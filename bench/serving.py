"""Assemble the serving stack once, so the server and the load test cannot diverge.

Extracted from ``scripts/serve_rag.py`` when Phase 5e needed the same stack without HTTP. The
reason is the one already stated in ``bench/pipeline.py``: two experiments that share a pipeline
must share the *implementation* of it. A load test that assembles its own executor, scheduler and
pool is measuring a system that resembles the server rather than the system that is the server,
and the resemblance stops at the first bug fix applied to only one of them.

**No FastAPI import anywhere below.** The serving extras are optional -- installed with
``pip install -e ".[serve]"`` -- and a measurement run that dies on ``No module named 'fastapi'``
after a 4.5 GiB weight download is a bad way to learn that. ``scripts/serve_rag.py`` adds the HTTP
layer on top of what this returns; ``scripts/colab_poisson.py`` drives the engine directly and
never needs it.

The quantization arm arrives as a plain ``{component: bits}`` mapping rather than an arm name.
That keeps the arm vocabulary (``fp16``, ``LM8+ViT4``, ...) in ``scripts/`` where the ablation
defines it, instead of making ``bench/`` depend on a script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from bench.pipeline import free_duplicate_hf_decoder
from edgerag.cache.allocator import BlockAllocator
from edgerag.core.budget import PoolPlan, plan_pool_for_budget
from edgerag.core.loader import HEADLINE_MODEL, load_model
from edgerag.core.model import load_from_hf
from edgerag.core.quant import QuantConfig
from edgerag.retrieval.corpus import load_corpus
from edgerag.retrieval.index import FlatIndex
from edgerag.sched.scheduler import Scheduler, SchedulerConfig
from edgerag.serve.engine import InferenceEngine
from edgerag.serve.executor import ModelExecutor
from edgerag.serve.pipeline import RagPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3

#: Where the measured activation term comes from. Phase 5e is the first run that recorded a peak
#: alongside the exact weights and pool it was measured with, which is what makes the subtraction
#: possible at all -- earlier runs had to infer the term from the HF baseline (``CONTEXT.md`` D21).
SWEEP_PATH = REPO_ROOT / "results" / "poisson_sweep.jsonl"


def measured_activation_bytes(
    chunked: bool = True, path: Path | None = None, model_id: str = HEADLINE_MODEL
) -> int | None:
    """Transient peak above weights and pool, read off a Phase 5e cell **for this model**.

    ``peak - weights - pool`` on a record that stamps all three. Returns ``None`` when no matching
    sweep is on file rather than guessing, because a fabricated activation term is exactly the
    failure ``BUGS.md`` P-25 describes: the table sums under budget and the pipeline OOMs anyway.

    Three filters, and each one exists because ignoring it produces a plausible wrong number:

    * **``model_id``.** Activation scales with hidden size and prompt length. Sizing the 256M
      fixture's pool from the 2.2B's 0.324 GiB would silently reserve ~10x what it needs -- and
      the failure is invisible, because the server starts fine and just serves shorter prompts
      than it could.
    * **``chunked``.** 0.324 GiB against 0.796 unchunked; the wrong one mis-sizes by 484 MiB.
    * **``trusted``.** An untrusted-device residual is not publishable and must not silently
      become a production pool size (``CONTEXT.md`` D4).
    """
    path = path or SWEEP_PATH
    if not path.exists():
        return None
    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("model") != model_id:
            continue
        if row.get("chunked_prefill") is not chunked or not row.get("trusted"):
            continue
        residual = row["peak_allocated_bytes"] - row["weight_bytes"] - row["pool_bytes"]
        if residual > 0:
            candidates.append(residual)
    # The maximum, not the mean: this is a high-water mark being reserved against, and averaging
    # two peaks produces a number neither run stayed under.
    return max(candidates) if candidates else None

#: Blocks in the served pool. 640 x 16 tokens is ~1.9 GiB, which holds exactly **one** median RAG
#: request (~6,758 prompt tokens + generation = ~425 blocks). That is the ship configuration and
#: it is deliberate; see ``concurrency_supported`` for why a *concurrency* sweep must raise it.
DEFAULT_NUM_BLOCKS = 640
DEFAULT_BLOCK_SIZE = 16


def resident_weight_bytes(lm: Any, decoder: torch.nn.Module) -> int:
    """Bytes held by what actually runs: our decoder plus HuggingFace's surviving vision half.

    The HF text decoder is freed during assembly (``BUGS.md`` B-05), so summing ``lm.model`` alone
    would miss everything we own. Needed before the pool exists, because the pool's size is
    derived from what the weights left over.
    """
    inner = getattr(lm.model, "model", lm.model)
    total = 0
    for module in (decoder, inner.vision_model, inner.connector):
        total += sum(t.numel() * t.element_size() for t in module.parameters())
        total += sum(t.numel() * t.element_size() for t in module.buffers())
    return total


@dataclass
class ServingStack:
    """Everything the server needs, already wired and running.

    Returned as one object rather than a tuple because the load test needs pieces the HTTP server
    does not -- the scheduler's stats, the allocator's free count, the executor's chunk size -- and
    threading those through a growing tuple is how call sites start disagreeing about the order.
    """

    lm: Any
    decoder: torch.nn.Module
    index: FlatIndex
    docs_by_key: dict
    allocator: BlockAllocator
    executor: ModelExecutor
    scheduler: Scheduler
    engine: InferenceEngine
    rag: RagPipeline | None
    model_id: str
    arm: str

    #: Set when the pool was sized from a memory budget rather than from ``num_blocks``.
    pool_plan: PoolPlan | None = None

    @property
    def weight_bytes(self) -> int:
        """Bytes the model actually holds: our decoder plus the surviving vision half."""
        return resident_weight_bytes(self.lm, self.decoder)

    def blocks_per_request(self, prompt_tokens: int, max_new_tokens: int) -> int:
        return self.allocator.blocks_needed(prompt_tokens + max_new_tokens)

    def concurrency_supported(self, prompt_tokens: int, max_new_tokens: int) -> int:
        """How many such requests the pool admits at once.

        This is the number that decides whether a concurrency sweep can say anything at all. At
        the ship pool of 640 blocks a 6,758-token RAG request needs ~425, so the answer is **1**:
        admission blocks the second arrival by construction, and a throughput-vs-concurrency curve
        measured there would be a flat line describing the pool rather than the scheduler.
        """
        per_request = self.blocks_per_request(prompt_tokens, max_new_tokens)
        usable = self.allocator.num_blocks - self.scheduler.config.cow_reserve_blocks
        return max(0, usable // per_request)

    def stop(self) -> None:
        self.engine.stop()


def build_stack(
    model_id: str = HEADLINE_MODEL,
    quant_spec: dict[str, int] | None = None,
    *,
    arm: str = "fp16",
    group_size: int = 128,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    chunk_size: int = 512,
    max_batch_size: int = 8,
    max_prefills_per_step: int = 1,
    budget_gib: float | None = None,
    k: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    build_rag: bool = True,
    verbose: bool = True,
) -> ServingStack:
    """Load the checkpoint, quantize it, allocate the pool, and start the engine.

    ``quant_spec`` maps component name (``language`` / ``vision`` / ``connector``) to bit width;
    an empty mapping or ``None`` serves fp16. Assembly order is not arbitrary -- the decoder is
    built quantized rather than built dense and converted, and HuggingFace's duplicate text decoder
    is freed immediately after its weights are copied out, because holding an fp16 copy of a stack
    you have already quantized is ``BUGS.md`` B-05.
    """
    spec_bits = dict(quant_spec or {})
    language_bits = spec_bits.get("language")
    vision_bits = spec_bits.get("vision")

    if verbose:
        print(f"loading {model_id} as {arm} {spec_bits or '(no quantization)'}")

    lm = load_model(model_id, device=device, dtype=dtype)
    decoder = load_from_hf(
        lm.spec,
        lm.model,
        quant_config=(
            QuantConfig(group_size=group_size, bits=language_bits) if language_bits else None
        ),
    )
    free_duplicate_hf_decoder(lm.model)

    if vision_bits:
        from edgerag.core.linear import quantize_module_

        inner = lm.model.model
        quantize_module_(inner.vision_model, QuantConfig(group_size=group_size, bits=vision_bits))
        quantize_module_(inner.connector, QuantConfig(group_size=group_size, bits=vision_bits))

    docs = load_corpus()
    docs_by_key = {d.doc_key: d for d in docs}
    # No image embeddings: D22 measured that signal as noise, and FlatIndex degrades to its text
    # score cleanly rather than erroring. Startup is instant as a result.
    index = FlatIndex.build(docs, image_embeddings={})
    if verbose:
        print(f"index: {len(docs)} pages, {sum(1 for d in docs if d.text)} with OCR text, "
              f"vocab {index.vectorizer.vocab_size}")

    # Size the pool *from* the budget when one is given, rather than accepting a block count and
    # discovering afterwards what it summed to. The activation term is measured, not assumed, and
    # the run refuses to invent one -- a budget table with a fabricated transient line is
    # `BUGS.md` P-25: it sums under budget and the pipeline OOMs anyway.
    pool_plan = None
    if budget_gib is not None:
        activation = measured_activation_bytes(chunked=True, model_id=model_id)
        if activation is None:
            raise FileNotFoundError(
                f"--budget-gib needs an activation term measured for {model_id!r} with chunked "
                f"prefill, and {SWEEP_PATH.name} has no such record. Run scripts.colab_poisson "
                "against this model on a T4, or size the pool with --num-blocks instead. "
                "Refusing to borrow another model's activation term: it would start cleanly and "
                "reserve the wrong amount."
            )
        pool_plan = plan_pool_for_budget(
            kv_bytes_per_token=lm.spec.kv_bytes_per_token(),
            weights_bytes=resident_weight_bytes(lm, decoder),
            activation_bytes=activation,
            budget_gib=budget_gib,
            block_size=block_size,
            reserve_blocks=SchedulerConfig().cow_reserve_blocks,
        )
        num_blocks = pool_plan.num_blocks
        if verbose:
            print(f"pool sized from budget: {pool_plan.describe()}")

    allocator = BlockAllocator(num_blocks, block_size)
    executor = ModelExecutor(
        decoder, lm.spec, allocator, lm.device, dtype, chunk_size=chunk_size
    )
    scheduler = Scheduler(
        allocator,
        SchedulerConfig(
            max_batch_size=max_batch_size,
            # Set for completeness; the scheduler does not read it. `ModelExecutor.chunk_size`
            # above is what actually slices a prefill, and `tests/test_load.py` pins the
            # difference so a sweep cannot turn the inert one by mistake.
            prefill_chunk_size=chunk_size,
            max_prefills_per_step=max_prefills_per_step,
            eos_token_id=lm.processor.tokenizer.eos_token_id,
        ),
    )
    engine = InferenceEngine(scheduler, executor)
    engine.start()

    rag = None
    if build_rag:
        rag = RagPipeline(
            index=index, docs_by_key=docs_by_key, hf_model=lm.model, decoder=decoder,
            processor=lm.processor, spec=lm.spec, device=lm.device, k=k, repo_root=REPO_ROOT,
        )

    if verbose and torch.cuda.is_available():
        print(f"resident: {torch.cuda.memory_allocated() / GIB:.2f} GiB "
              f"(weights + {executor.pool_bytes / GIB:.2f} GiB block pool)")

    return ServingStack(
        lm=lm, decoder=decoder, index=index, docs_by_key=docs_by_key, allocator=allocator,
        executor=executor, scheduler=scheduler, engine=engine, rag=rag,
        model_id=model_id, arm=arm, pool_plan=pool_plan,
    )
