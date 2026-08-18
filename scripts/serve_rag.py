"""Boot the RAG server: retrieval, quantized model, paged cache, streaming HTTP.

    python -m scripts.serve_rag                     # ship config, port 8000
    python -m scripts.serve_rag --arm fp16          # unquantized, for comparison

This is the assembly point -- it constructs every piece the earlier phases built and wires them
together, and contains no logic of its own worth testing beyond that. Defaults are the *measured*
ones, not plausible ones:

* **``LM8+ViT4`` by default** (INT8 language, INT4 vision). ``CONTEXT.md`` D24: 2.296 GiB against
  fp16's 4.185, at 0.4193 ANLS against 0.4378 -- a 0.31 sigma difference. The uniform INT4
  alternative is smaller still and keeps 60% of quality, which is not a trade worth defaulting to.
* **Text-only retrieval.** D22 measured the image-space query signal as noise (0.14x the text
  side's spread, mean per-query maximum indistinguishable from zero), so ``FlatIndex`` is built
  with no image embeddings at all. That also makes startup instant: embedding 362 pages costs ~50
  minutes and buys nothing measurable.
* **Chunked prefill at 512** (D18), so one 7,000-token retrieved prompt cannot monopolise the GPU
  while other requests decode (P-18).

The block pool is the memory knob. 640 blocks x 16 tokens is ~1.9 GiB, enough for one ~7k-token
RAG request with room to grow; raise it for concurrency, and the budget arithmetic is in
``scripts/measure_memory_ledger.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from bench.pipeline import free_duplicate_hf_decoder
from edgerag.cache.allocator import BlockAllocator
from edgerag.core.loader import HEADLINE_MODEL, load_model
from edgerag.core.model import load_from_hf
from edgerag.core.quant import QuantConfig
from edgerag.retrieval.corpus import load_corpus
from edgerag.retrieval.index import FlatIndex
from edgerag.sched.scheduler import Scheduler, SchedulerConfig
from edgerag.serve.app import ServerState, create_app
from edgerag.serve.engine import InferenceEngine
from edgerag.serve.executor import ModelExecutor
from edgerag.serve.pipeline import RagPipeline
from scripts.colab_quant_ablation import MIXED_ARMS, arm_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def build_server(
    model_id: str = HEADLINE_MODEL,
    arm: str = "LM8+ViT4",
    bits: int = 4,
    group_size: int = 128,
    num_blocks: int = 640,
    block_size: int = 16,
    chunk_size: int = 512,
    k: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Assemble everything and return ``(app, engine)``. The engine is already running."""
    spec_bits = {} if arm == "fp16" else arm_spec(arm, bits)
    language_bits, vision_bits = spec_bits.get("language"), spec_bits.get("vision")

    print(f"loading {model_id} as {arm} {spec_bits or '(no quantization)'}")
    lm = load_model(model_id, device=device, dtype=dtype)
    decoder = load_from_hf(
        lm.spec,
        lm.model,
        quant_config=(
            QuantConfig(group_size=group_size, bits=language_bits) if language_bits else None
        ),
    )
    # The HF text decoder is now a duplicate of weights we own -- 3.6 GiB held for nothing, which
    # is what made every Phase 4 request OOM (BUGS.md B-05).
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
    print(f"index: {len(docs)} pages, {sum(1 for d in docs if d.text)} with OCR text, "
          f"vocab {index.vectorizer.vocab_size}")

    allocator = BlockAllocator(num_blocks, block_size)
    executor = ModelExecutor(
        decoder, lm.spec, allocator, lm.device, dtype, chunk_size=chunk_size
    )
    scheduler = Scheduler(
        allocator, SchedulerConfig(eos_token_id=lm.processor.tokenizer.eos_token_id)
    )
    engine = InferenceEngine(scheduler, executor)
    engine.start()

    rag = RagPipeline(
        index=index, docs_by_key=docs_by_key, hf_model=lm.model, decoder=decoder,
        processor=lm.processor, spec=lm.spec, device=lm.device, k=k, repo_root=REPO_ROOT,
    )
    app = create_app(
        ServerState(
            engine=engine, tokenizer=lm.processor.tokenizer, model_id=f"{model_id}:{arm}", rag=rag
        )
    )

    if torch.cuda.is_available():
        print(f"resident: {torch.cuda.memory_allocated() / GIB:.2f} GiB "
              f"(weights + {executor.pool_bytes / GIB:.2f} GiB block pool)")
    return app, engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve EdgeRAG over HTTP")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument(
        "--arm", default="LM8+ViT4",
        choices=["fp16", "LM", "LM+ViT", "ViT", *MIXED_ARMS],
        help="quantization configuration; the default is D24's measured ship config",
    )
    parser.add_argument("--bits", type=int, default=4, help="ignored for mixed arms")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=640)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("-k", type=int, default=5, help="pages retrieved per question")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed: pip install -e '.[serve]'", file=sys.stderr)
        return 1

    app, engine = build_server(
        model_id=args.model, arm=args.arm, bits=args.bits, group_size=args.group_size,
        num_blocks=args.num_blocks, chunk_size=args.chunk_size, k=args.k, device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    print(f"\nserving on http://{args.host}:{args.port}  --  POST /v1/chat/completions\n")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
