"""Phase 1 baseline measurement. Runs on a Colab T4; will not run anywhere else.

    python -m scripts.colab_baseline --drive /content/drive/MyDrive/edgerag

Produces the numbers every later phase is measured against: HuggingFace ``generate()`` on the
headline 2.2B model, replaying the frozen trace.

Designed around free-tier Colab rather than in spite of it:

* **Resumable.** Each measurement is a named cell recorded in a manifest as it completes. A
  disconnect at cell 4 of 6 costs cell 4, not cells 1-3. Re-running skips completed cells.
* **Checkpointed to Drive on every cell**, fsync'd. The runtime is disposable; the results are not.
* **GPU time is spent last.** Corpus rebuild and trace verification are CPU work and happen before
  weights are loaded, so a disconnect during setup costs no GPU quota.
* **The device gate runs first.** Landing on a K80 or a CPU runtime fails in one second rather
  than after twenty minutes of downloading weights.

The OOM probe is a deliverable, not diagnostics: "how many concurrent k=5 RAG requests does a
naive HF pipeline sustain on a 16 GB T4" is the denominator for every concurrency claim in
Phase 3, and CONTEXT.md D11 predicts it is small.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from bench.bench import BenchRecord, JsonlWriter, TrialResult, run_benchmark
from bench.metrics import DeviceInfo, HeldConstant, assert_device_trusted, sync
from edgerag.core.loader import HEADLINE_MODEL, load_model
from edgerag.retrieval.corpus import CorpusDoc, load_corpus
from edgerag.retrieval.trace import (
    PROMPT_FORMAT_VERSION,
    TraceEntry,
    build_prompt_messages,
    load_trace,
    trace_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEW_TOKENS = 64


@dataclass
class Checkpoint:
    """Manifest of completed cells, so a killed session resumes instead of restarting."""

    path: Path

    def completed(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            return set(json.loads(self.path.read_text(encoding="utf-8"))["completed"])
        except (json.JSONDecodeError, KeyError, OSError):
            return set()

    def mark(self, cell: str) -> None:
        done = self.completed() | {cell}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump({"completed": sorted(done)}, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())


class PromptBuilder:
    """Turns trace entries into model inputs. One place, so every cell prompts identically."""

    def __init__(self, processor: Any, corpus: list[CorpusDoc], device: torch.device) -> None:
        self.processor = processor
        self.by_key = {d.doc_key: d for d in corpus}
        self.device = device

    def build(self, entry: TraceEntry) -> dict[str, torch.Tensor]:
        docs = [self.by_key[k] for k in entry.retrieved_doc_keys if k in self.by_key]
        images = [Image.open(REPO_ROOT / d.image_path).convert("RGB") for d in docs]
        messages = build_prompt_messages(docs, entry.question)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        batch = self.processor(text=prompt, images=images, return_tensors="pt")
        for im in images:
            im.close()
        return {k: v.to(self.device) for k, v in batch.items()}

    def build_batched(self, entries: list[TraceEntry]) -> dict[str, torch.Tensor]:
        """Batch several requests.

        Left-padding, deliberately: decode reads the *last* position, so right-padding would make
        the model continue from a pad token. The pad id comes from the tokenizer, never from the
        composite config -- see ``BUGS.md`` L-01.
        """
        docs_per = [
            [self.by_key[k] for k in e.retrieved_doc_keys if k in self.by_key] for e in entries
        ]
        prompts = [
            self.processor.apply_chat_template(
                build_prompt_messages(docs, e.question), add_generation_prompt=True
            )
            for docs, e in zip(docs_per, entries, strict=True)
        ]
        images = [
            [Image.open(REPO_ROOT / d.image_path).convert("RGB") for d in docs]
            for docs in docs_per
        ]

        previous_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            batch = self.processor(
                text=prompts, images=images, padding=True, return_tensors="pt"
            )
        finally:
            self.processor.tokenizer.padding_side = previous_side

        for group in images:
            for im in group:
                im.close()
        return {k: v.to(self.device) for k, v in batch.items()}


def measure_generate(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    new_tokens: int,
    use_cache: bool = True,
) -> TrialResult:
    """One HF ``generate()`` trial, TTFT and decode measured separately.

    TTFT comes from a ``max_new_tokens=1`` call and decode from a full call, rather than from
    timestamping inside the loop. Instrumenting each step needs a ``cuda.synchronize()`` per token,
    which slows the baseline and would inflate every speedup measured against it (``BUGS.md``
    P-13's mirror image: over-instrumenting the thing you are trying to beat).

    ``use_cache`` is explicit and recorded. A baseline accidentally run without the KV cache
    produces a fake 5-10x win that collapses under one question -- ``BUGS.md`` P-16.
    """
    n_prompt = int(inputs["input_ids"].shape[-1])
    common = {
        "do_sample": False,
        "use_cache": use_cache,
        "pad_token_id": model.config.text_config.pad_token_id,
    }

    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=1, **common)
    sync()
    ttft = time.perf_counter() - t0

    sync()
    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=new_tokens, **common)
    sync()
    total = time.perf_counter() - t1

    generated = int(out.shape[-1]) - n_prompt
    decode_total = max(total - ttft, 1e-9)

    return TrialResult.from_aggregate(
        ttft_s=ttft,
        decode_total_s=decode_total,
        n_prompt_tokens=n_prompt,
        n_generated_tokens=max(generated, 1),
        metadata={"use_cache": use_cache, "batch": int(inputs["input_ids"].shape[0])},
    )


def probe_max_batch(
    model: torch.nn.Module,
    builder: PromptBuilder,
    entries: list[TraceEntry],
    new_tokens: int,
    ceiling: int = 16,
) -> dict[str, Any]:
    """Find the largest batch a naive HF pipeline sustains before OOM.

    This is the **denominator for every concurrency claim in Phase 3**, so it has to be the actual
    maximum, not a lower bound. Doubling alone (1, 2, 4, 8) only proves "at least 4, fewer than 8";
    reporting that 4 as *the* baseline would inflate a later "Nx more concurrent sequences" claim
    by up to 75%. So the doubling phase is followed by a **linear refinement** between the last
    success and the first failure.

    CUDA OOM is a recoverable Python exception (unlike a device-side assert, which poisons the
    context -- ``BUGS.md`` L-01), so the probe can keep going after a failure.
    """
    results: dict[str, Any] = {
        "attempted": [],
        "max_ok": 0,
        "oom_at": None,
        "refined": False,
    }

    def attempt(batch: int) -> bool:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            inputs = builder.build_batched(entries[:batch])
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=model.config.text_config.pad_token_id,
                )
            sync()
            peak = torch.cuda.max_memory_allocated()
            results["attempted"].append(
                {"batch": batch, "ok": True, "peak_gib": round(peak / 1024**3, 3)}
            )
            del inputs
            return True
        except torch.cuda.OutOfMemoryError:
            results["attempted"].append({"batch": batch, "ok": False})
            torch.cuda.empty_cache()
            return False

    # Phase 1: double until failure, to bracket the answer cheaply.
    batch = 1
    while batch <= ceiling and len(entries) >= batch:
        if attempt(batch):
            results["max_ok"] = batch
            batch *= 2
        else:
            results["oom_at"] = batch
            break

    # Phase 2: walk the gap linearly. Without this, max_ok is a lower bound wearing the name of
    # a maximum.
    if results["oom_at"] is not None:
        for candidate in range(results["max_ok"] + 1, results["oom_at"]):
            if len(entries) < candidate:
                break
            if attempt(candidate):
                results["max_ok"] = candidate
                results["refined"] = True
            else:
                results["oom_at"] = candidate
                results["refined"] = True
                break

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 baseline on a Colab T4")
    parser.add_argument("--drive", type=str, default="", help="Drive dir for checkpointed results")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--new-tokens", type=int, default=DEFAULT_NEW_TOKENS)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--skip-oom-probe", action="store_true")
    parser.add_argument(
        "--allow-untrusted-device",
        action="store_true",
        help="record on non-T4 hardware; results stamped trusted=false and NOT publishable",
    )
    args = parser.parse_args(argv)

    # --- gate first: fail in a second, not after downloading 4.5 GB of weights ---
    info: DeviceInfo = assert_device_trusted(allow_untrusted=args.allow_untrusted_device)
    vram_gib = (info.total_memory_bytes or 0) / 1024**3
    print(
        f"device: {info.name} | cap {info.compute_capability} | {vram_gib:.1f} GiB "
        f"| tensor cores {info.has_tensor_cores}"
    )

    out_dir = Path(args.drive) if args.drive else (REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(out_dir / "baseline.jsonl")
    checkpoint = Checkpoint(out_dir / "baseline_manifest.json")
    done = checkpoint.completed()
    if done:
        print(f"resuming -- already complete: {sorted(done)}")

    # --- CPU-side setup, before any GPU quota is spent ---
    corpus = load_corpus()
    trace = load_trace()
    fingerprint = trace_fingerprint(trace)
    print(f"corpus {len(corpus)} pages | trace {len(trace)} entries | "
          f"fingerprint {fingerprint} | prompt format v{PROMPT_FORMAT_VERSION}")

    eval_entries = [e for e in trace if e.split == "eval"]
    if len(eval_entries) < max(args.batches):
        print("not enough eval entries for the requested batch sizes", file=sys.stderr)
        return 1

    # --- GPU work starts here ---
    print(f"\nloading {args.model} ...")
    lm = load_model(args.model, device="cuda", dtype=torch.float16)
    print(f"  {lm.n_params:,} params | {lm.weight_bytes / 1024**3:.2f} GiB weights")
    builder = PromptBuilder(lm.processor, corpus, lm.device)

    held_base = {
        "model_id": args.model,
        "dtype": "float16",
        "prompt_tokens": -1,  # variable per request; the trace fingerprint pins the workload
        "max_new_tokens": args.new_tokens,
    }

    def record(rec: BenchRecord, cell: str) -> None:
        writer.append(rec)
        checkpoint.mark(cell)
        print(
            f"  {cell}: TTFT p50 {rec.ttft['p50'] * 1e3:.0f} ms | "
            f"decode {rec.decode_tokens_per_s['p50']:.1f} tok/s | "
            f"peak {rec.memory['peak_allocated_gib']:.2f} GiB"
        )

    for batch in args.batches:
        cell = f"hf_generate_b{batch}"
        if cell in done:
            print(f"  {cell}: skipped (already recorded)")
            continue
        entries = eval_entries[:batch]
        # An OOM at batch 4 must not cost the nocache and oom_probe cells that follow. CUDA OOM
        # is a recoverable Python exception, so the run continues after clearing the allocator.
        try:
            inputs = builder.build_batched(entries) if batch > 1 else builder.build(entries[0])
            rec = run_benchmark(
                name=cell,
                fn=lambda i=inputs: measure_generate(lm.model, i, args.new_tokens, use_cache=True),
                held_constant=HeldConstant(**held_base, batch_size=batch),
                warmup=args.warmup,
                trials=args.trials,
                allow_untrusted_device=args.allow_untrusted_device,
                workload_fingerprint=fingerprint,
                notes=(
                    "HuggingFace generate(), use_cache=True. "
                    "The baseline every later phase is measured against."
                ),
            )
            record(rec, cell)
            del inputs
        except torch.cuda.OutOfMemoryError:
            # Not a failure -- it is the headline result. A naive pipeline that cannot hold this
            # batch is precisely what Phase 3 exists to fix.
            print(f"  {cell}: OOM -- naive pipeline cannot sustain batch {batch}")
            oom_note = out_dir / "baseline_oom.json"
            prior = (
                json.loads(oom_note.read_text(encoding="utf-8")) if oom_note.exists() else []
            )
            prior.append({"cell": cell, "batch": batch, "device": info.name})
            oom_note.write_text(json.dumps(prior, indent=2), encoding="utf-8")
            checkpoint.mark(cell)
        torch.cuda.empty_cache()

    # Decode without the KV cache, for the Phase 2 "why the cache matters" plot. Run at batch 1
    # and fewer tokens: without a cache every step re-attends the whole prefix, so this is
    # quadratic and would otherwise eat the session.
    cell = "hf_generate_nocache_b1"
    if cell not in done:
        inputs = builder.build(eval_entries[0])
        rec = run_benchmark(
            name=cell,
            fn=lambda i=inputs: measure_generate(lm.model, i, 16, use_cache=False),
            held_constant=HeldConstant(**{**held_base, "max_new_tokens": 16}, batch_size=1),
            warmup=1,
            trials=args.trials,
            allow_untrusted_device=args.allow_untrusted_device,
            workload_fingerprint=fingerprint,
            notes="use_cache=False. Quadratic in prefix length; 16 tokens only.",
        )
        record(rec, cell)
        del inputs
        torch.cuda.empty_cache()

    cell = "oom_probe"
    if not args.skip_oom_probe and cell not in done:
        print("\nprobing max concurrent requests before OOM ...")
        probe = probe_max_batch(lm.model, builder, eval_entries, args.new_tokens)
        probe.update({"device": info.name, "workload_fingerprint": fingerprint})
        path = out_dir / "oom_probe.json"
        path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
        checkpoint.mark(cell)
        print(f"  max batch OK: {probe['max_ok']} | OOM at: {probe['oom_at']}")
        print(f"  wrote {path}")

    print(f"\ndone. results in {out_dir}")
    print("Copy baseline.jsonl back into the repo's results/ and commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
