"""Phase 6 gate: memory, throughput and quality across {fp16, int8, int4} x {LM, LM+ViT, ViT}.

    python -m scripts.colab_quant_ablation --drive /content/drive/MyDrive/edgerag

The two columns that need a GPU. The third — memory — is already computed exactly and locally by
``scripts.measure_memory_ledger`` (``CONTEXT.md`` D21), so this script does not re-derive it. What
it does instead is **check it**: for every arm it sums the bytes the loaded model actually holds
and compares them against what the ledger predicted, and prints the delta. A ledger that is never
checked against a running model is a spreadsheet.

**Set expectations before reading the throughput column: INT4 is expected to be SLOWER than fp16
here, and that is the documented result, not a bug** (``CONTEXT.md`` D7). ``QuantLinear.forward``
dequantizes to fp16 and calls a normal matmul, because a T4 has no INT4 tensor cores. The packed
bytes are expanded *before* they reach the multiplier, so the narrower load buys nothing and the
expansion costs. The speed win needs the dequantize fused into the GEMV — the Triton kernel D7
time-boxed and this project has not spent. Reporting "4x smaller, N% slower, here is the roofline
reason, here is what the fused kernel recovers" is the honest position and the one D7 chose in
advance.

Two controls are built in, and both are falsifiable:

* **The ViT arms must not move decode throughput.** The vision tower runs once during prefill and
  is not in the decode loop at all, so quantizing it can change TTFT and quality but cannot change
  tok/s. If it does, the measurement is wrong.
* **Pruning is off everywhere** (``keep_ratio=1.0``, proven bit-identical to no compressor). Phase
  4 measured pruning; this measures quantization. Varying both at once would produce a table where
  no cell explains anything (``00_FOUNDATIONS.md`` §4 rule 5).

Resumable at arm granularity: each arm reloads the checkpoint, and a completed arm is appended to
the output file and skipped on the next run. A disconnect costs one arm.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

from bench.metrics import MemoryProbe, Percentiles, anls, assert_device_trusted
from bench.pipeline import free_duplicate_hf_decoder, generate
from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.paged import PagedKVCache
from edgerag.compress.fastv import FastVConfig
from edgerag.core.linear import SKIP_RAGGED, quantize_module_
from edgerag.core.loader import HEADLINE_MODEL, load_model
from edgerag.core.model import load_from_hf
from edgerag.core.quant import QuantConfig
from edgerag.retrieval.corpus import load_corpus
from edgerag.retrieval.trace import load_trace, trace_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
MIB = 1024**2

#: Arm name -> which components it quantizes. Must match ``scripts/measure_memory_ledger.ARMS``,
#: or the cross-check below compares two different experiments and reports the difference as a
#: ledger error.
ARMS: dict[str, tuple[str, ...]] = {
    "fp16": (),
    "LM": ("language",),
    "LM+ViT": ("language", "vision", "connector"),
    "ViT": ("vision", "connector"),
}

BLOCK_SIZE = 16


def _vision_parts(hf_model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    inner = getattr(hf_model, "model", hf_model)
    return inner.vision_model, inner.connector


def resident_bytes(*modules: torch.nn.Module) -> int:
    """Bytes these modules actually hold on the device, at whatever dtype they ended up in.

    Deliberately *not* the ledger's arithmetic repeated: it reads the tensors that exist. That is
    the whole value of the comparison — one number comes from shapes on a laptop, the other from
    storage on a T4, and they are allowed to disagree.
    """
    total = 0
    for module in modules:
        total += sum(t.numel() * t.element_size() for t in module.parameters())
        total += sum(t.numel() * t.element_size() for t in module.buffers())
    return total


def ledger_prediction(arm: str, bits: int) -> int | None:
    """What ``results/memory_ledger.json`` says this arm should weigh, if it has been computed."""
    path = REPO_ROOT / "results" / "memory_ledger.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    wanted = ("none", 16) if arm == "fp16" else (arm, bits)
    for row in payload.get("matrix", []):
        if (row["arm"], row["bits"]) == wanted:
            return int(row["total_bytes"])
    return None


def build_arm(
    model_id: str,
    arm: str,
    bits: int,
    group_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[Any, torch.nn.Module]:
    """Load the checkpoint and assemble one arm of the ablation.

    Order matters and is not arbitrary. The decoder is built **quantized from the start** via the
    ``load_from_hf`` flag rather than built dense and converted, and the duplicate HF text decoder
    is freed immediately afterwards -- on a 14.6 GiB card, holding an fp16 copy of a stack you have
    already quantized is how the Phase 4 run OOM'd on every request (``BUGS.md`` B-05).

    ``device``/``dtype`` exist so a CPU test can drive this on the 256M fixture. That is not a
    convenience: the last script written for a T4 and never executed anywhere else produced a
    complete, well-formed file of zeros (B-05 again). The runs that matter are still T4-only, and
    ``main`` still refuses anything else.
    """
    components = ARMS[arm]
    config = QuantConfig(group_size=group_size, bits=bits) if components else None

    lm = load_model(model_id, device=device, dtype=dtype)
    decoder = load_from_hf(
        lm.spec, lm.model, quant_config=config if "language" in components else None
    )
    free_duplicate_hf_decoder(lm.model)

    if config is not None and "vision" in components:
        vision, connector = _vision_parts(lm.model)
        plan = quantize_module_(vision, config)
        quantize_module_(connector, config)
        ragged = len(plan.skipped_for(SKIP_RAGGED))
        if ragged:
            # Stated, never silent: these are the in_features=4304 MLP layers, and they are the
            # reason the ViT arm compresses 2.0x rather than 3.9x (CONTEXT.md D21 finding 2).
            print(f"    {ragged} vision layers left fp16: in_features not divisible by "
                  f"{group_size}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return lm, decoder


def measure_arm(
    lm: Any,
    decoder: torch.nn.Module,
    requests: list[Any],
    docs_by_key: dict,
    max_new_tokens: int,
    num_blocks: int,
    trials: int,
) -> dict[str, Any]:
    """Throughput on a repeated request, then quality over the held-out set.

    Throughput first and on **one** request repeated: it is the measurement that varies with the
    thing being ablated, and pinning the input means the only difference between arms is the
    weights. Quality then runs over all of them, because a single question's ANLS is noise.
    """
    cache = PagedKVCache(
        lm.spec, BlockAllocator(num_blocks, BLOCK_SIZE), lm.device, lm.dtype
    )
    no_pruning = FastVConfig(keep_ratio=1.0)

    def run(entry: Any) -> tuple[str, dict[str, Any]]:
        return generate(
            decoder, lm.model, lm.processor, lm.spec, entry, docs_by_key,
            no_pruning, "attention", max_new_tokens, lm.device, cache,
        )

    pinned = requests[0]
    run(pinned)  # warmup: first-call kernel selection and autotuning, discarded (§4 rule 1)

    ttfts, tok_s, prefill_tokens = [], [], 0
    peak_bytes = 0
    for _ in range(trials):
        with MemoryProbe() as probe:
            _, stats = run(pinned)
        ttfts.append(stats["ttft_s"])
        if stats["generated_tokens"]:
            tok_s.append(stats["generated_tokens"] / stats["decode_s"])
        prefill_tokens = stats["prefill_tokens"]
        sample = probe.require()
        peak_bytes = max(peak_bytes, sample.peak_allocated_bytes)

    scores, oom = [], 0
    for entry in requests:
        try:
            text, _ = run(entry)
        except torch.cuda.OutOfMemoryError:
            oom += 1
            cache.reset()
            torch.cuda.empty_cache()
            continue
        scores.append(anls(text, entry.answers))

    return {
        "prompt_tokens": prefill_tokens,
        "peak_allocated_bytes": peak_bytes,
        "ttft_s": Percentiles.of(ttfts).to_dict(),
        "decode_tokens_per_s": Percentiles.of(tok_s).to_dict() if tok_s else None,
        "anls": sum(scores) / len(scores) if scores else 0.0,
        "n_scored": len(scores),
        # The count this *specific* measurement was run against. Recorded per row because arms
        # measured in different sessions can carry different --n-queries -- resuming skips an
        # arm outright, it does not check that the settings match, so a blanket CLI value at
        # print time would silently misdescribe every row measured somewhere else.
        "n_requested": len(requests),
        "n_oom": oom,
    }


def completed_arms(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done.add((row["arm"], row["bits"]))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 quantization ablation (T4)")
    parser.add_argument("--drive", default="")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--arms", nargs="+", default=["fp16", "LM", "LM+ViT", "ViT"])
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--n-queries", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=640,
        help=(
            "block pool size. One pool, not two -- pruning is off here, so the compressed cache's "
            "second half would be allocated and left empty, inflating the memory column this "
            "script exists to measure. 640 x 16 tokens covers a 7k prompt at ~1.9 GiB."
        ),
    )
    parser.add_argument("--allow-untrusted-device", action="store_true")
    args = parser.parse_args(argv)

    info = assert_device_trusted(allow_untrusted=args.allow_untrusted_device)
    print(f"device: {info.name}")
    print("EXPECT INT4 TO BE SLOWER THAN FP16 (CONTEXT.md D7): dequantize-then-matmul on a card")
    print("with no INT4 tensor cores. The memory win is real; the speed win needs a fused kernel.")
    print("EXPECT THE ViT ARMS TO MATCH FP16 ON tok/s: the tower is not in the decode loop.\n")

    corpus = load_corpus()
    docs_by_key = {d.doc_key: d for d in corpus}
    trace = load_trace()
    fingerprint = trace_fingerprint(trace)
    heldout = [e for e in trace if e.split == "heldout"][: args.n_queries]
    print(f"trace {fingerprint} | {len(heldout)} held-out requests")

    longest = max(
        sum(len(docs_by_key[k].text) // 4 + 900 for k in e.retrieved_doc_keys if k in docs_by_key)
        for e in heldout
    )
    needed = (longest + args.max_new_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    print(f"block pool: {args.num_blocks} x {BLOCK_SIZE} tokens "
          f"(~{args.num_blocks * 3 / 1024:.1f} GiB); longest request needs ~{needed}")
    if needed > args.num_blocks:
        print(f"\nFAILED: the longest request needs ~{needed} blocks against a pool of "
              f"{args.num_blocks}. Re-run with --num-blocks {int(needed * 1.3)}.", file=sys.stderr)
        return 1

    out_dir = Path(args.drive) if args.drive else (REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quant_ablation.jsonl"
    done = completed_arms(out_path)
    if done:
        print(f"resuming: {len(done)} arm(s) already in {out_path.name}")
    print()

    # fp16 is one arm regardless of bit width -- "quantize nothing at 4 bits" and "quantize
    # nothing at 8 bits" are the same model, and running it twice would put two samples of the
    # same configuration in a table read as a comparison.
    plan: list[tuple[str, int]] = [("fp16", 16)] if "fp16" in args.arms else []
    plan += [(arm, bits) for bits in args.bits for arm in args.arms if arm != "fp16"]

    for arm, bits in plan:
        if (arm, bits) in done:
            print(f"  {arm} @ int{bits}: already measured, skipping")
            continue
        label = "fp16" if bits == 16 else f"{arm} @ int{bits}"
        print(f"  {label}: loading")

        lm, decoder = build_arm(args.model, arm, bits, args.group_size)
        vision, connector = _vision_parts(lm.model)
        weight_bytes = resident_bytes(decoder, vision, connector)

        predicted = ledger_prediction(arm, bits)
        delta = None if predicted is None else weight_bytes - predicted
        if delta is None:
            check = "no ledger on file"
        elif abs(delta) < MIB:
            check = f"ledger agrees (delta {delta:+,} B)"
        else:
            # Not an assertion. A disagreement is a finding about the ledger, and aborting the
            # session would throw away the measurement that produced it.
            check = f"** LEDGER DISAGREES by {delta / MIB:+.1f} MiB **"
        print(f"    weights {weight_bytes / GIB:.3f} GiB -- {check}")

        try:
            measured = measure_arm(
                lm, decoder, heldout, docs_by_key,
                args.max_new_tokens, args.num_blocks, args.trials,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM: {label} does not fit. Recording nothing and moving on.",
                  file=sys.stderr)
            del decoder, lm
            gc.collect()
            torch.cuda.empty_cache()
            continue

        record = {
            "arm": arm,
            "bits": bits,
            "group_size": args.group_size if bits != 16 else None,
            "weight_bytes": weight_bytes,
            "weight_gib": round(weight_bytes / GIB, 4),
            "ledger_predicted_bytes": predicted,
            "ledger_delta_bytes": delta,
            "pruning": "off (keep_ratio=1.0)",
            "max_new_tokens": args.max_new_tokens,
            "trials": args.trials,
            "workload_fingerprint": fingerprint,
            "device": info.name,
            "trusted": info.trusted,
            **measured,
        }
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()

        decode = record["decode_tokens_per_s"]
        print(f"    ANLS={record['anls']:.4f} (n={record['n_scored']})  "
              f"tok/s={decode['p50']:.2f}  TTFT={record['ttft_s']['p50']:.2f}s  "
              f"peak={record['peak_allocated_bytes'] / GIB:.2f} GiB\n"
              if decode else "    no decode samples\n")

        del decoder, lm
        gc.collect()
        torch.cuda.empty_cache()

    print(f"wrote {out_path}\n")
    return summarise(out_path, args.n_queries)


def summarise(out_path: Path, n_queries: int) -> int:
    """Print the gate's table, and refuse to call an empty run a result.

    ``n_queries`` describes only *this invocation* -- resuming skips an arm outright rather than
    re-measuring it, so a row already on disk may have been scored under a different session's
    ``--n-queries``. Reading ``n_queries`` here instead of each row's own ``n_scored`` is exactly
    how a leftover smoke-test row would get printed with a full run's sample-size caveat, or the
    reverse. The per-row ``n`` column is the only number that is ever true of the row it sits on.
    """
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        print("FAILED: nothing was measured. Do not use this file.", file=sys.stderr)
        return 1

    baseline = next((r for r in rows if r["bits"] == 16), None)
    print(f"{'arm':>8}{'bits':>6}{'weights':>10}{'peak':>8}{'tok/s':>9}{'vs fp16':>9}"
          f"{'TTFT':>8}{'ANLS':>8}{'n':>5}")
    for row in sorted(rows, key=lambda r: (-r["bits"], r["arm"])):
        decode = row["decode_tokens_per_s"]
        speed = decode["p50"] if decode else float("nan")
        relative = (
            f"{speed / baseline['decode_tokens_per_s']['p50']:.2f}x"
            if baseline and baseline["decode_tokens_per_s"] else "--"
        )
        print(f"{row['arm']:>8}{row['bits']:>6}{row['weight_gib']:>9.3f}G"
              f"{row['peak_allocated_bytes'] / GIB:>7.2f}G{speed:>9.2f}{relative:>9}"
              f"{row['ttft_s']['p50']:>7.2f}s{row['anls']:>8.4f}{row['n_scored']:>5}")

    # sqrt-scaled off CONTEXT.md D20's n=40 measurement, not re-derived from theory: ANLS is not a
    # clean binomial proportion, so this is a rough guide for reading the table, not a real CI.
    print("\n  SE per row ~= 0.06 * sqrt(40 / n) (calibrated at n=40, CONTEXT.md D20) -- read the")
    print("  n column per row, not one blanket figure: arms resumed from an earlier session may")
    print("  carry a different sample size than this invocation's --n-queries.")
    print("  `peak` is measured, so CONTEXT.md D21's inferred activation line can now be replaced")
    print("  with arithmetic on these numbers rather than on a median request length.")

    # `n_requested` is per-row and present on anything measured after this field was added; older
    # rows fall back to this invocation's --n-queries, the best available guess for them.
    thin = [r for r in rows if r["n_scored"] < r.get("n_requested", n_queries) * 0.5]
    for row in thin:
        expected = row.get("n_requested", n_queries)
        print(f"\nWARNING: {row['arm']} @ int{row['bits']} scored only {row['n_scored']}/"
              f"{expected} requests -- that mean is over a biased subset.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
