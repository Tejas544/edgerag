"""Does the fused kernel work, and does it buy back the gather? T4 only.

    python -m scripts.colab_fused_attention --drive /content/drive/MyDrive/edgerag

``CONTEXT.md`` D3 chose gather-into-scratch over a fused kernel and promised to revisit if the copy
ever exceeded ~25% of the paged attention path. D19 measured **72.7%** at the median request length
-- 23.5 ms of gather against 8.8 ms of attention per decode step -- and the decision has been
outstanding ever since, because Triton has no Windows wheel and this project is developed on
Windows. This is the script that settles it.

**Correctness runs first and the benchmark does not start if it fails.** That ordering is the
whole design. The kernel has never executed anywhere at the time of writing: it was written against
:func:`~edgerag.cache.fused.paged_attention_reference`, which is tested locally across every block
boundary, but a Triton translation of a correct algorithm is still an unrun program. Timing an
incorrect kernel produces a fast wrong number that looks like a result -- which is ``BUGS.md``
B-05's lesson stated in advance rather than after another wasted session.

**No model weights are loaded**, for the same reason ``scripts/colab_gather_overhead.py`` loads
none: gather cost and attention cost are functions of tensor shape and memory layout, not of the
values in the tensors. Two minutes and no nine-gigabyte download, and the sweep can cover lengths
a real model would not fit.

**Prediction, recorded before the run.** The fused path deletes the copy -- 72.7% of the measured
path -- but replaces SDPA's tuned kernel with a hand-written block loop, so it does not get the
whole 3.7x that removing 72.7% would imply. Expect **2-3x on the attention path at the median
request length, and possibly a loss at short lengths**, where the copy is small and kernel launch
overhead is the larger term. A result above 3.7x means the comparison is wrong, not that the kernel
is remarkable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from bench.metrics import assert_device_trusted, sync
from edgerag.cache.fused import (
    HAS_TRITON,
    fused_paged_attention,
    paged_attention_reference,
    triton_unavailable_reason,
)
from edgerag.core.loader import HEADLINE_MODEL, load_spec
from edgerag.core.spec import ModelSpec
from scripts.colab_quant_ablation import code_version

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = uuid.uuid4().hex[:12]

#: Lengths the equivalence gate sweeps. Every one either side of a block boundary, because the
#: partial final block is where ``BUGS.md`` P-01 lives and where a mask off by one hides.
GATE_LENGTHS = (1, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 511, 512, 513, 6758, 6759)

#: Lengths the benchmark sweeps. 6,758 is the frozen trace's median request.
BENCH_LENGTHS = (128, 512, 1024, 2048, 4096, 6758, 7992)


def build_pool(
    spec: ModelSpec, num_blocks: int, block_size: int, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Head-major synthetic pool, the layout the real one uses (``CONTEXT.md`` P6)."""
    shape = (spec.n_kv_heads, num_blocks, block_size, spec.head_dim)
    return (
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(shape, device=device, dtype=dtype),
    )


def gather_then_sdpa(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
    scaling: float,
    n_rep: int,
) -> torch.Tensor:
    """The shipping path, reproduced exactly: one copy out of the pool, then SDPA over it."""
    n_kv_heads, _, _, head_dim = key_pool.shape
    keys = key_pool[:, block_ids].reshape(n_kv_heads, -1, head_dim)[:, :seq_len].unsqueeze(0)
    values = value_pool[:, block_ids].reshape(n_kv_heads, -1, head_dim)[:, :seq_len].unsqueeze(0)
    if n_rep > 1:
        keys = keys.repeat_interleave(n_rep, dim=1)
        values = values.repeat_interleave(n_rep, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(query, keys, values, scale=scaling)
    return out.transpose(1, 2).contiguous()


def _time(fn, iters: int, warmup: int) -> list[float]:
    """Warmup discarded, ``cuda.synchronize`` on both sides (``00_FOUNDATIONS.md`` §4 rules 1-2)."""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - start)
    return samples


def run_gate(
    spec: ModelSpec, block_size: int, device: str, dtype: torch.dtype, tolerance: float
) -> list[dict[str, Any]]:
    """Fused == reference across every block boundary. Returns one row per length.

    fp16 rather than fp32 because that is what ships, and because a kernel that agrees in fp32 and
    diverges in fp16 has an accumulation-order problem the shipping path would actually hit.
    """
    scaling = spec.head_dim**-0.5
    n_rep = spec.n_q_heads // spec.n_kv_heads
    rows = []

    for seq_len in GATE_LENGTHS:
        n_blocks = math.ceil(seq_len / block_size)
        pool_blocks = n_blocks + 4  # slack the sequence does not own, and must not read
        key_pool, value_pool = build_pool(spec, pool_blocks, block_size, device, dtype)
        # Poison the slack, so a mask that is off by one shows up as a large error rather than a
        # subtle one. The realistic version -- plausible KV from a previous tenant -- is far
        # harder to see, which is exactly why the gate makes it loud.
        key_pool[:, n_blocks:] = 1e4
        value_pool[:, n_blocks:] = 1e4

        block_ids = torch.randperm(pool_blocks, device=device)[:n_blocks]
        query = torch.randn(1, spec.n_q_heads, 1, spec.head_dim, device=device, dtype=dtype)

        reference = paged_attention_reference(
            query, key_pool, value_pool, block_ids, seq_len, scaling, n_rep
        )
        fused = fused_paged_attention(
            query, key_pool, value_pool, block_ids, seq_len, scaling, n_rep
        )
        # Against the shipping path too, not only against the reference. The reference is tested
        # locally against gather+SDPA, but re-checking here means a shared misunderstanding of the
        # pool layout cannot pass by agreeing with itself.
        shipping = gather_then_sdpa(
            query, key_pool, value_pool, block_ids, seq_len, scaling, n_rep
        )

        delta_ref = (fused - reference).abs().max().item()
        delta_ship = (fused - shipping).abs().max().item()
        rows.append({
            "seq_len": seq_len,
            "max_abs_delta_vs_reference": delta_ref,
            "max_abs_delta_vs_gather_sdpa": delta_ship,
            "passed": delta_ref <= tolerance and delta_ship <= tolerance,
        })
        del key_pool, value_pool
        torch.cuda.empty_cache()
    return rows


def benchmark(
    spec: ModelSpec,
    block_size: int,
    device: str,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
) -> list[dict[str, Any]]:
    """Time both paths at each length, plus the gather alone so D19's fraction is reproduced."""
    scaling = spec.head_dim**-0.5
    n_rep = spec.n_q_heads // spec.n_kv_heads
    rows = []

    for seq_len in BENCH_LENGTHS:
        n_blocks = math.ceil(seq_len / block_size)
        key_pool, value_pool = build_pool(spec, n_blocks + 2, block_size, device, dtype)
        block_ids = torch.randperm(n_blocks + 2, device=device)[:n_blocks]
        query = torch.randn(1, spec.n_q_heads, 1, spec.head_dim, device=device, dtype=dtype)

        n_kv_heads, _, _, head_dim = key_pool.shape

        def gather_only(
            kp=key_pool, vp=value_pool, ids=block_ids, sl=seq_len, h=n_kv_heads, d=head_dim
        ):
            """Exactly what ``PagedKVCache.gather`` does: **both** pools, and nothing else.

            Two corrections live in this function, and both of them moved the number it reports.

            It once ended in ``.contiguous()``, which the shipping path never calls -- it hands
            the sliced view straight to SDPA -- so it was timing a second copy nothing performs.

            It then gathered only the *key* pool. ``PagedKVCache.gather`` calls ``_gather_pool``
            twice, once per pool, so half the copy was missing from a column whose entire job is
            to say how large the copy is. That under-reported the gather by ~2x in the first T4
            run, and the fraction printed there (22.1% at the median) should be read as roughly
            half of the truth.

            The advanced index is the copy; the reshape, the slice and the unsqueeze are views.
            """
            keys = kp[:, ids].reshape(h, -1, d)[:, :sl].unsqueeze(0)
            values = vp[:, ids].reshape(h, -1, d)[:, :sl].unsqueeze(0)
            return keys, values

        shipping = _time(
            lambda q=query, kp=key_pool, vp=value_pool, ids=block_ids, sl=seq_len: (
                gather_then_sdpa(q, kp, vp, ids, sl, scaling, n_rep)
            ),
            iters, warmup,
        )
        fused = _time(
            lambda q=query, kp=key_pool, vp=value_pool, ids=block_ids, sl=seq_len: (
                fused_paged_attention(q, kp, vp, ids, sl, scaling, n_rep)
            ),
            iters, warmup,
        )
        gather = _time(gather_only, iters, warmup)

        ship_ms = statistics.median(shipping) * 1e3
        fused_ms = statistics.median(fused) * 1e3
        gather_ms = statistics.median(gather) * 1e3
        # "A single number is not a result" (00_FOUNDATIONS.md §4 rule 4). The first T4 run showed
        # the *baseline* running faster at 7,992 tokens than at 6,758, which is impossible and was
        # simply run-to-run spread being reported as a point estimate. Carrying the relative
        # deviation makes a speedup that sits inside the noise visible as one.
        ship_rsd = statistics.stdev(shipping) / statistics.mean(shipping) if iters > 1 else 0.0
        fused_rsd = statistics.stdev(fused) / statistics.mean(fused) if iters > 1 else 0.0
        rows.append({
            "seq_len": seq_len,
            "gather_sdpa_ms": ship_ms,
            "fused_ms": fused_ms,
            "gather_alone_ms": gather_ms,
            "gather_sdpa_rsd": ship_rsd,
            "fused_rsd": fused_rsd,
            # D19's headline fraction, recomputed here so the two runs can be compared directly.
            "gather_fraction_of_shipping": gather_ms / ship_ms if ship_ms else 0.0,
            "speedup": ship_ms / fused_ms if fused_ms else 0.0,
            "iters": iters,
        })
        del key_pool, value_pool
        torch.cuda.empty_cache()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fused paged attention: gate then benchmark (T4)")
    parser.add_argument("--drive", default="")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=2e-3,
                        help="fp16 max abs delta; the shipping path is fp16 and so is this")
    parser.add_argument("--out-name", default="fused_attention.jsonl")
    parser.add_argument("--skip-gate", action="store_true",
                        help="benchmark without checking correctness first. Do not.")
    parser.add_argument("--allow-untrusted-device", action="store_true")
    args = parser.parse_args(argv)

    info = assert_device_trusted(allow_untrusted=args.allow_untrusted_device)
    print(f"device: {info.name}")

    if not HAS_TRITON:
        print(f"\nFAILED: no Triton -- {triton_unavailable_reason()}\n"
              "The fused kernel is the entire subject of this script; there is nothing to measure "
              "without it.", file=sys.stderr)
        return 1

    spec = load_spec(args.model)
    print(f"{args.model}: {spec.n_q_heads} q heads, {spec.n_kv_heads} kv heads, "
          f"head_dim {spec.head_dim}, block size {args.block_size}")
    print("\nPREDICTION (recorded before the run): 2-3x on the attention path at the median")
    print("request length, possibly a loss at short lengths. Removing 72.7% of the path implies")
    print("3.7x; a hand-written loop does not get all of it, and >3.7x means the comparison is")
    print("wrong rather than the kernel remarkable.\n")

    gate: list[dict[str, Any]] = []
    if not args.skip_gate:
        print("=== equivalence gate: fused vs reference vs gather+SDPA, fp16 ===")
        gate = run_gate(spec, args.block_size, "cuda", torch.float16, args.tolerance)
        for row in gate:
            mark = "ok " if row["passed"] else "FAIL"
            print(f"  {mark} seq_len {row['seq_len']:>5}  vs reference "
                  f"{row['max_abs_delta_vs_reference']:.2e}  vs gather+SDPA "
                  f"{row['max_abs_delta_vs_gather_sdpa']:.2e}")
        if not all(row["passed"] for row in gate):
            failed = [row["seq_len"] for row in gate if not row["passed"]]
            print(f"\nFAILED at seq_len {failed}. Not timing an incorrect kernel -- a fast wrong "
                  "number\nlooks exactly like a result. Lengths straddling a block boundary "
                  f"({args.block_size}) failing\nmeans the P-01 mask; lengths failing everywhere "
                  "means the accumulation.", file=sys.stderr)
            return 1
        print("  gate GREEN -- the kernel agrees with both the reference and the shipping path.\n")

    print("=== benchmark ===")
    rows = benchmark(spec, args.block_size, "cuda", torch.float16, args.iters, args.warmup)
    print(f"{'seq_len':>8}{'gather+SDPA':>13}{'+/-':>7}{'fused':>9}{'+/-':>7}"
          f"{'gather K+V':>12}{'gather %':>10}{'speedup':>9}")
    for row in rows:
        print(f"{row['seq_len']:>8}{row['gather_sdpa_ms']:>12.3f}m"
              f"{row['gather_sdpa_rsd']:>6.1%}{row['fused_ms']:>8.3f}m{row['fused_rsd']:>6.1%}"
              f"{row['gather_alone_ms']:>11.3f}m"
              f"{row['gather_fraction_of_shipping']:>9.1%}{row['speedup']:>8.2f}x")

    noisy = [r for r in rows if max(r["gather_sdpa_rsd"], r["fused_rsd"]) > 0.10]
    if noisy:
        print(f"\n  {len(noisy)} row(s) with >10% run-to-run spread: "
              f"{', '.join(str(r['seq_len']) for r in noisy)}. Their speedups are not")
        print("  distinguishable from their neighbours' -- read the trend, not the individual "
              "cells.")

    median_row = next((r for r in rows if r["seq_len"] == 6758), rows[-1])
    print(f"\n  At the trace's median request ({median_row['seq_len']} tokens): "
          f"**{median_row['speedup']:.2f}x**, with the gather")
    print(f"  measured at {median_row['gather_fraction_of_shipping']:.1%} of the shipping path "
          "(D19 measured 72.7%).")
    print("  D3's revisit threshold was 25%. This is the number that closes that decision.")

    payload = {
        "model": args.model,
        "block_size": args.block_size,
        "n_q_heads": spec.n_q_heads,
        "n_kv_heads": spec.n_kv_heads,
        "head_dim": spec.head_dim,
        "dtype": "float16",
        "gate": gate,
        "gate_skipped": args.skip_gate,
        "benchmark": rows,
        "code_version": code_version(),
        "session_id": SESSION_ID,
        "device": info.name,
        "trusted": info.trusted,
    }
    out_dir = Path(args.drive) if args.drive else (REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
        fh.flush()
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
