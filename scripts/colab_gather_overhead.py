"""What the gather in paged attention actually costs. T4 only.

    python -m scripts.colab_gather_overhead --drive /content/drive/MyDrive/edgerag

``CONTEXT.md`` D3 chose gather-into-scratch plus standard attention over a fused Triton kernel, on
the grounds that the memory thesis is fully preserved and only a copy is added -- and promised to
**measure that copy and publish it**. This is that measurement. Without it, "why didn't you write
a fused kernel?" is answered with a shrug, which is the weakest possible answer to the most
obvious question about the design.

**No model weights are loaded.** Gather cost is a function of tensor shape and memory layout, not
of the values in the tensors, so this runs from the spec alone with synthetic KV. That makes it a
two-minute cell rather than a nine-gigabyte download, and it lets the sweep cover sequence lengths
and block sizes that a real model would not fit.

Three things are timed, all with ``cuda.synchronize`` bracketing (``00_FOUNDATIONS.md`` §4 rule 2):

1. ``gather`` -- ``pool[block_ids]`` plus the reshape/transpose/contiguous that follows.
2. ``attention`` -- the two matmuls and the softmax over the gathered tensors.
3. ``naive`` -- the same attention over an already-contiguous cache, which is what the gather is
   overhead *relative to*.

The headline figure is ``gather / (gather + attention)``: the fraction of the paged attention path
spent reassembling blocks. D3 says revisit the fused-kernel decision if it exceeds ~25%.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from bench.metrics import assert_device_trusted, sync
from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.paged import PagedKVCache
from edgerag.core.layers import eager_attention
from edgerag.core.loader import HEADLINE_MODEL, load_spec
from edgerag.core.spec import ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _time(fn, iters: int, warmup: int) -> list[float]:
    """Median-of-many timing with warmup discarded and syncs on both sides."""
    for _ in range(warmup):
        fn()
    sync()

    samples: list[float] = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return samples


def measure(
    spec: ModelSpec,
    seq_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    """Time gather, paged attention, and contiguous attention at one operating point."""
    blocks_needed = (seq_len + block_size - 1) // block_size
    allocator = BlockAllocator(num_blocks=blocks_needed + 8, block_size=block_size)
    cache = PagedKVCache(spec, allocator, device, dtype)

    # Fill with synthetic KV -- values are irrelevant to gather cost, shapes are not.
    kv = torch.randn(1, spec.n_kv_heads, seq_len, spec.head_dim, device=device, dtype=dtype)
    for layer in range(spec.n_layers):
        cache.update(kv, kv, layer)

    # One decode step: a single query attending over the whole history.
    query = torch.randn(1, spec.n_q_heads, 1, spec.head_dim, device=device, dtype=dtype)
    scaling = spec.head_dim**-0.5

    gathered_k, gathered_v = cache.gather(0)
    contiguous_k = gathered_k.clone()
    contiguous_v = gathered_v.clone()

    gather_s = _time(lambda: cache.gather(0), iters, warmup)
    paged_attn_s = _time(
        lambda: eager_attention(query, gathered_k, gathered_v, None, scaling, spec.n_rep),
        iters,
        warmup,
    )
    naive_attn_s = _time(
        lambda: eager_attention(query, contiguous_k, contiguous_v, None, scaling, spec.n_rep),
        iters,
        warmup,
    )

    gather = statistics.median(gather_s)
    attn = statistics.median(paged_attn_s)
    naive = statistics.median(naive_attn_s)

    # Per-layer costs scale to a whole decode step: every layer gathers and attends.
    step_gather = gather * spec.n_layers
    step_attn = attn * spec.n_layers

    result = {
        "seq_len": seq_len,
        "block_size": block_size,
        "n_blocks": blocks_needed,
        "gather_ms": gather * 1e3,
        "paged_attention_ms": attn * 1e3,
        "contiguous_attention_ms": naive * 1e3,
        "gather_fraction": gather / (gather + attn) if (gather + attn) else 0.0,
        "step_gather_ms": step_gather * 1e3,
        "step_attention_ms": step_attn * 1e3,
        "kv_mib": cache.nbytes / (1024**2),
    }
    cache.free()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gather overhead in paged attention (T4)")
    parser.add_argument("--drive", default="")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[512, 2048, 4096, 6800, 8192]
    )
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--allow-untrusted-device", action="store_true")
    args = parser.parse_args(argv)

    info = assert_device_trusted(allow_untrusted=args.allow_untrusted_device)
    print(f"device: {info.name} | tensor cores {info.has_tensor_cores}")

    spec = load_spec(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"{spec.model_id}: {spec.n_layers} layers, {spec.n_kv_heads} kv-heads, "
          f"{spec.kv_bytes_per_token() / 1024:.0f} KiB/token\n")

    rows: list[dict[str, Any]] = []
    print(f"{'seq':>6} {'block':>6} {'gather ms':>10} {'attn ms':>9} "
          f"{'gather %':>9} {'step gather ms':>15}")

    for block_size in args.block_sizes:
        for seq_len in args.seq_lens:
            try:
                row = measure(
                    spec, seq_len, block_size, device, dtype, args.iters, args.warmup
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{seq_len:>6} {block_size:>6}  OOM")
                continue
            rows.append(row)
            print(f"{seq_len:>6} {block_size:>6} {row['gather_ms']:>10.3f} "
                  f"{row['paged_attention_ms']:>9.3f} {row['gather_fraction']:>8.1%} "
                  f"{row['step_gather_ms']:>15.2f}")
            torch.cuda.empty_cache()

    out_dir = Path(args.drive) if args.drive else (REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gather_overhead.json"
    out_path.write_text(
        json.dumps({"device": info.name, "model": spec.model_id, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    if rows:
        realistic = [r for r in rows if r["block_size"] == 16 and r["seq_len"] >= 4096]
        if realistic:
            worst = max(r["gather_fraction"] for r in realistic)
            print(f"\nAt block_size=16 and realistic lengths, gather is at most {worst:.1%} "
                  "of the paged attention path.")
            verdict = (
                "exceeds the 25% threshold -- D3 says revisit the fused-kernel decision"
                if worst > 0.25
                else "below the 25% threshold -- D3's gather-plus-SDPA choice stands"
            )
            print(f"  {verdict}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
