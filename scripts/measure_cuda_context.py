"""What the CUDA context costs, measured instead of cited. Closes ``CONTEXT.md`` P3.

    python -m scripts.measure_cuda_context

Every memory table in this project ends with the same sentence: *excludes the CUDA context, 300-600
MiB on Turing, **not measured on this device***. That parenthetical has been carried since Phase 0
and it is the last unmeasured term in the budget. D26 made it urgent rather than tidy: the serving
configuration now lands at **4.000 GiB of a 4.00 GiB budget**, so there is exactly zero slack to
absorb a term nobody has weighed.

**The context is not one number, and that is the finding this script is shaped around.** The driver
allocates when the context is created, and again as CUDA modules are loaded -- modern CUDA defaults
to `CUDA_MODULE_LOADING=LAZY`, so kernels arrive as they are first used. Measuring straight after
``torch.cuda.init()`` gives a floor that a real pipeline blows straight through. So this walks
stages and reports each: bare init, after a realistic block pool exists, after actual attention and
GEMM kernels have run.

**How it is measured.** ``torch.cuda.mem_get_info()`` reports free and total memory *for the
device*, which includes everything the driver holds. PyTorch's own reservation is
``torch.cuda.memory_reserved()``. The difference is everything else::

    context = (total - free) - memory_reserved()

That subtraction is why this cannot use ``max_memory_allocated``: the whole point is the memory
that sits **outside** what PyTorch can see, which is exactly the memory a budget defined on
``max_memory_allocated`` (P3) silently omits.

**Not gated on the reference device, deliberately.** ``CONTEXT.md`` D4 keeps *latency* off
untrusted hardware because a GTX 1650 has no tensor cores and its timings are architecturally
incomparable. Context size is not a speed measurement -- it is a property of the driver, the device
and the CUDA version, and the interesting question is precisely what it costs on *different*
hardware. The GTX 1650 is the more informative of the two here, because it is a physically 4 GB
card and therefore the literal subject of "would this fit on a 4 GB device".
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2
GIB = 1024**3


def device_used_bytes_via_smi() -> int | None:
    """Device memory in use, asked of the driver **without creating a context**.

    This is the baseline the whole measurement rests on. ``torch.cuda.mem_get_info`` cannot
    provide it: calling it creates the very context being measured. ``nvidia-smi`` is a separate
    process and reads the driver directly, so it can be asked *before* this one has touched CUDA.

    Without this, the measurement reports every other tenant as though it were the context. On the
    development box -- a Windows desktop with a display attached to the same card -- that is
    801 MiB of compositor and browser, which is not what the budget needs to know about.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(result.stdout.strip().splitlines()[0]) * MIB
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


def context_bytes(baseline: int | None) -> int | None:
    """Device memory this process holds outside PyTorch's caching allocator.

    **One instrument, differenced.** ``nvidia-smi`` supplies both the pre-init baseline and every
    later reading; PyTorch's own arena comes from ``memory_reserved()`` and is subtracted off::

        context = (smi_now - smi_before_this_process) - memory_reserved()

    An earlier version took the baseline from ``nvidia-smi`` and the later readings from
    ``torch.cuda.mem_get_info()``, and produced a **negative context**. The two do not share an
    accounting: under Windows WDDM the OS manages GPU memory and ``nvidia-smi`` reports allocations
    that ``mem_get_info``'s CUDA-visible view does not. Subtracting one from the other is not a
    measurement, and the negative number was the only reason that was obvious.

    ``max_memory_allocated`` is unusable here by construction: the quantity of interest is
    precisely the memory PyTorch *cannot* see, which is what a budget defined on it omits (P3).
    """
    now = device_used_bytes_via_smi()
    if now is None or baseline is None:
        return None
    return (now - baseline) - torch.cuda.memory_reserved()


def other_processes() -> list[dict[str, Any]]:
    """Anything else holding memory on this GPU, because it would land in the measurement.

    Best-effort: a missing or unparseable ``nvidia-smi`` returns nothing rather than raising, and
    the caller reports that it could not check instead of claiming the GPU was clean.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        result = subprocess.run(
            [exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    apps = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        pid, _, used = line.partition(",")
        # Windows/WDDM reports per-process memory as "[N/A]" -- the driver owns the allocation and
        # will not attribute it. The *count* of processes is still the thing that matters here
        # (a second tenant invalidates the measurement), so an unparseable size records None
        # rather than raising and taking the whole run down with it.
        try:
            used_mib: int | None = int(used.strip())
        except ValueError:
            used_mib = None
        apps.append({"pid": pid.strip(), "used_mib": used_mib})
    return apps


def stage_init() -> None:
    """Create the context and nothing else."""
    torch.cuda.init()
    torch.cuda.synchronize()


def stage_pool(num_blocks: int, block_size: int, kv_bytes_per_token: int) -> torch.Tensor:
    """Allocate a realistic block pool, so the caching allocator is in its steady state."""
    n_elements = num_blocks * block_size * kv_bytes_per_token // 2  # fp16
    pool = torch.zeros(n_elements, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    return pool


def stage_kernels(pool: torch.Tensor) -> None:
    """Run the kernel families the decode path actually uses, so lazy modules are loaded.

    A GEMM, an SDPA call and an advanced-index gather. Under ``CUDA_MODULE_LOADING=LAZY`` -- the
    default on modern CUDA -- none of these are resident until first use, so a context measured
    before them understates what a running pipeline holds. That understatement is the entire
    failure mode this stage exists to expose.
    """
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    (a @ a).sum()

    q = torch.randn(1, 8, 1, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
    torch.nn.functional.scaled_dot_product_attention(q, k, k)

    index = torch.randint(0, max(1, pool.numel() // 1024), (64,), device="cuda")
    pool.view(-1, 1024)[index].sum()
    torch.cuda.synchronize()


def measure(
    num_blocks: int, block_size: int, kv_bytes_per_token: int, baseline: int | None
) -> dict[str, Any]:
    """Walk the stages, recording the context after each.

    ``baseline`` must have been taken *before* this process touched CUDA, or it is not a baseline.
    """
    stages: dict[str, int] = {}

    stage_init()
    stages["after_init"] = context_bytes(baseline)

    pool = stage_pool(num_blocks, block_size, kv_bytes_per_token)
    stages["after_pool"] = context_bytes(baseline)

    stage_kernels(pool)
    stages["after_kernels"] = context_bytes(baseline)

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    del pool
    torch.cuda.empty_cache()

    return {
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": total,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "baseline_used_bytes": baseline,
        "stages_bytes": stages,
        "context_bytes": stages["after_kernels"],
        "pool_blocks": num_blocks,
        "block_size": block_size,
        "other_processes": other_processes(),
        "free_after_bytes": free,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the CUDA context (P3). Any device.")
    parser.add_argument("--drive", default="")
    parser.add_argument("--num-blocks", type=int, default=471,
                        help="D26's budget-derived pool; only its size matters, not its contents")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-bytes-per-token", type=int, default=192 * 1024,
                        help="SmolVLM2-2.2B: 24 layers x 2 x 32 kv-heads x 64 dim x fp16")
    parser.add_argument("--budget-gib", type=float, default=4.0)
    parser.add_argument("--out-name", default="cuda_context.json")
    args = parser.parse_args(argv)

    # Taken first, before anything in this process has spoken to CUDA. Any torch call that
    # initialises the driver -- including torch.cuda.is_available() on some builds -- would fold
    # the context into its own baseline and measure it as zero.
    #
    # Two reads, because the baseline is only a baseline if it holds still. A card with a display
    # attached moves by hundreds of MiB on its own, and that drift lands directly on a figure
    # derived by subtraction.
    baseline = device_used_bytes_via_smi()
    time.sleep(0.4)
    baseline_again = device_used_bytes_via_smi()
    drift = (
        abs(baseline_again - baseline)
        if baseline is not None and baseline_again is not None
        else None
    )
    before = other_processes()

    if not torch.cuda.is_available():
        print("no CUDA device: the context is a driver allocation and there is nothing to "
              "measure without one.", file=sys.stderr)
        return 1

    payload = measure(args.num_blocks, args.block_size, args.kv_bytes_per_token, baseline)

    print(f"device: {payload['device']} "
          f"({payload['total_memory_bytes'] / GIB:.2f} GiB, CUDA {payload['cuda_version']}, "
          f"torch {payload['torch_version']})\n")

    if payload["baseline_used_bytes"] is None:
        print("FAILED: no pre-init baseline -- nvidia-smi is unavailable, and without it the "
              "context\ncannot be separated from every other tenant on the card.", file=sys.stderr)
        return 1

    print(f"  baseline before this process touched CUDA: "
          f"{payload['baseline_used_bytes'] / MIB:.0f} MiB held by "
          f"{len(before)} other process(es), subtracted from everything below.")
    if drift:
        print(f"  baseline drift over 0.4 s: {drift / MIB:.0f} MiB")

    print(f"\n{'stage':>16}{'context':>12}{'delta':>10}")
    previous = 0
    labels = {
        "after_init": "bare context",
        "after_pool": "+ block pool",
        "after_kernels": "+ real kernels",
    }
    for stage, value in payload["stages_bytes"].items():
        if value is None:
            continue
        print(f"{labels[stage]:>16}{value / MIB:>10.0f}M{(value - previous) / MIB:>9.0f}M")
        previous = value

    context = payload["context_bytes"]

    # Two different refusals, and keeping them apart matters. The first is about the *value* being
    # impossible; the second is about the *instrument* being unfit. Refusing on "the number looks
    # too small" would be assuming the answer -- the whole reason to measure this is that nobody
    # here knows what it is on a given card. So the quality gate reads the card, not the result.
    if context is None or context < 0:
        print(f"\n  ** REFUSING: residual is {context}, and a context cannot be negative. **",
              file=sys.stderr)
        print("  The baseline and the readings disagree about what counts as used memory.",
              file=sys.stderr)
        return 1

    tenants = len(before)
    noisy = drift is not None and drift > 0.10 * max(context, 1)
    if tenants > 1 or noisy:
        print(f"\n  ** REFUSING TO PUBLISH: this card is not a usable instrument for a "
              f"{context / MIB:.0f} MiB signal. **")
        if tenants > 1:
            print(f"  {tenants} other processes hold memory on it -- a display is attached and "
                  "the desktop")
            print("  compositor moves more than the quantity being measured.")
        if noisy:
            print(f"  The baseline drifted {drift / MIB:.0f} MiB while idle, against a "
                  f"{context / MIB:.0f} MiB residual.")
        print(f"\n  The residual came out at {context / MIB:.0f} MiB. It is recorded below as "
              "indicative and")
        print("  marked untrusted; it is NOT the figure to put in the budget table.")
        print("  Measure on a card with one tenant and no display -- a Colab T4 is exactly that,")
        print("  and is the device the budget is defined against anyway.")
        payload["trusted"] = False
    else:
        payload["trusted"] = True
    budget = args.budget_gib * GIB
    claim = (
        f"**The CUDA context costs {context / MIB:.0f} MiB**"
        if payload["trusted"]
        else f"The residual was {context / MIB:.0f} MiB"
    )
    print(f"\n  {claim} once the kernels a decode step uses are resident.")
    stages = payload["stages_bytes"]
    kernel_cost = stages["after_kernels"] - stages["after_pool"]
    print(f"  Loading them adds {kernel_cost / MIB:.0f} MiB over a bare context, so a figure "
          "taken at init understates it.")
    if not payload["trusted"]:
        print("  (untrusted, per the refusal above -- indicative only)")
    print(f"\n  Against the {args.budget_gib:.0f} GiB pipeline budget that is "
          f"{context / budget:.1%} of it, and the budget")
    print("  is defined on max_memory_allocated (P3), which cannot see this memory at all.")

    total = payload["total_memory_bytes"]
    print(f"  On this {total / GIB:.2f} GiB card, {(total - context) / GIB:.2f} GiB remains for "
          "the pipeline.")
    if budget > total - context:
        print(f"  ** A {args.budget_gib:.0f} GiB pipeline does NOT fit on this device once the "
              f"context is counted. **")
        print(f"  The largest pipeline that does is {(total - context) / GIB:.2f} GiB.")

    payload["budget_gib"] = args.budget_gib
    payload["pipeline_headroom_bytes"] = total - context
    out_dir = Path(args.drive) if args.drive else (REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8")).get("measurements", [])
    # Keyed by device: the context is a property of driver and card, so two devices are two
    # findings rather than one superseding the other.
    existing = [m for m in existing if m["device"] != payload["device"]]
    out_path.write_text(
        json.dumps({"measurements": [*existing, payload]}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
