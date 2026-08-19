"""Phase 5e: the serving layer under Poisson load. Throughput and p99 TTFT against offered load.

    python -m scripts.colab_poisson --drive /content/drive/MyDrive/edgerag

The one gate the project built and never measured. Phases 5a-5d proved the scheduler *correct* --
admission, chunked prefill, preemption, pool conservation, all property-tested without a GPU --
and then Phase 5e, the half that turns a correct scheduler into a number, was cut when the
schedule ran out. Everything downstream of that is still blank: the throughput-vs-concurrency
plot, the p99 TTFT figure, and the two empty slots in ``01_EDGERAG.md`` §8's CV bullet.

**Predictions recorded before measuring, so the result is allowed to disagree** (the same
discipline ``PLAN.md`` Phase 5 applied to the batching work):

1. **Aggregate throughput will rise far less than concurrency does.** D14 measured 1.32x aggregate
   for 4x batch on this checkpoint, because it is MHA (32 query heads, 32 KV heads) and decode is
   bandwidth-bound on KV reads at 192 KiB/token. A scheduler moves work around; it does not move
   bytes faster. If throughput scales near-linearly here, distrust the measurement before
   believing it.
2. **Chunked prefill will win on p99 TTFT and lose slightly on throughput.** Fourteen 512-token
   chunks cost more in kernel-launch and attention-setup overhead than one 6,758-token pass. What
   they buy is that a decoding request waits one chunk rather than one whole prefill.
3. **p99 TTFT will degrade superlinearly once offered load passes service rate**, because an open
   loop has no backpressure -- that is queueing theory, not a property of this code, and seeing it
   is a check that the driver is really open-loop.

**Measured 2026-08-19 -- ``CONTEXT.md`` D25. Predictions 1 and 3 hold; prediction 2 is wrong.**
Chunked prefill costs **1.51x on p95 TTFT** and buys **4.2x on the decode phase**, netting 1.23x
end to end. The prediction had the mechanism right and the metric wrong: chunking protects
*decode* from head-of-line blocking exactly as ``BUGS.md`` P-18 describes, and the TTFT it loses
is admission queueing caused by ``max_prefills_per_step=1`` -- a 15-chunk prefill holds the single
prefill slot for 15 iterations, during which nothing new is admitted. The predictions above are
left exactly as they were written; editing them after the fact would destroy the only thing that
makes recording them worth doing.

**Two traps this script is shaped around, both of which produce a confident null result.**

*The chunk-size knob is on the executor, not the scheduler.* ``SchedulerConfig.prefill_chunk_size``
exists and is read by nothing; ``ModelExecutor.chunk_size`` is what actually slices a prefill.
Sweeping the former would produce two identical arms and the conclusion "chunked prefill does not
help".

*The ship pool holds exactly one request.* 640 blocks x 16 tokens is 10,240 token-slots against a
median RAG request of ~6,758 + generation = ~425 blocks. A concurrency sweep there is a flat line
describing the pool, not the scheduler. So the sweep deliberately runs a **larger pool than the
4 GiB budget allows** and reports both: what the scheduler does when given room, and the fact that
the budget -- not the scheduler -- is what caps concurrency at the shipped configuration. That
contrast is the finding, and it is printed whether or not anyone reads the JSON.

Resumable per cell, fsync'd per cell, and every record carries its session id and git SHA
(``CONTEXT.md`` D24 findings 3 and 5: latency compared across sessions or across code versions is
not comparable, and a file that cannot be checked for it will be misread).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from bench.load import LoadResult, drain, replay_poisson
from bench.metrics import MemoryProbe, Percentiles, assert_device_trusted
from bench.serving import build_stack
from edgerag.core.loader import HEADLINE_MODEL
from edgerag.retrieval.trace import load_trace, trace_fingerprint
from edgerag.sched.request import Request
from scripts.colab_quant_ablation import arm_spec, code_version, host_and_gpu_free

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3

#: One value per process, stamped into every record. See D24 finding 3.
SESSION_ID = uuid.uuid4().hex[:12]

#: Offered load as a multiple of the *measured* single-request service rate. Expressed as a ratio
#: rather than in requests/second so the sweep self-calibrates: the same numbers mean the same
#: thing on fp16 (13.9 tok/s) and INT4 (3.3 tok/s), which they would not if the rates were fixed.
DEFAULT_LOAD_FACTORS = (0.5, 1.0, 2.0, 4.0)

#: Effectively "no chunking": larger than any prompt in the trace, so a prefill completes in one
#: iteration and blocks every decoding request behind it for its full duration.
UNCHUNKED = 32768


def build_prompts(stack: Any, questions: list[str], verbose: bool = True) -> list[Any]:
    """Retrieve, encode and merge every prompt **before** the timed window opens.

    Deliberate, and the reason is that these two costs do not vary with concurrency. Retrieval and
    the vision tower run once per request on the caller's thread -- ``edgerag/serve/pipeline.py``
    says so and explains why -- so including them would add a near-constant to every TTFT and
    shrink the *relative* effect of the queueing this experiment exists to measure. The constant is
    separately measurable and reported as ``prompt_build_s`` below rather than dropped.

    The cost of doing it this way is memory: one prompt's merged embeddings are
    ``prompt_len x hidden x 2`` bytes -- about 27 MiB at 6,758 tokens and hidden 2048 -- and they
    are all held for the whole sweep. That is why the default count is small and why the figure is
    printed rather than left for someone to discover as an OOM.
    """
    prompts = []
    started = time.perf_counter()
    for i, question in enumerate(questions):
        prompts.append(stack.rag.build(question))
        if verbose and (i + 1) % 4 == 0:
            print(f"    built {i + 1}/{len(questions)} prompts")
    elapsed = time.perf_counter() - started

    lengths = [p.n_tokens for p in prompts]
    embed_bytes = sum(p.embeds.numel() * p.embeds.element_size() for p in prompts)
    if verbose:
        print(f"    {len(prompts)} prompts, {min(lengths)}-{max(lengths)} tokens "
              f"(median {sorted(lengths)[len(lengths) // 2]}), holding "
              f"{embed_bytes / GIB:.2f} GiB of embeddings, built in {elapsed:.1f}s "
              f"({elapsed / len(prompts):.2f}s each)")
    return prompts


def request_factory(prompts: list[Any], cell: str, max_new_tokens: int):
    """Build request *i* of a cell, cycling through the prepared prompts.

    Request ids carry the cell name so that a JSON file holding several cells can be read back
    per-request without a join, and so two cells' ids can never collide in the driver's dict.
    """

    def make(i: int) -> Request:
        prompt = prompts[i % len(prompts)]
        return Request(
            request_id=f"{cell}-{i:03d}",
            prompt_token_ids=list(prompt.token_ids),
            max_new_tokens=max_new_tokens,
            prompt_embeds=prompt.embeds,
        )

    return make


def calibrate(stack: Any, prompts: list[Any], max_new_tokens: int, verbose: bool = True) -> float:
    """Serve one request alone, twice, and return the second one's end-to-end seconds.

    The first call is warmup and discarded -- first-call kernel selection and autotuning otherwise
    land inside the number every offered load is then derived from (``00_FOUNDATIONS.md`` §4
    rule 1). Two calls, not five: this is a scale factor for the sweep, not a published latency,
    and the published single-request latencies come from the sweep's own load=0.5 cell.
    """
    for attempt in range(2):
        result = replay_poisson(
            stack.engine, stack.scheduler,
            request_factory(prompts, f"calib{attempt}", max_new_tokens),
            n_requests=1, arrival_rate_hz=1000.0, timeout_s=600.0,
        )
        drain(stack.scheduler, stack.engine)
        service_s = result.outcomes[0].e2e_s if result.outcomes else None
        if service_s is None:
            raise RuntimeError(
                "calibration request never finished. Nothing downstream is measurable; check the "
                "pool size and the console above for an executor error."
            )
        if verbose:
            label = "warmup" if attempt == 0 else "service time"
            print(f"    {label}: {service_s:.2f}s end-to-end, TTFT "
                  f"{result.outcomes[0].ttft_s:.2f}s, {result.outcomes[0].n_tokens} tokens")
    return service_s


def cell_name(load_factor: float, chunk_size: int) -> str:
    """The identity a cell is resumed on."""
    chunk = "unchunked" if chunk_size >= UNCHUNKED else f"chunk{chunk_size}"
    return f"load{load_factor:g}-{chunk}"


def completed_cells(path: Path, n_requests: int) -> set[str]:
    """Cells already measured to at least this invocation's request count.

    Identity alone is not enough -- resuming on "a row for this cell exists" is what let a
    two-request smoke row survive a real run and become the denominator of a summary table
    (``BUGS.md`` B-10). A thin cell is re-measured rather than skipped.
    """
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("n_requests", 0) >= n_requests and not row.get("timed_out"):
            done.add(row["cell"])
    return done


def run_cell(
    stack: Any,
    prompts: list[Any],
    load_factor: float,
    service_s: float,
    chunk_size: int,
    n_requests: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[LoadResult, dict[str, Any]]:
    """One offered-load cell. Returns the driver's result plus what the stack reported around it."""
    name = cell_name(load_factor, chunk_size)
    rate = load_factor / service_s

    # The knob that matters. Mutating it between cells rather than rebuilding the executor is
    # deliberate: a rebuilt executor reallocates the block pool, which would put a different
    # memory state under each arm of a comparison whose whole subject is latency.
    stack.executor.chunk_size = chunk_size
    stack.scheduler.config.prefill_chunk_size = chunk_size

    free_before = stack.allocator.num_free
    # Three times the serial time plus a floor: enough that a saturated cell finishes draining,
    # short enough that a deadlocked one is reported rather than eating the session.
    timeout_s = n_requests * service_s * 3.0 + 120.0

    with MemoryProbe() as probe:
        result = replay_poisson(
            stack.engine, stack.scheduler,
            request_factory(prompts, name, max_new_tokens),
            n_requests=n_requests, arrival_rate_hz=rate, seed=seed, timeout_s=timeout_s,
        )
    idle = drain(stack.scheduler, stack.engine)
    free_after = stack.allocator.num_free

    return result, {
        "peak_allocated_bytes": probe.require().peak_allocated_bytes,
        # Pool conservation across a real load, which the property tests assert against a fake
        # engine and nothing had ever checked against the real one. A leak here silently reduces
        # every *later* cell's concurrency, so it must be checked per cell, not at the end.
        "blocks_free_before": free_before,
        "blocks_free_after": free_after,
        "blocks_leaked": free_before - free_after,
        "engine_idle_after": idle,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5e: Poisson load sweep (T4)")
    parser.add_argument("--drive", default="")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--arm", default="LM8+ViT4",
                        help="quantization arm; the default is D24's measured ship config")
    parser.add_argument("--bits", type=int, default=4, help="ignored for mixed arms")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--load-factors", type=float, nargs="+", default=list(DEFAULT_LOAD_FACTORS))
    parser.add_argument("--n-requests", type=int, default=12,
                        help="requests per cell. p99 needs >=100 and this default cannot give one")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--n-prompts", type=int, default=8,
                        help="distinct prompts held in memory and cycled; ~27 MiB of embeds each")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--num-blocks", type=int, default=1792,
        help=(
            "block pool. 1792 x 16 tokens is ~5.25 GiB and admits 4 concurrent RAG requests; the "
            "shipped 640 admits exactly 1, which would make a concurrency sweep meaningless. This "
            "deliberately exceeds the 4 GiB budget -- see the header note the run prints."
        ),
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-prefills-per-step", type=int, default=1,
        help=(
            "prefills admitted per iteration. 1 is the ship default and is frequently the *real* "
            "concurrency limit rather than the pool: a 6,758-token prompt takes 14 chunked "
            "iterations, so at one prefill per step arrivals queue on the prefill slot long "
            "before they queue on blocks. Raise it to separate the two effects."
        ),
    )
    parser.add_argument("--skip-chunk-comparison", action="store_true")
    parser.add_argument("--chunk-comparison-load", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-name", default="poisson_sweep.jsonl")
    parser.add_argument("--allow-untrusted-device", action="store_true")
    args = parser.parse_args(argv)

    info = assert_device_trusted(allow_untrusted=args.allow_untrusted_device)
    print(f"device: {info.name}\n")
    print("PREDICTIONS, recorded before the run (CONTEXT.md D14, D18):")
    print("  1. aggregate throughput rises far less than concurrency -- decode is KV-bandwidth")
    print("     bound on this MHA checkpoint, and a scheduler cannot move bytes faster.")
    print("  2. chunked prefill wins p99 TTFT and loses a little throughput.")
    print("  3. p99 TTFT degrades superlinearly past the service rate -- an open loop has no")
    print("     backpressure, so this is a check on the driver as much as on the server.\n")

    trace = load_trace()
    fingerprint = trace_fingerprint(trace)
    heldout = [e for e in trace if e.split == "heldout"][: args.n_prompts]
    if not heldout:
        print("FAILED: no held-out trace entries. Run scripts.build_trace first.", file=sys.stderr)
        return 1
    print(f"trace {fingerprint} | {len(heldout)} distinct prompts, cycled\n")

    out_dir = Path(args.drive) if args.drive else (REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    done = completed_cells(out_path, args.n_requests)
    if done:
        print(f"resuming: {len(done)} cell(s) already measured to this standard "
              f"(n>={args.n_requests})\n")

    stack = build_stack(
        model_id=args.model,
        quant_spec={} if args.arm == "fp16" else arm_spec(args.arm, args.bits),
        arm=args.arm,
        group_size=args.group_size,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        chunk_size=args.chunk_size,
        max_batch_size=args.max_batch_size,
        max_prefills_per_step=args.max_prefills_per_step,
    )
    run_started = time.perf_counter()

    try:
        print("\nbuilding prompts (outside every timed window -- see build_prompts):")
        prompt_build_started = time.perf_counter()
        prompts = build_prompts(stack, [e.question for e in heldout])
        prompt_build_s = (time.perf_counter() - prompt_build_started) / len(prompts)

        median_tokens = sorted(p.n_tokens for p in prompts)[len(prompts) // 2]
        supported = stack.concurrency_supported(median_tokens, args.max_new_tokens)
        per_request = stack.blocks_per_request(median_tokens, args.max_new_tokens)
        ship_supported = (640 - stack.scheduler.config.cow_reserve_blocks) // per_request

        print(f"\npool: {args.num_blocks} blocks x {args.block_size} tokens = "
              f"{stack.executor.pool_bytes / GIB:.2f} GiB")
        print(f"  a median request needs {per_request} blocks -> "
              f"**{supported} concurrent** at this pool size")
        print(f"  at the shipped 640-block pool that number is {ship_supported}. The 4 GiB budget,")
        print("  not the scheduler, is what caps concurrency on the ship configuration -- which is")
        print("  why this sweep runs a deliberately over-budget pool to measure the scheduler.")
        if supported < 2:
            print(f"\nFAILED: a pool admitting {supported} request(s) cannot show a concurrency "
                  f"effect. Re-run with --num-blocks {per_request * 4 + 8}.", file=sys.stderr)
            return 1

        print("\ncalibrating single-request service time:")
        service_s = calibrate(stack, prompts, args.max_new_tokens)

        plan: list[tuple[float, int]] = [(lf, args.chunk_size) for lf in args.load_factors]
        if not args.skip_chunk_comparison:
            plan.append((args.chunk_comparison_load, UNCHUNKED))

        # Arrival span at loads below 1x, service time above it -- whichever dominates. A 0.5x
        # cell is bounded by how slowly requests arrive; a 4x cell by how fast the GPU drains them.
        estimate = sum(args.n_requests * service_s / min(lf, 1.0) for lf, _ in plan)
        span = (f"{estimate:.0f}-{estimate * 2:.0f} s" if estimate < 120
                else f"{estimate / 60:.0f}-{estimate * 2 / 60:.0f} min")
        print(f"\n{len(plan)} cell(s), roughly {span} "
              f"(service time {service_s:.1f}s/request)\n")

        for load_factor, chunk_size in plan:
            name = cell_name(load_factor, chunk_size)
            if name in done:
                print(f"  {name}: already measured, skipping")
                continue
            rate = load_factor / service_s
            print(f"  {name}: offering {rate:.3f} req/s ({load_factor:g}x service rate), "
                  f"{args.n_requests} requests")

            result, stack_stats = run_cell(
                stack, prompts, load_factor, service_s, chunk_size,
                args.n_requests, args.max_new_tokens, args.seed,
            )

            record = {
                "cell": name,
                "load_factor": load_factor,
                "chunk_size": chunk_size,
                "chunked_prefill": chunk_size < UNCHUNKED,
                "service_time_s": service_s,
                "arm": args.arm,
                "model": args.model,
                "num_blocks": args.num_blocks,
                "block_size": args.block_size,
                "max_batch_size": args.max_batch_size,
                "max_prefills_per_step": args.max_prefills_per_step,
                "max_new_tokens": args.max_new_tokens,
                "n_prompts": len(prompts),
                "median_prompt_tokens": median_tokens,
                "blocks_per_request": per_request,
                "concurrency_supported": supported,
                "ship_pool_concurrency": ship_supported,
                "pool_bytes": stack.executor.pool_bytes,
                "weight_bytes": stack.weight_bytes,
                "prompt_build_s": round(prompt_build_s, 3),
                "workload_fingerprint": fingerprint,
                "code_version": code_version(),
                "session_id": SESSION_ID,
                "device": info.name,
                "trusted": info.trusted,
                **stack_stats,
                **result.to_dict(),
            }
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
                fh.flush()

            ttft, e2e = record["ttft_s"], record["e2e_s"]
            failed = f" ({record['n_failed']} failed)" if record["n_failed"] else ""
            print(f"    completed {record['n_completed']}/{record['n_requests']}{failed}"
                  f"  |  {record['output_tokens_per_s']:.1f} tok/s aggregate"
                  f"  |  {record['completed_per_s'] * 60:.1f} req/min")
            if ttft:
                flag = "" if ttft["p99_is_reliable"] else "*"
                tail = f"  |  e2e p50 {e2e['p50']:.2f}s" if e2e else ""
                print(f"    TTFT p50 {ttft['p50']:.2f}s  p95 {ttft['p95']:.2f}s  "
                      f"p99 {ttft['p99']:.2f}s{flag}{tail}")
            print(f"    in-flight max {record['max_inflight']} mean "
                  f"{record['mean_inflight']:.2f}  |  admission blocked "
                  f"{record['scheduler_delta']['admission_blocked']}x  |  preempted "
                  f"{record['scheduler_delta']['preempted']}x")
            if record["blocks_leaked"]:
                print(f"    ** {record['blocks_leaked']} BLOCKS LEAKED ** -- every later cell "
                      f"runs with a smaller pool. Treat subsequent rows as suspect.",
                      file=sys.stderr)
            if record["timed_out"]:
                print("    ** CELL TIMED OUT ** -- recorded anyway; a deadlock is a finding.",
                      file=sys.stderr)
            print(f"    [{time.perf_counter() - run_started:.0f}s elapsed] {host_and_gpu_free()}\n")
    finally:
        stack.stop()

    print(f"wrote {out_path}\n")
    return summarise(out_path)


#: Tolerance above the analytically-expected stable rho before a cell is called saturated.
SATURATION_MARGIN = 1.05


def expected_stable_rho(row: dict[str, Any]) -> float:
    """What ``utilisation`` reads on a cell that is keeping up perfectly. It is not 1.0.

    ``completed_per_s`` divides by a window running from the first *submission* to the last
    *completion*, so a server that keeps up exactly still shows a window one service time longer
    than the arrival span: ``served = n / (n/lambda + service)``, hence
    ``rho = 1 + service * lambda / n``.

    That bias is large where it matters most -- 8.6% at n=12 and 2x load -- so a flat ``rho > 1.05``
    threshold condemns a healthy cell at small n and clears a sick one at large n. Correcting it
    analytically rather than widening the threshold is what lets the 1.0x cell be read as stable
    (measured 1.09 against an expected 1.09) while the 2x cell at 1.67 is not.
    """
    return 1.0 + row["service_time_s"] * row["offered_per_s"] / max(1, row["n_requests"])


def is_saturated(row: dict[str, Any]) -> bool:
    return utilisation(row) > expected_stable_rho(row) * SATURATION_MARGIN


def utilisation(row: dict[str, Any]) -> float:
    """Offered rate over achieved rate. Above 1, the queue grows for as long as the run lasts.

    **This is the number that decides whether a cell's latency percentiles mean anything.** An
    open loop past saturation has no backpressure, so the k-th arrival waits roughly
    ``k * (1/served - 1/offered)`` and TTFT grows *linearly in the number of requests offered*.
    A p99 measured there is a property of how long the run was, not of the server -- and it is
    reported with the same confident decimal places as a converged one.

    Measured on this stack: at 2x offered load, n=12 gives p99 TTFT 26.1 s and n=100 gives 221.7 s.
    Same code, same load, same session. Adding samples does not converge an unbounded quantity;
    it walks further up the ramp.
    """
    served = row.get("completed_per_s", 0.0)
    return row["offered_per_s"] / served if served > 0 else float("inf")


def decode_phase(row: dict[str, Any]) -> dict[str, Any] | None:
    """Per-request ``e2e - ttft``, summarised. The generation half, with queueing removed.

    Recomputed from the stored outcomes rather than taken as ``e2e p50 - ttft p50``, which is a
    different quantity: the median of a difference is not the difference of medians unless the two
    are perfectly rank-correlated, and under load they are not.

    This decomposition is the reason chunked prefill is legible at all. TTFT is dominated by how
    long a request waited to be *admitted*; the decode phase is how fast it ran once it was. Those
    two respond to chunking in opposite directions, and a table that reports only their sum shows
    a feature doing nothing.
    """
    phases = [
        o["e2e_s"] - o["ttft_s"]
        for o in row.get("outcomes", [])
        if o.get("e2e_s") is not None and o.get("ttft_s") is not None and o.get("error") is None
    ]
    return Percentiles.of(phases).to_dict() if phases else None


def summarise(out_path: Path) -> int:
    """Print the gate's two tables: the tradeoff curve, then the chunked-prefill comparison."""
    parsed = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not parsed:
        print("FAILED: nothing was measured. Do not use this file.", file=sys.stderr)
        return 1

    by_cell: dict[str, dict[str, Any]] = {row["cell"]: row for row in parsed}
    rows = list(by_cell.values())
    if len(parsed) > len(rows):
        print(f"  ({len(parsed) - len(rows)} superseded row(s); using the latest of each cell)")

    chunked = sorted([r for r in rows if r["chunked_prefill"]], key=lambda r: r["load_factor"])
    print(f"\n{'offered':>9}{'n':>5}{'req/s in':>9}{'out':>7}{'rho':>6}{'tok/s':>7}"
          f"{'in-flight':>11}{'TTFT p50':>10}{'p95':>8}{'p99':>9}{'decode p50':>12}")
    for row in chunked:
        ttft, dec, rho = row["ttft_s"], decode_phase(row), utilisation(row)
        flag = "" if ttft and ttft["p99_is_reliable"] else "*"
        mark = " SAT" if is_saturated(row) else ""
        print(f"{row['load_factor']:>8.1f}x{row['n_requests']:>5}{row['offered_per_s']:>9.3f}"
              f"{row['completed_per_s']:>7.3f}{rho:>6.2f}{row['output_tokens_per_s']:>7.1f}"
              f"{row['mean_inflight']:>11.2f}{ttft['p50'] if ttft else 0:>10.2f}"
              f"{ttft['p95'] if ttft else 0:>8.2f}{ttft['p99'] if ttft else 0:>8.2f}{flag}"
              f"{dec['p50'] if dec else 0:>12.2f}{mark}")

    # Capacity, and the line past which the latency columns above stop describing the server.
    saturated = [r for r in chunked if is_saturated(r)]
    stable = [r for r in chunked if not is_saturated(r)]
    if saturated:
        capacity = sum(r["completed_per_s"] for r in saturated) / len(saturated)
        serial = 1.0 / chunked[0]["service_time_s"]
        print(f"\n  Capacity {capacity:.3f} req/s = {capacity * 60:.1f} req/min, read off the "
              f"drain rate of the {len(saturated)} saturated cell(s).")
        print(f"  Strictly serial service would be {serial * 60:.1f} req/min, so concurrency buys "
              f"{capacity / serial:.2f}x and saturation")
        print(f"  sits near {capacity / serial:.2f}x offered load.")
        print(f"\n  ** The {len(saturated)} row(s) marked SAT are past saturation. Their TTFT "
              "columns are NOT server")
        print("  properties -- an open loop with no backpressure queues without bound, so the k-th")
        print("  arrival waits ~k x (1/served - 1/offered) and the percentiles grow linearly with")
        print("  --n-requests. Quote latency from the unsaturated rows only; more samples do not")
        print("  fix a saturated row, they walk further up the ramp.")
        for row in saturated:
            served, offered = row["completed_per_s"], row["offered_per_s"]
            predicted = row["n_requests"] * (1 / served - 1 / offered)
            actual = row["ttft_s"]["p99"] if row["ttft_s"] else 0.0
            print(f"      {row['load_factor']:g}x n={row['n_requests']:<4} queueing predicts "
                  f"{predicted:>6.0f}s at the tail, measured p99 {actual:>7.2f}s")
    if stable:
        best = stable[-1]
        print(f"\n  Publishable latency comes from the unsaturated rows. At "
              f"{best['load_factor']:g}x offered load:")
        print(f"      TTFT p50 {best['ttft_s']['p50']:.2f}s, p95 {best['ttft_s']['p95']:.2f}s "
              f"(n={best['n_requests']}), against {best['service_time_s']:.2f}s served alone.")

    if len(chunked) >= 2:
        base, top = chunked[0], chunked[-1]
        peak = max(chunked, key=lambda r: r["output_tokens_per_s"])
        gain = top["output_tokens_per_s"] / max(1e-9, base["output_tokens_per_s"])
        # Mean, not max. `max_inflight` saturates at the first cell that ever reached depth 2 and
        # then reports "1x" for every increase after it -- which is how a 3.8x rise in actual
        # concurrency gets printed as no change at all.
        conc = top["mean_inflight"] / max(1e-9, base["mean_inflight"])
        print(f"\n  Offered load {base['load_factor']:g}x -> {top['load_factor']:g}x: throughput "
              f"{gain:.2f}x for {conc:.2f}x the mean in-flight depth.")
        if peak is not top:
            # Not a footnote: throughput peaking below maximum load means the server is past
            # saturation and the extra arrivals are buying queue, not work. Reporting the best
            # cell as if it were the endpoint would hide exactly that.
            print(f"  Throughput peaks at {peak['load_factor']:g}x "
                  f"({peak['output_tokens_per_s']:.1f} tok/s) and falls after it -- past that "
                  "point offered load buys queueing, not work.")
        print("  Prediction 1 said the throughput multiple would land well under the concurrency")
        print("  multiple, because decode is KV-bandwidth bound on an MHA checkpoint (D14).")

        # Which resource actually formed the queue. Getting this wrong is how a sweep concludes
        # "the pool is the limit" from a run where no request was ever short of a block.
        for row in chunked:
            blocked = row["scheduler_delta"]["admission_blocked"]
            prefills = row.get("max_prefills_per_step", 1)
            if blocked == 0 and row["max_inflight"] <= max(2, prefills + 1):
                print(f"  {row['load_factor']:g}x: admission never blocked and in-flight held at "
                      f"{row['max_inflight']} -- the binding constraint was "
                      f"max_prefills_per_step={prefills}, not the block pool.")
                break

    # The chunked-prefill comparison: same offered load, same arrival seed, one variable.
    pairs = [
        (c, u) for c in chunked
        for u in rows
        if not u["chunked_prefill"] and u["load_factor"] == c["load_factor"]
    ]
    if pairs:
        print(f"\n{'chunked prefill':>20}{'tok/s':>9}{'in-flight':>11}{'blocked':>9}"
              f"{'TTFT p95':>10}{'decode p50':>12}{'e2e p50':>9}")
        for on, off in pairs:
            for label, row in ((f"on ({on['chunk_size']})", on), ("off (one pass)", off)):
                ttft, e2e, dec = row["ttft_s"], row["e2e_s"], decode_phase(row)
                print(f"{label:>20}{row['output_tokens_per_s']:>9.1f}"
                      f"{row['mean_inflight']:>11.2f}"
                      f"{row['scheduler_delta']['admission_blocked']:>9}"
                      f"{ttft['p95'] if ttft else 0:>10.2f}{dec['p50'] if dec else 0:>12.2f}"
                      f"{e2e['p50'] if e2e else 0:>9.2f}")

            on_dec, off_dec = decode_phase(on), decode_phase(off)
            if not (on["ttft_s"] and off["ttft_s"] and on_dec and off_dec):
                continue
            # Reported as two separate ratios in whichever direction they fall, because the
            # feature moves TTFT and decode *opposite* ways and a single "chunking wins/loses"
            # number is the average of a saving and a cost.
            ttft_ratio = on["ttft_s"]["p95"] / off["ttft_s"]["p95"]
            dec_ratio = off_dec["p50"] / on_dec["p50"]
            e2e_ratio = on["e2e_s"]["p50"] / off["e2e_s"]["p50"]
            print(f"\n  At {on['load_factor']:g}x load, chunking costs {ttft_ratio:.2f}x on p95 "
                  f"TTFT and buys {dec_ratio:.2f}x on the decode phase,")
            print(f"  netting {e2e_ratio:.2f}x end to end. The two halves move in opposite "
                  "directions and only the")
            print("  decomposition shows it -- TTFT is mostly *admission* queueing, decode is what")
            print("  BUGS.md P-18 is actually about. Mean in-flight and the admission-blocked")
            print("  count say which resource each arm ran out of.")

    thin = [r for r in rows if r["ttft_s"] and not r["ttft_s"]["p99_is_reliable"]]
    if thin:
        print(f"\n  `*` p99 from under 100 samples ({len(thin)}/{len(rows)} cells) -- it is the")
        print("  max, not a percentile. Raising --n-requests fixes that **only on an unsaturated")
        print("  row**. On a saturated one the sample size was never the problem: the quantity")
        print("  itself diverges with run length, so a bigger n buys a bigger number, not a")
        print("  better estimate. Lower the offered load instead.")

    sessions = {row.get("session_id", "unstamped") for row in rows}
    versions = {row.get("code_version", "unknown") for row in rows}
    if len(sessions) == 1:
        print(f"\n  SINGLE SESSION ({sessions.pop()}): every row here is internally comparable.")
    else:
        names = ", ".join(sorted(sessions))
        print(f"\n  ** {len(sessions)} SESSIONS ({names}) ** -- cross-session latency carries")
        print("  ~7.5% variance (D24), so only rows sharing a session are comparable.")
    if len(versions) > 1:
        print(f"  ** {len(versions)} CODE VERSIONS ({', '.join(sorted(versions))}) **")

    leaked = [r for r in rows if r.get("blocks_leaked")]
    if leaked:
        print(f"\n  ** {len(leaked)} cell(s) leaked blocks ** -- pool conservation failed under "
              "real load", file=sys.stderr)
    else:
        print("  Pool conservation held on every cell: blocks free before == free after.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
