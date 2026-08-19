"""Phase 5e's load driver, tested without a GPU.

``bench/load.py`` is the thing that will produce the throughput and p99 TTFT numbers, and a
measurement driver is exactly the kind of code that fails silently: a TTFT measured from the wrong
instant, a cell that reports 12 completions because it counted submissions, a concurrency figure
that is really the pool's block count. None of that raises. All of it survives a T4 session and
lands in the README.

So the driver is driven here against a fake executor that allocates *real* blocks and sleeps
instead of computing. That covers everything except "the forward pass produces sensible tokens",
which ``tests/test_anls.py`` and the serving test already cover end to end.

``BUGS.md`` B-05 is the standing reason: the last script written for a T4 and executed nowhere
else produced a complete, well-formed file of zeros.
"""

from __future__ import annotations

import statistics
import threading
import time

import pytest

from bench.load import drain, poisson_arrivals, replay_poisson
from edgerag.cache.allocator import BlockAllocator, OutOfBlocksError
from edgerag.cache.block_table import BlockTable
from edgerag.sched.request import Request
from edgerag.sched.scheduler import Batch, Scheduler, SchedulerConfig
from edgerag.serve.engine import InferenceEngine, StepOutput

EOS = 2


class _FakeCache:
    """Just enough cache for the scheduler: real blocks, no tensors.

    Real ``BlockTable`` rather than a stub, because pool conservation is one of the properties
    under test and a stub that counts nothing would pass it vacuously.
    """

    def __init__(self, allocator: BlockAllocator) -> None:
        self.table = BlockTable(allocator=allocator)

    @property
    def seq_len(self) -> int:
        return self.table.num_tokens

    def free(self) -> int:
        return self.table.free()


class PacedExecutor:
    """Chunked prefill and one-token decode, with a configurable per-iteration cost.

    Mirrors ``ModelExecutor``'s *structure* -- chunk the prefill, emit the first token from the
    final chunk, one token per decoding request -- because the driver's numbers are made of that
    structure. The sleep stands in for a forward pass so that arrivals genuinely overlap; with an
    instant executor every request is served before the next arrives and concurrency is never
    exercised.
    """

    def __init__(
        self,
        allocator: BlockAllocator,
        chunk_size: int = 512,
        step_s: float = 0.004,
        fail_on: set[str] | None = None,
    ) -> None:
        self.allocator = allocator
        self.chunk_size = chunk_size
        self.step_s = step_s
        self.fail_on = fail_on or set()
        self.iterations = 0
        self.threads: set[str] = set()

    def _cache(self, request: Request) -> _FakeCache:
        if request.cache is None:
            request.cache = _FakeCache(self.allocator)
        return request.cache

    def execute(self, batch: Batch) -> StepOutput:
        self.threads.add(threading.current_thread().name)
        self.iterations += 1
        time.sleep(self.step_s)
        output = StepOutput()

        for request in batch.prefill:
            if request.request_id in self.fail_on:
                raise RuntimeError("synthetic executor failure")
            cache = self._cache(request)
            start = request.prefill_offset
            end = min(start + self.chunk_size, request.prompt_len)
            n_tokens = end - start
            if n_tokens <= 0:
                continue
            cache.table.append(n_tokens)
            output.prefilled[request.request_id] = n_tokens
            if end >= request.prompt_len:
                output.tokens[request.request_id] = 100 + request.num_generated

        for request in batch.decode:
            cache = self._cache(request)
            cache.table.append(1)
            output.tokens[request.request_id] = 100 + request.num_generated
        return output


def _stack(num_blocks: int = 64, block_size: int = 16, **executor_kwargs):
    allocator = BlockAllocator(num_blocks, block_size)
    scheduler = Scheduler(allocator, SchedulerConfig(eos_token_id=EOS))
    executor = PacedExecutor(allocator, **executor_kwargs)
    engine = InferenceEngine(scheduler, executor, idle_poll_seconds=0.001)
    engine.start()
    return allocator, scheduler, executor, engine


def _factory(prompt_len: int = 64, max_new_tokens: int = 4, prefix: str = "r"):
    def make(i: int) -> Request:
        return Request(
            request_id=f"{prefix}{i:03d}",
            prompt_token_ids=list(range(prompt_len)),
            max_new_tokens=max_new_tokens,
        )

    return make


# --- the arrival process ------------------------------------------------------------------------


def test_arrivals_are_deterministic_for_a_seed() -> None:
    """Two cells at one rate must receive the same arrival pattern, or they differ in two things."""
    assert poisson_arrivals(20, 5.0, seed=7) == poisson_arrivals(20, 5.0, seed=7)
    assert poisson_arrivals(20, 5.0, seed=7) != poisson_arrivals(20, 5.0, seed=8)


def test_arrivals_are_monotonic_and_hit_the_requested_rate() -> None:
    offsets = poisson_arrivals(4000, 10.0, seed=1)
    assert offsets == sorted(offsets)
    gaps = [b - a for a, b in zip([0.0, *offsets], offsets, strict=False)]
    # Exponential with rate 10 has mean gap 0.1. 4000 samples puts the standard error near 0.0016.
    assert statistics.fmean(gaps) == pytest.approx(0.1, rel=0.06)


def test_a_non_positive_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        poisson_arrivals(5, 0.0, seed=1)


# --- what the driver reports ----------------------------------------------------------------------


def test_every_request_gets_a_ttft_an_end_to_end_and_a_token_count() -> None:
    _, scheduler, _, engine = _stack()
    try:
        result = replay_poisson(
            engine, scheduler, _factory(), n_requests=6, arrival_rate_hz=50.0, timeout_s=30.0
        )
    finally:
        engine.stop()

    assert result.n_completed == 6
    assert result.n_failed == 0
    assert result.n_never_finished == 0
    assert not result.timed_out
    for outcome in result.outcomes:
        assert outcome.ttft_s is not None and outcome.ttft_s > 0
        assert outcome.e2e_s is not None and outcome.e2e_s >= outcome.ttft_s
        assert outcome.n_tokens == 4
    assert result.total_tokens == 24


def test_ttft_is_measured_from_submission_so_queueing_is_counted() -> None:
    """The load-bearing measurement choice.

    Six requests arrive at once into a pool with room for two. The last admitted waits for an
    earlier one to finish, and that wait is part of its TTFT -- measuring from *admission* would
    delete exactly the queueing this experiment exists to observe, and would do it silently.
    """
    prompt_len, block_size = 256, 16
    per_request = (prompt_len + 4 + block_size - 1) // block_size  # 17 blocks
    _, scheduler, _, engine = _stack(
        num_blocks=per_request * 2 + 4, block_size=block_size, step_s=0.01
    )
    try:
        result = replay_poisson(
            engine, scheduler, _factory(prompt_len=prompt_len),
            n_requests=6, arrival_rate_hz=500.0, timeout_s=60.0,
        )
    finally:
        engine.stop()

    ttfts = sorted(o.ttft_s for o in result.outcomes)
    assert result.n_completed == 6
    assert result.scheduler_delta["admission_blocked"] > 0, "the pool should have blocked someone"
    assert ttfts[-1] > ttfts[0] * 1.5, (
        f"queued requests show no extra TTFT ({ttfts[0]:.3f}s..{ttfts[-1]:.3f}s) -- TTFT is "
        "probably being measured from admission rather than from arrival"
    )


def test_the_pool_is_what_caps_concurrency() -> None:
    """The claim ``scripts/colab_poisson.py`` refuses to run a sweep without.

    A pool sized for one request admits one at a time no matter how many arrive; the same
    arrivals against a pool sized for three overlap. If this ever inverts, a concurrency sweep is
    measuring the block count and calling it the scheduler.
    """
    prompt_len, block_size = 256, 16
    per_request = (prompt_len + 4 + block_size - 1) // block_size

    def measure(num_blocks: int) -> int:
        _, scheduler, _, engine = _stack(
            num_blocks=num_blocks, block_size=block_size, step_s=0.01
        )
        try:
            result = replay_poisson(
                engine, scheduler, _factory(prompt_len=prompt_len),
                n_requests=6, arrival_rate_hz=500.0, timeout_s=60.0, sample_hz=200.0,
            )
        finally:
            engine.stop()
        assert result.n_completed == 6
        return result.max_inflight

    assert measure(per_request + 4) == 1
    assert measure(per_request * 3 + 4) > 1


def test_a_failing_request_is_recorded_rather_than_raised() -> None:
    """A load sweep that dies on one bad request throws away the whole cell.

    Also pins the **blast radius**, which is bigger than one request and is worth knowing before
    reading a sweep's failure counts. ``InferenceEngine._step`` fails *every request in the batch*
    when the executor raises -- deliberately, so the worker thread survives and no stream hangs
    forever -- so under load a single bad request can take out up to ``max_batch_size`` of its
    neighbours. That is why ``n_failed`` in a cell is not a count of bad requests.
    """
    _, scheduler, _, engine = _stack(fail_on={"r002"})
    try:
        result = replay_poisson(
            engine, scheduler, _factory(), n_requests=4, arrival_rate_hz=50.0, timeout_s=30.0
        )
    finally:
        engine.stop()

    assert result.n_failed >= 1
    assert result.n_completed + result.n_failed == 4, "every request must reach a terminal state"
    assert result.n_never_finished == 0
    failed = next(o for o in result.outcomes if o.request_id == "r002")
    assert failed.error is not None and "synthetic" in failed.error
    for outcome in result.outcomes:
        if outcome.error is not None:
            assert outcome.finish_t is not None, "a failed stream must still be closed"


def test_blocks_are_conserved_across_a_run() -> None:
    """Pool conservation under real submission traffic, not against a hand-driven scheduler.

    A leak here does not fail a cell -- it silently shrinks every *later* cell's concurrency, so
    the sweep would report a declining curve that is an artifact of its own earlier rows.
    """
    allocator, scheduler, _, engine = _stack(num_blocks=64)
    free_before = allocator.num_free
    try:
        replay_poisson(
            engine, scheduler, _factory(), n_requests=8, arrival_rate_hz=80.0, timeout_s=30.0
        )
        drain(scheduler, engine)
    finally:
        engine.stop()

    assert allocator.num_free == free_before
    allocator.check_invariants()


def test_a_request_too_large_for_the_pool_times_out_with_partial_results() -> None:
    """A deadlocked cell must return evidence, not hang until the runtime is reclaimed."""
    _, scheduler, _, engine = _stack(num_blocks=8, block_size=16)
    try:
        result = replay_poisson(
            engine, scheduler, _factory(prompt_len=4096),
            n_requests=2, arrival_rate_hz=100.0, timeout_s=1.0,
        )
    finally:
        engine.stop()

    assert result.timed_out
    assert result.n_never_finished == 2
    assert result.scheduler_delta["admission_blocked"] > 0


# --- the knob the sweep actually turns -----------------------------------------------------------


def test_chunk_size_changes_the_number_of_prefill_iterations() -> None:
    """The trap the runner is shaped around, asserted.

    ``SchedulerConfig.prefill_chunk_size`` is read by nothing; the executor's ``chunk_size`` is
    what slices a prefill. A sweep that turned the scheduler's copy would produce two identical
    arms and the confident conclusion that chunked prefill does not help.
    """

    def prefill_chunks(chunk_size: int) -> int:
        _, scheduler, _, engine = _stack(num_blocks=128, chunk_size=chunk_size)
        try:
            result = replay_poisson(
                engine, scheduler, _factory(prompt_len=512, max_new_tokens=2),
                n_requests=1, arrival_rate_hz=100.0, timeout_s=30.0,
            )
        finally:
            engine.stop()
        assert result.n_completed == 1
        return result.scheduler_delta["prefill_chunks"]

    assert prefill_chunks(64) == 8, "512 tokens in 64-token chunks is 8 iterations"
    assert prefill_chunks(4096) == 1, "one pass when the chunk exceeds the prompt"


def test_scheduler_config_prefill_chunk_size_is_inert() -> None:
    """Pins the trap in place: if this ever starts working, the runner's comment is wrong.

    Not an endorsement of the dead field -- it is a regression guard on a documented hazard. The
    day someone wires ``SchedulerConfig.prefill_chunk_size`` up, this fails and points at the
    docstring that has to change with it.
    """
    allocator = BlockAllocator(128, 16)
    scheduler = Scheduler(allocator, SchedulerConfig(eos_token_id=EOS, prefill_chunk_size=64))
    executor = PacedExecutor(allocator, chunk_size=512, step_s=0.0)
    engine = InferenceEngine(scheduler, executor, idle_poll_seconds=0.001)
    engine.start()
    try:
        result = replay_poisson(
            engine, scheduler, _factory(prompt_len=512, max_new_tokens=2),
            n_requests=1, arrival_rate_hz=100.0, timeout_s=30.0,
        )
    finally:
        engine.stop()

    assert result.scheduler_delta["prefill_chunks"] == 1, (
        "the scheduler's prefill_chunk_size now affects chunking -- update the hazard note in "
        "scripts/colab_poisson.py, which tells the reader it does not"
    )


def test_drain_clears_finished_requests() -> None:
    """Each finished request pins a prompt-embedding tensor; a sweep that never clears them OOMs."""
    _, scheduler, _, engine = _stack()
    try:
        replay_poisson(
            engine, scheduler, _factory(), n_requests=4, arrival_rate_hz=100.0, timeout_s=30.0
        )
        assert scheduler.finished, "requests should have accumulated"
        assert drain(scheduler, engine)
    finally:
        engine.stop()
    assert scheduler.finished == []


def test_the_driver_never_runs_the_executor_on_its_own_thread() -> None:
    """P-17 again, from the load driver's side: submission must not become execution."""
    _, scheduler, executor, engine = _stack()
    driver_thread = threading.current_thread().name
    try:
        replay_poisson(
            engine, scheduler, _factory(), n_requests=3, arrival_rate_hz=100.0, timeout_s=30.0
        )
    finally:
        engine.stop()

    assert executor.threads == {"edgerag-engine"}
    assert driver_thread not in executor.threads


def test_zero_requests_is_refused() -> None:
    _, scheduler, _, engine = _stack()
    try:
        with pytest.raises(ValueError, match="n_requests"):
            replay_poisson(
                engine, scheduler, _factory(), n_requests=0, arrival_rate_hz=10.0
            )
    finally:
        engine.stop()


def test_out_of_blocks_is_an_allocator_error_not_a_driver_one() -> None:
    """Guards the assumption behind the timeout test: exhaustion raises where it is handled."""
    allocator = BlockAllocator(2, 16)
    table = BlockTable(allocator=allocator)
    with pytest.raises(OutOfBlocksError):
        table.append(1000)
