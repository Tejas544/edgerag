"""Phase 7: the asyncio-to-GPU bridge, tested without a GPU.

``BUGS.md`` P-17 is the reason this component exists at all, and it is a *concurrency* claim --
"the scheduler owns the GPU on a dedicated thread; asyncio touches it only through queues". A
concurrency claim asserted in a docstring is a hope. These tests drive the real threading with a
fake executor, so ordering, shutdown, error propagation and the thread boundary itself are checked
in milliseconds rather than discovered under load.

The one thing a fake executor cannot check -- that a real forward pass produces sensible tokens --
is already covered end to end by ``tests/test_anls.py``.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from edgerag.cache.allocator import BlockAllocator
from edgerag.sched.request import Request, RequestState
from edgerag.sched.scheduler import Batch, Scheduler, SchedulerConfig
from edgerag.serve.engine import InferenceEngine, StepOutput

EOS = 2


class ScriptedExecutor:
    """Advances prefill in one chunk, then emits a fixed token sequence per request.

    Records the thread it ran on, because "GPU work never happens on the event loop" is exactly
    the property that is invisible unless something checks which thread executed.
    """

    def __init__(self, script: list[int] | None = None) -> None:
        self.script = script if script is not None else [10, 11, 12]
        self.threads: set[str] = set()
        self.calls = 0

    def execute(self, batch: Batch) -> StepOutput:
        self.threads.add(threading.current_thread().name)
        self.calls += 1
        output = StepOutput()
        for request in batch.prefill:
            output.prefilled[request.request_id] = request.prompt_len - request.prefill_offset
        for request in batch.decode:
            index = request.num_generated
            if index < len(self.script):
                output.tokens[request.request_id] = self.script[index]
        return output


class ExplodingExecutor:
    def execute(self, batch: Batch) -> StepOutput:
        raise RuntimeError("CUDA out of memory")


def _engine(executor, max_new_tokens_eos: int | None = EOS, num_blocks: int = 64):
    scheduler = Scheduler(
        BlockAllocator(num_blocks, 16),
        SchedulerConfig(eos_token_id=max_new_tokens_eos),
    )
    return InferenceEngine(scheduler, executor, idle_poll_seconds=0.001)


def _request(request_id: str = "r1", prompt_len: int = 8, max_new_tokens: int = 3) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(100, 100 + prompt_len)),
        max_new_tokens=max_new_tokens,
    )


async def _collect(engine, request, timeout: float = 5.0):
    events = []
    async def run():
        async for event in engine.stream(request):
            events.append(event)
    await asyncio.wait_for(run(), timeout=timeout)
    return events


# --- the thread boundary, which is the whole point ---------------------------------------------


def test_the_executor_never_runs_on_the_event_loop() -> None:
    """P-17 itself. If this fails, every other request's TTFT inflates by one forward pass."""
    executor = ScriptedExecutor()
    engine = _engine(executor)
    engine.start()
    try:
        loop_thread = threading.current_thread().name
        asyncio.run(_collect(engine, _request()))
        assert executor.threads, "the executor never ran"
        assert loop_thread not in executor.threads, (
            f"executor ran on {executor.threads}, which includes the calling thread -- "
            "GPU work is on the event loop"
        )
        assert executor.threads == {"edgerag-engine"}
    finally:
        engine.stop()


def test_submit_does_not_block_the_caller() -> None:
    """The event loop must hand off and return, not wait for the GPU."""

    class SlowExecutor(ScriptedExecutor):
        def execute(self, batch: Batch) -> StepOutput:
            time.sleep(0.05)
            return super().execute(batch)

    engine = _engine(SlowExecutor())
    engine.start()
    try:
        started = time.perf_counter()
        engine.submit(_request(), lambda event: None)
        assert time.perf_counter() - started < 0.02, "submit() waited for the worker"
    finally:
        engine.stop()


# --- streaming --------------------------------------------------------------------------------


def test_tokens_arrive_in_order_and_the_stream_terminates() -> None:
    engine = _engine(ScriptedExecutor([10, 11, 12]))
    engine.start()
    try:
        events = asyncio.run(_collect(engine, _request(max_new_tokens=3)))
    finally:
        engine.stop()

    tokens = [e.token_id for e in events if e.token_id is not None]
    assert tokens == [10, 11, 12]
    assert events[-1].done and events[-1].error is None


def test_hitting_max_new_tokens_reports_length() -> None:
    engine = _engine(ScriptedExecutor([10, 11, 12, 13, 14]))
    engine.start()
    try:
        events = asyncio.run(_collect(engine, _request(max_new_tokens=2)))
    finally:
        engine.stop()

    assert [e.token_id for e in events if e.token_id is not None] == [10, 11]
    assert events[-1].finish_reason == "length"


def test_hitting_eos_reports_stop_and_truncates() -> None:
    """The scheduler owns the stop decision; the engine must report the reason it actually was."""
    engine = _engine(ScriptedExecutor([10, EOS, 12]))
    engine.start()
    try:
        events = asyncio.run(_collect(engine, _request(max_new_tokens=5)))
    finally:
        engine.stop()

    assert [e.token_id for e in events if e.token_id is not None] == [10, EOS]
    assert events[-1].finish_reason == "stop"


def test_two_requests_stream_concurrently_without_interleaving_their_tokens() -> None:
    """Continuous batching: both run in one batch, and neither receives the other's tokens."""
    engine = _engine(ScriptedExecutor([10, 11, 12]))
    engine.start()

    async def both():
        first, second = _request("a", max_new_tokens=3), _request("b", max_new_tokens=3)
        return await asyncio.gather(_collect(engine, first), _collect(engine, second))

    try:
        events_a, events_b = asyncio.run(both())
    finally:
        engine.stop()

    for request_id, events in (("a", events_a), ("b", events_b)):
        assert {e.request_id for e in events} == {request_id}, "streams crossed"
        assert [e.token_id for e in events if e.token_id is not None] == [10, 11, 12]


# --- failure, which must reach the client rather than hang it ----------------------------------


def test_an_executor_crash_ends_the_stream_with_an_error() -> None:
    """A silent hang is the worst failure here: the client waits forever on a dead generation."""
    engine = _engine(ExplodingExecutor())
    engine.start()
    try:
        events = asyncio.run(_collect(engine, _request(), timeout=5.0))
    finally:
        engine.stop()

    assert events[-1].done
    assert "CUDA out of memory" in events[-1].error


def test_the_worker_survives_a_crash_and_serves_the_next_request() -> None:
    """One request's OOM must not take the server down with it."""

    class FlakyExecutor(ScriptedExecutor):
        def execute(self, batch: Batch) -> StepOutput:
            if any(r.request_id == "boom" for r in batch.requests):
                raise RuntimeError("boom")
            return super().execute(batch)

    engine = _engine(FlakyExecutor())
    engine.start()
    try:
        failed = asyncio.run(_collect(engine, _request("boom")))
        assert failed[-1].error is not None
        assert engine.is_running, "the worker thread died with the request"

        ok = asyncio.run(_collect(engine, _request("fine", max_new_tokens=2)))
        assert ok[-1].error is None
        assert [e.token_id for e in ok if e.token_id is not None] == [10, 11]
    finally:
        engine.stop()


def test_stopping_the_engine_releases_a_waiting_stream() -> None:
    """A stream whose engine went away must be told, not left awaiting a token forever."""
    engine = _engine(ScriptedExecutor([]))  # never produces a token
    engine.start()

    async def scenario():
        request = _request(max_new_tokens=100)
        events = []

        async def consume():
            async for event in engine.stream(request):
                events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        await asyncio.to_thread(engine.stop)
        await asyncio.wait_for(task, timeout=5.0)
        return events

    events = asyncio.run(scenario())
    assert events[-1].done and "stopped" in events[-1].error


# --- lifecycle ---------------------------------------------------------------------------------


def test_submitting_before_start_is_refused_rather_than_silently_queued() -> None:
    """Queuing into a thread that does not exist yet is a hang with extra steps."""
    engine = _engine(ScriptedExecutor())
    with pytest.raises(RuntimeError, match="not running"):
        engine.submit(_request(), lambda event: None)


def test_starting_twice_is_refused() -> None:
    engine = _engine(ScriptedExecutor())
    engine.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            engine.start()
    finally:
        engine.stop()


def test_stopping_twice_is_harmless() -> None:
    """Shutdown paths get called twice; the second one should be free, not an exception."""
    engine = _engine(ScriptedExecutor())
    engine.start()
    engine.stop()
    engine.stop()
    assert not engine.is_running


def test_stats_count_what_happened() -> None:
    engine = _engine(ScriptedExecutor([10, 11]))
    engine.start()
    try:
        asyncio.run(_collect(engine, _request(max_new_tokens=2)))
    finally:
        engine.stop()
    assert engine.stats.submitted == 1
    assert engine.stats.completed == 1
    assert engine.stats.failed == 0
    assert engine.stats.steps > 0


def test_a_finished_request_reaches_a_terminal_state_in_the_scheduler() -> None:
    """The engine and the scheduler must agree the request is over, or blocks leak."""
    engine = _engine(ScriptedExecutor([10, 11]))
    engine.start()
    try:
        asyncio.run(_collect(engine, _request(max_new_tokens=2)))
    finally:
        engine.stop()

    assert len(engine.scheduler.finished) == 1
    assert engine.scheduler.finished[0].state is RequestState.FINISHED
    assert not engine.scheduler.running
