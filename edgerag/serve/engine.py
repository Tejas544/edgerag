"""The bridge between asyncio and the GPU. Phase 7.

``BUGS.md`` P-17 in one sentence: **the scheduler owns the GPU on a dedicated thread, and asyncio
touches it only through queues.** Running a forward pass inside an ``async def`` handler blocks the
event loop for the duration, so every *other* in-flight request's time-to-first-token inflates by
however long that pass took. It looks like a model problem and it is a threading problem, which is
why it is worth building the seam explicitly rather than discovering it under load.

The shape:

* **One worker thread** runs the scheduler loop -- drain submissions, ``schedule()``, execute,
  emit tokens, repeat. Every tensor operation happens on this thread and nowhere else.
* **Submissions cross in** through a ``queue.Queue``, which is thread-safe by construction.
* **Tokens cross back out** through ``loop.call_soon_threadsafe``, because an ``asyncio.Queue`` is
  *not* thread-safe and calling ``put_nowait`` on it from the worker thread is a race that will
  survive every test you write and fail in production.

The model itself is injected as a :class:`BatchExecutor` rather than constructed here. That is not
decoration: it means the entire concurrency story -- ordering, backpressure, cancellation,
shutdown, error propagation -- is testable on CPU in milliseconds with a fake executor, and the
one thing a fake cannot check (that the real forward pass produces sensible tokens) is already
covered by ``tests/test_anls.py`` end to end.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from edgerag.sched.request import Request
from edgerag.sched.scheduler import Batch, Scheduler

#: How long the worker sleeps when the scheduler has nothing to do. Short enough that a newly
#: arrived request is picked up promptly, long enough that an idle server does not spin a core.
IDLE_POLL_SECONDS = 0.002


@dataclass
class TokenEvent:
    """One thing that happened to one request, delivered to whoever is streaming it."""

    request_id: str
    token_id: int | None = None
    #: Set when the request is over. ``error`` distinguishes "finished" from "failed"; a stream
    #: that ends silently on an exception is indistinguishable from one that ended normally, which
    #: is how a serving bug becomes a truncated answer nobody notices.
    done: bool = False
    error: str | None = None
    #: Why generation stopped, once ``done``. OpenAI calls this ``finish_reason``.
    finish_reason: str | None = None


@dataclass
class StepOutput:
    """What one executed batch produced, split the way the scheduler's transitions are split.

    Two dictionaries rather than one, because ``prefilled`` and ``tokens`` drive two *different*
    state transitions (``on_prefill_chunk`` and ``on_token``) and merging them would force the
    engine to guess which one a number meant. A chunked prefill reports progress and no token; a
    decode step reports a token and no progress.
    """

    #: request_id -> newly generated token.
    tokens: dict[str, int] = field(default_factory=dict)
    #: request_id -> prompt tokens consumed this step (chunked prefill, ``CONTEXT.md`` D18).
    prefilled: dict[str, int] = field(default_factory=dict)


class BatchExecutor(Protocol):
    """Runs one scheduled batch on the GPU. **The only place tensors are touched.**

    Implemented for real by the model path; implemented trivially by tests. The scheduler decides
    *what* to run (``edgerag/sched/scheduler.py`` is pure decision-making, no tensors); this is
    the seam where that decision becomes arithmetic.
    """

    def execute(self, batch: Batch) -> StepOutput: ...


@dataclass
class _Submission:
    request: Request
    emit: Callable[[TokenEvent], None]


@dataclass
class EngineStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "steps": self.steps,
        }


class InferenceEngine:
    """Owns the GPU thread. Everything public here is safe to call from the event loop."""

    def __init__(
        self,
        scheduler: Scheduler,
        executor: BatchExecutor,
        idle_poll_seconds: float = IDLE_POLL_SECONDS,
    ) -> None:
        self.scheduler = scheduler
        self.executor = executor
        self.idle_poll_seconds = idle_poll_seconds
        self.stats = EngineStats()

        self._inbox: queue.Queue[_Submission] = queue.Queue()
        self._emitters: dict[str, Callable[[TokenEvent], None]] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("engine already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="edgerag-engine", daemon=True)
        self._thread.start()
        self._started.wait(timeout=5.0)

    def stop(self, timeout: float = 5.0) -> None:
        """Idempotent, because shutdown paths get called twice and a second stop should be free."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- the worker thread -- every tensor operation happens below this line ------------------

    def _run(self) -> None:
        self._started.set()
        while not self._stop.is_set():
            self._drain_inbox()
            if not self.scheduler.has_work:
                time.sleep(self.idle_poll_seconds)
                continue
            self._step()
        self._fail_everything_still_open("engine stopped")

    def _drain_inbox(self) -> None:
        while True:
            try:
                submission = self._inbox.get_nowait()
            except queue.Empty:
                return
            self._emitters[submission.request.request_id] = submission.emit
            self.scheduler.add_request(submission.request)

    def _step(self) -> None:
        batch = self.scheduler.schedule()
        if batch.is_empty:
            # Nothing admissible this iteration -- the pool is full of longer-lived requests.
            # Sleeping rather than spinning matters: this is the state a server sits in when it is
            # saturated, which is exactly when a busy-wait would steal the GPU thread's own cycles.
            time.sleep(self.idle_poll_seconds)
            return

        try:
            output = self.executor.execute(batch)
        except Exception as exc:
            # One request's failure must not take down the worker thread. If it did, every other
            # in-flight stream would hang forever waiting for a token from a thread that no longer
            # exists -- the failure mode is a server that stops responding rather than one that
            # returns an error.
            for request in batch.requests:
                self._fail(request, f"{type(exc).__name__}: {exc}")
            self.scheduler.end_step()
            self.stats.steps += 1
            return

        for request in batch.prefill:
            consumed = output.prefilled.get(request.request_id, 0)
            if consumed:
                self.scheduler.on_prefill_chunk(request, consumed)

        for request in batch.decode:
            token_id = output.tokens.get(request.request_id)
            if token_id is None:
                continue
            # `on_token` appends, and finishes the request itself when it hits the stop condition
            # -- the scheduler owns that decision (its own `config.eos_token_id`) so that there is
            # exactly one place stopping is decided. Calling `scheduler.finish` again here would
            # double-count it into `finished` and the stats.
            self.scheduler.on_token(request, token_id)
            self._emit(request.request_id, TokenEvent(request.request_id, token_id=token_id))
            if request.is_terminal:
                self._complete(request)

        self.scheduler.end_step()
        self.stats.steps += 1

    def _finish_reason(self, request: Request) -> str:
        eos = self.scheduler.config.eos_token_id
        hit_eos = bool(
            eos is not None and request.generated_token_ids
            and request.generated_token_ids[-1] == eos
        )
        return "stop" if hit_eos else "length"

    def _complete(self, request: Request) -> None:
        """The request stopped on its own terms; the scheduler has already finished it."""
        self.stats.completed += 1
        self._emit(
            request.request_id,
            TokenEvent(
                request.request_id, done=True, finish_reason=self._finish_reason(request)
            ),
        )
        self._emitters.pop(request.request_id, None)

    def _fail(self, request: Request, error: str) -> None:
        """The request died in the executor, so nothing has finished it yet -- this must."""
        if not request.is_terminal:
            self.scheduler.finish(request)
        self.stats.failed += 1
        self._emit(request.request_id, TokenEvent(request.request_id, done=True, error=error))
        self._emitters.pop(request.request_id, None)

    def _fail_everything_still_open(self, reason: str) -> None:
        """A stream whose engine went away must be told, not left hanging on an await."""
        for request_id, emit in list(self._emitters.items()):
            emit(TokenEvent(request_id, done=True, error=reason))
        self._emitters.clear()

    def _emit(self, request_id: str, event: TokenEvent) -> None:
        emit = self._emitters.get(request_id)
        if emit is not None:
            emit(event)

    # --- the asyncio side -- safe to call from the event loop ---------------------------------

    def submit(self, request: Request, emit: Callable[[TokenEvent], None]) -> None:
        """Hand a request to the worker thread. Returns immediately; does no GPU work."""
        if not self.is_running:
            raise RuntimeError("engine is not running -- call start() first")
        self.stats.submitted += 1
        self._inbox.put(_Submission(request=request, emit=emit))

    async def stream(self, request: Request) -> AsyncIterator[TokenEvent]:
        """Yield this request's tokens as the worker thread produces them.

        ``call_soon_threadsafe`` is the load-bearing line. ``asyncio.Queue`` is not thread-safe;
        calling ``put_nowait`` on it directly from the worker would be a race that passes every
        test on a quiet machine and corrupts the queue under load.
        """
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[TokenEvent] = asyncio.Queue()

        def emit(event: TokenEvent) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        self.submit(request, emit)
        while True:
            event = await events.get()
            yield event
            if event.done:
                return
