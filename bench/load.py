"""Open-loop Poisson replay against a running engine. Phase 5e.

    from bench.load import replay_poisson

The scheduler has been correct since Phase 5a and unmeasured under load ever since. This is the
driver that measures it: requests arrive on a Poisson process at a chosen rate, the engine serves
them concurrently, and every request's arrival, first token and completion are timestamped.

**Open loop, not closed.** A closed loop (N clients, each waiting for its own reply before sending
the next) cannot produce a queue, because arrivals are throttled by service. Queueing delay is the
entire subject here -- it is what p99 TTFT is made of and what chunked prefill exists to bound --
so arrivals must be independent of service, which is what an open loop with exponential gaps
gives. The cost is that an overloaded cell never reaches steady state, and that is reported
(``offered_per_s`` against ``completed_per_s``) rather than hidden by rescaling.

**Nothing here touches a tensor or knows what a model is.** The driver takes a request factory and
an engine; ``tests/test_load.py`` drives it with a scripted fake executor in milliseconds. The
script written for a T4 and executed nowhere else is ``BUGS.md`` B-05, and it produced a
well-formed file of zeros.

Two measurement points worth stating because they are choices:

* **TTFT is measured from submission, not from admission.** A request that waits 4 seconds for a
  block has a 4-second-worse TTFT and the user experiences all of it. Measuring from admission
  would report the scheduler's own latency and silently delete the queueing it caused.
* **The throughput window runs from the first submission to the last completion.** At low offered
  load that window contains idle GPU, so throughput reads low -- correctly. It is goodput under an
  offered load, not a peak capacity figure, and ``mean_inflight`` is reported alongside so the
  idle is visible rather than inferred.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bench.metrics import Percentiles
from edgerag.sched.request import Request
from edgerag.sched.scheduler import Scheduler
from edgerag.serve.engine import InferenceEngine, TokenEvent

#: How often the driver samples in-flight depth. 20 Hz is fine against decode steps measured in
#: tens of milliseconds and costs nothing; the sampler holds no lock the engine wants.
DEFAULT_SAMPLE_HZ = 20.0


@dataclass
class RequestOutcome:
    """One request's timeline. All times are ``perf_counter`` seconds on the driver's clock."""

    request_id: str
    submit_t: float
    first_token_t: float | None = None
    finish_t: float | None = None
    n_tokens: int = 0
    error: str | None = None

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_t is None:
            return None
        return self.first_token_t - self.submit_t

    @property
    def e2e_s(self) -> float | None:
        if self.finish_t is None:
            return None
        return self.finish_t - self.submit_t

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ttft_s": self.ttft_s,
            "e2e_s": self.e2e_s,
            "n_tokens": self.n_tokens,
            "error": self.error,
        }


@dataclass
class LoadResult:
    """What one offered-load cell produced."""

    offered_per_s: float
    n_requests: int
    n_completed: int
    n_failed: int
    n_never_finished: int
    window_s: float
    outcomes: list[RequestOutcome] = field(default_factory=list)
    max_inflight: int = 0
    mean_inflight: float = 0.0
    scheduler_delta: dict[str, int] = field(default_factory=dict)
    timed_out: bool = False

    @property
    def total_tokens(self) -> int:
        return sum(o.n_tokens for o in self.outcomes)

    def _ttfts(self) -> list[float]:
        return [o.ttft_s for o in self.outcomes if o.ttft_s is not None]

    def _e2es(self) -> list[float]:
        return [o.e2e_s for o in self.outcomes if o.e2e_s is not None and o.error is None]

    def to_dict(self) -> dict[str, Any]:
        ttfts, e2es = self._ttfts(), self._e2es()
        return {
            "offered_per_s": self.offered_per_s,
            "n_requests": self.n_requests,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "n_never_finished": self.n_never_finished,
            "window_s": self.window_s,
            "total_generated_tokens": self.total_tokens,
            # Goodput under this offered load, not peak capacity -- the window includes whatever
            # idle the arrival process left.
            "output_tokens_per_s": self.total_tokens / self.window_s if self.window_s else 0.0,
            "completed_per_s": self.n_completed / self.window_s if self.window_s else 0.0,
            "ttft_s": Percentiles.of(ttfts).to_dict() if ttfts else None,
            "e2e_s": Percentiles.of(e2es).to_dict() if e2es else None,
            "max_inflight": self.max_inflight,
            "mean_inflight": round(self.mean_inflight, 3),
            "scheduler_delta": self.scheduler_delta,
            "timed_out": self.timed_out,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def poisson_arrivals(n: int, rate_hz: float, seed: int) -> list[float]:
    """Cumulative arrival offsets for ``n`` requests at ``rate_hz``, exponential gaps.

    Seeded and returned up front rather than sampled as the run proceeds, so the arrival schedule
    is a property of ``(n, rate, seed)`` alone. Two cells run at the same rate then receive the
    *same* arrival pattern, which removes one source of difference between them -- the point of
    ``00_FOUNDATIONS.md`` §4 rule 5 applied to the workload's timing rather than its content.
    """
    if rate_hz <= 0:
        raise ValueError(f"arrival rate must be positive, got {rate_hz}")
    rng = random.Random(seed)
    offsets, clock = [], 0.0
    for _ in range(n):
        clock += rng.expovariate(rate_hz)
        offsets.append(clock)
    return offsets


def _stats_snapshot(scheduler: Scheduler) -> dict[str, int]:
    stats = scheduler.stats
    return {
        "steps": stats.steps,
        "admitted": stats.admitted,
        "finished": stats.finished,
        "rejected": stats.rejected,
        "preempted": stats.preempted,
        "prefill_chunks": stats.prefill_chunks,
        "decode_steps": stats.decode_steps,
        "admission_blocked": stats.admission_blocked,
    }


class _InflightSampler:
    """Polls ``len(scheduler.running)`` on its own thread.

    Sampled rather than derived from the outcome timeline because the timeline records what the
    *driver* saw -- submissions and tokens -- and the quantity of interest is what the scheduler
    was actually holding, which differs by exactly the queueing this measurement is about.
    """

    def __init__(self, scheduler: Scheduler, hz: float) -> None:
        self.scheduler = scheduler
        self.interval = 1.0 / hz
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="edgerag-inflight", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(len(self.scheduler.running))
            time.sleep(self.interval)

    def __enter__(self) -> _InflightSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def max(self) -> int:
        return max(self.samples) if self.samples else 0

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def replay_poisson(
    engine: InferenceEngine,
    scheduler: Scheduler,
    make_request: Callable[[int], Request],
    *,
    n_requests: int,
    arrival_rate_hz: float,
    seed: int = 1234,
    timeout_s: float = 600.0,
    sample_hz: float = DEFAULT_SAMPLE_HZ,
) -> LoadResult:
    """Submit ``n_requests`` on a Poisson process and wait for them all to finish.

    ``make_request(i)`` builds request *i*. It is called on the driver thread immediately before
    submission, so anything expensive in it (retrieval, a vision tower) lands inside the measured
    window -- callers who do not want that must build ahead and have the factory hand back a
    prepared object. ``scripts/colab_poisson.py`` does exactly that, and says why.

    Returns whatever was measured even on timeout. A cell that deadlocks because the pool cannot
    fit its requests is a finding, and raising would discard the evidence for it.
    """
    if n_requests <= 0:
        raise ValueError(f"n_requests must be positive, got {n_requests}")

    offsets = poisson_arrivals(n_requests, arrival_rate_hz, seed)
    outcomes: dict[str, RequestOutcome] = {}
    all_done = threading.Event()
    remaining = threading.Semaphore(0)
    finished_count = 0
    lock = threading.Lock()

    def make_emit(request_id: str) -> Callable[[TokenEvent], None]:
        # Called on the engine's worker thread, inside its step loop -- it must stay cheap.
        # A perf_counter read and a couple of field writes is the whole budget.
        def emit(event: TokenEvent) -> None:
            nonlocal finished_count
            now = time.perf_counter()
            outcome = outcomes[request_id]
            if event.token_id is not None:
                if outcome.first_token_t is None:
                    outcome.first_token_t = now
                outcome.n_tokens += 1
            if event.done:
                outcome.finish_t = now
                outcome.error = event.error
                with lock:
                    finished_count += 1
                    if finished_count >= n_requests:
                        all_done.set()
                remaining.release()

        return emit

    baseline = _stats_snapshot(scheduler)
    started = time.perf_counter()

    with _InflightSampler(scheduler, sample_hz) as sampler:
        for i, offset in enumerate(offsets):
            delay = started + offset - time.perf_counter()
            if delay > 0:
                # Sleeping the *residual* rather than the gap keeps the schedule absolute: a slow
                # make_request would otherwise push every later arrival back by its own cost and
                # quietly reduce the offered load being reported.
                time.sleep(delay)
            request = make_request(i)
            outcomes[request.request_id] = RequestOutcome(
                request_id=request.request_id, submit_t=time.perf_counter()
            )
            engine.submit(request, make_emit(request.request_id))

        deadline = started + timeout_s
        timed_out = not all_done.wait(timeout=max(0.0, deadline - time.perf_counter()))
        ended = time.perf_counter()

    ordered = [outcomes[r.request_id] for r in sorted(outcomes.values(), key=lambda o: o.submit_t)]
    first_submit = min((o.submit_t for o in ordered), default=started)
    last_finish = max((o.finish_t for o in ordered if o.finish_t is not None), default=ended)

    delta = _stats_snapshot(scheduler)
    return LoadResult(
        offered_per_s=arrival_rate_hz,
        n_requests=n_requests,
        n_completed=sum(1 for o in ordered if o.finish_t is not None and o.error is None),
        n_failed=sum(1 for o in ordered if o.error is not None),
        n_never_finished=sum(1 for o in ordered if o.finish_t is None),
        window_s=max(1e-9, last_finish - first_submit),
        outcomes=ordered,
        max_inflight=sampler.max,
        mean_inflight=sampler.mean,
        scheduler_delta={k: delta[k] - baseline[k] for k in delta},
        timed_out=timed_out,
    )


def drain(scheduler: Scheduler, engine: InferenceEngine, timeout_s: float = 30.0) -> bool:
    """Wait for the engine to go idle, then clear the finished list.

    Called between cells for two reasons. ``scheduler.finished`` grows without bound and each
    entry pins a ``Request`` -- which, in this project, pins a prompt-embedding tensor of tens of
    MiB. And the pool-conservation check that follows is only meaningful once nothing is running.
    """
    deadline = time.perf_counter() + timeout_s
    while scheduler.has_work and time.perf_counter() < deadline:
        time.sleep(0.05)
    scheduler.finished.clear()
    return not scheduler.has_work
