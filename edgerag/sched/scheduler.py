"""Continuous-batching scheduler: iteration-level admission with chunked prefill.

Phase 5a/5c/5d. The scheduler decides, once per model iteration, *what runs next* -- it does not
run it. Keeping the policy free of tensors means the interesting logic (admission under pressure,
chunked-prefill progress, preemption and readmission) is testable in milliseconds against a fake
engine, which is the same split that made the allocator debuggable.

Three decisions worth stating, because each has an alternative that looks reasonable:

**Prefill is chunked and shares the iteration with decode.** A 6,800-token prefill run to
completion blocks every decoding request behind it, and D14 measured what that costs: TTFT went
from 3.7 s at batch 1 to 25 s at batch 4. Chunking bounds the damage one arrival can do to
everyone already running -- the classic head-of-line problem (`BUGS.md` P-18).

**Admission reserves headroom rather than spending the last block.** Copy-on-write can itself raise
``OutOfBlocksError`` (found in Phase 3a): writing to a shared prefix block needs a *free* block at
the moment the pool is fullest. A scheduler that admits until the pool is empty deadlocks exactly
when prefix sharing is doing the most good, so ``cow_reserve_blocks`` is held back.

**Preempted requests re-enter the waiting queue rather than resuming directly.** ``swap_in`` needs
blocks that may have been taken while the request was parked, so restoration is an admission
decision, not a guaranteed operation (`CONTEXT.md` D16).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from edgerag.cache.allocator import BlockAllocator, OutOfBlocksError
from edgerag.cache.preemption import Preemptor
from edgerag.sched.request import PREEMPTABLE, Request, RequestState

#: Tokens of prompt processed per iteration. 512 keeps a worst-case prefill chunk comparable in
#: cost to a decode step over a long context, so neither starves the other. Swept in Phase 8.
DEFAULT_PREFILL_CHUNK = 512


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8
    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK
    #: Blocks never handed to a new admission, kept for copy-on-write splits by running requests.
    cow_reserve_blocks: int = 4
    #: A batch may mix prefill and decode, but not unboundedly -- one chunked prefill per
    #: iteration keeps decode latency predictable.
    max_prefills_per_step: int = 1
    eos_token_id: int | None = None


@dataclass
class SchedulerStats:
    steps: int = 0
    admitted: int = 0
    finished: int = 0
    rejected: int = 0
    preempted: int = 0
    prefill_chunks: int = 0
    decode_steps: int = 0
    admission_blocked: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """What the engine should run this iteration.

    Prefill and decode are separate lists because they are different shapes -- a prefill request
    contributes many tokens and a decoding one contributes exactly one.
    """

    step: int
    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode

    @property
    def size(self) -> int:
        return len(self.prefill) + len(self.decode)

    @property
    def requests(self) -> list[Request]:
        """Everything in this batch, when the caller does not care which phase it is in.

        The engine needs this on the error path: if a forward pass raises, every request that was
        in flight has to be failed, prefilling and decoding alike.
        """
        return [*self.prefill, *self.decode]


class Scheduler:
    """Iteration-level scheduler over a shared block pool."""

    def __init__(
        self,
        allocator: BlockAllocator,
        config: SchedulerConfig | None = None,
        preemptor: Preemptor | None = None,
    ) -> None:
        self.allocator = allocator
        self.config = config or SchedulerConfig()
        self.preemptor = preemptor
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.step_index = 0
        self.stats = SchedulerStats()

    # --- queue ---------------------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        request.arrival_step = self.step_index
        request.state = RequestState.WAITING
        self.waiting.append(request)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def available_blocks(self) -> int:
        """Blocks a *new* admission may use, after holding back the CoW reserve."""
        return max(0, self.allocator.num_free - self.config.cow_reserve_blocks)

    def blocks_required(self, request: Request) -> int:
        """Blocks to hold this request's whole prompt plus everything it will generate.

        Admission budgets for the **final** size, not the first chunk. Admitting on the first
        chunk and discovering at token 4,000 that the pool is gone converts a clean rejection into
        a preemption -- and a preemption costs a swap or a full recompute (D16).
        """
        return self.allocator.blocks_needed(request.prompt_len + request.max_new_tokens)

    # --- the scheduling decision ------------------------------------------------------------------

    def schedule(self) -> Batch:
        """Choose this iteration's work. Pure decision-making: nothing is executed here."""
        batch = Batch(step=self.step_index)

        # Decode first. Requests already past prefill have consumed resources and are closest to
        # releasing them, so draining them takes priority over admitting more work.
        for request in self.running:
            if request.state is RequestState.DECODING:
                batch.decode.append(request)
            elif (
                request.state is RequestState.PREFILLING
                and len(batch.prefill) < self.config.max_prefills_per_step
            ):
                batch.prefill.append(request)

        # Then admit, while capacity allows.
        # `self.running` already contains everything in `batch.prefill` and `batch.decode`, so the
        # capacity check reads it alone. Adding the batch lists to it double-counts every admitted
        # request and silently caps the batch at half `max_batch_size`.
        while (
            self.waiting
            and len(self.running) < self.config.max_batch_size
            and len(batch.prefill) < self.config.max_prefills_per_step
        ):
            request = self.waiting[0]
            needed = self.blocks_required(request)
            if needed > self.available_blocks:
                self.stats.admission_blocked += 1
                break
            self.waiting.popleft()
            request.state = RequestState.PREFILLING
            request.admitted_step = self.step_index
            self.running.append(request)
            self.stats.admitted += 1
            batch.prefill.append(request)

        return batch

    # --- progress bookkeeping ---------------------------------------------------------------------

    def on_prefill_chunk(self, request: Request, n_tokens: int) -> None:
        """Record that ``n_tokens`` of prompt were processed."""
        request.advance_prefill(n_tokens)
        self.stats.prefill_chunks += 1
        if request.prefill_done:
            request.state = RequestState.DECODING

    def on_token(self, request: Request, token_id: int) -> None:
        request.append_token(token_id, self.step_index)
        self.stats.decode_steps += 1
        if request.should_stop(self.config.eos_token_id):
            self.finish(request)

    def finish(self, request: Request) -> None:
        request.state = RequestState.FINISHED
        request.finished_step = self.step_index
        if request.cache is not None:
            request.cache.free()
            request.cache = None
        if request in self.running:
            self.running.remove(request)
        self.finished.append(request)
        self.stats.finished += 1

    def end_step(self) -> None:
        self.step_index += 1
        self.stats.steps += 1

    # --- pressure -----------------------------------------------------------------------------

    def preempt_to_free(self, blocks_needed: int) -> list[Request]:
        """Evict running requests, newest first, until ``blocks_needed`` blocks are free.

        Newest-first protects accumulated work and avoids the livelock where the request that has
        waited longest is repeatedly the one reset (D16).

        Victims return to the **front** of the waiting queue: they arrived before everything behind
        them, and sending them to the back would let a steady stream of new arrivals starve a
        request indefinitely.
        """
        victims: list[Request] = []
        candidates = [r for r in self.running if r.state in PREEMPTABLE]

        for request in reversed(candidates):
            if self.allocator.num_free >= blocks_needed:
                break
            victims.append(request)
            self._preempt_one(request)

        for request in reversed(victims):
            self.waiting.appendleft(request)
        return victims

    def _preempt_one(self, request: Request) -> None:
        blocks = list(request.cache.table.blocks) if request.cache is not None else []
        if self.preemptor is not None and blocks:
            self.preemptor.preempt(
                request.request_id,
                blocks,
                request.total_tokens,
                key_pool=getattr(request.cache, "key_pool", None),
                value_pool=getattr(request.cache, "value_pool", None),
            )
            if request.cache is not None:
                request.cache.table.blocks = []
                request.cache.table.num_tokens = 0
        elif request.cache is not None:
            request.cache.free()

        request.mark_preempted()
        request.state = RequestState.WAITING
        request.cache = None
        # Swapping preserves KV, but this scheduler re-prefills on readmission: restoring is an
        # admission decision that can itself fail, and a request that believes it has KV it no
        # longer holds is a far worse failure than a repeated prefill.
        request.prefill_offset = 0
        self.running.remove(request)
        self.stats.preempted += 1

    def reject(self, request: Request, reason: str = "") -> None:
        """Refuse a request outright -- honest under sustained overload."""
        request.state = RequestState.REJECTED
        request.finished_step = self.step_index
        if request in self.waiting:
            self.waiting.remove(request)
        self.stats.rejected += 1
        self.stats.extra.setdefault("reject_reasons", []).append(reason)
        self.finished.append(request)

    # --- reporting ----------------------------------------------------------------------------

    def try_allocate(self, request: Request, n_tokens: int) -> bool:
        """Grow a request's cache, preempting others if that is what it takes.

        Returns ``False`` when even preemption cannot make room, which the caller must treat as a
        rejection rather than retrying forever.
        """
        if request.cache is None:
            return False
        try:
            request.cache.table.ensure_capacity(request.cache.seq_len + n_tokens)
            return True
        except OutOfBlocksError:
            needed = self.allocator.blocks_needed(request.cache.seq_len + n_tokens)
            self.preempt_to_free(needed)
            try:
                request.cache.table.ensure_capacity(request.cache.seq_len + n_tokens)
                return True
            except OutOfBlocksError:
                return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_index,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
            "free_blocks": self.allocator.num_free,
            "available_for_admission": self.available_blocks,
            "stats": {
                "steps": self.stats.steps,
                "admitted": self.stats.admitted,
                "finished": self.stats.finished,
                "rejected": self.stats.rejected,
                "preempted": self.stats.preempted,
                "prefill_chunks": self.stats.prefill_chunks,
                "decode_steps": self.stats.decode_steps,
                "admission_blocked": self.stats.admission_blocked,
            },
        }
