"""Phase 5a/5d: request lifecycle and scheduling policy.

No tensors, no GPU, no model -- the whole module runs in milliseconds, the same discipline as
``tests/test_allocator.py``. A scheduler bug that presents as wrong output gets misdiagnosed as a
cache bug for hours; proving the policy in isolation means the batched-decode work can exclude it.
"""

from __future__ import annotations

import random

import pytest

from edgerag.cache.allocator import BlockAllocator
from edgerag.sched.request import Request, RequestState
from edgerag.sched.scheduler import Batch, Scheduler, SchedulerConfig

BLOCK = 16


def make_scheduler(
    num_blocks: int = 64, block_size: int = BLOCK, **config: object
) -> Scheduler:
    return Scheduler(BlockAllocator(num_blocks, block_size), SchedulerConfig(**config))  # type: ignore[arg-type]


def make_request(rid: str, prompt: int = 100, new: int = 20) -> Request:
    return Request(request_id=rid, prompt_token_ids=list(range(prompt)), max_new_tokens=new)


# --- request lifecycle ---------------------------------------------------------------------------


def test_prefill_chunks_cover_the_prompt_exactly() -> None:
    request = make_request("r", prompt=250)
    seen = 0
    while not request.prefill_done:
        chunk = request.next_prefill_chunk(100)
        assert chunk == request.prompt_token_ids[seen : seen + len(chunk)]
        request.advance_prefill(len(chunk))
        seen += len(chunk)
    assert seen == 250


def test_next_chunk_does_not_advance() -> None:
    """Peek and commit are separate so a failed forward does not fake progress."""
    request = make_request("r", prompt=100)
    assert len(request.next_prefill_chunk(30)) == 30
    assert request.prefill_offset == 0


def test_final_chunk_is_short_not_padded() -> None:
    request = make_request("r", prompt=250)
    request.advance_prefill(200)
    assert len(request.next_prefill_chunk(100)) == 50


def test_advancing_past_the_prompt_is_rejected() -> None:
    request = make_request("r", prompt=10)
    with pytest.raises(ValueError, match="past prompt length"):
        request.advance_prefill(11)


def test_total_tokens_counts_prompt_plus_generated() -> None:
    request = make_request("r", prompt=100)
    request.append_token(5, step=1)
    request.append_token(6, step=2)
    assert request.total_tokens == 102


def test_stops_at_max_new_tokens() -> None:
    request = make_request("r", new=3)
    for i in range(3):
        request.append_token(i, step=i)
    assert request.should_stop() is True


def test_stops_on_eos() -> None:
    request = make_request("r", new=100)
    request.append_token(7, step=0)
    assert request.should_stop(eos_token_id=7) is True
    assert request.should_stop(eos_token_id=99) is False


def test_ttft_measured_from_arrival_not_admission() -> None:
    """Queue time is part of what a user waits, so it belongs in TTFT."""
    request = make_request("r")
    request.arrival_step = 10
    request.admitted_step = 25
    request.append_token(1, step=30)
    assert request.ttft_steps() == 20


# --- admission ------------------------------------------------------------------------------------


def test_admits_up_to_the_batch_limit() -> None:
    sched = make_scheduler(num_blocks=256, max_batch_size=3, max_prefills_per_step=8)
    for i in range(5):
        sched.add_request(make_request(f"r{i}", prompt=32, new=16))

    batch = sched.schedule()
    assert len(batch.prefill) == 3
    assert len(sched.waiting) == 2


def test_admission_budgets_for_the_final_size_not_the_first_chunk() -> None:
    """Admitting on the first chunk turns a clean rejection into a preemption later."""
    sched = make_scheduler(num_blocks=8, block_size=16, cow_reserve_blocks=0)
    # 100 prompt + 60 generated = 160 tokens = 10 blocks > the 8 available.
    sched.add_request(make_request("big", prompt=100, new=60))

    batch = sched.schedule()
    assert batch.is_empty
    assert sched.stats.admission_blocked == 1


def test_cow_reserve_is_withheld_from_admission() -> None:
    """CoW can itself raise OutOfBlocksError; spending the last block deadlocks prefix sharing."""
    sched = make_scheduler(num_blocks=10, cow_reserve_blocks=4)
    assert sched.available_blocks == 6

    sched.allocator.allocate(6)
    assert sched.available_blocks == 0
    assert sched.allocator.num_free == 4  # the reserve survives


def test_queue_order_is_preserved() -> None:
    """Head-of-line blocking is deliberate: reordering starves whoever does not fit."""
    sched = make_scheduler(num_blocks=256, max_prefills_per_step=8)
    for i in range(4):
        sched.add_request(make_request(f"r{i}", prompt=32, new=16))

    batch = sched.schedule()
    assert [r.request_id for r in batch.prefill] == ["r0", "r1", "r2", "r3"]


def test_decode_is_scheduled_before_new_admissions() -> None:
    """Running requests are closest to releasing blocks, so they drain first."""
    sched = make_scheduler(num_blocks=256, max_batch_size=2, max_prefills_per_step=8)
    running = make_request("old", prompt=32, new=16)
    running.state = RequestState.DECODING
    sched.running.append(running)
    sched.add_request(make_request("new", prompt=32, new=16))

    batch = sched.schedule()
    assert [r.request_id for r in batch.decode] == ["old"]
    assert len(batch.prefill) == 1


def test_prefills_per_step_is_capped() -> None:
    """One chunked prefill per iteration keeps decode latency predictable (BUGS.md P-18)."""
    sched = make_scheduler(num_blocks=512, max_batch_size=8, max_prefills_per_step=1)
    for i in range(4):
        sched.add_request(make_request(f"r{i}", prompt=32, new=16))

    assert len(sched.schedule().prefill) == 1


# --- progress -------------------------------------------------------------------------------------


def test_prefill_completion_moves_to_decoding() -> None:
    sched = make_scheduler(num_blocks=256, max_prefills_per_step=8)
    sched.add_request(make_request("r", prompt=100, new=10))
    request = sched.schedule().prefill[0]

    sched.on_prefill_chunk(request, 60)
    assert request.state is RequestState.PREFILLING
    sched.on_prefill_chunk(request, 40)
    assert request.state is RequestState.DECODING


def test_finishing_releases_blocks_and_leaves_running() -> None:
    sched = make_scheduler(num_blocks=64, max_prefills_per_step=8)
    sched.add_request(make_request("r", prompt=32, new=1))
    request = sched.schedule().prefill[0]
    sched.on_prefill_chunk(request, 32)

    sched.on_token(request, token_id=5)
    assert request.state is RequestState.FINISHED
    assert request not in sched.running
    assert sched.stats.finished == 1


def test_a_finished_request_frees_capacity_for_the_next(monkeypatch) -> None:
    """The property a contiguous cache cannot offer: reuse without draining the batch."""
    sched = make_scheduler(num_blocks=8, block_size=16, max_batch_size=1, cow_reserve_blocks=0)
    sched.add_request(make_request("a", prompt=64, new=32))  # 6 blocks
    sched.add_request(make_request("b", prompt=64, new=32))

    first = sched.schedule().prefill[0]
    assert first.request_id == "a"

    # "b" must not be admitted while "a" holds the pool. The batch is not empty -- "a" is still
    # prefilling in it -- so assert on the admission count, which is the thing under test.
    admitted_before = sched.stats.admitted
    sched.schedule()
    assert sched.stats.admitted == admitted_before, "second request was admitted too early"

    sched.finish(first)
    assert sched.schedule().prefill[0].request_id == "b"


# --- preemption -----------------------------------------------------------------------------------


def _fill_running(sched: Scheduler, names: tuple[str, ...], blocks_each: int) -> None:
    """Put running requests in the pool and exhaust it.

    Exhausting matters: ``preempt_to_free`` correctly evicts *nobody* when the requested blocks are
    already free, so a test that leaves slack proves nothing about victim selection.
    """
    for name in names:
        request = make_request(name, prompt=32, new=16)
        request.state = RequestState.DECODING
        request.cache = _FakeCache(sched.allocator, blocks=blocks_each)
        sched.running.append(request)
    if sched.allocator.num_free:
        sched.allocator.allocate(sched.allocator.num_free)  # soak up the remainder
    assert sched.allocator.num_free == 0


def test_no_victims_when_the_pool_already_has_room() -> None:
    """Preemption is a last resort, not a reflex."""
    sched = make_scheduler(num_blocks=32, block_size=16, cow_reserve_blocks=0)
    request = make_request("r")
    request.state = RequestState.DECODING
    request.cache = _FakeCache(sched.allocator, blocks=2)
    sched.running.append(request)

    assert sched.preempt_to_free(blocks_needed=2) == []
    assert request.state is RequestState.DECODING


def test_victims_are_newest_first() -> None:
    """Oldest-first discards the most accumulated work and livelocks long requests."""
    sched = make_scheduler(num_blocks=12, block_size=16, cow_reserve_blocks=0)
    _fill_running(sched, ("old", "mid", "new"), blocks_each=3)

    victims = sched.preempt_to_free(blocks_needed=3)
    assert [v.request_id for v in victims] == ["new"]


def test_victims_return_to_the_front_of_the_queue() -> None:
    """Sending them to the back lets new arrivals starve them indefinitely."""
    sched = make_scheduler(num_blocks=12, block_size=16, cow_reserve_blocks=0)
    _fill_running(sched, ("victim",), blocks_each=3)
    sched.add_request(make_request("newcomer"))

    sched.preempt_to_free(blocks_needed=3)
    assert sched.waiting[0].request_id == "victim"


def test_preemption_resets_prefill_progress() -> None:
    """This scheduler re-prefills on readmission; believing in KV it no longer holds is worse."""
    sched = make_scheduler(num_blocks=12, block_size=16, cow_reserve_blocks=0)
    _fill_running(sched, ("r",), blocks_each=3)
    request = sched.running[0]
    request.prefill_offset = 32

    sched.preempt_to_free(blocks_needed=3)
    assert request.prefill_offset == 0
    assert request.cache is None
    assert request.preemptions == 1


def test_preemption_stops_once_enough_is_free() -> None:
    sched = make_scheduler(num_blocks=12, block_size=16, cow_reserve_blocks=0)
    _fill_running(sched, ("a", "b", "c"), blocks_each=2)

    victims = sched.preempt_to_free(blocks_needed=2)
    assert len(victims) == 1, "freed more than was asked for"


def test_rejection_is_terminal_and_counted() -> None:
    sched = make_scheduler()
    request = make_request("r")
    sched.add_request(request)
    sched.reject(request, reason="overloaded")

    assert request.state is RequestState.REJECTED
    assert request.is_terminal
    assert sched.stats.rejected == 1
    assert request not in sched.waiting


# --- invariants under load ------------------------------------------------------------------------


class _FakeCache:
    """Holds real blocks from the real allocator, without any tensors."""

    def __init__(self, allocator: BlockAllocator, blocks: int) -> None:
        self.allocator = allocator
        self.table = _FakeTable(allocator.allocate(blocks))

    @property
    def seq_len(self) -> int:
        return len(self.table.blocks) * self.allocator.block_size

    def free(self) -> int:
        reclaimed = self.allocator.free(self.table.blocks)
        self.table.blocks = []
        return reclaimed


class _FakeTable:
    def __init__(self, blocks: list[int]) -> None:
        self.blocks = blocks
        self.num_tokens = 0


def test_pool_is_conserved_across_a_randomized_load() -> None:
    """The soak that catches leaks on unusual paths -- BUGS.md P-04's failure mode.

    Admission, completion, preemption and rejection interleave randomly; the pool must return to
    pristine once every request is terminal.
    """
    rng = random.Random(20260811)
    sched = make_scheduler(num_blocks=48, block_size=16, max_batch_size=4, cow_reserve_blocks=2)

    for step in range(400):
        if rng.random() < 0.4:
            sched.add_request(make_request(f"r{step}", prompt=rng.randint(16, 96), new=16))

        batch = sched.schedule()
        for request in batch.prefill:
            if request.cache is None:
                request.cache = _FakeCache(sched.allocator, blocks=1)
            sched.on_prefill_chunk(request, len(request.next_prefill_chunk(64)))
        for request in list(batch.decode):
            sched.on_token(request, token_id=rng.randint(0, 99))

        if rng.random() < 0.08 and sched.running:
            sched.preempt_to_free(blocks_needed=2)

        sched.end_step()
        sched.allocator.check_invariants()

    for request in list(sched.running):
        sched.finish(request)
    for request in list(sched.waiting):
        sched.reject(request, "drain")

    sched.allocator.check_invariants()
    assert sched.allocator.num_free == sched.allocator.num_blocks, "pool leaked"


def test_batch_reports_emptiness_and_size() -> None:
    batch = Batch(step=0)
    assert batch.is_empty and batch.size == 0
    batch.decode.append(make_request("r"))
    assert not batch.is_empty and batch.size == 1
