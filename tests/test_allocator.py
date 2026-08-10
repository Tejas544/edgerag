"""Tests for the physical block allocator.

No GPU, no tensors, no model -- the whole suite runs in milliseconds, which is the point. The
randomized tests at the bottom are the ones that matter: refcount bugs surface under long
interleavings of allocate/free/fork, not under hand-written sequences.
"""

from __future__ import annotations

import contextlib
import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edgerag.cache.allocator import (
    AllocatorError,
    BlockAllocator,
    DoubleFreeError,
    OutOfBlocksError,
)


def alloc(num_blocks: int = 16, block_size: int = 16) -> BlockAllocator:
    return BlockAllocator(num_blocks=num_blocks, block_size=block_size)


# --- construction -----------------------------------------------------------------------------


def test_starts_empty_and_consistent() -> None:
    a = alloc(8)
    assert a.num_free == 8
    assert a.num_used == 0
    assert a.utilization == 0.0
    a.check_invariants()


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_nonsense_dimensions(bad: int) -> None:
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=bad, block_size=16)
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=16, block_size=bad)


# --- allocate / free --------------------------------------------------------------------------


def test_allocate_hands_out_distinct_blocks() -> None:
    a = alloc(8)
    blocks = a.allocate(4)
    assert len(set(blocks)) == 4
    assert a.num_used == 4
    assert all(a.refcount(b) == 1 for b in blocks)
    a.check_invariants()


def test_free_returns_blocks_to_the_pool() -> None:
    a = alloc(8)
    blocks = a.allocate(4)
    assert a.free(blocks) == 4
    assert a.num_free == 8
    a.check_invariants()


def test_allocating_zero_is_a_noop() -> None:
    a = alloc(4)
    assert a.allocate(0) == []
    assert a.num_free == 4


def test_exhaustion_raises_and_is_counted() -> None:
    a = alloc(4)
    a.allocate(4)
    with pytest.raises(OutOfBlocksError, match="only 0 free"):
        a.allocate(1)
    assert a.stats.oom_events == 1


def test_failed_allocation_is_all_or_nothing() -> None:
    """A partial allocation would leave every caller needing unwind logic."""
    a = alloc(4)
    a.allocate(3)
    with pytest.raises(OutOfBlocksError):
        a.allocate(2)
    assert a.num_free == 1  # the one remaining block was not handed out
    a.check_invariants()


def test_free_list_is_lifo_for_cache_locality() -> None:
    """Freeing then reallocating returns the same block, still warm."""
    a = alloc(8)
    first = a.allocate(1)
    a.free(first)
    assert a.allocate(1) == first


def test_double_free_is_rejected() -> None:
    """BUGS.md P-05 -- the block would enter the free list twice."""
    a = alloc(4)
    blocks = a.allocate(2)
    a.free(blocks)
    with pytest.raises(DoubleFreeError, match="already free"):
        a.free(blocks)
    a.check_invariants()


def test_out_of_range_block_is_rejected() -> None:
    a = alloc(4)
    for bad in (-1, 4, 999):
        with pytest.raises(AllocatorError, match="out of range"):
            a.free([bad])


# --- reference counting / sharing ---------------------------------------------------------------


def test_incref_delays_reclamation() -> None:
    """The mechanic behind prefix sharing: two holders, one block."""
    a = alloc(4)
    blocks = a.allocate(1)
    a.incref(blocks)
    assert a.refcount(blocks[0]) == 2

    assert a.free(blocks) == 0  # first holder releases; nothing reclaimed
    assert a.num_free == 3
    assert a.free(blocks) == 1  # last holder releases; now reclaimed
    assert a.num_free == 4
    a.check_invariants()


def test_incref_on_a_free_block_is_rejected() -> None:
    """It would resurrect a block that is still listed as free."""
    a = alloc(4)
    blocks = a.allocate(1)
    a.free(blocks)
    with pytest.raises(AllocatorError, match="still in the free list"):
        a.incref(blocks)


def test_is_shared_tracks_refcount() -> None:
    a = alloc(4)
    b = a.allocate(1)
    assert a.is_shared(b[0]) is False
    a.incref(b)
    assert a.is_shared(b[0]) is True


# --- copy on write ------------------------------------------------------------------------------


def test_copy_on_write_splits_a_shared_block() -> None:
    a = alloc(8)
    b = a.allocate(1)[0]
    a.incref([b])  # two holders

    new = a.copy_on_write(b)
    assert new != b
    assert a.refcount(b) == 1  # the other holder keeps the original
    assert a.refcount(new) == 1
    assert a.stats.copy_on_writes == 1
    a.check_invariants()


def test_copy_on_write_is_a_noop_when_unshared() -> None:
    """Callers invoke it unconditionally; the allocator decides. Each caller reimplementing the
    'is it shared?' test is how one of them gets it wrong."""
    a = alloc(8)
    b = a.allocate(1)[0]
    assert a.copy_on_write(b) == b
    assert a.stats.copy_on_writes == 0


def test_copy_on_write_on_a_free_block_is_rejected() -> None:
    a = alloc(4)
    b = a.allocate(1)[0]
    a.free([b])
    with pytest.raises(AllocatorError, match="free block"):
        a.copy_on_write(b)


def test_copy_on_write_propagates_exhaustion() -> None:
    """CoW needs a spare block; under pressure it must fail loudly, not silently share on.

    This is a real operational hazard, not just an error path. A write to a shared prefix block
    demands a *free* block at precisely the moment the pool is fullest, so admission control has
    to reserve headroom for CoW rather than budgeting only for new sequences -- otherwise the
    system deadlocks exactly when prefix sharing is doing the most good. Feeds CONTEXT.md P1.
    """
    a = alloc(2)
    blocks = a.allocate(2)
    a.incref([blocks[0]])
    with pytest.raises(OutOfBlocksError):
        a.copy_on_write(blocks[0])
    # Critically, the failed CoW left the original sharing intact rather than half-splitting it.
    assert a.refcount(blocks[0]) == 2
    a.check_invariants()


# --- block arithmetic (BUGS.md P-02) --------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [(0, 0), (1, 1), (15, 1), (16, 1), (17, 2), (31, 2), (32, 2), (33, 3), (128, 8), (129, 9)],
)
def test_blocks_needed_at_boundaries(tokens: int, expected: int) -> None:
    """The exact-multiple case is the bug: an extra block leaks, a missing one IndexErrors."""
    assert alloc(64, 16).blocks_needed(tokens) == expected


def test_blocks_needed_rejects_negative() -> None:
    with pytest.raises(ValueError):
        alloc().blocks_needed(-1)


@pytest.mark.parametrize("block_size", [1, 4, 8, 16, 32])
def test_exact_multiples_never_over_allocate(block_size: int) -> None:
    a = alloc(256, block_size)
    for multiple in range(1, 9):
        assert a.blocks_needed(block_size * multiple) == multiple


# --- fragmentation accounting ---------------------------------------------------------------------


def test_internal_fragmentation_of_exact_fits_is_zero() -> None:
    a = alloc(64, 16)
    assert a.internal_fragmentation([16, 32, 48])["wasted_tokens"] == 0.0


def test_internal_fragmentation_counts_the_partial_tail() -> None:
    a = alloc(64, 16)
    # 17 tokens occupies 2 blocks = 32 slots, wasting 15.
    frag = a.internal_fragmentation([17])
    assert frag["wasted_tokens"] == 15.0
    assert frag["waste_fraction"] == pytest.approx(15 / 32)


def test_smaller_blocks_waste_less() -> None:
    """The quantified argument for tuning block_size rather than copying vLLM's default.

    At 192 KiB/token (CONTEXT.md D11) each wasted slot is real memory.
    """
    lengths = [100, 250, 617, 1003]
    waste_16 = alloc(4096, 16).internal_fragmentation(lengths)["wasted_tokens"]
    waste_4 = alloc(4096, 4).internal_fragmentation(lengths)["wasted_tokens"]
    assert waste_4 < waste_16


def test_empty_fragmentation_query_is_safe() -> None:
    assert alloc().internal_fragmentation([])["waste_fraction"] == 0.0


# --- randomized property tests ------------------------------------------------------------------


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    ops=st.lists(
        st.tuples(
            st.sampled_from(["alloc", "free", "incref", "cow"]),
            st.integers(min_value=1, max_value=4),
        ),
        min_size=1,
        max_size=120,
    )
)
def test_invariants_hold_under_arbitrary_operation_sequences(ops) -> None:
    """The test that actually catches refcount bugs.

    Hand-written sequences exercise the paths the author already thought about. Randomized
    interleavings of allocate/free/incref/copy-on-write are what surface a leak on an unusual
    path -- which is exactly BUGS.md P-04's failure mode, invisible until the pool drains hours
    into a load test.
    """
    a = BlockAllocator(num_blocks=12, block_size=8)
    held: list[int] = []

    for op, count in ops:
        if op == "alloc":
            with contextlib.suppress(OutOfBlocksError):
                held.extend(a.allocate(min(count, max(a.num_free, 0))))
        elif op == "free" and held:
            take = held[:count]
            held = held[count:]
            a.free(take)
        elif op == "incref" and held:
            targets = held[:count]
            a.incref(targets)
            held.extend(targets)  # the new references are also held
        elif op == "cow" and held:
            # CoW needs a spare block and can legitimately fail when the pool is full -- the
            # operational hazard documented in CONTEXT.md P1. Hypothesis finds this quickly.
            with contextlib.suppress(OutOfBlocksError):
                target = held[0]
                replacement = a.copy_on_write(target)
                if replacement != target:
                    held[0] = replacement
        a.check_invariants()

    # Releasing every held reference must return the pool to pristine.
    a.free(held)
    a.check_invariants()
    assert a.num_free == a.num_blocks, "pool leaked -- BUGS.md P-04"


def test_long_random_workload_conserves_blocks() -> None:
    """A deterministic soak, mirroring the request churn of a serving loop."""
    rng = random.Random(20260810)
    a = BlockAllocator(num_blocks=64, block_size=16)
    sequences: dict[int, list[int]] = {}

    for step in range(3000):
        action = rng.random()
        if action < 0.45 and a.num_free >= 4:
            sequences[step] = a.allocate(rng.randint(1, 4))
        elif action < 0.75 and sequences:
            key = rng.choice(list(sequences))
            a.free(sequences.pop(key))
        elif action < 0.9 and sequences:
            # Fork: a new sequence shares an existing one's blocks, as in prefix reuse.
            key = rng.choice(list(sequences))
            shared = sequences[key]
            a.incref(shared)
            sequences[f"fork{step}"] = list(shared)  # type: ignore[index]
        elif sequences:
            key = rng.choice(list(sequences))
            blocks = sequences[key]
            if blocks:
                # Expected under pressure, and the reason CoW needs an admission-control
                # reservation: splitting a shared block requires a spare at the moment the pool
                # is fullest. See CONTEXT.md P1.
                with contextlib.suppress(OutOfBlocksError):
                    blocks[0] = a.copy_on_write(blocks[0])
        a.check_invariants()

    for blocks in sequences.values():
        a.free(blocks)
    a.check_invariants()
    assert a.num_free == a.num_blocks
    assert a.stats.peak_used_blocks <= a.num_blocks
