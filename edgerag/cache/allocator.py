"""Physical block allocator for the paged KV cache.

Phase 3a. **Deliberately contains no tensors and no CUDA.** The allocator is a data structure --
a free list plus reference counts -- and testing it as one is the single highest-leverage decision
in this phase.

The reason is diagnostic, not aesthetic. Most of the pain in a paged attention implementation is
allocator bugs *presenting as* attention bugs: output is subtly wrong, so the search starts at the
masking and the block gather, and the actual defect is a refcount that leaked three requests ago.
Proving the allocator correct in isolation, with randomized property tests that run in
milliseconds, means that when paged attention later disagrees with the naive cache the allocator
is already excluded.

Sizing, from ``CONTEXT.md`` D11: the headline model costs 192 KiB of KV per token, so a 16-token
block is **3 MiB** and a 2 GiB pool holds only ~680 blocks. Two consequences that differ from the
usual vLLM intuition, which was tuned on GQA models with ~8x cheaper tokens:

* Block-table overhead is negligible -- hundreds of blocks, not hundreds of thousands.
* Internal fragmentation is expensive -- one wasted block is 3 MiB, and at ~680 blocks total,
  wasting half a block per sequence is real money.

Both argue for *smaller* blocks than the default 16, which is why ``block_size`` is a parameter
with a sweep planned rather than a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Sentinel for "no physical block". Using -1 rather than None keeps block tables as plain int
#: lists, which matters once they become device tensors in Phase 3b (``BUGS.md`` P-06).
NO_BLOCK = -1


class AllocatorError(RuntimeError):
    """Base class for allocator misuse. Always a bug in the caller, never a resource condition."""


class OutOfBlocksError(AllocatorError):
    """The pool is exhausted.

    Distinct from the other errors: this one is an expected runtime condition that the Phase 3d
    preemption policy handles, not a programming mistake.
    """


class DoubleFreeError(AllocatorError):
    """A block was freed while already free.

    Its own exception type because the consequence is uniquely nasty: the block re-enters the free
    list twice, is handed to two live sequences, and they silently overwrite each other's KV. The
    symptom is corruption in an *unrelated* request (``BUGS.md`` P-05).
    """


@dataclass
class AllocatorStats:
    """Counters for the fragmentation and utilisation story.

    Cheap to maintain and impossible to reconstruct after the fact, which is why they are built in
    from the start rather than added when a plot is needed.
    """

    total_blocks: int
    block_size: int
    allocations: int = 0
    frees: int = 0
    forks: int = 0
    copy_on_writes: int = 0
    peak_used_blocks: int = 0
    oom_events: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class BlockAllocator:
    """Fixed-size physical block pool with reference counting.

    Reference counts, not plain ownership, because copy-on-write prefix sharing (Phase 3c) is the
    point of the whole design: when 11 questions share one retrieved document -- which the frozen
    trace actually contains -- they share its blocks, and a block is only reclaimed when the last
    holder releases it.

    The free list is a **stack**, not a queue. Freeing and immediately reallocating returns the
    same block, which is warm in cache and in the allocator's own metadata. A FIFO queue would
    cycle through the entire pool before reuse, maximising cache misses for no benefit.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self._refcount = [0] * num_blocks
        # Descending so the first allocations come from block 0 upward, which makes traces and
        # debug dumps far easier to read.
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self.stats = AllocatorStats(total_blocks=num_blocks, block_size=block_size)

    # --- inspection ---------------------------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    @property
    def utilization(self) -> float:
        return self.num_used / self.num_blocks

    def refcount(self, block: int) -> int:
        return self._refcount[block]

    def check_invariants(self) -> None:
        """Assert the allocator's structural invariants.

        Called after every operation in tests, and after every load test in the benchmark suite.
        Conservation is the one that catches leaks: a refcount that is never decremented on the
        error path drains the pool over hours and is invisible to unit tests (``BUGS.md`` P-04).
        """
        free_set = set(self._free)
        if len(free_set) != len(self._free):
            raise AllocatorError(
                f"free list contains duplicates: {len(self._free)} entries, "
                f"{len(free_set)} distinct. A block handed out twice corrupts two sequences."
            )

        for block in range(self.num_blocks):
            is_free = block in free_set
            count = self._refcount[block]
            if is_free and count != 0:
                raise AllocatorError(f"block {block} is free but has refcount {count}")
            if not is_free and count <= 0:
                raise AllocatorError(f"block {block} is allocated but has refcount {count}")

        if self.num_used + self.num_free != self.num_blocks:
            raise AllocatorError(
                f"conservation violated: {self.num_used} used + {self.num_free} free "
                f"!= {self.num_blocks} total"
            )

    # --- allocation ---------------------------------------------------------------------------

    def allocate(self, n_blocks: int = 1) -> list[int]:
        """Take ``n_blocks`` from the pool, each at refcount 1.

        All-or-nothing. A partial allocation would leave the caller holding blocks for a sequence
        it cannot admit, and every caller would need unwinding logic -- so the pool is checked
        first and nothing is handed out on failure.
        """
        if n_blocks < 0:
            raise ValueError(f"cannot allocate {n_blocks} blocks")
        if n_blocks == 0:
            return []
        if n_blocks > self.num_free:
            self.stats.oom_events += 1
            raise OutOfBlocksError(
                f"requested {n_blocks} blocks, only {self.num_free} free "
                f"({self.num_used}/{self.num_blocks} in use)"
            )

        blocks = [self._free.pop() for _ in range(n_blocks)]
        for block in blocks:
            self._refcount[block] = 1

        self.stats.allocations += n_blocks
        self.stats.peak_used_blocks = max(self.stats.peak_used_blocks, self.num_used)
        return blocks

    def free(self, blocks: list[int]) -> int:
        """Drop one reference to each block; return how many actually returned to the pool.

        A shared block's refcount falls without it being reclaimed -- that is the whole point of
        prefix sharing, and the return value makes it observable.
        """
        reclaimed = 0
        for block in blocks:
            self._validate(block)
            if self._refcount[block] == 0:
                raise DoubleFreeError(
                    f"block {block} freed while already free. It would enter the free list twice "
                    "and be handed to two live sequences (BUGS.md P-05)."
                )
            self._refcount[block] -= 1
            if self._refcount[block] == 0:
                self._free.append(block)
                reclaimed += 1

        self.stats.frees += len(blocks)
        return reclaimed

    def incref(self, blocks: list[int]) -> None:
        """Add a reference -- the sharing half of copy-on-write.

        Increfing a free block is rejected rather than silently resurrecting it: that would hand
        a block with no owner to a new holder while the free list still lists it.
        """
        for block in blocks:
            self._validate(block)
            if self._refcount[block] == 0:
                raise AllocatorError(
                    f"cannot incref free block {block} -- it is still in the free list"
                )
        for block in blocks:
            self._refcount[block] += 1

    def is_shared(self, block: int) -> bool:
        """Whether a write to this block must copy first (Phase 3c)."""
        self._validate(block)
        return self._refcount[block] > 1

    def copy_on_write(self, block: int) -> int:
        """Split a shared block: allocate a fresh one and drop this holder's reference.

        Returns the new block id, or ``block`` unchanged when it was not shared -- so callers can
        invoke it unconditionally and let the allocator decide, rather than each caller
        reimplementing the "is it shared?" test and getting it wrong somewhere.

        The caller still has to copy the *contents*; the allocator only owns the bookkeeping.
        """
        self._validate(block)
        if self._refcount[block] == 0:
            raise AllocatorError(f"cannot copy-on-write free block {block}")
        if self._refcount[block] == 1:
            return block

        new_block = self.allocate(1)[0]
        self._refcount[block] -= 1
        self.stats.copy_on_writes += 1
        return new_block

    def _validate(self, block: int) -> None:
        if not 0 <= block < self.num_blocks:
            raise AllocatorError(f"block id {block} out of range [0, {self.num_blocks})")

    # --- reporting ----------------------------------------------------------------------------

    def blocks_needed(self, n_tokens: int) -> int:
        """Blocks required to hold ``n_tokens``.

        ``ceil`` division, and the case that matters is ``n_tokens % block_size == 0``: it must
        produce exactly ``n_tokens // block_size`` blocks and not one spare. Getting this wrong
        either leaks a block per sequence forever or leaves the next token with nowhere to go
        (``BUGS.md`` P-02).
        """
        if n_tokens < 0:
            raise ValueError(f"negative token count {n_tokens}")
        return (n_tokens + self.block_size - 1) // self.block_size

    def internal_fragmentation(self, sequence_lengths: list[int]) -> dict[str, float]:
        """Tokens of capacity wasted in the final partial block of each sequence.

        This is the cost paging *adds* relative to a perfectly-packed cache, and it is the honest
        counterweight to the external-fragmentation savings. Reporting only the win would be the
        kind of selective accounting a careful reader catches.
        """
        if not sequence_lengths:
            return {"wasted_tokens": 0.0, "wasted_blocks_equiv": 0.0, "waste_fraction": 0.0}

        wasted = sum(
            (self.blocks_needed(n) * self.block_size) - n for n in sequence_lengths
        )
        capacity = sum(self.blocks_needed(n) * self.block_size for n in sequence_lengths)
        return {
            "wasted_tokens": float(wasted),
            "wasted_blocks_equiv": wasted / self.block_size,
            "waste_fraction": wasted / capacity if capacity else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "num_free": self.num_free,
            "num_used": self.num_used,
            "utilization": round(self.utilization, 4),
            "stats": {
                "allocations": self.stats.allocations,
                "frees": self.stats.frees,
                "forks": self.stats.forks,
                "copy_on_writes": self.stats.copy_on_writes,
                "peak_used_blocks": self.stats.peak_used_blocks,
                "oom_events": self.stats.oom_events,
            },
        }
