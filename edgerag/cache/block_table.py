"""Per-sequence logical-to-physical block mapping.

A sequence sees a contiguous run of token positions ``0..n-1``. The pool sees scattered physical
blocks. The block table is the entire translation layer, and it is where the classic paged-cache
off-by-ones live -- so it is a plain Python object with no tensors, testable in microseconds,
exactly like the allocator (``CONTEXT.md`` Phase 3a rationale).

Two invariants carry the design:

* ``blocks`` holds *exactly* ``ceil(num_tokens / block_size)`` entries -- never a spare. A spare
  block leaks one block per sequence forever; a missing one means the next token has nowhere to go
  (``BUGS.md`` P-02). The case that breaks both is ``num_tokens % block_size == 0``.
* ``num_tokens`` is the only source of truth for sequence length. The physical capacity
  (``len(blocks) * block_size``) is always ``>=`` it, and the gap is the internal fragmentation
  paging trades for the removal of external fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edgerag.cache.allocator import BlockAllocator


@dataclass
class BlockTable:
    """Logical token positions -> physical block ids, for one sequence."""

    allocator: BlockAllocator
    blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0

    @property
    def block_size(self) -> int:
        return self.allocator.block_size

    @property
    def capacity(self) -> int:
        """Token slots physically held, including the unused tail of the final block."""
        return len(self.blocks) * self.block_size

    @property
    def slack(self) -> int:
        """Unused slots in the final block -- this sequence's internal fragmentation."""
        return self.capacity - self.num_tokens

    def locate(self, position: int) -> tuple[int, int]:
        """Map a logical position to ``(physical_block, offset_within_block)``.

        Rejects positions at or beyond ``num_tokens``. Allowing reads into the unwritten tail of
        the final block is precisely ``BUGS.md`` P-01 -- it returns whatever the previous tenant
        left, with no crash and no NaN.
        """
        if not 0 <= position < self.num_tokens:
            raise IndexError(
                f"position {position} outside the written region [0, {self.num_tokens}). "
                "Reading past it returns the previous tenant's data (BUGS.md P-01)."
            )
        return self.blocks[position // self.block_size], position % self.block_size

    def ensure_capacity(self, n_tokens: int) -> list[int]:
        """Grow so ``n_tokens`` fit. Returns the newly allocated blocks (empty if none needed)."""
        needed = self.allocator.blocks_needed(n_tokens)
        if needed <= len(self.blocks):
            return []
        fresh = self.allocator.allocate(needed - len(self.blocks))
        self.blocks.extend(fresh)
        return fresh

    def append(self, n_tokens: int = 1) -> list[int]:
        """Extend the sequence by ``n_tokens``, allocating blocks as required.

        Capacity is grown *before* the length is advanced, so an ``OutOfBlocksError`` leaves the
        table exactly as it was rather than claiming tokens it has nowhere to store.
        """
        fresh = self.ensure_capacity(self.num_tokens + n_tokens)
        self.num_tokens += n_tokens
        return fresh

    def fork(self) -> BlockTable:
        """Share every block with a new sequence (copy-on-write, Phase 3c).

        No KV is copied -- both tables point at the same physical blocks and each block's refcount
        rises. This is the operation that makes RAG cheap: 11 questions against one retrieved
        document, which the frozen trace actually contains, share one copy of its ~2,000 visual
        tokens rather than storing 11.
        """
        self.allocator.incref(self.blocks)
        self.allocator.stats.forks += 1
        return BlockTable(
            allocator=self.allocator, blocks=list(self.blocks), num_tokens=self.num_tokens
        )

    def unshare(self, logical_block: int) -> tuple[int, int] | None:
        """Make ``logical_block`` privately writable. Returns ``(old, new)`` if a copy happened.

        The caller must copy the block's *contents*; the table and allocator only handle
        bookkeeping. Forgetting the content copy is the classic CoW bug -- the new block is
        allocated, the mapping is updated, and the data is stale.
        """
        old = self.blocks[logical_block]
        new = self.allocator.copy_on_write(old)
        if new == old:
            return None
        self.blocks[logical_block] = new
        return old, new

    def writable_block_for(self, position: int) -> tuple[int, int, tuple[int, int] | None]:
        """Resolve a position for writing, splitting the block first if it is shared.

        Returns ``(physical_block, offset, copied)``. The single entry point for writes, because
        the bug this prevents -- appending into a block another sequence is still reading -- is
        silent and corrupts the *other* sequence.
        """
        logical_block = position // self.block_size
        copied = self.unshare(logical_block)
        return self.blocks[logical_block], position % self.block_size, copied

    def free(self) -> int:
        """Release every block. Returns how many actually returned to the pool.

        Shared blocks merely lose a reference, so the return value is less than ``len(blocks)``
        whenever a fork is still alive.
        """
        reclaimed = self.allocator.free(self.blocks)
        self.blocks = []
        self.num_tokens = 0
        return reclaimed

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "num_blocks": len(self.blocks),
            "capacity": self.capacity,
            "slack": self.slack,
            "blocks": list(self.blocks),
        }
