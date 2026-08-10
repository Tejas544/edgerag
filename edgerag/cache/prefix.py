"""Automatic prefix sharing by content hash.

Phase 3c. ``BlockTable.fork`` shares blocks when the caller already knows two sequences are
related. This finds that relationship automatically: a new request hashes its prompt block by
block, and any leading run that some earlier request already computed is reused instead of
recomputed.

That is the RAG-specific win. ``CONTEXT.md`` D15 measured it at **8.3%** of KV on the frozen trace
(14.8% with canonical document ordering), and it also removes prefill *compute* for the matched
prefix -- which at a measured 3.7 s TTFT (D14) is the more valuable half.

**Only full blocks are cacheable.** A partially-filled block will be appended to, so sharing it
would immediately trigger copy-on-write and gain nothing while risking the corruption demonstrated
in ``BUGS.md`` B-03. The hash chain therefore covers whole blocks only.

**The cache owns a reference to every block it holds.** Without that, a cached block could be
freed by its last sequence, returned to the pool, reallocated to unrelated data, and then handed
out by a stale cache entry -- silently serving one request another's context. Owning a reference
makes that impossible; the cost is that cached blocks are not reclaimable until evicted, which is
what ``max_blocks`` bounds.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from edgerag.cache.allocator import BlockAllocator

#: Chain seed. Distinguishes "empty prefix" from "no prefix", which would otherwise collide.
ROOT_HASH = "root"


def chain_hash(parent: str, tokens: tuple[int, ...]) -> str:
    """Hash of a block's contents *and* everything before it.

    Chaining is what makes the match a genuine *prefix* match. Hashing each block independently
    would let a block match at the wrong position -- two requests sharing their third block but
    not their first would incorrectly share KV, and the reused keys would carry RoPE for the
    wrong positions.
    """
    digest = hashlib.sha256()
    digest.update(parent.encode())
    digest.update(b"|")
    digest.update(",".join(map(str, tokens)).encode())
    return digest.hexdigest()[:32]


@dataclass
class PrefixStats:
    lookups: int = 0
    hits: int = 0
    blocks_reused: int = 0
    tokens_reused: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class PrefixCache:
    """Maps chained block hashes to physical blocks, so identical prefixes are computed once.

    LRU-bounded. Eviction order matters: a document retrieved by many queries should outlive a
    one-off prompt, and recency approximates that well on the observed trace, where 95 of 362
    pages carry multiple questions.
    """

    def __init__(self, allocator: BlockAllocator, max_blocks: int | None = None) -> None:
        self.allocator = allocator
        # Cached blocks are pinned, so an unbounded cache would starve the pool. Default to a
        # quarter of it -- enough to hold the hot documents, small enough to leave working room.
        default_cap = max(1, allocator.num_blocks // 4)
        self.max_blocks = max_blocks if max_blocks is not None else default_cap
        self._entries: OrderedDict[str, int] = OrderedDict()
        self.stats = PrefixStats()

    @property
    def num_cached(self) -> int:
        return len(self._entries)

    def block_hashes(self, token_ids: list[int]) -> list[tuple[str, tuple[int, ...]]]:
        """Chained hash for each *full* block of ``token_ids``.

        The trailing partial block is deliberately excluded -- it is not shareable.
        """
        block_size = self.allocator.block_size
        n_full = len(token_ids) // block_size

        hashes: list[tuple[str, tuple[int, ...]]] = []
        parent = ROOT_HASH
        for i in range(n_full):
            chunk = tuple(token_ids[i * block_size : (i + 1) * block_size])
            parent = chain_hash(parent, chunk)
            hashes.append((parent, chunk))
        return hashes

    def lookup(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Longest cached prefix of ``token_ids``. Returns ``(blocks, tokens_matched)``.

        Matched blocks are **increfed for the caller**, who now owns those references and must
        free them with the rest of the sequence. Returning borrowed blocks would be a use-after-free
        waiting for the cache to evict.

        Matching stops at the first miss. A later block cannot be reused without the blocks before
        it, since the sequence's positions must stay contiguous.
        """
        self.stats.lookups += 1
        matched: list[int] = []

        for block_hash, _ in self.block_hashes(token_ids):
            block = self._entries.get(block_hash)
            if block is None:
                break
            self._entries.move_to_end(block_hash)
            matched.append(block)

        if not matched:
            return [], 0

        self.allocator.incref(matched)
        self.stats.hits += 1
        self.stats.blocks_reused += len(matched)
        tokens = len(matched) * self.allocator.block_size
        self.stats.tokens_reused += tokens
        return matched, tokens

    def register(self, token_ids: list[int], blocks: list[int]) -> int:
        """Cache the full blocks of a completed prefill. Returns how many entries were added.

        The cache takes its own reference on each block it stores, so registration does not
        transfer ownership -- the caller keeps its own reference and frees it normally.
        """
        added = 0
        for i, (block_hash, _) in enumerate(self.block_hashes(token_ids)):
            if i >= len(blocks) or block_hash in self._entries:
                continue
            self._evict_if_needed()
            if self.num_cached >= self.max_blocks:
                break
            block = blocks[i]
            self.allocator.incref([block])
            self._entries[block_hash] = block
            added += 1
        return added

    def _evict_if_needed(self) -> None:
        while self.num_cached >= self.max_blocks and self._entries:
            _, block = self._entries.popitem(last=False)  # least recently used
            self.allocator.free([block])
            self.stats.evictions += 1

    def clear(self) -> None:
        """Drop every entry, releasing the cache's references."""
        for block in self._entries.values():
            self.allocator.free([block])
        self._entries.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_cached": self.num_cached,
            "max_blocks": self.max_blocks,
            "lookups": self.stats.lookups,
            "hits": self.stats.hits,
            "hit_rate": round(self.stats.hit_rate, 4),
            "blocks_reused": self.stats.blocks_reused,
            "tokens_reused": self.stats.tokens_reused,
            "evictions": self.stats.evictions,
        }
