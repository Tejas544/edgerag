"""Paged KV cache: scattered physical blocks presented as a contiguous history.

Phase 3b. Implements ``CONTEXT.md`` D3 -- gather the sequence's blocks into a scratch buffer, then
hand the result to the same eager attention the naive cache feeds. No fused kernel.

The gather costs a copy. It buys the entire memory thesis: no external fragmentation, per-sequence
growth on demand, and copy-on-write prefix sharing. Only the copy is added, and its cost is
measured and published rather than hidden -- which is a better answer to *"why didn't you fuse
it?"* than a shrug (D3).

**Why P-01 cannot happen in this design.** The single most dangerous bug in a paged cache is
attending to the unwritten tail of the final partial block: no crash, no NaN, fluent and wrong
output, and only when ``seq_len % block_size != 0``. Here :meth:`gather` slices the reassembled
buffer to ``num_tokens`` before returning it, so the padding slots are never handed to attention
at all. That is a structural guarantee rather than a mask that has to be right -- and the boundary
sweep in the tests exists to keep it that way.

Pool layout is ``(num_blocks, block_size, kv_heads, head_dim)`` per layer. Chosen so that
gathering is ``pool[block_ids]`` followed by a *free* reshape: the block and slot axes are already
adjacent and contiguous, so flattening them copies nothing extra.
"""

from __future__ import annotations

import torch

from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.block_table import BlockTable
from edgerag.core.spec import ModelSpec


class PagedKVCache:
    """Block-paged KV storage for a single sequence, with the naive cache's interface.

    Presenting the same ``update(key, value, layer_idx) -> (keys, values)`` contract as
    :class:`~edgerag.cache.naive.NaiveKVCache` is deliberate: the decoder is written once and does
    not know which cache it holds, so the equivalence test compares two storage strategies rather
    than two model implementations.
    """

    def __init__(
        self,
        spec: ModelSpec,
        allocator: BlockAllocator,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        pool: list[torch.Tensor] | None = None,
        value_pool: list[torch.Tensor] | None = None,
    ) -> None:
        self.spec = spec
        self.allocator = allocator
        self.device = device
        self.dtype = dtype
        self.table = BlockTable(allocator=allocator)

        shape = (allocator.num_blocks, allocator.block_size, spec.n_kv_heads, spec.head_dim)
        # Pools are shareable so several sequences can page into one physical arena -- the whole
        # point of the allocator. Phase 3c passes them in explicitly.
        self.key_pool = pool or [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(spec.n_layers)
        ]
        self.value_pool = value_pool or [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(spec.n_layers)
        ]
        self._pending_tokens = 0

    # --- interface shared with NaiveKVCache ---------------------------------------------------

    @property
    def seq_len(self) -> int:
        return self.table.num_tokens

    @property
    def nbytes(self) -> int:
        """Bytes *reserved by this sequence* -- blocks held, not pool size.

        The contrast with the naive cache is the headline: naive reserves ``max_seq_len`` per
        sequence whatever it uses; this reserves ``ceil(len / block_size)`` blocks.
        """
        per_block = self.allocator.block_size * self.spec.n_kv_heads * self.spec.head_dim
        itemsize = torch.finfo(self.dtype).bits // 8
        return 2 * len(self.table.blocks) * per_block * itemsize * self.spec.n_layers

    def used_bytes(self) -> int:
        """Bytes actually written, excluding the final block's slack."""
        if not self.table.blocks:
            return 0
        return self.nbytes * self.seq_len // self.table.capacity

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's K/V into blocks and return the full history.

        ``key``/``value`` are ``(1, kv_heads, n_new, head_dim)``, matching the attention layout.

        Growth happens once, on layer 0, not per layer. Every layer of one forward pass covers the
        same tokens, so allocating per layer would claim ``n_layers`` times the blocks and advance
        the length ``n_layers`` times -- a bug that survives any single-layer test.
        """
        if key.shape[0] != 1:
            raise NotImplementedError(
                "PagedKVCache holds one sequence; batching is the scheduler's job (Phase 3c)"
            )
        n_new = key.shape[2]

        if layer_idx == 0:
            self._pending_tokens = n_new
            self.table.append(n_new)
            # Split any shared block this write will touch, once, across every layer's pool.
            self._unshare_write_range(self.seq_len - n_new, n_new)

        start = self.seq_len - self._pending_tokens
        self._write(self.key_pool[layer_idx], key, start, n_new)
        self._write(self.value_pool[layer_idx], value, start, n_new)

        return self.gather(layer_idx)

    def _unshare_write_range(self, start: int, n_new: int) -> None:
        """Copy-on-write every shared block the range ``[start, start+n_new)`` will write into.

        In practice this is at most one block -- the partially-filled tail a fork left behind.
        Full blocks are never rewritten, and blocks allocated by this very ``append`` are private
        by construction. But the range is walked rather than assumed, because "at most one" is a
        property of the current append pattern, not an invariant of the class.

        **The content copy is the part that is easy to omit.** ``BlockTable.unshare`` allocates a
        fresh block and repoints the mapping; without copying the old block's KV into it, the
        sequence keeps a valid-looking mapping to a block full of zeros, and the answer degrades
        rather than crashing.

        Done at layer 0 for every layer's pool at once. Doing it per layer would leave layers 1..N
        pointing at the old block while layer 0 points at the new one.
        """
        if not self.table.blocks:
            return

        block_size = self.allocator.block_size
        first = start // block_size
        last = (start + n_new - 1) // block_size

        for logical in range(first, last + 1):
            copied = self.table.unshare(logical)
            if copied is None:
                continue
            old, new = copied
            for layer in range(self.spec.n_layers):
                self.key_pool[layer][new].copy_(self.key_pool[layer][old])
                self.value_pool[layer][new].copy_(self.value_pool[layer][old])

    # --- block-level primitives ---------------------------------------------------------------

    def _write(
        self, pool: torch.Tensor, source: torch.Tensor, start: int, n_new: int
    ) -> None:
        """Scatter ``n_new`` tokens into their blocks, starting at logical position ``start``.

        ``source`` is ``(1, kv_heads, n_new, head_dim)``; the pool wants ``(slot, kv_heads,
        head_dim)``, hence the transpose.

        Index tensors are built on the device in one shot. Building them as Python lists and
        converting per step is a hidden host-to-device sync in the hot loop, which makes paged
        *slower* than naive and gets misattributed to the gather (``BUGS.md`` P-06).
        """
        positions = torch.arange(start, start + n_new, device=self.device)
        block_ids = torch.tensor(self.table.blocks, device=self.device, dtype=torch.long)

        block_of = block_ids[positions // self.allocator.block_size]
        offset_of = positions % self.allocator.block_size

        pool[block_of, offset_of] = source[0].transpose(0, 1).to(pool.dtype)

    def gather(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reassemble the sequence's blocks into contiguous ``(1, kv_heads, seq_len, head_dim)``.

        The ``[:seq_len]`` slice is what makes ``BUGS.md`` P-01 structurally impossible: the final
        block's unwritten slots are dropped before attention ever sees them.
        """
        if not self.table.blocks:
            empty = torch.zeros(
                (1, self.spec.n_kv_heads, 0, self.spec.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty

        block_ids = torch.tensor(self.table.blocks, device=self.device, dtype=torch.long)
        keys = self._gather_pool(self.key_pool[layer_idx], block_ids)
        values = self._gather_pool(self.value_pool[layer_idx], block_ids)
        return keys, values

    def _gather_pool(self, pool: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        # (n_blocks, block_size, kv_heads, head_dim) -- block and slot axes adjacent, so the
        # flatten below is a view, not a copy.
        blocks = pool[block_ids]
        flat = blocks.reshape(-1, self.spec.n_kv_heads, self.spec.head_dim)
        # Drop the final block's slack *before* attention sees it. This is the P-01 guarantee.
        flat = flat[: self.seq_len]
        return flat.transpose(0, 1).unsqueeze(0).contiguous()

    # --- sharing (Phase 3c) -------------------------------------------------------------------

    def fork(self) -> PagedKVCache:
        """Branch a new sequence sharing this one's blocks, copying no KV."""
        child = PagedKVCache(
            self.spec,
            self.allocator,
            self.device,
            self.dtype,
            pool=self.key_pool,
            value_pool=self.value_pool,
        )
        child.table = self.table.fork()
        return child

    def free(self) -> int:
        return self.table.free()

    def reset(self) -> None:
        self.free()
        self._pending_tokens = 0
