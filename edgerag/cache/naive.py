"""Contiguous KV cache -- the reference implementation.

This is the simplest thing that works: one preallocated ``(batch, kv_heads, max_seq, head_dim)``
tensor per layer, filled left to right.

It is **not** throwaway code. Phase 3's paged allocator is validated against it for the rest of the
project (``PLAN.md`` Phase 3 gate: paged logits == naive logits == HF logits). A reference you
delete is a reference you cannot diff against at 1 a.m. when the paged version is off by 1e-3.

Its limitations *are* the argument for paging, and they should be quoted rather than hidden:

* Capacity is reserved at ``max_seq_len`` per sequence, whatever the sequence actually uses. A
  request that stops at 300 tokens with a 2048 reservation wastes 85% of its allocation.
* Batch size is fixed at construction, so a finished sequence's memory cannot be reused by a new
  arrival until the whole batch drains.
* Nothing is shared between sequences, so 20 queries against one retrieved document store 20
  identical copies of its KV.

For the headline model at 192 KiB/token (``CONTEXT.md`` D11) those are not rounding errors --
one 2048-token reservation is 384 MiB.
"""

from __future__ import annotations

import torch

from edgerag.core.spec import ModelSpec


class NaiveKVCache:
    """Preallocated contiguous KV cache.

    Layout is ``(batch, kv_heads, max_seq_len, head_dim)`` per layer, matching the attention
    tensors so ``update`` is a slice-assign with no transpose.

    Caches **post-RoPE** keys, matching HF. Storing pre-RoPE keys would force re-rotating the
    entire prefix at every decode step, which is precisely the O(n) per-token work a cache exists
    to remove.
    """

    def __init__(
        self,
        spec: ModelSpec,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.spec = spec
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.seq_len = 0  # tokens currently held

        shape = (batch_size, spec.n_kv_heads, max_seq_len, spec.head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(spec.n_layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(spec.n_layers)]

    @property
    def nbytes(self) -> int:
        """Bytes reserved -- not bytes used. The gap is the fragmentation paging removes."""
        per = self.keys[0].numel() * self.keys[0].element_size()
        return 2 * per * self.spec.n_layers

    def used_bytes(self) -> int:
        if self.max_seq_len == 0:
            return 0
        return self.nbytes * self.seq_len // self.max_seq_len

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's K/V and return the full history to attend over.

        The length bookkeeping advances only on layer 0. Every layer sees the same tokens in the
        same forward pass, so advancing per layer would multiply the position by ``n_layers`` --
        a bug that survives a single-layer test and detonates on a real model.
        """
        new_len = key.shape[2]
        start = self.seq_len
        end = start + new_len

        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: {end} tokens exceeds max_seq_len={self.max_seq_len}. "
                "A contiguous cache cannot grow -- this is the failure paging removes."
            )

        self.keys[layer_idx][:, :, start:end] = key
        self.values[layer_idx][:, :, start:end] = value

        if layer_idx == self.spec.n_layers - 1:
            self.seq_len = end

        # Return a *view* of the filled region only. Returning the whole buffer would attend over
        # uninitialised zeros -- the contiguous-cache cousin of BUGS.md P-01.
        return self.keys[layer_idx][:, :, :end], self.values[layer_idx][:, :, :end]

    def reset(self) -> None:
        """Drop all history without reallocating.

        Buffers are not zeroed: every slot is overwritten before it is read, because ``update``
        only ever returns the filled prefix. Zeroing 384 MiB between requests would be pure waste.
        """
        self.seq_len = 0
