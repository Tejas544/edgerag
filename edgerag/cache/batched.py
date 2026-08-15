"""Several sequences, one forward pass.

Phase 5b. :class:`~edgerag.cache.paged.PagedKVCache` deliberately refuses batch > 1 -- one cache
holds one sequence, and combining them is the scheduler's problem. This is that combination.

Each sequence keeps its own block table and they all page into **one physical pool**, so a
finished request's blocks are immediately available to a new arrival. That is the property a
contiguous cache cannot offer, and the reason continuous batching is possible at all: a contiguous
cache cannot hand back a finished sequence's memory until the whole batch drains.

**The padding cost is real and is measured, not hidden.** Sequences in a batch have different
lengths, so the gathered buffer is padded to the batch maximum and the padded columns are masked
out. Those columns are wasted memory bandwidth -- in exactly the regime that D14 identified as the
bottleneck. :meth:`BatchedPagedCache.padding_waste` reports it, so the Phase 5 write-up can state
what batching costs as well as what it saves. If the waste is large, the answer is length-bucketed
admission rather than a bigger batch.
"""

from __future__ import annotations

from typing import Any

import torch

from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.paged import PagedKVCache
from edgerag.core.spec import ModelSpec


class BatchedPagedCache:
    """A batch of independent paged sequences presented as one cache.

    Implements the same ``update(key, value, layer_idx)`` contract as the single-sequence cache, so
    the decoder is unchanged. What the decoder *does* need from the scheduler is per-sequence
    ``position_ids`` and a ``padding_mask`` -- with unequal lengths there is no single scalar
    ``past_len``, and assuming one is how a batched cache silently attends to the wrong positions.
    """

    def __init__(
        self,
        spec: ModelSpec,
        allocator: BlockAllocator,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.spec = spec
        self.allocator = allocator
        self.device = device
        self.dtype = dtype
        self.sequences: list[PagedKVCache] = []
        # One pool shared by every member, created once and handed to each sequence.
        shape = (allocator.num_blocks, allocator.block_size, spec.n_kv_heads, spec.head_dim)
        self.key_pool = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(spec.n_layers)
        ]
        self.value_pool = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(spec.n_layers)
        ]

    # --- membership ---------------------------------------------------------------------------

    def new_sequence(self) -> PagedKVCache:
        """A cache for one sequence, paging into the shared pool. Not yet part of the batch."""
        return PagedKVCache(
            self.spec,
            self.allocator,
            self.device,
            self.dtype,
            pool=self.key_pool,
            value_pool=self.value_pool,
        )

    def set_batch(self, sequences: list[PagedKVCache]) -> None:
        """Define which sequences participate in the next forward.

        Membership is set per iteration rather than mutated incrementally, because that is exactly
        what iteration-level scheduling means: the batch is whatever is runnable *now*, and a
        request admitted this step joins without waiting for the current batch to drain.
        """
        for seq in sequences:
            if seq.key_pool is not self.key_pool:
                raise ValueError("sequence pages into a different pool")
        self.sequences = list(sequences)

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def lengths(self) -> list[int]:
        return [seq.seq_len for seq in self.sequences]

    @property
    def max_len(self) -> int:
        return max(self.lengths, default=0)

    @property
    def seq_len(self) -> int:
        """The **padded** batch width, which is what the causal mask must be sized against.

        The decoder reads this to size its mask, so it has to describe the gathered buffer rather
        than any individual sequence. Per-sequence differences are expressed by
        :meth:`padding_mask` and :meth:`position_ids`; omitting this property entirely leaves the
        decoder inferring ``past_len = 0`` and building a mask one column wide against a history
        of hundreds.
        """
        return self.max_len

    # --- the batched forward contract -----------------------------------------------------------

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write one step for every sequence and return the padded batch history.

        ``key``/``value`` are ``(batch, kv_heads, n_new, head_dim)``. Every sequence in a batched
        *decode* contributes exactly one token; batched prefill of differing chunk lengths is not
        supported and is rejected rather than silently mis-slicing.
        """
        if key.shape[0] != self.batch_size:
            raise ValueError(
                f"key batch {key.shape[0]} does not match batch size {self.batch_size}"
            )
        if self.batch_size == 0:
            raise RuntimeError("no sequences in the batch")

        for i, seq in enumerate(self.sequences):
            seq.update(key[i : i + 1], value[i : i + 1], layer_idx)

        return self.gather(layer_idx)

    def gather(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reassemble every sequence, right-padded to the batch maximum.

        **Right**-padding, not left: each sequence's tokens must stay at positions ``0..len_i-1``
        so the causal mask and the per-sequence ``position_ids`` line up. The padded tail is hidden
        by :meth:`padding_mask`.
        """
        max_len = self.max_len
        keys = torch.zeros(
            (self.batch_size, self.spec.n_kv_heads, max_len, self.spec.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        values = torch.zeros_like(keys)

        for i, seq in enumerate(self.sequences):
            seq_keys, seq_values = seq.gather(layer_idx)
            length = seq_keys.shape[2]
            keys[i, :, :length] = seq_keys[0]
            values[i, :, :length] = seq_values[0]
        return keys, values

    def padding_mask(self, n_new: int = 1) -> torch.Tensor:
        """``(batch, max_len + n_new)`` -- ``True`` where a position holds a real token.

        Without this the shorter sequences attend to zero-filled padding. That is `BUGS.md` P-01
        wearing a different hat: no crash, no NaN, and an answer conditioned partly on nothing.

        ``n_new`` accounts for the tokens *about to be written*. The caller builds this before the
        forward pass, but the mask must describe the cache **after** it, because that is what
        attention will see. Defaulting to 1 matches a decode step; pass 0 to describe the current
        state.
        """
        width = self.max_len + n_new
        mask = torch.zeros((self.batch_size, width), dtype=torch.bool, device=self.device)
        for i, length in enumerate(self.lengths):
            mask[i, : length + n_new] = True
        return mask

    def position_ids(self) -> torch.Tensor:
        """``(batch, 1)`` next position per sequence, for the decode step.

        Each sequence continues from its own length. A single scalar ``past_len`` -- which is what
        the decoder infers when it is not told otherwise -- would give every sequence the longest
        member's positions and rotate their RoPE wrongly (`BUGS.md` P-03/P-09).
        """
        return torch.tensor(
            [[length] for length in self.lengths], dtype=torch.long, device=self.device
        )

    # --- accounting -----------------------------------------------------------------------------

    def padding_waste(self) -> dict[str, Any]:
        """How much of the gathered buffer is padding.

        The honest counterweight to any batching speedup: these bytes cross the memory bus and
        contribute nothing. Reported rather than assumed, because D14 established that decode is
        bandwidth-bound and this is bandwidth spent on zeros.
        """
        lengths = self.lengths
        if not lengths:
            return {"batch_size": 0, "padded_tokens": 0, "waste_fraction": 0.0}

        padded = self.max_len * len(lengths)
        real = sum(lengths)
        return {
            "batch_size": len(lengths),
            "max_len": self.max_len,
            "min_len": min(lengths),
            "real_tokens": real,
            "padded_tokens": padded,
            "wasted_tokens": padded - real,
            "waste_fraction": (padded - real) / padded if padded else 0.0,
        }

    @property
    def nbytes(self) -> int:
        return sum(seq.nbytes for seq in self.sequences)

    def free_all(self) -> int:
        reclaimed = sum(seq.free() for seq in self.sequences)
        self.sequences = []
        return reclaimed
