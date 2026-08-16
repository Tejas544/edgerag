"""Two-tier KV cache: full history below the pruning layer, pruned history above it.

Phase 4. FastV drops visual tokens after layer *k*, so layers ``0..k-1`` and ``k..L-1`` hold
*different numbers of tokens*. ``CONTEXT.md`` D5 chose to prune once, at the prefill/decode
boundary, precisely so that this is **two** block tables rather than one per layer.

Routing by layer index keeps the decoder oblivious: it still calls
``cache.update(k, v, layer_idx)`` and does not know whether the sequence was pruned.

The memory win is proportional to how much of the model sits above ``k``. With ``k=2`` on a
24-layer model, **92% of layers** hold the pruned set, so the saving is close to the raw pruning
ratio.
"""

from __future__ import annotations

from typing import Any

import torch

from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.paged import PagedKVCache
from edgerag.core.spec import ModelSpec


class CompressedKVCache:
    """Routes layer ``i`` to the full cache when ``i < score_layer``, else to the pruned cache.

    Both halves page into the **same physical pool**, so freed blocks from either are immediately
    reusable by the other -- the property a contiguous cache cannot offer and the reason this is
    built on the allocator rather than on two independent buffers.
    """

    def __init__(
        self,
        spec: ModelSpec,
        allocator: BlockAllocator,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        score_layer: int = 2,
    ) -> None:
        if not 0 <= score_layer <= spec.n_layers:
            raise ValueError(f"score_layer {score_layer} outside [0, {spec.n_layers}]")

        self.spec = spec
        self.allocator = allocator
        self.score_layer = score_layer

        # Each half stores only the layers it serves. Sharing one full-stack pool -- the obvious
        # design, and the original one -- means the `full` cache reserves slots in all L layers
        # while writing only `score_layer` of them. At the default cut that is 22 of 24 layers
        # wasted on one half, and it doubled the block demand: at keep_ratio=1.0 both halves hold
        # the entire sequence, so a 7,000-token prompt needed 874 blocks from a 576-block pool.
        # Split this way the two pools together cost exactly one full stack.
        self.full = PagedKVCache(
            spec, allocator, device, dtype, first_layer=0, n_pool_layers=score_layer
        )
        self.pruned = PagedKVCache(
            spec,
            allocator,
            device,
            dtype,
            first_layer=score_layer,
            n_pool_layers=spec.n_layers - score_layer,
        )

    def _route(self, layer_idx: int) -> PagedKVCache:
        return self.full if layer_idx < self.score_layer else self.pruned

    @property
    def seq_len(self) -> int:
        """The **full** sequence length, which is what position ids must follow.

        Deliberately not the pruned length. Pruned tokens are removed from storage, not
        renumbered: a surviving token keeps the RoPE rotation it was computed with, and the next
        generated token continues from the original count. Reporting the pruned length here would
        renumber positions and silently change every relative distance the model was trained on.
        """
        return self.full.seq_len

    @property
    def pruned_len(self) -> int:
        return self.pruned.seq_len

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._route(layer_idx).update(key, value, layer_idx)

    @property
    def nbytes(self) -> int:
        return self.full.nbytes + self.pruned.nbytes

    def used_bytes(self) -> int:
        return self.full.used_bytes() + self.pruned.used_bytes()

    def uncompressed_nbytes(self) -> int:
        """What this sequence would cost with no pruning, for the saving ratio.

        Every layer would hold the full token count, so it is the full cache's per-layer cost
        scaled across the whole stack.
        """
        per_layer_full = self.full.nbytes / self.spec.n_layers
        return int(per_layer_full * self.spec.n_layers)

    def savings(self) -> dict[str, Any]:
        """Bytes reclaimed by pruning, and the fraction of the stack that benefits.

        ``CONTEXT.md`` D15 asks for the Phase 4 curve in **MiB reclaimed**, not just percent of
        tokens removed -- the token ratio understates the win because most layers sit above the
        pruning point.
        """
        layers_above = self.spec.n_layers - self.score_layer
        per_token_per_layer = 2 * self.spec.n_kv_heads * self.spec.head_dim * 2
        dropped = max(self.full.seq_len - self.pruned.seq_len, 0)
        reclaimed = dropped * layers_above * per_token_per_layer
        would_be = self.full.seq_len * self.spec.n_layers * per_token_per_layer
        return {
            "full_tokens": self.full.seq_len,
            "pruned_tokens": self.pruned.seq_len,
            "dropped_tokens": dropped,
            "layers_above_cut": layers_above,
            "bytes_reclaimed": reclaimed,
            "mib_reclaimed": reclaimed / (1024**2),
            "uncompressed_bytes": would_be,
            "saving_fraction": reclaimed / would_be if would_be else 0.0,
        }

    def free(self) -> int:
        return self.full.free() + self.pruned.free()

    def reset(self) -> None:
        self.full.reset()
        self.pruned.reset()
