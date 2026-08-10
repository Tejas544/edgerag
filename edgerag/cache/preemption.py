"""Preemption: what happens when the block pool runs out.

Phase 3d. Three policies are implementable; the choice is argued from this project's own measured
numbers rather than from the vLLM paper's defaults, because the workloads differ sharply.

| policy | cost to resume | pool freed | notes |
|---|---|---|---|
| **SWAP** | ~1.27 GiB over PCIe, about **100 ms** | all victim blocks | default |
| RECOMPUTE | one full prefill, **3,730 ms** (D14) | all victim blocks | 37x worse here |
| REJECT | n/a, the request dies | all victim blocks | honest under sustained overload |

**Why SWAP, when the plan originally leaned RECOMPUTE.** The original argument was "prefill is
compute-bound, so a T4 has spare FLOPs and recompute is cheap." The measurement in ``CONTEXT.md``
D14 kills it: TTFT is **3.73 s at batch 1**, because a k=5 RAG prompt is ~6,758 tokens of which
75% are visual and must traverse the vision tower again. Recompute does not cost microseconds
here, it costs seconds, and a preempted request would blow p99 far past the point where the
scheduler's throughput gain is worth anything.

Swapping the same sequence's KV to pinned host memory moves ~1.27 GiB, which over PCIe 3.0 x16
(~12 GB/s achievable) is roughly 100 ms — **an order of magnitude cheaper than recomputing it**,
and it does not re-run the vision tower at all.

**Where that flips.** Swap wins because these sequences are long and expensive to rebuild. For
short prompts, recompute is cheaper than a round trip and vLLM's default is right. The policy is
therefore a parameter with a measured justification, not a constant.

**Victim selection is newest-first.** Preempting the oldest request would discard the most
accumulated work and risks starving long requests indefinitely under load — each preemption
resets the request that has waited longest. Newest-first preserves progress already paid for, at
the cost of penalising arrivals during a spike, which is the fairer trade when the alternative is
livelock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import torch

from edgerag.cache.allocator import BlockAllocator


class PreemptionPolicy(StrEnum):
    SWAP = "swap"
    RECOMPUTE = "recompute"
    REJECT = "reject"


class Preemptable(Protocol):
    """What the preemptor needs from a sequence. Keeps this module free of scheduler types."""

    @property
    def seq_len(self) -> int: ...

    def free(self) -> int: ...


@dataclass
class SwappedSequence:
    """A victim's KV, parked in host memory.

    Stored per layer as ``(n_blocks, block_size, kv_heads, head_dim)`` CPU tensors -- the same
    layout as the device pool, so restoring is a straight copy with no reshaping.
    """

    request_id: str
    num_tokens: int
    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    @property
    def nbytes(self) -> int:
        per = self.keys[0].numel() * self.keys[0].element_size()
        return 2 * per * len(self.keys)


@dataclass
class PreemptionStats:
    preemptions: int = 0
    swaps_out: int = 0
    swaps_in: int = 0
    recomputes: int = 0
    rejects: int = 0
    bytes_swapped: int = 0
    blocks_reclaimed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Preemptor:
    """Frees pool capacity by evicting live sequences under the configured policy."""

    def __init__(
        self,
        allocator: BlockAllocator,
        policy: PreemptionPolicy = PreemptionPolicy.SWAP,
        pin_memory: bool = True,
    ) -> None:
        self.allocator = allocator
        self.policy = policy
        # Pinned host memory makes the device-to-host copy DMA-able and roughly 2x faster. It is
        # also non-pageable, so an unbounded swap area would exhaust host RAM -- the scheduler
        # bounds how many sequences may sit swapped.
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.swapped: dict[str, SwappedSequence] = {}
        self.stats = PreemptionStats()

    def swap_out(
        self,
        request_id: str,
        blocks: list[int],
        num_tokens: int,
        key_pool: list[torch.Tensor],
        value_pool: list[torch.Tensor],
    ) -> SwappedSequence:
        """Copy a victim's blocks to host memory and release them.

        The copy happens **before** the blocks are freed. Freeing first would return them to the
        pool where another sequence could claim and overwrite them mid-copy -- a race that shows
        up as one request answering with another's context, exactly the shape of ``BUGS.md`` B-03.
        """
        keys = [pool[blocks].to("cpu", copy=True) for pool in key_pool]
        values = [pool[blocks].to("cpu", copy=True) for pool in value_pool]

        if self.pin_memory:
            keys = [k.pin_memory() for k in keys]
            values = [v.pin_memory() for v in values]

        record = SwappedSequence(request_id, num_tokens, keys, values)
        self.swapped[request_id] = record

        reclaimed = self.allocator.free(blocks)
        self.stats.preemptions += 1
        self.stats.swaps_out += 1
        self.stats.bytes_swapped += record.nbytes
        self.stats.blocks_reclaimed += reclaimed
        return record

    def swap_in(
        self,
        request_id: str,
        key_pool: list[torch.Tensor],
        value_pool: list[torch.Tensor],
    ) -> tuple[list[int], int]:
        """Reallocate blocks and restore a swapped sequence. Returns ``(blocks, num_tokens)``.

        Raises ``OutOfBlocksError`` if the pool cannot take it back, which the scheduler must
        handle -- swapping in is itself an admission decision, not a guaranteed operation.
        """
        record = self.swapped.get(request_id)
        if record is None:
            raise KeyError(f"no swapped sequence {request_id!r}")

        n_blocks = record.keys[0].shape[0]
        blocks = self.allocator.allocate(n_blocks)

        for layer, (keys, values) in enumerate(zip(record.keys, record.values, strict=True)):
            key_pool[layer][blocks] = keys.to(key_pool[layer].device, non_blocking=self.pin_memory)
            value_pool[layer][blocks] = values.to(
                value_pool[layer].device, non_blocking=self.pin_memory
            )

        del self.swapped[request_id]
        self.stats.swaps_in += 1
        self.stats.bytes_swapped += record.nbytes
        return blocks, record.num_tokens

    def preempt(
        self,
        request_id: str,
        blocks: list[int],
        num_tokens: int,
        key_pool: list[torch.Tensor] | None = None,
        value_pool: list[torch.Tensor] | None = None,
    ) -> int:
        """Apply the configured policy. Returns blocks returned to the pool."""
        if self.policy is PreemptionPolicy.SWAP:
            if key_pool is None or value_pool is None:
                raise ValueError("SWAP requires the KV pools")
            record = self.swap_out(request_id, blocks, num_tokens, key_pool, value_pool)
            return len(record.keys[0])

        reclaimed = self.allocator.free(blocks)
        self.stats.preemptions += 1
        self.stats.blocks_reclaimed += reclaimed
        if self.policy is PreemptionPolicy.RECOMPUTE:
            self.stats.recomputes += 1
        else:
            self.stats.rejects += 1
        return reclaimed

    @staticmethod
    def select_victims(
        candidates: list[tuple[str, int]], blocks_needed: int
    ) -> list[str]:
        """Choose whom to evict, newest first, until ``blocks_needed`` is covered.

        ``candidates`` is ``(request_id, blocks_held)`` in arrival order. Newest-first protects
        accumulated work and avoids the livelock where the longest-waiting request is repeatedly
        the one reset.
        """
        victims: list[str] = []
        freed = 0
        for request_id, held in reversed(candidates):
            if freed >= blocks_needed:
                break
            victims.append(request_id)
            freed += held
        return victims

    @property
    def swapped_bytes(self) -> int:
        return sum(record.nbytes for record in self.swapped.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "pin_memory": self.pin_memory,
            "num_swapped": len(self.swapped),
            "swapped_bytes": self.swapped_bytes,
            "preemptions": self.stats.preemptions,
            "swaps_out": self.stats.swaps_out,
            "swaps_in": self.stats.swaps_in,
            "recomputes": self.stats.recomputes,
            "rejects": self.stats.rejects,
            "bytes_swapped": self.stats.bytes_swapped,
            "blocks_reclaimed": self.stats.blocks_reclaimed,
        }
