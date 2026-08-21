"""Fused paged attention: read KV straight out of the pool, never materialise the gather.

``CONTEXT.md`` D3 set a threshold -- if the gather ever exceeded 25% of the paged attention path,
write the fused kernel instead of arguing about it. D19 measured **72.7%** at the median request
length (23.5 ms of gather against 8.8 ms of attention per decode step) after a head-major pool
layout had already taken ~20% off it. That is ~3x the threshold, and it has been the largest
un-actioned performance lever in the project since Phase 3.

The mechanism is not subtle. ``PagedKVCache.gather`` reads the whole sequence's KV out of the pool
and *writes a copy*, then attention reads that copy -- so decode's memory traffic is roughly
doubled in exactly the regime that is already memory-bandwidth-bound (D14). Attention over an
already-contiguous cache costs +0.55% over paged indirection; the indirection is free and the copy
is the entire bill.

**Two implementations of one algorithm, and that is deliberate.**

:func:`paged_attention_reference` is plain PyTorch. It is not a fallback that happens to exist --
it is the executable specification. Triton does not install on Windows, so the kernel cannot be
run, let alone debugged, on this project's development machine; writing the Triton first and
testing it only on Colab is precisely ``BUGS.md`` B-05, which produced a complete, well-formed
file of zeros. The reference implements the identical block-walk and the identical online-softmax
recurrence, runs on CPU, and is tested against ``gather`` + SDPA across every block boundary. The
kernel then has one job -- translate a tested algorithm -- rather than two.

**The recurrence**, which is the only interesting part. Attention cannot be computed blockwise
naively because softmax needs a global maximum and a global sum. The standard trick (FlashAttention
/ online softmax) carries a running maximum ``m`` and running denominator ``l`` and rescales the
accumulator whenever the maximum moves::

    m_new = max(m, max(s_block))
    alpha = exp(m - m_new)                 # how much the old accumulator must shrink
    l     = l * alpha + sum(exp(s - m_new))
    acc   = acc * alpha + exp(s - m_new) @ v
    m     = m_new

Every partial result stays bounded, so this is numerically *better* than materialising the scores,
not merely equal to it.

**The partial final block is the whole risk surface**, and it is the same hazard as ``BUGS.md``
P-01: a sequence of 6,758 tokens in blocks of 16 leaves 10 unwritten slots in its last block, and
those slots hold whatever the previous tenant left. ``gather`` handles this structurally by
slicing ``[:seq_len]`` before attention sees anything. A fused kernel has no such slice -- it must
mask, and a mask that is off by one attends to a dead token and produces plausible wrong text
rather than a crash. Both implementations here take ``seq_len`` and mask against it, and the
tests sweep sequence lengths either side of every block boundary.

Nothing imports Triton at module load: ``HAS_TRITON`` is checked lazily so this module is
importable on Windows, in CI, and on a CPU-only box.
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = [
    "HAS_TRITON",
    "fused_paged_attention",
    "paged_attention_reference",
    "triton_unavailable_reason",
]


def _probe_triton() -> tuple[Any, str]:
    """Import Triton if it exists, and remember *why* if it does not.

    A bare ``HAS_TRITON = False`` sends someone hunting for a bug in their install when the answer
    is "Triton has no Windows wheel". The reason is carried so the benchmark can print it.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover -- exercised by not having Triton
        return None, f"{exc}. Triton ships no Windows wheel; this path is Colab/Linux only."
    return (triton, tl), ""


_TRITON, _TRITON_REASON = _probe_triton()
HAS_TRITON = _TRITON is not None


def triton_unavailable_reason() -> str:
    """Empty when Triton imported. Otherwise the reason, for printing rather than guessing."""
    return _TRITON_REASON


def paged_attention_reference(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
    scaling: float,
    n_rep: int = 1,
) -> torch.Tensor:
    """The executable specification: blockwise online softmax, straight out of the pool.

    Args:
        query: ``(1, n_q_heads, 1, head_dim)`` -- decode, exactly one query position.
        key_pool, value_pool: ``(n_kv_heads, num_blocks, block_size, head_dim)``, head-major
            (``CONTEXT.md`` P6 -- the layout that made ``pool[:, ids]`` a free view).
        block_ids: ``(n_blocks,)`` physical block ids, in logical order.
        seq_len: valid token count. Slots at or past this in the final block are **masked**, and
            getting that wrong is ``BUGS.md`` P-01 with no crash to announce it.
        scaling: ``1/sqrt(head_dim)``, passed in rather than derived so it cannot drift from the
            attention layer's own value.
        n_rep: query heads per KV head. 1 for MHA (this checkpoint), >1 under GQA.

    Returns ``(1, 1, n_q_heads, head_dim)`` -- the same layout ``sdpa_attention`` returns after its
    transpose, so this is a drop-in for the decode path.

    Accumulates in fp32 regardless of the pool's dtype. The recurrence multiplies the accumulator
    by ``alpha`` once per block; at ~425 blocks for a median request, fp16 rounding on that chain
    is a real error source and fp32 costs nothing here because the accumulator is one row.
    """
    if query.shape[2] != 1:
        raise ValueError(
            f"fused paged attention is the decode path: expected 1 query position, got "
            f"{query.shape[2]}. Prefill is compute-bound and keeps the SDPA path."
        )
    n_kv_heads, _, block_size, head_dim = key_pool.shape
    n_q_heads = query.shape[1]
    if n_q_heads != n_kv_heads * n_rep:
        raise ValueError(
            f"{n_q_heads} query heads against {n_kv_heads} kv heads at n_rep={n_rep}"
        )
    if seq_len <= 0:
        return torch.zeros(
            (1, 1, n_q_heads, head_dim), dtype=query.dtype, device=query.device
        )

    q = query[0, :, 0, :].to(torch.float32)                       # (n_q_heads, head_dim)
    running_max = torch.full((n_q_heads,), -float("inf"), device=q.device, dtype=torch.float32)
    denom = torch.zeros((n_q_heads,), device=q.device, dtype=torch.float32)
    acc = torch.zeros((n_q_heads, head_dim), device=q.device, dtype=torch.float32)

    for logical, block in enumerate(block_ids.tolist()):
        start = logical * block_size
        valid = min(block_size, seq_len - start)
        if valid <= 0:
            break  # blocks past the written region: allocated, never written, never attended

        keys = key_pool[:, block, :valid].to(torch.float32)       # (n_kv_heads, valid, head_dim)
        values = value_pool[:, block, :valid].to(torch.float32)
        if n_rep > 1:
            keys = keys.repeat_interleave(n_rep, dim=0)
            values = values.repeat_interleave(n_rep, dim=0)

        # (n_q_heads, valid): one query row against this block's keys.
        scores = torch.einsum("hd,hkd->hk", q, keys) * scaling

        block_max = scores.max(dim=-1).values
        new_max = torch.maximum(running_max, block_max)
        alpha = torch.exp(running_max - new_max)
        # exp(-inf - -inf) is nan, and the first block always hits it. The accumulator is zero
        # there anyway, so the rescale is a no-op and 0 is the right substitute.
        alpha = torch.nan_to_num(alpha, nan=0.0)

        probs = torch.exp(scores - new_max.unsqueeze(-1))
        denom = denom * alpha + probs.sum(dim=-1)
        acc = acc * alpha.unsqueeze(-1) + torch.einsum("hk,hkd->hd", probs, values)
        running_max = new_max

    out = acc / denom.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    return out.to(query.dtype).reshape(1, 1, n_q_heads, head_dim)


# --- the Triton kernel -------------------------------------------------------------------------
#
# Written to the contract the reference establishes and the tests pin. It is NOT exercised by the
# local suite, because Triton does not install here -- `scripts/colab_fused_attention.py` runs the
# equivalence gate as the first thing it does on a T4, before any timing, for that reason.

if HAS_TRITON:  # pragma: no cover -- no Triton on the development machine
    triton, tl = _TRITON

    @triton.jit
    def _paged_attn_decode_kernel(
        out_ptr, q_ptr, k_pool_ptr, v_pool_ptr, block_table_ptr,
        seq_len, n_blocks, scaling, n_rep,
        stride_qh, stride_qd,
        stride_kh, stride_kb, stride_ks, stride_kd,
        stride_oh, stride_od,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        """One program per query head; walks that sequence's blocks once, carrying the softmax.

        The loop reads each block's K and V exactly once and never writes them anywhere -- which
        is the entire point, since the gather's cost is the write, not the indirection.
        """
        head = tl.program_id(0)
        kv_head = head // n_rep

        dim = tl.arange(0, HEAD_DIM)
        slots = tl.arange(0, BLOCK_SIZE)

        q = tl.load(q_ptr + head * stride_qh + dim * stride_qd).to(tl.float32)

        running_max = float("-inf")
        denom = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for logical in range(0, n_blocks):
            block = tl.load(block_table_ptr + logical)
            start = logical * BLOCK_SIZE
            # The P-01 mask. Slots at or past seq_len were allocated but never written and hold
            # the previous tenant's data; attending to them is silent corruption, not a crash.
            valid = (start + slots) < seq_len

            base = kv_head * stride_kh + block * stride_kb
            offsets = base + slots[:, None] * stride_ks + dim[None, :] * stride_kd
            keys = tl.load(k_pool_ptr + offsets, mask=valid[:, None], other=0.0).to(tl.float32)
            values = tl.load(v_pool_ptr + offsets, mask=valid[:, None], other=0.0).to(tl.float32)

            scores = tl.sum(keys * q[None, :], axis=1) * scaling
            scores = tl.where(valid, scores, float("-inf"))

            new_max = tl.maximum(running_max, tl.max(scores, axis=0))
            alpha = tl.exp(running_max - new_max)
            # A block entirely past seq_len leaves new_max at -inf, making alpha nan. It cannot
            # happen while n_blocks is derived from seq_len, but a caller that over-reports would
            # otherwise poison the accumulator silently rather than contributing nothing.
            alpha = tl.where(new_max == float("-inf"), 0.0, alpha)

            probs = tl.exp(scores - new_max)
            probs = tl.where(valid, probs, 0.0)

            denom = denom * alpha + tl.sum(probs, axis=0)
            acc = acc * alpha + tl.sum(probs[:, None] * values, axis=0)
            running_max = new_max

        out = acc / tl.maximum(denom, 1e-38)
        tl.store(out_ptr + head * stride_oh + dim * stride_od, out.to(tl.float32))


def _next_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def fused_paged_attention(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
    scaling: float,
    n_rep: int = 1,
    force_reference: bool = False,
) -> torch.Tensor:
    """Decode attention over the paged pool, fused when Triton is available.

    Falls back to :func:`paged_attention_reference` rather than raising, so the serving stack runs
    unchanged on a machine without Triton -- which is every Windows machine, including the one this
    project is developed on. ``force_reference`` exists so the equivalence gate can compare the two
    against each other on hardware where both are available.
    """
    if force_reference or not HAS_TRITON or not query.is_cuda:
        return paged_attention_reference(
            query, key_pool, value_pool, block_ids, seq_len, scaling, n_rep
        )

    _, _, block_size, head_dim = key_pool.shape
    n_q_heads = query.shape[1]
    if seq_len <= 0:
        return torch.zeros((1, 1, n_q_heads, head_dim), dtype=query.dtype, device=query.device)

    # Only the blocks that actually hold written tokens. Passing the whole table would make every
    # request pay for its allocated-but-unwritten tail on every decode step.
    n_blocks = math.ceil(seq_len / block_size)
    table = block_ids[:n_blocks].to(torch.int32)

    q = query[0, :, 0, :].contiguous()
    out = torch.empty((n_q_heads, head_dim), dtype=torch.float32, device=query.device)

    _paged_attn_decode_kernel[(n_q_heads,)](
        out, q, key_pool, value_pool, table,
        seq_len, n_blocks, scaling, n_rep,
        q.stride(0), q.stride(1),
        key_pool.stride(0), key_pool.stride(1), key_pool.stride(2), key_pool.stride(3),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_DIM=_next_pow2(head_dim),
    )
    return out.to(query.dtype).reshape(1, 1, n_q_heads, head_dim)
