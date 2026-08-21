"""The fused paged-attention algorithm, tested where Triton cannot run.

``edgerag/cache/fused.py`` carries two implementations of one algorithm. The Triton kernel is the
one that will matter -- D19 measured the gather at 72.7% of the paged attention path -- and it is
also the one that cannot be executed on this project's development machine, because Triton ships
no Windows wheel.

So the reference implementation is treated as the specification and tested to the standard the
kernel will have to meet: **identical output to `gather` + SDPA**, swept across sequence lengths
either side of every block boundary. If the reference is right, porting it is a translation
problem. If it is not, no amount of T4 time will find that out cheaply.

``BUGS.md`` P-01 is the specific hazard. A sequence of 6,758 tokens in blocks of 16 leaves 10
unwritten slots in its final block holding the previous tenant's data. `gather` drops them
structurally with a `[:seq_len]` slice; a fused kernel has to mask, and an off-by-one there
attends to a dead token and produces fluent wrong text with nothing to announce it. Every test
below that sweeps a length is sweeping for that.
"""

from __future__ import annotations

import math

import pytest
import torch

from edgerag.cache.fused import (
    HAS_TRITON,
    fused_paged_attention,
    paged_attention_reference,
)

torch.manual_seed(0)

BLOCK_SIZE = 16
HEAD_DIM = 8


def _pool(n_kv_heads: int, num_blocks: int, dtype=torch.float32):
    shape = (n_kv_heads, num_blocks, BLOCK_SIZE, HEAD_DIM)
    return torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype)


def _sdpa_over_gather(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
    scaling: float,
    n_rep: int,
) -> torch.Tensor:
    """Exactly what the shipping path does: gather the blocks, slice, then SDPA.

    Written out here rather than imported so the comparison cannot silently become a comparison of
    the fused path against itself.
    """
    n_kv_heads = key_pool.shape[0]
    keys = key_pool[:, block_ids].reshape(n_kv_heads, -1, HEAD_DIM)[:, :seq_len].unsqueeze(0)
    values = value_pool[:, block_ids].reshape(n_kv_heads, -1, HEAD_DIM)[:, :seq_len].unsqueeze(0)
    if n_rep > 1:
        keys = keys.repeat_interleave(n_rep, dim=1)
        values = values.repeat_interleave(n_rep, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(query, keys, values, scale=scaling)
    return out.transpose(1, 2).contiguous()


# --- the algorithm against the path it replaces -------------------------------------------------


@pytest.mark.parametrize(
    "seq_len",
    # Either side of every block boundary the sweep can reach, plus 1 and a long tail. 16, 32 and
    # 64 are exactly-full blocks (no mask); 15, 17, 31, 33 straddle. P-01 lives at 17 and 33.
    [1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 100, 127, 128, 129],
)
def test_reference_matches_gather_plus_sdpa(seq_len: int) -> None:
    n_kv_heads, n_rep = 4, 1
    num_blocks = math.ceil(seq_len / BLOCK_SIZE) + 3  # allocate slack past the written region
    key_pool, value_pool = _pool(n_kv_heads, num_blocks)
    block_ids = torch.randperm(num_blocks)[: math.ceil(seq_len / BLOCK_SIZE)]
    query = torch.randn(1, n_kv_heads * n_rep, 1, HEAD_DIM)
    scaling = HEAD_DIM**-0.5

    expected = _sdpa_over_gather(
        query, key_pool, value_pool, block_ids, seq_len, scaling, n_rep
    )
    actual = paged_attention_reference(
        query, key_pool, value_pool, block_ids, seq_len, scaling, n_rep
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_the_unwritten_tail_of_the_final_block_is_never_attended() -> None:
    """P-01 stated as a property rather than hoped for.

    The slack slots are filled with a value large enough that attending to even one of them would
    dominate the softmax. If the mask is off by one, the output moves by orders of magnitude --
    which is a far better failure than the realistic version, where the slack holds plausible KV
    from the previous tenant and the answer is merely wrong.
    """
    n_kv_heads, seq_len = 2, 17  # one full block plus a single token: 15 slack slots
    key_pool, value_pool = _pool(n_kv_heads, 4)
    block_ids = torch.tensor([0, 1])
    query = torch.randn(1, n_kv_heads, 1, HEAD_DIM)

    clean = paged_attention_reference(
        query, key_pool, value_pool, block_ids, seq_len, HEAD_DIM**-0.5
    )
    # Poison every slot the sequence does not own.
    key_pool[:, 1, 1:] = 1e4
    value_pool[:, 1, 1:] = 1e4
    poisoned = paged_attention_reference(
        query, key_pool, value_pool, block_ids, seq_len, HEAD_DIM**-0.5
    )
    torch.testing.assert_close(poisoned, clean, rtol=1e-6, atol=1e-6)


def test_blocks_past_the_written_region_are_not_read() -> None:
    """A block table longer than the sequence must contribute nothing, not garbage."""
    n_kv_heads, seq_len = 2, 20
    key_pool, value_pool = _pool(n_kv_heads, 6)
    short = torch.tensor([0, 1])
    long = torch.tensor([0, 1, 2, 3])  # two extra allocated-but-unwritten blocks
    key_pool[:, 2:] = 1e4
    value_pool[:, 2:] = 1e4
    query = torch.randn(1, n_kv_heads, 1, HEAD_DIM)

    a = paged_attention_reference(query, key_pool, value_pool, short, seq_len, HEAD_DIM**-0.5)
    b = paged_attention_reference(query, key_pool, value_pool, long, seq_len, HEAD_DIM**-0.5)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_rep", [1, 2, 4])
def test_grouped_query_attention_maps_heads_correctly(n_rep: int) -> None:
    """This checkpoint is MHA (n_rep=1), so GQA is the untested-by-the-workload path.

    Getting ``kv_head = q_head // n_rep`` backwards produces output that is the right shape and
    entirely wrong, on a model nobody here runs -- exactly the kind of thing that survives until
    someone swaps the checkpoint.
    """
    n_kv_heads, seq_len = 3, 40
    key_pool, value_pool = _pool(n_kv_heads, 5)
    block_ids = torch.tensor([0, 1, 2])
    query = torch.randn(1, n_kv_heads * n_rep, 1, HEAD_DIM)
    scaling = HEAD_DIM**-0.5

    expected = _sdpa_over_gather(
        query, key_pool, value_pool, block_ids, seq_len, scaling, n_rep
    )
    actual = paged_attention_reference(
        query, key_pool, value_pool, block_ids, seq_len, scaling, n_rep
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_non_contiguous_block_tables_are_handled() -> None:
    """Blocks arrive in whatever order the allocator handed them out, not in ascending order."""
    n_kv_heads, seq_len = 2, 48
    key_pool, value_pool = _pool(n_kv_heads, 8)
    block_ids = torch.tensor([7, 2, 5])  # scattered, which is the normal case after churn
    query = torch.randn(1, n_kv_heads, 1, HEAD_DIM)
    scaling = HEAD_DIM**-0.5

    torch.testing.assert_close(
        paged_attention_reference(query, key_pool, value_pool, block_ids, seq_len, scaling),
        _sdpa_over_gather(query, key_pool, value_pool, block_ids, seq_len, scaling, 1),
        rtol=1e-5, atol=1e-5,
    )


# --- numerical behaviour the recurrence is supposed to have --------------------------------------


def test_the_online_softmax_survives_scores_that_would_overflow_a_naive_one() -> None:
    """The recurrence subtracts a running maximum, so large logits must not become inf/nan.

    A naive blockwise implementation that exponentiates before rescaling produces nan here, and
    nan propagates to fluent-looking garbage rather than to a crash.
    """
    n_kv_heads, seq_len = 2, 33
    key_pool, value_pool = _pool(n_kv_heads, 4)
    key_pool *= 200.0  # logits in the hundreds after scaling
    query = torch.randn(1, n_kv_heads, 1, HEAD_DIM) * 200.0
    block_ids = torch.tensor([0, 1, 2])

    out = paged_attention_reference(
        query, key_pool, value_pool, block_ids, seq_len, HEAD_DIM**-0.5
    )
    assert torch.isfinite(out).all(), "the running-maximum subtraction did not hold"


def test_an_empty_sequence_returns_zeros_rather_than_nan() -> None:
    """Dividing by an all-zero denominator is the obvious way to get nan out of this."""
    key_pool, value_pool = _pool(2, 2)
    out = paged_attention_reference(
        torch.randn(1, 2, 1, HEAD_DIM), key_pool, value_pool,
        torch.tensor([0]), 0, HEAD_DIM**-0.5,
    )
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) == 0


# --- the contract, and the guard rails around it -------------------------------------------------


def test_prefill_is_refused_rather_than_silently_wrong() -> None:
    """This is the decode path. Prefill is compute-bound and keeps SDPA (D3)."""
    key_pool, value_pool = _pool(2, 4)
    with pytest.raises(ValueError, match="decode path"):
        paged_attention_reference(
            torch.randn(1, 2, 7, HEAD_DIM), key_pool, value_pool,
            torch.tensor([0]), 16, HEAD_DIM**-0.5,
        )


def test_a_head_count_mismatch_is_refused() -> None:
    key_pool, value_pool = _pool(4, 4)
    with pytest.raises(ValueError, match="query heads"):
        paged_attention_reference(
            torch.randn(1, 6, 1, HEAD_DIM), key_pool, value_pool,
            torch.tensor([0]), 16, HEAD_DIM**-0.5, n_rep=1,
        )


def test_the_dispatcher_falls_back_instead_of_raising_without_triton() -> None:
    """The stack must run unchanged where there is no Triton -- i.e. every Windows box."""
    key_pool, value_pool = _pool(2, 4)
    query = torch.randn(1, 2, 1, HEAD_DIM)
    block_ids = torch.tensor([0, 1])

    fused = fused_paged_attention(query, key_pool, value_pool, block_ids, 20, HEAD_DIM**-0.5)
    reference = paged_attention_reference(
        query, key_pool, value_pool, block_ids, 20, HEAD_DIM**-0.5
    )
    torch.testing.assert_close(fused, reference)


def test_force_reference_bypasses_triton_even_when_it_exists() -> None:
    """The equivalence gate needs to compare the two on a box where both are available."""
    key_pool, value_pool = _pool(2, 4)
    query = torch.randn(1, 2, 1, HEAD_DIM)
    out = fused_paged_attention(
        query, key_pool, value_pool, torch.tensor([0, 1]), 20, HEAD_DIM**-0.5,
        force_reference=True,
    )
    assert out.shape == (1, 1, 2, HEAD_DIM)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is Linux-only; this runs on Colab")
@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 33, 128, 129])
def test_triton_kernel_matches_the_reference(seq_len: int) -> None:  # pragma: no cover
    """The gate that closes the loop. Skipped here, and the whole point of the Colab runner.

    Kept in the suite rather than only in ``scripts/colab_fused_attention.py`` so that anyone who
    runs pytest on a Linux box with Triton gets the check for free.
    """
    device = "cuda"
    n_kv_heads = 4
    num_blocks = math.ceil(seq_len / BLOCK_SIZE) + 2
    key_pool, value_pool = _pool(n_kv_heads, num_blocks, dtype=torch.float16)
    key_pool, value_pool = key_pool.to(device), value_pool.to(device)
    block_ids = torch.randperm(num_blocks, device=device)[: math.ceil(seq_len / BLOCK_SIZE)]
    query = torch.randn(1, n_kv_heads, 1, HEAD_DIM, dtype=torch.float16, device=device)
    scaling = HEAD_DIM**-0.5

    fused = fused_paged_attention(query, key_pool, value_pool, block_ids, seq_len, scaling)
    reference = paged_attention_reference(
        query, key_pool, value_pool, block_ids, seq_len, scaling
    )
    torch.testing.assert_close(fused, reference, rtol=2e-3, atol=2e-3)
