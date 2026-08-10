"""PHASE 3 GATE -- paged logits == naive logits == HF logits.

Run in **fp32** (``CONTEXT.md`` D13). The paged cache gathers blocks into a scratch buffer, which
changes GEMM shape relative to the naive cache, so the expected agreement is the ~1e-5
accumulation band measured in Phase 2 rather than zero. That band is known in advance, which is
what stops ``BUGS.md`` P-07 -- "is 1e-3 acceptable?" -- from costing an evening.

The sweep over ``seq_len x block_size`` is the whole point. Every off-by-one in a paged cache
lives at a block boundary, and lengths that are exact multiples of the block size are where both
directions of ``BUGS.md`` P-02 hide.
"""

from __future__ import annotations

import pytest
import torch

from edgerag.cache.allocator import BlockAllocator, OutOfBlocksError
from edgerag.cache.block_table import BlockTable
from edgerag.cache.naive import NaiveKVCache
from edgerag.cache.paged import PagedKVCache
from edgerag.core.loader import FIXTURE_MODEL
from edgerag.core.model import load_from_hf
from edgerag.core.spec import ModelSpec

# Same band as tests/test_equivalence.py CACHE_ATOL -- gather changes GEMM shape, not arithmetic.
PAGED_ATOL = 1e-4
PAGED_RTOL = 1e-4

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bundle():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.AutoConfig.from_pretrained(FIXTURE_MODEL)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        FIXTURE_MODEL, config=config, dtype=torch.float32
    )
    model.eval()
    spec = ModelSpec.from_hf_config(FIXTURE_MODEL, config)
    return model, spec, torch.device("cpu"), torch.float32


@pytest.fixture(scope="module")
def ours(bundle):
    model, spec, _, _ = bundle
    return load_from_hf(spec, model)


def _ids(spec: ModelSpec, seq: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seq * 31 + 7)
    return torch.randint(100, spec.vocab_size - 100, (1, seq), generator=g).to(device)


def _paged(spec, device, dtype, block_size: int, num_blocks: int = 64) -> PagedKVCache:
    return PagedKVCache(spec, BlockAllocator(num_blocks, block_size), device, dtype)


# --- the gate ---------------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 31, 32, 33, 64])
def test_paged_prefill_matches_naive(bundle, ours, block_size, seq_len) -> None:
    """The boundary sweep. Exact multiples of block_size are where P-02 hides in both directions."""
    _, spec, device, dtype = bundle
    ids = _ids(spec, seq_len, device)

    with torch.inference_mode():
        naive = NaiveKVCache(spec, 1, 128, device, dtype)
        expected = ours(input_ids=ids, cache=naive)

        paged = _paged(spec, device, dtype, block_size)
        actual = ours(input_ids=ids, cache=paged)

    assert paged.seq_len == naive.seq_len == seq_len
    torch.testing.assert_close(actual, expected, atol=PAGED_ATOL, rtol=PAGED_RTOL)


@pytest.mark.parametrize("block_size", [8, 16])
@pytest.mark.parametrize(("prefix", "total"), [(8, 16), (16, 20), (15, 33), (32, 40)])
def test_paged_decode_matches_naive(bundle, ours, block_size, prefix, total) -> None:
    """Step-by-step decode across block boundaries.

    ``(16, 20)`` with ``block_size=16`` is the important row: the prefill exactly fills a block, so
    the first decoded token must trigger a fresh allocation.
    """
    _, spec, device, dtype = bundle
    ids = _ids(spec, total, device)

    def run(cache):
        out = [ours(input_ids=ids[:, :prefix], cache=cache)]
        for pos in range(prefix, total):
            out.append(ours(input_ids=ids[:, pos : pos + 1], cache=cache))
        return torch.cat(out, dim=1)

    with torch.inference_mode():
        expected = run(NaiveKVCache(spec, 1, 128, device, dtype))
        paged = _paged(spec, device, dtype, block_size)
        actual = run(paged)

    assert paged.seq_len == total
    torch.testing.assert_close(actual, expected, atol=PAGED_ATOL, rtol=PAGED_RTOL)


def test_paged_matches_hf_end_to_end(bundle, ours) -> None:
    """Closes the chain: paged == naive == HuggingFace."""
    model, spec, device, dtype = bundle
    ids = _ids(spec, 24, device)

    with torch.inference_mode():
        expected = model.model.text_model(input_ids=ids).last_hidden_state @ model.lm_head.weight.T
        actual = ours(input_ids=ids, cache=_paged(spec, device, dtype, 16))

    torch.testing.assert_close(actual, expected, atol=PAGED_ATOL, rtol=PAGED_RTOL)


def test_block_size_one_is_degenerate_but_correct(bundle, ours) -> None:
    """Every token its own block -- zero internal fragmentation, maximum table overhead.

    Worth pinning because it exercises the block-boundary path on every single token.
    """
    _, spec, device, dtype = bundle
    ids = _ids(spec, 12, device)

    with torch.inference_mode():
        expected = ours(input_ids=ids, cache=NaiveKVCache(spec, 1, 32, device, dtype))
        paged = _paged(spec, device, dtype, block_size=1, num_blocks=32)
        actual = ours(input_ids=ids, cache=paged)

    assert len(paged.table.blocks) == 12
    torch.testing.assert_close(actual, expected, atol=PAGED_ATOL, rtol=PAGED_RTOL)


# --- P-01: the unwritten tail must never reach attention -----------------------------------------


def test_gather_excludes_the_unwritten_tail(bundle) -> None:
    """BUGS.md P-01, asserted structurally.

    The final block is poisoned with a recognisable value. If ``gather`` returned the whole block
    the poison would appear in the output; slicing to ``seq_len`` means it cannot.
    """
    _, spec, device, dtype = bundle
    cache = _paged(spec, device, dtype, block_size=16)

    k = torch.ones(1, spec.n_kv_heads, 17, spec.head_dim, device=device, dtype=dtype)
    for layer in range(spec.n_layers):
        cache.update(k, k, layer)

    # Two blocks are held (17 tokens), so slots 17..31 of block 1 are unwritten. Poison them.
    poison = 12345.0
    second = cache.table.blocks[1]
    cache.key_pool[0][second, 1:] = poison

    keys, _ = cache.gather(0)
    assert keys.shape[2] == 17
    assert not (keys == poison).any(), "unwritten block slots leaked into attention (P-01)"


def test_locate_refuses_positions_past_the_written_region(bundle) -> None:
    table = BlockTable(allocator=BlockAllocator(8, 16))
    table.append(17)

    table.locate(16)  # last written position is fine
    for bad in (17, 31, 100):
        with pytest.raises(IndexError, match="outside the written region"):
            table.locate(bad)


# --- block table arithmetic (no model needed) ----------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "block_size", "expected_blocks"),
    [(1, 16, 1), (16, 16, 1), (17, 16, 2), (32, 16, 2), (33, 16, 3), (8, 8, 1), (9, 8, 2)],
)
def test_table_holds_exactly_the_blocks_needed(tokens, block_size, expected_blocks) -> None:
    """No spare block at exact multiples -- a spare leaks one block per sequence forever."""
    table = BlockTable(allocator=BlockAllocator(16, block_size))
    table.append(tokens)
    assert len(table.blocks) == expected_blocks
    assert table.num_tokens == tokens


def test_appending_one_token_past_a_full_block_allocates_exactly_one() -> None:
    allocator = BlockAllocator(8, 16)
    table = BlockTable(allocator=allocator)
    table.append(16)
    assert len(table.blocks) == 1

    fresh = table.append(1)
    assert len(fresh) == 1
    assert len(table.blocks) == 2


def test_slack_reports_internal_fragmentation() -> None:
    table = BlockTable(allocator=BlockAllocator(8, 16))
    table.append(17)
    assert table.capacity == 32
    assert table.slack == 15


def test_failed_growth_leaves_the_table_untouched() -> None:
    """An OOM must not advance num_tokens past storage the sequence does not have."""
    table = BlockTable(allocator=BlockAllocator(2, 16))
    table.append(32)  # consumes the whole pool
    with pytest.raises(OutOfBlocksError):
        table.append(1)
    assert table.num_tokens == 32
    assert len(table.blocks) == 2


def test_free_returns_blocks_and_resets() -> None:
    allocator = BlockAllocator(8, 16)
    table = BlockTable(allocator=allocator)
    table.append(40)
    assert allocator.num_used == 3

    assert table.free() == 3
    assert allocator.num_free == 8
    assert table.num_tokens == 0
    allocator.check_invariants()


# --- fork / copy-on-write bookkeeping ------------------------------------------------------------


def test_fork_shares_blocks_without_copying() -> None:
    allocator = BlockAllocator(16, 16)
    parent = BlockTable(allocator=allocator)
    parent.append(32)
    used_before = allocator.num_used

    child = parent.fork()
    assert child.blocks == parent.blocks
    assert allocator.num_used == used_before, "fork allocated -- it must only incref"
    assert all(allocator.refcount(b) == 2 for b in parent.blocks)
    allocator.check_invariants()


def test_writing_to_a_forked_block_splits_it() -> None:
    allocator = BlockAllocator(16, 16)
    parent = BlockTable(allocator=allocator)
    parent.append(16)
    child = parent.fork()

    block, _, copied = child.writable_block_for(0)
    assert copied is not None, "write to a shared block did not copy"
    old, new = copied
    assert block == new != old
    assert parent.blocks[0] == old, "the parent must keep the original"
    assert allocator.refcount(old) == 1
    allocator.check_invariants()


def test_writing_to_an_unshared_block_does_not_copy() -> None:
    allocator = BlockAllocator(16, 16)
    table = BlockTable(allocator=allocator)
    table.append(16)

    _, _, copied = table.writable_block_for(0)
    assert copied is None
    assert allocator.stats.copy_on_writes == 0


def test_freeing_a_fork_leaves_the_parent_intact() -> None:
    allocator = BlockAllocator(16, 16)
    parent = BlockTable(allocator=allocator)
    parent.append(32)
    child = parent.fork()

    assert child.free() == 0  # shared, so nothing reclaimed
    assert all(allocator.refcount(b) == 1 for b in parent.blocks)
    assert parent.free() == 2
    assert allocator.num_free == 16
    allocator.check_invariants()


# --- memory accounting: the argument for paging ---------------------------------------------------


def test_paged_reserves_far_less_than_naive_for_a_short_sequence(bundle) -> None:
    """The headline comparison, quantified.

    Naive reserves ``max_seq_len`` per sequence regardless of use; paged reserves only the blocks
    a sequence actually holds. At 192 KiB/token on the headline model this is the difference
    between fitting one request and fitting several.
    """
    _, spec, device, dtype = bundle
    naive = NaiveKVCache(spec, 1, 2048, device, dtype)
    paged = _paged(spec, device, dtype, block_size=16, num_blocks=256)

    k = torch.zeros(1, spec.n_kv_heads, 300, spec.head_dim, device=device, dtype=dtype)
    for layer in range(spec.n_layers):
        naive.update(k, k, layer)
        paged.update(k, k, layer)

    assert paged.nbytes < naive.nbytes / 6
    assert paged.seq_len == naive.seq_len == 300


def test_fork_makes_shared_prefixes_nearly_free(bundle) -> None:
    """20 queries against one retrieved document, which the frozen trace actually contains."""
    _, spec, device, dtype = bundle
    allocator = BlockAllocator(512, 16)
    base = PagedKVCache(spec, allocator, device, dtype)

    k = torch.zeros(1, spec.n_kv_heads, 320, spec.head_dim, device=device, dtype=dtype)
    for layer in range(spec.n_layers):
        base.update(k, k, layer)

    blocks_for_one = allocator.num_used
    forks = [base.fork() for _ in range(19)]

    assert allocator.num_used == blocks_for_one, "sharing 20 sequences allocated new blocks"
    assert len(forks) == 19
    allocator.check_invariants()
