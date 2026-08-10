"""Phase 3c/3d: copy-on-write correctness, prefix sharing, and preemption.

The CoW tests operate on real tensors, not just refcounts. ``tests/test_paged.py`` already covers
the bookkeeping, and the bookkeeping was correct while the data path was not -- see ``BUGS.md``
B-03. Asserting on refcounts alone is what let that through.
"""

from __future__ import annotations

import pytest
import torch

from edgerag.cache.allocator import BlockAllocator, OutOfBlocksError
from edgerag.cache.paged import PagedKVCache
from edgerag.cache.preemption import PreemptionPolicy, Preemptor
from edgerag.cache.prefix import ROOT_HASH, PrefixCache, chain_hash
from edgerag.core.spec import ModelSpec

BLOCK = 8


def tiny_spec(n_layers: int = 2) -> ModelSpec:
    return ModelSpec(
        model_id="tiny",
        model_type="test",
        n_layers=n_layers,
        hidden_size=32,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=8,
        vocab_size=100,
        max_position_embeddings=128,
        rope_theta=10000.0,
        vision_layers=1,
        vision_hidden=8,
        vision_image_size=16,
        vision_patch_size=8,
        scale_factor=1,
        image_token_id=99,
        pad_token_id=0,
        rms_norm_eps=1e-5,
        intermediate_size=64,
        hidden_act="silu",
    )


def make_cache(allocator: BlockAllocator, spec: ModelSpec | None = None) -> PagedKVCache:
    spec = spec or tiny_spec()
    return PagedKVCache(spec, allocator, torch.device("cpu"), torch.float32)


def fill(cache: PagedKVCache, n_tokens: int, value: float) -> None:
    spec = cache.spec
    kv = torch.full((1, spec.n_kv_heads, n_tokens, spec.head_dim), value)
    for layer in range(spec.n_layers):
        cache.update(kv, kv, layer)


# --- BUGS.md B-03: fork-then-append must not corrupt the sibling -----------------------------


def test_sibling_writes_do_not_corrupt_each_other() -> None:
    """The bug B-03 records, asserted on tensor contents rather than refcounts.

    Both sequences append into what was a shared partially-filled block. Whichever writes second
    would otherwise overwrite the first's KV, with no exception and no NaN -- the request simply
    answers using another request's context.
    """
    allocator = BlockAllocator(16, BLOCK)
    parent = make_cache(allocator)
    fill(parent, 10, 1.0)  # 2 blocks; block 1 holds 2 tokens and has 6 free slots

    child = parent.fork()
    child_before = child.gather(0)[0].clone()

    fill(child, 3, 7.0)  # writes into the shared tail block
    parent_after_child = parent.gather(0)[0]
    assert torch.equal(parent_after_child, child_before[:, :10]), "child corrupted the parent"

    fill(parent, 3, -5.0)  # parent writes the same physical slots
    child_after = child.gather(0)[0]

    assert not (child_after == -5.0).any(), "parent's write leaked into the child (B-03)"
    assert (child_after[:, 10:13] == 7.0).all(), "child lost its own tokens"
    allocator.check_invariants()


def test_copy_on_write_preserves_the_shared_prefix_contents() -> None:
    """The content copy, not just the mapping.

    ``unshare`` allocates a fresh block and repoints the table. Omitting the data copy leaves a
    valid-looking mapping to a block of zeros -- degraded answers, no crash.
    """
    allocator = BlockAllocator(16, BLOCK)
    parent = make_cache(allocator)
    fill(parent, 10, 3.0)

    child = parent.fork()
    fill(child, 1, 9.0)

    keys, _ = child.gather(0)
    assert (keys[0, :, :10] == 3.0).all(), "shared prefix lost in the copy"
    assert (keys[0, :, 10] == 9.0).all()
    assert allocator.stats.copy_on_writes == 1


def test_copy_on_write_fires_once_not_per_layer() -> None:
    """Splitting per layer would leave layers disagreeing about which block holds the sequence."""
    allocator = BlockAllocator(16, BLOCK)
    parent = make_cache(allocator, tiny_spec(n_layers=6))
    fill(parent, 10, 1.0)

    child = parent.fork()
    fill(child, 2, 2.0)
    assert allocator.stats.copy_on_writes == 1


def test_full_block_boundary_needs_no_copy() -> None:
    """A fork whose length is an exact multiple has no partial tail to split.

    The next append allocates a fresh private block, so CoW must not fire.
    """
    allocator = BlockAllocator(16, BLOCK)
    parent = make_cache(allocator)
    fill(parent, BLOCK, 1.0)

    child = parent.fork()
    fill(child, 1, 5.0)
    assert allocator.stats.copy_on_writes == 0
    assert (child.gather(0)[0][0, :, :BLOCK] == 1.0).all()


def test_forks_are_independent_after_divergence() -> None:
    allocator = BlockAllocator(32, BLOCK)
    base = make_cache(allocator)
    fill(base, 10, 1.0)

    children = [base.fork() for _ in range(3)]
    for i, child in enumerate(children):
        fill(child, 2, float(i + 10))

    for i, child in enumerate(children):
        keys = child.gather(0)[0]
        assert (keys[0, :, :10] == 1.0).all()
        assert (keys[0, :, 10:12] == float(i + 10)).all()
    allocator.check_invariants()


# --- prefix cache -----------------------------------------------------------------------------


def test_chain_hash_is_position_sensitive() -> None:
    """Independent per-block hashes would let a block match at the wrong position.

    Reused keys carry RoPE for the positions they were computed at, so a mid-sequence match would
    attend with wrong relative positions.
    """
    a = chain_hash(ROOT_HASH, (1, 2, 3))
    b = chain_hash(ROOT_HASH, (9, 9, 9))
    assert chain_hash(a, (4, 5, 6)) != chain_hash(b, (4, 5, 6))


def test_lookup_misses_on_an_empty_cache() -> None:
    cache = PrefixCache(BlockAllocator(16, BLOCK))
    assert cache.lookup(list(range(32))) == ([], 0)
    assert cache.stats.hit_rate == 0.0


def test_identical_prefix_is_reused() -> None:
    allocator = BlockAllocator(16, BLOCK)
    cache = PrefixCache(allocator)
    tokens = list(range(24))  # 3 full blocks
    blocks = allocator.allocate(3)

    assert cache.register(tokens, blocks) == 3
    matched, n_tokens = cache.lookup(tokens)
    assert matched == blocks
    assert n_tokens == 24
    allocator.check_invariants()


def test_partial_prefix_matches_only_the_common_run() -> None:
    """Matching stops at the first divergent block -- later blocks cannot be used without it."""
    allocator = BlockAllocator(32, BLOCK)
    cache = PrefixCache(allocator)
    original = list(range(24))
    cache.register(original, allocator.allocate(3))

    divergent = list(range(16)) + [999] * 8  # first two blocks shared, third differs
    matched, n_tokens = cache.lookup(divergent)
    assert len(matched) == 2
    assert n_tokens == 16


def test_partial_trailing_block_is_never_cached() -> None:
    """It will be appended to, so sharing it triggers CoW immediately and gains nothing."""
    allocator = BlockAllocator(16, BLOCK)
    cache = PrefixCache(allocator)
    tokens = list(range(20))  # 2 full blocks + 4 leftover tokens
    assert len(cache.block_hashes(tokens)) == 2
    assert cache.register(tokens, allocator.allocate(3)) == 2


def test_lookup_increfs_so_the_caller_owns_what_it_gets() -> None:
    """Returning borrowed blocks would be a use-after-free waiting for an eviction."""
    allocator = BlockAllocator(16, BLOCK)
    cache = PrefixCache(allocator)
    tokens = list(range(16))
    blocks = allocator.allocate(2)
    cache.register(tokens, blocks)

    before = [allocator.refcount(b) for b in blocks]
    matched, _ = cache.lookup(tokens)
    after = [allocator.refcount(b) for b in matched]
    assert all(a == b + 1 for a, b in zip(after, before, strict=True))


def test_cached_blocks_survive_the_owning_sequence() -> None:
    """The cache holds its own reference, so a block cannot be recycled under a stale entry."""
    allocator = BlockAllocator(16, BLOCK)
    cache = PrefixCache(allocator)
    tokens = list(range(16))
    blocks = allocator.allocate(2)
    cache.register(tokens, blocks)

    allocator.free(blocks)  # the original sequence finishes
    assert all(allocator.refcount(b) == 1 for b in blocks), "cache lost its reference"

    matched, _ = cache.lookup(tokens)
    assert matched == blocks
    allocator.check_invariants()


def test_lru_eviction_bounds_pinned_blocks() -> None:
    """Cached blocks are pinned; unbounded caching would starve the pool it serves."""
    allocator = BlockAllocator(32, BLOCK)
    cache = PrefixCache(allocator, max_blocks=2)

    cache.register(list(range(8)), allocator.allocate(1))
    cache.register(list(range(100, 108)), allocator.allocate(1))
    assert cache.num_cached == 2

    cache.register(list(range(200, 208)), allocator.allocate(1))
    assert cache.num_cached <= 2
    assert cache.stats.evictions >= 1
    allocator.check_invariants()


def test_clear_releases_every_reference() -> None:
    allocator = BlockAllocator(16, BLOCK)
    cache = PrefixCache(allocator)
    blocks = allocator.allocate(2)
    cache.register(list(range(16)), blocks)

    cache.clear()
    allocator.free(blocks)
    assert allocator.num_free == 16
    allocator.check_invariants()


# --- preemption -------------------------------------------------------------------------------


def test_swap_out_preserves_kv_and_frees_blocks() -> None:
    allocator = BlockAllocator(16, BLOCK)
    cache = make_cache(allocator)
    fill(cache, 16, 4.0)
    blocks = list(cache.table.blocks)

    preemptor = Preemptor(allocator, PreemptionPolicy.SWAP, pin_memory=False)
    record = preemptor.swap_out("r1", blocks, 16, cache.key_pool, cache.value_pool)

    assert allocator.num_free == 16, "victim blocks were not reclaimed"
    assert record.num_tokens == 16
    assert (record.keys[0] == 4.0).all()
    allocator.check_invariants()


def test_swap_round_trip_restores_exact_contents() -> None:
    allocator = BlockAllocator(16, BLOCK)
    cache = make_cache(allocator)
    fill(cache, 16, 4.0)
    before = cache.gather(0)[0].clone()
    blocks = list(cache.table.blocks)

    preemptor = Preemptor(allocator, PreemptionPolicy.SWAP, pin_memory=False)
    preemptor.swap_out("r1", blocks, 16, cache.key_pool, cache.value_pool)

    # Something else uses the pool in the meantime, so restoration cannot rely on luck.
    other = allocator.allocate(2)
    cache.key_pool[0][other] = -99.0
    allocator.free(other)

    new_blocks, num_tokens = preemptor.swap_in("r1", cache.key_pool, cache.value_pool)
    cache.table.blocks = new_blocks
    cache.table.num_tokens = num_tokens

    torch.testing.assert_close(cache.gather(0)[0], before)
    assert preemptor.stats.swaps_in == 1


def test_swap_copies_before_freeing() -> None:
    """Freeing first would let another sequence claim and overwrite the blocks mid-copy."""
    allocator = BlockAllocator(8, BLOCK)
    cache = make_cache(allocator)
    fill(cache, 8, 2.0)

    preemptor = Preemptor(allocator, PreemptionPolicy.SWAP, pin_memory=False)
    record = preemptor.swap_out("r1", list(cache.table.blocks), 8, cache.key_pool, cache.value_pool)
    assert (record.keys[0] == 2.0).all(), "data was lost -- blocks freed before the copy"


def test_swap_in_can_fail_when_the_pool_is_full() -> None:
    """Swapping back in is itself an admission decision, not a guaranteed operation."""
    allocator = BlockAllocator(4, BLOCK)
    cache = make_cache(allocator)
    fill(cache, 16, 1.0)

    preemptor = Preemptor(allocator, PreemptionPolicy.SWAP, pin_memory=False)
    preemptor.swap_out("r1", list(cache.table.blocks), 16, cache.key_pool, cache.value_pool)
    allocator.allocate(4)  # someone else took the pool

    with pytest.raises(OutOfBlocksError):
        preemptor.swap_in("r1", cache.key_pool, cache.value_pool)


def test_recompute_policy_frees_without_copying() -> None:
    allocator = BlockAllocator(16, BLOCK)
    cache = make_cache(allocator)
    fill(cache, 16, 1.0)

    preemptor = Preemptor(allocator, PreemptionPolicy.RECOMPUTE, pin_memory=False)
    preemptor.preempt("r1", list(cache.table.blocks), 16)

    assert allocator.num_free == 16
    assert preemptor.stats.recomputes == 1
    assert preemptor.swapped == {}


def test_victims_are_selected_newest_first() -> None:
    """Oldest-first would repeatedly reset the longest-waiting request -- livelock."""
    candidates = [("old", 2), ("mid", 2), ("new", 2)]
    assert Preemptor.select_victims(candidates, blocks_needed=2) == ["new"]
    assert Preemptor.select_victims(candidates, blocks_needed=3) == ["new", "mid"]


def test_victim_selection_stops_once_the_need_is_met() -> None:
    candidates = [("a", 5), ("b", 5), ("c", 5)]
    assert Preemptor.select_victims(candidates, blocks_needed=1) == ["c"]


def test_unknown_swap_in_is_rejected() -> None:
    preemptor = Preemptor(BlockAllocator(8, BLOCK), PreemptionPolicy.SWAP, pin_memory=False)
    with pytest.raises(KeyError):
        preemptor.swap_in("nope", [], [])
