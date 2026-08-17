"""Phase 4: visual token pruning.

The selection logic is pure tensor arithmetic and is tested without a model, in milliseconds --
the same discipline as the allocator. The end-to-end tests then confirm that pruning actually
changes what the decoder stores, and that ``keep_ratio=1.0`` is an *exact* no-op rather than an
approximation, because the whole ablation table depends on that row being the true baseline.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.compressed import CompressedKVCache
from edgerag.compress.fastv import (
    FastVCompressor,
    FastVConfig,
    build_visual_mask,
    select_kept_indices,
    uniform_stride_indices,
    visual_token_scores,
)
from tests.test_cow import tiny_spec

# --- config -----------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_invalid_keep_ratio_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="keep_ratio"):
        FastVConfig(keep_ratio=bad)


def test_keep_ratio_one_is_disabled() -> None:
    assert FastVConfig(keep_ratio=1.0).enabled is False
    assert FastVConfig(keep_ratio=0.99).enabled is True


# --- scoring ----------------------------------------------------------------------------------


def _weights(seq: int, heads: int = 2) -> torch.Tensor:
    """Causal attention weights: row q distributes 1.0 over keys 0..q."""
    w = torch.zeros(1, heads, seq, seq)
    for q in range(seq):
        w[:, :, q, : q + 1] = 1.0 / (q + 1)
    return w


def test_scores_are_minus_inf_for_non_visual_tokens() -> None:
    """So a top-k over all positions can never select a text token."""
    mask = torch.tensor([[False, True, True, False]])
    scores = visual_token_scores(_weights(4), mask)
    assert torch.isneginf(scores[0, 0])
    assert torch.isneginf(scores[0, 3])
    assert torch.isfinite(scores[0, 1:3]).all()


def test_mean_mode_normalises_away_the_query_count_bias() -> None:
    """Token j is visible to only S-j queries, so a plain column mean penalises late tokens.

    Constructed so every key receives an identical amount *per attending query*: the normalised
    score must then be flat, while the un-normalised mean decays purely with position. In a RAG
    prompt the late tokens are the most recently retrieved document, so that artifact would prune
    exactly the wrong pages.
    """
    seq, heads = 8, 2
    w = torch.zeros(1, heads, seq, seq)
    for q in range(seq):
        w[:, :, q, : q + 1] = 0.1  # constant per (query, visible key)

    mask = torch.ones(1, seq, dtype=torch.bool)
    scores = visual_token_scores(w, mask, mode="mean")[0]
    torch.testing.assert_close(scores, torch.full((seq,), 0.1), atol=1e-6, rtol=1e-6)

    naive = w.float().sum(dim=1).sum(dim=1)[0] / seq
    assert naive[0] > naive[-1] * 4, "the artifact this normalisation removes should be large"


def test_last_row_mode_scores_by_the_final_token_only() -> None:
    """FastV's choice: decode queries resemble the last prompt token more than the average one."""
    seq = 6
    w = torch.zeros(1, 1, seq, seq)
    for q in range(seq):
        w[0, 0, q, : q + 1] = 0.01
        if q >= 1:
            w[0, 0, q, 1] = 0.5  # token 1: broad, steady support from every query
    w[0, 0, -1, 4] = 0.6  # token 4: ignored until the final query, which wants it badly

    mask = torch.ones(1, seq, dtype=torch.bool)
    # The two modes answer different questions and here they disagree, which is the point:
    # "what did the prompt attend to" is not "what will generation need".
    assert int(visual_token_scores(w, mask, mode="mean")[0].argmax()) == 1
    assert int(visual_token_scores(w, mask, mode="last_row")[0].argmax()) == 4


def test_scores_reward_tokens_that_receive_attention() -> None:
    seq = 6
    w = torch.full((1, 1, seq, seq), 0.0)
    for q in range(seq):
        w[0, 0, q, : q + 1] = 0.01
        w[0, 0, q, 2] = 1.0  # token 2 is heavily attended
    mask = torch.ones(1, seq, dtype=torch.bool)
    for mode in ("mean", "last_row"):
        assert int(visual_token_scores(w, mask, mode=mode)[0].argmax()) == 2


def test_unknown_scoring_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scoring mode"):
        visual_token_scores(_weights(4), torch.ones(1, 4, dtype=torch.bool), mode="vibes")


def test_rejects_wrongly_shaped_weights() -> None:
    with pytest.raises(ValueError, match="batch, heads, queries, keys"):
        visual_token_scores(torch.zeros(2, 3, 4), torch.ones(1, 4, dtype=torch.bool))


# --- selection --------------------------------------------------------------------------------


def _mask(pattern: str) -> torch.Tensor:
    """'ttvvvvtt' -> text/visual layout."""
    return torch.tensor([c == "v" for c in pattern])


def test_text_tokens_are_never_pruned() -> None:
    """They are 25% of prefill and carry the question."""
    mask = _mask("ttvvvvvvtt")
    scores = torch.where(mask, torch.rand(10), torch.tensor(float("-inf")))
    kept = select_kept_indices(scores, mask, FastVConfig(keep_ratio=0.25))
    text_positions = torch.nonzero(~mask).flatten()
    assert set(text_positions.tolist()).issubset(set(kept.tolist()))


def test_last_position_is_always_kept() -> None:
    """Mechanical, not heuristic: generation reads logits from it."""
    mask = _mask("tvvvvvvvvv")  # prompt ends on an image token
    scores = torch.where(mask, torch.zeros(10), torch.tensor(float("-inf")))
    scores[-1] = -1e9  # worst possible score
    kept = select_kept_indices(scores, mask, FastVConfig(keep_ratio=0.25))
    assert 9 in kept.tolist()


def test_keep_ratio_one_keeps_everything_exactly() -> None:
    """The ablation baseline row must be a true no-op."""
    mask = _mask("ttvvvvvvtt")
    scores = torch.where(mask, torch.rand(10), torch.tensor(float("-inf")))
    kept = select_kept_indices(scores, mask, FastVConfig(keep_ratio=1.0))
    assert kept.tolist() == list(range(10))


def test_indices_are_ascending_so_causality_survives() -> None:
    """A standard causal mask over the survivors is only valid if their order is preserved."""
    mask = _mask("tvvvvvvvvt")
    scores = torch.where(mask, torch.rand(10), torch.tensor(float("-inf")))
    kept = select_kept_indices(scores, mask, FastVConfig(keep_ratio=0.5))
    assert kept.tolist() == sorted(kept.tolist())


@pytest.mark.parametrize("ratio", [0.125, 0.25, 0.5, 0.75])
def test_visual_survivors_track_the_ratio(ratio: float) -> None:
    mask = _mask("tt" + "v" * 16 + "tt")
    scores = torch.where(mask, torch.rand(20), torch.tensor(float("-inf")))
    kept = select_kept_indices(scores, mask, FastVConfig(keep_ratio=ratio, protect_last_n=0))

    kept_visual = sum(1 for i in kept.tolist() if mask[i])
    assert kept_visual == pytest.approx(16 * ratio, abs=1)


def test_at_least_one_visual_token_survives() -> None:
    """An extreme ratio must not strip the image entirely -- that is a different experiment."""
    mask = _mask("t" + "v" * 4 + "t")
    scores = torch.where(mask, torch.rand(6), torch.tensor(float("-inf")))
    kept = select_kept_indices(scores, mask, FastVConfig(keep_ratio=0.01, protect_last_n=0))
    assert any(mask[i] for i in kept.tolist())


def test_prompt_without_images_is_untouched() -> None:
    mask = _mask("tttttt")
    scores = torch.full((6,), float("-inf"))
    kept = select_kept_indices(scores, mask, FastVConfig(keep_ratio=0.25))
    assert kept.tolist() == list(range(6))


# --- uniform-stride control --------------------------------------------------------------------


def test_uniform_stride_spreads_survivors_evenly() -> None:
    """The control that keeps the published curve honest.

    If attention scoring does not beat evenly-spaced selection, the finding is "visual tokens are
    redundant", not "FastV works".
    """
    mask = _mask("t" + "v" * 16 + "t")
    kept = uniform_stride_indices(mask, FastVConfig(keep_ratio=0.25, protect_last_n=0))
    visual_kept = [i for i in kept.tolist() if mask[i]]
    assert len(visual_kept) == 4
    gaps = [b - a for a, b in itertools.pairwise(visual_kept)]
    assert max(gaps) - min(gaps) <= 1, f"survivors not evenly spaced: {visual_kept}"


def test_uniform_stride_also_keeps_all_text() -> None:
    mask = _mask("ttvvvvvvtt")
    kept = uniform_stride_indices(mask, FastVConfig(keep_ratio=0.5))
    assert set(torch.nonzero(~mask).flatten().tolist()).issubset(set(kept.tolist()))


# --- visual mask ---------------------------------------------------------------------------------


def test_build_visual_mask_finds_image_tokens() -> None:
    ids = torch.tensor([[1, 99, 99, 5]])
    assert build_visual_mask(ids, 99).tolist() == [[False, True, True, False]]


# --- compressed cache ----------------------------------------------------------------------------


def test_cache_routes_layers_by_the_cut() -> None:
    spec = tiny_spec(n_layers=6)
    cache = CompressedKVCache(
        spec, BlockAllocator(64, 8), torch.device("cpu"), torch.float32, score_layer=2
    )
    assert cache._route(0) is cache.full
    assert cache._route(1) is cache.full
    assert cache._route(2) is cache.pruned
    assert cache._route(5) is cache.pruned


def test_cache_halves_track_different_lengths() -> None:
    spec = tiny_spec(n_layers=6)
    cache = CompressedKVCache(
        spec, BlockAllocator(64, 8), torch.device("cpu"), torch.float32, score_layer=2
    )

    full_kv = torch.ones(1, spec.n_kv_heads, 16, spec.head_dim)
    pruned_kv = torch.ones(1, spec.n_kv_heads, 6, spec.head_dim)
    for layer in range(2):
        cache.update(full_kv, full_kv, layer)
    for layer in range(2, 6):
        cache.update(pruned_kv, pruned_kv, layer)

    assert cache.full.seq_len == 16
    assert cache.pruned.seq_len == 6
    assert cache.seq_len == 16, "position ids must follow the FULL length, not the pruned one"


def test_savings_counts_only_layers_above_the_cut() -> None:
    """The token ratio understates the win: most of the stack sits above the cut."""
    spec = tiny_spec(n_layers=10)
    cache = CompressedKVCache(
        spec, BlockAllocator(128, 8), torch.device("cpu"), torch.float32, score_layer=2
    )
    full_kv = torch.ones(1, spec.n_kv_heads, 16, spec.head_dim)
    pruned_kv = torch.ones(1, spec.n_kv_heads, 4, spec.head_dim)
    for layer in range(2):
        cache.update(full_kv, full_kv, layer)
    for layer in range(2, 10):
        cache.update(pruned_kv, pruned_kv, layer)

    saving = cache.savings()
    assert saving["dropped_tokens"] == 12
    assert saving["layers_above_cut"] == 8
    # 12 dropped tokens over 8 of 10 layers = 60% of total KV.
    assert saving["saving_fraction"] == pytest.approx(0.6)


def test_score_layer_outside_the_stack_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        CompressedKVCache(
            tiny_spec(n_layers=4), BlockAllocator(16, 8), torch.device("cpu"),
            torch.float32, score_layer=9,
        )


def test_both_halves_share_one_allocator_not_one_pool() -> None:
    """Freed blocks from either half must be reusable by the other -- that needs a shared
    *allocator*, not a shared pool.

    The pools are deliberately separate and right-sized: each half stores only the layers it
    serves. Sharing one full-stack pool made the `full` cache reserve slots in all L layers while
    writing only `score_layer` of them, and doubled the block demand at keep_ratio=1.0.
    """
    spec = tiny_spec(n_layers=4)
    cache = CompressedKVCache(
        spec, BlockAllocator(32, 8), torch.device("cpu"), torch.float32, score_layer=2
    )
    assert cache.pruned.allocator is cache.full.allocator
    assert cache.pruned.key_pool is not cache.full.key_pool
    assert len(cache.full.key_pool) == 2
    assert len(cache.pruned.key_pool) == 2
    # Together they cost exactly one full stack, not two.
    assert len(cache.full.key_pool) + len(cache.pruned.key_pool) == spec.n_layers


def test_pool_memory_is_one_full_stack_not_two() -> None:
    """The bug that stopped the Phase 4 quality run, expressed as an assertion.

    At keep_ratio=1.0 nothing is pruned, so both halves hold the entire sequence -- which is the
    *worst* case for block demand, counter-intuitively, since the baseline row of the ablation is
    the one most likely to exhaust the pool.
    """
    spec = tiny_spec(n_layers=24)
    cache = CompressedKVCache(
        spec, BlockAllocator(64, 8), torch.device("cpu"), torch.float32, score_layer=2
    )
    per_layer_bytes = cache.full.key_pool[0].numel() * cache.full.key_pool[0].element_size()
    total = (len(cache.full.key_pool) + len(cache.pruned.key_pool)) * per_layer_bytes * 2
    one_stack = spec.n_layers * per_layer_bytes * 2
    assert total == one_stack


# --- compressor facade ----------------------------------------------------------------------------


def test_compressor_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        FastVCompressor(strategy="magic")


def test_compressor_records_its_last_selection() -> None:
    mask = _mask("tt" + "v" * 8 + "t")
    compressor = FastVCompressor(FastVConfig(keep_ratio=0.5))
    kept = compressor.select(_weights(11)[0], mask)
    assert compressor.last_kept is kept
    assert kept.numel() < 11


# --- end to end through the real decoder ---------------------------------------------------------


@pytest.fixture(scope="module")
def bundle():
    transformers = pytest.importorskip("transformers")
    from edgerag.core.loader import FIXTURE_MODEL
    from edgerag.core.model import load_from_hf
    from edgerag.core.spec import ModelSpec

    torch.manual_seed(0)
    config = transformers.AutoConfig.from_pretrained(FIXTURE_MODEL)
    # The *text* path is eager so it is bit-comparable with our decoder. The **vision tower is
    # not** -- it is HuggingFace's on both sides of every comparison here, so its attention
    # implementation cancels out, while eager costs +0.76 GiB of transient score matrix against
    # SDPA's +0.17 (measured). That is B-05's lesson applied to the suite: the peak that killed
    # the T4 quality run is the same peak that makes this file die on a loaded dev box (P-27).
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "sdpa"
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        FIXTURE_MODEL, config=config, dtype=torch.float32
    )
    model.eval()
    spec = ModelSpec.from_hf_config(FIXTURE_MODEL, config)
    # SDPA -- so the FastV end-to-end tests cover last_row_scores(), the cheap scoring path that
    # replaced materialising a 9 GiB score matrix to read one row of it.
    return load_from_hf(spec, model), spec


def _prompt(spec, n_text: int = 6, n_visual: int = 40):
    """Text prefix, a run of image tokens, then a text question -- the RAG prompt shape."""
    g = torch.Generator().manual_seed(11)
    text = torch.randint(100, spec.vocab_size - 100, (1, n_text), generator=g)
    images = torch.full((1, n_visual), spec.image_token_id)
    tail = torch.randint(100, spec.vocab_size - 100, (1, n_text), generator=g)
    ids = torch.cat([text, images, tail], dim=1)
    return ids, build_visual_mask(ids, spec.image_token_id)


@pytest.mark.slow
def test_keep_ratio_one_is_bit_identical_to_no_compressor(bundle) -> None:
    """The ablation baseline row must be the true baseline, not a near-miss.

    Every saving in the Phase 4 table is measured against this row; if it differs at all from
    plain inference, every number above it is offset by an unknown amount.
    """
    ours, spec = bundle
    ids, visual = _prompt(spec)

    with torch.inference_mode():
        plain = ours(input_ids=ids)
        noop = ours(
            input_ids=ids,
            compressor=FastVCompressor(FastVConfig(keep_ratio=1.0)),
            visual_mask=visual,
        )
    torch.testing.assert_close(noop, plain, atol=0.0, rtol=0.0)


@pytest.mark.slow
@pytest.mark.parametrize("keep_ratio", [0.25, 0.5, 0.75])
def test_pruning_shrinks_the_cache_above_the_cut(bundle, keep_ratio) -> None:
    """The memory win, measured through the real forward pass rather than asserted."""
    from edgerag.cache.allocator import BlockAllocator

    ours, spec = bundle
    ids, visual = _prompt(spec)
    cut = 2

    cache = CompressedKVCache(
        spec, BlockAllocator(512, 8), torch.device("cpu"), torch.float32, score_layer=cut
    )
    with torch.inference_mode():
        ours(
            input_ids=ids,
            cache=cache,
            compressor=FastVCompressor(FastVConfig(keep_ratio=keep_ratio, score_layer=cut)),
            visual_mask=visual,
        )

    saving = cache.savings()
    assert cache.full.seq_len == ids.shape[1], "layers below the cut must keep everything"
    assert cache.pruned.seq_len < cache.full.seq_len, "layers above the cut did not shrink"
    assert saving["layers_above_cut"] == spec.n_layers - cut
    assert 0.0 < saving["saving_fraction"] < 1.0
    # 40 visual tokens at this ratio, plus 12 text and the protected last position.
    expected_visual = max(1, round(40 * keep_ratio))
    assert abs(cache.pruned.seq_len - (expected_visual + 12)) <= 2


@pytest.mark.slow
def test_pruned_decode_still_produces_finite_logits(bundle) -> None:
    """Decode after pruning: layers below the cut see the full history, layers above see less.

    Position ids must follow the FULL length, so this also guards the renumbering trap in
    CompressedKVCache.seq_len.
    """
    from edgerag.cache.allocator import BlockAllocator

    ours, spec = bundle
    ids, visual = _prompt(spec)
    cache = CompressedKVCache(
        spec, BlockAllocator(512, 8), torch.device("cpu"), torch.float32, score_layer=2
    )

    with torch.inference_mode():
        ours(
            input_ids=ids,
            cache=cache,
            compressor=FastVCompressor(FastVConfig(keep_ratio=0.5, score_layer=2)),
            visual_mask=visual,
        )
        full_before = cache.full.seq_len
        step = ours(input_ids=ids[:, -1:], cache=cache)

    assert torch.isfinite(step).all()
    assert cache.full.seq_len == full_before + 1
    assert cache.pruned.seq_len < cache.full.seq_len


@pytest.mark.slow
def test_uniform_strategy_runs_end_to_end(bundle) -> None:
    """The control has to work through the same path, or the comparison is not like-for-like."""
    ours, spec = bundle
    ids, visual = _prompt(spec)

    with torch.inference_mode():
        out = ours(
            input_ids=ids,
            compressor=FastVCompressor(FastVConfig(keep_ratio=0.5), strategy="uniform"),
            visual_mask=visual,
        )
    assert torch.isfinite(out).all()
