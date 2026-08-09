"""Tests for model introspection and KV-cache arithmetic.

These pin the numbers that the whole project argues from. If the KV math is wrong, the memory
budget is wrong, the block sizing is wrong, and the interview answer is wrong.

Values below are the *real* configs, fetched 2026-08-09 (see ``CONTEXT.md`` D10). They are
hardcoded deliberately: the test must fail if a checkpoint silently changes shape under us.
"""

from __future__ import annotations

import pytest

from edgerag.core.spec import GIB, ModelSpec

# SmolVLM2-2.2B-Instruct -- the headline model. Note n_kv_heads == n_q_heads: full MHA.
SPEC_2B = ModelSpec(
    model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    model_type="smolvlm",
    n_layers=24,
    hidden_size=2048,
    n_q_heads=32,
    n_kv_heads=32,
    head_dim=64,
    vocab_size=49280,
    max_position_embeddings=8192,
    rope_theta=10000.0,
    vision_layers=27,
    vision_hidden=1152,
    vision_image_size=384,
    vision_patch_size=14,
    scale_factor=3,
    image_token_id=49190,
)

# SmolVLM2-256M -- the local fixture. GQA 3:1.
SPEC_256M = ModelSpec(
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    model_type="smolvlm",
    n_layers=30,
    hidden_size=576,
    n_q_heads=9,
    n_kv_heads=3,
    head_dim=64,
    vocab_size=49280,
    max_position_embeddings=8192,
    rope_theta=10000.0,
    vision_layers=12,
    vision_hidden=768,
    vision_image_size=512,
    vision_patch_size=16,
    scale_factor=4,
    image_token_id=49190,
)


# --- attention shape ------------------------------------------------------------------------


def test_headline_model_is_mha_not_gqa() -> None:
    """The 2.2B checkpoint has 32 query heads and 32 kv heads.

    This is the finding that sizes the whole project: no GQA saving, so its KV cache is 4x what
    an 8-kv-head GQA model of the same depth would cost. See CONTEXT.md D10.
    """
    assert SPEC_2B.uses_gqa is False
    assert SPEC_2B.n_rep == 1


def test_fixture_model_uses_gqa() -> None:
    assert SPEC_256M.uses_gqa is True
    assert SPEC_256M.n_rep == 3


def test_n_rep_one_is_the_mha_degenerate_case() -> None:
    """MHA is GQA with n_rep=1, so one code path serves both tiers.

    Worth pinning: the local fixture only exercises n_rep=3, and the headline model only
    exercises n_rep=1. Neither tier covers the other, so the shared path must handle both.
    """
    assert SPEC_2B.n_rep == 1
    assert SPEC_256M.n_rep == 3


def test_indivisible_head_counts_rejected() -> None:
    bad = ModelSpec(**{**SPEC_2B.__dict__, "n_q_heads": 9, "n_kv_heads": 2})
    with pytest.raises(ValueError, match="not divisible"):
        _ = bad.n_rep


# --- BUGS.md L-01 -----------------------------------------------------------------------------


def test_out_of_range_pad_token_rejected_at_construction() -> None:
    """The shipped composite config carries pad_token_id=128002 against a 49280 vocabulary.

    Padding with it is an out-of-bounds embedding index -- on CUDA an async device-side assert
    that poisons the context and reports an unrelated traceback. Fail in Python instead.
    """
    with pytest.raises(ValueError, match="outside vocab_size"):
        ModelSpec(**{**SPEC_2B.__dict__, "pad_token_id": 128002})


def test_both_tiers_have_a_valid_pad_token() -> None:
    for spec in (SPEC_2B, SPEC_256M):
        assert 0 <= spec.pad_token_id < spec.vocab_size


# --- KV arithmetic (01_EDGERAG.md §7 question 1) ---------------------------------------------


def test_kv_bytes_per_token_2b() -> None:
    """2 (K,V) x 24 layers x 32 kv-heads x 64 head-dim x 2 bytes = 196,608 B = 192 KiB/token."""
    assert SPEC_2B.kv_bytes_per_token("float16") == 196_608
    assert SPEC_2B.kv_bytes_per_token("float16") / 1024 == 192.0


def test_kv_cache_at_2048_batch_8_nearly_fills_the_budget() -> None:
    """The number the project exists for: 3 GiB of KV cache against a 4 GiB total budget."""
    total = SPEC_2B.kv_bytes(seq_len=2048, batch=8, dtype="float16")
    assert total == 3 * GIB
    assert total / GIB == pytest.approx(3.0)


def test_single_sequence_at_2048_costs_384_mib() -> None:
    assert SPEC_2B.kv_bytes(seq_len=2048, batch=1) / (1024**2) == pytest.approx(384.0)


def test_fixture_kv_is_dramatically_smaller() -> None:
    """2 x 30 x 3 x 64 x 2 = 23,040 B/token -- 8.5x cheaper than the 2.2B despite more layers."""
    assert SPEC_256M.kv_bytes_per_token("float16") == 23_040
    ratio = SPEC_2B.kv_bytes_per_token() / SPEC_256M.kv_bytes_per_token()
    assert ratio == pytest.approx(8.53, abs=0.01)


def test_kv_scales_linearly_in_seq_and_batch() -> None:
    base = SPEC_2B.kv_bytes(1024, 1)
    assert SPEC_2B.kv_bytes(2048, 1) == 2 * base
    assert SPEC_2B.kv_bytes(1024, 4) == 4 * base


def test_fp32_kv_is_double_fp16() -> None:
    assert SPEC_2B.kv_bytes_per_token("float32") == 2 * SPEC_2B.kv_bytes_per_token("float16")


def test_kv_report_shows_the_arithmetic() -> None:
    report = SPEC_2B.kv_report(seq_len=2048, batch=8)
    assert "196,608 bytes" in report
    assert "192.0 KiB/token" in report
    assert "3.000 GiB" in report
    assert "MHA (no GQA saving)" in report
    assert "GQA 3:1" in SPEC_256M.kv_report()


# --- visual token arithmetic ------------------------------------------------------------------


def test_visual_tokens_per_subimage_2b() -> None:
    """384/14 = 27 patches per side -> 729 patches, pixel-shuffled by 3^2 -> 81 tokens."""
    assert SPEC_2B.patches_per_side == 27
    assert SPEC_2B.visual_tokens_per_subimage == 81


def test_visual_tokens_per_subimage_fixture() -> None:
    """512/16 = 32 per side -> 1024 patches, shuffled by 4^2 -> 64 tokens."""
    assert SPEC_256M.patches_per_side == 32
    assert SPEC_256M.visual_tokens_per_subimage == 64


def test_image_splitting_drives_the_visual_token_count() -> None:
    """A 2x2 split costs five sub-images: four tiles plus the global view.

    This is the lever the Phase 1 gate depends on. Without splitting, one image is 81 tokens and
    the 'visual tokens dominate the KV cache' thesis is false for this model.
    """
    assert SPEC_2B.visual_tokens_for(n_subimages=0) == 81
    assert SPEC_2B.visual_tokens_for(n_subimages=4) == 405
    assert SPEC_2B.visual_tokens_for(n_subimages=9) == 810
    assert SPEC_2B.visual_tokens_for(n_subimages=4, include_global=False) == 324


def test_five_retrieved_doc_pages_cost_two_thousand_visual_tokens() -> None:
    """k=5 retrieval at a 2x2 split -> 2025 visual tokens -> 380 MiB of KV cache on its own."""
    per_image = SPEC_2B.visual_tokens_for(n_subimages=4)
    total = per_image * 5
    assert total == 2025
    kv_mib = SPEC_2B.kv_bytes(total) / (1024**2)
    assert kv_mib == pytest.approx(379.7, abs=0.5)


# --- weights ------------------------------------------------------------------------------------


def test_int4_packs_two_values_per_byte() -> None:
    n = 2_200_000_000
    assert ModelSpec.effective_weight_bytes(n, "float16") == 2 * n
    assert ModelSpec.effective_weight_bytes(n, "int8") == n
    assert ModelSpec.effective_weight_bytes(n, "int4") == n // 2


def test_int4_rounds_up_on_odd_counts() -> None:
    assert ModelSpec.effective_weight_bytes(3, "int4") == 2


def test_2b_weights_only_fit_the_budget_at_int4() -> None:
    """2.2B fp16 is 4.1 GiB -- over budget before a single KV block is allocated."""
    n = 2_200_000_000
    assert ModelSpec.effective_weight_bytes(n, "float16") / GIB > 4.0
    assert ModelSpec.effective_weight_bytes(n, "int4") / GIB < 1.1


def test_spec_serialises_with_derived_fields() -> None:
    payload = SPEC_2B.to_dict()
    assert payload["uses_gqa"] is False
    assert payload["kv_bytes_per_token_fp16"] == 196_608
    assert payload["visual_tokens_per_subimage"] == 81
