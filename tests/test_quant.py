"""Phase 6: group-wise INT4 weight quantization.

Pure tensor arithmetic, no model, milliseconds -- the same discipline as the allocator, and for
the same reason. ``BUGS.md`` P-20 is a *silent* failure: a wrong grouping axis dequantizes without
error, produces tensors of exactly the right shape, and destroys quality. An end-to-end evaluation
would show only that quality collapsed, with no clue which of a hundred layers did it. These tests
assert on **error magnitude**, which localises it in milliseconds.
"""

from __future__ import annotations

import pytest
import torch

from edgerag.core.linear import (
    QUANTIZATION_SKIP_LIST,
    FP16Linear,
    QuantLinear,
    should_quantize,
)
from edgerag.core.quant import (
    INT4_MAX,
    QuantConfig,
    dequantize_groupwise,
    pack_int4,
    quantization_error,
    quantize_groupwise,
    unpack_int4,
)


def _weight(out_features: int = 64, in_features: int = 256, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(out_features, in_features, generator=g) * 0.02


# --- config -------------------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [1, 3, 16])
def test_unsupported_bit_widths_rejected(bits: int) -> None:
    with pytest.raises(ValueError, match="only 4- and 8-bit"):
        QuantConfig(bits=bits)


def test_group_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="group_size"):
        QuantConfig(group_size=0)


def test_ragged_groups_are_rejected_not_silently_truncated() -> None:
    """A final short group would need its own scale; failing beats quietly dropping weights."""
    with pytest.raises(ValueError, match="not divisible"):
        quantize_groupwise(_weight(4, 100), QuantConfig(group_size=128))


# --- round trip ---------------------------------------------------------------------------------


def test_shapes_survive_the_round_trip() -> None:
    weight = _weight()
    codes, scales = quantize_groupwise(weight)
    assert codes.shape == weight.shape
    assert codes.dtype == torch.int8
    assert scales.shape == (weight.shape[0], weight.shape[1] // 128)
    assert dequantize_groupwise(codes, scales).shape == weight.shape


def test_codes_stay_inside_the_signed_4bit_grid() -> None:
    codes, _ = quantize_groupwise(_weight())
    assert int(codes.max()) <= INT4_MAX
    assert int(codes.min()) >= -INT4_MAX


def test_round_trip_error_matches_4bit_theory() -> None:
    """Bounded by what 4 bits can actually do, not by an arbitrary target.

    For Gaussian weights the group max is around 3sigma, so the step is ~3sigma/7 and uniform
    quantization error is ~step/sqrt(12) ~ 0.124 sigma. Mean |w| is 0.798 sigma, so relative
    error lands near
    0.155. Anything much *below* that would mean the grid is not being used; anything far above
    means something is wrong. 18 dB SNR is the floor for a usable 4-bit representation.
    """
    stats = quantization_error(_weight())
    assert 0.05 < stats["relative_error"] < 0.18
    assert stats["signal_to_noise_db"] > 18.0


def test_grouping_along_the_wrong_axis_is_dramatically_worse() -> None:
    """BUGS.md P-20, encoded directly rather than hoped for by a threshold.

    Groups must run along the *input* dimension, so each covers 128 weights feeding one output.
    Grouping along the output dimension yields tensors of exactly the right shape and dequantizes
    without error -- it simply destroys the model.

    Writing this test corrected the diagnostic it relies on. The tensor-wide ``relative_error``
    barely moves (0.125 -> 0.149, a 19% change) because it is dominated by large-magnitude
    weights, while the damage falls on the *small* ones. ``max_row_relative_error`` sees it
    immediately. An aggregate that is nearly blind to the failure it exists to catch is worse
    than no metric at all.
    """
    # Square, so both orientations are legal group sizes and only the *meaning* differs.
    weight = _weight(256, 256)
    # Skew each output row's magnitude, so rows and columns are genuinely not interchangeable.
    # Without this the weight is isotropic and a flipped axis costs nothing -- which is exactly
    # why a naive round-trip test fails to catch P-20.
    weight = weight * torch.linspace(0.01, 4.0, 256).unsqueeze(1)
    config = QuantConfig()

    def restore(w: torch.Tensor) -> torch.Tensor:
        codes, scales = quantize_groupwise(w, config)
        return dequantize_groupwise(codes, scales, config).float()

    right = restore(weight)
    # The mistake: group along the other axis, then put the tensor back the way it was. Every
    # shape is correct and nothing raises.
    wrong = restore(weight.T.contiguous()).T.contiguous()

    def worst_row(restored: torch.Tensor) -> float:
        row_error = (restored - weight).abs().mean(dim=1)
        return float((row_error / weight.abs().mean(dim=1)).max())

    right_worst, wrong_worst = worst_row(right), worst_row(wrong)
    assert wrong_worst > right_worst * 5, (
        f"wrong-axis worst row scored {wrong_worst:.3f} against {right_worst:.3f} -- too close "
        "to distinguish, so this test would not catch the axis being flipped"
    )


def test_max_row_error_exposes_what_the_tensor_average_hides() -> None:
    """The metric fix, asserted on its own.

    One quiet row among loud ones is destroyed by a group spanning both, and a tensor-wide mean
    shrugs. This is why per-layer round-trip checks report the worst row.
    """
    weight = _weight(256, 256)  # square, so the transposed orientation is also a legal width
    weight[0] *= 0.001  # one very quiet output row

    # Group across rows -- the wrong axis -- so the quiet row shares a scale with loud ones.
    config = QuantConfig()
    codes, scales = quantize_groupwise(weight.T.contiguous(), config)
    restored = dequantize_groupwise(codes, scales, config).float().T.contiguous()

    quiet_row_error = float(
        (restored[0] - weight[0]).abs().mean() / weight[0].abs().mean()
    )
    tensor_wide = float((restored - weight).abs().mean() / weight.abs().mean())

    assert quiet_row_error > 0.5, "the quiet row should be wrecked"
    assert tensor_wide < quiet_row_error / 2, "the tensor-wide mean should barely notice"


def test_smaller_groups_are_more_accurate() -> None:
    """The whole argument for group-wise over per-tensor, stated as a monotonic property."""
    weight = _weight(32, 512)
    errors = [
        quantization_error(weight, QuantConfig(group_size=g))["relative_error"]
        for g in (512, 128, 32)
    ]
    assert errors[0] > errors[1] > errors[2], f"error did not fall with group size: {errors}"


def test_an_outlier_is_contained_within_its_group() -> None:
    """The failure mode per-tensor scaling has: one huge value flattens everything else.

    With grouping, the damage stops at the group boundary -- so the *untouched* groups must
    quantize just as well as they did before the outlier existed.
    """
    weight = _weight(8, 256)
    clean = quantization_error(weight[:, 128:], QuantConfig(group_size=128))["relative_error"]

    weight[0, 5] = 500.0  # a wild outlier, in the first group only
    codes, scales = quantize_groupwise(weight, QuantConfig(group_size=128))
    restored = dequantize_groupwise(codes, scales, QuantConfig(group_size=128))

    tail_error = (restored[:, 128:].float() - weight[:, 128:]).abs().mean()
    tail_denom = weight[:, 128:].abs().mean()
    assert float(tail_error / tail_denom) == pytest.approx(clean, rel=0.25)


def test_all_zero_group_does_not_divide_by_zero() -> None:
    weight = torch.zeros(4, 128)
    codes, scales = quantize_groupwise(weight)
    assert torch.isfinite(scales).all()
    restored = dequantize_groupwise(codes, scales)
    assert torch.equal(restored, torch.zeros(4, 128, dtype=torch.float16))


def test_exactly_representable_values_survive_unchanged() -> None:
    """Multiples of the group scale must round-trip exactly, or the grid is misaligned."""
    config = QuantConfig(group_size=120)  # 15 levels x 8 repeats
    weight = torch.tensor([[float(v) for v in range(-INT4_MAX, INT4_MAX + 1)] * 8])
    assert weight.shape[1] == 120

    codes, scales = quantize_groupwise(weight, config)
    restored = dequantize_groupwise(codes, scales, config)
    torch.testing.assert_close(restored.float(), weight, atol=1e-3, rtol=1e-3)


# --- packing: the step that makes "INT4" actually four bits ---------------------------------------


def test_packing_halves_the_byte_count() -> None:
    """Without this, codes are int8 and 'INT4' saves nothing -- the memory claim rests here."""
    codes, _ = quantize_groupwise(_weight(64, 256))
    packed = pack_int4(codes)
    assert packed.shape == (64, 128)
    assert packed.dtype == torch.uint8
    assert packed.numel() == codes.numel() // 2


def test_pack_unpack_is_lossless() -> None:
    codes, _ = quantize_groupwise(_weight(16, 256))
    assert torch.equal(unpack_int4(pack_int4(codes), 256), codes)


def test_packing_preserves_negative_values() -> None:
    """Nibbles are biased by +8 so the low one cannot borrow the high one's sign bit."""
    codes = torch.tensor([[-7, 7, -1, 1, 0, -4, 4, -7]], dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(codes), 8), codes)


def test_odd_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be even"):
        pack_int4(torch.zeros(4, 7, dtype=torch.int8))


# --- the layer ------------------------------------------------------------------------------------


def test_quant_linear_matches_fp16_within_quantization_error() -> None:
    """The layer is only useful if its output tracks the unquantized one."""
    weight = _weight(64, 256)
    x = torch.randn(4, 256)

    reference = FP16Linear(256, 64)
    reference.load_weight(weight)

    quantized = QuantLinear(256, 64)
    quantized.load_weight(weight)

    expected = reference(x)
    actual = quantized(x)
    relative = (actual - expected).abs().mean() / expected.abs().mean()
    assert float(relative) < 0.12, f"quantized output drifted {relative:.1%}"


def test_quant_linear_reports_real_bytes_including_scales() -> None:
    """Counting only packed weights overstates the saving. Real INT4 is 4.125 bits, not 4."""
    quantized = QuantLinear(256, 64)
    # FP16Linear names the intended dtype but does not enforce it -- nn.Linear defaults to fp32,
    # and the model is cast to fp16 wholesale at load. Cast explicitly so this compares 4 bits
    # against 16 rather than against 32.
    reference = FP16Linear(256, 64).half()
    reference.load_weight(_weight(64, 256).half())

    packed_only = 64 * 128
    assert quantized.weight_bytes() > packed_only, "scales were not counted"

    ratio = reference.weight_bytes() / quantized.weight_bytes()
    assert 3.5 < ratio < 4.0, f"expected just under 4x against fp16, got {ratio:.2f}x"


def test_quant_linear_rejects_a_mismatched_weight() -> None:
    quantized = QuantLinear(256, 64)
    with pytest.raises(ValueError, match="does not match layer"):
        quantized.load_weight(_weight(32, 256))


def test_quant_linear_rejects_bias() -> None:
    with pytest.raises(NotImplementedError):
        QuantLinear(256, 64, bias=True)


def test_quantization_happens_at_load_not_in_forward() -> None:
    """Re-quantizing per call would pay the cost on every token."""
    quantized = QuantLinear(256, 64)
    quantized.load_weight(_weight(64, 256))
    before = quantized.packed.clone()

    quantized(torch.randn(2, 256))
    assert torch.equal(quantized.packed, before)


# --- the skip list --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["embed_tokens", "lm_head", "input_layernorm", "norm", "patch_embedding"]
)
def test_sensitive_layers_are_skipped(name: str) -> None:
    """BUGS.md P-21. Quantizing these produces a cliff that reads as 'INT4 breaks VLMs'."""
    assert should_quantize(name) is False


@pytest.mark.parametrize(
    "name", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
def test_projection_layers_are_quantized(name: str) -> None:
    """These are the bulk of the parameters and the entire point of the exercise."""
    assert should_quantize(name) is True


def test_skip_list_matching_is_case_insensitive() -> None:
    assert should_quantize("LM_Head") is False
    assert should_quantize("model.layers.0.input_layerNorm") is False


def test_skip_list_is_not_empty() -> None:
    """A silently empty skip list would quantize everything and look like a quality collapse."""
    assert len(QUANTIZATION_SKIP_LIST) >= 4
