"""Phase 6: what happens when the quantization flag is turned on.

``tests/test_quant.py`` proves the arithmetic; this proves the *wiring*, which fails differently.
The arithmetic failures are loud and local -- a wrong axis, a ragged group. The wiring failures are
quiet and global: a decoder that quantized its ``lm_head`` and lost 20 ANLS, or a memory ledger
that reported a saving the running model never made. Two of the tests below exist only to make
those two silences impossible.

Everything except the two ``slow`` cases runs on the ``meta`` device or a 2-layer toy spec, so the
file costs milliseconds and no download (``CONTEXT.md`` D4).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from edgerag.core import linear as linear_mod
from edgerag.core.linear import (
    SKIP_RAGGED,
    SKIP_SENSITIVE,
    FP16Linear,
    LinearBase,
    QuantLinear,
    plan_quantization,
    quantize_module_,
)
from edgerag.core.model import EdgeRagDecoder, load_from_hf
from edgerag.core.quant import QuantConfig
from edgerag.core.spec import ModelSpec

#: A decoder small enough to build in a millisecond and shaped so group 128 divides it.
TOY = ModelSpec(
    model_id="toy",
    model_type="smolvlm",
    n_layers=2,
    hidden_size=256,
    n_q_heads=4,
    n_kv_heads=2,
    head_dim=64,
    vocab_size=512,
    max_position_embeddings=1024,
    rope_theta=10000.0,
    vision_layers=2,
    vision_hidden=128,
    vision_image_size=64,
    vision_patch_size=16,
    scale_factor=2,
    image_token_id=511,
    pad_token_id=0,
    intermediate_size=512,
)


def _projections(decoder: EdgeRagDecoder) -> list[LinearBase]:
    return [
        decoder.layers[0].self_attn.q_proj,
        decoder.layers[0].mlp.gate_proj,
        decoder.layers[1].self_attn.o_proj,
        decoder.layers[1].mlp.down_proj,
    ]


# --- the flag -------------------------------------------------------------------------------


def test_flag_off_leaves_the_fp16_stack_untouched() -> None:
    """The default path must not change at all -- every earlier phase is measured against it."""
    decoder = EdgeRagDecoder(TOY)
    assert decoder.quant_config is None
    assert all(isinstance(p, FP16Linear) for p in _projections(decoder))
    assert isinstance(decoder.lm_head, FP16Linear)


def test_flag_on_quantizes_every_projection() -> None:
    decoder = EdgeRagDecoder(TOY, quant_config=QuantConfig())
    assert all(isinstance(p, QuantLinear) for p in _projections(decoder))


def test_the_lm_head_stays_fp16() -> None:
    """``BUGS.md`` P-21. Its error reaches argmax with nothing downstream to attenuate it."""
    decoder = EdgeRagDecoder(TOY, quant_config=QuantConfig())
    assert isinstance(decoder.lm_head, FP16Linear)


def test_the_head_asks_the_skip_list_rather_than_hardcoding_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip list is the single source of truth, so removing ``lm_head`` from it must land.

    Without this, someone edits ``QUANTIZATION_SKIP_LIST``, the README prints the new list, and
    the model quietly keeps the old behaviour -- the exact failure mode the list exists to prevent.
    """
    monkeypatch.setattr(linear_mod, "QUANTIZATION_SKIP_LIST", ("embed_tokens", "norm"))
    decoder = EdgeRagDecoder(TOY, quant_config=QuantConfig())
    assert isinstance(decoder.lm_head, QuantLinear)


def test_quantized_decoder_is_smaller_and_still_runs() -> None:
    fp16 = EdgeRagDecoder(TOY).half()
    quantized = EdgeRagDecoder(TOY, quant_config=QuantConfig()).half()

    def weight_bytes(decoder: EdgeRagDecoder) -> int:
        return sum(m.weight_bytes() for m in decoder.modules() if isinstance(m, LinearBase))

    assert weight_bytes(quantized) < weight_bytes(fp16) / 2

    ids = torch.randint(0, TOY.vocab_size, (1, 8))
    assert quantized(ids).shape == (1, 8, TOY.vocab_size)


# --- the plan, and why the ledger can be trusted ---------------------------------------------


def test_the_plan_and_the_built_model_agree_byte_for_byte() -> None:
    """The memory ledger is computed from a *plan* over a dense model on ``meta``.

    That is only legitimate if the plan predicts what building the thing actually produces. This
    is the assertion that makes ``scripts/measure_memory_ledger.py`` a measurement of the model
    rather than a second, unverified copy of the packing arithmetic.
    """
    config = QuantConfig()
    with torch.device("meta"):
        # fp16 on both sides: the plan prices the deployment dtype, and a module built under the
        # fp32 default would report an lm_head twice the size the ledger claims.
        dense = EdgeRagDecoder(TOY).half()
        quantized = EdgeRagDecoder(TOY, quant_config=config).half()

    plan = plan_quantization(dense, config)
    built = sum(m.weight_bytes() for m in quantized.modules() if isinstance(m, LinearBase))
    assert plan.planned_bytes == built


def test_plan_classifies_each_layer_and_prices_both_outcomes() -> None:
    tree = nn.Sequential()
    tree.add_module("q_proj", nn.Linear(256, 64, bias=False))
    tree.add_module("fc2", nn.Linear(300, 64, bias=True))  # 300 % 128 != 0
    tree.add_module("lm_head", nn.Linear(256, 64, bias=False))

    plan = plan_quantization(tree, QuantConfig())
    by_name = {layer.name: layer for layer in plan.layers}

    assert by_name["q_proj"].skipped is None
    assert by_name["fc2"].skipped == SKIP_RAGGED
    assert by_name["lm_head"].skipped == SKIP_SENSITIVE
    # A skipped layer costs what it costs: dense in, dense out, no phantom saving.
    for name in ("fc2", "lm_head"):
        assert by_name[name].planned_bytes == by_name[name].dense_bytes
    assert 4.0 < plan.bits_per_weight < 4.2, "group 128 should land at 4.125 bits plus bias"


def test_plan_needs_no_weights() -> None:
    """The whole ablation table has to be computable from a config on a laptop."""
    with torch.device("meta"):
        decoder = EdgeRagDecoder(TOY)
    plan = plan_quantization(decoder, QuantConfig())
    assert plan.planned_bytes > 0
    assert plan.saved_bytes > 0


# --- in-place replacement, for the vision tower ----------------------------------------------


def test_quantize_module_replaces_in_place_and_keeps_the_bias() -> None:
    """SigLIP's projections all carry a bias; the language decoder's carry none."""
    torch.manual_seed(0)
    tree = nn.Sequential()
    tree.add_module("fc1", nn.Linear(256, 64, bias=True))
    reference = tree.fc1.weight.detach().clone(), tree.fc1.bias.detach().clone()
    x = torch.randn(3, 256)
    expected = tree(x)

    quantize_module_(tree, QuantConfig())

    assert isinstance(tree.fc1, QuantLinear)
    assert torch.equal(tree.fc1.bias.float(), reference[1].half().float())
    relative = (tree(x) - expected).abs().mean() / expected.abs().mean()
    assert float(relative) < 0.15


def test_quantize_module_refuses_meta_tensors() -> None:
    """Accounting is what ``plan_quantization`` is for; this one needs real values."""
    with torch.device("meta"):
        tree = nn.Sequential()
        tree.add_module("fc1", nn.Linear(256, 64))
    with pytest.raises(ValueError, match="meta device"):
        quantize_module_(tree, QuantConfig())


def test_a_layer_the_group_size_cannot_divide_is_refused_at_construction() -> None:
    """SmolVLM2's vision MLP is the real case: in_features=4304 = 16 x 269.

    Failing here rather than at ``load_weight`` matters because the memory ledger prices layers it
    never loads. A layer that constructs and then refuses its weights would be counted as
    quantized and shipped as fp16.
    """
    with pytest.raises(ValueError, match="not divisible by group_size"):
        QuantLinear(4304, 1152, config=QuantConfig(group_size=128))


# --- through the real checkpoint --------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_bundle():
    transformers = pytest.importorskip("transformers")
    from edgerag.core.loader import FIXTURE_MODEL

    config = transformers.AutoConfig.from_pretrained(FIXTURE_MODEL)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        FIXTURE_MODEL, config=config, dtype=torch.float32
    )
    model.eval()
    return model, ModelSpec.from_hf_config(FIXTURE_MODEL, config)


def _agreement(reference: EdgeRagDecoder, quantized: EdgeRagDecoder, spec: ModelSpec):
    ids = torch.randint(
        100, spec.vocab_size - 100, (1, 64), generator=torch.Generator().manual_seed(3)
    )
    with torch.no_grad():
        expected, actual = reference(ids), quantized(ids)
    relative = float((actual - expected).abs().mean() / expected.abs().mean())
    agreed = float((actual.argmax(-1) == expected.argmax(-1)).float().mean())
    return relative, agreed


@pytest.mark.slow
def test_int8_through_the_real_checkpoint_is_nearly_lossless(fixture_bundle) -> None:
    """INT8 is the control. If the wiring were wrong, *this* would break, and loudly.

    Measured on the 256M fixture: 1.3% logit drift, 98% of greedy tokens unchanged. A weight
    quantizer that damages an 8-bit model has a bug, not a precision limit -- which is what makes
    this the right assertion to hang the wiring on, and the INT4 case below the wrong one.

    Group 64 rather than the 128 default throughout this file: the fixture's hidden size is
    576 = 4.5 x 128, so 128 does not divide it. Not a workaround -- it is the same divisibility
    constraint the headline model's vision MLP hits, surfacing on the tier that is cheap to test.
    """
    model, spec = fixture_bundle
    reference = load_from_hf(spec, model)
    quantized = load_from_hf(spec, model, quant_config=QuantConfig(group_size=64, bits=8))

    relative, agreed = _agreement(reference, quantized, spec)
    assert relative < 0.05, f"INT8 logits drifted {relative:.1%}"
    assert agreed >= 0.90, f"INT8 changed {1 - agreed:.0%} of greedy tokens"


@pytest.mark.slow
def test_int4_on_the_fixture_degrades_but_does_not_collapse(fixture_bundle) -> None:
    """**This is a wiring test, not a quality result**, and the distinction is the point.

    Measured: INT4 at group 64 leaves 66% of greedy tokens unchanged on the 256M fixture, against
    98% for INT8. A 256M model is the hardest case for 4-bit weights -- there are fewer parameters
    for the error to average out over -- so this number says nothing about the 2.2B headline model
    and must never be quoted as if it did. The quality claim comes from the T4 ANLS sweep.

    What is asserted here is only what the wiring can guarantee: the model still tracks its fp16
    self far above chance, and a smaller group is better than a larger one -- the same monotonic
    property ``test_quant.py`` asserts on a single tensor, holding end to end through 30 layers.
    """
    model, spec = fixture_bundle
    reference = load_from_hf(spec, model)
    coarse = load_from_hf(spec, model, quant_config=QuantConfig(group_size=64))
    fine = load_from_hf(spec, model, quant_config=QuantConfig(group_size=16))

    coarse_relative, coarse_agreed = _agreement(reference, coarse, spec)
    _, fine_agreed = _agreement(reference, fine, spec)

    assert coarse_relative < 0.30, f"INT4 logits drifted {coarse_relative:.1%} -- collapsed"
    assert coarse_agreed >= 0.50, f"INT4 changed {1 - coarse_agreed:.0%} of greedy tokens"
    assert fine_agreed > coarse_agreed, (
        f"group 16 ({fine_agreed:.0%}) should beat group 64 ({coarse_agreed:.0%})"
    )


@pytest.mark.slow
def test_load_from_hf_quantized_actually_holds_fewer_bytes(fixture_bundle) -> None:
    """A quantized model that is not smaller has failed at the only thing it was for."""
    model, spec = fixture_bundle
    reference = load_from_hf(spec, model).half()
    quantized = load_from_hf(spec, model, quant_config=QuantConfig(group_size=64)).half()

    def weight_bytes(decoder: EdgeRagDecoder) -> int:
        return sum(m.weight_bytes() for m in decoder.modules() if isinstance(m, LinearBase))

    ratio = weight_bytes(reference) / weight_bytes(quantized)
    # Not 4x: the fixture's lm_head is 28% of its linear weights and stays fp16 by design.
    assert 1.5 < ratio < 4.0, f"expected a real but skip-list-limited saving, got {ratio:.2f}x"
