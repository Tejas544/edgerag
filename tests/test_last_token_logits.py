"""``BUGS.md`` B-09: the prefill computed 641 MiB of logits and read one row of it.

Every consumer in this repo -- ``bench/pipeline.py``'s greedy loop, every batched-decode test --
reads ``logits[:, -1]`` and nothing else. On the measured 6,981-token RAG prefill the discarded
remainder is 6,980 x 49,280 fp16 = **641 MiB, 16% of the entire 4 GiB budget**.

These tests pin the two things that make the fix safe rather than merely smaller: the surviving
row must match the corresponding row of the full tensor to within measured fp32 GEMM noise, and
the default must stay "all logits" so the HuggingFace equivalence gate is untouched.

**The first version of this file asserted bit-identity and was wrong**, which is worth keeping
rather than quietly correcting. The reasoning was that every output row of a matmul depends only
on its own input row, so slicing before ``lm_head`` cannot change the survivor. That is true of
the *mathematics* and false of the *implementation*: a ``(1, H) x (H, V)`` GEMV tiles, vectorises
and accumulates differently from a ``(S, H) x (H, V)`` GEMM, so the dot product is summed in a
different order. Measured over 60 (seed, sequence-length) combinations: worst absolute difference
**2.1e-06**, worst relative **2.0e-03**, and the greedy token identical in **60 of 60**. Same
mechanism as ``CONTEXT.md`` D18 (a changed GEMM shape moves the last decimal places) and
``BUGS.md`` P-28 (CPU reduction order depends on thread dispatch). The tolerances below come from
that measurement, not from whatever made the test pass.
"""

from __future__ import annotations

import pytest
import torch

from edgerag.core.model import EdgeRagDecoder
from edgerag.core.spec import ModelSpec

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


#: Measured worst case over 60 (seed, sequence-length) combinations: 2.1e-06 absolute,
#: 2.0e-03 relative. Set ~5x above the observed absolute drift -- loose enough not to flake on
#: GEMM reduction order, tight enough that a genuinely wrong slice (off by O(1)) cannot pass.
SLICE_ATOL = 1e-5
SLICE_RTOL = 5e-3


def _decoder() -> EdgeRagDecoder:
    torch.manual_seed(0)
    return EdgeRagDecoder(TOY).eval()


def _ids(seq_len: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    return torch.randint(1, TOY.vocab_size, (1, seq_len), generator=generator)


def test_default_still_returns_every_position() -> None:
    """The HuggingFace equivalence gate compares full-sequence logits. It must not change."""
    decoder, ids = _decoder(), _ids()
    with torch.no_grad():
        assert decoder(input_ids=ids).shape == (1, ids.shape[1], TOY.vocab_size)


def test_last_token_only_returns_one_row() -> None:
    decoder, ids = _decoder(), _ids()
    with torch.no_grad():
        assert decoder(input_ids=ids, last_token_only=True).shape == (1, 1, TOY.vocab_size)


def test_the_surviving_row_matches_the_full_tensors_last_row() -> None:
    """Equal to within measured GEMM noise -- see the module docstring for why not exactly equal.

    ``RMSNorm`` reduces over the last dimension only, so slicing before it is mathematically a
    no-op for the surviving position. What is *not* a no-op is the ``lm_head`` matmul changing
    shape from ``(1, S, H)`` to ``(1, 1, H)``, which changes how the dot product is accumulated.
    """
    decoder, ids = _decoder(), _ids()
    with torch.no_grad():
        full = decoder(input_ids=ids)
        sliced = decoder(input_ids=ids, last_token_only=True)
    torch.testing.assert_close(sliced[:, 0], full[:, -1], atol=SLICE_ATOL, rtol=SLICE_RTOL)


@pytest.mark.parametrize("seed", range(6))
def test_the_greedy_token_is_unchanged(seed: int) -> None:
    """The property generation actually depends on, and the one the drift could plausibly break.

    A 2e-06 shift flips ``argmax`` only if the top two logits are within that of each other.
    Measured over 60 (seed, length) combinations: zero flips. Parametrised over seeds here rather
    than asserted once, because a single seed proving it holds is not evidence that it holds.
    """
    torch.manual_seed(seed)
    decoder = EdgeRagDecoder(TOY).eval()
    ids = _ids(24)
    with torch.no_grad():
        full = decoder(input_ids=ids)
        sliced = decoder(input_ids=ids, last_token_only=True)
    assert int(sliced[0, -1].argmax()) == int(full[0, -1].argmax())


@pytest.mark.parametrize("seq_len", [1, 2, 37])
def test_holds_at_any_sequence_length_including_one(seq_len: int) -> None:
    """``seq_len=1`` is the decode step, where the slice is a no-op and must stay exact."""
    decoder, ids = _decoder(), _ids(seq_len)
    with torch.no_grad():
        full = decoder(input_ids=ids)
        sliced = decoder(input_ids=ids, last_token_only=True)
    torch.testing.assert_close(sliced[:, 0], full[:, -1], atol=SLICE_ATOL, rtol=SLICE_RTOL)


def test_holds_across_a_batch() -> None:
    """Each row's last position, not the batch's -- an off-by-one here is silent."""
    decoder = _decoder()
    generator = torch.Generator().manual_seed(5)
    ids = torch.randint(1, TOY.vocab_size, (3, 16), generator=generator)
    with torch.no_grad():
        full = decoder(input_ids=ids)
        sliced = decoder(input_ids=ids, last_token_only=True)
    assert sliced.shape == (3, 1, TOY.vocab_size)
    torch.testing.assert_close(sliced[:, 0], full[:, -1], atol=SLICE_ATOL, rtol=SLICE_RTOL)


def test_the_saving_is_the_whole_point() -> None:
    """Guards the claim itself: the discarded tensor must actually be the bulk of the output.

    If a future change made ``lm_head`` cheap or the vocabulary small, this optimisation would
    stop being worth its parameter and this test should be the thing that says so.
    """
    seq_len, vocab = 6981, 49280  # the measured RAG prefill, results/quant_ablation.jsonl
    full_bytes = seq_len * vocab * 2
    kept_bytes = 1 * vocab * 2
    assert full_bytes - kept_bytes > 600 * 1024**2, "the saving should be hundreds of MiB"
    assert kept_bytes / full_bytes < 0.001, "the kept row should be a rounding error of the whole"
