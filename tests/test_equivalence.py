"""PHASE 2 GATE -- our forward pass must agree with HuggingFace's.

This is the load-bearing test of the whole project. Phase 3 replaces the KV cache with a paged
allocator, and the only thing standing between a subtly-wrong allocator and six days of poisoned
benchmarks is a reference implementation that is known-correct to a tight tolerance.

Run against the 256M fixture (``CONTEXT.md`` D4) so the suite finishes in seconds. A test that
takes four minutes gets run once a day; a test that takes four seconds gets run every commit, and
that difference is what actually catches the bug.

**Correctness is asserted in fp32, where the expected difference is exactly zero** (``CONTEXT.md``
D13). Measured 2026-08-10: our logits and hidden states are *bit-identical* to HuggingFace's in
fp32. So the tolerance here is 1e-6, not a fudge factor, and any nonzero difference is a bug with
no judgement call attached -- which is what makes this gate usable for the paged allocator in
Phase 3, where "is 1e-3 acceptable?" would otherwise cost an evening (``BUGS.md`` P-07).

fp16 is tested separately and only for the property that matters functionally: identical greedy
token ids. Its elementwise drift (mean ~2e-2 over 30 layers) is cuBLAS accumulation order, not
arithmetic disagreement.

Both sides are forced to the **eager** attention path, so the comparison is against the same
algorithm rather than against a fused kernel.
"""

from __future__ import annotations

import pytest
import torch

from edgerag.cache.naive import NaiveKVCache
from edgerag.core.layers import build_causal_mask, repeat_kv
from edgerag.core.loader import FIXTURE_MODEL
from edgerag.core.model import (
    EdgeRagDecoder,
    encode_images_chunked,
    load_from_hf,
    merge_image_features,
)
from edgerag.core.spec import ModelSpec

#: Same-shape comparisons (our prefill vs HF prefill) are bit-identical in fp32. 1e-6 leaves room
#: for cross-platform kernel variation and none for a bug: a wrong n_rep, RoPE base, or mask offset
#: moves logits by 1e-1 or more.
EXACT_ATOL = 1e-6
EXACT_RTOL = 1e-6

#: Cross-shape comparisons (cached decode vs full prefill) are *not* bit-identical, and cannot be.
#: Full prefill attends through one (1, H, S, D) x (1, H, D, S) GEMM; cached decode does S separate
#: (1, H, 1, D) x (1, H, D, t) GEMMs. Identical mathematics, different reduction order, so fp32
#: accumulation differs -- measured at 2.1e-05 on the fixture.
#:
#: This band matters beyond Phase 2: the paged allocator gathers blocks into a scratch buffer and
#: therefore also changes GEMM shape relative to the naive cache. Phase 3 should expect ~1e-5
#: against naive, not 0, and must not spend an evening hunting it (``BUGS.md`` P-07). 1e-4 is two
#: orders above the observed noise and three below what any real bug produces.
CACHE_ATOL = 1e-4
CACHE_RTOL = 1e-4

#: SDPA vs eager. A *different algorithm*, not a reordering: online softmax with tiled
#: accumulation. Measured 8.8e-04 max abs at seq 64 in fp32. Wider than CACHE_ATOL on purpose --
#: greedy-token equality is the assertion that matters, and this only catches gross regressions.
SDPA_ATOL = 5e-3

pytestmark = pytest.mark.slow


def _load(device: torch.device, dtype: torch.dtype):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)

    config = transformers.AutoConfig.from_pretrained(FIXTURE_MODEL)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        FIXTURE_MODEL, config=config, dtype=dtype
    )
    model.to(device).eval()
    spec = ModelSpec.from_hf_config(FIXTURE_MODEL, config)
    return model, spec, device, dtype


@pytest.fixture(scope="module")
def hf_bundle():
    """fp32 on CPU -- the correctness tier. Expected difference is zero."""
    return _load(torch.device("cpu"), torch.float32)


@pytest.fixture(scope="module")
def ours(hf_bundle):
    """Eager, to be bit-comparable with HuggingFace's eager path.

    The *shipping* path is SDPA -- eager materialises the full score matrix and cannot run a
    6,800-token prefill (``BUGS.md`` B-05). SDPA is covered by
    :func:`test_sdpa_matches_eager_output`, so neither path is untested.
    """
    model, spec, _, _ = hf_bundle
    return load_from_hf(spec, model, use_eager=True)


@pytest.fixture(scope="module")
def hf_bundle_fp16():
    """fp16 on CUDA -- the deployment tier. Only greedy-token agreement is asserted."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return _load(torch.device("cuda"), torch.float16)


@pytest.fixture(scope="module")
def ours_fp16(hf_bundle_fp16):
    model, spec, _, _ = hf_bundle_fp16
    return load_from_hf(spec, model, use_eager=True)


def _random_ids(spec: ModelSpec, batch: int, seq: int, device: torch.device) -> torch.Tensor:
    """Token ids drawn away from special ids, so nothing takes an unintended branch."""
    generator = torch.Generator(device="cpu").manual_seed(seq * 1000 + batch)
    return torch.randint(
        100, spec.vocab_size - 100, (batch, seq), generator=generator, device="cpu"
    ).to(device)


def _hf_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model.model.text_model(input_ids=input_ids).last_hidden_state @ (
            model.lm_head.weight.T
        )


# --- the gate ---------------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [1, 2, 7, 16, 17, 31, 32, 33, 64])
def test_prefill_logits_match_hf(hf_bundle, ours, seq_len) -> None:
    """Full-sequence forward, no cache. The base case everything else builds on.

    Lengths straddle powers of two deliberately -- Phase 3 introduces block boundaries at 16 and
    32, and this same sweep becomes the paged-cache gate.
    """
    model, spec, device, _ = hf_bundle
    ids = _random_ids(spec, 1, seq_len, device)

    expected = _hf_logits(model, ids)
    with torch.inference_mode():
        actual = ours(input_ids=ids)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=EXACT_ATOL, rtol=EXACT_RTOL)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_batched_prefill_matches_hf(hf_bundle, ours, batch) -> None:
    model, spec, device, _ = hf_bundle
    ids = _random_ids(spec, batch, 24, device)

    expected = _hf_logits(model, ids)
    with torch.inference_mode():
        actual = ours(input_ids=ids)

    torch.testing.assert_close(actual, expected, atol=EXACT_ATOL, rtol=EXACT_RTOL)


def test_hidden_states_are_bit_identical_in_fp32(hf_bundle, ours) -> None:
    """The claim behind D13, asserted rather than assumed.

    Comparing hidden states before ``lm_head`` localises any future regression to the decoder
    stack rather than to the output projection.
    """
    model, spec, device, _ = hf_bundle
    ids = _random_ids(spec, 1, 48, device)

    with torch.inference_mode():
        expected = model.model.text_model(input_ids=ids).last_hidden_state
        hidden = ours.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        cos, sin = ours.rotary(hidden, pos)
        mask = build_causal_mask(ids.shape[1], 0, hidden.dtype, device)
        for layer in ours.layers:
            hidden, _ = layer(hidden, cos, sin, mask, None)
        actual = ours.norm(hidden)

    torch.testing.assert_close(actual, expected, atol=EXACT_ATOL, rtol=EXACT_RTOL)


def test_incremental_decode_matches_full_prefill(hf_bundle, ours) -> None:
    """The KV cache must be transparent: decoding token-by-token equals one full forward.

    This is the property paging must preserve in Phase 3, and it is where a stale position offset
    or a mis-sliced cache view shows up. Prefilling a prefix and then stepping one token at a time
    must land on exactly the logits a single full-sequence pass produces.
    """
    _, spec, device, dtype = hf_bundle
    total, prefix = 40, 24
    ids = _random_ids(spec, 1, total, device)

    with torch.inference_mode():
        reference = ours(input_ids=ids)

        cache = NaiveKVCache(spec, batch_size=1, max_seq_len=total, device=device, dtype=dtype)
        stepped = [ours(input_ids=ids[:, :prefix], cache=cache)]
        for pos in range(prefix, total):
            stepped.append(ours(input_ids=ids[:, pos : pos + 1], cache=cache))

    got = torch.cat(stepped, dim=1)
    assert cache.seq_len == total
    # Prefilled positions go through the identical GEMM either way, so they must be bit-exact.
    torch.testing.assert_close(
        got[:, :prefix], reference[:, :prefix], atol=EXACT_ATOL, rtol=EXACT_RTOL
    )
    # Decoded positions differ only by reduction order -- see CACHE_ATOL.
    torch.testing.assert_close(got, reference, atol=CACHE_ATOL, rtol=CACHE_RTOL)


def test_decode_with_cache_matches_hf(hf_bundle, ours) -> None:
    """Same as above, but the ground truth is HF rather than our own prefill."""
    model, spec, device, dtype = hf_bundle
    total, prefix = 33, 20
    ids = _random_ids(spec, 1, total, device)
    expected = _hf_logits(model, ids)

    with torch.inference_mode():
        cache = NaiveKVCache(spec, 1, total, device, dtype)
        out = [ours(input_ids=ids[:, :prefix], cache=cache)]
        for pos in range(prefix, total):
            out.append(ours(input_ids=ids[:, pos : pos + 1], cache=cache))

    torch.testing.assert_close(
        torch.cat(out, dim=1), expected, atol=CACHE_ATOL, rtol=CACHE_RTOL
    )


def test_cached_prefill_is_bit_identical_to_uncached(hf_bundle, ours) -> None:
    """Writing to the cache must not perturb the computation at all.

    Isolates cache *bookkeeping* from cache *arithmetic*: this path has the same GEMM shape as the
    uncached one, so unlike decode it admits no accumulation excuse. Any nonzero difference here
    is a real bug.
    """
    _, spec, device, dtype = hf_bundle
    ids = _random_ids(spec, 1, 24, device)

    with torch.inference_mode():
        plain = ours(input_ids=ids)
        cache = NaiveKVCache(spec, 1, 64, device, dtype)
        cached = ours(input_ids=ids, cache=cache)

    assert cache.seq_len == 24
    torch.testing.assert_close(cached, plain, atol=EXACT_ATOL, rtol=EXACT_RTOL)


# --- fp16 deployment tier (CONTEXT.md D13) ---------------------------------------------------


@pytest.mark.parametrize("seq_len", [16, 33, 64])
def test_fp16_greedy_tokens_match_where_the_choice_is_not_a_coin_flip(
    hf_bundle_fp16, ours_fp16, seq_len
) -> None:
    """In fp16 the functional contract is greedy tokens, not logits.

    But *unconditional* argmax equality is the wrong assertion, and asserting it was a mistake:
    where the top-1 and top-2 logits sit within the ~2e-2 accumulation band, which candidate wins
    is decided by rounding, and either answer is equally defensible. Demanding equality there
    tests floating-point luck.

    So: exact agreement is required wherever the top-2 margin clears the noise band, and the
    coin-flip positions are counted rather than asserted on. A real bug (wrong ``n_rep``, wrong
    RoPE base) breaks the confident positions, not just the ties.
    """
    model, spec, device, _ = hf_bundle_fp16
    ids = _random_ids(spec, 1, seq_len, device)

    with torch.inference_mode():
        expected_logits = _hf_logits(model, ids).float()
        actual_logits = ours_fp16(input_ids=ids).float()

    top2 = expected_logits.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    confident = margin > 0.2  # an order of magnitude above the measured 2e-2 drift

    expected = expected_logits.argmax(dim=-1)
    actual = actual_logits.argmax(dim=-1)

    assert confident.any(), "no confidently-decided positions -- test proves nothing"
    assert torch.equal(actual[confident], expected[confident]), (
        "greedy token disagreement at a position where the top-2 margin exceeds fp16 noise"
    )

    agreement = (actual == expected).float().mean()
    assert agreement > 0.90, f"overall greedy agreement {agreement:.1%} is too low for tie-breaking"


@pytest.mark.parametrize("seq_len", [16, 64])
def test_fp16_drift_stays_in_the_accumulation_noise_band(
    hf_bundle_fp16, ours_fp16, seq_len
) -> None:
    """A guard rail, not a correctness claim -- correctness is the fp32 tests.

    Measured **relative** to logit magnitude rather than as an absolute bound. Absolute drift
    varies with the input: 2.0e-2 mean on one sequence, 5.3e-2 on another, purely because logit
    scale differs. An absolute threshold calibrated on one sample is the same mistake the harness
    rejects elsewhere -- a single number is not a result.

    Ratio-to-signal is stable across sequences, so it can be bounded honestly. A real defect moves
    this by orders of magnitude, not percent.
    """
    model, spec, device, _ = hf_bundle_fp16
    ids = _random_ids(spec, 1, seq_len, device)

    with torch.inference_mode():
        expected = _hf_logits(model, ids).float()
        actual = ours_fp16(input_ids=ids).float()

    signal = expected.abs().mean()
    relative = (actual - expected).abs().mean() / signal
    assert relative < 0.05, (
        f"fp16 drift is {relative:.2%} of logit magnitude (mean |logit| = {signal:.2f}); "
        "accumulation order alone should stay near 1%"
    )


# --- unit-level checks for the individually dangerous pieces --------------------------------


def test_repeat_kv_uses_interleave_semantics() -> None:
    """BUGS.md P-08. ``repeat`` tiles (h0 h1 h0 h1); ``repeat_interleave`` groups (h0 h0 h1 h1).

    Both produce identical shapes, so only the values distinguish them -- and the wrong one yields
    grammatical, wrong text.
    """
    # (batch=1, kv_heads=2, seq=1, head_dim=3); head 0 is all zeros, head 1 all ones.
    x = torch.stack([torch.zeros(1, 1, 3), torch.ones(1, 1, 3)], dim=1)
    assert x.shape == (1, 2, 1, 3)

    out = repeat_kv(x, 3)
    assert out.shape == (1, 6, 1, 3)
    head_values = [float(out[0, h, 0, 0]) for h in range(6)]
    assert head_values == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert head_values != [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # what `repeat` would give


def test_repeat_kv_is_identity_for_mha() -> None:
    """n_rep == 1 is the 2.2B headline model's path and must not copy."""
    x = torch.randn(1, 4, 3, 8)
    assert repeat_kv(x, 1) is x


def test_causal_mask_blocks_the_future() -> None:
    mask = build_causal_mask(4, 0, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 1] < -1e30  # query 0 cannot see key 1
    assert mask[0, 0, 3, 3] == 0.0


def test_causal_mask_uses_finfo_min_not_neg_inf() -> None:
    """BUGS.md P-10: -inf softmaxes a fully-masked row to NaN, which poisons the whole batch."""
    mask = build_causal_mask(3, 0, torch.float16, torch.device("cpu"))
    assert torch.isfinite(mask).all()
    assert mask.min() == torch.finfo(torch.float16).min


def test_decode_mask_row_sees_entire_history() -> None:
    """During decode seq_len == 1 and every past position is visible.

    Stated as a test because it is why mask bugs survive single-token testing and only appear
    during prefill.
    """
    mask = build_causal_mask(1, 10, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 1, 1, 11)
    assert (mask == 0.0).all()


def test_padding_mask_hides_left_padding() -> None:
    padding = torch.tensor([[0, 0, 1, 1]])
    mask = build_causal_mask(4, 0, torch.float32, torch.device("cpu"), padding_mask=padding)
    assert mask[0, 0, 3, 0] < -1e30  # padded key stays hidden
    assert mask[0, 0, 3, 2] == 0.0


# --- naive cache ------------------------------------------------------------------------------


def test_cache_advances_once_per_forward_not_once_per_layer(hf_bundle) -> None:
    """Length bookkeeping advances on the last layer only.

    Advancing per layer multiplies position by n_layers -- invisible in a one-layer test.
    """
    _, spec, device, dtype = hf_bundle
    cache = NaiveKVCache(spec, 1, 32, device, dtype)
    k = torch.zeros(1, spec.n_kv_heads, 5, spec.head_dim, device=device, dtype=dtype)

    for layer in range(spec.n_layers):
        cache.update(k, k, layer)
    assert cache.seq_len == 5


def test_cache_returns_only_the_filled_prefix(hf_bundle) -> None:
    """Returning the whole buffer would attend over uninitialised zeros -- P-01's cousin."""
    _, spec, device, dtype = hf_bundle
    cache = NaiveKVCache(spec, 1, 64, device, dtype)
    k = torch.ones(1, spec.n_kv_heads, 7, spec.head_dim, device=device, dtype=dtype)

    keys, values = cache.update(k, k, 0)
    assert keys.shape[2] == 7
    assert values.shape[2] == 7


def test_cache_overflow_raises_rather_than_corrupting(hf_bundle) -> None:
    _, spec, device, dtype = hf_bundle
    cache = NaiveKVCache(spec, 1, 8, device, dtype)
    k = torch.zeros(1, spec.n_kv_heads, 9, spec.head_dim, device=device, dtype=dtype)

    with pytest.raises(RuntimeError, match="overflow"):
        cache.update(k, k, 0)


def test_cache_reports_reservation_versus_use(hf_bundle) -> None:
    """The waste a contiguous cache cannot avoid, and the argument for paging."""
    _, spec, device, dtype = hf_bundle
    cache = NaiveKVCache(spec, 1, 2048, device, dtype)
    k = torch.zeros(1, spec.n_kv_heads, 300, spec.head_dim, device=device, dtype=dtype)
    for layer in range(spec.n_layers):
        cache.update(k, k, layer)

    assert cache.used_bytes() / cache.nbytes == pytest.approx(300 / 2048, abs=1e-3)


# --- SDPA, the shipping path (BUGS.md B-05) -------------------------------------------------------


@pytest.fixture(scope="module")
def ours_sdpa(hf_bundle):
    model, spec, _, _ = hf_bundle
    return load_from_hf(spec, model, use_eager=False)


@pytest.mark.parametrize("seq_len", [1, 17, 64])
def test_sdpa_matches_eager_output(hf_bundle, ours, ours_sdpa, seq_len) -> None:
    """The production path must agree with the reference one.

    Eager is what the HF comparison is made against; SDPA is what actually runs, because eager
    materialises a ``(heads, queries, keys)`` matrix that is 9 GiB at a realistic prompt length.
    Testing only eager would leave the shipped path uncovered -- the mistake ``BUGS.md`` B-04
    records.

    SDPA is not a reordering of the eager arithmetic -- it is a different algorithm (online
    softmax, tiled accumulation), so bit-exactness was never on offer. Measured 8.8e-04 max abs on
    0.4% of elements at seq 64, in fp32. **Greedy tokens are the real assertion here**; the
    numeric band only guards against a gross regression, and holding it near the eager tolerance
    would just invite loosening the tests that genuinely are exact.
    """
    _, spec, device, _ = hf_bundle
    ids = _random_ids(spec, 1, seq_len, device)

    with torch.inference_mode():
        expected = ours(input_ids=ids)
        actual = ours_sdpa(input_ids=ids)

    torch.testing.assert_close(actual, expected, atol=SDPA_ATOL, rtol=SDPA_ATOL)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1)), "SDPA changed the greedy tokens"


def test_sdpa_does_not_materialise_the_score_matrix(hf_bundle, ours_sdpa) -> None:
    """The whole point: memory must not scale with sequence length squared.

    Doubling the prompt roughly doubles activation memory under SDPA and roughly *quadruples* it
    under eager. Asserted as a ratio rather than an absolute, so it holds on any device.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to measure allocation")

    _, spec, _, _ = hf_bundle
    cuda_model = ours_sdpa.to("cuda")
    peaks = []
    for seq_len in (256, 512):
        ids = _random_ids(spec, 1, seq_len, torch.device("cuda"))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            cuda_model(input_ids=ids)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    ours_sdpa.to("cpu")

    growth = peaks[1] / peaks[0]
    assert growth < 3.0, f"peak memory grew {growth:.1f}x for 2x sequence -- looks quadratic"


# --- vision tower (BUGS.md B-04) ----------------------------------------------------------------


@pytest.fixture(scope="module")
def image_batch(hf_bundle):
    """One real image through the real processor.

    Real, because synthetic *pixel* tensors would not have caught B-04 -- that bug was in mask
    construction, which depends on the processor's actual output shapes and sub-image splitting.

    Deliberately small (640x480, so few sub-images). The vision tower is the most expensive thing
    in the suite by a wide margin: a 900x1200 page splits into ~16 sub-images of 1024 patches
    each, and a 12-layer ViT over 16k tokens in fp32 on CPU takes minutes. A suite that takes ten
    minutes stops being run every commit, which is the same argument D4 makes for the 256M
    fixture in the first place.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    from PIL import Image

    processor = transformers.AutoProcessor.from_pretrained(FIXTURE_MODEL)
    image = Image.new("RGB", (640, 480), color=(30, 60, 90))
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what?"}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(text=prompt, images=[image], return_tensors="pt")
    image.close()
    return batch


@pytest.fixture(scope="module")
def unchunked_features(hf_bundle, image_batch):
    """Encode once, unchunked, and reuse as the reference for every chunk-size case."""
    model, _, _, _ = hf_bundle
    with torch.inference_mode():
        return encode_images_chunked(model, image_batch["pixel_values"], chunk_size=4096)


def test_chunked_vision_encoding_matches_huggingface(hf_bundle, image_batch) -> None:
    """Our chunked tower must equal HF's ``get_image_features``.

    This test did not exist until the function failed on its first real use (``BUGS.md`` B-04).
    Nothing in the local suite passed real images through the vision path, so a mask-shape bug
    shipped and was found on a T4 after a 9 GB download.

    Compared at ``CACHE_ATOL``, not bit-exactly, and the reason is worth stating: whenever the
    sub-image count is not a multiple of the chunk size, the final chunk is smaller -- and a
    chunk holding a *single* image takes a different GEMM path from a batched one, shifting fp32
    rounding by ~5e-06. That is the same effect ``CACHE_ATOL`` documents for the KV cache.
    Whether any given input happens to divide evenly is not a property to design around.
    """
    model, _, _, _ = hf_bundle
    pixel_values = image_batch["pixel_values"]

    with torch.inference_mode():
        expected = model.model.get_image_features(pixel_values).pooler_output
        actual = encode_images_chunked(model, pixel_values, chunk_size=4)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=CACHE_ATOL, rtol=CACHE_RTOL)


@pytest.mark.parametrize("chunk_size", [1, 8])
def test_chunk_size_does_not_change_the_result(
    hf_bundle, image_batch, unchunked_features, chunk_size
) -> None:
    """D12 claims chunking is equivalent, not an approximation. Verified here, with a caveat.

    It holds because no attention crosses sub-image boundaries inside the tower. But *"it should"*
    and *"it does"* are different statements and only one is a test -- and running the test twice,
    on different images, refined the claim twice.

    Chunking is **mathematically** equivalent and agrees to ~5e-06, not bit-exactly. A chunk
    holding a single image takes a different GEMM path from a batched one, so reduction order and
    fp32 rounding differ. Any input whose sub-image count is not a multiple of the chunk size ends
    with such a chunk.

    A first run on a larger image *appeared* bit-exact at chunk sizes >= 3 -- that image simply
    divided evenly and produced no batch-1 remainder. Reading agreement as a property when it was
    a coincidence of one input is exactly the mistake ``CACHE_ATOL`` exists to prevent.
    """
    model, _, _, _ = hf_bundle
    with torch.inference_mode():
        chunked = encode_images_chunked(model, image_batch["pixel_values"], chunk_size=chunk_size)

    # 1e-4 is four orders above the measured 5e-06 and far below anything a real bug produces.
    torch.testing.assert_close(chunked, unchunked_features, atol=CACHE_ATOL, rtol=CACHE_RTOL)


def test_image_features_count_matches_the_image_token_slots(
    hf_bundle, image_batch, unchunked_features
) -> None:
    """The invariant that catches P-11 and P-26 before they corrupt an answer.

    If the tower returns a different number of features than the prompt has ``<image>`` slots,
    every embedding after the mismatch shifts by one and the model answers confidently wrong.
    """
    _, spec, _, _ = hf_bundle
    n_slots = int((image_batch["input_ids"] == spec.image_token_id).sum())
    assert unchunked_features.shape[0] * unchunked_features.shape[1] == n_slots


def test_merge_rejects_a_feature_count_mismatch(
    hf_bundle, image_batch, unchunked_features
) -> None:
    """The guard fires loudly rather than silently shifting embeddings."""
    _, spec, device, dtype = hf_bundle
    input_ids = image_batch["input_ids"]
    embeds = torch.zeros(
        input_ids.shape[0], input_ids.shape[1], spec.hidden_size, dtype=dtype, device=device
    )

    with pytest.raises(ValueError, match="image features"):
        merge_image_features(input_ids, embeds, unchunked_features[:-1], spec.image_token_id)


def test_decoder_rejects_both_ids_and_embeds(hf_bundle) -> None:
    _, spec, device, dtype = hf_bundle
    model = EdgeRagDecoder(spec).to(device=device, dtype=dtype)
    ids = torch.zeros(1, 3, dtype=torch.long, device=device)
    embeds = torch.zeros(1, 3, spec.hidden_size, device=device, dtype=dtype)

    with pytest.raises(ValueError, match="exactly one"):
        model(input_ids=ids, inputs_embeds=embeds)
    with pytest.raises(ValueError, match="exactly one"):
        model()
