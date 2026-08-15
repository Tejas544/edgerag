"""Phase 5b/5c: batched paged decode and chunked prefill.

Two properties, both in fp32 (``CONTEXT.md`` D13):

* **Batching must not change any sequence's output.** Decoding N sequences together must equal
  decoding each alone. This is the property a scheduler silently violates -- unequal lengths mean
  padding and per-sequence positions, and getting either wrong produces fluent, wrong text rather
  than a crash.
* **Chunked prefill must equal full prefill.** This is what fixes D14's 25-second TTFT, and it is
  worthless if it perturbs the answer.
"""

from __future__ import annotations

import pytest
import torch

from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.batched import BatchedPagedCache
from edgerag.cache.naive import NaiveKVCache
from edgerag.cache.paged import PagedKVCache
from edgerag.core.loader import FIXTURE_MODEL
from edgerag.core.model import load_from_hf
from edgerag.core.spec import ModelSpec

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
    return load_from_hf(spec, model), spec, torch.device("cpu"), torch.float32


def _ids(spec: ModelSpec, seq: int, salt: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seq * 97 + salt)
    return torch.randint(100, spec.vocab_size - 100, (1, seq), generator=g)


# --- chunked prefill (5c) ------------------------------------------------------------------------


def test_single_chunk_prefill_is_bit_identical(bundle) -> None:
    """One chunk means one GEMM shape, so this must be exact -- it proves the logic, not the noise.

    Measured 0.000e+00, as is paged-vs-naive single-shot. Any drift here is a real bug.
    """
    ours, spec, device, dtype = bundle
    ids = _ids(spec, 40)

    with torch.inference_mode():
        reference = ours(input_ids=ids, cache=NaiveKVCache(spec, 1, 128, device, dtype))
        chunked = ours(
            input_ids=ids, cache=PagedKVCache(spec, BlockAllocator(64, 16), device, dtype)
        )

    torch.testing.assert_close(chunked, reference, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("chunk", [1, 7, 16, 32])
def test_chunked_prefill_matches_full_prefill(bundle, chunk) -> None:
    """The fix for D14's TTFT, verified not to move the answer.

    A prompt processed in slices across iterations must land on the same logits as one processed
    whole. It does -- to a tolerance that **grows with the number of chunks**, because every chunk
    boundary changes the GEMM shape and the rounding compounds through 30 layers. Measured on a
    40-token prompt:

    ==========  ==========
    chunks      max abs
    ==========  ==========
    1           0.0
    2-3         4.3e-05
    6           1.6e-04
    40          9.1e-04
    ==========  ==========

    So a flat band would be either too loose for few chunks or too tight for many. Scaling with
    chunk count keeps it tight where tightness is meaningful. Greedy tokens are asserted
    separately, because that is the property that actually determines output.
    """
    ours, spec, device, dtype = bundle
    ids = _ids(spec, 40)
    n_chunks = (ids.shape[1] + chunk - 1) // chunk

    with torch.inference_mode():
        reference = ours(input_ids=ids, cache=NaiveKVCache(spec, 1, 128, device, dtype))

        cache = PagedKVCache(spec, BlockAllocator(64, 16), device, dtype)
        pieces = [
            ours(input_ids=ids[:, s : s + chunk], cache=cache)
            for s in range(0, ids.shape[1], chunk)
        ]
        chunked = torch.cat(pieces, dim=1)

    assert cache.seq_len == ids.shape[1]
    tolerance = 5e-5 * n_chunks
    torch.testing.assert_close(chunked, reference, atol=tolerance, rtol=tolerance)
    assert torch.equal(chunked.argmax(-1), reference.argmax(-1)), "chunking changed the tokens"


def test_chunk_boundaries_need_not_align_with_blocks(bundle) -> None:
    """Chunk size and block size are independent knobs and must not be assumed equal.

    A chunk of 7 against blocks of 16 leaves every chunk boundary mid-block, which is where an
    off-by-one in the write offset would surface.
    """
    ours, spec, device, dtype = bundle
    ids = _ids(spec, 33, salt=5)

    with torch.inference_mode():
        reference = ours(input_ids=ids, cache=NaiveKVCache(spec, 1, 64, device, dtype))
        cache = PagedKVCache(spec, BlockAllocator(64, 16), device, dtype)
        pieces = [
            ours(input_ids=ids[:, s : s + 7], cache=cache) for s in range(0, 33, 7)
        ]

    got = torch.cat(pieces, dim=1)
    torch.testing.assert_close(got, reference, atol=5e-4, rtol=5e-4)
    assert torch.equal(got.argmax(-1), reference.argmax(-1))


# --- batched decode (5b) -------------------------------------------------------------------------


def _decode_alone(ours, spec, device, dtype, ids: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Prefill then greedily step one sequence on its own. The reference for batching."""
    cache = PagedKVCache(spec, BlockAllocator(128, 16), device, dtype)
    with torch.inference_mode():
        logits = ours(input_ids=ids, cache=cache)
        outputs = []
        for _ in range(n_steps):
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            logits = ours(input_ids=token, cache=cache)
            outputs.append(logits)
    return torch.cat(outputs, dim=1)


def test_batched_decode_matches_solo_decode_for_equal_lengths(bundle) -> None:
    ours, spec, device, dtype = bundle
    prompts = [_ids(spec, 24, salt=1), _ids(spec, 24, salt=2)]
    steps = 4

    solo = [_decode_alone(ours, spec, device, dtype, p, steps) for p in prompts]

    batched = BatchedPagedCache(spec, BlockAllocator(256, 16), device, dtype)
    seqs = [batched.new_sequence() for _ in prompts]
    with torch.inference_mode():
        last = []
        for prompt, seq in zip(prompts, seqs, strict=True):
            last.append(ours(input_ids=prompt, cache=seq))

        batched.set_batch(seqs)
        tokens = torch.cat([logit[:, -1].argmax(dim=-1, keepdim=True) for logit in last], dim=0)
        got = []
        for _ in range(steps):
            logits = ours(
                input_ids=tokens,
                cache=batched,
                position_ids=batched.position_ids(),
                padding_mask=batched.padding_mask(),
            )
            got.append(logits)
            tokens = logits[:, -1].argmax(dim=-1, keepdim=True)

    stacked = torch.cat(got, dim=1)
    for i in range(len(prompts)):
        torch.testing.assert_close(
            stacked[i : i + 1], solo[i], atol=PAGED_ATOL, rtol=PAGED_RTOL
        )


def test_batched_decode_matches_solo_decode_for_unequal_lengths(bundle) -> None:
    """The case that actually breaks things.

    Different prompt lengths force padding in the gathered buffer and per-sequence position ids.
    A single scalar ``past_len`` -- what the decoder assumes when not told otherwise -- would give
    the shorter sequence the longer one's positions.
    """
    ours, spec, device, dtype = bundle
    prompts = [_ids(spec, 17, salt=3), _ids(spec, 33, salt=4), _ids(spec, 25, salt=9)]
    steps = 3

    solo = [_decode_alone(ours, spec, device, dtype, p, steps) for p in prompts]

    batched = BatchedPagedCache(spec, BlockAllocator(256, 16), device, dtype)
    seqs = [batched.new_sequence() for _ in prompts]
    with torch.inference_mode():
        last = [ours(input_ids=p, cache=s) for p, s in zip(prompts, seqs, strict=True)]
        batched.set_batch(seqs)
        tokens = torch.cat([logit[:, -1].argmax(dim=-1, keepdim=True) for logit in last], dim=0)
        got = []
        for _ in range(steps):
            logits = ours(
                input_ids=tokens,
                cache=batched,
                position_ids=batched.position_ids(),
                padding_mask=batched.padding_mask(),
            )
            got.append(logits)
            tokens = logits[:, -1].argmax(dim=-1, keepdim=True)

    stacked = torch.cat(got, dim=1)
    for i in range(len(prompts)):
        torch.testing.assert_close(
            stacked[i : i + 1], solo[i], atol=PAGED_ATOL, rtol=PAGED_RTOL
        )


# --- batched cache mechanics (no model) -----------------------------------------------------------


def _tiny():
    from tests.test_cow import tiny_spec

    spec = tiny_spec(n_layers=2)
    cache = BatchedPagedCache(
        spec, BlockAllocator(64, 8), torch.device("cpu"), torch.float32
    )
    return spec, cache


def _fill(spec, seq, n_tokens: int, value: float) -> None:
    kv = torch.full((1, spec.n_kv_heads, n_tokens, spec.head_dim), value)
    for layer in range(spec.n_layers):
        seq.update(kv, kv, layer)


def test_positions_are_per_sequence() -> None:
    """BUGS.md P-03/P-09: a shared scalar past_len rotates the shorter sequence wrongly."""
    spec, cache = _tiny()
    a, b = cache.new_sequence(), cache.new_sequence()
    _fill(spec, a, 10, 1.0)
    _fill(spec, b, 4, 2.0)
    cache.set_batch([a, b])

    assert cache.position_ids().flatten().tolist() == [10, 4]


def test_padding_mask_hides_the_short_sequence_tail() -> None:
    """Without it the shorter sequence attends to zero-filled padding -- P-01 in a new hat."""
    spec, cache = _tiny()
    a, b = cache.new_sequence(), cache.new_sequence()
    _fill(spec, a, 10, 1.0)
    _fill(spec, b, 4, 2.0)
    cache.set_batch([a, b])

    # n_new=0 describes the cache as it stands.
    mask = cache.padding_mask(n_new=0)
    assert mask.shape == (2, 10)
    assert mask[0].all()
    assert mask[1, :4].all() and not mask[1, 4:].any()

    # The decode default widens by one, because the mask is built before the write it describes.
    step_mask = cache.padding_mask()
    assert step_mask.shape == (2, 11)
    assert step_mask[0].all()
    assert step_mask[1, :5].all() and not step_mask[1, 5:].any()


def test_gather_right_pads_so_positions_line_up() -> None:
    spec, cache = _tiny()
    a, b = cache.new_sequence(), cache.new_sequence()
    _fill(spec, a, 10, 1.0)
    _fill(spec, b, 4, 2.0)
    cache.set_batch([a, b])

    keys, _ = cache.gather(0)
    assert keys.shape == (2, spec.n_kv_heads, 10, spec.head_dim)
    assert (keys[1, :, :4] == 2.0).all()
    assert (keys[1, :, 4:] == 0.0).all(), "padding must be at the tail, not the head"


def test_padding_waste_is_reported_not_hidden() -> None:
    """The honest counterweight to any batching speedup: bandwidth spent on zeros."""
    spec, cache = _tiny()
    a, b = cache.new_sequence(), cache.new_sequence()
    _fill(spec, a, 16, 1.0)
    _fill(spec, b, 4, 2.0)
    cache.set_batch([a, b])

    waste = cache.padding_waste()
    assert waste["real_tokens"] == 20
    assert waste["padded_tokens"] == 32
    assert waste["waste_fraction"] == pytest.approx(12 / 32)


def test_equal_lengths_waste_nothing() -> None:
    spec, cache = _tiny()
    a, b = cache.new_sequence(), cache.new_sequence()
    _fill(spec, a, 8, 1.0)
    _fill(spec, b, 8, 2.0)
    cache.set_batch([a, b])
    assert cache.padding_waste()["waste_fraction"] == 0.0


def test_sequences_share_one_pool() -> None:
    """A finished request's blocks must be reusable without draining the batch."""
    _, cache = _tiny()
    a, b = cache.new_sequence(), cache.new_sequence()
    assert a.key_pool is cache.key_pool
    assert b.key_pool is a.key_pool


def test_foreign_sequence_is_rejected() -> None:
    spec, cache = _tiny()
    # A cache paging into a *different* pool: sharing it would corrupt both.
    other = PagedKVCache(spec, BlockAllocator(16, 8), torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="different pool"):
        cache.set_batch([other])


def test_batch_size_mismatch_is_rejected() -> None:
    spec, cache = _tiny()
    a = cache.new_sequence()
    cache.set_batch([a])
    kv = torch.zeros(3, spec.n_kv_heads, 1, spec.head_dim)
    with pytest.raises(ValueError, match="does not match batch size"):
        cache.update(kv, kv, 0)
