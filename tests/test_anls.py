"""Tests for the ANLS scorer used by the Phase 4 quality curve.

Written *before* the curve is measured, deliberately. ``BUGS.md`` B-04 shipped because a function
whose only caller lived in a script was never executed by a test; `levenshtein` and `anls` are in
exactly that position, and a defect in either would not crash — it would produce a plausible,
wrong quality curve. That is worse than a crash, because the number would be published.
"""

from __future__ import annotations

import pytest

from bench.metrics import anls, levenshtein

# --- edit distance ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),  # the textbook case
        ("flaw", "lawn", 2),
        ("0.28", "0.28%", 1),
    ],
)
def test_levenshtein_known_distances(a: str, b: str, expected: int) -> None:
    assert levenshtein(a, b) == expected


def test_levenshtein_is_symmetric() -> None:
    """The implementation swaps arguments internally to bound memory; that must not change it."""
    for a, b in [("pinterest", "interest"), ("abcdef", "az"), ("", "xyz")]:
        assert levenshtein(a, b) == levenshtein(b, a)


def test_levenshtein_handles_long_inputs_without_recursion_limits() -> None:
    """Generative answers can be long; a recursive implementation would blow the stack."""
    assert levenshtein("a" * 2000, "a" * 2000) == 0
    assert levenshtein("a" * 2000, "b" * 2000) == 2000


# --- ANLS ---------------------------------------------------------------------------------------


def test_exact_match_scores_one() -> None:
    assert anls("0.28", ["0.28"]) == 1.0


def test_scoring_is_case_and_whitespace_insensitive() -> None:
    """Generated answers carry incidental whitespace and capitalisation."""
    assert anls("  Pinterest ", ["pinterest"]) == 1.0


def test_best_of_several_gold_answers_wins() -> None:
    """DocVQA ships multiple acceptable answers; matching any one of them is correct."""
    assert anls("pinterest", ["facebook", "pinterest", "twitter"]) == 1.0


def test_near_miss_scores_partially_not_zero() -> None:
    """The reason ANLS is used instead of exact match.

    "0.28" against "0.28%" is a real answer with a formatting difference. Exact match calls that a
    total failure, which would make the pruning curve look like a cliff where there is none.
    """
    score = anls("0.28%", ["0.28"])
    assert 0.5 <= score < 1.0


def test_answers_below_the_threshold_score_zero_not_partial() -> None:
    """Without the cutoff, a long wrong answer collects partial credit for incidental overlap."""
    assert anls("completely different text here", ["0.28"]) == 0.0


def test_threshold_boundary_is_inclusive() -> None:
    """A similarity of exactly the threshold counts, matching the standard definition."""
    # "abcd" vs "abxy": distance 2, denom 4 -> similarity exactly 0.5.
    assert anls("abcd", ["abxy"], threshold=0.5) == pytest.approx(0.5)
    assert anls("abcd", ["abxy"], threshold=0.51) == 0.0


def test_empty_prediction_scores_zero_against_a_real_answer() -> None:
    """An empty generation is a failure, and must not be rewarded for having no wrong characters."""
    assert anls("", ["0.28"]) == 0.0


def test_empty_prediction_and_empty_gold_agree() -> None:
    assert anls("", [""]) == 1.0


def test_no_gold_answers_scores_zero() -> None:
    """Guards the divide-by-zero path rather than letting it raise mid-sweep."""
    assert anls("anything", []) == 0.0


def test_score_is_bounded() -> None:
    """A curve axis that can exceed 1.0 or go negative is unreadable."""
    cases = [("", ["x"]), ("x", [""]), ("abc", ["abc"]), ("zzz", ["abc"]), ("ab", ["abcdefgh"])]
    for prediction, answers in cases:
        assert 0.0 <= anls(prediction, answers) <= 1.0


# --- the generation path (BUGS.md B-05) -----------------------------------------------------------


@pytest.mark.slow
def test_generate_runs_end_to_end_on_the_fixture(tmp_path) -> None:
    """Covers ``generate()`` itself, which no test touched until it failed on a T4.

    B-05: it allocated a 3.2 GiB block pool *per request*, OOM'd on every call, and the caught
    exception turned into a file of zeros. Both halves of that were invisible locally because the
    only caller was a script. This runs the whole path -- processor, chunked vision encoding,
    feature merge, compressor, paged decode -- on the 256M fixture at CPU/fp32.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    import torch
    from PIL import Image

    from bench.pipeline import generate
    from edgerag.cache.allocator import BlockAllocator
    from edgerag.cache.compressed import CompressedKVCache
    from edgerag.compress.fastv import FastVConfig
    from edgerag.core.loader import FIXTURE_MODEL
    from edgerag.core.model import load_from_hf
    from edgerag.core.spec import ModelSpec
    from edgerag.retrieval.corpus import CorpusDoc
    from edgerag.retrieval.trace import TraceEntry

    config_obj = transformers.AutoConfig.from_pretrained(FIXTURE_MODEL)
    # The *text* path is eager so it is bit-comparable with our decoder. The **vision tower is
    # not** -- it is HuggingFace's on both sides of every comparison here, so its attention
    # implementation cancels out, while eager costs +0.76 GiB of transient score matrix against
    # SDPA's +0.17 (measured). That is B-05's lesson applied to the suite: the peak that killed
    # the T4 quality run is the same peak that makes this file die on a loaded dev box (P-27).
    config_obj._attn_implementation = "eager"
    config_obj.text_config._attn_implementation = "eager"
    config_obj.vision_config._attn_implementation = "sdpa"
    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        FIXTURE_MODEL, config=config_obj, dtype=torch.float32
    )
    hf.eval()
    processor = transformers.AutoProcessor.from_pretrained(FIXTURE_MODEL)
    spec = ModelSpec.from_hf_config(FIXTURE_MODEL, config_obj)
    decoder = load_from_hf(spec, hf)

    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (480, 360), color=(200, 200, 200)).save(image_path, "JPEG")
    doc = CorpusDoc(
        doc_key="t:1:0", source="t", doc_id="1", page_no=0,
        image_path=str(image_path), width=480, height=360, text="total is 42", n_text_chars=11,
    )
    entry = TraceEntry(
        query_id="q1", question="What is the total?", answers=["42"],
        gold_doc_key=doc.doc_key, retrieved_doc_keys=[doc.doc_key], split="heldout", k=1,
    )

    cache = CompressedKVCache(
        spec, BlockAllocator(256, 16), torch.device("cpu"), torch.float32, score_layer=2
    )

    # Two requests through ONE cache: the reuse that B-05 got wrong.
    for _ in range(2):
        text, stats = generate(
            decoder, hf, processor, spec, entry, {doc.doc_key: doc},
            FastVConfig(keep_ratio=0.5, score_layer=2), "attention", 4,
            torch.device("cpu"), cache,
        )
        assert isinstance(text, str)
        assert stats["prefill_tokens"] > 0
        assert stats["ttft_s"] > 0
        assert stats["dropped_tokens"] > 0, "pruning did not drop anything"

    # The pool must not leak across requests -- that is what per-request allocation hid.
    cache.reset()
    cache.allocator.check_invariants()
