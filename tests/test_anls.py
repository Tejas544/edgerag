"""Tests for the ANLS scorer used by the Phase 4 quality curve.

Written *before* the curve is measured, deliberately. ``BUGS.md`` B-04 shipped because a function
whose only caller lived in a script was never executed by a test; `levenshtein` and `anls` are in
exactly that position, and a defect in either would not crash — it would produce a plausible,
wrong quality curve. That is worse than a crash, because the number would be published.
"""

from __future__ import annotations

import pytest

from scripts.colab_pruning_quality import anls, levenshtein

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
