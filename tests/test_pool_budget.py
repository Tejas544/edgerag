"""Sizing the block pool from the memory budget, rather than choosing it and hoping.

The bug this exists to prevent is not a crash. The shipped default was 640 blocks -- a reasonable
number, arrived at as "enough for one ~7k request with room to grow" -- and against the measured
weights and activation it put the pipeline at 4.495 GiB while the README's headline said 4 GiB,
with the two figures a paragraph apart and nobody subtracting one from the other.

So the arithmetic is a function with tests rather than a constant in a docstring, and the property
that matters is the one an interviewer would check with a calculator: **weights + activation +
pool <= budget**, for whatever the weights actually turn out to be.
"""

from __future__ import annotations

import pytest

from edgerag.core.budget import (
    GIB,
    MemoryBudgetExceeded,
    plan_pool_for_budget,
)

#: SmolVLM2-2.2B: 24 layers, K and V, 32 KV heads, head dim 64, fp16. MHA, so no GQA saving.
KV_PER_TOKEN = 24 * 2 * 32 * 64 * 2

#: LM8+ViT4, measured on a T4 and ledger-confirmed to the byte (CONTEXT.md D24).
SHIP_WEIGHTS = 2_465_137_152

#: Measured on the serving path with chunked prefill at 512 (CONTEXT.md D26).
SHIP_ACTIVATION = 347_734_016


def test_the_kv_arithmetic_matches_the_documented_rate() -> None:
    """192 KiB/token is quoted throughout the README; if it drifts, every table above is wrong."""
    assert KV_PER_TOKEN == 192 * 1024


def test_the_plan_fits_the_budget_it_was_given() -> None:
    plan = plan_pool_for_budget(KV_PER_TOKEN, SHIP_WEIGHTS, SHIP_ACTIVATION, budget_gib=4.0)
    assert plan.fits
    assert plan.total_bytes <= 4 * GIB
    # And it is the *largest* such pool -- one more block must break it, or the budget is being
    # under-spent and the concurrency claim understated.
    one_more = plan.pool_bytes + plan.block_size * KV_PER_TOKEN
    assert SHIP_WEIGHTS + SHIP_ACTIVATION + one_more > 4 * GIB


def test_the_shipped_640_block_default_did_not_fit() -> None:
    """Pins the actual defect, so nobody restores 640 as a default without seeing this fail."""
    over = SHIP_WEIGHTS + SHIP_ACTIVATION + 640 * 16 * KV_PER_TOKEN
    assert over > 4 * GIB
    assert (over - 4 * GIB) / 1024**2 == pytest.approx(507, abs=1)


def test_a_budget_buys_a_prompt_length_not_an_unlimited_prompt() -> None:
    """The consequence that has to be stated, not the one discovered as an OOM mid-decode."""
    plan = plan_pool_for_budget(KV_PER_TOKEN, SHIP_WEIGHTS, SHIP_ACTIVATION, budget_gib=4.0)
    assert plan.max_prompt_tokens == (plan.num_blocks - plan.reserve_blocks) * plan.block_size
    # Sanity against the trace: the median request is ~6,758 tokens and fits; the p90 is ~7,795
    # and does not. Both halves of that are the finding.
    assert plan.max_prompt_tokens > 6_758
    assert plan.max_prompt_tokens < 7_795


def test_chunked_prefill_is_what_makes_the_budget_reachable() -> None:
    """D26's load-bearing claim: unchunked, no pool at all fits beside the ship weights.

    Chunked prefill costs p95 TTFT (D25) and buys 484 MiB of activation headroom. Without that
    headroom the 4 GiB budget cannot hold one median request, so the feature is not a latency
    tradeoff that happens to save memory -- it is load-bearing for the project's headline.
    """
    unchunked_activation = 855_051_776  # measured, same session, chunking off
    chunked = plan_pool_for_budget(KV_PER_TOKEN, SHIP_WEIGHTS, SHIP_ACTIVATION, budget_gib=4.0)
    unchunked = plan_pool_for_budget(
        KV_PER_TOKEN, SHIP_WEIGHTS, unchunked_activation, budget_gib=4.0
    )
    assert chunked.max_prompt_tokens > 6_758, "the median request must fit when chunked"
    assert unchunked.max_prompt_tokens < 6_758, "and must not fit when it is not"


def test_a_budget_too_small_for_one_block_is_refused() -> None:
    """Returning a zero-block pool would start a server that rejects every request.

    2.62 GiB is chosen to sit just above weights + activation (2.6197 GiB) and just below one more
    block, which is the only interesting boundary here -- a wildly small budget would pass this
    test without exercising the off-by-one.
    """
    with pytest.raises(MemoryBudgetExceeded, match="not one"):
        plan_pool_for_budget(KV_PER_TOKEN, SHIP_WEIGHTS, SHIP_ACTIVATION, budget_gib=2.62)


def test_a_bigger_budget_buys_proportionally_more_blocks() -> None:
    four = plan_pool_for_budget(KV_PER_TOKEN, SHIP_WEIGHTS, SHIP_ACTIVATION, budget_gib=4.0)
    eight = plan_pool_for_budget(KV_PER_TOKEN, SHIP_WEIGHTS, SHIP_ACTIVATION, budget_gib=8.0)
    # The extra 4 GiB is spent entirely on blocks, because weights and activation do not move.
    assert (eight.num_blocks - four.num_blocks) * eight.block_size * KV_PER_TOKEN == pytest.approx(
        4 * GIB, rel=0.01
    )


def test_the_plan_serialises_every_term_it_summed() -> None:
    """A budget table that cannot be re-added by its reader is a claim, not an accounting."""
    plan = plan_pool_for_budget(KV_PER_TOKEN, SHIP_WEIGHTS, SHIP_ACTIVATION, budget_gib=4.0)
    payload = plan.to_dict()
    assert (
        payload["weights_bytes"] + payload["activation_bytes"] + payload["pool_bytes"]
        == payload["total_bytes"]
    )
    assert payload["total_bytes"] <= payload["budget_bytes"]


def test_the_measured_activation_term_is_read_from_results_not_invented() -> None:
    """If the sweep is absent the caller must be told, not handed a plausible number."""
    from pathlib import Path

    from bench.serving import measured_activation_bytes

    assert measured_activation_bytes(chunked=True, path=Path("nope")) is None

    onfile = measured_activation_bytes(chunked=True)
    if onfile is not None:  # the sweep is committed, so this is the live path
        assert onfile == SHIP_ACTIVATION, "results/poisson_sweep.jsonl disagrees with D26"


def test_another_model_does_not_borrow_the_headline_model_s_activation_term() -> None:
    """Activation scales with the model; borrowing a term is a silent, plausible wrong answer.

    The 256M fixture sized from the 2.2B's 0.324 GiB reserves roughly ten times what it needs, and
    nothing fails -- the server starts and quietly serves shorter prompts than it could.
    """
    from bench.serving import measured_activation_bytes

    assert measured_activation_bytes(chunked=True, model_id="HuggingFaceTB/SmolVLM2-256M") is None


def test_the_unchunked_term_is_the_larger_one() -> None:
    """Guards the direction of D26 finding 1: chunking bounds the biggest tensor, so it is lower."""
    from bench.serving import measured_activation_bytes

    chunked = measured_activation_bytes(chunked=True)
    unchunked = measured_activation_bytes(chunked=False)
    if chunked is None or unchunked is None:
        pytest.skip("no sweep on file")
    assert unchunked > chunked
    assert (unchunked - chunked) / 1024**2 == pytest.approx(484, abs=2)
