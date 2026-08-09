"""Tests for the benchmark harness itself.

The harness measures everything else, so it is the one component whose correctness cannot be
established by the thing it measures. Each test below pins one rule from ``00_FOUNDATIONS.md`` §4.
"""

from __future__ import annotations

import json
import random

import pytest

from bench.bench import (
    IncomparableRecordsError,
    JsonlWriter,
    TrialResult,
    _fake_generation,
    check_comparable,
    records_to_markdown,
    run_benchmark,
)
from bench.metrics import (
    HeldConstant,
    Percentiles,
    UntrustedDeviceError,
    _has_tensor_cores,
    _percentile,
    assert_device_trusted,
)


def _held() -> HeldConstant:
    return HeldConstant(
        model_id="fake/dry-run",
        dtype="float16",
        batch_size=1,
        prompt_tokens=512,
        max_new_tokens=8,
    )


def _fn(seed: int = 0):
    rng = random.Random(seed)
    return lambda: _fake_generation(new_tokens=8, rng=rng)


# --- percentiles --------------------------------------------------------------------------


def test_percentile_matches_linear_interpolation() -> None:
    ordered = [float(i) for i in range(1, 101)]  # 1..100
    assert _percentile(ordered, 50.0) == pytest.approx(50.5)
    assert _percentile(ordered, 0.0) == 1.0
    assert _percentile(ordered, 100.0) == 100.0


def test_percentile_single_sample() -> None:
    assert _percentile([7.0], 99.0) == 7.0


def test_p99_flagged_unreliable_below_100_samples() -> None:
    """A 'p99' over 20 samples is the max wearing a hat. The record has to say so."""
    assert Percentiles.of([float(i) for i in range(20)]).p99_is_reliable is False
    assert Percentiles.of([float(i) for i in range(150)]).p99_is_reliable is True


def test_percentiles_reject_empty() -> None:
    with pytest.raises(ValueError):
        Percentiles.of([])


# --- device trust gate (CONTEXT.md D4) ------------------------------------------------------


def test_gtx_16_series_reported_without_tensor_cores() -> None:
    """A GTX 1650 and a Tesla T4 both report compute capability 7.5. Only one has tensor cores."""
    assert _has_tensor_cores("NVIDIA GeForce GTX 1650", 7) is False
    assert _has_tensor_cores("NVIDIA GeForce GTX 1660 Ti", 7) is False
    assert _has_tensor_cores("Tesla T4", 7) is True


def test_untrusted_device_blocks_recording() -> None:
    """Neither CPU nor the local GTX 1650 may silently produce a publishable number."""
    with pytest.raises(UntrustedDeviceError, match="Refusing to record"):
        assert_device_trusted(allow_untrusted=False)


def test_untrusted_device_override_warns_and_stamps() -> None:
    with pytest.warns(UserWarning, match="NOT publishable"):
        info = assert_device_trusted(allow_untrusted=True)
    assert info.trusted is False


# --- the protocol the harness enforces ------------------------------------------------------


def test_fewer_than_five_trials_rejected() -> None:
    """§4 rule 4: a single number is not a result."""
    with pytest.raises(ValueError, match="not a result"):
        run_benchmark("x", _fn(), _held(), trials=1, allow_untrusted_device=True)


def test_trial_result_catches_step_count_mismatch() -> None:
    bad = TrialResult(
        ttft_s=0.01, decode_step_times_s=[0.001, 0.001], n_prompt_tokens=8, n_generated_tokens=5
    )
    with pytest.raises(ValueError, match="step times"):
        bad.validate()


def test_trial_result_catches_nonpositive_ttft() -> None:
    """A zero TTFT almost always means a missing sync(), not an infinitely fast prefill."""
    bad = TrialResult(
        ttft_s=0.0, decode_step_times_s=[0.001], n_prompt_tokens=8, n_generated_tokens=1
    )
    with pytest.raises(ValueError, match="missing a sync"):
        bad.validate()


def test_run_benchmark_produces_a_complete_record() -> None:
    with pytest.warns(UserWarning):
        rec = run_benchmark("dry", _fn(), _held(), warmup=2, trials=5, allow_untrusted_device=True)

    assert rec.trials == 5
    assert rec.warmup_iters == 2
    assert rec.trusted is False
    # TTFT and decode are aggregated separately -- §4 rule 7.
    assert rec.ttft["p50"] > 0
    assert rec.decode_tokens_per_s["p50"] > 0
    assert rec.decode_step_latency["n"] == 5 * 8
    assert rec.ttft["n"] == 5
    # Provenance is stamped on every record -- BUGS.md P-15.
    assert rec.device["name"]
    assert rec.held_constant["batch_size"] == 1


def test_record_is_json_serialisable() -> None:
    with pytest.warns(UserWarning):
        rec = run_benchmark("dry", _fn(), _held(), warmup=1, trials=5, allow_untrusted_device=True)
    json.dumps(rec.to_dict())  # must not raise


# --- incremental persistence (§3 rule 5) ----------------------------------------------------


def test_jsonl_writer_appends_incrementally(tmp_path) -> None:
    """A disconnect at hour 3 must not cost hours 1 and 2."""
    writer = JsonlWriter(tmp_path / "nested" / "out.jsonl")
    assert writer.read_all() == []

    with pytest.warns(UserWarning):
        for i in range(3):
            writer.append(
                run_benchmark(
                    f"r{i}", _fn(i), _held(), warmup=1, trials=5, allow_untrusted_device=True
                )
            )
            # Readable after every single append, not just at the end.
            assert len(writer.read_all()) == i + 1


# --- held-constant enforcement (§4 rule 5) --------------------------------------------------


def _rec_dict(name: str, **held_overrides) -> dict:
    held = _held().to_dict() | held_overrides
    return {"name": name, "held_constant": held}


def test_comparable_records_pass() -> None:
    check_comparable([_rec_dict("a"), _rec_dict("b")])


def test_incomparable_records_rejected() -> None:
    """The quiet failure: an ablation whose rows differ in batch size as well as the feature."""
    with pytest.raises(IncomparableRecordsError, match="batch_size"):
        check_comparable([_rec_dict("a"), _rec_dict("b", batch_size=8)])


def test_variable_under_test_can_be_ignored() -> None:
    check_comparable(
        [_rec_dict("a"), _rec_dict("b", batch_size=8)], ignore=["batch_size"]
    )


def test_single_record_is_trivially_comparable() -> None:
    check_comparable([_rec_dict("a")])


# --- markdown emitter -----------------------------------------------------------------------


def test_markdown_table_flags_untrusted_and_unreliable_p99() -> None:
    with pytest.warns(UserWarning):
        rec = run_benchmark("dry", _fn(), _held(), warmup=1, trials=5, allow_untrusted_device=True)
    md = records_to_markdown([rec.to_dict()])

    assert "| dry |" in md
    assert "**NO**" in md  # untrusted device flagged in the row
    assert "not publishable" in md.lower()
    assert "p99 computed from fewer than 100 samples" in md


def test_markdown_handles_no_records() -> None:
    assert "no records" in records_to_markdown([])


# --- workload provenance --------------------------------------------------------------------


def test_records_from_different_traces_are_incomparable() -> None:
    """The one incomparability a held-constant manifest cannot express.

    Two runs against different workloads report the same model, dtype, batch, and lengths. Only
    the trace fingerprint distinguishes them.
    """
    a = {**_rec_dict("a"), "workload_fingerprint": "aaaa1111"}
    b = {**_rec_dict("b"), "workload_fingerprint": "bbbb2222"}
    with pytest.raises(IncomparableRecordsError, match="different workload traces"):
        check_comparable([a, b])


def test_records_from_the_same_trace_are_comparable() -> None:
    a = {**_rec_dict("a"), "workload_fingerprint": "aaaa1111"}
    b = {**_rec_dict("b"), "workload_fingerprint": "aaaa1111"}
    check_comparable([a, b])


def test_fingerprint_is_stamped_onto_the_record() -> None:
    with pytest.warns(UserWarning):
        rec = run_benchmark(
            "dry",
            _fn(),
            _held(),
            warmup=1,
            trials=5,
            allow_untrusted_device=True,
            workload_fingerprint="94b148a0b9f5006e",
        )
    assert rec.workload_fingerprint == "94b148a0b9f5006e"
    assert "94b148a0b9f5006e" in records_to_markdown([rec.to_dict()])


# --- aggregate decode timing ------------------------------------------------------------------


def test_from_aggregate_spreads_decode_time_evenly() -> None:
    result = TrialResult.from_aggregate(
        ttft_s=0.5, decode_total_s=2.0, n_prompt_tokens=100, n_generated_tokens=8
    )
    result.validate()
    assert result.decode_timing == "aggregate"
    assert result.decode_step_times_s == [0.25] * 8
    assert result.decode_s == pytest.approx(2.0)
    assert result.end_to_end_s == pytest.approx(2.5)


def test_from_aggregate_rejects_zero_tokens() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TrialResult.from_aggregate(0.5, 2.0, 100, 0)


def test_aggregate_timing_is_flagged_in_the_markdown() -> None:
    """A flat per-step distribution must not be mistaken for a measured one."""
    fn = lambda: TrialResult.from_aggregate(0.01, 0.08, 512, 8)  # noqa: E731
    with pytest.warns(UserWarning):
        rec = run_benchmark("agg", fn, _held(), warmup=1, trials=5, allow_untrusted_device=True)
    assert rec.decode_timing == "aggregate"

    md = records_to_markdown([rec.to_dict()])
    assert "aggregate" in md
    assert "inter-token latency percentiles for those rows are not" in md


def test_mixing_timing_modes_within_one_record_is_rejected() -> None:
    """Pooling real per-step latencies with uniformly-filled ones yields a distribution that
    describes neither."""
    calls = {"n": 0}

    def alternating() -> TrialResult:
        calls["n"] += 1
        if calls["n"] % 2:
            return TrialResult.from_aggregate(0.01, 0.08, 512, 8)
        return _fake_generation(new_tokens=8, rng=random.Random(0))

    with pytest.warns(UserWarning), pytest.raises(ValueError, match="mixed decode timing"):
        run_benchmark("mix", alternating, _held(), warmup=0, trials=5, allow_untrusted_device=True)
