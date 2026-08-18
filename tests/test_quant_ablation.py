"""The Phase 6 ablation runner, exercised on CPU before it is trusted on a T4.

``BUGS.md`` B-05 is the reason this file exists. A script written for a T4 and executed nowhere
else came back with a complete, well-formed results file of zeros -- indistinguishable from a real
measurement showing catastrophic quality loss. The prevention was never "be careful"; it was to
run the real path on the 256M fixture, on CPU, in the suite.

So the end-to-end test below drives the actual ``build_arm`` / ``measure_arm`` the T4 will call,
with a quantized decoder *and* a quantized vision tower, and asserts the record it produces
contains measurements rather than defaults.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from scripts import measure_memory_ledger
from scripts.colab_quant_ablation import (
    ARMS,
    build_arm,
    ledger_prediction,
    measure_arm,
    resident_bytes,
    summarise,
)


def test_the_two_scripts_agree_on_what_an_arm_is() -> None:
    """The ledger predicts per arm and this script checks per arm -- against the same definition.

    If ``LM+ViT`` meant "language and vision" in one file and "language, vision and connector" in
    the other, the cross-check would report a 40 MiB ledger error every run and the ledger would be
    "fixed" to match a bug.
    """
    ours = {name: parts for name, parts in ARMS.items() if name != "fp16"}
    theirs = {name: parts for name, parts in measure_memory_ledger.ARMS.items() if name != "none"}
    assert ours == theirs


def test_resident_bytes_counts_buffers_as_well_as_parameters() -> None:
    """A quantized layer keeps its weights in *buffers*; counting only parameters reports zero."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(10, 10, dtype=torch.float16))
            self.register_buffer("packed", torch.zeros(10, 5, dtype=torch.uint8))

    assert resident_bytes(Tiny()) == 10 * 10 * 2 + 10 * 5


def test_ledger_lookup_finds_the_arm_and_maps_fp16_to_the_ledger_name() -> None:
    """The ledger calls the unquantized row ``none``; this script calls it ``fp16``."""
    predicted = ledger_prediction("fp16", 16)
    if predicted is None:
        pytest.skip("results/memory_ledger.json has not been generated")
    assert predicted > 4 * 1024**3, "the fp16 checkpoint is over 4 GiB -- that is the whole point"
    assert ledger_prediction("LM+ViT", 4) < predicted


def test_ledger_lookup_returns_none_for_an_arm_that_was_not_computed() -> None:
    """None means 'no prediction', and the runner prints that rather than inventing a delta."""
    assert ledger_prediction("nonexistent-arm", 4) is None


def test_summarise_refuses_to_call_an_empty_run_a_result(tmp_path) -> None:
    """B-05 in one assertion: a file with no measurements must exit non-zero."""
    empty = tmp_path / "quant_ablation.jsonl"
    empty.write_text("", encoding="utf-8")
    assert summarise(empty, 40) == 1


def test_summarise_reports_speed_relative_to_the_fp16_row(tmp_path, capsys) -> None:
    path = tmp_path / "quant_ablation.jsonl"
    rows = [
        {
            "arm": "fp16", "bits": 16, "weight_gib": 4.185, "peak_allocated_bytes": 6 * 1024**3,
            "decode_tokens_per_s": {"p50": 26.0}, "ttft_s": {"p50": 3.7},
            "anls": 0.44, "n_scored": 40,
        },
        {
            "arm": "LM+ViT", "bits": 4, "weight_gib": 1.546, "peak_allocated_bytes": 3 * 1024**3,
            "decode_tokens_per_s": {"p50": 13.0}, "ttft_s": {"p50": 4.1},
            "anls": 0.41, "n_scored": 40,
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    assert summarise(path, 40) == 0
    out = capsys.readouterr().out
    assert "0.50x" in out, "the INT4 row is half the speed and the table must say so"


def test_summarise_flags_a_row_scored_on_a_thin_subset(tmp_path, capsys) -> None:
    """A mean over 3 of 40 requests is a different number wearing the same column heading."""
    path = tmp_path / "quant_ablation.jsonl"
    path.write_text(
        json.dumps(
            {
                "arm": "LM", "bits": 4, "weight_gib": 1.9, "peak_allocated_bytes": 0,
                "decode_tokens_per_s": {"p50": 10.0}, "ttft_s": {"p50": 1.0},
                "anls": 0.9, "n_scored": 3,
            }
        ),
        encoding="utf-8",
    )
    summarise(path, 40)
    assert "biased subset" in capsys.readouterr().err


# --- the real path, on the fixture --------------------------------------------------------------


@pytest.mark.slow
def test_one_arm_runs_end_to_end_on_the_fixture(tmp_path) -> None:
    """Build a quantized arm and measure it, on CPU, exactly as the T4 run will.

    ``LM+ViT`` deliberately: it is the only arm that exercises both halves of the wiring -- the
    ``load_from_hf`` flag for our decoder and ``quantize_module_`` for HuggingFace's tower -- and
    the tower half has no other end-to-end coverage.

    Group 64 because the fixture's hidden size is 576, which 128 does not divide. Any vision layer
    that 64 does not divide either is skipped as ragged, which is itself the behaviour under test:
    the run must complete rather than raise.
    """
    pytest.importorskip("transformers")
    from PIL import Image

    from edgerag.core.loader import FIXTURE_MODEL
    from edgerag.retrieval.corpus import CorpusDoc
    from edgerag.retrieval.trace import TraceEntry

    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (480, 360), color=(200, 200, 200)).save(image_path, "JPEG")
    doc = CorpusDoc(
        doc_key="t:1:0", source="t", doc_id="1", page_no=0, image_path=str(image_path),
        width=480, height=360, text="total is 42", n_text_chars=11,
    )
    entry = TraceEntry(
        query_id="q1", question="What is the total?", answers=["42"], gold_doc_key=doc.doc_key,
        retrieved_doc_keys=[doc.doc_key], split="heldout", k=1,
    )

    lm, decoder = build_arm(
        FIXTURE_MODEL, "LM+ViT", bits=4, group_size=64, device="cpu", dtype=torch.float32
    )
    try:
        # The tower must have been replaced in place, or the ViT half of every arm is a no-op that
        # still reports a saving.
        from edgerag.core.linear import QuantLinear

        inner = lm.model.model
        assert any(isinstance(m, QuantLinear) for m in inner.vision_model.modules())
        assert any(isinstance(m, QuantLinear) for m in decoder.modules())

        measured = measure_arm(
            lm, decoder, [entry], {doc.doc_key: doc},
            max_new_tokens=2, num_blocks=256, trials=1,
        )
    finally:
        del decoder, lm

    assert measured["prompt_tokens"] > 0, "prefill length must be recorded, not defaulted"
    assert measured["n_scored"] == 1
    assert measured["n_oom"] == 0
    assert measured["ttft_s"]["p50"] > 0
    assert measured["decode_tokens_per_s"] is not None
    assert measured["decode_tokens_per_s"]["p50"] > 0


# --- resumability must not enshrine an inadequate row (BUGS.md B-10) ---------------------------


def test_a_thinner_row_does_not_count_as_done(tmp_path) -> None:
    """The trap that produced a 2-query fp16 baseline for a 40-query table.

    A row exists for the arm, so identity-based resuming skips it -- and it was measured to a
    weaker standard than the run now being asked for.
    """
    from scripts.colab_quant_ablation import completed_arms

    path = tmp_path / "quant_ablation.jsonl"
    path.write_text(
        json.dumps({"arm": "fp16", "bits": 16, "n_requested": 2, "trials": 1}),
        encoding="utf-8",
    )
    assert completed_arms(path, n_queries=40, trials=3) == set()
    assert completed_arms(path, n_queries=2, trials=1) == {("fp16", 16)}


def test_a_row_measured_more_thoroughly_still_counts_as_done(tmp_path) -> None:
    """Resuming must not re-measure an arm that already exceeds what was asked for."""
    from scripts.colab_quant_ablation import completed_arms

    path = tmp_path / "quant_ablation.jsonl"
    path.write_text(
        json.dumps({"arm": "LM", "bits": 4, "n_requested": 60, "trials": 5}), encoding="utf-8"
    )
    assert completed_arms(path, n_queries=40, trials=3) == {("LM", 4)}


def test_rows_predating_the_fields_are_treated_as_adequate(tmp_path) -> None:
    """Nothing to compare against, and refusing to resume would be worse than trusting them."""
    from scripts.colab_quant_ablation import completed_arms

    path = tmp_path / "quant_ablation.jsonl"
    path.write_text(json.dumps({"arm": "LM", "bits": 8}), encoding="utf-8")
    assert completed_arms(path, n_queries=40, trials=3) == {("LM", 8)}


def test_summarise_prefers_the_latest_measurement_of_an_arm(tmp_path, capsys) -> None:
    """Append-only + re-measure means duplicates; the newer row has to win or the fix is moot."""
    path = tmp_path / "quant_ablation.jsonl"
    stale = {
        "arm": "fp16", "bits": 16, "weight_gib": 4.185, "peak_allocated_bytes": 0,
        "decode_tokens_per_s": {"p50": 14.0}, "ttft_s": {"p50": 2.6},
        "anls": 0.8889, "n_scored": 2,
    }
    fresh = {**stale, "anls": 0.4378, "n_scored": 40, "decode_tokens_per_s": {"p50": 12.0}}
    path.write_text(json.dumps(stale) + "\n" + json.dumps(fresh) + "\n", encoding="utf-8")

    assert summarise(path, 40) == 0
    out = capsys.readouterr().out
    assert "0.4378" in out and "0.8889" not in out, "the superseded row must not be the one shown"
    assert "superseded" in out
