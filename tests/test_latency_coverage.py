"""Which arms have single-session latency coverage -- the question D24 answered by hand.

The failure this guards is not a crash. It is a table that *looks* complete because eight rows
exist, while the ratios in it were divided across sessions carrying ~7.5% clock variance -- wider
than several of the gaps being reported. D24 caught that only by re-reading console output from
three separate runs.
"""

from __future__ import annotations

from scripts.colab_quant_ablation import arm_label
from scripts.latency_coverage import (
    ALL_ARMS,
    ANCHOR,
    by_session,
    finish_command,
    label_to_spec,
)


def _row(label: str, session: str, tok_s: float = 10.0) -> dict:
    arm, bits = label_to_spec(label)
    return {
        "label": label,
        "arm": arm,
        "bits": bits if bits is not None else 0,
        "session_id": session,
        "decode_tokens_per_s": {"p50": tok_s},
        "_file": "quant_latency_test.jsonl",
        "_label": label,
    }


def test_every_known_arm_label_round_trips() -> None:
    """``finish_command`` rebuilds CLI arguments from labels; a lossy parse builds a wrong run."""
    for label in ALL_ARMS:
        arm, bits = label_to_spec(label)
        if bits is None:  # mixed arms are self-naming
            assert arm == label
            continue
        assert arm_label(arm, bits) == label


def test_sessions_do_not_merge() -> None:
    rows = [_row("fp16", "aaa"), _row("LM@int8", "aaa"), _row("fp16", "bbb")]
    grouped = by_session(rows)
    assert set(grouped) == {"aaa", "bbb"}
    assert set(grouped["aaa"]) == {"fp16", "LM@int8"}


def test_unstamped_rows_are_not_treated_as_one_session() -> None:
    """They are *some* set of sessions. Calling them one is the laundering this script prevents."""
    rows = [{"label": "fp16", "_label": "fp16", "_file": "x", "decode_tokens_per_s": {"p50": 1}}]
    grouped = by_session(rows)
    assert "pre-stamping" in grouped
    assert "pre-stamping" not in {a for a in ALL_ARMS}


def test_the_finish_command_covers_everything_missing() -> None:
    """The point of emitting a command is that running it actually closes the gap."""
    missing = {"LM@int8", "LM+ViT@int8", "ViT@int8", "ViT@int4", "LM8+ViT4"}
    command = "\n".join(finish_command(missing))

    assert ANCHOR in command, "without an anchor the new rows have nothing to divide by"
    for label in missing:
        arm, bits = label_to_spec(label)
        assert arm in command, f"{label} needs arm {arm!r} in --arms"
        if bits is not None:
            assert str(bits) in command.split("--bits")[1], f"{label} needs {bits} in --bits"


def test_the_cross_product_extras_are_named_not_hidden() -> None:
    """An unexpected arm in the output otherwise reads as a resume bug rather than as arithmetic."""
    # ViT@int4 and LM@int8 together force both widths, which also re-measures LM@int4 and ViT@int8.
    command = "\n".join(finish_command({"ViT@int4", "LM@int8"}))
    assert "also re-measures" in command
    assert "LM@int4" in command
    assert "ViT@int8" in command


def test_a_single_missing_uniform_arm_still_emits_a_valid_bits_flag() -> None:
    """A mixed-arm-only gap leaves `--bits` with nothing to say; it must not emit an empty flag."""
    command = "\n".join(finish_command({"LM8+ViT4"}))
    assert "--bits" in command
    assert "--bits \n" not in command and "--bits  " not in command
    assert "LM8+ViT4" in command
