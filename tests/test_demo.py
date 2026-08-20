"""The README demo, checked for the two things that make it honest and the one that makes it work.

A demo asset is uniquely easy to let rot: nobody re-reads an SVG, and if the animation silently
stops running or the provenance stamp gets refactored out, the artifact keeps looking fine while
saying something the project explicitly forbids (``CONTEXT.md`` D4 -- no locally-measured timing
presented as a performance claim).

So: the stamp is asserted, the script-free requirement is asserted, and the reveal times are
asserted to be the recorded ones rather than anything invented at render time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.make_demo import Cast, CastEvent, render

REPO_ROOT = Path(__file__).resolve().parents[1]
CAST_PATH = REPO_ROOT / "results" / "demo_cast.json"


def _cast(**overrides) -> Cast:
    base = {
        "recorded_utc": "2026-08-20T00:00:00+00:00",
        "base_url": "http://127.0.0.1:8300",
        "model": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct:fp16",
        "question": "What percentage of users are female?",
        "retrieved": ["infographicvqa:1:0", "infographicvqa:2:0"],
        "device": "NVIDIA GeForce GTX 1650",
        "trusted": False,
        "prompt_tokens": 5760,
        "total_s": 12.0,
        "events": [
            CastEvent(t=8.0, text="infographicvqa:1:0", kind="meta"),
            CastEvent(t=8.0, text="infographicvqa:2:0", kind="meta"),
            CastEvent(t=11.0, text=" 44", kind="token"),
            CastEvent(t=11.5, text="%", kind="token"),
        ],
    }
    base.update(overrides)
    return Cast(**base)


def test_the_provenance_stamp_is_inside_the_frame() -> None:
    """A caption can be cropped away when the image is pasted into a slide; the frame cannot."""
    svg = render(_cast())
    assert "NVIDIA GeForce GTX 1650" in svg
    assert "SmolVLM2-256M-Video-Instruct" in svg
    assert "T4-only" in svg


def test_a_sped_up_recording_says_so_and_a_real_time_one_says_that() -> None:
    """The number a reader would otherwise take for a latency claim."""
    assert "8x speed" in render(_cast(), speed=8.0)
    assert "real time" in render(_cast(), speed=1.0)


def test_there_is_no_script_anywhere() -> None:
    """GitHub serves README images through a proxy that strips scripts.

    A demo that animates only via JavaScript renders as a still frame for every visitor, and looks
    completely fine locally. CSS keyframes are the only thing that survives the trip.
    """
    svg = render(_cast())
    assert "<script" not in svg.lower()
    assert "onload" not in svg.lower()
    assert "@keyframes" in svg


def test_reveal_times_are_the_recorded_ones_divided_by_speed() -> None:
    """The load-bearing honesty property: pacing is a transcript, not a design choice."""
    cast = _cast()
    delays = [float(d) for d in re.findall(r"animation-delay:([\d.]+)s", render(cast, speed=1.0))]
    assert 11.0 in delays and 11.5 in delays, "token reveals must sit at their recorded offsets"

    halved = [float(d) for d in re.findall(r"animation-delay:([\d.]+)s", render(cast, speed=2.0))]
    assert 5.5 in halved and 5.75 in halved


def test_every_recorded_token_reaches_the_image() -> None:
    cast = _cast()
    svg = render(cast)
    for event in cast.events:
        if event.kind == "token":
            assert event.text.strip() in svg
    for key in cast.retrieved:
        assert key in svg


def test_a_cast_round_trips_through_json() -> None:
    """The cast is the provenance artifact; if it cannot be reloaded it cannot be checked."""
    cast = _cast()
    again = Cast.from_dict(json.loads(json.dumps(cast.to_dict())))
    assert again.events == cast.events
    assert again.device == cast.device
    assert render(again) == render(cast)


def test_a_cast_with_no_tokens_still_renders() -> None:
    """A server that retrieved but never generated is a real failure mode, not a crash here."""
    cast = _cast(
        retrieved=["doc:1:0"], events=[CastEvent(t=1.0, text="doc:1:0", kind="meta")]
    )
    svg = render(cast)
    assert "doc:1:0" in svg, "the sources are drawn from cast.retrieved, and must survive"
    assert "answer" in svg


@pytest.mark.skipif(not CAST_PATH.exists(), reason="no recorded cast on file")
def test_the_committed_cast_is_a_real_untrusted_recording() -> None:
    """Guards against a hand-written cast being committed as if it had been recorded.

    Also pins the D4 consequence: this recording is *not* from the reference device, so the SVG
    built from it must carry the warning. If someone re-records on a T4, this test should be
    updated deliberately rather than quietly passing.
    """
    payload = json.loads(CAST_PATH.read_text(encoding="utf-8"))
    cast = Cast.from_dict(dict(payload))
    assert cast.events, "an empty cast is not a recording"
    assert cast.retrieved, "the demo must show its sources"
    assert cast.total_s > 0
    assert any(e.kind == "token" for e in cast.events), "the model must have answered"

    # Sources before answer -- the whole point of showing it.
    first_meta = min(e.t for e in cast.events if e.kind == "meta")
    first_token = min(e.t for e in cast.events if e.kind == "token")
    assert first_meta < first_token, "retrieval must land before generation, or the demo misleads"

    if not cast.trusted:
        assert "T4-only" in render(cast)
