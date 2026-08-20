"""Record a real request against a running server, and render it as an animated SVG.

    python -m scripts.serve_rag --model HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
        --arm fp16 --budget-gib 0 --num-blocks 1536 --port 8300     # in one shell
    python -m scripts.make_demo --port 8300                          # in another

A reader gives a repository about ninety seconds. Everything this one has to show for itself is
currently a table, and a table does not answer the first question someone actually has, which is
*does this thing run*. So: one real question, streamed, with the pages it was answered from.

**Recorded, not mocked.** The cast is a transcript of an actual HTTP round trip -- the retrieved
document keys are whatever the index returned, the tokens are whatever the model emitted, and the
inter-token gaps are ``perf_counter`` deltas from the wire. ``results/demo_cast.json`` is kept
beside the SVG so the animation can be checked against its source, the same arrangement as
``scripts/make_plots.py`` and for the same reason: a hand-drawn demo is a demo that stops being
true and does it silently, in the one artifact people actually look at.

**This is not a performance measurement, and the SVG says so in the frame rather than in a
caption underneath it.** ``CONTEXT.md`` D4 keeps locally-measured timings out of `results/`, the
plots and the CV bullet, because a GTX 1650 has no tensor cores and its timings are
architecturally incomparable to the reference T4. A demo is a different claim -- *this path works
end to end* -- but it is one a reader can easily mistake for a speed claim, so the device and the
model are burned into the image where they cannot be cropped away.

No JavaScript: GitHub renders SVG through an image proxy that strips scripts but honours CSS
keyframes, which is what makes a pure-CSS cast the format that actually works in a README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
CAST_PATH = RESULTS / "demo_cast.json"
SVG_PATH = RESULTS / "demo.svg"

DEFAULT_QUESTION = "What percentage of users are female?"

# --- the palette, shared with scripts/make_plots.py so the README reads as one artifact ---------
SURFACE = "#11161a"
INK = "#e8ecef"
MUTED = "#7d8b96"
BLUE = "#5aa9ff"
ORANGE = "#ff9457"
GREEN = "#4ec99a"
WARN = "#ffcf5a"

CHAR_W = 8.4
LINE_H = 21.0
PAD = 22.0


@dataclass
class CastEvent:
    """One thing that appeared on screen, at the moment it appeared."""

    t: float
    text: str
    kind: str = "out"  # out | prompt | token | meta


@dataclass
class Cast:
    """A recorded session. Every field here came off the wire or out of `/health`."""

    recorded_utc: str
    base_url: str
    model: str
    question: str
    retrieved: list[str]
    device: str
    trusted: bool
    prompt_tokens: int
    total_s: float
    events: list[CastEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = [asdict(e) for e in self.events]
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> Cast:
        events = [CastEvent(**e) for e in payload.pop("events", [])]
        return Cast(**payload, events=events)


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def record(base_url: str, question: str, max_tokens: int, timeout: float) -> Cast:
    """Stream one question and timestamp everything that comes back.

    The first SSE chunk carries the retrieved page keys and nothing else, which is what makes the
    demo worth watching: it shows the answer arriving *after* its sources, in that order, so a
    viewer can see this is retrieval-augmented rather than a model reciting from weights.
    """
    health = _get_json(f"{base_url}/health", timeout)
    device = "unknown device"
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover -- provenance is not worth a crash
        pass

    body = json.dumps(
        {"messages": [{"role": "user", "content": question}],
         "max_tokens": max_tokens, "stream": True}
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )

    events: list[CastEvent] = []
    retrieved: list[str] = []
    started = time.perf_counter()

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"]
            now = time.perf_counter() - started
            if "retrieved" in delta:
                retrieved = list(delta["retrieved"])
                for key in retrieved:
                    events.append(CastEvent(t=now, text=key, kind="meta"))
            if delta.get("content"):
                events.append(CastEvent(t=now, text=delta["content"], kind="token"))

    total = time.perf_counter() - started
    # Prompt length is not on the streaming path -- SSE chunks carry no usage block -- so it comes
    # from a second, non-streaming call. Reported because "5,760 prompt tokens" is the number that
    # makes the 4 GiB budget mean something to someone skimming.
    prompt_tokens = 0
    try:
        probe = json.dumps(
            {"messages": [{"role": "user", "content": question}], "max_tokens": 1}
        ).encode()
        probe_request = urllib.request.Request(
            f"{base_url}/v1/chat/completions", data=probe,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(probe_request, timeout=timeout) as response:
            prompt_tokens = json.loads(response.read())["usage"]["prompt_tokens"]
    except (urllib.error.URLError, KeyError, TimeoutError):
        pass

    return Cast(
        recorded_utc=datetime.now(UTC).isoformat(),
        base_url=base_url,
        model=health.get("model", "unknown"),
        question=question,
        retrieved=retrieved,
        device=device,
        trusted="Tesla T4" in device,
        prompt_tokens=prompt_tokens,
        total_s=total,
        events=events,
    )


def _text(x: float, y: float, content: str, fill: str, *, weight: str = "400",
          opacity: str = "1", extra: str = "") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-weight="{weight}" '
        f'opacity="{opacity}"{extra}>{escape(content)}</text>'
    )


def render(cast: Cast, width_chars: int = 92, speed: float = 1.0) -> str:
    """Turn a recorded cast into a self-contained, looping, script-free SVG.

    Reveal times are the recorded ones divided by ``speed``; the loop then pauses on the finished
    frame so the answer is readable rather than flashing past. Every timing visible here is the
    one that was recorded, which is the only reason it is allowed to be shown at all.
    """
    lines: list[str] = []
    token_events = [e for e in cast.events if e.kind == "token"]
    first_token_t = token_events[0].t if token_events else cast.total_s
    retrieved_t = min((e.t for e in cast.events if e.kind == "meta"), default=0.0)

    y = PAD + LINE_H
    prompt = 'curl -N localhost:8000/v1/chat/completions -d \'{"stream":true,...}\''
    lines.append(_text(PAD, y, "$ ", GREEN, weight="700"))
    lines.append(_text(PAD + CHAR_W * 2, y, prompt, INK))

    y += LINE_H * 1.6
    lines.append(_text(PAD, y, f"? {cast.question}", ORANGE, weight="700"))

    y += LINE_H * 1.6
    delay = retrieved_t / speed
    lines.append(
        f'<g opacity="0" style="animation:r var(--d) steps(1,end) infinite;'
        f'animation-delay:{delay:.2f}s">'
        + _text(PAD, y, f"retrieved {len(cast.retrieved)} pages", MUTED)
        + "</g>"
    )
    for i, key in enumerate(cast.retrieved):
        y += LINE_H
        lines.append(
            f'<g opacity="0" style="animation:r var(--d) steps(1,end) infinite;'
            f'animation-delay:{delay + 0.05 * i:.2f}s">'
            + _text(PAD + CHAR_W * 2, y, f"- {key}", BLUE)
            + "</g>"
        )

    y += LINE_H * 1.7
    lines.append(
        f'<g opacity="0" style="animation:r var(--d) steps(1,end) infinite;'
        f'animation-delay:{first_token_t / speed:.2f}s">'
        + _text(PAD, y, "answer", MUTED)
        + "</g>"
    )

    # Tokens reveal one at a time, each at the offset it actually arrived at.
    y += LINE_H
    x = PAD + CHAR_W * 2
    for event in token_events:
        lines.append(
            f'<g opacity="0" style="animation:r var(--d) steps(1,end) infinite;'
            f'animation-delay:{event.t / speed:.2f}s">'
            + _text(x, y, event.text, INK, weight="700")
            + "</g>"
        )
        x += CHAR_W * len(event.text)

    y += LINE_H * 1.8
    footer = (
        f"{cast.prompt_tokens:,} prompt tokens from {len(cast.retrieved)} retrieved pages"
        if cast.prompt_tokens
        else f"{len(cast.retrieved)} retrieved pages"
    )
    lines.append(_text(PAD, y, footer, MUTED))

    # The stamp. Inside the frame, not in a caption -- a caption does not survive being cropped
    # into a slide deck, and this is the sentence that stops a demo being read as a benchmark.
    y += LINE_H * 1.35
    pace = "real time" if abs(speed - 1.0) < 1e-9 else f"{speed:g}x speed"
    stamp = f"{cast.device} - {cast.model.split('/')[-1]} - recorded {cast.total_s:.0f}s, {pace}"
    lines.append(_text(PAD, y, stamp, WARN))
    y += LINE_H
    lines.append(
        _text(PAD, y, "demonstrates the path, not the speed - every benchmark here is T4-only",
              MUTED)
    )

    height = y + PAD
    width = width_chars * CHAR_W + PAD * 2
    loop = max(cast.total_s / speed + 3.2, 4.0)
    # A 0.4 s fade expressed as a percentage of the loop, because CSS keyframe stops are
    # percentages and the loop length varies with the recording.
    fade_pct = 100 * 0.4 / loop

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" \
width="{width:.0f}" height="{height:.0f}" font-family="ui-monospace,'SF Mono','Cascadia Mono',\
Menlo,Consolas,monospace" font-size="13.5">
  <style>
    :root {{ --d: {loop:.2f}s; }}
    text {{ white-space: pre; }}
    @keyframes r {{
      0% {{ opacity: 0 }}
      {fade_pct:.3f}% {{ opacity: 1 }}
      100% {{ opacity: 1 }}
    }}
  </style>
  <rect width="100%" height="100%" rx="10" fill="{SURFACE}"/>
  <rect x="0.5" y="0.5" width="{width - 1:.0f}" height="{height - 1:.0f}" rx="10" fill="none" \
stroke="#243038"/>
  {"".join(lines)}
</svg>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record and render the README demo")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="divide recorded delays by this; 1.0 is real time")
    parser.add_argument("--render-only", action="store_true",
                        help="re-render results/demo_cast.json without a server")
    args = parser.parse_args(argv)

    if args.render_only:
        if not CAST_PATH.exists():
            print(f"no cast on file at {CAST_PATH}; run without --render-only", file=sys.stderr)
            return 1
        cast = Cast.from_dict(json.loads(CAST_PATH.read_text(encoding="utf-8")))
    else:
        base = f"http://{args.host}:{args.port}"
        try:
            cast = record(base, args.question, args.max_tokens, args.timeout)
        except urllib.error.URLError as exc:
            print(f"no server at {base}: {exc.reason}\n\nStart one first:\n"
                  "    python -m scripts.serve_rag --model "
                  "HuggingFaceTB/SmolVLM2-256M-Video-Instruct \\\n"
                  f"        --arm fp16 --budget-gib 0 --num-blocks 1536 --port {args.port}",
                  file=sys.stderr)
            return 1
        if not cast.events:
            print("the server streamed nothing -- not writing a cast of an empty run",
                  file=sys.stderr)
            return 1
        RESULTS.mkdir(parents=True, exist_ok=True)
        CAST_PATH.write_text(json.dumps(cast.to_dict(), indent=2), encoding="utf-8")
        print(f"recorded {len(cast.events)} events in {cast.total_s:.2f}s -> {CAST_PATH.name}")

    SVG_PATH.write_text(render(cast, speed=args.speed), encoding="utf-8")
    print(f"wrote {SVG_PATH.relative_to(REPO_ROOT)}")
    if not cast.trusted:
        print(f"  stamped '{cast.device}' -- this is a demo, not a measurement (CONTEXT.md D4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
