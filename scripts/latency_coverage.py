"""Which quantization arms have latency measured *in one session*, and what it takes to finish.

    python -m scripts.latency_coverage

``CONTEXT.md`` D24 finding 3 is the reason this exists. Decode throughput compared across Colab
sessions carries ~7.5% clock variance, which is wider than several of the differences the table is
trying to show -- so a ``vs fp16`` ratio is only meaningful between rows sharing a ``session_id``.
The table has been assembled from three sessions, resolved by hand from console output, and
partially re-run twice; what has never existed is a way to *ask the files* which arms are covered.

That is the whole job here: read every ``quant_latency*.jsonl`` in ``results/``, group by session,
and answer three questions a T4 session should not have to guess at.

* Which session has the widest coverage, and does it carry an ``fp16`` anchor? Without an anchor
  the rows are absolute numbers with nothing to divide by, and D24's whole finding is a ratio.
* Which arms are still missing from it?
* What is the exact command that would close the gap?

Reads only; runs anywhere; no GPU and no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.colab_quant_ablation import MIXED_ARMS, arm_label

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

#: The eight rows D24's throughput column is supposed to contain.
ALL_ARMS: tuple[str, ...] = (
    "fp16",
    "LM@int8", "LM+ViT@int8", "ViT@int8",
    "LM@int4", "LM+ViT@int4", "ViT@int4",
    "LM8+ViT4",
)

ANCHOR = "fp16"


def label_to_spec(label: str) -> tuple[str, int | None]:
    """``"LM@int8"`` -> ``("LM", 8)``. Mixed arms carry their own widths and report ``None``."""
    if label in MIXED_ARMS:
        return label, None
    if label == ANCHOR:
        return ANCHOR, 16
    arm, _, bits = label.partition("@int")
    return arm, int(bits)


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Every latency record on file, stamped with which file it came from."""
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_file"] = path.name
            row["_label"] = row.get("label") or arm_label(row["arm"], row["bits"])
            rows.append(row)
    return rows


def by_session(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """session id -> label -> the latest record for it.

    Records written before session stamping existed collapse into one ``pre-stamping`` bucket,
    which is exactly right: they are known to be *some* set of sessions and cannot be treated as
    one. Reporting them as a session would launder the thing this script exists to detect.
    """
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row.get("session_id", "pre-stamping")][row["_label"]] = row
    return dict(grouped)


def finish_command(missing: set[str], out_name: str = "quant_latency_finish.jsonl") -> list[str]:
    """The invocation that would measure ``missing`` plus an anchor, in one session.

    ``--arms`` and ``--bits`` form a cross product in the runner, so asking for two arms at two
    widths measures four cells. The extras are named rather than hidden -- an unexpected arm
    appearing in the output otherwise reads as a bug in the resume logic.
    """
    uniform = {label_to_spec(m) for m in missing if m not in MIXED_ARMS and m != ANCHOR}
    arms = sorted({arm for arm, _ in uniform})
    widths = sorted({bits for _, bits in uniform if bits is not None}, reverse=True)
    mixed = sorted({m for m in missing if m in MIXED_ARMS})

    requested = [ANCHOR, *arms, *mixed]
    produced = {ANCHOR, *mixed} | {arm_label(a, b) for a in arms for b in widths}
    extras = sorted(produced - missing - {ANCHOR})

    lines = [
        "    python -m scripts.colab_quant_ablation --drive /content/drive/MyDrive/edgerag \\",
        f"        --out-name {out_name} \\",
        f"        --arms {' '.join(requested)} \\",
        f"        --bits {' '.join(str(b) for b in widths) or '8'} "
        "--n-queries 2 --trials 5",
    ]
    if extras:
        lines.append("")
        lines.append(f"  (also re-measures {', '.join(extras)} -- --arms x --bits is a cross")
        lines.append("   product. Harmless, and it widens the single-session table.)")
    return lines


def report(paths: list[Path]) -> int:
    rows = load_rows(paths)
    if not rows:
        print("no latency records on file. Run scripts.colab_quant_ablation with --out-name "
              "quant_latency*.jsonl on a T4.", file=sys.stderr)
        return 1

    sessions = by_session(rows)
    print(f"read {len(rows)} record(s) from {len(paths)} file(s)\n")

    ranked = sorted(sessions.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for session, labels in ranked:
        anchored = "anchored" if ANCHOR in labels else "**NO fp16 ANCHOR**"
        files = sorted({row["_file"] for row in labels.values()})
        print(f"  session {session}  ({len(labels)}/{len(ALL_ARMS)} arms, {anchored})")
        print(f"    file(s): {', '.join(files)}")
        base = labels.get(ANCHOR, {}).get("decode_tokens_per_s")
        for label in ALL_ARMS:
            row = labels.get(label)
            if row is None:
                continue
            decode = row.get("decode_tokens_per_s")
            speed = decode["p50"] if decode else float("nan")
            ratio = f"{speed / base['p50']:.3f}x" if base and decode else "--"
            print(f"      {label:>12}  {speed:>6.2f} tok/s  {ratio:>8}")
        print()

    best_session, best_labels = ranked[0]
    covered = {label for label in best_labels if label in ALL_ARMS}
    missing = set(ALL_ARMS) - covered

    if not missing and ANCHOR in covered:
        print(f"  COMPLETE: session {best_session} covers all {len(ALL_ARMS)} arms with an "
              "anchor.\n  D24's cross-session caveat can be retired.")
        return 0

    print(f"  Widest single session is {best_session} at {len(covered)}/{len(ALL_ARMS)} arms.")
    print(f"  MISSING: {', '.join(sorted(missing)) or 'nothing'}")
    if ANCHOR not in covered:
        print(f"  It has no {ANCHOR} anchor, so even its covered arms have no ratio to report.")

    # Arms measured somewhere, but not in the best session -- the trap D24 fell into.
    elsewhere = {label for labels in sessions.values() for label in labels} & missing
    if elsewhere:
        print(f"\n  {', '.join(sorted(elsewhere))} exist in *other* sessions. Those numbers are")
        print("  real but not comparable to the ones above -- ~7.5% cross-session clock variance")
        print("  (D24 finding 3) is wider than several of the gaps in this table.")

    print("\n  To close it, with an anchor, in one session:\n")
    print("\n".join(finish_command(missing)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-session latency coverage (no GPU)")
    parser.add_argument(
        "--glob", default="quant_latency*.jsonl",
        help="which results files to read; the quality table (quant_ablation.jsonl) is excluded "
             "by default because its rows were measured at a different --n-queries",
    )
    args = parser.parse_args(argv)

    paths = sorted(RESULTS.glob(args.glob))
    if not paths:
        print(f"no files matching {args.glob!r} in {RESULTS}", file=sys.stderr)
        return 1
    return report(paths)


if __name__ == "__main__":
    sys.exit(main())
