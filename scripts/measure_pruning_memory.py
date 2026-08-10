"""Phase 4 memory curve: KV reclaimed versus visual-token pruning ratio.

    python -m scripts.measure_pruning_memory

Memory only, exact arithmetic, so it runs on the local tier under ``CONTEXT.md`` D4. The
*quality* half of the curve needs the 2.2B model and therefore a T4 -- see
``scripts/colab_pruning_quality.py``. Publishing the memory axis without the quality axis would
be the dishonest half of this result, so the two are deliberately separate files with the
dependency stated.

``CONTEXT.md`` D15 asks for this curve in **MiB reclaimed**, not percent of tokens removed. The
token percentage understates the win, because pruning happens at layer *k* and every layer above
it -- 22 of 24 on the headline model -- stores the reduced set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean

from edgerag.core.loader import HEADLINE_MODEL, load_spec
from edgerag.core.spec import ModelSpec

MIB = 1024**2
GIB = 1024**3
GATE_PATH = Path("data/token_ratio_gate.json")


def reclaimed_bytes(
    spec: ModelSpec, total_tokens: int, visual_tokens: int, keep_ratio: float, score_layer: int
) -> dict[str, float]:
    """KV bytes saved for one request at a given pruning ratio."""
    per_token_per_layer = 2 * spec.n_kv_heads * spec.head_dim * 2  # K and V, fp16
    layers_above = spec.n_layers - score_layer

    dropped = round(visual_tokens * (1.0 - keep_ratio))
    saved = dropped * layers_above * per_token_per_layer
    baseline = total_tokens * spec.n_layers * per_token_per_layer

    return {
        "dropped_tokens": dropped,
        "kept_tokens": total_tokens - dropped,
        "baseline_bytes": baseline,
        "saved_bytes": saved,
        "saved_mib": saved / MIB,
        "remaining_mib": (baseline - saved) / MIB,
        "saved_fraction": saved / baseline if baseline else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 memory curve")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--score-layer", type=int, default=2)
    parser.add_argument(
        "--keep-ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.375, 0.25, 0.125],
    )
    args = parser.parse_args(argv)

    spec = load_spec(args.model)
    if not GATE_PATH.exists():
        print(f"missing {GATE_PATH}; run scripts.measure_token_ratio first", file=sys.stderr)
        return 1

    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    per_query = gate["per_query"][args.model]
    total = int(fmean(q["total_tokens"] for q in per_query))
    visual = int(fmean(q["image_tokens"] for q in per_query))

    layers_above = spec.n_layers - args.score_layer
    print(f"model {spec.model_id}")
    print(f"  {spec.kv_bytes_per_token() / 1024:.0f} KiB KV/token, {spec.n_layers} layers")
    print(f"  pruning at layer {args.score_layer}: "
          f"{layers_above}/{spec.n_layers} layers ({layers_above / spec.n_layers:.0%}) store "
          "the reduced set")
    print(f"  mean request: {total} tokens, {visual} visual ({visual / total:.1%})\n")

    print(f"{'keep':>6} {'visual kept':>12} {'dropped':>8} {'KV MiB':>9} "
          f"{'reclaimed':>10} {'saving':>8}")
    rows = []
    for ratio in args.keep_ratios:
        r = reclaimed_bytes(spec, total, visual, ratio, args.score_layer)
        rows.append({"keep_ratio": ratio, **r})
        print(f"{ratio:>6.3f} {int(visual * ratio):>12,} {r['dropped_tokens']:>8,} "
              f"{r['remaining_mib']:>9.0f} {r['saved_mib']:>10.0f} "
              f"{r['saved_fraction']:>7.1%}")

    # What the saving means against the project's actual ceiling.
    weights_int4 = ModelSpec.effective_weight_bytes(2_246_784_880, "int4")
    print(f"\n=== against the 4 GiB budget (INT4 weights = {weights_int4 / GIB:.2f} GiB) ===")
    for row in rows:
        kv = row["remaining_mib"] * MIB
        total_gib = (weights_int4 + kv) / GIB
        verdict = "fits" if total_gib <= 4.0 else "OVER"
        print(
            f"  keep {row['keep_ratio']:.3f}: weights + 1 request = {total_gib:.2f} GiB  {verdict}"
        )

    out = Path("results/pruning_memory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": spec.model_id,
                "score_layer": args.score_layer,
                "layers_above_cut": layers_above,
                "mean_total_tokens": total,
                "mean_visual_tokens": visual,
                "int4_weight_bytes": weights_int4,
                "curve": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    print("\nQuality axis requires the 2.2B model -- run scripts/colab_pruning_quality.py on a T4.")
    print("This curve is HALF the result; publishing it alone would be the dishonest half.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
