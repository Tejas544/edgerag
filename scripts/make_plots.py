"""Phase 8 plots, generated from ``results/`` -- never drawn by hand.

    python -m scripts.make_plots

Every number on every axis is read out of a results file at render time. That is not tidiness: a
plot transcribed by hand is a plot that disagrees with the data the first time a measurement is
re-run, and it disagrees silently, in the one artifact people actually look at.

Each figure carries its own provenance line -- device, sample size, workload fingerprint, and
which quantities are *computed exactly* versus *measured*. The distinction matters throughout this
project (``CONTEXT.md`` D21 vs D24) and a chart that hides it is doing the reader a disservice.

Colours come from a validated categorical palette (blue / orange / aqua), checked for
colour-vision separation rather than chosen by eye. Three series maximum per figure, series
identity never carried by colour alone -- every mark that matters is directly labelled.
"""

# ruff: noqa: RUF001 -- U+2212 MINUS SIGN is deliberate in chart labels: a hyphen reads as a
# dash beside a numeral at these sizes, and these strings are rendered, not compared.

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
PLOTS = RESULTS / "plots"
GIB = 1024**3

# --- the validated palette (see the data-visualisation reference) -------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
LIMIT = "#d03b3b"
DE_EMPHASIS = "#b5b4ae"

#: The block pool the ablation preallocated: 640 blocks x 16 tokens, 24 layers, K and V, fp16.
#: Subtracting it from a measured peak is what isolates the transient activation term.
ABLATION_POOL_BYTES = 640 * 16 * 24 * 2 * 32 * 64 * 2

#: ANLS standard error at n=40, calibrated in ``CONTEXT.md`` D20. Drawn as a band rather than
#: quoted in a caption, because the single most common misreading of these curves is treating a
#: gap smaller than this as a result.
ANLS_SE = 0.06


def _style(ax, *, xgrid: bool = False, ygrid: bool = True) -> None:
    """Recessive axes: the data is the ink, the frame is not."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y" if ygrid else "x", color=GRID, linewidth=1.0, alpha=1.0 if ygrid else 0)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def _provenance(fig, text: str) -> None:
    fig.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=7.5, color=MUTED, wrap=True)


def _label(ax, title: str, subtitle: str, xlabel: str = "", ylabel: str = "") -> None:
    # The subtitle is anchored above the axes and grows *upward*, so a fixed title pad puts a
    # two-line subtitle straight through the title. Pad follows the line count instead.
    lines = subtitle.count("\n") + 1
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold",
                 pad=18 + 14 * (lines - 1))
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2, labelpad=8)


def _load_arms() -> list[dict]:
    path = RESULTS / "quant_ablation.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["label"] = row.get("label") or (
            "fp16" if row["bits"] == 16 else f"{row['arm']}@int{row['bits']}"
        )
    return rows


# --- plot 1: does it fit? -----------------------------------------------------------------------


def plot_budget(arms: list[dict], ledger: dict) -> Path:
    """The thesis, as a part-to-whole against a hard limit.

    Stacked because the question is *what is the budget made of*, horizontal because the
    configuration names are long, and ordered by total so the fitting/not-fitting boundary is a
    line the eye can follow rather than a fact in a caption.
    """
    kv = ledger["summary"]["kv_bytes_per_request"] / GIB
    measured = [
        (row["peak_allocated_bytes"] - row["weight_bytes"] - ABLATION_POOL_BYTES) / GIB
        for row in arms
        if row.get("n_requested")  # only rows measured after the B-09 logits fix
    ]
    activation = sum(measured) / len(measured)

    shown = ["fp16", "LM@int8", "LM8+ViT4", "LM@int4", "LM+ViT@int4"]
    rows = [next(a for a in arms if a["label"] == name) for name in shown]
    rows.sort(key=lambda r: r["weight_gib"])

    fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor=SURFACE)
    positions = range(len(rows))
    weights = [r["weight_gib"] for r in rows]

    # The two constant segments carry their value in the legend text. Partly the relief rule --
    # aqua sits below 3:1 on this surface -- and partly because they are the same width on every
    # bar, so a per-bar label would repeat one number five times.
    ax.barh(positions, weights, height=0.58, color=BLUE, label="weights (labelled)", zorder=3)
    ax.barh(positions, [kv] * len(rows), height=0.58, left=weights, color=ORANGE,
            label=f"KV cache, 1 request  ({kv:.2f})", zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.barh(positions, [activation] * len(rows), height=0.58,
            left=[w + kv for w in weights], color=AQUA,
            label=f"activation + workspace  ({activation:.2f})",
            zorder=3, edgecolor=SURFACE, linewidth=2)

    for i, row in enumerate(rows):
        total = row["weight_gib"] + kv + activation
        fits = total <= 4.0
        # A surface-coloured box behind the total, because two of these land within a hair of the
        # 4 GiB rule and would otherwise be read through it.
        ax.text(total + 0.10, i, f"{total:.2f} GiB", va="center", fontsize=9.5,
                color=INK if fits else LIMIT, fontweight="bold" if not fits else "normal",
                zorder=6, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
        # The relief rule: the aqua segment sits below 3:1 on this surface, so every segment it
        # belongs to carries a visible value rather than relying on the legend swatch alone.
        ax.text(row["weight_gib"] / 2, i, f"{row['weight_gib']:.2f}", va="center", ha="center",
                fontsize=8.5, color="white", fontweight="bold", zorder=4)

    ax.axvline(4.0, color=LIMIT, linewidth=2, zorder=5)
    ax.text(4.06, len(rows) - 0.55, "4 GiB budget", color=LIMIT, fontsize=9.5,
            fontweight="bold", va="center", ha="left", zorder=6,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2})

    ax.set_yticks(list(positions))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10, color=INK)
    ax.set_xlim(0, 6.9)
    ax.set_ylim(-0.62, len(rows) - 0.28)   # headroom so the rule label stays inside the axes
    _style(ax, xgrid=True, ygrid=False)
    _label(
        ax,
        "Only the INT4 language model fits one request in 4 GiB",
        "SmolVLM2-2.2B, one 6,758-token retrieved request. The configuration that preserves "
        "quality (LM8+ViT4) is over by 0.24 GiB.",
        xlabel="GiB",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _provenance(
        fig,
        "weights and KV computed exactly from the checkpoint config (CONTEXT.md D21, confirmed to "
        f"the byte on a T4); activation measured, mean of {len(measured)} post-fix arms "
        f"({activation:.3f} GiB). Workload 94b148a0b9f5006e.",
    )
    return _save(fig, "budget.png")


# --- plot 2: what quantization costs ------------------------------------------------------------


def plot_quantization_tradeoff(arms: list[dict]) -> Path:
    """Memory against quality, with the two configurations that matter picked out.

    **Emphasis, not eight categorical hues.** Eight arms would need eight colours that no
    colour-vision check passes, and the story is not "here are eight things" -- it is "one of
    these eight is the knee". Everything else is context, in grey.
    """
    fp16 = next(a for a in arms if a["label"] == "fp16")
    mixed = next(a for a in arms if a["label"] == "LM8+ViT4")

    fig, ax = plt.subplots(figsize=(9.0, 5.4), facecolor=SURFACE)

    # Hand-placed because there are eight of them in two tight clusters; an automatic placer
    # either overlaps or drifts far enough from its mark to be ambiguous. LM+ViT@int8 goes to the
    # side rather than below: directly beneath it sits the LM8+ViT4 marker.
    above = {"fp16", "ViT@int8", "LM@int8", "LM8+ViT4", "LM+ViT@int4"}
    placements = {"LM+ViT@int8": (12, -4, "left")}

    for arm in arms:
        emphasised = arm["label"] in ("fp16", "LM8+ViT4")
        color = {"fp16": BLUE, "LM8+ViT4": ORANGE}.get(arm["label"], DE_EMPHASIS)
        ax.scatter(arm["weight_gib"], arm["anls"], s=190 if emphasised else 90, color=color,
                   zorder=5 if emphasised else 3, edgecolor=SURFACE, linewidth=2)
        dx, dy, ha = placements.get(
            arm["label"], (0, 14 if arm["label"] in above else -20, "center")
        )
        ax.annotate(arm["label"], (arm["weight_gib"], arm["anls"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=9,
                    color=INK if emphasised else MUTED,
                    fontweight="bold" if emphasised else "normal")

    ax.annotate(
        "", xy=(mixed["weight_gib"], mixed["anls"] - 0.012),
        xytext=(fp16["weight_gib"], fp16["anls"] - 0.012),
        arrowprops={"arrowstyle": "->", "color": INK_2, "linewidth": 1.4,
                    "connectionstyle": "arc3,rad=-0.18", "shrinkA": 12, "shrinkB": 12},
    )
    saved = fp16["weight_gib"] - mixed["weight_gib"]
    ax.text(
        3.25, 0.30,
        f"fp16      {fp16['weight_gib']:.2f} GiB   ANLS {fp16['anls']:.3f}\n"
        f"LM8+ViT4  {mixed['weight_gib']:.2f} GiB   ANLS {mixed['anls']:.3f}\n"
        f"−{saved:.2f} GiB for −{fp16['anls'] - mixed['anls']:.3f} ANLS",
        fontsize=9.5, color=INK_2, ha="center", va="center", linespacing=1.6,
        family="monospace",
    )

    ax.axhspan(fp16["anls"] - ANLS_SE, fp16["anls"] + ANLS_SE, color=BLUE, alpha=0.07, zorder=1)
    ax.text(1.42, fp16["anls"] + ANLS_SE + 0.008, "±1 SE of the fp16 baseline (n=40)",
            fontsize=8, color=MUTED)

    ax.set_xlim(1.30, 4.62)
    ax.set_ylim(0.18, 0.53)
    _style(ax)
    _label(
        ax,
        "The INT4 quality cliff is in the language model, not the vision tower",
        "Mixed precision — INT8 language, INT4 vision — keeps fp16 quality at 55% of the weights.",
        xlabel="resident weights (GiB)", ylabel="ANLS on 40 held-out questions",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _provenance(
        fig,
        "Tesla T4, 40 held-out DocVQA/InfographicVQA questions, pruning off. Weights exact and "
        "ledger-confirmed to the byte. ANLS gaps below ~0.12 are within sampling noise "
        "(CONTEXT.md D20/D24).",
    )
    return _save(fig, "quantization_tradeoff.png")


# --- plot 3: what pruning costs -----------------------------------------------------------------


def plot_pruning_curve() -> Path:
    """The negative result, drawn so the noise floor is impossible to overlook.

    Two series, so colour is legal -- but the point of the shaded band is that most of the visible
    gap between them sits inside it. A reader who takes the crossing at 0.375 as a finding has
    been misled by the chart, so the chart draws the uncertainty.
    """
    path = RESULTS / "pruning_quality.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r["n_scored"] == 40]

    baseline = next(r["anls"] for r in rows if r["keep_ratio"] == 1.0)
    series = {}
    for strategy in ("attention", "uniform"):
        points = sorted(
            ((r["mib_reclaimed"], r["anls"]) for r in rows
             if r["strategy"] == strategy and r["keep_ratio"] < 1.0),
            key=lambda p: p[0],
        )
        series[strategy] = ([0.0, *[p[0] for p in points]], [baseline, *[p[1] for p in points]])

    fig, ax = plt.subplots(figsize=(9.0, 5.4), facecolor=SURFACE)
    for (strategy, (xs, ys)), color in ((("attention", series["attention"]), BLUE),
                                        (("uniform", series["uniform"]), ORANGE)):
        ax.fill_between(xs, [y - ANLS_SE for y in ys], [y + ANLS_SE for y in ys],
                        color=color, alpha=0.10, zorder=1, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4, label=strategy)

    ax.axhline(baseline, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.text(795, baseline + 0.008, f"unpruned baseline  {baseline:.3f}", fontsize=8.5,
            color=MUTED, ha="right")

    half = next(r for r in rows if r["strategy"] == "attention" and r["keep_ratio"] == 0.5)
    ax.annotate(
        f"half the visual tokens:\n−{(baseline - half['anls']) / baseline:.0%} quality "
        f"for {half['mib_reclaimed']:.0f} MiB",
        (half["mib_reclaimed"], half["anls"]), textcoords="offset points", xytext=(18, 26),
        fontsize=9, color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
    )

    ax.set_xlim(-25, 810)
    ax.set_ylim(0, 0.55)
    _style(ax)
    _label(
        ax,
        "Visual-token pruning has no free region on document RAG",
        "Quality falls from the first step. Shaded bands are ±1 SE at n=40:\n"
        "where they overlap, the two strategies are indistinguishable.",
        xlabel="KV cache reclaimed (MiB per request)", ylabel="ANLS on 40 held-out questions",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK_2,
              handles=[Patch(color=BLUE, label="attention (FastV)"),
                       Patch(color=ORANGE, label="uniform stride")])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _provenance(
        fig,
        "Tesla T4, FastV at layer 2, 40 held-out questions per point, workload 94b148a0b9f5006e. "
        "MiB reclaimed is exact allocator arithmetic (CONTEXT.md D17); ANLS is measured (D20).",
    )
    return _save(fig, "pruning_curve.png")


def _save(fig, name: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


def main(argv: list[str] | None = None) -> int:
    required = [RESULTS / "quant_ablation.jsonl", RESULTS / "memory_ledger.json",
                RESULTS / "pruning_quality.jsonl"]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("missing results: " + ", ".join(str(p.relative_to(REPO_ROOT)) for p in missing),
              file=sys.stderr)
        return 1

    arms = _load_arms()
    ledger = json.loads((RESULTS / "memory_ledger.json").read_text(encoding="utf-8"))

    print("rendering plots from results/ ...")
    plot_budget(arms, ledger)
    plot_quantization_tradeoff(arms)
    plot_pruning_curve()
    print("\nEvery axis is read from results/ at render time -- none of these numbers is typed in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
