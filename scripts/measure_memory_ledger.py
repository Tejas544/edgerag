"""Phase 6's memory column, and the 4 GiB accounting table. Exact, local, no GPU, no weights.

    python -m scripts.measure_memory_ledger

The Phase 6 gate wants memory / throughput / quality across {fp16, int8, int4} x {LM, LM+ViT,
ViT}. **The memory third of that table is arithmetic, not a benchmark**, so it does not belong on
a T4 at all: quantized bytes are a function of tensor shape and group size, and neither depends on
which card the tensor would have sat on. This script computes it from the checkpoint *config*
alone -- the model is instantiated on the ``meta`` device, which allocates shapes and dtypes with
no storage, so the 2.2B headline model is "loaded" in about a second without downloading nine
gigabytes (``CONTEXT.md`` D4, the same discipline as ``scripts/colab_gather_overhead.py``).

Two properties are worth stating plainly, because they are what make the numbers trustworthy:

* **Nothing here re-derives the packing arithmetic.** Every quantized figure is
  :meth:`QuantLinear.weight_bytes` called on a meta instance of the layer that would actually be
  built -- scales and bias included. A ledger with its own copy of the byte math is a ledger that
  drifts from the model the afternoon someone changes the group size.
* **Exactly one line is not exact**, and it is labelled: the transient activation peak, which is
  inferred from the measured T4 baseline rather than computed. ``BUGS.md`` P-25 is precisely the
  failure of leaving that line out -- the table sums under budget and the pipeline OOMs anyway.

The composition being priced is what actually runs after ``BUGS.md`` B-05: **our** decoder, plus
HuggingFace's vision tower and connector, with the duplicate HF text decoder freed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import nn
from transformers import AutoModelForImageTextToText

from edgerag.core.budget import GIB, BudgetLedger
from edgerag.core.linear import (
    FP16_BYTES,
    SKIP_RAGGED,
    SKIP_SENSITIVE,
    QuantizationPlan,
    QuantLinear,
    plan_quantization,
)
from edgerag.core.loader import HEADLINE_MODEL, load_config, load_spec
from edgerag.core.model import EdgeRagDecoder
from edgerag.core.quant import QuantConfig
from edgerag.core.spec import ModelSpec

MIB = 1024**2
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Which components each arm of the ablation quantizes. The connector travels with the vision
#: tower: it is the modality projection, it only ever runs on tower output, and separating them
#: would produce a fourth arm nobody asked for.
ARMS: dict[str, tuple[str, ...]] = {
    "none": (),
    "LM": ("language",),
    "LM+ViT": ("language", "vision", "connector"),
    "ViT": ("vision", "connector"),
}


#: This table is computed, not measured, and says so where a reader would otherwise assume the
#: opposite. The budget is *defined* on the measured peak; this is the prediction it gets checked
#: against, which is a different and weaker claim.
COMPUTED_BASIS = (
    "Computed exactly from tensor shapes, not measured -- the budget itself is defined on "
    "`torch.cuda.max_memory_allocated()` (`CONTEXT.md` P3), and this table is the prediction "
    "that measurement will be checked against."
)


@dataclass(frozen=True)
class Component:
    """One resident piece of the model, priced dense and quantized.

    ``other_bytes`` is everything that is not an ``nn.Linear``: token and position embeddings, the
    norms, and the vision tower's patch-embedding Conv2d. It never quantizes (``BUGS.md`` P-21)
    and it is not negligible -- on the headline model it is 193 MiB, which is 4.6% of the fp16
    checkpoint and would be an embarrassing omission from a table titled "memory budget".
    """

    name: str
    n_params: int
    plan: QuantizationPlan
    other_bytes: int

    @property
    def dense_bytes(self) -> int:
        return self.plan.dense_bytes + self.other_bytes

    @property
    def quantized_bytes(self) -> int:
        return self.plan.planned_bytes + self.other_bytes

    def bytes_when(self, quantized: bool) -> int:
        return self.quantized_bytes if quantized else self.dense_bytes


def _linear_param_numel(root: nn.Module) -> int:
    return sum(
        t.numel() for m in root.modules() if isinstance(m, nn.Linear) for t in m.parameters()
    )


def _param_numel(root: nn.Module) -> int:
    return sum(t.numel() for t in root.parameters())


def _all_numel(root: nn.Module) -> int:
    return _param_numel(root) + sum(t.numel() for t in root.buffers())


def build_component(name: str, root: nn.Module, config: QuantConfig) -> Component:
    """Price one module tree at fp16 and at ``config``.

    Buffers are counted at the fp16 rate alongside parameters. The only non-parameter buffers in
    either tree are RoPE inverse frequencies -- 128 bytes -- so the choice is a rounding error and
    counting them at all is the point (``BudgetLedger.register_module`` makes the same argument).
    """
    plan = plan_quantization(root, config)
    linear_numel = _linear_param_numel(root)
    other_bytes = (_all_numel(root) - linear_numel) * FP16_BYTES

    # The plan and the module tree must agree about what a dense linear costs, or every saving
    # below is measured against the wrong baseline.
    assert plan.dense_bytes == linear_numel * FP16_BYTES, name
    return Component(name=name, n_params=_all_numel(root), plan=plan, other_bytes=other_bytes)


def build_components(model_id: str, spec: ModelSpec, config: QuantConfig) -> list[Component]:
    """Instantiate the deployed composition on ``meta`` and price each piece.

    Our decoder is built from :class:`ModelSpec`, not lifted from the checkpoint, and the
    assertion below is a real check rather than a formality: it says the from-scratch stack is
    parameter-identical to the one HuggingFace would have built, which is the claim
    ``tests/test_equivalence.py`` makes numerically and this makes structurally.
    """
    with torch.device("meta"):
        hf = AutoModelForImageTextToText.from_config(load_config(model_id))
        ours = EdgeRagDecoder(spec)

    # Parameters, not parameters-plus-buffers: HuggingFace's rotary embedding caches a different
    # number of inverse frequencies than ours does, which is a 128-byte implementation detail and
    # not a divergence in the model being priced.
    inner = hf.model
    hf_text_params = _param_numel(inner.text_model) + _param_numel(hf.lm_head)
    if _param_numel(ours) != hf_text_params:
        raise RuntimeError(
            f"our decoder holds {_param_numel(ours):,} parameters against the checkpoint's "
            f"{hf_text_params:,}. The spec and the checkpoint have diverged -- the ledger would "
            "be pricing a model that does not exist."
        )

    return [
        build_component("language", ours, config),
        build_component("vision", inner.vision_model, config),
        build_component("connector", inner.connector, config),
    ]


# --- section 1: the ablation matrix ---------------------------------------------------------


def arm_totals(components: list[Component], arm: tuple[str, ...]) -> dict[str, int]:
    return {c.name: c.bytes_when(c.name in arm) for c in components}


def print_matrix(
    per_bits: dict[int, list[Component]], model_id: str, group_size: int, budget_gib: float
) -> list[dict[str, Any]]:
    """The Phase 6 gate's memory column: every arm at every bit width, in one table."""
    first = next(iter(per_bits.values()))
    names = [c.name for c in first]
    fp16_total = sum(c.dense_bytes for c in first)

    print(f"=== weights by ablation arm -- {model_id} ===")
    print(f"    group size {group_size}, fp16 deployment dtype, HF text decoder freed (B-05)\n")
    header = f"{'arm':>8} {'bits':>5}" + "".join(f"{n:>11}" for n in names)
    print(header + f"{'total':>10}{'vs fp16':>9}{'of budget':>11}")

    rows: list[dict[str, Any]] = []
    todo: list[tuple[str, int]] = [("none", 16)]
    todo += [
        (arm, bits)
        for bits in sorted(per_bits, reverse=True)
        for arm in ARMS
        if arm != "none"
    ]

    for arm_name, bits in todo:
        components = first if bits == 16 else per_bits[bits]
        totals = arm_totals(components, ARMS[arm_name])
        total = sum(totals.values())
        label = "fp16" if bits == 16 else arm_name
        cells = "".join(f"{totals[n] / GIB:>11.3f}" for n in names)
        print(
            f"{label:>8} {bits:>5}{cells}{total / GIB:>10.3f}"
            f"{fp16_total / total:>8.2f}x{total / (budget_gib * GIB):>10.0%}"
        )
        rows.append(
            {
                "arm": arm_name,
                "bits": bits,
                "components": totals,
                "total_bytes": total,
                "total_gib": round(total / GIB, 4),
                "compression_vs_fp16": round(fp16_total / total, 4),
            }
        )
    print("\n  The fp16 row is over the budget on weights alone, before one KV block is "
          "allocated -- which\n  is D14 finding 1 restated as arithmetic rather than as an OOM.\n")
    return rows


def print_skip_costs(components: list[Component], group_size: int) -> dict[str, Any]:
    """What stays fp16, why, and what it costs. The skip list is part of the result (P-21)."""
    sensitive: list[tuple[str, Any]] = []
    ragged: list[tuple[str, Any]] = []
    for c in components:
        sensitive += [(c.name, layer) for layer in c.plan.skipped_for(SKIP_SENSITIVE)]
        ragged += [(c.name, layer) for layer in c.plan.skipped_for(SKIP_RAGGED)]

    print("=== what the skip list costs (BUGS.md P-21) ===")
    sensitive_bytes = 0
    for component, layer in sensitive:
        would_be = layer.dense_bytes / 4  # ~4x, close enough for a "what it would have cost"
        sensitive_bytes += layer.dense_bytes
        print(
            f"  {component}.{layer.name:<28} {layer.out_features:>6} x {layer.in_features:<5} "
            f"{layer.dense_bytes / MIB:>8.1f} MiB fp16   (~{would_be / MIB:.1f} MiB if quantized)"
        )
    if not sensitive:
        print("  (none -- every linear layer was eligible)")
    print(f"  deliberately dense: {sensitive_bytes / MIB:.1f} MiB\n")

    print("=== layers the group size cannot cover ===")
    ragged_bytes = sum(layer.dense_bytes for _, layer in ragged)
    if ragged:
        shapes = {(layer.out_features, layer.in_features) for _, layer in ragged}
        for out_f, in_f in sorted(shapes):
            n = sum(
                1 for _, layer in ragged if (layer.out_features, layer.in_features) == (out_f, in_f)
            )
            print(f"  {n:>3} x [{out_f} x {in_f}] -- in_features={in_f} is not divisible by "
                  f"{group_size}")
        print(f"  forced to stay fp16: {ragged_bytes / MIB:.1f} MiB")

        # What a group size that *does* divide them would recover. Reported rather than applied:
        # silently shrinking the group for one layer would make this table describe a model that
        # was never run.
        divisor = max(g for g in (16, 32, 64, 128) if all(
            layer.in_features % g == 0 for _, layer in ragged))
        alt = QuantConfig(group_size=divisor)
        recovered = 0
        with torch.device("meta"):
            for _, layer in ragged:
                recovered += QuantLinear(
                    layer.in_features, layer.out_features, bias=layer.has_bias, config=alt
                ).weight_bytes()
        bits_per_weight = 8.0 * recovered / sum(
            layer.in_features * layer.out_features for _, layer in ragged
        )
        print(
            f"  at group {divisor} they would cost {recovered / MIB:.1f} MiB "
            f"({bits_per_weight:.2f} bits/weight, against {8 * FP16_BYTES} dense) -- "
            f"a further {(ragged_bytes - recovered) / MIB:.1f} MiB"
        )
    else:
        print(f"  (none -- every eligible layer's in_features divides by {group_size})")
    print()

    return {
        "sensitive_bytes": sensitive_bytes,
        "sensitive_layers": [f"{c}.{layer.name}" for c, layer in sensitive],
        "ragged_bytes": ragged_bytes,
        "ragged_layers": [f"{c}.{layer.name}" for c, layer in ragged],
    }


# --- section 2: the 4 GiB accounting table ---------------------------------------------------


def measured_baseline_peak() -> float | None:
    """Peak allocated bytes from the batch-1 T4 baseline, if it has been run."""
    path = REPO_ROOT / "results" / "baseline.jsonl"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("name") == "hf_generate_b1" and row.get("trusted"):
            return float(row["memory"]["peak_allocated_bytes"])
    return None


def request_lengths() -> list[int]:
    """Total prompt tokens per request on the frozen trace, from the Phase 1 gate."""
    path = REPO_ROOT / "data" / "token_ratio_gate.json"
    if not path.exists():
        return []
    gate = json.loads(path.read_text(encoding="utf-8"))
    per_query = gate["per_query"].get(HEADLINE_MODEL, [])
    return [int(q["total_tokens"]) for q in per_query]


def infer_activation_bytes(
    spec: ModelSpec, fp16_weight_bytes: int, new_tokens: int
) -> tuple[int, int, str]:
    """Transient peak above resident weights + KV, inferred from the measured T4 baseline.

    This is the one line in the ledger that is not exact, so here is exactly how soft it is: the
    baseline harness records ``prompt_tokens: -1``, so the request's own length is not on file and
    the KV term has to use the trace's median. The p10-p90 spread of request lengths is carried
    through as the error bar rather than hidden.

    It comes out small -- a few hundred MiB against a 4 GiB budget -- and the mechanism is the
    same one as ``BUGS.md`` B-05: HuggingFace's tower runs SDPA, so no ``(heads, patches,
    patches)`` score matrix is ever materialised. An eager tower would put a gigabyte on this line.
    """
    peak = measured_baseline_peak()
    lengths = request_lengths()
    if peak is None or not lengths:
        return 0, 0, "no measured baseline on file"

    median = statistics.median(lengths)
    ordered = sorted(lengths)
    p10, p90 = ordered[len(ordered) // 10], ordered[9 * len(ordered) // 10]

    def residual(prompt_tokens: float) -> float:
        return peak - fp16_weight_bytes - (prompt_tokens + new_tokens) * spec.kv_bytes_per_token()

    # A longer assumed prompt means more of the peak was KV, so *less* was activation.
    best = residual(median)
    spread = (residual(p10) - residual(p90)) / 2
    note = (
        f"inferred: measured T4 peak {peak / GIB:.3f} GiB minus fp16 weights and the KV of a "
        f"median {median:.0f}-token request; +/-{spread / GIB:.2f} GiB for the p10-p90 length "
        f"spread ({p10}-{p90} tokens)"
    )
    return int(best), int(spread), note


def _component_note(component: Component) -> str:
    """Say *which* bytes resisted quantization, not just how many.

    "259 MiB stays fp16" invites the reader to assume the skip list. For the vision tower almost
    all of it is the ragged MLP instead, and those are different problems with different fixes --
    one is a deliberate quality decision, the other is a divisibility accident.
    """
    parts = []
    for reason, label in ((SKIP_RAGGED, "ragged groups"), (SKIP_SENSITIVE, "skip list")):
        held = sum(layer.dense_bytes for layer in component.plan.skipped_for(reason))
        if held:
            parts.append(f"{held / MIB:.0f} MiB {label}")
    if component.other_bytes:
        parts.append(f"{component.other_bytes / MIB:.0f} MiB embeddings/norms")

    note = f"quantized at {component.plan.bits_per_weight:.3f} bits/weight"
    return f"{note}; still fp16: {', '.join(parts)}" if parts else note


def build_ledger(
    components: list[Component],
    arm_name: str,
    spec: ModelSpec,
    budget_gib: float,
    new_tokens: int,
) -> tuple[BudgetLedger, dict[str, Any]]:
    """The accounting table for one arm, sized for a **single** in-flight request.

    Sizing the KV pool at one request rather than at "whatever is left over" is the difference
    between a verdict and a tautology: a pool defined as the remainder makes ``within_budget``
    true by construction, no matter how large the weights are. One request is the minimum viable
    system, so the table now answers "does this run at all?", and the headroom underneath it
    answers "how much concurrency does the budget buy?".
    """
    arm = ARMS[arm_name]
    bits = components[0].plan.config.bits
    ledger = BudgetLedger(limit_gib=budget_gib)

    for component in components:
        quantized = component.name in arm
        note = _component_note(component) if quantized else "left dense by this arm"
        ledger.register(
            component.name,
            component.bytes_when(quantized),
            f"int{bits}" if quantized else "float16",
            note,
        )

    fp16_weights = sum(c.dense_bytes for c in components)
    activation, spread, note = infer_activation_bytes(spec, fp16_weights, new_tokens)
    ledger.register("activation + workspace", activation, "float16", note)

    lengths = request_lengths()
    median_tokens = int(statistics.median(lengths)) if lengths else 0
    per_request = spec.kv_bytes(median_tokens + new_tokens)
    ledger.register(
        "KV cache, 1 request",
        per_request,
        "float16",
        f"{median_tokens} median prompt tokens + {new_tokens} generated, at "
        f"{spec.kv_bytes_per_token() / 1024:.0f} KiB/token (MHA -- D10)",
    )

    headroom = int(budget_gib * GIB) - ledger.total_bytes
    extra = headroom / per_request if per_request else 0.0

    return ledger, {
        "arm": arm_name,
        "bits": bits,
        "weights_bytes": sum(c.bytes_when(c.name in arm) for c in components),
        "activation_bytes": activation,
        "activation_uncertainty_bytes": spread,
        "kv_bytes_per_request": per_request,
        "median_prompt_tokens": median_tokens,
        "headroom_bytes": headroom,
        "concurrent_requests": round(1 + extra, 2) if headroom > 0 else 1.0,
        "within_budget": ledger.within_budget,
    }


def render_markdown(
    model_id: str,
    matrix: list[dict[str, Any]],
    components: list[Component],
    ledger: BudgetLedger,
    summary: dict[str, Any],
    group_size: int,
) -> str:
    """The two tables in the form they will be pasted into the README.

    Written by the script that computed them rather than transcribed by hand, because a
    transcribed table is a table that disagrees with `results/` by Phase 8.
    """
    names = [c.name for c in components]
    lines = [
        f"# Memory ledger — {model_id}",
        "",
        f"Computed exactly from the checkpoint config at group size {group_size}; no weights are "
        "loaded and no GPU is involved (`python -m scripts.measure_memory_ledger`).",
        "",
        "## Weights, by ablation arm (GiB)",
        "",
        "| arm | bits | " + " | ".join(names) + " | total | vs fp16 |",
        "|---|---:|" + "---:|" * (len(names) + 2),
    ]
    for row in matrix:
        cells = " | ".join(f"{row['components'][n] / GIB:.3f}" for n in names)
        label = "fp16" if row["bits"] == 16 else row["arm"]
        lines.append(
            f"| {label} | {row['bits']} | {cells} | {row['total_gib']:.3f} | "
            f"{row['compression_vs_fp16']:.2f}x |"
        )

    lines += [
        "",
        "The fp16 row is over a 4 GiB budget on weights alone, before one KV block is allocated.",
        "",
        f"## The budget, arm {summary['arm']} at int{summary['bits']}, one request in flight",
        "",
        ledger.to_markdown(COMPUTED_BASIS),
        f"Headroom {summary['headroom_bytes'] / GIB:.2f} GiB = "
        f"**{summary['concurrent_requests']:.1f} concurrent requests** at "
        f"{summary['kv_bytes_per_request'] / GIB:.2f} GiB of KV each. The fp16 arm supports none.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 memory ledger -- exact, local, no GPU")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    parser.add_argument("--ship-arm", default="LM+ViT", choices=sorted(ARMS))
    parser.add_argument("--ship-bits", type=int, default=4)
    parser.add_argument("--budget-gib", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="results/memory_ledger.json")
    args = parser.parse_args(argv)

    transformers.logging.set_verbosity_error()

    spec = load_spec(args.model)
    ship_config = QuantConfig(group_size=args.group_size, bits=args.ship_bits)

    per_bits: dict[int, list[Component]] = {
        bits: build_components(args.model, spec, QuantConfig(group_size=args.group_size, bits=bits))
        for bits in args.bits
    }
    matrix = print_matrix(per_bits, args.model, args.group_size, args.budget_gib)
    skips = print_skip_costs(per_bits[args.ship_bits], args.group_size)

    ledger, summary = build_ledger(
        per_bits[args.ship_bits], args.ship_arm, spec, args.budget_gib, args.max_new_tokens
    )
    print(f"=== the {args.budget_gib:.0f} GiB budget, arm {args.ship_arm} at "
          f"int{args.ship_bits}, one request in flight ===\n")
    print(ledger.to_markdown(COMPUTED_BASIS))

    verdict = "fits" if ledger.within_budget else "OVER BUDGET"
    print(f"  total {ledger.total_gib:.3f} GiB of {args.budget_gib:.1f} -- {verdict}, "
          f"{summary['headroom_bytes'] / GIB:.2f} GiB spare")
    print(f"  that headroom is {summary['concurrent_requests']:.1f} concurrent requests at "
          f"{summary['kv_bytes_per_request'] / GIB:.2f} GiB of KV each.")
    print("  The fp16 arm supports ZERO: its weights alone are over the budget, which is why the "
          "comparison\n  is not '4x less memory' but 'runs at all versus does not'.")

    payload = {
        "model": args.model,
        "computed": "exact from checkpoint config; meta-device instantiation, no weights loaded",
        "group_size": args.group_size,
        "quant_config": {"bits": ship_config.bits, "group_size": ship_config.group_size},
        "components": {
            c.name: {
                "params": c.n_params,
                "fp16_bytes": c.dense_bytes,
                "quantized_bytes": c.quantized_bytes,
                "non_linear_bytes": c.other_bytes,
                "plan": c.plan.to_dict(),
            }
            for c in per_bits[args.ship_bits]
        },
        # Per component, per bit width, in bytes. The matrix above prices whole *arms*, which can
        # only express one bit width at a time; D23 finding 2 says the configuration worth
        # shipping is mixed (INT8 language + INT4 vision), and pricing that needs the components
        # priced independently. 16 is the dense cost.
        "component_bytes": {
            c.name: {
                "16": c.dense_bytes,
                **{str(bits): comp.quantized_bytes
                   for bits in sorted(per_bits) for comp in per_bits[bits] if comp.name == c.name},
            }
            for c in per_bits[args.ship_bits]
        },
        "matrix": matrix,
        "skips": skips,
        "ledger": ledger.to_dict(),
        "summary": summary,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    md.write_text(
        render_markdown(
            args.model, matrix, per_bits[args.ship_bits], ledger, summary, args.group_size
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out} and {md}")
    print("Throughput and quality for these arms need the 2.2B on a T4; this is the memory third.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
