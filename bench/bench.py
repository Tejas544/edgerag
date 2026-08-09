"""The EdgeRAG benchmark harness.

Written before any feature work, per ``00_FOUNDATIONS.md`` §4: *"Before any feature work on any
project, write bench.py. Every subsequent commit gets measured against it."*

Usage::

    python -m bench.bench --dry-run              # prove the harness against a fake model
    python -m bench.bench --dry-run --md         # ... and render the markdown table

Design notes:

* A benchmarked callable returns a :class:`TrialResult`. It reports *when* tokens arrived; the
  harness owns warmup, repetition, synchronisation, percentiles, and memory. Callers cannot
  accidentally skip a rule.
* TTFT and decode throughput are aggregated separately (§4 rule 7). Prefill is compute-bound and
  decode is memory-bound; a single "latency" number conflates two different regimes.
* Records are appended to JSONL and fsync'd as they complete (§3 rule 5), so a Colab disconnect at
  hour 3 does not cost hours 1 and 2.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.metrics import (
    DeviceInfo,
    HeldConstant,
    MemoryProbe,
    MemorySample,
    Percentiles,
    assert_device_trusted,
    sync,
)

DEFAULT_WARMUP = 10
DEFAULT_TRIALS = 5
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class TrialResult:
    """What one benchmarked run reports back to the harness.

    ``decode_step_times_s`` is per-step so the harness can report inter-token latency percentiles,
    which is what actually governs perceived streaming smoothness. A single aggregate tok/s hides
    a stalling decode loop completely.
    """

    ttft_s: float
    decode_step_times_s: list[float]
    n_prompt_tokens: int
    n_generated_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decode_s(self) -> float:
        return sum(self.decode_step_times_s)

    @property
    def end_to_end_s(self) -> float:
        return self.ttft_s + self.decode_s

    def validate(self) -> None:
        if self.ttft_s <= 0:
            raise ValueError(f"non-positive TTFT ({self.ttft_s}s) -- missing a sync()?")
        if self.n_generated_tokens != len(self.decode_step_times_s):
            raise ValueError(
                f"reported {self.n_generated_tokens} generated tokens but "
                f"{len(self.decode_step_times_s)} step times"
            )


@dataclass
class BenchRecord:
    """One fully-specified measurement. This is the unit that lands in ``results/*.jsonl``."""

    run_id: str
    name: str
    timestamp_utc: str
    device: dict[str, Any]
    held_constant: dict[str, Any]
    warmup_iters: int
    trials: int
    ttft: dict[str, Any]
    decode_step_latency: dict[str, Any]
    decode_tokens_per_s: dict[str, Any]
    end_to_end: dict[str, Any]
    memory: dict[str, Any]
    trusted: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(
    name: str,
    fn: Callable[[], TrialResult],
    held_constant: HeldConstant,
    *,
    warmup: int = DEFAULT_WARMUP,
    trials: int = DEFAULT_TRIALS,
    allow_untrusted_device: bool = False,
    notes: str = "",
) -> BenchRecord:
    """Run ``fn`` under the full measurement protocol and return a record.

    The device gate fires *before* any work, so an untrusted-device run fails in a second rather
    than after a twenty-minute sweep.
    """
    if trials < DEFAULT_TRIALS:
        raise ValueError(
            f"trials={trials} < {DEFAULT_TRIALS}: a single number is not a result "
            "(00_FOUNDATIONS.md §4 rule 4)"
        )

    device: DeviceInfo = assert_device_trusted(allow_untrusted=allow_untrusted_device)

    # Warmup. Discarded entirely -- first-call kernel compilation and cache warming otherwise
    # dominate the measurement (§4 rule 1).
    for _ in range(warmup):
        fn()
    sync()

    ttfts: list[float] = []
    all_step_times: list[float] = []
    e2es: list[float] = []
    throughputs: list[float] = []

    with MemoryProbe() as probe:
        for _ in range(trials):
            sync()
            result = fn()
            sync()
            result.validate()

            ttfts.append(result.ttft_s)
            all_step_times.extend(result.decode_step_times_s)
            e2es.append(result.end_to_end_s)
            if result.decode_s > 0:
                throughputs.append(result.n_generated_tokens / result.decode_s)

    memory: MemorySample = probe.require()

    return BenchRecord(
        run_id=uuid.uuid4().hex[:12],
        name=name,
        timestamp_utc=datetime.now(UTC).isoformat(),
        device=asdict(device),
        held_constant=held_constant.to_dict(),
        warmup_iters=warmup,
        trials=trials,
        ttft=Percentiles.of(ttfts).to_dict(),
        decode_step_latency=Percentiles.of(all_step_times).to_dict(),
        decode_tokens_per_s=Percentiles.of(throughputs).to_dict(),
        end_to_end=Percentiles.of(e2es).to_dict(),
        memory=memory.to_dict(),
        trusted=device.trusted,
        notes=notes,
    )


class JsonlWriter:
    """Append-and-fsync writer.

    ``00_FOUNDATIONS.md`` §3 rule 5: long sweeps write incrementally so a disconnect at hour 3
    does not cost hours 1 and 2. The ``fsync`` is the point -- buffered writes are lost with the
    runtime.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: BenchRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


class IncomparableRecordsError(ValueError):
    """Raised when records that differ in held-constant fields are put in one table."""


def check_comparable(records: Sequence[dict[str, Any]], ignore: Sequence[str] = ()) -> None:
    """Refuse to tabulate runs that did not hold the same things constant.

    ``00_FOUNDATIONS.md`` §4 rule 5, enforced. The failure this prevents is the quiet one: an
    ablation table whose rows differ in batch size as well as in the feature being ablated, so the
    "speedup" column measures two things at once.
    """
    if len(records) < 2:
        return
    ignore_set = set(ignore)
    base = records[0]["held_constant"]
    for rec in records[1:]:
        for key, value in base.items():
            if key in ignore_set or key == "extra":
                continue
            other = rec["held_constant"].get(key)
            if other != value:
                raise IncomparableRecordsError(
                    f"records {records[0]['name']!r} and {rec['name']!r} differ in held-constant "
                    f"field {key!r}: {value!r} vs {other!r}. Add it to `ignore` only if it is the "
                    "variable under test."
                )


def records_to_markdown(
    records: Sequence[dict[str, Any]], ignore: Sequence[str] = ()
) -> str:
    """Render records as the markdown table that goes in the README."""
    if not records:
        return "_no records_\n"
    check_comparable(records, ignore=ignore)

    header = (
        "| run | TTFT p50 (ms) | TTFT p99 (ms) | decode (tok/s) | "
        "e2e p50 (ms) | peak alloc (GiB) | trusted |"
    )
    sep = "|---|---:|---:|---:|---:|---:|:--:|"
    lines = [header, sep]
    for rec in records:
        p99_flag = "" if rec["ttft"]["p99_is_reliable"] else "*"
        lines.append(
            "| {name} | {t50:.1f} | {t99:.1f}{flag} | {tps:.1f} ± {tps_sd:.1f} | "
            "{e50:.1f} | {mem:.3f} | {tr} |".format(
                name=rec["name"],
                t50=rec["ttft"]["p50"] * 1e3,
                t99=rec["ttft"]["p99"] * 1e3,
                flag=p99_flag,
                tps=rec["decode_tokens_per_s"]["p50"],
                tps_sd=rec["decode_tokens_per_s"]["std"],
                e50=rec["end_to_end"]["p50"] * 1e3,
                mem=rec["memory"]["peak_allocated_gib"],
                tr="yes" if rec["trusted"] else "**NO**",
            )
        )

    footnotes = ["", f"Device: `{records[0]['device']['name']}`.", ""]
    if any(not r["ttft"]["p99_is_reliable"] for r in records):
        footnotes.append(
            "`*` p99 computed from fewer than 100 samples -- it is effectively the max, "
            "not a percentile. Treat as indicative only."
        )
    if any(not r["trusted"] for r in records):
        footnotes.append(
            "**Untrusted-device rows are not publishable.** Measured on hardware that is not the "
            "reference T4; see `CONTEXT.md` D4."
        )
    return "\n".join(lines + footnotes) + "\n"


# --- Dry run: prove the harness before it measures anything real -----------------------------


def _fake_generation(
    prompt_tokens: int = 512, new_tokens: int = 32, rng: random.Random | None = None
) -> TrialResult:
    """A model-shaped callable that only sleeps.

    The harness has to be trustworthy before it measures anything real, so it gets exercised
    end-to-end against a fake whose timings are known by construction.
    """
    rng = rng or random.Random(0)
    t0 = time.perf_counter()
    time.sleep(0.004 + rng.random() * 0.001)  # "prefill"
    ttft = time.perf_counter() - t0

    steps: list[float] = []
    for _ in range(new_tokens):
        s0 = time.perf_counter()
        time.sleep(0.0004 + rng.random() * 0.0002)  # "decode step"
        steps.append(time.perf_counter() - s0)

    return TrialResult(
        ttft_s=ttft,
        decode_step_times_s=steps,
        n_prompt_tokens=prompt_tokens,
        n_generated_tokens=new_tokens,
        metadata={"fake": True},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EdgeRAG benchmark harness")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise the harness against a fake model; writes to results/dry_run.jsonl",
    )
    parser.add_argument("--md", action="store_true", help="print the markdown table")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--allow-untrusted-device",
        action="store_true",
        help=(
            "record a perf number on non-reference hardware. The record is stamped "
            "trusted=false and MUST NOT be published. See CONTEXT.md D4."
        ),
    )
    args = parser.parse_args(argv)

    if not args.dry_run:
        parser.error("no benchmark selected; --dry-run is the only target until Phase 1")

    rng = random.Random(1234)
    held = HeldConstant(
        model_id="fake/dry-run",
        dtype="float16",
        batch_size=1,
        prompt_tokens=512,
        max_new_tokens=32,
    )
    record = run_benchmark(
        name="dry-run",
        fn=lambda: _fake_generation(rng=rng),
        held_constant=held,
        warmup=args.warmup,
        trials=args.trials,
        allow_untrusted_device=args.allow_untrusted_device,
        notes="Harness self-test against a sleeping fake model. Not a real measurement.",
    )

    writer = JsonlWriter(RESULTS_DIR / "dry_run.jsonl")
    writer.append(record)
    print(f"wrote {writer.path}")

    if args.md:
        print()
        print(records_to_markdown([record.to_dict()]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
