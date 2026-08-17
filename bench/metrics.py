"""Measurement primitives for the EdgeRAG benchmark harness.

Everything here exists to enforce one of the rules in ``00_FOUNDATIONS.md`` §4. The harness is
written before any feature work, and every rule it enforces is a rule that is otherwise violated
silently:

* warmup discard                      -> first-call kernel compilation dominates otherwise
* ``cuda.synchronize()`` bracketing   -> otherwise you time kernel *launches*, not execution
* percentiles, not means              -> a mean hides the tail that matters in serving
* peak-memory reset between runs      -> otherwise every run reports the worst prior run's peak
* device provenance on every record   -> see ``DeviceInfo`` and D4 in ``CONTEXT.md``
"""

from __future__ import annotations

import os
import platform
import shutil
import statistics
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

# --- Device trust (CONTEXT.md D4) ------------------------------------------------------------
#
# The local dev GPU is a GTX 1650: compute capability 7.5, same Turing family as the T4, but the
# GTX 16-series ships *without tensor cores*. Correctness ports between the two; performance does
# not. A latency measured locally is architecturally incomparable to a T4 and must never reach
# results/, a plot, the README, or the CV bullet.
#
# This is enforced in code rather than by discipline, because discipline fails at 2 a.m.

TRUSTED_DEVICE_SUBSTRINGS: tuple[str, ...] = ("Tesla T4",)

#: Percentile reporting below this sample count is dominated by the single worst sample.
MIN_SAMPLES_FOR_P99 = 100


class UntrustedDeviceError(RuntimeError):
    """Raised when a performance record would be written from a non-reference device."""


@dataclass(frozen=True)
class DeviceInfo:
    """Provenance stamped into every benchmark record.

    ``00_FOUNDATIONS.md`` §3 rule 4: you don't always get a T4. Silent hardware variation
    otherwise shows up as a phantom regression and costs an evening (see ``BUGS.md`` P-15).
    """

    name: str
    compute_capability: str | None
    total_memory_bytes: int | None
    driver_version: str | None
    torch_version: str
    cuda_version: str | None
    platform: str
    has_tensor_cores: bool
    trusted: bool

    @staticmethod
    def collect() -> DeviceInfo:
        if not torch.cuda.is_available():
            return DeviceInfo(
                name="cpu",
                compute_capability=None,
                total_memory_bytes=None,
                driver_version=None,
                torch_version=torch.__version__,
                cuda_version=None,
                platform=platform.platform(),
                has_tensor_cores=False,
                trusted=False,
            )

        props = torch.cuda.get_device_properties(0)
        cap = f"{props.major}.{props.minor}"
        name = props.name
        return DeviceInfo(
            name=name,
            compute_capability=cap,
            total_memory_bytes=props.total_memory,
            driver_version=_nvidia_smi_query("driver_version"),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            platform=platform.platform(),
            has_tensor_cores=_has_tensor_cores(name, props.major),
            trusted=any(s in name for s in TRUSTED_DEVICE_SUBSTRINGS),
        )


def _has_tensor_cores(device_name: str, major: int) -> bool:
    """Turing GTX 16-series parts (TU116/TU117) ship without tensor cores.

    Compute capability alone does not answer this: a GTX 1650 and a Tesla T4 both report 7.5.
    """
    if major < 7:
        return False
    # TU116/TU117 ship as "NVIDIA GeForce GTX 16xx"; every 16-series part lacks tensor cores.
    return "GTX 16" not in device_name


def _nvidia_smi_query(field_name: str) -> str | None:
    """Best-effort ``nvidia-smi`` scalar query. Never raises -- provenance is not worth a crash."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={field_name}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def assert_device_trusted(allow_untrusted: bool = False) -> DeviceInfo:
    """Gate on device provenance before a perf record is written.

    Set ``EDGERAG_ALLOW_UNTRUSTED_DEVICE=1`` or pass ``allow_untrusted`` to override; the record
    is then stamped ``trusted=false`` and is not publishable.
    """
    info = DeviceInfo.collect()
    if info.trusted:
        return info

    override = allow_untrusted or os.environ.get("EDGERAG_ALLOW_UNTRUSTED_DEVICE") == "1"
    if not override:
        raise UntrustedDeviceError(
            f"Refusing to record performance on {info.name!r} "
            f"(tensor cores: {info.has_tensor_cores}). "
            f"Reference device is one of {TRUSTED_DEVICE_SUBSTRINGS}. "
            "Pass --allow-untrusted-device to record anyway; the result will be stamped "
            "trusted=false and must not be published. See CONTEXT.md D4."
        )
    warnings.warn(
        f"Recording performance on untrusted device {info.name!r}. "
        "This number is NOT publishable. See CONTEXT.md D4.",
        UserWarning,
        stacklevel=2,
    )
    return info


# --- Timing ----------------------------------------------------------------------------------


def sync() -> None:
    """``torch.cuda.synchronize()`` when there is a CUDA device, else a no-op.

    ``00_FOUNDATIONS.md`` §4 rule 2. CUDA is asynchronous; without this you time launches.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class Percentiles:
    """Latency summary. Means are deliberately reported alongside, never instead of, percentiles."""

    p50: float
    p95: float
    p99: float
    mean: float
    std: float
    min: float
    max: float
    n: int
    p99_is_reliable: bool

    @staticmethod
    def of(samples: list[float]) -> Percentiles:
        if not samples:
            raise ValueError("cannot summarise an empty sample list")
        ordered = sorted(samples)
        n = len(ordered)
        return Percentiles(
            p50=_percentile(ordered, 50.0),
            p95=_percentile(ordered, 95.0),
            p99=_percentile(ordered, 99.0),
            mean=statistics.fmean(ordered),
            std=statistics.stdev(ordered) if n > 1 else 0.0,
            min=ordered[0],
            max=ordered[-1],
            n=n,
            # With <100 samples the "p99" is effectively the max. Reporting it as a percentile
            # without saying so is the kind of thing a careful reader catches immediately.
            p99_is_reliable=n >= MIN_SAMPLES_FOR_P99,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(ordered: list[float], q: float) -> float:
    """Linear-interpolated percentile over a pre-sorted list. Matches ``numpy.percentile``."""
    if len(ordered) == 1:
        return ordered[0]
    pos = (q / 100.0) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


# --- Memory ----------------------------------------------------------------------------------


@dataclass
class MemorySample:
    """Peak memory over a measured region.

    ``allocated`` is what the model actually used. ``reserved`` is what the caching allocator
    holds from the driver. ``nvidia-smi`` shows reserved *plus* the CUDA context (~300-600 MB),
    which is why the three numbers never agree and why the README has to say which one it means
    (see ``CONTEXT.md`` P3).
    """

    peak_allocated_bytes: int
    peak_reserved_bytes: int

    @property
    def peak_allocated_gib(self) -> float:
        return self.peak_allocated_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "peak_allocated_gib": round(self.peak_allocated_gib, 4),
        }


class MemoryProbe:
    """Context manager capturing peak memory over a region.

    Resets the peak counter on entry -- ``BUGS.md`` P-14. Without the reset every run reports the
    worst prior run's peak and memory ablations become monotonic-looking nonsense.
    """

    def __init__(self) -> None:
        self.sample: MemorySample | None = None

    def __enter__(self) -> MemoryProbe:
        if torch.cuda.is_available():
            sync()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc: object) -> None:
        if torch.cuda.is_available():
            sync()
            self.sample = MemorySample(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        else:
            self.sample = MemorySample(peak_allocated_bytes=0, peak_reserved_bytes=0)

    def require(self) -> MemorySample:
        if self.sample is None:
            raise RuntimeError("MemoryProbe was not exited")
        return self.sample


# --- Held-constant manifest ------------------------------------------------------------------


@dataclass
class HeldConstant:
    """What was fixed across compared runs.

    ``00_FOUNDATIONS.md`` §4 rule 5. This is serialised into every record so that a comparison
    between two records can be *checked* rather than assumed. ``bench.compare`` refuses to put two
    records in the same table if their held-constant manifests disagree.
    """

    model_id: str
    dtype: str
    batch_size: int
    prompt_tokens: int
    max_new_tokens: int
    block_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Answer quality --------------------------------------------------------------------------
#
# Lived in `scripts/colab_pruning_quality.py` until Phase 6 needed the same metric for the
# quantization ablation. A metric that two experiments must agree on cannot live inside one of
# them: the moment it is copied, the two curves stop being comparable and nobody notices.


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance, iterative to avoid recursion limits on long answers."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def anls(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity against the best of several gold answers.

    The standard DocVQA metric (``CONTEXT.md`` P4). Exact match is too brittle for generative
    answers -- "0.28" against "0.28%" is a real answer and exact match scores it zero -- while an
    unthresholded similarity hands partial credit to long wrong answers for incidental character
    overlap. Below ``threshold`` the score is zeroed rather than allowed to decay smoothly.
    """
    prediction = prediction.strip().lower()
    best = 0.0
    for answer in answers:
        gold = answer.strip().lower()
        if not gold and not prediction:
            best = max(best, 1.0)
            continue
        denom = max(len(prediction), len(gold))
        if denom == 0:
            continue
        similarity = 1.0 - levenshtein(prediction, gold) / denom
        best = max(best, similarity)
    return best if best >= threshold else 0.0
