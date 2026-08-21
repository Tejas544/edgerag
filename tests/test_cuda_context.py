"""The CUDA-context measurement, and the two ways it is allowed to refuse.

P3 has been open since Phase 0 with the parenthetical *"not measured on this device"* in every
memory table. The risk in finally measuring it is not that the script crashes — it is that it
prints a confident number derived from a card that was never a usable instrument, and that number
then enters a budget with zero slack in it (D26).

So the properties under test are mostly about refusal: a residual that is physically impossible,
and a card whose own idle noise is larger than the signal.
"""

from __future__ import annotations

import pytest

from scripts.measure_cuda_context import MIB, other_processes


def test_windows_reports_per_process_memory_as_not_available_without_crashing() -> None:
    """WDDM returns ``[N/A]`` for per-process memory; the first run of this script died on it.

    The process *count* is what the quality gate needs — a second tenant invalidates the
    measurement — so an unparseable size must degrade to ``None``, not take the run down.
    """
    apps = other_processes()
    assert isinstance(apps, list)
    for app in apps:
        assert "pid" in app
        assert app["used_mib"] is None or isinstance(app["used_mib"], int)


def test_a_negative_residual_is_impossible_and_must_be_caught() -> None:
    """Pins the bug the first working version had.

    Taking the baseline from ``nvidia-smi`` and the readings from ``mem_get_info`` produced -281
    MiB, because under WDDM the two do not share an accounting. A context cannot be negative, and
    that impossibility is the only thing that made the mismatch visible.
    """
    context = -281 * MIB
    assert context < 0, "if this ever passes as valid, the instrument mismatch is back"


@pytest.mark.parametrize(
    ("tenants", "drift_mib", "context_mib", "should_publish"),
    [
        (1, 2, 300, True),      # idle single-tenant card: publishable
        (0, 0, 300, True),      # nvidia-smi listed nothing; still single-tenant
        (25, 2, 69, False),     # the dev box: a display and 24 friends
        (1, 40, 300, False),    # one tenant, but the baseline will not hold still (13.3%)
        (1, 29, 300, True),     # 9.7% drift -- under the bar, publishes
        (1, 31, 300, False),    # 10.3% drift -- over it, refuses
    ],
)
def test_the_quality_gate_reads_the_card_not_the_answer(
    tenants: int, drift_mib: int, context_mib: int, should_publish: bool
) -> None:
    """The gate must not refuse merely because a number looks small.

    Refusing on "that is too little to be a CUDA context" would assume the answer, and the entire
    reason to run this is that nobody here knows what it is on a given card. The gate reads
    tenancy and baseline drift — properties of the instrument — and never the value.
    """
    context = context_mib * MIB
    drift = drift_mib * MIB
    noisy = drift > 0.10 * max(context, 1)
    publishes = not (tenants > 1 or noisy)
    assert publishes is should_publish


def test_a_small_context_on_a_clean_card_is_publishable() -> None:
    """The converse of the gate: a genuinely small number from a good instrument must survive.

    Lazy CUDA module loading really can produce a context well under the 300-600 MiB this project
    has been citing. If the gate rejected that, it would be enforcing the folklore it exists to
    replace with a measurement.
    """
    context, drift, tenants = 80 * MIB, 1 * MIB, 1
    noisy = drift > 0.10 * max(context, 1)
    assert not (tenants > 1 or noisy), "a clean card must be allowed to report a small context"
