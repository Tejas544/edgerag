"""Tests for memory budget enforcement and accounting.

The 4 GB ceiling is the project's thesis. These tests exist so that the ceiling is a *failing
test* from Phase 0 rather than an audit on the last afternoon.
"""

from __future__ import annotations

import pytest
import torch

from edgerag.core.budget import (
    GIB,
    BudgetLedger,
    MemoryBudget,
    MemoryBudgetExceeded,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_budget_passes_when_under_ceiling() -> None:
    with MemoryBudget(limit_gib=4.0, label="test") as budget:
        pass
    assert budget.peak_gib <= 4.0
    assert budget.headroom_gib >= 0


@CUDA
def test_budget_raises_when_over_ceiling() -> None:
    """A tiny ceiling and a real allocation: the guard must actually fire."""
    with pytest.raises(MemoryBudgetExceeded, match="exceeds budget"), MemoryBudget(
        limit_gib=0.001, label="tiny"
    ):
        _hold = torch.empty(16 * 1024 * 1024, dtype=torch.float16, device="cuda")
        del _hold


@CUDA
def test_budget_resets_peak_between_regions() -> None:
    """BUGS.md P-14: without the reset, region 2 inherits region 1's peak."""
    with MemoryBudget(limit_gib=4.0) as first:
        big = torch.empty(64 * 1024 * 1024, dtype=torch.float16, device="cuda")
        del big
    with MemoryBudget(limit_gib=4.0) as second:
        pass
    assert first.peak_gib > second.peak_gib


def test_budget_does_not_mask_inflight_exception() -> None:
    """An OOM inside the region is more informative than the budget failure it causes."""
    with pytest.raises(ZeroDivisionError), MemoryBudget(limit_gib=0.0, label="strict"):
        _ = 1 / 0


def test_non_strict_budget_records_without_raising() -> None:
    with MemoryBudget(limit_gib=0.0, strict=False) as budget:
        pass
    assert budget.peak_gib >= 0.0


# --- ledger ---------------------------------------------------------------------------------


def test_ledger_totals_and_verdict() -> None:
    ledger = BudgetLedger(limit_gib=4.0)
    ledger.register("lm weights", int(2.2 * GIB), "int4", "2.2B @ 4-bit")
    ledger.register("kv pool", int(0.5 * GIB), "float16", "512 blocks x 16 tokens")
    assert ledger.total_gib == pytest.approx(2.7, abs=1e-3)
    assert ledger.within_budget is True

    ledger.register("vision tower", int(2.0 * GIB), "float16")
    assert ledger.within_budget is False


def test_ledger_counts_buffers_not_just_parameters() -> None:
    """Rotary caches and masks are buffers, and are routinely forgotten in these tables."""

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(10, 10, bias=False, dtype=torch.float32)  # 400 B
            self.register_buffer("rope_cache", torch.zeros(100, dtype=torch.float32))  # 400 B

    ledger = BudgetLedger()
    ledger.register_module("tiny", Tiny())
    assert ledger.entries[0].bytes_ == 800
    assert ledger.entries[0].dtype == "float32"


def test_ledger_markdown_states_the_measurement_definition() -> None:
    """CONTEXT.md P3: silently excluding the CUDA context is what unravels the conversation."""
    ledger = BudgetLedger(limit_gib=4.0)
    ledger.register("lm weights", int(1.5 * GIB), "int4")
    ledger.cuda_context_bytes = int(0.4 * GIB)

    md = ledger.to_markdown()
    assert "max_memory_allocated" in md
    assert "Excludes the CUDA context" in md
    assert "0.400 GiB" in md
    assert "within budget" in md


def test_ledger_markdown_flags_overage() -> None:
    ledger = BudgetLedger(limit_gib=1.0)
    ledger.register("lm weights", int(2.0 * GIB), "float16")
    assert "**OVER BUDGET**" in ledger.to_markdown()


def test_ledger_serialises() -> None:
    ledger = BudgetLedger()
    ledger.register("x", 1024, "int8", "note")
    payload = ledger.to_dict()
    assert payload["entries"][0]["bytes"] == 1024
    assert payload["within_budget"] is True
