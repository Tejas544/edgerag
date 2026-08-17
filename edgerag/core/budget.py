"""Memory budget enforcement and accounting.

The thesis of this project is a *constraint*: the whole pipeline fits in <= 4 GB. A constraint that
is audited once at the end is not a constraint -- it is a hope. So it is enforced from Phase 0,
in tests, and every component registers its own footprint as it is built.

Two things live here:

* :class:`MemoryBudget` -- a context manager that fails loudly when a region exceeds its ceiling.
* :class:`BudgetLedger` -- the running accounting table that becomes the README deliverable
  (``01_EDGERAG.md`` §9: *"Memory-budget accounting table summing under 4 GB"*).

Building the ledger incrementally is deliberate. Reconstructing "where did the 4.3 GB go?" on the
last afternoon is a bad afternoon; having each component declare its cost at the moment it is
written costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

GIB = 1024**3

#: The project-wide ceiling from ``01_EDGERAG.md`` §6.
DEFAULT_BUDGET_GIB = 4.0


class MemoryBudgetExceeded(RuntimeError):
    """Raised when a measured region's peak allocation exceeds its declared ceiling."""


class MemoryBudget:
    """Assert that peak *allocated* memory inside the region stays under ``limit_gib``.

    Measures ``torch.cuda.max_memory_allocated`` -- what the model actually used -- and resets the
    peak counter on entry (``BUGS.md`` P-14). Deliberately *not* ``max_memory_reserved``: the
    caching allocator's high-water mark is an allocator artifact, not a model property, and using
    it would make the budget depend on allocation history. The README states which one it means
    (``CONTEXT.md`` P3).

    The CUDA context (~300-600 MB) sits outside ``max_memory_allocated`` entirely. It is reported
    separately rather than folded in, because folding it in silently is exactly the kind of thing
    that unravels an otherwise good interview conversation.

    On CPU this is a no-op that still records zero, so tests run identically on both tiers.
    """

    def __init__(
        self,
        limit_gib: float = DEFAULT_BUDGET_GIB,
        label: str = "pipeline",
        *,
        strict: bool = True,
    ) -> None:
        self.limit_gib = limit_gib
        self.label = label
        self.strict = strict
        self.peak_gib: float = 0.0

    def __enter__(self) -> MemoryBudget:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_gib = torch.cuda.max_memory_allocated() / GIB

        # Do not mask an in-flight exception with a budget failure -- the original traceback is
        # more useful, and an OOM will often be the in-flight exception.
        if exc_type is not None:
            return
        if self.strict and self.peak_gib > self.limit_gib:
            raise MemoryBudgetExceeded(
                f"{self.label}: peak allocated {self.peak_gib:.3f} GiB "
                f"exceeds budget of {self.limit_gib:.3f} GiB "
                f"(over by {self.peak_gib - self.limit_gib:.3f} GiB)"
            )

    @property
    def headroom_gib(self) -> float:
        return self.limit_gib - self.peak_gib


@dataclass
class LedgerEntry:
    component: str
    bytes_: int
    dtype: str
    note: str = ""

    @property
    def gib(self) -> float:
        return self.bytes_ / GIB


@dataclass
class BudgetLedger:
    """Running memory accounting, one entry per pipeline component.

    Components register themselves as they are constructed. The table is a deliverable, so it is
    built by the code that knows the answer rather than assembled by hand at the end.
    """

    limit_gib: float = DEFAULT_BUDGET_GIB
    entries: list[LedgerEntry] = field(default_factory=list)
    cuda_context_bytes: int | None = None

    def register(self, component: str, bytes_: int, dtype: str, note: str = "") -> None:
        self.entries.append(LedgerEntry(component, bytes_, dtype, note))

    def register_module(self, component: str, module: torch.nn.Module, note: str = "") -> None:
        """Register a module's parameter + buffer footprint.

        Counts buffers as well as parameters: rotary caches and attention masks are real memory
        and are routinely forgotten in these tables.
        """
        total = sum(p.numel() * p.element_size() for p in module.parameters())
        total += sum(b.numel() * b.element_size() for b in module.buffers())
        dtypes = {str(p.dtype).replace("torch.", "") for p in module.parameters()}
        self.register(component, total, "/".join(sorted(dtypes)) or "n/a", note)

    @property
    def total_bytes(self) -> int:
        return sum(e.bytes_ for e in self.entries)

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def within_budget(self) -> bool:
        return self.total_gib <= self.limit_gib

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit_gib": self.limit_gib,
            "total_gib": round(self.total_gib, 4),
            "within_budget": self.within_budget,
            "cuda_context_bytes": self.cuda_context_bytes,
            "entries": [
                {
                    "component": e.component,
                    "bytes": e.bytes_,
                    "gib": round(e.gib, 4),
                    "dtype": e.dtype,
                    "note": e.note,
                }
                for e in self.entries
            ],
        }

    def to_markdown(self, basis: str = "Measured as `torch.cuda.max_memory_allocated()`.") -> str:
        """``basis`` states where the numbers came from, and it is not decoration.

        The same table is produced two ways: measured from a running pipeline, and *computed*
        exactly from shapes before anything runs (``scripts/measure_memory_ledger.py``). Printing
        "measured" over a computed table is a small lie that invalidates the whole document, so
        the caller that knows must say.
        """
        lines = [
            "| component | dtype | bytes | GiB | note |",
            "|---|:--:|---:|---:|---|",
        ]
        for e in sorted(self.entries, key=lambda x: -x.bytes_):
            lines.append(
                f"| {e.component} | {e.dtype} | {e.bytes_:,} | {e.gib:.4f} | {e.note} |"
            )
        verdict = "within budget" if self.within_budget else "**OVER BUDGET**"
        lines.append(
            f"| **total** | | **{self.total_bytes:,}** | **{self.total_gib:.4f}** | "
            f"budget {self.limit_gib:.2f} GiB -- {verdict} |"
        )
        context = (
            f"~{self.cuda_context_bytes / GIB:.3f} GiB"
            if self.cuda_context_bytes is not None
            # Printing "~0.000 GiB" for something never measured is worse than saying so: the
            # context is 300-600 MiB on these cards, which is 8-15% of a 4 GiB budget.
            else "**not measured on this device** -- 300-600 MiB on Turing, per CONTEXT.md P3"
        )
        lines.append("")
        lines.append(
            f"{basis} **Excludes the CUDA context** ({context}), which is reported separately "
            "because it is a driver cost, not a pipeline cost."
        )
        return "\n".join(lines) + "\n"
