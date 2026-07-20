"""Lightweight per-step timing so each pipeline stage's cost is visible.

Usage::

    timer = StepTimer(LOGGER)
    with timer.step("legend: roboflow detect"):
        ...
    LOGGER.info(timer.summary())         # table, sorted in recorded order
    timer.as_dict()                      # JSON-friendly artefact

Each ``step`` logs its own duration as it finishes (so you see progress live),
and :meth:`summary` prints a table with per-step share of the total.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple


class StepTimer:
    """Records named durations and renders a summary table."""

    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._records: List[Tuple[str, float]] = []

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Time a block and log its duration when it completes."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._records.append((name, dt))
            self._logger.info("[timing] %-30s %8.2fs", name, dt)

    def add(self, name: str, seconds: float) -> None:
        """Record a pre-measured duration (for work not wrapped by ``step``)."""
        self._records.append((name, seconds))
        self._logger.info("[timing] %-30s %8.2fs", name, seconds)

    @property
    def total(self) -> float:
        return sum(d for _, d in self._records)

    def summary(self) -> str:
        """A ranked-by-recording-order table, each step's share of the total."""
        total = self.total
        lines: List[str] = ["", "===== Timing summary ====="]
        for name, d in self._records:
            pct = (100.0 * d / total) if total else 0.0
            lines.append(f"  {name:<32}{d:8.2f}s  {pct:5.1f}%")
        lines.append("  " + "-" * 47)
        lines.append(f"  {'TOTAL':<32}{total:8.2f}s  100.0%")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly view for a ``timings.json`` artefact."""
        return {
            "steps": [
                {"name": n, "seconds": round(d, 3)} for n, d in self._records
            ],
            "total_seconds": round(self.total, 3),
        }
