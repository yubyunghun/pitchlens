"""Where does the time go? A tiny per-stage stopwatch for a run.

    timer = StageTimer()
    with timer("detection"):
        ...
    print(timer.summary(frames=750))
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class StageTimer:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.seconds: dict[str, float] = defaultdict(float)
        self._order: list[str] = []

    @contextmanager
    def __call__(self, stage: str):
        if stage not in self.seconds:
            self._order.append(stage)
        start = self.clock()
        try:
            yield
        finally:
            self.seconds[stage] += self.clock() - start

    def add(self, stage: str, seconds: float) -> None:
        """Record time measured elsewhere (e.g. around a large block)."""
        if stage not in self.seconds:
            self._order.append(stage)
        self.seconds[stage] += seconds

    @property
    def total(self) -> float:
        return sum(self.seconds.values())

    def summary(self, frames: int | None = None) -> str:
        total = self.total
        if total <= 0:
            return "Time per stage: nothing measured"
        head = f"Time per stage ({total:.1f} s"
        if frames:
            head += f", {frames / total:.1f} frames/s overall"
        lines = [head + "):"]
        for stage in sorted(self._order, key=lambda s: -self.seconds[s]):
            sec = self.seconds[stage]
            lines.append(f"  {stage:<22} {sec:7.1f} s  {sec / total:4.0%}")
        return "\n".join(lines)
