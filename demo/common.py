"""Shared timing utilities for the latency demonstrations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class LatencyLog:
    """Collects per-decision latencies and reports summary stats."""

    label: str
    samples_ms: list[float] = field(default_factory=list)

    def record(self, started: float) -> float:
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.samples_ms.append(elapsed_ms)
        return elapsed_ms

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    @property
    def mean(self) -> float:
        return sum(self.samples_ms) / max(1, len(self.samples_ms))

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def min(self) -> float:
        return min(self.samples_ms) if self.samples_ms else 0.0

    @property
    def max(self) -> float:
        return max(self.samples_ms) if self.samples_ms else 0.0

    def _percentile(self, pct: float) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
        return ordered[idx]

    def summary(self) -> str:
        return (
            f"{self.label}: n={self.count}  "
            f"mean={self.mean:.1f}ms  p50={self.p50:.1f}ms  "
            f"p95={self.p95:.1f}ms  min={self.min:.1f}ms  max={self.max:.1f}ms"
        )


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
