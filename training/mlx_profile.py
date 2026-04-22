"""Lightweight rolling profiling helpers for MLX training loops."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


PROFILE_BUCKETS = (
    "batch_fetch",
    "numpy_to_mlx",
    "forward_backward",
    "opt_step",
    "eval",
    "checkpoint",
)

PROFILE_LABELS = {
    "batch_fetch": "batch_fetch",
    "numpy_to_mlx": "numpy_to_mlx",
    "forward_backward": "forward_backward",
    "opt_step": "opt_step",
    "eval": "eval",
    "checkpoint": "checkpoint",
}


@dataclass
class MLXProfile:
    """Rolling timing buckets for optional MLX profiling output."""

    enabled: bool = False
    totals: dict[str, float] = field(
        default_factory=lambda: {bucket: 0.0 for bucket in PROFILE_BUCKETS}
    )

    def add(self, bucket: str, elapsed: float) -> None:
        if not self.enabled:
            return
        if bucket not in self.totals:
            raise KeyError(f"Unknown MLX profile bucket: {bucket}")
        self.totals[bucket] += max(0.0, elapsed)

    def reset(self) -> None:
        for bucket in PROFILE_BUCKETS:
            self.totals[bucket] = 0.0

    def format_report(self, *, window_label: str) -> str:
        if not self.enabled:
            return ""
        total = sum(self.totals.values())
        if total <= 0.0:
            return f"MLX profile | {window_label} | no timing samples collected"

        parts = [f"MLX profile | {window_label}"]
        for bucket in PROFILE_BUCKETS:
            bucket_total = self.totals[bucket]
            pct = (bucket_total / total) * 100.0 if total > 0.0 else 0.0
            parts.append(f"{PROFILE_LABELS[bucket]} {bucket_total * 1000.0:.1f}ms ({pct:.1f}%)")
        return " | ".join(parts)


def now() -> float:
    """Small wrapper so callers use the same high-resolution clock."""
    return time.perf_counter()
