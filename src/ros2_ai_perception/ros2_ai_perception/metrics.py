"""Bounded timing aggregation shared by ROS diagnostics and benchmarks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean, median


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Descriptive statistics in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    sample_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "sample_count": self.sample_count,
        }


def summarize(values: list[float] | tuple[float, ...]) -> DistributionSummary:
    """Summarize measured samples using nearest-rank p95."""
    if not values:
        raise ValueError("at least one sample is required")
    ordered = sorted(float(value) for value in values)
    rank = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return DistributionSummary(mean(ordered), median(ordered), ordered[rank], len(ordered))


class MetricsWindow:
    """Track recent inference, processing, and optional end-to-end latencies."""

    def __init__(self, max_samples: int = 500) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.inference_ms: deque[float] = deque(maxlen=max_samples)
        self.processing_ms: deque[float] = deque(maxlen=max_samples)
        self.end_to_end_ms: deque[float] = deque(maxlen=max_samples)
        self.processed_frame_count = 0

    def add(
        self, inference_ms: float, processing_ms: float, end_to_end_ms: float | None = None
    ) -> None:
        """Add one successful frame without conflating distinct timings."""
        if (
            inference_ms < 0
            or processing_ms < 0
            or (end_to_end_ms is not None and end_to_end_ms < 0)
        ):
            raise ValueError("latencies must be non-negative")
        self.inference_ms.append(float(inference_ms))
        self.processing_ms.append(float(processing_ms))
        if end_to_end_ms is not None:
            self.end_to_end_ms.append(float(end_to_end_ms))
        self.processed_frame_count += 1

    def snapshot(self) -> dict[str, object]:
        """Return JSON-safe statistics for all populated timing streams."""
        if not self.processing_ms:
            return {"processed_frame_count": self.processed_frame_count}
        processing = summarize(tuple(self.processing_ms))
        result: dict[str, object] = {
            "processed_frame_count": self.processed_frame_count,
            "window_sample_count": len(self.processing_ms),
            "approx_throughput_fps": 1000.0 / processing.mean_ms if processing.mean_ms else 0.0,
            "inference": summarize(tuple(self.inference_ms)).as_dict(),
            "processing": processing.as_dict(),
        }
        if self.end_to_end_ms:
            result["end_to_end"] = summarize(tuple(self.end_to_end_ms)).as_dict()
        return result
