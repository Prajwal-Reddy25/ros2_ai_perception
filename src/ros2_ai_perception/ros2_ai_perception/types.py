"""Dependency-light domain types used across ROS and standalone paths."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """One axis-aligned image-space object detection."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label_id: int
    label: str

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("detection box must satisfy x2 >= x1 and y2 >= y1")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("detection score must be in [0, 1]")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "bbox_xyxy": [self.x1, self.y1, self.x2, self.y2],
            "score": self.score,
            "label_id": self.label_id,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Detector output and timing for one frame."""

    detections: tuple[Detection, ...]
    inference_ms: float
    processing_ms: float
