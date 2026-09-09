"""OpenCV visualization utilities."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


def annotate_frame(frame: np.ndarray, detections: tuple[Detection, ...]) -> np.ndarray:
    """Draw detections on a copy of a BGR frame."""
    output = frame.copy()
    height, width = output.shape[:2]
    for detection in detections:
        x1 = int(max(0, min(width - 1, round(detection.x1))))
        y1 = int(max(0, min(height - 1, round(detection.y1))))
        x2 = int(max(0, min(width - 1, round(detection.x2))))
        y2 = int(max(0, min(height - 1, round(detection.y2))))
        color = _label_color(detection.label_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        text = f"{detection.label} {detection.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(output, (x1, top), (min(width - 1, x1 + text_width + 4), y1), color, -1)
        cv2.putText(
            output,
            text,
            (x1 + 2, max(text_height + 1, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def _label_color(label_id: int) -> tuple[int, int, int]:
    return (
        48 + (label_id * 37) % 176,
        48 + (label_id * 67) % 176,
        48 + (label_id * 97) % 176,
    )
