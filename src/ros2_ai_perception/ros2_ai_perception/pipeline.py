"""Reusable detector pipeline independent of ROS 2."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import validate_threshold
from .models import bgr_to_tensor, filter_predictions, load_torchvision_model
from .types import ProcessResult


class Detector:
    """Own a torchvision model and run measured inference on BGR frames."""

    def __init__(
        self,
        model_name: str,
        device: str,
        confidence_threshold: float,
        *,
        model: Any | None = None,
        categories: tuple[str, ...] | None = None,
    ) -> None:
        self.device = device
        self.confidence_threshold = validate_threshold(confidence_threshold)
        if model is None:
            model, categories = load_torchvision_model(model_name, device)
        if categories is None:
            raise ValueError("categories are required with an injected model")
        self.model = model
        self.categories = categories

    def _synchronize(self) -> None:
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()

    def process(self, frame: np.ndarray) -> ProcessResult:
        """Preprocess, infer, filter, and return separate timing measurements."""
        import torch

        processing_start = time.perf_counter()
        tensor = bgr_to_tensor(frame, torch).to(self.device)
        self._synchronize()
        inference_start = time.perf_counter()
        try:
            with torch.inference_mode():
                outputs = self.model([tensor])
            self._synchronize()
        except Exception as exc:
            raise RuntimeError(f"model inference failed on {self.device}: {exc}") from exc
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError("detector must return one output mapping for one input frame")
        detections = filter_predictions(outputs[0], self.categories, self.confidence_threshold)
        processing_ms = (time.perf_counter() - processing_start) * 1000.0
        return ProcessResult(detections, inference_ms, processing_ms)
