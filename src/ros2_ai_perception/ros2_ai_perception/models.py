"""Supported torchvision detector construction and output conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .config import validate_model_name, validate_threshold
from .types import Detection


def load_torchvision_model(name: str, device: str) -> tuple[Any, tuple[str, ...]]:
    """Load an official pretrained torchvision detector and COCO categories."""
    validate_model_name(name)
    try:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            SSDLite320_MobileNet_V3_Large_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
            ssdlite320_mobilenet_v3_large,
        )
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "torchvision detection operators could not be imported; install matching "
            "official torch and torchvision builds"
        ) from exc

    if name == "ssdlite320_mobilenet_v3_large":
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        constructor = ssdlite320_mobilenet_v3_large
    else:
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        constructor = fasterrcnn_mobilenet_v3_large_320_fpn
    try:
        model = constructor(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load official pretrained weights for {name}; "
            "check network access and the PyTorch model cache"
        ) from exc
    model.to(device)
    model.eval()
    categories = tuple(str(item) for item in weights.meta["categories"])
    return model, categories


def bgr_to_tensor(image: np.ndarray, torch_module: Any | None = None) -> Any:
    """Convert an OpenCV BGR uint8 frame to a contiguous RGB float tensor."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if torch_module is None:
        import torch as torch_module
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    return torch_module.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)


def filter_predictions(
    output: Mapping[str, Any], categories: tuple[str, ...], threshold: float
) -> tuple[Detection, ...]:
    """Convert a torchvision output mapping into validated domain detections."""
    threshold = validate_threshold(threshold)
    required = ("boxes", "scores", "labels")
    missing = [key for key in required if key not in output]
    if missing:
        raise ValueError(f"model output missing keys: {', '.join(missing)}")
    boxes = output["boxes"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    labels = output["labels"].detach().cpu().tolist()
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("model output boxes, scores, and labels differ in length")
    detections: list[Detection] = []
    for box, score, label_id in zip(boxes, scores, labels, strict=True):
        score = float(score)
        if score < threshold:
            continue
        label_id = int(label_id)
        label = categories[label_id] if 0 <= label_id < len(categories) else str(label_id)
        detections.append(
            Detection(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                score=score,
                label_id=label_id,
                label=label,
            )
        )
    return tuple(detections)
