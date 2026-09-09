"""Small-dataset detector evaluation with dependency-light AP50 metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import SUPPORTED_MODELS
from .device import select_device
from .pipeline import Detector
from .types import Detection


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """One labeled xyxy ground-truth box."""

    image_id: str
    label: str
    box: tuple[float, float, float, float]


def intersection_over_union(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    """Calculate IoU for two axis-aligned xyxy boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def evaluate_ap50(
    predictions: dict[str, tuple[Detection, ...]], ground_truth: tuple[GroundTruth, ...]
) -> dict[str, object]:
    """Compute class AP50 and micro precision/recall using greedy score matching."""
    labels = sorted({item.label for item in ground_truth})
    per_class: dict[str, object] = {}
    total_tp = total_fp = total_gt = 0
    aps: list[float] = []
    for label in labels:
        truths = [item for item in ground_truth if item.label == label]
        total_gt += len(truths)
        candidates = sorted(
            (
                (image_id, detection)
                for image_id, detections in predictions.items()
                for detection in detections
                if detection.label == label
            ),
            key=lambda pair: pair[1].score,
            reverse=True,
        )
        matched: set[int] = set()
        true_positive: list[int] = []
        false_positive: list[int] = []
        for image_id, detection in candidates:
            best_index = -1
            best_iou = 0.0
            box = (detection.x1, detection.y1, detection.x2, detection.y2)
            for index, truth in enumerate(truths):
                if index in matched or truth.image_id != image_id:
                    continue
                iou = intersection_over_union(box, truth.box)
                if iou > best_iou:
                    best_iou, best_index = iou, index
            is_match = best_index >= 0 and best_iou >= 0.5
            if is_match:
                matched.add(best_index)
            true_positive.append(int(is_match))
            false_positive.append(int(not is_match))
        tp = sum(true_positive)
        fp = sum(false_positive)
        total_tp += tp
        total_fp += fp
        ap = _average_precision(true_positive, false_positive, len(truths))
        aps.append(ap)
        per_class[label] = {
            "ap50": ap,
            "ground_truth_count": len(truths),
            "prediction_count": len(candidates),
            "true_positive": tp,
            "false_positive": fp,
        }
    return {
        "map50": sum(aps) / len(aps) if aps else 0.0,
        "micro_precision50": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "micro_recall50": total_tp / total_gt if total_gt else 0.0,
        "ground_truth_count": total_gt,
        "per_class": per_class,
    }


def _average_precision(true_positive: list[int], false_positive: list[int], gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    recall = [0.0]
    precision = [1.0]
    for tp, fp in zip(true_positive, false_positive, strict=True):
        cumulative_tp += tp
        cumulative_fp += fp
        recall.append(cumulative_tp / gt_count)
        precision.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    recall.append(1.0)
    precision.append(0.0)
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return sum(
        (recall[index] - recall[index - 1]) * precision[index]
        for index in range(1, len(recall))
        if recall[index] != recall[index - 1]
    )


def load_manifest(
    path: Path,
) -> tuple[Path, tuple[dict[str, object], ...], tuple[GroundTruth, ...]]:
    """Load the documented compact evaluation manifest and validate its shape."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("images"), list):
        raise ValueError("manifest must contain an 'images' list")
    images: list[dict[str, object]] = []
    truths: list[GroundTruth] = []
    for entry in raw["images"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError("each image entry must contain a string 'file'")
        image_id = str(entry.get("id", entry["file"]))
        annotations = entry.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError("image annotations must be a list")
        images.append({"id": image_id, "file": entry["file"]})
        for annotation in annotations:
            if not isinstance(annotation, dict) or not isinstance(annotation.get("label"), str):
                raise ValueError("each annotation needs a string label")
            box = annotation.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("bbox_xyxy must be a four-number list")
            numeric_box = tuple(float(value) for value in box)
            if numeric_box[2] < numeric_box[0] or numeric_box[3] < numeric_box[1]:
                raise ValueError("ground-truth boxes must satisfy x2 >= x1 and y2 >= y1")
            truths.append(GroundTruth(image_id, annotation["label"], numeric_box))
    return path.parent, tuple(images), tuple(truths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detector AP at IoU 0.50")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS[0])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--confidence", type=float, default=0.05)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run real model predictions for all manifest images and calculate AP50."""
    base, images, truths = load_manifest(args.manifest)
    device = select_device(args.device)
    detector = Detector(args.model, device, args.confidence)
    predictions: dict[str, tuple[Detection, ...]] = {}
    for entry in images:
        image_path = base / str(entry["file"])
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None or not frame.size:
            raise ValueError(f"could not decode evaluation image: {image_path}")
        predictions[str(entry["id"])] = detector.process(frame).detections
    report = {
        "schema_version": 1,
        "model": args.model,
        "device": device,
        "confidence_threshold": args.confidence,
        "iou_threshold": 0.5,
        "image_count": len(images),
        "metrics": evaluate_ap50(predictions, truths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"perception_evaluate: {exc}")
        raise SystemExit(2) from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
