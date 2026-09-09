"""Standalone image inference command."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2

from .annotation import annotate_frame
from .config import SUPPORTED_MODELS
from .device import DeviceUnavailableError, select_device
from .media import MediaSource, UnsupportedMediaError
from .pipeline import Detector

LOGGER = logging.getLogger("ros2_ai_perception.standalone")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run torchvision object detection on media")
    parser.add_argument("input", help="Image, video, or image directory")
    parser.add_argument("--output", type=Path, required=True, help="Annotated image output path")
    parser.add_argument("--json-output", type=Path, help="Optional detections/timing JSON")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS[0])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--confidence", type=float, default=0.5)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one-frame standalone inference and write genuine output."""
    device = select_device(args.device)
    detector = Detector(args.model, device, args.confidence)
    with MediaSource(args.input, loop=False) as source:
        frame = source.read()
    if frame is None:
        raise ValueError("input contained no decodable frame")
    result = detector.process(frame)
    annotated = annotate_frame(frame, result.detections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise OSError(f"failed to write annotated image: {args.output}")
    report: dict[str, object] = {
        "model": args.model,
        "device": device,
        "confidence_threshold": args.confidence,
        "inference_ms": result.inference_ms,
        "processing_ms": result.processing_ms,
        "detections": [item.as_dict() for item in result.detections],
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (
        DeviceUnavailableError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        UnsupportedMediaError,
        ValueError,
    ) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
