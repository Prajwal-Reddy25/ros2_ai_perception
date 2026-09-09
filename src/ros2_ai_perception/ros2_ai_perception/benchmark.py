"""Reproducible single-frame model benchmark command."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from .config import SUPPORTED_MODELS
from .device import select_device
from .media import MediaSource
from .metrics import MetricsWindow
from .pipeline import Detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a supported detector")
    parser.add_argument("input", help="Input image or media")
    parser.add_argument("--output", type=Path, help="Write report JSON")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS[0])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Benchmark repeated inference while excluding explicit warm-up samples."""
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    device = select_device(args.device)
    detector = Detector(args.model, device, args.confidence)
    with MediaSource(args.input) as source:
        frame = source.read()
    if frame is None:
        raise ValueError("input contained no frame")
    for _ in range(args.warmup):
        detector.process(frame)
    metrics = MetricsWindow(max_samples=args.iterations)
    detection_counts: list[int] = []
    for _ in range(args.iterations):
        result = detector.process(frame)
        metrics.add(result.inference_ms, result.processing_ms)
        detection_counts.append(len(result.detections))
    import torch
    import torchvision

    report: dict[str, object] = {
        "schema_version": 1,
        "model": args.model,
        "official_pretrained_weights": True,
        "device": device,
        "confidence_threshold": args.confidence,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "detections_per_iteration": detection_counts,
        "metrics": metrics.snapshot(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"perception_benchmark: {exc}")
        raise SystemExit(2) from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
