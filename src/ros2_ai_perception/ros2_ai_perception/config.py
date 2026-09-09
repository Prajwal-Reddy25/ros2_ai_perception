"""Shared validation for CLI and ROS parameters."""

from pathlib import Path

SUPPORTED_MODELS = (
    "ssdlite320_mobilenet_v3_large",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
)
SUPPORTED_DEVICES = ("auto", "cpu", "cuda")


def validate_threshold(value: float) -> float:
    """Validate and normalize a confidence threshold."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    return value


def validate_model_name(name: str) -> str:
    """Reject unknown model identifiers before loading dependencies or weights."""
    if name not in SUPPORTED_MODELS:
        choices = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"unsupported model {name!r}; choose one of: {choices}")
    return name


def validate_input_path(path: str | Path) -> Path:
    """Resolve a media path and require a regular file or directory."""
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"input media does not exist: {candidate}")
    if not (candidate.is_file() or candidate.is_dir()):
        raise ValueError(f"input media is neither a file nor directory: {candidate}")
    return candidate.resolve()
