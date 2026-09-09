"""PyTorch device selection policy."""

from __future__ import annotations

from typing import Any

from .config import SUPPORTED_DEVICES


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested accelerator is unavailable."""


def select_device(requested: str, torch_module: Any | None = None) -> str:
    """Resolve ``auto``, ``cpu``, or ``cuda`` with strict explicit-CUDA behavior.

    ``auto`` selects CUDA only when PyTorch reports it available. Explicit
    ``cuda`` fails clearly rather than silently changing benchmark semantics.
    """
    normalized = requested.strip().lower()
    if normalized not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(SUPPORTED_DEVICES)}")
    if normalized == "cpu":
        return "cpu"
    if torch_module is None:
        import torch as torch_module

    available = bool(torch_module.cuda.is_available())
    if normalized == "cuda" and not available:
        raise DeviceUnavailableError(
            "CUDA was requested but is unavailable to this PyTorch installation; "
            "use device:=cpu/auto or install a compatible CUDA-enabled PyTorch wheel"
        )
    return "cuda" if available else "cpu"
