from types import SimpleNamespace

import pytest
from ros2_ai_perception.device import DeviceUnavailableError, select_device

from ros2_ai_perception.config import (
    validate_input_path,
    validate_model_name,
    validate_threshold,
)


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_threshold_accepts_closed_unit_interval(value):
    assert validate_threshold(value) == value


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_threshold_rejects_out_of_range(value):
    with pytest.raises(ValueError, match="threshold"):
        validate_threshold(value)


def test_model_name_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported model"):
        validate_model_name("not-a-model")


def test_missing_input_path_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_input_path(tmp_path / "missing.mp4")


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [("cpu", False, "cpu"), ("auto", False, "cpu"), ("auto", True, "cuda"), ("cuda", True, "cuda")],
)
def test_device_selection(requested, available, expected):
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
    assert select_device(requested, torch_stub) == expected


def test_explicit_unavailable_cuda_fails():
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(DeviceUnavailableError, match="CUDA was requested"):
        select_device("cuda", torch_stub)


def test_invalid_device_fails_before_importing_torch():
    with pytest.raises(ValueError, match="device"):
        select_device("tpu")
