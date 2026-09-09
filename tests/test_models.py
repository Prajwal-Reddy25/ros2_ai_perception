import numpy as np
import pytest
import torch
from ros2_ai_perception.models import bgr_to_tensor, filter_predictions


def test_bgr_to_tensor_converts_channel_order_and_scale():
    image = np.array([[[0, 127, 255]]], dtype=np.uint8)
    tensor = bgr_to_tensor(image)
    assert tuple(tensor.shape) == (3, 1, 1)
    assert tensor[0, 0, 0].item() == pytest.approx(1.0)
    assert tensor[1, 0, 0].item() == pytest.approx(127 / 255)
    assert tensor[2, 0, 0].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 0, 3), dtype=np.uint8),
        np.zeros((3, 3), dtype=np.uint8),
        np.zeros((3, 3, 3), dtype=np.float32),
    ],
)
def test_bgr_to_tensor_rejects_malformed_images(image):
    with pytest.raises(ValueError):
        bgr_to_tensor(image)


def test_filter_predictions_applies_threshold_and_categories():
    output = {
        "boxes": torch.tensor([[1, 2, 11, 22], [3, 4, 8, 9]], dtype=torch.float32),
        "scores": torch.tensor([0.9, 0.2]),
        "labels": torch.tensor([1, 2]),
    }
    detections = filter_predictions(output, ("background", "person", "car"), 0.5)
    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].center == pytest.approx((6.0, 12.0))


def test_filter_predictions_requires_complete_equal_length_output():
    with pytest.raises(ValueError, match="missing keys"):
        filter_predictions({"boxes": torch.empty((0, 4))}, (), 0.5)
    with pytest.raises(ValueError, match="differ in length"):
        filter_predictions(
            {
                "boxes": torch.empty((1, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.int64),
            },
            (),
            0.5,
        )
