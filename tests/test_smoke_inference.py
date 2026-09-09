import os
from pathlib import Path

import cv2
import pytest
from ros2_ai_perception.pipeline import Detector


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_MODEL_SMOKE") != "1", reason="set RUN_MODEL_SMOKE=1")
def test_real_cpu_inference_path():
    root = Path(__file__).parents[1]
    image = cv2.imread(str(root / "media" / "downloads" / "demo_dog.jpg"))
    assert image is not None, "run scripts/prepare_sample_media.sh first"
    detector = Detector("ssdlite320_mobilenet_v3_large", "cpu", 0.5)
    result = detector.process(image)
    assert result.inference_ms > 0
    assert result.processing_ms >= result.inference_ms
    assert any(item.label == "dog" for item in result.detections)
