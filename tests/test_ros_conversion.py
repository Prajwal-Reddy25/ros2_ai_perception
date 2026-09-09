import pytest

pytest.importorskip("builtin_interfaces.msg", reason="ROS 2 messages are not installed")

from builtin_interfaces.msg import Time
from ros2_ai_perception.ros_conversion import detection_array_message, diagnostic_message
from ros2_ai_perception.types import Detection
from std_msgs.msg import Header


def test_detection_conversion_matches_jazzy_schema():
    header = Header(stamp=Time(sec=12, nanosec=34), frame_id="camera")
    source = Detection(10, 20, 30, 60, 0.875, 18, "dog")
    message = detection_array_message((source,), header)
    assert message.header == header
    assert len(message.detections) == 1
    detection = message.detections[0]
    assert detection.header == header
    assert detection.bbox.center.position.x == 20.0
    assert detection.bbox.center.position.y == 40.0
    assert detection.bbox.size_x == 20.0
    assert detection.bbox.size_y == 40.0
    assert detection.results[0].hypothesis.class_id == "dog"
    assert detection.results[0].hypothesis.score == 0.875


def test_diagnostic_conversion_flattens_metrics():
    message = diagnostic_message(
        {"processed_frame_count": 2, "inference": {"mean_ms": 4.5}},
        Time(sec=1),
        "cpu",
        "model",
    )
    values = {item.key: item.value for item in message.status[0].values}
    assert values["model"] == "model"
    assert values["device"] == "cpu"
    assert values["inference.mean_ms"] == "4.5"
