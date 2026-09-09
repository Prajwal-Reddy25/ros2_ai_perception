"""Conversion from domain detections to the installed Jazzy vision_msgs schema."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .types import Detection


def detection_array_message(detections: Iterable[Detection], header: Any) -> Any:
    """Build a Jazzy ``vision_msgs/Detection2DArray`` preserving the image header."""
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    output = Detection2DArray()
    output.header = header
    for index, item in enumerate(detections):
        message = Detection2D()
        message.header = header
        message.id = str(index)
        center_x, center_y = item.center
        message.bbox.center.position.x = center_x
        message.bbox.center.position.y = center_y
        message.bbox.center.theta = 0.0
        message.bbox.size_x = item.width
        message.bbox.size_y = item.height
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = item.label
        hypothesis.hypothesis.score = item.score
        message.results.append(hypothesis)
        output.detections.append(message)
    return output


def diagnostic_message(snapshot: dict[str, object], stamp: Any, device: str, model: str) -> Any:
    """Build a standard ROS diagnostic array from a metrics snapshot."""
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    output = DiagnosticArray()
    output.header.stamp = stamp
    status = DiagnosticStatus()
    status.level = DiagnosticStatus.OK
    status.name = "ros2_ai_perception/perception"
    status.hardware_id = device
    status.message = "perception active"
    values = {"model": model, "device": device, **_flatten(snapshot)}
    status.values = [KeyValue(key=str(key), value=str(value)) for key, value in values.items()]
    output.status.append(status)
    return output


def _flatten(value: dict[str, object], prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten(item, full_key))
        else:
            flattened[full_key] = item
    return flattened
