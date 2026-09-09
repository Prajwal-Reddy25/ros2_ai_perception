"""Small package-local suite executed by ``colcon test``."""

import unittest

from builtin_interfaces.msg import Time
from ros2_ai_perception.metrics import summarize
from ros2_ai_perception.ros_conversion import detection_array_message
from ros2_ai_perception.types import Detection
from std_msgs.msg import Header

from ros2_ai_perception.config import validate_threshold


class PackageRuntimeTest(unittest.TestCase):
    def test_core_runtime_and_jazzy_message_conversion(self):
        source = Detection(0, 10, 20, 30, 0.9, 18, "dog")
        header = Header(stamp=Time(sec=1), frame_id="camera")
        message = detection_array_message((source,), header)

        self.assertEqual(validate_threshold(0.5), 0.5)
        self.assertEqual(summarize([1.0, 2.0, 3.0]).median_ms, 2.0)
        self.assertEqual(message.detections[0].bbox.center.position.x, 10.0)
        self.assertEqual(message.detections[0].results[0].hypothesis.class_id, "dog")
