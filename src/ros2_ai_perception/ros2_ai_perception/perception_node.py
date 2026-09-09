"""ROS 2 torchvision object-detection node."""

from __future__ import annotations

import rclpy
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from .annotation import annotate_frame
from .config import validate_model_name, validate_threshold
from .device import DeviceUnavailableError, select_device
from .metrics import MetricsWindow
from .pipeline import Detector
from .ros_conversion import detection_array_message, diagnostic_message


class PerceptionNode(Node):
    """Subscribe to BGR images and publish detections, annotations, and diagnostics."""

    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("model", "ssdlite320_mobilenet_v3_large")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("input_image_topic", "/perception/image_raw")
        self.declare_parameter("detections_topic", "/perception/detections")
        self.declare_parameter("annotated_image_topic", "/perception/annotated")
        self.declare_parameter("diagnostics_topic", "/perception/diagnostics")
        self.declare_parameter("metrics_interval_frames", 10)
        self.declare_parameter("metrics_window_size", 500)

        self._model_name = validate_model_name(str(self.get_parameter("model").value))
        requested_device = str(self.get_parameter("device").value)
        self._device = select_device(requested_device)
        threshold = validate_threshold(float(self.get_parameter("confidence_threshold").value))
        interval = int(self.get_parameter("metrics_interval_frames").value)
        window_size = int(self.get_parameter("metrics_window_size").value)
        if interval < 1 or window_size < 1:
            raise ValueError("metrics_interval_frames and metrics_window_size must be positive")
        self._metrics_interval = interval

        self.get_logger().info(
            f"Loading {self._model_name} with official weights on {self._device}"
        )
        self._detector = Detector(self._model_name, self._device, threshold)
        self._bridge = CvBridge()
        self._metrics = MetricsWindow(window_size)
        self._failures = 0

        input_topic = str(self.get_parameter("input_image_topic").value)
        detection_topic = str(self.get_parameter("detections_topic").value)
        annotated_topic = str(self.get_parameter("annotated_image_topic").value)
        diagnostics_topic = str(self.get_parameter("diagnostics_topic").value)
        for name, value in (
            ("input_image_topic", input_topic),
            ("detections_topic", detection_topic),
            ("annotated_image_topic", annotated_topic),
            ("diagnostics_topic", diagnostics_topic),
        ):
            if not value.strip():
                raise ValueError(f"parameter {name!r} must not be empty")

        self._detection_publisher = self.create_publisher(
            Detection2DArray, detection_topic, qos_profile_sensor_data
        )
        self._image_publisher = self.create_publisher(
            Image, annotated_topic, qos_profile_sensor_data
        )
        diagnostics_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray, diagnostics_topic, diagnostics_qos
        )
        self._subscription = self.create_subscription(
            Image, input_topic, self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(f"Ready: {input_topic} -> {detection_topic}, {annotated_topic}")

    def _on_image(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            if frame is None or not frame.size:
                raise ValueError("received an empty image")
            result = self._detector.process(frame)
            annotated = annotate_frame(frame, result.detections)
        except (CvBridgeError, RuntimeError, TypeError, ValueError) as exc:
            self._failures += 1
            self.get_logger().error(f"Dropped frame after processing error: {exc}")
            return

        self._detection_publisher.publish(
            detection_array_message(result.detections, message.header)
        )
        annotated_message = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_message.header = message.header
        self._image_publisher.publish(annotated_message)

        end_to_end_ms = self._end_to_end_ms(message)
        self._metrics.add(result.inference_ms, result.processing_ms, end_to_end_ms)
        if self._metrics.processed_frame_count % self._metrics_interval == 0:
            snapshot = self._metrics.snapshot()
            snapshot["failed_frame_count"] = self._failures
            now = self.get_clock().now()
            self._diagnostics_publisher.publish(
                diagnostic_message(snapshot, now.to_msg(), self._device, self._model_name)
            )
            self.get_logger().info(
                f"frames={self._metrics.processed_frame_count} "
                f"inference_mean={snapshot['inference']['mean_ms']:.1f}ms "
                f"processing_mean={snapshot['processing']['mean_ms']:.1f}ms"
            )

    def _end_to_end_ms(self, message: Image) -> float | None:
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp_ns == 0:
            return None
        latency_ms = (self.get_clock().now().nanoseconds - stamp_ns) / 1_000_000.0
        return latency_ms if latency_ms >= 0.0 else None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PerceptionNode | None = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except (DeviceUnavailableError, RuntimeError, ValueError) as exc:
        if node is not None:
            node.get_logger().fatal(str(exc))
        else:
            print(f"perception_node: {exc}")
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
