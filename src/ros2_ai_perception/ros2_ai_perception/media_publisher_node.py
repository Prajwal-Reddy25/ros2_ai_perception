"""ROS 2 node that publishes prerecorded media as sensor_msgs/Image."""

from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .media import MediaSource, UnsupportedMediaError


class MediaPublisherNode(Node):
    """Publish a file or image sequence at a configurable rate."""

    def __init__(self) -> None:
        super().__init__("media_publisher")
        self.declare_parameter("path", "")
        self.declare_parameter("frame_rate", 5.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("output_topic", "/perception/image_raw")
        self.declare_parameter("frame_id", "camera")

        path = str(self.get_parameter("path").value)
        frame_rate = float(self.get_parameter("frame_rate").value)
        if not path.strip():
            raise ValueError("parameter 'path' must identify an image, video, or image directory")
        if frame_rate <= 0.0:
            raise ValueError("parameter 'frame_rate' must be greater than zero")
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._source = MediaSource(path, loop=bool(self.get_parameter("loop").value))
        topic = str(self.get_parameter("output_topic").value)
        if not topic.strip():
            raise ValueError("parameter 'output_topic' must not be empty")
        self._publisher = self.create_publisher(Image, topic, qos_profile_sensor_data)
        self._bridge = CvBridge()
        self._published = 0
        self._timer = self.create_timer(1.0 / frame_rate, self._publish_next)
        self.get_logger().info(f"Publishing {path!r} to {topic!r} at {frame_rate:.3f} Hz")

    def _publish_next(self) -> None:
        try:
            frame = self._source.read()
        except UnsupportedMediaError as exc:
            self.get_logger().error(str(exc))
            self._stop()
            return
        if frame is None:
            self.get_logger().info(f"End of media after {self._published} published frame(s)")
            self._stop()
            return
        message = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        self._publisher.publish(message)
        self._published += 1

    def _stop(self) -> None:
        self._timer.cancel()
        self._source.close()
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        self._source.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: MediaPublisherNode | None = None
    try:
        node = MediaPublisherNode()
        rclpy.spin(node)
    except (FileNotFoundError, ValueError, UnsupportedMediaError) as exc:
        if node is not None:
            node.get_logger().fatal(str(exc))
        else:
            print(f"media_publisher: {exc}")
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
