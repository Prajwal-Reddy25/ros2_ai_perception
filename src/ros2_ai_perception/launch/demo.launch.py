"""Launch prerecorded media and torchvision object detection together."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    input_path = LaunchConfiguration("input_path")
    model = LaunchConfiguration("model")
    device = LaunchConfiguration("device")
    confidence = LaunchConfiguration("confidence_threshold")
    frame_rate = LaunchConfiguration("frame_rate")
    loop = LaunchConfiguration("loop")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_path",
                description=(
                    "Absolute or shell-resolved path to an image, video, or image directory"
                ),
            ),
            DeclareLaunchArgument("model", default_value="ssdlite320_mobilenet_v3_large"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("confidence_threshold", default_value="0.50"),
            DeclareLaunchArgument("frame_rate", default_value="1.0"),
            DeclareLaunchArgument("loop", default_value="true"),
            Node(
                package="ros2_ai_perception",
                executable="media_publisher",
                name="media_publisher",
                output="screen",
                parameters=[
                    {
                        "path": input_path,
                        "frame_rate": frame_rate,
                        "loop": loop,
                        "output_topic": "/perception/image_raw",
                        "frame_id": "camera",
                    }
                ],
            ),
            Node(
                package="ros2_ai_perception",
                executable="perception_node",
                name="perception_node",
                output="screen",
                parameters=[
                    {
                        "model": model,
                        "device": device,
                        "confidence_threshold": confidence,
                        "input_image_topic": "/perception/image_raw",
                        "detections_topic": "/perception/detections",
                        "annotated_image_topic": "/perception/annotated",
                        "diagnostics_topic": "/perception/diagnostics",
                    }
                ],
            ),
        ]
    )
