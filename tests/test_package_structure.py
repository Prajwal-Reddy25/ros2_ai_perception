from pathlib import Path


def test_ros_package_structure_and_entry_points_exist():
    root = Path(__file__).parents[1]
    package = root / "src" / "ros2_ai_perception"
    assert (package / "package.xml").is_file()
    assert (package / "resource" / "ros2_ai_perception").is_file()
    assert (package / "launch" / "demo.launch.py").is_file()
    setup = (package / "setup.py").read_text(encoding="utf-8")
    for executable in (
        "media_publisher",
        "perception_node",
        "standalone_inference",
        "perception_benchmark",
        "perception_evaluate",
    ):
        assert f'"{executable} =' in setup
