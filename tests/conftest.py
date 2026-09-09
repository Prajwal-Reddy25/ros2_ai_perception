"""Make the ROS Python package importable without an editable install."""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "ros2_ai_perception"
sys.path.insert(0, str(PACKAGE_ROOT))
