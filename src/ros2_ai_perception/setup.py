from glob import glob

from setuptools import find_packages, setup

package_name = "ros2_ai_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Prajwal HB",
    maintainer_email="prajwalhb31@gmail.com",
    description="CPU-first ROS 2 object detection with torchvision.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "media_publisher = ros2_ai_perception.media_publisher_node:main",
            "perception_node = ros2_ai_perception.perception_node:main",
            "standalone_inference = ros2_ai_perception.standalone:main",
            "perception_benchmark = ros2_ai_perception.benchmark:main",
            "perception_evaluate = ros2_ai_perception.evaluation:main",
        ],
    },
)
