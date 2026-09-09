# Local validation record

Validation was executed on 2026-09-09 on Ubuntu 24.04.4 LTS, ROS 2 Jazzy,
Python 3.12.3, AMD Ryzen 9 8940HX (16 cores / 32 threads), PyTorch 2.12.1+cpu,
torchvision 0.27.1+cpu, OpenCV 4.6.0, and NumPy 1.26.4.

## Commands and outcomes

| Check | Command | Actual outcome |
|---|---|---|
| Formatting | `.venv/bin/ruff format --check .` | passed |
| Lint | `.venv/bin/ruff check .` | passed |
| Unit suite | `.venv/bin/pytest -q` | 31 passed, 1 model smoke skipped |
| Real CPU inference | `RUN_MODEL_SMOKE=1 .venv/bin/pytest -q tests/test_smoke_inference.py -s` | 1 passed; real official weights |
| ROS build | `.venv/bin/python -m colcon build --symlink-install` | 1 package finished |
| ROS executable discovery | `ros2 pkg executables ros2_ai_perception` | 5 executables found |
| Launch parsing | `ros2 launch ros2_ai_perception demo.launch.py --show-args` | passed; 6 arguments shown |
| ROS graph smoke | `./scripts/ros_smoke_test.sh` | passed; all three output topics received |
| Package tests | `.venv/bin/python -m colcon test` | 1 passed; zero errors/failures/skips |

The development graph run received all three outputs from independent ROS CLI
subscriptions:

- `Detection2DArray`: one `dog`, score 0.9912348985671997, width 1076.035 px, height 1239.130 px
- annotated `Image` header: original `camera` frame ID and timestamp preserved
- `DiagnosticArray`: frame count, approximate FPS, and mean inference, processing, and end-to-end latency keys

## Measured model comparison

Both models used identical input, threshold 0.5, 3 excluded warm-ups, and 15
measured CPU iterations. SSDLite measured 30.15 ms mean inference and 41.68 ms
mean processing. Faster R-CNN MobileNet V3 320 FPN measured 53.03 ms mean
inference and 61.15 ms mean processing. Full environment data and distributions
are retained under `results/`.

CUDA was not tested. The installed NVIDIA GPU was visible through `nvidia-smi`,
but the validated PyTorch wheel is CPU-only. No training or dataset accuracy
evaluation was performed.
