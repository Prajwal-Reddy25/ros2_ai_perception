#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

image="${1:-${repo_root}/media/downloads/demo_dog.jpg}"
if [[ ! -f "${image}" ]]; then
  echo "Input image not found: ${image}" >&2
  echo "Run scripts/prepare_sample_media.sh first." >&2
  exit 2
fi

smoke_dir="$(mktemp -d /tmp/ros2_ai_perception_smoke.XXXXXX)"
launch_pid=""
cleanup() {
  if [[ -n "${launch_pid}" ]]; then
    kill "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
  fi
  rm -rf "${smoke_dir}"
}
trap cleanup EXIT

ros2 launch ros2_ai_perception demo.launch.py \
  input_path:="${image}" frame_rate:=2.0 loop:=true device:=cpu \
  >"${smoke_dir}/launch.log" 2>&1 &
launch_pid=$!

for _ in $(seq 1 30); do
  if ros2 topic list | grep -qx '/perception/detections'; then
    break
  fi
  sleep 1
done

timeout 30 ros2 topic echo --once /perception/detections >"${smoke_dir}/detections.yaml"
timeout 30 ros2 topic echo --once /perception/annotated --field header \
  >"${smoke_dir}/annotated_header.yaml"
timeout 30 ros2 topic echo --once /perception/diagnostics >"${smoke_dir}/diagnostics.yaml"

grep -q 'class_id: dog' "${smoke_dir}/detections.yaml"
grep -q 'frame_id: camera' "${smoke_dir}/annotated_header.yaml"
grep -q 'inference.mean_ms' "${smoke_dir}/diagnostics.yaml"
grep -q 'processing.mean_ms' "${smoke_dir}/diagnostics.yaml"
grep -q 'end_to_end.mean_ms' "${smoke_dir}/diagnostics.yaml"

echo "ROS smoke test passed: detections, annotated image, and diagnostics received."
