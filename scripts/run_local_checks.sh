#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv; follow the README setup first." >&2
  exit 2
fi

set +u
source /opt/ros/jazzy/setup.bash
set -u
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/python -m colcon build --symlink-install
set +u
source install/setup.bash
set -u
.venv/bin/python -m colcon test --event-handlers console_direct+
.venv/bin/python -m colcon test-result --verbose

if [[ "${1:-}" == "--with-model" ]]; then
  ./scripts/prepare_sample_media.sh
  RUN_MODEL_SMOKE=1 .venv/bin/pytest -q tests/test_smoke_inference.py
fi
