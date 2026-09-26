#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-inference.txt
# Original GraphCast API, before the WeatherNext reorganization.
# Install the inference dependencies above; omit upstream Colab-only extras.
.venv/bin/python -m pip install --no-deps \
  'graphcast @ git+https://github.com/google-deepmind/weathernext.git@97d1ad50b0b7af4aaed7790167dffa769bae1f2c'
.venv/bin/python -c 'from graphcast import graphcast, rollout, casting, normalization; print("GraphCast imports OK")'
printf '\nSetup complete. Next: bash scripts/run_inference.sh --check-device\n'
