#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Use Python 3.11 or 3.12"'
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements-inference.txt
# Original GraphCast API, before the WeatherNext reorganization.
# Install the inference dependencies above; omit upstream Colab-only extras.
"$PYTHON_BIN" -m pip install --no-deps \
  'graphcast @ git+https://github.com/google-deepmind/weathernext.git@97d1ad50b0b7af4aaed7790167dffa769bae1f2c'
"$PYTHON_BIN" -c 'from graphcast import graphcast, rollout, casting, normalization; print("GraphCast imports OK")'
printf '\nSetup complete. Next: bash scripts/run_inference.sh --check-device\n'
