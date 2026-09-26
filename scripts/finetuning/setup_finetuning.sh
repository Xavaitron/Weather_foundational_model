#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Use Python 3.11 or 3.12"'

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements-training.txt
"$PYTHON_BIN" -m pip install --no-deps \
  'graphcast @ git+https://github.com/google-deepmind/weathernext.git@97d1ad50b0b7af4aaed7790167dffa769bae1f2c'
"$PYTHON_BIN" -c 'import gcsfs, jax, optax, zarr; from graphcast import graphcast, checkpoint; print(f"Training imports OK; JAX {jax.__version__}; {jax.devices()}")'

if [[ "${1:-}" == "--skip-data" ]]; then
  if [[ $# -ne 1 ]]; then
    printf 'ERROR: --skip-data cannot be combined with data preparation options.\n' >&2
    exit 2
  fi
  printf '\nDependencies ready. Dataset staging skipped.\n'
else
  OUTPUT_DATA_DIR="${ERA5_DATA_DIR:-${DATA_DIR:-data/era5_graphcast_small_1deg}}"
  "$PYTHON_BIN" scripts/finetuning/prepare_era5.py --output-dir "$OUTPUT_DATA_DIR" "$@"
fi