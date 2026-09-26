#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Resolve the nvidia-smi index to a UUID to avoid CUDA enumeration ambiguity.
GRAPHCAST_GPU="${GRAPHCAST_GPU:-6}"
GPU_UUID="$(nvidia-smi -i "$GRAPHCAST_GPU" --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export MPLBACKEND=Agg
exec .venv/bin/python -u scripts/infer_graphcast.py "$@"
