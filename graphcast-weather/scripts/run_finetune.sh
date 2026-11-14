#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GRAPHCAST_GPU:?Set GRAPHCAST_GPU to an available assigned GPU index}"
GPU_UUID="$(nvidia-smi -i "$GRAPHCAST_GPU" --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${PYTHON_BIN:-.venv/bin/python}" -u -m finetuning.train "$@"
