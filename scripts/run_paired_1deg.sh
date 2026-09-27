#!/usr/bin/env bash
# Matched initial fine-tuning stage; real sampled windows, no repeated pilot file.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ROOT="${RUN_ROOT:-runs/full-1deg-stage1-20260927}"
BASELINE_GPU="${BASELINE_GPU:-2}"
ADVECTION_GPU="${ADVECTION_GPU:-6}"
if [[ -e "$RUN_ROOT" ]]; then
  echo "Choose a new RUN_ROOT; refusing to overwrite $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/code-revision.txt"
for variant in baseline advection; do
  gpu="$BASELINE_GPU"
  if [[ "$variant" == advection ]]; then gpu="$ADVECTION_GPU"; fi
  GRAPHCAST_GPU="$gpu" GRAPHCAST_GEOMETRY_CACHE=work/geometry-cache \
  XLA_FLAGS=--xla_gpu_autotune_level=0 \
  nohup bash scripts/run_finetune.sh \
    --checkpoint 'data/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz' \
    --stats-dir data/graphcast/stats --resolution 1.0 --variant "$variant" \
    --updates 1000 --steps 1 --learning-rate 1e-5 --seed 0 \
    --window-cache work/era5-1deg-window-cache --cache-max-gib 32 \
    --validate-initial --validation-count 4 --validate-every 50 --save-every 25 \
    --output "$RUN_ROOT/$variant" > "$RUN_ROOT/$variant.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN_ROOT/$variant.pid"
  echo "Started $variant on GPU $gpu; PID $!; log $RUN_ROOT/$variant.log"
done
