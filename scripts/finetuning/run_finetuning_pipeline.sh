#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${DATA_DIR:-data/era5_graphcast_small_1deg}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GRAPHCAST_GPU:-6}}"
EPOCHS="${EPOCHS:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
TARGET_STEPS="${TARGET_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SEED="${SEED:-0}"
EVAL_INITIALIZATIONS="${EVAL_INITIALIZATIONS:-4}"
FINETUNING_OUTPUT_DIR="${FINETUNING_OUTPUT_DIR:-outputs/finetuning}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN_DIR="$FINETUNING_OUTPUT_DIR/run_$RUN_ID"

export PYTHON_BIN ERA5_DATA_DIR CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

on_exit() {
  status=$?
  if (( status != 0 )); then
    printf 'Pipeline failed (exit %s). Log: %s\n' "$status" "$LOG_FILE" >&2
  fi
}
trap on_exit EXIT

printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Data directory: %s\n' "$ERA5_DATA_DIR"
printf 'Visible GPU: %s\n' "$CUDA_VISIBLE_DEVICES"

bash scripts/finetuning/setup_finetuning.sh --skip-data

stores_complete=true
for store in train validation test climatology; do
  if [[ ! -f "$ERA5_DATA_DIR/$store.zarr/.zmetadata" ]]; then
    stores_complete=false
  fi
done

if [[ "$stores_complete" == true ]]; then
  printf 'All ERA5 split and climatology stores already exist; skipping staging.\n'
else
  shopt -s nullglob dotglob
  existing_items=("$ERA5_DATA_DIR"/*)
  shopt -u nullglob dotglob
  if [[ -e "$ERA5_DATA_DIR" ]] && (( ${#existing_items[@]} > 0 )); then
    printf 'ERROR: Dataset directory is nonempty but incomplete: %s\n' "$ERA5_DATA_DIR" >&2
    printf 'Inspect or move the partial stores before rerunning; nothing was deleted.\n' >&2
    exit 2
  fi
  "$PYTHON_BIN" scripts/finetuning/prepare_era5.py --estimate-only
  "$PYTHON_BIN" scripts/finetuning/prepare_era5.py --yes
fi

printf 'Running one-step GPU smoke test.\n'
"$PYTHON_BIN" scripts/finetuning/finetune_graphcast.py \
  --data-dir "$ERA5_DATA_DIR" \
  --output-dir "$RUN_DIR/smoke_test" \
  --epochs 1 \
  --steps-per-epoch 1 \
  --batch-size 1 \
  --target-steps 1 \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED"

printf 'Starting fine-tuning.\n'
"$PYTHON_BIN" scripts/finetuning/finetune_graphcast.py \
  --data-dir "$ERA5_DATA_DIR" \
  --output-dir "$RUN_DIR/model" \
  --epochs "$EPOCHS" \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --batch-size "$BATCH_SIZE" \
  --target-steps "$TARGET_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED"

CHECKPOINT="$RUN_DIR/model/graphcast_small_finetuned.npz"
for split in validation test; do
  printf 'Evaluating %s split.\n' "$split"
  "$PYTHON_BIN" scripts/finetuning/evaluate_graphcast.py \
    --split "$split" \
    --data-dir "$ERA5_DATA_DIR" \
    --finetuned-checkpoint "$CHECKPOINT" \
    --num-initializations "$EVAL_INITIALIZATIONS" \
    --seed "$SEED" \
    --output "$RUN_DIR/${split}_benchmark.json"
done

printf 'Pipeline complete. Results and log: %s\n' "$RUN_DIR"