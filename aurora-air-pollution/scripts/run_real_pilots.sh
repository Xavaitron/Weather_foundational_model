#!/usr/bin/env bash
# Run only after run_data_pilots.sh has validated both source products.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Select the allocated GPU explicitly}"
py="${AIRADV_PYTHON:-python}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=backend:native
mkdir -p validation/server
trap 'printf "FAILED at line %s; inspect method log\n" "$LINENO" > validation/server/real_trials.status' ERR
printf 'Waiting for validated pilot data; GPU memory not allocated\n' > validation/server/real_trials.status
while [[ ! -f validation/server/data_pilots.json ]]; do
  if [[ -f validation/server/pilot_pipeline.status ]] && grep -q '^FAILED' validation/server/pilot_pipeline.status; then
    printf 'STOPPED: data pilot failed; inspect pilot_pipeline.log\n' > validation/server/real_trials.status
    exit 1
  fi
  sleep 30
done
for method in baseline nowcast_evolution adr_latent debezenac_gaussian wind_pull column_push; do
  printf 'Running %s on real EAC4 pilot data\n' "$method" > validation/server/real_trials.status
  "$py" -u scripts/gpu_trial.py --config validation/server/pilot_eac4_config.json --method "$method" --output "validation/server/real_${method}.json" > "validation/server/real_${method}.log" 2>&1
done
printf 'PASSED: two real-data optimizer steps per method; weights discarded; no skill estimate\n' > validation/server/real_trials.status
