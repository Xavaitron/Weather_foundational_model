#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=backend:native
stage2_config="${1:-configs/analysis.json}"
stage2_name="$(basename "$stage2_config" .json)"
run_stage() {
  local config="$1" method="$2" seed="$3" output="$4"
  shift 4
  if [[ -f "$output/last.pt" ]]; then
    python -m airadv.train --config "$config" --method "$method" --seed "$seed" --output "$output" --resume "$output/last.pt"
  else
    python -m airadv.train --config "$config" --method "$method" --seed "$seed" --output "$output" "$@"
  fi
}
for seed in 42 43 44; do
  for method in baseline nowcast_evolution adr_latent debezenac_gaussian; do
    pre="runs/eac4_${method}_seed${seed}"
    run="runs/${stage2_name}_${method}_seed${seed}"
    run_stage configs/eac4.json "$method" "$seed" "$pre"
    run_stage "$stage2_config" "$method" "$seed" "$run" --initial-state "$pre/best.pt"
    python -m airadv.evaluate --checkpoint "$run/best.pt" --output "$run/evaluation.json"
  done
done
