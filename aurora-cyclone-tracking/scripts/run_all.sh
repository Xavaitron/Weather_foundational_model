#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task="${1:-cyclone}"
config="configs/${task}.json"
nproc="${NPROC_PER_NODE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Data preparation runs once before launching distributed training workers.
if [ "$task" = "cyclone" ]; then
  python scripts/download_datasets.py --source "${DATA_SOURCE:-hres}" --config "$config"
fi
for approach in standard paradis nowcastnet adrnet de_bezenac; do
  if [ "$nproc" -gt 1 ]; then
    python -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
      "scripts/finetune_${approach}.py" --config "$config"
  else
    python "scripts/finetune_${approach}.py" --config "$config"
  fi
done
