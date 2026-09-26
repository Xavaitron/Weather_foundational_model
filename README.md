# GraphCast Workflows

Run commands on Deathstar's Linux GPU server, not Windows PowerShell. The scripts use the active Python 3.11/3.12 environment; no virtual environment is created, and package installation may change that environment. Select one assigned GPU with `CUDA_VISIBLE_DEVICES`.

## Fine-Tune GraphCast_small

This workflow uses the official 1-degree, 13-pressure-level GraphCast_small checkpoint. ERA5 is remapped from 0.25 degrees to 1 degree and split into 2016-2019 train, 2020 validation, and 2021-2022 test. Regridding reduces grid resolution; it does not add weather information.

### 1. Install and estimate

```bash
export PYTHON_BIN="$(command -v python)"
export ERA5_DATA_DIR="/path/to/writable/era5_graphcast_small_1deg"
bash scripts/finetuning/setup_finetuning.sh --skip-data
"$PYTHON_BIN" scripts/finetuning/prepare_era5.py --estimate-only
```

Set `ERA5_DATA_DIR` once in this shell (or add the export to `~/.bashrc`). Preparation, training, evaluation, and the setup wrapper use it by default. The existing `DATA_DIR` variable is also accepted. Explicit `--output-dir` or `--data-dir` arguments override the environment setting.

### 2. Stage ERA5

```bash
"$PYTHON_BIN" scripts/finetuning/prepare_era5.py --yes
```

This needs GCS access to `storage.googleapis.com`, about 256 GB (238 GiB) free for the conservative uncompressed estimate, and extra disk headroom. Actual Zarr size depends on compression.

### 3. Fine-tune

```bash
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" scripts/finetuning/finetune_graphcast.py --epochs 1 --steps-per-epoch 1 --target-steps 1
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" scripts/finetuning/finetune_graphcast.py --epochs 1 --steps-per-epoch 1000 --target-steps 2 --overwrite
```

Replace GPU `6` with the assigned GPU. The smoke test and full run use the same output directory, so the second command overwrites the smoke-test checkpoint.

### 4. Evaluate

Evaluate validation before making decisions; run test only after settings are fixed.

```bash
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" scripts/finetuning/evaluate_graphcast.py --split validation --num-initializations 4
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" scripts/finetuning/evaluate_graphcast.py --split test --num-initializations 4
```

Four evenly spaced starts are a pilot. Use `--num-initializations 0` to evaluate every eligible 12-hour start; this takes substantially longer.

## Metrics

Compare the fine-tuned checkpoint with frozen GraphCast_small on identical initializations. Report global area-weighted RMSE (lower is better) and WeatherBench 2 ACC (higher is better), for the 69 non-precipitation targets at 12-hour leads through 240 hours. ACC uses WB2's 1990-2017 ERA5 six-hour climatology. The overall summary is macro-mean percent RMSE improvement and mean ACC change across target/lead pairs; detailed per-variable and per-level scores are also saved because target units differ. Do not use test results to select settings.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/finetuning/setup_finetuning.sh` | Installs training dependencies and pinned GraphCast in the active Python environment. Pass `--skip-data` to install without staging ERA5. |
| `scripts/finetuning/prepare_era5.py` | Estimates disk requirements, selects required ERA5 fields, remaps them to 1 degree, writes the three split stores, and stages the WB2 climatology. |
| `scripts/finetuning/finetune_graphcast.py` | Loads GraphCast_small and its statistics, trains only on the train split with autoregressive gradient checkpointing, and saves a checkpoint plus `training.json`. |
| `scripts/finetuning/evaluate_graphcast.py` | Compares frozen and fine-tuned checkpoints on validation or test and writes detailed metrics to JSON. |
| `scripts/inference/setup_inference.sh` | Installs the pinned inference environment and GraphCast code. |
| `scripts/inference/run_inference.sh` | Selects one GPU (default index 6), configures JAX, and launches the inference runner. |
| `scripts/inference/infer_graphcast.py` | Runs the separate 0.25-degree, 37-level pretrained inference smoke test on a January 2022 sample; this is not benchmark evidence. |

Inference forecasts are written under `outputs/inference/`; fine-tuned checkpoints, training provenance, and benchmark metrics are written under `outputs/finetuning/`.

For inference only, run `bash scripts/inference/setup_inference.sh`, then `bash scripts/inference/run_inference.sh --check-device` and `bash scripts/inference/run_inference.sh --steps 1`. Each inference step is six hours; supplied samples cover up to 72 hours.
