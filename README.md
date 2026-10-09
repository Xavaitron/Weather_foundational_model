# GraphCast inference and fine-tuning

This repository includes pretrained GraphCast inference and matched **1° fine-tuning experiments with and without a learned latent advection adapter**, using the normal GraphCast model with 37 pressure levels. Both variants completed 1,000 updates with ERA5 training windows from 2016–2019, validation in 2020, and a sampled test benchmark in 2021–2022. The active experiment uses 1°, superseding the initial 0.1° target.

- [Fine-tuning setup and commands](docs/finetuning.md)
- [Completed training run](docs/training-stage1.md)
- [RMSE and ACC results](docs/evaluation-results.md) and [evaluation methodology](docs/evaluation-methodology.md)
- [Displacement before/after training and direct fine-tuning comparison](docs/advection-diagnostics.md)

Direct fine-tuning improves substantially over the pretrained model evaluated at 1°. The current six-hour benchmark does not establish a meaningful additional benefit from the adapter. The adapter is inspired by PARADIS; this is not a reproduction of its complete architecture. Small result tables and figures are included under `docs/results/`; ERA5 data, downloaded weights, trained checkpoints, caches, and optimizer state remain outside Git.

Run these commands on a Linux GPU server from the repository root, using an existing Python 3.11 or 3.12 environment. Select a GPU assigned to you. For inference, use its `nvidia-smi` index with `GRAPHCAST_GPU`. For direct fine-tuning and evaluation, set `CUDA_VISIBLE_DEVICES` to the GPU UUID reported by `nvidia-smi -L`; CUDA device ordinals can differ from `nvidia-smi` indices.

## Inference

This uses the official pretrained GraphCast model at 0.25-degree resolution with 37 pressure levels and the published January 2022 sample. It is a smoke test for the inference setup, not the GraphCast_small fine-tuning baseline.

```bash
bash scripts/inference/setup_inference.sh
GRAPHCAST_GPU=<nvidia_smi_gpu_index> bash scripts/inference/run_inference.sh --check-device
GRAPHCAST_GPU=<nvidia_smi_gpu_index> bash scripts/inference/run_inference.sh --steps 1 --cache-dir data/graphcast
```

The final-lead diagnostics are 2 m temperature, mean sea-level pressure, 850 hPa temperature and 500 hPa geopotential, compared with ERA5 using cosine-latitude-weighted RMSE. The January 2022 sample falls within the planned test years; do not use it to select settings or claim benchmark skill.

Each step is a six-hour forecast and `--steps` accepts 1 to 12 (up to 72 hours). Downloads are cached in `data/graphcast/`; predictions, plots, and the run manifest are saved under `outputs/inference/`.

The 0.25-degree / 37-level model requires substantially more GPU memory than GraphCast_small. Confirm available memory before running it.

## Fine-tuning

### Full GraphCast: baseline and learned advection

The completed experiment uses the normal GraphCast checkpoint, its full mesh and 37 pressure levels, adapted to a 1-degree input/output grid. Baseline and advection variants use the same 2016–2019 training, 2020 validation, and 2021–2022 test split. The original checkpoint was pretrained through 2017, overlapping the first two fine-tuning years.

Use the [full-model reproduction guide](docs/finetuning.md) for environment setup, individual training commands, adapter-aware checkpoint loading, and paired evaluation. Its older setup command `scripts/setup_inference.sh` now resides at `scripts/inference/setup_inference.sh`. Keep this workflow's `requirements-finetuning.txt` environment separate from the small-model `requirements-training.txt` environment because their dependency versions differ.

Both variants completed 1,000 updates. The recorded evaluation covers 36 monthly initializations at a six-hour lead. Direct fine-tuning improves substantially over the pretrained model evaluated at 1 degree; an additional advection benefit is not established. See the linked training, evaluation and displacement reports above.

### GraphCast_small: staged-data workflow


Fine-tuning uses the official **GraphCast_small** checkpoint at 1-degree resolution and 13 pressure levels. ERA5 is split into train (2016-2019), validation (2020), and test (2021-2022). The pretrained checkpoint ends in 2015.

Set the staged dataset location and install the dependencies once. The staged dataset is stored at `/media/data_dump/Anwar/era5_graphcast_small_1deg`; it must contain `train.zarr`, `validation.zarr`, `test.zarr`, and `climatology.zarr`.

```bash
export PYTHON_BIN="$(command -v python)"
export ERA5_DATA_DIR="/media/data_dump/Anwar/era5_graphcast_small_1deg"
bash scripts/finetuning/setup_finetuning.sh --skip-data
```

If the four Zarr stores do not already exist, inspect the required space and stage them. Do not rerun staging into a nonempty partial directory.

```bash
"$PYTHON_BIN" scripts/finetuning/prepare_era5.py --estimate-only
"$PYTHON_BIN" scripts/finetuning/prepare_era5.py --yes
```

Copy the UUID for your assigned GPU from `nvidia-smi -L`, then set it before running direct fine-tuning or evaluation commands:

```bash
export CUDA_VISIBLE_DEVICES=GPU-<uuid_from_nvidia_smi>
```

Start with a one-step smoke run. `target-steps=1` and `batch-size=1` are the lowest-memory training configuration. A successful run writes `training.json` and `graphcast_small_finetuned.npz`.

```bash
"$PYTHON_BIN" scripts/finetuning/finetune_graphcast.py \
  --data-dir "$ERA5_DATA_DIR" \
  --output-dir outputs/finetuning/smoke_test \
  --epochs 1 --steps-per-epoch 1 --batch-size 1 --target-steps 1
```

For a longer run, use a new output directory. Increase `target-steps` from 1 to 2 only after confirming the smoke run has memory headroom. If JAX reports an out-of-memory error, keep `batch-size=1` and `target-steps=1`, or use a GPU with more memory.

```bash
"$PYTHON_BIN" scripts/finetuning/finetune_graphcast.py \
  --data-dir "$ERA5_DATA_DIR" \
  --output-dir outputs/finetuning/run_1 \
  --epochs 1 --steps-per-epoch 1000 --batch-size 1 --target-steps 1
```

Evaluate validation first, then test after you have fixed the training settings. Replace the checkpoint path if you use a different output directory.

```bash
CHECKPOINT=outputs/finetuning/run_1/graphcast_small_finetuned.npz
"$PYTHON_BIN" scripts/finetuning/evaluate_graphcast.py \
  --split validation --data-dir "$ERA5_DATA_DIR" \
  --finetuned-checkpoint "$CHECKPOINT" --num-initializations 4 \
  --output outputs/finetuning/validation.json
"$PYTHON_BIN" scripts/finetuning/evaluate_graphcast.py \
  --split test --data-dir "$ERA5_DATA_DIR" \
  --finetuned-checkpoint "$CHECKPOINT" --num-initializations 4 \
  --output outputs/finetuning/test.json
```

Evaluation saves area-weighted RMSE and anomaly correlation coefficient for 69 non-precipitation targets at 12-hour leads through 240 hours.
