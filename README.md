# GraphCast Workflows

Run these commands on a Linux GPU server from the repository root, using an existing Python 3.11 or 3.12 environment. Select a GPU assigned to you. For inference, use its `nvidia-smi` index with `GRAPHCAST_GPU`. For direct fine-tuning and evaluation, resolve that index to its GPU UUID because CUDA device ordinals can differ from `nvidia-smi` indices.

## Inference

This uses the official pretrained GraphCast model at 0.25-degree resolution with 37 pressure levels and the published January 2022 sample. It is a smoke test for the inference setup, not the GraphCast_small fine-tuning baseline.

```bash
bash scripts/inference/setup_inference.sh
GRAPHCAST_GPU=<nvidia_smi_gpu_index> bash scripts/inference/run_inference.sh --check-device
GRAPHCAST_GPU=<nvidia_smi_gpu_index> bash scripts/inference/run_inference.sh --steps 1
```

Each step is a six-hour forecast and `--steps` accepts 1 to 12. Downloads are cached in `data/graphcast/`; predictions, plots, and the run manifest are saved under `outputs/inference/`.

The 0.25-degree / 37-level model requires substantially more GPU memory than GraphCast_small. Confirm available memory before running it.

## Fine-tuning

Fine-tuning uses the official **GraphCast_small** checkpoint at 1-degree resolution and 13 pressure levels. ERA5 is split into train (2016-2019), validation (2020), and test (2021-2022). The pretrained checkpoint ends in 2015.

Set the staged dataset location and install the dependencies once:

```bash
export PYTHON_BIN="$(command -v python)"
export ERA5_DATA_DIR="/path/to/era5_graphcast_small_1deg"
bash scripts/finetuning/setup_finetuning.sh --skip-data
```

If the four Zarr stores do not already exist, inspect the required space and stage them. Do not rerun staging into a nonempty partial directory.

```bash
"$PYTHON_BIN" scripts/finetuning/prepare_era5.py --estimate-only
"$PYTHON_BIN" scripts/finetuning/prepare_era5.py --yes
```

Set `GPU_ID` to the GPU index displayed by `nvidia-smi`, then resolve its UUID before running direct fine-tuning or evaluation commands:

```bash
GPU_ID=<nvidia_smi_gpu_index>
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$GPU_ID" --query-gpu=uuid --format=csv,noheader)"
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