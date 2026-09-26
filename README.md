# GraphCast Inference

Run the original pretrained GraphCast model on an official ERA5 sample. This does not train or fine-tune the model. Run these Bash commands on the Linux GPU server, not in Windows PowerShell.

## Run

Use your existing Python 3.11 or 3.12 environment. Setup installs packages into it, so it may change existing package versions.

```bash
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" --version
bash scripts/setup_inference.sh
bash scripts/run_inference.sh --check-device
bash scripts/run_inference.sh --steps 1
```

The launcher defaults to GPU index 6. To select another GPU assigned to you, set `GRAPHCAST_GPU`, for example:

```bash
GRAPHCAST_GPU=2 bash scripts/run_inference.sh --steps 1
```

`--steps 1` forecasts six hours. Each step is six hours; `--steps 4` forecasts 24 hours and `--steps 12` forecasts 72 hours.

## What It Loads

- **Model:** pretrained 0.25-degree, 37-pressure-level GraphCast checkpoint, trained on ERA5 from 1979-2017. It downloads from DeepMind's public `dm_graphcast` bucket.
- **Normalization:** three statistics files matched to the checkpoint, from the same bucket.
- **Weather data:** DeepMind's prepared global ERA5 sample for January 1, 2022, at 0.25-degree resolution and 37 pressure levels. The runner downloads the smallest available sample that covers the requested forecast length. Downloads are cached in `data/` and checked by size and MD5.

For a one-step forecast, the model gets two weather states six hours apart and predicts the next state. The runner compares that prediction with the ERA5 reference; future weather values are not passed to the model, though known time and radiation forcings are. It reports 2-metre temperature RMSE.

## What It Checks

`--check-device` runs a small JAX calculation on the selected GPU. It checks GPU access but does not load the model or weather data. `--steps 1` is the actual inference smoke test: it loads the checkpoint, compiles and runs GraphCast on the GPU, then writes a timestamped folder under `outputs/` containing:

- `prediction_006h.nc`: forecast weather fields at +6 hours.
- `temperature.png`: ERA5 reference, predicted 2-metre temperature, and their difference.
- `run.json`: model and data identifiers, initialization time, device, run status, and RMSE.

The NetCDF forecast is ignored by Git because it is large. The plot and `run.json` are not ignored and can be committed.

## Limits

- This does not train or fine-tune GraphCast.
- January 2022 is a smoke-test sample, not a benchmark. It falls within the planned 2021-2022 test period; do not use this sample's RMSE to tune the model or claim benchmark performance.
- Supplied samples support forecasts up to 72 hours. This does not test ten-day forecasting or the proposed 0.1-degree/advection changes.
