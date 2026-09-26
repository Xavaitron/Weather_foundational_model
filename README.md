# GraphCast Inference

The experimental baseline/advection fine-tuning implementation is documented in [docs/finetuning.md](docs/finetuning.md). Its feasibility checks and pilot runs must not be confused with a completed 0.1° training experiment.

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

For a forecast, the model gets two weather states six hours apart and predicts the next state. The runner compares predictions with ERA5 references; future weather values are not passed to the model, though known time and radiation forcings are.

## What It Checks

`--check-device` runs a small JAX calculation on the selected GPU. It checks GPU access but does not load the model or weather data. `--steps 1` is the actual inference smoke test: it loads the checkpoint, compiles and runs GraphCast on the GPU. For longer runs, a NetCDF forecast is written at each lead; four plots and four area-weighted RMSE metrics are produced for the final requested lead. The output folder uses the forecast initialization time (latest input time) and requested number of steps, for example `graphcast_20220101_0000Z_1step` for a 00:00 initialization and one-step forecast to 06:00. Forecast files include their valid time and lead. The folder contains:

- `prediction_<valid-time>_lead_<hours>.nc`: forecast weather fields at that lead.
- `<field>_<valid-time>_lead_<hours>.png`: ERA5 reference, prediction, and difference for each selected field.
- `run.json`: model and data identifiers, initialization and valid times, device, run status, and four RMSE metrics (including units).

The `.nc` forecasts are ignored by Git because they are large. The four plots and `run.json` are not ignored and can be committed.

The selected diagnostics are 2-metre temperature and mean sea-level pressure at the surface, plus temperature at 850 hPa and geopotential at 500 hPa. Each RMSE is cosine-latitude-weighted and compares the final forecast lead with the corresponding ERA5 sample.

## Limits

- This does not train or fine-tune GraphCast.
- January 2022 is a smoke-test sample, not a benchmark. It falls within the planned 2021-2022 test period; do not use this sample's RMSE to tune the model or claim benchmark performance.
- Supplied samples support forecasts up to 72 hours. This does not test ten-day forecasting or the proposed 0.1-degree/advection changes.
