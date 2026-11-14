# Aurora air pollution

This workflow extends Aurora to pollutant forecasting. It covers the completed January 2003 baseline, the six model variants, and the full historical EAC4-to-CAMS protocol.

## Architecture and transport

Horizontal wind moves pollutant plumes, while emissions, chemistry, deposition and vertical mixing can change their concentrations. The model starts from the released **12-hour Aurora weather checkpoint**. It adds pollutant input and output channels and predicts the next weather and pollutant state from two 12-hourly history frames. The project uses a 120 × 240 global 1.5° grid with 12 surface channels and 10 atmospheric variables at 13 levels. Of the 142 forecast channels, 73 are pollutant channels.

The additional branches correct the decoded pollutant forecast. A scale fitted on training data normalizes the recent pollutant history for the correction branch, then restores the correction to physical units before addition.

~~~text
pollutant_forecast = Aurora(history).pollutants
                   + pollutant_scale * correction(normalized_history, wind_displacement)
~~~

| Variant | Operator |
|---|---|
| `baseline` | Full fine-tuning of the expanded 12-hour weather checkpoint |
| `wind_pull` | Fixed backward transport using estimated displacement from current winds |
| `column_push` | Area-weighted forward scatter for the five total columns |
| `nowcast_evolution` | U-Net estimates a displacement residual and an additive source, with a small initial transport gate |
| `adr_latent` | Three reaction, bounded explicit diffusion, and pull-advection blocks in a 32-channel representation |
| `debezenac_gaussian` | U-Net estimates motion residuals for a normalized 6 × 6 Gaussian sampler with learned width |

The wind estimate uses 10 m wind for surface particulate matter, wind at the matching level for pressure-level species, and a pressure-thickness weighted proxy for total columns. These trajectories approximate horizontal motion with wind held fixed for 12 hours. The conservative property applies to the isolated areal-density push for total columns. Concentrations and mixing ratios require a different mass budget. See [operator definitions and source papers](docs/METHODS.md).

The loss is the mean over channels of area-weighted MAE divided by each channel's training-period persistence MAE. This normalizes variables with different units and magnitudes. Report physical RMSE separately for each pollutant.

## Reproduce the January pilot and full comparison

Use a separate Linux Python 3.12 environment. Install the appropriate PyTorch 2.8.0 build, configure ADS access on the data host, and choose an assigned GPU by UUID. The first block installs the package, runs tests and downloads the pinned static and weather checkpoint assets.

~~~bash
cd aurora-air-pollution
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,download]'
python -m pytest -q
HF_HUB_DISABLE_XET=1 python scripts/fetch_assets.py
python scripts/run_training_pipeline.py --plan
~~~

With ADS credentials configured, prepare the January EAC4 store, fit training-only statistics, and check the split before launching. The download end date is exclusive. The result uses 1–20 January for training, 21–24 January for validation, and 25–31 January for testing.

~~~bash
python scripts/build_dataset.py --product eac4 --start 2003-01-01 \
  --end 2003-02-01 --raw data/raw_grouped --days-per-request 31 \
  --output data/processed/eac4_jan2003_1p5.zarr --execute
python scripts/stats.py --data data/processed/eac4_jan2003_1p5.zarr \
  --start 2003-01-01 --end 2003-01-21 \
  --output data/processed/eac4_jan2003_stats.json
python scripts/preflight.py --config configs/starter_eac4_jan2003.json

export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader)"
export PYTORCH_CUDA_ALLOC_CONF=backend:native
python -m airadv.train --config configs/starter_eac4_jan2003.json
python -m airadv.evaluate \
  --checkpoint runs/starter_eac4_jan2003_baseline_seed42/best.pt \
  --output runs/starter_eac4_jan2003_baseline_seed42/evaluation.json \
  --steps 10 --stride 2
~~~

The full comparison stages EAC4 first and CAMS forecast lead zero second. Prepare each store and the matching training-only statistics before starting the matrix. The matrix trains the baseline and three learned variants over seeds 42, 43, and 44. The fixed-wind controls can be selected with explicit `--method` flags. Use the CAMS lead-zero config explicitly to match the intended data product.

~~~bash
python scripts/build_dataset.py --product eac4 --start 2003-01-01 \
  --end 2022-01-01 --raw data/raw_grouped --days-per-request 31 \
  --output data/processed/eac4_1p5.zarr --execute
python scripts/build_dataset.py --product cams_forecast_t0 --start 2017-10-01 \
  --end 2022-12-01 --raw data/raw_grouped --days-per-request 31 \
  --output data/processed/cams_t0_1p5.zarr --execute
python scripts/stats.py --data data/processed/eac4_1p5.zarr \
  --start 2003-01-01 --end 2021-01-01 --output data/processed/eac4_stats.json
python scripts/stats.py --data data/processed/cams_t0_1p5.zarr \
  --start 2017-10-01 --end 2022-04-01 --output data/processed/cams_t0_stats.json
python scripts/preflight.py --config configs/eac4.json
python scripts/preflight.py --config configs/cams_t0.json
bash scripts/run_matrix.sh configs/cams_t0.json
python scripts/compare.py runs/cams_t0_*/evaluation.json --output runs/comparison.csv
~~~

The official **0.4° air-pollution checkpoint** is a distinct initialization supported by the model builder. Its use is guarded because the planned historical CAMS validation overlaps its training exposure. The January results below use the 12-hour weather checkpoint. See the [GPU guide](docs/GPU_UPLOAD.md) for ADS setup, historical retrieval, preflight checks, GPU trials and resume commands.

## Results

See the [results-only README](results/README.md) for this evidence without the run steps.

### Training and evaluation record

| Item | January 2003 EAC4 pilot |
|---|---|
| Initialization | Official 12-hour meteorological Aurora, extended with pollutant channels |
| Parameters updated | 1,256,857,440 |
| Grid and interval | Global 120 × 240 at 1.5°, 12-hour steps |
| Train | 1–20 January, 38 valid windows |
| Validation | 21–24 January, 6 valid windows |
| Test | 25–31 January, two forecast origins with complete 120-hour rollouts |
| Optimization | Seed 42, 200 updates, four microbatches per update, one-step supervision |
| Checkpoint selection | Best validation loss at update 200 |

| Optimizer step | 50 | 100 | 150 | 200 |
|---|---:|---:|---:|---:|
| Normalized validation objective | 2.8825 | 2.1317 | 1.9421 | 1.8197 |

The normalized objective averages across the 142 weather and pollution channels. A lower value during fitting does not directly translate to lower PM2.5 RMSE. Forecast origins were 25 and 26 January 2003 at 12:00 UTC. Both forecasts run from +12 h to +120 h in twelve-hour increments and overlap in time.

For each lead, area-weighted MAE averages absolute error, RMSE is the square root of mean squared error, and signed bias averages forecast minus reference. The negative fraction divides the weighted area below zero by the total scored area. Fixed persistence repeats the latest field, and diurnal persistence repeats the matching time-of-day field. The India-region scores use a bounding box, including grid cells outside national borders. The reference is EAC4 reanalysis.

### Held-out PM2.5 results

RMSE is in µg/m³. Negative area is the area-weighted fraction of raw forecasts below zero.

| Region | Lead | Aurora RMSE | Fixed persistence | Diurnal persistence | Negative forecast area |
|---|---:|---:|---:|---:|---:|
| Global | 12 h | 24.251 | 17.207 | 18.828 | 15.52% |
| Global | 24 h | 24.595 | 19.555 | 19.555 | 17.96% |
| Global | 72 h | 26.026 | 24.599 | 24.599 | 20.20% |
| Global | 120 h | 20.888 | 22.530 | 22.530 | 19.74% |
| India | 12 h | 50.785 | 38.981 | 19.199 | 5.07% |
| India | 24 h | 30.950 | 13.293 | 13.293 | 8.72% |
| India | 72 h | 24.710 | 20.193 | 20.193 | 8.21% |
| India | 120 h | 25.933 | 21.321 | 21.321 | 7.98% |

India PM2.5 RMSE exceeds both persistence controls at every reported lead. Global PM2.5 beats both controls at +120 h on these two cases. The sample is too small to estimate seasonal skill.

### Implementation checks and missing scores

All six variants completed two real EAC4 optimizer updates with finite gradients. Trial weights were discarded. The real-data baseline peak allocation was about 23.50 GiB. The recorded 23.53–23.54 GiB range for full-size synthetic trials covers checkpoint loading, forward/backward and AdamW allocation. These are memory and integration measurements.

The 9 November 2026 snapshot records 62 completed January frames and 7,368 historical EAC4 frames through 31 January 2013. The historical worker stopped on the February 2013 ADS request after a TLS certificate verification failure. CAMS forecast lead zero had two pilot frames. The completed evaluation's full per-pollutant, all-lead JSON was on the server and is absent from the tracked evidence. The table above includes all published PM2.5 rows in this checkout. There are no recorded held-out comparison scores for the learned advection variants or the official 0.4° air-pollution checkpoint. See the [dated evidence](docs/RESULTS_2026-11-09.txt) and [validation record](validation/VALIDATION.md).

## Directory layout

- `src/airadv/` contains model integration, numerical transport and evaluation.
- `scripts/` contains ADS retrieval, preparation, training and comparison utilities.
- `configs/` contains the January starter, full EAC4 and CAMS protocols.
- `docs/` and `validation/` contain recorded experiment evidence.
