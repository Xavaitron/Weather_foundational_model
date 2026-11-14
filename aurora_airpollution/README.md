# Aurora air-pollution advection experiments

Air-pollution fine-tuning of Aurora at **1.5 degrees**, with baseline and paper-inspired advection branches. Install and run commands below from this directory, in a separate environment from the repository's GraphCast workflows.

**Last verified results: 9 November 2026.** The January 2003 baseline completed 200 updates and held-out evaluation. Only two forecast origins were evaluated; PM2.5 often trails persistence, and negative predictions remain a limitation. No held-out advection improvement has been demonstrated. The full historical data pipeline stopped on 4 November following an ADS TLS certificate error; the EAC4 store reached January 2013, and the full CAMS stage is incomplete. This is a dated status, not a live server check.

All six variants previously passed two real EAC4 optimizer steps. These are integration checks, not comparative training results. See [results and limitations](docs/RESULTS_2026-11-09.txt), [training history](docs/TRAINING_RUN.md), and [GPU setup](docs/GPU_UPLOAD.md). Datasets, model weights, credentials, and operational logs are excluded.

## Implemented experiments

| Method flag | Model trained |
|---|---|
| `baseline` | Standard full fine-tuning of Aurora, extended with pollutants |
| `nowcast_evolution` | Aurora plus U-Net motion and source evolution branch |
| `adr_latent` | Aurora plus latent reaction, diffusion and advection blocks |
| `debezenac_gaussian` | Aurora plus learned motion and a differentiable Gaussian warp |
| `wind_pull` | Aurora plus a fixed wind-driven pull increment |
| `column_push` | Aurora plus an area-conservative transport increment for five total columns |

The three learned branches are **paper-inspired adaptations**, with intentional architecture changes documented in [METHODS.md](docs/METHODS.md). Full NowcastNet GAN training and original-paper benchmark reproduction are not implemented. GraphCast and GenCast integrations are subsequent work.

## What is ready

- Global 120 × 240 cell-centre grid, 12-hour histories and forecast steps.
- All 12 surface variables and 10 atmospheric variables on 13 pressure levels; the adapters cover all 73 pollutant channels.
- CAMS analysis and EAC4 retrieval plans, conservative regridding, strict units/schema checks, train-only statistics, and chronological window construction.
- Full-parameter training, gradient accumulation, activation checkpointing, bfloat16 backbone execution, checkpoint/resume, and multi-step training option.
- Autoregressive evaluation at 12–120 hours with global and India-region MAE/RMSE/bias; fixed and diurnal persistence controls.
- A two-stage EAC4 → CAMS experiment matrix with three seeds and matching training budgets.

## Quick software check

Python 3.12 is the tested version. Install a suitable PyTorch 2.8.0 build first, then:

```bash
pip install -e '.[dev,download]'
python -m pytest -q
python scripts/smoke.py --full-grid
```

The smoke test uses the actual Aurora implementation with a tiny random configuration. It validates integration on the requested grid without downloading 5 GB weights. See [VALIDATION.md](validation/VALIDATION.md) for precisely what was checked.

## Main experiment decision

The main comparison starts from `Aurora12hPretrained`, which has a public checkpoint. New pollution embeddings and outputs are trained along with the backbone. All methods share the same starting weights, data, splits, loss, steps and seed. The released `AuroraAirPollution` is also supported, but it was already fine-tuned on historical CAMS data; its use is explicitly guarded because an April–May 2022 validation set would be pre-exposed.

The default protocol uses the same **dataset families** as Aurora's pollution work, with an explicit validation holdout and coarser grid. It is not an exact recreation of the published training schedule or split. [DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md) records every distinction.

## Files

`src/airadv/` contains transport, neural branches, Aurora integration, training and evaluation. `scripts/` contains data preparation and launch tools. `configs/` contains stage settings. `tests/` contains numerical and end-to-end integration checks. `validation/` contains the saved CPU verification record.

