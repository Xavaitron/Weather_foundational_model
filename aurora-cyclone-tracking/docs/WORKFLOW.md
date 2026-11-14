# Aurora cyclone tracking

Five independent training entry points share the same data, weather loss, rollout engine, checkpoint format and cyclone evaluator:

| Script | Experiment |
|---|---|
| `scripts/finetune_standard.py` | Standard Aurora fine-tuning |
| `scripts/finetune_paradis.py` | PARADIS-inspired spherical backward transport, learned mixing and reaction |
| `scripts/finetune_nowcastnet.py` | NowcastNet-inspired nearest transport, source update and a separately supervised bilinear path |
| `scripts/finetune_adrnet.py` | ADRNet-inspired forward push, implicit grid diffusion and reaction; optional backward-color variant |
| `scripts/finetune_de_bezenac.py` | de Bezenac-inspired Gaussian-kernel backward warping |

The default cyclone model is **original Aurora 0.25-degree Pretrained**, fine-tuned here on a **1.5-degree output grid**. Aurora 1.5 (the model version) is a different model interface and is not selected by these scripts. Full fine-tuning is the default for all five experiments. The adapter weights start from new initialization; the Aurora weights come from Microsoft's checkpoint.

These are implementations of the papers' transport ideas adapted to Aurora, not reproductions of the original authors' architectures/results. Read [methods](METHODS.md) for exact differences. No trained advection adapters, atmospheric datasets or IBTrACS downloads are included. Local tests use a tiny random instance of the actual Aurora architecture. GPU training, released-weight inference, DDP performance and forecast skill must be established on your server.

## 1. Copy and set up on the SSH GPU

Copy the supplied `aurora_cyclone_tracking.zip`, or copy the source directory while excluding the local `.venv`, caches, data and runs.

```bash
unzip aurora_cyclone_tracking.zip
cd aurora-cyclone-tracking
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Tested torch/torchvision pair. Select a wheel compatible with your GPU driver.
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python scripts/smoke_test.py

# Optional two-process CPU check, without a GPU or weight downloads:
python scripts/smoke_ddp.py
```

Python 3.10–3.12 and Git are required. The Aurora source is pinned to commit `7179f501b9f1a66e0444b782ecd06aa07062c8bc`, which was used to test the integration. Its checkpoint revision is controlled by the corresponding official model class. `pip-freeze.txt`, the source hash, initial checkpoint configuration, GPU information and training settings are saved with each run. Checkpoint downloads use Hugging Face's normal cache; for an offline machine, pre-stage the checkpoint and set `initial_checkpoint` in the JSON configuration to that file.

For multi-GPU runs each process holds the full model and optimizer (DDP, not FSDP). Start with a single 80 GB GPU if available. At 1.5 degrees the activations are much smaller than at native resolution, but optimizer/model storage remains substantial. Reduce `batch_size`, `rollout_steps`, or use the same `trainable="decoder"` configuration across all variants if memory is insufficient. `model="small"` is available for debugging, but is a different checkpoint and not the final large-model experiment.

## 2. Automatic cyclone data download

```bash
# Downloads/prepares data automatically, then trains all five approaches:
NPROC_PER_NODE=4 bash scripts/run_all.sh cyclone

# Or prepare the data separately before running individual training scripts:
python scripts/download_datasets.py

# Optional metadata-only connectivity/schema/date check:
python scripts/download_datasets.py --check-only
```

The downloader anonymously reads public WeatherBench2 HRES-T0 surface/pressure-level
fields, downloads ERA5 land mask, surface geopotential and soil type from ARCO,
and downloads/normalizes NOAA IBTrACS v04r01. No CDS login or cloud credentials
are needed. Weather data is conservatively regridded from the WeatherBench
121x240 grid to the pipeline's cell-centred 120x240 grid; winds use Cartesian
vector regridding and soil type uses nearest-neighbour interpolation.

Completed monthly shards are reused on rerun. Partial months are not accepted.
The final local `data/cyclone_1p5.zarr` is assembled only after all months pass
schema, six-hour cadence and finite-value checks. The monthly cache is retained
at `data/cyclone_1p5_download_cache`; allow space for both cache and final store
(roughly 160 GB uncompressed combined for seven years; actual compressed size varies).
Preparation is CPU/network/disk work; GPUs are used after it finishes.

The default split is train 2016–2020, validation 2021, test 2022, matching public
HRES-T0 availability. This is a coarse-grid research experiment, not a full
reproduction of Aurora's published cyclone evaluation. Existing experiments and
checkpoints using the previous split must retain their saved configuration.
Use a distinct config/store path for ERA5-only experiments and run
`python scripts/download_datasets.py --source era5 --config YOUR_CONFIG.json` first.
For the runner to use ERA5, set DATA_SOURCE=era5 and edit configs/cyclone.json to point to that ERA5 store. The downloader refuses to mix sources or date ranges in an existing managed store.

This automatic downloader covers **cyclone data only**. Air pollution still
requires CAMS pollutant fields and emission inputs; WeatherBench ERA5/HRES cannot
supply those. `scripts/run_all.sh air_pollution` retains the manual preparation workflow.

### Manual sources and preparation

To match the **data source** in Aurora's cyclone experiment, supply **IFS HRES-T0** atmospheric/surface fields and the ERA5 static fields. IBTrACS supplies observed storm positions, not atmospheric training targets. HRES-T0 archive access is separate from ERA5. ERA5/ARCO can also be used, but label that experiment as ERA5 fine-tuning rather than a reproduction of the HRES-T0 experiment.

The default cyclone configuration has:

- Training: 2016–2020.
- Validation: 2021.
- Test: 2022.

The public WeatherBench HRES-T0 archive ends in 2022. This reserves 2021 for model selection and tests on **2022**. It does not reproduce that paper's complete test population. Audit inherited training exposure before changing dates. All inputs and targets of a training/validation window must remain in their split, and windows spanning missing time steps are rejected.

Required interface:

| Kind | Variables |
|---|---|
| Surface | `2t`, `10u`, `10v`, `msl` |
| Atmospheric | `z`, `u`, `v`, `t`, `q` |
| Pressure levels in hPa | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |
| Static | `lsm`, surface `z`, `slt` |

Use Aurora/ECMWF physical units: temperature K, pressure Pa, wind m/s, geopotential m²/s² (not geopotential height), specific humidity kg/kg. The preparer does not guess conversions or fill missing values. Fields below terrain must follow the source product's documented extrapolation policy. All variables need the same valid timestamps. Select a single member/experiment before preparation.

```bash
python scripts/prepare_data.py \
  --task cyclone \
  --source /data/hres_pressure_levels.zarr \
  --source /data/hres_surface.zarr \
  --source /data/era5_static.nc \
  --start 2016-01-01 --end 2023-12-31 \
  --data-vintage "IFS HRES-T0 2016-2023; ERA5 static fields" \
  --output data/cyclone_1p5.zarr
```

Both `.nc` and consolidated Zarr inputs are supported. Repeat `--source` for additional stores. Common ERA5 names are recognized; use `--mapping` for other names, units or ambiguous fields. For example, a JSON mapping may contain:

```json
{
  "atmos__z": {"name": "geopotential", "source_index": 0},
  "surf__msl": {"name": "msl", "source_index": 1},
  "static__z": {"name": "z", "source_index": 2},
  "static__slt": {"name": "slt", "source_index": 2}
}
```

`source_index` is zero-based in command-line order. A mapping can additionally specify `scale` and `offset` for an explicit conversion. For global ARCO Zarr, `gs://...` URLs use anonymous GCS access. Select the requested date range before staging; these archives are very large. The historical WeatherBench `1959-2023_01_10...` example does not contain the full 2023 test year.

Regridding uses area-overlap weights for scalars, regrids u/v as a Cartesian tangent vector, and uses nearest interpolation for categorical soil type. The target is a **120 × 240 cell-centred grid**: latitude 89.25 to -89.25, longitude 0.75 to 359.25. This is shifted relative to WeatherBench's 121 × 240 pole-including grid. The symmetric cell-centred grid enables unambiguous polar transport padding. All five variants use exactly this grid. Their scores are not directly interchangeable with a different grid's scores.

Aurora retains its original patch size 4 for cyclone runs. Therefore the transport feature lattice is **30 × 60, nominally 6 degrees**, while the reconstructed forecast is 1.5 degrees. This is an initial coarse-resolution mechanism study; do not interpret it as transport at 1.5-degree latent spacing. The alternative, changing patch size/weights, needs a separately controlled experiment.

## 3. Prepare IBTrACS

Download the global CSV `ibtracs.ALL.list.v04r01.csv` from the [official NOAA v04r01 CSV directory](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/), then run:

```bash
python scripts/prepare_ibtracs.py \
  --input /data/ibtracs.ALL.list.v04r01.csv \
  --output data/ibtracs.csv
```

This keeps main/provisional tracks, checks duplicates and valid coordinates, and retains nature/basin labels. The evaluator uses `NATURE` TS and DS by default and requires at least 24 hours of subsequent track coverage. Observed future positions are used only for verification. Storm initialization positions are permitted tracker inputs.

## 4. Run each fine-tuning approach

Review `configs/cyclone.json` first. The five commands are:

```bash
python scripts/finetune_standard.py   --config configs/cyclone.json
python scripts/finetune_paradis.py    --config configs/cyclone.json
python scripts/finetune_nowcastnet.py --config configs/cyclone.json
python scripts/finetune_adrnet.py     --config configs/cyclone.json
python scripts/finetune_de_bezenac.py --config configs/cyclone.json
```

Run a short pilot with `--max-steps 10 --output runs/pilot_paradis` before committing to a long run. A production run with different settings needs a new output directory. To run all five sequentially:

```bash
bash scripts/run_all.sh cyclone

# Or use all four GPUs for each approach, sequentially:
NPROC_PER_NODE=4 bash scripts/run_all.sh cyclone

# One approach with torchrun:
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/finetune_paradis.py --config configs/cyclone.json
```

The default objective is Aurora-style variable-weighted normalized MAE, two six-hour rollout steps, and full-model optimization. The cyclone configuration supplies the paper's HRES fine-tuning weight adjustments in `loss_overrides`; remove those overrides when deliberately using the original ERA5/pretraining weighting. By default it does not add latitude weighting to the training loss; set `area_weighted_loss=true` consistently across all variants to make that a separate choice. Evaluation always uses geographic distance or area weighting as appropriate. New adapters get a separate learning rate. NowcastNet adds its necessary bilinear auxiliary forecast loss; the other transport variants add a weak motion penalty. These objective differences and training cost must be reported alongside the comparison.

The default run directories are `runs/cyclone/<approach>_seed42/`. Each contains:

- `config.json`, `manifest.json`, `pip-freeze.txt`.
- `train.jsonl`: training/validation loss, optimizer-step time and peak GPU memory.
- `latest.pt`, `best.pt`: complete model/optimizer states, RNG state and epoch offset.
- `training_complete.json` on successful completion.

`best.pt` is selected using validation **weather loss**, not test cyclone error. For a final study, use the same fixed validation sample budget for every variant, increase it beyond the default 32 batches if needed, and repeat configurations with several seeds. For example, add `--seed 43` to all five commands for a second matched comparison.

Resume with the exact configuration, code and GPU count:

```bash
python scripts/finetune_paradis.py \
  --config runs/cyclone/paradis_seed42/config.json \
  --output runs/cyclone/paradis_seed42 \
  --resume runs/cyclone/paradis_seed42/latest.pt
```

Only load checkpoints you created or trust. Optimizer/RNG restoration uses PyTorch's full checkpoint loader. Resume reproduces epoch order/RNG state, but CUDA scatter accumulation and hardware kernels can still be nondeterministic.

## 5. Evaluate cyclone tracks with matching metrics

Run the identical command for each trained approach:

```bash
for approach in standard paradis nowcastnet adrnet de_bezenac; do
  python scripts/evaluate_cyclones.py \
    --checkpoint "runs/cyclone/${approach}_seed42/best.pt" \
    --ibtracs data/ibtracs.csv \
    --leads 24 48 72 96 120
done
```

For the original paper's operational availability convention, add `--issue-delay-hours 6` to every evaluator. Reported 24/48/72/96/120 h leads then use **30/54/78/102/126 h from the atmospheric initialization**, also matching the raw lead times listed by Aurora 1.5. Both raw and reported lead times are saved. This does not reproduce all operational issue-time/tracker choices in either paper; it makes our timing explicit. Directly requesting `--leads 30 54 78 102 126` with delay zero is also supported and labels them as raw leads.

Outputs under each run's `cyclone_evaluation/`:

- `tracks.csv`: storm ID, basin, initialization/issue/valid times, raw/reported lead, predicted/observed positions, error in km and failure flag.
- `summary.csv`: mean haversine track error, 95% storm-bootstrap interval, cases, storms and failure rate by lead and basin.
- `evaluation.json`: protocol, IBTrACS file hash, model/data configuration and tracker version.

The tracker is an explicit **coarse-grid adaptation** of Aurora's MSL/Z700 minimum tracker with quadratic subgrid refinement. It extrapolates previous predictions for its search center; on failure it persists the previous predicted location, flags the failure and includes the resulting distance in scores. It never uses a future observed position to recenter a forecast. The search-box/local-minimum logic differs from the 0.25-degree paper because the grid is coarser. This evaluates tracks of already-known cyclones; it does not evaluate genesis or cyclone detection.

The summary uses the native IBTrACS basin codes and fixes each forecast's basin at initialization. Aurora 1.5's special South Indian/Australian regrouping is not applied. There is no ensemble generation or ensemble rank histogram in this deterministic experiment package.

Compare on exactly matched cases, retaining failures:

```bash
python scripts/compare_results.py \
  --reference runs/cyclone/standard_seed42/cyclone_evaluation \
  --candidate runs/cyclone/paradis_seed42/cyclone_evaluation \
  --candidate runs/cyclone/nowcastnet_seed42/cyclone_evaluation \
  --candidate runs/cyclone/adrnet_seed42/cyclone_evaluation \
  --candidate runs/cyclone/de_bezenac_seed42/cyclone_evaluation \
  --output runs/cyclone_comparison.csv
```

The comparison reports `100 × (candidate MAE − standard MAE) / standard MAE`; negative is improvement. It computes paired 95% confidence intervals using 1,000 storm resamplings stratified by basin. Missing cases or mismatched protocols raise an error rather than quietly restricting the sample. `--max-inits` on evaluators is for debugging only and must match across approaches.

## 6. Air pollution

The same five entry points accept `--config configs/air_pollution.json`. They start from Microsoft's **AuroraAirPollution** checkpoint and therefore perform **further fine-tuning** of the published air-quality model, rather than reproducing its full earlier adaptation from the foundation model.

Supply CAMS analysis/reanalysis fields and all prescribed static emissions/context fields. The required interface is defined explicitly in `advection_ft/schema.py` and matches the official air-pollution class. It includes PM1/PM2.5/PM10, total-column gases and pressure-level gases alongside meteorology. ERA5 alone is insufficient. Missing fields fail explicitly. Do not replace missing pollutant/emissions fields with zeros. Use `--mapping` to specify names and exact unit conversions (for example, µg/m³ to kg/m³ is a factor of 1e-9; verify each variable's native convention). Do not treat total-column quantities as concentrations.

```bash
python scripts/prepare_data.py --task air_pollution \
  --source /data/cams_all_fields.zarr --source /data/aurora_pollution_statics.nc \
  --mapping /data/cams_to_aurora.json \
  --start 2017-10-01 --end 2024-12-31 \
  --data-vintage "CAMS analyses; include model-cycle provenance in your source metadata" \
  --output data/cams_1p5.zarr

bash scripts/run_all.sh air_pollution
python scripts/evaluate_air_pollution.py \
  --checkpoint runs/air_pollution/paradis_seed42/best.pt \
  --horizon-hours 120
```

The published checkpoint already saw training data through May 2022, so the example uses 2023 for validation and 2024 for testing. This avoids calling an inherited training period unseen validation. Those dates are **not** the original paper's June–November 2022 test period. For an additional evaluation on that historical test, use `--start 2022-06-01 --end 2022-11-30` and a separate `--output` directory. CAMS model changes create distribution shifts; record which analysis/reanalysis product and cycles supplied each period. A faithful from-foundation reproduction would additionally require the earlier CAMS-reanalysis adaptation stage and its original loss weights.

The evaluator writes physical-unit, area-weighted RMSE, MAE and bias separately for each variable, pressure level and 12-hour lead. It never averages different physical units into one headline score. Aurora's pollution-specific decoder hooks, differencing, transforms and clipping remain in use for the released air-pollution class. The tiny pollution integration test exercises these hooks. The pollution configuration uses float32, following the paper's numerical-stability motivation. The default generic loss weighting for new pollution variables is 1 unless overridden; this is not the paper's persistence-error-calibrated pollution weighting.

## 7. Copy the results back

```bash
mkdir -p local-artifacts
python scripts/export_results.py --runs runs --output local-artifacts/advection_results.zip
```

Copy `advection_results.zip` back to this workspace. It contains configs, manifests, logs, per-case errors and comparisons, and excludes large model checkpoints by default. Use `--include-checkpoints` only if you also want to transfer the trained models. Keep the large checkpoints on the GPU server until the results have been reviewed.

To make a new source bundle after editing:

```bash
python scripts/package_project.py --output ../aurora_cyclone_tracking.zip
```

The source bundle excludes `.venv`, data and runs. The package refuses to overwrite existing training/evaluation/result directories by default so an earlier experiment is not lost.
