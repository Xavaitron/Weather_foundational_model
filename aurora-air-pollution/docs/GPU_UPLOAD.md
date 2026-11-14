> Historical record. See [last verified results, 9 November 2026](RESULTS_2026-11-09.txt) for the completed pilot and stopped historical pipeline. Server log paths below refer to the deployment; operational logs are not included in Git.

# What to upload and run later

## Server continuation on 30 October 2026

Both data pilots passed after fixing CAMS forecast-period metadata decoding. All six full variants completed two optimizer steps on real EAC4 data. A durable real-training pipeline is now launched; see [TRAINING_RUN.md](TRAINING_RUN.md) for its preliminary January run, full-data stages and live status files. The shared NAS reported 5.1 TiB free at launch. `/home` had about 13 GB free and `/media/data_dump` about 24 GB free; neither is used for this project's datasets, weights or environment. Shared free space is not a reserved project quota.

Code is deployed at `/media/nas_mount/anwar2/aurora_airpollution_shohom_20261029/aurora_airpollution` on the supplied IIITD host, with an isolated Python environment in its parent `.venv`. Use the network disk for datasets/checkpoints; `/home` is nearly full. GPU 2 is an RTX A6000 and hosts an existing unrelated process. Validation uses only the remaining memory and must not stop other processes.

All six full pretrained variants passed two synthetic optimizer updates, using 23.53–23.54 GiB peak allocated memory, followed by real-data trials. Both official assets have verified publisher hashes. `validation/server/real_trials.status` and `real_<method>.log` preserve the completed pilot evidence. The active training supervisor reports to `runs/operations/training_pipeline.json`. Do not launch duplicate downloads or GPU training while it is active. The token and raw data are not included in the local backup.

All ADS requests are sequential, following [ECMWF's current service guidance](https://forum.ecmwf.int/t/data-stores-service-protection-measures-affecting-data-downloads-and-account-access-cds-ads-ewds-ecds/15481). Slow archive requests can remain queued/running for some time; inspect existing request status before resubmitting.

The live ADS audit found that literal CAMS `type=analysis` lacks historical PM fields at the beginning of the planned interval. See DATA_PROTOCOL.md. For the full-period option consistent with Microsoft's public input example, use `cams_forecast_t0` and `configs/cams_t0.json` explicitly. Do not combine those outputs with literal-analysis experiment results.

The ADS account is active and both dataset licences are accepted. The project API token is already configured on this server with owner-only permissions. For a different host, configure the token privately before running:

```bash
cd /media/nas_mount/anwar2/aurora_airpollution_shohom_20261029/aurora_airpollution
../.venv/bin/python scripts/download.py --product eac4 --start 2003-01-01 --end 2003-01-02 --execute
../.venv/bin/python scripts/download.py --product cams_forecast_t0 --start 2017-10-01 --end 2017-10-02 --execute
```

The token remains in `.secrets/ads.json` with private filesystem permissions, excluded from version control and bundles. `ADS_API_KEY` is also supported. After inspecting and validating both pilot responses, use the monthly builder for EAC4 as below and for `cams_forecast_t0` with output `data/processed/cams_t0_1p5.zarr`. Fit its training statistics into `data/processed/cams_t0_stats.json`. Run `bash scripts/run_matrix.sh configs/cams_t0.json` to select that product explicitly.

`scripts/pretrained_check.py` runs two optimizer updates on the real pretrained model using synthetic inputs. It measures integration and memory only, discards updates, and reports no forecast skill. `scripts/gpu_trial.py` remains the required real-data trial before scientific training.

## Upload

Upload **`aurora_airpollution_gpu_bundle.zip`** and unzip it in your GPU workspace. It contains code, configs, tests, research notes and validation evidence. It contains **no CAMS data, pretrained weights, API credentials, local virtual environment or trained model**.

If data are prepared on another machine, also transfer the two Zarr directories, their manifests, both statistics JSON files, and `data/assets/`. Keep their relative paths as below. Otherwise download and prepare data directly on a machine with access to the GPU's storage.

```text
aurora-air-pollution/
  data/assets/aurora-0.25-12h-pretrained.ckpt
  data/assets/aurora-0.4-air-pollution-static.pickle
  data/processed/eac4_1p5.zarr/
  data/processed/eac4_1p5.zarr.manifest.json
  data/processed/eac4_stats.json
  data/processed/cams_t0_1p5.zarr/
  data/processed/cams_t0_1p5.zarr.manifest.json
  data/processed/cams_t0_stats.json
```

The optional warm-start checkpoint is `aurora-0.4-air-pollution.ckpt`. It is unnecessary for the main experiment.

## Machine and installation

Use Linux, Python 3.12, and an NVIDIA GPU supporting bfloat16. An **A100/H100 with 80 GB** is a sensible first allocation: [Aurora's fine-tuning guidance](https://microsoft.github.io/aurora/finetuning.html) uses an A100 80 GB. Our coarser grid should reduce activations, but full-model weights, gradients and Adam states remain large. On the shared RTX A6000, the full 1.26B-parameter baseline completed two synthetic optimizer steps at 1.5° with 23.54 GiB allocated (23.99 GiB reserved). This leaves little headroom on a 24 GB card. Real-data and larger-rollout checks remain separate requirements.

At 142 dynamic channels × 120 × 240 float32 values, a frame is about **16.36 MB**, or **11.95 GB per year** at two frames/day before compression. EAC4 2003–2021 is about 227 GB; the full CAMS period here adds about 62 GB. These are calculated storage estimates, not observed compression ratios. Budget extra space for checkpoints and raw monthly staging; roughly 1 TB working space is a practical starting allowance if raw data are staged and removed only after verification. Keeping all native-resolution years requires substantially more.

From the unzipped package root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[dev,download]'
python -m pytest -q
python scripts/smoke.py --device cuda --full-grid
```

Use the CUDA 12.6 or 12.9 wheel instead if required by the host driver; keep the PyTorch versions fixed. The commands are from [PyTorch's versioned installation guide](https://pytorch.org/get-started/previous-versions/). Server CUDA smoke validation has now passed for all six tiny-model variants; full pretrained memory validation is a separate check.

## Obtain assets and pilot the data request

```bash
HF_HUB_DISABLE_XET=1 python scripts/fetch_assets.py
python scripts/download.py --product cams_forecast_t0 --start 2017-10-01 --end 2017-10-02
```

The second command only writes request JSON. Check the ADS account's dataset terms, then set `ADS_API_KEY` privately in the GPU/data host's environment and append `--execute` to actually retrieve. Do not paste credentials into chat or source files.

```bash
python scripts/download.py --product cams_forecast_t0 --start 2017-10-01 --end 2017-10-02 --execute
python scripts/download.py --product eac4 --start 2003-01-01 --end 2003-01-02 --execute
```

On a new deployment, verify variable names, dimensions, units, pressure levels, timestamps and explicit forecast-lead-zero provenance. These pilot checks have passed on the supplied account/server. If the archive contains multiple NetCDF files, the downloader stops with their names for explicit handling. If analysis is unavailable for a required field/date, establish the appropriate ECMWF retrieval route before bulk downloading. Do not silently replace missing data or substitute future forecasts.

After pilot acceptance, retrieve **one month at a time**:

```bash
python scripts/download.py --product cams_forecast_t0 --start 2017-10-01 --end 2017-11-01 --execute
python scripts/download.py --product eac4 --start 2003-01-01 --end 2003-02-01 --execute
```

Complete CAMS through 2022-11-30 and EAC4 through 2021-12-31. Preserve the saved request and hash files.

After the pilots pass, `build_dataset.py` can automate the full monthly loop:

```bash
python scripts/build_dataset.py --product cams_forecast_t0 --start 2017-10-01 --end 2022-12-01 --raw data/raw_grouped --days-per-request 31 --output data/processed/cams_t0_1p5.zarr --execute --discard-verified-raw
python scripts/build_dataset.py --product eac4 --start 2003-01-01 --end 2022-01-01 --raw data/raw_grouped --days-per-request 31 --output data/processed/eac4_1p5.zarr --execute --discard-verified-raw
```

Omit `--execute` to inspect its plan. The optional discard flag removes only that month's downloaded NetCDF files after checking exact timestamps and every prepared value for finiteness; request/hash records remain. Without it, native files accumulate. The loop skips already completed months and rejects partial months. It does not make an interrupted Zarr append transactional.

## Prepare and validate the stores

Example for the first CAMS month:

```bash
python scripts/prepare.py \
  --surface 'data/raw/cams_forecast_t0/2017-10/*-sfc.nc' \
  --atmos 'data/raw/cams_forecast_t0/2017-10/*-plev.nc' \
  --static data/assets/aurora-0.4-air-pollution-static.pickle \
  --trust-official-static-pickle \
  --product cams_forecast_t0 \
  --output data/processed/cams_t0_1p5.zarr
```

Repeat for subsequent months with `--append`; the script rejects overlapping timestamps and changed static data/products. Use corresponding EAC4 paths and `--product eac4` for `data/processed/eac4_1p5.zarr`. Pickle loading is restricted to the explicitly trusted official asset. A NetCDF static file is also supported.

Append writes are not transactional: keep the current raw month until successful completion and verification. If an append is interrupted, restore the preceding known-good store before retrying. Do not run concurrent appends.

After completing both stores:

```bash
python scripts/stats.py --data data/processed/eac4_1p5.zarr --start 2003-01-01 --end 2021-01-01 --output data/processed/eac4_stats.json
python scripts/stats.py --data data/processed/cams_t0_1p5.zarr --start 2017-10-01 --end 2022-04-01 --output data/processed/cams_t0_stats.json
python scripts/preflight.py --config configs/eac4.json
python scripts/preflight.py --config configs/cams_t0.json
```

Statistics cover training only. They validate all training values for finiteness and reject negative pollution inputs. Investigate errors rather than clipping them away. `preflight.py` samples each split; full validation/test values are checked as loaded during evaluation.

## Real GPU trial, then training

First load actual weights and real data for two optimizer updates, including allocation of Adam state:

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:native
python scripts/gpu_trial.py --config configs/cams_t0.json --method baseline
python scripts/gpu_trial.py --config configs/cams_t0.json --method debezenac_gaussian --output runs/gpu_trial_gaussian.json
```

These trials discard updated weights and save only diagnostics. If memory is insufficient, obtain a larger GPU or add a separately tested distributed strategy. Lowering gradient accumulation does not reduce single-microbatch activation memory. `trainable=adapter` is a different experiment; it should use an already pollution-trained backbone and must not be reported as full fine-tuning.

Run one two-stage baseline first:

```bash
python -m airadv.train --config configs/eac4.json
python -m airadv.train --config configs/cams_t0.json --initial-state runs/eac4_baseline_seed42/best.pt
python -m airadv.evaluate --checkpoint runs/cams_t0_baseline_seed42/best.pt --output runs/cams_t0_baseline_seed42/evaluation.json
```

Then repeat with `--method nowcast_evolution`, `adr_latent`, or `debezenac_gaussian`, giving each run its own `--output` path and using the matching EAC4 checkpoint. The complete three-seed matrix is:

```bash
bash scripts/run_matrix.sh configs/cams_t0.json
python scripts/compare.py runs/cams_t0_*/evaluation.json --output runs/comparison.csv
```

The matrix resumes compatible `last.pt` checkpoints when present. A new run refuses to overwrite an existing experiment directory. To continue one interrupted run manually, use:

```bash
python -m airadv.train --config configs/cams_t0.json --resume runs/cams_t0_baseline_seed42/last.pt
```

For non-baseline runs, repeat the original `--method`, `--seed` and `--output` overrides when resuming. Resume checks data paths, stats hash, model settings and optimizer settings. Exact optimizer-boundary replay was verified on CPU; CUDA kernel nondeterminism can still affect bitwise reproducibility.

The default training budgets are initial research settings, not paper hyperparameters or claims of convergence. Modify config step counts based on validation behaviour before running the full matrix. For two-step backpropagation, set `rollout_steps=2` and rerun the GPU memory trial with an equivalent rollout check; the current trial measures one-step memory. Saved `best.pt` and `last.pt` include optimizer state and can be large.

## Current continuation

GPU access, official weights and authenticated data pilots are ready. The active pipeline prepares the first month and then starts saved fine-tuning, with historical retrieval and comparative training following automatically. Server validation establishes integration and memory fit, not forecast accuracy or an advantage of advection. Keep the test interval untouched while choosing settings.

