# GraphCast fine-tuning experiment

This is a research implementation for two matched experiments: pretrained GraphCast fine-tuning and the same fine-tuning with a learned latent advection adapter. The active global input/output grid is **1°**, with the **normal GraphCast checkpoint, full mesh and 37 pressure levels**. This supersedes the original 0.1° target and the small-model fallback; GraphCast-small is not selected for the active experiment. This is not a reproduction of the complete PARADIS architecture. Every training configuration must pass the memory feasibility gate below before execution.

## Data and experiment protocol

- Training: 2016–2019; validation: 2020; final testing: 2021–2022.
- Use the original checkpoint trained through 2017, not a later checkpoint that has seen validation/test years. Its overlap with 2016–2017 fine-tuning data is expected and must be disclosed.
- Keep both historical input states, every forecast target, and each precipitation accumulation inside its split. Initialization times are six-hourly. No 2021/2022 weather values are used for development here.
- Read the public WeatherBench 2 **hourly, 37-level, 0.25°** ERA5 store one window at a time. The commonly used preprocessed six-hourly store has only 13 levels and is not interchangeable with this checkpoint.
- Sum the six hourly precipitation accumulations ending at each state time. Select the other variables at the state time. Generate calendar and solar forcings with the official GraphCast utilities.
- Interpolate periodically in longitude onto **181 × 360 points (1°)** from the 0.25° ERA5 source. This changes the input/output grid while retaining all 37 pressure levels and the pretrained mesh. It is not conservative regridding; wind components are interpolated as components. This limitation must accompany precipitation and small-scale interpretations.
- Retain the pretrained normalization files. Use the same data ordering, seed, optimization, rollout length, weighted MSE and validation dates for both variants.
- Current defaults (10 updates, four validation dates, one forecast step) are **engineering pilot settings**, not an adequate training schedule or benchmark. Validation uses physical 2020 data, but the reported loss remains the normalized GraphCast training objective.
- Freeze the schedule and model selection rule on 2020 before evaluating 2021–2022. The paired benchmark now implements physical-unit, latitude-area-weighted RMSE and ACC by variable, level and lead. ACC uses the published 1990–2019 ERA5 climatology, excluding validation/test years. Spherical spectral diagnostics remain future work.

## Adapter

`finetuning/advection.py` inserts one residual transport adapter after the existing mesh processor. It compresses the 512-channel mesh state to 16 learned modes, predicts separate bounded east/north angular displacements for each mode, transfers modes to a regular helper grid, traces backwards on the sphere, and samples with differentiable bicubic interpolation. Longitude is periodic; interpolation stencils reflect across poles with a half-turn in longitude.

The correction is the transported sample minus the zero-displacement sample. A zero-initialized projection lifts this difference back to 512 channels, making the adapter a mathematical identity at initialization. The small-model test matches exactly; separately compiled full-size BF16 runs can differ slightly through numerical rounding. Displacement-network gradients become active after the lift first changes. The helper grid defaults to 1°; this is internal to the adapter, while the active model input/output grid is also 1°.

Project-specific choices include three-neighbor inverse-distance mesh transfer, the helper-grid resolution, one adapter after the processor, the displacement bound, and retaining GraphCast's loss. PARADIS instead composes dedicated advection, diffusion and reaction operators in its own architecture. Learned displacements here are latent transport parameters, not measured winds. This prototype does not establish physical conservation or long-rollout stability.

The pretrained mesh remains unchanged when changing the latitude/longitude grid. The number of grid points feeding each mesh neighborhood changes, which changes aggregate statistics. A resolution-transfer ablation (including aggregation normalization) remains necessary before interpreting model quality; shape compatibility alone does not establish resolution invariance.

## Environment

Use a separate Python 3.11/3.12 environment. The existing inference dependency versions and upstream GraphCast commit are retained. On the experiment server the isolated `.venv` inherits the existing working inference packages and adds `requirements-finetuning.txt`; the shared base environment was not modified.

For a clean environment:

```bash
python3.12 -m venv .venv
PYTHON_BIN=.venv/bin/python bash scripts/setup_inference.sh
.venv/bin/python -m pip install -r requirements-finetuning.txt pytest
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_finetuning.py
```

The server worktree is `/home/anwar/Weather_foundational_model_finetune`, branch `codex/finetune-0p1`. Its `data` symlink reuses the original checkout's downloaded checkpoint/statistics without modifying the original checkout. Select a free GPU explicitly; do not terminate unrelated processes.

## Memory feasibility

The trainer and profiler now default to 1°. The explicit 0.1° command below reproduces the earlier memory investigation; it is not the active training configuration.

`finetuning.profile` compiles a one-step, full-parameter update from synthetic array shapes. It reads only schema/coordinates from the existing sample, discards weather arrays, and executes no optimization. A 2022 filename used as schema does not constitute test-weather evaluation.

```bash
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader)"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
# Compilation-only profiling avoids GPU kernel-benchmark allocations.
export XLA_FLAGS=--xla_gpu_autotune_level=0
.venv/bin/python -u -m finetuning.profile \
  --checkpoint 'data/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz' \
  --stats-dir data/graphcast/stats \
  --schema data/graphcast/dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc \
  --resolution 0.1 --variant baseline --output work/memory-0p1.json
```

The implementation checkpoints each message-passing block, each graph embedding, and each output network. Float32 forward and gradient equivalence is tested against stock GraphCast; BF16 backward rounding may change with recomputation.

BF16 computation is already enabled through the official `Bfloat16Cast` wrapper. Pretrained parameters and Adam moments remain float32. The trainer now also donates parameter and optimizer buffers to the compiled update, allowing XLA to reuse their storage. Weather arrays are not donated, so a pilot can reuse its batch. A two-update regression test checks numerical agreement with donation disabled and verifies the input batch remains usable. This saves parameter/optimizer storage; it does not remove the much larger grid and edge activations. Use `--no-donate` with `finetuning.profile` for a comparison.

### Historical GraphCast-small fallback checks

The official `GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz` checkpoint can use the same trainer and profiler. Its mesh has 10,242 vertices and its task has 13 pressure levels, but it still has 512 latent channels and 16 message-passing steps. Switching its input/output grid to 0.1° therefore does not guarantee that it fits a 48 GiB GPU. Profile that exact configuration first. A native 1° pilot is an engineering fallback and did not satisfy the original 0.1° experiment or the full checkpoint's 37-level task. The same checkpoint and task must be used for both baseline and adapter in any eventual comparison.

The downloaded checkpoint at `work/graphcast-small.npz` has SHA-256 `e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29`. Its checksum was verified against the official Cloud Storage object's MD5 metadata before transfer and against SHA-256 after transfer. Example native-grid engineering run:

```bash
GRAPHCAST_GPU=6 GRAPHCAST_GEOMETRY_CACHE=work/geometry-cache \
XLA_FLAGS=--xla_gpu_autotune_level=0 bash scripts/run_finetune.sh \
  --checkpoint work/graphcast-small.npz --stats-dir data/graphcast/stats \
  --resolution 1.0 --variant baseline --updates 2 \
  --train-window work/era5-train-pilot-20160101.nc \
  --output runs/engineering-small-1deg-baseline
```

Replace `baseline` with `advection` and use a new output directory for its matched engineering run. Disable kernel autotuning for the memory investigation; performance measurements with this flag are not throughput benchmarks.

Compiler memory estimates are not measured training peak memory and exclude some runtime overhead. The trainer also checks free GPU memory before its first execution, reserving 2 GiB. A memory rejection must be resolved through implementation/compute changes; do not silently lower the required resolution. Merely having two 48 GiB cards does not pool their memory for the current single-device implementation.

## Active full-model 1° training commands

Run each variant into a new directory, using the same seed and schedule. The following is a pilot command, not a recommendation for final convergence:

```bash
GRAPHCAST_GPU=2 bash scripts/run_finetune.sh \
  --checkpoint 'data/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz' \
  --stats-dir data/graphcast/stats --resolution 1.0 \
  --variant baseline --updates 10 --seed 0 --output runs/baseline-pilot
```

Use `--variant advection --output runs/advection-pilot` for the matched treatment. `--train-window work/era5-train-pilot-20160101.nc` repeats a single training window for engineering only; it does not represent four years of training. The normal path streams training windows from the full source. Do not use a single-window pilot to draw skill conclusions.

`run.json` records hashes, configuration, memory estimate, status and completed updates. `metrics.jsonl` records loss, gradient norm, update time and scheduled validation loss. `latest.npz` and `best.npz` contain model weights. `training-state.pkl` contains optimizer/sampler state for `--resume` into a new directory; load only trusted local files. Pilot checkpoints are not final experiment results.

Adapter checkpoints embed their architecture metadata. For inference use `finetuning.checkpoints.predictor` inside `hk.transform_with_state`, with loaded checkpoint parameters and the matching statistics. The original repository inference script constructs stock GraphCast and must not be used to evaluate adapter checkpoints, because it can ignore the extra weights. The tests verify saving, loading and predicting with the adapter.

## Paired stage-one fine-tuning

`bash scripts/run_paired_1deg.sh` launches normal GraphCast baseline on GPU 2 and advection on GPU 6. It refuses to reuse an existing output root. Override `RUN_ROOT`, `BASELINE_GPU`, or `ADVECTION_GPU` as needed before launch. The default run root is `runs/full-1deg-stage1-20260927`.

This initial stage uses 1,000 full-parameter updates per variant, batch size one, one six-hour forecast step, learning rate 1e-5, AdamW, clipping at norm 32, and seed 0. Each update samples from eligible 2016–2019 initializations using the same random sequence for both variants. Sampling is with replacement; this is not a complete pass through all four years or a final convergence schedule. Both runs start from the original normal pretrained checkpoint, not the engineering pilot weights.

Initial validation and validation every 50 updates use the same four evenly spaced eligible 2020 initializations. Save a resumable state after the first update, every 25 updates and at completion. `best.npz` is selected by normalized validation loss and may remain the initial pretrained checkpoint if no improvement is seen. Four validation windows provide an initial training diagnostic; they are not the final benchmark. No 2021–2022 weather is accessed.

The shared `work/era5-1deg-window-cache` is capped at 32 GiB of completed NetCDF files. Its identity includes source, split, date, levels, grid and rollout, and per-window locks prevent duplicate downloads between processes. Writes are atomic; least-recently-used, unlocked files are removed as needed. A small temporary overage can occur during concurrent writes. The cache preserves bilinear point-sampling semantics: for an exactly aligned 1° grid, select those points before materializing source arrays. The upstream cloud chunks still span the complete spatial grid, so network transfer can dominate training time.

The run root contains one log and PID file per variant plus the source-code revision. Each variant's `run.json` records progress, the exact validation dates and loading state. `metrics.jsonl` records each training initialization, data-loading time, update time, loss and gradient norm. Both jobs use `nohup` and continue after SSH disconnects. No recurring monitor is installed.

## Paired RMSE and ACC evaluation

Both 1,000-update checkpoints have completed the frozen monthly benchmark: 12 dates per year in 2020, 2021 and 2022, with one six-hour forecast on the full-model 1° grid. See [results](evaluation-results.md) and [methodology](evaluation-methodology.md). This is a sampled benchmark, not continuous evaluation of every date.

`finetuning.paired_evaluate freeze --protocol PATH` records exact dates, forecast leads, checkpoint hashes, grid, climatology and aggregation before reading evaluation weather. `finetuning.paired_evaluate run --protocol PATH` verifies checkpoint hashes, forecasts both models from identical inputs, and saves per-date MSE/ACC plus yearly and combined-test aggregates. Pass these module names to `.venv/bin/python -m`, and use the GPU environment documented above. Existing completed metrics can be resumed only with the same protocol hash. The evaluated protocol is `runs/paired-evaluation-20260928-frozen.json`; do not regenerate it when resuming.

RMSE covers all 37 pressure levels; ACC covers the 13 available climatology levels and all five predicted surface variables. Forecasts receive zero target templates; verification weather is never supplied as future input. RMSE pools squared errors before taking the square root, while ACC averages per-initialization anomaly correlations. Undefined scores stop the run for inspection. See `tests/test_verification.py` for analytical formula, alignment, aggregation and calendar checks.

The older `finetuning.evaluate` remains a standalone RMSE-only diagnostic runner. Use the paired runner for the completed RMSE/ACC benchmark. No test results were used for checkpoint selection or further tuning.

Optional `GRAPHCAST_GEOMETRY_CACHE=work/geometry-cache` persists the immutable graph geometry between processes. The key includes coordinates, model configuration and the pinned upstream version. Only use a trusted local directory: these cache files use Python pickle. No weather fields or trainable parameters are stored in the geometry cache.

The loader explicitly ignores one unused legacy checkpoint leaf, `mesh2grid_gnn/~_networks_builder/decoder_nodes_mesh_nodes_mlp/~/linear_1`. All active pretrained parameters must match by name and shape; no random replacement of missing active weights is permitted.

## Measured status

See [training-stage1.md](training-stage1.md) for completed training and [evaluation-results.md](evaluation-results.md) for RMSE/ACC scores. [full-model-1deg-results.md](full-model-1deg-results.md) records the earlier full-model 1° engineering check. [experiment-status.md](experiment-status.md) also preserves historical 0.1° memory results and small-model checks.

## References and reading scope

- [GraphCast, arXiv:2212.12794v2](https://arxiv.org/abs/2212.12794v2): main paper and supplement, including normalization, graph construction, training, evaluation and spectral analyses. The original model uses two historical states and six-hour residual forecasts; the requested split and resolution are project-specific adaptations.
- [Neural semi-Lagrangian advection / PARADIS, arXiv:2601.21151v3](https://arxiv.org/abs/2601.21151v3): main paper and appendices, especially learned latent transport, spherical backtracing, interpolation and ADR composition. Version 3 (September 2026) differs substantially from the initial version, so do not mix their model/compute claims.
- [Official GraphCast source](https://github.com/google-deepmind/weathernext/tree/97d1ad50b0b7af4aaed7790167dffa769bae1f2c/graphcast) and [PARADIS author source](https://github.com/Wx-Alliance-Alliance-Meteo/paradis_model) were checked against the implementation design.
- [Official GraphCast checkpoint variants](https://github.com/google-deepmind/weathernext/blob/main/docs/weathernext1_graph/README.md) and [JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html).
- [WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) identifies the public ERA5 stores. Data provenance/units and chronology still need to be checked on every new source.
