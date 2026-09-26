# GraphCast fine-tuning experiment

This is a research implementation for two matched experiments: pretrained GraphCast fine-tuning and the same fine-tuning with a learned latent advection adapter. The requested global input/output grid is **0.1°**, with 37 pressure levels. This is not a reproduction of the complete PARADIS architecture. Full-resolution training must pass the memory feasibility gate below before execution.

## Data and experiment protocol

- Training: 2016–2019; validation: 2020; final testing: 2021–2022.
- Use the original checkpoint trained through 2017, not a later checkpoint that has seen validation/test years. Its overlap with 2016–2017 fine-tuning data is expected and must be disclosed.
- Keep both historical input states, every forecast target, and each precipitation accumulation inside its split. Initialization times are six-hourly. No 2021/2022 weather values are used for development here.
- Read the public WeatherBench 2 **hourly, 37-level, 0.25°** ERA5 store one window at a time. The commonly used preprocessed six-hourly store has only 13 levels and is not interchangeable with this checkpoint.
- Sum the six hourly precipitation accumulations ending at each state time. Select the other variables at the state time. Generate calendar and solar forcings with the official GraphCast utilities.
- Interpolate periodically in longitude onto 1801 × 3600 points. This is bilinearly interpolated ERA5, not independent native 0.1° information. It is not conservative regridding; wind components are interpolated as components. This limitation must accompany precipitation and small-scale interpretations.
- Retain the pretrained normalization files. Use the same data ordering, seed, optimization, rollout length, weighted MSE and validation dates for both variants.
- Current defaults (10 updates, four validation dates, one forecast step) are **engineering pilot settings**, not an adequate training schedule or benchmark. Validation uses physical 2020 data, but the reported loss remains the normalized GraphCast training objective.
- Freeze the schedule and model selection rule on 2020 before opening the 2021–2022 test evaluation. Planned final metrics: physical-unit latitude-weighted RMSE and ACC by variable/level/lead, training-period climatology, and spherical spectral amplitude/coherence. The diagnostic runner currently implements RMSE; ACC, climatology and spherical spectra remain to be implemented for the final benchmark.

## Adapter

`finetuning/advection.py` inserts one residual transport adapter after the existing mesh processor. It compresses the 512-channel mesh state to 16 learned modes, predicts separate bounded east/north angular displacements for each mode, transfers modes to a regular helper grid, traces backwards on the sphere, and samples with differentiable bicubic interpolation. Longitude is periodic; interpolation stencils reflect across poles with a half-turn in longitude.

The correction is the transported sample minus the zero-displacement sample. A zero-initialized projection lifts this difference back to 512 channels, making the adapter a mathematical identity at initialization. The small-model test matches exactly; separately compiled full-size BF16 runs can differ slightly through numerical rounding. Displacement-network gradients become active after the lift first changes. The helper grid defaults to 1°; this is internal to the adapter, while the requested model input/output grid remains 0.1°.

Project-specific choices include three-neighbor inverse-distance mesh transfer, the helper-grid resolution, one adapter after the processor, the displacement bound, and retaining GraphCast's loss. PARADIS instead composes dedicated advection, diffusion and reaction operators in its own architecture. Learned displacements here are latent transport parameters, not measured winds. This prototype does not establish physical conservation or long-rollout stability.

The pretrained mesh remains unchanged when refining the latitude/longitude grid. More grid points feed each mesh neighborhood, changing aggregate statistics. A resolution-transfer ablation (including aggregation normalization) remains necessary before interpreting model quality; shape compatibility alone does not establish resolution invariance.

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

Compiler memory estimates are not measured training peak memory and exclude some runtime overhead. The trainer also checks free GPU memory before its first execution, reserving 2 GiB. A memory rejection must be resolved through implementation/compute changes; do not silently lower the required resolution. Merely having two 48 GiB cards does not pool their memory for the current single-device implementation.

## Training commands after feasibility passes

Run each variant into a new directory, using the same seed and schedule. The following is a pilot command, not a recommendation for final convergence:

```bash
GRAPHCAST_GPU=2 bash scripts/run_finetune.sh \
  --checkpoint 'data/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz' \
  --stats-dir data/graphcast/stats --resolution 0.1 \
  --variant baseline --updates 10 --seed 0 --output runs/baseline-pilot
```

Use `--variant advection --output runs/advection-pilot` for the matched treatment. `--train-window work/era5-train-pilot-20160101.nc` repeats a single training window for engineering only; it does not represent four years of training. The normal path streams training windows from the full source. Do not use a single-window pilot to draw skill conclusions.

`run.json` records hashes, configuration, memory estimate, status and completed updates. `metrics.jsonl` records loss, gradient norm, update time and scheduled validation loss. `latest.npz` and `best.npz` contain model weights. `training-state.pkl` contains optimizer/sampler state for `--resume` into a new directory; load only trusted local files. Pilot checkpoints are not final experiment results.

Adapter checkpoints embed their architecture metadata. For inference use `finetuning.checkpoints.predictor` inside `hk.transform_with_state`, with loaded checkpoint parameters and the matching statistics. The original repository inference script constructs stock GraphCast and must not be used to evaluate adapter checkpoints, because it can ignore the extra weights. The tests verify saving, loading and predicting with the adapter.

## Diagnostic evaluation

`python -m finetuning.evaluate --checkpoint runs/VARIANT/best.npz --stats-dir data/graphcast/stats --split val --steps 4 --count 4 --output runs/VARIANT-validation` runs physical-unit, area-weighted RMSE by variable, pressure level and forecast lead. Select a GPU with `CUDA_VISIBLE_DEVICES` and disable JAX preallocation as above. This runner passes a zero target template into prediction; verification targets are not forecast inputs. Its RMSE aggregates squared errors across dates before taking the square root. The complete runner has not yet been exercised on a full checkpoint; its metric/alignment logic is tested.

Use a new output directory on each run. Test evaluation additionally requires `--frozen-protocol PATH`, recording a hash of the protocol already chosen on validation. This does not prove the research choices were frozen: the team must actually preserve the protocol and avoid tuning on test results. No test evaluation has been launched.

Optional `GRAPHCAST_GEOMETRY_CACHE=work/geometry-cache` persists the immutable graph geometry between processes. The key includes coordinates, model configuration and the pinned upstream version. Only use a trusted local directory: these cache files use Python pickle. No weather fields or trainable parameters are stored in the geometry cache.

The loader explicitly ignores one unused legacy checkpoint leaf, `mesh2grid_gnn/~_networks_builder/decoder_nodes_mesh_nodes_mlp/~/linear_1`. All active pretrained parameters must match by name and shape; no random replacement of missing active weights is permitted.

## Measured status

See [experiment-status.md](experiment-status.md) for completed engineering runs, test results and the measured 0.1° memory blocker.

## References and reading scope

- [GraphCast, arXiv:2212.12794v2](https://arxiv.org/abs/2212.12794v2): main paper and supplement, including normalization, graph construction, training, evaluation and spectral analyses. The original model uses two historical states and six-hour residual forecasts; the requested split and resolution are project-specific adaptations.
- [Neural semi-Lagrangian advection / PARADIS, arXiv:2601.21151v3](https://arxiv.org/abs/2601.21151v3): main paper and appendices, especially learned latent transport, spherical backtracing, interpolation and ADR composition. Version 3 (September 2026) differs substantially from the initial version, so do not mix their model/compute claims.
- [Official GraphCast source](https://github.com/google-deepmind/weathernext/tree/97d1ad50b0b7af4aaed7790167dffa769bae1f2c/graphcast) and [PARADIS author source](https://github.com/Wx-Alliance-Alliance-Meteo/paradis_model) were checked against the implementation design.
- [WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) identifies the public ERA5 stores. Data provenance/units and chronology still need to be checked on every new source.
