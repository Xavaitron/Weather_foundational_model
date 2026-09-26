# Pretrained GraphCast inference

First milestone: run the original, unmodified GraphCast checkpoint on official ERA5 sample data. Model: **0.25 degrees, 37 pressure levels, trained on 1979-2017**, six-hour forecast steps. This runs on the Linux SSH server after you push and pull this folder.

## Quick start

This runner targets **Linux x86_64 with an NVIDIA GPU**. From Windows, first connect to your Linux GPU server over SSH; these Bash commands do not run in PowerShell. From the repository root on that server, set up the environment, verify the GPU, and make a six-hour forecast:

```bash
conda create -n graphcast-bootstrap python=3.11 -y
conda activate graphcast-bootstrap
PYTHON_BIN="$(command -v python)" bash scripts/setup_inference.sh
bash scripts/run_inference.sh --check-device
bash scripts/run_inference.sh --steps 1
```

The first prediction downloads the checkpoint and sample data, then saves NetCDF predictions, a temperature plot, and run metadata in a timestamped folder under `outputs/`. See the steps below for GPU selection, longer forecasts, and troubleshooting.

## 1. Install on the server

From the root of your pulled repository:

```bash
# Use a separate environment rather than the server's existing geomotion env.
conda create -n graphcast-bootstrap python=3.11 -y
conda activate graphcast-bootstrap
PYTHON_BIN="$(command -v python)" bash scripts/setup_inference.sh
```

If `python3.11` is already installed, just run `bash scripts/setup_inference.sh`. The setup creates `.venv` and installs a pinned original GraphCast revision, its inference dependencies, and JAX with pip-provided CUDA 12 libraries. The newer CUDA-capable driver reported by `nvidia-smi` does not require us to use the same CUDA version for pip packages. Do not install or change the server GPU driver for this setup. See [JAX's installation guidance](https://docs.jax.dev/en/latest/installation.html).

The upstream GraphCast package is installed with `--no-deps` after installing the dependencies in `requirements-inference.txt`. This deliberately avoids its notebook-only `colabtools`/Cartopy dependencies. `pip check` can consequently report those unused upstream requirements; model imports are checked by the setup script.

## 2. Check the selected GPU

```bash
nvidia-smi -i 6
bash scripts/run_inference.sh --check-device
```

GPU 6 was an idle 48 GB RTX A6000 in your snapshot. Check that it is still available under your server's allocation rules. The launcher resolves its NVIDIA index to a GPU UUID and exposes only that GPU to JAX. It disables JAX's large startup memory preallocation. GPU 4's reported error is not repaired by this project.

To select another available device, use `GRAPHCAST_GPU=2 bash scripts/run_inference.sh --check-device` (the variable also accepts a GPU UUID). In a scheduler allocation, select the GPU assigned to your job.

## 3. Run one six-hour prediction

```bash
bash scripts/run_inference.sh --steps 1
```

This downloads the original pretrained weights (~146 MB), three small normalization files, and the official 37-level ERA5 sample (~2.85 GB). Downloading uses the public `dm_graphcast` bucket anonymously; no CDS account or Google Cloud credentials are required. Files are cached under `data/`, with size and MD5 verified. An interrupted download restarts on the next run; it is never mistaken for a completed file.

Allow several minutes for the first JAX compilation. Predictions are saved in a timestamped directory under `outputs/`:

| File | Meaning |
| --- | --- |
| `prediction_006h.nc` | All predicted fields at +6 h, including pressure levels and valid time |
| `temperature.png` | ERA5 reference, predicted 2 m temperature, and their difference |
| `run.json` | Checkpoint/data identifiers, package versions, initialization time, completion status, and 2 m temperature RMSE |

Each full-resolution prediction contains about 236 million values (~0.94 GB at float32 before file overhead). Have at least **10 GB free disk and preferably 32 GB or more host RAM** for the first run; this is planning guidance, not a measured peak. GPU and host memory requirements differ. The A6000 is the intended device, but the full run still needs to be validated on your server.

## 4. Extend after the first run succeeds

```bash
# 24 hours: official input file is about 5.69 GB
bash scripts/run_inference.sh --steps 4

# Keep a named result directory (it must be empty or new)
bash scripts/run_inference.sh --steps 4 --output-dir outputs/baseline_24h
```

The script supports **1-12 steps (6-72 hours)** using the available full-resolution sample files. The 12-step sample is about 13.26 GB. It selects the smallest sample covering the requested horizon, keeps the first two states as inputs, and writes one forecast step per file. A ten-day run will need prepared data/forcings beyond these samples; it is not enabled by silently extending this demo's data.

The model only receives two input weather states plus known forcings. Future ERA5 weather fields provide the output structure and verification reference; their values are replaced with NaNs in the template passed into the forecast. The RMSE is a sample-run sanity check, not a 2021-2022 benchmark result. Do not tune the model or choose hyperparameters from this 2022 example when you later establish the held-out test.

## Troubleshooting

| Symptom | Next action |
| --- | --- |
| CUDA backend initialization fails | Run `--check-device`; confirm the selected GPU is healthy and available. Record the complete traceback and `.venv/bin/python -m pip freeze`. |
| CUDA library conflict | Check whether `LD_LIBRARY_PATH` injects an incompatible system CUDA/cuDNN into the pip environment. See the JAX link above. |
| Out of memory | Start with `--steps 1` on an available A6000, inspect both host RAM and VRAM, and share the traceback. Fewer steps reduce host/output storage; they do not eliminate the one-step model's GPU memory requirement. |
| Download interruption | Rerun the same command; verified cached files are reused and the incomplete object restarts. |
| Output directory already exists | Use a new directory, or omit `--output-dir` for a timestamped one. |
| Installation failure | Send the first pip error and Python version. The specified setup targets Linux x86_64 and Python 3.11. |

To download assets separately, without initializing JAX: `.venv/bin/python scripts/infer_graphcast.py --download-only --steps 1`.

## Project context and provenance

Local verification: Python compilation, CLI help, Bash syntax, sample selection, checksum encoding, and corrupt-cache replacement passed. The CPU integration test and full pretrained GPU run have **not** been executed here; local installation of the numerical dependencies was interrupted because downloads were too slow. After server setup, the included small random-model test exercises the upstream wrappers, two-step rollout, NetCDF round trip, and plot generation:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m unittest discover -s tests -v
```

This test checks pipeline behavior only. The real checkpoint is used by `scripts/run_inference.sh`.

See [the research notes](docs/research_context.md) for the Aurora/advection context, dataset choices, experiment split, and unresolved 0.1-degree requirement. See [the GraphCast model reference](docs/graphcast_model_reference.md) for the paper's architecture, data dimensions, and inference details.

Inference follows [DeepMind's original notebook at the pinned revision](https://github.com/google-deepmind/weathernext/blob/97d1ad50b0b7af4aaed7790167dffa769bae1f2c/graphcast_demo.ipynb). The [current upstream repository](https://github.com/google-deepmind/weathernext) has reorganized GraphCast under WeatherNext. We pin the original API so those changes do not silently change this runner. Upstream code, weights, and ERA5 data retain their respective licenses and attribution requirements.

`data/`, `outputs/`, and Python environments are excluded by `.gitignore` and should not be committed.
