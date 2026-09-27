# GraphCast memory optimization checks — 27 September 2026

SSH access is restored. Both full GraphCast and GraphCast-small exceed the available 48 GiB GPUs at 0.1°, even with BF16, activation checkpointing and buffer donation. The small model successfully runs at its native 1° resolution; those runs are engineering pilots.

## Changes and validation

- Confirmed the trainer already uses GraphCast's BF16 wrapper and checkpoints message-passing blocks, graph embeddings and output networks. Parameters and Adam moments remain float32.
- Enabled parameter/optimizer buffer donation at the outer compiled update. Weather batches are retained so repeated pilot updates remain valid. This follows [JAX's buffer-donation interface](https://docs.jax.dev/en/latest/buffer_donation.html).
- Selected the checkpoint's pressure levels before interpolation, avoiding interpolation of unused levels when the small model receives a cached 37-level window.
- Added model configuration, pressure levels and precision/donation settings to new run/profile metadata. The profiler accepts `--no-donate` for comparisons.
- Nine server tests passed in 95.88 seconds. The new regression checks two consecutive donated versus non-donated Adam updates with xarray inputs and confirms that weather inputs remain reusable. It also passed locally.

## Memory measurements

These are XLA compiler estimates, not measured runtime peaks. Runs use batch size one, a single six-hour forecast step, BF16 computation and full-parameter AdamW. Kernel autotuning was disabled during these checks.

| Configuration | Resolution | Pressure levels | Estimated device memory | Real updates |
|---|---:|---:|---:|---:|
| Full GraphCast, existing checkpointing | 0.1° | 37 | 199.38 GiB | 0 |
| Full GraphCast, plus buffer donation | 0.1° | 37 | 198.98 GiB | 0 |
| GraphCast-small baseline, donation | 0.1° | 13 | **180.81 GiB** | 0 |
| GraphCast-small baseline, donation | 1° | 13 | 3.96 GiB | 2 |
| GraphCast-small + advection, donation | 1° | 13 | 4.09 GiB | 2 |

Buffer donation saves approximately 0.40 GiB (0.20%) for the full model at 0.1°. Graph activations dominate its memory; parameter and optimizer storage are a small fraction.

## Small-model engineering pilots

The official small checkpoint was downloaded and checksum-verified. It was pretrained on ERA5 1979–2015 and natively uses a 1° grid with 13 levels and a smaller mesh. Inspection confirms it still has 512 latent channels and 16 message-passing steps. See the [official checkpoint descriptions](https://github.com/google-deepmind/weathernext/blob/main/docs/weathernext1_graph/README.md).

Both variants used the same 2016-01-01 window, seed and learning rate, and completed two updates at 1°. Baseline losses were 0.80497 and 0.82677; adapter losses were 0.80497 and 0.82803. These are training losses on a repeated engineering window, not validation scores or evidence of improved forecasting. The pilot learning rate is untuned.

Reloaded weights were finite float32 values, and 258 pretrained tensors changed in each run. The baseline has 35,979,347 active parameters. The adapter checkpoint has 36,012,163 parameters, includes all three adapter modules, and has a nonzero lift norm of 0.00148277.

Server outputs are under `/home/anwar/Weather_foundational_model_finetune/runs/engineering-small-1deg-baseline` and `runs/engineering-small-1deg-advection`.

## What remains

The measured 0.1° small-model estimate is 194,148,170,980 bytes. Its smaller mesh and reduced pressure-level count help, but the high-resolution grid and 512-wide edge activations remain large. No 0.1° optimizer update was executed for either checkpoint.

The next implementation work is to stream/chunk the grid-to-mesh and mesh-to-grid computations with verified gradients, or implement model sharding across sufficient aggregate GPU memory. Gradient accumulation or ordinary data parallelism cannot make a single global example fit.

A 1°/13-level pilot does not meet the requested 0.1° experiment. The full experiment still needs a workable full-resolution implementation or sufficient compute, reliable full-source training data access, a schedule chosen using 2020 validation, and frozen-protocol evaluation on 2021–2022. No test-weather values were used in these checks; synthetic profiles read only schema/coordinates.

All profiling and pilot jobs have finished. Code is saved in the isolated experiment worktree on branch `codex/finetune-0p1`; it has not been pushed to GitHub. [Raw optimization evidence](optimization-evidence.tar.gz) includes compiler profiles, run manifests, losses, checkpoint audits and the test log.
