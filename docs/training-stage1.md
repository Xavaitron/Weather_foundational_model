# Normal GraphCast 1° fine-tuning — stage 1

The baseline and learned-advection fine-tuning jobs have been launched on the server, using the normal pretrained GraphCast checkpoint with all 37 pressure levels and the full mesh. This is real sampling from the ERA5 training years, not the earlier repeated-window engineering test.

| Setting | Both runs |
|---|---|
| Grid | 1°, 181 × 360 points |
| Initial weights | Normal GraphCast, ERA5 1979–2017, originally 0.25° |
| Training source | Public WeatherBench 2 hourly ERA5, 37 levels; windows restricted to 2016–2019 |
| Validation | Four fixed initializations in 2020, before training and every 50 updates |
| Test period | 2021–2022 reserved; no test-weather access |
| Stage length | 1,000 updates each, sampled with replacement using seed 0 |
| Batch / forecast horizon | One global example / one six-hour step |
| Optimizer | AdamW, learning rate 1e-5, beta1 0.9, beta2 0.95, matrix weight decay 0.1, gradient clipping at norm 32 |
| Memory settings | BF16 computation, float32 parameters/moments, activation checkpointing and buffer reuse |
| Checkpointing | After update 1, every 25 updates, and at completion; best model selected by validation loss |
| Shared data cache | 32 GiB cap on completed regridded window files; atomic writes and per-window locks |

The baseline runs on physical GPU 2 and advection on physical GPU 6. Each model uses one A6000. Both start from the original checkpoint, with identical dates and optimization settings; the adapter uses 16 latent modes and a 1° helper grid. The upstream checkpoint's 2016–2017 pretraining overlap remains disclosed.

This initial 1,000-update stage is not a complete pass over all eligible four-year windows or a claim of convergence. Four validation windows are an initial diagnostic, not the final benchmark. The initial validation checkpoint can remain `best.npz` if subsequent fine-tuning does not improve its score.

## Location and progress

Server run root: `/home/anwar/Weather_foundational_model_finetune/runs/full-1deg-stage1-20260927`.

- Baseline PID at launch: `3550683`; log: `baseline.log`; progress: `baseline/run.json`; losses and actual sampled dates: `baseline/metrics.jsonl`.
- Advection PID at launch: `3550684`; log: `advection.log`; progress: `advection/run.json`; losses and sampled dates: `advection/metrics.jsonl`.
- Model weights are written under each variant as `latest.npz` and `best.npz`; resumable optimizer/sampler state is `training-state.pkl`.
- Both processes use `nohup` and continue after SSH disconnects. No recurring monitoring automation has been installed.
- Source-code revision at launch: `b6fb717`, recorded in `code-revision.txt`. Changes have not been pushed to GitHub.

## Verification and throughput

The server's 11-test suite passed in 96.47 seconds. After removing the archive-wide Dask task graph, the two data/cache tests passed again. The first two fresh training windows are 2019-05-27 12:00 UTC and 2018-07-19 12:00 UTC; their downloads and preparation took approximately 89 and 84 seconds. The first 2020 validation window took approximately 76 seconds. Compilation reproduced the earlier estimates: 10.77 GiB for baseline and 10.98 GiB for advection. These are compiler estimates, not measured peaks.

The source stores large 0.25° chunks, so selecting a 1° output grid does not proportionally reduce cloud transfer. Actual data loading is currently much slower than GPU computation. Both jobs share cached windows, and cached validation windows are reused. Completion time depends on cloud throughput and concurrent server use.

[Reproduction guide](finetuning.md) includes the matched launcher and resume instructions. Both runs have now completed at least two actual fine-tuning updates on distinct dates. Initial mean normalized validation loss was 14.84619 for baseline and 14.84814 for advection. These are scores before training; no improvement claim is made. Training losses on different sampled dates are not directly comparable as a learning curve. The jobs continue toward their 1,000-update stage target.

[Progress snapshot, manifests, metrics, checkpoint audit and test logs](../work/training-stage1-evidence.tar.gz) record the verified launch. The snapshot is a point-in-time record; the server logs continue updating.
