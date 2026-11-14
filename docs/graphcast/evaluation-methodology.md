# Paired GraphCast verification

Both full GraphCast models finished 1,000 fine-tuning updates on ERA5 windows from 2016–2019. Their best checkpoints were selected using the same four 2020 validation windows. The final training-validation losses were 0.74183655 for baseline and 0.74172211 for advection; these are normalized training losses, not physical-unit RMSE.

The new benchmark freezes those two checkpoints and evaluates 6-hour forecasts initialized at 12 UTC on the 15th of every month in 2020, 2021 and 2022. This gives 12 dates per year, 36 paired dates and 72 model forecasts. It is a sampled benchmark, not continuous evaluation of every forecast in those years. Evaluation inputs and reference weather are identical for both variants. No test results are used for checkpoint selection or further tuning.

Models retain the full mesh and all 37 pressure levels on a 1° latitude/longitude grid. The network uses BF16 arithmetic. ERA5 is selected at the exact coincident 1° grid points from the 0.25° source; this is point sampling, not conservative area averaging. Six-hour precipitation is the sum of the six hourly accumulations ending at the verification time.

RMSE is calculated in physical units for every predicted variable, pressure level and lead time. Squared errors receive latitude cell-area weights, including the polar caps, and are averaged over space and initialization dates before taking the square root. Errors with different units are not combined into one score.

ACC uses the published WeatherBench 1990–2019 ERA5 climatology, selected by forecast valid-time day of year and UTC hour. That reference excludes 2020–2022. Forecast and observed anomalies are each measured relative to this same climatology; ACC is their area-weighted inner product divided by the product of their area-weighted norms. There is no extra subtraction of a spatial mean. Correlations are calculated separately for each initialization and then averaged equally over dates.

The climatology provides 13 pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925 and 1000 hPa. ACC is reported at these levels and for the five predicted surface variables. RMSE is reported at all 37 levels. Blank ACC entries at other pressure levels mean unsupported climatology, not zero skill. Zero anomaly norms are undefined and cause the run to stop for inspection rather than silently omitting dates.

Results are aggregated separately for 2020, 2021, 2022 and combined 2021–2022. Larger ACC and smaller RMSE are better. A small difference on this monthly sample is not evidence of a statistically established advection benefit. Six-hour skill does not establish multi-day stability or skill. The original pretrained checkpoint includes 1979–2017, overlapping the 2016–2017 fine-tuning years, but not the validation/test years.

Source definitions: [WeatherBench data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), [official ACC implementation](https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py).

## Reproduction

Server checkout: `/home/anwar/Weather_foundational_model_finetune`.

Initial evaluation code revision: `d10022c`; cache preparation revision: `4056225`. Revision `9ca473d` adds a 60-second cloud-request timeout and preserves resume provenance. The run was resumed from its completed paired dates after a climatology request stalled without a timeout. The frozen forecast/metric protocol and checkpoint hashes were unchanged.

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_verification.py
JAX_PLATFORMS=cpu .venv/bin/python -m finetuning.paired_evaluate freeze \
  --protocol runs/paired-evaluation-20260928-frozen.json
CUDA_VISIBLE_DEVICES=GPU-bfe80d06-6876-075e-4681-5de95d94b179 \
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_autotune_level=0 OMP_NUM_THREADS=4 \
  GRAPHCAST_GEOMETRY_CACHE=work/geometry-cache \
  .venv/bin/python -u -m finetuning.paired_evaluate run \
  --protocol runs/paired-evaluation-20260928-frozen.json
```

The freeze command refuses to replace an existing protocol. To resume this run, use its existing protocol and run command. Checkpoint hashes must match. Completed date/model metrics are reused only within the same protocol. The evaluation was launched using `nohup`; disconnecting SSH or VPN does not stop it.

Five CPU workers, in pools of three and two, prepare the frozen evaluation windows into the shared weather and climatology caches. The second pool visits the same frozen date list in reverse order; this changes preparation order only. Only the evaluator uses GPU 2. NetCDF writes are serialized within each process and protected by per-cache-entry locks between processes.
