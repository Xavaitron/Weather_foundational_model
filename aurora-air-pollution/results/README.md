# Aurora air pollution results

Return to the [workflow README](../README.md) for architecture and run commands.

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

The 9 November 2026 snapshot records 62 completed January frames and 7,368 historical EAC4 frames through 31 January 2013. The historical worker stopped on the February 2013 ADS request after a TLS certificate verification failure. CAMS forecast lead zero had two pilot frames. The completed evaluation's full per-pollutant, all-lead JSON was on the server and is absent from the tracked evidence. The table above includes all published PM2.5 rows in this checkout. There are no recorded held-out comparison scores for the learned advection variants or the official 0.4° air-pollution checkpoint. See the [dated evidence](../docs/RESULTS_2026-11-09.txt) and [validation record](../validation/VALIDATION.md).
