# GraphCast baseline vs advection: completed RMSE and ACC benchmark

Both checkpoints were evaluated successfully on 36 matched initializations: 12 monthly dates in each of 2020, 2021 and 2022. Forecasts are six hours ahead on the 1° grid. This is a sampled benchmark, not every forecast in those years.

Lower RMSE and higher ACC are better. RMSE is in the units shown; ACC is dimensionless. Both models completed 1,000 fine-tuning updates. Best checkpoints were chosen using 2020 validation only, before test evaluation.

## 2020 validation

| Variable | Units | Baseline RMSE | Advection RMSE | Baseline ACC | Advection ACC |
|---|---|---:|---:|---:|---:|
| 2 m temperature | K | 0.58369 | 0.58381 | 0.974572 | 0.974559 |
| 850 hPa temperature | K | 0.33977 | 0.33980 | 0.995053 | 0.995052 |
| 500 hPa geopotential | m²/s² | 22.66075 | 22.63036 | 0.999629 | 0.999630 |
| 10 m eastward wind | m/s | 0.48730 | 0.48717 | 0.992412 | 0.992415 |
| 10 m northward wind | m/s | 0.49767 | 0.49725 | 0.992509 | 0.992522 |
| Mean sea-level pressure | Pa | 32.50906 | 32.48933 | 0.998999 | 0.999000 |
| 6 h precipitation | mm | 1.16013 | 1.16032 | 0.879827 | 0.879835 |

## 2021 test

| Variable | Units | Baseline RMSE | Advection RMSE | Baseline ACC | Advection ACC |
|---|---|---:|---:|---:|---:|
| 2 m temperature | K | 0.59267 | 0.59262 | 0.975433 | 0.975437 |
| 850 hPa temperature | K | 0.33541 | 0.33545 | 0.995381 | 0.995382 |
| 500 hPa geopotential | m²/s² | 21.98004 | 21.98575 | 0.999601 | 0.999601 |
| 10 m eastward wind | m/s | 0.48198 | 0.48183 | 0.992056 | 0.992061 |
| 10 m northward wind | m/s | 0.49511 | 0.49462 | 0.992128 | 0.992144 |
| Mean sea-level pressure | Pa | 31.65369 | 31.68169 | 0.998920 | 0.998918 |
| 6 h precipitation | mm | 1.11768 | 1.11784 | 0.880217 | 0.880205 |

## 2022 test

| Variable | Units | Baseline RMSE | Advection RMSE | Baseline ACC | Advection ACC |
|---|---|---:|---:|---:|---:|
| 2 m temperature | K | 0.59732 | 0.59706 | 0.973112 | 0.973134 |
| 850 hPa temperature | K | 0.34418 | 0.34423 | 0.995157 | 0.995155 |
| 500 hPa geopotential | m²/s² | 23.57518 | 23.56355 | 0.999599 | 0.999600 |
| 10 m eastward wind | m/s | 0.48991 | 0.48988 | 0.992288 | 0.992289 |
| 10 m northward wind | m/s | 0.49888 | 0.49847 | 0.992746 | 0.992758 |
| Mean sea-level pressure | Pa | 33.21294 | 33.20532 | 0.998976 | 0.998977 |
| 6 h precipitation | mm | 1.07179 | 1.07178 | 0.883607 | 0.883641 |

## 2021-2022 test

| Variable | Units | Baseline RMSE | Advection RMSE | Baseline ACC | Advection ACC |
|---|---|---:|---:|---:|---:|
| 2 m temperature | K | 0.59500 | 0.59485 | 0.974273 | 0.974285 |
| 850 hPa temperature | K | 0.33982 | 0.33987 | 0.995269 | 0.995269 |
| 500 hPa geopotential | m²/s² | 22.79157 | 22.78831 | 0.999600 | 0.999601 |
| 10 m eastward wind | m/s | 0.48596 | 0.48587 | 0.992172 | 0.992175 |
| 10 m northward wind | m/s | 0.49700 | 0.49655 | 0.992437 | 0.992451 |
| Mean sea-level pressure | Pa | 32.44268 | 32.45245 | 0.998948 | 0.998948 |
| 6 h precipitation | mm | 1.09497 | 1.09506 | 0.881912 | 0.881923 |

## Scope and audit

RMSE covers all 37 pressure levels. ACC uses the published 1990–2019 ERA5 climatology at its 13 supported pressure levels, plus the five surface outputs. No climatology from validation/test years is used. All aggregate results were recomputed independently from saved per-date metric files; all 36 dates were present for both models and all reported metrics were finite.

The models have small, variable-dependent differences. This short-horizon monthly sample does not establish an overall advection advantage or multi-day skill. No statistical significance claim is made.

[All variables and levels (CSV)](results/evaluation/scores.csv) · [Frozen protocol](results/evaluation/protocol.json) · [Methodology](evaluation-methodology.md)
