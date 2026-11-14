# GraphCast results

The matched benchmark uses the same six-hour initialization dates and ERA5 targets for both models. RMSE is the square root of area-weighted mean squared error in the variable's physical units. Anomaly correlation (ACC) measures agreement after removing the climatology. Higher ACC and lower RMSE are better.

| 2 m temperature | Direct RMSE (K) | Advection RMSE (K) | Direct ACC | Advection ACC |
|---|---:|---:|---:|---:|
| 2020 validation | 0.58369 | 0.58381 | 0.974572 | 0.974559 |
| 2021–2022 test | 0.59500 | 0.59485 | 0.974273 | 0.974285 |

The held-out RMSE difference is 0.00015 K. The diagnostic ablation and repeated inference show why the measured change is insufficient to establish an adapter benefit. The benchmark samples 12 monthly initializations per year and covers one six-hour lead.

- [All yearly and combined tables](../docs/evaluation-results.md) and [full scores CSV](../docs/results/evaluation/scores.csv)
- [Displacement, correction magnitude, and repeatability](../docs/advection-diagnostics.md)
- [Six-hour inference example](../docs/results/inference/graphcast_20220101_0600Z_1step/) and [24-hour example](../docs/results/inference/graphcast_20220101_0600Z_4step/)
