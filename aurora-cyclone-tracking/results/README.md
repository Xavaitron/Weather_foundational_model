# Aurora cyclone tracking results

Five models trained for 10,000 optimizer updates each (seed 42), with two autoregressive six-hour training steps. The 2022 benchmark evaluates the same initialized storms at +24 to +120 hours. Track MAE averages great-circle position errors in km. Track RMSE squares the same distances before averaging and gives large misses more weight. A failed tracker step persists the previous predicted position and remains in the error calculation.

| Global track MAE (km) | 24 h | 48 h | 72 h | 96 h | 120 h |
|---|---:|---:|---:|---:|---:|
| Standard | 209.67 | 325.86 | 468.82 | 585.35 | 720.98 |
| PARADIS | 207.99 | 319.74 | 464.01 | 569.27 | 723.54 |
| NowcastNet | 204.84 | 319.58 | 450.43 | 568.65 | 714.10 |
| ADRNet | 205.77 | 321.28 | 462.35 | 575.90 | 719.57 |
| de Bézenac | 207.93 | 321.78 | 463.30 | 579.32 | 722.02 |

NowcastNet's largest mean improvement was 3.92% at 72 h, with a nominal paired storm-bootstrap interval of −6.73% to −1.46% for the relative change. The interval is pointwise, without adjustment across methods or leads. The 120 h tracker failure rate rose from 10.77% for standard to 11.41% for NowcastNet. The four-step run and matched native-resolution evaluation are pending in the recorded results.

- [Full report with plots, RMSE, and failure rates](../cyclone_report_20261106/Aurora_Cyclone_Results.md)
- [All MAE scores](../cyclone_report_20261106/data/track_mae_km.csv), [RMSE scores](../cyclone_report_20261106/data/track_rmse_km.csv), and [paired basin comparisons](../cyclone_report_20261106/data/paired_comparisons.csv)
- [Per-variant training cost](../cyclone_report_20261106/data/training_summary.csv)
