# Aurora air pollution results

The saved January 2003 EAC4 baseline completed 200 optimizer updates. The test evaluation used two overlapping initializations and rolled out at 12-hour intervals to 120 hours. RMSE is area-weighted and reported for PM2.5 below in µg/m³. Fixed persistence retains the latest observation. Diurnal persistence repeats the corresponding time of day. The India result uses the 6–38° N, 68–98° E bounding box.

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

India PM2.5 trails both persistence controls at each lead shown. Global PM2.5 beats both controls at 120 hours on these two cases. Negative forecasts are included as emitted. The underlying reference is reanalysis. The six variants completed real-data integration checks, and a held-out advection comparison remains pending.

- [Full recorded pilot results and historical download status](../docs/RESULTS_2026-11-09.txt)
- [Verification record](../validation/VALIDATION.md) and [training protocol](../docs/TRAINING_RUN.md)
