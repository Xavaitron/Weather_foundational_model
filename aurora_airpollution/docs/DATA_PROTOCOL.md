# Data and experiment protocol

## Published reference and our choices

[Aurora's Nature paper](https://www.nature.com/articles/s41586-025-09005-y) uses CAMS analysis and EAC4 reanalysis for pollution. The [public earlier paper and supplement](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/Aurora__A_Foundation_Model_of_the_Atmosphere.pdf), Table 5 and section C.3, explicitly give June–November 2022 testing; the main text's May wording is less precise. That supplement is an earlier version, not the inaccessible final Nature supplement.

| Stage | Training, inclusive start / exclusive end | Validation | Test/diagnostic |
|---|---|---|---|
| EAC4 | 2003-01-01 / 2021-01-01 | 2021-01-01 / 2021-07-01 | 2021-07-01 / 2022-01-01 |
| CAMS analysis | 2017-10-01 / 2022-04-01 | 2022-04-01 / 2022-06-01 | 2022-06-01 / 2022-12-01 |

These validation holdouts are **our choices**. They shorten the training periods relative to the paper. All dates are UTC. The entire input history and all forecast targets must be within the chosen split. Missing 12-hour timestamps invalidate a window. Evaluation origins with targets beyond the test boundary are excluded. No random temporal split is used.

The EAC4 test is a stage diagnostic. The final benchmark is the chosen CAMS product, consistently used across every method (see the live availability distinction below). Fit scales independently on each stage's training interval. Stage transfer starts from the preceding stage's weights, resets the optimizer, and fits the new stage's pollution scales. The code preserves meteorological pretrained normalisation; pollution normalisation and loss scaling use training-only statistics. Dataset scaling changes at the transition are intentional fine-tuning changes.

## Data products and retrieval

**Live catalogue audit, 29 October 2026:** the current ADS form uses `data_format=netcdf_zip`; the downloader now follows it and validates requests against saved public catalogue constraints. EAC4 covers every requested field/date. The literal `type=analysis` CAMS request lacks PM1/PM2.5/PM10 for 269 requested 00UTC dates and 268 requested 12UTC dates at the beginning of the 2017–2022 interval. It must not be bulk-submitted as a complete dataset. The original `configs/analysis.json` is retained as the intended literal-analysis protocol, not as evidence of full availability.

The explicitly named `configs/cams_t0.json` uses **only forecast lead zero** for the full CAMS interval, matching the input product in Microsoft's public Aurora example. Public catalogue coverage passes for all required fields and dates. This is the practical full-period retrieval option; its provenance stays `cams_forecast_t0` throughout preprocessing, training and evaluation. It is not relabelled as a literal `type=analysis` retrieval or an exact paper reproduction. All methods in a comparison must use the same chosen product. Authenticated pilot retrieval is still required. See `validation/server/data_access.json` and its saved catalogue snapshots.

- [CAMS global forecasts/analysis](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview): request **type=analysis**, leadtime=0, at 00/12 UTC. [ECMWF documentation](https://confluence.ecmwf.int/pages/viewpage.action?navigatingVersions=true&pageId=340756552) describes analysis access through the API even when absent from the form.
- [EAC4](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4?tab=overview): select the same times and variables. Verify a one-day retrieval before requesting years of data.
- [Official Aurora CAMS example](https://microsoft.github.io/aurora/example_cams.html) uses a zero-lead forecast product as its analysis input. This package can request that explicitly as `cams_forecast_t0`, but records it separately; it is never silently substituted for `cams_analysis`.
- Static fields come from the pinned official Aurora pollution static asset. They include the fixed emissions features. Do not replace them with zeros. Those historical fixed inventories are part of the model setup, not time-varying emission observations.

Requests are saved alongside downloaded data. The downloader creates a SHA-256 for each completed file. The preprocessing manifest lists source files and the static-file digest. Public metadata were audited without credentials; actual data responses and archive layouts remain to be verified by an authenticated pilot.

## Data contract

The source of truth is `src/airadv/schema.py`. Meteorological surface fields are `2t`, `10u`, `10v`, `msl`; pollutant surface fields are PM1, PM2.5, PM10 and total columns of CO, NO, NO2, O3, SO2. Atmospheric fields are `z,u,v,t,q,co,no,no2,go3,so2` at 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa.

| Field | Stored unit | Reporting conversion |
|---|---|---|
| PM | kg m-3 | multiply by 1e9 for µg m-3 |
| Total columns | kg m-2 | multiply by 1e6 for mg m-2 |
| Pressure-level species | kg kg-1 | do not relabel as µg m-3; density is required |
| Temperature | K | unchanged |
| Wind | m s-1 | unchanged |
| Sea-level pressure | Pa | unchanged |
| Geopotential | m2 s-2 | not geopotential height |

These units follow [ECMWF's EAC4 documentation](https://confluence.ecmwf.int/pages/viewpage.action?navigatingVersions=true&pageId=222478021). Missing variables, unit mismatches, ambiguous ensembles/experiment versions, duplicate times, nonfinite training values and missing pressure levels raise errors. No pollutant is inferred from another species. In particular, NO is not NO2 or NOx, and PM is not total aerosol optical depth.

## Regridding

The 1.5° grid has latitude centres `89.25,87.75,...,-89.25` and longitude centres `0.75,2.25,...,359.25`, shape **120 × 240**. Both dimensions divide Aurora's patch sizes 4 and 3, avoiding an asymmetric dropped polar row. This is 1.5-degree spatial resolution; it does not refer to the separate Aurora 1.5 model family.

The first-order conservative remap uses longitude overlap and sine-latitude area overlap. Source gridpoint values are interpreted as means of cells inferred from coordinate midpoints. Its integral conservation is with respect to that inferred-cell convention. Continuous static features use the same remap; categorical soil type uses nearest neighbour. Native grids must cover the full globe. The implementation remaps scalar components, not a full spherical vector finite-volume system.

## Comparison rules

Use matching data, seeds 42/43/44, initial checkpoint, training steps, gradient accumulation and validation samples for all methods. Select best checkpoints by validation loss only. Evaluate identical initialisation times at 12,24,...,120 hours and save all per-variable/level metrics. `compare.py` rejects different origin lists, products or horizons.

The default evaluation covers the globe and an India bounding box (6–38°N, 68–98°E); the latter is an illustrative region, not a political-boundary mask. Report area-weighted RMSE, MAE and bias in physical units, with persistence and alternating 24-hour-cycle persistence. Aggregate errors before taking square roots. Do not average RMSEs from unequal sample sizes.

Actual pollution skill, high-pollution-event skill, confidence intervals and comparison to operational CAMS forecasts require later experiments. No CAMS forecast baseline results are included. Station observations, chemistry-aware mass budgets, exact published replay-buffer LoRA training, and GraphCast/GenCast integration are outside this first implementation.

