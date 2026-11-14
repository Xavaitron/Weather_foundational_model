# Weather foundation model experiments

This repository studies whether explicit horizontal transport improves forecasts from GraphCast and Aurora. It covers three tasks with different targets and evaluation protocols. Each workflow README includes its architecture, data splits, run commands, and recorded results. The results-only READMEs collect the same evidence without the setup steps.

| Workflow | Data and target | Workflow guide | Results |
|---|---|---|---|
| GraphCast weather | 1° ERA5, six-hour weather fields | [Theory and commands](graphcast-weather/README.md) | [Weather RMSE, ACC and latent diagnostics](graphcast-weather/results/README.md) |
| Aurora cyclone tracking | 1.5° HRES-T0 weather forecasts, verified against 2022 IBTrACS storm positions | [Theory and commands](aurora-cyclone-tracking/README.md) | [Track MAE, RMSE, basin results and failure rates](aurora-cyclone-tracking/results/README.md) |
| Aurora air pollution | 1.5° CAMS EAC4 and planned CAMS forecast lead zero, twelve-hour pollutant fields | [Theory and commands](aurora-air-pollution/README.md) | [January EAC4 pilot and persistence scores](aurora-air-pollution/results/README.md) |

## Submission files

- [Project report (PDF)](report/Weather_Foundation_Model_Report.pdf)
- [Editable report LaTeX](report/latex/main.tex)
- [Overleaf project (ZIP)](report/Weather_Foundation_Model_Overleaf.zip)
- [Source code snapshot (ZIP)](report/Weather_Foundation_Model_Code.zip)

The ZIP contains the tracked files at commit e348e2a, the experiment snapshot cited in the report.

## What the experiments test

Advection changes the location of a field. A pull operator traces a destination backwards and samples nearby source values. A push operator moves source content forward and distributes it among destination cells. Mixing, diffusion, and reaction or source terms model changes beyond translation. The learned feature channels in these experiments do not have a direct interpretation as conserved physical quantities.

| Workflow | Where the transport correction enters | Verification |
|---|---|---|
| GraphCast | Mesh features after the GNN processor and before the grid decoder | Area-weighted field RMSE and anomaly correlation against ERA5 |
| Aurora cyclone tracking | Patch-grid features before Aurora's decoder | Great-circle track distance and tracker failure rate against IBTrACS |
| Aurora air pollution | Pollutant fields after Aurora's decoder | Area-weighted pollutant RMSE, MAE, bias and negative area against reanalysis, with persistence controls |

## Recorded results

- **GraphCast:** direct fine-tuning lowered two-metre temperature RMSE from 2.239944 K to 0.583608 K in the fixed 2020 diagnostic. In the separate 2021–2022 paired benchmark, two-metre temperature RMSE was 0.59500 K for direct fine-tuning and 0.59485 K with advection. The adapter ablation and repeatability checks support only a small measured effect on this six-hour benchmark. [Full GraphCast results](graphcast-weather/results/README.md)
- **Aurora cyclone tracking:** NowcastNet-inspired transport lowered 72-hour mean track error from 468.82 to 450.43 km and track RMSE from 661.80 to 632.20 km. The matched 2022 evaluation uses one training seed, with pointwise intervals and a separate count of tracker failures. [Full cyclone results](aurora-cyclone-tracking/results/README.md)
- **Aurora air pollution:** the January 2003 EAC4 baseline completed 200 updates. At +12 h, the global PM2.5 RMSE was 24.251 µg/m³ versus 17.207 for fixed persistence and 18.828 for diurnal persistence. The evaluation covers two overlapping forecast origins. All transport variants passed integration trials, and a held-out adapter comparison is pending in the recorded evidence. [Full air-pollution pilot results](aurora-air-pollution/results/README.md)

## Reproduce an experiment

Start in the corresponding workflow folder and use its Python environment. Follow the order in its README: stage data and model assets, run a short validation or smoke check, train or select a checkpoint, then evaluate on the defined held-out split. Keep each comparison on the same cases, lead times and metric implementation. Weather error, storm position error and pollutant concentration error have different units and answer different questions.

## Repository layout and existing checkouts

- `graphcast-weather/` contains the GraphCast JAX model integration, run scripts, evaluation protocol and score files.
- `aurora-cyclone-tracking/` contains the PyTorch training code under `src/`, runnable commands under `scripts/`, experiment configs and track evidence.
- `aurora-air-pollution/` contains the `airadv` package, ADS data tools, separate dataset configs and the pilot evidence.

Earlier checkouts used `advection finetuning/` and `aurora_airpollution/`, with GraphCast scripts at the repository root. Git leaves ignored data, checkpoints and virtual environments at their old paths. Point the new workflow at existing storage or use absolute paths. Recreate environments moved from their original installation paths. For an interrupted cyclone training run, resume from the original code revision because the trainer checks the source fingerprint.
