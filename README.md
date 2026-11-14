# Weather foundation model experiments

This repository tests advection adapters for weather forecasting, cyclone tracking, and air pollution. Each workflow has its own code, environment, and evaluation guide.

| Workflow | Data and task | Results |
|---|---|---|
| [GraphCast weather](graphcast-weather/README.md) | ERA5 weather forecasting at 1° | [RMSE and ACC](graphcast-weather/docs/evaluation-results.md) |
| [Aurora cyclone tracking](aurora-cyclone-tracking/README.md) | HRES-T0 forecasts at 1.5°, verified with IBTrACS | [Track errors and plots](aurora-cyclone-tracking/cyclone_report_20261106/Aurora_Cyclone_Results.md) |
| [Aurora air pollution](aurora-air-pollution/README.md) | CAMS EAC4 and CAMS lead-zero data at 1.5° | [January 2003 pilot](aurora-air-pollution/docs/RESULTS_2026-11-09.txt) |

## Running an experiment

Open the workflow's README and run its commands from that folder. Create a separate Python environment for each workflow. Keep downloaded data, virtual environments, and checkpoints outside Git.

## Migrating an existing checkout

The old `advection finetuning/` and `aurora_airpollution/` paths are now `aurora-cyclone-tracking/` and `aurora-air-pollution/`. GraphCast scripts and requirements moved from the repository root into `graphcast-weather/`. In an existing GPU checkout, point the new folders at your existing data, runs, and model caches, or pass their absolute paths. Git does not move ignored files when you pull. Recreate a virtual environment if its old path changed.
