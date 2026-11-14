# Weather foundation model experiments

| Experiment | Setup and commands | Results |
|---|---|---|
| **GraphCast weather forecasting** | [GraphCast guide](docs/graphcast/README.md) | [RMSE and ACC](docs/graphcast/evaluation-results.md), [advection diagnostics](docs/graphcast/advection-diagnostics.md) |
| **Aurora cyclone tracking** | [Aurora fine-tuning guide](advection%20finetuning/README.md) | [IBTrACS track errors and plots](advection%20finetuning/cyclone_report_20261106/Aurora_Cyclone_Results.md) |
| **Aurora air pollution** | [Air pollution guide](aurora_airpollution/README.md) | [January 2003 pilot](aurora_airpollution/docs/RESULTS_2026-11-09.txt) |

## Where things live

- **GraphCast** uses `scripts/` for entry points, `finetuning/` for the Python package, and `requirements-*.txt` for its environments. Notes, benchmark evidence, and the checked-in inference examples are in [docs/graphcast](docs/graphcast/).
- **Aurora cyclone tracking** lives in [advection finetuning](advection%20finetuning/), including its configs, scripts, and results. Quote the directory name when using it in a shell.
- **Aurora air pollution** lives in [aurora_airpollution](aurora_airpollution/) and uses its own Python environment.

Use each guide's working directory when running its commands. GraphCast writes new inference runs under `outputs/inference/`. Each Aurora workflow has its own `data/` and `runs/` directories.
