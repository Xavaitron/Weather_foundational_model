# Aurora cyclone tracking

## Overview

This workflow fine-tunes the original Aurora checkpoint at 1.5° using WeatherBench 2 HRES-T0 fields. It compares standard fine-tuning with PARADIS-, NowcastNet-, ADRNet-, and de Bézenac-inspired transport adapters. Cyclone tracks are verified against NOAA IBTrACS.

## Setup and run

From the repository root:

```bash
cd aurora-cyclone-tracking
```

Follow the [command guide](WORKFLOW.md) for the Python environment, dataset preparation, training, cyclone evaluation, and result export. The scripts use local `data/` and `runs/` paths by default.

## Results

Five variants completed matched fine-tuning and held-out 2022 evaluation. The [cyclone results](cyclone_report_20261106/Aurora_Cyclone_Results.md) include track MAE, RMSE, tracker failures, and plots. See [methods](METHODS.md) and [validation](VALIDATION.md) for implementation and checks.

## Directory layout

| Path | Contents |
|---|---|
| `finetune_*.py` | One training entry point per approach |
| `configs/` | Dataset, model, and rollout settings |
| `advection_ft/` | Shared training, transport, and tracking code |
| `tests/` | Cyclone workflow tests |
| `cyclone_report_20261106/` | Result tables and figures |
