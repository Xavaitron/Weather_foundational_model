# Aurora air pollution

## Overview

This workflow adapts the released 12-hour Aurora weather checkpoint to pollutant forecasting at 1.5°. It compares standard fine-tuning, fixed wind transport controls, and learned motion branches inspired by NowcastNet, ADRNet, and de Bézenac.

## Setup and run

From the repository root:

```bash
cd aurora-air-pollution
```

Use a separate Python 3.12 environment. Follow the [command guide](docs/workflow.md) for installation and checks, and the [GPU setup guide](docs/GPU_UPLOAD.md) for data preparation and training.

## Results

The January 2003 EAC4 pilot completed 200 baseline updates and evaluated two overlapping forecast origins. The advection variants passed real-data integration checks. See the [pilot results](docs/RESULTS_2026-11-09.txt), [data protocol](docs/DATA_PROTOCOL.md), and [validation record](validation/VALIDATION.md).

## Directory layout

| Path | Contents |
|---|---|
| `scripts/` | Data retrieval, preparation, and launch tools |
| `src/airadv/` | Aurora integration, transport, training, and evaluation |
| `configs/` | EAC4 and CAMS experiment settings |
| `tests/` and `validation/` | Integration tests and saved checks |
