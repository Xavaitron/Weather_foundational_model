# GraphCast weather forecasting

## Overview

This workflow compares direct fine-tuning with a PARADIS-inspired latent advection adapter on 1° ERA5 weather forecasts. It also includes native 0.25° pretrained inference and a separate GraphCast-small workflow.

## Setup and run

From the repository root:

```bash
cd graphcast-weather
```

Use Python 3.11 or 3.12. Choose the environment for your workflow and follow the [command guide](docs/workflow.md) for inference, training, and evaluation.

## Results

The matched six-hour evaluation reports area-weighted RMSE and anomaly correlation. The measured adapter effect on held-out 2 m temperature RMSE was 0.00015 K. See the [evaluation results](docs/evaluation-results.md), [advection diagnostics](docs/advection-diagnostics.md), and [fine-tuning protocol](docs/finetuning.md).

## Directory layout

| Path | Contents |
|---|---|
| `scripts/` | Setup, inference, and training entry points |
| `finetuning/` | GraphCast training, transport, and verification code |
| `tests/` | GraphCast tests |
| `docs/` | Protocols, benchmarks, and saved inference examples |
