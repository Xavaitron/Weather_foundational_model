# Aurora air pollution

## Theory

The model extends the released 12-hour Aurora weather checkpoint with pollutant inputs and outputs. It reads two recent global states on a 1.5° grid and forecasts the next state. The learned branches use past pollutant fields and wind to predict a correction after Aurora decodes the forecast. They implement NowcastNet-style motion and source evolution, ADRNet-style latent dynamics, and de Bézenac-style Gaussian transport. Two fixed wind-transport controls and a direct fine-tuning baseline use the same backbone.

The loss averages area-weighted MAE across weather and pollution channels after scaling each channel by its training-period persistence MAE. The planned full comparison uses EAC4 reanalysis followed by CAMS forecast lead zero. See the [methods](docs/METHODS.md) and [data protocol](docs/DATA_PROTOCOL.md).

## Steps to run

1. From the repository root, run `cd aurora-air-pollution`. Activate a Python 3.12 environment with the project's PyTorch build.
2. Install the project, run its software tests, and inspect the training plan.

   ```bash
   python -m pip install -e '.[dev,download]'
   python -m pytest -q
   python scripts/run_training_pipeline.py --plan
   ```

3. On Linux, configure ADS, fetch the assets, and build the required stores and training-only statistics using the [GPU setup guide](docs/GPU_UPLOAD.md). The [training record](docs/TRAINING_RUN.md) describes the starter and full-data stages.
4. With the January store and statistics in place, run the starter and evaluate its selected checkpoint.

   ```bash
   python -m airadv.train --config configs/starter_eac4_jan2003.json
   python -m airadv.evaluate --checkpoint runs/starter_eac4_jan2003_baseline_seed42/best.pt \
     --output runs/starter_eac4_jan2003_baseline_seed42/evaluation.json
   ```

The [workflow guide](docs/workflow.md) lists all model flags and the full comparison setup.

## Results

See [results/README.md](results/README.md) for PM2.5 errors, both persistence controls, and the status of the advection comparison.

## Directory layout

- `src/airadv/` contains the Aurora integration, transport, training, and evaluation.
- `scripts/` contains data preparation, launch, and comparison commands.
- `configs/` contains the starter, EAC4, and CAMS settings.
- `tests/` and `validation/` contain tests and archived checks.
