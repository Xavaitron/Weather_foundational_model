# GraphCast weather forecasting

## Theory

GraphCast encodes two consecutive weather states from the latitude–longitude grid onto a multiresolution graph mesh, updates mesh features with message passing, and decodes the next six-hour weather state back to the grid. The PARADIS-inspired adapter learns a displacement in feature space and samples features at displaced positions. Its correction lets the experiment test whether explicitly moving features adds value beyond direct fine-tuning.

The main comparison keeps the normal GraphCast mesh and 37 pressure levels and uses ERA5 on a 1° grid. It trains on 2016–2019, validates on 2020 and tests on 2021–2022. The 0.25° pretrained inference example and the 13-level GraphCast-small workflow have separate setups. See the [model reference](docs/graphcast_model_reference.md) and [fine-tuning protocol](docs/finetuning.md).

## Steps to run

1. From the repository root, run `cd graphcast-weather`. Use Python 3.11 or 3.12 and activate the environment for the selected workflow.
2. For the native-resolution inference smoke test, run the following after replacing the example GPU index with your assigned index:

   ```bash
   bash scripts/inference/setup_inference.sh
   export GRAPHCAST_GPU=2
   bash scripts/inference/run_inference.sh --check-device
   bash scripts/inference/run_inference.sh --steps 1
   ```

3. For the 1° comparison, install the full-model environment and prepare the pretrained assets as shown in the [fine-tuning protocol](docs/finetuning.md). Set `BASELINE_GPU` and `ADVECTION_GPU` to two assigned device indices and run `bash scripts/run_paired_1deg.sh`. The commands for individual runs are in the same guide.
4. Freeze the paired evaluation protocol and run it using `python -m finetuning.paired_evaluate`. The [evaluation guide](docs/evaluation-methodology.md) explains the fixed cases and aggregation. See the [command guide](docs/workflow.md) for GraphCast-small and inference options.

## Results

See [results/README.md](results/README.md) for the paired benchmark, adapter diagnostics, and saved inference examples.

## Directory layout

- `scripts/` contains setup, inference, and training commands.
- `finetuning/` contains the model integration, transport, and evaluation code.
- `tests/` contains workflow and numerical checks.
- `docs/` contains protocols and benchmark artifacts. `results/` is the reader's index.
