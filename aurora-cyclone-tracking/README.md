# Aurora cyclone tracking

## Theory

Aurora encodes the two most recent weather states, evolves their features with a Swin Transformer backbone, and decodes the next six-hour forecast. The transport adapters estimate motion from the encoded history and backbone context, move features across the latent grid, then add a learned correction before decoding. At 1.5°, the 120 × 240 weather grid becomes a 30 × 60 transport grid after 4 × 4 patching.

The comparison covers standard Aurora, PARADIS-inspired backward transport, NowcastNet-inspired motion and source prediction, ADRNet-inspired forward transport and diffusion, and de Bézenac-inspired Gaussian warping. The trained models use WeatherBench 2 HRES-T0 and ERA5 static fields. Training uses 2016–2020, validation uses 2021, and NOAA IBTrACS verifies initialized storms in 2022. See [methods](docs/METHODS.md) for the operators and objectives.

## Steps to run

1. From the repository root, run `cd aurora-cyclone-tracking`. Create the environment and install dependencies with the [setup instructions](docs/WORKFLOW.md#1-copy-and-set-up-on-the-ssh-gpu).
2. Select the assigned GPU by UUID and prepare the data. Check that the IBTrACS CSV contains the configured 2022 test period before evaluation.
3. Run one method or use the five-method launcher.

   ```bash
   python scripts/download_datasets.py
   python scripts/finetune_standard.py --config configs/cyclone.json
   # To launch all five methods from fresh output directories instead:
   # bash scripts/run_all.sh cyclone
   ```

4. Evaluate each selected checkpoint with `scripts/evaluate_cyclones.py`, using the same IBTrACS file and lead times. Use `scripts/compare_results.py` for paired comparisons and `scripts/export_results.py` to package the outputs. The [command guide](docs/WORKFLOW.md) has the exact flags, resume command, and output paths.

The source layout changes the trainer's resume fingerprint. For a training job started before this change, use the original checkout or source bundle to resume. Existing model checkpoints can still be passed to the evaluator.

## Results

See [results/README.md](results/README.md) for the five-way comparison, MAE, RMSE, failure rates, figures, and outstanding experiments.

## Directory layout

- `src/advection_ft/` contains the shared model, operators, trainer, and tracker.
- `advection_ft/` connects the existing standalone commands to the source package.
- `scripts/finetune_*.py` are training entry points. Other commands also live in scripts/.
- `configs/` and `tests/` contain experiment settings and checks.
- `results/` indexes the tables and figures in `cyclone_report_20261106/`.
