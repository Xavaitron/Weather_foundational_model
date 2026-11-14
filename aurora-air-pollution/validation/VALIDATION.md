> Historical record. See [last verified results, 9 November 2026](../docs/RESULTS_2026-11-09.txt) for the completed pilot and stopped historical pipeline. Server log paths below refer to the deployment; operational logs are not included in Git.

# Verification record

Verified locally on **29 October 2026**, Python 3.12.14, macOS ARM64, CPU. Exact dependency versions and check exit codes are in `environment_and_checks.json`.

**17 tests passed in 28.56 seconds**, with runtime warnings treated as errors:

```bash
python -m pytest -q -W error::RuntimeWarning
```

The suite checks:

- Identity, eastward translation and longitude seam behaviour.
- Pole reflection with the required 180-degree longitude shift.
- Finite-difference gradient checks of bilinear and Gaussian transport.
- Constant-field preservation of the Gaussian warp.
- Area-integral conservation and non-negativity of the isolated column push operator.
- Wind units/direction and finite polar trajectories.
- Conservative remapping, lazy-array execution and preserved units.
- Chronological split boundaries, gap rejection and zero-lead selection.
- Gradient propagation and optimizer updates in all three learned branches.
- NetCDF3 and HDF5/NetCDF4 ingestion, conversion, and chronological Zarr append.
- Full-grid integration of a reduced specialised `AuroraAirPollution` architecture.
- Real Aurora training code on synthetic 1.5-degree fields, activation checkpointing, checkpoint save/load, exact CPU resume equality, two-step rollout and rejection of incompatible resume settings.

`smoke_cpu_full_grid.json` records forward/backward passes for **all six variants on 120 × 240 inputs**. It uses actual Aurora code with three backbone stages, reduced width/depth and random weights: approximately 0.79 million backbone parameters, not the 1.3-billion-parameter released model. Recorded losses/timings are software-test measurements and must not be presented as pollutant forecasting performance or GPU speed estimates.

The evaluation CLI also completed two two-step synthetic rollouts and the comparison CLI generated its metrics CSV. These temporary synthetic forecasts and trained tiny checkpoints are intentionally excluded from the upload bundle. Python compilation, shell syntax validation and `pip check` passed. Data request generation and monthly-build dry planning were exercised without submitting ADS requests.

## Server validation, 29 October 2026

The isolated Ubuntu/Python 3.12 environment passed 19 tests (including two catalogue checks), followed by two new download-integrity tests. `pip check` passed. Tiny CUDA smoke checks passed for all six variants. The NVIDIA driver emits an NVML warning because another GPU is unhealthy; actual CUDA computation on the selected A6000 works.

The official 5,025,574,829-byte 12-hour checkpoint and static asset were downloaded and verified against their publisher SHA-256 hashes at pinned revisions. Every full-size variant loaded the checkpoint through the missing-key audit and completed **two AdamW updates** with finite gradients on **synthetic 120 × 240 inputs**. Peak allocated memory was **23.53–23.54 GiB**, with a cap based on current free memory and a 4 GiB reserve. Weights were discarded. These are integration/memory checks, not forecast scores. Individual records are in `server/pretrained_*.json`.

The ADS profile is active and both dataset licences are accepted. The private API key is configured only on the server. On 30 October, both data pilots passed schema, 1.5-degree grid and finite-value validation: four EAC4 frames and two CAMS forecast-lead-zero frames. CAMS preprocessing initially stopped on conflicting forecast-period dtype metadata. Keeping lead times numeric while still decoding absolute timestamps fixes ingestion; the existing zero-lead requirement remains enforced. Seven relevant regression checks passed locally and on the server. Original downloaded files were preserved.

All six full variants then completed two optimizer updates on real EAC4 data, with finite gradients. The baseline peak allocation was 25.23 GB (23.50 GiB). Updated weights were discarded; see `server/real_trials.status` and `real_*.json`. These small pilots provide no held-out skill estimate.

On 30 October, a saved-training supervisor was launched. It waits for the complete January 2003 EAC4 store, fits training-only statistics, performs preflight checks, and runs the preliminary baseline before full historical comparisons. See `docs/TRAINING_RUN.md`. Six focused tests passed in 28.56 seconds after the grouped-download and training-resource changes, including request-period boundaries and exact optimizer-boundary checkpoint resume. Four training/preparation tests had also passed after the raw-time chunking change. Runtime status and actual optimizer progress remain in the server logs; launching a supervisor is not proof that a checkpoint has been produced.

## Remaining GPU/data validation

| Check | Status |
|---|---|
| Authentic ADS downloads and complete variable availability | Both bounded pilots passed; full-period coverage still needs retrieval |
| Actual 5 GB pretrained checkpoint loading and missing-key audit | Passed for all six variants |
| Full-size CUDA/bfloat16 forward, backward and optimizer step | Synthetic and real EAC4 checks passed for all six |
| Peak memory, throughput and GPU reproducibility | Real EAC4 baseline peak measured; full-run throughput/reproducibility pending |
| Full historical dataset preparation | Sequential worker queued after January preflight |
| Fine-tuned checkpoint on real pollution data | Preliminary saved-training run queued; inspect live status |
| Held-out skill, extreme-event performance, advection improvement | Not measured |
| Exact reproductions of the three source-paper models | Not claimed |

Use `scripts/gpu_trial.py` and the upload guide for the next checks. The final data I/O explicitly selects h5netcdf for HDF5/NetCDF4 and SciPy for NetCDF3; both paths were tested. Local log copies are snapshots; server logs continue to change.

