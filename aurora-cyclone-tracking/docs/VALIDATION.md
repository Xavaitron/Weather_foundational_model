# Local validation record

Date: 29 October 2026. Platform: Windows, CPU only. Python: 3.12.14.

Test environment:

| Package | Version |
|---|---|
| torch | 2.6.0 |
| torchvision | 0.21.0 |
| microsoft-aurora | 2.0.2.dev2+g7179f501b |
| numpy | 2.3.5 |
| xarray | 2025.12.0 |
| pytest | 8.4.2 |

Checks completed:

- `python -m pytest -q tests`: **22 passed**, final run 41.83 seconds.
- `python scripts/smoke_ddp.py`: two CPU/Gloo processes completed two multistep optimizer updates, checked synchronized motion gradients, and ran validation.
- `ruff check --select F,E9`: passed.
- Python compilation: passed.
- `bash -n scripts/run_all.sh`: passed.

The tests cover each of the five approaches through the actual tiny Aurora encoder/backbone/decoder at the global 120×240 1.5-degree grid, with backward propagation and nonzero finite motion gradients. They also cover the specialized air-pollution class, two-step gradient flow with activation checkpointing, the exact zero-gain baseline path, transport identity/constant fields, seam/pole coordinates, area-integral preservation of the push operation, stable implicit grid diffusion, conservative regridding, chronology/gap filtering, spherical distance, paired storm resampling and failed-track accounting.

The end-to-end workflow check prepares a synthetic Zarr store, trains the NowcastNet-inspired variant for one optimizer step, saves and reloads the optimizer/model checkpoint, executes its resume path, evaluates a synthetic cyclone against a prepared observation table and exports the result archive.

The distributed check emitted performance warnings about unused-parameter detection and convolution gradient layout; synchronized gradients and validation still passed. This test does not benchmark GPU/NCCL performance.

No released large checkpoint was downloaded for these tests. No real HRES/ERA5/CAMS forecast, GPU fine-tuning, cyclone benchmark score, ensemble calibration experiment or claimed scientific improvement has been produced. Those runs are to be performed on the SSH GPU with the source datasets and recorded settings. The paper-to-code adaptations and evaluation differences are documented in `METHODS.md` and `README.md`.

## Automatic cyclone downloader validation (2026-10-29)

- New downloader and existing workflow tests: 7 passed (83.44 s).
- End-to-end test runs the real data-preparation and IBTrACS CLIs against small local sources, assembles two monthly shards, validates the final store, resumes without accessing remote fields, and rejects source changes.
- Live HRES-T0/ARCO metadata check passed: all required variables and 10,228 six-hour timestamps for 2016-2022.
- Read one actual HRES 2-metre temperature field and all three ERA5 static fields (land mask, surface geopotential, soil type); all samples were finite.
- Selected variables/dates before constructing Dask arrays to avoid a task graph over the entire ARCO corpus. Public GCS stores use anonymous HTTPS object reads.
- Full seven-year downloads and GPU training were not run locally. Monthly cache and final store require separate disk space. Air-pollution/CAMS data downloading is not implemented by this cyclone downloader.