> Historical launch/recovery record. Superseded by the [9 November results](RESULTS_2026-11-09.txt): the starter finished; the full historical pipeline stopped. References below to running jobs describe earlier checks.

# Real-data training launch, 30 October 2026

## Recovery on 2 November 2026

The original starter failed before an optimizer update: PyTorch's allocator-limit API rejected the unindexed `cuda` device. The trainer now resolves it to an explicit process-local device index and selects that device before GPU checks. Two device-selection regression tests and the existing exact-resume/two-step-rollout integration test passed on the server (3 tests). An actual A6000 allocation with the memory cap also passed.

January 2003 preparation is complete (62 twelve-hour frames). The historical store contains 4,870 frames from 1 January 2003 through 31 August 2009. Retrieval stopped at September 2009 after an ADS TLS certificate-verification error. Fresh requests now pass certificate verification with both the environment and system CA bundles; verification remains enabled.

The supervisor was restarted on physical GPU 5, UUID `GPU-28c4d352-b984-ed19-6dbe-cd0a411c8ea0`, because the original GPU 2 no longer had enough free memory. The 28 GiB cap and 4 GiB reserve remain in force. It reuses completed data, restarts the historical worker under its lock, and resumes any saved training checkpoint. See the live status and training log for completed optimizer steps; a running process alone is not a completed run.

Real training subsequently completed its first three optimizer updates with finite losses and gradient norms. Recent updates took approximately 11 seconds with four microbatches per update. The first scheduled validation/checkpoint is at step 50; no completed fine-tuning result or held-out score is implied. The recovery snapshot is `validation/server/recovery_20261102.json`.

The original launch details below are retained as a historical record.

Server project: `/media/nas_mount/anwar2/aurora_airpollution_shohom_20261029/aurora_airpollution` on the supplied IIITD server. The environment, data, weights and checkpoints all reside on the NAS. The selected device is physical GPU 2, an RTX A6000, selected by UUID to avoid the server's unhealthy GPU changing ordinal mappings.

The launched supervisor is `scripts/run_training_pipeline.py`. It survives closing SSH, writes status to `runs/operations/training_pipeline.json`, and holds an exclusive lock against duplicate supervisors. Consult the live status before starting another process. The launch-time phase is `waiting_for_starter_data`; a queued supervisor alone does not mean optimizer updates have begun.

## First saved run

`configs/starter_eac4_jan2003.json` trains every parameter of the extended Aurora12hPretrained baseline on actual EAC4 data, at 120 × 240 cells and 12-hour time steps. It uses 200 optimizer updates, four microbatches per update, bfloat16 backbone computation and activation checkpointing.

| Split | Inclusive calendar dates |
|---|---|
| Training | 1–20 January 2003 |
| Validation | 21–24 January 2003 |
| Held-out test | 25–31 January 2003 |

Training statistics use only the training interval. Validation runs every 50 updates. `best.pt` and `last.pt` include optimizer and random-generator states for resuming. The test is evaluated after training and checkpoint selection. This month-long experiment is preliminary and cannot establish general seasonal performance or reproduce the paper's benchmark.

The data path is `data/processed/eac4_jan2003_1p5.zarr`. The output directory is `runs/starter_eac4_jan2003_baseline_seed42`. `run.json` records the actual model, data and memory allocation; `train.jsonl` records completed optimizer steps. A checkpoint is confirmed only after a `checkpoint_saved` log event and the corresponding `.pt` file exist.

## Following stages

After the January data checks pass, a separate worker retrieves full EAC4 (2003–2021), then CAMS forecast lead zero (October 2017–November 2022). ADS requests are sequential and grouped within a calendar month. Completed months are regridded and checked before their raw NetCDF files are removed; request JSON and hashes remain. The worker reuses January's downloaded raw data.

The training supervisor then runs `baseline`, `nowcast_evolution`, `adr_latent` and `debezenac_gaussian`, with seed 42. Each full comparison starts afresh from the same official 12-hour checkpoint: 10,000 EAC4 updates, followed by 5,000 CAMS updates initialized from that method's selected EAC4 checkpoint. January's preliminary weights are not reused in this comparison. The three advection branches are paper-inspired adaptations, not exact source-paper reproductions. The separate three-seed `run_matrix.sh` is available but is not the active supervisor's plan.

Full EAC4 and CAMS split dates are recorded in `configs/eac4.json` and `configs/cams_t0.json`. Training cannot advance to a stage until its complete data store, training statistics and preflight checks succeed. A download failure stops dependent training with an explicit failure status. Dataset availability and archive queue time determine when these stages can begin.

## Inspect the running jobs

From the server project directory:

```bash
cat runs/operations/training_pipeline.json
cat runs/operations/historical_data.json
tail -n 5 runs/starter_eac4_jan2003_baseline_seed42/train.jsonl
tail -n 10 runs/operations/starter_eac4_jan2003_baseline_seed42.log
```

Files for later phases appear only after those phases start. `starter_data.log` contains the first month's ADS/preparation output; `historical_supervisor.log` and `*_historical_data.log` contain the historical worker's output. On failure, inspect the indicated log before restarting. Run `python scripts/run_training_pipeline.py --plan` to inspect the schedule without launching it.

GPU allocation uses the native allocator and caps this process at the smaller of 28 GiB and current free memory minus 4 GiB. Dataset building and checkpoint saves stop if shared free storage falls below 200 GiB. These are process-side limits, not a scheduler reservation; unrelated processes and other NAS users can still affect available resources.

The portable bundle excludes all datasets, weights, checkpoints and credentials. Nothing needs uploading again to the existing server.
