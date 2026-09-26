# Experiment status — 27 September 2026

The requested 0.1° experiment is **not trained**. The baseline and learned-advection implementation is present, with strict 2016–2019 / 2020 / 2021–2022 chronology. Full-size memory profiling identifies the current blocker.

## Verified results

- Read the GraphCast v2 paper and supplement and the advection/PARADIS v3 paper and appendices, and checked the relevant official implementations. Version-specific sources and design differences are in `finetuning.md`.
- Isolated the work in `/home/anwar/Weather_foundational_model_finetune`, branch `codex/finetune-0p1`; the original teammate inference checkout is preserved.
- Downloaded one real 37-level training window: 2016-01-01 06:00, 12:00 and 18:00 UTC. It is an engineering sample, not the full dataset.
- Eight tests passed in 106.51 seconds on the server. They cover chronology, precipitation accumulation, periodic 0.1° regridding, spherical interpolation/gradients, identity initialization, an optimizer update, checkpoint round-trip prediction, physical RMSE alignment, and float32 equivalence of activation checkpointing to stock GraphCast.
- The requested global grid has 6,483,600 nodes, about 6.245 times the pretrained 0.25° grid. No test-weather values have been used for training or model selection in this work.

## Memory evidence

The 0.1° profiles below are for the baseline; the 0.1° adapter has not been compiled or trained. All numbers are compiler estimates for one six-hour full-parameter update, batch size one, with BF16 activations and AdamW. They are not measured peak usage. Autotuning was disabled for the completed 0.1° profiles to avoid allocating large GPU benchmarking buffers during compilation.

| Implementation | Grid | Estimated device memory |
|---|---:|---:|
| Checkpoint message-passing blocks | 0.1° | 303.72 GiB |
| Also checkpoint graph embeddings and outputs | 0.1° | **199.38 GiB** |
| Same final checkpointing, native-grid baseline | 0.25° | 33.82 GiB |
| Native-grid advection adapter | 0.25° | 34.10 GiB |

The final 0.1° breakdown is 18,227,262,400 bytes of arguments, 436,184,028 bytes of outputs, and 195,417,022,640 temporary bytes, with no buffer aliases: **214,080,469,068 bytes total**. The profile is `work/memory-0p1-full-remat.json` on the server.

The server has two 48 GB RTX A6000s, already shared with other workloads. The current implementation runs on one device. Merely selecting two devices does not combine their memory, and their combined capacity is below this estimate anyway. A larger node would still require explicit model sharding unless a single device has sufficient memory.

This is a limitation of the current implementation, not a theoretical lower bound for GraphCast. Further work could stream/chunk grid-to-mesh and mesh-to-grid edge computations, recompute their activations, and/or offload or shard tensors. Ordinary data parallelism and gradient accumulation do not solve the memory required for one global example. Such changes need numerical-equivalence and gradient tests before long training runs.

## Engineering runs and remaining work

The 0.25° runs are explicitly native-grid smoke tests on one 2016 window. They do not replace the 0.1° research experiment, and their training losses must not be reported as validation or test skill. The baseline completed two full-parameter GPU updates and saved a checkpoint containing 36,348,131 active parameters. Reloading confirmed finite weights and changes in 258 parameter tensors. Its normalized training loss increased from 0.53912 to 0.80981; the pilot learning rate (1e-5) and schedule are not validated for training. This is evidence that execution works, not evidence of forecast improvement. The matched adapter checkpoint contains 36,380,947 finite parameters, including all three adapter modules and a nonzero lift projection. It also completed two updates, with losses 0.53924 and 0.80807. Both checkpoints are engineering artifacts only. The first losses differ by about 0.023%, consistent with BF16 compilation differences; zero initialization is a mathematical identity, not a promise of bitwise equivalence across large compiled programs.

Before the requested experiment can run: resolve single-example memory through additional implementation work or suitable compute; establish reliable access to the complete four-year training source; select the training/rollout schedule on 2020; then freeze the protocol and evaluate 2021–2022. The RMSE diagnostic runner is implemented, but full-checkpoint validation, ACC, training-only climatology and spherical spectra remain pending.

Server DNS to Google Cloud failed intermittently. The one-window download succeeded with a process-local DNS override preserving HTTPS hostname/certificate validation; no system DNS settings were changed. A durable data-access setup is still needed for unattended full-source streaming.

There are no trained 0.1° model weights, final benchmark scores, or evidence yet that this adapter improves forecasting.
