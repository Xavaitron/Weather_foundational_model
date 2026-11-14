# Normal GraphCast at 1° — verified engineering run

The normal pretrained GraphCast successfully completed training updates at 1°, both with and without the learned advection adapter. GraphCast-small is not used for this configuration.

| Configuration | GPU | Compiler memory estimate | Completed updates |
|---|---:|---:|---:|
| Normal GraphCast baseline | 2 | 10.77 GiB | 2 |
| Normal GraphCast + advection | 6 | 10.98 GiB | 2 |

The runs were launched concurrently, with a separate model on each GPU. Each configuration fits one A6000; no model sharding is needed for this tested setting. These are compiler estimates, not measured peaks, and apply to batch size one with a single six-hour forecast step. Longer rollouts or larger batches need another memory check.

The normal checkpoint is `GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz`. Its full mesh, 512 latent channels, 16 message-passing steps and all 37 pressure levels are retained. Only the input/output grid changes to 181 × 360 points. Computation uses BF16, activation checkpointing and parameter/optimizer buffer reuse; master parameters and optimizer moments remain float32.

Both runs used the same real 2016-01-01 window, seed 0 and learning rate 1e-5. Both reported normalized training loss 15.40039 on the first update and 11.49121 on the second, with finite gradients. Gradient clipping is enabled at norm 32. These are two updates on a repeated engineering window, not four-year training or evidence of forecast skill. Moving a checkpoint from its native 0.25° grid to 1° changes graph aggregation statistics, so accuracy must be established on 2020 validation.

Reloaded checkpoints contain finite weights, with 258 pretrained parameter tensors changed in each run. The baseline has 36,348,131 active parameters; the adapter has 36,380,947 and all three adapter modules, including a nonzero lift norm of 0.00155505.

Server checkpoints:

- `/home/anwar/Weather_foundational_model_finetune/runs/engineering-full-1deg-baseline/latest.npz`
- `/home/anwar/Weather_foundational_model_finetune/runs/engineering-full-1deg-advection/latest.npz`

The active resolution is now 1° using normal GraphCast. Trainer and profiler defaults and the reproduction guide have been updated. The year split remains training 2016–2019, validation 2020, testing 2021–2022. Full-data training, validation and test evaluation have not been run. All jobs for this check have finished.

[Run manifests, metrics, checkpoint audit and logs](full-model-1deg-evidence.tar.gz) preserve the evidence. [Reproduction guide](finetuning-guide.md) gives the full-model training command with `--resolution 1.0`.
