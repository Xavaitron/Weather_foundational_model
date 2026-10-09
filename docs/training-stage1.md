# Normal GraphCast 1° fine-tuning — stage 1 completed

Both the baseline and learned-advection models completed **1,000 fine-tuning updates** on ERA5 windows from 2016–2019. Both server runs report `requested_updates_complete`. Their best validation checkpoints are from update 1,000.

| Setting | Both runs |
|---|---|
| Architecture | Normal pretrained GraphCast, full mesh, 37 pressure levels |
| Forecast grid | 1°, 181 × 360 points |
| Initial checkpoint | ERA5 1979–2017, originally 0.25° |
| Fine-tuning years | 2016–2019 |
| Checkpoint selection | Four fixed 2020 validation windows, checked every 50 updates |
| Stage length | 1,000 updates, windows sampled with replacement using seed 0 |
| Batch / horizon | One global example / one six-hour forecast |
| Optimizer | AdamW, learning rate 1e-5, betas 0.9/0.95, matrix weight decay 0.1, gradient clipping at norm 32 |
| Memory settings | BF16 computation, float32 parameters/moments, activation checkpointing and buffer reuse |
| Advection adapter | 16 latent modes, 1° helper grid |

| Normalized validation loss | Baseline | Advection |
|---|---:|---:|
| Before fine-tuning | 14.84619141 | 14.84814453 |
| Best / final update 1,000 | 0.74183655 | 0.74172211 |

These normalized losses are training diagnostics, not physical-unit RMSE or ACC. The two models are essentially tied on the four checkpoint-selection windows. The 1,000-update stage is not a complete pass through every training window or a claim of convergence. The original checkpoint's pretraining overlaps the 2016–2017 fine-tuning years.

The training jobs used separate A6000s: baseline on GPU 2, advection on GPU 6. Both jobs have finished. Checkpoints and resumable optimizer state remain under `/home/anwar/Weather_foundational_model_finetune/runs/full-1deg-stage1-20260927/{baseline,advection}/`. Source revision at training launch was `b6fb717`.

The user subsequently authorized held-out evaluation. A separate frozen benchmark now covers twelve monthly dates in each of 2020, 2021 and 2022, with physical-unit RMSE and climatological ACC for both checkpoints. See [evaluation methodology](evaluation-methodology.md) and [completed results](evaluation-results.md).

[Reproduction guide](finetuning.md).
