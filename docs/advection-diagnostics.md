# Displacement and direct fine-tuning audit

Six forecast cases were compared on the same 12 monthly 2020 validation dates, at 1° and a six-hour horizon. Latent fields were measured on January, April, July and October dates. No training updates were run and no new test-year results were used for these diagnostics.

## Displacement before and after

- Mean latent departure distance: **215.39 → 263.08 km equivalent** (uniform mean over mesh nodes, 16 modes and four validation dates).
- The east/north displacement vectors changed by **149.82%** in relative L2 norm, averaged over dates. This includes both backbone-feature changes and displacement-network learning.
- Holding the final backbone features fixed and swapping only initial versus trained displacement weights changes the vectors by **6.64%** in relative L2 norm.
- Initial displacements are random and nonzero. The initial output projection is zero, so they initially add no latent correction.
- The trained correction RMS is **0.13411%** of backbone feature RMS.
- After materializing the actual BF16 decoder-input tensors, **41.917%** of mesh-feature components change. This is the fraction of components whose represented value changes, not the fraction of correction energy retained.

![Displacement maps](../runs/advection-diagnostics-20260928/displacement-maps.png)

## Direct fine-tuning and controlled forecast comparisons

Direct fine-tuning changed 36,346,030 of 36,348,131 active parameter values (1.826% relative L2 change). The two trained backbones differ by 0.0962% in relative L2 norm.

### 2 m temperature

| Case | RMSE (K) | ACC |
|---|---:|---:|
| Pretrained at 1° | 2.239944 | 0.6790606 |
| Direct fine-tuning | 0.583608 | 0.9745794 |
| Initial adapter replay | 2.239890 | 0.6790693 |
| Trained advection | 0.583768 | 0.9745622 |
| Trained advection, zero correction | 0.583783 | 0.9745618 |
| Trained advection, initial displacement weights | 0.583815 | 0.9745591 |

### 850 hPa temperature

| Case | RMSE (K) | ACC |
|---|---:|---:|
| Pretrained at 1° | 2.518006 | 0.7050126 |
| Direct fine-tuning | 0.339805 | 0.9950519 |
| Initial adapter replay | 2.517938 | 0.7050297 |
| Trained advection | 0.339822 | 0.9950517 |
| Trained advection, zero correction | 0.339779 | 0.9950531 |
| Trained advection, initial displacement weights | 0.339808 | 0.9950522 |

### 500 hPa geopotential

| Case | RMSE (m²/s²) | ACC |
|---|---:|---:|
| Pretrained at 1° | 247.263596 | 0.9543545 |
| Direct fine-tuning | 22.655146 | 0.9996292 |
| Initial adapter replay | 247.239898 | 0.9543639 |
| Trained advection | 22.625121 | 0.9996302 |
| Trained advection, zero correction | 22.626090 | 0.9996301 |
| Trained advection, initial displacement weights | 22.622214 | 0.9996303 |

The adapter-off comparison retains the trained advection backbone and zeros only the final lift matrix. The reset-displacement comparison retains that backbone, learned projection and learned lift, but restores the initial displacement linear weights/biases. These are evaluation-only ablations, not additional trained models.

![Validation learning curves](../runs/advection-diagnostics-20260928/learning-curves.png)

## Repeatability check

Eight repetitions per case used identical 15 January 2020 inputs, weights and PRNG key. The current GPU/BF16 execution shows numerical variation between repeats. This does not establish BF16 casting alone as the cause; rounding itself is deterministic.

| Case | Mean 2 m temperature RMSE (K) | Repeat standard deviation (K) | Range (K) |
|---|---:|---:|---:|
| baseline | 0.62690550 | 0.00005672 | 0.62681256–0.62700354 |
| advection | 0.62722273 | 0.00006873 | 0.62714325–0.62733740 |
| advection_off | 0.62724559 | 0.00007207 | 0.62712937–0.62737533 |

The adapter-on/off RMSE gap is smaller than the per-date repeat variation measured here. The sample does not establish a meaningful incremental advection benefit. The direct-fine-tuning improvement is much larger. Recomputed baseline/advection scores differ slightly from the earlier benchmark, so the table above is the matched diagnostic rerun; it does not replace the original benchmark record.

## Reproducibility and limits

No epoch-zero adapter checkpoint was retained. Its parameters were reconstructed from the original pretrained backbone, the recorded seed and the exact training initializer; the Haiku shape and traversal order were checked. The original initial validation loss was 14.848144531; the replay produced 14.846191406. This small BF16 numerical difference is disclosed; the replay is not a saved epoch-zero artifact.

Only initial and final adapter parameters are available, so the plots do not establish the displacement trajectory at intermediate updates. The learning curves use the actual saved validation logs at every 50 updates. Latent capture calls the same normalized single-step network inside the autoregressive wrapper; all physical forecast comparisons use the unchanged production rollout. Checkpoint hashes were checked against the frozen evaluation.

[All validation scores](../runs/advection-diagnostics-20260928/scores.csv) · [Displacement statistics](../runs/advection-diagnostics-20260928/displacement.json) · [Parameter changes](../runs/advection-diagnostics-20260928/parameter_changes.json) · [Forecast differences](../runs/advection-diagnostics-20260928/prediction_effects.json)
