# Method mapping and implementation boundaries

All variants keep the official Aurora encoder, backbone, decoder and variable preprocessing. The baseline fine-tunes these directly. Each transport variant adds a small current-state feature encoder and a prediction of horizontal motion. The current-state encoder receives normalized dynamic variables only; a common convolution/downsample/upsample motion network sees previous/current features and projected Aurora forecast context.

Only the additional current-state branch clips normalized input features to [-20,20] to bound rare spikes. The original Aurora path retains its own preprocessing. This additional clipping is an engineering choice shared by all four adapters, not a claim from the source papers.

For selected features `c`, backbone output `b`, learned projection `P`, lift `U`, and a trainable small gain `g`, fusion is:

```text
candidate = transport(current_features, estimated_motion) + optional_local_update
updated_backbone = b + tanh(g) * U(candidate - P(b))
forecast = original_Aurora_decoder(updated_backbone)
```

The initial gain is 0.01 so motion receives gradients immediately. Setting the gain to zero exactly restores the original decoder path; this is tested. Unlike the earlier research proposal's orthonormal-subspace formulation, P and U are independently learned here. The formula is a low-rank residual fusion and has no exact orthogonal-complement preservation guarantee.

The content slots are learned features, not physical pressure surfaces. Every slot has four flow groups by default. Velocities are bounded to ±120 m/s in each local tangent component, converted to spherical displacements using the actual forecast interval (6 h for cyclone, 12 h for pollution). A frozen-velocity spherical exponential map supplies source/destination coordinates. Latitude boundaries use scalar reflection plus a half-longitude roll; longitude is periodic. Vector meteorological inputs are encoded as features; latent channels are transported as scalars. No physical moisture/energy conservation claim follows from this.

| Approach | Implemented mechanism | Changes from the paper |
|---|---|---|
| Standard | Official Aurora forecast, normalized variable-weighted MAE, autoregressive windows | Coarsened 1.5-degree inputs and the explicitly configured train/validation/test years |
| PARADIS-inspired | Bicubic backward transport of grouped compressed features; separate depthwise mixing and pointwise reaction/source updates | Aurora supplies the large-scale model and decoder; transport runs on its patch lattice rather than PARADIS's native-grid prognostic state; small learned mixers rather than the complete PARADIS network |
| NowcastNet-inspired | Nearest-neighbour backward transport plus additive source, and a second bilinear candidate decoded with an auxiliary forecast loss | Transports learned atmospheric features; no radar-specific weighting, adversarial generator or precipitation-specific accumulation loss; the bilinear auxiliary is supervised through Aurora's decoder |
| ADRNet-inspired | Forward area-weighted bilinear push/splat, backward-Euler grid diffusion, local reaction/source | Chooses the paper's conservative push formulation by default; the paper's main implementation uses the backward color alternative; `adr_transport="pull"` selects that alternative. Uses spherical transport with Neumann-latitude/periodic-longitude grid diffusion |
| de Bezenac-inspired | A learned motion field and differentiable Gaussian sampling around each departure point, with learned positive sigma | Local 7×7 truncated, normalized Gaussian in latent-cell coordinates; spherical departure calculation; two-frame Aurora history instead of the original SST history; common MAE by default rather than changing the objective to Charbonnier |

ADRNet's implicit diffusion uses an orthonormal DCT-II in latitude and a Fourier transform in longitude to invert the five-point grid stencil. Its coefficient is positive. It is a grid regularizer, not the physical spherical diffusion equation. The push alone preserves the area-weighted latent scalar integral by construction, including polar/dateline transport. Subsequent diffusion/reaction/fusion need not preserve that integral. CUDA scatter addition can be nondeterministic.

NowcastNet's nearest branch gives zero coordinate derivatives almost everywhere. It is deliberately not replaced by an undocumented straight-through estimator: training decodes a second bilinear branch and adds its forecast loss, allowing motion to learn. Evaluation decodes only the nearest branch. This has extra training cost, recorded in timing. The existing multistep training graph is retained unless the common `detach_every` setting explicitly truncates it.

The de Bezenac kernel uses a Gaussian rather than a bilinear grid sampler. Sigma stays positive and both its value and the motion field receive gradients. Increasing its support or defining its width in physical distance is a further experiment; neither is implied by this prototype.

Shared motion regularization penalizes magnitude and local roughness. It does not enforce zero divergence. `loss="charbonnier"` enables a robust alternative objective, but change it for the standard baseline too if comparing architectures at matched loss. Fine-tuning the models in these papers on their original datasets, transferring their non-Aurora checkpoints, GraphCast/GenCast integration and ensemble verification are outside this Aurora experiment implementation.

## Sources

- [Aurora paper](https://arxiv.org/html/2405.13063v3) and [pinned official implementation](https://github.com/microsoft/aurora/tree/7179f501b9f1a66e0444b782ecd06aa07062c8bc).
- [Aurora 1.5: evaluation definitions, including cyclone track verification](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf).
- [PARADIS, revised May 2026 paper](https://arxiv.org/html/2601.21151v2).
- [NowcastNet, original paper](https://www.nature.com/articles/s41586-023-06184-4), [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10356617/).
- [ADRNet paper](https://arxiv.org/html/2406.19253), [authors' implementation](https://github.com/Siddharth-Rout/deepADRnet).
- [de Bezenac et al., Deep Learning for Physical Processes](https://arxiv.org/html/1711.07970).
- [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive).

The adapter implementation is written independently from the mathematical descriptions; it does not vendor ADRNet's GPL implementation. The upstream Aurora package and downloaded checkpoints retain their own licenses/conditions.
