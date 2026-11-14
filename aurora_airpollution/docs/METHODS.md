# Methods and implementation scope

Checked 29 October 2026. The implementation was written for this package; original paper repositories are linked for attribution and comparison, not vendored.

## Paper findings

| Source | Relevant idea and available assets | Transfer decision |
|---|---|---|
| [NowcastNet, Nature 2023](https://www.nature.com/articles/s41586-023-06184-4) | Deterministic evolution conditions a stochastic generative precipitation model. The paper links code and pretrained weights through [Code Ocean](https://doi.org/10.24433/CO.0832447.v1). Capsule access was not verified here. | Transfer the motion/source evolution idea. Radar weights are not an air-pollution checkpoint. No GAN/discriminator or probabilistic reproduction is claimed. |
| [ADRNet, NeurIPS 2024](https://arxiv.org/pdf/2406.19253) | Latent features evolve through reaction, diffusion and transport. The paper discusses mass-preserving push and intensity-preserving pull; the described implementation uses pull. [Official code](https://github.com/Siddharth-Rout/deepADRnet), inspected at `c5cb62c2780b013548da21773e13517702d54f6e`, contains no checkpoint in its tree. | Implement latent ADR blocks with per-channel motion. This uses bounded explicit diffusion, a different width/depth, and no optional denoiser. It is not the released architecture verbatim. |
| [de Bézenac et al.](https://arxiv.org/pdf/1711.07970) | A convolution/deconvolution motion predictor drives a differentiable Gaussian warp of the latest field. Original experiments concern sea-surface temperature. An authoritative downloadable pretrained checkpoint was not verified. | Implement the Gaussian departure-point kernel and a compact U-Net. Two input times, wind conditioning and pollution channels differ from the paper. |

## Architecture used here

Aurora receives meteorology, pollutant histories and static fields. The main initialisation uses the released 12-hour meteorological backbone with added pollutant variables. It is **not** a from-scratch recreation of the specialised `AuroraAirPollution` architecture.

Let `F_theta(X)` be Aurora's prediction, `q` the latest pollutant state, and `s` a scale fitted only on the training period. The learned variants output:

`q_next = F_theta(X)_pollution + s * C_phi(q_history / s, wind_displacements)`.

Meteorological predictions come from Aurora. The correction operates after decoding in physical variable space. The baseline trains the same Aurora model without this extra correction. The objective is equal-channel area-weighted MAE divided by each channel's training 12-hour persistence MAE. This is a deliberate comparison objective, not the publication's exact weighting.

`nowcast_evolution`: a two-resolution U-Net predicts a per-channel displacement residual and source increment. Its correction is a gated difference between wind-plus-residual transport and the current state, plus the learned source. The initial transport gate is small; source/motion output weights start at zero.

`debezenac_gaussian`: a U-Net predicts displacement residuals. A discrete Gaussian, centred on each fractional departure point, weights nearby source cells. The 6 × 6 support is renormalised; learnable sigma is bounded to 0.2–1.0 pixels. The correction is the gated transported-minus-current difference. Sigma is a numerical smoothing parameter, not a calibrated atmospheric diffusivity.

`adr_latent`: history and wind-displacement features are lifted to 32 latent channels, passed through three reaction → diffusion → advection blocks, then projected to pollutant corrections. Reaction uses pointwise convolutions; diffusion uses a five-point stencil with coefficient below 1/4; advection uses bounded per-channel pull displacements. The final projection starts at zero. Its transport is in feature space, with no tracer-mass interpretation.

## Extra physical controls

An additional relevant paper is [ClimODE, ICLR 2024](https://arxiv.org/abs/2404.10024), with [author-maintained code](https://github.com/Aalto-QuML/ClimODE). It learns continuous-time transport with a neural ODE and provides a useful future comparison for learned global flow. Adapting it would require pollutant channels, an appropriate velocity initialisation and chemistry/source treatment. It is listed for follow-up; no ClimODE implementation or air-pollution checkpoint is claimed in this package.

`wind_pull` adds an ungated transport increment based only on current winds. `column_push` scatters **cell-area-weighted mass** for total columns and converts back to areal density. They help distinguish benefits of learned motion from a simple transport bias. These are additional original controls, not claims of reproducing another paper.

The wind trajectory uses a great-circle endpoint with locally frozen velocity for 12 hours. PM uses 10-m winds, pressure-level species use winds on their own level, and total columns use a pressure-thickness weighted wind proxy. The column proxy is not concentration-weighted species flux. This is a first-order horizontal approximation; vertical transport, emissions, chemistry and deposition are left to the learned forecast/correction.

Longitude is periodic. Scalar sampling folds across poles with a half-world longitude shift. Pole-folded CNN features are not a proof of vector rotational equivariance. Pull and Gaussian transport preserve constants, not mass under divergent flow. Only the isolated areal-density push operator has a tested area-integral conservation property. PM in kg/m³ and species mixing ratios in kg/kg must not be described as mass conserved by that test.

Raw predictions are evaluated without clipping, including their negative fraction. No positive concentration guarantee or operational air-quality validity is claimed. Improvements require real held-out data, multi-seed comparisons, and review of regional/extreme-event behaviour.

