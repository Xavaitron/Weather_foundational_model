# Research context for the first GraphCast run

Reviewed 2026-09-26. Immediate milestone: ordinary pretrained GraphCast inference. The original research brief focuses on Aurora and improved latent transport; the later meeting specifies a GraphCast experiment, so the runnable baseline here is GraphCast.

## What the papers imply for the project

| Model / method | Main computation | Relevance |
|---|---|---|
| GraphCast | Grid-to-mesh encoder, 16 message-passing processor layers, mesh-to-grid decoder; six-hour residual prediction | Establish an unchanged pretrained baseline before any transport modification. |
| Aurora | Flexible encoder/decoder around a multiscale 3D Swin Transformer backbone; pretraining on heterogeneous atmospheric datasets | The original project target; its latent grid differs from GraphCast's spherical mesh. An operator needs a model-specific integration. |
| Aurora 1.5 | Fine-tuning for expanded variables and temporal resolution, stochastic AdaLN perturbations, CRPS, and autoregressive ensemble training | Ensemble methodology and deterministic GraphCast require distinct evaluation protocols. The paper alone does not establish availability of compatible public weights. |
| Pangu-Weather | Earth-specific 3D Transformer with hierarchical temporal aggregation | A useful architectural comparison for spatial representation and temporal error accumulation. |
| ClimaX | Variable tokenization and aggregation in a Transformer foundation model | Illustrates how a model can handle different variable sets and adapt to downstream tasks. |
| Learning to Advect | Neural semi-Lagrangian transport in latent space, with separate diffusion and reaction components | Transport is a learned architectural operation requiring training, geometry, and interpolation decisions. It is not an inference flag in stock GraphCast. |

Sources: [GraphCast](https://arxiv.org/html/2212.12794v2), [Aurora](https://arxiv.org/html/2405.13063v3), [Aurora 1.5](https://www.microsoft.com/en-us/research/publication/aurora-1-5-fine-tuning-a-foundation-model-for-medium-range-ensemble-weather-prediction/), [Pangu official implementation](https://github.com/198808xc/Pangu-Weather), [ClimaX official implementation](https://github.com/microsoft/ClimaX), [Learning to Advect v3](https://arxiv.org/html/2601.21151v3).

The advection paper's latest linked version uses 0.25-degree inputs/outputs and a 0.5-degree latent processor grid. It transports learned latent modes using differentiable spherical interpolation, then applies local mixing and channel interactions. The authors assess spectral amplitude and coherence as well as forecast error. For this project, this suggests measuring whether improved fine-scale variance has correct spatial phase; sharper pictures alone do not establish more accurate weather. The paper has multiple versions, so freeze the intended version when designing the modified model. [Method and spectral evaluation](https://arxiv.org/html/2601.21151v3).

## Data choices

| Source | What it provides | What still needs preparation for GraphCast |
|---|---|---|
| CDS ERA5 pressure levels | Atmospheric variables on pressure levels | Surface variables, static fields, forcing radiation, temporal alignment, and precipitation accumulation also need to be obtained. The supplied pressure-level link alone is insufficient. |
| ARCO-ERA5 | Cloud-optimized ERA5 stores, including analysis-ready arrays on 0.25-degree grids | Select the right store, variables, levels, hours, units, and chunk strategy before making local subsets. |
| WeatherBench 2 | Documented ERA5 and forecast data plus evaluation conventions | Dataset variants differ in resolution, levels, variables, and period. Validate the chosen store against the checkpoint; a 13-level dataset does not directly satisfy the 37-level model. |
| Official GraphCast sample | Already prepared input/reference data matching the model | Best first execution check; too small and too narrow in time for the project benchmark. |

Sources: [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview), [ARCO-ERA5](https://github.com/google-research/arco-era5), [WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), [GraphCast demo](https://github.com/google-deepmind/weathernext/blob/97d1ad50b0b7af4aaed7790167dffa769bae1f2c/graphcast_demo.ipynb).

## Resolution and the requested split

ERA5's native atmospheric scale is approximately 31 km. A 0.1-degree interpolation has more grid points but does not create new resolved atmospheric information. Standard GraphCast is trained at 0.25 degrees. Running or fine-tuning a model on a 0.1-degree grid requires an explicit decision about target data, graph construction, compute, and what improvement will be measured. A global 0.1-degree latitude/longitude grid has 1801 x 3600 = 6,483,600 points, about 6.24 times the standard grid. This ratio concerns grid values, not a reliable estimate of total model runtime or memory. [ERA5 specification](https://www.ecmwf.int/en/forecasts/datasets/complete-era5-global-atmospheric-reanalysis), [GraphCast paper](https://arxiv.org/html/2212.12794v2).

The planned fine-tuning split is **2016-2019 train, 2020 validation, 2021-2022 test**. The selected standard checkpoint was pretrained through 2017; that overlaps the fine-tuning training interval, but does not overlap validation/test. Document pretraining separately so the experiment is not described as training exclusively on 2016-2019. Avoid the operational checkpoint for this split because its training extends through 2021. Use identical initializations, verification targets, forecast horizons, and preprocessing for the unchanged and modified models. Ensure each training rollout's target times remain inside its split. [Checkpoint descriptions](https://github.com/google-deepmind/weathernext/blob/main/docs/weathernext1_graph/README.md).

## What this code establishes

The runner loads the exact pretrained parameters and their original model/task configuration, applies the official casting and normalization wrappers, and uses the official autoregressive rollout. It saves physical predictions, valid times, provenance, a temperature comparison plot, and a sample RMSE. The default is one six-hour prediction; larger official samples support up to 72 hours here.

After server execution succeeds, the next research decisions are the data source for the complete split, the meaning of 0.1 degrees, the transport operator's location in the network, and the evaluation protocol. The baseline code makes none of those architectural changes yet.
