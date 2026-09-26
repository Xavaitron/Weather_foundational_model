"""Compare pretrained and fine-tuned GraphCast_small on a held-out ERA5 split."""
import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from finetune_graphcast import SMALL_PARAMS, _fetch_assets, _load_window
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from infer_graphcast import REVISION, make_forward

SCORED_SURFACE = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)
SCORED_ATMOSPHERIC = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-dir", type=Path, default=Path("data/era5_graphcast_small_1deg"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/graphcast"))
    parser.add_argument("--finetuned-checkpoint", type=Path,
                        default=Path("outputs/finetuning/graphcast_small_finetuned/graphcast_small_finetuned.npz"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-initializations", type=int, default=4,
                        help="Evenly sample this many starts; use 0 to evaluate every 12-hour start")
    parser.add_argument("--max-lead-hours", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _metric_pair(forecast, truth, climatology, latitude_weights):
    forecast_anomaly = forecast - climatology
    truth_anomaly = truth - climatology
    covariance = (forecast_anomaly * truth_anomaly).weighted(latitude_weights).mean(("lat", "lon"))
    forecast_variance = (forecast_anomaly ** 2).weighted(latitude_weights).mean(("lat", "lon"))
    truth_variance = (truth_anomaly ** 2).weighted(latitude_weights).mean(("lat", "lon"))
    acc = covariance / np.sqrt(forecast_variance * truth_variance)
    mse = ((forecast - truth) ** 2).weighted(latitude_weights).mean(("lat", "lon"))
    return mse, acc


def _climatology_for_times(climatology, valid_times):
    import xarray as xr

    valid_times = np.asarray(valid_times).astype("datetime64[s]")
    datetimes = xr.DataArray(valid_times, dims=("time",))
    dayofyear = datetimes.dt.dayofyear
    hour = datetimes.dt.hour
    if "dayofyear" not in climatology.dims:
        raise ValueError("Climatology store must have a dayofyear dimension.")
    selection = {"dayofyear": dayofyear}
    if "hour" in climatology.coords:
        selection["hour"] = hour
    return climatology.sel(selection)


def _load_checkpoint(path, checkpoint_module, graphcast_module):
    with path.open("rb") as stream:
        return checkpoint_module.load(stream, graphcast_module.CheckPoint)


def run(args):
    if args.num_initializations < 0:
        raise ValueError("Number of initializations must be zero (all) or positive.")
    if args.max_lead_hours < 12 or args.max_lead_hours > 240 or args.max_lead_hours % 12:
        raise ValueError("Maximum lead must be a multiple of 12 hours between 12 and 240.")
    split_path = args.data_dir / f"{args.split}.zarr"
    climatology_path = args.data_dir / "climatology.zarr"
    for path in (split_path, climatology_path, args.finetuned_checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation asset is missing: {path}")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    import jax
    import pandas as pd
    import xarray as xr
    from graphcast import checkpoint, data_utils, graphcast, rollout

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expose exactly one GPU to JAX. Found: {devices}")
    print(f"JAX {jax.__version__}: {devices[0]} ({devices[0].device_kind})", flush=True)

    paths, assets = _fetch_assets(args.cache_dir)
    pretrained = _load_checkpoint(paths[SMALL_PARAMS], checkpoint, graphcast)
    finetuned = _load_checkpoint(args.finetuned_checkpoint, checkpoint, graphcast)
    if (pretrained.model_config.resolution != 1.0 or
            len(pretrained.task_config.pressure_levels) != 13 or
            pretrained.task_config != finetuned.task_config or
            pretrained.model_config != finetuned.model_config):
        raise ValueError("Fine-tuned and official checkpoints must use the same 1-degree GraphCast_small configuration.")
    stats = {name: xr.load_dataset(paths[f"graphcast/stats/{name}.nc"])
             for name in ("mean_by_level", "stddev_by_level", "diffs_stddev_by_level")}
    forward = make_forward(pretrained.model_config, pretrained.task_config, stats)
    climatology = xr.open_zarr(climatology_path, chunks=None)
    split = xr.open_zarr(split_path, chunks=None)
    duration = pd.Timedelta(pretrained.model_config.input_duration)
    six_hours = pd.Timedelta("6h")
    frame_count = int(duration / six_hours) + args.max_lead_hours // 6
    candidate_starts = np.arange(
        1, split.sizes["time"] - frame_count + 1, 2, dtype=np.int64)
    if candidate_starts.size == 0:
        raise ValueError(f"{args.split} split does not contain a full {args.max_lead_hours}-hour forecast window.")
    if args.num_initializations:
        selected = np.unique(np.linspace(
            0, candidate_starts.size - 1,
            min(args.num_initializations, candidate_starts.size), dtype=np.int64))
        candidate_starts = candidate_starts[selected]
    lead_indices = np.arange(2, args.max_lead_hours // 6 + 1, 2, dtype=np.int64)
    variable_names = list(SCORED_SURFACE + SCORED_ATMOSPHERIC)
    if not set(variable_names).issubset(pretrained.task_config.target_variables):
        raise ValueError("The GraphCast_small checkpoint does not predict all 69 benchmark targets.")

    checkpoints = {"pretrained": pretrained, "finetuned": finetuned}
    apply_functions = {
        name: functools.partial(
            jax.jit(forward.apply), params=model_checkpoint.params,
            state=model_checkpoint.state)
        for name, model_checkpoint in checkpoints.items()
    }
    sums = {
        model: {variable: {int(index * 6): {"mse_sum": None, "acc_sum": None, "count": 0}
                          for index in lead_indices} for variable in variable_names}
        for model in checkpoints
    }
    latitude_radians = np.deg2rad(split.lat.values)
    latitude_bounds = np.concatenate((
        [-np.pi / 2],
        (latitude_radians[:-1] + latitude_radians[1:]) / 2,
        [np.pi / 2],
    ))
    latitude_weights = split.lat.copy(data=np.diff(np.sin(latitude_bounds)))
    started = time.monotonic()
    for case_index, start in enumerate(candidate_starts, start=1):
        batch = _load_window(split, int(start), frame_count)
        absolute = np.asarray(batch.datetime.isel(batch=0).values).astype("datetime64[s]")
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            batch,
            target_lead_times=slice("6h", f"{args.max_lead_hours}h"),
            **dataclasses.asdict(pretrained.task_config),
        )
        valid_times = absolute[2:]
        if valid_times.size != targets.sizes["time"]:
            raise ValueError("Target times do not align with the split's absolute datetime coordinate.")
        case_climatology = _climatology_for_times(climatology, valid_times)
        score_targets = targets[variable_names]
        for model_name, model_checkpoint in checkpoints.items():
            apply = apply_functions[model_name]

            def predict(**kwargs):
                return apply(**kwargs)[0]

            print(f"{args.split} initialization {case_index}/{candidate_starts.size}: "
                  f"{np.datetime_as_string(absolute[1], unit='h')} ({model_name})", flush=True)
            generator = rollout.chunked_prediction_generator(
                predictor_fn=predict,
                rng=jax.random.PRNGKey(args.seed + case_index),
                inputs=inputs,
                targets_template=targets * np.nan,
                forcings=forcings,
                num_steps_per_chunk=1,
            )
            prediction_chunks = []
            for chunk in generator:
                prediction_chunks.append(chunk.compute())
            predictions = xr.concat(prediction_chunks, dim="time").isel(time=lead_indices - 1)
            selected_targets = score_targets.isel(time=lead_indices - 1)
            selected_climatology = case_climatology.isel(time=lead_indices - 1)
            predictions = predictions[variable_names]
            for variable in variable_names:
                for time_index, lead_hours in enumerate(lead_indices * 6):
                    mse, acc = _metric_pair(
                        predictions[variable].isel(time=time_index),
                        selected_targets[variable].isel(time=time_index),
                        selected_climatology[variable].isel(time=time_index),
                        latitude_weights,
                    )
                    entry = sums[model_name][variable][int(lead_hours)]
                    mse_value = np.asarray(mse.values, dtype=np.float64)
                    acc_value = np.asarray(acc.values, dtype=np.float64)
                    entry["mse_sum"] = mse_value if entry["mse_sum"] is None else entry["mse_sum"] + mse_value
                    entry["acc_sum"] = acc_value if entry["acc_sum"] is None else entry["acc_sum"] + acc_value
                    entry["count"] += 1

    output = args.output or Path("outputs/finetuning") / f"graphcast_{args.split}_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete",
        "upstream_commit": REVISION,
        "split": args.split,
        "initialization_count": int(candidate_starts.size),
        "initialization_sampling": "evenly sampled at 12-hour intervals; use --num-initializations 0 for all eligible starts",
        "lead_hours": (lead_indices * 6).tolist(),
        "scored_targets": variable_names,
        "metrics": {model: {} for model in checkpoints},
        "climatology": "WeatherBench 2 ERA5 1990-2017 six-hour climatology",
        "assets": assets,
        "elapsed_seconds": time.monotonic() - started,
    }
    for model in checkpoints:
        for variable in variable_names:
            result["metrics"][model][variable] = {}
            for lead_hours, entry in sums[model][variable].items():
                mean_mse = entry["mse_sum"] / entry["count"]
                mean_acc = entry["acc_sum"] / entry["count"]
                result["metrics"][model][variable][str(lead_hours)] = {
                    "rmse": np.sqrt(mean_mse).tolist(),
                    "acc": mean_acc.tolist(),
                    "initialization_count": entry["count"],
                }
    relative_rmse_changes = []
    acc_changes = []
    for variable in variable_names:
        for lead_hours in result["lead_hours"]:
            pretrained_scores = result["metrics"]["pretrained"][variable][str(lead_hours)]
            finetuned_scores = result["metrics"]["finetuned"][variable][str(lead_hours)]
            pretrained_rmse = np.asarray(pretrained_scores["rmse"], dtype=np.float64)
            finetuned_rmse = np.asarray(finetuned_scores["rmse"], dtype=np.float64)
            valid_rmse = (np.isfinite(pretrained_rmse) & np.isfinite(finetuned_rmse)
                          & (pretrained_rmse > 0))
            relative_rmse_changes.extend(
                (100 * (pretrained_rmse[valid_rmse] - finetuned_rmse[valid_rmse])
                 / pretrained_rmse[valid_rmse]).tolist())
            pretrained_acc = np.asarray(pretrained_scores["acc"], dtype=np.float64)
            finetuned_acc = np.asarray(finetuned_scores["acc"], dtype=np.float64)
            valid_acc = np.isfinite(pretrained_acc) & np.isfinite(finetuned_acc)
            acc_changes.extend((finetuned_acc[valid_acc] - pretrained_acc[valid_acc]).tolist())
    result["overall_summary"] = {
        "macro_mean_rmse_improvement_percent": float(np.mean(relative_rmse_changes)),
        "macro_mean_acc_change": float(np.mean(acc_changes)),
        "target_lead_pairs": len(relative_rmse_changes),
        "positive_rmse_improvement_percent_is_better": True,
        "positive_acc_change_is_better": True,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.split} metrics: {output.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)