"""Fine-tune official GraphCast_small weights on locally staged ERA5 data."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from infer_graphcast import REVISION, STATS, fetch, make_forward

SMALL_PARAMS = (
    "graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = os.environ.get(
        "ERA5_DATA_DIR", os.environ.get("DATA_DIR", "data/era5_graphcast_small_1deg"))
    parser.add_argument("--data-dir", type=Path, default=Path(default_data_dir))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/graphcast"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/finetuning/graphcast_small_finetuned"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-steps", type=int, default=2,
                        help="Autoregressive six-hour targets per training window (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _fetch_assets(cache_dir):
    assets = {}
    names = [SMALL_PARAMS, *(f"graphcast/stats/{name}.nc" for name in STATS)]
    paths = {}
    for name in names:
        paths[name], metadata = fetch(name, cache_dir)
        assets[name] = {key: metadata[key] for key in ("generation", "size", "md5Hash")}
    return paths, assets


def _load_window(store, start, frame_count):
    import xarray as xr

    absolute = np.asarray(
        store.datetime.isel(time=slice(start, start + frame_count)).values
    ).astype("datetime64[s]")
    window = store.isel(time=slice(start, start + frame_count)).expand_dims(batch=[0]).load()
    if absolute.size != frame_count or not np.all(np.diff(absolute) == np.timedelta64(6, "h")):
        raise ValueError("Training window is incomplete or not on a six-hour grid.")
    elapsed = absolute - absolute[0]
    window = window.assign_coords(time=("time", elapsed))
    window = window.assign_coords(datetime=(("batch", "time"), absolute[None, :]))
    return window


def run(args):
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.target_steps) < 1:
        raise ValueError("Epochs, steps per epoch, batch size, and target steps must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")
    for split_name in ("train", "validation", "test"):
        if not (args.data_dir / f"{split_name}.zarr").is_dir():
            raise FileNotFoundError(f"Missing staged split: {args.data_dir / f'{split_name}.zarr'}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise ValueError(f"Output directory is nonempty; pass --overwrite to reuse it: {args.output_dir}")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    import jax
    import jax.numpy as jnp
    import optax
    import pandas as pd
    import xarray as xr
    from graphcast import checkpoint, data_utils, graphcast, losses

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expose exactly one GPU to JAX. Found: {devices}")
    print(f"JAX {jax.__version__}: {devices[0]} ({devices[0].device_kind})", flush=True)

    paths, assets = _fetch_assets(args.cache_dir)
    with paths[SMALL_PARAMS].open("rb") as stream:
        pretrained = checkpoint.load(stream, graphcast.CheckPoint)
    if pretrained.model_config.resolution != 1.0 or len(pretrained.task_config.pressure_levels) != 13:
        raise ValueError("Expected the official 1-degree, 13-level GraphCast_small checkpoint.")
    stats = {name: xr.load_dataset(paths[f"graphcast/stats/{name}.nc"]) for name in STATS}
    model_config, task_config = pretrained.model_config, pretrained.task_config
    forward = make_forward(model_config, task_config, stats, gradient_checkpointing=True)
    apply = forward.apply
    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(pretrained.params)
    params = pretrained.params
    # The pinned GraphCast CheckPoint stores parameters and configuration only.
    # The Haiku transform is still stateful at the API boundary, but GraphCast
    # itself has no persistent Haiku state, so start each run with an empty tree.
    state = {}
    rng = jax.random.PRNGKey(args.seed)
    input_duration = pd.Timedelta(model_config.input_duration)
    six_hours = pd.Timedelta("6h")
    if input_duration % six_hours:
        raise ValueError(f"Model input duration is not a multiple of six hours: {input_duration}")
    frame_count = int(input_duration / six_hours) + args.target_steps
    train_store = xr.open_zarr(args.data_dir / "train.zarr", chunks=None)
    available_starts = train_store.sizes["time"] - frame_count + 1
    if available_starts < 1:
        raise ValueError("Training split is too short for the model input window and target step.")
    per_variable_weights = {name: 1.0 for name in task_config.target_variables}

    def loss_fn(current_params, current_state, step_rng, inputs, targets, forcings):
        predictions, updated_state = apply(
            current_params, current_state, step_rng,
            inputs, targets * np.nan, forcings)
        batch_loss, _ = losses.weighted_mse_per_level(
            predictions, targets, per_variable_weights)
        return jnp.mean(batch_loss.data), updated_state

    @jax.jit
    def train_step(current_params, current_state, current_opt_state, step_rng,
                   inputs, targets, forcings):
        (loss, updated_state), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(
                current_params, current_state, step_rng, inputs, targets, forcings)
        updates, updated_opt_state = optimizer.update(
            gradients, current_opt_state, current_params)
        updated_params = optax.apply_updates(current_params, updates)
        return updated_params, updated_state, updated_opt_state, loss

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "training.json"
    manifest = {
        "status": "running",
        "upstream_commit": REVISION,
        "pretrained_checkpoint": pretrained.description,
        "assets": assets,
        "model_resolution_degrees": 1.0,
        "pressure_levels_hpa": list(task_config.pressure_levels),
        "train_period": "2016-01-01 through 2019-12-31",
        "validation_period": "2020-01-01 through 2020-12-31 (not used for optimization)",
        "test_period": "2021-01-01 through 2022-12-31 (not used for optimization)",
        "objective": "GraphCast weighted latitude- and pressure-level MSE over autoregressive six-hour targets",
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "target_steps": args.target_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(devices[0]),
        "history": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    step_index = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for _ in range(args.steps_per_epoch):
            rng, step_rng = jax.random.split(rng)
            starts = np.random.default_rng(args.seed + step_index).integers(
                0, available_starts, size=args.batch_size)
            batch = xr.concat(
                [_load_window(train_store, int(start), frame_count) for start in starts],
                dim="batch")
            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                batch,
                target_lead_times=slice("6h", f"{args.target_steps * 6}h"),
                **dataclasses.asdict(task_config),
            )
            params, state, opt_state, loss = train_step(
                params, state, opt_state, step_rng, inputs, targets, forcings)
            loss_value = float(loss)
            if not np.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite training loss at step {step_index + 1}.")
            epoch_losses.append(loss_value)
            step_index += 1
            if step_index == 1 or step_index % 10 == 0:
                print(f"epoch={epoch} step={step_index} loss={loss_value:.6g}", flush=True)
        record = {"epoch": epoch, "mean_training_loss": float(np.mean(epoch_losses))}
        manifest["history"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}: mean loss={record['mean_training_loss']:.6g}", flush=True)

    trained = dataclasses.replace(pretrained, params=params)
    checkpoint_path = args.output_dir / "graphcast_small_finetuned.npz"
    with checkpoint_path.open("wb") as stream:
        checkpoint.dump(stream, trained)
    manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                    checkpoint=str(checkpoint_path))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Fine-tuned checkpoint: {checkpoint_path.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (OSError, RuntimeError, ValueError, FloatingPointError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
