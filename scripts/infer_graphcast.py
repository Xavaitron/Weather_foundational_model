"""Run the original pretrained GraphCast on official ERA5 sample data.

Based on the workflow in DeepMind's graphcast_demo.ipynb. Model implementation
is supplied by the pinned upstream dependency, not copied into this project.
"""
import argparse
import base64
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

REVISION = "97d1ad50b0b7af4aaed7790167dffa769bae1f2c"
BUCKET = "dm_graphcast"
PARAMS = (
    "graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - "
    "pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
)
STATS = ("mean_by_level", "stddev_by_level", "diffs_stddev_by_level")


def sample_name(steps):
    """Select the smallest published 37-level sample covering this rollout."""
    if not 1 <= steps <= 12:
        raise ValueError("Official full-resolution samples support 1-12 steps (6-72 h).")
    available = next(n for n in (1, 4, 12) if n >= steps)
    return ("graphcast/dataset/source-era5_date-2022-01-01_res-0.25_"
            f"levels-37_steps-{available:02d}.nc")


def checksum(path):
    digest = hashlib.md5()  # Integrity comparison with public GCS metadata.
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def fetch(name, cache):
    """Cache an official object, verifying its published size and MD5."""
    path = cache / name
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = urllib.parse.quote(name, safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{encoded}"
    with urllib.request.urlopen(url, timeout=60) as response:
        meta = json.load(response)
    size, md5 = int(meta["size"]), meta["md5Hash"]
    if path.exists() and path.stat().st_size == size and checksum(path) == md5:
        print(f"Cached: {path.name}", flush=True)
        return path, meta
    print(f"Downloading {path.name} ({size / 1e9:.2f} GB)", flush=True)
    partial = path.with_suffix(path.suffix + ".part")
    media = f"{url}?alt=media&generation={meta['generation']}"
    received, last_print = 0, time.monotonic()
    with urllib.request.urlopen(media, timeout=180) as response, partial.open("wb") as out:
        while block := response.read(8 * 1024 * 1024):
            out.write(block)
            received += len(block)
            if time.monotonic() - last_print > 10:
                print(f"  {received / size:.0%}", flush=True)
                last_print = time.monotonic()
    if received != size or checksum(partial) != md5:
        raise RuntimeError(f"Download integrity check failed: {partial}. Run again to retry.")
    partial.replace(path)
    return path, meta


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of six-hour steps, 1-12 (default: 1)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New/empty directory; defaults to a timestamped outputs/ folder")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--check-device", action="store_true")
    return parser.parse_args()


def run(args):
    sample = sample_name(args.steps)
    # Set before importing JAX. The shell launcher selects one physical GPU.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    if not args.download_only:
        import jax
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise RuntimeError(f"Expose exactly one GPU using the launcher. Found: {devices}")
        print(f"JAX {jax.__version__}: {devices[0]} ({devices[0].device_kind})", flush=True)
        if args.check_device:
            import jax.numpy as jnp
            result = jax.jit(lambda x: x @ x)(jnp.ones((32, 32)))
            result.block_until_ready()
            print("GPU computation OK", flush=True)
            return

    objects = [PARAMS, sample] + [f"graphcast/stats/{s}.nc" for s in STATS]
    assets = {}
    paths = {}
    for name in objects:
        paths[name], metadata = fetch(name, args.cache_dir)
        assets[name] = {k: metadata[k] for k in ("generation", "size", "md5Hash")}
    if args.download_only:
        print("All inference assets downloaded and verified.")
        return

    import functools
    import numpy as np
    import xarray as xr
    from graphcast import checkpoint, data_utils, graphcast, rollout

    with paths[PARAMS].open("rb") as stream:
        ckpt = checkpoint.load(stream, graphcast.CheckPoint)
    if ckpt.model_config.resolution != 0.25 or len(ckpt.task_config.pressure_levels) != 37:
        raise ValueError("Checkpoint must be the original 0.25-degree, 37-level model.")
    print(ckpt.description, flush=True)
    stats = {s: xr.load_dataset(paths[f"graphcast/stats/{s}.nc"]) for s in STATS}

    # Slice BEFORE loading and extracting: upstream aligns the last timestamp
    # with the forecast horizon. Keep the first two states as fixed inputs.
    with xr.open_dataset(paths[sample]) as source:
        if source.sizes.get("time", 0) < args.steps + 2:
            raise ValueError("Sample does not contain enough timestamps.")
        batch = source.isel(time=slice(0, args.steps + 2), batch=slice(0, 1)).load()
    if (batch.sizes["lat"], batch.sizes["lon"], batch.sizes["level"]) != (721, 1440, 37):
        raise ValueError(f"Unexpected sample dimensions: {dict(batch.sizes)}")
    if not np.all(np.diff(batch.time.values) == np.timedelta64(6, "h")):
        raise ValueError("Sample must have six-hour intervals.")
    initialization = np.asarray(batch.datetime.isel(time=1)).reshape(-1)[0]
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", f"{args.steps * 6}h"),
        **dataclasses.asdict(ckpt.task_config))
    if inputs.sizes["time"] != 2 or targets.sizes["time"] != args.steps:
        raise ValueError("Unexpected input or target time selection.")
    for label, dataset in (("inputs", inputs), ("forcings", forcings)):
        for name, value in dataset.data_vars.items():
            if not np.isfinite(value.values).all():
                raise ValueError(f"Non-finite {label} variable: {name}")
    print(f"Initialized at {initialization}; inputs={dict(inputs.sizes)}", flush=True)

    forward = make_forward(ckpt.model_config, ckpt.task_config, stats)
    apply = functools.partial(jax.jit(forward.apply), params=ckpt.params, state={})

    def predict(**kwargs):
        return apply(**kwargs)[0]

    output = args.output_dir or Path("outputs") / time.strftime("graphcast_%Y%m%d_%H%M%S")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running", "upstream_commit": REVISION, "initialization_utc": str(initialization),
        "steps": args.steps, "resolution": 0.25, "pressure_levels": list(ckpt.task_config.pressure_levels),
        "assets": assets, "device": str(devices[0]), "device_kind": devices[0].device_kind,
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                for d in importlib.metadata.distributions()
                                if d.metadata["Name"])),
        "metrics": [],
    }
    manifest = output / "run.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Compiling the first step; this can take several minutes.", flush=True)
    start = time.monotonic()
    # The template carries shapes/coordinates, never future weather values.
    generator = rollout.chunked_prediction_generator(
        predictor_fn=predict, rng=jax.random.PRNGKey(0), inputs=inputs,
        targets_template=targets * np.nan, forcings=forcings, num_steps_per_chunk=1)
    for step, chunk in enumerate(generator, start=1):
        chunk = to_host(chunk)
        for name, value in chunk.data_vars.items():
            if not np.isfinite(value.values).all():
                raise RuntimeError(f"Non-finite prediction: {name}, step {step}")
        chunk = chunk.assign_coords(valid_time=("time", initialization + chunk.time.values))
        chunk.attrs.update(initialization_utc=str(initialization), source="Pretrained GraphCast ERA5 1979-2017")
        dest = output / f"prediction_{step * 6:03d}h.nc"
        chunk.to_netcdf(dest, engine="netcdf4")
        truth = targets["2m_temperature"].isel(time=step - 1)
        forecast = chunk["2m_temperature"].isel(time=0)
        weights = np.cos(np.deg2rad(truth.lat)).clip(min=0)
        rmse = float(np.sqrt(((forecast - truth) ** 2).weighted(weights).mean()))
        metadata["metrics"].append({"lead_hours": step * 6, "t2m_area_weighted_rmse_K": rmse})
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {dest}; 2m temperature RMSE={rmse:.3f} K", flush=True)
        if step == args.steps:
            plot_temperature(forecast, truth, output / "temperature.png", step * 6)
    metadata.update(status="complete", elapsed_seconds_including_compile_and_io=time.monotonic() - start)
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed. Results: {output.resolve()}", flush=True)


def make_forward(model_config, task_config, stats):
    """The official predictor wrapper order, shared with the CPU pipeline test."""
    import haiku as hk
    from graphcast import autoregressive, casting, graphcast, normalization

    @hk.transform_with_state
    def forward(inputs, targets_template, forcings):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, **stats)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    return forward


def to_host(dataset):
    """Transfer arrays while retaining xarray dimensions and coordinates."""
    import jax
    import numpy as np
    from graphcast import xarray_jax, xarray_tree
    return xarray_tree.map_structure(
        lambda a: a.copy(data=np.asarray(jax.device_get(xarray_jax.unwrap_data(a)))),
        dataset)


def plot_temperature(prediction, truth, path, hours):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prediction, truth = prediction.squeeze(), truth.squeeze()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    low, high = float(truth.min()), float(truth.max())
    for ax, field, title in zip(axes[:2], (truth, prediction), ("ERA5 reference", "GraphCast")):
        field.plot(ax=ax, x="lon", y="lat", vmin=low, vmax=high, cmap="coolwarm")
        ax.set_title(f"{title}: 2m temperature (K)")
    (prediction - truth).plot(ax=axes[2], x="lon", y="lat", cmap="RdBu_r", center=0)
    axes[2].set_title("Forecast minus reference (K)")
    fig.suptitle(f"Lead time: {hours} hours")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
