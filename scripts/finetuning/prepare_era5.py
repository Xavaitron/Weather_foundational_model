"""Stage split-safe ERA5 GraphCast_small data from the public WeatherBench 2 bucket."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.sparse import csr_matrix
import xarray as xr

SOURCE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
CLIMATOLOGY_SOURCE = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/"
    "1990-2017_6h_1440x721.zarr"
)
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURFACE = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "toa_incident_solar_radiation",
)
ATMOSPHERIC = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
STATIC = ("geopotential_at_surface", "land_sea_mask")
SCORE_SURFACE = SURFACE[:4]
SCORE_ATMOSPHERIC = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)
SPLITS = {
    "train": ("2016-01-01T00:00", "2019-12-31T18:00"),
    "validation": ("2020-01-01T00:00", "2020-12-31T18:00"),
    "test": ("2021-01-01T00:00", "2022-12-31T18:00"),
}


def _cell_edges(centers, lower, upper):
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0], edges[-1] = lower, upper
    return edges


def _latitude_weights(source_lat, target_lat):
    source_edges = _cell_edges(source_lat, -90.0, 90.0)
    target_edges = _cell_edges(target_lat, -90.0, 90.0)
    source_area = np.sin(np.deg2rad(source_edges))
    target_area = np.sin(np.deg2rad(target_edges))
    weights = np.zeros((target_lat.size, source_lat.size), dtype=np.float64)
    for target_index in range(target_lat.size):
        lower, upper = target_area[target_index:target_index + 2]
        overlap = np.maximum(
            0.0,
            np.minimum(upper, source_area[1:]) - np.maximum(lower, source_area[:-1]),
        )
        weights[target_index] = overlap / overlap.sum()
    return csr_matrix(weights.astype(np.float32))


def _longitude_weights(source_lon, target_lon):
    source_spacing = float(np.median(np.diff(source_lon)))
    offsets = ((source_lon[None, :] - target_lon[:, None] + 180) % 360) - 180
    overlap = np.maximum(
        0.0,
        np.minimum(offsets + source_spacing / 2, 0.5)
        - np.maximum(offsets - source_spacing / 2, -0.5),
    )
    overlap /= overlap.sum(axis=1, keepdims=True)
    return csr_matrix(overlap.astype(np.float32))


def _regrid_block(values, latitude_weights, longitude_weights):
    leading_shape = values.shape[:-2]
    frames = values.reshape((-1, values.shape[-2], values.shape[-1]))
    result = np.empty(
        (frames.shape[0], latitude_weights.shape[0], longitude_weights.shape[0]),
        dtype=np.float32,
    )
    for index, frame in enumerate(frames):
        latitude_regridded = latitude_weights @ frame
        result[index] = (longitude_weights @ latitude_regridded.T).T
    return result.reshape(leading_shape + result.shape[-2:])


def _regrid_array(array, latitude_weights, longitude_weights, target_lat, target_lon):
    array = array.chunk({"latitude": -1, "longitude": -1})
    result = xr.apply_ufunc(
        _regrid_block,
        array,
        input_core_dims=[["latitude", "longitude"]],
        output_core_dims=[["latitude_1deg", "longitude_1deg"]],
        exclude_dims={"latitude", "longitude"},
        kwargs={
            "latitude_weights": latitude_weights,
            "longitude_weights": longitude_weights,
        },
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={
            "output_sizes": {
                "latitude_1deg": target_lat.size,
                "longitude_1deg": target_lon.size,
            },
            "allow_rechunk": True,
        },
    )
    return result.rename({"latitude_1deg": "latitude", "longitude_1deg": "longitude"})\
        .assign_coords(latitude=target_lat, longitude=target_lon)


def regrid_dataset(dataset):
    source_lat = np.asarray(dataset.latitude.values, dtype=np.float64)
    source_lon = np.asarray(dataset.longitude.values, dtype=np.float64)
    dataset = dataset.sortby("latitude")
    source_lat = np.asarray(dataset.latitude.values, dtype=np.float64)
    if not np.allclose(np.diff(source_lat), 0.25) or not np.allclose(np.diff(source_lon), 0.25):
        raise ValueError("ERA5 source grid must be regular 0.25-degree latitude/longitude.")
    target_lat = np.arange(-90, 91, dtype=np.float32)
    target_lon = np.arange(360, dtype=np.float32)
    latitude_weights = _latitude_weights(source_lat, target_lat)
    longitude_weights = _longitude_weights(source_lon, target_lon)
    remapped = {
        name: _regrid_array(value, latitude_weights, longitude_weights, target_lat, target_lon)
        if "latitude" in value.dims and "longitude" in value.dims else value
        for name, value in dataset.data_vars.items()
    }
    result = xr.Dataset(remapped, attrs=dict(dataset.attrs))
    result = result.assign_coords(
        latitude=target_lat,
        longitude=target_lon,
    )
    result.attrs.update(source_resolution_degrees=0.25, model_resolution_degrees=1.0,
                        regridding="first-order conservative area-overlap averaging")
    return result


def estimate_uncompressed_bytes(frame_count):
    scalar_values_per_grid_cell = len(SURFACE) + len(ATMOSPHERIC) * len(LEVELS)
    values = frame_count * 181 * 360 * scalar_values_per_grid_cell
    static_values = 2 * 181 * 360
    return (values + static_values) * np.dtype(np.float32).itemsize


def prepare_climatology(output_dir, latitude_weights, longitude_weights, target_lat, target_lon):
    try:
        climatology = xr.open_zarr(
            CLIMATOLOGY_SOURCE,
            storage_options={"token": "anon"},
            chunks={},
            consolidated=True,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not open the public ERA5 climatology Zarr; check DNS/network access."
        ) from error
    variables = list(SCORE_SURFACE + SCORE_ATMOSPHERIC)
    missing = sorted(set(variables) - set(climatology.data_vars))
    if missing:
        raise ValueError(f"ERA5 climatology is missing benchmark variables: {missing}")
    climatology = climatology[variables].sel(level=list(LEVELS))
    climatology = climatology.sortby("latitude")
    climatology = xr.Dataset({
        name: _regrid_array(value, latitude_weights, longitude_weights, target_lat, target_lon)
        if "latitude" in value.dims and "longitude" in value.dims else value
        for name, value in climatology.data_vars.items()
    }, attrs=dict(climatology.attrs))
    climatology = climatology.assign_coords(latitude=target_lat, longitude=target_lon)
    climatology.attrs.update(
        source=CLIMATOLOGY_SOURCE,
        regridding="first-order conservative area-overlap averaging",
    )
    destination = output_dir / "climatology.zarr"
    if destination.exists():
        raise ValueError(f"Climatology store already exists: {destination}")
    climatology = climatology.chunk({"dayofyear": 1, "hour": 1, "latitude": 181, "longitude": 360})
    climatology.to_zarr(destination, mode="w", consolidated=True)
    print(f"Staged WB2 climatology: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/era5_graphcast_small_1deg"))
    parser.add_argument("--source-zarr", default=SOURCE)
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--yes", action="store_true", help="Skip the large-data staging confirmation")
    parser.add_argument("--estimate-only", action="store_true", help="Inspect split sizes and required fields only")
    parser.add_argument("--skip-climatology", action="store_true", help="Skip ACC reference climatology (RMSE-only evaluation)")
    return parser.parse_args()


def run(args):
    try:
        source = xr.open_zarr(
            args.source_zarr,
            storage_options={"token": "anon"},
            chunks={},
            consolidated=True,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not open public ERA5 Zarr. Check DNS/network access to storage.googleapis.com."
        ) from error

    if "latitude" not in source.coords or "longitude" not in source.coords or "time" not in source.coords:
        raise ValueError("ERA5 Zarr must contain time, latitude, and longitude coordinates.")
    if "level" not in source.coords or not set(LEVELS).issubset(set(source.level.values.tolist())):
        raise ValueError("ERA5 source is missing one or more GraphCast_small pressure levels.")
    required = list(SURFACE + ATMOSPHERIC + STATIC)
    missing = sorted(set(required) - set(source.data_vars))
    if missing:
        raise ValueError(f"ERA5 Zarr is missing required GraphCast_small variables: {missing}")

    source = source.sel(level=list(LEVELS))
    selected_splits = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
    frame_counts = {}
    for name, (start, stop) in selected_splits.items():
        frame_counts[name] = int(source.sel(time=slice(start, stop)).sizes["time"])
    estimated_bytes = estimate_uncompressed_bytes(sum(frame_counts.values()))
    if not args.skip_climatology:
        estimated_bytes += estimate_uncompressed_bytes(4 * 366)
    output_parent = args.output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_parent).free
    print(f"Public source: {args.source_zarr}")
    for name, frames in frame_counts.items():
        print(f"{name}: {frames} six-hour frames")
    print(f"Output grid: 181 x 360 x 13 levels; estimated uncompressed size: {estimated_bytes / 1e9:.1f} GB")
    print(f"Free space at destination: {free_bytes / 1e9:.1f} GB")
    if args.estimate_only:
        return
    if free_bytes < estimated_bytes:
        raise RuntimeError("Insufficient free disk space for the conservative uncompressed estimate.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty or new: {args.output_dir}")
    if not args.yes:
        if not sys.stdin.isatty():
            raise RuntimeError("Pass --yes to begin staging this multi-year dataset non-interactively.")
        answer = input("Stage the selected ERA5 splits locally? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Dataset staging cancelled.")
            return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variables = list(SURFACE + ATMOSPHERIC + STATIC)
    for name, (start, stop) in selected_splits.items():
        destination = args.output_dir / f"{name}.zarr"
        if destination.exists():
            raise ValueError(f"Split store already exists: {destination}")
        split = source[variables].sel(time=slice(start, stop))
        split = split.assign_coords(datetime=("time", split.time.values))
        split = regrid_dataset(split)
        split = split.rename({"latitude": "lat", "longitude": "lon"})
        split.attrs.update(split=name, start_utc=start, stop_utc=stop)
        split = split.chunk({"time": 1, "lat": 181, "lon": 360})
        split.to_zarr(destination, mode="w", consolidated=True)
        print(f"Staged {name}: {destination}", flush=True)
    if not args.skip_climatology:
        target_lat = np.arange(-90, 91, dtype=np.float32)
        target_lon = np.arange(360, dtype=np.float32)
        source_lat = np.arange(-90, 90.0001, 0.25, dtype=np.float64)
        source_lon = np.arange(0, 360, 0.25, dtype=np.float64)
        prepare_climatology(
            args.output_dir,
            _latitude_weights(source_lat, target_lat),
            _longitude_weights(source_lon, target_lon),
            target_lat,
            target_lon,
        )


if __name__ == "__main__":
    try:
        run(parse_args())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)