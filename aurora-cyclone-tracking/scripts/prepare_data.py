"""Prepare global, cell-centred 1.5-degree Aurora inputs from NetCDF/Zarr.

Uses separable first-order conservative regridding for scalar fields and
Cartesian-vector regridding for u/v. No automatic unit guessing or NaN filling.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

if not __package__:
    import _bootstrap
from advection_ft.schema import ALIASES, LEVELS, canonical_names, interface
from scripts.public_store import PublicGCSStore


def open_source(path):
    if str(path).endswith((".nc", ".nc4")):
        ds = xr.open_dataset(path, chunks=None)
    else:
        kwargs = {"storage_options": {"token": "anon"}} if str(path).startswith("gs://") else {}
        ds = xr.open_zarr(PublicGCSStore(str(path)) if kwargs else path, consolidated=True, chunks=None)
    renames = {a: b for a, b in [("lat", "latitude"), ("lon", "longitude"),
                                 ("pressure_level", "level"), ("valid_time", "time")]
               if a in ds.dims and b not in ds.dims}
    ds = ds.rename(renames)
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude").sortby("latitude")
    return ds


def conservative_weights(source_lat, source_lon):
    """Exact overlap areas for rectilinear latitude/longitude cell bounds."""
    source_lat, source_lon = np.asarray(source_lat), np.asarray(source_lon)
    if not np.allclose(np.diff(source_lon), np.diff(source_lon)[0]):
        raise ValueError("Source longitude must be regular.")
    step = np.diff(source_lon)[0]
    if not np.isclose(step * len(source_lon), 360):
        raise ValueError("Source must cover the globe with no duplicate seam column.")
    lat_bounds = np.r_[source_lat[0] - (source_lat[1]-source_lat[0])/2,
                       (source_lat[:-1]+source_lat[1:])/2,
                       source_lat[-1] + (source_lat[-1]-source_lat[-2])/2].clip(-90, 90)
    if not np.isclose(lat_bounds[0], -90) or not np.isclose(lat_bounds[-1], 90):
        raise ValueError("Source latitude cells must cover both poles.")
    target_bounds = np.linspace(-90, 90, 121)
    south = np.maximum(target_bounds[:-1, None], lat_bounds[None, :-1])
    north = np.minimum(target_bounds[1:, None], lat_bounds[None, 1:])
    wy = np.maximum(0, np.sin(np.deg2rad(north))-np.sin(np.deg2rad(south)))
    wy /= np.diff(np.sin(np.deg2rad(target_bounds)))[:, None]
    target_lon_bounds = np.linspace(0, 360, 241)
    wx = np.zeros((240, len(source_lon)))
    for shift in (-360, 0, 360):
        left = np.maximum(target_lon_bounds[:-1, None], source_lon[None] - step/2 + shift)
        right = np.minimum(target_lon_bounds[1:, None], source_lon[None] + step/2 + shift)
        wx += np.maximum(0, right-left)/1.5
    if not np.allclose(wy.sum(1), 1) or not np.allclose(wx.sum(1), 1):
        raise ValueError("Regridding weights do not cover the target grid.")
    return wy[::-1].copy(), wx


def regrid(field, categorical=False):
    lat = np.arange(89.25, -90, -1.5)
    lon = np.arange(0.75, 360, 1.5)
    if categorical:
        # Append periodic seam copies to avoid longitude-edge extrapolation.
        ext = xr.concat((field.isel(longitude=[-1]).assign_coords(longitude=[float(field.longitude[-1])-360]),
                         field, field.isel(longitude=[0]).assign_coords(longitude=[float(field.longitude[0])+360])),
                        "longitude")
        return ext.interp(latitude=lat, longitude=lon, method="nearest").astype("float32")
    wy, wx = conservative_weights(field.latitude.values, field.longitude.values)
    a = xr.DataArray(wy, dims=("out_lat", "latitude"),
                     coords={"out_lat": lat, "latitude": field.latitude})
    b = xr.DataArray(wx, dims=("out_lon", "longitude"),
                     coords={"out_lon": lon, "longitude": field.longitude})
    result = xr.dot(xr.dot(field, a, dim="latitude"), b, dim="longitude")
    return result.rename(out_lat="latitude", out_lon="longitude").astype("float32")


def regrid_vector(u, v):
    phi, lam = np.deg2rad(u.latitude), np.deg2rad(u.longitude)
    vx = regrid(-u*np.sin(lam) - v*np.sin(phi)*np.cos(lam))
    vy = regrid(u*np.cos(lam) - v*np.sin(phi)*np.sin(lam))
    vz = regrid(v*np.cos(phi))
    phi, lam = np.deg2rad(vx.latitude), np.deg2rad(vx.longitude)
    return (-vx*np.sin(lam)+vy*np.cos(lam),
            -vx*np.sin(phi)*np.cos(lam)-vy*np.sin(phi)*np.sin(lam)+vz*np.cos(phi))


def resolve(canonical, datasets, mapping):
    specification = mapping.get(canonical, {})
    if isinstance(specification, str):
        specification = {"name": specification}
    candidates = [specification["name"]] if "name" in specification else [canonical] + ALIASES.get(
        canonical, [canonical.split("__", 1)[1]])
    indices = [specification["source_index"]] if "source_index" in specification else range(len(datasets))
    for i in indices:
        ds = datasets[i]
        for name in candidates:
            if name not in ds:
                continue
            f = ds[name]
            kind = canonical.split("__")[0]
            if kind == "atmos" and "level" not in f.dims:
                continue
            if kind != "atmos" and "level" in f.dims:
                continue
            if kind == "static" and "time" in f.dims:
                f = f.isel(time=0, drop=True)
            if kind != "static" and "time" not in f.dims:
                continue
            for dim in list(f.dims):
                if dim not in ("time", "level", "latitude", "longitude"):
                    if f.sizes[dim] != 1:
                        raise ValueError(f"Select a single member/experiment before preparing {name}: {dim}")
                    f = f.isel({dim: 0}, drop=True)
            if kind == "atmos":
                f = f.sel(level=list(LEVELS))
            # Build Dask tasks only after selecting variables, dates and levels.
            f = f.chunk({'time': 4} if 'time' in f.dims else {})
            return f * specification.get("scale", 1.0) + specification.get("offset", 0.0)
    raise ValueError(f"Cannot resolve {canonical}. Supply --mapping with an explicit name/source_index. "
                     "Missing pollution/static fields must be supplied; they are never fabricated.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", required=True, help="Repeat for surface, pressure-level and static stores.")
    p.add_argument("--output", required=True)
    p.add_argument("--task", choices=["cyclone", "air_pollution"], default="cyclone")
    p.add_argument("--mapping", help="JSON canonical key -> source variable or {name,source_index,scale,offset}.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--data-vintage", required=True, help="E.g. HRES-T0 2016-2023; ERA5 is a different experiment.")
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    mapping = json.loads(Path(args.mapping).read_text()) if args.mapping else {}
    datasets = [open_source(path) for path in args.source]
    _, _, _, hours = interface(args.task)
    for i, ds in enumerate(datasets):
        if "time" in ds.dims:
            ds = ds.sel(time=slice(args.start, args.end))
            ds = ds.sel(time=(ds.time.dt.hour % hours == 0) & (ds.time.dt.minute == 0))
        datasets[i] = ds
    raw = {key: resolve(key, datasets, mapping) for key in canonical_names(args.task)}
    # Exact time alignment prevents implicit unions and NaN-filled targets.
    keys = [k for k in raw if not k.startswith("static__")]
    aligned = xr.align(*(raw[k] for k in keys), join="exact", copy=False)
    raw.update(dict(zip(keys, aligned)))
    output = {}
    for u, v in [("surf__10u", "surf__10v"), ("atmos__u", "atmos__v")]:
        output[u], output[v] = regrid_vector(raw[u], raw[v])
    for name, value in raw.items():
        if name not in output:
            output[name] = regrid(value, categorical=name == "static__slt")
    ds = xr.Dataset(output)
    ds = ds.assign_coords(level=list(LEVELS))
    ds.attrs.update(task=args.task, resolution_degrees=1.5, grid="cell_centred_120x240",
                    regridding="first_order_conservative_scalars_cartesian_vectors_nearest_soil_type",
                    data_vintage=args.data_vintage, sources=json.dumps(args.source),
                    variable_mapping=json.dumps(mapping), created_utc=datetime.now(timezone.utc).isoformat())
    ds = ds.chunk({"time": 4, "latitude": 120, "longitude": 240, "level": 13})
    for name in ds:
        order = [d for d in ("time", "level", "latitude", "longitude") if d in ds[name].dims]
        ds[name] = ds[name].transpose(*order)
        ds[name].encoding = {}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(args.output, mode="w-", consolidated=True)
    print(f"Prepared {args.output}: {dict(ds.sizes)}; inspect units and source vintage before training.")


if __name__ == "__main__":
    main()
