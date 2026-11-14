"""Download and prepare public cyclone inputs; resume completed monthly shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import dask
import numpy as np
import pandas as pd
import xarray as xr

from advection_ft.schema import LEVELS, canonical_names
from prepare_data import open_source, resolve

HRES = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
STATIC = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
IBTRACS = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
ROOT = Path(__file__).resolve().parent


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def download_file(url, path, attempts=4):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Aurora-advection-research/1.0"})
            with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            partial.replace(path)
            return
        except (OSError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def expected_times(start, end):
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1),
                         freq="6h", inclusive="left").values


def check_times(actual, start, end):
    if not np.array_equal(np.asarray(actual, dtype="datetime64[ns]"), expected_times(start, end)):
        raise ValueError(f"Source does not contain every six-hour timestamp in {start}..{end}; "
                         "choose dates within its coverage. No data will be fabricated.")


def monthly_ranges(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    current = start
    while current <= end:
        stop = min(current + pd.offsets.MonthEnd(0), end)
        yield current.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")
        current = stop + pd.Timedelta(days=1)


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


def check_prepared(ds, start, end):
    if set(canonical_names("cyclone")) - set(ds.data_vars):
        raise ValueError("Prepared shard is missing required variables")
    check_times(ds.time.values, start, end)
    if tuple(ds.level.values) != LEVELS or ds.sizes["latitude"] != 120 or ds.sizes["longitude"] != 240:
        raise ValueError("Prepared shard has incorrect levels or grid")
    # Reduction is chunked: missing source values must fail before training starts.
    for name in ds.data_vars:
        if not bool(np.isfinite(ds[name]).all().compute()):
            raise ValueError(f"Nonfinite values in {name}; source needs an explicit missing-data policy")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["hres", "era5"], default="hres")
    p.add_argument("--config", default="configs/cyclone.json")
    p.add_argument("--workers", type=int, default=2, help="Dask threads per preparation process")
    p.add_argument("--check-only", action="store_true", help="Check remote schema/dates without downloading fields")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    config = json.loads(Path(args.config).read_text())
    if config["task"] != "cyclone":
        p.error("This public downloader supports cyclone inputs. CAMS pollution data requires a separate source.")
    intervals = [(pd.Timestamp(config[f"{s}_start"]), pd.Timestamp(config[f"{s}_end"]))
                 for s in ("train", "val", "test")]
    if any(a > b for a, b in intervals) or any(intervals[i][1] >= intervals[i+1][0] for i in (0, 1)):
        p.error("Train/validation/test dates must be ordered and disjoint")
    start, end = intervals[0][0].strftime("%Y-%m-%d"), intervals[-1][1].strftime("%Y-%m-%d")
    output = Path(config["store"]).resolve()
    cache = output.parent / (output.stem + "_download_cache")
    spec = {"version": 1, "source": HRES if args.source == "hres" else ERA5,
            "static_source": STATIC, "static_time": start, "start": start, "end": end,
            "grid": "cell_centred_120x240", "levels": list(LEVELS), "ibtracs_source": IBTRACS}
    marker = output.with_name(output.name + ".download.json")
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError("Existing output belongs to another source/date request. Use a different config store path.")
    weather_ready = output.exists() and marker.exists()
    if output.exists() and not weather_ready:
        raise FileExistsError(f"Unmanaged or incomplete output {output}; choose a new store path")
    if not weather_ready:
        print(f"Checking {spec['source']}", flush=True)
        with open_source(spec["source"]) as weather, open_source(STATIC) as static:
            selected = weather.sel(time=slice(start, end))
            check_times(selected.time.values, start, end)
            for name in canonical_names("cyclone"):
                resolve(name, [static.sel(time=[start], method="nearest")] if name.startswith("static__")
                        else [selected], {})
            print(f"Verified required fields and {len(selected.time)} timestamps. "
                  "Static fields use ERA5 at the start date.", flush=True)
        if args.check_only:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        request_file = cache / "request.json"
        if request_file.exists() and json.loads(request_file.read_text()) != spec:
            raise ValueError("Download cache belongs to a different request; use a different store path")
        atomic_json(request_file, spec)
        os.environ["DASK_SCHEDULER"] = "threads"
        os.environ["DASK_NUM_WORKERS"] = str(args.workers)
        dask.config.set(scheduler="threads", num_workers=args.workers)
        static_file = cache / "static.nc"
        if not static_file.exists():
            print("Downloading land mask, surface geopotential and soil type...", flush=True)
            with open_source(STATIC) as remote:
                fields = {name: resolve(name, [remote.sel(time=[start], method="nearest")], {})
                          for name in ("static__lsm", "static__z", "static__slt")}
                xr.Dataset(fields).to_netcdf(cache / "static.partial.nc")
            (cache / "static.partial.nc").replace(static_file)
        mapping_file = cache / "mapping.json"
        atomic_json(mapping_file, {name: {"name": name, "source_index": 1}
                                  for name in ("static__lsm", "static__z", "static__slt")})
        shards = []
        for lo, hi in monthly_ranges(start, end):
            shard = cache / f"{lo}_{hi}.zarr"
            if not shard.exists():
                print(f"Downloading and regridding {lo}..{hi}", flush=True)
                with tempfile.TemporaryDirectory(prefix="month_", dir=cache) as temp:
                    staged = Path(temp) / "prepared.zarr"
                    run("prepare_data.py", "--source", spec["source"], "--source", static_file,
                        "--mapping", mapping_file, "--output", staged, "--start", lo, "--end", hi,
                        "--data-vintage", f"{args.source.upper()} WeatherBench2; ERA5 static fields {start}")
                    with xr.open_zarr(staged, consolidated=True) as ds:
                        check_prepared(ds, lo, hi)
                    staged.rename(shard)
            shards.append(shard)
        print("Assembling local training store (monthly cache is retained for resume)...", flush=True)
        datasets = [xr.open_zarr(path, consolidated=True) for path in shards]
        try:
            combined = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="equals")
            check_times(combined.time.values, start, end)
            combined.attrs.update(download_sources=json.dumps(spec),
                                  data_vintage=f"{args.source.upper()} WeatherBench2; ERA5 static fields {start}")
            combined = combined.chunk({"time": 4, "level": 13, "latitude": 120, "longitude": 240})
            for name in combined.variables:
                combined[name].encoding = {}
            with tempfile.TemporaryDirectory(prefix="assemble_", dir=cache) as temp:
                staged = Path(temp) / "complete.zarr"
                combined.to_zarr(staged, mode="w-", consolidated=True)
                staged.rename(output)
            atomic_json(marker, spec)
        finally:
            for ds in datasets:
                ds.close()
    elif args.check_only:
        print(f"Already prepared: {output}")
        return
    raw = cache / "ibtracs.ALL.list.v04r01.csv"
    download_file(IBTRACS, raw)
    tracks = output.parent / "ibtracs.csv"
    if not tracks.exists():
        with tempfile.TemporaryDirectory(prefix="tracks_", dir=output.parent) as temp:
            staged = Path(temp) / "ibtracs.csv"
            run("prepare_ibtracs.py", "--input", raw, "--output", staged)
            staged.replace(tracks)
    provenance = dict(spec, ibtracs_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    atomic_json(output.parent / "download_manifest.json", provenance)
    print(f"Ready: {output}\nIBTrACS: {tracks}\nSource: {args.source.upper()}", flush=True)


if __name__ == "__main__":
    main()
