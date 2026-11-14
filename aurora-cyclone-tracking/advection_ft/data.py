import json
from datetime import datetime, timezone

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .schema import LEVELS, canonical_names, interface


def open_store(path):
    if str(path).endswith((".nc", ".nc4")):
        return xr.open_dataset(path, chunks={})
    return xr.open_zarr(path, consolidated=True)


def validate_store(ds, task):
    missing = set(canonical_names(task)) - set(ds.data_vars)
    if missing:
        raise ValueError(f"Missing canonical variables: {sorted(missing)}. Run prepare_data.py.")
    expected_lat = np.arange(89.25, -90, -1.5)
    expected_lon = np.arange(0.75, 360, 1.5)
    if ds.sizes.get("latitude") != 120 or ds.sizes.get("longitude") != 240:
        raise ValueError("Expected global cell-centred 1.5-degree grid (120,240).")
    if not np.allclose(ds.latitude, expected_lat) or not np.allclose(ds.longitude, expected_lon):
        raise ValueError("Coordinate values do not match the prepared 1.5-degree grid.")
    if tuple(ds.level.values.tolist()) != LEVELS:
        raise ValueError(f"Expected pressure levels in hPa: {LEVELS}")
    times = ds.time.values.astype("datetime64[ns]")
    if len(times) < 3 or np.any(np.diff(times).astype("timedelta64[s]").astype(int) <= 0):
        raise ValueError("Times must be unique and strictly ascending.")


class WeatherDataset(Dataset):
    def __init__(self, path, task, start, end, rollout_steps=2, stride=1):
        self.path, self.task, self.rollout_steps = path, task, rollout_steps
        self.ds = open_store(path)
        validate_store(self.ds, task)
        self.surf, self.atmos, self.static, self.hours = interface(task)
        times = self.ds.time.values.astype("datetime64[ns]")
        self.times = times
        lo, hi = np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D")
        cadence = np.timedelta64(self.hours, "h")
        self.indices = []
        for i in range(1, len(times) - rollout_steps, stride):
            window = times[i-1:i+rollout_steps+1]
            if window[0] >= lo and window[-1] < hi and np.all(np.diff(window) == cadence):
                self.indices.append(i)
        if not self.indices:
            raise ValueError(f"No complete {self.hours}h windows in {start}..{end} for {path}")
        self.lat = torch.tensor(self.ds.latitude.values.copy(), dtype=torch.float32)
        self.lon = torch.tensor(self.ds.longitude.values.copy(), dtype=torch.float32)
        self.static_fields = {v: self._tensor(self.ds[f"static__{v}"].transpose("latitude", "longitude").values)
                              for v in self.static}
        self.source_metadata = dict(self.ds.attrs)

    @staticmethod
    def _tensor(arr):
        a = np.array(arr, dtype=np.float32, copy=True)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite data found. Handle missing/below-ground data explicitly before training.")
        return torch.from_numpy(a)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ds"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.ds is None:
            self.ds = open_store(self.path)
        i = self.indices[index]
        stop = i + self.rollout_steps + 1
        surf = {v: self._tensor(self.ds[f"surf__{v}"].isel(time=slice(i-1, stop)).transpose(
            "time", "latitude", "longitude").values) for v in self.surf}
        atmos = {v: self._tensor(self.ds[f"atmos__{v}"].isel(time=slice(i-1, stop)).transpose(
            "time", "level", "latitude", "longitude").values) for v in self.atmos}
        return {"surf": surf, "atmos": atmos, "time_ns": int(self.times[i].astype(np.int64))}

    def make_batch(self, item, device):
        from aurora import Batch, Metadata
        # Dataset windows have 2 history states followed by rollout targets.
        times = tuple(datetime.fromisoformat(str(np.datetime64(int(t), "ns"))).replace(tzinfo=timezone.utc)
                      for t in item["time_ns"].tolist())
        return Batch(surf_vars={v: x[:, :2].to(device) for v, x in item["surf"].items()},
                     atmos_vars={v: x[:, :2].to(device) for v, x in item["atmos"].items()},
                     static_vars={v: x.to(device) for v, x in self.static_fields.items()},
                     metadata=Metadata(lat=self.lat.to(device), lon=self.lon.to(device),
                                       time=times, atmos_levels=LEVELS))


def read_config(path):
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    train_lo, train_hi = np.datetime64(config["train_start"]), np.datetime64(config["train_end"])
    val_lo, val_hi = np.datetime64(config["val_start"]), np.datetime64(config["val_end"])
    test_lo, test_hi = np.datetime64(config["test_start"]), np.datetime64(config["test_end"])
    if not train_lo <= train_hi < val_lo <= val_hi < test_lo <= test_hi:
        raise ValueError("Require disjoint chronological train, validation and test date ranges.")
    return config
