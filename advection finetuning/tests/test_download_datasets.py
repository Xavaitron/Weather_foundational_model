import json
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

import download_datasets as downloader
from advection_ft.schema import LEVELS, canonical_names


def test_download_prepare_and_resume(tmp_path, monkeypatch):
    """Real preparation/IBTrACS CLIs; local sources replace public network stores."""
    times = downloader.expected_times("2020-01-30", "2020-02-02")
    shape = (len(times), 6, 12)
    fields = {}
    for name in canonical_names("cyclone"):
        if name.startswith("static__"):
            fields[name] = (("latitude", "longitude"), np.ones(shape[1:], dtype="float32"))
        elif name.startswith("atmos__"):
            fields[name] = (("time", "level", "latitude", "longitude"),
                            np.ones((shape[0], 13, *shape[1:]), dtype="float32"))
        else:
            fields[name] = (("time", "latitude", "longitude"), np.ones(shape, dtype="float32"))
    source = tmp_path / "source.nc"
    xr.Dataset(fields, coords={"time": times, "level": list(LEVELS),
                              "latitude": np.linspace(-90, 90, 6),
                              "longitude": np.arange(0, 360, 30)}).to_netcdf(source)
    tracks = tmp_path / "official.csv"
    tracks.write_text("SID,ISO_TIME,LAT,LON,BASIN,NATURE,TRACK_TYPE\n"
                      "storm,2020-02-01 00:00:00,10,20,NA,TS,MAIN\n")
    output = tmp_path / "data" / "weather.zarr"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(task="cyclone", store=str(output),
        train_start="2020-01-30", train_end="2020-01-30", val_start="2020-01-31",
        val_end="2020-01-31", test_start="2020-02-01", test_end="2020-02-02")))
    monkeypatch.setattr(downloader, "HRES", str(source))
    monkeypatch.setattr(downloader, "STATIC", str(source))
    monkeypatch.setattr(downloader, "IBTRACS", tracks.as_uri())
    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--config", str(config)])
    downloader.main()
    with xr.open_zarr(output, consolidated=True) as ds:
        downloader.check_prepared(ds, "2020-01-30", "2020-02-02")
        assert ds.static__lsm.dims == ("latitude", "longitude")
    assert (output.parent / "ibtracs.csv").exists()
    assert len(list((output.parent / "weather_download_cache").glob("*.zarr"))) == 2
    monkeypatch.setattr(downloader, "open_source", lambda _: pytest.fail("Resume should use local data"))
    downloader.main()
    monkeypatch.setattr(downloader, "HRES", "another-source")
    with pytest.raises(ValueError, match="another source"):
        downloader.main()


def test_missing_timestamps_rejected():
    with pytest.raises(ValueError, match="every six-hour"):
        downloader.check_times(downloader.expected_times("2022-01-01", "2022-01-01")[:-1],
                               "2022-01-01", "2022-01-01")


def test_month_boundaries():
    assert list(downloader.monthly_ranges("2020-01-31", "2020-03-01")) == [
        ("2020-01-31", "2020-01-31"), ("2020-02-01", "2020-02-29"),
        ("2020-03-01", "2020-03-01")]
