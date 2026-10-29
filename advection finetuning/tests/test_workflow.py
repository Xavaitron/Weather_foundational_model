import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from advection_ft.data import WeatherDataset
from advection_ft.metrics import paired_comparison, summarise_tracks
from advection_ft.schema import LEVELS
from advection_ft.tracking import CycloneTracker
from tests.test_integration import synthetic_batch


def make_store(path):
    batch = synthetic_batch()
    times = np.concatenate([np.arange(np.datetime64(f"{year}-01-01T00"),
                            np.datetime64(f"{year}-01-04T00"), np.timedelta64(6, "h"))
                            for year in (2016, 2022, 2023)])
    variables = {}
    for key, value in batch.surf_vars.items():
        variables[f"surf__{key}"] = (("time", "latitude", "longitude"),
                                     np.repeat(value[0, :1].numpy(), len(times), axis=0))
    for key, value in batch.atmos_vars.items():
        variables[f"atmos__{key}"] = (("time", "level", "latitude", "longitude"),
                                      np.repeat(value[0, :1].numpy(), len(times), axis=0))
    for key, value in batch.static_vars.items():
        variables[f"static__{key}"] = (("latitude", "longitude"), value.numpy())
    ds = xr.Dataset(variables, coords={"time": times, "level": list(LEVELS),
        "latitude": batch.metadata.lat.numpy(), "longitude": batch.metadata.lon.numpy()},
        attrs={"data_vintage": "SYNTHETIC_TEST_ONLY"})
    ds.to_zarr(path, consolidated=True)


def test_dataset_windows_do_not_cross_gaps_or_split_boundaries(tmp_path):
    path = tmp_path/"data.zarr"
    make_store(path)
    data = WeatherDataset(str(path), "cyclone", "2016-01-01", "2016-12-31", rollout_steps=2)
    assert len(data) == 9
    for i in data.indices:
        assert np.all(np.diff(data.times[i-1:i+3]) == np.timedelta64(6, "h"))
        assert data.times[i+2] < np.datetime64("2017-01-01")
    with pytest.raises(ValueError, match="No complete"):
        WeatherDataset(str(path), "cyclone", "2016-12-31", "2021-12-31", rollout_steps=2)


def test_storm_bootstrap_and_relative_improvement():
    rows = [dict(sid=f"storm{s}", basin="NA", lead_hours=24, init_time=str(i), valid_time=str(i+1),
                 error_km=100.+s, tracker_failed=False) for s in range(3) for i in range(4)]
    ref = pd.DataFrame(rows)
    candidate = ref.copy()
    candidate.error_km *= .8
    comparison = paired_comparison(candidate, ref, [24], repetitions=50)
    assert np.allclose(comparison.relative_error_percent, -20)
    assert comparison.significant_improvement.all()
    summary = summarise_tracks(ref, [24], repetitions=50)
    assert (summary.n_cases == 12).all() and (summary.n_storms == 3).all()


def test_tracker_keeps_failed_case_and_handles_dateline():
    lat, lon = np.arange(89.25, -90, -1.5), np.arange(.75, 360, 1.5)
    tracker = CycloneTracker(lat, lon, 15, 359)
    field = np.full((120, 240), np.nan)
    point, failed = tracker.step(field, field)
    assert failed and point == (15, 359)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    msl = (yy-15)**2 + ((xx-.75+180)%360-180)**2
    point, failed = tracker.step(msl, msl)
    assert not failed and abs(point[0]-15) < 1 and min(point[1], 360-point[1]) < 2


def test_training_checkpoint_resume_evaluation_and_export(tmp_path):
    root = Path(__file__).resolve().parents[1]
    store, run = tmp_path/"data.zarr", tmp_path/"runs"/"nowcastnet"
    make_store(store)
    config = dict(task="cyclone", model="tiny_random", store=str(store),
        train_start="2016-01-01", train_end="2016-12-31", val_start="2022-01-01", val_end="2022-12-31",
        test_start="2023-01-01", test_end="2023-12-31", precision="fp32", activation_checkpointing=True,
        batch_size=1, accumulation_steps=1, rollout_steps=2, max_steps=1, warmup_steps=0,
        learning_rate=5e-6, weight_decay=5e-6, seed=42, workers=0, validate_every=1,
        validation_batches=1, save_every=1, adapter_channels=8, flow_groups=2)
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(config))
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    def command(*args):
        subprocess.run([sys.executable, *map(str, args)], cwd=root, env=env,
                       check=True, capture_output=True, text=True)
    prepared = tmp_path/"prepared.zarr"
    command("prepare_data.py", "--source", store, "--output", prepared, "--task", "cyclone",
            "--start", "2016-01-01", "--end", "2023-12-31", "--data-vintage", "SYNTHETIC_TEST_ONLY")
    validate = WeatherDataset(str(prepared), "cyclone", "2016-01-01", "2016-12-31", rollout_steps=2)
    assert len(validate) == 9
    command("finetune_nowcastnet.py", "--config", config_path, "--output", run)
    checkpoint = torch.load(run/"latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 1 and checkpoint["approach"] == "nowcastnet"
    command("finetune_nowcastnet.py", "--config", config_path, "--output", run, "--resume", run/"latest.pt")
    truth = pd.DataFrame([dict(sid="test", time=f"2023-01-0{day}T{hour:02}:00:00Z", lat=15,
                               lon=150, basin="WP", nature="TS")
                         for day in (1, 2, 3) for hour in (0, 6, 12, 18)])
    ibtracs = tmp_path/"ibtracs.csv"
    truth.to_csv(ibtracs, index=False)
    command("evaluate_cyclones.py", "--checkpoint", run/"best.pt", "--ibtracs", ibtracs,
            "--leads", 6, "--max-inits", 1, "--bootstrap", 10)
    tracks = pd.read_csv(run/"cyclone_evaluation"/"tracks.csv")
    assert len(tracks) == 1 and np.isfinite(tracks.error_km).all()
    command("export_results.py", "--runs", tmp_path/"runs", "--output", tmp_path/"results.zip")
    from zipfile import ZipFile
    with ZipFile(tmp_path/"results.zip") as archive:
        assert any(name.endswith("tracks.csv") for name in archive.namelist())
        assert not any(name.endswith(".pt") for name in archive.namelist())
