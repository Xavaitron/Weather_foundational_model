"""CPU checks with a tiny random model; not a pretrained forecast benchmark."""
import dataclasses
import functools
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import infer_graphcast as runner


class AssetTests(unittest.TestCase):
    def test_horizon_selects_covering_sample(self):
        for steps, suffix in ((1, "01"), (2, "04"), (4, "04"), (5, "12"), (12, "12")):
            self.assertTrue(runner.sample_name(steps).endswith(f"steps-{suffix}.nc"))
        for steps in (0, -1, 13):
            with self.assertRaises(ValueError):
                runner.sample_name(steps)

    def test_checksum_matches_published_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "asset"
            file.write_bytes(b"abc")
            self.assertEqual(runner.checksum(file), "kAFQmDzST7DWlj99KOF/cg==")

    def test_corrupt_cache_is_replaced_and_verified_cache_is_reused(self):
        metadata = {"size": "3", "md5Hash": "kAFQmDzST7DWlj99KOF/cg==", "generation": "1"}
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            (cache / "asset").write_bytes(b"bad")
            with mock.patch.object(runner.urllib.request, "urlopen", side_effect=[
                io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(b"abc")
            ]):
                path, _ = runner.fetch("asset", cache)
            self.assertEqual(path.read_bytes(), b"abc")
            with mock.patch.object(runner.urllib.request, "urlopen", return_value=
                io.BytesIO(json.dumps(metadata).encode())) as request:
                runner.fetch("asset", cache)
                self.assertEqual(request.call_count, 1)


class PipelineTest(unittest.TestCase):
    def test_two_step_rollout_serialization_and_plot(self):
        import jax
        import numpy as np
        import xarray as xr
        from graphcast import data_utils, graphcast, rollout

        task = graphcast.TASK_13
        coords = {
            "batch": [0], "time": np.arange(4) * np.timedelta64(6, "h"),
            "lat": np.linspace(-90, 90, 31, dtype=np.float32),
            "lon": np.arange(60, dtype=np.float32) * 6,
            "level": list(task.pressure_levels),
        }
        ds = xr.Dataset(coords=coords)
        ds = ds.assign_coords(datetime=(("batch", "time"),
            (np.datetime64("2022-01-01") + coords["time"])[None, :]))
        rng = np.random.default_rng(1)
        for name in graphcast.TARGET_SURFACE_VARS + graphcast.TARGET_ATMOSPHERIC_VARS:
            dims = ("batch", "time", "lat", "lon")
            if name in graphcast.TARGET_ATMOSPHERIC_VARS:
                dims += ("level",)
            ds[name] = (dims, rng.normal(size=tuple(ds.sizes[d] for d in dims)).astype("float32"))
        ds["toa_incident_solar_radiation"] = xr.zeros_like(ds["2m_temperature"])
        for name in graphcast.STATIC_VARS:
            ds[name] = (("lat", "lon"), np.zeros((31, 60), dtype=np.float32))
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            ds, target_lead_times=slice("6h", "12h"), **dataclasses.asdict(task))
        self.assertEqual(inputs.sizes["time"], 2)
        # Statistics for all physical fields; generated clock features are
        # already dimensionless and the official wrapper leaves them unchanged.
        variables = graphcast.TARGET_SURFACE_VARS + graphcast.TARGET_ATMOSPHERIC_VARS + graphcast.STATIC_VARS + graphcast.EXTERNAL_FORCING_VARS
        stats = {
            name: xr.Dataset({v: xr.DataArray(np.float32(value)) for v in variables})
            for name, value in (("mean_by_level", 0), ("stddev_by_level", 1), ("diffs_stddev_by_level", 1))
        }
        config = graphcast.ModelConfig(resolution=6, mesh_size=1, latent_size=8,
            gnn_msg_steps=2, hidden_layers=1, radius_query_fraction_edge_length=0.6)
        forward = runner.make_forward(config, task, stats)
        template = targets * np.nan
        params, state = forward.init(jax.random.PRNGKey(0), inputs,
            template.isel(time=slice(0, 1)), forcings.isel(time=slice(0, 1)))
        apply = functools.partial(jax.jit(forward.apply), params=params, state=state)
        chunks = list(rollout.chunked_prediction_generator(
            predictor_fn=lambda **kw: apply(**kw)[0], rng=jax.random.PRNGKey(1),
            inputs=inputs, targets_template=template, forcings=forcings,
            num_steps_per_chunk=1))
        self.assertEqual(len(chunks), 2)
        host = runner.to_host(chunks[-1])
        self.assertIsInstance(host, xr.Dataset)
        self.assertEqual(set(host.data_vars), set(task.target_variables))
        self.assertEqual(host.time.values[0], np.timedelta64(12, "h"))
        for field in host.data_vars.values():
            self.assertTrue(np.isfinite(field.values).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            host.to_netcdf(path / "prediction.nc", engine="netcdf4")
            with xr.open_dataset(path / "prediction.nc") as loaded:
                xr.testing.assert_allclose(host, loaded)
            runner.plot_temperature(host["2m_temperature"].isel(time=0),
                targets["2m_temperature"].isel(time=1), path / "plot.png", 12)
            self.assertGreater((path / "plot.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
