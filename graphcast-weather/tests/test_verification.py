import numpy as np
import pytest
import xarray as xr
from graphcast import losses
from finetuning.evaluate import spatial_mse
from finetuning.verification import aggregate, spatial_acc, select_climatology


def field(values):
    return xr.DataArray(np.asarray(values, dtype=float).reshape(3, 2),
                        dims=('lat', 'lon'), coords={'lat': [-90., 0., 90.], 'lon': [0., 180.]})


def test_acc_analytic_and_no_extra_spatial_centering():
    truth = field([1, 2, 3, 4, 5, 6])
    climate = xr.ones_like(truth) * 10
    assert float(spatial_acc(truth, truth, climate)) == pytest.approx(1)
    assert float(spatial_acc(climate - 2*(truth-climate), truth, climate)) == pytest.approx(-1)
    forecast = truth + 2
    weights = losses.normalized_latitude_weights(truth).values[:, None]
    fa, ta = (forecast-climate).values, (truth-climate).values
    expected = np.sum(fa*ta*weights) / np.sqrt(np.sum(fa**2*weights)*np.sum(ta**2*weights))
    assert float(spatial_acc(forecast, truth, climate)) == pytest.approx(expected)
    assert expected < 0.999  # Spatial Pearson centering would incorrectly give 1.
    assert np.isnan(spatial_acc(climate, truth, climate))
    with pytest.raises(ValueError):
        spatial_acc(truth, truth.assign_coords(lon=[1., 181.]), climate)


def test_rmse_and_acc_aggregation_distinct():
    mse = xr.Dataset({'a': ('initialization', [1., 9.])})
    acc = xr.Dataset({'a': ('initialization', [1., -0.5])})
    error, score = aggregate(mse, acc)
    assert float(error.a) == pytest.approx(np.sqrt(5))
    assert float(score.a) == pytest.approx(.25)
    ref = field([0]*6)
    prediction = field([2]*6)
    assert float(spatial_mse(prediction, ref)) == pytest.approx(4)


def test_climatology_valid_time_leap_mapping_and_exact_levels():
    # Valid dates, not initialization dates: midnight after Feb 28 differs by leap year.
    coords = dict(dayofyear=[59, 60, 61], hour=[0, 6, 12, 18], level=[500, 850],
                  latitude=[-90., 0., 90.], longitude=[0., 180.])
    values = np.broadcast_to(np.array([59., 60., 61.])[:, None, None, None, None], (3, 4, 2, 3, 2))
    source = xr.Dataset({'temperature': (tuple(coords), values)}, coords=coords)
    times = np.array(['2020-02-29T00', '2021-03-01T00', '2020-03-01T06'], dtype='datetime64[ns]')
    selected = select_climatology(source, times, [850], [-90., 0., 90.], [0., 180.], ['temperature'])
    np.testing.assert_array_equal(selected.temperature[:, 0, 0, 0].values, [60, 60, 61])
    np.testing.assert_array_equal(selected.time.values, times)
    assert selected.sizes['level'] == 1
    with pytest.raises(KeyError):
        select_climatology(source, times, [700], [0.], [0.], ['temperature'])
