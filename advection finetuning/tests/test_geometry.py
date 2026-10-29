import numpy as np
import pytest
import torch

from advection_ft.geometry import displaced_coordinates, gaussian_pull, pull, push
from advection_ft.metrics import haversine_km, paired_comparison
from advection_ft.operators import implicit_grid_heat
from prepare_data import conservative_weights


def grid():
    return torch.arange(82.5, -90, -15), torch.arange(7.5, 360, 15)


@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_zero_motion_identity(mode):
    lat, lon = grid()
    x = torch.randn(2, 3, len(lat), len(lon))
    result = pull(x, torch.zeros(2, 2, len(lat), len(lon)), lat, lon, 21600, mode)
    torch.testing.assert_close(result, x, atol=2e-5, rtol=2e-5)


def test_sphere_seam_and_pole_are_finite_with_gradients():
    lat, lon = grid()
    velocity = torch.full((1, 2, len(lat), len(lon)), 100., requires_grad=True)
    plat, plon = displaced_coordinates(lat, lon, velocity, 86400)
    assert plat.abs().max() <= 90
    assert plon.min() >= 0 and plon.max() < 360
    x = torch.randn(1, 2, len(lat), len(lon))
    result = pull(x, velocity, lat, lon, 86400)
    result.square().mean().backward()
    assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum() > 0


def test_pull_and_gaussian_preserve_constant_with_nonzero_motion():
    lat, lon = grid()
    v = torch.randn(2, 2, len(lat), len(lon))*75
    x = torch.ones(2, 2, len(lat), len(lon))
    torch.testing.assert_close(pull(x, v, lat, lon, 43200), x)
    torch.testing.assert_close(gaussian_pull(x, v, lat, lon, 43200, torch.ones(2)), x)


def test_push_conserves_spherical_area_integral_and_has_motion_gradient():
    lat, lon = grid()
    x = torch.randn(2, 3, len(lat), len(lon))
    v = (torch.randn(2, 2, len(lat), len(lon))*50).requires_grad_()
    area = torch.deg2rad(lat).cos()[None, None, :, None]
    out = push(x, v, lat, lon, 86400)
    torch.testing.assert_close((out*area).sum((-2, -1)), (x*area).sum((-2, -1)), atol=2e-5, rtol=2e-5)
    out.square().mean().backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0


def test_implicit_diffusion_constant_and_energy():
    x = torch.randn(2, 3, 12, 24)
    result = implicit_grid_heat(x, torch.ones(3))
    assert result.square().sum() <= x.square().sum()
    torch.testing.assert_close(implicit_grid_heat(torch.ones_like(x), torch.ones(3)), torch.ones_like(x),
                               atol=1e-5, rtol=1e-5)


def test_conservative_grid_weights_preserve_area_integral():
    lat = np.arange(-90, 90.01, .25)
    lon = np.arange(0, 360, .25)
    wy, wx = conservative_weights(lat, lon)
    np.testing.assert_allclose(wy.sum(1), 1)
    np.testing.assert_allclose(wx.sum(1), 1)
    # Unit field remains unit under both separable operations.
    np.testing.assert_allclose(wy @ np.ones((len(lat), len(lon))) @ wx.T, 1)


def test_haversine_handles_dateline():
    assert haversine_km(0, 179, 0, -179) == pytest.approx(222.38985, rel=1e-6)
    assert haversine_km(20, 20, 20, 20) == 0


def test_comparison_refuses_missing_forecasts():
    import pandas as pd
    frame = pd.DataFrame([dict(sid="a", init_time="0", valid_time="1", lead_hours=24,
                               basin="NA", error_km=100, tracker_failed=False)])
    with pytest.raises(ValueError, match="cases differ"):
        paired_comparison(frame, frame.iloc[:0], [24], repetitions=10)
