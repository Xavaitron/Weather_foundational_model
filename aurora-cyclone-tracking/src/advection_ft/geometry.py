"""Scalar transport on a symmetric, cell-centred latitude/longitude lattice.

Coordinates are degrees. Velocities are eastward/northward m/s. The regular
1.5-degree grid uses latitude 89.25..-89.25 and longitude 0.75..359.25.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

EARTH_RADIUS = 6_371_000.0


def sphere_pad(x, pad=1):
    if pad == 0:
        return x
    h, w = x.shape[-2:]
    if w % 2 or h < pad:
        raise ValueError("Polar padding requires even longitude count and H >= pad.")
    north = x[..., :pad, :].flip(-2).roll(w // 2, -1)
    south = x[..., -pad:, :].flip(-2).roll(w // 2, -1)
    return F.pad(torch.cat((north, x, south), -2), (pad, pad, 0, 0), mode="circular")


class SphereConv(nn.Conv2d):
    def __init__(self, cin, cout, kernel_size=3, **kwargs):
        super().__init__(cin, cout, kernel_size, padding=0, **kwargs)
        self.sphere_padding = kernel_size // 2

    def forward(self, x):
        return super().forward(sphere_pad(x, self.sphere_padding))


def check_grid(lat, lon):
    if lat.numel() < 2 or lon.numel() < 4 or lon.numel() % 2:
        raise ValueError("Need H>=2 and an even W>=4.")
    dlat, dlon = lat[0] - lat[1], lon[1] - lon[0]
    if not torch.allclose(lat[:-1] - lat[1:], dlat.expand_as(lat[1:]), atol=1e-3):
        raise ValueError("Latitude must be regular and descending.")
    if not torch.allclose(lon[1:] - lon[:-1], dlon.expand_as(lon[1:]), atol=1e-3):
        raise ValueError("Longitude must be regular and ascending.")
    if abs(float(dlon * len(lon)) - 360) > 1e-2:
        raise ValueError("Transport requires a full global longitude grid.")
    if abs(float(lat[0] + dlat / 2) - 90) > 1e-2 or abs(float(lat[-1] - dlat / 2) + 90) > 1e-2:
        raise ValueError("Use the prepared symmetric cell-centred global grid.")


def displaced_coordinates(lat, lon, velocity, seconds, direction=-1):
    """Exponential-map trajectory with a frozen local tangent velocity."""
    phi, lam = torch.meshgrid(torch.deg2rad(lat.float()), torch.deg2rad(lon.float()), indexing="ij")
    cp, sp, cl, sl = phi.cos(), phi.sin(), lam.cos(), lam.sin()
    r = torch.stack((cp * cl, cp * sl, sp), 0)[None]
    east = torch.stack((-sl, cl, torch.zeros_like(sl)), 0)[None]
    north = torch.stack((-sp * cl, -sp * sl, cp), 0)[None]
    tangent = (east * velocity[:, :1].float() + north * velocity[:, 1:2].float())
    tangent = tangent * (direction * seconds / EARTH_RADIUS)
    theta = (tangent.square().sum(1, keepdim=True) + 1e-16).sqrt()
    moved = r * theta.cos() + tangent * torch.sinc(theta / math.pi)
    plat = torch.rad2deg(torch.atan2(moved[:, 2], moved[:, :2].square().sum(1).clamp_min(1e-16).sqrt()))
    plon = torch.rad2deg(torch.atan2(moved[:, 1], moved[:, 0])).remainder(360)
    return plat, plon


def fractional_indices(lat, lon, plat, plon):
    y = (lat[0] - plat) / (lat[0] - lat[1])
    x = ((plon - lon[0]) / (lon[1] - lon[0])).remainder(len(lon))
    return y, x


def pull(x, velocity, lat, lon, seconds, mode="bilinear"):
    plat, plon = displaced_coordinates(lat, lon, velocity, seconds)
    y, col = fractional_indices(lat, lon, plat, plon)
    pad = 2
    h, w = x.shape[-2:]
    grid = torch.stack((2 * (col + pad) / (w + 2 * pad - 1) - 1,
                        2 * (y + pad) / (h + 2 * pad - 1) - 1), -1)
    # grid_sample requires equal dtypes; float32 also avoids coarse BF16 coordinates.
    with torch.autocast(device_type=x.device.type, enabled=False):
        sampled = F.grid_sample(sphere_pad(x.float(), pad), grid.float(), mode=mode,
                                padding_mode="border", align_corners=True)
    return sampled.to(x.dtype)


def fold_indices(y, x, h, w):
    """Map integer scalar sample locations through polar reflection and wrapping."""
    folded = y.remainder(2 * h)
    reflected = folded >= h
    iy = torch.where(reflected, 2 * h - 1 - folded, folded)
    ix = (x + reflected.to(x.dtype) * (w // 2)).remainder(w)
    return iy.long(), ix.long()


def gather(x, y, col):
    n, c, h, w = x.shape
    iy, ix = fold_indices(y, col, h, w)
    indices = (iy * w + ix).reshape(n, 1, -1).expand(-1, c, -1)
    return x.flatten(2).gather(2, indices).reshape(n, c, h, w)


def gaussian_pull(x, velocity, lat, lon, seconds, sigma, radius=3):
    """Truncated, normalized Gaussian around a continuous departure location.

    Sigma is in latent-grid cells (a deliberate adaptation of the planar paper).
    Gaussian weights, rather than grid_sample, carry motion gradients.
    """
    plat, plon = displaced_coordinates(lat, lon, velocity, seconds)
    y, col = fractional_indices(lat, lon, plat, plon)
    by, bx = y.floor(), col.floor()
    sigma = sigma.float().reshape(1, -1, 1, 1)
    result, total = torch.zeros_like(x.float()), torch.zeros_like(x.float())
    for j in range(-radius, radius + 1):
        for i in range(-radius, radius + 1):
            yy, xx = by + j, bx + i
            weight = torch.exp(-((y[:, None] - yy[:, None]).square()
                                 + (col[:, None] - xx[:, None]).square()) / (2 * sigma.square()))
            result = result + weight * gather(x.float(), yy, xx)
            total = total + weight
    return (result / total.clamp_min(1e-12)).to(x.dtype)


def push(x, velocity, lat, lon, seconds):
    """Forward bilinear splat preserving the area-weighted scalar integral.

    This conserves the transported latent scalar, not physical atmospheric mass.
    Scatter accumulation may be nondeterministic on CUDA.
    """
    n, c, h, w = x.shape
    plat, plon = displaced_coordinates(lat, lon, velocity, seconds, direction=1)
    y, col = fractional_indices(lat, lon, plat, plon)
    by, bx = y.floor(), col.floor()
    fy, fx = y - by, col - bx
    area = torch.deg2rad(lat.float()).cos()[None, None, :, None].clamp_min(1e-8)
    mass = x.float() * area
    out = torch.zeros_like(mass).flatten(2)
    for dy, dx, weight in [(0, 0, (1-fy)*(1-fx)), (0, 1, (1-fy)*fx),
                            (1, 0, fy*(1-fx)), (1, 1, fy*fx)]:
        iy, ix = fold_indices(by + dy, bx + dx, h, w)
        index = (iy * w + ix).reshape(n, 1, -1).expand(-1, c, -1)
        out = out.scatter_add(2, index, (mass * weight[:, None]).flatten(2))
    return (out.reshape(n, c, h, w) / area).to(x.dtype)
