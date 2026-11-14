"""Independent implementations of paper-inspired transport mechanisms.

These are Aurora adapters, not the authors' original models or checkpoints.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import SphereConv, gaussian_pull, pull, push


class MotionNet(nn.Module):
    """Small convolution/downsample/upsample network with a skip connection."""
    def __init__(self, channels, groups, width=48, source=True):
        super().__init__()
        self.groups = groups
        self.encoder = nn.Sequential(SphereConv(3 * channels, width), nn.GroupNorm(8, width), nn.SiLU(),
                                     SphereConv(width, width), nn.SiLU())
        self.coarse = nn.Sequential(SphereConv(width, 2 * width), nn.GroupNorm(8, 2 * width), nn.SiLU(),
                                    SphereConv(2 * width, width), nn.SiLU())
        self.decoder = nn.Sequential(SphereConv(2 * width, width), nn.SiLU())
        self.motion = nn.Conv2d(width, 2 * groups, 1)
        self.source = nn.Conv2d(width, channels, 1) if source else None
        nn.init.normal_(self.motion.weight, std=1e-3)
        nn.init.zeros_(self.motion.bias)
        if self.source is not None:
            nn.init.zeros_(self.source.weight)
            nn.init.zeros_(self.source.bias)

    def forward(self, current, previous, context):
        fine = self.encoder(torch.cat((current, previous, context), 1))
        coarse = self.coarse(F.avg_pool2d(fine, 2))
        hidden = self.decoder(torch.cat((fine, F.interpolate(coarse, size=fine.shape[-2:],
                                                            mode="bilinear", align_corners=False)), 1))
        velocity = self.motion(hidden).tanh() * 120.0
        source = self.source(hidden) if self.source is not None else None
        return velocity, source


def implicit_grid_heat(x, diffusivity):
    """Backward-Euler 5-point diffusion; Neumann latitude, periodic longitude.

    Solved exactly in a DCT-II x Fourier basis, in latent-grid units. This is
    ADRNet's grid diffusion idea; it is not the spherical physical Laplacian.
    """
    _, _, h, w = x.shape
    with torch.autocast(device_type=x.device.type, enabled=False):
        k = torch.arange(h, device=x.device, dtype=torch.float32)
        j = torch.arange(h, device=x.device, dtype=torch.float32)
        dct = torch.cos(math.pi / h * k[:, None] * (j[None] + 0.5)) * math.sqrt(2 / h)
        dct[0] *= 1 / math.sqrt(2)
        transformed = torch.einsum("kh,nchw->nckw", dct, x.float())
        transformed = torch.fft.rfft(transformed, dim=-1)
        eig = 4 * (math.pi * k / (2 * h)).sin().square()[:, None]
        eig = eig + 4 * (math.pi * torch.arange(w // 2 + 1, device=x.device) / w).sin().square()[None]
        transformed = transformed / (1 + diffusivity.float()[None, :, None, None] * eig)
        result = torch.fft.irfft(transformed, n=w, dim=-1)
        result = torch.einsum("kh,nckw->nchw", dct, result)
    return result.to(x.dtype)


class TransportOperator(nn.Module):
    def __init__(self, approach, channels=32, groups=4, substeps=1, adr_transport="push"):
        super().__init__()
        if channels % groups:
            raise ValueError("Adapter channels must be divisible by flow groups.")
        self.approach, self.groups, self.substeps = approach, groups, substeps
        self.adr_transport = adr_transport
        self.net = MotionNet(channels, groups, source=approach != "de_bezenac")
        if approach == "paradis":
            self.diffusion = SphereConv(channels, channels, groups=channels, bias=False)
            nn.init.zeros_(self.diffusion.weight)
            self.reaction = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1), nn.SiLU(),
                                          nn.Conv2d(2 * channels, channels, 1))
            nn.init.zeros_(self.reaction[-1].weight)
            nn.init.zeros_(self.reaction[-1].bias)
        if approach == "adrnet":
            self.log_kappa = nn.Parameter(torch.full((channels,), -4.0))
        if approach == "de_bezenac":
            self.log_sigma = nn.Parameter(torch.full((channels // groups,), -0.5))

    def _transport(self, c, velocity, lat, lon, seconds, method):
        n, channels, h, w = c.shape
        grouped = c.reshape(n * self.groups, channels // self.groups, h, w)
        velocity = velocity.reshape(n * self.groups, 2, h, w)
        if method == "push":
            out = push(grouped, velocity, lat, lon, seconds)
        elif method == "gaussian":
            out = gaussian_pull(grouped, velocity, lat, lon, seconds,
                                F.softplus(self.log_sigma) + 0.15)
        else:
            out = pull(grouped, velocity, lat, lon, seconds, method)
        return out.reshape(n, channels, h, w)

    def forward(self, current, previous, context, lat, lon, seconds):
        velocity, source = self.net(current, previous, context)
        # Weak amplitude and local smoothness penalties; no incompressibility constraint.
        regularizer = (velocity.square().mean() / 120**2
                       + (velocity[..., 1:, :] - velocity[..., :-1, :]).square().mean() / 120**2
                       + (velocity - velocity.roll(1, -1)).square().mean() / 120**2)
        auxiliary = None
        if self.approach == "nowcastnet":
            # Main path uses nearest interpolation; bilinear path trains motion.
            # The caller applies a separate decoded forecast loss to the auxiliary path.
            out = self._transport(current, velocity, lat, lon, seconds, "nearest")
            auxiliary = self._transport(current, velocity, lat, lon, seconds, "bilinear")
            return out + source, auxiliary + source, regularizer
        if self.approach == "de_bezenac":
            return self._transport(current, velocity, lat, lon, seconds, "gaussian"), None, regularizer
        state = current
        for _ in range(self.substeps):
            if self.approach == "paradis":
                state = self._transport(state, velocity, lat, lon, seconds / self.substeps, "bicubic")
                state = state + (self.diffusion(state) + self.reaction(state) + source) / self.substeps
            elif self.approach == "adrnet":
                state = self._transport(state, velocity, lat, lon, seconds / self.substeps,
                                        "push" if self.adr_transport == "push" else "bilinear")
                state = implicit_grid_heat(state, F.softplus(self.log_kappa) / self.substeps)
                state = state + source / self.substeps
            else:
                raise ValueError(f"Unknown transport approach {self.approach}")
        return state, auxiliary, regularizer
