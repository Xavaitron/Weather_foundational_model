"""Research adapter inspired by arXiv:2601.21151v3, not a PARADIS reproduction.

Mesh features -> compressed regular-grid modes -> learned spherical departure
points -> bicubic samples -> mesh residual. IDW transfer and the post-processor
insertion are project-specific choices. All sampling arithmetic is float32.
"""
from functools import lru_cache
import haiku as hk
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from graphcast import graphcast
from finetuning.geometry import CachedGraphCast


def cubic_weight(x):
    # Keys cubic convolution, a=-0.75, as in PyTorch bicubic grid_sample.
    a = -0.75
    x = jnp.abs(x)
    return jnp.where(x <= 1, (a + 2)*x**3-(a + 3)*x**2+1,
                     jnp.where(x < 2, a*x**3-5*a*x**2+8*a*x-4*a, 0))


def sample_sphere(field, latitude, longitude):
    """Sample [B,H,W,C] at [B,N,C] radians, with periodic/pole continuation."""
    b, h, w, c = field.shape
    field = field.at[:, 0].set(jnp.mean(field[:, 0], axis=1, keepdims=True))
    field = field.at[:, -1].set(jnp.mean(field[:, -1], axis=1, keepdims=True))
    y = (latitude + jnp.pi/2) * ((h-1)/jnp.pi)
    x = jnp.mod(longitude, 2*jnp.pi) * (w/(2*jnp.pi))
    iy, ix = jnp.floor(y).astype(jnp.int32), jnp.floor(x).astype(jnp.int32)
    result = jnp.zeros_like(x)
    batch = jnp.arange(b)[:, None, None]
    channel = jnp.arange(c)[None, None, :]
    for dy in (-1, 0, 1, 2):
        yy = iy + dy
        crossed = (yy < 0) | (yy >= h)
        yy_reflected = jnp.where(yy < 0, -yy, jnp.where(yy >= h, 2*(h-1)-yy, yy))
        for dx in (-1, 0, 1, 2):
            xx = jnp.mod(ix + dx + crossed*(w//2), w)
            result += (field[batch, yy_reflected, xx, channel]
                       * cubic_weight(y-(iy+dy)) * cubic_weight(x-(ix+dx)))
    return result


def departure(lat, lon, east, north):
    """Local angular displacements already include the learned substep duration."""
    phi, lam = -north, -east
    sinlat = jnp.sin(phi)*jnp.cos(lat) + jnp.cos(phi)*jnp.cos(lam)*jnp.sin(lat)
    # Mesh nodes avoid exact poles; clipping protects extreme learned displacements.
    outlat = jnp.arcsin(jnp.clip(sinlat, -1+1e-7, 1-1e-7))
    outlon = lon + jnp.arctan2(jnp.cos(phi)*jnp.sin(lam),
                             jnp.cos(phi)*jnp.cos(lam)*jnp.cos(lat)-jnp.sin(phi)*jnp.sin(lat))
    return outlat, jnp.mod(outlon, 2*jnp.pi)


@lru_cache(maxsize=8)
def transfer(vertices_bytes, count, resolution):
    vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(count, 3)
    h, w = round(180/resolution)+1, round(360/resolution)
    lat, lon = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, h),
                          np.arange(w)*2*np.pi/w, indexing='ij')
    xyz = np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], -1)
    distance, indices = cKDTree(vertices).query(xyz.reshape(-1, 3), k=3)
    weight = 1/np.maximum(distance, 1e-10)**2
    weight /= weight.sum(axis=-1, keepdims=True)
    return indices, weight.astype(np.float32), h, w


class AdvectedGraphCast(CachedGraphCast):
    def __init__(self, model_config, task_config, modes=16, adapter_resolution=1.0):
        if modes < 1 or not np.isfinite(adapter_resolution) or adapter_resolution <= 0:
            raise ValueError('Adapter modes and resolution must be positive')
        width = round(360/adapter_resolution)
        if width < 4 or width % 2 or not np.isclose(width*adapter_resolution,360):
            raise ValueError('Adapter grid needs an even longitude count, at least four, dividing 360 degrees')
        super().__init__(model_config, task_config)
        self.modes = modes
        self.adapter_resolution = adapter_resolution

    def _run_mesh_gnn(self, latent_mesh_nodes):
        hidden = super()._run_mesh_gnn(latent_mesh_nodes)
        vertices = np.asarray(self._finest_mesh.vertices, dtype=np.float32)
        indices, weights, h, w = transfer(vertices.tobytes(), len(vertices), self.adapter_resolution)
        # GraphCast features have [nodes,batch,channels] layout.
        features = jnp.swapaxes(hidden, 0, 1).astype(jnp.float32)
        z = hk.Linear(self.modes, name='advection_project')(features)
        velocity = hk.Linear(2*self.modes, w_init=hk.initializers.RandomNormal(0.01),
                             name='advection_velocity')(features)
        east, north = jnp.split(0.15*jnp.tanh(velocity), 2, axis=-1)
        gridded = jnp.sum(z[:, indices, :] * jnp.asarray(weights)[None, :, :, None], axis=2)
        gridded = gridded.reshape(features.shape[0], h, w, self.modes)
        lat = jnp.asarray(np.arcsin(np.clip(vertices[:, 2], -1, 1)))[None, :, None]
        lon = jnp.asarray(np.arctan2(vertices[:, 1], vertices[:, 0]))[None, :, None]
        lat0, lon0 = jnp.broadcast_arrays(lat+jnp.zeros_like(east), lon+jnp.zeros_like(east))
        latd, lond = departure(lat, lon, east, north)
        # Subtract the same mesh-grid-mesh transfer at zero displacement so the
        # residual does not introduce smoothing merely through grid conversion.
        delta = sample_sphere(gridded, latd, lond) - sample_sphere(gridded, lat0, lon0)
        correction = hk.Linear(features.shape[-1], w_init=jnp.zeros, with_bias=False,
                               name='advection_lift')(delta)
        return hidden + jnp.swapaxes(correction, 0, 1).astype(hidden.dtype)
