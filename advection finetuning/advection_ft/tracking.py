"""Coarse-grid adaptation of Aurora's MSL/Z700 heuristic; no future truth input."""

import numpy as np
from scipy.ndimage import minimum_filter

from .metrics import haversine_km, wrap_delta


def refine_minimum(field, iy, ix, lat, lon):
    h, w = field.shape
    def offset(left, mid, right):
        denominator = left - 2*mid + right
        if denominator <= 0 or not np.isfinite(denominator):
            return 0.
        return float(np.clip(0.5*(left-right)/denominator, -0.5, 0.5))
    dy = offset(field[iy-1, ix], field[iy, ix], field[iy+1, ix]) if 0 < iy < h-1 else 0.
    dx = offset(field[iy, (ix-1)%w], field[iy, ix], field[iy, (ix+1)%w])
    return float(np.clip(lat[iy]+dy*(lat[1]-lat[0]), -90, 90)), float((lon[ix]+dx*(lon[1]-lon[0]))%360)


class CycloneTracker:
    def __init__(self, lat, lon, start_lat, start_lon, search_radius=5.0):
        self.lat, self.lon = np.asarray(lat), np.asarray(lon)
        self.positions = [(float(start_lat), float(start_lon)%360)]
        self.radius = search_radius

    def step(self, msl, z700):
        previous = self.positions[-1]
        if len(self.positions) >= 2:
            recent = self.positions[-min(8, len(self.positions)):]
            coords = np.asarray(recent)
            unwrapped = np.rad2deg(np.unwrap(np.deg2rad(coords[:, 1])))
            t = np.arange(len(coords))
            guess = (float(np.clip(np.polyval(np.polyfit(t, coords[:, 0], 1), len(t)), -89.25, 89.25)),
                     float(np.polyval(np.polyfit(t, unwrapped, 1), len(t)))%360)
        else:
            guess = previous
        yy, xx = np.meshgrid(self.lat, self.lon, indexing="ij")
        inside = (np.abs(yy-guess[0]) <= self.radius) & (np.abs(wrap_delta(xx-guess[1])) <= self.radius)
        for field in (msl, z700):
            minimum = minimum_filter(field, size=3, mode=("nearest", "wrap"))
            mask = inside & np.isfinite(field) & (field == minimum)
            iy, ix = np.where(mask)
            if len(iy):
                # Reject minima at the search boundary, as in the paper's heuristic.
                interior = ((np.abs(self.lat[iy]-guess[0]) < self.radius-0.75)
                            & (np.abs(wrap_delta(self.lon[ix]-guess[1])) < self.radius-0.75))
                iy, ix = iy[interior], ix[interior]
                if len(iy):
                    distances = haversine_km(guess[0], guess[1], self.lat[iy], self.lon[ix])
                    selected = np.argmin(distances)
                    point = refine_minimum(field, iy[selected], ix[selected], self.lat, self.lon)
                    self.positions.append(point)
                    return point, False
        # Persist the previous prediction on failure and keep it in scoring.
        self.positions.append(previous)
        return previous, True
