"""Physical-unit, area-weighted forecast verification on exact matched grids."""
import numpy as np
import xarray as xr
from graphcast import losses


def spatial_acc(prediction, reference, climatology):
    """WeatherBench-style uncentered anomaly correlation for each date/lead/level.

    Climatology must already be selected at forecast valid time. A zero anomaly
    norm is undefined and returns NaN; there is no additional spatial centering.
    """
    prediction, reference, climatology = xr.align(
        prediction, reference, climatology, join='exact')
    weight = losses.normalized_latitude_weights(reference).astype(np.float64)
    fa = prediction.astype(np.float64) - climatology.astype(np.float64)
    ta = reference.astype(np.float64) - climatology.astype(np.float64)
    axes = [d for d in ('lat', 'lon', 'batch') if d in fa.dims]
    numerator = (fa * ta * weight).mean(axes, skipna=False)
    denominator = np.sqrt((fa**2 * weight).mean(axes, skipna=False)
                          * (ta**2 * weight).mean(axes, skipna=False))
    return numerator / denominator.where(denominator > 0)


def aggregate(mse, acc):
    """Pool squared errors before sqrt; average per-initialization ACC equally."""
    return (np.sqrt(mse.mean('initialization', skipna=False)),
            acc.mean('initialization', skipna=False))


def select_climatology(source, valid_times, levels, lat, lon, variables):
    """Exact grid/pressure selection and official day-of-year/hour convention.

    This function remains lazy. Use a bounded source opened with chunks=None.
    The reference has 366 day-of-year entries, including the leap-day slot.
    """
    times = xr.DataArray(np.asarray(valid_times, dtype='datetime64[ns]'), dims='time')
    selected = source[list(variables)].sel(
        dayofyear=times.dt.dayofyear, hour=times.dt.hour,
        level=list(levels), latitude=np.asarray(lat), longitude=np.asarray(lon))
    selected = selected.rename(latitude='lat', longitude='lon')
    selected = selected.drop_vars(['dayofyear', 'hour'], errors='ignore')
    return selected.assign_coords(time=times.values)
