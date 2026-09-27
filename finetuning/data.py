"""Bounded ERA5 reads and strictly chronological forecast windows."""
import dataclasses
import numpy as np
import xarray as xr

SOURCE = ('gs://weatherbench2/datasets/era5/'
          '1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr')
SPLITS = {'train': ('2016-01-01', '2020-01-01'),
          'val': ('2020-01-01', '2021-01-01'),
          'test': ('2021-01-01', '2023-01-01')}
ATMOSPHERE = ('temperature', 'geopotential', 'u_component_of_wind',
              'v_component_of_wind', 'vertical_velocity', 'specific_humidity')
SURFACE = ('2m_temperature', 'mean_sea_level_pressure',
           '10m_u_component_of_wind', '10m_v_component_of_wind')
STATIC = ('geopotential_at_surface', 'land_sea_mask')


def valid_initializations(split, steps):
    if steps < 1:
        raise ValueError('Forecast steps must be positive')
    start, stop = map(np.datetime64, SPLITS[split])
    # Keep both states AND all six-hour precipitation history inside the split.
    return np.arange(start + np.timedelta64(12, 'h'),
                     stop - np.timedelta64(6 * steps, 'h'),
                     np.timedelta64(6, 'h'))


def assert_window(initialization, split, steps):
    init = np.datetime64(initialization, 'h')
    candidates = valid_initializations(split, steps)
    if np.datetime64(initialization) != init or not np.any(candidates == init):
        raise ValueError(f'{initialization} with {steps} steps crosses {split} boundaries '
                         'or is not a synoptic initialization')
    return init


def open_source(source=SOURCE):
    options = {'storage_options': {'token': 'anon'}} if source.startswith('gs://') else {}
    data = xr.open_zarr(source, consolidated=True, **options)
    data.attrs['source_store'] = source
    return data


def precipitation_six_hours(precip, times):
    result = []
    for time in times:
        hours = time - np.arange(5, -1, -1) * np.timedelta64(1, 'h')
        # Exact selection fails on a missing hour; skipna=False rejects gaps.
        result.append(precip.sel(time=hours).sum('time', skipna=False))
    output = xr.concat(result, dim=xr.IndexVariable('time', times.astype('datetime64[ns]')))
    output.attrs.update(precip.attrs)
    output.attrs['accumulation_hours'] = 6
    return output


def read_window(source, initialization, split, steps, levels):
    init = assert_window(initialization, split, steps)
    times = init + np.arange(-1, steps + 1) * np.timedelta64(6, 'h')
    dynamic = source[list(ATMOSPHERE + SURFACE)].sel(time=times, level=list(levels))
    dynamic['total_precipitation_6hr'] = precipitation_six_hours(source.total_precipitation, times)
    data = xr.merge([dynamic, source[list(STATIC)]], join='exact')
    data = data.rename({'latitude': 'lat', 'longitude': 'lon'}).sortby('lat')
    # One window only, never .load() on the full source store.
    data = data.compute()
    for name, value in data.data_vars.items():
        if not np.isfinite(value.values).all():
            raise ValueError(f'Missing/non-finite ERA5 data in {name}')
    data.attrs.update(source=source.attrs.get('source_store', 'unspecified'), split=split, initialization_utc=str(init),
                      precipitation='sum of hourly accumulations ending at t-5h,...,t')
    return data


def regrid(data, resolution):
    """Periodic bilinear interpolation onto the requested global grid."""
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('Resolution must be positive and finite')
    nlat, nlon = round(180 / resolution) + 1, round(360 / resolution)
    if not np.isclose((nlat - 1) * resolution, 180):
        raise ValueError('Resolution must divide 180 degrees')
    # Coordinate precision matters: float32 0.1-degree increments fail the
    # official loss's uniform-latitude-spacing check near the poles.
    lat = np.linspace(-90, 90, nlat, dtype=np.float64)
    lon = np.arange(nlon, dtype=np.float64) * resolution
    data = data.assign_coords(lon=np.mod(data.lon, 360)).sortby('lat').sortby('lon')
    if len(np.unique(data.lon)) != data.sizes['lon']:
        raise ValueError('Duplicate longitude endpoint')
    if np.array_equal(data.lat.values, lat) and np.array_equal(data.lon.values, lon):
        return data
    output = xr.Dataset(attrs=data.attrs)
    # Process variables separately to avoid one large concatenated temporary.
    for name, field in data.data_vars.items():
        if not {'lat', 'lon'}.issubset(field.dims):
            output[name] = field
            continue
        wrapped = xr.concat([field.isel(lon=[-1]).assign_coords(lon=[float(data.lon[-1]) - 360]),
                             field,
                             field.isel(lon=[0]).assign_coords(lon=[float(data.lon[0]) + 360])], dim='lon')
        output[name] = wrapped.interp(lat=lat, lon=lon, method='linear').astype(np.float32)
    output.attrs.update(grid_resolution_degrees=resolution,
                        regridding='periodic bilinear interpolation of ERA5; not conservative')
    return output


def model_batch(data, task, steps):
    from graphcast import data_utils
    times = data.time.values
    batch = data.expand_dims(batch=[0]).assign_coords(
        datetime=(('batch', 'time'), times[None, :]), time=times-times[0])
    # Forcings, including solar radiation, are generated by official GraphCast.
    return data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice('6h', f'{steps * 6}h'), **dataclasses.asdict(task))
