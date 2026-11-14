"""Frozen, resumable paired RMSE/ACC evaluation of the two stage-1 checkpoints.

Freeze writes a protocol without reading evaluation weather. Run consumes that
protocol, verifies checkpoint hashes, and shares every weather window between
the two model variants. Published pre-2020 climatology is used only for ACC.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time
import haiku as hk
import jax
import numpy as np
import xarray as xr
from graphcast import checkpoint, graphcast
from finetuning.checkpoints import configuration, predictor
from finetuning.data import SOURCE, assert_window, read_window, model_batch
from finetuning.evaluate import spatial_mse
from finetuning.verification import aggregate, spatial_acc, select_climatology
from finetuning.window_cache import cached_window

CLIMATOLOGY = ('gs://weatherbench2/datasets/era5-hourly-climatology/'
               '1990-2019_6h_1440x721.zarr')
CLIMATOLOGY_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
UNITS = {'geopotential': 'm2 s-2', 'temperature': 'K', '2m_temperature': 'K',
         'u_component_of_wind': 'm s-1', 'v_component_of_wind': 'm s-1',
         '10m_u_component_of_wind': 'm s-1', '10m_v_component_of_wind': 'm s-1',
         'vertical_velocity': 'Pa s-1', 'specific_humidity': 'kg kg-1',
         'mean_sea_level_pressure': 'Pa', 'total_precipitation_6hr': 'm'}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def load_model(path):
    with Path(path).open('rb') as f:
        return checkpoint.load(f, graphcast.CheckPoint)


def open_evaluation_source(path):
    # gcsfs otherwise passes timeout=None to aiohttp, permitting a stalled
    # object request to block indefinitely. Its existing retry policy handles
    # bounded request failures without changing the selected data.
    options = {'storage_options': {'token': 'anon', 'requests_timeout': 60}} if path.startswith('gs://') else {}
    source = xr.open_zarr(path, consolidated=True, chunks=None, **options)
    source.attrs['source_store'] = path
    return source


def freeze(args):
    if args.protocol.exists():
        raise FileExistsError(args.protocol)
    if args.steps < 1:
        raise ValueError('steps must be positive')
    checkpoints = {}
    models = []
    for variant in ('baseline', 'advection'):
        path = (args.run_root / variant / 'best.npz').resolve()
        model = load_model(path)
        if configuration(model)['variant'] != variant:
            raise ValueError('Checkpoint variant mismatch')
        models.append(model)
        checkpoints[variant] = {'path': str(path), 'sha256': digest(path)}
    if models[0].task_config != models[1].task_config or models[0].model_config != models[1].model_config:
        raise ValueError('Models must have identical task and base model configurations')
    dates = [f'{year}-{month:02d}-15T12' for year in (2020, 2021, 2022)
             for month in range(1, 13)]
    for date in dates:
        assert_window(date, 'val' if date.startswith('2020') else 'test', args.steps)
    protocol = dict(
        format_version=1, frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        checkpoints=checkpoints, source=SOURCE, climatology=CLIMATOLOGY,
        climatology_years=[1990, 2019], climatology_pressure_levels=CLIMATOLOGY_LEVELS,
        resolution=models[0].model_config.resolution,
        model_pressure_levels=list(models[0].task_config.pressure_levels),
        target_variables=list(models[0].task_config.target_variables),
        steps=args.steps, lead_hours=list(range(6, 6*args.steps+1, 6)),
        initializations=dates, sampling='15th of every month at 12 UTC; 12 initializations/year',
        aggregation={'rmse': 'sqrt(mean across initializations of latitude-area-weighted spatial MSE)',
                     'acc': 'mean of per-initialization uncentered area-weighted anomaly correlations'},
        climatology_selection='forecast valid time dayofyear and hour; exact pressure/grid points',
        spatial_weights='GraphCast normalized latitude weights, including polar-cap cell area',
        undefined_acc='NaN for zero anomaly norm; never silently omitted from averages',
        checkpoint_selection='Best 2020 training-validation checkpoint; no tuning using 2021-2022',
        forecast_precision='bfloat16 network with physical-unit outputs',
        benchmark_scope='Sampled single-initialization forecasts; not a full-year or multi-day benchmark')
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.protocol, protocol)
    print(json.dumps(protocol, indent=2), flush=True)


def make_predict(model, stats):
    @hk.transform_with_state
    def forward(inputs, template, forcings):
        return predictor(model, stats)(inputs, template, forcings)
    return jax.jit(forward.apply)


def load_climatology(source, init, targets, protocol, folder):
    valid = np.datetime64(init, 'ns') + targets.time.values
    clock = xr.DataArray(valid, dims='time')
    identity = dict(format_version=1, source=protocol['climatology'],
                    days=clock.dt.dayofyear.values.tolist(), hours=clock.dt.hour.values.tolist(),
                    levels=protocol['climatology_pressure_levels'],
                    resolution=protocol['resolution'], variables=list(targets.data_vars))

    def producer():
        data = select_climatology(source, valid, identity['levels'],
                                  targets.lat, targets.lon, identity['variables'])
        # Variables are independent lazy backend arrays, not the full archive.
        # Only loading is threaded; NetCDF cache access stays on the main thread.
        with ThreadPoolExecutor(max_workers=4) as pool:
            loaded = list(pool.map(lambda n: (n, data[n].load()), data.data_vars))
        result = xr.Dataset(dict(loaded)).assign_coords(time=targets.time.values)
        for name in result:
            if not np.isfinite(result[name].values).all():
                raise ValueError(f'Non-finite climatology: {name}')
        return result

    return cached_window(folder, identity, producer, max_bytes=8*1024**3).assign_coords(time=targets.time.values)


def metric_paths(output, variant, date):
    prefix = date.replace(':', '').replace('-', '')
    folder = output / 'per_initialization' / variant
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (prefix + '.mse.nc'), folder / (prefix + '.acc.nc')


def save_metrics(path, dataset):
    temp = path.with_suffix('.tmp')
    dataset.to_netcdf(temp)
    temp.replace(path)


def summarize(output, protocol, completed):
    rows = []
    for variant in protocol['checkpoints']:
        for group in ('2020', '2021', '2022', '2021-2022'):
            dates = [d for d in completed if d[:4] == group or
                     (group == '2021-2022' and d[:4] in ('2021', '2022'))]
            if not dates:
                continue
            mse, acc = [], []
            for date in dates:
                paths = metric_paths(output, variant, date)
                mse.append(xr.load_dataset(paths[0]))
                acc.append(xr.load_dataset(paths[1]))
            axis = xr.IndexVariable('initialization', np.array(dates, dtype='datetime64[ns]'))
            rmse, averaged_acc = aggregate(xr.concat(mse, dim=axis), xr.concat(acc, dim=axis))
            for ds, metric in ((rmse, 'rmse'), (averaged_acc, 'acc')):
                ds.attrs.update(model=variant, period=group, count=len(dates),
                                aggregation=protocol['aggregation'][metric],
                                scope='sampled monthly dates; all configured dates required for final results')
                for name in ds:
                    ds[name].attrs['units'] = UNITS[name] if metric == 'rmse' else '1'
                save_metrics(output / f'{variant}-{group}-{metric}.nc', ds)
            for name in rmse:
                for lead in rmse.time.values:
                    field = rmse[name].sel(time=lead)
                    levels = field.level.values.tolist() if 'level' in field.dims else [None]
                    for level in levels:
                        error = float(field.sel(level=level) if level is not None else field)
                        correlation = ''
                        if level is None or level in protocol['climatology_pressure_levels']:
                            score = averaged_acc[name].sel(time=lead)
                            correlation = float(score.sel(level=level) if level is not None else score)
                        rows.append(dict(model=variant, period=group, initializations=len(dates),
                                         variable=name, level_hpa='' if level is None else level,
                                         lead_hours=int(lead / np.timedelta64(1, 'h')),
                                         rmse=error, units=UNITS[name], acc=correlation))
    if rows:
        temp = output / 'scores.tmp'
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(output / 'scores.csv')


def run(args):
    protocol = json.loads(args.protocol.read_text())
    if protocol['format_version'] != 1:
        raise ValueError('Unsupported protocol')
    protocol_hash = digest(args.protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    status_file = args.output / 'run.json'
    previous = json.loads(status_file.read_text()) if status_file.exists() else {}
    if previous and previous['protocol_sha256'] != protocol_hash:
        raise ValueError('Output already belongs to another protocol')
    write_json(args.output / 'protocol.json', protocol)
    status = dict(status='running', protocol_sha256=protocol_hash, completed_initializations=[],
                  requested_initializations=len(protocol['initializations']),
                  started_at_utc=previous.get('started_at_utc', time.time()),
                  attempt=previous.get('attempt', 1 if previous else 0)+1,
                  resumed_at_utc=time.time())
    write_json(status_file, status)
    models, predictions = {}, {}
    try:
        stats = {n: xr.load_dataset(args.stats_dir / (n + '.nc')) for n in
                 ('mean_by_level', 'stddev_by_level', 'diffs_stddev_by_level')}
        for variant, entry in protocol['checkpoints'].items():
            if digest(entry['path']) != entry['sha256']:
                raise ValueError(f'Frozen checkpoint changed: {variant}')
            models[variant] = model = load_model(entry['path'])
            if configuration(model)['variant'] != variant:
                raise ValueError('Variant mismatch')
            predictions[variant] = make_predict(model, stats)
        model = models['baseline']
        source = open_evaluation_source(protocol['source'])
        climate = open_evaluation_source(protocol['climatology'])
        # Fail on unsupported variables or coordinates before forecast data access.
        select_climatology(climate, [np.datetime64(protocol['initializations'][0])],
                           protocol['climatology_pressure_levels'], [-90, 0, 90], [0],
                           protocol['target_variables'])
        for date in protocol['initializations']:
            before = time.monotonic()
            status.update(current_initialization=date, phase='weather')
            write_json(status_file, status)
            paths = {v: metric_paths(args.output, v, date) for v in models}
            if not all(p.exists() for pair in paths.values() for p in pair):
                init = np.datetime64(date, 'h')
                split = 'val' if date.startswith('2020') else 'test'
                identity = dict(format_version=1, source=protocol['source'], split=split,
                                initialization=str(init), steps=protocol['steps'],
                                levels=protocol['model_pressure_levels'], resolution=protocol['resolution'])
                data = cached_window(args.weather_cache, identity,
                                     lambda: read_window(source, init, split, protocol['steps'],
                                                         identity['levels'], identity['resolution']))
                inputs, targets, forcings = model_batch(data, model.task_config, protocol['steps'])
                status.update(phase='climatology', weather_seconds=time.monotonic()-before)
                write_json(status_file, status)
                climo = load_climatology(climate, init, targets, protocol, args.climatology_cache)
                for variant, trained in models.items():
                    if all(p.exists() for p in paths[variant]):
                        continue
                    status.update(phase='inference', current_model=variant)
                    write_json(status_file, status)
                    forecast, _ = predictions[variant](trained.params, {}, jax.random.PRNGKey(0),
                                                       inputs, xr.zeros_like(targets), forcings)
                    # Bring physical fields to CPU before float64 reductions.
                    forecast = jax.device_get(forecast)
                    mse, acc = {}, {}
                    for name in targets:
                        mse[name] = spatial_mse(forecast[name], targets[name])
                        if not np.isfinite(mse[name].values).all():
                            raise FloatingPointError(f'Non-finite MSE: {variant}/{name}')
                        f, t = forecast[name], targets[name]
                        if 'level' in t.dims:
                            f, t = f.sel(level=climo.level), t.sel(level=climo.level)
                        acc[name] = spatial_acc(f, t, climo[name])
                        if not np.isfinite(acc[name].values).all():
                            raise FloatingPointError(f'Undefined ACC: {variant}/{name}; inspect anomaly norms')
                    save_metrics(paths[variant][0], xr.Dataset(mse))
                    save_metrics(paths[variant][1], xr.Dataset(acc))
                    del forecast
            status['completed_initializations'].append(date)
            summarize(args.output, protocol, status['completed_initializations'])
            status.update(phase='date_complete', last_date_seconds=time.monotonic()-before)
            write_json(status_file, status)
            print(f'Completed {date}: {len(status["completed_initializations"])}/'
                  f'{len(protocol["initializations"])} paired dates in {time.monotonic()-before:.1f}s', flush=True)
        status.update(status='complete', phase='complete', finished_at_utc=time.time())
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(status_file, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, default=Path('runs/full-1deg-stage1-20260927'))
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--stats-dir', type=Path, default=Path('data/graphcast/stats'))
    parser.add_argument('--weather-cache', type=Path, default=Path('work/era5-1deg-window-cache'))
    parser.add_argument('--climatology-cache', type=Path, default=Path('work/era5-1deg-climatology-cache'))
    parser.add_argument('--output', type=Path, default=Path('runs/paired-evaluation-20260928'))
    args = parser.parse_args()
    (freeze if args.action == 'freeze' else run)(args)


if __name__ == '__main__':
    main()
