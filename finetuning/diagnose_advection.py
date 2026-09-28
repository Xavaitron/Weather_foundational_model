"""Read-only displacement, BF16 correction and direct-fine-tuning audit.

Uses the frozen 2020 monthly validation dates only. Initial adapter parameters
are replayed from the training seed and initialization code, not a saved epoch-0
checkpoint. No optimizer updates or checkpoint mutations are performed.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import time
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from graphcast import icosahedral_mesh
from finetuning.advection import departure
from finetuning.checkpoints import predictor
from finetuning.data import read_window, model_batch
from finetuning.evaluate import spatial_mse
from finetuning.paired_evaluate import (load_model, make_predict, digest, write_json,
    open_evaluation_source, load_climatology, save_metrics, summarize)
from finetuning.train import transformed_loss
from finetuning.verification import spatial_acc
from finetuning.window_cache import cached_window


def initial_parameters(original, trained, seed):
    """Replay the sorted dictionary order produced by jax.eval_shape in train.py."""
    rng = np.random.default_rng(seed)
    params = {}
    for module in sorted(trained.params):
        if module in original.params:
            params[module] = {k:original.params[module][k] for k in sorted(original.params[module])}
            continue
        if module not in ('advection_lift', 'advection_project', 'advection_velocity'):
            raise ValueError(f'Unexpected new module {module}')
        params[module] = {}
        for name in sorted(trained.params[module]):
            shape = trained.params[module][name].shape
            if name == 'b' or module.startswith('advection_lift'):
                value = np.zeros(shape, np.float32)
            else:
                scale = .01 if module.startswith('advection_velocity') else 1/np.sqrt(shape[0])
                value = rng.normal(0, scale, shape).astype(np.float32)
            params[module][name] = value
    return params


def parameter_changes(before, after):
    rows = []
    for module in after:
        a = np.concatenate([np.asarray(before[module][k], dtype=np.float64).ravel() for k in sorted(after[module])])
        b = np.concatenate([np.asarray(after[module][k], dtype=np.float64).ravel() for k in sorted(after[module])])
        na, nb, nd = map(float, [np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(b-a)])
        rows.append(dict(module=module, count=a.size, changed=int(np.count_nonzero(a != b)),
                         before_l2=na, after_l2=nb, delta_l2=nd,
                         relative_delta=nd/na if na else None, max_abs_delta=float(np.max(np.abs(b-a)))))
    initial_norm = np.sqrt(sum(r['before_l2']**2 for r in rows))
    difference_norm = np.sqrt(sum(r['delta_l2']**2 for r in rows))
    return dict(parameters=sum(r['count'] for r in rows), changed=sum(r['changed'] for r in rows),
                changed_modules=sum(r['changed'] > 0 for r in rows), modules=len(rows),
                delta_l2=float(difference_norm), relative_delta=float(difference_norm/initial_norm), per_module=rows)


def make_diagnostics(model, stats, initial_velocity):
    vertices = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(model.model_config.mesh_size)[-1].vertices
    latitude = np.arcsin(np.clip(vertices[:, 2], -1, 1)).astype(np.float32)
    longitude = np.arctan2(vertices[:, 1], vertices[:, 0]).astype(np.float32)
    @hk.transform_with_state
    def observe(inputs, template, forcings):
        captured = {}
        def intercept(next_fun, args, kwargs, context):
            output = next_fun(*args, **kwargs)
            name = context.module.module_name
            if context.method_name == '__call__' and name == 'advection_velocity':
                captured['features'] = args[0]
                captured['velocity'] = output
            elif context.method_name == '__call__' and name == 'advection_lift':
                captured['delta'] = args[0]
                captured['correction'] = output
            elif context.method_name == '__call__' and name == 'mesh2grid_gnn':
                captured['summed'] = jnp.swapaxes(args[0].nodes['mesh_nodes'].features, 0, 1)
            return output
        with hk.intercept_methods(intercept):
            if template.sizes['time'] != 1:
                raise ValueError('Latent diagnostics support exactly one forecast step')
            # Call the same normalized one-step network inside the rollout.
            # Captured intermediates cannot escape an autoregressive hk.scan.
            predictor(model, stats)._predictor(inputs, template, forcings)
        features = captured['features']
        correction = captured['correction']
        # Capture the actual decoder input. Return BF16 arrays themselves so
        # host-side comparisons observe stored precision, not an XLA-widened
        # intermediate feeding a fused scalar reduction.
        hidden_bf16 = features.astype(jnp.bfloat16)
        assert captured['summed'].dtype == jnp.bfloat16
        summed = captured['summed'].astype(jnp.float32)
        effective = summed - features
        rms = lambda x: jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32))))
        scalars = dict(hidden_rms=rms(features), transport_delta_rms=rms(captured['delta']),
                       correction_rms=rms(correction), effective_correction_rms=rms(effective),
                       correction_to_hidden_rms=rms(correction)/rms(features),
                       changed_hidden_fraction=jnp.mean(summed != features),
                       nonzero_correction_fraction=jnp.mean(correction != 0))
        reference_velocity = (jnp.dot(features, jnp.asarray(initial_velocity['w']))
                              + jnp.asarray(initial_velocity['b']))
        fields = {}
        for label, velocity in [('current', captured['velocity']), ('initial_velocity_same_features', reference_velocity)]:
            east, north = jnp.split(.15*jnp.tanh(velocity), 2, axis=-1)
            lat = jnp.asarray(latitude)[None, :, None]
            lon = jnp.asarray(longitude)[None, :, None]
            latd, lond = departure(lat, lon, east, north)
            hav = jnp.sin((latd-lat)/2)**2 + jnp.cos(lat)*jnp.cos(latd)*jnp.sin((lond-lon)/2)**2
            distance = 2*6371*jnp.arcsin(jnp.sqrt(jnp.clip(hav, 0, 1)))
            fields[label] = dict(east=east[0], north=north[0], distance_km=distance[0])
        return scalars, fields, {'before':hidden_bf16, 'after':captured['summed']}
    return jax.jit(observe.apply), latitude, longitude


def materialized_scalars(scalars, boundary):
    before = np.asarray(boundary['before'], dtype=np.float32)
    after = np.asarray(boundary['after'], dtype=np.float32)
    result = {k:float(v) for k,v in scalars.items()}
    result['changed_hidden_fraction'] = float(np.mean(before != after))
    result['effective_correction_rms'] = float(np.sqrt(np.mean((after-before)**2, dtype=np.float64)))
    return result


def field_statistics(fields):
    result = {}
    for label, field in fields.items():
        d = np.asarray(field['distance_km'], dtype=np.float64)
        result[label] = dict(mean_km=float(d.mean()), rms_km=float(np.sqrt(np.mean(d*d))),
                             p50_km=float(np.quantile(d, .5)), p95_km=float(np.quantile(d, .95)),
                             p99_km=float(np.quantile(d, .99)), max_km=float(d.max()),
                             saturation_fraction=float(np.mean(np.maximum(np.abs(field['east']), np.abs(field['north'])) > .1425)),
                             per_mode_mean_km=d.mean(axis=0).tolist())
    a = np.concatenate([fields['current'][k].ravel() for k in ('east', 'north')]).astype(float)
    b = np.concatenate([fields['initial_velocity_same_features'][k].ravel() for k in ('east', 'north')]).astype(float)
    result['learned_velocity_effect_same_features'] = dict(
        relative_vector_change=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
        component_correlation=float(np.corrcoef(a,b)[0,1]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=Path('runs/full-1deg-stage1-20260927'))
    p.add_argument('--evaluation-root', type=Path, default=Path('runs/paired-evaluation-20260928'))
    p.add_argument('--output', type=Path, default=Path('runs/advection-diagnostics-20260928'))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = dict(status='preparing', completed_dates=[], started_at_utc=time.time())
    try:
        protocol = json.loads((args.evaluation_root/'protocol.json').read_text())
        metadata = json.loads((args.run_root/'advection/run.json').read_text())
        original = load_model(metadata['checkpoint'])
        assert digest(metadata['checkpoint']) == metadata['checkpoint_sha256']
        baseline = load_model(args.run_root/'baseline/best.npz')
        trained = load_model(args.run_root/'advection/best.npz')
        for variant in ('baseline', 'advection'):
            assert digest(args.run_root/variant/'best.npz') == protocol['checkpoints'][variant]['sha256']
        init_params = initial_parameters(original, trained, metadata['seed'])
        initial = dataclasses.replace(trained, params=init_params)
        pretrained = dataclasses.replace(baseline, params={k:original.params[k] for k in baseline.params})
        off_params = dict(trained.params)
        off_params['advection_lift'] = {k:np.zeros_like(v) for k,v in trained.params['advection_lift'].items()}
        reset_params = dict(trained.params)
        reset_params['advection_velocity'] = init_params['advection_velocity']
        cases = dict(pretrained=pretrained, baseline_trained=baseline, advection_initial=initial,
                     advection_trained=trained, advection_off=dataclasses.replace(trained, params=off_params),
                     advection_reset_velocity=dataclasses.replace(trained, params=reset_params))
        changes = dict(direct_finetuning=parameter_changes(pretrained.params, baseline.params),
                       advection_training=parameter_changes(init_params, trained.params),
                       trained_backbones_difference=parameter_changes(baseline.params,
                           {k:trained.params[k] for k in baseline.params}))
        write_json(args.output/'parameter_changes.json', changes)
        stats = {n:xr.load_dataset(Path(metadata['stats_dir'])/(n+'.nc')) for n in
                 ('mean_by_level', 'stddev_by_level', 'diffs_stddev_by_level')}
        base_predict = make_predict(baseline, stats)
        adv_predict = make_predict(trained, stats)
        diagnose, lat, lon = make_diagnostics(trained, stats, init_params['advection_velocity'])
        source = open_evaluation_source(protocol['source'])
        climate = open_evaluation_source(protocol['climatology'])
        def batch(date):
            init = np.datetime64(date, 'h')
            identity = dict(format_version=1, source=protocol['source'], split='val', initialization=str(init),
                            steps=1, levels=protocol['model_pressure_levels'], resolution=protocol['resolution'])
            data = cached_window(metadata['window_cache'], identity,
                lambda:read_window(source, init, 'val', 1, identity['levels'], identity['resolution']))
            return model_batch(data, trained.task_config, 1)
        manifest.update(status='checking_initialization', settings='12 frozen 2020 dates, 6h; no test data or optimization',
                        initial_parameters='Reconstructed from original pretrained backbone, seed=0 and train.py initializer',
                        checkpoints={v:entry['sha256'] for v,entry in protocol['checkpoints'].items()})
        write_json(args.output/'run.json', manifest)
        # Verify the reconstructed Haiku shape/dictionary order and reproduce
        # the logged epoch-zero loss on the original four validation windows.
        loss = transformed_loss(trained.model_config, trained.task_config, stats, 'advection',
                                metadata['adapter_modes'], metadata['adapter_resolution'])
        first = batch(metadata['validation_initializations_utc'][0])
        shapes, _ = jax.eval_shape(loss.init, jax.random.PRNGKey(metadata['seed']), *first)
        assert list(shapes) == list(init_params)
        for module in shapes:
            assert list(shapes[module]) == list(init_params[module])
            for name in shapes[module]:
                assert shapes[module][name].shape == init_params[module][name].shape
        loss_apply = jax.jit(loss.apply)
        values = []
        for date in metadata['validation_initializations_utc']:
            (value, _), _ = loss_apply(init_params, {}, jax.random.PRNGKey(0), *batch(date))
            values.append(float(value))
        reproduced = float(np.mean(values))
        manifest.update(initial_loss_recorded=metadata['initial_val_loss'], initial_loss_reproduced=reproduced)
        if abs(reproduced-metadata['initial_val_loss']) > .002:
            raise ValueError('Reconstructed initial validation loss differs from the training record')
        del loss_apply
        audit_protocol = dict(protocol, checkpoints={name:{} for name in cases},
                              initializations=[d for d in protocol['initializations'] if d.startswith('2020')])
        write_json(args.output/'protocol.json', audit_protocol)
        diagnostic_rows = []
        prediction_effects = []
        diagnostic_dates = audit_protocol['initializations'][::3]
        for date in audit_protocol['initializations']:
            manifest.update(status='running', current_date=date)
            write_json(args.output/'run.json', manifest)
            inputs, targets, forcings = batch(date)
            template = xr.zeros_like(targets)
            climo = load_climatology(climate, np.datetime64(date), targets, protocol, 'work/era5-1deg-climatology-cache')
            forecasts = {}
            for name, model in cases.items():
                predict_fn = base_predict if name in ('pretrained', 'baseline_trained') else adv_predict
                forecast, _ = predict_fn(model.params, {}, jax.random.PRNGKey(0), inputs, template, forcings)
                forecast = jax.device_get(forecast)
                forecasts[name] = forecast
                mse, acc = {}, {}
                for variable in targets:
                    mse[variable] = spatial_mse(forecast[variable], targets[variable])
                    f, t = forecast[variable], targets[variable]
                    if 'level' in t.dims:
                        f, t = f.sel(level=climo.level), t.sel(level=climo.level)
                    acc[variable] = spatial_acc(f, t, climo[variable])
                    assert np.isfinite(mse[variable]).all() and np.isfinite(acc[variable]).all()
                prefix = date.replace('-', '').replace(':', '')
                folder = args.output/'per_initialization'/name
                folder.mkdir(parents=True, exist_ok=True)
                save_metrics(folder/(prefix+'.mse.nc'), xr.Dataset(mse))
                save_metrics(folder/(prefix+'.acc.nc'), xr.Dataset(acc))
            for left, right in [('advection_trained','advection_off'),
                                ('advection_trained','advection_reset_velocity'),
                                ('advection_initial','pretrained'),
                                ('baseline_trained','pretrained')]:
                for variable in targets:
                    field = spatial_mse(forecasts[left][variable], forecasts[right][variable])
                    if 'level' in field.dims:
                        field = field.sel(level=[500,850])
                    prediction_effects.append(dict(date=date, left=left, right=right, variable=variable,
                        levels=field.level.values.tolist() if 'level' in field.dims else None,
                        spatial_mse=np.asarray(field).tolist()))
            if date in diagnostic_dates:
                endpoints = {}
                for label, model in [('initial', initial), ('trained', trained)]:
                    (scalars, fields, boundary), _ = diagnose(model.params, {}, jax.random.PRNGKey(0), inputs, template, forcings)
                    scalars, fields, boundary = jax.device_get((scalars, fields, boundary))
                    diagnostic_rows.append(dict(date=date, endpoint=label,
                        scalars=materialized_scalars(scalars, boundary), fields=field_statistics(fields)))
                    endpoints[label] = fields['current']
                    if date == diagnostic_dates[0]:
                        np.savez_compressed(args.output/f'map-{label}.npz', latitude=lat, longitude=lon,
                            mean_distance_km=fields['current']['distance_km'].mean(axis=1),
                            mean_east=fields['current']['east'].mean(axis=1),
                            mean_north=fields['current']['north'].mean(axis=1))
                a = np.concatenate([endpoints['initial'][k].ravel() for k in ('east','north')]).astype(float)
                b = np.concatenate([endpoints['trained'][k].ravel() for k in ('east','north')]).astype(float)
                diagnostic_rows[-1]['endpoint_vector_comparison'] = dict(
                    relative_change=float(np.linalg.norm(b-a)/np.linalg.norm(a)),
                    correlation=float(np.corrcoef(a,b)[0,1]))
                write_json(args.output/'displacement.json', diagnostic_rows)
            manifest['completed_dates'].append(date)
            summarize(args.output, audit_protocol, manifest['completed_dates'])
            write_json(args.output/'prediction_effects.json', prediction_effects)
            write_json(args.output/'run.json', manifest)
            print(f'Completed {date}: {len(manifest["completed_dates"])}/12, six forecast cases', flush=True)
            del forecasts
        manifest.update(status='complete', finished_at_utc=time.time())
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(args.output/'run.json', manifest)


if __name__ == '__main__':
    main()
