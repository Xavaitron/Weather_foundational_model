"""GraphCast fine-tuning with bounded ERA5 windows; see docs/finetuning.md."""
import argparse
import dataclasses
import json
import hashlib
import os
from pathlib import Path
import pickle
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from graphcast import autoregressive, casting, checkpoint, graphcast, normalization
from graphcast import xarray_jax, xarray_tree
from finetuning.data import (SOURCE, SPLITS, open_source, read_window, regrid,
                             model_batch, valid_initializations, assert_window)
from finetuning.advection import AdvectedGraphCast
from finetuning.checkpoints import save_model
from finetuning.memory import guard, estimate
from finetuning.geometry import CachedGraphCast
from finetuning.window_cache import cached_window


def transformed_loss(model_config, task, stats, variant, modes=16, adapter_resolution=1.):
    @hk.transform_with_state
    def loss(inputs, targets, forcings):
        if variant == 'advection':
            predictor = AdvectedGraphCast(model_config, task, modes, adapter_resolution)
        else:
            predictor = CachedGraphCast(model_config, task)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, **stats)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        result = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), result)
    return loss


def merge_pretrained(initialized, pretrained):
    """Match every active parameter; permit one known unused legacy decoder head."""
    # The public checkpoint contains an unused mesh-node output head. Upstream
    # GraphCast decodes grid nodes only, so Haiku does not instantiate this leaf.
    legacy_unused = {'mesh2grid_gnn/~_networks_builder/decoder_nodes_mesh_nodes_mlp/~/linear_1'}
    missing = set(pretrained)-set(initialized)-legacy_unused
    if missing:
        raise ValueError(f'Architecture is missing pretrained modules: {sorted(missing)}')
    merged = hk.data_structures.to_mutable_dict(initialized)
    for module, values in initialized.items():
        if module not in pretrained:
            if not module.startswith('advection_'):
                raise ValueError(f'Unexpected new module {module}')
            continue
        if set(values) != set(pretrained[module]):
            raise ValueError(f'Parameter keys differ in {module}')
        for name, value in values.items():
            old = pretrained[module][name]
            if old.shape != value.shape:
                raise ValueError(f'Pretrained shape mismatch: {module}/{name}')
            merged[module][name] = old
    return merged


def make_update(loss, optimizer):
    @jax.jit
    def update(params, state, opt_state, rng, inputs, targets, forcings):
        def objective(p):
            (value, diagnostics), next_state = loss.apply(p, state, rng, inputs, targets, forcings)
            return value, (diagnostics, next_state)
        (value, (diagnostics, next_state)), grads = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_state, next_opt_state, value, diagnostics, optax.global_norm(grads)
    return update


def compile_update(update, *arguments, donate=False):
    """Lower flattened xarray inputs, optionally reusing parameter/optimizer buffers.

    With donation, callers must replace params and opt_state with returned values.
    Weather inputs, model state and RNG remain reusable across pilot updates.
    """
    leaves, structure = jax.tree.flatten(arguments)
    donated = []
    offset = 0
    for index, argument in enumerate(arguments):
        count = len(jax.tree.leaves(argument))
        if donate and index in (0, 2):
            donated.extend(range(offset, offset+count))
        offset += count
    def flat_update(*flat_arguments):
        return update(*jax.tree.unflatten(structure,flat_arguments))
    executable = jax.jit(flat_update, donate_argnums=tuple(donated)).lower(*leaves).compile()
    def execute(*values):
        flat, actual = jax.tree.flatten(values)
        if actual != structure:
            raise ValueError('Compiled update input structure/coordinates changed')
        return executable(*flat)
    execute.memory_analysis = executable.memory_analysis
    return execute


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stats-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', default=SOURCE)
    p.add_argument('--variant', choices=['baseline', 'advection'], required=True)
    p.add_argument('--resolution', type=float, default=1.0)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--updates', type=int, default=10)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--adapter-modes', type=int, default=16)
    p.add_argument('--adapter-resolution', type=float, default=1.)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--validate-every', type=int, default=10)
    p.add_argument('--validation-count', type=int, default=4)
    p.add_argument('--validate-initial', action='store_true', help='Evaluate pretrained weights on the fixed validation windows')
    p.add_argument('--window-cache', type=Path, help='Shared local cache for regridded ERA5 windows')
    p.add_argument('--cache-max-gib', type=float, default=32.)
    p.add_argument('--train-window', type=Path, help='Prepared absolute-time NetCDF for a pilot only')
    p.add_argument('--resume', type=Path, help='Trusted local training-state pickle from this code')
    p.add_argument('--compile-only', action='store_true', help='Compile and report memory, no optimizer execution')
    return p.parse_args()


def main():
    args = arguments()
    if min(args.updates, args.steps, args.save_every, args.validate_every, args.validation_count) < 1:
        raise ValueError('Counts must be positive')
    if not np.isfinite(args.cache_max_gib) or args.cache_max_gib <= 0:
        raise ValueError('Cache size must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output directory; resume can read state from an older directory')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(splits=SPLITS, status='preparing', devices=[str(d) for d in jax.devices()],
                    regridding='periodic bilinear interpolation of ERA5; not conservative',
                    message_passing_checkpointing=True,
                    embedding_output_checkpointing=True,
                    parameter_optimizer_buffer_donation=True,
                    compute_dtype='bfloat16', parameter_dtype='float32',
                    xla_flags=os.environ.get('XLA_FLAGS',''),
                    package_versions=dict(jax=jax.__version__,numpy=np.__version__,xarray=xr.__version__,optax=optax.__version__),
                    pilot_only=bool(args.train_window), completed_updates=0)
    manifest = args.output/'run.json'
    manifest.write_text(json.dumps(metadata, indent=2))
    try:
        with args.checkpoint.open('rb') as f:
            metadata['checkpoint_sha256'] = hashlib.file_digest(f,'sha256').hexdigest()
        metadata['stats_sha256'] = {}
        for name in ('mean_by_level','stddev_by_level','diffs_stddev_by_level'):
            with (args.stats_dir/(name+'.nc')).open('rb') as f:
                metadata['stats_sha256'][name] = hashlib.file_digest(f,'sha256').hexdigest()
        with args.checkpoint.open('rb') as f:
            original = checkpoint.load(f, graphcast.CheckPoint)
        config = dataclasses.replace(original.model_config, resolution=args.resolution)
        metadata['model_config'] = dataclasses.asdict(config)
        metadata['pressure_levels'] = list(original.task_config.pressure_levels)
        stats = {n: xr.load_dataset(args.stats_dir/(n+'.nc')) for n in
                 ('mean_by_level', 'stddev_by_level', 'diffs_stddev_by_level')}
        generator = np.random.default_rng(args.seed)
        train_dates = valid_initializations('train', args.steps)
        val_dates = valid_initializations('val', args.steps)
        # Fixed evenly spaced dates across 2020, identical for both variants.
        val_dates = val_dates[np.linspace(0, len(val_dates)-1, args.validation_count, dtype=int)]
        metadata['validation_initializations_utc'] = [str(d) for d in val_dates]
        source = None
        def batch(split, init):
            nonlocal source
            if split == 'train' and args.train_window:
                with xr.open_dataset(args.train_window) as f:
                    data = f.load()
                if data.sizes['time'] != args.steps+2:
                    raise ValueError('Pilot window length differs from rollout')
                init = data.time.values[1]
                assert_window(init, 'train', args.steps)
                expected = init + np.arange(-1, args.steps+1)*np.timedelta64(6,'h')
                if not np.array_equal(data.time.values, expected):
                    raise ValueError('Pilot data must contain consecutive six-hour states')
            else:
                assert_window(init,split,args.steps)
                metadata.update(status='loading_window',loading_split=split,loading_initialization_utc=str(init))
                manifest.write_text(json.dumps(metadata,indent=2))
                def fetch():
                    nonlocal source
                    if source is None:
                        source = open_source(args.source)
                    return read_window(source,init,split,args.steps,original.task_config.pressure_levels,
                                       resolution=args.resolution)
                identity=dict(format_version=1,source=args.source,split=split,initialization=str(init),
                              steps=args.steps,levels=list(original.task_config.pressure_levels),
                              resolution=args.resolution)
                data = (cached_window(args.window_cache,identity,fetch,int(args.cache_max_gib*1024**3))
                        if args.window_cache else fetch())
            # A cached 37-level pilot can also serve the 13-level small model.
            # Discard unused levels before allocating the interpolated grid.
            data = data.sel(level=list(original.task_config.pressure_levels))
            return model_batch(regrid(data, args.resolution), original.task_config, args.steps)

        first_init = generator.choice(train_dates)
        if args.train_window:
            with xr.open_dataset(args.train_window) as pilot:
                first_init = pilot.time.values[1]
        metadata['first_training_initialization'] = str(first_init)
        print('Preparing first training window', str(first_init), flush=True)
        first = batch('train', first_init)
        loss = transformed_loss(config, original.task_config, stats, args.variant,
                                args.adapter_modes, args.adapter_resolution)
        # eval_shape builds parameter shapes without a full random-weight forward pass.
        # Adapter params are initialized separately below using their small shapes.
        shapes, state = jax.eval_shape(loss.init, jax.random.PRNGKey(args.seed), *first)
        new_params = {}
        init_rng = np.random.default_rng(args.seed)
        for module, values in shapes.items():
            if module in original.params:
                new_params[module] = original.params[module]
            else:
                new_params[module] = {}
                for name, value in values.items():
                    if name == 'b' or module.startswith('advection_lift'):
                        array = np.zeros(value.shape, np.float32)
                    else:
                        scale = .01 if module.startswith('advection_velocity') else 1/np.sqrt(value.shape[0])
                        array = init_rng.normal(0, scale, value.shape).astype(np.float32)
                    new_params[module][name] = array
        params = merge_pretrained(shapes, original.params)
        for module in new_params:
            if module not in original.params:
                params[module] = new_params[module]
        state = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), state)
        decay_mask = jax.tree.map(lambda x: x.ndim > 1, params)
        optimizer = optax.chain(optax.clip_by_global_norm(32.),
                               optax.adamw(args.learning_rate, b1=.9, b2=.95,
                                           weight_decay=.1, mask=decay_mask))
        opt_state = optimizer.init(params)
        step = 0
        best = float('inf')
        if args.resume:
            with args.resume.open('rb') as f:
                saved = pickle.load(f)
            for key in ('variant','resolution','steps','learning_rate','adapter_modes','adapter_resolution','source','seed','train_window','checkpoint_sha256','stats_sha256','validation_count'):
                if saved['config'][key] != metadata[key]:
                    raise ValueError(f'Resume configuration mismatch: {key}')
            params,state,opt_state,step,best = (saved[k] for k in ('params','state','opt_state','step','best'))
            generator.bit_generator.state = saved['sampler_state']
        update = make_update(loss, optimizer)
        print('Compiling update', flush=True)
        metadata['status']='compiling'
        manifest.write_text(json.dumps(metadata,indent=2))
        t0 = time.monotonic()
        compiled = compile_update(update,params,state,opt_state,jax.random.PRNGKey(args.seed),*first,
                                  donate=True)
        memory = compiled.memory_analysis()
        metadata['compiled_memory'] = str(memory)
        metadata['estimated_device_bytes'] = estimate(memory)
        metadata['compile_seconds'] = time.monotonic()-t0
        print('Compiled memory:', memory, flush=True)
        if args.compile_only:
            metadata['status'] = 'compiled_only'
            return
        guard(memory)
        def save(path):
            save_model(path,original,config,params,args.variant,args.adapter_modes,args.adapter_resolution)
        validation = jax.jit(loss.apply)
        log = (args.output/'metrics.jsonl').open('a')
        def validate():
            values=[]
            for date in val_dates:
                (v,_),_ = validation(params,state,jax.random.PRNGKey(0),*batch('val',date))
                values.append(float(v))
            result=float(np.mean(values))
            if not np.isfinite(result):
                raise FloatingPointError('Non-finite validation loss')
            return result
        if args.validate_initial and not args.resume:
            best=validate()
            row=dict(update=0,val_loss=best,kind='pretrained_validation')
            print(json.dumps(row),flush=True); log.write(json.dumps(row)+'\n'); log.flush()
            save(args.output/'best.npz')
            metadata['initial_val_loss']=best
        for index in range(step, args.updates):
            data_t0=time.monotonic()
            use_first=args.train_window or (index == 0 and not args.resume)
            current_init=first_init if use_first else generator.choice(train_dates)
            data = first if use_first else batch('train',current_init)
            data_seconds=time.monotonic()-data_t0
            metadata['status']='training'
            t0 = time.monotonic()
            params,state,opt_state,value,diag,gradnorm = compiled(
                params,state,opt_state,jax.random.fold_in(jax.random.PRNGKey(args.seed),index),*data)
            value, gradnorm = float(value), float(gradnorm)
            if not np.isfinite(value+gradnorm):
                raise FloatingPointError('Non-finite training loss/gradient; stopping')
            row = dict(update=index+1, loss=value, gradient_norm=gradnorm,
                       seconds=time.monotonic()-t0,data_seconds=data_seconds,
                       initialization_utc=str(current_init))
            if (index+1) % args.validate_every == 0:
                row['val_loss'] = validate()
                if row['val_loss'] < best:
                    best = row['val_loss']
                    save(args.output/'best.npz')
            print(json.dumps(row),flush=True); log.write(json.dumps(row)+'\n'); log.flush()
            metadata['completed_updates'] = index+1
            if index == 0 or (index+1) % args.save_every == 0 or index+1 == args.updates:
                save(args.output/'latest.npz')
                training_state = dict(params=jax.device_get(params),state=jax.device_get(state),
                                      opt_state=jax.device_get(opt_state),step=index+1,best=best,
                                      sampler_state=generator.bit_generator.state,config=metadata)
                temp = args.output/'training-state.tmp'
                with temp.open('wb') as f: pickle.dump(training_state,f)
                temp.replace(args.output/'training-state.pkl')
            metadata.update(status='training',best_val_loss=best)
            manifest.write_text(json.dumps(metadata,indent=2))
        metadata['status'] = 'pilot_complete' if args.train_window else 'requested_updates_complete'
        log.close()
    except Exception as error:
        metadata.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest.write_text(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
