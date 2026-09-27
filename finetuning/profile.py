"""Compile a full-checkpoint update from synthetic shapes, without training data.

Reads only NetCDF schema/coordinates, never weather values. No optimizer update
is executed. This separates a memory feasibility check from test-set use.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import time
import jax
import numpy as np
import optax
import xarray as xr
from graphcast import checkpoint, data_utils, graphcast
from finetuning.train import transformed_loss, make_update, merge_pretrained, compile_update


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--stats-dir',type=Path,required=True)
    p.add_argument('--schema',type=Path,required=True)
    p.add_argument('--resolution',type=float,default=.1)
    p.add_argument('--variant',choices=['baseline','advection'],default='baseline')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-donate',action='store_true',help='Disable buffer donation for comparison')
    a=p.parse_args()
    with a.checkpoint.open('rb') as f: ckpt=checkpoint.load(f,graphcast.CheckPoint)
    config=dataclasses.replace(ckpt.model_config,resolution=a.resolution)
    stats={n:xr.load_dataset(a.stats_dir/(n+'.nc')) for n in
           ('mean_by_level','stddev_by_level','diffs_stddev_by_level')}
    with xr.open_dataset(a.schema) as src:
        # Official prepared samples already contain forcings; selecting these
        # datasets is lazy. Discard weather arrays immediately after reading shapes.
        parts=data_utils.extract_inputs_targets_forcings(src.isel(time=slice(0,3),batch=slice(0,1)),
                  target_lead_times='6h',**dataclasses.asdict(ckpt.task_config))
        batch=[]
        for part in parts:
            coords={k:v for k,v in part.coords.items() if k not in ('lat','lon')}
            coords.update(lat=np.linspace(-90,90,round(180/a.resolution)+1,dtype=np.float64),
                          lon=np.arange(round(360/a.resolution),dtype=np.float64)*a.resolution)
            values={}
            for name,field in part.data_vars.items():
                shape=tuple(len(coords[d]) if d in coords else field.sizes[d] for d in field.dims)
                values[name]=(field.dims,np.broadcast_to(np.float32(0),shape))
            batch.append(xr.Dataset(values,coords=coords))
    loss=transformed_loss(config,ckpt.task_config,stats,a.variant)
    print('Building graph and tracing parameter shapes',flush=True)
    started=time.monotonic()
    shapes,state=jax.eval_shape(loss.init,jax.random.PRNGKey(0),*batch)
    params=merge_pretrained(shapes,ckpt.params)
    for m in set(params)-set(ckpt.params):
        params[m]={n:np.zeros(s.shape,np.float32) for n,s in shapes[m].items()}
    mask=jax.tree.map(lambda x:x.ndim>1,params)
    opt=optax.chain(optax.clip_by_global_norm(32),optax.adamw(1e-5,b1=.9,b2=.95,weight_decay=.1,mask=mask))
    opt_state=jax.eval_shape(opt.init,params)
    # Abstract optimizer leaves avoid unnecessary device allocations during profiling.
    step=make_update(loss,opt)
    print('Compiling full update',flush=True)
    executable=compile_update(step,params,state,opt_state,jax.random.PRNGKey(0),*batch,
                              donate=not a.no_donate)
    memory=executable.memory_analysis()
    report=dict(resolution=a.resolution,variant=a.variant,seconds=time.monotonic()-started,
                message_passing_checkpointing=True,
                embedding_output_checkpointing=True, xla_flags=os.environ.get('XLA_FLAGS',''),
                parameter_optimizer_buffer_donation=not a.no_donate,
                compute_dtype='bfloat16', parameter_dtype='float32',
                model_config=dataclasses.asdict(config),
                pressure_levels=list(ckpt.task_config.pressure_levels),
                kind='synthetic-shape compilation only; no training or weather values',
                device=str(jax.devices()[0]),device_kind=jax.devices()[0].device_kind)
    for key in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes',
                'alias_size_in_bytes','host_argument_size_in_bytes','host_temp_size_in_bytes'):
        report[key]=getattr(memory,key,None)
    report['estimated_device_bytes']=(memory.argument_size_in_bytes+memory.output_size_in_bytes+
                                      memory.temp_size_in_bytes-memory.alias_size_in_bytes)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
