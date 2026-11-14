"""Physical-unit RMSE diagnostics; final test runs require a frozen protocol file.

ACC and spectral verification are deliberately not approximated here. This is
not a complete research benchmark and does not automatically authorize test use.
"""
import argparse
import hashlib
import json
from pathlib import Path
import haiku as hk
import jax
import numpy as np
import xarray as xr
from graphcast import checkpoint, graphcast, losses
from finetuning.checkpoints import predictor, configuration
from finetuning.data import SOURCE, open_source, read_window, regrid, model_batch, valid_initializations


def spatial_mse(prediction, reference):
    """Area-weighted squared error, preserving time and pressure-level axes."""
    prediction, reference = xr.align(prediction,reference,join='exact')
    weight = losses.normalized_latitude_weights(reference)
    error = prediction.astype(np.float64)-reference.astype(np.float64)
    axes = [d for d in ('lat','lon','batch') if d in error.dims]
    return (error**2*weight).mean(axes,skipna=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--stats-dir',type=Path,required=True)
    p.add_argument('--source',default=SOURCE)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--frozen-protocol',type=Path,
                   help='Research protocol frozen before test evaluation; its hash is recorded')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.count < 1 or a.steps < 1:
        raise ValueError('Counts must be positive')
    if a.split == 'test' and not a.frozen_protocol:
        raise ValueError('Freeze the experiment protocol on 2020 before evaluating 2021–2022')
    if a.output.exists():
        raise FileExistsError(a.output)
    with a.checkpoint.open('rb') as f:
        model=checkpoint.load(f,graphcast.CheckPoint)
    stats={n:xr.load_dataset(a.stats_dir/(n+'.nc')) for n in
           ('mean_by_level','stddev_by_level','diffs_stddev_by_level')}
    metadata=dict(checkpoint=str(a.checkpoint),architecture=configuration(model),split=a.split,
                  resolution=model.model_config.resolution,steps=a.steps,count=a.count,
                  source=a.source,status='running',completed_initializations=0,
                  aggregation='sqrt(mean over initializations of area-weighted spatial MSE)',
                  metrics_scope='RMSE only; ACC and spherical spectra pending')
    with a.checkpoint.open('rb') as f:
        metadata['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    if a.frozen_protocol:
        metadata['frozen_protocol_sha256']=hashlib.sha256(a.frozen_protocol.read_bytes()).hexdigest()
    dates=valid_initializations(a.split,a.steps)
    if a.count > len(dates):
        raise ValueError('Requested more initializations than the split contains')
    dates=dates[np.linspace(0,len(dates)-1,a.count,dtype=int)]
    metadata['initializations']=[str(d) for d in dates]
    a.output.mkdir(parents=True)
    @hk.transform_with_state
    def forward(inputs,template,forcings):
        return predictor(model,stats)(inputs,template,forcings)
    predict=jax.jit(forward.apply)
    accumulated={}
    try:
        source=open_source(a.source)
        for index,date in enumerate(dates):
            data=read_window(source,date,a.split,a.steps,model.task_config.pressure_levels)
            inputs,targets,forcings=model_batch(regrid(data,model.model_config.resolution),model.task_config,a.steps)
            # Future weather values are used for verification only.
            predictions,_=predict(model.params,{},jax.random.PRNGKey(0),inputs,xr.zeros_like(targets),forcings)
            for name in targets:
                mse=spatial_mse(predictions[name],targets[name])
                if not np.isfinite(mse.values).all():
                    raise FloatingPointError(f'Non-finite metric in {name}')
                accumulated[name]=mse if name not in accumulated else accumulated[name]+mse
            metadata['completed_initializations']=index+1
            (a.output/'run.json').write_text(json.dumps(metadata,indent=2))
            print(f'Completed {date}',flush=True)
        rmse=xr.Dataset({name:np.sqrt(value/a.count) for name,value in accumulated.items()})
        for name in rmse:
            rmse[name].attrs['units']=targets[name].attrs.get('units','unspecified in source')
        rmse.attrs.update(aggregation=metadata['aggregation'],split=a.split)
        rmse.to_netcdf(a.output/'rmse.nc')
        metadata['status']='complete'
    except Exception as error:
        metadata.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (a.output/'run.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
