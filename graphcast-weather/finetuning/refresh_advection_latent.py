"""Materialize captured BF16 boundaries and check repeated-inference variability."""
import dataclasses
import json
from pathlib import Path
import jax
import numpy as np
import xarray as xr
from finetuning.diagnose_advection import (initial_parameters,make_diagnostics,
    materialized_scalars,field_statistics)
from finetuning.data import read_window,model_batch
from finetuning.evaluate import spatial_mse
from finetuning.paired_evaluate import load_model,open_evaluation_source,write_json,make_predict
from finetuning.window_cache import cached_window


def main():
    root=Path('runs/advection-diagnostics-20260928')
    train=Path('runs/full-1deg-stage1-20260927')
    metadata=json.loads((train/'advection/run.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    original=load_model(metadata['checkpoint'])
    trained=load_model(train/'advection/best.npz')
    initial=dataclasses.replace(trained,params=initial_parameters(original,trained,metadata['seed']))
    stats={n:xr.load_dataset(Path(metadata['stats_dir'])/(n+'.nc')) for n in
           ('mean_by_level','stddev_by_level','diffs_stddev_by_level')}
    source=open_evaluation_source(protocol['source'])
    def batch(date):
        init=np.datetime64(date,'h')
        identity=dict(format_version=1,source=protocol['source'],split='val',initialization=str(init),
                      steps=1,levels=protocol['model_pressure_levels'],resolution=protocol['resolution'])
        data=cached_window(metadata['window_cache'],identity,
            lambda:read_window(source,init,'val',1,identity['levels'],identity['resolution']))
        return model_batch(data,trained.task_config,1)
    observe,lat,lon=make_diagnostics(trained,stats,initial.params['advection_velocity'])
    rows=[]
    for date in protocol['initializations'][::3]:
        inputs,targets,forcings=batch(date)
        endpoints={}
        for label,model in [('initial',initial),('trained',trained)]:
            (s,f,boundary),_=observe(model.params,{},jax.random.PRNGKey(0),inputs,xr.zeros_like(targets),forcings)
            s,f,boundary=jax.device_get((s,f,boundary))
            rows.append(dict(date=date,endpoint=label,scalars=materialized_scalars(s,boundary),
                             fields=field_statistics(f),measurement='actual BF16 decoder-input tensors materialized; host reductions'))
            endpoints[label]=f['current']
            if date==protocol['initializations'][0]:
                np.savez_compressed(root/f'map-{label}.npz',latitude=lat,longitude=lon,
                    mean_distance_km=f['current']['distance_km'].mean(axis=1),
                    mean_east=f['current']['east'].mean(axis=1),mean_north=f['current']['north'].mean(axis=1))
            del boundary
        a=np.concatenate([endpoints['initial'][k].ravel() for k in ('east','north')]).astype(float)
        b=np.concatenate([endpoints['trained'][k].ravel() for k in ('east','north')]).astype(float)
        rows[-1]['endpoint_vector_comparison']=dict(relative_change=float(np.linalg.norm(b-a)/np.linalg.norm(a)),
                                                  correlation=float(np.corrcoef(a,b)[0,1]))
        print('Measured BF16 boundary',date,rows[-1]['scalars'],flush=True)
    write_json(root/'displacement.json',rows)
    baseline=load_model(train/'baseline/best.npz')
    off=dict(trained.params)
    off['advection_lift']={k:np.zeros_like(v) for k,v in trained.params['advection_lift'].items()}
    models=dict(baseline=baseline,advection=trained,advection_off=dataclasses.replace(trained,params=off))
    functions=dict(baseline=make_predict(baseline,stats),advection=make_predict(trained,stats))
    inputs,targets,forcings=batch(protocol['initializations'][0]);template=xr.zeros_like(targets)
    first={};records=[];pairs=[]
    for repeat in range(8):
        current={}
        for name,model in models.items():
            forecast,_=functions['baseline' if name=='baseline' else 'advection'](
                model.params,{},jax.random.PRNGKey(0),inputs,template,forcings)
            forecast=jax.device_get(forecast)
            fields={}
            for variable,level in [('2m_temperature',None),('temperature',850),('geopotential',500)]:
                f,t=forecast[variable],targets[variable]
                if level is not None:f,t=f.sel(level=level),t.sel(level=level)
                label=variable+('' if level is None else '_'+str(level))
                fields[label]=f
                records.append(dict(repeat=repeat,model=name,variable=label,
                    rmse=float(np.sqrt(spatial_mse(f,t))),
                    rms_difference_from_first=float(np.sqrt(spatial_mse(f,first[name][label]))) if name in first else 0))
            current[name]=fields
            first.setdefault(name,fields)
        for variable in current['advection']:
            pairs.append(dict(repeat=repeat,variable=variable,
                on_off_rms_difference=float(np.sqrt(spatial_mse(current['advection'][variable],current['advection_off'][variable])))))
        print('Repeated inference',repeat+1,'/8',flush=True)
    write_json(root/'repeatability.json',dict(date=protocol['initializations'][0],repeats=8,records=records,on_off_pairs=pairs))


if __name__=='__main__':main()
