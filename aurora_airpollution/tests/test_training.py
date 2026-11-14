"""Integration: real Aurora, production grid, data loader, optimizer, resume."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from aurora.normalisation import scales
from airadv.synthetic import synthetic_batch
from airadv.schema import SURF,ATMOS,STATIC,LEVELS,UNITS
from airadv.train import run
from airadv.data import CAMSDataset,advance
from airadv.model import build_model

def make_store(root):
    b=synthetic_batch(120,240); times=np.datetime64("2022-01-01")+np.arange(20)*np.timedelta64(12,"h")
    coords={"time":times,"latitude":b.metadata.lat.numpy(),"longitude":b.metadata.lon.numpy(),"level":list(LEVELS)}
    ds=xr.Dataset(coords=coords)
    for v in SURF:
        base=b.surf_vars[v][0,-1].numpy()
        values=base[None]+np.arange(20,dtype=np.float32)[:,None,None]*.001*scales[v]
        ds[v]=xr.DataArray(values,dims=("time","latitude","longitude"),attrs={"units":UNITS[v]})
    for v in ATMOS:
        base=b.atmos_vars[v][0,-1].numpy()
        scale=np.array([scales[f"{v}_{p}"] for p in LEVELS],dtype=np.float32)
        values=base[None]+np.arange(20,dtype=np.float32)[:,None,None,None]*.001*scale[None,:,None,None]
        ds[v]=xr.DataArray(values,dims=("time","level","latitude","longitude"),attrs={"units":UNITS[v]})
    for v in STATIC: ds["static__"+v]=xr.DataArray(b.static_vars[v].numpy(),dims=("latitude","longitude"))
    ds.attrs["product"]="cams_analysis"; path=root/"synthetic.zarr"
    ds.chunk({"time":2}).to_zarr(path,consolidated=True)
    return path

def test_training_resume_exact_and_two_step_rollout(tmp_path):
    torch.set_num_threads(2)
    path=make_store(tmp_path)
    spec=importlib.util.spec_from_file_location("stats_script",Path(__file__).parents[1]/"scripts/stats.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    stats=mod.compute_stats(str(path),"2022-01-01","2022-01-05")
    stats_path=tmp_path/"stats.json"; stats_path.write_text(json.dumps(stats))
    cfg={"model":{"initialisation":"tiny_random","method":"nowcast_evolution","amp":False,"checkpointing":True,"width":8},"data":{"path":str(path),"stats":str(stats_path),"product":"cams_analysis","train_start":"2022-01-01","train_end":"2022-01-05","val_start":"2022-01-05","val_end":"2022-01-07","test_start":"2022-01-07","test_end":"2022-01-11"},"training":{"device":"cpu","seed":12,"steps":1,"lr":1e-4,"grad_accum":1,"validate_every":1,"val_samples":1,"rollout_steps":1},"output":str(tmp_path/"resumed")}
    run(copy.deepcopy(cfg))
    cfg["training"]["steps"]=2
    model=run(copy.deepcopy(cfg),resume=str(tmp_path/"resumed/last.pt"))
    other=copy.deepcopy(cfg); other["output"]=str(tmp_path/"continuous")
    continuous=run(other)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,continuous.state_dict()[key],rtol=0,atol=0)
    ck=torch.load(tmp_path/"resumed/best.pt",weights_only=False)
    cfg["model"]["checkpointing"]=False
    evaluation=build_model(cfg,stats,load_weights=False); evaluation.load_state_dict(ck["model"])
    ds=CAMSDataset(path,"2022-01-07","2022-01-11",steps=2)
    history,targets=ds[0]
    with torch.no_grad():
        for target in targets:
            pred=evaluation(history)
            assert pred.metadata.time==target.metadata.time
            history=advance(history,pred)
    # Different statistics/split must not silently resume an old run.
    cfg["training"]["lr"]=2e-4
    import pytest
    with pytest.raises(ValueError,match="Resume settings"):
        run(cfg,resume=str(tmp_path/"resumed/last.pt"))

