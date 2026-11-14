import dataclasses
from datetime import timedelta
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from aurora import Batch, Metadata
from .schema import SURF, ATMOS, STATIC, LEVELS, UNITS, ALIASES, channel_names

def canonicalise(ds):
    """Normalise known ADS names without flattening forecast lead times into truth."""
    renames={k:v for k,v in {"lat":"latitude","lon":"longitude","pressure_level":"level",**ALIASES}.items() if k in ds and v not in ds}
    ds=ds.rename(renames)
    for lead in ("forecast_period","step"):
        if lead in ds.dims:
            vals=np.asarray(ds[lead])
            zeros=np.flatnonzero(vals==np.timedelta64(0,"ns")) if vals.dtype.kind=="m" else np.flatnonzero(vals==0)
            if len(zeros)!=1:
                raise ValueError(f"{lead}: require exactly one zero-lead state")
            ds=ds.isel({lead:int(zeros[0])},drop=True)
    if "time" not in ds.dims:
        for name in ("valid_time","forecast_reference_time"):
            if name in ds and ds[name].ndim==1:
                dim=ds[name].dims[0]
                if dim!=name:
                    ds=ds.swap_dims({dim:name})
                ds=ds.rename({name:"time"})
                break
    for dim in ("number","expver"):
        if dim in ds.dims:
            if ds.sizes[dim]!=1:
                raise ValueError(f"Resolve multiple {dim} values explicitly before preprocessing")
            ds=ds.isel({dim:0},drop=True)
    ds=ds.assign_coords(longitude=np.mod(ds.longitude,360)).sortby("longitude").sortby("latitude",ascending=False)
    if "time" in ds:
        ds=ds.sortby("time")
        if len(np.unique(ds.time))!=ds.sizes["time"]:
            raise ValueError("Duplicate valid timestamps; resolve source records explicitly")
    if "level" in ds:
        unit=str(ds.level.attrs.get("units","hPa")).lower()
        if unit=="pa": ds=ds.assign_coords(level=ds.level/100)
        ds=ds.sel(level=list(LEVELS))
    return ds

def unit_key(u):
    return str(u).lower().replace("**","").replace("^","").replace(" ","").replace("/","")

def validate_schema(ds,require_grid=True):
    missing=set(SURF+ATMOS)-set(ds.data_vars)
    missing.update("static__"+v for v in STATIC if "static__"+v not in ds)
    if missing: raise ValueError(f"Missing required data (never zero-filled): {sorted(missing)}")
    for v in SURF+ATMOS:
        expected=("time","latitude","longitude") if v in SURF else ("time","level","latitude","longitude")
        if ds[v].dims!=expected: raise ValueError(f"{v}: expected dims {expected}, got {ds[v].dims}")
        if unit_key(ds[v].attrs.get("units",""))!=unit_key(UNITS[v]):
            raise ValueError(f"{v}: units {ds[v].attrs.get('units')} do not match {UNITS[v]}; convert explicitly")
    if tuple(ds.level.values)!=LEVELS: raise ValueError("Pressure levels/order do not match schema")
    times=np.asarray(ds.time,dtype="datetime64[s]")
    if len(times)<1 or np.isnat(times).any() or np.any(np.diff(times)<=np.timedelta64(0,"s")):
        raise ValueError("Times must be finite, unique and increasing")
    if np.any(times.astype("int64")%(12*3600)!=0): raise ValueError("Expected 00/12 UTC timestamps")
    for v in STATIC:
        if ds["static__"+v].dims!=("latitude","longitude"):
            raise ValueError(f"Static variable {v} must have no time dimension")
    if require_grid:
        from .regrid import target_grid
        lat,lon=target_grid()
        if ds.sizes["latitude"]!=120 or ds.sizes["longitude"]!=240 or not np.allclose(ds.latitude,lat) or not np.allclose(ds.longitude,lon):
            raise ValueError("Expected global 1.5-degree cell-centre grid (120,240)")

def valid_windows(times,start,end,steps=1):
    """The entire two-frame history and all targets must lie inside the split."""
    times=np.asarray(times,dtype="datetime64[s]")
    start,end=np.datetime64(start),np.datetime64(end)
    good=[]
    for i in range(1,len(times)-steps):
        seq=times[i-1:i+steps+1]
        if seq[0]>=start and seq[-1]<end and np.all(np.diff(seq)==np.timedelta64(12,"h")):
            good.append(i)
    return good

def pack_pollutants(batch):
    from .schema import SURF_POLL,ATMOS_POLL
    return torch.cat([batch.surf_vars[v].unsqueeze(2) for v in SURF_POLL]+[batch.atmos_vars[v] for v in ATMOS_POLL],2)

def unpack_pollutants(pred,q):
    from .schema import SURF_POLL,ATMOS_POLL
    surf=dict(pred.surf_vars); atmos=dict(pred.atmos_vars); offset=0
    for v in SURF_POLL: surf[v]=q[:,None,offset]; offset+=1
    for v in ATMOS_POLL:
        n=len(pred.metadata.atmos_levels); atmos[v]=q[:,None,offset:offset+n]; offset+=n
    return dataclasses.replace(pred,surf_vars=surf,atmos_vars=atmos)

def advance(history,pred,detach=False):
    """Replace history with [latest input, prediction]; no future truth is used."""
    def combine(old,new):
        value=torch.cat((old[:,-1:],new),1)
        return value.detach() if detach else value
    return dataclasses.replace(pred,surf_vars={k:combine(history.surf_vars[k],v) for k,v in pred.surf_vars.items()},atmos_vars={k:combine(history.atmos_vars[k],v) for k,v in pred.atmos_vars.items()})

class CAMSDataset(Dataset):
    def __init__(self,path,start,end,steps=1,require_grid=True):
        self.ds=xr.open_zarr(path,consolidated=True)
        validate_schema(self.ds,require_grid)
        self.indices=valid_windows(self.ds.time.values,start,end,steps)
        self.steps=steps
        if not self.indices: raise ValueError("No complete 12-hour windows in requested split")
        self.lat=torch.tensor(self.ds.latitude.values,dtype=torch.float32)
        self.lon=torch.tensor(self.ds.longitude.values,dtype=torch.float32)
        self.static={k:torch.tensor(self.ds["static__"+k].values,dtype=torch.float32) for k in STATIC}
        if any(not torch.isfinite(x).all() for x in self.static.values()): raise ValueError("Nonfinite static data")
    def __len__(self): return len(self.indices)
    def batch(self,indices):
        state=self.ds.isel(time=indices).load()
        def tensor(v):
            x=torch.tensor(state[v].values,dtype=torch.float32).unsqueeze(0)
            if not torch.isfinite(x).all(): raise ValueError(f"Nonfinite {v} at {state.time.values}")
            return x
        return Batch(surf_vars={k:tensor(k) for k in SURF},atmos_vars={k:tensor(k) for k in ATMOS},static_vars=self.static,metadata=Metadata(lat=self.lat,lon=self.lon,time=(state.time.values[-1].astype("datetime64[s]").item(),),atmos_levels=LEVELS))
    def __getitem__(self,index):
        i=self.indices[index]
        return self.batch([i-1,i]),[self.batch([i+j]) for j in range(1,self.steps+1)]

