"""Fit scales exclusively on a declared training interval; stream via Dask."""
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
from airadv.schema import SURF,ATMOS,SURF_POLL,ATMOS_POLL
from airadv.data import validate_schema

def compute_stats(path,start,end):
    ds=xr.open_zarr(path,consolidated=True); validate_schema(ds)
    ds=ds.sel(time=(ds.time>=np.datetime64(start))&(ds.time<np.datetime64(end)))
    if ds.sizes["time"]<3: raise ValueError("At least three training frames needed")
    consecutive=np.diff(ds.time.values)==np.timedelta64(12,"h")
    if not consecutive.any(): raise ValueError("No consecutive 12-hour training states")
    answer={"scale":{},"loss_scale":{},"source":str(path),"start":start,"end_exclusive":end,"frames":ds.sizes["time"],"definition":"scale: half mean spatial max for pollution, standard deviation otherwise; loss_scale: 12h persistence MAE"}
    for v in SURF+ATMOS:
        x=ds[v]
        if bool((~np.isfinite(x)).any().compute()): raise ValueError(f"Nonfinite training variable {v}")
        if v in SURF_POLL+ATMOS_POLL:
            if bool((x<0).any().compute()): raise ValueError(f"Negative training concentration: {v}; investigate source")
            scale=0.5*x.max(("latitude","longitude")).mean("time")
        else: scale=x.std(("time","latitude","longitude"))
        error=abs(x.diff("time")).isel(time=np.flatnonzero(consecutive)).mean(("time","latitude","longitude"))
        scale,error=scale.compute(),error.compute()
        for i,key in enumerate([v] if v in SURF else [f"{v}@{int(p)}" for p in ds.level.values]):
            s=float(scale) if scale.ndim==0 else float(scale[i])
            e=float(error) if error.ndim==0 else float(error[i])
            if not np.isfinite(s+e) or s<=0: raise ValueError(f"Degenerate scale for {key}")
            answer["scale"][key]=s; answer["loss_scale"][key]=max(e,s*1e-4)
    return answer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True); p.add_argument("--start",required=True); p.add_argument("--end",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); result=compute_stats(a.data,a.start,a.end)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(f"Saved {len(result['scale'])} channel scales to {a.output}")
if __name__=="__main__": main()

