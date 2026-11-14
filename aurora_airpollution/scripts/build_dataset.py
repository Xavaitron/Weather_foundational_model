"""Retrieve and preprocess chronological months with bounded raw staging."""
import argparse
from datetime import date
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import numpy as np
import xarray as xr
from airadv.data import validate_schema

def next_month(day):
    return date(day.year+(day.month==12),day.month%12+1,1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product",choices=["cams_analysis","cams_forecast_t0","eac4"],required=True)
    p.add_argument("--start",required=True); p.add_argument("--end",required=True)
    p.add_argument("--raw",default="data/raw"); p.add_argument("--output",required=True)
    p.add_argument("--static",default="data/assets/aurora-0.4-air-pollution-static.pickle")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--days-per-request",type=int,default=1,choices=range(1,32))
    p.add_argument("--min-free-gib",type=float,default=200.)
    p.add_argument("--discard-verified-raw",action="store_true",help="Delete only staged monthly .nc files after full finite-value verification; keep request/hash files")
    a=p.parse_args(); day=date.fromisoformat(a.start); end=date.fromisoformat(a.end)
    if day.day!=1 or end.day!=1 or end<=day: p.error("Use increasing first-of-month start/end (end exclusive)")
    scripts=Path(__file__).resolve().parent; output=Path(a.output)
    while day<end:
        if a.execute and shutil.disk_usage(Path.cwd()).free<a.min_free_gib*2**30:
            raise RuntimeError("Free storage below reserve; stopping before the next month")
        stop=min(next_month(day),end); month=Path(a.raw)/a.product/day.strftime("%Y-%m")
        already=False
        if a.execute and output.exists():
            ds=xr.open_zarr(output,consolidated=True); validate_schema(ds)
            if ds.attrs.get("product")!=a.product: raise ValueError("Store product mismatch")
            existing=ds.sel(time=(ds.time>=np.datetime64(day))&(ds.time<np.datetime64(stop)))
            expected=np.arange(np.datetime64(day,"h"),np.datetime64(stop,"h"),np.timedelta64(12,"h"))
            if existing.sizes["time"]:
                if not np.array_equal(existing.time.values,expected): raise ValueError("Partially written month: restore a verified store before retrying")
                already=True
            ds.close()
        commands=[
            [sys.executable,str(scripts/"download.py"),"--product",a.product,"--start",str(day),"--end",str(stop),"--output",a.raw,"--days-per-request",str(a.days_per_request),"--execute"],
            [sys.executable,str(scripts/"prepare.py"),"--surface",str(month/"*-sfc.nc"),"--atmos",str(month/"*-plev.nc"),"--static",a.static,"--product",a.product,"--output",a.output],
        ]
        if Path(a.static).suffix==".pickle": commands[1].append("--trust-official-static-pickle")
        if output.exists() or day!=date.fromisoformat(a.start): commands[1].append("--append")
        if not already:
            for command in commands:
                print(shlex.join(command),flush=True)
                if a.execute: subprocess.run(command,check=True)
        if a.execute:
            ds=xr.open_zarr(output,consolidated=True)
            frame=ds.sel(time=(ds.time>=np.datetime64(day))&(ds.time<np.datetime64(stop)))
            expected=np.arange(np.datetime64(day,"h"),np.datetime64(stop,"h"),np.timedelta64(12,"h"))
            if not np.array_equal(frame.time.values,expected): raise ValueError("Incomplete month; raw files retained")
            for v in frame.data_vars:
                if bool((~np.isfinite(frame[v])).any().compute()): raise ValueError(f"Nonfinite {v}; raw files retained")
            ds.close()
            if a.discard_verified_raw:
                for f in list(month.glob("*-sfc.nc"))+list(month.glob("*-plev.nc")): f.unlink()
        day=stop
    print("Completed" if a.execute else "Dry plan only; add --execute after pilot acceptance")
if __name__=="__main__": main()

