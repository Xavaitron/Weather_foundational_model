"""Process downloaded NetCDF groups to a validated 1.5-degree Zarr store."""
import argparse
import glob
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import xarray as xr
from airadv.schema import SURF,ATMOS,STATIC
from airadv.data import canonicalise,validate_schema
from airadv.regrid import ConservativeRegridder

def netcdf_engine(path):
    with open(path,"rb") as f: magic=f.read(8)
    if magic==b"\x89HDF\r\n\x1a\n": return "h5netcdf"
    if magic[:3]==b"CDF": return "scipy"
    raise ValueError(f"Not a supported NetCDF file: {path}")

def open_group(pattern):
    files=sorted(glob.glob(pattern,recursive=True))
    if not files: raise FileNotFoundError(pattern)
    # Preprocess each file before combine: newer ADS uses forecast_reference_time.
    engines={netcdf_engine(f) for f in files}
    if len(engines)!=1: raise ValueError("Mixed NetCDF3/4 formats in one group; process them separately")
    # Current CAMS files encode forecast_period with both numeric storage dtype
    # and a timedelta dtype attribute, which conflicts in xarray's decoder.
    # Keep leads numeric: canonicalise requires exactly zero (unit-independent)
    # and drops that axis. Absolute timestamps still use normal CF decoding.
    ds=xr.open_mfdataset(files,engine=engines.pop(),decode_timedelta=False,combine="by_coords",preprocess=canonicalise,chunks={"time":2,"valid_time":2,"forecast_reference_time":2},join="exact",compat="no_conflicts")
    return ds,files

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface",required=True); p.add_argument("--atmos",required=True)
    p.add_argument("--static",required=True); p.add_argument("--output",required=True)
    p.add_argument("--product",choices=["cams_analysis","cams_forecast_t0","eac4"],required=True)
    p.add_argument("--trust-official-static-pickle",action="store_true")
    p.add_argument("--append",action="store_true",help="Append a later month to an existing store")
    a=p.parse_args()
    exists=Path(a.output).exists()
    if exists and not a.append: raise FileExistsError(a.output)
    if a.append and not exists: raise FileNotFoundError("Create the initial store without --append")
    surf,sfiles=open_group(a.surface); atmos,afiles=open_group(a.atmos)
    if not np.array_equal(surf.time,atmos.time): raise ValueError("Surface/atmosphere timestamps differ")
    if not np.array_equal(surf.latitude,atmos.latitude) or not np.array_equal(surf.longitude,atmos.longitude): raise ValueError("Surface/atmosphere grids differ")
    ds=xr.merge([surf[list(SURF)],atmos[list(ATMOS)]],join="exact")
    # Select 00/12 UTC only. The window builder later rejects gaps.
    ds=ds.sel(time=ds.time.dt.hour.isin([0,12]))
    remap=ConservativeRegridder(ds.latitude.values,ds.longitude.values)
    out=xr.Dataset({v:remap.apply(ds[v]) for v in SURF+ATMOS})
    static_path=Path(a.static)
    if static_path.suffix==".pickle":
        if not a.trust_official_static_pickle: raise ValueError("Use only the pinned official static pickle; pass --trust-official-static-pickle after checking provenance")
        with static_path.open("rb") as f: values=pickle.load(f)
        h,w=np.asarray(values["z"]).shape
        static=xr.Dataset({k:(("latitude","longitude"),np.asarray(values[k])) for k in STATIC},coords={"latitude":np.linspace(90,-90,h),"longitude":np.arange(w)*360/w})
    else:
        static=canonicalise(xr.open_dataset(static_path,engine=netcdf_engine(static_path)))
    sr=ConservativeRegridder(static.latitude.values,static.longitude.values)
    for k in STATIC:
        if k=="slt":
            # Categories must never be averaged into fractional soil classes.
            out["static__"+k]=static[k].interp(latitude=sr.lat,longitude=sr.lon,method="nearest",kwargs={"fill_value":"extrapolate"}).astype("float32")
        else: out["static__"+k]=sr.apply(static[k])
    for k in SURF: out[k]=out[k].transpose("time","latitude","longitude")
    for k in ATMOS: out[k]=out[k].transpose("time","level","latitude","longitude")
    out.attrs.update(product=a.product,grid="global_cell_centres_1.5deg",regridding="first_order_spherical_conservative",static_sha256=hashlib.sha256(static_path.read_bytes()).hexdigest())
    validate_schema(out)
    out=out.chunk({"time":2,"level":13,"latitude":120,"longitude":240})
    for v in out: out[v].encoding={}
    old_manifest=None
    if a.append:
        existing=xr.open_zarr(a.output,consolidated=True)
        validate_schema(existing)
        if existing.attrs.get("product")!=a.product or existing.attrs.get("static_sha256")!=out.attrs["static_sha256"]:
            raise ValueError("Append requires matching product and static data")
        if not (out.time.values[0]>existing.time.values[-1]): raise ValueError("Append would duplicate or reorder times")
        for v in STATIC:
            if not np.allclose(existing["static__"+v],out["static__"+v]): raise ValueError("Static fields changed")
        old_manifest=json.loads(Path(a.output+".manifest.json").read_text())
        existing.close()
        out.to_zarr(a.output,mode="a",append_dim="time",consolidated=True)
    else: out.to_zarr(a.output,mode="w",consolidated=True)
    manifest={"product":a.product,"surface_files":sfiles,"atmos_files":afiles,"static_sha256":out.attrs["static_sha256"],"time_start":str(out.time.values[0]),"time_end":str(out.time.values[-1]),"frames":out.sizes["time"],"grid":[120,240]}
    if old_manifest:
        manifest["surface_files"]=old_manifest["surface_files"]+sfiles
        manifest["atmos_files"]=old_manifest["atmos_files"]+afiles
        manifest["time_start"]=old_manifest["time_start"]
        manifest["frames"]+=old_manifest["frames"]
    Path(a.output+".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()

