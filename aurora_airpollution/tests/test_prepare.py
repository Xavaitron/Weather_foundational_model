import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import xarray as xr
from airadv.synthetic import synthetic_batch
from airadv.schema import SURF,ATMOS,STATIC,UNITS,LEVELS
from airadv.data import validate_schema

@pytest.mark.parametrize("engine,forecast",[("h5netcdf",False),("scipy",False),("h5netcdf",True)])
def test_prepare_and_append_from_netcdf(tmp_path,engine,forecast):
    # A coarser global raw grid keeps the integration test compact.
    b=synthetic_batch(12,24)
    coords={"latitude":b.metadata.lat.numpy(),"longitude":b.metadata.lon.numpy()}
    static=xr.Dataset({k:(("latitude","longitude"),b.static_vars[k].numpy()) for k in STATIC},coords=coords)
    static["slt"]=xr.zeros_like(static["slt"])+2
    static.to_netcdf(tmp_path/"static.nc",engine=engine)
    script=Path(__file__).parents[1]/"scripts/prepare.py"
    output=tmp_path/"prepared.zarr"
    for day in (1,2):
        times=np.array([f"2022-01-0{day}T00",f"2022-01-0{day}T12"],dtype="datetime64[s]")
        surf=xr.Dataset({k:(("time","latitude","longitude"),b.surf_vars[k][0].numpy(),{"units":UNITS[k]}) for k in SURF},coords={**coords,"time":times})
        atmos=xr.Dataset({k:(("time","level","latitude","longitude"),b.atmos_vars[k][0].numpy(),{"units":UNITS[k]}) for k in ATMOS},coords={**coords,"time":times,"level":list(LEVELS)})
        if forecast:
            # Reproduce the actual ADS CAMS metadata, including the redundant
            # timedelta dtype attribute that breaks automatic timedelta decode.
            def cams_layout(ds):
                ds=ds.rename(time="forecast_reference_time").expand_dims(forecast_period=[0.0],axis=1)
                ds.forecast_period.attrs.update(units="hours",dtype="timedelta64[ns]")
                ds=ds.assign_coords(valid_time=(("forecast_reference_time","forecast_period"),times[:,None]))
                return ds
            surf,atmos=cams_layout(surf),cams_layout(atmos)
        surf.to_netcdf(tmp_path/f"{day}-sfc.nc",engine=engine); atmos.to_netcdf(tmp_path/f"{day}-plev.nc",engine=engine)
        command=[sys.executable,str(script),"--surface",str(tmp_path/f"{day}-sfc.nc"),"--atmos",str(tmp_path/f"{day}-plev.nc"),"--static",str(tmp_path/"static.nc"),"--output",str(output),"--product","cams_forecast_t0" if forecast else "cams_analysis"]
        if day==2: command.append("--append")
        result=subprocess.run(command,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    ds=xr.open_zarr(output,consolidated=True); validate_schema(ds)
    assert ds.sizes["time"]==4
    np.testing.assert_allclose(ds["static__slt"],2)
    manifest=json.loads(Path(str(output)+".manifest.json").read_text())
    assert manifest["frames"]==4

