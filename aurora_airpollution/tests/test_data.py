import numpy as np
import pytest
import xarray as xr
from airadv.data import valid_windows,canonicalise
from airadv.regrid import ConservativeRegridder,latitude_edges

def test_window_boundaries_and_gaps():
    t=np.datetime64("2022-01-01")+np.arange(12)*np.timedelta64(12,"h")
    idx=valid_windows(t,"2022-01-02","2022-01-05",2)
    assert idx==[3,4,5]
    broken=np.delete(t,5)
    for i in valid_windows(broken,"2022-01-01","2022-01-07",2):
        assert np.all(np.diff(broken[i-1:i+3])==np.timedelta64(12,"h"))

def test_remap_mass_and_constant():
    lat=np.linspace(90,-90,241); lon=np.arange(480)*.75
    rg=ConservativeRegridder(lat,lon)
    rng=np.random.default_rng(42); x=rng.random((241,480))
    out=rg.array(x)
    src_area=-np.diff(np.sin(np.deg2rad(latitude_edges(lat))))[:,None]*.75
    dst_area=-np.diff(np.sin(np.deg2rad(latitude_edges(rg.lat))))[:,None]*1.5
    np.testing.assert_allclose((x*src_area).sum(),(out*dst_area).sum(),rtol=1e-7)
    np.testing.assert_allclose(rg.array(np.ones_like(x)),1,rtol=1e-7)
    da=xr.DataArray(x,dims=("latitude","longitude"),coords={"latitude":lat,"longitude":lon},attrs={"units":"kg m-2"})
    result=rg.apply(da.chunk({"latitude":241,"longitude":480})).compute()
    assert result.shape==(120,240) and result.attrs["units"]=="kg m-2"

def test_canonicalise_selects_zero_lead():
    ds=xr.Dataset({"t2m":(("forecast_reference_time","forecast_period","latitude","longitude"),np.zeros((2,2,3,4)))},coords={"forecast_reference_time":np.array(["2022-01-01","2022-01-02"],dtype="datetime64[s]"),"forecast_period":[0,12],"latitude":[-90,0,90],"longitude":[-180,-90,0,90]})
    result=canonicalise(ds)
    assert result["2t"].dims==("time","latitude","longitude")
    assert list(result.latitude)==[90,0,-90]
    assert list(result.longitude)==[0,90,180,270]

def test_canonicalise_refuses_no_zero_lead():
    ds=xr.Dataset(coords={"step":[6,12],"latitude":[90,0,-90],"longitude":[0,90,180,270]})
    with pytest.raises(ValueError,match="zero-lead"): canonicalise(ds)

