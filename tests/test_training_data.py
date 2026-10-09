from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import xarray as xr
from finetuning.data import ATMOSPHERE,SURFACE,STATIC,read_window,regrid
from finetuning.window_cache import cached_window


def test_coarse_read_matches_interpolation_after_precip_accumulation():
    rng=np.random.default_rng(8)
    times=np.arange(np.datetime64('2016-01-01','ns'),np.datetime64('2016-01-02','ns'),
                    np.timedelta64(1,'h'))
    coords=dict(time=times,level=[500,850],latitude=[90.,45.,0.,-45.,-90.],
                longitude=np.arange(8)*45.)
    variables={n:(('time','level','latitude','longitude'),
                    rng.normal(size=(24,2,5,8)).astype(np.float32)) for n in ATMOSPHERE}
    variables.update({n:(('time','latitude','longitude'),
                        rng.uniform(size=(24,5,8)).astype(np.float32))
                      for n in SURFACE+('total_precipitation',)})
    variables.update({n:(('latitude','longitude'),rng.uniform(size=(5,8)).astype(np.float32))
                      for n in STATIC})
    source=xr.Dataset(variables,coords=coords)
    source.total_precipitation.attrs['units']='m'
    full=read_window(source,'2016-01-01T12','train',1,[500,850])
    expected=regrid(full,90.)
    actual=read_window(source,'2016-01-01T12','train',1,[500,850],resolution=90.)
    xr.testing.assert_allclose(actual,expected)
    assert actual.total_precipitation_6hr.attrs['units']=='m'
    assert actual.lat.dtype==np.float64 and actual.lon.dtype==np.float64


def test_window_cache_serializes_paired_reads_and_separates_dates(tmp_path):
    calls=[]
    def fetch():
        calls.append(1)
        time.sleep(.05)
        return xr.Dataset({'a':(('x',),np.arange(10,dtype=np.float32))})
    identity={'source':'test','split':'train','date':'2016-01-01','levels':[500],
              'resolution':1.,'steps':1}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(cached_window,tmp_path,identity,fetch,1024**2) for _ in range(2)]
        results=[f.result() for f in futures]
    assert len(calls)==1
    xr.testing.assert_identical(*results)
    cached_window(tmp_path,{**identity,'date':'2017-01-01'},fetch,1024**2)
    assert len(calls)==2
    cached_window(tmp_path,identity,fetch,1024**2)
    assert len(calls)==2
    # NetCDF file overhead also counts against the persistent cache bound.
    cached_window(tmp_path,{**identity,'date':'2018-01-01'},fetch,64)
    assert sum(p.stat().st_size for p in tmp_path.glob('*.nc'))<=64
