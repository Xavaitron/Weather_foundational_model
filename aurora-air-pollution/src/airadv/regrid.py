"""First-order conservative rectilinear remap using exact spherical overlaps."""
import numpy as np
import xarray as xr
from scipy import sparse

def target_grid(resolution=1.5):
    h,w=round(180/resolution),round(360/resolution)
    if not np.isclose(h*resolution,180) or not np.isclose(w*resolution,360):
        raise ValueError("Resolution must divide both 180 and 360")
    return 90-(np.arange(h)+0.5)*resolution,(np.arange(w)+0.5)*resolution

def latitude_edges(lat):
    lat=np.asarray(lat)
    if len(lat)<2 or np.any(np.diff(lat)>=0):
        raise ValueError("Latitude must be strictly descending")
    if lat[0]<90-abs(lat[1]-lat[0])/2-1e-5 or lat[-1]>-90+abs(lat[-1]-lat[-2])/2+1e-5:
        raise ValueError("Source latitude cells must cover both poles; regional grids are unsupported")
    return np.r_[90.,(lat[:-1]+lat[1:])/2,-90.]

def longitude_edges(lon):
    lon=np.asarray(lon)
    d=np.diff(lon)
    if not np.allclose(d,d[0]) or not np.isclose(d[0]*len(lon),360):
        raise ValueError("Require a regular GLOBAL longitude grid without duplicate 360")
    return np.r_[lon-d[0]/2,lon[-1]+d[0]/2]

def overlap(a,b,periodic=False):
    """Input edge arrays increasing; output average weights, shape (target,source)."""
    result=np.zeros((len(b)-1,len(a)-1))
    for shift in (-360,0,360) if periodic else (0,):
        lo=np.maximum(b[:-1,None],a[None,:-1]+shift)
        hi=np.minimum(b[1:,None],a[None,1:]+shift)
        result+=np.maximum(hi-lo,0)
    result/=np.diff(b)[:,None]
    if not np.allclose(result.sum(1),1,atol=1e-8):
        raise ValueError("Source grid does not cover all target cells")
    return sparse.csr_matrix(result)

class ConservativeRegridder:
    def __init__(self,lat,lon,resolution=1.5):
        self.lat,self.lon=target_grid(resolution)
        a=np.sin(np.deg2rad(latitude_edges(lat)[::-1]))
        b=np.sin(np.deg2rad(latitude_edges(self.lat)[::-1]))
        self.wlat=overlap(a,b)[::-1,::-1].tocsr()
        self.wlon=overlap(longitude_edges(lon),longitude_edges(self.lon),True)
    def array(self,values):
        shape=values.shape
        flat=np.asarray(values).reshape(-1,*shape[-2:])
        out=np.empty((len(flat),len(self.lat),len(self.lon)),dtype=np.float32)
        for i,x in enumerate(flat):
            out[i]=self.wlon.dot(self.wlat.dot(x).T).T
        return out.reshape(*shape[:-2],len(self.lat),len(self.lon))
    def apply(self,da):
        other=[d for d in da.dims if d not in ("latitude","longitude")]
        da=da.transpose(*other,"latitude","longitude")
        result=xr.apply_ufunc(self.array,da,input_core_dims=[["latitude","longitude"]],output_core_dims=[["new_lat","new_lon"]],dask="parallelized",output_dtypes=[np.float32],dask_gufunc_kwargs={"output_sizes":{"new_lat":len(self.lat),"new_lon":len(self.lon)},"allow_rechunk":True},keep_attrs=True)
        return result.rename(new_lat="latitude",new_lon="longitude").assign_coords(latitude=self.lat,longitude=self.lon)

