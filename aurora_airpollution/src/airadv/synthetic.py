"""Synthetic physically scaled inputs solely for software validation."""
from datetime import datetime
import torch
from aurora import Batch,Metadata
from aurora.normalisation import locations,scales
from .schema import SURF,ATMOS,STATIC,LEVELS,SURF_POLL,ATMOS_POLL,channel_names

def synthetic_batch(h=24,w=48,seed=7):
    gen=torch.Generator().manual_seed(seed)
    def field(key,shape,positive=False):
        mu=locations[key]; scale=scales[key]
        x=mu+scale*(0.4+0.1*torch.rand(shape,generator=gen) if positive else 0.1*torch.randn(shape,generator=gen))
        return x.float()
    return Batch(surf_vars={v:field(v,(1,2,h,w),v in SURF_POLL) for v in SURF},atmos_vars={v:torch.stack([field(f"{v}_{p}",(1,2,h,w),v in ATMOS_POLL or v=="q") for p in LEVELS],2) for v in ATMOS},static_vars={v:field(v,(h,w)) for v in STATIC},metadata=Metadata(lat=90-(torch.arange(h)+.5)*180/h,lon=(torch.arange(w)+.5)*360/w,time=(datetime(2022,1,1,12),),atmos_levels=LEVELS))

def synthetic_stats():
    return {"scale":{k:scales[k.replace("@","_")] for k in channel_names()},"loss_scale":{k:scales[k.replace("@","_")] for k in channel_names()}}

