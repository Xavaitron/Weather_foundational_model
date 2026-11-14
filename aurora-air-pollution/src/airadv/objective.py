import torch
from .schema import SURF,ATMOS
from .transport import area_weights

def fields(batch):
    for name in SURF:
        yield name,batch.surf_vars[name][:,-1]
    for name in ATMOS:
        for i,level in enumerate(batch.metadata.atmos_levels):
            yield f"{name}@{level}",batch.atmos_vars[name][:,-1,i]

def weighted_loss(pred,target,stats):
    """Equal-channel, area-weighted physical MAE / training persistence MAE.

    Deliberate simplified objective, not the paper's exact variable weighting.
    """
    truth=dict(fields(target)); pieces=[]
    for key,value in fields(pred):
        y=truth[key]
        a=area_weights(pred.metadata.lat).to(value)[None,:,None]
        denom=a.sum()*value.shape[0]*value.shape[-1]
        error=(value.float()-y.float()).abs()
        pieces.append((error*a).sum()/denom/stats["loss_scale"][key])
    loss=torch.stack(pieces).mean()
    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss; inspect data/normalisation")
    return loss

class MetricAccumulator:
    def __init__(self): self.sums={}
    def update(self,pred,target,region="global"):
        truth=dict(fields(target)); lat=pred.metadata.lat; lon=pred.metadata.lon
        area=area_weights(lat)[:,None].expand(-1,len(lon))
        if region=="india": area=area*((lat[:,None]>=6)&(lat[:,None]<=38)&(lon[None,:]>=68)&(lon[None,:]<=98))
        if area.sum()==0: raise ValueError("Region has no cells")
        for key,x in fields(pred):
            x=x.detach().double(); y=truth[key].double(); a=area.to(x)[None].expand_as(x)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise FloatingPointError(f"Nonfinite evaluation: {key}")
            err=x-y
            values=torch.stack(((a*err.abs()).sum(),(a*err**2).sum(),(a*err).sum(),a.sum(),(a*(x<0)).sum())).cpu().tolist()
            old=self.sums.setdefault(key,[0.]*5)
            self.sums[key]=[u+v for u,v in zip(old,values)]
    def compute(self):
        return {k:{"mae":s[0]/s[3],"rmse":(s[1]/s[3])**0.5,"bias":s[2]/s[3],"negative_fraction":s[4]/s[3]} for k,s in self.sums.items()}

