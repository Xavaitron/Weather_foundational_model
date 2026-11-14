"""Differentiable scalar transport on a global cell-centred lat/lon grid.

Coordinates use x east, y south. Learned flows are grid cells per step.
Pull transport preserves constants, not area-weighted tracer mass.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

EARTH_RADIUS = 6_371_000.0

def area_weights(lat):
    """Exact latitude-band areas up to a common longitude/R^2 factor."""
    edges = torch.cat((lat.new_tensor([90.]), (lat[:-1]+lat[1:])/2, lat.new_tensor([-90.])))
    return (torch.sin(torch.deg2rad(edges[:-1])) - torch.sin(torch.deg2rad(edges[1:]))).abs()

def base_grid(q):
    h, w = q.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=q.device, dtype=q.dtype), torch.arange(w, device=q.device, dtype=q.dtype), indexing="ij")
    return x, y

def fold_poles(x, y, h, w):
    """Reflect cell indices at pole edges, with the required 180-degree shift."""
    t = torch.remainder(y + 0.5, 2*h)
    reflected = t > h
    return x + reflected.to(x.dtype)*(w/2), torch.where(reflected, 2*h-t, t)-0.5

def sample_bilinear(q, x, y):
    """Sample B,C,H,W at per-channel coordinates; periodic longitude, folded poles."""
    b, c, h, w = q.shape
    x, y = torch.broadcast_tensors(x, y)
    x = x.expand(b, c, h, w)
    y = y.expand(b, c, h, w)
    x0, y0 = torch.floor(x), torch.floor(y)
    fx, fy = x-x0, y-y0
    flat = q.reshape(b, c, h*w)
    result = torch.zeros_like(q)
    for dx, dy, weight in ((0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)), (0,1,(1-fx)*fy), (1,1,fx*fy)):
        xx, yy = fold_poles(x0+dx, y0+dy, h, w)
        idx = (yy.long().clamp(0,h-1)*w + xx.long().remainder(w)).reshape(b,c,-1)
        result = result + flat.gather(-1, idx).reshape_as(q)*weight
    return result

def pull(q, flow):
    """flow is B,C,2,H,W or B,1,2,H,W; positive x moves image east."""
    x, y = base_grid(q)
    return sample_bilinear(q, x-flow[:,:,0], y-flow[:,:,1])

def gaussian_pull(q, flow, sigma, radius=2):
    """Truncated, renormalised Gaussian around the *fractional departure point*.

    This is a direct discrete Gaussian warp, not Gaussian blur after warping.
    sigma is pixels (not an atmospheric diffusivity in m^2/s).
    """
    b,c,h,w = q.shape
    x,y = base_grid(q)
    x, y = x-flow[:,:,0], y-flow[:,:,1]
    x0,y0 = torch.floor(x), torch.floor(y)
    sigma = torch.as_tensor(sigma, device=q.device, dtype=q.dtype).clamp_min(0.15)
    out, norm = torch.zeros_like(q), torch.zeros_like(q)
    flat = q.reshape(b,c,-1)
    for dy in range(-radius, radius+2):
        for dx in range(-radius, radius+2):
            xi, yi = x0+dx, y0+dy
            weight = torch.exp(-((xi-x)**2+(yi-y)**2)/(2*sigma**2))
            xx, yy = fold_poles(xi,yi,h,w)
            idx=(yy.long().clamp(0,h-1)*w+xx.long().remainder(w)).expand(b,c,h,w).reshape(b,c,-1)
            out=out+flat.gather(-1,idx).reshape_as(q)*weight
            norm=norm+weight
    return out/norm.clamp_min(1e-12)

def conservative_push(q, flow, lat):
    """Area-conservative splat for an AREAL DENSITY, e.g. kg/m^2 total column.

    Does not establish conservation for kg/kg mixing ratios or kg/m^3 PM.
    CUDA scatter_add may be nondeterministic in floating-point summation order.
    """
    b,c,h,w = q.shape
    x,y=base_grid(q)
    x,y=(x+flow[:,:,0]).expand_as(q),(y+flow[:,:,1]).expand_as(q)
    x0,y0=torch.floor(x),torch.floor(y)
    fx,fy=x-x0,y-y0
    area=area_weights(lat).to(q)[None,None,:,None]
    mass=q*area
    result=torch.zeros(b,c,h*w,device=q.device,dtype=q.dtype)
    for dx,dy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        xx,yy=fold_poles(x0+dx,y0+dy,h,w)
        idx=(yy.long().clamp(0,h-1)*w+xx.long().remainder(w)).reshape(b,c,-1)
        result=result.scatter_add(-1,idx,(mass*weight).reshape(b,c,-1))
    return result.reshape_as(q)/area

def wind_displacement(u, v, lat, lon, seconds=43200., backward=True):
    """Great-circle endpoint under locally frozen east/north wind (m/s).

    Backward flow is departure displacement for pull(); forward for push().
    A first-order trajectory approximation, with no vertical transport.
    """
    phi=torch.deg2rad(lat.to(u))[None,None,:,None]
    lam=torch.deg2rad(lon.to(u))[None,None,None,:]
    speed=torch.sqrt(u*u+v*v+1e-20)
    sign=-1 if backward else 1
    angle=sign*speed*seconds/EARTH_RADIUS
    north,east=v/speed,u/speed
    p2=torch.asin((phi.sin()*angle.cos()+phi.cos()*angle.sin()*north).clamp(-1+1e-7,1-1e-7))
    l2=lam+torch.atan2(east*angle.sin()*phi.cos(),angle.cos()-phi.sin()*p2.sin())
    dl=torch.remainder(l2-lam+math.pi,2*math.pi)-math.pi
    dx=torch.rad2deg(dl)/(360/u.shape[-1])
    dy=-(torch.rad2deg(p2)-torch.rad2deg(phi))/(180/u.shape[-2])
    # pull expects destination minus source; push expects destination minus origin.
    return torch.stack((-dx if backward else dx, -dy if backward else dy),dim=2)

class GlobeConv(nn.Module):
    """Scalar-feature convolution with circular longitude and polar-fold padding."""
    def __init__(self,cin,cout,k=3):
        super().__init__(); self.p=k//2; self.conv=nn.Conv2d(cin,cout,k)
    def forward(self,x):
        p=self.p
        if p:
            w=x.shape[-1]
            top=torch.roll(x[...,:p,:].flip(-2),w//2,-1)
            bottom=torch.roll(x[...,-p:,:].flip(-2),w//2,-1)
            x=torch.cat((top,x,bottom),dim=-2)
            x=F.pad(x,(p,p,0,0),mode="circular")
        return self.conv(x)

