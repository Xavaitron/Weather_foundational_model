"""Paper-inspired transport branches. These are adaptations, not reproductions."""
import torch
from torch import nn
from torch.nn import functional as F
from .transport import GlobeConv, pull, gaussian_pull

class Block(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(GlobeConv(cin,cout),nn.GroupNorm(4,cout),nn.SiLU(),GlobeConv(cout,cout),nn.SiLU())

class MotionUNet(nn.Module):
    def __init__(self,cin,cout,width=32):
        super().__init__()
        self.enc=Block(cin,width); self.low=Block(width,2*width)
        self.dec=Block(3*width,width); self.out=nn.Conv2d(width,cout,1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self,x):
        h=self.enc(x)
        z=self.low(F.avg_pool2d(h,2))
        z=F.interpolate(z,size=h.shape[-2:],mode="bilinear",align_corners=False)
        return self.out(self.dec(torch.cat((h,z),1)))

class EvolutionAdapter(nn.Module):
    """NowcastNet evolution idea: U-Net motion plus a learned source increment.

    No stochastic generator/discriminator: deterministic air-quality ablation.
    """
    def __init__(self,n,width=32):
        super().__init__(); self.n=n
        self.net=MotionUNet(4*n,3*n,width)
        self.gate=nn.Parameter(torch.full((1,n,1,1),-4.))
    def forward(self,history,wind_flow):
        old,now=history.unbind(1)
        features=torch.cat((old,now,wind_flow[:,:,0]/20,wind_flow[:,:,1]/20),1)
        delta=self.net(features)
        residual=2*torch.tanh(delta[:,:2*self.n].reshape(now.shape[0],self.n,2,*now.shape[-2:]))
        moved=pull(now,wind_flow+residual)
        return self.gate.sigmoid()*(moved-now)+0.1*delta[:,2*self.n:]

class GaussianAdapter(nn.Module):
    """de Bezenac Gaussian warp idea; bounded learned displacement/diffusion."""
    def __init__(self,n,width=32):
        super().__init__(); self.n=n
        self.motion=MotionUNet(4*n,2*n,width)
        self.log_sigma=nn.Parameter(torch.zeros(1,n,1,1))
        self.gate=nn.Parameter(torch.full((1,n,1,1),-4.))
    def forward(self,history,wind_flow):
        old,now=history.unbind(1)
        z=torch.cat((old,now,wind_flow[:,:,0]/20,wind_flow[:,:,1]/20),1)
        residual=2*torch.tanh(self.motion(z).reshape(now.shape[0],self.n,2,*now.shape[-2:]))
        sigma=0.2+0.8*self.log_sigma.sigmoid()
        moved=gaussian_pull(now,wind_flow+residual,sigma)
        return self.gate.sigmoid()*(moved-now)

class ADRLayer(nn.Module):
    """Latent reaction -> stable explicit diffusion -> learned pull advection."""
    def __init__(self,width):
        super().__init__(); self.width=width
        self.reaction=nn.Sequential(nn.Conv2d(width,width,1),nn.SiLU(),nn.Conv2d(width,width,1))
        self.motion=nn.Sequential(GlobeConv(width,width),nn.SiLU(),nn.Conv2d(width,2*width,1))
        self.diffusivity=nn.Parameter(torch.full((1,width,1,1),-2.))
        nn.init.zeros_(self.motion[-1].weight); nn.init.zeros_(self.motion[-1].bias)
    def forward(self,x):
        x=x+0.1*self.reaction(x)
        # Bound coefficient below 1/4: stable 5-point explicit diffusion in pixel units.
        north=torch.cat((torch.roll(x[...,:1,:],x.shape[-1]//2,-1),x[...,:-1,:]),-2)
        south=torch.cat((x[...,1:,:],torch.roll(x[...,-1:,:],x.shape[-1]//2,-1)),-2)
        lap=north+south+x.roll(1,-1)+x.roll(-1,-1)-4*x
        x=x+0.24*self.diffusivity.sigmoid()*lap
        flow=2*torch.tanh(self.motion(x).reshape(x.shape[0],self.width,2,*x.shape[-2:]))
        return pull(x,flow)

class ADRAdapter(nn.Module):
    def __init__(self,n,width=32,depth=3):
        super().__init__()
        self.lift=nn.Conv2d(4*n,width,1)
        self.layers=nn.Sequential(*(ADRLayer(width) for _ in range(depth)))
        self.project=nn.Conv2d(width,n,1)
        nn.init.zeros_(self.project.weight); nn.init.zeros_(self.project.bias)
    def forward(self,history,wind_flow):
        old,now=history.unbind(1)
        z=self.lift(torch.cat((old,now,wind_flow[:,:,0]/20,wind_flow[:,:,1]/20),1))
        return 0.1*self.project(self.layers(z))

def make_adapter(method,n,width=32):
    factories={"nowcast_evolution":EvolutionAdapter,"debezenac_gaussian":GaussianAdapter,"adr_latent":ADRAdapter}
    if method not in factories:
        raise ValueError(f"Unknown method {method}; expected {list(factories)}")
    return factories[method](n,width)
