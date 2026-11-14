import math
import pytest
import torch
from airadv.transport import pull,gaussian_pull,conservative_push,area_weights,wind_displacement

def test_identity_and_periodic_shift():
    q=torch.randn(2,3,12,24,dtype=torch.double)
    flow=torch.zeros(2,3,2,12,24,dtype=torch.double)
    torch.testing.assert_close(pull(q,flow),q)
    flow[:,:,0]=1
    torch.testing.assert_close(pull(q,flow),q.roll(1,-1))

def test_pole_crossing_rotates_longitude():
    q=torch.arange(8.).reshape(1,1,1,8).expand(1,1,4,8)
    f=torch.zeros(1,1,2,4,8); f[:,:,1]=1
    torch.testing.assert_close(pull(q,f)[...,0,:],q[...,0,:].roll(4,-1))

def test_gaussian_constants_and_differentiable_motion():
    flow=(torch.rand(1,1,2,6,12,dtype=torch.double)*.2+.2).requires_grad_()
    q=torch.rand(1,1,6,12,dtype=torch.double,requires_grad=True)
    torch.testing.assert_close(gaussian_pull(torch.ones_like(q),flow,.7),torch.ones_like(q))
    assert torch.autograd.gradcheck(lambda x,f:gaussian_pull(x,f,.7,radius=1),(q,flow),fast_mode=True)

def test_pull_gradcheck():
    q=torch.rand(1,1,4,8,dtype=torch.double,requires_grad=True)
    f=(torch.rand(1,1,2,4,8,dtype=torch.double)*.2+.3).requires_grad_()
    assert torch.autograd.gradcheck(pull,(q,f),fast_mode=True)

def test_spherical_push_conserves_area_mass():
    torch.manual_seed(4)
    lat=90-(torch.arange(12,dtype=torch.double)+.5)*15
    q=torch.rand(2,3,12,24,dtype=torch.double)
    f=torch.randn(2,3,2,12,24,dtype=torch.double)*3
    out=conservative_push(q,f,lat)
    a=area_weights(lat)[None,None,:,None]
    torch.testing.assert_close((q*a).sum((-2,-1)),(out*a).sum((-2,-1)),rtol=1e-12,atol=1e-12)
    assert (out>=0).all()

def test_wind_direction_and_finite_poles():
    lat=90-(torch.arange(120)+.5)*1.5; lon=(torch.arange(240)+.5)*1.5
    u=torch.full((1,1,120,240),10.); zero=torch.zeros_like(u)
    f=wind_displacement(u,zero,lat,lon)
    assert torch.isfinite(f).all()
    expected=10*43200/(6_371_000*math.radians(1.5))
    assert abs(float(f[0,0,0,60,0])-expected)<.01
    g=wind_displacement(zero,u,lat,lon)
    assert float(g[0,0,1,60,0])<0  # north = decreasing row index

