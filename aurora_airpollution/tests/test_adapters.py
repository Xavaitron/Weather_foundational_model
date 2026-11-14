import pytest
import torch
from airadv.adapters import make_adapter

@pytest.mark.parametrize("method",["nowcast_evolution","debezenac_gaussian","adr_latent"])
def test_adapter_backprop_and_optimizer_update(method):
    torch.set_num_threads(2); torch.manual_seed(5)
    m=make_adapter(method,3,width=8)
    history=torch.rand(1,2,3,8,16); flow=torch.randn(1,3,2,8,16)*.3
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    before=[p.detach().clone() for p in m.parameters()]
    for _ in range(2):
        opt.zero_grad(); y=m(history,flow)
        assert y.shape==(1,3,8,16)
        loss=(y-.2).square().mean(); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
    assert any(not torch.equal(p,b) for p,b in zip(m.parameters(),before))

