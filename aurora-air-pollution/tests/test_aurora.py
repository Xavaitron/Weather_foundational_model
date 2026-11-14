import torch
from aurora import AuroraAirPollution
from airadv.synthetic import synthetic_batch,synthetic_stats
from airadv.model import ForecastModel
from airadv.schema import channel_names

def test_specialised_air_architecture_at_requested_grid():
    torch.set_num_threads(2)
    backbone=AuroraAirPollution(embed_dim=32,num_heads=4,encoder_depths=(1,1,1),decoder_depths=(1,1,1),encoder_num_heads=(2,4,8),decoder_num_heads=(8,4,2),autocast=False,use_lora=False)
    stats=synthetic_stats()
    model=ForecastModel(backbone,"debezenac_gaussian",[stats["scale"][k] for k in channel_names(pollutants_only=True)],width=8,train_normalisation=False)
    batch=synthetic_batch(120,240)
    pred=model(batch)
    loss=(pred.surf_vars["pm2p5"]/stats["scale"]["pm2p5"]).square().mean()
    loss.backward()
    assert pred.surf_vars["pm2p5"].shape==(1,1,120,240)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
