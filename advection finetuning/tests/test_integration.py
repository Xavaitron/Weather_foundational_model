from datetime import datetime, timezone

import pytest
import torch

from advection_ft import APPROACHES
from advection_ft.model import advance, build_model
from advection_ft.schema import LEVELS, interface
from advection_ft.train import forecast_loss

torch.set_num_threads(2)


def synthetic_batch(task="cyclone"):
    from aurora import Batch, Metadata
    surf, atmos, static, _ = interface(task)
    h, w = 120, 240
    scales = {"2t": 280, "t": 260, "msl": 101000, "z": 50000, "q": .005}
    surface = {v: torch.randn(1, 2, h, w)*.01 + scales.get(v, .01) for v in surf}
    air = {v: torch.randn(1, 2, 13, h, w)*.001 + scales.get(v, .01) for v in atmos}
    return Batch(surf_vars=surface, atmos_vars=air,
                 static_vars={v: torch.zeros(h, w) for v in static},
                 metadata=Metadata(lat=torch.arange(89.25, -90, -1.5),
                                   lon=torch.arange(.75, 360, 1.5),
                                   time=(datetime(2025, 1, 1, 6, tzinfo=timezone.utc),), atmos_levels=LEVELS))


def config(task="cyclone"):
    return dict(task=task, model="tiny_random", precision="fp32", activation_checkpointing=False,
                trainable="full", adapter_channels=8, flow_groups=2, loss="mae")


@pytest.mark.parametrize("approach", APPROACHES)
def test_real_aurora_forward_backward_at_1p5_degrees(approach):
    torch.manual_seed(4)
    model = build_model(config(), approach, load_weights=False).train()
    batch = synthetic_batch()
    pred, aux, reg = model(batch)
    assert pred.surf_vars["msl"].shape == (1, 1, 120, 240)
    assert pred.metadata.time[0].hour == 12
    target = {"surf": {v: torch.cat((x, x[:, -1:]), 1) for v, x in batch.surf_vars.items()},
              "atmos": {v: torch.cat((x, x[:, -1:]), 1) for v, x in batch.atmos_vars.items()}}
    loss = forecast_loss(pred, target, 0, model.base, config()) + 1e-4*reg
    if aux is not None:
        loss = loss + .5*forecast_loss(aux, target, 0, model.base, config())
    assert torch.isfinite(loss)
    loss.backward()
    if approach != "standard":
        gradient = model.base.decoder.operator.net.motion.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0, f"{approach}: motion branch has no forecast gradient"
    assert any(p.grad is not None for p in model.base.backbone.parameters())


def test_zero_gate_exactly_restores_original_decoder():
    model = build_model(config(), "paradis", load_weights=False).eval()
    batch = synthetic_batch()
    adapter = model.base.decoder
    adapter.gain.data.zero_()
    with torch.no_grad():
        with_adapter = model(batch)[0]
        model.base.decoder = adapter.original
        original = model(batch)[0]
    torch.testing.assert_close(with_adapter.surf_vars["msl"], original.surf_vars["msl"], atol=0, rtol=0)


def test_two_step_history_and_checkpointing():
    c = config()
    c["activation_checkpointing"] = True
    model = build_model(c, "nowcastnet", load_weights=False).train()
    batch = synthetic_batch()
    first, _, _ = model(batch)
    first.surf_vars["2t"].retain_grad()
    second, aux, _ = model(advance(batch, first))
    loss = second.surf_vars["2t"].mean()+aux.surf_vars["2t"].mean()
    loss.backward()
    assert first.surf_vars["2t"].grad is not None


def test_pollution_interface_forward_backward():
    model = build_model(config("air_pollution"), "adrnet", load_weights=False).train()
    batch = synthetic_batch("air_pollution")
    pred, _, reg = model(batch)
    assert "pm2p5" in pred.surf_vars and "so2" in pred.atmos_vars
    (pred.surf_vars["pm2p5"].mean()+reg).backward()
    assert model.base.decoder.content.weight.grad is not None
