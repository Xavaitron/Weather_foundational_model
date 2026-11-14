import dataclasses
from datetime import timedelta

import torch
from torch import nn

from .geometry import check_grid
from .operators import TransportOperator
from .schema import LEVELS, interface


class TransportDecoder(nn.Module):
    """Fuse an explicit current-state forecast before Aurora's original decoder.

    A - projection(b) correction compares two estimates of the selected features,
    rather than advecting the already advanced whole backbone a second time.
    """
    def __init__(self, decoder, base, config, approach):
        super().__init__()
        self.original = decoder
        self.surf_vars, self.atmos_vars = tuple(base.surf_vars), tuple(base.atmos_vars)
        self.patch_size, self.levels = base.patch_size, base.encoder.latent_levels
        channels = config.get("adapter_channels", 32)
        cin = len(self.surf_vars) + len(self.atmos_vars) * len(LEVELS)
        self.content = nn.Conv2d(cin, self.levels * channels, self.patch_size, stride=self.patch_size)
        dim = 2 * base.encoder.embed_dim
        self.project = nn.Conv2d(dim, channels, 1, bias=False)
        self.lift = nn.Conv2d(channels, dim, 1, bias=False)
        nn.init.orthogonal_(self.lift.weight.flatten(1))
        self.gain = nn.Parameter(torch.tensor(0.01))
        self.operator = TransportOperator(approach, channels, config.get("flow_groups", 4),
                                          config.get("transport_substeps", 1),
                                          config.get("adr_transport", "push"))
        self.auxiliary = None
        self.normalised_input = None
        self.regularizer = None

    def _content(self, batch, history):
        fields = [batch.surf_vars[v][:, history:history+1] if history >= 0
                  else batch.surf_vars[v][:, history][:, None] for v in self.surf_vars]
        fields += [batch.atmos_vars[v][:, history] for v in self.atmos_vars]
        # Normalization is Aurora's own normalization; robust clipping protects rare
        # pollutant spikes in this auxiliary representation only.
        maps = torch.cat(fields, 1).float().clamp(-20, 20)
        c = self.content(maps)
        b, _, h, w = c.shape
        return c.reshape(b * self.levels, -1, h, w)

    def forward(self, x, batch, patch_res, lead_times):
        self.auxiliary = self.normalised_input = self.regularizer = None
        b, _, dim = x.shape
        levels, h, w = patch_res
        if levels != self.levels:
            raise ValueError("Latent-level shape changed unexpectedly.")
        lat = batch.metadata.lat.reshape(h, self.patch_size).mean(-1).to(x.device)
        lon = batch.metadata.lon.reshape(w, self.patch_size).mean(-1).to(x.device)
        check_grid(lat, lon)
        if not torch.allclose(lead_times, lead_times[:1].expand_as(lead_times)):
            raise ValueError("This experiment uses one common lead time per batch.")
        features = x.reshape(b, levels, h, w, dim).permute(0, 1, 4, 2, 3).reshape(b*levels, dim, h, w)
        # Keep transport/sampling and coordinate calculations in float32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            current, previous = self._content(batch, -1), self._content(batch, 0)
            reference = self.project(features.float())
            candidate, auxiliary, reg = self.operator(current, previous, reference, lat, lon,
                                                       float(lead_times[0]) * 3600)

            def fuse(c):
                f = features.float() + self.gain.tanh() * self.lift(c - reference)
                return f.reshape(b, levels, dim, h, w).permute(0, 1, 3, 4, 2).reshape(b, -1, dim).to(x.dtype)

            updated = fuse(candidate)
            auxiliary_x = fuse(auxiliary) if auxiliary is not None and self.training else None
        self.regularizer = reg
        if auxiliary_x is not None:
            self.auxiliary = self.original(auxiliary_x, batch, patch_res=patch_res, lead_times=lead_times)
            self.normalised_input = batch
        return self.original(updated, batch, patch_res=patch_res, lead_times=lead_times)


class ExperimentModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, batch, targets=None, config=None):
        if targets is not None:
            # One DDP forward owns the entire rollout graph. Calling DDP separately
            # at every step before a shared backward can confuse unused-parameter
            # bookkeeping when the same parameters are reused through time.
            from .train import forecast_loss
            primary, total = [], []
            for s in range(config["rollout_steps"]):
                pred, auxiliary, reg = self._forecast(batch)
                score = forecast_loss(pred, targets, s, self.base, config)
                objective = score + config.get("motion_regularization", 1e-4) * reg
                if auxiliary is not None:
                    objective = objective + config.get("nowcast_aux_weight", 0.5) * forecast_loss(
                        auxiliary, targets, s, self.base, config)
                primary.append(score)
                total.append(objective)
                detach_every = config.get("detach_every", 0)
                batch = advance(batch, pred, detach=bool(detach_every and (s+1) % detach_every == 0))
            return torch.stack(total).mean(), torch.stack(primary).mean()
        return self._forecast(batch)

    def _forecast(self, batch):
        leads = None
        if self.base.variable_lead_time:
            n = next(iter(batch.surf_vars.values())).shape[0]
            leads = torch.full((n,), self.base.timestep.total_seconds()/3600,
                               device=next(self.parameters()).device)
        pred = self.base(batch, lead_times=leads)
        if not isinstance(self.base.decoder, TransportDecoder):
            return pred, None, next(iter(pred.surf_vars.values())).new_zeros(())
        decoder = self.base.decoder
        auxiliary = decoder.auxiliary
        if auxiliary is not None:
            auxiliary = dataclasses.replace(
                auxiliary,
                surf_vars={k: v[:, None] for k, v in auxiliary.surf_vars.items()},
                atmos_vars={k: v[:, None] for k, v in auxiliary.atmos_vars.items()},
            )
            auxiliary = self.base._post_decoder_hook(decoder.normalised_input, auxiliary)
            auxiliary = auxiliary.type(torch.float32).unnormalise(surf_stats=self.base.surf_stats)
            auxiliary = self.base._post_unnorm_hook(decoder.normalised_input, auxiliary)
        return pred, auxiliary, decoder.regularizer


def build_model(config, approach, load_weights=True):
    from aurora import Aurora, AuroraAirPollution, AuroraPretrained, AuroraSmallPretrained

    name = config.get("model", "pretrained")
    kw = {"autocast": config.get("precision", "bf16") == "bf16", "use_lora": False}
    if name == "tiny_random":
        surf, atmos, static, hours = interface(config["task"])
        # Real Aurora architecture, deliberately tiny/random for integration tests only.
        constructor = AuroraAirPollution if config["task"] == "air_pollution" else Aurora
        base = constructor(surf_vars=surf, atmos_vars=atmos, static_vars=static,
                      encoder_depths=(1, 1), decoder_depths=(1, 1),
                      encoder_num_heads=(2, 4), decoder_num_heads=(4, 2),
                      embed_dim=32, num_heads=4, window_size=(2, 3, 5),
                      timestep=timedelta(hours=hours), **kw)
    elif name == "pretrained":
        if config["task"] != "cyclone":
            raise ValueError("Use model=air_pollution for the full CAMS variable interface.")
        base = AuroraPretrained(**kw)
    elif name == "small":
        if config["task"] != "cyclone":
            raise ValueError("The released small checkpoint uses the core weather interface.")
        base = AuroraSmallPretrained(**kw)
    elif name == "air_pollution":
        base = AuroraAirPollution(**kw)
    else:
        raise ValueError(f"Unsupported model {name}; use pretrained, small, air_pollution or tiny_random.")
    if load_weights and name != "tiny_random":
        if config.get("initial_checkpoint"):
            base.load_checkpoint_local(config["initial_checkpoint"], strict=True)
        else:
            base.load_checkpoint()
    # Insert the adapter only AFTER strict loading of the upstream checkpoint.
    if approach != "standard":
        base.decoder = TransportDecoder(base.decoder, base, config, approach)
    mode = config.get("trainable", "full")
    if mode not in ("full", "decoder", "adapter"):
        raise ValueError("trainable must be full, decoder or adapter")
    if mode != "full":
        base.requires_grad_(False)
        if mode == "decoder":
            base.decoder.requires_grad_(True)
        elif approach == "standard":
            raise ValueError("Standard has no adapter. Use full or decoder for every comparison.")
        else:
            base.decoder.requires_grad_(True)
            base.decoder.original.requires_grad_(False)
    if config.get("activation_checkpointing", True):
        # Checkpoint only the backbone. A stateful auxiliary decoder is not re-entered.
        base.configure_activation_checkpointing(module_names=("Swin3DTransformerBlock",))
    return ExperimentModel(base)


def advance(batch, pred, detach=False):
    surf = {k: torch.cat((batch.surf_vars[k][:, 1:], v), 1) for k, v in pred.surf_vars.items()
            if k in batch.surf_vars}
    atmos = {k: torch.cat((batch.atmos_vars[k][:, 1:], v), 1) for k, v in pred.atmos_vars.items()
             if k in batch.atmos_vars}
    if detach:
        surf = {k: v.detach() for k, v in surf.items()}
        atmos = {k: v.detach() for k, v in atmos.items()}
    return dataclasses.replace(pred, surf_vars=surf, atmos_vars=atmos, static_vars=batch.static_vars)
