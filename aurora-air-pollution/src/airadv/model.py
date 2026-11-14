import dataclasses
from pathlib import Path
import torch
from torch import nn
from aurora import Aurora, Aurora12hPretrained, AuroraAirPollution
from .schema import SURF, ATMOS, STATIC, SURF_POLL, ATMOS_POLL, LEVELS, channel_names
from .data import pack_pollutants, unpack_pollutants
from .adapters import make_adapter
from .transport import wind_displacement, pull, conservative_push

def pollution_winds(batch):
    """Column winds are a pressure-thickness weighted proxy, not species fluxes."""
    levels=batch.surf_vars["10u"].new_tensor(batch.metadata.atmos_levels)
    edges=torch.cat((levels[:1],(levels[:-1]+levels[1:])/2,levels[-1:]))
    weights=torch.diff(edges); weights=weights/weights.sum()
    def get(surf_key,atmos_key):
        profile=batch.atmos_vars[atmos_key][:,-1]
        col=(profile*weights[None,:,None,None]).sum(1,keepdim=True)
        return torch.cat([batch.surf_vars[surf_key][:,-1:].expand(-1,3,-1,-1),col.expand(-1,5,-1,-1),*[profile for _ in ATMOS_POLL]],1)
    return get("10u","u"),get("10v","v")

class ForecastModel(nn.Module):
    def __init__(self,backbone,method,scale,width=32,train_normalisation=True):
        super().__init__(); self.backbone=backbone; self.method=method
        self.register_buffer("poll_scale",torch.tensor(scale,dtype=torch.float32)[None,:,None,None])
        from aurora.normalisation import scales,locations
        keys=[k.replace("@","_") for k in channel_names(pollutants_only=True)]
        if any(locations[k]!=0 for k in keys): raise ValueError("Expected zero-centred pollution normalisation")
        published=self.poll_scale.new_tensor([scales[k] for k in keys])[None,:,None,None]
        # Implement training-only pollution normalisation without mutating Aurora globals.
        ratio=published/self.poll_scale if train_normalisation else torch.ones_like(self.poll_scale)
        self.register_buffer("norm_ratio",ratio)
        self.adapter=None if method in ("baseline","wind_pull","column_push") else make_adapter(method,len(scale),width)
        if method not in ("baseline","wind_pull","column_push") and self.adapter is None: raise ValueError(method)
    def forward(self,batch):
        pollution=pack_pollutants(batch)*self.norm_ratio[:,None]
        surf=dict(batch.surf_vars); atmos=dict(batch.atmos_vars); offset=0
        for v in SURF_POLL: surf[v]=pollution[:,:,offset]; offset+=1
        for v in ATMOS_POLL:
            n=len(batch.metadata.atmos_levels); atmos[v]=pollution[:,:,offset:offset+n]; offset+=n
        internal=dataclasses.replace(batch,surf_vars=surf,atmos_vars=atmos)
        pred=self.backbone(internal)
        pred=unpack_pollutants(pred,pack_pollutants(pred)[:,0]/self.norm_ratio)
        if pred.surf_vars["2t"].shape[-2:]!=batch.surf_vars["2t"].shape[-2:]:
            raise ValueError("Aurora cropped the grid; use the supplied cell-centre grid")
        if self.method=="baseline": return pred
        q=pack_pollutants(batch).float()/self.poll_scale[:,None]
        u,v=pollution_winds(batch)
        flow=wind_displacement(u.float(),v.float(),batch.metadata.lat,batch.metadata.lon)
        if self.method=="wind_pull":
            delta=pull(q[:,-1],flow)-q[:,-1]
        elif self.method=="column_push":
            delta=torch.zeros_like(q[:,-1])
            forward=wind_displacement(u[:,3:8].float(),v[:,3:8].float(),batch.metadata.lat,batch.metadata.lon,backward=False)
            delta[:,3:8]=conservative_push(q[:,-1,3:8],forward,batch.metadata.lat)-q[:,-1,3:8]
        else:
            # Transport and Gaussian weights stay in float32 even when Aurora uses AMP.
            with torch.autocast(device_type=q.device.type,enabled=False): delta=self.adapter(q,flow)
        out=pack_pollutants(pred)[:,0].float()+self.poll_scale*delta
        return unpack_pollutants(pred,out)

def build_model(config,stats,load_weights=True):
    cfg=config["model"]; init=cfg["initialisation"]
    common=dict(surf_vars=SURF,atmos_vars=ATMOS,static_vars=STATIC,use_lora=False,autocast=cfg.get("amp",True))
    if init=="tiny_random":
        from datetime import timedelta
        backbone=Aurora(**common,patch_size=4,embed_dim=32,num_heads=4,encoder_depths=(1,1,1),decoder_depths=(1,1,1),encoder_num_heads=(2,4,8),decoder_num_heads=(8,4,2),window_size=(2,6,12),latent_levels=4,timestep=timedelta(hours=12))
    elif init=="pretrained12h":
        backbone=Aurora12hPretrained(**common)
    elif init=="air_pollution":
        if not config.get("allow_preexposed_validation",False):
            raise ValueError("Released air-pollution weights saw historical validation data. Set allow_preexposed_validation only for an explicitly labelled warm-start study.")
        common["use_lora"]=True
        backbone=AuroraAirPollution(**common)
    else: raise ValueError(f"Unknown initialisation: {init}")
    if load_weights and init!="tiny_random":
        # Audit every omitted parameter rather than blindly accepting strict=False.
        def audit(module,incompatible):
            allowed=set(SURF_POLL+ATMOS_POLL+STATIC[3:])
            bad=[k for k in incompatible.missing_keys if not (set(k.split("."))&allowed)]
            if bad or incompatible.unexpected_keys:
                raise RuntimeError(f"Unplanned checkpoint mismatch: missing={bad}, unexpected={incompatible.unexpected_keys}")
        hook=backbone.register_load_state_dict_post_hook(audit)
        try:
            path=cfg.get("checkpoint_path")
            if path:
                if not Path(path).is_file(): raise FileNotFoundError(f"Run scripts/fetch_assets.py first: {path}")
                backbone.load_checkpoint_local(path,strict=(init=="air_pollution"))
            else: backbone.load_checkpoint(strict=(init=="air_pollution"))
        finally: hook.remove()
        if init=="pretrained12h":
            # Preserve pretrained meteorological encoder behaviour on first update.
            with torch.no_grad():
                for name,p in backbone.named_parameters():
                    if name.startswith("encoder.surf_token_embeds.weights.") and name.rsplit(".",1)[-1] in SURF_POLL+STATIC[3:]: p.zero_()
                    if name.startswith("encoder.atmos_token_embeds.weights.") and name.rsplit(".",1)[-1] in ATMOS_POLL: p.zero_()
    if cfg.get("checkpointing",True): backbone.configure_activation_checkpointing()
    scale=[stats["scale"][k] for k in channel_names(pollutants_only=True)]
    model=ForecastModel(backbone,cfg["method"],scale,cfg.get("width",32),train_normalisation=init!="air_pollution")
    mode=cfg.get("trainable","all")
    if mode=="adapter":
        if model.adapter is None: raise ValueError("Adapter-only training requires a learned adapter")
        for p in backbone.parameters(): p.requires_grad_(False)
    elif mode!="all": raise ValueError("trainable must be all or adapter")
    return model
