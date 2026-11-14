"""Single-GPU full fine-tuning with exact optimizer-boundary resume."""
import argparse
import copy
import hashlib
import json
import platform
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .data import CAMSDataset,advance
from .model import build_model
from .objective import weighted_loss

def read_json(path): return json.loads(Path(path).read_text())
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def resolve_device(name):
    device=torch.device(name)
    if device.type=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; use scripts/smoke.py for CPU validation")
        # Allocator limits require an explicit process-local CUDA index.
        device=torch.device("cuda",torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    return device

def validate_config(cfg,stats):
    d=cfg["data"]
    if not (d["train_start"]<d["train_end"]<=d["val_start"]<d["val_end"]<=d["test_start"]<d["test_end"]):
        raise ValueError("Chronological split boundaries overlap or are not ordered")
    if stats["start"]<d["train_start"] or stats["end_exclusive"]>d["train_end"]:
        raise ValueError("Statistics include dates outside the training split")
    if str(Path(stats["source"]).resolve())!=str(Path(d["path"]).resolve()):
        raise ValueError("Statistics belong to a different data store")
    if cfg["training"]["grad_accum"]<1: raise ValueError("grad_accum must be positive")

@torch.no_grad()
def validation(model,ds,stats,device,limit):
    model.eval(); total=0
    idx=np.unique(np.linspace(0,len(ds)-1,min(limit,len(ds)),dtype=int))
    for i in idx:
        history,targets=ds[int(i)]; history=history.to(device)
        for target in targets:
            pred=model(history); total+=float(weighted_loss(pred,target.to(device),stats))/len(targets)
            history=advance(history,pred,detach=True)
    model.train()
    return total/len(idx)

def run(cfg,resume=None,initial_state=None):
    stats=read_json(cfg["data"]["stats"]); validate_config(cfg,stats)
    run_dir=Path(cfg["output"])
    if (run_dir/"run.json").exists() and not resume:
        raise FileExistsError("Run directory already contains an experiment; use --resume or a new --output")
    device=resolve_device(cfg["training"].get("device","cuda"))
    if device.type=="cuda" and cfg["model"].get("amp",True) and not torch.cuda.is_bf16_supported(): raise RuntimeError("This AMP configuration needs bfloat16 support")
    memory_limit=None
    if device.type=="cuda" and cfg["training"].get("max_memory_gib") is not None:
        if torch.cuda.memory.get_allocator_backend()!="native": raise RuntimeError("Use native CUDA allocation to enforce the shared-GPU memory cap")
        free,total=torch.cuda.mem_get_info(device)
        memory_limit=min(cfg["training"]["max_memory_gib"]*2**30,free-cfg["training"].get("reserve_gib",4.)*2**30)
        if memory_limit<=0: raise RuntimeError("Insufficient free GPU memory after reserve")
        torch.cuda.set_per_process_memory_fraction(memory_limit/total,device)
    if device.type!="cuda": cfg["model"]["amp"]=False
    seed=cfg["training"]["seed"]; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng=np.random.default_rng(seed)
    d=cfg["data"]; steps=cfg["training"].get("rollout_steps",1)
    train=CAMSDataset(d["path"],d["train_start"],d["train_end"],steps)
    val=CAMSDataset(d["path"],d["val_start"],d["val_end"],steps)
    if train.ds.attrs.get("product")!=cfg["data"]["product"]: raise ValueError("Dataset provenance/product mismatch")
    if d["product"]=="eac4" and d["test_end"]>"2022-01-01": raise ValueError("EAC4 stage must not silently become the CAMS analysis benchmark")
    model=build_model(cfg,stats,load_weights=not (resume or initial_state)).to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg["training"]["lr"],weight_decay=cfg["training"].get("weight_decay",0.))
    run_dir.mkdir(parents=True,exist_ok=True)
    fingerprint={"model":cfg["model"],"data":d,"stats":digest(stats),"seed":seed,"rollout_steps":steps,"grad_accum":cfg["training"]["grad_accum"],"lr":cfg["training"]["lr"],"weight_decay":cfg["training"].get("weight_decay",0.),"warmup":cfg["training"].get("warmup",100)}
    start=0; best=float("inf")
    if resume:
        # Load only checkpoints created by this project, never unknown pickle files.
        ck=torch.load(resume,map_location="cpu",weights_only=False)
        if ck["fingerprint"]!=fingerprint: raise ValueError("Resume settings/data/stats differ from checkpoint")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start=ck["step"]; best=ck["best_val"]; rng.bit_generator.state=ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        if device.type=="cuda": torch.cuda.set_rng_state_all(ck["cuda_rng"])
    elif initial_state:
        ck=torch.load(initial_state,map_location="cpu",weights_only=False)
        if ck["config"]["model"]["initialisation"]!=cfg["model"]["initialisation"]: raise ValueError("Stage initialisation architecture mismatch")
        state=ck["model"]
        # Dataset-dependent branch scales must come from this stage's training data.
        state={k:v for k,v in state.items() if k not in ("poll_scale","norm_ratio")}
        result=model.load_state_dict(state,strict=False)
        if set(result.missing_keys)!={"poll_scale","norm_ratio"} or result.unexpected_keys: raise ValueError(str(result))
    meta={"config":cfg,"stats_hash":digest(stats),"torch":torch.__version__,"python":platform.python_version(),"trainable_parameters":sum(p.numel() for p in params),"train_windows":len(train),"val_windows":len(val),"data_provenance":dict(train.ds.attrs),"memory_limit_GiB":None if memory_limit is None else memory_limit/2**30}
    (run_dir/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
    def save(name,step):
        if shutil.disk_usage(run_dir).free<cfg["training"].get("min_free_disk_gib",0.)*2**30:
            raise RuntimeError("Storage below checkpoint reserve; last completed checkpoint is preserved")
        payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,"best_val":best,"config":cfg,"stats":stats,"fingerprint":fingerprint,"rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else []}
        print(json.dumps({"event":"checkpoint_started","path":str(run_dir/name),"step":step}),flush=True)
        tmp=run_dir/(name+".tmp"); torch.save(payload,tmp); tmp.replace(run_dir/name)
        print(json.dumps({"event":"checkpoint_saved","path":str(run_dir/name),"step":step}),flush=True)
    model.train()
    total_steps=cfg["training"]["steps"]; accum=cfg["training"]["grad_accum"]
    for step in range(start,total_steps):
        started=time.perf_counter()
        opt.zero_grad(set_to_none=True); train_loss=0.
        warmup=cfg["training"].get("warmup",100)
        for g in opt.param_groups: g["lr"]=cfg["training"]["lr"]*min(1.,(step+1)/max(1,warmup))
        for _ in range(accum):
            history,targets=train[int(rng.integers(len(train)))]; history=history.to(device)
            loss=0
            for target in targets:
                pred=model(history); loss=loss+weighted_loss(pred,target.to(device),stats)/len(targets)
                history=advance(history,pred)
            train_loss+=float(loss.detach())/accum; (loss/accum).backward()
        norm=torch.nn.utils.clip_grad_norm_(params,cfg["training"].get("clip",1.),error_if_nonfinite=True)
        opt.step()
        record={"step":step+1,"train_loss":train_loss,"grad_norm":float(norm),"lr":opt.param_groups[0]["lr"],"optimizer_step_seconds":time.perf_counter()-started}
        if (step+1)%cfg["training"]["validate_every"]==0 or step+1==total_steps:
            score=validation(model,val,stats,device,cfg["training"].get("val_samples",64)); record["val_loss"]=score
            if score<best: best=score; save("best.pt",step+1)
            save("last.pt",step+1)
        print(json.dumps(record),flush=True)
        with (run_dir/"train.jsonl").open("a") as f: f.write(json.dumps(record)+"\n")
    return model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True); p.add_argument("--resume"); p.add_argument("--initial-state")
    p.add_argument("--method",choices=["baseline","nowcast_evolution","debezenac_gaussian","adr_latent","wind_pull","column_push"])
    p.add_argument("--seed",type=int); p.add_argument("--output")
    a=p.parse_args()
    if a.resume and a.initial_state: p.error("Use resume or initial-state, not both")
    cfg=read_json(a.config)
    if a.method: cfg["model"]["method"]=a.method
    if a.seed is not None: cfg["training"]["seed"]=a.seed
    if a.output: cfg["output"]=a.output
    run(cfg,a.resume,a.initial_state)
if __name__=="__main__": main()

