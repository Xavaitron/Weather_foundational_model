"""Real Aurora code, tiny random architecture, no checkpoint/data download."""
import argparse
import dataclasses
import json
import time
from datetime import timedelta
from pathlib import Path
import torch
from airadv.synthetic import synthetic_batch,synthetic_stats
from airadv.model import build_model
from airadv.objective import weighted_loss
from airadv.data import advance

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu"); p.add_argument("--full-grid",action="store_true")
    p.add_argument("--output",default="runs/smoke.json")
    a=p.parse_args(); torch.set_num_threads(2)
    stats=synthetic_stats(); batch=synthetic_batch(120,240) if a.full_grid else synthetic_batch()
    batch=batch.to(a.device)
    target=dataclasses.replace(batch,surf_vars={k:v[:,-1:] for k,v in batch.surf_vars.items()},atmos_vars={k:v[:,-1:] for k,v in batch.atmos_vars.items()},metadata=dataclasses.replace(batch.metadata,time=(batch.metadata.time[0]+timedelta(hours=12),)))
    results=[]
    for method in ("baseline","nowcast_evolution","adr_latent","debezenac_gaussian","wind_pull","column_push"):
        torch.manual_seed(4)
        cfg={"model":{"initialisation":"tiny_random","method":method,"amp":False,"checkpointing":False,"width":8}}
        model=build_model(cfg,stats).to(a.device); model.train()
        t=time.perf_counter(); pred=model(batch)
        loss=weighted_loss(pred,target,stats); loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert pred.metadata.time[0]==target.metadata.time[0]
        assert pred.surf_vars["pm2p5"].shape==(1,1,*batch.surf_vars["pm2p5"].shape[-2:])
        assert advance(batch,pred).surf_vars["pm2p5"].shape[1]==2
        record={"method":method,"loss_synthetic":float(loss.detach()),"seconds":time.perf_counter()-t,"finite_gradients":True,"parameters":sum(p.numel() for p in model.parameters())}
        results.append(record); print(json.dumps(record),flush=True)
        del model,pred,loss
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"data":"SYNTHETIC_ONLY","weights":"RANDOM_TINY_AURORA","device":a.device,"grid":list(batch.surf_vars["2t"].shape[-2:]),"results":results},indent=2)+"\n")
if __name__=="__main__": main()

