"""Autoregressive forecasts compared to fixed and diurnal persistence."""
import argparse
import dataclasses
import json
from pathlib import Path
import torch
from .data import CAMSDataset,advance
from .model import build_model
from .objective import MetricAccumulator

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int,default=10); p.add_argument("--stride",type=int,default=2)
    p.add_argument("--device",default="cuda"); p.add_argument("--limit",type=int)
    a=p.parse_args()
    if a.steps<1 or a.stride<1: p.error("steps and stride must be positive")
    if a.limit is not None and a.limit<1: p.error("limit must be positive")
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    cfg,stats=ck["config"],ck["stats"]; cfg["model"]["checkpointing"]=False
    cfg["model"]["amp"]=a.device.startswith("cuda")
    model=build_model(cfg,stats,load_weights=False).to(a.device).eval()
    model.load_state_dict(ck["model"])
    d=cfg["data"]; ds=CAMSDataset(d["path"],d["test_start"],d["test_end"],a.steps)
    if ds.ds.attrs.get("product")!=d["product"]: raise ValueError("Product mismatch")
    meters={(method,region,step):MetricAccumulator() for method in ("model","persistence","diurnal_persistence") for region in ("global","india") for step in range(1,a.steps+1)}
    indices=list(range(0,len(ds),a.stride)); indices=indices[:a.limit] if a.limit else indices
    for count,i in enumerate(indices):
        initial,targets=ds[i]; initial=initial.to(a.device); history=initial
        for step,target in enumerate(targets,1):
            target=target.to(a.device); pred=model(history)
            def persistence(index):
                return dataclasses.replace(target,surf_vars={k:v[:,index:index+1] for k,v in initial.surf_vars.items()},atmos_vars={k:v[:,index:index+1] for k,v in initial.atmos_vars.items()})
            predictions={"model":pred,"persistence":persistence(1),"diurnal_persistence":persistence(0 if step%2 else 1)}
            for method,value in predictions.items():
                for region in ("global","india"): meters[method,region,step].update(value,target,region)
            history=advance(history,pred,detach=True)
        print(f"Evaluated {count+1}/{len(indices)} initialisations",flush=True)
    results=[]
    for (method,region,step),meter in meters.items():
        for variable,values in meter.compute().items():
            results.append(dict(method=method,region=region,lead_hours=step*12,variable=variable,**values))
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    origins=[str(ds.ds.time.values[ds.indices[i]]) for i in indices]
    output.write_text(json.dumps({"initialisations":len(indices),"origins":origins,"product":d["product"],"stride":a.stride,"steps":a.steps,"checkpoint":a.checkpoint,"test":[d["test_start"],d["test_end"]],"metrics":results},indent=2)+"\n")
if __name__=="__main__": main()
