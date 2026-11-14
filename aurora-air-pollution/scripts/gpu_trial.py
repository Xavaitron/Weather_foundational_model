"""Run two real pretrained optimizer steps and measure peak CUDA memory."""
import argparse
import json
from pathlib import Path
import torch
from airadv.train import read_json,validate_config
from airadv.data import CAMSDataset
from airadv.model import build_model
from airadv.objective import weighted_loss

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True); p.add_argument("--method",default="baseline")
    p.add_argument("--output",default="runs/gpu_trial.json")
    p.add_argument("--max-memory-gib",type=float,default=28.)
    p.add_argument("--reserve-gib",type=float,default=4.)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("A CUDA GPU is required")
    free,total=torch.cuda.mem_get_info()
    limit=min(a.max_memory_gib*2**30,free-a.reserve_gib*2**30)
    if limit<=0: raise RuntimeError("Insufficient free GPU memory after safety reserve")
    if torch.cuda.memory.get_allocator_backend()!="native":
        raise RuntimeError("Use the native allocator so the shared-GPU memory cap is enforced")
    torch.cuda.set_per_process_memory_fraction(limit/total)
    cfg=read_json(a.config); cfg["model"]["method"]=a.method
    stats=read_json(cfg["data"]["stats"]); validate_config(cfg,stats)
    if cfg["model"]["initialisation"]=="tiny_random": raise ValueError("Use a real checkpoint for this test")
    if cfg["model"].get("amp",True) and not torch.cuda.is_bf16_supported(): raise ValueError("bfloat16 is unsupported")
    d=cfg["data"]; data=CAMSDataset(d["path"],d["train_start"],d["train_end"])
    if data.ds.attrs.get("product")!=d["product"]: raise ValueError("Dataset product mismatch")
    torch.manual_seed(cfg["training"]["seed"]); torch.cuda.reset_peak_memory_stats()
    model=build_model(cfg,stats).cuda().train()
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg["training"]["lr"],weight_decay=cfg["training"].get("weight_decay",0.))
    losses=[]
    for i in range(2):
        batch,targets=data[i%len(data)]; opt.zero_grad(set_to_none=True)
        pred=model(batch.to("cuda")); loss=weighted_loss(pred,targets[0].to("cuda"),stats)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); opt.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    result={"gpu":torch.cuda.get_device_name(),"method":a.method,"data":d["path"],"product":d["product"],"losses":losses,"memory_limit_GiB":limit/2**30,"peak_allocated_GB":torch.cuda.max_memory_allocated()/1e9,"peak_reserved_GB":torch.cuda.max_memory_reserved()/1e9,"weights_saved":False,"forecast_skill_measured":False}
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()

