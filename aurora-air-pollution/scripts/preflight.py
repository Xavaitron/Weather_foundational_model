"""Read-only checks before allocating an expensive training run."""
import argparse
import json
import shutil
from pathlib import Path
from importlib.metadata import version
import torch
from airadv.data import CAMSDataset
from airadv.train import validate_config,read_json

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True); a=p.parse_args()
    cfg=read_json(a.config); stats=read_json(cfg["data"]["stats"]); validate_config(cfg,stats)
    report={"torch":torch.__version__,"aurora":version("microsoft-aurora"),"cuda":torch.cuda.is_available(),"free_disk_GB":shutil.disk_usage(Path.cwd()).free/1e9,"splits":{}}
    if torch.cuda.is_available(): report.update(gpu=torch.cuda.get_device_name(0),vram_GB=torch.cuda.get_device_properties(0).total_memory/1e9,bf16=torch.cuda.is_bf16_supported())
    d=cfg["data"]
    for split in ("train","val","test"):
        ds=CAMSDataset(d["path"],d[split+"_start"],d[split+"_end"],1 if split!="test" else 10)
        ds[0]; ds[len(ds)//2]; ds[-1]
        report["splits"][split]={"windows":len(ds),"product":ds.ds.attrs.get("product"),"first_origin":str(ds.ds.time.values[ds.indices[0]]),"last_origin":str(ds.ds.time.values[ds.indices[-1]])}
        if ds.ds.attrs.get("product")!=d["product"]: raise ValueError("Unexpected data product")
    print(json.dumps(report,indent=2))
    print("Data samples passed; peak GPU memory and pretrained forward/backward still require a GPU trial.")
if __name__=="__main__": main()

