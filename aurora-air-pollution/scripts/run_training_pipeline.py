"""Supervise saved real-data training and sequential historical ADS retrieval.

Linux server only. No synthetic data or pilot-only optimizer script is used.
The January run is preliminary; full comparisons restart from official weights.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OPS=ROOT/"runs/operations"
METHODS=("baseline","nowcast_evolution","adr_latent","debezenac_gaussian")

def status(worker,phase,**details):
    payload=dict(phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**details)
    p=OPS/f"{worker}.json"; tmp=p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload,indent=2)+"\n"); tmp.replace(p)
    print(json.dumps(payload),flush=True)

def call(args,log):
    with (OPS/log).open("a",buffering=1) as out:
        out.write(json.dumps({"command":[sys.executable,*args]})+"\n")
        subprocess.run([sys.executable,"-u",*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)

def prepare_stats(config,prefix):
    d=json.loads((ROOT/config).read_text())["data"]
    call(["scripts/stats.py","--data",d["path"],"--start",d["train_start"],"--end",d["train_end"],"--output",d["stats"]],prefix+"_stats.log")
    call(["scripts/preflight.py","--config",config],prefix+"_preflight.log")

def historical_data():
    for product,start,end,output,config,ready in [
        ("eac4","2003-01-01","2022-01-01","data/processed/eac4_1p5.zarr","configs/eac4.json","eac4"),
        ("cams_forecast_t0","2017-10-01","2022-12-01","data/processed/cams_t0_1p5.zarr","configs/cams_t0.json","cams_t0"),
    ]:
        if (OPS/f"{ready}_ready.json").exists(): continue
        status("historical_data","building",product=product,start=start,end_exclusive=end)
        call(["scripts/build_dataset.py","--product",product,"--start",start,"--end",end,"--raw","data/raw_grouped","--days-per-request","31","--output",output,"--execute","--discard-verified-raw"],ready+"_historical_data.log")
        status("historical_data","fitting_training_statistics",product=product)
        prepare_stats(config,ready)
        (OPS/f"{ready}_ready.json").write_text(json.dumps(dict(product=product,path=output,config=config,full_store_verified=True))+"\n")
    status("historical_data","completed")

def wait_for_data(ready,worker):
    while not (OPS/f"{ready}_ready.json").exists():
        if worker.poll() is not None:
            raise RuntimeError(f"Historical data worker exited before {ready} was ready; inspect historical_data.json and its log")
        time.sleep(30)

def train(config,method,output,initial=None):
    args=["-m","airadv.train","--config",config,"--method",method,"--output",output]
    last=ROOT/output/"last.pt"
    if last.exists(): args.extend(["--resume",str(last)])
    elif initial: args.extend(["--initial-state",initial])
    call(args,Path(output).name+".log")

def evaluate(output):
    # Test data are evaluated once per completed run, after model selection.
    report=ROOT/output/"evaluation.json"
    if not report.exists():
        call(["-m","airadv.evaluate","--checkpoint",str(ROOT/output/"best.pt"),"--output",str(report)],Path(output).name+"_evaluate.log")

def training(starter_pid):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Select the allocated GPU UUID in CUDA_VISIBLE_DEVICES")
    status("training_pipeline","waiting_for_starter_data",data_pid=starter_pid)
    if starter_pid:
        while True:
            try: os.kill(starter_pid,0)
            except ProcessLookupError: break
            time.sleep(15)
    if "Completed" not in (OPS/"starter_data.log").read_text().splitlines():
        raise RuntimeError("Starter data builder did not complete successfully; inspect starter_data.log")
    status("training_pipeline","starter_statistics_and_preflight")
    prepare_stats("configs/starter_eac4_jan2003.json","starter")
    history_log=(OPS/"historical_supervisor.log").open("a",buffering=1)
    history=subprocess.Popen([sys.executable,"-u",__file__,"--historical-data-worker"],cwd=ROOT,stdout=history_log,stderr=subprocess.STDOUT,start_new_session=True)
    (OPS/"historical_data.pid").write_text(str(history.pid)+"\n")
    starter="runs/starter_eac4_jan2003_baseline_seed42"
    status("training_pipeline","starter_training",output=starter,steps=200,historical_data_pid=history.pid)
    train("configs/starter_eac4_jan2003.json","baseline",starter)
    status("training_pipeline","starter_evaluation",output=starter)
    evaluate(starter)
    for method in METHODS:
        status("training_pipeline","waiting_for_eac4_history",method=method)
        wait_for_data("eac4",history)
        pre=f"runs/eac4_{method}_seed42"
        status("training_pipeline","full_eac4_training",method=method,output=pre)
        # Every full experiment starts afresh from the official pretrained model.
        train("configs/eac4.json",method,pre)
        status("training_pipeline","waiting_for_cams_history",method=method)
        wait_for_data("cams_t0",history)
        run=f"runs/cams_t0_{method}_seed42"
        status("training_pipeline","cams_finetuning",method=method,output=run)
        train("configs/cams_t0.json",method,run,str(ROOT/pre/"best.pt"))
        status("training_pipeline","held_out_evaluation",method=method,output=run)
        evaluate(run)
    reports=[f"runs/cams_t0_{method}_seed42/evaluation.json" for method in METHODS]
    call(["scripts/compare.py",*reports,"--output","runs/comparison_seed42.csv"],"comparison_seed42.log")
    status("training_pipeline","completed",comparison="runs/comparison_seed42.csv",seeds=[42])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-data-worker",action="store_true")
    p.add_argument("--starter-data-pid",type=int)
    p.add_argument("--plan",action="store_true")
    a=p.parse_args()
    if a.plan:
        print(json.dumps({"starter":"configs/starter_eac4_jan2003.json","stages":["configs/eac4.json","configs/cams_t0.json"],"methods":METHODS,"seeds":[42],"historical_downloads":"sequential monthly requests; delete raw only after verified preprocessing","gpu":"explicit UUID required; native allocator; configured memory cap","status":"runs/operations/training_pipeline.json"},indent=2)); return
    os.chdir(ROOT); OPS.mkdir(parents=True,exist_ok=True)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","DASK_NUM_WORKERS"):
        os.environ[key]="4"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="backend:native"
    worker="historical_data" if a.historical_data_worker else "training_pipeline"
    with (OPS/f"{worker}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.historical_data_worker: historical_data()
            else: training(a.starter_data_pid)
        except Exception as exc:
            status(worker,"failed",error_type=type(exc).__name__,error=str(exc))
            traceback.print_exc(); raise

if __name__=="__main__": main()
