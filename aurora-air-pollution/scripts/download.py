"""Write explicit ADS retrieval requests; download only with --execute.

The product is always explicit; forecast lead zero is recorded separately.
API credentials are read from ADS_API_KEY or the private .secrets/ads.json file
created interactively by scripts/configure_ads.py on the machine that runs this.
"""
import argparse
from datetime import date,timedelta
import hashlib
import json
import os
from pathlib import Path
import zipfile
from airadv.schema import SURF,ATMOS,ADS_NAMES,LEVELS

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def finalise_download(temp,final):
    # Keep partial extractions distinct from completed files across interruption.
    partial=final.with_suffix(".nc.partial")
    if zipfile.is_zipfile(temp):
        with zipfile.ZipFile(temp) as z:
            files=[n for n in z.namelist() if n.endswith(".nc")]
            if len(files)!=1: raise ValueError(f"Expected one NetCDF group in {temp}, found {files}; inspect archive layout")
            with z.open(files[0]) as inp,partial.open("wb") as out:
                import shutil
                shutil.copyfileobj(inp,out)
    else:
        import shutil
        shutil.copyfile(temp,partial)
    digest=sha256(partial)
    partial.replace(final)
    final.with_suffix(".sha256").write_text(digest+"\n")
    temp.unlink()

def requests_for_day(day,product):
    dataset="cams-global-reanalysis-eac4" if product=="eac4" else "cams-global-atmospheric-composition-forecasts"
    common={"date":day,"time":["00:00","12:00"],"data_format":"netcdf_zip"}
    if product!="eac4": common.update(type="analysis" if product=="cams_analysis" else "forecast",leadtime_hour="0")
    for group,variables in (("sfc",SURF),("plev",ATMOS)):
        request=dict(common,variable=[ADS_NAMES[v] for v in variables])
        if group=="plev": request["pressure_level"]=[str(x) for x in LEVELS]
        yield dataset,group,request

def request_periods(start,end,days_per_request):
    """Keep requests within a calendar month and the explicit exclusive end."""
    if not 1<=days_per_request<=31: raise ValueError("days_per_request must be 1..31")
    while start<end:
        month_end=date(start.year+(start.month==12),start.month%12+1,1)
        stop=min(start+timedelta(days=days_per_request),month_end,end)
        yield start,stop
        start=stop

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product",choices=["cams_analysis","cams_forecast_t0","eac4"],required=True)
    p.add_argument("--start",required=True); p.add_argument("--end",required=True,help="Exclusive end date")
    p.add_argument("--output",default="data/raw"); p.add_argument("--execute",action="store_true")
    p.add_argument("--days-per-request",type=int,default=1,choices=range(1,32))
    a=p.parse_args(); day=date.fromisoformat(a.start); end=date.fromisoformat(a.end)
    if end<=day: p.error("end must be after start")
    root=Path(a.output)/a.product; root.mkdir(parents=True,exist_ok=True)
    layout=root/"request_layout.json"
    settings={"product":a.product,"days_per_request":a.days_per_request}
    if layout.exists() and json.loads(layout.read_text())!=settings:
        raise ValueError("Request grouping changed: use a separate raw output directory to avoid overlapping files")
    if not layout.exists() and a.days_per_request!=1 and any(root.glob("*/*.request.json")):
        raise ValueError("Use a separate raw output directory for grouped requests")
    layout.write_text(json.dumps(settings,indent=2)+"\n")
    client=None; catalogues={}
    if a.execute:
        import cdsapi
        key=os.environ.get("ADS_API_KEY")
        if not key:
            private=Path(".secrets/ads.json")
            if private.exists():
                credentials=json.loads(private.read_text())
                if credentials.get("url")!="https://ads.atmosphere.copernicus.eu/api":
                    raise ValueError("Private configuration must use the official ADS endpoint")
                key=credentials.get("key")
        if not key: raise RuntimeError("Run scripts/configure_ads.py privately on the data/GPU host and accept dataset terms in ADS")
        client=cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api",key=key)
    count=0
    for day,stop in request_periods(day,end,a.days_per_request):
        folder=root/day.strftime("%Y-%m"); folder.mkdir(exist_ok=True)
        last=stop-timedelta(days=1)
        for dataset,group,request in requests_for_day(day.isoformat(),a.product):
            period=str(day) if last==day else f"{day}_to_{last}"
            if last!=day: request["date"]=[f"{day}/{last}"]
            prefix=folder/f"{period}-{group}"; request_path=prefix.with_suffix(".request.json")
            plan={"dataset":dataset,"request":request,"product":a.product}
            if request_path.exists() and json.loads(request_path.read_text())!=plan: raise ValueError(f"Existing request differs: {request_path}")
            request_path.write_text(json.dumps(plan,indent=2)+"\n")
            final=prefix.with_suffix(".nc")
            if client and final.exists():
                receipt=prefix.with_suffix(".sha256")
                if not receipt.exists() or receipt.read_text().strip()!=sha256(final):
                    raise ValueError(f"Unverified or changed completed file: {final}; inspect before resuming")
            if client and not final.exists():
                from airadv.catalogue import fetch_catalogue,unavailable_fields
                if dataset not in catalogues:
                    catalogues[dataset]=fetch_catalogue(dataset,root/"catalogue")
                catalogue=catalogues[dataset]
                formats=next(x["details"]["values"] for x in catalogue["form"] if x.get("name")=="data_format")
                if request["data_format"] not in formats: raise ValueError("ADS format schema changed; inspect saved catalogue")
                check_day=day
                while check_day<stop:
                    missing=unavailable_fields(dict(request,date=check_day.isoformat()),group,catalogue["constraints"])
                    if missing: raise ValueError(f"ADS catalogue has no requested {a.product} fields on {check_day}: {missing}. Do not silently substitute forecast values.")
                    check_day+=timedelta(days=1)
                temp=prefix.with_suffix(".download")
                client.retrieve(dataset,request,str(temp))
                finalise_download(temp,final)
            count+=1
    print(f"{'Executed' if client else 'Prepared (no network requests)'} {count} requests in {root}")
    print("Pilot one day first: ADS availability/schema changes must be checked before a historical bulk retrieval.")
if __name__=="__main__": main()

