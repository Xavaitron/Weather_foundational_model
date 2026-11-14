"""Convert evaluation files to a comparable CSV with persistence skill."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("files",nargs="+"); p.add_argument("--output",default="runs/comparison.csv"); a=p.parse_args()
    rows=[]; reference=None
    for file in a.files:
        data=json.loads(Path(file).read_text()); identity=(data["test"],data["origins"],data["product"],data["steps"])
        if reference is None: reference=identity
        if identity!=reference: raise ValueError("Different evaluation windows/counts/stride; rerun on identical origins")
        lookup={(m["method"],m["region"],m["lead_hours"],m["variable"]):m for m in data["metrics"]}
        for r in data["metrics"]:
            if r["method"]!="model": continue
            key=(r["region"],r["lead_hours"],r["variable"])
            persist=lookup[("persistence",*key)]["rmse"]
            diurnal=lookup[("diurnal_persistence",*key)]["rmse"]
            rows.append(dict(run=Path(file).stem,**r,persistence_rmse=persist,diurnal_rmse=diurnal,skill_vs_persistence=1-r["rmse"]/persist if persist else "",skill_vs_diurnal=1-r["rmse"]/diurnal if diurnal else ""))
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)
if __name__=="__main__": main()

