"""Fetch pinned official Aurora assets directly on the GPU/data host."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata

ASSETS={
    "static":("aurora-0.4-air-pollution-static.pickle","1764d5630a53d3d7a7d169ca335236fc343e4bfc"),
    "pretrained12h":("aurora-0.25-12h-pretrained.ckpt","15e76e47b65bf4b28fd2246b7b5b951d6e2443b9"),
    "air_pollution":("aurora-0.4-air-pollution.ckpt","1764d5630a53d3d7a7d169ca335236fc343e4bfc"),
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",nargs="+",choices=list(ASSETS),default=["static","pretrained12h"])
    p.add_argument("--output",default="data/assets")
    a=p.parse_args(); root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    records=[]
    for key in a.assets:
        name,revision=ASSETS[key]
        metadata=get_hf_file_metadata(hf_hub_url("microsoft/aurora",filename=name,revision=revision))
        expected=metadata.etag
        if not expected or len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Expected an official LFS SHA-256 for {name}")
        print(f"Fetching {name} ({metadata.size} bytes)",flush=True)
        file=Path(hf_hub_download("microsoft/aurora",filename=name,revision=revision,local_dir=root))
        h=hashlib.sha256()
        with file.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
        if h.hexdigest()!=expected or file.stat().st_size!=metadata.size:
            raise ValueError(f"Publisher hash/size mismatch for {name}; do not load this asset")
        records.append(dict(asset=key,file=str(file),revision=revision,sha256=h.hexdigest(),bytes=file.stat().st_size,publisher_hash_verified=True))
    (root/"assets.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps(records,indent=2))
if __name__=="__main__": main()

