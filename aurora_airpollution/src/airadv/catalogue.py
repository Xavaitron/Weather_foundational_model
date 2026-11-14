"""Check public ADS availability before submitting authenticated retrievals."""
import hashlib
import json
from pathlib import Path


def fetch_catalogue(dataset, output):
    import requests
    base = "https://ads.atmosphere.copernicus.eu/api/catalogue/v1/collections/"
    response = requests.get(base + dataset, timeout=60)
    response.raise_for_status()
    collection = response.json()
    documents = {"collection": collection}
    urls = {"collection": base + dataset}
    for kind in ("form", "constraints"):
        url = next(x["href"] for x in collection["links"] if x["rel"] == kind)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        documents[kind] = r.json()
        urls[kind] = url
    root = Path(output) / dataset
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for kind, document in documents.items():
        encoded = (json.dumps(document, indent=2) + "\n").encode()
        (root / (kind + ".json")).write_bytes(encoded)
        hashes[kind] = {"url": urls[kind], "sha256": hashlib.sha256(encoded).hexdigest()}
    (root / "sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return documents


def unavailable_fields(request, group, constraints):
    """Return variable/time pairs missing any requested level for one UTC day.

    Analysis is intentionally validated against the constraints, because ADS's
    web form hides it. ECMWF documents analysis API access with leadtime=0,
    while analysis constraint rows omit the leadtime key.
    """
    day = request["date"]
    rows = []
    for row in constraints:
        if not any(span.split("/")[0] <= day <= span.split("/")[-1] for span in row["date"]):
            continue
        if "type" in request and request["type"] not in row.get("type", []):
            continue
        if request.get("type") == "forecast" and request["leadtime_hour"] not in row.get("leadtime_hour", []):
            continue
        if group == "sfc" and ("model_level" in row or "pressure_level" in row):
            continue
        if group == "plev" and "pressure_level" not in row:
            continue
        rows.append(row)
    missing = []
    for variable in request["variable"]:
        for hour in request["time"]:
            matching = [row for row in rows if variable in row["variable"] and hour in row["time"]]
            if group == "plev":
                levels = {x for row in matching for x in row["pressure_level"]}
                okay = set(request["pressure_level"]) <= levels
            else:
                okay = bool(matching)
            if not okay:
                missing.append(variable + "@" + hour)
    return missing
