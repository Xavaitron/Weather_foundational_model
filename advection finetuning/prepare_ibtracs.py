import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Normalize an official IBTrACS v04r01 CSV; no gridding of positions.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/ibtracs.csv")
    args = p.parse_args()
    # NOAA CSV has a second header row containing units. Parsing dates below
    # naturally excludes it, and also accepts ordinary one-header CSV files.
    raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    raw.columns = raw.columns.str.strip()
    for column in raw:
        raw[column] = raw[column].str.strip()
    required = {"SID", "ISO_TIME", "LAT", "LON", "BASIN", "NATURE"}
    if not required.issubset(raw):
        raise ValueError(f"Missing IBTrACS columns: {required-set(raw)}")
    if "TRACK_TYPE" in raw:
        raw = raw[raw.TRACK_TYPE.isin(["MAIN", "PROVISIONAL"])]
    out = pd.DataFrame({"sid": raw.SID, "time": pd.to_datetime(raw.ISO_TIME, utc=True, errors="coerce"),
                        "lat": pd.to_numeric(raw.LAT, errors="coerce"),
                        "lon": pd.to_numeric(raw.LON, errors="coerce"), "basin": raw.BASIN, "nature": raw.NATURE})
    out = out.dropna(subset=["time", "lat", "lon"])
    out = out[(out.lat.abs() <= 90) & (out.lon.abs() <= 360)]
    out["lon"] = out.lon % 360
    if out.duplicated(["sid", "time"]).any():
        raise ValueError("Duplicate storm/time rows remain. Select one authoritative track before scoring.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    out.sort_values(["sid", "time"]).to_csv(output, index=False)
    print(f"Saved {len(out)} observations from {out.sid.nunique()} storms to {output}")


if __name__ == "__main__":
    main()
