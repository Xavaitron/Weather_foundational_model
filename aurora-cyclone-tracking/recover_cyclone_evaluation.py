"""Recover a completed cyclone evaluation after the NA-basin parsing failure.

Run from advection finetuning. Requires confirmation that all initializations
finished; this script cannot resume partially completed forecasting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


def patch_csv_readers(root):
    fixes = {
        "evaluate_cyclones.py": [
            ('pd.read_csv(args.ibtracs)', 'pd.read_csv(args.ibtracs, keep_default_na=False)'),
        ],
        "compare_results.py": [
            ('pd.read_csv(reference_dir/"tracks.csv")',
             'pd.read_csv(reference_dir/"tracks.csv", keep_default_na=False)'),
            ('pd.read_csv(directory/"tracks.csv")',
             'pd.read_csv(directory/"tracks.csv", keep_default_na=False)'),
        ],
    }
    staged = []
    for filename, replacements in fixes.items():
        path = root / filename
        source = path.read_text()
        for old, new in replacements:
            if old in source:
                source = source.replace(old, new)
            elif new not in source:
                raise ValueError(f"Unexpected source in {filename}; no automatic patch applied")
        compile(source, str(path), "exec")
        staged.append((path, source))
    for path, source in staged:
        path.write_text(source)


def repair_basins(frame, truth):
    """Recover labels from the exact storm/initialization, never guess a basin."""
    frame = frame.copy()
    truth = truth.copy()
    truth["time"] = pd.to_datetime(truth["time"], utc=True)
    if truth.duplicated(["sid", "time"]).any():
        raise ValueError("IBTrACS contains duplicate storm/time pairs")
    index = pd.MultiIndex.from_arrays([frame["sid"], pd.to_datetime(frame["init_time"], utc=True)])
    expected = truth.set_index(["sid", "time"])["basin"].reindex(index).reset_index(drop=True)
    expected = expected.astype("string").str.strip().str.upper()
    if not expected.isin(["NA", "EP", "WP", "NI", "SI", "SP", "SA"]).all():
        raise ValueError("Cannot recover every basin from IBTrACS at the recorded initialization times")
    current = frame["basin"].fillna("").astype(str).str.strip().str.upper().reset_index(drop=True)
    if ((current != "") & (current != expected)).any():
        raise ValueError("Saved basin labels conflict with supplied IBTrACS; verify the source file")
    frame["basin"] = expected.to_numpy()
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/cyclone/standard_seed42")
    p.add_argument("--ibtracs", default="data/ibtracs_corrected.csv")
    p.add_argument("--forecasts-complete", action="store_true",
                   help="Confirm the log reached the final initialization before summary failed")
    p.add_argument("--leads", type=int, nargs="+", default=[24, 48, 72, 96, 120])
    p.add_argument("--issue-delay-hours", type=int, choices=[0, 6], default=0)
    p.add_argument("--nature", nargs="+", default=["TS", "DS"])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--max-inits", type=int)
    args = p.parse_args()
    if not args.forecasts_complete:
        p.error("Use --forecasts-complete only after the final initialization finished")
    root = Path.cwd()
    patch_csv_readers(root)
    # Imports follow patching; no GPU model or weather store is loaded.
    sys.path.insert(0, str(root))
    from advection_ft.metrics import summarise_tracks
    run = Path(args.run)
    output = run / "cyclone_evaluation"
    if (output / "evaluation.json").exists():
        raise FileExistsError("Evaluation already finalized; existing results were preserved")
    tracks = output / "tracks.csv"
    frame = pd.read_csv(tracks, keep_default_na=False)
    if frame.empty:
        raise ValueError("No saved forecast tracks to recover")
    truth = pd.read_csv(args.ibtracs, keep_default_na=False)
    frame = repair_basins(frame, truth)
    if not set(args.leads).issubset(set(frame["lead_hours"])):
        raise ValueError("Saved records do not cover every requested lead")
    if "tracker_failed" in frame and frame["tracker_failed"].dtype != bool:
        values = frame["tracker_failed"].astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError("Invalid tracker failure flags")
        frame["tracker_failed"] = values.eq("true")
    # saved config.json is the configuration used by this run.
    config = json.loads((run / "config.json").read_text())
    approach = json.loads((run / "manifest.json").read_text())["approach"]
    config["activation_checkpointing"] = False
    summary = summarise_tracks(frame, args.leads, args.bootstrap, config["seed"])
    if summary.empty:
        raise ValueError("No scoreable cases")
    metadata = {"approach": approach, "leads": args.leads,
                "issue_delay_hours": args.issue_delay_hours, "nature": args.nature,
                "bootstrap_repetitions": args.bootstrap, "max_inits": args.max_inits,
                "ibtracs_sha256": hashlib.sha256(Path(args.ibtracs).read_bytes()).hexdigest(),
                "config": config,
                "tracker": "coarse_MSL_Z700_local_minimum_quadratic_refinement_persist_failure",
                "note": "Deterministic, initialized-track verification; no genesis/detection or ensemble calibration claim.",
                "recovery": "Summary regenerated from completed saved tracks; basin labels matched to IBTrACS at initialization."}
    backup = output / "tracks_before_basin_fix.csv"
    if not backup.exists():
        backup.write_bytes(tracks.read_bytes())
    for name, data in [("tracks.csv", frame), ("summary.csv", summary)]:
        temporary = output / (name + ".tmp")
        data.to_csv(temporary, index=False)
        temporary.replace(output / name)
    temporary = output / "evaluation.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    temporary.replace(output / "evaluation.json")
    print(summary.to_string(index=False))
    print(f"Recovered {output}; no forecast inference was repeated.")


if __name__ == "__main__":
    main()
