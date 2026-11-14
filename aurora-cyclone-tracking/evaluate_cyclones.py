import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from advection_ft.data import WeatherDataset
from advection_ft.metrics import haversine_km, summarise_tracks
from advection_ft.model import advance, build_model
from advection_ft.tracking import CycloneTracker
from advection_ft.train import seed_all, write_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Deterministic cyclone track verification against IBTrACS.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ibtracs", required=True)
    p.add_argument("--store")
    p.add_argument("--output")
    p.add_argument("--leads", type=int, nargs="+", default=[24, 48, 72, 96, 120])
    p.add_argument("--issue-delay-hours", type=int, choices=[0, 6], default=0)
    p.add_argument("--nature", nargs="+", default=["TS", "DS"])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--max-inits", type=int, help="Debug only; identical limit required for every approach.")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, approach = checkpoint["config"].copy(), checkpoint["approach"]
    if config["task"] != "cyclone":
        raise ValueError("This evaluator requires a cyclone run.")
    if args.store:
        config["store"] = args.store
    if any(lead <= 0 or lead % 6 for lead in args.leads):
        raise ValueError("Lead times must be positive multiples of 6 hours.")
    data = WeatherDataset(config["store"], "cyclone", config["test_start"], config["test_end"], rollout_steps=1)
    truth = pd.read_csv(args.ibtracs, keep_default_na=False)
    truth["time"] = pd.to_datetime(truth.time, utc=True)
    truth = truth.sort_values(["sid", "time"])
    start, end = pd.Timestamp(config["test_start"], tz="UTC"), pd.Timestamp(config["test_end"], tz="UTC")+pd.Timedelta(days=1)
    if truth.duplicated(["sid", "time"]).any():
        raise ValueError("Duplicate IBTrACS storm/time pairs.")
    observations = {(row.sid, row.time): row for row in truth.itertuples()}
    available = {pd.Timestamp(data.times[i]).tz_localize("UTC"): j for j, i in enumerate(data.indices)}
    eligible = truth[(truth.time >= start) & (truth.time < end) & truth.nature.isin(args.nature)]
    eligible = eligible[eligible.time.isin(available)]
    final_time = truth.groupby("sid").time.max()
    eligible = eligible.loc[np.asarray([
        final_time[sid] >= t + pd.Timedelta(hours=24)
        for sid, t in zip(eligible.sid, eligible.time)
    ], dtype=bool)]
    initializations = sorted(eligible.time.unique())
    if args.max_inits:
        initializations = initializations[:args.max_inits]
    if not initializations:
        raise ValueError("No IBTrACS initializations overlap the prepared test data.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config["precision"] = "fp32"
        torch.set_num_threads(2)
    config["activation_checkpointing"] = False
    seed_all(config["seed"])
    model = build_model(config, approach, load_weights=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    output = Path(args.output or Path(args.checkpoint).parent/"cyclone_evaluation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation {output}")
    output.mkdir(parents=True, exist_ok=True)
    horizon = max(args.leads)+args.issue_delay_hours
    records = []
    for count, timestamp in enumerate(initializations):
        item = data[available[timestamp]]
        item = {"surf": {v: t[None] for v, t in item["surf"].items()},
                "atmos": {v: t[None] for v, t in item["atmos"].items()},
                "time_ns": torch.tensor([item["time_ns"]])}
        batch = data.make_batch(item, device)
        active = eligible[eligible.time == timestamp]
        trackers = {row.sid: CycloneTracker(data.lat.numpy(), data.lon.numpy(), row.lat, row.lon)
                    for row in active.itertuples()}
        basins = dict(zip(active.sid, active.basin))
        for step in range(1, horizon//6+1):
            pred, _, _ = model(batch)
            msl = pred.surf_vars["msl"][0, 0].float().cpu().numpy()
            z700 = pred.atmos_vars["z"][0, 0, pred.metadata.atmos_levels.index(700)].float().cpu().numpy()
            if not np.isfinite(msl).all() or not np.isfinite(z700).all():
                raise FloatingPointError("Nonfinite forecast. Evaluation aborted rather than dropping cases.")
            raw_lead, lead = step*6, step*6-args.issue_delay_hours
            valid = timestamp+pd.Timedelta(hours=raw_lead)
            for sid, tracker in trackers.items():
                position, failed = tracker.step(msl, z700)
                observed = observations.get((sid, valid))
                if lead <= 0 or valid >= end or observed is None or observed.nature not in args.nature:
                    continue
                records.append({"sid": sid, "basin": basins[sid], "init_time": timestamp.isoformat(),
                    "issue_time": (timestamp+pd.Timedelta(hours=args.issue_delay_hours)).isoformat(),
                    "valid_time": valid.isoformat(), "lead_hours": lead, "raw_lead_hours": raw_lead,
                    "pred_lat": position[0], "pred_lon": position[1], "true_lat": observed.lat,
                    "true_lon": observed.lon, "tracker_failed": failed,
                    "error_km": float(haversine_km(*position, observed.lat, observed.lon))})
            batch = advance(batch, pred)
        print(f"{count+1}/{len(initializations)} initialization times", flush=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No verifying track positions; check dates, nature and lead times.")
    frame.to_csv(output/"tracks.csv", index=False)
    summary = summarise_tracks(frame, args.leads, args.bootstrap, config["seed"])
    if summary.empty:
        raise ValueError("No scored cases at the requested leads; check storm coverage/nature and evaluation dates.")
    summary.to_csv(output/"summary.csv", index=False)
    write_json(output/"evaluation.json", {"approach": approach, "leads": args.leads,
        "issue_delay_hours": args.issue_delay_hours, "nature": args.nature,
        "bootstrap_repetitions": args.bootstrap, "max_inits": args.max_inits,
        "ibtracs_sha256": hashlib.sha256(Path(args.ibtracs).read_bytes()).hexdigest(),
        "config": config, "tracker": "coarse_MSL_Z700_local_minimum_quadratic_refinement_persist_failure",
        "note": "Deterministic, initialized-track verification; no genesis/detection or ensemble calibration claim."})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
