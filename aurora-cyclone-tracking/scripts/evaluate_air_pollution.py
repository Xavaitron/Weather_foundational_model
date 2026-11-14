import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap  # Make the project package importable when run as a script.
from advection_ft.data import WeatherDataset
from advection_ft.model import advance, build_model
from advection_ft.train import seed_all, write_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Physical-unit, area-weighted CAMS RMSE/MAE/bias by variable/level/lead.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--store")
    p.add_argument("--output")
    p.add_argument("--start", help="Optional explicit evaluation date override, recorded in metadata.")
    p.add_argument("--end")
    p.add_argument("--horizon-hours", type=int, default=120)
    p.add_argument("--stride", type=int, default=2, help="2 means daily initialization for 12-hour data.")
    p.add_argument("--max-inits", type=int)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, approach = checkpoint["config"].copy(), checkpoint["approach"]
    if config["task"] != "air_pollution":
        raise ValueError("Requires an air-pollution checkpoint.")
    if args.horizon_hours <= 0 or args.horizon_hours % 12:
        raise ValueError("Air-pollution horizon must be a positive multiple of 12 hours.")
    if args.store:
        config["store"] = args.store
    if args.start:
        config["test_start"] = args.start
    if args.end:
        config["test_end"] = args.end
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config["precision"] = "fp32"
        torch.set_num_threads(2)
    config["activation_checkpointing"] = False
    seed_all(config["seed"])
    model = build_model(config, approach, load_weights=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    steps = args.horizon_hours//12
    data = WeatherDataset(config["store"], "air_pollution", config["test_start"], config["test_end"],
                          rollout_steps=steps, stride=args.stride)
    output = Path(args.output or Path(args.checkpoint).parent/"air_pollution_evaluation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    area = torch.deg2rad(data.lat).cos().to(device)
    area = area[:, None]/(area.sum()*len(data.lon))
    records = []
    for i in range(min(len(data), args.max_inits or len(data))):
        item = data[i]
        item = {"surf": {v: t[None] for v, t in item["surf"].items()},
                "atmos": {v: t[None] for v, t in item["atmos"].items()},
                "time_ns": torch.tensor([item["time_ns"]])}
        batch = data.make_batch(item, device)
        for s in range(steps):
            pred, _, _ = model(batch)
            for kind, fields in (("surf", pred.surf_vars), ("atmos", pred.atmos_vars)):
                for variable, tensor in fields.items():
                    truth = item[kind][variable][:, 2+s:3+s].to(device)
                    error = (tensor-truth).float()[0, 0]
                    if not torch.isfinite(error).all():
                        raise FloatingPointError(f"Nonfinite forecast for {variable}; evaluation stopped.")
                    if kind == "surf":
                        error = error[None]
                    for level, field in zip([None] if kind == "surf" else pred.metadata.atmos_levels, error):
                        records.append({"init_time_ns": int(item["time_ns"][0]), "lead_hours": (s+1)*12,
                            "kind": kind, "variable": variable, "level_hpa": level,
                            "mse": float((field.square()*area).sum()),
                            "mae": float((field.abs()*area).sum()), "bias": float((field*area).sum())})
            batch = advance(batch, pred)
        print(f"Evaluated {i+1} initializations", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output/"per_initialization.csv", index=False)
    keys = ["kind", "variable", "level_hpa", "lead_hours"]
    summary = frame.groupby(keys, dropna=False).agg(mse=("mse", "mean"), mae=("mae", "mean"),
                         bias=("bias", "mean"), n_initializations=("mse", "size")).reset_index()
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    summary.to_csv(output/"summary.csv", index=False)
    write_json(output/"evaluation.json", {"config": config, "approach": approach,
        "horizon_hours": args.horizon_hours, "stride": args.stride, "max_inits": args.max_inits,
        "metrics": "physical-unit area-weighted RMSE, MAE and bias; no pooled cross-unit average"})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
