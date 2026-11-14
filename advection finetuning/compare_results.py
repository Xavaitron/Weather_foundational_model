import argparse
import json
from pathlib import Path

import pandas as pd

from advection_ft.metrics import paired_comparison


def main():
    p = argparse.ArgumentParser(description="Paired storm-bootstrap comparisons to standard fine-tuning.")
    p.add_argument("--reference", required=True, help="Standard run's cyclone_evaluation directory.")
    p.add_argument("--candidate", action="append", required=True, help="Repeat for each candidate evaluation directory.")
    p.add_argument("--output", default="runs/comparison.csv")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    reference_dir = Path(args.reference)
    ref_meta = json.loads((reference_dir/"evaluation.json").read_text())
    reference = pd.read_csv(reference_dir/"tracks.csv", keep_default_na=False)
    all_results = []
    for path in args.candidate:
        directory = Path(path)
        metadata = json.loads((directory/"evaluation.json").read_text())
        for key in ("leads", "issue_delay_hours", "nature", "ibtracs_sha256", "tracker", "max_inits"):
            if metadata[key] != ref_meta[key]:
                raise ValueError(f"Comparison protocol mismatch for {key}: {directory}")
        for key in ("task", "store", "train_start", "train_end", "val_start", "val_end",
                    "test_start", "test_end", "model", "initial_checkpoint", "seed", "trainable",
                    "batch_size", "accumulation_steps", "rollout_steps", "detach_every", "max_steps",
                    "learning_rate", "weight_decay", "loss", "area_weighted_loss", "loss_overrides"):
            if metadata["config"].get(key) != ref_meta["config"].get(key):
                raise ValueError(f"Comparison checkpoint/data mismatch: {key}")
        result = paired_comparison(pd.read_csv(directory/"tracks.csv", keep_default_na=False), reference,
                                  ref_meta["leads"], args.bootstrap, args.seed)
        result.insert(0, "approach", metadata["approach"])
        all_results.append(result)
    result = pd.concat(all_results, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
