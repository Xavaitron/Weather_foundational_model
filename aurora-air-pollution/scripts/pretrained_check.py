"""Validate real checkpoint loading and CUDA optimizer steps on SYNTHETIC inputs.

This is an integration/memory check, never a forecast-skill experiment. Updated
weights are discarded. Run one method per process to release all CUDA state.
"""
import argparse
import dataclasses
from datetime import timedelta
import json
from pathlib import Path
import time

import torch

from airadv.model import build_model
from airadv.objective import weighted_loss
from airadv.synthetic import synthetic_batch, synthetic_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="baseline", choices=["baseline", "nowcast_evolution", "adr_latent", "debezenac_gaussian", "wind_pull", "column_push"])
    p.add_argument("--checkpoint", default="data/assets/aurora-0.25-12h-pretrained.ckpt")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--reserve-gib", type=float, default=4.0)
    p.add_argument("--max-memory-gib", type=float, default=28.0)
    a = p.parse_args()
    if a.steps < 2:
        p.error("At least two steps are required to measure populated Adam state")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A working bfloat16-capable CUDA device is required")
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("Use the native allocator so the shared-GPU memory cap is enforced")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    free, total = torch.cuda.mem_get_info()
    limit = min(a.max_memory_gib * 2**30, free - a.reserve_gib * 2**30)
    if limit <= 0:
        raise RuntimeError("Insufficient free GPU memory after safety reserve")
    torch.cuda.set_per_process_memory_fraction(limit / total)
    result = {"data": "SYNTHETIC_ONLY", "weights": "OFFICIAL_12H_PRETRAINED",
              "checkpoint": a.checkpoint, "method": a.method, "grid": [120, 240],
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "memory_limit_GiB": limit / 2**30, "weights_saved": False,
              "forecast_skill_measured": False, "steps": []}
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        stats = synthetic_stats()
        cfg = {"model": {"initialisation": "pretrained12h", "method": a.method,
                         "checkpoint_path": a.checkpoint, "amp": True,
                         "checkpointing": True, "trainable": "all", "width": 32}}
        model = build_model(cfg, stats).cuda().train()
        result["parameters"] = sum(x.numel() for x in model.parameters())
        result["checkpoint_loaded"] = True
        print(json.dumps({"stage": "checkpoint_loaded", "parameters": result["parameters"]}), flush=True)
        batch = synthetic_batch(120, 240).to("cuda")
        target = dataclasses.replace(batch,
            surf_vars={k: v[:, -1:] for k, v in batch.surf_vars.items()},
            atmos_vars={k: v[:, -1:] for k, v in batch.atmos_vars.items()},
            metadata=dataclasses.replace(batch.metadata, time=(batch.metadata.time[0] + timedelta(hours=12),)))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        torch.cuda.reset_peak_memory_stats()
        for i in range(a.steps):
            step_started = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = weighted_loss(pred, target, stats)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite synthetic loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            torch.cuda.synchronize()
            record = {"step": i + 1, "synthetic_loss": float(loss.detach()),
                      "gradient_norm_before_clip": float(grad_norm),
                      "seconds": time.perf_counter() - step_started}
            result["steps"].append(record)
            print(json.dumps(record), flush=True)
            del pred, loss
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        raise
    finally:
        result["seconds"] = time.perf_counter() - started
        result["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
