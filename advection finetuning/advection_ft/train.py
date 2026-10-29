"""Single-GPU/DDP training, validation, resumable checkpoints and JSONL logs."""

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from . import AURORA_COMMIT
from .data import WeatherDataset, read_config
from .model import TransportDecoder, build_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def code_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    files = list(root.glob("*.py")) + list((root/"advection_ft").glob("*.py"))
    for p in sorted(files):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def forecast_loss(pred, item, step, base, config):
    target = dataclasses.replace(pred,
        surf_vars={v: item["surf"][v][:, 2+step:3+step].to(next(base.parameters()).device)
                   for v in pred.surf_vars if v in item["surf"]},
        atmos_vars={v: item["atmos"][v][:, 2+step:3+step].to(next(base.parameters()).device)
                    for v in pred.atmos_vars if v in item["atmos"]})
    prediction = pred.normalise(surf_stats=base.surf_stats)
    target = target.normalise(surf_stats=base.surf_stats)
    surface_weights = {"2t": 3.0, "10u": 0.77, "10v": 0.66, "msl": 1.5}
    atmosphere_weights = {"z": 2.8, "q": 0.78, "t": 1.7, "u": 0.87, "v": 0.6}
    surface, atmosphere = [], []
    for kind, predicted, observed, weights, results in [
        ("surf", prediction.surf_vars, target.surf_vars, surface_weights, surface),
        ("atmos", prediction.atmos_vars, target.atmos_vars, atmosphere_weights, atmosphere)]:
        for name, truth in observed.items():
            error = (predicted[name].float() - truth.float())
            if config.get("loss", "mae") == "mae":
                error = error.abs()
            elif config["loss"] == "charbonnier":
                error = (error.square()+1e-6).sqrt()
            else:
                raise ValueError("loss must be mae or charbonnier")
            if config.get("area_weighted_loss", False):
                area = torch.deg2rad(pred.metadata.lat.float()).cos()
                area = area / area.mean()
                error = error * area[:, None]
            weight = config.get("loss_overrides", {}).get(f"{kind}__{name}", weights.get(name, 1.0))
            results.append(error.mean() * weight)
    return (0.25*sum(surface)+sum(atmosphere))/(len(surface)+len(atmosphere))


def window_loss(model, dataset, item, config, device):
    batch = dataset.make_batch(item, device)
    return model(batch, targets=item, config=config)


def reduce_mean(value, world):
    tensor = value.detach().clone()
    if world > 1:
        dist.all_reduce(tensor)
        tensor /= world
    return float(tensor)


@torch.no_grad()
def validate(model, dataset, loader, config, device, world):
    model.eval()
    scores = []
    for i, item in enumerate(loader):
        if i >= config.get("validation_batches", 32):
            break
        _, score = window_loss(model, dataset, item, config, device)
        scores.append(score)
    if not scores:
        raise ValueError("Validation loader has no batches.")
    mean = reduce_mean(torch.stack(scores).mean(), world)
    model.train()
    return mean


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path, model, optimizer, step, best, config, approach, epoch, batch_offset, rank, world):
    states = [None] * world
    state = rng_state()
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if rank == 0:
        checkpoint = {"model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                      "step": step, "best_validation": best, "config": config, "approach": approach,
                      "epoch": epoch, "batch_offset": batch_offset, "rng": states,
                      "world_size": world, "code_hash": code_hash(), "aurora_commit": AURORA_COMMIT}
        temporary = Path(str(path)+".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)


def main(approach):
    parser = argparse.ArgumentParser(description=f"Aurora {approach} fine-tuning")
    parser.add_argument("--config", default="configs/cyclone.json")
    parser.add_argument("--store")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", choices=["pretrained", "small", "air_pollution", "tiny_random"])
    args = parser.parse_args()
    config = read_config(args.config)
    for key in ("store", "max_steps", "seed", "model"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if approach == "nowcastnet" and config.get("nowcast_aux_weight", 0.5) <= 0:
        raise ValueError("NowcastNet's nearest path needs nowcast_aux_weight>0 to learn motion.")
    rank, local, world = int(os.getenv("RANK", 0)), int(os.getenv("LOCAL_RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif config.get("precision") != "fp32":
        raise ValueError("CPU training requires precision=fp32. Use smoke_test.py for local checks.")
    if device.type == "cpu":
        torch.set_num_threads(config.get("cpu_threads", 2))
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    seed_all(config["seed"])
    directory = Path(args.output or f"runs/{config['task']}/{approach}_seed{config['seed']}")
    if rank == 0:
        if directory.exists() and any(directory.iterdir()) and not args.resume:
            raise FileExistsError(f"{directory} is not empty; use --resume or choose a new --output.")
        directory.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    train = WeatherDataset(config["store"], config["task"], config["train_start"], config["train_end"],
                           config["rollout_steps"])
    val = WeatherDataset(config["store"], config["task"], config["val_start"], config["val_end"],
                         config["rollout_steps"], stride=max(1, config.get("validation_stride", 4)))
    sampler = DistributedSampler(train, seed=config["seed"]) if world > 1 else None
    # Explicit generator allows deterministic epoch order and resume skipping.
    generator = torch.Generator()
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=sampler is None, sampler=sampler,
                        num_workers=config.get("workers", 0), generator=generator, drop_last=True)
    val_sampler = DistributedSampler(val, shuffle=False) if world > 1 else None
    val_loader = DataLoader(val, batch_size=config["batch_size"], sampler=val_sampler,
                            num_workers=config.get("workers", 0))
    if not len(loader):
        raise ValueError("Not enough training windows for the selected batch size.")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint:
        if checkpoint["approach"] != approach or checkpoint["world_size"] != world:
            raise ValueError("Resume requires the same approach and GPU/world count.")
        original = checkpoint["config"]
        if original != config:
            raise ValueError("Resume config must match exactly; use the saved config.json.")
        if checkpoint.get("code_hash") != code_hash():
            raise ValueError("Code differs from checkpoint. Resume with the original source bundle.")
    model = build_model(config, approach, load_weights=checkpoint is None).to(device)
    base_parameters, adapter_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            is_adapter = ("base.decoder." in name and "base.decoder.original." not in name
                          and isinstance(model.base.decoder, TransportDecoder))
            (adapter_parameters if is_adapter else base_parameters).append(parameter)
    groups = []
    if base_parameters:
        groups.append({"params": base_parameters, "lr": config["learning_rate"]})
    if adapter_parameters:
        groups.append({"params": adapter_parameters, "lr": config.get("adapter_learning_rate", 1e-4)})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    start, best, epoch, offset = 0, math.inf, 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["step"], checkpoint["best_validation"]
        epoch, offset = checkpoint["epoch"], checkpoint["batch_offset"]
        state = checkpoint["rng"][rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if device.type == "cuda" and state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    if world > 1:
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None,
                    find_unused_parameters=True)
    if rank == 0:
        write_json(directory/"config.json", config)
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "numpy", "xarray", "microsoft-aurora")}
        write_json(directory/"manifest.json", {"approach": approach, "aurora_commit": AURORA_COMMIT,
                   "code_hash": code_hash(), "versions": versions, "world_size": world,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "source_metadata": train.source_metadata,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   "validation_note": "Subset weather loss selects checkpoints; cyclone metrics are separate."})
        (directory/"pip-freeze.txt").write_bytes(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"]))
    model.train()
    accumulation = config.get("accumulation_steps", 1)
    micro, step, running = 0, start, 0.0
    optimizer.zero_grad(set_to_none=True)
    # Optimizer group order follows the explicit construction above.
    initial_lrs = ([config["learning_rate"]] if base_parameters else []) + (
        [config.get("adapter_learning_rate", 1e-4)] if adapter_parameters else [])
    tick = time.monotonic()
    while step < config["max_steps"]:
        generator.manual_seed(config["seed"] + epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_number, item in enumerate(loader):
            if batch_number < offset:
                continue
            sync = not isinstance(model, DDP) or (micro+1) % accumulation == 0
            ctx = contextlib.nullcontext() if sync else model.no_sync()
            with ctx:
                loss, score = window_loss(model, train, item, config, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at optimizer step {step}.")
                (loss/accumulation).backward()
            running += float(score.detach())
            micro += 1
            if micro % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                          config.get("clip_grad_norm", 1.0), error_if_nonfinite=True)
            warmup = config.get("warmup_steps", 0)
            scale = min(1.0, (step+1)/max(1, warmup))
            if step >= warmup:
                scale *= 0.1 + 0.9*0.5*(1+math.cos(math.pi*(step-warmup)/max(1, config["max_steps"]-warmup)))
            for g, lr in zip(optimizer.param_groups, initial_lrs):
                g["lr"] = lr*scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = {"step": step, "train_weather_loss": reduce_mean(
                torch.tensor(running/accumulation, device=device), world),
                "seconds_per_optimizer_step": time.monotonic()-tick,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
            running = 0.0
            do_validate = step % config.get("validate_every", 250) == 0 or step == config["max_steps"]
            improved = False
            if do_validate:
                record["validation_weather_loss"] = validate(model, val, val_loader, config, device, world)
                improved = record["validation_weather_loss"] < best
                best = min(best, record["validation_weather_loss"])
            if rank == 0:
                with (directory/"train.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record)+"\n")
                print(json.dumps(record), flush=True)
            if step % config.get("save_every", 250) == 0 or do_validate:
                save_checkpoint(directory/"latest.pt", model, optimizer, step, best, config, approach,
                                epoch, batch_number+1, rank, world)
                if improved:
                    # Atomic copy avoids serializing another live model state.
                    if rank == 0:
                        import shutil
                        shutil.copyfile(directory/"latest.pt", directory/"best.pt")
            tick = time.monotonic()
            if step >= config["max_steps"]:
                break
        epoch, offset = epoch+1, 0
    if rank == 0:
        write_json(directory/"training_complete.json", {"step": step, "best_validation_weather_loss": best})
    if world > 1:
        dist.destroy_process_group()
