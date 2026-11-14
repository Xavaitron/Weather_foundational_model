"""Two-process CPU/Gloo check of multistep distributed training; no GPU/downloads."""
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import _bootstrap  # Make the project package importable when run as a script.
from advection_ft.model import build_model
from tests.test_integration import config, synthetic_batch


def worker(rank, store):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    settings = config()
    settings.update(rollout_steps=2, activation_checkpointing=True)
    model = DistributedDataParallel(build_model(settings, "nowcastnet", load_weights=False),
                                   find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    batch = synthetic_batch()
    targets = {"surf": {v: torch.cat((x, x[:, -1:], x[:, -1:]), 1) for v, x in batch.surf_vars.items()},
               "atmos": {v: torch.cat((x, x[:, -1:], x[:, -1:]), 1) for v, x in batch.atmos_vars.items()}}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        total, score = model(batch, targets=targets, config=settings)
        total.backward()
        gradient = model.module.base.decoder.operator.net.motion.weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise AssertionError("Missing distributed motion gradient")
        gathered = [torch.zeros_like(gradient) for _ in range(2)]
        dist.all_gather(gathered, gradient)
        torch.testing.assert_close(gathered[0], gathered[1])
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, validation = model(batch, targets=targets, config=settings)
        assert torch.isfinite(validation)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="aurora_ddp_") as directory:
        mp.spawn(worker, args=(str(Path(directory)/"store"),), nprocs=2, join=True)
    print("Two-process CPU DDP forward/backward, gradient synchronization and validation passed.")
