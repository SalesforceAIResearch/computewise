#!/usr/bin/env python3
"""Multi-process distributed test: a torchrun --nproc_per_node=N job behaves like
N GPUs on NCCL, with collectives correct across the real processes.

This is gpu-tasks' core requirement: track-3 and MLS-Bench jobs run under
torchrun with NCCL all_reduce / all_gather / broadcast, and every rank must get
the same (correct) value. Run as a worker under torchrun:

    NVRT_SOCKET=... torchrun --nproc_per_node=2 tests/test_distributed.py

It prints one PASS/FAIL line per rank; the runner (run_distributed.sh) checks all
ranks agree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "torch_patch"))

import cuda_hooks  # noqa: F401  (patches torch)
import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    local = rank % max(1, torch.cuda.device_count())
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    results = {}

    results["backend_is_nccl"] = (dist.get_backend() == "nccl")
    results["world_size"] = (dist.get_world_size() == ws)
    results["rank"] = (dist.get_rank() == rank)

    # all_reduce SUM: every rank must see sum(1..ws)
    t = torch.tensor([float(rank + 1)], device=dev)
    dist.all_reduce(t)
    expected_sum = ws * (ws + 1) / 2
    results["all_reduce_sum"] = (abs(t.item() - expected_sum) < 1e-4)

    # broadcast from rank 0: every rank gets rank 0's value (0.0)
    b = torch.tensor([float(rank * 100)], device=dev)
    dist.broadcast(b, 0)
    results["broadcast"] = (abs(b.item() - 0.0) < 1e-4)

    # all_gather: every rank gets the full list [1,2,...,ws]
    src = torch.tensor([float(rank + 1)], device=dev)
    gathered = [torch.zeros(1, device=dev) for _ in range(ws)]
    dist.all_gather(gathered, src)
    got = sorted(int(x.item()) for x in gathered)
    results["all_gather"] = (got == list(range(1, ws + 1)))

    dist.barrier()
    ok = all(results.values())
    print(f"RANK {rank}/{ws} {'PASS' if ok else 'FAIL'} "
          + " ".join(f"{k}={'ok' if v else 'BAD'}" for k, v in results.items())
          + f" all_reduce={t.item()}")
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
