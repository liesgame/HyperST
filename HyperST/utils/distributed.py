#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from __future__ import annotations

from typing import Callable

import torch
from loguru import logger
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.distributed.nn import all_gather as nn_all_gather


def launch(
    job_fn: Callable,
    num_machines: int = 1,
    num_gpus_per_machine: int = 1,
    machine_rank: int = 0,
    dist_url: str = "tcp://127.0.0.1:23457",
    args=(),
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found! Cannot launch distributed processes.")

    world_size = num_machines * num_gpus_per_machine

    if world_size > 1:
        mp.spawn(
            _job_worker,
            nprocs=num_gpus_per_machine,
            args=(
                job_fn, world_size, num_gpus_per_machine, machine_rank, dist_url, args
            ),
            daemon=False,
        )
    else:

        _job_worker(0, job_fn, 1, 1, 0, dist_url, args)


def _job_worker(
    local_rank: int,
    job_fn: Callable,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: tuple,
):

    # Adjust global rank of process based on its machine rank.
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as e:
        pass
        raise e

    synchronize()
    torch.cuda.set_device(local_rank)
    job_fn(*args)


def synchronize() -> None:
    if dist.is_initialized():
        dist.barrier()


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def gather_across_processes(t: torch.Tensor) -> list[torch.Tensor]:

    world_size = get_world_size()
    if world_size == 1:
        return [t]

    output = list(nn_all_gather(t))
    return output


def gpu_mem_usage() -> int:

    if torch.cuda.is_available():
        # This will be in bytes, so we divide by (1024 * 1024).
        return torch.cuda.max_memory_allocated() // 1048576
    else:
        return 0
