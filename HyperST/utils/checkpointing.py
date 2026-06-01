#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from __future__ import annotations

import copy
from pathlib import Path

import torch
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from . import distributed as dist


class CheckpointManager:
    def __init__(
        self, output_dir: str = "/tmp", keep_recent: int = 100, **checkpointables
    ):

        self.output_dir = Path(output_dir)
        self.keep_recent = keep_recent
        self._recent_iterations = []

        self.checkpointables = copy.copy(checkpointables)

    def step(self, iteration: int):

        out_state_dict = {}
        for key in self.checkpointables:
            if isinstance(self.checkpointables[key], DistributedDataParallel):
                out_state_dict[key] = self.checkpointables[key].module.state_dict()
            else:
                out_state_dict[key] = self.checkpointables[key].state_dict()
        out_state_dict["iteration"] = iteration

        iter_str = f"{iteration:0>8d}"
        torch.save(out_state_dict, self.output_dir / f"checkpoint_{iter_str}.pth")
        with (self.output_dir / "last_checkpoint.txt").open("w") as f:
            f.write(f"checkpoint_{iter_str}.pth")

        self._recent_iterations.append(iter_str)
        if len(self._recent_iterations) > self.keep_recent:
            oldest_iteration = self._recent_iterations.pop(0)
            (self.output_dir / f"checkpoint_{oldest_iteration}.pth").unlink()

    def final_step(self):

        out_state_dict = {}
        for key in self.checkpointables:
            if isinstance(self.checkpointables[key], DistributedDataParallel):
                out_state_dict[key] = self.checkpointables[key].module.state_dict()
            else:
                out_state_dict[key] = self.checkpointables[key].state_dict()
        torch.save(out_state_dict, self.output_dir / f"checkpoint_final.pth")

    def resume(self) -> int:


        last_ckpt_info_file = self.output_dir / "last_checkpoint.txt"
        if last_ckpt_info_file.exists():
            ckpt_path = last_ckpt_info_file.read_text().strip()
            return self.load(self.output_dir / ckpt_path)
        else:
            return 0

    def load(self, path: str | Path) -> int:
        rank = dist.get_rank()

        checkpoint = torch.load(path, map_location="cpu")
        iteration = checkpoint.pop("iteration", -1)
        is_loaded = {key: False for key in self.checkpointables}

        for key in checkpoint:
            if key in self.checkpointables:
                if isinstance(self.checkpointables[key], DistributedDataParallel):
                    self.checkpointables[key].module.load_state_dict(checkpoint[key])
                else:
                    self.checkpointables[key].load_state_dict(checkpoint[key])

                is_loaded[key] = True
            else:
                pass

        not_loaded: list[str] = [key for key in is_loaded if not is_loaded[key]]
        if len(not_loaded) > 0:
            pass
        return iteration
