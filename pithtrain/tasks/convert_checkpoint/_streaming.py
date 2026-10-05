import json
from collections.abc import Callable
from logging import Logger
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint.planner import WriteItem


class HfCheckpoint:
    def __init__(self, path: Path, stdout: Logger):
        self.path = path
        with open(path / "model.safetensors.index.json") as f:
            self.weight_map = json.load(f)["weight_map"]

        self.tensors: dict[str, torch.Tensor] = {}
        shard_files = sorted(set(self.weight_map.values()))
        for i, shard_file in enumerate(shard_files, 1):
            stdout.info("Indexing shard %d/%d: %s" % (i, len(shard_files), shard_file))
            with safe_open(path / shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    view = f.get_slice(key)
                    shape = view.get_shape()
                    # An empty slice preserves the dtype without reading the weight's data.
                    dtype = (view[:0] if shape else view[...]).dtype
                    self.tensors[key] = torch.empty(shape, dtype=dtype, device="meta")
                    self.weight_map[key] = shard_file

    def load(self, key: str, expert: int | None = None) -> torch.Tensor:
        with safe_open(self.path / self.weight_map[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key) if expert is None else f.get_slice(key)[expert]


class _StreamingSavePlanner(dcp.DefaultSavePlanner):
    def __init__(self, load_tensor: Callable[[str], torch.Tensor]):
        super().__init__()
        self.load_tensor = load_tensor

    def resolve_data(self, write_item: WriteItem) -> torch.Tensor:
        return self.load_tensor(write_item.index.fqn.removeprefix("app.model."))


def save_dcp(
    tensors: dict[str, torch.Tensor],
    load_tensor: Callable[[str], torch.Tensor],
    save_path: Path,
    stdout: Logger,
) -> None:
    stdout.info("Writing DCP checkpoint to %s (%d weights)" % (save_path, len(tensors)))
    # The filesystem writer retains tensors until their output file is complete.
    writer = dcp.FileSystemWriter(save_path, single_file_per_rank=False, per_thread_copy_ahead=0)
    dcp.save(
        {"app": {"model": tensors}},
        storage_writer=writer,
        planner=_StreamingSavePlanner(load_tensor),
        no_dist=True,
    )
    stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(tensors)))
