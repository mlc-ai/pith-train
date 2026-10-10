import io
import itertools
import json
import math
import os
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from logging import Logger
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan, WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

DEFAULT_CHUNK_SIZE = 64 * 1024**2
DEFAULT_SHARD_SIZE = 8 * 1024**3


def tensor_chunks(shape: tuple[int, ...], max_numel: int):
    if not shape or math.prod(shape) == 0:
        yield (0,) * len(shape), shape
        return
    steps = list(shape)
    for axis in range(len(shape)):
        if math.prod(steps) <= max_numel:
            break
        steps[axis] = max(1, max_numel // math.prod(steps[axis + 1 :]))
    for offsets in itertools.product(*(range(0, size, step) for size, step in zip(shape, steps))):
        sizes = tuple(min(step, size - offset) for size, step, offset in zip(shape, steps, offsets))
        yield offsets, sizes


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

    def load(
        self, key: str, expert: int | None = None, slices: tuple[slice, ...] = ()
    ) -> torch.Tensor:
        with safe_open(self.path / self.weight_map[key], framework="pt", device="cpu") as f:
            selection = (() if expert is None else (expert,)) + slices
            return f.get_slice(key)[selection] if selection else f.get_tensor(key)


class _StreamingSavePlanner(dcp.DefaultSavePlanner):
    def __init__(
        self, load_tensor: Callable[[str, tuple[slice, ...]], torch.Tensor], chunk_size: int
    ):
        super().__init__()
        self.load_tensor = load_tensor
        self.chunk_size = chunk_size

    def create_local_plan(self) -> SavePlan:
        plan = super().create_local_plan()
        items = []
        for item in plan.items:
            tensor = self.state_dict[item.index.fqn]
            max_numel = self.chunk_size // tensor.element_size()
            if max_numel == 0:
                raise ValueError("max_chunk_size must fit one element of %s" % tensor.dtype)
            for offsets, sizes in tensor_chunks(tuple(tensor.shape), max_numel):
                chunk = ChunkStorageMetadata(torch.Size(offsets), torch.Size(sizes))
                items.append(
                    replace(
                        item,
                        index=MetadataIndex(item.index.fqn, torch.Size(offsets)),
                        type=WriteItemType.SHARD,
                        tensor_data=replace(item.tensor_data, chunk=chunk),
                    )
                )
        return replace(plan, items=items)

    def resolve_data(self, write_item: WriteItem) -> torch.Tensor:
        chunk = write_item.tensor_data.chunk
        slices = tuple(
            slice(offset, offset + size) for offset, size in zip(chunk.offsets, chunk.sizes)
        )
        return self.load_tensor(write_item.index.fqn.removeprefix("app.model."), slices)


class _StreamingFileSystemWriter(dcp.FileSystemWriter):
    def __init__(self, path: Path, shard_size: int):
        super().__init__(path, per_thread_copy_ahead=0)
        self.shard_size = shard_size

    def write_data(self, plan: SavePlan, planner) -> Future[list[WriteResult]]:
        results, stream, file_count = [], None, 0

        def sync_file():
            stream.flush()
            if self.sync_files:
                os.fsync(stream.fileno())

        with ExitStack() as stack:
            for item in plan.items:
                tensor = planner.resolve_data(item).detach().cpu()
                # A slice may retain the whole source storage through torch.save.
                if tensor.untyped_storage().nbytes() != tensor.nbytes:
                    tensor = tensor.clone()
                with io.BytesIO() as buffer:
                    torch.save(tensor, buffer)
                    del tensor
                    with buffer.getbuffer() as data:
                        length = data.nbytes
                        if length > self.shard_size:
                            raise ValueError("max_shard_size must fit a serialized chunk")
                        if stream is None or stream.tell() + length > self.shard_size:
                            if stream is not None:
                                sync_file()
                                stream.close()
                            file_name = "%s%d.distcp" % (plan.storage_data.prefix, file_count)
                            file_count += 1
                            path = self.fs.concat_path(self.path, file_name)
                            stream = stack.enter_context(self.fs.create_stream(path, "wb"))
                        offset = stream.tell()
                        stream.write(data)
                        results.append(
                            WriteResult(item.index, length, _StorageInfo(file_name, offset, length))
                        )
            if stream is not None:
                sync_file()
        future = Future()
        future.set_result(results)
        return future


def save_dcp(
    tensors: dict[str, torch.Tensor],
    load_tensor: Callable[[str, tuple[slice, ...]], torch.Tensor],
    save_path: Path,
    stdout: Logger,
    *,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_shard_size: int = DEFAULT_SHARD_SIZE,
) -> None:
    if max_chunk_size <= 0 or max_shard_size < 4096:
        raise ValueError("max_chunk_size must be positive and max_shard_size must be at least 4096")
    stdout.info("Writing DCP checkpoint to %s (%d weights)" % (save_path, len(tensors)))
    writer = _StreamingFileSystemWriter(save_path, max_shard_size)
    dcp.save(
        {"app": {"model": tensors}},
        storage_writer=writer,
        planner=_StreamingSavePlanner(load_tensor, min(max_chunk_size, max_shard_size // 2)),
        no_dist=True,
    )
    stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(tensors)))
