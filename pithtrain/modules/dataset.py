"""
Memory-mapped token datasets for distributed training.

Only the pages a sample touches are read. Global rank 0 writes the sequence offsets and the
shuffled indices to offsets.npy and indices.npy in the corpus directory at startup, and every rank
maps them after a barrier. The shuffle runs on the GPU, so the index array must fit in GPU memory.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

__all__ = ["MemmapDataset", "ConcatDataset"]


class MemmapDataset:
    def __init__(self, path: Path, sequence_length: int):
        self.root = path.parent
        self.sequence_length = sequence_length
        self.tokens = np.load(path, mmap_mode="r")

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.sequence_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.sequence_length
        end = start + self.sequence_length
        tokens = torch.tensor(self.tokens[start:end])
        labels = torch.tensor(self.tokens[start + 1 : end + 1])
        return tokens, labels

    def get_chunk(
        self, idx: int, seq_offset: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.sequence_length + seq_offset
        tokens = torch.tensor(self.tokens[start : start + seq_length])
        labels = torch.tensor(self.tokens[start + 1 : start + seq_length + 1])
        return tokens, labels


class ConcatDataset:
    def __init__(self, memmap_datasets: List[MemmapDataset], seed: int):
        self.memmap_datasets = memmap_datasets
        root = os.path.commonpath([str(d.root) for d in memmap_datasets])
        offsets_path = Path(root, "offsets.npy")
        indices_path = Path(root, "indices.npy")
        if torch.distributed.get_rank() == 0:
            offsets = np.cumsum([len(ds) for ds in memmap_datasets])
            np.save(offsets_path, offsets)
            kwargs = dict()
            kwargs["device"] = torch.cuda.current_device()
            generator = torch.Generator(kwargs["device"])
            generator = generator.manual_seed(seed)
            kwargs["generator"] = generator
            indices = torch.randperm(offsets[-1], **kwargs)
            np.save(indices_path, indices.cpu().numpy())
        torch.distributed.barrier()
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self.indices = np.load(indices_path, mmap_mode="r")

    def __len__(self):
        return self.offsets[-1]

    def resolve_index(self, idx: int) -> Tuple[MemmapDataset, int]:
        p = self.indices[idx]
        x = np.searchsorted(self.offsets, p, side="right")
        y = p if x == 0 else p - self.offsets[x - 1]
        return self.memmap_datasets[x], y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ds, local_idx = self.resolve_index(idx)
        return ds[local_idx]

    def get_chunk(
        self, idx: int, seq_offset: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ds, local_idx = self.resolve_index(idx)
        return ds.get_chunk(local_idx, seq_offset, seq_length)
