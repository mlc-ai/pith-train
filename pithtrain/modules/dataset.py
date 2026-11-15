"""
Token datasets for distributed training.

MemmapDataset and ConcatDataset read a tokenized corpus: only the pages a sample touches are read.
Global rank 0 writes the sequence offsets and the shuffled indices to offsets.npy and indices.npy
in the corpus directory at startup, and every rank maps them after a barrier. The shuffle runs on
the GPU, so the index array must fit in GPU memory.

SyntheticDataset reads nothing: it draws seeded token ids on the GPU, for benchmarks.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

__all__ = ["MemmapDataset", "ConcatDataset", "SyntheticDataset"]


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


class SyntheticDataset:
    """
    Seeded synthetic token ids drawn on the GPU, so a benchmark needs no corpus and its steps touch
    no file. Ids follow a Zipf law over the vocabulary (id k has weight 1 / (k + 1)), so the loss
    falls from about ln(vocab_size) toward the distribution's entropy as the model learns the
    unigram, and two builds can be compared step by step. It is not a convergence signal.

    The samples of a step depend only on the seed and the step, never on the mesh: every rank draws
    the whole global batch and keeps its own rows and columns.
    """

    def __init__(self, vocab_size: int, sequence_length: int, global_batch_size: int, seed: int):
        self.sequence_length = sequence_length
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.device = torch.device("cuda", torch.cuda.current_device())
        weights = 1.0 / torch.arange(1, vocab_size + 1, device=self.device, dtype=torch.float64)
        cdf = torch.cumsum(weights, 0)
        self.cdf = (cdf / cdf[-1]).to(torch.float32)
        self.cdf[-1] = 1.0

    def get_step(self, step: int) -> torch.Tensor:
        """
        Every sample of the step as int64 ids of shape (global_batch_size, sequence_length + 1);
        a sample's labels are its ids shifted left by one.
        """
        generator = torch.Generator(self.device).manual_seed(self.seed * 1_000_003 + step)
        shape = (self.global_batch_size, self.sequence_length + 1)
        uniform = torch.rand(shape, generator=generator, device=self.device)
        ids = torch.searchsorted(self.cdf, uniform, right=True)
        return ids.clamp_max_(len(self.cdf) - 1)
