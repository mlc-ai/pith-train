"""
Dataset utilities for distributed training.

All data is memory-mapped so only accessed pages are read into memory. Precomputed metadata
(sequence offsets, shuffle indices) is written to disk by local rank 0 and memory-mapped by
all other ranks after a barrier. Global shuffling is done on GPU for speed.

TODO: if the shuffled index array exceeds GPU memory, implement block-wise shuffling.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


class MemmapDataset:
    """Fixed-length next-token samples from the token stream in a packed .bin file.

    tokenize_corpus writes tokens followed by document offsets as two .npy arrays; this
    reader uses only the tokens. Samples can cross document boundaries, including EOS.
    Inputs and labels have equal length, with labels shifted one token ahead here, before
    any context-parallel slicing. The loss must not shift them again. A trailing fragment
    without enough tokens for a full input plus its last target is dropped, not padded.
    """

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
        """Read only [seq_offset, seq_offset + seq_length) of a sequence."""
        start = idx * self.sequence_length + seq_offset
        tokens = torch.tensor(self.tokens[start : start + seq_length])
        labels = torch.tensor(self.tokens[start + 1 : start + seq_length + 1])
        return tokens, labels


class ConcatDataset:
    """Concatenates multiple MemmapDatasets with global shuffling."""

    OFFSETS = "offsets.npy"
    INDICES = "indices.npy"

    def __init__(self, memmap_datasets: List[MemmapDataset], seed: int):
        self.memmap_datasets = memmap_datasets
        root = os.path.commonpath([str(d.root) for d in memmap_datasets])
        offsets_path = Path(root, ConcatDataset.OFFSETS)
        indices_path = Path(root, ConcatDataset.INDICES)
        # The first rank on each node computes offsets and shuffled indices.
        # All other ranks wait at the barrier until the results are ready for mmap.
        if int(os.environ["LOCAL_RANK"]) == 0:
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

    def _resolve(self, idx: int) -> Tuple[MemmapDataset, int]:
        """Map a global shuffled index to (dataset, local_index)."""
        p = self.indices[idx]
        x = np.searchsorted(self.offsets, p, side="right")
        y = p if x == 0 else p - self.offsets[x - 1]
        return self.memmap_datasets[x], y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ds, local_idx = self._resolve(idx)
        return ds[local_idx]

    def get_chunk(
        self, idx: int, seq_offset: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a sub-range of a sequence by index, delegating to the underlying dataset."""
        ds, local_idx = self._resolve(idx)
        return ds.get_chunk(local_idx, seq_offset, seq_length)
