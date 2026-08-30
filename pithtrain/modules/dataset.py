"""
Dataset utilities for distributed training.

All data is memory-mapped so only accessed pages are read into memory. Precomputed metadata
(sequence offsets, shuffle indices) is written to disk by local rank 0 and memory-mapped by
all other ranks after a barrier. Global shuffling is done on GPU for speed.

TODO: if the shuffled index array exceeds GPU memory, implement block-wise shuffling.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


class MemmapDataset:
    """Memory-mapped dataset backed by a packed .bin file of token IDs."""

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


class SourceDataset:
    """Named corpus source composed of one or more memory-mapped shards."""

    def __init__(self, name: str, memmap_datasets: List[MemmapDataset]):
        if not memmap_datasets:
            raise ValueError(f"Source '{name}' has no dataset shards.")
        self.name = name
        self.memmap_datasets = memmap_datasets
        lengths = np.array([len(ds) for ds in memmap_datasets], dtype=np.int64)
        self.offsets = np.cumsum(lengths)
        self.total_len = int(self.offsets[-1])
        if self.total_len <= 0:
            raise ValueError(f"Source '{name}' has zero samples.")

    def __len__(self) -> int:
        return self.total_len

    def _resolve(self, idx: int) -> Tuple[MemmapDataset, int]:
        idx = int(idx)
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Source '{self.name}' index out of range: {idx}")
        x = int(np.searchsorted(self.offsets, idx, side="right"))
        y = idx if x == 0 else idx - int(self.offsets[x - 1])
        return self.memmap_datasets[x], y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ds, local_idx = self._resolve(idx)
        return ds[local_idx]

    def get_chunk(
        self, idx: int, seq_offset: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ds, local_idx = self._resolve(idx)
        return ds.get_chunk(local_idx, seq_offset, seq_length)


class WeightedMixtureDataset:
    """Weighted multi-source sampling with replacement."""

    _U64_MASK = (1 << 64) - 1

    def __init__(
        self,
        source_datasets: Dict[str, SourceDataset],
        seed: int,
        weights: Dict[str, float],
    ):
        if not source_datasets:
            raise ValueError("dataset_sources cannot be empty for weighted mixture mode.")
        self.source_datasets = dict(source_datasets)
        self.source_names = sorted(self.source_datasets.keys())
        self.seed = int(seed) & WeightedMixtureDataset._U64_MASK
        self._weights: Dict[str, float] = {}
        self._cdf = np.empty(0, dtype=np.float64)
        self.update_weights(weights)

    @staticmethod
    def _splitmix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & WeightedMixtureDataset._U64_MASK
        z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & WeightedMixtureDataset._U64_MASK
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & WeightedMixtureDataset._U64_MASK
        return (z ^ (z >> 31)) & WeightedMixtureDataset._U64_MASK

    def _rand_u64(self, idx: int, salt: int) -> int:
        x = (self.seed ^ int(idx) ^ int(salt)) & WeightedMixtureDataset._U64_MASK
        return WeightedMixtureDataset._splitmix64(x)

    def _validate_weights(self, weights: Dict[str, float]) -> np.ndarray:
        if set(weights.keys()) != set(self.source_names):
            expected = ", ".join(self.source_names)
            actual = ", ".join(sorted(weights.keys()))
            raise ValueError(
                f"Mixture keys must match dataset_sources. expected=[{expected}] actual=[{actual}]"
            )
        vec = np.array([float(weights[k]) for k in self.source_names], dtype=np.float64)
        if not np.all(np.isfinite(vec)):
            raise ValueError("Mixture weights must be finite numbers.")
        if np.any(vec < 0):
            raise ValueError("Mixture weights must be non-negative.")
        total = float(np.sum(vec))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Mixture weights must sum to 1.0 (got {total:.8f}).")
        if np.all(vec == 0.0):
            raise ValueError("At least one mixture weight must be > 0.")
        return vec

    def update_weights(self, weights: Dict[str, float]) -> None:
        vec = self._validate_weights(weights)
        self._cdf = np.cumsum(vec)
        self._cdf[-1] = 1.0
        self._weights = {k: float(weights[k]) for k in self.source_names}

    def current_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def source_lengths(self) -> Dict[str, int]:
        return {name: len(ds) for name, ds in self.source_datasets.items()}

    def __len__(self) -> int:
        return int(sum(len(ds) for ds in self.source_datasets.values()))

    def _sample_source_name(self, idx: int) -> str:
        u = self._rand_u64(idx, salt=0xD6E8FEB86659FD93)
        p = u / float(1 << 64)
        src_idx = int(np.searchsorted(self._cdf, p, side="right"))
        src_idx = min(src_idx, len(self.source_names) - 1)
        return self.source_names[src_idx]

    def _sample_local_index(self, idx: int, source_size: int) -> int:
        u = self._rand_u64(idx, salt=0xA0761D6478BD642F)
        return int(u % source_size)

    def _resolve(self, idx: int) -> Tuple[SourceDataset, int]:
        idx = int(idx)
        if idx < 0:
            raise IndexError(f"Mixture index must be non-negative, got {idx}")
        source_name = self._sample_source_name(idx)
        source_dataset = self.source_datasets[source_name]
        local_idx = self._sample_local_index(idx, len(source_dataset))
        return source_dataset, local_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ds, local_idx = self._resolve(idx)
        return ds[local_idx]

    def get_chunk(
        self, idx: int, seq_offset: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ds, local_idx = self._resolve(idx)
        return ds.get_chunk(local_idx, seq_offset, seq_length)
