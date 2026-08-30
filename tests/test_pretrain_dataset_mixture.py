from pathlib import Path

import numpy as np
import pytest
import torch

from pithtrain.modules.dataset import MemmapDataset, SourceDataset, WeightedMixtureDataset

if not torch.cuda.is_available():
    pytest.skip("pretrain_lm imports CUDA-only modules", allow_module_level=True)

from pithtrain.contexts import distributed, training
from pithtrain.modules.training import setup_dataset
from pithtrain.tasks.pretrain_lm import (
    PretrainLMCfg,
    get_global_batch,
    raise_if_dataset_insufficient,
)


def _write_tokens(path: Path, start: int, sequence_length: int = 8, samples: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.arange(start, start + sequence_length * samples + 1, dtype=np.int64)
    with open(path, "wb") as f:
        np.save(f, tokens)


def _build_weighted_dataset(tmp_path: Path) -> WeightedMixtureDataset:
    sequence_length = 8
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    _write_tokens(path_a, 0)
    _write_tokens(path_b, 1_000_000)
    source_a = SourceDataset("a", [MemmapDataset(path_a, sequence_length)])
    source_b = SourceDataset("b", [MemmapDataset(path_b, sequence_length)])
    return WeightedMixtureDataset(
        {"a": source_a, "b": source_b}, seed=1, weights={"a": 0.5, "b": 0.5}
    )


def test_setup_dataset_builds_weighted_mixture(tmp_path: Path):
    _write_tokens(tmp_path / "a" / "shard.bin", 0)
    _write_tokens(tmp_path / "b" / "shard.bin", 1_000_000)
    cfg = PretrainLMCfg()
    cfg.training.sequence_length = 8
    cfg.training.dataset_sources = {"a": tmp_path / "a", "b": tmp_path / "b"}
    cfg.training.dataset_mixture = {"a": 0.25, "b": 0.75}

    setup_dataset(cfg.training)

    assert isinstance(training.dataset, WeightedMixtureDataset)
    assert training.dataset.source_lengths() == {"a": 64, "b": 64}
    assert training.dataset.current_weights() == {"a": 0.25, "b": 0.75}


def test_get_global_batch_reads_weighted_mixture(tmp_path: Path):
    cfg = PretrainLMCfg()
    cfg.training.micro_batch_size = 1
    cfg.training.global_batch_size = 4
    cfg.training.sequence_length = 8
    training.dataset = _build_weighted_dataset(tmp_path)
    training.step = 0
    distributed.pp_rank = 0
    distributed.dp_rank = 0
    distributed.dp_size = 1
    distributed.cp_rank = 0
    distributed.cp_size = 1
    distributed.ep_rank = 0
    distributed.ep_size = 1

    tokens, labels = get_global_batch(cfg, torch.device("cpu"))

    assert tokens is not None
    assert labels is not None
    assert tokens.shape == labels.shape == (4, 8)
    assert torch.equal(tokens[:, 1:], labels[:, :-1])


def test_weighted_mixture_is_not_limited_by_combined_source_length(tmp_path: Path):
    cfg = PretrainLMCfg()
    cfg.training.global_batch_size = 1024
    cfg.training.max_steps = 100
    training.dataset = _build_weighted_dataset(tmp_path)

    raise_if_dataset_insufficient(cfg)
