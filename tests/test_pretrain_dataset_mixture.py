from pathlib import Path

import numpy as np
import pytest
import torch

from pithtrain.modules.dataset import MemmapDataset, SourceDataset, WeightedMixtureDataset

if not torch.cuda.is_available():
    pytest.skip("pretrain_lm imports CUDA-only modules", allow_module_level=True)

from pithtrain.contexts import distributed
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, get_global_batch, setup_dataset


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
    cfg.dataset_sources = {"a": tmp_path / "a", "b": tmp_path / "b"}
    cfg.dataset_mixture = {"a": 0.25, "b": 0.75}

    dataset = setup_dataset(cfg)

    assert isinstance(dataset, WeightedMixtureDataset)
    assert dataset.source_lengths() == {"a": 64, "b": 64}
    assert dataset.current_weights() == {"a": 0.25, "b": 0.75}


def test_get_global_batch_reads_weighted_mixture(tmp_path: Path):
    cfg = PretrainLMCfg()
    cfg.training.micro_batch_size = 1
    cfg.training.global_batch_size = 4
    cfg.training.sequence_length = 8
    dataset = _build_weighted_dataset(tmp_path)
    distributed.pp_rank = 0
    distributed.dp_rank = 0
    distributed.dp_size = 1
    distributed.cp_rank = 0
    distributed.cp_size = 1
    distributed.ep_rank = 0
    distributed.ep_size = 1

    microbatches = get_global_batch(cfg, dataset, step=0, device=torch.device("cpu"))

    assert len(microbatches) == 4
    for index, batch in enumerate(microbatches):
        (tokens,) = batch.model_inputs
        (labels,) = batch.objective_inputs
        expected_tokens, expected_labels = dataset[index]
        assert tokens.shape == labels.shape == (1, 8)
        assert torch.equal(tokens[0], expected_tokens)
        assert torch.equal(labels[0], expected_labels)
        assert batch.cu_seqlens is None
