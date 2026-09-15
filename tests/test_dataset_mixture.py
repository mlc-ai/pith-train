from pathlib import Path

import numpy as np
import pytest
import torch

from pithtrain.modules.dataset import (
    MemmapDataset,
    SourceDataset,
    WeightedMixtureDataset,
)


def _write_tokens(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, tokens)


def _build_source(
    root: Path, name: str, start: int, sequence_length: int, samples: int
) -> SourceDataset:
    tokens = np.arange(start, start + samples * sequence_length + 1, dtype=np.int64)
    path = root / name / "shard0.bin"
    _write_tokens(path, tokens)
    return SourceDataset(name, [MemmapDataset(path, sequence_length)])


def _estimate_source_share(dataset: WeightedMixtureDataset, begin: int, end: int) -> float:
    count_a = 0
    for idx in range(begin, end):
        tokens, _ = dataset[idx]
        assert isinstance(tokens, torch.Tensor)
        if int(tokens[0]) < 1_000_000:
            count_a += 1
    return count_a / max(1, end - begin)


def test_source_dataset_resolves_multiple_shards(tmp_path: Path):
    sequence_length = 4
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    _write_tokens(path_a, np.arange(0, 9, dtype=np.int64))
    _write_tokens(path_b, np.arange(100, 109, dtype=np.int64))
    source = SourceDataset(
        "source",
        [MemmapDataset(path_a, sequence_length), MemmapDataset(path_b, sequence_length)],
    )

    assert [int(source[idx][0][0]) for idx in range(len(source))] == [0, 4, 100, 104]
    tokens, labels = source.get_chunk(2, seq_offset=1, seq_length=2)
    assert tokens.tolist() == [101, 102]
    assert labels.tolist() == [102, 103]


def test_weighted_mixture_basic_and_validation(tmp_path: Path):
    source_a = _build_source(tmp_path, "a", 0, sequence_length=8, samples=128)
    source_b = _build_source(tmp_path, "b", 1_000_000, sequence_length=8, samples=128)
    dataset = WeightedMixtureDataset(
        {"a": source_a, "b": source_b}, seed=1234, weights={"a": 0.7, "b": 0.3}
    )

    assert dataset.current_weights() == {"a": 0.7, "b": 0.3}
    assert dataset.source_lengths() == {"a": 128, "b": 128}
    assert len(dataset) == 256

    with pytest.raises(ValueError):
        dataset.update_weights({"a": 0.5, "c": 0.5})
    with pytest.raises(ValueError):
        dataset.update_weights({"a": 0.5, "b": 0.49})
    with pytest.raises(ValueError):
        dataset.update_weights({"a": -0.1, "b": 1.1})


def test_weighted_mixture_is_deterministic_for_fixed_seed(tmp_path: Path):
    source_a = _build_source(tmp_path, "a", 0, sequence_length=8, samples=64)
    source_b = _build_source(tmp_path, "b", 1_000_000, sequence_length=8, samples=64)
    weights = {"a": 0.6, "b": 0.4}

    dataset1 = WeightedMixtureDataset({"a": source_a, "b": source_b}, seed=7, weights=weights)
    dataset2 = WeightedMixtureDataset({"a": source_a, "b": source_b}, seed=7, weights=weights)

    for idx in [0, 1, 7, 31, 100, 777]:
        tokens1, labels1 = dataset1[idx]
        tokens2, labels2 = dataset2[idx]
        assert torch.equal(tokens1, tokens2)
        assert torch.equal(labels1, labels2)


def test_weighted_mixture_respects_weights_and_updates(tmp_path: Path):
    source_a = _build_source(tmp_path, "a", 0, sequence_length=8, samples=256)
    source_b = _build_source(tmp_path, "b", 1_000_000, sequence_length=8, samples=256)
    dataset = WeightedMixtureDataset(
        {"a": source_a, "b": source_b}, seed=2026, weights={"a": 0.8, "b": 0.2}
    )

    assert abs(_estimate_source_share(dataset, begin=0, end=20000) - 0.8) < 0.03

    dataset.update_weights({"a": 0.2, "b": 0.8})
    assert abs(_estimate_source_share(dataset, begin=20000, end=40000) - 0.2) < 0.03


def test_weighted_mixture_get_chunk_matches_item(tmp_path: Path):
    source_a = _build_source(tmp_path, "a", 0, sequence_length=8, samples=64)
    source_b = _build_source(tmp_path, "b", 1_000_000, sequence_length=8, samples=64)
    dataset = WeightedMixtureDataset(
        {"a": source_a, "b": source_b}, seed=99, weights={"a": 0.5, "b": 0.5}
    )

    tokens, labels = dataset[13]
    chunk_tokens, chunk_labels = dataset.get_chunk(13, seq_offset=2, seq_length=3)

    assert torch.equal(chunk_tokens, tokens[2:5])
    assert torch.equal(chunk_labels, labels[2:5])
