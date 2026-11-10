"""Real-processor CPU checks of explicit modality selection and deterministic replay."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("av", reason="Install the omni-data extra")
pytest.importorskip("librosa", reason="Install the omni-data extra")
pytest.importorskip("torchvision", reason="Install the omni-data extra")

from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.qwen3_omni_data import create_omni_dataloader
from pithtrain.modules.training_data import PreparedBundleData
from tests import test_omni_training as media_fixtures

manifest = media_fixtures.manifest
training_bundle = media_fixtures.training_bundle
processor_config = media_fixtures.processor_config


@pytest.mark.parametrize("stage", ["text", "image", "audio", "video"])
def test_explicit_modalities_preserve_named_stage_sampling(
    training_bundle, processor_config, stage
):
    recipe = json.loads((training_bundle / "bundle.json").read_text())["recipe"]
    kinds = recipe["stages"][stage]
    common = dict(
        split="train",
        num_samples=8,
        seed=341,
        epoch=2,
        rank=1,
        world_size=2,
        start_sample=2,
        batch_cfg=dict(recipe["batch"], max_length=1024),
    )
    named = create_omni_dataloader(training_bundle, *processor_config, stage=stage, **common)
    explicit = create_omni_dataloader(
        training_bundle, *processor_config, modalities=tuple(reversed(kinds)), **common
    )
    named_batches, explicit_batches = list(named), list(explicit)
    assert [batch.sample_ids for batch in named_batches] == [
        batch.sample_ids for batch in explicit_batches
    ]
    for left, right in zip(named_batches, explicit_batches, strict=True):
        torch.testing.assert_close(left.labels, right.labels, rtol=0, atol=0)
        torch.testing.assert_close(left.model_inputs, right.model_inputs, rtol=0, atol=0)


def test_explicit_subset_replays_by_step(training_bundle, processor_config):
    tc = SimpleNamespace(global_batch_size=4, micro_batch_size=1, sequence_length=1024, seed=1234)
    cfg = DataCfg()
    cfg.dataset, cfg.format = training_bundle, "prepared_bundle"
    cfg.modalities, cfg.epoch_samples = ("text", "audio"), 8
    data = PreparedBundleData(cfg, tc)
    assert data.stage is None and data.modalities == ["audio", "text"]
    first = media_fixtures.consume(data, 0)
    assert all("pixel_values" not in batch.context_for_stage(0) for batch in first)
    expected = media_fixtures.consume(data, 1)
    cfg.modalities = ("audio", "text")
    restored = PreparedBundleData(cfg, tc)
    media_fixtures.compare_batches(media_fixtures.consume(restored, 1), expected)
