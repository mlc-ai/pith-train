"""Data-boundary tests for our Omni reader/collator, without model weights or GPUs."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("av", reason="Install the omni-data extra")
pytest.importorskip("librosa", reason="Install the omni-data extra")
pytest.importorskip("torchvision", reason="Install the omni-data extra")

import av
import soundfile as sf
from PIL import Image
from transformers import AutoConfig, Qwen3OmniMoeProcessor

from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset


@pytest.fixture(scope="module")
def processor_config():
    kwargs = dict(revision="26291f793822fb6be9555850f06dfe95f2d7e695", local_files_only=True)
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_id, **kwargs)
        config = AutoConfig.from_pretrained(model_id, **kwargs).thinker_config
    except OSError:
        pytest.skip("Run the Omni data preparation recipe to cache the small processor files")
    return processor, config


@pytest.fixture
def manifest(tmp_path):
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(tmp_path / "image.png")
    waveform = np.sin(2 * np.pi * 440 * np.arange(9600) / 8000).astype(np.float32) * 0.1
    sf.write(tmp_path / "audio.wav", waveform, 8000)
    with av.open(str(tmp_path / "video.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=2)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for color in [(0, 255, 0), (0, 0, 255)]:
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), color=color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    rows = [
        dict(id="text", text="A text example.", media=[]),
        dict(id="image", text="A red square.", media=[dict(type="image", path="image.png")]),
        dict(id="audio", text="A short tone.", media=[dict(type="audio", path="audio.wav")]),
        dict(
            id="video", text="Green changes to blue.", media=[dict(type="video", path="video.mp4")]
        ),
    ]
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_mixed_media_keeps_samples_targets_and_feature_counts(manifest, processor_config):
    processor, config = processor_config
    dataset = Qwen3OmniDataset(manifest)
    collate = Qwen3OmniCollator(processor, config)
    samples = [dataset[i] for i in [3, 1, 0, 2]]
    batch = collate(samples)
    assert batch.sample_ids == ("video", "image", "text", "audio")
    assert batch.model_inputs["input_ids"].shape == batch.labels.shape
    assert (
        batch.model_inputs["pixel_values"].shape[0]
        == batch.model_inputs["image_grid_thw"].prod().item()
    )
    assert (
        batch.model_inputs["pixel_values_videos"].shape[0]
        == batch.model_inputs["video_grid_thw"].prod().item()
    )
    for row, sample in enumerate(samples):
        single = collate([sample])
        length = single.labels.shape[1]
        torch.testing.assert_close(
            batch.model_inputs["input_ids"][row, :length], single.model_inputs["input_ids"][0]
        )
        torch.testing.assert_close(batch.labels[row, :length], single.labels[0])
        assert (batch.labels[row, length:] == -100).all()
        actual = batch.labels[row][batch.labels[row] != -100].tolist()
        expected = processor.tokenizer.encode(
            sample["text"] + processor.tokenizer.eos_token, add_special_tokens=False
        )
        assert actual == expected, sample["id"]
        for token_id in collate.media_token_ids:
            assert token_id not in actual
    # The 8 kHz input is resampled, not falsely declared to be 16 kHz.
    assert batch.media_info[3][0]["sampling_rate"] == 16000
    assert batch.media_info[3][0]["samples"] == 19200
    assert batch.media_info[0][0]["timestamps"] == [0.0, 0.5]
    assert batch.model_inputs["video_second_per_grid"].tolist() == [1.0]


def test_video_resampling_matches_processor_timing(manifest, processor_config):
    processor, config = processor_config
    sample = Qwen3OmniDataset(manifest)[3]
    batch = Qwen3OmniCollator(processor, config, video_fps=4)([sample])
    info = batch.media_info[0][0]
    assert info["timestamps"] == [0.0, 0.25, 0.5, 0.75]
    assert info["source_timestamps"] == [0.0, 0.0, 0.5, 0.5]
    assert batch.model_inputs["video_second_per_grid"].tolist() == [0.5]


def test_multiple_media_preserve_manifest_order(manifest, processor_config):
    processor, config = processor_config
    dataset = Qwen3OmniDataset(manifest)
    sample = dict(
        id="mixed",
        text="A tone and a red square.",
        media=[dataset[2]["media"][0], dataset[1]["media"][0]],
    )
    batch = Qwen3OmniCollator(processor, config)([sample])
    ids = batch.model_inputs["input_ids"][0].tolist()
    audio_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    assert ids.index(audio_id) < ids.index(image_id)
    assert [item["type"] for item in batch.media_info[0]] == ["audio", "image"]


def test_rejects_truncating_expanded_media(manifest, processor_config):
    processor, config = processor_config
    sample = Qwen3OmniDataset(manifest)[1]
    with pytest.raises(ValueError, match="do not truncate media tokens"):
        Qwen3OmniCollator(processor, config, max_length=2)([sample])


@pytest.mark.parametrize(
    "media",
    [
        [dict(type="image", path="missing.png")],
        [dict(type="audio_video", path="clip.mp4")],
    ],
)
def test_rejects_unusable_media_at_manifest_read(tmp_path, media):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(dict(id="bad", text="Text.", media=media)))
    with pytest.raises(ValueError, match="bad:"):
        Qwen3OmniDataset(path)


def test_rejects_reserved_placeholder_in_paired_text(manifest, processor_config):
    processor, config = processor_config
    sample = dict(Qwen3OmniDataset(manifest)[0], text="Unexpected <|image_pad|> token")
    with pytest.raises(ValueError, match="reserved media tokens"):
        Qwen3OmniCollator(processor, config)([sample])


def test_indexed_shards_and_stage_guard(manifest, tmp_path):
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    shards = [tmp_path / "one.jsonl", tmp_path / "two.jsonl"]
    shards[0].write_text("\n".join(json.dumps(row) for row in rows[:2]))
    shards[1].write_text("\n".join(json.dumps(row) for row in rows[2:]))
    dataset = Qwen3OmniDataset(shards, media_root=tmp_path)
    assert [dataset[i]["id"] for i in range(len(dataset))] == [row["id"] for row in rows]
    assert dataset[-1]["id"] == "video"
    assert len(dataset.offsets) == 2
    with pytest.raises(ValueError, match="not enabled for this stage"):
        Qwen3OmniDataset(shards, media_root=tmp_path, allowed_modalities={"text", "image"})


def test_weighted_sampling_reproduces_dp_and_resume():
    from collections import Counter

    from pithtrain.modules.qwen3_omni_data import OmniMixtureSampler

    groups, weights = {"text": [0, 1], "image": [2, 3]}, {"text": 1, "image": 3}
    full = list(OmniMixtureSampler(groups, weights, 2000))
    ranks = [
        list(OmniMixtureSampler(groups, weights, 2000, rank=rank, world_size=2))
        for rank in range(2)
    ]
    assert [value for pair in zip(*ranks) for value in pair] == full
    assert list(OmniMixtureSampler(groups, weights, 2000, start_sample=12)) == full[12:]
    assert (
        list(OmniMixtureSampler(groups, weights, 2000, rank=1, world_size=2, start_sample=12))
        == ranks[1][6:]
    )
    counts = Counter("text" if index < 2 else "image" for index in full)
    assert 0.70 < counts["image"] / len(full) < 0.80
    assert list(OmniMixtureSampler(groups, weights, 2000, epoch=1)) != full
    with pytest.raises(ValueError, match="positive finite weight"):
        OmniMixtureSampler(groups, {"text": 1, "image": -1}, 10)


def test_training_preparation_splits_resume_and_integrity(tmp_path, processor_config, monkeypatch):
    from pathlib import Path

    import pithtrain.tasks.prepare_omni_data as prep
    from pithtrain.modules.qwen3_omni_data import create_omni_dataloader

    recipe_path = (
        Path(__file__).parents[1] / "examples/prepare_omni_data/qwen3-omni-training/config.json"
    )
    recipe = json.loads(recipe_path.read_text())
    recipe["samples_per_modality"] = {"train": 2, "validation": 1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(recipe))
    monkeypatch.setattr(prep, "processor_for", lambda *args: processor_config)

    def source_records(modality, split, source, cache, max_scan):
        rows = (
            [("one", "speaker-a"), ("one", "speaker-a"), ("two", "speaker-b")]
            if split == "train"
            else [("leak", "speaker-a"), ("three", "speaker-c")]
        )
        for sample_id, group in rows:
            yield dict(id=sample_id, group=group, text=f"Text for {sample_id}.", origin=sample_id)

    def interrupted(*args):
        if args[1] == "validation":
            raise RuntimeError("Simulated interrupted preparation")
        yield from source_records(*args)

    cfg = prep.PrepareOmniDataCfg()
    cfg.recipe, cfg.output, cfg.cache = (
        str(config_path),
        str(tmp_path / "bundle"),
        str(tmp_path / "cache"),
    )
    cfg.stage, cfg.offline = "text", True
    monkeypatch.setattr(prep, "records", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        prep.launch(cfg)
    assert not (Path(cfg.output) / "bundle.json").exists()
    monkeypatch.setattr(prep, "records", source_records)
    bundle = prep.launch(cfg)
    assert bundle["statistics"]["train"]["text"]["samples"] == 2
    assert bundle["statistics"]["train"]["text"]["rejected"] == {"duplicate sample": 1}
    assert bundle["statistics"]["validation"]["text"]["rejected"] == {
        "source group overlaps train/validation": 1
    }
    assert (Path(cfg.output) / "tokens/train/00000.bin").exists()
    assert (Path(cfg.output) / "tokens/validation/00000.bin").exists()
    assert prep.launch(cfg) == bundle
    # The GPU batch checker consumes this same train-only export; verify its file
    # reference path on CPU without importing CUDA-dependent training modules.
    import runpy

    read_reference = runpy.run_path(str(Path(__file__).with_name("test_pretrain_data.py")))[
        "read_text_reference"
    ]
    inputs, labels, metadata = read_reference(cfg.output, 4)
    assert len(inputs) > 0 and inputs.shape == labels.shape
    torch.testing.assert_close(inputs[:, 1:], labels[:, :-1])
    assert metadata["vocab_size"] == processor_config[1].text_config.vocab_size
    processor, config = processor_config
    loader = create_omni_dataloader(cfg.output, processor, config, stage="text", split="validation")
    assert [sample_id for batch in loader for sample_id in batch.sample_ids] == ["three"]
    with pytest.raises(ValueError, match="not been prepared"):
        create_omni_dataloader(cfg.output, processor, config, stage="image", split="train")
    manifest = Path(cfg.output) / bundle["manifests"]["train"]["text"][0]
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        prep.verify_bundle(cfg.output)


def test_source_cache_detects_changes_and_offline_misses(tmp_path):
    from pithtrain.tasks.prepare_omni_data import SourceCache

    cache = SourceCache(tmp_path)
    path = cache.cached("source", ".txt", lambda dest: dest.write_text("original"))
    offline = SourceCache(tmp_path, offline=True)
    assert offline.cached("source", ".txt", None).read_text() == "original"
    with pytest.raises(FileNotFoundError, match="Offline source cache miss"):
        offline.cached("missing", ".txt", None)
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="Corrupt source cache"):
        offline.cached("source", ".txt", None)


def test_interrupted_preparation_repairs_partial_media(
    tmp_path, manifest, processor_config, monkeypatch
):
    from pathlib import Path

    import pithtrain.tasks.prepare_omni_data as prep

    recipe_path = (
        Path(__file__).parents[1] / "examples/prepare_omni_data/qwen3-omni-training/config.json"
    )
    recipe = json.loads(recipe_path.read_text())
    recipe["samples_per_modality"] = {"train": 2, "validation": 1}
    recipe["modality_sample_limits"] = {"image": {"train": 1}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(recipe))
    Image.new("RGB", (64, 64), color=(0, 0, 255)).save(tmp_path / "validation.png")
    source_paths = {"train": tmp_path / "image.png", "validation": tmp_path / "validation.png"}
    monkeypatch.setattr(prep, "processor_for", lambda *args: processor_config)
    monkeypatch.setattr(prep.SourceCache, "http_file", lambda self, url: source_paths[url])
    interrupted = True

    def records(modality, split, source, cache, max_scan):
        if split == "validation" and interrupted:
            raise RuntimeError("Simulated interruption")
        count = 2 if modality == "text" and split == "train" else 1
        for index in range(count):
            identity = f"{modality}-{split}-{index}"
            yield dict(
                id=identity, group=identity, text="A colored square.", origin=split, url=split
            )

    monkeypatch.setattr(prep, "records", records)
    cfg = prep.PrepareOmniDataCfg()
    cfg.recipe, cfg.output, cfg.cache = (
        str(config_path),
        str(tmp_path / "bundle"),
        str(tmp_path / "cache"),
    )
    cfg.stage, cfg.offline = "image", True
    with pytest.raises(RuntimeError, match="interruption"):
        prep.launch(cfg)
    partial_media = next((Path(cfg.output) / "media/image").iterdir())
    partial_media.write_bytes(b"interrupted media write")
    interrupted = False
    bundle = prep.launch(cfg)
    assert partial_media.read_bytes() == source_paths["train"].read_bytes()
    assert bundle["statistics"]["train"]["text"]["samples"] == 2
    assert bundle["statistics"]["train"]["image"]["samples"] == 1
    assert bundle["statistics"]["validation"]["image"]["samples"] == 1
    assert prep.verify_bundle(cfg.output) == bundle
    partial_media.write_bytes(b"corruption after completion")
    with pytest.raises(ValueError, match="hash mismatch"):
        prep.launch(cfg)
