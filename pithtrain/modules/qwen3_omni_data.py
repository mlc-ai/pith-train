"""Read local media/text pairs and build Qwen3-Omni Thinker pretraining inputs.

The batch keeps HF's media field names. The training data adapter wraps these
inputs in a DualPipeV Microbatch; native Omni must consume the features and
construct multimodal positions.
"""

import hashlib
import json
import math
import random
from array import array
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import av
import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


class Qwen3OmniDataset(Dataset):
    """Index JSONL shards by byte offset; decode records only when requested.

    Media paths are relative to media_root (the manifest's directory by default).
    allowed_modalities rejects incompatible samples rather than deleting their media.
    """

    def __init__(self, manifest, *, media_root=None, allowed_modalities=None):
        manifests = [manifest] if isinstance(manifest, (str, Path)) else manifest
        self.manifests = [Path(path).resolve() for path in manifests]
        if not self.manifests:
            raise ValueError("No manifest shards")
        self.root = (
            Path(media_root).resolve() if media_root is not None else self.manifests[0].parent
        )
        self.allowed_modalities = None if allowed_modalities is None else set(allowed_modalities)
        self.offsets, self.ends, self.groups = [], [], {}
        seen, count = set(), 0
        for manifest_path in self.manifests:
            offsets = array("Q")
            with manifest_path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    sample = self._record(line)
                    if sample["id"] in seen:
                        raise ValueError(f"Duplicate sample id: {sample['id']}")
                    seen.add(sample["id"])
                    kinds = {item["type"] for item in sample["media"]} or {"text"}
                    kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
                    self.groups.setdefault(kind, array("Q")).append(count)
                    offsets.append(offset)
                    count += 1
            self.offsets.append(offsets)
            self.ends.append(count)
        if not count:
            raise ValueError("The manifest has no samples")

    def _record(self, line):
        sample = json.loads(line)
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Missing sample id")
        if not isinstance(sample.get("text"), str) or not sample["text"].strip():
            raise ValueError(f"{sample_id}: expected nonempty paired text")
        if not isinstance(sample.get("media"), list):
            raise ValueError(f"{sample_id}: media must be a list (empty for text-only)")
        media = []
        for item in sample["media"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"type", "path"}
                or item["type"] not in {"image", "audio", "video"}
            ):
                raise ValueError(f"{sample_id}: expected image/audio/video with a local path")
            if not isinstance(item["path"], str):
                raise ValueError(f"{sample_id}: media path must be a string")
            path = (self.root / item["path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"{sample_id}: media is missing or outside the manifest directory")
            media.append(dict(type=item["type"], path=path))
        kinds = {item["type"] for item in media} or {"text"}
        if self.allowed_modalities is not None and not kinds <= self.allowed_modalities:
            raise ValueError(
                f"{sample_id}: modalities {sorted(kinds)} are not enabled for this stage"
            )
        return dict(id=sample_id, text=sample["text"], media=media)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        local_index = index - (self.ends[shard - 1] if shard else 0)
        with self.manifests[shard].open("rb") as stream:
            stream.seek(self.offsets[shard][local_index])
            return self._record(stream.readline())


@dataclass
class Qwen3OmniBatch:
    """Labels are ALREADY shifted for PithTrain's external next-token CE.

    For HF comparison, pass model_inputs without labels and calculate CE against
    this labels tensor externally; passing it as HF labels would shift twice.
    media_info is grouped by sample, in the manifest's original media order.
    """

    sample_ids: tuple[str, ...]
    model_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    media_info: tuple[list[dict], ...]


class Qwen3OmniCollator:
    """Media-prefix text prediction, without an instruction/chat template.

    Prefix a document boundary (the tokenizer's EOS), then predict all paired
    text tokens and EOS. Ignore padding and media wrapper/placeholder targets.
    Media tensors stay attached, never flattened into .bin.
    Video currently means visual frames only; clips with audio are rejected so
    an audio track cannot silently disappear from an Omni training sample.
    """

    def __init__(
        self,
        processor,
        thinker_config,
        *,
        max_length=2048,
        min_pixels=4096,
        max_pixels=65536,
        video_fps=2.0,
        max_video_frames=32,
        max_audio_seconds=30.0,
    ):
        if max_length < 2 or video_fps <= 0 or max_video_frames < 2:
            raise ValueError("Invalid sequence length or video sampling limits")
        if min_pixels <= 0 or max_pixels < min_pixels or max_audio_seconds <= 0:
            raise ValueError("Invalid media size/duration limits")
        self.processor = processor
        self.config = thinker_config
        self.max_length = max_length
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.video_fps, self.max_video_frames = video_fps, max_video_frames
        self.max_audio_seconds = max_audio_seconds
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.media_tokens = [
            getattr(processor, name)
            for name in (
                "image_token",
                "audio_token",
                "video_token",
                "vision_bos_token",
                "vision_eos_token",
                "audio_bos_token",
                "audio_eos_token",
            )
        ]
        self.media_token_ids = processor.tokenizer.convert_tokens_to_ids(self.media_tokens)

    def _audio(self, path):
        with sf.SoundFile(path) as stream:
            rate = stream.samplerate
            if len(stream) / rate > self.max_audio_seconds:
                raise ValueError(f"Audio exceeds {self.max_audio_seconds}s: {path}")
            audio = stream.read(dtype="float32", always_2d=True).mean(axis=1)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Empty or nonfinite audio: {path}")
        if rate != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=self.sampling_rate)
        return audio, dict(type="audio", sampling_rate=self.sampling_rate, samples=len(audio))

    def _video(self, path):
        frames, timestamps, source_timestamps = [], [], []

        def append(frame, source_time):
            if len(frames) == self.max_video_frames:
                raise ValueError(f"Video exceeds {self.max_video_frames} sampled frames: {path}")
            timestamps.append(len(frames) / self.video_fps)
            source_timestamps.append(source_time)
            frames.append(frame.to_ndarray(format="rgb24"))

        with av.open(str(path)) as container:
            if container.streams.audio:
                raise ValueError(
                    f"Video has an audio track: {path}. Synchronized audio/video is not supported yet."
                )
            start, previous, previous_time = None, None, None
            for frame in container.decode(video=0):
                if frame.time is None:
                    raise ValueError(f"Video frame has no timestamp: {path}")
                if start is None:
                    start = float(frame.time)
                timestamp = float(frame.time) - start
                if previous is not None:
                    if timestamp <= previous_time:
                        raise ValueError(f"Video timestamps are not increasing: {path}")
                    # Hold the previous frame on a uniform timeline. This also
                    # handles source FPS below the requested sampling rate.
                    while len(frames) / self.video_fps < timestamp - 1e-6:
                        append(previous, previous_time)
                previous, previous_time = frame, timestamp
            if previous is not None:
                duration = float(previous.duration * previous.time_base)
                if duration <= 0:
                    rate = container.streams.video[0].average_rate
                    if rate is None or rate <= 0:
                        raise ValueError(f"Video's last frame has no duration: {path}")
                    duration = 1 / float(rate)
                end = previous_time + duration
                while len(frames) / self.video_fps < end - 1e-6:
                    append(previous, previous_time)
        if len(frames) < 2:
            raise ValueError(f"Video needs at least two sampled frames: {path}")
        return np.stack(frames), dict(
            type="video",
            fps=self.video_fps,
            timestamps=timestamps,
            source_timestamps=source_timestamps,
        )

    def __call__(self, samples):
        with torch.device("cpu"):
            return self._collate(samples)

    def _collate(self, samples):
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        processor = self.processor
        texts, images, audio, videos, media_info = [], [], [], [], []
        for sample in samples:
            if any(token in sample["text"] for token in self.media_tokens):
                raise ValueError(f"{sample['id']}: paired text contains reserved media tokens")
            # A boundary also gives the first text-only token a preceding input.
            prefix, info = [processor.tokenizer.eos_token], []
            for item in sample["media"]:
                kind, path = item["type"], item["path"]
                if kind == "image":
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        images.append(image)
                        info.append(dict(type=kind, width=image.width, height=image.height))
                    prefix.append(
                        processor.vision_bos_token
                        + processor.image_token
                        + processor.vision_eos_token
                    )
                elif kind == "audio":
                    waveform, details = self._audio(path)
                    audio.append(waveform)
                    info.append(details)
                    prefix.append(
                        processor.audio_bos_token
                        + processor.audio_token
                        + processor.audio_eos_token
                    )
                else:
                    video, details = self._video(path)
                    videos.append(video)
                    info.append(details)
                    prefix.append(
                        processor.vision_bos_token
                        + processor.video_token
                        + processor.vision_eos_token
                    )
            texts.append("".join(prefix) + sample["text"] + processor.tokenizer.eos_token)
            media_info.append(info)
        inputs = dict(
            processor(
                text=texts,
                images=images or None,
                audio=audio or None,
                videos=videos or None,
                text_kwargs=dict(padding=True, padding_side="right", add_special_tokens=False),
                images_kwargs=dict(min_pixels=self.min_pixels, max_pixels=self.max_pixels),
                audio_kwargs=dict(
                    sampling_rate=self.sampling_rate, n_window=self.config.audio_config.n_window
                ),
                videos_kwargs=dict(
                    fps=self.video_fps,
                    do_sample_frames=False,
                    cap_pixels_per_frame=True,
                    size=dict(shortest_edge=self.min_pixels, longest_edge=self.max_pixels),
                    position_id_per_seconds=self.config.position_id_per_seconds,
                    use_audio_in_video=False,
                ),
                return_tensors="pt",
            )
        )
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if ids.shape[1] - 1 > self.max_length:
            raise ValueError(
                f"Expanded media/text length {ids.shape[1] - 1} exceeds {self.max_length}; do not truncate media tokens"
            )
        if ids.min() < 0 or ids.max() >= self.config.text_config.vocab_size:
            raise ValueError("Processor token IDs exceed the model vocabulary")
        labels = ids[:, 1:].clone()
        valid = mask[:, 1:].bool() & mask[:, :-1].bool()
        for token_id in self.media_token_ids:
            valid &= labels != token_id
        labels.masked_fill_(~valid, -100)
        if not (labels != -100).any(dim=1).all():
            raise ValueError("Each sample must have a text target")
        inputs["input_ids"] = ids[:, :-1].contiguous()
        inputs["attention_mask"] = mask[:, :-1].contiguous()
        return Qwen3OmniBatch(
            tuple(sample["id"] for sample in samples),
            inputs,
            labels.contiguous(),
            tuple(media_info),
        )


class OmniMixtureSampler(Sampler):
    """Deterministic weighted draws WITH replacement, partitioned by data rank.

    start_sample counts globally consumed samples, not DataLoader prefetches.
    A draw depends only on seed/epoch/global position, so resume recreates it.
    Pass data-parallel rank/size; EP and PP ranks must reuse their DP stream.
    """

    def __init__(
        self,
        groups,
        weights,
        num_samples,
        *,
        seed=1234,
        epoch=0,
        rank=0,
        world_size=1,
        start_sample=0,
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid data-parallel rank/size")
        if num_samples <= 0 or num_samples % world_size or start_sample % world_size:
            raise ValueError(
                "Global sample counts must be positive and divisible by data world size"
            )
        if not 0 <= start_sample <= num_samples:
            raise ValueError("Invalid globally consumed sample count")
        if set(weights) != set(groups) or any(
            not math.isfinite(w) or w <= 0 for w in weights.values()
        ):
            raise ValueError("Specify one positive finite weight for each selected modality")
        if any(not len(indices) for indices in groups.values()):
            raise ValueError("Cannot sample an empty modality")
        self.groups, self.weights = groups, weights
        self.num_samples, self.start_sample = num_samples, start_sample
        self.seed, self.epoch, self.rank, self.world_size = seed, epoch, rank, world_size

    def __len__(self):
        return (self.num_samples - self.start_sample) // self.world_size

    def __iter__(self):
        names = sorted(self.groups)
        weights = [self.weights[name] for name in names]
        for position in range(self.start_sample + self.rank, self.num_samples, self.world_size):
            rng = random.Random(f"{self.seed}:{self.epoch}:{position}")
            modality = rng.choices(names, weights=weights, k=1)[0]
            yield rng.choice(self.groups[modality])


def create_omni_dataloader(
    root,
    processor,
    thinker_config,
    *,
    stage,
    split,
    batch_size=1,
    num_samples=None,
    weights=None,
    epoch=0,
    rank=0,
    world_size=1,
    start_sample=0,
    num_workers=0,
    seed=None,
    batch_cfg=None,
):
    """Read a prepared bundle and select only modalities supported by this stage.

    Training uses weighted sampling. Validation reads each selected record once,
    in order; this initial evaluation reader is single-process (no DP padding).
    This returns data batches, not an already-connected DualPipeV training task.
    """
    root = Path(root).resolve()
    checksums = dict(
        (name, digest)
        for digest, name in (
            line.split("  ", 1) for line in (root / "checksums.sha256").read_text().splitlines()
        )
    )

    def verify_metadata(path):
        name = str(path.relative_to(root))
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksums.get(name) != actual:
            raise ValueError(f"Dataset metadata hash mismatch: {name}")

    verify_metadata(root / "bundle.json")
    bundle = json.loads((root / "bundle.json").read_text())
    if bundle["status"] != "complete" or split not in {"train", "validation"}:
        raise ValueError("Expected a completed bundle and train/validation split")
    recipe = bundle["recipe"]
    modalities = recipe["stages"][stage]
    available = bundle["manifests"][split]
    missing = set(modalities) - set(available)
    if missing:
        raise ValueError(f"Data for this stage has not been prepared: {sorted(missing)}")
    manifests = [(root / name).resolve() for kind in modalities for name in available[kind]]
    if any(not path.is_relative_to(root) for path in manifests):
        raise ValueError("Manifest outside the bundle directory")
    for path in manifests:
        verify_metadata(path)
    dataset = Qwen3OmniDataset(manifests, media_root=root, allowed_modalities=modalities)
    collator = Qwen3OmniCollator(
        processor, thinker_config, **(recipe["batch"] if batch_cfg is None else batch_cfg)
    )
    if split == "train":
        selected_weights = (
            weights
            if weights is not None
            else {kind: recipe["sampling_weights"][kind] for kind in modalities}
        )
        sampler = OmniMixtureSampler(
            dataset.groups,
            selected_weights,
            num_samples if num_samples is not None else len(dataset),
            seed=recipe["seed"] if seed is None else seed,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            start_sample=start_sample,
        )
    else:
        if (
            world_size != 1
            or rank != 0
            or start_sample
            or weights is not None
            or num_samples is not None
        ):
            raise ValueError(
                "Validation reads each record once on one data rank; no mixture/resume overrides"
            )
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
        generator=torch.Generator(device="cpu").manual_seed(
            recipe["seed"] if seed is None else seed
        ),
    )
