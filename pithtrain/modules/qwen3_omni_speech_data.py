"""Qwen3-Omni chat SFT and explicitly supplied external RVQ supervision.

No codec encoder or waveform training loss is inferred from the checkpoint. The
processor handles input media; output codec targets must come from a separately
identified encoder. Labels are shifted exactly once, after media expansion.
"""

import hashlib
import importlib.metadata
import json
import os
import random
import re
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from pithtrain.modules.microbatch import Microbatch

_MEDIA = frozenset({"pixel_values", "pixel_values_videos", "input_features"})
_POSITIONS = frozenset(
    {
        "image_grid_thw",
        "video_grid_thw",
        "video_second_per_grid",
        "audio_feature_lengths",
        "use_audio_in_video",
    }
)
_BLOCK = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n", re.DOTALL)


def _local(root, name):
    if not isinstance(name, str) or not name or "://" in name:
        raise ValueError("Expected a local relative file path")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("File missing or outside the dataset root")
    return path


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identity_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _identity_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return dict(
            dtype=str(value.dtype),
            shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
    if isinstance(value, np.generic):
        return value.item()
    if type(value).__name__ == "AddedToken":
        return {
            name: getattr(value, name)
            for name in ("content", "single_word", "lstrip", "rstrip", "normalized", "special")
        }
    if hasattr(value, "to_dict"):
        return _identity_value(value.to_dict())
    raise ValueError(f"Unsupported identity value: {type(value).__name__}")


def _fingerprint(value):
    encoded = json.dumps(
        _identity_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _class_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _file_digest(path):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    current = path.stat()
    key = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if key(before) != key(after) or key(after) != key(current):
        raise ValueError("Input file changed while its identity was being read")
    return after.st_size, digest


def _sample_assets(row, root):
    assets = []
    for message in row["messages"]:
        content = message["content"]
        for item in content if isinstance(content, list) else ():
            if item["type"] != "text":
                assets.append(
                    (item["type"], item["path"], *_file_digest(_local(root, item["path"])))
                )
    if "codec" in row:
        name = row["codec"]["path"]
        assets.append(("codec", name, *_file_digest(_local(root, name))))
    return tuple(assets)


def _validate_assets(root, assets):
    for kind, name, size, digest in assets:
        if _file_digest(_local(root, name)) != (size, digest):
            raise ValueError(f"{kind} media identity changed: {name}")


@contextmanager
def _preserve_rng_on_failure():
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    except BaseException:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        raise


class Qwen3OmniSpeechDataset:
    """Indexed, local JSONL; media paths cannot escape the manifest directory.

    A row contains ``id``, ``messages`` and optional ``codec``. The last message
    is assistant text. Media is supported only in user messages, matching the
    Talker conditioning position contract. No remote payload is downloaded.
    """

    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.offsets = []
        self.row_hashes = []
        self.asset_identities = []
        ids = set()
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                row = self._validate(json.loads(line))
                if row["id"] in ids:
                    raise ValueError("Duplicate sample id")
                ids.add(row["id"])
                self.offsets.append(offset)
                self.row_hashes.append(hashlib.sha256(line).digest())
                self.asset_identities.append(_sample_assets(row, self.root))
        if not self.offsets:
            raise ValueError("Empty speech dataset")
        self.manifest_sha256 = digest.hexdigest()
        self.asset_identities = tuple(self.asset_identities)
        self.fingerprint = _fingerprint(
            dict(schema=1, manifest=self.manifest_sha256, assets=self.asset_identities)
        )
        self.validate_identity()

    def validate_identity(self, indices=None):
        """Re-read bytes, not timestamps; restore checks every referenced asset."""
        if _file_digest(self.path)[1] != self.manifest_sha256:
            raise ValueError("Manifest changed after indexing")
        indices = range(len(self)) if indices is None else indices
        for index in indices:
            _validate_assets(self.root, self.asset_identities[index])

    def _validate(self, row):
        if not isinstance(row, dict) or set(row) - {"id", "messages", "codec"}:
            raise ValueError("Expected id/messages/optional codec")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("A nonempty sample id is required")
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(messages[-1], dict)
            or messages[-1].get("role") != "assistant"
        ):
            raise ValueError("Messages must end with an assistant response")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("Messages require exactly role and content; tools are unsupported")
            role, content = message["role"], message["content"]
            if role not in {"system", "user", "assistant"} or (role == "system" and index != 0):
                raise ValueError("Unsupported role or misplaced system message")
            if isinstance(content, str):
                continue
            if role != "user" or not isinstance(content, list) or not content:
                raise ValueError("Only user messages may contain media lists")
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("Invalid content item")
                if item.get("type") == "text":
                    if set(item) != {"type", "text"} or not isinstance(item["text"], str):
                        raise ValueError("Invalid text content")
                elif item.get("type") in {"image", "audio", "video"}:
                    if set(item) != {"type", "path"}:
                        raise ValueError("Media requires exactly type and path")
                    _local(self.root, item["path"])
                else:
                    raise ValueError("Unsupported media type")
        if not any(message["role"] == "user" for message in messages):
            raise ValueError("At least one user message is required")
        if "codec" in row:
            if not isinstance(row["codec"], dict):
                raise ValueError("Codec supervision requires an explicit manifest")
            _local(self.root, row["codec"].get("path"))
        return row

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        self.validate_identity((index,))
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            line = stream.readline()
        if hashlib.sha256(line).digest() != self.row_hashes[index]:
            raise ValueError("Manifest changed after indexing")
        row = self._validate(json.loads(line))
        return dict(row, root=self.root, _asset_identity=self.asset_identities[index])


class ExternalRVQEncoder(Protocol):
    """External preparation interface, not an encoder shipped by this package.

    Implementations supply a provenance manifest and a local NPZ containing
    integer ``codes[T,16]`` and boolean ``codebook_mask[T,16]``. Persist these and
    their SHA256 before training; raw waveform input is never substituted.
    """

    def encode(self, waveform: np.ndarray, *, sample_rate: int, sample_id: str) -> dict: ...


def load_codec_supervision(sample, *, max_bytes=64 * 1024 * 1024):
    """Read a hash-pinned, pickle-free external label archive for the final answer.

    Alignment describes the *source* waveform segment, not a fabricated decoder
    output length: external encoders can pad frames. Optional waveform targets
    must exactly match the declared source segment length.
    """
    manifest = sample["codec"]
    required = {
        "path",
        "sha256",
        "sample_id",
        "assistant_index",
        "transcript",
        "source",
        "num_codebooks",
        "codebook_size",
        "num_frames",
        "alignment",
    }
    if set(manifest) != required:
        raise ValueError("Incomplete or unknown codec manifest fields")
    if (
        manifest["sample_id"] != sample["id"]
        or manifest["assistant_index"] != len(sample["messages"]) - 1
    ):
        raise ValueError("Codec labels must identify this sample's final assistant response")
    if manifest["transcript"] != sample["messages"][-1]["content"]:
        raise ValueError("Codec transcript differs from the final assistant text")
    source = manifest["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"model_id", "revision"}
        or not all(isinstance(v, str) and v.strip() for v in source.values())
    ):
        raise ValueError("Explicit external codec model_id and immutable revision are required")
    if source["revision"].lower() in {"main", "master", "latest"}:
        raise ValueError("Pin the external codec revision")
    if manifest["num_codebooks"] != 16 or manifest["codebook_size"] != 2048:
        raise ValueError("Qwen3-Omni supervision requires 16 codebooks of size 2048")
    frames = _integer(manifest["num_frames"], "num_frames")
    alignment = manifest["alignment"]
    if not isinstance(alignment, dict) or set(alignment) != {
        "sample_rate",
        "frame_hop_samples",
        "start_sample",
        "num_samples",
    }:
        raise ValueError("Explicit source waveform alignment is required")
    if alignment["sample_rate"] != 24000 or alignment["frame_hop_samples"] != 1920:
        raise ValueError("Qwen3-Omni codec timing is 24 kHz / 1920 samples per frame")
    _integer(alignment["start_sample"], "start_sample")
    samples = _integer(alignment["num_samples"], "num_samples")
    if (frames == 0) != (samples == 0):
        raise ValueError("Empty codec EOS requires an empty source segment")
    path = _local(Path(sample["root"]).resolve(), manifest["path"])
    if path.suffix != ".npz" or path.stat().st_size > max_bytes:
        raise ValueError("Expected a bounded NPZ archive")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
            raise ValueError("Codec archive SHA256 mismatch")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (
            len({m.filename for m in members}) != len(members)
            or sum(m.file_size for m in members) > max_bytes
        ):
            raise ValueError("Codec archive exceeds the uncompressed limit or repeats a member")
    with np.load(path, allow_pickle=False) as archive:
        if not {"codes", "codebook_mask"} <= set(archive.files) or set(archive.files) - {
            "codes",
            "codebook_mask",
            "waveform",
        }:
            raise ValueError("NPZ requires codes/codebook_mask and optional waveform")
        codes, mask = archive["codes"], archive["codebook_mask"]
        if (
            codes.dtype != np.int64
            or codes.shape != (frames, 16)
            or mask.dtype != np.bool_
            or mask.shape != codes.shape
        ):
            raise ValueError("Codec codes/mask must be int64/bool [time,16]")
        # Loss masking does not remove autoregressive input dependencies.
        # Every active frame needs all known groups, including unsupervised ones.
        if np.any((codes < 0) | (codes >= 2048)):
            raise ValueError(
                "All active codec IDs, including unsupervised groups, must be in [0,2048)"
            )
        result = dict(
            codes=torch.from_numpy(codes.copy())[None],
            codebook_mask=torch.from_numpy(mask.copy())[None],
            code_lengths=torch.tensor([frames], dtype=torch.long),
            codec_provenance=manifest,
        )
        if "waveform" in archive:
            waveform = archive["waveform"]
            if (
                waveform.dtype != np.float32
                or waveform.shape != (samples,)
                or not np.isfinite(waveform).all()
            ):
                raise ValueError("Waveform target must be finite float32 matching alignment length")
            result["target_waveform"] = torch.from_numpy(waveform.copy())[None]
    return result


class Qwen3OmniSpeechCollator:
    """One unpadded conversation, with exact expanded-token assistant boundaries.

    Complete rendered ChatML is parsed before processing. Every assistant start
    and end is proved by processing the corresponding exact rendered substring
    with preceding user media and comparing all prefix tokens. This also handles
    templates whose thinking text changes with conversation context. Unsupported
    templates fail closed; tokenizer offsets or a guessed assistant mask are not
    used. Inputs end immediately before the final im_end target.
    """

    def __init__(self, processor, thinker_config, **media_limits):
        from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator

        self.processor, self.config = processor, thinker_config
        self.media = Qwen3OmniCollator(processor, thinker_config, **media_limits)

    def checkpoint_identity(self):
        """Pure identity of preprocessing semantics, assets and public provenance."""
        tokenizer = self.processor.tokenizer
        if not hasattr(tokenizer, "backend_tokenizer"):
            raise ValueError("Strict speech data requires a serializable tokenizer backend")
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        # These are mutable per-call caches. _process supplies both explicitly.
        backend.pop("padding", None)
        backend.pop("truncation", None)
        tokenizer_options = {
            key: value for key, value in tokenizer.init_kwargs.items() if key != "backend"
        }
        return dict(
            schema=1,
            collator=_class_name(self),
            processor=_class_name(self.processor),
            processor_config=self.processor.to_dict(),
            processor_arrays={
                name: {
                    key: value
                    for key, value in vars(getattr(self.processor, name)).items()
                    if isinstance(value, np.ndarray)
                }
                for name in ("image_processor", "video_processor", "feature_extractor")
            },
            chat_template=self.processor.chat_template,
            tokenizer=dict(
                implementation=_class_name(tokenizer),
                backend=backend,
                options=tokenizer_options,
                specials=tokenizer.special_tokens_map,
                padding_side=tokenizer.padding_side,
                truncation_side=tokenizer.truncation_side,
                model_max_length=tokenizer.model_max_length,
            ),
            media_tokens={
                name: getattr(self.processor, name)
                for name in (
                    "image_token",
                    "audio_token",
                    "video_token",
                    "vision_bos_token",
                    "vision_eos_token",
                    "audio_bos_token",
                    "audio_eos_token",
                )
            },
            thinker_config=self.config.to_dict(),
            config_revision=getattr(self.config, "_commit_hash", None),
            media_limits={
                name: getattr(self.media, name)
                for name in (
                    "max_length",
                    "min_pixels",
                    "max_pixels",
                    "video_fps",
                    "max_video_frames",
                    "max_audio_seconds",
                    "use_audio_in_video",
                    "sampling_rate",
                    "media_tokens",
                    "media_token_ids",
                )
            },
            versions={
                name: importlib.metadata.version(name)
                for name in (
                    "transformers",
                    "tokenizers",
                    "numpy",
                    "Pillow",
                    "soundfile",
                    "av",
                    "librosa",
                )
            },
        )

    def _load_media(self, messages, root):
        from PIL import Image

        from pithtrain.modules.qwen3_omni_data import decode_video_audio

        by_message, video_modes = [], []
        for message in messages:
            payload = {"images": [], "audio": [], "videos": []}
            content = message["content"]
            for item in content if isinstance(content, list) else []:
                kind = item["type"]
                if kind == "text":
                    continue
                path = _local(root, item["path"])
                if kind == "image":
                    with Image.open(path) as image:
                        payload["images"].append(image.convert("RGB"))
                elif kind == "audio":
                    payload["audio"].append(self.media._audio(path)[0])
                else:
                    frames, info = self.media._video(path)
                    payload["videos"].append(frames)
                    video_modes.append(info["has_audio"])
                    if info["has_audio"]:
                        waveform, _ = decode_video_audio(
                            path,
                            start_seconds=info["start_seconds"],
                            duration_seconds=info["duration_seconds"],
                            sampling_rate=self.media.sampling_rate,
                            max_seconds=self.media.max_audio_seconds,
                        )
                        payload["audio"].append(waveform)
            by_message.append(payload)
        if any(video_modes) and not all(video_modes):
            raise ValueError("Cannot mix videos with and without audio")
        return by_message, any(video_modes)

    def _process(self, text, payloads, synchronized):
        media = {
            key: [value for payload in payloads for value in payload[key]]
            for key in ("images", "audio", "videos")
        }
        m = self.media
        inputs = dict(
            self.processor(
                text=[text],
                **{key: value or None for key, value in media.items()},
                text_kwargs=dict(padding=False, add_special_tokens=False),
                images_kwargs=dict(min_pixels=m.min_pixels, max_pixels=m.max_pixels),
                audio_kwargs=dict(
                    sampling_rate=m.sampling_rate, n_window=self.config.audio_config.n_window
                ),
                videos_kwargs=dict(
                    fps=m.video_fps,
                    do_sample_frames=False,
                    cap_pixels_per_frame=True,
                    size=dict(shortest_edge=m.min_pixels, longest_edge=m.max_pixels),
                    position_id_per_seconds=self.config.position_id_per_seconds,
                    use_audio_in_video=synchronized,
                ),
                return_tensors="pt",
            )
        )
        if synchronized and media["videos"]:
            inputs["use_audio_in_video"] = torch.tensor(True)
        return inputs

    def __call__(self, sample):
        with _preserve_rng_on_failure(), torch.device("cpu"):
            root = Path(sample["root"]).resolve()
            assets = sample.get("_asset_identity")
            if assets is None:
                assets = _sample_assets(sample, root)
            _validate_assets(root, assets)
            result = self._collate(sample)
            # Decoders reopen paths; reject any replacement spanning preparation.
            _validate_assets(root, assets)
            return result

    def _collate(self, sample):
        root = Path(sample["root"]).resolve()
        # Recheck prepared-tensor callers too, not only JSONL users.
        validator = object.__new__(Qwen3OmniSpeechDataset)
        validator.root = root
        validator._validate(
            {key: value for key, value in sample.items() if key not in {"root", "_asset_identity"}}
        )
        messages = sample["messages"]
        reserved = (
            set(self.processor.tokenizer.all_special_tokens)
            | set(self.media.media_tokens)
            | {"<|im_start|>", "<|im_end|>"}
        )
        for message in messages:
            content = message["content"]
            texts = (
                [content]
                if isinstance(content, str)
                else [item["text"] for item in content if item["type"] == "text"]
            )
            if any(token in text for token in reserved for text in texts):
                raise ValueError("Message text contains reserved template/media tokens")
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        blocks = list(_BLOCK.finditer(rendered))
        if "".join(match.group(0) for match in blocks) != rendered or [
            match.group(1) for match in blocks
        ] != [message["role"] for message in messages]:
            raise ValueError("Unsupported chat template: exact ChatML role boundaries required")
        payloads, synchronized = self._load_media(messages, root)
        inputs = self._process(rendered, payloads, synchronized)
        ids, attention = inputs["input_ids"], inputs["attention_mask"]
        if ids.ndim != 2 or ids.shape[0] != 1 or not attention.bool().all():
            raise ValueError("Expected one unpadded conversation")
        if ids.min() < 0 or ids.max() >= self.config.text_config.vocab_size:
            raise ValueError("Processor token IDs exceed model vocabulary")
        valid = torch.zeros_like(ids, dtype=torch.bool)
        prompt_length, terminal = None, None
        end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        for index, block in enumerate(blocks):
            if block.group(1) != "assistant":
                continue
            header_end = block.start(2)
            # Exclude only known template newline following the terminal token.
            response_end = block.end() - 1
            boundaries = []
            for char_end in (header_end, response_end):
                prefix = self._process(rendered[:char_end], payloads[:index], synchronized)[
                    "input_ids"
                ]
                length = prefix.shape[1]
                if length > ids.shape[1] or not torch.equal(prefix, ids[:, :length]):
                    raise ValueError("Expanded assistant prefix does not exactly align")
                boundaries.append(length)
            start, end = boundaries
            header = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            if len(header) != 3 or ids[0, start - 3 : start].tolist() != header:
                raise ValueError("Talker requires the exact three-token assistant header")
            if end <= start or ids[0, end - 1] != end_id:
                raise ValueError("Assistant response must end in im_end")
            valid[:, start:end] = True
            prompt_length, terminal = start, end
        if terminal is None or blocks[-1].group(1) != "assistant":
            raise ValueError("Missing final assistant response")
        if terminal - 1 > self.media.max_length:
            raise ValueError("Expanded conversation exceeds max_length; never truncate media")
        labels = ids[:, 1:terminal].clone()
        labels.masked_fill_(~valid[:, 1:terminal], -100)
        inputs["input_ids"] = ids[:, : terminal - 1].contiguous()
        inputs["attention_mask"] = attention[:, : terminal - 1].contiguous()
        context = {key: value for key, value in inputs.items() if key not in _MEDIA}
        features = {key: value for key, value in inputs.items() if key in _MEDIA}
        positions = {key: value for key, value in context.items() if key in _POSITIONS}
        if "feature_attention_mask" in context and "audio_feature_lengths" not in positions:
            positions["audio_feature_lengths"] = context["feature_attention_mask"].sum(-1)
        # Thinker processor names differ from Talker's public RoPE API.
        # Rename this separate dictionary only; preserve the native Thinker context.
        for source, target in (
            ("audio_feature_lengths", "audio_seqlens"),
            ("video_second_per_grid", "second_per_grids"),
        ):
            if source in positions:
                positions[target] = positions.pop(source)
        result = dict(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
            labels=labels.contiguous(),
            model_context=context,
            media_inputs=features,
            prompt_lengths=torch.tensor([prompt_length], dtype=torch.long),
            sample_ids=(sample["id"],),
            position_kwargs=positions,
        )
        if "codec" in sample:
            result.update(load_codec_supervision(sample))
        return result


class Qwen3OmniSpeechSFTData:
    """Finite indexed scans for Thinker SFT; persist alongside task/optimizer/RNG.

    get_batch advances scans only after successful preparation. A globally empty
    target batch can be skipped without commit_update; updates never select data.
    Speech codec targets are exposed by the collator for the speech trainer, not
    silently attached to the text-only SFT objective. Each scan prepares exactly
    microbatches_per_scan consecutive samples atomically. Incomplete tail groups
    are rejected at construction; no rows are dropped, duplicated or padded.
    """

    def __init__(self, dataset, collator, *, microbatches_per_scan=1):
        self.dataset, self.collator = dataset, collator
        self.microbatches_per_scan = _integer(microbatches_per_scan, "microbatches_per_scan", 1)
        if len(dataset) % self.microbatches_per_scan:
            raise ValueError(
                "Dataset length must be divisible by microbatches_per_scan; "
                "an incomplete final scan is not padded or discarded"
            )
        if not callable(collator) or not callable(getattr(collator, "checkpoint_identity", None)):
            raise ValueError(
                "Strict data requires a callable collator with a pure checkpoint_identity protocol"
            )
        self.collator_fingerprint = _fingerprint(collator.checkpoint_identity())
        dataset.validate_identity()
        self.total_scans = len(dataset) // self.microbatches_per_scan
        self.scanned_batches = self.completed_updates = 0
        self.last_update_scan = -1

    def _validate_pipeline(self, indices=None):
        self.dataset.validate_identity(indices)
        if _fingerprint(self.collator.checkpoint_identity()) != self.collator_fingerprint:
            raise ValueError("Processor/template/collator identity changed")

    def get_batch(self, scan_index, device):
        with _preserve_rng_on_failure():
            return self._get_batch(scan_index, device)

    def _get_batch(self, scan_index, device):
        if type(scan_index) is not int or scan_index != self.scanned_batches:
            raise ValueError("scan_index must match the next checkpointed scan")
        if scan_index >= self.total_scans:
            raise IndexError("Finite dataset exhausted; no repeated or fabricated microbatches")
        start = scan_index * self.microbatches_per_scan
        indices = range(start, start + self.microbatches_per_scan)
        self._validate_pipeline(indices)
        batches = []
        for index in indices:
            prepared = self.collator(self.dataset[index])
            context = {key: value.to(device) for key, value in prepared["model_context"].items()}
            media = {key: value.to(device) for key, value in prepared["media_inputs"].items()}
            batches.append(
                Microbatch(
                    model_inputs=(context["input_ids"],),
                    cu_seqlens=None,
                    objective_inputs=(prepared["labels"].to(device),),
                    model_context=context,
                    media_inputs=media,
                    sample_ids=prepared["sample_ids"],
                )
            )
        # No cursor moves unless preparation and post-read identity checks succeed.
        self._validate_pipeline(indices)
        self.scanned_batches += 1
        return batches

    def commit_update(self, *, scan_index, update_index):
        if (
            type(scan_index) is not int
            or type(update_index) is not int
            or scan_index != self.scanned_batches - 1
            or scan_index <= self.last_update_scan
            or update_index != self.completed_updates
        ):
            raise ValueError("Update must commit the most recent scan exactly once")
        self.last_update_scan = scan_index
        self.completed_updates += 1

    def state_dict(self):
        return dict(
            fingerprint=self.dataset.fingerprint,
            collator_fingerprint=self.collator_fingerprint,
            microbatches_per_scan=self.microbatches_per_scan,
            scanned_batches=self.scanned_batches,
            completed_updates=self.completed_updates,
            last_update_scan=self.last_update_scan,
        )

    def validate_state_dict(self, state):
        """Pure checkpoint preflight; no counters change on success or failure."""
        self._validate_pipeline()
        if (
            set(state)
            != {
                "fingerprint",
                "collator_fingerprint",
                "microbatches_per_scan",
                "scanned_batches",
                "completed_updates",
                "last_update_scan",
            }
            or state["fingerprint"] != self.dataset.fingerprint
            or state["collator_fingerprint"] != self.collator_fingerprint
            or type(state["microbatches_per_scan"]) is not int
            or state["microbatches_per_scan"] != self.microbatches_per_scan
        ):
            raise ValueError("Data checkpoint does not match this manifest")
        scans = _integer(state["scanned_batches"], "scanned_batches")
        updates = _integer(state["completed_updates"], "completed_updates")
        last = _integer(state["last_update_scan"], "last_update_scan", -1)
        if (
            not updates <= scans <= self.total_scans
            or not -1 <= last < scans
            or (updates == 0) != (last == -1)
            or (updates > 0 and last + 1 < updates)
        ):
            raise ValueError("Invalid scan/update checkpoint counters")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.scanned_batches = state["scanned_batches"]
        self.completed_updates = state["completed_updates"]
        self.last_update_scan = state["last_update_scan"]


class Qwen3OmniSpeechData(Qwen3OmniSpeechSFTData):
    """Native speech task adapter: one explicit forward-kwargs dict per microbatch.

    User media tensors are merged into model_context for native speech forward.
    Codec targets are mandatory. The same scan/update checkpoint protocol is used
    by the text adapter, including scans without an optimizer update. This adapter
    deliberately retains one sample per scan, independent of SFT pipeline grouping.
    """

    def __init__(self, dataset, collator):
        super().__init__(dataset, collator, microbatches_per_scan=1)

    def _get_batch(self, scan_index, device):
        if type(scan_index) is not int or scan_index != self.scanned_batches:
            raise ValueError("scan_index must match the next checkpointed scan")
        self._validate_pipeline((scan_index,))
        prepared = self.collator(self.dataset[scan_index])
        if not {"codes", "code_lengths", "codebook_mask", "codec_provenance"} <= prepared.keys():
            raise ValueError("Speech training needs explicit external codec supervision")

        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value

        context = dict(prepared["model_context"], **prepared["media_inputs"])
        keys = {
            "input_ids",
            "prompt_lengths",
            "codes",
            "code_lengths",
            "codebook_mask",
            "position_kwargs",
            "labels",
            "sample_ids",
            "codec_provenance",
            "target_waveform",
        }
        batch = move({key: value for key, value in prepared.items() if key in keys})
        batch["model_context"] = move(context)
        self._validate_pipeline((scan_index,))
        self.scanned_batches += 1
        return [batch]
