"""Prepare versioned, split-aware Omni media/text corpora with bounded defaults."""

import fnmatch
import hashlib
import io
import json
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

import av
import pyarrow.parquet as pq
import soundfile as sf
import zstandard as zstd
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, snapshot_download
from PIL import UnidentifiedImageError
from transformers import AutoConfig, Qwen3OmniMoeProcessor

from pithtrain.config import SlottedDefault
from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator
from pithtrain.tasks.tokenize_corpus import Worker, Writer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path, data):
    atomic_write(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())


class SourceCache:
    """Cache only requested files/Parquet row groups, checking cache hashes on reuse."""

    def __init__(self, root, offline=False):
        self.root, self.offline = Path(root), offline
        self.fs = None

    def cached(self, key, suffix, create):
        digest = hashlib.sha256(key.encode()).hexdigest()
        path = self.root / (digest + suffix)
        checksum = path.with_suffix(path.suffix + ".sha256")
        if path.exists() and checksum.exists():
            if sha256(path) != checksum.read_text().strip():
                raise RuntimeError(f"Corrupt source cache: {path}")
            return path
        if self.offline:
            raise FileNotFoundError(f"Offline source cache miss: {key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        create(temporary)
        temporary.replace(path)
        atomic_write(checksum, (sha256(path) + "\n").encode())
        return path

    def hf_path(self, source, filename):
        return f"datasets/{source['repo_id']}@{source['revision']}/{filename}"

    def open_hf(self, source, filename):
        if self.offline:
            raise RuntimeError("Network access requested in offline mode")
        if self.fs is None:
            self.fs = HfFileSystem()
        return self.fs.open(self.hf_path(source, filename), "rb", block_size=65536)

    def files(self, source, patterns):
        if not any(any(char in pattern for char in "*?[") for pattern in patterns):
            return patterns
        key = f"inventory:{source['repo_id']}@{source['revision']}"
        path = self.cached(
            key,
            ".json",
            lambda dest: write_json(
                dest,
                HfApi().list_repo_files(
                    source["repo_id"], repo_type="dataset", revision=source["revision"]
                ),
            ),
        )
        files = json.loads(path.read_text())
        selected = sorted(
            {name for name in files for pattern in patterns if fnmatch.fnmatch(name, pattern)}
        )
        if not selected:
            raise ValueError(f"No source files match {patterns}")
        return selected

    def file(self, source, filename):
        def create(dest):
            with self.open_hf(source, filename) as stream, dest.open("wb") as output:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(chunk)

        return self.cached(self.hf_path(source, filename), Path(filename).suffix, create)

    def http_file(self, url):
        def create(dest):
            with urlopen(url, timeout=45) as stream, dest.open("wb") as output:
                for chunk in iter(lambda: stream.read(65536), b""):
                    output.write(chunk)

        return self.cached(url, ".jpg", create)

    def parquet(self, source, patterns):
        for filename in self.files(source, patterns):
            key = self.hf_path(source, filename)

            def metadata(dest):
                with self.open_hf(source, filename) as stream:
                    write_json(dest, {"groups": pq.ParquetFile(stream).metadata.num_row_groups})

            count = json.loads(self.cached(key + ":metadata", ".json", metadata).read_text())[
                "groups"
            ]
            for group in range(count):

                def create(dest):
                    with self.open_hf(source, filename) as stream:
                        table = pq.ParquetFile(stream).read_row_group(group)
                        pq.write_table(table, dest)

                path = self.cached(f"{key}:row-group:{group}", ".parquet", create)
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                    yield from batch.to_pylist()

    def text(self, source, max_scan):
        for filename in self.files(source, source["files"]):

            def create(dest):
                with self.open_hf(source, filename) as stream:
                    with zstd.ZstdDecompressor().stream_reader(stream) as decoded:
                        with (
                            io.TextIOWrapper(decoded, encoding="utf-8") as lines,
                            dest.open("w") as output,
                        ):
                            for index, line in enumerate(lines):
                                if index == max_scan:
                                    break
                                output.write(json.dumps({"text": json.loads(line)["text"]}) + "\n")

            key = f"{self.hf_path(source, filename)}:documents:{max_scan}"
            path = self.cached(key, ".jsonl", create)
            with path.open() as stream:
                for index, line in enumerate(stream):
                    yield filename, index, json.loads(line)["text"]


def records(modality, split, source, cache, max_scan):
    """Yield source identities and original captions; no new annotations or chat roles."""
    if modality == "text":
        for filename, index, text in cache.text(source, max_scan):
            normalized = " ".join(text.split())
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            is_validation = int(digest[:16], 16) / 2**64 < source["validation_fraction"]
            if is_validation != (split == "validation"):
                continue
            yield dict(
                id="dclm-" + digest, group="dclm-" + digest, text=text, origin=f"{filename}:{index}"
            )
    elif modality in {"image", "audio"}:
        for row in cache.parquet(source, source["files"][split]):
            if modality == "image":
                yield dict(
                    id=f"coco-{row['cocoid']}",
                    group=f"coco-{row['cocoid']}",
                    text=row["sentences"][0],
                    origin=row["filename"],
                    url=source["image_base_url"] + row["filepath"] + "/" + row["filename"],
                )
            else:
                yield dict(
                    id="librispeech-" + row["id"],
                    group=f"librispeech-speaker-{row['speaker_id']}",
                    text=row["text"],
                    origin=row["id"],
                    audio=row["audio"]["bytes"],
                )
    else:
        for filename in cache.files(source, source["files"][split]):
            for row in json.loads(cache.file(source, filename).read_text()):
                yield dict(
                    id="msvd-" + row["video_id"],
                    group="msvd-" + row["video_id"].rsplit("_", 2)[0],
                    text=row["caption"][0],
                    origin=row["video_id"],
                    filename=source["video_prefix"] + "/" + row["video"],
                )


def processor_for(recipe, offline=False):
    snapshot = snapshot_download(
        recipe["model_id"],
        revision=recipe["revision"],
        local_files_only=offline,
        allow_patterns=[
            "config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
        ],
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot, local_files_only=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).thinker_config
    return processor, config


def verify_bundle(root):
    root = Path(root).resolve()
    bundle = json.loads((root / "bundle.json").read_text())
    if bundle["status"] != "complete":
        raise ValueError("Data preparation has not completed")
    for line in (root / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    return bundle


@dataclass(init=False, slots=True)
class PrepareOmniDataCfg(SlottedDefault):
    recipe: str
    output: str
    cache: str
    stage: str = "video"
    offline: bool = False


def launch(cfg: PrepareOmniDataCfg):
    recipe = json.loads(Path(cfg.recipe).read_text())
    modalities = recipe["stages"][cfg.stage]
    overrides = recipe.get("modality_sample_limits", {})
    if set(overrides) - set(recipe["sources"]):
        raise ValueError("Sample limits name an unknown modality")
    limits = {
        kind: dict(recipe["samples_per_modality"], **overrides.get(kind, {})) for kind in modalities
    }
    for counts in limits.values():
        if set(counts) != {"train", "validation"} or any(
            type(n) is not int or n <= 0 for n in counts.values()
        ):
            raise ValueError("Train/validation sample counts must be positive integers")
    if recipe["records_per_shard"] <= 0 or recipe["max_scan_per_modality"] <= 0:
        raise ValueError("Shard size and scan limits must be positive")
    effective = dict(recipe=recipe, stage=cfg.stage, format_version=3)
    identity = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    root = Path(cfg.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    recipe_path = root / "recipe.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != effective:
        raise ValueError("Output belongs to another recipe/stage; choose another output directory")
    if (root / "bundle.json").exists() and (root / "checksums.sha256").exists():
        return verify_bundle(root)
    write_json(recipe_path, effective)
    processor, config = processor_for(recipe, cfg.offline)
    collator = Qwen3OmniCollator(processor, config, **recipe["batch"])
    tokenizer_config = hf_hub_download(
        recipe["model_id"],
        "tokenizer_config.json",
        revision=recipe["revision"],
        local_files_only=True,
    )
    Worker(str(Path(tokenizer_config).parent))
    cache = SourceCache(cfg.cache, cfg.offline)
    groups, content_splits, ids = {}, {}, set()
    statistics, manifests, files = {}, {}, {"recipe.json"}
    rejected_path = root / "rejected.jsonl"
    with rejected_path.open("w") as rejected:
        for split in ("train", "validation"):
            manifests[split], statistics[split] = {}, {}
            for modality in modalities:
                source = recipe["sources"][modality]
                accepted, scanned, reasons, shard_rows, shard_paths = 0, 0, Counter(), [], []

                def flush():
                    if not shard_rows:
                        return
                    name = f"{split}/{modality}-{len(shard_paths):05d}.jsonl"
                    atomic_write(
                        root / name,
                        (
                            "".join(
                                json.dumps(row, ensure_ascii=False) + "\n" for row in shard_rows
                            )
                        ).encode(),
                    )
                    if modality == "text":
                        token_name = f"tokens/{split}/{len(shard_paths):05d}.bin"
                        token_path = root / token_name
                        token_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = Writer(token_path)
                        for row in shard_rows:
                            tokens, _ = Worker.encode(row["text"])
                            writer.append(tokens)
                        writer.flush()
                        files.add(token_name)
                    shard_paths.append(name)
                    files.add(name)
                    shard_rows.clear()

                for record in records(
                    modality, split, source, cache, recipe["max_scan_per_modality"]
                ):
                    scanned += 1
                    if scanned > recipe["max_scan_per_modality"]:
                        break
                    try:
                        if record["id"] in ids:
                            raise ValueError("duplicate sample")
                        if groups.get(record["group"], split) != split:
                            raise ValueError("source group overlaps train/validation")
                        if not isinstance(record["text"], str) or not record["text"].strip():
                            raise ValueError("empty paired text")
                        if len(record["text"]) > recipe.get("max_text_chars", 20000):
                            raise ValueError(
                                f"text exceeds {recipe.get('max_text_chars', 20000)}-character preparation limit"
                            )
                        media, media_hash = [], None
                        if modality != "text":
                            if modality == "image":
                                source_path, suffix = cache.http_file(record["url"]), ".jpg"
                            elif modality == "audio":
                                source_path, suffix = None, ".flac"
                            else:
                                source_path = cache.file(source, record["filename"])
                                suffix = Path(record["filename"]).suffix
                            content = (
                                record["audio"] if source_path is None else source_path.read_bytes()
                            )
                            media_hash = hashlib.sha256(content).hexdigest()
                            if media_hash in content_splits:
                                raise ValueError(
                                    "duplicate media content (within or across splits)"
                                )
                            name = f"media/{modality}/{media_hash}{suffix}"
                            if not (root / name).exists() or sha256(root / name) != media_hash:
                                atomic_write(root / name, content)
                            media = [dict(type=modality, path=name)]
                        sample = dict(id=record["id"], text=record["text"].strip(), media=media)
                        runtime = dict(
                            sample, media=[dict(item, path=root / item["path"]) for item in media]
                        )
                        batch = collator([runtime])
                    except (
                        ValueError,
                        UnidentifiedImageError,
                        sf.LibsndfileError,
                        av.error.FFmpegError,
                    ) as exc:
                        reason = str(exc).replace(str(root) + "/", "")
                        reasons[reason] += 1
                        rejected.write(
                            json.dumps(
                                dict(id=record["id"], split=split, modality=modality, reason=reason)
                            )
                            + "\n"
                        )
                        continue
                    sample.update(
                        source=dict(
                            repo_id=source["repo_id"],
                            revision=source["revision"],
                            split=split,
                            record=record["origin"],
                            group=record["group"],
                        ),
                        media_sha256=media_hash,
                        input_tokens=batch.labels.numel(),
                        target_tokens=int((batch.labels != -100).sum()),
                    )
                    shard_rows.append(sample)
                    ids.add(record["id"])
                    groups[record["group"]] = split
                    if media_hash:
                        content_splits[media_hash] = split
                        files.add(media[0]["path"])
                    accepted += 1
                    if len(shard_rows) == recipe["records_per_shard"]:
                        flush()
                    if accepted == limits[modality][split]:
                        break
                flush()
                if accepted != limits[modality][split]:
                    raise ValueError(
                        f"{split}/{modality}: only {accepted}/{limits[modality][split]} usable samples after scanning {scanned}; increase scan/source limits. Rejections: {dict(reasons)}"
                    )
                manifests[split][modality] = shard_paths
                statistics[split][modality] = dict(
                    samples=accepted, scanned=scanned, rejected=dict(reasons)
                )
                print(f"Prepared {split}/{modality}: {accepted} samples", flush=True)
    files.add("rejected.jsonl")
    checksums = {name: sha256(root / name) for name in sorted(files)}
    content_id = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()
    bundle = dict(
        status="complete",
        format_version=3,
        id=identity,
        content_id=content_id,
        stage=cfg.stage,
        recipe=recipe,
        manifests=manifests,
        statistics=statistics,
        prepared_bytes=sum((root / name).stat().st_size for name in files),
        split_policy="Original media splits; DCLM normalized-text hash holdout; shared source groups/content rejected",
        versions={
            name: version(name)
            for name in (
                "transformers",
                "torch",
                "torchvision",
                "pyarrow",
                "av",
                "soundfile",
                "librosa",
            )
        },
        native_training_executed=False,
    )
    write_json(root / "bundle.json", bundle)
    checksums["bundle.json"] = sha256(root / "bundle.json")
    atomic_write(
        root / "checksums.sha256",
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()).encode(),
    )
    return verify_bundle(root)
