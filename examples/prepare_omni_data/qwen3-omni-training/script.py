"""Prepare, check, publish, or stage a Qwen3-Omni training data bundle."""

import argparse
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

from pithtrain.modules.qwen3_omni_data import create_omni_dataloader
from pithtrain.tasks.prepare_omni_data import (
    PrepareOmniDataCfg,
    launch,
    processor_for,
    sha256,
    verify_bundle,
    write_json,
)


def check(root, report_path):
    bundle = verify_bundle(root)
    processor, config = processor_for(bundle["recipe"], offline=True)
    report = {"content_id": bundle["content_id"], "stages": {}, "native_training_executed": False}
    available = set(bundle["manifests"]["train"])
    for stage, modalities in bundle["recipe"]["stages"].items():
        if not set(modalities) <= available:
            continue
        report["stages"][stage] = {}
        for split in ("train", "validation"):
            kwargs = dict(num_samples=16) if split == "train" else {}
            loader = create_omni_dataloader(
                root, processor, config, stage=stage, split=split, batch_size=2, **kwargs
            )
            batches, targets, sample_ids = 0, 0, []
            for batch in loader:
                if (batch.labels != -100).sum().item() == 0:
                    raise ValueError("A batch has no text targets")
                batches += 1
                targets += int((batch.labels != -100).sum())
                sample_ids.extend(batch.sample_ids)
            report["stages"][stage][split] = dict(
                batches=batches, target_tokens=targets, sample_ids=sample_ids
            )
    write_json(report_path, report)
    return report


def archive_bundle(root, archive):
    root = Path(root)
    verify_bundle(root)
    names = [
        line.split("  ", 1)[1] for line in (root / "checksums.sha256").read_text().splitlines()
    ]
    names.append("checksums.sha256")
    with tarfile.open(archive, "w") as output:
        for name in sorted(names):
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Expected a regular bundle file: {name}")
            info = tarfile.TarInfo(name)
            info.size, info.mode = path.stat().st_size, 0o644
            # Canonical timestamps/ownership make a re-upload byte-identical.
            with path.open("rb") as stream:
                output.addfile(info, stream)


def publish(root, prefix):
    if not prefix.startswith("gs://"):
        raise ValueError("Expected a GCS destination prefix")
    bundle = verify_bundle(root)
    with tempfile.TemporaryDirectory(prefix="omni-publish-", dir=Path(root).parent) as temp:
        archive = Path(temp) / (bundle["content_id"] + ".tar")
        archive_bundle(root, archive)
        checksum = archive.with_suffix(".tar.sha256")
        checksum.write_text(sha256(archive) + "\n")
        uri = prefix.rstrip("/") + "/" + sha256(archive) + ".tar"
        subprocess.run(["gcloud", "storage", "cp", "--no-clobber", str(archive), uri], check=True)
        subprocess.run(
            ["gcloud", "storage", "cp", "--no-clobber", str(checksum), uri + ".sha256"], check=True
        )
    return {"uri": uri, "content_id": bundle["content_id"]}


def stage(uri, destination):
    if not uri.startswith("gs://") or not uri.endswith(".tar"):
        raise ValueError("Expected the published GCS .tar URI")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Use a new staging directory; existing data is never overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="omni-stage-", dir=destination.parent) as temp:
        temp = Path(temp)
        archive, checksum = temp / "bundle.tar", temp / "bundle.tar.sha256"
        subprocess.run(["gcloud", "storage", "cp", uri, str(archive)], check=True)
        subprocess.run(["gcloud", "storage", "cp", uri + ".sha256", str(checksum)], check=True)
        if sha256(archive) != checksum.read_text().strip():
            raise ValueError("Downloaded archive checksum mismatch")
        extracted = temp / "data"
        extracted.mkdir()
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                if not member.isfile() or not (extracted / member.name).resolve().is_relative_to(
                    extracted
                ):
                    raise ValueError("Unexpected archive member")
            source.extractall(extracted, filter="data")
        bundle = verify_bundle(extracted)
        extracted.rename(destination)
    return {"path": str(destination), "content_id": bundle["content_id"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    prepare.add_argument("--stage", choices=["text", "image", "audio", "video"], default="video")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--cache", default="workspace/datasets/omni-source-cache")
    prepare.add_argument("--offline", action="store_true")
    validate = subcommands.add_parser("check")
    validate.add_argument("--data", required=True)
    validate.add_argument("--report", required=True)
    upload = subcommands.add_parser("publish")
    upload.add_argument("--data", required=True)
    upload.add_argument("--prefix", required=True)
    download = subcommands.add_parser("stage")
    download.add_argument("--uri", required=True)
    download.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        cfg = PrepareOmniDataCfg()
        cfg.recipe, cfg.output, cfg.cache = args.config, args.output, args.cache
        cfg.stage, cfg.offline = args.stage, args.offline
        bundle = launch(cfg)
        result = {key: bundle[key] for key in ("content_id", "statistics", "prepared_bytes")}
    elif args.command == "check":
        result = check(args.data, args.report)
    elif args.command == "publish":
        result = publish(args.data, args.prefix)
    else:
        result = stage(args.uri, args.output)
    print(json.dumps(result, indent=2))
