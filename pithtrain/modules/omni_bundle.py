"""Bundle identity and verification shared by preparation and training."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_id(checksums):
    return hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()


def read_verified_bundle(root):
    """Return metadata and the hash of those same verified bytes for checkpoint identity."""
    root = Path(root).resolve()
    bundle_bytes = (root / "bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    if bundle["status"] != "complete":
        raise ValueError("Data preparation has not completed")
    checksums = {}
    for line in (root / "checksums.sha256").read_text().splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise ValueError("Invalid dataset checksum inventory")
        expected, name = fields
        path = (root / name).resolve()
        if (
            name in checksums
            or not name
            or Path(name).as_posix() != name
            or not path.is_relative_to(root)
            or path.relative_to(root).as_posix() != name
        ):
            raise ValueError(f"Invalid dataset checksum inventory entry: {name}")
        actual = bundle_sha256 if name == "bundle.json" else sha256(path)
        if actual != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
        checksums[name] = actual
    required = {"bundle.json", "recipe.json", "rejected.jsonl"}
    token_files = set()
    for split, modalities in bundle["manifests"].items():
        for kind, manifests in modalities.items():
            for index, name in enumerate(manifests):
                if name not in checksums:
                    raise ValueError(f"Missing dataset checksum inventory entry: {name}")
                required.add(name)
                with (root / name).open() as stream:
                    for line in stream:
                        if line.strip():
                            required.update(item["path"] for item in json.loads(line)["media"])
                if kind == "text":
                    token_files.add(f"tokens/{split}/{index:05d}.bin")
    # TokenBinData recursively consumes every .bin, so an unlisted extra
    # shard must fail even when all the listed hashes still match.
    actual_tokens = {path.relative_to(root).as_posix() for path in (root / "tokens").rglob("*.bin")}
    if actual_tokens != token_files or set(checksums) != required | token_files:
        raise ValueError("Dataset checksum inventory differs from the prepared files")
    checksums.pop("bundle.json")
    if bundle.get("content_id") != _content_id(checksums):
        raise ValueError("Dataset content_id does not match verified content")
    return bundle, bundle_sha256


def verify_bundle(root):
    return read_verified_bundle(root)[0]
