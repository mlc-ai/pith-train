"""Assemble a new local asset directory; never modify the frozen reuse source."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(path, item):
    assert path.is_file() and path.stat().st_size == item["bytes"], str(path)
    assert digest(path) == item["sha256"], str(path)


def relative(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts and "\\" not in name, name
    return path


def assemble(archive, manifest, reuse, output):
    spec = json.loads(manifest.read_text())
    assert not output.exists(), "Choose a new directory"
    assert not output.resolve().is_relative_to(reuse.resolve())
    assert not reuse.resolve().is_relative_to(output.resolve())
    verify(archive, spec["assets_archive"])
    for name, item in spec["reused_data_files"].items():
        verify(reuse / relative(name), item)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        assert len(members) == len(spec["uploaded_files"])
        assert {m.name for m in members} == spec["uploaded_files"].keys()
        assert all(m.isfile() for m in members), "Only regular files are allowed"
        for member in members:
            relative(member.name)
        output.mkdir(parents=True)
        for name in spec["reused_data_files"]:
            target = output / "data" / relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reuse / name, target)
        for member in members:
            target = output / relative(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            verify(target, spec["uploaded_files"][member.name])
    for name, item in spec["reused_data_files"].items():
        verify(output / "data" / name, item)
    print(
        json.dumps(
            dict(
                status="ASSET_ASSEMBLY_PASS",
                files=len(members) + len(spec["reused_data_files"]),
                cuda_used=False,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "manifest", "reuse", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assemble(args.archive, args.manifest, args.reuse, args.output)
