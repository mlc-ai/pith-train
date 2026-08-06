import gzip
import importlib.util
import io
from pathlib import Path

import pytest


def _load_script(name: str):
    path = Path(__file__).parents[1] / "examples/tokenize_corpus" / name / "script.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nemotron_manifest_filtering(monkeypatch):
    module = _load_script("nemotron-cc-v1-qwen3")
    manifest = "\n".join(
        [
            module.MANIFEST_ROOT
            + "quality=high/kind=actual/kind2=actual/CC-MAIN-part-00000.jsonl.zstd",
            module.MANIFEST_ROOT
            + "quality=high/kind=synthetic/kind2=distill/CC-MAIN-part-00001.jsonl.zstd",
            module.MANIFEST_ROOT
            + "quality=medium/kind=actual/kind2=actual/CC-MAIN-part-00002.jsonl.zstd",
        ]
    )
    payload = gzip.compress(manifest.encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: io.BytesIO(payload))
    monkeypatch.setenv(
        "NEMOTRON_CC_PARTITIONS",
        "quality=high/kind=synthetic/kind2=distill,quality=medium/kind=actual/kind2=actual",
    )
    monkeypatch.setenv("NEMOTRON_CC_MAX_FILES", "0")

    paths = module.selected_paths()

    assert [rel.as_posix() for _, rel in paths] == [
        "quality=high/kind=synthetic/kind2=distill/CC-MAIN-part-00001.jsonl.zstd",
        "quality=medium/kind=actual/kind2=actual/CC-MAIN-part-00002.jsonl.zstd",
    ]


def test_nemotron_v1_manifest_filtering_rejects_empty_matches(monkeypatch):
    module = _load_script("nemotron-cc-v1-qwen3")
    manifest = module.MANIFEST_ROOT + "quality=medium/kind=actual/kind2=actual/file.jsonl.zstd"
    payload = gzip.compress(manifest.encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: io.BytesIO(payload))
    monkeypatch.setenv("NEMOTRON_CC_PARTITIONS", "quality=high/kind=actual/kind2=actual")

    with pytest.raises(ValueError, match="No Nemotron-CC v1 files matched"):
        module.selected_paths()


def test_nemotron_v1_download_preserves_partition_path(monkeypatch, tmp_path: Path):
    module = _load_script("nemotron-cc-v1-qwen3")
    rel = Path("quality=high/kind=actual/kind2=actual/part.jsonl.zstd")
    manifest_path = module.MANIFEST_ROOT + rel.as_posix()
    calls = []

    monkeypatch.setattr(module, "selected_paths", lambda: [(manifest_path, rel)])

    def urlopen(url: str):
        calls.append(url)
        return io.BytesIO(b"shard")

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)

    module.download_raw_files(tmp_path)
    module.download_raw_files(tmp_path)

    assert (tmp_path / rel).read_bytes() == b"shard"
    assert calls == [module.BASE_URL + manifest_path]


def test_nemotron_v2_path_selection_and_download(monkeypatch, tmp_path: Path):
    module = _load_script("nemotron-cc-v2-qwen3")

    class Item:
        def __init__(self, path: str):
            self.path = path

    class FakeApi:
        def list_repo_tree(self, repo_id: str, repo_type: str, path_in_repo: str):
            assert repo_id == module.REPO_ID
            assert repo_type == "dataset"
            return [
                Item(f"{path_in_repo}/part_000002.parquet"),
                Item(f"{path_in_repo}/README.md"),
                Item(f"{path_in_repo}/part_000000.parquet"),
                Item(f"{path_in_repo}/part_000001.parquet"),
            ]

    calls = {}

    def fake_snapshot_download(repo_id: str, **kwargs):
        calls["repo_id"] = repo_id
        calls.update(kwargs)

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("NEMOTRON_CC_V2_SUBSETS", "High-Quality,Translated-Diverse-QA")
    monkeypatch.setenv("NEMOTRON_CC_V2_MAX_FILES_PER_SUBSET", "2")

    paths = module.selected_paths()

    assert paths == [
        "High-Quality/part_000000.parquet",
        "High-Quality/part_000001.parquet",
        "Translated-Diverse-QA/part_000000.parquet",
        "Translated-Diverse-QA/part_000001.parquet",
    ]

    module.download_raw_files(tmp_path)

    assert calls["repo_id"] == module.REPO_ID
    assert calls["repo_type"] == "dataset"
    assert calls["local_dir"] == tmp_path
    assert calls["allow_patterns"] == paths


def test_nemotron_v2_rejects_unknown_subset(monkeypatch):
    module = _load_script("nemotron-cc-v2-qwen3")

    monkeypatch.setenv("NEMOTRON_CC_V2_SUBSETS", "High-Quality,Nope")

    with pytest.raises(ValueError, match="Unknown Nemotron-CC v2 subsets"):
        module.selected_subsets()
