"""Download Nemotron-CC v1 files and tokenize with the Qwen3 tokenizer."""

import gzip
import os
import shutil
import urllib.request
from pathlib import Path

from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg, launch

BASE_URL = "https://data.commoncrawl.org/"
MANIFEST_URL = BASE_URL + "contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"
MANIFEST_ROOT = "contrib/Nemotron/Nemotron-CC/data-jsonl/"
DEFAULT_PARTITIONS = ("quality=high/kind=actual/kind2=actual",)
ALL_PARTITIONS = (
    "quality=high/kind=actual/kind2=actual",
    "quality=high/kind=synthetic/kind2=distill",
    "quality=high/kind=synthetic/kind2=diverse_qa_pairs",
    "quality=high/kind=synthetic/kind2=extract_knowledge",
    "quality=high/kind=synthetic/kind2=knowledge_list",
    "quality=high/kind=synthetic/kind2=wrap_medium",
    "quality=low/kind=actual/kind2=actual",
    "quality=low/kind=synthetic/kind2=wrap_medium",
    "quality=medium-high/kind=actual/kind2=actual",
    "quality=medium-low/kind=actual/kind2=actual",
    "quality=medium/kind=actual/kind2=actual",
)


def selected_partitions() -> tuple[str, ...]:
    value = os.environ.get("NEMOTRON_CC_PARTITIONS")
    if value is None:
        return DEFAULT_PARTITIONS
    if value.strip().lower() == "all":
        return ALL_PARTITIONS
    partitions = tuple(part.strip().strip("/") for part in value.split(",") if part.strip())
    if not partitions:
        raise ValueError("NEMOTRON_CC_PARTITIONS must include at least one partition.")
    return partitions


def selected_paths() -> list[tuple[str, Path]]:
    partitions = selected_partitions()
    limit = int(os.environ.get("NEMOTRON_CC_MAX_FILES", "8"))
    with urllib.request.urlopen(MANIFEST_URL) as response:
        lines = gzip.decompress(response.read()).decode().splitlines()
    paths = []
    for path in lines:
        if not path.startswith(MANIFEST_ROOT):
            continue
        rel = path.removeprefix(MANIFEST_ROOT)
        if not any(rel.startswith(part + "/") for part in partitions):
            continue
        paths.append((path, Path(rel)))
        if limit > 0 and len(paths) >= limit:
            break
    if not paths:
        requested = ", ".join(partitions)
        raise ValueError(f"No Nemotron-CC v1 files matched partitions: {requested}")
    return paths


def download_raw_files(raw_root: Path) -> None:
    for path, rel in selected_paths():
        dest = raw_root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(BASE_URL + path) as response, open(tmp, "wb") as f:
            shutil.copyfileobj(response, f)
        tmp.replace(dest)


if __name__ == "__main__":
    raw_root = Path("workspace/datasets/nemotron-cc-v1/rawtxt")
    download_raw_files(raw_root)

    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = "Qwen/Qwen3-30B-A3B"
    cfg.source_path = raw_root
    cfg.output_path = Path("workspace/datasets/nemotron-cc-v1/toktxt/qwen3")
    launch(cfg)
