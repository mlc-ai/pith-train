"""Download Nemotron-CC v2 files and tokenize with the Qwen3 tokenizer."""

import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg, launch

REPO_ID = "nvidia/Nemotron-CC-v2"
DEFAULT_SUBSETS = ("High-Quality",)
ALL_SUBSETS = (
    "Diverse-QA",
    "High-Quality",
    "High-Quality-Synthetic",
    "Medium-High-Quality",
    "Medium-Quality",
    "Translated-Diverse-QA",
)


def selected_subsets() -> tuple[str, ...]:
    value = os.environ.get("NEMOTRON_CC_V2_SUBSETS")
    if value is None:
        return DEFAULT_SUBSETS
    if value.strip().lower() == "all":
        return ALL_SUBSETS
    subsets = tuple(part.strip().strip("/") for part in value.split(",") if part.strip())
    if not subsets:
        raise ValueError("NEMOTRON_CC_V2_SUBSETS must include at least one subset.")
    unknown = sorted(set(subsets) - set(ALL_SUBSETS))
    if unknown:
        raise ValueError("Unknown Nemotron-CC v2 subsets: %s" % ", ".join(unknown))
    return subsets


def selected_paths() -> list[str]:
    limit = int(os.environ.get("NEMOTRON_CC_V2_MAX_FILES_PER_SUBSET", "1"))
    api = HfApi()
    paths = []
    for subset in selected_subsets():
        subset_paths = [
            item.path
            for item in api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=subset)
            if item.path.endswith(".parquet")
        ]
        subset_paths.sort()
        if not subset_paths:
            raise ValueError(f"No Nemotron-CC v2 parquet files found for subset: {subset}")
        if limit > 0:
            subset_paths = subset_paths[:limit]
        paths.extend(subset_paths)
    return paths


def download_raw_files(raw_root: Path) -> None:
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=raw_root,
        allow_patterns=selected_paths(),
    )


if __name__ == "__main__":
    raw_root = Path("workspace/datasets/nemotron-cc-v2/rawtxt")
    download_raw_files(raw_root)

    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = "Qwen/Qwen3-30B-A3B"
    cfg.source_path = raw_root
    cfg.output_path = Path("workspace/datasets/nemotron-cc-v2/toktxt/qwen3")
    launch(cfg)
