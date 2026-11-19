from pathlib import Path

from huggingface_hub import snapshot_download

# The released weights, loaded by setting training.hf_import_path to this directory.
local_dir = Path("workspace/checkpoints/qwen3-30b-a3b-base/hf-import")

if __name__ == "__main__":
    snapshot_download(repo_id="Qwen/Qwen3-30B-A3B-Base", local_dir=local_dir)
