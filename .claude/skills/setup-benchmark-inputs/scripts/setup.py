"""Setup minimal benchmark inputs: corpus shard, tokenize, HF checkpoint, HF->DCP."""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from pithtrain.tasks import convert_checkpoint, tokenize_corpus
from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg
from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg

CORPUS_RAWTXT = "workspace/datasets/dclm-baseline/rawtxt"

MODELS = {
    "deepseek-v2-lite": {
        "model_id": "deepseek-ai/DeepSeek-V2-Lite",
        "corpus_toktxt": "workspace/datasets/dclm-baseline/toktxt/deepseek-v2",
        "hf_import": "workspace/checkpoints/deepseek-v2-lite/hf-import",
        "torch_dcp": "workspace/checkpoints/deepseek-v2-lite/torch-dcp/step-00000000",
    },
    "qwen3-30b-a3b": {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "corpus_toktxt": "workspace/datasets/dclm-baseline/toktxt/qwen3",
        "hf_import": "workspace/checkpoints/qwen3-30b-a3b/hf-import",
        "torch_dcp": "workspace/checkpoints/qwen3-30b-a3b/torch-dcp/step-00000000",
    },
}


def fetch_corpus():
    corpus_file = "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"
    shard = Path(CORPUS_RAWTXT, corpus_file)
    if shard.exists():
        print(f"[skip] corpus shard already at {shard}")
        return
    corpus_id = "mlfoundations/dclm-baseline-1.0"
    hf_hub_download(corpus_id, corpus_file, repo_type="dataset", local_dir=CORPUS_RAWTXT)


def build_toktxt(specs):
    toktxt = Path(specs["corpus_toktxt"])
    if toktxt.exists() and any(toktxt.glob("*.bin")):
        print(f"[skip] corpus already tokenized at {toktxt}")
        return
    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = specs["model_id"]
    cfg.source_path = Path(CORPUS_RAWTXT)
    cfg.output_path = toktxt
    cfg.num_workers = min(os.cpu_count() or 1, 24)
    tokenize_corpus.launch(cfg)


def fetch_checkpoint(specs):
    hf_import = Path(specs["hf_import"])
    if hf_import.exists() and any(hf_import.glob("*.safetensors")):
        print(f"[skip] HF checkpoint already at {hf_import}")
        return
    snapshot_download(specs["model_id"], local_dir=hf_import)


def import_checkpoint(specs):
    torch_dcp = Path(specs["torch_dcp"])
    if Path(torch_dcp, ".metadata").exists():
        print(f"[skip] DCP checkpoint already at {torch_dcp}")
        return
    cfg = ConvertCheckpointCfg()
    cfg.operation = "hf2dcp"
    cfg.load_path = Path(specs["hf_import"])
    cfg.save_path = torch_dcp
    convert_checkpoint.launch(cfg)


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODELS))
parsed = parser.parse_args()
specs = MODELS[parsed.model]

if __name__ == "__main__":
    fetch_corpus()
    build_toktxt(specs)
    fetch_checkpoint(specs)
    import_checkpoint(specs)
