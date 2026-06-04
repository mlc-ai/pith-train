"""Download one shard of the DCLM corpus and tokenize with the DeepSeek-V2 model."""

from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg, launch

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_type"] = "dataset"
    kwargs["local_dir"] = "workspace/datasets/dclm-baseline/rawtxt"
    kwargs["allow_patterns"] = "global-shard_03_of_10/local-shard_1_of_10/*.jsonl.zst"
    snapshot_download("mlfoundations/dclm-baseline-1.0", **kwargs)

if __name__ == "__main__":
    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = "deepseek-ai/DeepSeek-V2-Lite"
    cfg.source_path = Path("workspace/datasets/dclm-baseline/rawtxt")
    cfg.output_path = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")
    launch(cfg)
