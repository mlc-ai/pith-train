"""Download a Nemotron-CC v2 sample and tokenize with the GPT-OSS tokenizer."""

from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg, launch

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_type"] = "dataset"
    kwargs["local_dir"] = "workspace/datasets/nemotron-cc-v2/rawtxt"
    kwargs["allow_patterns"] = "High-Quality/part_0000[0-3]*.parquet"
    snapshot_download("nvidia/Nemotron-CC-v2", **kwargs)

if __name__ == "__main__":
    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = "openai/gpt-oss-20b"
    cfg.source_path = Path("workspace/datasets/nemotron-cc-v2/rawtxt")
    cfg.output_path = Path("workspace/datasets/nemotron-cc-v2/toktxt/gpt-oss")
    launch(cfg)
