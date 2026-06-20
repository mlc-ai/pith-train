#!/usr/bin/env python3
"""
Prepare the model checkpoint and the pretraining dataset.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.tasks import convert_checkpoint, tokenize_corpus
from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg
from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg

if __name__ == "__main__":
    Path("workspace").resolve().mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "Qwen/Qwen3-30B-A3B-Base"
    kwargs["local_dir"] = "workspace/checkpoints/qwen3-30b-a3b/hf-import"
    snapshot_download(**kwargs, repo_type="model")

if __name__ == "__main__":
    cfg = ConvertCheckpointCfg()
    cfg.operation = "hf2dcp"
    cfg.load_path = Path("workspace/checkpoints/qwen3-30b-a3b/hf-import")
    cfg.save_path = Path("workspace/checkpoints/qwen3-30b-a3b/torch-dcp/step-00000000")
    if not Path(cfg.save_path, ".metadata").exists():
        convert_checkpoint.launch(cfg)

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "mlfoundations/dclm-baseline-1.0"
    kwargs["local_dir"] = "workspace/datasets/dclm-baseline/rawtxt"
    pattern = "global-shard_03_of_10/local-shard_1_of_10/shard_0000000[0-7]_processed.jsonl.zst"
    kwargs["allow_patterns"] = [pattern]
    snapshot_download(**kwargs, repo_type="dataset")

if __name__ == "__main__":
    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = "workspace/checkpoints/qwen3-30b-a3b/hf-import"
    cfg.source_path = Path("workspace/datasets/dclm-baseline/rawtxt")
    cfg.output_path = Path("workspace/datasets/dclm-baseline/toktxt/qwen3")
    n_zst = sum(1 for _ in cfg.source_path.rglob("*.jsonl.zst"))
    n_bin = sum(1 for _ in cfg.output_path.rglob("*.bin"))
    if n_bin < n_zst or any(cfg.output_path.rglob("*.lock")):
        tokenize_corpus.launch(cfg)

from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
from pithtrain.tasks.pretrain_lm import PretrainLMCfg

cfg = PretrainLMCfg()
training = cfg.training
training.model = Path("benchmarks/pretraining/qwen3-30b-a3b/model.json")
training.optimizer = make_muon_optimizer
training.scheduler = make_constant_scheduler
training.lr = 1.0e-6
training.max_steps = 25
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/qwen3")
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.save_location = Path("workspace/checkpoints/qwen3-30b-a3b")
