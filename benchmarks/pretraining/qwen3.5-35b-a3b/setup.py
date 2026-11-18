#!/usr/bin/env python3
"""
Prepare the model checkpoint and the pretraining dataset.

Qwen3.5-35B-A3B is released as a vision-language checkpoint; only the text
tower is loaded (the vision tower and the MTP head are skipped). Setting
training.hf_import_path to it and observing a low first-step LM loss (well
below the ~12 of random init) is the correctness signal for the model + loader.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.tasks import tokenize_corpus
from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg

if __name__ == "__main__":
    Path("workspace").resolve().mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "Qwen/Qwen3.5-35B-A3B"
    kwargs["local_dir"] = "workspace/checkpoints/qwen3.5-35b-a3b/hf-import"
    snapshot_download(**kwargs, repo_type="model")

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "mlfoundations/dclm-baseline-1.0"
    kwargs["local_dir"] = "workspace/datasets/dclm-baseline/rawtxt"
    pattern = "global-shard_03_of_10/local-shard_1_of_10/shard_0000000[0-7]_processed.jsonl.zst"
    kwargs["allow_patterns"] = [pattern]
    snapshot_download(**kwargs, repo_type="dataset")

if __name__ == "__main__":
    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = "workspace/checkpoints/qwen3.5-35b-a3b/hf-import"
    cfg.source_path = Path("workspace/datasets/dclm-baseline/rawtxt")
    cfg.output_path = Path("workspace/datasets/dclm-baseline/toktxt/qwen3.5")
    n_zst = sum(1 for _ in cfg.source_path.rglob("*.jsonl.zst"))
    n_bin = sum(1 for _ in cfg.output_path.rglob("*.bin"))
    if n_bin < n_zst or any(cfg.output_path.rglob("*.lock")):
        tokenize_corpus.launch(cfg)

from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
from pithtrain.tasks.pretrain_lm import PretrainLMCfg

cfg = PretrainLMCfg()

cfg.dataset = Path("workspace/datasets/dclm-baseline/toktxt/qwen3.5")

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-35b-a3b")
training.optimizer = make_muon_optimizer
training.scheduler = make_constant_scheduler
training.lr = 1.0e-6
training.max_steps = 25
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.benchmark = True
