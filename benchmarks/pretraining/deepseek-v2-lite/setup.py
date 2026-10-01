#!/usr/bin/env python3
"""
Prepare the tokenizer and the pretraining dataset.
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from pithtrain.tasks import tokenize_corpus
from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg

if __name__ == "__main__":
    Path("workspace").resolve().mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
    tokenizer.save_pretrained("workspace/checkpoints/deepseek-v2-lite/hf-import")

if __name__ == "__main__":
    kwargs = dict()
    kwargs["repo_id"] = "mlfoundations/dclm-baseline-1.0"
    kwargs["local_dir"] = "workspace/datasets/dclm-baseline/rawtxt"
    pattern = "global-shard_03_of_10/local-shard_1_of_10/shard_0000000[0-7]_processed.jsonl.zst"
    kwargs["allow_patterns"] = [pattern]
    snapshot_download(**kwargs, repo_type="dataset")

if __name__ == "__main__":
    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = "workspace/checkpoints/deepseek-v2-lite/hf-import"
    cfg.source_path = Path("workspace/datasets/dclm-baseline/rawtxt")
    cfg.output_path = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")
    n_zst = sum(1 for _ in cfg.source_path.rglob("*.jsonl.zst"))
    n_bin = sum(1 for _ in cfg.output_path.rglob("*.bin"))
    if n_bin < n_zst or any(cfg.output_path.rglob("*.lock")):
        tokenize_corpus.launch(cfg)

from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
from pithtrain.tasks.pretrain_lm import PretrainLMCfg

cfg = PretrainLMCfg()

cfg.data.dataset = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")

training = cfg.training
training.model = Path("benchmarks/pretraining/deepseek-v2-lite")
training.optimizer = make_muon_optimizer
training.scheduler = make_constant_scheduler
training.lr = 1.0e-6
training.max_steps = 25
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.benchmark = True
