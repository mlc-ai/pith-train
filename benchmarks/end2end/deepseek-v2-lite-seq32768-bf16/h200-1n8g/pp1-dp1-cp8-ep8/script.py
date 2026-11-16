"""
deepseek-v2-lite, 32768 tokens, BF16, 1x8 H200: pp 1, attention dp 1 x cp 8, experts ep 8 x dp 1.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("deepseek-v2-lite", sequence_length=32768, fp8=False, world_size=8)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 1
distributed.context_parallel_size = 8
distributed.expert_parallel_size = 8

if __name__ == "__main__":
    launch(cfg)
