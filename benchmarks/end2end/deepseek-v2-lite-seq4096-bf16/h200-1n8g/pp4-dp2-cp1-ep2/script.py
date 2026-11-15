"""
deepseek-v2-lite, 4096 tokens, BF16, 1x8 H200: pp 4, attention dp 2 x cp 1, experts ep 2 x dp 1.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("deepseek-v2-lite", sequence_length=4096, fp8=False, world_size=8)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 4
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 2

if __name__ == "__main__":
    launch(cfg)
