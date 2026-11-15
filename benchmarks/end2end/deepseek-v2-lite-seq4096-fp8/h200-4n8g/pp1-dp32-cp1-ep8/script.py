"""
deepseek-v2-lite, 4096 tokens, FP8, 4x8 H200: pp 1, attention dp 32 x cp 1, experts ep 8 x dp 4.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("deepseek-v2-lite", sequence_length=4096, fp8=True, world_size=32)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 1
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 8

if __name__ == "__main__":
    launch(cfg)
