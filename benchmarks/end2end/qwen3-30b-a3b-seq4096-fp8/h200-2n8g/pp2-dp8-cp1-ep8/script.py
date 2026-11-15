"""
qwen3-30b-a3b, 4096 tokens, FP8, 2x8 H200: pp 2, attention dp 8 x cp 1, experts ep 8 x dp 1.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("qwen3-30b-a3b", sequence_length=4096, fp8=True, world_size=16)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 2
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 8

if __name__ == "__main__":
    launch(cfg)
