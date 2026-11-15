"""
qwen3.5-122b-a10b, 4096 tokens, BF16, 2x8 H200: pp 1, attention dp 16 x cp 1, experts ep 8 x dp 2.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("qwen3.5-122b-a10b", sequence_length=4096, fp8=False, world_size=16)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 1
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 8

if __name__ == "__main__":
    launch(cfg)
