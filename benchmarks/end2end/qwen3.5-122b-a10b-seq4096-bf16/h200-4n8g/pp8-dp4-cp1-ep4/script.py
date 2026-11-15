"""
qwen3.5-122b-a10b, 4096 tokens, BF16, 4x8 H200: pp 8, attention dp 4 x cp 1, experts ep 4 x dp 1.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("qwen3.5-122b-a10b", sequence_length=4096, fp8=False, world_size=32)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 8
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 4

if __name__ == "__main__":
    launch(cfg)
