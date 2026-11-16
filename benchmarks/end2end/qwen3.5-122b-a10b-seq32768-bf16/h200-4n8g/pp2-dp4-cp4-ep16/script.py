"""
qwen3.5-122b-a10b, 32768 tokens, BF16, 4x8 H200: pp 2, attention dp 4 x cp 4, experts ep 16 x dp 1.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("qwen3.5-122b-a10b", sequence_length=32768, fp8=False, world_size=32)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 2
distributed.context_parallel_size = 4
distributed.expert_parallel_size = 16

if __name__ == "__main__":
    launch(cfg)
