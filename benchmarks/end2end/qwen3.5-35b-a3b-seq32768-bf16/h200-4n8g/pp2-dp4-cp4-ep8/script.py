"""
qwen3.5-35b-a3b, 32768 tokens, BF16, 4x8 H200: pp 2, attention dp 4 x cp 4, experts ep 8 x dp 2.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("qwen3.5-35b-a3b", sequence_length=32768, fp8=False, world_size=32)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 2
distributed.context_parallel_size = 4
distributed.expert_parallel_size = 8

if __name__ == "__main__":
    launch(cfg)
