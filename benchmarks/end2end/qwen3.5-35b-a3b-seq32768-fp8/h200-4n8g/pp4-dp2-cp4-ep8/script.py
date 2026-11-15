"""
qwen3.5-35b-a3b, 32768 tokens, FP8, 4x8 H200: pp 4, attention dp 2 x cp 4, experts ep 8 x dp 1.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("qwen3.5-35b-a3b", sequence_length=32768, fp8=True, world_size=32)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 4
distributed.context_parallel_size = 4
distributed.expert_parallel_size = 8

if __name__ == "__main__":
    launch(cfg)
