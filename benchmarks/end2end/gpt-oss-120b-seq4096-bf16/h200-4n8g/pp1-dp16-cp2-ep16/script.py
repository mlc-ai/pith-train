"""
gpt-oss-120b, 4096 tokens, BF16, 4x8 H200: pp 1, attention dp 16 x cp 2, experts ep 16 x dp 2.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("gpt-oss-120b", sequence_length=4096, fp8=False, world_size=32)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 1
distributed.context_parallel_size = 2
distributed.expert_parallel_size = 16

if __name__ == "__main__":
    launch(cfg)
