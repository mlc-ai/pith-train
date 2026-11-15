"""
gpt-oss-120b, 4096 tokens, BF16, 1x8 H200: pp 2, attention dp 2 x cp 2, experts ep 2 x dp 2.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("gpt-oss-120b", sequence_length=4096, fp8=False, world_size=8)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 2
distributed.context_parallel_size = 2
distributed.expert_parallel_size = 2

if __name__ == "__main__":
    launch(cfg)
