"""
gpt-oss-20b, 4096 tokens, BF16, 1x8 H200: pp 1, attention dp 2 x cp 4, experts ep 2 x dp 4.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("gpt-oss-20b", sequence_length=4096, fp8=False, world_size=8)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 1
distributed.context_parallel_size = 4
distributed.expert_parallel_size = 2

if __name__ == "__main__":
    launch(cfg)
