"""
gpt-oss-20b, 4096 tokens, FP8, 4x8 H200: pp 4, attention dp 4 x cp 2, experts ep 4 x dp 2.
"""

from benchmarks.end2end.common import end2end_cfg
from pithtrain.tasks.pretrain_lm import launch

cfg = end2end_cfg("gpt-oss-20b", sequence_length=4096, fp8=True, world_size=32)

distributed = cfg.distributed
distributed.pipeline_parallel_size = 4
distributed.context_parallel_size = 2
distributed.expert_parallel_size = 4

if __name__ == "__main__":
    launch(cfg)
