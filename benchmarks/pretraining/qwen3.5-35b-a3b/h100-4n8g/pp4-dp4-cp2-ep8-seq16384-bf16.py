#!/usr/bin/env python3

from setup import cfg

from pithtrain.tasks import pretrain_lm

distributed = cfg.distributed
distributed.pipeline_parallel_size = 4
distributed.expert_parallel_size = 8
distributed.context_parallel_size = 2

training = cfg.training
training.micro_batch_size = 1
training.global_batch_size = 128
training.sequence_length = 16384

if __name__ == "__main__":
    pretrain_lm.launch(cfg)
