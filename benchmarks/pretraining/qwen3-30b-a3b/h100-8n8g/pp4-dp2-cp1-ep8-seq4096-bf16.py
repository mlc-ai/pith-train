#!/usr/bin/env python3

from setup import cfg

from pithtrain.tasks import pretrain_lm

distributed = cfg.distributed
distributed.pipeline_parallel_size = 4
distributed.expert_parallel_size = 8
distributed.sharding_strategy = "hsdp"

training = cfg.training
training.micro_batch_size = 1
training.global_batch_size = 1024
training.sequence_length = 4096

if __name__ == "__main__":
    pretrain_lm.launch(cfg)
