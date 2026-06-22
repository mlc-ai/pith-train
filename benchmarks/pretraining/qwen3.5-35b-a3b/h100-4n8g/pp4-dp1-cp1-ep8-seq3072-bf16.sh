#!/bin/bash
# Launch the training.

TRUN_ARGS=()
TRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --node-rank=${SLURM_NODEID:-0} --nproc-per-node=gpu)
TRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=${SLURM_LAUNCH_NODE_IPADDR:-localhost}:15213)

torchrun ${TRUN_ARGS[@]} benchmarks/pretraining/qwen3.5-35b-a3b/h100-4n8g/pp4-dp1-cp1-ep8-seq3072-bf16.py
