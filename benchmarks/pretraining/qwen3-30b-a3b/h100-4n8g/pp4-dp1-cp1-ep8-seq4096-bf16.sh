#!/bin/bash
# Launch the training.

set -euo pipefail

TRUN_ARGS=()
TRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --node-rank=${SLURM_NODEID:-0} --nproc-per-node=gpu)
RDZV_HOST=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-localhost}" | head -1 || echo localhost)
TRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$RDZV_HOST:15213)

torchrun ${TRUN_ARGS[@]} benchmarks/pretraining/qwen3-30b-a3b/h100-4n8g/pp4-dp1-cp1-ep8-seq4096-bf16.py
