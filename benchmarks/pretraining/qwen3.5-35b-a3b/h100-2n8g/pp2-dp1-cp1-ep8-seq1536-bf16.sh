#!/bin/bash
# Launch the training.

set -euo pipefail

TRUN_ARGS=()
TRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --node-rank=${SLURM_NODEID:-0} --nproc-per-node=gpu)
RDZV_HOST=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-localhost}" | head -1 || echo localhost)
TRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$RDZV_HOST:15213)

torchrun ${TRUN_ARGS[@]} benchmarks/pretraining/qwen3.5-35b-a3b/h100-2n8g/pp2-dp1-cp1-ep8-seq1536-bf16.py
