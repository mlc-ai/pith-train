#!/bin/bash
# Launch the training.

set -euo pipefail

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-localhost}" | head -1):15213)

torchrun ${TORCHRUN_ARGS[@]} benchmarks/pretraining/gpt-oss-120b/h100-8n8g/pp8-dp1-cp1-ep8-seq2048-bf16.py
