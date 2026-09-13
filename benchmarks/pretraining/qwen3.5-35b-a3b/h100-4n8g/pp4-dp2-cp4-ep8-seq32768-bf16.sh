#!/bin/bash
# Launch the training.

set -euo pipefail

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-localhost}" | head -1):15213)

torchrun ${TORCHRUN_ARGS[@]} benchmarks/pretraining/qwen3.5-35b-a3b/h100-4n8g/pp4-dp2-cp4-ep8-seq32768-bf16.py
