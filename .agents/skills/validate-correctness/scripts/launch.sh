#!/bin/bash
# Run a short PithTrain training run for correctness validation.

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-localhost}" | head -1):15213)

torchrun ${TORCHRUN_ARGS[@]} $1
