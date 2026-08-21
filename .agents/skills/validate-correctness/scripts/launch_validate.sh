#!/bin/bash
# Run a short PithTrain training run for correctness validation.

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

SCRIPT=.agents/skills/validate-correctness/scripts/validate.py

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --node-rank=${SLURM_NODEID:-0} --nproc-per-node=gpu)
RDZV_HOST=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-localhost}" | head -1 || echo localhost)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$RDZV_HOST:15213)

torchrun ${TORCHRUN_ARGS[@]} $SCRIPT $@
