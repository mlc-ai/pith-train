#!/bin/bash
# Run a short PithTrain training run for correctness validation.

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

SCRIPT=.claude/skills/validate-correctness/scripts/validate.py

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --node-rank=${SLURM_NODEID:-0} --nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=${SLURM_LAUNCH_NODE_IPADDR:-localhost}:15213)

torchrun ${TORCHRUN_ARGS[@]} $SCRIPT $@
