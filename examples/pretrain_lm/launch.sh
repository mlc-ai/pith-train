#!/bin/bash
# Pretrain a Mixture-of-Experts (MoE) language model.
#
# Usage:
#   bash examples/pretrain_lm/launch.sh qwen3-30b-a3b
#   bash examples/pretrain_lm/launch.sh deepseek-v2-lite
#
# For multi-node training with SLURM:
#   srun -W 0 examples/pretrain_lm/launch.sh qwen3-30b-a3b

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

if [ $# -ne 1 ]; then
    echo "Usage: launch.sh <model>" >&2
    exit 1
fi

# Setup distributed.
TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-localhost}" | head -1):15213)

# Launch the training.
SCRIPT=examples/pretrain_lm/$1/script.py
OUTPUT=logging/pretrain_lm/${1}_node${SLURM_NODEID:-0}.log

mkdir -p $(dirname $OUTPUT) && exec > >(tee $OUTPUT) 2>&1
torchrun ${TORCHRUN_ARGS[@]} $SCRIPT
