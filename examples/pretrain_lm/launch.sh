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
LAUNCH_ARGS=()
LAUNCH_ARGS+=(--nnodes=${SLURM_NNODES:-1} --node-rank=${SLURM_NODEID:-0} --nproc-per-node=gpu)
LAUNCH_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=${SLURM_LAUNCH_NODE_IPADDR:-localhost}:15213)

# Launch the training.
SCRIPT=examples/pretrain_lm/$1/script.py
OUTPUT=logging/pretrain_lm/${1}_node${SLURM_NODEID:-0}.log

mkdir -p $(dirname $OUTPUT) && exec > >(tee $OUTPUT) 2>&1
torchrun ${LAUNCH_ARGS[@]} $SCRIPT
