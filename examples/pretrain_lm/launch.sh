#!/bin/bash
# Pretrain a Mixture-of-Experts (MoE) language model.
#
# Every example pretrains at 4K context, sized for 8x H200 nodes.
#
# Usage (one node: deepseek-v2-lite, gpt-oss-20b, qwen3-30b-a3b, qwen3.5-35b-a3b):
#   bash examples/pretrain_lm/launch.sh qwen3-30b-a3b
#   bash examples/pretrain_lm/launch.sh deepseek-v2-lite
#
# For multi-node training with SLURM (qwen3.5-27b: two nodes, gpt-oss-120b: four):
#   srun -N 2 --ntasks-per-node=1 -W 0 examples/pretrain_lm/launch.sh qwen3.5-27b
#   srun -N 4 --ntasks-per-node=1 -W 0 examples/pretrain_lm/launch.sh gpt-oss-120b

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
