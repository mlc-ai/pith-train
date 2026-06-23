#!/bin/bash
# Benchmark the training throughput of Qwen3-30B-A3B with 8x8 H100.
# The workspace is a node-isolated storage that provides fast access.

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/benchmarks/pretraining/qwen3-30b-a3b

SRUN_ARGS=()
SRUN_ARGS+=(--nodes=8 --gpus-per-node=8)
SRUN_ARGS+=(--wait=0 --time=00-01:00:00)

STEP=benchmarks/pretraining/qwen3-30b-a3b/setup
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.py

STEP=benchmarks/pretraining/qwen3-30b-a3b/h100-8n8g/pp4-dp2-cp1-ep8-seq4096-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
