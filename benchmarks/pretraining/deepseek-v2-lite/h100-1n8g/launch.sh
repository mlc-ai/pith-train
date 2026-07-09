#!/bin/bash
# Benchmark the training throughput of DeepSeek-V2-Lite with 1x8 H100.
# The workspace is a node-isolated storage that provides fast access.

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/benchmarks/pretraining/deepseek-v2-lite

SRUN_ARGS=()
SRUN_ARGS+=(--nodes=1 --gpus-per-node=8)
SRUN_ARGS+=(--wait=0 --time=00-01:00:00)

STEP=benchmarks/pretraining/deepseek-v2-lite/setup
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.py

STEP=benchmarks/pretraining/deepseek-v2-lite/h100-1n8g/pp1-dp1-cp1-ep8-seq2048-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
