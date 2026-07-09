#!/bin/bash
# Benchmark the training throughput of Qwen3-30B-A3B with 2x8 H100.
# The workspace is a node-isolated storage that provides fast access.

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/benchmarks/pretraining/qwen3-30b-a3b

SRUN_ARGS=()
SRUN_ARGS+=(--nodes=2 --gpus-per-node=8)
SRUN_ARGS+=(--wait=0 --time=00-01:00:00)

STEP=benchmarks/pretraining/qwen3-30b-a3b/setup
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.py

STEP=benchmarks/pretraining/qwen3-30b-a3b/h100-2n8g/pp2-dp1-cp1-ep8-seq2048-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
