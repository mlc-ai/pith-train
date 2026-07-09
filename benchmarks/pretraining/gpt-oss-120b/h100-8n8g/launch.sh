#!/bin/bash
# Benchmark the training throughput of GPT-OSS-120B with 8x8 H100.
# The workspace is a node-isolated storage that provides fast access.

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/benchmarks/pretraining/gpt-oss-120b

SRUN_ARGS=()
SRUN_ARGS+=(--nodes=8 --gpus-per-node=8)
SRUN_ARGS+=(--wait=0 --time=00-01:00:00)

STEP=benchmarks/pretraining/gpt-oss-120b/setup
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.py

STEP=benchmarks/pretraining/gpt-oss-120b/h100-8n8g/pp8-dp1-cp1-ep8-seq2048-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
