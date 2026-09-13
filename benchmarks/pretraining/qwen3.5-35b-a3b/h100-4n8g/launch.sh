#!/bin/bash
# Benchmark the training of Qwen3.5-35B-A3B with 4x8 H100/H200.
# The workspace is a node-isolated storage that provides fast access.

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/benchmarks/pretraining/qwen3.5-35b-a3b

SRUN_ARGS=()
SRUN_ARGS+=(--nodes=4 --gpus-per-node=8)
SRUN_ARGS+=(--wait=0 --time=00-01:00:00)

STEP=benchmarks/pretraining/qwen3.5-35b-a3b/setup
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.py

STEP=benchmarks/pretraining/qwen3.5-35b-a3b/h100-4n8g/pp4-dp1-cp1-ep8-seq3072-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh

STEP=benchmarks/pretraining/qwen3.5-35b-a3b/h100-4n8g/pp4-dp8-cp1-ep8-seq8192-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
STEP=benchmarks/pretraining/qwen3.5-35b-a3b/h100-4n8g/pp4-dp4-cp2-ep8-seq16384-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
STEP=benchmarks/pretraining/qwen3.5-35b-a3b/h100-4n8g/pp4-dp2-cp4-ep8-seq32768-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
STEP=benchmarks/pretraining/qwen3.5-35b-a3b/h100-4n8g/pp4-dp1-cp8-ep8-seq65536-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
