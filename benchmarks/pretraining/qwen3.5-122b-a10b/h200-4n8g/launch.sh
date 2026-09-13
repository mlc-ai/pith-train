#!/bin/bash
# Benchmark the training of Qwen3.5-122B-A10B with 4x8 H200.
# The workspace is a node-isolated storage that provides fast access.
#
# One pipeline stage owns 8 ranks, which the attention view factors as dp=2 x cp=4 and the
# expert view as expt_dp=1 x ep=8. Qwen3.5 is hybrid, so cp shards the sequence two ways,
# zigzag through the attention layers and contiguous through the GatedDeltaNet ones.

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/benchmarks/pretraining/qwen3.5-122b-a10b

SRUN_ARGS=()
SRUN_ARGS+=(--nodes=4 --gpus-per-node=8)
SRUN_ARGS+=(--wait=0 --time=00-01:00:00)

STEP=benchmarks/pretraining/qwen3.5-122b-a10b/setup
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.py

STEP=benchmarks/pretraining/qwen3.5-122b-a10b/h200-4n8g/pp4-dp2-cp4-ep8-seq8192-bf16
srun ${SRUN_ARGS[@]} --output logging/$STEP.log $STEP.sh
