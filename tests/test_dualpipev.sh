#!/bin/bash
# Test DualPipeV against a single-device reference.

set -euo pipefail

export WORKSPACE=$(readlink -f ${WORKSPACE:-$PWD/workspace})
export OMP_NUM_THREADS=8

TRUN_ARGS=()
TRUN_ARGS+=(--nnodes=1 --nproc-per-node=8)
TRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=localhost:15213)

MODEL="${1:-examples/pretrain_lm/deepseek-v2-lite/config.json}"

MAIN_ARGS=()
MAIN_ARGS+=(--pp-size 2 --ep-size 2)
MAIN_ARGS+=(--model "$MODEL")

SCRIPT=tests/test_dualpipev.py
TAG=$(basename "$(dirname "$MODEL")")
OUTPUT=$PWD/logging/test_dualpipev_${TAG}.log; mkdir -p $(dirname $OUTPUT)

torchrun ${TRUN_ARGS[@]} $SCRIPT ${MAIN_ARGS[@]} 2>&1 | tee $OUTPUT
