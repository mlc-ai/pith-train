#!/bin/bash
# Test load_hf_into_model bit-exact against the hf2dcp -> load_checkpoint path.

set -euo pipefail

export WORKSPACE=$(readlink -f ${WORKSPACE:-$PWD/workspace})
export OMP_NUM_THREADS=8

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=1 --nproc-per-node=8)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=localhost:15214)

MODEL="${1:-examples/pretrain_lm/deepseek-v2-lite/config.json}"
HF_CKPT="${HF_CKPT:-$WORKSPACE/checkpoints/deepseek-v2-lite/hf-import}"

# Mesh degrees, overridable from the environment. The default 8-GPU mesh is pp=2 ep=2 cp=1,
# which leaves attn dp=4 and expt dp=2. Set CP_SIZE=2 to exercise the folded layout.
PP_SIZE="${PP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-1}"

MAIN_ARGS=()
MAIN_ARGS+=(--pp-size "$PP_SIZE" --ep-size "$EP_SIZE" --cp-size "$CP_SIZE")
MAIN_ARGS+=(--model "$MODEL" --hf-ckpt "$HF_CKPT")

SCRIPT=tests/test_hf_loader.py
TAG=$(basename "$(dirname "$MODEL")")
OUTPUT=$PWD/logging/test_hf_loader_${TAG}.log; mkdir -p $(dirname $OUTPUT)

torchrun ${TORCHRUN_ARGS[@]} $SCRIPT ${MAIN_ARGS[@]} 2>&1 | tee $OUTPUT
