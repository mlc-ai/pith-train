#!/bin/bash
# Benchmark ring attention. Always captures an nsys profile of the final iteration.
#
# Usage:
#   bash benchmarks/operators/bench_ring_attention.sh qwen3-30b-a3b-cp4-s32k

set -euo pipefail
export OMP_NUM_THREADS=8

SCRIPT=benchmarks/operators/bench_ring_attention.py
OUTDIR=workspace/benchmarks/operators
mkdir -p $OUTDIR

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scenario> (e.g. qwen3-30b-a3b-cp4-s32k)" >&2
    exit 1
fi
SCENARIO=$1

# Extract cp_size from the scenario for torchrun's --nproc-per-node.
if [[ ! $SCENARIO =~ -cp([0-9]+)- ]]; then
    echo "Scenario '$SCENARIO' missing -cp<N>- segment" >&2
    exit 1
fi
NPROC=${BASH_REMATCH[1]}

NSYS_ARGS=()
NSYS_ARGS+=(profile)
NSYS_ARGS+=(--stats=false)
NSYS_ARGS+=(--trace=cuda,osrt,nvtx)
NSYS_ARGS+=(--force-overwrite=true)
NSYS_ARGS+=(--output=$OUTDIR/ring_attention.$SCENARIO)
NSYS_ARGS+=(--cuda-graph-trace=node)
NSYS_ARGS+=(--capture-range=cudaProfilerApi)
NSYS_ARGS+=(--capture-range-end=stop-shutdown)
NSYS_ARGS+=(--delay=0)

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=1)
TORCHRUN_ARGS+=(--nproc-per-node=$NPROC)
TORCHRUN_ARGS+=(--rdzv-backend=c10d)
TORCHRUN_ARGS+=(--rdzv-endpoint=localhost:15213)

nsys ${NSYS_ARGS[@]} torchrun ${TORCHRUN_ARGS[@]} $SCRIPT $SCENARIO
