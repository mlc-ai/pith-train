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

# Base nsys args; scenario branch adds --output.
NSYS_ARGS=()
NSYS_ARGS+=(profile)
NSYS_ARGS+=(--stats=false)
NSYS_ARGS+=(--trace=cuda,osrt,nvtx)
NSYS_ARGS+=(--force-overwrite=true)
NSYS_ARGS+=(--cuda-graph-trace=node)
NSYS_ARGS+=(--capture-range=cudaProfilerApi)
NSYS_ARGS+=(--capture-range-end=stop-shutdown)
NSYS_ARGS+=(--delay=0)

# Base torchrun args; scenario branch adds --nproc-per-node.
TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=1)
TORCHRUN_ARGS+=(--rdzv-backend=c10d)
TORCHRUN_ARGS+=(--rdzv-endpoint=localhost:15213)

case "${1:-}" in
    qwen3-30b-a3b-cp4-s32k)
        NSYS_ARGS+=(--output=$OUTDIR/ring_attention.$1)
        TORCHRUN_ARGS+=(--nproc-per-node=4)
        BENCH_ARGS=(--B 1 --S 32768 --HQ 32 --HK 4 --D 128 --cp-size 4)
        ;;
    *)
        echo "Usage: $0 <scenario>" >&2
        echo "Known scenarios:" >&2
        echo "  qwen3-30b-a3b-cp4-s32k" >&2
        exit 1
        ;;
esac

nsys ${NSYS_ARGS[@]} torchrun ${TORCHRUN_ARGS[@]} $SCRIPT ${BENCH_ARGS[@]}
