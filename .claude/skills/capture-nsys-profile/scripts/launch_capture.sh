#!/bin/bash
# Capture an nsys profile of a short PithTrain run.

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

OUTDIR=workspace/capture-nsys-profile; mkdir -p $OUTDIR
SCRIPT=.claude/skills/capture-nsys-profile/scripts/capture.py

NSYS_ARGS=()
NSYS_ARGS+=(profile)
NSYS_ARGS+=(--stats=false)
NSYS_ARGS+=(--trace=cuda,nvtx)
NSYS_ARGS+=(--force-overwrite=true)
NSYS_ARGS+=(--output=$OUTDIR/pithtrain_node${SLURM_NODEID:-0})
NSYS_ARGS+=(--cuda-graph-trace=node)
NSYS_ARGS+=(--capture-range=cudaProfilerApi)
NSYS_ARGS+=(--capture-range-end=stop-shutdown)
NSYS_ARGS+=(--delay=0)

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=${SLURM_NNODES:-1} --node-rank=${SLURM_NODEID:-0} --nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=${SLURM_LAUNCH_NODE_IPADDR:-localhost}:15213)

nsys ${NSYS_ARGS[@]} torchrun ${TORCHRUN_ARGS[@]} $SCRIPT $@
