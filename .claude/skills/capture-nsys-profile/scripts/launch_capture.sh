#!/bin/bash
# Capture an nsys profile of a short PithTrain run.

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

OUTDIR=workspace/capture-nsys-profile; mkdir -p $OUTDIR
SCRIPT=.claude/skills/capture-nsys-profile/scripts/capture.py

SLURM_NNODES=${SLURM_NNODES:-1}
SLURM_NODEID=${SLURM_NODEID:-0}
SLURM_STEP_NODELIST=${SLURM_STEP_NODELIST:-$(hostname)}
RDZV_HOST=$(command -v scontrol &>/dev/null && scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1 || echo localhost)

NSYS_ARGS=()
NSYS_ARGS+=(profile)
NSYS_ARGS+=(--stats=false)
NSYS_ARGS+=(--trace=cuda,nvtx)
NSYS_ARGS+=(--force-overwrite=true)
NSYS_ARGS+=(--output=$OUTDIR/pithtrain_node${SLURM_NODEID})
NSYS_ARGS+=(--cuda-graph-trace=node)
NSYS_ARGS+=(--capture-range=cudaProfilerApi)
NSYS_ARGS+=(--capture-range-end=stop-shutdown)
NSYS_ARGS+=(--delay=0)

TORCHRUN_ARGS=()
TORCHRUN_ARGS+=(--nnodes=$SLURM_NNODES)
TORCHRUN_ARGS+=(--node-rank=$SLURM_NODEID)
TORCHRUN_ARGS+=(--nproc-per-node=gpu)
TORCHRUN_ARGS+=(--rdzv-backend=c10d)
TORCHRUN_ARGS+=(--rdzv-endpoint=$RDZV_HOST:15213)

nsys ${NSYS_ARGS[@]} torchrun ${TORCHRUN_ARGS[@]} $SCRIPT $@
