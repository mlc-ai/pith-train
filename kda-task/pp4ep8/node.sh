#!/bin/bash
# One node of a 4-node PP4-EP8 A/B run, inside the CUDA container (srun_pith.sh, one task per node).
# Usage: node.sh <arm-worktree> <model> <bf16|fp8> <steps> <run-id> <master-host> <port> <out-prefix>
# Fresh Triton / Inductor / PithTrain tuning caches per run, node-local; rc written to <out-prefix>.node<N>.rc.
D=$1; MODEL=$2; DTYPE=$3; STEPS=$4; RUN=$5; MASTER=$6; PORT=$7; OUT=$8
N=${SLURM_NODEID:-0}
C=/tmp/pp4ep8-$USER-$RUN
rm -rf $C && mkdir -p $C
cd $D && source .venv/bin/activate
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=$D
export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor PITHTRAIN_TUNING_DIR=$C/tuning
echo "node $N $(hostname) arm $(git -C $D rev-parse --short HEAD) $(date -u +%FT%TZ)"
torchrun --nnodes=4 --node-rank=$N --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=$MASTER:$PORT \
  /mnt/weka/home/hao.kang/haok-pith-train-kda90/kda-task/pp4ep8/ab_run.py $MODEL $DTYPE $STEPS $C/ckpt
rc=$?
echo $rc > $OUT.node$N.rc
rm -rf $C
exit $rc
