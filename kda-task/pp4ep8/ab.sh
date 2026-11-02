#!/bin/bash
# Login-node orchestrator (light: squeue + srun clients only) of the 4-node PP4-EP8 A/B, base (kda/sm90-base b723719)
# vs feat (haok/kda-sm90 1bc9ee6), clean detached worktrees in workspace/kda-sm90/ab-pp4ep8/{base,feat}.
# Order per model x dtype: base1, feat1, base2 (base2 = base-vs-base noise for step time and loss). Each run is one
# srun step over all 4 nodes of the RUNNING PithTrain task (-N4, 8 GPUs per node, no --overlap). A finished run leaves
# runs/<run>.done; a preempted run is moved aside (*.preempted.<job>) and redone on the next task; a run that fails
# while its task is still RUNNING leaves runs/<run>.failed and stops the orchestrator (exit 3) for a look.
# Usage: ab.sh [skip-job-id]   (runs start only on a task whose job id differs from skip-job-id)
K=/mnt/weka/home/hao.kang/haok-pith-train-kda90/kda-task/pp4ep8
W=/mnt/weka/home/hao.kang/haok-pith-train/workspace/kda-sm90/ab-pp4ep8
SRUN=/mnt/weka/home/hao.kang/haok-pith-train/workspace/srun_pith.sh
SKIP=${1:-}; STEPS=32; PORT=15251
MODELS=${MODELS:-"qwen3.5-35b-a3b qwen3-30b-a3b gpt-oss-20b gpt-oss-120b"}
DTYPES=${DTYPES:-"bf16 fp8"}
running_job() { squeue -j 2394207 -t RUNNING -h -o %A | head -1; }
for MODEL in $MODELS; do for DTYPE in $DTYPES; do for ARM in base1 feat1 base2; do
  RUN=$MODEL-$DTYPE-$ARM; O=$K/runs/$RUN
  [ -f $O.done ] && { echo "$RUN done-before"; continue; }
  while true; do
    while J=$(running_job); [ -z "$J" ] || [ "$J" = "$SKIP" ]; do sleep 150; done
    NODES=$(scontrol show hostnames $(squeue -j $J -h -o %N) | sort); MASTER=$(echo "$NODES" | head -1)
    rm -f $O.node*.log $O.node*.rc
    echo "$RUN start job $J $(echo $NODES | tr '\n' ' ') $(date -u +%FT%TZ)"
    $SRUN --jobid=$J -N4 --ntasks-per-node=1 --gres=gpu:8 --output=$O.node%n.log -- \
      bash $K/node.sh $W/${ARM%[0-9]} $MODEL $DTYPE $STEPS $RUN $MASTER $PORT $O
    rc=$?; [ $rc != 0 ] && sleep 90; NOW=$(running_job)
    RCS=$(cat $O.node*.rc 2>/dev/null | tr '\n' ' ')
    if [ $rc = 0 ] && grep -q "step 000000$STEPS/000000$STEPS" $O.node0.log; then
      echo "job $J nodes $(echo $NODES | tr '\n' ' ') rcs $RCS end $(date -u +%FT%TZ)" > $O.done
      echo "$RUN done rc=0 $(date -u +%FT%TZ)"; break
    fi
    if [ "$NOW" != "$J" ]; then
      for f in $O.node*; do mv $f $f.preempted.$J; done 2>/dev/null
      echo "$RUN preempted on job $J (rc=$rc) $(date -u +%FT%TZ); redo on next task"; continue
    fi
    echo "job $J rc=$rc rcs $RCS $(date -u +%FT%TZ)" > $O.failed
    echo "$RUN FAILED rc=$rc rcs $RCS while job $J still running; stop"; exit 3
  done
done; done; done
echo "ALL DONE $(date -u +%FT%TZ)"
