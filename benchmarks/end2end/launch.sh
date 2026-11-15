#!/bin/bash
# Run end-to-end benchmark cases, one after another, and record one result.json per case.
#
# Usage, from the repo root with .venv active, one srun step per node count:
#   srun -N 1 -W 0 bash benchmarks/end2end/launch.sh benchmarks/end2end
#   srun -N 4 -W 0 bash benchmarks/end2end/launch.sh benchmarks/end2end/qwen3-30b-a3b-seq4096-bf16
#   bash benchmarks/end2end/launch.sh <case-dir>          # one node, no SLURM
#
# Each argument is a case directory (one holding script.py) or any directory above cases. Only the
# cases whose h200-<nodes>n8g matches the node count of this step run; the rest are skipped.
#
# Results land in $BENCH_OUTPUT/<case>/{result.json,node<k>.log}, by default
# workspace/benchmarks/end2end/<commit>. A case with a result.json is skipped, so a preempted
# set resumes where it stopped when rerun; a failed case is retried once (BENCH_ATTEMPTS=2). A case
# the model rejects with NotImplementedError is recorded as unsupported and not retried.
# Compare two output directories with: python3 -m benchmarks.end2end.compare <base> <feat>

set -uo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

if [ $# -lt 1 ]; then
    sed -n '2,16p' "$0" >&2
    exit 1
fi

NNODES=${SLURM_NNODES:-1}
NODEID=${SLURM_NODEID:-0}
GPUS=$(nvidia-smi -L | wc -l)
COMMIT=$(git rev-parse --short=12 HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | grep -q . && echo 1 || echo 0)
OUTPUT=${BENCH_OUTPUT:-workspace/benchmarks/end2end/$COMMIT}
ATTEMPTS=${BENCH_ATTEMPTS:-2}
CASE_TIMEOUT=${BENCH_CASE_TIMEOUT:-45m}

# Compile caches on node-local disk, so no step of a case waits on shared storage.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/$USER/triton}

# One rendezvous port per step, so concurrent steps on one node do not collide.
PORT=$((20000 + ${SLURM_STEP_ID:-0} % 20000))
MASTER=$(scontrol show hostnames "${SLURM_STEP_NODELIST:-localhost}" 2>/dev/null | head -1)
MASTER=${MASTER:-localhost}

CASES=$(for root in "$@"; do find "$root" -name script.py -path "*/h200-${NNODES}n${GPUS}g/*"; done | sort -u)
if [ -z "$CASES" ]; then
    echo "no case under $* matches h200-${NNODES}n${GPUS}g" >&2
    exit 1
fi

for script in $CASES; do
    case=$(dirname "${script#benchmarks/end2end/}")
    dir=$OUTPUT/$case
    result=$dir/result.json
    if [ -f "$result" ] && grep -q '"status": "\(ok\|unsupported\)"' "$result"; then
        echo "skip $case: done"
        continue
    fi
    attempt=$(cat "$dir/attempts" 2>/dev/null || echo 0)
    if [ "$attempt" -ge "$ATTEMPTS" ]; then
        echo "skip $case: failed $attempt times"
        continue
    fi
    mkdir -p "$dir"
    echo "run $case (attempt $((attempt + 1)))"
    TORCHRUN_ARGS=(--nnodes=$NNODES --nproc-per-node=gpu)
    TORCHRUN_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=$MASTER:$PORT --rdzv-id=$PORT)
    start=$(date +%s)
    timeout --signal=TERM --kill-after=60 $CASE_TIMEOUT \
        torchrun "${TORCHRUN_ARGS[@]}" "$script" > "$dir/node$NODEID.log" 2>&1
    code=$?
    if [ "$NODEID" = 0 ]; then
        python3 -m benchmarks.end2end.summarize "$dir/node0.log" --case "$case" \
            --exit-code $code --commit $COMMIT --dirty $DIRTY \
            --wall-seconds $(($(date +%s) - start)) --output "$result"
        # Bumped only once the case has ended on this node, after every node read it above. A
        # preempted case never gets here, so preemption does not use up its attempts.
        echo $((attempt + 1)) > "$dir/attempts"
    fi
    PORT=$((PORT + 1))
done
