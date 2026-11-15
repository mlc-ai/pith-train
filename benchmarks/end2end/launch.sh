#!/bin/bash
# Run end-to-end benchmark cases, one after another, and record one result.json per case.
#
# Usage, from the repo root with .venv active, one srun step per node count:
#   srun -N 1 -W 0 bash benchmarks/end2end/launch.sh                  # the default set (default.txt)
#   srun -N 4 -W 0 bash benchmarks/end2end/launch.sh benchmarks/end2end/qwen3-30b-a3b-seq4096-bf16
#   srun -N 2 -W 0 bash benchmarks/end2end/launch.sh benchmarks/end2end   # the full-grid sweep
#   bash benchmarks/end2end/launch.sh <case-dir>          # one node, no SLURM
#
# With no argument it runs the default set, the cases listed in benchmarks/end2end/default.txt.
# Otherwise each argument is a case directory (one holding script.py) or any directory above
# cases. Either way only the cases whose h200-<nodes>n8g matches the node count of this step run.
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

if [ $# -eq 0 ]; then
    set -- $(grep -v '^\s*\(#\|$\)' benchmarks/end2end/default.txt | sed 's#^#benchmarks/end2end/#')
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
    # A node whose ranks fail leaves abort.<attempt>; the other nodes stop their own ranks on seeing
    # it, rather than wait out the collective timeout in a step that can no longer finish.
    abort=$dir/abort.$((attempt + 1))
    timeout --signal=TERM --kill-after=60 $CASE_TIMEOUT \
        torchrun "${TORCHRUN_ARGS[@]}" "$script" > "$dir/node$NODEID.log" 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
        [ -e "$abort" ] && kill -TERM $pid 2>/dev/null
        sleep 5
    done
    wait $pid
    code=$?
    [ $code -ne 0 ] && touch "$abort"
    if [ "$NODEID" = 0 ]; then
        # Every node's log, so a case that failed on another node records that node's error.
        sleep 10
        python3 -m benchmarks.end2end.summarize "$dir"/node*.log --case "$case" \
            --exit-code $code --commit $COMMIT --dirty $DIRTY \
            --wall-seconds $(($(date +%s) - start)) --output "$result"
        # Bumped only once the case has ended on this node, after every node read it above. A
        # preempted case never gets here, so preemption does not use up its attempts.
        echo $((attempt + 1)) > "$dir/attempts"
    fi
    PORT=$((PORT + 1))
done
