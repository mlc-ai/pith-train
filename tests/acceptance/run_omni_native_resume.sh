#!/usr/bin/env bash
# Invoke only from the sole coordinated GPU owner, inside an existing allocation.
# Each mode starts a separate process. Never submits or uploads anything.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash tests/acceptance/run_omni_native_resume.sh PYTHON DATASET NEW_OUTPUT' >&2
  exit 2
fi
python=$(realpath "$1")
data=$(realpath "$2")
out=$(realpath -m "$3")
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ "${PITHTRAIN_RUN_OMNI_GPU_TESTS:-}" == 1 ]] || { echo 'Explicit GPU scheduling opt-in required' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo 'Use a new evidence directory' >&2; exit 2; }
cd "$repo"
export PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
timeout --signal=TERM --kill-after=15s 180s "$python" tests/acceptance/omni_native_resume.py --mode preflight --dataset "$data" --output "$out"
for mode in update produce restore negative; do
  timeout --signal=TERM --kill-after=15s 180s "$python" tests/acceptance/omni_native_resume.py --mode "$mode" --dataset "$data" --output "$out" > "$out/$mode.log" 2>&1
done
echo 'All native unsharded update/restart gates passed; no distributed-resume claim.'
