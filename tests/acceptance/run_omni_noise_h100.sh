#!/usr/bin/env bash
# Explicitly scheduled step only: never sbatch, upload, requeue or switch sources.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: run_omni_noise_h100.sh PYTHON PLAN_JSON' >&2; exit 2; }
python=$(realpath "$1")
plan=$(realpath "$2")
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ ${PITHTRAIN_RUN_OMNI_GPU_TESTS:-} == 1 ]] || { echo 'GPU scheduling opt-in required' >&2; exit 2; }
[[ ${SLURM_RESTART_COUNT:-0} == 0 ]] || { echo 'No automatic requeue/retry' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTEST_ADDOPTS PYTEST_PLUGINS
"$python" -c 'import torch; assert torch.cuda.device_count() == 4; assert all("H100" in torch.cuda.get_device_name(i) for i in range(4))'
exec timeout --signal=TERM --kill-after=15s 3330s "$python" "$repo/tests/acceptance/omni_noise_regression.py" run --plan "$plan"
