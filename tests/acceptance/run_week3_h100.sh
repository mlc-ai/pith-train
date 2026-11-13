#!/usr/bin/env bash
# Prepared next-round command only. Requires the sole owner's existing allocation.
# No uploads, submission, environment installation, or changes to frozen packages.
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_week3_h100.sh PYTHON DATASET NEW_OUTPUT' >&2; exit 2; }
python=$(realpath "$1")
data=$(realpath "$2")
out=$(realpath -m "$3")
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ ${PITHTRAIN_RUN_OMNI_GPU_TESTS:-} == 1 ]] || { echo 'Coordinated GPU opt-in required' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo 'Use a new evidence directory' >&2; exit 2; }
cd "$repo"
git diff --quiet HEAD
mkdir -p "$out"
export PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS PYTEST_PLUGINS
start=$SECONDS
run_gate() {
  local name=$1 cap=$2 remaining
  shift 2
  remaining=$((3300 - SECONDS + start))
  (( remaining > 0 )) || { echo '55-minute budget exhausted; remaining gates NOT RUN' >&2; exit 124; }
  (( cap <= remaining )) || cap=$remaining
  printf '%s START %s cap=%ss\n' "$(date -u +%FT%TZ)" "$name" "$cap" >> "$out/timeline.log"
  if timeout --signal=TERM --kill-after=15s "${cap}s" "$@" > "$out/$name.log" 2>&1; then
    printf '%s PASS %s\n' "$(date -u +%FT%TZ)" "$name" >> "$out/timeline.log"
  else
    local status=$?
    printf '%s FAIL %s exit=%s; subsequent gates NOT RUN\n' "$(date -u +%FT%TZ)" "$name" "$status" >> "$out/timeline.log"
    exit "$status"
  fi
}
run_gate target 30 "$python" -c 'import torch; assert torch.cuda.device_count() >= 4; assert all("H100" in torch.cuda.get_device_name(i) for i in range(4)); print(torch.__version__, torch.cuda.get_device_name(0))'
run_gate converter-native 120 "$python" -m pytest tests/test_qwen3_omni_checkpoint_gpu.py -q -s
run_gate reference 120 "$python" -m pytest tests/test_qwen3_omni_reference.py -q -s
driver=tests/acceptance/omni_distributed_resume.py
converter=examples/convert_checkpoint/qwen3-omni-text/script.py
for profile in '1 1 1' '2 1 1' '2 2 1' '2 1 2' '4 2 2'; do
  read -r world pp ep <<< "$profile"
  name=w${world}-pp${pp}-ep${ep}
  location=$out/$name
  common=(--dataset "$data" --output "$location" --world-size "$world" --pp "$pp" --ep "$ep")
  run_gate "$name-prepare" 120 env CUDA_VISIBLE_DEVICES= "$python" "$driver" --mode prepare "${common[@]}"
  limit=180
  (( world != 1 )) || limit=120
  run_gate "$name-produce" "$limit" "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$world" "$driver" --mode produce "${common[@]}"
  run_gate "$name-export" 120 env CUDA_VISIBLE_DEVICES= "$python" "$converter" dcp2hf "$location/checkpoints/torch-dcp/00000002" "$location/hf"
  run_gate "$name-import" 120 env CUDA_VISIBLE_DEVICES= "$python" "$converter" hf2dcp "$location/hf" "$location/converted/torch-dcp/00000002" --restore-resume
  run_gate "$name-restore" "$limit" "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$world" "$driver" --mode restore "${common[@]}" --restore-root "$location/converted"
done
# Five separate fresh-process, one-rank corruptions on the combined PP2EP2 mesh.
for fault in missing-rank missing-cuda-rng shape topology optimizer-order; do
  run_gate "negative-$fault" 180 "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 "$driver" --mode negative --negative-case "$fault" "${common[@]}" --restore-root "$location/converted"
done
run_gate reports 30 "$python" -c 'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); profiles=((1,1,1),(2,1,1),(2,2,1),(2,1,2),(4,2,2)); count=0
for world,pp,ep in profiles:
    folder=root/f"w{world}-pp{pp}-ep{ep}"
    modes=["produce", "restore"]
    if (world,pp,ep)==(4,2,2): modes += ["negative-"+s for s in ("missing-rank","missing-cuda-rng","shape","topology","optimizer-order")]
    for mode in modes:
        for rank in range(world):
            report=json.loads((folder/f"{mode}-rank-{rank:05d}.json").read_text()); assert report["status"]=="PASSED"; count+=1
print(f"PASS: {count} per-rank reports; 32-step noise regression remains separate")' "$out"
echo 'All selected native/reference and strict-HF-strict continuation gates passed. 32-step regression is NOT covered.'
