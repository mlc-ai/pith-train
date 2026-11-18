#!/usr/bin/env bash
# Offline gates only. This script neither uploads nor requests an allocation.
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_week4_h100.sh PYTHON DATASET NEW_OUTPUT' >&2; exit 2; }
python=$(realpath "$1")
data=$(realpath "$2")
out=$(realpath -m "$3")
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ ${PITHTRAIN_RUN_OMNI_GPU_TESTS:-} == 1 ]] || { echo 'Explicit coordinated GPU opt-in required' >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run inside the approved Flame allocation' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo 'Use a fresh evidence directory' >&2; exit 2; }
cd "$repo"
git diff --quiet HEAD
[[ -z $(git ls-files --others --exclude-standard) ]] || { echo 'Untracked source files are not frozen' >&2; exit 2; }
mkdir -p "$out"
export PYTHONPATH="$repo" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WANDB_MODE=disabled
unset PYTEST_ADDOPTS PYTEST_PLUGINS
git rev-parse HEAD > "$out/commit.txt"
start=$SECONDS
run_gate() {
  local name=$1 cap=$2 remaining
  shift 2
  remaining=$((2400 - SECONDS + start))
  (( remaining > 0 )) || { echo '40-minute budget exhausted; subsequent gates NOT_RUN' >&2; exit 124; }
  (( cap <= remaining )) || cap=$remaining
  printf '%s START %s\n' "$(date -u +%FT%TZ)" "$name" >> "$out/timeline.log"
  if timeout --signal=TERM --kill-after=15s "${cap}s" "$@" > "$out/$name.log" 2>&1; then
    printf '%s PASS %s\n' "$(date -u +%FT%TZ)" "$name" >> "$out/timeline.log"
  else
    local status=$?
    printf '%s FAIL %s exit=%s; subsequent gates NOT_RUN\n' "$(date -u +%FT%TZ)" "$name" "$status" >> "$out/timeline.log"
    exit "$status"
  fi
}
gpu_free() {
  # Recheck allocation-visible devices before every CUDA command. Do not borrow
  # a GPU hidden by Slurm or change CUDA_VISIBLE_DEVICES outside the allocation.
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
  "$python" -c 'import torch; assert torch.cuda.device_count() >= 4
for i in range(4):
 assert "H100" in torch.cuda.get_device_name(i)
 free,total = torch.cuda.mem_get_info(i)
 assert total-free < 1000*1024*1024, f"GPU {i} is busy"'
}
run_gate cpu-tests 180 env CUDA_VISIBLE_DEVICES= "$python" -m pytest \
  tests/test_qwen3_omni_vision.py tests/test_omni_image_contract.py tests/test_omni_image_acceptance_evidence.py tests/test_omni_image_rank_zero.py tests/test_omni_training.py \
  tests/test_qwen3_omni_data.py tests/test_omni_model_registration.py -q --junitxml="$out/cpu.xml"
run_gate cpu-no-skips 15 "$python" -c 'import sys,xml.etree.ElementTree as E
r=E.parse(sys.argv[1]); suites=list(r.iter("testsuite")); assert suites
assert all(int(s.get("tests",0))>0 and all(int(s.get(k,0))==0 for k in ("skipped","failures","errors")) for s in suites)' "$out/cpu.xml"
driver=tests/acceptance/omni_image_training.py
run_gate real-audit 120 env CUDA_VISIBLE_DEVICES= "$python" "$driver" --mode cpu --dataset "$data" --output "$out/audit"
gpu_free > "$out/gpu-before-text.log"
run_gate text-hf-regression 120 "$python" -m pytest tests/test_qwen3_omni_reference.py -q --junitxml="$out/text.xml"
run_gate text-no-skips 15 "$python" -c 'import sys,xml.etree.ElementTree as E
suites=list(E.parse(sys.argv[1]).iter("testsuite")); assert suites
assert all(int(s.get("tests",0))>0 and all(int(s.get(k,0))==0 for k in ("skipped","failures","errors")) for s in suites)' "$out/text.xml"
gpu_free > "$out/gpu-before-reference.log"
run_gate image-hf-reference 120 "$python" "$driver" --mode reference --dataset "$data" --output "$out/reference"
for profile in '1 1 1 nonuniform' '2 1 1 nonuniform' '2 2 1 nonuniform' '2 1 2 nonuniform' '4 2 2 nonuniform' '2 1 1 rank-zero' '2 1 2 rank-zero' '4 2 2 rank-zero'; do
  read -r world pp ep scenario <<< "$profile"
  name=w${world}-pp${pp}-ep${ep}
  if [[ $scenario != nonuniform ]]; then name+=-$scenario; fi
  gpu_free > "$out/gpu-before-$name.log"
  cap=180
  (( world != 1 )) || cap=120
  run_gate "$name" "$cap" "$python" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node="$world" "$driver" --mode staged --dataset "$data" --output "$out/$name" --pp "$pp" --ep "$ep" --scenario "$scenario"
done
run_gate reports 15 "$python" -c 'import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); commit=(r/"commit.txt").read_text().strip()
paths=[(r/"audit/cpu.json",None),(r/"reference/reference.json",None)]
for world,pp,ep in ((1,1,1),(2,1,1),(2,2,1),(2,1,2),(4,2,2)):
 paths += [(r/f"w{world}-pp{pp}-ep{ep}"/f"rank-{rank:05d}.json","nonuniform") for rank in range(world)]
for world,pp,ep in ((2,1,1),(2,1,2),(4,2,2)):
 paths += [(r/f"w{world}-pp{pp}-ep{ep}-rank-zero"/f"rank-{rank:05d}.json","rank-zero") for rank in range(world)]
for path,scenario in paths:
 report=json.loads(path.read_text()); assert report["status"]=="PASSED" and report["commit"]==commit, path
 if scenario:
  assert report["scenario"]==scenario, path
 if scenario=="rank-zero":
  totals=report["target_layout"]["rank_target_totals"]
  assert totals[0]==0 and all(value>0 for value in totals[1:]), path
  assert report["target_count"]==sum(totals)>0, path
  for row in report["target_partitions"]:
   assert row["local_target_count"]==totals[row["dp_rank"]], path
   assert row["global_target_count"]==sum(totals), path
print(f"PASS: {len(paths)} reports. This does not cover full released weights, CP, mixed-modality DP, or image checkpoint conversion.")' "$out"
