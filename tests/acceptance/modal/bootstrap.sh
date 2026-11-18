#!/usr/bin/env bash
set -euo pipefail
: "${OMNI_SOURCE_COMMIT:?The controller must pin the final source commit}"
test "${#OMNI_SOURCE_COMMIT}" -eq 40
test ! -e /work/repo
mkdir -p /work
git clone /tmp/omni-source.bundle /work/repo
git -C /work/repo checkout --detach "$OMNI_SOURCE_COMMIT"
test "$(git -C /work/repo rev-parse HEAD)" = "$OMNI_SOURCE_COMMIT"
test -z "$(git -C /work/repo status --porcelain)"
git -C /work/repo submodule status
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
# Some sandbox runtimes expose GPU compute but not the NVML topology matrix.
# Record the unavailable inventory; actual distributed topology gates stay required.
if nvidia-smi topo -m; then
  echo "NVML topology inventory available"
else
  topology_status=$?
  echo "NVML topology inventory unavailable (exit $topology_status); distributed validation remains required" >&2
fi
python - <<'PY'
import importlib.metadata
import json
import platform
import torch

print(json.dumps({
    "python": platform.python_version(),
    "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
}, sort_keys=True))
assert torch.__version__ == "2.13.0+cu130"
assert importlib.metadata.version("torchvision") == "0.28.0+cu130"
assert torch.cuda.is_available()
PY
