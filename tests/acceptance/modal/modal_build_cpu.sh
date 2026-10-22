set -euo pipefail
export DEBIAN_FRONTEND=noninteractive CC=gcc CXX=g++ MAX_JOBS=4
chmod 1777 /tmp
python -m pip install --disable-pip-version-check uv==0.9.4
timeout 120 apt-get update
timeout 120 apt-get install -y --no-install-recommends git
uv pip install --system --no-deps --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130'
sed '/^torch==/d' /tmp/runtime-requirements.txt > /tmp/pypi-requirements.txt
uv pip install --system --no-deps --index-url https://pypi.org/simple \
  -r /tmp/pypi-requirements.txt -r /tmp/media-requirements.txt
DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 uv pip install --system --no-build-isolation \
  'deep-gemm @ git+https://github.com/deepseek-ai/DeepGEMM@559d79fb6994a58b8a15b4b93bf13ccc16edf247'
python -m pip check
python - <<'PY'
import importlib.metadata as m, json, torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
assert torch.__version__ == "2.13.0+cu130"
assert m.version("torchvision") == "0.28.0+cu130"
assert not torch.cuda.is_available(), "Image construction must be CPU only"
print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()},sort_keys=True))
PY
