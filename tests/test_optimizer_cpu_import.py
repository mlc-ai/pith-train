"""The CPU optimizer API must not import the GPU training graph."""

import os
import subprocess
import sys
from pathlib import Path


def test_optimizer_classifier_import_does_not_load_gpu_training_graph():
    repo = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys
import torch
for prefix in ('pithtrain.modules.training', 'pithtrain.models', 'pithtrain.pipeline', 'pithtrain.operators'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
class RejectGPUTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in (
            'pithtrain.modules.training', 'pithtrain.models', 'pithtrain.pipeline',
            'pithtrain.operators', 'deep_gemm', 'triton',
        )):
            raise AssertionError('CPU optimizer attempted unrelated import: ' + fullname)
sys.meta_path.insert(0, RejectGPUTrainingImports())
from pithtrain.modules.optimizer import Muon, is_muon_param
assert callable(is_muon_param) and issubclass(Muon, torch.optim.Optimizer)
assert not torch.cuda.is_initialized()
"""
    env = dict(
        os.environ, PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
