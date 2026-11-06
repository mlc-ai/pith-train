"""CPU policy contracts; these do not claim CUDA numerical reproducibility."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.modules.reproducibility import (
    configure_deterministic_training,
    runtime_determinism_snapshot,
)
from tests.cpu_helpers import setup_training_fixture as setup_training_fixture

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def restore_policy():
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(algorithms, warn_only=warn)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn


@pytest.mark.parametrize(
    "enabled,warn,benchmark,cudnn",
    [
        (False, False, True, False),
        (True, True, True, False),
        (True, False, False, True),
    ],
)
def test_ordinary_setup_preserves_caller_policy(monkeypatch, enabled, warn, benchmark, cudnn):
    torch.use_deterministic_algorithms(enabled, warn_only=warn)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = cudnn
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "caller-owned-incompatible")
    before = runtime_determinism_snapshot()
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("Ordinary path probed CUDA")
    )
    configure_deterministic_training(False)
    assert runtime_determinism_snapshot() == before


@pytest.mark.parametrize(
    "cuda_available,workspace",
    [
        (False, None),
        (True, ":4096:8"),
        (True, ":16:8"),
    ],
)
def test_strict_policy_enables_errors_and_deterministic_convolutions(
    monkeypatch, cuda_available, workspace
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
    precision = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    configure_deterministic_training(True)
    actual = runtime_determinism_snapshot()
    assert actual["deterministic_algorithms"] is True
    assert actual["deterministic_warn_only"] is False
    assert actual["cudnn_benchmark"] is False
    assert actual["cudnn_deterministic"] is True
    assert actual["cublas_workspace_config"] == workspace
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    ) == precision


@pytest.mark.parametrize("workspace", [None, "", ":4096:2", "4096:8", "caller-owned"])
def test_cuda_workspace_precondition_rejects_without_mutation(monkeypatch, workspace):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    before = runtime_determinism_snapshot()
    with pytest.raises(RuntimeError, match="before starting Python"):
        configure_deterministic_training(True)
    assert runtime_determinism_snapshot() == before


def test_configuration_error_is_not_suppressed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def reject(*args, **kwargs):
        raise RuntimeError("deterministic backend unavailable")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", reject)
    with pytest.raises(RuntimeError, match="deterministic backend unavailable"):
        configure_deterministic_training(True)


@pytest.mark.parametrize("strict", [None, False, True])
@pytest.mark.parametrize("deterministic", [None, False, True])
def test_setup_training_applies_policy_before_constructing_model(
    setup_training, monkeypatch, strict, deterministic
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
    runtime, seen = SimpleNamespace(deterministic_attention=None), []
    cfg, setup = setup_training(
        runtime,
        lambda *_: seen.append(
            (
                torch.are_deterministic_algorithms_enabled(),
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                runtime.deterministic_attention,
                runtime.deterministic_token_layout,
            )
        ),
        configure_deterministic_training,
    )
    if strict is not None:
        cfg.training.strict_resume = strict
    if deterministic is not None:
        cfg.training.deterministic = deterministic
    setup(cfg)
    enabled = bool(strict or deterministic)
    assert seen == [(enabled, not enabled, enabled, enabled, enabled)]


def test_rank_snapshot_is_whitelisted_read_only_and_json_serializable(monkeypatch):
    monkeypatch.setenv("UNRELATED_TEST_SECRET", "must-not-appear")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(training, "deterministic_attention", True)
    monkeypatch.setattr(training, "deterministic_token_layout", True)
    actual = runtime_determinism_snapshot()
    assert set(actual) == {
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
        "float32_matmul_precision",
        "cublas_workspace_config",
        "fa4_deterministic",
        "token_layout_deterministic",
    }
    assert actual["cublas_workspace_config"] == ":4096:8"
    assert actual["fa4_deterministic"] is True
    assert actual["token_layout_deterministic"] is True
    assert "must-not-appear" not in json.dumps(actual)
    assert runtime_determinism_snapshot() == actual


def test_deterministic_config_is_independent_and_opt_in():
    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    config = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "TrainingCfg"
    )
    field = next(
        item
        for item in config.body
        if isinstance(item, ast.AnnAssign) and item.target.id == "deterministic"
    )
    assert isinstance(field.value, ast.Constant) and field.value.value is False
