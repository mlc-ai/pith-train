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

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def restore_policy():
    runtime = training.deterministic_attention, training.deterministic_token_layout
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        yield
    finally:
        training.deterministic_attention, training.deterministic_token_layout = runtime
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
    monkeypatch.setattr(training, "deterministic_attention", False)
    monkeypatch.setattr(training, "deterministic_token_layout", False)
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
    assert training.deterministic_attention is True
    assert training.deterministic_token_layout is True
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
    monkeypatch, strict, deterministic
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
    node = next(
        item
        for item in ast.parse((ROOT / "pithtrain/modules/training.py").read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "setup_training"
    )
    cfg = SimpleNamespace(
        training=SimpleNamespace(seed=1234, optimizer=lambda _: (), scheduler=lambda _: ()),
        distributed=SimpleNamespace(),
    )
    if strict is not None:
        cfg.training.strict_resume = strict
    if deterministic is not None:
        cfg.training.deterministic = deterministic
    runtime, seen = training, []
    monkeypatch.setattr(training, "deterministic_attention", None)
    monkeypatch.setattr(training, "deterministic_token_layout", None)
    monkeypatch.setattr(training, "optimizers", None, raising=False)
    monkeypatch.setattr(training, "schedulers", None, raising=False)
    noop_seed = SimpleNamespace(seed=lambda _: None)
    namespace = dict(
        TrainingCfg=SimpleNamespace,
        training=runtime,
        random=noop_seed,
        np=SimpleNamespace(random=noop_seed),
        torch=SimpleNamespace(
            manual_seed=lambda _: None, cuda=SimpleNamespace(manual_seed_all=lambda _: None)
        ),
        configure_deterministic_training=configure_deterministic_training,
        setup_model=lambda *_: seen.append(
            (
                torch.are_deterministic_algorithms_enabled(),
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                runtime.deterministic_attention,
                runtime.deterministic_token_layout,
            )
        ),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<setup_training>", "exec"), namespace)
    namespace["setup_training"](cfg)
    enabled = bool(strict or deterministic)
    assert seen == [(enabled, not enabled, enabled, enabled, enabled)]


def test_rank_snapshot_is_whitelisted_read_only_and_json_serializable(monkeypatch):
    monkeypatch.setenv("UNRELATED_TEST_SECRET", "must-not-appear")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(training, "deterministic_attention", True)
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
    }
    assert actual["cublas_workspace_config"] == ":4096:8"
    assert actual["fa4_deterministic"] is True
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


def test_public_initializer_freezes_model_mode_and_semantic_identity(monkeypatch):
    from pithtrain.modules.resume_semantics import model_contract

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = ROOT / "pithtrain/models/qwen3_omni_moe.py"
    names = {
        "Qwen3OmniMoeThinkerTextExperts",
        "Qwen3OmniMoeThinkerTextTopKRouter",
        "Qwen3OmniMoeThinkerTextSparseMoeBlock",
    }
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name in names
    ]
    namespace = dict(
        __name__=__name__,
        torch=torch,
        nn=torch.nn,
        training=training,
        distributed=SimpleNamespace(ep_size=1),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    block = namespace["Qwen3OmniMoeThinkerTextSparseMoeBlock"]
    cfg = SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=8,
        moe_intermediate_size=8,
        norm_topk_prob=True,
    )
    configure_deterministic_training(False)
    fast = block(cfg)
    configure_deterministic_training(True)
    compatible = block(cfg)
    assert fast.deterministic_token_layout is False
    assert compatible.deterministic_token_layout is True
    left, right = model_contract(fast), model_contract(compatible)
    assert left != right
    assert (
        left["identity"]["modules"][""]["numerical_attributes"]["deterministic_token_layout"]
        is False
    )
    configure_deterministic_training(False)
    assert model_contract(compatible) == right  # No global flip changes an existing model.
