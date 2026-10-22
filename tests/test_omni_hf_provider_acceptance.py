"""CPU fixture/CLI contracts only; native provider GPU integration is NOT_RUN."""

import importlib.util
import json
from types import SimpleNamespace

import pytest
import torch

from tests.acceptance import omni_hf_provider as gate


def script():
    spec = importlib.util.spec_from_file_location("hf_example_test", gate.SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_prepare_local_hf_tensors_and_cli_plan_no_cuda(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        gate, "source_identity", lambda expected: dict(commit=expected, tree="fixture-only")
    )
    args = SimpleNamespace(
        expected_commit="a" * 40,
        hardware="NVIDIA H100 80GB HBM3",
        output=tmp_path / "reports",
        workspace=tmp_path / "state",
    )
    result = gate.prepare(args)
    assert result["status"] == "PREPARED_GPU_NOT_RUN"
    assert all(
        gate.sha(args.workspace / name) == digest for name, digest in result["files"].items()
    )
    assert all(len(set(scores)) > 1 for scores in result["first_reward_groups"])
    assert not list(args.output.rglob("*.pt")) and not list(args.output.rglob("*.safetensors"))
    from safetensors.torch import load_file

    values = load_file(str(args.workspace / "model/model.safetensors"))
    assert values and all(
        v.dtype == torch.float32 and torch.isfinite(v).all() for v in values.values()
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    script().main(gate.cli_arguments(args.workspace) + ["--plan"])
    plan = json.loads(capsys.readouterr().out)
    assert (plan["world"], plan["pp"], plan["ep"]) == (4, 2, 2)
    assert plan["gpu_validation"] == "NOT_RUN"
    assert not torch.cuda.is_initialized()


def test_producer_restore_contract_and_nonoverwriting_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    producer = script().parse_args(gate.cli_arguments(tmp_path))
    restored = script().parse_args(gate.cli_arguments(tmp_path, True))
    assert producer.requests == restored.requests == 3
    assert producer.updates == restored.updates == 3
    assert producer.checkpoint == restored.checkpoint
    assert not producer.no_save and restored.no_save and restored.resume_request == 2
    assert gate.CAPS == dict(prepare=120, produce=600, restore=300, kill_grace=10)


@pytest.mark.parametrize(
    "extra", [["--lr", "nan"], ["--resume-request", "3"], ["--group-size", "2"]]
)
def test_invalid_native_cli_bound_rejected(tmp_path, monkeypatch, extra):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(SystemExit):
        script().parse_args(gate.cli_arguments(tmp_path) + extra)


def test_plan_requires_actual_exact_clean_source(tmp_path):
    with pytest.raises(ValueError, match="Source commit differs"):
        gate.main(
            [
                "plan",
                "--expected-commit",
                "0" * 40,
                "--hardware",
                "NVIDIA H100 80GB HBM3",
                "--output",
                str(tmp_path / "reports"),
                "--workspace",
                str(tmp_path / "state"),
            ]
        )


def test_report_namespace_cannot_include_heavy_state(tmp_path):
    args = SimpleNamespace(output=tmp_path, workspace=tmp_path / "state")
    with pytest.raises(ValueError, match="disjoint"):
        gate.binding(args)
