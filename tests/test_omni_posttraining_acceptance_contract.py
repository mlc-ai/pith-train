"""CPU-only native posttraining gate contracts; no GPU acceptance is implied."""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from tests.acceptance import omni_posttraining as gate


@pytest.mark.parametrize("mesh", gate.PROFILES)
def test_complete_groups_pipeline_bounds_and_five_request_four_update_schedule(mesh):
    profile = gate.profile(*mesh)
    assert profile["local_batch"] >= 2 * profile["pp"]
    assert profile["local_batch"] % 2 == 0
    counts = [
        [int(gate.token_mask(request, rank, profile).sum()) for rank in range(profile["dp"])]
        for request in range(gate.REQUESTS)
    ]
    assert sum(sum(values) > 0 for values in counts) == 4
    assert not any(counts[2])
    assert len(set(counts[0])) > 1 or profile["dp"] == 1
    if profile["dp"] > 1:
        assert counts[1][0] == 0 and sum(counts[1][1:]) > 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mesh", [(2, 1, 1), (4, 2, 4), (8, 2, 2), (4, 4, 1), (0, 1, 1)])
def test_unreviewed_mesh_is_rejected(mesh):
    with pytest.raises(ValueError, match="Unreviewed"):
        gate.profile(*mesh)


def test_masks_are_nonuniform_only_completion_targets_and_no_invalid_request():
    topology = gate.profile(4, 2, 2)
    mask = gate.token_mask(0, 1, topology)
    assert not mask[:, : gate.PROMPT - 1].any()
    assert not mask[:, -1].any()
    assert len(set(mask.sum(-1).tolist())) > 1
    for request in (-1, 5, True):
        with pytest.raises(ValueError):
            gate.token_mask(request, 0, topology)


def test_all_gradients_are_reported_after_failure_including_zero_and_small():
    expected = {"bad": torch.ones(2), "zero": torch.zeros(2), "tiny": torch.full((2,), 1e-9)}
    actual = {"bad": -expected["bad"], "zero": expected["zero"], "tiny": -expected["tiny"]}
    report = gate.gradient_evidence(actual, expected, {key: (2,) for key in expected})
    assert report["status"] == "FAIL" and report["checked_count"] == 3
    assert report["parameters"]["zero"]["status"] == "PASS"
    assert report["parameters"]["tiny"]["status"] == "FAIL"
    assert report["threshold"] == 1e-2
    assert all("nse" in row for row in report["parameters"].values())


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "nonfinite"])
def test_gradient_structure_and_nonfinite_never_pass(fault):
    expected, actual = (
        {"a": torch.ones(2), "b": torch.ones(2)},
        {"a": torch.ones(2), "b": torch.ones(2)},
    )
    if fault == "missing":
        del actual["a"]
    elif fault == "shape":
        actual["a"] = torch.ones(1)
    elif fault == "dtype":
        actual["a"] = actual["a"].double()
    else:
        actual["a"][0] = float("nan")
    report = gate.gradient_evidence(actual, expected, {"a": (2,), "b": (2,)})
    assert report["status"] == "FAIL"
    assert report["parameters"]["b"]["status"] == "PASS"


def test_canonical_merge_requires_complete_identical_replicas():
    part = {"a": torch.arange(2.0)}
    assert torch.equal(gate.merge_canonical([part, part], {"a": (2,)})["a"], part["a"])
    with pytest.raises(ValueError, match="Replica"):
        gate.merge_canonical([part, {"a": torch.ones(2)}], {"a": (2,)})
    with pytest.raises(ValueError, match="Missing"):
        gate.merge_canonical([{}], {"a": (2,)})


def test_source_binding_rejects_dirty_or_different_commit(monkeypatch):
    expected = "a" * 40
    answers = {
        ("rev-parse", "HEAD"): expected,
        ("status", "--porcelain"): " M file",
        ("rev-parse", "HEAD^{tree}"): "b" * 40,
    }
    monkeypatch.setattr(gate, "git", lambda *args: answers[args])
    with pytest.raises(ValueError, match="clean"):
        gate.source_identity(expected)
    answers[("status", "--porcelain")] = ""
    assert gate.source_identity(expected)["commit"] == expected
    with pytest.raises(ValueError, match="differs"):
        gate.source_identity("c" * 40)


def test_exact_plan_caps_and_native_routes_are_fixed(tmp_path):
    args = SimpleNamespace(
        task="sft",
        world=4,
        pp=2,
        ep=2,
        expected_commit="a" * 40,
        hardware="NVIDIA H100 80GB HBM3",
        output=tmp_path,
    )
    commands = gate.plan_commands(args)
    assert set(commands) == {"produce", "restore"}
    for mode, command in commands.items():
        assert command[:3] == ["timeout", "--kill-after=10s", "180s"]
        assert command[3:7] == ["torchrun", "--standalone", "--nproc_per_node=4", "-m"]
        assert command[8] == mode and args.expected_commit in command
    assert gate.GRADIENT_NSE == 1e-2 and gate.LOSS_RTOL == gate.LOSS_ATOL == 1e-3
    assert (
        gate.TRANSFORMERS == "5.17.0"
        and gate.REVISION == "26291f793822fb6be9555850f06dfe95f2d7e695"
    )
    source = inspect.getsource(gate.run_gpu)
    assert "sft_lm.train_step" in source and "strict=True" in source
    assert "observer.before_clip()" in source and "original_clip(*positional, **keyword)" in source
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    restores = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_checkpoint"
    ]
    assert (
        len(restores) == 1
        and isinstance(restores[0].args[1], ast.Constant)
        and restores[0].args[1].value == 3
    )
    assert not torch.cuda.is_initialized()


def test_exact_nse_boundary_is_rejected_for_gradient_and_update_report(monkeypatch):
    from tests.acceptance import omni_topology_gradients

    monkeypatch.setattr(omni_topology_gradients, "calculate_difference", lambda left, right: 1e-2)
    report = gate.gradient_evidence({"p": torch.ones(2)}, {"p": torch.ones(2)}, {"p": (2,)})
    assert report["status"] == "FAIL"
    assert report["parameters"]["p"]["status"] == "FAIL"
    assert report["parameters"]["p"]["nse"] == report["threshold"] == 1e-2
