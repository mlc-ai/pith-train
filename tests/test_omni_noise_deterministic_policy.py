"""CPU rejection contracts; synthetic reports do not establish GPU correctness."""

import json
from types import SimpleNamespace

import pytest

from tests.acceptance import omni_noise_regression as noise
from tests.test_omni_noise_regression import write_arm


def policy(**overrides):
    runtime = dict(
        deterministic_algorithms=True,
        deterministic_warn_only=False,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        cudnn_allow_tf32=True,
        cuda_matmul_allow_tf32=False,
        float32_matmul_precision="highest",
        cublas_workspace_config=":4096:8",
        fa4_deterministic=True,
    )
    runtime.update(overrides)
    return dict(
        numerical_policy=noise.COMMON_NUMERICAL_POLICY,
        runtime_determinism=runtime,
    )


@pytest.fixture
def common():
    assert noise.NUMERICAL_POLICY == noise.LEGACY_NUMERICAL_POLICY
    noise.configure_numerical_policy(noise.COMMON_NUMERICAL_POLICY)
    try:
        yield
    finally:
        noise.configure_numerical_policy()


def config(has_field=True):
    return SimpleNamespace(
        distributed=SimpleNamespace(),
        data=SimpleNamespace(),
        logging=SimpleNamespace(),
        training=SimpleNamespace(
            strict_resume=False, **({"deterministic": False} if has_field else {})
        ),
    )


def test_legacy_entry_does_not_require_new_public_field(tmp_path):
    cfg = config(False)
    noise.configure_cfg(
        cfg, {"dataset": tmp_path, "model": tmp_path}, (1, 1, 1), lambda: None, lambda: None
    )
    assert not hasattr(cfg.training, "deterministic")
    assert "numerical_policy" not in noise.SETTINGS
    noise.validate_runtime_policy({})


@pytest.mark.parametrize("checkpoint", [False, True])
def test_common_config_enables_same_public_policy_without_changing_intervention(
    common, tmp_path, checkpoint
):
    cfg = config()
    noise.configure_cfg(
        cfg,
        {"dataset": tmp_path, "model": tmp_path},
        (1, 1, 1),
        lambda: None,
        lambda: None,
        tmp_path if checkpoint else None,
    )
    assert cfg.training.deterministic is True
    assert cfg.training.strict_resume is checkpoint
    assert cfg.training.save_interval == (16 if checkpoint else None)
    assert cfg.training.max_steps == 32
    assert cfg.training.benchmark is False


def test_common_config_rejects_missing_public_policy(common, tmp_path):
    with pytest.raises(ValueError, match="Common deterministic config"):
        noise.configure_cfg(
            config(False),
            {"dataset": tmp_path, "model": tmp_path},
            (1, 1, 1),
            lambda: None,
            lambda: None,
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"deterministic_algorithms": False},
        {"deterministic_warn_only": True},
        {"cudnn_benchmark": True},
        {"cudnn_deterministic": False},
        {"fa4_deterministic": False},
        {"cublas_workspace_config": None},
        {"cublas_workspace_config": ":4096:2"},
        {"cuda_matmul_allow_tf32": 0},
    ],
)
def test_actual_policy_must_be_enabled_and_well_formed(common, changed):
    with pytest.raises(ValueError):
        noise.validate_runtime_policy(policy(**changed))


def test_policy_must_be_complete_and_whitelisted(common):
    for mutation in ("missing", "extra", "wrong-version"):
        report = policy()
        if mutation == "missing":
            del report["runtime_determinism"]["cudnn_allow_tf32"]
        elif mutation == "extra":
            report["runtime_determinism"]["unrelated_environment"] = "not-allowed"
        else:
            report["numerical_policy"] = "legacy-v1"
        with pytest.raises(ValueError):
            noise.validate_runtime_policy(report)


@pytest.mark.parametrize(
    "changed",
    [
        {"cudnn_allow_tf32": False},
        {"cuda_matmul_allow_tf32": True},
        {"float32_matmul_precision": "high"},
        {"cublas_workspace_config": ":16:8"},
    ],
)
def test_valid_but_different_policies_cannot_compare(common, changed):
    with pytest.raises(ValueError, match="policies differ"):
        noise.require_same_runtime_policy([policy(), policy(**changed)])


@pytest.mark.parametrize("mutation", ["none", "missing", "rank", "arm", "bridge"])
def test_full_arm_evidence_requires_all_ranks_arms_and_bridge_policy(common, tmp_path, mutation):
    plan = dict(base={"commit": "base"}, feature={"commit": "feature"}, input={"digest": "input"})
    profile = (2, 1, 1)
    for arm in noise.ARMS:
        write_arm(tmp_path / arm, plan, arm, profile)
        for path in (tmp_path / arm).glob("rank-*.json"):
            report = json.loads(path.read_text())
            report.update(policy())
            if report["bridge"]:
                report["bridge"].update(policy())
            if arm == "feat0" and report["rank"] == 1:
                if mutation == "missing":
                    del report["runtime_determinism"]
                elif mutation == "rank":
                    report.update(policy(cudnn_allow_tf32=False))
                elif mutation == "bridge":
                    report["bridge"].update(policy(cublas_workspace_config=":16:8"))
            if arm == "base1" and mutation == "arm":
                report.update(policy(cuda_matmul_allow_tf32=True))
            path.write_text(json.dumps(report))
    if mutation == "none":
        assert noise.compare_profile(plan, tmp_path, profile)["status"] == "PASS"
    else:
        with pytest.raises(ValueError):
            noise.compare_profile(plan, tmp_path, profile)


def test_unknown_policy_rejected_without_mutation():
    with pytest.raises(ValueError, match="Unknown numerical policy"):
        noise.configure_numerical_policy("balanced-routing")
    assert noise.NUMERICAL_POLICY == noise.LEGACY_NUMERICAL_POLICY
