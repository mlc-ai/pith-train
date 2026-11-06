"""CPU evidence validation. Synthetic logs in these tests are not GPU acceptance."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.acceptance import omni_noise_regression as noise


def logs(ce=10.0, lb=1.0, steps=32):
    lines = []
    for step in range(1, steps + 1):
        lr = 1e-6 + (1e-5 - 1e-6) * step / 32
        lines.append(
            f"step {step:08d}/00000032 | cross-entropy-loss {ce:.4f} | "
            f"load-balance-loss {lb:.6f} | learning-rate {lr:.6e} | gradient-norm 0.2500"
        )
    return "\n".join(lines)


@pytest.mark.parametrize("count", [0, 1, 3, 31, 33])
def test_reject_incomplete_or_extra_steps(count):
    with pytest.raises(ValueError):
        noise.parse_training_log(logs(steps=count))


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("step 00000002/", "step 00000001/"),
        lambda s: s.replace("/00000032", "/00000003"),
        lambda s: s.replace("load-balance-loss 1.000000", ""),
        lambda s: s.replace("cross-entropy-loss 10.0000", "cross-entropy-loss nan"),
        lambda s: s.replace("cross-entropy-loss 10.0000", "cross-entropy-loss inf"),
        lambda s: s.replace("cross-entropy-loss 10.0000", "cross-entropy-loss -1"),
        lambda s: s.replace("gradient-norm 0.2500", "gradient-norm nan"),
        lambda s: s.replace("learning-rate 1.281250e-06", "learning-rate 1.000000e-05"),
        lambda s: s + "\n" + s.splitlines()[0],
        lambda s: s.replace(
            "load-balance-loss 1.000000", "load-balance-loss 1 load-balance-loss 1"
        ),
    ],
)
def test_reject_corrupt_training_evidence(change):
    with pytest.raises(ValueError):
        noise.parse_training_log(change(logs()))


@pytest.mark.parametrize(
    "signal,status",
    [(0, "PASS"), (2, "PASS"), (3, "INVESTIGATE"), (4, "INVESTIGATE"), (5, "FAIL"), (6, "FAIL")],
)
def test_exact_repository_thresholds(signal, status):
    runs = [noise.parse_training_log(logs(ce=x)) for x in (10, 11, 10 + signal)]
    result = noise.compare_records(runs)
    assert result["status"] == status
    assert result["metrics"]["cross-entropy-loss"]["ratio"] == signal
    assert result["step_1_forward"]["gated"] is False


def test_zero_noise_floor_is_exact_and_load_balance_is_gated():
    base = noise.parse_training_log(logs())
    assert noise.compare_records([base, base, base])["status"] == "PASS"
    result = noise.compare_records([base, base, noise.parse_training_log(logs(lb=1.000001))])
    assert result["status"] == "FAIL"
    assert result["metrics"]["load-balance-loss"]["ratio"] == "infinity"


def test_short_or_nonfinite_records_do_not_compare():
    base = noise.parse_training_log(logs())
    with pytest.raises(ValueError):
        noise.compare_records([base, base, base[:-1]])
    bad = copy.deepcopy(base)
    bad[7]["load-balance-loss"] = float("nan")
    with pytest.raises(ValueError):
        noise.compare_records([base, bad, base])


def write_arm(folder, plan, arm, profile=(1, 1, 1)):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "training.log").write_text(logs())
    for rank in range(profile[0]):
        report = dict(
            status="COMPLETED_32_NATIVE_UPDATES",
            arm=arm,
            rank=rank,
            source=plan["feature" if arm == "feat0" else "base"],
            settings=noise.SETTINGS,
            profile=list(profile),
            input=plan["input"],
            consumed_samples=256,
            batches={str(i): str(i) for i in range(32)},
            initial_weights_sha256="same",
            gpu="NVIDIA H100",
            native_fa4=True,
            torch="test-fixture",
            records=noise.parse_training_log(logs()) if rank == 0 else None,
            bridge=(
                dict(
                    protocol=noise.BRIDGE,
                    producer_pid=1,
                    restorer_pid=2,
                    before_sha256="boundary",
                    restored_sha256="boundary",
                    exact_boundary=True,
                    strict_hf_strict=True,
                )
                if arm == "feat0"
                else None
            ),
        )
        (folder / f"rank-{rank:05d}.json").write_text(json.dumps(report))


@pytest.fixture
def experiment(tmp_path):
    plan = dict(
        base={"commit": "base"}, feature={"commit": "feature"}, input={"digest": "real-input"}
    )
    for arm in noise.ARMS:
        write_arm(tmp_path / arm, plan, arm)
    return plan, tmp_path


def test_three_complete_reports_compare(experiment):
    plan, folder = experiment
    assert noise.compare_profile(plan, folder, (1, 1, 1))["status"] == "PASS"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", {"commit": "wrong"}),
        ("consumed_samples", 24),
        ("rank", 1),
        ("profile", [4, 2, 2]),
        ("input", {"digest": "other"}),
        ("initial_weights_sha256", "changed"),
        ("batches", {str(i): "changed" for i in range(32)}),
        ("settings", {"steps": 3}),
        ("status", "PARTIAL"),
        ("gpu", "RTX 5080"),
        ("native_fa4", False),
    ],
)
def test_cross_arm_identity_and_completion_fail_closed(experiment, field, value):
    plan, folder = experiment
    path = folder / "feat0/rank-00000.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        noise.compare_profile(plan, folder, (1, 1, 1))


def test_missing_or_unexpected_rank_is_rejected(experiment):
    plan, folder = experiment
    with pytest.raises(ValueError, match="rank"):
        noise.validate_arm(plan, folder / "base0", (2, 1, 1), "base0")
    (folder / "base0/rank-00001.json").write_text("{}")
    with pytest.raises(ValueError, match="rank"):
        noise.validate_arm(plan, folder / "base0", (1, 1, 1), "base0")


def test_live_metrics_must_match_captured_log(experiment):
    plan, folder = experiment
    (folder / "feat0/training.log").write_text(logs(ce=11))
    with pytest.raises(ValueError, match="metrics"):
        noise.compare_profile(plan, folder, (1, 1, 1))


def test_timeout_terminates_arm_and_preserves_log(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        noise.run_command(
            [sys.executable, "-c", "import time; print('started',flush=True); time.sleep(10)"],
            tmp_path / "timeout.log",
            os.environ.copy(),
            0.2,
        )
    assert "started" in (tmp_path / "timeout.log").read_text()


def test_gpu_entry_cannot_run_without_opt_in(tmp_path):
    env = noise.child_env(noise.REPO, cpu=True)
    env.pop("PITHTRAIN_RUN_OMNI_GPU_TESTS", None)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(noise.__file__)),
            "run",
            "--plan",
            str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0 and "opt-in" in result.stderr
    assert not (tmp_path / "runs").exists()


def test_no_runtime_step_or_tolerance_override():
    with pytest.raises(SystemExit):
        noise.parse_args(["run", "--plan", "plan.json", "--steps", "3"])
    assert noise.SETTINGS["steps"] == 32
    assert noise.SETTINGS["benchmark"] is False
    assert noise.SETTINGS["warmup_ratio"] == 1.0
    assert noise.PROFILES[-1] == (4, 2, 2)


def test_real_native_config_structure_without_gpu_import():
    result = noise.verify_config_structure(
        noise.REPO, dict(dataset="real-prepared-data", model="tiny-omni")
    )
    assert result.startswith("AST_FIELDS_AND_SHARED_CONFIG_BUILDER_PASS")
    assert "pithtrain.operators.deepgemm_quantize" not in sys.modules


def test_source_identity_requires_clean_real_git_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "pithtrain").mkdir()
    (tmp_path / "pithtrain/model.py").write_text("value=1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=CPU test",
            "-c",
            "user.email=cpu@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    assert noise.source_identity(tmp_path)["production_sha256"]
    (tmp_path / "pithtrain/model.py").write_text("value=2\n")
    with pytest.raises(ValueError, match="Dirty"):
        noise.source_identity(tmp_path)


@pytest.mark.parametrize(
    "mode,expected_calls,status",
    [
        ("complete", 15, "PASS"),
        ("failure", 2, "FAIL_OR_INCOMPLETE"),
        ("investigate", 3, "INVESTIGATE"),
    ],
)
def test_orchestration_is_sequential_and_fails_closed(
    tmp_path, monkeypatch, mode, expected_calls, status
):
    from types import SimpleNamespace

    plan = dict(
        base={"commit": "base", "path": "base-code"},
        feature={"commit": "feature", "path": "feature-code"},
        input={"digest": "same-real-input"},
    )
    monkeypatch.setenv("PITHTRAIN_RUN_OMNI_GPU_TESTS", "1")
    monkeypatch.setattr(noise, "verify_plan", lambda path: plan)
    calls = []

    def cpu_arm_fixture(plan_path, plan, profile, arm, arm_output, deadline):
        calls.append((profile, arm))
        if mode == "failure" and len(calls) == 2:
            raise RuntimeError("CPU-injected arm failure; no GPU ran")
        write_arm(arm_output, plan, arm, profile)
        if mode == "investigate":
            ce = {"base0": 10, "base1": 11, "feat0": 13}[arm]
            (arm_output / "training.log").write_text(logs(ce=ce))
            report_path = arm_output / "rank-00000.json"
            report = json.loads(report_path.read_text())
            report["records"] = noise.parse_training_log(logs(ce=ce))
            report_path.write_text(json.dumps(report))

    monkeypatch.setattr(noise, "execute_arm", cpu_arm_fixture)
    code = noise.run(SimpleNamespace(plan=tmp_path / "plan.json"))
    assert len(calls) == expected_calls
    assert calls == [(p, a) for p in noise.PROFILES for a in noise.ARMS][:expected_calls]
    summary = json.loads((tmp_path / "runs/summary.json").read_text())
    assert summary["status"] == status
    assert code == (0 if status == "PASS" else 2 if status == "INVESTIGATE" else 1)
    if mode != "complete":
        assert "NOT_RUN" in summary["arms"].values()
    if mode == "failure":
        assert summary["arms"]["w1-pp1-ep1/base1"] == "FAILED_OR_INTERRUPTED"


def test_bridge_segments_still_require_all_32_steps():
    lines = logs().splitlines()
    prefix = noise.parse_training_log("\n".join(lines[:16]), 1, 16)
    suffix = noise.parse_training_log("\n".join(lines[16:]), 17, 32)
    assert prefix + suffix == noise.parse_training_log(logs())
    with pytest.raises(ValueError):
        noise.parse_training_log("\n".join(lines[:16]))
    with pytest.raises(ValueError):
        noise.parse_training_log("\n".join(lines[:3]), 1, 3)


@pytest.mark.parametrize(
    "change",
    [
        lambda b: None,
        lambda b: dict(b, exact_boundary=False),
        lambda b: dict(b, restored_sha256="wrong"),
        lambda b: dict(b, restorer_pid=1),
        lambda b: dict(b, strict_hf_strict=False),
        lambda b: dict(b, protocol=dict(noise.BRIDGE, step=3)),
    ],
)
def test_feature_cannot_bypass_or_falsify_bridge(experiment, change):
    plan, folder = experiment
    path = folder / "feat0/rank-00000.json"
    report = json.loads(path.read_text())
    report["bridge"] = change(report["bridge"])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        noise.compare_profile(plan, folder, (1, 1, 1))


def test_bridge_executes_native_save_cpu_export_import_then_fresh_restore(tmp_path, monkeypatch):
    import time

    plan = dict(base=dict(path="base"), feature=dict(path="candidate"))
    calls = []

    def command(command, log, env, timeout):
        calls.append((command, env, timeout))
        lines = logs().splitlines()
        log.write_text(
            "\n".join(
                lines[:16]
                if log.name == "produce.log"
                else lines[16:]
                if log.name == "restore.log"
                else []
            )
        )

    monkeypatch.setattr(noise, "run_command", command)
    noise.execute_arm(
        tmp_path / "plan.json", plan, (4, 2, 2), "feat0", tmp_path, time.monotonic() + 3300
    )
    assert len(calls) == 4
    assert calls[0][0][calls[0][0].index("--segment") + 1] == "produce"
    assert calls[1][0][2] == "dcp2hf" and "fused" in calls[1][0]
    assert calls[2][0][2] == "hf2dcp" and "--restore-resume" in calls[2][0]
    assert calls[3][0][calls[3][0].index("--segment") + 1] == "restore"
    assert calls[1][1]["CUDA_VISIBLE_DEVICES"] == calls[2][1]["CUDA_VISIBLE_DEVICES"] == ""
    assert all(0 < c[2] <= 600 for c in calls)
    assert len(noise.parse_training_log((tmp_path / "training.log").read_text())) == 32


def test_bridge_failed_conversion_never_starts_restore(tmp_path, monkeypatch):
    import time

    calls = []

    def command(cmd, log, env, timeout):
        calls.append(cmd)
        log.write_text("CPU fixture; no GPU executed")
        if len(calls) == 2:
            raise RuntimeError("conversion failed")

    monkeypatch.setattr(noise, "run_command", command)
    with pytest.raises(RuntimeError):
        noise.execute_arm(
            tmp_path / "plan.json",
            dict(feature=dict(path="candidate")),
            (4, 2, 2),
            "feat0",
            tmp_path,
            time.monotonic() + 3300,
        )
    assert len(calls) == 2
    assert not (tmp_path / "training.log").exists()


def test_full_numpy_rng_and_tensor_bits_are_hashed():
    import numpy as np
    import torch

    value = dict(numpy=np.arange(624, dtype=np.uint32), tensor=torch.tensor([1.0]))
    original = noise.tensor_digest(value)
    value["numpy"][300] += 1  # Interior values disappear in a truncated ndarray repr.
    assert noise.tensor_digest(value) != original
    value["numpy"][300] -= 1
    value["tensor"][0] = torch.nextafter(value["tensor"][0], torch.tensor(float("inf")))
    assert noise.tensor_digest(value) != original
    assert noise.tensor_digest([1, 23]) != noise.tensor_digest([12, 3])
