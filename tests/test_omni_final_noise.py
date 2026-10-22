"""CPU fail-closed tests for final-commit binding of the frozen noise engine."""

import argparse
import ast
import copy
import json
import subprocess
from pathlib import Path

import pytest

from tests.acceptance import omni_final_noise as new


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit(repo, message):
    git(repo, "add", ".")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def binding(tmp_path, monkeypatch):
    base = tmp_path / "base"
    feature = tmp_path / "feature"
    base.mkdir()
    git(base, "init", "-q")
    git(base, "config", "user.email", "acceptance@example.invalid")
    git(base, "config", "user.name", "Acceptance fixture")
    for name in new.UNCHANGED_FILES:
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("VALUE = 1\n")
    for name, methods in new.UNCHANGED_FUNCTIONS.items():
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("training_data.py"):
            p.write_text(
                "class DensePretrainData:\n    def __init__(self): pass\n"
                "class OmniPretrainData:\n"
                "    def get_batch(self, step, device):\n"
                "        self.begin_step(step)\n"
                "        if self._dense is not None:\n"
                "            return self._dense.get_batch(step, device)\n"
                "        return self._media_microbatches(device)\n"
                + "".join(
                    f"    def {method.split('.')[1]}(self): pass\n"
                    for method in methods
                    if "." in method
                )
            )
        else:
            p.write_text("".join(f"def {method}(): return 0\n" for method in methods))
    training = base / "pithtrain/modules/training.py"
    training.write_text(
        training.read_text() + "class TrainingCfg:\n    native_vision: bool = False\n"
        "    deterministic: bool = False\n"
    )
    base_sha = commit(base, "baseline")
    subprocess.run(["git", "clone", "-q", str(base), str(feature)], check=True)
    git(feature, "config", "user.email", "acceptance@example.invalid")
    git(feature, "config", "user.name", "Acceptance fixture")
    (feature / "pithtrain/new_resume.py").write_text("STRICT_BRIDGE = True\n")
    feature_sha = commit(feature, "feature")
    monkeypatch.setattr(new, "BASE_SHA", base_sha)
    output = tmp_path / "binding.json"
    args = argparse.Namespace(
        base_repo=base, feature_repo=feature, expected_feature_commit=feature_sha, output=output
    )
    new.create_binding(args)
    return args, json.loads(output.read_text())


def test_reviewed_binding_captures_distinct_sources_and_exclusions(binding):
    args, bound = binding
    assert new.load_binding(args.output) == bound
    assert bound["baseline_provenance"] == new.BASE_PROVENANCE
    assert bound["base"]["production_sha256"] != bound["feature"]["production_sha256"]
    assert bound["numerical_execution"] == "NOT_RUN"
    assert len(bound["profiles"]) == 5 and bound["arms"] == ["base0", "base1", "feat0"]
    assert bound["settings"]["steps"] == 32
    assert bound["settings"]["benchmark"] is False
    assert bound["bridge"]["step"] == 16
    assert bound["source_audit"]["media_opt_in_defaults"] == {"native_vision": False}
    assert any("Whole PR100/PR101" in s for s in bound["does_not_cover"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "legacy"),
        ("baseline_provenance", {"parent": "unreviewed"}),
        ("engine_sha256", "0" * 64),
        ("adapter_sha256", "0" * 64),
        ("arms", ["base0", "feat0"]),
        ("profiles", [[1, 1, 1]]),
        ("bridge", {"step": 8}),
    ],
)
def test_binding_tampering_rejected(binding, field, value):
    args, data = binding
    data[field] = value
    args.output.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        new.load_binding(args.output)


@pytest.mark.parametrize(
    "relative,contents,error",
    [
        ("pithtrain/models/qwen3_omni_moe.py", "VALUE = 2\n", "Numerical file changed"),
        ("pithtrain/modules/optimizer.py", "VALUE = 2\n", "Numerical file changed"),
        ("pithtrain/modules/reproducibility.py", "VALUE = 2\n", "Numerical file changed"),
        ("pithtrain/contexts/training.py", "VALUE = 2\n", "Numerical file changed"),
        ("pithtrain/operators/new_kernel.py", "VALUE = 2\n", "Numerical tree file set changed"),
        ("pithtrain/pipeline/new_schedule.py", "VALUE = 2\n", "Numerical tree file set changed"),
    ],
)
def test_numerical_source_change_requires_new_review(binding, relative, contents, error):
    args, _ = binding
    p = args.feature_repo / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(contents)
    commit(args.feature_repo, "changed numerics")
    with pytest.raises(ValueError, match=error):
        new.common_audit(args.base_repo, args.feature_repo)


@pytest.mark.parametrize("kind", ["dirty", "new-commit", "wrong-feature"])
def test_stale_or_dirty_source_rejected(binding, kind):
    args, _ = binding
    p = args.feature_repo / "pithtrain/new_resume.py"
    if kind == "wrong-feature":
        args.expected_feature_commit = "f" * 40
        args.output = args.output.with_name("wrong.json")
        with pytest.raises(ValueError, match="Feature commit mismatch"):
            new.create_binding(args)
    else:
        p.write_text("STRICT_BRIDGE = False\n")
        if kind == "new-commit":
            commit(args.feature_repo, "new revision")
        with pytest.raises(ValueError):
            new.load_binding(args.output)


def test_changed_numeric_function_rejected(binding):
    args, _ = binding
    p = args.feature_repo / "pithtrain/modules/training.py"
    p.write_text(
        p.read_text().replace(
            "def make_adamw_optimizer(): return 0", "def make_adamw_optimizer(): return 1"
        )
    )
    commit(args.feature_repo, "optimizer changed")
    with pytest.raises(ValueError, match="Numerical function changed"):
        new.common_audit(args.base_repo, args.feature_repo)


def test_text_early_return_cannot_be_changed(binding):
    args, _ = binding
    p = args.feature_repo / "pithtrain/modules/training_data.py"
    p.write_text(p.read_text().replace("self.begin_step(step)", "self.begin_step(step + 1)"))
    commit(args.feature_repo, "data order changed")
    with pytest.raises(ValueError, match="Text batch early-return"):
        new.common_audit(args.base_repo, args.feature_repo)


def test_media_optin_must_default_false(binding):
    args, _ = binding
    p = args.feature_repo / "pithtrain/modules/training.py"
    p.write_text(p.read_text().replace("native_vision: bool = False", "native_vision: bool = True"))
    commit(args.feature_repo, "media enabled")
    with pytest.raises(ValueError, match="Media selector enabled"):
        new.common_audit(args.base_repo, args.feature_repo)


def test_old_entry_retains_original_locked_range():
    engine = new.engine_module()
    assert engine.BASE_SHA == "a0b14f7abbf4f84c7a9fbb76eb5a759a63f65010"
    assert engine.FEATURE_SHA == "d237e4c3d5ce9e38cf80e16a38dbd8f2210d0e53"
    assert engine.SETTINGS["steps"] == 32 and engine.SETTINGS["benchmark"] is False
    assert (engine.comparator().RATIO_WARN, engine.comparator().RATIO_FAIL) == (3.0, 5.0)


def test_bound_engine_reenters_new_adapter_and_retains_workload(binding, monkeypatch):
    args, data = binding
    monkeypatch.delenv(new.BINDING_ENV, raising=False)
    engine, bound = new.configured_engine(args.output)
    assert bound == data and engine.BASE_SHA == data["base"]["commit"]
    assert engine.FEATURE_SHA == data["feature"]["commit"]
    assert Path(engine.__file__).resolve() == Path(new.__file__).resolve()
    assert new.engine_module().BASE_SHA != engine.BASE_SHA


def test_probe_requires_explicit_binding(monkeypatch, tmp_path):
    monkeypatch.delenv(new.BINDING_ENV, raising=False)
    args = [
        "probe",
        "--target-repo",
        str(tmp_path),
        "--dataset",
        str(tmp_path),
        "--model",
        str(tmp_path),
        "--result",
        str(tmp_path / "result"),
    ]
    with pytest.raises(ValueError, match="explicit inherited"):
        new.main(args)


@pytest.mark.parametrize(
    "snapshot",
    [
        "",
        "0, NVIDIA A100, 0, 80000\n",
        "0, NVIDIA H100, 1000, 80000\n",
    ],
)
def test_gpu_preflight_rejects_missing_wrong_or_busy_device(monkeypatch, tmp_path, snapshot):
    monkeypatch.setattr(new.subprocess, "check_output", lambda *a, **k: snapshot)
    with pytest.raises(ValueError):
        new.gpu_before(
            ["python", "-m", "torch.distributed.run", "--nproc-per-node=1"],
            tmp_path / "run.log",
            {},
        )


def test_cpu_conversion_does_not_query_gpu(monkeypatch, tmp_path):
    def forbidden(*a, **k):
        raise AssertionError("No GPU query for CPU converter")

    monkeypatch.setattr(new.subprocess, "check_output", forbidden)
    new.gpu_before(
        ["python", "script.py", "dcp2hf"], tmp_path / "convert.log", {"CUDA_VISIBLE_DEVICES": ""}
    )


@pytest.fixture
def fake_arm(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    plan = {"final_binding_path": str(tmp_path / "binding.json")}
    Path(plan["final_binding_path"]).write_text("{}")
    engine = SimpleNamespace(
        PROFILES=((1, 1, 1),),
        ARMS=("base0", "base1", "feat0"),
        validate_arm=lambda *a: calls.append(("validate", a[-1])),
        execute_arm=lambda *a: calls.append(("execute", a[3], a[-1])),
        run_command=lambda *a: None,
    )
    monkeypatch.setattr(new, "engine_for_plan", lambda path: (engine, {}, plan))
    monkeypatch.setenv("PITHTRAIN_RUN_OMNI_GPU_TESTS", "1")
    args = argparse.Namespace(plan=tmp_path / "plan.json", profile="1,1,1", arm="feat0")
    return args, engine, calls


def test_single_arm_preserves_predecessor_order_and_deadline(fake_arm):
    args, _, calls = fake_arm
    before = new.time.monotonic()
    new.run_one(args)
    assert [call[:2] for call in calls] == [
        ("validate", "base0"),
        ("validate", "base1"),
        ("execute", "feat0"),
        ("validate", "feat0"),
    ]
    assert before < calls[2][2] <= new.time.monotonic() + 600
    report = json.loads((args.plan.parent / "runs/w1-pp1-ep1/feat0/execution.json").read_text())
    assert report["status"] == "COMPLETED_32_NATIVE_UPDATES"
    assert report["timeout_seconds"] == 600
    assert not (args.plan.parent / "runs/.sequential-arm-lock").exists()


def test_incomplete_predecessor_does_not_launch_feature(fake_arm):
    args, engine, calls = fake_arm

    def reject(*a):
        raise ValueError("Missing baseline report")

    engine.validate_arm = reject
    with pytest.raises(ValueError, match="Missing baseline report"):
        new.run_one(args)
    assert calls == []
    assert not (args.plan.parent / "runs/.sequential-arm-lock").exists()


def test_global_55_minute_bound_blocks_later_arm(fake_arm):
    args, _, calls = fake_arm
    out = args.plan.parent / "runs"
    out.mkdir()
    (out / "experiment-clock.json").write_text(
        json.dumps(
            {
                "started_unix": new.time.time() - 3301,
                "total_timeout_seconds": 3300,
            }
        )
    )
    with pytest.raises(ValueError, match="55-minute experiment budget exhausted"):
        new.run_one(args)
    assert calls == []


def test_arm_failure_retained_and_not_silently_retried(fake_arm):
    args, engine, calls = fake_arm
    args.arm = "base0"

    def fail(*a):
        calls.append(("failed-execution",))
        raise RuntimeError("Worker failed")

    engine.execute_arm = fail
    with pytest.raises(RuntimeError, match="Worker failed"):
        new.run_one(args)
    report = json.loads((args.plan.parent / "runs/w1-pp1-ep1/base0/execution.json").read_text())
    assert report["status"] == "FAILED_OR_INTERRUPTED"
    with pytest.raises(FileExistsError):
        new.run_one(args)
    assert calls == [("failed-execution",)]


def test_parallel_attempt_preserves_active_lock(fake_arm):
    args, _, calls = fake_arm
    lock = args.plan.parent / "runs/.sequential-arm-lock"
    lock.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        new.run_one(args)
    assert lock.is_dir() and calls == []


def test_gpu_opt_in_required_before_running_arm(fake_arm, monkeypatch):
    args, _, calls = fake_arm
    monkeypatch.delenv("PITHTRAIN_RUN_OMNI_GPU_TESTS")
    with pytest.raises(ValueError, match="GPU opt-in"):
        new.run_one(args)
    assert calls == []


@pytest.mark.parametrize(
    "replacement",
    ["deterministic: bool = True", "deterministic: int = False", "other_field: bool = False"],
)
def test_common_numerical_optin_cannot_change(binding, replacement):
    args, _ = binding
    p = args.base_repo / "pithtrain/modules/training.py"
    p.write_text(p.read_text().replace("deterministic: bool = False", replacement))
    with pytest.raises(ValueError, match="Common deterministic config"):
        new.common_audit(args.base_repo, args.feature_repo)


@pytest.fixture
def reviewed_binding(binding):
    """Use actual reviewed source, reconstructing the hash-pinned original ASTs.

    No external checkout or GPU is needed in CI. The two reconstructed baseline
    functions must match the independently recorded 1611d085 function hashes.
    """
    args, _ = binding
    path = "pithtrain/modules/training.py"
    current = new.ROOT / path
    feature_nodes = {
        name: new.function_node(current, name) for name in ("make_wsd_scheduler", "apply_fsdp")
    }
    originals = copy.deepcopy(feature_nodes)
    wsd = originals["make_wsd_scheduler"]
    wsd.body[-2:] = [ast.Return(value=copy.deepcopy(wsd.body[-1].value.args[0]))]
    fsdp = originals["apply_fsdp"]
    loop = next(n for n in fsdp.body if isinstance(n, ast.For))
    # The exact baseline edge loop is independently pinned as part of the entire
    # function, not accepted merely because the production projection accepts it.
    loop.body[1:3] = ast.parse(new.OLD_FSDP_EDGE).body
    assert new.ast_digest(wsd) == new.WSD_BASE_AST
    assert new.ast_digest(fsdp) == new.FSDP_BASE_AST
    for repo, nodes in ((args.base_repo, originals), (args.feature_repo, feature_nodes)):
        source = (repo / path).read_text()
        for name, node in nodes.items():
            source = source.replace(f"def {name}(): return 0", ast.unparse(node))
        (repo / path).write_text(source)
    for name in (
        "pithtrain/modules/resume_semantics.py",
        "pithtrain/models/qwen3_omni_talker_pipeline.py",
    ):
        target = args.feature_repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((new.ROOT / name).read_bytes())
    return args


def test_actual_reviewed_function_binding_and_roundtrip(reviewed_binding, monkeypatch):
    args = reviewed_binding
    monkeypatch.setattr(new, "BASE_SHA", commit(args.base_repo, "exact reviewed baseline ASTs"))
    args.expected_feature_commit = commit(args.feature_repo, "exact reviewed feature ASTs")
    args.output = args.output.with_name("reviewed-binding.json")
    new.create_binding(args)
    bound = new.load_binding(args.output)
    audit = bound["source_audit"]
    assert len(audit["identical_functions"]) == 9
    assert len(audit["scoped_equivalent_functions"]) == 2
    for report in audit["scoped_equivalent_functions"].values():
        assert report["raw_ast_equal"] is False
        assert report["baseline_function_ast_sha256"] == report["projected_function_ast_sha256"]
    assert bound["numerical_execution"] == "NOT_RUN"
    assert bound["settings"]["steps"] == 32 and len(bound["profiles"]) == 5


@pytest.mark.parametrize(
    "before,after,error",
    [
        ("return 1.0", "return 0.5", "feature make_wsd_scheduler"),
        ("LambdaLR(opt, lr_lambda)", "LambdaLR(opt, lambda _: 1.0)", "feature make_wsd_scheduler"),
        ("start_lr=start_lr", "start_lr=0.0", "feature make_wsd_scheduler"),
        ("reduce_dtype=torch.float32", "reduce_dtype=torch.bfloat16", "feature apply_fsdp"),
        (
            "set_gradient_divide_factor(1.0)",
            "set_gradient_divide_factor(2.0)",
            "feature apply_fsdp",
        ),
        ("reshard_after_forward=True", "reshard_after_forward=False", "feature apply_fsdp"),
        ('name != "layers"', 'name != "changed"', "feature apply_fsdp"),
    ],
)
def test_reviewed_numerical_or_constructor_drift_rejected(reviewed_binding, before, after, error):
    args = reviewed_binding
    p = args.feature_repo / "pithtrain/modules/training.py"
    source = p.read_text()
    # ast.unparse uses single quotes; normalize the one string-expression case.
    before, after = before.replace('"', "'"), after.replace('"', "'")
    assert before in source
    p.write_text(source.replace(before, after, 1))
    with pytest.raises(ValueError, match=error):
        new.common_audit(args.base_repo, args.feature_repo)


@pytest.mark.parametrize(
    "before,after",
    [
        ("scheduler._pith_factory_contract = contract", "scheduler.base_lrs = [0.0]"),
        ("return schedulers", "return schedulers[::-1]"),
        ('separators=(",", ":")', 'separators=(", ", ":")'),
        ("Path(source).read_bytes()", "b'changed implementation'"),
        ("return str(value)", "return 'changed canonical value'"),
        ("import hashlib", "import json as hashlib"),
    ],
)
def test_metadata_helper_dependencies_are_not_a_waiver(reviewed_binding, before, after):
    args = reviewed_binding
    p = args.feature_repo / "pithtrain/modules/resume_semantics.py"
    source = p.read_text()
    assert before in source
    p.write_text(source.replace(before, after, 1))
    with pytest.raises(ValueError, match="Scheduler metadata helper changed"):
        new.common_audit(args.base_repo, args.feature_repo)


def test_metadata_helper_rebinding_rejected(reviewed_binding):
    args = reviewed_binding
    p = args.feature_repo / "pithtrain/modules/resume_semantics.py"
    p.write_text(p.read_text() + "\nbind_scheduler_factory = lambda *args: ()\n")
    with pytest.raises(ValueError, match="Scheduler metadata helper changed"):
        new.common_audit(args.base_repo, args.feature_repo)


def test_unrelated_contract_body_does_not_claim_numerical_change(reviewed_binding):
    args = reviewed_binding
    p = args.feature_repo / "pithtrain/modules/resume_semantics.py"
    tree = ast.parse(p.read_text())
    function = next(n for n in tree.body if getattr(n, "name", None) == "callable_contract")
    function.body.append(
        ast.Raise(exc=ast.Call(func=ast.Name(id="RuntimeError"), args=[], keywords=[]))
    )
    p.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    assert (
        len(new.common_audit(args.base_repo, args.feature_repo)["scoped_equivalent_functions"]) == 2
    )


def test_unrelated_contract_decorator_cannot_rebind_helper(reviewed_binding):
    args = reviewed_binding
    p = args.feature_repo / "pithtrain/modules/resume_semantics.py"
    p.write_text(
        p.read_text().replace("def callable_contract", "@unsafe_rebind\ndef callable_contract")
    )
    with pytest.raises(ValueError, match="Scheduler metadata helper changed"):
        new.common_audit(args.base_repo, args.feature_repo)


def test_changed_talker_edge_selector_still_requires_review(reviewed_binding):
    args = reviewed_binding
    p = args.feature_repo / "pithtrain/models/qwen3_omni_talker_pipeline.py"
    tree = ast.parse(p.read_text())
    owner = next(n for n in tree.body if getattr(n, "name", None) == "Qwen3OmniMoeTalkerTextModel")
    method = next(n for n in owner.body if getattr(n, "name", None) == "fsdp_edge_modules")
    method.body = [ast.Return(value=ast.Tuple(elts=[], ctx=ast.Load()))]
    p.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    with pytest.raises(ValueError, match="Talker FSDP edge selector changed"):
        new.common_audit(args.base_repo, args.feature_repo)


@pytest.mark.parametrize("shape", ["cosine", "linear"])
def test_metadata_wrapper_preserves_actual_cpu_lr_sequence(reviewed_binding, shape):
    import math
    from types import SimpleNamespace

    import torch

    args = reviewed_binding
    rates, schedulers = [], []
    for repo in (args.base_repo, args.feature_repo):
        path = repo / "pithtrain/modules/training.py"
        node = new.function_node(path, "make_wsd_scheduler")
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        namespace = {
            "__name__": __name__,
            "math": math,
            "Literal": __import__("typing").Literal,
            "TrainingCfg": SimpleNamespace,
            "LRScheduler": torch.optim.lr_scheduler.LRScheduler,
            "LambdaLR": torch.optim.lr_scheduler.LambdaLR,
            "training": SimpleNamespace(optimizers=(optimizer,)),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        (scheduler,) = namespace["make_wsd_scheduler"](
            SimpleNamespace(lr=1e-4, max_steps=32),
            start_lr=1e-6,
            warmup_ratio=0.25,
            final_lr=1e-7,
            decay_ratio=0.25,
            decay_shape=shape,
        )
        sequence = [scheduler.get_last_lr()]
        for _ in range(32):
            optimizer.step()
            scheduler.step()
            sequence.append(scheduler.get_last_lr())
        rates.append(sequence)
        schedulers.append(scheduler)
    assert rates[0] == rates[1]
    assert not hasattr(schedulers[0], "_pith_factory_contract")
    assert schedulers[1]._pith_factory_contract["identity"]["arguments"]["s:max_steps"] == 32


@pytest.mark.parametrize(
    "before,after,error",
    [
        ("return 1.0", "return 0.5", "baseline make_wsd_scheduler"),
        (
            "set_gradient_divide_factor(1.0)",
            "set_gradient_divide_factor(2.0)",
            "baseline apply_fsdp",
        ),
    ],
)
def test_reviewed_baseline_drift_rejected(reviewed_binding, before, after, error):
    args = reviewed_binding
    p = args.base_repo / "pithtrain/modules/training.py"
    source = p.read_text()
    assert before in source
    p.write_text(source.replace(before, after, 1))
    with pytest.raises(ValueError, match=error):
        new.common_audit(args.base_repo, args.feature_repo)
