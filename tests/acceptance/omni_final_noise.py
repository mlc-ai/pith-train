"""Reviewed final-commit binding for the unchanged 32-update noise experiment.

The historical a0b14f7 -> d237e4c entry retains its default policy. This adapter explicitly
binds a current-main baseline and a reviewed final feature commit, fingerprints
the workload engine with an explicit common deterministic opt-in, and retains all its runtime/data/restore/comparison
checks. It is a text-path strict-resume/ordinary-training regression, not proof
of whole-PR100/101 or image/audio/video numerical correctness.

No allocation, credentials, uploads, or network access are performed here.
"""

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = ROOT / "tests/acceptance/omni_noise_regression.py"
BASE_SHA = "1611d085a7c244bb01a34caf8b58d19aef9eeeff"
ENGINE_SHA256 = "2b1a8f231c4d1c6543697073f4961edf134aada924be90ffa271c49e080259ff"
SCHEMA = "omni_final_text_noise_binding_v2"
BINDING_ENV = "PITHTRAIN_OMNI_FINAL_NOISE_BINDING"
SHA = re.compile(r"[0-9a-f]{40}")
UNCHANGED_FILES = (
    "pithtrain/contexts/training.py",
    "pithtrain/modules/reproducibility.py",
    "pithtrain/models/qwen3_omni_moe.py",
    "pithtrain/modules/dataset.py",
    "pithtrain/modules/data_config.py",
    "pithtrain/modules/model_config.py",
    "pithtrain/modules/optimizer.py",
    "pithtrain/modules/load_balance.py",
    "pithtrain/tasks/prepare_omni_data.py",
    "pithtrain/tasks/convert_checkpoint/qwen3_omni_moe.py",
    "tests/configs/qwen3_omni_text/tiny.json",
)
UNCHANGED_TREES = (
    "pithtrain/operators",
    "pithtrain/pipeline",
    "pithtrain/distributed",
)
UNCHANGED_FUNCTIONS = {
    "pithtrain/modules/training.py": (
        "make_adamw_optimizer",
        "make_wsd_scheduler",
        "apply_fsdp",
        "setup_training",
    ),
    "pithtrain/tasks/pretrain_lm.py": ("objective", "get_global_batch"),
    "pithtrain/modules/training_data.py": (
        "DensePretrainData",
        "OmniPretrainData.begin_step",
        "OmniPretrainData.commit_step",
        "OmniPretrainData.state_dict",
        "OmniPretrainData.load_state_dict",
    ),
}
EXCLUSIONS = [
    "Whole PR100/PR101 and changes already present in the reviewed baseline",
    "Image/audio/video/mixed modalities, encoder gradients and AV positions",
    "Full-size/released weights, CP>1, HSDP>1, Muon, FP8, changed topology",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def function_dump(path, qualified_name):
    nodes = ast.parse(Path(path).read_text()).body
    found = None
    for part in qualified_name.split("."):
        found = next((n for n in nodes if getattr(n, "name", None) == part), None)
        require(found is not None, f"Missing audited function {qualified_name}")
        nodes = getattr(found, "body", ())
    return ast.dump(found, include_attributes=False)


def common_audit(base, feature):
    """Fail closed if the previously reviewed numerical implementation changes."""
    names = set(UNCHANGED_FILES)
    for tree in UNCHANGED_TREES:
        base_names = set(git(base, "ls-files", tree).splitlines())
        feature_names = set(git(feature, "ls-files", tree).splitlines())
        require(base_names == feature_names, f"Numerical tree file set changed: {tree}")
        names |= base_names
    files = {}
    for name in sorted(names):
        a, b = digest(base / name), digest(feature / name)
        require(a == b, f"Numerical file changed; review a new baseline/scope: {name}")
        files[name] = a
    functions = {}
    for name, symbols in UNCHANGED_FUNCTIONS.items():
        for symbol in symbols:
            a = function_dump(base / name, symbol)
            b = function_dump(feature / name, symbol)
            require(
                a == b, f"Numerical function changed; explicit re-review required: {name}:{symbol}"
            )
            functions[f"{name}:{symbol}"] = hashlib.sha256(a.encode()).hexdigest()
    # The public numerical opt-in exists independently of checkpoint features.
    for repo in (base, feature):
        cfg_tree = ast.parse((repo / "pithtrain/modules/training.py").read_text())
        config = next(
            n for n in cfg_tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingCfg"
        )
        field = next(
            (
                n
                for n in config.body
                if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == "deterministic"
            ),
            None,
        )
        require(
            field is not None
            and isinstance(field.annotation, ast.Name)
            and field.annotation.id == "bool"
            and isinstance(field.value, ast.Constant)
            and field.value.value is False,
            "Common deterministic config must be bool and default False",
        )
    # Media-only extensions occur after the unmodified text early return.
    prefixes = []
    for repo in (base, feature):
        tree = ast.parse((repo / "pithtrain/modules/training_data.py").read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniPretrainData"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_batch"
        )
        prefixes.append(
            ast.dump(ast.Module(body=method.body[:2], type_ignores=[]), include_attributes=False)
        )
    require(prefixes[0] == prefixes[1], "Text batch early-return path changed")
    # Every media selector must remain an explicit false-by-default opt-in.
    tree = ast.parse((feature / "pithtrain/modules/training.py").read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingCfg")
    defaults = {
        n.target.id: ast.literal_eval(n.value)
        for n in cfg.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("native_")
    }
    require(
        all(value is False for value in defaults.values()),
        "Media selector enabled in text experiment",
    )
    return {
        "common_numerical_policy": "common-deterministic-v2",
        "common_deterministic_default": False,
        "identical_files": files,
        "identical_functions": functions,
        "text_get_batch_prefix_sha256": hashlib.sha256(prefixes[0].encode()).hexdigest(),
        "media_opt_in_defaults": defaults,
    }


def engine_module():
    require(digest(ENGINE_PATH) == ENGINE_SHA256, "Reviewed workload engine changed")
    spec = importlib.util.spec_from_file_location("omni_fixed_noise_engine", ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.configure_numerical_policy(engine.COMMON_NUMERICAL_POLICY)
    return engine


def create_binding(args):
    engine = engine_module()
    base, feature = args.base_repo.resolve(), args.feature_repo.resolve()
    require(SHA.fullmatch(args.expected_feature_commit), "Expected a full feature commit")
    require(args.expected_feature_commit != BASE_SHA, "Identical code is only a noise control")
    base_id, feature_id = engine.source_identity(base), engine.source_identity(feature)
    require(base_id["commit"] == BASE_SHA, "Unreviewed baseline commit")
    require(feature_id["commit"] == args.expected_feature_commit, "Feature commit mismatch")
    require(
        base_id["production_sha256"] != feature_id["production_sha256"],
        "Identical production sources",
    )
    audit = common_audit(base, feature)
    base_files = set(git(base, "ls-files", "pithtrain").splitlines())
    feature_files = set(git(feature, "ls-files", "pithtrain").splitlines())
    changed = [
        name
        for name in sorted(base_files | feature_files)
        if name not in base_files
        or name not in feature_files
        or digest(base / name) != digest(feature / name)
    ]
    require(changed, "No production increment")
    payload = dict(
        schema=SCHEMA,
        base=base_id,
        feature=feature_id,
        source_audit=audit,
        changed_production_files=changed,
        engine_sha256=ENGINE_SHA256,
        adapter_sha256=digest(__file__),
        settings=engine.SETTINGS,
        profiles=[list(p) for p in engine.PROFILES],
        bridge=engine.BRIDGE,
        arms=list(engine.ARMS),
        scope="Text training plus strict/HF/strict continuation under common-deterministic-v2",
        scope_changed_declared="Both baseline arms and feature explicitly enable the same public deterministic policy; legacy results are not inherited",
        does_not_cover=EXCLUSIONS,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        numerical_execution="NOT_RUN",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, payload)
    print(
        json.dumps(
            {
                "status": "BOUND_NOT_EXECUTED",
                "binding": str(args.output),
                "base": BASE_SHA,
                "feature": feature_id["commit"],
            }
        )
    )


def load_binding(path):
    path = Path(path).resolve()
    b = json.loads(path.read_text())
    require(b["schema"] == SCHEMA, "Wrong binding schema")
    require(b["adapter_sha256"] == digest(__file__), "Binding adapter changed")
    require(b["engine_sha256"] == ENGINE_SHA256 == digest(ENGINE_PATH), "Workload engine changed")
    require(b["base"]["commit"] == BASE_SHA, "Unreviewed base")
    require(SHA.fullmatch(b["feature"]["commit"]), "Invalid feature SHA")
    engine = engine_module()
    require(
        (b["settings"], b["profiles"], b["bridge"], b["arms"])
        == (engine.SETTINGS, [list(p) for p in engine.PROFILES], engine.BRIDGE, list(engine.ARMS)),
        "Workload or intervention changed",
    )
    for key in ("base", "feature"):
        require(engine.source_identity(b[key]["path"]) == b[key], f"{key} changed after binding")
    require(
        common_audit(Path(b["base"]["path"]), Path(b["feature"]["path"])) == b["source_audit"],
        "Source audit changed",
    )
    return b


def configured_engine(binding_path):
    """Bind the reviewed engine and explicit common numerical policy."""
    b = load_binding(binding_path)
    engine = engine_module()
    engine.BASE_SHA, engine.FEATURE_SHA = b["base"]["commit"], b["feature"]["commit"]
    # The engine uses its entry path for fresh-process probe/arm subprocesses
    # and its driver fingerprint. Re-enter THIS explicit adapter in children.
    engine.__file__ = str(Path(__file__).resolve())
    original_verify = engine.verify_plan

    def verify_plan(path):
        plan = original_verify(path)
        require(plan.get("final_binding") == b, "Missing or changed final comparison binding")
        require(plan.get("final_binding_sha256") == digest(binding_path), "Binding file changed")
        load_binding(binding_path)
        return plan

    engine.verify_plan = verify_plan
    os.environ[BINDING_ENV] = str(Path(binding_path).resolve())
    return engine, b


def prepare(args):
    engine, binding = configured_engine(args.binding)
    args.base_repo = Path(binding["base"]["path"])
    args.feature_repo = Path(binding["feature"]["path"])
    engine.prepare(args)
    plan_path = args.output / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan.update(
        final_binding=binding,
        final_binding_path=str(args.binding.resolve()),
        final_binding_sha256=digest(args.binding),
    )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    engine.verify_plan(plan_path)


def engine_for_plan(path):
    preliminary = json.loads(Path(path).read_text())
    engine, binding = configured_engine(preliminary["final_binding_path"])
    return engine, binding, engine.verify_plan(path)


def gpu_before(command, log_path, env):
    if "torch.distributed.run" not in command:
        return
    world = next(int(arg.split("=")[1]) for arg in command if arg.startswith("--nproc-per-node="))
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
        env=env,
    )
    Path(str(log_path) + ".gpu.csv").write_text(result)
    rows = [line.split(",") for line in result.splitlines() if line.strip()]
    require(len(rows) >= world, "Insufficient allocated GPUs")
    require(
        all("H100" in row[1] and int(row[2]) < 1000 for row in rows[:world]),
        "GPUs are wrong hardware or busy",
    )


def run_one(args):
    engine, _, plan = engine_for_plan(args.plan)
    require(os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", "GPU opt-in required")
    profile = tuple(map(int, args.profile.split(",")))
    require(profile in engine.PROFILES, "Unsupported topology")
    output = args.plan.parent / "runs"
    output.mkdir(exist_ok=True)
    lock = output / ".sequential-arm-lock"
    lock.mkdir(exist_ok=False)
    name = f"w{profile[0]}-pp{profile[1]}-ep{profile[2]}"
    folder = output / name
    arm_dir = folder / args.arm
    folder.mkdir(exist_ok=True)
    try:
        clock_path = output / "experiment-clock.json"
        if not clock_path.exists():
            write(clock_path, {"started_unix": time.time(), "total_timeout_seconds": 3300})
        clock = json.loads(clock_path.read_text())
        require(clock["total_timeout_seconds"] == 3300, "Global timeout changed")
        remaining = clock["started_unix"] + 3300 - time.time()
        require(remaining > 0, "55-minute experiment budget exhausted; remaining arms NOT_RUN")
        # The parent coordinator selects topologies, but each profile always
        # executes base0 -> base1 -> feature in fresh, strictly sequential arms.
        for predecessor in engine.ARMS[: engine.ARMS.index(args.arm)]:
            engine.validate_arm(plan, folder / predecessor, profile, predecessor)
        arm_dir.mkdir(exist_ok=False)
        original = engine.run_command

        def checked_run(command, log_path, env, timeout):
            gpu_before(command, log_path, env)
            return original(command, log_path, env, timeout)

        engine.run_command = checked_run
        started = time.monotonic()
        status = "FAILED_OR_INTERRUPTED"
        try:
            engine.execute_arm(
                args.plan, plan, profile, args.arm, arm_dir, started + min(600, remaining)
            )
            engine.validate_arm(plan, arm_dir, profile, args.arm)
            status = "COMPLETED_32_NATIVE_UPDATES"
        finally:
            write(
                arm_dir / "execution.json",
                dict(
                    status=status,
                    profile=list(profile),
                    arm=args.arm,
                    elapsed_seconds=time.monotonic() - started,
                    timeout_seconds=600,
                    binding_sha256=digest(plan["final_binding_path"]),
                ),
            )
        print(json.dumps({"status": status, "profile": profile, "arm": args.arm}))
    finally:
        lock.rmdir()


def compare(args):
    engine, binding, plan = engine_for_plan(args.plan)
    profiles = (
        engine.PROFILES if args.profile == "all" else (tuple(map(int, args.profile.split(","))),)
    )
    require(all(p in engine.PROFILES for p in profiles), "Unsupported topology")
    results = {}
    for profile in profiles:
        name = f"w{profile[0]}-pp{profile[1]}-ep{profile[2]}"
        results[name] = engine.compare_profile(plan, args.plan.parent / "runs" / name, profile)
    passed = all(r["status"] == "PASS" for r in results.values())
    report = dict(
        status=("PASS_ALL_FIVE" if len(profiles) == 5 else "PASS_SELECTED")
        if passed
        else "FAIL_OR_INVESTIGATE",
        profiles=results,
        remaining_profiles="NOT_RUN_OR_NOT_COMPARED" if len(profiles) < 5 else None,
        binding=binding,
        whole_omni_acceptance=False,
    )
    write(args.output, report)
    print(json.dumps(report))
    return 0 if passed else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="mode", required=True)
    b = commands.add_parser("bind")
    for name in ("base-repo", "feature-repo", "output"):
        b.add_argument("--" + name, type=Path, required=True)
    b.add_argument("--expected-feature-commit", required=True)
    prep = commands.add_parser("prepare")
    for name in ("binding", "dataset", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    one = commands.add_parser("run-arm")
    one.add_argument("--plan", type=Path, required=True)
    one.add_argument("--profile", required=True)
    one.add_argument("--arm", choices=("base0", "base1", "feat0"), required=True)
    comp = commands.add_parser("compare")
    comp.add_argument("--plan", type=Path, required=True)
    comp.add_argument("--profile", default="all")
    comp.add_argument("--output", type=Path, required=True)
    for mode in ("probe", "arm"):
        commands.add_parser(mode, add_help=False)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("probe", "arm"):
        return engine_module().parse_args(argv)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "bind":
        return create_binding(args)
    if args.mode in ("probe", "arm"):
        require(
            BINDING_ENV in os.environ, "Fresh worker requires explicit inherited experiment binding"
        )
        engine, _ = configured_engine(os.environ[BINDING_ENV])
        return getattr(engine, args.mode)(args)
    return {"prepare": prepare, "run-arm": run_one, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
