"""Three fresh 32-step native Omni runs, judged against base-vs-base noise.

prepare is CPU-only; run requires an already authorized H100 allocation.
This driver never submits jobs, switches branches, installs packages or uploads.
"""

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = ("base0", "base1", "feat0")
BASE_SHA = "a0b14f7abbf4f84c7a9fbb76eb5a759a63f65010"
FEATURE_SHA = "d237e4c3d5ce9e38cf80e16a38dbd8f2210d0e53"
BRIDGE = dict(step=16, layout="fused", prefix="thinker.", restore="fresh_process_strict")
PROFILES = ((1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 1, 2), (4, 2, 2))
SETTINGS = dict(
    steps=32,
    sequence_length=32,
    global_batch_size=8,
    micro_batch_size=1,
    seed=1234,
    dtype="bfloat16",
    lr=1e-5,
    start_lr=1e-6,
    warmup_ratio=1.0,
    decay_ratio=0.0,
    moe_load_balance_type="global-batch",
    moe_load_balance_coef=1e-3,
    benchmark=False,
    fp8=False,
    cp=1,
    hsdp_replica=1,
)
LEGACY_NUMERICAL_POLICY = "legacy-v1"
COMMON_NUMERICAL_POLICY = "common-deterministic-v2"
NUMERICAL_POLICY = LEGACY_NUMERICAL_POLICY
RUNTIME_KEYS = {
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
METRICS = ("cross-entropy-loss", "load-balance-loss")
STEP = re.compile(r"\bstep (\d+)/(\d+)\b")
NUMBER = r"([-+\w.]+)"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure_numerical_policy(policy=LEGACY_NUMERICAL_POLICY):
    """Explicit adapter opt-in; the historical entry keeps its original policy."""
    global NUMERICAL_POLICY
    require(
        policy in (LEGACY_NUMERICAL_POLICY, COMMON_NUMERICAL_POLICY), "Unknown numerical policy"
    )
    NUMERICAL_POLICY = policy
    SETTINGS.pop("numerical_policy", None)
    if policy == COMMON_NUMERICAL_POLICY:
        SETTINGS["numerical_policy"] = policy


def validate_runtime_policy(report):
    """Require actual rank settings, not inferred settings from source/config."""
    if NUMERICAL_POLICY == LEGACY_NUMERICAL_POLICY:
        return
    require(report.get("numerical_policy") == COMMON_NUMERICAL_POLICY, "Missing numerical policy")
    runtime = report.get("runtime_determinism", {})
    require(set(runtime) == RUNTIME_KEYS, "Missing/unexpected runtime policy fields")
    require(
        runtime["deterministic_algorithms"] is True
        and runtime["deterministic_warn_only"] is False
        and runtime["cudnn_benchmark"] is False
        and runtime["cudnn_deterministic"] is True
        and runtime["fa4_deterministic"] is True
        and runtime["cublas_workspace_config"] in (":4096:8", ":16:8"),
        "Actual deterministic policy was not enabled",
    )
    require(
        type(runtime["cudnn_allow_tf32"]) is bool
        and type(runtime["cuda_matmul_allow_tf32"]) is bool
        and runtime["float32_matmul_precision"] in ("highest", "high", "medium"),
        "Malformed actual precision policy",
    )


def runtime_policy_metadata():
    if NUMERICAL_POLICY == LEGACY_NUMERICAL_POLICY:
        return {}
    from pithtrain.modules.reproducibility import runtime_determinism_snapshot

    metadata = dict(
        numerical_policy=NUMERICAL_POLICY,
        runtime_determinism=runtime_determinism_snapshot(),
    )
    validate_runtime_policy(metadata)
    return metadata


def require_same_runtime_policy(reports):
    if NUMERICAL_POLICY == LEGACY_NUMERICAL_POLICY:
        return
    for report in reports:
        validate_runtime_policy(report)
    require(
        all(r["runtime_determinism"] == reports[0]["runtime_determinism"] for r in reports),
        "Actual numerical policies differ across arms/ranks/processes",
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_identity(repo):
    repo = Path(repo).resolve()
    git = lambda *args: subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    require(not git("status", "--porcelain", "--untracked-files=all"), f"Dirty source: {repo}")
    files = git("ls-files", "pithtrain").splitlines()
    require(files, "No tracked production files")
    production = hashlib.sha256()
    for name in files:
        production.update((name + "\0" + sha(repo / name) + "\n").encode())
    return dict(
        path=str(repo), commit=git("rev-parse", "HEAD"), production_sha256=production.hexdigest()
    )


def select_source(repo):
    require(
        not any(n == "pithtrain" or n.startswith("pithtrain.") for n in sys.modules),
        "Use a fresh process for each source/arm",
    )
    sys.path.insert(0, str(Path(repo).resolve()))
    extra = os.environ.get("PITHTRAIN_EXTRA_SITE_PACKAGES")
    if extra:
        require(
            Path(extra).name == "site-packages" and Path(extra).is_dir(),
            "Invalid dependency supplement",
        )
        sys.path.append(extra)


def configure_cfg(
    cfg, plan, profile, make_adamw_optimizer, make_wsd_scheduler, checkpoint_root=None
):
    from datetime import timedelta
    from functools import partial

    world, pp, ep = profile
    require(profile in PROFILES, "Unsupported acceptance mesh")
    require(8 % (world // pp) == 0 and 8 // (world // pp) >= 2 * pp, "Invalid microbatch mesh")
    cfg.distributed.pipeline_parallel_size = pp
    cfg.distributed.expert_parallel_size = ep
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.hsdp_replica = 1
    cfg.distributed.timeout = timedelta(seconds=90)
    cfg.data.dataset = Path(plan["dataset"])
    cfg.data.format, cfg.data.modalities, cfg.data.num_workers = "prepared_bundle", ("text",), 0
    training = cfg.training
    training.model = Path(plan["model"])
    training.optimizer = make_adamw_optimizer
    training.scheduler = partial(
        make_wsd_scheduler, start_lr=1e-6, warmup_ratio=1.0, decay_ratio=0.0
    )
    training.max_steps, training.sequence_length = 32, 32
    training.global_batch_size, training.micro_batch_size = 8, 1
    training.seed, training.lr = 1234, 1e-5
    training.moe_load_balance_type, training.moe_load_balance_coef = "global-batch", 1e-3
    training.benchmark = training.fp8 = False
    if NUMERICAL_POLICY == COMMON_NUMERICAL_POLICY:
        require(hasattr(training, "deterministic"), "Common deterministic config is missing")
        training.deterministic = True
    training.save_interval = training.save_location = None
    if hasattr(training, "strict_resume"):
        training.strict_resume = False
    if checkpoint_root is not None:
        require(hasattr(training, "strict_resume"), "Candidate must support strict resume")
        training.strict_resume = True
        training.save_interval = 16
        training.save_location = Path(checkpoint_root)
    cfg.logging.wandb = None
    return cfg


def build_cfg(plan, profile, checkpoint_root=None):
    from pithtrain.modules.training import make_adamw_optimizer, make_wsd_scheduler
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg

    return configure_cfg(
        PretrainLMCfg(), plan, profile, make_adamw_optimizer, make_wsd_scheduler, checkpoint_root
    )


def verify_config_structure(repo, plan):
    """Check real source fields without importing CUDA-querying production operators."""
    from types import SimpleNamespace

    sections = {
        "training": ("pithtrain/modules/training.py", "TrainingCfg"),
        "distributed": ("pithtrain/modules/distributed.py", "DistributedCfg"),
        "data": ("pithtrain/modules/data_config.py", "DataCfg"),
        "logging": ("pithtrain/modules/logging.py", "LoggingCfg"),
    }
    fields = {}
    for key, (name, classname) in sections.items():
        tree = ast.parse((Path(repo) / name).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
        fields[key] = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    tree = ast.parse((Path(repo) / "pithtrain/tasks/pretrain_lm.py").read_text())
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    require({"setup_dataset", "train_step"} <= functions, "Native training/data interface missing")
    for profile in PROFILES:
        cfg = SimpleNamespace(
            **{key: SimpleNamespace(**dict.fromkeys(names)) for key, names in fields.items()}
        )
        configure_cfg(cfg, plan, profile, lambda: None, lambda: None)
        for key, names in fields.items():
            require(set(vars(getattr(cfg, key))) == names, f"Unknown native {key} field")
        require(cfg.training.max_steps == 32 and cfg.training.benchmark is False, "Wrong workload")
        if "strict_resume" in fields["training"]:
            configure_cfg(cfg, plan, profile, lambda: None, lambda: None, Path("new-checkpoints"))
            require(
                cfg.training.strict_resume is True
                and cfg.training.save_interval == 16
                and cfg.training.max_steps == 32,
                "Feature checkpoint intervention changed",
            )
        require(
            cfg.training.scheduler.keywords
            == dict(start_lr=1e-6, warmup_ratio=1.0, decay_ratio=0.0),
            "Wrong warmup",
        )
    return "AST_FIELDS_AND_SHARED_CONFIG_BUILDER_PASS; GPU_NATIVE_IMPORT_NOT_RUN"


def verify_dataset(dataset, model):
    import torch
    from transformers import AutoConfig

    from pithtrain.modules.dataset import MemmapDataset
    from pithtrain.modules.model_config import resolve_text_model_config
    from pithtrain.tasks.prepare_omni_data import processor_for, verify_bundle

    require(not torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    bundle = verify_bundle(dataset)
    require("text" in bundle["manifests"]["train"], "Real prepared text is required")
    config = resolve_text_model_config(AutoConfig.from_pretrained(model))
    require(config.model_type == "qwen3_omni_moe_text", "Expected native Omni Thinker text")
    _, thinker = processor_for(bundle["recipe"], offline=True)
    require(config.vocab_size >= thinker.text_config.vocab_size, "Processor vocabulary mismatch")
    shards = [MemmapDataset(p, 32) for p in sorted((dataset / "tokens/train").rglob("*.bin"))]
    samples = sum(map(len, shards))
    require(samples >= 32 * 8, f"Need 256 real samples per arm, found {samples}")
    require(
        all(0 <= int(s.tokens.min()) <= int(s.tokens.max()) < config.vocab_size for s in shards),
        "Out-of-vocabulary real token",
    )
    require(not torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    return dict(
        bundle_sha256=sha(dataset / "bundle.json"),
        checksums_sha256=sha(dataset / "checksums.sha256"),
        samples=samples,
        required_samples=256,
        model_sha256=sha(model / "config.json"),
    )


def probe(args):
    select_source(args.target_repo)
    import torch
    import transformers

    plan = dict(dataset=str(args.dataset.resolve()), model=str(args.model.resolve()))
    result = dict(
        source=source_identity(args.target_repo),
        input=verify_dataset(args.dataset, args.model),
        torch=torch.__version__,
        transformers=transformers.__version__,
        cuda_initialized=False,
        config_structure=verify_config_structure(args.target_repo, plan),
    )
    write_json(args.result, result)


def child_env(repo, *, cpu=False):
    env = os.environ.copy()
    # Permit only explicit dependency site-packages supplements, never pytest/attention adapters.
    env.update(
        PYTHONPATH=str(repo),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP"):
        env.pop(name, None)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def prepare(args):
    base, feature = source_identity(args.base_repo), source_identity(args.feature_repo)
    require(
        base["commit"] == BASE_SHA and feature["commit"] == FEATURE_SHA,
        "This experiment locks a0b14f7 -> d237e4c; do not substitute another comparison",
    )
    require(
        base["production_sha256"] != feature["production_sha256"],
        "Identical production code is a noise control, not this regression baseline",
    )
    require(not args.output.exists(), "Use a new output directory")
    roots = (args.dataset.resolve(), args.base_repo.resolve(), args.feature_repo.resolve(), REPO)
    require(
        all(not args.output.resolve().is_relative_to(p) for p in roots),
        "Evidence must be outside data/source trees",
    )
    args.output.mkdir(parents=True)
    model = args.output / "model"
    model.mkdir()
    raw = (REPO / "tests/configs/qwen3_omni_text/tiny.json").read_bytes()
    (model / "config.json").write_bytes(raw)
    for arm, source in (("base", base), ("feature", feature)):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "probe",
            "--target-repo",
            source["path"],
            "--dataset",
            str(args.dataset),
            "--model",
            str(model),
            "--result",
            str(args.output / f"{arm}-probe.json"),
        ]
        with (args.output / f"{arm}-probe.log").open("x") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env(source["path"], cpu=True),
                timeout=120,
            )
    probes = [
        json.loads((args.output / f"{arm}-probe.json").read_text()) for arm in ("base", "feature")
    ]
    require(probes[0]["input"] == probes[1]["input"], "Baseline/candidate data or model mismatch")
    require(
        (probes[0]["torch"], probes[0]["transformers"])
        == (probes[1]["torch"], probes[1]["transformers"]),
        "Runtime mismatch",
    )
    plan = dict(
        schema="omni_strict_bridge_noise_32_v2",
        bridge=BRIDGE,
        base=base,
        feature=feature,
        dataset=str(args.dataset.resolve()),
        model=str(model.resolve()),
        input=probes[0]["input"],
        settings=SETTINGS,
        profiles=[list(p) for p in PROFILES],
        arms=list(ARMS),
        driver_sha256=sha(__file__),
        comparator_sha256=sha(REPO / ".agents/skills/validate-correctness/scripts/compare.py"),
        cpu_verified=True,
        gpu_status="NOT_RUN",
        arm_timeout_seconds=600,
        total_timeout_seconds=3300,
    )
    write_json(args.output / "plan.json", plan)
    print("CPU_PREFLIGHT_PASS; 3 x 32 steps per mesh; GPU_NOT_RUN")


def parse_training_log(text, first=1, last=32):
    require(
        (first, last) in {(1, 32), (1, 16), (17, 32)},
        "Only full runs or the fixed bridge segments are valid",
    )
    records = []
    for line in text.splitlines():
        step = STEP.search(line)
        if not step:
            continue
        index, total = map(int, step.groups())
        require(
            total == 32 and index == len(records) + first, "Missing/duplicate/out-of-order step"
        )
        item = dict(step=index)
        for metric in (*METRICS, "learning-rate", "gradient-norm"):
            values = re.findall(rf"(?<![\w-]){metric} {NUMBER}(?=\s|$)", line)
            require(len(values) == 1, f"Missing/duplicate metric: {metric}")
            value = float(values[0])
            require(math.isfinite(value) and value >= 0, f"Invalid metric: {metric}")
            item[metric] = value
        expected_lr = 1e-6 + (1e-5 - 1e-6) * index / 32
        require(
            math.isclose(item["learning-rate"], expected_lr, rel_tol=1e-6), "Wrong warmup schedule"
        )
        records.append(item)
    require(
        len(records) == last - first + 1,
        f"Expected exactly steps {first}..{last}, found {len(records)}",
    )
    return records


def comparator():
    path = REPO / ".agents/skills/validate-correctness/scripts/compare.py"
    spec = importlib.util.spec_from_file_location("existing_noise_comparator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require((module.RATIO_WARN, module.RATIO_FAIL) == (3.0, 5.0), "Comparator thresholds changed")
    require(tuple(module.METRICS) == METRICS, "Comparator metric scope changed")
    return module


def compare_records(runs):
    require(len(runs) == 3 and all(len(r) == 32 for r in runs), "Three full arms are required")
    compare = comparator()
    result = {}
    for metric in METRICS:
        values = [[row[metric] for row in run] for run in runs]
        require(all(math.isfinite(x) and x >= 0 for v in values for x in v), "Nonfinite metric")
        envelope = compare.mean_delta(values[0], values[1])
        signal_value = compare.mean_delta(values[0], values[2])
        ratio = signal_value / envelope if envelope else (0.0 if signal_value == 0 else math.inf)
        status = (
            "PASS"
            if ratio < compare.RATIO_WARN
            else "INVESTIGATE"
            if ratio < compare.RATIO_FAIL
            else "FAIL"
        )
        result[metric] = dict(
            envelope=envelope,
            signal=signal_value,
            ratio=ratio if math.isfinite(ratio) else "infinity",
            status=status,
        )
    status = (
        "FAIL"
        if any(m["status"] == "FAIL" for m in result.values())
        else (
            "INVESTIGATE" if any(m["status"] == "INVESTIGATE" for m in result.values()) else "PASS"
        )
    )
    return dict(
        status=status,
        steps=32,
        metrics=result,
        step_1_forward=dict(
            envelope=abs(runs[0][0][METRICS[0]] - runs[1][0][METRICS[0]]),
            signal=abs(runs[0][0][METRICS[0]] - runs[2][0][METRICS[0]]),
            gated=False,
        ),
    )


def verify_plan(path):
    plan = json.loads(path.read_text())
    require(
        plan["schema"] == "omni_strict_bridge_noise_32_v2"
        and plan["settings"] == SETTINGS
        and plan["bridge"] == BRIDGE,
        "Workload changed",
    )
    require(
        plan["profiles"] == [list(p) for p in PROFILES] and plan["arms"] == list(ARMS),
        "Incomplete experiment",
    )
    require(plan["driver_sha256"] == sha(__file__), "Driver changed after preflight")
    require(
        plan["comparator_sha256"]
        == sha(REPO / ".agents/skills/validate-correctness/scripts/compare.py"),
        "Comparator changed",
    )
    require(
        plan["arm_timeout_seconds"] == 600 and plan["total_timeout_seconds"] == 3300,
        "Budget changed",
    )
    for key in ("base", "feature"):
        require(source_identity(plan[key]["path"]) == plan[key], f"{key} source changed")
    require(
        plan["base"]["commit"] == BASE_SHA and plan["feature"]["commit"] == FEATURE_SHA,
        "Comparison range changed",
    )
    require(
        sha(Path(plan["model"]) / "config.json") == plan["input"]["model_sha256"], "Model changed"
    )
    return plan


def tensor_digest(value):
    import numpy as np
    import torch

    h = hashlib.sha256()

    def block(value):
        h.update(len(value).to_bytes(8, "big"))
        h.update(value)

    def add(item):
        if isinstance(item, torch.Tensor):
            if hasattr(item, "to_local"):
                item = item.to_local()
            item = item.detach().cpu().contiguous()
            block(b"tensor")
            block(str((str(item.dtype), list(item.shape))).encode())
            block(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            block(b"numpy")
            block(str((str(item.dtype), list(item.shape))).encode())
            block(item.tobytes(order="C"))
        elif isinstance(item, dict):
            block(b"dict")
            block(str(len(item)).encode())
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            block(type(item).__name__.encode())
            block(str(len(item)).encode())
            for element in item:
                add(element)
        else:
            block(type(item).__name__.encode())
            block(repr(item).encode())

    add(value)
    return h.hexdigest()


class ObservedData:
    def __init__(self, source):
        self.source, self.batches = source, {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_batch(self, step, device):
        require(step not in self.batches, "A training batch was requested twice")
        batches = self.source.get_batch(step, device)
        self.batches[step] = tensor_digest(
            [
                dict(
                    inputs=m.model_inputs,
                    targets=m.objective_inputs,
                    cu_seqlens=m.cu_seqlens,
                    sample_ids=m.sample_ids,
                )
                for m in batches
            ]
        )
        return batches


class StepCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        message = record.getMessage()
        if STEP.search(message):
            self.lines.append(message)


def runtime_digest(data):
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.distributed_checkpoint import _pack, _runtime

    values, _ = _runtime(
        training.model,
        training.optimizers,
        training.schedulers,
        data.checkpoint_state,
        distributed.device,
    )
    return tensor_digest(_pack(values))


def arm(args):
    require(os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", "GPU scheduling opt-in required")
    plan = verify_plan(args.plan)
    profile = tuple(map(int, args.profile.split(",")))
    require(profile in PROFILES, "Unknown profile")
    require(
        args.segment in (("produce", "restore") if args.arm == "feat0" else ("full",)),
        "Only feat0 may perform the fixed step16 bridge; it cannot bypass it",
    )
    source = plan["feature" if args.arm == "feat0" else "base"]
    select_source(source["path"])
    import torch
    import torch.distributed as dist

    require(
        verify_dataset(Path(plan["dataset"]), Path(plan["model"])) == plan["input"], "Input changed"
    )
    from pithtrain.contexts import distributed, training
    from pithtrain.contexts import logging as log_context
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import setup_training
    from pithtrain.tasks import pretrain_lm

    require(
        Path(pretrain_lm.__file__).resolve().is_relative_to(Path(source["path"])),
        "Wrong code imported",
    )
    require(int(os.environ.get("WORLD_SIZE", "0")) == profile[0], "Wrong allocation world size")
    checkpoint_root = args.output / "checkpoints" if args.arm == "feat0" else None
    cfg = build_cfg(plan, profile, checkpoint_root)
    setup_logging(cfg)
    capture = StepCapture()
    log_context.stdout.addHandler(capture)
    setup_distributed(cfg)
    require("H100" in torch.cuda.get_device_name(), "Expected authorized H100 allocation")
    try:
        data = ObservedData(pretrain_lm.setup_dataset(cfg))
        setup_training(cfg)
        policy = runtime_policy_metadata()
        initial = tensor_digest(dict(training.model.named_parameters()))
        first, last = (
            (16, 32)
            if args.segment == "restore"
            else (0, 16)
            if args.segment == "produce"
            else (0, 32)
        )
        boundary_path = args.output / f"boundary-rank-{distributed.rank:05d}.json"
        boundary = None
        bridge = None
        if args.segment == "restore":
            from pithtrain.modules.checkpoint import load_checkpoint
            from pithtrain.modules.distributed_checkpoint import resume_contract

            boundary = json.loads(boundary_path.read_text())
            require_same_runtime_policy([policy, boundary])
            require(boundary["pid"] != os.getpid(), "Restore must run in a fresh process")
            require(
                boundary["source"] == source
                and boundary["input"] == plan["input"]
                and boundary["settings"] == SETTINGS,
                "Stale boundary metadata",
            )
            require(
                boundary["profile"] == list(profile) and boundary["rank"] == distributed.rank,
                "Boundary topology mismatch",
            )
            require(
                all(not opt.state for opt in training.optimizers), "Fresh optimizer must be empty"
            )
            load_checkpoint(
                args.output / "converted",
                16,
                data_state=data.checkpoint_state,
                strict=True,
                resume_config=resume_contract(cfg),
            )
            restored = runtime_digest(data)
            require(
                restored == boundary["runtime_sha256"],
                "Exact step16 state/RNG/cursor restoration failed",
            )
            require(data.consumed_samples == 128, "Wrong restored cursor")
            require(initial == boundary["initial_weights_sha256"], "Fresh initialization changed")
            bridge = dict(
                protocol=BRIDGE,
                producer_pid=boundary["pid"],
                restorer_pid=os.getpid(),
                before_sha256=boundary["runtime_sha256"],
                restored_sha256=restored,
                exact_boundary=True,
                strict_hf_strict=True,
                **policy,
            )
        gc.disable()
        for step in range(first, last):
            pretrain_lm.train_step(cfg, data, step)
        gc.enable()
        require_same_runtime_policy([policy, runtime_policy_metadata()])
        require(
            data.consumed_samples == last * 8 and list(data.batches) == list(range(first, last)),
            "Incomplete real-data consumption",
        )
        require(all(s.last_epoch == last for s in training.schedulers), "Scheduler skipped updates")
        require(
            all(
                opt.state and all(int(v["step"].item()) == last for v in opt.state.values())
                for opt in training.optimizers
            ),
            "Optimizer skipped updates",
        )
        records = (
            parse_training_log("\n".join(capture.lines), first + 1, last)
            if distributed.rank == 0
            else None
        )
        if args.segment == "produce":
            from pithtrain.modules.checkpoint import find_checkpoint

            require(
                find_checkpoint(checkpoint_root) == 16, "Native step16 strict checkpoint missing"
            )
            require(
                (checkpoint_root / "torch-dcp/00000016/strict-manifest.json").is_file(),
                "Strict publication missing",
            )
            write_json(
                boundary_path,
                dict(
                    source=source,
                    input=plan["input"],
                    settings=SETTINGS,
                    profile=list(profile),
                    rank=distributed.rank,
                    pid=os.getpid(),
                    runtime_sha256=runtime_digest(data),
                    initial_weights_sha256=initial,
                    records=records,
                    batches=data.batches,
                    **policy,
                ),
            )
            dist.barrier()
            return
        batches = {str(k): v for k, v in data.batches.items()}
        if boundary is not None:
            require(
                set(boundary["batches"]) == {str(n) for n in range(16)},
                "Incomplete prefix evidence",
            )
            batches = {**boundary["batches"], **batches}
            records = boundary["records"] + records if records is not None else None
        dist.barrier()
        write_json(
            args.output / f"rank-{distributed.rank:05d}.json",
            dict(
                status="COMPLETED_32_NATIVE_UPDATES",
                arm=args.arm,
                rank=distributed.rank,
                source=source,
                settings=SETTINGS,
                profile=list(profile),
                input=plan["input"],
                initial_weights_sha256=initial,
                batches=batches,
                consumed_samples=data.consumed_samples,
                records=records,
                bridge=bridge,
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),
                native_fa4=True,
                **policy,
            ),
        )
    finally:
        gc.enable()
        if dist.is_initialized():
            dist.destroy_process_group()


def validate_arm(plan, folder, profile, arm_name):
    expected_source = plan["feature" if arm_name == "feat0" else "base"]
    paths = sorted(folder.glob("rank-*.json"))
    require(
        [p.name for p in paths] == [f"rank-{r:05d}.json" for r in range(profile[0])],
        "Missing/unexpected rank report",
    )
    reports = [json.loads(p.read_text()) for p in paths]
    require_same_runtime_policy(reports)
    for rank, report in enumerate(reports):
        require(
            report["status"] == "COMPLETED_32_NATIVE_UPDATES"
            and report["rank"] == rank
            and report["arm"] == arm_name,
            "Wrong report",
        )
        require(
            report["source"] == expected_source
            and report["settings"] == SETTINGS
            and report["profile"] == list(profile),
            "Arm configuration mismatch",
        )
        require(
            report["input"] == plan["input"] and report["consumed_samples"] == 256,
            "Input/consumption mismatch",
        )
        require(set(report["batches"]) == {str(n) for n in range(32)}, "Missing batch evidence")
        require("H100" in report["gpu"] and report["native_fa4"] is True, "Wrong execution backend")
        bridge = report.get("bridge")
        if arm_name == "feat0":
            require(
                bridge
                and bridge["protocol"] == BRIDGE
                and bridge["exact_boundary"] is True
                and bridge["strict_hf_strict"] is True
                and bridge["producer_pid"] != bridge["restorer_pid"]
                and bridge["before_sha256"] == bridge["restored_sha256"],
                "Missing or failed strict/HF boundary",
            )
            require_same_runtime_policy([report, bridge])
        else:
            require(bridge is None, "Baseline must remain a continuous native run")
    records = parse_training_log((folder / "training.log").read_text())
    require(records == reports[0]["records"], "Live rank-zero metrics disagree with training log")
    return reports, records


def compare_profile(plan, folder, profile):
    arms = [validate_arm(plan, folder / name, profile, name) for name in ARMS]
    for rank in range(profile[0]):
        reports = [item[0][rank] for item in arms]
        require(
            all(r["batches"] == reports[0]["batches"] for r in reports),
            "Arms consumed different batches",
        )
        require(
            all(
                r["initial_weights_sha256"] == reports[0]["initial_weights_sha256"] for r in reports
            ),
            "Arms started from different weights",
        )
        require(all(r["torch"] == reports[0]["torch"] for r in reports), "Arm runtime mismatch")
    require_same_runtime_policy([report for reports, _ in arms for report in reports])
    return compare_records([item[1] for item in arms])


def run_command(command, log_path, env, timeout):
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    require(status == 0, f"Native arm failed with exit {status}; later arms NOT RUN")


def execute_arm(plan_path, plan, profile, arm_name, arm_output, deadline):
    target = plan["feature" if arm_name == "feat0" else "base"]["path"]
    arm_deadline = min(deadline, time.monotonic() + 600)

    def execute(command, filename, cpu=False):
        remaining = arm_deadline - time.monotonic()
        require(remaining > 0, "Arm/global budget exhausted; no shortened run")
        run_command(command, arm_output / filename, child_env(target, cpu=cpu), remaining)

    def native(segment, filename):
        execute(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={profile[0]}",
                str(Path(__file__).resolve()),
                "arm",
                "--plan",
                str(plan_path),
                "--profile",
                ",".join(map(str, profile)),
                "--arm",
                arm_name,
                "--segment",
                segment,
                "--output",
                str(arm_output),
            ],
            filename,
        )

    if arm_name != "feat0":
        native("full", "training.log")
        return
    native("produce", "produce.log")
    converter = str(Path(target) / "examples/convert_checkpoint/qwen3-omni-text/script.py")
    execute(
        [
            sys.executable,
            converter,
            "dcp2hf",
            str(arm_output / "checkpoints/torch-dcp/00000016"),
            str(arm_output / "hf"),
            "--expert-layout",
            "fused",
            "--prefix",
            "thinker.",
        ],
        "export.log",
        cpu=True,
    )
    execute(
        [
            sys.executable,
            converter,
            "hf2dcp",
            str(arm_output / "hf"),
            str(arm_output / "converted/torch-dcp/00000016"),
            "--restore-resume",
            "--prefix",
            "thinker.",
        ],
        "import.log",
        cpu=True,
    )
    native("restore", "restore.log")
    # This is one logical 32-step feature arm; either missing half fails strict parsing.
    with (arm_output / "training.log").open("x") as log:
        log.write(
            (arm_output / "produce.log").read_text()
            + "\n"
            + (arm_output / "restore.log").read_text()
        )


def run(args):
    require(os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", "GPU scheduling opt-in required")
    plan = verify_plan(args.plan)
    output = args.plan.parent / "runs"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    results = {}
    arm_status = {f"w{w}-pp{pp}-ep{ep}/{a}": "NOT_RUN" for w, pp, ep in PROFILES for a in ARMS}
    status = "INCOMPLETE"
    error = None
    current_arm = None

    def interrupted(signum, _frame):
        raise InterruptedError(f"Signal {signum}; no automatic retry")

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        for profile in PROFILES:
            world, pp, ep = profile
            name = f"w{world}-pp{pp}-ep{ep}"
            folder = output / name
            folder.mkdir()
            for arm_name in ARMS:
                remaining = 3300 - (time.monotonic() - started)
                require(remaining > 0, "55-minute budget exhausted; remaining arms NOT RUN")
                arm_output = folder / arm_name
                arm_output.mkdir()
                current_arm = f"{name}/{arm_name}"
                arm_status[current_arm] = "RUNNING"
                execute_arm(args.plan, plan, profile, arm_name, arm_output, started + 3300)
                validate_arm(plan, arm_output, profile, arm_name)
                arm_status[current_arm] = "COMPLETED_32_NATIVE_UPDATES"
            result = compare_profile(plan, folder, profile)
            write_json(folder / "comparison.json", result)
            results[name] = result
            if result["status"] != "PASS":
                status = result["status"]
                break
        else:
            status = "PASS"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        status = "FAIL_OR_INCOMPLETE"
        if current_arm and arm_status[current_arm] == "RUNNING":
            arm_status[current_arm] = "FAILED_OR_INTERRUPTED"
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        write_json(
            output / "summary.json",
            dict(
                status=status,
                profiles=results,
                error=error,
                arms=arm_status,
                remaining_profiles="NOT_RUN" if len(results) < len(PROFILES) else None,
                source=plan,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
    print(status)
    return 0 if status == "PASS" else 2 if status == "INVESTIGATE" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = commands.add_parser("prepare", help="CPU-only full workload preflight")
    for name in ("base-repo", "feature-repo", "dataset", "output"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    probe_parser = commands.add_parser("probe", help=argparse.SUPPRESS)
    for name in ("target-repo", "dataset", "model", "result"):
        probe_parser.add_argument("--" + name, type=Path, required=True)
    run_parser = commands.add_parser(
        "run", help="Run all three arms sequentially in an existing allocation"
    )
    run_parser.add_argument("--plan", type=Path, required=True)
    arm_parser = commands.add_parser("arm", help=argparse.SUPPRESS)
    arm_parser.add_argument("--plan", type=Path, required=True)
    arm_parser.add_argument("--profile", required=True)
    arm_parser.add_argument("--arm", choices=ARMS, required=True)
    arm_parser.add_argument("--segment", choices=("full", "produce", "restore"), required=True)
    arm_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    result = {"prepare": prepare, "probe": probe, "run": run, "arm": arm}[args.mode](args)
    return result or 0


if __name__ == "__main__":
    raise SystemExit(main())
