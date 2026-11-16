"""Released Thinker under the documented native-multimodal recipe, with strict resume.

Validation row 23. The configuration is the one
examples/pretrain_lm/qwen3-omni-native-multimodal/script.py builds: its main() runs
with pretrain_lm.launch replaced by a capture, so every recipe value
(native_multimodal; strict_resume, which also enables the deterministic policies;
four mixed modalities; sequence 2048; the production AdamW factory; constant LR
1e-4; seed 1234; a save every two steps) is the recipe's own. Only run-size
arguments are chosen: PP2/EP4 on eight GPUs, global batch 16 (DualPipeV needs
2*PP microbatches per data rank) and --steps 4, of which three run, so the
recipe's train_step saves once, at step two, through its own strict path.

``produce`` follows launch: setup, model-only initialization from the prepared
released DCP, tune_kernels where launch calls it, and the recipe's train_step with
all-parameter observation. ``restore`` is a fresh process: find_checkpoint must
report step two and the strict load must reproduce the boundary state; step three
must reproduce its input batch and its complete post-update state bitwise. Loss
values and parameter deltas are diagnostics. No download, allocation or upload.
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tests.acceptance import omni_released_continuation as continuation
from tests.acceptance import omni_released_update_support as support

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "examples/pretrain_lm/qwen3-omni-native-multimodal/script.py"
SCHEMA = "qwen3_omni_released_default_recipe_v1"
MODALITIES = ("text", "image", "audio", "video")
WORLD, PP, EP = 8, 2, 4
GLOBAL_BATCH = 16
CONFIGURED_STEPS = 4
RUN_STEPS = 3
SAVE_STEP = 2
VISUAL, VISUAL_KEYS = support.ENCODERS["image"]
AUDIO, AUDIO_KEYS = support.ENCODERS["audio"]
SECONDS = dict(initialization=600, save=900, load=900, update=600, snapshot=900)
# Every value the recipe itself fixes, as documented in its README and script.
DOCUMENTED = dict(
    native_multimodal=True,
    strict_resume=True,
    deterministic_field=False,
    optimizer="make_adamw_optimizer",
    scheduler="make_constant_scheduler",
    lr=1e-4,
    seed=1234,
    micro_batch_size=1,
    sequence_length=2048,
    fp8=False,
    moe_load_balance_coef=0.0,
    save_interval=2,
    data_format="prepared_bundle",
    modalities=list(MODALITIES),
    sampling_weights=None,
    num_workers=0,
    cp=1,
    initialize_step=0,
)


def recipe_argv(*, dataset, model, checkpoint, epoch_samples, initialize_from=None):
    argv = [
        "--dataset",
        str(dataset),
        "--model",
        str(model),
        "--checkpoint",
        str(checkpoint),
        "--epoch-samples",
        str(epoch_samples),
        "--global-batch-size",
        str(GLOBAL_BATCH),
        "--steps",
        str(CONFIGURED_STEPS),
        "--pp",
        str(PP),
        "--ep",
        str(EP),
    ]
    if initialize_from is not None:
        argv += ["--initialize-from", str(initialize_from)]
    return argv


def recipe_cfg(argv):
    """Run the recipe's own main() and capture the cfg it would launch."""
    from pithtrain.tasks import pretrain_lm

    spec = importlib.util.spec_from_file_location("omni_native_multimodal_recipe", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured, launch, saved = [], pretrain_lm.launch, sys.argv
    pretrain_lm.launch, sys.argv = captured.append, [str(RECIPE), *argv]
    try:
        module.main()
    finally:
        pretrain_lm.launch, sys.argv = launch, saved
    if len(captured) != 1:
        raise AssertionError("The recipe must call launch exactly once")
    return captured[0]


def recipe_settings(cfg):
    t, d = cfg.training, cfg.distributed
    return dict(
        native_multimodal=t.native_multimodal,
        strict_resume=t.strict_resume,
        deterministic_field=t.deterministic,
        optimizer=t.optimizer.__name__,
        scheduler=t.scheduler.__name__,
        lr=t.lr,
        seed=t.seed,
        max_steps=t.max_steps,
        micro_batch_size=t.micro_batch_size,
        global_batch_size=t.global_batch_size,
        sequence_length=t.sequence_length,
        fp8=t.fp8,
        moe_load_balance_coef=t.moe_load_balance_coef,
        save_interval=t.save_interval,
        save_location=str(t.save_location),
        model=str(t.model),
        dataset=str(cfg.data.dataset),
        data_format=cfg.data.format,
        modalities=list(cfg.data.modalities),
        sampling_weights=cfg.data.sampling_weights,
        epoch_samples=cfg.data.epoch_samples,
        num_workers=cfg.data.num_workers,
        pp=d.pipeline_parallel_size,
        cp=d.context_parallel_size,
        ep=d.expert_parallel_size,
        initialize_from=None if cfg.initialize_from is None else str(cfg.initialize_from),
        initialize_step=cfg.initialize_step,
    )


def check_recipe(settings, *, dataset, model, checkpoint, epoch_samples, initialize_from):
    """The documented recipe values plus exactly the chosen run-size arguments."""
    expected = dict(
        DOCUMENTED,
        max_steps=CONFIGURED_STEPS,
        global_batch_size=GLOBAL_BATCH,
        pp=PP,
        ep=EP,
        save_location=str(checkpoint),
        model=str(model),
        dataset=str(dataset),
        epoch_samples=epoch_samples,
        initialize_from=None if initialize_from is None else str(initialize_from),
    )
    if set(settings) != set(expected):
        raise AssertionError("Recipe settings inventory differs")
    differences = {k: [settings[k], v] for k, v in expected.items() if settings[k] != v}
    if differences:
        raise AssertionError(f"Recipe configuration differs: {differences}")
    return expected


def estimate():
    """omni_released_update_support.estimate with both encoders on the heavy stage."""
    encoders = VISUAL + AUDIO
    heavy = 24 * support.NONEXPERT_LAYER + support.EDGE + encoders
    light = 24 * support.NONEXPERT_LAYER
    result = dict(
        parameters=support.TEXT + encoders,
        parameter_keys=531 + VISUAL_KEYS + AUDIO_KEYS,
        topology=dict(world=WORLD, pp=PP, ep=EP, cp=1, attention_dp=WORLD // PP),
        microbatches_per_attention_dp_rank=GLOBAL_BATCH // (WORLD // PP),
        checkpoint_bytes=12 * (support.TEXT + encoders),
    )
    for pp, nonexpert in ((0, heavy), (1, light)):
        shard = support.EXPERTS // 8 + nonexpert // 4
        unsharded = support.EXPERTS // 8 + nonexpert
        result[f"pp{pp}_sharded_elements"] = shard
        result[f"pp{pp}_initial_free_gpu_admission_bytes"] = (
            20 * shard + 6 * unsharded + 8 * support.GIB
        )
    return result


def plan():
    return dict(
        schema=SCHEMA,
        status="NOT_RUN",
        row=23,
        recipe=str(RECIPE.relative_to(REPO)),
        documented=DOCUMENTED,
        run_size=dict(
            pp=PP,
            ep=EP,
            world=WORLD,
            global_batch_size=GLOBAL_BATCH,
            configured_steps=CONFIGURED_STEPS,
            executed_steps=RUN_STEPS,
            save_step=SAVE_STEP,
        ),
        weights="pinned public released Thinker from the reviewed R0.2 prepared DCP",
        criteria=[
            "the recipe's own cfg, unchanged except the run-size arguments",
            "every parameter: finite FP32 gradient at every update; each component group with"
            " input in that global batch has a nonzero gradient and changed values",
            "Adam step equals the update count for every parameter",
            "restore: find_checkpoint is step 2; boundary state bitwise equal",
            "restore: step-3 input batch and complete post-update state bitwise equal",
        ],
        diagnostics=["objective loss sums", "per-parameter norms and digests", "tuned kernels"],
        estimate=estimate(),
    )


def multimodal_groups(config):
    """Component prefixes of both encoders and the text decoder, each prefix once."""
    groups = support.component_groups(config, "image") + support.component_groups(config, "audio")
    return tuple(dict.fromkeys(groups))


def required_groups(groups, present):
    """Encoder groups need a nonzero gradient only when that global batch fed them."""
    return tuple(
        prefix
        for prefix in groups
        if not (prefix.startswith("visual.") and not present["visual"])
        and not (prefix.startswith("audio_tower.") and not present["audio"])
    )


def media_presence(microbatches):
    present = dict(visual=False, audio=False)
    for microbatch in microbatches:
        inputs = microbatch.media_inputs or {}
        present["visual"] |= any("pixel" in key and inputs[key] is not None for key in inputs)
        present["audio"] |= inputs.get("input_features") is not None
    return present


def update_rows(rows, groups, required, expected_step):
    """omni_released_update_support.validate_rows for any update and data-dependent groups."""
    errors = []
    summary = {
        prefix: dict(
            elements=0, gradient_nonzero_elements=0, changed_parameter_shards=0, parameter_shards=0
        )
        for prefix in groups
    }
    for rank in rows:
        for name, item in rank["parameters"].items():
            owners = [prefix for prefix in groups if name.startswith(prefix)]
            if len(owners) != 1:
                errors.append(f"rank {rank['rank']}: missing/ambiguous component {name}")
                continue
            result = summary[owners[0]]
            result["elements"] += item["before"]["elements"]
            result["parameter_shards"] += 1
            grad = item.get("gradient")
            if grad is None or not grad["finite"] or grad["dtype"] != "torch.float32":
                errors.append(f"rank {rank['rank']}: missing/nonfinite/non-FP32 gradient {name}")
            else:
                result["gradient_nonzero_elements"] += grad["nonzero_elements"]
            if not item["before"]["finite"] or not item.get("after", {}).get("finite", False):
                errors.append(f"rank {rank['rank']}: nonfinite/missing parameter values {name}")
            if item.get("changed"):
                result["changed_parameter_shards"] += 1
            state = item.get("optimizer_state", {})
            if state.get("step") != expected_step:
                errors.append(f"rank {rank['rank']}: Adam step is not {expected_step} for {name}")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if value is None or not value["finite"] or value["dtype"] != "torch.float32":
                    errors.append(f"rank {rank['rank']}: invalid FP32 Adam {key} for {name}")
    for prefix, result in summary.items():
        if result["elements"] <= 0:
            errors.append(f"Component without parameters {prefix}")
        elif prefix in required and (
            result["gradient_nonzero_elements"] <= 0 or result["changed_parameter_shards"] <= 0
        ):
            errors.append(f"Missing/zero gradient/no actual update component {prefix}")
    return summary, errors


def observed_step(cfg, data, step, parameters, groups, output, seconds):
    """The recipe's train_step with all-parameter rows; saving happens inside it."""
    import torch.distributed as dist

    from pithtrain.contexts import distributed, training
    from pithtrain.tasks import pretrain_lm

    rank, world = dist.get_rank(), dist.get_world_size()
    report = dict(status="RUNNING", step=step + 1, rank=rank, parameters={}, errors=[])
    optimizer = training.optimizers[0]
    old_step, old_batch = training.model.step, data.get_batch
    requested, presence = [], dict(visual=False, audio=False)
    expected_outputs = cfg.training.global_batch_size // (
        cfg.training.micro_batch_size * distributed.dp_size
    )

    def batch(data_step, device):
        result = old_batch(data_step, device)
        requested.append(data_step)
        presence.update(media_presence(result))
        continuation.write_rows(
            output / f"batch-{step}-{rank}",
            continuation.state_rows(
                [
                    dict(
                        model_inputs=m.model_inputs,
                        objective_inputs=m.objective_inputs,
                        media_context=m.media_context,
                        media_inputs=m.media_inputs,
                        cu_seqlens=m.cu_seqlens,
                        sample_ids=m.sample_ids,
                    )
                    for m in result
                ]
            ),
        )
        return result

    def forward(*args, **kwargs):
        values = old_step(*args, **kwargs)
        report["objective_loss_sums"] = [float(v.detach().item()) for v in values]
        assert len(values) == (expected_outputs if distributed.pp_rank == 0 else 0)
        assert all(math.isfinite(v) for v in report["objective_loss_sums"])
        return values

    def before(opt, _args, _kwargs):
        assert bool(opt.state) == (step > 0)
        failures = []
        for name, p in parameters.items():
            grad = support.scan_tensor(p.grad) if p.grad is not None else None
            value = support.scan_tensor(p, digest=True)
            report["parameters"][name] = dict(before=value, gradient=grad)
            if (
                not value["finite"]
                or value["dtype"] != "torch.float32"
                or grad is None
                or not grad["finite"]
                or grad["dtype"] != "torch.float32"
            ):
                failures.append(name)
        everyone = [None] * world
        dist.all_gather_object(everyone, failures)
        assert not any(everyone), everyone

    def after(opt, _args, _kwargs):
        for name, p in parameters.items():
            row = report["parameters"][name]
            row["after"] = support.scan_tensor(p, digest=True)
            row["changed"] = row["before"]["sha256"] != row["after"]["sha256"]
            state = opt.state.get(p, {})
            row["optimizer_state"] = dict(
                step=float(state["step"].item()) if "step" in state else None,
                **{
                    k: support.scan_tensor(state[k])
                    for k in ("exp_avg", "exp_avg_sq")
                    if k in state
                },
            )

    pre, post = optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)
    training.model.step, data.get_batch = forward, batch
    try:
        with continuation.deadline(seconds):
            pretrain_lm.train_step(cfg, data, step)
        assert requested == [step], requested
        assert all(p.grad is None for p in parameters.values())
        gathered = [None] * world
        dist.all_gather_object(
            gathered, dict(rank=rank, parameters=report["parameters"], presence=presence)
        )
        present = {k: any(item["presence"][k] for item in gathered) for k in ("visual", "audio")}
        report["global_media_presence"] = present
        required = required_groups(groups, present)
        report["components"], report["errors"] = update_rows(gathered, groups, required, step + 1)
        assert not report["errors"], report["errors"][:20]
        report["status"] = "PASS"
    finally:
        pre.remove()
        post.remove()
        training.model.step, data.get_batch = old_step, old_batch
        rows = report.pop("parameters")
        continuation.write_rows(
            output / f"update-{step}-{rank}", (dict(key=k, **v) for k, v in rows.items())
        )
        continuation.write_json(output / f"update-{step}-{rank}.json", report)
    return report


def kernel_choices(chosen):
    return sorted([repr(key), repr(value[0]), repr(value[1])] for key, value in chosen.items())


def prepared_identity(args):
    """R0.2's reviewed prepare output; its producer commit may precede this head."""
    if continuation.file_sha(args.prepared / "prepared.json") != args.prepared_sha256:
        raise AssertionError("Prepared receipt checksum differs")
    prepared = continuation.read_json(args.prepared / "prepared.json")
    if (
        prepared["status"] != "PASS_RELEASED_THINKER_PREPARE_ONLY"
        or prepared["revision"] != support.REVISION
        or prepared["config_sha256"] != support.CONFIG_SHA
        or prepared["source_commit"] != args.prepared_commit
    ):
        raise AssertionError("Prepared receipt identity differs")
    if (
        continuation.file_sha(args.prepared / "expected/index.json")
        != prepared["expected_index_sha256"]
    ):
        raise AssertionError("Prepared expected index differs")
    # The converter, model, training and recipe code must be byte-identical to the
    # commit that produced the DCP; only test files may differ.
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", args.prepared_commit, "HEAD", "--", "pithtrain", "examples"],
        cwd=REPO,
        text=True,
    ).split()
    if changed:
        raise AssertionError(f"Production code changed since the prepared DCP: {changed}")
    return prepared


def run_gpu(args):
    import gc
    import importlib.metadata
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from pithtrain.contexts import training
    from pithtrain.kernels import tune_kernels
    from pithtrain.modules.checkpoint import (
        find_checkpoint,
        initialize_checkpoint,
        to_canonical_model,
    )
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.distributed_checkpoint import load_checkpoint, resume_contract
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import setup_training
    from pithtrain.tasks import pretrain_lm
    from tests.acceptance.omni_full_media import full_inventory, official_config

    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1"
    assert importlib.metadata.version("transformers") == "5.17.0"
    assert importlib.metadata.version("safetensors") == "0.8.0"
    assert torch.__version__ == "2.13.0+cu130" and torch.get_default_dtype() == torch.float32
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8")
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == WORLD and 0 <= rank < WORLD
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    assert "H200" in torch.cuda.get_device_name()
    continuation.source_identity(args.expected_commit)
    prepared = prepared_identity(args)
    work = Path(prepared["work_directory"])
    assert continuation.file_sha(work / "model/config.json") == support.CONFIG_SHA
    for path in (args.output, args.checkpoint, args.tuning_dir):
        assert not path.resolve().is_relative_to(work.resolve())
    expected_all = {row["key"]: row for row in continuation.read_rows(args.prepared / "expected")}
    plan_values = estimate()
    if args.phase == "produce":
        assert not args.checkpoint.exists()
        shapes = {k: v["shape"] for k, v in expected_all.items()}
        observed = continuation.inspect_initialization_dcp(work / "initialization", shapes)
        assert observed == prepared["dcp"]
        free = shutil.disk_usage(args.checkpoint.parent).free
        assert free >= math.ceil(1.1 * plan_values["checkpoint_bytes"]), free
        assert not args.tuning_dir.exists(), "Use a new job-owned kernel tuning directory"
    else:
        assert args.tuning_dir.is_dir(), "Restore reuses the producer's tuning directory"
    resources = support.resource_inventory()
    assert not support.resource_errors(resources), support.resource_errors(resources)
    stage = 0 if rank < WORLD // PP else 1
    assert (
        torch.cuda.mem_get_info()[0] >= plan_values[f"pp{stage}_initial_free_gpu_admission_bytes"]
    )
    os.environ["PITHTRAIN_TUNING_DIR"] = str(args.tuning_dir)
    argv = recipe_argv(
        dataset=args.dataset,
        model=work / "model",
        checkpoint=args.checkpoint,
        epoch_samples=args.epoch_samples,
        initialize_from=work / "initialization" if args.phase == "produce" else None,
    )
    cfg = recipe_cfg(argv)
    settings = recipe_settings(cfg)
    check_recipe(
        settings,
        dataset=args.dataset,
        model=work / "model",
        checkpoint=args.checkpoint,
        epoch_samples=args.epoch_samples,
        initialize_from=work / "initialization" if args.phase == "produce" else None,
    )
    # Infrastructure only: owner-only I/O must fit while the peers wait.
    cfg.distributed.timeout = timedelta(
        seconds=continuation.collective_timeout_seconds(
            args.phase, args.initialization_seconds, args.save_seconds, args.load_seconds
        )
    )
    setup_logging(cfg)
    setup_distributed(cfg)
    if rank == 0:
        args.output.mkdir()
        if args.phase == "produce":
            args.tuning_dir.mkdir()
    dist.barrier()
    timings = {}
    continuation.write_json(
        args.output / f"runtime-{rank}.json",
        dict(
            source_commit=args.expected_commit,
            prepared_commit=args.prepared_commit,
            recipe_sha256=continuation.file_sha(RECIPE),
            recipe_argv=argv,
            recipe_settings=settings,
            rank=rank,
            gpu_name=torch.cuda.get_device_name(),
            host=resources,
            distributions={
                d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
            },
            torch_cuda=torch.version.cuda,
            torch_git=torch.version.git_version,
            tuning_dir=str(args.tuning_dir),
            estimate=plan_values,
        ),
    )
    data = pretrain_lm.setup_dataset(cfg)
    statistics = data.bundle["statistics"]["train"]
    assert all(statistics[m]["samples"] > 0 for m in MODALITIES), statistics
    assert data.epoch_samples >= RUN_STEPS * GLOBAL_BATCH
    data_contract = dict(
        data_configuration=cfg.data.to_json_dict(),
        recipe=data.recipe,
        bundle_sha256=continuation.file_sha(args.dataset / "bundle.json"),
        fingerprint=data.data_fingerprint,
        epoch_samples=data.epoch_samples,
        sampler_seed=data.seed,
        weights=data.weights,
        modalities=list(data.modalities),
        source_records={m: statistics[m]["samples"] for m in MODALITIES},
        sequence_contract=continuation.validate_dataset_sequence_length(args.dataset, 2048),
        records_fabricated=0,
    )
    if rank == 0:
        continuation.write_json(args.output / "data-contract.json", data_contract)
    config = official_config(work / "model/config.json", REPO)
    shapes = {**full_inventory(config, "image"), **full_inventory(config, "audio")}
    groups = multimodal_groups(config)
    started = time.monotonic()
    with continuation.deadline(args.initialization_seconds):
        setup_training(cfg)
        parameters = {}
        for module in training.model.module:
            for name, value in module.named_parameters():
                key = support.canonical(name)
                assert key not in parameters and key in shapes, key
                shape = list(shapes[key])
                if ".experts." in key:
                    shape[0] //= EP
                assert list(value.shape) == shape, key
                parameters[key] = value
        inventories = [None] * WORLD
        dist.all_gather_object(
            inventories,
            dict(
                keys=sorted(parameters),
                elements=sum(support.local_tensor(p).numel() for p in parameters.values()),
            ),
        )
        assert set().union(*(set(v["keys"]) for v in inventories)) == set(shapes)
        assert len(shapes) == plan_values["parameter_keys"]
        assert sum(v["elements"] for v in inventories) == plan_values["parameters"]
        assert len(training.optimizers) == 1
        support.inspect_production_adamw(training.optimizers[0], parameters, expected_lr=1e-4)
        local_keys = set(to_canonical_model(training.model.state_dict(), training.model))
        all_keys = [None] * WORLD
        dist.all_gather_object(all_keys, sorted(local_keys))
        assert set().union(*(set(v) for v in all_keys)) == set(expected_all)
        if args.phase == "produce":
            assert find_checkpoint(cfg.training.save_location) is None
            initialize_checkpoint(cfg.initialize_from, cfg.initialize_step)
            assert not training.optimizers[0].state
            continuation.verify_loaded_parameters(expected_all, args.output / f"initialized-{rank}")
    timings["initialization"] = time.monotonic() - started
    start = 0
    if args.phase == "restore":
        validate_producer(args, rank)
        step = find_checkpoint(cfg.training.save_location)
        assert step == SAVE_STEP, step
        started = time.monotonic()
        with continuation.deadline(args.load_seconds):
            load_checkpoint(
                cfg.training.save_location,
                step,
                data_fingerprint=data.data_fingerprint,
                strict=True,
                resume_config=resume_contract(cfg),
            )
        timings["load"] = time.monotonic() - started
        with continuation.deadline(args.snapshot_seconds):
            rows = list(continuation.state_rows(continuation.capture_state(data, device)))
        continuation.compare_rows(rows, continuation.read_rows(args.producer / f"boundary-{rank}"))
        continuation.write_rows(args.output / f"boundary-{rank}", rows)
        start = SAVE_STEP
    # launch's order: freeze the startup heap, tune before the first step and after it.
    gc.collect()
    gc.freeze()
    gc.disable()
    kernels = dict(before_first_step=kernel_choices(tune_kernels()))
    try:
        for step in range(start, RUN_STEPS):
            saves = step + 1 == SAVE_STEP
            seconds = args.update_seconds + (args.save_seconds if saves else 0)
            started = time.monotonic()
            observed_step(cfg, data, step, parameters, groups, args.output, seconds)
            timings[f"step-{step + 1}"] = time.monotonic() - started
            if step == start:
                kernels["after_first_step"] = kernel_choices(tune_kernels())
            if args.phase == "produce" and saves:
                assert find_checkpoint(cfg.training.save_location) == SAVE_STEP
                with continuation.deadline(args.snapshot_seconds):
                    continuation.write_rows(
                        args.output / f"boundary-{rank}",
                        continuation.state_rows(continuation.capture_state(data, device)),
                    )
    finally:
        gc.enable()
    assert find_checkpoint(cfg.training.save_location) == SAVE_STEP
    with continuation.deadline(args.snapshot_seconds):
        rows = list(continuation.state_rows(continuation.capture_state(data, device)))
    if args.phase == "restore":
        continuation.compare_rows(rows, continuation.read_rows(args.producer / f"after-{rank}"))
        continuation.compare_rows(
            continuation.read_rows(args.output / f"batch-{SAVE_STEP}-{rank}"),
            continuation.read_rows(args.producer / f"batch-{SAVE_STEP}-{rank}"),
        )
    continuation.write_rows(args.output / f"after-{rank}", rows)
    continuation.write_json(
        args.output / f"rank-{rank}.json",
        dict(
            status="PASS",
            rank=rank,
            phase=args.phase,
            source_commit=args.expected_commit,
            prepared_sha256=args.prepared_sha256,
            pid=os.getpid(),
            process_start=Path("/proc/self/stat").read_text().split()[21],
            updates=list(range(start + 1, RUN_STEPS + 1)),
            kernels=kernels,
            timings=timings,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    dist.barrier()
    if rank == 0:
        files = [
            dict(path=str(p.relative_to(args.output)), sha256=continuation.file_sha(p))
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        ]
        continuation.write_json(
            args.output / "complete.json",
            dict(
                status="PASS_RELEASED_DEFAULT_RECIPE_PRODUCER"
                if args.phase == "produce"
                else "PASS_RELEASED_DEFAULT_RECIPE_FRESH_RESTORE",
                source_commit=args.expected_commit,
                prepared_sha256=args.prepared_sha256,
                recipe_sha256=continuation.file_sha(RECIPE),
                files=files,
                hf_parity="SEPARATE_GATE_ROW_17",
            ),
        )
    dist.barrier()


def validate_producer(args, rank):
    if continuation.file_sha(args.producer / "complete.json") != args.producer_sha256:
        raise AssertionError("Producer completion checksum differs")
    receipt = continuation.read_json(args.producer / "complete.json")
    assert receipt["status"] == "PASS_RELEASED_DEFAULT_RECIPE_PRODUCER"
    assert receipt["source_commit"] == args.expected_commit
    assert receipt["prepared_sha256"] == args.prepared_sha256
    assert receipt["recipe_sha256"] == continuation.file_sha(RECIPE)
    for item in receipt["files"]:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert continuation.file_sha(args.producer / relative) == item["sha256"]
    prior = continuation.read_json(args.producer / f"rank-{rank}.json")
    identity = (os.getpid(), Path("/proc/self/stat").read_text().split()[21])
    assert identity != (prior["pid"], prior["process_start"]), "Restore must use a fresh process"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("phase", choices=("plan", "produce", "restore"))
    result.add_argument("--expected-commit")
    result.add_argument("--prepared", type=Path)
    result.add_argument("--prepared-sha256")
    result.add_argument("--prepared-commit")
    result.add_argument("--dataset", type=Path)
    result.add_argument("--epoch-samples", type=int)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--tuning-dir", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--producer", type=Path)
    result.add_argument("--producer-sha256")
    for key, value in SECONDS.items():
        result.add_argument(f"--{key}-seconds", type=int, default=value)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.phase == "plan":
        print(json.dumps(plan(), indent=2))
        return
    required = (
        "expected_commit",
        "prepared",
        "prepared_sha256",
        "prepared_commit",
        "dataset",
        "epoch_samples",
        "checkpoint",
        "tuning_dir",
        "output",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if args.phase == "restore":
        missing += [n for n in ("producer", "producer_sha256") if getattr(args, n) is None]
    elif args.producer is not None or args.producer_sha256 is not None:
        missing.append("no producer for produce")
    if missing:
        parser().error(f"Missing or invalid arguments: {missing}")
    if args.epoch_samples % GLOBAL_BATCH or args.epoch_samples < RUN_STEPS * GLOBAL_BATCH:
        parser().error("--epoch-samples must be a multiple of 16 and at least 48")
    for path in (args.checkpoint, args.tuning_dir, args.output):
        if not path.is_absolute():
            parser().error("Checkpoint, tuning and output paths must be absolute")
    run_gpu(args)


if __name__ == "__main__":
    main()
