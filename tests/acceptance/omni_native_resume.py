"""Native Omni update/restart acceptance over PR100's real prepared-text loader.

Only preflight runs on CPU. Other modes require explicit GPU opt-in and a fresh
process. This covers phase=-1 (unsharded native model), not DualPipeV/FSDP.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "omni_native_resume_v1"
PROTOCOL = "save_then_runtime_state_v1"
SETTINGS = dict(
    sequence_length=32, global_batch_size=4, micro_batch_size=1, max_steps=3, seed=1234, lr=1e-4
)
SOURCE_FILES = (
    "tests/acceptance/omni_native_resume.py",
    "tests/configs/qwen3_omni_text/tiny.json",
    "pithtrain/modules/checkpoint.py",
    "pithtrain/modules/strict_checkpoint.py",
    "pithtrain/modules/training_data.py",
    "pithtrain/modules/dataset.py",
    "pithtrain/modules/model_config.py",
    "pithtrain/modules/training.py",
    "pithtrain/modules/data_config.py",
    "pithtrain/tasks/pretrain_lm.py",
    "pithtrain/tasks/prepare_omni_data.py",
    "pithtrain/models/qwen3_omni_moe.py",
    "pithtrain/tasks/convert_checkpoint/qwen3_omni_moe.py",
)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def provenance():
    return dict(
        commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        files={name: sha(REPO / name) for name in SOURCE_FILES},
    )


def equal(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        torch.testing.assert_close(
            actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0, msg=path
        )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            equal(left, right, f"{path}.{index}")
    else:
        assert type(actual) is type(expected) and actual == expected, path


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def difference(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    left, right = actual.double(), expected.double()
    denominator = (left.square() + right.square()).sum()
    return 0.0 if denominator == 0 else ((left - right).square().sum() / denominator).item()


def compare_named(actual, expected, threshold, label):
    assert actual.keys() == expected.keys(), f"{label}: missing/unexpected parameter names"
    errors = {}
    for name in actual:
        assert actual[name] is not None and expected[name] is not None, f"{label}: missing {name}"
        errors[name] = difference(actual[name], expected[name])
        assert errors[name] < threshold, f"{label}: {name}: error={errors[name]} >= {threshold}"
    assert errors, f"{label}: no parameters compared"
    return errors


def verify_input(root):
    from pithtrain.tasks.prepare_omni_data import verify_bundle

    bundle = verify_bundle(root)
    assert "text" in bundle["manifests"]["train"], "Prepared text is required"
    return dict(bundle=sha(root / "bundle.json"), checksums=sha(root / "checksums.sha256"))


def model_config():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTextConfig,
    )

    from pithtrain.modules.model_config import resolve_text_model_config

    raw = json.loads((REPO / "tests/configs/qwen3_omni_text/tiny.json").read_text())
    return resolve_text_model_config(Qwen3OmniMoeTextConfig(**raw))


def preflight(args):
    from pithtrain.modules.dataset import MemmapDataset
    from pithtrain.tasks.prepare_omni_data import processor_for

    assert not torch.cuda.is_initialized(), "CPU preflight must not initialize CUDA"
    assert not args.output.exists(), "Use a fresh output directory"
    assert not args.output.is_relative_to(args.dataset) and not args.dataset.is_relative_to(
        args.output
    )
    identity = verify_input(args.dataset)
    config = model_config()
    recipe = json.loads((args.dataset / "bundle.json").read_text())["recipe"]
    processor, thinker = processor_for(recipe, offline=True)
    assert config.vocab_size >= thinker.text_config.vocab_size, (
        "Processor vocabulary exceeds the native model"
    )
    processor_info = dict(
        model_id=recipe["model_id"],
        revision=recipe["revision"],
        processor_class=type(processor).__name__,
        vocab_size=thinker.text_config.vocab_size,
    )
    shards = [
        MemmapDataset(p, SETTINGS["sequence_length"])
        for p in sorted((args.dataset / "tokens/train").rglob("*.bin"))
    ]
    assert shards, "No real train token shards"
    count = sum(map(len, shards))
    required = SETTINGS["max_steps"] * SETTINGS["global_batch_size"]
    assert count >= required, f"Real corpus has {count} samples; requires {required}"
    minimum = min(int(shard.tokens.min()) for shard in shards)
    maximum = max(int(shard.tokens.max()) for shard in shards)
    assert 0 <= minimum <= maximum < config.vocab_size, "Real token IDs exceed model vocabulary"
    args.output.mkdir(parents=True)
    shutil.copytree(args.dataset, args.output / "data")
    assert verify_input(args.output / "data") == identity
    assert not torch.cuda.is_initialized()
    return dict(
        schema=SCHEMA,
        protocol=PROTOCOL,
        settings=SETTINGS,
        source=provenance(),
        input=identity,
        original_data=str(args.dataset),
        samples=count,
        required_samples=required,
        token_range=[minimum, maximum],
        model_config=config.to_dict(),
        gpu_used=False,
        processor=processor_info,
    )


def validate_preflight(prepared, args):
    assert prepared["status"] == "PASSED" and prepared["gpu_used"] is False
    assert prepared["schema"] == SCHEMA and prepared["protocol"] == PROTOCOL
    assert prepared["settings"] == SETTINGS, "Training configuration changed"
    assert prepared["source"] == provenance(), "Source changed after preflight"
    assert prepared["original_data"] == str(args.dataset), "Input location changed"
    assert prepared["input"] == verify_input(args.dataset) == verify_input(args.output / "data")


def source(args):
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, setup_dataset

    cfg = PretrainLMCfg()
    cfg.data.dataset, cfg.data.format, cfg.data.modalities = (
        args.output / "data",
        "prepared_bundle",
        ("text",),
    )
    cfg.data.num_workers = 0
    for name, value in SETTINGS.items():
        setattr(cfg.training, name, value)
    cfg.training.model = args.output / "model"
    model_config().save_pretrained(cfg.training.model)
    # Reuse the production data entrypoint, including model/processor vocabulary
    # and modality validation. The loader's own state owns consumption.
    data = setup_dataset(cfg)
    assert data.is_text and data.checkpoint_state is data
    return data


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def setup_device(args):
    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", (
        "GPU execution requires explicit opt-in"
    )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise NotImplementedError("Distributed native strict resume is not implemented")
    assert torch.cuda.is_available(), "Native kernels and the real loader require CUDA"
    assert torch.cuda.get_device_capability(0)[0] in (9, 10), (
        "Use the supported Hopper/Blackwell target"
    )
    assert not dist.is_initialized(), "Use a fresh process"
    torch.cuda.set_device(0)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    rendezvous = args.output / f"rendezvous-{args.mode}"
    assert not rendezvous.exists(), "Stale rendezvous; use new evidence"
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=45),
        device_id=torch.device("cuda", 0),
    )
    from pithtrain.contexts import distributed, training
    from pithtrain.operators.grouped_linear import GroupedLinear

    distributed.pp_size = distributed.ep_size = distributed.cp_size = distributed.dp_size = 1
    distributed.pp_rank = distributed.ep_rank = distributed.cp_rank = distributed.dp_rank = 0
    distributed.pp_group = distributed.ep_group = distributed.cp_group = None
    distributed.device = torch.device("cuda", 0)
    training.fp8, training.Linear, training.GroupedLinear = False, torch.nn.Linear, GroupedLinear
    seed(SETTINGS["seed"])


def models():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextModel as HFText,
    )

    from pithtrain.models.qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel
    from pithtrain.modules.checkpoint import to_localized_model
    from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import hf_to_canonical

    config = model_config()
    hf_config = copy.deepcopy(config)
    hf_config._attn_implementation = hf_config._experts_implementation = "eager"
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.manual_seed(SETTINGS["seed"])
            hf = torch.nn.ModuleDict(
                {
                    "model": HFText(hf_config),
                    "lm_head": torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False),
                }
            )
            for name, parameter in hf.named_parameters():
                if parameter.ndim > 1:
                    torch.nn.init.normal_(parameter, std=config.initializer_range)
                elif name.endswith("bias"):
                    torch.nn.init.zeros_(parameter)
                else:
                    torch.nn.init.ones_(parameter)
    finally:
        torch.set_default_dtype(old_dtype)
    # Device-only transfer keeps the official HF RoPE buffers in FP32.
    hf = hf.cuda().train()
    with torch.device("cuda"):
        native = Qwen3OmniMoeThinkerTextModel(config, phase=-1).to(dtype=torch.bfloat16).train()
    canonical, _ = hf_to_canonical(hf.state_dict(), config, prefix="")
    native.load_state_dict(to_localized_model(canonical, native), strict=True)
    actual = native_parameters(native)
    expected = dict(hf.named_parameters())
    equal(actual, expected, "initial-converted-weights")
    return native, hf


def native_parameters(model):
    return {
        name if name.startswith("lm_head.") else "model." + name: p
        for name, p in model.named_parameters()
    }


def optimizer(model):
    opt = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"], weight_decay=0.1, foreach=False)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    return opt, sch


def state(parts, data, step):
    from pithtrain.modules.strict_checkpoint import StrictResumeState

    model, opt, sch = parts
    return StrictResumeState(
        model,
        (opt,),
        (sch,),
        data_state=data.checkpoint_state,
        step=step,
        global_batch_size=SETTINGS["global_batch_size"],
    )


def get_batch(data, step):
    microbatches = data.get_batch(step, torch.device("cuda", 0))
    indices = [
        int(data._dense.corpus.indices[i])
        for i in range(
            step * SETTINGS["global_batch_size"], (step + 1) * SETTINGS["global_batch_size"]
        )
    ]
    observed = []
    for micro in microbatches:
        assert micro.cu_seqlens is None and micro.media_inputs is None
        assert (
            micro.model_inputs[0].shape
            == micro.objective_inputs[0].shape
            == (1, SETTINGS["sequence_length"])
        )
        observed.append(
            dict(
                tokens=cpu_tree(micro.model_inputs[0]),
                labels=cpu_tree(micro.objective_inputs[0]),
                sample_ids=micro.sample_ids,
            )
        )
    return microbatches, dict(indices=indices, microbatches=observed)


def backward(model, batches, *, hf=False):
    # Same PR100 label contract and single global-token normalization. Labels
    # are already shifted by the real loader; never shift them a second time.
    from pithtrain.modules.training_data import global_target_count

    count = global_target_count(batches)
    loss = torch.zeros((), device="cuda", dtype=torch.float32)
    logits_record = []
    for micro in batches:
        inputs, labels = micro.model_inputs[0], micro.objective_inputs[0]
        logits = (
            model["lm_head"](model["model"](input_ids=inputs, use_cache=False).last_hidden_state)
            if hf
            else model.reference_forward(inputs)
        )
        objective = (
            F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum") / count
        )
        assert torch.isfinite(objective) and torch.isfinite(logits).all()
        objective.backward()
        loss += objective.detach()
        logits_record.append(cpu_tree(logits))
    params = dict(model.named_parameters()) if hf else native_parameters(model)
    for name, parameter in params.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (
            f"Missing/nonfinite gradient {name}"
        )
    return loss, logits_record


def apply_update(parts):
    model, opt, sch = parts
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    sch.step()
    opt.zero_grad(set_to_none=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def rng_probe():
    # Audit all restored generators after each real update. These observations
    # are not synthetic training inputs and do not replace native gradients.
    return dict(
        python=random.random(),
        numpy=float(np.random.random()),
        cpu=torch.rand(3),
        cuda=torch.rand(3, device="cuda").cpu(),
    )


def update(args):
    data = source(args)
    native, hf = models()
    actual_parts, expected_parts = (native, *optimizer(native)), (hf, *optimizer(hf))
    actual, expected = native_parameters(native), dict(hf.named_parameters())
    initial = {name: parameter.detach().float().clone() for name, parameter in actual.items()}
    batches, observed = get_batch(data, 0)
    actual_loss, actual_logits = backward(native, batches)
    expected_loss, expected_logits = backward(hf, batches, hf=True)
    logits_error = [difference(a, b) for a, b in zip(actual_logits, expected_logits, strict=True)]
    assert max(logits_error) < 1e-3
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-3)
    grad_error = compare_named(
        {n: p.grad for n, p in actual.items()},
        {n: p.grad for n, p in expected.items()},
        1e-2,
        "gradient",
    )
    apply_update(actual_parts)
    apply_update(expected_parts)
    delta_error = compare_named(
        {n: p.detach().float() - initial[n] for n, p in actual.items()},
        {n: p.detach().float() - initial[n] for n, p in expected.items()},
        1e-2,
        "optimizer-update",
    )
    assert any(not torch.equal(p.detach().float(), initial[n]) for n, p in actual.items()), (
        "No native parameter changed"
    )
    for parts in (actual_parts, expected_parts):
        assert len(parts[1].state) == len(list(parts[0].parameters()))
        assert all(item["step"].item() == 1 for item in parts[1].state.values())
        assert parts[2].last_epoch == 1
    data.commit_step(0)
    return dict(
        loss=actual_loss.item(),
        hf_loss=expected_loss.item(),
        logits_error=logits_error,
        gradient_errors=grad_error,
        update_errors=delta_error,
        parameter_count=len(actual),
        data=data.state_dict(),
        batch_sha256=tensor_tree_digest(observed),
    )


def tensor_tree_digest(value):
    # Evidence identity, not a substitute for exact tensor comparison on restore.
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return dict(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            sha256=hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        )
    if isinstance(value, dict):
        return {key: tensor_tree_digest(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_tree_digest(item) for item in value]
    return value


def step_once(parts, data, step):
    batches, observed = get_batch(data, step)
    loss, _ = backward(parts[0], batches)
    apply_update(parts)
    data.commit_step(step)
    return dict(batch=observed, loss=cpu_tree(loss), rng=cpu_tree(rng_probe()))


def build_run(args):
    data = source(args)
    native, hf = models()
    del hf
    parts = (native, *optimizer(native))
    seed(SETTINGS["seed"])
    return parts, data


def produce(args):
    from pithtrain.modules.strict_checkpoint import save_strict_checkpoint

    parts, data = build_run(args)
    records = [step_once(parts, data, i) for i in range(2)]
    boundary = state(parts, data, 2)
    location = save_strict_checkpoint(args.output / "checkpoints", boundary)
    # Same PR100 observation order: save, then inspect live runtime state.
    expected = dict(
        schema=SCHEMA, protocol=PROTOCOL, pid=os.getpid(), boundary=cpu_tree(boundary.state_dict())
    )
    expected["next"] = step_once(parts, data, 2)
    expected["after"] = cpu_tree(state(parts, data, 3).state_dict())
    torch.save(expected, args.output / "continuous.pt")
    return dict(
        checkpoint=str(location),
        checkpoint_step=2,
        final_step=3,
        losses=[record["loss"].item() for record in records] + [expected["next"]["loss"].item()],
        next_batch=tensor_tree_digest(expected["next"]["batch"]),
        consumed_samples=data.consumed_samples,
    )


def restore(args):
    from pithtrain.modules.strict_checkpoint import load_strict_checkpoint

    expected = torch.load(args.output / "continuous.pt", map_location="cpu", weights_only=True)
    assert expected["schema"] == SCHEMA and expected["protocol"] == PROTOCOL
    assert expected["pid"] != os.getpid(), "Restore must execute in a new process"
    parts, data = build_run(args)
    # Prove the checkpoint replaces model and all RNG state, not just a cursor.
    with torch.no_grad():
        for parameter in parts[0].parameters():
            parameter.zero_()
    seed(9917)
    boundary = state(parts, data, 2)
    assert not parts[1].state and data.consumed_samples == 0
    assert load_strict_checkpoint(args.output / "checkpoints", boundary) == 2
    equal(cpu_tree(boundary.state_dict()), expected["boundary"], "restored-boundary")
    observed = step_once(parts, data, 2)
    equal(observed, expected["next"], "next-update")
    equal(cpu_tree(state(parts, data, 3).state_dict()), expected["after"], "after-update")
    return dict(
        restored_step=2,
        final_step=3,
        consumed_samples=data.consumed_samples,
        max_error=0,
        producer_pid=expected["pid"],
        next_batch=tensor_tree_digest(observed["batch"]),
    )


def negative(args):
    from pithtrain.modules.strict_checkpoint import load_strict_checkpoint

    parts, data = build_run(args)
    boundary = state(parts, data, 2)
    load_strict_checkpoint(args.output / "checkpoints", boundary)
    good = cpu_tree(boundary.state_dict())
    names = [
        "missing-model",
        "unexpected-model",
        "shape",
        "dtype",
        "missing-optimizer",
        "missing-scheduler",
        "missing-data",
        "missing-cuda-rng",
        "fingerprint",
        "cursor",
        "optimizer-order",
    ]
    for name in names:
        corrupt = copy.deepcopy(good)
        key = next(iter(corrupt["model"]))
        if name == "missing-model":
            del corrupt["model"][key]
        elif name == "unexpected-model":
            corrupt["model"]["unexpected.weight"] = corrupt["model"][key]
        elif name == "shape":
            corrupt["model"][key] = corrupt["model"][key].flatten()[:1]
        elif name == "dtype":
            corrupt["model"][key] = corrupt["model"][key].double()
        elif name in ("missing-optimizer", "missing-scheduler", "missing-data"):
            del corrupt[name.removeprefix("missing-")]
        elif name == "missing-cuda-rng":
            del corrupt["resume"]["rng"]["cuda"]
        elif name == "fingerprint":
            corrupt["data"]["fingerprint"] = "wrong"
        elif name == "cursor":
            corrupt["data"]["consumed_samples"] += 1
        elif name == "optimizer-order":
            corrupt["optimizer"]["param_groups"][0]["params"].reverse()
        try:
            boundary.load_state_dict(corrupt)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Corrupt checkpoint accepted: {name}")
        equal(cpu_tree(boundary.state_dict()), good, f"failure-atomicity-{name}")
    return dict(rejected=names, cases=len(names), state_error=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preflight", "update", "produce", "restore", "negative"), required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    return args


def main():
    args = parse_args()
    output_existed = args.output.exists()
    validated_output = False
    report = args.output / f"{args.mode}.json"
    assert not report.exists(), "Never overwrite prior acceptance evidence"
    try:
        if args.mode == "preflight":
            result = preflight(args)
        else:
            prepared = json.loads((args.output / "preflight.json").read_text())
            validate_preflight(prepared, args)
            validated_output = True
            subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"], check=True)
            setup_device(args)
            result = {
                "update": update,
                "produce": produce,
                "restore": restore,
                "negative": negative,
            }[args.mode](args)
            assert verify_input(args.dataset) == prepared["input"]
            torch.cuda.synchronize()
            result.update(
                gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
        result.update(
            status="PASSED",
            mode=args.mode,
            pid=os.getpid(),
            utc=datetime.now(timezone.utc).isoformat(),
            scope="unsharded native Omni, real PR100 text; distributed strict resume unimplemented",
        )
        report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: result[key] for key in ("status", "mode", "pid", "scope")}))
    except Exception as error:
        created_output = args.mode == "preflight" and not output_existed
        if (validated_output or created_output) and args.output.is_dir() and not report.exists():
            report.write_text(
                json.dumps(
                    dict(status="FAILED", mode=args.mode, error=f"{type(error).__name__}: {error}"),
                    indent=2,
                )
                + "\n"
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
