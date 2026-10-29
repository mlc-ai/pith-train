"""Complete Qwen3 speech training with explicit replica or FSDP2 contracts.

CP=EP=PP=1 is required. Ordinary replicas use local verified checkpoints;
apply_speech_fsdp models use native strict task checkpoints and FP32 SUM
sharded gradients. Codec labels are already aligned. Waveform training requires
a caller-defined differentiable sum/count objective. Runtime/memory feasibility
and native GPU parity need independent validation for the chosen model size.
"""

import copy
import hashlib
import math
import os
import random
import struct
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from pithtrain.modules.resume_semantics import model_contract as _model_contract
from pithtrain.modules.resume_semantics import scheduler_contract
from pithtrain.tasks.omni_objectives import (
    TargetLoss,
    global_target_counts,
    normalized_loss,
    speech_loss_sums,
    text_loss_sum,
)
from pithtrain.tasks.sft_lm import (
    SFTCheckpointState,
    SFTState,
    reject_nonfinite_gradients,
    reject_unsupported_auxiliary_loss,
)


@dataclass
class OmniSpeechCfg:
    mode: str = "codec_only"
    codec_weighting: str = "combined"
    text_weight: float = 1.0
    max_updates: int = 1
    max_batches: int = 1
    max_grad_norm: float | None = 1.0
    speaker: str = "ethan"
    cp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    moe_load_balance_coef: float = 0.0
    waveform_objective_id: str | None = None


def _base(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def configure_trainable_components(model, mode):
    """Call before constructing the optimizer (and before a DDP wrapper).

    The optimizer must contain exactly these trainable parameters. A frozen
    Thinker still supplies conditioning, while code2wav is absent from codec
    forward and must never receive accidental AdamW weight decay.
    """
    if any(isinstance(module, FSDPModule) for module in model.modules()) or any(
        isinstance(parameter, DTensor) for parameter in model.parameters()
    ):
        raise NotImplementedError(
            "FSDP/DTensor speech training needs registered methods and sharded checkpoint integration"
        )
    if isinstance(model, DistributedDataParallel):
        raise ValueError("Configure components before wrapping DDP")
    active = {
        "codec_only": {"talker"},
        "text_and_codec": {"thinker", "talker"},
        "waveform": {"code2wav"},
    }.get(mode)
    if active is None:
        raise ValueError("Choose codec_only, text_and_codec or waveform")
    if getattr(model.config, "model_type", None) != "qwen3_omni_moe":
        raise ValueError("Only complete Qwen3-Omni speech models are supported")
    owned = set()
    for name in ("thinker", "talker", "code2wav"):
        component = getattr(model, name)
        ids = {id(p) for p in component.parameters()}
        if owned & ids:
            raise ValueError(
                "Cross-component parameter aliases require an explicit training policy"
            )
        owned |= ids
    if owned != {id(p) for p in model.parameters()}:
        raise ValueError("All parameters must belong to named speech components")
    for name in ("thinker", "talker", "code2wav"):
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(name in active)
        getattr(model, name).train(name in active)
    return tuple(p for p in model.parameters() if p.requires_grad)


def _fsdp_plan(model):
    plan = getattr(_base(model), "_omni_speech_fsdp_plan", None)
    if plan is None:
        return None
    required = {
        "schema",
        "mode",
        "mesh",
        "group",
        "gradient_reduction",
        "master_dtype",
        "param_dtype",
        "reduce_dtype",
        "require_backward",
        "owned_modules",
        "fixed_topology",
        "thinker_sharding",
    }
    if (
        not isinstance(plan, dict)
        or not required <= set(plan)
        or (
            plan["schema"] != 1
            or plan["gradient_reduction"] != "sum"
            or plan["master_dtype"] != torch.float32
            or plan["reduce_dtype"] != torch.float32
            or plan["param_dtype"] != torch.bfloat16
            or not plan["require_backward"]
            or not plan["fixed_topology"]
            or not isinstance(plan["thinker_sharding"], str)
            or not plan["thinker_sharding"]
        )
    ):
        raise ValueError("Invalid declared speech FSDP plan")
    modules = dict(_base(model).named_modules())
    if not plan["owned_modules"] or any(
        name not in modules or not isinstance(modules[name], FSDPModule)
        for name in plan["owned_modules"]
    ):
        raise ValueError("Declared FSDP modules do not match the actual model")
    return plan


def _collective_preflight(error, *, device, group, message):
    rejected = torch.tensor(int(error is not None), dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(rejected, op=dist.ReduceOp.MAX, group=group)
    if rejected.item():
        raise ValueError(message) from error


def clip_sharded_grad_norm(model, maximum, *, group):
    """One-dimensional FSDP shard norm; no replica gradient all-reduce."""
    device = next(model.parameters()).device
    square_sum = torch.zeros((), dtype=torch.float64, device=device)
    gradients = []
    error = None
    try:
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if (
                not isinstance(parameter.grad, DTensor)
                or len(parameter.grad.placements) != 1
                or not parameter.grad.placements[0].is_shard()
            ):
                raise ValueError("FSDP gradient norm requires one-dimensional sharded gradients")
            local = parameter.grad.to_local()
            square_sum += torch.linalg.vector_norm(local, dtype=torch.float64).square()
            gradients.append(local)
    except Exception as exc:
        error = exc
    _collective_preflight(
        error, device=device, group=group, message="Sharded gradient norm preflight rejected"
    )
    dist.all_reduce(square_sum, op=dist.ReduceOp.SUM, group=group)
    norm = square_sum.sqrt()
    reject_nonfinite_gradients(model, device=device, group=group, norm=norm)
    scale = (maximum / norm.clamp_min(1e-12)).clamp(max=1)
    for gradient in gradients:
        gradient.mul_(scale.to(gradient.dtype))
    return norm


def _native_checkpoint(
    action, *, root, scan, cfg, model, optimizer, scheduler, data, state, contract, token_group
):
    from pithtrain.modules import checkpoint as native
    from pithtrain.modules.omni_speech_fsdp import (
        native_speech_checkpoint_state,
        speech_checkpoint_context,
    )

    device = next(model.parameters()).device
    group = token_group
    error = None
    with ExitStack() as stack:
        try:
            plan = _fsdp_plan(model)
            stable = dict(
                contract,
                speech_task=dict(
                    schema=2,
                    model_contract=_model_contract(model),
                    mode=cfg.mode,
                    codec_weighting=cfg.codec_weighting,
                    text_weight=cfg.text_weight,
                    speaker=cfg.speaker,
                    max_grad_norm=cfg.max_grad_norm,
                    waveform_objective_id=cfg.waveform_objective_id,
                    fsdp=dict(
                        schema=plan["schema"],
                        thinker_sharding=plan["thinker_sharding"],
                        reduction="sum",
                        master_dtype="float32",
                        param_dtype="bfloat16",
                        reduce_dtype="float32",
                    ),
                ),
            )
            adapter = native_speech_checkpoint_state(
                SFTCheckpointState(state, data), contract=stable
            )
            stack.enter_context(speech_checkpoint_context(model, optimizer, scheduler))
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError("Native checkpoint requires no pending gradients")
            adapter.validate_state_dict(adapter.state_dict())
        except Exception as exc:
            error = exc
        _collective_preflight(
            error,
            device=device,
            group=group,
            message="Native speech checkpoint wrapper preflight rejected on at least one rank",
        )
        function = native.save_checkpoint if action == "save" else native.load_checkpoint
        function(root, scan, data_state=adapter, strict=True, resume_config=stable)


def _validate(
    cfg, model, optimizer, scheduler, token_group, gradient_reduction, waveform_objective
):
    plan = _fsdp_plan(model)
    sharded = any(isinstance(module, FSDPModule) for module in model.modules()) or any(
        isinstance(parameter, DTensor) for parameter in model.parameters()
    )
    if sharded and plan is None:
        raise NotImplementedError(
            "FSDP/DTensor speech training requires apply_speech_fsdp's declared plan"
        )
    if plan is not None:
        if (
            plan["mode"] != cfg.mode
            or plan["group"] is not token_group
            or gradient_reduction != "sum"
        ):
            raise ValueError("FSDP mode/group must match helper and reduction must be SUM")
        if isinstance(model, DistributedDataParallel) or type(optimizer) is not torch.optim.AdamW:
            raise NotImplementedError("Speech FSDP uses native AdamW without a DDP wrapper")
    if (cfg.cp_size, cfg.ep_size, cfg.pp_size) != (1, 1, 1):
        raise NotImplementedError("Complete speech trainer currently requires CP=EP=PP=1")
    if cfg.codec_weighting not in ("combined", "per_head"):
        raise ValueError("Choose an explicit codec weighting")
    if gradient_reduction not in ("sum", "mean"):
        raise ValueError("Declare the replica gradient reduction as sum or mean")
    if any(type(v) is not int or v < 0 for v in (cfg.max_updates, cfg.max_batches)):
        raise ValueError("Update and scan limits must be nonnegative integers")
    if not 0 < cfg.text_weight < float("inf"):
        raise ValueError("text_weight must be positive and finite")
    if cfg.max_grad_norm is not None and not 0 < cfg.max_grad_norm < float("inf"):
        raise ValueError("max_grad_norm must be positive and finite or None")
    if cfg.mode == "waveform" and (
        waveform_objective is None
        or not isinstance(cfg.waveform_objective_id, str)
        or not cfg.waveform_objective_id.strip()
    ):
        raise ValueError(
            "Waveform mode requires a caller-defined sum/count objective and stable objective ID"
        )
    if dist.is_initialized() and token_group is None:
        raise ValueError("Explicit DP process group required")
    if isinstance(model, DistributedDataParallel) and model.process_group is not token_group:
        raise ValueError("DDP and objective must use the same process group")
    if isinstance(model, DistributedDataParallel) and not model.find_unused_parameters:
        raise ValueError("DDP must enable find_unused_parameters for conditional speech heads")
    base = _base(model)
    active = {
        "codec_only": {"talker"},
        "text_and_codec": {"thinker", "talker"},
        "waveform": {"code2wav"},
    }.get(cfg.mode)
    if active is None or getattr(base.config, "model_type", None) != "qwen3_omni_moe":
        raise ValueError("Select a Qwen3-Omni training mode")
    for name in ("thinker", "talker", "code2wav"):
        if any(p.requires_grad != (name in active) for p in getattr(base, name).parameters()):
            raise ValueError("Call configure_trainable_components before creating the optimizer")
    expected = {id(p) for p in base.parameters() if p.requires_grad}
    members = [p for group in optimizer.param_groups for p in group["params"]]
    if len(members) != len(expected) or {id(p) for p in members} != expected:
        raise ValueError(
            "Optimizer must contain every active parameter exactly once and no frozen ones"
        )
    if scheduler is not None and scheduler.optimizer is not optimizer:
        raise ValueError("Scheduler belongs to another optimizer")
    if type(optimizer) not in (torch.optim.AdamW, torch.optim.SGD):
        raise NotImplementedError("Strict local checkpoint supports torch AdamW and SGD")


def _rng_state():
    np_state = np.random.get_state()
    return dict(
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        python=random.getstate(),
        numpy=(np_state[0], np_state[1].tolist(), *np_state[2:]),
    )


def _set_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(
        (state["numpy"][0], np.asarray(state["numpy"][1], dtype=np.uint32), *state["numpy"][2:])
    )


def _topology(group):
    return dict(
        dp=dist.get_world_size(group) if dist.is_initialized() else 1,
        rank=dist.get_rank(group) if dist.is_initialized() else 0,
        cp=1,
        ep=1,
        pp=1,
    )


def _manifest(cfg, model, optimizer, scheduler, group, reduction):
    base = _base(model)
    names = {id(p): name for name, p in base.named_parameters()}
    return dict(
        version=2,
        kind="qwen3_omni_complete_autograd",
        model_contract=_model_contract(model),
        mode=cfg.mode,
        codec_weighting=cfg.codec_weighting,
        text_weight=cfg.text_weight,
        speaker=cfg.speaker,
        waveform_objective_id=cfg.waveform_objective_id,
        max_grad_norm=cfg.max_grad_norm,
        reduction=reduction,
        topology=_topology(group),
        model=[(n, list(t.shape), str(t.dtype)) for n, t in base.state_dict().items()],
        trainable=[(n, p.requires_grad) for n, p in base.named_parameters()],
        optimizer=type(optimizer).__module__ + "." + type(optimizer).__qualname__,
        param_groups=[[names[id(p)] for p in g["params"]] for g in optimizer.param_groups],
        scheduler=scheduler_contract(scheduler),
    )


def _rank_path(path, group):
    path = Path(path)
    topology = _topology(group)
    return (
        path if topology["dp"] == 1 else path.with_name(path.name + f".rank{topology['rank']:05d}")
    )


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _agree_checkpoint(payload, group):
    if not dist.is_initialized():
        return
    manifest = copy.deepcopy(payload["manifest"])
    manifest["topology"].pop("rank")
    identity = dict(
        manifest=manifest,
        task=payload["task"],
        learning_rates=[g["lr"] for g in payload["optimizer"]["param_groups"]],
    )
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, identity, group=group)
    if any(peer != peers[0] for peer in peers):
        raise ValueError("Speech checkpoint ranks disagree on global transaction/contract")


_INTEGRITY_MAGIC = b"PITH_SPEECH_SHA256_V1\n"
_INTEGRITY_BYTES = len(_INTEGRITY_MAGIC) + 8 + 32


def _payload_hash(stream, size):
    stream.seek(0)
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        block = stream.read(min(remaining, 1024 * 1024))
        if not block:
            raise ValueError("Truncated checkpoint payload")
        digest.update(block)
        remaining -= len(block)
    return digest.digest()


def _write_verified_checkpoint(target, payload):
    """One atomic file, with a streaming SHA256/length footer over all payload bytes."""
    target = Path(target)
    temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x+b") as stream:
            torch.save(payload, stream)
            size = stream.tell()
            digest = _payload_hash(stream, size)
            stream.seek(size)
            stream.write(_INTEGRITY_MAGIC + struct.pack(">Q", size) + digest)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_verified_checkpoint(target):
    # Keep the same file descriptor for hash verification and weights_only load;
    # a concurrent atomic replacement cannot substitute another inode mid-load.
    with Path(target).open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        if size < _INTEGRITY_BYTES:
            raise ValueError("Missing checkpoint integrity footer")
        stream.seek(size - _INTEGRITY_BYTES)
        footer = stream.read(_INTEGRITY_BYTES)
        offset = len(_INTEGRITY_MAGIC)
        if footer[:offset] != _INTEGRITY_MAGIC:
            raise ValueError("Missing checkpoint integrity footer")
        payload_size = struct.unpack(">Q", footer[offset : offset + 8])[0]
        if (
            payload_size != size - _INTEGRITY_BYTES
            or _payload_hash(stream, payload_size) != footer[-32:]
        ):
            raise ValueError("Checkpoint SHA256/length mismatch")
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=True)


def _optimizer_schema(saved, names):
    """Presence and exact counters include legitimately conditional unused params."""
    schema = []
    for group, group_names in zip(saved["param_groups"], names, strict=True):
        for index, name in zip(group["params"], group_names, strict=True):
            values = saved["state"].get(index, {})
            step = values.get("step")
            schema.append(
                dict(
                    id=index,
                    name=name,
                    present=index in saved["state"],
                    step=None if step is None else step.item(),
                    tensors=[
                        (key, list(value.shape), str(value.dtype)) for key, value in values.items()
                    ],
                )
            )
    return schema


def save_checkpoint(
    path, *, cfg, model, optimizer, scheduler, data, state, token_group, gradient_reduction
):
    """Atomic per-rank local save; no remote I/O or production DCP compatibility."""
    device = next(_base(model).parameters()).device
    error = None
    payload = None
    try:
        if any(isinstance(p, DTensor) for p in _base(model).parameters()):
            raise ValueError("Sharded speech models require the native checkpoint interface")
        if any(parameter.grad is not None for parameter in _base(model).parameters()):
            raise ValueError("Checkpoint requires a clean scan boundary with no pending gradients")
        state.validate_state_dict(state.state_dict())
        data.validate_state_dict(data.state_dict())
        if any(data.state_dict().get(key) != value for key, value in state.state_dict().items()):
            raise ValueError("Checkpoint data/task cursor mismatch")
        payload = dict(
            manifest=_manifest(cfg, model, optimizer, scheduler, token_group, gradient_reduction),
            model=_base(model).state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=None if scheduler is None else scheduler.state_dict(),
            data=data.state_dict(),
            task=state.state_dict(),
            rng=_rng_state(),
            modes=[(name, module.training) for name, module in _base(model).named_modules()],
        )
        payload["optimizer_schema"] = _optimizer_schema(
            payload["optimizer"], payload["manifest"]["param_groups"]
        )
        _preflight(
            payload,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            state=state,
            token_group=token_group,
            gradient_reduction=gradient_reduction,
        )
    except Exception as exc:
        error = exc
    _collective_preflight(
        error,
        device=device,
        group=token_group,
        message="Speech checkpoint save preflight rejected on at least one rank",
    )
    _agree_checkpoint(payload, token_group)
    error = None
    try:
        _write_verified_checkpoint(_rank_path(path, token_group), payload)
    except Exception as exc:
        error = exc
    # A failed writer may leave successful peer files. Report collective failure;
    # these are not an accepted transaction and must not be claimed as a save.
    _collective_preflight(
        error,
        device=device,
        group=token_group,
        message="Speech checkpoint write failed on at least one rank",
    )


def _preflight(
    payload, *, cfg, model, optimizer, scheduler, data, state, token_group, gradient_reduction
):
    if set(payload) != {
        "manifest",
        "model",
        "optimizer",
        "optimizer_schema",
        "scheduler",
        "data",
        "task",
        "rng",
        "modes",
    }:
        raise ValueError("Incomplete local speech checkpoint")
    if payload["manifest"] != _manifest(
        cfg, model, optimizer, scheduler, token_group, gradient_reduction
    ):
        raise ValueError("Speech checkpoint topology/names/shapes/dtypes/order/contract mismatch")
    state.validate_state_dict(payload["task"])
    data.validate_state_dict(payload["data"])
    for key in ("scanned_batches", "completed_updates"):
        if payload["data"].get(key) != payload["task"][key]:
            raise ValueError("Data and task checkpoint cursors differ")
    current = _base(model).state_dict()
    if list(payload["model"]) != list(current):
        raise ValueError("Model state order mismatch")
    for name, value in payload["model"].items():
        reference = current[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
        ):
            raise ValueError("Model tensor shape/dtype mismatch")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Nonfinite model checkpoint")
    expected_modes = [
        (name, bool(module.training)) for name, module in _base(model).named_modules()
    ]
    if [n for n, _ in payload["modes"]] != [n for n, _ in expected_modes] or any(
        type(value) is not bool for _, value in payload["modes"]
    ):
        raise ValueError("Module mode names/order mismatch")
    saved = payload["optimizer"]
    if payload["optimizer_schema"] != _optimizer_schema(saved, payload["manifest"]["param_groups"]):
        raise ValueError("Optimizer per-parameter presence/update schema mismatch")
    if set(saved) != {"state", "param_groups"} or len(saved["param_groups"]) != len(
        optimizer.param_groups
    ):
        raise ValueError("Optimizer group mismatch")
    ids, parameters = [], []
    for group, live in zip(saved["param_groups"], optimizer.param_groups, strict=True):
        if set(group) != set(live) or len(group["params"]) != len(live["params"]):
            raise ValueError("Optimizer parameter group schema mismatch")
        for key, value in group.items():
            if key not in {"params", "lr"} and value != live[key]:
                raise ValueError("Optimizer hyperparameter contract mismatch")
        if (
            type(group["lr"]) not in (float, int)
            or not math.isfinite(group["lr"])
            or group["lr"] < 0
        ):
            raise ValueError("Invalid saved learning rate")
        ids.extend(group["params"])
        parameters.extend(live["params"])
    if ids != list(range(len(ids))) or set(saved["state"]) - set(ids):
        raise ValueError("Optimizer parameter IDs/order mismatch")
    steps = payload["task"]["completed_updates"]
    for index, parameter in zip(ids, parameters, strict=True):
        values = saved["state"].get(index, {})
        allowed = (
            {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
            if type(optimizer) is torch.optim.AdamW
            else {"momentum_buffer"}
        )
        if set(values) - allowed:
            raise ValueError("Unknown optimizer state")
        if (
            type(optimizer) is torch.optim.AdamW
            and values
            and not {"step", "exp_avg", "exp_avg_sq"} <= set(values)
        ):
            raise ValueError("Incomplete AdamW state")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError("Invalid optimizer state tensor")
            if name == "step":
                if (
                    value.ndim != 0
                    or value.item() != int(value.item())
                    or not 0 < value.item() <= steps
                ):
                    raise ValueError("Invalid optimizer update counter")
            elif value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("Optimizer tensor shape/dtype mismatch")
    if (scheduler is None) != (payload["scheduler"] is None):
        raise ValueError("Scheduler presence mismatch")
    if scheduler is not None:
        if (
            set(payload["scheduler"]) != set(scheduler.state_dict())
            or payload["scheduler"].get("last_epoch") != steps
        ):
            raise ValueError("Scheduler update counter/schema mismatch")
        for key, value in payload["scheduler"].items():
            if (
                key not in {"last_epoch", "_step_count", "_last_lr"}
                and value != scheduler.state_dict()[key]
            ):
                raise ValueError("Scheduler policy contract mismatch")
        if payload["scheduler"].get("_step_count") != steps + 1 or payload["scheduler"].get(
            "_last_lr"
        ) != [g["lr"] for g in saved["param_groups"]]:
            raise ValueError("Scheduler/optimizer learning rate or update counter mismatch")
    elif any(
        group["lr"] != live["lr"]
        for group, live in zip(saved["param_groups"], optimizer.param_groups, strict=True)
    ):
        raise ValueError("Unscheduled learning rate contract mismatch")
    rng = payload["rng"]
    if set(rng) != {"torch", "cuda", "python", "numpy"}:
        raise ValueError("Incomplete RNG state")
    torch.Generator().set_state(rng["torch"])
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(
        (rng["numpy"][0], np.asarray(rng["numpy"][1], dtype=np.uint32), *rng["numpy"][2:])
    )
    expected_cuda = len(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else 0
    if len(rng["cuda"]) != expected_cuda:
        raise ValueError("CUDA RNG device topology mismatch")
    for index, value in enumerate(rng["cuda"]):
        torch.Generator(device=f"cuda:{index}").set_state(value)


def load_checkpoint(
    path, *, cfg, model, optimizer, scheduler, data, state, token_group, gradient_reduction
):
    """Validate all state before loading, with rollback on apply errors.

    Each rank must use its own file and the exact saved DP topology. No optimizer
    ID remapping, dtype coercion, partial model loads, or cursor guesses occur.
    Rollback snapshots are CPU copies of model/optimizer state, requiring host
    memory proportional to full state; this is not a sharded checkpoint format.
    """
    payload = None
    error = None
    try:
        if any(isinstance(p, DTensor) for p in _base(model).parameters()):
            raise ValueError("Sharded speech models require the native checkpoint interface")
        if any(parameter.grad is not None for parameter in _base(model).parameters()):
            raise ValueError("Resume requires a clean scan boundary with no pending gradients")
        payload = _read_verified_checkpoint(_rank_path(path, token_group))
        _preflight(
            payload,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            state=state,
            token_group=token_group,
            gradient_reduction=gradient_reduction,
        )
    except Exception as exc:
        error = exc
    device = next(_base(model).parameters()).device
    rejected = torch.tensor(int(error is not None), dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(rejected, op=dist.ReduceOp.MAX, group=token_group)
    if rejected.item():
        raise ValueError("Speech checkpoint preflight rejected on at least one rank") from error
    _agree_checkpoint(payload, token_group)
    error = None
    before = None
    try:
        before = dict(
            model=_cpu_snapshot(_base(model).state_dict()),
            optimizer=_cpu_snapshot(optimizer.state_dict()),
            scheduler=None if scheduler is None else copy.deepcopy(scheduler.state_dict()),
            data=copy.deepcopy(data.state_dict()),
            task=state.state_dict(),
            rng=_rng_state(),
            modes=[(n, m.training) for n, m in _base(model).named_modules()],
        )
    except Exception as exc:
        error = exc
    _collective_preflight(
        error,
        device=device,
        group=token_group,
        message="Speech checkpoint rollback snapshot failed before any state was applied",
    )

    def apply(saved):
        _base(model).load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        data.load_state_dict(saved["data"])
        state.load_state_dict(saved["task"])
        for (name, module), (saved_name, mode) in zip(
            _base(model).named_modules(), saved["modes"], strict=True
        ):
            assert name == saved_name
            module.training = mode
        _set_rng(saved["rng"])

    error = None
    try:
        apply(payload)
    except Exception as exc:
        error = exc
    rejected.fill_(int(error is not None))
    if dist.is_initialized():
        dist.all_reduce(rejected, op=dist.ReduceOp.MAX, group=token_group)
    if rejected.item():
        apply(before)
        raise RuntimeError("Speech checkpoint application failed; live state restored") from error


def _step(
    cfg,
    batches,
    *,
    model,
    optimizer,
    scheduler,
    token_group,
    gradient_reduction,
    waveform_objective,
):
    base = _base(model)
    device = next(base.parameters()).device
    error = None
    plan = None
    try:
        plan = _fsdp_plan(model)
        if not batches:
            raise ValueError("Every rank supplies a microbatch, even with zero targets")
        if isinstance(model, DistributedDataParallel) and (
            len(batches) != 1 or cfg.mode == "waveform"
        ):
            raise NotImplementedError(
                "DDP wrapper supports one codec microbatch; use explicit manual DP otherwise"
            )
    except Exception as exc:
        error = exc
    if not dist.is_initialized() and error is not None:
        raise error
    _collective_preflight(
        error,
        device=device,
        group=token_group,
        message="Speech step admission rejected on at least one rank",
    )
    if plan is not None:
        from pithtrain.modules.omni_speech_fsdp import validate_speech_collective_plan

        validate_speech_collective_plan(
            batches, cfg.mode, token_group, model=base, speaker=cfg.speaker
        )
    reject_unsupported_auxiliary_loss(cfg, base, device=device, include_training_context=False)
    # Counts are produced by the official teacher-forcing path. Preserve logical
    # mutable forward state when its complete global objective is empty.
    rng = _rng_state()
    buffers = [(b, b.detach().clone()) for b in base.buffers()]
    prior_gradients = {
        name: (p.grad, None if p.grad is None else p.grad.detach().clone())
        for name, p in base.named_parameters()
    }
    modes = [(m, m.training) for m in base.modules()]
    active = {
        "codec_only": {"talker"},
        "text_and_codec": {"thinker", "talker"},
        "waveform": {"code2wav"},
    }[cfg.mode]
    for name in ("thinker", "talker", "code2wav"):
        getattr(base, name).train(name in active)
    codec, texts = [], []
    for batch in batches:
        error = None
        try:
            if cfg.mode == "waveform":
                wave = base.decode_training_codes(batch["codes"].transpose(1, 2))
                item = waveform_objective(wave, batch)
                if (
                    not isinstance(item, TargetLoss)
                    or item.loss_sum.ndim != 0
                    or item.target_count.ndim != 0
                ):
                    raise ValueError(
                        "Waveform objective returns scalar TargetLoss(sum, int64 count)"
                    )
                codec.append(item)
            else:
                output = model(
                    batch["input_ids"],
                    batch["model_context"],
                    batch["codes"],
                    batch["code_lengths"],
                    prompt_lengths=batch["prompt_lengths"],
                    speaker=cfg.speaker,
                    codebook_mask=batch.get("codebook_mask"),
                    position_kwargs=batch.get("position_kwargs"),
                )
                codec.append(
                    speech_loss_sums(
                        output.first_logits,
                        output.first_labels,
                        output.residual_logits,
                        output.residual_labels,
                    )
                )
                if cfg.mode == "text_and_codec":
                    texts.append(text_loss_sum(output.thinker_logits, batch["labels"]))
            for item in [codec[-1]] + ([texts[-1]] if cfg.mode == "text_and_codec" else []):
                if (
                    item.target_count.dtype != torch.int64
                    or item.target_count.device != device
                    or item.loss_sum.device != device
                    or (item.target_count < 0).any()
                    or not item.loss_sum.requires_grad
                    or not torch.isfinite(item.loss_sum).all()
                ):
                    raise ValueError(
                        "Objective needs finite differentiable sums and nonnegative int64 counts"
                    )
        except Exception as exc:
            error = exc
        if plan is not None:
            _collective_preflight(
                error,
                device=device,
                group=token_group,
                message="Speech forward/objective preflight rejected on at least one rank",
            )
        elif error is not None:
            break
    if plan is None:
        _collective_preflight(
            error,
            device=device,
            group=token_group,
            message="Speech forward/objective preflight rejected on at least one rank",
        )
    sums = torch.stack([item.loss_sum for item in codec]).sum(0)
    counts = global_target_counts(
        torch.stack([item.target_count for item in codec]).sum(0), token_group=token_group
    )
    text_count = global_target_counts(
        sum(
            (item.target_count for item in texts),
            start=torch.zeros((), device=device, dtype=torch.int64),
        ),
        token_group=token_group,
    )
    if int(counts.sum()) == 0 and int(text_count) == 0:
        if isinstance(model, DistributedDataParallel) or plan is not None:
            # Complete DDP/FSDP's begun backward/reshard cycle, then restore
            # logical gradient values/None by name after sharded parameters return.
            zero = sums.sum()
            if texts:
                zero = zero + torch.stack([item.loss_sum for item in texts]).sum()
            (zero * 0).backward()
            with torch.no_grad():
                for name, parameter in base.named_parameters():
                    gradient, saved = prior_gradients[name]
                    if gradient is not None:
                        gradient.copy_(saved)
                    parameter.grad = gradient
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, mode in modes:
            module.training = mode
        _set_rng(rng)
        return False
    loss = normalized_loss(sums, counts, weighting=cfg.codec_weighting)
    if texts:
        text_sum = torch.stack([item.loss_sum for item in texts]).sum()
        loss = loss + cfg.text_weight * normalized_loss(text_sum, text_count, weighting="combined")
    world = _topology(token_group)["dp"]
    # A mean-reducing DDP hook needs world compensation; manual mode emulates
    # precisely the declared reducer, then applies the same compensation.
    if plan is None and gradient_reduction == "mean":
        loss = loss * world
    optimizer.zero_grad(set_to_none=True)
    if plan is None:
        loss.backward()
    else:
        # A single backward accumulates repeated BF16 compute leaves in BF16.
        # Separate backwards reduce each contribution into the existing FP32
        # sharded gradients, with the same global-target normalization.
        for index, item in enumerate(codec):
            micro_loss = normalized_loss(item.loss_sum, counts, weighting=cfg.codec_weighting)
            if texts:
                micro_loss = micro_loss + cfg.text_weight * normalized_loss(
                    texts[index].loss_sum, text_count, weighting="combined"
                )
            micro_loss.backward()
    reject_nonfinite_gradients(base, device=device, group=token_group)
    if plan is None and dist.is_initialized() and not isinstance(model, DistributedDataParallel):
        for parameter in base.parameters():
            if not parameter.requires_grad:
                continue
            present = torch.tensor(
                int(parameter.grad is not None), device=device, dtype=torch.int64
            )
            dist.all_reduce(present, op=dist.ReduceOp.SUM, group=token_group)
            if not present.item():
                continue  # Globally unused parameters get no decay/moment advancement.
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=token_group)
            if gradient_reduction == "mean":
                parameter.grad.div_(world)
    reject_nonfinite_gradients(base, device=device, group=token_group)
    if cfg.max_grad_norm is not None:
        if plan is not None:
            clip_sharded_grad_norm(base, cfg.max_grad_norm, group=token_group)
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for p in base.parameters() if p.requires_grad],
                cfg.max_grad_norm,
                error_if_nonfinite=True,
            )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()
    return True


def launch(
    cfg,
    data,
    *,
    model,
    optimizer,
    device,
    scheduler=None,
    token_group=None,
    gradient_reduction="sum",
    state=None,
    checkpoint_path=None,
    resume_from=None,
    waveform_objective=None,
    checkpoint_root=None,
    resume_scan=None,
    checkpoint_contract=None,
):
    """Run explicit complete-model training with separate scan/update cursors.

    Configure components before wrapping DDP or applying speech FSDP, then
    construct the optimizer from the current parameter objects. DDP declares its
    actual SUM/mean reducer; plain models use manual reduction. Declared FSDP uses
    only its native SUM reduction and sharded norm. Caller owns process-group
    initialization/timeouts. Replica checkpoint_path/resume_from and native
    FSDP checkpoint_root/resume_scan are separate scan-boundary contracts.
    """
    device = torch.device(device)
    state = SFTState() if state is None else state
    error = None
    try:
        _validate(
            cfg, model, optimizer, scheduler, token_group, gradient_reduction, waveform_objective
        )
        plan = _fsdp_plan(model)
        if plan is not None:
            if checkpoint_path is not None or resume_from is not None:
                raise ValueError(
                    "FSDP uses native checkpoint_root/resume_scan, not full-replica files"
                )
            if checkpoint_root is not None and (
                scheduler is None
                or not isinstance(checkpoint_contract, dict)
                or not checkpoint_contract
            ):
                raise ValueError(
                    "Native FSDP checkpoint needs scheduler and explicit stable contract"
                )
            if resume_scan is not None and (
                checkpoint_root is None or type(resume_scan) is not int or resume_scan < 0
            ):
                raise ValueError("Native resume needs checkpoint_root and a nonnegative scan")
        elif (
            checkpoint_root is not None
            or resume_scan is not None
            or checkpoint_contract is not None
        ):
            raise ValueError("Native speech checkpoint arguments require declared FSDP")
        state.validate_state_dict(state.state_dict())
        for parameter in _base(model).parameters():
            if parameter.device != torch.device(device):
                raise ValueError("Move the complete model to the explicit device before launch")
    except Exception as exc:
        error = exc
    if not dist.is_initialized() and error is not None:
        raise error
    _collective_preflight(
        error,
        device=device,
        group=token_group,
        message="Speech launch preflight rejected on at least one rank",
    )
    options = dict(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        state=state,
        token_group=token_group,
        gradient_reduction=gradient_reduction,
    )
    if resume_scan is not None:
        _native_checkpoint(
            "load",
            root=checkpoint_root,
            scan=resume_scan,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            state=state,
            contract=checkpoint_contract,
            token_group=token_group,
        )
    elif resume_from is not None:
        load_checkpoint(resume_from, **options)
    error = None
    try:
        if any(data.state_dict().get(key) != value for key, value in state.state_dict().items()):
            raise ValueError("Data and task live cursors differ")
        data.validate_state_dict(data.state_dict())
    except Exception as exc:
        error = exc
    _collective_preflight(
        error,
        device=device,
        group=token_group,
        message="Speech reader/task cursor preflight rejected on at least one rank",
    )
    if dist.is_initialized():
        peers = [None] * dist.get_world_size(token_group)
        dist.all_gather_object(
            peers,
            dict(config=vars(cfg), cursors=state.state_dict(), reduction=gradient_reduction),
            group=token_group,
        )
        if any(peer != peers[0] for peer in peers):
            raise ValueError(
                "Speech ranks disagree on task configuration or global scan/update cursors"
            )
    while state.completed_updates < cfg.max_updates and state.scanned_batches < cfg.max_batches:
        scan, update = state.scanned_batches, state.completed_updates
        error = None
        batches = None
        try:
            batches = data.get_batch(scan, device)
        except Exception as exc:
            error = exc
        _collective_preflight(
            error,
            device=device,
            group=token_group,
            message="Speech batch preparation rejected on at least one rank",
        )
        updated = _step(
            cfg,
            batches,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            token_group=token_group,
            gradient_reduction=gradient_reduction,
            waveform_objective=waveform_objective,
        )
        if updated:
            data.commit_update(scan_index=scan, update_index=update)
            state.completed_updates += 1
        state.scanned_batches += 1
        if checkpoint_root is not None:
            _native_checkpoint(
                "save",
                root=checkpoint_root,
                scan=state.scanned_batches,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data=data,
                state=state,
                contract=checkpoint_contract,
                token_group=token_group,
            )
        elif checkpoint_path is not None:
            save_checkpoint(checkpoint_path, **options)
    return state
