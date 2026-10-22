"""Transactional, fixed-topology strict resume over real rank-local shards.

Weights use checkpoint.py's canonical expert mapping. Per-rank optimizer
membership and RNG are never deduplicated across PP/EP ranks. This version
supports AdamW and StepLR/LambdaLR only; it is not a resharding weight export.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from pithtrain.contexts import distributed, training
from pithtrain.modules.checkpoint import to_canonical_model, to_localized_model
from pithtrain.modules.resume_semantics import scheduler_contract
from pithtrain.modules.strict_checkpoint import (
    _check_schema,
    _restore_rng,
    _rng_state,
    _validate_rng,
)
from pithtrain.modules.task_checkpoint import (
    CONDITIONAL_OPTIMIZER_POLICY,
    TASK_STATE_VERSION,
    TaskCheckpointState,
)

MANIFEST = "strict-manifest.json"
VERSION = 1


class DistributedCheckpointError(ValueError):
    """All participating ranks reject the checkpoint or restore transaction."""


def _phase(label, function):
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    failures = [item for item in errors if item is not None]
    if failures:
        raise DistributedCheckpointError(f"{label}: " + "; ".join(failures))
    return value


def _agree(value, label):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    if any(item != values[0] for item in values):
        raise DistributedCheckpointError(f"Ranks disagree on {label}")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resume_contract(cfg):
    """Exact numerical configuration and production Python source identity."""
    config = cfg.training.to_json_dict()
    for key in list(config):
        if key in {"model", "save_location", "save_interval"} or key.startswith(
            ("nsys_", "memory_profile_")
        ):
            del config[key]
    root = Path(__file__).parents[1]
    code = {str(path.relative_to(root)): _hash(path) for path in sorted(root.rglob("*.py"))}
    return dict(
        training=config,
        model=json.loads((cfg.training.model / "config.json").read_text()),
        code=code,
        torch=str(torch.__version__),
        hsdp_replica=cfg.distributed.hsdp_replica,
    )


def _topology():
    result = dict(rank=dist.get_rank(), world_size=dist.get_world_size())
    for axis in ("pp", "dp", "cp", "ep"):
        result[axis + "_size"] = getattr(distributed, axis + "_size")
        result[axis + "_rank"] = getattr(distributed, axis + "_rank")
    for name in ("attn_mesh", "expt_mesh"):
        mesh = getattr(distributed, name)
        result[name] = dict(
            ranks=mesh.mesh.tolist(), names=mesh.mesh_dim_names, device=mesh.device_type
        )
    return result


def _describe(tensor):
    result = dict(shape=tuple(tensor.shape), dtype=str(tensor.dtype), device=tensor.device.type)
    if isinstance(tensor, DTensor):
        mesh = tensor.device_mesh
        result.update(
            stride=tuple(tensor.stride()),
            mesh=mesh.mesh.tolist(),
            names=mesh.mesh_dim_names,
            placements=[str(p) for p in tensor.placements],
            mesh_device=mesh.device_type,
        )
    return result


def _pack(value):
    if isinstance(value, torch.Tensor):
        local = value.to_local() if isinstance(value, DTensor) else value
        tensor = local.detach().cpu().clone()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise DistributedCheckpointError("Nonfinite checkpoint tensor")
        return {"tensor": _describe(value), "value": tensor}
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_pack(item) for item in value)
    return copy.deepcopy(value)


def _unpack(saved, live, path="state"):
    if isinstance(live, torch.Tensor):
        if (
            not isinstance(saved, dict)
            or set(saved) != {"tensor", "value"}
            or saved["tensor"] != _describe(live)
        ):
            raise DistributedCheckpointError(f"{path}: tensor layout/shape/dtype mismatch")
        value = saved["value"]
        local = live.to_local() if isinstance(live, DTensor) else live
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != local.shape
            or value.dtype != local.dtype
        ):
            raise DistributedCheckpointError(f"{path}: local shard shape/dtype mismatch")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise DistributedCheckpointError(f"{path}: nonfinite payload")
        value = value.to(local.device).clone()
        if isinstance(live, DTensor):
            return DTensor.from_local(
                value,
                device_mesh=live.device_mesh,
                placements=live.placements,
                shape=live.shape,
                stride=live.stride(),
                run_check=False,
            )
        return value
    if isinstance(live, dict):
        if not isinstance(saved, dict) or saved.keys() != live.keys():
            raise DistributedCheckpointError(f"{path}: missing/unexpected keys")
        return {key: _unpack(saved[key], item, f"{path}.{key}") for key, item in live.items()}
    if isinstance(live, (list, tuple)):
        if type(saved) is not type(live) or len(saved) != len(live):
            raise DistributedCheckpointError(f"{path}: sequence type/count mismatch")
        return type(live)(
            _unpack(left, right, f"{path}.{index}")
            for index, (left, right) in enumerate(zip(saved, live, strict=True))
        )
    if type(saved) is not type(live) or (isinstance(saved, float) and not math.isfinite(saved)):
        raise DistributedCheckpointError(f"{path}: scalar type/value mismatch")
    return saved


def _validate_packed(saved, live, path="state"):
    """Pure CPU payload validation: no full-device restored state is allocated."""
    if isinstance(live, torch.Tensor):
        if (
            type(saved) is not dict
            or set(saved) != {"tensor", "value"}
            or saved["tensor"] != _describe(live)
        ):
            raise DistributedCheckpointError(f"{path}: tensor layout/shape/dtype mismatch")
        value = saved["value"]
        local = live.to_local() if isinstance(live, DTensor) else live
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.shape != local.shape
            or value.dtype != local.dtype
        ):
            raise DistributedCheckpointError(f"{path}: local shard shape/dtype mismatch")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise DistributedCheckpointError(f"{path}: nonfinite payload")
        return
    if isinstance(live, dict):
        if type(saved) is not dict or saved.keys() != live.keys():
            raise DistributedCheckpointError(f"{path}: missing/unexpected keys")
        for key, value in live.items():
            _validate_packed(saved[key], value, f"{path}.{key}")
        return
    if isinstance(live, (list, tuple)):
        if type(saved) is not type(live) or len(saved) != len(live):
            raise DistributedCheckpointError(f"{path}: sequence type/count mismatch")
        for index, (value, reference) in enumerate(zip(saved, live, strict=True)):
            _validate_packed(value, reference, f"{path}.{index}")
        return
    if type(saved) is not type(live) or (isinstance(saved, float) and not math.isfinite(saved)):
        raise DistributedCheckpointError(f"{path}: scalar type/value mismatch")


def _restore_packed_inplace(saved, template, live=None):
    """Reuse existing local storage; allocate only actually missing Adam tensors.

    Called only after every rank passed _validate_packed and a CPU rollback
    snapshot exists. The template may alias parameter storage, so it is NEVER
    used as a destination for a missing optimizer moment.
    """
    if isinstance(template, torch.Tensor):
        if isinstance(live, torch.Tensor) and _describe(live) == _describe(template):
            local = live.to_local() if isinstance(live, DTensor) else live
            with torch.no_grad():
                local.copy_(saved["value"])
            return live
        return _unpack(saved, template)  # One missing tensor, not an entire state.
    if isinstance(template, dict):
        actual = live if isinstance(live, dict) else {}
        return {
            key: _restore_packed_inplace(saved[key], item, actual.get(key))
            for key, item in template.items()
        }
    if isinstance(template, (list, tuple)):
        actual = live if isinstance(live, (list, tuple)) else [None] * len(template)
        return type(template)(
            _restore_packed_inplace(value, item, old)
            for value, item, old in zip(saved, template, actual, strict=True)
        )
    return copy.deepcopy(saved)


def _membership(model, optimizers):
    names = {parameter: name for name, parameter in model.named_parameters()}
    return [
        [[names[p] for p in group["params"]] for group in optimizer.param_groups]
        for optimizer in optimizers
    ]


def _optimizer_template(optimizer, *, empty=False):
    # No get_state_dict dummy optimizer step: the fresh live optimizer stays empty.
    template = optimizer.state_dict()
    if empty:
        return dict(state={}, param_groups=template["param_groups"])
    state = {}
    for saved_group, group in zip(template["param_groups"], optimizer.param_groups, strict=True):
        for index, parameter in zip(saved_group["params"], group["params"], strict=True):
            device = parameter.device if group.get("capturable") or group.get("fused") else "cpu"
            state[index] = dict(
                step=torch.tensor(0.0, device=device),
                exp_avg=parameter.detach(),
                exp_avg_sq=parameter.detach(),
            )
            if group["amsgrad"]:
                state[index]["max_exp_avg_sq"] = parameter.detach()
    return dict(state=state, param_groups=template["param_groups"])


def _conditional_optimizer_schema(values, optimizers, updates):
    """Capture actual local Adam presence/counters, protected by the rank checksum."""
    groups = []
    for value, optimizer in zip(values, optimizers, strict=True):
        full = _optimizer_template(optimizer)
        if not value["state"].keys() <= full["state"].keys():
            raise DistributedCheckpointError("Unexpected conditional optimizer parameter")
        parameters = []
        for key in full["state"]:
            present = key in value["state"]
            step = None
            if present:
                counter = value["state"][key].get("step")
                if not isinstance(counter, torch.Tensor) or counter.numel() != 1:
                    raise DistributedCheckpointError("Invalid conditional Adam step tensor")
                number = counter.item()
                if not math.isfinite(number) or int(number) != number or not 0 <= number <= updates:
                    raise DistributedCheckpointError(
                        "Conditional Adam step exceeds committed updates"
                    )
                step = int(number)
            parameters.append(dict(id=key, present=present, step=step))
        groups.append(parameters)
    return dict(version=1, policy=CONDITIONAL_OPTIMIZER_POLICY, optimizers=groups)


def _conditional_optimizer_templates(optimizers, schema, updates):
    """Validate every explicit presence row before constructing payload templates."""
    if (
        type(schema) is not dict
        or set(schema) != {"version", "policy", "optimizers"}
        or type(schema["version"]) is not int
        or schema["version"] != 1
        or schema["policy"] != CONDITIONAL_OPTIMIZER_POLICY
        or type(schema["optimizers"]) is not list
        or len(schema["optimizers"]) != len(optimizers)
    ):
        raise DistributedCheckpointError("Invalid conditional optimizer schema")
    templates, counters = [], []
    for optimizer, rows in zip(optimizers, schema["optimizers"], strict=True):
        full = _optimizer_template(optimizer)
        expected = list(full["state"])
        if type(rows) is not list or len(rows) != len(expected):
            raise DistributedCheckpointError("Incomplete conditional optimizer membership")
        present = {}
        for row, key in zip(rows, expected, strict=True):
            if (
                type(row) is not dict
                or set(row) != {"id", "present", "step"}
                or type(row["id"]) is not int
                or row["id"] != key
                or type(row["present"]) is not bool
            ):
                raise DistributedCheckpointError("Conditional optimizer identity/presence differs")
            if row["present"]:
                if type(row["step"]) is not int or not 0 <= row["step"] <= updates:
                    raise DistributedCheckpointError("Conditional optimizer counter out of range")
                present[key] = row["step"]
            elif row["step"] is not None:
                raise DistributedCheckpointError("Absent optimizer parameter has a counter")
        templates.append(
            dict(
                state={key: full["state"][key] for key in present},
                param_groups=full["param_groups"],
            )
        )
        counters.append(present)
    return templates, counters


def _validate_data(value, *, step, global_batch_size, fingerprint, data_state=None):
    if isinstance(data_state, TaskCheckpointState):
        data_state.validate_state_dict(value)
        if value["cursors"]["transaction"] != step:
            raise DistributedCheckpointError("Checkpoint transaction and task cursor disagree")
        return value["cursors"]["updates"]
    if (
        set(value) != {"version", "fingerprint", "consumed_samples"}
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise DistributedCheckpointError("Expected complete PR100 version-1 data state")
    if value["fingerprint"] != fingerprint or not isinstance(fingerprint, str):
        raise DistributedCheckpointError("Data fingerprint mismatch")
    if (
        type(value["consumed_samples"]) is not int
        or value["consumed_samples"] != step * global_batch_size
    ):
        raise DistributedCheckpointError("Step and committed data cursor disagree")
    return step


def _objects(data_state, step, config):
    if type(step) is not int or step < 1 or data_state is None or not isinstance(config, dict):
        raise DistributedCheckpointError(
            "Strict resume requires completed step, data and configuration"
        )
    if getattr(distributed, "cp_size", 1) != 1 or config.get("hsdp_replica", 1) != 1:
        raise NotImplementedError("Strict resume currently requires CP=1 and HSDP replicas=1")
    # Require serializable, rank-independent numerical configuration before I/O.
    json.dumps(config, sort_keys=True, allow_nan=False)
    model, optimizers, schedulers = training.model, training.optimizers, training.schedulers
    if not optimizers or len(optimizers) != len(schedulers):
        raise DistributedCheckpointError("Optimizer/scheduler membership is incomplete")
    if any(type(opt) is not torch.optim.AdamW for opt in optimizers):
        raise NotImplementedError("Fixed-topology strict resume currently supports AdamW only")
    if any(
        type(sch) not in (torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.LambdaLR)
        for sch in schedulers
    ):
        raise NotImplementedError("Strict resume currently supports StepLR/LambdaLR only")
    names = [
        name for groups in _membership(model, optimizers) for group in groups for name in group
    ]
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(names) != len(set(names)) or set(names) != required:
        raise DistributedCheckpointError(
            "Optimizer parameter coverage is missing, duplicated or unexpected"
        )
    device = next(model.parameters()).device
    global_batch_size = None
    if not isinstance(data_state, TaskCheckpointState):
        global_batch_size = config["training"]["global_batch_size"]
        if type(global_batch_size) is not int or global_batch_size < 1:
            raise DistributedCheckpointError("Invalid global batch size")
    data = (
        data_state.state_dict()
    )  # Reject pending, uncommitted reads before collectives with state.
    return model, optimizers, schedulers, device, global_batch_size, data


def _runtime(model, optimizers, schedulers, data_state, device):
    # FSDP state_dict exposes sharded DTensors. No full_tensor/all-gather is used.
    state = model.state_dict()
    return dict(
        model=to_canonical_model(state, model),
        optimizer=[opt.state_dict() for opt in optimizers],
        scheduler=[sch.state_dict() for sch in schedulers],
        data=data_state.state_dict(),
        rng=_rng_state(device),
    ), {name: _describe(value) for name, value in state.items()}


def _apply(values, model, optimizers, schedulers, data_state, device):
    model.load_state_dict(to_localized_model(values["model"], model), strict=True)
    for opt, saved in zip(optimizers, values["optimizer"], strict=True):
        opt.load_state_dict(saved)
    for sch, saved in zip(schedulers, values["scheduler"], strict=True):
        sch.load_state_dict(saved)
    data_state.load_state_dict(values["data"])
    _restore_rng(values["rng"], device)


def _check_training_state(values, templates, step, *, optimizer_steps=None, packed=False):
    for opt_index, (value, template) in enumerate(zip(values["optimizer"], templates, strict=True)):
        if value["state"].keys() != template["state"].keys():
            raise DistributedCheckpointError("Missing/unexpected optimizer parameter state")
        for key, expected in template["state"].items():
            actual = value["state"][key]
            if actual.keys() != expected.keys() or any(
                (actual[name]["tensor"] != _describe(tensor))
                if packed
                else (
                    not isinstance(actual[name], torch.Tensor)
                    or _describe(actual[name]) != _describe(tensor)
                )
                for name, tensor in expected.items()
            ):
                raise DistributedCheckpointError("Optimizer state tensor schema differs")
        for saved, live in zip(value["param_groups"], template["param_groups"], strict=True):
            if {k: v for k, v in saved.items() if k != "lr"} != {
                k: v for k, v in live.items() if k != "lr"
            }:
                raise DistributedCheckpointError("Optimizer group configuration/membership differs")
        expected_steps = (
            {key: step for key in value["state"]}
            if optimizer_steps is None
            else optimizer_steps[opt_index]
        )
        if any(
            (item["step"]["value"] if packed else item["step"]).item() != expected_steps[key]
            for key, item in value["state"].items()
        ):
            raise DistributedCheckpointError("Incomplete optimizer update counters")
    for scheduler, optimizer in zip(values["scheduler"], values["optimizer"], strict=True):
        if scheduler["last_epoch"] != step or scheduler["_last_lr"] != [
            group["lr"] for group in optimizer["param_groups"]
        ]:
            raise DistributedCheckpointError("Scheduler/optimizer update position differs")


def _write_tensor_file(path, payload):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save(root, step, *, data_state, config):
    if not dist.is_initialized():
        raise DistributedCheckpointError("Initialize the real process group before strict saving")
    objects = _phase("save arguments", lambda: _objects(data_state, step, config))
    model, opts, scheds, device, batch, data = objects
    task_mode = isinstance(data_state, TaskCheckpointState)
    conditional = task_mode and data_state.optimizer_policy == CONDITIONAL_OPTIMIZER_POLICY
    version = TASK_STATE_VERSION if task_mode else VERSION
    updates = _phase(
        "committed cursor",
        lambda: _validate_data(
            data,
            step=step,
            global_batch_size=batch,
            fingerprint=data["fingerprint"],
            data_state=data_state,
        ),
    )
    contract = _phase(
        "save contract arguments",
        lambda: dict(
            step=step,
            config=config,
            data=data_state.checkpoint_contract(data, include_cursors=True) if task_mode else data,
            root=str(Path(root).resolve()),
        ),
    )
    _agree(contract, "save contract")
    values, layout = _phase(
        "capture shards", lambda: _runtime(model, opts, scheds, data_state, device)
    )
    optimizer_schema = None
    optimizer_steps = None
    expected_optimizers = [
        _optimizer_template(opt, empty=task_mode and updates == 0) for opt in opts
    ]
    if conditional:
        optimizer_schema = _phase(
            "capture conditional optimizer",
            lambda: _conditional_optimizer_schema(values["optimizer"], opts, updates),
        )
        expected_optimizers, optimizer_steps = _phase(
            "conditional optimizer templates",
            lambda: _conditional_optimizer_templates(opts, optimizer_schema, updates),
        )
    _phase(
        "complete optimizer",
        lambda: _check_training_state(
            values,
            expected_optimizers,
            updates,
            optimizer_steps=optimizer_steps,
        ),
    )
    header = _phase(
        "rank header",
        lambda: dict(
            version=version,
            step=step,
            config=config,
            topology=_topology(),
            membership=_membership(model, opts),
            optimizer_types=[type(opt).__qualname__ for opt in opts],
            scheduler_types=[type(sch).__qualname__ for sch in scheds],
            scheduler_contracts=[scheduler_contract(sch) for sch in scheds],
            layout=layout,
        ),
    )
    payload = _phase(
        "stage shards",
        lambda: dict(
            header=header,
            state=_pack(
                {k: v for k, v in values.items() if k != "rng" and (not task_mode or k != "data")}
            ),
            rng=values["rng"],
            **({"task": values["data"]} if task_mode else {}),
            **({"optimizer_schema": optimizer_schema} if conditional else {}),
        ),
    )
    location = Path(root, "torch-dcp", f"{step:08d}")
    _phase(
        "reserve step",
        lambda: location.mkdir(parents=True, exist_ok=False) if dist.get_rank() == 0 else None,
    )
    name = f"rank-{dist.get_rank():05d}.pt"
    _phase("write shards", lambda: _write_tensor_file(location / name, payload))
    checksum = _phase("hash shards", lambda: _hash(location / name))
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, checksum)

    def commit():
        if dist.get_rank() != 0:
            return
        manifest = dict(
            version=version,
            world_size=dist.get_world_size(),
            step=step,
            files={f"rank-{rank:05d}.pt": digest for rank, digest in enumerate(hashes)},
        )
        temporary = location / (MANIFEST + ".partial")
        with temporary.open("w") as stream:
            json.dump(manifest, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(location / MANIFEST)

    _phase("publish completion", commit)
    return location


def load(root, step, *, data_state, config):
    if not dist.is_initialized():
        raise DistributedCheckpointError("Initialize the real process group before strict loading")
    model, opts, scheds, device, batch, data = _phase(
        "load arguments", lambda: _objects(data_state, step, config)
    )
    task_mode = isinstance(data_state, TaskCheckpointState)
    conditional = task_mode and data_state.optimizer_policy == CONDITIONAL_OPTIMIZER_POLICY
    version = TASK_STATE_VERSION if task_mode else VERSION
    bounded = task_mode and data_state.bounded_load
    contract = _phase(
        "load contract arguments",
        lambda: dict(
            step=step,
            config=config,
            fingerprint=(
                data_state.checkpoint_contract(data, include_cursors=False)
                if task_mode
                else data["fingerprint"]
            ),
            root=str(Path(root).resolve()),
        ),
    )
    _agree(contract, "load contract")
    location = Path(root, "torch-dcp", f"{step:08d}")

    def read():
        manifest = json.loads((location / MANIFEST).read_text())
        if not isinstance(manifest, dict) or any(
            type(manifest.get(name)) is not int for name in ("version", "world_size", "step")
        ):
            raise DistributedCheckpointError("Invalid strict manifest scalar schema")
        names = {f"rank-{rank:05d}.pt" for rank in range(dist.get_world_size())}
        if (
            set(manifest) != {"version", "world_size", "step", "files"}
            or manifest["version"] != version
            or manifest["world_size"] != dist.get_world_size()
            or manifest["step"] != step
            or set(manifest["files"]) != names
        ):
            raise DistributedCheckpointError("Incomplete or incompatible strict manifest")
        if {p.name for p in location.iterdir()} != names | {MANIFEST}:
            raise DistributedCheckpointError("Missing/unexpected rank files")
        name = f"rank-{dist.get_rank():05d}.pt"
        if _hash(location / name) != manifest["files"][name]:
            raise DistributedCheckpointError("Rank shard checksum mismatch")
        return torch.load(location / name, map_location="cpu", weights_only=True)

    payload = _phase("read and verify files", read)
    before, layout = _phase(
        "capture live layout", lambda: _runtime(model, opts, scheds, data_state, device)
    )
    templates = _phase("optimizer templates", lambda: [_optimizer_template(opt) for opt in opts])
    header = _phase(
        "rank header",
        lambda: dict(
            version=version,
            step=step,
            config=config,
            topology=_topology(),
            membership=_membership(model, opts),
            optimizer_types=[type(opt).__qualname__ for opt in opts],
            scheduler_types=[type(sch).__qualname__ for sch in scheds],
            scheduler_contracts=[scheduler_contract(sch) for sch in scheds],
            layout=layout,
        ),
    )

    bounded_template = None

    def validate():
        nonlocal bounded_template
        if not isinstance(payload, dict) or "header" not in payload:
            raise DistributedCheckpointError("Invalid rank payload")
        _check_schema(payload["header"], header, path="header")
        fields = {"header", "state", "rng"} | ({"task"} if task_mode else set())
        if conditional:
            fields.add("optimizer_schema")
        if set(payload) != fields or payload["header"] != header:
            raise DistributedCheckpointError(
                "Rank topology, configuration, membership or model layout differs"
            )
        updates = step
        expected_optimizers = templates
        optimizer_steps = None
        if task_mode:
            updates = _validate_data(
                payload["task"],
                step=step,
                global_batch_size=None,
                fingerprint=data["fingerprint"],
                data_state=data_state,
            )
            if updates == 0:
                expected_optimizers = [_optimizer_template(opt, empty=True) for opt in opts]
        if conditional:
            expected_optimizers, optimizer_steps = _conditional_optimizer_templates(
                opts, payload["optimizer_schema"], updates
            )
        template = dict(before, optimizer=expected_optimizers)
        del template["rng"]
        if task_mode:
            del template["data"]
        if bounded:
            _validate_packed(payload["state"], template)
            bounded_template = template
            restored = dict(payload["state"])
        else:
            restored = _unpack(payload["state"], template)
        if task_mode:
            restored["data"] = copy.deepcopy(payload["task"])
        else:
            _validate_data(
                restored["data"],
                step=step,
                global_batch_size=batch,
                fingerprint=data["fingerprint"],
            )
        _check_training_state(
            restored, expected_optimizers, updates, optimizer_steps=optimizer_steps, packed=bounded
        )
        _check_schema(payload["rng"], before["rng"], path="app.resume.rng")
        _validate_rng(payload["rng"], device)
        restored["rng"] = payload["rng"]
        return restored

    restored = _phase("validate every rank before mutation", validate)
    if task_mode:
        restored_contract = _phase(
            "restored task contract",
            lambda: data_state.checkpoint_contract(restored["data"], include_cursors=True),
        )
        _agree(restored_contract, "restored task global cursors")
    rollback = _phase(
        "stage rollback", lambda: _pack({k: v for k, v in before.items() if k != "rng"})
    )

    def apply_bounded(values):
        # Canonical model tensors are local views of the actual live shards.
        # _restore_packed_inplace already copied them. Do not restack all MoE
        # experts through to_localized_model and allocate another model copy.
        for optimizer, saved in zip(opts, values["optimizer"], strict=True):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(scheds, values["scheduler"], strict=True):
            scheduler.load_state_dict(saved)
        data_state.load_state_dict(values["data"])
        _restore_rng(values["rng"], device)

    def apply_restored():
        if not bounded:
            return _apply(restored, model, opts, scheds, data_state, device)
        values = _restore_packed_inplace(payload["state"], bounded_template, before)
        values["data"] = restored["data"]
        values["rng"] = restored["rng"]
        return apply_bounded(values)

    try:
        _phase("apply strict checkpoint", apply_restored)
    except DistributedCheckpointError:

        def undo():
            original = {k: v for k, v in before.items() if k != "rng"}
            old = (
                _restore_packed_inplace(rollback, original, original)
                if bounded
                else _unpack(rollback, original)
            )
            old["rng"] = before["rng"]
            if bounded:
                apply_bounded(old)
            else:
                _apply(old, model, opts, scheds, data_state, device)

        _phase("rollback all ranks", undo)
        raise
    return step
