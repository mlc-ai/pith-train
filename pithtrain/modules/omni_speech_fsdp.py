"""Explicit complete-speech FSDP2 setup and fixed-topology resume context.

This helper is separate from DualPipeV's two-stage setup. It does not initialize
process groups, load weights, choose hardware, or claim full-size GPU validation.
Configure trainable components before sharding, then construct the optimizer.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)
from torch.distributed.tensor import DTensor, Shard

from pithtrain.contexts import distributed, training
from pithtrain.modules.task_checkpoint import TaskCheckpointState

_MODES = {
    "codec_only": {"talker"},
    "text_and_codec": {"thinker", "talker"},
    "waveform": {"code2wav"},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _children(sequence):
    return tuple(sequence.values()) if isinstance(sequence, nn.ModuleDict) else tuple(sequence)


def _thinker_layer_units(layer):
    # Keep residual joins outside FSDP's input identity hook. Wrapping the whole
    # decoder regroups the BF16 text and speech cotangents at a shared hidden
    # state. These children still share one all-gather/reduce-scatter group.
    return (layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp)


def _shard_units(model):
    """Build a bottom-up plan without mutating modules or parameter identities."""
    _require(
        set(dict(model.named_children())) == {"thinker", "talker", "code2wav"},
        "Complete speech must own exactly thinker, talker and code2wav",
    )
    thinker, talker = model.thinker, model.talker
    _require(
        thinker.stage_count == 1 and thinker.stage_index == 0,
        "Speech FSDP requires a complete phase=-1 Thinker",
    )
    units = []

    def add(module, methods=(), *, container=False):
        _require(isinstance(module, nn.Module), "Missing native speech module")
        for method in methods:
            _require(callable(getattr(module, method, None)), "Missing speech method: " + method)
        units.append((module, tuple(methods), container))

    for edge in (thinker.embed_tokens, thinker.norm, thinker.lm_head):
        add(edge)
    # Whole towers preserve their own internal chunk loop without introducing
    # rank-dependent FSDP collectives inside that loop.
    for name in ("visual", "audio_tower"):
        if getattr(thinker, name, None) is not None:
            add(getattr(thinker, name))
    for layer in _children(thinker.layers):
        norm, attention, post_norm, mlp = _thinker_layer_units(layer)
        for child in (norm, attention, post_norm):
            add(child)
        add(mlp, ("reference_forward",))
    add(thinker, ("forward_prolog", "forward_posemb", "forward_epilog"), container=True)

    for layer in _children(talker.model.layers):
        add(layer)
    add(talker.model.codec_embedding)
    add(talker.model.norm)
    add(talker.model, container=True)
    for name in ("text_projection", "hidden_projection", "codec_head"):
        add(getattr(talker, name))
    predictor = talker.code_predictor
    for layer in _children(predictor.model.layers):
        add(layer, ("_forward_per_head",))
    for embedding in predictor.model.codec_embedding:
        add(embedding)
    add(predictor.model.norm, ("_forward_per_head",))
    add(predictor.model, ("_forward_per_head",), container=True)
    for head in predictor.lm_head:
        add(head)
    add(predictor, ("teacher_force",), container=True)
    add(talker, container=True)
    add(model.code2wav)
    # A common root tracks calls to direct-method children and repeated edges.
    add(model, ("decode_training_codes",), container=True)
    _require(len({id(m) for m, _, _ in units}) == len(units), "Aliased speech modules")
    return units


def _world_mesh(mesh):
    _require(dist.is_initialized(), "Initialize the process group before speech FSDP")
    _require(
        mesh.ndim == 1 and tuple(mesh.mesh_dim_names or ()) == ("dp",),
        "Speech FSDP requires a one-dimensional named dp mesh",
    )
    _require(
        mesh.mesh.tolist() == list(range(dist.get_world_size())),
        "Native speech strict resume requires the complete ordered world DP mesh",
    )
    for axis in ("cp", "ep", "pp"):
        _require(getattr(distributed, axis + "_size", 1) == 1, "Speech FSDP requires CP=EP=PP=1")


def apply_speech_fsdp(model, mesh, *, mode, gradient_reduction="sum", param_dtype=torch.bfloat16):
    """Shard the complete native speech tree; return the same model.

    Parameters stay FP32 outside forward/backward. Each compute unit all-gathers
    BF16 parameters, reduces FP32 gradients with SUM, and reshards after forward.
    A failed FSDP runtime setup may have modified modules: discard that model;
    do not retry wrapping it. This never creates optimizer state or weights.
    """
    _world_mesh(mesh)
    if mesh.device_type == "cpu":
        _require(
            hasattr(FSDPModule, "set_force_sum_reduction_for_comms"),
            "This CPU FSDP runtime has no public Gloo SUM communication support",
        )
    _require(mode in _MODES, "Unknown speech training mode")
    _require(gradient_reduction == "sum", "Speech FSDP uses SUM with global-target normalization")
    _require(param_dtype == torch.bfloat16, "Speech FSDP compute must be BF16")
    _require(not hasattr(model, "_omni_speech_fsdp_plan"), "Speech model is already configured")
    _require(
        not any(isinstance(m, FSDPModule) for m in model.modules()), "Cannot rewrap FSDP modules"
    )
    named = list(model.named_parameters(remove_duplicate=False))
    _require(
        named and len({id(p) for _, p in named}) == len(named),
        "Tied/aliased parameters need an explicit shared-unit contract",
    )
    for name, parameter in named:
        _require(
            not isinstance(parameter, DTensor) and parameter.dtype == torch.float32,
            "Construct FP32 master parameters before sharding",
        )
        _require(
            parameter.requires_grad == (name.split(".", 1)[0] in _MODES[mode]),
            "Configure all active/frozen components before FSDP and optimizer setup",
        )
    units = _shard_units(model)
    names = {id(m): name for name, m in model.named_modules()}
    compute = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    context = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=False,
    )
    layer_groups = [_thinker_layer_units(layer) for layer in _children(model.thinker.layers)]
    group_starts = {id(group[0]): list(group) for group in layer_groups}
    group_members = {id(child) for group in layer_groups for child in group[1:]}
    for module, methods, container in units:
        if id(module) not in group_members:
            fully_shard(
                group_starts.get(id(module), module),
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=context if container else compute,
            )
        for method in methods:
            register_fsdp_forward_method(module, method)
        module.set_gradient_divide_factor(1.0)
        if mesh.device_type == "cpu":
            module.set_force_sum_reduction_for_comms(True)
    model._omni_speech_fsdp_plan = dict(
        schema=1,
        materialized=not any(p.is_meta for p in model.parameters()),
        weights_initialized=not any(p.is_meta for p in model.parameters()),
        mode=mode,
        mesh=mesh,
        group=mesh.get_group(),
        gradient_reduction="sum",
        master_dtype=torch.float32,
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        require_backward=True,
        fixed_topology=True,
        thinker_sharding="grouped_residual_children_v1",
        owned_modules=tuple(names[id(m)] for m, _, _ in units),
        registered_methods=tuple((names[id(m)], methods) for m, methods, _ in units if methods),
    )
    return model


def _local_collective_plan(batches, mode):
    _require(
        mode in _MODES and isinstance(batches, (list, tuple)) and len(batches) > 0,
        "Supply a supported mode and at least one microbatch per rank",
    )
    sequence = []
    for batch in batches:
        _require(isinstance(batch, dict), "Speech microbatch must be a dictionary")
        codes = batch.get("codes")
        _require(
            isinstance(codes, torch.Tensor) and codes.ndim == 3 and codes.shape[0] == 1,
            "Speech FSDP currently requires B=1 codes[B,time,groups]",
        )
        if mode == "waveform":
            _require(codes.shape[1] > 0, "Waveform codes need at least one frame")
            media = ()
        else:
            ids = batch.get("input_ids")
            _require(
                isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[0] == 1,
                "Speech FSDP currently requires B=1 input_ids",
            )
            context = batch.get("model_context")
            _require(isinstance(context, dict), "Explicit model context is required")
            media = tuple(
                name
                for name in ("pixel_values", "pixel_values_videos", "input_features")
                if context.get(name) is not None
            )
        # Local lengths/labels/masks and target counts may differ. These fields
        # alone control this first-cut collective call sequence.
        sequence.append((mode, 1, codes.shape[1] > 0, codes.shape[2], media))
    return tuple(sequence)


def validate_speech_collective_plan(batches, mode, group=None, *, model=None, speaker="ethan"):
    """All-rank reject heterogeneous invocation plans before model forward.

    No labels, tensor contents or RNG values are exchanged. Zero local targets
    are supported; a globally empty objective must still finish zero backward
    after FSDP forward and then restore logical state without an optimizer step.
    """
    # model=None is only the shape-level protocol probe. Native training callers
    # always supply the actual model for the complete parameter-free data contract.
    try:
        plan = _local_collective_plan(batches, mode)
        if model is not None:
            descriptor = (
                getattr(model, "_omni_speech_fsdp_plan", None)
                if isinstance(model, nn.Module)
                else None
            )
            if descriptor is not None:
                _require(
                    descriptor.get("weights_initialized") is True,
                    "Speech shards need explicit initialization",
                )
            from pithtrain.modules.omni_speech_batch_contract import validate_speech_batch

            for batch in batches:
                validate_speech_batch(batch, model, mode, speaker)
        local = dict(plan=plan, error=None)
    except Exception as error:
        local = dict(plan=None, error=str(error))
    if dist.is_initialized():
        _require(group is not None, "Explicit speech DP group is required")
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, local, group=group)
    else:
        gathered = [local]
    _require(
        not any(item["error"] for item in gathered),
        "Invalid rank-local speech plan: " + repr([item["error"] for item in gathered]),
    )
    _require(
        all(item["plan"] == gathered[0]["plan"] for item in gathered),
        "Unsupported heterogeneous speech FSDP invocation plans",
    )
    return local["plan"]


def native_speech_checkpoint_state(task, *, contract):
    """Wrap actual task+reader state, never infer data position from batch size."""
    return TaskCheckpointState(
        task,
        task_kind="omni_speech",
        task_schema=1,
        contract=contract,
        transaction_key="scanned_batches",
        update_key="completed_updates",
        data_key="data_cursor",
        optimizer_policy="conditional_adamw_v1",
        bounded_load=True,
    )


def require_complete_optimizer_state(model, optimizer, updates):
    """Default native strict contract; conditional Adam histories are unsupported.

    Do not manufacture zero gradients or optimizer entries to pass this check.
    An independently versioned conditional-state contract is required to relax it.
    """
    _require(type(updates) is int and updates >= 0, "Invalid completed update count")
    parameters = {p for p in model.parameters() if p.requires_grad}
    members = [p for group in optimizer.param_groups for p in group["params"]]
    _require(
        type(optimizer) is torch.optim.AdamW
        and set(members) == parameters
        and len(members) == len(parameters),
        "Native speech strict resume requires exact active AdamW membership",
    )
    if updates == 0:
        _require(not optimizer.state, "Zero-update checkpoint must have empty Adam state")
        return
    _require(
        set(optimizer.state) == parameters,
        "Conditional/missing Adam state needs a versioned checkpoint contract",
    )
    for parameter, state in optimizer.state.items():
        step = state.get("step")
        _require(
            isinstance(step, torch.Tensor) and step.numel() == 1 and float(step.item()) == updates,
            "Conditional Adam per-parameter steps are unsupported by default strict resume",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            _require(
                isinstance(value, torch.Tensor)
                and value.shape == parameter.shape
                and value.dtype == parameter.dtype,
                "Adam moment layout differs from master parameter",
            )
            if isinstance(parameter, DTensor):
                _require(
                    isinstance(value, DTensor)
                    and value.placements == parameter.placements
                    and value.device_mesh == parameter.device_mesh,
                    "Adam moments must retain the parameter shard layout",
                )


@contextmanager
def speech_checkpoint_context(model, optimizer, scheduler):
    """Temporarily bind actual FSDP objects to native strict save/load entrypoints."""
    plan = getattr(model, "_omni_speech_fsdp_plan", None)
    _require(
        isinstance(plan, dict) and plan.get("schema") == 1 and plan.get("fixed_topology") is True,
        "Use the reviewed complete-speech FSDP setup first",
    )
    _require(plan.get("weights_initialized") is True, "Speech shards need explicit initialization")
    mesh = plan["mesh"]
    _world_mesh(mesh)
    _require(
        type(optimizer) is torch.optim.AdamW
        and type(scheduler) in (torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.LambdaLR)
        and scheduler.optimizer is optimizer,
        "Native strict speech resume requires AdamW and its StepLR/LambdaLR",
    )
    for parameter in model.parameters():
        _require(
            isinstance(parameter, DTensor)
            and parameter.placements == (Shard(0),)
            and parameter.device_mesh == mesh,
            "Every speech parameter must use the exact one-dimensional DP shard mesh",
        )
    values = [
        (training, "model", model),
        (training, "optimizers", (optimizer,)),
        (training, "schedulers", (scheduler,)),
        (distributed, "attn_mesh", mesh),
        (distributed, "expt_mesh", mesh),
    ]
    for axis in ("pp", "cp", "ep"):
        values.extend([(distributed, axis + "_size", 1), (distributed, axis + "_rank", 0)])
    values.extend(
        [(distributed, "dp_size", dist.get_world_size()), (distributed, "dp_rank", dist.get_rank())]
    )
    missing = object()
    previous = [(owner, name, getattr(owner, name, missing)) for owner, name, _ in values]
    try:
        for owner, name, value in values:
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in reversed(previous):
            if value is missing:
                delattr(owner, name)
            else:
                setattr(owner, name, value)


def _local_speech_tensors(model):
    """Return views of parameters and buffers, never a canonical stacked copy."""
    result = {}
    for kind, entries in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in entries:
            _require(name not in result, "Duplicate speech state name")
            local = value.to_local() if isinstance(value, DTensor) else value
            result[name] = (kind, value, local)
    return result


def materialize_speech_shards(model, *, device):
    """Allocate already-sharded meta parameters without a full model replica.

    Call after apply_speech_fsdp and before constructing the optimizer. This
    intentionally creates UNINITIALIZED local storage: use load_speech_local_shards
    with actual rank-local weights, or an explicitly generated local test state.
    It does not convert released checkpoints or claim random weights are released
    weights. On allocation/load failure discard this not-yet-trainable model.
    """
    plan = getattr(model, "_omni_speech_fsdp_plan", None)
    _require(isinstance(plan, dict), "Apply speech FSDP to the meta model first")
    _world_mesh(plan["mesh"])
    device = torch.device(device)
    _require(device.type == plan["mesh"].device_type, "Materialization device differs from mesh")
    if device.type == "cuda":
        _require(
            device.index in (None, torch.cuda.current_device()),
            "Materialization must use this rank's current CUDA device",
        )
    _require(
        all(isinstance(p, DTensor) and p.is_meta for p in model.parameters()),
        "Materialization requires an entirely meta, already-sharded model",
    )
    # to_empty also replaces buffers; preserve concrete constructor constants.
    concrete_buffers = {
        name: local.detach().cpu().clone()
        for name, (kind, _, local) in _local_speech_tensors(model).items()
        if kind == "buffer" and not local.is_meta
    }
    model.to_empty(device=device)
    values = _local_speech_tensors(model)
    with torch.no_grad():
        for name, value in concrete_buffers.items():
            values[name][2].copy_(value)
    _require(
        all(isinstance(p, DTensor) and not p.is_meta for p in model.parameters()),
        "Runtime did not preserve materialized parameter shards",
    )
    plan["weights_initialized"] = False
    plan["materialized"] = True
    return model


def load_speech_local_shards(model, shards, *, rank, world_size):
    """Load exact local tensors into a newly materialized model, with no gather.

    ``shards`` maps every native parameter and buffer name to a CPU tensor of
    that rank's local shape and exact dtype. Rank/world identity is mandatory;
    conversion from a released layout is a separate explicit producer contract.
    Complete local and all-rank validation precedes the first copy. No optimizer
    may exist yet. For transactional resume use the bounded native checkpoint
    loader, not this initialization-only API. A failed copy leaves the model
    uninitialized and must be discarded; do not retry or train that instance.
    """
    plan = getattr(model, "_omni_speech_fsdp_plan", None)
    _require(isinstance(plan, dict), "Use the complete-speech FSDP setup first")
    error = None
    try:
        _require(
            plan.get("weights_initialized") is False, "Local shard load is initialization-only"
        )
        _require(
            type(rank) is int
            and type(world_size) is int
            and rank == dist.get_rank()
            and world_size == dist.get_world_size(),
            "Local shard rank/world identity mismatch",
        )
        values = _local_speech_tensors(model)
        _require(
            isinstance(shards, dict) and shards.keys() == values.keys(), "Local shard keys differ"
        )
        for name, (_, _, local) in values.items():
            value = shards[name]
            _require(
                isinstance(value, torch.Tensor)
                and not isinstance(value, DTensor)
                and value.device.type == "cpu"
                and value.shape == local.shape
                and value.dtype == local.dtype
                and not local.is_meta,
                "Invalid rank-local tensor: " + name,
            )
            _require(
                not value.is_floating_point() or torch.isfinite(value).all().item(),
                "Nonfinite rank-local tensor: " + name,
            )
    except Exception as failure:
        error = str(failure)
    errors = [None] * dist.get_world_size(plan["group"])
    dist.all_gather_object(errors, error, group=plan["group"])
    _require(not any(errors), "Invalid speech local shards: " + repr(errors))
    error = None
    try:
        with torch.no_grad():
            for name, (_, _, local) in values.items():
                local.copy_(shards[name])
    except Exception as failure:
        error = str(failure)
    dist.all_gather_object(errors, error, group=plan["group"])
    _require(not any(errors), "Speech initialization copy failed; discard model: " + repr(errors))
    plan["weights_initialized"] = True
    return model
