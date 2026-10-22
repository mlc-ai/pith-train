"""CPU bridge for Omni fixed-topology strict containers and HF resume bundles.

The HF companion stores training state plus model-shard descriptors/digests,
not duplicate model weights. Import reconstructs each original shard from HF
and rejects changed weights instead of pairing them with stale optimizer/RNG.
"""

import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.distributed.tensor import Replicate, Shard

from pithtrain.modules.checkpoint import strip_prefix
from pithtrain.modules.distributed_checkpoint import MANIFEST, VERSION, _write_tensor_file
from pithtrain.modules.resume_semantics import _canonical_model_value, seal
from pithtrain.modules.strict_checkpoint import _numpy_state

from .qwen3_omni_moe import (
    ConversionError,
    read_json,
    require_empty_destination,
    text_config,
    validate_tensors,
)
from .qwen3_omni_moe import (
    parameter_shapes as text_parameter_shapes,
)


def parameter_shapes(config):
    if config.model_type == "qwen3_omni_moe_thinker":
        from .qwen3_omni_thinker import parameter_shapes as thinker_shapes

        return thinker_shapes(config)
    return text_parameter_shapes(config)


RESUME_DIR = "pithtrain-resume"
RESUME_MANIFEST = "resume-manifest.json"
_EXPERT = re.compile(r"(layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_up_proj|down_proj)$")


def require(condition, message):
    if not condition:
        raise ConversionError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu", "Expected CPU tensor")
    return hashlib.sha256(
        value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def same_config(left, right):
    left, right = left.to_dict(), right.to_dict()
    ignored = {"transformers_version", "_name_or_path"}
    require(
        all(left.get(k) == right.get(k) for k in (left.keys() | right.keys()) - ignored),
        "Text config disagrees with strict checkpoint",
    )


def _descriptor(desc, rank):
    plain = {"shape", "dtype", "device"}
    sharded = plain | {"stride", "mesh", "names", "placements", "mesh_device"}
    require(isinstance(desc, dict) and set(desc) in (plain, sharded), "Invalid tensor descriptor")
    shape = desc["shape"]
    require(
        type(shape) is tuple and all(type(n) is int and n >= 0 for n in shape),
        "Invalid tensor shape",
    )
    require(
        desc["dtype"] in {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"},
        "Invalid model/moment dtype",
    )
    require(desc["device"] in {"cpu", "cuda"}, "Invalid tensor device")
    local, begin, end = shape, 0, shape[0] if shape else 1
    if "mesh" in desc:
        mesh = desc["mesh"]
        require(
            isinstance(mesh, list)
            and mesh
            and all(type(i) is int and i >= 0 for i in mesh)
            and len(set(mesh)) == len(mesh)
            and rank in mesh,
            "Only one-dimensional, rank-owned DTensor meshes are supported",
        )
        require(
            desc["mesh_device"] == desc["device"]
            and type(desc["names"]) is tuple
            and len(desc["names"]) == 1
            and isinstance(desc["names"][0], str),
            "Invalid mesh descriptor",
        )
        require(
            type(desc["stride"]) is tuple
            and len(desc["stride"]) == len(shape)
            and all(type(n) is int and n >= 0 for n in desc["stride"]),
            "Invalid global stride",
        )
        require(
            desc["placements"] in ([str(Shard(0))], [str(Replicate())]),
            "Unsupported DTensor placement",
        )
        if desc["placements"] == [str(Shard(0))]:
            require(bool(shape), "Cannot shard a scalar")
            size, begin = Shard.local_shard_size_and_offset(shape[0], len(mesh), mesh.index(rank))
            end, local = begin + size, (size, *shape[1:])
    return shape, local, begin, end


def _tensor(entry, rank, expected=None, *, skeleton=False):
    field = "sha256" if skeleton else "value"
    require(
        isinstance(entry, dict) and set(entry) == {"tensor", field},
        "Missing/unexpected tensor fields",
    )
    shape, local, begin, end = _descriptor(entry["tensor"], rank)
    if expected is not None:
        require(entry["tensor"] == expected, "Tensor layout/moment dtype differs from model")
    if skeleton:
        require(
            isinstance(entry[field], str) and re.fullmatch(r"[0-9a-f]{64}", entry[field]),
            "Invalid model shard digest",
        )
    else:
        value = entry["value"]
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and tuple(value.shape) == local
            and str(value.dtype) == entry["tensor"]["dtype"],
            "Local tensor shape/dtype mismatch",
        )
        require(bool(torch.isfinite(value).all()), "Nonfinite checkpoint tensor")
    return shape, begin, end


def _topology(topology, rank, world):
    fields = {
        "rank",
        "world_size",
        "pp_size",
        "pp_rank",
        "dp_size",
        "dp_rank",
        "cp_size",
        "cp_rank",
        "ep_size",
        "ep_rank",
        "attn_mesh",
        "expt_mesh",
    }
    require(isinstance(topology, dict) and set(topology) == fields, "Incomplete topology")
    require(
        all(type(topology[name]) is int for name in fields - {"attn_mesh", "expt_mesh"}),
        "Invalid topology scalar type",
    )
    pp, ep = topology["pp_size"], topology["ep_size"]
    require(
        pp > 0 and ep > 0 and world % pp == 0 and (world // pp) % ep == 0, "Invalid PP/EP topology"
    )
    stage = world // pp
    require(
        (
            topology["rank"],
            topology["world_size"],
            topology["pp_rank"],
            topology["dp_size"],
            topology["dp_rank"],
            topology["cp_size"],
            topology["cp_rank"],
            topology["ep_rank"],
        )
        == (rank, world, rank // stage, stage, rank % stage, 1, 0, rank % ep),
        "Wrong rank coordinates; CP must remain one",
    )
    for name, shape, names in (
        ("attn_mesh", (pp, stage, 1), ("pp", "dp", "cp")),
        ("expt_mesh", (pp, stage // ep, ep), ("pp", "dp", "ep")),
    ):
        mesh = topology[name]
        require(
            isinstance(mesh, dict) and set(mesh) == {"ranks", "names", "device"},
            "Invalid topology mesh",
        )
        require(
            mesh["ranks"] == torch.arange(world).reshape(shape).tolist()
            and mesh["names"] == names
            and mesh["device"] in {"cpu", "cuda"},
            "Mesh rank ordering differs",
        )
    require(
        topology["attn_mesh"]["device"] == topology["expt_mesh"]["device"],
        "Inconsistent mesh devices",
    )
    return pp, ep


def _model_plan(header, config):
    topology, layout = header["topology"], header["layout"]
    rank, ep = topology["rank"], topology["ep_size"]
    text_cfg = getattr(config, "text_config", config)
    expected, plan, names = parameter_shapes(config), {}, {}
    require(text_cfg.num_experts % ep == 0, "Expert count must divide EP")
    per_ep = text_cfg.num_experts // ep
    require(isinstance(layout, dict) and layout, "Missing runtime model layout")
    for name, desc in layout.items():
        require(isinstance(name, str), "Invalid model name")
        key = strip_prefix(name)
        if config.model_type == "qwen3_omni_moe_thinker" and key.startswith(
            ("visual.", "audio_tower.")
        ):
            require(topology["pp_rank"] == 0, "Media encoders must belong to the first PP rank")
        require(key not in names, "Duplicate canonical parameter across local pipeline chunks")
        names[key] = name
        shape, _, begin, end = _descriptor(desc, rank)
        require(
            desc["device"] == topology["attn_mesh"]["device"], "Model device differs from topology"
        )
        stage_start = topology["pp_rank"] * topology["dp_size"]
        if ".experts." in key:
            root, proj = key.rsplit(".", 1)
            require(
                proj in {"gate_up_proj", "down_proj"} and f"{root}.0.{proj}" in expected,
                "Unknown native expert key",
            )
            require(
                shape == (per_ep, *expected[f"{root}.0.{proj}"]),
                "Native [experts,out,in] shape mismatch",
            )
            if "mesh" in desc:
                require(
                    desc["mesh"]
                    == list(
                        range(
                            stage_start + topology["ep_rank"], stage_start + topology["dp_size"], ep
                        )
                    ),
                    "Expert shard mesh differs from EP ownership",
                )
            # Canonical expert entries contain whole [out,in] slices. Their
            # original stacked FSDP descriptor determines which indices exist.
            for index in range(begin, end):
                canonical = f"{root}.{topology['ep_rank'] * per_ep + index}.{proj}"
                plan[canonical] = expected[canonical]
        else:
            require(key in expected and shape == expected[key], "Runtime model key/shape mismatch")
            if "mesh" in desc:
                require(
                    desc["mesh"] == list(range(stage_start, stage_start + topology["dp_size"])),
                    "Attention FSDP mesh differs from topology",
                )
            plan[key] = expected[key]
    return plan


def _rng(rng, device):
    require(
        isinstance(rng, dict) and set(rng) == {"python", "numpy", "torch_cpu", "cuda"},
        "Missing/unexpected RNG components",
    )
    try:
        torch.Generator(device="cpu").set_state(rng["torch_cpu"])
        random.Random(0).setstate(rng["python"])
        np.random.RandomState(0).set_state(_numpy_state(rng["numpy"]))
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ConversionError("Invalid CPU/Python/NumPy RNG state") from error
    cuda = rng["cuda"]
    if device == "cpu":
        require(cuda is None, "Unexpected CUDA RNG on CPU checkpoint")
    else:
        # CUDA generator state is opaque here: verify integrity/structure on CPU;
        # the existing runtime validates its native representation on CUDA load.
        require(
            isinstance(cuda, torch.Tensor)
            and cuda.device.type == "cpu"
            and cuda.dtype == torch.uint8
            and cuda.ndim == 1
            and cuda.numel() > 0,
            "Missing/invalid CUDA RNG bytes",
        )


def _training(payload):
    header, state = payload["header"], payload["state"]
    rank, step = header["topology"]["rank"], header["step"]
    require(
        isinstance(state, dict) and set(state) == {"model", "optimizer", "scheduler", "data"},
        "Missing/unexpected training state",
    )
    batch = header["config"]["training"]["global_batch_size"]
    require(type(batch) is int and batch > 0, "Invalid global batch size")
    data = state["data"]
    require(
        isinstance(data, dict)
        and set(data) == {"version", "fingerprint", "consumed_samples"}
        and type(data["version"]) is int
        and data["version"] == 1
        and isinstance(data["fingerprint"], str)
        and type(data["consumed_samples"]) is int
        and data["consumed_samples"] == step * batch,
        "Invalid committed data position/fingerprint",
    )
    membership, optimizers, schedulers = (
        header["membership"],
        state["optimizer"],
        state["scheduler"],
    )
    require(
        isinstance(membership, list)
        and membership
        and isinstance(optimizers, list)
        and isinstance(schedulers, list)
        and len(membership) == len(optimizers) == len(schedulers),
        "Missing optimizer/scheduler membership",
    )
    require(
        header["optimizer_types"] == ["AdamW"] * len(optimizers)
        and len(header["scheduler_types"]) == len(schedulers)
        and isinstance(header["scheduler_contracts"], list)
        and len(header["scheduler_contracts"]) == len(schedulers)
        and all(name in {"StepLR", "LambdaLR"} for name in header["scheduler_types"]),
        "Unsupported optimizer/scheduler class",
    )
    flat_names = []
    probe = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    group_fields = set(probe.param_groups[0]) | {"initial_lr"}
    for groups, optimizer, scheduler, scheduler_type, policy in zip(
        membership,
        optimizers,
        schedulers,
        header["scheduler_types"],
        header["scheduler_contracts"],
        strict=True,
    ):
        require(
            isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"},
            "Incomplete optimizer state",
        )
        require(isinstance(optimizer["state"], dict), "Invalid optimizer state mapping")
        require(
            isinstance(groups, list)
            and isinstance(optimizer["param_groups"], list)
            and len(groups) == len(optimizer["param_groups"]),
            "Optimizer group count mismatch",
        )
        ids = []
        for names, group in zip(groups, optimizer["param_groups"], strict=True):
            require(
                isinstance(names, list)
                and isinstance(group, dict)
                and isinstance(group.get("params"), list)
                and len(names) == len(group["params"]),
                "Invalid optimizer parameter group",
            )
            require(set(group) == group_fields, "Missing/unexpected AdamW group settings")
            require(
                type(group["betas"]) is tuple
                and len(group["betas"]) == 2
                and all(isinstance(v, (int, float)) and 0 <= v < 1 for v in group["betas"]),
                "Invalid AdamW betas",
            )
            require(
                all(
                    type(group[k]) in (int, float) and math.isfinite(group[k]) and group[k] >= 0
                    for k in ("eps", "weight_decay", "initial_lr")
                ),
                "Invalid AdamW scalar settings",
            )
            require(
                all(
                    type(group[k]) is bool
                    for k in ("maximize", "capturable", "differentiable", "decoupled_weight_decay")
                )
                and all(group[k] is None or type(group[k]) is bool for k in ("foreach", "fused")),
                "Invalid AdamW flags",
            )
            require(
                all(isinstance(name, str) and name in header["layout"] for name in names),
                "Unknown optimizer parameter name",
            )
            require(
                type(group.get("amsgrad")) is bool
                and isinstance(group.get("lr"), (int, float))
                and not isinstance(group["lr"], bool)
                and math.isfinite(group["lr"])
                and group["lr"] >= 0,
                "Invalid AdamW settings",
            )
            flat_names.extend(names)
            for index, name in zip(group["params"], names, strict=True):
                require(
                    type(index) is int and index >= 0 and index in optimizer["state"],
                    "Missing optimizer parameter state",
                )
                ids.append(index)
                entry = optimizer["state"][index]
                required = {"step", "exp_avg", "exp_avg_sq"} | (
                    {"max_exp_avg_sq"} if group["amsgrad"] else set()
                )
                require(
                    isinstance(entry, dict) and set(entry) == required,
                    "Missing/unexpected AdamW moments",
                )
                for key in required - {"step"}:
                    _tensor(entry[key], rank, header["layout"][name])
                _tensor(entry["step"], rank)
                require(
                    entry["step"]["tensor"]["shape"] == ()
                    and entry["step"]["value"].item() == step,
                    "Optimizer update counter differs",
                )
        require(
            ids == list(range(len(ids))) and set(ids) == set(optimizer["state"]),
            "Missing/reordered/unexpected optimizer parameter IDs",
        )
        template = (
            torch.optim.lr_scheduler.StepLR(probe, 1)
            if scheduler_type == "StepLR"
            else torch.optim.lr_scheduler.LambdaLR(probe, lambda _: 1)
        ).state_dict()
        require(
            isinstance(scheduler, dict)
            and set(scheduler) - {"_pith_factory_contract"} == set(template),
            "Missing/unexpected scheduler fields",
        )
        _scheduler_policy(policy, scheduler, scheduler_type, groups)
        require(
            all(type(scheduler[k]) is bool for k in ("_is_initial", "_get_lr_called_within_step")),
            "Invalid scheduler flags",
        )
        if scheduler_type == "StepLR":
            require(
                type(scheduler["step_size"]) is int
                and scheduler["step_size"] > 0
                and type(scheduler["gamma"]) in (int, float)
                and math.isfinite(scheduler["gamma"])
                and scheduler["gamma"] >= 0,
                "Invalid StepLR settings",
            )
        else:
            require(
                isinstance(scheduler["lr_lambdas"], list)
                and len(scheduler["lr_lambdas"]) == len(groups)
                and all(
                    value is None or isinstance(value, dict) for value in scheduler["lr_lambdas"]
                ),
                "Invalid LambdaLR callable state",
            )
        require(
            scheduler["base_lrs"] == [g["initial_lr"] for g in optimizer["param_groups"]]
            and type(scheduler["_step_count"]) is int
            and scheduler["_step_count"] == step + 1,
            "Scheduler base learning rates/counter differ",
        )
        require(
            isinstance(scheduler, dict)
            and type(scheduler.get("last_epoch")) is int
            and scheduler["last_epoch"] == step
            and scheduler.get("_last_lr") == [g["lr"] for g in optimizer["param_groups"]],
            "Scheduler update position differs",
        )
    require(
        len(flat_names) == len(set(flat_names)) and set(flat_names) == set(header["layout"]),
        "Optimizer model coverage differs",
    )
    _rng(payload["rng"], header["topology"]["attn_mesh"]["device"])


def _sealed_identity(contract, label):
    require(
        isinstance(contract, dict) and set(contract) == {"identity", "sha256"},
        f"Missing/invalid {label} contract",
    )
    try:
        expected = seal(contract["identity"])
    except (TypeError, ValueError) as error:
        raise ConversionError(f"Invalid {label} contract encoding") from error
    require(contract == expected, f"{label} contract digest differs")
    require(isinstance(contract["identity"], dict), f"Invalid {label} identity")
    return contract["identity"]


def _scheduler_policy(contract, state, name, groups):
    """Validate and retain opaque callable semantics without executing checkpoint code.

    Conversion can run on a different CPU/CUDA build. Runtime compatibility and
    the actual callable equality remain mandatory at strict restore admission.
    """
    identity = _sealed_identity(contract, "scheduler")
    require(
        set(identity)
        == {"schema", "kind", "torch", "implementation", "configuration", "callables", "factory"}
        and type(identity["schema"]) is int
        and identity["schema"] == 1
        and identity["kind"] == "scheduler_policy"
        and isinstance(identity["torch"], str)
        and bool(identity["torch"]),
        "Invalid scheduler policy identity",
    )
    implementation = identity["implementation"]
    require(
        isinstance(implementation, dict)
        and set(implementation) == {"name", "source_sha256"}
        and implementation["name"] == f"torch.optim.lr_scheduler.{name}"
        and isinstance(implementation["source_sha256"], str)
        and re.fullmatch("[0-9a-f]{64}", implementation["source_sha256"]) is not None,
        "Invalid scheduler implementation identity",
    )
    mutable = {"last_epoch", "_step_count", "_last_lr", "_get_lr_called_within_step"}
    configuration = {
        key: value
        for key, value in state.items()
        if key not in mutable | {"lr_lambdas", "_pith_factory_contract"}
    }
    try:
        canonical = _canonical_model_value(configuration)
    except (TypeError, ValueError) as error:
        raise ConversionError("Invalid scheduler configuration") from error
    require(identity["configuration"] == canonical, "Scheduler policy/state configuration differs")
    factory = identity["factory"]
    require(factory == state.get("_pith_factory_contract"), "Scheduler factory/state differs")
    if factory is not None:
        factory_identity = _sealed_identity(factory, "scheduler factory")
        require(
            set(factory_identity) == {"schema", "implementation", "arguments"}
            and type(factory_identity["schema"]) is int
            and factory_identity["schema"] == 1
            and isinstance(factory_identity["implementation"], dict)
            and set(factory_identity["implementation"]) == {"name", "source_sha256"}
            and isinstance(factory_identity["implementation"]["name"], str)
            and isinstance(factory_identity["implementation"]["source_sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", factory_identity["implementation"]["source_sha256"])
            is not None,
            "Invalid scheduler factory identity",
        )
    callables = identity["callables"]
    if name == "StepLR":
        require(callables is None, "Unexpected StepLR callable policy")
    else:
        require(
            isinstance(callables, list) and len(callables) == len(groups),
            "Missing LambdaLR callable policies",
        )
        for contract in callables:
            callable_identity = _sealed_identity(contract, "scheduler callable")
            require(
                set(callable_identity) == {"schema", "kind", "python", "callable"}
                and type(callable_identity["schema"]) is int
                and callable_identity["schema"] == 1
                and callable_identity["kind"] == "python_callable"
                and isinstance(callable_identity["python"], list)
                and len(callable_identity["python"]) == 2
                and all(type(v) is int for v in callable_identity["python"])
                and isinstance(callable_identity["callable"], dict)
                and bool(callable_identity["callable"]),
                "Invalid scheduler callable identity",
            )


def validate_payloads(payloads, step, *, skeleton=False, multimodal=False):
    require(
        type(step) is int and step > 0 and isinstance(payloads, list) and payloads,
        "Invalid strict step/ranks",
    )
    config, common, data, geometry = None, None, None, None
    pieces = {}
    for rank, payload in enumerate(payloads):
        require(
            isinstance(payload, dict) and set(payload) == {"header", "state", "rng"},
            "Invalid rank payload",
        )
        header = payload["header"]
        require(
            isinstance(header, dict)
            and set(header)
            == {
                "version",
                "step",
                "config",
                "topology",
                "membership",
                "optimizer_types",
                "scheduler_types",
                "scheduler_contracts",
                "layout",
            }
            and type(header["version"]) is int
            and header["version"] == VERSION
            and type(header["step"]) is int
            and header["step"] == step,
            "Invalid strict header",
        )
        pp_ep = _topology(header["topology"], rank, len(payloads))
        require(
            isinstance(header["config"], dict)
            and "model" in header["config"]
            and isinstance(header["config"].get("training"), dict)
            and "global_batch_size" in header["config"]["training"],
            "Missing strict model/training config",
        )
        require(
            type(header["config"].get("hsdp_replica", 1)) is int
            and header["config"].get("hsdp_replica", 1) == 1,
            "HSDP>1 remains unsupported",
        )
        if multimodal:
            from .qwen3_omni_thinker import thinker_config

            current = thinker_config(header["config"]["model"])
        else:
            current = text_config(header["config"]["model"])
        if config is None:
            config, common, geometry = current, header["config"], pp_ep
        else:
            require(
                header["config"] == common and pp_ep == geometry,
                "Rank configuration/topology disagreement",
            )
            same_config(current, config)
        plan = _model_plan(header, config)
        _training(payload)
        if data is None:
            data = payload["state"]["data"]
        require(payload["state"]["data"] == data, "Ranks disagree on committed data")
        model = payload["state"]["model"]
        require(isinstance(model, dict), f"Rank {rank}: invalid canonical model mapping")
        require(
            model.keys() == plan.keys(),
            f"Rank {rank}: missing={sorted(plan.keys() - model.keys())}; unexpected={sorted(model.keys() - plan.keys())}",
        )
        layouts = {strip_prefix(name): desc for name, desc in header["layout"].items()}
        for key, entry in model.items():
            try:
                shape, begin, end = _tensor(entry, rank, skeleton=skeleton)
            except ConversionError as error:
                raise ConversionError(f"rank {rank}, {key}: {error}") from error
            require(shape == plan[key], f"{key}: canonical global shape mismatch")
            require(
                entry["tensor"]["device"] == header["topology"]["attn_mesh"]["device"],
                f"{key}: model device differs",
            )
            if _EXPERT.fullmatch(key):
                require(
                    begin == 0 and end == shape[0],
                    "Canonical experts must be complete [out,in] slices",
                )
            else:
                descriptor = layouts[key]
                require(
                    entry["tensor"] == descriptor,
                    f"{key}: canonical layout differs from runtime model",
                )
            pieces.setdefault(key, []).append((begin, end, rank, entry))
    expected = parameter_shapes(config)
    require(pieces.keys() == expected.keys(), "Missing/unexpected global text weights")
    for key, items in pieces.items():
        require(
            len({payloads[rank]["header"]["topology"]["pp_rank"] for _, _, rank, _ in items}) == 1,
            f"{key}: duplicated across PP stages",
        )
        intervals, cursor, dtype = {}, 0, None
        for begin, end, rank, entry in sorted(items, key=lambda item: (item[0], item[1], item[2])):
            require(
                dtype is None or dtype == entry["tensor"]["dtype"],
                f"{key}: mixed replica/shard dtype",
            )
            dtype = entry["tensor"]["dtype"]
            if begin == end:
                continue
            checksum = entry["sha256"] if skeleton else tensor_hash(entry["value"])
            interval = (begin, end)
            if interval in intervals:
                require(intervals[interval] == checksum, f"{key}: inconsistent replicated weights")
            else:
                require(begin == cursor, f"{key}: overlapping or missing shard range")
                intervals[interval], cursor = checksum, end
        require(cursor == expected[key][0], f"{key}: incomplete shard coverage")
    return config, pieces


def read_payloads(directory, *, companion=False, multimodal=False):
    directory = Path(directory)
    manifest_name = RESUME_MANIFEST if companion else MANIFEST
    manifest = read_json(directory / manifest_name)
    require(
        set(manifest) == {"version", "world_size", "step", "files"}
        and all(type(manifest[k]) is int for k in ("version", "world_size", "step"))
        and manifest["version"] == VERSION
        and manifest["world_size"] > 0
        and manifest["step"] > 0,
        "Invalid strict manifest",
    )
    names = [f"rank-{rank:05d}.pt" for rank in range(manifest["world_size"])]
    require(
        isinstance(manifest["files"], dict) and set(manifest["files"]) == set(names),
        "Missing/unexpected manifest ranks",
    )
    require(
        {p.name for p in directory.iterdir()} == set(names) | {manifest_name},
        "Missing/unexpected rank files or incomplete container",
    )
    for name in names:
        require(
            (directory / name).is_file()
            and not (directory / name).is_symlink()
            and file_hash(directory / name) == manifest["files"][name],
            f"{name}: rank checksum mismatch",
        )
    try:
        payloads = [
            torch.load(directory / name, map_location="cpu", weights_only=True) for name in names
        ]
        config, pieces = validate_payloads(
            payloads, manifest["step"], skeleton=companion, multimodal=multimodal
        )
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        raise ConversionError(f"{directory}: invalid strict state: {error}") from error
    return manifest, payloads, config, pieces


def assemble(config, pieces):
    result = {}
    for key, shape in parameter_shapes(config).items():
        first = pieces[key][0][3]["value"]
        value = torch.empty(shape, dtype=first.dtype, device="cpu")
        for begin, end, _, entry in pieces[key]:
            value[begin:end].copy_(entry["value"])
        result[key] = value
    validate_tensors(result, parameter_shapes(config), "strict canonical text")
    return result


def _write_payloads(directory, payloads, step, *, companion=False):
    directory = Path(directory)
    require_empty_destination(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for rank, payload in enumerate(payloads):
        name = f"rank-{rank:05d}.pt"
        _write_tensor_file(directory / name, payload)
        hashes[name] = file_hash(directory / name)
    manifest = dict(version=VERSION, world_size=len(payloads), step=step, files=hashes)
    name = RESUME_MANIFEST if companion else MANIFEST
    temporary = directory / (name + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(directory / name)


def write_companion(hf_path, payloads, step):
    skeletons = []
    for payload in payloads:
        skeleton = dict(payload, state=dict(payload["state"]))
        skeleton["state"]["model"] = {
            key: dict(tensor=entry["tensor"], sha256=tensor_hash(entry["value"]))
            for key, entry in payload["state"]["model"].items()
        }
        skeletons.append(skeleton)
    _write_payloads(Path(hf_path) / RESUME_DIR, skeletons, step, companion=True)


def restore_from_hf(hf_path, destination, config, canonical, *, multimodal=False):
    require_empty_destination(destination)
    manifest, skeletons, saved_config, _ = read_payloads(
        Path(hf_path) / RESUME_DIR, companion=True, multimodal=multimodal
    )
    same_config(config, saved_config)
    validate_tensors(canonical, parameter_shapes(config), "HF resume weights")
    restored = []
    for rank, skeleton in enumerate(skeletons):
        payload = copy.deepcopy(skeleton)
        for key, entry in payload["state"]["model"].items():
            _, _, begin, end = _descriptor(entry["tensor"], rank)
            value = canonical[key][begin:end].contiguous().clone()
            require(
                str(value.dtype) == entry["tensor"]["dtype"]
                and tensor_hash(value) == entry["sha256"],
                f"{key}: HF weights changed; cannot reuse saved optimizer/data/RNG",
            )
            entry["value"] = value
            del entry["sha256"]
        restored.append(payload)
    validate_payloads(restored, manifest["step"], multimodal=multimodal)
    _write_payloads(destination, restored, manifest["step"])
