"""Bounded full Qwen3-Omni HF -> canonical DCP -> local speech FSDP initialization.

No downloads, optimizer state or model replica is created. DCP keeps the existing
app.model keys and component sidecar. Initialization is not transactional resume:
all ranks validate CPU candidates before copying; discard a model after copy error.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list
from torch.distributed.tensor import DTensor, Shard

from pithtrain.modules.omni_speech_fsdp import (
    _local_speech_tensors,
    _require,
    _world_mesh,
    load_speech_local_shards,
)
from pithtrain.tasks.convert_checkpoint import qwen3_omni_full as full

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
CONFIG_SHA256 = "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"
EXPERT = re.compile(r"(.*\.experts)\.(gate_up_proj|down_proj)$")


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def _files(root):
    result = {}
    for path in sorted(Path(root).iterdir()):
        _require(path.is_file() and not path.is_symlink(), "Only regular checkpoint files allowed")
        result[path.name] = dict(bytes=path.stat().st_size, sha256=_sha(path))
    return result


def _check_files(root, expected, *, sidecar=False):
    actual = {p.name for p in Path(root).iterdir()}
    _require(
        actual == set(expected) | ({full.SIDECAR} if sidecar else set()), "File inventory differs"
    )
    for name, row in expected.items():
        _require(
            Path(name).name == name and "/" not in name and "\\" not in name, "Invalid filename"
        )
        path = Path(root) / name
        _require(path.is_file() and not path.is_symlink(), "Expected regular file: " + name)
        _require(
            path.stat().st_size == row["bytes"] and _sha(path) == row["sha256"],
            "File changed: " + name,
        )


def _canonical_sources(config, layouts):
    """Derive slices using the existing exact native/HF schema, then cover every key."""
    shapes = full.parameter_shapes(config)
    fused = full.parameter_shapes(config, "fused")
    result = {}
    for native, hf in full.native_key_map(config).items():
        match = EXPERT.fullmatch(native)
        if match is None:
            _require(native in shapes, "Unmapped canonical parameter: " + native)
            result[native] = [(hf, None)]
            continue
        root, projection = match.groups()
        hf_root = hf.rsplit(".", 1)[0]
        for index in range(fused[hf][0]):
            key = f"{root}.{index}.{projection}"
            if layouts[native.split(".", 1)[0]] == "fused":
                result[key] = [(hf, index)]
            elif projection == "down_proj":
                result[key] = [(f"{hf_root}.{index}.down_proj.weight", None)]
            else:
                result[key] = [
                    (f"{hf_root}.{index}.{part}.weight", None) for part in ("gate_proj", "up_proj")
                ]
    _require(result.keys() == shapes.keys(), "Incomplete canonical source mapping")
    return result


class _HFSavePlanner(DefaultSavePlanner):
    def __init__(self, root, owners, mapping, shapes, dtypes, max_tensor_bytes):
        super().__init__()
        self.root, self.owners, self.mapping = root, owners, mapping
        self.shapes, self.dtypes, self.max_tensor_bytes = shapes, dtypes, max_tensor_bytes

    def resolve_data(self, write_item):
        key = write_item.index.fqn.removeprefix("app.model.")
        pieces = []
        for name, index in self.mapping[key]:
            with safe_open(self.root / self.owners[name], framework="pt", device="cpu") as handle:
                # Slice fused experts before materializing; never load the expert stack.
                value = (
                    handle.get_slice(name)[index] if index is not None else handle.get_tensor(name)
                )
                pieces.append(value)
        result = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        _require(tuple(result.shape) == self.shapes[key], "Canonical slice shape differs: " + key)
        _require(result.dtype == self.dtypes[key], "Canonical slice dtype differs: " + key)
        _require(
            result.numel() * result.element_size() <= self.max_tensor_bytes, "Tensor memory cap"
        )
        _require(torch.isfinite(result).all().item(), "Nonfinite released tensor: " + key)
        return result


def prepare_speech_initialization(source, destination, *, expected_source, max_tensor_bytes):
    """Convert pinned local full components to standard DCP using bounded tensors.

    expected_source contains kind, model_id, revision, config_sha256 and exact
    files{name:{bytes,sha256}}. kind='released' enforces the official revision,
    config and BF16 inventory; 'test_fixture' is explicitly not released acceptance.
    The caller resolves/downloads provenance separately. Destination must be absent.
    Peak live payload is a few canonical tensors, never the full model. The public
    DCP writer uses one tensor per file because its per-file cache retains tensors.
    """
    source, destination = Path(source), Path(destination)
    _require(type(max_tensor_bytes) is int and max_tensor_bytes > 0, "Positive tensor cap required")
    _require(
        set(expected_source) == {"kind", "model_id", "revision", "config_sha256", "files"},
        "Source identity fields differ",
    )
    _require(
        expected_source["kind"] in ("released", "test_fixture"), "Explicit source kind required"
    )
    _require(not destination.exists(), "Use a new DCP destination")
    _require(
        not destination.resolve().is_relative_to(source.resolve()), "Destination overlaps source"
    )
    _check_files(source, expected_source["files"])
    _require(
        _sha(source / "config.json") == expected_source["config_sha256"], "Source config changed"
    )
    raw = full.text.read_json(source / "config.json")
    config = full.configuration(raw, "all")
    _, owners, hf_dtypes, excluded, layouts = full.read_hf_headers(source, config)
    _require(not excluded, "Full speech cannot omit component keys")
    if expected_source["kind"] == "released":
        _require(
            expected_source["model_id"] == MODEL and expected_source["revision"] == REVISION,
            "Released revision mismatch",
        )
        _require(expected_source["config_sha256"] == CONFIG_SHA256, "Released config mismatch")
        _require(
            len(owners) == 28010 and len(set(owners.values())) == 15, "Released inventory mismatch"
        )
        _require(set(hf_dtypes.values()) == {torch.bfloat16}, "Released weights must be BF16")
    mapping = _canonical_sources(config, layouts)
    shapes = full.parameter_shapes(config)
    dtypes = {}
    for key, parts in mapping.items():
        types = {hf_dtypes[name] for name, _ in parts}
        _require(len(types) == 1, "Mixed dtype within canonical tensor")
        dtypes[key] = next(iter(types))
        _require(
            math.prod(shapes[key]) * torch.empty((), dtype=dtypes[key]).element_size()
            <= max_tensor_bytes,
            "Tensor memory cap: " + key,
        )
    # Meta is only the DCP plan; payload is resolved lazily by the save planner.
    state = {
        key: torch.empty(shape, dtype=dtypes[key], device="meta") for key, shape in shapes.items()
    }
    planner = _HFSavePlanner(source, owners, mapping, shapes, dtypes, max_tensor_bytes)
    writer = FileSystemWriter(
        destination,
        single_file_per_rank=False,
        thread_count=1,
        per_thread_copy_ahead=0,
        overwrite=False,
    )
    dcp.save({"app": {"model": state}}, storage_writer=writer, planner=planner, no_dist=True)
    _check_files(source, expected_source["files"])
    manifest = full._manifest(config, "component", layouts, [])
    manifest["bounded_initialization"] = dict(
        schema=1,
        source=expected_source,
        max_tensor_bytes=max_tensor_bytes,
        checkpoint_files=_files(destination),
        canonical_dtypes={k: str(v) for k, v in dtypes.items()},
        nonpersistent_constants="Native modules recompute rotary, audio positions and codec offsets; no checkpoint tensors are invented",
    )
    (destination / full.SIDECAR).write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(
        manifest_sha256=_sha(destination / full.SIDECAR),
        canonical_tensors=len(shapes),
        source_kind=expected_source["kind"],
        full_model_replica=False,
    )


class _SliceLoadPlanner(DefaultLoadPlanner):
    def __init__(self, offsets):
        super().__init__(flatten_state_dict=False, flatten_sharded_tensors=False)
        self.offsets = offsets

    def create_local_plan(self):
        items = []
        for key, tensor in self.state_dict.items():
            chunk = ChunkStorageMetadata(torch.Size(self.offsets[key]), tensor.size())
            items.extend(
                create_read_items_for_chunk_list(
                    key, self.metadata.state_dict_metadata[key], [chunk]
                )
            )
        return LoadPlan(items)

    def resolve_tensor(self, read_item):
        value = self.state_dict[read_item.dest_index.fqn]
        for dim, (offset, length) in enumerate(
            zip(read_item.dest_offsets, read_item.lengths, strict=True)
        ):
            value = value.narrow(dim, offset, length)
        return value


def _agree(plan, error, label):
    errors = [None] * dist.get_world_size(plan["group"])
    dist.all_gather_object(errors, error, group=plan["group"])
    _require(not any(errors), label + ": " + repr(errors))


def initialize_speech_from_dcp(
    model,
    directory,
    *,
    expected_manifest_sha256,
    rank,
    world_size,
    max_local_bytes,
    max_tensor_bytes,
):
    """Initialize only materialized local FP32 shards, without an all-gather.

    Complete metadata/layout/file and all-rank CPU finite validation precedes
    the first model copy. CPU staging is at most this rank's local storage plus
    one canonical DCP tensor, plus finite-check boolean workspace (the reader
    deserializes its saved tensor). Unknown
    buffers/constants are rejected. All current native constants are recomputed
    from config, so only released parameter/persistent-buffer keys are loaded.
    No optimizer may be created before this initialization-only operation.
    """
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    directory = Path(directory)
    plan = getattr(model, "_omni_speech_fsdp_plan", None)
    _require(isinstance(plan, dict), "Apply speech FSDP first")
    error = None
    try:
        _world_mesh(plan["mesh"])
        _require(
            plan.get("materialized") is True and plan.get("weights_initialized") is False,
            "Initialize newly materialized shards only",
        )
        _require(
            type(rank) is int
            and type(world_size) is int
            and rank == dist.get_rank()
            and world_size == dist.get_world_size(),
            "Rank/world identity mismatch",
        )
        _require(
            all(type(v) is int and v > 0 for v in (max_local_bytes, max_tensor_bytes)),
            "Positive memory caps required",
        )
        _require(
            _sha(directory / full.SIDECAR) == expected_manifest_sha256, "Pinned manifest changed"
        )
        manifest = full._read_manifest(directory / full.SIDECAR, "all")
        bounded = manifest["bounded_initialization"]
        _require(bounded["schema"] == 1, "Bounded initialization contract mismatch")
        config = full.configuration(manifest["source_config"], "all")
        _require(
            model.config.to_dict() == Qwen3OmniMoeConfig(**config.raw).to_dict(),
            "Model configuration differs",
        )
        metadata = FileSystemReader(directory).read_metadata()
        shapes = full.parameter_shapes(config)
        expected = {"app.model." + k: v for k, v in shapes.items()}
        _require(
            metadata.state_dict_metadata.keys() == expected.keys(),
            "Canonical DCP key inventory differs",
        )
        for key, shape in expected.items():
            entry = metadata.state_dict_metadata[key]
            _require(tuple(entry.size) == shape, "Canonical DCP shape differs: " + key)
            _require(
                str(entry.properties.dtype)
                == bounded["canonical_dtypes"][key.removeprefix("app.model.")],
                "Canonical DCP dtype differs",
            )
            _require(
                math.prod(shape) * torch.empty((), dtype=entry.properties.dtype).element_size()
                <= max_tensor_bytes,
                "DCP tensor read memory cap",
            )
        values = _local_speech_tensors(model)
        mapping = full.native_key_map(config)
        _require(
            values.keys() == mapping.keys(),
            "Native parameters/buffers differ; unknown constants need an explicit contract",
        )
        fused = full.parameter_shapes(config, "fused")
        total = 0
        for name, (kind, value, local) in values.items():
            _require(tuple(value.shape) == fused[mapping[name]], "Native global shape differs")
            _require(
                not local.is_meta and local.dtype == torch.float32,
                "Materialized FP32 master required",
            )
            if kind == "parameter":
                _require(
                    isinstance(value, DTensor)
                    and value.placements == (Shard(0),)
                    and value.device_mesh == plan["mesh"],
                    "Exact DP shard layout required",
                )
                width = math.ceil(value.shape[0] / world_size)
                local_shape = (max(0, min(width, value.shape[0] - rank * width)), *value.shape[1:])
                _require(tuple(local.shape) == local_shape, "Native local shard shape differs")
            else:
                _require(not isinstance(value, DTensor), "Persistent buffers must be replicated")
            total += local.numel() * local.element_size()
        _require(total <= max_local_bytes, "Rank-local staging memory cap")
    except Exception as exc:
        error = str(exc)
    _agree(plan, error, "Speech initialization preflight failed")
    identities = [None] * world_size
    dist.all_gather_object(
        identities,
        (expected_manifest_sha256, max_local_bytes, max_tensor_bytes),
        group=plan["group"],
    )
    _require(len(set(identities)) == 1, "All ranks must use the same source and memory contract")
    # Each rank may see a different local mount, even with an identical sidecar.
    # Validate that rank's actual payload bytes before staging and before copying.
    error = None
    try:
        _check_files(directory, bounded["checkpoint_files"], sidecar=True)
    except Exception as exc:
        error = str(exc)
    _agree(plan, error, "DCP file preflight failed")
    error = None
    shards = {}
    try:
        targets, offsets = {}, {}
        for name, (_, value, local) in values.items():
            staged = torch.empty(local.shape, dtype=local.dtype, device="cpu")
            shards[name] = staged
            start = (
                min(math.ceil(value.shape[0] / world_size) * rank, value.shape[0])
                if isinstance(value, DTensor)
                else 0
            )
            match = EXPERT.fullmatch(name)
            if match:
                root, projection = match.groups()
                for i in range(local.shape[0]):
                    key = f"app.model.{root}.{start + i}.{projection}"
                    targets[key], offsets[key] = staged[i], [0] * (value.ndim - 1)
            else:
                key = "app.model." + name
                targets[key], offsets[key] = staged, [start] + [0] * (value.ndim - 1)
        # No cross-rank collectives in DCP; each plan targets only this rank's slices.
        dcp.load(targets, checkpoint_id=directory, planner=_SliceLoadPlanner(offsets), no_dist=True)
        _require(
            all(torch.isfinite(v).all().item() for v in shards.values()),
            "Nonfinite local candidate",
        )
    except BaseException as exc:
        error = str(exc)
    _agree(plan, error, "DCP local staging failed")
    error = None
    try:
        _require(
            _sha(directory / full.SIDECAR) == expected_manifest_sha256,
            "Manifest changed while reading",
        )
        _check_files(directory, bounded["checkpoint_files"], sidecar=True)
    except Exception as exc:
        error = str(exc)
    _agree(plan, error, "DCP changed before application")
    load_speech_local_shards(model, shards, rank=rank, world_size=world_size)
    receipt = dict(
        schema=1,
        kind="full_speech_model_initialization_only",
        rank=rank,
        world_size=world_size,
        mesh=list(plan["mesh"].mesh.tolist()),
        placement="Shard(0)",
        manifest_sha256=expected_manifest_sha256,
        source=bounded["source"],
        local_staging_bytes=total,
        parameters=sum(k == "parameter" for k, _, _ in values.values()),
        buffers=sum(k == "buffer" for k, _, _ in values.values()),
        weights_initialized=True,
        full_model_replica=False,
    )
    plan["initialization"] = receipt
    return receipt
