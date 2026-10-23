"""Read complete native speech model weights from a strict DP checkpoint.

This is model-only initialization for a separate unsharded generation process,
not training resume or in-flight generation restore. No live model is mutated.
The caller may load the returned FP32 state into its fresh/meta native model.
"""

import json
from pathlib import Path

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Shard

from pithtrain.modules.checkpoint import find_moe
from pithtrain.modules.distributed_checkpoint import MANIFEST
from pithtrain.modules.omni_speech_initialization import _require, _sha
from pithtrain.modules.resume_semantics import model_contract
from pithtrain.modules.task_checkpoint import TASK_STATE_VERSION
from pithtrain.tasks.convert_checkpoint import qwen3_omni_full as full


def read_speech_generation_state(
    model, location, *, expected_manifest_sha256, max_state_bytes, max_rank_file_bytes
):
    """Validate every rank, then assemble FP32 native weights without collectives.

    Supports the complete speech DP-only, Shard(0), task-v2 format produced by
    omni_speech's native checkpoint path. PP/CP/EP, replicated/HSDP layouts and
    alternate model semantics are rejected. Model may be meta; parameters,
    buffers, gradients, RNG and optimizer state are never changed by this read.
    The manifest SHA must come from the caller's completed checkpoint receipt.

    Payload storage is mmap-backed, but every model tensor is checked and copied
    into owned CPU memory before return. The explicit state cap covers the full
    returned model, not caller-owned models, OS page cache or later device moves.
    File hashes are checked around each pass; incomplete/changed input cannot
    yield a partially loaded model. Optimizer/data/RNG contents are not restored.
    """
    for cap in (max_state_bytes, max_rank_file_bytes):
        _require(type(cap) is int and cap > 0, "Positive generation memory/file caps required")
    _require(
        not any(isinstance(m, FSDPModule) for m in model.modules())
        and not any(isinstance(p, DTensor) for p in model.parameters()),
        "Generation target must be an unsharded complete native model",
    )
    cfg = full.configuration(model.config.to_dict(), "all")
    mapping = full.native_key_map(cfg)
    shapes = full.parameter_shapes(cfg, "fused")
    target = model.state_dict()
    _require(target.keys() == mapping.keys(), "Generation native parameter inventory differs")
    _require(
        all(
            tuple(v.shape) == shapes[mapping[k]] and v.dtype == torch.float32
            for k, v in target.items()
        ),
        "Generation target requires exact complete FP32 master shapes",
    )
    nbytes = sum(v.numel() * v.element_size() for v in target.values())
    _require(nbytes <= max_state_bytes, "Generation state byte cap exceeded")
    contract = model_contract(model)
    modules = dict(model.named_modules())
    location = Path(location)
    manifest_path = location / MANIFEST
    _require(
        manifest_path.is_file()
        and not manifest_path.is_symlink()
        and _sha(manifest_path) == expected_manifest_sha256,
        "Pinned generation manifest differs",
    )
    manifest = json.loads(manifest_path.read_text())
    _require(set(manifest) == {"version", "world_size", "step", "files"}, "Invalid strict manifest")
    world, step = manifest["world_size"], manifest["step"]
    _require(
        manifest["version"] == TASK_STATE_VERSION
        and type(world) is int
        and world > 0
        and type(step) is int
        and step > 0,
        "Generation requires a completed task-v2 checkpoint",
    )
    names = {f"rank-{r:05d}.pt" for r in range(world)}
    _require(set(manifest["files"]) == names, "Generation rank inventory differs")

    def verify_files():
        _require(_sha(manifest_path) == expected_manifest_sha256, "Generation manifest changed")
        _require(
            {p.name for p in location.iterdir()} == names | {MANIFEST},
            "Unexpected checkpoint files",
        )
        for name in sorted(names):
            path = location / name
            _require(path.is_file() and not path.is_symlink(), "Invalid checkpoint rank file")
            _require(
                path.stat().st_size <= max_rank_file_bytes, "Generation rank-file byte cap exceeded"
            )
            _require(_sha(path) == manifest["files"][name], "Generation rank checksum differs")

    saved_config = None
    saved_updates = None

    def visit(rank, destination=None):
        nonlocal saved_config, saved_updates
        payload = torch.load(
            location / f"rank-{rank:05d}.pt", weights_only=True, mmap=True, map_location="cpu"
        )
        header = payload["header"]
        topology = header["topology"]
        _require(
            header["version"] == TASK_STATE_VERSION
            and header["step"] == step
            and topology["rank"] == topology["dp_rank"] == rank
            and topology["world_size"] == topology["dp_size"] == world
            and all(
                topology[axis + "_size"] == 1 and topology[axis + "_rank"] == 0
                for axis in ("cp", "ep", "pp")
            ),
            "Generation supports complete DP-only checkpoint topology",
        )
        for key in ("attn_mesh", "expt_mesh"):
            mesh = topology[key]
            _require(
                mesh["ranks"] == list(range(world)) and list(mesh["names"]) == ["dp"],
                "Generation shard mesh differs",
            )
        stored = header["config"]
        speech = stored.get("speech_task", {})
        _require(
            speech.get("schema") == 2 and speech.get("model_contract") == contract,
            "Generation model numerical contract differs",
        )
        _require(stored.get("hsdp_replica", 1) == 1, "Generation HSDP checkpoint unsupported")
        _require(payload["task"]["task_kind"] == "omni_speech", "Expected native speech task")
        updates = payload["task"]["cursors"]["updates"]
        if saved_config is None:
            saved_config, saved_updates = stored, updates
        _require(
            stored == saved_config and updates == saved_updates,
            "Checkpoint ranks disagree on model/task contract",
        )
        layout = header["layout"]
        _require(layout.keys() == target.keys(), "Native checkpoint layout inventory differs")
        packed = payload["state"]["model"]
        expected_keys = set()
        for name, tensor in target.items():
            desc = layout[name]
            _require(
                tuple(desc["shape"]) == tuple(tensor.shape)
                and desc["dtype"] == "torch.float32"
                and desc.get("placements") == ["S(0)"]
                and desc.get("mesh") == list(range(world))
                and list(desc.get("names", ())) == ["dp"]
                and tuple(desc.get("stride", ())) == tuple(tensor.stride()),
                "Generation parameter shard layout differs: " + name,
            )
            count, offset = Shard.local_shard_size_and_offset(tensor.shape[0], world, rank)
            expert = find_moe(name, modules) is not None
            parts = (
                [
                    (name.replace(".experts.", f".experts.{i}.", 1), (i,), tuple(tensor.shape[1:]))
                    for i in range(offset, offset + count)
                ]
                if expert
                else [(name, (slice(offset, offset + count),), (count, *tensor.shape[1:]))]
            )
            for key, index, shape in parts:
                expected_keys.add(key)
                item = packed[key]
                _require(set(item) == {"tensor", "value"}, "Invalid packed generation tensor")
                value = item["value"]
                expected_desc = (
                    {
                        "shape": tuple(tensor.shape[1:]),
                        "dtype": "torch.float32",
                        "device": desc["device"],
                    }
                    if expert
                    else desc
                )
                _require(
                    item["tensor"] == expected_desc, "Canonical generation layout differs: " + key
                )
                _require(
                    type(value) is torch.Tensor
                    and value.device.type == "cpu"
                    and value.dtype == torch.float32
                    and tuple(value.shape) == shape
                    and torch.isfinite(value).all().item(),
                    "Invalid generation tensor: " + key,
                )
                if destination is not None:
                    destination[name][index].copy_(value)
        _require(packed.keys() == expected_keys, "Missing or unexpected canonical generation keys")

    verify_files()
    for rank in range(world):
        visit(rank)
    verify_files()
    result = {name: torch.empty_like(value, device="cpu") for name, value in target.items()}
    for rank in range(world):
        visit(rank, result)
    verify_files()
    return result, dict(
        schema=1,
        kind="native_speech_checkpoint_model_only",
        step=step,
        updates=saved_updates,
        checkpoint_world_size=world,
        model_contract_sha256=contract["sha256"],
        manifest_sha256=expected_manifest_sha256,
        fp32_state_bytes=nbytes,
        restored_training_state=False,
        restored_generation_cache=False,
    )
