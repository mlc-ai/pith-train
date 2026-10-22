"""Opt-in single-process strict training resume, separate from weight initialization.

Reuses checkpoint.py's canonical model/optimizer mapping and PR100's Stateful
data boundary. Production task wiring and distributed resume remain integration
work: this entrypoint explicitly rejects PP/EP/CP/FSDP and multiple processes.
"""

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor

from pithtrain.contexts import distributed
from pithtrain.modules.checkpoint import CheckpointState


class StrictResumeError(ValueError):
    """A checkpoint cannot satisfy the requested training continuation."""


def _type_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _check_schema(actual, expected, path="app"):
    if path == "app.resume.rng.python.2":
        # random.gauss alternates this cache between None and a float. It is
        # runtime state, not a mismatch with a fresh process's empty cache.
        if actual is not None and type(actual) is not float:
            raise StrictResumeError(f"{path}: invalid cached Gaussian")
    elif isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise StrictResumeError(f"{path}: expected tensor")
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise StrictResumeError(
                f"{path}: shape/dtype mismatch: {actual.shape}/{actual.dtype}, "
                f"expected {expected.shape}/{expected.dtype}"
            )
    elif isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise StrictResumeError(f"{path}: expected dictionary")
        missing, unexpected = expected.keys() - actual.keys(), actual.keys() - expected.keys()
        if missing or unexpected:
            raise StrictResumeError(
                f"{path}: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for key in expected:
            _check_schema(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise StrictResumeError(f"{path}: sequence type/count mismatch")
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _check_schema(left, right, f"{path}.{i}")
    elif type(actual) is not type(expected):
        raise StrictResumeError(f"{path}: type mismatch")


def _rng_state(device):
    numpy = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        # Avoid NumPy pickle globals; DCP / weights_only-compatible builtins.
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _numpy_state(state):
    return state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]


def _validate_rng(state, device):
    try:
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(_numpy_state(state["numpy"]))
        if device.type == "cuda":
            torch.Generator(device=device).set_state(state["cuda"])
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise StrictResumeError(f"Invalid RNG state: {error}") from error


def _restore_rng(state, device):
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    np.random.set_state(_numpy_state(state["numpy"]))
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


class StrictResumeState:
    """Model + optimizer + scheduler + committed data + RNG, all required.

    ``step`` counts completed updates. ``data_state`` is the existing loader's
    checkpoint_state, whose version-1 schema includes fingerprint and
    consumed_samples. No loader, sampler or cursor convention is introduced.
    The caller must construct the same optimizer/scheduler classes and layout.
    """

    def __init__(self, model, optimizers, schedulers, *, data_state, step, global_batch_size):
        if type(step) is not int or step < 1:
            raise StrictResumeError("Strict resume requires at least one completed update")
        if type(global_batch_size) is not int or global_batch_size < 1:
            raise StrictResumeError("global_batch_size must be a positive integer")
        if data_state is None or not optimizers or not schedulers:
            raise StrictResumeError("Strict resume requires data, optimizer and scheduler state")
        self.base = CheckpointState(model, tuple(optimizers), tuple(schedulers))
        self.data_state, self.step = data_state, step
        self.global_batch_size = global_batch_size
        self._check_scope()

    def _check_scope(self):
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise NotImplementedError("Distributed strict resume requires PP/EP integration")
        if any(getattr(distributed, axis, 1) != 1 for axis in ("pp_size", "ep_size", "cp_size")):
            raise NotImplementedError("Strict resume currently requires PP=EP=CP=1")
        params = list(self.base.model.parameters())
        if any(isinstance(p, DTensor) for p in params):
            raise NotImplementedError("FSDP strict resume requires distributed integration")
        devices = {p.device for p in params}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise StrictResumeError("Strict resume requires one CPU or CUDA device")
        self.device = next(iter(devices))

    def _snapshot(self):
        self._check_scope()
        # PyTorch DCP lazily materializes empty optimizer state by calling a
        # zero-LR step. Build that template on an isolated model/optimizer graph
        # so even a metadata failure cannot initialize the live optimizer.
        base = self.base
        if any(not optimizer.state for optimizer in base.optimizers):
            base = copy.deepcopy(base)
        result = base.state_dict()
        result["data"] = self.data_state.state_dict()
        result["resume"] = {
            "version": 1,
            "step": self.step,
            "global_batch_size": self.global_batch_size,
            "optimizer_types": [_type_name(opt) for opt in self.base.optimizers],
            "scheduler_types": [_type_name(sch) for sch in self.base.schedulers],
            "rng": _rng_state(self.device),
        }
        return result

    def _check_data_position(self, data):
        if (
            type(data.get("version")) is not int
            or data["version"] != 1
            or not isinstance(data.get("fingerprint"), str)
        ):
            raise StrictResumeError("Expected PR100 version-1 data checkpoint_state")
        consumed = data.get("consumed_samples")
        if type(consumed) is not int or consumed != self.step * self.global_batch_size:
            raise StrictResumeError("Checkpoint step and committed data position disagree")

    def state_dict(self):
        result = self._snapshot()
        self._check_data_position(result["data"])
        return result

    def _validate(self, loaded, expected):
        _check_schema(loaded, expected)
        for field in ("version", "step", "global_batch_size", "optimizer_types", "scheduler_types"):
            if loaded["resume"][field] != expected["resume"][field]:
                raise StrictResumeError(f"resume.{field}: incompatible training continuation")
        self._check_data_position(loaded["data"])
        if loaded["data"]["fingerprint"] != expected["data"]["fingerprint"]:
            raise StrictResumeError("Checkpoint data fingerprint differs from this run")
        # Existing localization reconstructs membership; validate first so it
        # cannot hide dropped/reordered parameters in a corrupt checkpoint.
        for saved, live in zip(
            loaded["optimizer"]["param_groups"], expected["optimizer"]["param_groups"], strict=True
        ):
            if saved["params"] != live["params"]:
                raise StrictResumeError("optimizer param_group membership/order mismatch")
        _validate_rng(loaded["resume"]["rng"], self.device)

    def load_state_dict(self, state_dict):
        expected = self._snapshot()
        self._validate(state_dict, expected)
        # PR100 validates its own data contract before touching the model. If it
        # rejects extra schema constraints, retain the caller's original cursor.
        previous_data = copy.deepcopy(expected["data"])
        try:
            self.data_state.load_state_dict(state_dict["data"])
        except Exception:
            self.data_state.load_state_dict(previous_data)
            raise
        self.base.load_state_dict(state_dict)
        _restore_rng(state_dict["resume"]["rng"], self.device)


def _location(root, state):
    return Path(root, "torch-dcp", f"{state.step:08d}")


def save_strict_checkpoint(root, state: StrictResumeState):
    """Save a committed single-process update; never overwrite an existing step."""
    snapshot = state.state_dict()
    location = _location(root, state)
    if location.exists():
        raise FileExistsError(f"Checkpoint already exists: {location}")
    dcp.save({"app": snapshot}, checkpoint_id=location, no_dist=True)
    return location


def load_strict_checkpoint(root, state: StrictResumeState):
    """Preflight every key/shape/dtype, stage separately, validate, then apply.

    DCP normally ignores unexpected keys and may partially read mismatched
    shapes. Its own flattening routine is used to compare the exact planner
    schema, including nested optimizer/scheduler keys, before payload reads.
    """
    template = state._snapshot()
    location = _location(root, state)
    metadata = dcp.FileSystemReader(location).read_metadata()
    expected, _ = flatten_state_dict({"app": template})
    actual = metadata.state_dict_metadata
    missing, unexpected = expected.keys() - actual.keys(), actual.keys() - expected.keys()
    if missing or unexpected:
        raise StrictResumeError(f"DCP missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in expected.items():
        stored = actual[key]
        if isinstance(value, torch.Tensor):
            if not isinstance(stored, TensorStorageMetadata):
                raise StrictResumeError(f"DCP {key}: expected tensor metadata")
            if stored.size != value.shape or stored.properties.dtype != value.dtype:
                raise StrictResumeError(f"DCP {key}: shape/dtype mismatch")
        elif isinstance(stored, TensorStorageMetadata):
            raise StrictResumeError(f"DCP {key}: unexpected tensor metadata")
    # get_state_dict can alias live parameters and optimizer buffers. Reading
    # into those would mutate the training state before semantic validation.
    staged = {"app": copy.deepcopy(template)}
    dcp.load(staged, checkpoint_id=location, no_dist=True)
    state.load_state_dict(staged["app"])
    return state.step
