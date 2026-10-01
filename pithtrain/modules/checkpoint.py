"""
Checkpoint save and load, with resharding between the disk and runtime formats.

Disk format (canonical):
  - No module.{N}. prefix
  - Experts individually indexed: layers.1.mlp.experts.3.gate_proj.weight

Runtime format (localized):
  - DualPipeV prefix: module.0.layers.1.mlp.experts.gate_proj.weight
  - Experts stacked per EP rank: shape [experts_per_rank, ...]

A checkpoint is one DCP directory holding the model, optimizer and scheduler state plus a
per-rank CUDA RNG file, kept at root/torch-dcp/XXXXXXXX. A checkpoint is identified by its
step, so all three entry points take a root and a step: save_checkpoint writes one, load_checkpoint
reads one, and find_checkpoint reports the newest step under a root, or None when there is none.
The layout itself never leaves this module.

The step counts completed units of work, which is also the number a resuming run continues from:
with five steps done, the next one is five. Callers therefore need no offset arithmetic, and none
of these functions touches run position. Only the caller knows whether loading means resuming its
own run or filling a frozen slot such as an RL reference policy.
"""

import gc
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pithtrain.contexts import distributed, logging, training

__all__ = [
    "find_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
    "to_canonical_model",
    "to_canonical_optim",
    "to_localized_model",
    "to_localized_optim",
]

MODULE_PREFIX_RE = re.compile(r"^module\.\d+\.")
INDEXED_EXPERT_RE = re.compile(r"(.*)\.experts\.(\d+)\.(.*)")


def strip_prefix(key: str) -> str:
    """
    Strip module.{N}. prefix from a DualPipeV FQN.
    """
    return MODULE_PREFIX_RE.sub("", key)


def find_moe(key: str, named_modules: Dict[str, nn.Module]) -> Optional[nn.Module]:
    """
    Return the MoE module for a stacked expert key, or None.

    A stacked key looks like module.0.layers.1.mlp.experts.gate_proj.weight, with no numeric index
    after .experts. Already-indexed keys like layers.1.mlp.experts.3.gate_proj.weight return None.
    """
    if ".experts." not in key:
        return None
    moe_path, _, after = key.partition(".experts.")
    if after and after.split(".")[0].isdigit():
        return None
    mod = named_modules.get(moe_path)
    if mod and hasattr(mod, "experts_per_rank"):
        return mod
    return None


def expert_range(mod: nn.Module) -> Tuple[int, int]:
    """
    Global expert index range [start, end) for this EP rank.
    """
    start = distributed.ep_rank * mod.experts_per_rank
    return start, start + mod.experts_per_rank


def unwrap_dtensor_experts(value: Any, expected_n: int) -> Optional[Tuple[Any, int, int]]:
    """
    Extract local expert data from a DTensor without triggering all_gather.

    When FSDP shards the stacked expert tensor along dim 0 via Shard(0), each DP rank holds a
    contiguous subset of experts. This helper extracts that local subset and computes the global
    expert offset so that unpack can emit per-expert keys for only the experts this rank actually
    owns, with zero GPU communication.

    Works for both model tensors (a single DTensor) and optimizer state entries (a dict whose
    tensor values are DTensors). Returns (localized_value, local_expert_count, dp_expert_offset),
    or None if value is not a sharded DTensor.
    """

    def _info(dt: Any) -> Optional[Tuple[torch.Tensor, int, int]]:
        """
        Return (local_tensor, local_n, dp_offset) for a Shard(0) DTensor, or None.
        """
        if not isinstance(dt, DTensor) or dt.dim() == 0 or dt.shape[0] != expected_n:
            return None
        if not dt.placements:
            return None
        if not isinstance(dt.placements[0], Shard):
            return None
        if dt.placements[0].dim != 0:
            return None
        local = dt._local_tensor
        if local.shape[0] >= expected_n:
            return None
        dp_rank = dt.device_mesh.get_local_rank()
        dp_size = dt.device_mesh.size()
        chunk, remainder = divmod(expected_n, dp_size)
        dp_offset = dp_rank * chunk + min(dp_rank, remainder)
        return local, local.shape[0], dp_offset

    if isinstance(value, DTensor):
        return _info(value)

    if isinstance(value, dict):
        ref = None
        for v in value.values():
            ref = _info(v) if isinstance(v, DTensor) else None
            if ref is not None:
                break
        if ref is None:
            return None
        _, local_n, dp_offset = ref
        localized = {k: v._local_tensor if isinstance(v, DTensor) else v for k, v in value.items()}
        return localized, local_n, dp_offset

    return None


def unpack(
    entries: Dict[str, Any],
    named_modules: Dict[str, nn.Module],
    unstack: Callable[[Any, int, int], Any],
) -> Dict[str, Any]:
    """
    Strip module prefix and unpack stacked experts into individual entries.

    unstack(value, num_local_experts, local_idx) extracts one expert slice from a stacked value.
    For model tensors this is v[i]; for optimizer state dicts it slices each sub-tensor.

    When values are FSDP-sharded DTensors the expert dimension is extracted locally by
    unwrap_dtensor_experts, so each DP rank emits keys only for the experts it owns and no GPU
    all_gather is triggered.
    """
    result: Dict[str, Any] = {}
    for key, value in entries.items():
        canon = strip_prefix(key)
        moe = find_moe(key, named_modules)
        if moe is None:
            result[canon] = value
            continue
        start, _ = expert_range(moe)
        n = moe.experts_per_rank

        local_info = unwrap_dtensor_experts(value, n)
        if local_info is not None:
            local_value, local_n, dp_offset = local_info
            for i in range(local_n):
                global_idx = start + dp_offset + i
                ckey = canon.replace(".experts.", ".experts.%d." % global_idx, 1)
                result[ckey] = unstack(local_value, local_n, i)
        else:
            for i in range(n):
                ckey = canon.replace(".experts.", ".experts.%d." % (start + i), 1)
                result[ckey] = unstack(value, n, i)
    return result


def repack(
    entries: Dict[str, Any],
    fqn_map: Dict[str, str],
    named_modules: Dict[str, nn.Module],
    restack: Callable[[Dict[int, Any]], Any],
) -> Dict[str, Any]:
    """
    Remap canonical FQNs to localized FQNs and repack individual experts into stacked format.

    fqn_map maps {canonical_fqn: localized_fqn}.
    restack(by_global_idx) stacks individual expert values back into one.
    """
    result: Dict[str, Any] = {}
    to_stack: Dict[str, Dict[int, Any]] = {}

    for canon_fqn, value in entries.items():
        m = INDEXED_EXPERT_RE.match(canon_fqn)
        if m:
            prefix, idx_str, suffix = m.groups()
            stacked_canon = "%s.experts.%s" % (prefix, suffix)
            localized = fqn_map.get(stacked_canon)
            if localized is not None:
                moe_path = localized.partition(".experts.")[0]
                moe = named_modules.get(moe_path)
                if moe and hasattr(moe, "experts_per_rank"):
                    s, e = expert_range(moe)
                    idx = int(idx_str)
                    if s <= idx < e:
                        to_stack.setdefault(localized, {})[idx] = value
        else:
            localized = fqn_map.get(canon_fqn)
            if localized is not None:
                result[localized] = value

    for localized, by_idx in to_stack.items():
        result[localized] = restack(by_idx)
    return result


def unstack_optim(entry: Dict[str, Any], n: int, i: int) -> Dict[str, Any]:
    """
    Extract one expert slice from a stacked optimizer state entry.
    """
    return {
        k: v[i] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == n else v
        for k, v in entry.items()
    }


def restack_tensors(by_idx: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Stack individual expert tensors back into one.
    """
    return torch.stack([v for _, v in sorted(by_idx.items())])


def restack_optim(by_idx: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stack individual expert optimizer state entries back into one.
    """
    items = sorted(by_idx.items())
    sample = items[0][1]
    return {
        k: torch.stack([by_idx[i][k] for i, _ in items])
        if isinstance(sample[k], torch.Tensor) and sample[k].dim() > 0
        else sample[k]
        for k in sample
    }


def to_canonical_model(
    state_dict: Dict[str, torch.Tensor], model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Canonicalize model state: strip module prefix, unstack experts.
    """
    return unpack(state_dict, dict(model.named_modules()), lambda v, n, i: v[i])


def _expand_localized_fqn(localized_fqn: str, named_modules: Dict[str, nn.Module]) -> list:
    """
    Map a localized (runtime) FQN to its canonical (disk) FQN(s), expanding stacked experts.
    """
    canon = strip_prefix(localized_fqn)
    moe = find_moe(localized_fqn, named_modules)
    if moe is None:
        return [canon]
    start, end = expert_range(moe)
    return [canon.replace(".experts.", ".experts.%d." % idx, 1) for idx in range(start, end)]


def _localized_param_group_fqns(model: nn.Module, optimizers) -> list:
    """
    This rank's localized FQNs per param group, across the optimizers in order (the order DCP
    combines them in). Needed on load: DCP dedups param_groups metadata across PP ranks, so the
    loaded params lists aren't reliably this rank's, and we rebuild membership from the live
    optimizers instead.
    """
    param_to_fqn = {p: n for n, p in model.named_parameters()}
    groups = []
    for opt in optimizers:
        for g in opt.param_groups:
            groups.append([param_to_fqn[p] for p in g["params"]])
    return groups


def to_canonical_optim(optim_state: Dict, model: nn.Module) -> Dict:
    """
    Canonicalize optimizer state: strip module prefix, unstack expert states.

    Each param group keeps its own membership, meaning its loaded FQNs mapped to canonical with
    experts expanded, rather than collapsing to the full param set. That way a composed
    multi-optimizer state dict such as Muon plus AdamW, combined by DCP, round-trips without one
    group's hyperparameters leaking onto another's. Plain Adam is the degenerate case: its one
    group already owns every param.
    """
    named_modules = dict(model.named_modules())
    state = unpack(optim_state["state"], named_modules, unstack_optim)
    param_groups = []
    for g in optim_state["param_groups"]:
        group = {k: v for k, v in g.items() if k != "params"}
        group["params"] = [
            cf for lf in g["params"] for cf in _expand_localized_fqn(lf, named_modules)
        ]
        param_groups.append(group)
    return {"state": state, "param_groups": param_groups}


def rewrap_dtensor_experts(result: Dict[str, Any], model: nn.Module) -> None:
    """
    Re-wrap restacked plain tensors as DTensors to match model parameters.

    When unpack extracts local FSDP shards via unwrap_dtensor_experts, the individual expert tensors
    are plain rather than DTensors. After repack stacks them the result is a plain tensor whose
    shape equals the FSDP local shard, so [16, ...] when the full stacked parameter is [32, ...].

    model.load_state_dict compares against the DTensor global shape, so these plain tensors must be
    wrapped back into DTensors with the same mesh and placements as the model parameter. For
    optimizer state entries, which are dicts of tensors, every tensor with dim() > 0 is wrapped.
    """
    param_dtensors = {n: p for n, p in model.named_parameters() if isinstance(p, DTensor)}
    for name, param in param_dtensors.items():
        value = result.get(name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor) and not isinstance(value, DTensor):
            result[name] = DTensor.from_local(
                value,
                device_mesh=param.device_mesh,
                placements=param.placements,
                run_check=False,
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, torch.Tensor) and not isinstance(v, DTensor) and v.dim() > 0:
                    value[k] = DTensor.from_local(
                        v,
                        device_mesh=param.device_mesh,
                        placements=param.placements,
                        run_check=False,
                    )


def to_localized_model(
    canonical: Dict[str, torch.Tensor], model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Localize model state: remap FQNs to this rank's localized keys, restack experts.
    """
    named_modules = dict(model.named_modules())
    model_keys = set(model.state_dict().keys())
    fqn_map = {strip_prefix(k): k for k in model_keys}
    result = repack(canonical, fqn_map, named_modules, restack_tensors)
    rewrap_dtensor_experts(result, model)
    return result


def to_localized_optim(optim_state: Dict, model: nn.Module, optimizers) -> Dict:
    """
    Localize optimizer state: remap FQNs, restack experts, rebuild param_groups.

    Membership is rebuilt from the live optimizers, because DCP dedups param_groups across PP ranks
    and the loaded lists aren't reliably this rank's. Each loaded group is matched by position to
    the live group, in the order DCP combined them; its hyperparameters are kept, and its
    membership is replaced with that group's localized FQNs.
    """
    named_modules = dict(model.named_modules())
    fqn_map = {strip_prefix(n): n for n, _ in model.named_parameters()}
    state = repack(optim_state["state"], fqn_map, named_modules, restack_optim)
    rewrap_dtensor_experts(state, model)
    localized_groups = _localized_param_group_fqns(model, optimizers)
    loaded_groups = optim_state["param_groups"]
    assert len(loaded_groups) == len(localized_groups), (
        f"param_group count mismatch: checkpoint has {len(loaded_groups)}, "
        f"live optimizers have {len(localized_groups)}"
    )
    param_groups = []
    for g, members in zip(loaded_groups, localized_groups):
        group = {k: v for k, v in g.items() if k != "params"}
        group["params"] = members
        param_groups.append(group)
    return {"state": state, "param_groups": param_groups}


def find_checkpoint(root: Optional[Path]) -> Optional[int]:
    """
    The step of the newest checkpoint under root, or None when there is none to load.

    A checkpoint is identified by its step, so the step is what this returns and what save and load
    take. The root/torch-dcp/XXXXXXXX layout stays inside this module. Nothing is read into the
    runtime state here: the caller loads, and decides what the step means for its own loop.

    The step counts completed units of work rather than naming the last one, so a resuming run
    continues at exactly this number. With five steps done, the next one to run is five. Storing the
    count rather than the index is what keeps every caller free of offset arithmetic.
    """
    if root is None: return None  # fmt: skip
    latest = max(Path(root, "torch-dcp").glob("[0-9]" * 8), default=None)
    return int(latest.name) if latest is not None else None


class CheckpointState(Stateful):
    """
    Stateful object to save and load the checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizers: tuple[Optimizer, ...],
        schedulers: tuple[LRScheduler, ...],
        model_only: bool = False,
        data_state: Stateful | None = None,
    ):
        self.model, self.optimizers, self.schedulers = model, optimizers, schedulers
        self.model_only = model_only
        self.data_state = data_state

    def state_dict(self):
        """
        Serialize the model, optimizer, and scheduler to a state dictionary.

        Both model and optimizer states are converted to canonical, PP-independent format: the
        module.{N}. prefix is stripped so FQNs use global layer IDs such as layers.0.weight, and
        stacked expert weights are expanded to individual expert tensors with global IDs.

        When model_only is set, as when loading a checkpoint converted from HuggingFace that has no
        optimizer or scheduler, only model keys are advertised so DCP's planner does not look for
        missing optimizer keys.
        """
        if self.model_only:
            model_state, _ = get_state_dict(self.model, self.optimizers)
            return {"model": to_canonical_model(model_state, self.model)}
        model_state, optim_state = get_state_dict(self.model, self.optimizers)
        model_state = to_canonical_model(model_state, self.model)
        optim_state = to_canonical_optim(optim_state, self.model)
        sched_state = [s.state_dict() for s in self.schedulers]
        result = {"model": model_state, "optimizer": optim_state, "scheduler": sched_state}
        if self.data_state is not None:
            result["data"] = self.data_state.state_dict()
        return result

    def load_state_dict(self, state_dict):
        """
        Restore the model, optimizer, and scheduler from the checkpoint.

        Canonical, PP-independent FQNs are mapped back to localized FQNs using the current model
        structure. The optimizer param_groups are rebuilt from the current model so that DCP's
        cross-rank deduplication of non-tensor metadata does not cause FQN mismatches.

        Released checkpoints from HuggingFace do not necessarily include optimizer and scheduler
        state, so those are skipped when missing.
        """
        if self.data_state is not None:
            self.data_state.load_state_dict(state_dict["data"])
        model_state = to_localized_model(state_dict["model"], self.model)
        optim_state = state_dict.get("optimizer")
        sched_state = state_dict.get("scheduler")

        if optim_state:
            optim_state = to_localized_optim(optim_state, self.model, self.optimizers)
            kwargs = dict(model_state_dict=model_state, optim_state_dict=optim_state)
            set_state_dict(self.model, self.optimizers, **kwargs)
        else:
            options = StateDictOptions(strict=False)
            set_model_state_dict(self.model, model_state, options=options)
        if sched_state:
            for scheduler, st in zip(self.schedulers, sched_state):
                scheduler.load_state_dict(st)


def save_checkpoint(root: Path, step: int, *, data_state: Stateful | None = None) -> None:
    """
    Save the runtime model, optimizers and schedulers as the step checkpoint under root. The step is
    a count of completed work, the same number find_checkpoint reports and load_checkpoint takes.

    Uses cpu_offload=True, with the default full_state_dict=False, so that each rank's local FSDP
    shards move to CPU without any GPU all-gather. Expert DTensors are split into per-expert
    entries locally by unwrap_dtensor_experts, so each rank writes only the
    expert keys it owns. Non-expert DTensors stay as CPU DTensors and DCP saves each rank's shard.
    """
    stdout = logging.stdout
    model, optimizers, schedulers = training.model, training.optimizers, training.schedulers
    location = Path(root, "torch-dcp", "%08d" % step)

    options = StateDictOptions(cpu_offload=True)
    model_state, optim_state = get_state_dict(model, optimizers, options=options)
    state_dict = dict()
    state_dict["app"] = dict()
    state_dict["app"]["model"] = to_canonical_model(model_state, model)
    state_dict["app"]["optimizer"] = to_canonical_optim(optim_state, model)
    state_dict["app"]["scheduler"] = [s.state_dict() for s in schedulers]
    if data_state is not None:
        state_dict["app"]["data"] = data_state.state_dict()

    stdout.info("Save checkpoint: %s" % location)
    t0 = time.monotonic()
    gc.collect()
    torch.cuda.empty_cache()
    dcp.save(state_dict, checkpoint_id=location)
    rank = torch.distributed.get_rank()
    rng_path = Path(location, "rng-rank-%05d.pt" % rank)
    torch.save(torch.cuda.get_rng_state(), rng_path)

    dt = torch.tensor(time.monotonic() - t0, device="cuda")
    dt_min, dt_max = dt.clone(), dt.clone()
    torch.distributed.all_reduce(dt_min, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(dt_max, op=torch.distributed.ReduceOp.MAX)
    stdout.info("Save checkpoint: elapsed min=%.1fs, max=%.1fs" % (dt_min.item(), dt_max.item()))


def load_checkpoint(root: Path, step: int, *, data_state: Stateful | None = None) -> None:
    """
    Load the step checkpoint under root into the runtime state: the model, plus the optimizers and
    schedulers when the checkpoint carries them. One converted from HuggingFace carries model keys
    only, which the DCP metadata reveals before anything is read. Requires only that the model,
    optimizers and schedulers already exist in the training context, whichever setup built them.

    Run position is never written. The step names a checkpoint, not where the caller is, and only
    the caller knows whether loading these weights means resuming its own run or filling a frozen
    slot such as an RL reference policy.
    """
    stdout = logging.stdout
    location = Path(root, "torch-dcp", "%08d" % step)
    stdout.info("Load checkpoint: %s" % location)

    t0 = time.monotonic()
    torch.cuda.empty_cache()
    metadata = FileSystemReader(str(location)).read_metadata()
    model_only = all(k.startswith("app.model.") for k in metadata.state_dict_metadata)
    model, optimizers, schedulers = training.model, training.optimizers, training.schedulers
    if data_state is not None and not any(
        k.startswith("app.data.") for k in metadata.state_dict_metadata
    ):
        raise ValueError("This checkpoint has no data state; it cannot resume prepared Omni data")
    state = CheckpointState(
        model, optimizers, schedulers, model_only=model_only, data_state=data_state
    )
    dcp.load({"app": state}, checkpoint_id=location)
    rank = torch.distributed.get_rank()
    rng_path = Path(location, "rng-rank-%05d.pt" % rank)
    if rng_path.exists():
        rng_state = torch.load(rng_path, weights_only=True)
        torch.cuda.set_rng_state(rng_state)

    dt = torch.tensor(time.monotonic() - t0, device="cuda")
    dt_min, dt_max = dt.clone(), dt.clone()
    torch.distributed.all_reduce(dt_min, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(dt_max, op=torch.distributed.ReduceOp.MAX)
    stdout.info("Load checkpoint: elapsed min=%.1fs, max=%.1fs" % (dt_min.item(), dt_max.item()))
