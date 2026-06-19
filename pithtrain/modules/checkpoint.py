"""
Checkpoint resharding utilities.

Disk format (canonical):
  - No module.{N}. prefix
  - Experts individually indexed: layers.1.mlp.experts.3.gate_proj.weight

Runtime format (local):
  - DualPipeV prefix: module.0.layers.1.mlp.experts.gate_proj.weight
  - Experts stacked per EP rank: shape [experts_per_rank, ...]
"""

import re
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

__all__ = ["to_canonical_model", "to_canonical_optim", "to_localized_model", "to_localized_optim"]

MODULE_PREFIX_RE = re.compile(r"^module\.\d+\.")
INDEXED_EXPERT_RE = re.compile(r"(.*)\.experts\.(\d+)\.(.*)")


def strip_prefix(key: str) -> str:
    """Strip module.{N}. prefix from a DualPipeV FQN."""
    return MODULE_PREFIX_RE.sub("", key)


def find_moe(key: str, named_modules: Dict[str, nn.Module]) -> Optional[nn.Module]:
    """
    Return the MoE module for a stacked expert key, or None.

    A stacked key looks like module.0.layers.1.mlp.experts.gate_proj.weight
    (no numeric index after .experts.).  Already-indexed keys like
    layers.1.mlp.experts.3.gate_proj.weight return None.
    """
    if ".experts." not in key:
        return None
    moe_path, _, after = key.partition(".experts.")
    if after and after.split(".")[0].isdigit():
        return None
    mod = named_modules.get(moe_path)
    if mod and hasattr(mod, "ep_rank") and hasattr(mod, "experts_per_rank"):
        return mod
    return None


def expert_range(mod: nn.Module) -> Tuple[int, int]:
    """Global expert index range [start, end) for this EP rank."""
    start = mod.ep_rank * mod.experts_per_rank
    return start, start + mod.experts_per_rank


def unwrap_dtensor_experts(value: Any, expected_n: int) -> Optional[Tuple[Any, int, int]]:
    """
    Extract local expert data from a DTensor without triggering all_gather.

    When FSDP shards the stacked expert tensor along dim 0 via Shard(0),
    each DP rank holds a contiguous subset of experts.  This helper extracts
    that local subset and computes the global expert offset so that unpack
    can emit per-expert keys for only the experts this rank actually owns,
    with zero GPU communication.

    Works for both model tensors (a single DTensor) and optimizer state
    entries (a dict whose tensor values are DTensors).

    Returns (localized_value, local_expert_count, dp_expert_offset)
    or None if value is not a sharded DTensor.
    """

    def _info(dt: Any) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Return (local_tensor, local_n, dp_offset) for a Shard(0) DTensor, or None."""
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

    unstack(value, num_local_experts, local_idx) extracts one expert slice
    from a stacked value.  For model tensors this is v[i]; for optimizer
    state dicts it slices each sub-tensor.

    When values are FSDP-sharded DTensors the expert dimension is extracted
    locally (via unwrap_dtensor_experts) so that each DP rank emits keys
    only for the experts it owns -- no GPU all_gather is triggered.
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
    Remap canonical FQNs to local FQNs and repack individual experts into stacked format.

    fqn_map maps {canonical_fqn: local_fqn}.
    restack(by_global_idx) stacks individual expert values back into one.
    """
    result: Dict[str, Any] = {}
    to_stack: Dict[str, Dict[int, Any]] = {}

    for canon_fqn, value in entries.items():
        m = INDEXED_EXPERT_RE.match(canon_fqn)
        if m:
            prefix, idx_str, suffix = m.groups()
            stacked_canon = "%s.experts.%s" % (prefix, suffix)
            local = fqn_map.get(stacked_canon)
            if local is not None:
                moe_path = local.partition(".experts.")[0]
                moe = named_modules.get(moe_path)
                if moe and hasattr(moe, "ep_rank"):
                    s, e = expert_range(moe)
                    idx = int(idx_str)
                    if s <= idx < e:
                        to_stack.setdefault(local, {})[idx] = value
        else:
            local = fqn_map.get(canon_fqn)
            if local is not None:
                result[local] = value

    for local, by_idx in to_stack.items():
        result[local] = restack(by_idx)
    return result


def unstack_optim(entry: Dict[str, Any], n: int, i: int) -> Dict[str, Any]:
    """Extract one expert slice from a stacked optimizer state entry."""
    return {
        k: v[i] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == n else v
        for k, v in entry.items()
    }


def restack_tensors(by_idx: Dict[int, torch.Tensor]) -> torch.Tensor:
    """Stack individual expert tensors back into one."""
    return torch.stack([v for _, v in sorted(by_idx.items())])


def restack_optim(by_idx: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Stack individual expert optimizer state entries back into one."""
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
    """Canonicalize model state: strip module prefix, unstack experts."""
    return unpack(state_dict, dict(model.named_modules()), lambda v, n, i: v[i])


def _expand_local_fqn(local_fqn: str, named_modules: Dict[str, nn.Module]) -> list:
    """Map a local (runtime) FQN to its canonical (disk) FQN(s), expanding
    stacked experts."""
    canon = strip_prefix(local_fqn)
    moe = find_moe(local_fqn, named_modules)
    if moe is None:
        return [canon]
    start, end = expert_range(moe)
    return [canon.replace(".experts.", ".experts.%d." % idx, 1) for idx in range(start, end)]


def _local_param_group_fqns(model: nn.Module, optimizers) -> list:
    """This rank's local FQNs per param group, across ``optimizers`` in order
    (the order DCP combines them in). Needed on load: DCP dedups param_groups
    metadata across PP ranks, so the loaded params lists aren't reliably this
    rank's -- we rebuild membership from the live optimizers instead."""
    param_to_fqn = {p: n for n, p in model.named_parameters()}
    groups = []
    for opt in optimizers:
        for g in opt.param_groups:
            groups.append([param_to_fqn[p] for p in g["params"]])
    return groups


def to_canonical_optim(optim_state: Dict, model: nn.Module) -> Dict:
    """Canonicalize optimizer state: strip module prefix, unstack expert states.

    Each param group keeps its *own* membership (its loaded FQNs mapped to
    canonical, experts expanded) rather than collapsing to the full param set, so a
    composed multi-optimizer state dict (Muon + AdamW, combined by DCP) round-trips
    without one group's hyperparameters leaking onto another's. Plain Adam is the
    degenerate case: its one group already owns every param.
    """
    named_modules = dict(model.named_modules())
    state = unpack(optim_state["state"], named_modules, unstack_optim)
    param_groups = []
    for g in optim_state["param_groups"]:
        group = {k: v for k, v in g.items() if k != "params"}
        group["params"] = [cf for lf in g["params"] for cf in _expand_local_fqn(lf, named_modules)]
        param_groups.append(group)
    return {"state": state, "param_groups": param_groups}


def rewrap_dtensor_experts(result: Dict[str, Any], model: nn.Module) -> None:
    """
    Re-wrap restacked plain tensors as DTensors to match model parameters.

    When unpack extracts local FSDP shards via unwrap_dtensor_experts,
    the individual expert tensors are plain (non-DTensor).  After repack
    stacks them the result is a plain tensor whose shape equals the FSDP local
    shard (e.g. [16, ...] when the full stacked parameter is [32, ...]).

    model.load_state_dict compares against the DTensor global shape, so
    we must wrap these plain tensors back into DTensors with the same mesh and
    placements as the model parameter.  For optimizer state entries (dicts of
    tensors), every tensor with dim() > 0 is wrapped.
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
    """Localize model state: remap FQNs to this rank's local keys, restack experts."""
    named_modules = dict(model.named_modules())
    model_keys = set(model.state_dict().keys())
    fqn_map = {strip_prefix(k): k for k in model_keys}
    result = repack(canonical, fqn_map, named_modules, restack_tensors)
    rewrap_dtensor_experts(result, model)
    return result


def to_localized_optim(optim_state: Dict, model: nn.Module, optimizers) -> Dict:
    """
    Localize optimizer state: remap FQNs, restack experts, rebuild param_groups.

    Membership is rebuilt from the live ``optimizers`` (DCP dedups param_groups
    across PP ranks, so the loaded lists aren't reliably this rank's). Each loaded
    group is matched by position to the live group (the order DCP combined them
    in); its hyperparameters are kept, its membership replaced with that group's
    local FQNs.
    """
    named_modules = dict(model.named_modules())
    fqn_map = {strip_prefix(n): n for n, _ in model.named_parameters()}
    state = repack(optim_state["state"], fqn_map, named_modules, restack_optim)
    rewrap_dtensor_experts(state, model)
    local_groups = _local_param_group_fqns(model, optimizers)
    loaded_groups = optim_state["param_groups"]
    assert len(loaded_groups) == len(local_groups), (
        f"param_group count mismatch: checkpoint has {len(loaded_groups)}, "
        f"live optimizers have {len(local_groups)}"
    )
    param_groups = []
    for g, members in zip(loaded_groups, local_groups):
        group = {k: v for k, v in g.items() if k != "params"}
        group["params"] = members
        param_groups.append(group)
    return {"state": state, "param_groups": param_groups}
