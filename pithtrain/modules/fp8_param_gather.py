"""
FP8 parameter all-gather for FSDP2: gather the FP8 bytes of 128x128 block-scaled weights instead
of bf16, so no bf16 copy of an FP8 weight exists and the all-gather moves one byte per element.

precompute_fp8_param_scales computes the per-tile amax of each rank's fp32 shard, reduces it with
one packed all-reduce MAX per FSDP shard group, derives the scales and casts the shard to FP8. It
must run after every change to the fp32 weights (init, checkpoint load, optimizer step). FSDP
then all-gathers that FP8 shard, and the post-all-gather hook packs the gathered weight and its
transpose into one buffer of the bf16 weight's size. The unsharded parameter is a bf16 view of
that buffer, so autograd and FSDP's gradient path are unchanged, and the module reads the operands
through the weight_fp8 and weight_t_fp8 buffers that view the same storage.

The FP8 bytes and scales are bit-identical to fp8_quant_weight of the bf16 weight. The
hooks are plain functions set on the sharded local tensor, so no tensor subclass is involved.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, free_storage

from pithtrain.kernels import (
    fp8_pack_transpose,
    fp8_scales,
    fp8_scales_multi,
    fp8_shard_amax,
    fp8_shard_amax_multi,
    fp8_shard_cast,
)

__all__ = [
    "fp8_param_operands",
    "install_fp8_param_gather",
    "install_raw_fp8_param_gather",
    "precompute_fp8_param_scales",
]

BLOCK = 128


@dataclass
class FP8ParamState:
    module: nn.Module
    fsdp_param: FSDPParam
    shape: torch.Size
    """
    Global shape, (N, K) or (G, N, K).
    """
    row_offset: int
    """
    First global row of N this rank holds; 0 for a 3-D weight, which is sharded over G.
    """
    group_offset: int
    """
    First global group of G this rank holds; 0 for a 2-D weight.
    """
    amax: torch.Tensor
    """
    The (G, ceil(N / 128), ceil(K / 128)) amax grid, a view into its shard group's packed buffer.
    """
    version: int = -1
    """
    Version counter of the local shard when amax was last computed.
    """
    packed: Optional[torch.Tensor] = None
    """
    Flat byte buffer [fp8 | fp8_t] behind the unsharded parameter.
    """
    fp8_shard: Optional[torch.Tensor] = None
    """
    This rank's shard cast to FP8 at FSDP's padded shard size: the all-gather input.
    """

    def local_shard(self) -> torch.Tensor:
        """
        The local fp32 shard as (local groups, local rows, K).
        """
        local = self.fsdp_param._sharded_local_tensor
        return local.view(1, *local.shape) if len(self.shape) == 2 else local

    def pre_all_gather(
        self,
        mesh: Any,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: Any,
    ) -> tuple[tuple[torch.Tensor, ...], None]:
        if self.shard_freed():
            self.cast_shard()
        assert self.fsdp_param._sharded_local_tensor._version == self.version, (
            "FP8 shard is stale: call precompute_fp8_param_scales after changing the weights"
        )
        return (self.fp8_shard,), None

    def shard_freed(self) -> bool:
        return self.fp8_shard is not None and self.fp8_shard.untyped_storage().size() == 0

    def cast_shard(self) -> None:
        padded = self.fsdp_param.padded_sharded_param_size
        if self.shard_freed():
            self.fp8_shard.untyped_storage().resize_(self.fp8_shard.numel())
        if self.fp8_shard is None:
            # FSDP needs the padded size even for an uneven shard; the padding rows are dropped
            # after the gather, so they stay uninitialised.
            device = self.fsdp_param.sharded_param.device
            self.fp8_shard = torch.empty(padded, dtype=torch.uint8, device=device)
        out = self.fp8_shard.view(torch.float8_e4m3fn)
        shard = self.local_shard()
        if len(self.shape) == 2:
            dst = out.view(1, *padded)[:, : shard.shape[1]]
        else:
            dst = out[: shard.shape[0]]
        fp8_shard_cast(shard, self.amax, dst, self.row_offset, self.group_offset)

    def post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: None,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> Optional[tuple[torch.Tensor, tuple[torch.Tensor, ...]]]:
        """
        FSDP passes out=None only on the first all-gather, which allocates the packed buffer and
        returns its bf16 view as the unsharded parameter; later calls refill it in
        place and return None.
        """
        assert param_dtype == torch.bfloat16, param_dtype
        (gathered,) = all_gather_outputs
        gathered = gathered[: self.shape[0]].view(torch.float8_e4m3fn).view(self.shape)
        gathered = gathered.view(-1, *self.shape[-2:])
        numel = gathered.numel()
        if out is None:
            self.packed = torch.empty(2 * numel, dtype=torch.uint8, device=gathered.device)
            data = self.packed.view(torch.float8_e4m3fn)
            *lead, N, K = self.shape
            fp8, fp8_t = data[:numel].view(*lead, N, K), data[numel:].view(*lead, K, N)
            self.module.register_buffer("weight_fp8", fp8, persistent=False)
            self.module.register_buffer("weight_t_fp8", fp8_t, persistent=False)
        fp8_pack_transpose(gathered, self.packed)
        # The bytes now live in the packed buffer; the next unshard reallocates the gather output.
        # Both run on the current stream, so the free is ordered.
        for output in self.fsdp_param.all_gather_outputs:
            free_storage(output)
        # FSDP copied the FP8 shard into its all-gather buffer, so it is dead until the next
        # pre_all_gather recasts it.
        free_storage(self.fp8_shard)
        if out is not None:
            return None
        return self.packed.view(torch.bfloat16).view(self.shape), (self.packed,)


fp8_states: list[FP8ParamState] = []
fp8_groups: list[tuple[dist.ProcessGroup, torch.Tensor, list[FP8ParamState]]] = []


def install_fp8_param_gather(model: nn.Module, fp8_classes: tuple[type, ...]) -> None:
    """
    Make FSDP2 all-gather the weight of every module of fp8_classes in FP8, registering the
    weight_scale and weight_scale_t buffers on each module and computing the scales once.

    Call after fully_shard and before the first forward. Collective over each shard group.
    """
    fp8_states.clear()
    fp8_groups.clear()
    by_group: dict[int, tuple[dist.ProcessGroup, list[tuple[FP8ParamState, int]]]] = {}
    for fsdp_module in model.modules():
        if not isinstance(fsdp_module, FSDPModule):
            continue
        param_group = fsdp_module._get_fsdp_state()._fsdp_param_group
        if param_group is None:
            continue
        for fsdp_param in param_group.fsdp_params:
            module = fsdp_param._module_info.module
            if not isinstance(module, fp8_classes) or fsdp_param._module_info.param_name != "weight":  # fmt: skip
                continue
            assert fsdp_param.fsdp_placement.dim == 0
            shape = fsdp_param._orig_size
            assert len(shape) in (2, 3), shape
            mesh_info = fsdp_param.mesh_info
            rank, size = mesh_info.shard_mesh_rank, mesh_info.shard_mesh_size
            chunk = (shape[0] + size - 1) // size
            offset = min(rank * chunk, shape[0])
            G, N, K = (1, *shape) if len(shape) == 2 else shape
            grid = (G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)
            device = fsdp_param.sharded_param.device
            scale = torch.empty(grid, dtype=torch.float32, device=device)
            scale_t = torch.empty(grid[0], grid[2], grid[1], dtype=torch.float32, device=device)
            if len(shape) == 2:
                scale, scale_t = scale[0], scale_t[0]
            module.register_buffer("weight_scale", scale, persistent=False)
            module.register_buffer("weight_scale_t", scale_t, persistent=False)
            module.fp8_param_gather = True
            row_offset, group_offset = (offset, 0) if len(shape) == 2 else (0, offset)
            state = FP8ParamState(module, fsdp_param, shape, row_offset, group_offset, torch.empty(0))  # fmt: skip
            group = mesh_info.shard_process_group
            entry = by_group.setdefault(id(group), (group, []))
            entry[1].append((state, G * grid[1] * grid[2]))
            local = fsdp_param._sharded_local_tensor
            local.fsdp_pre_all_gather = state.pre_all_gather
            local.fsdp_post_all_gather = state.post_all_gather
            fsdp_param._init_extensions()
            fp8_states.append(state)

    # One packed amax buffer per shard group, so each group reduces in a single collective.
    for group, entries in by_group.values():
        device = entries[0][0].fsdp_param.sharded_param.device
        buffer = torch.zeros(sum(num_tiles for _, num_tiles in entries), dtype=torch.float32, device=device)  # fmt: skip
        start = 0
        for state, num_tiles in entries:
            G, N, K = (1, *state.shape) if len(state.shape) == 2 else state.shape
            state.amax = buffer[start : start + num_tiles].view(G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)  # fmt: skip
            start += num_tiles
        fp8_groups.append((group, buffer, [state for state, _ in entries]))

    precompute_fp8_param_scales()


@torch.no_grad()
def precompute_fp8_param_scales() -> None:
    """
    Recompute the FP8 scales and the FP8 shard of every installed weight from its fp32 shard.
    Collective over each FSDP shard group, so every rank must call it at the same point.
    """
    for group, buffer, states in fp8_groups:
        buffer.zero_()
        launches = multi_launches(states)
        if launches is not None:
            launches[0]()
        else:
            for state in states:
                shard, amax = state.local_shard(), state.amax
                fp8_shard_amax(shard, amax, state.row_offset, state.group_offset)
        if group.size() > 1:
            dist.all_reduce(buffer, op=dist.ReduceOp.MAX, group=group)
        if launches is not None:
            launches[1]()
        for state in states:
            if launches is None:
                fp8_scales(state.amax, *scale_grids(state.module))
            state.cast_shard()
            state.version = state.fsdp_param._sharded_local_tensor._version


def scale_grids(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The module's scale buffers as 3-D grids (G, R, C) and (G, C, R).
    """
    scale, scale_t = module.weight_scale, module.weight_scale_t
    if scale.dim() == 2:
        scale, scale_t = scale.unsqueeze(0), scale_t.unsqueeze(0)
    return scale, scale_t


multi_launch_cache: dict[int, tuple[tuple[int, ...], tuple[Any, Any]]] = {}


def multi_launches(states: list[FP8ParamState]) -> Optional[tuple[Any, Any]]:
    """
    One amax launch and one scales launch covering every weight of a shard group, or None where
    the kernel library has no such launch, in which case the caller loops over the weights. Cached
    per group and rebuilt when any shard moves.
    """
    shards = [state.local_shard() for state in states]
    key = tuple(shard.data_ptr() for shard in shards)
    cached = multi_launch_cache.get(id(states))
    if cached is None or cached[0] != key:
        amaxes = [state.amax for state in states]
        grids = [scale_grids(state.module) for state in states]
        amax_launch = fp8_shard_amax_multi(
            shards,
            amaxes,
            [state.row_offset for state in states],
            [state.group_offset for state in states],
        )
        if amax_launch is None:
            return None
        scales_launch = fp8_scales_multi(amaxes, [grid[0] for grid in grids], [grid[1] for grid in grids])  # fmt: skip
        cached = (key, (amax_launch, scales_launch))
        multi_launch_cache[id(states)] = cached
    return cached[1]


# Raw expert parameters (gpt-oss's gate_up_proj and down_proj): a module lists them in its
# fp8_raw_params class attribute, and each gets a holder module that plays the part of the FP8
# module for the gather. The holder sits in a plain dict in the module's __dict__, outside the
# module tree, so parameter names and checkpoints are unchanged.
HOLDERS_KEY = "_fp8_param_holders"


class RawParamHolder(nn.Module):
    fp8_param_gather: bool = True


def fp8_param_operands(module: nn.Module, name: str) -> Optional[tuple[torch.Tensor, ...]]:
    """
    The FP8 operands (fp8, scale, fp8_t, scale_t) of the raw weight name, or None when it is not
    gathered in FP8.
    """
    holder = module.__dict__.get(HOLDERS_KEY, {}).get(name)
    if holder is None:
        return None
    return holder.weight_fp8, holder.weight_scale, holder.weight_t_fp8, holder.weight_scale_t


def install_raw_fp8_param_gather(model: nn.Module) -> None:
    """
    Make FSDP2 all-gather in FP8 every raw parameter named in a module's fp8_raw_params.

    Call after install_fp8_param_gather, which resets the registry, and before the first forward.
    Collective over each shard group.
    """
    by_group: dict[int, tuple[dist.ProcessGroup, list[tuple[FP8ParamState, int]]]] = {}
    for fsdp_module in model.modules():
        if not isinstance(fsdp_module, FSDPModule):
            continue
        param_group = fsdp_module._get_fsdp_state()._fsdp_param_group
        if param_group is None:
            continue
        for fsdp_param in param_group.fsdp_params:
            module = fsdp_param._module_info.module
            name = fsdp_param._module_info.param_name
            if name not in getattr(module, "fp8_raw_params", ()):
                continue
            assert fsdp_param.fsdp_placement.dim == 0
            shape = fsdp_param._orig_size
            assert len(shape) == 3, shape
            mesh_info = fsdp_param.mesh_info
            rank, size = mesh_info.shard_mesh_rank, mesh_info.shard_mesh_size
            chunk = (shape[0] + size - 1) // size
            offset = min(rank * chunk, shape[0])
            G, N, K = shape
            grid = (G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)
            device = fsdp_param.sharded_param.device
            holder = RawParamHolder()
            holder.register_buffer("weight_scale", torch.empty(grid, dtype=torch.float32, device=device), persistent=False)  # fmt: skip
            holder.register_buffer("weight_scale_t", torch.empty(G, grid[2], grid[1], dtype=torch.float32, device=device), persistent=False)  # fmt: skip
            module.__dict__.setdefault(HOLDERS_KEY, {})[name] = holder
            state = FP8ParamState(holder, fsdp_param, shape, 0, offset, torch.empty(0))
            group = mesh_info.shard_process_group
            entry = by_group.setdefault(id(group), (group, []))
            entry[1].append((state, G * grid[1] * grid[2]))
            local = fsdp_param._sharded_local_tensor
            local.fsdp_pre_all_gather = state.pre_all_gather
            local.fsdp_post_all_gather = state.post_all_gather
            fsdp_param._init_extensions()
            fp8_states.append(state)

    for group, entries in by_group.values():
        device = entries[0][0].fsdp_param.sharded_param.device
        buffer = torch.zeros(sum(num_tiles for _, num_tiles in entries), dtype=torch.float32, device=device)  # fmt: skip
        start = 0
        for state, num_tiles in entries:
            G, N, K = state.shape
            state.amax = buffer[start : start + num_tiles].view(G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)  # fmt: skip
            start += num_tiles
        fp8_groups.append((group, buffer, [state for state, _ in entries]))

    if by_group:
        precompute_fp8_param_scales()
