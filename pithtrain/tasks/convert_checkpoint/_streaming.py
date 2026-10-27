"""
Copy plans for streaming a HuggingFace checkpoint into FSDP-sharded params (see
``pithtrain.modules.hf_loader``). A plan is a list of CopyOps; the per-model converters build
theirs from the two planners here, one per HF expert layout.
"""

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from pithtrain.modules.checkpoint import _expand_localized_fqn, expert_range, find_moe, strip_prefix


class CopyOp(NamedTuple):
    """
    One write into a local param: rows ``row_slice`` of ``hf_keys``, combined by ``transform``
    when set (MXFP4 dequant), copied into ``dst``.
    """

    local_fqn: str
    dst: torch.Tensor
    hf_keys: Tuple[str, ...]
    shard_files: Tuple[str, ...]
    row_slice: slice
    transform: Optional[Callable[..., torch.Tensor]]


def generic_plan_param(
    local_fqn: str,
    param: DTensor,
    weight_map: Dict[str, str],
    named_modules: Dict[str, nn.Module],
    remap: Callable[[str], str] = lambda k: k,
) -> List[CopyOp]:
    """
    Plan a dense param, or a stacked-expert param whose HF checkpoint stores one tensor per expert
    (Qwen3, DeepSeek-V2). ``remap`` maps a canonical key to its HF key.
    """
    local = param._local_tensor
    canon_keys = _expand_localized_fqn(local_fqn, named_modules)
    moe = find_moe(local_fqn, named_modules)
    dp_rank = param.device_mesh.get_local_rank()
    dp_size = param.device_mesh.size()

    if moe is None:
        (canon,) = canon_keys
        hf_key = remap(canon)
        chunk = -(-param.shape[0] // dp_size)
        start = min(dp_rank * chunk, param.shape[0])
        length = max(0, min(chunk, param.shape[0] - start))
        return [
            CopyOp(
                local_fqn=local_fqn,
                dst=local,
                hf_keys=(hf_key,),
                shard_files=(weight_map[hf_key],),
                row_slice=slice(start, start + length),
                transform=None,
            )
        ]

    chunk = -(-moe.experts_per_rank // dp_size)
    dp_offset = min(dp_rank * chunk, moe.experts_per_rank)
    dp_len = max(0, min(chunk, moe.experts_per_rank - dp_offset))
    ops: List[CopyOp] = []
    for i, canon in enumerate(canon_keys[dp_offset : dp_offset + dp_len]):
        hf_key = remap(canon)
        ops.append(
            CopyOp(
                local_fqn=local_fqn,
                dst=local[i],
                hf_keys=(hf_key,),
                shard_files=(weight_map[hf_key],),
                row_slice=slice(None),
                transform=None,
            )
        )
    return ops


def fused_expert_ops(
    local_fqn: str,
    param: DTensor,
    weight_map: Dict[str, str],
    named_modules: Dict[str, nn.Module],
    hf_keys: Callable[[str], Tuple[str, ...]],
    transform: Optional[Callable[..., torch.Tensor]] = None,
) -> List[CopyOp]:
    """
    Plan a stacked-expert param whose HF checkpoint fuses every expert into one ``[E, ...]``
    tensor (GPT-OSS, Qwen3.5), one CopyOp per local expert. ``hf_keys`` maps the fused canonical
    key to the HF key(s) read for it.
    """
    local = param._local_tensor
    moe = find_moe(local_fqn, named_modules)
    keys = hf_keys(strip_prefix(local_fqn).removesuffix(".weight"))
    dp_rank = param.device_mesh.get_local_rank()
    dp_size = param.device_mesh.size()
    chunk = -(-moe.experts_per_rank // dp_size)
    dp_offset = min(dp_rank * chunk, moe.experts_per_rank)
    dp_len = max(0, min(chunk, moe.experts_per_rank - dp_offset))
    shard_files = tuple(weight_map[key] for key in keys)
    first = expert_range(moe)[0] + dp_offset
    return [
        CopyOp(
            local_fqn=local_fqn,
            dst=local[i : i + 1],
            hf_keys=keys,
            shard_files=shard_files,
            row_slice=slice(first + i, first + i + 1),
            transform=transform,
        )
        for i in range(dp_len)
    ]
