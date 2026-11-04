"""
Picker of the weight-side 128x128 blockwise FP8 quantization: the blockwise transposes of a weight
and the FP8 parameter all-gather kernels.

The blockwise transposes run the Triton kernel on both architectures: they are called inside
torch.compile regions, which cannot trace the CuTe-DSL launchers. The all-gather kernels run the
CuTe-DSL port on SM90 and the Triton kernel on SM100. Every scale is row-major: the E8M0
2^ceil(log2(max(amax, 1e-4) / 448)) of its 128x128 tile's amax.
"""

from typing import Any, Optional, Sequence

import torch

from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.kernels.fp8_weight_quant_sm100 import fp8_quant_weight_grouped_sm100, fp8_quant_weight_sm100

__all__ = ["fp8_pack_transpose", "fp8_scales", "fp8_scales_multi", "fp8_shard_amax", "fp8_shard_amax_multi", "fp8_shard_cast", "fp8_quant_weight", "fp8_quant_weight_grouped"]


def fp8_quant_weight(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    128x128 blockwise FP8 of a weight and its transpose from one read.

    x is 2-D (N, K) bf16 (or fp32), any N, K and strides (a partial tile is quantized over its valid
    elements). w_fp8 is e4m3 (N, K), w_scale fp32 (ceil(N / 128), ceil(K / 128)); wt_fp8 is e4m3
    (K, N), exactly w_fp8.T, and wt_scale is w_scale.T. All outputs are fresh and contiguous.
    """
    return fp8_quant_weight_sm100(x)


def fp8_quant_weight_grouped(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    fp8_quant_weight of each expert of a grouped weight x, 3-D (G, N, K) bf16 (or fp32),
    any strides. Returns w_fp8 e4m3 (G, N, K), w_scale fp32 (G, ceil(N / 128), ceil(K / 128)),
    wt_fp8 e4m3 (G, K, N) and wt_scale fp32 (G, ceil(K / 128), ceil(N / 128)), all contiguous.
    Tiles never straddle experts.
    """
    return fp8_quant_weight_grouped_sm100(x)


def fp8_shard_amax(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:
    """
    In place: the 128x128-tile amax of one row shard of a (G, N, K) weight, written into the
    shared amax grid.

    shard is fp32 (g_local, local_rows, K) contiguous: rows [row_offset, row_offset + local_rows) of
    experts [group_offset, group_offset + g_local) (a 2-D weight is G = 1). amax is fp32
    (G, ceil(N / 128), ceil(K / 128)) with contiguous rows. Each tile the shard touches is
    overwritten with max(|bf16(x)|) over the shard's elements in it; other entries are left alone.
    Zero the grid first; a MAX all-reduce over the shards then gives the amax of the whole bf16
    weight. NaN inputs are out of contract.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_amax_sm90
            return fp8_shard_amax_sm90(shard, amax, row_offset, group_offset)
        case 10:
            from pithtrain.kernels.fp8_weight_quant_sm100 import fp8_shard_amax_sm100
            fp8_shard_amax_sm100(shard, amax, row_offset, group_offset)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_shard_cast(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:
    """
    In place: the FP8 cast of one row shard with the scales of the reduced amax grid.

    shard, row_offset and group_offset are as in fp8_shard_amax; amax is the reduced grid (read
    only). out, e4m3 of the shard's shape and contiguous, is overwritten with e4m3(bf16(x) / scale).
    The gathered shards equal the w_fp8 of fp8_quant_weight_grouped on the bf16 weight.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_cast_sm90
            return fp8_shard_cast_sm90(shard, amax, out, row_offset, group_offset)
        case 10:
            from pithtrain.kernels.fp8_weight_quant_sm100 import fp8_shard_cast_sm100
            fp8_shard_cast_sm100(shard, amax, out, row_offset, group_offset)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_scales(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    """
    In place: scale fp32 (G, R, C) and scale_t fp32 (G, C, R), both contiguous, overwritten with the
    E8M0 scales of the contiguous fp32 (G, R, C) amax grid (scale_t[g] = scale[g].T).
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_scales_sm90
            return fp8_scales_sm90(amax, scale, scale_t)
        case 10:
            from pithtrain.kernels.fp8_weight_quant_sm100 import fp8_scales_sm100
            fp8_scales_sm100(amax, scale, scale_t)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_pack_transpose(x: torch.Tensor, out: torch.Tensor) -> None:
    """
    In place: x, a one-byte-element (G, N, K) contiguous tensor, and its per-expert transpose
    (G, K, N), written back to back into out, a contiguous one-byte-element buffer of at least
    2 * x.numel() elements.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_pack_transpose_sm90
            return fp8_pack_transpose_sm90(x, out)
        case 10:
            from pithtrain.kernels.fp8_weight_quant_sm100 import fp8_pack_transpose_sm100
            fp8_pack_transpose_sm100(x, out)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_shard_amax_multi(shards: Sequence[torch.Tensor], amaxes: Sequence[torch.Tensor], row_offsets: Sequence[int], group_offsets: Sequence[int]) -> Optional[Any]:
    """
    A callable that runs fp8_shard_amax for every (shard, amax, row_offset, group_offset) of the
    lists in one launch, or None on SM100, where the caller loops over fp8_shard_amax itself.

    It records the tensors' addresses, so it stays valid only while none of them is reallocated.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_amax_multi_sm90
            return fp8_shard_amax_multi_sm90(shards, amaxes, row_offsets, group_offsets)
        case 10:
            return None
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_scales_multi(amaxes: Sequence[torch.Tensor], scales: Sequence[torch.Tensor], scales_t: Sequence[torch.Tensor]) -> Optional[Any]:
    """
    A callable that runs fp8_scales for every (amax, scale, scale_t) of the lists in one launch,
    or None on SM100, where the caller loops over fp8_scales itself. Same address
    caveat as fp8_shard_amax_multi.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_scales_multi_sm90
            return fp8_scales_multi_sm90(amaxes, scales, scales_t)
        case 10:
            return None
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
