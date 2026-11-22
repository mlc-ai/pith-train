"""
FP8 weight quantization at every weight_quant shape, bit-exact against the 128x128 E8M0 recipe:
fp8_quant_weight(_grouped) and the FP8 parameter all-gather kernels over SHARDS simulated ranks.
"""

import pytest
import torch

from pithtrain.kernels import (
    fp8_pack_transpose,
    fp8_quant_weight,
    fp8_quant_weight_grouped,
    fp8_scales,
    fp8_scales_multi,
    fp8_shard_amax,
    fp8_shard_amax_multi,
    fp8_shard_cast,
)
from tests.kernels.cases import params
from tests.kernels.fp8_reference import (
    BLOCK,
    assert_fp8_equal,
    assert_scale_equal,
    blocks,
    e8m0_scale,
    quant_blockwise,
)

DEV = "cuda"
SHARDS = 8


def master_weight(case: dict) -> torch.Tensor:
    """
    fp32 (G, N, K) (G = 1 for a dense weight), scaled per expert over a few binades.
    """
    axes = case["axes"]
    G, N, K = max(axes["G"], 1), axes["N"], axes["K"]
    gen = torch.Generator(device=DEV).manual_seed(N * 31 + K)
    w = torch.randn(G, N, K, device=DEV, generator=gen) * 0.02
    w *= torch.exp2(torch.randint(-3, 4, (G, 1, 1), device=DEV, generator=gen).float())
    return w


@pytest.mark.parametrize("case", params("weight_quant"))
def test_quant_weight(case: dict) -> None:
    w = master_weight(case).to(torch.bfloat16)
    ref, ref_scale = quant_blockwise(w)
    if case["axes"]["G"] == 0:
        w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(w[0])
        ref, ref_scale = ref[0], ref_scale[0]
    else:
        w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight_grouped(w)
    assert_fp8_equal(w_fp8, ref, "w_fp8")
    assert_scale_equal(w_scale, ref_scale, "w_scale")
    assert_fp8_equal(wt_fp8, ref.transpose(-1, -2).contiguous(), "wt_fp8")
    assert_scale_equal(wt_scale, ref_scale.transpose(-1, -2).contiguous(), "wt_scale")


def fsdp_shards(w: torch.Tensor, dense: bool) -> list[tuple[torch.Tensor, int, int]]:
    """
    (shard, row_offset, group_offset) per non-empty rank, cut on dim 0 as FSDP does.
    """
    shards = []
    if dense:
        rows = w.shape[1]
        chunk = -(-rows // SHARDS)
        for start in range(0, rows, chunk):
            shards.append((w[:, start : start + chunk].contiguous(), start, 0))
    else:
        G = w.shape[0]
        chunk = -(-G // SHARDS)
        for start in range(0, G, chunk):
            shards.append((w[start : start + chunk].contiguous(), 0, start))
    return shards


def shard_tile_amax(shard: torch.Tensor, row_offset: int, group_offset: int, grid_shape) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """
    The reference tile-amax grid of one shard (zero outside its tiles) and the mask of its tiles.
    """
    G, R, C = grid_shape
    g_local, rows, K = shard.shape
    grid = torch.zeros(grid_shape, dtype=torch.float32, device=DEV)
    touched = torch.zeros(grid_shape, dtype=torch.bool, device=DEV)
    first, last = row_offset // BLOCK, (row_offset + rows - 1) // BLOCK
    padded = torch.zeros(g_local, (last + 1) * BLOCK, C * BLOCK, dtype=torch.float32, device=DEV)
    padded[:, row_offset : row_offset + rows, :K] = shard.to(torch.bfloat16).float().abs()
    tiles = padded.view(g_local, last + 1, BLOCK, C, BLOCK).amax(dim=(2, 4))
    grid[group_offset : group_offset + g_local, first : last + 1] = tiles[:, first:]
    touched[group_offset : group_offset + g_local, first : last + 1] = True
    return grid, touched


@pytest.mark.parametrize("case", params("weight_quant"))
def test_param_gather(case: dict) -> None:
    dense = case["axes"]["G"] == 0
    w = master_weight(case)
    G, N, K = w.shape
    grid_shape = (G, blocks(N), blocks(K))
    shards = fsdp_shards(w, dense)

    # torch.maximum stands in for the all-reduce's MAX
    grids, reduced = [], torch.zeros(grid_shape, dtype=torch.float32, device=DEV)
    for shard, row_offset, group_offset in shards:
        grid = torch.zeros(grid_shape, dtype=torch.float32, device=DEV)
        fp8_shard_amax(shard, grid, row_offset, group_offset)
        want, _ = shard_tile_amax(shard, row_offset, group_offset, grid_shape)
        assert_scale_equal(grid, want, f"amax of shard at rows {row_offset} groups {group_offset}")
        grids.append(grid)
        reduced = torch.maximum(reduced, grid)
    w_bf16 = w.to(torch.bfloat16)
    ref, ref_scale = quant_blockwise(w_bf16)
    padded = torch.zeros(G, grid_shape[1] * BLOCK, grid_shape[2] * BLOCK, device=DEV)
    padded[:, :N, :K] = w_bf16.float().abs()
    whole_amax = padded.view(G, grid_shape[1], BLOCK, grid_shape[2], BLOCK).amax(dim=(2, 4))
    assert_scale_equal(reduced, whole_amax, "reduced amax")

    multi_grids = [torch.zeros(grid_shape, dtype=torch.float32, device=DEV) for _ in shards]
    launch = fp8_shard_amax_multi([s for s, _, _ in shards], multi_grids, [r for _, r, _ in shards], [g for _, _, g in shards])  # fmt: skip
    if launch is not None:
        launch()
        for index, (got, want) in enumerate(zip(multi_grids, grids)):
            assert_scale_equal(got, want, f"fp8_shard_amax_multi shard {index}")

    scale = torch.empty(grid_shape, dtype=torch.float32, device=DEV)
    scale_t = torch.empty(G, grid_shape[2], grid_shape[1], dtype=torch.float32, device=DEV)
    fp8_scales(reduced, scale, scale_t)
    assert_scale_equal(scale, e8m0_scale(whole_amax), "fp8_scales")
    assert_scale_equal(scale, ref_scale, "fp8_scales vs recipe")
    assert_scale_equal(scale_t, ref_scale.transpose(1, 2).contiguous(), "fp8_scales transposed")
    scale_m, scale_t_m = torch.empty_like(scale), torch.empty_like(scale_t)
    launch = fp8_scales_multi([reduced], [scale_m], [scale_t_m])
    if launch is not None:
        launch()
        assert_scale_equal(scale_m, scale, "fp8_scales_multi")
        assert_scale_equal(scale_t_m, scale_t, "fp8_scales_multi transposed")

    outs = []
    for shard, row_offset, group_offset in shards:
        out = torch.empty(shard.shape, dtype=torch.float8_e4m3fn, device=DEV)
        fp8_shard_cast(shard, reduced, out, row_offset, group_offset)
        outs.append(out)
    gathered = torch.cat(outs, dim=1 if dense else 0)
    assert_fp8_equal(gathered, ref, "gathered fp8_shard_cast")

    # may exceed 2^31 bytes
    packed = torch.empty(2 * gathered.numel(), dtype=torch.uint8, device=DEV)
    fp8_pack_transpose(gathered, packed)
    numel = gathered.numel()
    assert_fp8_equal(packed[:numel].view(torch.float8_e4m3fn).view(G, N, K), ref, "packed fp8")
    assert_fp8_equal(packed[numel:].view(torch.float8_e4m3fn).view(G, K, N), ref.transpose(1, 2).contiguous(), "packed fp8_t")  # fmt: skip
