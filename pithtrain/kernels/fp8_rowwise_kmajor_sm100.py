"""
The reference the SM90 kernel of fp8cast_rowwise_kmajor is bit-identical to.
"""

import torch

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_rowwise_colwise import fp8cast_rowwise_colwise_kernel

__all__ = ["fp8cast_rowwise_kmajor_sm100", "kmajor_empty_outputs"]

BLOCK = 128


def fp8cast_rowwise_kmajor_sm100(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2 and x.shape[0] % 128 == 0, f"Expected 2D tensor with M%128==0, got shape {x.shape}"
    M, N = x.shape

    num_row_blocks, scale_tok_cols = M // BLOCK, (N + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    kmajor_fp8 = torch.empty(M * N, dtype=torch.float8_e4m3fn, device=x.device)
    kmajor_scale = torch.empty((N, num_row_blocks), dtype=torch.float32, device=x.device)

    # groups are 128-aligned, so the group of a row block is the group of its first row
    row_block_starts = torch.arange(0, M, BLOCK, device=x.device, dtype=torch.int64)
    block_to_group = torch.searchsorted(grouped_mm_offs.to(torch.int64), row_block_starts, right=True).to(torch.int32)
    block_to_group.clamp_(max=grouped_mm_offs.shape[0] - 1)

    BLOCK_N = 128
    grid = (num_row_blocks, (N + BLOCK_N - 1) // BLOCK_N)

    fp8cast_rowwise_colwise_kernel[grid](x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_ch_ptr=kmajor_fp8, scale_ch_ptr=kmajor_scale, M=M, N=N, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_tok_row=N, scale_tok_cols=scale_tok_cols, stride_out_ch_row=0, scale_ch_cols=num_row_blocks, block_to_group_ptr=block_to_group, cumsum_ptr=grouped_mm_offs, BLOCK_N=BLOCK_N, NATIVE_E8M0=NATIVE_E8M0, WRITE_KMAJOR=True, num_warps=4, num_stages=2)

    return out_tok, scale_tok, kmajor_fp8, kmajor_scale


def kmajor_empty_outputs(M: int, N: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), torch.empty((M, (N + BLOCK - 1) // BLOCK), dtype=torch.float32, device=device), torch.empty(M * N, dtype=torch.float8_e4m3fn, device=device), torch.empty((N, M // BLOCK), dtype=torch.float32, device=device))
