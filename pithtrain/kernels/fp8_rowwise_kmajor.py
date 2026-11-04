"""
Picker of fp8cast_rowwise_kmajor, the SM90 rowwise plus K-major per-group colwise FP8 quantization
(CuTe-DSL kernels in fp8_rowwise_kmajor_sm90); SM90 only.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["fp8cast_rowwise_kmajor"]


def fp8cast_rowwise_kmajor(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise and K-major colwise FP8 quantization of x from one read.

    x is bf16 or fp32 (M, N) with M % 128 == 0. grouped_mm_offs is int32 (num_groups,), the
    cumulative group ends (no leading zero, last entry M), each a multiple of 128. It also takes
    fp16, any strides and base alignment of x, and int64 or non-contiguous offsets; M == 0 or
    N == 0 gives empty outputs for any num_groups, and otherwise num_groups == 0 or M > 2^31 - 1
    raises ValueError. Inputs are not modified.

    Returns (tok_fp8, tok_scale, kmajor_fp8, kmajor_scale):
    - tok_fp8 float8_e4m3fn (M, N) and tok_scale float32 (M, ceil(N / 128)): 1x128
      blocks along each row.
    - kmajor_fp8 float8_e4m3fn (M * N,) and kmajor_scale float32 (N, M / 128): 128x1 blocks down
      each column within a group; group g, with K_g rows, is stored as its (N, K_g) row-major block
      starting at grouped_mm_offs[g - 1] * N (0 for g = 0).

    SM90 only: the K-major per-channel layout is what SM90's k-grouped GEMM reads; on SM100
    fp8_quant_grouped writes the (M, N) layout instead, and this raises there. A block holding NaN
    gets the scale bits 0x80000000 (its amax compares |x| bit patterns, where NaN is largest), and a
    block whose amax is infinite gets scale inf. Both scales are MN-major (see fp8_scale_layout),
    the layout DeepGEMM's SM90 GEMMs read without a transpose.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import fp8cast_rowwise_kmajor_sm90
            return fp8cast_rowwise_kmajor_sm90(x, grouped_mm_offs)
        case _:
            raise NotImplementedError(f"fp8cast_rowwise_kmajor runs on SM90 only (SM100's fp8_quant_grouped writes the (M, K) per-channel layout instead), not compute capability {ARCH_MAJOR}.x")
