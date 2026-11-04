import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["fp8cast_rowwise_kmajor"]


def fp8cast_rowwise_kmajor(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise and K-major colwise FP8 quantization of x from one read.

    x is bf16 or fp32 (M, N) with M % 128 == 0. grouped_mm_offs is int32 (num_groups,), the
    cumulative group ends (no leading zero, last entry M), each a multiple of 128.

    Returns (tok_fp8, tok_scale, kmajor_fp8, kmajor_scale):
    - tok_fp8 float8_e4m3fn (M, N) and tok_scale float32 (M, ceil(N / 128)): 1x128
      blocks along each row.
    - kmajor_fp8 float8_e4m3fn (M * N,) and kmajor_scale float32 (N, M / 128): 128x1 blocks down
      each column within a group; group g, with K_g rows, is stored as its (N, K_g) row-major block
      starting at grouped_mm_offs[g - 1] * N (0 for g = 0).

    FP8 bytes and scale values are the same on both architectures, the scale strides are not: SM90
    returns both scales MN-major (see fp8_scale_layout), the layout DeepGEMM's SM90 GEMMs read
    without a transpose; SM100 returns them row-major.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import fp8cast_rowwise_kmajor_sm90
            return fp8cast_rowwise_kmajor_sm90(x, grouped_mm_offs)
        case 10:
            from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100
            return fp8cast_rowwise_kmajor_sm100(x, grouped_mm_offs)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
