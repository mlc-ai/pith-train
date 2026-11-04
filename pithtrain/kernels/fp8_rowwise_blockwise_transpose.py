import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["fp8_quant_act"]


def fp8_quant_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise 1x128 FP8 of x and 128x128 blockwise FP8 of x.T from one read of x, returning (x_fp8,
    x_scale, xt_fp8, xt_scale): the activation operands of the dense FP8 linear.

    x is 2-D (M, K) bf16 (fp16 / fp32 also accepted), any M and K, any strides; not modified. x_fp8
    is e4m3 (M, K) contiguous with fp32 x_scale of logical shape (M, ceil(K / 128)), strides
    unspecified (MN-major on SM90, row-major on SM100). xt_fp8 is e4m3 (K, M) contiguous with one
    scale per 128x128 tile: fp32 xt_scale (ceil(K / 128), ceil(M / 128)), row-major.

    Per block (a partial block uses its valid elements): scale = 2^ceil(log2(max(amax, 1e-4) / 448))
    (E8M0), fp8 = e4m3(x / scale) rounded to nearest even with saturation. Both architectures return
    the same bytes and scale values.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_rowwise_blockwise_transpose_sm90 import fp8_quant_act_sm90
            return fp8_quant_act_sm90(x)
        case 10:
            from pithtrain.kernels.fp8_rowwise_blockwise_transpose_sm100 import fp8_quant_act_sm100
            return fp8_quant_act_sm100(x)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
