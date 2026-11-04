import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["fp8_quant_grad"]


def fp8_quant_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise 1x128 FP8 of x and of x.T from one read of x, returning (x_fp8, x_scale, xt_fp8,
    xt_scale): the grad_output operands of the dense FP8 linear's dgrad and wgrad.

    x is 2-D (M, N) bf16 (fp32 also accepted), any M and N, any strides; not modified. x_fp8 is e4m3
    (M, N) contiguous with fp32 x_scale of logical shape (M, ceil(N / 128)). xt_fp8 is e4m3 (N, M)
    contiguous with fp32 xt_scale of logical shape (N, ceil(M / 128)). Scale strides are
    unspecified: both MN-major on SM90, row-major on SM100.

    Per 128-element block (a partial block uses its valid elements): scale = 2^ceil(log2(max(amax,
    1e-4) / 448)) (E8M0), fp8 = e4m3(x / scale) rounded to nearest even with saturation. Both
    architectures return the same bytes and scale values.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_rowwise_transpose_sm90 import fp8_quant_grad_sm90
            return fp8_quant_grad_sm90(x)
        case 10:
            from pithtrain.kernels.fp8_rowwise_transpose_sm100 import fp8_quant_grad_sm100
            return fp8_quant_grad_sm100(x)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
