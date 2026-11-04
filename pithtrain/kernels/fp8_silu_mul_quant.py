import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["fp8_silu_mul_bwd", "fp8_silu_mul_gu_bwd", "fp8_silu_mul_fwd", "fp8_silu_mul_gu_fwd"]

BLOCK = 128


def check_operands(tensors: list[torch.Tensor], grouped_mm_offs: torch.Tensor) -> None:
    for x in tensors:
        assert x.dtype == torch.bfloat16 and x.ndim == 2 and x.is_contiguous()
        assert x.shape == tensors[0].shape
    assert tensors[0].shape[0] % BLOCK == 0, f"M must be a multiple of 128, got {tensors[0].shape}"
    assert grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.is_contiguous()


def check_gate_up(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> None:
    assert gu.dtype == torch.bfloat16 and gu.ndim == 2 and gu.is_contiguous()
    assert gu.shape[1] % 2 == 0
    assert grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.is_contiguous()


def fp8_silu_mul_fwd(g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FP8 grouped-GEMM input of h = silu(g) * u without writing h: the 4-tuple (h_fp8, h_scale,
    h_ch_fp8, h_ch_scale) of fp8_quant_grouped(silu_mul_fwd(g, u), grouped_mm_offs) on SM90,
    bit for bit.

    g, u: bf16 (M, N), contiguous, same shape; rows grouped by expert. grouped_mm_offs: int32 (G,)
    contiguous cumulative group ends, each a multiple of 128, last == M. Inputs are not modified; M
    == 0 or N == 0 gives empty outputs. Callers use it where fp8_fused_experts_supported(); on SM100
    it returns the same values with row-major scales (scale strides are unspecified). Deterministic.
    """
    check_operands([g, u], grouped_mm_offs)
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_silu_mul_quant_sm90 import fp8_silu_mul_fwd_sm90
            return fp8_silu_mul_fwd_sm90(g, u, grouped_mm_offs)
        case 10:
            from pithtrain.kernels.fp8_silu_mul_quant_sm100 import fp8_silu_mul_fwd_sm100
            return fp8_silu_mul_fwd_sm100(g, u, grouped_mm_offs)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_silu_mul_bwd(dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """
    Backward of fp8_silu_mul_fwd: (dg_quant, du_quant), each the 4-tuple of fp8_quant_grouped
    of one gradient of silu_mul_bwd(dh, g, u), without writing the bf16 gradients.

    dh, g, u: bf16 (M, N), contiguous, same shape; dh the gradient of h, g / u the saved forward
    inputs. grouped_mm_offs, layout and SM100 behaviour as in fp8_silu_mul_fwd. Inputs are
    not modified. Deterministic.
    """
    check_operands([dh, g, u], grouped_mm_offs)
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_silu_mul_quant_sm90 import fp8_silu_mul_bwd_sm90
            return fp8_silu_mul_bwd_sm90(dh, g, u, grouped_mm_offs)
        case 10:
            from pithtrain.kernels.fp8_silu_mul_quant_sm100 import fp8_silu_mul_bwd_sm100
            return fp8_silu_mul_bwd_sm100(dh, g, u, grouped_mm_offs)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_silu_mul_gu_fwd(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    fp8_silu_mul_fwd(gu[:, :N].contiguous(), gu[:, N:].contiguous(), grouped_mm_offs) of a fused
    gate_up output gu: bf16 (M, 2N) contiguous, halves concatenated (not interleaved), not modified.
    Returns the 4-tuple for the (M, N) h.
    """
    check_gate_up(gu, grouped_mm_offs)
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_silu_mul_quant_sm90 import fp8_silu_mul_gu_fwd_sm90
            return fp8_silu_mul_gu_fwd_sm90(gu, grouped_mm_offs)
        case 10:
            from pithtrain.kernels.fp8_silu_mul_quant_sm100 import fp8_silu_mul_gu_fwd_sm100
            return fp8_silu_mul_gu_fwd_sm100(gu, grouped_mm_offs)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_silu_mul_gu_bwd(dh: torch.Tensor, gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward of fp8_silu_mul_gu_fwd: the 4-tuple of fp8_quant_grouped(cat([dg, du], 1),
    grouped_mm_offs) for the (M, 2N) gate_up gradient, dg, du = silu_mul_bwd(dh, g, u).

    dh: bf16 (M, N) contiguous, the gradient of h. gu: bf16 (M, 2N) contiguous, the saved gate_up
    output. Neither is modified.
    """
    check_gate_up(gu, grouped_mm_offs)
    assert dh.dtype == torch.bfloat16 and dh.is_contiguous()
    assert dh.shape == (gu.shape[0], gu.shape[1] // 2)
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_silu_mul_quant_sm90 import fp8_silu_mul_gu_bwd_sm90
            return fp8_silu_mul_gu_bwd_sm90(dh, gu, grouped_mm_offs)
        case 10:
            from pithtrain.kernels.fp8_silu_mul_quant_sm100 import fp8_silu_mul_gu_bwd_sm100
            return fp8_silu_mul_gu_bwd_sm100(dh, gu, grouped_mm_offs)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
