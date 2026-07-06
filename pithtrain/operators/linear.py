"""
FP8 linear layer (DeepGEMM) and the shared FP8 GEMM recipe.

``FP8Linear`` is a drop-in ``nn.Linear`` replacement backed by DeepGEMM's Float8 E4M3 GEMM with
per-block (128-element) scales -- E8M0 power-of-2 scales on Blackwell (SM100+), FP32 scales on
Hopper (SM90). ``fp8_act_weight_gemm`` / ``fp8_dgrad_wgrad`` own the forward/backward GEMM
convention and are shared with the MLA pass-latent ring attention.
"""

from typing import Tuple

import deep_gemm
import torch
import torch.nn as nn

from pithtrain.dualpipe.utils import FP8WeightCacheControl
from pithtrain.operators.deepgemm_quantize import (
    fused_blockwise_transpose_cast_to_fp8,
    fused_rowwise_blockwise_transpose_cast_to_fp8,
    fused_rowwise_transpose_cast_to_fp8,
)

ARCH_MAJOR, _ = torch.cuda.get_device_capability()


def _fp8_gemm_nt(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    m: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``out[m, n] = a @ b.T`` via the DeepGEMM fp8 NT GEMM. The single place that owns the
    allocate-then-``fp8_fp4_gemm_nt`` convention shared by the FP8 linear layer and the MLA
    pass-latent ring decompress."""
    out = torch.empty((m, n), device=device, dtype=dtype)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, a_scale), (b_fp8, b_scale), out)
    return out


def fp8_act_weight_gemm(
    input_2d: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 forward GEMM ``y = x @ W.T``: rowwise-quantize the activation, run the NT GEMM, and
    return ``(output, input_t_fp8, scale_input_t)`` -- the transposed activation is handed back
    so a later wgrad can reuse it. Shared by ``_fp8_linear_fwd`` and the MLA ring so the recipe
    has one source of truth."""
    m, n = input_2d.shape[0], weight_fp8.shape[0]
    input_fp8, scale_input, input_t_fp8, scale_input_t = (
        fused_rowwise_blockwise_transpose_cast_to_fp8(input_2d)
    )
    output = _fp8_gemm_nt(
        input_fp8, scale_input, weight_fp8, scale_weight, m, n, input_2d.device, input_2d.dtype
    )
    return output, input_t_fp8, scale_input_t


def fp8_dgrad_wgrad(
    grad_2d: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP8 backward GEMMs: ``grad_input = grad @ W`` (via the transposed weight) and
    ``weight_grad = grad.T @ x`` (via the transposed activation), both returned in ``grad_2d``'s
    dtype. Shared by ``_fp8_linear_bwd`` and the MLA ring so dgrad/wgrad have one source of
    truth; the caller supplies ``input_t_fp8`` (saved from the forward, or recomputed)."""
    m, n = grad_2d.shape
    grad_fp8, scale_grad, grad_t_fp8, scale_grad_t = fused_rowwise_transpose_cast_to_fp8(grad_2d)
    grad_input = _fp8_gemm_nt(
        grad_fp8, scale_grad, weight_t_fp8, scale_weight_t, m, k, grad_2d.device, grad_2d.dtype
    )
    weight_grad = _fp8_gemm_nt(
        grad_t_fp8, scale_grad_t, input_t_fp8, scale_input_t, n, k, grad_2d.device, grad_2d.dtype
    )
    return grad_input, weight_grad


@torch.library.custom_op("pithtrain::fp8_linear_fwd", mutates_args=())
def _fp8_linear_fwd(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_weight: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FP8 linear forward: output = input @ weight.T via fp8_fp4_gemm_nt.

    Returns
    ----
    output : torch.Tensor
        (M, N) GEMM result in input dtype.
    input_t_fp8 : torch.Tensor
        (K, M) transposed FP8 input, saved for wgrad.
    scale_input_t : torch.Tensor
        Block scales for input_t_fp8, saved for wgrad.
    """
    return fp8_act_weight_gemm(input_2d, weight_fp8, scale_weight)


@_fp8_linear_fwd.register_fake
def _(input_2d, weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t):
    (M, K), N = input_2d.shape, weight.shape[0]
    output = torch.empty((M, N), dtype=input_2d.dtype, device=input_2d.device)
    input_t_fp8 = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=input_2d.device)
    size = ((K + 127) // 128, (M + 127) // 128)
    scale_input_t = torch.empty(size, dtype=torch.float32, device=input_2d.device)
    return output, input_t_fp8, scale_input_t


@torch.library.custom_op("pithtrain::fp8_linear_bwd", mutates_args=())
def _fp8_linear_bwd(
    grad_output_2d: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 linear backward: computes dgrad and wgrad.

    Returns
    ----
    grad_input : torch.Tensor
        (M, K) input gradient.
    weight_grad : torch.Tensor
        (N, K) weight gradient.
    """
    return fp8_dgrad_wgrad(
        grad_output_2d, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, K
    )


@_fp8_linear_bwd.register_fake
def _(grad_output_2d, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, K):
    M, N = grad_output_2d.shape
    grad_input = torch.empty((M, K), dtype=grad_output_2d.dtype, device=grad_output_2d.device)
    weight_grad = torch.empty((N, K), dtype=grad_output_2d.dtype, device=grad_output_2d.device)
    return grad_input, weight_grad


def _fp8_linear_setup_context(ctx, inputs, output):
    input_2d, _, _, _, weight_t_fp8, scale_weight_t = inputs
    _, input_t_fp8, scale_input_t = output
    ctx.save_for_backward(weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t)
    ctx.K = input_2d.shape[1]


def _fp8_linear_backward(ctx, grad_output, grad_input_t_fp8, grad_scale_input_t):
    weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t = ctx.saved_tensors
    grad_input, weight_grad = _fp8_linear_bwd(
        grad_output, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, ctx.K
    )
    return grad_input, weight_grad, None, None, None, None


_fp8_linear_fwd.register_autograd(_fp8_linear_backward, setup_context=_fp8_linear_setup_context)


class FP8Linear(nn.Linear):
    """
    Drop-in replacement for ``nn.Linear`` using FP8 GEMM via DeepGEMM.

    Weights are stored in BF16 and quantized to MXFP8 on-the-fly each forward pass.
    Quantized weights are cached and reused across micro-batches within a single pipeline step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wq_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._wq_version: int = -1

    def _get_quantized_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return fused_blockwise_transpose_cast_to_fp8(self.weight)
        ver = FP8WeightCacheControl.version
        if self._wq_version == ver:
            return self._wq_cache
        result = fused_blockwise_transpose_cast_to_fp8(self.weight)
        self._wq_cache = result
        self._wq_version = ver
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.numel() == 0:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        quantized_weight = self._get_quantized_weight()
        weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = quantized_weight
        input_2d = input.flatten(0, -2)
        output_2d, _, _ = _fp8_linear_fwd(
            input_2d, self.weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t
        )
        output = output_2d.view(*input.shape[:-1], self.weight.shape[0])
        if self.bias is not None:
            output = output + self.bias
        return output
