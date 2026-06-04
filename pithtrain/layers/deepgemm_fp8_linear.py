"""
DeepGEMM style FP8 linear layers.

Float8 E4M3 data with per-block (128-element) scales.
Blackwell (SM100+) uses E8M0 power-of-2 scales; Hopper (SM90) uses FP32 scales.
"""

from functools import partial
from typing import Tuple

import deep_gemm
import torch
import torch.nn as nn

from pithtrain.dualpipe.utils import FP8WeightCacheControl, WeightGradStore
from pithtrain.operators.deepgemm_fp8_quantize import (
    fused_blockwise_transpose_cast_to_fp8,
    fused_blockwise_transpose_cast_to_fp8_batched,
    fused_rowwise_blockwise_transpose_cast_to_fp8,
    fused_rowwise_colwise_cast_to_fp8,
    fused_rowwise_kmajor_cast_to_fp8,
    fused_rowwise_transpose_cast_to_fp8,
)

ARCH_MAJOR, _ = torch.cuda.get_device_capability()


def _m_grouped_fp8_gemm_nt(a, b, d, grouped_mm_offs, M, group_indices=None):
    """Dispatch m_grouped FP8 GEMM NT to the right API for the current GPU arch."""
    if ARCH_MAJOR >= 10:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a,
            b,
            d,
            grouped_mm_offs,
            use_psum_layout=True,
            expected_m_for_psum_layout=M,
        )
    else:
        assert group_indices is not None
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, group_indices)


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
    When ``FP8WeightCacheControl.enabled`` is True, quantized weights are cached
    and reused across micro-batches within a single pipeline step.
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
        ver = FP8WeightCacheControl._version
        if FP8WeightCacheControl.enabled and self._wq_version == ver:
            return self._wq_cache
        result = fused_blockwise_transpose_cast_to_fp8(self.weight)
        if FP8WeightCacheControl.enabled:
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


class FP8GroupLinearFunc(torch.autograd.Function):
    """
    Custom autograd Function for FP8 grouped linear layer (MoE experts).

    Forward:  output = grouped_mm(input, weight.T)  via m_grouped FP8 GEMM NT
    Dgrad:    grad_input = grouped_mm(grad_output, weight_t.T)  via m_grouped FP8 GEMM NT
    Wgrad:    weight_grad = grouped_mm(grad_output.T, input)  via k_grouped FP8 GEMM
              (Blackwell: TN/MN-Major, Hopper: NT/K-Major)
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        quantized_weight: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        group_indices: torch.Tensor | None = None,
    ):
        weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = quantized_weight
        M, K = input.shape
        num_groups, N, _ = weight.shape

        # Quantize input to FP8 (fused: single read for both forward and wgrad).
        # Produces per-token (for m_grouped forward) and per-channel (for k_grouped wgrad).
        # On Hopper the colwise output is written directly in K-major layout
        # (fused transpose), eliminating a separate kernel launch in backward.
        if ARCH_MAJOR >= 10:
            input_fp8, scale_input, input_ch_fp8, scale_input_ch = (
                fused_rowwise_colwise_cast_to_fp8(input)
            )
        else:
            input_fp8, scale_input, input_ch_fp8, scale_input_ch = fused_rowwise_kmajor_cast_to_fp8(
                input, grouped_mm_offs
            )

        assert ARCH_MAJOR >= 10 or group_indices is not None, (
            "group_indices is required on Hopper (SM90); call precompute_group_indices() once "
            "at the caller level and pass it to all grouped projections"
        )

        # Forward: m_grouped GEMM NT contiguous
        output = torch.empty((M, N), device=input.device, dtype=input.dtype)
        _m_grouped_fp8_gemm_nt(
            (input_fp8, scale_input),
            (weight_fp8, scale_weight),
            output,
            grouped_mm_offs,
            M,
            group_indices=group_indices,
        )

        ctx.save_for_backward(
            weight_t_fp8, scale_weight_t, grouped_mm_offs, input_ch_fp8, scale_input_ch, ks_tensor
        )
        ctx.weight_ref = weight
        ctx.ks = ks
        ctx.group_indices = group_indices
        ctx.M = M
        ctx.K = K
        ctx.N = N
        ctx.num_groups = num_groups

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight_t_fp8, scale_weight_t, grouped_mm_offs, input_ch_fp8, scale_input_ch, ks_tensor = (
            ctx.saved_tensors
        )
        weight = ctx.weight_ref
        ks = ctx.ks
        group_indices = ctx.group_indices
        M = ctx.M
        K, N, num_groups = ctx.K, ctx.N, ctx.num_groups

        # Quantize grad_output (fused: single read).
        # Produces both per-token (for dgrad) and per-channel (for wgrad)
        # FP8 tensors in one pass, eliminating a redundant BF16 read.
        # On Hopper the colwise output is K-major (fused transpose).
        if ARCH_MAJOR >= 10:
            grad_fp8, scale_grad, grad_ch_fp8, scale_grad_ch = fused_rowwise_colwise_cast_to_fp8(
                grad_output
            )
        else:
            grad_fp8, scale_grad, grad_ch_fp8, scale_grad_ch = fused_rowwise_kmajor_cast_to_fp8(
                grad_output, grouped_mm_offs
            )

        # Dgrad: m_grouped GEMM NT contiguous with pre-transposed weight
        grad_input = torch.empty((M, K), device=grad_output.device, dtype=grad_output.dtype)
        _m_grouped_fp8_gemm_nt(
            (grad_fp8, scale_grad),
            (weight_t_fp8, scale_weight_t),
            grad_input,
            grouped_mm_offs,
            M,
            group_indices=group_indices,
        )

        # Wgrad: k_grouped GEMM
        # Blackwell (TN, MN-Major): pass per-channel data directly.
        # Hopper (NT, K-Major): data is already in K-major from the fused quantization kernel.
        if ARCH_MAJOR >= 10:
            k_grouped_gemm = deep_gemm.k_grouped_fp8_gemm_tn_contiguous
        else:
            k_grouped_gemm = deep_gemm.k_grouped_fp8_gemm_nt_contiguous

        a_wgrad = (grad_ch_fp8, scale_grad_ch)
        b_wgrad = (input_ch_fp8, scale_input_ch)

        def grad_weight_fn(a, b, ks, ks_tensor):
            c = torch.zeros(num_groups, N, K, device=a[0].device, dtype=torch.float32)
            weight_grad = c
            k_grouped_gemm(a, b, weight_grad, ks, ks_tensor, c=c)
            weight_grad_bf16 = weight_grad.to(torch.bfloat16)
            if weight.grad is None:
                weight.grad = weight_grad_bf16
            else:
                weight.grad += weight_grad_bf16

        if WeightGradStore.enabled:
            WeightGradStore.put(
                partial(
                    grad_weight_fn,
                    (a_wgrad[0].detach(), a_wgrad[1].detach()),
                    (b_wgrad[0].detach(), b_wgrad[1].detach()),
                    ks,
                    ks_tensor.detach(),
                )
            )
        else:
            grad_weight_fn(a_wgrad, b_wgrad, ks, ks_tensor)

        return grad_input, None, None, None, None, None, None


class FP8GroupLinear(nn.Module):
    """
    FP8 grouped linear layer for MoE experts.

    Drop-in replacement for ``GroupLinear`` using FP8 GEMM via DeepGEMM.
    Weight shape: ``(num_groups, out_features, in_features)``.
    When ``FP8WeightCacheControl.enabled`` is True, quantized weights are cached
    and reused across micro-batches within a single pipeline step.
    """

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))
        self._wq_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._wq_version: int = -1

    def _get_quantized_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return fused_blockwise_transpose_cast_to_fp8_batched(self.weight)
        ver = FP8WeightCacheControl._version
        if FP8WeightCacheControl.enabled and self._wq_version == ver:
            return self._wq_cache
        result = fused_blockwise_transpose_cast_to_fp8_batched(self.weight)
        if FP8WeightCacheControl.enabled:
            self._wq_cache = result
            self._wq_version = ver
        return result

    def forward(
        self,
        input: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input.shape[0] == 0:
            # Preserve autograd graph with 0 tokens (same pattern as GroupLinear).
            return input @ self.weight[0].T
        quantized_weight = self._get_quantized_weight()
        return FP8GroupLinearFunc.apply(
            input, self.weight, grouped_mm_offs, ks, ks_tensor, quantized_weight, group_indices
        )
