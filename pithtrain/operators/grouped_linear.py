from functools import partial
from typing import Optional

import deep_gemm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pithtrain.dualpipe.utils import FP8WeightCacheControl, WeightGradStore
from pithtrain.operators.deepgemm_quantize import (
    fused_blockwise_transpose_cast_to_fp8_batched,
    fused_rowwise_colwise_cast_to_fp8,
    fused_rowwise_kmajor_cast_to_fp8,
)
from pithtrain.operators.linear import ARCH_MAJOR


class GroupedLinearFunc(torch.autograd.Function):
    """
    Custom autograd Function for BF16 grouped linear layer (MoE experts).

    Forward: output      = grouped_mm(input, weight.T)      [2D-3D, jagged on M]
    Dgrad:   grad_input  = grouped_mm(grad_output, weight)  [2D-3D, jagged on M]
    Wgrad:   weight_grad = grouped_mm(grad_output.T, input) [2D-2D, jagged on K]

    The wgrad is split from dgrad so DualPipeV's zero-bubble W-phase can defer
    it via WeightGradStore, freeing the critical path during stage3_b.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
    ) -> torch.Tensor:
        output = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)
        ctx.save_for_backward(input, weight, grouped_mm_offs)
        return output

    @staticmethod
    def backward(ctx, dy):
        input, weight, offs = ctx.saved_tensors
        dgrad = F.grouped_mm(dy, weight, offs=offs)

        def wgrad_fn(dy, x, offs):
            wgrad = F.grouped_mm(dy.transpose(0, 1), x, offs=offs)
            weight.grad = wgrad if weight.grad is None else weight.grad.add_(wgrad)

        if WeightGradStore.enabled:
            WeightGradStore.put(partial(wgrad_fn, dy.detach(), input.detach(), offs.detach()))
        else:
            wgrad_fn(dy, input, offs)

        return dgrad, None, None


class GroupedLinear(nn.Module):
    """
    Grouped linear layer that partitions input data and applies a distinct
    linear transformation per group. This is useful for the MLP layers in
    the mixture-of-experts models.
    """

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))

    def forward(
        self,
        input: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: Optional[list] = None,
        ks_tensor: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input.shape[0] == 0:
            # Use a matmul instead of new_empty to preserve the autograd graph.
            # With 0 tokens the result is (0, out_features) and gradients are zero,
            # but the grad_fn must exist so that run_backward does not crash.
            return input @ self.weight[0].T
        return GroupedLinearFunc.apply(input, self.weight, grouped_mm_offs)


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


class FP8GroupedLinearFunc(torch.autograd.Function):
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


class FP8GroupedLinear(nn.Module):
    """
    FP8 grouped linear layer for MoE experts.

    Drop-in replacement for ``GroupedLinear`` using FP8 GEMM via DeepGEMM.
    Weight shape: ``(num_groups, out_features, in_features)``.
    Quantized weights are cached and reused across micro-batches within a single pipeline step.
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
        ver = FP8WeightCacheControl.version
        if self._wq_version == ver:
            return self._wq_cache
        result = fused_blockwise_transpose_cast_to_fp8_batched(self.weight)
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
            # Preserve autograd graph with 0 tokens (same pattern as GroupedLinear).
            return input @ self.weight[0].T
        quantized_weight = self._get_quantized_weight()
        return FP8GroupedLinearFunc.apply(
            input, self.weight, grouped_mm_offs, ks, ks_tensor, quantized_weight, group_indices
        )
