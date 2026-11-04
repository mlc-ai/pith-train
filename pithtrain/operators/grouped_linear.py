from functools import partial
from typing import List, Optional

import deep_gemm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels import (
    fp8_grouped_gemm,
    fp8_quant_grouped,
    fp8_quant_weight_grouped,
    k_grouped_fp8_gemm,
)

__all__ = [
    "FP8GroupedLinear",
    "FP8GroupedLinearFunc",
    "GroupedLinear",
    "GroupedLinearFunc",
    "fp8_grouped_wgrad",
]


class GroupedLinearFunc(torch.autograd.Function):
    """
    The weight gradient accumulates into main_grad in fp32 when one is given; ks and ks_tensor
    are then required.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        main_grad: Optional[torch.Tensor] = None,
        ks: Optional[List[int]] = None,
        ks_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)
        ctx.save_for_backward(input, weight, grouped_mm_offs)
        ctx.main_grad, ctx.ks, ctx.ks_tensor = main_grad, ks, ks_tensor
        return output

    @staticmethod
    def backward(ctx, dy):
        input, weight, offs = ctx.saved_tensors
        dgrad = F.grouped_mm(dy, weight, offs=offs)
        main_grad, ks, ks_tensor = ctx.main_grad, ctx.ks, ctx.ks_tensor

        def wgrad_fn(dy: torch.Tensor, x: torch.Tensor, offs: torch.Tensor) -> None:
            if main_grad is not None:
                dy, x = dy.contiguous(), x.contiguous()
                deep_gemm.k_grouped_bf16_gemm_tn_contiguous(dy, x, main_grad, ks, ks_tensor, c=main_grad)  # fmt: skip
                return
            wgrad = F.grouped_mm(dy.transpose(0, 1), x, offs=offs)
            weight.grad = wgrad if weight.grad is None else weight.grad.add_(wgrad)

        if WeightGradStore.enabled:
            WeightGradStore.put(partial(wgrad_fn, dy.detach(), input.detach(), offs.detach()))
        else:
            wgrad_fn(dy, input, offs)

        return dgrad, None, None, None, None, None


class GroupedLinear(nn.Module):
    """
    The input rows are sorted by group, with group boundaries given by grouped_mm_offs.
    """

    main_grad: Optional[torch.Tensor] = None

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
        ks: Optional[List[int]] = None,
        ks_tensor: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
        quantized_input: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        if input.shape[0] == 0:
            # A matmul rather than new_empty, so the (0, out_features) result has a grad_fn and
            # the backward does not fail.
            weight = self.weight[0] if self.main_grad is None else self.weight[0].detach()
            return input @ weight.T
        if self.main_grad is not None and torch.is_grad_enabled():
            assert ks is not None and ks_tensor is not None
            return GroupedLinearFunc.apply(input, self.weight, grouped_mm_offs, self.main_grad, ks, ks_tensor)  # fmt: skip
        return GroupedLinearFunc.apply(input, self.weight, grouped_mm_offs)


def fp8_grouped_wgrad(
    grad_ch: tuple[torch.Tensor, torch.Tensor],
    input_ch: tuple[torch.Tensor, torch.Tensor],
    ks: list,
    ks_tensor: torch.Tensor,
    weight: torch.Tensor,
    shape: tuple[int, int, int],
) -> None:
    """
    grad_ch and input_ch are the per-channel FP8 (data, scale) pairs that fp8_quant_grouped gives
    for grad_output and input; shape is the (G, N, K) weight shape.
    """
    main_grad = getattr(weight, "main_grad", None)

    def grad_weight_fn(grad_ch, input_ch, ks, ks_tensor):
        if main_grad is not None:
            k_grouped_fp8_gemm(grad_ch, input_ch, main_grad, ks, ks_tensor)
            return
        weight_grad = torch.zeros(shape, device=grad_ch[0].device, dtype=torch.float32)
        k_grouped_fp8_gemm(grad_ch, input_ch, weight_grad, ks, ks_tensor)
        weight_grad_bf16 = weight_grad.to(torch.bfloat16)
        if weight.grad is None:
            weight.grad = weight_grad_bf16
        else:
            weight.grad += weight_grad_bf16

    if WeightGradStore.enabled:
        grad_ch = (grad_ch[0].detach(), grad_ch[1].detach())
        input_ch = (input_ch[0].detach(), input_ch[1].detach())
        WeightGradStore.put(partial(grad_weight_fn, grad_ch, input_ch, ks, ks_tensor.detach()))
    else:
        grad_weight_fn(grad_ch, input_ch, ks, ks_tensor)


class FP8GroupedLinearFunc(torch.autograd.Function):
    """
    quantized_weight is what FP8GroupedLinear.get_quantized_weight returns. quantized_input, when
    given, is fp8_quant_grouped(input) computed once and shared by several projections.
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
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ):
        weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = quantized_weight
        K = input.shape[1]
        num_groups, N, _ = weight.shape

        if quantized_input is None:
            quantized_input = fp8_quant_grouped(input, grouped_mm_offs)
        input_fp8, scale_input, input_ch_fp8, scale_input_ch = quantized_input

        output = fp8_grouped_gemm(
            (input_fp8, scale_input), (weight_fp8, scale_weight), N, grouped_mm_offs, group_indices
        )

        ctx.save_for_backward(
            weight_t_fp8, scale_weight_t, grouped_mm_offs, input_ch_fp8, scale_input_ch, ks_tensor
        )
        ctx.weight_ref = weight
        ctx.ks = ks
        ctx.group_indices = group_indices
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
        K, N, num_groups = ctx.K, ctx.N, ctx.num_groups

        grad_fp8, scale_grad, grad_ch_fp8, scale_grad_ch = fp8_quant_grouped(
            grad_output, grouped_mm_offs
        )

        grad_input = fp8_grouped_gemm(
            (grad_fp8, scale_grad),
            (weight_t_fp8, scale_weight_t),
            K,
            grouped_mm_offs,
            group_indices,
        )

        a_wgrad = (grad_ch_fp8, scale_grad_ch)
        b_wgrad = (input_ch_fp8, scale_input_ch)

        fp8_grouped_wgrad(a_wgrad, b_wgrad, ks, ks_tensor, weight, (num_groups, N, K))

        return grad_input, None, None, None, None, None, None, None


class FP8GroupedLinear(nn.Module):
    """
    Drop-in GroupedLinear replacement with FP8 GEMMs. The bf16 weight is quantized once per
    pipeline step and reused by every micro-batch of it; with fp8_param_gather set, the weight
    arrives already quantized.
    """

    fp8_param_gather: bool = False
    main_grad: Optional[torch.Tensor] = None
    """
    fp32 gradient buffer set by wgrad_accumulation.
    """

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))
        self.quantized_weight_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None  # fmt: skip
        self.quantized_weight_version: int = -1

    def get_quantized_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Every FP8 expert path reads the weight through here, so the weight the backward sees
        # gets its main_grad here (the unsharded parameter outlives the step).
        if self.main_grad is not None:
            self.weight.main_grad = self.main_grad
        if self.fp8_param_gather:
            return self.weight_fp8, self.weight_scale, self.weight_t_fp8, self.weight_scale_t
        if torch.compiler.is_compiling():
            return fp8_quant_weight_grouped(self.weight)
        ver = FP8WeightCacheControl.version
        if self.quantized_weight_version == ver:
            return self.quantized_weight_cache
        result = fp8_quant_weight_grouped(self.weight)
        self.quantized_weight_cache = result
        self.quantized_weight_version = ver
        return result

    def forward(
        self,
        input: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor | None = None,
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if input.shape[0] == 0:
            # A matmul rather than new_empty, as in GroupedLinear, to keep a grad_fn.
            weight = self.weight[0] if self.main_grad is None else self.weight[0].detach()
            return input @ weight.T
        quantized_weight = self.get_quantized_weight()
        return FP8GroupedLinearFunc.apply(
            input,
            self.weight,
            grouped_mm_offs,
            ks,
            ks_tensor,
            quantized_weight,
            group_indices,
            quantized_input,
        )
