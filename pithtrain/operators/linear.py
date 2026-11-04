"""
FP8Linear runs its GEMMs in FP8 E4M3 with one E8M0 scale per 128-element block.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.lowering import fallback_handler, register_lowering

from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels import (
    act_scale_like,
    fp8_gemm_bwd,
    fp8_gemm_fwd,
    fp8_gemm_nt,
    fp8_quant_act,
    fp8_quant_weight,
)

__all__ = ["BF16Linear", "BF16LinearFunc", "FP8Linear"]


@torch.library.custom_op("pithtrain::fp8_linear_fwd", mutates_args=())
def fp8_linear_fwd(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_weight: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    output = input_2d @ weight.T for an (M, K) input_2d. Returns output (M, N) in the input dtype
    and, saved for the wgrad, input_t_fp8 (K, M) with its scale_input_t. weight is taken only to
    give it a gradient; the GEMM reads weight_fp8.
    """
    return fp8_gemm_fwd(input_2d, weight_fp8, scale_weight)


@fp8_linear_fwd.register_fake
def _(input_2d, weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t):
    (M, K), N = input_2d.shape, weight.shape[0]
    output = torch.empty((M, N), dtype=input_2d.dtype, device=input_2d.device)
    input_t_fp8 = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=input_2d.device)
    size = ((K + 127) // 128, (M + 127) // 128)
    scale_input_t = torch.empty(size, dtype=torch.float32, device=input_2d.device)
    return output, input_t_fp8, scale_input_t


@torch.library.custom_op("pithtrain::fp8_linear_bwd", mutates_args=())
def fp8_linear_bwd(
    grad_output_2d: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fp8_gemm_bwd(grad_output_2d, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, K)


@fp8_linear_bwd.register_fake
def _(grad_output_2d, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, K):
    M, N = grad_output_2d.shape
    grad_input = torch.empty((M, K), dtype=grad_output_2d.dtype, device=grad_output_2d.device)
    weight_grad = torch.empty((N, K), dtype=grad_output_2d.dtype, device=grad_output_2d.device)
    return grad_input, weight_grad


def fp8_linear_setup_context(ctx, inputs, output):
    input_2d, _, _, _, weight_t_fp8, scale_weight_t = inputs
    _, input_t_fp8, scale_input_t = output
    ctx.save_for_backward(weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t)
    ctx.K = input_2d.shape[1]


def fp8_linear_backward(ctx, grad_output, grad_input_t_fp8, grad_scale_input_t):
    weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t = ctx.saved_tensors
    grad_input, weight_grad = fp8_linear_bwd(
        grad_output, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, ctx.K
    )
    return grad_input, weight_grad, None, None, None, None


fp8_linear_fwd.register_autograd(fp8_linear_backward, setup_context=fp8_linear_setup_context)


@torch.library.custom_op("pithtrain::fp8_quantize_activation", mutates_args=())
def fp8_quantize_activation(
    input_2d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The input quantization of fp8_linear_fwd on its own, so projections that share an input
    quantize it once. Returns (input_fp8, scale_input, input_t_fp8, scale_input_t).
    """
    return fp8_quant_act(input_2d)


@fp8_quantize_activation.register_fake
def _(input_2d):
    M, K = input_2d.shape
    device = input_2d.device
    input_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device)
    scale_input = act_scale_like(M, (K + 127) // 128, device)
    input_t_fp8 = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=device)
    size = ((K + 127) // 128, (M + 127) // 128)
    scale_input_t = torch.empty(size, dtype=torch.float32, device=device)
    return input_fp8, scale_input, input_t_fp8, scale_input_t


@torch.library.custom_op("pithtrain::fp8_linear_fwd_prequantized", mutates_args=())
def fp8_linear_fwd_prequantized(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_weight: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_fp8: torch.Tensor,
    scale_input: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
) -> torch.Tensor:
    """
    The output of fp8_linear_fwd for an input already quantized by fp8_quantize_activation.
    input_2d and weight are taken only to give them gradients; the GEMM reads the FP8 operands.
    """
    M, N = input_fp8.shape[0], weight_fp8.shape[0]
    return fp8_gemm_nt(
        input_fp8, scale_input, weight_fp8, scale_weight, M, N, input_2d.device, input_2d.dtype
    )


@fp8_linear_fwd_prequantized.register_fake
def _(input_2d, weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t, *quantized_input):
    M, N = input_2d.shape[0], weight.shape[0]
    return torch.empty((M, N), dtype=input_2d.dtype, device=input_2d.device)


def fp8_linear_prequantized_setup_context(ctx, inputs, output):
    input_2d, _, _, _, weight_t_fp8, scale_weight_t, _, _, input_t_fp8, scale_input_t = inputs
    ctx.save_for_backward(weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t)
    ctx.K = input_2d.shape[1]


def fp8_linear_prequantized_backward(ctx, grad_output):
    weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t = ctx.saved_tensors
    grad_input, weight_grad = fp8_linear_bwd(
        grad_output, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, ctx.K
    )
    return grad_input, weight_grad, None, None, None, None, None, None, None, None


fp8_linear_fwd_prequantized.register_autograd(
    fp8_linear_prequantized_backward, setup_context=fp8_linear_prequantized_setup_context
)


class FP8Linear(nn.Linear):
    """
    Outside torch.compile, the bf16 weight is quantized once per step and reused by every
    micro-batch of it; with fp8_param_gather set, the weight arrives from the
    all-gather already quantized.
    """

    fp8_param_gather: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantized_weight_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None  # fmt: skip
        self.quantized_weight_version: int = -1

    def get_quantized_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fp8_param_gather:
            return self.weight_fp8, self.weight_scale, self.weight_t_fp8, self.weight_scale_t
        if torch.compiler.is_compiling():
            return fp8_quant_weight(self.weight)
        ver = FP8WeightCacheControl.version
        if self.quantized_weight_version == ver:
            return self.quantized_weight_cache
        result = fp8_quant_weight(self.weight)
        self.quantized_weight_cache = result
        self.quantized_weight_version = ver
        return result

    @staticmethod
    def quantize_input(x: torch.Tensor) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Quantize an activation once for several FP8Linear layers that read it, to pass to each as
        quantized_input. The bytes are the same as each layer quantizing it itself.
        """
        if x.numel() == 0:
            return None
        # Not differentiable: each layer still takes x itself, which carries the gradient.
        return fp8_quantize_activation(x.detach().flatten(0, -2))

    def forward(
        self, input: torch.Tensor, quantized_input: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        if input.numel() == 0:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        quantized_weight = self.get_quantized_weight()
        weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = quantized_weight
        input_2d = input.flatten(0, -2)
        if quantized_input is not None:
            output_2d = fp8_linear_fwd_prequantized(
                input_2d,
                self.weight,
                weight_fp8,
                scale_weight,
                weight_t_fp8,
                scale_weight_t,
                *quantized_input,
            )
        else:
            output_2d, _, _ = fp8_linear_fwd(
                input_2d, self.weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t
            )
        output = output_2d.view(*input.shape[:-1], self.weight.shape[0])
        if self.bias is not None:
            output = output + self.bias
        return output


# Inductor has no single-GEMM lowering for addmm with out_dtype, so the fused GEMM is its own op.
wgrad_accumulate_library = torch.library.Library("pithtrain", "FRAGMENT")
wgrad_accumulate_library.define("wgrad_accumulate_(Tensor(a!) main_grad, Tensor a, Tensor b) -> ()")


def wgrad_accumulate_cuda(main_grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    torch.addmm(main_grad, a, b, out_dtype=torch.float32, out=main_grad)


def wgrad_accumulate_meta(main_grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    return None


wgrad_accumulate_library.impl("wgrad_accumulate_", wgrad_accumulate_cuda, "CUDA")
wgrad_accumulate_library.impl("wgrad_accumulate_", wgrad_accumulate_meta, "Meta")
addmm_dtype_out_fallback = fallback_handler(torch.ops.aten.addmm.dtype_out, add_to_fallback_set=False)  # fmt: skip


@register_lowering(torch.ops.pithtrain.wgrad_accumulate_.default, type_promotion_kind=None)
def lower_wgrad_accumulate(main_grad: Any, a: Any, b: Any) -> None:
    addmm_dtype_out_fallback(main_grad, a, b, torch.float32, beta=1, alpha=1, out=main_grad)


class BF16LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        main_grad: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        # Kept off the saved tensors: other micro-batches update main_grad in place meanwhile.
        ctx.main_grad = main_grad
        return F.linear(input, weight)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        input, weight = ctx.saved_tensors
        main_grad = ctx.main_grad
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        grad_input = (grad_output_2d @ weight).view(*input.shape)
        torch.ops.pithtrain.wgrad_accumulate_(main_grad, grad_output_2d.t(), input_2d)
        return grad_input, None, None


class BF16Linear(nn.Linear):
    """
    With main_grad set, the weight gradient accumulates into it in fp32 instead of a .grad.
    """

    main_grad: Optional[torch.Tensor] = None

    @staticmethod
    def quantize_input(x: torch.Tensor) -> None:
        return None

    def forward(self, input: torch.Tensor, quantized_input: None = None) -> torch.Tensor:
        if self.main_grad is None or not torch.is_grad_enabled():
            return F.linear(input, self.weight, self.bias)
        output = BF16LinearFunc.apply(input, self.weight, self.main_grad)
        return output if self.bias is None else output + self.bias
