"""
FP8 routed experts as single autograd nodes, with the activation fused into the FP8 quantization
of its neighbours, and the fused expert input that feeds them.

- fp8_swiglu_experts: separate gate, up and down weights (DeepSeek-V2-Lite, Qwen3).
- fp8_swiglu_gate_up_experts: one fused (G, 2I, H) gate_up weight (Qwen3.5).
- fp8_clamped_swiglu_experts: gpt-oss's raw biased, interleaved gate_up and down parameters and
  its clamped SwiGLU.
- fp8_scatter_quant: received tokens to the quantized grouped expert input in one pass.

Use these only where fp8_fused_experts_supported() is true; elsewhere FP8 experts take the
unfused FP8GroupedLinear path. Every FP8 operand is bit-identical to that path, so the outputs
and gradients are too, except gpt-oss's gate_up bias gradient, which sums the same bf16 values in
a different fp32 order.
"""

import torch

from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels import (
    fp8_clamped_swiglu_bwd,
    fp8_clamped_swiglu_kmajor,
    fp8_clamped_swiglu_rowwise,
    fp8_grouped_gemm,
    fp8_quant_grouped,
    fp8_silu_mul_bwd,
    fp8_silu_mul_fwd,
    fp8_silu_mul_gu_bwd,
    fp8_silu_mul_gu_fwd,
    group_bias_add,
    k_grouped_fp8_gemm,
    scatter_quant,
    segment_sum,
)
from pithtrain.operators.grouped_linear import FP8GroupedLinear, fp8_grouped_wgrad

__all__ = [
    "FP8ClampedSwiGLUExpertsFunc",
    "FP8ScatterQuantFunc",
    "FP8SwiGLUExpertsFunc",
    "FP8SwiGLUGateUpExpertsFunc",
    "fp8_clamped_swiglu_experts",
    "fp8_padded_wgrad",
    "fp8_scatter_quant",
    "fp8_swiglu_experts",
    "fp8_swiglu_gate_up_experts",
    "scatter_quant_backward",
]


class FP8SwiGLUExpertsFunc(torch.autograd.Function):
    """
    down_proj(silu(gate_proj(x)) * up_proj(x)) for FP8 experts with separate gate, up and down
    weights, as one autograd node. The SwiGLU output h and its gradients are produced directly in
    FP8, never in bf16.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor,
        quantized_gate: tuple[torch.Tensor, ...],
        quantized_up: tuple[torch.Tensor, ...],
        quantized_down: tuple[torch.Tensor, ...],
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if quantized_input is None:
            quantized_input = fp8_quant_grouped(x, grouped_mm_offs)
        x_fp8, x_scale, x_kmajor, x_kmajor_scale = quantized_input
        inter, hidden = w_gate.shape[1], w_down.shape[1]
        gate = fp8_grouped_gemm(
            (x_fp8, x_scale), quantized_gate[:2], inter, grouped_mm_offs, group_indices
        )
        up = fp8_grouped_gemm(
            (x_fp8, x_scale), quantized_up[:2], inter, grouped_mm_offs, group_indices
        )
        h_fp8, h_scale, _, _ = fp8_silu_mul_fwd(gate, up, grouped_mm_offs)
        y = fp8_grouped_gemm(
            (h_fp8, h_scale), quantized_down[:2], hidden, grouped_mm_offs, group_indices
        )
        del h_fp8, h_scale
        # The K-major h for down_proj's wgrad is not saved: the backward quantizes it again from
        # gate and up, the same bytes.
        ctx.save_for_backward(
            gate, up, x_kmajor, x_kmajor_scale,
            grouped_mm_offs, ks_tensor,
            quantized_gate[2], quantized_gate[3], quantized_up[2], quantized_up[3],
            quantized_down[2], quantized_down[3],
        )  # fmt: skip
        ctx.weights = (w_gate, w_up, w_down)
        ctx.ks, ctx.group_indices = ks, group_indices
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (
            gate, up, x_kmajor, x_kmajor_scale,
            grouped_mm_offs, ks_tensor,
            w_gate_t, w_gate_scale_t, w_up_t, w_up_scale_t, w_down_t, w_down_scale_t,
        ) = ctx.saved_tensors  # fmt: skip
        w_gate, w_up, w_down = ctx.weights
        ks, group_indices = ctx.ks, ctx.group_indices
        inter, hidden = w_gate.shape[1], w_down.shape[1]
        x_kmajor_pair = (x_kmajor, x_kmajor_scale)

        # Each local is deleted once its last reader is queued, to keep the backward's peak memory
        # low; the deferred wgrads hold only the K-major operands they need.
        _, _, h_kmajor, h_kmajor_scale = fp8_silu_mul_fwd(gate, up, grouped_mm_offs)
        dy_fp8, dy_scale, dy_kmajor, dy_kmajor_scale = fp8_quant_grouped(dy, grouped_mm_offs)
        del dy
        dh = fp8_grouped_gemm(
            (dy_fp8, dy_scale), (w_down_t, w_down_scale_t), inter, grouped_mm_offs, group_indices
        )
        del dy_fp8, dy_scale
        fp8_grouped_wgrad(
            (dy_kmajor, dy_kmajor_scale),
            (h_kmajor, h_kmajor_scale),
            ks,
            ks_tensor,
            w_down,
            tuple(w_down.shape),
        )
        del dy_kmajor, dy_kmajor_scale, h_kmajor, h_kmajor_scale
        dg_quant, du_quant = fp8_silu_mul_bwd(dh, gate, up, grouped_mm_offs)
        del dh, gate, up
        dx = fp8_grouped_gemm(
            dg_quant[:2], (w_gate_t, w_gate_scale_t), hidden, grouped_mm_offs, group_indices
        )
        dg_quant = dg_quant[2:]
        dx += fp8_grouped_gemm(
            du_quant[:2], (w_up_t, w_up_scale_t), hidden, grouped_mm_offs, group_indices
        )
        du_quant = du_quant[2:]
        fp8_grouped_wgrad(dg_quant, x_kmajor_pair, ks, ks_tensor, w_gate, tuple(w_gate.shape))
        fp8_grouped_wgrad(du_quant, x_kmajor_pair, ks, ks_tensor, w_up, tuple(w_up.shape))
        return dx, None, None, None, None, None, None, None, None, None, None, None


def fp8_swiglu_experts(
    gate_proj: "FP8GroupedLinear",
    up_proj: "FP8GroupedLinear",
    down_proj: "FP8GroupedLinear",
    x: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    group_indices: torch.Tensor,
    quantized_input: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """
    down_proj(silu(gate_proj(x)) * up_proj(x)). With quantized_input (from fp8_scatter_quant) x is
    only a gradient carrier and is never read.
    """
    return FP8SwiGLUExpertsFunc.apply(
        x, gate_proj.weight, up_proj.weight, down_proj.weight, grouped_mm_offs, ks, ks_tensor,
        group_indices, gate_proj.get_quantized_weight(), up_proj.get_quantized_weight(),
        down_proj.get_quantized_weight(), quantized_input,
    )  # fmt: skip


def scatter_quant_backward(
    grad_x: torch.Tensor,
    reverse_shuffle_idxs: torch.Tensor,
    expand_idx: torch.Tensor | None,
    num_src_rows: int,
) -> torch.Tensor:
    """
    Gradient of fp8_scatter_quant's gathered input from grad_x (M, N), the gradient of the grouped
    expert input: rows gathered by reverse_shuffle_idxs, then, with expand_idx, scatter-added into
    num_src_rows rows. Padding rows of grad_x are ignored.
    """
    grad_expanded = grad_x[reverse_shuffle_idxs]
    if expand_idx is None:
        return grad_expanded
    out = grad_x.new_zeros(num_src_rows, grad_x.shape[1])
    # scatter_add_ rather than index_add_: several times faster for bf16 rows at these shapes.
    out.scatter_add_(0, expand_idx.unsqueeze(-1).expand_as(grad_expanded), grad_expanded)
    return out


class FP8ScatterQuantFunc(torch.autograd.Function):
    """
    Received tokens to the quantized grouped expert input in one pass: the dedup expansion, the
    scatter by expert and the FP8 quantization, without writing the bf16 expert input.

    Returns a zero-storage (M, N) stand-in for that bf16 input, which only carries its gradient back
    to the received tokens, then reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks and the
    quantized tuple, all as scatter_quant returns them, row placement within an expert included.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        gathered: torch.Tensor,
        expert_idxs: torch.Tensor,
        num_groups: int,
        expand_idx: torch.Tensor | None,
        group_counts: list[int] | None,
    ):
        quantized_x, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_quant(
            gathered, expert_idxs, num_groups, expand_idx, group_counts=group_counts
        )
        x = gathered.new_zeros(()).expand(quantized_x[0].shape)
        empty = reverse_shuffle_idxs.new_empty(0)
        ctx.save_for_backward(reverse_shuffle_idxs, expand_idx if expand_idx is not None else empty)
        ctx.has_expand, ctx.num_src_rows = expand_idx is not None, gathered.shape[0]
        ctx.mark_non_differentiable(reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, *quantized_x)
        return x, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks, quantized_x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor, *_):
        reverse_shuffle_idxs, expand_idx = ctx.saved_tensors
        expand_idx = expand_idx if ctx.has_expand else None
        return (
            scatter_quant_backward(grad_x, reverse_shuffle_idxs, expand_idx, ctx.num_src_rows),
            None,
            None,
            None,
            None,
        )


def fp8_scatter_quant(
    gathered: torch.Tensor,
    expert_idxs: torch.Tensor,
    num_groups: int,
    expand_idx: torch.Tensor | None = None,
    group_counts: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor, tuple[torch.Tensor, ...]]:
    """
    Returns (x, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor, quantized_x); see
    FP8ScatterQuantFunc. Hand the experts x, the gradient carrier, together with quantized_x.
    """
    x, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks, quantized_x = (
        FP8ScatterQuantFunc.apply(gathered, expert_idxs, num_groups, expand_idx, group_counts)
    )
    return x, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor, quantized_x


class FP8SwiGLUGateUpExpertsFunc(torch.autograd.Function):
    """
    down_proj(silu(g) * u) with [g | u] = gate_up_proj(x) for FP8 experts with a fused (G, 2I, H)
    gate_up weight, as one autograd node, SwiGLU fused into the FP8 quantizations.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor,
        quantized_gate_up: tuple[torch.Tensor, ...],
        quantized_down: tuple[torch.Tensor, ...],
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if quantized_input is None:
            quantized_input = fp8_quant_grouped(x, grouped_mm_offs)
        x_fp8, x_scale, x_kmajor, x_kmajor_scale = quantized_input
        two_inter, hidden = w_gate_up.shape[1], w_down.shape[1]
        gate_up = fp8_grouped_gemm(
            (x_fp8, x_scale), quantized_gate_up[:2], two_inter, grouped_mm_offs, group_indices
        )
        h_fp8, h_scale, _, _ = fp8_silu_mul_gu_fwd(gate_up, grouped_mm_offs)
        y = fp8_grouped_gemm(
            (h_fp8, h_scale), quantized_down[:2], hidden, grouped_mm_offs, group_indices
        )
        # The K-major h for down_proj's wgrad is not saved: the backward quantizes it again from
        # gate_up, the same bytes.
        ctx.save_for_backward(
            gate_up, x_kmajor, x_kmajor_scale, grouped_mm_offs, ks_tensor,
            quantized_gate_up[2], quantized_gate_up[3], quantized_down[2], quantized_down[3],
        )  # fmt: skip
        ctx.weights = (w_gate_up, w_down)
        ctx.ks, ctx.group_indices = ks, group_indices
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (
            gate_up, x_kmajor, x_kmajor_scale, grouped_mm_offs, ks_tensor,
            w_gate_up_t, w_gate_up_scale_t, w_down_t, w_down_scale_t,
        ) = ctx.saved_tensors  # fmt: skip
        w_gate_up, w_down = ctx.weights
        ks, group_indices = ctx.ks, ctx.group_indices
        inter, hidden = w_down.shape[2], w_down.shape[1]

        dy_fp8, dy_scale, dy_kmajor, dy_kmajor_scale = fp8_quant_grouped(dy, grouped_mm_offs)
        dh = fp8_grouped_gemm(
            (dy_fp8, dy_scale), (w_down_t, w_down_scale_t), inter, grouped_mm_offs, group_indices
        )
        _, _, h_kmajor, h_kmajor_scale = fp8_silu_mul_gu_fwd(gate_up, grouped_mm_offs)
        fp8_grouped_wgrad(
            (dy_kmajor, dy_kmajor_scale),
            (h_kmajor, h_kmajor_scale),
            ks,
            ks_tensor,
            w_down,
            tuple(w_down.shape),
        )
        dgu_quant = fp8_silu_mul_gu_bwd(dh, gate_up, grouped_mm_offs)
        del dh
        dx = fp8_grouped_gemm(
            dgu_quant[:2], (w_gate_up_t, w_gate_up_scale_t), hidden, grouped_mm_offs, group_indices
        )
        fp8_grouped_wgrad(
            dgu_quant[2:],
            (x_kmajor, x_kmajor_scale),
            ks,
            ks_tensor,
            w_gate_up,
            tuple(w_gate_up.shape),
        )
        return dx, None, None, None, None, None, None, None, None, None


def fp8_swiglu_gate_up_experts(
    gate_up_proj: torch.nn.Module,
    down_proj: torch.nn.Module,
    x: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    group_indices: torch.Tensor,
    quantized_input: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """
    down_proj(silu(g) * u) with [g | u] = gate_up_proj(x), both FP8GroupedLinear. With
    quantized_input (from fp8_scatter_quant) x is only a gradient carrier and is never read.
    """
    return FP8SwiGLUGateUpExpertsFunc.apply(
        x, gate_up_proj.weight, down_proj.weight, grouped_mm_offs, ks, ks_tensor, group_indices,
        gate_up_proj.get_quantized_weight(), down_proj.get_quantized_weight(), quantized_input,
    )  # fmt: skip


def fp8_padded_wgrad(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    ks: list,
    ks_tensor: torch.Tensor,
    weight: torch.Tensor,
    main_grad: torch.Tensor | None = None,
) -> None:
    """
    weight.grad += a.T @ b per expert for weight (G, N, K), or into main_grad in fp32 when it has
    storage. a and b are per-channel FP8 (data, scale) pairs with at least N and K channels (the
    fused ops' padded=True layout); channels past N and K are ignored. Deferred to DualPipeV's W
    phase when WeightGradStore is enabled.
    """
    G, N, K = weight.shape

    def accumulate_wgrad(a, b, ks, ks_tensor):
        if main_grad is not None and main_grad.untyped_storage().size() > 0:
            k_grouped_fp8_gemm(a, b, main_grad, ks, ks_tensor)
            return
        grad = torch.zeros((G, N, K), device=a[0].device, dtype=torch.float32)
        k_grouped_fp8_gemm(a, b, grad, ks, ks_tensor)
        grad = grad.to(torch.bfloat16)
        if weight.grad is None:
            weight.grad = grad.contiguous()
        else:
            weight.grad += grad

    if WeightGradStore.enabled:
        a = (a[0].detach(), a[1].detach())
        b = (b[0].detach(), b[1].detach())
        WeightGradStore.put(lambda: accumulate_wgrad(a, b, ks, ks_tensor.detach()))
    else:
        accumulate_wgrad(a, b, ks, ks_tensor)


class FP8ClampedSwiGLUExpertsFunc(torch.autograd.Function):
    """
    down(clamped_swiglu(gate_up(x) + b_gate_up)) + b_down for gpt-oss FP8 experts, as one autograd
    node. With quantized_input (from fp8_scatter_quant), x is only a gradient carrier.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        w_gate_up: torch.Tensor,
        b_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor,
        quantized_gate_up: tuple[torch.Tensor, ...],
        quantized_down: tuple[torch.Tensor, ...],
        alpha: float,
        limit: float,
        quantized_input: tuple[torch.Tensor, ...] | None = None,
        main_grads: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
    ) -> torch.Tensor:
        if quantized_input is None:
            quantized_input = fp8_quant_grouped(x, grouped_mm_offs)
        x_fp8, x_scale, x_kmajor, x_kmajor_scale = quantized_input
        two_inter, hidden = w_gate_up.shape[1], w_down.shape[1]
        gate_up = fp8_grouped_gemm(
            (x_fp8, x_scale), quantized_gate_up[:2], two_inter, grouped_mm_offs, group_indices
        )
        # Only the rowwise h: the backward quantizes its K-major copy again from gate_up.
        h_fp8, h_scale = fp8_clamped_swiglu_rowwise(
            gate_up, b_gate_up, grouped_mm_offs, alpha, limit
        )
        y = fp8_grouped_gemm(
            (h_fp8, h_scale), quantized_down[:2], hidden, grouped_mm_offs, group_indices
        )
        group_bias_add(y, b_down, group_indices)
        ctx.save_for_backward(
            b_gate_up, x_kmajor, x_kmajor_scale, grouped_mm_offs, ks_tensor,
            quantized_gate_up[2], quantized_gate_up[3], quantized_down[2], quantized_down[3],
            gate_up,
        )  # fmt: skip
        ctx.weights = (w_gate_up, w_down)
        ctx.main_grads = main_grads
        ctx.ks, ctx.group_indices = ks, group_indices
        ctx.alpha, ctx.limit = alpha, limit
        ctx.b_down_dtype = b_down.dtype
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (
            b_gate_up, x_kmajor, x_kmajor_scale, grouped_mm_offs, ks_tensor,
            w_gate_up_t, w_gate_up_scale_t, w_down_t, w_down_scale_t, gate_up,
        ) = ctx.saved_tensors  # fmt: skip
        w_gate_up, w_down = ctx.weights
        ks, group_indices = ctx.ks, ctx.group_indices
        inter, hidden = w_down.shape[2], w_down.shape[1]
        dy = dy.contiguous()
        d_b_down = segment_sum(dy, grouped_mm_offs, ctx.b_down_dtype)
        dy_fp8, dy_scale, dy_kmajor, dy_kmajor_scale = fp8_quant_grouped(dy, grouped_mm_offs)
        dh = fp8_grouped_gemm(
            (dy_fp8, dy_scale), (w_down_t, w_down_scale_t), inter, grouped_mm_offs, group_indices
        )
        # Padded so the k-grouped wgrad takes it without a copy.
        h_kmajor, h_kmajor_scale = fp8_clamped_swiglu_kmajor(
            gate_up, b_gate_up, grouped_mm_offs, ctx.alpha, ctx.limit, padded=True
        )
        main_grad_gate_up, main_grad_down = ctx.main_grads
        fp8_padded_wgrad(
            (dy_kmajor, dy_kmajor_scale),
            (h_kmajor, h_kmajor_scale),
            ks,
            ks_tensor,
            w_down,
            main_grad_down,
        )
        del h_kmajor, h_kmajor_scale
        del dy_fp8, dy_scale, dy_kmajor, dy_kmajor_scale
        dgu_quant, d_b_gate_up = fp8_clamped_swiglu_bwd(
            dh, gate_up, b_gate_up, grouped_mm_offs, ctx.alpha, ctx.limit, padded=True
        )
        del dh
        dx = fp8_grouped_gemm(
            dgu_quant[:2], (w_gate_up_t, w_gate_up_scale_t), hidden, grouped_mm_offs, group_indices
        )
        fp8_padded_wgrad(
            dgu_quant[2:], (x_kmajor, x_kmajor_scale), ks, ks_tensor, w_gate_up, main_grad_gate_up
        )
        return dx, None, d_b_gate_up, None, d_b_down, *([None] * 10)


def fp8_clamped_swiglu_experts(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    b_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    group_indices: torch.Tensor,
    quantized_gate_up: tuple[torch.Tensor, ...],
    quantized_down: tuple[torch.Tensor, ...],
    alpha: float,
    limit: float,
    quantized_input: tuple[torch.Tensor, ...] | None = None,
    main_grads: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
) -> torch.Tensor:
    """
    down(clamped_swiglu(gate_up(x) + b_gate_up)) + b_down. quantized_gate_up and quantized_down are
    the (fp8, scale, fp8_t, scale_t) quantized weights.
    """
    return FP8ClampedSwiGLUExpertsFunc.apply(
        x, w_gate_up, b_gate_up, w_down, b_down, grouped_mm_offs, ks, ks_tensor, group_indices,
        quantized_gate_up, quantized_down, alpha, limit, quantized_input, main_grads,
    )  # fmt: skip
