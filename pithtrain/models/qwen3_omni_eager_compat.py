"""Explicit CPU eager expert compatibility helper, independent of the pipeline.

This module does not select a model backend or change a production default.
Only dense CPU BF16/FP32 tensors and unique top-k routes are supported. Compile,
EP communication, deferred weight gradients and FSDP integration are unsupported.
"""

import torch
from torch.nn import functional as F


def eager_experts_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Apply the HF eager expert graph to caller-selected routes, on CPU.

    Shapes are hidden_states [N,H], routing_weights/expert_ids [N,K],
    gate_up_proj [E,2*I,H], and down_proj [E,H,I]. Floating tensors share a
    BF16 or FP32 dtype; IDs are int64, in [0,E), and unique within each row.
    Weights are used as supplied, without normalization or router computation.

    Nonempty inputs preserve separate per-expert input gathers, slot-major
    rows, ordinary F.linear autograd and ascending-expert index_add arithmetic.
    N=0 returns [0,H] with zero VJPs for all four floating inputs; this explicit
    empty-input convention extends HF eager's otherwise disconnected output.
    """
    tensors = {
        "hidden_states": hidden_states,
        "routing_weights": routing_weights,
        "expert_ids": expert_ids,
        "gate_up_proj": gate_up_proj,
        "down_proj": down_proj,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a dense CPU tensor")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [N,H]")
    if routing_weights.ndim != 2 or expert_ids.ndim != 2:
        raise ValueError("routing_weights and expert_ids must have shape [N,K]")
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("expert projections must have shapes [E,2*I,H] and [E,H,I]")
    if hidden_states.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("floating tensors must use BF16 or FP32")
    if any(t.dtype != hidden_states.dtype for t in (routing_weights, gate_up_proj, down_proj)):
        raise TypeError("all floating tensors must have the same dtype")
    if expert_ids.dtype != torch.int64:
        raise TypeError("expert_ids must use int64")

    n, hidden = hidden_states.shape
    experts, twice_width, projected_hidden = gate_up_proj.shape
    if experts < 1 or hidden < 1 or twice_width < 2 or twice_width % 2:
        raise ValueError("E, H and I must be positive, with an even gate_up width")
    width = twice_width // 2
    if projected_hidden != hidden or down_proj.shape != (experts, hidden, width):
        raise ValueError("expert projection dimensions must match [E,2*I,H] and [E,H,I]")
    if routing_weights.shape != expert_ids.shape or expert_ids.shape[0] != n:
        raise ValueError("routing_weights and expert_ids must share shape [N,K]")
    k = expert_ids.shape[1]
    if not 1 <= k <= experts:
        raise ValueError("K must satisfy 1 <= K <= E")
    if torch.any((expert_ids < 0) | (expert_ids >= experts)):
        raise ValueError("expert_ids must be in [0,E)")
    ordered_ids = expert_ids.sort(dim=-1).values
    if torch.any(ordered_ids[:, 1:] == ordered_ids[:, :-1]):
        raise ValueError("expert_ids must be unique within each token's top-k routes")

    if n == 0:
        gate, up = F.linear(hidden_states, gate_up_proj[0]).chunk(2, dim=-1)
        projected = F.linear(F.silu(gate) * up, down_proj[0])
        return projected + routing_weights.sum(dim=-1, keepdim=True)

    result = torch.zeros_like(hidden_states)
    for expert in range(gate_up_proj.shape[0]):
        slot, token = torch.where((expert_ids == expert).transpose(0, 1))
        if token.numel() == 0:
            continue
        selected = hidden_states[token]
        gate, up = F.linear(selected, gate_up_proj[expert]).chunk(2, dim=-1)
        activated = F.silu(gate) * up
        projected = F.linear(activated, down_proj[expert])
        weighted = projected * routing_weights[token, slot, None]
        result.index_add_(0, token, weighted.to(result.dtype))
    return result


def eager_stage5_from_routes(
    moe_outs: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    residual: torch.Tensor,
    *,
    num_experts: int,
    returned_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit CPU HF-eager aggregation of caller-supplied expert outputs.

    This compatibility/reference entry point is not selected by the pipeline.
    It keeps the original router slots: IDs/weights are [N,K], outputs [M,H],
    residual [...,H] with N flattened tokens. Optional unique int64 return
    indices map each output to its original token*K+slot; missing routes make
    no local contribution. Without indices all N*K routes must be supplied.

    Inputs are dense CPU tensors, with float tensors sharing BF16/FP32 dtype.
    Experts accumulate in ascending ID, with each expert's rows slot-major,
    matching HF eager's multiply/index_add graph at this aggregation boundary.
    The outputs may originate elsewhere: native expert GEMM/input-gradient
    equivalence, EP collectives, GPU and fullgraph compilation are unqualified.
    Empty input/return sets explicitly yield zero output/weight VJPs.
    """
    values = (moe_outs, routing_weights, expert_ids, residual)
    if returned_indices is not None:
        values += (returned_indices,)
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError("all tensor arguments must be torch.Tensor values")
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("all tensor arguments must be dense CPU tensors")
    if type(num_experts) is not int or num_experts < 1:
        raise ValueError("num_experts must be a positive integer")
    if moe_outs.ndim != 2 or routing_weights.ndim != 2 or expert_ids.ndim != 2:
        raise ValueError("outputs must be [M,H] and routes/weights [N,K]")
    if residual.ndim < 2 or residual.shape[-1] < 1:
        raise ValueError("residual must have shape [...,H] with positive H")
    n, k = routing_weights.shape
    hidden = residual.shape[-1]
    if expert_ids.shape != (n, k) or not 1 <= k <= num_experts:
        raise ValueError("routes/weights must agree with 1 <= K <= num_experts")
    if moe_outs.shape[-1] != hidden or residual.numel() != n * hidden:
        raise ValueError("output and residual hidden/token dimensions differ")
    if moe_outs.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("float inputs must share BF16 or FP32 dtype")
    if routing_weights.dtype != moe_outs.dtype or residual.dtype != moe_outs.dtype:
        raise TypeError("float inputs must share BF16 or FP32 dtype")
    if expert_ids.dtype != torch.int64:
        raise TypeError("expert_ids must be int64")
    if torch.any((expert_ids < 0) | (expert_ids >= num_experts)):
        raise ValueError("expert_ids must lie within the configured expert range")
    ordered = expert_ids.sort(-1).values
    if torch.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("each token's expert IDs must be unique")
    if returned_indices is None:
        if moe_outs.shape[0] != n * k:
            raise ValueError("all routes are required without returned_indices")
        positions = torch.arange(n * k, dtype=torch.int64)
    else:
        positions = returned_indices
        if positions.dtype != torch.int64:
            raise TypeError("returned_indices must be int64")
        if positions.ndim != 1 or positions.numel() != moe_outs.shape[0]:
            raise ValueError("returned_indices must map every returned output row")
        if torch.any((positions < 0) | (positions >= n * k)):
            raise ValueError("returned_indices are outside the router slot range")
        if positions.unique().numel() != positions.numel():
            raise ValueError("returned_indices must be unique")
    result = moe_outs.new_zeros(n, hidden)
    if positions.numel() == 0:
        weights = routing_weights.reshape(-1).index_select(0, positions)
        result = result.index_add(0, positions // k, moe_outs * weights[:, None])
    else:
        slots = moe_outs.new_zeros(n * k, hidden).index_copy(0, positions, moe_outs)
        present = torch.zeros(n * k, dtype=torch.bool).index_fill(0, positions, True)
        active = expert_ids.reshape(-1).index_select(0, positions).unique(sorted=True).tolist()
        for expert in active:
            slot, token = torch.where(((expert_ids == expert) & present.view(n, k)).T)
            output = slots.view(n, k, hidden)[token, slot]
            weighted = output * routing_weights[token, slot, None]
            result.index_add_(0, token, weighted.to(result.dtype))
    return residual + result.reshape_as(residual)
