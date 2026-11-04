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
