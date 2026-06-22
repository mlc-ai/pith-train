"""
Gated DeltaNet chunked delta rule as a torch.library custom op.

FLA's chunk_gated_delta_rule is @torch.compiler.disable'd (a hard Dynamo graph break).
Wrapping its low-level fwd/bwd in a custom_op + register_fake makes it an opaque,
shape-known graph node so the linear-attention region stays fullgraph-compilable.
"""

from typing import Tuple

import fla.ops.common.chunk_o
import torch
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd

# fmt: off
# mypy: ignore-errors

# Disable FLA's issue#640 guard: the BK=64 backward miscompile doesn't apply at head_k_dim=128.
fla.ops.common.chunk_o.TRITON_ABOVE_3_4_0 = False

@torch.library.custom_op("pithtrain::gated_delta_rule_fwd", mutates_args=())
def _gdr_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    g_out, o, A, _, _, _ = chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=q.shape[-1] ** -0.5, initial_state=None, output_final_state=False, cu_seqlens=None, cp_context=None, chunk_indices=None, state_v_first=False, use_gate_in_kernel=False, A_log=None, dt_bias=None)
    return o.to(q.dtype), g_out, A

@_gdr_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
    b, s, hv, vd = v.shape
    o = torch.empty((b, s, hv, vd), dtype=q.dtype, device=q.device)
    A = torch.empty((b, s, hv, 64), dtype=q.dtype, device=q.device) # 64 = FLA chunk size (BT)
    return o, torch.empty_like(g, dtype=torch.float32), A

@torch.library.custom_op("pithtrain::gated_delta_rule_bwd", mutates_args=())
def _gdr_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, g_out, beta, A, do = (t.contiguous() for t in (q, k, v, g_out, beta, A, do))
    dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(q=q, k=k, v=v, g=g_out, beta=beta, A=A, scale=q.shape[-1] ** -0.5, initial_state=None, do=do, dht=None, cu_seqlens=None, cp_context=None, chunk_indices=None, state_v_first=False, use_gate_in_kernel=False, g_input=None, A_log=None, dt_bias=None)
    return dq.to(q.dtype).contiguous(), dk.to(k.dtype).contiguous(), dv.to(v.dtype).contiguous(), dg.to(g_out.dtype).contiguous(), db.to(beta.dtype).contiguous()

@_gdr_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor):
    cf = torch.contiguous_format # match the .contiguous() grads; empty_like would inherit input strides
    return torch.empty_like(q, memory_format=cf), torch.empty_like(k, memory_format=cf), torch.empty_like(v, memory_format=cf), torch.empty_like(g_out, memory_format=cf), torch.empty_like(beta, memory_format=cf)

def _gdr_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:
    q, k, v, _g, beta = inputs
    _o, g_out, A = output
    ctx.save_for_backward(q, k, v, g_out, beta, A)

def _gdr_backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor, *_unused) -> Tuple:
    q, k, v, g_out, beta, A = ctx.saved_tensors
    return _gdr_bwd(q, k, v, g_out, beta, A, do)

_gdr_fwd.register_autograd(_gdr_backward, setup_context=_gdr_setup_context)

def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    o, *_ = _gdr_fwd(q, k, v, g, beta)
    return o
