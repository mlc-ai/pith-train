"""Gated DeltaNet chunked delta rule as a ``torch.library`` custom op.

FLA's ``chunk_gated_delta_rule`` is ``@torch.compiler.disable``d (a hard Dynamo
graph break). Wrapping its low-level fwd/bwd kernels FA4-style (``custom_op`` +
``register_fake``, see :mod:`pithtrain.operators.flash_attn_v4`) makes it an
opaque, shape-known graph node so the linear-attention region stays
fullgraph-compilable.

Specialized to the qwen3.5 usage: in-kernel q/k L2-norm, post-sigmoid ``beta``,
log-space ``g``, no cu_seqlens / initial_state / gate-in-kernel.
"""

from typing import Tuple

import fla.ops.common.chunk_o
import torch
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)

# mypy: ignore-errors

# FLA hard-raises in the gated delta-rule backward on Hopper + Triton>=3.4
# (fla-org#640). That miscompile is specific to a 64-wide K tile; head_k_dim=128
# tiles K at 128 and is correct (grads match the torch reference to ~1e-5), so
# neutralize the over-broad guard rather than depend on the unrelated tilelang
# backend (which also fails to import under Python 3.14 here).
fla.ops.common.chunk_o.TRITON_ABOVE_3_4_0 = False

_CHUNK = 64  # FLA chunk size (BT); trailing dim of the saved WY matrix A


@torch.library.custom_op("pithtrain::gated_delta_rule_fwd", mutates_args=())
def _gdr_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    scale = q.shape[-1] ** -0.5
    ql, q_rstd = l2norm_fwd(q)
    kl, k_rstd = l2norm_fwd(k)
    g_out, o, A, _final_state, _initial_state, _g_input = chunk_gated_delta_rule_fwd(
        q=ql,
        k=kl,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        cp_context=None,
        chunk_indices=None,
        state_v_first=False,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
    )
    return o.to(q.dtype), ql, kl, q_rstd, k_rstd, g_out, A


@_gdr_fwd.register_fake
def _(q, k, v, g, beta):
    b, s, hv, vd = v.shape
    o = torch.empty((b, s, hv, vd), dtype=q.dtype, device=q.device)
    ql = torch.empty_like(q)
    kl = torch.empty_like(k)
    q_rstd = torch.empty((b, s, q.shape[2]), dtype=torch.float32, device=q.device)
    k_rstd = torch.empty((b, s, k.shape[2]), dtype=torch.float32, device=q.device)
    g_out = torch.empty_like(g, dtype=torch.float32)
    A = torch.empty((b, s, hv, _CHUNK), dtype=q.dtype, device=q.device)
    return o, ql, kl, q_rstd, k_rstd, g_out, A


@torch.library.custom_op("pithtrain::gated_delta_rule_bwd", mutates_args=())
def _gdr_bwd(
    ql: torch.Tensor,
    kl: torch.Tensor,
    v: torch.Tensor,
    g_out: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    do: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = ql.shape[-1] ** -0.5
    dq, dk, dv, db, dg, _dh0, _dA_log, _ddt_bias = chunk_gated_delta_rule_bwd(
        q=ql,
        k=kl,
        v=v,
        g=g_out,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=None,
        do=do.contiguous(),
        dht=None,
        cu_seqlens=None,
        cp_context=None,
        chunk_indices=None,
        state_v_first=False,
        use_gate_in_kernel=False,
        g_input=None,
        A_log=None,
        dt_bias=None,
    )
    dq = l2norm_bwd(ql, q_rstd, dq)
    dk = l2norm_bwd(kl, k_rstd, dk)
    return dq.to(ql.dtype), dk.to(kl.dtype), dv.to(v.dtype), dg.to(g_out.dtype), db.to(beta.dtype)


@_gdr_bwd.register_fake
def _(ql, kl, v, g_out, beta, A, q_rstd, k_rstd, do):
    return (
        torch.empty_like(ql),
        torch.empty_like(kl),
        torch.empty_like(v),
        torch.empty_like(g_out),
        torch.empty_like(beta),
    )


def _gdr_setup_context(ctx, inputs, output):
    o, ql, kl, q_rstd, k_rstd, g_out, A = output
    _q, _k, v, _g, beta = inputs
    ctx.save_for_backward(ql, kl, v, g_out, beta, A, q_rstd, k_rstd)


def _gdr_backward(ctx, do, dql, dkl, dq_rstd, dk_rstd, dg_out, dA):
    # Only ``do`` carries gradient; the saved-intermediate outputs do not.
    ql, kl, v, g_out, beta, A, q_rstd, k_rstd = ctx.saved_tensors
    dq, dk, dv, dg, dbeta = _gdr_bwd(ql, kl, v, g_out, beta, A, q_rstd, k_rstd, do)
    return dq, dk, dv, dg, dbeta


_gdr_fwd.register_autograd(_gdr_backward, setup_context=_gdr_setup_context)


def gated_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Chunked gated delta rule (BSHD q/k/v, ``[b, s, hv]`` g/beta) → BSHD output.

    ``g`` is the log-space decay and ``beta`` is post-sigmoid; q/k are L2-normed
    inside the kernel. Compile-safe drop-in for FLA's ``chunk_gated_delta_rule``.
    """
    return _gdr_fwd(q, k, v, g, beta)[0]
