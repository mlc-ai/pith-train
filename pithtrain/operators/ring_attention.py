"""
Zigzag ring attention for context parallelism.

Causal flash attention sharded across cp_size ranks. Two design choices: zigzag chunking for
load balance, and async ring P2P for compute/comm overlap.

The global sequence is split into 2 * cp_size equal chunks of size block = S / (2 * cp_size).
Rank r holds two of them, chunk r (front block) and chunk 2 * cp_size - r - 1 (back block).
The chunk-to-rank assignment is mirrored; for cp_size = 4 it looks like

    chunk:  0  1  2  3  4  5  6  7
    rank:   0  1  2  3  3  2  1  0

Lower-indexed ranks pair one very-early chunk with one very-late chunk; higher-indexed ranks
pair two near-middle chunks. The early chunk has few tokens before it (light causal work),
the late one has many (heavy). They cancel, so every rank ends up doing the same amount of
attention work.

Q stays on its home rank; K/V rotate one hop per step in the +1 direction. Each step posts
its next-step batch_isend_irecv before launching its flash kernel, so the transfer overlaps
with compute. The backward runs two rings concurrently: K/V rotates as in the forward (we
re-derive partial outputs rather than save cp_size copies), and partial dK/dV rotates in the
same direction so every contribution reaches its originating rank after cp_size hops.

Q, K, V, and the returned output are all in the zigzag local layout; the caller (data loader
and RoPE) does the permutation.

The non-obvious part is which flash call covers each step. At step s, rank r holds K/V
originating from kv = (r - s) mod cp_size; the rotated K has its own front block (chunk kv)
and back block (chunk 2 * cp_size - kv - 1). Comparing global chunk positions, exactly one
of three pictures holds:

    step == 0     (kv == r)   flash(q,            k,            v,            causal=True)
    1 <= s <= r   (kv <  r)   flash(q,            k[:, :block], v[:, :block], causal=False)
    s >  r        (kv >  r)   flash(q[:, block:], k,            v,            causal=False)

  step 0: K is local. The four (q_part, k_part) sub-blocks line up with a 2*block local
          causal mask, so one causal call handles everything.

  kv < r: K came from a lower-indexed rank, whose chunks live at extreme positions. Only its
          front block survives the global causal mask; both halves of Q attend to it fully.

  kv > r: K came from a higher-indexed rank, whose chunks live at central positions. Only
          Q's back block attends, sees the full rotated K, and only the back-block positions
          of out / lse are updated.

Every step costs the same: one causal pass on length 2*block, or one non-causal pass on a
2*block-by-block rectangle.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from torch.distributed import (
    P2POp,
    ProcessGroup,
    Work,
    batch_isend_irecv,
    get_global_rank,
    get_rank,
    get_world_size,
    irecv,
    isend,
)
from torch.distributed.distributed_c10d import _resolve_process_group

# FP8 (deep-gemm) shared GEMM recipe + the activation quantizer for the in-ring kv_b
# decompression; reached only when the caller passes a quantized kv_b weight
# (i.e. fp8_training="deep-gemm").
from pithtrain.layers.deepgemm_fp8_linear import fp8_act_weight_gemm, fp8_dgrad_wgrad
from pithtrain.operators.deepgemm_fp8_quantize import (
    fused_rowwise_blockwise_transpose_cast_to_fp8,
)


def post_ring_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: ProcessGroup,
    dst: int,
    src: int,
    k_recv: Optional[torch.Tensor] = None,
    v_recv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Work]]:
    """
    Async (K, V) ring hop. Pre-allocated recv buffers let the backward recycle just-sent
    dK/dV buffers as the next iteration's recv slots.
    """
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("ring P2P requires contiguous send buffers")
    if k_recv is None:
        k_recv = torch.empty_like(k)
    if v_recv is None:
        v_recv = torch.empty_like(v)
    ops = []
    ops.append(P2POp(isend, k, dst, group=cp_group))
    ops.append(P2POp(isend, v, dst, group=cp_group))
    ops.append(P2POp(irecv, k_recv, src, group=cp_group))
    ops.append(P2POp(irecv, v_recv, src, group=cp_group))
    work = batch_isend_irecv(ops)
    return k_recv, v_recv, work


def wait_ring(work: List[Work]) -> None:
    for req in work:
        req.wait()


@torch.compile(fullgraph=True)
def combine_partial(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    start: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Online-softmax merge of a partial flash output into the running fp32 accumulator. When
    start > 0 only positions [start:] are updated; the first call must have start == 0.
    """
    partial_out = partial_out.to(torch.float32)
    partial_lse = partial_lse.transpose(-2, -1).unsqueeze(-1)
    if out is None:
        if start != 0:
            raise ValueError("first combine_partial call must update the full sequence")
        return partial_out, partial_lse
    if start == 0:
        weight = torch.sigmoid(partial_lse - lse)
        new_out = out + weight * (partial_out - out)
        new_lse = lse + F.softplus(partial_lse - lse)
        return new_out, new_lse
    cur_out = out[:, start:]
    cur_lse = lse[:, start:]
    weight = torch.sigmoid(partial_lse - cur_lse)
    out[:, start:] = cur_out + weight * (partial_out - cur_out)
    lse[:, start:] = cur_lse + F.softplus(partial_lse - cur_lse)
    return out, lse


def zigzag_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    cp_group: ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cp_rank, cp_size = get_rank(cp_group), get_world_size(cp_group)
    dst = get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    src = get_global_rank(cp_group, (cp_rank - 1) % cp_size)
    block = q.shape[1] // 2
    q_back = q[:, block:]

    out: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None
    next_k: Optional[torch.Tensor] = None
    next_v: Optional[torch.Tensor] = None
    kv_work: Optional[List[Work]] = None

    for step in range(cp_size):
        if step + 1 < cp_size:
            next_k, next_v, kv_work = post_ring_kv(k, v, cp_group, dst, src)
        if step == 0:
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q, k, v, softmax_scale=sm_scale, causal=True, return_lse=True
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse)
        elif step <= cp_rank:
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q,
                k[:, :block],
                v[:, :block],
                softmax_scale=sm_scale,
                causal=False,
                return_lse=True,
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse)
        else:
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q_back, k, v, softmax_scale=sm_scale, causal=False, return_lse=True
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse, start=block)
        if step + 1 < cp_size:
            wait_ring(kv_work)
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(-1).transpose(1, 2).contiguous()
    return out, lse


def zigzag_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    cp_group: ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_rank, cp_size = get_rank(cp_group), get_world_size(cp_group)
    dst = get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    src = get_global_rank(cp_group, (cp_rank - 1) % cp_size)
    block = q.shape[1] // 2

    dout_back = dout[:, block:].contiguous()
    q_back = q[:, block:].contiguous()
    out_back = out[:, block:].contiguous()
    lse_back = lse[:, :, block:].contiguous()

    dq: Optional[torch.Tensor] = None
    dk: Optional[torch.Tensor] = None
    dv: Optional[torch.Tensor] = None
    next_k: Optional[torch.Tensor] = None
    next_v: Optional[torch.Tensor] = None
    kv_work: Optional[List[Work]] = None
    incoming_dk: Optional[torch.Tensor] = None
    incoming_dv: Optional[torch.Tensor] = None
    grad_recv_slot_k: Optional[torch.Tensor] = None
    grad_recv_slot_v: Optional[torch.Tensor] = None
    grad_work: Optional[List[Work]] = None

    for step in range(cp_size):
        if step + 1 < cp_size:
            next_k, next_v, kv_work = post_ring_kv(k, v, cp_group, dst, src)
        if step == 0:
            dq_step, dk_step, dv_step = _flash_attn_bwd(
                q, k, v, out, dout, lse, softmax_scale=sm_scale, causal=True
            )
            dq = dq_step.to(torch.float32)
            dk = dk_step.to(torch.float32)
            dv = dv_step.to(torch.float32)
        else:
            if step <= cp_rank:
                dq_step, dk_step, dv_step = _flash_attn_bwd(
                    q,
                    k[:, :block],
                    v[:, :block],
                    out,
                    dout,
                    lse,
                    softmax_scale=sm_scale,
                    causal=False,
                )
                dq += dq_step
            else:
                dq_step, dk_step, dv_step = _flash_attn_bwd(
                    q_back,
                    k,
                    v,
                    out_back,
                    dout_back,
                    lse_back,
                    softmax_scale=sm_scale,
                    causal=False,
                )
                dq[:, block:] += dq_step
            # Adopt the previous hop's dK/dV as the working accumulator.
            # The buffers we shipped one hop ago are free to recycle.
            wait_ring(grad_work)
            grad_recv_slot_k, grad_recv_slot_v = dk, dv
            dk, dv = incoming_dk, incoming_dv
            if step <= cp_rank:
                dk[:, :block] += dk_step
                dv[:, :block] += dv_step
            else:
                dk += dk_step
                dv += dv_step
        if step + 1 < cp_size:
            wait_ring(kv_work)
            k, v = next_k, next_v
        incoming_dk, incoming_dv, grad_work = post_ring_kv(
            dk, dv, cp_group, dst, src, grad_recv_slot_k, grad_recv_slot_v
        )

    wait_ring(grad_work)
    return dq.to(q.dtype), incoming_dk.to(q.dtype), incoming_dv.to(q.dtype)


# ---------------------------------------------------------------------------
# Opaque custom-op wrappers.
#
# The ring forward/backward are wrapped as torch.library.custom_op so torch.compile sees
# the whole ring -- P2P collectives, flash kernels, online-softmax merge -- as one black
# box, exactly like the FA4 kernels in flash_attn_v4.py. This lets the decoder layer keep
# `_forward_attn_compute` under @torch.compile(fullgraph=True) even under context
# parallelism (Dynamo fuses the surrounding LN/RoPE/residual and emits an opaque call for
# the ring) instead of falling back to eager. The collectives never enter the Inductor
# graph; the op body runs eagerly at runtime.
#
# A ProcessGroup is not a legal custom-op argument, so the public funcs pass the group's
# registered name (`cp_group.group_name`) and the op body resolves it via
# `_resolve_process_group` -- the same name<->group registry the functional collectives use.
# ---------------------------------------------------------------------------


@torch.library.custom_op("pithtrain::zigzag_ring_fwd", mutates_args=())
def _zigzag_ring_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    group_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("ring attention requires contiguous k and v")
    if q.shape[1] % 2:
        raise ValueError(f"zigzag layout needs even local seq len, got {q.shape[1]}")
    return zigzag_forward(q, k, v, sm_scale, _resolve_process_group(group_name))


@_zigzag_ring_fwd.register_fake
def _(q, k, v, sm_scale, group_name):
    b, s, hq, _ = q.shape
    out = torch.empty((b, s, hq, v.shape[-1]), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, hq, s), dtype=torch.float32, device=q.device)
    return out, lse


@torch.library.custom_op("pithtrain::zigzag_ring_bwd", mutates_args=())
def _zigzag_ring_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    group_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return zigzag_backward(dout, q, k, v, out, lse, sm_scale, _resolve_process_group(group_name))


@_zigzag_ring_bwd.register_fake
def _(dout, q, k, v, out, lse, sm_scale, group_name):
    # dk/dv take q.dtype to match zigzag_backward's `.to(q.dtype)` cast, not k/v's dtype
    # (identical today since q/k/v share a dtype, but the meta must track the real contract).
    return (
        torch.empty_like(q),
        torch.empty(k.shape, dtype=q.dtype, device=q.device),
        torch.empty(v.shape, dtype=q.dtype, device=q.device),
    )


def _zigzag_setup_context(ctx, inputs, output):
    q, k, v, sm_scale, group_name = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.sm_scale = sm_scale
    ctx.group_name = group_name


def _zigzag_backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _zigzag_ring_bwd(grad_out, q, k, v, out, lse, ctx.sm_scale, ctx.group_name)
    return dq, dk, dv, None, None


_zigzag_ring_fwd.register_autograd(_zigzag_backward, setup_context=_zigzag_setup_context)


def ring_attention_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    cp_group: ProcessGroup,
) -> torch.Tensor:
    """
    Causal zigzag ring attention across context-parallel ranks.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape [batch, S_local, num_q_heads, head_dim] in zigzag local
        layout, where S_local = S / cp_size for a global sequence of length S.
    k : torch.Tensor
        Key tensor of shape [batch, S_local, num_kv_heads, head_dim] in the same zigzag
        layout as q. Must be contiguous; rotated around the ring during forward and
        re-rotated during backward.
    v : torch.Tensor
        Value tensor of shape [batch, S_local, num_kv_heads, head_dim_v] in the same
        zigzag layout as q. Must be contiguous.
    sm_scale : float
        Softmax scale, typically head_dim ** -0.5.
    cp_group : torch.distributed.ProcessGroup
        Context-parallel process group. Must contain at least two ranks; the single-rank
        case is handled upstream by skipping the ring entirely.

    Returns
    -------
    torch.Tensor
        Attention output of shape [batch, S_local, num_q_heads, head_dim_v] in q.dtype,
        returned in the same zigzag local layout as q.
    """
    out, _ = _zigzag_ring_fwd(q, k, v, sm_scale, cp_group.group_name)
    return out


# ---------------------------------------------------------------------------
# MLA (Multi-head Latent Attention) zigzag ring attention -- "pass the latent".
#
# Standard ring attention (above) would require decompressing the MLA latent
# into full per-head K/V and rotating that around the ring:
#     num_heads * q_head_dim (key) + num_heads * v_head_dim (value)
# elements per token. Instead we rotate the *compressed latent* itself --
# normed_kv (kv_lora_rank) plus the shared rope key k_pe (qk_rope_head_dim) --
# and decompress locally on each rank via the kv_b projection. For
# DeepSeek-V2-Lite this cuts the per-token ring payload from 5120 to 576
# elements (~8.9x), in both the forward K/V ring and the backward gradient ring.
#
# kv_a_layernorm and RoPE are applied by the caller (both are per-token,
# position-only ops), so the rotated latent already carries them; RoPE's
# relative-position property makes pre-rotation correct. The decompression is a
# bf16 matmul with the (FSDP-gathered) kv_b weight. Its weight gradient is
# accumulated locally and summed across CP ranks by FSDP's reduce; only the
# latent *activation* gradient rings back to each token's origin rank.
#
# The three-case flash schedule and the online-softmax merge are identical to
# the standard ring -- the only addition is a local decompression before each
# flash call (and its backprop on the way back).
# ---------------------------------------------------------------------------


def _mla_decompress(
    normed_kv: torch.Tensor,
    kv_b_weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    *,
    use_fp8: bool = False,
    weight_fp8: Optional[torch.Tensor] = None,
    scale_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompress a latent block into per-head (k_nope, value) via kv_b: y = x @ Wt.

    When ``use_fp8`` the matmul uses the DeepGEMM FP8 path (mirroring FP8Linear): the latent
    block is rowwise-quantized to FP8 here (it differs every ring hop), and the kv_b weight
    is pre-quantized once per micro-batch by the caller (``weight_fp8``/``scale_weight``).
    """
    b, n, _ = normed_kv.shape
    if use_fp8:
        x2d = normed_kv.reshape(b * n, normed_kv.shape[-1])
        kv2d, _, _ = fp8_act_weight_gemm(x2d, weight_fp8, scale_weight)
        kv = kv2d.view(b, n, num_heads, qk_nope_head_dim + v_head_dim)
    else:
        kv = F.linear(normed_kv, kv_b_weight).view(b, n, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope, value = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
    # k_nope is consumed only by `torch.cat([k_nope, k_pe.expand(...)], dim=-1)`, which
    # materialises a fresh contiguous output regardless of input strides -- the .contiguous()
    # would be a redundant copy. value goes straight into _flash_attn_{fwd,bwd}, which require
    # contiguous K/V on the FA4 cute path, so the asymmetry is intentional.
    return k_nope, value.contiguous()


def mla_zigzag_forward(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    normed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    kv_b_weight: torch.Tensor,
    sm_scale: float,
    qk_nope_head_dim: int,
    v_head_dim: int,
    cp_group: ProcessGroup,
    *,
    use_fp8: bool = False,
    weight_fp8: Optional[torch.Tensor] = None,
    scale_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cp_rank, cp_size = get_rank(cp_group), get_world_size(cp_group)
    dst = get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    src = get_global_rank(cp_group, (cp_rank - 1) % cp_size)
    num_heads = q_nope.shape[2]
    block = q_nope.shape[1] // 2
    _fp8_kw = dict(use_fp8=use_fp8, weight_fp8=weight_fp8, scale_weight=scale_weight)

    q = torch.cat([q_nope, q_pe], dim=-1)
    # q_back is a strided view (kept non-contiguous in forward to match the standard zigzag
    # ring at line 144 -- FA4 forward tolerates strided q). Backward explicitly contiguous-ifies
    # the same slice (line 462) because FA4 backward has tighter strider requirements on q.
    q_back = q[:, block:]

    out: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None
    next_kv: Optional[torch.Tensor] = None
    next_pe: Optional[torch.Tensor] = None
    kv_work: Optional[List[Work]] = None

    for step in range(cp_size):
        if step + 1 < cp_size:
            next_kv, next_pe, kv_work = post_ring_kv(normed_kv, k_pe, cp_group, dst, src)
        if step == 0:
            k_nope, value = _mla_decompress(
                normed_kv, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **_fp8_kw
            )
            key = torch.cat([k_nope, k_pe.expand(-1, -1, num_heads, -1)], dim=-1)
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q, key, value, softmax_scale=sm_scale, causal=True, return_lse=True
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse)
        elif step <= cp_rank:
            k_nope, value = _mla_decompress(
                normed_kv[:, :block],
                kv_b_weight,
                num_heads,
                qk_nope_head_dim,
                v_head_dim,
                **_fp8_kw,
            )
            key = torch.cat([k_nope, k_pe[:, :block].expand(-1, -1, num_heads, -1)], dim=-1)
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q, key, value, softmax_scale=sm_scale, causal=False, return_lse=True
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse)
        else:
            k_nope, value = _mla_decompress(
                normed_kv, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **_fp8_kw
            )
            key = torch.cat([k_nope, k_pe.expand(-1, -1, num_heads, -1)], dim=-1)
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q_back, key, value, softmax_scale=sm_scale, causal=False, return_lse=True
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse, start=block)
        if step + 1 < cp_size:
            wait_ring(kv_work)
            normed_kv, k_pe = next_kv, next_pe

    out = out.to(q.dtype)
    lse = lse.squeeze(-1).transpose(1, 2).contiguous()
    return out, lse


def mla_zigzag_backward(
    dout: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    normed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    kv_b_weight: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    qk_nope_head_dim: int,
    v_head_dim: int,
    cp_group: ProcessGroup,
    *,
    use_fp8: bool = False,
    weight_fp8: Optional[torch.Tensor] = None,
    scale_weight: Optional[torch.Tensor] = None,
    weight_t_fp8: Optional[torch.Tensor] = None,
    scale_weight_t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_rank, cp_size = get_rank(cp_group), get_world_size(cp_group)
    dst = get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    src = get_global_rank(cp_group, (cp_rank - 1) % cp_size)
    b = q_nope.shape[0]
    num_heads = q_nope.shape[2]
    block = q_nope.shape[1] // 2
    rope_dim = k_pe.shape[-1]
    lora_rank = normed_kv.shape[-1]
    out_features = kv_b_weight.shape[0]
    weight_f = None if use_fp8 else kv_b_weight.to(torch.float32)
    _fp8_kw = dict(use_fp8=use_fp8, weight_fp8=weight_fp8, scale_weight=scale_weight)

    q = torch.cat([q_nope, q_pe], dim=-1)
    dout = dout.contiguous()
    q_back = q[:, block:].contiguous()
    dout_back = dout[:, block:].contiguous()
    out_back = out[:, block:].contiguous()
    lse_back = lse[:, :, block:].contiguous()

    dq: Optional[torch.Tensor] = None
    dW = torch.zeros((out_features, lora_rank), device=kv_b_weight.device, dtype=torch.float32)
    d_nkv: Optional[torch.Tensor] = None
    d_kpe: Optional[torch.Tensor] = None
    next_kv: Optional[torch.Tensor] = None
    next_pe: Optional[torch.Tensor] = None
    kv_work: Optional[List[Work]] = None
    incoming_dnkv: Optional[torch.Tensor] = None
    incoming_dkpe: Optional[torch.Tensor] = None
    grad_slot_nkv: Optional[torch.Tensor] = None
    grad_slot_kpe: Optional[torch.Tensor] = None
    grad_work: Optional[List[Work]] = None

    for step in range(cp_size):
        if step + 1 < cp_size:
            next_kv, next_pe, kv_work = post_ring_kv(normed_kv, k_pe, cp_group, dst, src)

        # Recompute the decompression for the resident block and run flash backward.
        if step == 0:
            nkv_blk = normed_kv
            k_nope, value = _mla_decompress(
                normed_kv, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **_fp8_kw
            )
            key = torch.cat([k_nope, k_pe.expand(-1, -1, num_heads, -1)], dim=-1)
            dq_step, dkey_step, dvalue_step = _flash_attn_bwd(
                q, key, value, out, dout, lse, softmax_scale=sm_scale, causal=True
            )
            front = False
        elif step <= cp_rank:
            nkv_blk = normed_kv[:, :block]
            k_nope, value = _mla_decompress(
                nkv_blk, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **_fp8_kw
            )
            key = torch.cat([k_nope, k_pe[:, :block].expand(-1, -1, num_heads, -1)], dim=-1)
            dq_step, dkey_step, dvalue_step = _flash_attn_bwd(
                q, key, value, out, dout, lse, softmax_scale=sm_scale, causal=False
            )
            front = True
        else:
            nkv_blk = normed_kv
            k_nope, value = _mla_decompress(
                normed_kv, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **_fp8_kw
            )
            key = torch.cat([k_nope, k_pe.expand(-1, -1, num_heads, -1)], dim=-1)
            dq_step, dkey_step, dvalue_step = _flash_attn_bwd(
                q_back,
                key,
                value,
                out_back,
                dout_back,
                lse_back,
                softmax_scale=sm_scale,
                causal=False,
            )
            front = False

        # Backprop the decompression: (dk_nope, dvalue) -> d(normed_kv) + kv_b weight grad.
        dk_nope, dk_pe = torch.split(dkey_step, [qk_nope_head_dim, rope_dim], dim=-1)
        n = dk_nope.shape[1]
        if use_fp8:
            # FP8 dgrad/wgrad via the shared recipe. dy = [dk_nope | dvalue] (M, out_features).
            # The wgrad needs this hop's latent block transposed; we re-quantize it here rather
            # than stash it from the forward (recompute is the deliberate memory/compute trade).
            dy2d = (
                torch.cat([dk_nope, dvalue_step], dim=-1)
                .reshape(b * n, out_features)
                .to(normed_kv.dtype)
            )
            _, _, x_t_fp8, scale_x_t = fused_rowwise_blockwise_transpose_cast_to_fp8(
                nkv_blk.reshape(b * n, lora_rank)
            )
            d_nkv_2d, dW_blk = fp8_dgrad_wgrad(
                dy2d, weight_t_fp8, scale_weight_t, x_t_fp8, scale_x_t, lora_rank
            )
            d_nkv_blk = d_nkv_2d.view(b, n, lora_rank).to(torch.float32)
            dW += dW_blk.to(torch.float32)
        else:
            d_kv = (
                torch.cat([dk_nope, dvalue_step], dim=-1)
                .reshape(b * n, out_features)
                .to(torch.float32)
            )
            d_nkv_blk = (d_kv @ weight_f).view(b, n, lora_rank)
            dW += d_kv.t() @ nkv_blk.reshape(b * n, lora_rank).to(torch.float32)
        d_kpe_blk = dk_pe.sum(dim=2, keepdim=True).to(torch.float32).contiguous()

        # dq accumulates locally (queries never leave their home rank).
        if step == 0:
            dq = dq_step.to(torch.float32)
        elif step <= cp_rank:
            dq += dq_step
        else:
            dq[:, block:] += dq_step

        # Latent gradient ring (mirrors the standard dk/dv ring, on the 576-d latent).
        if step == 0:
            # `d_nkv` aliases `d_nkv_blk` (the fp32 matmul output). Safe as a send buffer
            # for `post_ring_kv` below because subsequent iterations rebind d_nkv_blk to a
            # fresh matmul result -- the step-0 buffer is never mutated before step-1's
            # wait_ring(grad_work). If a future refactor adds in-place ops on d_nkv_blk,
            # replace these with .clone() to break the alias.
            d_nkv = d_nkv_blk
            d_kpe = d_kpe_blk
        else:
            wait_ring(grad_work)
            grad_slot_nkv, grad_slot_kpe = d_nkv, d_kpe
            d_nkv, d_kpe = incoming_dnkv, incoming_dkpe
            if front:
                d_nkv[:, :block] += d_nkv_blk
                d_kpe[:, :block] += d_kpe_blk
            else:
                d_nkv += d_nkv_blk
                d_kpe += d_kpe_blk

        if step + 1 < cp_size:
            wait_ring(kv_work)
            normed_kv, k_pe = next_kv, next_pe

        incoming_dnkv, incoming_dkpe, grad_work = post_ring_kv(
            d_nkv, d_kpe, cp_group, dst, src, grad_slot_nkv, grad_slot_kpe
        )

    wait_ring(grad_work)
    dq_nope, dq_pe = torch.split(dq, [qk_nope_head_dim, rope_dim], dim=-1)
    dtype = q_nope.dtype
    return (
        dq_nope.contiguous().to(dtype),
        dq_pe.contiguous().to(dtype),
        incoming_dnkv.to(dtype),
        incoming_dkpe.to(dtype),
        dW,
    )


@torch.library.custom_op("pithtrain::mla_zigzag_ring_fwd", mutates_args=())
def _mla_zigzag_ring_fwd(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    normed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    kv_b_weight: torch.Tensor,
    sm_scale: float,
    qk_nope_head_dim: int,
    v_head_dim: int,
    group_name: str,
    use_fp8: bool,
    weight_fp8: Optional[torch.Tensor],
    scale_weight: Optional[torch.Tensor],
    weight_t_fp8: Optional[torch.Tensor],
    scale_weight_t: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (normed_kv.is_contiguous() and k_pe.is_contiguous()):
        raise ValueError("MLA ring attention requires contiguous normed_kv and k_pe")
    if q_nope.shape[1] % 2:
        raise ValueError(f"zigzag layout needs even local seq len, got {q_nope.shape[1]}")
    return mla_zigzag_forward(
        q_nope,
        q_pe,
        normed_kv,
        k_pe,
        kv_b_weight,
        sm_scale,
        qk_nope_head_dim,
        v_head_dim,
        _resolve_process_group(group_name),
        use_fp8=use_fp8,
        weight_fp8=weight_fp8,
        scale_weight=scale_weight,
    )


@_mla_zigzag_ring_fwd.register_fake
def _(
    q_nope,
    q_pe,
    normed_kv,
    k_pe,
    kv_b_weight,
    sm_scale,
    qk_nope_head_dim,
    v_head_dim,
    group_name,
    use_fp8,
    weight_fp8,
    scale_weight,
    weight_t_fp8,
    scale_weight_t,
):
    b, s, h, _ = q_nope.shape
    out = torch.empty((b, s, h, v_head_dim), dtype=q_nope.dtype, device=q_nope.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q_nope.device)
    return out, lse


@torch.library.custom_op("pithtrain::mla_zigzag_ring_bwd", mutates_args=())
def _mla_zigzag_ring_bwd(
    dout: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    normed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    kv_b_weight: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    qk_nope_head_dim: int,
    v_head_dim: int,
    group_name: str,
    use_fp8: bool,
    weight_fp8: Optional[torch.Tensor],
    scale_weight: Optional[torch.Tensor],
    weight_t_fp8: Optional[torch.Tensor],
    scale_weight_t: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return mla_zigzag_backward(
        dout,
        q_nope,
        q_pe,
        normed_kv,
        k_pe,
        kv_b_weight,
        out,
        lse,
        sm_scale,
        qk_nope_head_dim,
        v_head_dim,
        _resolve_process_group(group_name),
        use_fp8=use_fp8,
        weight_fp8=weight_fp8,
        scale_weight=scale_weight,
        weight_t_fp8=weight_t_fp8,
        scale_weight_t=scale_weight_t,
    )


@_mla_zigzag_ring_bwd.register_fake
def _(
    dout,
    q_nope,
    q_pe,
    normed_kv,
    k_pe,
    kv_b_weight,
    out,
    lse,
    sm_scale,
    qk_nope_head_dim,
    v_head_dim,
    group_name,
    use_fp8,
    weight_fp8,
    scale_weight,
    weight_t_fp8,
    scale_weight_t,
):
    out_features, lora_rank = kv_b_weight.shape
    dW = torch.empty((out_features, lora_rank), dtype=torch.float32, device=kv_b_weight.device)
    # d_normed_kv / d_k_pe take q_nope.dtype to match mla_zigzag_backward's `.to(q_nope.dtype)`
    # cast (identical today since latent and query share a dtype, but the meta must track it).
    return (
        torch.empty_like(q_nope),
        torch.empty_like(q_pe),
        torch.empty(normed_kv.shape, dtype=q_nope.dtype, device=q_nope.device),
        torch.empty(k_pe.shape, dtype=q_nope.dtype, device=q_nope.device),
        dW,
    )


def _mla_zigzag_setup_context(ctx, inputs, output):
    (
        q_nope,
        q_pe,
        normed_kv,
        k_pe,
        kv_b_weight,
        sm_scale,
        qk_nope_head_dim,
        v_head_dim,
        group_name,
        use_fp8,
        weight_fp8,
        scale_weight,
        weight_t_fp8,
        scale_weight_t,
    ) = inputs
    out, lse = output
    ctx.save_for_backward(q_nope, q_pe, normed_kv, k_pe, kv_b_weight, out, lse)
    ctx.sm_scale = sm_scale
    ctx.qk_nope_head_dim = qk_nope_head_dim
    ctx.v_head_dim = v_head_dim
    ctx.group_name = group_name
    ctx.use_fp8 = use_fp8
    # FP8 weight tensors (None in bf16) are non-differentiable inputs reused in backward;
    # stash on ctx as the autograd.Function this replaced did. dW routes to kv_b_weight.
    ctx.weight_fp8 = weight_fp8
    ctx.scale_weight = scale_weight
    ctx.weight_t_fp8 = weight_t_fp8
    ctx.scale_weight_t = scale_weight_t


def _mla_zigzag_backward(ctx, grad_out, grad_lse):
    q_nope, q_pe, normed_kv, k_pe, kv_b_weight, out, lse = ctx.saved_tensors
    dq_nope, dq_pe, d_normed_kv, d_k_pe, dW = _mla_zigzag_ring_bwd(
        grad_out,
        q_nope,
        q_pe,
        normed_kv,
        k_pe,
        kv_b_weight,
        out,
        lse,
        ctx.sm_scale,
        ctx.qk_nope_head_dim,
        ctx.v_head_dim,
        ctx.group_name,
        ctx.use_fp8,
        ctx.weight_fp8,
        ctx.scale_weight,
        ctx.weight_t_fp8,
        ctx.scale_weight_t,
    )
    # Grads positionally match the fwd inputs; only the five differentiable tensors carry one.
    return (
        dq_nope,
        dq_pe,
        d_normed_kv,
        d_k_pe,
        dW,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_mla_zigzag_ring_fwd.register_autograd(
    _mla_zigzag_backward, setup_context=_mla_zigzag_setup_context
)


def mla_ring_attention_func(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    normed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    kv_b_weight: torch.Tensor,
    sm_scale: float,
    qk_nope_head_dim: int,
    v_head_dim: int,
    cp_group: ProcessGroup,
    kv_b_quant: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Causal zigzag ring attention for MLA, rotating the compressed latent.

    Parameters
    ----------
    q_nope, q_pe : torch.Tensor
        Query no-position / rope parts, shapes [batch, S_local, num_heads, qk_nope_head_dim]
        and [batch, S_local, num_heads, qk_rope_head_dim], in zigzag local layout. RoPE is
        already applied to q_pe by the caller. Queries stay on their home rank.
    normed_kv : torch.Tensor
        The compressed latent after kv_a_layernorm, shape [batch, S_local, kv_lora_rank], in
        zigzag local layout. Rotated around the ring; decompressed locally via kv_b_weight.
    k_pe : torch.Tensor
        Shared rope key, shape [batch, S_local, 1, qk_rope_head_dim], RoPE already applied.
        Rotated around the ring alongside normed_kv.
    kv_b_weight : torch.Tensor
        kv_b projection weight, shape [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank].
        FSDP-gathered (full) inside forward_attn; its gradient is reduced across CP by FSDP.
    sm_scale : float
        Softmax scale, typically q_head_dim ** -0.5.
    qk_nope_head_dim, v_head_dim : int
        Head dims used to split the decompressed kv into (k_nope, value).
    cp_group : torch.distributed.ProcessGroup
        Context-parallel process group (>= 2 ranks; cp_size == 1 is handled upstream).
    kv_b_quant : optional tuple
        (weight_fp8, scale_weight, weight_t_fp8, scale_weight_t) -- the kv_b weight
        pre-quantized once per micro-batch via FP8Linear._get_quantized_weight(). When given,
        the in-ring decompression uses the FP8 DeepGEMM path (matmul + dgrad/wgrad) instead of
        a bf16 F.linear. None -> bf16 path. ``kv_b_weight`` (the bf16 master) is still passed
        so its gradient (dW) routes to kv_b_proj.weight for FSDP.

    Returns
    -------
    torch.Tensor
        Attention output [batch, S_local, num_heads, v_head_dim] in q dtype, zigzag layout.
    """
    use_fp8 = kv_b_quant is not None
    weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = (
        kv_b_quant if use_fp8 else (None, None, None, None)
    )
    out, _ = _mla_zigzag_ring_fwd(
        q_nope,
        q_pe,
        normed_kv,
        k_pe,
        kv_b_weight,
        sm_scale,
        qk_nope_head_dim,
        v_head_dim,
        cp_group.group_name,
        use_fp8,
        weight_fp8,
        scale_weight,
        weight_t_fp8,
        scale_weight_t,
    )
    return out
