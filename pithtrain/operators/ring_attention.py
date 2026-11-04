"""
Zigzag ring attention for context parallelism.

Causal flash attention sharded across cp_size ranks. The global sequence is split into
2 * cp_size chunks of block = S / (2 * cp_size) tokens, and rank r holds chunk r (its front
block) and chunk 2 * cp_size - r - 1 (its back block), which balances the causal work across
ranks. For cp_size = 4:

    chunk:  0  1  2  3  4  5  6  7
    rank:   0  1  2  3  3  2  1  0

Q stays on its rank; K and V rotate one hop per step to rank r + 1, each hop posted before the
step's flash call so the transfer overlaps it. The backward rotates K and V again and, in the
same direction, the partial dK and dV, which reach their own rank after cp_size hops. Q, K, V and
the output are in this zigzag layout; the caller permutes the sequence.

At step s, rank r holds the K and V of rank kv = (r - s) mod cp_size, and exactly one of three
flash calls covers the step:

    s == 0        (kv == r)   flash(q,            k,            v,            causal=True)
    1 <= s <= r   (kv <  r)   flash(q,            k[:, :block], v[:, :block], causal=False)
    s >  r        (kv >  r)   flash(q[:, block:], k,            v,            causal=False)

  step 0: K is local, and the local causal mask is the global one.
  kv < r: only K's front block precedes Q, and all of Q attends to it.
  kv > r: only Q's back block follows K, and it attends to all of K.

Packed sequences use the same schedule one document at a time (zigzag_varlen_index), with one
varlen call per step: local boundaries are cu_seqlens // cp_size and block boundaries
cu_seqlens // (2 * cp_size).

The MLA variant rotates the compressed latent (normed_kv and k_pe) instead of per-head K and V,
and decompresses it through kv_b on every rank. The kv_b weight gradient stays on the rank (FSDP
reduces it across CP); only the latent gradient rotates back.
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

from pithtrain.kernels import (
    fp8_gemm_bwd,
    fp8_gemm_fwd,
    fp8_quant_act,
)

__all__ = ["mla_ring_attention_func", "ring_attention_func", "ring_attention_varlen_func"]


def post_ring_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: ProcessGroup,
    dst: int,
    src: int,
    k_recv: Optional[torch.Tensor] = None,
    v_recv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Work]]:
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
    Online-softmax merge of a partial flash output into the running fp32 accumulator.
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
            # Take over the dK and dV that arrived; the buffers sent one hop ago are now free to
            # receive the next ones.
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


# Each ring is a custom op so that a fullgraph torch.compile region can call it as one opaque
# node. A ProcessGroup cannot be a custom-op argument, so the op takes the group's name.
@torch.library.custom_op("pithtrain::zigzag_ring_fwd", mutates_args=())
def zigzag_ring_fwd(
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


@zigzag_ring_fwd.register_fake
def _(q, k, v, sm_scale, group_name):
    b, s, hq, _ = q.shape
    out = torch.empty((b, s, hq, v.shape[-1]), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, hq, s), dtype=torch.float32, device=q.device)
    return out, lse


@torch.library.custom_op("pithtrain::zigzag_ring_bwd", mutates_args=())
def zigzag_ring_bwd(
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


@zigzag_ring_bwd.register_fake
def _(dout, q, k, v, out, lse, sm_scale, group_name):
    # dk and dv take q.dtype, as zigzag_backward casts them to it.
    return (
        torch.empty_like(q),
        torch.empty(k.shape, dtype=q.dtype, device=q.device),
        torch.empty(v.shape, dtype=q.dtype, device=q.device),
    )


def zigzag_ring_setup_context(ctx, inputs, output):
    q, k, v, sm_scale, group_name = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.sm_scale = sm_scale
    ctx.group_name = group_name


def zigzag_ring_backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = zigzag_ring_bwd(grad_out, q, k, v, out, lse, ctx.sm_scale, ctx.group_name)
    return dq, dk, dv, None, None


zigzag_ring_fwd.register_autograd(zigzag_ring_backward, setup_context=zigzag_ring_setup_context)


def ring_attention_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    cp_group: ProcessGroup,
) -> torch.Tensor:
    """
    Causal zigzag ring attention across the cp_group ranks (at least two). q (B, S_local, Hq, D),
    k (B, S_local, Hkv, D) and v (B, S_local, Hkv, Dv) are in the zigzag layout, S_local being
    S / cp_size; k and v must be contiguous. Returns the output (B, S_local, Hq, Dv) in q.dtype,
    in the same layout.
    """
    out, _ = zigzag_ring_fwd(q, k, v, sm_scale, cp_group.group_name)
    return out


def document_blocks(cu_blocks: torch.Tensor, total: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The local indices of the front and back block tokens of every document, for documents that
    span [2 * cu_blocks[d], 2 * cu_blocks[d + 1]) of the total local tokens.
    """
    token_idx = torch.arange(total // 2, device=cu_blocks.device, dtype=cu_blocks.dtype)
    doc_idx = torch.searchsorted(cu_blocks, token_idx, right=True) - 1
    front = token_idx + cu_blocks[doc_idx]
    return front, front + cu_blocks[doc_idx + 1] - cu_blocks[doc_idx]


def zigzag_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sm_scale: float,
    cp_group: ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cp_rank, cp_size = get_rank(cp_group), get_world_size(cp_group)
    dst = get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    src = get_global_rank(cp_group, (cp_rank - 1) % cp_size)
    cu_local, cu_blocks = cu_seqlens // cp_size, cu_seqlens // (2 * cp_size)
    front, back = document_blocks(cu_blocks, q.shape[0])

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
                q,
                k,
                v,
                cu_seqlens_q=cu_local,
                cu_seqlens_k=cu_local,
                softmax_scale=sm_scale,
                causal=True,
                return_lse=True,
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse)
        elif step <= cp_rank:
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q,
                k[front],
                v[front],
                cu_seqlens_q=cu_local,
                cu_seqlens_k=cu_blocks,
                softmax_scale=sm_scale,
                causal=False,
                return_lse=True,
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse)
        else:
            # The remaining steps update back blocks only, so gather them once and scatter once.
            if step == cp_rank + 1:
                q_back = q[back]
                out_back, lse_back = out[back], lse[back]
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q_back,
                k,
                v,
                cu_seqlens_q=cu_blocks,
                cu_seqlens_k=cu_local,
                softmax_scale=sm_scale,
                causal=False,
                return_lse=True,
            )
            out_back, lse_back = combine_partial(out_back, lse_back, partial_out, partial_lse)
        if step + 1 < cp_size:
            wait_ring(kv_work)
            k, v = next_k, next_v

    if cp_rank + 1 < cp_size:
        out[back], lse[back] = out_back, lse_back
    out = out.to(q.dtype)
    lse = lse.squeeze(-1).transpose(0, 1).contiguous()
    return out, lse


def zigzag_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sm_scale: float,
    cp_group: ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_rank, cp_size = get_rank(cp_group), get_world_size(cp_group)
    dst = get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    src = get_global_rank(cp_group, (cp_rank - 1) % cp_size)
    cu_local, cu_blocks = cu_seqlens // cp_size, cu_seqlens // (2 * cp_size)
    front, back = document_blocks(cu_blocks, q.shape[0])

    dout = dout.contiguous()
    if cp_rank + 1 < cp_size:
        dout_back = dout[back]
        q_back = q[back]
        out_back = out[back]
        lse_back = lse[:, back]

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
                q,
                k,
                v,
                out,
                dout,
                lse,
                cu_seqlens_q=cu_local,
                cu_seqlens_k=cu_local,
                softmax_scale=sm_scale,
                causal=True,
            )
            dq = dq_step.to(torch.float32)
            dk = dk_step.to(torch.float32)
            dv = dv_step.to(torch.float32)
        else:
            if step <= cp_rank:
                dq_step, dk_step, dv_step = _flash_attn_bwd(
                    q,
                    k[front],
                    v[front],
                    out,
                    dout,
                    lse,
                    cu_seqlens_q=cu_local,
                    cu_seqlens_k=cu_blocks,
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
                    cu_seqlens_q=cu_blocks,
                    cu_seqlens_k=cu_local,
                    softmax_scale=sm_scale,
                    causal=False,
                )
                if step == cp_rank + 1:
                    dq_back = dq_step.to(torch.float32)
                else:
                    dq_back += dq_step
            # Take over the dK and dV that arrived; the buffers sent one hop ago are now free to
            # receive the next ones.
            wait_ring(grad_work)
            grad_recv_slot_k, grad_recv_slot_v = dk, dv
            dk, dv = incoming_dk, incoming_dv
            if step <= cp_rank:
                dk[front] += dk_step
                dv[front] += dv_step
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
    if cp_rank + 1 < cp_size:
        dq.index_add_(0, back, dq_back)
    return dq.to(q.dtype), incoming_dk.to(q.dtype), incoming_dv.to(q.dtype)


@torch.library.custom_op("pithtrain::zigzag_ring_varlen_fwd", mutates_args=())
def zigzag_ring_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sm_scale: float,
    group_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("ring attention requires contiguous k and v")
    if q.shape[0] % 2:
        raise ValueError(f"packed zigzag layout needs an even local token count, got {q.shape[0]}")
    return zigzag_varlen_forward(q, k, v, cu_seqlens, sm_scale, _resolve_process_group(group_name))


@zigzag_ring_varlen_fwd.register_fake
def _(q, k, v, cu_seqlens, sm_scale, group_name):
    t, hq, _ = q.shape
    out = torch.empty((t, hq, v.shape[-1]), dtype=q.dtype, device=q.device)
    lse = torch.empty((hq, t), dtype=torch.float32, device=q.device)
    return out, lse


@torch.library.custom_op("pithtrain::zigzag_ring_varlen_bwd", mutates_args=())
def zigzag_ring_varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sm_scale: float,
    group_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _resolve_process_group(group_name)
    return zigzag_varlen_backward(dout, q, k, v, out, lse, cu_seqlens, sm_scale, group)


@zigzag_ring_varlen_bwd.register_fake
def _(dout, q, k, v, out, lse, cu_seqlens, sm_scale, group_name):
    return (
        torch.empty_like(q),
        torch.empty(k.shape, dtype=q.dtype, device=q.device),
        torch.empty(v.shape, dtype=q.dtype, device=q.device),
    )


def zigzag_ring_varlen_setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, sm_scale, group_name = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
    ctx.sm_scale = sm_scale
    ctx.group_name = group_name


def zigzag_ring_varlen_backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
    dq, dk, dv = zigzag_ring_varlen_bwd(
        grad_out, q, k, v, out, lse, cu_seqlens, ctx.sm_scale, ctx.group_name
    )
    return dq, dk, dv, None, None, None


zigzag_ring_varlen_fwd.register_autograd(
    zigzag_ring_varlen_backward, setup_context=zigzag_ring_varlen_setup_context
)


def ring_attention_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sm_scale: float,
    cp_group: ProcessGroup,
) -> torch.Tensor:
    """
    Causal zigzag ring attention over a packed sequence, block-diagonal by document. q (S_local, Hq,
    D), k and v (S_local, Hkv, D or Dv) hold this rank's tokens in the order zigzag_varlen_index
    gives; k and v must be contiguous. cu_seqlens holds the int32 document boundaries of the whole
    packed sequence, every document a multiple of 2 * cp_size tokens long, so S_local ==
    cu_seqlens[-1] // cp_size. cp_group has at least two ranks. Returns the output (S_local, Hq, Dv)
    in q.dtype, in the same layout.
    """
    out, _ = zigzag_ring_varlen_fwd(q, k, v, cu_seqlens, sm_scale, cp_group.group_name)
    return out


def mla_decompress(
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
    """
    Decompress a latent block (B, S, kv_lora_rank) into per-head k_nope and value through kv_b.
    """
    batch, seq_len, _ = normed_kv.shape
    if use_fp8:
        x_2d = normed_kv.reshape(batch * seq_len, normed_kv.shape[-1])
        kv_2d, _, _ = fp8_gemm_fwd(x_2d, weight_fp8, scale_weight)
        kv = kv_2d.view(batch, seq_len, num_heads, qk_nope_head_dim + v_head_dim)
    else:
        kv = F.linear(normed_kv, kv_b_weight).view(batch, seq_len, num_heads, qk_nope_head_dim + v_head_dim)  # fmt: skip
    k_nope, value = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
    # Only value needs to be contiguous: FA4 reads it directly, while k_nope goes through a cat.
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
    fp8_kwargs = dict(use_fp8=use_fp8, weight_fp8=weight_fp8, scale_weight=scale_weight)

    q = torch.cat([q_nope, q_pe], dim=-1)
    # A strided view is fine for FA4's forward; the backward makes its copy contiguous.
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
            k_nope, value = mla_decompress(
                normed_kv, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **fp8_kwargs
            )
            key = torch.cat([k_nope, k_pe.expand(-1, -1, num_heads, -1)], dim=-1)
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q, key, value, softmax_scale=sm_scale, causal=True, return_lse=True
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse)
        elif step <= cp_rank:
            k_nope, value = mla_decompress(
                normed_kv[:, :block],
                kv_b_weight,
                num_heads,
                qk_nope_head_dim,
                v_head_dim,
                **fp8_kwargs,
            )
            key = torch.cat([k_nope, k_pe[:, :block].expand(-1, -1, num_heads, -1)], dim=-1)
            partial_out, partial_lse, *_ = _flash_attn_fwd(
                q, key, value, softmax_scale=sm_scale, causal=False, return_lse=True
            )
            out, lse = combine_partial(out, lse, partial_out, partial_lse)
        else:
            k_nope, value = mla_decompress(
                normed_kv, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **fp8_kwargs
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
    batch = q_nope.shape[0]
    num_heads = q_nope.shape[2]
    block = q_nope.shape[1] // 2
    rope_dim = k_pe.shape[-1]
    lora_rank = normed_kv.shape[-1]
    out_features = kv_b_weight.shape[0]
    weight_fp32 = None if use_fp8 else kv_b_weight.to(torch.float32)
    fp8_kwargs = dict(use_fp8=use_fp8, weight_fp8=weight_fp8, scale_weight=scale_weight)

    q = torch.cat([q_nope, q_pe], dim=-1)
    dout = dout.contiguous()
    q_back = q[:, block:].contiguous()
    dout_back = dout[:, block:].contiguous()
    out_back = out[:, block:].contiguous()
    lse_back = lse[:, :, block:].contiguous()

    dq: Optional[torch.Tensor] = None
    dW = torch.zeros((out_features, lora_rank), device=kv_b_weight.device, dtype=torch.float32)
    d_normed_kv: Optional[torch.Tensor] = None
    d_k_pe: Optional[torch.Tensor] = None
    next_kv: Optional[torch.Tensor] = None
    next_pe: Optional[torch.Tensor] = None
    kv_work: Optional[List[Work]] = None
    incoming_d_normed_kv: Optional[torch.Tensor] = None
    incoming_d_k_pe: Optional[torch.Tensor] = None
    grad_recv_slot_normed_kv: Optional[torch.Tensor] = None
    grad_recv_slot_k_pe: Optional[torch.Tensor] = None
    grad_work: Optional[List[Work]] = None

    for step in range(cp_size):
        if step + 1 < cp_size:
            next_kv, next_pe, kv_work = post_ring_kv(normed_kv, k_pe, cp_group, dst, src)

        if step == 0:
            normed_kv_block = normed_kv
            k_nope, value = mla_decompress(
                normed_kv, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **fp8_kwargs
            )
            key = torch.cat([k_nope, k_pe.expand(-1, -1, num_heads, -1)], dim=-1)
            dq_step, dkey_step, dvalue_step = _flash_attn_bwd(
                q, key, value, out, dout, lse, softmax_scale=sm_scale, causal=True
            )
            front = False
        elif step <= cp_rank:
            normed_kv_block = normed_kv[:, :block]
            k_nope, value = mla_decompress(
                normed_kv_block, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **fp8_kwargs
            )
            key = torch.cat([k_nope, k_pe[:, :block].expand(-1, -1, num_heads, -1)], dim=-1)
            dq_step, dkey_step, dvalue_step = _flash_attn_bwd(
                q, key, value, out, dout, lse, softmax_scale=sm_scale, causal=False
            )
            front = True
        else:
            normed_kv_block = normed_kv
            k_nope, value = mla_decompress(
                normed_kv, kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, **fp8_kwargs
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

        dk_nope, dk_pe = torch.split(dkey_step, [qk_nope_head_dim, rope_dim], dim=-1)
        kv_len = dk_nope.shape[1]
        if use_fp8:
            # The wgrad's transposed FP8 latent block is quantized again rather than saved.
            dy_2d = (
                torch.cat([dk_nope, dvalue_step], dim=-1)
                .reshape(batch * kv_len, out_features)
                .to(normed_kv.dtype)
            )
            _, _, x_t_fp8, scale_x_t = fp8_quant_act(
                normed_kv_block.reshape(batch * kv_len, lora_rank)
            )
            d_normed_kv_2d, dW_block = fp8_gemm_bwd(
                dy_2d, weight_t_fp8, scale_weight_t, x_t_fp8, scale_x_t, lora_rank
            )
            d_normed_kv_block = d_normed_kv_2d.view(batch, kv_len, lora_rank).to(torch.float32)
            dW += dW_block.to(torch.float32)
        else:
            d_kv = (
                torch.cat([dk_nope, dvalue_step], dim=-1)
                .reshape(batch * kv_len, out_features)
                .to(torch.float32)
            )
            d_normed_kv_block = (d_kv @ weight_fp32).view(batch, kv_len, lora_rank)
            dW += d_kv.t() @ normed_kv_block.reshape(batch * kv_len, lora_rank).to(torch.float32)
        d_k_pe_block = dk_pe.sum(dim=2, keepdim=True).to(torch.float32).contiguous()

        if step == 0:
            dq = dq_step.to(torch.float32)
        elif step <= cp_rank:
            dq += dq_step
        else:
            dq[:, block:] += dq_step

        if step == 0:
            # Aliasing is safe: later steps rebind d_normed_kv_block to a new tensor and never
            # write this send buffer in place.
            d_normed_kv = d_normed_kv_block
            d_k_pe = d_k_pe_block
        else:
            wait_ring(grad_work)
            grad_recv_slot_normed_kv, grad_recv_slot_k_pe = d_normed_kv, d_k_pe
            d_normed_kv, d_k_pe = incoming_d_normed_kv, incoming_d_k_pe
            if front:
                d_normed_kv[:, :block] += d_normed_kv_block
                d_k_pe[:, :block] += d_k_pe_block
            else:
                d_normed_kv += d_normed_kv_block
                d_k_pe += d_k_pe_block

        if step + 1 < cp_size:
            wait_ring(kv_work)
            normed_kv, k_pe = next_kv, next_pe

        incoming_d_normed_kv, incoming_d_k_pe, grad_work = post_ring_kv(
            d_normed_kv, d_k_pe, cp_group, dst, src, grad_recv_slot_normed_kv, grad_recv_slot_k_pe
        )

    wait_ring(grad_work)
    dq_nope, dq_pe = torch.split(dq, [qk_nope_head_dim, rope_dim], dim=-1)
    dtype = q_nope.dtype
    return (
        dq_nope.contiguous().to(dtype),
        dq_pe.contiguous().to(dtype),
        incoming_d_normed_kv.to(dtype),
        incoming_d_k_pe.to(dtype),
        dW,
    )


@torch.library.custom_op("pithtrain::mla_zigzag_ring_fwd", mutates_args=())
def mla_zigzag_ring_fwd(
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


@mla_zigzag_ring_fwd.register_fake
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
def mla_zigzag_ring_bwd(
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


@mla_zigzag_ring_bwd.register_fake
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
    # d_normed_kv and d_k_pe take q_nope.dtype, as mla_zigzag_backward casts them to it.
    return (
        torch.empty_like(q_nope),
        torch.empty_like(q_pe),
        torch.empty(normed_kv.shape, dtype=q_nope.dtype, device=q_nope.device),
        torch.empty(k_pe.shape, dtype=q_nope.dtype, device=q_nope.device),
        dW,
    )


def mla_zigzag_ring_setup_context(ctx, inputs, output):
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
    # The FP8 weights (None in BF16) are non-differentiable, so they stay off the saved tensors.
    ctx.weight_fp8 = weight_fp8
    ctx.scale_weight = scale_weight
    ctx.weight_t_fp8 = weight_t_fp8
    ctx.scale_weight_t = scale_weight_t


def mla_zigzag_ring_backward(ctx, grad_out, grad_lse):
    q_nope, q_pe, normed_kv, k_pe, kv_b_weight, out, lse = ctx.saved_tensors
    dq_nope, dq_pe, d_normed_kv, d_k_pe, dW = mla_zigzag_ring_bwd(
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


mla_zigzag_ring_fwd.register_autograd(
    mla_zigzag_ring_backward, setup_context=mla_zigzag_ring_setup_context
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
    Causal zigzag ring attention for MLA across the cp_group ranks (at least two), rotating the
    compressed latent. In the zigzag layout, with RoPE already applied: q_nope (B, S_local, H,
    qk_nope_head_dim), q_pe (B, S_local, H, qk_rope_head_dim), normed_kv (B, S_local, kv_lora_rank)
    after kv_a_layernorm, and k_pe (B, S_local, 1, qk_rope_head_dim); normed_kv and k_pe must be
    contiguous. kv_b_weight is the full (H * (qk_nope_head_dim + v_head_dim), kv_lora_rank) bf16
    weight, which receives the gradient. With kv_b_quant, the quantized kv_b weight (weight_fp8,
    scale_weight, weight_t_fp8, scale_weight_t), the decompression runs in FP8. Returns the output
    (B, S_local, H, v_head_dim) in q_nope's dtype, in the same layout.
    """
    use_fp8 = kv_b_quant is not None
    weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = (
        kv_b_quant if use_fp8 else (None, None, None, None)
    )
    out, _ = mla_zigzag_ring_fwd(
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
