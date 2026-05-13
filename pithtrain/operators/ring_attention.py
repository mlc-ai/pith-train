"""
Ring attention for context parallelism.

Splits the sequence across CP ranks. Each rank holds Q for its local chunk
and passes K/V around a ring so every rank computes full causal attention.
Partial outputs are combined with online softmax (log-sum-exp rescaling).

Forward:  KV travels next->prev (send to rank+1, recv from rank-1).
Backward: re-uses saved KV chunks and passes the *combined* out/lse to
          flash_attn_backward so it reconstructs the correct global attention
          weights.  dK/dV contributions are redistributed to originating
          ranks via distance-based P2P exchange.

Known limitations
-----------------
* **Causal load imbalance**: With contiguous chunking, rank 0 processes 1 KV
  chunk while rank C-1 processes C chunks.  Zigzag/striped partitioning would
  fix this but is not yet implemented.
* **O(cp_size) KV memory in backward**: All received KV chunks are saved from
  the forward pass for reuse in the backward.  For large cp_size a ring-based
  KV recomputation scheme would reduce this to O(1).
"""

import torch
import torch.distributed as dist
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


def _ring_send_recv_kv(
    send_k: torch.Tensor,
    send_v: torch.Tensor,
    next_global: int,
    prev_global: int,
):
    assert send_k.is_contiguous() and send_v.is_contiguous(), (
        "ring attention send_k or send_v is not contiguous"
    )
    recv_k = torch.empty_like(send_k)
    recv_v = torch.empty_like(send_v)
    ops = [
        dist.P2POp(dist.isend, send_k, next_global),
        dist.P2POp(dist.isend, send_v, next_global),
        dist.P2POp(dist.irecv, recv_k, prev_global),
        dist.P2POp(dist.irecv, recv_v, prev_global),
    ]
    reqs = dist.batch_isend_irecv(ops)
    # NCCL runs on its own internal CUDA stream
    for req in reqs:
        req.wait()
    return recv_k, recv_v


def _online_softmax_combine(out1, lse1, out2, lse2):
    """Combine two partial attention outputs via online softmax rescaling.

    out: [B, S, H, D],  lse: [B, H, S]  (float32).
    """
    max_lse = torch.maximum(lse1, lse2)
    exp1 = torch.exp(lse1 - max_lse)
    exp2 = torch.exp(lse2 - max_lse)
    # [B, H, S] -> [B, S, H, 1] for broadcast with [B, S, H, D]
    e1 = exp1.transpose(1, 2).unsqueeze(-1)
    e2 = exp2.transpose(1, 2).unsqueeze(-1)
    new_out = (e1 * out1 + e2 * out2) / (e1 + e2)
    new_lse = max_lse + torch.log(exp1 + exp2)
    return new_out, new_lse


class RingAttentionFunc(torch.autograd.Function):
    """Causal ring attention with flash-attn kernels."""

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, cp_rank, cp_size, global_ranks):
        B, S, H, _ = q.shape
        DV = v.shape[-1]
        combined_out = torch.zeros((B, S, H, DV), dtype=q.dtype, device=q.device)
        combined_lse = torch.full(
            (B, H, S), torch.finfo(torch.float32).min, device=q.device, dtype=torch.float32
        )

        next_global = global_ranks[(cp_rank + 1) % cp_size]
        prev_global = global_ranks[(cp_rank - 1) % cp_size]
        assert k.is_contiguous() and v.is_contiguous(), "ring attention k or v is not contiguous"
        cur_k, cur_v = k, v
        saved_k, saved_v, saved_ranks = [], [], []

        for step in range(cp_size):
            kv_rank = (cp_rank - step) % cp_size

            if kv_rank <= cp_rank:
                use_causal = kv_rank == cp_rank
                step_out, step_lse = _flash_attn_fwd(
                    q, cur_k, cur_v, softmax_scale=softmax_scale, causal=use_causal, return_lse=True
                )
                combined_out, combined_lse = _online_softmax_combine(
                    combined_out, combined_lse, step_out, step_lse
                )
                saved_k.append(cur_k)
                saved_v.append(cur_v)
                saved_ranks.append(kv_rank)

            if step < cp_size - 1:
                cur_k, cur_v = _ring_send_recv_kv(cur_k, cur_v, next_global, prev_global)

        combined_out = combined_out.to(q.dtype)

        assert combined_lse.is_contiguous()
        ctx.save_for_backward(q, combined_out, combined_lse, *saved_k, *saved_v)
        ctx.softmax_scale = softmax_scale
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.global_ranks = global_ranks
        ctx.saved_ranks = saved_ranks
        ctx.n_saved = len(saved_ranks)
        return combined_out

    @staticmethod
    def backward(ctx, dout):
        n = ctx.n_saved
        saved = ctx.saved_tensors
        q, combined_out, combined_lse = saved[0], saved[1], saved[2]
        all_k = list(saved[3 : 3 + n])
        all_v = list(saved[3 + n : 3 + 2 * n])
        cp_rank, cp_size, global_ranks = ctx.cp_rank, ctx.cp_size, ctx.global_ranks

        dq = torch.zeros_like(q)
        local_dk = torch.zeros_like(all_k[0])
        local_dv = torch.zeros_like(all_v[0])
        remote_dk, remote_dv = {}, {}

        for i, kv_rank in enumerate(ctx.saved_ranks):
            dq_s, dk_s, dv_s = _flash_attn_bwd(
                q,
                all_k[i],
                all_v[i],
                combined_out,
                dout,
                combined_lse,
                softmax_scale=ctx.softmax_scale,
                causal=(kv_rank == cp_rank),
            )
            dq += dq_s
            if kv_rank == cp_rank:
                local_dk += dk_s
                local_dv += dv_s
            else:
                remote_dk[kv_rank] = dk_s
                remote_dv[kv_rank] = dv_s

        # Redistribute dK/dV to originating ranks via P2P.
        # At distance d: rank r sends dK for rank (r-d) and receives dK from rank (r+d).
        for d in range(1, cp_size):
            target, source = cp_rank - d, cp_rank + d
            ops, recv_dk_buf, recv_dv_buf = [], None, None

            if target >= 0 and target in remote_dk:
                dst = global_ranks[target]
                assert remote_dk[target].is_contiguous() and remote_dv[target].is_contiguous(), (
                    "ring attention remote_dk or remote_dv is not contiguous"
                )
                ops.append(dist.P2POp(dist.isend, remote_dk[target], dst))
                ops.append(dist.P2POp(dist.isend, remote_dv[target], dst))
            if source < cp_size:
                src = global_ranks[source]
                recv_dk_buf = torch.empty_like(local_dk)
                recv_dv_buf = torch.empty_like(local_dv)
                ops.append(dist.P2POp(dist.irecv, recv_dk_buf, src))
                ops.append(dist.P2POp(dist.irecv, recv_dv_buf, src))

            if ops:
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
            if recv_dk_buf is not None:
                local_dk += recv_dk_buf
                local_dv += recv_dv_buf

        return dq, local_dk, local_dv, None, None, None, None


def ring_attention_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    cp_group: dist.ProcessGroup,
):
    """Causal ring attention across context-parallel ranks."""
    cp_rank = cp_group.rank()
    cp_size = cp_group.size()
    global_ranks = [dist.distributed_c10d.get_global_rank(cp_group, r) for r in range(cp_size)]
    return RingAttentionFunc.apply(q, k, v, softmax_scale, cp_rank, cp_size, global_ranks)
