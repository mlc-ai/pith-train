"""
fp32 softmax attention and its gradients, one block of query rows at a time.
"""

import torch

MAX_BLOCK_ELEMENTS = 1 << 26  # caps each fp32 (heads x rows x keys) score block's memory


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    window: int | None = None,
    sink: torch.Tensor | None = None,
    dout: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
    o_in: torch.Tensor | None = None,
    lse_in: torch.Tensor | None = None,
) -> dict:
    """
    Attention of one sequence in fp32: out and lse, plus dq, dk, dv (and dsink) when dout is given.
    """
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return attention_blocks(q, k, v, scale, causal, window, sink, dout, q_positions, o_in, lse_in)  # fmt: skip
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def attention_blocks(q, k, v, scale, causal, window, sink, dout, q_positions, o_in, lse_in) -> dict:
    Sq, Hq, D = q.shape
    Skv, Hkv, Dv = v.shape
    group = Hq // Hkv
    f32 = torch.float32
    qh = q.to(f32).transpose(0, 1).contiguous()
    kh = k.to(f32).repeat_interleave(group, dim=1).transpose(0, 1).contiguous()
    vh = v.to(f32).repeat_interleave(group, dim=1).transpose(0, 1).contiguous()
    if q_positions is None:
        q_positions = torch.arange(Sq, device=q.device) + (Skv - Sq)
    q_positions = q_positions.to(q.device)
    out = torch.empty(Hq, Sq, Dv, dtype=f32, device=q.device)
    lse = torch.empty(Hq, Sq, dtype=f32, device=q.device)
    grads = dout is not None
    if grads:
        doh = dout.to(f32).transpose(0, 1).contiguous()
        o_used_all = o_in.to(f32).transpose(0, 1) if o_in is not None else None
        dq = torch.zeros_like(qh)
        dk = torch.zeros_like(kh)
        dv = torch.zeros_like(vh)
        dsink = torch.zeros(Hq, dtype=f32, device=q.device) if sink is not None else None
    sink_f = sink.to(f32)[:, None] if sink is not None else None
    rows = max(16, MAX_BLOCK_ELEMENTS // (Hq * Skv) // 16 * 16)
    key_index = torch.arange(Skv, device=q.device)
    for r0 in range(0, Sq, rows):
        r1 = min(Sq, r0 + rows)
        pos = q_positions[r0:r1]
        lo, hi = 0, Skv
        if causal:
            hi = min(Skv, int(pos.max()) + 1)
            if window is not None:
                lo = max(0, int(pos.min()) - window)
        if hi <= lo:
            raise ValueError("a query block sees no key")
        qb, kb, vb = qh[:, r0:r1], kh[:, lo:hi], vh[:, lo:hi]
        s = torch.matmul(qb, kb.transpose(1, 2)).mul_(scale)
        if causal:
            j = key_index[lo:hi][None, :]
            allowed = j <= pos[:, None]
            if window is not None:
                allowed &= j >= pos[:, None] - window
            s.masked_fill_(~allowed[None], float("-inf"))
        m = torch.logsumexp(s, dim=-1)
        if sink_f is not None:
            m = torch.logaddexp(m, sink_f.expand_as(m))
        p = torch.exp(s - m[..., None])
        o = torch.matmul(p, vb)
        out[:, r0:r1], lse[:, r0:r1] = o, m
        if not grads:
            continue
        if lse_in is not None:
            m_used = lse_in[:, r0:r1].to(f32)
            p = torch.exp(s - m_used[..., None])
        else:
            m_used = m
        o_used = o_used_all[:, r0:r1] if o_used_all is not None else o
        dob = doh[:, r0:r1]
        delta = (dob * o_used).sum(-1, keepdim=True)
        dp = torch.matmul(dob, vb.transpose(1, 2))
        ds = p * (dp - delta)
        del dp, s
        dq[:, r0:r1] = torch.matmul(ds, kb).mul_(scale)
        dk[:, lo:hi] += torch.matmul(ds.transpose(1, 2), qb).mul_(scale)
        dv[:, lo:hi] += torch.matmul(p.transpose(1, 2), dob)
        if dsink is not None:
            dsink -= (torch.exp(sink_f - m_used) * delta.squeeze(-1)).sum(-1)
    result = dict(out=out.transpose(0, 1), lse=lse)
    if grads:
        result["dq"] = dq.transpose(0, 1)
        result["dk"] = dk.reshape(Hkv, group, Skv, D).sum(1).transpose(0, 1)
        result["dv"] = dv.reshape(Hkv, group, Skv, Dv).sum(1).transpose(0, 1)
        if dsink is not None:
            result["dsink"] = dsink
    return result
