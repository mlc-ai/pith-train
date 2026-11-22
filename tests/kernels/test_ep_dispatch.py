"""
The EP dispatch index kernels at the ep_dispatch cases, exact against PyTorch (one GPU).
"""

import pytest
import torch

from pithtrain.kernels import (
    ep_dispatch_recv_idx,
    ep_dispatch_send_idx,
)
from tests.kernels.cases import params, seed_of
from tests.kernels.common import I64, REALISTIC, group_topk, receive


@pytest.fixture(autouse=True)
def empty_cache():
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()


def send_reference(topk_ids: torch.Tensor, ep: int, epr: int) -> dict:
    """
    ep_dispatch_send_idx's outputs, placed stably.
    """
    m, k = topk_ids.shape
    flat = topk_ids.reshape(-1)
    idxs = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ep * epr)
    visit = torch.zeros(m, ep, dtype=torch.bool, device="cuda")
    visit.scatter_(1, topk_ids // epr, True)
    dedup = visit.sum(0)
    position = visit.long().cumsum(0) - 1
    return dict(
        tokens_per_ep_rank=counts.view(ep, epr).sum(1),
        dedup=dedup,
        dispatch=visit.T.nonzero()[:, 1],
        idxs=idxs,
        expand=position[idxs // k, flat[idxs] // epr],
        send_meta=torch.cat([counts.view(ep, epr), dedup[:, None]], 1).reshape(-1),
    )


@pytest.mark.parametrize("case", params("ep_dispatch", keep=lambda c: REALISTIC(c) and c["axes"]["ep"] > 1))  # fmt: skip
def test_ep_dispatch(case: dict) -> None:
    a = case["axes"]
    T, ep, E, k, G = a["T"], a["ep"], a["E"], a["top_k"], a["G"]
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    topk = group_topk(case, gen)
    recv_meta, recv_expand, dedup_tokens = [], [], []
    for sender in range(ep):
        ids = topk[sender * T : (sender + 1) * T]
        meta = torch.full((2 * ep + ep * (G + 1),), -7, **I64)
        outs = ep_dispatch_send_idx(ids, E, ep, G, meta)
        tokens_per_ep_rank, dedup, dispatch, idxs, expand, send_meta = outs
        assert all(out.dtype == torch.int64 for out in outs)
        assert tokens_per_ep_rank.data_ptr() == meta.data_ptr() and dedup.data_ptr() == meta.data_ptr() + 8 * ep  # fmt: skip
        assert dispatch.shape == (T * ep,) and idxs.shape == expand.shape == (T * k,)
        assert send_meta.shape == (ep * (G + 1),) and bool((meta[2 * ep :] == -7).all())
        ref = send_reference(ids, ep, G)
        assert torch.equal(tokens_per_ep_rank, ref["tokens_per_ep_rank"])
        assert torch.equal(dedup, ref["dedup"]) and torch.equal(send_meta, ref["send_meta"])
        total = int(ref["dedup"].sum())
        flat = ids.reshape(-1)
        assert torch.equal(idxs, ref["idxs"]) and torch.equal(expand, ref["expand"])
        assert torch.equal(dispatch[:total], ref["dispatch"])
        rank = flat[idxs] // G
        starts = ref["dedup"].cumsum(0) - ref["dedup"]
        assert bool(((expand >= 0) & (expand < ref["dedup"][rank])).all())
        assert torch.equal(dispatch[starts[rank] + expand], idxs // k)
        recv_meta.append(send_meta.view(ep, G + 1)[0].clone())
        recv_expand.append(expand[: int(tokens_per_ep_rank[0])].clone())
        dedup_tokens.append(dispatch[: int(dedup[0])] + sender * T)
    recv_meta, recv_expand, dedup_tokens = torch.cat(recv_meta), torch.cat(recv_expand), torch.cat(dedup_tokens)  # fmt: skip
    expert_idxs, expand_idx = ep_dispatch_recv_idx(recv_meta, recv_expand, ep, G)
    rows = recv_meta.view(ep, G + 1)
    want_expert = torch.arange(G, **I64).repeat(ep).repeat_interleave(rows[:, :G].reshape(-1))
    dedup_start = rows[:, G].cumsum(0) - rows[:, G]
    want_expand = recv_expand + dedup_start.repeat_interleave(rows[:, :G].sum(1))
    assert expert_idxs.dtype == expand_idx.dtype == torch.int64
    assert torch.equal(expert_idxs, want_expert) and torch.equal(expand_idx, want_expand)
    recv = receive(topk, T, G)
    assert torch.bincount(expert_idxs, minlength=G).tolist() == a["group_sizes"]
    assert torch.equal(expert_idxs, recv["expert"])
    token = dedup_tokens[expand_idx]
    assert torch.equal(torch.sort(token * G + expert_idxs).values, torch.sort(recv["token"] * G + recv["expert"]).values)  # fmt: skip
    assert bool((topk[token] == expert_idxs[:, None]).any(1).all())
    assert torch.equal(token, recv["token"]) and torch.equal(expand_idx, recv["expand"])
