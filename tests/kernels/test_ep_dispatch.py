"""
Verifies the ep_dispatch kernels, ep_dispatch_send_idx and ep_dispatch_recv_idx, against
PyTorch. Names: m tokens, k experts per token, ne experts, ep EP ranks, epr experts per rank.

Sender side, where a token routed to several experts of one EP rank is sent to it once:

- tokens_per_ep_rank, dedup_tokens_per_gpu and send_meta carry no order and are compared exactly.
- idxs, expand_idx and dispatch_token_idxs carry an order. On the inputs the H200 path covers they
  are compared exactly with a stable reference (argsort(stable=True), nonzero), since the CuTe
  kernel places tokens in order within each bucket. Other inputs fall back to atomic placement,
  where they are compared as per-bucket sets plus the invariant
  dispatch_token_idxs[rank_start + expand_idx] == idxs // k.

Receiver side, ep_dispatch_recv_idx carries no order and is compared exactly. End-to-end tests
emulate the operator's all-to-alls in one process and check that every received slot resolves to
its token and expert.

The routing contract: a token's top-k experts are distinct, as torch.topk returns them.
"""

import pytest
import torch

from pithtrain.kernels import ep_dispatch_recv_idx, ep_dispatch_send_idx

# (num_experts, top-k) of the models: deepseek-v2-lite, qwen3-30b-a3b, qwen3.5-35b-a3b,
# gpt-oss-20b, gpt-oss-120b
MODELS = [(64, 6), (128, 8), (256, 8), (32, 4), (128, 4)]


def stable_reference(topk_ids: torch.Tensor, ep: int, epr: int) -> dict:
    """
    Every sender-side output, placed stably (token order within each bucket).
    """
    m, k = topk_ids.shape
    ne = ep * epr
    flat = topk_ids.reshape(-1).long()
    idxs = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ne)
    visit = torch.zeros(m, ep, dtype=torch.bool, device=topk_ids.device)
    if m:
        visit.scatter_(1, topk_ids.long() // epr, True)
    dedup = visit.sum(0)
    dispatch_token_idxs = visit.T.nonzero()[:, 1]  # rank-major, token order within a rank
    rank = visit.long().cumsum(0) - 1  # position of a token within its rank's dedup chunk
    expand = rank[idxs // k, flat[idxs] // epr] if m else idxs
    return {
        "tokens_per_ep_rank": counts.view(ep, epr).sum(1),
        "dedup": dedup,
        "dti": dispatch_token_idxs,
        "idxs": idxs,
        "expand": expand,
        "send_meta": torch.cat([counts.view(ep, epr), dedup[:, None]], 1).reshape(-1),
    }


def routing(m: int, k: int, ne: int, epr: int, kind: str = "random", seed: int = 0):
    """
    (m, k) int64 distinct expert ids per token, spread as kind says.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.rand(m, ne, generator=gen).argsort(1)[:, :k]
    if kind == "one_expert" and m:
        ids[:, 0] = 0  # every token picks expert 0 (its other picks stay distinct)
        ids[:, 1:] = (torch.rand(m, ne - 1, generator=gen).argsort(1)[:, : k - 1]) + 1
    elif kind == "one_rank" and m:
        assert epr >= k
        ids = torch.rand(m, epr, generator=gen).argsort(1)[:, :k]  # every pick on rank 0
    elif kind == "ragged" and m:
        # a few hot experts, and every third expert and all of EP rank 1 empty
        weights = torch.exp(torch.randn(ne, generator=gen) * 3)
        weights[1::3] = 0
        weights[epr : 2 * epr] = 0
        assert int((weights > 0).sum()) >= k
        ids = torch.multinomial(weights.expand(m, ne), k, replacement=False, generator=gen)
    return ids.cuda()


def check_shapes(outs, m: int, k: int, ep: int, epr: int) -> None:
    tokens_per_ep_rank, dedup, dispatch_token_idxs, idxs, expand, send_meta = outs
    assert tokens_per_ep_rank.shape == dedup.shape == (ep,)
    # over-allocated; the first sum(dedup) entries are valid
    assert dispatch_token_idxs.shape == (m * ep,)
    assert idxs.shape == expand.shape == (m * k,)
    assert send_meta.shape == (ep * (epr + 1),)
    assert all(out.dtype == torch.int64 for out in outs)


def check_order_free(outs, ref) -> None:
    assert torch.equal(outs[0], ref["tokens_per_ep_rank"])
    assert torch.equal(outs[1], ref["dedup"])
    assert torch.equal(outs[5], ref["send_meta"])


def check_stable(outs, ref) -> None:
    """
    Exact: the H200 path places stably.
    """
    check_order_free(outs, ref)
    total = int(ref["dedup"].sum())
    assert torch.equal(outs[3], ref["idxs"])
    assert torch.equal(outs[4], ref["expand"])
    assert torch.equal(outs[2][:total], ref["dti"])


def buckets(values: torch.Tensor, counts: list[int]) -> list[list[int]]:
    value_list, out, start = values.cpu().tolist(), [], 0
    for count in counts:
        out.append(sorted(value_list[start : start + count]))
        start += count
    return out


def check_any_order(outs, ref, topk_ids: torch.Tensor, ep: int, epr: int) -> None:
    """
    Order-free outputs exactly; order-carrying ones as per-bucket sets plus the invariant.
    """
    check_order_free(outs, ref)
    k = topk_ids.shape[1]
    _, dedup, dispatch_token_idxs, idxs, expand, _ = outs
    counts = torch.bincount(topk_ids.reshape(-1).long(), minlength=ep * epr).tolist()
    assert buckets(idxs, counts) == buckets(ref["idxs"], counts)
    dedup_counts = dedup.tolist()
    total = sum(dedup_counts)
    assert buckets(dispatch_token_idxs[:total], dedup_counts) == buckets(ref["dti"], dedup_counts)
    rank = topk_ids.reshape(-1).long()[idxs] // epr
    rank_starts = torch.cumsum(dedup, 0) - dedup
    assert bool(((expand >= 0) & (expand < dedup[rank])).all())
    assert torch.equal(dispatch_token_idxs[rank_starts[rank] + expand], idxs // k)


def run_dispatch(topk_ids: torch.Tensor, ep: int, epr: int, meta_out=None):
    outs = ep_dispatch_send_idx(topk_ids, ep * epr, ep, epr, meta_out)
    check_shapes(outs, topk_ids.shape[0], topk_ids.shape[1], ep, epr)
    return outs


@pytest.mark.parametrize("m", [0, 1, 511, 513, 4133])  # the CuTe kernel's chunk is 512 tokens
@pytest.mark.parametrize("ep", [1, 2, 4, 8])
@pytest.mark.parametrize("ne,k", MODELS)
def test_dispatch_models(ne: int, k: int, ep: int, m: int) -> None:
    epr = ne // ep
    topk_ids = routing(m, k, ne, epr, seed=m + k + ep)
    check_stable(run_dispatch(topk_ids, ep, epr), stable_reference(topk_ids, ep, epr))


@pytest.mark.parametrize("k", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("m", [4096, 65536, 70000])
def test_dispatch_odd_experts_per_rank(k: int, ep: int, m: int) -> None:
    """
    The token counts span many CTAs.
    """
    epr = 5 if ep <= 4 else 3
    topk_ids = routing(m, k, ep * epr, epr, seed=m + k)
    check_stable(run_dispatch(topk_ids, ep, epr), stable_reference(topk_ids, ep, epr))


@pytest.mark.parametrize("kind", ["one_expert", "one_rank", "ragged"])
@pytest.mark.parametrize(
    "ep,epr,k", [(4, 32, 8), (8, 32, 8), (4, 64, 8), (8, 8, 6), (4, 8, 4), (8, 16, 4)]
)
def test_dispatch_skewed(kind: str, ep: int, epr: int, k: int) -> None:
    if kind == "one_rank" and epr < k:
        pytest.skip("needs k experts on one rank")
    topk_ids = routing(4096, k, ep * epr, epr, kind, seed=3)
    ref = stable_reference(topk_ids, ep, epr)
    if kind == "ragged":
        assert int((ref["send_meta"] == 0).sum()) > 0  # some empty experts
        assert int(ref["dedup"][1]) == 0  # an empty EP rank
    check_stable(run_dispatch(topk_ids, ep, epr), ref)


def test_dispatch_m_zero() -> None:
    for ep, epr, k in ((4, 4, 4), (1, 64, 6), (8, 32, 8)):
        topk_ids = torch.empty((0, k), dtype=torch.int64, device="cuda")
        outs = run_dispatch(topk_ids, ep, epr)
        assert int(outs[0].abs().sum()) == int(outs[1].abs().sum()) == 0
        assert int(outs[5].abs().sum()) == 0
        assert all(out.numel() == 0 for out in outs[2:5])


def test_dispatch_meta_out() -> None:
    """
    With meta_out, tokens_per_ep_rank and dedup_tokens_per_gpu are its first 2 * ep_size entries,
    as the operator lays out its metadata buffer; the rest is untouched.
    """
    ep, epr, k = 4, 32, 8
    for m in (0, 4133):
        topk_ids = routing(m, k, ep * epr, epr, seed=9)
        meta = torch.full((2 * ep + ep * (epr + 1),), -7, dtype=torch.int64, device="cuda")
        outs = run_dispatch(topk_ids, ep, epr, meta)
        assert outs[0].data_ptr() == meta.data_ptr()
        assert outs[1].data_ptr() == meta.data_ptr() + ep * 8
        ref = stable_reference(topk_ids, ep, epr)
        assert torch.equal(meta[:ep], ref["tokens_per_ep_rank"])
        assert torch.equal(meta[ep : 2 * ep], ref["dedup"])
        assert bool((meta[2 * ep :] == -7).all())
        check_stable(outs, ref)


def test_dispatch_row_stride() -> None:
    """
    The models hand over a contiguous (m, k) topk; a wider row stride is honoured too.
    """
    ep, epr, k = 4, 32, 8
    wide = routing(4096, k + 3, ep * epr, epr, seed=5)
    topk_ids = wide[:, :k]
    assert topk_ids.stride(0) == k + 3
    check_stable(run_dispatch(topk_ids, ep, epr), stable_reference(topk_ids.contiguous(), ep, epr))


def test_dispatch_back_to_back() -> None:
    """
    Back-to-back calls with different m on one stream (reused scratch) stay exact, and later calls
    do not overwrite the outputs of earlier ones.
    """
    ep, epr, k = 8, 16, 8
    cases = [routing(m, k, ep * epr, epr, seed=m) for m in (70000, 1, 4096, 33, 65536, 4096)]
    outs = [run_dispatch(topk_ids, ep, epr) for topk_ids in cases]
    for topk_ids, case_outs in zip(cases, outs):
        check_stable(case_outs, stable_reference(topk_ids, ep, epr))


def test_dispatch_grid_barrier_stress() -> None:
    """
    Repeated launches whose grid spans every SM (many 512-token chunks per CTA), alternating with
    small ones: each result stays exact.
    """
    ep, epr, k = 8, 16, 8
    big = routing(300_000, k, ep * epr, epr, kind="ragged", seed=21)
    small = routing(700, k, ep * epr, epr, seed=22)
    refs = {id(topk_ids): stable_reference(topk_ids, ep, epr) for topk_ids in (big, small)}
    for i in range(12):
        topk_ids = big if i % 2 == 0 else small
        check_stable(run_dispatch(topk_ids, ep, epr), refs[id(topk_ids)])


def test_dispatch_stability_is_enforced() -> None:
    """
    A placement atomics could produce (two rows of one bucket swapped) passes the per-bucket
    comparison but not the stable one, so the exact comparison checks order.
    """
    ep, epr, k = 4, 8, 4
    topk_ids = routing(1024, k, ep * epr, epr, seed=11)
    outs = list(run_dispatch(topk_ids, ep, epr))
    ref = stable_reference(topk_ids, ep, epr)
    assert int(outs[5][0]) >= 2
    swapped = outs[3].clone()
    swapped[[0, 1]] = swapped[[1, 0]]  # both in expert 0's bucket
    assert not torch.equal(swapped, ref["idxs"])
    counts = torch.bincount(topk_ids.reshape(-1), minlength=ep * epr).tolist()
    assert buckets(swapped, counts) == buckets(ref["idxs"], counts)


def test_dispatch_fallback_int32_ids() -> None:
    for ep, epr, k in ((4, 32, 8), (8, 8, 6), (1, 64, 6)):
        topk_ids = routing(4133, k, ep * epr, epr, seed=2).int()
        outs = run_dispatch(topk_ids, ep, epr)
        check_any_order(outs, stable_reference(topk_ids, ep, epr), topk_ids, ep, epr)


def test_dispatch_fallback_many_bins() -> None:
    """
    More than 1024 bins, ep_size * (experts_per_rank + 1), with as few experts as reach that, since
    the fallback unrolls a loop per expert and compiles slowly for more.
    """
    ep, epr, k = 32, 32, 8
    for m in (513,):
        topk_ids = routing(m, k, ep * epr, epr, seed=m)
        outs = run_dispatch(topk_ids, ep, epr)
        check_any_order(outs, stable_reference(topk_ids, ep, epr), topk_ids, ep, epr)


def expand_reference(meta: torch.Tensor, received_expand_idx: torch.Tensor, ep: int, epr: int):
    """
    Each slot's local expert, and its sender-local expand_idx plus the dedup tokens of the preceding
    senders; slots are sender-major and expert-minor.
    """
    meta_rows = meta.view(ep, epr + 1)
    segment_sizes = meta_rows[:, :epr].reshape(-1)
    expert = torch.arange(epr, device=meta.device).repeat(ep).repeat_interleave(segment_sizes)
    dedup_start = meta_rows[:, epr].cumsum(0) - meta_rows[:, epr]
    offset = dedup_start.repeat_interleave(meta_rows[:, :epr].sum(1))
    return expert, received_expand_idx + offset


def recv_case(ep: int, epr: int, total_target: int, seed: int, zero_frac: float = 0.3):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ne = ep * epr
    weights = torch.rand(ne, generator=gen) * (torch.rand(ne, generator=gen) > zero_frac)
    segment_sizes = (weights / weights.sum().clamp_min(1e-9) * total_target).long()
    dedup = torch.randint(1, 4000, (ep,), generator=gen)
    meta = torch.cat([segment_sizes.view(ep, epr), dedup[:, None]], 1).reshape(-1).contiguous().cuda()  # fmt: skip
    received_expand_idx = torch.randint(0, 4000, (int(segment_sizes.sum()),), generator=gen).cuda()
    return meta, received_expand_idx


def check_expand(meta, received_expand_idx, ep, epr, ref_expand_idx=None) -> None:
    got = ep_dispatch_recv_idx(meta, received_expand_idx, ep, epr)
    ref = expand_reference(meta, received_expand_idx if ref_expand_idx is None else ref_expand_idx, ep, epr)  # fmt: skip
    assert got[0].dtype == got[1].dtype == torch.int64
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


# (ep_size, experts_per_rank) up to 2048 segments: a larger count leaves the H200's CuTe kernel
# for a Triton kernel that unrolls every segment (minutes to compile), which no model reaches.
EXPAND_GEOMETRIES = [
    (ep, epr)
    for ep in (1, 2, 4, 8, 16, 32)
    for epr in (1, 3, 5, 8, 32, 64, 128, 256)
    if ep * epr <= 2048
]


@pytest.mark.parametrize("ep,epr", EXPAND_GEOMETRIES)
def test_ep_dispatch_recv_idx(ep: int, epr: int) -> None:
    for seed, total in ((0, 3000), (1, 70000), (2, 1), (3, 1023), (4, 1025)):
        meta, received_expand_idx = recv_case(ep, epr, total, seed)
        check_expand(meta, received_expand_idx, ep, epr)


def test_ep_dispatch_recv_idx_empty() -> None:
    for ep, epr in ((4, 8), (1, 64), (8, 32)):
        meta = torch.zeros(ep * (epr + 1), dtype=torch.int64, device="cuda")
        received_expand_idx = torch.empty(0, dtype=torch.int64, device="cuda")
        expert_idxs, expand_idx = ep_dispatch_recv_idx(meta, received_expand_idx, ep, epr)
        assert expert_idxs.shape == expand_idx.shape == (0,) and expert_idxs.dtype == expand_idx.dtype == torch.int64  # fmt: skip


def test_ep_dispatch_recv_idx_sensitive() -> None:
    """
    One segment count off by one shifts later slots, and the comparison notices.
    """
    meta, received_expand_idx = recv_case(4, 32, 5000, 7, zero_frac=0.0)
    bad = meta.clone()
    bad[0] += 1
    bad[1] -= 1
    got = ep_dispatch_recv_idx(bad, received_expand_idx, 4, 32)
    assert not torch.equal(got[0], expand_reference(meta, received_expand_idx, 4, 32)[0])


def test_ep_dispatch_recv_idx_fallback_strided() -> None:
    """
    A non-contiguous received expand_idx, which the CuTe kernel does not cover.
    """
    for ep, epr, total in ((4, 8, 300), (8, 32, 70000)):
        meta, received_expand_idx = recv_case(ep, epr, total, 0)
        strided = torch.empty(2 * received_expand_idx.numel(), dtype=torch.int64, device="cuda")[::2]  # fmt: skip
        strided.copy_(received_expand_idx)
        assert not strided.is_contiguous()
        check_expand(meta, strided, ep, epr, ref_expand_idx=received_expand_idx)


@pytest.mark.parametrize("ne,k", MODELS)
@pytest.mark.parametrize("ep", [2, 4, 8])
def test_dispatch_round_trip(ne: int, k: int, ep: int) -> None:
    """
    ep_size senders with their own routing (ragged token counts, one empty): each receiver gathers
    the senders' send_meta rows and expand_idx segments for it, as the operator's all-to-alls do,
    and every received slot must name its sender's token (through the concatenated dedup chunks and
    the global expand_idx) and that token's expert.
    """
    epr = ne // ep
    tokens_per_sender = [0 if sender == 1 else 37 + 1000 * sender for sender in range(ep)]
    senders = []
    for sender, m in enumerate(tokens_per_sender):
        topk_ids = routing(m, k, ne, epr, kind="ragged" if sender % 2 else "random", seed=100 + sender)  # fmt: skip
        outs = run_dispatch(topk_ids, ep, epr)
        check_stable(outs, stable_reference(topk_ids, ep, epr))
        senders.append((topk_ids, outs))
    for recv_rank in range(ep):
        recv_meta = torch.cat([sender_outs[5].view(ep, epr + 1)[recv_rank] for _, sender_outs in senders])  # fmt: skip
        received_expand_idx, tokens, experts, dedup_tokens = [], [], [], []
        for sender, (topk_ids, sender_outs) in enumerate(senders):
            tokens_per_ep_rank, dedup, dispatch_token_idxs, idxs, expand, _ = sender_outs
            slot_start, slot_end = int(tokens_per_ep_rank[:recv_rank].sum()), int(tokens_per_ep_rank[: recv_rank + 1].sum())  # fmt: skip
            received_expand_idx.append(expand[slot_start:slot_end])
            # tag tokens with their sender
            tokens.append(idxs[slot_start:slot_end] // k + sender * 1_000_000)
            experts.append(topk_ids.reshape(-1)[idxs[slot_start:slot_end]] - recv_rank * epr)
            dedup_start, dedup_end = int(dedup[:recv_rank].sum()), int(dedup[: recv_rank + 1].sum())
            dedup_tokens.append(dispatch_token_idxs[dedup_start:dedup_end] + sender * 1_000_000)
        received_expand_idx = torch.cat(received_expand_idx)
        expert_idxs, expand_idx = ep_dispatch_recv_idx(recv_meta, received_expand_idx, ep, epr)
        assert torch.equal(expert_idxs, torch.cat(experts))
        assert torch.equal(torch.cat(dedup_tokens)[expand_idx], torch.cat(tokens))


# (tokens per sender, top-k, num_experts, ep_size)
SIM_CONFIGS = [
    (32, 8, 32, 8),
    (64, 4, 16, 4),
    (128, 8, 64, 8),
    (16, 2, 8, 4),
    (100, 8, 32, 4),
    (256, 8, 128, 16),
    (10, 4, 8, 2),
    (32, 1, 8, 4),  # k=1: no dedup
    (32, 4, 8, 2),
    (2, 2, 4, 2),
    (1, 2, 8, 4),  # single token
    (2048, 8, 128, 2),  # Qwen3-30B-A3B: many experts, few EP ranks
]


def randperm_routing(ms: int, k: int, num_experts: int) -> torch.Tensor:
    return torch.stack([torch.randperm(num_experts, device="cuda")[:k] for _ in range(ms)])


@pytest.mark.parametrize("ms,k,num_experts,ep_size", SIM_CONFIGS)
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_fused_dedup_dispatch(ms: int, k: int, num_experts: int, ep_size: int, seed: int) -> None:
    torch.manual_seed(seed)
    epr = num_experts // ep_size
    topk_ids = randperm_routing(ms, k, num_experts)
    check_stable(run_dispatch(topk_ids, ep_size, epr), stable_reference(topk_ids, ep_size, epr))


@pytest.mark.parametrize("ms,k,num_experts,ep_size", SIM_CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_fused_dedup_end_to_end(ms: int, k: int, num_experts: int, ep_size: int, seed: int) -> None:
    """
    Full sender -> receiver pipeline on hidden states: each receiver gathers the senders' dedup
    token chunks and expand_idx segments, makes expand_idx global through ep_dispatch_recv_idx,
    and the expanded rows equal each slot's own token row.
    """
    torch.manual_seed(seed)
    epr = num_experts // ep_size
    senders = []
    for _ in range(ep_size):
        hidden = torch.randn(ms, 64, device="cuda")
        ids = randperm_routing(ms, k, num_experts)
        tokens_per_ep_rank, dedup, dispatch_token_idxs, idxs, expand, send_meta = run_dispatch(ids, ep_size, epr)  # fmt: skip
        dedup_rows = hidden[dispatch_token_idxs[: int(dedup.sum())]]
        senders.append(
            {
                "meta": send_meta.view(ep_size, epr + 1),
                "dedup_chunks": dedup_rows.split(dedup.tolist()),
                "expand_chunks": expand.split(tokens_per_ep_rank.tolist()),
                "ref_chunks": hidden[idxs // k].split(tokens_per_ep_rank.tolist()),
            }
        )
    for recv_rank in range(ep_size):
        recv_meta = torch.cat([sender["meta"][recv_rank] for sender in senders])
        received_expand_idx = torch.cat([sender["expand_chunks"][recv_rank] for sender in senders])
        _, expand_idx = ep_dispatch_recv_idx(recv_meta, received_expand_idx, ep_size, epr)
        gathered = torch.cat([sender["dedup_chunks"][recv_rank] for sender in senders])
        if expand_idx.numel():
            assert int(expand_idx.min()) >= 0 and int(expand_idx.max()) < gathered.shape[0]
        reference = torch.cat([sender["ref_chunks"][recv_rank] for sender in senders])
        assert torch.equal(gathered[expand_idx], reference), f"GPU {recv_rank}: expanded != reference"  # fmt: skip


def test_fused_dedup_dispatch_m_zero() -> None:
    topk_ids = torch.empty((0, 4), dtype=torch.int64, device="cuda")
    outs = run_dispatch(topk_ids, 4, 4)
    assert int(outs[5].sum()) == 0 and outs[3].numel() == outs[4].numel() == 0


def recv_meta_of(tokens_per_expert: torch.Tensor, dedup: torch.Tensor) -> torch.Tensor:
    """
    Interleaved received metadata: per sender, its per-expert counts then its dedup count.
    """
    return torch.cat([tokens_per_expert, dedup[:, None]], dim=1).reshape(-1).contiguous()


@pytest.mark.parametrize("ms,k,num_experts,ep_size", SIM_CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_ep_dispatch_recv_idx_sim(
    ms: int, k: int, num_experts: int, ep_size: int, seed: int
) -> None:
    torch.manual_seed(seed)
    epr = num_experts // ep_size
    tokens_per_expert = torch.randint(0, max(ms, 2), (ep_size, epr), device="cuda")
    dedup = torch.randint(1, max(ms, 2) + 1, (ep_size,), device="cuda")
    splits = tokens_per_expert.sum(dim=1)
    total = int(splits.sum())
    sender = torch.arange(ep_size, device="cuda").repeat_interleave(splits)
    received_expand_idx = (torch.rand(total, device="cuda") * dedup[sender]).long()
    meta = recv_meta_of(tokens_per_expert, dedup)
    expert_idxs, _ = ep_dispatch_recv_idx(meta, received_expand_idx, ep_size, epr)
    assert expert_idxs.numel() == total, "one entry per received token slot"
    check_expand(meta, received_expand_idx, ep_size, epr)


def test_ep_dispatch_recv_idx_slot_count_from_input() -> None:
    """
    The slot count is the received expand_idx's length, decided by the senders, even where the
    metadata's segment counts add up to more: the outputs have exactly that many entries, those
    entries are right, and a tail not a multiple of any CTA size is handled.
    """
    ep_size, epr = 8, 16
    total = 5501
    tokens_per_expert = torch.full((ep_size, epr), total // (ep_size * epr), device="cuda")
    tokens_per_expert.view(-1)[: total % (ep_size * epr)] += 1
    assert int(tokens_per_expert.sum()) == total
    dedup = torch.full((ep_size,), 64, device="cuda")
    meta = recv_meta_of(tokens_per_expert, dedup)
    received_expand_idx = torch.randint(0, 64, (total,), device="cuda")
    check_expand(meta, received_expand_idx, ep_size, epr)
    ref_expert, ref_expand = expand_reference(meta, received_expand_idx, ep_size, epr)
    for short in (total - 1, 4096, 1000, 1):
        expert_idxs, expand_idx = ep_dispatch_recv_idx(meta, received_expand_idx[:short], ep_size, epr)  # fmt: skip
        assert expert_idxs.shape == expand_idx.shape == (short,)
        assert torch.equal(expert_idxs, ref_expert[:short]) and torch.equal(expand_idx, ref_expand[:short])  # fmt: skip
