"""
Verifies the slot maps and row reductions of the BF16 CuTe experts (SM90 only) against PyTorch
bit for bit.

- bf16_expert_group_slots: cu_seqlens holds the expert offsets, grouped rows are expert-major and
  stable (slot order within an expert), a_idx / row_of / slot_of agree, and repeated calls match.
  Routings are laid out as the EP-dedup dispatch delivers them (slots sender-major, expert-minor,
  a token repeated once per local expert it chose), with empty experts, expand_idx None or not,
  0 slots and 0 tokens, for 8 / 16 / 32 / 64 local experts.
- bf16_expert_gather_sum: each token's grouped rows summed in fp32 in ascending expert order and
  rounded once to bf16.
- bf16_expert_gather_rows: x[idx] (the no-expand input gradient dx[row_of]).
- bf16_expert_segment_sum: per-segment fp32 row sums, 256-row chunks summed in row order and then
  the chunk sums in order, as the contract fixes.
"""

import pytest
import torch

from pithtrain.kernels import (
    bf16_expert_gather_rows,
    bf16_expert_gather_sum,
    bf16_expert_group_slots,
    bf16_expert_segment_sum,
    bf16_experts_supported,
)

pytestmark = pytest.mark.skipif(
    not bf16_experts_supported(),
    reason="needs the fused BF16 CuTe experts (bf16_experts_supported)",
)


def ep_routing(G: int, ep: int, tokens_per_sender: int, k: int, empty=(), seed=0):
    """
    (expert_idxs, expand_idx, T): per sender, each token picks 1..k of the G local experts (none
    of empty), slots sender-major and expert-minor. expand_idx indexes the concatenated deduplicated
    tokens of all senders.
    """
    gen = torch.Generator().manual_seed(seed)
    allowed = torch.tensor([expert for expert in range(G) if expert not in empty])
    expert_list, expand_list, token_base = [], [], 0
    for _ in range(ep):
        num_chosen = torch.randint(1, k + 1, (tokens_per_sender,), generator=gen)
        choice = []
        for token in range(tokens_per_sender):
            perm = torch.randperm(len(allowed), generator=gen)[: int(num_chosen[token])]
            choice += [(expert, token) for expert in allowed[perm].tolist()]
        choice.sort()
        expert_list += [expert for expert, _ in choice]
        expand_list += [token_base + token for _, token in choice]
        token_base += tokens_per_sender
    cuda = dict(device="cuda", dtype=torch.int64)
    return torch.tensor(expert_list, **cuda), torch.tensor(expand_list, **cuda), token_base


def check_maps(maps, expert_idxs, expand_idx, G, T):
    S = expert_idxs.shape[0]
    slot_range = torch.arange(S, device="cuda")
    counts = torch.bincount(expert_idxs, minlength=G)
    cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    assert maps.cu_seqlens.dtype == torch.int32
    assert maps.cu_seqlens.tolist() == cu.tolist()
    for index_map in (maps.a_idx, maps.row_of, maps.slot_of):
        assert index_map.dtype == torch.int32 and index_map.shape == (S,)
    # expert-major and stable: grouped row r holds the r-th slot of a stable sort by expert
    order = torch.argsort(expert_idxs, stable=True)
    assert torch.equal(maps.slot_of.long(), order)
    rows = maps.row_of.long()
    assert torch.equal(torch.sort(rows).values, slot_range)
    assert torch.equal(maps.slot_of.long()[rows], slot_range)
    src = expand_idx if expand_idx is not None else slot_range
    assert torch.equal(maps.a_idx.long()[rows], src)
    assert torch.equal(maps.a_idx.long(), src[order])
    if expand_idx is None:
        assert maps.mask is None
    else:
        # the token bitmask of its local experts
        ref = torch.zeros(T, dtype=torch.int64, device="cuda")
        ref.index_add_(0, expand_idx, torch.ones_like(expert_idxs) << expert_idxs)
        assert maps.mask.shape == (T,) and torch.equal(maps.mask, ref)


ROUTINGS = [
    # G, ep, tokens per sender, k, empty experts
    (16, 8, 1000, 6, (3, 7)),  # qwen3 at EP 8
    (32, 4, 2048, 8, ()),  # qwen3 at EP 4
    (64, 2, 1500, 8, (0, 63)),  # qwen3 at EP 2
    (64, 4, 300, 8, tuple(range(10, 40))),  # many empty experts
    (8, 2, 3, 4, ()),  # a few slots (less than one chunk)
    (32, 1, 4096, 4, ()),  # gpt-oss, EP 1 (expand still repeats tokens)
]


@pytest.mark.parametrize("G,ep,tps,k,empty", ROUTINGS)
def test_group_slots(G, ep, tps, k, empty):
    expert_idxs, expand_idx, T = ep_routing(G, ep, tps, k, empty)
    maps = bf16_expert_group_slots(expert_idxs, expand_idx, G, T)
    check_maps(maps, expert_idxs, expand_idx, G, T)
    again = bf16_expert_group_slots(expert_idxs, expand_idx, G, T)
    for name in ("cu_seqlens", "a_idx", "row_of", "slot_of", "mask"):
        assert torch.equal(getattr(maps, name), getattr(again, name)), name


@pytest.mark.parametrize("G,S", [(16, 5000), (64, 70000), (8, 1), (32, 2047), (32, 2049)])
def test_group_slots_without_expand(G, S):
    gen = torch.Generator().manual_seed(S)
    expert_idxs = torch.randint(0, G, (S,), generator=gen).cuda()
    maps = bf16_expert_group_slots(expert_idxs, None, G, S)
    check_maps(maps, expert_idxs, None, G, S)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("T", [0, 5])
def test_group_slots_no_slots(expand, T):
    no_slots = torch.empty(0, dtype=torch.int64, device="cuda")
    maps = bf16_expert_group_slots(no_slots, no_slots if expand else None, 16, T)
    assert maps.cu_seqlens.tolist() == [0] * 17
    assert maps.a_idx.shape == maps.row_of.shape == maps.slot_of.shape == (0,)
    if expand:
        assert maps.mask.shape == (T,) and not maps.mask.any()
        # no slots: every token's gradient is the empty sum
        out = bf16_expert_gather_sum(torch.empty(0, 2048, **BF16), maps, 16, 4)
        assert out.shape == (T, 2048) and not out.any()
    else:
        assert maps.mask is None


BF16 = dict(device="cuda", dtype=torch.bfloat16)


def gather_sum_reference(grad, expert_idxs, expand_idx, row_of, T):
    out = torch.zeros(T, grad.shape[1], dtype=torch.float32, device=grad.device)
    for expert in torch.unique(expert_idxs).tolist():  # ascending
        # a token chose each expert at most once, so index_add_ adds one row per token
        slots = (expert_idxs == expert).nonzero().squeeze(1)
        out.index_add_(0, expand_idx[slots], grad[row_of[slots].long()].float())
    return out.to(grad.dtype)


@pytest.mark.parametrize(
    "G,ep,tps,k,H",
    [
        (16, 8, 600, 8, 2048),  # qwen3 / qwen3.5 / deepseek-v2-lite hidden
        (32, 4, 2048, 8, 2048),
        (64, 2, 700, 8, 2048),
        (8, 4, 1500, 4, 2880),  # gpt-oss hidden
        (32, 1, 1000, 4, 2880),
        (4, 1, 3, 2, 8),  # tiny rows, fewer tokens than one CTA's share
    ],
)
def test_gather_sum(G, ep, tps, k, H):
    expert_idxs, expand_idx, T = ep_routing(G, ep, tps, k, seed=2)
    maps = bf16_expert_group_slots(expert_idxs, expand_idx, G, T)
    torch.manual_seed(0)
    grad = torch.randn(expert_idxs.shape[0], H, **BF16)
    grad[::7] *= 1000  # wide dynamic range, so the summation order shows
    out = bf16_expert_gather_sum(grad, maps, G, min(k, G))
    ref = gather_sum_reference(grad, expert_idxs, expand_idx, maps.row_of, T)
    assert out.dtype == torch.bfloat16 and out.shape == (T, H)
    assert torch.equal(out, ref)
    assert torch.equal(out, bf16_expert_gather_sum(grad, maps, G, min(k, G)))


def test_gather_sum_zero_tokens():
    no_slots = torch.empty(0, dtype=torch.int64, device="cuda")
    maps = bf16_expert_group_slots(no_slots, no_slots, 16, 0)
    out = bf16_expert_gather_sum(torch.empty(0, 2048, **BF16), maps, 16, 8)
    assert out.shape == (0, 2048)


@pytest.mark.parametrize("N", [2048, 2880, 1536, 8])
@pytest.mark.parametrize("rows", [0, 1, 5001])
def test_gather_rows(N, rows):
    torch.manual_seed(N)
    x = torch.randn(3000, N, **BF16)
    idx = torch.randint(0, 3000, (rows,), device="cuda", dtype=torch.int32)
    out = bf16_expert_gather_rows(x, idx)
    assert out.shape == (rows, N) and out.dtype == torch.bfloat16
    assert torch.equal(out, x[idx.long()])


def test_gather_rows_no_expand_input_grad():
    G, S = 16, 4000
    expert_idxs = torch.randint(0, G, (S,), device="cuda")
    maps = bf16_expert_group_slots(expert_idxs, None, G, S)
    dx = torch.randn(S, 2048, **BF16)
    out = bf16_expert_gather_rows(dx, maps.row_of)
    ref = torch.empty_like(dx)
    ref[maps.slot_of.long()] = dx  # grouped row r belongs to slot slot_of[r]
    assert torch.equal(out, ref)


def segment_sum_reference(x, counts, idx=None):
    G, N = len(counts), x.shape[1]
    rows = x[idx.long()] if idx is not None else x
    out = torch.zeros(G, N, dtype=torch.float32, device=x.device)
    row_start = 0
    for group_idx, count in enumerate(counts):
        segment = rows[row_start : row_start + count].float()
        row_start += count
        num_chunks = -(-count // 256)
        if num_chunks == 0:
            continue
        padded = segment.new_zeros(num_chunks * 256, N)
        padded[:count] = segment
        padded = padded.view(num_chunks, 256, N)
        acc_chunk = torch.zeros(num_chunks, N, dtype=torch.float32, device=x.device)
        for row_idx in range(256):
            acc_chunk += padded[:, row_idx]
        acc_sum = torch.zeros(N, dtype=torch.float32, device=x.device)
        for chunk_idx in range(num_chunks):
            acc_sum += acc_chunk[chunk_idx]
        out[group_idx] = acc_sum
    return out


SEGMENTS = {
    "gpt-oss-ragged": ([700, 0, 1, 2048, 300, 129, 900, 64], 5760),  # db_gate_up (2I)
    "gpt-oss-down": ([700, 0, 1, 2048, 300, 129, 900, 64], 2880),  # db_down (H)
    "gpt-oss-32": ([256] * 16 + [0] * 8 + [1000] * 8, 2880),
    "qwen3": ([0, 1, 300, 6000, 255, 256, 257, 0], 2048),
    "tiny": ([3, 0, 5], 8),
    "all-empty": ([0, 0, 0, 0], 2048),
}


@pytest.mark.parametrize("case", list(SEGMENTS))
@pytest.mark.parametrize("gather", [False, True])
def test_segment_sum(case, gather):
    counts, N = SEGMENTS[case]
    M = sum(counts)
    torch.manual_seed(M + N)
    x = torch.randn(M + 17, N, **BF16)
    x[::5] *= 300
    idx = torch.randperm(M + 17, device="cuda")[:M].to(torch.int32) if gather else None
    cu_host = [0]
    for count in counts:
        cu_host.append(cu_host[-1] + count)
    cu = torch.tensor(cu_host, dtype=torch.int32, device="cuda")
    out = bf16_expert_segment_sum(x if gather else x[:M], cu_host, cu, idx)
    ref = segment_sum_reference(x if gather else x[:M], counts, idx)
    assert out.dtype == torch.float32 and out.shape == (len(counts), N)
    assert torch.equal(out, ref)
    assert torch.equal(out, bf16_expert_segment_sum(x if gather else x[:M], cu_host, cu, idx))
