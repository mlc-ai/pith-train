"""
The token movement, index math and small reductions around the routed experts of one EP rank, at
every expert shape of the spec (family moe_experts of tests/kernels/cases.py; the boundary cases
only matter to the BF16 GEMMs and are left out). Nothing here restates a shape: T, ep, E, top_k, G,
hidden, inter and the rows of every local expert (group_sizes) come from the case.

Routing. A case gives only the rows per local expert on the busiest rank. Each test synthesizes the
whole EP group's top-k routing from it: N = T * ep tokens (T per sender), each with top_k distinct
experts, where the G experts of rank 0 receive exactly group_sizes slots and the other E - G experts
share the remaining N * top_k - rows slots evenly (each at most N). The slots are laid out expert by
expert and dealt column-major over a (top_k, N) grid, so a run of one expert (at most N slots) never
hits a token twice; then the tokens and each token's k columns are shuffled. From it, in plain
PyTorch, the stage-3 input of rank 0 as the EP dispatch delivers it: at EP 1 the expanded slots in
token-major order (expert_idxs = topk_ids.view(-1), no expand_idx, no host counts); at EP > 1 the
received slots sender-major and expert-minor (token order within an expert), the deduplicated
received tokens (every token of a sender with a pick on rank 0, sender-major, token order), each
slot's row among them (expand_idx) and the host counts group_sizes. Its per-expert counts are
asserted to be group_sizes.

Checks (exact unless a bound is given):

- test_token_scatter (cases where training scatters, G > 64: the unfused BF16 path):
  scatter_by_expert as scatter_for_grouped_gemm calls it; every slot's row copied byte for byte into
  its expert's first count rows, reverse_shuffle_idxs a bijection onto them, padding and the
  allocation tail up to 1024 rows zero bytes, offsets and padded sizes exact, token order within an
  expert (the SM90 kernels place stably).
- test_scatter_quant (every case, the FP8 path): scatter_quant on the received tokens as
  fp8_scatter_quant calls it; placement as above, and the rowwise and per-channel FP8 bytes and
  scales byte-exact against tests/kernels/fp8_reference.py applied to the scattered bf16 rows
  (rebuilt through reverse_shuffle_idxs), padding included; then fp8_group_indices on its offsets
  (the hint precompute_group_indices takes): every grouped row's expert.
- test_bias_kernels (gpt-oss, biases): group_bias_add on the down output (M = padded rows, hidden)
  bit-identical to y + bias[group] (one fp32 add, one rounding); segment_sum of the gradient (hidden,
  and 2 * inter where the unfused path's indexed_bias_add runs) with the analytic bound of its fp32
  accumulation, |sum - exact| <= gamma_n * sum |x| with gamma_n = n u / (1 - n u), u = 2^-24, n =
  the group's 128-row tiles + 126 (the depth of any order inside a tile, then the tile fold), and
  the bf16 result exactly the fp32 result rounded once.
- test_clamped_swiglu (gpt-oss where training runs it, G > 64: gpt-oss-120b at EP 1): forward and
  backward on the padded grouped (M, 2 * inter) gate_up against float64 of the same bf16 inputs.
  The contract is fp32 arithmetic rounded once to bf16, so |out - exact| <= half a bf16 ulp of the
  exact value + 2 * REL * a, a the magnitude the fp32 evaluation works at (|exact| for the forward
  and the up gradient; |dh| |up_c + 1| (sigma + |alpha gate_c| sigma) for the gate gradient, whose
  factor 1 + alpha gate_c (1 - sigma) cancels near the swish minimum). The fp32 error REL has no
  tight analytic value (the sigmoid's exp argument, |alpha gate_c| up to 12, is itself rounded, so
  its relative error grows with it): REL = 2^-18, about 3x the measured 2^-19.8 (1.11e-6, the
  smallest REL every case passes; the same in all three outputs). Zero gradients exactly where
  clamped.
- test_ep_dispatch (EP > 1): ep_dispatch_send_idx on each sender's T tokens (meta_out as the
  operator passes it): counts and send_meta exact, the slot and token orders exact against a stable
  reference, and dispatch_token_idxs[rank start + expand_idx] == idxs // k; then ep_dispatch_recv_idx on rank 0's received metadata and
  expand_idx segments (the all-to-alls emulated in one process, no communication), exact against
  plain PyTorch, and end to end: per-expert counts == group_sizes, every received slot resolving to
  its sender's token and expert.
- test_bf16_expert_slots (cases with bf16_fused, where bf16_experts_supported(G)): the maps of
  bf16_expert_group_slots (cu_seqlens, the stable expert-major order, a_idx, row_of, slot_of, the
  token bitmask), bf16_expert_gather_rows (EP 1, no expand_idx) and bf16_expert_gather_sum (EP > 1,
  fp32 in ascending local-expert order from zero, rounded once) bit for bit, and for gpt-oss
  bf16_expert_segment_sum of grad_preact (2 * inter) and of grad_out read through slot_of (hidden)
  bit for bit against the contract's order (256-row chunks from each segment's start summed row by
  row in fp32, then the chunk sums in order).

Measured on the H200 over every smoke, spec and stress case (PITHTRAIN_SPEC_ERRORS records them),
as a fraction of the bound: segment_sum's fp32 error 0.0036 of its analytic bound (worst case,
gate_up width; 0.0029 at hidden); clamped_swiglu forward 0.998, gate gradient 0.998, up gradient
0.998, each dominated by its half-ulp term (the bf16 rounding itself), with the fp32 term at 1 /
3.4 of 2 * REL * a.
"""

import pytest
import torch

from pithtrain.kernels import (
    bf16_expert_gather_rows,
    bf16_expert_gather_sum,
    bf16_expert_group_slots,
    bf16_expert_segment_sum,
    bf16_experts_supported,
    clamped_swiglu_bwd,
    clamped_swiglu_fwd,
    ep_dispatch_recv_idx,
    ep_dispatch_send_idx,
    fp8_group_indices,
    group_bias_add,
    scatter_by_expert,
    scatter_quant,
    segment_sum,
)
from tests.kernels.cases import params, seed_of
from tests.kernels.fp8_reference import (
    assert_fp8_equal,
    assert_scale_equal,
    half_ulp_bf16,
    quant_rowwise,
)
from tests.kernels.measure import record

BLOCK = 128
CHUNK = 16384  # rows per reference chunk: a multiple of 128 and of 256
F64 = torch.float64
I64 = dict(device="cuda", dtype=torch.int64)
BF16 = dict(device="cuda", dtype=torch.bfloat16)
ALPHA, LIMIT = 1.702, 7.0  # gpt-oss clamped SwiGLU; asserted against the case
REL = 2.0**-18
U32 = 2.0**-24
REALISTIC = lambda c: c["axes"]["routing"] != "boundary"


@pytest.fixture(autouse=True)
def empty_cache():
    """
    Each test starts and ends with an empty allocator cache (the stress cases move GBs of rows).
    """
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()


def pad(count: int) -> int:
    return -(-count // BLOCK) * BLOCK


def cumsum_list(values: list[int]) -> list[int]:
    out = [0]
    for value in values:
        out.append(out[-1] + value)
    return out


def row_chunks(sizes: list[int], chunk: int = CHUNK):
    """
    (group, group start, chunk start, chunk end) over grouped rows, chunks inside one group.
    """
    start = 0
    for group, size in enumerate(sizes):
        for a in range(start, start + size, chunk):
            yield group, start, a, min(a + chunk, start + size)
        start += size


# Routing -----------------------------------------------------------------------------------------


def group_topk(case: dict, gen: torch.Generator) -> torch.Tensor:
    """
    (T * ep, top_k) int64 distinct expert ids per token for the whole EP group (module docstring):
    rank 0's experts 0 .. G - 1 receive exactly group_sizes slots.
    """
    a = case["axes"]
    T, ep, E, k, G = a["T"], a["ep"], a["E"], a["top_k"], a["G"]
    N, local = T * ep, a["group_sizes"]
    rest, others = N * k - sum(local), E - G
    if others == 0:
        assert rest == 0, (rest, case["id"])
        spread = []
    else:
        base, extra = divmod(rest, others)
        spread = [base + 1] * extra + [base] * (others - extra)
    counts = local + spread
    assert max(counts) <= N and sum(counts) == N * k, case["id"]
    ids = torch.repeat_interleave(torch.arange(E, **I64), torch.tensor(counts, **I64), output_size=N * k)  # fmt: skip
    topk = ids.view(k, N).T  # a run of one expert (<= N slots) hits distinct tokens
    topk = topk[torch.randperm(N, generator=gen, device="cuda")]
    cols = torch.rand(N, k, generator=gen, device="cuda").argsort(1)
    return topk.gather(1, cols).contiguous()


def receive(topk: torch.Tensor, T: int, G: int) -> dict:
    """
    Rank 0's received slots in plain PyTorch: expert (local), the global token of each slot, the
    slot's row among the deduplicated received tokens (expand), and their count.
    """
    N, _ = topk.shape
    local = topk < G
    token, col = local.nonzero(as_tuple=True)
    expert = topk[token, col]
    order = torch.argsort(((token // T) * G + expert) * N + token)  # sender, expert, token
    token, expert = token[order], expert[order]
    visit = local.any(1)
    dedup_pos = visit.long().cumsum(0) - 1
    return dict(expert=expert, token=token, expand=dedup_pos[token], num_tokens=int(visit.sum()))


def stage3_input(case: dict, gen: torch.Generator) -> dict:
    """
    What stage 3 of rank 0 hands the expert kernels: expert_idxs (S,), expand_idx (S,) or None, the
    source row count and the host counts (None at EP 1), as the dispatch delivers them.
    """
    a = case["axes"]
    T, ep, G = a["T"], a["ep"], a["G"]
    topk = group_topk(case, gen)
    if ep == 1:
        expert_idxs = topk.view(-1)
        out = dict(expert_idxs=expert_idxs, expand_idx=None, num_src=expert_idxs.numel(), group_counts=None)  # fmt: skip
    else:
        recv = receive(topk, T, G)
        out = dict(expert_idxs=recv["expert"], expand_idx=recv["expand"], num_src=recv["num_tokens"], group_counts=list(a["group_sizes"]))  # fmt: skip
    counts = torch.bincount(out["expert_idxs"], minlength=G)
    assert counts.tolist() == a["group_sizes"], case["id"]
    return out


def stable_rank(expert_idxs: torch.Tensor, G: int) -> torch.Tensor:
    """
    Each slot's position among the slots of its expert, in slot order.
    """
    order = torch.argsort(expert_idxs, stable=True)
    first = torch.searchsorted(expert_idxs[order], torch.arange(G, **I64))
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), **I64) - first[expert_idxs[order]]
    return rank


def check_placement(reverse: torch.Tensor, expert_idxs: torch.Tensor, counts: list[int], ks: list[int], M: int) -> None:  # fmt: skip
    """
    Each slot inside its expert's first count rows, no two slots on one row, and slots in token order
    within an expert (SM90 places stably).
    """
    assert reverse.dtype == torch.int64 and reverse.shape == expert_idxs.shape
    starts = torch.tensor(cumsum_list(ks)[:-1], **I64)
    pos = reverse - starts[expert_idxs]
    assert bool((pos >= 0).all()) and bool((pos < torch.tensor(counts, **I64)[expert_idxs]).all())
    if reverse.numel():
        assert int(torch.bincount(reverse, minlength=M).max()) <= 1, "two slots on one row"
    assert torch.equal(pos, stable_rank(expert_idxs, len(counts))), "not in slot order"


def check_layout(offs: torch.Tensor, ks_tensor: torch.Tensor, ks: list[int], counts: list[int]) -> None:  # fmt: skip
    want = [pad(c) for c in counts]
    assert ks == want
    assert ks_tensor.dtype == torch.int32 and ks_tensor.tolist() == want
    assert offs.dtype == torch.int32 and offs.tolist() == cumsum_list(want)[1:]


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


# Token scatter and the FP8 stage-3 input ---------------------------------------------------------


@pytest.mark.parametrize("case", params("moe_experts", keep=lambda c: REALISTIC(c) and not c["axes"]["bf16_fused"]))  # fmt: skip
def test_token_scatter(case: dict) -> None:
    a = case["axes"]
    G, H, counts = a["G"], a["hidden"], a["group_sizes"]
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    inp = stage3_input(case, gen)
    expert_idxs = inp["expert_idxs"]
    m = expert_idxs.numel()
    # the slots' rows, as padded_index_gather (EP > 1) or the top-k expansion (EP 1) gives them
    x = torch.randn(m, H, generator=gen, **BF16)
    ids_before = expert_idxs.clone()
    out, reverse, offs, ks_tensor, ks = scatter_by_expert(x, expert_idxs, G, BLOCK, inp["group_counts"])  # fmt: skip
    check_layout(offs, ks_tensor, ks, counts)
    M = sum(ks)
    assert M == a["fp8_rows"] and out.shape == (M, H) and out.dtype == x.dtype and out.stride() == (H, 1)  # fmt: skip
    check_placement(reverse, expert_idxs, counts, ks, M)
    for start in range(0, m, CHUNK):
        end = min(start + CHUNK, m)
        assert torch.equal(bits(out[reverse[start:end]]), bits(x[start:end])), "rows not copied exactly"  # fmt: skip
    padding = torch.ones(M, dtype=torch.bool, device="cuda")
    padding[reverse] = False
    assert not bool(bits(out[padding]).any()), "padding rows must be zero bytes"
    rows_alloc = -(-M // 1024) * 1024
    assert out.storage_offset() == 0 and out.untyped_storage().nbytes() >= rows_alloc * H * 2
    tail = torch.as_strided(out, (rows_alloc - M, H), (H, 1), M * H)
    assert not bool(bits(tail).any()), "allocation tail rows must be zero bytes"
    assert torch.equal(expert_idxs, ids_before), "expert_idxs must not change"


def received_tokens(rows: int, H: int, gen: torch.Generator) -> torch.Tensor:
    """
    bf16 tokens with per-row magnitudes from 2^-8 to 2^8 and some zero rows, so the block scales
    vary.
    """
    x = torch.randn(rows, H, generator=gen, device="cuda")
    x *= torch.exp2(torch.randint(-8, 9, (rows, 1), generator=gen, device="cuda").float())
    x[3::97] = 0
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("case", params("moe_experts", keep=REALISTIC))
def test_scatter_quant(case: dict) -> None:
    a = case["axes"]
    G, H, counts = a["G"], a["hidden"], a["group_sizes"]
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    inp = stage3_input(case, gen)
    expert_idxs, expand_idx = inp["expert_idxs"], inp["expand_idx"]
    gathered = received_tokens(inp["num_src"], H, gen)
    quantized, reverse, offs, ks, ks_tensor = scatter_quant(gathered, expert_idxs, G, expand_idx, group_counts=inp["group_counts"])  # fmt: skip
    check_layout(offs, ks_tensor, ks, counts)
    M = sum(ks)
    assert M == a["fp8_rows"]
    check_placement(reverse, expert_idxs, counts, ks, M)
    x_fp8, x_scale, x_ch, x_ch_scale = quantized
    assert x_fp8.shape == (M, H) and x_scale.shape == (M, -(-H // BLOCK))
    assert x_ch.shape == (M * H,) and x_ch_scale.shape == (H, M // BLOCK)  # K-major per group
    src = torch.full((M,), -1, **I64)
    src[reverse] = expand_idx if expand_idx is not None else torch.arange(expert_idxs.numel(), **I64)  # fmt: skip
    for group, group_start, start, end in row_chunks(ks):
        rows = src[start:end]
        dense = torch.zeros(end - start, H, **BF16)
        valid = rows >= 0
        dense[valid] = gathered[rows[valid]]
        q, s = quant_rowwise(dense)
        assert_fp8_equal(x_fp8[start:end], q, "x_fp8")
        assert_scale_equal(x_scale[start:end], s, "x_scale")
        q, s = quant_rowwise(dense.T)  # per channel: 128 rows x 1 column blocks
        size, first, last = ks[group], start - group_start, end - group_start
        data = x_ch[group_start * H : (group_start + size) * H].view(H, size)[:, first:last]
        scale = x_ch_scale[:, start // BLOCK : end // BLOCK]
        assert_fp8_equal(data, q, "x_ch_fp8")
        assert_scale_equal(scale, s, "x_ch_scale")
    hint = fp8_group_indices(offs, M)
    want = torch.repeat_interleave(torch.arange(G, device="cuda", dtype=torch.int32), ks_tensor, output_size=M)  # fmt: skip
    assert hint.dtype == torch.int32 and hint.shape == (M,) and torch.equal(hint, want)


# Expert biases (gpt-oss) -------------------------------------------------------------------------


def grouped_rows(counts: list[int], width: int, gen: torch.Generator, scale: float = 1.0) -> torch.Tensor:  # fmt: skip
    """
    bf16 (sum of padded counts, width): random rows for each group's real rows, zeros in its padding.
    """
    ks = [pad(c) for c in counts]
    x = torch.zeros(sum(ks), width, **BF16)
    for count, start in zip(counts, cumsum_list(ks)):
        if count:
            x[start : start + count] = torch.randn(count, width, generator=gen, device="cuda") * scale  # fmt: skip
    return x


def check_segment_sum(case: dict, what: str, x: torch.Tensor, ks: list[int]) -> None:
    offs = torch.tensor(cumsum_list(ks)[1:], device="cuda", dtype=torch.int32)
    G, n = len(ks), x.shape[1]
    got32 = segment_sum(x, offs, torch.float32)
    got16 = segment_sum(x, offs, torch.bfloat16)
    assert got32.shape == got16.shape == (G, n)
    assert got32.dtype == torch.float32 and got16.dtype == torch.bfloat16
    assert torch.equal(bits(got16), bits(got32.to(torch.bfloat16))), f"{what}: bf16 is not the fp32 sum rounded once"  # fmt: skip
    worst = 0.0
    for group, (size, start) in enumerate(zip(ks, cumsum_list(ks))):
        ref = torch.zeros(n, device="cuda", dtype=F64)
        absref = torch.zeros_like(ref)
        for a in range(start, start + size, CHUNK):
            part = x[a : min(a + CHUNK, start + size)].to(F64)
            ref += part.sum(0)
            absref += part.abs().sum(0)
        depth = -(-size // BLOCK) + 126
        gamma = depth * U32 / (1 - depth * U32)
        err = (got32[group].to(F64) - ref).abs()
        worst = max(worst, float((err / (gamma * absref + 1e-300)).max()))
    record(case["id"], f"segment_sum {what}", ratio=worst)
    assert worst <= 1, f"segment_sum {what}: error {worst:.3g}x the fp32 accumulation bound"


@pytest.mark.parametrize("case", params("moe_experts", keep=lambda c: REALISTIC(c) and c["axes"]["bias"]))  # fmt: skip
def test_bias_kernels(case: dict) -> None:
    a = case["axes"]
    G, H, inter, counts = a["G"], a["hidden"], a["inter"], a["group_sizes"]
    ks = [pad(c) for c in counts]
    M = sum(ks)
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    groups = torch.repeat_interleave(torch.arange(G, device="cuda", dtype=torch.int32), torch.tensor(ks, device="cuda"), output_size=M)  # fmt: skip
    y = grouped_rows(counts, H, gen, scale=4.0)
    bias = torch.randn(G, H, generator=gen, **BF16)
    before = y.clone()
    assert group_bias_add(y, bias, groups) is None
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        want = (before[start:end].float() + bias[groups[start:end].long()].float()).to(torch.bfloat16)  # fmt: skip
        assert torch.equal(bits(y[start:end]), bits(want)), "group_bias_add is not y + bias[group]"
    del y, before
    # d_bias_down (the fused FP8 experts' backward), and on the unfused path d_bias_gate_up too
    widths = [("down", H)] + ([("gate_up", 2 * inter)] if not a["bf16_fused"] else [])
    for what, width in widths:
        check_segment_sum(case, what, grouped_rows(counts, width, gen), ks)


# Clamped SwiGLU (gpt-oss, unfused path) ----------------------------------------------------------


@pytest.mark.parametrize("case", params("moe_experts", keep=lambda c: REALISTIC(c) and c["axes"]["act"] == "clamped_swiglu" and not c["axes"]["bf16_fused"]))  # fmt: skip
def test_clamped_swiglu(case: dict) -> None:
    a = case["axes"]
    assert (a["swiglu_alpha"], a["swiglu_limit"]) == (ALPHA, LIMIT)
    inter, counts = a["inter"], a["group_sizes"]
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    gate_up = grouped_rows(counts, 2 * inter, gen, scale=4.0)  # reaches the clamps (7.0 is exact)
    dh = grouped_rows(counts, inter, gen)
    M = gate_up.shape[0]
    assert M == a["fp8_rows"]
    out = clamped_swiglu_fwd(gate_up, ALPHA, LIMIT)
    grad = clamped_swiglu_bwd(dh, gate_up, ALPHA, LIMIT)
    assert out.shape == (M, inter) and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert grad.shape == (M, 2 * inter) and grad.dtype == torch.bfloat16
    worst = dict(fwd=0.0, dgate=0.0, dup=0.0)
    needed = dict(fwd=0.0, dgate=0.0, dup=0.0)  # the smallest REL that would pass, for calibration

    def check(name, got, ref, scale):
        err, half = (got.to(F64) - ref).abs(), half_ulp_bf16(ref)
        worst[name] = max(worst[name], float((err / (half + 2 * REL * scale)).max()))
        needed[name] = max(needed[name], float(((err - half).clamp(min=0) / (2 * scale + 1e-300)).max()))  # fmt: skip

    for start in range(0, M, CHUNK // 2):
        end = min(start + CHUNK // 2, M)
        g, u, d = gate_up[start:end, 0::2].to(F64), gate_up[start:end, 1::2].to(F64), dh[start:end].to(F64)  # fmt: skip
        gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
        sig = torch.sigmoid(ALPHA * gc)
        h = (uc + 1) * gc * sig
        check("fwd", out[start:end], h, h.abs())
        dgate = d * (uc + 1) * sig * (1 + ALPHA * gc * (1 - sig)) * (g <= LIMIT)
        dup = d * gc * sig * (u.abs() <= LIMIT)
        check("dgate", grad[start:end, 0::2], dgate, d.abs() * (uc + 1).abs() * sig * (1 + ALPHA * gc.abs()))  # fmt: skip
        check("dup", grad[start:end, 1::2], dup, dup.abs())
        assert not bool(bits(grad[start:end, 0::2][g > LIMIT]).any()), (
            "gate gradient past the limit"
        )
        assert not bool(bits(grad[start:end, 1::2][u.abs() > LIMIT]).any()), "up gradient past the limit"  # fmt: skip
    for name, value in worst.items():
        record(case["id"], f"clamped_swiglu {name}", ratio=value, rel_needed=needed[name])
    assert bool((gate_up[:, 0::2].float() > LIMIT).any()), "the clamp must be exercised"
    for name, value in worst.items():
        assert value <= 1, f"clamped_swiglu {name}: error {value:.3g}x the bound"


# EP dispatch index math (single GPU) -------------------------------------------------------------


def send_reference(topk_ids: torch.Tensor, ep: int, epr: int) -> dict:
    """
    Every sender-side output of ep_dispatch_send_idx, placed stably (token order in each bucket).
    """
    m, k = topk_ids.shape
    flat = topk_ids.reshape(-1)
    idxs = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ep * epr)
    visit = torch.zeros(m, ep, dtype=torch.bool, device="cuda")
    visit.scatter_(1, topk_ids // epr, True)
    dedup = visit.sum(0)
    position = visit.long().cumsum(0) - 1  # of a token within its rank's dedup chunk
    return dict(
        tokens_per_ep_rank=counts.view(ep, epr).sum(1),
        dedup=dedup,
        dispatch=visit.T.nonzero()[:, 1],  # rank-major, token order within a rank
        idxs=idxs,
        expand=position[idxs // k, flat[idxs] // epr],
        send_meta=torch.cat([counts.view(ep, epr), dedup[:, None]], 1).reshape(-1),
    )


@pytest.mark.parametrize("case", params("moe_experts", keep=lambda c: REALISTIC(c) and c["axes"]["ep"] > 1))  # fmt: skip
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
        # what rank 0 receives from this sender (the operator's all-to-alls)
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
    # end to end: rank 0's experts get group_sizes, each slot its sender's token and expert
    recv = receive(topk, T, G)
    assert torch.bincount(expert_idxs, minlength=G).tolist() == a["group_sizes"]
    assert torch.equal(expert_idxs, recv["expert"])  # sender-major, expert-minor
    token = dedup_tokens[expand_idx]
    assert torch.equal(torch.sort(token * G + expert_idxs).values, torch.sort(recv["token"] * G + recv["expert"]).values)  # fmt: skip
    assert bool((topk[token] == expert_idxs[:, None]).any(1).all())
    assert torch.equal(token, recv["token"]) and torch.equal(expand_idx, recv["expand"])


# BF16 fused experts: slot maps, gathers, segment sums --------------------------------------------


def segment_sum_order(x: torch.Tensor, cu_host: list[int], idx: torch.Tensor | None) -> torch.Tensor:  # fmt: skip
    """
    bf16_expert_segment_sum's documented order in plain PyTorch: per segment, 256-row chunks from
    its start each summed row by row in fp32 from zero, then the chunk sums added in order.
    """
    G, n = len(cu_host) - 1, x.shape[1]
    out = torch.zeros(G, n, device="cuda", dtype=torch.float32)
    for group in range(G):
        start, end = cu_host[group], cu_host[group + 1]
        total = torch.zeros(n, device="cuda", dtype=torch.float32)
        for a in range(start, end, CHUNK):
            b = min(a + CHUNK, end)
            rows = x[idx[a:b].long()] if idx is not None else x[a:b]
            chunks = -(-(b - a) // 256)
            padded = torch.zeros(chunks * 256, n, device="cuda", dtype=torch.float32)
            padded[: b - a] = rows.float()
            padded = padded.view(chunks, 256, n)
            acc = torch.zeros(chunks, n, device="cuda", dtype=torch.float32)
            for row in range(256):
                acc += padded[:, row]
            for chunk in range(chunks):
                total += acc[chunk]
        out[group] = total
    return out


@pytest.mark.parametrize("case", params("moe_experts", keep=lambda c: REALISTIC(c) and c["axes"]["bf16_fused"]))  # fmt: skip
def test_bf16_expert_slots(case: dict) -> None:
    a = case["axes"]
    G, H, inter, k, counts = a["G"], a["hidden"], a["inter"], a["top_k"], a["group_sizes"]
    if not bf16_experts_supported(G):
        pytest.skip(f"bf16_experts_supported({G}) is False here")
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    inp = stage3_input(case, gen)
    expert_idxs, expand_idx, num_tokens = inp["expert_idxs"], inp["expand_idx"], inp["num_src"]
    S = expert_idxs.numel()
    maps = bf16_expert_group_slots(expert_idxs, expand_idx, G, num_tokens)
    cu_host = cumsum_list(counts)
    assert maps.cu_seqlens.dtype == torch.int32 and maps.cu_seqlens.tolist() == cu_host
    order = torch.argsort(expert_idxs, stable=True)  # grouped row r holds slot order[r]
    row_of = torch.empty_like(order)
    row_of[order] = torch.arange(S, **I64)
    src = expand_idx if expand_idx is not None else torch.arange(S, **I64)
    for name, want in (("slot_of", order), ("row_of", row_of), ("a_idx", src[order])):
        got = getattr(maps, name)
        assert got.dtype == torch.int32 and got.shape == (S,) and torch.equal(got.long(), want), name  # fmt: skip
    if expand_idx is None:
        assert maps.mask is None
        grad_x = torch.randn(S, H, generator=gen, **BF16)
        out = bf16_expert_gather_rows(grad_x, maps.row_of)
        assert out.shape == (S, H) and out.dtype == torch.bfloat16
        for start in range(0, S, CHUNK):
            end = min(start + CHUNK, S)
            assert torch.equal(bits(out[start:end]), bits(grad_x[row_of[start:end]])), "gather_rows"
    else:
        mask = torch.zeros(num_tokens, **I64)
        mask.index_add_(0, expand_idx, torch.ones_like(expert_idxs) << expert_idxs)
        assert maps.mask.shape == (num_tokens,) and torch.equal(maps.mask, mask)
        max_slots = min(k, G)
        grad = torch.randn(S, H, generator=gen, **BF16)
        grad[::7] *= 1000  # a wide dynamic range, so the summation order shows
        out = bf16_expert_gather_sum(grad, maps, G, max_slots)
        assert out.shape == (num_tokens, H) and out.dtype == torch.bfloat16
        # each token's grouped rows in ascending local expert, padded with -1 to max_slots
        by_token = torch.argsort(expand_idx * G + expert_idxs)
        token = expand_idx[by_token]
        per_token = torch.bincount(expand_idx, minlength=num_tokens)
        slot = torch.arange(S, **I64) - (per_token.cumsum(0) - per_token)[token]
        assert int(per_token.max()) <= max_slots
        table = torch.full((num_tokens, max_slots), -1, **I64)
        table[token, slot] = row_of[by_token]
        for start in range(0, num_tokens, CHUNK):
            end = min(start + CHUNK, num_tokens)
            acc = torch.zeros(end - start, H, device="cuda", dtype=torch.float32)
            for j in range(max_slots):
                rows = table[start:end, j]
                valid = rows >= 0
                acc[valid] = acc[valid] + grad[rows[valid]].float()
            assert torch.equal(bits(out[start:end]), bits(acc.to(torch.bfloat16))), "gather_sum"
    if a["bias"]:
        cu = maps.cu_seqlens
        grad_preact = torch.randn(S, 2 * inter, generator=gen, **BF16)
        got = bf16_expert_segment_sum(grad_preact, cu_host, cu)
        assert got.dtype == torch.float32 and got.shape == (G, 2 * inter)
        assert torch.equal(got, segment_sum_order(grad_preact, cu_host, None)), "segment_sum gate_up"  # fmt: skip
        del grad_preact
        grad_out = torch.randn(S, H, generator=gen, **BF16)
        got = bf16_expert_segment_sum(grad_out, cu_host, cu, maps.slot_of)
        assert got.dtype == torch.float32 and got.shape == (G, H)
        assert torch.equal(got, segment_sum_order(grad_out, cu_host, order)), "segment_sum down"
