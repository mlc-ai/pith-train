"""
Contract: expand the deduplicated rows (gathered[expand_idx]), scatter the tokens into per-expert
groups padded with zero rows to a multiple of 128, and quantize the result as fp8_quant_grouped
does on SM90 (rowwise 1x128 plus the per-channel FP8 K-major per group). Returns (quantized 4-tuple,
reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor).

Row placement within a group is the kernel's choice, so reverse_shuffle_idxs only has to be a valid
placement: each token inside its expert's rows, all rows distinct. The reference scatters the
tokens through it into a zero bf16 matrix and quantizes that in PyTorch; every quantized output,
padding rows included (zero bytes), must match exactly. On SM90 the placement is also checked to
keep token order within each expert.

Recipe: per 128-element block, scale = 2^ceil(log2(max(amax, 1e-4) / 448)), at least 2^-126, and
fp8 = e4m3_satfinite(x * (1 / scale)), rounded to nearest even.
"""

import pytest
import torch

from pithtrain.kernels import fp8_fused_experts_supported, scatter_quant

pytestmark = pytest.mark.skipif(
    not fp8_fused_experts_supported(),
    reason="needs the fused FP8 expert path (fp8_fused_experts_supported)",
)

BLOCK = 128


def e8m0_scale_rcp(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the recipe's power-of-two scale of an fp32 amax, and its exact reciprocal.
    """
    clamped = torch.maximum(amax, torch.full_like(amax, 1e-4))
    ratio = clamped * torch.full_like(clamped, 1.0 / 448.0)
    bits = ratio.view(torch.int32)
    bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    scale = torch.maximum(bits.view(torch.float32), torch.full_like(clamped, 2.0**-126))
    rcp = ((254 << 23) - scale.view(torch.int32)).view(torch.float32)
    return scale, rcp


def quantize_bytes(x: torch.Tensor, rcp: torch.Tensor) -> torch.Tensor:
    return (x * rcp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def quantize_grouped_ref(x: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Return fp8_quant_grouped on SM90 in PyTorch: x_fp8 (M, K) as uint8, x_scale (M,
    ceil(K/128)), x_ch flat (M*K,) uint8 K-major per group, and x_ch_scale (K, M/128).
    """
    M, K = x.shape
    num_blocks = -(-K // BLOCK)
    x_pad = torch.zeros(M, num_blocks * BLOCK, dtype=torch.float32, device=x.device)
    x_pad[:, :K] = x.float()
    scale_row, rcp_row = e8m0_scale_rcp(x_pad.abs().view(M, num_blocks, BLOCK).amax(-1))
    fp8_row = quantize_bytes(x_pad.view(M, num_blocks, BLOCK), rcp_row[..., None]).view(M, num_blocks * BLOCK)[:, :K]  # fmt: skip
    x_col = x_pad[:, :K].reshape(M // BLOCK, BLOCK, K)
    scale_ch, rcp_ch = e8m0_scale_rcp(x_col.abs().amax(1))
    fp8_ch = quantize_bytes(x_col, rcp_ch[:, None, :]).view(M, K)
    flat = torch.zeros(M * K, dtype=torch.uint8, device=x.device)
    start = 0
    for end in offs.tolist():
        flat[start * K : end * K].view(K, end - start).copy_(fp8_ch[start:end].T)
        start = end
    return fp8_row.contiguous(), scale_row, flat, scale_ch.T.contiguous()


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def make_routing(
    sizes: list[int], dedup: int, segmented: bool, seed: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Return expert_idxs with the given per-expert sizes, an expand_idx with about dedup slots per
    source row, and the source row count. segmented mimics EP dispatch (runs per sender);
    otherwise the order is random.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    expert_idxs = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    num_tokens = expert_idxs.numel()
    if segmented:
        sender = torch.randint(0, 8, (num_tokens,), generator=gen)
        expert_idxs = expert_idxs[torch.argsort(sender * len(sizes) + expert_idxs, stable=True)]
    else:
        expert_idxs = expert_idxs[torch.randperm(num_tokens, generator=gen)]
    m_src = max(1, (num_tokens + dedup - 1) // dedup)
    expand = torch.randint(0, m_src, (num_tokens,), generator=gen)
    # Every source row is used at least once.
    expand[: min(num_tokens, m_src)] = torch.arange(min(num_tokens, m_src))
    expand = expand[torch.randperm(num_tokens, generator=gen)]
    return expert_idxs.cuda(), expand.cuda(), m_src


def make_tokens(rows: int, N: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, N, device="cuda", generator=gen)
    row_scale = torch.exp2(torch.randint(-12, 12, (rows, 1), device="cuda", generator=gen).float())
    x = x * row_scale
    mask = torch.rand(rows, N, device="cuda", generator=gen) < 1e-3
    x = torch.where(mask, x * 1e4, x)
    x[3::97] = 0.0
    x[:, : min(N, 3)] = 0.0
    x[5::101] = -x[5::101].abs()
    return x.to(torch.bfloat16)


def ref_group_layout(expert_idxs: torch.Tensor, G: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:  # fmt: skip
    counts = torch.bincount(expert_idxs, minlength=G)
    ks_t = ((counts + BLOCK - 1) // BLOCK * BLOCK).to(torch.int32)
    return ks_t.cumsum(0).to(torch.int32), ks_t, ks_t.tolist()


def check(
    gathered: torch.Tensor,
    expert_idxs: torch.Tensor,
    G: int,
    expand: torch.Tensor | None,
    counts: list[int] | None,
    out: tuple,
) -> None:
    quantized, reverse_idxs, offs, ks, ks_tensor = out
    num_tokens, N = expert_idxs.shape[0], gathered.shape[1]
    x = gathered if expand is None else gathered[expand]

    ref_offs, ref_ks_tensor, ref_ks = ref_group_layout(expert_idxs, G)
    assert offs.dtype == torch.int32 and torch.equal(offs, ref_offs)
    assert ks_tensor.dtype == torch.int32 and torch.equal(ks_tensor, ref_ks_tensor)
    assert isinstance(ks, list) and ks == ref_ks
    if counts is not None:
        assert ks == [(count + BLOCK - 1) // BLOCK * BLOCK for count in counts]
    M = sum(ref_ks)

    assert reverse_idxs.dtype == torch.int64 and reverse_idxs.shape == (num_tokens,)
    starts = (ref_offs - ref_ks_tensor).long()
    counts_t = torch.bincount(expert_idxs, minlength=G)
    rel = reverse_idxs - starts[expert_idxs]
    assert bool((rel >= 0).all()) and bool((rel < counts_t[expert_idxs]).all())
    assert torch.unique(reverse_idxs).numel() == num_tokens

    dense = torch.zeros(M, N, dtype=torch.bfloat16, device=gathered.device)
    dense[reverse_idxs] = x
    ref = quantize_grouped_ref(dense, ref_offs)
    shapes = [(M, N), (M, -(-N // BLOCK)), (M * N,), (N, M // BLOCK)]
    names = ("x_fp8", "x_scale", "x_ch_fp8", "x_ch_scale")
    for name, got, want, shape in zip(names, quantized, ref, shapes):
        assert tuple(got.shape) == shape, (name, got.shape, shape)
        assert got.dtype == (torch.float8_e4m3fn if name.endswith("fp8") else torch.float32), name
        if not torch.equal(as_bytes(got), want):
            raise AssertionError(f"{name}: {(as_bytes(got) != want).sum().item()} mismatches")
    pad = torch.ones(M, dtype=torch.bool, device=gathered.device)
    pad[reverse_idxs] = False
    assert not bool(as_bytes(quantized[0])[pad].any()), "padding rows must be zero bytes"


def check_stable(expert_idxs: torch.Tensor, G: int, reverse_idxs: torch.Tensor, offs: torch.Tensor) -> None:  # fmt: skip
    """
    Check that tokens keep their order within each expert (the SM90 placement).
    """
    ks_tensor = torch.diff(offs, prepend=offs.new_zeros(1))
    starts = (offs - ks_tensor).long()
    order = torch.argsort(expert_idxs, stable=True)
    rank = torch.empty_like(order)
    pos = torch.arange(expert_idxs.numel(), device=expert_idxs.device)
    first = torch.searchsorted(expert_idxs[order], torch.arange(G, device=expert_idxs.device))
    rank[order] = pos - first[expert_idxs[order]]
    assert torch.equal(reverse_idxs, starts[expert_idxs] + rank)


def run_case(sizes: list[int], N: int, dedup: int, segmented: bool, host_counts: bool) -> None:
    G = len(sizes)
    seed = sum(sizes) * 7 + N + dedup
    expert_idxs, expand, m_src = make_routing(sizes, dedup, segmented, seed)
    use_expand = dedup > 1 or segmented
    counts = sizes if host_counts else None
    if use_expand:
        gathered = make_tokens(m_src, N, seed)
        out = scatter_quant(gathered, expert_idxs, G, expand, group_counts=counts)
        check(gathered, expert_idxs, G, expand, counts, out)
    else:
        gathered = make_tokens(expert_idxs.numel(), N, seed)
        out = scatter_quant(gathered, expert_idxs, G, padding_alignment=128, group_counts=counts)
        check(gathered, expert_idxs, G, None, counts, out)
    check_stable(expert_idxs, G, out[1], out[2])


BALANCED = [2048] * 16
IMBALANCED = [0, 128, 384, 0, 0, 1024, 256, 128, 0, 2944, 128, 0, 640, 384, 128, 0]
RAGGED = [1, 0, 300, 129, 0, 77, 1000, 5, 0, 0, 255, 256, 17, 0, 640, 3]


@pytest.mark.parametrize("dedup", [1, 2, 4, 8])
@pytest.mark.parametrize("host_counts", [True, False])
@pytest.mark.parametrize("sizes", [BALANCED, IMBALANCED, RAGGED], ids=["bal", "imb", "ragged"])
def test_scatter_quant(sizes: list[int], host_counts: bool, dedup: int) -> None:
    run_case(sizes, 2048, dedup, segmented=True, host_counts=host_counts)


@pytest.mark.parametrize(
    "sizes,N",
    [
        ([4096] * 16, 2048),
        ([8192, 0, 0, 0, 8192, 0, 0, 16384], 2048),
        (RAGGED, 768),
        (RAGGED, 2880),  # gpt-oss hidden
        (RAGGED, 200),  # N not a multiple of 128
        ([5, 0, 3], 136),
        ([0, 0, 1], 8),
        ([300] * 128, 256),  # many groups
        ([7] * 300, 128),
        ([70000], 64),  # one group, > 2^16 tokens
    ],
)
def test_scatter_quant_shapes(sizes: list[int], N: int) -> None:
    run_case(sizes, N, 3, segmented=False, host_counts=False)


@pytest.mark.parametrize("host_counts", [True, False])
def test_scatter_quant_identity(host_counts: bool) -> None:
    """
    Without expand_idx, gathered holds one row per token.
    """
    run_case(IMBALANCED, 2048, 1, segmented=False, host_counts=host_counts)
    run_case(RAGGED, 2880, 1, segmented=False, host_counts=host_counts)


def test_scatter_quant_n_not_multiple_of_8() -> None:
    """
    N % 8 != 0 takes the unfused path, whose row placement is not stable, so
    check_stable is skipped.
    """
    sizes, G, N = RAGGED, len(RAGGED), 100
    expert_idxs, expand, m_src = make_routing(sizes, 2, True, 9)
    gathered = make_tokens(m_src, N, 9)
    check(gathered, expert_idxs, G, expand, None, scatter_quant(gathered, expert_idxs, G, expand))


def test_scatter_quant_determinism() -> None:
    expert_idxs, _, _ = make_routing(IMBALANCED, 1, False, 0)
    x = make_tokens(expert_idxs.numel(), 2048, 1)
    first = scatter_quant(x, expert_idxs, 16)
    second = scatter_quant(x, expert_idxs, 16)
    for tensor_a, tensor_b in zip(first[0], second[0]):
        assert torch.equal(as_bytes(tensor_a), as_bytes(tensor_b))
    assert torch.equal(first[1], second[1]) and torch.equal(first[2], second[2]) and first[3] == second[3]  # fmt: skip


def test_scatter_quant_back_to_back() -> None:
    for it, sizes in enumerate([[2048] * 16, [5, 0, 3] + [0] * 13, [4096] * 16, [1] * 16] * 3):
        run_case(sizes, 128, 2, segmented=it % 2 == 0, host_counts=True)


def test_scatter_quant_cuda_graph() -> None:
    """
    With host counts the op is capturable; replays see fresh tokens and routing.
    """
    sizes = IMBALANCED
    expert_idxs, expand, m_src = make_routing(sizes, 2, True, 5)
    gathered = make_tokens(m_src, 512, 5)
    # Compile outside the capture.
    scatter_quant(gathered, expert_idxs, 16, expand, group_counts=sizes)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = scatter_quant(gathered, expert_idxs, 16, expand, group_counts=sizes)
    for seed in (6, 7):
        new_expert_idxs, new_expand, _ = make_routing(sizes, 2, True, seed)
        expert_idxs.copy_(new_expert_idxs)
        expand.copy_(new_expand)
        gathered.copy_(make_tokens(m_src, 512, seed))
        graph.replay()
        check(gathered, expert_idxs, 16, expand, sizes, out)


def test_scatter_quant_zeros() -> None:
    """
    All-zero tokens give zero bytes everywhere and the scale floor.
    """
    expert_idxs, expand, m_src = make_routing(RAGGED, 2, True, 11)
    gathered = torch.zeros(m_src, 768, dtype=torch.bfloat16, device="cuda")
    out = scatter_quant(gathered, expert_idxs, len(RAGGED), expand)
    check(gathered, expert_idxs, len(RAGGED), expand, None, out)
    floor = e8m0_scale_rcp(torch.zeros(1, device="cuda"))[0].item()
    assert bool((out[0][1] == floor).all()) and bool((out[0][3] == floor).all())
