"""
The BF16 expert slot maps, gathers and segment sums at the bf16_expert_slots cases, exact.
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
from tests.kernels.cases import params, seed_of
from tests.kernels.common import (
    BF16,
    I64,
    REALISTIC,
    ROUTING_CHUNK,
    bits,
    cumsum_list,
    stage3_input,
)


@pytest.fixture(autouse=True)
def empty_cache():
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()


def segment_sum_order(x: torch.Tensor, cu_host: list[int], idx: torch.Tensor | None) -> torch.Tensor:  # fmt: skip
    """
    bf16_expert_segment_sum's order: per segment, fp32 sums of 256-row chunks, added in order.
    """
    G, n = len(cu_host) - 1, x.shape[1]
    out = torch.zeros(G, n, device="cuda", dtype=torch.float32)
    for group in range(G):
        start, end = cu_host[group], cu_host[group + 1]
        total = torch.zeros(n, device="cuda", dtype=torch.float32)
        for a in range(start, end, ROUTING_CHUNK):
            b = min(a + ROUTING_CHUNK, end)
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


@pytest.mark.parametrize("case", params("bf16_expert_slots", keep=lambda c: REALISTIC(c) and c["axes"]["bf16_fused"]))  # fmt: skip
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
    order = torch.argsort(expert_idxs, stable=True)
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
        for start in range(0, S, ROUTING_CHUNK):
            end = min(start + ROUTING_CHUNK, S)
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
        by_token = torch.argsort(expand_idx * G + expert_idxs)
        token = expand_idx[by_token]
        per_token = torch.bincount(expand_idx, minlength=num_tokens)
        slot = torch.arange(S, **I64) - (per_token.cumsum(0) - per_token)[token]
        assert int(per_token.max()) <= max_slots
        table = torch.full((num_tokens, max_slots), -1, **I64)
        table[token, slot] = row_of[by_token]
        for start in range(0, num_tokens, ROUTING_CHUNK):
            end = min(start + ROUTING_CHUNK, num_tokens)
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
