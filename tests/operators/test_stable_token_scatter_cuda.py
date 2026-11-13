"""CUDA contracts for the opt-in candidate; execute only in a GPU allocation."""

import pytest
import torch


@pytest.mark.parametrize("rows,experts", [(0, 4), (1, 4), (64, 4), (129, 16), (513, 4)])
def test_stable_scatter_layout_padding_and_backward(rows, experts):
    if not torch.cuda.is_available():
        pytest.skip("CUDA allocation required")
    from pithtrain.operators.token_scatter import scatter_for_grouped_gemm

    generator = torch.Generator(device="cuda").manual_seed(117)
    ids = torch.randint(experts, (rows,), device="cuda", generator=generator)
    x = torch.randn(rows, 128, device="cuda", generator=generator, requires_grad=True)
    first_reverse = None
    for _ in range(16):
        grouped, reverse, offsets, sizes, size_tensor = scatter_for_grouped_gemm(
            x, ids, experts, deterministic=True
        )
        if rows == 0:
            assert grouped.shape == x.shape
            assert reverse.numel() == offsets.numel() == size_tensor.numel() == 0
            assert sizes == []
            continue
        counts = torch.bincount(ids, minlength=experts)
        expected_sizes = ((counts + 127) // 128) * 128
        torch.testing.assert_close(offsets, expected_sizes.cumsum(0).int(), rtol=0, atol=0)
        assert sizes == expected_sizes.tolist()
        torch.testing.assert_close(size_tensor, expected_sizes.int(), rtol=0, atol=0)
        assert reverse.unique().numel() == rows
        torch.testing.assert_close(grouped[reverse], x, rtol=0, atol=0)
        padding = torch.ones(grouped.shape[0], device="cuda", dtype=torch.bool)
        padding[reverse] = False
        assert torch.count_nonzero(grouped[padding]) == 0
        for expert in range(experts):
            selected = reverse[ids == expert]
            assert bool(torch.all(selected[1:] > selected[:-1]))
        if first_reverse is None:
            first_reverse = reverse.clone()
        else:
            assert torch.equal(reverse, first_reverse)
    gradient = torch.randn_like(grouped)
    grouped.backward(gradient)
    torch.testing.assert_close(x.grad, gradient[reverse], rtol=0, atol=0)
