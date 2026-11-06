"""Independent stable-sort oracles for the fused row-placement candidate."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocation required")


@pytest.mark.parametrize("rows,groups", [(1, 4), (257, 8), (1025, 4)])
@pytest.mark.parametrize("canonical", [False, True])
def test_scatter_policies_preserve_values_padding_and_backward(rows, groups, canonical):
    from pithtrain.operators.token_scatter import scatter_for_grouped_gemm

    generator = torch.Generator(device="cuda").manual_seed(37)
    ids = torch.randint(groups - 1, (rows,), device="cuda", generator=generator)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    x.requires_grad_()
    expected = scatter_for_grouped_gemm(x, ids, groups, deterministic=True)
    actual = scatter_for_grouped_gemm(x, ids, groups, deterministic=False, canonical=canonical)
    assert torch.equal(actual[0][actual[1]], x)
    padding = torch.ones(actual[0].shape[0], device="cuda", dtype=torch.bool)
    padding[actual[1]] = False
    assert torch.count_nonzero(actual[0][padding]) == 0
    fields = range(5) if canonical else range(2, 5)
    for field in fields:
        a, b = actual[field], expected[field]
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    dy = torch.randn_like(actual[0])
    actual[0].backward(dy)
    assert torch.equal(x.grad, dy[actual[1]])
