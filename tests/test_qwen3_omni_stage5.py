"""Check production compiled text stage5 and its eager arithmetic body.

Extracting these pure methods avoids unrelated CUDA-only kernel imports. The
production bodies are executed without replacement in both controls.
"""

import pytest
import torch

from tests.cpu_helpers import layer_type


@pytest.mark.parametrize("is_moe,ep", [(False, 1), (True, 1), (True, 2)])
def test_text_stage5_fullgraph_output_and_all_gradients(is_moe, ep):
    torch.manual_seed(19)
    layer = layer_type(ep)()
    layer.is_moe = is_moe
    residual = torch.randn(1, 4, 8, requires_grad=True)
    weights = torch.randn(4, 2, requires_grad=True) if is_moe else None
    indexes = torch.tensor([6, 1, 4, 3, 2]) if ep == 2 else torch.arange(8)
    output = (
        torch.randn(len(indexes), 8, requires_grad=True)
        if is_moe
        else torch.randn_like(residual, requires_grad=True)
    )
    if is_moe:
        # Independent per-token weighted sum also covers sparse, unordered EP2
        # assignments and tokens with no local experts.
        contributions = []
        for token in range(4):
            selected = indexes // 2 == token
            contributions.append(
                (output[selected] * weights.reshape(-1)[indexes[selected], None]).sum(0)
            )
        expected = residual + torch.stack(contributions).reshape_as(residual)
    else:
        expected = residual + output
    values = (residual, output, weights) if is_moe else (residual, output)
    expected_grads = torch.autograd.grad(expected.square().sum(), values)

    def eager(*args):
        return type(layer).forward_stage5.__wrapped__(layer, *args)

    for call in (eager, layer.forward_stage5):
        actual = call(output, indexes if is_moe else None, weights, residual)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        actual_grads = torch.autograd.grad(actual.square().sum(), values)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=2e-6)
