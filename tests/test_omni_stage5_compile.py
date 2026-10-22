"""Execute the production stage5 bodies through CPU fullgraph Dynamo/Inductor.

Extracting these pure methods avoids importing unrelated CUDA-only operators.
The bodies and compile decorators are unchanged; no model math is stubbed.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pithtrain.models.qwen3_omni_image_inputs import inject_deepstack

ROOT = Path(__file__).parents[1]


def layer_type(ep):
    namespace = dict(
        torch=torch, distributed=SimpleNamespace(ep_size=ep), inject_deepstack=inject_deepstack
    )
    parent_name = "Qwen3OmniMoeThinkerTextDecoderLayer"
    parent = type(parent_name, (torch.nn.Module,), {})
    namespace[parent_name] = parent
    for file, name, cls, methods in (
        ("qwen3_omni_moe.py", parent_name, parent, ("_forward_stage5", "forward_stage5")),
        (
            "qwen3_omni_image.py",
            "Qwen3OmniMoeThinkerImageDecoderLayer",
            type("ImageLayer", (parent,), {}),
            ("forward_stage5",),
        ),
    ):
        path = ROOT / "pithtrain/models" / file
        tree = ast.parse(path.read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        for method in methods:
            node = next(
                n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == method
            )
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
            setattr(cls, method, namespace[method])
    return cls


@pytest.mark.parametrize("is_moe,ep", [(False, 1), (True, 1), (True, 2)])
def test_image_stage5_preserves_fullgraph_output_and_all_gradients(is_moe, ep):
    torch.manual_seed(19)
    layer = layer_type(ep)()
    layer.is_moe, layer.idx, layer.text_hidden_size, layer.deepstack_count = is_moe, 1, 8, 2
    residual = torch.randn(1, 4, 24, requires_grad=True)
    weights = torch.randn(4, 2, requires_grad=True) if is_moe else None
    output = (
        torch.randn(8, 8, requires_grad=True)
        if is_moe
        else torch.randn(1, 4, 8, requires_grad=True)
    )
    indexes = torch.arange(8) if is_moe else None
    actual = layer.forward_stage5(output, indexes, weights, residual)
    hidden = residual[..., :8]
    if is_moe:
        expected = hidden + (output.reshape(4, 2, 8) * weights[..., None]).sum(1).reshape_as(hidden)
    else:
        expected = hidden + output
    expected = torch.cat((expected + residual[..., 16:24], residual[..., 8:]), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    values = (residual, output, weights) if is_moe else (residual, output)
    actual_grads = torch.autograd.grad(actual.square().sum(), values, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), values)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=2e-6)
