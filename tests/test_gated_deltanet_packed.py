from dataclasses import dataclass
from itertools import accumulate

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from pithtrain.contexts import training
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.linear import BF16Linear
from tests.utilities import launch


@dataclass
class PackedRequest:
    lengths: tuple[int, ...]
    compile: bool = False


def relative_error(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double(), got.double()
    return float((got - ref).norm() / ref.norm().clamp_min(1e-12))


def verify_packed(req: PackedRequest) -> None:
    """
    Compare output and gradients against separate documents; an unmasked row is the control.
    """
    # Imported here: the model module needs a GPU at import, and collection must not.
    from pithtrain.models.qwen35_moe import Qwen35MoeGatedDeltaNet

    training.Linear, training.fp8 = BF16Linear, False  # what setup_model binds for BF16
    device = torch.cuda.current_device()
    config = Qwen3_5MoeTextConfig(
        hidden_size=512,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    torch.manual_seed(42)
    layer = Qwen35MoeGatedDeltaNet(config, 0).to(device, torch.bfloat16)
    for module in layer.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
    # Slow the decay, so a state leaked across a boundary outlives the first few tokens.
    nn.init.constant_(layer.A_log, -2.0)
    forward = torch.compile(layer, fullgraph=True) if req.compile else layer

    S = sum(req.lengths)
    bounds = list(accumulate(req.lengths))
    cu_seqlens = torch.tensor([0, *bounds], dtype=torch.int32, device=device)
    tokens = torch.randn(1, S, config.hidden_size, device=device, dtype=torch.bfloat16)
    grad_out = torch.randn(1, S, config.hidden_size, device=device, dtype=torch.bfloat16)

    def run(packed: bool, seqlens: torch.Tensor | None) -> dict:
        layer.zero_grad(set_to_none=True)
        x = tokens.clone().requires_grad_(True)
        if packed:
            out = forward(x, seqlens)
        else:
            spans = zip([0, *bounds[:-1]], bounds)
            out = torch.cat([layer(x[:, start:end]) for start, end in spans], dim=1)
        out.backward(grad_out)
        grads = {name: param.grad for name, param in layer.named_parameters()}
        return {"out": out.detach(), "dx": x.grad, **grads}

    reference = run(packed=False, seqlens=None)
    packed = run(packed=True, seqlens=cu_seqlens)
    leaky = run(packed=True, seqlens=None)
    for name, ref in reference.items():
        error = relative_error(ref, packed[name])
        control = relative_error(ref, leaky[name])
        if not error < 2e-2:
            raise AssertionError(f"{name} leaks across documents: {error=:.2e} {control=:.2e}")
        if name in ("out", "dx") and not control > 10 * error:
            raise AssertionError(f"{name} control too close: {error=:.2e} {control=:.2e}")

    # Isolate cross-document gradients below the bf16 tolerance used above.
    last = bounds[-2]
    x = tokens.clone().requires_grad_(True)
    forward(x, cu_seqlens)[:, last:].backward(grad_out[:, last:])
    leak = float(x.grad[:, :last].double().norm() / x.grad[:, last:].double().norm())
    if not leak < 1e-5:
        raise AssertionError(f"the last document's gradient reaches earlier ones: {leak=:.2e}")


PACKED = [
    # Shorter than the conv kernel, crossing and ending on chunk boundaries (FLA chunks are 64).
    pytest.param(PackedRequest(lengths=(1, 3, 60, 64, 130, 2, 252)), id="mixed"),
    pytest.param(PackedRequest(lengths=(128, 128, 128, 128)), id="chunk-aligned"),
    pytest.param(PackedRequest(lengths=(37, 91, 200, 184), compile=True), id="compiled"),
]


@pytest.mark.parametrize("req", PACKED)
def test_packed_gated_deltanet(req: PackedRequest) -> None:
    launch(DistributedCfg(), verify_packed, req)
