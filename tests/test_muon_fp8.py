"""Smoke test: Muon steps cleanly on gradients from the DeepGEMM FP8 path.

FP8 quantizes only the forward matmuls; grads are bf16/fp32 and the Muon math is
unchanged. Confirms a real fp8 forward/backward feeds finite grads into a Muon
step. Requires CUDA + ``deep_gemm`` (skips otherwise)."""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from pithtrain.modules.optimizer import Muon
from pithtrain.modules.training import is_muon_param


def test_muon_step_on_fp8_gradients():
    pytest.importorskip("deep_gemm")
    from pithtrain.contexts import training
    from pithtrain.operators.linear import FP8Linear

    prev = getattr(training, "fp8", False)
    training.fp8 = True
    try:
        torch.manual_seed(0)
        linear_cls = FP8Linear
        # Dims are multiples of 128 for DeepGEMM block scaling.
        net = nn.Sequential(
            linear_cls(256, 512, bias=False),
            linear_cls(512, 256, bias=False),
        ).to(device="cuda", dtype=torch.bfloat16)

        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        net(x).float().pow(2).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())

        muon_params, adamw_params = [], []
        for name, p in net.named_parameters():
            (muon_params if is_muon_param(name, p) else adamw_params).append(p)
        optimizers = [Muon(muon_params, lr=0.02)]
        if adamw_params:
            optimizers.append(AdamW(adamw_params, lr=0.02, weight_decay=0.0))
        for opt in optimizers:
            opt.step()
        for name, p in net.named_parameters():
            assert torch.isfinite(p).all(), name
    finally:
        training.fp8 = prev
