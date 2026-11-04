"""
act_scale_like must return the strides this device's activation quantization returns: fake
implementations of the FP8 ops allocate their scales with it, so a fake and a real scale must agree.
"""

import pytest
import torch

from pithtrain.kernels import (
    act_scale_like,
    fp8_quant_act,
    fp8_quant_grad,
)

BLOCK = 128

MN_SFK = [(1, 1), (5, 1), (4096, 16), (1001, 3), (2, 7), (4096, 23), (1000, 86), (0, 2)]

# (M, K) of the activations the four models quantize, plus unaligned and tiny rows.
MK = [
    (4096, 2048),
    (4096, 1408),
    (4096, 768),
    (4096, 512),
    (4096, 2880),
    (1000, 10944),
    (4100, 576),
    (7, 300),
    (1, 128),
    (130, 129),
    (2, 1),
]


@pytest.mark.parametrize("mn,sf_k", MN_SFK)
def test_shape_dtype_device(mn: int, sf_k: int) -> None:
    scale = act_scale_like(mn, sf_k, torch.device("cuda"))
    assert scale.shape == (mn, sf_k)
    assert scale.dtype == torch.float32
    assert scale.device.type == "cuda"
    # Every logical element has its own storage slot (writable without aliasing).
    if mn * sf_k:
        scale.copy_(torch.arange(mn * sf_k, device="cuda", dtype=torch.float32).view(mn, sf_k))
        assert torch.equal(scale, torch.arange(mn * sf_k, device="cuda").float().view(mn, sf_k))


@pytest.mark.parametrize("mn,sf_k", MN_SFK)
def test_meta_device(mn: int, sf_k: int) -> None:
    meta = act_scale_like(mn, sf_k, torch.device("meta"))
    real = act_scale_like(mn, sf_k, torch.device("cuda"))
    assert meta.device.type == "meta" and meta.dtype == torch.float32
    assert meta.shape == real.shape and meta.stride() == real.stride()


@pytest.mark.parametrize("M,K", MK)
def test_matches_rowwise_blockwise_transpose(M: int, K: int) -> None:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    _, x_scale, _, _ = fp8_quant_act(x)
    fake = act_scale_like(M, (K + BLOCK - 1) // BLOCK, x.device)
    assert fake.shape == x_scale.shape and fake.dtype == x_scale.dtype
    assert fake.stride() == x_scale.stride(), (fake.stride(), x_scale.stride())


@pytest.mark.parametrize("M,K", MK)
def test_matches_rowwise_transpose(M: int, K: int) -> None:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    _, x_scale, _, xt_scale = fp8_quant_grad(x)
    for scale, mn, k in ((x_scale, M, K), (xt_scale, K, M)):
        fake = act_scale_like(mn, (k + BLOCK - 1) // BLOCK, x.device)
        assert fake.shape == scale.shape and fake.dtype == scale.dtype
        assert fake.stride() == scale.stride(), (mn, k, fake.stride(), scale.stride())
