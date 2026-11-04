"""
FP8 bytes and scale values must match the PyTorch reference exactly; scale strides are unspecified,
so scales are compared by logical value.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import fp8_quant_grad

BLOCK = 128


def e8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """
    Return 2 ** ceil(log2(max(amax, 1e-4) / 448)), computed exactly with frexp.
    """
    ratio = amax.float().clamp(min=1e-4) / 448.0
    mantissa, exponent = torch.frexp(ratio)  # ratio = mantissa * 2**exponent, mantissa in [0.5, 1)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    return ((exponent + 127) << 23).to(torch.int32).view(torch.float32)


def ref_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor rowwise in 1x128 blocks: (fp8 (M, K), scale (M, ceil(K/128))).
    """
    M, K = x.shape
    num_blocks = (K + BLOCK - 1) // BLOCK
    x_blocks = F.pad(x.float(), (0, num_blocks * BLOCK - K)).view(M, num_blocks, BLOCK)
    scale = e8m0_scale(x_blocks.abs().amax(dim=-1)) if M * num_blocks else torch.empty(M, num_blocks, device=x.device)  # fmt: skip
    fp8 = (x_blocks / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return fp8.view(M, num_blocks * BLOCK)[:, :K].contiguous(), scale


def assert_fp8_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float8_e4m3fn, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    out_bytes, ref_bytes = out.view(torch.uint8), ref.view(torch.uint8)
    assert torch.equal(out_bytes, ref_bytes), f"{name}: {(out_bytes != ref_bytes).sum().item()} mismatching bytes"  # fmt: skip


def assert_scale_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float32, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    assert torch.equal(out, ref), f"{name}: {(out != ref).sum().item()} mismatching scales"


def check(x: torch.Tensor) -> None:
    x_fp8, x_scale, xt_fp8, xt_scale = fp8_quant_grad(x)
    ref_fp8, ref_scale = ref_rowwise(x)
    ref_fp8_t, ref_scale_t = ref_rowwise(x.T)
    assert_fp8_equal("x_fp8", x_fp8, ref_fp8)
    assert_scale_equal("x_scale", x_scale, ref_scale)
    assert_fp8_equal("xt_fp8", xt_fp8, ref_fp8_t)
    assert_scale_equal("xt_scale", xt_scale, ref_scale_t)


# (M, N): model hidden / intermediate / dense sizes, tile edges, unaligned sizes.
SHAPES = [
    (4096, 4096),
    (4096, 2048),
    (4096, 1408),
    (4096, 768),
    (4096, 512),
    (4096, 2880),
    (1000, 10944),
    (4100, 576),
    (128, 128),
    (128, 256),
    (256, 256),
    (64, 256),
    (1, 128),
    (7, 64),
    (7, 300),
    (16, 8),
    (300, 200),
    (130, 1000),
    (257, 129),
    (64, 4104),
    (333, 1001),
    (96, 1028),
    (258, 1030),
    (4101, 512),
    (136, 384),
    (2056, 2048),
    (8, 4096),
    (256, 7168),
]


def make_input(m: int, n: int, kind: str, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, n, device="cuda", dtype=torch.float32, generator=gen)
    if kind == "outliers":
        mask = torch.rand(m, n, device="cuda", generator=gen) < 0.001
        x = torch.where(mask, x * 1e4, x)
        x[:, ::7] *= 1e-3
        x[::5, :] *= 30.0
    elif kind == "zeros":
        x[:, : max(1, n // 3)] = 0.0
        x[: max(1, m // 4), :] = 0.0
    elif kind == "tiny":
        x = x * 1e-7
    elif kind == "pow2":
        # amax exactly 448 * 2**k takes the exact-power-of-two (no round-up) branch of the ceil.
        x = x.clamp(-1, 1)
        x[:, 0] = 448.0 * 2.0**-3
        x[0, :] = -448.0
    elif kind == "wide":
        x = x * torch.exp2(torch.randint(-40, 40, (m, n), device="cuda", generator=gen).float())
    elif kind == "huge":
        x = x * 1e30
    elif kind == "negative":
        x = -x.abs()
    return x.to(torch.bfloat16)


KINDS = ["randn", "outliers", "zeros", "tiny", "pow2", "wide", "huge", "negative"]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", KINDS)
def test_matches_reference(shape: tuple[int, int], kind: str) -> None:
    check(make_input(*shape, kind))


def test_all_zeros() -> None:
    check(torch.zeros(64, 256, device="cuda", dtype=torch.bfloat16))


def test_misaligned_base_pointer() -> None:
    base = make_input(512, 1032, "outliers")
    for off in (3, 4, 8):
        check(base.view(-1)[off : off + 512 * 1024].view(512, 1024))


def test_non_contiguous() -> None:
    check(make_input(256, 512, "randn")[:, :384])
    check(make_input(512, 300, "outliers").T)


def test_empty() -> None:
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    out = fp8_quant_grad(x)
    assert [tuple(t.shape) for t in out] == [(0, 256), (0, 2), (256, 0), (256, 0)]
