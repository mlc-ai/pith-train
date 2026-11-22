"""
The PithTrain FP8 recipe in plain PyTorch: E8M0 scales per 1x128 block or 128x128 tile, e4m3
rounded to nearest even and saturated to +-448.
"""

import torch

BLOCK = 128
INV_448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)


def e8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """
    The E8M0 scale: 2^ceil(log2(max(amax, 1e-4) / 448)) in fp32.
    """
    raw = amax.float().clamp(min=1e-4) * INV_448.to(amax.device)
    mantissa, exponent = torch.frexp(raw)  # mantissa in [0.5, 1)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    return torch.ldexp(torch.ones_like(raw), exponent)


def cast(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() * scale.reciprocal()).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def blocks(n: int) -> int:
    return -(-n // BLOCK)


def quant_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1x128 blocks along each row of a 2-D x: fp8 (M, K), scale (M, ceil(K / 128)).
    """
    M, K = x.shape
    padded = x.new_zeros(M, blocks(K) * BLOCK, dtype=torch.float32)
    padded[:, :K] = x.float()
    tiles = padded.view(M, blocks(K), BLOCK)
    scale = e8m0_scale(tiles.abs().amax(-1))
    q = cast(tiles, scale[..., None]).view(M, -1)[:, :K].contiguous()
    return q, scale


def quant_blockwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    128x128 tiles of x (..., R, C): fp8 (..., R, C), scale (..., ceil(R / 128), ceil(C / 128)).
    """
    *lead, R, C = x.shape
    padded = x.new_zeros(*lead, blocks(R) * BLOCK, blocks(C) * BLOCK, dtype=torch.float32)
    padded[..., :R, :C] = x.float()
    tiles = padded.view(*lead, blocks(R), BLOCK, blocks(C), BLOCK)
    scale = e8m0_scale(tiles.abs().amax(dim=(-3, -1)))
    q = cast(tiles, scale[..., :, None, :, None]).view(*lead, blocks(R) * BLOCK, blocks(C) * BLOCK)
    return q[..., :R, :C].contiguous(), scale


def dequant_rowwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.repeat_interleave(BLOCK, dim=-1)[..., : q.shape[-1]]


def dequant_blockwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    R, C = q.shape[-2:]
    full = scale.repeat_interleave(BLOCK, dim=-2).repeat_interleave(BLOCK, dim=-1)
    return q.float() * full[..., :R, :C]


def assert_fp8_equal(got: torch.Tensor, want: torch.Tensor, name: str) -> None:
    assert got.dtype == torch.float8_e4m3fn and got.shape == want.shape, (name, got.dtype, got.shape, want.shape)  # fmt: skip
    differ = (
        (got.contiguous().view(torch.uint8) != want.contiguous().view(torch.uint8)).sum().item()
    )
    assert differ == 0, f"{name}: {differ} of {got.numel()} fp8 bytes differ"


def assert_scale_equal(got: torch.Tensor, want: torch.Tensor, name: str) -> None:
    assert got.dtype == torch.float32 and got.shape == want.shape, (name, got.shape, want.shape)
    differ = (got != want).sum().item()
    assert differ == 0, f"{name}: {differ} of {got.numel()} scales differ"


def half_ulp_bf16(x: torch.Tensor) -> torch.Tensor:
    x = x.double().abs()
    _, exponent = torch.frexp(x)  # ulp = 2^(exponent - 8)
    exponent = exponent.clamp(min=-125)
    return torch.ldexp(torch.full_like(x, 0.5), exponent - 8)
