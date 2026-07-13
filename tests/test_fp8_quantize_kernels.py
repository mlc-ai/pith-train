"""
Correctness tests for Triton FP8 quantization kernels.

Compares each Triton kernel in ``pithtrain.operators.deepgemm_quantize``
against the reference pure-PyTorch implementation in ``deep_gemm.utils.math``.
"""

import pytest
import torch
from deep_gemm.utils.math import (
    per_block_cast_to_fp8 as ref_per_block,
)
from deep_gemm.utils.math import (
    per_channel_cast_to_fp8 as ref_per_channel,
)
from deep_gemm.utils.math import (
    per_token_cast_to_fp8 as ref_per_token,
)

_USE_E8M0 = torch.cuda.get_device_capability()[0] >= 10


def calc_diff(x, y):
    """Normalized squared-error similarity metric."""
    x, y = x.detach().double(), y.detach().double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _make_bf16(shape, device="cuda"):
    return torch.randn(shape, device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# fused_rowwise_colwise_cast_to_fp8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,N",
    [
        (128, 128),
        (128, 256),
        (256, 256),
        (256, 512),
        (384, 384),
        (128, 300),  # N not multiple of 128
        (256, 7168),  # typical shape
    ],
)
def test_fused_per_token_per_channel(M, N):
    """Fused per_token+per_channel matches calling the two kernels separately."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_rowwise_colwise_cast_to_fp8,
    )

    x = _make_bf16((M, N))

    tok_fp8, tok_scale, ch_fp8, ch_scale = fused_rowwise_colwise_cast_to_fp8(x)

    ref_tok_fp8, ref_tok_scale = ref_per_token(x, use_ue8m0=_USE_E8M0, gran_k=128)
    ref_ch_fp8, ref_ch_scale = ref_per_channel(x, use_ue8m0=_USE_E8M0, gran_k=128)

    # Per-token outputs should match exactly
    assert tok_fp8.shape == ref_tok_fp8.shape == (M, N)
    assert torch.equal(tok_fp8.float(), ref_tok_fp8.float()), (
        f"tok FP8 diff = {calc_diff(tok_fp8.float(), ref_tok_fp8.float())}"
    )
    # Both are float32
    assert tok_scale.dtype == torch.float32
    assert ref_tok_scale.dtype == torch.float32
    assert tok_scale.shape == ref_tok_scale.shape
    assert torch.equal(tok_scale, ref_tok_scale), (
        f"tok scale max diff = {(tok_scale - ref_tok_scale).abs().max().item()}"
    )

    # Per-channel outputs should match exactly (float32 scales, unchanged)
    assert ch_fp8.shape == ref_ch_fp8.shape == (M, N)
    assert torch.equal(ch_fp8.float(), ref_ch_fp8.float()), (
        f"ch FP8 diff = {calc_diff(ch_fp8.float(), ref_ch_fp8.float())}"
    )
    assert ch_scale.dtype == torch.float32
    assert torch.equal(ch_scale, ref_ch_scale), (
        f"ch scale max diff = {(ch_scale - ref_ch_scale).abs().max().item()}"
    )


def test_fused_per_token_per_channel_all_zeros():
    """Fused kernel handles all-zero input (scale clamp prevents div-by-zero)."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_rowwise_colwise_cast_to_fp8,
    )

    x = torch.zeros((128, 256), device="cuda", dtype=torch.bfloat16)
    tok_fp8, tok_scale, ch_fp8, ch_scale = fused_rowwise_colwise_cast_to_fp8(x)

    assert tok_scale.dtype == torch.float32
    assert torch.isfinite(ch_scale).all()
    assert (tok_fp8.float() == 0.0).all()
    assert (ch_fp8.float() == 0.0).all()


# ---------------------------------------------------------------------------
# fused_rowwise_transpose_cast_to_fp8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,N",
    [
        (128, 128),
        (128, 256),
        (256, 256),
        (64, 256),
        (7, 300),  # odd M, N not multiple of 128
        (1, 128),
        (256, 7168),  # typical shape
    ],
)
def test_fused_per_token_and_transpose(M, N):
    """Fused per_token+transpose matches calling per_token on x and x.T separately."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_rowwise_transpose_cast_to_fp8,
    )

    x = _make_bf16((M, N))

    tok_fp8, tok_scale, t_fp8, t_scale = fused_rowwise_transpose_cast_to_fp8(x)

    ref_tok_fp8, ref_tok_scale = ref_per_token(x, use_ue8m0=_USE_E8M0, gran_k=128)
    ref_t_fp8, ref_t_scale = ref_per_token(x.T, use_ue8m0=_USE_E8M0, gran_k=128)

    # Per-token outputs should match exactly
    assert tok_fp8.shape == ref_tok_fp8.shape == (M, N)
    assert torch.equal(tok_fp8.float(), ref_tok_fp8.float()), (
        f"tok FP8 diff = {calc_diff(tok_fp8.float(), ref_tok_fp8.float())}"
    )
    assert tok_scale.dtype == torch.float32
    assert torch.equal(tok_scale, ref_tok_scale), (
        f"tok scale max diff = {(tok_scale - ref_tok_scale).abs().max().item()}"
    )

    # Transposed per-token outputs should match exactly
    assert t_fp8.shape == ref_t_fp8.shape == (N, M)
    assert torch.equal(t_fp8.float(), ref_t_fp8.float()), (
        f"t FP8 diff = {calc_diff(t_fp8.float(), ref_t_fp8.float())}"
    )
    assert t_scale.dtype == torch.float32
    assert torch.equal(t_scale, ref_t_scale), (
        f"t scale max diff = {(t_scale - ref_t_scale).abs().max().item()}"
    )


def test_fused_per_token_and_transpose_all_zeros():
    """Fused transpose kernel handles all-zero input."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_rowwise_transpose_cast_to_fp8,
    )

    x = torch.zeros((64, 256), device="cuda", dtype=torch.bfloat16)
    tok_fp8, tok_scale, t_fp8, t_scale = fused_rowwise_transpose_cast_to_fp8(x)

    assert tok_scale.dtype == torch.float32
    assert t_scale.dtype == torch.float32
    assert (tok_fp8.float() == 0.0).all()
    assert (t_fp8.float() == 0.0).all()


# ---------------------------------------------------------------------------
# fused_rowwise_blockwise_transpose_cast_to_fp8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,K",
    [
        (128, 128),
        (256, 128),
        (128, 256),
        (256, 256),
        (300, 384),
        (1, 128),
        (64, 64),
        (7, 300),
        (256, 7168),
    ],
)
def test_fused_per_token_and_per_block_transpose(M, K):
    """Fused per_token+per_block_transpose matches separate per_token(x) + per_block(x.T)."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_rowwise_blockwise_transpose_cast_to_fp8,
    )

    x = _make_bf16((M, K))

    tok_fp8, tok_scale, blk_t_fp8, blk_t_scale = fused_rowwise_blockwise_transpose_cast_to_fp8(x)

    ref_tok_fp8, ref_tok_scale = ref_per_token(x, use_ue8m0=_USE_E8M0, gran_k=128)
    ref_blk_t_fp8, ref_blk_t_scale = ref_per_block(x.T, use_ue8m0=_USE_E8M0, gran_k=128)

    # Per-token outputs should match exactly
    assert tok_fp8.shape == ref_tok_fp8.shape == (M, K)
    assert torch.equal(tok_fp8.float(), ref_tok_fp8.float()), (
        f"tok FP8 diff = {calc_diff(tok_fp8.float(), ref_tok_fp8.float())}"
    )
    # Both are float32
    assert tok_scale.dtype == torch.float32
    assert torch.equal(tok_scale, ref_tok_scale), (
        f"tok scale max diff = {(tok_scale - ref_tok_scale).abs().max().item()}"
    )

    # Per-block transpose outputs should match exactly
    assert blk_t_fp8.shape == ref_blk_t_fp8.shape == (K, M)
    assert torch.equal(blk_t_fp8.float(), ref_blk_t_fp8.float()), (
        f"blk_t FP8 diff = {calc_diff(blk_t_fp8.float(), ref_blk_t_fp8.float())}"
    )
    assert torch.equal(blk_t_scale, ref_blk_t_scale), (
        f"blk_t scale max diff = {(blk_t_scale - ref_blk_t_scale).abs().max().item()}"
    )


def test_fused_per_token_and_per_block_transpose_all_zeros():
    """Fused per_token+per_block_transpose handles all-zero input."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_rowwise_blockwise_transpose_cast_to_fp8,
    )

    x = torch.zeros((64, 256), device="cuda", dtype=torch.bfloat16)
    tok_fp8, tok_scale, blk_t_fp8, blk_t_scale = fused_rowwise_blockwise_transpose_cast_to_fp8(x)

    assert torch.isfinite(tok_scale).all()
    assert torch.isfinite(blk_t_scale).all()
    assert (tok_fp8.float() == 0.0).all()
    assert (blk_t_fp8.float() == 0.0).all()


# ---------------------------------------------------------------------------
# fused_blockwise_transpose_cast_to_fp8 (2D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,K",
    [
        (128, 128),  # exact tile
        (256, 256),  # multi-tile
        (300, 384),  # non-multiples of 128
        (1, 128),  # single row
        (64, 64),  # sub-tile
        (128, 256),
        (7, 300),  # odd M, K not multiple of 128
        (256, 7168),  # typical weight shape
    ],
)
def test_fused_per_block_and_transpose(M, K):
    """Fused per_block+transpose matches separate per_block(x) + per_block(x.T)."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_blockwise_transpose_cast_to_fp8,
    )

    x = _make_bf16((M, K))

    fp8, scale, fp8_t, scale_t = fused_blockwise_transpose_cast_to_fp8(x)

    ref_fp8, ref_scale = ref_per_block(x, use_ue8m0=_USE_E8M0, gran_k=128)
    ref_fp8_t, ref_scale_t = ref_per_block(x.T, use_ue8m0=_USE_E8M0, gran_k=128)

    # Original per-block outputs should match exactly
    assert fp8.shape == ref_fp8.shape == (M, K)
    assert torch.equal(fp8.float(), ref_fp8.float()), (
        f"FP8 diff = {calc_diff(fp8.float(), ref_fp8.float())}"
    )
    assert scale.dtype == torch.float32
    assert torch.equal(scale, ref_scale), (
        f"scale max diff = {(scale - ref_scale).abs().max().item()}"
    )

    # Transposed per-block outputs should match exactly
    assert fp8_t.shape == ref_fp8_t.shape == (K, M)
    assert torch.equal(fp8_t.float(), ref_fp8_t.float()), (
        f"FP8_t diff = {calc_diff(fp8_t.float(), ref_fp8_t.float())}"
    )
    assert torch.equal(scale_t, ref_scale_t), (
        f"scale_t max diff = {(scale_t - ref_scale_t).abs().max().item()}"
    )


def test_fused_per_block_and_transpose_all_zeros():
    """Fused per_block+transpose handles all-zero input (finite scales)."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_blockwise_transpose_cast_to_fp8,
    )

    x = torch.zeros((128, 256), device="cuda", dtype=torch.bfloat16)
    fp8, scale, fp8_t, scale_t = fused_blockwise_transpose_cast_to_fp8(x)

    assert torch.isfinite(scale).all()
    assert torch.isfinite(scale_t).all()
    assert (fp8.float() == 0.0).all()
    assert (fp8_t.float() == 0.0).all()


@pytest.mark.parametrize(
    "M,K",
    [
        (128, 128),
        (256, 256),
        (300, 384),
        (128, 512),
    ],
)
def test_fused_per_block_and_transpose_scale_symmetry(M, K):
    """Verifies scale == scale_t.T for the fused per_block+transpose kernel."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_blockwise_transpose_cast_to_fp8,
    )

    x = _make_bf16((M, K))
    _, scale, _, scale_t = fused_blockwise_transpose_cast_to_fp8(x)

    assert torch.equal(scale, scale_t.T), (
        f"scale vs scale_t.T max diff = {(scale - scale_t.T).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# fused_blockwise_transpose_cast_to_fp8_batched (3D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "G,N,K",
    [
        (1, 128, 128),
        (4, 256, 128),
        (8, 256, 512),
        (2, 300, 384),  # non-multiples of 128
        (4, 64, 64),  # sub-tile
        (2, 128, 256),
    ],
)
def test_fused_per_block_and_transpose_batched(G, N, K):
    """Fused batched per_block+transpose matches separate per_block calls per group."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_blockwise_transpose_cast_to_fp8_batched,
    )

    x = _make_bf16((G, N, K))

    fp8, scale, fp8_t, scale_t = fused_blockwise_transpose_cast_to_fp8_batched(x)

    # Reference: loop + stack of ref_per_block per group
    ref_fp8_list, ref_scale_list = [], []
    ref_fp8_t_list, ref_scale_t_list = [], []
    for g in range(G):
        fp8_g, s_g = ref_per_block(x[g], use_ue8m0=_USE_E8M0, gran_k=128)
        ref_fp8_list.append(fp8_g)
        ref_scale_list.append(s_g)
        fp8_t_g, s_t_g = ref_per_block(x[g].T, use_ue8m0=_USE_E8M0, gran_k=128)
        ref_fp8_t_list.append(fp8_t_g)
        ref_scale_t_list.append(s_t_g)
    ref_fp8 = torch.stack(ref_fp8_list)
    ref_scale = torch.stack(ref_scale_list)
    ref_fp8_t = torch.stack(ref_fp8_t_list)
    ref_scale_t = torch.stack(ref_scale_t_list)

    # Original per-block outputs should match exactly
    assert fp8.shape == ref_fp8.shape == (G, N, K)
    assert torch.equal(fp8.float(), ref_fp8.float()), (
        f"FP8 diff = {calc_diff(fp8.float(), ref_fp8.float())}"
    )
    assert scale.dtype == torch.float32
    assert torch.equal(scale, ref_scale), (
        f"scale max diff = {(scale - ref_scale).abs().max().item()}"
    )

    # Transposed per-block outputs should match exactly
    assert fp8_t.shape == ref_fp8_t.shape == (G, K, N)
    assert torch.equal(fp8_t.float(), ref_fp8_t.float()), (
        f"FP8_t diff = {calc_diff(fp8_t.float(), ref_fp8_t.float())}"
    )
    assert torch.equal(scale_t, ref_scale_t), (
        f"scale_t max diff = {(scale_t - ref_scale_t).abs().max().item()}"
    )


def test_fused_per_block_and_transpose_batched_all_zeros():
    """Fused batched per_block+transpose handles all-zero input (finite scales)."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_blockwise_transpose_cast_to_fp8_batched,
    )

    x = torch.zeros((4, 128, 256), device="cuda", dtype=torch.bfloat16)
    fp8, scale, fp8_t, scale_t = fused_blockwise_transpose_cast_to_fp8_batched(x)

    assert torch.isfinite(scale).all()
    assert torch.isfinite(scale_t).all()
    assert (fp8.float() == 0.0).all()
    assert (fp8_t.float() == 0.0).all()


@pytest.mark.parametrize(
    "G,N,K",
    [
        (2, 128, 128),
        (4, 256, 256),
        (2, 300, 384),
    ],
)
def test_fused_per_block_and_transpose_batched_scale_symmetry(G, N, K):
    """Verifies scale == scale_t.transpose(1,2) for the batched fused kernel."""
    from pithtrain.operators.deepgemm_quantize import (
        fused_blockwise_transpose_cast_to_fp8_batched,
    )

    x = _make_bf16((G, N, K))
    _, scale, _, scale_t = fused_blockwise_transpose_cast_to_fp8_batched(x)

    assert torch.equal(scale, scale_t.transpose(1, 2)), (
        f"scale vs scale_t.T max diff = {(scale - scale_t.transpose(1, 2)).abs().max().item()}"
    )
