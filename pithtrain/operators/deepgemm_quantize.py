"""
Fused Triton kernels for FP8 quantization with architecture-aware scaling.

Replaces the pure-PyTorch quantization utilities from ``deep_gemm.utils.math``
with single-pass Triton kernels that fuse pad -> abs -> amax -> scale -> cast.

On Blackwell (SM100+), produces E8M0 power-of-2 scaling factors for MXFP8 via
PTX ``cvt.rp.satfinite.ue8m0x2.f32``.  On Hopper (SM90), produces plain
float32 scales (``amax / 448``).  Block granularity is fixed at 128 elements.

Public kernels:
    1. fused_rowwise_colwise_cast_to_fp8 -- rowwise(x) + colwise(x)
    2. fused_rowwise_transpose_cast_to_fp8 -- rowwise(x) + rowwise(x.T)
    3. fused_rowwise_blockwise_transpose_cast_to_fp8 -- rowwise(x) + blockwise(x.T)
    4. fused_blockwise_transpose_cast_to_fp8 -- blockwise(x) + blockwise(x.T), 2D
    5. fused_blockwise_transpose_cast_to_fp8_batched -- blockwise(x) + blockwise(x.T), 3D
"""

import torch
import triton
import triton.language as tl

# DeepGEMM block granularity -- hardcoded, all callers use 128
_BLOCK_K: tl.constexpr = 128

# FP8 E4M3 max representable value
_FP8_MAX: tl.constexpr = 448.0

# Detect SM version once at import time
ARCH_MAJOR, _ = torch.cuda.get_device_capability()
_USE_E8M0_SCALES = ARCH_MAJOR >= 10


# ---------------------------------------------------------------------------
# Shared: FP8 scale computation
# ---------------------------------------------------------------------------


@triton.jit
def _compute_fp8_scale(amax, SCALING_MODE: tl.constexpr):
    """Compute float32 scale from per-group amax values.

    Given ``amax`` (the max absolute value of a group), computes a scaling
    factor for FP8 quantization.

    Both modes produce power-of-2 scales so that quantize and dequantize
    multiplications are exact (IEEE-754 exponent shift, no mantissa rounding).

    - ``"e8m0"``: SM100+ E8M0 scale via PTX ``cvt.rp.satfinite.ue8m0x2.f32``.
    - ``"fp32"``: Equivalent power-of-2 ceil via IEEE-754 bit manipulation
      (used on Hopper / SM90 where the PTX instruction is unavailable).

    Returns (scale, reciprocal_scale), both exact powers of 2.
    """
    FP8_MAX_RCP: tl.constexpr = 1.0 / 448.0

    amax_clamped = tl.maximum(amax.to(tl.float32), 1e-4)
    scale_input = amax_clamped * FP8_MAX_RCP

    if SCALING_MODE == "e8m0":
        # SM100: use PTX cvt.rp (round-positive = ceil) to E8M0
        scale_e8m0_biased = tl.inline_asm_elementwise(
            asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            constraints="=h,r",
            args=[scale_input],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)

        scale_fp = (scale_e8m0_biased.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    else:
        tl.static_assert(SCALING_MODE == "fp32")
        # Ceil to nearest power of 2 via IEEE-754 bit manipulation:
        # clear mantissa bits (floor to pow2), then increment exponent if
        # the original value wasn't already an exact power of 2.
        bits = scale_input.to(tl.int32, bitcast=True)
        mantissa = bits & 0x007FFFFF
        scale_fp = ((bits & 0x7F800000) + tl.where(mantissa != 0, 0x00800000, 0)).to(
            tl.float32, bitcast=True
        )

    # Common tail for both modes: clamp + exact reciprocal
    fp32_min_normal = tl.exp2(-126.0)
    scale_fp = tl.maximum(scale_fp, fp32_min_normal)

    # Exact reciprocal via exponent flip: 2^(e-127) -> 2^(127-e)
    rcp_bits = (254 << 23) - scale_fp.to(tl.int32, bitcast=True)
    rcp_scale = rcp_bits.to(tl.float32, bitcast=True)

    return scale_fp, rcp_scale


# ---------------------------------------------------------------------------
# fused_rowwise_colwise: rowwise(x) + colwise(x)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_rowwise_colwise_fp8_kernel(
    x_ptr,
    out_tok_ptr,
    scale_tok_ptr,
    out_ch_ptr,
    scale_ch_ptr,
    M,
    N,
    stride_x_row,
    stride_x_col,
    stride_out_tok_row,
    scale_tok_cols,
    stride_out_ch_row,
    scale_ch_cols,
    block_to_group_ptr,
    cumsum_ptr,
    BLOCK_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    WRITE_KMAJOR: tl.constexpr = False,
):
    """Fused rowwise + colwise FP8 quantization from a single tile load.

    Each program processes a 128 x BLOCK_N tile.  M must be divisible by 128.

    Rowwise path: 128-element column-block scaling (axis=1 reduction).
    Colwise path: 128-row group, per-column scaling (axis=0 reduction).

    When WRITE_KMAJOR=True (Hopper grouped wgrad), the colwise FP8 output is
    written directly in K-major flat layout - per-group (K_g, N) blocks are
    transposed to (N, K_g) and concatenated - eliminating a separate transpose
    kernel.  Colwise scales are written transposed: (N, M//128) instead of
    (M//128, N).  Requires block_to_group_ptr and cumsum_ptr (cumulative
    row counts without a leading zero, i.e. grouped_mm_offs directly).
    """
    BLOCK_ROWS: tl.constexpr = 128
    CHUNKS: tl.constexpr = BLOCK_N // BLOCK_ROWS

    pid_row = tl.program_id(0)  # which 128-row group
    pid_col = tl.program_id(1)  # which BLOCK_N column block

    row_start = pid_row * BLOCK_ROWS
    col_start = pid_col * BLOCK_N

    row_offs = row_start + tl.arange(0, BLOCK_ROWS)
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # Load 128 x BLOCK_N tile (single global-memory read) -- keep bf16
    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)

    # --- Rowwise path (axis=1 reduction, per-row, per-128-col-block) ---
    abs_reshaped = tl.reshape(abs_bf16, BLOCK_ROWS * CHUNKS, BLOCK_ROWS)
    tok_amax = tl.max(abs_reshaped, axis=1)  # (128 * CHUNKS,)
    tok_scale, tok_rcp = _compute_fp8_scale(tok_amax, SCALING_MODE)

    x_f32 = x_bf16.to(tl.float32)
    x_reshaped = tl.reshape(x_f32, BLOCK_ROWS * CHUNKS, BLOCK_ROWS)
    tok_fp8_flat = (x_reshaped * tok_rcp[:, None]).to(tl.float8e4nv)
    tok_fp8 = tl.reshape(tok_fp8_flat, BLOCK_ROWS, BLOCK_N)

    # Store rowwise FP8 data: out_tok (M, N)
    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=mask)

    # Store rowwise float32 scales: (M, scale_tok_cols)
    tok_scale_2d = tl.reshape(tok_scale, BLOCK_ROWS, CHUNKS)
    tok_col_offs = pid_col * CHUNKS + tl.arange(0, CHUNKS)
    tok_scale_ptrs = scale_tok_ptr + row_offs[:, None] * scale_tok_cols + tok_col_offs[None, :]
    tok_scale_mask = (row_offs[:, None] < M) & (tok_col_offs[None, :] < scale_tok_cols)
    tl.store(tok_scale_ptrs, tok_scale_2d, mask=tok_scale_mask)

    # --- Colwise path (axis=0 reduction, per-column within 128-row group) ---
    ch_amax = tl.max(abs_bf16, axis=0)  # (BLOCK_N,)
    ch_scale, ch_rcp = _compute_fp8_scale(ch_amax, SCALING_MODE)

    ch_fp8 = (x_f32 * ch_rcp[None, :]).to(tl.float8e4nv)

    ch_scale_mask = col_offs < N

    if WRITE_KMAJOR:
        # K-major: write per-group (K_g, N) blocks transposed to (N, K_g),
        # concatenated into a 1D flat buffer.  Groups are 128-aligned so each
        # pid_row belongs to exactly one group.
        group_id = tl.load(block_to_group_ptr + pid_row).to(tl.int64)
        group_end = tl.load(cumsum_ptr + group_id).to(tl.int64)
        group_start = tl.load(cumsum_ptr + group_id - 1, mask=group_id > 0, other=0).to(tl.int64)
        k_g = group_end - group_start
        k_offset = row_start - group_start

        flat_base = group_start * N
        row_local = tl.arange(0, BLOCK_ROWS).to(tl.int64)
        out_offs = (
            flat_base + col_offs[None, :].to(tl.int64) * k_g + (k_offset + row_local)[:, None]
        )
        tl.store(out_ch_ptr + out_offs, ch_fp8, mask=mask)

        # Scales: (N, num_row_groups) transposed layout.
        # scale_ch_cols = num_row_groups when WRITE_KMAJOR=True.
        scale_ch_ptrs = scale_ch_ptr + col_offs * scale_ch_cols + pid_row
        tl.store(scale_ch_ptrs, ch_scale, mask=ch_scale_mask)
    else:
        # MN-major: out_ch (M, N), scale_ch (M//128, N)
        out_ch_ptrs = out_ch_ptr + row_offs[:, None] * stride_out_ch_row + col_offs[None, :]
        tl.store(out_ch_ptrs, ch_fp8, mask=mask)

        scale_ch_ptrs = scale_ch_ptr + pid_row * scale_ch_cols + col_offs
        tl.store(scale_ch_ptrs, ch_scale, mask=ch_scale_mask)


def fused_rowwise_colwise_cast_to_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise + colwise FP8 quantization (single read of *x*).

    Args:
        x: 2D tensor (M, N) in BF16/FP32.  M must be divisible by 128.

    Returns:
        (tok_fp8, tok_scale, ch_fp8, ch_scale):
            tok_fp8 -- float8_e4m3fn (M, N),
            tok_scale -- float32 (M, ceil(N/128)),
            ch_fp8  -- float8_e4m3fn (M, N),
            ch_scale -- float32 (M//128, N).
    """
    assert x.ndim == 2 and x.shape[0] % 128 == 0, (
        f"Expected 2D tensor with M%128==0, got shape {x.shape}"
    )
    M, N = x.shape
    BLOCK = 128

    num_row_groups = M // BLOCK
    scale_tok_cols = (N + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_ch = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_ch = torch.empty((num_row_groups, N), dtype=torch.float32, device=x.device)

    scaling_mode = "e8m0" if _USE_E8M0_SCALES else "fp32"

    _BLOCK_N = 128
    grid = (num_row_groups, (N + _BLOCK_N - 1) // _BLOCK_N)

    _fused_rowwise_colwise_fp8_kernel[grid](
        x_ptr=x,
        out_tok_ptr=out_tok,
        scale_tok_ptr=scale_tok,
        out_ch_ptr=out_ch,
        scale_ch_ptr=scale_ch,
        M=M,
        N=N,
        stride_x_row=x.stride(0),
        stride_x_col=x.stride(1),
        stride_out_tok_row=N,
        scale_tok_cols=scale_tok_cols,
        stride_out_ch_row=N,
        scale_ch_cols=N,
        block_to_group_ptr=out_tok,
        cumsum_ptr=out_tok,
        BLOCK_N=_BLOCK_N,
        SCALING_MODE=scaling_mode,
        WRITE_KMAJOR=False,
        num_warps=4,
        num_stages=2,
    )

    return out_tok, scale_tok, out_ch, scale_ch


def fused_rowwise_kmajor_cast_to_fp8(
    x: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise + K-major colwise FP8 quantization (single read of *x*).

    Same as :func:`fused_rowwise_colwise_cast_to_fp8` for the rowwise outputs,
    but the colwise FP8 data is written directly in K-major flat layout used by
    the grouped wgrad GEMM on Hopper, and colwise scales are transposed.

    Args:
        x: 2D tensor (M, N) in BF16/FP32.  M must be divisible by 128.
        grouped_mm_offs: 1D int32 tensor (num_groups,) of cumulative row counts.
            Each value must be divisible by 128.

    Returns:
        (tok_fp8, tok_scale, kmajor_fp8, kmajor_scale):
            tok_fp8    -- float8_e4m3fn (M, N),
            tok_scale  -- float32 (M, ceil(N/128)),
            kmajor_fp8 -- float8_e4m3fn 1D (M*N,) in K-major flat layout,
            kmajor_scale -- float32 (N, M//128) transposed colwise scales.
    """
    assert x.ndim == 2 and x.shape[0] % 128 == 0, (
        f"Expected 2D tensor with M%128==0, got shape {x.shape}"
    )
    M, N = x.shape
    BLOCK = 128

    num_row_groups = M // BLOCK
    scale_tok_cols = (N + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    kmajor_fp8 = torch.empty(M * N, dtype=torch.float8_e4m3fn, device=x.device)
    kmajor_scale = torch.empty((N, num_row_groups), dtype=torch.float32, device=x.device)

    # Precompute per-block group mapping (tiny - num_row_groups elements)
    block_starts = torch.arange(0, M, BLOCK, device=x.device, dtype=torch.int64)
    block_to_group = torch.searchsorted(
        grouped_mm_offs.to(torch.int64),
        block_starts,
        right=True,
    ).to(torch.int32)
    block_to_group.clamp_(max=grouped_mm_offs.shape[0] - 1)

    scaling_mode = "e8m0" if _USE_E8M0_SCALES else "fp32"

    _BLOCK_N = 128
    grid = (num_row_groups, (N + _BLOCK_N - 1) // _BLOCK_N)

    _fused_rowwise_colwise_fp8_kernel[grid](
        x_ptr=x,
        out_tok_ptr=out_tok,
        scale_tok_ptr=scale_tok,
        out_ch_ptr=kmajor_fp8,
        scale_ch_ptr=kmajor_scale,
        M=M,
        N=N,
        stride_x_row=x.stride(0),
        stride_x_col=x.stride(1),
        stride_out_tok_row=N,
        scale_tok_cols=scale_tok_cols,
        stride_out_ch_row=0,
        scale_ch_cols=num_row_groups,
        block_to_group_ptr=block_to_group,
        cumsum_ptr=grouped_mm_offs,
        BLOCK_N=_BLOCK_N,
        SCALING_MODE=scaling_mode,
        WRITE_KMAJOR=True,
        num_warps=4,
        num_stages=2,
    )

    return out_tok, scale_tok, kmajor_fp8, kmajor_scale


# ---------------------------------------------------------------------------
# fused_rowwise_transpose: rowwise(x) + rowwise(x.T)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_rowwise_transpose_fp8_kernel(
    x_ptr,
    out_tok_ptr,
    scale_tok_ptr,
    out_t_ptr,
    scale_t_ptr,
    M,
    N,
    stride_x_row,
    stride_x_col,
    stride_out_tok_row,
    scale_tok_cols,
    stride_out_t_row,
    stride_scale_t_row,
    BLOCK_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
):
    """Fused rowwise quantization of x (M, N) and x.T (N, M).

    Each program processes a 128 x BLOCK_N tile of x.

    Original rowwise: axis=1 reduction -> (M, N) FP8 + float32 scales.
    Transposed rowwise: axis=0 reduction -> (N, M) FP8 + float32 scales.
    """
    BLOCK_ROWS: tl.constexpr = 128
    CHUNKS: tl.constexpr = BLOCK_N // BLOCK_ROWS

    pid_row = tl.program_id(0)  # which 128-row group of x
    pid_col = tl.program_id(1)  # which BLOCK_N column block of x

    row_start = pid_row * BLOCK_ROWS
    col_start = pid_col * BLOCK_N

    row_offs = row_start + tl.arange(0, BLOCK_ROWS)
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # Load 128 x BLOCK_N tile (single global-memory read) -- keep bf16
    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)

    # --- Rowwise on x (axis=1 reduction) ---
    abs_reshaped = tl.reshape(abs_bf16, BLOCK_ROWS * CHUNKS, BLOCK_ROWS)
    tok_amax = tl.max(abs_reshaped, axis=1)  # (128 * CHUNKS,)
    tok_scale, tok_rcp = _compute_fp8_scale(tok_amax, SCALING_MODE)

    x_f32 = x_bf16.to(tl.float32)
    x_reshaped = tl.reshape(x_f32, BLOCK_ROWS * CHUNKS, BLOCK_ROWS)
    tok_fp8_flat = (x_reshaped * tok_rcp[:, None]).to(tl.float8e4nv)
    tok_fp8 = tl.reshape(tok_fp8_flat, BLOCK_ROWS, BLOCK_N)

    # Store rowwise FP8 data: out_tok (M, N)
    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=mask)

    # Store rowwise float32 scales: (M, scale_tok_cols)
    tok_scale_2d = tl.reshape(tok_scale, BLOCK_ROWS, CHUNKS)
    tok_col_offs = pid_col * CHUNKS + tl.arange(0, CHUNKS)
    tok_scale_ptrs = scale_tok_ptr + row_offs[:, None] * scale_tok_cols + tok_col_offs[None, :]
    tok_scale_mask = (row_offs[:, None] < M) & (tok_col_offs[None, :] < scale_tok_cols)
    tl.store(tok_scale_ptrs, tok_scale_2d, mask=tok_scale_mask)

    # --- Transposed rowwise on x.T (axis=0 reduction of x tile) ---
    t_amax = tl.max(abs_bf16, axis=0)  # (BLOCK_N,)
    t_scale, t_rcp = _compute_fp8_scale(t_amax, SCALING_MODE)

    # Scale each column of x by its transpose-token scale, cast to FP8
    t_fp8 = (x_f32 * t_rcp[None, :]).to(tl.float8e4nv)

    # Transpose to (BLOCK_N, 128) for coalesced store to out_t (N, M)
    t_fp8_t = tl.trans(t_fp8)  # (BLOCK_N, 128)

    t_row_offs = col_start + tl.arange(0, BLOCK_N)
    t_col_offs = row_start + tl.arange(0, BLOCK_ROWS)
    out_t_ptrs = out_t_ptr + t_row_offs[:, None] * stride_out_t_row + t_col_offs[None, :]
    t_mask = (t_row_offs[:, None] < N) & (t_col_offs[None, :] < M)
    tl.store(out_t_ptrs, t_fp8_t, mask=t_mask)

    # Store transposed float32 scales: scale_t (N, ceil(M/128))
    # Each program contributes one column (pid_row) per transposed row.
    scale_t_ptrs = scale_t_ptr + col_offs * stride_scale_t_row + pid_row
    scale_t_mask = col_offs < N
    tl.store(scale_t_ptrs, t_scale, mask=scale_t_mask)


def fused_rowwise_transpose_cast_to_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise FP8 quantization of x *and* x.T (single read of *x*).

    Args:
        x: 2D tensor (M, N) in BF16/FP32.

    Returns:
        (tok_fp8, tok_scale, t_fp8, t_scale):
            tok_fp8   -- float8_e4m3fn (M, N),
            tok_scale -- float32 (M, ceil(N/128)),
            t_fp8     -- float8_e4m3fn (N, M),
            t_scale   -- float32 (N, ceil(M/128)).
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    BLOCK = 128

    scale_tok_cols = (N + BLOCK - 1) // BLOCK
    scale_t_cols = (M + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((N, scale_t_cols), dtype=torch.float32, device=x.device)

    scaling_mode = "e8m0" if _USE_E8M0_SCALES else "fp32"

    _BLOCK_N = 128
    grid = (
        (M + BLOCK - 1) // BLOCK,
        (N + _BLOCK_N - 1) // _BLOCK_N,
    )

    _fused_rowwise_transpose_fp8_kernel[grid](
        x_ptr=x,
        out_tok_ptr=out_tok,
        scale_tok_ptr=scale_tok,
        out_t_ptr=out_t,
        scale_t_ptr=scale_t,
        M=M,
        N=N,
        stride_x_row=x.stride(0),
        stride_x_col=x.stride(1),
        stride_out_tok_row=N,
        scale_tok_cols=scale_tok_cols,
        stride_out_t_row=M,
        stride_scale_t_row=scale_t_cols,
        BLOCK_N=_BLOCK_N,
        SCALING_MODE=scaling_mode,
        num_warps=4,
        num_stages=2,
    )

    return out_tok, scale_tok, out_t, scale_t


# ---------------------------------------------------------------------------
# fused_rowwise_blockwise_transpose: rowwise(x) + blockwise(x.T)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_rowwise_blockwise_transpose_fp8_kernel(
    x_ptr,
    out_tok_ptr,
    scale_tok_ptr,
    out_blk_t_ptr,
    scale_blk_t_ptr,
    M,
    K,
    stride_x_row,
    stride_x_col,
    stride_out_tok_row,
    stride_scale_tok_row,
    stride_out_blk_t_row,
    scale_blk_t_cols,
    SCALING_MODE: tl.constexpr,
):
    """Fused rowwise quantization of x and blockwise quantization of x.T.

    Each program processes a single 128x128 tile of x.

    Rowwise path: row-wise amax over 128 columns -> (128,) scales ->
        FP8 output to ``out_tok (M, K)`` + scales to ``scale_tok (M, ceil(K/128))``.
    Blockwise transpose path: tile-wide amax (reuses row amax) -> scalar scale ->
        transposed FP8 output to ``out_blk_t (K, M)`` + scale to
        ``scale_blk_t (ceil(K/128), ceil(M/128))``.
    """
    BLOCK: tl.constexpr = 128

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_start = pid_row * BLOCK
    col_start = pid_col * BLOCK

    row_offs = row_start + tl.arange(0, BLOCK)
    col_offs = col_start + tl.arange(0, BLOCK)

    # Load 128x128 tile (single global-memory read) -- keep bf16
    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < K)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)

    # --- Rowwise path (one 128-element column block per tile) ---
    tok_amax = tl.max(abs_bf16, axis=1)  # (128,)
    tok_scale, tok_rcp = _compute_fp8_scale(tok_amax, SCALING_MODE)  # (128,)

    x_f32 = x_bf16.to(tl.float32)
    tok_fp8 = (x_f32 * tok_rcp[:, None]).to(tl.float8e4nv)

    # Store rowwise FP8 data: out_tok (M, K)
    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=mask)

    # Store rowwise scale: (M, ceil(K/128)), one value per row at column pid_col
    scale_tok_ptrs = scale_tok_ptr + row_offs * stride_scale_tok_row + pid_col
    scale_tok_mask = row_offs < M
    tl.store(scale_tok_ptrs, tok_scale, mask=scale_tok_mask)

    # --- Blockwise transpose path (reuse tok_amax) ---
    block_amax = tl.max(tok_amax, axis=0)  # scalar
    blk_scale, blk_rcp = _compute_fp8_scale(block_amax, SCALING_MODE)  # scalar

    blk_fp8 = (x_f32 * blk_rcp).to(tl.float8e4nv)

    # Transpose to (128, 128) for coalesced store to out_blk_t (K, M)
    blk_fp8_t = tl.trans(blk_fp8)

    # Store transposed FP8 data: rows [col_start..+128), cols [row_start..+128)
    t_row_offs = col_start + tl.arange(0, BLOCK)
    t_col_offs = row_start + tl.arange(0, BLOCK)
    out_blk_t_ptrs = (
        out_blk_t_ptr + t_row_offs[:, None] * stride_out_blk_t_row + t_col_offs[None, :]
    )
    t_mask = (t_row_offs[:, None] < K) & (t_col_offs[None, :] < M)
    tl.store(out_blk_t_ptrs, blk_fp8_t, mask=t_mask)

    # Store block scale: scale_blk_t (ceil(K/128), ceil(M/128))
    # Block (pid_col, pid_row) of x.T maps to tile (pid_row, pid_col) of x
    scale_blk_t_offset = pid_col * scale_blk_t_cols + pid_row
    tl.store(scale_blk_t_ptr + scale_blk_t_offset, blk_scale)


def fused_rowwise_blockwise_transpose_cast_to_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise FP8 quant of *x* and blockwise FP8 quant of *x*.T.

    Args:
        x: 2D tensor (M, K) in BF16/FP32.

    Returns:
        (tok_fp8, tok_scale, blk_t_fp8, blk_t_scale):
            tok_fp8     -- float8_e4m3fn (M, K),
            tok_scale   -- float32 (M, ceil(K/128)),
            blk_t_fp8   -- float8_e4m3fn (K, M),
            blk_t_scale -- float32 (ceil(K/128), ceil(M/128)).
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    BLOCK = 128

    scale_tok_cols = (K + BLOCK - 1) // BLOCK
    scale_blk_t_rows = (K + BLOCK - 1) // BLOCK
    scale_blk_t_cols = (M + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_blk_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_blk_t = torch.empty(
        (scale_blk_t_rows, scale_blk_t_cols), dtype=torch.float32, device=x.device
    )

    scaling_mode = "e8m0" if _USE_E8M0_SCALES else "fp32"

    grid = (
        (M + BLOCK - 1) // BLOCK,
        (K + BLOCK - 1) // BLOCK,
    )

    _fused_rowwise_blockwise_transpose_fp8_kernel[grid](
        x_ptr=x,
        out_tok_ptr=out_tok,
        scale_tok_ptr=scale_tok,
        out_blk_t_ptr=out_blk_t,
        scale_blk_t_ptr=scale_blk_t,
        M=M,
        K=K,
        stride_x_row=x.stride(0),
        stride_x_col=x.stride(1),
        stride_out_tok_row=K,
        stride_scale_tok_row=scale_tok_cols,
        stride_out_blk_t_row=M,
        scale_blk_t_cols=scale_blk_t_cols,
        SCALING_MODE=scaling_mode,
        num_warps=4,
        num_stages=2,
    )

    return out_tok, scale_tok, out_blk_t, scale_blk_t


# ---------------------------------------------------------------------------
# fused_blockwise_transpose: blockwise(x) + blockwise(x.T), 2D
# ---------------------------------------------------------------------------


@triton.jit
def _fused_blockwise_transpose_fp8_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    out_t_ptr,
    scale_t_ptr,
    M,
    K,
    stride_x_row,
    stride_x_col,
    stride_out_row,
    scale_cols,
    stride_out_t_row,
    scale_t_cols,
    SCALING_MODE: tl.constexpr,
):
    """Fused blockwise (128x128) FP8 quantization of x and x.T.

    Each program processes one 128x128 tile of x, computing a single amax
    and FP8 scale for the tile.  The quantized tile is stored to both
    ``out (M, K)`` and, transposed via ``tl.trans``, to ``out_t (K, M)``.
    The same scalar scale is written to both ``scale`` and ``scale_t`` at
    swapped indices.
    """
    BLOCK: tl.constexpr = 128

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_start = pid_row * BLOCK
    col_start = pid_col * BLOCK

    row_offs = row_start + tl.arange(0, BLOCK)
    col_offs = col_start + tl.arange(0, BLOCK)

    # Load 128x128 tile (single global-memory read) -- keep bf16
    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < K)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    # Single scalar amax for the 128x128 block
    abs_bf16 = tl.abs(x_bf16)
    row_max = tl.max(abs_bf16, axis=1)  # (BLOCK,)
    block_amax = tl.max(row_max, axis=0)  # scalar

    # Compute scalar scale
    scale_val, rcp_val = _compute_fp8_scale(block_amax, SCALING_MODE)

    # Scale and cast (once)
    x_f32 = x_bf16.to(tl.float32)
    fp8_tile = (x_f32 * rcp_val).to(tl.float8e4nv)

    # Store FP8 data: out(M, K) at tile (pid_row, pid_col)
    out_ptrs = out_ptr + row_offs[:, None] * stride_out_row + col_offs[None, :]
    tl.store(out_ptrs, fp8_tile, mask=mask)

    # Transpose in registers
    fp8_tile_t = tl.trans(fp8_tile)  # (BLOCK, BLOCK)

    # Store transposed FP8 data: out_t(K, M) at tile (pid_col, pid_row)
    t_row_offs = col_start + tl.arange(0, BLOCK)
    t_col_offs = row_start + tl.arange(0, BLOCK)
    out_t_ptrs = out_t_ptr + t_row_offs[:, None] * stride_out_t_row + t_col_offs[None, :]
    t_mask = (t_row_offs[:, None] < K) & (t_col_offs[None, :] < M)
    tl.store(out_t_ptrs, fp8_tile_t, mask=t_mask)

    # Store scale: one value at (pid_row, pid_col)
    scale_offset = pid_row * scale_cols + pid_col
    tl.store(scale_ptr + scale_offset, scale_val)

    # Store scale_t: same value at (pid_col, pid_row) -- swapped indices
    scale_t_offset = pid_col * scale_t_cols + pid_row
    tl.store(scale_t_ptr + scale_t_offset, scale_val)


def fused_blockwise_transpose_cast_to_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused blockwise FP8 quantization of x *and* x.T (single read of *x*).

    Args:
        x: 2D tensor (M, K) in BF16/FP32.

    Returns:
        (fp8, scale, fp8_t, scale_t):
            fp8     -- float8_e4m3fn (M, K),
            scale   -- float32 (ceil(M/128), ceil(K/128)),
            fp8_t   -- float8_e4m3fn (K, M),
            scale_t -- float32 (ceil(K/128), ceil(M/128)).
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    BLOCK = 128
    scale_rows = (M + BLOCK - 1) // BLOCK
    scale_cols = (K + BLOCK - 1) // BLOCK

    out = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((scale_rows, scale_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((scale_cols, scale_rows), dtype=torch.float32, device=x.device)

    scaling_mode = "e8m0" if _USE_E8M0_SCALES else "fp32"

    grid = (scale_rows, scale_cols)

    _fused_blockwise_transpose_fp8_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        scale_ptr=scale,
        out_t_ptr=out_t,
        scale_t_ptr=scale_t,
        M=M,
        K=K,
        stride_x_row=x.stride(0),
        stride_x_col=x.stride(1),
        stride_out_row=K,
        scale_cols=scale_cols,
        stride_out_t_row=M,
        scale_t_cols=scale_rows,
        SCALING_MODE=scaling_mode,
        num_warps=4,
        num_stages=2,
    )

    return out, scale, out_t, scale_t


# ---------------------------------------------------------------------------
# fused_blockwise_transpose_batched: blockwise(x) + blockwise(x.T), 3D
# ---------------------------------------------------------------------------


@triton.jit
def _fused_blockwise_transpose_fp8_batched_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    out_t_ptr,
    scale_t_ptr,
    G,
    N,
    K,
    stride_x_g,
    stride_x_row,
    stride_x_col,
    stride_out_g,
    stride_out_row,
    scale_cols,
    stride_scale_g,
    stride_out_t_g,
    stride_out_t_row,
    scale_t_cols,
    stride_scale_t_g,
    SCALING_MODE: tl.constexpr,
):
    """Fused batched blockwise (128x128) FP8 quantization of x and x.transpose(1,2).

    Grid: (ceil(N/128), ceil(K/128), G)

    Each program processes one 128x128 tile within one group, computing a
    single amax and FP8 scale.  The quantized tile is stored to both
    ``out (G, N, K)`` and, transposed, to ``out_t (G, K, N)``.  The same
    scalar scale is written to ``scale`` and ``scale_t`` at swapped indices.
    """
    BLOCK: tl.constexpr = 128

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    pid_g = tl.program_id(2)

    row_start = pid_row * BLOCK
    col_start = pid_col * BLOCK

    row_offs = row_start + tl.arange(0, BLOCK)
    col_offs = col_start + tl.arange(0, BLOCK)

    # Load 128x128 tile from group pid_g -- keep bf16
    base = x_ptr + pid_g * stride_x_g
    ptrs = base + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < N) & (col_offs[None, :] < K)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    # Single scalar amax for the block
    abs_bf16 = tl.abs(x_bf16)
    row_max = tl.max(abs_bf16, axis=1)  # (BLOCK,)
    block_amax = tl.max(row_max, axis=0)  # scalar

    scale_val, rcp_val = _compute_fp8_scale(block_amax, SCALING_MODE)

    # Scale and cast (once)
    x_f32 = x_bf16.to(tl.float32)
    fp8_tile = (x_f32 * rcp_val).to(tl.float8e4nv)

    # Store FP8 data: out(G, N, K)
    out_base = out_ptr + pid_g * stride_out_g
    out_ptrs = out_base + row_offs[:, None] * stride_out_row + col_offs[None, :]
    tl.store(out_ptrs, fp8_tile, mask=mask)

    # Transpose in registers
    fp8_tile_t = tl.trans(fp8_tile)  # (BLOCK, BLOCK)

    # Store transposed FP8 data: out_t(G, K, N)
    t_row_offs = col_start + tl.arange(0, BLOCK)
    t_col_offs = row_start + tl.arange(0, BLOCK)
    out_t_base = out_t_ptr + pid_g * stride_out_t_g
    out_t_ptrs = out_t_base + t_row_offs[:, None] * stride_out_t_row + t_col_offs[None, :]
    t_mask = (t_row_offs[:, None] < K) & (t_col_offs[None, :] < N)
    tl.store(out_t_ptrs, fp8_tile_t, mask=t_mask)

    # Store scale: (G, ceil(N/128), ceil(K/128)) at (pid_g, pid_row, pid_col)
    scale_offset = pid_g * stride_scale_g + pid_row * scale_cols + pid_col
    tl.store(scale_ptr + scale_offset, scale_val)

    # Store scale_t: (G, ceil(K/128), ceil(N/128)) at (pid_g, pid_col, pid_row)
    scale_t_offset = pid_g * stride_scale_t_g + pid_col * scale_t_cols + pid_row
    tl.store(scale_t_ptr + scale_t_offset, scale_val)


def fused_blockwise_transpose_cast_to_fp8_batched(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused blockwise FP8 quantization of x *and* x.transpose(1,2) for batched (G, N, K).

    Args:
        x: 3D tensor (G, N, K) in BF16/FP32.

    Returns:
        (fp8, scale, fp8_t, scale_t):
            fp8     -- float8_e4m3fn (G, N, K),
            scale   -- float32 (G, ceil(N/128), ceil(K/128)),
            fp8_t   -- float8_e4m3fn (G, K, N),
            scale_t -- float32 (G, ceil(K/128), ceil(N/128)).
    """
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    G, N, K = x.shape
    BLOCK = 128
    scale_rows = (N + BLOCK - 1) // BLOCK
    scale_cols = (K + BLOCK - 1) // BLOCK

    out = torch.empty((G, N, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((G, scale_rows, scale_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((G, scale_cols, scale_rows), dtype=torch.float32, device=x.device)

    scaling_mode = "e8m0" if _USE_E8M0_SCALES else "fp32"

    grid = (scale_rows, scale_cols, G)

    _fused_blockwise_transpose_fp8_batched_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        scale_ptr=scale,
        out_t_ptr=out_t,
        scale_t_ptr=scale_t,
        G=G,
        N=N,
        K=K,
        stride_x_g=x.stride(0),
        stride_x_row=x.stride(1),
        stride_x_col=x.stride(2),
        stride_out_g=N * K,
        stride_out_row=K,
        scale_cols=scale_cols,
        stride_scale_g=scale_rows * scale_cols,
        stride_out_t_g=K * N,
        stride_out_t_row=N,
        scale_t_cols=scale_rows,
        stride_scale_t_g=scale_cols * scale_rows,
        SCALING_MODE=scaling_mode,
        num_warps=4,
        num_stages=2,
    )

    return out, scale, out_t, scale_t
