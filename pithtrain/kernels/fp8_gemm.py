"""
Every 1x128 activation scale is fp32 of logical shape (rows, ceil(K / 128)) with unspecified strides
(MN-major on SM90, row-major on SM100); the GEMMs take either.
"""

from typing import Tuple

import deep_gemm
import torch

from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.kernels.fp8_rowwise_blockwise_transpose import fp8_quant_act
from pithtrain.kernels.fp8_rowwise_colwise import fp8cast_rowwise_colwise
from pithtrain.kernels.fp8_rowwise_kmajor import fp8cast_rowwise_kmajor
from pithtrain.kernels.fp8_rowwise_transpose import fp8_quant_grad

__all__ = ["fp8_gemm_fwd", "fp8_gemm_bwd", "fp8_fused_experts_supported", "fp8_gemm_nt", "fp8_grouped_wgrad_empty", "fp8_grouped_gemm", "k_grouped_fp8_gemm", "k_grouped_safe_size", "fp8_group_indices", "fp8_quant_grouped"]


def fp8_gemm_nt(a_fp8: torch.Tensor, a_scale: torch.Tensor, b_fp8: torch.Tensor, b_scale: torch.Tensor, m: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    out = a @ b.T, fresh, fp32 accumulation. dtype must be bf16 (DeepGEMM on SM90 rejects others).

    - a_fp8: e4m3 (m, K) row-major; a_scale: its 1x128 scale, fp32 (m, ceil(K / 128)), any strides.
    - b_fp8: e4m3 (n, K) row-major; b_scale: its 128x128 scale, fp32 (ceil(n / 128), ceil(K / 128))
      row-major, as fp8_quant_weight returns it.
    """
    out = torch.empty((m, n), device=device, dtype=dtype)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, a_scale), (b_fp8, b_scale), out)
    return out


def fp8_gemm_fwd(input_2d: torch.Tensor, weight_fp8: torch.Tensor, scale_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward FP8 linear y = x @ W.T, returning (output, input_t_fp8, scale_input_t).

    - input_2d: x, bf16 (m, K), quantized rowwise 1x128.
    - weight_fp8, scale_weight: W, e4m3 (n, K) with its 128x128 scale as
      fp8_quant_weight returns it.
    - output: fresh (m, n) in input_2d's dtype.
    - input_t_fp8, scale_input_t: x.T (K, m) e4m3 quantized per 128x128 tile, scale fp32 (ceil(K /
      128), ceil(m / 128)) row-major; the activation operand of fp8_gemm_bwd.
    """
    M, N = input_2d.shape[0], weight_fp8.shape[0]
    input_fp8, scale_input, input_t_fp8, scale_input_t = fp8_quant_act(input_2d)
    output = fp8_gemm_nt(input_fp8, scale_input, weight_fp8, scale_weight, M, N, input_2d.device, input_2d.dtype)
    return output, input_t_fp8, scale_input_t


def fp8_gemm_bwd(grad_2d: torch.Tensor, weight_t_fp8: torch.Tensor, scale_weight_t: torch.Tensor, input_t_fp8: torch.Tensor, scale_input_t: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward FP8 linear of y = x @ W.T, returning (grad_input, weight_grad).

    - grad_2d: the gradient of y, bf16 (m, n).
    - weight_t_fp8, scale_weight_t: W.T, e4m3 (k, n) with its 128x128 scale (the transpose
      pair of fp8_quant_weight).
    - input_t_fp8, scale_input_t: x.T (k, m) blockwise, as fp8_gemm_fwd returns them.
    - k: the input features.
    - grad_input = grad @ W, fresh (m, k); weight_grad = grad.T @ x, fresh (n, k) (not accumulated);
      both in grad_2d's dtype.
    """
    M, N = grad_2d.shape
    grad_fp8, scale_grad, grad_t_fp8, scale_grad_t = fp8_quant_grad(grad_2d)
    grad_input = fp8_gemm_nt(grad_fp8, scale_grad, weight_t_fp8, scale_weight_t, M, k, grad_2d.device, grad_2d.dtype)
    weight_grad = fp8_gemm_nt(grad_t_fp8, scale_grad_t, input_t_fp8, scale_input_t, N, k, grad_2d.device, grad_2d.dtype)
    return grad_input, weight_grad


def m_grouped_fp8_gemm_nt(a, b, out, grouped_mm_offs, M, group_indices=None):
    match ARCH_MAJOR:
        case 9:
            if group_indices is None:
                group_indices = fp8_group_indices(grouped_mm_offs, M)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, out, group_indices)
        case 10:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, out, grouped_mm_offs, use_psum_layout=True, expected_m_for_psum_layout=M)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


K_GROUPED_MAX_ROWS = 65536
"""
Rows per call of DeepGEMM's SM90 k-grouped FP8 GEMM: it fails to launch (CUDA_ERROR_INVALID_VALUE)
once all groups' rows reach 116224, which imbalanced routing reaches, so larger calls are split.
"""


def pad_k_major_channels(x: tuple[torch.Tensor, torch.Tensor], ks: list, num_padded_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A K-major grouped operand (flat data of (n, k_g) row-major blocks, scales (n, rows / 128))
    re-laid out with num_padded_channels >= n; the extra channels are zero data with unit scales.
    """
    data, scale = x
    num_channels, num_blocks = scale.shape
    num_rows = sum(ks)
    out = data.new_zeros(num_padded_channels * num_rows)
    row_start = 0
    for group_rows in ks:
        if group_rows:
            row_end = row_start + group_rows
            out_block = out[row_start * num_padded_channels : row_end * num_padded_channels].view(num_padded_channels, group_rows)
            out_block[:num_channels].copy_(data[row_start * num_channels : row_end * num_channels].view(num_channels, group_rows))
        row_start += group_rows
    out_scale = torch.ones((num_blocks, num_padded_channels), device=scale.device, dtype=scale.dtype).T
    out_scale[:num_channels].copy_(scale)
    return out, out_scale


def k_grouped_runs_transposed(M: int, N: int) -> bool:
    """
    Whether the SM90 k-grouped FP8 GEMM of an (M, N) output runs transposed (see k_grouped_call).
    """
    return -(-M // 128) % 2 == 1 and -(-N // 128) % 2 == 0


def k_grouped_call(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], d: torch.Tensor, ks: list, ks_tensor: torch.Tensor) -> None:
    """
    One deep_gemm.k_grouped_fp8_gemm_nt_contiguous call accumulating into d, made safe from a
    DeepGEMM SM90 deadlock: the kernel runs 2-CTA clusters along M (block_m = 128, cluster_m = 2)
    and never finishes when ceil(M / 128) * ceil(N / block_n) is odd, M being the first operand's
    channels (e.g. (M, N) = (2880, 2880), (2880, 1024), (5760, 2880) hang). An even M-block count is
    always safe, so when a's is odd the call computes the transposed product b a^T if b's is even,
    and otherwise pads a's channels to a multiple of 256 with zeros.
    """
    M, N = a[1].shape[0], b[1].shape[0]
    if -(-M // 128) % 2 == 0:
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, ks_tensor, c=d)
    elif k_grouped_runs_transposed(M, N):
        # fp8_grouped_wgrad_empty stores d transposed for this case, so it accumulates in place.
        d_t = d.transpose(1, 2)
        if d_t.is_contiguous():
            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(b, a, d_t, ks, ks_tensor, c=d_t)
            return
        d_t = d_t.contiguous()
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(b, a, d_t, ks, ks_tensor, c=d_t)
        d.copy_(d_t.transpose(1, 2))
    else:
        M_padded = -(-M // 256) * 256
        d_padded = d.new_zeros((d.shape[0], M_padded, d.shape[2]))
        d_padded[:, :M].copy_(d)
        a_padded = pad_k_major_channels(a, ks, M_padded)
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a_padded, b, d_padded, ks, ks_tensor, c=d_padded)
        d.copy_(d_padded[:, :M])


def k_grouped_fp8_gemm_nt(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], d: torch.Tensor, ks: list, ks_tensor: torch.Tensor) -> None:
    """
    Accumulates into d in calls of at most K_GROUPED_MAX_ROWS rows. Runs of whole groups are
    contiguous slices of the K-major operands; a group longer than the limit is cut along k and
    accumulates chunk by chunk. Scale slices keep the parent's MN-major strides, so DeepGEMM reads
    them as is.
    """
    if sum(ks) <= K_GROUPED_MAX_ROWS:
        k_grouped_call(a, b, d, ks, ks_tensor)
        return
    channels = (a[1].shape[0], b[1].shape[0])

    def call_run(group_start: int, group_end: int, row_start: int, row_end: int, chunk: tuple[int, int] | None) -> None:
        """
        One k_grouped_call over groups [group_start, group_end), which span rows [row_start,
        row_end); with chunk = (chunk_start, chunk_end), only those rows of the single group.
        """
        operands = []
        for (data, scale), num_channels in zip((a, b), channels):
            if chunk is None:
                data, scale = data[row_start * num_channels : row_end * num_channels], scale[:, row_start // 128 : row_end // 128]
            else:
                group_rows, (chunk_start, chunk_end) = row_end - row_start, chunk
                data = data[row_start * num_channels : row_end * num_channels].view(num_channels, group_rows)[:, chunk_start:chunk_end].contiguous().view(-1)
                scale = scale[:, (row_start + chunk_start) // 128 : (row_start + chunk_end) // 128]
            operands.append((data, scale))
        sub_ks = ks[group_start:group_end] if chunk is None else [chunk[1] - chunk[0]]
        sub_ks_tensor = ks_tensor[group_start:group_end] if chunk is None else torch.tensor(sub_ks, dtype=ks_tensor.dtype, device=ks_tensor.device)
        out = d[group_start:group_end]
        k_grouped_call(operands[0], operands[1], out, sub_ks, sub_ks_tensor)

    # A run is the groups since group_start (from row_start); group_row_start is group_idx's first row.
    group_start, row_start, group_row_start = 0, 0, 0
    for group_idx, group_rows in enumerate(ks):
        group_row_end = group_row_start + group_rows
        if group_rows > K_GROUPED_MAX_ROWS:
            if group_idx > group_start:
                call_run(group_start, group_idx, row_start, group_row_start, None)
            for chunk_start in range(0, group_rows, K_GROUPED_MAX_ROWS):
                call_run(group_idx, group_idx + 1, group_row_start, group_row_end, (chunk_start, min(chunk_start + K_GROUPED_MAX_ROWS, group_rows)))
            group_start, row_start = group_idx + 1, group_row_end
        elif group_row_end - row_start > K_GROUPED_MAX_ROWS:
            call_run(group_start, group_idx, row_start, group_row_start, None)
            group_start, row_start = group_idx, group_row_start
        group_row_start = group_row_end
    if len(ks) > group_start:
        call_run(group_start, len(ks), row_start, group_row_start, None)


def k_grouped_fp8_gemm(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], d: torch.Tensor, ks: list, ks_tensor: torch.Tensor) -> None:
    """
    FP8 grouped weight gradient, accumulated in place: d[g] += a_g^T b_g in fp32 for each group g
    of ks rows.

    - d: fp32 (G, N, K), contiguous or as fp8_grouped_wgrad_empty returns it.
    - a, b: the per-channel FP8 data and scales (x_ch_fp8, x_ch_scale) of fp8_quant_grouped (or
      of the fused quantization ops) of the grad_output (N channels) and of the input (K channels),
      over the same grouped rows.
    - ks (host list) and ks_tensor (int32, device): per-group row counts, multiples of 128.
    - Either operand may have more channels than d (zero-padded, as the fused ops' padded=True
      writes it); the extra channels are dropped. Only the SM90 fused ops produce such operands.
    """
    match ARCH_MAJOR:
        case 9:
            G, N, K = d.shape
            NA, KB = a[1].shape[0], b[1].shape[0]
            if (NA, KB) == (N, K):
                k_grouped_fp8_gemm_nt(a, b, d, ks, ks_tensor)
                return
            if NA % 256 != 0 and KB % 256 == 0:
                d_padded = torch.zeros((G, KB, NA), device=d.device, dtype=torch.float32)
                k_grouped_fp8_gemm_nt(b, a, d_padded, ks, ks_tensor)
                d.add_(d_padded[:, :K, :N].transpose(1, 2))
            else:
                d_padded = torch.zeros((G, NA, KB), device=d.device, dtype=torch.float32)
                k_grouped_fp8_gemm_nt(a, b, d_padded, ks, ks_tensor)
                d.add_(d_padded[:, :N, :K])
        case 10:
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c=d)
            return
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_grouped_wgrad_empty(size: torch.Size, device: torch.device) -> torch.Tensor:
    """
    Uninitialized fp32 main_grad of shape size for an FP8 grouped weight (G, N, K), laid out so
    k_grouped_fp8_gemm accumulates into it in place: a transposed view of (G, K, N) storage where
    SM90 computes the product transposed, otherwise contiguous.
    """
    if len(size) == 3 and ARCH_MAJOR == 9:
        G, N, K = size
        if k_grouped_runs_transposed(N, K):
            return torch.empty((G, K, N), dtype=torch.float32, device=device).transpose(1, 2)
    return torch.empty(size, dtype=torch.float32, device=device)


def fp8_fused_experts_supported() -> bool:
    """
    Whether the fused FP8 expert ops run on this device (the activation-fused quantizations,
    scatter_quant, the fused dense SwiGLU MLP): True on SM90 only.
    """
    return ARCH_MAJOR == 9


def fp8_quant_grouped(input: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FP8 operands of a grouped-GEMM input from one read: (x_fp8, x_scale, x_ch_fp8, x_ch_scale).

    - input: bf16 (M, K) contiguous, rows grouped by expert, each group padded to a multiple of 128
      rows; grouped_mm_offs: int32 cumulative group ends (last = M).
    - x_fp8 e4m3 (M, K), x_scale fp32 (M, ceil(K / 128)): rowwise 1x128, the a
      operand of fp8_grouped_gemm.
    - x_ch_fp8, x_ch_scale: per-channel (128 rows x 1 within each group), the operand of
      k_grouped_fp8_gemm. SM90: K-major per group (group g's (K, ks[g]) row-major block flat at
      offs[g-1] * K), scale (K, M / 128). SM100: (M, K), scale (M / 128, K).

    Per 128-element block, scale = the power of two at or above max(amax, 1e-4) / 448 (at least
    2^-126) and fp8 = e4m3_satfinite(x / scale).
    """
    match ARCH_MAJOR:
        case 9:
            return fp8cast_rowwise_kmajor(input, grouped_mm_offs)
        case 10:
            return fp8cast_rowwise_colwise(input)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def fp8_grouped_gemm(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], n: int, grouped_mm_offs: torch.Tensor, group_indices: torch.Tensor | None = None) -> torch.Tensor:
    """
    m-grouped FP8 GEMM into a fresh bf16 (M, n) tensor: rows [offs[g-1], offs[g]) are a_g @
    b[g].T, fp32 accumulation.

    - a: (fp8 (M, K), 1x128 scale (M, ceil(K / 128))) as fp8_quant_grouped returns them.
    - b: (fp8 (G, n, K), 128x128 scale (G, ceil(n / 128), ceil(K / 128))) as
      fp8_quant_weight_grouped returns them.
    - grouped_mm_offs: int32 (G,) cumulative group ends, each group a multiple of 128 rows.
    - group_indices: optional hint from fp8_group_indices, computed when None.
    """
    M = a[0].shape[0]
    out = torch.empty((M, n), device=a[0].device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt(a, b, out, grouped_mm_offs, M, group_indices=group_indices)
    return out


def k_grouped_safe_size(n: int) -> int:
    """
    Channel count (a multiple of 256) to lay out a k-grouped wgrad operand with so DeepGEMM's SM90
    k-grouped GEMM can take it as the first operand without the 2-CTA deadlock (see k_grouped_call).
    """
    return -(-n // 256) * 256


def fp8_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor | None:
    """
    The group-id hint of fp8_grouped_gemm, reusable by every m-grouped GEMM over the same rows;
    None on SM100, which takes none.

    On SM90: int32 (M,) per-row group ids, e.g. grouped_mm_offs = [128, 256, 384, 512] gives [0,
    ..., 1, ..., 2, ..., 3, ...]; rows at or past the last offset get the last group.
    """
    if ARCH_MAJOR != 9:
        return None
    from pithtrain.kernels.group_indices_sm90 import group_indices, supports_group_indices
    if supports_group_indices(grouped_mm_offs, M):
        return group_indices(grouped_mm_offs, M)
    row_indices = torch.arange(M, device=grouped_mm_offs.device)
    group_ids = torch.searchsorted(grouped_mm_offs, row_indices, right=True).to(torch.int32)
    group_ids.clamp_(max=grouped_mm_offs.shape[0] - 1)
    return group_ids
