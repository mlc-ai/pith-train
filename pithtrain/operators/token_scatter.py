from typing import Optional

import torch
import triton
import triton.language as tl

from pithtrain.contexts import training
from pithtrain.operators.linear import ARCH_MAJOR

# -- Shared async D-to-H copy infrastructure --
# Used by both ScatterForGroupedGemm and prepare_dispatch to avoid
# per-call cudaStreamSynchronize overhead from .tolist() / .item().
#
# These are process-global, not per-thread, which is fine because the MoE ops only ever run
# on one thread at a time. DualPipeV launches forwards from a single thread and runs every
# backward through run_backward(), which disables autograd multithreading (dualpipe/utils.py),
# and nothing recomputes the MoE forward on a separate thread. If that ever changes (two
# models stepping in one process, or a plain multithreaded .backward() overlapping a forward)
# the shared buffers would race and corrupt the ks / EP-splits read-back, so key the buffer
# by threading.get_ident() and drop the shared copy stream/event at that point.

_pinned_buffers: dict[tuple[str, torch.dtype, int], torch.Tensor] = {}
_GEMM_ALLOC_ALIGNMENT = 1024


def get_pinned_buffer(name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
    # Cache pinned-memory buffers to avoid per-call allocation.
    # Freeing a pinned tensor triggers cudaEventRecordWithFlags (~10 us)
    # in PyTorch's CachingHostAllocator.
    key = (name, dtype, numel)
    buf = _pinned_buffers.get(key)
    if buf is None:
        buf = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
        _pinned_buffers[key] = buf
    return buf


@triton.jit
def _compute_group_offsets_kernel(
    expert_idxs_ptr,  # [m] int64
    group_counts_ptr,  # [num_groups] int64, pre-zeroed
    grouped_mm_offs_ptr,  # [num_groups] int32, output
    ks_ptr,  # [num_groups] int32, output: per-group padded sizes
    completion_counter_ptr,  # [1] int64, pre-zeroed
    m,
    num_ctas,
    PADDING_ALIGNMENT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # Phase 1: Bincount via atomic adds (parallel across CTAs)
    for start in range(pid * BLOCK, m, num_ctas * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < m
        eids = tl.load(expert_idxs_ptr + offs, mask=mask, other=0)
        tl.atomic_add(group_counts_ptr + eids, 1, mask=mask)

    # Barrier: last CTA to finish Phase 1 runs Phase 2
    is_last = tl.atomic_add(completion_counter_ptr, 1) == num_ctas - 1
    if is_last:
        # Phase 2: Pad + prefix-sum (sequential over num_groups, trivial)
        running_sum: tl.int64 = 0
        for g in tl.static_range(NUM_GROUPS):
            count = tl.load(group_counts_ptr + g)
            padded = ((count + PADDING_ALIGNMENT - 1) // PADDING_ALIGNMENT) * PADDING_ALIGNMENT
            running_sum = running_sum + padded
            off_mm = g + tl.arange(0, 1)
            tl.store(grouped_mm_offs_ptr + off_mm, running_sum.to(tl.int32))
            tl.store(ks_ptr + off_mm, padded.to(tl.int32))


@triton.jit
def _scatter_for_grouped_gemm_kernel(
    input_tokens_ptr,  # [m, hidden_size]
    expert_idxs_ptr,  # [m] int64
    grouped_mm_offs_ptr,  # [num_groups] int32, padded cumulative offsets
    group_counters_ptr,  # [num_groups] int64, zero-initialized (for atomic positioning)
    actual_counts_ptr,  # [num_groups] int64, actual counts from prep kernel
    output_tokens_ptr,  # [m_padded, hidden_size]
    reverse_shuffle_idxs_ptr,  # [m] int64
    hidden_size,
    stride_src_m,  # input_tokens.stride(0)
    stride_dst_m,  # output_tokens.stride(0)
    m,
    num_groups,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    GEMM_ALLOC_ALIGNMENT: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    expert_id = tl.load(expert_idxs_ptr + pid)
    pos_in_group = tl.atomic_add(group_counters_ptr + expert_id, 1)
    safe_idx = tl.maximum(expert_id - 1, 0)
    base_offset = tl.load(grouped_mm_offs_ptr + safe_idx, mask=expert_id > 0, other=0)
    global_pos = pos_in_group + base_offset.to(tl.int64)
    tl.store(reverse_shuffle_idxs_ptr + pid, global_pos)

    src_base = pid * stride_src_m
    dst_base = global_pos * stride_dst_m
    for i in tl.static_range(NUM_H_BLOCKS):
        h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = h_offs < hidden_size
        vals = tl.load(input_tokens_ptr + src_base + h_offs, mask=mask)
        tl.store(output_tokens_ptr + dst_base + h_offs, vals, mask=mask)

    # --- Zero padding rows for this group (distributed across programs) ---
    actual_count = tl.load(actual_counts_ptr + expert_id)
    end_offset = tl.load(grouped_mm_offs_ptr + expert_id).to(tl.int64)
    padded_count = end_offset - base_offset.to(tl.int64)
    pad_rows = padded_count - actual_count

    if pad_rows > 0:
        # Distribute pad_rows among actual_count programs in this group
        rows_per = (pad_rows + actual_count - 1) // actual_count  # ceil div
        my_start = pos_in_group * rows_per
        my_end = tl.minimum(my_start + rows_per, pad_rows)

        pad_base = base_offset.to(tl.int64) + actual_count
        for row_off in range(my_start, my_end):
            row_base = (pad_base + row_off) * stride_dst_m
            for i in tl.static_range(NUM_H_BLOCKS):
                h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
                mask = h_offs < hidden_size
                tl.store(
                    output_tokens_ptr + row_base + h_offs,
                    tl.zeros([BLOCK_H], dtype=input_tokens_ptr.dtype.element_ty),
                    mask=mask,
                )

    # --- Zero rows [actual_M, M_rounded) for GEMM allocation alignment ---
    last_group_end = tl.load(grouped_mm_offs_ptr + num_groups - 1).to(tl.int64)
    M_rounded = (
        (last_group_end + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT
    )
    align_pad = M_rounded - last_group_end

    if align_pad > 0:
        rows_per = (align_pad + m - 1) // m
        my_start = pid * rows_per
        my_end = tl.minimum(my_start + rows_per, align_pad)
        for row_off in range(my_start, my_end):
            row_base = (last_group_end + row_off) * stride_dst_m
            for i in tl.static_range(NUM_H_BLOCKS):
                h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
                mask = h_offs < hidden_size
                tl.store(
                    output_tokens_ptr + row_base + h_offs,
                    tl.zeros([BLOCK_H], dtype=input_tokens_ptr.dtype.element_ty),
                    mask=mask,
                )


class ScatterForGroupedGemm(torch.autograd.Function):
    """
    Custom autograd function wrapping the Triton scatter kernels so the
    scatter operation is differentiable.

    Forward: reorders input_tokens by expert assignment with zero-padding.
    Backward: gathers gradients back using reverse_shuffle_idxs.
    """

    _copy_stream: Optional[torch.cuda.Stream] = None
    _copy_event: Optional[torch.cuda.Event] = None

    @staticmethod
    def _get_copy_stream_and_event(device):
        if ScatterForGroupedGemm._copy_stream is None:
            ScatterForGroupedGemm._copy_stream = torch.cuda.Stream(device=device)
            ScatterForGroupedGemm._copy_event = torch.cuda.Event()
        return ScatterForGroupedGemm._copy_stream, ScatterForGroupedGemm._copy_event

    @staticmethod
    def forward(ctx, input_tokens, expert_idxs, num_groups, padding_alignment=128):
        m = input_tokens.shape[0]
        hidden_size = input_tokens.shape[1]
        device = input_tokens.device

        if m == 0:
            output_tokens = torch.empty((0, hidden_size), device=device, dtype=input_tokens.dtype)
            reverse_shuffle_idxs = torch.empty((0,), device=device, dtype=torch.int64)
            grouped_mm_offs = torch.empty((0,), device=device, dtype=torch.int32)
            ks_tensor = torch.empty((0,), device=device, dtype=torch.int32)
            ctx.save_for_backward(reverse_shuffle_idxs)
            return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, []

        # Over-allocate: tight upper bound, no GPU data needed (no .item() sync)
        max_m_padded = m + num_groups * (padding_alignment - 1)
        max_m_padded = (
            (max_m_padded + _GEMM_ALLOC_ALIGNMENT - 1)
            // _GEMM_ALLOC_ALIGNMENT
            * _GEMM_ALLOC_ALIGNMENT
        )

        # Over-allocated output - padding rows zeroed inside the scatter kernel
        # `output_tokens` will be zeroed inside the scatter kernel.
        output_tokens = torch.empty(
            (max_m_padded, hidden_size), device=device, dtype=input_tokens.dtype
        )
        # Small buffers for prep kernel
        reverse_shuffle_idxs = torch.empty((m,), device=device, dtype=torch.int64)
        grouped_mm_offs = torch.empty((num_groups,), device=device, dtype=torch.int32)
        ks_tensor = torch.empty((num_groups,), device=device, dtype=torch.int32)
        all_zeros = torch.zeros((num_groups * 2 + 1,), device=device, dtype=torch.int64)
        group_counts_1 = all_zeros[:num_groups]
        group_counts_2 = all_zeros[num_groups : num_groups * 2]
        completion_counter = all_zeros[num_groups * 2 :]

        # Prep kernel: fused bincount + pad + cumsum (1 launch)
        BLOCK = 1024
        num_ctas = min(triton.cdiv(m, BLOCK), 32)
        _compute_group_offsets_kernel[(num_ctas,)](
            expert_idxs,
            group_counts_1,
            grouped_mm_offs,
            ks_tensor,
            completion_counter,
            m,
            num_ctas,
            PADDING_ALIGNMENT=padding_alignment,
            NUM_GROUPS=num_groups,
            BLOCK=BLOCK,
        )

        # Async D-to-H: ks_tensor is done after the prep kernel; copy it on
        # a separate stream so the memcpy overlaps with the scatter kernel.
        copy_stream, copy_event = ScatterForGroupedGemm._get_copy_stream_and_event(device)
        copy_event.record()  # marks prep kernel completion on default stream

        # Scatter kernel (on default stream, overlaps with D-to-H copy)
        BLOCK_H = 256
        NUM_H_BLOCKS = triton.cdiv(hidden_size, BLOCK_H)
        _scatter_for_grouped_gemm_kernel[(m,)](
            input_tokens,
            expert_idxs,
            grouped_mm_offs,
            group_counts_2,
            group_counts_1,  # actual counts from prep kernel
            output_tokens,
            reverse_shuffle_idxs,
            hidden_size,
            input_tokens.stride(0),
            output_tokens.stride(0),
            m,
            num_groups,
            BLOCK_H=BLOCK_H,
            NUM_H_BLOCKS=NUM_H_BLOCKS,
            GEMM_ALLOC_ALIGNMENT=_GEMM_ALLOC_ALIGNMENT,
        )

        # Narrow to actual padded size (removes over-allocated tail).
        # ``actual_M = sum(ks)`` is already a multiple of ``padding_alignment``
        # (each group is rounded up to that alignment inside the prep kernel),
        # so this size is sufficient for BF16 grouped GEMM AND for DeepGEMM's
        # FP8 ``k_grouped_fp8_gemm_tn_contiguous`` which asserts the data row
        # count equals ``sum(ks)`` exactly.
        ks_cpu = get_pinned_buffer("ks", num_groups, torch.int32)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(copy_event)
            ks_cpu.copy_(ks_tensor, non_blocking=True)
        copy_stream.synchronize()
        ks = ks_cpu.tolist()
        actual_M = sum(ks)
        output_tokens = output_tokens[:actual_M]

        ctx.save_for_backward(reverse_shuffle_idxs)
        return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks

    @staticmethod
    def backward(
        ctx,
        grad_output_tokens,
        grad_reverse_shuffle_idxs,
        grad_grouped_mm_offs,
        grad_ks_tensor,
        grad_ks,
    ):
        (reverse_shuffle_idxs,) = ctx.saved_tensors
        # Forward: output_tokens[reverse_shuffle_idxs[i]] = input_tokens[i]
        # Backward: grad_input_tokens[i] = grad_output_tokens[reverse_shuffle_idxs[i]]
        grad_input_tokens = grad_output_tokens[reverse_shuffle_idxs]
        return grad_input_tokens, None, None, None


class _PaddedIndexGather(torch.autograd.Function):
    """Gather rows from a 2-D tensor by a 1-D index, with padded allocation.

    Two optimizations over plain ``input[index]``:

    1. **Avoids saving the input tensor** - the backward is a scatter-add that
       only needs the indices and the input shape, not the input values.
    2. **Pads the output allocation** to a fixed row alignment so the CUDA
       caching allocator sees consistent block sizes across micro-batches,
       reducing memory fragmentation from dynamic MoE token counts.
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, index: torch.Tensor, pad_to_multiple: int
    ) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.input_shape = input.shape
        ctx.pad_to_multiple = pad_to_multiple
        actual = index.shape[0]
        padded = (actual + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple
        buf = input.new_empty(padded, *input.shape[1:])
        # Write directly into buf without materializing a temporary.
        torch.index_select(input, 0, index, out=buf[:actual])
        return buf[:actual]  # view - keeps the padded allocation alive

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (index,) = ctx.saved_tensors
        rows = ctx.input_shape[0]
        padded_rows = (rows + ctx.pad_to_multiple - 1) // ctx.pad_to_multiple * ctx.pad_to_multiple
        grad_buf = torch.zeros(
            padded_rows,
            *ctx.input_shape[1:],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        grad_buf[:rows].scatter_add_(0, index.unsqueeze(-1).expand_as(grad_output), grad_output)
        return grad_buf[:rows], None, None


def padded_index_gather(
    input: torch.Tensor, index: torch.Tensor, pad_to_multiple: int = 1024
) -> torch.Tensor:
    """Memory-efficient ``input[index]`` with padded allocation.

    Does not save *input* for backward, and pads the output to
    *pad_to_multiple* rows to reduce CUDA allocator fragmentation.
    """
    return _PaddedIndexGather.apply(input, index, pad_to_multiple)


def scatter_for_grouped_gemm(
    input_tokens: torch.Tensor,  # [m, hidden_size]
    expert_idxs: torch.Tensor,  # [m] int64, values in [0, num_groups-1]
    num_groups: int,
    padding_alignment: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """
    Fused MoE token scatter: computes output tokens grouped by expert,
    reverse shuffle indices, grouped_mm offsets, and per-group padded sizes
    in a single kernel launch.

    Args:
        input_tokens: [m, hidden_size] input tokens
        expert_idxs: [m] expert assignment for each token (int64, values in [0, num_groups-1])
        num_groups: number of expert groups
        padding_alignment: pad each group's row count to this alignment (default 128)

    Returns:
        output_tokens: [actual_M, hidden_size] tokens reordered by expert with zero-padding,
            where actual_M = grouped_mm_offs[-1] (no over-allocation)
        reverse_shuffle_idxs: [m] int64, maps each input token to its position in output
        grouped_mm_offs: [num_groups] int32, cumulative padded offsets for grouped_mm
        ks: list[int], per-group padded sizes (derived from ks_tensor)
        ks_tensor: [num_groups] int32, per-group padded sizes
    """
    output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = (
        ScatterForGroupedGemm.apply(input_tokens, expert_idxs, num_groups, padding_alignment)
    )
    return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor


def precompute_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> Optional[torch.Tensor]:
    """
    Precompute per-row group indices for reuse across multiple grouped FP8 projections.

    Converts cumulative offsets to per-row group IDs, e.g.
    ``grouped_mm_offs = [128, 256, 384, 512]`` -> ``[0,0,...,1,1,...,2,2,...,3,3,...]``.

    Only needed on Hopper with the DeepGEMM backend; returns None otherwise.
    """
    if training.fp8 and ARCH_MAJOR < 10:
        row_indices = torch.arange(M, device=grouped_mm_offs.device)
        gi = torch.searchsorted(grouped_mm_offs, row_indices, right=True).to(torch.int32)
        gi.clamp_(max=grouped_mm_offs.shape[0] - 1)
        return gi
    return None
