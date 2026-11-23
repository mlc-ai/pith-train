"""
Compute the SM90 chunked gated delta rule forward and the operands shared with backward.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from flash_attn.cute.ampere_helpers import gemm, gemm_rs, get_smem_layout_atom
from quack.layout_utils import reshape_acc_to_frgA

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream

__all__ = [
    "CHUNK_ROWS", "HEAD_DIM", "STATE_COLS", "STATE_STAGES", "STATE_THREADS", "apply_gdr_wy_matrix", "chunk_gdr_fwd_pinned_sm90", "compute_gdr_state_output", "compute_gdr_wy", "copy_gdr_rows", "make_gdr_chunk_copies", "make_gdr_mma_copies",
    "scale_gdr_operands",
]

CHUNK_ROWS = 64
RCP_LN2 = 1.4426950216  # 1 / ln(2), rounded to FP32 for the log2 gate conversion
HEAD_DIM = 128
TF32_STRIDE = 68  # Pads the solve rows to avoid bank conflicts in TF32 A-fragment loads.
WY_THREADS = 128
STATE_THREADS = 128
STATE_STAGES = 3  # The recurrence prefetches two chunks ahead of the one it is computing.
STATE_COLS = 32
RECURRENCE_BARRIER, OUTPUT_BARRIER, STATE_FULL_BARRIER, STATE_EMPTY_BARRIER = 1, 2, 3, 5  # Each handoff reserves one ID per double-buffer slot.


@cute.jit
def make_gdr_input_tma(tensor, layout, width: cutlass.Constexpr[int], row_stride: cutlass.Constexpr[int]):
    head_columns = cute.make_tensor(tensor.iterator, cute.make_layout((tensor.shape[0], row_stride), stride=(row_stride, 1)))  # (token, head * width + column)
    return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), head_columns, layout, (CHUNK_ROWS, width))


@cute.jit
def accumulate_tf32_block_product(mma_thread: cute.ThrMma, mma_tf32: cute.TiledMma, acc_product: cute.Tensor, s_left: cute.Tensor, s_right: cute.Tensor, use_even_odd_k: cutlass.Constexpr[bool]):
    """
    Accumulate a 16 x 16 block product with one warp's TF32 MMA.
    """
    stride = TF32_STRIDE
    # Grouped K modes split the two k8 instructions into even and odd lanes for the inner merge.
    left_k_layout = cute.make_layout(((4, 2), 2), stride=((2, 8), 1)) if const_expr(use_even_odd_k) else cute.make_layout(16, stride=1)
    right_k_layout = cute.make_layout(((4, 2), 2), stride=((2 * stride, 8 * stride), stride)) if const_expr(use_even_odd_k) else cute.make_layout(16, stride=stride)
    s_left_k = cute.make_tensor(s_left.iterator, cute.make_layout((16, left_k_layout.shape), stride=(stride, left_k_layout.stride)))  # (block row, K)
    s_right_t = cute.make_tensor(s_right.iterator, cute.make_layout((16, right_k_layout.shape), stride=(1, right_k_layout.stride)))  # (block column, K)
    left_partition, right_partition = mma_thread.partition_A(s_left_k), mma_thread.partition_B(s_right_t)  # A: (MMA, MMA_M, MMA_K), B: (MMA, MMA_N, MMA_K)
    left_fragment = cute.make_rmem_tensor(mma_thread.partition_shape_A((16, 16)), Float32)  # (MMA, MMA_M, MMA_K)
    right_fragment = cute.make_rmem_tensor(mma_thread.partition_shape_B((16, 16)), Float32)  # (MMA, MMA_N, MMA_K)
    cute.autovec_copy(left_partition, left_fragment)
    cute.autovec_copy(right_partition, right_fragment)
    # Reinterpret FP32 registers so the MMA truncates the operands to TF32.
    reinterpret_tf32 = lambda fragment: cute.make_tensor(cute.recast_ptr(fragment.iterator, dtype=cutlass.TFloat32), fragment.layout)
    cute.gemm(mma_tf32, acc_product, reinterpret_tf32(left_fragment), reinterpret_tf32(right_fragment), acc_product)


def make_gdr_chunk_copies(threads: int, tidx) -> tuple:
    """
    Build the 16-byte chunk load and stores for 128- and 64-column BF16 tiles.
    """
    atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
    atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
    load_chunk = cute.make_tiled_copy_tv(atom_load, cute.make_ordered_layout((threads // 16, 16), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
    store_wide = cute.make_tiled_copy_tv(atom_store, cute.make_ordered_layout((threads // 16, 16), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
    store_square = cute.make_tiled_copy_tv(atom_store, cute.make_ordered_layout((threads // 8, 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
    return load_chunk, store_wide, store_square


def make_gdr_mma_copies(mma: cute.TiledMma, tidx) -> tuple:
    """
    Build matrix loads for A and B in both orientations, plus loads and stores for C.
    """
    plain_load_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)
    transposed_load_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16)
    load_a, load_b = cute.make_tiled_copy_A(plain_load_atom, mma).get_slice(tidx), cute.make_tiled_copy_B(plain_load_atom, mma).get_slice(tidx)
    load_a_t, load_b_t = cute.make_tiled_copy_A(transposed_load_atom, mma).get_slice(tidx), cute.make_tiled_copy_B(transposed_load_atom, mma).get_slice(tidx)
    store_c = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16), mma).get_slice(tidx)
    store_c_t = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16), mma).get_slice(tidx)
    load_c = cute.make_tiled_copy_C(plain_load_atom, mma).get_slice(tidx)
    return load_a, load_b, load_a_t, load_b_t, store_c, store_c_t, load_c


@cute.jit
def copy_gdr_rows(tiled_copy: cute.TiledCopy, source: cute.Tensor, destination: cute.Tensor, element_coords: cute.Tensor, num_valid_rows, use_zero_fill: cutlass.Constexpr[bool]):
    """
    Copy 16-byte vectors from valid rows and optionally zero padded rows in place.
    """
    # Skipped copies leave stale data. The caller's barrier publishes the shared zero fill.
    if num_valid_rows >= CHUNK_ROWS:
        cute.copy(tiled_copy, source, destination)
    else:
        for copy_row_idx in cutlass.range_constexpr(cute.size(source, mode=[1])):
            for copy_col_idx in cutlass.range_constexpr(cute.size(source, mode=[2])):
                if element_coords[0, copy_row_idx, copy_col_idx][0] < num_valid_rows:
                    cute.copy(tiled_copy, source[None, copy_row_idx, copy_col_idx], destination[None, copy_row_idx, copy_col_idx])
                else:
                    if const_expr(use_zero_fill):
                        destination[None, copy_row_idx, copy_col_idx].fill(0.0)


@cute.jit
def scale_gdr_operands(load_chunk: cute.TiledCopy, s_key: cute.Tensor, s_value: cute.Tensor, s_gate_cum: cute.Tensor, s_beta: cute.Tensor, use_key_rounding: cutlass.Constexpr[bool]):
    """
    Scale k and v in place by beta, and k by the base-two exponential of the cumulative gate.
    Key rounding selects whether k * beta rounds to BF16 before multiplying by the decay.
    """
    key_partition = load_chunk.partition_D(s_key)  # (copy vector, copy rows, copy columns)
    value_partition = load_chunk.partition_D(s_value)  # (copy vector, copy rows, copy columns)
    element_coords = load_chunk.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM)))  # (copy vector, copy rows, copy columns)
    for copy_row_idx in cutlass.range_constexpr(cute.size(key_partition, mode=[1])):
        for copy_col_idx in cutlass.range_constexpr(cute.size(key_partition, mode=[2])):
            row_idx = element_coords[0, copy_row_idx, copy_col_idx][0]
            beta, decay = s_beta[row_idx], cute.math.exp2(s_gate_cum[row_idx], fastmath=True)
            key_beta = key_partition[None, copy_row_idx, copy_col_idx].load().to(Float32) * beta
            if const_expr(use_key_rounding):
                key_beta = key_beta.to(BFloat16).to(Float32)
            key_partition[None, copy_row_idx, copy_col_idx].store((key_beta * decay).to(BFloat16))
            value_partition[None, copy_row_idx, copy_col_idx].store((value_partition[None, copy_row_idx, copy_col_idx].load().to(Float32) * beta).to(BFloat16))
    cute.arch.barrier()


@cute.jit
def apply_gdr_wy_matrix(mma: cute.TiledMma, mma_thread: cute.ThrMma, load_a: cute.TiledCopy, load_b_t: cute.TiledCopy, s_wy: cute.Tensor, s_operand: cute.Tensor):
    """
    Replace the shared operand with its BF16 product by A.
    """
    s_operand_t = cute.composition(s_operand, cute.make_ordered_layout((HEAD_DIM, CHUNK_ROWS), order=(1, 0)))  # (head column, chunk row)
    acc_product = cute.make_rmem_tensor(mma_thread.partition_shape_C((CHUNK_ROWS, HEAD_DIM)), Float32)
    acc_product.fill(0.0)
    gemm(mma, acc_product, mma_thread.make_fragment_A(mma_thread.partition_A(s_wy)), mma_thread.make_fragment_B(mma_thread.partition_B(s_operand_t)), load_a.partition_S(s_wy), load_b_t.partition_S(s_operand_t), load_a, load_b_t)
    cute.arch.barrier()
    operand_partition = mma_thread.partition_C(s_operand)  # (MMA, MMA_M, MMA_N)
    for elem_idx in cutlass.range_constexpr(cute.size(acc_product)):
        operand_partition[elem_idx] = BFloat16(acc_product[elem_idx])
    cute.arch.barrier()


@cute.jit
def compute_causal_scores(mma, mma_thread, load_a, load_b, s_query, s_key, s_gate_cum):
    acc_scores = cute.make_rmem_tensor(mma_thread.partition_shape_C((CHUNK_ROWS, CHUNK_ROWS)), Float32)
    acc_scores.fill(0.0)
    gemm(mma, acc_scores, mma_thread.make_fragment_A(mma_thread.partition_A(s_query)), mma_thread.make_fragment_B(mma_thread.partition_B(s_key)), load_a.partition_S(s_query), load_b.partition_S(s_key), load_a, load_b)
    element_coords = mma_thread.partition_C(cute.make_identity_tensor((CHUNK_ROWS, CHUNK_ROWS)))
    for elem_idx in cutlass.range_constexpr(cute.size(acc_scores)):
        row_idx, col_idx = element_coords[elem_idx]
        value = Float32(0.0)
        if row_idx >= col_idx:
            value = acc_scores[elem_idx] * cute.math.exp2(s_gate_cum[row_idx] - s_gate_cum[col_idx], fastmath=True)
        acc_scores[elem_idx] = value
    score_fragment = cute.make_fragment_like(acc_scores, BFloat16)
    score_fragment.store(acc_scores.load().to(BFloat16))
    return score_fragment


class WyRepresentation:
    """
    Solve A = (I + L)^-1 and form W and U with one CTA per chunk and value head.
    """
    def __init__(self, num_key_heads: int, num_value_heads: int, is_ragged: bool = False):
        self.num_key_heads, self.num_value_heads, self.is_ragged = num_key_heads, num_value_heads, is_ragged

    @cute.jit
    def __call__(self, mK: cute.Tensor, mV: cute.Tensor, mGate: cute.Tensor, mBeta: cute.Tensor, mGateCum: cute.Tensor, mWy: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, stream, mQ=None, mScores=None):
        mma_bf16 = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        mma_tf32 = cute.make_tiled_mma(warp.MmaTF32Op((16, 8, 8)), (1, 1, 1), permutation_mnk=(16, 16, 8))
        chunk_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (CHUNK_ROWS, HEAD_DIM), (0, 1))
        wy_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, CHUNK_ROWS), (CHUNK_ROWS, CHUNK_ROWS), (0, 1))
        num_chunks = (mGate.shape[0] + CHUNK_ROWS - 1) // CHUNK_ROWS if const_expr(self.is_ragged) else mGate.shape[0] // CHUNK_ROWS
        self.kernel(mK, mV, mGate, mBeta, mGateCum, mWy, mW, mU, mma_bf16, mma_tf32, chunk_layout, wy_layout, mQ, mScores).launch(grid=[num_chunks, self.num_value_heads, 1], block=[WY_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self, mK: cute.Tensor, mV: cute.Tensor, mGate: cute.Tensor, mBeta: cute.Tensor, mGateCum: cute.Tensor, mWy: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mma_bf16: cute.TiledMma, mma_tf32: cute.TiledMma,
        chunk_layout: cute.ComposedLayout, wy_layout: cute.ComposedLayout, mQ, mScores,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, _ = cute.arch.block_idx()
        key_head_idx = head_idx // (self.num_value_heads // self.num_key_heads)
        solve_layout = cute.make_layout((CHUNK_ROWS, CHUNK_ROWS), stride=(TF32_STRIDE, 1))
        allocator = cutlass.utils.SmemAllocator()
        s_inverse = allocator.allocate_tensor(Float32, solve_layout, 1024)  # holds k first
        s_value_storage = allocator.allocate_tensor(Float32, cute.make_layout(CHUNK_ROWS * TF32_STRIDE), 1024)
        s_lower = allocator.allocate_tensor(Float32, solve_layout, 1024)  # holds k again after the inverse
        s_gate_cum = allocator.allocate_tensor(Float32, cute.make_layout(CHUNK_ROWS), 16)
        s_beta = allocator.allocate_tensor(Float32, cute.make_layout(CHUNK_ROWS), 16)
        if const_expr(mScores is not None):
            s_wy = cute.make_tensor(cute.recast_ptr(s_inverse.iterator, dtype=BFloat16), wy_layout)
        else:
            s_wy = allocator.allocate_tensor(BFloat16, wy_layout, 1024)
        s_key = cute.make_tensor(cute.recast_ptr(s_inverse.iterator, dtype=BFloat16), chunk_layout)  # (chunk row, key column)
        s_key_reload = cute.make_tensor(cute.recast_ptr(s_lower.iterator, dtype=BFloat16), chunk_layout)  # (chunk row, key column)
        s_value = cute.make_tensor(cute.recast_ptr(s_value_storage.iterator, dtype=BFloat16), chunk_layout)  # (chunk row, value column)
        load_chunk, store_wide, store_square = make_gdr_chunk_copies(WY_THREADS, tidx)
        load_a, load_b, _, load_b_t, _, _, _ = make_gdr_mma_copies(mma_bf16, tidx)
        mma_thread = mma_bf16.get_slice(tidx)
        key_tile = cute.local_tile(mK[None, key_head_idx, None], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))
        value_tile = cute.local_tile(mV[None, head_idx, None], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))
        wy_tile = cute.local_tile(mWy[None, head_idx, None], (CHUNK_ROWS, CHUNK_ROWS), (chunk_idx, 0))

        tile_coords = (chunk_idx, head_idx, key_head_idx)
        copies = (load_chunk, store_wide, store_square)
        operands = (mma_bf16, mma_thread, load_a, load_b, load_b_t)
        tail = self.load_chunk_inputs(mGate, mQ, (key_tile, value_tile), (s_key, s_value), copies, tile_coords)
        self.scan_log_gate(mGate, mBeta, mGateCum, s_gate_cum, s_beta, tidx, chunk_idx * CHUNK_ROWS, head_idx)
        self.prepare_chunk_scores(mScores, value_tile, (s_key, s_value, s_lower, s_gate_cum, s_beta), copies, tile_coords, wy_layout, tidx)
        self.solve_chunk_inverse(mScores is not None, operands, (s_key, s_lower, s_inverse, s_wy, s_gate_cum, s_beta), mma_tf32, tidx)
        self.form_wy_operands((mW, mU), (key_tile, wy_tile), (s_key_reload, s_value, s_wy, s_gate_cum, s_beta), copies, operands, tile_coords, tail)

    @cute.jit
    def load_chunk_inputs(self, mGate, mQ, tiles, buffers, copies, tile_coords):
        key_tile, value_tile = tiles
        s_key, s_value = buffers
        load_chunk, _, store_square = copies
        chunk_idx, _, key_head_idx = tile_coords
        if const_expr(self.is_ragged):
            num_valid_rows = mGate.shape[0] - chunk_idx * CHUNK_ROWS
            wide_row_coords = load_chunk.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM)))
            square_row_coords = store_square.partition_D(cute.make_identity_tensor((CHUNK_ROWS, CHUNK_ROWS)))
            copy_gdr_rows(load_chunk, load_chunk.partition_S(key_tile), load_chunk.partition_D(s_key), wide_row_coords, num_valid_rows, True)
            cute.arch.cp_async_commit_group()
            copy_gdr_rows(load_chunk, load_chunk.partition_S(value_tile), load_chunk.partition_D(s_value), wide_row_coords, num_valid_rows, True)
            cute.arch.cp_async_commit_group()
        else:
            cute.copy(load_chunk, load_chunk.partition_S(key_tile), load_chunk.partition_D(s_key))
            cute.arch.cp_async_commit_group()
            if const_expr(mQ is not None):
                query_tile = cute.local_tile(mQ[None, key_head_idx, None], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))
                cute.copy(load_chunk, load_chunk.partition_S(query_tile), load_chunk.partition_D(s_value))
            else:
                cute.copy(load_chunk, load_chunk.partition_S(value_tile), load_chunk.partition_D(s_value))
            cute.arch.cp_async_commit_group()
        return (wide_row_coords, square_row_coords, num_valid_rows) if const_expr(self.is_ragged) else None

    @cute.jit
    def prepare_chunk_scores(self, mScores, value_tile, buffers, copies, tile_coords, wy_layout, tidx):
        s_key, s_value, s_lower, s_gate_cum, s_beta = buffers
        load_chunk, _, store_square = copies
        chunk_idx, head_idx, _ = tile_coords
        if const_expr(mScores is not None):
            # Q occupies the V region until the score MMA has consumed it.
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            s_scores = cute.make_tensor(cute.recast_ptr(s_lower.iterator, dtype=BFloat16), wy_layout)  # (query row, key row)
            self.compute_triangular_product(s_value, s_key, s_scores, s_gate_cum, s_beta, tidx, True)
            cute.arch.barrier()
            cute.copy(load_chunk, load_chunk.partition_S(value_tile), load_chunk.partition_D(s_value))
            cute.arch.cp_async_commit_group()
            score_tile = cute.local_tile(mScores[None, head_idx, None], (CHUNK_ROWS, CHUNK_ROWS), (chunk_idx, 0))
            cute.copy(store_square, store_square.partition_S(s_scores), store_square.partition_D(score_tile))
            # L reuses the score buffer.
            cute.arch.barrier()
        else:
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier()

    @cute.jit
    def solve_chunk_inverse(self, has_scores: cutlass.Constexpr[bool], operands, buffers, mma_tf32, tidx):
        mma_bf16, mma_thread, load_a, load_b, _ = operands
        s_key, s_lower, s_inverse, s_wy, s_gate_cum, s_beta = buffers
        if const_expr(has_scores):
            self.compute_triangular_product(s_key, s_key, s_lower, s_gate_cum, s_beta, tidx, False)
            cute.arch.barrier()
        else:
            self.compute_wy_lower(mma_bf16, mma_thread, load_a, load_b, s_key, s_lower, s_gate_cum, s_beta)
        self.invert_diagonal_blocks(s_lower, s_inverse, tidx)
        self.merge_blocks(mma_tf32, s_lower, s_inverse, tidx)
        self.convert_inverse(s_inverse, s_wy, tidx, has_scores)

    @cute.jit
    def form_wy_operands(self, outputs, tiles, buffers, copies, operands, tile_coords, tail):
        mW, mU = outputs
        key_tile, wy_tile = tiles
        s_key_reload, s_value, s_wy, s_gate_cum, s_beta = buffers
        load_chunk, store_wide, store_square = copies
        mma_bf16, mma_thread, load_a, _, load_b_t = operands
        chunk_idx, head_idx, _ = tile_coords
        if const_expr(self.is_ragged):
            wide_row_coords, square_row_coords, num_valid_rows = tail
        # Reload K into the space freed by the lower triangle.
        if const_expr(self.is_ragged):
            copy_gdr_rows(load_chunk, load_chunk.partition_S(key_tile), load_chunk.partition_D(s_key_reload), wide_row_coords, num_valid_rows, True)
        else:
            cute.copy(load_chunk, load_chunk.partition_S(key_tile), load_chunk.partition_D(s_key_reload))
        cute.arch.cp_async_commit_group()
        cute.arch.barrier()
        if const_expr(self.is_ragged):
            copy_gdr_rows(store_square, store_square.partition_S(s_wy), store_square.partition_D(wy_tile), square_row_coords, num_valid_rows, False)
        else:
            cute.copy(store_square, store_square.partition_S(s_wy), store_square.partition_D(wy_tile))
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        scale_gdr_operands(load_chunk, s_key_reload, s_value, s_gate_cum, s_beta, True)
        for operand_idx in cutlass.range_constexpr(2):
            s_operand = s_value if const_expr(operand_idx == 0) else s_key_reload
            mOut = mU if const_expr(operand_idx == 0) else mW
            apply_gdr_wy_matrix(mma_bf16, mma_thread, load_a, load_b_t, s_wy, s_operand)
            if const_expr(self.is_ragged):
                copy_gdr_rows(store_wide, store_wide.partition_S(s_operand), store_wide.partition_D(cute.local_tile(mOut[None, head_idx, None], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))), wide_row_coords, num_valid_rows, False)
            else:
                cute.copy(store_wide, store_wide.partition_S(s_operand), store_wide.partition_D(cute.local_tile(mOut[None, head_idx, None], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))))

    @cute.jit
    def scan_log_gate(self, mGate: cute.Tensor, mBeta: cute.Tensor, mGateCum: cute.Tensor, s_gate_cum: cute.Tensor, s_beta: cute.Tensor, tidx, row_start, head_idx):
        """
        Scan both halves of the log gate in warp 0 and convert the result to log2 units.
        """
        if tidx < 32:
            row_idx = row_start + tidx
            if const_expr(self.is_ragged):
                T = mGate.shape[0]
                gate_lo, gate_hi, beta_lo, beta_hi = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                if row_idx < T:
                    gate_lo, beta_lo = Float32(mGate[row_idx, head_idx]), Float32(mBeta[row_idx, head_idx])
                if row_idx + 32 < T:
                    gate_hi, beta_hi = Float32(mGate[row_idx + 32, head_idx]), Float32(mBeta[row_idx + 32, head_idx])
            else:
                gate_lo, gate_hi = Float32(mGate[row_idx, head_idx]), Float32(mGate[row_idx + 32, head_idx])
            for i in cutlass.range_constexpr(5):
                shuffled_lo = cute.arch.shuffle_sync_up(gate_lo, offset=1 << i, mask_and_clamp=0)
                shuffled_hi = cute.arch.shuffle_sync_up(gate_hi, offset=1 << i, mask_and_clamp=0)
                if tidx >= (1 << i):
                    gate_lo, gate_hi = gate_lo + shuffled_lo, gate_hi + shuffled_hi
            gate_hi = gate_hi + cute.arch.shuffle_sync(gate_lo, offset=31)
            gate_cum_lo, gate_cum_hi = gate_lo * RCP_LN2, gate_hi * RCP_LN2
            s_gate_cum[tidx], s_gate_cum[tidx + 32] = gate_cum_lo, gate_cum_hi
            if const_expr(self.is_ragged):
                if row_idx < T:
                    mGateCum[row_idx, head_idx] = gate_cum_lo
                if row_idx + 32 < T:
                    mGateCum[row_idx + 32, head_idx] = gate_cum_hi
                s_beta[tidx], s_beta[tidx + 32] = beta_lo, beta_hi
            else:
                mGateCum[row_idx, head_idx], mGateCum[row_idx + 32, head_idx] = gate_cum_lo, gate_cum_hi
                s_beta[tidx], s_beta[tidx + 32] = Float32(mBeta[row_idx, head_idx]), Float32(mBeta[row_idx + 32, head_idx])

    @cute.jit
    def compute_triangular_product(self, s_left, s_key, s_triangle, s_gate_cum, s_beta, tidx, use_score: cutlass.Constexpr[bool]):
        mma = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (1, 1, 1), permutation_mnk=(16, 16, 16))
        warp_idx, lane = tidx // 32, tidx % 32
        mma_thread = mma.get_slice(lane)
        load_a, load_b, _, _, _, _, _ = make_gdr_mma_copies(mma, lane)
        for wave in cutlass.range_constexpr(3):
            if const_expr(wave == 0):
                row_tile_idx, col_tile_idx = warp_idx, warp_idx
            elif const_expr(wave == 1):
                row_tile_idx, col_tile_idx = warp_idx + 1 - warp_idx // 3, warp_idx // 3
            else:
                row_tile_idx, col_tile_idx = warp_idx + 2, warp_idx + 1
            if wave < 2 or warp_idx < 2:
                s_left_tile = cute.local_tile(s_left, (16, HEAD_DIM), (row_tile_idx, 0))  # (16 query or key rows, head column)
                s_right_tile = cute.local_tile(s_key, (16, HEAD_DIM), (col_tile_idx, 0))  # (16 key rows, head column)
                acc_product = cute.make_rmem_tensor(mma_thread.partition_shape_C((16, 16)), Float32)
                acc_product.fill(0.0)
                gemm(mma, acc_product, mma_thread.make_fragment_A(mma_thread.partition_A(s_left_tile)), mma_thread.make_fragment_B(mma_thread.partition_B(s_right_tile)), load_a.partition_S(s_left_tile), load_b.partition_S(s_right_tile), load_a, load_b)
                element_coords = mma_thread.partition_C(cute.make_identity_tensor((16, 16)))
                for elem_idx in cutlass.range_constexpr(cute.size(acc_product)):
                    row_idx, col_idx = row_tile_idx * 16 + element_coords[elem_idx][0], col_tile_idx * 16 + element_coords[elem_idx][1]
                    value = Float32(0.0)
                    if const_expr(wave != 0) or row_idx >= col_idx + (0 if const_expr(use_score) else 1):
                        value = acc_product[elem_idx] * cute.math.exp2(s_gate_cum[row_idx] - s_gate_cum[col_idx], fastmath=True)
                        if const_expr(not use_score):
                            value = value * s_beta[row_idx]
                    # Keep the original swizzle origin for dynamically assigned tiles.
                    s_triangle[row_idx, col_idx] = value.to(s_triangle.element_type)
                    if const_expr(wave != 0):
                        s_triangle[col_idx, row_idx] = s_triangle.element_type(0.0)

    @cute.jit
    def compute_wy_lower(self, mma: cute.TiledMma, mma_thread: cute.ThrMma, load_a: cute.TiledCopy, load_b: cute.TiledCopy, s_key: cute.Tensor, s_lower: cute.Tensor, s_gate_cum: cute.Tensor, s_beta: cute.Tensor):
        """
        Form L, the strictly lower triangle of beta_i dot(k_i, k_j) exp2(g_i - g_j).
        """
        acc_scores = cute.make_rmem_tensor(mma_thread.partition_shape_C((CHUNK_ROWS, CHUNK_ROWS)), Float32)
        acc_scores.fill(0.0)
        gemm(mma, acc_scores, mma_thread.make_fragment_A(mma_thread.partition_A(s_key)), mma_thread.make_fragment_B(mma_thread.partition_B(s_key)), load_a.partition_S(s_key), load_b.partition_S(s_key), load_a, load_b)
        element_coords, lower_partition = mma_thread.partition_C(cute.make_identity_tensor((CHUNK_ROWS, CHUNK_ROWS))), mma_thread.partition_C(s_lower)  # (MMA, MMA_M, MMA_N)
        for elem_idx in cutlass.range_constexpr(cute.size(acc_scores)):
            row_idx, col_idx = element_coords[elem_idx]
            value = Float32(0.0)
            if row_idx > col_idx:
                value = acc_scores[elem_idx] * cute.math.exp2(s_gate_cum[row_idx] - s_gate_cum[col_idx], fastmath=True) * s_beta[row_idx]
            lower_partition[elem_idx] = value
        cute.arch.barrier()

    @cute.jit
    def invert_diagonal_blocks(self, s_lower: cute.Tensor, s_inverse: cute.Tensor, tidx):
        """
        Invert each 16 x 16 diagonal block of I + L by forward substitution in its owning warp.
        """
        warp_idx, lane = tidx // 32, tidx % 32
        block_start = warp_idx * 16
        if lane < 16:
            column = [-s_lower[block_start + row_idx, block_start + lane] for row_idx in range(16)]
            for row_idx in cutlass.range_constexpr(2, 16):
                lower_row = [-s_lower[block_start + row_idx, block_start + col_idx] if col_idx < row_idx else Float32(0.0) for col_idx in range(16)]
                # Pair rows separated by eight, then reduce by row-group bits 4, 2 and 1.
                pairs = [cute.math.fma(lower_row[pair_idx], column[pair_idx], lower_row[pair_idx + 8] * column[pair_idx + 8]) for pair_idx in range(8)]
                total = ((pairs[0] + pairs[4]) + (pairs[2] + pairs[6])) + ((pairs[1] + pairs[5]) + (pairs[3] + pairs[7]))
                lower_elem = Float32(0.0)
                if lane < row_idx:
                    lower_elem = -s_lower[block_start + row_idx, block_start + lane]
                column[row_idx] = lower_elem + total
            for row_idx in cutlass.range_constexpr(16):
                identity = Float32(0.0)
                if lane == row_idx:
                    identity = Float32(1.0)
                s_inverse[block_start + row_idx, block_start + lane] = column[row_idx] + identity
        cute.arch.barrier()

    @cute.jit
    def merge_blocks(self, mma_tf32: cute.TiledMma, s_lower: cute.Tensor, s_inverse: cute.Tensor, tidx):
        """
        Form the strictly lower inverse blocks from preceding block rows.
        """
        warp_idx, lane = tidx // 32, tidx % 32
        mma_thread = mma_tf32.get_slice(lane)
        select_solve_block = lambda tensor, row_idx, col_idx: cute.local_tile(tensor, (16, 16), (row_idx, col_idx))
        for block_row_idx in cutlass.range_constexpr(1, 4):
            for block_col_idx in cutlass.range_constexpr(block_row_idx):
                if warp_idx == block_col_idx:
                    acc_product = cute.make_rmem_tensor(mma_thread.partition_shape_C((16, 16)), Float32)
                    acc_product.fill(0.0)
                    # Adjacent blocks multiply from the left. Distant blocks fuse their sum.
                    if const_expr(block_col_idx == block_row_idx - 1):
                        accumulate_tf32_block_product(mma_thread, mma_tf32, acc_product, select_solve_block(s_inverse, block_row_idx, block_row_idx), select_solve_block(s_lower, block_row_idx, block_col_idx), True)
                    else:
                        for block_inner_idx in cutlass.range_constexpr(block_col_idx, block_row_idx):
                            accumulate_tf32_block_product(mma_thread, mma_tf32, acc_product, select_solve_block(s_lower, block_row_idx, block_inner_idx), select_solve_block(s_inverse, block_inner_idx, block_col_idx), True)
                    # Use the unused upper block of L as scratch for the inner product.
                    inner_partition = mma_thread.partition_C(select_solve_block(s_lower, block_col_idx, block_row_idx))  # (MMA, MMA_M, MMA_N)
                    for elem_idx in cutlass.range_constexpr(cute.size(acc_product)):
                        inner_partition[elem_idx] = acc_product[elem_idx]
                    cute.arch.sync_warp()
                    acc_product.fill(0.0)
                    if const_expr(block_col_idx == block_row_idx - 1):
                        accumulate_tf32_block_product(mma_thread, mma_tf32, acc_product, select_solve_block(s_lower, block_col_idx, block_row_idx), select_solve_block(s_inverse, block_col_idx, block_col_idx), False)
                    else:
                        accumulate_tf32_block_product(mma_thread, mma_tf32, acc_product, select_solve_block(s_inverse, block_row_idx, block_row_idx), select_solve_block(s_lower, block_col_idx, block_row_idx), False)
                    inverse_partition = mma_thread.partition_C(select_solve_block(s_inverse, block_row_idx, block_col_idx))  # (MMA, MMA_M, MMA_N)
                    for elem_idx in cutlass.range_constexpr(cute.size(acc_product)):
                        inverse_partition[elem_idx] = -acc_product[elem_idx]
            cute.arch.barrier()

    @cute.jit
    def convert_inverse(self, s_inverse: cute.Tensor, s_wy: cute.Tensor, tidx, use_in_place: cutlass.Constexpr[bool]):
        """
        Convert the solved lower block triangle to BF16 and zero the upper blocks.
        """
        num_elements = CHUNK_ROWS * CHUNK_ROWS // WY_THREADS
        num_aliased_elements = num_elements // 2 if const_expr(use_in_place) else 0
        if const_expr(use_in_place):
            packed_inverse = cute.make_rmem_tensor(cute.make_layout(num_aliased_elements), BFloat16)
            for thread_elem_idx in cutlass.range_constexpr(num_aliased_elements):
                elem_idx = thread_elem_idx * WY_THREADS + tidx
                row_idx, col_idx = elem_idx // CHUNK_ROWS, elem_idx % CHUNK_ROWS
                value = Float32(0.0)
                if row_idx // 16 >= col_idx // 16:
                    value = s_inverse[row_idx, col_idx]
                packed_inverse[thread_elem_idx] = BFloat16(value)
            # The entire BF16 tile ends before FP32 row 32. Only the first
            # half of the inverse must be read before the first BF16 store.
            cute.arch.barrier()
            for thread_elem_idx in cutlass.range_constexpr(num_aliased_elements):
                elem_idx = thread_elem_idx * WY_THREADS + tidx
                s_wy[elem_idx // CHUNK_ROWS, elem_idx % CHUNK_ROWS] = packed_inverse[thread_elem_idx]
        for thread_elem_idx in cutlass.range_constexpr(num_aliased_elements, num_elements):
            elem_idx = thread_elem_idx * WY_THREADS + tidx
            row_idx, col_idx = elem_idx // CHUNK_ROWS, elem_idx % CHUNK_ROWS
            value = Float32(0.0)
            if row_idx // 16 >= col_idx // 16:
                value = s_inverse[row_idx, col_idx]
            s_wy[row_idx, col_idx] = BFloat16(value)


class ChunkStateOutput:
    """
    Fuse the chunk-state recurrence and output for one value head and a tile of state columns.
    Ragged chunks read zero-padded inputs, repeat the last cumulative gate, and store valid rows.
    """
    def __init__(self, num_key_heads: int, num_value_heads: int, has_initial_state: bool, is_ragged: bool = False):
        self.num_key_heads, self.num_value_heads, self.has_initial_state, self.is_ragged = num_key_heads, num_value_heads, has_initial_state, is_ragged
        self.scale = HEAD_DIM ** -0.5
        self.recurrence_threads, self.output_threads = STATE_THREADS, STATE_THREADS
        self.output_warp_start = self.recurrence_threads // 32
        self.tma_warp = (self.recurrence_threads + self.output_threads) // 32

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGateCum: cute.Tensor, mOut: cute.Tensor, mInitialState: cute.Tensor, stream, mScores=None):
        mma = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        layouts = (
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (CHUNK_ROWS, HEAD_DIM, STATE_STAGES), (0, 1, 2)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (CHUNK_ROWS, HEAD_DIM, 2), (0, 1, 2)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_COLS), (CHUNK_ROWS, STATE_COLS, STATE_STAGES), (0, 1, 2)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (STATE_COLS, HEAD_DIM, 2), (0, 1, 2)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, CHUNK_ROWS), (STATE_COLS, CHUNK_ROWS, 2), (0, 1, 2)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_COLS), (CHUNK_ROWS, STATE_COLS), (0, 1)),
        )
        score_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, CHUNK_ROWS), (CHUNK_ROWS, CHUNK_ROWS, 2), (0, 1, 2)) if const_expr(mScores is not None) else None
        use_tma = const_expr(mScores is not None and not self.is_ragged and not self.has_initial_state and (self.num_key_heads, self.num_value_heads) in ((16, 48), (32, 96)))
        tma_descriptors = None
        if const_expr(use_tma):
            atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, BFloat16)
            narrow_layout_atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW64, BFloat16)
            key_ring_layout = cute.tile_to_shape(atom, (CHUNK_ROWS, HEAD_DIM, STATE_STAGES), (0, 1, 2))
            query_layout = cute.tile_to_shape(atom, (CHUNK_ROWS, HEAD_DIM, 2), (0, 1, 2))
            score_layout = cute.tile_to_shape(atom, (CHUNK_ROWS, CHUNK_ROWS, 2), (0, 1, 2))
            wy_value_layout = cute.tile_to_shape(narrow_layout_atom, (CHUNK_ROWS, STATE_COLS, STATE_STAGES), (0, 1, 2))
            tma_descriptors = (
                make_gdr_input_tma(mQ, query_layout, HEAD_DIM, self.num_key_heads * HEAD_DIM),
                make_gdr_input_tma(mScores, score_layout, CHUNK_ROWS, self.num_value_heads * CHUNK_ROWS),
                make_gdr_input_tma(mK, key_ring_layout, HEAD_DIM, self.num_key_heads * HEAD_DIM),
                make_gdr_input_tma(mW, key_ring_layout, HEAD_DIM, self.num_value_heads * HEAD_DIM),
                make_gdr_input_tma(mU, wy_value_layout, STATE_COLS, self.num_value_heads * HEAD_DIM),
            )
            layouts = (key_ring_layout, query_layout, wy_value_layout, *layouts[3:])
        threads = self.recurrence_threads + self.output_threads + (32 if const_expr(use_tma) else 0)
        self.kernel(mQ, mK, mW, mU, mGateCum, mOut, mInitialState, mma, *layouts, mScores, score_layout, tma_descriptors).launch(grid=[HEAD_DIM // STATE_COLS, self.num_value_heads, 1], block=[threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGateCum: cute.Tensor, mOut: cute.Tensor, mInitialState: cute.Tensor, mma: cute.TiledMma, key_ring_layout: cute.ComposedLayout,
        query_layout: cute.ComposedLayout, wy_value_layout: cute.ComposedLayout, state_layout: cute.ComposedLayout, residual_layout: cute.ComposedLayout, out_layout: cute.ComposedLayout, mScores, score_layout, tma_descriptors,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        col_tile_idx, head_idx, _ = cute.arch.block_idx()
        allocator = cutlass.utils.SmemAllocator()
        if const_expr(tma_descriptors is not None):
            s_key = allocator.allocate_tensor(BFloat16, key_ring_layout.outer, 1024, swizzle=key_ring_layout.inner)  # (chunk row, key column, ring stage)
            s_wy_key = allocator.allocate_tensor(BFloat16, key_ring_layout.outer, 1024, swizzle=key_ring_layout.inner)  # (chunk row, key column, ring stage)
            s_query = allocator.allocate_tensor(BFloat16, query_layout.outer, 1024, swizzle=query_layout.inner)  # (chunk row, key column, double-buffer slot)
            s_wy_value = allocator.allocate_tensor(BFloat16, wy_value_layout.outer, 1024, swizzle=wy_value_layout.inner)  # (chunk row, state column, ring stage)
        else:
            s_key = allocator.allocate_tensor(BFloat16, key_ring_layout, 1024)  # (chunk row, key column, ring stage)
            s_wy_key = allocator.allocate_tensor(BFloat16, key_ring_layout, 1024)  # (chunk row, key column, ring stage)
            s_query = allocator.allocate_tensor(BFloat16, query_layout, 1024)  # (chunk row, key column, double-buffer slot)
            s_wy_value = allocator.allocate_tensor(BFloat16, wy_value_layout, 1024)  # (chunk row, state column, ring stage)
        s_state = allocator.allocate_tensor(BFloat16, state_layout, 1024)  # (state column, key column, double-buffer slot)
        s_residual = allocator.allocate_tensor(BFloat16, residual_layout, 1024)  # (state column, chunk row, double-buffer slot)
        s_decayed_residual = allocator.allocate_tensor(BFloat16, residual_layout, 1024)  # (state column, chunk row, double-buffer slot)
        s_out = allocator.allocate_tensor(BFloat16, out_layout, 1024)  # (chunk row, state column)
        s_gate_cum = allocator.allocate_tensor(Float32, cute.make_layout((CHUNK_ROWS, STATE_STAGES), stride=(1, CHUNK_ROWS)), 16)  # (chunk row, ring stage)
        s_scores = None
        if const_expr(tma_descriptors is not None):
            s_scores = allocator.allocate_tensor(BFloat16, score_layout.outer, 1024, swizzle=score_layout.inner)  # (query row, key row, double-buffer slot)
        elif const_expr(mScores is not None):
            s_scores = allocator.allocate_tensor(BFloat16, score_layout, 1024)  # (query row, key row, double-buffer slot)
        recurrence_buffers = (s_key, s_wy_key, s_wy_value, s_state, s_residual, s_decayed_residual, s_gate_cum)
        output_buffers = (s_key, s_query, s_state, s_residual, s_out, s_gate_cum, s_scores)
        tile_coords = (col_tile_idx, head_idx)
        num_chunks = (mGateCum.shape[0] + CHUNK_ROWS - 1) // CHUNK_ROWS if const_expr(self.is_ragged) else mGateCum.shape[0] // CHUNK_ROWS
        recurrence_barriers = None
        if const_expr(tma_descriptors is not None):
            score_ready = allocator.allocate_array(cutlass.Int64, 2)
            score_released = allocator.allocate_array(cutlass.Int64, 2)
            state_ready = allocator.allocate_array(cutlass.Int64, STATE_STAGES)
            state_released = allocator.allocate_array(cutlass.Int64, STATE_STAGES)
            recurrence_barriers = (state_ready, state_released)
            if tidx == 0:
                for slot_idx in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(score_ready + slot_idx, 1)
                    cute.arch.mbarrier_init(score_released + slot_idx, self.output_threads)
                for slot_idx in cutlass.range_constexpr(STATE_STAGES):
                    cute.arch.mbarrier_init(state_ready + slot_idx, 1)
                    cute.arch.mbarrier_init(state_released + slot_idx, self.recurrence_threads)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

        if tidx < self.output_warp_start * 32:
            self.recurrence_loop((mK, mW, mU, mGateCum), mInitialState, mma, recurrence_buffers, tidx, tile_coords, recurrence_barriers)
        else:
            if const_expr(tma_descriptors is not None):
                if tidx < self.tma_warp * 32:
                    self.output_loop((mQ, mOut, mScores), mma, output_buffers, tidx - self.output_warp_start * 32, tile_coords, num_chunks, (score_ready, score_released))
                elif tidx == self.tma_warp * 32:
                    self.load_loop(tma_descriptors, (s_query, s_scores, s_key, s_wy_key, s_wy_value), (score_ready, score_released, state_ready, state_released), tile_coords, num_chunks)
            else:
                self.output_loop((mQ, mOut, mScores), mma, output_buffers, tidx - self.output_warp_start * 32, tile_coords, num_chunks)

    @cute.jit
    def load_loop(self, descriptors, input_buffers, barriers, tile_coords, num_chunks):
        col_tile_idx, head_idx = tile_coords
        (query_atom, query_tensor), (score_atom, score_tensor), (key_atom, key_tensor), (wy_key_atom, wy_key_tensor), (wy_value_atom, wy_value_tensor) = descriptors
        s_query, s_scores, s_key, s_wy_key, s_wy_value = input_buffers
        key_head_idx = head_idx // (self.num_value_heads // self.num_key_heads)
        query_tiles = cute.local_tile(query_tensor, (CHUNK_ROWS, HEAD_DIM), (None, key_head_idx))
        score_tiles = cute.local_tile(score_tensor, (CHUNK_ROWS, CHUNK_ROWS), (None, head_idx))
        key_tiles = cute.local_tile(key_tensor, (CHUNK_ROWS, HEAD_DIM), (None, key_head_idx))
        wy_key_tiles = cute.local_tile(wy_key_tensor, (CHUNK_ROWS, HEAD_DIM), (None, head_idx))
        wy_value_tiles = cute.local_tile(wy_value_tensor, (CHUNK_ROWS, STATE_COLS), (None, head_idx * (HEAD_DIM // STATE_COLS) + col_tile_idx))
        s_query_partition, query_partition = cpasync.tma_partition(query_atom, 0, cute.make_layout(1), cute.group_modes(s_query, 0, 2), cute.group_modes(query_tiles, 0, 2))  # shared: (TMA tile, buffer slot), global: (TMA tile, chunk)
        s_score_partition, score_partition = cpasync.tma_partition(score_atom, 0, cute.make_layout(1), cute.group_modes(s_scores, 0, 2), cute.group_modes(score_tiles, 0, 2))  # shared: (TMA tile, buffer slot), global: (TMA tile, chunk)
        s_key_partition, key_partition = cpasync.tma_partition(key_atom, 0, cute.make_layout(1), cute.group_modes(s_key, 0, 2), cute.group_modes(key_tiles, 0, 2))  # shared: (TMA tile, buffer slot), global: (TMA tile, chunk)
        s_wy_key_partition, wy_key_partition = cpasync.tma_partition(wy_key_atom, 0, cute.make_layout(1), cute.group_modes(s_wy_key, 0, 2), cute.group_modes(wy_key_tiles, 0, 2))  # shared: (TMA tile, buffer slot), global: (TMA tile, chunk)
        s_wy_value_partition, wy_value_partition = cpasync.tma_partition(wy_value_atom, 0, cute.make_layout(1), cute.group_modes(s_wy_value, 0, 2), cute.group_modes(wy_value_tiles, 0, 2))  # shared: (TMA tile, buffer slot), global: (TMA tile, chunk)
        cpasync.prefetch_descriptor(query_atom)
        cpasync.prefetch_descriptor(score_atom)
        cpasync.prefetch_descriptor(key_atom)
        cpasync.prefetch_descriptor(wy_key_atom)
        cpasync.prefetch_descriptor(wy_value_atom)
        atoms = (query_atom, score_atom, key_atom, wy_key_atom, wy_value_atom)
        tile_pairs = ((s_query_partition, query_partition), (s_score_partition, score_partition), (s_key_partition, key_partition), (s_wy_key_partition, wy_key_partition), (s_wy_value_partition, wy_value_partition))
        for chunk_idx in cutlass.range(num_chunks, unroll=1):
            self.load_step(chunk_idx, atoms, tile_pairs, barriers)

    @cute.jit
    def load_step(self, chunk_idx, atoms, tile_pairs, barriers):
        query_atom, score_atom, key_atom, wy_key_atom, wy_value_atom = atoms
        (s_query_partition, query_partition), (s_score_partition, score_partition), (s_key_partition, key_partition), (s_wy_key_partition, wy_key_partition), (s_wy_value_partition, wy_value_partition) = tile_pairs
        score_ready, score_released, state_ready, state_released = barriers
        state_slot, state_phase = chunk_idx % STATE_STAGES, (chunk_idx // STATE_STAGES) % 2
        if chunk_idx >= STATE_STAGES:
            cute.arch.mbarrier_wait(state_released + state_slot, state_phase ^ 1)
        state_ready_addr = state_ready + state_slot
        cute.arch.mbarrier_arrive_and_expect_tx(state_ready_addr, CHUNK_ROWS * (2 * HEAD_DIM + STATE_COLS) * 2)
        cute.copy(key_atom, key_partition[None, chunk_idx], s_key_partition[None, state_slot], tma_bar_ptr=state_ready_addr)
        cute.copy(wy_key_atom, wy_key_partition[None, chunk_idx], s_wy_key_partition[None, state_slot], tma_bar_ptr=state_ready_addr)
        cute.copy(wy_value_atom, wy_value_partition[None, chunk_idx], s_wy_value_partition[None, state_slot], tma_bar_ptr=state_ready_addr)
        slot_idx, barrier_phase = chunk_idx % 2, (chunk_idx // 2) % 2
        if chunk_idx >= 2:
            cute.arch.mbarrier_wait(score_released + slot_idx, barrier_phase ^ 1)
        ready_addr = score_ready + slot_idx
        cute.arch.mbarrier_arrive_and_expect_tx(ready_addr, CHUNK_ROWS * (HEAD_DIM + CHUNK_ROWS) * 2)
        cute.copy(query_atom, query_partition[None, chunk_idx], s_query_partition[None, slot_idx], tma_bar_ptr=ready_addr)
        cute.copy(score_atom, score_partition[None, chunk_idx], s_score_partition[None, slot_idx], tma_bar_ptr=ready_addr)

    @cute.jit
    def issue_chunk(self, copies, sources, targets, chunk_idx: int, stage_idx: int, tidx, tail, use_gate_only: cutlass.Constexpr[bool]):
        """
        Queue the chunk inputs in one asynchronous copy group.
        """
        load_chunk, load_wy_value, load_gate = copies
        key_source, wy_key_source, wy_value_source, gate_source = sources
        s_key_target, s_wy_key_target, s_wy_value_target, s_gate_target = targets
        if const_expr(use_gate_only):
            if tidx < CHUNK_ROWS:
                cute.copy(load_gate, gate_source[None, None, chunk_idx], s_gate_target[None, None, stage_idx])
        elif const_expr(self.is_ragged):
            wide_row_coords, narrow_row_coords, mGateCum, head_idx, s_gate_cum = tail
            T = mGateCum.shape[0]
            num_valid_rows = T - chunk_idx * CHUNK_ROWS
            copy_gdr_rows(load_chunk, key_source[None, None, None, chunk_idx], s_key_target[None, None, None, stage_idx], wide_row_coords, num_valid_rows, True)
            copy_gdr_rows(load_chunk, wy_key_source[None, None, None, chunk_idx], s_wy_key_target[None, None, None, stage_idx], wide_row_coords, num_valid_rows, True)
            copy_gdr_rows(load_wy_value, wy_value_source[None, None, None, chunk_idx], s_wy_value_target[None, None, None, stage_idx], narrow_row_coords, num_valid_rows, True)
            if tidx < CHUNK_ROWS:
                if tidx < num_valid_rows:
                    cute.copy(load_gate, gate_source[None, None, chunk_idx], s_gate_target[None, None, stage_idx])
                else:
                    s_gate_cum[tidx, stage_idx] = mGateCum[T - 1, head_idx]
        else:
            cute.copy(load_chunk, key_source[None, None, None, chunk_idx], s_key_target[None, None, None, stage_idx])
            cute.copy(load_chunk, wy_key_source[None, None, None, chunk_idx], s_wy_key_target[None, None, None, stage_idx])
            cute.copy(load_wy_value, wy_value_source[None, None, None, chunk_idx], s_wy_value_target[None, None, None, stage_idx])
            if tidx < CHUNK_ROWS:
                cute.copy(load_gate, gate_source[None, None, chunk_idx], s_gate_target[None, None, stage_idx])
        cute.arch.cp_async_commit_group()

    @cute.jit
    def recurrence_loop(self, inputs, mInitialState: cute.Tensor, mma: cute.TiledMma, buffers, tidx, tile_coords, tma_barriers):
        mK, mW, mU, mGateCum = inputs
        col_tile_idx, head_idx = tile_coords
        s_key, s_wy_key, s_wy_value, s_state, s_residual, s_decayed_residual, s_gate_cum = buffers
        num_chunks = (mGateCum.shape[0] + CHUNK_ROWS - 1) // CHUNK_ROWS if const_expr(self.is_ragged) else mGateCum.shape[0] // CHUNK_ROWS
        key_head_idx = head_idx // (self.num_value_heads // self.num_key_heads)
        threads = self.recurrence_threads
        load_chunk, _, _ = make_gdr_chunk_copies(threads, tidx)
        atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
        load_wy_value = cute.make_tiled_copy_tv(atom_load, cute.make_ordered_layout((threads // (STATE_COLS // 8), STATE_COLS // 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        load_gate = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32), cute.make_layout(CHUNK_ROWS), cute.make_layout(1)).get_slice(tidx % CHUNK_ROWS)
        key_tiles = cute.local_tile(mK[None, key_head_idx, None], (CHUNK_ROWS, HEAD_DIM), (None, 0))
        wy_key_tiles = cute.local_tile(mW[None, head_idx, None], (CHUNK_ROWS, HEAD_DIM), (None, 0))
        wy_value_tiles = cute.local_tile(mU[None, head_idx, None], (CHUNK_ROWS, STATE_COLS), (None, col_tile_idx))
        gate_tiles = cute.local_tile(mGateCum[None, head_idx], (CHUNK_ROWS,), (None,))
        sources = (load_chunk.partition_S(key_tiles), load_chunk.partition_S(wy_key_tiles), load_wy_value.partition_S(wy_value_tiles), load_gate.partition_S(gate_tiles))  # (copy vector, copy rows, copy columns, chunk), gate omits columns
        targets = (load_chunk.partition_D(s_key), load_chunk.partition_D(s_wy_key), load_wy_value.partition_D(s_wy_value), load_gate.partition_D(s_gate_cum))  # (copy vector, copy rows, copy columns, stage), gate omits columns
        tail = (load_chunk.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM))), load_wy_value.partition_D(cute.make_identity_tensor((CHUNK_ROWS, STATE_COLS))), mGateCum, head_idx, s_gate_cum) if const_expr(self.is_ragged) else None
        copies = (load_chunk, load_wy_value, load_gate)
        # Empty commits keep the wait-group distance fixed even for short sequences.
        for stage_idx in cutlass.range_constexpr(STATE_STAGES - 1):
            if stage_idx < num_chunks:
                self.issue_chunk(copies, sources, targets, stage_idx, stage_idx, tidx, tail, const_expr(tma_barriers is not None))
            else:
                cute.arch.cp_async_commit_group()
        mma_thread = mma.get_slice(tidx)
        load_a, load_b, load_a_t, _, _, store_c_t, load_c = make_gdr_mma_copies(mma, tidx)
        acc_state = cute.make_rmem_tensor(mma_thread.partition_shape_C((HEAD_DIM, STATE_COLS)), Float32)  # (MMA, MMA_K, MMA_V)
        acc_state.fill(0.0)
        state_coords = mma_thread.partition_C(cute.make_identity_tensor((HEAD_DIM, STATE_COLS)))  # (MMA, MMA_K, MMA_V)
        if const_expr(self.has_initial_state):
            for elem_idx in cutlass.range_constexpr(cute.size(acc_state)):
                acc_state[elem_idx] = mInitialState[head_idx, state_coords[elem_idx][0], col_tile_idx * STATE_COLS + state_coords[elem_idx][1]]
        residual_coords = mma_thread.partition_C(cute.make_identity_tensor((CHUNK_ROWS, STATE_COLS)))  # (MMA, MMA_T, MMA_V)
        matrix_ops = (mma, mma_thread, load_a, load_b, load_a_t, store_c_t, load_c)

        for chunk_idx in cutlass.range(num_chunks, unroll=1):
            stage_idx, double_buffer_idx = chunk_idx % STATE_STAGES, chunk_idx % 2
            self.recurrence_step(chunk_idx, (stage_idx, double_buffer_idx), buffers, matrix_ops, acc_state, residual_coords, tma_barriers)
            next_chunk_idx = chunk_idx + STATE_STAGES - 1
            if next_chunk_idx < num_chunks:
                self.issue_chunk(copies, sources, targets, next_chunk_idx, next_chunk_idx % STATE_STAGES, tidx, tail, const_expr(tma_barriers is not None))
            else:
                cute.arch.cp_async_commit_group()
            cute.arch.barrier(barrier_id=RECURRENCE_BARRIER, number_of_threads=threads)

    @cute.jit
    def recurrence_step(self, chunk_idx, stages, buffers, matrix_ops, acc_state, residual_coords, tma_barriers):
        stage_idx, double_buffer_idx = stages
        s_key, s_wy_key, s_wy_value, s_state, s_residual, s_decayed_residual, s_gate_cum = buffers
        mma, mma_thread, load_a, load_b, load_a_t, store_c_t, load_c = matrix_ops
        threads = self.recurrence_threads
        s_key_chunk, s_wy_key_chunk, s_wy_value_chunk = s_key[None, None, stage_idx], s_wy_key[None, None, stage_idx], s_wy_value[None, None, stage_idx]
        s_state_chunk = s_state[None, None, double_buffer_idx]  # (state column, key column)
        s_residual_chunk = s_residual[None, None, double_buffer_idx]  # (state column, chunk row)
        s_decayed_residual_chunk = s_decayed_residual[None, None, 0]  # (state column, chunk row)
        # The output warpgroup released this state slot at chunk - 2.
        state_fragment = cute.make_fragment_like(acc_state, BFloat16)
        state_fragment.store(acc_state.load().to(BFloat16))
        cute.copy(store_c_t, store_c_t.retile(state_fragment), store_c_t.partition_D(cute.composition(s_state_chunk, cute.make_ordered_layout((HEAD_DIM, STATE_COLS), order=(1, 0)))))
        cute.arch.cp_async_wait_group(STATE_STAGES - 2)
        if const_expr(tma_barriers is not None):
            ready_addr, released_addr = tma_barriers
            cute.arch.mbarrier_wait(ready_addr + stage_idx, (chunk_idx // STATE_STAGES) % 2)
        cute.arch.barrier(barrier_id=RECURRENCE_BARRIER, number_of_threads=threads)
        acc_residual = cute.make_rmem_tensor(mma_thread.partition_shape_C((CHUNK_ROWS, STATE_COLS)), Float32)
        acc_residual.fill(0.0)
        gemm(mma, acc_residual, mma_thread.make_fragment_A(mma_thread.partition_A(s_wy_key_chunk)), mma_thread.make_fragment_B(mma_thread.partition_B(s_state_chunk)), load_a.partition_S(s_wy_key_chunk), load_b.partition_S(s_state_chunk), load_a, load_b)
        last_gate = s_gate_cum[CHUNK_ROWS - 1, stage_idx]
        wy_value_fragment = cute.make_fragment_like(acc_residual, BFloat16)
        cute.copy(load_c, load_c.partition_S(s_wy_value_chunk), load_c.retile(wy_value_fragment))
        residual = wy_value_fragment.load().to(Float32) - acc_residual.load()
        residual_fragment = cute.make_fragment_like(acc_residual, BFloat16)
        residual_fragment.store(residual.to(BFloat16))
        acc_residual.store(residual)
        for elem_idx in cutlass.range_constexpr(cute.size(acc_residual)):
            acc_residual[elem_idx] = acc_residual[elem_idx] * cute.math.exp2(last_gate - s_gate_cum[residual_coords[elem_idx][0], stage_idx], fastmath=True)
        decayed_residual_fragment = cute.make_fragment_like(acc_residual, BFloat16)
        decayed_residual_fragment.store(acc_residual.load().to(BFloat16))
        cute.copy(store_c_t, store_c_t.retile(residual_fragment), store_c_t.partition_D(cute.composition(s_residual_chunk, cute.make_ordered_layout((CHUNK_ROWS, STATE_COLS), order=(1, 0)))))
        cute.copy(store_c_t, store_c_t.retile(decayed_residual_fragment), store_c_t.partition_D(cute.composition(s_decayed_residual_chunk, cute.make_ordered_layout((CHUNK_ROWS, STATE_COLS), order=(1, 0)))))
        decay = cute.math.exp2(last_gate, fastmath=True)
        for elem_idx in cutlass.range_constexpr(cute.size(acc_state)):
            acc_state[elem_idx] = acc_state[elem_idx] * decay
        cute.arch.barrier(barrier_id=RECURRENCE_BARRIER, number_of_threads=threads)
        cute.arch.barrier_arrive(barrier_id=STATE_FULL_BARRIER + double_buffer_idx, number_of_threads=threads + self.output_threads)
        s_key_chunk_t = cute.composition(s_key_chunk, cute.make_ordered_layout((HEAD_DIM, CHUNK_ROWS), order=(1, 0)))  # (key column, chunk row)
        gemm(
            mma, acc_state, mma_thread.make_fragment_A(mma_thread.partition_A(s_key_chunk_t)), mma_thread.make_fragment_B(mma_thread.partition_B(s_decayed_residual_chunk)), load_a_t.partition_S(s_key_chunk_t),
            load_b.partition_S(s_decayed_residual_chunk), load_a_t, load_b,
        )
        if const_expr(tma_barriers is not None):
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(released_addr + stage_idx)
        # The output warpgroup still reads the preceding chunk's K and state tiles.
        if chunk_idx > 0:
            cute.arch.barrier(barrier_id=STATE_EMPTY_BARRIER + 1 - double_buffer_idx, number_of_threads=threads + self.output_threads)

    @cute.jit
    def output_loop(self, tensors, mma: cute.TiledMma, buffers, tidx, tile_coords, num_chunks, score_barriers=None):
        mQ, mOut, mScores = tensors
        col_tile_idx, head_idx = tile_coords
        _, s_query, _, _, _, _, s_scores = buffers
        key_head_idx = head_idx // (self.num_value_heads // self.num_key_heads)
        threads = self.output_threads
        load_chunk, _, _ = make_gdr_chunk_copies(threads, tidx)
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        store_out = cute.make_tiled_copy_tv(atom_store, cute.make_ordered_layout((threads // (STATE_COLS // 8), STATE_COLS // 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        mma_thread = mma.get_slice(tidx)
        load_a, load_b, _, _, store_c, _, _ = make_gdr_mma_copies(mma, tidx)
        query_tiles = cute.local_tile(mQ[None, key_head_idx, None], (CHUNK_ROWS, HEAD_DIM), (None, 0))
        out_tiles = cute.local_tile(mOut[None, head_idx, None], (CHUNK_ROWS, STATE_COLS), (None, col_tile_idx))
        out_coords = mma_thread.partition_C(cute.make_identity_tensor((CHUNK_ROWS, STATE_COLS)))  # (MMA, MMA_T, MMA_V)
        if const_expr(score_barriers is None):
            query_source, s_query_target = load_chunk.partition_S(query_tiles), load_chunk.partition_D(s_query)  # global ends in chunk, shared ends in double-buffer slot
            if const_expr(mScores is not None):
                atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
                load_scores = cute.make_tiled_copy_tv(atom_load, cute.make_ordered_layout((threads // 8, 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
                score_tile = cute.local_tile(mScores[None, head_idx, None], (CHUNK_ROWS, CHUNK_ROWS), (None, 0))
                score_source, s_score_target = load_scores.partition_S(score_tile), load_scores.partition_D(s_scores)  # global ends in chunk, shared ends in double-buffer slot
                cute.copy(load_scores, score_source[None, None, None, 0], s_score_target[None, None, None, 0])
            if const_expr(self.is_ragged):
                T = mQ.shape[0]
                query_row_coords = load_chunk.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM)))
                out_row_coords = store_out.partition_D(cute.make_identity_tensor((CHUNK_ROWS, STATE_COLS)))
                copy_gdr_rows(load_chunk, query_source[None, None, None, 0], s_query_target[None, None, None, 0], query_row_coords, T, True)
            else:
                cute.copy(load_chunk, query_source[None, None, None, 0], s_query_target[None, None, None, 0])
            cute.arch.cp_async_commit_group()
        matrix_ops = (mma, mma_thread, load_a, load_b, store_c)
        store_info = (store_out, out_tiles, out_coords)
        query_partitions = (query_source, s_query_target) if const_expr(score_barriers is None) else None
        score_copy = (load_scores, score_source, s_score_target) if const_expr(score_barriers is None and mScores is not None) else None
        tail = (query_row_coords, out_row_coords, T) if const_expr(self.is_ragged) else None
        prefetch = (load_chunk, query_partitions, score_copy, tail)

        for chunk_idx in cutlass.range(num_chunks, unroll=1):
            stage_idx, double_buffer_idx = chunk_idx % STATE_STAGES, chunk_idx % 2
            self.output_step((chunk_idx, stage_idx, double_buffer_idx, num_chunks), buffers, matrix_ops, store_info, prefetch, mScores is not None, score_barriers)

    @cute.jit
    def output_step(self, chunk_coords, buffers, matrix_ops, store_info, prefetch, has_scores: cutlass.Constexpr[bool], score_barriers):
        chunk_idx, stage_idx, double_buffer_idx, num_chunks = chunk_coords
        s_key, s_query, s_state, s_residual, s_out, s_gate_cum, s_scores = buffers
        mma, mma_thread, load_a, load_b, store_c = matrix_ops
        store_out, out_tiles, out_coords = store_info
        load_chunk, query_partitions, score_copy, tail = prefetch
        threads = self.output_threads
        if const_expr(score_barriers is None):
            query_source, s_query_target = query_partitions
            if const_expr(has_scores):
                load_scores, score_source, s_score_target = score_copy
            if const_expr(self.is_ragged):
                query_row_coords, out_row_coords, T = tail
            if chunk_idx + 1 < num_chunks:
                if const_expr(self.is_ragged):
                    copy_gdr_rows(load_chunk, query_source[None, None, None, chunk_idx + 1], s_query_target[None, None, None, 1 - double_buffer_idx], query_row_coords, T - (chunk_idx + 1) * CHUNK_ROWS, True)
                else:
                    cute.copy(load_chunk, query_source[None, None, None, chunk_idx + 1], s_query_target[None, None, None, 1 - double_buffer_idx])
                    if const_expr(has_scores):
                        cute.copy(load_scores, score_source[None, None, None, chunk_idx + 1], s_score_target[None, None, None, 1 - double_buffer_idx])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
        else:
            score_ready, score_released = score_barriers
            cute.arch.mbarrier_wait(score_ready + double_buffer_idx, (chunk_idx // 2) % 2)
        cute.arch.barrier(barrier_id=STATE_FULL_BARRIER + double_buffer_idx, number_of_threads=self.recurrence_threads + threads)
        s_query_chunk = s_query[None, None, double_buffer_idx]
        s_key_chunk = s_key[None, None, stage_idx]
        s_state_chunk = s_state[None, None, double_buffer_idx]  # (state column, key column)
        s_residual_chunk = s_residual[None, None, double_buffer_idx]  # (state column, chunk row)
        acc_state_output = cute.make_rmem_tensor(mma_thread.partition_shape_C((CHUNK_ROWS, STATE_COLS)), Float32)
        acc_state_output.fill(0.0)
        gemm(mma, acc_state_output, mma_thread.make_fragment_A(mma_thread.partition_A(s_query_chunk)), mma_thread.make_fragment_B(mma_thread.partition_B(s_state_chunk)), load_a.partition_S(s_query_chunk), load_b.partition_S(s_state_chunk), load_a, load_b)
        acc_local_output = cute.make_rmem_tensor(mma_thread.partition_shape_C((CHUNK_ROWS, STATE_COLS)), Float32)
        acc_local_output.fill(0.0)
        if const_expr(has_scores):
            s_score_chunk = s_scores[None, None, double_buffer_idx]
            gemm(
                mma, acc_local_output, mma_thread.make_fragment_A(mma_thread.partition_A(s_score_chunk)), mma_thread.make_fragment_B(mma_thread.partition_B(s_residual_chunk)), load_a.partition_S(s_score_chunk),
                load_b.partition_S(s_residual_chunk), load_a, load_b,
            )
        else:
            score_fragment = compute_causal_scores(mma, mma_thread, load_a, load_b, s_query_chunk, s_key_chunk, s_gate_cum[None, stage_idx])
            gemm_rs(mma, acc_local_output, reshape_acc_to_frgA(score_fragment), mma_thread.make_fragment_B(mma_thread.partition_B(s_residual_chunk)), load_b.partition_S(s_residual_chunk), load_b)
        if const_expr(score_barriers is not None):
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(score_released + double_buffer_idx)
        for elem_idx in cutlass.range_constexpr(cute.size(acc_state_output)):
            acc_state_output[elem_idx] = acc_state_output[elem_idx] * cute.math.exp2(s_gate_cum[out_coords[elem_idx][0], stage_idx], fastmath=True)
        cute.arch.barrier_arrive(barrier_id=STATE_EMPTY_BARRIER + double_buffer_idx, number_of_threads=self.recurrence_threads + threads)
        out_fragment = cute.make_fragment_like(acc_state_output, BFloat16)
        out_fragment.store((acc_state_output.load() * self.scale + acc_local_output.load() * self.scale).to(BFloat16))
        cute.copy(store_c, store_c.retile(out_fragment), store_c.partition_D(s_out))
        cute.arch.barrier(barrier_id=OUTPUT_BARRIER, number_of_threads=threads)
        if const_expr(self.is_ragged):
            copy_gdr_rows(store_out, store_out.partition_S(s_out), store_out.partition_D(out_tiles[None, None, chunk_idx]), out_row_coords, T - chunk_idx * CHUNK_ROWS, False)
        else:
            cute.copy(store_out, store_out.partition_S(s_out), store_out.partition_D(out_tiles[None, None, chunk_idx]))
        cute.arch.barrier(barrier_id=OUTPUT_BARRIER, number_of_threads=threads)


COMPILED: dict[tuple, object] = {}


def compile_gdr_wy(num_key_heads: int, num_value_heads: int, is_ragged: bool = False, has_scores: bool = False) -> object:
    T = cute.sym_int()
    key_shape, value_shape, gate_shape = (T, num_key_heads, HEAD_DIM), (T, num_value_heads, HEAD_DIM), (T, num_value_heads)
    key_stride, value_stride, gate_stride = (num_key_heads * HEAD_DIM, HEAD_DIM, 1), (num_value_heads * HEAD_DIM, HEAD_DIM, 1), (num_value_heads, 1)
    keys = fake(BFloat16, key_shape, key_stride)
    values = fake(BFloat16, value_shape, value_stride)
    gate = fake(Float32, gate_shape, gate_stride, 4)
    beta = fake(BFloat16, gate_shape, gate_stride, 2)
    a_matrix = fake(BFloat16, (T, num_value_heads, CHUNK_ROWS), (num_value_heads * CHUNK_ROWS, CHUNK_ROWS, 1))
    score_args = (keys, a_matrix) if has_scores else ()
    return cute.compile(WyRepresentation(num_key_heads, num_value_heads, is_ragged), keys, values, gate, beta, gate, a_matrix, values, values, fake_stream(), *score_args, options="--enable-tvm-ffi")


def compile_gdr_state_output(num_key_heads: int, num_value_heads: int, has_initial_state: bool, is_ragged: bool = False, has_scores: bool = False) -> object:
    T = cute.sym_int()
    keys = fake(BFloat16, (T, num_key_heads, HEAD_DIM), (num_key_heads * HEAD_DIM, HEAD_DIM, 1))
    values = fake(BFloat16, (T, num_value_heads, HEAD_DIM), (num_value_heads * HEAD_DIM, HEAD_DIM, 1))
    gate = fake(Float32, (T, num_value_heads), (num_value_heads, 1), 4)
    state = fake(Float32, (num_value_heads, HEAD_DIM, HEAD_DIM), (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1))
    score_args = (fake(BFloat16, (T, num_value_heads, CHUNK_ROWS), (num_value_heads * CHUNK_ROWS, CHUNK_ROWS, 1)),) if has_scores else ()
    return cute.compile(ChunkStateOutput(num_key_heads, num_value_heads, has_initial_state, is_ragged), keys, keys, values, values, gate, values, state if has_initial_state else None, fake_stream(), *score_args, options="--enable-tvm-ffi")


def compute_gdr_wy(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, *, score_args: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return fresh chunk gates, solved WY matrices and W/U operands for batch size one.
    Inputs are contiguous: k is BF16 (1, T, H, 128), v is BF16 (1, T, HV, 128), and g and beta
    are respectively FP32 and BF16 (1, T, HV). Gates are FP32 (1, T, HV), A is BF16
    (1, T, HV, 64), and W/U match the value shape and dtype. Optional score arguments are
    the BF16 query shaped like k and a caller-provided BF16 score buffer shaped like A.
    The score buffer is filled in place on the aligned score path.
    """
    _, T, num_key_heads, _ = k.shape
    num_value_heads = v.shape[2]
    is_ragged = T % CHUNK_ROWS != 0
    gate_cum = torch.empty(1, T, num_value_heads, device=k.device, dtype=torch.float32)
    A = torch.empty(1, T, num_value_heads, CHUNK_ROWS, device=k.device, dtype=k.dtype)
    w, u = k.new_empty(1, T, num_value_heads, HEAD_DIM), torch.empty_like(v)
    has_scores = score_args is not None
    key = ("wy_representation", num_key_heads, num_value_heads, k.device.index, is_ragged, has_scores)
    if key not in COMPILED:
        COMPILED[key] = compile_gdr_wy(num_key_heads, num_value_heads, is_ragged, has_scores)
    score_views = (score_args[0][0], score_args[1][0]) if has_scores else ()
    COMPILED[key](k[0], v[0], g[0], beta[0], gate_cum[0], A[0], w[0], u[0], *score_views)
    return gate_cum, A, w, u


def compute_gdr_state_output(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, gate_cum: torch.Tensor, initial_state: torch.Tensor | None = None, *, scores: torch.Tensor | None = None) -> torch.Tensor:
    """
    Return a fresh BF16 output shaped like U from the given FP32 initial state or zero.
    Queries and keys are contiguous BF16 (1, T, H, 128). W and U are contiguous BF16
    (1, T, HV, 128), the cumulative gate is FP32 (1, T, HV), and an initial state,
    if given, is contiguous FP32 (HV, 128, 128). Optional BF16 scores are (1, T, HV, 64).
    """
    _, T, num_key_heads, _ = k.shape
    num_value_heads = u.shape[2]
    is_ragged = T % CHUNK_ROWS != 0
    out = torch.empty_like(u)
    has_initial_state = initial_state is not None
    has_scores = scores is not None
    key = ("chunk_state_output", num_key_heads, num_value_heads, k.device.index, has_initial_state, is_ragged, has_scores)
    if key not in COMPILED:
        COMPILED[key] = compile_gdr_state_output(num_key_heads, num_value_heads, has_initial_state, is_ragged, has_scores)
    score_views = (scores[0],) if has_scores else ()
    COMPILED[key](q[0], k[0], w[0], u[0], gate_cum[0], out[0], initial_state, *score_views)
    return out


def chunk_gdr_fwd_pinned_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute chunked gated delta rule forward with SM90 CuTe and keep the WY data for backward.
    """
    _, T, num_key_heads, _ = k.shape
    num_value_heads = v.shape[2]
    if T == 4096 and (num_key_heads, num_value_heads) in ((16, 48), (32, 96)):
        scores = q.new_empty((1, T, num_value_heads, CHUNK_ROWS))
        gate_cum, A, w, u = compute_gdr_wy(k, v, g, beta, score_args=(q, scores))
        return compute_gdr_state_output(q, k, w, u, gate_cum, scores=scores), gate_cum, A
    gate_cum, A, w, u = compute_gdr_wy(k, v, g, beta)
    return compute_gdr_state_output(q, k, w, u, gate_cum), gate_cum, A
