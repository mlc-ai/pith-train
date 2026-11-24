"""
SM90 gated delta rule backward with 64-row chunks and head dimensions of 128.
Ragged chunks zero-pad data rows and extend the last cumulative gate through the padding.
Only valid rows are stored. Products accumulate in fp32, with TF32 operands for the
state-gradient query/output product and bf16 operands for the remaining products.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.utils import LayoutEnum
from flash_attn.cute.ampere_helpers import gemm, gemm_rs, get_smem_layout_atom
from quack import sm90_utils
from quack.layout_utils import reshape_acc_to_frgA

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.gated_delta_rule_pinned_fwd_sm90 import CHUNK_ROWS, HEAD_DIM, STATE_COLS, STATE_STAGES, STATE_THREADS, make_gdr_chunk_copies, copy_gdr_rows, apply_gdr_wy_matrix, make_gdr_mma_copies, scale_gdr_operands

__all__ = ["bwd_prologue", "chunk_gdr_bwd_pinned_sm90", "chunk_gradients", "state_final", "state_gradient", "state_gradient_initial", "state_recompute"]

PROLOGUE_THREADS = 128
GRAD_BV = 32  # state-gradient columns per CTA
F32_PAD = 4  # fp32 row padding of the TF32 product tiles
GRAD_STAGES = 3  # chunks of k, w, q, g and dv_local / do columns in flight
BAR_CHAIN, BAR_PRODUCER = 1, 2
BAR_READY, BAR_FREE_P, BAR_FREE = 3, 5, 7  # READY, FREE_P: ids +0 / +1 by P buffer; FREE: +0 .. +2 by stage
GRAD_THREADS = 256  # ChunkGradients: 8 warps, 4 along the rows and 2 along the columns of each tile


def make_head_tma_atom(operation, tensor, layout, width):
    reordered = cute.make_tensor(tensor.iterator, cute.select(tensor.layout, mode=[0, 2, 1]))  # (token, feature, head)
    return cpasync.make_tiled_tma_atom(operation, reordered, layout, (CHUNK_ROWS, width))


@cute.jit
def load_tma_tile(atom: cute.CopyAtom, source: cute.Tensor, target: cute.Tensor, barrier):
    target_partition, source_partition = cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(target, 0, 2), cute.group_modes(source, 0, 2))
    cute.copy(atom, source_partition, target_partition, tma_bar_ptr=barrier)


@cute.jit
def store_tma_tile(atom: cute.CopyAtom, source: cute.Tensor, target: cute.Tensor):
    source_partition, target_partition = cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(source, 0, 2), cute.group_modes(target, 0, 2))
    cute.copy(atom, source_partition, target_partition)


class BwdPrologue:
    """
    Recompute the WY operands w and u from saved A and compute the local value gradient.
    """

    def __init__(self, H: int, HV: int, ragged: bool = False, tma: bool = False):
        self.H, self.HV, self.ragged = H, HV, ragged
        self.tma = tma
        self.scale = HEAD_DIM ** -0.5

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mDo: cute.Tensor, mGcum: cute.Tensor, mBeta: cute.Tensor, mA: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mDvLocal: cute.Tensor, stream):
        mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        if const_expr(self.tma):
            chunk_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, HEAD_DIM), major_mode_size=64)
            a_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, CHUNK_ROWS), major_mode_size=32)
        else:
            chunk_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (CHUNK_ROWS, HEAD_DIM), (0, 1))
            a_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, CHUNK_ROWS), (CHUNK_ROWS, CHUNK_ROWS), (0, 1))
        num_chunks = (mGcum.shape[0] + CHUNK_ROWS - 1) // CHUNK_ROWS if const_expr(self.ragged) else mGcum.shape[0] // CHUNK_ROWS
        tma_inputs, tma_outputs = None, None
        if const_expr(self.tma):
            tma_inputs = tuple(
                make_head_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, layout, width) for tensor, layout,
                width in ((mK, chunk_layout, HEAD_DIM), (mQ, chunk_layout, HEAD_DIM), (mDo, chunk_layout, HEAD_DIM), (mV, chunk_layout, HEAD_DIM), (mA, a_layout, CHUNK_ROWS),)
            )
            tma_outputs = tuple(make_head_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), tensor, chunk_layout, HEAD_DIM) for tensor in (mDvLocal, mU, mW))
        self.kernel(mQ, mK, mV, mDo, mGcum, mBeta, mA, mW, mU, mDvLocal, mma, chunk_layout, a_layout, tma_inputs, tma_outputs).launch(grid=[num_chunks, self.HV, 1], block=[PROLOGUE_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mDo: cute.Tensor, mGcum: cute.Tensor, mBeta: cute.Tensor, mA: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor,
        mDvLocal: cute.Tensor, mma: cute.TiledMma, chunk_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout, tma_inputs, tma_outputs
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, _ = cute.arch.block_idx()
        key_head_idx = head_idx // (self.HV // self.H)
        smem = cutlass.utils.SmemAllocator()
        if const_expr(self.tma):
            s_k = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
            s_q = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
            s_v = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
            s_do = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
            s_a = smem.allocate_tensor(BFloat16, a_layout.outer, 1024, swizzle=a_layout.inner)
        else:
            s_k, s_q = smem.allocate_tensor(BFloat16, chunk_layout, 1024), smem.allocate_tensor(BFloat16, chunk_layout, 1024)
            s_v, s_do = smem.allocate_tensor(BFloat16, chunk_layout, 1024), smem.allocate_tensor(BFloat16, chunk_layout, 1024)
            s_a = smem.allocate_tensor(BFloat16, a_layout, 1024)
        s_gcum, s_beta = smem.allocate_tensor(Float32, cute.make_layout(CHUNK_ROWS), 16), smem.allocate_tensor(Float32, cute.make_layout(CHUNK_ROWS), 16)
        load, store_wide, _ = make_gdr_chunk_copies(PROLOGUE_THREADS, tidx)
        load_square = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128),
            cute.make_ordered_layout((PROLOGUE_THREADS // 8, 8), order=(1, 0)), cute.make_layout((1, 8))
        ).get_slice(tidx)
        load_a, load_b, _, load_b_t, store_c, _, _ = make_gdr_mma_copies(mma, tidx)
        thread_mma = mma.get_slice(tidx)

        coords = (tidx, chunk_idx, head_idx, key_head_idx)
        inputs = (mK, mQ, mV, mDo, mA, mGcum, mBeta)
        tiles = (s_k, s_q, s_v, s_do, s_a, s_gcum, s_beta)
        ready = smem.allocate_array(cutlass.Int64, 2) if const_expr(self.tma) else None
        tail = self.load_inputs(coords, inputs, tiles, (load, load_square), tma_inputs, ready)
        self.compute_local_value_gradient(tiles, (mma, thread_mma, load_a, load_b, load_b_t, store_c), ready)
        self.store_local_value_gradient(coords, s_q, mDvLocal, store_wide, tma_outputs, tail)
        self.recompute_wy_operands(coords, (s_k, s_v, s_a, s_gcum, s_beta), (mU, mW), (mma, thread_mma, load, load_a, load_b_t, store_wide), tma_outputs, tail)

    @cute.jit
    def load_inputs(self, coords, inputs, tiles, copies, tma_inputs, ready):
        tidx, chunk_idx, head_idx, key_head_idx = coords
        mK, mQ, mV, mDo, mA, mGcum, mBeta = inputs
        s_k, s_q, s_v, s_do, s_a, s_gcum, s_beta = tiles
        load, load_square = copies
        wide_rows, num_valid_rows = None, None
        if const_expr(self.tma):
            if tidx == 0:
                cute.arch.mbarrier_init(ready, 1)
                cute.arch.mbarrier_init(ready + 1, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive_and_expect_tx(ready, 4 * CHUNK_ROWS * HEAD_DIM)
                cute.arch.mbarrier_arrive_and_expect_tx(ready + 1, 2 * CHUNK_ROWS * (2 * HEAD_DIM + CHUNK_ROWS))
            if tidx < 32:
                # Bulk copies elect their issuer from this warp.
                cute.arch.sync_warp()
                for i in cutlass.range_constexpr(5):
                    atom, tma_coords = tma_inputs[i]
                    head = key_head_idx if const_expr(i < 2) else head_idx
                    width = CHUNK_ROWS if const_expr(i == 4) else HEAD_DIM
                    source = cute.local_tile(tma_coords[None, None, head], (CHUNK_ROWS, width), (chunk_idx, 0))
                    load_tma_tile(atom, source, (s_k, s_q, s_do, s_v, s_a)[i], ready + (0 if const_expr(i < 2) else 1))
            if tidx < CHUNK_ROWS:
                row = chunk_idx * CHUNK_ROWS + tidx
                s_gcum[tidx], s_beta[tidx] = mGcum[row, head_idx], Float32(mBeta[row, head_idx])
        elif const_expr(self.ragged):
            T = mGcum.shape[0]
            num_valid_rows = T - chunk_idx * CHUNK_ROWS
            wide_rows = load.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM)))
            square_rows = load_square.partition_D(cute.make_identity_tensor((CHUNK_ROWS, CHUNK_ROWS)))
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mK, key_head_idx, chunk_idx)), load.partition_D(s_k), wide_rows, num_valid_rows, True)
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mQ, key_head_idx, chunk_idx)), load.partition_D(s_q), wide_rows, num_valid_rows, True)
            cute.arch.cp_async_commit_group()
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mDo, head_idx, chunk_idx)), load.partition_D(s_do), wide_rows, num_valid_rows, True)
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mV, head_idx, chunk_idx)), load.partition_D(s_v), wide_rows, num_valid_rows, True)
            copy_gdr_rows(load_square, load_square.partition_S(cute.local_tile(mA[None, head_idx, None], (CHUNK_ROWS, CHUNK_ROWS), (chunk_idx, 0))), load_square.partition_D(s_a), square_rows, num_valid_rows, True)
            cute.arch.cp_async_commit_group()
            if tidx < CHUNK_ROWS:
                # both loads at the clamped row, issued together; beta past T is then
                # zeroed in registers
                row = chunk_idx * CHUNK_ROWS + tidx
                clamped_row = cutlass.min(row, T - 1)
                gate, beta = mGcum[clamped_row, head_idx], Float32(mBeta[clamped_row, head_idx])
                if row >= T:
                    beta = Float32(0.0)
                s_gcum[tidx], s_beta[tidx] = gate, beta
        else:
            cute.copy(load, load.partition_S(select_chunk_tile(mK, key_head_idx, chunk_idx)), load.partition_D(s_k))
            cute.copy(load, load.partition_S(select_chunk_tile(mQ, key_head_idx, chunk_idx)), load.partition_D(s_q))
            cute.arch.cp_async_commit_group()
            cute.copy(load, load.partition_S(select_chunk_tile(mDo, head_idx, chunk_idx)), load.partition_D(s_do))
            cute.copy(load, load.partition_S(select_chunk_tile(mV, head_idx, chunk_idx)), load.partition_D(s_v))
            cute.copy(load_square, load_square.partition_S(cute.local_tile(mA[None, head_idx, None], (CHUNK_ROWS, CHUNK_ROWS), (chunk_idx, 0))), load_square.partition_D(s_a))
            cute.arch.cp_async_commit_group()
            if tidx < CHUNK_ROWS:
                row = chunk_idx * CHUNK_ROWS + tidx
                s_gcum[tidx], s_beta[tidx] = mGcum[row, head_idx], Float32(mBeta[row, head_idx])
        return wide_rows, num_valid_rows

    @cute.jit
    def compute_local_value_gradient(self, tiles, mma_copies, ready):
        s_k, s_q, _, s_do, _, s_gcum, _ = tiles
        mma, thread_mma, load_a, load_b, load_b_t, store_c = mma_copies
        if const_expr(self.tma):
            cute.arch.barrier()
            cute.arch.mbarrier_wait(ready, 0)
        else:
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier()
        # S = k q^T, then the upper-triangular (key i <= query j) decayed and scaled block
        acc_s = cute.make_rmem_tensor(thread_mma.partition_shape_C((CHUNK_ROWS, CHUNK_ROWS)), Float32)
        acc_s.fill(0.0)
        gemm(mma, acc_s, thread_mma.make_fragment_A(thread_mma.partition_A(s_k)), thread_mma.make_fragment_B(thread_mma.partition_B(s_q)), load_a.partition_S(s_k), load_b.partition_S(s_q), load_a, load_b)
        s_coords = thread_mma.partition_C(cute.make_identity_tensor((CHUNK_ROWS, CHUNK_ROWS)))
        for elem in cutlass.range_constexpr(cute.size(acc_s)):
            row, column_idx = s_coords[elem]
            value = Float32(0.0)
            if row <= column_idx:
                value = acc_s[elem] * self.scale * cute.math.exp2(s_gcum[column_idx] - s_gcum[row], fastmath=True)
            acc_s[elem] = value
        r_decay_scores = cute.make_fragment_like(acc_s, BFloat16)
        r_decay_scores.store(acc_s.load().to(BFloat16))
        if const_expr(self.tma):
            cute.arch.mbarrier_wait(ready + 1, 0)
            cute.arch.barrier()
        else:
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
        s_do_t = cute.composition(s_do, cute.make_ordered_layout((HEAD_DIM, CHUNK_ROWS), order=(1, 0)))  # (value, token)
        acc_dv = cute.make_rmem_tensor(thread_mma.partition_shape_C((CHUNK_ROWS, HEAD_DIM)), Float32)
        acc_dv.fill(0.0)
        gemm_rs(mma, acc_dv, reshape_acc_to_frgA(r_decay_scores), thread_mma.make_fragment_B(thread_mma.partition_B(s_do_t)), load_b_t.partition_S(s_do_t), load_b_t)
        r_dv = cute.make_fragment_like(acc_dv, BFloat16)
        r_dv.store(acc_dv.load().to(BFloat16))
        cute.copy(store_c, store_c.retile(r_dv), store_c.partition_D(s_q))  # q is no longer read
        if const_expr(self.tma):
            cute.arch.fence_view_async_shared()
        cute.arch.barrier()

    @cute.jit
    def store_local_value_gradient(self, coords, s_q, mDvLocal, store_wide, tma_outputs, tail):
        tidx, chunk_idx, head_idx, _ = coords
        wide_rows, num_valid_rows = tail
        if const_expr(self.tma):
            if tidx < 32:
                atom, tma_coords = tma_outputs[0]
                target = cute.local_tile(tma_coords[None, None, head_idx], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))
                store_tma_tile(atom, s_q, target)
                cute.arch.cp_async_bulk_commit_group()
        elif const_expr(self.ragged):
            copy_gdr_rows(store_wide, store_wide.partition_S(s_q), store_wide.partition_D(select_chunk_tile(mDvLocal, head_idx, chunk_idx)), wide_rows, num_valid_rows, False)
        else:
            cute.copy(store_wide, store_wide.partition_S(s_q), store_wide.partition_D(select_chunk_tile(mDvLocal, head_idx, chunk_idx)))

    @cute.jit
    def recompute_wy_operands(self, coords, tiles, outputs, copies, tma_outputs, tail):
        tidx, chunk_idx, head_idx, _ = coords
        s_k, s_v, s_a, s_gcum, s_beta = tiles
        mU, mW = outputs
        mma, thread_mma, load, load_a, load_b_t, store_wide = copies
        wide_rows, num_valid_rows = tail
        # Round k beta to bf16 before applying the gate, matching the forward WY operands.
        scale_gdr_operands(load, s_k, s_v, s_gcum, s_beta, True)
        for operand_idx in cutlass.range_constexpr(2):
            s_x = s_v if const_expr(operand_idx == 0) else s_k
            mOut = mU if const_expr(operand_idx == 0) else mW
            apply_gdr_wy_matrix(mma, thread_mma, load_a, load_b_t, s_a, s_x)
            if const_expr(self.tma):
                cute.arch.fence_view_async_shared()
                cute.arch.barrier()
                if tidx < 32:
                    atom, tma_coords = tma_outputs[1 + operand_idx]
                    target = cute.local_tile(tma_coords[None, None, head_idx], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))
                    store_tma_tile(atom, s_x, target)
            elif const_expr(self.ragged):
                copy_gdr_rows(store_wide, store_wide.partition_S(s_x), store_wide.partition_D(select_chunk_tile(mOut, head_idx, chunk_idx)), wide_rows, num_valid_rows, False)
            else:
                cute.copy(store_wide, store_wide.partition_S(s_x), store_wide.partition_D(select_chunk_tile(mOut, head_idx, chunk_idx)))
        if const_expr(self.tma):
            if tidx < 32:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)


class StateRecompute:
    """
    Recompute chunk-entry states and v_new, optionally from an initial state.
    An optional final-state output holds fp32 state in (value head, key, value) order.
    """

    def __init__(self, H: int, HV: int, ragged: bool = False, has_h0: bool = False, has_ht: bool = False, store_states: bool = True):
        self.H, self.HV, self.ragged = H, HV, ragged
        self.has_h0, self.has_ht, self.store_states = has_h0, has_ht, store_states

    @cute.jit
    def __call__(self, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGcum: cute.Tensor, mH: cute.Tensor, mVnew: cute.Tensor, mH0: cute.Tensor, mHt: cute.Tensor, stream):
        mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        layouts = (
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (CHUNK_ROWS, HEAD_DIM, STATE_STAGES), (0, 1, 2)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_COLS), (CHUNK_ROWS, STATE_COLS, STATE_STAGES), (0, 1, 2)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (STATE_COLS, HEAD_DIM), (0, 1)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, CHUNK_ROWS), (STATE_COLS, CHUNK_ROWS), (0, 1)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_COLS), (CHUNK_ROWS, STATE_COLS), (0, 1))
        )
        self.kernel(mK, mW, mU, mGcum, mH, mVnew, mH0, mHt, mma, *layouts).launch(grid=[HEAD_DIM // STATE_COLS, self.HV, 1], block=[STATE_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGcum: cute.Tensor, mH: cute.Tensor, mVnew: cute.Tensor, mH0: cute.Tensor, mHt: cute.Tensor, mma: cute.TiledMma,
        ring_layout: cute.ComposedLayout, u_layout: cute.ComposedLayout, h_layout: cute.ComposedLayout, v_layout: cute.ComposedLayout, o_layout: cute.ComposedLayout
    ):
        tidx, _, _ = cute.arch.thread_idx()
        column_tile_idx, head_idx, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s_k, s_w = smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, ring_layout, 1024)
        s_u, s_h = smem.allocate_tensor(BFloat16, u_layout, 1024), smem.allocate_tensor(BFloat16, h_layout, 1024)
        s_vscaled, s_vnew = smem.allocate_tensor(BFloat16, v_layout, 1024), smem.allocate_tensor(BFloat16, o_layout, 1024)
        s_gcum = smem.allocate_tensor(Float32, cute.make_layout((CHUNK_ROWS, STATE_STAGES), stride=(1, CHUNK_ROWS)), 16)
        coords = (tidx, column_tile_idx, head_idx)
        self.recompute_states(coords, (mK, mW, mU, mGcum), (mH, mVnew, mH0, mHt), mma, (s_k, s_w, s_u, s_h, s_vscaled, s_vnew, s_gcum))

    @cute.jit
    def recompute_states(self, coords, inputs, outputs, mma, buffers):
        tidx, column_tile_idx, head_idx = coords
        mK, mW, mU, mGcum = inputs
        mH, mVnew, mH0, mHt = outputs
        s_k, s_w, s_u, s_h, s_vscaled, s_vnew, s_gcum = buffers
        num_chunks = (mGcum.shape[0] + CHUNK_ROWS - 1) // CHUNK_ROWS if const_expr(self.ragged) else mGcum.shape[0] // CHUNK_ROWS
        key_head_idx = head_idx // (self.HV // self.H)
        load, store_wide, _ = make_gdr_chunk_copies(STATE_THREADS, tidx)
        atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        narrow = cute.make_ordered_layout((STATE_THREADS // (STATE_COLS // 8), STATE_COLS // 8), order=(1, 0))
        load_u = cute.make_tiled_copy_tv(atom_load, narrow, cute.make_layout((1, 8))).get_slice(tidx)
        store_v = cute.make_tiled_copy_tv(atom_store, narrow, cute.make_layout((1, 8))).get_slice(tidx)
        load_gate = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32), cute.make_layout(CHUNK_ROWS), cute.make_layout(1)).get_slice(tidx % CHUNK_ROWS)
        g_key = cute.local_tile(mK[None, key_head_idx, None], (CHUNK_ROWS, HEAD_DIM), (None, 0))  # (chunk row, key, chunk)
        g_w = cute.local_tile(mW[None, head_idx, None], (CHUNK_ROWS, HEAD_DIM), (None, 0))  # (chunk row, key, chunk)
        g_u = cute.local_tile(mU[None, head_idx, None], (CHUNK_ROWS, STATE_COLS), (None, column_tile_idx))  # (chunk row, value tile, chunk)
        if const_expr(self.store_states):
            g_vnew = cute.local_tile(mVnew[None, head_idx, None], (CHUNK_ROWS, STATE_COLS), (None, column_tile_idx))  # (chunk row, value tile, chunk)
        g_gcum = cute.local_tile(mGcum[None, head_idx], (CHUNK_ROWS,), (None,))  # (chunk row, chunk)
        t_k_src, t_w_src, t_u_src, t_g_src = load.partition_S(g_key), load.partition_S(g_w), load_u.partition_S(g_u), load_gate.partition_S(g_gcum)
        t_k_dst, t_w_dst, t_u_dst, t_g_dst = load.partition_D(s_k), load.partition_D(s_w), load_u.partition_D(s_u), load_gate.partition_D(s_gcum)
        if const_expr(self.ragged):
            T = mGcum.shape[0]
            rows_wide = load.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM)))
            rows_narrow = load_u.partition_D(cute.make_identity_tensor((CHUNK_ROWS, STATE_COLS)))
            rows_v = store_v.partition_D(cute.make_identity_tensor((CHUNK_ROWS, STATE_COLS)))
        copies = (load, load_u, load_gate)
        sources = (t_k_src, t_w_src, t_u_src, t_g_src)
        targets = (t_k_dst, t_w_dst, t_u_dst, t_g_dst)
        tail = (rows_wide, rows_narrow, mGcum, s_gcum, head_idx) if const_expr(self.ragged) else None
        self.prefetch_chunks(copies, sources, targets, tail, num_chunks, tidx)
        thread_mma = mma.get_slice(tidx)
        load_a, load_b, load_a_t, _, store_c, store_c_t, load_c = make_gdr_mma_copies(mma, tidx)
        acc_h = cute.make_rmem_tensor(thread_mma.partition_shape_C((HEAD_DIM, STATE_COLS)), Float32)
        acc_h.fill(0.0)
        h_coords = thread_mma.partition_C(cute.make_identity_tensor((HEAD_DIM, STATE_COLS)))
        if const_expr(self.has_h0):
            for elem in cutlass.range_constexpr(cute.size(acc_h)):
                acc_h[elem] = mH0[head_idx, h_coords[elem][0], column_tile_idx * STATE_COLS + h_coords[elem][1]]
        v_coords = thread_mma.partition_C(cute.make_identity_tensor((CHUNK_ROWS, STATE_COLS)))
        s_h_kv = cute.composition(s_h, cute.make_ordered_layout((HEAD_DIM, STATE_COLS), order=(1, 0)))  # (key, value tile)
        s_vscaled_t = cute.composition(s_vscaled, cute.make_ordered_layout((CHUNK_ROWS, STATE_COLS), order=(1, 0)))  # (chunk row, value tile)
        t_h_store, t_vnew_store = store_wide.partition_S(s_h), store_v.partition_S(s_vnew)
        mma_copies = (mma, thread_mma, load_a, load_b, load_a_t, store_c, store_c_t, load_c)
        store_copies = (store_wide, store_v, t_h_store, t_vnew_store)
        for chunk in cutlass.range(num_chunks, unroll=1):
            state_outputs = (mH, g_vnew) if const_expr(self.store_states) else None
            store_tail = (rows_v, T) if const_expr(self.ragged and self.store_states) else None
            self.recompute_chunk(coords, chunk, buffers, (s_h_kv, s_vscaled_t), mma_copies, store_copies, (acc_h, v_coords), state_outputs, store_tail)
            upcoming = chunk + STATE_STAGES - 1
            if upcoming < num_chunks:
                self.load_chunk(copies, sources, targets, tail, upcoming, upcoming % STATE_STAGES, tidx)
            cute.arch.cp_async_commit_group()
            cute.arch.barrier()
        if const_expr(self.has_ht):
            self.store_final_state(coords, mHt, acc_h, h_coords)

    @cute.jit
    def prefetch_chunks(self, copies, sources, targets, tail, num_chunks, tidx):
        for stage in cutlass.range_constexpr(STATE_STAGES - 1):
            if stage < num_chunks:
                self.load_chunk(copies, sources, targets, tail, stage, stage, tidx)
            cute.arch.cp_async_commit_group()

    @cute.jit
    def load_chunk(self, copies, sources, targets, tail, chunk, stage, tidx):
        load, load_u, load_gate = copies
        if const_expr(self.ragged):
            rows_wide, rows_narrow, mGcum, s_gcum, head_idx = tail
            self.issue_tail_chunk(load, load_u, load_gate, sources, targets, (rows_wide, rows_narrow), mGcum, s_gcum, head_idx, chunk, stage, tidx)
        else:
            t_k_src, t_w_src, t_u_src, t_g_src = sources
            t_k_dst, t_w_dst, t_u_dst, t_g_dst = targets
            cute.copy(load, t_k_src[None, None, None, chunk], t_k_dst[None, None, None, stage])
            cute.copy(load, t_w_src[None, None, None, chunk], t_w_dst[None, None, None, stage])
            cute.copy(load_u, t_u_src[None, None, None, chunk], t_u_dst[None, None, None, stage])
            if tidx < CHUNK_ROWS:
                cute.copy(load_gate, t_g_src[None, None, chunk], t_g_dst[None, None, stage])

    @cute.jit
    def recompute_chunk(self, coords, chunk, buffers, views, mma_copies, store_copies, state, outputs, tail):
        _, column_tile_idx, head_idx = coords
        s_k, s_w, s_u, s_h, s_vscaled, s_vnew, s_gcum = buffers
        s_h_kv, s_vscaled_t = views
        mma, thread_mma, load_a, load_b, load_a_t, store_c, store_c_t, load_c = mma_copies
        store_wide, store_v, t_h_store, t_vnew_store = store_copies
        acc_h, v_coords = state
        if const_expr(self.store_states):
            mH, g_vnew = outputs
            if const_expr(self.ragged):
                rows_v, T = tail
        stage = chunk % STATE_STAGES
        s_k_n, s_w_n, s_u_n = s_k[None, None, stage], s_w[None, None, stage], s_u[None, None, stage]
        r_h = cute.make_fragment_like(acc_h, BFloat16)
        r_h.store(acc_h.load().to(BFloat16))
        cute.copy(store_c_t, store_c_t.retile(r_h), store_c_t.partition_D(s_h_kv))
        cute.arch.cp_async_wait_group(STATE_STAGES - 2)
        cute.arch.barrier()
        if const_expr(self.store_states):
            cute.copy(store_wide, t_h_store, store_wide.partition_D(cute.local_tile(mH[chunk, head_idx, None, None], (STATE_COLS, HEAD_DIM), (column_tile_idx, 0))))
        acc_v = cute.make_rmem_tensor(thread_mma.partition_shape_C((CHUNK_ROWS, STATE_COLS)), Float32)
        acc_v.fill(0.0)
        gemm(mma, acc_v, thread_mma.make_fragment_A(thread_mma.partition_A(s_w_n)), thread_mma.make_fragment_B(thread_mma.partition_B(s_h)), load_a.partition_S(s_w_n), load_b.partition_S(s_h), load_a, load_b)
        gate_last = s_gcum[CHUNK_ROWS - 1, stage]
        r_u = cute.make_fragment_like(acc_v, BFloat16)
        cute.copy(load_c, load_c.partition_S(s_u_n), load_c.retile(r_u))
        v_value = r_u.load().to(Float32) - acc_v.load()
        r_vnew = cute.make_fragment_like(acc_v, BFloat16)
        r_vnew.store(v_value.to(BFloat16))
        acc_v.store(v_value)
        for elem in cutlass.range_constexpr(cute.size(acc_v)):
            acc_v[elem] = acc_v[elem] * cute.math.exp2(gate_last - s_gcum[v_coords[elem][0], stage], fastmath=True)
        r_vscaled = cute.make_fragment_like(acc_v, BFloat16)
        r_vscaled.store(acc_v.load().to(BFloat16))
        cute.copy(store_c, store_c.retile(r_vnew), store_c.partition_D(s_vnew))
        cute.copy(store_c_t, store_c_t.retile(r_vscaled), store_c_t.partition_D(s_vscaled_t))
        decay = cute.math.exp2(gate_last, fastmath=True)
        for elem in cutlass.range_constexpr(cute.size(acc_h)):
            acc_h[elem] = acc_h[elem] * decay
        cute.arch.barrier()
        if const_expr(self.store_states):
            if const_expr(self.ragged):
                copy_gdr_rows(store_v, t_vnew_store, store_v.partition_D(g_vnew[None, None, chunk]), rows_v, T - chunk * CHUNK_ROWS, False)
            else:
                cute.copy(store_v, t_vnew_store, store_v.partition_D(g_vnew[None, None, chunk]))
        s_k_n_t = cute.composition(s_k_n, cute.make_ordered_layout((HEAD_DIM, CHUNK_ROWS), order=(1, 0)))
        gemm(mma, acc_h, thread_mma.make_fragment_A(thread_mma.partition_A(s_k_n_t)), thread_mma.make_fragment_B(thread_mma.partition_B(s_vscaled)), load_a_t.partition_S(s_k_n_t), load_b.partition_S(s_vscaled), load_a_t, load_b)

    @cute.jit
    def store_final_state(self, coords, mHt, acc_h, h_coords):
        _, column_tile_idx, head_idx = coords
        for elem in cutlass.range_constexpr(cute.size(acc_h)):
            mHt[head_idx, h_coords[elem][0], column_tile_idx * STATE_COLS + h_coords[elem][1]] = acc_h[elem]

    @cute.jit
    def issue_tail_chunk(self, load: cute.TiledCopy, load_u: cute.TiledCopy, load_gate: cute.TiledCopy, sources: tuple, targets: tuple, rows: tuple, mGcum: cute.Tensor, s_gcum: cute.Tensor, head_idx, chunk, stage, tidx):
        """
        Copy the padded tail into a ring stage, extending the last valid cumulative gate.
        """
        T = mGcum.shape[0]
        num_valid_rows = T - chunk * CHUNK_ROWS
        t_k_src, t_w_src, t_u_src, t_g_src = sources
        t_k_dst, t_w_dst, t_u_dst, t_g_dst = targets
        rows_wide, rows_narrow = rows
        copy_gdr_rows(load, t_k_src[None, None, None, chunk], t_k_dst[None, None, None, stage], rows_wide, num_valid_rows, True)
        copy_gdr_rows(load, t_w_src[None, None, None, chunk], t_w_dst[None, None, None, stage], rows_wide, num_valid_rows, True)
        copy_gdr_rows(load_u, t_u_src[None, None, None, chunk], t_u_dst[None, None, None, stage], rows_narrow, num_valid_rows, True)
        if tidx < CHUNK_ROWS:
            if tidx < num_valid_rows:
                cute.copy(load_gate, t_g_src[None, None, chunk], t_g_dst[None, None, stage])
            else:
                s_gcum[tidx, stage] = mGcum[T - 1, head_idx]


class StateRecomputeTma:
    """
    Dense state recurrence with a dedicated producer warp and two TMA stages.
    """

    stages, gate_tile_width = 2, 4

    def __init__(self, H: int, HV: int):
        self.H, self.HV = H, HV
        self.num_compute_warps = STATE_THREADS // 32
        self.load_warp = self.num_compute_warps

    @cute.jit
    def __call__(self, mK, mW, mU, mGcum, mH, mVnew, stream):
        mmas = (sm90_utils.make_tiled_mma(BFloat16, "K", "K", STATE_COLS), sm90_utils.make_tiled_mma(BFloat16, "MN", "K", STATE_COLS),)
        wide = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, HEAD_DIM), self.stages, major_mode_size=64)
        narrow = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, STATE_COLS), self.stages, major_mode_size=STATE_COLS)
        gate = cute.make_ordered_layout((CHUNK_ROWS, self.gate_tile_width, self.stages), order=(1, 0, 2))
        states = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (STATE_COLS, HEAD_DIM), major_mode_size=64)
        scaled = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (STATE_COLS, CHUNK_ROWS), major_mode_size=64)
        output = cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_COLS), (CHUNK_ROWS, STATE_COLS), (0, 1))
        operands = tuple(make_head_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, cute.slice_(layout, (None, None, 0)), width) for tensor, layout, width in ((mK, wide, HEAD_DIM), (mW, wide, HEAD_DIM), (mU, narrow, STATE_COLS)))
        gates = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mGcum, cute.slice_(gate, (None, None, 0)), (CHUNK_ROWS, self.gate_tile_width),)
        self.kernel(operands, gates, mH, mVnew, mmas, (wide, narrow, gate, states, scaled, output), mGcum.shape[0] // CHUNK_ROWS,).launch(grid=[HEAD_DIM // STATE_COLS, self.HV, 1], block=[STATE_THREADS + 32, 1, 1], stream=stream,)

    @cute.kernel
    def kernel(self, operands, gates, mH, mVnew, mmas, layouts, num_chunks):
        tidx, _, _ = cute.arch.thread_idx()
        column_tile_idx, head_idx, _ = cute.arch.block_idx()
        wide, narrow, gate, states, scaled, output = layouts
        smem = cutlass.utils.SmemAllocator()
        s_k, s_w = smem.allocate_tensor(BFloat16, wide.outer, 1024, swizzle=wide.inner), smem.allocate_tensor(BFloat16, wide.outer, 1024, swizzle=wide.inner)
        s_u, s_gate_ring = smem.allocate_tensor(BFloat16, narrow.outer, 1024, swizzle=narrow.inner), smem.allocate_tensor(Float32, gate, 128)
        s_h = smem.allocate_tensor(BFloat16, states.outer, 1024, swizzle=states.inner)
        s_vscaled, s_vnew = smem.allocate_tensor(BFloat16, scaled.outer, 1024, swizzle=scaled.inner), smem.allocate_tensor(BFloat16, output, 1024)
        full, free = smem.allocate_array(cutlass.Int64, self.stages), smem.allocate_array(cutlass.Int64, self.stages)
        if tidx == 0:
            for slot in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(full + slot, 1)
                cute.arch.mbarrier_init(free + slot, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        coords = (tidx, column_tile_idx, head_idx)
        barriers = (full, free)
        if tidx < self.num_compute_warps * 32:
            self.compute_loop((mH, mVnew), mmas, (s_k, s_w, s_u, s_gate_ring, s_h, s_vscaled, s_vnew), barriers, num_chunks, coords)
        else:
            self.load_loop(operands, gates, (s_k, s_w, s_u, s_gate_ring), barriers, num_chunks, coords)

    @cute.jit
    def load_loop(self, operands, gates, buffers, barriers, num_chunks, coords):
        tidx, column_tile_idx, head_idx = coords
        full, free = barriers
        key_head_idx = head_idx // (self.HV // self.H)
        for chunk in cutlass.range(num_chunks, unroll=1):
            slot = chunk % self.stages
            if chunk >= self.stages:
                cute.arch.mbarrier_wait(free + slot, (chunk // self.stages - 1) % 2)
            self.load_chunk(operands, gates, buffers, full + slot, (tidx, column_tile_idx, head_idx, key_head_idx), chunk, slot)

    @cute.jit
    def load_chunk(self, operands, gates, buffers, barrier, coords, chunk, slot):
        tidx, column_tile_idx, head_idx, key_head_idx = coords
        s_k, s_w, s_u, s_gate_ring = buffers
        (atom_k, coords_k), (atom_w, coords_w), (atom_u, coords_u) = operands
        atom_g, coords_g = gates
        if tidx == self.load_warp * 32:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * CHUNK_ROWS * (2 * HEAD_DIM + STATE_COLS) + 4 * CHUNK_ROWS * self.gate_tile_width)
        # CuTe elects the issuer from the whole producer warp.
        cute.arch.sync_warp()
        k_tile = cute.local_tile(coords_k[None, None, key_head_idx], (CHUNK_ROWS, HEAD_DIM), (chunk, 0))
        w_tile = cute.local_tile(coords_w[None, None, head_idx], (CHUNK_ROWS, HEAD_DIM), (chunk, 0))
        u_tile = cute.local_tile(coords_u[None, None, head_idx], (CHUNK_ROWS, STATE_COLS), (chunk, column_tile_idx))
        g_tile = cute.local_tile(coords_g, (CHUNK_ROWS, self.gate_tile_width), (chunk, head_idx // self.gate_tile_width))
        load_tma_tile(atom_k, k_tile, s_k[None, None, slot], barrier)
        load_tma_tile(atom_w, w_tile, s_w[None, None, slot], barrier)
        load_tma_tile(atom_u, u_tile, s_u[None, None, slot], barrier)
        load_tma_tile(atom_g, g_tile, s_gate_ring[None, None, slot], barrier)

    @cute.jit
    def compute_loop(self, outputs, mmas, buffers, barriers, num_chunks, coords):
        tidx, column_tile_idx, head_idx = coords
        mH, mVnew = outputs
        _, _, _, _, s_h, s_vscaled, s_vnew = buffers
        mma_v, mma_h = mmas
        thread_mma_v, thread_mma_h = mma_v.get_slice(tidx), mma_h.get_slice(tidx)
        warpgroup_mma_v, warpgroup_mma_h = mma_v.get_slice(0), mma_h.get_slice(0)
        _, store_wide, _ = make_gdr_chunk_copies(STATE_THREADS, tidx)
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        narrow = cute.make_ordered_layout((STATE_THREADS // (STATE_COLS // 8), STATE_COLS // 8), order=(1, 0))
        store_v = cute.make_tiled_copy_tv(atom_store, narrow, cute.make_layout((1, 8))).get_slice(tidx)
        acc_h = cute.make_rmem_tensor(thread_mma_h.partition_shape_C((HEAD_DIM, STATE_COLS)), Float32)
        acc_h.fill(0.0)
        v_coords = thread_mma_v.partition_C(cute.make_identity_tensor((CHUNK_ROWS, STATE_COLS)))
        s_h_kv = cute.composition(s_h, cute.make_ordered_layout((HEAD_DIM, STATE_COLS), order=(1, 0)))  # (key, value tile)
        s_vscaled_t = cute.composition(s_vscaled, cute.make_ordered_layout((CHUNK_ROWS, STATE_COLS), order=(1, 0)))  # (chunk row, value tile)
        t_h_store = store_wide.partition_S(s_h)
        t_vnew_store = store_v.partition_S(s_vnew)
        g_vnew = cute.local_tile(mVnew[None, head_idx, None], (CHUNK_ROWS, STATE_COLS), (None, column_tile_idx))  # (chunk row, value tile, chunk)
        store_copies = (store_wide, store_v, t_h_store, t_vnew_store)
        slices = (thread_mma_v, thread_mma_h, warpgroup_mma_v, warpgroup_mma_h)
        views = (s_h_kv, s_vscaled_t, g_vnew)
        for chunk in cutlass.range(num_chunks, unroll=1):
            self.recompute_chunk((mH, chunk), mmas, buffers, barriers, coords, store_copies, slices, views, (acc_h, v_coords))

    @cute.jit
    def recompute_chunk(self, output, mmas, buffers, barriers, coords, store_copies, slices, views, state):
        mH, chunk = output
        tidx, column_tile_idx, head_idx = coords
        s_k, s_w, s_u, s_gate_ring, s_h, s_vscaled, s_vnew = buffers
        full, free = barriers
        mma_v, mma_h = mmas
        thread_mma_v, thread_mma_h, warpgroup_mma_v, warpgroup_mma_h = slices
        s_h_kv, s_vscaled_t, g_vnew = views
        acc_h, v_coords = state
        store_wide, store_v, t_h_store, t_vnew_store = store_copies
        slot = chunk % self.stages
        r_h = cute.make_fragment_like(acc_h, BFloat16)
        r_h.store(acc_h.load().to(BFloat16))
        cute.autovec_copy(r_h, thread_mma_h.partition_C(s_h_kv))
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_wait(full + slot, (chunk // self.stages) % 2)
        cute.arch.barrier(barrier_id=BAR_CHAIN, number_of_threads=STATE_THREADS)
        h_tile = cute.local_tile(mH[chunk, head_idx, None, None], (STATE_COLS, HEAD_DIM), (column_tile_idx, 0))
        cute.copy(store_wide, t_h_store, store_wide.partition_D(h_tile))
        s_w_n, s_u_n = s_w[None, None, slot], s_u[None, None, slot]
        acc_v = cute.make_rmem_tensor(thread_mma_v.partition_shape_C((CHUNK_ROWS, STATE_COLS)), Float32)
        acc_v.fill(0.0)
        sm90_utils.gemm(mma_v, acc_v, warpgroup_mma_v.make_fragment_A(warpgroup_mma_v.partition_A(s_w_n)), warpgroup_mma_v.make_fragment_B(warpgroup_mma_v.partition_B(s_h)), wg_wait=0,)
        gate_head_idx = head_idx % self.gate_tile_width
        gate_last = s_gate_ring[CHUNK_ROWS - 1, gate_head_idx, slot]
        r_u = cute.make_fragment_like(acc_v, BFloat16)
        cute.autovec_copy(thread_mma_v.partition_C(s_u_n), r_u)
        v_value = r_u.load().to(Float32) - acc_v.load()
        r_vnew = cute.make_fragment_like(acc_v, BFloat16)
        r_vnew.store(v_value.to(BFloat16))
        acc_v.store(v_value)
        for elem in cutlass.range_constexpr(cute.size(acc_v)):
            gate_row = s_gate_ring[v_coords[elem][0], gate_head_idx, slot]
            acc_v[elem] = acc_v[elem] * cute.math.exp2(gate_last - gate_row, fastmath=True)
        r_vscaled = cute.make_fragment_like(acc_v, BFloat16)
        r_vscaled.store(acc_v.load().to(BFloat16))
        cute.autovec_copy(r_vnew, thread_mma_v.partition_C(s_vnew))
        cute.autovec_copy(r_vscaled, thread_mma_v.partition_C(s_vscaled_t))
        cute.arch.fence_view_async_shared()
        decay = cute.math.exp2(gate_last, fastmath=True)
        for elem in cutlass.range_constexpr(cute.size(acc_h)):
            acc_h[elem] = acc_h[elem] * decay
        cute.arch.barrier(barrier_id=BAR_CHAIN, number_of_threads=STATE_THREADS)
        cute.copy(store_v, t_vnew_store, store_v.partition_D(g_vnew[None, None, chunk]))
        s_k_n_t = cute.composition(s_k[None, None, slot], cute.make_ordered_layout((HEAD_DIM, CHUNK_ROWS), order=(1, 0)))
        sm90_utils.gemm(mma_h, acc_h, warpgroup_mma_h.make_fragment_A(warpgroup_mma_h.partition_A(s_k_n_t)), warpgroup_mma_h.make_fragment_B(warpgroup_mma_h.partition_B(s_vscaled)), wg_wait=0,)
        cute.arch.barrier(barrier_id=BAR_CHAIN, number_of_threads=STATE_THREADS)
        if tidx == 0:
            cute.arch.mbarrier_arrive(free + slot)


class StateGradient:
    """
    Run the reverse recurrence for dh and du, starting from an optional final-state gradient.
    Optionally store only the initial-state gradient for the context-parallel transition.
    The product P = scale times (q exp2(g)) transposed times do is independent of dh.
    A separate warpgroup computes P from truncated TF32 operands while the chain advances.
    """

    def __init__(self, H: int, HV: int, ragged: bool = False, has_dht: bool = False, has_dh0: bool = False, store_states: bool = True, use_tma: bool = False):
        self.H, self.HV, self.ragged = H, HV, ragged
        self.has_dht, self.has_dh0, self.store_states = has_dht, has_dh0, store_states
        self.use_tma = use_tma
        self.scale = HEAD_DIM ** -0.5
        self.num_chain_warps = STATE_THREADS // 32
        self.product_start_warp = self.num_chain_warps
        self.load_warp = 2 * self.num_chain_warps

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mDo: cute.Tensor, mDvLocal: cute.Tensor, mGcum: cute.Tensor, mDh: cute.Tensor, mDu: cute.Tensor, mDht: cute.Tensor, mDh0: cute.Tensor, stream):
        mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        mma_tf32 = cute.make_tiled_mma(cute.nvgpu.warp.MmaTF32Op((16, 8, 8)), (4, 1, 1), permutation_mnk=(64, 8, 8))
        if const_expr(self.use_tma):
            wide_ring = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, HEAD_DIM), stage=GRAD_STAGES, major_mode_size=HEAD_DIM // 2,)
        else:
            wide_ring = cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (CHUNK_ROWS, HEAD_DIM, GRAD_STAGES), (0, 1, 2))
        gate_layout = None
        if const_expr(self.use_tma):
            cols_ring = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, GRAD_BV), stage=GRAD_STAGES, major_mode_size=GRAD_BV,)
            gate_layout = cute.make_ordered_layout((CHUNK_ROWS, 4, GRAD_STAGES), order=(1, 0, 2))
        else:
            cols_ring = cute.tile_to_shape(get_smem_layout_atom(BFloat16, GRAD_BV), (CHUNK_ROWS, GRAD_BV, GRAD_STAGES), (0, 1, 2))
        layouts = (
            wide_ring, cols_ring, cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (GRAD_BV, HEAD_DIM), (0, 1)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, CHUNK_ROWS), (GRAD_BV, CHUNK_ROWS), (0, 1)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, GRAD_BV), (CHUNK_ROWS, GRAD_BV), (0, 1))
        )
        tma_inputs = None
        if const_expr(self.use_tma):
            wide_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, HEAD_DIM), major_mode_size=HEAD_DIM // 2,)
            tma_inputs = tuple(make_head_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, wide_layout, HEAD_DIM) for tensor in (mK, mW, mQ))
        if const_expr(self.use_tma):
            cols_tile = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, GRAD_BV), major_mode_size=GRAD_BV)
            narrow = tuple(make_head_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, cols_tile, GRAD_BV) for tensor in (mDvLocal, mDo))
            gates = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mGcum, cute.slice_(gate_layout, (None, None, 0)), (CHUNK_ROWS, 4),)
            tma_inputs = (*tma_inputs, *narrow, gates)
        threads = 2 * STATE_THREADS + (32 if self.use_tma else 0)
        self.kernel(mQ, mK, mW, mDo, mDvLocal, mGcum, mDh, mDu, mDht, mDh0, mma, mma_tf32, tma_inputs, gate_layout, *layouts).launch(grid=[HEAD_DIM // GRAD_BV, self.HV, 1], block=[threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mDo: cute.Tensor, mDvLocal: cute.Tensor, mGcum: cute.Tensor, mDh: cute.Tensor, mDu: cute.Tensor, mDht: cute.Tensor,
        mDh0: cute.Tensor, mma: cute.TiledMma, mma_tf32: cute.TiledMma, tma_inputs, gate_layout, ring_layout: cute.ComposedLayout, cols_layout: cute.ComposedLayout, dh_layout: cute.ComposedLayout,
        du_t_layout: cute.ComposedLayout, du_layout: cute.ComposedLayout
    ):
        tidx, _, _ = cute.arch.thread_idx()
        column_tile_idx, head_idx, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        if const_expr(self.use_tma):
            # Keep the ring's swizzle in the pointer so TMA can vectorize each stage.
            s_k, s_w, s_q = (smem.allocate_tensor(BFloat16, ring_layout.outer, 1024, swizzle=ring_layout.inner) for _ in range(3))
        else:
            s_k, s_w = smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, ring_layout, 1024)
            s_q = smem.allocate_tensor(BFloat16, ring_layout, 1024)
        if const_expr(self.use_tma):
            s_dv_local, s_do = (smem.allocate_tensor(BFloat16, cols_layout.outer, 1024, swizzle=cols_layout.inner) for _ in range(2))
        else:
            s_dv_local, s_do = smem.allocate_tensor(BFloat16, cols_layout, 1024), smem.allocate_tensor(BFloat16, cols_layout, 1024)
        s_dh, s_du_t = smem.allocate_tensor(BFloat16, dh_layout, 1024), smem.allocate_tensor(BFloat16, du_t_layout, 1024)
        s_du = smem.allocate_tensor(BFloat16, du_layout, 1024)
        if const_expr(self.use_tma):
            s_gate = smem.allocate_tensor(Float32, gate_layout, 128)
            s_gcum = s_gate[None, head_idx % 4, None]  # (chunk row, ring stage)
        else:
            s_gcum = smem.allocate_tensor(Float32, cute.make_layout((CHUNK_ROWS, GRAD_STAGES), stride=(1, CHUNK_ROWS)), 16)
        s_query_product = smem.allocate_tensor(Float32, cute.make_layout((HEAD_DIM, GRAD_BV, 2), stride=(GRAD_BV + F32_PAD, 1, HEAD_DIM * (GRAD_BV + F32_PAD))), 16)  # (key, value tile, parity)
        s_decay = smem.allocate_tensor(Float32, cute.make_layout(CHUNK_ROWS), 16)  # 2^g of the producer's chunk
        buffers = (s_k, s_w, s_q, s_dv_local, s_do, s_dh, s_du_t, s_du, s_gcum, s_query_product, s_decay)
        tma_barriers = None
        if const_expr(self.use_tma):
            tma_barriers = smem.allocate_array(cutlass.Int64, GRAD_STAGES)
            if tidx == self.product_start_warp * 32:
                for i in cutlass.range_constexpr(GRAD_STAGES):
                    cute.arch.mbarrier_init(tma_barriers + i, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.fence_view_async_shared()
            cute.arch.barrier()
        num_chunks = (mGcum.shape[0] + CHUNK_ROWS - 1) // CHUNK_ROWS if const_expr(self.ragged) else mGcum.shape[0] // CHUNK_ROWS
        coords = (tidx, column_tile_idx, head_idx)
        inputs = (mQ, mK, mW, mDo, mDvLocal, mGcum)
        outputs = (mDh, mDu, mDht, mDh0)
        if const_expr(self.use_tma):
            if tidx >= self.load_warp * 32:
                self.load_loop(tma_inputs, (s_k, s_w, s_q, s_dv_local, s_do, s_gate), tma_barriers, num_chunks, coords)
            elif tidx < self.num_chain_warps * 32:
                self.chain_loop(outputs, mma, buffers, coords, num_chunks, tma_barriers)
            else:
                self.product_loop(inputs, mma_tf32, buffers, (tidx - self.product_start_warp * 32, column_tile_idx, head_idx), tma_barriers)
        else:
            if tidx < self.num_chain_warps * 32:
                self.chain_loop(outputs, mma, buffers, coords, num_chunks, tma_barriers)
            else:
                self.product_loop(inputs, mma_tf32, buffers, (tidx - self.product_start_warp * 32, column_tile_idx, head_idx), tma_barriers)

    @cute.jit
    def load_loop(self, inputs, tiles, full, num_chunks, coords):
        _, _, head_idx = coords
        key_head_idx = head_idx // (self.HV // self.H)
        for step in cutlass.range(num_chunks, unroll=1):
            slot = step % GRAD_STAGES
            if step >= GRAD_STAGES:
                cute.arch.barrier(barrier_id=BAR_FREE + slot, number_of_threads=STATE_THREADS + 32)
            barrier = full + slot
            self.load_reverse_chunk(inputs, tiles, barrier, coords, key_head_idx, num_chunks - 1 - step, slot)

    @cute.jit
    def load_reverse_chunk(self, inputs, tiles, barrier, coords, key_head_idx, chunk, slot):
        tidx, column_tile_idx, head_idx = coords
        if tidx == self.load_warp * 32:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * CHUNK_ROWS * (3 * HEAD_DIM + 2 * GRAD_BV) + 4 * CHUNK_ROWS * 4)
        cute.arch.sync_warp()
        for tile_index in cutlass.range_constexpr(5):
            atom, tma_coords = inputs[tile_index]
            wide = const_expr(tile_index < 3)
            width = HEAD_DIM if wide else GRAD_BV
            column = 0 if wide else column_tile_idx
            head = key_head_idx if const_expr(tile_index == 0 or tile_index == 2) else head_idx
            source = cute.local_tile(tma_coords[None, None, head], (CHUNK_ROWS, width), (chunk, column))
            load_tma_tile(atom, source, tiles[tile_index][None, None, slot], barrier)
        atom, tma_coords = inputs[5]
        gate = cute.local_tile(tma_coords, (CHUNK_ROWS, 4), (chunk, head_idx // 4))
        load_tma_tile(atom, gate, tiles[5][None, None, slot], barrier)

    @cute.jit
    def load_chunk(self, copies: tuple, sources: tuple, targets: tuple, chunk, buffer, tidx, tail: tuple):
        """
        Populate one reverse ring stage with cp.async and commit its group.
        """
        load, load_cols, load_gate = copies
        if const_expr(self.ragged):
            rows_wide, rows_cols, mGcum, head_idx, s_gcum = tail
            T = mGcum.shape[0]
            num_valid_rows = T - chunk * CHUNK_ROWS
            for i in cutlass.range_constexpr(3):
                copy_gdr_rows(load, sources[i][None, None, None, chunk], targets[i][None, None, None, buffer], rows_wide, num_valid_rows, True)
            for i in cutlass.range_constexpr(3, 5):
                copy_gdr_rows(load_cols, sources[i][None, None, None, chunk], targets[i][None, None, None, buffer], rows_cols, num_valid_rows, True)
            if tidx < CHUNK_ROWS:
                if tidx < num_valid_rows:
                    cute.copy(load_gate, sources[5][None, None, chunk], targets[5][None, None, buffer])
                else:
                    s_gcum[tidx, buffer] = mGcum[T - 1, head_idx]
        else:
            for i in cutlass.range_constexpr(3):
                cute.copy(load, sources[i][None, None, None, chunk], targets[i][None, None, None, buffer])
            for i in cutlass.range_constexpr(3, 5):
                cute.copy(load_cols, sources[i][None, None, None, chunk], targets[i][None, None, None, buffer])
            if tidx < CHUNK_ROWS:
                cute.copy(load_gate, sources[5][None, None, chunk], targets[5][None, None, buffer])
        cute.arch.cp_async_commit_group()

    @cute.jit
    def product_loop(self, inputs, mma_tf32, buffers, coords, tma_barriers):
        mQ, mK, mW, mDo, mDvLocal, mGcum = inputs
        tidx, column_tile_idx, head_idx = coords
        s_k, s_w, s_q, s_dv_local, s_do, _, _, _, s_gcum, s_query_product, s_decay = buffers
        num_chunks = (mGcum.shape[0] + CHUNK_ROWS - 1) // CHUNK_ROWS if const_expr(self.ragged) else mGcum.shape[0] // CHUNK_ROWS
        key_head_idx = head_idx // (self.HV // self.H)
        threads = STATE_THREADS
        if const_expr(not self.use_tma):
            load, _, _ = make_gdr_chunk_copies(threads, tidx)
            atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
            load_cols = cute.make_tiled_copy_tv(atom_load, cute.make_ordered_layout((threads // (GRAD_BV // 8), GRAD_BV // 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
            load_gate = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32), cute.make_layout(CHUNK_ROWS), cute.make_layout(1)).get_slice(tidx % CHUNK_ROWS)
            g_key = cute.local_tile(mK[None, key_head_idx, None], (CHUNK_ROWS, HEAD_DIM), (None, 0))  # (chunk row, key, chunk)
            g_w = cute.local_tile(mW[None, head_idx, None], (CHUNK_ROWS, HEAD_DIM), (None, 0))  # (chunk row, key, chunk)
            g_query = cute.local_tile(mQ[None, key_head_idx, None], (CHUNK_ROWS, HEAD_DIM), (None, 0))  # (chunk row, key, chunk)
            g_dv_local = cute.local_tile(mDvLocal[None, head_idx, None], (CHUNK_ROWS, GRAD_BV), (None, column_tile_idx))  # (chunk row, value tile, chunk)
            g_do = cute.local_tile(mDo[None, head_idx, None], (CHUNK_ROWS, GRAD_BV), (None, column_tile_idx))  # (chunk row, value tile, chunk)
            g_gcum = cute.local_tile(mGcum[None, head_idx], (CHUNK_ROWS,), (None,))  # (chunk row, chunk)
            sources = (load.partition_S(g_key), load.partition_S(g_w), load.partition_S(g_query), load_cols.partition_S(g_dv_local), load_cols.partition_S(g_do), load_gate.partition_S(g_gcum))
            targets = (load.partition_D(s_k), load.partition_D(s_w), load.partition_D(s_q), load_cols.partition_D(s_dv_local), load_cols.partition_D(s_do), load_gate.partition_D(s_gcum))
            copies = (load, load_cols, load_gate)
            tail = (load.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM))), load_cols.partition_D(cute.make_identity_tensor((CHUNK_ROWS, GRAD_BV))), mGcum, head_idx, s_gcum) if const_expr(self.ragged) else None
        thread_mma = mma_tf32.get_slice(tidx)
        a_coords = thread_mma.partition_A(cute.make_identity_tensor((HEAD_DIM, CHUNK_ROWS)))
        r_a = cute.make_rmem_tensor(thread_mma.partition_shape_A((HEAD_DIM, CHUNK_ROWS)), Float32)
        r_b = cute.make_rmem_tensor(thread_mma.partition_shape_B((GRAD_BV, CHUNK_ROWS)), Float32)
        # Recast the fp32 operands so the MMA truncates them to TF32 without rounding.
        r_a_tf32 = cute.make_tensor(cute.recast_ptr(r_a.iterator, dtype=cutlass.TFloat32), r_a.layout)
        r_b_tf32 = cute.make_tensor(cute.recast_ptr(r_b.iterator, dtype=cutlass.TFloat32), r_b.layout)
        if const_expr(not self.use_tma):
            # an empty commit group stands in for a missing chunk, so the group count stays fixed
            for step in cutlass.range_constexpr(GRAD_STAGES - 1):
                if step < num_chunks:
                    self.load_chunk(copies, sources, targets, num_chunks - 1 - step, step, tidx, tail)
                else:
                    cute.arch.cp_async_commit_group()
        for step in cutlass.range(num_chunks, unroll=1):
            buffer, parity = step % GRAD_STAGES, step % 2
            if const_expr(self.use_tma):
                cute.arch.mbarrier_wait(tma_barriers + buffer, (step // GRAD_STAGES) % 2)
            else:
                cute.arch.cp_async_wait_group(GRAD_STAGES - 2)
                cute.arch.barrier(barrier_id=BAR_PRODUCER, number_of_threads=threads)
            self.compute_query_product(mma_tf32, (thread_mma, a_coords), (r_a, r_b, r_a_tf32, r_b_tf32), (s_q, s_do, s_gcum, s_decay, s_query_product), (buffer, parity), step, tidx)
            if const_expr(not self.use_tma):
                # chunk step + 2 goes into the stage of step - 1 once the chain warpgroup released it
                upcoming = step + GRAD_STAGES - 1
                if upcoming < num_chunks:
                    if step >= 1:
                        cute.arch.barrier(barrier_id=BAR_FREE + upcoming % GRAD_STAGES, number_of_threads=2 * threads)
                    self.load_chunk(copies, sources, targets, num_chunks - 1 - upcoming, upcoming % GRAD_STAGES, tidx, tail)
                else:
                    cute.arch.cp_async_commit_group()

    @cute.jit
    def compute_query_product(self, mma_tf32, mma_slice, fragments, buffers, stage, step, tidx):
        thread_mma, a_coords = mma_slice
        r_a, r_b, r_a_tf32, r_b_tf32 = fragments
        s_q, s_do, s_gcum, s_decay, s_query_product = buffers
        buffer, parity = stage
        threads = STATE_THREADS
        if tidx < CHUNK_ROWS:
            s_decay[tidx] = cute.math.exp2(s_gcum[tidx, buffer], fastmath=True)
        cute.arch.barrier(barrier_id=BAR_PRODUCER, number_of_threads=threads)
        # The TF32 MMA reads the transposed views in (key / value, token) order.
        t_a = thread_mma.partition_A(cute.composition(s_q[None, None, buffer], cute.make_ordered_layout((HEAD_DIM, CHUNK_ROWS), order=(1, 0))))  # (MMA, MMA_M, MMA_K)
        t_b = thread_mma.partition_B(cute.composition(s_do[None, None, buffer], cute.make_ordered_layout((GRAD_BV, CHUNK_ROWS), order=(1, 0))))  # (MMA, MMA_N, MMA_K)
        for elem in cutlass.range_constexpr(cute.size(r_a)):
            r_a[elem] = Float32(t_a[elem]) * s_decay[a_coords[elem][1]]
        for elem in cutlass.range_constexpr(cute.size(r_b)):
            r_b[elem] = Float32(t_b[elem])
        acc_query_product = cute.make_rmem_tensor(thread_mma.partition_shape_C((HEAD_DIM, GRAD_BV)), Float32)
        acc_query_product.fill(0.0)
        cute.gemm(mma_tf32, acc_query_product, r_a_tf32, r_b_tf32, acc_query_product)
        # The chain must release this buffer's previous product before the overwrite.
        if step >= 2:
            cute.arch.barrier(barrier_id=BAR_FREE_P + parity, number_of_threads=2 * threads)
        t_query_product = thread_mma.partition_C(s_query_product[None, None, parity])  # (MMA, MMA_M, MMA_N)
        for elem in cutlass.range_constexpr(cute.size(acc_query_product)):
            t_query_product[elem] = acc_query_product[elem] * self.scale
        cute.arch.barrier_arrive(barrier_id=BAR_READY + parity, number_of_threads=2 * threads)

    @cute.jit
    def chain_loop(self, outputs, mma, buffers, coords, num_chunks, tma_barriers):
        mDh, mDu, mDht, mDh0 = outputs
        tidx, column_tile_idx, head_idx = coords
        _, _, _, _, _, s_dh, s_du_t, s_du, _, _, _ = buffers
        threads = STATE_THREADS
        _, store_wide, _ = make_gdr_chunk_copies(threads, tidx)
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        store_cols = cute.make_tiled_copy_tv(atom_store, cute.make_ordered_layout((threads // (GRAD_BV // 8), GRAD_BV // 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        thread_mma = mma.get_slice(tidx)
        load_a, load_b, load_a_t, _, store_c, store_c_t, load_c = make_gdr_mma_copies(mma, tidx)
        if const_expr(self.store_states):
            g_du = cute.local_tile(mDu[None, head_idx, None], (CHUNK_ROWS, GRAD_BV), (None, column_tile_idx))  # (chunk row, value tile, chunk)
        acc_dh = cute.make_rmem_tensor(thread_mma.partition_shape_C((HEAD_DIM, GRAD_BV)), Float32)
        acc_dh.fill(0.0)
        dh_coords = thread_mma.partition_C(cute.make_identity_tensor((HEAD_DIM, GRAD_BV)))
        if const_expr(self.has_dht):
            for elem in cutlass.range_constexpr(cute.size(acc_dh)):
                acc_dh[elem] = mDht[head_idx, dh_coords[elem][0], column_tile_idx * GRAD_BV + dh_coords[elem][1]]
        u_coords = thread_mma.partition_C(cute.make_identity_tensor((CHUNK_ROWS, GRAD_BV)))
        s_dh_kv = cute.composition(s_dh, cute.make_ordered_layout((HEAD_DIM, GRAD_BV), order=(1, 0)))  # (key, value tile)
        s_du_tv = cute.composition(s_du_t, cute.make_ordered_layout((CHUNK_ROWS, GRAD_BV), order=(1, 0)))  # (chunk row, value tile)
        t_dh_store, t_du_store = store_wide.partition_S(s_dh), store_cols.partition_S(s_du)
        if const_expr(self.ragged and self.store_states):
            T = mDu.shape[0]
            rows_du = store_cols.partition_D(cute.make_identity_tensor((CHUNK_ROWS, GRAD_BV)))
        mma_copies = (mma, thread_mma, load_a, load_b, load_a_t, store_c, store_c_t, load_c)
        stores = (store_wide, store_cols, t_dh_store, t_du_store)
        views = (s_dh_kv, s_du_tv)
        chunk_outputs = (mDh, g_du) if const_expr(self.store_states) else None
        tail = (rows_du, T) if const_expr(self.ragged and self.store_states) else None
        for step in cutlass.range(num_chunks, unroll=1):
            self.advance_chain(step, num_chunks, coords, (chunk_outputs, tail), (acc_dh, u_coords), (buffers, views), (mma_copies, stores), tma_barriers)
        if const_expr(self.has_dh0):
            for elem in cutlass.range_constexpr(cute.size(acc_dh)):
                mDh0[head_idx, dh_coords[elem][0], column_tile_idx * GRAD_BV + dh_coords[elem][1]] = acc_dh[elem]

    @cute.jit
    def advance_chain(self, step, num_chunks, coords, outputs, state, shared, copies, tma_barriers):
        _, column_tile_idx, head_idx = coords
        chunk_outputs, tail = outputs
        if const_expr(self.store_states):
            mDh, g_du = chunk_outputs
            if const_expr(self.ragged):
                rows_du, T = tail
        acc_dh, u_coords = state
        buffers, views = shared
        s_k, s_w, _, s_dv_local, _, s_dh, s_du_t, s_du, s_gcum, s_query_product, _ = buffers
        s_dh_kv, s_du_tv = views
        mma_copies, stores = copies
        mma, thread_mma, load_a, load_b, load_a_t, store_c, store_c_t, load_c = mma_copies
        store_wide, store_cols, t_dh_store, t_du_store = stores
        threads = STATE_THREADS
        chunk, buffer, parity = num_chunks - 1 - step, step % GRAD_STAGES, step % 2
        s_k_n, s_w_n, s_dv_local_n = s_k[None, None, buffer], s_w[None, None, buffer], s_dv_local[None, None, buffer]
        r_dh = cute.make_fragment_like(acc_dh, BFloat16)
        r_dh.store(acc_dh.load().to(BFloat16))
        cute.copy(store_c_t, store_c_t.retile(r_dh), store_c_t.partition_D(s_dh_kv))
        cute.arch.barrier(barrier_id=BAR_CHAIN, number_of_threads=threads)
        if const_expr(self.store_states):
            cute.copy(store_wide, t_dh_store, store_wide.partition_D(cute.local_tile(mDh[chunk, head_idx, None, None], (GRAD_BV, HEAD_DIM), (column_tile_idx, 0))))
        if const_expr(self.use_tma):
            cute.arch.mbarrier_wait(tma_barriers + buffer, (step // GRAD_STAGES) % 2)
        else:
            cute.arch.barrier(barrier_id=BAR_READY + parity, number_of_threads=2 * threads)
        acc_du = cute.make_rmem_tensor(thread_mma.partition_shape_C((CHUNK_ROWS, GRAD_BV)), Float32)
        acc_du.fill(0.0)
        gemm(mma, acc_du, thread_mma.make_fragment_A(thread_mma.partition_A(s_k_n)), thread_mma.make_fragment_B(thread_mma.partition_B(s_dh)), load_a.partition_S(s_k_n), load_b.partition_S(s_dh), load_a, load_b)
        gate_last = s_gcum[CHUNK_ROWS - 1, buffer]
        r_dv_local = cute.make_fragment_like(acc_du, BFloat16)
        cute.copy(load_c, load_c.partition_S(s_dv_local_n), load_c.retile(r_dv_local))
        dv_local = r_dv_local.load().to(Float32)
        for elem in cutlass.range_constexpr(cute.size(acc_du)):
            acc_du[elem] = acc_du[elem] * cute.math.exp2(gate_last - s_gcum[u_coords[elem][0], buffer], fastmath=True)
        acc_du.store(acc_du.load() + dv_local)
        r_du = cute.make_fragment_like(acc_du, BFloat16)
        r_du.store(acc_du.load().to(BFloat16))
        r_du_neg = cute.make_fragment_like(acc_du, BFloat16)
        r_du_neg.store((-(r_du.load().to(Float32))).to(BFloat16))
        cute.copy(store_c, store_c.retile(r_du), store_c.partition_D(s_du))
        cute.copy(store_c_t, store_c_t.retile(r_du_neg), store_c_t.partition_D(s_du_tv))
        cute.arch.barrier(barrier_id=BAR_CHAIN, number_of_threads=threads)
        if const_expr(self.store_states):
            if const_expr(self.ragged):
                copy_gdr_rows(store_cols, t_du_store, store_cols.partition_D(g_du[None, None, chunk]), rows_du, T - chunk * CHUNK_ROWS, False)
            else:
                cute.copy(store_cols, t_du_store, store_cols.partition_D(g_du[None, None, chunk]))
        s_w_n_t = cute.composition(s_w_n, cute.make_ordered_layout((HEAD_DIM, CHUNK_ROWS), order=(1, 0)))
        acc_wdu = cute.make_rmem_tensor(thread_mma.partition_shape_C((HEAD_DIM, GRAD_BV)), Float32)
        acc_wdu.fill(0.0)
        gemm(mma, acc_wdu, thread_mma.make_fragment_A(thread_mma.partition_A(s_w_n_t)), thread_mma.make_fragment_B(thread_mma.partition_B(s_du_t)), load_a_t.partition_S(s_w_n_t), load_b.partition_S(s_du_t), load_a_t, load_b)
        decay = cute.math.exp2(gate_last, fastmath=True)
        if const_expr(self.use_tma):
            # dU and W.T*dU only require the transport. P joins here.
            cute.arch.barrier(barrier_id=BAR_READY + parity, number_of_threads=2 * threads)
        t_query_product = thread_mma.partition_C(s_query_product[None, None, parity])  # (MMA, MMA_M, MMA_N)
        for elem in cutlass.range_constexpr(cute.size(acc_dh)):
            acc_dh[elem] = acc_dh[elem] * decay + (t_query_product[elem] + acc_wdu[elem])
        if step + 2 < num_chunks:
            cute.arch.barrier_arrive(barrier_id=BAR_FREE_P + parity, number_of_threads=2 * threads)
        # Release the stage only when the loader or product warpgroup will refill it.
        if step + GRAD_STAGES < num_chunks:
            participants = threads + 32 if self.use_tma else 2 * threads
            cute.arch.barrier_arrive(barrier_id=BAR_FREE + buffer, number_of_threads=participants)


def select_chunk_tile(tensor, head_idx, chunk_idx):
    return cute.local_tile(tensor[None, head_idx, None], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))


def transpose_tile(s_tile, num_rows, num_cols):
    return cute.composition(s_tile, cute.make_ordered_layout((num_rows, num_cols), order=(1, 0)))


def fence_chunk_shared():
    """
    Publish shared writes to all threads and to the async proxy used by warpgroup MMA.
    """
    cute.arch.fence_view_async_shared()
    cute.arch.barrier()


def accumulate_chunk_product(mma_tiles, warpgroup_idx, acc, s_left, s_right, is_left_transposed):
    """
    Accumulate s_left times the transpose of s_right, in shared memory on both warpgroups.
    """
    mma_wide, mma_wide_t, mma_square, mma_square_t = mma_tiles
    is_wide = cute.size(acc) == CHUNK_ROWS * HEAD_DIM // GRAD_THREADS
    tiled_mma = (mma_wide_t if is_left_transposed else mma_wide) if is_wide else (mma_square_t if is_left_transposed else mma_square)
    warpgroup_mma = tiled_mma.get_slice(warpgroup_idx)
    sm90_utils.gemm(tiled_mma, acc, warpgroup_mma.make_fragment_A(warpgroup_mma.partition_A(s_left)), warpgroup_mma.make_fragment_B(warpgroup_mma.partition_B(s_right)), zero_init=False, wg_wait=0)


def load_accumulator_values(thread_mma, load_c, s_tile):
    """
    Load a shared tile in the MMA accumulator's thread and fragment order for row reductions.
    """
    r_tile = cute.make_rmem_tensor(thread_mma.partition_shape_C((CHUNK_ROWS, HEAD_DIM)), BFloat16)
    cute.copy(load_c, load_c.partition_S(s_tile), load_c.retile(r_tile))
    return r_tile


class ChunkGradients:
    """
    Compute per-value-head dq, dk, dv, dg and dbeta from the saved and recomputed chunk states.
    In-chunk score gradients precede the WY backward through the saved triangular factor A.
    Reduce gate and beta contributions in a fixed order, then reverse-scan the gate gradient.
    """

    def __init__(self, H: int, HV: int, ragged: bool = False, use_tma: bool = False):
        self.H, self.HV, self.ragged = H, HV, ragged
        self.use_tma = use_tma
        self.scale = HEAD_DIM ** -0.5
        self.state_load_warp, self.operand_load_warp = 0, 1
        self.store_warp, self.gate_scan_warp = 0, 0

    @cute.jit
    def __call__(
        self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mVnew: cute.Tensor, mDo: cute.Tensor, mDu: cute.Tensor, mH: cute.Tensor, mDh: cute.Tensor, mGcum: cute.Tensor,
        mBeta: cute.Tensor, mA: cute.Tensor, mDq: cute.Tensor, mDk: cute.Tensor, mDv: cute.Tensor, mDg: cute.Tensor, mDbeta: cute.Tensor, stream
    ):
        # warpgroup MMAs, the two warpgroups splitting N: (A major, B major) x (N 64 for the
        # (64, 128) products, N 32 for the (64, 64) ones); every operand stays in shared memory
        mmas = tuple(sm90_utils.make_tiled_mma(BFloat16, a_major, b_major, tile_n, "SS", atom_layout_mnk=(1, 2, 1)) for a_major, b_major, tile_n in (("K", "MN", HEAD_DIM // 2), ("MN", "MN", HEAD_DIM // 2), ("K", "K", CHUNK_ROWS // 2), ("MN", "MN", CHUNK_ROWS // 2)))
        # row-major tiles whose swizzle atoms also serve their transposed (MN-major) use
        # per warpgroup
        chunk_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, HEAD_DIM), major_mode_size=HEAD_DIM // 2)
        state_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (HEAD_DIM, HEAD_DIM), major_mode_size=HEAD_DIM // 2)
        square_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (CHUNK_ROWS, CHUNK_ROWS), major_mode_size=CHUNK_ROWS // 2)
        tma_inputs = None
        if const_expr(self.use_tma):
            value_inputs = tuple(make_head_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, chunk_layout, HEAD_DIM) for tensor in (mDu, mDo, mVnew, mK, mQ, mV))
            state_inputs = tuple(cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), cute.make_tensor(tensor.iterator, cute.select(tensor.layout, mode=[2, 3, 1, 0])), state_layout, (HEAD_DIM, HEAD_DIM),) for tensor in (mH, mDh))
            wy_input = make_head_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mA, square_layout, CHUNK_ROWS)
            tma_inputs = (*value_inputs, *state_inputs, wy_input)
        tma_outputs = None
        if const_expr(self.use_tma):
            tma_outputs = tuple(make_head_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), tensor, chunk_layout, HEAD_DIM) for tensor in (mDq, mDv, mDk))
        num_chunks = (mGcum.shape[0] + CHUNK_ROWS - 1) // CHUNK_ROWS if const_expr(self.ragged) else mGcum.shape[0] // CHUNK_ROWS
        self.kernel(mQ, mK, mV, mVnew, mDo, mDu, mH, mDh, mGcum, mBeta, mA, mDq, mDk, mDv, mDg, mDbeta, mmas, chunk_layout, state_layout, square_layout, tma_inputs, tma_outputs).launch(grid=[num_chunks, self.HV, 1], block=[GRAD_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mVnew: cute.Tensor, mDo: cute.Tensor, mDu: cute.Tensor, mH: cute.Tensor, mDh: cute.Tensor, mGcum: cute.Tensor,
        mBeta: cute.Tensor, mA: cute.Tensor, mDq: cute.Tensor, mDk: cute.Tensor, mDv: cute.Tensor, mDg: cute.Tensor, mDbeta: cute.Tensor, mmas: tuple, chunk_layout: cute.ComposedLayout,
        state_layout: cute.ComposedLayout, square_layout: cute.ComposedLayout, tma_inputs, tma_outputs
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, _ = cute.arch.block_idx()
        key_head_idx = head_idx // (self.HV // self.H)
        warp_idx, lane = tidx // 32, tidx % 32
        smem = cutlass.utils.SmemAllocator()
        s_q = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
        s_k = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
        s_v = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
        s_vnew = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
        s_do = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
        s_du = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
        s_h = smem.allocate_tensor(BFloat16, state_layout.outer, 1024, swizzle=state_layout.inner)
        s_dh = smem.allocate_tensor(BFloat16, state_layout.outer, 1024, swizzle=state_layout.inner)
        s_dw = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner)
        s_a = smem.allocate_tensor(BFloat16, square_layout.outer, 1024, swizzle=square_layout.inner)
        s_scores = smem.allocate_tensor(BFloat16, square_layout.outer, 1024, swizzle=square_layout.inner)
        if const_expr(self.use_tma):
            s_wy_mid = smem.allocate_tensor(BFloat16, square_layout.outer, 1024, swizzle=square_layout.inner)
        else:
            s_wy_mid = s_scores
        s_gcum, s_beta = smem.allocate_tensor(Float32, cute.make_layout(CHUNK_ROWS), 16), smem.allocate_tensor(Float32, cute.make_layout(CHUNK_ROWS), 16)
        s_row_sums = smem.allocate_tensor(Float32, cute.make_layout((CHUNK_ROWS, 2, 2), stride=(1, CHUNK_ROWS, 2 * CHUNK_ROWS)), 16)  # (row, column warp, quantity)
        s_warp_sums = smem.allocate_tensor(Float32, cute.make_layout(GRAD_THREADS // 32), 16)
        s_col_sums = smem.allocate_tensor(Float32, cute.make_layout((4, CHUNK_ROWS), stride=(CHUNK_ROWS, 1)), 16)  # (row warp, column)
        # Reuse the state tiles once the state products are complete.
        s_k_beta_gate = cute.make_tensor(s_h.iterator, chunk_layout.outer)
        s_k_beta, s_v_beta = cute.make_tensor(s_h.iterator + CHUNK_ROWS * HEAD_DIM, chunk_layout.outer), cute.make_tensor(s_dh.iterator, chunk_layout.outer)
        load, store_wide, _ = make_gdr_chunk_copies(GRAD_THREADS, tidx)
        load_square = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128),
            cute.make_ordered_layout((GRAD_THREADS // 8, 8), order=(1, 0)), cute.make_layout((1, 8))
        ).get_slice(tidx)
        mma_wide, _, mma_square, _ = mmas
        warpgroup_idx = cute.make_layout(GRAD_THREADS // 128, stride=128)(cute.arch.make_warp_uniform(tidx // 128))
        thread_mma, thread_mma_square = mma_wide.get_slice(tidx), mma_square.get_slice(tidx)
        store_c, load_c = (cute.make_tiled_copy_C(atom, mma_wide).get_slice(tidx) for atom in (cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16), cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)))
        store_c_square = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16), mma_square).get_slice(tidx)

        coords = (tidx, chunk_idx, head_idx, key_head_idx, warp_idx, lane)
        chunk_tiles = (s_q, s_k, s_v, s_vnew, s_do, s_du, s_h, s_dh, s_dw)
        wy_tiles = (s_a, s_scores, s_wy_mid, s_k_beta_gate, s_k_beta, s_v_beta)
        reductions = (s_gcum, s_beta, s_row_sums, s_warp_sums, s_col_sums)
        mma_slices = (mmas, warpgroup_idx, thread_mma, thread_mma_square)
        mma_copies = (store_c, load_c, store_c_square)
        operand_inputs = (mDu, mDo, mVnew, mK, mQ, mV, mH, mDh, mA)
        tma_barrier = smem.allocate_array(cutlass.Int64, 2) if const_expr(self.use_tma and not self.ragged) else None
        tail = self.load_inputs(coords, operand_inputs, (mGcum, mBeta), chunk_tiles, (s_a, s_gcum, s_beta), (load, load_square), tma_inputs, tma_barrier)

        acc_coords = thread_mma.partition_C(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM)))  # (MMA, MMA_M, MMA_N) for wide products
        square_coords = thread_mma_square.partition_C(cute.make_identity_tensor((CHUNK_ROWS, CHUNK_ROWS)))  # (MMA, MMA_M, MMA_N) for square products
        gate_last = s_gcum[CHUNK_ROWS - 1]
        dg_first, dg_second, dbeta_first, dbeta_second = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        # Each thread owns two rows of every accumulator, eight rows apart.
        first_row = acc_coords[0][0]
        gate_first, gate_second = s_gcum[first_row], s_gcum[first_row + 8]
        beta_first, beta_second = s_beta[first_row], s_beta[first_row + 8]
        decay_first, decay_second = cute.math.exp2(gate_first, fastmath=True), cute.math.exp2(gate_second, fastmath=True)
        tail_first, tail_second = cute.math.exp2(gate_last - gate_first, fastmath=True), cute.math.exp2(gate_last - gate_second, fastmath=True)
        row_scales = (beta_first, beta_second, decay_first, decay_second)
        fragment_coords = (acc_coords, square_coords)

        accumulators, r_out, state_sum = self.accumulate_state_products(chunk_tiles, mma_slices, mma_copies, load)
        dg_first, dg_second, dg_last, transposes = self.accumulate_score_gradients(
            (chunk_tiles, wy_tiles, reductions), (mma_slices, mma_copies, fragment_coords), (accumulators, r_out), (dg_first, dg_second),
            (gate_last, tail_first, tail_second, decay_first, decay_second), state_sum, (lane, warp_idx), tma_barrier
        )
        self.scale_wy_operands(chunk_tiles, wy_tiles, (s_gcum, s_beta), load)
        stores = (coords, store_wide, tma_outputs, tail)
        self.store_chunk_tile(stores, 0, s_do, mDq)
        wy_setup = (mma_slices, mma_copies, square_coords)
        wy_gradients = (accumulators, r_out)
        row_sums = (dg_first, dg_second, dbeta_first, dbeta_second)
        da_fragments, s_a_t, row_sums = self.differentiate_wy_operands((chunk_tiles, wy_tiles), wy_setup, wy_gradients, row_sums, row_scales, (stores, mDv))
        dg_first, dg_second, dbeta_first, dbeta_second = self.differentiate_wy_factor((chunk_tiles, wy_tiles, reductions), wy_setup, (accumulators, da_fragments), row_sums, (beta_first, beta_second), (s_a_t, transposes), (lane, warp_idx))
        self.store_gradients(coords, (mDk, mDg, mDbeta), (s_q, s_row_sums, s_col_sums), stores, (store_c, r_out, accumulators[2], acc_coords), (dg_first, dg_second, dbeta_first, dbeta_second, dg_last))

    @cute.jit
    def load_inputs(self, coords, inputs, scalars, chunk_tiles, scalar_tiles, copies, tma_inputs, tma_barrier):
        tidx, chunk_idx, head_idx, key_head_idx, warp_idx, _ = coords
        mDu, mDo, mVnew, mK, mQ, mV, mH, mDh, mA = inputs
        mGcum, mBeta = scalars
        s_q, s_k, s_v, s_vnew, s_do, s_du, s_h, s_dh, _ = chunk_tiles
        s_a, s_gcum, s_beta = scalar_tiles
        load, load_square = copies
        state_tile = lambda m: m[chunk_idx, head_idx, None, None]
        wide_rows, num_valid_rows = None, None
        if const_expr(self.ragged):
            T = mGcum.shape[0]
            num_valid_rows = T - chunk_idx * CHUNK_ROWS
            wide_rows = load.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM)))
            square_rows = load_square.partition_D(cute.make_identity_tensor((CHUNK_ROWS, CHUNK_ROWS)))
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mDu, head_idx, chunk_idx)), load.partition_D(s_du), wide_rows, num_valid_rows, True)
            cute.copy(load, load.partition_S(state_tile(mH)), load.partition_D(s_h))
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mDo, head_idx, chunk_idx)), load.partition_D(s_do), wide_rows, num_valid_rows, True)
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mVnew, head_idx, chunk_idx)), load.partition_D(s_vnew), wide_rows, num_valid_rows, True)
            cute.copy(load, load.partition_S(state_tile(mDh)), load.partition_D(s_dh))
            cute.arch.cp_async_commit_group()
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mK, key_head_idx, chunk_idx)), load.partition_D(s_k), wide_rows, num_valid_rows, True)
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mQ, key_head_idx, chunk_idx)), load.partition_D(s_q), wide_rows, num_valid_rows, True)
            copy_gdr_rows(load, load.partition_S(select_chunk_tile(mV, head_idx, chunk_idx)), load.partition_D(s_v), wide_rows, num_valid_rows, True)
            copy_gdr_rows(load_square, load_square.partition_S(cute.local_tile(mA[None, head_idx, None], (CHUNK_ROWS, CHUNK_ROWS), (chunk_idx, 0))), load_square.partition_D(s_a), square_rows, num_valid_rows, True)
            cute.arch.cp_async_commit_group()
            if tidx < CHUNK_ROWS:
                # both loads at the clamped row, issued together; beta past T is then
                # zeroed in registers
                row = chunk_idx * CHUNK_ROWS + tidx
                clamped_row = cutlass.min(row, T - 1)
                gate, beta = mGcum[clamped_row, head_idx], Float32(mBeta[clamped_row, head_idx])
                if row >= T:
                    beta = Float32(0.0)
                s_gcum[tidx], s_beta[tidx] = gate, beta
        else:
            if const_expr(self.use_tma):
                if tidx == self.state_load_warp * 32:
                    cute.arch.mbarrier_init(tma_barrier, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_barrier, 2 * (3 * CHUNK_ROWS * HEAD_DIM + 2 * HEAD_DIM * HEAD_DIM))
                if tidx == self.operand_load_warp * 32:
                    cute.arch.mbarrier_init(tma_barrier + 1, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_barrier + 1, 2 * (3 * CHUNK_ROWS * HEAD_DIM + CHUNK_ROWS * CHUNK_ROWS))
                if warp_idx == self.state_load_warp:
                    self.load_state_tiles(tma_inputs, (s_du, s_do, s_vnew), (s_h, s_dh), (chunk_idx, head_idx), tma_barrier)
                elif warp_idx == self.operand_load_warp:
                    self.load_wy_tiles(tma_inputs, (s_k, s_q, s_v, s_a), (chunk_idx, head_idx, key_head_idx), tma_barrier)
            else:
                cute.copy(load, load.partition_S(select_chunk_tile(mDu, head_idx, chunk_idx)), load.partition_D(s_du))
                cute.copy(load, load.partition_S(state_tile(mH)), load.partition_D(s_h))
                cute.copy(load, load.partition_S(select_chunk_tile(mDo, head_idx, chunk_idx)), load.partition_D(s_do))
                cute.copy(load, load.partition_S(select_chunk_tile(mVnew, head_idx, chunk_idx)), load.partition_D(s_vnew))
                cute.copy(load, load.partition_S(state_tile(mDh)), load.partition_D(s_dh))
                cute.arch.cp_async_commit_group()
                cute.copy(load, load.partition_S(select_chunk_tile(mK, key_head_idx, chunk_idx)), load.partition_D(s_k))
                cute.copy(load, load.partition_S(select_chunk_tile(mQ, key_head_idx, chunk_idx)), load.partition_D(s_q))
                cute.copy(load, load.partition_S(select_chunk_tile(mV, head_idx, chunk_idx)), load.partition_D(s_v))
                cute.copy(load_square, load_square.partition_S(cute.local_tile(mA[None, head_idx, None], (CHUNK_ROWS, CHUNK_ROWS), (chunk_idx, 0))), load_square.partition_D(s_a))
                cute.arch.cp_async_commit_group()
            if tidx < CHUNK_ROWS:
                row = chunk_idx * CHUNK_ROWS + tidx
                s_gcum[tidx], s_beta[tidx] = mGcum[row, head_idx], Float32(mBeta[row, head_idx])
        if const_expr(self.use_tma):
            # Publish scalar gate/beta writes and both initialized barriers.
            cute.arch.barrier()
            cute.arch.mbarrier_wait(tma_barrier, 0)
        else:
            cute.arch.cp_async_wait_group(1)
            fence_chunk_shared()
        return wide_rows, num_valid_rows

    @cute.jit
    def load_state_tiles(self, tma_inputs, value_tiles, state_tiles, coords, tma_barrier):
        chunk_idx, head_idx = coords
        # CuTe elects the issuing lane inside each copy. The whole
        # producer warp must reach that election after lane 0 arms the barrier.
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(3):
            atom, tma_coords = tma_inputs[i]
            source = cute.local_tile(tma_coords[None, None, head_idx], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))
            load_tma_tile(atom, source, value_tiles[i], tma_barrier)
        for i in cutlass.range_constexpr(2):
            atom, tma_coords = tma_inputs[6 + i]
            load_tma_tile(atom, tma_coords[None, None, head_idx, chunk_idx], state_tiles[i], tma_barrier)

    @cute.jit
    def load_wy_tiles(self, tma_inputs, tiles, coords, tma_barrier):
        chunk_idx, head_idx, key_head_idx = coords
        s_k, s_q, s_v, s_a = tiles
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(3):
            atom, tma_coords = tma_inputs[3 + i]
            head = key_head_idx if const_expr(i < 2) else head_idx
            source = cute.local_tile(tma_coords[None, None, head], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))
            load_tma_tile(atom, source, (s_k, s_q, s_v)[i], tma_barrier + 1)
        atom, tma_coords = tma_inputs[8]
        source = cute.local_tile(tma_coords[None, None, head_idx], (CHUNK_ROWS, CHUNK_ROWS), (chunk_idx, 0))
        load_tma_tile(atom, source, s_a, tma_barrier + 1)

    @cute.jit
    def accumulate_state_products(self, chunk_tiles, mma_slices, mma_copies, load):
        _, _, _, s_vnew, s_do, s_du, s_h, s_dh, s_dw = chunk_tiles
        mmas, warpgroup_idx, thread_mma, thread_mma_square = mma_slices
        store_c, _, _ = mma_copies
        s_h_t, s_dh_t = transpose_tile(s_h, HEAD_DIM, HEAD_DIM), transpose_tile(s_dh, HEAD_DIM, HEAD_DIM)
        acc = cute.make_rmem_tensor(thread_mma.partition_shape_C((CHUNK_ROWS, HEAD_DIM)), Float32)
        acc.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc, s_du, s_h_t, False)
        r_out = cute.make_fragment_like(acc, BFloat16)
        r_out.store((-(acc.load().to(BFloat16).to(Float32))).to(BFloat16))
        cute.copy(store_c, store_c.retile(r_out), store_c.partition_D(s_dw))
        acc_dq = cute.make_rmem_tensor(thread_mma.partition_shape_C((CHUNK_ROWS, HEAD_DIM)), Float32)
        acc_dq.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_dq, s_do, s_h_t, False)
        acc_ds = cute.make_rmem_tensor(thread_mma_square.partition_shape_C((CHUNK_ROWS, CHUNK_ROWS)), Float32)
        acc_ds.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_ds, s_do, s_vnew, False)
        acc_dk = cute.make_rmem_tensor(thread_mma.partition_shape_C((CHUNK_ROWS, HEAD_DIM)), Float32)
        acc_dk.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_dk, s_vnew, s_dh_t, False)
        t_h, t_dh = load.partition_D(s_h), load.partition_D(s_dh)
        state_sum = Float32(0.0)
        for i in cutlass.range_constexpr(cute.size(t_h, mode=[1])):
            for j in cutlass.range_constexpr(cute.size(t_h, mode=[2])):
                pairs = t_h[None, i, j].load().to(Float32) * t_dh[None, i, j].load().to(Float32)
                for elem in cutlass.range_constexpr(8):
                    state_sum = state_sum + pairs[elem]
        return (acc, acc_dq, acc_dk, acc_ds), r_out, state_sum

    @cute.jit
    def accumulate_score_gradients(self, tiles, mma_setup, gradients, gate_sums, gate_scales, state_sum, warp_coords, tma_barrier):
        chunk_tiles, wy_tiles, reductions = tiles
        mma_slices, mma_copies, fragment_coords = mma_setup
        accumulators, r_out = gradients
        lane, warp_idx = warp_coords
        s_q, s_k, _, _, s_do, _, _, _, _ = chunk_tiles
        _, s_scores, _, _, _, _ = wy_tiles
        s_gcum, _, s_row_sums, s_warp_sums, _ = reductions
        mmas, warpgroup_idx, thread_mma, _ = mma_slices
        store_c, load_c, store_c_square = mma_copies
        _, square_coords = fragment_coords
        _, acc_dq, acc_dk, acc_ds = accumulators
        dg_first, dg_second = gate_sums
        gate_last, tail_first, tail_second, decay_first, decay_second = gate_scales
        if const_expr(self.use_tma):
            cute.arch.mbarrier_wait(tma_barrier + 1, 0)
        else:
            cute.arch.cp_async_wait_group(0)
            fence_chunk_shared()
        key_sum = Float32(0.0)
        r_k_c = load_accumulator_values(thread_mma, load_c, s_k)
        for elem in cutlass.range_constexpr(cute.size(acc_dq)):
            decay, tail = (decay_second, tail_second) if elem % 4 >= 2 else (decay_first, tail_first)
            acc_dq[elem], acc_dk[elem] = acc_dq[elem] * decay * self.scale, acc_dk[elem] * tail
            key_sum = key_sum + acc_dk[elem] * Float32(r_k_c[elem])
        for elem in cutlass.range_constexpr(cute.size(acc_ds)):
            row, column_idx = square_coords[elem]
            value = Float32(0.0)
            if row >= column_idx:
                value = acc_ds[elem] * cute.math.exp2(s_gcum[row] - s_gcum[column_idx], fastmath=True) * self.scale
            acc_ds[elem] = value
        r_ds = cute.make_fragment_like(acc_ds, BFloat16)
        r_ds.store(acc_ds.load().to(BFloat16))
        cute.copy(store_c_square, store_c_square.retile(r_ds), store_c_square.partition_D(s_scores))
        # dg_last: a block sum in a fixed order
        partial = state_sum
        for offset in cutlass.range_constexpr(5):
            partial = partial + cute.arch.shuffle_sync_bfly(partial, offset=1 << (4 - offset))
        partial_keys = key_sum
        for offset in cutlass.range_constexpr(5):
            partial_keys = partial_keys + cute.arch.shuffle_sync_bfly(partial_keys, offset=1 << (4 - offset))
        if lane == 0:
            s_warp_sums[warp_idx] = partial
            s_row_sums[warp_idx, 0, 0] = partial_keys
        fence_chunk_shared()
        s_k_t, s_q_t, s_scores_t = transpose_tile(s_k, HEAD_DIM, CHUNK_ROWS), transpose_tile(s_q, HEAD_DIM, CHUNK_ROWS), transpose_tile(s_scores, CHUNK_ROWS, CHUNK_ROWS)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_dq, s_scores, s_k_t, False)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_dk, s_scores_t, s_q_t, True)
        state_total, key_total = Float32(0.0), Float32(0.0)
        for partial_idx in cutlass.range_constexpr(GRAD_THREADS // 32):
            state_total, key_total = state_total + s_warp_sums[partial_idx], key_total + s_row_sums[partial_idx, 0, 0]
        dg_last = state_total * cute.math.exp2(gate_last, fastmath=True) + key_total
        r_q_c, r_k_c = load_accumulator_values(thread_mma, load_c, s_q), load_accumulator_values(thread_mma, load_c, s_k)
        for elem in cutlass.range_constexpr(cute.size(acc_dq)):
            contribution = acc_dq[elem] * Float32(r_q_c[elem]) - acc_dk[elem] * Float32(r_k_c[elem])
            if const_expr(elem % 4 < 2):
                dg_first = dg_first + contribution
            else:
                dg_second = dg_second + contribution
        r_out.store(acc_dq.load().to(BFloat16))
        fence_chunk_shared()  # every thread is done with do, v_new, h, dh and the dg_last partials
        cute.copy(store_c, store_c.retile(r_out), store_c.partition_D(s_do))
        return dg_first, dg_second, dg_last, (s_k_t, s_scores_t)

    @cute.jit
    def scale_wy_operands(self, chunk_tiles, wy_tiles, scalars, load):
        _, s_k, s_v, _, _, _, _, _, _ = chunk_tiles
        _, _, _, s_k_beta_gate, s_k_beta, s_v_beta = wy_tiles
        s_gcum, s_beta = scalars
        t_k = load.partition_D(s_k)
        t_v, t_k_beta_gate, t_k_beta, t_v_beta = load.partition_D(s_v), load.partition_D(s_k_beta_gate), load.partition_D(s_k_beta), load.partition_D(s_v_beta)
        row_coords = load.partition_D(cute.make_identity_tensor((CHUNK_ROWS, HEAD_DIM)))
        for i in cutlass.range_constexpr(cute.size(t_k, mode=[1])):
            for j in cutlass.range_constexpr(cute.size(t_k, mode=[2])):
                row = row_coords[0, i, j][0]
                beta = s_beta[row]
                keys = t_k[None, i, j].load().to(Float32)
                t_k_beta_gate[None, i, j].store((keys * (beta * cute.math.exp2(s_gcum[row], fastmath=True))).to(BFloat16))
                t_k_beta[None, i, j].store((keys * beta).to(BFloat16))
                t_v_beta[None, i, j].store((t_v[None, i, j].load().to(Float32) * beta).to(BFloat16))
        fence_chunk_shared()

    @cute.jit
    def differentiate_wy_operands(self, tiles, mma_setup, gradients, row_sums, row_scales, value_output):
        chunk_tiles, wy_tiles = tiles
        mma_slices, mma_copies, square_coords = mma_setup
        accumulators, r_out = gradients
        stores, mDv = value_output
        _, s_k, s_v, s_vnew, _, s_du, _, _, s_dw = chunk_tiles
        s_a, s_scores, _, s_k_beta_gate, _, s_v_beta = wy_tiles
        mmas, warpgroup_idx, thread_mma, thread_mma_square = mma_slices
        store_c, load_c, store_c_square = mma_copies
        acc, _, acc_dk, _ = accumulators
        dg_first, dg_second, dbeta_first, dbeta_second = row_sums
        beta_first, beta_second, decay_first, decay_second = row_scales
        acc_da = cute.make_rmem_tensor(thread_mma_square.partition_shape_C((CHUNK_ROWS, CHUNK_ROWS)), Float32)
        acc_da.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_da, s_dw, s_k_beta_gate, False)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_da, s_du, s_v_beta, False)
        s_a_t, s_dw_t, s_du_t = transpose_tile(s_a, CHUNK_ROWS, CHUNK_ROWS), transpose_tile(s_dw, HEAD_DIM, CHUNK_ROWS), transpose_tile(s_du, HEAD_DIM, CHUNK_ROWS)
        acc.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc, s_a_t, s_dw_t, True)
        r_k_c = load_accumulator_values(thread_mma, load_c, s_k)
        for elem in cutlass.range_constexpr(cute.size(acc)):
            # the fragment's elements 2 and 3 of every 4 sit on the thread's second row
            beta, decay = (beta_second, decay_second) if elem % 4 >= 2 else (beta_first, decay_first)
            keys = Float32(r_k_c[elem])
            acc_dk[elem] = acc_dk[elem] + acc[elem] * (decay * beta)
            beta_term, gate_term = acc[elem] * keys * decay, acc[elem] * (keys * (beta * decay))
            if const_expr(elem % 4 < 2):
                dbeta_first, dg_first = dbeta_first + beta_term, dg_first + gate_term
            else:
                dbeta_second, dg_second = dbeta_second + beta_term, dg_second + gate_term
        acc.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc, s_a_t, s_du_t, True)
        r_v_c = load_accumulator_values(thread_mma, load_c, s_v)
        for elem in cutlass.range_constexpr(cute.size(acc)):
            beta = beta_second if elem % 4 >= 2 else beta_first
            beta_term = acc[elem] * Float32(r_v_c[elem])
            if const_expr(elem % 4 < 2):
                dbeta_first = dbeta_first + beta_term
            else:
                dbeta_second = dbeta_second + beta_term
            acc[elem] = acc[elem] * beta
        r_out.store(acc.load().to(BFloat16))
        for elem in cutlass.range_constexpr(cute.size(acc_da)):
            row, column_idx = square_coords[elem]
            value = Float32(0.0)
            if row > column_idx:
                value = acc_da[elem]
            acc_da[elem] = value
        r_da = cute.make_fragment_like(acc_da, BFloat16)
        r_da.store(acc_da.load().to(BFloat16))
        if const_expr(not self.use_tma):
            fence_chunk_shared()  # the aliased global-store staging is no longer read
        cute.copy(store_c, store_c.retile(r_out), store_c.partition_D(s_vnew))
        cute.copy(store_c_square, store_c_square.retile(r_da), store_c_square.partition_D(s_scores))
        fence_chunk_shared()
        self.store_chunk_tile(stores, 1, s_vnew, mDv)
        return (acc_da, r_da), s_a_t, (dg_first, dg_second, dbeta_first, dbeta_second)

    @cute.jit
    def differentiate_wy_factor(self, tiles, mma_setup, gradients, row_sums, beta_rows, transposes, warp_coords):
        chunk_tiles, wy_tiles, reductions = tiles
        mma_slices, mma_copies, square_coords = mma_setup
        accumulators, (acc_da, r_da) = gradients
        _, s_k, _, _, _, _, _, _, _ = chunk_tiles
        s_a, s_scores, s_wy_mid, _, s_k_beta, _ = wy_tiles
        s_gcum, _, _, _, s_col_sums = reductions
        mmas, warpgroup_idx, thread_mma, thread_mma_square = mma_slices
        _, load_c, store_c_square = mma_copies
        acc, _, acc_dk, _ = accumulators
        dg_first, dg_second, dbeta_first, dbeta_second = row_sums
        beta_first, beta_second = beta_rows
        s_a_t, (s_k_t, s_scores_t) = transposes
        lane, warp_idx = warp_coords
        acc_da.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_da, s_scores, s_a, False)
        r_da.store(acc_da.load().to(BFloat16))
        if const_expr(not self.use_tma):
            fence_chunk_shared()  # the intermediate overwrites its input
        cute.copy(store_c_square, store_c_square.retile(r_da), store_c_square.partition_D(s_wy_mid))
        fence_chunk_shared()
        acc_da.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_da, s_a_t, transpose_tile(s_wy_mid, CHUNK_ROWS, CHUNK_ROWS), True)
        for elem in cutlass.range_constexpr(cute.size(acc_da)):
            row, column_idx = square_coords[elem]
            value = Float32(0.0)
            if row > column_idx:
                value = -acc_da[elem] * cute.math.exp2(s_gcum[row] - s_gcum[column_idx], fastmath=True)
            acc_da[elem] = value
        r_da.store(acc_da.load().to(BFloat16))
        acc_kk = cute.make_rmem_tensor(thread_mma_square.partition_shape_C((CHUNK_ROWS, CHUNK_ROWS)), Float32)
        acc_kk.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_kk, s_k, s_k, False)
        # C = dA * beta_i k_i.k_j in acc_kk; dg += rowsum(C) - colsum(C)
        for elem in cutlass.range_constexpr(cute.size(acc_kk)):
            beta = beta_second if elem % 4 >= 2 else beta_first
            acc_kk[elem] = Float32(r_da[elem]) * (acc_kk[elem] * beta)
            if const_expr(elem % 4 < 2):
                dg_first = dg_first + acc_kk[elem]
            else:
                dg_second = dg_second + acc_kk[elem]
        # an MMA C fragment is (2 columns, 2 rows 8 apart) per atom: elements e and e + 2
        # share a column
        for atom in cutlass.range_constexpr(cute.size(acc_kk) // 4):
            for half in cutlass.range_constexpr(2):
                part = acc_kk[4 * atom + half] + acc_kk[4 * atom + half + 2]
                for offset in cutlass.range_constexpr(3):
                    part = part + cute.arch.shuffle_sync_bfly(part, offset=4 << offset)
                if lane < 4:
                    s_col_sums[warp_idx % 4, square_coords[4 * atom + half][1]] = part
        if const_expr(not self.use_tma):
            fence_chunk_shared()  # the final WY tile aliases the previous MMA input
        cute.copy(store_c_square, store_c_square.retile(r_da), store_c_square.partition_D(s_scores))
        fence_chunk_shared()
        acc.fill(0.0)
        accumulate_chunk_product(mmas, warpgroup_idx, acc, s_scores, s_k_t, False)
        r_k_c = load_accumulator_values(thread_mma, load_c, s_k)
        for elem in cutlass.range_constexpr(cute.size(acc)):
            beta = beta_second if elem % 4 >= 2 else beta_first
            beta_term = acc[elem] * Float32(r_k_c[elem])
            if const_expr(elem % 4 < 2):
                dbeta_first = dbeta_first + beta_term
            else:
                dbeta_second = dbeta_second + beta_term
            acc_dk[elem] = acc_dk[elem] + acc[elem] * beta
        s_k_beta_t = transpose_tile(s_k_beta, HEAD_DIM, CHUNK_ROWS)
        accumulate_chunk_product(mmas, warpgroup_idx, acc_dk, s_scores_t, s_k_beta_t, True)
        return dg_first, dg_second, dbeta_first, dbeta_second

    @cute.jit
    def store_chunk_tile(self, stores, output_idx: cutlass.Constexpr[int], s_tile, mOut):
        coords, store_wide, tma_outputs, tail = stores
        _, chunk_idx, head_idx, _, warp_idx, _ = coords
        wide_rows, num_valid_rows = tail
        if const_expr(self.use_tma):
            if warp_idx == self.store_warp:
                atom, tma_coords = tma_outputs[output_idx]
                output = cute.local_tile(tma_coords[None, None, head_idx], (CHUNK_ROWS, HEAD_DIM), (chunk_idx, 0))
                store_tma_tile(atom, s_tile, output)
                cute.arch.cp_async_bulk_commit_group()
        elif const_expr(self.ragged):
            copy_gdr_rows(store_wide, store_wide.partition_S(s_tile), store_wide.partition_D(select_chunk_tile(mOut, head_idx, chunk_idx)), wide_rows, num_valid_rows, False)
        else:
            cute.copy(store_wide, store_wide.partition_S(s_tile), store_wide.partition_D(select_chunk_tile(mOut, head_idx, chunk_idx)))

    @cute.jit
    def store_gradients(self, coords, outputs, tiles, stores, fragments, row_sums):
        tidx, chunk_idx, head_idx, _, warp_idx, lane = coords
        mDk, mDg, mDbeta = outputs
        s_q, s_row_sums, s_col_sums = tiles
        store_c, r_out, acc_dk, acc_coords = fragments
        dg_first, dg_second, dbeta_first, dbeta_second, dg_last = row_sums
        _, _, _, tail = stores
        _, num_valid_rows = tail
        for offset in cutlass.range_constexpr(2):
            dg_first = dg_first + cute.arch.shuffle_sync_bfly(dg_first, offset=1 << offset)
            dg_second = dg_second + cute.arch.shuffle_sync_bfly(dg_second, offset=1 << offset)
            dbeta_first = dbeta_first + cute.arch.shuffle_sync_bfly(dbeta_first, offset=1 << offset)
            dbeta_second = dbeta_second + cute.arch.shuffle_sync_bfly(dbeta_second, offset=1 << offset)
        first_row = acc_coords[0][0]
        column_warp = warp_idx // 4
        if lane % 4 == 0:
            s_row_sums[first_row, column_warp, 0], s_row_sums[first_row + 8, column_warp, 0] = dg_first, dg_second
            s_row_sums[first_row, column_warp, 1], s_row_sums[first_row + 8, column_warp, 1] = dbeta_first, dbeta_second
        r_out.store(acc_dk.load().to(BFloat16))
        cute.copy(store_c, store_c.retile(r_out), store_c.partition_D(s_q))
        fence_chunk_shared()
        self.store_chunk_tile(stores, 2, s_q, mDk)
        if tidx < CHUNK_ROWS:
            gate_sum = s_row_sums[tidx, 0, 0] + s_row_sums[tidx, 1, 0] - (s_col_sums[0, tidx] + s_col_sums[1, tidx] + s_col_sums[2, tidx] + s_col_sums[3, tidx])
            if tidx == CHUNK_ROWS - 1:
                gate_sum = gate_sum + dg_last
            beta_sum = s_row_sums[tidx, 0, 1] + s_row_sums[tidx, 1, 1]
            if const_expr(self.ragged):
                if tidx < num_valid_rows:
                    mDbeta[chunk_idx * CHUNK_ROWS + tidx, head_idx] = BFloat16(beta_sum)
            else:
                mDbeta[chunk_idx * CHUNK_ROWS + tidx, head_idx] = BFloat16(beta_sum)
            s_row_sums[tidx, 0, 0] = gate_sum
        fence_chunk_shared()
        if warp_idx == self.gate_scan_warp:
            self.scan_gate_gradient(mDg, s_row_sums, (chunk_idx, head_idx, lane), num_valid_rows)
        if const_expr(self.use_tma):
            if warp_idx == self.store_warp:
                cute.arch.cp_async_bulk_wait_group(0)

    @cute.jit
    def scan_gate_gradient(self, mDg, s_row_sums, coords, num_valid_rows):
        chunk_idx, head_idx, lane = coords
        # Lane l owns rows 2l and 2l + 1, so each shuffle adds a pair of rows.
        even, odd = s_row_sums[2 * lane, 0, 0], s_row_sums[2 * lane + 1, 0, 0]
        running = even + odd
        for i in cutlass.range_constexpr(5):
            from_above = cute.arch.shuffle_sync_down(running, offset=1 << i)
            if lane + (1 << i) < 32:
                running = running + from_above
        after = cute.arch.shuffle_sync_down(running, offset=1)
        if lane == 31:
            after = Float32(0.0)
        odd_total, row = odd + after, chunk_idx * CHUNK_ROWS + 2 * lane
        if const_expr(self.ragged):
            if 2 * lane < num_valid_rows:
                mDg[row, head_idx] = even + odd_total
            if 2 * lane + 1 < num_valid_rows:
                mDg[row + 1, head_idx] = odd_total
        else:
            mDg[row, head_idx], mDg[row + 1, head_idx] = even + odd_total, odd_total


COMPILED: dict[tuple, object] = {}


def compile_backward_kernel(name: str, H: int, HV: int, device: int, ragged: bool, flags: tuple = ()) -> object:
    """
    Cache the compiled launch by kernel, head counts, device and optional state I/O.
    T is symbolic, so each ragged variant serves every sequence length with a partial chunk.
    """
    key = (name, H, HV, device, flags, ragged)
    if key not in COMPILED:
        T, num_chunks = cute.sym_int(), cute.sym_int()
        keys, values = fake(BFloat16, (T, H, HEAD_DIM), (H * HEAD_DIM, HEAD_DIM, 1)), fake(BFloat16, (T, HV, HEAD_DIM), (HV * HEAD_DIM, HEAD_DIM, 1))
        gate = fake(Float32, (T, HV), (HV, 1), 4)
        states = fake(BFloat16, (num_chunks, HV, HEAD_DIM, HEAD_DIM), (HV * HEAD_DIM * HEAD_DIM, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1))
        state_in = fake(Float32, (HV, HEAD_DIM, HEAD_DIM), (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1))
        state_out = fake(Float32, (HV, HEAD_DIM, HEAD_DIM), (2 * HEAD_DIM * HEAD_DIM, 2 * HEAD_DIM, 1))
        has_state_in, has_state_out, store_states = flags if flags else (False, False, True)
        if name in ("bwd_prologue", "bwd_prologue_tma", "chunk_gradients", "chunk_gradients_epilogue"):
            A, beta = fake(BFloat16, (T, HV, CHUNK_ROWS), (HV * CHUNK_ROWS, CHUNK_ROWS, 1)), fake(BFloat16, (T, HV), (HV, 1), 2)
        match name:
            case "bwd_prologue" | "bwd_prologue_tma":
                COMPILED[key] = cute.compile(BwdPrologue(H, HV, ragged, tma=name == "bwd_prologue_tma"), keys, keys, values, values, gate, beta, A, values, values, values, fake_stream(), options="--enable-tvm-ffi")
            case "state_gradient" | "state_gradient_loader":
                COMPILED[key] = cute.compile(
                    StateGradient(H, HV, ragged, has_state_in, has_state_out, store_states, use_tma=name == "state_gradient_loader"), keys, keys,
                    values, values, values, gate if name != "state_gradient_loader" else fake(Float32, (T, HV), (HV, 1), 16), states if store_states else None, values if store_states else None,
                    state_in if has_state_in else None, state_out if has_state_out else None, fake_stream(), options="--enable-tvm-ffi"
                )
            case "chunk_gradients" | "chunk_gradients_epilogue":
                COMPILED[key] = cute.compile(
                    ChunkGradients(H, HV, ragged, use_tma=name == "chunk_gradients_epilogue"), keys, keys, values, values, values, values, states,
                    states, gate, beta, A, values, values, values, gate, beta, fake_stream(), options="--enable-tvm-ffi"
                )
            case "state_recompute_tma":
                aligned_gate = fake(Float32, (T, HV), (HV, 1), 16)
                COMPILED[key] = cute.compile(StateRecomputeTma(H, HV), keys, values, values, aligned_gate, states, values, fake_stream(), options="--enable-tvm-ffi")
            case "state_recompute":
                COMPILED[key] = cute.compile(
                    StateRecompute(H, HV, ragged, has_state_in, has_state_out, store_states), keys, values, values, gate, states if store_states else None,
                    values if store_states else None, state_in if has_state_in else None, state_out if has_state_out else None, fake_stream(), options="--enable-tvm-ffi"
                )
    return COMPILED[key]


def bwd_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recompute w, u and dv_local in bf16 from the saved WY factor and output gradient.
    """
    T, H, HV = k.shape[1], k.shape[2], v.shape[2]
    w, u, dv_local = k.new_empty(1, T, HV, HEAD_DIM), torch.empty_like(v), torch.empty_like(v)
    compile_backward_kernel("bwd_prologue", H, HV, k.device.index, T % CHUNK_ROWS != 0)(q[0], k[0], v[0], do[0], g_cum[0], beta[0], A[0], w[0], u[0], dv_local[0])
    return w, u, dv_local


def state_recompute(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cum: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return V-first bf16 chunk-entry states and v_new, optionally starting at fp32 h0.
    """
    T, H, HV = k.shape[1], k.shape[2], u.shape[2]
    h, v_new = k.new_empty(1, (T + CHUNK_ROWS - 1) // CHUNK_ROWS, HV, HEAD_DIM, HEAD_DIM), torch.empty_like(u)
    flags = () if h0 is None else (True, False, True)
    compile_backward_kernel("state_recompute", H, HV, k.device.index, T % CHUNK_ROWS != 0, flags)(k[0], w[0], u[0], g_cum[0], h[0], v_new[0], h0, None)
    return h, v_new


def state_final(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cum: torch.Tensor, ht: torch.Tensor) -> None:
    """
    Write the fp32 final state into ht, a (HV, K, V) view with row stride 2 K.
    """
    T, H, HV = k.shape[1], k.shape[2], u.shape[2]
    compile_backward_kernel("state_recompute", H, HV, k.device.index, T % CHUNK_ROWS != 0, (False, True, False))(k[0], w[0], u[0], g_cum[0], None, None, None, ht)


def state_gradient(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, do: torch.Tensor, dv_local: torch.Tensor, g_cum: torch.Tensor, du: torch.Tensor | None = None, dht: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return V-first bf16 chunk-state gradients and du, optionally starting at fp32 dht.
    Reuse the supplied du buffer when present.
    """
    return launch_state_gradient("state_gradient", q, k, w, do, dv_local, g_cum, du, dht)


def launch_state_gradient(name, q, k, w, do, dv_local, g_cum, du, dht):
    T, H, HV = k.shape[1], k.shape[2], do.shape[2]
    dh = k.new_empty(1, (T + CHUNK_ROWS - 1) // CHUNK_ROWS, HV, HEAD_DIM, HEAD_DIM)
    du = torch.empty_like(do) if du is None else du
    flags = () if dht is None else (True, False, True)
    compile_backward_kernel(name, H, HV, k.device.index, T % CHUNK_ROWS != 0, flags)(q[0], k[0], w[0], do[0], dv_local[0], g_cum[0], dh[0], du[0], dht, None)
    return dh, du


def state_gradient_initial(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, do: torch.Tensor, dv_local: torch.Tensor, g_cum: torch.Tensor, dh0: torch.Tensor) -> None:
    """
    Write the fp32 initial-state gradient into dh0, a (HV, K, V) view with row stride 2 K.
    """
    T, H, HV = k.shape[1], k.shape[2], do.shape[2]
    compile_backward_kernel("state_gradient", H, HV, k.device.index, T % CHUNK_ROWS != 0, (False, True, False))(q[0], k[0], w[0], do[0], dv_local[0], g_cum[0], None, None, None, dh0)


def chunk_gradients(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, v_new: torch.Tensor, do: torch.Tensor, du: torch.Tensor, h: torch.Tensor, dh: torch.Tensor, g_cum: torch.Tensor,
    beta: torch.Tensor, A: torch.Tensor, dv: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return per-value-head dq, dk, dv, dg and dbeta, optionally storing dv in the supplied buffer.
    """
    return launch_chunk_gradients("chunk_gradients", q, k, v, v_new, do, du, h, dh, g_cum, beta, A, dv)


def launch_chunk_gradients(name, q, k, v, v_new, do, du, h, dh, g_cum, beta, A, dv):
    T, H, HV = k.shape[1], k.shape[2], v.shape[2]
    dq, dk = q.new_empty(1, T, HV, HEAD_DIM), k.new_empty(1, T, HV, HEAD_DIM)
    dv = torch.empty_like(v) if dv is None else dv
    dg, dbeta = torch.empty_like(g_cum), torch.empty_like(beta)
    compile_backward_kernel(name, H, HV, k.device.index, T % CHUNK_ROWS != 0)(q[0], k[0], v[0], v_new[0], do[0], du[0], h[0], dh[0], g_cum[0], beta[0], A[0], dq[0], dk[0], dv[0], dg[0], dbeta[0])
    return dq, dk, dv, dg, dbeta


def chunk_gdr_bwd_pinned_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the SM90 pinned backward, using a zero-padded chunk for a partial tail.
    """
    T, H, HV = k.shape[1], k.shape[2], v.shape[2]
    use_folded_tma = (T, H, HV) == (4096, 32, 96)

    w, u, dv_local = launch_dense_prologue(q, k, v, do, g_cum, beta, A, use_folded_tma)
    h, v_new = launch_dense_state_recompute(k, w, u, g_cum, use_folded_tma)
    # The reverse recurrence reuses u for du, and the last kernel reuses dv_local for dv.
    dh, du = launch_dense_state_gradient(q, k, w, do, dv_local, g_cum, u, use_folded_tma)
    return launch_dense_chunk_gradients(q, k, v, v_new, do, du, h, dh, g_cum, beta, A, dv_local, use_folded_tma)


def has_aligned_tma_inputs(tensors):
    return all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)


def launch_dense_prologue(q, k, v, do, g_cum, beta, A, use_folded_tma):
    if use_folded_tma and has_aligned_tma_inputs((q, k, v, do, A)):
        T, H, HV = k.shape[1], k.shape[2], v.shape[2]
        w = k.new_empty(1, T, HV, HEAD_DIM)
        u, dv_local = torch.empty_like(v), torch.empty_like(v)
        launch = compile_backward_kernel("bwd_prologue_tma", H, HV, k.device.index, False)
        launch(q[0], k[0], v[0], do[0], g_cum[0], beta[0], A[0], w[0], u[0], dv_local[0])
        return w, u, dv_local
    return bwd_prologue(q, k, v, do, g_cum, beta, A)


def launch_dense_state_recompute(k, w, u, g_cum, use_folded_tma):
    if use_folded_tma and has_aligned_tma_inputs((k, w, u, g_cum)):
        T, H, HV = k.shape[1], k.shape[2], u.shape[2]
        h = k.new_empty(1, T // CHUNK_ROWS, HV, HEAD_DIM, HEAD_DIM)
        v_new = torch.empty_like(u)
        launch = compile_backward_kernel("state_recompute_tma", H, HV, k.device.index, False)
        launch(k[0], w[0], u[0], g_cum[0], h[0], v_new[0])
        return h, v_new
    return state_recompute(k, w, u, g_cum)


def launch_dense_state_gradient(q, k, w, do, dv_local, g_cum, u, use_folded_tma):
    if use_folded_tma and has_aligned_tma_inputs((q, k, w, do, dv_local, g_cum)):
        return launch_state_gradient("state_gradient_loader", q, k, w, do, dv_local, g_cum, u, None)
    return state_gradient(q, k, w, do, dv_local, g_cum, du=u)


def launch_dense_chunk_gradients(q, k, v, v_new, do, du, h, dh, g_cum, beta, A, dv_local, use_folded_tma):
    name = "chunk_gradients"
    if use_folded_tma and has_aligned_tma_inputs((du, do, v_new, h, dh, k, q, v, A)):
        name = "chunk_gradients_epilogue"
    return launch_chunk_gradients(name, q, k, v, v_new, do, du, h, dh, g_cum, beta, A, dv_local)
