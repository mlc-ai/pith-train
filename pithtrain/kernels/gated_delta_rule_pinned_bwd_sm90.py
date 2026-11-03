"""
SM90 dense chunked gated delta rule backward (chunk size 64), four CuTe-DSL kernels at every T:
BwdPrologue (w, u from the saved A, and the intra-chunk dv_local), StateRecompute (the chunk states
h and v_new again), StateGradient (the reverse state-gradient recurrence, dh and du) and
ChunkGradients (dq, dk, dv, dg, dbeta per chunk, the WY backward and the in-chunk reverse cumsum of
dg fused in). They follow FLA 0.5.2's chunk algorithm and rounding points: bf16 w, u, h, v_new,
dv_local, dh, du, dw, ds and dA, fp32 accumulation, exp2 of the log2 gate, and a TF32 (q 2^g)^T do
product in the dh update with truncated operands, as FLA's tl.dot on Hopper; dk and dg stay in fp32
where FLA adds them through bf16 / fp32 buffers. When T is not a multiple of the chunk size each
kernel runs its ragged variant, whose last chunk behaves as if zero-padded to 64 rows (copy_rows'
zero-filled loads, gate rows past T reading g_cum[T - 1], stores of rows below T only); the
intermediates keep T rows.
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
from pithtrain.kernels.gated_delta_rule_pinned_fwd_sm90 import BT, HEAD_DIM, STATE_BV, STATE_STAGES, STATE_THREADS, chunk_copies, copy_rows, multiply_a, operand_copies, scale_kv

__all__ = ["BwdPrologue", "ChunkGradients", "StateGradient", "StateRecompute", "bwd_prologue", "chunk_gdr_bwd_pinned_sm90", "chunk_gradients", "state_final", "state_gradient", "state_gradient_initial", "state_recompute"]

PROLOGUE_THREADS = 128
GRAD_BV = 32  # state-gradient columns per CTA
F32_PAD = 4  # fp32 row padding of the TF32 product tiles
GRAD_STAGES = 3  # chunks of k, w, q, g and dv_local / do columns in flight
BAR_CHAIN, BAR_PRODUCER, BAR_READY, BAR_FREE_P, BAR_FREE = 1, 2, 3, 5, 7  # READY, FREE_P: ids +0 / +1 by P buffer; FREE: +0 .. +2 by stage
GRAD_THREADS = 256  # ChunkGradients: 8 warps, 4 along the rows and 2 along the columns of each tile


class BwdPrologue:
    """
    One CTA per (chunk, value head), four warps, everything in shared memory, no cross-chunk state:
        dv_local = bf16(bf16(mask_{i <= j}(k_i.q_j scale 2^(g_j - g_i))) @ do)
        u = bf16(A bf16(v beta)),  w = bf16(A bf16(k beta 2^g))
    from the forward's saved A and g_cum (WyRepresentation's scale and multiply without the solve).
    w, u and dv_local are FLA's recompute_w_u and chunk_bwd_kernel_dv_local bit for bit (w rounds
    the bf16 x bf16 product k beta to bf16, then bf16(that 2^g), as FLA does).

    With ragged (T not a multiple of BT) the last chunk runs as if zero-padded to BT rows: q, k, v,
    do and A rows past T load as zeros, beta as 0 and g_cum as g_cum[T - 1]; only rows below T are
    stored.
    """

    def __init__(self, H: int, HV: int, ragged: bool = False):
        self.H, self.HV, self.ragged = H, HV, ragged
        self.scale = HEAD_DIM ** -0.5

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mDo: cute.Tensor, mGcum: cute.Tensor, mBeta: cute.Tensor, mA: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mDvLocal: cute.Tensor, stream):
        mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        chunk_layout, a_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (BT, HEAD_DIM), (0, 1)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, BT), (BT, BT), (0, 1))
        num_chunks = (mGcum.shape[0] + BT - 1) // BT if const_expr(self.ragged) else mGcum.shape[0] // BT
        self.kernel(mQ, mK, mV, mDo, mGcum, mBeta, mA, mW, mU, mDvLocal, mma, chunk_layout, a_layout).launch(grid=[num_chunks, self.HV, 1], block=[PROLOGUE_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mDo: cute.Tensor, mGcum: cute.Tensor, mBeta: cute.Tensor, mA: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mDvLocal: cute.Tensor, mma: cute.TiledMma, chunk_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, _ = cute.arch.block_idx()
        key_head_idx = head_idx // (self.HV // self.H)
        smem = cutlass.utils.SmemAllocator()
        s_k, s_q, s_v, s_do, s_a, s_gcum, s_beta = smem.allocate_tensor(BFloat16, chunk_layout, 1024), smem.allocate_tensor(BFloat16, chunk_layout, 1024), smem.allocate_tensor(BFloat16, chunk_layout, 1024), smem.allocate_tensor(BFloat16, chunk_layout, 1024), smem.allocate_tensor(BFloat16, a_layout, 1024), smem.allocate_tensor(Float32, cute.make_layout(BT), 16), smem.allocate_tensor(Float32, cute.make_layout(BT), 16)
        load, store_128, _ = chunk_copies(PROLOGUE_THREADS, tidx)
        load_64 = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128), cute.make_ordered_layout((PROLOGUE_THREADS // 8, 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        ld_A, ld_B, _, ldt_B, st_C, _, _ = operand_copies(mma, tidx)
        thr = mma.get_slice(tidx)
        tile = lambda m, head: cute.local_tile(m[None, head, None], (BT, HEAD_DIM), (chunk_idx, 0))
        if const_expr(self.ragged):
            T = mGcum.shape[0]
            valid = T - chunk_idx * BT
            rows_128, rows_64 = load.partition_D(cute.make_identity_tensor((BT, HEAD_DIM))), load_64.partition_D(cute.make_identity_tensor((BT, BT)))
            copy_rows(load, load.partition_S(tile(mK, key_head_idx)), load.partition_D(s_k), rows_128, valid, True)
            copy_rows(load, load.partition_S(tile(mQ, key_head_idx)), load.partition_D(s_q), rows_128, valid, True)
            cute.arch.cp_async_commit_group()
            copy_rows(load, load.partition_S(tile(mDo, head_idx)), load.partition_D(s_do), rows_128, valid, True)
            copy_rows(load, load.partition_S(tile(mV, head_idx)), load.partition_D(s_v), rows_128, valid, True)
            copy_rows(load_64, load_64.partition_S(cute.local_tile(mA[None, head_idx, None], (BT, BT), (chunk_idx, 0))), load_64.partition_D(s_a), rows_64, valid, True)
            cute.arch.cp_async_commit_group()
            if tidx < BT:
                # both loads at the clamped row, issued together; beta past T is then zeroed in registers
                row = chunk_idx * BT + tidx
                clamped = cutlass.min(row, T - 1)
                gate, beta = mGcum[clamped, head_idx], Float32(mBeta[clamped, head_idx])
                if row >= T:
                    beta = Float32(0.0)
                s_gcum[tidx], s_beta[tidx] = gate, beta
        else:
            cute.copy(load, load.partition_S(tile(mK, key_head_idx)), load.partition_D(s_k))
            cute.copy(load, load.partition_S(tile(mQ, key_head_idx)), load.partition_D(s_q))
            cute.arch.cp_async_commit_group()
            cute.copy(load, load.partition_S(tile(mDo, head_idx)), load.partition_D(s_do))
            cute.copy(load, load.partition_S(tile(mV, head_idx)), load.partition_D(s_v))
            cute.copy(load_64, load_64.partition_S(cute.local_tile(mA[None, head_idx, None], (BT, BT), (chunk_idx, 0))), load_64.partition_D(s_a))
            cute.arch.cp_async_commit_group()
            if tidx < BT:
                row = chunk_idx * BT + tidx
                s_gcum[tidx], s_beta[tidx] = mGcum[row, head_idx], Float32(mBeta[row, head_idx])
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()
        # S = k q^T, then the upper-triangular (key i <= query j) decayed and scaled block
        acc_s = cute.make_rmem_tensor(thr.partition_shape_C((BT, BT)), Float32)
        acc_s.fill(0.0)
        gemm(mma, acc_s, thr.make_fragment_A(thr.partition_A(s_k)), thr.make_fragment_B(thr.partition_B(s_q)), ld_A.partition_S(s_k), ld_B.partition_S(s_q), ld_A, ld_B)
        s_coords = thr.partition_C(cute.make_identity_tensor((BT, BT)))
        for elem in cutlass.range_constexpr(cute.size(acc_s)):
            row, col = s_coords[elem]
            value = Float32(0.0)
            if row <= col:
                value = acc_s[elem] * self.scale * cute.math.exp2(s_gcum[col] - s_gcum[row], fastmath=True)
            acc_s[elem] = value
        r_p = cute.make_fragment_like(acc_s, BFloat16)
        r_p.store(acc_s.load().to(BFloat16))
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        s_do_t = cute.composition(s_do, cute.make_ordered_layout((HEAD_DIM, BT), order=(1, 0)))
        acc_dv = cute.make_rmem_tensor(thr.partition_shape_C((BT, HEAD_DIM)), Float32)
        acc_dv.fill(0.0)
        gemm_rs(mma, acc_dv, reshape_acc_to_frgA(r_p), thr.make_fragment_B(thr.partition_B(s_do_t)), ldt_B.partition_S(s_do_t), ldt_B)
        r_dv = cute.make_fragment_like(acc_dv, BFloat16)
        r_dv.store(acc_dv.load().to(BFloat16))
        cute.copy(st_C, st_C.retile(r_dv), st_C.partition_D(s_q))  # q is no longer read
        cute.arch.barrier()
        if const_expr(self.ragged):
            copy_rows(store_128, store_128.partition_S(s_q), store_128.partition_D(tile(mDvLocal, head_idx)), rows_128, valid, False)
        else:
            cute.copy(store_128, store_128.partition_S(s_q), store_128.partition_D(tile(mDvLocal, head_idx)))
        scale_kv(load, s_k, s_v, s_gcum, s_beta, True)
        for which in cutlass.range_constexpr(2):
            s_x = s_v if const_expr(which == 0) else s_k
            mOut = mU if const_expr(which == 0) else mW
            multiply_a(mma, thr, ld_A, ldt_B, s_a, s_x)
            if const_expr(self.ragged):
                copy_rows(store_128, store_128.partition_S(s_x), store_128.partition_D(tile(mOut, head_idx)), rows_128, valid, False)
            else:
                cute.copy(store_128, store_128.partition_S(s_x), store_128.partition_D(tile(mOut, head_idx)))


class StateRecompute:
    """
    The forward's chunk-state recurrence rerun for the backward, one CTA (one warpgroup) per
    (STATE_BV state columns, value head), h the HEAD_DIM x STATE_BV fp32 state in registers, chunk
    by chunk:
        h_n = bf16(h);  v = u_n - w_n h_n;  v_new = bf16(v)
        h = h 2^(g_last) + k_n^T bf16(v 2^(g_last - g))
    h_n is stored V-first, (chunks, HV, V, K), and v_new like u; both bit-identical to FLA's fwd_h
    given the same w and u. Loads of k, w, u and g run STATE_STAGES chunks ahead.

    With ragged (T not a multiple of BT) the last chunk runs as if zero-padded to BT rows: k, w and u
    rows past T load as zeros and its gate rows as g_cum[T - 1]; only v_new's rows below T are
    stored.

    With has_h0 the state starts at h0, fp32 (HV, K, V), instead of zero; with has_ht the final
    state is stored, fp32 (HV, K, V) with any row stride; without store_states neither h_n nor
    v_new is stored (mH and mVnew may be None), the recurrence alone: the context-parallel
    transition's state half (the rank's h from zero to its end, FLA's pre-process h).
    """

    def __init__(self, H: int, HV: int, ragged: bool = False, has_h0: bool = False, has_ht: bool = False, store_states: bool = True):
        self.H, self.HV, self.ragged = H, HV, ragged
        self.has_h0, self.has_ht, self.store_states = has_h0, has_ht, store_states

    @cute.jit
    def __call__(self, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGcum: cute.Tensor, mH: cute.Tensor, mVnew: cute.Tensor, mH0: cute.Tensor, mHt: cute.Tensor, stream):
        mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        layouts = (cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (BT, HEAD_DIM, STATE_STAGES), (0, 1, 2)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_BV), (BT, STATE_BV, STATE_STAGES), (0, 1, 2)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (STATE_BV, HEAD_DIM), (0, 1)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, BT), (STATE_BV, BT), (0, 1)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_BV), (BT, STATE_BV), (0, 1)))
        self.kernel(mK, mW, mU, mGcum, mH, mVnew, mH0, mHt, mma, *layouts).launch(grid=[HEAD_DIM // STATE_BV, self.HV, 1], block=[STATE_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGcum: cute.Tensor, mH: cute.Tensor, mVnew: cute.Tensor, mH0: cute.Tensor, mHt: cute.Tensor, mma: cute.TiledMma, ring_layout: cute.ComposedLayout, u_layout: cute.ComposedLayout, h_layout: cute.ComposedLayout, v_layout: cute.ComposedLayout, o_layout: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        col_tile, head_idx, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s_k, s_w, s_u, s_h, s_vscaled, s_vnew, s_gcum = smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, u_layout, 1024), smem.allocate_tensor(BFloat16, h_layout, 1024), smem.allocate_tensor(BFloat16, v_layout, 1024), smem.allocate_tensor(BFloat16, o_layout, 1024), smem.allocate_tensor(Float32, cute.make_layout((BT, STATE_STAGES), stride=(1, BT)), 16)
        num_chunks = (mGcum.shape[0] + BT - 1) // BT if const_expr(self.ragged) else mGcum.shape[0] // BT
        key_head_idx = head_idx // (self.HV // self.H)
        load, store_128, _ = chunk_copies(STATE_THREADS, tidx)
        atom_load, atom_store = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128), cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        narrow = cute.make_ordered_layout((STATE_THREADS // (STATE_BV // 8), STATE_BV // 8), order=(1, 0))
        load_u = cute.make_tiled_copy_tv(atom_load, narrow, cute.make_layout((1, 8))).get_slice(tidx)
        store_v = cute.make_tiled_copy_tv(atom_store, narrow, cute.make_layout((1, 8))).get_slice(tidx)
        load_gate = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32), cute.make_layout(BT), cute.make_layout(1)).get_slice(tidx % BT)
        gK, gW, gU = cute.local_tile(mK[None, key_head_idx, None], (BT, HEAD_DIM), (None, 0)), cute.local_tile(mW[None, head_idx, None], (BT, HEAD_DIM), (None, 0)), cute.local_tile(mU[None, head_idx, None], (BT, STATE_BV), (None, col_tile))
        if const_expr(self.store_states):
            gVnew = cute.local_tile(mVnew[None, head_idx, None], (BT, STATE_BV), (None, col_tile))
        gGcum = cute.local_tile(mGcum[None, head_idx], (BT,), (None,))
        t_k_src, t_w_src, t_u_src, t_g_src = load.partition_S(gK), load.partition_S(gW), load_u.partition_S(gU), load_gate.partition_S(gGcum)
        t_k_dst, t_w_dst, t_u_dst, t_g_dst = load.partition_D(s_k), load.partition_D(s_w), load_u.partition_D(s_u), load_gate.partition_D(s_gcum)
        if const_expr(self.ragged):
            T = mGcum.shape[0]
            rows_wide, rows_narrow, rows_v = load.partition_D(cute.make_identity_tensor((BT, HEAD_DIM))), load_u.partition_D(cute.make_identity_tensor((BT, STATE_BV))), store_v.partition_D(cute.make_identity_tensor((BT, STATE_BV)))
        for stage in cutlass.range_constexpr(STATE_STAGES - 1):
            if stage < num_chunks:
                if const_expr(self.ragged):
                    self.issue_tail_chunk(load, load_u, load_gate, (t_k_src, t_w_src, t_u_src, t_g_src), (t_k_dst, t_w_dst, t_u_dst, t_g_dst), (rows_wide, rows_narrow), mGcum, s_gcum, head_idx, stage, stage, tidx)
                else:
                    cute.copy(load, t_k_src[None, None, None, stage], t_k_dst[None, None, None, stage])
                    cute.copy(load, t_w_src[None, None, None, stage], t_w_dst[None, None, None, stage])
                    cute.copy(load_u, t_u_src[None, None, None, stage], t_u_dst[None, None, None, stage])
                    if tidx < BT:
                        cute.copy(load_gate, t_g_src[None, None, stage], t_g_dst[None, None, stage])
            cute.arch.cp_async_commit_group()
        thr = mma.get_slice(tidx)
        ld_A, ld_B, ldt_A, _, st_C, st_C_t, ld_C = operand_copies(mma, tidx)
        acc_h = cute.make_rmem_tensor(thr.partition_shape_C((HEAD_DIM, STATE_BV)), Float32)
        acc_h.fill(0.0)
        h_coords = thr.partition_C(cute.make_identity_tensor((HEAD_DIM, STATE_BV)))
        if const_expr(self.has_h0):
            for elem in cutlass.range_constexpr(cute.size(acc_h)):
                acc_h[elem] = mH0[head_idx, h_coords[elem][0], col_tile * STATE_BV + h_coords[elem][1]]
        v_coords = thr.partition_C(cute.make_identity_tensor((BT, STATE_BV)))
        s_h_kv, s_vscaled_t = cute.composition(s_h, cute.make_ordered_layout((HEAD_DIM, STATE_BV), order=(1, 0))), cute.composition(s_vscaled, cute.make_ordered_layout((BT, STATE_BV), order=(1, 0)))
        t_h_store, t_vnew_store = store_128.partition_S(s_h), store_v.partition_S(s_vnew)
        for chunk in cutlass.range(num_chunks, unroll=1):
            stage = chunk % STATE_STAGES
            s_k_n, s_w_n, s_u_n = s_k[None, None, stage], s_w[None, None, stage], s_u[None, None, stage]
            r_h = cute.make_fragment_like(acc_h, BFloat16)
            r_h.store(acc_h.load().to(BFloat16))
            cute.copy(st_C_t, st_C_t.retile(r_h), st_C_t.partition_D(s_h_kv))
            cute.arch.cp_async_wait_group(STATE_STAGES - 2)
            cute.arch.barrier()
            if const_expr(self.store_states):
                cute.copy(store_128, t_h_store, store_128.partition_D(cute.local_tile(mH[chunk, head_idx, None, None], (STATE_BV, HEAD_DIM), (col_tile, 0))))
            acc_v = cute.make_rmem_tensor(thr.partition_shape_C((BT, STATE_BV)), Float32)
            acc_v.fill(0.0)
            gemm(mma, acc_v, thr.make_fragment_A(thr.partition_A(s_w_n)), thr.make_fragment_B(thr.partition_B(s_h)), ld_A.partition_S(s_w_n), ld_B.partition_S(s_h), ld_A, ld_B)
            gate_last = s_gcum[BT - 1, stage]
            r_u = cute.make_fragment_like(acc_v, BFloat16)
            cute.copy(ld_C, ld_C.partition_S(s_u_n), ld_C.retile(r_u))
            v_value = r_u.load().to(Float32) - acc_v.load()
            r_vnew = cute.make_fragment_like(acc_v, BFloat16)
            r_vnew.store(v_value.to(BFloat16))
            acc_v.store(v_value)
            for elem in cutlass.range_constexpr(cute.size(acc_v)):
                acc_v[elem] = acc_v[elem] * cute.math.exp2(gate_last - s_gcum[v_coords[elem][0], stage], fastmath=True)
            r_vscaled = cute.make_fragment_like(acc_v, BFloat16)
            r_vscaled.store(acc_v.load().to(BFloat16))
            cute.copy(st_C, st_C.retile(r_vnew), st_C.partition_D(s_vnew))
            cute.copy(st_C_t, st_C_t.retile(r_vscaled), st_C_t.partition_D(s_vscaled_t))
            decay = cute.math.exp2(gate_last, fastmath=True)
            for elem in cutlass.range_constexpr(cute.size(acc_h)):
                acc_h[elem] = acc_h[elem] * decay
            cute.arch.barrier()
            if const_expr(self.store_states):
                if const_expr(self.ragged):
                    copy_rows(store_v, t_vnew_store, store_v.partition_D(gVnew[None, None, chunk]), rows_v, T - chunk * BT, False)
                else:
                    cute.copy(store_v, t_vnew_store, store_v.partition_D(gVnew[None, None, chunk]))
            s_k_n_t = cute.composition(s_k_n, cute.make_ordered_layout((HEAD_DIM, BT), order=(1, 0)))
            gemm(mma, acc_h, thr.make_fragment_A(thr.partition_A(s_k_n_t)), thr.make_fragment_B(thr.partition_B(s_vscaled)), ldt_A.partition_S(s_k_n_t), ld_B.partition_S(s_vscaled), ldt_A, ld_B)
            upcoming = chunk + STATE_STAGES - 1
            if upcoming < num_chunks:
                next_stage = upcoming % STATE_STAGES
                if const_expr(self.ragged):
                    self.issue_tail_chunk(load, load_u, load_gate, (t_k_src, t_w_src, t_u_src, t_g_src), (t_k_dst, t_w_dst, t_u_dst, t_g_dst), (rows_wide, rows_narrow), mGcum, s_gcum, head_idx, upcoming, next_stage, tidx)
                else:
                    cute.copy(load, t_k_src[None, None, None, upcoming], t_k_dst[None, None, None, next_stage])
                    cute.copy(load, t_w_src[None, None, None, upcoming], t_w_dst[None, None, None, next_stage])
                    cute.copy(load_u, t_u_src[None, None, None, upcoming], t_u_dst[None, None, None, next_stage])
                    if tidx < BT:
                        cute.copy(load_gate, t_g_src[None, None, upcoming], t_g_dst[None, None, next_stage])
            cute.arch.cp_async_commit_group()
            cute.arch.barrier()
        if const_expr(self.has_ht):
            for elem in cutlass.range_constexpr(cute.size(acc_h)):
                mHt[head_idx, h_coords[elem][0], col_tile * STATE_BV + h_coords[elem][1]] = acc_h[elem]

    @cute.jit
    def issue_tail_chunk(self, load: cute.TiledCopy, load_u: cute.TiledCopy, load_gate: cute.TiledCopy, sources: tuple, targets: tuple, rows: tuple, mGcum: cute.Tensor, s_gcum: cute.Tensor, head_idx, chunk, stage, tidx):
        """
        The ragged variant's cp.async of chunk's k, w, u and g into ring stage: rows past T zeroed,
        gate rows past T set to g_cum[T - 1]. The caller commits the group.
        """
        T = mGcum.shape[0]
        valid = T - chunk * BT
        t_k_src, t_w_src, t_u_src, t_g_src = sources
        t_k_dst, t_w_dst, t_u_dst, t_g_dst = targets
        rows_wide, rows_narrow = rows
        copy_rows(load, t_k_src[None, None, None, chunk], t_k_dst[None, None, None, stage], rows_wide, valid, True)
        copy_rows(load, t_w_src[None, None, None, chunk], t_w_dst[None, None, None, stage], rows_wide, valid, True)
        copy_rows(load_u, t_u_src[None, None, None, chunk], t_u_dst[None, None, None, stage], rows_narrow, valid, True)
        if tidx < BT:
            if tidx < valid:
                cute.copy(load_gate, t_g_src[None, None, chunk], t_g_dst[None, None, stage])
            else:
                s_gcum[tidx, stage] = mGcum[T - 1, head_idx]


class StateGradient:
    """
    The reverse state-gradient recurrence, one CTA per (GRAD_BV columns, value head), two
    warpgroups. The chain warpgroup, chunk n = NT - 1 .. 0, with dh the HEAD_DIM x GRAD_BV fp32
    gradient of the state in registers (zero after the last chunk):
        dh_n = bf16(dh), stored V-first like h;  du = bf16(dv_local + (k_n dh_n) 2^(g_last - g))
        dh = dh 2^(g_last) + (P_n - w_n^T du)
    The producer warpgroup loads chunk n's k, w, q, g and dv_local / do columns into a GRAD_STAGES
    ring (two chunks ahead of their use) and computes P_n = scale tf32((q_n 2^g)^T) tf32(do_n), the
    product that does not depend on dh (TF32 operands truncated, FLA's tl.dot on Hopper), off the
    chain into a double buffer. READY[n % 2] says chunk n's stage and P are ready, FREE_P[n % 2]
    that the chain warpgroup has read P_n, FREE[stage] that it is done with that stage. P_n is
    computed as soon as chunk n has landed, before the producer waits to refill the stage of the
    previous chunk.

    With ragged (T not a multiple of BT) the last chunk, the first one processed, runs as if
    zero-padded to BT rows: k, w, q, dv_local and do rows past T load as zeros and its gate rows as
    g_cum[T - 1]; only du's rows below T are stored. The barrier sequence is the aligned one.

    With has_dht dh starts at dht, fp32 (HV, K, V), the gradient of the final state, instead of
    zero; with has_dh0 the gradient of the initial state (dh after chunk 0) is stored, fp32 (HV, K,
    V) with any row stride; without store_states neither dh_n nor du is stored (mDh and mDu may be
    None): the context-parallel transition's state half (FLA's pre-process dh).
    """

    def __init__(self, H: int, HV: int, ragged: bool = False, has_dht: bool = False, has_dh0: bool = False, store_states: bool = True):
        self.H, self.HV, self.ragged = H, HV, ragged
        self.has_dht, self.has_dh0, self.store_states = has_dht, has_dh0, store_states
        self.scale = HEAD_DIM ** -0.5

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mDo: cute.Tensor, mDvLocal: cute.Tensor, mGcum: cute.Tensor, mDh: cute.Tensor, mDu: cute.Tensor, mDht: cute.Tensor, mDh0: cute.Tensor, stream):
        mma, mma_tf32 = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16)), cute.make_tiled_mma(cute.nvgpu.warp.MmaTF32Op((16, 8, 8)), (4, 1, 1), permutation_mnk=(64, 8, 8))
        layouts = (cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (BT, HEAD_DIM, GRAD_STAGES), (0, 1, 2)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, GRAD_BV), (BT, GRAD_BV, GRAD_STAGES), (0, 1, 2)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (GRAD_BV, HEAD_DIM), (0, 1)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, BT), (GRAD_BV, BT), (0, 1)), cute.tile_to_shape(get_smem_layout_atom(BFloat16, GRAD_BV), (BT, GRAD_BV), (0, 1)))
        self.kernel(mQ, mK, mW, mDo, mDvLocal, mGcum, mDh, mDu, mDht, mDh0, mma, mma_tf32, *layouts).launch(grid=[HEAD_DIM // GRAD_BV, self.HV, 1], block=[2 * STATE_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mDo: cute.Tensor, mDvLocal: cute.Tensor, mGcum: cute.Tensor, mDh: cute.Tensor, mDu: cute.Tensor, mDht: cute.Tensor, mDh0: cute.Tensor, mma: cute.TiledMma, mma_tf32: cute.TiledMma, ring_layout: cute.ComposedLayout, cols_layout: cute.ComposedLayout, dh_layout: cute.ComposedLayout, du_t_layout: cute.ComposedLayout, du_layout: cute.ComposedLayout):
        tid, _, _ = cute.arch.thread_idx()
        col_tile, head_idx, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s_k, s_w, s_q, s_dvl, s_do, s_dh, s_du_t, s_du, s_gcum, s_p = smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, cols_layout, 1024), smem.allocate_tensor(BFloat16, cols_layout, 1024), smem.allocate_tensor(BFloat16, dh_layout, 1024), smem.allocate_tensor(BFloat16, du_t_layout, 1024), smem.allocate_tensor(BFloat16, du_layout, 1024), smem.allocate_tensor(Float32, cute.make_layout((BT, GRAD_STAGES), stride=(1, BT)), 16), smem.allocate_tensor(Float32, cute.make_layout((HEAD_DIM, GRAD_BV, 2), stride=(GRAD_BV + F32_PAD, 1, HEAD_DIM * (GRAD_BV + F32_PAD))), 16)
        s_decay = smem.allocate_tensor(Float32, cute.make_layout(BT), 16)  # 2^g of the producer's chunk
        buffers = (s_k, s_w, s_q, s_dvl, s_do, s_dh, s_du_t, s_du, s_gcum, s_p, s_decay)
        if tid < STATE_THREADS:
            self.chain_loop(mDh, mDu, mDht, mDh0, mma, buffers, tid, col_tile, head_idx, (mGcum.shape[0] + BT - 1) // BT if const_expr(self.ragged) else mGcum.shape[0] // BT)
        else:
            self.producer_loop(mQ, mK, mW, mDo, mDvLocal, mGcum, mma_tf32, buffers, tid - STATE_THREADS, col_tile, head_idx)

    @cute.jit
    def issue_chunk(self, copies: tuple, sources: tuple, targets: tuple, chunk, buffer, tidx, tail: tuple):
        """
        cp.async of chunk's k, w, q, dv_local and do columns and g into ring stage buffer, one
        commit group. With ragged, tail is (rows of load's vectors, rows of load_cols's, g_cum, its
        head, s_gcum) for the chunk cut by T.
        """
        load, load_cols, load_gate = copies
        if const_expr(self.ragged):
            rows_wide, rows_cols, mGcum, head_idx, s_gcum = tail
            T = mGcum.shape[0]
            valid = T - chunk * BT
            for i in cutlass.range_constexpr(3):
                copy_rows(load, sources[i][None, None, None, chunk], targets[i][None, None, None, buffer], rows_wide, valid, True)
            for i in cutlass.range_constexpr(3, 5):
                copy_rows(load_cols, sources[i][None, None, None, chunk], targets[i][None, None, None, buffer], rows_cols, valid, True)
            if tidx < BT:
                if tidx < valid:
                    cute.copy(load_gate, sources[5][None, None, chunk], targets[5][None, None, buffer])
                else:
                    s_gcum[tidx, buffer] = mGcum[T - 1, head_idx]
        else:
            for i in cutlass.range_constexpr(3):
                cute.copy(load, sources[i][None, None, None, chunk], targets[i][None, None, None, buffer])
            for i in cutlass.range_constexpr(3, 5):
                cute.copy(load_cols, sources[i][None, None, None, chunk], targets[i][None, None, None, buffer])
            if tidx < BT:
                cute.copy(load_gate, sources[5][None, None, chunk], targets[5][None, None, buffer])
        cute.arch.cp_async_commit_group()

    @cute.jit
    def producer_loop(self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mDo: cute.Tensor, mDvLocal: cute.Tensor, mGcum: cute.Tensor, mma_tf32: cute.TiledMma, buffers: tuple, tidx, col_tile, head_idx):
        s_k, s_w, s_q, s_dvl, s_do, _, _, _, s_gcum, s_p, s_decay = buffers
        num_chunks = (mGcum.shape[0] + BT - 1) // BT if const_expr(self.ragged) else mGcum.shape[0] // BT
        key_head_idx = head_idx // (self.HV // self.H)
        threads = STATE_THREADS
        load, _, _ = chunk_copies(threads, tidx)
        atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
        load_cols = cute.make_tiled_copy_tv(atom_load, cute.make_ordered_layout((threads // (GRAD_BV // 8), GRAD_BV // 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        load_gate = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32), cute.make_layout(BT), cute.make_layout(1)).get_slice(tidx % BT)
        gK, gW, gQ, gDvl, gDo, gGcum = cute.local_tile(mK[None, key_head_idx, None], (BT, HEAD_DIM), (None, 0)), cute.local_tile(mW[None, head_idx, None], (BT, HEAD_DIM), (None, 0)), cute.local_tile(mQ[None, key_head_idx, None], (BT, HEAD_DIM), (None, 0)), cute.local_tile(mDvLocal[None, head_idx, None], (BT, GRAD_BV), (None, col_tile)), cute.local_tile(mDo[None, head_idx, None], (BT, GRAD_BV), (None, col_tile)), cute.local_tile(mGcum[None, head_idx], (BT,), (None,))
        sources = (load.partition_S(gK), load.partition_S(gW), load.partition_S(gQ), load_cols.partition_S(gDvl), load_cols.partition_S(gDo), load_gate.partition_S(gGcum))
        targets = (load.partition_D(s_k), load.partition_D(s_w), load.partition_D(s_q), load_cols.partition_D(s_dvl), load_cols.partition_D(s_do), load_gate.partition_D(s_gcum))
        copies = (load, load_cols, load_gate)
        tail = (load.partition_D(cute.make_identity_tensor((BT, HEAD_DIM))), load_cols.partition_D(cute.make_identity_tensor((BT, GRAD_BV))), mGcum, head_idx, s_gcum) if const_expr(self.ragged) else None
        thr = mma_tf32.get_slice(tidx)
        a_coords = thr.partition_A(cute.make_identity_tensor((HEAD_DIM, BT)))
        r_a, r_b = cute.make_rmem_tensor(thr.partition_shape_A((HEAD_DIM, BT)), Float32), cute.make_rmem_tensor(thr.partition_shape_B((GRAD_BV, BT)), Float32)
        # the TF32 MMA reads the fp32 bits as TF32 (the low mantissa bits dropped), as FLA's tl.dot does
        r_a_tf32, r_b_tf32 = cute.make_tensor(cute.recast_ptr(r_a.iterator, dtype=cutlass.TFloat32), r_a.layout), cute.make_tensor(cute.recast_ptr(r_b.iterator, dtype=cutlass.TFloat32), r_b.layout)
        # an empty commit group stands in for a missing chunk, so the group count stays fixed
        for step in cutlass.range_constexpr(GRAD_STAGES - 1):
            if step < num_chunks:
                self.issue_chunk(copies, sources, targets, num_chunks - 1 - step, step, tidx, tail)
            else:
                cute.arch.cp_async_commit_group()
        for step in cutlass.range(num_chunks, unroll=1):
            buffer, parity = step % GRAD_STAGES, step % 2
            cute.arch.cp_async_wait_group(GRAD_STAGES - 2)
            cute.arch.barrier(barrier_id=BAR_PRODUCER, number_of_threads=threads)
            if tidx < BT:
                s_decay[tidx] = cute.math.exp2(s_gcum[tidx, buffer], fastmath=True)
            cute.arch.barrier(barrier_id=BAR_PRODUCER, number_of_threads=threads)
            # A = (q 2^g)^T and B = do from the bf16 tiles, through their transposed views
            t_a = thr.partition_A(cute.composition(s_q[None, None, buffer], cute.make_ordered_layout((HEAD_DIM, BT), order=(1, 0))))
            t_b = thr.partition_B(cute.composition(s_do[None, None, buffer], cute.make_ordered_layout((GRAD_BV, BT), order=(1, 0))))
            for elem in cutlass.range_constexpr(cute.size(r_a)):
                r_a[elem] = Float32(t_a[elem]) * s_decay[a_coords[elem][1]]
            for elem in cutlass.range_constexpr(cute.size(r_b)):
                r_b[elem] = Float32(t_b[elem])
            acc_p = cute.make_rmem_tensor(thr.partition_shape_C((HEAD_DIM, GRAD_BV)), Float32)
            acc_p.fill(0.0)
            cute.gemm(mma_tf32, acc_p, r_a_tf32, r_b_tf32, acc_p)
            # P_n's buffer held P of two chunks before
            if step >= 2:
                cute.arch.barrier(barrier_id=BAR_FREE_P + parity, number_of_threads=2 * threads)
            t_p = thr.partition_C(s_p[None, None, parity])
            for elem in cutlass.range_constexpr(cute.size(acc_p)):
                t_p[elem] = acc_p[elem] * self.scale
            cute.arch.barrier_arrive(barrier_id=BAR_READY + parity, number_of_threads=2 * threads)
            # chunk step + 2 goes into the stage of step - 1 once the chain warpgroup released it
            upcoming = step + GRAD_STAGES - 1
            if upcoming < num_chunks:
                if step >= 1:
                    cute.arch.barrier(barrier_id=BAR_FREE + upcoming % GRAD_STAGES, number_of_threads=2 * threads)
                self.issue_chunk(copies, sources, targets, num_chunks - 1 - upcoming, upcoming % GRAD_STAGES, tidx, tail)
            else:
                cute.arch.cp_async_commit_group()

    @cute.jit
    def chain_loop(self, mDh: cute.Tensor, mDu: cute.Tensor, mDht: cute.Tensor, mDh0: cute.Tensor, mma: cute.TiledMma, buffers: tuple, tidx, col_tile, head_idx, num_chunks):
        s_k, s_w, _, s_dvl, _, s_dh, s_du_t, s_du, s_gcum, s_p, _ = buffers
        threads = STATE_THREADS
        _, store_128, _ = chunk_copies(threads, tidx)
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        store_cols = cute.make_tiled_copy_tv(atom_store, cute.make_ordered_layout((threads // (GRAD_BV // 8), GRAD_BV // 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        thr = mma.get_slice(tidx)
        ld_A, ld_B, ldt_A, _, st_C, st_C_t, ld_C = operand_copies(mma, tidx)
        if const_expr(self.store_states):
            gDu = cute.local_tile(mDu[None, head_idx, None], (BT, GRAD_BV), (None, col_tile))
        acc_dh = cute.make_rmem_tensor(thr.partition_shape_C((HEAD_DIM, GRAD_BV)), Float32)
        acc_dh.fill(0.0)
        dh_coords = thr.partition_C(cute.make_identity_tensor((HEAD_DIM, GRAD_BV)))
        if const_expr(self.has_dht):
            for elem in cutlass.range_constexpr(cute.size(acc_dh)):
                acc_dh[elem] = mDht[head_idx, dh_coords[elem][0], col_tile * GRAD_BV + dh_coords[elem][1]]
        u_coords = thr.partition_C(cute.make_identity_tensor((BT, GRAD_BV)))
        s_dh_kv, s_du_tv = cute.composition(s_dh, cute.make_ordered_layout((HEAD_DIM, GRAD_BV), order=(1, 0))), cute.composition(s_du_t, cute.make_ordered_layout((BT, GRAD_BV), order=(1, 0)))
        t_dh_store, t_du_store = store_128.partition_S(s_dh), store_cols.partition_S(s_du)
        if const_expr(self.ragged and self.store_states):
            T, rows_du = mDu.shape[0], store_cols.partition_D(cute.make_identity_tensor((BT, GRAD_BV)))
        for step in cutlass.range(num_chunks, unroll=1):
            chunk, buffer, parity = num_chunks - 1 - step, step % GRAD_STAGES, step % 2
            s_k_n, s_w_n, s_dvl_n = s_k[None, None, buffer], s_w[None, None, buffer], s_dvl[None, None, buffer]
            r_dh = cute.make_fragment_like(acc_dh, BFloat16)
            r_dh.store(acc_dh.load().to(BFloat16))
            cute.copy(st_C_t, st_C_t.retile(r_dh), st_C_t.partition_D(s_dh_kv))
            cute.arch.barrier(barrier_id=BAR_CHAIN, number_of_threads=threads)
            if const_expr(self.store_states):
                cute.copy(store_128, t_dh_store, store_128.partition_D(cute.local_tile(mDh[chunk, head_idx, None, None], (GRAD_BV, HEAD_DIM), (col_tile, 0))))
            cute.arch.barrier(barrier_id=BAR_READY + parity, number_of_threads=2 * threads)
            acc_du = cute.make_rmem_tensor(thr.partition_shape_C((BT, GRAD_BV)), Float32)
            acc_du.fill(0.0)
            gemm(mma, acc_du, thr.make_fragment_A(thr.partition_A(s_k_n)), thr.make_fragment_B(thr.partition_B(s_dh)), ld_A.partition_S(s_k_n), ld_B.partition_S(s_dh), ld_A, ld_B)
            gate_last = s_gcum[BT - 1, buffer]
            r_dvl = cute.make_fragment_like(acc_du, BFloat16)
            cute.copy(ld_C, ld_C.partition_S(s_dvl_n), ld_C.retile(r_dvl))
            dvl = r_dvl.load().to(Float32)
            for elem in cutlass.range_constexpr(cute.size(acc_du)):
                acc_du[elem] = acc_du[elem] * cute.math.exp2(gate_last - s_gcum[u_coords[elem][0], buffer], fastmath=True)
            acc_du.store(acc_du.load() + dvl)
            r_du = cute.make_fragment_like(acc_du, BFloat16)
            r_du.store(acc_du.load().to(BFloat16))
            r_du_neg = cute.make_fragment_like(acc_du, BFloat16)
            r_du_neg.store((-(r_du.load().to(Float32))).to(BFloat16))
            cute.copy(st_C, st_C.retile(r_du), st_C.partition_D(s_du))
            cute.copy(st_C_t, st_C_t.retile(r_du_neg), st_C_t.partition_D(s_du_tv))
            cute.arch.barrier(barrier_id=BAR_CHAIN, number_of_threads=threads)
            if const_expr(self.store_states):
                if const_expr(self.ragged):
                    copy_rows(store_cols, t_du_store, store_cols.partition_D(gDu[None, None, chunk]), rows_du, T - chunk * BT, False)
                else:
                    cute.copy(store_cols, t_du_store, store_cols.partition_D(gDu[None, None, chunk]))
            s_w_n_t = cute.composition(s_w_n, cute.make_ordered_layout((HEAD_DIM, BT), order=(1, 0)))
            acc_wdu = cute.make_rmem_tensor(thr.partition_shape_C((HEAD_DIM, GRAD_BV)), Float32)
            acc_wdu.fill(0.0)
            gemm(mma, acc_wdu, thr.make_fragment_A(thr.partition_A(s_w_n_t)), thr.make_fragment_B(thr.partition_B(s_du_t)), ldt_A.partition_S(s_w_n_t), ld_B.partition_S(s_du_t), ldt_A, ld_B)
            decay = cute.math.exp2(gate_last, fastmath=True)
            t_p = thr.partition_C(s_p[None, None, parity])
            for elem in cutlass.range_constexpr(cute.size(acc_dh)):
                acc_dh[elem] = acc_dh[elem] * decay + (t_p[elem] + acc_wdu[elem])
            if step + 2 < num_chunks:
                cute.arch.barrier_arrive(barrier_id=BAR_FREE_P + parity, number_of_threads=2 * threads)
            # the producer waits for this stage only if it loads another chunk into it
            if step + GRAD_STAGES < num_chunks:
                cute.arch.barrier_arrive(barrier_id=BAR_FREE + buffer, number_of_threads=2 * threads)
        if const_expr(self.has_dh0):
            for elem in cutlass.range_constexpr(cute.size(acc_dh)):
                mDh0[head_idx, dh_coords[elem][0], col_tile * GRAD_BV + dh_coords[elem][1]] = acc_dh[elem]


class ChunkGradients:
    """
    One CTA per (chunk, value head), two warpgroups splitting the columns of every product
    (warpgroup MMAs, operands in shared memory), every gradient of the chunk from the recomputed h,
    v_new and the state gradients dh, du, in four steps through shared memory (bf16 operands, fp32
    accumulators):
      1. dw = -bf16(du h_n^T);  dq = do h_n^T;  dk = v_new dh_n^T;  ds = do v_new^T;
         sum(h_n * dh_n)
      2. dq = dq 2^g scale + bf16(mask_{i >= j}(ds 2^(g_i - g_j)) scale) k;
         dk = dk 2^(g_last - g) + ds^T q
         dg_last = sum(h_n * dh_n) 2^(g_last) + sum(dk 2^(g_last - g) * k), on the last row only
         dg = sum(dq q) - sum(dk k) per row;  dq stored
      3. the WY backward through the saved A: dA = dw bf16(k beta 2^g)^T + du bf16(v beta)^T;
         dk += (A^T dw) beta 2^g;  dv = bf16((A^T du) beta);  dA = bf16(mask_{i > j}(-A^T
         bf16(bf16(dA) A^T) 2^(g_i - g_j)));  dk += (dA k) beta + dA^T bf16(k beta);  dbeta and dg
         from the same terms
      4. dk stored; dg = reverse in-chunk cumsum, stored with dbeta
    Row sums are reduced in a fixed order (per thread, quad shuffles, then the two column warps), so
    the result is deterministic. dq, dk per value head; dg with respect to the raw gate.

    With ragged (T not a multiple of BT) the last chunk runs as if zero-padded to BT rows: every
    row-indexed input past T loads as zeros (beta as 0, g_cum as g_cum[T - 1]), so every padded row
    term is an exact zero and the padded rows' gate sums are zero before the reverse cumsum, which
    adds dg_last on row BT - 1 as the aligned kernel does; only rows below T are stored.
    """

    def __init__(self, H: int, HV: int, ragged: bool = False):
        self.H, self.HV, self.ragged = H, HV, ragged
        self.scale = HEAD_DIM ** -0.5

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mVnew: cute.Tensor, mDo: cute.Tensor, mDu: cute.Tensor, mH: cute.Tensor, mDh: cute.Tensor, mGcum: cute.Tensor, mBeta: cute.Tensor, mA: cute.Tensor, mDq: cute.Tensor, mDk: cute.Tensor, mDv: cute.Tensor, mDg: cute.Tensor, mDbeta: cute.Tensor, stream):
        # warpgroup MMAs, the two warpgroups splitting N: (A major, B major) x (N 64 for the (64, 128)
        # products, N 32 for the (64, 64) ones); every operand stays in shared memory
        mmas = tuple(sm90_utils.make_tiled_mma(BFloat16, a_major, b_major, tile_n, "SS", atom_layout_mnk=(1, 2, 1)) for a_major, b_major, tile_n in (("K", "MN", HEAD_DIM // 2), ("MN", "MN", HEAD_DIM // 2), ("K", "K", BT // 2), ("MN", "MN", BT // 2)))
        # row-major tiles whose swizzle atoms also serve their transposed (MN-major) use per warpgroup
        chunk_layout, state_layout, square_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (BT, HEAD_DIM), major_mode_size=HEAD_DIM // 2), sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (HEAD_DIM, HEAD_DIM), major_mode_size=HEAD_DIM // 2), sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (BT, BT), major_mode_size=BT // 2)
        num_chunks = (mGcum.shape[0] + BT - 1) // BT if const_expr(self.ragged) else mGcum.shape[0] // BT
        self.kernel(mQ, mK, mV, mVnew, mDo, mDu, mH, mDh, mGcum, mBeta, mA, mDq, mDk, mDv, mDg, mDbeta, mmas, chunk_layout, state_layout, square_layout).launch(grid=[num_chunks, self.HV, 1], block=[GRAD_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mVnew: cute.Tensor, mDo: cute.Tensor, mDu: cute.Tensor, mH: cute.Tensor, mDh: cute.Tensor, mGcum: cute.Tensor, mBeta: cute.Tensor, mA: cute.Tensor, mDq: cute.Tensor, mDk: cute.Tensor, mDv: cute.Tensor, mDg: cute.Tensor, mDbeta: cute.Tensor, mmas: tuple, chunk_layout: cute.ComposedLayout, state_layout: cute.ComposedLayout, square_layout: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, _ = cute.arch.block_idx()
        key_head_idx = head_idx // (self.HV // self.H)
        warp_idx, lane = tidx // 32, tidx % 32
        smem = cutlass.utils.SmemAllocator()
        s_q, s_k, s_v, s_vnew, s_do, s_du, s_h, s_dh, s_dw, s_a, s_sq, s_gcum, s_beta = smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner), smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner), smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner), smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner), smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner), smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner), smem.allocate_tensor(BFloat16, state_layout.outer, 1024, swizzle=state_layout.inner), smem.allocate_tensor(BFloat16, state_layout.outer, 1024, swizzle=state_layout.inner), smem.allocate_tensor(BFloat16, chunk_layout.outer, 1024, swizzle=chunk_layout.inner), smem.allocate_tensor(BFloat16, square_layout.outer, 1024, swizzle=square_layout.inner), smem.allocate_tensor(BFloat16, square_layout.outer, 1024, swizzle=square_layout.inner), smem.allocate_tensor(Float32, cute.make_layout(BT), 16), smem.allocate_tensor(Float32, cute.make_layout(BT), 16)
        s_red = smem.allocate_tensor(Float32, cute.make_layout((BT, 2, 2), stride=(1, BT, 2 * BT)), 16)  # (row, column warp, quantity)
        s_warp = smem.allocate_tensor(Float32, cute.make_layout(GRAD_THREADS // 32), 16)
        s_col = smem.allocate_tensor(Float32, cute.make_layout((4, BT), stride=(BT, 1)), 16)  # (row warp, column)
        # the WY operands live where h and dh were, once step 1 is done with them
        s_kbg, s_kb, s_vb = cute.make_tensor(s_h.iterator, chunk_layout.outer), cute.make_tensor(s_h.iterator + BT * HEAD_DIM, chunk_layout.outer), cute.make_tensor(s_dh.iterator, chunk_layout.outer)
        load, store_128, _ = chunk_copies(GRAD_THREADS, tidx)
        load_64 = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128), cute.make_ordered_layout((GRAD_THREADS // 8, 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        mma_wide, mma_wide_t, mma_square, mma_square_t = mmas
        warp_group = cute.make_layout(GRAD_THREADS // 128, stride=128)(cute.arch.make_warp_uniform(tidx // 128))
        thr, thr_square = mma_wide.get_slice(tidx), mma_square.get_slice(tidx)
        st_C, ld_C = (cute.make_tiled_copy_C(atom, mma_wide).get_slice(tidx) for atom in (cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16), cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)))
        st_C_square = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16), mma_square).get_slice(tidx)

        def product(acc: cute.Tensor, s_a: cute.Tensor, s_b: cute.Tensor, a_transposed: bool, b_transposed: bool):
            """
            acc += s_a s_b^T on both warpgroups (s_a (M, K), s_b (N, K)); a transposed operand is a
            view of a row-major tile, MN-major for the MMA. Reads shared memory that every thread
            fenced.
            """
            wide = const_expr(cute.size(acc) == BT * HEAD_DIM // GRAD_THREADS)
            # the products here: wide ones (K, MN) or (MN, MN), square ones (K, K) or (MN, MN)
            tiled = (mma_wide_t if a_transposed else mma_wide) if wide else (mma_square_t if a_transposed else mma_square)
            wg = tiled.get_slice(warp_group)
            sm90_utils.gemm(tiled, acc, wg.make_fragment_A(wg.partition_A(s_a)), wg.make_fragment_B(wg.partition_B(s_b)), zero_init=False, wg_wait=0)

        def sync():
            """
            Shared-memory writes (cp.async, st.shared) visible to every thread and to the warpgroup
            MMAs.
            """
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()
        tile = lambda m, head: cute.local_tile(m[None, head, None], (BT, HEAD_DIM), (chunk_idx, 0))
        state_tile = lambda m: m[chunk_idx, head_idx, None, None]
        transposed = lambda s, rows, cols: cute.composition(s, cute.make_ordered_layout((rows, cols), order=(1, 0)))

        def c_values(s_x: cute.Tensor) -> cute.Tensor:
            """
            s_x (BT, HEAD_DIM) at the accumulators' coordinates, loaded where a row sum needs it.
            """
            r_x = cute.make_rmem_tensor(thr.partition_shape_C((BT, HEAD_DIM)), BFloat16)
            cute.copy(ld_C, ld_C.partition_S(s_x), ld_C.retile(r_x))
            return r_x
        # loads: what step 1 reads first, then the rest
        if const_expr(self.ragged):
            T = mGcum.shape[0]
            valid = T - chunk_idx * BT
            rows_128, rows_64 = load.partition_D(cute.make_identity_tensor((BT, HEAD_DIM))), load_64.partition_D(cute.make_identity_tensor((BT, BT)))
            copy_rows(load, load.partition_S(tile(mDu, head_idx)), load.partition_D(s_du), rows_128, valid, True)
            cute.copy(load, load.partition_S(state_tile(mH)), load.partition_D(s_h))
            copy_rows(load, load.partition_S(tile(mDo, head_idx)), load.partition_D(s_do), rows_128, valid, True)
            copy_rows(load, load.partition_S(tile(mVnew, head_idx)), load.partition_D(s_vnew), rows_128, valid, True)
            cute.copy(load, load.partition_S(state_tile(mDh)), load.partition_D(s_dh))
            cute.arch.cp_async_commit_group()
            copy_rows(load, load.partition_S(tile(mK, key_head_idx)), load.partition_D(s_k), rows_128, valid, True)
            copy_rows(load, load.partition_S(tile(mQ, key_head_idx)), load.partition_D(s_q), rows_128, valid, True)
            copy_rows(load, load.partition_S(tile(mV, head_idx)), load.partition_D(s_v), rows_128, valid, True)
            copy_rows(load_64, load_64.partition_S(cute.local_tile(mA[None, head_idx, None], (BT, BT), (chunk_idx, 0))), load_64.partition_D(s_a), rows_64, valid, True)
            cute.arch.cp_async_commit_group()
            if tidx < BT:
                # both loads at the clamped row, issued together; beta past T is then zeroed in registers
                row = chunk_idx * BT + tidx
                clamped = cutlass.min(row, T - 1)
                gate, beta = mGcum[clamped, head_idx], Float32(mBeta[clamped, head_idx])
                if row >= T:
                    beta = Float32(0.0)
                s_gcum[tidx], s_beta[tidx] = gate, beta
        else:
            cute.copy(load, load.partition_S(tile(mDu, head_idx)), load.partition_D(s_du))
            cute.copy(load, load.partition_S(state_tile(mH)), load.partition_D(s_h))
            cute.copy(load, load.partition_S(tile(mDo, head_idx)), load.partition_D(s_do))
            cute.copy(load, load.partition_S(tile(mVnew, head_idx)), load.partition_D(s_vnew))
            cute.copy(load, load.partition_S(state_tile(mDh)), load.partition_D(s_dh))
            cute.arch.cp_async_commit_group()
            cute.copy(load, load.partition_S(tile(mK, key_head_idx)), load.partition_D(s_k))
            cute.copy(load, load.partition_S(tile(mQ, key_head_idx)), load.partition_D(s_q))
            cute.copy(load, load.partition_S(tile(mV, head_idx)), load.partition_D(s_v))
            cute.copy(load_64, load_64.partition_S(cute.local_tile(mA[None, head_idx, None], (BT, BT), (chunk_idx, 0))), load_64.partition_D(s_a))
            cute.arch.cp_async_commit_group()
            if tidx < BT:
                row = chunk_idx * BT + tidx
                s_gcum[tidx], s_beta[tidx] = mGcum[row, head_idx], Float32(mBeta[row, head_idx])
        cute.arch.cp_async_wait_group(1)
        sync()
        c_coords = thr.partition_C(cute.make_identity_tensor((BT, HEAD_DIM)))
        sq_coords = thr_square.partition_C(cute.make_identity_tensor((BT, BT)))
        gate_last = s_gcum[BT - 1]
        dg_first, dg_second, db_first, db_second = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        # each thread holds two rows of every 64-row accumulator: their gate factors, once
        first_row = c_coords[0][0]
        gate_first, gate_second = s_gcum[first_row], s_gcum[first_row + 8]
        beta_first, beta_second = s_beta[first_row], s_beta[first_row + 8]
        decay_first, decay_second, tail_first, tail_second = cute.math.exp2(gate_first, fastmath=True), cute.math.exp2(gate_second, fastmath=True), cute.math.exp2(gate_last - gate_first, fastmath=True), cute.math.exp2(gate_last - gate_second, fastmath=True)
        # step 1: dw first (its accumulator is stored right away), then dq, dk, ds
        s_h_t, s_dh_t = transposed(s_h, HEAD_DIM, HEAD_DIM), transposed(s_dh, HEAD_DIM, HEAD_DIM)
        acc = cute.make_rmem_tensor(thr.partition_shape_C((BT, HEAD_DIM)), Float32)
        acc.fill(0.0)
        product(acc, s_du, s_h_t, False, True)
        r_out = cute.make_fragment_like(acc, BFloat16)
        r_out.store((-(acc.load().to(BFloat16).to(Float32))).to(BFloat16))
        cute.copy(st_C, st_C.retile(r_out), st_C.partition_D(s_dw))
        acc_dq = cute.make_rmem_tensor(thr.partition_shape_C((BT, HEAD_DIM)), Float32)
        acc_dq.fill(0.0)
        product(acc_dq, s_do, s_h_t, False, True)
        acc_ds = cute.make_rmem_tensor(thr_square.partition_shape_C((BT, BT)), Float32)
        acc_ds.fill(0.0)
        product(acc_ds, s_do, s_vnew, False, False)
        acc_dk = cute.make_rmem_tensor(thr.partition_shape_C((BT, HEAD_DIM)), Float32)
        acc_dk.fill(0.0)
        product(acc_dk, s_vnew, s_dh_t, False, True)
        t_h, t_dh = load.partition_D(s_h), load.partition_D(s_dh)
        state_sum = Float32(0.0)
        for i in cutlass.range_constexpr(cute.size(t_h, mode=[1])):
            for j in cutlass.range_constexpr(cute.size(t_h, mode=[2])):
                pairs = t_h[None, i, j].load().to(Float32) * t_dh[None, i, j].load().to(Float32)
                for e in cutlass.range_constexpr(8):
                    state_sum = state_sum + pairs[e]
        # step 2: gates, ds, dq, dk and the gate gradient's state terms
        cute.arch.cp_async_wait_group(0)
        sync()
        key_sum = Float32(0.0)
        r_k_c = c_values(s_k)
        for elem in cutlass.range_constexpr(cute.size(acc_dq)):
            decay, tail = (decay_second, tail_second) if elem % 4 >= 2 else (decay_first, tail_first)
            acc_dq[elem], acc_dk[elem] = acc_dq[elem] * decay * self.scale, acc_dk[elem] * tail
            key_sum = key_sum + acc_dk[elem] * Float32(r_k_c[elem])
        for elem in cutlass.range_constexpr(cute.size(acc_ds)):
            row, col = sq_coords[elem]
            value = Float32(0.0)
            if row >= col:
                value = acc_ds[elem] * cute.math.exp2(s_gcum[row] - s_gcum[col], fastmath=True) * self.scale
            acc_ds[elem] = value
        r_ds = cute.make_fragment_like(acc_ds, BFloat16)
        r_ds.store(acc_ds.load().to(BFloat16))
        cute.copy(st_C_square, st_C_square.retile(r_ds), st_C_square.partition_D(s_sq))
        # dg_last: a block sum in a fixed order
        partial = state_sum
        for offset in cutlass.range_constexpr(5):
            partial = partial + cute.arch.shuffle_sync_bfly(partial, offset=1 << (4 - offset))
        partial_keys = key_sum
        for offset in cutlass.range_constexpr(5):
            partial_keys = partial_keys + cute.arch.shuffle_sync_bfly(partial_keys, offset=1 << (4 - offset))
        if lane == 0:
            s_warp[warp_idx] = partial
            s_red[warp_idx, 0, 0] = partial_keys
        sync()
        s_k_t, s_q_t, s_sq_t = transposed(s_k, HEAD_DIM, BT), transposed(s_q, HEAD_DIM, BT), transposed(s_sq, BT, BT)
        product(acc_dq, s_sq, s_k_t, False, True)
        product(acc_dk, s_sq_t, s_q_t, True, True)
        state_total, key_total = Float32(0.0), Float32(0.0)
        for w in cutlass.range_constexpr(GRAD_THREADS // 32):
            state_total, key_total = state_total + s_warp[w], key_total + s_red[w, 0, 0]
        dg_last = state_total * cute.math.exp2(gate_last, fastmath=True) + key_total
        r_q_c, r_k_c = c_values(s_q), c_values(s_k)
        for elem in cutlass.range_constexpr(cute.size(acc_dq)):
            contribution = acc_dq[elem] * Float32(r_q_c[elem]) - acc_dk[elem] * Float32(r_k_c[elem])
            if const_expr(elem % 4 < 2):
                dg_first = dg_first + contribution
            else:
                dg_second = dg_second + contribution
        r_out.store(acc_dq.load().to(BFloat16))
        sync()  # every thread is done with do, v_new, h, dh and the dg_last partials
        cute.copy(st_C, st_C.retile(r_out), st_C.partition_D(s_do))
        # step 3: the WY backward; k beta 2^g, k beta and v beta where h and dh were
        t_k, t_v, t_kbg, t_kb, t_vb, row_coords = load.partition_D(s_k), load.partition_D(s_v), load.partition_D(s_kbg), load.partition_D(s_kb), load.partition_D(s_vb), load.partition_D(cute.make_identity_tensor((BT, HEAD_DIM)))
        for i in cutlass.range_constexpr(cute.size(t_k, mode=[1])):
            for j in cutlass.range_constexpr(cute.size(t_k, mode=[2])):
                row = row_coords[0, i, j][0]
                beta = s_beta[row]
                keys = t_k[None, i, j].load().to(Float32)
                t_kbg[None, i, j].store((keys * (beta * cute.math.exp2(s_gcum[row], fastmath=True))).to(BFloat16))
                t_kb[None, i, j].store((keys * beta).to(BFloat16))
                t_vb[None, i, j].store((t_v[None, i, j].load().to(Float32) * beta).to(BFloat16))
        sync()
        if const_expr(self.ragged):
            copy_rows(store_128, store_128.partition_S(s_do), store_128.partition_D(tile(mDq, head_idx)), rows_128, valid, False)
        else:
            cute.copy(store_128, store_128.partition_S(s_do), store_128.partition_D(tile(mDq, head_idx)))
        acc_da = cute.make_rmem_tensor(thr_square.partition_shape_C((BT, BT)), Float32)
        acc_da.fill(0.0)
        product(acc_da, s_dw, s_kbg, False, False)
        product(acc_da, s_du, s_vb, False, False)
        s_a_t, s_dw_t, s_du_t = transposed(s_a, BT, BT), transposed(s_dw, HEAD_DIM, BT), transposed(s_du, HEAD_DIM, BT)
        acc.fill(0.0)
        product(acc, s_a_t, s_dw_t, True, True)
        r_k_c = c_values(s_k)
        for elem in cutlass.range_constexpr(cute.size(acc)):
            # the fragment's elements 2 and 3 of every 4 sit on the thread's second row
            beta, decay = (beta_second, decay_second) if elem % 4 >= 2 else (beta_first, decay_first)
            keys = Float32(r_k_c[elem])
            acc_dk[elem] = acc_dk[elem] + acc[elem] * (decay * beta)
            beta_term, gate_term = acc[elem] * keys * decay, acc[elem] * (keys * (beta * decay))
            if const_expr(elem % 4 < 2):
                db_first, dg_first = db_first + beta_term, dg_first + gate_term
            else:
                db_second, dg_second = db_second + beta_term, dg_second + gate_term
        acc.fill(0.0)
        product(acc, s_a_t, s_du_t, True, True)
        r_v_c = c_values(s_v)
        for elem in cutlass.range_constexpr(cute.size(acc)):
            beta = beta_second if elem % 4 >= 2 else beta_first
            beta_term = acc[elem] * Float32(r_v_c[elem])
            if const_expr(elem % 4 < 2):
                db_first = db_first + beta_term
            else:
                db_second = db_second + beta_term
            acc[elem] = acc[elem] * beta
        r_out.store(acc.load().to(BFloat16))
        for elem in cutlass.range_constexpr(cute.size(acc_da)):
            row, col = sq_coords[elem]
            value = Float32(0.0)
            if row > col:
                value = acc_da[elem]
            acc_da[elem] = value
        r_da = cute.make_fragment_like(acc_da, BFloat16)
        r_da.store(acc_da.load().to(BFloat16))
        sync()  # ds (in s_sq) and q, v_new are no longer read
        cute.copy(st_C, st_C.retile(r_out), st_C.partition_D(s_vnew))
        cute.copy(st_C_square, st_C_square.retile(r_da), st_C_square.partition_D(s_sq))
        sync()
        if const_expr(self.ragged):
            copy_rows(store_128, store_128.partition_S(s_vnew), store_128.partition_D(tile(mDv, head_idx)), rows_128, valid, False)
        else:
            cute.copy(store_128, store_128.partition_S(s_vnew), store_128.partition_D(tile(mDv, head_idx)))
        acc_da.fill(0.0)
        product(acc_da, s_sq, s_a, False, False)
        r_da.store(acc_da.load().to(BFloat16))
        sync()
        cute.copy(st_C_square, st_C_square.retile(r_da), st_C_square.partition_D(s_sq))
        sync()
        acc_da.fill(0.0)
        product(acc_da, s_a_t, s_sq_t, True, True)
        for elem in cutlass.range_constexpr(cute.size(acc_da)):
            row, col = sq_coords[elem]
            value = Float32(0.0)
            if row > col:
                value = -acc_da[elem] * cute.math.exp2(s_gcum[row] - s_gcum[col], fastmath=True)
            acc_da[elem] = value
        r_da.store(acc_da.load().to(BFloat16))
        acc_kk = cute.make_rmem_tensor(thr_square.partition_shape_C((BT, BT)), Float32)
        acc_kk.fill(0.0)
        product(acc_kk, s_k, s_k, False, False)
        # C = dA * beta_i k_i.k_j in acc_kk; dg += rowsum(C) - colsum(C)
        for elem in cutlass.range_constexpr(cute.size(acc_kk)):
            beta = beta_second if elem % 4 >= 2 else beta_first
            acc_kk[elem] = Float32(r_da[elem]) * (acc_kk[elem] * beta)
            if const_expr(elem % 4 < 2):
                dg_first = dg_first + acc_kk[elem]
            else:
                dg_second = dg_second + acc_kk[elem]
        # an MMA C fragment is (2 columns, 2 rows 8 apart) per atom: elements e and e + 2 share a column
        for atom in cutlass.range_constexpr(cute.size(acc_kk) // 4):
            for half in cutlass.range_constexpr(2):
                part = acc_kk[4 * atom + half] + acc_kk[4 * atom + half + 2]
                for offset in cutlass.range_constexpr(3):
                    part = part + cute.arch.shuffle_sync_bfly(part, offset=4 << offset)
                if lane < 4:
                    s_col[warp_idx % 4, sq_coords[4 * atom + half][1]] = part
        sync()
        cute.copy(st_C_square, st_C_square.retile(r_da), st_C_square.partition_D(s_sq))
        sync()
        acc.fill(0.0)
        product(acc, s_sq, s_k_t, False, True)
        r_k_c = c_values(s_k)
        for elem in cutlass.range_constexpr(cute.size(acc)):
            beta = beta_second if elem % 4 >= 2 else beta_first
            beta_term = acc[elem] * Float32(r_k_c[elem])
            if const_expr(elem % 4 < 2):
                db_first = db_first + beta_term
            else:
                db_second = db_second + beta_term
            acc_dk[elem] = acc_dk[elem] + acc[elem] * beta
        s_kb_t = transposed(s_kb, HEAD_DIM, BT)
        product(acc_dk, s_sq_t, s_kb_t, True, True)
        # step 4: the row sums, dk, then dg's reverse cumsum and dbeta
        for offset in cutlass.range_constexpr(2):
            dg_first, dg_second, db_first, db_second = dg_first + cute.arch.shuffle_sync_bfly(dg_first, offset=1 << offset), dg_second + cute.arch.shuffle_sync_bfly(dg_second, offset=1 << offset), db_first + cute.arch.shuffle_sync_bfly(db_first, offset=1 << offset), db_second + cute.arch.shuffle_sync_bfly(db_second, offset=1 << offset)
        first_row, column_warp = c_coords[0][0], warp_idx // 4
        if lane % 4 == 0:
            s_red[first_row, column_warp, 0], s_red[first_row + 8, column_warp, 0] = dg_first, dg_second
            s_red[first_row, column_warp, 1], s_red[first_row + 8, column_warp, 1] = db_first, db_second
        r_out.store(acc_dk.load().to(BFloat16))
        cute.copy(st_C, st_C.retile(r_out), st_C.partition_D(s_q))
        sync()
        if const_expr(self.ragged):
            copy_rows(store_128, store_128.partition_S(s_q), store_128.partition_D(tile(mDk, head_idx)), rows_128, valid, False)
        else:
            cute.copy(store_128, store_128.partition_S(s_q), store_128.partition_D(tile(mDk, head_idx)))
        if tidx < BT:
            gate_sum = s_red[tidx, 0, 0] + s_red[tidx, 1, 0] - (s_col[0, tidx] + s_col[1, tidx] + s_col[2, tidx] + s_col[3, tidx])
            if tidx == BT - 1:
                gate_sum = gate_sum + dg_last
            beta_sum = s_red[tidx, 0, 1] + s_red[tidx, 1, 1]
            if const_expr(self.ragged):
                if tidx < valid:
                    mDbeta[chunk_idx * BT + tidx, head_idx] = BFloat16(beta_sum)
            else:
                mDbeta[chunk_idx * BT + tidx, head_idx] = BFloat16(beta_sum)
            s_red[tidx, 0, 0] = gate_sum
        sync()
        # the reverse in-chunk cumsum, warp 0: lane l owns rows 2l and 2l + 1
        if warp_idx == 0:
            even, odd = s_red[2 * lane, 0, 0], s_red[2 * lane + 1, 0, 0]
            running = even + odd
            for i in cutlass.range_constexpr(5):
                from_above = cute.arch.shuffle_sync_down(running, offset=1 << i)
                if lane + (1 << i) < 32:
                    running = running + from_above
            after = cute.arch.shuffle_sync_down(running, offset=1)
            if lane == 31:
                after = Float32(0.0)
            odd_total, row = odd + after, chunk_idx * BT + 2 * lane
            if const_expr(self.ragged):
                if 2 * lane < valid:
                    mDg[row, head_idx] = even + odd_total
                if 2 * lane + 1 < valid:
                    mDg[row + 1, head_idx] = odd_total
            else:
                mDg[row, head_idx], mDg[row + 1, head_idx] = even + odd_total, odd_total


COMPILED: dict[tuple, object] = {}


def backward_kernel(name: str, H: int, HV: int, device: int, ragged: bool, flags: tuple = ()) -> object:
    """
    The backward kernel name, compiled once per (name, H, HV, device, flags, ragged) with T
    symbolic: its ragged variant serves every T that is not a multiple of BT. flags are the
    context-parallel compile flags, empty for the dense backward: (has_h0, has_ht, store_states)
    for state_recompute, (has_dht, has_dh0, store_states) for state_gradient. A final state or an
    initial-state gradient is written into the (HV, K, V + K) transition buffer's first V columns.
    """
    key = (name, H, HV, device, flags, ragged)
    if key not in COMPILED:
        T, NT = cute.sym_int(), cute.sym_int()
        keys, values, gate, states, state_in, state_out = fake(BFloat16, (T, H, HEAD_DIM), (H * HEAD_DIM, HEAD_DIM, 1)), fake(BFloat16, (T, HV, HEAD_DIM), (HV * HEAD_DIM, HEAD_DIM, 1)), fake(Float32, (T, HV), (HV, 1), 4), fake(BFloat16, (NT, HV, HEAD_DIM, HEAD_DIM), (HV * HEAD_DIM * HEAD_DIM, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)), fake(Float32, (HV, HEAD_DIM, HEAD_DIM), (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)), fake(Float32, (HV, HEAD_DIM, HEAD_DIM), (2 * HEAD_DIM * HEAD_DIM, 2 * HEAD_DIM, 1))
        first, last, store = flags if flags else (False, False, True)
        match name:
            case "bwd_prologue":
                a, beta = fake(BFloat16, (T, HV, BT), (HV * BT, BT, 1)), fake(BFloat16, (T, HV), (HV, 1), 2)
                COMPILED[key] = cute.compile(BwdPrologue(H, HV, ragged), keys, keys, values, values, gate, beta, a, values, values, values, fake_stream(), options="--enable-tvm-ffi")
            case "state_gradient":
                COMPILED[key] = cute.compile(StateGradient(H, HV, ragged, first, last, store), keys, keys, values, values, values, gate, states if store else None, values if store else None, state_in if first else None, state_out if last else None, fake_stream(), options="--enable-tvm-ffi")
            case "chunk_gradients":
                a, beta = fake(BFloat16, (T, HV, BT), (HV * BT, BT, 1)), fake(BFloat16, (T, HV), (HV, 1), 2)
                COMPILED[key] = cute.compile(ChunkGradients(H, HV, ragged), keys, keys, values, values, values, values, states, states, gate, beta, a, values, values, values, gate, beta, fake_stream(), options="--enable-tvm-ffi")
            case "state_recompute":
                COMPILED[key] = cute.compile(StateRecompute(H, HV, ragged, first, last, store), keys, values, values, gate, states if store else None, values if store else None, state_in if first else None, state_out if last else None, fake_stream(), options="--enable-tvm-ffi")
    return COMPILED[key]


def bwd_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (w, u, dv_local), bf16 (1, T, HV, K / V).
    """
    T, H, HV = k.shape[1], k.shape[2], v.shape[2]
    w, u, dv_local = k.new_empty(1, T, HV, HEAD_DIM), torch.empty_like(v), torch.empty_like(v)
    backward_kernel("bwd_prologue", H, HV, k.device.index, T % BT != 0)(q[0], k[0], v[0], do[0], g_cum[0], beta[0], A[0], w[0], u[0], dv_local[0])
    return w, u, dv_local


def state_recompute(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cum: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (h, v_new): h bf16 (1, ceil(T / 64), HV, V, K), the chunk-entry states V-first; v_new bf16 like
    u. From h0, fp32 (HV, K, V), when one is passed, else from zero.
    """
    T, H, HV = k.shape[1], k.shape[2], u.shape[2]
    h, v_new = k.new_empty(1, (T + BT - 1) // BT, HV, HEAD_DIM, HEAD_DIM), torch.empty_like(u)
    flags = () if h0 is None else (True, False, True)
    backward_kernel("state_recompute", H, HV, k.device.index, T % BT != 0, flags)(k[0], w[0], u[0], g_cum[0], h[0], v_new[0], h0, None)
    return h, v_new


def state_final(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cum: torch.Tensor, ht: torch.Tensor) -> None:
    """
    The recurrence alone from zero, the final state into ht, fp32 (HV, K, V) with row stride 2 K
    (the first V columns of an (HV, K, V + K) buffer); nothing else is stored.
    """
    T, H, HV = k.shape[1], k.shape[2], u.shape[2]
    backward_kernel("state_recompute", H, HV, k.device.index, T % BT != 0, (False, True, False))(k[0], w[0], u[0], g_cum[0], None, None, None, ht)


def state_gradient(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, do: torch.Tensor, dv_local: torch.Tensor, g_cum: torch.Tensor, du: torch.Tensor | None = None, dht: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (dh, du): dh bf16 (1, ceil(T / 64), HV, V, K), the chunk states' gradients V-first; du bf16 like
    do, into the given buffer when one is passed. From dht, fp32 (HV, K, V), the gradient of the
    final state, when one is passed, else from zero.
    """
    T, H, HV = k.shape[1], k.shape[2], do.shape[2]
    dh = k.new_empty(1, (T + BT - 1) // BT, HV, HEAD_DIM, HEAD_DIM)
    du = torch.empty_like(do) if du is None else du
    flags = () if dht is None else (True, False, True)
    backward_kernel("state_gradient", H, HV, k.device.index, T % BT != 0, flags)(q[0], k[0], w[0], do[0], dv_local[0], g_cum[0], dh[0], du[0], dht, None)
    return dh, du


def state_gradient_initial(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, do: torch.Tensor, dv_local: torch.Tensor, g_cum: torch.Tensor, dh0: torch.Tensor) -> None:
    """
    The reverse recurrence alone from zero, the initial state's gradient into dh0, fp32 (HV, K, V)
    with row stride 2 K (the first V columns of an (HV, K, V + K) buffer); nothing else is stored.
    """
    T, H, HV = k.shape[1], k.shape[2], do.shape[2]
    backward_kernel("state_gradient", H, HV, k.device.index, T % BT != 0, (False, True, False))(q[0], k[0], w[0], do[0], dv_local[0], g_cum[0], None, None, None, dh0)


def chunk_gradients(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, v_new: torch.Tensor, do: torch.Tensor, du: torch.Tensor, h: torch.Tensor, dh: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, dv: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (dq, dk, dv, dg, dbeta), the backward's outputs; dv into the given buffer when one is passed.
    """
    T, H, HV = k.shape[1], k.shape[2], v.shape[2]
    dq, dk = q.new_empty(1, T, HV, HEAD_DIM), k.new_empty(1, T, HV, HEAD_DIM)
    dv = torch.empty_like(v) if dv is None else dv
    dg, dbeta = torch.empty_like(g_cum), torch.empty_like(beta)
    backward_kernel("chunk_gradients", H, HV, k.device.index, T % BT != 0)(q[0], k[0], v[0], v_new[0], do[0], du[0], h[0], dh[0], g_cum[0], beta[0], A[0], dq[0], dk[0], dv[0], dg[0], dbeta[0])
    return dq, dk, dv, dg, dbeta


def chunk_gdr_bwd_cute(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The backward on the four kernels, B == 1 (their ragged variants when T is not a multiple of BT).
    du reuses u's buffer (read only by StateRecompute) and dv dv_local's (read only by
    StateGradient); every intermediate keeps T rows, h and dh ceil(T / 64) chunks.
    """
    w, u, dv_local = bwd_prologue(q, k, v, do, g_cum, beta, A)
    h, v_new = state_recompute(k, w, u, g_cum)
    dh, du = state_gradient(q, k, w, do, dv_local, g_cum, du=u)
    return chunk_gradients(q, k, v, v_new, do, du, h, dh, g_cum, beta, A, dv=dv_local)


def chunk_gdr_bwd_pinned_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward of chunk_gdr_fwd_pinned_sm90: fresh (dq, dk, dv, dg, dbeta).

    q, k, v, beta: the forward inputs; g_cum, A: the forward outputs; do: bf16 (B, T, HV, V), the
    gradient of o. Inputs are not modified.

    dq, dk: (B, T, HV, K) in q's / k's dtype, per value head (the caller sums each group of HV / H
    heads, in repeat_interleave order, for the key-head gradients). dv like v; dg: fp32 (B, T, HV),
    with respect to the raw gate g; dbeta like beta. The CuTe kernels at every T (the ragged
    variants when T is not a multiple of 64), on FLA 0.5.2's chunk algorithm and rounding points
    (fp32 accumulation, so dk and dg skip the bf16 / fp32 buffers FLA adds them through). Matches
    the fp32 recurrence's autograd to calc_diff < 1e-3 and a largest absolute error within 2e-2 of
    the reference's largest magnitude (tests/kernels/test_gated_delta_rule_pinned.py).
    Deterministic. Only where gated_delta_rule_pinned_supported is True.
    """
    return chunk_gdr_bwd_cute(q, k, v, g_cum, beta, A, do)
