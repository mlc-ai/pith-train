"""
SM90 context-parallel chunked gated delta rule (chunk size 64): FLA 0.5.2's context-parallel
structure (FLA's ops/cp/chunk_delta_h.py) on CuTe-DSL kernels. Each rank holds a
contiguous slice of one sequence. In the forward it computes its affine state map from zero,
h_end = M h_start + he, where he is the recurrence run from a zero state (StateRecompute with a
final state and no per-chunk stores, FLA's pre-process h half) and M is the product of the per-chunk
transitions M_c = diag(2^g_last) - bf16(k 2^(g_last - g))^T w (TransitionChain, FLA's pre-process M
half). The caller all-gathers the maps (hm, HV x K x (V + K) fp32 per rank); StateMerge folds the
predecessors' maps into the rank's initial state h0, which ChunkStateOutput starts from. The
backward mirrors this: the reverse recurrence from a zero gradient (StateGradient with an initial
gradient and no per-chunk stores) and the reverse chain of M_c^T computed as diag(2^g_last) - w^T
bf16(k 2^(g_last - g)) give dhm; the caller all-gathers it; StateMerge folds the successors' maps
into dht, which StateGradient starts from. The collectives stay with the caller.

Rounding follows FLA's kernels: bf16 k~ = bf16(k 2^(g_last - g)) (exp2 with ftz, round to nearest
even), the M_c product on bf16 tensor cores with fp32 accumulation in increasing K order, the
subtraction from the diagonal in fp32 after the product, and the chain M <- M_c M and the merge h
<- M_s h + he_s as TF32 tensor-core products of the raw fp32 bits (truncated, no rounding
conversion) in increasing K8 order, each instruction's slot p reading k = 8 j + p of block j, the
merge's he_s the accumulator the products are added to. The verifier
(tests/kernels/test_gated_delta_rule_cp.py) compares every piece with a float64 PyTorch evaluation.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from flash_attn.cute.ampere_helpers import gemm, get_smem_layout_atom

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.gated_delta_rule_pinned_bwd_sm90 import bwd_prologue, chunk_gradients, state_final, state_gradient, state_gradient_initial, state_recompute
from pithtrain.kernels.gated_delta_rule_pinned_fwd_sm90 import BT, HEAD_DIM, copy_rows, operand_copies, state_output, wy_representation

__all__ = ["StateMerge", "TransitionChain", "gdr_cp_bwd_local_sm90", "gdr_cp_bwd_output_sm90", "gdr_cp_bwd_states_sm90", "gdr_cp_fwd_local_sm90", "gdr_cp_fwd_output_sm90", "gdr_cp_merge_sm90", "gated_delta_rule_cp_supported_sm90"]

CHAIN_BV = 32  # transition columns per CTA
CHAIN_THREADS = 128
CHAIN_STAGES = 2  # chunks of k, w and g in flight
MERGE_BV = 32  # state columns per CTA of the merge
MERGE_THREADS = 128
F32_PAD = 4  # fp32 row padding of the TF32 operand tiles


@cute.jit
def tf32_product(mma_tf32: cute.TiledMma, thr: cute.ThrMma, acc: cute.Tensor, s_a: cute.Tensor, s_b: cute.Tensor, rows: cutlass.Constexpr[int], cols: cutlass.Constexpr[int]):
    """
    acc (rows x cols) += s_a @ s_b on the warps' TF32 MMA: s_a (rows, HEAD_DIM) fp32 row-major with
    row stride HEAD_DIM + F32_PAD, s_b (HEAD_DIM, cols) fp32 row-major with row stride cols +
    F32_PAD. The fp32 values go to the MMA as they are (recast, not converted), so it reads them
    truncated to TF32. K8 blocks in increasing order, slot p of each instruction reading k = 8 j + p.
    """
    s_a_k, s_b_k = cute.make_tensor(s_a.iterator, cute.make_layout((rows, HEAD_DIM), stride=(HEAD_DIM + F32_PAD, 1))), cute.make_tensor(s_b.iterator, cute.make_layout((cols, HEAD_DIM), stride=(1, cols + F32_PAD)))
    t_a, t_b = thr.partition_A(s_a_k), thr.partition_B(s_b_k)
    tf32 = lambda r: cute.make_tensor(cute.recast_ptr(r.iterator, dtype=cutlass.TFloat32), r.layout)
    # registers in the atom's fragment order (partition_shape), one K8 block at a time
    r_a, r_b = cute.make_rmem_tensor(thr.partition_shape_A((rows, 8)), Float32), cute.make_rmem_tensor(thr.partition_shape_B((cols, 8)), Float32)
    for kb in cutlass.range_constexpr(cute.size(t_a, mode=[2])):
        cute.autovec_copy(t_a[None, None, kb], r_a[None, None, 0])
        cute.autovec_copy(t_b[None, None, kb], r_b[None, None, 0])
        cute.gemm(mma_tf32, acc, tf32(r_a)[None, None, 0], tf32(r_b)[None, None, 0], acc)


class TransitionChain:
    """
    The transition of a rank's chunk sequence, one CTA per (CHAIN_BV columns, value head), four
    warps: starting at the identity, chunk by chunk (n = 0 .. NT - 1, or NT - 1 .. 0 with reverse)
        k~ = bf16(k_n 2^(g_last - g));  P = k~^T w_n (reverse: w_n^T k~), bf16 MMA, fp32
        M_c = diag(2^g_last) - P;  M = tf32(M_c) tf32(M)
    and M's columns stored, fp32, into out (HV, K, K) with any row stride: forward the product
    M_{NT-1} .. M_0 of FLA's pre_process_fwd_kernel_merged, reverse M_0^T .. M_{NT-1}^T of its
    pre_process_bwd_kernel_merged. k and w ride a CHAIN_STAGES ring of cp.async loads.

    With ragged (T not a multiple of BT) the last chunk runs as if zero-padded to BT rows: k and w
    rows past T load as zeros, gate rows as g_cum[T - 1] (so g_last is the last valid row's).
    """

    def __init__(self, H: int, HV: int, reverse: bool, ragged: bool = False):
        self.H, self.HV, self.reverse, self.ragged = H, HV, reverse, ragged

    @cute.jit
    def __call__(self, mK: cute.Tensor, mW: cute.Tensor, mGcum: cute.Tensor, mOut: cute.Tensor, stream):
        mma, mma_tf32 = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16)), cute.make_tiled_mma(warp.MmaTF32Op((16, 8, 8)), (4, 1, 1), permutation_mnk=(64, 8, 8))
        ring_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (BT, HEAD_DIM, CHAIN_STAGES), (0, 1, 2))
        self.kernel(mK, mW, mGcum, mOut, mma, mma_tf32, ring_layout).launch(grid=[HEAD_DIM // CHAIN_BV, self.HV, 1], block=[CHAIN_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mK: cute.Tensor, mW: cute.Tensor, mGcum: cute.Tensor, mOut: cute.Tensor, mma: cute.TiledMma, mma_tf32: cute.TiledMma, ring_layout: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        col_tile, head_idx, _ = cute.arch.block_idx()
        key_head_idx = head_idx // (self.HV // self.H)
        smem = cutlass.utils.SmemAllocator()
        s_k, s_w, s_gcum, s_mc, s_m = smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(Float32, cute.make_layout((BT, CHAIN_STAGES), stride=(1, BT)), 16), smem.allocate_tensor(Float32, cute.make_layout((HEAD_DIM, HEAD_DIM), stride=(HEAD_DIM + F32_PAD, 1)), 16), smem.allocate_tensor(Float32, cute.make_layout((HEAD_DIM, CHAIN_BV), stride=(CHAIN_BV + F32_PAD, 1)), 16)
        num_chunks = (mGcum.shape[0] + BT - 1) // BT if const_expr(self.ragged) else mGcum.shape[0] // BT
        threads = CHAIN_THREADS
        atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
        load = cute.make_tiled_copy_tv(atom_load, cute.make_ordered_layout((threads // 16, 16), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        load_gate = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32), cute.make_layout(BT), cute.make_layout(1)).get_slice(tidx % BT)
        gK, gW, gGcum = cute.local_tile(mK[None, key_head_idx, None], (BT, HEAD_DIM), (None, 0)), cute.local_tile(mW[None, head_idx, None], (BT, HEAD_DIM), (None, 0)), cute.local_tile(mGcum[None, head_idx], (BT,), (None,))
        t_k_src, t_w_src, t_g_src = load.partition_S(gK), load.partition_S(gW), load_gate.partition_S(gGcum)
        t_k_dst, t_w_dst, t_g_dst = load.partition_D(s_k), load.partition_D(s_w), load_gate.partition_D(s_gcum)
        rows_wide = load.partition_D(cute.make_identity_tensor((BT, HEAD_DIM)))
        # the identity's columns col_tile * CHAIN_BV ..
        for i in cutlass.range_constexpr(HEAD_DIM * CHAIN_BV // threads):
            idx = i * threads + tidx
            row, col = idx // CHAIN_BV, idx % CHAIN_BV
            value = Float32(0.0)
            if row == col_tile * CHAIN_BV + col:
                value = Float32(1.0)
            s_m[row, col] = value
        for step in cutlass.range_constexpr(CHAIN_STAGES - 1):
            if step < num_chunks:
                chunk = num_chunks - 1 - step if const_expr(self.reverse) else step
                self.issue_chunk(load, load_gate, (t_k_src, t_w_src, t_g_src), (t_k_dst, t_w_dst, t_g_dst), rows_wide, mGcum, s_gcum, head_idx, chunk, step, tidx)
            cute.arch.cp_async_commit_group()
        thr, thr_tf32 = mma.get_slice(tidx), mma_tf32.get_slice(tidx)
        _, _, ldt_A, ldt_B, _, _, _ = operand_copies(mma, tidx)
        c_coords = thr.partition_C(cute.make_identity_tensor((HEAD_DIM, HEAD_DIM // 2)))
        acc_m = cute.make_rmem_tensor(thr_tf32.partition_shape_C((HEAD_DIM, CHAIN_BV)), Float32)
        for step in cutlass.range(num_chunks, unroll=1):
            stage = step % CHAIN_STAGES
            upcoming = step + CHAIN_STAGES - 1
            cute.arch.cp_async_wait_group(CHAIN_STAGES - 2)
            cute.arch.barrier()
            # every thread is done with the previous step's stage: refill it
            if upcoming < num_chunks:
                chunk = num_chunks - 1 - upcoming if const_expr(self.reverse) else upcoming
                self.issue_chunk(load, load_gate, (t_k_src, t_w_src, t_g_src), (t_k_dst, t_w_dst, t_g_dst), rows_wide, mGcum, s_gcum, head_idx, chunk, upcoming % CHAIN_STAGES, tidx)
            cute.arch.cp_async_commit_group()
            gate_last = s_gcum[BT - 1, stage]
            s_k_n, s_w_n = s_k[None, None, stage], s_w[None, None, stage]
            # k~ = bf16(k 2^(g_last - g)) in place, 8 columns per access
            t_k = load.partition_D(s_k_n)
            for i in cutlass.range_constexpr(cute.size(t_k, mode=[1])):
                for j in cutlass.range_constexpr(cute.size(t_k, mode=[2])):
                    factor = cute.math.exp2(gate_last - s_gcum[rows_wide[0, i, j][0], stage], fastmath=True)
                    t_k[None, i, j].store((t_k[None, i, j].load().to(Float32) * factor).to(BFloat16))
            cute.arch.barrier()
            # M_c = diag(2^g_last) - P, P = left^T right over the chunk's rows, by column halves
            decay = cute.math.exp2(gate_last, fastmath=True)
            s_left, s_right = (s_w_n, s_k_n) if const_expr(self.reverse) else (s_k_n, s_w_n)
            s_left_t, s_right_t = cute.composition(s_left, cute.make_ordered_layout((HEAD_DIM, BT), order=(1, 0))), cute.composition(s_right, cute.make_ordered_layout((HEAD_DIM, BT), order=(1, 0)))
            for half in cutlass.range_constexpr(2):
                s_right_half = cute.local_tile(s_right_t, (HEAD_DIM // 2, BT), (half, 0))
                acc_p = cute.make_rmem_tensor(thr.partition_shape_C((HEAD_DIM, HEAD_DIM // 2)), Float32)
                acc_p.fill(0.0)
                gemm(mma, acc_p, thr.make_fragment_A(thr.partition_A(s_left_t)), thr.make_fragment_B(thr.partition_B(s_right_half)), ldt_A.partition_S(s_left_t), ldt_B.partition_S(s_right_half), ldt_A, ldt_B)
                for elem in cutlass.range_constexpr(cute.size(acc_p)):
                    row, col = c_coords[elem][0], c_coords[elem][1] + half * (HEAD_DIM // 2)
                    diagonal = Float32(0.0)
                    if row == col:
                        diagonal = decay
                    s_mc[row, col] = diagonal - acc_p[elem]
            cute.arch.barrier()
            acc_m.fill(0.0)
            tf32_product(mma_tf32, thr_tf32, acc_m, s_mc, s_m, HEAD_DIM, CHAIN_BV)
            cute.arch.barrier()
            t_m = thr_tf32.partition_C(s_m)
            for elem in cutlass.range_constexpr(cute.size(acc_m)):
                t_m[elem] = acc_m[elem]
        cute.arch.barrier()
        for i in cutlass.range_constexpr(HEAD_DIM * CHAIN_BV // threads):
            idx = i * threads + tidx
            row, col = idx // CHAIN_BV, idx % CHAIN_BV
            mOut[head_idx, row, col_tile * CHAIN_BV + col] = s_m[row, col]

    @cute.jit
    def issue_chunk(self, load: cute.TiledCopy, load_gate: cute.TiledCopy, sources: tuple, targets: tuple, rows_wide: cute.Tensor, mGcum: cute.Tensor, s_gcum: cute.Tensor, head_idx, chunk, stage, tidx):
        """
        cp.async of chunk's k, w and g into ring stage; with ragged, rows past T zeroed and gate rows
        past T set to g_cum[T - 1]. The caller commits the group.
        """
        t_k_src, t_w_src, t_g_src = sources
        t_k_dst, t_w_dst, t_g_dst = targets
        if const_expr(self.ragged):
            T = mGcum.shape[0]
            valid = T - chunk * BT
            copy_rows(load, t_k_src[None, None, None, chunk], t_k_dst[None, None, None, stage], rows_wide, valid, True)
            copy_rows(load, t_w_src[None, None, None, chunk], t_w_dst[None, None, None, stage], rows_wide, valid, True)
            if tidx < BT:
                if tidx < valid:
                    cute.copy(load_gate, t_g_src[None, None, chunk], t_g_dst[None, None, stage])
                else:
                    s_gcum[tidx, stage] = mGcum[T - 1, head_idx]
        else:
            cute.copy(load, t_k_src[None, None, None, chunk], t_k_dst[None, None, None, stage])
            cute.copy(load, t_w_src[None, None, None, chunk], t_w_dst[None, None, None, stage])
            if tidx < BT:
                cute.copy(load_gate, t_g_src[None, None, chunk], t_g_dst[None, None, stage])


class StateMerge:
    """
    The fold of the gathered ranks' affine maps into one state, one CTA per (MERGE_BV state columns,
    value head), four warps, FLA's merge_fwd_bwd_kernel in its context-parallel mode: starting at h
    = 0, for s over count ranks (forward: rank - count .. rank - 1; reverse: rank + count .. rank +
    1)
        h = tf32(M_s) tf32(h) + he_s
    with ag (W, HV, K, V + K) fp32 holding (he_s | M_s) per rank s, he_s the accumulator the TF32
    products are added to, in increasing K8 order. h is stored, fp32 (HV, K, V). M_s rides a
    double buffer of cp.async loads, the next rank's loaded during the current product.
    """

    def __init__(self, HV: int, reverse: bool):
        self.HV, self.reverse = HV, reverse

    @cute.jit
    def __call__(self, mAg: cute.Tensor, mOut: cute.Tensor, rank: Int32, count: Int32, stream):
        mma_tf32 = cute.make_tiled_mma(warp.MmaTF32Op((16, 8, 8)), (4, 1, 1), permutation_mnk=(64, 8, 8))
        self.kernel(mAg, mOut, rank, count, mma_tf32).launch(grid=[HEAD_DIM // MERGE_BV, self.HV, 1], block=[MERGE_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mAg: cute.Tensor, mOut: cute.Tensor, rank: Int32, count: Int32, mma_tf32: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        col_tile, head_idx, _ = cute.arch.block_idx()
        threads = MERGE_THREADS
        smem = cutlass.utils.SmemAllocator()
        s_m, s_h = smem.allocate_tensor(Float32, cute.make_layout((HEAD_DIM, HEAD_DIM, 2), stride=(HEAD_DIM + F32_PAD, 1, HEAD_DIM * (HEAD_DIM + F32_PAD))), 16), smem.allocate_tensor(Float32, cute.make_layout((HEAD_DIM, MERGE_BV), stride=(MERGE_BV + F32_PAD, 1)), 16)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), Float32, num_bits_per_copy=128)
        load = cute.make_tiled_copy_tv(atom, cute.make_ordered_layout((threads // 32, 32), order=(1, 0)), cute.make_layout((1, 4))).get_slice(tidx)
        t_m_dst = load.partition_D(s_m)
        for i in cutlass.range_constexpr(HEAD_DIM * MERGE_BV // threads):
            idx = i * threads + tidx
            s_h[idx // MERGE_BV, idx % MERGE_BV] = Float32(0.0)
        thr = mma_tf32.get_slice(tidx)
        coords = thr.partition_C(cute.make_identity_tensor((HEAD_DIM, MERGE_BV)))
        acc = cute.make_rmem_tensor(thr.partition_shape_C((HEAD_DIM, MERGE_BV)), Float32)
        acc.fill(0.0)
        source = lambda idx: rank + count - idx if const_expr(self.reverse) else rank - count + idx
        if count > 0:
            cute.copy(load, load.partition_S(cute.local_tile(mAg[source(0), head_idx, None, None], (HEAD_DIM, HEAD_DIM), (0, 1))), t_m_dst[None, None, None, 0])
        cute.arch.cp_async_commit_group()
        for idx in cutlass.range(count, unroll=1):
            buffer = idx % 2
            if idx + 1 < count:
                cute.copy(load, load.partition_S(cute.local_tile(mAg[source(idx + 1), head_idx, None, None], (HEAD_DIM, HEAD_DIM), (0, 1))), t_m_dst[None, None, None, 1 - buffer])
            cute.arch.cp_async_commit_group()
            # he_s as the accumulator, loaded at the fragment's coordinates
            for elem in cutlass.range_constexpr(cute.size(acc)):
                acc[elem] = mAg[source(idx), head_idx, coords[elem][0], col_tile * MERGE_BV + coords[elem][1]]
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier()
            tf32_product(mma_tf32, thr, acc, s_m[None, None, buffer], s_h, HEAD_DIM, MERGE_BV)
            cute.arch.barrier()
            t_h = thr.partition_C(s_h)
            for elem in cutlass.range_constexpr(cute.size(acc)):
                t_h[elem] = acc[elem]
        cute.arch.barrier()
        for i in cutlass.range_constexpr(HEAD_DIM * MERGE_BV // threads):
            idx = i * threads + tidx
            row, col = idx // MERGE_BV, idx % MERGE_BV
            mOut[head_idx, row, col_tile * MERGE_BV + col] = s_h[row, col]


COMPILED: dict[tuple, object] = {}


def gated_delta_rule_cp_supported_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    The geometry the context-parallel kernels take: ranks 4 / 4 / 4 / 3 / 3, all contiguous on one
    CUDA device, B == 1, T > 0, HV a multiple of 32 and H == HV / 2 (Qwen3.5's 16 / 32 heads, a
    batch folded into the heads), K == V == 128, bf16 q / k / v / beta and an fp32 gate.
    """
    tensors = (q, k, v, g, beta)
    if tuple(t.dim() for t in tensors) != (4, 4, 4, 3, 3):
        return False
    B, T, H, K = k.shape
    HV, V = v.shape[2:]
    consistent = T > 0 and tuple(v.shape[:2]) == (B, T) and tuple(g.shape) == (B, T, HV) and tuple(beta.shape) == (B, T, HV) and len({t.device for t in tensors}) == 1 and k.device.type == "cuda" and all(t.is_contiguous() for t in tensors)
    return consistent and B == 1 and HV % 32 == 0 and HV == 2 * H and K == 128 and V == 128 and q.shape == k.shape and q.dtype == k.dtype == v.dtype == beta.dtype == torch.bfloat16 and g.dtype == torch.float32


def chain_kernel(H: int, HV: int, reverse: bool, device: int, ragged: bool) -> object:
    """
    TransitionChain compiled once per (H, HV, reverse, device, ragged), T symbolic.
    """
    key = ("transition_chain", H, HV, device, reverse, ragged)
    if key not in COMPILED:
        T = cute.sym_int()
        keys, values, gate, out = fake(BFloat16, (T, H, HEAD_DIM), (H * HEAD_DIM, HEAD_DIM, 1)), fake(BFloat16, (T, HV, HEAD_DIM), (HV * HEAD_DIM, HEAD_DIM, 1)), fake(Float32, (T, HV), (HV, 1), 4), fake(Float32, (HV, HEAD_DIM, HEAD_DIM), (2 * HEAD_DIM * HEAD_DIM, 2 * HEAD_DIM, 1))
        COMPILED[key] = cute.compile(TransitionChain(H, HV, reverse, ragged), keys, values, gate, out, fake_stream(), options="--enable-tvm-ffi")
    return COMPILED[key]


def merge_kernel(HV: int, reverse: bool, device: int) -> object:
    """
    StateMerge compiled once per (HV, reverse, device); the gathered rank count, the rank
    and the predecessor / successor count are runtime arguments.
    """
    key = ("state_merge", 0, HV, device, reverse, False)
    if key not in COMPILED:
        W = cute.sym_int()
        ag, out = fake(Float32, (W, HV, HEAD_DIM, 2 * HEAD_DIM), (HV * 2 * HEAD_DIM * HEAD_DIM, 2 * HEAD_DIM * HEAD_DIM, 2 * HEAD_DIM, 1)), fake(Float32, (HV, HEAD_DIM, HEAD_DIM), (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1))
        COMPILED[key] = cute.compile(StateMerge(HV, reverse), ag, out, Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")
    return COMPILED[key]


def gdr_cp_fwd_local_sm90(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, transition: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (g_cum, A, w, u, hm): WyRepresentation, and with transition the rank's forward map into hm, fp32
    (HV, K, V + K) = (he | M); without it hm is allocated and left unwritten (the last rank's slot,
    which no rank reads).
    """
    _, T, H, _ = k.shape
    HV = v.shape[2]
    g_cum, A, w, u = wy_representation(k, v, g, beta)
    hm = torch.empty(HV, HEAD_DIM, 2 * HEAD_DIM, device=k.device, dtype=torch.float32)
    if transition:
        state_final(k, w, u, g_cum, hm[:, :, :HEAD_DIM])
        chain_kernel(H, HV, False, k.device.index, T % BT != 0)(k[0], w[0], g_cum[0], hm[:, :, HEAD_DIM:])
    return g_cum, A, w, u, hm


def gdr_cp_merge_sm90(ag: torch.Tensor, rank: int, count: int, reverse: bool) -> torch.Tensor:
    """
    The folded state, fp32 (HV, K, V): of the count ranks before rank (forward) or after it
    (reverse) in ag, (W, HV, K, V + K) fp32; zeros when count is 0.
    """
    W, HV = ag.shape[:2]
    out = torch.zeros(HV, HEAD_DIM, HEAD_DIM, device=ag.device, dtype=torch.float32) if count == 0 else torch.empty(HV, HEAD_DIM, HEAD_DIM, device=ag.device, dtype=torch.float32)
    if count > 0:
        merge_kernel(HV, reverse, ag.device.index)(ag, out, rank, count)
    return out


def gdr_cp_fwd_output_sm90(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cum: torch.Tensor, h0: torch.Tensor | None) -> torch.Tensor:
    """
    o from ChunkStateOutput starting at h0, fp32 (HV, K, V), or at zero when h0 is None.
    """
    return state_output(q, k, w, u, g_cum, h0)


def gdr_cp_bwd_local_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, transition: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (w, u, dv_local, dhm): BwdPrologue, and with transition the rank's backward map into dhm, fp32
    (HV, K, V + K) = (dh from zero | M^T); without it dhm is allocated and left unwritten (the first
    rank's slot, which no rank reads).
    """
    _, T, H, _ = k.shape
    HV = v.shape[2]
    w, u, dv_local = bwd_prologue(q, k, v, do, g_cum, beta, A)
    dhm = torch.empty(HV, HEAD_DIM, 2 * HEAD_DIM, device=k.device, dtype=torch.float32)
    if transition:
        state_gradient_initial(q, k, w, do, dv_local, g_cum, dhm[:, :, :HEAD_DIM])
        chain_kernel(H, HV, True, k.device.index, T % BT != 0)(k[0], w[0], g_cum[0], dhm[:, :, HEAD_DIM:])
    return w, u, dv_local, dhm


def gdr_cp_bwd_states_sm90(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cum: torch.Tensor, h0: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (h, v_new) from StateRecompute starting at h0 (or zero when None).
    """
    return state_recompute(k, w, u, g_cum, h0)


def gdr_cp_bwd_output_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, v_new: torch.Tensor, do: torch.Tensor, dv_local: torch.Tensor, h: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, u: torch.Tensor, dht: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (dq, dk, dv, dg, dbeta): StateGradient starting at dht (or zero when None), du into u's buffer,
    then ChunkGradients, dv into dv_local's.
    """
    dh, du = state_gradient(q, k, w, do, dv_local, g_cum, du=u, dht=dht)
    return chunk_gradients(q, k, v, v_new, do, du, h, dh, g_cum, beta, A, dv=dv_local)
