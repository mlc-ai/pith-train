"""
gated_delta_rule_cp_sm100: one rank's share of the chunked gated delta rule (chunk size 64, scale
K ** -0.5) under context parallelism on SM100, in TIRx-lite kernels. The rank holds a
contiguous shard of the sequence split over the CP group, at Qwen3.5's geometry (batch 1, 32
heads, head dims 128).

Numerically it follows the chunked CP algorithm of FLA 0.5.2 (the same bf16 / TF32 rounding
points): the sharded recurrence matches the fp64 PyTorch reference of the CP verifier, and its
all-gather payloads are (HV, K, V + K) fp32 state summaries of each rank. Each direction is two
stages around the one all-gather the caller runs over the CP group:

- forward, before: the persistent intra-chunk kernel (intra2: cumulative gate, WY matrix A, w, u
  and the masked q k^T, q / k / v staged by TMA) and, unless the rank is the last of its last local
  sequence, that sequence's state summary (fsum7; with several local sequences it also emits the
  last sequence's output, which starts from zero);
- forward, after: the merge of the predecessors' summaries into the initial state, fused into the
  state recurrence and output (rec7m), which also checkpoints the exact fp32 state entering up to
  three chunks into A's always-zero upper triangle for the backward;
- backward, before: w, u and dv_local (bprep2), the h recompute as independent sub-chains (rec_h,
  from the initial state, a checkpoint or a segment start; launched behind the dh summary chain
  with programmatic dependent launch when the first local sequence is long) and, unless the rank is
  the first of its first local sequence, that sequence's gradient summary (bsum7, which also stores
  the real dh / dv2 when that pass starts from a zero gradient);
- backward, after: the merge of the successors' gradient summaries, fused into the dh recurrence
  (brec7m), and the persistent gradient kernel (bgrad2: dq, dk, dv, dg, dbeta).

h / dh and v_new / dv2 travel between the kernels in bgrad2's 128 B XOR-swizzled UMMA operand
layouts (private scratch); A, the cumulative gate and the initial state are private to this
forward / backward pair.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin", "nvcc": []}}})  # nvcc without --use_fast_math

__all__ = ["chunk_gdr_cp_bwd_post_sm100", "chunk_gdr_cp_bwd_pre_sm100", "chunk_gdr_cp_fwd_post_sm100", "chunk_gdr_cp_fwd_pre_sm100", "gated_delta_rule_cp_plan_sm100"]


RCP_LN2 = 1.4426950408889634
HEAD_DIM = 128
CP_ASYNC_16 = 'cp.async.cg.shared.global'
CP_ASYNC_4 = 'cp.async.ca.shared.global'
UMMA_F16 = 'tcgen05.mma.cta_group::1.kind::f16'
UMMA_TF32 = 'tcgen05.mma.cta_group::1.kind::tf32'
UMMA_COMMIT = 'tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64'
TMEM_LD_32X32B_X32 = 'tcgen05.ld.sync.aligned.32x32b.x32.b32'
TMEM_ST_32X32B_X32 = 'tcgen05.st.sync.aligned.32x32b.x32.b32'
TMEM_LD_16X256B_X8 = 'tcgen05.ld.sync.aligned.16x256b.x8.b32'
TMEM_LD_16X256B_X4 = 'tcgen05.ld.sync.aligned.16x256b.x4.b32'
TMEM_ST_16X256B_X8 = 'tcgen05.st.sync.aligned.16x256b.x8.b32'
TMEM_LD_32X32B_X16 = 'tcgen05.ld.sync.aligned.32x32b.x16.b32'
BULK_COPY_G2S = 'cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes'
UMMA_DESC_K = 4611756662049538048
UMMA_DESC_MN = 4611756662083026944
UMMA_DESC_MN_TF32 = 2305913515463933952
KIB = 1024


def ld_global_s32(ptr):
    value = txl.local_scalar('int32')
    txl.ptx.ld.global_.s32(value, ptr)
    return value


def ld_global_f32(ptr):
    value = txl.local_scalar('float32')
    txl.ptx.ld.global_.f32(value, ptr)
    return value


def ld_global_bf16_as_f32(ptr):
    bits = txl.local_scalar('uint16')
    txl.ptx.ld.global_.b16(bits, ptr)
    value = txl.local_scalar('float32')
    txl.ptx.cvt.f32.bf16(value, bits)
    return value


def ld_shared_f32(smem, off):
    value = txl.local_scalar('float32')
    txl.ptx.ld.shared.f32(value, smem.ptr_to([off]))
    return value


def st_shared_f32(smem, off, val):
    txl.ptx.st.shared.f32(smem.ptr_to([off]), val)


def tmem_lane_offset(t):
    """
    TMEM address offset (lane field) of the 32-lane quarter that thread t of a warpgroup reads.
    """
    return txl.shift_left(txl.bitwise_and(t, 96), 16)


def umma_desc(smem, off, base):
    """
    UMMA shared-memory descriptor of the tile at byte offset off: base with the address field set.
    """
    addr = txl.cuda.cvta_generic_to_shared(smem.ptr_to([off]))
    lo = txl.cast(txl.bitwise_and(txl.shift_right(addr, txl.uint32(4)), txl.uint32(16383)), 'uint64')
    return txl.bitwise_or(txl.uint64(base), lo)


def round_to_bf16(x):
    """
    fp32 -> bf16 (RN) -> fp32.
    """
    bits = txl.reinterpret(txl.cast(x, 'bfloat16'), ty='uint16')
    value = txl.local_scalar('float32')
    txl.ptx.cvt.f32.bf16(value, bits)
    return value


def exp2_approx(x):
    value = txl.local_scalar('float32')
    txl.ptx['ex2.approx.ftz.f32'](value, x)
    return value


def pack_bf16x2(lo, hi):
    packed = txl.local_scalar('uint32')
    txl.ptx.cvt.rn.bf16x2.f32(packed, hi, lo)
    return packed


def unpack_bf16x2(packed, lo, hi):
    low, high = txl.cast(txl.bitwise_and(packed, txl.uint32(65535)), 'uint16'), txl.cast(txl.shift_right(packed, txl.uint32(16)), 'uint16')
    txl.ptx.cvt.f32.bf16(lo, low)
    txl.ptx.cvt.f32.bf16(hi, high)


def tf32_operand(x, mode):
    """
    Operand conversion of an fp32 value fed to a TF32 tensor-core dot.
    """
    if mode == 'trunc':
        return txl.reinterpret(txl.bitwise_and(txl.reinterpret(x, ty='uint32'), txl.uint32(4294959104)), ty='float32')
    if mode == 'rna':
        bits = txl.local_scalar('uint32')
        txl.ptx.cvt.rna.tf32.f32(bits, x)
        return txl.reinterpret(bits, ty='float32')
    return x


def mma_bf16_m16n8k16(acc, base, a, b0, b1):
    txl.ptx['mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32'](acc[base], acc[base + 1], acc[base + 2], acc[base + 3], a[0], a[1], a[2], a[3], b0, b1, acc[base], acc[base + 1], acc[base + 2], acc[base + 3])


def mma_tf32_m16n8k8(acc, base, a, b0, b1):
    txl.ptx['mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32'](acc[base], acc[base + 1], acc[base + 2], acc[base + 3], a[0], a[1], a[2], a[3], b0, b1, acc[base], acc[base + 1], acc[base + 2], acc[base + 3])


def ldmatrix_x4(dst, ptr, trans=False):
    if trans:
        txl.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(dst[0], dst[1], dst[2], dst[3], ptr)
    else:
        txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(dst[0], dst[1], dst[2], dst[3], ptr)


def swizzle_chunk(row, chunk):
    if isinstance(row, int) and isinstance(chunk, int):
        return chunk ^ row & 7
    return txl.bitwise_xor(chunk, txl.bitwise_and(row, 7))


def sw256_offset(row, chunk):
    """
    Byte offset in a [rows x 256B] bf16 tile (16 x 16B chunks/row), 128B XOR swizzle.
    """
    return row * 256 + swizzle_chunk(row, chunk) * 16


def sw128_offset(row, chunk):
    """
    Byte offset in a [rows x 128B] bf16 tile (8 x 16B chunks/row), 128B XOR swizzle.
    """
    return row * 128 + swizzle_chunk(row, chunk) * 16


def umma128_offset(row, chunk):
    """
    Byte offset in a [128 x 128] bf16 UMMA tile made of two SW128 row atoms.
    """
    return chunk // 8 * 16384 + row * 128 + swizzle_chunk(row, chunk % 8) * 16


def umma64_offset(row, chunk):
    """
    Byte offset in a [64 x 128] bf16 UMMA tile made of two SW128 row atoms.
    """
    return chunk // 8 * 8192 + row * 128 + swizzle_chunk(row, chunk % 8) * 16


def stmatrix_acc(smem, addr, acc, n_tiles, r0, lane):
    """
    Store mma-layout bf16 pairs for 16 rows and n_tiles * 8 columns with stmatrix.x4.
    """
    matrix_idx, matrix_row = lane // 8, lane % 8
    row = r0 + matrix_row + 8 * (matrix_idx % 2)
    for pair in range(n_tiles // 2):
        regs = [pack_bf16x2(acc[(2 * pair + j // 2) * 4 + (j % 2) * 2], acc[(2 * pair + j // 2) * 4 + (j % 2) * 2 + 1]) for j in range(4)]
        txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(smem.ptr_to([addr(row, 2 * pair + matrix_idx // 2)]), regs[0], regs[1], regs[2], regs[3])


def zero_acc(acc, n):
    for i in range(n):
        txl.assign(acc[i], txl.float32(0.0))


def all_of(*conds):
    out = conds[0]
    for cond in conds[1:]:
        out = txl.And(out, cond)
    return out


def build_zero_payload(HV):
    """
    Zero an unused (HV, K, V + K) fp32 all-gather payload, so every returned tensor is defined.
    """

    @txl.kernel()
    def zero_payload(p: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1)):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[HV, 1], block=128))
        head = txl.cta_id()[0]
        tidx = txl.thread_id()
        for it in range(HEAD_DIM * 2 * HEAD_DIM // 4 // 128):
            zero = txl.uint32(0)
            txl.ptx.st.global_.v4.b32(p.ptr_to([head * (HEAD_DIM * 2 * HEAD_DIM) + (it * 128 + tidx) * 4]), zero, zero, zero, zero)
    return zero_payload


def chunk_lookup(cu, nseg, chunk_idx):
    """
    Map global chunk chunk_idx to (chunk_start, num_valid): its first token and valid row count.
    """
    chunk_start = txl.local_scalar('int32', init=txl.int32(0))
    num_valid = txl.local_scalar('int32', init=txl.int32(0))
    chunks_before = txl.local_scalar('int32', init=txl.int32(0))
    with txl.serial(nseg) as seg_idx:
        seq_start, seq_end = ld_global_s32(cu.ptr_to([seg_idx])), ld_global_s32(cu.ptr_to([seg_idx + 1]))
        seq_chunks = (seq_end - seq_start + 63) // 64
        with txl.if_(all_of(chunk_idx >= chunks_before, chunk_idx < chunks_before + seq_chunks)), txl.then_():
            txl.assign(chunk_start, seq_start + (chunk_idx - chunks_before) * 64)
            txl.assign(num_valid, txl.min(seq_end - seq_start - (chunk_idx - chunks_before) * 64, 64))
        txl.assign(chunks_before, chunks_before + seq_chunks)
    return (chunk_start, num_valid)


# Per-kernel layout constants: *_S are shared-memory byte offsets, *_TM_* TMEM columns, *_B_*
# mbarrier indices and *_ID_* UMMA instruction descriptors.
INTRA2_K_S = 0
INTRA2_Q_S = 16 * KIB
INTRA2_V_S = 32 * KIB
INTRA2_KB_S = 48 * KIB
INTRA2_AF_S = 0
INTRA2_AQ_S = 16 * KIB
INTRA2_AI_S0 = 64 * KIB
INTRA2_AB_S0 = 80 * KIB
INTRA2_SCR_S0 = 88 * KIB
INTRA2_G_S0 = 92 * KIB


def build_intra2(HV, tf32_mode, kperm=False, diet=False, fsw=False, tg=False, persistent=False):
    """
    The intra-chunk kernel: cumulative gate, WY matrix A, w, u and the masked q k^T per chunk.
    diet: 72.5 KB smem (3 CTAs / SM): Aqk stored from registers, the inverse in the dead q tile,
    per-warp merge scratch then bf16 A in one 8 KB region.
    """
    if diet:
        AI_S, SCR_S, AB_S = (INTRA2_Q_S, 64 * KIB, 64 * KIB)
        G_S = 72 * KIB
    else:
        AI_S, SCR_S, AB_S, G_S = (INTRA2_AI_S0, INTRA2_SCR_S0, INTRA2_AB_S0, INTRA2_G_S0)
    B_S = G_S + 256
    SMEM = B_S + 256
    BARS_S = SMEM
    HOLD_S = SMEM + 64
    if tg:
        SMEM += 128
    sA = umma64_offset if tg else sw256_offset

    def prepare_host(params):
        maps = []
        for name in ('q', 'k', 'v'):
            desc = txl.stack_alloca('tensormap', 1)
            txl.cu_tensor_map_encode_tiled(desc, 'bfloat16', 3, params[name].data, 64, params['S'], 2 * HV, HV * 256, 128, 64, 64, 2, 1, 1, 1, 0, 3, 2, 0)
            maps.append(desc)
        return tuple(maps)

    @txl.kernel()
    def intra2(q: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), k: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), v: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), g: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), beta: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), nseg: txl.int32, n_chunks: txl.int32, g_cum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), a_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), u_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), aqk_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), payload: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), zero_payload: txl.int32, num_ctas: txl.int32, S: txl.int32):
        host = prepare_host(dict(q=q, k=k, v=v, S=S))
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[num_ctas] if persistent else [n_chunks, HV], block=4 * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=3) if diet else None)
        q_map, k_map, v_map = host
        smem = txl.alloc_tensor((SMEM,), txl.u8, scope='shared.dyn', align=1024)
        cta = txl.cta_id()
        tile_parity = txl.local_scalar('int32', init=txl.int32(0))
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        lane_group, lane_in_group = lane // 4, lane % 4
        r0 = warp * 16
        scr = SCR_S + warp * 1024
        with txl.if_(txl.And(zero_payload != 0, tidx < 32)), txl.then_():
            zi = txl.local_scalar('int32', init=cta * 32 + tidx)
            zstride = num_ctas * 32
            with txl.while_(zi < HV * 128 * 64):
                txl.ptx.st.global_.v4.f32(payload.ptr_to([zi * 4]), txl.float32(0.0), txl.float32(0.0), txl.float32(0.0), txl.float32(0.0))
                txl.assign(zi, zi + zstride)

        def shfl(x, src):
            r = txl.local_scalar('uint32')
            txl.ptx.shfl_sync.idx.b32(r, txl.reinterpret(x, ty='uint32'), txl.cast(src, 'uint32'), txl.uint32(31), txl.uint32(4294967295))
            return txl.reinterpret(r, ty='float32')

        def xr(a, b):
            return a ^ b if isinstance(a, int) and isinstance(b, int) else txl.bitwise_xor(a, b)

        def fa(base, row, col, stride):
            """
            Byte address of fp32 (row, col) in a [.][stride] fp32 tile. fsw: 16 B chunks
            XOR-swizzled by row % 8 (stride 64) / (row // 2) % 4 (stride 16) so the 8 fragment rows
            of a warp hit distinct banks.
            """
            if not fsw:
                return base + (row * stride + col) * 4
            sel = row % 8 if stride == 64 else row // 2 % 4
            return base + row * (stride * 4) + xr(col // 4, sel) * 16 + col % 4 * 4

        def tf32_mm(acc, terms, perm=False):
            """
            acc(16x16, C frags of 2 n-tiles) += sum of tf32(X) @ tf32(Y); X, Y are fp32 smem tiles
            given as (byte offset, row stride in floats). perm: Triton's kWidth=2 operand encoding
            (FLA's inner merge dots, read from its PTX): MMA ks takes the k of parity ks; slot
            lane_in_group carries k = 2 lane_in_group + ks and slot lane_in_group + 4 carries
            k = 8 + 2 lane_in_group + ks.
            """
            dk = 8 if perm else 4
            for (xb, xr0, xc0, xs), (yb, yr0, yc0, ys) in terms:
                for ks in range(2):
                    kb_ = 2 * lane_in_group + ks if perm else ks * 8 + lane_in_group
                    a = txl.alloc_local((4,), 'uint32')
                    for j, (dr, dc) in enumerate(((0, 0), (8, 0), (0, dk), (8, dk))):
                        txl.assign(a[j], txl.reinterpret(tf32_operand(ld_shared_f32(smem, fa(xb, xr0 + lane_group + dr, xc0 + kb_ + dc, xs)), tf32_mode), ty='uint32'))
                    for nt in range(2):
                        b0 = txl.reinterpret(tf32_operand(ld_shared_f32(smem, fa(yb, yr0 + kb_, yc0 + nt * 8 + lane_group, ys)), tf32_mode), ty='uint32')
                        b1 = txl.reinterpret(tf32_operand(ld_shared_f32(smem, fa(yb, yr0 + kb_ + dk, yc0 + nt * 8 + lane_group, ys)), tf32_mode), ty='uint32')
                        mma_tf32_m16n8k8(acc, nt * 4, a, b0, b1)

        def st_c16(acc, tile, neg):
            tb, tr0, tc0, stride = tile
            for nt in range(2):
                for e in range(4):
                    x = acc[nt * 4 + e]
                    st_shared_f32(smem, fa(tb, tr0 + lane_group + 8 * (e // 2), tc0 + nt * 8 + 2 * lane_in_group + e % 2, stride), -x if neg else x)

        def warp_sync():
            txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
        if tg:
            with txl.if_(tidx == 0), txl.then_():
                for i_ in range(5):
                    txl.ptx.mbarrier.init.shared.b64(smem.ptr_to([BARS_S + i_ * 8]), txl.uint32(1))
            txl.ptx.fence.mbarrier_init.release.cluster()
            with txl.if_(warp == 0), txl.then_():
                txl.ptx['tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32'](smem.ptr_to([HOLD_S]), txl.uint32(128))
                txl.ptx['tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned']()
            txl.ptx['tcgen05.fence::before_thread_sync']()
            txl.cuda.cta_sync()
            txl.ptx['tcgen05.fence::after_thread_sync']()
            tmem_base = txl.local_scalar('uint32')
            txl.ptx.ld.volatile.shared.u32(tmem_base, smem.ptr_to([HOLD_S]))

            def udesc(off, base):
                addr = txl.cuda.cvta_generic_to_shared(smem.ptr_to([off]))
                return txl.bitwise_or(txl.uint64(base), txl.cast(txl.bitwise_and(txl.shift_right(addr, txl.uint32(4)), txl.uint32(16383)), 'uint64'))

            def issue_kk(d_col, a_off, b_off, bar_i):
                """
                D[64 x 64] = A[64 x 128] @ B[64 x 128]^T, both umma64 K-major, with the K steps
                in increasing order.
                """
                with txl.if_(warp == 0), txl.then_():
                    leader = txl.cuda.elect_sync()
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    txl.ptx['fence.proxy.async.shared::cta']()
                    a_d, b_d = udesc(a_off, UMMA_DESC_K), udesc(b_off, UMMA_DESC_K)
                    for kp in range(8):
                        txl.ptx[UMMA_F16](txl.cast(tmem_base + d_col, 'uint32'), a_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), b_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), txl.uint32(BGRAD2_ID_64x64_KK), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                    txl.ptx[UMMA_COMMIT](smem.ptr_to([BARS_S + bar_i * 8]), pred=leader)

            def issue_ab(d_col, b_off, bar_i):
                """
                D[64 x 128] = A[64 x 64] (sw128 K-major bf16 A) @ B[64 tok][128] (umma64, MN-major),
                K = 64.
                """
                with txl.if_(warp == 0), txl.then_():
                    leader = txl.cuda.elect_sync()
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    txl.ptx['fence.proxy.async.shared::cta']()
                    a_d, b_d = udesc(AB_S, UMMA_DESC_K), udesc(b_off, UMMA_DESC_MN)
                    for kp in range(4):
                        txl.ptx[UMMA_F16](txl.cast(tmem_base + d_col, 'uint32'), a_d + txl.uint64(kp * 2), b_d + txl.uint64(kp * 128), txl.uint32(BGRAD2_ID_64x128_KMN), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                    txl.ptx[UMMA_COMMIT](smem.ptr_to([BARS_S + bar_i * 8]), pred=leader)

            def tm_ld(dst, d_col, bar_i, fence=True):
                """
                Rows r0.. (lanes 32 warp ..), 64 columns from d_col of a finished M = 64
                accumulator, mma.sync C order.
                """
                ready = txl.local_scalar('uint32', init=txl.uint32(0))
                with txl.while_(ready == txl.uint32(0)):
                    txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready, smem.ptr_to([BARS_S + bar_i * 8]), txl.cast(tile_parity, 'uint32'))
                if fence:
                    txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                raw = txl.alloc_local((32,), 'uint32')
                txl.ptx[TMEM_LD_16X256B_X8](*[raw[i] for i in range(32)], txl.cast(tmem_base + txl.shift_left(txl.cast(warp * 32, 'uint32'), txl.uint32(16)) + d_col, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                for i in range(32):
                    txl.assign(dst[i], txl.reinterpret(raw[i], ty='float32'))
        def tile(c_in, head_in):
            c, head = txl.local_scalar('int32', init=c_in), txl.local_scalar('int32', init=head_in)
            chunk_start, num_valid = chunk_lookup(cu, nseg, c)
            with txl.if_(num_valid > 0), txl.then_():
                with txl.if_(tidx == 0), txl.then_():
                    txl.ptx['fence.proxy.async.shared::cta']()
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(smem.ptr_to([BARS_S + 32]), txl.uint32(3 * 16384))
                    for tmap, dst in ((k_map, INTRA2_K_S), (q_map, INTRA2_Q_S), (v_map, INTRA2_V_S)):
                        txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([dst]), txl.address_of(tmap), txl.int32(0), chunk_start, txl.cast(2 * head, 'int32'), smem.ptr_to([BARS_S + 32]), txl.uint64(0x12F0000000000000))
                with txl.if_(tidx < 64), txl.then_():
                    gv, bv = txl.local_scalar('float32', init=txl.float32(0.0)), txl.local_scalar('float32', init=txl.float32(0.0))
                    with txl.if_(tidx < num_valid), txl.then_():
                        txl.assign(gv, ld_global_f32(g.ptr_to([(chunk_start + tidx) * HV + head])))
                        txl.assign(bv, ld_global_bf16_as_f32(beta.ptr_to([(chunk_start + tidx) * HV + head])))
                    st_shared_f32(smem, G_S + tidx * 4, gv)
                    st_shared_f32(smem, B_S + tidx * 4, bv)
                txl.cuda.cta_sync()
                with txl.if_(warp == 0), txl.then_():
                    x0 = txl.local_scalar('float32', init=ld_shared_f32(smem, G_S + lane * 4))
                    x1 = txl.local_scalar('float32', init=ld_shared_f32(smem, G_S + 128 + lane * 4))
                    for off in (1, 2, 4, 8, 16):
                        for x in (x0, x1):
                            up = txl.local_scalar('uint32')
                            txl.ptx.shfl_sync.up.b32(up, txl.reinterpret(x, ty='uint32'), txl.uint32(off), txl.uint32(0), txl.uint32(4294967295))
                            txl.assign(x, txl.if_then_else(lane >= off, txl.reinterpret(up, ty='float32') + x, x))
                    tot = shfl(x0, 31)
                    st_shared_f32(smem, G_S + lane * 4, x0 * txl.float32(RCP_LN2))
                    st_shared_f32(smem, G_S + 128 + lane * 4, (x1 + tot) * txl.float32(RCP_LN2))
                txl.cuda.cta_sync()
                with txl.if_(tidx < num_valid), txl.then_():
                    txl.ptx.st.global_.f32(g_cum.ptr_to([(chunk_start + tidx) * HV + head]), ld_shared_f32(smem, G_S + tidx * 4))
                with txl.if_(tidx == 0), txl.then_():
                    ready_tma = txl.local_scalar('uint32', init=txl.uint32(0))
                    with txl.while_(ready_tma == txl.uint32(0)):
                        txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready_tma, smem.ptr_to([BARS_S + 32]), txl.cast(tile_parity, 'uint32'))
                txl.cuda.cta_sync()
                for it in range(8):
                    idx = it * 128 + tidx
                    row, ch = idx // 16, idx % 16
                    off = sA(row, ch)
                    kw, vw = txl.alloc_local((4,), 'uint32'), txl.alloc_local((4,), 'uint32')
                    txl.ptx.ld.shared.v4.b32(kw[0], kw[1], kw[2], kw[3], smem.ptr_to([INTRA2_K_S + off]))
                    txl.ptx.ld.shared.v4.b32(vw[0], vw[1], vw[2], vw[3], smem.ptr_to([INTRA2_V_S + off]))
                    b_r = ld_shared_f32(smem, B_S + row * 4)
                    eg = exp2_approx(ld_shared_f32(smem, G_S + row * 4))
                    kb, vb = txl.alloc_local((4,), 'uint32'), txl.alloc_local((4,), 'uint32')
                    bb2 = txl.local_scalar('uint32', init=pack_bf16x2(b_r, b_r))
                    for j in range(4):
                        lo, hi, kbb = txl.local_scalar('float32'), txl.local_scalar('float32'), txl.local_scalar('uint32')
                        txl.ptx['mul.rn.bf16x2'](kbb, kw[j], bb2)
                        unpack_bf16x2(kbb, lo, hi)
                        kb1 = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](kb1, txl.cuda.make_float2(lo, hi), txl.cuda.make_float2(eg, eg))
                        txl.assign(kb[j], pack_bf16x2(txl.cuda.float2_x(kb1), txl.cuda.float2_y(kb1)))
                        txl.ptx['mul.rn.bf16x2'](vb[j], vw[j], bb2)
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([INTRA2_KB_S + off]), kb[0], kb[1], kb[2], kb[3])
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([INTRA2_V_S + off]), vb[0], vb[1], vb[2], vb[3])
                txl.cuda.cta_sync()
                acc, aq = txl.alloc_local((32,), 'float32'), txl.alloc_local((32,), 'float32')
                if tg:
                    issue_kk(0, INTRA2_K_S, INTRA2_K_S, 0)
                    issue_kk(64, INTRA2_Q_S, INTRA2_K_S, 1)
                    tm_ld(acc, 0, 0, fence=False)
                    tm_ld(aq, 64, 1)
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                else:
                    zero_acc(acc, 32)
                    zero_acc(aq, 32)
                    for kk in range(8):
                        a = txl.alloc_local((4,), 'uint32')
                        ldmatrix_x4(a, smem.ptr_to([INTRA2_K_S + sw256_offset(r0 + lane % 8 + 8 * (lane // 8 % 2), 2 * kk + lane // 16)]))
                        aqf = txl.alloc_local((4,), 'uint32')
                        ldmatrix_x4(aqf, smem.ptr_to([INTRA2_Q_S + sw256_offset(r0 + lane % 8 + 8 * (lane // 8 % 2), 2 * kk + lane // 16)]))
                        for np_ in range(4):
                            b = txl.alloc_local((4,), 'uint32')
                            ldmatrix_x4(b, smem.ptr_to([INTRA2_K_S + sw256_offset(np_ * 16 + lane % 8 + 8 * (lane // 16), 2 * kk + lane // 8 % 2)]))
                            mma_bf16_m16n8k16(acc, 2 * np_ * 4, a, b[0], b[1])
                            mma_bf16_m16n8k16(acc, (2 * np_ + 1) * 4, a, b[2], b[3])
                            mma_bf16_m16n8k16(aq, 2 * np_ * 4, aqf, b[0], b[1])
                            mma_bf16_m16n8k16(aq, (2 * np_ + 1) * 4, aqf, b[2], b[3])
                txl.cuda.cta_sync()
                gi_r = [txl.local_scalar('float32', init=ld_shared_f32(smem, G_S + (r0 + lane_group + 8 * h_) * 4)) for h_ in range(2)]
                bi_r = [txl.local_scalar('float32', init=ld_shared_f32(smem, B_S + (r0 + lane_group + 8 * h_) * 4)) for h_ in range(2)]
                gj_r = [txl.local_scalar('float32', init=ld_shared_f32(smem, G_S + (nt_ * 8 + 2 * lane_in_group + e_) * 4)) for nt_ in range(8) for e_ in range(2)]
                for nt in range(8):
                    for half in range(2):
                        i = r0 + lane_group + 8 * half
                        j = nt * 8 + 2 * lane_in_group
                        gi = gi_r[half]
                        egd0, egd1 = exp2_approx(gi - gj_r[nt * 2]), exp2_approx(gi - gj_r[nt * 2 + 1])
                        gate0 = txl.if_then_else(all_of(i > j, i < num_valid, j < num_valid), egd0, txl.float32(0.0))
                        gate1 = txl.if_then_else(all_of(i > j + 1, i < num_valid, j + 1 < num_valid), egd1, txl.float32(0.0))
                        base = nt * 4 + half * 2
                        af0, af1 = txl.local_scalar('uint64'), txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](af0, txl.cuda.make_float2(acc[base], acc[base + 1]), txl.cuda.make_float2(gate0, gate1))
                        bi = bi_r[half]
                        txl.ptx['mul.rn.f32x2'](af1, af0, txl.cuda.make_float2(bi, bi))
                        st_shared_f32(smem, fa(INTRA2_AF_S, i, j, 64), txl.cuda.float2_x(af1))
                        st_shared_f32(smem, fa(INTRA2_AF_S, i, j + 1, 64), txl.cuda.float2_y(af1))
                        aq2 = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](aq2, txl.cuda.make_float2(aq[base], aq[base + 1]), txl.cuda.make_float2(egd0, egd1))
                        txl.assign(aq[base], txl.if_then_else(all_of(i >= j, i < num_valid, j < num_valid), txl.cuda.float2_x(aq2), txl.float32(0.0)))
                        txl.assign(aq[base + 1], txl.if_then_else(all_of(i >= j + 1, i < num_valid, j + 1 < num_valid), txl.cuda.float2_y(aq2), txl.float32(0.0)))
                for nt in range(8):
                    for half in range(2):
                        i = r0 + lane_group + 8 * half
                        if diet:
                            txl.ptx.st.shared.b32(smem.ptr_to([SCR_S + i * 128 + txl.bitwise_xor(nt, txl.bitwise_and(i, 7)) * 16 + lane_in_group * 4]), pack_bf16x2(aq[nt * 4 + half * 2], aq[nt * 4 + half * 2 + 1]))
                        else:
                            txl.ptx.st.shared.b32(smem.ptr_to([INTRA2_AQ_S + i * 128 + (nt * 8 + 2 * lane_in_group) * 2]), pack_bf16x2(aq[nt * 4 + half * 2], aq[nt * 4 + half * 2 + 1]))
                txl.cuda.cta_sync()
                if diet:
                    for it in range(4):
                        idx = it * 128 + tidx
                        row, ch = idx // 8, idx % 8
                        with txl.if_(row < num_valid), txl.then_():
                            wd = txl.alloc_local((4,), 'uint32')
                            txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([SCR_S + row * 128 + txl.bitwise_xor(ch, txl.bitwise_and(row, 7)) * 16]))
                            txl.ptx.st.global_.v4.b32(aqk_out.ptr_to([((chunk_start + row) * HV + head) * 64 + ch * 8]), wd[0], wd[1], wd[2], wd[3])
                d0 = r0
                X = txl.alloc_local((16,), 'float32')
                colc = lane % 16
                for r in range(16):
                    txl.assign(X[r], -ld_shared_f32(smem, fa(INTRA2_AF_S, d0 + r, d0 + colc, 64)))

                def fma_rn(a_, b_, c_):
                    r_ = txl.local_scalar('float32')
                    txl.ptx['fma.rn.f32'](r_, a_, b_, c_)
                    return r_

                def mul_rn(a_, b_):
                    r_ = txl.local_scalar('float32')
                    txl.ptx['mul.rn.f32'](r_, a_, b_)
                    return r_

                def add_rn(a_, b_):
                    r_ = txl.local_scalar('float32')
                    txl.ptx['add.rn.f32'](r_, a_, b_)
                    return r_
                for i in range(2, 16):
                    av = [shfl(X[i], j) if j < i else txl.float32(0.0) for j in range(16)]
                    L = [fma_rn(av[g], X[g], mul_rn(av[g + 8], X[g + 8])) for g in range(8)]
                    s_ = add_rn(add_rn(add_rn(L[0], L[4]), add_rn(L[2], L[6])), add_rn(add_rn(L[1], L[5]), add_rn(L[3], L[7])))
                    a_c = txl.if_then_else(colc < i, X[i], txl.float32(0.0))
                    txl.assign(X[i], add_rn(a_c, s_))
                with txl.if_(lane < 16), txl.then_():
                    for r in range(16):
                        st_shared_f32(smem, fa(AI_S, d0 + r, d0 + colc, 64), X[r] + txl.if_then_else(colc == r, txl.float32(1.0), txl.float32(0.0)))
                for it in range(0 if diet else 4):
                    idx = it * 128 + tidx
                    row, ch = idx // 8, idx % 8
                    with txl.if_(row < num_valid), txl.then_():
                        wd = txl.alloc_local((4,), 'uint32')
                        txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([INTRA2_AQ_S + row * 128 + ch * 16]))
                        txl.ptx.st.global_.v4.b32(aqk_out.ptr_to([((chunk_start + row) * HV + head) * 64 + ch * 8]), wd[0], wd[1], wd[2], wd[3])
                txl.cuda.cta_sync()

                def AF(br, bc):
                    return (INTRA2_AF_S, br * 16, bc * 16, 64)

                def AI(br, bc):
                    return (AI_S, br * 16, bc * 16, 64)

                def SCR():
                    return (scr, 0, 0, 16)
                for d in range(3):
                    with txl.if_(warp == d + 1), txl.then_():
                        p = txl.alloc_local((8,), 'float32')
                        zero_acc(p, 8)
                        tf32_mm(p, [(AI(d + 1, d + 1), AF(d + 1, d))], perm=kperm)
                        st_c16(p, SCR(), False)
                        warp_sync()
                        r = txl.alloc_local((8,), 'float32')
                        zero_acc(r, 8)
                        tf32_mm(r, [(SCR(), AI(d, d))])
                        st_c16(r, AI(d + 1, d), True)
                txl.cuda.cta_sync()
                for br, bc, wsel in ((2, 0, 2), (3, 1, 3)):
                    with txl.if_(warp == wsel), txl.then_():
                        p = txl.alloc_local((8,), 'float32')
                        zero_acc(p, 8)
                        tf32_mm(p, [(AF(br, bc), AI(bc, bc)), (AF(br, bc + 1), AI(bc + 1, bc))], perm=kperm)
                        st_c16(p, SCR(), False)
                        warp_sync()
                        r = txl.alloc_local((8,), 'float32')
                        zero_acc(r, 8)
                        tf32_mm(r, [(AI(br, br), SCR())])
                        st_c16(r, AI(br, bc), True)
                txl.cuda.cta_sync()
                with txl.if_(warp == 3), txl.then_():
                    p = txl.alloc_local((8,), 'float32')
                    zero_acc(p, 8)
                    tf32_mm(p, [(AF(3, 0), AI(0, 0)), (AF(3, 1), AI(1, 0)), (AF(3, 2), AI(2, 0))], perm=kperm)
                    st_c16(p, SCR(), False)
                    warp_sync()
                    r = txl.alloc_local((8,), 'float32')
                    zero_acc(r, 8)
                    tf32_mm(r, [(AI(3, 3), SCR())])
                    st_c16(r, AI(3, 0), True)
                txl.cuda.cta_sync()
                for it in range(4):
                    idx = it * 128 + tidx
                    row, ch = idx // 8, idx % 8
                    words = txl.alloc_local((4,), 'uint32')
                    keep = ch // 2 <= row // 16
                    for j in range(4):
                        cl = ch * 8 + 2 * j
                        lo = txl.if_then_else(keep, ld_shared_f32(smem, fa(AI_S, row, cl, 64)), txl.float32(0.0))
                        hi = txl.if_then_else(keep, ld_shared_f32(smem, fa(AI_S, row, cl + 1, 64)), txl.float32(0.0))
                        txl.assign(words[j], pack_bf16x2(lo, hi))
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([AB_S + sw128_offset(row, ch)]), words[0], words[1], words[2], words[3])
                    with txl.if_(row < num_valid), txl.then_():
                        txl.ptx.st.global_.v4.b32(a_out.ptr_to([((chunk_start + row) * HV + head) * 64 + ch * 8]), words[0], words[1], words[2], words[3])
                txl.cuda.cta_sync()
                if tg:
                    for bi, (src, stg) in enumerate(((INTRA2_V_S, INTRA2_AF_S), (INTRA2_KB_S, AI_S))):
                        if bi:
                            txl.ptx['tcgen05.fence::before_thread_sync']()
                            txl.cuda.cta_sync()
                        issue_ab(0, src, 2 + bi)
                        for c64 in range(2):
                            o = txl.alloc_local((32,), 'float32')
                            tm_ld(o, c64 * 64, 2 + bi)
                            for nt in range(8):
                                for half in range(2):
                                    row = r0 + lane_group + 8 * half
                                    txl.ptx.st.shared.b32(smem.ptr_to([stg + sw256_offset(row, c64 * 8 + nt) + 2 * lane_in_group * 2]), pack_bf16x2(o[nt * 4 + half * 2], o[nt * 4 + half * 2 + 1]))
                else:
                    af = txl.alloc_local((16,), 'uint32')
                    for ks in range(4):
                        tmp = txl.alloc_local((4,), 'uint32')
                        ldmatrix_x4(tmp, smem.ptr_to([AB_S + sw128_offset(r0 + lane % 8 + 8 * (lane // 8 % 2), 2 * ks + lane // 16)]))
                        for j in range(4):
                            txl.assign(af[ks * 4 + j], tmp[j])
                    for src, stg in ((INTRA2_V_S, INTRA2_AF_S), (INTRA2_KB_S, AI_S)):
                        for np_ in range(8):
                            o = txl.alloc_local((8,), 'float32')
                            zero_acc(o, 8)
                            for ks in range(4):
                                b = txl.alloc_local((4,), 'uint32')
                                ldmatrix_x4(b, smem.ptr_to([src + sw256_offset(ks * 16 + lane % 8 + 8 * (lane // 8 % 2), 2 * np_ + lane // 16)]), trans=True)
                                a4 = [af[ks * 4 + j] for j in range(4)]
                                mma_bf16_m16n8k16(o, 0, a4, b[0], b[1])
                                mma_bf16_m16n8k16(o, 4, a4, b[2], b[3])
                            for nt in range(2):
                                for half in range(2):
                                    row = r0 + lane_group + 8 * half
                                    txl.ptx.st.shared.b32(smem.ptr_to([stg + sw256_offset(row, 2 * np_ + nt) + 2 * lane_in_group * 2]), pack_bf16x2(o[nt * 4 + half * 2], o[nt * 4 + half * 2 + 1]))
                txl.cuda.cta_sync()
                for it in range(8):
                    idx = it * 128 + tidx
                    row, ch = idx // 16, idx % 16
                    with txl.if_(row < num_valid), txl.then_():
                        for stg, dst in ((INTRA2_AF_S, u_out), (AI_S, w_out)):
                            wd = txl.alloc_local((4,), 'uint32')
                            txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([stg + sw256_offset(row, ch)]))
                            txl.ptx.st.global_.v4.b32(dst.ptr_to([((chunk_start + row) * HV + head) * 128 + ch * 8]), wd[0], wd[1], wd[2], wd[3])
        ntiles = txl.local_scalar('int32', init=(n_chunks * HV - cta + num_ctas - 1) // num_ctas)
        with txl.serial(ntiles) as it:
            txl.assign(tile_parity, it % 2)
            tl = txl.local_scalar('int32', init=cta + it * num_ctas)
            tile(tl // HV, tl % HV)
            txl.ptx['tcgen05.fence::before_thread_sync']()
            txl.cuda.cta_sync()
            txl.ptx['tcgen05.fence::after_thread_sync']()
        if tg:
            txl.ptx['tcgen05.fence::before_thread_sync']()
            txl.cuda.cta_sync()
            txl.ptx['tcgen05.fence::after_thread_sync']()
            with txl.if_(warp == 0), txl.then_():
                txl.ptx['tcgen05.dealloc.cta_group::1.sync.aligned.b32'](txl.cast(tmem_base, 'uint32'), txl.uint32(128))
    return intra2


def chunk_info(cu, nseg, chunk_idx):
    """
    (chunk_start, num_valid, bos, seg) of global chunk chunk_idx.
    """
    chunk_start = txl.local_scalar('int32', init=txl.int32(0))
    num_valid = txl.local_scalar('int32', init=txl.int32(0))
    bos = txl.local_scalar('int32', init=txl.int32(0))
    seg = txl.local_scalar('int32', init=txl.int32(0))
    chunks_before = txl.local_scalar('int32', init=txl.int32(0))
    with txl.serial(nseg) as seg_idx:
        seq_start, seq_end = ld_global_s32(cu.ptr_to([seg_idx])), ld_global_s32(cu.ptr_to([seg_idx + 1]))
        seq_chunks = (seq_end - seq_start + 63) // 64
        with txl.if_(txl.And(chunk_idx >= chunks_before, chunk_idx < chunks_before + seq_chunks)), txl.then_():
            txl.assign(chunk_start, seq_start + (chunk_idx - chunks_before) * 64)
            txl.assign(num_valid, txl.min(seq_end - seq_start - (chunk_idx - chunks_before) * 64, 64))
            txl.assign(bos, seq_start)
            txl.assign(seg, seg_idx)
        txl.assign(chunks_before, chunks_before + seq_chunks)
    return (chunk_start, num_valid, bos, seg)


class ChunkCursor:
    """
    Walk chunks in global order without rescanning cu, keeping seg, bos, eos and chunk_start
    in local scalars.
    """

    def __init__(self, cu, nseg, c):
        chunk_start, nvc, bos, seg = chunk_info(cu, nseg, c)
        self.cu = cu
        self.nseg = nseg
        self.seg = txl.local_scalar('int32', init=seg)
        self.bos = txl.local_scalar('int32', init=bos)
        self.eos = txl.local_scalar('int32', init=ld_global_s32(cu.ptr_to([seg + 1])))
        self.chunk_start = txl.local_scalar('int32', init=chunk_start)

    def num_valid(self):
        return txl.min(self.eos - self.chunk_start, 64)

    def advance(self):
        txl.assign(self.chunk_start, self.chunk_start + 64)
        with txl.while_(txl.And(self.chunk_start >= self.eos, self.seg < self.nseg - 1)):
            txl.assign(self.seg, self.seg + 1)
            txl.assign(self.bos, self.eos)
            txl.assign(self.eos, ld_global_s32(self.cu.ptr_to([self.seg + 1])))
            txl.assign(self.chunk_start, self.bos)

    def retreat(self):
        txl.assign(self.chunk_start, self.chunk_start - 64)
        with txl.while_(txl.And(self.chunk_start < self.bos, self.seg > 0)):
            txl.assign(self.seg, self.seg - 1)
            txl.assign(self.eos, self.bos)
            txl.assign(self.bos, ld_global_s32(self.cu.ptr_to([self.seg])))
            txl.assign(self.chunk_start, self.bos + ((self.eos - self.bos + 63) // 64 - 1) * 64)


def mn_major_offset(row, ch):
    """
    Byte offset of 16 B chunk ch (0..15) of a row in a [64][128] bf16 MN-major SW128 tile.
    """
    if isinstance(row, int) and isinstance(ch, int):
        return ch // 8 * 8192 + row * 128 + (ch % 8 ^ row % 8) * 16
    return ch // 8 * 8192 + row * 128 + txl.bitwise_xor(ch % 8, txl.bitwise_and(row, 7)) * 16


def merge_block_offset(row, col4):
    """
    Byte offset of fp32 (row, 4 * col4 ..) in the [128][32] fp32 M block, in the layout the TF32
    MN-major descriptor (128 B swizzle, 32 B atoms) reads: 32 B unit (col / 8) XOR (row % 4).
    """
    return row * 128 + txl.bitwise_xor(col4 // 2, txl.bitwise_and(row, 3)) * 32 + col4 % 2 * 16


REC7_NS = 3


def rec7_smem_layout(with_o, vb=32):
    """
    Stage ring (q / aqk only with_o), then HB / VN / VG; the merge-only tf32(h) block MB aliases
    HB..VG, which are first written after the merge has finished.
    """
    L = {}
    off = 0
    if with_o:
        L['Q_S'] = off
        off += 16384
    else:
        L['Q_S'] = 0
    L['K_S'] = off
    L['W_S'] = off + 16384
    off += 32768
    if with_o:
        L['AQ_S'] = off
        off += 8192
    else:
        L['AQ_S'] = 0
    L['U_S'] = off
    L['G_S'] = off + 64 * (2 * vb + 16)
    L['STAGE'] = (L['G_S'] + 256 + 1023) // 1024 * 1024
    L['HB_S'] = REC7_NS * L['STAGE']
    NA = max(1, vb // 64)
    L['VN_S'] = L['HB_S'] + 16384 * NA
    L['VG_S'] = L['VN_S'] + 8192 * NA
    L['MB_S'] = L['HB_S']
    L['BAR_S'] = L['VG_S'] + 8192 * NA
    L['HOLD_S'] = L['BAR_S'] + 256
    L['SMEM'] = L['HOLD_S'] + 64
    return L


REC7_TM_H = 0
REC7_TM_VD = 32
REC7_TM_OI = 64
REC7_TM_OX = 96
REC7_TM_MA = 128
REC7_TM_COLS = 256
REC7_ID_M64 = 67699856
REC7_ID_UPD = 134841488
REC7_ID_MERGE = 134809872
REC7_B_FULL = 0
REC7_B_EMPTY = REC7_NS
REC7_B_HB = 2 * REC7_NS
REC7_B_VD = REC7_B_HB + 1
REC7_B_VN = REC7_B_VD + 1
REC7_B_UPD = REC7_B_VN + 1
REC7_B_MRDY = REC7_B_UPD + 2
REC7_B_MDONE = REC7_B_MRDY + 1
REC7_B_OX = REC7_B_MDONE + 1
REC7_B_OFREE = REC7_B_OX + 2
REC7_CK_CHUNKS = 19   # full chunks of A holes per fp32 checkpoint: 512 groups per head / 28 per (chunk, head)


def checkpoint_base(kt, head, t, vb32, HV):
    """
    bf16 element index in A of piece 0 of the checkpoint group of state row t, 32-col block vb32
    and head (pieces i8 = 0..7 at + i8 * HV * 64). Groups fill the full chunks starting at token kt.
    """
    gi = (head * 128 + t) * 4 + vb32
    ci, rem = gi // (HV * 28), gi % (HV * 28)
    ha, m = rem // 28, rem % 28
    uu = 1 + txl.if_then_else(m >= 1, 1, 0) + txl.if_then_else(m >= 3, 1, 0) + txl.if_then_else(m >= 6, 1, 0) + txl.if_then_else(m >= 10, 1, 0) + txl.if_then_else(m >= 15, 1, 0) + txl.if_then_else(m >= 21, 1, 0)
    r0 = 8 * (m - uu * (uu - 1) // 2)
    return ((kt + ci * 64 + r0) * HV + ha) * 64 + uu * 8


def build_rec7(HV, with_o=True, merge=False, summary=False, store_h=False, hkv=True, hst=False, v16=False, vb=32, nz=1, ckw=False, static_nz=None, pdl=False):
    """
    The state recurrence: rec7m (merge, output, checkpoints) forward, rec_h (h recompute) backward.
    vb: value columns per CTA (NSG = vb / 32 state warpgroups, each the 32-column state WG on its
    own column block; N = vb UMMAs). nz > 1: up to nz sub-chains per (v-block, head) (nzr at run
    time, grid NVB * nzr): sub-chain z covers [b_z, b_{z+1}) with b = (c_lo, ck1, ck2, ck3)[:nzr] +
    (c_hi,), starting from h0 (z = 0), the checkpoint at token kt_z of A (passed as aqk; bit z of
    ckm) or zero. ckw: write the fp32 state entering chunk ck_z (bit z of ckm) to checkpoint kt_z of
    A (passed as h_out). pdl: launched with programmatic dependent launch.
    """
    hst = hst and store_h and not with_o
    v16 = v16 and (hst or not store_h)
    NSG = vb // 32
    NA = max(1, vb // 64)
    assert NSG == 1 or (store_h and hst and v16 and not with_o and not merge and not summary)
    assert nz == 1 or (store_h and not with_o)
    UR = 2 * vb + 16
    NVB = 128 // vb
    TMVD = REC7_TM_VD if vb == 32 else vb
    ID_M64 = REC7_ID_M64 + (vb // 8 - 4) * (1 << 17)
    ID_UPD = REC7_ID_UPD + (vb // 8 - 4) * (1 << 17)
    DESC_HB = UMMA_DESC_MN + (((16384 >> 4) - 512) << 16 if NA == 2 else 0)
    REC7_B_HBF = REC7_B_OFREE + 2
    TM_COLS = 512 if ckw else REC7_TM_COLS
    WM = 4 * NSG + 4
    NWARPS = WM + 3
    L = rec7_smem_layout(with_o, vb)
    Q_S, K_S, W_S, AQ_S, U_S, G_S, STAGE = (L['Q_S'], L['K_S'], L['W_S'], L['AQ_S'], L['U_S'], L['G_S'], L['STAGE'])
    HB_S, VN_S, VG_S, MB_S, BAR_S, HOLD_S, SMEM = (L['HB_S'], L['VN_S'], L['VG_S'], L['MB_S'], L['BAR_S'], L['HOLD_S'], L['SMEM'])
    assert SMEM <= 227 * 1024, SMEM
    NCONS = 3 if with_o else NSG + 1

    @txl.kernel()
    def rec7(q: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), k: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), u: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), aqk: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), g_cum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), nseg: txl.int32, c_lo0: txl.int32, c_hi0: txl.int32, h0: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), use_h0: txl.int32, scale: txl.float32, o: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), h_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), vn_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), summ: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), ag: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), m_first: txl.int32, m_step: txl.int32, m_n: txl.int32, nzr: txl.int32, ck1: txl.int32, ck2: txl.int32, ck3: txl.int32, kt1: txl.int32, kt2: txl.int32, kt3: txl.int32, ckm: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[NVB * nzr, HV] if (nz > 1 and static_nz is None) else [NVB * (static_nz or 1), HV], block=NWARPS * 32, programmatic_stream_serialization=True if pdl else None))
        smem = txl.alloc_tensor((SMEM,), txl.u8, scope='shared.dyn', align=1024)
        cta = txl.cta_id()
        m_n1 = txl.local_scalar('int32', init=txl.if_then_else(m_n > 1, m_n - 1, 0))
        if nz > 1:
            vblk, zc = txl.local_scalar('int32', init=cta[0] % NVB), txl.local_scalar('int32', init=cta[0] // NVB)
            c_lo = txl.local_scalar('int32', init=txl.if_then_else(zc == 0, c_lo0, txl.if_then_else(zc == 1, ck1, txl.if_then_else(zc == 2, ck2, ck3))))
            c_hi = txl.local_scalar('int32', init=txl.if_then_else(zc + 1 == nzr, c_hi0, txl.if_then_else(zc == 0, ck1, txl.if_then_else(zc == 1, ck2, ck3))))
            zkt = txl.local_scalar('int32', init=txl.if_then_else(zc == 1, kt1, txl.if_then_else(zc == 2, kt2, kt3)))
            zck = txl.local_scalar('int32', init=txl.bitwise_and(txl.shift_right(ckm, zc), 1))
        else:
            vblk = cta[0]
            zc = 0
            c_lo = c_lo0
            c_hi = c_hi0
        head = cta[1]
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        nchunks = c_hi - c_lo
        v0 = vblk * vb

        def mbar_addr(i):
            return smem.ptr_to([BAR_S + i * 8])

        def mbar_wait(i, parity):
            ready = txl.local_scalar('uint32', init=txl.uint32(0))
            with txl.while_(ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready, mbar_addr(i), txl.cast(parity, 'uint32'))

        def arrive(i):
            txl.ptx.mbarrier.arrive.shared.b64(mbar_addr(i))

        def wg_sync_arrive_one(bar, t):
            txl.ptx.bar.sync(txl.uint32(1 + warp // 4), txl.uint32(128))
            with txl.if_(t == 0), txl.then_():
                arrive(bar)
        with txl.if_(tidx == 0), txl.then_():
            for s_ in range(REC7_NS):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(REC7_B_FULL + s_), txl.uint32(64))
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(REC7_B_EMPTY + s_), txl.uint32(NCONS))
            for i_, cnt in ((REC7_B_HB, 4 * NSG), (REC7_B_VD, 1), (REC7_B_VN, 4 * NSG), (REC7_B_UPD, 1), (REC7_B_MRDY, 128), (REC7_B_MDONE, 1), (REC7_B_OX, 1), (REC7_B_OX + 1, 1), (REC7_B_OFREE, 128), (REC7_B_OFREE + 1, 128), (REC7_B_HBF, 128)):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(i_), txl.uint32(cnt))
        txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.if_(warp == WM), txl.then_():
            txl.ptx['tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32'](smem.ptr_to([HOLD_S]), txl.uint32(TM_COLS))
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        tmem_base = txl.local_scalar('int32')
        txl.ptx.ld.volatile.shared.s32(tmem_base, smem.ptr_to([HOLD_S]))
        def load_loop():
            """
            Loader warps: copy each chunk's operands into the stage ring with cp.async.
            """
            loader_tidx = tidx - 32 * (WM + 1)
            cur = ChunkCursor(cu, nseg, c_lo)
            with txl.serial(nchunks) as it_c:
                stage = it_c % REC7_NS
                stage_base = stage * STAGE
                with txl.if_(it_c >= REC7_NS), txl.then_():
                    mbar_wait(REC7_B_EMPTY + stage, (it_c // REC7_NS - 1) % 2)
                    txl.ptx['fence.proxy.async.shared::cta']()
                chunk_start, num_valid = txl.local_scalar('int32', init=cur.chunk_start), txl.local_scalar('int32', init=cur.num_valid())
                tiles = ((q, Q_S), (k, K_S), (w, W_S)) if with_o else ((k, K_S), (w, W_S))
                for rep in range(16):
                    idx = rep * 64 + loader_tidx
                    row, ch = idx // 16, idx % 16
                    ok = row < num_valid
                    src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                    for src, dst in tiles:
                        txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + dst + mn_major_offset(row, ch)]), src.ptr_to([(src_row * HV + head) * 128 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                if with_o:
                    for rep in range(8):
                        idx = rep * 64 + loader_tidx
                        row, ch = idx // 8, idx % 8
                        ok = row < num_valid
                        src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                        txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + AQ_S + row * 128 + txl.bitwise_xor(ch, txl.bitwise_and(row, 7)) * 16]), aqk.ptr_to([(src_row * HV + head) * 64 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                for rep in range(4 * NSG):
                    idx = rep * 64 + loader_tidx
                    row, ch = idx // (4 * NSG), idx % (4 * NSG)
                    ok = row < num_valid
                    src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                    txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + U_S + row * UR + ch * 16]), u.ptr_to([(src_row * HV + head) * 128 + v0 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                ok = loader_tidx < num_valid
                src_row = txl.if_then_else(ok, chunk_start + loader_tidx, chunk_start)
                txl.ptx[CP_ASYNC_4](smem.ptr_to([stage_base + G_S + loader_tidx * 4]), g_cum.ptr_to([src_row * HV + head]), 4, txl.cast(txl.if_then_else(ok, 4, 0), 'uint32'))
                txl.ptx['cp.async.mbarrier.arrive.noinc.shared::cta.b64'](mbar_addr(REC7_B_FULL + stage))
                cur.advance()

        with txl.if_(warp >= WM + 1), txl.then_():
            load_loop()
        def mma_loop():
            """
            MMA warp: issue the merge and per-chunk UMMAs of the recurrence.
            """
            leader = txl.cuda.elect_sync()

            def mma(op, d, a, b, idesc, acc):
                txl.ptx[op](txl.cast(d, 'uint32'), a, b, txl.uint32(idesc), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(acc)), pred=leader)
            if merge:
                with txl.serial(m_n1) as j:
                    mbar_wait(REC7_B_MRDY, j % 2)
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    bd = umma_desc(smem, MB_S, UMMA_DESC_MN_TF32)
                    for kp in range(16):
                        mma(UMMA_TF32, tmem_base + REC7_TM_H, txl.cast(tmem_base + REC7_TM_MA + kp * 8, 'uint32'), bd + txl.uint64(kp * 64), REC7_ID_MERGE, 0 if kp == 0 else 1)
                    txl.ptx[UMMA_COMMIT](mbar_addr(REC7_B_MDONE), pred=leader)
            with txl.serial(nchunks) as it_c:
                stage = it_c % REC7_NS
                stage_base = stage * STAGE
                mbar_wait(REC7_B_FULL + stage, it_c // REC7_NS % 2)
                s2 = it_c % 2
                mbar_wait(REC7_B_HB, it_c % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                # with_o: HB(c) lives in half c % 2 of the 128 B swizzle-atom rows (N = 32 bf16
                # fills 64 B), so OI(c) can trail upd(c); HB half c % 2 is rewritten only after
                # UPD(c + 1), committed behind OI(c).
                hb_d = umma_desc(smem, HB_S + (s2 * 64 if with_o else 0), DESC_HB)
                w_d = umma_desc(smem, stage_base + W_S, UMMA_DESC_K)
                for kp in range(8):
                    mma(UMMA_F16, tmem_base + TMVD, w_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), hb_d + txl.uint64(kp * 128), ID_M64, 0 if kp == 0 else 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(REC7_B_VD), pred=leader)
                mbar_wait(REC7_B_VN, it_c % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                k_d, vg_d = umma_desc(smem, stage_base + K_S, UMMA_DESC_MN), umma_desc(smem, VG_S, UMMA_DESC_MN)
                for kp in range(4):
                    mma(UMMA_F16, tmem_base + REC7_TM_H, k_d + txl.uint64(kp * 128), vg_d + txl.uint64(kp * 128), ID_UPD, 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(REC7_B_UPD), pred=leader)
                if with_o:
                    with txl.if_(it_c >= 2), txl.then_():
                        mbar_wait(REC7_B_OFREE + s2, (it_c // 2 - 1) % 2)
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    q_d = umma_desc(smem, stage_base + Q_S, UMMA_DESC_K)
                    for kp in range(8):
                        mma(UMMA_F16, tmem_base + REC7_TM_OI + s2 * 64, q_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), hb_d + txl.uint64(kp * 128), REC7_ID_M64, 0 if kp == 0 else 1)
                    a_d, vn_d = umma_desc(smem, stage_base + AQ_S, UMMA_DESC_K), umma_desc(smem, VN_S, UMMA_DESC_MN)
                    for kp in range(4):
                        mma(UMMA_F16, tmem_base + REC7_TM_OX + s2 * 64, a_d + txl.uint64(kp * 2), vn_d + txl.uint64(kp * 128), REC7_ID_M64, 0 if kp == 0 else 1)
                    txl.ptx[UMMA_COMMIT](mbar_addr(REC7_B_OX + s2), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(REC7_B_EMPTY + stage), pred=leader)

        with txl.if_(warp == WM), txl.then_():
            mma_loop()
        def state_loop():
            """
            State warps: carry the state through the chunks and store h, v_new and the summary.
            """
            t, gq = tidx % 128, tidx // 128
            vq = v0 + 32 * gq
            hbq = HB_S + gq // 2 * 16384
            vnq = VN_S + gq // 2 * 8192
            vgq = VG_S + gq // 2 * 8192
            g4 = 4 * (gq % 2)
            tmem_addr = tmem_base + tmem_lane_offset(t) + 32 * gq
            hv = txl.alloc_local((32,), 'uint32')
            for i in range(32):
                txl.assign(hv[i], txl.uint32(0))
            if merge:
                hb_ = (txl.cast(m_first, 'int64') * HV + head) * (128 * 256)
                with txl.if_(m_n > 0), txl.then_():
                    agv = txl.alloc_local((32,), 'uint32')
                    for i4 in range(8):
                        txl.ptx.ld.global_.nc.v4.b32(agv[4 * i4], agv[4 * i4 + 1], agv[4 * i4 + 2], agv[4 * i4 + 3], ag.ptr_to([hb_ + t * 256 + vq + 4 * i4]))
                    for i in range(32):
                        txl.assign(hv[i], txl.reinterpret(txl.reinterpret(hv[i], ty='float32') + txl.reinterpret(agv[i], ty='float32'), ty='uint32'))
                with txl.serial(m_n1) as jj:
                    base = (txl.cast(m_first + (jj + 1) * m_step, 'int64') * HV + head) * (128 * 256)
                    with txl.if_(jj > 0), txl.then_():
                        mbar_wait(REC7_B_MDONE, (jj - 1) % 2)
                        txl.ptx['tcgen05.fence::after_thread_sync']()
                        txl.ptx[TMEM_LD_32X32B_X32](*[hv[i] for i in range(32)], txl.cast(tmem_addr + REC7_TM_H, 'uint32'))
                        txl.ptx['tcgen05.wait::ld.sync.aligned']()
                        agv = txl.alloc_local((32,), 'uint32')
                        for i4 in range(8):
                            txl.ptx.ld.global_.nc.v4.b32(agv[4 * i4], agv[4 * i4 + 1], agv[4 * i4 + 2], agv[4 * i4 + 3], ag.ptr_to([base - txl.cast(m_step, 'int64') * HV * (128 * 256) + t * 256 + vq + 4 * i4]))
                        for i in range(32):
                            txl.assign(hv[i], txl.reinterpret(txl.reinterpret(hv[i], ty='float32') + txl.reinterpret(agv[i], ty='float32'), ty='uint32'))
                    txl.ptx['fence.proxy.async.shared::cta']()
                    for c4 in range(8):
                        tv = [txl.bitwise_and(hv[c4 * 4 + e], txl.uint32(4294959104)) for e in range(4)]
                        txl.ptx.st.shared.v4.b32(smem.ptr_to([MB_S + merge_block_offset(t, c4)]), tv[0], tv[1], tv[2], tv[3])
                    ma = txl.alloc_local((32,), 'uint32')
                    for q4 in range(4):
                        for i4 in range(8):
                            txl.ptx.ld.global_.nc.v4.b32(ma[4 * i4], ma[4 * i4 + 1], ma[4 * i4 + 2], ma[4 * i4 + 3], ag.ptr_to([base + t * 256 + 128 + q4 * 32 + 4 * i4]))
                        for i in range(32):
                            txl.assign(ma[i], txl.bitwise_and(ma[i], txl.uint32(4294959104)))
                        txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + REC7_TM_MA + q4 * 32, 'uint32'), *[ma[i] for i in range(32)])
                    txl.ptx['tcgen05.wait::st.sync.aligned']()
                    txl.ptx['fence.proxy.async.shared::cta']()
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                    arrive(REC7_B_MRDY)
                with txl.if_(m_n > 1), txl.then_():
                    mbar_wait(REC7_B_MDONE, (m_n - 2) % 2)
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    txl.ptx[TMEM_LD_32X32B_X32](*[hv[i] for i in range(32)], txl.cast(tmem_addr + REC7_TM_H, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    lastb = (txl.cast(m_first + (m_n - 1) * m_step, 'int64') * HV + head) * (128 * 256)
                    agv = txl.alloc_local((32,), 'uint32')
                    for i4 in range(8):
                        txl.ptx.ld.global_.nc.v4.b32(agv[4 * i4], agv[4 * i4 + 1], agv[4 * i4 + 2], agv[4 * i4 + 3], ag.ptr_to([lastb + t * 256 + vq + 4 * i4]))
                    for i in range(32):
                        txl.assign(hv[i], txl.reinterpret(txl.reinterpret(hv[i], ty='float32') + txl.reinterpret(agv[i], ty='float32'), ty='uint32'))
                for i4 in range(8):
                    txl.ptx.st.global_.v4.b32(h0.ptr_to([(head * 128 + t) * 128 + vq + 4 * i4]), hv[4 * i4], hv[4 * i4 + 1], hv[4 * i4 + 2], hv[4 * i4 + 3])
            elif not summary:
                with txl.if_(txl.And(use_h0 != 0, zc == 0) if nz > 1 else use_h0 != 0), txl.then_():
                    for i4 in range(8):
                        txl.ptx.ld.global_.v4.b32(hv[4 * i4], hv[4 * i4 + 1], hv[4 * i4 + 2], hv[4 * i4 + 3], h0.ptr_to([(head * 128 + t) * 128 + vq + 4 * i4]))
            if nz > 1:
                with txl.if_(zck != 0), txl.then_():
                    ckb = txl.local_scalar('int32', init=checkpoint_base(zkt, head, t, vq // 32, HV))
                    for i8 in range(8):
                        txl.ptx.ld.global_.nc.v4.b32(hv[4 * i8], hv[4 * i8 + 1], hv[4 * i8 + 2], hv[4 * i8 + 3], aqk.ptr_to([ckb + i8 * HV * 64]))
            txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + REC7_TM_H, 'uint32'), *[hv[i] for i in range(32)])
            txl.ptx['tcgen05.wait::st.sync.aligned']()
            if ckw:
                cw1 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 2) != 0, ck1 - c_lo, -1))
                cw2 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 4) != 0, ck2 - c_lo, -1))
                cw3 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 8) != 0, ck3 - c_lo, -1))
            cur = ChunkCursor(cu, nseg, c_lo)
            mrow = 16 * (t // 32) + lane % 16
            act = lane < 16
            with txl.serial(nchunks) as it_c:
                stage = it_c % REC7_NS
                stage_base = stage * STAGE
                chunk_start, num_valid = txl.local_scalar('int32', init=cur.chunk_start), txl.local_scalar('int32', init=cur.num_valid())
                reset = txl.And(chunk_start == cur.bos, it_c != 0)
                mbar_wait(REC7_B_FULL + stage, it_c // REC7_NS % 2)
                with txl.if_(it_c >= 1), txl.then_():
                    mbar_wait(REC7_B_UPD, (it_c - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx[TMEM_LD_32X32B_X32](*[hv[i] for i in range(32)], txl.cast(tmem_addr + REC7_TM_H, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                with txl.if_(reset), txl.then_():
                    for i in range(32):
                        txl.assign(hv[i], txl.uint32(0))
                if hst:
                    with txl.if_(it_c >= 1), txl.then_():
                        mbar_wait(REC7_B_HBF, (it_c - 1) % 2)
                hb4 = txl.alloc_local((16,), 'uint32')
                for p in range(16):
                    txl.assign(hb4[p], pack_bf16x2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32')))
                txl.ptx['fence.proxy.async.shared::cta']()
                for c in range(4):
                    hc = c + 4 * (it_c % 2) if with_o else c + g4
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([hbq + t * 128 + txl.bitwise_xor(hc, txl.bitwise_and(t, 7)) * 16]), hb4[4 * c], hb4[4 * c + 1], hb4[4 * c + 2], hb4[4 * c + 3])
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(REC7_B_HB)
                gl = ld_shared_f32(smem, stage_base + G_S + (num_valid - 1) * 4)
                egl = exp2_approx(gl)
                if store_h and not hst:
                    ci = c_lo + it_c
                    for c in range(4):
                        txl.ptx.st.global_.v4.b32(h_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 16384 + umma128_offset(t, vq // 8 + c) // 2]), hb4[4 * c], hb4[4 * c + 1], hb4[4 * c + 2], hb4[4 * c + 3])
                if v16:
                    lane_group, lane_in_group = lane // 4, lane % 4
                    vrow = [16 * (t // 32) + lane_group + 8 * hh for hh in range(2)]
                    frh = [txl.local_scalar('float32', init=txl.if_then_else(vrow[hh] < num_valid, exp2_approx(gl - ld_shared_f32(smem, stage_base + G_S + vrow[hh] * 4)), txl.float32(0.0))) for hh in range(2)]
                    uw2 = txl.alloc_local((8,), 'uint32')
                    for j in range(4):
                        for hh in range(2):
                            txl.ptx.ld.shared.b32(uw2[j * 2 + hh], smem.ptr_to([stage_base + U_S + vrow[hh] * UR + 64 * gq + (8 * j + 2 * lane_in_group) * 2]))
                else:
                    uw = txl.alloc_local((16,), 'uint32')
                    fr = txl.local_scalar('float32', init=txl.float32(0.0))
                    with txl.if_(act), txl.then_():
                        for c in range(4):
                            txl.ptx.ld.shared.v4.b32(uw[4 * c], uw[4 * c + 1], uw[4 * c + 2], uw[4 * c + 3], smem.ptr_to([stage_base + U_S + mrow * UR + 64 * gq + c * 16]))
                        txl.assign(fr, txl.if_then_else(mrow < num_valid, exp2_approx(gl - ld_shared_f32(smem, stage_base + G_S + mrow * 4)), txl.float32(0.0)))
                if ckw:
                    with txl.if_(txl.Or(txl.Or(it_c == cw1, it_c == cw2), it_c == cw3)), txl.then_():
                        txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + 256 + txl.if_then_else(it_c == cw1, 0, txl.if_then_else(it_c == cw2, 32, 64)), 'uint32'), *[hv[i] for i in range(32)])
                for p in range(16):
                    scaled = txl.local_scalar('uint64')
                    txl.ptx['mul.rn.f32x2'](scaled, txl.cuda.make_float2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32')), txl.cuda.make_float2(egl, egl))
                    txl.assign(hv[2 * p], txl.reinterpret(txl.cuda.float2_x(scaled), ty='uint32'))
                    txl.assign(hv[2 * p + 1], txl.reinterpret(txl.cuda.float2_y(scaled), ty='uint32'))
                txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + REC7_TM_H, 'uint32'), *[hv[i] for i in range(32)])
                txl.ptx['tcgen05.wait::st.sync.aligned']()
                mbar_wait(REC7_B_VD, it_c % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                if v16:
                    vd16 = txl.alloc_local((16,), 'uint32')
                    txl.ptx['tcgen05.ld.sync.aligned.16x256b.x4.b32'](*[vd16[i] for i in range(16)], txl.cast(tmem_addr + TMVD, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    if with_o:
                        txl.ptx['fence.proxy.async.shared::cta']()
                    vnf, vgf = txl.alloc_local((16,), 'float32'), txl.alloc_local((16,), 'float32')
                    for j in range(4):
                        for hh in range(2):
                            lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                            unpack_bf16x2(uw2[j * 2 + hh], lo, hi)
                            vn2 = txl.local_scalar('uint64')
                            txl.ptx['sub.rn.f32x2'](vn2, txl.cuda.make_float2(lo, hi), txl.cuda.make_float2(txl.reinterpret(vd16[j * 4 + hh * 2], ty='float32'), txl.reinterpret(vd16[j * 4 + hh * 2 + 1], ty='float32')))
                            vg2 = txl.local_scalar('uint64')
                            txl.ptx['mul.rn.f32x2'](vg2, vn2, txl.cuda.make_float2(frh[hh], frh[hh]))
                            ii = j * 4 + hh * 2
                            txl.assign(vnf[ii], txl.cuda.float2_x(vn2))
                            txl.assign(vnf[ii + 1], txl.cuda.float2_y(vn2))
                            txl.assign(vgf[ii], txl.cuda.float2_x(vg2))
                            txl.assign(vgf[ii + 1], txl.cuda.float2_y(vg2))
                    rbase = 16 * (t // 32)
                    if with_o or hst:
                        stmatrix_acc(smem, lambda row, nt: vnq + row * 128 + txl.bitwise_xor(nt + g4, txl.bitwise_and(row, 7)) * 16, vnf, 4, rbase, lane)
                    stmatrix_acc(smem, lambda row, nt: vgq + row * 128 + txl.bitwise_xor(nt + g4, txl.bitwise_and(row, 7)) * 16, vgf, 4, rbase, lane)
                else:
                    vd = txl.alloc_local((32,), 'uint32')
                    txl.ptx[TMEM_LD_32X32B_X32](*[vd[i] for i in range(32)], txl.cast(tmem_addr + TMVD, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    txl.ptx['fence.proxy.async.shared::cta']()
                    vnb = txl.alloc_local((16,), 'uint32')
                    with txl.if_(act), txl.then_():
                        vgb = txl.alloc_local((16,), 'uint32')
                        for p in range(16):
                            lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                            unpack_bf16x2(uw[p], lo, hi)
                            vn2 = txl.local_scalar('uint64')
                            txl.ptx['sub.rn.f32x2'](vn2, txl.cuda.make_float2(lo, hi), txl.cuda.make_float2(txl.reinterpret(vd[2 * p], ty='float32'), txl.reinterpret(vd[2 * p + 1], ty='float32')))
                            vn0 = txl.cuda.float2_x(vn2)
                            vn1 = txl.cuda.float2_y(vn2)
                            txl.assign(vnb[p], pack_bf16x2(vn0, vn1))
                            vg2 = txl.local_scalar('uint64')
                            txl.ptx['mul.rn.f32x2'](vg2, vn2, txl.cuda.make_float2(fr, fr))
                            txl.assign(vgb[p], pack_bf16x2(txl.cuda.float2_x(vg2), txl.cuda.float2_y(vg2)))
                        for c in range(4):
                            off = mrow * 128 + txl.bitwise_xor(c + g4, txl.bitwise_and(mrow, 7)) * 16
                            if with_o or hst:
                                txl.ptx.st.shared.v4.b32(smem.ptr_to([vnq + off]), vnb[4 * c], vnb[4 * c + 1], vnb[4 * c + 2], vnb[4 * c + 3])
                            txl.ptx.st.shared.v4.b32(smem.ptr_to([vgq + off]), vgb[4 * c], vgb[4 * c + 1], vgb[4 * c + 2], vgb[4 * c + 3])
                txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(REC7_B_VN)
                if store_h and not hst:
                    with txl.if_(txl.And(act, mrow < num_valid)), txl.then_():
                        ci = c_lo + it_c
                        for c in range(4):
                            txl.ptx.st.global_.v4.b32(vn_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 8192 + umma64_offset(mrow, vq // 8 + c) // 2]), vnb[4 * c], vnb[4 * c + 1], vnb[4 * c + 2], vnb[4 * c + 3])
                wg_sync_arrive_one(REC7_B_EMPTY + stage, t)
                cur.advance()
            if summary:
                mbar_wait(REC7_B_UPD, (nchunks - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx[TMEM_LD_32X32B_X32](*[hv[i] for i in range(32)], txl.cast(tmem_addr + REC7_TM_H, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                for i4 in range(8):
                    txl.ptx.st.global_.v4.b32(summ.ptr_to([(head * 128 + t) * 256 + vq + 4 * i4]), hv[4 * i4], hv[4 * i4 + 1], hv[4 * i4 + 2], hv[4 * i4 + 3])

        with txl.if_(warp < 4 * NSG), txl.then_():
            state_loop()
        if hst:
            def store_loop():
                """
                Store warps: copy each chunk's h and v_new from shared memory to global memory.
                """
                u = tidx - 128 * NSG
                cur = ChunkCursor(cu, nseg, c_lo)
                with txl.serial(nchunks) as it_c:
                    _ = txl.local_scalar('int32', init=cur.chunk_start)  # unused; kept so the traced kernel stays the same
                    num_valid = txl.local_scalar('int32', init=cur.num_valid())
                    ci = c_lo + it_c
                    mbar_wait(REC7_B_HB, it_c % 2)
                    for j in range(4 * NSG):
                        p = u + 128 * j
                        row, c = p // (4 * NSG), p % (4 * NSG)
                        wd = txl.alloc_local((4,), 'uint32')
                        txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([HB_S + c // 8 * 16384 + row * 128 + txl.bitwise_xor(c % 8, txl.bitwise_and(row, 7)) * 16]))
                        txl.ptx.st.global_.v4.b32(h_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 16384 + umma128_offset(row, v0 // 8 + c) // 2]), wd[0], wd[1], wd[2], wd[3])
                    mbar_wait(REC7_B_VN, it_c % 2)
                    for j in range(2 * NSG):
                        p = u + 128 * j
                        row, c = p // (4 * NSG), p % (4 * NSG)
                        wd = txl.alloc_local((4,), 'uint32')
                        for e in range(4):
                            txl.assign(wd[e], txl.uint32(0))
                        with txl.if_(row < num_valid), txl.then_():
                            txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([VN_S + c // 8 * 8192 + row * 128 + txl.bitwise_xor(c % 8, txl.bitwise_and(row, 7)) * 16]))
                        txl.ptx.st.global_.v4.b32(vn_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 8192 + umma64_offset(row, v0 // 8 + c) // 2]), wd[0], wd[1], wd[2], wd[3])
                    arrive(REC7_B_HBF)
                    cur.advance()

            with txl.if_(txl.And(warp >= 4 * NSG, warp < 4 * NSG + 4)), txl.then_():
                store_loop()
        if with_o:
            def output_loop():
                """
                Output warps: finish o from the output accumulators and write the checkpoints.
                """
                t = tidx - 128
                tmem_addr = tmem_base + tmem_lane_offset(t)
                mrow = 16 * (t // 32) + lane % 16
                act = lane < 16
                if ckw:
                    ow1 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 2) != 0, ck1 - c_lo, -1))
                    ow2 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 4) != 0, ck2 - c_lo, -1))
                    ow3 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 8) != 0, ck3 - c_lo, -1))
                cur = ChunkCursor(cu, nseg, c_lo)
                with txl.serial(nchunks) as it_c:
                    stage = it_c % REC7_NS
                    stage_base = stage * STAGE
                    chunk_start, num_valid = txl.local_scalar('int32', init=cur.chunk_start), txl.local_scalar('int32', init=cur.num_valid())
                    s2 = it_c % 2
                    mbar_wait(REC7_B_OX + s2, it_c // 2 % 2)
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    oi, ox = txl.alloc_local((32,), 'uint32'), txl.alloc_local((32,), 'uint32')
                    txl.ptx[TMEM_LD_32X32B_X32](*[oi[i] for i in range(32)], txl.cast(tmem_addr + REC7_TM_OI + s2 * 64, 'uint32'))
                    txl.ptx[TMEM_LD_32X32B_X32](*[ox[i] for i in range(32)], txl.cast(tmem_addr + REC7_TM_OX + s2 * 64, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                    arrive(REC7_B_OFREE + s2)
                    eg = txl.local_scalar('float32', init=exp2_approx(ld_shared_f32(smem, stage_base + G_S + mrow * 4)))
                    wg_sync_arrive_one(REC7_B_EMPTY + stage, t)
                    with txl.if_(txl.And(act, mrow < num_valid)), txl.then_():
                        ob = txl.alloc_local((16,), 'uint32')
                        for p in range(16):
                            oi2, ox2, out2 = txl.local_scalar('uint64'), txl.local_scalar('uint64'), txl.local_scalar('uint64')
                            txl.ptx['mul.rn.f32x2'](oi2, txl.cuda.make_float2(txl.reinterpret(oi[2 * p], ty='float32'), txl.reinterpret(oi[2 * p + 1], ty='float32')), txl.cuda.make_float2(eg, eg))
                            txl.ptx['mul.rn.f32x2'](ox2, txl.cuda.make_float2(txl.reinterpret(ox[2 * p], ty='float32'), txl.reinterpret(ox[2 * p + 1], ty='float32')), txl.cuda.make_float2(scale, scale))
                            txl.ptx['fma.rn.f32x2'](out2, txl.cuda.make_float2(scale, scale), oi2, ox2)
                            txl.assign(ob[p], pack_bf16x2(txl.cuda.float2_x(out2), txl.cuda.float2_y(out2)))
                        for c in range(4):
                            txl.ptx.st.global_.v4.b32(o.ptr_to([((chunk_start + mrow) * HV + head) * 128 + v0 + c * 8]), ob[4 * c], ob[4 * c + 1], ob[4 * c + 2], ob[4 * c + 3])
                    if ckw:
                        with txl.if_(txl.Or(txl.Or(it_c == ow1, it_c == ow2), it_c == ow3)), txl.then_():
                            ckv = txl.alloc_local((32,), 'uint32')
                            txl.ptx[TMEM_LD_32X32B_X32](*[ckv[i] for i in range(32)], txl.cast(tmem_addr + 256 + txl.if_then_else(it_c == ow1, 0, txl.if_then_else(it_c == ow2, 32, 64)), 'uint32'))
                            txl.ptx['tcgen05.wait::ld.sync.aligned']()
                            ckb = txl.local_scalar('int32', init=checkpoint_base(txl.if_then_else(it_c == ow1, kt1, txl.if_then_else(it_c == ow2, kt2, kt3)), head, t, v0 // 32, HV))
                            for i8 in range(8):
                                txl.ptx.st.global_.v4.b32(h_out.ptr_to([ckb + i8 * HV * 64]), ckv[4 * i8], ckv[4 * i8 + 1], ckv[4 * i8 + 2], ckv[4 * i8 + 3])
                    cur.advance()

            with txl.if_(txl.And(warp >= 4, warp < 8)), txl.then_():
                output_loop()
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        with txl.if_(warp == WM), txl.then_():
            txl.ptx['tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned']()
            txl.ptx['tcgen05.dealloc.cta_group::1.sync.aligned.b32'](txl.cast(tmem_base, 'uint32'), txl.uint32(TM_COLS))
        if nz > 1:
            # PDL-launched behind bsum7: no data dependency, but the grid must not complete before
            # its prerequisite; only the last CTA (dispatched last) waits. No-op without PDL.
            with txl.if_(txl.And(cta[0] == NVB * nzr - 1, cta[1] == HV - 1)), txl.then_():
                txl.ptx.griddepcontrol.wait()
    return rec7


FSUM7_NS = 3
FSUM7_UR = 80
FSUM7_K_S = 0
FSUM7_W_S = 16384
FSUM7_U_S = 32768
FSUM7_G_S = FSUM7_U_S + 64 * FSUM7_UR
FSUM7_STAGE = (FSUM7_G_S + 256 + 1023) // 1024 * 1024
FSUM7_KG_S = FSUM7_NS * FSUM7_STAGE
FSUM7_MB_S = FSUM7_KG_S + 2 * 16384
FSUM7_HB_S = FSUM7_MB_S + 16384
FSUM7_VG_S = FSUM7_HB_S + 16384
FSUM7_BAR_S = FSUM7_VG_S + 8192
FSUM7_HOLD_S = FSUM7_BAR_S + 256
FSUM7_SMEM = FSUM7_HOLD_S + 64
FSUM7_NWARPS = 16
FSUM7_TM_DMI = 0
FSUM7_TM_A = 128
FSUM7_TM_DM = 384
FSUM7_TM_H = 416
FSUM7_TM_VD = 448
FSUM7_ID_MI = 136414352
FSUM7_ID_CHAIN = 134809872
FSUM7_ID_M64 = 67699856
FSUM7_ID_UPD = 134841488
FSUM7_B_FULL = 0
FSUM7_B_EMPTY = FSUM7_NS
FSUM7_B_KG = 2 * FSUM7_NS
FSUM7_B_MIACC = FSUM7_B_KG + 2
FSUM7_B_MIFREE = FSUM7_B_MIACC + 1
FSUM7_B_AREADY = FSUM7_B_MIFREE + 1
FSUM7_B_AFREE = FSUM7_B_AREADY + 2
FSUM7_B_BREADY = FSUM7_B_AFREE + 2
FSUM7_B_MCACC = FSUM7_B_BREADY + 1
FSUM7_B_HB = FSUM7_B_MCACC + 1
FSUM7_B_VD = FSUM7_B_HB + 1
FSUM7_B_VN = FSUM7_B_VD + 1
FSUM7_B_UPD = FSUM7_B_VN + 1


def build_fsum7(HV, rawneg=False, inplace=False, with_o=False):
    """
    The forward state summary of the rank's last local sequence (fsum7 / fsum7o).
    rawneg: copy D_Mi raw except the diagonal (D_ii - e^gl) and negate A in the chain UMMA (the TF32
    UMMA truncates the low 13 mantissa bits itself, so the result is the same). with_o (needs
    inplace): the segment's rec pass starts from zero, so it is also the REAL pass of that segment:
    emit o for it as well (rec7's oi / ox UMMAs and output epilogue on four extra warps W16-19; q /
    aqk join the ring, which drops to two stages to fit).
    """
    assert inplace or not with_o
    FSUM7_NS = 2 if with_o else 3
    if with_o:
        Q_S, FSUM7_K_S, FSUM7_W_S, AQ_S, FSUM7_U_S = (0, 16384, 32768, 49152, 57344)
    else:
        Q_S, FSUM7_K_S, FSUM7_W_S, AQ_S, FSUM7_U_S = (0, 0, 16384, 0, 32768)
    FSUM7_G_S = FSUM7_U_S + 64 * FSUM7_UR
    FSUM7_STAGE = (FSUM7_G_S + 256 + 1023) // 1024 * 1024
    FSUM7_KG_S = FSUM7_NS * FSUM7_STAGE
    FSUM7_MB_S = FSUM7_KG_S + 2 * 16384
    FSUM7_HB_S = FSUM7_MB_S + 16384
    FSUM7_VG_S = FSUM7_HB_S + 16384
    VN_S = FSUM7_VG_S + 8192
    FSUM7_BAR_S = VN_S + (8192 if with_o else 0)
    FSUM7_HOLD_S = FSUM7_BAR_S + 256
    FSUM7_SMEM = FSUM7_HOLD_S + 64
    assert FSUM7_SMEM <= 227 * 1024, FSUM7_SMEM
    NW = 20 if with_o else FSUM7_NWARPS
    FSUM7_B_FULL = 0
    FSUM7_B_EMPTY = FSUM7_NS
    FSUM7_B_KG = 2 * FSUM7_NS
    FSUM7_B_MIACC = FSUM7_B_KG + 2
    FSUM7_B_MIFREE = FSUM7_B_MIACC + 1
    FSUM7_B_AREADY = FSUM7_B_MIFREE + 1
    FSUM7_B_AFREE = FSUM7_B_AREADY + 2
    FSUM7_B_BREADY = FSUM7_B_AFREE + 2
    FSUM7_B_MCACC = FSUM7_B_BREADY + 1
    FSUM7_B_HB = FSUM7_B_MCACC + 1
    FSUM7_B_VD = FSUM7_B_HB + 1
    FSUM7_B_VN = FSUM7_B_VD + 1
    FSUM7_B_UPD = FSUM7_B_VN + 1
    B_OX = FSUM7_B_UPD + 1
    B_OFREE = B_OX + 1
    TM_OI, TM_OX = (0, 32)
    id_chain = FSUM7_ID_CHAIN | (1 << 13 if rawneg else 0)

    def prepare_host(params):
        maps = []
        for name in ('k', 'w', 'q'):
            desc = txl.stack_alloca('tensormap', 1)
            txl.cu_tensor_map_encode_tiled(desc, 'bfloat16', 3, params[name].data, 64, params['S'], 2 * HV, HV * 256, 128, 64, 64, 2, 1, 1, 1, 0, 3, 2, 0)
            maps.append(desc)
        desc = txl.stack_alloca('tensormap', 1)
        txl.cu_tensor_map_encode_tiled(desc, 'bfloat16', 3, params['aqk'].data, 64, params['S'], HV, HV * 128, 128, 64, 64, 1, 1, 1, 1, 0, 3, 2, 0)
        maps.append(desc)
        return tuple(maps)

    @txl.kernel()
    def fsum7(k: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), u: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), g_cum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), bos: txl.int32, eos: txl.int32, payload: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), q: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), aqk: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), o: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), scale: txl.float32, a_ck: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), ck1: txl.int32, ck2: txl.int32, ck3: txl.int32, kt1: txl.int32, kt2: txl.int32, kt3: txl.int32, ckm: txl.int32, S: txl.int32):
        host = prepare_host(dict(k=k, w=w, q=q, aqk=aqk, S=S))
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[4, HV], block=NW * 32))
        k_map, w_map, q_map, aqk_map = host
        smem = txl.alloc_tensor((FSUM7_SMEM,), txl.u8, scope='shared.dyn', align=1024)
        cta = txl.cta_id()
        cb = cta[0]
        head = cta[1]
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        nch = (eos - bos + 63) // 64
        v0 = cb * 32

        def mbar_addr(i):
            return smem.ptr_to([FSUM7_BAR_S + i * 8])

        def mbar_wait(i, parity):
            ready = txl.local_scalar('uint32', init=txl.uint32(0))
            with txl.while_(ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready, mbar_addr(i), txl.cast(parity, 'uint32'))

        def arrive(i):
            txl.ptx.mbarrier.arrive.shared.b64(mbar_addr(i))

        def group_arrive_one(bar_id, bar, t):
            txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
            txl.ptx.bar.sync(txl.uint32(bar_id), txl.uint32(128))
            with txl.if_(t == 0), txl.then_():
                arrive(bar)
        with txl.if_(tidx == 0), txl.then_():
            for s_ in range(FSUM7_NS):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(FSUM7_B_FULL + s_), txl.uint32(65))
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(FSUM7_B_EMPTY + s_), txl.uint32(5 if with_o else 4))
            for i_, cnt in ((FSUM7_B_KG, 4), (FSUM7_B_KG + 1, 4), (FSUM7_B_MIACC, 1), (FSUM7_B_MIFREE, 1 if inplace else 128), (FSUM7_B_AREADY, 128), (FSUM7_B_AREADY + 1, 128), (FSUM7_B_AFREE, 1), (FSUM7_B_AFREE + 1, 1), (FSUM7_B_BREADY, 4), (FSUM7_B_MCACC, 1), (FSUM7_B_HB, 4), (FSUM7_B_VD, 1), (FSUM7_B_VN, 4), (FSUM7_B_UPD, 1)) + (((B_OX, 1), (B_OFREE, 128)) if with_o else ()):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(i_), txl.uint32(cnt))
        txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.if_(warp == 8), txl.then_():
            txl.ptx['tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32'](smem.ptr_to([FSUM7_HOLD_S]), txl.uint32(512))
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        tmem_base = txl.local_scalar('int32')
        txl.ptx.ld.volatile.shared.s32(tmem_base, smem.ptr_to([FSUM7_HOLD_S]))
        def load_loop():
            """
            Loader warps: copy each chunk's operands into the stage ring.
            """
            loader_tidx = tidx - 288
            with txl.serial(nch) as it:
                stage = it % FSUM7_NS
                stage_base = stage * FSUM7_STAGE
                with txl.if_(it >= FSUM7_NS), txl.then_():
                    mbar_wait(FSUM7_B_EMPTY + stage, (it // FSUM7_NS - 1) % 2)
                    txl.ptx['fence.proxy.async.shared::cta']()
                chunk_start = txl.local_scalar('int32', init=bos + it * 64)
                num_valid = txl.local_scalar('int32', init=txl.min(eos - chunk_start, 64))
                with txl.if_(txl.And(num_valid == 64, loader_tidx == 0)), txl.then_():
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar_addr(FSUM7_B_FULL + stage), txl.uint32(16384 * (4 if with_o else 2) - (8192 if with_o else 0)))
                    for tmap, dst in ((k_map, FSUM7_K_S), (w_map, FSUM7_W_S)) + (((q_map, Q_S),) if with_o else ()):
                        txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([stage_base + dst]), txl.address_of(tmap), txl.int32(0), chunk_start, txl.cast(2 * head, 'int32'), mbar_addr(FSUM7_B_FULL + stage), txl.uint64(0x12F0000000000000))
                    if with_o:
                        txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([stage_base + AQ_S]), txl.address_of(aqk_map), txl.int32(0), chunk_start, txl.cast(head, 'int32'), mbar_addr(FSUM7_B_FULL + stage), txl.uint64(0x12F0000000000000))
                with txl.if_(num_valid < 64), txl.then_():
                    for rep in range(16):
                        idx = rep * 64 + loader_tidx
                        row, ch = idx // 16, idx % 16
                        ok = row < num_valid
                        src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                        for src, dst in ((k, FSUM7_K_S), (w, FSUM7_W_S)) + (((q, Q_S),) if with_o else ()):
                            txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + dst + mn_major_offset(row, ch)]), src.ptr_to([(src_row * HV + head) * 128 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                    if with_o:
                        for rep in range(8):
                            idx = rep * 64 + loader_tidx
                            row, ch = idx // 8, idx % 8
                            ok = row < num_valid
                            src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                            txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + AQ_S + row * 128 + txl.bitwise_xor(ch, txl.bitwise_and(row, 7)) * 16]), aqk.ptr_to([(src_row * HV + head) * 64 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                    with txl.if_(loader_tidx == 0), txl.then_():
                        txl.ptx.mbarrier.arrive.shared.b64(mbar_addr(FSUM7_B_FULL + stage))
                for rep in range(4):
                    idx = rep * 64 + loader_tidx
                    row, ch = idx // 4, idx % 4
                    ok = row < num_valid
                    src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                    txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + FSUM7_U_S + row * FSUM7_UR + ch * 16]), u.ptr_to([(src_row * HV + head) * 128 + v0 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                ok = loader_tidx < num_valid
                src_row = txl.if_then_else(ok, chunk_start + loader_tidx, chunk_start)
                txl.ptx[CP_ASYNC_4](smem.ptr_to([stage_base + FSUM7_G_S + loader_tidx * 4]), g_cum.ptr_to([src_row * HV + head]), 4, txl.cast(txl.if_then_else(ok, 4, 0), 'uint32'))
                txl.ptx['cp.async.mbarrier.arrive.noinc.shared::cta.b64'](mbar_addr(FSUM7_B_FULL + stage))

        with txl.if_(txl.And(warp >= 9, warp < 11)), txl.then_():
            load_loop()
        def chain_mma_loop():
            """
            MMA warp of the M chain: issue the Mi and chain UMMAs.
            """
            leader = txl.cuda.elect_sync()

            def issue_mi(it):
                stage = it % FSUM7_NS
                stage_base = stage * FSUM7_STAGE
                mbar_wait(FSUM7_B_FULL + stage, it // FSUM7_NS % 2)
                mbar_wait(FSUM7_B_KG + it % 2, it // 2 % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                if not inplace:
                    with txl.if_(it >= 1), txl.then_():
                        mbar_wait(FSUM7_B_MIFREE, (it - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                a_d, b_d = umma_desc(smem, FSUM7_KG_S + it % 2 * 16384, UMMA_DESC_MN), umma_desc(smem, stage_base + FSUM7_W_S, UMMA_DESC_MN)
                d_col = FSUM7_TM_A + it % 2 * 128 if inplace else FSUM7_TM_DMI
                for kp in range(4):
                    txl.ptx[UMMA_F16](txl.cast(tmem_base + d_col, 'uint32'), a_d + txl.uint64(kp * 128), b_d + txl.uint64(kp * 128), txl.uint32(FSUM7_ID_MI), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(FSUM7_B_MIACC + (it % 2 if inplace else 0)), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(FSUM7_B_EMPTY + stage), pred=leader)
            with txl.if_(nch > 0), txl.then_():
                issue_mi(0)
            if inplace:
                with txl.if_(nch > 1), txl.then_():
                    issue_mi(1)
            with txl.serial(nch) as it:
                s2 = it % 2
                mbar_wait(FSUM7_B_AREADY + s2, it // 2 % 2)
                mbar_wait(FSUM7_B_BREADY, it % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                b_d = umma_desc(smem, FSUM7_MB_S, UMMA_DESC_MN_TF32)
                for kp in range(16):
                    txl.ptx[UMMA_TF32](txl.cast(tmem_base + FSUM7_TM_DM, 'uint32'), txl.cast(tmem_base + FSUM7_TM_A + s2 * 128 + kp * 8, 'uint32'), b_d + txl.uint64(kp * 64), txl.uint32(id_chain), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(FSUM7_B_MCACC), pred=leader)
                if not inplace:
                    txl.ptx[UMMA_COMMIT](mbar_addr(FSUM7_B_AFREE + s2), pred=leader)
                if inplace:
                    # M_i(it + 2) overwrites the A slot chain(it) reads: tcgen05.mma A reads ->
                    # later D writes from the same warp are pipelined, so no AFREE wait is needed.
                    with txl.if_(it + 2 < nch), txl.then_():
                        issue_mi(it + 2)
                else:
                    with txl.if_(it + 1 < nch), txl.then_():
                        issue_mi(it + 1)

        with txl.if_(warp == 8), txl.then_():
            chain_mma_loop()
        def update_mma_loop():
            """
            MMA warp of the state update: issue the per-chunk update (and output) UMMAs.
            """
            leader = txl.cuda.elect_sync()

            def mma(op, d, a, b, idesc, acc):
                txl.ptx[op](txl.cast(d, 'uint32'), a, b, txl.uint32(idesc), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(acc)), pred=leader)
            with txl.serial(nch) as it:
                stage = it % FSUM7_NS
                stage_base = stage * FSUM7_STAGE
                mbar_wait(FSUM7_B_FULL + stage, it // FSUM7_NS % 2)
                mbar_wait(FSUM7_B_HB, it % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                if with_o:
                    with txl.if_(it >= 1), txl.then_():
                        mbar_wait(B_OFREE, (it - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                hb_d, w_d = umma_desc(smem, FSUM7_HB_S, UMMA_DESC_MN), umma_desc(smem, stage_base + FSUM7_W_S, UMMA_DESC_K)
                for kp in range(8):
                    mma(UMMA_F16, tmem_base + FSUM7_TM_VD, w_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), hb_d + txl.uint64(kp * 128), FSUM7_ID_M64, 0 if kp == 0 else 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(FSUM7_B_VD), pred=leader)
                if with_o:
                    q_d = umma_desc(smem, stage_base + Q_S, UMMA_DESC_K)
                    for kp in range(8):
                        mma(UMMA_F16, tmem_base + TM_OI, q_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), hb_d + txl.uint64(kp * 128), FSUM7_ID_M64, 0 if kp == 0 else 1)
                mbar_wait(FSUM7_B_VN, it % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                k_d, vg_d = umma_desc(smem, stage_base + FSUM7_K_S, UMMA_DESC_MN), umma_desc(smem, FSUM7_VG_S, UMMA_DESC_MN)
                for kp in range(4):
                    mma(UMMA_F16, tmem_base + FSUM7_TM_H, k_d + txl.uint64(kp * 128), vg_d + txl.uint64(kp * 128), FSUM7_ID_UPD, 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(FSUM7_B_UPD), pred=leader)
                if with_o:
                    a_d, vn_d = umma_desc(smem, stage_base + AQ_S, UMMA_DESC_K), umma_desc(smem, VN_S, UMMA_DESC_MN)
                    for kp in range(4):
                        mma(UMMA_F16, tmem_base + TM_OX, a_d + txl.uint64(kp * 2), vn_d + txl.uint64(kp * 128), FSUM7_ID_M64, 0 if kp == 0 else 1)
                    txl.ptx[UMMA_COMMIT](mbar_addr(B_OX), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(FSUM7_B_EMPTY + stage), pred=leader)

        with txl.if_(warp == 11), txl.then_():
            update_mma_loop()
        def chain_operand_loop():
            """
            Chain operand warps: build the scaled k / g tile and the A operand of each chain step.
            """
            r = tidx
            tmem_addr = tmem_base + tmem_lane_offset(tidx)

            def build_kg(it):
                stage = it % FSUM7_NS
                stage_base = stage * FSUM7_STAGE
                mbar_wait(FSUM7_B_FULL + stage, it // FSUM7_NS % 2)
                chunk_start = bos + it * 64
                num_valid = txl.min(eos - chunk_start, 64)
                gl_ = ld_shared_f32(smem, stage_base + FSUM7_G_S + (num_valid - 1) * 4)
                # all eight k / g loads first, then the scaling, then the stores (W0-3 patch the
                # next A slot right after this, so its latency is on the M chain's A path)
                kw = txl.alloc_local((32,), 'uint32')
                gr = txl.alloc_local((8,), 'float32')
                for rep in range(8):
                    idx = rep * 128 + tidx
                    row, ch = idx // 16, idx % 16
                    txl.ptx.ld.shared.v4.b32(kw[4 * rep], kw[4 * rep + 1], kw[4 * rep + 2], kw[4 * rep + 3], smem.ptr_to([stage_base + FSUM7_K_S + mn_major_offset(row, ch)]))
                    txl.ptx.ld.shared.f32(gr[rep], smem.ptr_to([stage_base + FSUM7_G_S + row * 4]))
                for rep in range(8):
                    idx = rep * 128 + tidx
                    row = idx // 16
                    fr = txl.if_then_else(row < num_valid, exp2_approx(gl_ - gr[rep]), txl.float32(0.0))
                    for j in range(4):
                        lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                        unpack_bf16x2(kw[4 * rep + j], lo, hi)
                        kg2 = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](kg2, txl.cuda.make_float2(lo, hi), txl.cuda.make_float2(fr, fr))
                        txl.assign(kw[4 * rep + j], pack_bf16x2(txl.cuda.float2_x(kg2), txl.cuda.float2_y(kg2)))
                for rep in range(8):
                    idx = rep * 128 + tidx
                    row, ch = idx // 16, idx % 16
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([FSUM7_KG_S + it % 2 * 16384 + mn_major_offset(row, ch)]), kw[4 * rep], kw[4 * rep + 1], kw[4 * rep + 2], kw[4 * rep + 3])
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(FSUM7_B_KG + it % 2)
                group_arrive_one(1, FSUM7_B_EMPTY + stage, tidx)
                return gl_
            glc, gln = txl.local_scalar('float32', init=txl.float32(0.0)), txl.local_scalar('float32', init=txl.float32(0.0))
            with txl.if_(nch > 0), txl.then_():
                txl.assign(glc, build_kg(0))
            with txl.if_(nch > 1), txl.then_():
                txl.assign(gln, build_kg(1))
            with txl.serial(nch) as it:
                s2 = it % 2
                if inplace:
                    mbar_wait(FSUM7_B_MIACC + s2, it // 2 % 2)
                else:
                    mbar_wait(FSUM7_B_MIACC, it % 2)
                    with txl.if_(it >= 2), txl.then_():
                        mbar_wait(FSUM7_B_AFREE + s2, (it // 2 - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                egl = exp2_approx(glc)
                dv = txl.alloc_local((32,), 'uint32')
                if inplace and rawneg:
                    txl.ptx[TMEM_LD_32X32B_X32](*[dv[i] for i in range(32)], txl.cast(tmem_addr + FSUM7_TM_A + s2 * 128 + warp * 32, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    for i in range(32):
                        txl.assign(dv[i], txl.if_then_else(lane == i, txl.reinterpret(txl.reinterpret(dv[i], ty='float32') - egl, ty='uint32'), dv[i]))
                    txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + FSUM7_TM_A + s2 * 128 + warp * 32, 'uint32'), *[dv[i] for i in range(32)])
                for q4 in range(0 if inplace and rawneg else 4):
                    txl.ptx[TMEM_LD_32X32B_X32](*[dv[i] for i in range(32)], txl.cast(tmem_addr + (FSUM7_TM_A + s2 * 128 if inplace else FSUM7_TM_DMI) + q4 * 32, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    if rawneg:
                        with txl.if_(r // 32 == q4), txl.then_():
                            for i in range(32):
                                txl.assign(dv[i], txl.if_then_else(r % 32 == i, txl.reinterpret(txl.reinterpret(dv[i], ty='float32') - egl, ty='uint32'), dv[i]))
                    else:
                        with txl.if_(r // 32 == q4), txl.then_():
                            for i in range(32):
                                d = txl.if_then_else(r % 32 == i, egl, txl.float32(0.0)) - txl.reinterpret(dv[i], ty='float32')
                                txl.assign(dv[i], txl.bitwise_and(txl.reinterpret(d, ty='uint32'), txl.uint32(4294959104)))
                        with txl.if_(r // 32 != q4), txl.then_():
                            for i in range(32):
                                d = txl.float32(0.0) - txl.reinterpret(dv[i], ty='float32')
                                txl.assign(dv[i], txl.bitwise_and(txl.reinterpret(d, ty='uint32'), txl.uint32(4294959104)))
                    txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + FSUM7_TM_A + s2 * 128 + q4 * 32, 'uint32'), *[dv[i] for i in range(32)])
                txl.ptx['tcgen05.wait::st.sync.aligned']()
                txl.ptx['tcgen05.fence::before_thread_sync']()
                if not inplace:
                    arrive(FSUM7_B_MIFREE)
                arrive(FSUM7_B_AREADY + s2)
                txl.assign(glc, gln)
                with txl.if_(it + 2 < nch), txl.then_():
                    txl.ptx['fence.proxy.async.shared::cta']()
                    txl.assign(gln, build_kg(it + 2))

        with txl.if_(warp < 4), txl.then_():
            chain_operand_loop()
        def chain_result_loop():
            """
            Chain result warps: seed the chain with the identity, store its result to the payload.
            """
            t = tidx - 128
            row = t
            tmem_addr = tmem_base + tmem_lane_offset(t)
            for c4 in range(8):
                z = []
                for e in range(4):
                    z.append(txl.if_then_else(cb * 32 + c4 * 4 + e == row, txl.float32(1.0), txl.float32(0.0)))
                txl.ptx.st.shared.v4.f32(smem.ptr_to([FSUM7_MB_S + merge_block_offset(row, c4)]), z[0], z[1], z[2], z[3])
            txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
            with txl.if_(lane == 0), txl.then_():
                arrive(FSUM7_B_BREADY)
            mv = txl.alloc_local((32,), 'uint32')
            with txl.serial(nch) as it:
                mbar_wait(FSUM7_B_MCACC, it % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx[TMEM_LD_32X32B_X32](*[mv[i] for i in range(32)], txl.cast(tmem_addr + FSUM7_TM_DM, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                with txl.if_(it + 1 < nch), txl.then_():
                    txl.ptx['fence.proxy.async.shared::cta']()
                    for c4 in range(8):
                        tv = [txl.bitwise_and(mv[c4 * 4 + e], txl.uint32(4294959104)) for e in range(4)]
                        txl.ptx.st.shared.v4.b32(smem.ptr_to([FSUM7_MB_S + merge_block_offset(row, c4)]), tv[0], tv[1], tv[2], tv[3])
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                    txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                    with txl.if_(lane == 0), txl.then_():
                        arrive(FSUM7_B_BREADY)
            with txl.if_(nch == 0), txl.then_():
                for c4 in range(8):
                    for e in range(4):
                        txl.assign(mv[c4 * 4 + e], txl.reinterpret(txl.if_then_else(cb * 32 + c4 * 4 + e == row, txl.float32(1.0), txl.float32(0.0)), ty='uint32'))
            for c4 in range(8):
                txl.ptx.st.global_.v4.b32(payload.ptr_to([(head * 128 + row) * 256 + 128 + cb * 32 + c4 * 4]), mv[c4 * 4], mv[c4 * 4 + 1], mv[c4 * 4 + 2], mv[c4 * 4 + 3])

        with txl.if_(txl.And(warp >= 4, warp < 8)), txl.then_():
            chain_result_loop()
        def state_loop():
            """
            State warps: carry the state through the chunks and store it to the payload.
            """
            t = tidx - 384
            tmem_addr = tmem_base + tmem_lane_offset(t)
            if with_o:
                cw1 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 2) != 0, ck1, -1))
                cw2 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 4) != 0, ck2, -1))
                cw3 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 8) != 0, ck3, -1))
            hv = txl.alloc_local((32,), 'uint32')
            for i in range(32):
                txl.assign(hv[i], txl.uint32(0))
            txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + FSUM7_TM_H, 'uint32'), *[hv[i] for i in range(32)])
            txl.ptx['tcgen05.wait::st.sync.aligned']()
            with txl.serial(nch) as it:
                stage = it % FSUM7_NS
                stage_base = stage * FSUM7_STAGE
                chunk_start = bos + it * 64
                num_valid = txl.min(eos - chunk_start, 64)
                mbar_wait(FSUM7_B_FULL + stage, it // FSUM7_NS % 2)
                with txl.if_(it >= 1), txl.then_():
                    mbar_wait(FSUM7_B_UPD, (it - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx[TMEM_LD_32X32B_X32](*[hv[i] for i in range(32)], txl.cast(tmem_addr + FSUM7_TM_H, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                hb4 = txl.alloc_local((16,), 'uint32')
                for p in range(16):
                    txl.assign(hb4[p], pack_bf16x2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32')))
                txl.ptx['fence.proxy.async.shared::cta']()
                for c in range(4):
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([FSUM7_HB_S + t * 128 + txl.bitwise_xor(c, txl.bitwise_and(t, 7)) * 16]), hb4[4 * c], hb4[4 * c + 1], hb4[4 * c + 2], hb4[4 * c + 3])
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(FSUM7_B_HB)
                gl = ld_shared_f32(smem, stage_base + FSUM7_G_S + (num_valid - 1) * 4)
                egl = exp2_approx(gl)
                lane_group, lane_in_group = lane // 4, lane % 4
                vrow = [16 * (t // 32) + lane_group + 8 * hh for hh in range(2)]
                frh = [txl.local_scalar('float32', init=txl.if_then_else(vrow[hh] < num_valid, exp2_approx(gl - ld_shared_f32(smem, stage_base + FSUM7_G_S + vrow[hh] * 4)), txl.float32(0.0))) for hh in range(2)]
                lw2 = txl.alloc_local((8,), 'uint32')
                for j in range(4):
                    for hh in range(2):
                        txl.ptx.ld.shared.b32(lw2[j * 2 + hh], smem.ptr_to([stage_base + FSUM7_U_S + vrow[hh] * FSUM7_UR + (8 * j + 2 * lane_in_group) * 2]))
                if with_o:
                    with txl.if_(txl.Or(txl.Or(it == cw1, it == cw2), it == cw3)), txl.then_():
                        txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + txl.if_then_else(it == cw1, 64, txl.if_then_else(it == cw2, 96, 480)), 'uint32'), *[hv[i] for i in range(32)])
                for p in range(16):
                    scaled = txl.local_scalar('uint64')
                    txl.ptx['mul.rn.f32x2'](scaled, txl.cuda.make_float2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32')), txl.cuda.make_float2(egl, egl))
                    txl.assign(hv[2 * p], txl.reinterpret(txl.cuda.float2_x(scaled), ty='uint32'))
                    txl.assign(hv[2 * p + 1], txl.reinterpret(txl.cuda.float2_y(scaled), ty='uint32'))
                txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + FSUM7_TM_H, 'uint32'), *[hv[i] for i in range(32)])
                txl.ptx['tcgen05.wait::st.sync.aligned']()
                mbar_wait(FSUM7_B_VD, it % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                acc16 = txl.alloc_local((16,), 'uint32')
                txl.ptx['tcgen05.ld.sync.aligned.16x256b.x4.b32'](*[acc16[i] for i in range(16)], txl.cast(tmem_addr + FSUM7_TM_VD, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                if with_o:
                    txl.ptx['fence.proxy.async.shared::cta']()
                vnf, vgf = txl.alloc_local((16,), 'float32'), txl.alloc_local((16,), 'float32')
                for j in range(4):
                    for hh in range(2):
                        lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                        unpack_bf16x2(lw2[j * 2 + hh], lo, hi)
                        vn2 = txl.local_scalar('uint64')
                        txl.ptx['sub.rn.f32x2'](vn2, txl.cuda.make_float2(lo, hi), txl.cuda.make_float2(txl.reinterpret(acc16[j * 4 + hh * 2], ty='float32'), txl.reinterpret(acc16[j * 4 + hh * 2 + 1], ty='float32')))
                        vg2 = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](vg2, vn2, txl.cuda.make_float2(frh[hh], frh[hh]))
                        ii = j * 4 + hh * 2
                        txl.assign(vnf[ii], txl.cuda.float2_x(vn2))
                        txl.assign(vnf[ii + 1], txl.cuda.float2_y(vn2))
                        txl.assign(vgf[ii], txl.cuda.float2_x(vg2))
                        txl.assign(vgf[ii + 1], txl.cuda.float2_y(vg2))
                rbase = 16 * (t // 32)
                if with_o:
                    stmatrix_acc(smem, lambda row, nt: VN_S + row * 128 + txl.bitwise_xor(nt, txl.bitwise_and(row, 7)) * 16, vnf, 4, rbase, lane)
                stmatrix_acc(smem, lambda row, nt: FSUM7_VG_S + row * 128 + txl.bitwise_xor(nt, txl.bitwise_and(row, 7)) * 16, vgf, 4, rbase, lane)
                txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(FSUM7_B_VN)
                group_arrive_one(2, FSUM7_B_EMPTY + stage, t)
            with txl.if_(nch > 0), txl.then_():
                mbar_wait(FSUM7_B_UPD, (nch - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx[TMEM_LD_32X32B_X32](*[hv[i] for i in range(32)], txl.cast(tmem_addr + FSUM7_TM_H, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
            for i4 in range(8):
                txl.ptx.st.global_.v4.b32(payload.ptr_to([(head * 128 + t) * 256 + v0 + 4 * i4]), hv[4 * i4], hv[4 * i4 + 1], hv[4 * i4 + 2], hv[4 * i4 + 3])

        with txl.if_(txl.And(warp >= 12, warp < 16)), txl.then_():
            state_loop()
        if with_o:
            def output_loop():
                """
                Output warps: finish o of the folded segment and write the checkpoints.
                """
                t = tidx - 512
                tmem_addr = tmem_base + tmem_lane_offset(t)
                cw1 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 2) != 0, ck1, -1))
                cw2 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 4) != 0, ck2, -1))
                cw3 = txl.local_scalar('int32', init=txl.if_then_else(txl.bitwise_and(ckm, 8) != 0, ck3, -1))
                mrow = 16 * (t // 32) + lane % 16
                act = lane < 16
                with txl.serial(nch) as it:
                    stage = it % FSUM7_NS
                    stage_base = stage * FSUM7_STAGE
                    chunk_start = bos + it * 64
                    num_valid = txl.min(eos - chunk_start, 64)
                    mbar_wait(B_OX, it % 2)
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    oi, ox = txl.alloc_local((32,), 'uint32'), txl.alloc_local((32,), 'uint32')
                    txl.ptx[TMEM_LD_32X32B_X32](*[oi[i] for i in range(32)], txl.cast(tmem_addr + TM_OI, 'uint32'))
                    txl.ptx[TMEM_LD_32X32B_X32](*[ox[i] for i in range(32)], txl.cast(tmem_addr + TM_OX, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                    arrive(B_OFREE)
                    # only the gate row comes from the stage: read it, then release the stage before
                    # the o math / stores (the two-stage ring otherwise waits on this release)
                    eg = txl.local_scalar('float32', init=exp2_approx(ld_shared_f32(smem, stage_base + FSUM7_G_S + mrow * 4)))
                    group_arrive_one(3, FSUM7_B_EMPTY + stage, t)
                    with txl.if_(txl.And(act, mrow < num_valid)), txl.then_():
                        ob = txl.alloc_local((16,), 'uint32')
                        for p in range(16):
                            oi2, ox2, out2 = txl.local_scalar('uint64'), txl.local_scalar('uint64'), txl.local_scalar('uint64')
                            txl.ptx['mul.rn.f32x2'](oi2, txl.cuda.make_float2(txl.reinterpret(oi[2 * p], ty='float32'), txl.reinterpret(oi[2 * p + 1], ty='float32')), txl.cuda.make_float2(eg, eg))
                            txl.ptx['mul.rn.f32x2'](ox2, txl.cuda.make_float2(txl.reinterpret(ox[2 * p], ty='float32'), txl.reinterpret(ox[2 * p + 1], ty='float32')), txl.cuda.make_float2(scale, scale))
                            txl.ptx['fma.rn.f32x2'](out2, txl.cuda.make_float2(scale, scale), oi2, ox2)
                            txl.assign(ob[p], pack_bf16x2(txl.cuda.float2_x(out2), txl.cuda.float2_y(out2)))
                        for c in range(4):
                            txl.ptx.st.global_.v4.b32(o.ptr_to([((chunk_start + mrow) * HV + head) * 128 + v0 + c * 8]), ob[4 * c], ob[4 * c + 1], ob[4 * c + 2], ob[4 * c + 3])
                    with txl.if_(txl.Or(txl.Or(it == cw1, it == cw2), it == cw3)), txl.then_():
                        ckv = txl.alloc_local((32,), 'uint32')
                        txl.ptx[TMEM_LD_32X32B_X32](*[ckv[i] for i in range(32)], txl.cast(tmem_addr + txl.if_then_else(it == cw1, 64, txl.if_then_else(it == cw2, 96, 480)), 'uint32'))
                        txl.ptx['tcgen05.wait::ld.sync.aligned']()
                        ckb = txl.local_scalar('int32', init=checkpoint_base(txl.if_then_else(it == cw1, kt1, txl.if_then_else(it == cw2, kt2, kt3)), head, t, v0 // 32, HV))
                        for i8 in range(8):
                            txl.ptx.st.global_.v4.b32(a_ck.ptr_to([ckb + i8 * HV * 64]), ckv[4 * i8], ckv[4 * i8 + 1], ckv[4 * i8 + 2], ckv[4 * i8 + 3])

            with txl.if_(warp >= 16), txl.then_():
                output_loop()
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        with txl.if_(warp == 8), txl.then_():
            txl.ptx['tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned']()
            txl.ptx['tcgen05.dealloc.cta_group::1.sync.aligned.b32'](txl.cast(tmem_base, 'uint32'), txl.uint32(512))
    return fsum7


BSUM7_NS = 3
BSUM7_QDO_NS = 2
BSUM7_UR = 80
BSUM7_QDO_UR = 64
BSUM7_K_S = 0
BSUM7_W_S = BSUM7_K_S + 16384
BSUM7_DV_S = BSUM7_W_S + 16384
BSUM7_G_S = BSUM7_DV_S + 64 * BSUM7_UR
BSUM7_STAGE = (BSUM7_G_S + 256 + 1023) // 1024 * 1024
BSUM7_QDO_BASE = BSUM7_NS * BSUM7_STAGE
BSUM7_Q_S = 0
BSUM7_DO_S = BSUM7_Q_S + 16384
BSUM7_QDO_STAGE = BSUM7_DO_S + 64 * BSUM7_QDO_UR
BSUM7_HB_S = BSUM7_QDO_BASE + BSUM7_QDO_NS * BSUM7_QDO_STAGE
BSUM7_DVA_S = BSUM7_HB_S + 16384
BSUM7_DOB_S = BSUM7_DVA_S + 8192
BSUM7_KG_S = BSUM7_DOB_S + 2 * 8192
BSUM7_MB_S = BSUM7_KG_S + 16384
BSUM7_BAR_S = BSUM7_MB_S + 16384
BSUM7_HOLD_S = BSUM7_BAR_S + 256
BSUM7_EG_S = BSUM7_HOLD_S + 64
BSUM7_SMEM = BSUM7_EG_S + 256
BSUM7_NWARPS = 16
BSUM7_TM_D = 0
BSUM7_TM_DM = 256
BSUM7_TM_DVP = 288
BSUM7_TM_T2 = 320
BSUM7_TM_T1 = 352
BSUM7_TM_QA = 416
BSUM7_ID_MI = 136414352
BSUM7_ID_TF32 = 134809872
BSUM7_ID_M64 = 67699856
BSUM7_ID_UPD = 134841488
BSUM7_B_FULL = 0
BSUM7_B_EMPTY = BSUM7_NS
BSUM7_B_QFULL = 2 * BSUM7_NS
BSUM7_B_QEMPTY = BSUM7_B_QFULL + BSUM7_QDO_NS
BSUM7_B_KG = BSUM7_B_QEMPTY + BSUM7_QDO_NS
BSUM7_B_MIACC = BSUM7_B_KG + 1
BSUM7_B_AREADY = BSUM7_B_MIACC + 2
BSUM7_B_AFREE = BSUM7_B_AREADY + 2
BSUM7_B_BREADY = BSUM7_B_AFREE + 2
BSUM7_B_MCACC = BSUM7_B_BREADY + 1
BSUM7_B_HB = BSUM7_B_MCACC + 1
BSUM7_B_DVP = BSUM7_B_HB + 1
BSUM7_B_DVA = BSUM7_B_DVP + 1
BSUM7_B_T2 = BSUM7_B_DVA + 1
BSUM7_B_TOP = BSUM7_B_T2 + 1
BSUM7_B_T1 = BSUM7_B_TOP + 1
BSUM7_B_T1F = BSUM7_B_T1 + 2
BSUM7_N_BARS = BSUM7_B_T1F + 2


def build_bsum7(HV, store=False, rawneg=False, lst=False):
    """
    The backward gradient summary of the rank's first local sequence (bsum7 / bsum7s). store: also
    write dh_out / dv2_out (the real reverse pass of the first segment, valid when it starts from a
    zero gradient: the last rank, or a segment that ends on this rank). rawneg: leave D_Mi raw
    except the diagonal (D_ii - e^gl) and negate A in the chain UMMA; the TF32 UMMA drops the low 13
    mantissa bits itself, so -(D - diag) = tf32(diag - D) without a 128-column pass.
    """
    lst = lst and store
    BSUM7_B_HBF = BSUM7_N_BARS
    assert (BSUM7_N_BARS + 4) * 8 <= 256

    def hb_bar(i):
        """
        lst: HB / DVA arrivals alternate between two barriers (one per smem half), waited with
        parity i // 2.
        """
        return BSUM7_B_HB + i % 2 * (BSUM7_B_HBF + 2 - BSUM7_B_HB) if lst else BSUM7_B_HB

    def dva_bar(i):
        return BSUM7_B_DVA + i % 2 * (BSUM7_B_HBF + 3 - BSUM7_B_DVA) if lst else BSUM7_B_DVA

    def hpar(i):
        return i // 2 % 2 if lst else i % 2
    id_chain = BSUM7_ID_TF32 | (1 << 13 if rawneg else 0)

    def prepare_host(params):
        maps = []
        for name in ('q', 'k', 'w'):
            desc = txl.stack_alloca('tensormap', 1)
            txl.cu_tensor_map_encode_tiled(desc, 'bfloat16', 3, params[name].data, 64, params['S'], 2 * HV, HV * 256, 128, 64, 64, 2, 1, 1, 1, 0, 3, 2, 0)
            maps.append(desc)
        return tuple(maps)

    @txl.kernel()
    def bsum7(q: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), k: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), do: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dv_local: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), g_cum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), bos: txl.int32, eos: txl.int32, scale: txl.float32, payload: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), dh_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dv2_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), S: txl.int32):
        host = prepare_host(dict(q=q, k=k, w=w, S=S))
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[4, HV], block=BSUM7_NWARPS * 32))
        q_map, k_map, w_map = host
        smem = txl.alloc_tensor((BSUM7_SMEM,), txl.u8, scope='shared.dyn', align=1024)
        cta = txl.cta_id()
        with txl.if_(txl.thread_id() == 0), txl.then_():
            txl.ptx.griddepcontrol.launch_dependents()
        cb = cta[0]
        head = cta[1]
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        nch = (eos - bos + 63) // 64
        v0 = cb * 32

        def mbar_addr(i):
            return smem.ptr_to([BSUM7_BAR_S + i * 8])

        def mbar_wait(i, parity):
            ready = txl.local_scalar('uint32', init=txl.uint32(0))
            with txl.while_(ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready, mbar_addr(i), txl.cast(parity, 'uint32'))

        def mbar_wait_n(name, i, parity):
            token = txl.alloc_local((1,), 'uint32')
            txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
            with txl.if_(lane == 0), txl.then_():
                txl.assign(token[0], txl.cuda.iket.range_start(name))
            mbar_wait(i, parity)
            txl.cuda.iket.range_end(token[0])

        def arrive(i):
            txl.ptx.mbarrier.arrive.shared.b64(mbar_addr(i))

        def chunk_t0(it):
            return bos + (nch - 1 - it) * 64

        def group_arrive_one(bar_id, bar, t):
            txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
            txl.ptx.bar.sync(txl.uint32(bar_id), txl.uint32(128))
            with txl.if_(t == 0), txl.then_():
                arrive(bar)
        with txl.if_(tidx == 0), txl.then_():
            for s_ in range(BSUM7_NS):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(BSUM7_B_FULL + s_), txl.uint32(65))
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(BSUM7_B_EMPTY + s_), txl.uint32(5))
            for s_ in range(BSUM7_QDO_NS):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(BSUM7_B_QFULL + s_), txl.uint32(65))
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(BSUM7_B_QEMPTY + s_), txl.uint32(1))
            for i_, cnt in ((BSUM7_B_KG, 4), (BSUM7_B_MIACC, 1), (BSUM7_B_MIACC + 1, 1), (BSUM7_B_AREADY, 128), (BSUM7_B_AREADY + 1, 128), (BSUM7_B_AFREE, 1), (BSUM7_B_AFREE + 1, 1), (BSUM7_B_BREADY, 4), (BSUM7_B_MCACC, 1), (BSUM7_B_HB, 4), (BSUM7_B_DVP, 1), (BSUM7_B_DVA, 4), (BSUM7_B_T2, 1), (BSUM7_B_TOP, 128), (BSUM7_B_T1, 1), (BSUM7_B_T1 + 1, 1), (BSUM7_B_T1F, 128), (BSUM7_B_T1F + 1, 128), (BSUM7_B_HBF, 64), (BSUM7_B_HBF + 1, 64)) + (((BSUM7_B_HBF + 2, 4), (BSUM7_B_HBF + 3, 4)) if lst else ()):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(i_), txl.uint32(cnt))
        txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.if_(warp == 8), txl.then_():
            txl.ptx['tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32'](smem.ptr_to([BSUM7_HOLD_S]), txl.uint32(512))
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        tmem_base = txl.local_scalar('int32')
        txl.ptx.ld.volatile.shared.s32(tmem_base, smem.ptr_to([BSUM7_HOLD_S]))
        def load_loop():
            """
            Loader warps: copy each chunk's operands into the stage rings.
            """
            loader_tidx = tidx - 288

            def copy_chunk(j):
                ci = nch - 1 - j
                t0c = chunk_t0(j)
                nvc = txl.min(eos - t0c, 64)
                mbar_wait(hb_bar(j), hpar(j))
                for r_ in range(8):
                    p = loader_tidx + 64 * r_
                    row, c = p // 4, p % 4
                    wd = txl.alloc_local((4,), 'uint32')
                    txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([BSUM7_HB_S + row * 128 + txl.bitwise_xor(c + 4 * (j % 2), txl.bitwise_and(row, 7)) * 16]))
                    txl.ptx.st.global_.v4.b32(dh_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 16384 + umma128_offset(row, v0 // 8 + c) // 2]), wd[0], wd[1], wd[2], wd[3])
                mbar_wait(dva_bar(j), hpar(j))
                for r_ in range(4):
                    p = loader_tidx + 64 * r_
                    row, c = p // 4, p % 4
                    wd = txl.alloc_local((4,), 'uint32')
                    for e in range(4):
                        txl.assign(wd[e], txl.uint32(0))
                    with txl.if_(row < nvc), txl.then_():
                        txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([BSUM7_DVA_S + row * 128 + txl.bitwise_xor(c + 4 * (j % 2), txl.bitwise_and(row, 7)) * 16]))
                    txl.ptx.st.global_.v4.b32(dv2_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 8192 + umma64_offset(row, v0 // 8 + c) // 2]), wd[0], wd[1], wd[2], wd[3])
                arrive(BSUM7_B_HBF + j % 2)
            with txl.serial(nch) as it:
                txl.cuda.iket.mark('loader-start')
                stage = it % BSUM7_NS
                qstage = it % BSUM7_QDO_NS
                stage_base = stage * BSUM7_STAGE
                qb = BSUM7_QDO_BASE + qstage * BSUM7_QDO_STAGE
                with txl.if_(it >= BSUM7_NS), txl.then_():
                    mbar_wait_n('loader-main-empty', BSUM7_B_EMPTY + stage, (it // BSUM7_NS - 1) % 2)
                    txl.ptx['fence.proxy.async.shared::cta']()
                with txl.if_(it >= BSUM7_QDO_NS), txl.then_():
                    mbar_wait_n('loader-q-empty', BSUM7_B_QEMPTY + qstage, (it // BSUM7_QDO_NS - 1) % 2)
                    txl.ptx['fence.proxy.async.shared::cta']()
                chunk_start = txl.local_scalar('int32', init=chunk_t0(it))
                num_valid = txl.local_scalar('int32', init=txl.min(eos - chunk_start, 64))
                with txl.if_(txl.And(num_valid == 64, loader_tidx == 0)), txl.then_():
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar_addr(BSUM7_B_FULL + stage), txl.uint32(32768))
                    for tmap, dst in ((k_map, stage_base + BSUM7_K_S), (w_map, stage_base + BSUM7_W_S)):
                        txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([dst]), txl.address_of(tmap), txl.int32(0), chunk_start, txl.cast(2 * head, 'int32'), mbar_addr(BSUM7_B_FULL + stage), txl.uint64(0x12F0000000000000))
                with txl.if_(num_valid < 64), txl.then_():
                    for rep in range(16):
                        idx = rep * 64 + loader_tidx
                        row, ch = idx // 16, idx % 16
                        ok = row < num_valid
                        src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                        for src, dst in ((k, BSUM7_K_S), (w, BSUM7_W_S)):
                            txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + dst + mn_major_offset(row, ch)]), src.ptr_to([(src_row * HV + head) * 128 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                    with txl.if_(loader_tidx == 0), txl.then_():
                        txl.ptx.mbarrier.arrive.shared.b64(mbar_addr(BSUM7_B_FULL + stage))
                for rep in range(4):
                    idx = rep * 64 + loader_tidx
                    row, ch = idx // 4, idx % 4
                    ok = row < num_valid
                    src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                    txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + BSUM7_DV_S + row * BSUM7_UR + ch * 16]), dv_local.ptr_to([(src_row * HV + head) * 128 + v0 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                ok = loader_tidx < num_valid
                src_row = txl.if_then_else(ok, chunk_start + loader_tidx, chunk_start)
                txl.ptx[CP_ASYNC_4](smem.ptr_to([stage_base + BSUM7_G_S + loader_tidx * 4]), g_cum.ptr_to([src_row * HV + head]), 4, txl.cast(txl.if_then_else(ok, 4, 0), 'uint32'))
                txl.ptx['cp.async.mbarrier.arrive.noinc.shared::cta.b64'](mbar_addr(BSUM7_B_FULL + stage))
                with txl.if_(txl.And(num_valid == 64, loader_tidx == 0)), txl.then_():
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar_addr(BSUM7_B_QFULL + qstage), txl.uint32(16384))
                    for tmap, dst in ((q_map, qb + BSUM7_Q_S),):
                        txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([dst]), txl.address_of(tmap), txl.int32(0), chunk_start, txl.cast(2 * head, 'int32'), mbar_addr(BSUM7_B_QFULL + qstage), txl.uint64(0x12F0000000000000))
                with txl.if_(num_valid < 64), txl.then_():
                    for rep in range(16):
                        idx = rep * 64 + loader_tidx
                        row, ch = idx // 16, idx % 16
                        ok = row < num_valid
                        src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                        txl.ptx[CP_ASYNC_16](smem.ptr_to([qb + BSUM7_Q_S + mn_major_offset(row, ch)]), q.ptr_to([(src_row * HV + head) * 128 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                    with txl.if_(loader_tidx == 0), txl.then_():
                        txl.ptx.mbarrier.arrive.shared.b64(mbar_addr(BSUM7_B_QFULL + qstage))
                for rep in range(4):
                    idx = rep * 64 + loader_tidx
                    row, ch = idx // 4, idx % 4
                    ok = row < num_valid
                    src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                    txl.ptx[CP_ASYNC_16](smem.ptr_to([qb + BSUM7_DO_S + row * BSUM7_QDO_UR + ch * 16]), do.ptr_to([(src_row * HV + head) * 128 + v0 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                txl.ptx['cp.async.mbarrier.arrive.noinc.shared::cta.b64'](mbar_addr(BSUM7_B_QFULL + qstage))
                if lst:
                    with txl.if_(it >= 2), txl.then_():
                        copy_chunk(it - 2)
                txl.cuda.iket.mark('loader-end')
            if lst:
                with txl.serial(2) as jt:
                    with txl.if_(nch - 2 + jt >= 0), txl.then_():
                        copy_chunk(nch - 2 + jt)

        with txl.if_(txl.And(warp >= 9, warp < 11)), txl.then_():
            load_loop()
        def chain_mma_loop():
            """
            MMA warp of the M chain: issue the Mi and chain UMMAs.
            """
            leader = txl.cuda.elect_sync()

            def issue_mi(it):
                stage = it % BSUM7_NS
                stage_base = stage * BSUM7_STAGE
                s2 = it % 2
                mbar_wait_n('issue-full', BSUM7_B_FULL + stage, it // BSUM7_NS % 2)
                mbar_wait_n('issue-kg', BSUM7_B_KG, it % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                with txl.if_(it >= 2), txl.then_():
                    mbar_wait_n('issue-afree', BSUM7_B_AFREE + s2, (it // 2 - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                a_d, b_d = umma_desc(smem, stage_base + BSUM7_W_S, UMMA_DESC_MN), umma_desc(smem, BSUM7_KG_S, UMMA_DESC_MN)
                for kp in range(4):
                    txl.ptx[UMMA_F16](txl.cast(tmem_base + BSUM7_TM_D + s2 * 128, 'uint32'), a_d + txl.uint64(kp * 128), b_d + txl.uint64(kp * 128), txl.uint32(BSUM7_ID_MI), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(BSUM7_B_MIACC + s2), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(BSUM7_B_EMPTY + stage), pred=leader)
            with txl.if_(nch > 0), txl.then_():
                issue_mi(0)
            with txl.serial(nch) as it:
                txl.cuda.iket.mark('issue-start')
                s2 = it % 2
                mbar_wait_n('issue-aready', BSUM7_B_AREADY + s2, it // 2 % 2)
                mbar_wait_n('issue-bready', BSUM7_B_BREADY, it % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                b_d = umma_desc(smem, BSUM7_MB_S, UMMA_DESC_MN_TF32)
                for kp in range(16):
                    txl.ptx[UMMA_TF32](txl.cast(tmem_base + BSUM7_TM_DM, 'uint32'), txl.cast(tmem_base + BSUM7_TM_D + s2 * 128 + kp * 8, 'uint32'), b_d + txl.uint64(kp * 64), txl.uint32(id_chain), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(BSUM7_B_MCACC), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(BSUM7_B_AFREE + s2), pred=leader)
                with txl.if_(it + 1 < nch), txl.then_():
                    issue_mi(it + 1)
                txl.cuda.iket.mark('issue-end')

        with txl.if_(warp == 8), txl.then_():
            chain_mma_loop()
        def update_mma_loop():
            """
            MMA warp of the gradient update: issue the per-chunk T1 / T2 / dv UMMAs.
            """
            leader = txl.cuda.elect_sync()

            def mma(op, d, a, b, idesc, acc):
                txl.ptx[op](txl.cast(d, 'uint32'), a, b, txl.uint32(idesc), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(acc)), pred=leader)

            def issue_t1(it):
                s2 = it % 2
                mbar_wait_n('issue2-top', BSUM7_B_TOP, it % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                with txl.if_(it >= 2), txl.then_():
                    mbar_wait_n('issue2-t1free', BSUM7_B_T1F + s2, (it // 2 - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                bd = umma_desc(smem, BSUM7_DOB_S + s2 * 8192, UMMA_DESC_MN_TF32)
                for kp in range(8):
                    mma(UMMA_TF32, tmem_base + BSUM7_TM_T1 + s2 * 32, txl.cast(tmem_base + BSUM7_TM_QA + kp * 8, 'uint32'), bd + txl.uint64(kp * 64), BSUM7_ID_TF32, 0 if kp == 0 else 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(BSUM7_B_T1 + s2), pred=leader)
            with txl.if_(nch > 0), txl.then_():
                issue_t1(0)
            with txl.serial(nch) as it:
                txl.cuda.iket.mark('issue2-start')
                stage = it % BSUM7_NS
                stage_base = stage * BSUM7_STAGE
                mbar_wait_n('issue2-full', BSUM7_B_FULL + stage, it // BSUM7_NS % 2)
                mbar_wait_n('issue2-hb', hb_bar(it), hpar(it))
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                hb_d, k_d = umma_desc(smem, BSUM7_HB_S + (it % 2 * 64 if lst else 0), UMMA_DESC_MN), umma_desc(smem, stage_base + BSUM7_K_S, UMMA_DESC_K)
                for kp in range(8):
                    mma(UMMA_F16, tmem_base + BSUM7_TM_DVP, k_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), hb_d + txl.uint64(kp * 128), BSUM7_ID_M64, 0 if kp == 0 else 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(BSUM7_B_DVP), pred=leader)
                mbar_wait_n('issue2-dva', dva_bar(it), hpar(it))
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                w_d, dva_d = umma_desc(smem, stage_base + BSUM7_W_S, UMMA_DESC_MN), umma_desc(smem, BSUM7_DVA_S + (it % 2 * 64 if lst else 0), UMMA_DESC_MN)
                for kp in range(4):
                    mma(UMMA_F16, tmem_base + BSUM7_TM_T2, w_d + txl.uint64(kp * 128), dva_d + txl.uint64(kp * 128), BSUM7_ID_UPD, 0 if kp == 0 else 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(BSUM7_B_T2), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(BSUM7_B_EMPTY + stage), pred=leader)
                with txl.if_(it + 1 < nch), txl.then_():
                    issue_t1(it + 1)
                txl.cuda.iket.mark('issue2-end')

        with txl.if_(warp == 11), txl.then_():
            update_mma_loop()
        def chain_operand_loop():
            """
            Chain operand warps: build the scaled k / g tile and the A operand of each chain step.
            """
            t = tidx
            tmem_addr = tmem_base + tmem_lane_offset(t)
            with txl.serial(nch) as it:
                txl.cuda.iket.mark('state-start')
                stage = it % BSUM7_NS
                stage_base = stage * BSUM7_STAGE
                s2 = it % 2
                mbar_wait_n('state-full', BSUM7_B_FULL + stage, it // BSUM7_NS % 2)
                chunk_start = chunk_t0(it)
                num_valid = txl.min(eos - chunk_start, 64)
                gl = ld_shared_f32(smem, stage_base + BSUM7_G_S + (num_valid - 1) * 4)
                with txl.if_(it >= 1), txl.then_():
                    mbar_wait_n('state-mi-prev', BSUM7_B_MIACC + (it - 1) % 2, (it - 1) // 2 % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                # KG(it) is on the M chain's critical path (single buffer): issue all eight k / g
                # loads first, then the scaling, then the stores, instead of eight dependent load ->
                # exp2 -> store rounds.
                kw = txl.alloc_local((32,), 'uint32')
                gr = txl.alloc_local((8,), 'float32')
                for rep in range(8):
                    idx = rep * 128 + t
                    row, ch = idx // 16, idx % 16
                    txl.ptx.ld.shared.v4.b32(kw[4 * rep], kw[4 * rep + 1], kw[4 * rep + 2], kw[4 * rep + 3], smem.ptr_to([stage_base + BSUM7_K_S + mn_major_offset(row, ch)]))
                    txl.ptx.ld.shared.f32(gr[rep], smem.ptr_to([stage_base + BSUM7_G_S + row * 4]))
                for rep in range(8):
                    idx = rep * 128 + t
                    row = idx // 16
                    fr = txl.if_then_else(row < num_valid, exp2_approx(gl - gr[rep]), txl.float32(0.0))
                    for j in range(4):
                        lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                        unpack_bf16x2(kw[4 * rep + j], lo, hi)
                        kg2 = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](kg2, txl.cuda.make_float2(lo, hi), txl.cuda.make_float2(fr, fr))
                        txl.assign(kw[4 * rep + j], pack_bf16x2(txl.cuda.float2_x(kg2), txl.cuda.float2_y(kg2)))
                for rep in range(8):
                    idx = rep * 128 + t
                    row, ch = idx // 16, idx % 16
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([BSUM7_KG_S + mn_major_offset(row, ch)]), kw[4 * rep], kw[4 * rep + 1], kw[4 * rep + 2], kw[4 * rep + 3])
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(BSUM7_B_KG)
                mbar_wait_n('state-mi', BSUM7_B_MIACC + s2, it // 2 % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                egl = exp2_approx(gl)
                dv = txl.alloc_local((32,), 'uint32')
                if rawneg:
                    txl.ptx[TMEM_LD_32X32B_X32](*[dv[i] for i in range(32)], txl.cast(tmem_addr + BSUM7_TM_D + s2 * 128 + warp * 32, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    for i in range(32):
                        txl.assign(dv[i], txl.if_then_else(lane == i, txl.reinterpret(txl.reinterpret(dv[i], ty='float32') - egl, ty='uint32'), dv[i]))
                    txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + BSUM7_TM_D + s2 * 128 + warp * 32, 'uint32'), *[dv[i] for i in range(32)])
                else:
                    for q4 in range(4):
                        txl.ptx[TMEM_LD_32X32B_X32](*[dv[i] for i in range(32)], txl.cast(tmem_addr + BSUM7_TM_D + s2 * 128 + q4 * 32, 'uint32'))
                        txl.ptx['tcgen05.wait::ld.sync.aligned']()
                        with txl.if_(warp == q4), txl.then_():
                            for i in range(32):
                                d = txl.if_then_else(lane == i, egl, txl.float32(0.0)) - txl.reinterpret(dv[i], ty='float32')
                                txl.assign(dv[i], txl.bitwise_and(txl.reinterpret(d, ty='uint32'), txl.uint32(4294959104)))
                        with txl.if_(warp != q4), txl.then_():
                            for i in range(32):
                                d = txl.float32(0.0) - txl.reinterpret(dv[i], ty='float32')
                                txl.assign(dv[i], txl.bitwise_and(txl.reinterpret(d, ty='uint32'), txl.uint32(4294959104)))
                        txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + BSUM7_TM_D + s2 * 128 + q4 * 32, 'uint32'), *[dv[i] for i in range(32)])
                txl.ptx['tcgen05.wait::st.sync.aligned']()
                txl.ptx['tcgen05.fence::before_thread_sync']()
                arrive(BSUM7_B_AREADY + s2)
                group_arrive_one(1, BSUM7_B_EMPTY + stage, t)
                txl.cuda.iket.mark('state-end')

        with txl.if_(warp < 4), txl.then_():
            chain_operand_loop()
        def qdo_loop():
            """
            q / do warps: build the gated q and do operands, store the chain result to the payload.
            """
            t = tidx - 128
            row = t
            tmem_addr = tmem_base + tmem_lane_offset(t)
            for c4 in range(8):
                z = []
                for e in range(4):
                    z.append(txl.if_then_else(cb * 32 + c4 * 4 + e == row, txl.float32(1.0), txl.float32(0.0)))
                txl.ptx.st.shared.v4.f32(smem.ptr_to([BSUM7_MB_S + merge_block_offset(row, c4)]), z[0], z[1], z[2], z[3])
            txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
            with txl.if_(lane == 0), txl.then_():
                arrive(BSUM7_B_BREADY)
            mv = txl.alloc_local((32,), 'uint32')

            def qdo_top(it):
                stage = it % BSUM7_NS
                qstage = it % BSUM7_QDO_NS
                stage_base = stage * BSUM7_STAGE
                qb = BSUM7_QDO_BASE + qstage * BSUM7_QDO_STAGE
                s2 = it % 2
                mbar_wait_n('qdo-full', BSUM7_B_FULL + stage, it // BSUM7_NS % 2)
                mbar_wait_n('qdo-qfull', BSUM7_B_QFULL + qstage, it // BSUM7_QDO_NS % 2)
                with txl.if_(t < 64), txl.then_():
                    txl.ptx.st.shared.f32(smem.ptr_to([BSUM7_EG_S + t * 4]), exp2_approx(ld_shared_f32(smem, stage_base + BSUM7_G_S + t * 4)))
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                txl.ptx.bar.sync(txl.uint32(3), txl.uint32(128))
                with txl.if_(it >= 1), txl.then_():
                    mbar_wait_n('qdo-t1-prev', BSUM7_B_T1 + (it - 1) % 2, (it - 1) // 2 % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                for h_ in range(2):
                    qa = txl.alloc_local((32,), 'uint32')
                    for ip in range(16):
                        i = ip * 2
                        tok = h_ * 32 + i
                        bits0, bits1 = txl.local_scalar('uint16'), txl.local_scalar('uint16')
                        txl.ptx.ld.shared.b16(bits0, smem.ptr_to([qb + BSUM7_Q_S + mn_major_offset(tok, t // 8) + t % 8 * 2]))
                        txl.ptx.ld.shared.b16(bits1, smem.ptr_to([qb + BSUM7_Q_S + mn_major_offset(tok + 1, t // 8) + t % 8 * 2]))
                        fv0, fv1 = txl.local_scalar('float32'), txl.local_scalar('float32')
                        txl.ptx.cvt.f32.bf16(fv0, bits0)
                        txl.ptx.cvt.f32.bf16(fv1, bits1)
                        qg2 = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](qg2, txl.cuda.make_float2(fv0, fv1), txl.cuda.make_float2(ld_shared_f32(smem, BSUM7_EG_S + tok * 4), ld_shared_f32(smem, BSUM7_EG_S + (tok + 1) * 4)))
                        txl.assign(qa[i], txl.bitwise_and(txl.reinterpret(txl.cuda.float2_x(qg2), ty='uint32'), txl.uint32(4294959104)))
                        txl.assign(qa[i + 1], txl.bitwise_and(txl.reinterpret(txl.cuda.float2_y(qg2), ty='uint32'), txl.uint32(4294959104)))
                    txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + BSUM7_TM_QA + h_ * 32, 'uint32'), *[qa[i] for i in range(32)])
                drow = t // 2
                c0 = t % 2 * 16
                dw = txl.alloc_local((8,), 'uint32')
                txl.ptx.ld.shared.v4.b32(dw[0], dw[1], dw[2], dw[3], smem.ptr_to([qb + BSUM7_DO_S + drow * BSUM7_QDO_UR + c0 * 2]))
                txl.ptx.ld.shared.v4.b32(dw[4], dw[5], dw[6], dw[7], smem.ptr_to([qb + BSUM7_DO_S + drow * BSUM7_QDO_UR + c0 * 2 + 16]))
                for c4 in range(4):
                    vals = []
                    for e in range(2):
                        lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                        unpack_bf16x2(dw[2 * c4 + e], lo, hi)
                        vals += [lo, hi]
                    txl.ptx.st.shared.v4.f32(smem.ptr_to([BSUM7_DOB_S + s2 * 8192 + merge_block_offset(drow, c0 // 4 + c4)]), vals[0], vals[1], vals[2], vals[3])
                txl.ptx['tcgen05.wait::st.sync.aligned']()
                txl.ptx['tcgen05.fence::before_thread_sync']()
                arrive(BSUM7_B_TOP)
                group_arrive_one(3, BSUM7_B_EMPTY + stage, t)
                with txl.if_(t == 0), txl.then_():
                    arrive(BSUM7_B_QEMPTY + qstage)

            with txl.if_(nch > 0), txl.then_():
                qdo_top(0)
            with txl.serial(nch) as it:
                txl.cuda.iket.mark('qdo-start')
                with txl.if_(it + 1 < nch), txl.then_():
                    qdo_top(it + 1)
                mbar_wait_n('qdo-mcacc', BSUM7_B_MCACC, it % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx[TMEM_LD_32X32B_X32](*[mv[i] for i in range(32)], txl.cast(tmem_addr + BSUM7_TM_DM, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                with txl.if_(it + 1 < nch), txl.then_():
                    txl.ptx['fence.proxy.async.shared::cta']()
                    for c4 in range(8):
                        tv = [txl.bitwise_and(mv[c4 * 4 + e], txl.uint32(4294959104)) for e in range(4)]
                        txl.ptx.st.shared.v4.b32(smem.ptr_to([BSUM7_MB_S + merge_block_offset(row, c4)]), tv[0], tv[1], tv[2], tv[3])
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                    txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                    with txl.if_(lane == 0), txl.then_():
                        arrive(BSUM7_B_BREADY)
                txl.cuda.iket.mark('qdo-end')
            with txl.if_(nch == 0), txl.then_():
                for c4 in range(8):
                    for e in range(4):
                        txl.assign(mv[c4 * 4 + e], txl.reinterpret(txl.if_then_else(cb * 32 + c4 * 4 + e == row, txl.float32(1.0), txl.float32(0.0)), ty='uint32'))
            for c4 in range(8):
                txl.ptx.st.global_.v4.b32(payload.ptr_to([(head * 128 + row) * 256 + 128 + cb * 32 + c4 * 4]), mv[c4 * 4], mv[c4 * 4 + 1], mv[c4 * 4 + 2], mv[c4 * 4 + 3])

        with txl.if_(txl.And(warp >= 4, warp < 8)), txl.then_():
            qdo_loop()
        def output_loop():
            """
            Output warps: carry dh through the chunks and store it.
            """
            t = tidx - 384
            tmem_addr = tmem_base + tmem_lane_offset(t)
            hv = txl.alloc_local((32,), 'uint32')
            for i in range(32):
                txl.assign(hv[i], txl.uint32(0))
            with txl.serial(nch) as it:
                txl.cuda.iket.mark('output-start')
                stage = it % BSUM7_NS
                stage_base = stage * BSUM7_STAGE
                s2 = it % 2
                chunk_start = chunk_t0(it)
                num_valid = txl.min(eos - chunk_start, 64)
                mbar_wait_n('output-full', BSUM7_B_FULL + stage, it // BSUM7_NS % 2)
                if lst:
                    # HB / DVA alternate between the two 64 B halves of their swizzle-atom rows, so
                    # HB(it) only waits for the loader-side copy of chunk it - 2 (one HBF barrier
                    # per half).
                    with txl.if_(it >= 2), txl.then_():
                        mbar_wait(BSUM7_B_HBF + it % 2, (it - 2) // 2 % 2)
                hb4 = txl.alloc_local((16,), 'uint32')
                for p in range(16):
                    txl.assign(hb4[p], pack_bf16x2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32')))
                txl.ptx['fence.proxy.async.shared::cta']()
                for c in range(4):
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([BSUM7_HB_S + t * 128 + txl.bitwise_xor(c + (it % 2 * 4 if lst else 0), txl.bitwise_and(t, 7)) * 16]), hb4[4 * c], hb4[4 * c + 1], hb4[4 * c + 2], hb4[4 * c + 3])
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(hb_bar(it))
                if store and not lst:
                    ci = nch - 1 - it
                    for c in range(4):
                        txl.ptx.st.global_.v4.b32(dh_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 16384 + umma128_offset(t, v0 // 8 + c) // 2]), hb4[4 * c], hb4[4 * c + 1], hb4[4 * c + 2], hb4[4 * c + 3])
                gl = ld_shared_f32(smem, stage_base + BSUM7_G_S + (num_valid - 1) * 4)
                egl = exp2_approx(gl)
                lane_group, lane_in_group = lane // 4, lane % 4
                vrow = [16 * (t // 32) + lane_group + 8 * hh for hh in range(2)]
                frh = [txl.local_scalar('float32', init=txl.if_then_else(vrow[hh] < num_valid, exp2_approx(gl - ld_shared_f32(smem, stage_base + BSUM7_G_S + vrow[hh] * 4)), txl.float32(0.0))) for hh in range(2)]
                lw2 = txl.alloc_local((8,), 'uint32')
                for j in range(4):
                    for hh in range(2):
                        txl.ptx.ld.shared.b32(lw2[j * 2 + hh], smem.ptr_to([stage_base + BSUM7_DV_S + vrow[hh] * BSUM7_UR + (8 * j + 2 * lane_in_group) * 2]))
                mbar_wait_n('output-dvp', BSUM7_B_DVP, it % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                acc16 = txl.alloc_local((16,), 'uint32')
                txl.ptx['tcgen05.ld.sync.aligned.16x256b.x4.b32'](*[acc16[i] for i in range(16)], txl.cast(tmem_addr + BSUM7_TM_DVP, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                dvf = txl.alloc_local((16,), 'float32')
                for j in range(4):
                    for hh in range(2):
                        lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                        unpack_bf16x2(lw2[j * 2 + hh], lo, hi)
                        dv2 = txl.local_scalar('uint64')
                        txl.ptx['fma.rn.f32x2'](dv2, txl.cuda.make_float2(txl.reinterpret(acc16[j * 4 + hh * 2], ty='float32'), txl.reinterpret(acc16[j * 4 + hh * 2 + 1], ty='float32')), txl.cuda.make_float2(frh[hh], frh[hh]), txl.cuda.make_float2(lo, hi))
                        ii = j * 4 + hh * 2
                        txl.assign(dvf[ii], txl.cuda.float2_x(dv2))
                        txl.assign(dvf[ii + 1], txl.cuda.float2_y(dv2))
                rbase = 16 * (t // 32)
                stmatrix_acc(smem, lambda row, nt: BSUM7_DVA_S + row * 128 + txl.bitwise_xor(nt + (it % 2 * 4 if lst else 0), txl.bitwise_and(row, 7)) * 16, dvf, 4, rbase, lane)
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(dva_bar(it))
                if store and not lst:
                    assert False, 'v16 bsum7 needs lst for dv2_out'
                mbar_wait_n('output-t2', BSUM7_B_T2, it % 2)
                mbar_wait_n('output-t1', BSUM7_B_T1 + s2, it // 2 % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                t1v, t2v = txl.alloc_local((32,), 'uint32'), txl.alloc_local((32,), 'uint32')
                txl.ptx[TMEM_LD_32X32B_X32](*[t1v[i] for i in range(32)], txl.cast(tmem_addr + BSUM7_TM_T1 + s2 * 32, 'uint32'))
                txl.ptx[TMEM_LD_32X32B_X32](*[t2v[i] for i in range(32)], txl.cast(tmem_addr + BSUM7_TM_T2, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                txl.ptx['tcgen05.fence::before_thread_sync']()
                arrive(BSUM7_B_T1F + s2)
                for p in range(16):
                    scaled = txl.local_scalar('uint64')
                    txl.ptx['mul.rn.f32x2'](scaled, txl.cuda.make_float2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32')), txl.cuda.make_float2(egl, egl))
                    step, updated = txl.local_scalar('uint64'), txl.local_scalar('uint64')
                    txl.ptx['fma.rn.f32x2'](step, txl.cuda.make_float2(txl.reinterpret(t1v[2 * p], ty='float32'), txl.reinterpret(t1v[2 * p + 1], ty='float32')), txl.cuda.make_float2(scale, scale), txl.cuda.make_float2(-txl.reinterpret(t2v[2 * p], ty='float32'), -txl.reinterpret(t2v[2 * p + 1], ty='float32')))
                    txl.ptx['add.rn.f32x2'](updated, scaled, step)
                    txl.assign(hv[2 * p], txl.reinterpret(txl.cuda.float2_x(updated), ty='uint32'))
                    txl.assign(hv[2 * p + 1], txl.reinterpret(txl.cuda.float2_y(updated), ty='uint32'))
                txl.cuda.iket.mark('output-end')
                group_arrive_one(2, BSUM7_B_EMPTY + stage, t)
            for i4 in range(8):
                txl.ptx.st.global_.v4.b32(payload.ptr_to([(head * 128 + t) * 256 + v0 + 4 * i4]), hv[4 * i4], hv[4 * i4 + 1], hv[4 * i4 + 2], hv[4 * i4 + 3])

        with txl.if_(warp >= 12), txl.then_():
            output_loop()
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        with txl.if_(warp == 8), txl.then_():
            txl.ptx['tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned']()
            txl.ptx['tcgen05.dealloc.cta_group::1.sync.aligned.b32'](txl.cast(tmem_base, 'uint32'), txl.uint32(512))
    return bsum7


BREC7_NS = 3
BREC7_UR = 80
BREC7_Q_S = 0
BREC7_K_S = BREC7_Q_S + 16384
BREC7_W_S = BREC7_K_S + 16384
BREC7_DO_S = BREC7_W_S + 16384
BREC7_DV_S = BREC7_DO_S + 64 * BREC7_UR
BREC7_G_S = BREC7_DV_S + 64 * BREC7_UR
BREC7_STAGE = (BREC7_G_S + 256 + 1023) // 1024 * 1024
BREC7_HB_S = BREC7_NS * BREC7_STAGE
BREC7_DVA_S = BREC7_HB_S + 16384
BREC7_DOB_S = BREC7_DVA_S + 8192
BREC7_MB_S = BREC7_HB_S
BREC7_BAR_S = BREC7_DOB_S + 2 * 8192
BREC7_HOLD_S = BREC7_BAR_S + 256
BREC7_EG_S = BREC7_HOLD_S + 64
BREC7_SMEM = BREC7_EG_S + 256
BREC7_TM_DH = 0
BREC7_TM_DVP = 32
BREC7_TM_T2 = 64
BREC7_TM_T1 = 96
BREC7_TM_QA = 160
BREC7_TM_MA = 288
BREC7_TM_COLS = 512
BREC7_ID_M64 = 67699856
BREC7_ID_UPD = 134841488
BREC7_ID_TF32 = 134809872
BREC7_B_FULL = 0
BREC7_B_EMPTY = BREC7_NS
BREC7_B_HB = 2 * BREC7_NS
BREC7_B_DVP = BREC7_B_HB + 1
BREC7_B_DVA = BREC7_B_DVP + 1
BREC7_B_T2 = BREC7_B_DVA + 1
BREC7_B_DH = BREC7_B_T2 + 1
BREC7_B_TOP = BREC7_B_DH + 1
BREC7_B_T1 = BREC7_B_TOP + 2
BREC7_B_T1F = BREC7_B_T1 + 2
BREC7_B_MRDY = BREC7_B_T1F + 2
BREC7_B_MDONE = BREC7_B_MRDY + 1


def build_brec7(HV, summary=False, merge=False, hkv=True, bst=False):
    """
    The dh recurrence, with the merge of the successors' gradient summaries (brec7m). bst: dh / dv2
    are stored by a separate store warp.
    """
    bst = bst and not summary
    NWARPS = 12 if bst else 11
    BREC7_B_HBF = BREC7_B_MDONE + 1

    def prepare_host(params):
        maps = []
        for name in ('q', 'k', 'w'):
            desc = txl.stack_alloca('tensormap', 1)
            txl.cu_tensor_map_encode_tiled(desc, 'bfloat16', 3, params[name].data, 64, params['S'], 2 * HV, HV * 256, 128, 64, 64, 2, 1, 1, 1, 0, 3, 2, 0)
            maps.append(desc)
        return tuple(maps)

    @txl.kernel()
    def brec7(q: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), k: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), do: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dv_local: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), g_cum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), nseg: txl.int32, c_lo: txl.int32, c_hi: txl.int32, scale: txl.float32, dh_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dv2_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), summ: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), ag: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), m_first: txl.int32, m_step: txl.int32, m_n: txl.int32, S: txl.int32):
        host = prepare_host(dict(q=q, k=k, w=w, S=S))
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[4, HV], block=NWARPS * 32))
        q_map, k_map, w_map = host
        smem = txl.alloc_tensor((BREC7_SMEM,), txl.u8, scope='shared.dyn', align=1024)
        cta = txl.cta_id()
        m_n1 = txl.local_scalar('int32', init=txl.if_then_else(m_n > 1, m_n - 1, 0))
        vblk = cta[0]
        head = cta[1]
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        nchunks = c_hi - c_lo
        v0 = vblk * 32
        cl = txl.local_scalar('int32', init=txl.max(c_hi - 1, 0))

        def mbar_addr(i):
            return smem.ptr_to([BREC7_BAR_S + i * 8])

        def mbar_wait(i, parity):
            ready = txl.local_scalar('uint32', init=txl.uint32(0))
            with txl.while_(ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready, mbar_addr(i), txl.cast(parity, 'uint32'))

        def arrive(i):
            txl.ptx.mbarrier.arrive.shared.b64(mbar_addr(i))

        def wg_sync_arrive_one(bar, t):
            txl.ptx.bar.sync(txl.uint32(1 + warp // 4), txl.uint32(128))
            with txl.if_(t == 0), txl.then_():
                arrive(bar)
        with txl.if_(tidx == 0), txl.then_():
            for s_ in range(BREC7_NS):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(BREC7_B_FULL + s_), txl.uint32(65))
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(BREC7_B_EMPTY + s_), txl.uint32(3))
            for i_, cnt in ((BREC7_B_HB, 4), (BREC7_B_DVP, 1), (BREC7_B_DVA, 4), (BREC7_B_T2, 1), (BREC7_B_DH, 128), (BREC7_B_TOP, 128), (BREC7_B_TOP + 1, 128), (BREC7_B_T1, 1), (BREC7_B_T1 + 1, 1), (BREC7_B_T1F, 128), (BREC7_B_T1F + 1, 128), (BREC7_B_MRDY, 128), (BREC7_B_MDONE, 1), (BREC7_B_HBF, 32)):
                txl.ptx.mbarrier.init.shared.b64(mbar_addr(i_), txl.uint32(cnt))
        txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.if_(warp == 8), txl.then_():
            txl.ptx['tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32'](smem.ptr_to([BREC7_HOLD_S]), txl.uint32(BREC7_TM_COLS))
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        tmem_base = txl.local_scalar('int32')
        txl.ptx.ld.volatile.shared.s32(tmem_base, smem.ptr_to([BREC7_HOLD_S]))
        def load_loop():
            """
            Loader warps: copy each chunk's operands into the stage ring.
            """
            loader_tidx = tidx - 288
            cur = ChunkCursor(cu, nseg, cl)
            with txl.serial(nchunks) as it_c:
                stage = it_c % BREC7_NS
                stage_base = stage * BREC7_STAGE
                with txl.if_(it_c >= BREC7_NS), txl.then_():
                    mbar_wait(BREC7_B_EMPTY + stage, (it_c // BREC7_NS - 1) % 2)
                    txl.ptx['fence.proxy.async.shared::cta']()
                chunk_start, num_valid = txl.local_scalar('int32', init=cur.chunk_start), txl.local_scalar('int32', init=cur.num_valid())
                with txl.if_(txl.And(num_valid == 64, loader_tidx == 0)), txl.then_():
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar_addr(BREC7_B_FULL + stage), txl.uint32(49152))
                    for tmap, dst in ((q_map, stage_base + BREC7_Q_S), (k_map, stage_base + BREC7_K_S), (w_map, stage_base + BREC7_W_S)):
                        txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([dst]), txl.address_of(tmap), txl.int32(0), chunk_start, txl.cast(2 * head, 'int32'), mbar_addr(BREC7_B_FULL + stage), txl.uint64(0x12F0000000000000))
                with txl.if_(num_valid < 64), txl.then_():
                    for rep in range(16):
                        idx = rep * 64 + loader_tidx
                        row, ch = idx // 16, idx % 16
                        ok = row < num_valid
                        src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                        for src, dst in ((q, BREC7_Q_S), (k, BREC7_K_S), (w, BREC7_W_S)):
                            txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + dst + mn_major_offset(row, ch)]), src.ptr_to([(src_row * HV + head) * 128 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                    with txl.if_(loader_tidx == 0), txl.then_():
                        txl.ptx.mbarrier.arrive.shared.b64(mbar_addr(BREC7_B_FULL + stage))
                for rep in range(4):
                    idx = rep * 64 + loader_tidx
                    row, ch = idx // 4, idx % 4
                    ok = row < num_valid
                    src_row = txl.if_then_else(ok, chunk_start + row, chunk_start)
                    for src, dst in ((do, BREC7_DO_S), (dv_local, BREC7_DV_S)):
                        txl.ptx[CP_ASYNC_16](smem.ptr_to([stage_base + dst + row * BREC7_UR + ch * 16]), src.ptr_to([(src_row * HV + head) * 128 + v0 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                ok = loader_tidx < num_valid
                src_row = txl.if_then_else(ok, chunk_start + loader_tidx, chunk_start)
                txl.ptx[CP_ASYNC_4](smem.ptr_to([stage_base + BREC7_G_S + loader_tidx * 4]), g_cum.ptr_to([src_row * HV + head]), 4, txl.cast(txl.if_then_else(ok, 4, 0), 'uint32'))
                txl.ptx['cp.async.mbarrier.arrive.noinc.shared::cta.b64'](mbar_addr(BREC7_B_FULL + stage))
                cur.retreat()

        with txl.if_(txl.And(warp >= 9, warp < 11)), txl.then_():
            load_loop()
        def mma_loop():
            """
            MMA warp: issue the merge and per-chunk UMMAs of the dh recurrence.
            """
            leader = txl.cuda.elect_sync()

            def mma(op, d, a, b, idesc, acc):
                txl.ptx[op](txl.cast(d, 'uint32'), a, b, txl.uint32(idesc), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(acc)), pred=leader)

            def issue_t1(it):
                s2 = it % 2
                mbar_wait(BREC7_B_TOP + s2, it // 2 % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                with txl.if_(it >= 2), txl.then_():
                    mbar_wait(BREC7_B_T1F + s2, (it // 2 - 1) % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                bd = umma_desc(smem, BREC7_DOB_S + s2 * 8192, UMMA_DESC_MN_TF32)
                for kp in range(8):
                    mma(UMMA_TF32, tmem_base + BREC7_TM_T1 + s2 * 32, txl.cast(tmem_base + BREC7_TM_QA + s2 * 64 + kp * 8, 'uint32'), bd + txl.uint64(kp * 64), BREC7_ID_TF32, 0 if kp == 0 else 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(BREC7_B_T1 + s2), pred=leader)
            if merge:
                with txl.serial(m_n1) as j:
                    mbar_wait(BREC7_B_MRDY, j % 2)
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    bd = umma_desc(smem, BREC7_MB_S, UMMA_DESC_MN_TF32)
                    for kp in range(16):
                        mma(UMMA_TF32, tmem_base + BREC7_TM_DH, txl.cast(tmem_base + BREC7_TM_MA + kp * 8, 'uint32'), bd + txl.uint64(kp * 64), BREC7_ID_TF32, 0 if kp == 0 else 1)
                    txl.ptx[UMMA_COMMIT](mbar_addr(BREC7_B_MDONE), pred=leader)
            with txl.if_(nchunks > 0), txl.then_():
                issue_t1(0)
            with txl.serial(nchunks) as it_c:
                stage = it_c % BREC7_NS
                stage_base = stage * BREC7_STAGE
                mbar_wait(BREC7_B_FULL + stage, it_c // BREC7_NS % 2)
                mbar_wait(BREC7_B_HB, it_c % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                hb_d, k_d = umma_desc(smem, BREC7_HB_S, UMMA_DESC_MN), umma_desc(smem, stage_base + BREC7_K_S, UMMA_DESC_K)
                for kp in range(8):
                    mma(UMMA_F16, tmem_base + BREC7_TM_DVP, k_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), hb_d + txl.uint64(kp * 128), BREC7_ID_M64, 0 if kp == 0 else 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(BREC7_B_DVP), pred=leader)
                mbar_wait(BREC7_B_DVA, it_c % 2)
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                w_d, dva_d = umma_desc(smem, stage_base + BREC7_W_S, UMMA_DESC_MN), umma_desc(smem, BREC7_DVA_S, UMMA_DESC_MN)
                for kp in range(4):
                    mma(UMMA_F16, tmem_base + BREC7_TM_T2, w_d + txl.uint64(kp * 128), dva_d + txl.uint64(kp * 128), BREC7_ID_UPD, 0 if kp == 0 else 1)
                txl.ptx[UMMA_COMMIT](mbar_addr(BREC7_B_T2), pred=leader)
                txl.ptx[UMMA_COMMIT](mbar_addr(BREC7_B_EMPTY + stage), pred=leader)
                with txl.if_(it_c + 1 < nchunks), txl.then_():
                    issue_t1(it_c + 1)

        with txl.if_(warp == 8), txl.then_():
            mma_loop()
        def qdo_loop():
            """
            q / do warps: build the gated q and do operands of each chunk.
            """
            t = tidx - 128
            tmem_addr = tmem_base + tmem_lane_offset(t)
            with txl.serial(nchunks) as it_c:
                stage = it_c % BREC7_NS
                stage_base = stage * BREC7_STAGE
                s2 = it_c % 2
                mbar_wait(BREC7_B_FULL + stage, it_c // BREC7_NS % 2)
                with txl.if_(it_c >= 2), txl.then_():
                    mbar_wait(BREC7_B_T1 + s2, (it_c // 2 - 1) % 2)
                with txl.if_(t < 64), txl.then_():
                    txl.ptx.st.shared.f32(smem.ptr_to([BREC7_EG_S + t * 4]), exp2_approx(ld_shared_f32(smem, stage_base + BREC7_G_S + t * 4)))
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))
                txl.ptx['tcgen05.fence::after_thread_sync']()
                qa = txl.alloc_local((64,), 'uint32')
                for tp in range(32):
                    tok = tp * 2
                    bits0, bits1 = txl.local_scalar('uint16'), txl.local_scalar('uint16')
                    txl.ptx.ld.shared.b16(bits0, smem.ptr_to([stage_base + BREC7_Q_S + mn_major_offset(tok, t // 8) + t % 8 * 2]))
                    txl.ptx.ld.shared.b16(bits1, smem.ptr_to([stage_base + BREC7_Q_S + mn_major_offset(tok + 1, t // 8) + t % 8 * 2]))
                    fv0, fv1 = txl.local_scalar('float32'), txl.local_scalar('float32')
                    txl.ptx.cvt.f32.bf16(fv0, bits0)
                    txl.ptx.cvt.f32.bf16(fv1, bits1)
                    qg2 = txl.local_scalar('uint64')
                    txl.ptx['mul.rn.f32x2'](qg2, txl.cuda.make_float2(fv0, fv1), txl.cuda.make_float2(ld_shared_f32(smem, BREC7_EG_S + tok * 4), ld_shared_f32(smem, BREC7_EG_S + (tok + 1) * 4)))
                    txl.assign(qa[tok], txl.bitwise_and(txl.reinterpret(txl.cuda.float2_x(qg2), ty='uint32'), txl.uint32(4294959104)))
                    txl.assign(qa[tok + 1], txl.bitwise_and(txl.reinterpret(txl.cuda.float2_y(qg2), ty='uint32'), txl.uint32(4294959104)))
                for h_ in range(2):
                    txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + BREC7_TM_QA + s2 * 64 + h_ * 32, 'uint32'), *[qa[h_ * 32 + i] for i in range(32)])
                row = t // 2
                c0 = t % 2 * 16
                dw = txl.alloc_local((8,), 'uint32')
                txl.ptx.ld.shared.v4.b32(dw[0], dw[1], dw[2], dw[3], smem.ptr_to([stage_base + BREC7_DO_S + row * BREC7_UR + c0 * 2]))
                txl.ptx.ld.shared.v4.b32(dw[4], dw[5], dw[6], dw[7], smem.ptr_to([stage_base + BREC7_DO_S + row * BREC7_UR + c0 * 2 + 16]))
                txl.ptx['fence.proxy.async.shared::cta']()
                for c4 in range(4):
                    vals = []
                    for e in range(2):
                        lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                        unpack_bf16x2(dw[2 * c4 + e], lo, hi)
                        vals += [lo, hi]
                    txl.ptx.st.shared.v4.f32(smem.ptr_to([BREC7_DOB_S + s2 * 8192 + merge_block_offset(row, c0 // 4 + c4)]), vals[0], vals[1], vals[2], vals[3])
                txl.ptx['tcgen05.wait::st.sync.aligned']()
                txl.ptx['tcgen05.fence::before_thread_sync']()
                arrive(BREC7_B_TOP + s2)
                wg_sync_arrive_one(BREC7_B_EMPTY + stage, t)

        with txl.if_(txl.And(warp >= 4, warp < 8)), txl.then_():
            qdo_loop()
        if bst:
            def store_loop():
                """
                Store warps: copy each chunk's dh and dv2 from shared memory to global memory.
                """
                curc = ChunkCursor(cu, nseg, cl)
                with txl.serial(nchunks) as it_c:
                    ci = c_hi - 1 - it_c
                    _ = txl.local_scalar('int32', init=curc.chunk_start)  # unused; kept so the traced kernel stays the same
                    num_valid = txl.local_scalar('int32', init=curc.num_valid())
                    mbar_wait(BREC7_B_HB, it_c % 2)
                    for j in range(16):
                        p = lane + 32 * j
                        row, c = p // 4, p % 4
                        wd = txl.alloc_local((4,), 'uint32')
                        txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([BREC7_HB_S + row * 128 + txl.bitwise_xor(c, txl.bitwise_and(row, 7)) * 16]))
                        txl.ptx.st.global_.v4.b32(dh_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 16384 + umma128_offset(row, v0 // 8 + c) // 2]), wd[0], wd[1], wd[2], wd[3])
                    mbar_wait(BREC7_B_DVA, it_c % 2)
                    for j in range(8):
                        p = lane + 32 * j
                        row, c = p // 4, p % 4
                        wd = txl.alloc_local((4,), 'uint32')
                        for e in range(4):
                            txl.assign(wd[e], txl.uint32(0))
                        with txl.if_(row < num_valid), txl.then_():
                            txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([BREC7_DVA_S + row * 128 + txl.bitwise_xor(c, txl.bitwise_and(row, 7)) * 16]))
                        txl.ptx.st.global_.v4.b32(dv2_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 8192 + umma64_offset(row, v0 // 8 + c) // 2]), wd[0], wd[1], wd[2], wd[3])
                    arrive(BREC7_B_HBF)
                    curc.retreat()

            with txl.if_(warp == 11), txl.then_():
                store_loop()
        def state_loop():
            """
            State warps: merge the successors' summaries and carry dh through the chunks.
            """
            t = tidx
            tmem_addr = tmem_base + tmem_lane_offset(t)
            hv = txl.alloc_local((32,), 'uint32')
            for i in range(32):
                txl.assign(hv[i], txl.uint32(0))
            if merge:
                hb_ = (txl.cast(m_first, 'int64') * HV + head) * (128 * 256)
                with txl.if_(m_n > 0), txl.then_():
                    agv = txl.alloc_local((32,), 'uint32')
                    for i4 in range(8):
                        txl.ptx.ld.global_.nc.v4.b32(agv[4 * i4], agv[4 * i4 + 1], agv[4 * i4 + 2], agv[4 * i4 + 3], ag.ptr_to([hb_ + t * 256 + v0 + 4 * i4]))
                    for i in range(32):
                        txl.assign(hv[i], txl.reinterpret(txl.reinterpret(hv[i], ty='float32') + txl.reinterpret(agv[i], ty='float32'), ty='uint32'))
                with txl.serial(m_n1) as jj:
                    base = (txl.cast(m_first + (jj + 1) * m_step, 'int64') * HV + head) * (128 * 256)
                    with txl.if_(jj > 0), txl.then_():
                        mbar_wait(BREC7_B_MDONE, (jj - 1) % 2)
                        txl.ptx['tcgen05.fence::after_thread_sync']()
                        txl.ptx[TMEM_LD_32X32B_X32](*[hv[i] for i in range(32)], txl.cast(tmem_addr + BREC7_TM_DH, 'uint32'))
                        txl.ptx['tcgen05.wait::ld.sync.aligned']()
                        agv = txl.alloc_local((32,), 'uint32')
                        for i4 in range(8):
                            txl.ptx.ld.global_.nc.v4.b32(agv[4 * i4], agv[4 * i4 + 1], agv[4 * i4 + 2], agv[4 * i4 + 3], ag.ptr_to([base - txl.cast(m_step, 'int64') * HV * (128 * 256) + t * 256 + v0 + 4 * i4]))
                        for i in range(32):
                            txl.assign(hv[i], txl.reinterpret(txl.reinterpret(hv[i], ty='float32') + txl.reinterpret(agv[i], ty='float32'), ty='uint32'))
                    txl.ptx['fence.proxy.async.shared::cta']()
                    for c4 in range(8):
                        tv = [txl.bitwise_and(hv[c4 * 4 + e], txl.uint32(4294959104)) for e in range(4)]
                        txl.ptx.st.shared.v4.b32(smem.ptr_to([BREC7_MB_S + merge_block_offset(t, c4)]), tv[0], tv[1], tv[2], tv[3])
                    ma = txl.alloc_local((32,), 'uint32')
                    for q4 in range(4):
                        for i4 in range(8):
                            txl.ptx.ld.global_.nc.v4.b32(ma[4 * i4], ma[4 * i4 + 1], ma[4 * i4 + 2], ma[4 * i4 + 3], ag.ptr_to([base + t * 256 + 128 + q4 * 32 + 4 * i4]))
                        for i in range(32):
                            txl.assign(ma[i], txl.bitwise_and(ma[i], txl.uint32(4294959104)))
                        txl.ptx[TMEM_ST_32X32B_X32](txl.cast(tmem_addr + BREC7_TM_MA + q4 * 32, 'uint32'), *[ma[i] for i in range(32)])
                    txl.ptx['tcgen05.wait::st.sync.aligned']()
                    txl.ptx['fence.proxy.async.shared::cta']()
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                    arrive(BREC7_B_MRDY)
                with txl.if_(m_n > 1), txl.then_():
                    mbar_wait(BREC7_B_MDONE, (m_n - 2) % 2)
                    txl.ptx['tcgen05.fence::after_thread_sync']()
                    txl.ptx[TMEM_LD_32X32B_X32](*[hv[i] for i in range(32)], txl.cast(tmem_addr + BREC7_TM_DH, 'uint32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    lastb = (txl.cast(m_first + (m_n - 1) * m_step, 'int64') * HV + head) * (128 * 256)
                    agv = txl.alloc_local((32,), 'uint32')
                    for i4 in range(8):
                        txl.ptx.ld.global_.nc.v4.b32(agv[4 * i4], agv[4 * i4 + 1], agv[4 * i4 + 2], agv[4 * i4 + 3], ag.ptr_to([lastb + t * 256 + v0 + 4 * i4]))
                    for i in range(32):
                        txl.assign(hv[i], txl.reinterpret(txl.reinterpret(hv[i], ty='float32') + txl.reinterpret(agv[i], ty='float32'), ty='uint32'))
            cur = ChunkCursor(cu, nseg, cl)
            with txl.serial(nchunks) as it_c:
                ci = c_hi - 1 - it_c
                stage = it_c % BREC7_NS
                stage_base = stage * BREC7_STAGE
                s2 = it_c % 2
                _ = txl.local_scalar('int32', init=cur.chunk_start)  # unused; kept so the traced kernel stays the same
                num_valid = txl.local_scalar('int32', init=cur.num_valid())
                with txl.if_(txl.And(it_c != 0, cur.eos - cur.chunk_start <= 64)), txl.then_():
                    for i in range(32):
                        txl.assign(hv[i], txl.uint32(0))
                mbar_wait(BREC7_B_FULL + stage, it_c // BREC7_NS % 2)
                if bst:
                    with txl.if_(it_c >= 1), txl.then_():
                        mbar_wait(BREC7_B_HBF, (it_c - 1) % 2)
                hb4 = txl.alloc_local((16,), 'uint32')
                for p in range(16):
                    txl.assign(hb4[p], pack_bf16x2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32')))
                txl.ptx['fence.proxy.async.shared::cta']()
                for c in range(4):
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([BREC7_HB_S + t * 128 + txl.bitwise_xor(c, txl.bitwise_and(t, 7)) * 16]), hb4[4 * c], hb4[4 * c + 1], hb4[4 * c + 2], hb4[4 * c + 3])
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(BREC7_B_HB)
                if not summary and not bst:
                    for c in range(4):
                        txl.ptx.st.global_.v4.b32(dh_out.ptr_to([(txl.cast(ci, 'int64') * HV + head) * 16384 + umma128_offset(t, v0 // 8 + c) // 2]), hb4[4 * c], hb4[4 * c + 1], hb4[4 * c + 2], hb4[4 * c + 3])
                gl = ld_shared_f32(smem, stage_base + BREC7_G_S + (num_valid - 1) * 4)
                egl = exp2_approx(gl)
                lane_group, lane_in_group = lane // 4, lane % 4
                vrow = [16 * (t // 32) + lane_group + 8 * hh for hh in range(2)]
                frh = [txl.local_scalar('float32', init=txl.if_then_else(vrow[hh] < num_valid, exp2_approx(gl - ld_shared_f32(smem, stage_base + BREC7_G_S + vrow[hh] * 4)), txl.float32(0.0))) for hh in range(2)]
                lw2 = txl.alloc_local((8,), 'uint32')
                for j in range(4):
                    for hh in range(2):
                        txl.ptx.ld.shared.b32(lw2[j * 2 + hh], smem.ptr_to([stage_base + BREC7_DV_S + vrow[hh] * BREC7_UR + (8 * j + 2 * lane_in_group) * 2]))
                mbar_wait(BREC7_B_DVP, it_c % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                acc16 = txl.alloc_local((16,), 'uint32')
                txl.ptx['tcgen05.ld.sync.aligned.16x256b.x4.b32'](*[acc16[i] for i in range(16)], txl.cast(tmem_addr + BREC7_TM_DVP, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                dvf = txl.alloc_local((16,), 'float32')
                for j in range(4):
                    for hh in range(2):
                        lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                        unpack_bf16x2(lw2[j * 2 + hh], lo, hi)
                        dv2 = txl.local_scalar('uint64')
                        txl.ptx['fma.rn.f32x2'](dv2, txl.cuda.make_float2(txl.reinterpret(acc16[j * 4 + hh * 2], ty='float32'), txl.reinterpret(acc16[j * 4 + hh * 2 + 1], ty='float32')), txl.cuda.make_float2(frh[hh], frh[hh]), txl.cuda.make_float2(lo, hi))
                        ii = j * 4 + hh * 2
                        txl.assign(dvf[ii], txl.cuda.float2_x(dv2))
                        txl.assign(dvf[ii + 1], txl.cuda.float2_y(dv2))
                rbase = 16 * (t // 32)
                stmatrix_acc(smem, lambda row, nt: BREC7_DVA_S + row * 128 + txl.bitwise_xor(nt, txl.bitwise_and(row, 7)) * 16, dvf, 4, rbase, lane)
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    arrive(BREC7_B_DVA)
                if not summary and not bst:
                    assert False, 'v16 brec7 needs bst for dv2_out'
                wg_sync_arrive_one(BREC7_B_EMPTY + stage, t)
                mbar_wait(BREC7_B_T1 + s2, it_c // 2 % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                t1v = txl.alloc_local((32,), 'uint32')
                txl.ptx[TMEM_LD_32X32B_X32](*[t1v[i] for i in range(32)], txl.cast(tmem_addr + BREC7_TM_T1 + s2 * 32, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                txl.ptx['tcgen05.fence::before_thread_sync']()
                arrive(BREC7_B_T1F + s2)
                for p in range(16):
                    pre = txl.local_scalar('uint64')
                    txl.ptx['mul.rn.f32x2'](pre, txl.cuda.make_float2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32')), txl.cuda.make_float2(egl, egl))
                    txl.assign(hv[2 * p], txl.reinterpret(txl.cuda.float2_x(pre), ty='uint32'))
                    txl.assign(hv[2 * p + 1], txl.reinterpret(txl.cuda.float2_y(pre), ty='uint32'))
                mbar_wait(BREC7_B_T2, it_c % 2)
                txl.ptx['tcgen05.fence::after_thread_sync']()
                t2v = txl.alloc_local((32,), 'uint32')
                txl.ptx[TMEM_LD_32X32B_X32](*[t2v[i] for i in range(32)], txl.cast(tmem_addr + BREC7_TM_T2, 'uint32'))
                txl.ptx['tcgen05.wait::ld.sync.aligned']()
                txl.ptx['tcgen05.fence::before_thread_sync']()
                for p in range(16):
                    scaled = txl.cuda.make_float2(txl.reinterpret(hv[2 * p], ty='float32'), txl.reinterpret(hv[2 * p + 1], ty='float32'))
                    step, updated = txl.local_scalar('uint64'), txl.local_scalar('uint64')
                    txl.ptx['fma.rn.f32x2'](step, txl.cuda.make_float2(txl.reinterpret(t1v[2 * p], ty='float32'), txl.reinterpret(t1v[2 * p + 1], ty='float32')), txl.cuda.make_float2(scale, scale), txl.cuda.make_float2(-txl.reinterpret(t2v[2 * p], ty='float32'), -txl.reinterpret(t2v[2 * p + 1], ty='float32')))
                    txl.ptx['add.rn.f32x2'](updated, scaled, step)
                    txl.assign(hv[2 * p], txl.reinterpret(txl.cuda.float2_x(updated), ty='uint32'))
                    txl.assign(hv[2 * p + 1], txl.reinterpret(txl.cuda.float2_y(updated), ty='uint32'))
                cur.retreat()
            if summary:
                for i4 in range(8):
                    txl.ptx.st.global_.v4.b32(summ.ptr_to([(head * 128 + t) * 256 + v0 + 4 * i4]), hv[4 * i4], hv[4 * i4 + 1], hv[4 * i4 + 2], hv[4 * i4 + 3])

        with txl.if_(warp < 4), txl.then_():
            state_loop()
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        with txl.if_(warp == 8), txl.then_():
            txl.ptx['tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned']()
            txl.ptx['tcgen05.dealloc.cta_group::1.sync.aligned.b32'](txl.cast(tmem_base, 'uint32'), txl.uint32(BREC7_TM_COLS))
    return brec7


BPREP2_NW = 8
BPREP2_NT = BPREP2_NW * 32
BPREP2_K_S = 0
BPREP2_V_S = 16 * KIB
BPREP2_Q_S = 32 * KIB
BPREP2_DO_S = 48 * KIB
BPREP2_KB_S = 64 * KIB
BPREP2_AB_S = 80 * KIB
BPREP2_KQ_S = 88 * KIB
BPREP2_G_S = 96 * KIB
BPREP2_B_S = BPREP2_G_S + 256
BPREP2_BARS_S = BPREP2_B_S + 256
BPREP2_SMEM = BPREP2_BARS_S + 64
BPREP2_ID_64x64_KK = 68158608
BPREP2_ID_64x128_KMN = 69272720


def build_bprep2(HV, ptg=False, persistent=False):
    """
    The backward preparation kernel: w, u and dv_local per chunk.
    ptg: kq = k q^T, u = A vb, w = A kb and dv_local = kq do run as tcgen05 UMMAs (fp32 TMEM
    accumulators, the mma.sync K order) issued by one thread; k / v / q / do / kb tiles then live
    in the UMMA SW128 layout.
    """
    sT = umma64_offset if ptg else sw256_offset

    @txl.kernel()
    def bprep2(q: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), k: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), v: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), do: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), g_cum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), beta: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), a_in: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), nseg: txl.int32, n_chunks: txl.int32, scale: txl.float32, w_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), u_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dvl_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[num_ctas] if persistent else [n_chunks, HV], block=BPREP2_NW * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=2))
        smem = txl.alloc_tensor((BPREP2_SMEM,), txl.u8, scope='shared.dyn', align=1024)
        cta = txl.cta_id()
        tile_parity = txl.local_scalar('int32', init=txl.int32(0))
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        lane_group, lane_in_group = lane // 4, lane % 4
        rs, chh = warp % 4, warp // 4
        r0 = rs * 16
        if ptg:
            with txl.if_(tidx == 0), txl.then_():
                for i_ in range(4):
                    txl.ptx.mbarrier.init.shared.b64(smem.ptr_to([BPREP2_BARS_S + i_ * 8]), txl.uint32(1))
            txl.ptx.fence.mbarrier_init.release.cluster()
            with txl.if_(warp == 0), txl.then_():
                txl.ptx['tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32'](smem.ptr_to([BPREP2_BARS_S + 32]), txl.uint32(256))
                txl.ptx['tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned']()
            txl.ptx['tcgen05.fence::before_thread_sync']()
            txl.cuda.cta_sync()
            txl.ptx['tcgen05.fence::after_thread_sync']()
        tmem_base = txl.local_scalar('uint32', init=txl.uint32(0))
        if ptg:
            txl.ptx.ld.volatile.shared.u32(tmem_base, smem.ptr_to([BPREP2_BARS_S + 32]))

        def sdesc(off, base):
            addr = txl.cuda.cvta_generic_to_shared(smem.ptr_to([off]))
            return txl.bitwise_or(txl.uint64(base), txl.cast(txl.bitwise_and(txl.shift_right(addr, txl.uint32(4)), txl.uint32(16383)), 'uint64'))

        def issue(d_col, a_off, a_base, a_step, b_off, b_base, b_step, nk, idesc, bar_i):
            with txl.if_(warp == 0), txl.then_():
                leader = txl.cuda.elect_sync()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx['fence.proxy.async.shared::cta']()
                a_d, b_d = sdesc(a_off, a_base), sdesc(b_off, b_base)
                for kp in range(nk):
                    txl.ptx[UMMA_F16](txl.cast(tmem_base + d_col, 'uint32'), a_d + txl.uint64(a_step(kp)), b_d + txl.uint64(b_step(kp)), txl.uint32(idesc), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](smem.ptr_to([BPREP2_BARS_S + bar_i * 8]), pred=leader)

        def tm_ld(dst, d_col, bar_i, ncols, fence=True):
            ready = txl.local_scalar('uint32', init=txl.uint32(0))
            with txl.while_(ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready, smem.ptr_to([BPREP2_BARS_S + bar_i * 8]), txl.cast(tile_parity, 'uint32'))
            if fence:
                txl.ptx['fence.proxy.async.shared::cta']()
            txl.ptx['tcgen05.fence::after_thread_sync']()
            nr = ncols // 2
            raw = txl.alloc_local((nr,), 'uint32')
            txl.ptx[TMEM_LD_16X256B_X8 if ncols == 64 else TMEM_LD_16X256B_X4](*[raw[i] for i in range(nr)], txl.cast(tmem_base + txl.shift_left(txl.cast(rs * 32, 'uint32'), txl.uint32(16)) + d_col + chh * ncols, 'uint32'))
            txl.ptx['tcgen05.wait::ld.sync.aligned']()
            for i in range(nr):
                txl.assign(dst[i], txl.reinterpret(raw[i], ty='float32'))

        def tile(c_in, head_in):
            c, head = txl.local_scalar('int32', init=c_in), txl.local_scalar('int32', init=head_in)
            chunk_start, num_valid = chunk_lookup(cu, nseg, c)
            with txl.if_(num_valid > 0), txl.then_():
                for it in range(16 * 64 // BPREP2_NT):
                    idx = it * BPREP2_NT + tidx
                    row, ch = idx // 16, idx % 16
                    ok = row < num_valid
                    srow = txl.if_then_else(ok, chunk_start + row, chunk_start)
                    for src, dst in ((k, BPREP2_K_S), (v, BPREP2_V_S), (q, BPREP2_Q_S), (do, BPREP2_DO_S)):
                        txl.ptx[CP_ASYNC_16](smem.ptr_to([dst + sT(row, ch)]), src.ptr_to([(srow * HV + head) * 128 + ch * 8]), 16, txl.cast(txl.if_then_else(ok, 16, 0), 'uint32'))
                for it in range(8 * 64 // BPREP2_NT):
                    idx = it * BPREP2_NT + tidx
                    row, ch = idx // 8, idx % 8
                    ok = row < num_valid
                    srow = txl.if_then_else(ok, chunk_start + row, chunk_start)
                    txl.ptx[CP_ASYNC_16](smem.ptr_to([BPREP2_AB_S + sw128_offset(row, ch)]), a_in.ptr_to([(srow * HV + head) * 64 + ch * 8]), 16, txl.cast(txl.if_then_else(txl.And(ok, ch <= row // 8), 16, 0), 'uint32'))
                with txl.if_(tidx < 64), txl.then_():
                    ok = tidx < num_valid
                    srow = txl.if_then_else(ok, chunk_start + tidx, chunk_start)
                    txl.ptx[CP_ASYNC_4](smem.ptr_to([BPREP2_G_S + tidx * 4]), g_cum.ptr_to([srow * HV + head]), 4, txl.cast(txl.if_then_else(ok, 4, 0), 'uint32'))
                txl.ptx.cp.async_.commit_group()
                with txl.if_(tidx < 64), txl.then_():
                    bv = txl.local_scalar('float32', init=txl.float32(0.0))
                    with txl.if_(tidx < num_valid), txl.then_():
                        bits = txl.local_scalar('uint16')
                        txl.ptx.ld.global_.b16(bits, beta.ptr_to([(chunk_start + tidx) * HV + head]))
                        txl.ptx.cvt.f32.bf16(bv, bits)
                    txl.ptx.st.shared.f32(smem.ptr_to([BPREP2_B_S + tidx * 4]), bv)
                txl.ptx.cp.async_.wait_group(0)
                txl.cuda.cta_sync()
                if ptg:
                    issue(0, BPREP2_K_S, UMMA_DESC_K, lambda kp: kp % 4 * 2 + kp // 4 * 512, BPREP2_Q_S, UMMA_DESC_K, lambda kp: kp % 4 * 2 + kp // 4 * 512, 8, BPREP2_ID_64x64_KK, 0)
                for it in range(16 * 64 // BPREP2_NT):
                    idx = it * BPREP2_NT + tidx
                    row, ch = idx // 16, idx % 16
                    off = sT(row, ch)
                    kw, vw = txl.alloc_local((4,), 'uint32'), txl.alloc_local((4,), 'uint32')
                    txl.ptx.ld.shared.v4.b32(kw[0], kw[1], kw[2], kw[3], smem.ptr_to([BPREP2_K_S + off]))
                    txl.ptx.ld.shared.v4.b32(vw[0], vw[1], vw[2], vw[3], smem.ptr_to([BPREP2_V_S + off]))
                    b_r = ld_shared_f32(smem, BPREP2_B_S + row * 4)
                    eg = exp2_approx(ld_shared_f32(smem, BPREP2_G_S + row * 4))
                    kb, vb = txl.alloc_local((4,), 'uint32'), txl.alloc_local((4,), 'uint32')
                    bb2 = txl.local_scalar('uint32', init=pack_bf16x2(b_r, b_r))
                    for j in range(4):
                        lo, hi, kbb = txl.local_scalar('float32'), txl.local_scalar('float32'), txl.local_scalar('uint32')
                        txl.ptx['mul.rn.bf16x2'](kbb, kw[j], bb2)
                        unpack_bf16x2(kbb, lo, hi)
                        kb1 = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](kb1, txl.cuda.make_float2(lo, hi), txl.cuda.make_float2(eg, eg))
                        txl.assign(kb[j], pack_bf16x2(txl.cuda.float2_x(kb1), txl.cuda.float2_y(kb1)))
                        txl.ptx['mul.rn.bf16x2'](vb[j], vw[j], bb2)
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([BPREP2_KB_S + off]), kb[0], kb[1], kb[2], kb[3])
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([BPREP2_V_S + off]), vb[0], vb[1], vb[2], vb[3])
                acc = txl.alloc_local((16,), 'float32')
                zero_acc(acc, 16)
                if ptg:
                    tm_ld(acc, 0, 0, 32, fence=False)
                for kk in range(0 if ptg else 8):
                    a = txl.alloc_local((4,), 'uint32')
                    ldmatrix_x4(a, smem.ptr_to([BPREP2_K_S + sw256_offset(r0 + lane % 8 + 8 * (lane // 8 % 2), 2 * kk + lane // 16)]))
                    for p in range(2):
                        np_ = chh * 2 + p
                        b = txl.alloc_local((4,), 'uint32')
                        ldmatrix_x4(b, smem.ptr_to([BPREP2_Q_S + sw256_offset(np_ * 16 + lane % 8 + 8 * (lane // 16), 2 * kk + lane // 8 % 2)]))
                        mma_bf16_m16n8k16(acc, 2 * p * 4, a, b[0], b[1])
                        mma_bf16_m16n8k16(acc, (2 * p + 1) * 4, a, b[2], b[3])
                gi = [txl.local_scalar('float32', init=ld_shared_f32(smem, BPREP2_G_S + (r0 + lane_group + 8 * h_) * 4)) for h_ in range(2)]
                gj = [txl.local_scalar('float32', init=ld_shared_f32(smem, BPREP2_G_S + (chh * 32 + nt_ * 8 + 2 * lane_in_group + e_) * 4)) for nt_ in range(4) for e_ in range(2)]
                for nt in range(4):
                    for half in range(2):
                        i = r0 + lane_group + 8 * half
                        j = chh * 32 + nt * 8 + 2 * lane_in_group
                        base = nt * 4 + half * 2
                        scaled, gated = txl.local_scalar('uint64'), txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](scaled, txl.cuda.make_float2(acc[base], acc[base + 1]), txl.cuda.make_float2(scale, scale))
                        txl.ptx['mul.rn.f32x2'](gated, scaled, txl.cuda.make_float2(exp2_approx(gj[nt * 2] - gi[half]), exp2_approx(gj[nt * 2 + 1] - gi[half])))
                        vals = [txl.if_then_else(all_of(i <= j, i < num_valid, j < num_valid), txl.cuda.float2_x(gated), txl.float32(0.0)), txl.if_then_else(all_of(i <= j + 1, i < num_valid, j + 1 < num_valid), txl.cuda.float2_y(gated), txl.float32(0.0))]
                        txl.ptx.st.shared.b32(smem.ptr_to([BPREP2_KQ_S + sw128_offset(i, chh * 4 + nt) + 2 * lane_in_group * 2]), pack_bf16x2(vals[0], vals[1]))
                if ptg:
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.cuda.cta_sync()
                if ptg:
                    ks_ = lambda kp: kp * 2
                    mn_ = lambda kp: kp * 128
                    issue(0, BPREP2_AB_S, UMMA_DESC_K, ks_, BPREP2_V_S, UMMA_DESC_MN, mn_, 4, BPREP2_ID_64x128_KMN, 1)
                    issue(128, BPREP2_AB_S, UMMA_DESC_K, ks_, BPREP2_KB_S, UMMA_DESC_MN, mn_, 4, BPREP2_ID_64x128_KMN, 2)
                    o_u, o_w, o_d = txl.alloc_local((32,), 'float32'), txl.alloc_local((32,), 'float32'), txl.alloc_local((32,), 'float32')
                    tm_ld(o_u, 0, 1, 64, fence=False)
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                    txl.cuda.cta_sync()
                    issue(0, BPREP2_KQ_S, UMMA_DESC_K, ks_, BPREP2_DO_S, UMMA_DESC_MN, mn_, 4, BPREP2_ID_64x128_KMN, 3)
                    tm_ld(o_w, 128, 2, 64, fence=False)
                    tm_ld(o_d, 0, 3, 64)
                    outs = [o_u, o_w, o_d]
                else:
                    outs = []
                    for asrc, src in ((BPREP2_AB_S, BPREP2_V_S), (BPREP2_AB_S, BPREP2_KB_S), (BPREP2_KQ_S, BPREP2_DO_S)):
                        af = txl.alloc_local((16,), 'uint32')
                        for ks in range(4):
                            tmp = txl.alloc_local((4,), 'uint32')
                            ldmatrix_x4(tmp, smem.ptr_to([asrc + sw128_offset(r0 + lane % 8 + 8 * (lane // 8 % 2), 2 * ks + lane // 16)]))
                            for j in range(4):
                                txl.assign(af[ks * 4 + j], tmp[j])
                        o = txl.alloc_local((32,), 'float32')
                        zero_acc(o, 32)
                        for ks in range(4):
                            a4 = [af[ks * 4 + j] for j in range(4)]
                            for p in range(4):
                                np_ = chh * 4 + p
                                b = txl.alloc_local((4,), 'uint32')
                                ldmatrix_x4(b, smem.ptr_to([src + sw256_offset(ks * 16 + lane % 8 + 8 * (lane // 8 % 2), 2 * np_ + lane // 16)]), trans=True)
                                mma_bf16_m16n8k16(o, 2 * p * 4, a4, b[0], b[1])
                                mma_bf16_m16n8k16(o, (2 * p + 1) * 4, a4, b[2], b[3])
                        outs.append(o)
                txl.cuda.cta_sync()
                for o, stg in zip(outs, (BPREP2_K_S, BPREP2_Q_S, BPREP2_DO_S)):
                    for nt in range(8):
                        for half in range(2):
                            row = r0 + lane_group + 8 * half
                            txl.ptx.st.shared.b32(smem.ptr_to([stg + sw256_offset(row, chh * 8 + nt) + 2 * lane_in_group * 2]), pack_bf16x2(o[nt * 4 + half * 2], o[nt * 4 + half * 2 + 1]))
                txl.cuda.cta_sync()
                for it in range(16 * 64 // BPREP2_NT):
                    idx = it * BPREP2_NT + tidx
                    row, ch = idx // 16, idx % 16
                    with txl.if_(row < num_valid), txl.then_():
                        for stg, dst in ((BPREP2_K_S, u_out), (BPREP2_Q_S, w_out), (BPREP2_DO_S, dvl_out)):
                            wd = txl.alloc_local((4,), 'uint32')
                            txl.ptx.ld.shared.v4.b32(wd[0], wd[1], wd[2], wd[3], smem.ptr_to([stg + sw256_offset(row, ch)]))
                            txl.ptx.st.global_.v4.b32(dst.ptr_to([((chunk_start + row) * HV + head) * 128 + ch * 8]), wd[0], wd[1], wd[2], wd[3])
        ntiles = txl.local_scalar('int32', init=(n_chunks * HV - cta + num_ctas - 1) // num_ctas)
        with txl.serial(ntiles) as it:
            txl.assign(tile_parity, it % 2)
            tl = txl.local_scalar('int32', init=cta + it * num_ctas)
            tile(tl // HV, tl % HV)
            txl.ptx['tcgen05.fence::before_thread_sync']()
            txl.cuda.cta_sync()
            txl.ptx['tcgen05.fence::after_thread_sync']()
        if ptg:
            txl.ptx['tcgen05.fence::before_thread_sync']()
            txl.cuda.cta_sync()
            txl.ptx['tcgen05.fence::after_thread_sync']()
            with txl.if_(warp == 0), txl.then_():
                txl.ptx['tcgen05.dealloc.cta_group::1.sync.aligned.b32'](tmem_base, txl.uint32(256))
    return bprep2


BGRAD2_NW = 8
BGRAD2_NT = BGRAD2_NW * 32
BGRAD2_Q_S = 0
BGRAD2_K_S = 16 * KIB
BGRAD2_DO_S = 32 * KIB
BGRAD2_VN_S = 48 * KIB
BGRAD2_DV2_S = 64 * KIB
BGRAD2_V_S = 80 * KIB
BGRAD2_HT_S = 96 * KIB
BGRAD2_DHT_S = 128 * KIB
BGRAD2_DS_S = 160 * KIB
BGRAD2_A_S = 168 * KIB
BGRAD2_G_S = 176 * KIB
BGRAD2_B_S = BGRAD2_G_S + 256
BGRAD2_RED_S = BGRAD2_B_S + 256
BGRAD2_BARS_S = BGRAD2_RED_S + 4096
BGRAD2_SMEM = BGRAD2_BARS_S + 128
BGRAD2_DW_S = BGRAD2_Q_S
BGRAD2_DKA_S = BGRAD2_DO_S
BGRAD2_DA0_S = BGRAD2_VN_S
BGRAD2_DA1_S = BGRAD2_VN_S + 8 * KIB
BGRAD2_KBG_S = BGRAD2_HT_S
BGRAD2_KBB_S = BGRAD2_HT_S + 16 * KIB
BGRAD2_VB_S = BGRAD2_DHT_S
BGRAD2_DA3_S = BGRAD2_DHT_S + 16 * KIB
BGRAD2_TMEM_COLS = 256
BGRAD2_ID_64x128_KK = 69207184
BGRAD2_ID_64x64_KK = 68158608
BGRAD2_TM_DS = 0
BGRAD2_TM_DA = 448
BGRAD2_TM_DW = 0
BGRAD2_ID_64x128_KMN = 69272720
BGRAD2_ID_64x64_MNMN = 68256912
BGRAD2_ID_64x128_MNMN = 69305488
BGRAD2_TM_ACC0 = 256
BGRAD2_TM_ACC1 = 384


def build_bgrad2(HV, kv=False, static_chunks=None, tgemm=False, persistent=False, static_ctas=None):
    """
    The gradient kernel: dq, dk, dv, dg and dbeta per chunk.
    tgemm: dq_h = do @ h^T, dk_h = vn @ dh^T and dw = dv2 @ h^T run as tcgen05 UMMAs (fp32 TMEM
    accumulators, the same K order as the mma.sync chains) issued by one thread as soon as their
    tiles land; do / vn / dv2 / h / dh then live in the UMMA K-major SW128 layout, which the
    remaining ldmatrix readers address directly. persistent: num_ctas CTAs loop over the (chunk,
    head) tiles; TMEM / mbarrier setup is paid once per CTA and no SM sits empty between tiles.
    """
    kvl = kv
    B3G_S = BGRAD2_SMEM
    B3B_S = BGRAD2_SMEM + 256
    assert kvl or not tgemm
    TCOLS = 512 if tgemm else BGRAD2_TMEM_COLS
    sA = umma64_offset if tgemm else sw256_offset
    sH = umma128_offset if tgemm else sw256_offset
    # kvl: h / dh arrive as [chunk, HV, K, V] (FLA's orientation): conflict-free tree loads.
    def prepare_host(params):
        maps = []
        for name in ('q', 'k', 'do', 'v'):
            desc = txl.stack_alloca('tensormap', 1)
            txl.cu_tensor_map_encode_tiled(desc, 'bfloat16', 3, params[name].data, 64, params['S'], 2 * HV, HV * 256, 128, 64, 64, 2, 1, 1, 1, 0, 3, 2, 0)
            maps.append(desc)
        return tuple(maps)

    @txl.kernel()
    def bgrad2(q: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), k: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), v: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), do: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), vn: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dv2: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), h: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dh: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), g_cum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), beta: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), a_in: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), nseg: txl.int32, n_chunks: txl.int32, num_ctas: txl.int32, scale: txl.float32, dq_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dk_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dv_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dg_out: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), db_out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), S: txl.int32):
        host = prepare_host(dict(q=q, k=k, do=do, v=v, S=S))
        if persistent:
            grid = [num_ctas] if static_ctas is None else [static_ctas]
        else:
            grid = [n_chunks, HV] if static_chunks is None else [static_chunks, HV]
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid, block=BGRAD2_NW * 32))
        q_map, k_map, do_map, v_map = host
        smem = txl.alloc_tensor((BGRAD2_SMEM + 512 + 64,), txl.u8, scope='shared.dyn', align=1024)
        bnext = txl.local_scalar('uint16', init=txl.uint16(0))
        cta = txl.cta_id()
        tile_parity = txl.local_scalar('int32', init=txl.int32(0))
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        lane_group, lane_in_group = lane // 4, lane % 4
        rs, chh = warp % 4, warp // 4
        r0 = rs * 16
        if tgemm:
            with txl.if_(tidx == 0), txl.then_():
                for i_ in range(14):
                    txl.ptx.mbarrier.init.shared.b64(smem.ptr_to([BGRAD2_BARS_S + i_ * 8]), txl.uint32(1))
                txl.ptx.mbarrier.init.shared.b64(smem.ptr_to([BGRAD2_BARS_S + 14 * 8]), txl.uint32(BGRAD2_NW))
                txl.ptx.mbarrier.init.shared.b64(smem.ptr_to([BGRAD2_BARS_S + 15 * 8]), txl.uint32(1))
                txl.ptx.mbarrier.init.shared.b64(smem.ptr_to([BGRAD2_SMEM + 512]), txl.uint32(1))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.if_(warp == 0), txl.then_():
            txl.ptx['tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32'](smem.ptr_to([BGRAD2_RED_S]), txl.uint32(TCOLS))
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        tmem_base = txl.local_scalar('uint32')
        txl.ptx.ld.volatile.shared.u32(tmem_base, smem.ptr_to([BGRAD2_RED_S]))

        def issue_64x128(d_col, a_off, b_off, bar_i, n=128, more=()):
            """
            D[64 x n] = A[64 x 128] @ B[n x 128]^T (+ the (A, B) pairs in more, accumulated in
            order), all K-major SW128 (A: 64-row atoms, B: n-row atoms); K steps in increasing order
            like the mma.sync chains.
            """
            with txl.if_(warp == 0), txl.then_():
                leader = txl.cuda.elect_sync()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx['fence.proxy.async.shared::cta']()
                for pi, (ao, bo) in enumerate(((a_off, b_off),) + tuple(more)):
                    a_d, b_d = umma_desc(smem, ao, UMMA_DESC_K), umma_desc(smem, bo, UMMA_DESC_K)
                    for kp in range(8):
                        txl.ptx[UMMA_F16](txl.cast(tmem_base + d_col, 'uint32'), a_d + txl.uint64(kp % 4 * 2 + kp // 4 * 512), b_d + txl.uint64(kp % 4 * 2 + kp // 4 * (8 * n)), txl.uint32(BGRAD2_ID_64x128_KK if n == 128 else BGRAD2_ID_64x64_KK), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 and pi == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](smem.ptr_to([BGRAD2_BARS_S + bar_i * 8]), pred=leader)

        def issue_at(d_col, b_off, bar_i):
            """
            D[64 x 128] = A^T @ B with A the [64 i][64 j] bf16 sw128 tile (A^T: M = j MN-major) and
            B a [64 i][128 n] umma64 tile (MN-major); K = i in 4 steps of 16 rows.
            """
            with txl.if_(warp == 0), txl.then_():
                leader = txl.cuda.elect_sync()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx['fence.proxy.async.shared::cta']()
                a_addr, b_addr = txl.cuda.cvta_generic_to_shared(smem.ptr_to([BGRAD2_A_S])), txl.cuda.cvta_generic_to_shared(smem.ptr_to([b_off]))
                a_d = txl.bitwise_or(txl.uint64(UMMA_DESC_MN), txl.cast(txl.bitwise_and(txl.shift_right(a_addr, txl.uint32(4)), txl.uint32(16383)), 'uint64'))
                b_d = txl.bitwise_or(txl.uint64(UMMA_DESC_MN), txl.cast(txl.bitwise_and(txl.shift_right(b_addr, txl.uint32(4)), txl.uint32(16383)), 'uint64'))
                for kp in range(4):
                    txl.ptx[UMMA_F16](txl.cast(tmem_base + d_col, 'uint32'), a_d + txl.uint64(kp * 128), b_d + txl.uint64(kp * 128), txl.uint32(BGRAD2_ID_64x128_MNMN), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](smem.ptr_to([BGRAD2_BARS_S + bar_i * 8]), pred=leader)

        def frag_addr(d_col, ncols=64):
            return txl.cast(tmem_base + txl.shift_left(txl.cast(rs * 32, 'uint32'), txl.uint32(16)) + d_col + chh * ncols, 'uint32')

        def tm_st(src, d_col):
            """
            Write a 64-column fragment back to its accumulator position (16x256b, the tm_ld layout).
            """
            txl.ptx[TMEM_ST_16X256B_X8](frag_addr(d_col), *[txl.reinterpret(src[i], ty='uint32') for i in range(32)])
            txl.ptx['tcgen05.wait::st.sync.aligned']()

        def issue_acc(d_col, a_off, a_base, b_off, idesc, bar_i, fresh=False):
            """
            D[64 x 128] += A @ B over K = 64 (4 steps of 16): A the [64][64] sw128 DS tile (K-major:
            ds, MN-major: ds^T), B a [64 tok][128] umma64 tile read MN-major, accumulated onto the
            written-back D accumulator.
            """
            with txl.if_(warp == 0), txl.then_():
                leader = txl.cuda.elect_sync()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx['fence.proxy.async.shared::cta']()
                a_addr, b_addr = txl.cuda.cvta_generic_to_shared(smem.ptr_to([a_off])), txl.cuda.cvta_generic_to_shared(smem.ptr_to([b_off]))
                a_d = txl.bitwise_or(txl.uint64(a_base), txl.cast(txl.bitwise_and(txl.shift_right(a_addr, txl.uint32(4)), txl.uint32(16383)), 'uint64'))
                b_d = txl.bitwise_or(txl.uint64(UMMA_DESC_MN), txl.cast(txl.bitwise_and(txl.shift_right(b_addr, txl.uint32(4)), txl.uint32(16383)), 'uint64'))
                for kp in range(4):
                    a_step = kp * 2 if a_base == UMMA_DESC_K else kp * 128
                    txl.ptx[UMMA_F16](txl.cast(tmem_base + d_col, 'uint32'), a_d + txl.uint64(a_step), b_d + txl.uint64(kp * 128), txl.uint32(idesc), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if fresh and kp == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](smem.ptr_to([BGRAD2_BARS_S + bar_i * 8]), pred=leader)

        def issue_6464(d_col, a_off, a_base, b_off, b_base, idesc, bar_i):
            """
            D[64 x 64] = A @ B over K = 64 with sw128 [64][64] operand tiles (K-major: +32 B per
            k-step, MN-major: +16 rows per k-step).
            """
            with txl.if_(warp == 0), txl.then_():
                leader = txl.cuda.elect_sync()
                txl.ptx['tcgen05.fence::after_thread_sync']()
                txl.ptx['fence.proxy.async.shared::cta']()
                ds_ = []
                for off, base in ((a_off, a_base), (b_off, b_base)):
                    addr = txl.cuda.cvta_generic_to_shared(smem.ptr_to([off]))
                    ds_.append(txl.bitwise_or(txl.uint64(base), txl.cast(txl.bitwise_and(txl.shift_right(addr, txl.uint32(4)), txl.uint32(16383)), 'uint64')))
                for kp in range(4):
                    a_step = kp * 2 if a_base == UMMA_DESC_K else kp * 128
                    b_step = kp * 2 if b_base == UMMA_DESC_K else kp * 128
                    txl.ptx[UMMA_F16](txl.cast(tmem_base + d_col, 'uint32'), ds_[0] + txl.uint64(a_step), ds_[1] + txl.uint64(b_step), txl.uint32(idesc), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.ptx.pred(txl.uint32(0 if kp == 0 else 1)), pred=leader)
                txl.ptx[UMMA_COMMIT](smem.ptr_to([BGRAD2_BARS_S + bar_i * 8]), pred=leader)

        def tm_ld(dst, d_col, bar_i, ncols=64, fence=True):
            """
            Rows r0.. (TMEM lanes 32 rs ..), columns chh * 64 .. of a finished 64 x 128 accumulator,
            in the mma.sync C-fragment order (16x256b: thread (lane_group, lane_in_group) gets rows
            lane_group / lane_group + 8, cols 2 lane_in_group, 2 lane_in_group + 1).
            """
            ready = txl.local_scalar('uint32', init=txl.uint32(0))
            with txl.while_(ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready, smem.ptr_to([BGRAD2_BARS_S + bar_i * 8]), txl.cast(tile_parity, 'uint32'))
            if fence:
                txl.ptx['fence.proxy.async.shared::cta']()
            txl.ptx['tcgen05.fence::after_thread_sync']()
            nr = ncols // 2
            raw = txl.alloc_local((nr,), 'uint32')
            txl.ptx[TMEM_LD_16X256B_X8 if ncols == 64 else TMEM_LD_16X256B_X4](*[raw[i] for i in range(nr)], txl.cast(tmem_base + txl.shift_left(txl.cast(rs * 32, 'uint32'), txl.uint32(16)) + d_col + chh * ncols, 'uint32'))
            txl.ptx['tcgen05.wait::ld.sync.aligned']()
            for i in range(nr):
                txl.assign(dst[i], txl.reinterpret(raw[i], ty='float32'))

        def a_rows(base, nks, sw, kc0=0):
            """
            A fragments of rows r0.. of a row-major tile, k-steps kc0 .. kc0+nks (16 cols each).
            """
            af = txl.alloc_local((4 * nks,), 'uint32')
            for ks in range(nks):
                tmp = txl.alloc_local((4,), 'uint32')
                ldmatrix_x4(tmp, smem.ptr_to([base + sw(r0 + lane % 8 + 8 * (lane // 8 % 2), 2 * (kc0 + ks) + lane // 16)]))
                for j in range(4):
                    txl.assign(af[ks * 4 + j], tmp[j])
            return af

        def a_trans(base, nks, sw):
            """
            A fragments of A^T (m = tile columns r0.., k = tile rows) of a row-major tile.
            """
            af = txl.alloc_local((4 * nks,), 'uint32')
            mi = lane // 8
            for ks in range(nks):
                tmp = txl.alloc_local((4,), 'uint32')
                ldmatrix_x4(tmp, smem.ptr_to([base + sw(ks * 16 + lane % 8 + 8 * (mi // 2), r0 // 8 + mi % 2)]), trans=True)
                for j in range(4):
                    txl.assign(af[ks * 4 + j], tmp[j])
            return af

        def mm_nt(acc, af, nks, bbase, np0, n_pairs, sw):
            """
            acc += A @ B, B[k][n] = tile[n][k] (rows n, starting at pair np0).
            """
            for ks in range(nks):
                a4 = [af[ks * 4 + j] for j in range(4)]
                for p in range(n_pairs):
                    np_ = np0 + p
                    b = txl.alloc_local((4,), 'uint32')
                    ldmatrix_x4(b, smem.ptr_to([bbase + sw(np_ * 16 + lane % 8 + 8 * (lane // 16), 2 * ks + lane // 8 % 2)]))
                    mma_bf16_m16n8k16(acc, 2 * p * 4, a4, b[0], b[1])
                    mma_bf16_m16n8k16(acc, (2 * p + 1) * 4, a4, b[2], b[3])

        def mm_t(acc, af, nks, bbase, np0, n_pairs, sw):
            """
            acc += A @ B, B[k][n] = tile[k][n] (rows k, column pairs from np0).
            """
            for ks in range(nks):
                a4 = [af[ks * 4 + j] for j in range(4)]
                for p in range(n_pairs):
                    np_ = np0 + p
                    b = txl.alloc_local((4,), 'uint32')
                    ldmatrix_x4(b, smem.ptr_to([bbase + sw(ks * 16 + lane % 8 + 8 * (lane // 8 % 2), 2 * np_ + lane // 16)]), trans=True)
                    mma_bf16_m16n8k16(acc, 2 * p * 4, a4, b[0], b[1])
                    mma_bf16_m16n8k16(acc, (2 * p + 1) * 4, a4, b[2], b[3])

        def c_pack(base, sw, kc0, nks):
            """
            bf16 pairs of rows r0.. in accumulator layout for k-steps kc0.. (16 cols each).
            """
            return a_rows(base, nks, sw, kc0)

        def c_val(pk, nt, e):
            word = pk[nt // 2 * 4 + nt % 2 * 2 + e // 2]
            bits = txl.bitwise_and(word, txl.uint32(65535)) if e % 2 == 0 else txl.shift_right(word, txl.uint32(16))
            r = txl.local_scalar('float32')
            txl.ptx.cvt.f32.bf16(r, txl.cast(bits, 'uint16'))
            return r

        def st_c(base, acc, n_tiles, col0, sw):
            for nt in range(n_tiles):
                for half in range(2):
                    row = r0 + lane_group + 8 * half
                    col = col0 + nt * 8
                    txl.ptx.st.shared.b32(smem.ptr_to([base + sw(row, col // 8) + 2 * lane_in_group * 2]), pack_bf16x2(acc[nt * 4 + half * 2], acc[nt * 4 + half * 2 + 1]))

        def quad_sum(x):
            out = txl.local_scalar('float32', init=x)
            for off in (1, 2):
                sh = txl.local_scalar('uint32')
                txl.ptx.shfl_sync.bfly.b32(sh, txl.reinterpret(out, ty='uint32'), txl.uint32(off), txl.uint32(31), txl.uint32(4294967295))
                txl.assign(out, out + txl.reinterpret(sh, ty='float32'))
            return out

        def warp_sum(x):
            out = txl.local_scalar('float32', init=x)
            for off in (16, 8, 4, 2, 1):
                sh = txl.local_scalar('uint32')
                txl.ptx.shfl_sync.bfly.b32(sh, txl.reinterpret(out, ty='uint32'), txl.uint32(off), txl.uint32(31), txl.uint32(4294967295))
                txl.assign(out, out + txl.reinterpret(sh, ty='float32'))
            return out

        def row_partial(slot, vals):
            """
            Quad-reduce two per-thread row partials and park them in RED_S[slot][half][row].
            """
            for hh in range(2):
                s_ = quad_sum(vals[hh])
                with txl.if_(lane_in_group == 0), txl.then_():
                    st_shared_f32(smem, BGRAD2_RED_S + ((slot * 2 + chh) * 64 + r0 + lane_group + 8 * hh) * 4, s_)

        def row_total(slot, row):
            return ld_shared_f32(smem, BGRAD2_RED_S + ((slot * 2 + 0) * 64 + row) * 4) + ld_shared_f32(smem, BGRAD2_RED_S + ((slot * 2 + 1) * 64 + row) * 4)

        p_t0 = txl.local_scalar('int32', init=txl.int32(0))
        p_nv = txl.local_scalar('int32', init=txl.int32(0))
        q_t0 = txl.local_scalar('int32', init=txl.int32(0))
        q_nv = txl.local_scalar('int32', init=txl.int32(0))

        def issue_loads(c_in, head_in, gS_, need_early):
            c = txl.local_scalar('int32', init=c_in)
            head = txl.local_scalar('int32', init=head_in)
            chunk_start = txl.local_scalar('int32', init=q_t0)
            num_valid = txl.local_scalar('int32', init=q_nv)
            gidx = txl.cast(c, 'int64')
            # Private recurrence scratch is produced in the exact physical umma64 / umma128 layouts
            # consumed here: six contiguous 16 KiB copies replace the h, dh, vn and dv2 per-lane
            # issue trains.
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(smem.ptr_to([BGRAD2_BARS_S + 15 * 8]), txl.uint32(98304 - 16384 + 2 * 16384))
                with txl.if_(need_early), txl.then_():
                    txl.ptx['fence.proxy.async.shared::cta']()
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(smem.ptr_to([BGRAD2_SMEM + 512]), txl.uint32(3 * 16384))
                    for tmap, dst in ((q_map, BGRAD2_Q_S), (v_map, BGRAD2_V_S)):
                        txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([dst]), txl.address_of(tmap), txl.int32(0), chunk_start, txl.cast(2 * head, 'int32'), smem.ptr_to([BGRAD2_SMEM + 512]), txl.uint64(0x12F0000000000000))
                    txl.ptx[BULK_COPY_G2S](smem.ptr_to([BGRAD2_DV2_S]), dv2.ptr_to([txl.cast(c * HV + head, 'int64') * 8192]), txl.uint32(16384), smem.ptr_to([BGRAD2_SMEM + 512]))
                for tmap, dst in ((do_map, BGRAD2_DO_S), (k_map, BGRAD2_K_S)):
                    txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([dst]), txl.address_of(tmap), txl.int32(0), chunk_start, txl.cast(2 * head, 'int32'), smem.ptr_to([BGRAD2_BARS_S + 15 * 8]), txl.uint64(0x12F0000000000000))
                for src, dst in ((h, BGRAD2_HT_S), (dh, BGRAD2_DHT_S)):
                    for half in range(2):
                        txl.ptx[BULK_COPY_G2S](smem.ptr_to([dst + half * 16384]), src.ptr_to([(gidx * HV + head) * 16384 + half * 8192]), txl.uint32(16384), smem.ptr_to([BGRAD2_BARS_S + 15 * 8]))
                for src, dst in ((vn, BGRAD2_VN_S),):
                    txl.ptx[BULK_COPY_G2S](smem.ptr_to([dst]), src.ptr_to([(gidx * HV + head) * 8192]), txl.uint32(16384), smem.ptr_to([BGRAD2_BARS_S + 15 * 8]))
            for it in range(8 * 64 // BGRAD2_NT):
                idx = it * BGRAD2_NT + tidx
                row, ch = idx // 8, idx % 8
                ok = row < num_valid
                srow = txl.if_then_else(ok, chunk_start + row, chunk_start)
                txl.ptx[CP_ASYNC_16](smem.ptr_to([BGRAD2_A_S + sw128_offset(row, ch)]), a_in.ptr_to([(srow * HV + head) * 64 + ch * 8]), 16, txl.cast(txl.if_then_else(txl.And(ok, ch <= row // 8), 16, 0), 'uint32'))
            with txl.if_(tidx < 64), txl.then_():
                ok = tidx < num_valid
                srow = txl.if_then_else(ok, chunk_start + tidx, chunk_start)
                txl.ptx[CP_ASYNC_4](smem.ptr_to([gS_ + tidx * 4]), g_cum.ptr_to([srow * HV + head]), 4, txl.cast(txl.if_then_else(ok, 4, 0), 'uint32'))
            txl.ptx.cp.async_.commit_group()
            with txl.if_(tidx < 64), txl.then_():
                txl.assign(bnext, txl.uint16(0))
                with txl.if_(tidx < num_valid), txl.then_():
                    txl.ptx.ld.global_.b16(bnext, beta.ptr_to([(chunk_start + tidx) * HV + head]))

        def tile(c_in, head_in, par, active, nxt_in):
            head, nxt = txl.local_scalar('int32', init=head_in), txl.local_scalar('int32', init=nxt_in)
            gS, bS = txl.if_then_else(tile_parity == 0, BGRAD2_G_S, B3G_S), txl.if_then_else(tile_parity == 0, BGRAD2_B_S, B3B_S)
            chunk_start, num_valid = txl.local_scalar('int32', init=p_t0), txl.local_scalar('int32', init=p_nv)
            with txl.if_(nxt >= 0), txl.then_():
                nx_c = txl.local_scalar('int32', init=nxt // HV)
                nt0_, nnv_ = chunk_lookup(cu, nseg, nx_c)
                txl.assign(q_t0, nt0_)
                txl.assign(q_nv, nnv_)
            # The persistent CTA will bulk-copy these four contiguous private tensors only after
            # finishing the current tile. Pull the next tile toward L2 now, while the current
            # tensor-core DAG executes.
            with txl.if_(txl.And(nxt >= 0, tidx == 0)), txl.then_():
                pf128 = txl.cast(nxt, 'int64') * 16384
                pf64 = txl.cast(nxt, 'int64') * 8192
                txl.ptx['cp.async.bulk.prefetch.L2.global'](h.ptr_to([pf128]), txl.uint32(32768))
                txl.ptx['cp.async.bulk.prefetch.L2.global'](dh.ptr_to([pf128]), txl.uint32(32768))
                txl.ptx['cp.async.bulk.prefetch.L2.global'](vn.ptr_to([pf64]), txl.uint32(16384))
                txl.ptx['cp.async.bulk.prefetch.L2.global'](dv2.ptr_to([pf64]), txl.uint32(16384))
            gl_t = txl.local_scalar('float32', init=txl.float32(0.0))
            with txl.if_(txl.And(active, num_valid > 0)), txl.then_():
                with txl.if_(tidx < 64), txl.then_():
                    bv = txl.local_scalar('float32')
                    txl.ptx.cvt.f32.bf16(bv, bnext)
                    st_shared_f32(smem, bS + tidx * 4, bv)
                txl.ptx.cp.async_.wait_group(0)
                bulk_ready = txl.local_scalar('uint32', init=txl.uint32(0))
                with txl.while_(bulk_ready == txl.uint32(0)):
                    txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(bulk_ready, smem.ptr_to([BGRAD2_BARS_S + 15 * 8]), txl.cast(tile_parity, 'uint32'))
                early_ready = txl.local_scalar('uint32', init=txl.uint32(0))
                with txl.while_(early_ready == txl.uint32(0)):
                    txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(early_ready, smem.ptr_to([BGRAD2_SMEM + 512]), txl.cast(tile_parity, 'uint32'))
                txl.ptx['fence.proxy.async.shared::cta']()
                txl.cuda.cta_sync()
                if tgemm:
                    issue_64x128(BGRAD2_TM_DS, BGRAD2_DO_S, BGRAD2_VN_S, 3, n=64)
                    issue_64x128(BGRAD2_TM_ACC0, BGRAD2_DO_S, BGRAD2_HT_S, 0)
                    issue_64x128(BGRAD2_TM_ACC1, BGRAD2_VN_S, BGRAD2_DHT_S, 1)
                with txl.if_(txl.And(nxt >= 0, tidx == 0)), txl.then_():
                    nx_h = txl.local_scalar('int32', init=nxt % HV)
                    for tmap in (do_map, q_map, k_map, v_map):
                        txl.ptx['cp.async.bulk.prefetch.tensor.3d.L2.global.tile'](txl.address_of(tmap), txl.int32(0), q_t0, txl.cast(2 * nx_h, 'int32'))
                gl = ld_shared_f32(smem, gS + (num_valid - 1) * 4)
                txl.assign(gl_t, gl)
                gi_r = [txl.local_scalar('float32', init=ld_shared_f32(smem, gS + (r0 + lane_group + 8 * h_) * 4)) for h_ in range(2)]
                bi_r = [txl.local_scalar('float32', init=ld_shared_f32(smem, bS + (r0 + lane_group + 8 * h_) * 4)) for h_ in range(2)]
                egi_r = [txl.local_scalar('float32', init=exp2_approx(gi_r[h_])) for h_ in range(2)]
                egli_r = [txl.local_scalar('float32', init=txl.if_then_else(r0 + lane_group + 8 * h_ < num_valid, exp2_approx(gl - gi_r[h_]), txl.float32(0.0))) for h_ in range(2)]
                jq0 = chh * 32
                gj_r = [txl.local_scalar('float32', init=ld_shared_f32(smem, gS + (jq0 + nt_ * 8 + 2 * lane_in_group + e_) * 4)) for nt_ in range(4) for e_ in range(2)]
                with txl.if_(warp < 4), txl.then_():
                    def vec_(base, p_):
                        r_, jj_ = p_ // 4, p_ % 4
                        kk_ = lane // 16 + 2 * warp + 8 * r_
                        vv_ = 8 * (lane % 16) + 2 * jj_
                        lo_, hi_ = txl.local_scalar('uint16'), txl.local_scalar('uint16')
                        txl.ptx.ld.shared.b16(lo_, smem.ptr_to([base + sw256_offset(vv_, kk_ // 8) + kk_ % 8 * 2]))
                        txl.ptx.ld.shared.b16(hi_, smem.ptr_to([base + sw256_offset(vv_ + 1, kk_ // 8) + kk_ % 8 * 2]))
                        return txl.bitwise_or(txl.cast(lo_, 'uint32'), txl.shift_left(txl.cast(hi_, 'uint32'), txl.uint32(16)))
                    nodes = txl.alloc_local((32,), 'uint32')
                    if kvl:
                        for r_ in range(16):
                            kk_ = lane // 16 + 2 * warp + 8 * r_
                            hv, dv_ = txl.alloc_local((4,), 'uint32'), txl.alloc_local((4,), 'uint32')
                            txl.ptx.ld.shared.v4.b32(hv[0], hv[1], hv[2], hv[3], smem.ptr_to([BGRAD2_HT_S + sH(kk_, lane % 16)]))
                            txl.ptx.ld.shared.v4.b32(dv_[0], dv_[1], dv_[2], dv_[3], smem.ptr_to([BGRAD2_DHT_S + sH(kk_, lane % 16)]))
                            for hm in range(2):
                                m_ = 2 * r_ + hm
                                pb_ = txl.local_scalar('uint32')
                                txl.ptx['mul.rn.bf16x2'](pb_, hv[2 * hm + 1], dv_[2 * hm + 1])
                                txl.ptx['fma.rn.bf16x2'](nodes[m_], hv[2 * hm], dv_[2 * hm], pb_)
                    for m_ in range(0 if kvl else 32):
                        pb_ = txl.local_scalar('uint32')
                        txl.ptx['mul.rn.bf16x2'](pb_, vec_(BGRAD2_HT_S, 2 * m_ + 1), vec_(BGRAD2_DHT_S, 2 * m_ + 1))
                        txl.ptx['fma.rn.bf16x2'](nodes[m_], vec_(BGRAD2_HT_S, 2 * m_), vec_(BGRAD2_DHT_S, 2 * m_), pb_)
                    n_ = 32
                    while n_ > 1:
                        for i_ in range(n_ // 2):
                            txl.ptx['add.rn.bf16x2'](nodes[i_], nodes[2 * i_], nodes[2 * i_ + 1])
                        n_ //= 2
                    tsum = txl.local_scalar('uint16')
                    txl.ptx['add.rn.bf16'](tsum, txl.cast(txl.bitwise_and(nodes[0], txl.uint32(65535)), 'uint16'), txl.cast(txl.shift_right(nodes[0], txl.uint32(16)), 'uint16'))
                    bacc = txl.local_scalar('float32')
                    txl.ptx.cvt.f32.bf16(bacc, tsum)
                    for off in (16, 8, 4, 2, 1):
                        sh = txl.local_scalar('uint32')
                        txl.ptx.shfl_sync.bfly.b32(sh, txl.reinterpret(bacc, ty='uint32'), txl.uint32(off), txl.uint32(31), txl.uint32(4294967295))
                        txl.assign(bacc, round_to_bf16(bacc + txl.reinterpret(sh, ty='float32')))
                    with txl.if_(lane == 0), txl.then_():
                        st_shared_f32(smem, BGRAD2_RED_S + (384 + warp) * 4, bacc)
                ds = txl.alloc_local((16,), 'float32')
                zero_acc(ds, 16)
                if tgemm:
                    tm_ld(ds, BGRAD2_TM_DS, 3, ncols=32, fence=False)
                else:
                    do_af = a_rows(BGRAD2_DO_S, 8, sA)
                    mm_nt(ds, do_af, 8, BGRAD2_VN_S, chh * 2, 2, sA)
                for nt in range(4):
                    for e in range(4):
                        i = r0 + lane_group + 8 * (e // 2)
                        j = jq0 + nt * 8 + 2 * lane_in_group + e % 2
                        x = ds[nt * 4 + e] * exp2_approx(gi_r[e // 2] - gj_r[nt * 2 + e % 2])
                        txl.assign(ds[nt * 4 + e], txl.if_then_else(all_of(i >= j, i < num_valid, j < num_valid), x, txl.float32(0.0)) * scale)
                st_c(BGRAD2_DS_S, ds, 4, jq0, sw128_offset)
                assert tgemm
                kc = chh * 64
                dq = txl.alloc_local((32,), 'float32')
                tm_ld(dq, BGRAD2_TM_ACC0, 0, fence=False)
                qpk = c_pack(BGRAD2_Q_S, sA, chh * 4, 4)
                for nt in range(8):
                    for half in range(2):
                        i = nt * 4 + half * 2
                        p0, p1 = txl.local_scalar('uint64'), txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](p0, txl.cuda.make_float2(dq[i], dq[i + 1]), txl.cuda.make_float2(egi_r[half], egi_r[half]))
                        txl.ptx['mul.rn.f32x2'](p1, p0, txl.cuda.make_float2(scale, scale))
                        txl.assign(dq[i], txl.cuda.float2_x(p1))
                        txl.assign(dq[i + 1], txl.cuda.float2_y(p1))
                tm_st(dq, BGRAD2_TM_ACC0)
                dk = txl.alloc_local((32,), 'float32')
                tm_ld(dk, BGRAD2_TM_ACC1, 1, fence=False)
                kpk = c_pack(BGRAD2_K_S, sA, chh * 4, 4)
                p2 = txl.local_scalar('float32', init=txl.float32(0.0))
                for nt in range(8):
                    for half in range(2):
                        i = nt * 4 + half * 2
                        pscaled = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](pscaled, txl.cuda.make_float2(dk[i], dk[i + 1]), txl.cuda.make_float2(egli_r[half], egli_r[half]))
                        txl.assign(dk[i], txl.cuda.float2_x(pscaled))
                        txl.assign(dk[i + 1], txl.cuda.float2_y(pscaled))
                    for e in range(4):
                        txl.assign(p2, p2 + dk[nt * 4 + e] * c_val(kpk, nt, e))
                wsum2 = warp_sum(p2)
                with txl.if_(lane == 0), txl.then_():
                    st_shared_f32(smem, BGRAD2_RED_S + (392 + warp) * 4, wsum2)
                tm_st(dk, BGRAD2_TM_ACC1)
                txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.cuda.cta_sync()
                issue_acc(BGRAD2_TM_ACC0, BGRAD2_DS_S, UMMA_DESC_K, BGRAD2_K_S, BGRAD2_ID_64x128_KMN, 8)
                issue_acc(BGRAD2_TM_ACC1, BGRAD2_DS_S, UMMA_DESC_MN, BGRAD2_Q_S, BGRAD2_ID_64x128_MNMN, 9)
                issue_64x128(BGRAD2_TM_DW, BGRAD2_DV2_S, BGRAD2_HT_S, 2)
                tm_ld(dq, BGRAD2_TM_ACC0, 8, fence=False)
                for nt in range(8):
                    for half in range(2):
                        row = r0 + lane_group + 8 * half
                        with txl.if_(row < num_valid), txl.then_():
                            txl.ptx.st.global_.b32(dq_out.ptr_to([((chunk_start + row) * HV + head) * 128 + kc + nt * 8 + 2 * lane_in_group]), pack_bf16x2(dq[nt * 4 + half * 2], dq[nt * 4 + half * 2 + 1]))
                tm_ld(dk, BGRAD2_TM_ACC1, 9, fence=False)
                dgp = txl.alloc_local((2,), 'float32')
                zero_acc(dgp, 2)
                for strip in range(2):
                    dqs = txl.alloc_local((16,), 'float32')
                    if tgemm:
                        rq = txl.alloc_local((16,), 'uint32')
                        txl.ptx[TMEM_LD_16X256B_X4](*[rq[i] for i in range(16)], frag_addr(BGRAD2_TM_ACC0 + strip * 32))
                        txl.ptx['tcgen05.wait::ld.sync.aligned']()
                        for i in range(16):
                            txl.assign(dqs[i], txl.reinterpret(rq[i], ty='float32'))
                    else:
                        txl.ptx[TMEM_LD_32X32B_X16](*(dqs[i] for i in range(16)), txl.cast(tmem_base + warp * 32 + strip * 16, 'uint32'))
                        txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    for j in range(16):
                        nt = strip * 4 + j // 4
                        e = j % 4
                        txl.assign(dgp[e // 2], dgp[e // 2] + dqs[j] * c_val(qpk, nt, e) - dk[nt * 4 + e] * c_val(kpk, nt, e))
                row_partial(0, dgp)
                txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.ptx['bar.warp.sync'](txl.uint32(4294967295))
                with txl.if_(lane == 0), txl.then_():
                    txl.ptx.mbarrier.arrive.shared.b64(smem.ptr_to([BGRAD2_BARS_S + 14 * 8]))
                with txl.if_(warp == 0), txl.then_():
                    retired = txl.local_scalar('uint32', init=txl.uint32(0))
                    with txl.while_(retired == txl.uint32(0)):
                        txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(retired, smem.ptr_to([BGRAD2_BARS_S + 14 * 8]), txl.cast(tile_parity, 'uint32'))
                if tgemm:
                    issue_64x128(BGRAD2_TM_ACC1, BGRAD2_K_S, BGRAD2_K_S, 4, n=64)
                st_c(BGRAD2_DKA_S, dk, 8, kc, sw256_offset)
                dwa = txl.alloc_local((32,), 'float32')
                zero_acc(dwa, 32)
                if tgemm:
                    tm_ld(dwa, BGRAD2_TM_DW, 2)
                else:
                    dv2_af_early = a_rows(BGRAD2_DV2_S, 8, sw256_offset)
                    (mm_nt if kvl else mm_t)(dwa, dv2_af_early, 8, BGRAD2_HT_S, chh * 4, 4, sw256_offset)
                for i in range(32):
                    txl.assign(dwa[i], -round_to_bf16(dwa[i]))
                st_c(BGRAD2_DW_S, dwa, 8, kc, sA)
                for it in range(16 * 64 // BGRAD2_NT):
                    idx = it * BGRAD2_NT + tidx
                    row, ch = idx // 16, idx % 16
                    kw, vw = txl.alloc_local((4,), 'uint32'), txl.alloc_local((4,), 'uint32')
                    txl.ptx.ld.shared.v4.b32(kw[0], kw[1], kw[2], kw[3], smem.ptr_to([BGRAD2_K_S + sA(row, ch)]))
                    txl.ptx.ld.shared.v4.b32(vw[0], vw[1], vw[2], vw[3], smem.ptr_to([BGRAD2_V_S + sA(row, ch)]))
                    b_r = ld_shared_f32(smem, bS + row * 4)
                    bg = b_r * exp2_approx(ld_shared_f32(smem, gS + row * 4))
                    o1, o2, o3 = txl.alloc_local((4,), 'uint32'), txl.alloc_local((4,), 'uint32'), txl.alloc_local((4,), 'uint32')
                    bb2 = txl.local_scalar('uint32', init=pack_bf16x2(b_r, b_r))
                    for j in range(4):
                        lo, hi = txl.local_scalar('float32'), txl.local_scalar('float32')
                        txl.ptx.cvt.f32.bf16(lo, txl.cast(txl.bitwise_and(kw[j], txl.uint32(65535)), 'uint16'))
                        txl.ptx.cvt.f32.bf16(hi, txl.cast(txl.shift_right(kw[j], txl.uint32(16)), 'uint16'))
                        pbg = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](pbg, txl.cuda.make_float2(lo, hi), txl.cuda.make_float2(bg, bg))
                        txl.assign(o1[j], pack_bf16x2(txl.cuda.float2_x(pbg), txl.cuda.float2_y(pbg)))
                        txl.ptx['mul.rn.bf16x2'](o2[j], kw[j], bb2)
                        txl.ptx['mul.rn.bf16x2'](o3[j], vw[j], bb2)
                    off = sA(row, ch)
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([BGRAD2_KBG_S + off]), o1[0], o1[1], o1[2], o1[3])
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([BGRAD2_KBB_S + off]), o2[0], o2[1], o2[2], o2[3])
                    txl.ptx.st.shared.v4.b32(smem.ptr_to([BGRAD2_VB_S + off]), o3[0], o3[1], o3[2], o3[3])
                txl.cuda.cta_sync()
                if tgemm:
                    issue_64x128(BGRAD2_TM_DA, BGRAD2_DW_S, BGRAD2_KBG_S, 5, n=64, more=((BGRAD2_DV2_S, BGRAD2_VB_S),))
                    issue_at(BGRAD2_TM_ACC0, BGRAD2_DW_S, 6)
                    issue_at(0, BGRAD2_DV2_S, 7)
                dA = txl.alloc_local((16,), 'float32')
                zero_acc(dA, 16)
                if tgemm:
                    tm_ld(dA, BGRAD2_TM_DA, 5, ncols=32, fence=False)
                else:
                    dw_af = a_rows(BGRAD2_DW_S, 8, sw256_offset)
                    mm_nt(dA, dw_af, 8, BGRAD2_KBG_S, chh * 2, 2, sw256_offset)
                    dv2_af = a_rows(BGRAD2_DV2_S, 8, sA)
                    mm_nt(dA, dv2_af, 8, BGRAD2_VB_S, chh * 2, 2, sw256_offset)
                at_af = None if tgemm else a_trans(BGRAD2_A_S, 4, sw128_offset)
                dkbg = txl.alloc_local((32,), 'float32')
                zero_acc(dkbg, 32)
                if tgemm:
                    tm_ld(dkbg, BGRAD2_TM_ACC0, 6, fence=False)
                else:
                    mm_t(dkbg, at_af, 4, BGRAD2_DW_S, chh * 4, 4, sA)
                dkw = txl.alloc_local((32,), 'float32')
                kpk_wy = c_pack(BGRAD2_K_S, sA, chh * 4, 4)
                dbp, g2p = txl.alloc_local((2,), 'float32'), txl.alloc_local((2,), 'float32')
                zero_acc(dbp, 2)
                zero_acc(g2p, 2)
                for nt in range(8):
                    for half in range(2):
                        i = nt * 4 + half * 2
                        gb = egi_r[half] * bi_r[half]
                        pdkw = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](pdkw, txl.cuda.make_float2(dkbg[i], dkbg[i + 1]), txl.cuda.make_float2(gb, gb))
                        txl.assign(dkw[i], round_to_bf16(txl.cuda.float2_x(pdkw)))
                        txl.assign(dkw[i + 1], round_to_bf16(txl.cuda.float2_y(pdkw)))
                        for ee in range(2):
                            e = half * 2 + ee
                            x = dkbg[nt * 4 + e]
                            kv = c_val(kpk_wy, nt, e)
                            gexp = egi_r[half]
                            b_r = bi_r[half]
                            txl.assign(dbp[half], dbp[half] + x * kv * gexp)
                            txl.assign(g2p[half], g2p[half] + x * (kv * (b_r * gexp)))
                tm_st(dkw, BGRAD2_TM_ACC0)
                dvb = txl.alloc_local((32,), 'float32')
                zero_acc(dvb, 32)
                if tgemm:
                    tm_ld(dvb, 0, 7, fence=False)
                else:
                    mm_t(dvb, at_af, 4, BGRAD2_DV2_S, chh * 4, 4, sA)
                vpk = c_pack(BGRAD2_V_S, sA, chh * 4, 4)
                for nt in range(8):
                    for half in range(2):
                        i = nt * 4 + half * 2
                        for ee in range(2):
                            e = half * 2 + ee
                            txl.assign(dbp[half], dbp[half] + dvb[nt * 4 + e] * c_val(vpk, nt, e))
                        pdvb = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](pdvb, txl.cuda.make_float2(dvb[i], dvb[i + 1]), txl.cuda.make_float2(bi_r[half], bi_r[half]))
                        txl.assign(dvb[i], txl.cuda.float2_x(pdvb))
                        txl.assign(dvb[i + 1], txl.cuda.float2_y(pdvb))
                for nt in range(8):
                    for half in range(2):
                        row = r0 + lane_group + 8 * half
                        with txl.if_(row < num_valid), txl.then_():
                            txl.ptx.st.global_.b32(dv_out.ptr_to([((chunk_start + row) * HV + head) * 128 + kc + nt * 8 + 2 * lane_in_group]), pack_bf16x2(dvb[nt * 4 + half * 2], dvb[nt * 4 + half * 2 + 1]))
                for nt in range(4):
                    for e in range(4):
                        i = r0 + lane_group + 8 * (e // 2)
                        j = jq0 + nt * 8 + 2 * lane_in_group + e % 2
                        txl.assign(dA[nt * 4 + e], txl.if_then_else(all_of(i > j, i < num_valid, j < num_valid), dA[nt * 4 + e], txl.float32(0.0)))
                st_c(BGRAD2_DA0_S, dA, 4, jq0, sw128_offset)
                akk = txl.alloc_local((16,), 'float32')
                zero_acc(akk, 16)
                if tgemm:
                    tm_ld(akk, BGRAD2_TM_ACC1, 4, ncols=32, fence=False)
                else:
                    k_af = a_rows(BGRAD2_K_S, 8, sw256_offset)
                    mm_nt(akk, k_af, 8, BGRAD2_K_S, chh * 2, 2, sw256_offset)
                if tgemm:
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.cuda.cta_sync()
                d1 = txl.alloc_local((16,), 'float32')
                zero_acc(d1, 16)
                if tgemm:
                    issue_6464(128, BGRAD2_DA0_S, UMMA_DESC_K, BGRAD2_A_S, UMMA_DESC_K, BGRAD2_ID_64x64_KK, 12)
                    with txl.if_(txl.And(nxt >= 0, tidx == 0)), txl.then_():
                        ex_h = txl.local_scalar('int32', init=nxt % HV)
                        txl.ptx['fence.proxy.async.shared::cta']()
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(smem.ptr_to([BGRAD2_SMEM + 512]), txl.uint32(3 * 16384))
                        for tmap, dst in ((q_map, BGRAD2_Q_S), (v_map, BGRAD2_V_S)):
                            txl.ptx['cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint'](smem.ptr_to([dst]), txl.address_of(tmap), txl.int32(0), q_t0, txl.cast(2 * ex_h, 'int32'), smem.ptr_to([BGRAD2_SMEM + 512]), txl.uint64(0x12F0000000000000))
                        txl.ptx[BULK_COPY_G2S](smem.ptr_to([BGRAD2_DV2_S]), dv2.ptr_to([txl.cast(nxt, 'int64') * 8192]), txl.uint32(16384), smem.ptr_to([BGRAD2_SMEM + 512]))
                    tm_ld(d1, 128, 12, ncols=32, fence=False)
                else:
                    da_af = a_rows(BGRAD2_DA0_S, 4, sw128_offset)
                    mm_nt(d1, da_af, 4, BGRAD2_A_S, chh * 2, 2, sw128_offset)
                st_c(BGRAD2_DA1_S, d1, 4, jq0, sw128_offset)
                if tgemm:
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.cuda.cta_sync()
                d2 = txl.alloc_local((16,), 'float32')
                zero_acc(d2, 16)
                if tgemm:
                    issue_6464(192, BGRAD2_A_S, UMMA_DESC_MN, BGRAD2_DA1_S, UMMA_DESC_MN, BGRAD2_ID_64x64_MNMN, 13)
                    tm_ld(d2, 192, 13, ncols=32, fence=False)
                else:
                    at2_af = a_trans(BGRAD2_A_S, 4, sw128_offset)
                    mm_t(d2, at2_af, 4, BGRAD2_DA1_S, chh * 2, 2, sw128_offset)
                colp = txl.alloc_local((8,), 'float32')
                zero_acc(colp, 8)
                for nt in range(4):
                    for e in range(4):
                        i = r0 + lane_group + 8 * (e // 2)
                        j = jq0 + nt * 8 + 2 * lane_in_group + e % 2
                        x = d2[nt * 4 + e] * exp2_approx(gi_r[e // 2] - gj_r[nt * 2 + e % 2])
                        y = round_to_bf16(txl.if_then_else(all_of(i > j, i < num_valid, j < num_valid), -x, txl.float32(0.0)))
                        txl.assign(d2[nt * 4 + e], y)
                        ada = y * (akk[nt * 4 + e] * bi_r[e // 2])
                        txl.assign(g2p[e // 2], g2p[e // 2] + ada)
                        txl.assign(colp[nt * 2 + e % 2], colp[nt * 2 + e % 2] + ada)
                st_c(BGRAD2_DA3_S, d2, 4, jq0, sw128_offset)
                for i in range(8):
                    for off in (4, 8, 16):
                        sh = txl.local_scalar('uint32')
                        txl.ptx.shfl_sync.bfly.b32(sh, txl.reinterpret(colp[i], ty='uint32'), txl.uint32(off), txl.uint32(31), txl.uint32(4294967295))
                        txl.assign(colp[i], colp[i] + txl.reinterpret(sh, ty='float32'))
                with txl.if_(lane_group == 0), txl.then_():
                    for nt in range(4):
                        for e2 in range(2):
                            st_shared_f32(smem, BGRAD2_RED_S + (512 + rs * 64 + jq0 + nt * 8 + 2 * lane_in_group + e2) * 4, colp[nt * 2 + e2])
                if tgemm:
                    txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.cuda.cta_sync()
                dkb = txl.alloc_local((32,), 'float32')
                zero_acc(dkb, 32)
                t3 = txl.alloc_local((32,), 'float32')
                zero_acc(t3, 32)
                if tgemm:
                    issue_acc(0, BGRAD2_DA3_S, UMMA_DESC_K, BGRAD2_K_S, BGRAD2_ID_64x128_KMN, 10, fresh=True)
                    issue_acc(BGRAD2_TM_ACC1, BGRAD2_DA3_S, UMMA_DESC_MN, BGRAD2_KBB_S, BGRAD2_ID_64x128_MNMN, 11, fresh=True)
                    tm_ld(dkb, 0, 10, fence=False)
                    tm_ld(t3, BGRAD2_TM_ACC1, 11)
                else:
                    d3_af = a_rows(BGRAD2_DA3_S, 4, sw128_offset)
                    mm_t(dkb, d3_af, 4, BGRAD2_K_S, chh * 4, 4, sA)
                    d3t = a_trans(BGRAD2_DA3_S, 4, sw128_offset)
                    mm_t(t3, d3t, 4, BGRAD2_KBB_S, chh * 4, 4, sA)
                dkap = c_pack(BGRAD2_DKA_S, sw256_offset, chh * 4, 4)
                for strip in range(2):
                    dkws, rk = txl.alloc_local((16,), 'float32'), txl.alloc_local((16,), 'uint32')
                    txl.ptx[TMEM_LD_16X256B_X4](*[rk[i] for i in range(16)], frag_addr(BGRAD2_TM_ACC0 + strip * 32))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    for i in range(16):
                        txl.assign(dkws[i], txl.reinterpret(rk[i], ty='float32'))
                    txl.ptx['tcgen05.wait::ld.sync.aligned']()
                    for jp in range(8):
                        j = jp * 2
                        ii = strip * 16 + j
                        nt = ii // 4
                        half = ii % 4 // 2
                        for ee in range(2):
                            e = half * 2 + ee
                            txl.assign(dbp[half], dbp[half] + dkb[ii + ee] * c_val(kpk_wy, nt, e))
                        pmul = txl.local_scalar('uint64')
                        padd1 = txl.local_scalar('uint64')
                        padd2 = txl.local_scalar('uint64')
                        pout = txl.local_scalar('uint64')
                        txl.ptx['mul.rn.f32x2'](pmul, txl.cuda.make_float2(dkb[ii], dkb[ii + 1]), txl.cuda.make_float2(bi_r[half], bi_r[half]))
                        txl.ptx['add.rn.f32x2'](padd1, pmul, txl.cuda.make_float2(t3[ii], t3[ii + 1]))
                        txl.ptx['add.rn.f32x2'](padd2, padd1, txl.cuda.make_float2(dkws[j], dkws[j + 1]))
                        dk20, dk21 = round_to_bf16(txl.cuda.float2_x(padd2)), round_to_bf16(txl.cuda.float2_y(padd2))
                        txl.ptx['add.rn.f32x2'](pout, txl.cuda.make_float2(c_val(dkap, nt, half * 2), c_val(dkap, nt, half * 2 + 1)), txl.cuda.make_float2(dk20, dk21))
                        txl.assign(dkb[ii], txl.cuda.float2_x(pout))
                        txl.assign(dkb[ii + 1], txl.cuda.float2_y(pout))
                for nt in range(8):
                    for half in range(2):
                        row = r0 + lane_group + 8 * half
                        with txl.if_(row < num_valid), txl.then_():
                            txl.ptx.st.global_.b32(dk_out.ptr_to([((chunk_start + row) * HV + head) * 128 + kc + nt * 8 + 2 * lane_in_group]), pack_bf16x2(dkb[nt * 4 + half * 2], dkb[nt * 4 + half * 2 + 1]))
                row_partial(1, dbp)
                row_partial(2, g2p)
                txl.cuda.cta_sync()
            with txl.if_(nxt >= 0), txl.then_():
                # Head-fast work IDs make neighboring persistent CTAs consume the same chunk's
                # contiguous head slabs. This preserves per-tile arithmetic while improving locality
                # for every [token, head, channel] input/output.
                issue_loads(nxt // HV, nxt % HV, txl.if_then_else(tile_parity == 0, B3G_S, BGRAD2_G_S), txl.Not(txl.And(active, num_valid > 0)))
            with txl.if_(txl.And(active, num_valid > 0)), txl.then_():
                with txl.if_(tidx < 64), txl.then_():
                    i = tidx
                    db_ = row_total(1, i)
                    with txl.if_(i < num_valid), txl.then_():
                        txl.ptx.st.global_.b16(db_out.ptr_to([(chunk_start + i) * HV + head]), txl.reinterpret(txl.cast(db_, 'bfloat16'), ty='uint16'))
                    hdh = round_to_bf16(round_to_bf16(ld_shared_f32(smem, BGRAD2_RED_S + 384 * 4) + ld_shared_f32(smem, BGRAD2_RED_S + 386 * 4)) + round_to_bf16(ld_shared_f32(smem, BGRAD2_RED_S + 385 * 4) + ld_shared_f32(smem, BGRAD2_RED_S + 387 * 4)))
                    dkk = txl.local_scalar('float32', init=txl.float32(0.0))
                    for w_ in range(BGRAD2_NW):
                        txl.assign(dkk, dkk + ld_shared_f32(smem, BGRAD2_RED_S + (392 + w_) * 4))
                    dgl = hdh * exp2_approx(gl_t) + dkk
                    colsum = ld_shared_f32(smem, BGRAD2_RED_S + (512 + i) * 4) + ld_shared_f32(smem, BGRAD2_RED_S + (576 + i) * 4) + ld_shared_f32(smem, BGRAD2_RED_S + (640 + i) * 4) + ld_shared_f32(smem, BGRAD2_RED_S + (704 + i) * 4)
                    dga = row_total(0, i)
                    dga2 = txl.if_then_else(i == num_valid - 1, dga + dgl, dga)
                    st_shared_f32(smem, gS + i * 4, txl.if_then_else(i < num_valid, dga2 + (row_total(2, i) - colsum), txl.float32(0.0)))
                txl.cuda.cta_sync()
                with txl.if_(warp == 0), txl.then_():
                    x0 = txl.local_scalar('float32', init=ld_shared_f32(smem, gS + lane * 4))
                    x1 = txl.local_scalar('float32', init=ld_shared_f32(smem, gS + 128 + lane * 4))
                    c0, c1 = txl.local_scalar('float32', init=x0), txl.local_scalar('float32', init=x1)
                    for off in (1, 2, 4, 8, 16):
                        for x in (c0, c1):
                            up = txl.local_scalar('uint32')
                            txl.ptx.shfl_sync.up.b32(up, txl.reinterpret(x, ty='uint32'), txl.uint32(off), txl.uint32(0), txl.uint32(4294967295))
                            txl.assign(x, txl.if_then_else(lane >= off, txl.reinterpret(up, ty='float32') + x, x))
                    sh = txl.local_scalar('uint32')
                    txl.ptx.shfl_sync.idx.b32(sh, txl.reinterpret(c0, ty='uint32'), txl.uint32(31), txl.uint32(31), txl.uint32(4294967295))
                    txl.assign(c1, c1 + txl.reinterpret(sh, ty='float32'))
                    z = warp_sum(x0 + x1)
                    with txl.if_(lane < num_valid), txl.then_():
                        txl.ptx.st.global_.f32(dg_out.ptr_to([(chunk_start + lane) * HV + head]), -c0 + z + x0)
                    with txl.if_(lane + 32 < num_valid), txl.then_():
                        txl.ptx.st.global_.f32(dg_out.ptr_to([(chunk_start + lane + 32) * HV + head]), -c1 + z + x1)
            txl.assign(p_t0, q_t0)
            txl.assign(p_nv, q_nv)

        if persistent:
            ntiles = txl.local_scalar('int32', init=(n_chunks * HV - cta + num_ctas - 1) // num_ctas)
            with txl.serial(ntiles + 1) as it:
                txl.assign(tile_parity, (it + 1) % 2)
                tl = cta + (it - 1) * num_ctas
                tile(txl.if_then_else(it > 0, tl // HV, 0), txl.if_then_else(it > 0, tl % HV, 0), (it + 1) % 2, it > 0, txl.if_then_else(it < ntiles, cta + it * num_ctas, -1))
                txl.ptx['tcgen05.fence::before_thread_sync']()
                txl.cuda.cta_sync()
                txl.ptx['tcgen05.fence::after_thread_sync']()
        else:
            issue_loads(cta[0], cta[1], BGRAD2_G_S)
            tile(cta[0], cta[1], 0, True, -1)
        txl.ptx['tcgen05.fence::before_thread_sync']()
        txl.cuda.cta_sync()
        txl.ptx['tcgen05.fence::after_thread_sync']()
        with txl.if_(warp == 0), txl.then_():
            txl.ptx['tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned']()
            txl.ptx['tcgen05.dealloc.cta_group::1.sync.aligned.b32'](tmem_base, txl.uint32(TCOLS))
    return bgrad2


RH_PMAX = 4
RH_PMAX_SHORT = 2  # below RH_LONG chunks, two 64-column sub-chains beat four 128-column ones
RH_LONG = 96
PACK_MIN = 48  # bsum7 chain length (chunks) from which rec_h is packed behind it
PACK_P = 3
EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def compile_kernel(kernel):
    """
    Compile a traced kernel once and return its entry function (Executable.__call__ re-resolves it
    on every call); the executable stays alive in EXECUTABLES. TARGET runs nvcc without fast
    math (no --use_fast_math).
    """
    ex = kernel.compile(TARGET)
    EXECUTABLES.append(ex)
    return ex["main"]


class LazyKernels:
    """
    Trace and compile each kernel on first use.
    """

    def __init__(self, builders):
        self.builders = builders
        self.done = {}

    def __getitem__(self, name):
        if name not in self.done:
            self.done[name] = compile_kernel(self.builders[name]())
        return self.done[name]


@functools.cache
def fwd_kernels(HV):
    return LazyKernels({"intra2": lambda: build_intra2(HV, "raw", kperm=True, diet=True, fsw=True, tg=True, persistent=True), "fsum7": lambda: build_fsum7(HV, rawneg=True, inplace=True), "fsum7o": lambda: build_fsum7(HV, rawneg=True, inplace=True, with_o=True), "rec7m": lambda: build_rec7(HV, with_o=True, merge=True, v16=True, ckw=True),})


@functools.cache
def bwd_kernels(HV):
    return LazyKernels({"bprep2": lambda: build_bprep2(HV, ptg=True, persistent=True), "rec_h": lambda: build_rec7(HV, with_o=False, store_h=True, hst=True, v16=True), "rec_h64": lambda: build_rec7(HV, with_o=False, store_h=True, hst=True, v16=True, vb=64, nz=2), "rec_h128": lambda: build_rec7(HV, with_o=False, store_h=True, hst=True, v16=True, vb=128, nz=4), "rec_h64x4_pdl": lambda: build_rec7(HV, with_o=False, store_h=True, hst=True, v16=True, vb=64, nz=4, pdl=True), "bsum7": lambda: build_bsum7(HV, rawneg=True), "bsum7s": lambda: build_bsum7(HV, store=True, rawneg=True, lst=True), "brec7m": lambda: build_brec7(HV, merge=True, bst=True), "bgrad2": lambda: build_bgrad2(HV, kv=True, tgemm=True, persistent=True), "zero": lambda: build_zero_payload(HV),})


@functools.cache
def num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def split_plan(cu, nch, ffold, pack=False):
    """
    Sub-chain plan for the backward rec_h: (nzr, (ck1, ck2, ck3), (kt1, kt2, kt3), ckm) or None.
    Split chunks are segment starts (zero state) or chunks the forward rec7m covers (chunks [0,
    cov)) whose entering state it checkpoints into A's holes at storage token kt (bit z of ckm).
    Minimizes the longest sub-chain.
    """
    if nch < 4:
        return None
    starts, slots = set(), []
    c = 0
    for a, b in zip(cu[:-1], cu[1:]):
        starts.add(c)
        slots += [a + 64 * REC7_CK_CHUNKS * i for i in range((b - a) // 64 // REC7_CK_CHUNKS)]
        c += (b - a + 63) // 64
    nlast = (cu[-1] - cu[-2] + 63) // 64
    cov = nch - nlast if ffold else nch

    def greedy(L):
        pos, sp, nck = 0, [], 0
        while nch - pos > L:
            if len(sp) == pmax - 1:
                return None
            hi = min(pos + L, nch - 1)
            st = [x for x in range(hi, pos, -1) if x in starts]
            ckx = [x for x in range(hi, pos, -1) if x < cov or (ffold and x > cov)] if nck < len(slots) else []
            if st and (not ckx or st[0] >= ckx[0] or nck + 1 > len(slots)):
                sp.append((st[0], 0))
            elif ckx:
                sp.append((ckx[0], 1))
                nck += 1
            else:
                return None
            pos = sp[-1][0]
        return sp

    pmax = PACK_P if pack else RH_PMAX if nch >= RH_LONG else RH_PMAX_SHORT
    for L in range(max(1, -(-nch // pmax)), nch):
        sp = greedy(L)
        if sp is not None:
            break
    else:
        return None
    if not sp:
        return None
    cks, kts, ckm, si = [], [], 0, 0
    for z, (x, mode) in enumerate(sp, start=1):
        cks.append(x)
        if mode:
            kts.append(slots[si])
            si += 1
            ckm |= 1 << z
        else:
            kts.append(0)
    while len(cks) < 3:
        cks.append(nch)
        kts.append(0)
    return (len(sp) + 1, tuple(cks), tuple(kts), ckm)


def fsum7_split_args(sp, c0):
    """
    The rec_h split plan relative to the folded segment's first chunk c0 (fsum7o writes checkpoints
    at or after it).
    """
    if sp is None:
        return (-1, -1, -1, 0, 0, 0, 0)
    return tuple(c - c0 for c in sp[1]) + sp[2] + (sp[3],)


def rec_h_split_args(sp):
    if sp is None:
        return (1, 0, 0, 0, 0, 0, 0, 0)
    return (sp[0],) + sp[1] + sp[2] + (sp[3],)


def aligned(t):
    """
    t contiguous with a 16 B aligned base (the TMA and 16 B vector loads need it): t itself when it
    is, else a copy.
    """
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def flat(t):
    return t.reshape(-1)


def gated_delta_rule_cp_plan_sm100(cu_seqlens, cu_seqlens_cpu, rank, is_first_rank, is_last_rank, pre_num_ranks, post_num_ranks):
    """
    The host-side plan of one rank's CP calls (local sequence table, chunk counts, the backward's
    rec_h split and packing), computed once per CP layout.
    """
    cu = [int(x) for x in cu_seqlens_cpu.tolist()]
    nch = sum((b - a + 63) // 64 for a, b in zip(cu[:-1], cu[1:]))
    nlast, nfirst = (cu[-1] - cu[-2] + 63) // 64, (cu[1] - cu[0] + 63) // 64
    pack = (not is_first_rank) and nfirst >= PACK_MIN
    split = split_plan(cu, nch, (not is_last_rank) and len(cu) > 2, pack)
    return dict(cu_gpu=cu_seqlens, cu=cu, nseg=len(cu) - 1, nch=nch, nlast=nlast, nfirst=nfirst, pack=pack, split=split, rank=rank, is_first_rank=bool(is_first_rank), is_last_rank=bool(is_last_rank), pre_num_ranks=pre_num_ranks, post_num_ranks=post_num_ranks)


def chunk_gdr_cp_fwd_pre_sm100(q, k, v, g, beta, plan):
    """
    The forward up to the all-gather: (hm, saved).
    """
    q, k, v, g, beta = (aligned(t) for t in (q, k, v, g, beta))
    B, S, HV, D = q.shape
    kernels, device = fwd_kernels(HV), q.device
    cu = plan["cu_gpu"]
    g_cum = torch.empty((B, S, HV), dtype=torch.float32, device=device)
    A, aqk = (torch.empty((B, S, HV, 64), dtype=torch.bfloat16, device=device) for _ in range(2))
    w, u, o = torch.empty_like(k), torch.empty_like(v), torch.empty_like(v)
    h0, hm = torch.empty((B, HV, D, D), dtype=torch.float32, device=device), torch.empty((HV, D, 2 * D), dtype=torch.float32, device=device)
    scale = float(D ** (-0.5))
    last = plan["is_last_rank"]
    ffold = (not last) and plan["nseg"] > 1
    kernels["intra2"](flat(q), flat(k), flat(v), flat(g), flat(beta), cu, plan["nseg"], plan["nch"], flat(g_cum), flat(A), flat(w), flat(u), flat(aqk), flat(hm), int(last), 3 * num_sms(device.index), S)
    if not last:
        if ffold:
            kernels["fsum7o"](flat(k), flat(w), flat(u), flat(g_cum), plan["cu"][-2], plan["cu"][-1], flat(hm), flat(q), flat(aqk), flat(o), scale, flat(A), *fsum7_split_args(plan["split"], plan["nch"] - plan["nlast"]), S)
        else:
            kernels["fsum7"](flat(k), flat(w), flat(u), flat(g_cum), plan["cu"][-2], plan["cu"][-1], flat(hm), flat(q), flat(aqk), flat(o), scale, flat(A), -1, -1, -1, 0, 0, 0, 0, S)
    return hm, (q, k, g_cum, A, aqk, w, u, o, h0, hm)


def chunk_gdr_cp_fwd_post_sm100(gathered, saved, plan):
    """
    The forward after the all-gather: (o, g_cum, A, initial_state).
    """
    q, k, g_cum, A, aqk, w, u, o, h0, hm = saved
    HV, D = q.shape[2:]
    kernels = fwd_kernels(HV)
    scale = float(D ** (-0.5))
    ffold = (not plan["is_last_rank"]) and plan["nseg"] > 1
    pre = 0 if plan["is_first_rank"] else plan["pre_num_ranks"]
    kernels["rec7m"](flat(q), flat(k), flat(w), flat(u), flat(aqk), flat(g_cum), plan["cu_gpu"], plan["nseg"], 0, plan["nch"] - (plan["nlast"] if ffold else 0), flat(h0), 0, scale, flat(o), flat(A), flat(o), flat(hm), flat(aligned(gathered)), plan["rank"] - pre, 1, pre, *rec_h_split_args(plan["split"]))
    return o, g_cum, A, h0


def chunk_gdr_cp_bwd_pre_sm100(q, k, v, g_cum, beta, A, do, initial_state, plan):
    """
    The backward up to the all-gather: (dhm, saved).
    """
    q, k, v, g_cum, beta, A, do, initial_state = (aligned(t) for t in (q, k, v, g_cum, beta, A, do, initial_state))
    _, S, HV, D = q.shape
    kernels, device = bwd_kernels(HV), q.device
    cu, nch, first = plan["cu_gpu"], plan["nch"], plan["is_first_rank"]
    scale = float(D ** (-0.5))
    w, u, dvl = torch.empty_like(k), torch.empty_like(v), torch.empty_like(v)
    vn, dv2 = (torch.empty((nch, HV, 64, D), dtype=torch.bfloat16, device=device) for _ in range(2))
    h, dh = (torch.empty((nch, HV, D, D), dtype=torch.bfloat16, device=device) for _ in range(2))
    dhm = torch.empty((HV, D, 2 * D), dtype=torch.float32, device=device)
    kernels["bprep2"](flat(q), flat(k), flat(v), flat(do), flat(g_cum), flat(beta), flat(A), cu, plan["nseg"], nch, scale, flat(w), flat(u), flat(dvl), 2 * num_sms(device.index))
    split = plan["split"]
    pack = plan["pack"] and split is not None
    rec_h_name = "rec_h64x4_pdl" if pack else "rec_h" if split is None else "rec_h64" if split[0] == 2 else "rec_h128"

    def run_rec_h():
        kernels[rec_h_name](flat(q), flat(k), flat(w), flat(u), flat(A), flat(g_cum), cu, plan["nseg"], 0, nch, flat(initial_state), int(not first), scale, flat(dv2), flat(h), flat(vn), flat(dhm), flat(dhm), 0, 0, 0, *rec_h_split_args(split))

    if not pack:
        run_rec_h()
    fold = (not first) and (plan["is_last_rank"] or plan["nseg"] > 1)
    if not first:
        kernels["bsum7s" if fold else "bsum7"](flat(q), flat(k), flat(w), flat(do), flat(dvl), flat(g_cum), plan["cu"][0], plan["cu"][1], scale, flat(dhm), flat(dh), flat(dv2), S)
    if pack:
        run_rec_h()
    if first:
        kernels["zero"](flat(dhm))
    return dhm, (q, k, v, g_cum, beta, A, do, w, dvl, vn, dv2, h, dh, dhm)


def chunk_gdr_cp_bwd_post_sm100(gathered, saved, plan):
    """
    The backward after the all-gather: (dq, dk, dv, dg, dbeta).
    """
    q, k, v, g_cum, beta, A, do, w, dvl, vn, dv2, h, dh, dhm = saved
    S, HV, D = q.shape[1:]
    kernels, device = bwd_kernels(HV), q.device
    cu, nch, first, last = plan["cu_gpu"], plan["nch"], plan["is_first_rank"], plan["is_last_rank"]
    scale = float(D ** (-0.5))
    dq, dk, dv, dg, db = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(g_cum), torch.empty_like(beta)
    post = 0 if last else plan["post_num_ranks"]
    fold = (not first) and (last or plan["nseg"] > 1)
    ag = aligned(gathered)
    if not fold or plan["nfirst"] < nch:
        c_lo = plan["nfirst"] if fold else 0
        kernels["brec7m"](flat(q), flat(k), flat(w), flat(do), flat(dvl), flat(g_cum), cu, plan["nseg"], c_lo, nch, scale, flat(dh), flat(dv2), flat(dhm), flat(ag), plan["rank"] + post, -1, post, S)
    kernels["bgrad2"](flat(q), flat(k), flat(v), flat(do), flat(vn), flat(dv2), flat(h), flat(dh), flat(g_cum), flat(beta), flat(A), cu, plan["nseg"], nch, min(num_sms(device.index), nch * HV), scale, flat(dq), flat(dk), flat(dv), flat(dg), flat(db), S)
    return dq, dk, dv, dg, db
