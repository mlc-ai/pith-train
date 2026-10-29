"""
pith-train SM100 FP8 expert GEMMs as fused TIRx-lite tcgen05 block-scaled kernels.

m-grouped (fwd / dgrad): out[rows of g] = a_g @ b[g].T, bf16.
k-grouped (wgrad):       d[g] += a_g^T @ b_g, fp32, accumulated in place by TMA reduce-add.

Both kernels are persistent, warp specialized (TMA / MMA / scale-factor / epilogue warps) and
read pith-train's fp32 E8M0 scales directly: a scale warp converts them to packed UE8M0 bytes in
shared memory, the MMA warp copies them to tensor memory with tcgen05.cp and issues
tcgen05.mma.kind::mxf8f6f4.block_scale (one scale per 128-wide K block, selected by sf_id).

Evolved with TIRx-harness: run fp8gemm, frontier member pair2cta-mk-disp (commit 74e5452, 1.445x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools

import torch
import tvm
import tvm_ffi

import tirx_kernels.tirx_lite as txl
from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_smem_descriptor_base_uint64,
)

__all__ = ["fp8_grouped_gemm_sm100", "k_grouped_fp8_gemm_sm100"]

NUM_SMS = 148
BM = 128
BK = 128
UMMA_K = 32
EPI_BAR = 8
EVICT_NORMAL = 0x1000000000000000
FP8 = "float8_e4m3fn"


def u64c(value):
    return txl.bitwise_or(
        txl.shift_left(txl.uint64((value >> 32) & 0xFFFFFFFF), txl.uint64(32)),
        txl.uint64(value & 0xFFFFFFFF),
    )


def u322(x):
    return txl.cast(x, "uint32")


def udiv(x, d):
    return txl.cast(u322(x) // u322(d), "int32")


def umod(x, d):
    return txl.cast(u322(x) % u322(d), "int32")


def smem_desc(base_const, addr_u32):
    """
    Matrix/SF descriptor: constant fields | (addr >> 4) in bits [0, 14).
    """
    return txl.bitwise_or(
        u64c(base_const),
        txl.cast(txl.bitwise_and(txl.shift_right(addr_u32, txl.uint32(4)), txl.uint32(0x3FFF)), "uint64"),
    )


def with_sf_id(desc, sf_id):
    out = txl.bitwise_and(desc, txl.uint32(0x9FFFFFCF))
    out = txl.bitwise_or(out, txl.shift_left(u322(sf_id), txl.uint32(29)))
    return txl.bitwise_or(out, txl.shift_left(u322(sf_id), txl.uint32(4)))


def e8(bits_u32):
    """
    Biased exponent byte of an fp32 power of two: its UE8M0 encoding.
    """
    return txl.bitwise_and(txl.shift_right(bits_u32, txl.uint32(23)), txl.uint32(0xFF))


def pack4(vals, base):
    w = e8(vals[base])
    for j in range(1, 4):
        w = txl.bitwise_or(w, txl.shift_left(e8(vals[base + j]), txl.uint32(8 * j)))
    return w


SF_DESC_BASE = encode_smem_descriptor_base_uint64(0, 8, 0)
AB_DESC_BASE = encode_smem_descriptor_base_uint64(0, 64, 3)  # SW128, 8 rows x 128 B per group
MMA_CHAIN = "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
UTCCP_CHAIN = "tcgen05.cp.cta_group::1.32x128b.warpx4"
LOAD2D = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
LOAD2D_MC = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
LOAD3D = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
STORE2D = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
REDUCE3D = "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"


def wait2(bar, stage, phase):
    txl.cuda.mbarrier_wait(bar.ptr_to([stage]), phase)


def rng(name):
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def end(token):
    txl.cuda.iket.range_end(token[0])


def twait(name, bar, stage, phase):
    tok = rng(name)
    wait2(bar, stage, phase)
    end(tok)


# ----------------------------------------------------------------------------------------------
# m-grouped: out (M, N) bf16, A (M, K) K-major, B (G, N, K) K-major
# ----------------------------------------------------------------------------------------------


def build_mgrouped(STAGES, BN):
    """
    2-CTA (cta_group::2) m-grouped kernel: a CTA pair computes 256 rows x 128 cols.

    Both 128-row halves of a pair belong to one group (pairs are formed per group); the odd last
    block of a group is computed by the pair's second CTA on rows it then does not store.
    """
    A_BYTES = BM * BK
    B_HALF = (BN // 2) * BK
    ACC_STAGES = 2 if BN <= 128 else 1
    N_SF_CHUNKS = BN // 128
    SFA_COL = ACC_STAGES * BN
    SFB_COL = SFA_COL + 4
    EPI_WARPS = 4 * N_SF_CHUNKS  # one warpgroup per 128 accumulator columns
    COLS_PER_WG = 128
    INSTR = encode_instr_descriptor_block_scaled_uint32(
        M=2 * BM, N=BN, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2,
    )  # fmt: skip
    MMA2 = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP2 = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LD2 = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
    LD3 = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def prelude(p):
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = p["M"], p["N"], p["K"], p["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, 2, 0)
        txl.cu_tensor_map_encode_tiled(
            b_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 2, 1,
            1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            d_map, "bfloat16", 2, p["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0
        )
        return a_map, b_map, d_map

    @txl.kernel(warps=4 + EPI_WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_SMS, host_prelude=prelude)
    def mgrouped(
        a: txl.gptr[FP8, 2],
        sfa: txl.gptr[txl.f32, 2],
        b: txl.gptr[FP8, 3],
        sfb: txl.gptr[txl.f32, 3],
        d: txl.gptr[txl.bf16, 2],
        offs: txl.gptr[txl.i32, 1],
        M: txl.i32,
        N: txl.i32,
        K: txl.i32,
        G: txl.i32,
        *,
        host,
    ):
        a_map, b_map, d_map = host
        rank = txl.cta_id_in_cluster([2])
        cluster = txl.cta_id() // 2
        warp = txl.warp_id()
        lane = txl.lane_id()
        tid = txl.thread_id()

        sp = txl.specialize()
        r_tma = sp.role("tma", warps=[0])
        r_mma = sp.role("mma", warps=[1])
        r_sf = sp.role("sf", warps=[2])
        r_idle = sp.role("idle", warps=[3])
        r_epi = sp.role("epi", warps=list(range(4, 4 + EPI_WARPS)))

        smem = txl.smem_pool()
        s_a = smem.alloc((STAGES * A_BYTES,), "uint8", align=1024)
        s_b = smem.alloc((STAGES * B_HALF,), "uint8", align=1024)
        s_cd = smem.alloc((N_SF_CHUNKS * BM * 16,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * BN,), "uint32", align=16)
        full = txl.TMABar(smem, STAGES)
        empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        t_full = txl.TCGen05Bar(smem, ACC_STAGES)
        t_empty = txl.MBarrier(smem, ACC_STAGES)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        full.init(1)
        empty.init(1)
        sf_full.init(64)
        t_full.init(1)
        t_empty.init(2 * 32 * EPI_WARPS)
        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.mbarrier_init.release.cluster()
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        full0 = full.remote_view(0)
        sf_full0 = sf_full.remote_view(0)

        n_nb = (N + (BN - 1)) // BN
        nk = (K + (BK - 1)) // BK
        kb_sf = nk

        def tiles(body):
            """
            Pair walk: t -> (pair p, n block); pairs are counted per group (ceil(rows/256)).
            """
            t = txl.local_scalar("int32", init=cluster)
            g = txl.local_scalar("int32", init=-1)
            gbeg = txl.local_scalar("int32", init=0)  # first row of group g
            gend = txl.local_scalar("int32", init=0)  # one past the last row of group g
            pbeg = txl.local_scalar("int32", init=0)  # first pair of group g
            pend = txl.local_scalar("int32", init=0)  # one past the last pair of group g
            # Total pairs: walk all groups once.
            total = txl.local_scalar("int32", init=0)
            prev = txl.local_scalar("int32", init=0)
            cur = txl.local_scalar("int32")
            with txl.serial(G) as gi:
                txl.ptx.ld.global_.s32(cur, offs.ptr_to([gi]))
                txl.assign(total, total + udiv(cur - prev + 255, 256))
                txl.assign(prev, cur)
            with txl.While(t < total * n_nb):
                pp = txl.local_scalar("int32", init=udiv(t, n_nb))
                nb = txl.local_scalar("int32", init=umod(t, n_nb))
                with txl.While(pp >= pend):
                    txl.assign(g, g + 1)
                    txl.assign(gbeg, gend)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([g]))
                    txl.assign(pbeg, pend)
                    txl.assign(pend, pend + udiv(gend - gbeg + 255, 256))
                m0 = txl.local_scalar("int32", init=gbeg + (pp - pbeg) * 256 + rank * BM)
                body(m0, nb, g, m0 < gend)
                txl.assign(t, t + NUM_SMS // 2)

        with r_tma:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                def body(m0, nb, g, valid):
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        bar = txl.cuda.cvta_generic_to_shared(full0.ptr_to([st.stage]))
                        with txl.If(rank == 0), txl.Then():
                            full0.arrive(st.stage, 2 * (A_BYTES + B_HALF))
                        txl.ptx[LD2](
                            s_a.ptr_to([st.stage * A_BYTES]),
                            txl.address_of(a_map),
                            txl.cast(k * BK, "int32"),
                            m0,
                            bar,
                        )
                        txl.ptx[LD3](
                            s_b.ptr_to([st.stage * B_HALF]),
                            txl.address_of(b_map),
                            txl.cast(k * BK, "int32"),
                            nb * BN + rank * (BN // 2),
                            g,
                            bar,
                        )
                        st.advance()

                tiles(body)

        with r_sf:
            st = txl.PipelineState(STAGES, 1)
            ra = txl.alloc_local((16,), "uint32")
            rb = txl.alloc_local((4 * N_SF_CHUNKS,), "uint32")
            fv = txl.local_scalar("float32")
            n_sfb = (N + 127) // 128

            def load_group(m0, nb, g, kb):
                for j in range(4):
                    kk = txl.min(kb + j, kb_sf - 1)
                    for i in range(4):
                        txl.ptx.ld.global_.nc.f32(fv, sfa.ptr_to([txl.min(m0 + lane + 32 * i, M - 1), kk]))
                        txl.assign(ra[i * 4 + j], txl.reinterpret("uint32", fv))
                    for c in range(N_SF_CHUNKS):
                        nsf = txl.min(nb * N_SF_CHUNKS + c, n_sfb - 1)
                        txl.ptx.ld.global_.nc.f32(fv, sfb.ptr_to([g, nsf, kk]))
                        txl.assign(rb[c * 4 + j], txl.reinterpret("uint32", fv))

            def body(m0, nb, g, valid):
                load_group(m0, nb, g, 0)
                with txl.serial(nk) as k:
                    twait("sf-wait-empty", empty, st.stage, st.phase)
                    with txl.If(umod(k, 4) == 0), txl.Then():
                        txl.ptx.st.shared.v4.u32(
                            s_sfa.ptr_to([st.stage * 128 + lane * 4]),
                            pack4(ra, 0), pack4(ra, 4), pack4(ra, 8), pack4(ra, 12),
                        )  # fmt: skip
                        for c in range(N_SF_CHUNKS):
                            wb = txl.local_scalar("uint32", init=pack4(rb, c * 4))
                            txl.ptx.st.shared.v4.u32(
                                s_sfb.ptr_to([st.stage * BN + c * 128 + lane * 4]), wb, wb, wb, wb
                            )
                        with txl.If(k + 4 < nk), txl.Then():
                            load_group(m0, nb, g, k + 4)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    sf_full0.arrive(st.stage)
                    st.advance()

            tiles(body)

        with r_mma:
            with txl.If(rank == 0), txl.Then():
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    st = txl.PipelineState(STAGES, 0)
                    acc = txl.PipelineState(ACC_STAGES, 0)
                    a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                    b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                    sfa_base = txl.local_scalar(
                        "uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0]))
                    )
                    sfb_base = txl.local_scalar(
                        "uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0]))
                    )

                    def body(m0, nb, g, valid):
                        twait("mma-wait-acc", t_empty, acc.stage, acc.phase ^ 1)
                        tile_tok = rng("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(nk) as k:
                            twait("mma-wait-tma", full, st.stage, st.phase)
                            twait("mma-wait-sf", sf_full, st.stage, st.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.If(umod(k, 4) == 0), txl.Then():
                                txl.ptx[UTCCP2](
                                    txl.uint32(SFA_COL),
                                    smem_desc(SF_DESC_BASE, sfa_base + u322(st.stage * 512)),
                                )
                                for c in range(N_SF_CHUNKS):
                                    txl.ptx[UTCCP2](
                                        txl.uint32(SFB_COL + 4 * c),
                                        smem_desc(SF_DESC_BASE, sfb_base + u322(st.stage * (BN * 4) + c * 512)),
                                    )
                            idesc = txl.local_scalar("uint32", init=with_sf_id(txl.uint32(INSTR), umod(k, 4)))
                            for ki in range(BK // UMMA_K):
                                txl.ptx[MMA2](
                                    u322(acc.stage * BN),
                                    smem_desc(AB_DESC_BASE, a_base + u322(st.stage * A_BYTES + ki * UMMA_K)),
                                    smem_desc(AB_DESC_BASE, b_base + u322(st.stage * B_HALF + ki * UMMA_K)),
                                    idesc,
                                    txl.uint32(SFA_COL),
                                    txl.uint32(SFB_COL),
                                    txl.Or(k > 0, ki > 0) if ki == 0 else txl.bool(True),
                                )
                            empty.arrive(st.stage, cta_group=2, cta_mask=3)
                            st.advance()
                        t_full.arrive(acc.stage, cta_group=2, cta_mask=3)
                        end(tile_tok)
                        acc.advance()

                    tiles(body)

        with r_epi:
            ew = warp - 4
            quad = warp % 4
            wg = ew // 4
            wtid = (ew % 4) * 32 + lane  # thread id within this warpgroup
            row = quad * 32 + lane
            acc = txl.PipelineState(ACC_STAGES, 0)
            vals = txl.alloc_local((32,), "uint32")
            packed = txl.alloc_local((COLS_PER_WG // 2,), "uint32")

            def body(m0, nb, g, valid):
                twait("epi-wait-acc", t_full, acc.stage, acc.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                epi_tok = rng("epi-drain")
                # Drain this warpgroup's 128 accumulator columns into packed bf16 registers.
                for c in range(COLS_PER_WG // 32):
                    taddr = txl.bitwise_or(
                        txl.shift_left(u322(quad * 32), txl.uint32(16)),
                        u322(acc.stage * BN + wg * COLS_PER_WG + c * 32),
                    )
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[vals[j] for j in range(32)], taddr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    for p in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(
                            packed[c * 16 + p],
                            txl.reinterpret("float32", vals[2 * p + 1]),
                            txl.reinterpret("float32", vals[2 * p]),
                        )
                txl.ptx.tcgen05.fence__before_thread_sync()
                t_empty.arrive(acc.stage, remote=0)
                end(epi_tok)
                st_tok = rng("epi-store")
                for c in range(COLS_PER_WG // 32):
                    buf = wg * (BM * 16)
                    with txl.If(wtid == 0), txl.Then():
                        txl.ptx.cp.async_.bulk.wait_group.read(0)
                    txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                    for q in range(4):
                        base = c * 16 + q * 4
                        txl.ptx.st.shared.v4.u32(
                            s_cd.ptr_to([buf + row * 16 + txl.bitwise_xor(q, (row // 2) % 4) * 4]),
                            packed[base], packed[base + 1], packed[base + 2], packed[base + 3],
                        )  # fmt: skip
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                    with txl.If(txl.And(wtid == 0, valid)), txl.Then():
                        txl.ptx[STORE2D](
                            txl.address_of(d_map),
                            nb * BN + wg * COLS_PER_WG + c * 32,
                            m0,
                            s_cd.ptr_to([buf]),
                        )
                    with txl.If(wtid == 0), txl.Then():
                        txl.ptx.cp.async_.bulk.commit_group()
                end(st_tok)
                acc.advance()

            tiles(body)
            with txl.If(wtid == 0), txl.Then():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with r_idle:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped


# ----------------------------------------------------------------------------------------------
# k-grouped: d (G, N, K) fp32 += A_g^T B_g; A (rows, N), B (rows, K), both MN-major operands
# ----------------------------------------------------------------------------------------------


def build_kgrouped(STAGES, CDB=1, REV=False, BN=256):
    """
    2-CTA k-grouped kernel: a CTA pair accumulates a 256 (N) x 256 (K) fp32 tile of d[g].
    """
    A_BYTES = BM * BK
    BH = BN // 2  # B columns held by each CTA of the pair
    B_HALF = BH * BK
    B_SW = 3 if BH == 128 else 2
    B_DESC_BASE = encode_smem_descriptor_base_uint64(0, 8 * BH // 16, B_SW)
    B_K_STEP = UMMA_K * BH
    N_SF_CHUNKS = (BN + 127) // 128
    ACC_STAGES = 1 if BN == 256 else 2
    SFA_COL = ACC_STAGES * BN
    SFB_COL = SFA_COL + 4
    COLS = BN // 2  # accumulator columns per epilogue warpgroup
    INSTR = encode_instr_descriptor_block_scaled_uint32(
        M=2 * BM, N=BN, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=True, trans_b=True, cta_group=2,
    )  # fmt: skip
    K_STEP = UMMA_K * 128  # bytes between UMMA K steps of an MN-major SW128 tile
    MMA2 = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP2 = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LD2 = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def prelude(p):
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = p["M"], p["N"], p["K"], p["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, N, M, N, 128, BK, 1, 1, 0, 3, 2, 0)
        txl.cu_tensor_map_encode_tiled(b_map, FP8, 2, p["b"].data, K, M, K, BH, BK, 1, 1, 0, B_SW, 2, 0)
        txl.cu_tensor_map_encode_tiled(
            d_map, "float32", 3, p["d"].data, K, N, G, K * 4, txl.cast(N, "int64") * K * 4,
            32, BM, 1, 1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        return a_map, b_map, d_map

    @txl.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_SMS, host_prelude=prelude)
    def kgrouped(
        a: txl.gptr[FP8, 2],
        sfa: txl.gptr[txl.f32, 2],
        b: txl.gptr[FP8, 2],
        sfb: txl.gptr[txl.f32, 2],
        d: txl.gptr[txl.f32, 3],
        ks: txl.gptr[txl.i32, 1],
        M: txl.i32,
        N: txl.i32,
        K: txl.i32,
        G: txl.i32,
        *,
        host,
    ):
        a_map, b_map, d_map = host
        rank = txl.cta_id_in_cluster([2])
        cluster = txl.cta_id() // 2
        warp = txl.warp_id()
        lane = txl.lane_id()
        tid = txl.thread_id()

        sp = txl.specialize()
        r_tma = sp.role("tma", warps=[0])
        r_mma = sp.role("mma", warps=[1])
        r_sf = sp.role("sf", warps=[2])
        r_idle = sp.role("idle", warps=[3])
        r_epi = sp.role("epi", warps=list(range(4, 12)))

        smem = txl.smem_pool()
        s_a = smem.alloc((STAGES * A_BYTES,), "uint8", align=1024)
        s_b = smem.alloc((STAGES * B_HALF,), "uint8", align=1024)
        s_cd = smem.alloc((2 * CDB * BM * 32,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * BN,), "uint32", align=16)
        full = txl.TMABar(smem, STAGES)
        empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        t_full = txl.TCGen05Bar(smem, ACC_STAGES)
        t_empty = txl.MBarrier(smem, ACC_STAGES)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        full.init(1)
        empty.init(1)
        sf_full.init(64)
        t_full.init(1)
        t_empty.init(512)
        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.mbarrier_init.release.cluster()
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        full0 = full.remote_view(0)
        sf_full0 = sf_full.remote_view(0)

        n_mt = (N + 255) // 256
        n_nt = (K + (BN - 1)) // BN
        per_group = n_mt * n_nt
        total = G * per_group

        def tiles(body):
            """
            Pair walk over (group, 256-row tile of d, 256-col tile of d); empty groups skipped.
            """
            t = txl.local_scalar("int32", init=cluster)
            kg = txl.local_scalar("int32")
            if REV:
                # Walk groups from the last one: d's tail is the most recently written memory.
                g = txl.local_scalar("int32", init=G - 1)
                row0 = txl.local_scalar("int32", init=M)
                txl.ptx.ld.global_.s32(kg, ks.ptr_to([G - 1]))
                txl.assign(row0, row0 - kg)
            else:
                g = txl.local_scalar("int32", init=0)
                row0 = txl.local_scalar("int32", init=0)
                txl.ptx.ld.global_.s32(kg, ks.ptr_to([0]))
            with txl.While(t < total):
                if REV:
                    tg = txl.local_scalar("int32", init=G - 1 - udiv(t, per_group))
                    with txl.While(g > tg):
                        txl.assign(g, g - 1)
                        txl.ptx.ld.global_.s32(kg, ks.ptr_to([g]))
                        txl.assign(row0, row0 - kg)
                    r = txl.local_scalar("int32", init=per_group - 1 - (t - (G - 1 - tg) * per_group))
                else:
                    tg = txl.local_scalar("int32", init=udiv(t, per_group))
                    with txl.While(g < tg):
                        txl.assign(row0, row0 + kg)
                        txl.assign(g, g + 1)
                        txl.ptx.ld.global_.s32(kg, ks.ptr_to([g]))
                    r = txl.local_scalar("int32", init=t - tg * per_group)
                m0 = txl.local_scalar("int32", init=udiv(r, n_nt) * 256 + rank * BM)
                n0 = txl.local_scalar("int32", init=umod(r, n_nt) * BN)
                with txl.If(kg > 0), txl.Then():
                    body(m0, n0, g, row0, udiv(kg, BK))
                txl.assign(t, t + NUM_SMS // 2)

        with r_tma:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                def body(m0, n0, g, row0, nk):
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        bar = txl.cuda.cvta_generic_to_shared(full0.ptr_to([st.stage]))
                        with txl.If(rank == 0), txl.Then():
                            full0.arrive(st.stage, 2 * (A_BYTES + B_HALF))
                        krow = row0 + txl.cast(k * BK, "int32")
                        txl.ptx[LD2](s_a.ptr_to([st.stage * A_BYTES]), txl.address_of(a_map), m0, krow, bar)
                        txl.ptx[LD2](
                            s_b.ptr_to([st.stage * B_HALF]), txl.address_of(b_map), n0 + rank * (BN // 2), krow, bar
                        )
                        st.advance()

                tiles(body)

        with r_sf:
            st = txl.PipelineState(STAGES, 1)
            ra = txl.alloc_local((16,), "uint32")
            rb = txl.alloc_local((16 * N_SF_CHUNKS,), "uint32")
            fv = txl.local_scalar("float32")

            def load_group(m0, n0, row0, nk, kb):
                for j in range(4):
                    kk = udiv(row0, BK) + txl.min(kb + j, nk - 1)
                    for i in range(4):
                        txl.ptx.ld.global_.nc.f32(fv, sfa.ptr_to([kk, txl.min(m0 + lane + 32 * i, N - 1)]))
                        txl.assign(ra[i * 4 + j], txl.reinterpret("uint32", fv))
                        for c in range(N_SF_CHUNKS):
                            txl.ptx.ld.global_.nc.f32(
                                fv, sfb.ptr_to([kk, txl.min(n0 + c * 128 + lane + 32 * i, K - 1)])
                            )
                            txl.assign(rb[c * 16 + i * 4 + j], txl.reinterpret("uint32", fv))

            def body(m0, n0, g, row0, nk):
                load_group(m0, n0, row0, nk, 0)
                with txl.serial(nk) as k:
                    twait("sf-wait-empty", empty, st.stage, st.phase)
                    with txl.If(umod(k, 4) == 0), txl.Then():
                        txl.ptx.st.shared.v4.u32(
                            s_sfa.ptr_to([st.stage * 128 + lane * 4]),
                            pack4(ra, 0), pack4(ra, 4), pack4(ra, 8), pack4(ra, 12),
                        )  # fmt: skip
                        for c in range(N_SF_CHUNKS):
                            txl.ptx.st.shared.v4.u32(
                                s_sfb.ptr_to([st.stage * BN + c * 128 + lane * 4]),
                                pack4(rb, c * 16), pack4(rb, c * 16 + 4),
                                pack4(rb, c * 16 + 8), pack4(rb, c * 16 + 12),
                            )  # fmt: skip
                        with txl.If(k + 4 < nk), txl.Then():
                            load_group(m0, n0, row0, nk, k + 4)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    sf_full0.arrive(st.stage)
                    st.advance()

            tiles(body)

        with r_mma:
            with txl.If(rank == 0), txl.Then():
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    st = txl.PipelineState(STAGES, 0)
                    acc = txl.PipelineState(ACC_STAGES, 0)
                    a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                    b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                    sfa_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0])))
                    sfb_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0])))

                    def body(m0, n0, g, row0, nk):
                        twait("mma-wait-acc", t_empty, acc.stage, acc.phase ^ 1)
                        tile_tok = rng("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(nk) as k:
                            twait("mma-wait-tma", full, st.stage, st.phase)
                            twait("mma-wait-sf", sf_full, st.stage, st.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.If(umod(k, 4) == 0), txl.Then():
                                txl.ptx[UTCCP2](
                                    txl.uint32(SFA_COL), smem_desc(SF_DESC_BASE, sfa_base + u322(st.stage * 512))
                                )
                                for c in range(N_SF_CHUNKS):
                                    txl.ptx[UTCCP2](
                                        txl.uint32(SFB_COL + 4 * c),
                                        smem_desc(SF_DESC_BASE, sfb_base + u322(st.stage * (BN * 4) + c * 512)),
                                    )
                            idesc = txl.local_scalar("uint32", init=with_sf_id(txl.uint32(INSTR), umod(k, 4)))
                            for ki in range(BK // UMMA_K):
                                txl.ptx[MMA2](
                                    u322(acc.stage * BN),
                                    smem_desc(AB_DESC_BASE, a_base + u322(st.stage * A_BYTES + ki * K_STEP)),
                                    smem_desc(B_DESC_BASE, b_base + u322(st.stage * B_HALF + ki * B_K_STEP)),
                                    idesc,
                                    txl.uint32(SFA_COL),
                                    txl.uint32(SFB_COL),
                                    txl.Or(k > 0, ki > 0) if ki == 0 else txl.bool(True),
                                )
                            empty.arrive(st.stage, cta_group=2, cta_mask=3)
                            st.advance()
                        t_full.arrive(acc.stage, cta_group=2, cta_mask=3)
                        end(tile_tok)
                        acc.advance()

                    tiles(body)

        with r_epi:
            ew = warp - 4
            quad = warp % 4
            wg = ew // 4
            wtid = (ew % 4) * 32 + lane
            row = quad * 32 + lane
            acc = txl.PipelineState(ACC_STAGES, 0)
            vals = txl.alloc_local((COLS,), "uint32")

            def body(m0, n0, g, row0, nk):
                twait("epi-wait-acc", t_full, acc.stage, acc.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                tok = rng("epi-drain")
                for c in range(COLS // 32):
                    taddr = txl.bitwise_or(
                        txl.shift_left(u322(quad * 32), txl.uint32(16)), u322(acc.stage * BN + wg * COLS + c * 32)
                    )
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[vals[c * 32 + j] for j in range(32)], taddr)
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx.tcgen05.fence__before_thread_sync()
                t_empty.arrive(acc.stage, remote=0)
                end(tok)
                tok2 = rng("epi-store")
                for c in range(COLS // 32):
                    buf = (wg * CDB + c % CDB) * (BM * 32)
                    with txl.If(wtid == 0), txl.Then():
                        txl.ptx.cp.async_.bulk.wait_group.read(CDB - 1)
                    txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                    for q in range(8):
                        txl.ptx.st.shared.v4.u32(
                            s_cd.ptr_to([buf + row * 32 + txl.bitwise_xor(q, row % 8) * 4]),
                            vals[c * 32 + q * 4], vals[c * 32 + q * 4 + 1],
                            vals[c * 32 + q * 4 + 2], vals[c * 32 + q * 4 + 3],
                        )  # fmt: skip
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                    with txl.If(wtid == 0), txl.Then():
                        txl.ptx[REDUCE3D](
                            txl.address_of(d_map), n0 + wg * COLS + c * 32, m0, g, s_cd.ptr_to([buf])
                        )
                        txl.ptx.cp.async_.bulk.commit_group()
                end(tok2)
                acc.advance()

            tiles(body)
            with txl.If(wtid == 0), txl.Then():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with r_idle:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return kgrouped


def compile_v(kernel):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    return tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")


@functools.cache
def mgrouped_exec(STAGES, BN):
    return compile_v(build_mgrouped(STAGES, BN))


def fp8_grouped_gemm_sm100(a, b, n, grouped_mm_offs):
    x, sx = a
    w, sw = b
    M, K = x.shape
    G = w.shape[0]
    out = torch.empty((M, n), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return out
    ex = mgrouped_exec(6, 256)
    with tvm_ffi.use_torch_stream():
        ex(x, sx, w, sw, out, grouped_mm_offs, M, n, K, G)
    return out


@functools.cache
def kgrouped_exec(STAGES, CDB=1, REV=False, BN=256):
    return compile_v(build_kgrouped(STAGES, CDB, REV, BN))


def k_grouped_fp8_gemm_sm100(a, b, d, ks, ks_tensor):
    xa, sa = a
    xb, sb = b
    M, N = xa.shape
    K = xb.shape[1]
    G = d.shape[0]
    if M == 0 or G == 0:
        return
    # Small groups are store-bound: a double-buffered 256x128 pair tile overlaps the reduce with
    # the next tile's MMA. Large groups are MMA-bound: the 256x256 pair tile is denser.
    ex = kgrouped_exec(6, 1, True, 128) if M < 1024 * G else kgrouped_exec(5, 1, True, 256)
    with tvm_ffi.use_torch_stream():
        ex(xa, sa, xb, sb, d, ks_tensor, M, N, K, G)
