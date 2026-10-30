"""
pith-train SM100 FP8 expert GEMMs as fused TIRx-lite tcgen05 block-scaled kernels.

m-grouped (fwd / dgrad): out[rows of g] = a_g @ b[g].T, bf16.
k-grouped (wgrad):       d[g] += a_g^T @ b_g, fp32, accumulated in place by TMA reduce-add.

Both kernels are persistent, warp specialized (TMA / MMA / scale-factor / epilogue warps) and
read pith-train's fp32 E8M0 scales directly: a scale warp converts them to packed UE8M0 bytes in
shared memory, the MMA warp copies them to tensor memory with tcgen05.cp and issues
tcgen05.mma.kind::mxf8f6f4.block_scale (one scale per 128-wide K block, selected by sf_id).

Evolved with TIRx-harness: run fp8gemm, frontier member pair2cta-hybrid (commit 528b7c4, 8458 us vs 11658 us, 1.378x over the
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
EVICT_LAST = 0x14F0000000000000
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
PREFETCH3D = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"
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


def build_mgrouped(STAGES, BN, CDB=1):
    """
    2-CTA (cta_group::2) m-grouped kernel: a CTA pair computes 256 rows x 128 cols.

    Both 128-row halves of a pair belong to one group (pairs are formed per group); the odd last
    block of a group is computed by the pair's second CTA on rows it then does not store.
    """
    A_BYTES = BM * BK
    B_HALF = (BN // 2) * BK
    ACC_STAGES = 2 if BN <= 192 else 1  # two accumulators + 12 scale columns fit in 512 TMEM columns
    N_SF_CHUNKS = (BN + 127) // 128  # 128-row SFB chunks per tile
    SFA_COL = ACC_STAGES * BN
    SFB_COL = SFA_COL + 4
    EPI_WGS = 2 if BN >= 192 else 1
    EPI_WARPS = 4 * EPI_WGS
    COLS_PER_WG = BN // EPI_WGS
    assert COLS_PER_WG % 32 == 0 and (COLS_PER_WG // 32) % CDB == 0
    INSTR = encode_instr_descriptor_block_scaled_uint32(
        M=2 * BM, N=BN, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2,
    )  # fmt: skip
    # Final-wave split (BN=256): the last partial wave's tiles run as two 128-column halves when
    # those halves fit in one wave, halving the makespan tail.
    SPLIT = BN == 256
    INSTR_H = encode_instr_descriptor_block_scaled_uint32(
        M=2 * BM, N=BN // 2, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2,
    )  # fmt: skip
    MMA2 = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP2 = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LD2 = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
    LD3 = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def prelude(p):
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        bh_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = p["M"], p["N"], p["K"], p["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, 2, 0)
        txl.cu_tensor_map_encode_tiled(
            b_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 2, 1,
            1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            bh_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 4, 1,
            1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            d_map, "bfloat16", 2, p["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0
        )
        return a_map, b_map, bh_map, d_map

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
        a_map, b_map, bh_map, d_map = host
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
        # Issue the first 32 group ends' loads now; their latency overlaps the setup below.
        cur0 = txl.local_scalar("int32")
        prev0 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(cur0, offs.ptr_to([txl.min(lane, G - 1)]))
        txl.ptx.ld.global_.s32(prev0, offs.ptr_to([txl.max(txl.min(lane, G - 1) - 1, 0)]))
        setup_tok = rng("setup")

        smem = txl.smem_pool()
        s_a = smem.alloc((STAGES * A_BYTES,), "uint8", align=1024)
        s_b = smem.alloc((STAGES * B_HALF,), "uint8", align=1024)
        s_cd = smem.alloc((EPI_WGS * CDB * BM * 16,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * N_SF_CHUNKS * 128,), "uint32", align=16)
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
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        full0 = full.remote_view(0)
        sf_full0 = sf_full.remote_view(0)
        end(setup_tok)

        n_nb = (N + (BN - 1)) // BN
        nk = (K + (BK - 1)) // BK
        kb_sf = nk

        # Lanes take 32 groups at a time: pairs per group (ceil(rows / 256)), their running sum
        # (shfl.up scan + carry) and, by warp-wide reductions, the total and the group holding this
        # cluster's first pair, so tiles() starts there instead of walking from group 0.
        walk_tok = rng("walk")
        carry = txl.local_scalar("uint32", init=txl.uint32(0))
        cur = txl.local_scalar("int32")
        prev = txl.local_scalar("int32")
        pp0 = txl.local_scalar("int32", init=udiv(cluster, n_nb))
        init = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(6)]  # hit g gbeg gend pbeg pend
        peer = txl.local_scalar("uint32")

        def warp_sum(v):
            for x in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(peer, v, txl.uint32(x), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                txl.assign(v, v + peer)

        with txl.serial((G + 31) // 32) as wc:
            gi = txl.local_scalar("int32", init=wc * 32 + lane)
            with txl.If(wc == 0):
                with txl.Then():
                    txl.assign(cur, cur0)
                    txl.assign(prev, prev0)
                with txl.Else():
                    txl.ptx.ld.global_.s32(cur, offs.ptr_to([txl.min(gi, G - 1)]))
                    txl.ptx.ld.global_.s32(prev, offs.ptr_to([txl.max(txl.min(gi, G - 1) - 1, 0)]))
            with txl.If(gi == 0), txl.Then():
                txl.assign(prev, 0)
            cnt = txl.local_scalar("uint32", init=txl.if_then_else(gi < G, u322(udiv(cur - prev + 255, 256)), txl.uint32(0)))
            incl = txl.local_scalar("uint32", init=cnt)
            for x in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(x), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.If(lane >= x), txl.Then():
                    txl.assign(incl, incl + peer)
            pe = txl.local_scalar("int32", init=txl.cast(carry + incl, "int32"))
            pb = txl.local_scalar("int32", init=pe - txl.cast(cnt, "int32"))
            hit = txl.local_scalar("uint32", init=txl.if_then_else(txl.And(pb <= pp0, pp0 < pe), txl.uint32(1), txl.uint32(0)))
            vals = [hit, hit * u322(gi), hit * u322(prev), hit * u322(cur), hit * u322(pb), hit * u322(pe)]
            for j in range(6):
                v = txl.local_scalar("uint32", init=vals[j])
                warp_sum(v)
                txl.assign(init[j], init[j] + v)
            chunk = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.idx.b32(chunk, incl, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(carry, carry + chunk)
        total = txl.local_scalar("int32", init=txl.cast(carry, "int32"))
        found = init[0] != txl.uint32(0)
        end(walk_tok)

        def tiles(body):
            """
            Pair walk: t -> (pair p, n block); pairs are counted per group (ceil(rows/256)).
            """
            t = txl.local_scalar("int32", init=cluster)
            g = txl.local_scalar("int32", init=txl.if_then_else(found, txl.cast(init[1], "int32"), -1))
            gbeg = txl.local_scalar("int32", init=txl.cast(init[2], "int32"))  # first row of group g
            gend = txl.local_scalar("int32", init=txl.cast(init[3], "int32"))  # one past its last row
            pbeg = txl.local_scalar("int32", init=txl.cast(init[4], "int32"))  # first pair of group g
            pend = txl.local_scalar("int32", init=txl.cast(init[5], "int32"))  # one past its last pair
            n_tiles = total * n_nb
            tail = umod(n_tiles, NUM_SMS // 2) if SPLIT else 0
            tail = txl.if_then_else(2 * tail <= NUM_SMS // 2, tail, 0)
            full_end = n_tiles - tail  # tiles [full_end, n_tiles) run as two column halves each
            with txl.While(t < full_end + 2 * tail):
                hf = txl.local_scalar("int32", init=txl.if_then_else(t >= full_end, 1, 0))
                tile = txl.local_scalar(
                    "int32", init=txl.if_then_else(hf == 1, full_end + udiv(t - full_end, 2), t)
                )
                pp = txl.local_scalar("int32", init=udiv(tile, n_nb))
                nb = txl.local_scalar("int32", init=umod(tile, n_nb))
                col0 = txl.local_scalar(
                    "int32", init=nb * BN + txl.if_then_else(hf == 1, umod(t - full_end, 2) * (BN // 2), 0)
                )
                # The walk's precomputed start assumes the first tile id is `cluster`; a final-wave split
                # covering the whole problem (< one wave of tiles) starts earlier: restart from group 0.
                with txl.If(pp < pbeg), txl.Then():
                    txl.assign(g, -1)
                    txl.assign(gbeg, 0)
                    txl.assign(gend, 0)
                    txl.assign(pbeg, 0)
                    txl.assign(pend, 0)
                with txl.While(pp >= pend):
                    txl.assign(g, g + 1)
                    txl.assign(gbeg, gend)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([g]))
                    txl.assign(pbeg, pend)
                    txl.assign(pend, pend + udiv(gend - gbeg + 255, 256))
                m0 = txl.local_scalar("int32", init=gbeg + (pp - pbeg) * 256 + rank * BM)
                body(m0, col0, g, m0 < gend, hf)
                txl.assign(t, t + NUM_SMS // 2)

        with r_tma:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                def body(m0, col0, g, valid, hf):
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        bar = txl.cuda.cvta_generic_to_shared(full0.ptr_to([st.stage]))
                        with txl.If(rank == 0), txl.Then():
                            full0.arrive(
                                st.stage, txl.if_then_else(hf == 1, 2 * (A_BYTES + B_HALF // 2), 2 * (A_BYTES + B_HALF))
                            )
                        txl.ptx[LD2](
                            s_a.ptr_to([st.stage * A_BYTES]),
                            txl.address_of(a_map),
                            txl.cast(k * BK, "int32"),
                            m0,
                            bar,
                        )
                        with txl.If(hf == 0):
                            with txl.Then():
                                txl.ptx[LD3](
                                    s_b.ptr_to([st.stage * B_HALF]),
                                    txl.address_of(b_map),
                                    txl.cast(k * BK, "int32"),
                                    col0 + rank * (BN // 2),
                                    g,
                                    bar,
                                )
                            with txl.Else():
                                txl.ptx[LD3](
                                    s_b.ptr_to([st.stage * B_HALF]),
                                    txl.address_of(bh_map),
                                    txl.cast(k * BK, "int32"),
                                    col0 + rank * (BN // 4),
                                    g,
                                    bar,
                                )
                        st.advance()

                tiles(body)

        with r_sf:
            st = txl.PipelineState(STAGES, 1)
            ra = txl.alloc_local((16,), "uint32")
            # The tile's B rows [nb*BN, nb*BN + BN) span at most 2 (BN <= 128: 1, 192: 2, 256: 2)
            # 128-row weight-scale blocks; 32-row groups never straddle a block (BN % 64 == 0).
            NBLK = 1 if BN <= 128 else 2
            rb = txl.alloc_local((4 * NBLK,), "uint32")
            fv = txl.local_scalar("float32")
            n_sfb = (N + 127) // 128

            def load_group(m0, col0, g, kb):
                for j in range(4):
                    kk = txl.min(kb + j, kb_sf - 1)
                    for i in range(4):
                        txl.ptx.ld.global_.nc.f32(fv, sfa.ptr_to([txl.min(m0 + lane + 32 * i, M - 1), kk]))
                        txl.assign(ra[i * 4 + j], txl.reinterpret("uint32", fv))
                    for c in range(NBLK):
                        nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                        txl.ptx.ld.global_.nc.f32(fv, sfb.ptr_to([g, nsf, kk]))
                        txl.assign(rb[c * 4 + j], txl.reinterpret("uint32", fv))

            def body(m0, col0, g, valid, hf):
                load_group(m0, col0, g, 0)
                with txl.serial(nk) as k:
                    twait("sf-wait-empty", empty, st.stage, st.phase)
                    # The previous tcgen05.cp of this stage (async proxy) completed; order it before our writes.
                    txl.ptx.fence.proxy.async_.shared__cta()
                    with txl.If(umod(k, 4) == 0), txl.Then():
                        txl.ptx.st.shared.v4.u32(
                            s_sfa.ptr_to([st.stage * 128 + lane * 4]),
                            pack4(ra, 0), pack4(ra, 4), pack4(ra, 8), pack4(ra, 12),
                        )  # fmt: skip
                        wb0 = txl.local_scalar("uint32", init=pack4(rb, 0))
                        wb1 = txl.local_scalar("uint32", init=pack4(rb, 4 * (NBLK - 1)))
                        boff = umod(col0, 128)  # tile start within its first scale block
                        for c in range(N_SF_CHUNKS):
                            # word i of chunk c holds B rows c*128 + 32*i + lane of the tile
                            ws = [
                                txl.if_then_else(boff + (c * 128 + 32 * i) >= 128, wb1, wb0) for i in range(4)
                            ]
                            txl.ptx.st.shared.v4.u32(
                                s_sfb.ptr_to([(st.stage * N_SF_CHUNKS + c) * 128 + lane * 4]), *ws
                            )
                        with txl.If(k + 4 < nk), txl.Then():
                            load_group(m0, col0, g, k + 4)
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

                    def body(m0, col0, g, valid, hf):
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
                                        smem_desc(
                                            SF_DESC_BASE, sfb_base + u322((st.stage * N_SF_CHUNKS + c) * 512)
                                        ),
                                    )
                            idesc = txl.local_scalar(
                                "uint32",
                                init=with_sf_id(
                                    txl.if_then_else(hf == 1, txl.uint32(INSTR_H), txl.uint32(INSTR)), umod(k, 4)
                                ),
                            )
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

            def body(m0, col0, g, valid, hf):
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
                    buf = (wg * CDB + c % CDB) * (BM * 16)
                    with txl.If(wtid == 0), txl.Then():
                        txl.ptx.cp.async_.bulk.wait_group.read(CDB - 1)
                    txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                    for q in range(4):
                        base = c * 16 + q * 4
                        txl.ptx.st.shared.v4.u32(
                            s_cd.ptr_to([buf + row * 16 + txl.bitwise_xor(q, (row // 2) % 4) * 4]),
                            packed[base], packed[base + 1], packed[base + 2], packed[base + 3],
                        )  # fmt: skip
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                    # A half tile's 128 columns all belong to warpgroup 0.
                    with txl.If(txl.And(txl.And(wtid == 0, valid), txl.Or(hf == 0, wg == 0))), txl.Then():
                        txl.ptx[STORE2D](
                            txl.address_of(d_map),
                            col0 + wg * COLS_PER_WG + c * 32,
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


def build_mgrouped_swap(STAGES):
    """
    Swap-AB 2-CTA m-grouped kernel: MMA M = 256 weight rows (128 per CTA), MMA N = tokens.

    A pair tile covers 256 tokens of one group, or 128 for a group's odd tail (MMA N = 128, half
    the work), times 256 output columns; CTA r stores columns [r*128, r*128+128) of the tile.
    """
    W_BYTES = 128 * BK  # weight rows per CTA
    T_BYTES = 128 * BK  # token rows per CTA (full tile; a tail tile uses half)
    SFA_COL = 256
    SFB_COL = SFA_COL + 4
    EPI_WARPS = 8
    INSTR_F = encode_instr_descriptor_block_scaled_uint32(
        M=256, N=256, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2,
    )  # fmt: skip
    INSTR_H = encode_instr_descriptor_block_scaled_uint32(
        M=256, N=128, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2,
    )  # fmt: skip
    MMA2 = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP2 = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LD2 = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
    LD3 = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def prelude(p):
        a_map = txl.stack_alloca("tensormap", 1)
        a64_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = p["M"], p["N"], p["K"], p["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, K, M, K, BK, 128, 1, 1, 0, 3, 2, 0)
        txl.cu_tensor_map_encode_tiled(a64_map, FP8, 2, p["a"].data, K, M, K, BK, 64, 1, 1, 0, 3, 2, 0)
        txl.cu_tensor_map_encode_tiled(
            b_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, 128, 1,
            1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            d_map, "bfloat16", 2, p["d"].data, N, M, N * 2, 128, 32, 1, 1, 0, 0, 2, 0
        )
        return a_map, a64_map, b_map, d_map

    @txl.kernel(warps=4 + EPI_WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_SMS, host_prelude=prelude)
    def mgrouped_swap(
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
        a_map, a64_map, b_map, d_map = host
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
        # Issue the first 32 group ends' loads now; their latency overlaps the setup below.
        cur0 = txl.local_scalar("int32")
        prev0 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(cur0, offs.ptr_to([txl.min(lane, G - 1)]))
        txl.ptx.ld.global_.s32(prev0, offs.ptr_to([txl.max(txl.min(lane, G - 1) - 1, 0)]))

        smem = txl.smem_pool()
        s_w = smem.alloc((STAGES * W_BYTES,), "uint8", align=1024)
        s_t = smem.alloc((STAGES * T_BYTES,), "uint8", align=1024)
        s_cd = smem.alloc((2 * 32 * 64,), "uint32", align=1024)  # per WG: 32 tokens x 128 cols bf16
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * 256,), "uint32", align=16)
        full = txl.TMABar(smem, STAGES)
        empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        t_full = txl.TCGen05Bar(smem, 1)
        t_empty = txl.MBarrier(smem, 1)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        full.init(1)
        empty.init(1)
        sf_full.init(64)
        t_full.init(1)
        t_empty.init(2 * 32 * EPI_WARPS)
        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        full0 = full.remote_view(0)
        sf_full0 = sf_full.remote_view(0)

        n_nb = (N + 255) // 256
        nk = (K + (BK - 1)) // BK
        kb_sf = nk

        # Lanes take 32 groups at a time: pairs per group (ceil(rows / 256)), their running sum
        # (shfl.up scan + carry) and, by warp-wide reductions, the total and the group holding this
        # cluster's first pair, so tiles() starts there instead of walking from group 0.
        walk_tok = rng("walk")
        carry = txl.local_scalar("uint32", init=txl.uint32(0))
        cur = txl.local_scalar("int32")
        prev = txl.local_scalar("int32")
        pp0 = txl.local_scalar("int32", init=udiv(cluster, n_nb))
        init = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(6)]  # hit g gbeg gend pbeg pend
        peer = txl.local_scalar("uint32")

        def warp_sum(v):
            for x in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(peer, v, txl.uint32(x), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                txl.assign(v, v + peer)

        with txl.serial((G + 31) // 32) as wc:
            gi = txl.local_scalar("int32", init=wc * 32 + lane)
            with txl.If(wc == 0):
                with txl.Then():
                    txl.assign(cur, cur0)
                    txl.assign(prev, prev0)
                with txl.Else():
                    txl.ptx.ld.global_.s32(cur, offs.ptr_to([txl.min(gi, G - 1)]))
                    txl.ptx.ld.global_.s32(prev, offs.ptr_to([txl.max(txl.min(gi, G - 1) - 1, 0)]))
            with txl.If(gi == 0), txl.Then():
                txl.assign(prev, 0)
            cnt = txl.local_scalar("uint32", init=txl.if_then_else(gi < G, u322(udiv(cur - prev + 255, 256)), txl.uint32(0)))
            incl = txl.local_scalar("uint32", init=cnt)
            for x in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(x), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.If(lane >= x), txl.Then():
                    txl.assign(incl, incl + peer)
            pe = txl.local_scalar("int32", init=txl.cast(carry + incl, "int32"))
            pb = txl.local_scalar("int32", init=pe - txl.cast(cnt, "int32"))
            hit = txl.local_scalar("uint32", init=txl.if_then_else(txl.And(pb <= pp0, pp0 < pe), txl.uint32(1), txl.uint32(0)))
            vals = [hit, hit * u322(gi), hit * u322(prev), hit * u322(cur), hit * u322(pb), hit * u322(pe)]
            for j in range(6):
                v = txl.local_scalar("uint32", init=vals[j])
                warp_sum(v)
                txl.assign(init[j], init[j] + v)
            chunk = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.idx.b32(chunk, incl, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(carry, carry + chunk)
        total = txl.local_scalar("int32", init=txl.cast(carry, "int32"))
        found = init[0] != txl.uint32(0)
        end(walk_tok)

        def tiles(body):
            """
            Pair walk: t -> (token tile p, n block); a group has ceil(rows/256) token tiles.
            """
            t = txl.local_scalar("int32", init=cluster)
            g = txl.local_scalar("int32", init=txl.if_then_else(found, txl.cast(init[1], "int32"), -1))
            gbeg = txl.local_scalar("int32", init=txl.cast(init[2], "int32"))
            gend = txl.local_scalar("int32", init=txl.cast(init[3], "int32"))
            pbeg = txl.local_scalar("int32", init=txl.cast(init[4], "int32"))
            pend = txl.local_scalar("int32", init=txl.cast(init[5], "int32"))
            with txl.While(t < total * n_nb):
                pp = txl.local_scalar("int32", init=udiv(t, n_nb))
                nb = txl.local_scalar("int32", init=umod(t, n_nb))
                with txl.While(pp >= pend):
                    txl.assign(g, g + 1)
                    txl.assign(gbeg, gend)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([g]))
                    txl.assign(pbeg, pend)
                    txl.assign(pend, pend + udiv(gend - gbeg + 255, 256))
                m0 = txl.local_scalar("int32", init=gbeg + (pp - pbeg) * 256)
                body(m0, nb, g, gend - m0 >= 256)
                txl.assign(t, t + NUM_SMS // 2)

        with r_tma:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(a64_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                def body(m0, nb, g, fullt):
                    tx = txl.local_scalar(
                        "int32", init=txl.if_then_else(fullt, 2 * (W_BYTES + T_BYTES), 2 * (W_BYTES + T_BYTES // 2))
                    )
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        bar = txl.cuda.cvta_generic_to_shared(full0.ptr_to([st.stage]))
                        with txl.If(rank == 0), txl.Then():
                            full0.arrive(st.stage, tx)
                        txl.ptx[LD3](
                            s_w.ptr_to([st.stage * W_BYTES]),
                            txl.address_of(b_map),
                            txl.cast(k * BK, "int32"),
                            nb * 256 + rank * 128,
                            g,
                            bar,
                        )
                        with txl.If(fullt):
                            with txl.Then():
                                txl.ptx[LD2](
                                    s_t.ptr_to([st.stage * T_BYTES]),
                                    txl.address_of(a_map),
                                    txl.cast(k * BK, "int32"),
                                    m0 + rank * 128,
                                    bar,
                                )
                            with txl.Else():
                                txl.ptx[LD2](
                                    s_t.ptr_to([st.stage * T_BYTES]),
                                    txl.address_of(a64_map),
                                    txl.cast(k * BK, "int32"),
                                    m0 + rank * 64,
                                    bar,
                                )
                        st.advance()

                tiles(body)

        with r_sf:
            st = txl.PipelineState(STAGES, 1)
            ra = txl.alloc_local((4,), "uint32")
            rb = txl.alloc_local((32,), "uint32")
            fv = txl.local_scalar("float32")
            n_sfb = (N + 127) // 128

            def load_group(m0, nb, g, kb):
                nsf = txl.min(nb * 2 + rank, n_sfb - 1)
                for j in range(4):
                    kk = txl.min(kb + j, kb_sf - 1)
                    txl.ptx.ld.global_.nc.f32(fv, sfb.ptr_to([g, nsf, kk]))
                    txl.assign(ra[j], txl.reinterpret("uint32", fv))
                    for c in range(2):
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(
                                fv, sfa.ptr_to([txl.min(m0 + c * 128 + lane + 32 * i, M - 1), kk])
                            )
                            txl.assign(rb[c * 16 + i * 4 + j], txl.reinterpret("uint32", fv))

            def body(m0, nb, g, fullt):
                load_group(m0, nb, g, 0)
                with txl.serial(nk) as k:
                    twait("sf-wait-empty", empty, st.stage, st.phase)
                    # The previous tcgen05.cp of this stage (async proxy) completed; order it before our writes.
                    txl.ptx.fence.proxy.async_.shared__cta()
                    with txl.If(umod(k, 4) == 0), txl.Then():
                        wa = txl.local_scalar("uint32", init=pack4(ra, 0))
                        txl.ptx.st.shared.v4.u32(s_sfa.ptr_to([st.stage * 128 + lane * 4]), wa, wa, wa, wa)
                        for c in range(2):
                            txl.ptx.st.shared.v4.u32(
                                s_sfb.ptr_to([st.stage * 256 + c * 128 + lane * 4]),
                                pack4(rb, c * 16), pack4(rb, c * 16 + 4),
                                pack4(rb, c * 16 + 8), pack4(rb, c * 16 + 12),
                            )  # fmt: skip
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
                    acc = txl.PipelineState(1, 0)
                    w_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_w.ptr_to([0])))
                    t_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_t.ptr_to([0])))
                    sfa_base = txl.local_scalar(
                        "uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0]))
                    )
                    sfb_base = txl.local_scalar(
                        "uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0]))
                    )

                    def body(m0, nb, g, fullt):
                        instr = txl.local_scalar(
                            "uint32", init=txl.if_then_else(fullt, txl.uint32(INSTR_F), txl.uint32(INSTR_H))
                        )
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
                                for c in range(2):
                                    txl.ptx[UTCCP2](
                                        txl.uint32(SFB_COL + 4 * c),
                                        smem_desc(SF_DESC_BASE, sfb_base + u322(st.stage * 1024 + c * 512)),
                                    )
                            idesc = txl.local_scalar("uint32", init=with_sf_id(instr, umod(k, 4)))
                            for ki in range(BK // UMMA_K):
                                txl.ptx[MMA2](
                                    txl.uint32(0),
                                    smem_desc(AB_DESC_BASE, w_base + u322(st.stage * W_BYTES + ki * UMMA_K)),
                                    smem_desc(AB_DESC_BASE, t_base + u322(st.stage * T_BYTES + ki * UMMA_K)),
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
            par = lane % 2
            col = quad * 32 + lane - par  # even output column of this lane's pair
            acc = txl.PipelineState(1, 0)
            vals = txl.alloc_local((128,), "uint32")
            packed = txl.alloc_local((16,), "uint32")
            peer = txl.local_scalar("uint32")
            buf = wg * (32 * 64)

            def body(m0, nb, g, fullt):
                live = txl.Or(fullt, wg == 0)
                twait("epi-wait-acc", t_full, acc.stage, acc.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                with txl.If(live), txl.Then():
                    for c in range(4):
                        taddr = txl.bitwise_or(
                            txl.shift_left(u322(quad * 32), txl.uint32(16)), u322(wg * 128 + c * 32)
                        )
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                            *[vals[c * 32 + j] for j in range(32)], taddr
                        )
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                # The raw accumulator is in registers: release TMEM before packing and storing.
                txl.ptx.tcgen05.fence__before_thread_sync()
                t_empty.arrive(acc.stage, remote=0)
                for c in range(4):
                    with txl.If(live), txl.Then():
                        # Lane pairs swap halves so each lane owns two adjacent columns of one token.
                        for p in range(16):
                            txl.ptx.shfl_sync.bfly.b32(
                                peer,
                                txl.if_then_else(par == 0, vals[c * 32 + 2 * p + 1], vals[c * 32 + 2 * p]),
                                txl.uint32(1),
                                txl.uint32(31),
                                txl.uint32(0xFFFFFFFF),
                            )
                            lo = txl.if_then_else(par == 0, vals[c * 32 + 2 * p], peer)
                            hi = txl.if_then_else(par == 0, peer, vals[c * 32 + 2 * p + 1])
                            txl.ptx.cvt.rn.bf16x2.f32(
                                packed[p], txl.reinterpret("float32", hi), txl.reinterpret("float32", lo)
                            )
                        with txl.If(wtid == 0), txl.Then():
                            txl.ptx.cp.async_.bulk.wait_group.read(0)
                        txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                        for p in range(16):
                            txl.ptx.st.shared.u32(s_cd.ptr_to([buf + (2 * p + par) * 64 + col // 2]), packed[p])
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                        with txl.If(wtid == 0), txl.Then():
                            txl.ptx[STORE2D](
                                txl.address_of(d_map), nb * 256 + rank * 128, m0 + wg * 128 + c * 32, s_cd.ptr_to([buf])
                            )
                            txl.ptx.cp.async_.bulk.commit_group()
                acc.advance()

            tiles(body)
            with txl.If(wtid == 0), txl.Then():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with r_idle:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped_swap


# ----------------------------------------------------------------------------------------------
# k-grouped: d (G, N, K) fp32 += A_g^T B_g; A (rows, N), B (rows, K), both MN-major operands
# ----------------------------------------------------------------------------------------------


SCHED_DEPTH = 3  # tile ids the scheduler may claim ahead of the epilogue
SCHED_CONSUMERS = 21  # per tile id: CTA0 tma + sf + mma + 8 epi warps, CTA1 tma + sf + 8 epi warps


def build_kgrouped(STAGES, CDB=1, REV=False, BN=256, PF=False, EPF=False):
    """
    2-CTA k-grouped kernel: a CTA pair accumulates a 256 (N) x 256 (K) fp32 tile of d[g].

    Tiles are claimed dynamically: the leader CTA's idle warp takes tile ids from a global atomic
    counter (ctr[0]) and hands them to every role of both CTAs through a shared-memory ring. The
    last pair to drain the counter resets ctr to zero for the next launch.
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
        ctr: txl.gptr[txl.i32, 1],
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
        s_tile = smem.alloc((SCHED_DEPTH,), "int32", align=16)
        q_full = txl.MBarrier(smem, SCHED_DEPTH)
        q_empty = txl.MBarrier(smem, SCHED_DEPTH)
        q_full.init(1)
        q_empty.init(SCHED_CONSUMERS)
        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        full0 = full.remote_view(0)
        sf_full0 = sf_full.remote_view(0)
        q_empty0 = q_empty.remote_view(0)

        n_mt = (N + 255) // 256
        n_nt = (K + (BN - 1)) // BN
        per_group = n_mt * n_nt
        total = G * per_group

        def tiles(body, one_lane=True):
            """
            Pair walk over (group, 256-row tile of d, 256-col tile of d); empty groups skipped.

            Tile ids arrive in increasing order from the scheduler ring; -1 ends the walk.
            one_lane: called by a single elected thread (else by a full warp, lane 0 releases).
            """
            qs = txl.PipelineState(SCHED_DEPTH, 0)
            t = txl.local_scalar("int32", init=0)

            def next_tile():
                txl.cuda.mbarrier_wait_acquire_cluster(q_full.ptr_to([qs.stage]), qs.phase)
                txl.ptx.ld.shared.s32(t, s_tile.ptr_to([qs.stage]))
                if not one_lane:
                    txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
                with txl.If(txl.Or(txl.bool(one_lane), lane == 0)), txl.Then():
                    q_empty0.arrive(qs.stage)
                qs.advance()

            next_tile()
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
            with txl.While(t >= 0):
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
                next_tile()

        with r_tma:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                txl.ptx.prefetch.tensormap(txl.address_of(d_map))

                def body(m0, n0, g, row0, nk):
                    # PF: pull this CTA's d slice into L2 ahead of the epilogue's reduce-add. A reduce that
                    # misses L2 waits on its own DRAM read, exposed when tiles are short (small groups).
                    if PF:
                        for c in range(BN // 32):
                            txl.ptx[PREFETCH3D](txl.address_of(d_map), n0 + c * 32, m0, g, u64c(EVICT_LAST))
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
                    # The previous tcgen05.cp of this stage (async proxy) completed; order it before our writes.
                    txl.ptx.fence.proxy.async_.shared__cta()
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

            tiles(body, one_lane=False)

        with r_idle:
            with txl.If(rank == 0), txl.Then():
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    qs = txl.PipelineState(SCHED_DEPTH, 1)
                    nt = txl.local_scalar("int32", init=0)
                    peer_tile = txl.local_scalar("uint32")
                    peer_full = txl.local_scalar("uint32")
                    with txl.While(nt >= 0):
                        txl.cuda.mbarrier_wait(q_empty.ptr_to([qs.stage]), qs.phase)
                        txl.ptx.atom.global_.add.s32(nt, ctr.ptr_to([0]), txl.int32(1))
                        with txl.If(nt >= total), txl.Then():
                            txl.assign(nt, -1)
                        txl.ptx.st.shared.s32(s_tile.ptr_to([qs.stage]), nt)
                        txl.ptx["mapa.shared::cluster.u32"](
                            peer_tile, txl.cuda.cvta_generic_to_shared(s_tile.ptr_to([qs.stage])), txl.uint32(1)
                        )
                        txl.ptx["st.shared::cluster.s32"](peer_tile, nt)
                        txl.ptx["mapa.shared::cluster.u32"](
                            peer_full, txl.cuda.cvta_generic_to_shared(q_full.ptr_to([qs.stage])), txl.uint32(1)
                        )
                        txl.ptx["mbarrier.arrive.release.cluster.shared::cluster.b64"](peer_full)
                        txl.ptx["mbarrier.arrive.release.cluster.shared::cta.b64"](
                            txl.cuda.cvta_generic_to_shared(q_full.ptr_to([qs.stage]))
                        )
                        qs.advance()
                    # The last pair to see the counter drained resets it for the next launch.
                    done = txl.local_scalar("int32")
                    # acq_rel: each pair releases its last ticket fetch; the last pair acquires them all
                    # before resetting the counter.
                    txl.ptx["atom.acq_rel.gpu.global.add.s32"](done, ctr.ptr_to([1]), txl.int32(1))
                    with txl.If(done == NUM_SMS // 2 - 1), txl.Then():
                        txl.ptx.st.global_.s32(ctr.ptr_to([0]), txl.int32(0))
                        txl.ptx.st.global_.s32(ctr.ptr_to([1]), txl.int32(0))

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
                if EPF:
                    # Long tiles: prefetch this warpgroup's d slice while the MMA accumulates it.
                    with txl.If(wtid == 0), txl.Then():
                        for c in range(COLS // 32):
                            txl.ptx[PREFETCH3D](
                                txl.address_of(d_map), n0 + wg * COLS + c * 32, m0, g, u64c(EVICT_LAST)
                            )
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
                        # evict_last keeps the reduced d lines resident in L2 while the stream runs
                        txl.ptx[REDUCE3D + ".L2::cache_hint"](
                            txl.address_of(d_map), n0 + wg * COLS + c * 32, m0, g, s_cd.ptr_to([buf]), u64c(EVICT_LAST)
                        )
                        txl.ptx.cp.async_.bulk.commit_group()
                end(tok2)
                acc.advance()

            tiles(body, one_lane=False)
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
def mgrouped_swap_exec(STAGES):
    return compile_v(build_mgrouped_swap(STAGES))


@functools.cache
def mgrouped_exec(STAGES, BN, CDB=1):
    return compile_v(build_mgrouped(STAGES, BN, CDB))


def mcfg(K, N):
    """
    (STAGES, BN, CDB) per shape: BN=192 (double-buffered accumulator, 7 stages) when it tiles N
    exactly and 256 does not; otherwise 256-wide tiles, with 4 store buffers when K is short.
    """
    if N % 256 != 0 and N % 192 == 0:
        return (7, 192, 1)
    return (5, 256, 2) if K <= 512 else (6, 256, 1)


def fp8_grouped_gemm_sm100(a, b, n, grouped_mm_offs):
    x, sx = a
    w, sw = b
    M, K = x.shape
    G = w.shape[0]
    out = torch.empty((M, n), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return out
    # Small groups (< 512 rows on average) waste most on odd 128-row tails: swap A/B so tokens are
    # the MMA N dimension and a tail costs half a tile. Weight-side padding (n % 256) costs far less.
    # With short K the swap epilogue (lane-pair transpose, 32-token store boxes) costs more than the
    # tail it saves (qwen3.5 EP2-like down, K=512: 64.5 vs 60.5 us).
    # BN=192 tiles (exact for n % 192 == 0) load 14% more operand bytes per FLOP than the swap kernel's
    # 256x256 tiles; when padding n up to 256 wastes <= 5% (n = 5760: 2.2%), the swap kernel wins.
    pad256 = (n + 255) // 256 * 256
    swap_pad = n % 256 != 0 and n % 192 == 0 and pad256 * 20 <= n * 21
    if (M < 512 * G or swap_pad) and K > 512:
        ex = mgrouped_swap_exec(6)
    else:
        ex = mgrouped_exec(*mcfg(K, n))
    with tvm_ffi.use_torch_stream():
        ex(x, sx, w, sw, out, grouped_mm_offs, M, n, K, G)
    return out


@functools.cache
def kgrouped_exec(STAGES, CDB=1, REV=False, BN=256, PF=False, EPF=False):
    return compile_v(build_kgrouped(STAGES, CDB, REV, BN, PF, EPF))


@txl.kernel(warps=1, arch="sm_100a", grid=1)
def zero2(ctr: txl.gptr[txl.i32, 1]):
    with txl.If(txl.thread_id() < 2), txl.Then():
        txl.ptx.st.global_.s32(ctr.ptr_to([txl.thread_id()]), txl.int32(0))


COUNTERS = {}


def tile_counter(device):
    """
    Per (device, stream) tile counter, zeroed once; each k-grouped launch leaves it zero.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    ctr = COUNTERS.get(key)
    if ctr is None:
        ctr = torch.empty(2, dtype=torch.int32, device=device)
        with tvm_ffi.use_torch_stream():
            zero_exec()(ctr)
        COUNTERS[key] = ctr
    return ctr


@functools.cache
def zero_exec():
    return compile_v(zero2)


def k_grouped_fp8_gemm_sm100(a, b, d, ks, ks_tensor):
    xa, sa = a
    xb, sb = b
    M, N = xa.shape
    K = xb.shape[1]
    G = d.shape[0]
    if M == 0 or G == 0:
        return
    # 256x256 pair tiles: half the A re-reads of 256x128 tiles; dynamic tile claiming balances
    # the per-SM reduce throughput.
    # d prefetch into L2 by average group size: small groups (short tiles) from the TMA warp, well ahead;
    # medium groups from the epilogue at tile start (about one MMA tile of lead); none for large groups.
    ex = kgrouped_exec(5, 1, True, 256, M < 512 * G, 512 * G <= M < 1024 * G)
    with tvm_ffi.use_torch_stream():
        ex(xa, sa, xb, sb, d, ks_tensor, tile_counter(d.device), M, N, K, G)
