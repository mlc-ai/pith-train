"""
pith-train SM100 FP8 expert GEMMs as fused TIRx-lite tcgen05 block-scaled kernels.

m-grouped (fwd / dgrad): out[rows of g] = a_g @ b[g].T, bf16.
k-grouped (wgrad):       d[g] += a_g^T @ b_g, fp32, accumulated in place by TMA reduce-add.

Both kernels are persistent, warp specialized (TMA / MMA / scale-factor / epilogue warps) and
read pith-train's fp32 E8M0 scales directly: a scale warp converts them to packed UE8M0 bytes in
shared memory, the MMA warp copies them to tensor memory with tcgen05.cp and issues
tcgen05.mma.kind::mxf8f6f4.block_scale (one scale per 128-wide K block, selected by sf_id).

The m-grouped GEMM picks a route by shape: 256 x 320 pair tiles as two N=160 sub-MMAs over
rotating TMEM slots (n a multiple of 320 but not of 256, gpt-oss), a 1-CTA 128 x 256 kernel for
small groups, swap-AB pair tiles (tokens on the MMA N side) for other small-group shapes, 256-wide
tiles as two N=128 sub-MMAs over rotating slots for long qwen rows, or 2-CTA BN=192 / BN=256
tiles. The k-grouped GEMM claims 256 x 256 pair tiles dynamically from a per-(device, stream)
counter and is launched with programmatic dependent launch (its prologue overlaps the previous
kernel's tail; every global access follows griddepcontrol.wait).

Evolved with TIRx-harness: run fp8gemm, frontier member pair2cta-hybrid (commit db8dbbc, 8021 us
vs 11612 us, 1.448x over the previous SM100 path in the task's benchmark). Compiled once per process
by tvm_tirx.tirx.compile(..., pipeline="tirx") and launched through tvm-ffi on torch's current
stream.
"""

import contextlib
import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

from tvm_tirx.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_smem_descriptor_base_uint64,
)

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc as before; a cubin (sm_100a SASS only, as the old -gencode build), since -arch sm_100a would also emit compute_100 PTX, which rejects sm_100a-only instructions

__all__ = ["fp8_grouped_gemm_sm100", "k_grouped_fp8_gemm_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


NUM_SMS = 148
BM = 128
BK = 128
UMMA_K = 32
EPI_BAR = 8
EVICT_NORMAL = 0x1000000000000000
EVICT_LAST = 0x14F0000000000000
FP8 = "float8_e4m3fn"
L2P = 3  # TMA L2 sector promotion for m-grouped operands: 256 B (a miss also fetches the next K block)


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
    lo = txl.local_scalar("uint16")
    hi = txl.local_scalar("uint16")
    out = txl.local_scalar("uint32")
    txl.ptx.cvt.rz.satfinite.ue8m0x2.f32(
        lo, txl.reinterpret(vals[base + 1], ty="float32"), txl.reinterpret(vals[base], ty="float32")
    )
    txl.ptx.cvt.rz.satfinite.ue8m0x2.f32(
        hi, txl.reinterpret(vals[base + 3], ty="float32"), txl.reinterpret(vals[base + 2], ty="float32")
    )
    txl.ptx.mov.b32(out, lo, hi)
    return out


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


@contextlib.contextmanager
def ktail(ki, cond):
    """
    Guard a K step past the first of a 128-wide K block: skipped beyond K (zero-filled K tail).
    """
    if ki == 0:
        yield
    else:
        with txl.if_(cond), txl.then_():
            yield


def dadd(desc, byte_off):
    """
    Descriptor of the operand at +byte_off (16-B units in the start-address field; no carry out).
    """
    return desc if byte_off == 0 else desc + txl.uint64(byte_off >> 4)


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


def build_mgrouped(STAGES, BN, CDB=1, SFT=False, VSF=False):
    """
    2-CTA (cta_group::2) m-grouped kernel: a CTA pair computes 256 rows x 128 cols.

    Both 128-row halves of a pair belong to one group (pairs are formed per group); the odd last
    block of a group is computed by the pair's second CTA on rows it then does not store.
    """
    A_BYTES = BM * BK
    B_HALF = (BN // 2) * BK
    ACC_STAGES = 2 if BN <= 192 else 1  # two accumulators + 12 scale columns fit in 512 TMEM columns
    # SFT (BN=256): two 256-column accumulators anyway; a tile's scale columns live in cols 240-251 of the
    # OTHER accumulator, freed when the previous tile's epilogue has read that tail chunk (drained first).
    SFT = SFT and BN == 256
    if SFT:
        ACC_STAGES = 2
    N_SF_CHUNKS = (BN + 127) // 128  # 128-row SFB chunks per tile
    SFA_COL = ACC_STAGES * BN
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
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, L2P, 0)
        txl.cu_tensor_map_encode_tiled(
            b_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 2, 1,
            1, 1, 1, 0, 3, L2P, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            bh_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 4, 1,
            1, 1, 1, 0, 3, L2P, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            d_map, "bfloat16", 2, p["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0
        )
        return a_map, b_map, bh_map, d_map

    @txl.kernel()
    def mgrouped(
        a: txl.Tensor((txl.int64(), txl.int64()), FP8),
        sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32),
        b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8),
        sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32),
        d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16),
        offs: txl.Tensor((txl.int64(),), txl.i32),
        M: txl.int32,
        N: txl.int32,
        K: txl.int32,
        G: txl.int32,
    ):
        host = prelude(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=(4 + EPI_WARPS) * 32, cluster=[2]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, bh_map, d_map = host
        rank = txl.cuda.cluster_cta_id("x")
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
        tail_free = txl.MBarrier(smem, 2)
        tail_free.init(2 * 32 * 4)  # warpgroup 1 (columns 128-255) of both CTAs
        with txl.if_(tid == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
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
            # one redux.sync per value: the six walk reductions are independent (no 5-deep shuffle chains)
            txl.ptx.redux_sync.add.u32(v, v, txl.uint32(0xFFFFFFFF))

        with txl.serial((G + 31) // 32) as wc:
            gi = txl.local_scalar("int32", init=wc * 32 + lane)
            with txl.if_(wc == 0):
                with txl.then_():
                    txl.assign(cur, cur0)
                    txl.assign(prev, prev0)
                with txl.else_():
                    txl.ptx.ld.global_.s32(cur, offs.ptr_to([txl.min(gi, G - 1)]))
                    txl.ptx.ld.global_.s32(prev, offs.ptr_to([txl.max(txl.min(gi, G - 1) - 1, 0)]))
            with txl.if_(gi == 0), txl.then_():
                txl.assign(prev, 0)
            cnt = txl.local_scalar("uint32", init=txl.if_then_else(gi < G, u322(udiv(cur - prev + 255, 256)), txl.uint32(0)))
            incl = txl.local_scalar("uint32", init=cnt)
            for x in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(x), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.if_(lane >= x), txl.then_():
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
        # the group walk (registers + global loads only) overlaps the cluster barrier latency
        txl.ptx.barrier.cluster.wait.acquire.aligned()

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
            with txl.while_(t < full_end + 2 * tail):
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
                with txl.if_(pp < pbeg), txl.then_():
                    txl.assign(g, -1)
                    txl.assign(gbeg, 0)
                    txl.assign(gend, 0)
                    txl.assign(pbeg, 0)
                    txl.assign(pend, 0)
                with txl.while_(pp >= pend):
                    txl.assign(g, g + 1)
                    txl.assign(gbeg, gend)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([g]))
                    txl.assign(pbeg, pend)
                    txl.assign(pend, pend + udiv(gend - gbeg + 255, 256))
                m0 = txl.local_scalar("int32", init=gbeg + (pp - pbeg) * 256 + rank * BM)
                body(m0, col0, g, m0 < gend, hf)
                txl.assign(t, t + NUM_SMS // 2)

        with r_tma:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                def body(m0, col0, g, valid, hf):
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        bar = txl.cuda.cvta_generic_to_shared(full0.ptr_to([st.stage]))
                        with txl.if_(rank == 0), txl.then_():
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
                        with txl.if_(hf == 0):
                            with txl.then_():
                                txl.ptx[LD3](
                                    s_b.ptr_to([st.stage * B_HALF]),
                                    txl.address_of(b_map),
                                    txl.cast(k * BK, "int32"),
                                    col0 + rank * (BN // 2),
                                    g,
                                    bar,
                                )
                            with txl.else_():
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
            ss = txl.PipelineState(STAGES, 1)
            ra = txl.alloc_local((16,), "uint32")
            # The tile's B rows [nb*BN, nb*BN + BN) span at most 2 (BN <= 128: 1, 192: 2, 256: 2)
            # 128-row weight-scale blocks; 32-row groups never straddle a block (BN % 64 == 0).
            NBLK = 1 if BN <= 128 else 2
            rb = txl.alloc_local((4 * NBLK,), "uint32")
            fv = txl.local_scalar("float32")
            n_sfb = (N + 127) // 128

            def load_group(m0, col0, g, kb):
                if VSF:
                    for i in range(4):
                        row = txl.min(m0 + lane + 32 * i, M - 1)
                        txl.ptx.ld.global_.nc.v4.b32(
                            ra[i * 4], ra[i * 4 + 1], ra[i * 4 + 2], ra[i * 4 + 3], sfa.ptr_to([row, kb])
                        )
                    for c in range(NBLK):
                        nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                        txl.ptx.ld.global_.nc.v4.b32(
                            rb[c * 4], rb[c * 4 + 1], rb[c * 4 + 2], rb[c * 4 + 3], sfb.ptr_to([g, nsf, kb])
                        )
                else:
                    for j in range(4):
                        kk = txl.min(kb + j, kb_sf - 1)
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(fv, sfa.ptr_to([txl.min(m0 + lane + 32 * i, M - 1), kk]))
                            txl.assign(ra[i * 4 + j], txl.reinterpret(fv, ty="uint32"))
                        for c in range(NBLK):
                            nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                            txl.ptx.ld.global_.nc.f32(fv, sfb.ptr_to([g, nsf, kk]))
                            txl.assign(rb[c * 4 + j], txl.reinterpret(fv, ty="uint32"))

            def body(m0, col0, g, valid, hf):
                load_group(m0, col0, g, 0)
                with txl.serial(nk) as k:
                    with txl.if_(umod(k, 4) == 0), txl.then_():
                        # Scale slots advance only when scales change. Reuse is protected by the data
                        # stage at k: after STAGES scale groups, k advanced 4*STAGES and is back on the
                        # same data stage, whose empty barrier includes the prior tcgen05.cp.
                        twait("sf-wait-empty", empty, st.stage, st.phase)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.st.shared.v4.u32(
                            s_sfa.ptr_to([ss.stage * 128 + lane * 4]),
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
                                s_sfb.ptr_to([(ss.stage * N_SF_CHUNKS + c) * 128 + lane * 4]), *ws
                            )
                        with txl.if_(k + 4 < nk), txl.then_():
                            load_group(m0, col0, g, k + 4)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        sf_full0.arrive(ss.stage)
                        ss.advance()
                    st.advance()

            tiles(body)

        with r_mma:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    st = txl.PipelineState(STAGES, 0)
                    ss = txl.PipelineState(STAGES, 0)
                    acc = txl.PipelineState(ACC_STAGES, 0)
                    q = txl.local_scalar("int32", init=0)  # tiles issued
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
                        if SFT:
                            sfa_col = txl.local_scalar("uint32", init=u322((1 - acc.stage) * 256 + 240))
                            with txl.if_(q > 0), txl.then_():
                                # the previous tile (other accumulator) has had its tail chunk read
                                twait("mma-wait-tail", tail_free, 1 - acc.stage, u322(umod(udiv(q - 1, 2), 2)))
                        else:
                            sfa_col = txl.uint32(SFA_COL)
                        tile_tok = rng("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(nk) as k:
                            d0 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, a_base + u322(st.stage * A_BYTES)))
                            d1 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, b_base + u322(st.stage * B_HALF)))
                            idesc = txl.local_scalar(
                                "uint32",
                                init=with_sf_id(
                                    txl.if_then_else(hf == 1, txl.uint32(INSTR_H), txl.uint32(INSTR)), umod(k, 4)
                                ),
                            )
                            sd0 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfa_base + u322(ss.stage * 512)))
                            sd1 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfb_base + u322(ss.stage * N_SF_CHUNKS * 512)))
                            twait("mma-wait-tma", full, st.stage, st.phase)
                            with txl.if_(umod(k, 4) == 0), txl.then_():
                                twait("mma-wait-sf", sf_full, ss.stage, ss.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.if_(umod(k, 4) == 0), txl.then_():
                                txl.ptx[UTCCP2](
                                    sfa_col,
                                    sd0,
                                )
                                for c in range(N_SF_CHUNKS):
                                    txl.ptx[UTCCP2](
                                        sfa_col + txl.uint32(4 + 4 * c),
                                        dadd(sd1, c * 512),
                                    )
                                ss.advance()
                            for ki in range(BK // UMMA_K):
                                txl.ptx[MMA2](
                                    u322(acc.stage * BN),
                                    dadd(d0, ki * UMMA_K),
                                    dadd(d1, ki * UMMA_K),
                                    idesc,
                                    sfa_col,
                                    sfa_col + txl.uint32(4),
                                    txl.Or(k > 0, ki > 0) if ki == 0 else txl.bool(True),
                                )
                            empty.arrive(st.stage, cta_group=2, cta_mask=3)
                            st.advance()
                        t_full.arrive(acc.stage, cta_group=2, cta_mask=3)
                        end(tile_tok)
                        acc.advance()
                        txl.assign(q, q + 1)

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
                # Drain this warpgroup's 128 accumulator columns into packed bf16 registers (SFT: the tail
                # chunk first; warpgroup 1 then frees it for the next tile's scale columns).
                nch = COLS_PER_WG // 32
                order = [nch - 1] + list(range(nch - 1)) if SFT else list(range(nch))
                for i, c in enumerate(order):
                    taddr = txl.bitwise_or(
                        txl.shift_left(u322(quad * 32), txl.uint32(16)),
                        u322(acc.stage * BN + wg * COLS_PER_WG + c * 32),
                    )
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[vals[j] for j in range(32)], taddr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    if SFT and i == 0:
                        with txl.if_(wg == 1), txl.then_():
                            txl.ptx.tcgen05.fence__before_thread_sync()
                            tail_free.arrive(acc.stage, remote=0)
                    for p in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(
                            packed[c * 16 + p],
                            txl.reinterpret(vals[2 * p + 1], ty="float32"),
                            txl.reinterpret(vals[2 * p], ty="float32"),
                        )
                txl.ptx.tcgen05.fence__before_thread_sync()
                t_empty.arrive(acc.stage, remote=0)
                end(epi_tok)
                st_tok = rng("epi-store")
                for c in range(COLS_PER_WG // 32):
                    buf = (wg * CDB + c % CDB) * (BM * 16)
                    with txl.if_(wtid == 0), txl.then_():
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
                    with txl.if_(txl.And(txl.And(wtid == 0, valid), txl.Or(hf == 0, wg == 0))), txl.then_():
                        txl.ptx[STORE2D](
                            txl.address_of(d_map),
                            col0 + wg * COLS_PER_WG + c * 32,
                            m0,
                            s_cd.ptr_to([buf]),
                        )
                    with txl.if_(wtid == 0), txl.then_():
                        txl.ptx.cp.async_.bulk.commit_group()
                end(st_tok)
                acc.advance()

            tiles(body)
            with txl.if_(wtid == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with r_idle:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped


def build_mgrouped_w320(STAGES, VSF=False):
    """
    BN=320 variant of the pair kernel (n % 320 == 0, e.g. gpt-oss 2880 = 9 x 320): each tile is two
    N=160 sub-MMAs (D cols 0 and 160). CTA r loads B rows [80 r, +80) and [160 + 80 r, +80) of the tile so
    both sub-MMAs land in natural column order; sub-MMA j reads its scales from its own SFB chunks.
    320 accumulator + 20 scale columns: single accumulator.

    Both 128-row halves of a pair belong to one group (pairs are formed per group); the odd last
    block of a group is computed by the pair's second CTA on rows it then does not store.
    """
    BN = 320
    CDB = 1
    SFT = False
    A_BYTES = BM * BK
    B_HALF = (BN // 2) * BK
    B_SUB = 80 * BK  # one 80-row B box
    ACC_STAGES = 1
    N_SF_CHUNKS = 4  # SFB chunks: tile rows [0,128), [128,256) for sub-MMA 0; [160,288), [288,416) for sub-MMA 1
    CHUNK_BASE = (0, 128, 160, 288)
    SFA_COL = 320
    EPI_WGS = 2 if BN >= 192 else 1
    EPI_WARPS = 4 * EPI_WGS
    COLS_PER_WG = BN // EPI_WGS
    assert COLS_PER_WG % 32 == 0 and (COLS_PER_WG // 32) % CDB == 0
    INSTR = encode_instr_descriptor_block_scaled_uint32(
        M=2 * BM, N=160, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2,
    )  # fmt: skip
    # Final-wave split (BN=256): the last partial wave's tiles run as two 128-column halves when
    # those halves fit in one wave, halving the makespan tail.
    SPLIT = False
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
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, L2P, 0)
        txl.cu_tensor_map_encode_tiled(
            b_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, 80, 1,
            1, 1, 1, 0, 3, L2P, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            bh_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 4, 1,
            1, 1, 1, 0, 3, L2P, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            d_map, "bfloat16", 2, p["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0
        )
        return a_map, b_map, bh_map, d_map

    @txl.kernel()
    def mgrouped_w320(
        a: txl.Tensor((txl.int64(), txl.int64()), FP8),
        sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32),
        b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8),
        sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32),
        d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16),
        offs: txl.Tensor((txl.int64(),), txl.i32),
        M: txl.int32,
        N: txl.int32,
        K: txl.int32,
        G: txl.int32,
    ):
        host = prelude(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=(4 + EPI_WARPS) * 32, cluster=[2]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, bh_map, d_map = host
        rank = txl.cuda.cluster_cta_id("x")
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
        tail_free = txl.MBarrier(smem, 2)
        tail_free.init(2 * 32 * 4)  # warpgroup 1 (columns 128-255) of both CTAs
        with txl.if_(tid == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
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
            # one redux.sync per value: the six walk reductions are independent (no 5-deep shuffle chains)
            txl.ptx.redux_sync.add.u32(v, v, txl.uint32(0xFFFFFFFF))

        with txl.serial((G + 31) // 32) as wc:
            gi = txl.local_scalar("int32", init=wc * 32 + lane)
            with txl.if_(wc == 0):
                with txl.then_():
                    txl.assign(cur, cur0)
                    txl.assign(prev, prev0)
                with txl.else_():
                    txl.ptx.ld.global_.s32(cur, offs.ptr_to([txl.min(gi, G - 1)]))
                    txl.ptx.ld.global_.s32(prev, offs.ptr_to([txl.max(txl.min(gi, G - 1) - 1, 0)]))
            with txl.if_(gi == 0), txl.then_():
                txl.assign(prev, 0)
            cnt = txl.local_scalar("uint32", init=txl.if_then_else(gi < G, u322(udiv(cur - prev + 255, 256)), txl.uint32(0)))
            incl = txl.local_scalar("uint32", init=cnt)
            for x in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(x), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.if_(lane >= x), txl.then_():
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
        # the group walk (registers + global loads only) overlaps the cluster barrier latency
        txl.ptx.barrier.cluster.wait.acquire.aligned()

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
            with txl.while_(t < full_end + 2 * tail):
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
                with txl.if_(pp < pbeg), txl.then_():
                    txl.assign(g, -1)
                    txl.assign(gbeg, 0)
                    txl.assign(gend, 0)
                    txl.assign(pbeg, 0)
                    txl.assign(pend, 0)
                with txl.while_(pp >= pend):
                    txl.assign(g, g + 1)
                    txl.assign(gbeg, gend)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([g]))
                    txl.assign(pbeg, pend)
                    txl.assign(pend, pend + udiv(gend - gbeg + 255, 256))
                m0 = txl.local_scalar("int32", init=gbeg + (pp - pbeg) * 256 + rank * BM)
                body(m0, col0, g, m0 < gend, hf)
                txl.assign(t, t + NUM_SMS // 2)

        with r_tma:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                def body(m0, col0, g, valid, hf):
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        bar = txl.cuda.cvta_generic_to_shared(full0.ptr_to([st.stage]))
                        with txl.if_(rank == 0), txl.then_():
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
                        for j in range(2):
                            txl.ptx[LD3](
                                s_b.ptr_to([st.stage * B_HALF + j * B_SUB]),
                                txl.address_of(b_map),
                                txl.cast(k * BK, "int32"),
                                col0 + j * 160 + rank * 80,
                                g,
                                bar,
                            )
                        st.advance()

                tiles(body)

        with r_sf:
            st = txl.PipelineState(STAGES, 1)
            ss = txl.PipelineState(STAGES, 1)
            ra = txl.alloc_local((16,), "uint32")
            # The tile's B rows [nb*BN, nb*BN + BN) span at most 2 (BN <= 128: 1, 192: 2, 256: 2)
            # 128-row weight-scale blocks; 32-row groups never straddle a block (BN % 64 == 0).
            NBLK = 3  # a 320-row tile starting at a multiple of 64 spans <= 3 weight-scale blocks
            rb = txl.alloc_local((4 * NBLK,), "uint32")
            fv = txl.local_scalar("float32")
            n_sfb = (N + 127) // 128

            def load_group(m0, col0, g, kb):
                if VSF:
                    for i in range(4):
                        row = txl.min(m0 + lane + 32 * i, M - 1)
                        txl.ptx.ld.global_.nc.v4.b32(
                            ra[i * 4], ra[i * 4 + 1], ra[i * 4 + 2], ra[i * 4 + 3], sfa.ptr_to([row, kb])
                        )
                    for c in range(NBLK):
                        nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                        txl.ptx.ld.global_.nc.v4.b32(
                            rb[c * 4], rb[c * 4 + 1], rb[c * 4 + 2], rb[c * 4 + 3], sfb.ptr_to([g, nsf, kb])
                        )
                else:
                    for j in range(4):
                        kk = txl.min(kb + j, kb_sf - 1)
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(fv, sfa.ptr_to([txl.min(m0 + lane + 32 * i, M - 1), kk]))
                            txl.assign(ra[i * 4 + j], txl.reinterpret(fv, ty="uint32"))
                        for c in range(NBLK):
                            nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                            txl.ptx.ld.global_.nc.f32(fv, sfb.ptr_to([g, nsf, kk]))
                            txl.assign(rb[c * 4 + j], txl.reinterpret(fv, ty="uint32"))

            def body(m0, col0, g, valid, hf):
                load_group(m0, col0, g, 0)
                with txl.serial(nk) as k:
                    with txl.if_(umod(k, 4) == 0), txl.then_():
                        # Scale slots advance only when scales change. Reuse is protected by the data
                        # stage at k: after STAGES scale groups, k advanced 4*STAGES and is back on the
                        # same data stage, whose empty barrier includes the prior tcgen05.cp.
                        twait("sf-wait-empty", empty, st.stage, st.phase)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.st.shared.v4.u32(
                            s_sfa.ptr_to([ss.stage * 128 + lane * 4]),
                            pack4(ra, 0), pack4(ra, 4), pack4(ra, 8), pack4(ra, 12),
                        )  # fmt: skip
                        wb = [txl.local_scalar("uint32", init=pack4(rb, 4 * j)) for j in range(NBLK)]
                        boff = umod(col0, 128)  # tile start within its first scale block
                        for c in range(N_SF_CHUNKS):
                            # word i of chunk c holds B rows CHUNK_BASE[c] + 32*i + lane of the tile
                            ws = []
                            for i in range(4):
                                o = boff + (CHUNK_BASE[c] + 32 * i)
                                ws.append(txl.if_then_else(o >= 256, wb[2], txl.if_then_else(o >= 128, wb[1], wb[0])))
                            txl.ptx.st.shared.v4.u32(
                                s_sfb.ptr_to([(ss.stage * N_SF_CHUNKS + c) * 128 + lane * 4]), *ws
                            )
                        with txl.if_(k + 4 < nk), txl.then_():
                            load_group(m0, col0, g, k + 4)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        sf_full0.arrive(ss.stage)
                        ss.advance()
                    st.advance()

            tiles(body)

        with r_mma:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    st = txl.PipelineState(STAGES, 0)
                    ss = txl.PipelineState(STAGES, 0)
                    acc = txl.PipelineState(ACC_STAGES, 0)
                    q = txl.local_scalar("int32", init=0)  # tiles issued
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
                        if SFT:
                            sfa_col = txl.local_scalar("uint32", init=u322((1 - acc.stage) * 256 + 240))
                            with txl.if_(q > 0), txl.then_():
                                # the previous tile (other accumulator) has had its tail chunk read
                                twait("mma-wait-tail", tail_free, 1 - acc.stage, u322(umod(udiv(q - 1, 2), 2)))
                        else:
                            sfa_col = txl.uint32(SFA_COL)
                        tile_tok = rng("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(nk) as k:
                            d0 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, a_base + u322(st.stage * A_BYTES)))
                            d1 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, b_base + u322(st.stage * B_HALF)))
                            idesc = txl.local_scalar(
                                "uint32",
                                init=with_sf_id(
                                    txl.if_then_else(hf == 1, txl.uint32(INSTR_H), txl.uint32(INSTR)), umod(k, 4)
                                ),
                            )
                            sd0 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfa_base + u322(ss.stage * 512)))
                            sd1 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfb_base + u322(ss.stage * N_SF_CHUNKS * 512)))
                            twait("mma-wait-tma", full, st.stage, st.phase)
                            with txl.if_(umod(k, 4) == 0), txl.then_():
                                twait("mma-wait-sf", sf_full, ss.stage, ss.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.if_(umod(k, 4) == 0), txl.then_():
                                txl.ptx[UTCCP2](
                                    sfa_col,
                                    sd0,
                                )
                                for c in range(N_SF_CHUNKS):
                                    txl.ptx[UTCCP2](
                                        sfa_col + txl.uint32(4 + 4 * c),
                                        dadd(sd1, c * 512),
                                    )
                                ss.advance()
                            for ki in range(BK // UMMA_K):
                                for j in range(2):
                                    txl.ptx[MMA2](
                                        txl.uint32(160 * j),
                                        dadd(d0, ki * UMMA_K),
                                        dadd(d1, j * B_SUB + ki * UMMA_K),
                                        idesc,
                                        sfa_col,
                                        sfa_col + txl.uint32(4 + 8 * j),
                                        txl.Or(k > 0, ki > 0) if ki == 0 else txl.bool(True),
                                    )
                            empty.arrive(st.stage, cta_group=2, cta_mask=3)
                            st.advance()
                        t_full.arrive(acc.stage, cta_group=2, cta_mask=3)
                        end(tile_tok)
                        acc.advance()
                        txl.assign(q, q + 1)

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
                # Drain this warpgroup's 128 accumulator columns into packed bf16 registers (SFT: the tail
                # chunk first; warpgroup 1 then frees it for the next tile's scale columns).
                nch = COLS_PER_WG // 32
                order = [nch - 1] + list(range(nch - 1)) if SFT else list(range(nch))
                for i, c in enumerate(order):
                    taddr = txl.bitwise_or(
                        txl.shift_left(u322(quad * 32), txl.uint32(16)),
                        u322(acc.stage * BN + wg * COLS_PER_WG + c * 32),
                    )
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[vals[j] for j in range(32)], taddr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    if SFT and i == 0:
                        with txl.if_(wg == 1), txl.then_():
                            txl.ptx.tcgen05.fence__before_thread_sync()
                            tail_free.arrive(acc.stage, remote=0)
                    for p in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(
                            packed[c * 16 + p],
                            txl.reinterpret(vals[2 * p + 1], ty="float32"),
                            txl.reinterpret(vals[2 * p], ty="float32"),
                        )
                txl.ptx.tcgen05.fence__before_thread_sync()
                t_empty.arrive(acc.stage, remote=0)
                end(epi_tok)
                st_tok = rng("epi-store")
                for c in range(COLS_PER_WG // 32):
                    buf = (wg * CDB + c % CDB) * (BM * 16)
                    with txl.if_(wtid == 0), txl.then_():
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
                    with txl.if_(txl.And(txl.And(wtid == 0, valid), txl.Or(hf == 0, wg == 0))), txl.then_():
                        txl.ptx[STORE2D](
                            txl.address_of(d_map),
                            col0 + wg * COLS_PER_WG + c * 32,
                            m0,
                            s_cd.ptr_to([buf]),
                        )
                    with txl.if_(wtid == 0), txl.then_():
                        txl.ptx.cp.async_.bulk.commit_group()
                end(st_tok)
                acc.advance()

            tiles(body)
            with txl.if_(wtid == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with r_idle:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped_w320


def build_mgrouped_w320r(STAGES, VSF=False, DEFER=2, SW=160, CDB=1, LASTCD=False):
    assert 1 <= DEFER <= 4  # deferred sub-MMA 1 steps must use the scale group copied at k = 0
    """
    BN=320 variant of the pair kernel (n % 320 == 0, e.g. gpt-oss 2880 = 9 x 320): each tile is two
    N=160 sub-MMAs (D cols 0 and 160). CTA r loads B rows [80 r, +80) and [160 + 80 r, +80) of the tile so
    both sub-MMAs land in natural column order; sub-MMA j reads its scales from its own SFB chunks.
    Rotating accumulator slots: TMEM holds three 160-column slots (cols 0/160/320) + 20 scale columns at 480.
    A tile uses (s0, s1); the next tile uses (free slot, s0). Its sub-MMA 0 starts at once in the free slot;
    its first DEFER K steps' sub-MMA 1 work is deferred (those stages stay held) until the epilogue frees s0,
    so the tensor pipe keeps working through the previous tile's commit -> drain round trip.

    Both 128-row halves of a pair belong to one group (pairs are formed per group); the odd last
    block of a group is computed by the pair's second CTA on rows it then does not store.
    """
    BN = 2 * SW  # two sub-MMAs of width SW (160: gpt-oss n; 128: 256-wide tiles)
    SCH = (SW + 127) // 128  # SFB chunks per sub-MMA
    SFT = False
    A_BYTES = BM * BK
    B_HALF = (BN // 2) * BK
    B_SUB = (SW // 2) * BK  # one B box (SW / 2 rows per CTA per sub-MMA)
    N_SF_CHUNKS = 2 * SCH  # SFB chunk c covers tile rows [SW * (c // SCH) + 128 * (c % SCH), +128)
    CHUNK_BASE = [SW * (c // SCH) + 128 * (c % SCH) for c in range(N_SF_CHUNKS)]
    SFA_COL = 3 * SW
    assert SFA_COL + 4 + 4 * N_SF_CHUNKS <= 512
    EPI_WGS = 2 if BN >= 192 else 1
    EPI_WARPS = 4 * EPI_WGS
    COLS_PER_WG = BN // EPI_WGS
    assert COLS_PER_WG % 32 == 0  # store buffers rotate on a running chunk count (any CDB)
    INSTR = encode_instr_descriptor_block_scaled_uint32(
        M=2 * BM, N=SW, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2,
    )  # fmt: skip
    # Final-wave split (BN=256): the last partial wave's tiles run as two 128-column halves when
    # those halves fit in one wave, halving the makespan tail.
    SPLIT = False
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
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, L2P, 0)
        txl.cu_tensor_map_encode_tiled(
            b_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, SW // 2, 1,
            1, 1, 1, 0, 3, L2P, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            bh_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 4, 1,
            1, 1, 1, 0, 3, L2P, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            d_map, "bfloat16", 2, p["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0
        )
        return a_map, b_map, bh_map, d_map

    @txl.kernel()
    def mgrouped_w320r(
        a: txl.Tensor((txl.int64(), txl.int64()), FP8),
        sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32),
        b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8),
        sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32),
        d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16),
        offs: txl.Tensor((txl.int64(),), txl.i32),
        M: txl.int32,
        N: txl.int32,
        K: txl.int32,
        G: txl.int32,
    ):
        host = prelude(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=(4 + EPI_WARPS) * 32, cluster=[2]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, bh_map, d_map = host
        rank = txl.cuda.cluster_cta_id("x")
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
        s_cd = smem.alloc((EPI_WGS * (2 if LASTCD else CDB) * BM * 16,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * N_SF_CHUNKS * 128,), "uint32", align=16)
        full = txl.TMABar(smem, STAGES)
        empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        t_full = txl.TCGen05Bar(smem, 2)
        slot_free = txl.MBarrier(smem, 3)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        full.init(1)
        empty.init(1)
        sf_full.init(64)
        t_full.init(1)
        slot_free.init(2 * 128)  # one epilogue warpgroup of each CTA drains a slot
        tail_free = txl.MBarrier(smem, 2)
        tail_free.init(2 * 32 * 4)  # warpgroup 1 (columns 128-255) of both CTAs
        with txl.if_(tid == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
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
            # one redux.sync per value: the six walk reductions are independent (no 5-deep shuffle chains)
            txl.ptx.redux_sync.add.u32(v, v, txl.uint32(0xFFFFFFFF))

        with txl.serial((G + 31) // 32) as wc:
            gi = txl.local_scalar("int32", init=wc * 32 + lane)
            with txl.if_(wc == 0):
                with txl.then_():
                    txl.assign(cur, cur0)
                    txl.assign(prev, prev0)
                with txl.else_():
                    txl.ptx.ld.global_.s32(cur, offs.ptr_to([txl.min(gi, G - 1)]))
                    txl.ptx.ld.global_.s32(prev, offs.ptr_to([txl.max(txl.min(gi, G - 1) - 1, 0)]))
            with txl.if_(gi == 0), txl.then_():
                txl.assign(prev, 0)
            cnt = txl.local_scalar("uint32", init=txl.if_then_else(gi < G, u322(udiv(cur - prev + 255, 256)), txl.uint32(0)))
            incl = txl.local_scalar("uint32", init=cnt)
            for x in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(x), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.if_(lane >= x), txl.then_():
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
        # the group walk (registers + global loads only) overlaps the cluster barrier latency
        txl.ptx.barrier.cluster.wait.acquire.aligned()

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
            with txl.while_(t < full_end + 2 * tail):
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
                with txl.if_(pp < pbeg), txl.then_():
                    txl.assign(g, -1)
                    txl.assign(gbeg, 0)
                    txl.assign(gend, 0)
                    txl.assign(pbeg, 0)
                    txl.assign(pend, 0)
                with txl.while_(pp >= pend):
                    txl.assign(g, g + 1)
                    txl.assign(gbeg, gend)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([g]))
                    txl.assign(pbeg, pend)
                    txl.assign(pend, pend + udiv(gend - gbeg + 255, 256))
                m0 = txl.local_scalar("int32", init=gbeg + (pp - pbeg) * 256 + rank * BM)
                body(m0, col0, g, m0 < gend, hf, t + NUM_SMS // 2 >= full_end + 2 * tail)
                txl.assign(t, t + NUM_SMS // 2)

        with r_tma:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                def body(m0, col0, g, valid, hf, last=None):
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        bar = txl.cuda.cvta_generic_to_shared(full0.ptr_to([st.stage]))
                        with txl.if_(rank == 0), txl.then_():
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
                        for j in range(2):
                            txl.ptx[LD3](
                                s_b.ptr_to([st.stage * B_HALF + j * B_SUB]),
                                txl.address_of(b_map),
                                txl.cast(k * BK, "int32"),
                                col0 + j * SW + rank * (SW // 2),
                                g,
                                bar,
                            )
                        st.advance()

                tiles(body)

        with r_sf:
            st = txl.PipelineState(STAGES, 1)
            ss = txl.PipelineState(STAGES, 1)
            ra = txl.alloc_local((16,), "uint32")
            # The tile's B rows [nb*BN, nb*BN + BN) span at most 2 (BN <= 128: 1, 192: 2, 256: 2)
            # 128-row weight-scale blocks; 32-row groups never straddle a block (BN % 64 == 0).
            NBLK = (BN + 64 + 127) // 128  # a tile starting at a multiple of 64 within a weight-scale block
            rb = txl.alloc_local((4 * NBLK,), "uint32")
            fv = txl.local_scalar("float32")
            n_sfb = (N + 127) // 128

            def load_group(m0, col0, g, kb):
                if VSF:
                    for i in range(4):
                        row = txl.min(m0 + lane + 32 * i, M - 1)
                        txl.ptx.ld.global_.nc.v4.b32(
                            ra[i * 4], ra[i * 4 + 1], ra[i * 4 + 2], ra[i * 4 + 3], sfa.ptr_to([row, kb])
                        )
                    for c in range(NBLK):
                        nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                        txl.ptx.ld.global_.nc.v4.b32(
                            rb[c * 4], rb[c * 4 + 1], rb[c * 4 + 2], rb[c * 4 + 3], sfb.ptr_to([g, nsf, kb])
                        )
                else:
                    for j in range(4):
                        kk = txl.min(kb + j, kb_sf - 1)
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(fv, sfa.ptr_to([txl.min(m0 + lane + 32 * i, M - 1), kk]))
                            txl.assign(ra[i * 4 + j], txl.reinterpret(fv, ty="uint32"))
                        for c in range(NBLK):
                            nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                            txl.ptx.ld.global_.nc.f32(fv, sfb.ptr_to([g, nsf, kk]))
                            txl.assign(rb[c * 4 + j], txl.reinterpret(fv, ty="uint32"))

            def body(m0, col0, g, valid, hf, last=None):
                load_group(m0, col0, g, 0)
                with txl.serial(nk) as k:
                    with txl.if_(umod(k, 4) == 0), txl.then_():
                        # Scale slots advance only when scales change. Reuse is protected by the data
                        # stage at k: after STAGES scale groups, k advanced 4*STAGES and is back on the
                        # same data stage, whose empty barrier includes the prior tcgen05.cp.
                        twait("sf-wait-empty", empty, st.stage, st.phase)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.st.shared.v4.u32(
                            s_sfa.ptr_to([ss.stage * 128 + lane * 4]),
                            pack4(ra, 0), pack4(ra, 4), pack4(ra, 8), pack4(ra, 12),
                        )  # fmt: skip
                        wb = [txl.local_scalar("uint32", init=pack4(rb, 4 * j)) for j in range(NBLK)]
                        boff = umod(col0, 128)  # tile start within its first scale block
                        for c in range(N_SF_CHUNKS):
                            # word i of chunk c holds B rows CHUNK_BASE[c] + 32*i + lane of the tile
                            ws = []
                            for i in range(4):
                                o = boff + (CHUNK_BASE[c] + 32 * i)
                                ws.append(txl.if_then_else(o >= 256, wb[2], txl.if_then_else(o >= 128, wb[1], wb[0])))
                            txl.ptx.st.shared.v4.u32(
                                s_sfb.ptr_to([(ss.stage * N_SF_CHUNKS + c) * 128 + lane * 4]), *ws
                            )
                        with txl.if_(k + 4 < nk), txl.then_():
                            load_group(m0, col0, g, k + 4)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        sf_full0.arrive(ss.stage)
                        ss.advance()
                    st.advance()

            tiles(body)

        with r_mma:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    st = txl.PipelineState(STAGES, 0)
                    ss = txl.PipelineState(STAGES, 0)
                    acc = txl.PipelineState(2, 0)
                    a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                    b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                    sfa_base = txl.local_scalar(
                        "uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0]))
                    )
                    sfb_base = txl.local_scalar(
                        "uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0]))
                    )

                    s0 = txl.local_scalar("int32", init=0)
                    s1 = txl.local_scalar("int32", init=1)
                    sph = txl.local_scalar("int32", init=0)  # bit s: slot_free phase of slot s
                    sfa_col = txl.uint32(SFA_COL)

                    def descs(stage, k):
                        d0 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, a_base + u322(stage * A_BYTES)))
                        d1 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, b_base + u322(stage * B_HALF)))
                        idesc = txl.local_scalar("uint32", init=with_sf_id(txl.uint32(INSTR), umod(k, 4)))
                        return d0, d1, idesc

                    def sub_mma(j, col, d0, d1, idesc, k):
                        for ki in range(BK // UMMA_K):
                            txl.ptx[MMA2](
                                col,
                                dadd(d0, ki * UMMA_K),
                                dadd(d1, j * B_SUB + ki * UMMA_K),
                                idesc,
                                sfa_col,
                                sfa_col + txl.uint32(4 + 4 * SCH * j),
                                txl.Or(k > 0, ki > 0) if ki == 0 else txl.bool(True),
                            )

                    def wait_slot(s):
                        twait("mma-wait-acc", slot_free, s, txl.bitwise_and(txl.shift_right(sph, s), 1) ^ 1)

                    def load_step(k):
                        """
                        Wait for stage st (K step k); copy the scale group at k % 4 == 0.
                        """
                        sd0 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfa_base + u322(ss.stage * 512)))
                        sd1 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfb_base + u322(ss.stage * N_SF_CHUNKS * 512)))
                        twait("mma-wait-tma", full, st.stage, st.phase)
                        with txl.if_(umod(k, 4) == 0), txl.then_():
                            twait("mma-wait-sf", sf_full, ss.stage, ss.phase)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.if_(umod(k, 4) == 0), txl.then_():
                            txl.ptx[UTCCP2](sfa_col, sd0)
                            for c in range(N_SF_CHUNKS):
                                txl.ptx[UTCCP2](sfa_col + txl.uint32(4 + 4 * c), dadd(sd1, c * 512))
                            ss.advance()

                    def body(m0, col0, g, valid, hf, last=None):
                        c0 = txl.local_scalar("uint32", init=u322(s0 * SW))
                        c1 = txl.local_scalar("uint32", init=u322(s1 * SW))
                        wait_slot(s0)  # the free slot: drained a tile ago
                        tile_tok = rng("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        nd = txl.local_scalar("int32", init=txl.min(nk, DEFER))
                        st0 = txl.local_scalar("int32", init=st.stage)
                        # first nd K steps: sub-MMA 0 only (stages stay held)
                        with txl.serial(nd) as k:
                            d0, d1, idesc = descs(st.stage, k)
                            load_step(k)
                            sub_mma(0, c0, d0, d1, idesc, k)
                            st.advance()
                        # the previous tile's first slot is drained: catch up sub-MMA 1, release those stages
                        wait_slot(s1)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(nd) as k:
                            stg = txl.local_scalar("int32", init=umod(st0 + k, STAGES))
                            d0, d1, idesc = descs(stg, k)
                            sub_mma(1, c1, d0, d1, idesc, k)
                            empty.arrive(stg, cta_group=2, cta_mask=3)
                        with txl.serial(nk - nd) as kk:
                            k = kk + nd
                            d0, d1, idesc = descs(st.stage, k)
                            load_step(k)
                            sub_mma(0, c0, d0, d1, idesc, k)
                            sub_mma(1, c1, d0, d1, idesc, k)
                            empty.arrive(st.stage, cta_group=2, cta_mask=3)
                            st.advance()
                        t_full.arrive(acc.stage, cta_group=2, cta_mask=3)
                        end(tile_tok)
                        acc.advance()
                        txl.assign(sph, txl.bitwise_xor(sph, txl.bitwise_or(txl.shift_left(txl.int32(1), s0), txl.shift_left(txl.int32(1), s1))))
                        nxt = txl.local_scalar("int32", init=3 - s0 - s1)
                        txl.assign(s1, s0)
                        txl.assign(s0, nxt)

                    tiles(body)

        with r_epi:
            ew = warp - 4
            quad = warp % 4
            wg = ew // 4
            wtid = (ew % 4) * 32 + lane  # thread id within this warpgroup
            row = quad * 32 + lane
            acc = txl.PipelineState(2, 0)
            e0 = txl.local_scalar("int32", init=0)
            cc = txl.local_scalar("int32", init=0)  # store chunks issued by this warpgroup (buffer rotation)
            e1 = txl.local_scalar("int32", init=1)
            vals = txl.alloc_local((32,), "uint32")
            packed = txl.alloc_local((COLS_PER_WG // 2,), "uint32")

            def body(m0, col0, g, valid, hf, last=None):
                twait("epi-wait-acc", t_full, acc.stage, acc.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                epi_tok = rng("epi-drain")
                myslot = txl.local_scalar("int32", init=txl.Select(wg == 0, e0, e1))
                # Drain this warpgroup's 128 accumulator columns into packed bf16 registers (SFT: the tail
                # chunk first; warpgroup 1 then frees it for the next tile's scale columns).
                nch = COLS_PER_WG // 32
                order = [nch - 1] + list(range(nch - 1)) if SFT else list(range(nch))
                for i, c in enumerate(order):
                    taddr = txl.bitwise_or(
                        txl.shift_left(u322(quad * 32), txl.uint32(16)),
                        u322(myslot * SW + c * 32),
                    )
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[vals[j] for j in range(32)], taddr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    if SFT and i == 0:
                        with txl.if_(wg == 1), txl.then_():
                            txl.ptx.tcgen05.fence__before_thread_sync()
                            tail_free.arrive(acc.stage, remote=0)
                    for p in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(
                            packed[c * 16 + p],
                            txl.reinterpret(vals[2 * p + 1], ty="float32"),
                            txl.reinterpret(vals[2 * p], ty="float32"),
                        )
                txl.ptx.tcgen05.fence__before_thread_sync()
                slot_free.arrive(myslot, remote=0)
                end(epi_tok)
                enx = txl.local_scalar("int32", init=3 - e0 - e1)
                txl.assign(e1, e0)
                txl.assign(e0, enx)
                st_tok = rng("epi-store")
                for c in range(COLS_PER_WG // 32):
                    if LASTCD:
                        # this CTA's last tile: no next tile's loads to compete with, so alternate two
                        # buffers (chunk c -> (c + 1) % 2: the previous tile's final store used buffer 0)
                        buf = (wg * 2 + txl.if_then_else(last, (c + 1) % 2, 0)) * (BM * 16)
                        with txl.if_(wtid == 0), txl.then_():
                            with txl.if_(last):
                                with txl.then_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(1)
                                with txl.else_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                    else:
                        buf = (wg * CDB + (umod(cc + c, CDB) if CDB > 1 else 0)) * (BM * 16)
                        with txl.if_(wtid == 0), txl.then_():
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
                    with txl.if_(txl.And(txl.And(wtid == 0, valid), txl.Or(hf == 0, wg == 0))), txl.then_():
                        txl.ptx[STORE2D](
                            txl.address_of(d_map),
                            col0 + wg * COLS_PER_WG + c * 32,
                            m0,
                            s_cd.ptr_to([buf]),
                        )
                    with txl.if_(wtid == 0), txl.then_():
                        txl.ptx.cp.async_.bulk.commit_group()
                end(st_tok)
                txl.assign(cc, cc + COLS_PER_WG // 32)
                acc.advance()

            tiles(body)
            with txl.if_(wtid == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with r_idle:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped_w320r


def build_mgrouped_1cta(STAGES, VSF=False):
    """
    1-CTA m-grouped kernel: a CTA computes 128 rows (one 128-row block of one group) x 256 columns.

    No pair padding: 128-row blocks tile the 128-padded groups exactly. One 256-column accumulator
    (+12 scale columns); two epilogue warpgroups each drain 128 columns to registers and release TMEM
    before packing and storing.
    """
    BN = 256
    A_BYTES = BM * BK
    B_BYTES = BN * BK
    N_SF_CHUNKS = BN // 128
    SFA_COL = BN
    SFB_COL = SFA_COL + 4
    INSTR = encode_instr_descriptor_block_scaled_uint32(
        M=BM, N=BN, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=1,
    )  # fmt: skip
    # Final-wave split: a last partial wave of <= half the CTAs runs its tiles as two 128-column halves.
    INSTR_H = encode_instr_descriptor_block_scaled_uint32(
        M=BM, N=BN // 2, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8,
        sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=1,
    )  # fmt: skip

    def prelude(p):
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = p["M"], p["N"], p["K"], p["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, L2P, 0)
        txl.cu_tensor_map_encode_tiled(
            b_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN, 1,
            1, 1, 1, 0, 3, L2P, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            d_map, "bfloat16", 2, p["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0
        )
        return a_map, b_map, d_map

    @txl.kernel()
    def mgrouped_1cta(
        a: txl.Tensor((txl.int64(), txl.int64()), FP8),
        sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32),
        b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8),
        sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32),
        d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16),
        offs: txl.Tensor((txl.int64(),), txl.i32),
        M: txl.int32,
        N: txl.int32,
        K: txl.int32,
        G: txl.int32,
    ):
        host = prelude(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=384), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, d_map = host
        cta = txl.cta_id()
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
        s_b = smem.alloc((STAGES * B_BYTES,), "uint8", align=1024)
        s_cd = smem.alloc((2 * BM * 16,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * BN,), "uint32", align=16)
        full = txl.TMABar(smem, STAGES)
        empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        t_full = txl.TCGen05Bar(smem, 1)
        t_empty = txl.MBarrier(smem, 1)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        full.init(1)
        empty.init(1)
        sf_full.init(32)
        t_full.init(1)
        t_empty.init(256)
        with txl.if_(tid == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
        txl.ptx.bar.sync(txl.uint32(0))

        n_nb = (N + (BN - 1)) // BN
        total = udiv(M, BM) * n_nb
        nk = (K + (BK - 1)) // BK
        kb_sf = nk

        def tiles(body):
            """
            Persistent walk: tile t -> (128-row block, n block); g follows the row block monotonically.
            """
            t = txl.local_scalar("int32", init=cta)
            g = txl.local_scalar("int32", init=0)
            gstart = txl.local_scalar("int32", init=0)  # first row of group g
            gend = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(gend, offs.ptr_to([0]))
            tail = umod(total, NUM_SMS)
            tail = txl.if_then_else(2 * tail <= NUM_SMS, tail, 0)
            full_end = total - tail  # tiles [full_end, total) run as two column halves each
            with txl.while_(t < full_end + 2 * tail):
                hf = txl.local_scalar("int32", init=txl.if_then_else(t >= full_end, 1, 0))
                tile = txl.local_scalar("int32", init=txl.if_then_else(hf == 1, full_end + udiv(t - full_end, 2), t))
                m0 = txl.local_scalar("int32", init=udiv(tile, n_nb) * BM)
                col0 = txl.local_scalar(
                    "int32", init=umod(tile, n_nb) * BN + txl.if_then_else(hf == 1, umod(t - full_end, 2) * (BN // 2), 0)
                )
                # half-tile ids can map below this CTA's previous tile (problem under one wave): restart the walk
                with txl.if_(m0 < gstart), txl.then_():
                    txl.assign(g, 0)
                    txl.assign(gstart, 0)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([0]))
                with txl.while_(m0 >= gend):
                    txl.assign(g, g + 1)
                    txl.assign(gstart, gend)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([g]))
                body(m0, col0, g, hf)
                txl.assign(t, t + NUM_SMS)

        def tiles_nowalk(body):
            """
            Same tile sequence without the group walk (for roles that only need the half-tile flag).
            """
            t = txl.local_scalar("int32", init=cta)
            tail = umod(total, NUM_SMS)
            tail = txl.if_then_else(2 * tail <= NUM_SMS, tail, 0)
            full_end = total - tail
            with txl.while_(t < full_end + 2 * tail):
                hf = txl.local_scalar("int32", init=txl.if_then_else(t >= full_end, 1, 0))
                body(0, 0, 0, hf)
                txl.assign(t, t + NUM_SMS)

        with r_tma:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                def body(m0, col0, g, hf):
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        full.arrive(st.stage, A_BYTES + B_BYTES)
                        txl.ptx[LOAD2D](
                            s_a.ptr_to([st.stage * A_BYTES]), txl.address_of(a_map), txl.cast(k * BK, "int32"), m0,
                            full.ptr_to([st.stage]), txl.uint64(EVICT_NORMAL),
                        )  # fmt: skip
                        txl.ptx[LOAD3D](
                            s_b.ptr_to([st.stage * B_BYTES]), txl.address_of(b_map), txl.cast(k * BK, "int32"),
                            col0, g, full.ptr_to([st.stage]), txl.uint64(EVICT_NORMAL),
                        )  # fmt: skip
                        st.advance()

                tiles(body)

        with r_sf:
            st = txl.PipelineState(STAGES, 1)
            ra = txl.alloc_local((16,), "uint32")
            rb = txl.alloc_local((4 * N_SF_CHUNKS,), "uint32")
            fv = txl.local_scalar("float32")
            n_sfb = (N + 127) // 128

            def load_group(m0, col0, g, kb):
                if VSF:
                    for i in range(4):
                        txl.ptx.ld.global_.nc.v4.b32(
                            ra[i * 4], ra[i * 4 + 1], ra[i * 4 + 2], ra[i * 4 + 3],
                            sfa.ptr_to([m0 + lane + 32 * i, kb]),
                        )
                    for c in range(N_SF_CHUNKS):
                        nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                        txl.ptx.ld.global_.nc.v4.b32(
                            rb[c * 4], rb[c * 4 + 1], rb[c * 4 + 2], rb[c * 4 + 3], sfb.ptr_to([g, nsf, kb])
                        )
                else:
                    for j in range(4):
                        kk = txl.min(kb + j, kb_sf - 1)
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(fv, sfa.ptr_to([m0 + lane + 32 * i, kk]))
                            txl.assign(ra[i * 4 + j], txl.reinterpret(fv, ty="uint32"))
                        for c in range(N_SF_CHUNKS):
                            nsf = txl.min(udiv(col0, 128) + c, n_sfb - 1)
                            txl.ptx.ld.global_.nc.f32(fv, sfb.ptr_to([g, nsf, kk]))
                            txl.assign(rb[c * 4 + j], txl.reinterpret(fv, ty="uint32"))

            def body(m0, col0, g, hf):
                load_group(m0, col0, g, 0)
                with txl.serial(nk) as k:
                    twait("sf-wait-empty", empty, st.stage, st.phase)
                    # the previous tcgen05.cp of this stage (async proxy) completed: order our writes after it
                    txl.ptx.fence.proxy.async_.shared__cta()
                    with txl.if_(umod(k, 4) == 0), txl.then_():
                        txl.ptx.st.shared.v4.u32(
                            s_sfa.ptr_to([st.stage * 128 + lane * 4]),
                            pack4(ra, 0), pack4(ra, 4), pack4(ra, 8), pack4(ra, 12),
                        )  # fmt: skip
                        for c in range(N_SF_CHUNKS):
                            wb = txl.local_scalar("uint32", init=pack4(rb, c * 4))
                            txl.ptx.st.shared.v4.u32(s_sfb.ptr_to([st.stage * BN + c * 128 + lane * 4]), wb, wb, wb, wb)
                        with txl.if_(k + 4 < nk), txl.then_():
                            load_group(m0, col0, g, k + 4)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    sf_full.arrive(st.stage)
                    st.advance()

            tiles(body)

        with r_mma:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                st = txl.PipelineState(STAGES, 0)
                acc = txl.PipelineState(1, 0)
                a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                sfa_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0])))
                sfb_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0])))

                def body(m0, col0, g, hf):
                    twait("mma-wait-acc", t_empty, acc.stage, acc.phase ^ 1)
                    tile_tok = rng("mma-tile")
                    txl.ptx.tcgen05.fence__after_thread_sync()
                    with txl.serial(nk) as k:
                        d0 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, a_base + u322(st.stage * A_BYTES)))
                        d1 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, b_base + u322(st.stage * B_BYTES)))
                        idesc = txl.local_scalar(
                            "uint32",
                            init=with_sf_id(txl.if_then_else(hf == 1, txl.uint32(INSTR_H), txl.uint32(INSTR)), umod(k, 4)),
                        )
                        sd0 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfa_base + u322(st.stage * 512)))
                        sd1 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfb_base + u322(st.stage * (BN * 4))))
                        twait("mma-wait-tma", full, st.stage, st.phase)
                        twait("mma-wait-sf", sf_full, st.stage, st.phase)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.if_(umod(k, 4) == 0), txl.then_():
                            txl.ptx[UTCCP_CHAIN](
                                txl.uint32(SFA_COL), sd0
                            )
                            for c in range(N_SF_CHUNKS):
                                txl.ptx[UTCCP_CHAIN](
                                    txl.uint32(SFB_COL + 4 * c),
                                    dadd(sd1, c * 512),
                                )
                        for ki in range(BK // UMMA_K):
                            txl.ptx[MMA_CHAIN](
                                txl.uint32(0),
                                dadd(d0, ki * UMMA_K),
                                dadd(d1, ki * UMMA_K),
                                idesc,
                                txl.uint32(SFA_COL),
                                txl.uint32(SFB_COL),
                                txl.Or(k > 0, ki > 0) if ki == 0 else txl.bool(True),
                            )
                        empty.arrive(st.stage)
                        st.advance()
                    t_full.arrive(acc.stage)
                    end(tile_tok)
                    acc.advance()

                tiles_nowalk(body)

        with r_epi:
            ew = warp - 4
            quad = warp % 4
            wg = ew // 4
            wtid = (ew % 4) * 32 + lane
            row = quad * 32 + lane
            acc = txl.PipelineState(1, 0)
            vals = txl.alloc_local((128,), "uint32")
            packed = txl.alloc_local((64,), "uint32")

            def body(m0, col0, g, hf):
                live = txl.Or(hf == 0, wg == 0)  # a half tile's 128 columns belong to warpgroup 0
                twait("epi-wait-acc", t_full, acc.stage, acc.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                with txl.if_(live), txl.then_():
                    for c in range(4):
                        taddr = txl.bitwise_or(txl.shift_left(u322(quad * 32), txl.uint32(16)), u322(wg * 128 + c * 32))
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[vals[c * 32 + j] for j in range(32)], taddr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx.tcgen05.fence__before_thread_sync()
                t_empty.arrive(acc.stage)
                with txl.if_(live), txl.then_():
                    for p in range(64):
                        txl.ptx.cvt.rn.bf16x2.f32(
                            packed[p], txl.reinterpret(vals[2 * p + 1], ty="float32"), txl.reinterpret(vals[2 * p], ty="float32")
                        )
                    buf = wg * (BM * 16)
                    for c in range(4):
                        with txl.if_(wtid == 0), txl.then_():
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
                        with txl.if_(wtid == 0), txl.then_():
                            txl.ptx[STORE2D](txl.address_of(d_map), col0 + wg * 128 + c * 32, m0, s_cd.ptr_to([buf]))
                            txl.ptx.cp.async_.bulk.commit_group()
                acc.advance()

            tiles(body)
            with txl.if_(wtid == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.bar.sync(txl.uint32(0))
        with r_idle:
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped_1cta



def build_mgrouped_swap(STAGES, VSF=False, TAIL_STEPS=4):
    """
    Swap-AB 2-CTA m-grouped kernel: MMA M = 256 weight rows (128 per CTA), MMA N = tokens.

    A pair tile covers 256 tokens of one group, or 128 for a group's odd tail (MMA N = 128, half
    the work), times 256 output columns; CTA r stores columns [r*128, r*128+128) of the tile.
    """
    assert 1 <= TAIL_STEPS <= BK // UMMA_K
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
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, p["a"].data, K, M, K, BK, 128, 1, 1, 0, 3, L2P, 0)
        txl.cu_tensor_map_encode_tiled(a64_map, FP8, 2, p["a"].data, K, M, K, BK, 64, 1, 1, 0, 3, L2P, 0)
        txl.cu_tensor_map_encode_tiled(
            b_map, FP8, 3, p["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, 128, 1,
            1, 1, 1, 0, 3, L2P, 0,
        )  # fmt: skip
        txl.cu_tensor_map_encode_tiled(
            d_map, "bfloat16", 2, p["d"].data, N, M, N * 2, 128, 32, 1, 1, 0, 0, 2, 0
        )
        return a_map, a64_map, b_map, d_map

    @txl.kernel()
    def mgrouped_swap(
        a: txl.Tensor((txl.int64(), txl.int64()), FP8),
        sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32),
        b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8),
        sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32),
        d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16),
        offs: txl.Tensor((txl.int64(),), txl.i32),
        M: txl.int32,
        N: txl.int32,
        K: txl.int32,
        G: txl.int32,
    ):
        host = prelude(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=(4 + EPI_WARPS) * 32, cluster=[2]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, a64_map, b_map, d_map = host
        rank = txl.cuda.cluster_cta_id("x")
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
        with txl.if_(tid == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
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
            # one redux.sync per value: the six walk reductions are independent (no 5-deep shuffle chains)
            txl.ptx.redux_sync.add.u32(v, v, txl.uint32(0xFFFFFFFF))

        with txl.serial((G + 31) // 32) as wc:
            gi = txl.local_scalar("int32", init=wc * 32 + lane)
            with txl.if_(wc == 0):
                with txl.then_():
                    txl.assign(cur, cur0)
                    txl.assign(prev, prev0)
                with txl.else_():
                    txl.ptx.ld.global_.s32(cur, offs.ptr_to([txl.min(gi, G - 1)]))
                    txl.ptx.ld.global_.s32(prev, offs.ptr_to([txl.max(txl.min(gi, G - 1) - 1, 0)]))
            with txl.if_(gi == 0), txl.then_():
                txl.assign(prev, 0)
            cnt = txl.local_scalar("uint32", init=txl.if_then_else(gi < G, u322(udiv(cur - prev + 255, 256)), txl.uint32(0)))
            incl = txl.local_scalar("uint32", init=cnt)
            for x in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(x), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.if_(lane >= x), txl.then_():
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
        # the group walk (registers + global loads only) overlaps the cluster barrier latency
        txl.ptx.barrier.cluster.wait.acquire.aligned()

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
            with txl.while_(t < total * n_nb):
                pp = txl.local_scalar("int32", init=udiv(t, n_nb))
                nb = txl.local_scalar("int32", init=umod(t, n_nb))
                with txl.while_(pp >= pend):
                    txl.assign(g, g + 1)
                    txl.assign(gbeg, gend)
                    txl.ptx.ld.global_.s32(gend, offs.ptr_to([g]))
                    txl.assign(pbeg, pend)
                    txl.assign(pend, pend + udiv(gend - gbeg + 255, 256))
                m0 = txl.local_scalar("int32", init=gbeg + (pp - pbeg) * 256)
                body(m0, nb, g, gend - m0 >= 256)
                txl.assign(t, t + NUM_SMS // 2)

        with r_tma:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
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
                        with txl.if_(rank == 0), txl.then_():
                            full0.arrive(st.stage, tx)
                        txl.ptx[LD3](
                            s_w.ptr_to([st.stage * W_BYTES]),
                            txl.address_of(b_map),
                            txl.cast(k * BK, "int32"),
                            nb * 256 + rank * 128,
                            g,
                            bar,
                        )
                        with txl.if_(fullt):
                            with txl.then_():
                                txl.ptx[LD2](
                                    s_t.ptr_to([st.stage * T_BYTES]),
                                    txl.address_of(a_map),
                                    txl.cast(k * BK, "int32"),
                                    m0 + rank * 128,
                                    bar,
                                )
                            with txl.else_():
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
                if VSF:
                    txl.ptx.ld.global_.nc.v4.b32(ra[0], ra[1], ra[2], ra[3], sfb.ptr_to([g, nsf, kb]))
                    for c in range(2):
                        for i in range(4):
                            row = txl.min(m0 + c * 128 + lane + 32 * i, M - 1)
                            base = c * 16 + i * 4
                            txl.ptx.ld.global_.nc.v4.b32(
                                rb[base], rb[base + 1], rb[base + 2], rb[base + 3], sfa.ptr_to([row, kb])
                            )
                else:
                    for j in range(4):
                        kk = txl.min(kb + j, kb_sf - 1)
                        txl.ptx.ld.global_.nc.f32(fv, sfb.ptr_to([g, nsf, kk]))
                        txl.assign(ra[j], txl.reinterpret(fv, ty="uint32"))
                        for c in range(2):
                            for i in range(4):
                                txl.ptx.ld.global_.nc.f32(
                                    fv, sfa.ptr_to([txl.min(m0 + c * 128 + lane + 32 * i, M - 1), kk])
                                )
                                txl.assign(rb[c * 16 + i * 4 + j], txl.reinterpret(fv, ty="uint32"))

            def body(m0, nb, g, fullt):
                load_group(m0, nb, g, 0)
                with txl.serial(nk) as k:
                    twait("sf-wait-empty", empty, st.stage, st.phase)
                    # The previous tcgen05.cp of this stage (async proxy) completed; order it before our writes.
                    txl.ptx.fence.proxy.async_.shared__cta()
                    with txl.if_(umod(k, 4) == 0), txl.then_():
                        wa = txl.local_scalar("uint32", init=pack4(ra, 0))
                        txl.ptx.st.shared.v4.u32(s_sfa.ptr_to([st.stage * 128 + lane * 4]), wa, wa, wa, wa)
                        for c in range(2):
                            txl.ptx.st.shared.v4.u32(
                                s_sfb.ptr_to([st.stage * 256 + c * 128 + lane * 4]),
                                pack4(rb, c * 16), pack4(rb, c * 16 + 4),
                                pack4(rb, c * 16 + 8), pack4(rb, c * 16 + 12),
                            )  # fmt: skip
                        with txl.if_(k + 4 < nk), txl.then_():
                            load_group(m0, nb, g, k + 4)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    sf_full0.arrive(st.stage)
                    st.advance()

            tiles(body)

        with r_mma:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
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
                        def issue_stage(k, mma_steps):
                            d0 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, w_base + u322(st.stage * W_BYTES)))
                            d1 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, t_base + u322(st.stage * T_BYTES)))
                            idesc = txl.local_scalar("uint32", init=with_sf_id(instr, umod(k, 4)))
                            sd0 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfa_base + u322(st.stage * 512)))
                            sd1 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfb_base + u322(st.stage * 1024)))
                            twait("mma-wait-tma", full, st.stage, st.phase)
                            twait("mma-wait-sf", sf_full, st.stage, st.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.if_(umod(k, 4) == 0), txl.then_():
                                txl.ptx[UTCCP2](
                                    txl.uint32(SFA_COL),
                                    sd0,
                                )
                                for c in range(2):
                                    txl.ptx[UTCCP2](
                                        txl.uint32(SFB_COL + 4 * c),
                                        dadd(sd1, c * 512),
                                    )
                            for ki in range(mma_steps):
                                txl.ptx[MMA2](
                                    txl.uint32(0),
                                    dadd(d0, ki * UMMA_K),
                                    dadd(d1, ki * UMMA_K),
                                    idesc,
                                    txl.uint32(SFA_COL),
                                    txl.uint32(SFB_COL),
                                    txl.Or(k > 0, ki > 0) if ki == 0 else txl.bool(True),
                                )
                            empty.arrive(st.stage, cta_group=2, cta_mask=3)
                            st.advance()
                        if TAIL_STEPS < BK // UMMA_K:
                            # Specialize by K: full stages have no tail predicates, and the peeled last
                            # stage omits the zero-filled 32-wide MMA steps.
                            with txl.serial(nk - 1) as k:
                                issue_stage(k, BK // UMMA_K)
                            issue_stage(nk - 1, TAIL_STEPS)
                        else:
                            with txl.serial(nk) as k:
                                issue_stage(k, BK // UMMA_K)
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
                with txl.if_(live), txl.then_():
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
                    with txl.if_(live), txl.then_():
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
                                packed[p], txl.reinterpret(hi, ty="float32"), txl.reinterpret(lo, ty="float32")
                            )
                        with txl.if_(wtid == 0), txl.then_():
                            txl.ptx.cp.async_.bulk.wait_group.read(0)
                        txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                        for p in range(16):
                            txl.ptx.st.shared.u32(s_cd.ptr_to([buf + (2 * p + par) * 64 + col // 2]), packed[p])
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                        with txl.if_(wtid == 0), txl.then_():
                            txl.ptx[STORE2D](
                                txl.address_of(d_map), nb * 256 + rank * 128, m0 + wg * 128 + c * 32, s_cd.ptr_to([buf])
                            )
                            txl.ptx.cp.async_.bulk.commit_group()
                acc.advance()

            tiles(body)
            with txl.if_(wtid == 0), txl.then_():
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


def build_kgrouped(STAGES, CDB=1, REV=False, BN=256, PF=False, EPF=False, RC=32, SHARECD=False):
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
    RC_SW = 3 if RC == 32 else 2  # reduce box: RC fp32 columns = 128 B (SW128) or 64 B (SW64) rows
    if SHARECD:
        # One 16 KiB SW128 staging box is time-shared by the two epilogue warpgroups.  This keeps
        # the six-stage operand ring while halving the number of reduce-add TMA instructions.
        assert RC == 32 and CDB == 1 and BN == 256
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
            RC, BM, 1, 1, 1, 1, 0, RC_SW, 2, 0,
        )  # fmt: skip
        return a_map, b_map, d_map

    @txl.kernel()
    def kgrouped(
        a: txl.Tensor((txl.int64(), txl.int64()), FP8),
        sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32),
        b: txl.Tensor((txl.int64(), txl.int64()), FP8),
        sfb: txl.Tensor((txl.int64(), txl.int64()), txl.f32),
        d: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32),
        ks: txl.Tensor((txl.int64(),), txl.i32),
        ctr: txl.Tensor((txl.int64(),), txl.i32),
        M: txl.int32,
        N: txl.int32,
        K: txl.int32,
        G: txl.int32,
    ):
        host = prelude(locals())
        # Programmatic dependent launch: the prologue (barrier init, TMEM allocation, cluster sync)
        # overlaps the tail of the preceding kernel; all global accesses follow griddepcontrol.wait.
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=384, cluster=[2], programmatic_stream_serialization=True), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, d_map = host
        rank = txl.cuda.cluster_cta_id("x")
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
        s_cd = smem.alloc(((1 if SHARECD else 2) * CDB * BM * RC,), "uint32", align=1024)
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
        with txl.if_(tid == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with r_idle:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        # PDL: the prologue above overlaps the preceding grid; every global access (operands, scales,
        # ks, the tile counter, d) follows its completion.
        txl.ptx.griddepcontrol.wait()
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
                with txl.if_(txl.Or(txl.bool(one_lane), lane == 0)), txl.then_():
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
            with txl.while_(t >= 0):
                if REV:
                    tg = txl.local_scalar("int32", init=G - 1 - udiv(t, per_group))
                    with txl.while_(g > tg):
                        txl.assign(g, g - 1)
                        txl.ptx.ld.global_.s32(kg, ks.ptr_to([g]))
                        txl.assign(row0, row0 - kg)
                    r = txl.local_scalar("int32", init=per_group - 1 - (t - (G - 1 - tg) * per_group))
                else:
                    tg = txl.local_scalar("int32", init=udiv(t, per_group))
                    with txl.while_(g < tg):
                        txl.assign(row0, row0 + kg)
                        txl.assign(g, g + 1)
                        txl.ptx.ld.global_.s32(kg, ks.ptr_to([g]))
                    r = txl.local_scalar("int32", init=t - tg * per_group)
                m0 = txl.local_scalar("int32", init=udiv(r, n_nt) * 256 + rank * BM)
                n0 = txl.local_scalar("int32", init=umod(r, n_nt) * BN)
                with txl.if_(kg > 0), txl.then_():
                    body(m0, n0, g, row0, udiv(kg, BK))
                next_tile()

        with r_tma:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                st = txl.PipelineState(STAGES, 1)

                txl.ptx.prefetch.tensormap(txl.address_of(d_map))

                def body(m0, n0, g, row0, nk):
                    # PF: pull this CTA's d slice into L2 ahead of the epilogue's reduce-add. A reduce that
                    # misses L2 waits on its own DRAM read, exposed when tiles are short (small groups).
                    if PF:
                        for c in range(BN // RC):
                            txl.ptx[PREFETCH3D](txl.address_of(d_map), n0 + c * RC, m0, g, u64c(EVICT_LAST))
                    with txl.serial(nk) as k:
                        twait("tma-wait-empty", empty, st.stage, st.phase)
                        bar = txl.cuda.cvta_generic_to_shared(full0.ptr_to([st.stage]))
                        with txl.if_(rank == 0), txl.then_():
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
                        txl.assign(ra[i * 4 + j], txl.reinterpret(fv, ty="uint32"))
                        for c in range(N_SF_CHUNKS):
                            txl.ptx.ld.global_.nc.f32(
                                fv, sfb.ptr_to([kk, txl.min(n0 + c * 128 + lane + 32 * i, K - 1)])
                            )
                            txl.assign(rb[c * 16 + i * 4 + j], txl.reinterpret(fv, ty="uint32"))

            def body(m0, n0, g, row0, nk):
                load_group(m0, n0, row0, nk, 0)
                with txl.serial(nk) as k:
                    twait("sf-wait-empty", empty, st.stage, st.phase)
                    # The previous tcgen05.cp of this stage (async proxy) completed; order it before our writes.
                    txl.ptx.fence.proxy.async_.shared__cta()
                    with txl.if_(umod(k, 4) == 0), txl.then_():
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
                        with txl.if_(k + 4 < nk), txl.then_():
                            load_group(m0, n0, row0, nk, k + 4)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    sf_full0.arrive(st.stage)
                    st.advance()

            tiles(body, one_lane=False)

        with r_idle:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    qs = txl.PipelineState(SCHED_DEPTH, 1)
                    nt = txl.local_scalar("int32", init=0)
                    peer_tile = txl.local_scalar("uint32")
                    peer_full = txl.local_scalar("uint32")
                    with txl.while_(nt >= 0):
                        txl.cuda.mbarrier_wait(q_empty.ptr_to([qs.stage]), qs.phase)
                        txl.ptx.atom.global_.add.s32(nt, ctr.ptr_to([0]), txl.int32(1))
                        with txl.if_(nt >= total), txl.then_():
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
                    with txl.if_(done == NUM_SMS // 2 - 1), txl.then_():
                        txl.ptx.st.global_.s32(ctr.ptr_to([0]), txl.int32(0))
                        txl.ptx.st.global_.s32(ctr.ptr_to([1]), txl.int32(0))

        with r_mma:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
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
                            d0 = txl.local_scalar("uint64", init=smem_desc(AB_DESC_BASE, a_base + u322(st.stage * A_BYTES)))
                            d1 = txl.local_scalar("uint64", init=smem_desc(B_DESC_BASE, b_base + u322(st.stage * B_HALF)))
                            idesc = txl.local_scalar("uint32", init=with_sf_id(txl.uint32(INSTR), umod(k, 4)))
                            sd0 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfa_base + u322(st.stage * 512)))
                            sd1 = txl.local_scalar("uint64", init=smem_desc(SF_DESC_BASE, sfb_base + u322(st.stage * (BN * 4))))
                            twait("mma-wait-tma", full, st.stage, st.phase)
                            twait("mma-wait-sf", sf_full, st.stage, st.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.if_(umod(k, 4) == 0), txl.then_():
                                txl.ptx[UTCCP2](
                                    txl.uint32(SFA_COL), sd0
                                )
                                for c in range(N_SF_CHUNKS):
                                    txl.ptx[UTCCP2](
                                        txl.uint32(SFB_COL + 4 * c),
                                        dadd(sd1, c * 512),
                                    )
                            for ki in range(BK // UMMA_K):
                                txl.ptx[MMA2](
                                    u322(acc.stage * BN),
                                    dadd(d0, ki * K_STEP),
                                    dadd(d1, ki * B_K_STEP),
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
                    with txl.if_(wtid == 0), txl.then_():
                        for c in range(COLS // RC):
                            txl.ptx[PREFETCH3D](
                                txl.address_of(d_map), n0 + wg * COLS + c * RC, m0, g, u64c(EVICT_LAST)
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
                if SHARECD:
                    # Batch all four chunks for one owner before handing the box to the other WG.  The
                    # final common barrier of a tile protects the first write of the next one, so only
                    # two cross-WG barriers are needed per tile (versus one per reduce in r170).
                    for owner in range(2):
                        with txl.if_(wg == owner), txl.then_():
                            for c in range(COLS // RC):
                                with txl.if_(wtid == 0), txl.then_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                                txl.ptx.bar.sync(u322(EPI_BAR + owner), txl.uint32(128))
                                for q in range(RC // 4):
                                    swz = txl.bitwise_xor(q, row % 8)
                                    txl.ptx.st.shared.v4.u32(
                                        s_cd.ptr_to([row * RC + swz * 4]),
                                        vals[c * RC + q * 4], vals[c * RC + q * 4 + 1],
                                        vals[c * RC + q * 4 + 2], vals[c * RC + q * 4 + 3],
                                    )  # fmt: skip
                                txl.ptx.fence.proxy.async_.shared__cta()
                                txl.ptx.bar.sync(u322(EPI_BAR + owner), txl.uint32(128))
                                with txl.if_(wtid == 0), txl.then_():
                                    txl.ptx[REDUCE3D + ".L2::cache_hint"](
                                        txl.address_of(d_map), n0 + owner * COLS + c * RC, m0, g,
                                        s_cd.ptr_to([0]), u64c(EVICT_LAST),
                                    )
                                    txl.ptx.cp.async_.bulk.commit_group()
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                        txl.ptx.bar.sync(u322(EPI_BAR + 2), txl.uint32(256))
                else:
                    for c in range(COLS // RC):
                        buf = (wg * CDB + c % CDB) * (BM * RC)
                        with txl.if_(wtid == 0), txl.then_():
                            txl.ptx.cp.async_.bulk.wait_group.read(CDB - 1)
                        txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                        for q in range(RC // 4):
                            swz = txl.bitwise_xor(q, row % 8) if RC == 32 else txl.bitwise_xor(q, (row // 2) % 4)
                            txl.ptx.st.shared.v4.u32(
                                s_cd.ptr_to([buf + row * RC + swz * 4]),
                                vals[c * RC + q * 4], vals[c * RC + q * 4 + 1],
                                vals[c * RC + q * 4 + 2], vals[c * RC + q * 4 + 3],
                            )  # fmt: skip
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(u322(EPI_BAR + wg), txl.uint32(128))
                        with txl.if_(wtid == 0), txl.then_():
                            # evict_last keeps the reduced d lines resident in L2 while the stream runs
                            txl.ptx[REDUCE3D + ".L2::cache_hint"](
                                txl.address_of(d_map), n0 + wg * COLS + c * RC, m0, g, s_cd.ptr_to([buf]), u64c(EVICT_LAST)
                            )
                            txl.ptx.cp.async_.bulk.commit_group()
                end(tok2)
                acc.advance()

            tiles(body, one_lane=False)
            with txl.if_(wtid == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with r_idle:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return kgrouped


def compile_v(kernel):
    return entry_function(kernel.compile(TARGET))


@functools.cache
def mgrouped_swap_exec(STAGES, VSF=False, TAIL_STEPS=4):
    return compile_v(build_mgrouped_swap(STAGES, VSF, TAIL_STEPS))


@functools.cache
def mgrouped_1cta_exec(STAGES, VSF=False):
    return compile_v(build_mgrouped_1cta(STAGES, VSF))


@functools.cache
def mgrouped_exec(STAGES, BN, CDB=1, SFT=False, VSF=False):
    return compile_v(build_mgrouped(STAGES, BN, CDB, SFT, VSF))


@functools.cache
def mgrouped_w320_exec(STAGES, VSF=False):
    return compile_v(build_mgrouped_w320(STAGES, VSF))


@functools.cache
def mgrouped_w320r_exec(STAGES, VSF=False, DEFER=2, SW=160, CDB=1, LASTCD=False):
    return compile_v(build_mgrouped_w320r(STAGES, VSF, DEFER, SW, CDB, LASTCD))


def mcfg(K, N):
    """
    (STAGES, BN, CDB) per shape: BN=192 (double-buffered accumulator, 7 stages) when it tiles N
    exactly and 256 does not; otherwise 256-wide tiles, with 4 store buffers when K is short.
    """
    if N % 256 != 0 and N % 192 == 0:
        return (7, 192, 1)
    return (5, 256, 2, True) if K <= 512 else (6, 256, 1, True)


ONECTA = True
W320_STAGES = 5
W320_DEFER = 3
ROT256 = True
ROT256_DEFER = 2


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
    vsf = ((K + 127) // 128) % 4 == 0 and sx.data_ptr() % 16 == 0 and sw.data_ptr() % 16 == 0
    if n % 320 == 0 and n % 256 != 0 and M >= 512 * G:
        ex = mgrouped_w320r_exec(W320_STAGES, vsf, W320_DEFER, 160, 1, True)
    elif M < 512 * G and n % 256 == 0 and ONECTA:
        # small groups, 256-multiple n: 1-CTA 128-row tiles (no pair padding)
        ex = mgrouped_1cta_exec(4, vsf)
    elif (M < 512 * G or swap_pad) and K > 512:
        tail_steps = (K % BK + UMMA_K - 1) // UMMA_K if K % BK else BK // UMMA_K
        ex = mgrouped_swap_exec(6, vsf, tail_steps)
    elif n % 320 == 0 and n % 256 != 0:
        # n = 2880: 320-wide tiles (two N=160 sub-MMAs) move 23% fewer operand bytes per FLOP than BN=192
        ex = mgrouped_w320_exec(W320_STAGES, vsf)
    elif n % 256 == 0 and K <= 512 and n >= 1024 and ROT256:
        ex = mgrouped_w320r_exec(5, vsf, ROT256_DEFER, 128, 2)
    elif n % 256 == 0 and K >= 1024 and n >= 1024 and ROT256:
        # 256-wide tiles as two N=128 sub-MMAs over three rotating 128-col TMEM slots (scales in dedicated
        # columns): no scale-tail round trip between tiles. Short K (512: store-bound) and n = 512 rows stay
        # on the BN256 scale-tail kernel (measured +8 / +1 us there).
        ex = mgrouped_w320r_exec(6, vsf, ROT256_DEFER, 128)
    else:
        ex = mgrouped_exec(*mcfg(K, n), VSF=vsf)
    ex(x, sx, w, sw, out, grouped_mm_offs, M, n, K, G)
    return out


@functools.cache
def kgrouped_exec(STAGES, CDB=1, REV=False, BN=256, PF=False, EPF=False, RC=32, SHARECD=False):
    return compile_v(build_kgrouped(STAGES, CDB, REV, BN, PF, EPF, RC, SHARECD))


@txl.kernel()
def zero2(ctr: txl.Tensor((txl.int64(),), txl.i32)):
    txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
    with txl.if_(txl.thread_id() < 2), txl.then_():
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
    # d prefetch into L2 when groups average < 2048 rows: the epilogue prefetches its tile's d slice at
    # tile start (about one MMA tile of lead), so the reduce-add hits L2 instead of stalling on its own
    # DRAM miss. Measured per wgrad row (official timer): -12..-54 us for 256-1080 rows/group; with
    # >= 2048 rows/group the tiles are long enough that the prefetched lines only add traffic (+2..+13 us).
    if M < 2048 * G:
        ex = kgrouped_exec(5, 1, True, 256, False, True)
    else:
        # long tiles (TC-bound): a sixth ring stage, funded by 16-column (8 KB per warpgroup) reduce chunks
        ex = kgrouped_exec(6, 1, True, 256, False, False, 32, True)
    ex(xa, sa, xb, sb, d, ks_tensor, tile_counter(d.device), M, N, K, G)
