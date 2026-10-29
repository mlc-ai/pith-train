"""
fp8_quant_act on SM100 in TIRx-lite.

Fast path (bf16, unit column stride, 16 B aligned base and rows, K % 16 == 0, M % 16 == 0): a
persistent warp-specialized kernel. One producer warp streams 128x128 bf16 tiles with TMA into a
multi-stage mbarrier ring; eight compute warps quantize each tile with packed bf16x2 math (the
power-of-two reciprocal multiply is exact), stage the rowwise FP8 tile in smem for a TMA store,
write the transposed FP8 through an XOR-swizzled smem tile as 16 B rows, and store the row scales
column-major (x_scale strides are unspecified by the contract). Everything else runs a generic
scalar kernel (any stride / dtype / alignment).

Evolved with TIRx-harness: run act, frontier member hybrid-persistent-oneshot (commit e9efe2b, 1.263x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools

import torch
import tvm_ffi

import tirx_kernels.tirx_lite as txl

__all__ = ["fp8_quant_act_sm100"]

BLOCK = 128
DT = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
RCP_448 = 1.0 / 448.0
FULL = 0xFFFFFFFF


def e8m0(amax):
    """
    (scale, rcp) of an f32 amax: 2^ceil(log2(max(amax, 1e-4) * fp32(1/448))), saturated at 2^127.
    """
    ratio = txl.local_scalar("float32")
    txl.ptx.max.f32(ratio, amax, txl.float32(1e-4))
    txl.ptx.mul.f32(ratio, ratio, txl.float32(RCP_448))
    bits = txl.reinterpret("uint32", ratio)
    sbits = txl.local_scalar("uint32", init=txl.bitwise_and(bits, txl.uint32(0x7F800000)))
    with txl.If(txl.bitwise_and(bits, txl.uint32(0x007FFFFF)) != txl.uint32(0)), txl.Then():
        txl.assign(sbits, sbits + txl.uint32(0x00800000))
    txl.assign(sbits, txl.min(sbits, txl.uint32(254 << 23)))
    scale = txl.reinterpret("float32", sbits)
    rcp = txl.reinterpret("float32", txl.uint32(254 << 23) - sbits)
    return scale, rcp


def fp8_byte(v):
    pair = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(pair, txl.float32(0.0), v)
    return txl.cast(txl.bitwise_and(txl.cast(pair, "uint32"), txl.uint32(0xFF)), "uint8")


def half_to_f32(h16, dtype):
    """
    f32 of the 16-bit value in the low half of a u32.
    """
    if dtype == "bfloat16":
        return txl.reinterpret("float32", txl.shift_left(h16, txl.uint32(16)))
    return txl.cast(txl.reinterpret("float16", txl.cast(h16, "uint16")), "float32")


def lo(word, dtype):
    return half_to_f32(txl.bitwise_and(word, txl.uint32(0xFFFF)), dtype)


def hi(word, dtype):
    if dtype == "bfloat16":
        return txl.reinterpret("float32", txl.bitwise_and(word, txl.uint32(0xFFFF0000)))
    return half_to_f32(txl.shift_right(word, txl.uint32(16)), dtype)


@functools.cache
def generic_kernel(dtype: str):
    @txl.kernel(warps=4, arch="sm_100a", grid=lambda p: p["nmb"] * p["nkb"])
    def fp8_quant_act_generic(
        x: txl.gptr[dtype],
        q: txl.gptr[txl.u8],
        s: txl.gptr[txl.f32],
        qt: txl.gptr[txl.u8],
        st: txl.gptr[txl.f32],
        M: txl.i32,
        K: txl.i32,
        nmb: txl.i32,
        nkb: txl.i32,
        sr: txl.i64,
        sc: txl.i64,
    ):
        b = txl.cta_id()
        t = txl.thread_id()
        smem = txl.smem_pool()
        red = smem.alloc([4], txl.f32)
        tm = txl.local_scalar("int32", init=txl.truncmod(b, nmb))
        tk = txl.local_scalar("int32", init=txl.truncdiv(b, nmb))
        r = txl.local_scalar("int32", init=tm * BLOCK + t)
        rvalid = r < M
        amax = txl.local_scalar("float32", init=txl.float32(0.0))
        v = txl.local_scalar("float32")
        h = txl.local_scalar("uint16")

        def load(c):
            col = tk * BLOCK + c
            txl.assign(v, txl.float32(0.0))
            with txl.If(rvalid & (col < K)), txl.Then():
                addr = x.ptr_to([txl.cast(r, "int64") * sr + txl.cast(col, "int64") * sc])
                if dtype == "float32":
                    txl.ptx["ld.global.nc.f32"](v, addr)
                else:
                    txl.ptx["ld.global.nc.b16"](h, addr)
                    txl.assign(v, half_to_f32(txl.cast(h, "uint32"), dtype))
            return col

        with txl.serial(BLOCK) as c:
            load(c)
            txl.ptx.max.f32(amax, amax, txl.abs(v))
        scale, rcp = e8m0(amax)
        rscale = txl.local_scalar("float32", init=scale)
        rrcp = txl.local_scalar("float32", init=rcp)
        with txl.If(rvalid), txl.Then():
            txl.ptx.st.global_.f32(s.ptr_to([txl.cast(r, "int64") * nkb + tk]), rscale)
        with txl.serial(BLOCK) as c:
            col = load(c)
            with txl.If(rvalid & (col < K)), txl.Then():
                txl.ptx.st.global_.b8(q.ptr_to([txl.cast(r, "int64") * K + col]), fp8_byte(v * rrcp))

        peer = txl.local_scalar("uint32")
        for d in (16, 8, 4, 2, 1):
            txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret("uint32", amax), txl.uint32(d), txl.uint32(31), txl.uint32(FULL))
            txl.ptx.max.f32(amax, amax, txl.reinterpret("float32", peer))
        with txl.If(txl.bitwise_and(t, 31) == 0), txl.Then():
            txl.ptx.st.shared.f32(red.ptr_to([t >> 5]), amax)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(128))
        bmax = txl.local_scalar("float32")
        tmp = txl.local_scalar("float32")
        txl.ptx.ld.shared.f32(bmax, red.ptr_to([0]))
        for w in range(1, 4):
            txl.ptx.ld.shared.f32(tmp, red.ptr_to([w]))
            txl.ptx.max.f32(bmax, bmax, tmp)
        bscale, brcp = e8m0(bmax)
        bb = txl.local_scalar("float32", init=brcp)
        with txl.If(t == 0), txl.Then():
            txl.ptx.st.global_.f32(st.ptr_to([tk * nmb + tm]), bscale)
        with txl.serial(BLOCK) as c:
            col = load(c)
            with txl.If(rvalid & (col < K)), txl.Then():
                txl.ptx.st.global_.b8(qt.ptr_to([txl.cast(col, "int64") * M + r]), fp8_byte(v * bb))

    return fp8_quant_act_generic.compile()


# ----------------------------------------------------------------------------- fast path


TMA_G2S = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
TMA_S2G = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"


def range_v(name):
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def end(token):
    txl.cuda.iket.range_end(token[0])


def bmul(d, a, r2):
    txl.ptx["mul.rn.bf16x2"](d, a, r2)


def cvt8(d16, a):
    txl.ptx["cvt.rn.satfinite.e4m3x2.bf16x2"](d16, a)


def pack16(lo16, hi16):
    return txl.bitwise_or(txl.cast(lo16, "uint32"), txl.shift_left(txl.cast(hi16, "uint32"), txl.uint32(16)))


def bf16x2_of(f32):
    """
    Both halves = the bf16 of an f32 power of two (exact).
    """
    u = txl.reinterpret("uint32", f32)
    return txl.bitwise_or(txl.shift_right(u, txl.uint32(16)), txl.bitwise_and(u, txl.uint32(0xFFFF0000)))


CFG = dict(stages=2, ctas_per_sm=2)
ONESHOT_CTAS_PER_SM = 3


@functools.cache
def fast_kernel(stages: int, ctas_per_sm: int):
    return fast_func(stages, ctas_per_sm).compile()


def fast_func(stages: int, ctas_per_sm: int):
    WARPS = 9  # 8 compute warps + 1 TMA producer warp
    S = stages
    TILE_BYTES = BLOCK * BLOCK * 2
    NP = 8  # TMA row parts per tile: part w holds compute warp w's 16 rows

    def host_prelude(params):
        x_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            x_map, "bfloat16", 2, params["x"].data,
            params["K"], params["M"], params["sr"] * 2, BLOCK, BLOCK // NP, 1, 1, 0, 0, 2, 0,
        )  # fmt: skip
        q_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            q_map, "uint8", 2, params["q"].data,
            params["K"], params["M"], params["K"], BLOCK, BLOCK, 1, 1, 0, 0, 2, 0,
        )  # fmt: skip
        qt_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            qt_map, "uint8", 2, params["qt"].data,
            params["M"], params["K"], params["M"], BLOCK, BLOCK, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        return x_map, q_map, qt_map

    @txl.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=ctas_per_sm, grid="grid_n", host_prelude=host_prelude)
    def fp8_quant_act_tma(
        x: txl.gptr[txl.bf16],
        q: txl.gptr[txl.u8],
        s: txl.gptr[txl.f32],
        qt: txl.gptr[txl.u8],
        st: txl.gptr[txl.f32],
        M: txl.i32,
        K: txl.i32,
        nmb: txl.i32,
        nkb: txl.i32,
        sr: txl.i64,
        grid_n: txl.i32,
        *,
        host,
    ):
        x_map, q_map, qt_map = host
        b = txl.cta_id()
        t = txl.thread_id()
        lane = t & 31
        warp = txl.warp_id()
        smem = txl.smem_pool()
        xs = smem.alloc([S * BLOCK * 64], txl.u32, align=1024)  # [stage][row][64 words]
        qs = smem.alloc([2 * BLOCK * 32], txl.u32, align=1024)  # [buf][row][32 words] rowwise FP8
        tile = smem.alloc([BLOCK * 32], txl.u32, align=1024)  # [k][128 B of m] FP8, TMA 128B swizzle
        red = smem.alloc([8], txl.f32, align=16)
        full = smem.alloc([S * NP], "uint64", align=8)
        empty = smem.alloc([S], "uint64", align=8)
        ntiles = nmb * nkb

        # The producer thread initializes the barriers and issues the first loads before the
        # CTA-wide prologue barrier, so the first TMA is not delayed by the compute warps' arrival.
        with txl.If(t == 8 * 32), txl.Then():
            for i in range(S):
                for pp in range(NP):
                    txl.ptx.mbarrier.init.shared.b64(full.ptr_to([i * NP + pp]), txl.uint32(1))
                txl.ptx.mbarrier.init.shared.b64(empty.ptr_to([i]), txl.uint32(8))
            txl.ptx.fence.mbarrier_init.release.cluster()

        stage = txl.local_scalar("int32", init=txl.int32(0))
        phase = txl.local_scalar("int32", init=txl.int32(0))
        tid = txl.local_scalar("int32", init=b)
        it_n = txl.local_scalar("int32", init=txl.int32(0))

        def advance():
            txl.assign(tid, tid + grid_n)
            txl.assign(it_n, it_n + 1)
            txl.assign(stage, stage + 1)
            with txl.If(stage == S), txl.Then():
                txl.assign(stage, txl.int32(0))
                txl.assign(phase, phase ^ 1)

        def produce(cond):
            with txl.While(cond()):
                tp = range_v("wait-empty")
                txl.cuda.mbarrier_wait(empty.ptr_to([stage]), phase ^ 1)
                end(tp)
                tm = txl.truncdiv(tid, nkb)
                tk = txl.truncmod(tid, nkb)
                for pp in range(NP):
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full.ptr_to([stage * NP + pp]), txl.uint32(TILE_BYTES // NP))
                    txl.ptx[TMA_G2S](
                        xs.ptr_to([stage * (BLOCK * 64) + pp * (BLOCK // NP) * 64]),
                        txl.address_of(x_map),
                        txl.Cast("int32", tk * BLOCK),
                        txl.Cast("int32", tm * BLOCK + pp * (BLOCK // NP)),
                        txl.cuda.cvta_generic_to_shared(full.ptr_to([stage * NP + pp])),
                    )
                advance()

        with txl.If(warp == 8):
            with txl.Then():
                with txl.If(lane == 0), txl.Then():
                    produce(lambda: (tid < ntiles) & (it_n < S))
                txl.ptx["bar.warp.sync"](txl.uint32(FULL))
                txl.ptx["bar.arrive"](txl.uint32(0), txl.uint32(WARPS * 32))
                with txl.If(lane == 0), txl.Then():
                    produce(lambda: tid < ntiles)
            with txl.Else():
                txl.ptx.bar.sync(txl.uint32(0), txl.uint32(WARPS * 32))
                consume(xs, qs, tile, red, full, empty, stage, phase, tid, it_n, advance, q_map, qt_map, s, st, M, K, nmb, nkb, ntiles, t, lane, warp)

    return fp8_quant_act_tma


@functools.cache
def oneshot_kernel(ctas_per_sm: int):
    return oneshot_func(ctas_per_sm).compile()


class Zero:
    def __and__(self, o):
        return 0


def oneshot_func(ctas_per_sm: int):
    NC = 256
    TILE_BYTES = BLOCK * BLOCK * 2

    def host_prelude(params):
        x_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            x_map, "bfloat16", 2, params["x"].data,
            params["K"], params["M"], params["sr"] * 2, BLOCK, BLOCK // 8, 1, 1, 0, 0, 2, 0,
        )  # fmt: skip
        q_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            q_map, "uint8", 2, params["q"].data,
            params["K"], params["M"], params["K"], BLOCK, BLOCK, 1, 1, 0, 0, 2, 0,
        )  # fmt: skip
        qt_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            qt_map, "uint8", 2, params["qt"].data,
            params["M"], params["K"], params["M"], BLOCK, BLOCK, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        return x_map, q_map, qt_map

    @txl.kernel(warps=8, arch="sm_100a", min_blocks_per_sm=ctas_per_sm, grid=lambda p: p["nmb"] * p["nkb"], host_prelude=host_prelude)
    def fp8_quant_act_oneshot(
        x: txl.gptr[txl.bf16],
        q: txl.gptr[txl.u8],
        s: txl.gptr[txl.f32],
        qt: txl.gptr[txl.u8],
        st: txl.gptr[txl.f32],
        M: txl.i32,
        K: txl.i32,
        nmb: txl.i32,
        nkb: txl.i32,
        sr: txl.i64,
        *,
        host,
    ):
        x_map, q_map, qt_map = host
        b = txl.cta_id()
        t = txl.thread_id()
        lane = t & 31
        warp = txl.warp_id()
        smem = txl.smem_pool()
        xs = smem.alloc([BLOCK * 64], txl.u32, align=1024)  # [row][64 words] bf16 tile
        qs = smem.alloc([BLOCK * 32], txl.u32, align=1024)  # [row][32 words] rowwise FP8
        tile = smem.alloc([BLOCK * 32], txl.u32, align=1024)  # [k][128 B of m] FP8, TMA 128B swizzle
        red = smem.alloc([8], txl.f32, align=16)
        full = smem.alloc([8], "uint64", align=8)
        tm = txl.truncdiv(b, nkb)
        tk = txl.truncmod(b, nkb)
        with txl.If(t == 0), txl.Then():
            for pp in range(8):
                txl.ptx.mbarrier.init.shared.b64(full.ptr_to([pp]), txl.uint32(1))
            txl.ptx.fence.mbarrier_init.release.cluster()
            for pp in range(8):
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full.ptr_to([pp]), txl.uint32(TILE_BYTES // 8))
                txl.ptx[TMA_G2S](
                    xs.ptr_to([pp * 16 * 64]),
                    txl.address_of(x_map),
                    txl.Cast("int32", tk * BLOCK),
                    txl.Cast("int32", tm * BLOCK + pp * 16),
                    txl.cuda.cvta_generic_to_shared(full.ptr_to([pp])),
                )
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NC))
        tid = txl.local_scalar("int32", init=b)
        consume(xs, qs, tile, red, full, None, txl.int32(0), txl.int32(0), tid, Zero(), None, q_map, qt_map, s, st, M, K, nmb, nkb, b + 1, t, lane, warp, oneshot=True)

    return fp8_quant_act_oneshot


def consume(xs, qs, tile, red, full, empty, stage, phase, tid, it_n, advance, q_map, qt_map, s, st, M, K, nmb, nkb, ntiles, t, lane, warp, oneshot=False):
    NC = 256
    rg = lane >> 3  # row group within the warp
    cc = lane & 7  # 16 B column chunk within each 64-column half
    g = warp * 4 + rg  # this thread owns rows 4g .. 4g+3, columns 8cc.. and 64+8cc..
    data = txl.alloc_local([32], "uint32")  # [half][row][word], word = 2 adjacent columns
    ramax = txl.alloc_local([4], "float32")
    tmp = txl.local_scalar("uint32")
    packed = txl.alloc_local([2], "uint32")
    peer = txl.local_scalar("uint32")
    tm = txl.local_scalar("int32")
    tk = txl.local_scalar("int32")
    sbase = txl.local_scalar("int64")
    m0 = txl.local_scalar("int32")
    qb = txl.local_scalar("int32")
    mw = txl.alloc_local([4], "uint32")
    r2 = txl.local_scalar("uint32")
    h16 = txl.alloc_local([4], "uint16")
    tw = txl.alloc_local([8], "uint32")
    rot = txl.alloc_local([8], "uint32")
    bmax = txl.local_scalar("float32")
    ftmp = txl.local_scalar("float32")
    loop = txl.If(tid < ntiles) if oneshot else txl.While(tid < ntiles)
    with loop:
        if oneshot:
            then = txl.Then()
            then.__enter__()
        txl.assign(tm, txl.truncdiv(tid, nkb))
        txl.assign(tk, txl.truncmod(tid, nkb))
        txl.assign(m0, tm * BLOCK)
        txl.assign(qb, (it_n & 1) * (BLOCK * 32))
        # x_scale stored column-major: element (m, kb) at kb * M + m
        txl.assign(sbase, txl.cast(tk, "int64") * txl.cast(M, "int64") + txl.cast(m0, "int64"))
        tk0 = range_v("wait-full")
        txl.cuda.mbarrier_wait(full.ptr_to([stage * 8 + warp]), phase)
        end(tk0)
        tk1 = range_v("rowwise")
        for h in range(2):
            for r in range(4):
                base = h * 16 + r * 4
                txl.ptx["ld.shared.v4.b32"](
                    data[base], data[base + 1], data[base + 2], data[base + 3],
                    xs.ptr_to([stage * (BLOCK * 64) + (g * 4 + r) * 64 + h * 32 + cc * 4]),
                )
        if not oneshot:
            # order this thread's generic-proxy reads of the stage before the producer's next TMA write
            txl.ptx.fence.proxy.async_.shared__cta()
            txl.ptx["bar.warp.sync"](txl.uint32(FULL))
            with txl.If(lane == 0), txl.Then():
                txl.ptx.mbarrier.arrive.shared.b64(empty.ptr_to([stage]), txl.uint32(1))

        for r in range(4):
            m01 = txl.local_scalar("uint32")
            m23 = txl.local_scalar("uint32")
            a, b_ = r * 4, 16 + r * 4
            txl.ptx["max.xorsign.abs.bf16x2"](m01, data[a], data[a + 1])
            txl.ptx["max.xorsign.abs.bf16x2"](m23, data[a + 2], data[a + 3])
            txl.ptx["max.xorsign.abs.bf16x2"](m01, m01, m23)
            txl.ptx["max.xorsign.abs.bf16x2"](m23, data[b_], data[b_ + 1])
            txl.ptx["max.xorsign.abs.bf16x2"](m01, m01, m23)
            txl.ptx["max.xorsign.abs.bf16x2"](m23, data[b_ + 2], data[b_ + 3])
            txl.ptx["max.xorsign.abs.bf16x2"](m01, m01, m23)
            txl.ptx["max.xorsign.abs.bf16x2"](tmp, m01, txl.shift_right(m01, txl.uint32(16)))
            if r % 2 == 0:
                txl.assign(packed[r // 2], txl.bitwise_and(tmp, txl.uint32(0x7FFF)))
            else:
                txl.assign(packed[r // 2], txl.bitwise_or(packed[r // 2], txl.shift_left(txl.bitwise_and(tmp, txl.uint32(0x7FFF)), txl.uint32(16))))
        for d in (1, 2, 4):
            for hh in range(2):
                txl.ptx.shfl_sync.bfly.b32(peer, packed[hh], txl.uint32(d), txl.uint32(31), txl.uint32(FULL))
                txl.ptx.max.bf16x2(packed[hh], packed[hh], peer)
        for r in range(4):
            txl.assign(ramax[r], lo(packed[r // 2], "bfloat16") if r % 2 == 0 else hi(packed[r // 2], "bfloat16"))
        for r in range(4):
            ml = g * 4 + r
            scale, rcp = e8m0(ramax[r])
            txl.assign(r2, bf16x2_of(rcp))
            with txl.If((cc == 0) & (m0 + ml < M)), txl.Then():
                txl.ptx.st.global_.f32(s.ptr_to([sbase + txl.cast(ml, "int64")]), scale)
            for h in range(2):
                base = h * 16 + r * 4
                for u in range(4):
                    bmul(mw[u], data[base + u], r2)
                    cvt8(h16[u], mw[u])
                txl.ptx["st.shared.v2.b32"](qs.ptr_to([qb + ml * 32 + h * 16 + cc * 2]), pack16(h16[0], h16[1]), pack16(h16[2], h16[3]))
        end(tk1)

        tk2 = range_v("blk-amax")
        txl.assign(bmax, ramax[0])
        for jj in range(1, 4):
            txl.ptx.max.f32(bmax, bmax, ramax[jj])
        for d in (8, 16):
            txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret("uint32", bmax), txl.uint32(d), txl.uint32(31), txl.uint32(FULL))
            txl.ptx.max.f32(bmax, bmax, txl.reinterpret("float32", peer))
        with txl.If(lane == 0), txl.Then():
            txl.ptx.st.shared.f32(red.ptr_to([warp]), bmax)
        if not oneshot:
            # the previous tile's TMA stores must have read qs / tile before they are rewritten
            with txl.If(t == 0), txl.Then():
                txl.ptx.cp.async_.bulk.wait_group.read(0)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NC))
        # every warp's rowwise FP8 is in qs now: store it while the transposed half is built
        with txl.If(t == 0), txl.Then():
            txl.ptx.fence.proxy.async_.shared__cta()
            txl.ptx[TMA_S2G](txl.address_of(q_map), txl.Cast("int32", tk * BLOCK), txl.Cast("int32", m0), qs.ptr_to([qb]))
            txl.ptx.cp.async_.bulk.commit_group()
        txl.ptx.ld.shared.f32(bmax, red.ptr_to([0]))
        for w in range(1, 8):
            txl.ptx.ld.shared.f32(ftmp, red.ptr_to([w]))
            txl.ptx.max.f32(bmax, bmax, ftmp)
        bscale, brcp = e8m0(bmax)
        txl.assign(r2, bf16x2_of(brcp))
        with txl.If(t == 0), txl.Then():
            txl.ptx.st.global_.f32(st.ptr_to([tk * nmb + tm]), bscale)
        end(tk2)

        tk3 = range_v("t-smem")
        # transposed FP8 in TMA 128B-swizzle layout: byte (k, m) at k*128 + ((m/16) ^ (k%8))*16 + m%16.
        # Word T[j] = rows 4g..4g+3 of column 64h + 8cc + j. Lane cc stores T[(step + cc) % 8] at each
        # step, so k % 8 differs across the 8 lanes of a row group: 32 distinct banks per store.
        for h in range(2):
            for r in range(4):
                for u in range(4):
                    bmul(data[h * 16 + r * 4 + u], data[h * 16 + r * 4 + u], r2)
            for u in range(4):
                w0, w1, w2, w3 = (data[h * 16 + r * 4 + u] for r in range(4))
                for half, sel in ((0, 0x5410), (1, 0x7632)):
                    pa = txl.local_scalar("uint32")
                    pb = txl.local_scalar("uint32")
                    txl.ptx["prmt.b32"](pa, w0, w1, txl.uint32(sel))
                    txl.ptx["prmt.b32"](pb, w2, w3, txl.uint32(sel))
                    cvt8(h16[0], pa)
                    cvt8(h16[1], pb)
                    txl.assign(tw[2 * u + half], pack16(h16[0], h16[1]))
            # rotate left by cc: rot[i] = tw[(i + cc) % 8]
            for bit in (1, 2, 4):
                src = tw
                for i in range(8):
                    txl.assign(rot[i], txl.Select((cc & bit) != 0, src[(i + bit) % 8], src[i]))
                for i in range(8):
                    txl.assign(tw[i], rot[i])
            for step in range(8):
                jc = (step + cc) & 7
                k = h * 64 + cc * 8 + jc
                txl.ptx["st.shared.b32"](tile.ptr_to([k * 32 + txl.bitwise_xor(warp, jc) * 4 + rg]), tw[step])
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NC))
        with txl.If(t == 0), txl.Then():
            txl.ptx.fence.proxy.async_.shared__cta()
            txl.ptx[TMA_S2G](txl.address_of(qt_map), txl.Cast("int32", m0), txl.Cast("int32", tk * BLOCK), tile.ptr_to([0]))
            txl.ptx.cp.async_.bulk.commit_group()
        end(tk3)
        if oneshot:
            then.__exit__(None, None, None)
        else:
            advance()
    with txl.If(t == 0), txl.Then():
        txl.ptx.cp.async_.bulk.wait_group.read(0)


NSM = None


def num_sms():
    global NSM
    if NSM is None:
        NSM = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return NSM


def is_fast(x):
    M, K = x.shape
    return (x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.data_ptr() % 16 == 0
            and x.stride(0) % 8 == 0 and K % 16 == 0 and M % 16 == 0)


def fp8_quant_act_sm100(x: torch.Tensor):
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    nkb, nmb = (K + BLOCK - 1) // BLOCK, (M + BLOCK - 1) // BLOCK
    dev = x.device
    fast = M > 0 and K > 0 and is_fast(x)
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    # x_scale strides are unspecified by the contract; the fast path stores it column-major so that
    # each tile's 128 row scales are one contiguous run.
    s_store = torch.empty((nkb, M) if fast else (M, nkb), dtype=torch.float32, device=dev)
    s = s_store.T if fast else s_store
    qt = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=dev)
    st = torch.empty((nkb, nmb), dtype=torch.float32, device=dev)
    if M == 0 or K == 0:
        return q, s, qt, st
    sr, sc = x.stride()
    span = (M - 1) * sr + (K - 1) * sc + 1
    flat = x.as_strided((span,), (1,))
    outs = (q.view(torch.uint8).view(-1), s_store.view(-1), qt.view(torch.uint8).view(-1), st.view(-1))
    with tvm_ffi.use_torch_stream():
        if fast:
            ntiles = nmb * nkb
            if ntiles <= ONESHOT_CTAS_PER_SM * num_sms():
                # one resident wave: one CTA per tile, all loads in flight at launch
                oneshot_kernel(ONESHOT_CTAS_PER_SM)(flat, *outs, M, K, nmb, nkb, sr)
            else:
                grid_n = min(ntiles, CFG["ctas_per_sm"] * num_sms())
                fast_kernel(CFG["stages"], CFG["ctas_per_sm"])(flat, *outs, M, K, nmb, nkb, sr, grid_n)
        else:
            generic_kernel(DT[x.dtype])(flat, *outs, M, K, nmb, nkb, sr, sc)
    return q, s, qt, st
