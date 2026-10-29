"""
SM100 fp8_quant_grad in TIRx-lite: rowwise 1x128 FP8 of x and of x.T from one read of x.

Fast path (bf16, unit column stride, 16B-aligned rows, N % 8 == 0, M % 16 == 0) dispatches on the
128x128 tile count T relative to the SM count S:
  T < 4S   rr16: persistent, 2 CTAs/SM x 16 warps; warps 0-7 run the row phase while warps 8-15
           run the column phase of the same staged tile (shorter per-tile latency chain).
  T < 10S  rr8:  persistent, 3 CTAs/SM x 8 warps, row phase then column phase.
  else     clc:  one CTA per tile with cluster-launch-control work stealing, TMA (3-D map,
           128B swizzle) loads into a 2-stage mbarrier ring (dynamic load balance).
rr8/rr16 cp.async 16 B (zero-filled edges) into a 2-stage XOR-swizzled ring. Row phase: 8 lanes
per row (128 B coalesced stores); column phase: 8 lanes per column pair over 16-row groups, column
abs-max reduced in-warp. Scales use an exact integer E8M0 formula; quantization is an exact
power-of-two bf16x2 multiply plus cvt.rn.satfinite.e4m3x2.bf16x2.
Everything else (fp32/fp16, odd shapes, arbitrary strides) takes a scalar generic kernel.

Evolved with TIRx-harness: run grad, frontier member tilecount-dispatch-hybrid (commit 9c7c73d, 1.532x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools

import torch
import tvm_ffi

import tirx_kernels.tirx_lite as txl

__all__ = ["fp8_quant_grad_sm100"]

BLOCK = 128
NTHREADS = 256


def cdiv(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------- trace-time helpers


def i642(v):
    return txl.Cast("int64", v)


def rc(v, ty):
    return txl.reinterpret(ty, v)


def u322(v):
    return txl.uint32(v)


def scale_bits(amax):
    """
    E8M0 scale of one block as (scale f32 bits, reciprocal f32 bits), both uint32.

    scale = 2^ceil(log2(max(amax, 1e-4) / 448)). With 448 = 7 * 2^6 and c = m * 2^E
    (1 <= m < 2), c / 448 rounds up to 2^(E-8) when m <= 1.75 and to 2^(E-7) otherwise; adding
    0x1FFFFF carries into the exponent exactly when the mantissa field exceeds 0x600000.
    """
    c = txl.local_scalar("float32")
    txl.ptx.max.f32(c, amax, txl.float32(1e-4))
    sbits = txl.local_scalar(
        "uint32", init=txl.bitwise_and(rc(c, "uint32") + u322(0x1FFFFF), u322(0x7F800000)) - u322(8 << 23)
    )
    rbits = txl.local_scalar("uint32", init=u322(254 << 23) - sbits)
    return sbits, rbits


def bf16x2_rcp(rbits):
    """
    Reciprocal (an exact power of two) broadcast into both bf16 halves.
    """
    hi = txl.shift_right(rbits, u322(16))
    return txl.local_scalar("uint32", init=txl.bitwise_or(hi, txl.shift_left(hi, u322(16))))


def q_bf16x2(word, rcp2):
    """
    Two bf16 -> two e4m3 bytes (low element in the low byte) as uint16.
    """
    s = txl.local_scalar("uint32")
    txl.ptx.mul.rn.bf16x2(s, word, rcp2)
    out = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(out, s)
    return out


def pack16(lo, hi):
    out = txl.local_scalar("uint32")
    txl.ptx.mov.b32(out, lo, hi)
    return out


# ---------------------------------------------------------------- fast bf16 tile kernel


def phys_chunk(r, c):
    """
    Swizzled 16-byte chunk of logical (row r, chunk c) in a 256-byte staged row.

    Row phase (8 lanes of a row read chunks k and k+8) and column phase (8 lanes = 8 row groups
    of 16 rows read one 4-word chunk) are both bank-conflict-free.
    """
    return c ^ ((c >> 3) & 1) ^ ((r >> 4) & 7)


def iket_range(name):
    """
    Lane-0 IKET range (stripped by the plain tirx pipeline).
    """
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
    with txl.If(txl.lane_id() == 0), txl.Then():
        txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def iket_end(token):
    txl.cuda.iket.range_end(token[0])


TILE_WORDS = BLOCK * 64


class TileWalk:
    """
    (tm, tn) of tile index L = cta + i * num_ctas, advanced incrementally (no per-tile division).
    """

    def __init__(self, L, nbn, dm, dn):
        self.tm = txl.local_scalar("int32", init=txl.truncdiv(L, nbn))
        self.tn = txl.local_scalar("int32", init=txl.truncmod(L, nbn))
        self.nbn, self.dm, self.dn = nbn, dm, dn

    def advance(self):
        txl.assign(self.tn, self.tn + self.dn)
        txl.assign(self.tm, self.tm + self.dm)
        with txl.If(self.tn >= self.nbn), txl.Then():
            txl.assign(self.tn, self.tn - self.nbn)
            txl.assign(self.tm, self.tm + 1)


TMA_3D = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
HALF_WORDS = BLOCK * 32  # one 64-column half of a staged tile (16 KB)


def x_tensor_map(params):
    """
    3-D view of x: (cols, row group g = r // 16, row in group e = r % 16) with box (64, 8, 16),
    so the staged row order is s = e * 8 + g and the 128B swizzle (chunk ^ s % 8) separates both
    the 8 lanes of a row (row phase) and the 8 row groups of a column (column phase).
    """
    x = params["x"]
    M, N, ld = params["M"], params["N"], params["ld"]
    tmap = txl.stack_alloca("tensormap", 1)
    txl.cu_tensor_map_encode_tiled(
        tmap, "bfloat16", 3, x.data,
        N, M // 16, 16,
        ld * 32, ld * 2,
        64, 8, 16,
        1, 1, 1,
        0, 3, 2, 0,
    )  # fmt: skip
    return tmap


def build_rr8():
    @txl.kernel(warps=NTHREADS // 32, arch="sm_100a", grid="num_ctas", min_blocks_per_sm=3)
    def fp8_quant_grad_persistent(
        x: txl.gptr[txl.bf16],
        xq: txl.gptr[txl.u8],
        xs: txl.gptr[txl.i32],
        xtq: txl.gptr[txl.u8],
        xts: txl.gptr[txl.i32],
        M: txl.i32,
        N: txl.i32,
        ld: txl.i64,
        off: txl.i64,
        num_ctas: txl.i32,
    ):
        cta = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        pool = txl.smem_pool()
        tiles = pool.alloc([2 * TILE_WORDS], "uint32", align=128)
        nbn = txl.local_scalar("int32", init=(N + 127) >> 7)
        nbm = txl.local_scalar("int32", init=(M + 127) >> 7)
        count = txl.local_scalar("int32", init=txl.truncdiv(nbn * nbm - cta + num_ctas - 1, num_ctas))
        dm = txl.local_scalar("int32", init=txl.truncdiv(num_ctas, nbn))
        dn = txl.local_scalar("int32", init=txl.truncmod(num_ctas, nbn))
        k = lane & 7
        r0 = warp * 4 + (lane >> 3)  # row of this thread in iteration it: r0 + 32 * it
        rg = lane & 7
        pl = lane >> 3
        M64 = i642(M)
        N64 = i642(N)

        # per-thread constant offsets (tile independent)
        xro = [txl.local_scalar("int64", init=i642(r0 + 32 * it) * ld + i642(k * 8)) for it in range(4)]
        # smem word offsets for row-phase / copy chunks: row r, chunk k (+8)
        soff = []
        for it in range(4):
            r = r0 + 32 * it
            g = (r >> 4) & 7
            soff.append([txl.local_scalar("int32", init=r * 64 + (((k + 8 * h) ^ h ^ g) << 2)) for h in range(2)])
        # column phase: pair p = warp*4 + pl + 32*it2, rows rg*16 + e
        coff = []
        for it2 in range(2):
            p = warp * 4 + pl + 32 * it2
            c = p >> 2
            coff.append(txl.local_scalar("int32", init=rg * 16 * 64 + ((c ^ ((c >> 3) & 1) ^ rg) << 2) + (p & 3)))

        def issue(tm_, tn_, stage):
            sb = stage * TILE_WORDS
            R0 = tm_ * 128
            C0 = tn_ * 128
            tbase = txl.local_scalar("int64", init=off + i642(R0) * ld + i642(C0))
            interior = (R0 + 128 <= M) & (C0 + 128 <= N)
            with txl.If(interior):
                with txl.Then():
                    for it in range(4):
                        for h in range(2):
                            txl.ptx["cp.async.cg.shared.global"](
                                tiles.ptr_to([sb + soff[it][h]]),
                                txl.address_of(x[tbase + xro[it] + 64 * h]),
                                16,
                                u322(16),
                            )
                with txl.Else():
                    for it in range(4):
                        for h in range(2):
                            ok = (R0 + r0 + 32 * it < M) & (C0 + k * 8 + 64 * h < N)
                            txl.ptx["cp.async.cg.shared.global"](
                                tiles.ptr_to([sb + soff[it][h]]),
                                txl.address_of(x[off + txl.Select(ok, tbase - off + xro[it] + 64 * h, i642(0))]),
                                16,
                                txl.Select(ok, u322(16), u322(0)),
                            )

        ahead = TileWalk(cta, nbn, dm, dn)
        for j in range(2 - 1):
            with txl.If(j < count), txl.Then():
                issue(ahead.tm, ahead.tn, j)
                ahead.advance()
            txl.ptx["cp.async.commit_group"]()

        cur = TileWalk(cta, nbn, dm, dn)
        stage = txl.local_scalar("int32", init=0)
        istage = txl.local_scalar("int32", init=2 - 1)
        w = txl.alloc_local([8], "uint32")
        v = txl.alloc_local([16], "uint32")
        with txl.serial(count) as i:
            txl.ptx["cp.async.wait_group"](2 - 2)
            txl.ptx.bar.sync(u322(0), u322(NTHREADS))
            with txl.If(i + 2 - 1 < count), txl.Then():
                issue(ahead.tm, ahead.tn, istage)
                ahead.advance()
            txl.ptx["cp.async.commit_group"]()
            tm = cur.tm
            tn = cur.tn
            sb = stage * TILE_WORDS
            R0 = tm * 128
            C0 = tn * 128

            # ---- row phase: 8 lanes per row; lane k holds elements 8k.. and 64+8k..
            qtile = txl.local_scalar("int64", init=i642(R0) * N64 + i642(C0))
            for it in range(4):
                grow = R0 + r0 + 32 * it
                for h in range(2):
                    txl.ptx.ld.shared.v4.b32(w[4 * h], w[4 * h + 1], w[4 * h + 2], w[4 * h + 3], tiles.ptr_to([sb + soff[it][h]]))
                m = txl.local_scalar("uint32", init=w[0])
                for e in range(1, 8):
                    txl.ptx.max.xorsign.abs.bf16x2(m, m, w[e])
                a = txl.local_scalar("uint32", init=txl.bitwise_and(m, u322(0x7FFF7FFF)))
                txl.ptx.max.u32(a, txl.bitwise_and(a, u322(0xFFFF)), txl.shift_right(a, u322(16)))
                peer = txl.local_scalar("uint32")
                for d in (1, 2, 4):
                    txl.ptx.shfl_sync.bfly.b32(peer, a, u322(d), u322(31), u322(0xFFFFFFFF))
                    txl.ptx.max.u32(a, a, peer)
                sbits, rbits = scale_bits(rc(txl.shift_left(a, u322(16)), "float32"))
                rcp2 = bf16x2_rcp(rbits)
                o = []
                for h in range(2):
                    o.append(pack16(q_bf16x2(w[4 * h], rcp2), q_bf16x2(w[4 * h + 1], rcp2)))
                    o.append(pack16(q_bf16x2(w[4 * h + 2], rcp2), q_bf16x2(w[4 * h + 3], rcp2)))
                with txl.If(grow < M), txl.Then():
                    qa = qtile + i642(r0 + 32 * it) * N64 + i642(k * 8)
                    with txl.If(C0 + 128 <= N):
                        with txl.Then():
                            txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa]), o[0], o[1])
                            txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64]), o[2], o[3])
                        with txl.Else():
                            for h in range(2):
                                with txl.If(C0 + k * 8 + 64 * h < N), txl.Then():
                                    txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64 * h]), o[2 * h], o[2 * h + 1])
                    with txl.If(k == 0), txl.Then():
                        txl.ptx.st.global_.b32(txl.address_of(xs[i642(grow) * i642(nbn) + i642(tn)]), sbits)

            # ---- column phase: lane = pl * 8 + rg; pair p = warp*4 + pl (+32), rows rg*16..+15
            for it2 in range(2):
                p = warp * 4 + pl + 32 * it2
                for e in range(16):
                    txl.ptx.ld.shared.b32(v[e], tiles.ptr_to([sb + coff[it2] + e * 64]))
                cm = txl.local_scalar("uint32", init=v[0])
                for e in range(1, 16):
                    txl.ptx.max.xorsign.abs.bf16x2(cm, cm, v[e])
                txl.assign(cm, txl.bitwise_and(cm, u322(0x7FFF7FFF)))
                pm = txl.local_scalar("uint32")
                for d in (1, 2, 4):
                    txl.ptx.shfl_sync.bfly.b32(pm, cm, u322(d), u322(31), u322(0xFFFFFFFF))
                    txl.ptx.max.bf16x2(cm, cm, pm)
                s_lo, r_lo = scale_bits(rc(txl.shift_left(cm, u322(16)), "float32"))
                s_hi, r_hi = scale_bits(rc(txl.bitwise_and(cm, u322(0xFFFF0000)), "float32"))
                rl2 = bf16x2_rcp(r_lo)
                rh2 = bf16x2_rcp(r_hi)
                lo = []
                hi = []
                for e in range(0, 16, 2):
                    plo = txl.local_scalar("uint32")
                    phi = txl.local_scalar("uint32")
                    txl.ptx.prmt.b32(plo, v[e], v[e + 1], u322(0x5410))
                    txl.ptx.prmt.b32(phi, v[e], v[e + 1], u322(0x7632))
                    lo.append(q_bf16x2(plo, rl2))
                    hi.append(q_bf16x2(phi, rh2))
                lw = [pack16(lo[2 * j], lo[2 * j + 1]) for j in range(4)]
                hw = [pack16(hi[2 * j], hi[2 * j + 1]) for j in range(4)]
                col = C0 + 2 * p
                with txl.If((col < N) & (R0 + rg * 16 < M)), txl.Then():
                    ta = txl.local_scalar("int64", init=i642(col) * M64 + i642(R0 + rg * 16))
                    txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta]), lw[0], lw[1], lw[2], lw[3])
                    txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta + M64]), hw[0], hw[1], hw[2], hw[3])
                    with txl.If(rg == 0), txl.Then():
                        sa = txl.local_scalar("int64", init=i642(col) * i642(nbm) + i642(tm))
                        txl.ptx.st.global_.b32(txl.address_of(xts[sa]), s_lo)
                        txl.ptx.st.global_.b32(txl.address_of(xts[sa + i642(nbm)]), s_hi)
            cur.advance()
            txl.assign(stage, txl.Select(stage == 2 - 1, 0, stage + 1))
            txl.assign(istage, txl.Select(istage == 2 - 1, 0, istage + 1))
        txl.ptx["cp.async.wait_group"](0)

    return fp8_quant_grad_persistent


def build_rr16():
    @txl.kernel(warps=512 // 32, arch="sm_100a", grid="num_ctas", min_blocks_per_sm=2)
    def fp8_quant_grad_persistent(
        x: txl.gptr[txl.bf16],
        xq: txl.gptr[txl.u8],
        xs: txl.gptr[txl.i32],
        xtq: txl.gptr[txl.u8],
        xts: txl.gptr[txl.i32],
        M: txl.i32,
        N: txl.i32,
        ld: txl.i64,
        off: txl.i64,
        num_ctas: txl.i32,
    ):
        cta = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        pool = txl.smem_pool()
        tiles = pool.alloc([2 * TILE_WORDS], "uint32", align=128)
        nbn = txl.local_scalar("int32", init=(N + 127) >> 7)
        nbm = txl.local_scalar("int32", init=(M + 127) >> 7)
        count = txl.local_scalar("int32", init=txl.truncdiv(nbn * nbm - cta + num_ctas - 1, num_ctas))
        dm = txl.local_scalar("int32", init=txl.truncdiv(num_ctas, nbn))
        dn = txl.local_scalar("int32", init=txl.truncmod(num_ctas, nbn))
        k = lane & 7
        role = warp >> 3  # 0: row phase, 1: column phase (both read the same staged tile)
        wr = warp & 7
        cr0 = warp * 4 + (lane >> 3)  # copy row of this thread: cr0 + 64 * it
        r0 = wr * 4 + (lane >> 3)  # row-phase row: r0 + 32 * it
        rg = lane & 7
        pl = lane >> 3
        M64 = i642(M)
        N64 = i642(N)

        # per-thread constant offsets (tile independent)
        xro = [txl.local_scalar("int64", init=i642(cr0 + 64 * it) * ld + i642(k * 8)) for it in range(2)]
        cso = []
        for it in range(2):
            r = cr0 + 64 * it
            g = (r >> 4) & 7
            cso.append([txl.local_scalar("int32", init=r * 64 + (((k + 8 * h) ^ h ^ g) << 2)) for h in range(2)])
        # smem word offsets for row-phase / copy chunks: row r, chunk k (+8)
        soff = []
        for it in range(4):
            r = r0 + 32 * it
            g = (r >> 4) & 7
            soff.append([txl.local_scalar("int32", init=r * 64 + (((k + 8 * h) ^ h ^ g) << 2)) for h in range(2)])
        # column phase: pair p = warp*4 + pl + 32*it2, rows rg*16 + e
        coff = []
        for it2 in range(2):
            p = wr * 4 + pl + 32 * it2
            c = p >> 2
            coff.append(txl.local_scalar("int32", init=rg * 16 * 64 + ((c ^ ((c >> 3) & 1) ^ rg) << 2) + (p & 3)))

        def issue(tm_, tn_, stage):
            sb = stage * TILE_WORDS
            R0 = tm_ * 128
            C0 = tn_ * 128
            tbase = txl.local_scalar("int64", init=off + i642(R0) * ld + i642(C0))
            interior = (R0 + 128 <= M) & (C0 + 128 <= N)
            with txl.If(interior):
                with txl.Then():
                    for it in range(2):
                        for h in range(2):
                            txl.ptx["cp.async.cg.shared.global"](
                                tiles.ptr_to([sb + cso[it][h]]),
                                txl.address_of(x[tbase + xro[it] + 64 * h]),
                                16,
                                u322(16),
                            )
                with txl.Else():
                    for it in range(2):
                        for h in range(2):
                            ok = (R0 + cr0 + 64 * it < M) & (C0 + k * 8 + 64 * h < N)
                            txl.ptx["cp.async.cg.shared.global"](
                                tiles.ptr_to([sb + cso[it][h]]),
                                txl.address_of(x[off + txl.Select(ok, tbase - off + xro[it] + 64 * h, i642(0))]),
                                16,
                                txl.Select(ok, u322(16), u322(0)),
                            )

        ahead = TileWalk(cta, nbn, dm, dn)
        for j in range(2 - 1):
            with txl.If(j < count), txl.Then():
                issue(ahead.tm, ahead.tn, j)
                ahead.advance()
            txl.ptx["cp.async.commit_group"]()

        cur = TileWalk(cta, nbn, dm, dn)
        stage = txl.local_scalar("int32", init=0)
        istage = txl.local_scalar("int32", init=2 - 1)
        w = txl.alloc_local([8], "uint32")
        v = txl.alloc_local([16], "uint32")
        with txl.serial(count) as i:
            txl.ptx["cp.async.wait_group"](2 - 2)
            txl.ptx.bar.sync(u322(0), u322(512))
            with txl.If(i + 2 - 1 < count), txl.Then():
                issue(ahead.tm, ahead.tn, istage)
                ahead.advance()
            txl.ptx["cp.async.commit_group"]()
            tm = cur.tm
            tn = cur.tn
            sb = stage * TILE_WORDS
            R0 = tm * 128
            C0 = tn * 128

            with txl.If(role == 0):
                with txl.Then():
                    # ---- row phase: 8 lanes per row; lane k holds elements 8k.. and 64+8k..
                    qtile = txl.local_scalar("int64", init=i642(R0) * N64 + i642(C0))
                    for it in range(4):
                        grow = R0 + r0 + 32 * it
                        for h in range(2):
                            txl.ptx.ld.shared.v4.b32(w[4 * h], w[4 * h + 1], w[4 * h + 2], w[4 * h + 3], tiles.ptr_to([sb + soff[it][h]]))
                        m = txl.local_scalar("uint32", init=w[0])
                        for e in range(1, 8):
                            txl.ptx.max.xorsign.abs.bf16x2(m, m, w[e])
                        a = txl.local_scalar("uint32", init=txl.bitwise_and(m, u322(0x7FFF7FFF)))
                        txl.ptx.max.u32(a, txl.bitwise_and(a, u322(0xFFFF)), txl.shift_right(a, u322(16)))
                        peer = txl.local_scalar("uint32")
                        for d in (1, 2, 4):
                            txl.ptx.shfl_sync.bfly.b32(peer, a, u322(d), u322(31), u322(0xFFFFFFFF))
                            txl.ptx.max.u32(a, a, peer)
                        sbits, rbits = scale_bits(rc(txl.shift_left(a, u322(16)), "float32"))
                        rcp2 = bf16x2_rcp(rbits)
                        o = []
                        for h in range(2):
                            o.append(pack16(q_bf16x2(w[4 * h], rcp2), q_bf16x2(w[4 * h + 1], rcp2)))
                            o.append(pack16(q_bf16x2(w[4 * h + 2], rcp2), q_bf16x2(w[4 * h + 3], rcp2)))
                        with txl.If(grow < M), txl.Then():
                            qa = qtile + i642(r0 + 32 * it) * N64 + i642(k * 8)
                            with txl.If(C0 + 128 <= N):
                                with txl.Then():
                                    txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa]), o[0], o[1])
                                    txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64]), o[2], o[3])
                                with txl.Else():
                                    for h in range(2):
                                        with txl.If(C0 + k * 8 + 64 * h < N), txl.Then():
                                            txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64 * h]), o[2 * h], o[2 * h + 1])
                            with txl.If(k == 0), txl.Then():
                                txl.ptx.st.global_.b32(txl.address_of(xs[i642(grow) * i642(nbn) + i642(tn)]), sbits)
                with txl.Else():
                    # ---- column phase: lane = pl * 8 + rg; pair p = warp*4 + pl (+32), rows rg*16..+15
                    for it2 in range(2):
                        p = wr * 4 + pl + 32 * it2
                        for e in range(16):
                            txl.ptx.ld.shared.b32(v[e], tiles.ptr_to([sb + coff[it2] + e * 64]))
                        cm = txl.local_scalar("uint32", init=v[0])
                        for e in range(1, 16):
                            txl.ptx.max.xorsign.abs.bf16x2(cm, cm, v[e])
                        txl.assign(cm, txl.bitwise_and(cm, u322(0x7FFF7FFF)))
                        pm = txl.local_scalar("uint32")
                        for d in (1, 2, 4):
                            txl.ptx.shfl_sync.bfly.b32(pm, cm, u322(d), u322(31), u322(0xFFFFFFFF))
                            txl.ptx.max.bf16x2(cm, cm, pm)
                        s_lo, r_lo = scale_bits(rc(txl.shift_left(cm, u322(16)), "float32"))
                        s_hi, r_hi = scale_bits(rc(txl.bitwise_and(cm, u322(0xFFFF0000)), "float32"))
                        rl2 = bf16x2_rcp(r_lo)
                        rh2 = bf16x2_rcp(r_hi)
                        lo = []
                        hi = []
                        for e in range(0, 16, 2):
                            plo = txl.local_scalar("uint32")
                            phi = txl.local_scalar("uint32")
                            txl.ptx.prmt.b32(plo, v[e], v[e + 1], u322(0x5410))
                            txl.ptx.prmt.b32(phi, v[e], v[e + 1], u322(0x7632))
                            lo.append(q_bf16x2(plo, rl2))
                            hi.append(q_bf16x2(phi, rh2))
                        lw = [pack16(lo[2 * j], lo[2 * j + 1]) for j in range(4)]
                        hw = [pack16(hi[2 * j], hi[2 * j + 1]) for j in range(4)]
                        col = C0 + 2 * p
                        with txl.If((col < N) & (R0 + rg * 16 < M)), txl.Then():
                            ta = txl.local_scalar("int64", init=i642(col) * M64 + i642(R0 + rg * 16))
                            txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta]), lw[0], lw[1], lw[2], lw[3])
                            txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta + M64]), hw[0], hw[1], hw[2], hw[3])
                            with txl.If(rg == 0), txl.Then():
                                sa = txl.local_scalar("int64", init=i642(col) * i642(nbm) + i642(tm))
                                txl.ptx.st.global_.b32(txl.address_of(xts[sa]), s_lo)
                                txl.ptx.st.global_.b32(txl.address_of(xts[sa + i642(nbm)]), s_hi)
            cur.advance()
            txl.assign(stage, txl.Select(stage == 2 - 1, 0, stage + 1))
            txl.assign(istage, txl.Select(istage == 2 - 1, 0, istage + 1))
        txl.ptx["cp.async.wait_group"](0)

    return fp8_quant_grad_persistent


def build_clc():
    @txl.kernel(
        warps=NTHREADS // 32,
        arch="sm_100a",
        grid=lambda p: ((p["N"] + 127) // 128) * ((p["M"] + 127) // 128),
        min_blocks_per_sm=2,
        host_prelude=x_tensor_map,
    )
    def fp8_quant_grad_clc(
        x: txl.gptr[txl.bf16],
        xq: txl.gptr[txl.u8],
        xs: txl.gptr[txl.i32],
        xtq: txl.gptr[txl.u8],
        xts: txl.gptr[txl.i32],
        M: txl.i32,
        N: txl.i32,
        ld: txl.i64,
        *,
        host,
    ):
        tmap = host
        cta = txl.cta_id()
        t = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        pool = txl.smem_pool()
        tiles = pool.alloc([2 * TILE_WORDS], "uint32", align=1024)
        full = pool.alloc([2], "uint64", align=8)
        clc_bar = pool.alloc([1], "uint64", align=8)
        clc_resp = pool.alloc([4], "uint32", align=16)
        slot = pool.alloc([2], "int32", align=8)
        nbn = txl.local_scalar("int32", init=(N + 127) >> 7)
        nbm = txl.local_scalar("int32", init=(M + 127) >> 7)
        k = lane & 7
        r0 = warp * 4 + (lane >> 3)
        rg = lane & 7
        pl = lane >> 3
        M64 = i642(M)
        N64 = i642(N)

        # staged word offsets: half h, staged row s, swizzled chunk
        soff = []
        for it in range(4):
            r = r0 + 32 * it
            srow = (r & 15) * 8 + (r >> 4)
            soff.append([txl.local_scalar("int32", init=h * HALF_WORDS + srow * 32 + ((k ^ (srow & 7)) << 2)) for h in range(2)])
        coff = []
        for it2 in range(2):
            p = warp * 4 + pl + 32 * it2
            coff.append(txl.local_scalar("int32", init=(p >> 5) * HALF_WORDS + rg * 32 + ((((p >> 2) & 7) ^ rg) << 2) + (p & 3)))

        with txl.If(t == 0), txl.Then():
            txl.ptx.prefetch.tensormap(txl.address_of(tmap))
            for j in range(2):
                txl.ptx.mbarrier.init.shared.b64(full.ptr_to([j]), u322(1))
            txl.ptx.mbarrier.init.shared.b64(clc_bar.ptr_to([0]), u322(1))
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(u322(0), u322(NTHREADS))

        def issue(L, stg):
            tm_ = txl.truncdiv(L, nbn)
            tn_ = L - tm_ * nbn
            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full.ptr_to([stg]), u322(BLOCK * BLOCK * 2))
            for h in range(2):
                txl.ptx[TMA_3D](
                    tiles.ptr_to([stg * TILE_WORDS + h * HALF_WORDS]),
                    txl.address_of(tmap),
                    txl.Cast("int32", tn_ * 128 + 64 * h),
                    txl.Cast("int32", tm_ * 8),
                    txl.int32(0),
                    full.ptr_to([stg]),
                )

        def request():
            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(clc_bar.ptr_to([0]), u322(16))
            txl.ptx["clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128"](
                clc_resp.ptr_to([0]), clc_bar.ptr_to([0])
            )

        cur = txl.local_scalar("int32", init=cta)
        with txl.If(t == 0), txl.Then():
            issue(cur, 0)
            request()

        i = txl.local_scalar("int32", init=0)
        nxt = txl.local_scalar("uint32")
        w = txl.alloc_local([8], "uint32")
        v = txl.alloc_local([16], "uint32")
        with txl.While(cur >= 0):
            stage = i & 1
            with txl.If(t == 0), txl.Then():
                txl.cuda.mbarrier_wait(clc_bar.ptr_to([0]), i & 1)
                txl.query_cancel_first_ctaid_x(nxt, clc_resp.ptr_to([0]))
                with txl.If(nxt != u322(0xFFFFFFFF)):
                    with txl.Then():
                        txl.ptx.st.shared.b32(slot.ptr_to([stage ^ 1]), txl.Cast("int32", nxt))
                        issue(txl.Cast("int32", nxt), stage ^ 1)
                        request()
                    with txl.Else():
                        txl.ptx.st.shared.b32(slot.ptr_to([stage ^ 1]), txl.int32(-1))
            txl.cuda.mbarrier_wait(full.ptr_to([stage]), (i >> 1) & 1)
            tm = txl.local_scalar("int32", init=txl.truncdiv(cur, nbn))
            tn = txl.local_scalar("int32", init=cur - tm * nbn)
            sb = stage * TILE_WORDS
            R0 = tm * 128
            C0 = tn * 128

            qtile = txl.local_scalar("int64", init=i642(R0) * N64 + i642(C0))
            for it in range(4):
                grow = R0 + r0 + 32 * it
                for h in range(2):
                    txl.ptx.ld.shared.v4.b32(w[4 * h], w[4 * h + 1], w[4 * h + 2], w[4 * h + 3], tiles.ptr_to([sb + soff[it][h]]))
                m = txl.local_scalar("uint32", init=w[0])
                for e in range(1, 8):
                    txl.ptx.max.xorsign.abs.bf16x2(m, m, w[e])
                a = txl.local_scalar("uint32", init=txl.bitwise_and(m, u322(0x7FFF7FFF)))
                txl.ptx.max.u32(a, txl.bitwise_and(a, u322(0xFFFF)), txl.shift_right(a, u322(16)))
                peer = txl.local_scalar("uint32")
                for d in (1, 2, 4):
                    txl.ptx.shfl_sync.bfly.b32(peer, a, u322(d), u322(31), u322(0xFFFFFFFF))
                    txl.ptx.max.u32(a, a, peer)
                sbits, rbits = scale_bits(rc(txl.shift_left(a, u322(16)), "float32"))
                rcp2 = bf16x2_rcp(rbits)
                o = []
                for h in range(2):
                    o.append(pack16(q_bf16x2(w[4 * h], rcp2), q_bf16x2(w[4 * h + 1], rcp2)))
                    o.append(pack16(q_bf16x2(w[4 * h + 2], rcp2), q_bf16x2(w[4 * h + 3], rcp2)))
                with txl.If(grow < M), txl.Then():
                    qa = qtile + i642(r0 + 32 * it) * N64 + i642(k * 8)
                    with txl.If(C0 + 128 <= N):
                        with txl.Then():
                            txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa]), o[0], o[1])
                            txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64]), o[2], o[3])
                        with txl.Else():
                            for h in range(2):
                                with txl.If(C0 + k * 8 + 64 * h < N), txl.Then():
                                    txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64 * h]), o[2 * h], o[2 * h + 1])
                    with txl.If(k == 0), txl.Then():
                        txl.ptx.st.global_.b32(txl.address_of(xs[i642(grow) * i642(nbn) + i642(tn)]), sbits)

            for it2 in range(2):
                p = warp * 4 + pl + 32 * it2
                for e in range(16):
                    txl.ptx.ld.shared.b32(v[e], tiles.ptr_to([sb + coff[it2] + e * 256]))
                cm = txl.local_scalar("uint32", init=v[0])
                for e in range(1, 16):
                    txl.ptx.max.xorsign.abs.bf16x2(cm, cm, v[e])
                txl.assign(cm, txl.bitwise_and(cm, u322(0x7FFF7FFF)))
                pm = txl.local_scalar("uint32")
                for d in (1, 2, 4):
                    txl.ptx.shfl_sync.bfly.b32(pm, cm, u322(d), u322(31), u322(0xFFFFFFFF))
                    txl.ptx.max.bf16x2(cm, cm, pm)
                s_lo, r_lo = scale_bits(rc(txl.shift_left(cm, u322(16)), "float32"))
                s_hi, r_hi = scale_bits(rc(txl.bitwise_and(cm, u322(0xFFFF0000)), "float32"))
                rl2 = bf16x2_rcp(r_lo)
                rh2 = bf16x2_rcp(r_hi)
                lo = []
                hi = []
                for e in range(0, 16, 2):
                    plo = txl.local_scalar("uint32")
                    phi = txl.local_scalar("uint32")
                    txl.ptx.prmt.b32(plo, v[e], v[e + 1], u322(0x5410))
                    txl.ptx.prmt.b32(phi, v[e], v[e + 1], u322(0x7632))
                    lo.append(q_bf16x2(plo, rl2))
                    hi.append(q_bf16x2(phi, rh2))
                lw = [pack16(lo[2 * j], lo[2 * j + 1]) for j in range(4)]
                hw = [pack16(hi[2 * j], hi[2 * j + 1]) for j in range(4)]
                col = C0 + 2 * p
                with txl.If((col < N) & (R0 + rg * 16 < M)), txl.Then():
                    ta = txl.local_scalar("int64", init=i642(col) * M64 + i642(R0 + rg * 16))
                    txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta]), lw[0], lw[1], lw[2], lw[3])
                    txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta + M64]), hw[0], hw[1], hw[2], hw[3])
                    with txl.If(rg == 0), txl.Then():
                        sa = txl.local_scalar("int64", init=i642(col) * i642(nbm) + i642(tm))
                        txl.ptx.st.global_.b32(txl.address_of(xts[sa]), s_lo)
                        txl.ptx.st.global_.b32(txl.address_of(xts[sa + i642(nbm)]), s_hi)
            txl.ptx.bar.sync(u322(0), u322(NTHREADS))
            txl.ptx.ld.shared.b32(cur, slot.ptr_to([stage ^ 1]))
            txl.assign(i, i + 1)

    return fp8_quant_grad_clc


RR8_CTAS_PER_SM = 3
RR16_CTAS_PER_SM = 2


@functools.cache
def rr8_kernel():
    return build_rr8().compile()


@functools.cache
def rr16_kernel():
    return build_rr16().compile()


@functools.cache
def clc_kernel():
    return build_clc().compile()


NUM_SMS = {}


def num_sms(dev):
    n = NUM_SMS.get(dev)
    if n is None:
        n = NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return n


# ---------------------------------------------------------------- generic scalar kernel


def load_f32(x, dtype, idx):
    """
    One element of x widened to f32.
    """
    if dtype == "float32":
        v = txl.local_scalar("float32")
        txl.ptx.ld.global_.f32(v, txl.address_of(x[idx]))
        return v
    raw = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(raw, txl.address_of(x[idx]))
    if dtype == "bfloat16":
        return rc(txl.shift_left(txl.Cast("uint32", raw), u322(16)), ty="float32")
    f = txl.local_scalar("float32")
    txl.ptx.cvt.f32.f16(f, raw)
    return f


def q_f32(val, rcp):
    s = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(s, val, rcp)
    out = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(out, txl.float32(0.0), s)
    return out


@functools.cache
def generic_kernel(dtype):
    PITCH = BLOCK + 1

    @txl.kernel(warps=NTHREADS // 32, arch="sm_100a", grid=lambda p: [(p["N"] + 127) // 128, (p["M"] + 127) // 128])
    def fp8_quant_grad_generic(
        x: txl.gptr[dtype],
        xq: txl.gptr[txl.u8],
        xs: txl.gptr[txl.i32],
        xtq: txl.gptr[txl.u8],
        xts: txl.gptr[txl.i32],
        M: txl.i32,
        N: txl.i32,
        s0: txl.i64,
        s1: txl.i64,
        off: txl.i64,
    ):
        cid = txl.cta_id()
        tn = cid[0]
        tm = cid[1]
        t = txl.thread_id()
        pool = txl.smem_pool()
        tile = pool.alloc([BLOCK * PITCH], "float32", align=16)
        red = pool.alloc([2 * BLOCK], "float32", align=16)
        nbn = (N + 127) // 128
        nbm = (M + 127) // 128
        R0 = tm * 128
        C0 = tn * 128

        with txl.serial(64) as it:
            e = it * NTHREADS + t
            rr = e >> 7
            c = e & 127
            val = txl.local_scalar("float32", init=txl.float32(0.0))
            with txl.If(R0 + rr < M), txl.Then():
                with txl.If(C0 + c < N), txl.Then():
                    txl.assign(val, load_f32(x, dtype, off + i642(R0 + rr) * s0 + i642(C0 + c) * s1))
            txl.ptx.st.shared.f32(txl.address_of(tile[rr * PITCH + c]), val)
        txl.ptx.bar.sync(u322(0), u322(NTHREADS))

        # rowwise: thread (row r, half h) covers 64 columns
        r = t >> 1
        h = t & 1
        amax = txl.local_scalar("float32", init=txl.float32(0.0))
        tmp = txl.local_scalar("float32")
        with txl.serial(64) as i:
            txl.ptx.ld.shared.f32(tmp, txl.address_of(tile[r * PITCH + h * 64 + i]))
            txl.ptx.max.f32(amax, amax, rc(txl.bitwise_and(rc(tmp, ty="uint32"), u322(0x7FFFFFFF)), ty="float32"))
        peer = txl.local_scalar("uint32")
        txl.ptx.shfl_sync.bfly.b32(peer, rc(amax, ty="uint32"), u322(1), u322(31), u322(0xFFFFFFFF))
        txl.ptx.max.f32(amax, amax, rc(peer, ty="float32"))
        sbits, rbits = scale_bits(amax)
        rcp = rc(rbits, ty="float32")
        with txl.If(R0 + r < M), txl.Then():
            with txl.serial(64) as i:
                cc = C0 + h * 64 + i
                with txl.If(cc < N), txl.Then():
                    txl.ptx.ld.shared.f32(tmp, txl.address_of(tile[r * PITCH + h * 64 + i]))
                    b = q_f32(tmp, rcp)
                    txl.ptx.st.global_.b8(txl.address_of(xq[i642(R0 + r) * i642(N) + i642(cc)]), txl.Cast("uint8", b))
            with txl.If(h == 0), txl.Then():
                txl.ptx.st.global_.b32(txl.address_of(xs[i642(R0 + r) * i642(nbn) + i642(tn)]), sbits)

        # columnwise: thread (column c, half g) covers 64 rows
        c = t & 127
        g = t >> 7
        camax = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.serial(64) as i:
            txl.ptx.ld.shared.f32(tmp, txl.address_of(tile[(g * 64 + i) * PITCH + c]))
            txl.ptx.max.f32(camax, camax, rc(txl.bitwise_and(rc(tmp, ty="uint32"), u322(0x7FFFFFFF)), ty="float32"))
        txl.ptx.st.shared.f32(txl.address_of(red[g * 128 + c]), camax)
        txl.ptx.bar.sync(u322(0), u322(NTHREADS))
        txl.ptx.ld.shared.f32(tmp, txl.address_of(red[(1 - g) * 128 + c]))
        txl.ptx.max.f32(camax, camax, tmp)
        csb, crb = scale_bits(camax)
        crcp = rc(crb, ty="float32")
        with txl.If(C0 + c < N), txl.Then():
            with txl.serial(64) as i:
                rr = R0 + g * 64 + i
                with txl.If(rr < M), txl.Then():
                    txl.ptx.ld.shared.f32(tmp, txl.address_of(tile[(g * 64 + i) * PITCH + c]))
                    b = q_f32(tmp, crcp)
                    txl.ptx.st.global_.b8(txl.address_of(xtq[i642(C0 + c) * i642(M) + i642(rr)]), txl.Cast("uint8", b))
            with txl.If(g == 0), txl.Then():
                txl.ptx.st.global_.b32(txl.address_of(xts[i642(C0 + c) * i642(nbm) + i642(tm)]), csb)

    return fp8_quant_grad_generic.compile()


GENERIC_DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def fp8_quant_grad_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    nbn, nbm = cdiv(N, BLOCK), cdiv(M, BLOCK)
    dev = x.device
    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev)
    scale_tok = torch.empty((M, nbn), dtype=torch.float32, device=dev)
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=dev)
    scale_t = torch.empty((N, nbm), dtype=torch.float32, device=dev)
    if M == 0 or N == 0:
        return out_tok, scale_tok, out_t, scale_t

    s0, s1 = x.stride()
    off = x.storage_offset()
    flat = x.as_strided(((M - 1) * s0 + (N - 1) * s1 + 1 + off,), (1,), 0)
    fast = (
        x.dtype == torch.bfloat16
        and s1 == 1
        and s0 % 8 == 0
        and off % 8 == 0
        and N % 8 == 0
        and M % 16 == 0
        and flat.data_ptr() % 16 == 0
    )
    with tvm_ffi.use_torch_stream():
        if fast:
            outs = (out_tok.view(torch.uint8).view(-1), scale_tok.view(-1).view(torch.int32), out_t.view(torch.uint8).view(-1), scale_t.view(-1).view(torch.int32))
            tiles, sms = nbn * nbm, num_sms(dev)
            if tiles < 4 * sms:
                rr16_kernel()(flat, *outs, M, N, s0, off, min(tiles, sms * RR16_CTAS_PER_SM))
            elif tiles < 10 * sms:
                rr8_kernel()(flat, *outs, M, N, s0, off, min(tiles, sms * RR8_CTAS_PER_SM))
            else:
                clc_kernel()(x.as_strided(((M - 1) * s0 + N,), (1,)), *outs, M, N, s0)
        else:
            generic_kernel(GENERIC_DTYPES[x.dtype])(
                flat,
                out_tok.view(torch.uint8).view(-1),
                scale_tok.view(-1).view(torch.int32),
                out_t.view(torch.uint8).view(-1),
                scale_t.view(-1).view(torch.int32),
                M,
                N,
                s0,
                s1,
                off,
            )
    return out_tok, scale_tok, out_t, scale_t
