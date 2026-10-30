"""
SM100 fused SwiGLU (+ gradients) -> grouped FP8 quantization (SM100 layout), TIRx-lite.

The output is cut into 128-row x 128-column tiles: one tile holds whole 1x128 row blocks
(x_fp8 / x_scale) and whole 128x1 column blocks (x_ch_fp8 / x_ch_scale), because groups are
128-row aligned (grouped_mm_offs is therefore not needed for the SM100 layout).

Persistent schedule: 512-thread CTAs (one per SM) walk tiles round-robin. Each thread owns the same
4 rows x 8 columns of every tile and prefetches them with cp.async into its own slots of a
shared-memory ring, so the HBM stream overlaps the arithmetic without any producer/consumer
barrier (a thread only ever reads bytes it copied itself). A tile load is split into row parts
(halves or quarters) that form a unit ring of per-mode depth: the forward keeps 1.5 tiles in
flight (three half-tile units), N=512 backward a quarter-tile ring one tile deep, N=1408 backward a
half-tile ring. The one CTA-wide exchange is the column amax: per-warp partials, 64 reducer
threads, an rcp broadcast.

Arithmetic runs on packed pairs (f32x2 / bf16x2). Triton's sigmoid, 1 / (1 + ex2.approx(x)) with
div.full.f32, is reproduced as 0.25 * rcp.approx.ftz(fma(ex2.approx.ftz(x), 0.25, 0.25)): this
is bit-identical for every bf16 input (checked exhaustively, including inf and NaN), since
1 + e absorbs any subnormal e and the 0.25 pre-scale reproduces div.full's large-divisor path.

gu_bwd with N % 128 != 0 (dg and du blocks offset by N % 128 columns) keeps one load and one
sigmoid per element by walking short ranges of consecutive 128-column windows (body_runL): CTA c
takes ranges c, c + G, ... so concurrent CTAs read adjacent columns of the same rows, carries
du of the previous window in thread-private shared memory and recomputes it (halo) at range starts.

N not a multiple of 8 falls back to a non-persistent kernel with narrower predicated loads.

Evolved with TIRx-harness: run silu, frontier member persistent-cpasync (commit 23cc854, 2126.9 us vs 8709.9 us, 4.095x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools
import math

import torch
import tvm
import tvm_ffi

import tirx_kernels.tirx_lite as txl

__all__ = ["fp8_silu_mul_fwd_sm100", "fp8_silu_mul_bwd_sm100", "fp8_silu_mul_gu_fwd_sm100", "fp8_silu_mul_gu_bwd_sm100"]

BLOCK = 128
NEG_LOG2E = -1.4426950408889634
FP8_MAX_RCP = 1.0 / 448.0
FULL = 0xFFFFFFFF
ABS2 = 0x7FFF7FFF
# Persistent schedule (measured best on B200): 512-thread CTAs, a one-tile cp.async lookahead
# (the tile is copied to registers before the next one is issued, so one ring slot suffices),
# two CTAs per SM for the forward (64 registers), one for the heavier backward.
CFG = {"threads": 512, "stages_fwd": 1, "occ_fwd": 1, "threads_bwd": 512, "stages_bwd": 1, "occ_bwd": 1,
       # cp.async unit ring per mode[:N]: a tile load is split into hp row parts, hu units in flight
       "hu": {"fwd": 3, "gu_fwd": 3, "bwd:1408": 2, "gu_bwd_pair:1408": 2, "bwd:512": 4, "gu_bwd_pair:512": 4},
       "hp": {"bwd:512": 4, "gu_bwd_pair:512": 4},
       "signed": ["bwd:1408", "gu_bwd_pair:1408", "bwd:512", "gu_bwd_pair:512"],
       # gu_bwd with N % 128 != 0: windows per interleaved range
       "run_l": 3}


def cdiv(a, b):
    return (a + b - 1) // b


def lo_f32(w):
    return txl.reinterpret("float32", txl.shift_left(w, txl.uint32(16)))


def hi_f32(w):
    return txl.reinterpret("float32", txl.bitwise_and(w, txl.uint32(0xFFFF0000)))


def f2(a, b):
    return txl.cuda.make_float2(a, b)


class F32Pair:
    """
    A packed f32 pair living in one 64-bit register.
    """

    def __init__(self):
        self.r = txl.local_scalar("uint64")

    @property
    def x(self):
        return txl.cuda.float2_x(self.r)

    @property
    def y(self):
        return txl.cuda.float2_y(self.r)


def unpack(w):
    p = F32Pair()
    txl.ptx.mov.b64(p.r, f2(lo_f32(w), hi_f32(w)))
    return p


def mul2(a, b):
    p = F32Pair()
    txl.ptx.mul.rn.f32x2(p.r, a, b)
    return p


def fma2(a, b, c):
    p = F32Pair()
    txl.ptx.fma.rn.f32x2(p.r, a, b, c)
    return p


def add2(a, b):
    p = F32Pair()
    txl.ptx.add.rn.f32x2(p.r, a, b)
    return p


def c2(v):
    return f2(txl.float32(v), txl.float32(v))


def sigmoid(g):
    """
    Packed Triton sigmoid of the pair g, times 4 (see module docstring).
    """
    x = mul2(g.r, c2(NEG_LOG2E))
    e0 = txl.local_scalar("float32")
    e1 = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(e0, x.x)
    txl.ptx.ex2.approx.ftz.f32(e1, x.y)
    d = fma2(f2(e0, e1), c2(0.25), c2(0.25))
    r0 = txl.local_scalar("float32")
    r1 = txl.local_scalar("float32")
    txl.ptx.rcp.approx.ftz.f32(r0, d.x)
    txl.ptx.rcp.approx.ftz.f32(r1, d.y)
    return f2(r0, r1)


def scale_rcp(amax):
    """
    E8M0 scale and its exact reciprocal (pith-train compute_fp8_scale, native cvt).
    """
    s16 = txl.local_scalar("uint16")
    arg = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(arg, txl.max(amax, txl.float32(1e-4)), txl.float32(FP8_MAX_RCP))
    txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(s16, txl.float32(0.0), arg)
    e = txl.bitwise_and(txl.cast(s16, "uint32"), txl.uint32(0xFF))
    sb = txl.local_scalar("uint32")
    txl.assign(sb, txl.Select(e == txl.uint32(0), txl.uint32(1 << 23), txl.shift_left(e, txl.uint32(23))))
    return txl.reinterpret("float32", sb), txl.reinterpret("float32", txl.uint32(254 << 23) - sb)


def quant2(dst16, xw, rcp2):
    """
    e4m3x2 of a bf16x2 word times a bf16x2 pair of powers of two.

    The bf16 product is exact wherever it is a normal number; where it is not, |x * rcp| is
    below 2^-126 and both this and the fp32 reference round it to a signed zero byte.
    """
    y = txl.local_scalar("uint32")
    txl.ptx["mul.rn.bf16x2"](y, xw, rcp2)
    txl.ptx["cvt.rn.satfinite.e4m3x2.bf16x2"](dst16, y)


def swiglu(mode_bwd, sel, N, ocol, wg, wu, wd, hw, W):
    """
    bf16 SwiGLU (or gradients) of the packed words, written to hw[out][t].
    """
    for t in range(W):
        gp = unpack(wg[t])
        up = unpack(wu[t])
        s = sigmoid(gp)
        sg = mul2(s, c2(0.25))
        silu = mul2(sg.r, gp.r)
        if not mode_bwd:
            h = mul2(silu.r, up.r)
            txl.ptx.cvt.rn.bf16x2.f32(hw[0][t], h.y, h.x)
            continue
        dp = unpack(wd[t])
        nsg = mul2(s, c2(-0.25))
        t2 = mul2(mul2(dp.r, up.r).r, sg.r)
        a = add2(gp.r, c2(1.0))
        b = fma2(nsg.r, gp.r, a.r)
        dg = mul2(t2.r, b.r)
        du = mul2(silu.r, dp.r)
        if not sel:
            txl.ptx.cvt.rn.bf16x2.f32(hw[0][t], dg.y, dg.x)
            txl.ptx.cvt.rn.bf16x2.f32(hw[1][t], du.y, du.x)
        else:
            c = ocol + 2 * (t % 4)
            lo = txl.Select(c < N, dg.x, du.x)
            hi = txl.Select(c + 1 < N, dg.y, du.y)
            txl.ptx.cvt.rn.bf16x2.f32(hw[0][t], hi, lo)


@functools.cache
def build(mode, N):
    """
    mode: fwd | gu_fwd | bwd | gu_bwd_pair | gu_bwd_run | gu_bwd_sel. Returns the compiled executable.
    """
    vec = math.gcd(8, N)
    persistent = vec == 8
    run = mode == "gu_bwd_run"
    NTHREADS = CFG["threads_bwd" if mode in ("bwd", "gu_bwd_pair", "gu_bwd_run", "gu_bwd_sel") else "threads"] if persistent else 256
    RPT = BLOCK // (NTHREADS // 16)  # rows per thread
    RSTRIDE = NTHREADS // 16
    NWARP = NTHREADS // 32
    gu = mode in ("gu_fwd", "gu_bwd_pair", "gu_bwd_run", "gu_bwd_sel")
    bwd = mode in ("bwd", "gu_bwd_pair", "gu_bwd_run", "gu_bwd_sel")
    sel = mode == "gu_bwd_sel"
    pair = mode in ("bwd", "gu_bwd_pair", "gu_bwd_run")
    IN_W = 2 * N if gu else N
    U_OFF = N if gu else 0
    K = 2 * N if mode in ("gu_bwd_pair", "gu_bwd_run", "gu_bwd_sel") else N  # output width
    KB = cdiv(K, BLOCK)
    LIM = K if sel else N  # column limit of this kernel's tiles (run: windows walk the N input columns)
    NT = cdiv(LIM, BLOCK)  # column tiles per row block
    full = LIM % BLOCK == 0
    # run (gu_bwd, N % 128 = HR != 0): input windows of 128 columns; du blocks start HR columns into a
    # window, HS = 128 - HR columns of each du block come from the previous window
    HQ, HR = N // BLOCK, N % BLOCK
    HS = BLOCK - HR
    n_out = 2 if pair else 1
    n_in = 3 if bwd else 2
    offs = [(0, 0), (N, N // BLOCK)] if mode == "gu_bwd_pair" else [(0, 0), (0, 0)]
    STAGES = CFG["stages_bwd" if bwd else "stages_fwd"]
    OCC = CFG["occ_bwd" if bwd else "occ_fwd"]
    W = RPT * 4

    def out_geom(oi, cb, ocol, tid):
        """
        (store column, row-scale block, reducer column, row-scale guard) of output oi.
        """
        col_off, blk_off = offs[oi]
        return ocol + col_off, cb + blk_off, cb * BLOCK + 2 * tid + col_off, None

    def quant_store(hw, outs, rb, cb, ocol0, row_base, sm, tid, lane, warp, cg, r0, buf, geo=None):
        """
        geo: optional [(words, out tensors, (ocol, sblk, ccol, sguard))] overriding hw/outs/out_geom.
        """
        part_sm, rcp_sm = sm
        peer = txl.local_scalar("uint32")
        items = geo if geo is not None else [(hw[oi], outs[oi], None) for oi in range(n_out)]
        for oi, (x, otens, g_) in enumerate(items):
            q_t, s_t, c_t, cs_t = otens
            xo = {}
            if isinstance(g_, dict):
                xo = g_
                ocol, sblk, ccol, sguard = g_["ocol"], g_["sblk"], g_["ccol"], g_["sguard"]
            else:
                ocol, sblk, ccol, sguard = g_ if g_ is not None else out_geom(oi, cb, ocol0, tid)
            olim = xo.get("olim", LIM + offs[oi][0])
            ofull = full and "olim" not in xo
            slane = xo.get("slane", 0)
            # |x| maxima on bf16x2 pairs; accumulators start at 0 so NaN is ignored
            rmax = txl.alloc_local([RPT], "uint32")
            cmax = txl.alloc_local([4], "uint32")
            for k in range(4):
                txl.assign(cmax[k], txl.uint32(0))
            for i in range(RPT):
                txl.assign(rmax[i], txl.uint32(0))
                for k in range(4):
                    txl.ptx["max.xorsign.abs.bf16x2"](rmax[i], rmax[i], x[i * 4 + k])
                    txl.ptx["max.xorsign.abs.bf16x2"](cmax[k], cmax[k], x[i * 4 + k])
            for i in range(RPT):
                txl.assign(rmax[i], txl.bitwise_and(rmax[i], txl.uint32(ABS2)))
            for k in range(4):
                txl.assign(cmax[k], txl.bitwise_and(cmax[k], txl.uint32(ABS2)))
            # column partials of this warp (rows r0 and r0 ^ 1 of each 32-row band)
            for k in range(4):
                txl.ptx.shfl_sync.bfly.b32(peer, cmax[k], txl.uint32(16), txl.uint32(31), txl.uint32(FULL))
                txl.ptx.max.bf16x2(cmax[k], cmax[k], peer)
            with txl.If(lane < 16), txl.Then():
                txl.ptx.st.shared.v4.b32(part_sm.ptr_to([buf, oi, warp, cg * 4]), cmax[0], cmax[1], cmax[2], cmax[3])
            # row maxima: fold the two column parities, two rows per bf16x2 word
            rpk = txl.alloc_local([RPT // 2], "uint32")
            for p in (range(RPT // 2) if "rw" not in xo else []):
                lo2 = txl.local_scalar("uint32")
                hi2 = txl.local_scalar("uint32")
                txl.ptx["prmt.b32"](lo2, rmax[2 * p], rmax[2 * p + 1], txl.uint32(0x5410))
                txl.ptx["prmt.b32"](hi2, rmax[2 * p], rmax[2 * p + 1], txl.uint32(0x7632))
                txl.ptx.max.bf16x2(rpk[p], lo2, hi2)
            for p in (range(RPT // 2) if "rw" not in xo else []):
                for d in (1, 2, 4, 8):
                    txl.ptx.shfl_sync.bfly.b32(peer, rpk[p], txl.uint32(d), txl.uint32(31), txl.uint32(FULL))
                    txl.ptx.max.bf16x2(rpk[p], rpk[p], peer)
            if "rw" in xo:
                for p in range(RPT // 2):
                    txl.assign(rpk[p], xo["rw"][p])
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))

            # 64 reducer threads: one bf16x2 column pair each
            with txl.If(tid < BLOCK // 2), txl.Then():
                acc = txl.local_scalar("uint32")
                tmpw = txl.alloc_local([NWARP], "uint32")
                for w_ in range(NWARP):
                    txl.ptx.ld.shared.b32(tmpw[w_], part_sm.ptr_to([buf, oi, w_, tid]))
                txl.assign(acc, tmpw[0])
                for w_ in range(1, NWARP):
                    txl.ptx.max.bf16x2(acc, acc, tmpw[w_])
                s0, rc0 = scale_rcp(lo_f32(acc))
                s1, rc1 = scale_rcp(hi_f32(acc))
                sc0 = txl.local_scalar("float32")
                sc1 = txl.local_scalar("float32")
                txl.assign(sc0, s0)
                txl.assign(sc1, s1)
                txl.ptx.st.shared.v2.f32(rcp_sm.ptr_to([buf, oi, 2 * tid]), rc0, rc1)
                if olim % 2 == 0:
                    with txl.If(ccol < olim), txl.Then():
                        txl.ptx.st.global_.v2.f32(cs_t.ptr_to([rb, ccol]), sc0, sc1)
                else:
                    with txl.If(ccol < olim), txl.Then():
                        txl.ptx.st.global_.f32(cs_t.ptr_to([rb, ccol]), sc0)
                    with txl.If(ccol + 1 < olim), txl.Then():
                        txl.ptx.st.global_.f32(cs_t.ptr_to([rb, ccol + 1]), sc1)

            def pack(q, a, b):
                return txl.bitwise_or(txl.cast(q[a], "uint32"), txl.shift_left(txl.cast(q[b], "uint32"), txl.uint32(16)))

            def store8(dst_t, row, q):
                if vec == 8:
                    if ofull:
                        txl.ptx.st.global_.v2.b32(dst_t.ptr_to([row, ocol]), pack(q, 0, 1), pack(q, 2, 3))
                    else:
                        with txl.If(ocol < olim), txl.Then():
                            txl.ptx.st.global_.v2.b32(dst_t.ptr_to([row, ocol]), pack(q, 0, 1), pack(q, 2, 3))
                elif vec == 4:
                    for sv in range(2):
                        with txl.If(ocol + 4 * sv < olim), txl.Then():
                            txl.ptx.st.global_.b32(dst_t.ptr_to([row, ocol + 4 * sv]), pack(q, 2 * sv, 2 * sv + 1))
                elif vec == 2:
                    for sv in range(4):
                        with txl.If(ocol + 2 * sv < olim), txl.Then():
                            txl.ptx.st.global_.b16(dst_t.ptr_to([row, ocol + 2 * sv]), q[sv])
                else:
                    for sv in range(8):
                        byte = txl.bitwise_and(txl.shift_right(txl.cast(q[sv // 2], "uint32"), txl.uint32(8 * (sv % 2))), txl.uint32(0xFF))
                        with txl.If(ocol + sv < olim), txl.Then():
                            txl.ptx.st.global_.b8(dst_t.ptr_to([row, ocol + sv]), txl.cast(byte, "uint8"))

            # row blocks, while the reducers finish the column scales
            for i in range(RPT):
                row = row_base + RSTRIDE * i
                w = rpk[i // 2]
                rs, rr = scale_rcp(lo_f32(w) if i % 2 == 0 else hi_f32(w))
                rr2 = txl.local_scalar("uint32")
                txl.ptx["prmt.b32"](rr2, txl.reinterpret("uint32", rr), txl.reinterpret("uint32", rr), txl.uint32(0x7632))
                with txl.If(cg == slane if sguard is None else txl.And(cg == slane, sguard)), txl.Then():
                    txl.ptx.st.global_.f32(s_t.ptr_to([row, sblk]), rs)
                qr = txl.alloc_local([4], "uint16")
                for k in range(4):
                    quant2(qr[k], x[i * 4 + k], rr2)
                store8(q_t, row, qr)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))

            crcp = txl.alloc_local([8], "uint32")
            txl.ptx.ld.shared.v4.b32(crcp[0], crcp[1], crcp[2], crcp[3], rcp_sm.ptr_to([buf, oi, cg * 8]))
            txl.ptx.ld.shared.v4.b32(crcp[4], crcp[5], crcp[6], crcp[7], rcp_sm.ptr_to([buf, oi, cg * 8 + 4]))
            crc2 = txl.alloc_local([4], "uint32")
            for k in range(4):
                txl.ptx["prmt.b32"](crc2[k], crcp[2 * k], crcp[2 * k + 1], txl.uint32(0x7632))
            for i in range(RPT):
                row = row_base + RSTRIDE * i
                qc = txl.alloc_local([4], "uint16")
                for k in range(4):
                    quant2(qc[k], x[i * 4 + k], crc2[k])
                store8(c_t, row, qc)

    def tile_coords(t, cg, r0):
        if f"{mode}:{N}" in CFG["signed"] or mode in CFG["signed"]:
            # Preserve the older signed coordinate expression for the half-tile
            # specialization.  Although it emits more address instructions, it
            # changes register scheduling materially for this pipeline.
            rb = t // NT
            cb = t % NT
            ocol = cb * BLOCK + cg * 8
            return rb, cb, ocol, rb * BLOCK + r0
        # tile indices are non-negative: unsigned division, 64-bit row so per-row offsets fold into addresses
        tu = txl.cast(t, "uint32")
        rb = txl.cast(tu // txl.uint32(NT), "int32")
        cb = txl.cast(tu % txl.uint32(NT), "int32")
        ocol = cb * BLOCK + cg * 8
        return rb, cb, ocol, txl.cast(rb, "int64") * BLOCK + txl.cast(r0, "int64")

    def body_persistent(dh, g, u, outs, MB, G):
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        cg = tid & 15
        r0 = tid >> 4
        cta = txl.cta_id()
        ntiles = MB * NT
        smem = txl.smem_pool()
        # DeepSeek's N=1408 backward rows benefit from dividing each tile load into two
        # independently pipelined 64-row units.  This exposes the second half's load
        # while the first half is copied to registers, without changing tile ownership
        # or the quantization protocol.  Other compile-time widths retain the frontier
        # schedule because the same split regressed qwen and dense rows.
        HU = CFG["hu"].get(f"{mode}:{N}", CFG["hu"].get(mode, 0))
        HP = CFG["hp"].get(f"{mode}:{N}", CFG["hp"].get(mode, 2))  # row parts per tile load
        HR2 = RPT // HP
        if HU:
            ring = smem.alloc([HU, n_in, BLOCK // HP, BLOCK], "bfloat16", align=128)
        else:
            ring = smem.alloc([STAGES, n_in, BLOCK, BLOCK], "bfloat16", align=128)
        sm = (smem.alloc([2, n_out, NWARP, BLOCK // 2], "uint32"), smem.alloc([2, n_out, BLOCK], "float32", align=16))
        srcs = [g, u] + ([dh] if bwd else [])

        def issue_h(uu):
            """
            Issue half-tile unit uu = 2 * tile_step + row_half.
            """
            t = cta + (uu // HP) * G
            hh = uu % HP
            slot = uu % HU
            with txl.If(t < ntiles), txl.Then():
                rb, cb, ocol, row_base = tile_coords(t, cg, r0)
                inside = None if full else ocol < LIM
                icol = txl.Select(ocol < N, ocol, ocol - N) if sel else ocol
                if not full:
                    icol = txl.Select(inside, icol, 0)
                for i2 in range(HR2):
                    rl = r0 + RSTRIDE * i2
                    row = row_base + txl.cast(RSTRIDE * HR2 * hh, "int64") + RSTRIDE * i2
                    for k, src in enumerate(srcs):
                        col = icol + U_OFF if k == 1 else icol
                        dst = ring.ptr_to([slot, k, rl, cg * 8])
                        if inside is None:
                            txl.ptx["cp.async.cg.shared.global"](dst, src.ptr_to([row, col]), 16, 16)
                        else:
                            txl.ptx["cp.async.cg.shared.global"](dst, src.ptr_to([row, col]), 16, txl.ptx.pred(ocol >= LIM))
            txl.ptx["cp.async.commit_group"]()

        def issue(j):
            t = cta + j * G
            slot = j % STAGES
            with txl.If(t < ntiles), txl.Then():
                rb, cb, ocol, row_base = tile_coords(t, cg, r0)
                inside = None if full else ocol < LIM
                icol = txl.Select(ocol < N, ocol, ocol - N) if sel else ocol
                if not full:
                    icol = txl.Select(inside, icol, 0)
                for i in range(RPT):
                    rl = r0 + RSTRIDE * i
                    row = row_base + RSTRIDE * i
                    for k, src in enumerate(srcs):
                        col = icol + U_OFF if k == 1 else icol
                        dst = ring.ptr_to([slot, k, rl, cg * 8])
                        if inside is None:
                            txl.ptx["cp.async.cg.shared.global"](dst, src.ptr_to([row, col]), 16, 16)
                        else:
                            txl.ptx["cp.async.cg.shared.global"](dst, src.ptr_to([row, col]), 16, txl.ptx.pred(ocol >= LIM))
            txl.ptx["cp.async.commit_group"]()

        for p in range(HU if HU else STAGES):
            (issue_h if HU else issue)(p)
        j = txl.local_scalar("int32")
        txl.assign(j, 0)
        hw = [txl.alloc_local([W], "uint32") for _ in range(n_out)]
        wg = txl.alloc_local([W], "uint32")
        wu = txl.alloc_local([W], "uint32")
        wd = txl.alloc_local([W], "uint32") if bwd else None
        regs = [wg, wu] + ([wd] if bwd else [])
        with txl.While(cta + j * G < ntiles):
            t = cta + j * G
            if HU:
                for hh in range(HP):
                    uu = HP * j + hh
                    txl.ptx["cp.async.wait_group"](HU - 1)
                    for i2 in range(HR2):
                        i = hh * HR2 + i2
                        rl = r0 + RSTRIDE * i2
                        for k in range(n_in):
                            dst = regs[k]
                            txl.ptx.ld.shared.v4.b32(dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], ring.ptr_to([uu % HU, k, rl, cg * 8]))
                    issue_h(uu + HU)
            else:
                slot = j % STAGES
                txl.ptx["cp.async.wait_group"](STAGES - 1)
                for i in range(RPT):
                    rl = r0 + RSTRIDE * i
                    for k in range(n_in):
                        dst = regs[k]
                        txl.ptx.ld.shared.v4.b32(dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], ring.ptr_to([slot, k, rl, cg * 8]))
                issue(j + STAGES)
            rb, cb, ocol, row_base = tile_coords(t, cg, r0)
            swiglu(bwd, sel, N, ocol, wg, wu, wd, hw, W)
            quant_store(hw, outs, rb, cb, ocol, row_base, sm, tid, lane, warp, cg, r0, j % 2)
            txl.assign(j, j + 1)
        txl.ptx["cp.async.wait_group"](0)

    def body_simple(dh, g, u, outs, MB, G):
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        cg = tid & 15
        r0 = tid >> 4
        rb, cb, ocol, row_base = tile_coords(txl.cta_id(), cg, r0)
        smem = txl.smem_pool()
        sm = (smem.alloc([1, n_out, NWARP, BLOCK // 2], "uint32"), smem.alloc([1, n_out, BLOCK], "float32", align=16))
        hw = [txl.alloc_local([W], "uint32") for _ in range(n_out)]
        wg = txl.alloc_local([W], "uint32")
        wu = txl.alloc_local([W], "uint32")
        wd = txl.alloc_local([W], "uint32") if bwd else None

        def load(dst, tensor, row, col, t0, pred, sv):
            ptr = tensor.ptr_to([row, col])
            kw = {} if pred is None else {"pred": pred}
            if vec == 1:
                t16 = txl.local_scalar("uint16")
                txl.ptx.ld.global_.b16(t16, ptr, **kw)
                val = txl.cast(t16, "uint32") if pred is None else txl.Select(pred, txl.cast(t16, "uint32"), txl.uint32(0))
                if sv % 2 == 0:
                    txl.assign(dst[t0], val)
                else:
                    txl.assign(dst[t0], txl.bitwise_or(dst[t0], txl.shift_left(val, txl.uint32(16))))
                return
            if vec == 4:
                txl.ptx.ld.global_.v2.b32(dst[t0], dst[t0 + 1], ptr, **kw)
            else:
                txl.ptx.ld.global_.b32(dst[t0], ptr, **kw)
            if pred is not None:
                # a skipped predicated load leaves its destination undefined
                for t in range(t0, t0 + vec // 2):
                    txl.assign(dst[t], txl.Select(pred, dst[t], txl.uint32(0)))

        for i in range(RPT):
            row = row_base + RSTRIDE * i
            for sv in range(8 // vec):
                c = ocol + sv * vec
                t0 = i * 4 + (sv * vec) // 2
                inside = None if full else c < LIM
                icol = txl.Select(c < N, c, c - N) if sel else c
                load(wg, g, row, icol, t0, inside, sv)
                load(wu, u, row, icol + U_OFF, t0, inside, sv)
                if bwd:
                    load(wd, dh, row, icol, t0, inside, sv)
        swiglu(bwd, sel, N, ocol, wg, wu, wd, hw, W)
        quant_store(hw, outs, rb, cb, ocol, row_base, sm, tid, lane, warp, cg, r0, 0)

    def body_run(dh, g, u, outs, MB, G):
        """
        gu_bwd, N % 128 = HR != 0. Each CTA walks one contiguous range of input windows k (a part of
        a row block). Window k loads g, u, dh once and computes dg and du (one sigmoid). dg block k is
        complete. du block HQ + k is du columns >= HS of window k - 1 followed by columns < HS of
        window k: thread cg keeps its du chunk of window k - 1 (in shared memory, private to the thread)
        and takes Select(8cg < HS, du_k, du_{k-1}), so the block is assembled without moving data
        between threads. The first window of a range starting at k > 0 recomputes du of window k - 1
        (columns >= HS) from a g/dh halo load; window HQ also loads du columns < HS of window 0 (aux)
        for the block where dg ends and du starts, and outputs the partial last du block.
        """
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        cg = tid & 15
        r0 = tid >> 4
        cta = txl.cta_id()
        total = MB * NT
        smem = txl.smem_pool()
        ring = smem.alloc([5, BLOCK, BLOCK], "bfloat16", align=128)
        sm = (smem.alloc([2, 2, NWARP, BLOCK // 2], "uint32"), smem.alloc([2, 2, BLOCK], "float32", align=16))
        out = outs[0]
        c8 = cg * 8
        isL = c8 < HS

        def lv(e, dt="int32"):
            v = txl.local_scalar(dt)
            txl.assign(v, e)
            return v

        # CTA cta owns the flattened windows f = rb * NT + k in [f0, f1): one contiguous range, at most a
        # few row-block segments; a segment starting at k > 0 recomputes du of window k - 1 (halo)
        # G >= MB: each row block is cut into PP = G // MB parts of PL windows (the host launches MB * PP
        # CTAs); otherwise flat contiguous ranges of ceil(total / G) windows. No range starts at window
        # HQ (it would need both a halo and the window-0 aux load).
        PP = lv(G // MB)
        PL = lv((NT + PP - 1) // txl.max(PP, 1))
        per = lv((total + G - 1) // G)

        def cut(part):
            c = txl.min(part * PL, NT)
            if HQ > 0:
                c = c + txl.Select(c == HQ, 1, 0)
            return c

        def start(c):
            f = c * per
            if HQ > 0:
                f = f + txl.Select(f % NT == HQ, 1, 0)
            return txl.min(f, total)

        aligned = PP >= 1
        rbc = cta // txl.max(PP, 1)
        pc = cta % txl.max(PP, 1)
        f0 = lv(txl.Select(aligned, rbc * NT + cut(pc), start(cta)))
        f1 = lv(txl.Select(aligned, rbc * NT + cut(pc + 1), start(cta + 1)))

        def win(f):
            fu = lv(txl.cast(f, "uint32"), "uint32")
            return lv(txl.cast(fu // txl.uint32(NT), "int32")), lv(txl.cast(fu % txl.uint32(NT), "int32"))

        def issue_main(f, ok_q):
            with txl.If(ok_q), txl.Then():
                rb, k = win(f)
                row_base = txl.cast(rb, "int64") * BLOCK + txl.cast(r0, "int64")
                ocol = k * BLOCK + c8
                ok = ocol < N
                col = txl.Select(ok, ocol, 0)
                for i in range(RPT):
                    rl = r0 + RSTRIDE * i
                    row = row_base + RSTRIDE * i
                    for kk, src, cc in ((0, g, col), (1, u, col + U_OFF), (2, dh, col)):
                        txl.ptx["cp.async.cg.shared.global"](ring.ptr_to([kk, rl, c8]), src.ptr_to([row, cc]), 16, txl.ptx.pred(txl.Not(ok)))
            txl.ptx["cp.async.commit_group"]()

        def aux_of(f, k):
            # halo (first window of a segment, k > 0): window k - 1, columns >= HS; window HQ: window 0, columns < HS
            halo_ = txl.And(f == f0, k > 0)
            last = k == HQ
            col = txl.Select(halo_, (k - 1) * BLOCK + c8, c8 - HR)
            ok = txl.Select(halo_, txl.Not(isL), txl.And(last, c8 >= HR))
            ok = txl.And(ok, col < N)
            return txl.Select(ok, col, 0), ok, txl.Or(halo_, last)

        def issue_aux(f, ok_q):
            with txl.If(ok_q), txl.Then():
                rb, k = win(f)
                row_base = txl.cast(rb, "int64") * BLOCK + txl.cast(r0, "int64")
                col, ok, need = aux_of(f, k)
                col = lv(col)
                okv = lv(txl.Select(ok, 1, 0))
                ok = okv == 1
                with txl.If(need), txl.Then():
                    for i in range(RPT):
                        rl = r0 + RSTRIDE * i
                        row = row_base + RSTRIDE * i
                        for kk, src in ((3, g), (4, dh)):
                            txl.ptx["cp.async.cg.shared.global"](ring.ptr_to([kk, rl, c8]), src.ptr_to([row, col]), 16, txl.ptx.pred(txl.Not(ok)))
            txl.ptx["cp.async.commit_group"]()

        def ld(regs, ks):
            for i in range(RPT):
                rl = r0 + RSTRIDE * i
                for dst, kk in zip(regs, ks):
                    txl.ptx.ld.shared.v4.b32(dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], ring.ptr_to([kk, rl, c8]))

        def du_only(t_, wg, wd):
            gp = unpack(wg[t_])
            dp = unpack(wd[t_])
            sg = mul2(sigmoid(gp), c2(0.25))
            du = mul2(mul2(sg.r, gp.r).r, dp.r)
            dw = txl.local_scalar("uint32")
            txl.ptx.cvt.rn.bf16x2.f32(dw, du.y, du.x)
            return dw

        def g_lin(k):  # dg block k, or (k == HQ) the block where dg ends and du starts: linear columns
            return {"ocol": k * BLOCK + c8, "sblk": k, "ccol": k * BLOCK + 2 * tid, "sguard": None, "olim": K}

        def g_du(b1):  # du block b1 (> HQ): its columns start HR into a window
            pos = lambda c: txl.bitwise_and(c + HR, BLOCK - 1)
            return {"ocol": b1 * BLOCK + pos(c8), "sblk": b1, "ccol": b1 * BLOCK + pos(2 * tid), "sguard": b1 < KB, "olim": K}

        issue_main(f0, f0 < f1)
        issue_aux(f0, f0 < f1)
        fc = txl.local_scalar("int32")
        txl.assign(fc, f0)
        hw = [txl.alloc_local([W], "uint32") for _ in range(2)]  # dg / du_k
        # du_{k-1} of this thread, kept in shared memory (only its owner reads or writes it)
        prev_sm = smem.alloc([W // 4, NTHREADS, 4], "uint32", align=16)

        def prev_st(x):
            for q4 in range(W // 4):
                txl.ptx.st.shared.v4.b32(prev_sm.ptr_to([q4, tid, 0]), x[4 * q4], x[4 * q4 + 1], x[4 * q4 + 2], x[4 * q4 + 3])

        def prev_ld(x):
            for q4 in range(W // 4):
                txl.ptx.ld.shared.v4.b32(x[4 * q4], x[4 * q4 + 1], x[4 * q4 + 2], x[4 * q4 + 3], prev_sm.ptr_to([q4, tid, 0]))

        wg = txl.alloc_local([W], "uint32")
        wu = txl.alloc_local([W], "uint32")
        wd = txl.alloc_local([W], "uint32")
        par = txl.local_scalar("int32")
        txl.assign(par, 0)
        with txl.While(fc < f1):
            rb, k = win(fc)
            row_base = txl.cast(rb, "int64") * BLOCK + txl.cast(r0, "int64")
            nf = lv(fc + 1)
            txl.ptx["cp.async.wait_group"](1)
            ld([wg, wu, wd], [0, 1, 2])
            issue_main(nf, nf < f1)
            swiglu(True, False, N, 0, wg, wu, wd, [hw[0], hw[1]], W)
            # du block HQ + k: du_k on lanes < HS, du_{k-1} (shared memory) on the others; du_k becomes
            # the next window's du_{k-1}
            prev_ld(wu)
            for t_ in range(W):
                txl.assign(wu[t_], txl.Select(isL, hw[1][t_], wu[t_]))
            prev_st(hw[1])
            txl.ptx["cp.async.wait_group"](1)
            acol, aok, need = aux_of(fc, k)
            ld([wg, wd], [3, 4])
            issue_aux(nf, nf < f1)
            last = k == HQ
            halo_ = txl.And(fc == f0, k > 0)
            with txl.If(need), txl.Then():
                # halo: du_{k-1} for lanes >= HS; last window: du of window 0 (columns < HS) for the dg|du block
                for t_ in range(W):
                    dw = du_only(t_, wg, wd)
                    txl.assign(wu[t_], txl.Select(txl.And(halo_, txl.Not(isL)), dw, wu[t_]))
                    txl.assign(hw[0][t_], txl.Select(txl.And(last, c8 >= HR), dw, hw[0][t_]))
            # (window 0 has no du block: its stores are masked off)
            gd = g_du(k + HQ)
            has = k > 0
            gd = dict(gd, ocol=txl.Select(has, gd["ocol"], K), ccol=txl.Select(has, gd["ccol"], K), sguard=txl.And(gd["sguard"], has))
            quant_store(None, None, rb, k, 0, row_base, sm, tid, lane, warp, cg, r0, 0,
                        geo=[(hw[0], out, g_lin(k)), (wu, out, gd)])
            if HR > 64:
                # window HQ: du columns >= HS (valid < HR) form the partial last block 2HQ + 1
                with txl.If(last), txl.Then():
                    prev_ld(wg)
                    for t_ in range(W):
                        txl.assign(wg[t_], txl.Select(isL, txl.uint32(0), wg[t_]))
                    quant_store(None, None, rb, k, 0, row_base, sm, tid, lane, warp, cg, r0, 1,
                                geo=[(wg, out, g_du(2 * HQ + 1))])
            txl.assign(fc, nf)
        txl.ptx["cp.async.wait_group"](0)

    def body_runL(dh, g, u, outs, MB, G):
        """
        gu_bwd, N % 128 = HR != 0, short interleaved window ranges. Each row block's NT windows are cut
        into R ranges of about RUN_L windows; CTA c walks ranges q = c, c + G, ... (consecutive CTAs take
        adjacent ranges of the same rows, so concurrent loads share DRAM pages) with one continuous
        cp.async pipeline across range boundaries. Per window as body_run: dg block k, and du block HQ + k
        from du_k (lanes < HS) and the thread's carried du_{k-1} (lanes >= HS); the first window of a
        range with k > 0 recomputes du_{k-1} from a g/dh halo load of window k - 1.
        """
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        cg = tid & 15
        r0 = tid >> 4
        cta = txl.cta_id()
        Q = MB * RUN_R
        smem = txl.smem_pool()
        ring = smem.alloc([5, BLOCK, BLOCK], "bfloat16", align=128)
        sm = (smem.alloc([2, 2, NWARP, BLOCK // 2], "uint32"), smem.alloc([2, 2, BLOCK], "float32", align=16))
        out = outs[0]
        c8 = cg * 8
        isL = c8 < HS

        def lv(e, dt="int32"):
            v = txl.local_scalar(dt)
            txl.assign(v, e)
            return v

        def cut(p):
            c = txl.min(p * RUN_L, NT)
            if HQ > 0:
                c = c + txl.Select(c == HQ, 1, 0)
            return c

        def rng(q):
            qu = lv(txl.cast(q, "uint32"), "uint32")
            rb = lv(txl.cast(qu // txl.uint32(RUN_R), "int32"))
            p = lv(txl.cast(qu % txl.uint32(RUN_R), "int32"))
            return rb, p

        def issue_main(rb, k, ok_q):
            with txl.If(ok_q), txl.Then():
                row_base = txl.cast(rb, "int64") * BLOCK + txl.cast(r0, "int64")
                ocol = k * BLOCK + c8
                ok = ocol < N
                col = txl.Select(ok, ocol, 0)
                for i in range(RPT):
                    rl = r0 + RSTRIDE * i
                    row = row_base + RSTRIDE * i
                    for kk, src, cc in ((0, g, col), (1, u, col + U_OFF), (2, dh, col)):
                        txl.ptx["cp.async.cg.shared.global"](ring.ptr_to([kk, rl, c8]), src.ptr_to([row, cc]), 16, txl.ptx.pred(txl.Not(ok)))
            txl.ptx["cp.async.commit_group"]()

        def aux_of(first, k):
            # halo (first window of a range, k > 0): window k - 1, columns >= HS; window HQ: window 0, columns < HS
            halo_ = txl.And(first == 1, k > 0)
            last = k == HQ
            col = txl.Select(halo_, (k - 1) * BLOCK + c8, c8 - HR)
            ok = txl.Select(halo_, txl.Not(isL), txl.And(last, c8 >= HR))
            ok = txl.And(ok, col < N)
            return txl.Select(ok, col, 0), ok, txl.Or(halo_, last)

        def issue_aux(rb, k, first, ok_q):
            with txl.If(ok_q), txl.Then():
                row_base = txl.cast(rb, "int64") * BLOCK + txl.cast(r0, "int64")
                col, ok, need = aux_of(first, k)
                col = lv(col)
                okv = lv(txl.Select(ok, 1, 0))
                ok = okv == 1
                with txl.If(need), txl.Then():
                    for i in range(RPT):
                        rl = r0 + RSTRIDE * i
                        row = row_base + RSTRIDE * i
                        for kk, src in ((3, g), (4, dh)):
                            txl.ptx["cp.async.cg.shared.global"](ring.ptr_to([kk, rl, c8]), src.ptr_to([row, col]), 16, txl.ptx.pred(txl.Not(ok)))
            txl.ptx["cp.async.commit_group"]()

        def ld(regs, ks):
            for i in range(RPT):
                rl = r0 + RSTRIDE * i
                for dst, kk in zip(regs, ks):
                    txl.ptx.ld.shared.v4.b32(dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], ring.ptr_to([kk, rl, c8]))

        def du_only(t_, wg, wd):
            gp = unpack(wg[t_])
            dp = unpack(wd[t_])
            sg = mul2(sigmoid(gp), c2(0.25))
            du = mul2(mul2(sg.r, gp.r).r, dp.r)
            dw = txl.local_scalar("uint32")
            txl.ptx.cvt.rn.bf16x2.f32(dw, du.y, du.x)
            return dw

        def g_lin(k):
            return {"ocol": k * BLOCK + c8, "sblk": k, "ccol": k * BLOCK + 2 * tid, "sguard": None, "olim": K}

        def g_du(b1):
            pos = lambda c: txl.bitwise_and(c + HR, BLOCK - 1)
            return {"ocol": b1 * BLOCK + pos(c8), "sblk": b1, "ccol": b1 * BLOCK + pos(2 * tid), "sguard": b1 < KB, "olim": K}

        # walk state: range q, window k, range end k1, first window of its range
        q = txl.local_scalar("int32")
        k = txl.local_scalar("int32")
        k1 = txl.local_scalar("int32")
        first = txl.local_scalar("int32")
        rbv = txl.local_scalar("int32")
        txl.assign(q, cta)
        rb0, p0 = rng(cta)
        txl.assign(rbv, rb0)
        txl.assign(k, cut(p0))
        txl.assign(k1, cut(p0 + 1))
        txl.assign(first, 1)
        issue_main(rbv, k, q < Q)
        issue_aux(rbv, k, first, q < Q)
        hw = [txl.alloc_local([W], "uint32") for _ in range(2)]
        prev_sm = smem.alloc([W // 4, NTHREADS, 4], "uint32", align=16)

        def prev_st(x):
            for q4 in range(W // 4):
                txl.ptx.st.shared.v4.b32(prev_sm.ptr_to([q4, tid, 0]), x[4 * q4], x[4 * q4 + 1], x[4 * q4 + 2], x[4 * q4 + 3])

        def prev_ld(x):
            for q4 in range(W // 4):
                txl.ptx.ld.shared.v4.b32(x[4 * q4], x[4 * q4 + 1], x[4 * q4 + 2], x[4 * q4 + 3], prev_sm.ptr_to([q4, tid, 0]))

        wg = txl.alloc_local([W], "uint32")
        wu = txl.alloc_local([W], "uint32")
        wd = txl.alloc_local([W], "uint32")
        with txl.While(q < Q):
            rb = rbv
            row_base = txl.cast(rb, "int64") * BLOCK + txl.cast(r0, "int64")
            # next window of this CTA's walk
            cont = lv(txl.Select(k + 1 < k1, 1, 0))
            nq = lv(txl.Select(cont == 1, q, q + G))
            nrb, np_ = rng(nq)
            nrb = lv(txl.Select(cont == 1, rb, nrb))
            nk = lv(txl.Select(cont == 1, k + 1, cut(np_)))
            nk1 = lv(txl.Select(cont == 1, k1, cut(np_ + 1)))
            nfirst = lv(1 - cont)
            txl.ptx["cp.async.wait_group"](1)
            ld([wg, wu, wd], [0, 1, 2])
            issue_main(nrb, nk, nq < Q)
            swiglu(True, False, N, 0, wg, wu, wd, [hw[0], hw[1]], W)
            prev_ld(wu)
            for t_ in range(W):
                txl.assign(wu[t_], txl.Select(isL, hw[1][t_], wu[t_]))
            prev_st(hw[1])
            txl.ptx["cp.async.wait_group"](1)
            acol, aok, need = aux_of(first, k)
            ld([wg, wd], [3, 4])
            issue_aux(nrb, nk, nfirst, nq < Q)
            last = k == HQ
            halo_ = txl.And(first == 1, k > 0)
            with txl.If(need), txl.Then():
                for t_ in range(W):
                    dw = du_only(t_, wg, wd)
                    txl.assign(wu[t_], txl.Select(txl.And(halo_, txl.Not(isL)), dw, wu[t_]))
                    txl.assign(hw[0][t_], txl.Select(txl.And(last, c8 >= HR), dw, hw[0][t_]))
            gd = g_du(k + HQ)
            has = k > 0
            gd = dict(gd, ocol=txl.Select(has, gd["ocol"], K), ccol=txl.Select(has, gd["ccol"], K), sguard=txl.And(gd["sguard"], has))
            quant_store(None, None, rb, k, 0, row_base, sm, tid, lane, warp, cg, r0, 0,
                        geo=[(hw[0], out, g_lin(k)), (wu, out, gd)])
            if HR > 64:
                with txl.If(last), txl.Then():
                    prev_ld(wg)
                    for t_ in range(W):
                        txl.assign(wg[t_], txl.Select(isL, txl.uint32(0), wg[t_]))
                    quant_store(None, None, rb, k, 0, row_base, sm, tid, lane, warp, cg, r0, 1,
                                geo=[(wg, out, g_du(2 * HQ + 1))])
            txl.assign(q, nq)
            txl.assign(rbv, nrb)
            txl.assign(k, nk)
            txl.assign(k1, nk1)
            txl.assign(first, nfirst)
        txl.ptx["cp.async.wait_group"](0)

    RUN_L = CFG["run_l"]
    if run and RUN_L:
        cuts = sorted({min(p * RUN_L, NT) + (1 if HQ > 0 and min(p * RUN_L, NT) == HQ else 0) for p in range(NT // RUN_L + 2)})
        RUN_R = len(cuts) - 1  # non-empty ranges per row block (the formula is monotone; empty ones only at the end)
        assert all(cuts[i] == min(i * RUN_L, NT) + (1 if HQ > 0 and min(i * RUN_L, NT) == HQ else 0) for i in range(RUN_R + 1))
    body = ((body_runL if RUN_L else body_run) if run else body_persistent) if persistent else body_simple

    def T(dt, f):
        return txl.gptr(dt, shape=f)

    QT = lambda p: (p["M"], K)
    ST = lambda p: (p["M"], KB)
    CST = lambda p: (p["MB"], K)
    kdec = txl.kernel(warps=NWARP, arch="sm_100a", grid=lambda p: p["G"], min_blocks_per_sm=OCC if persistent else None)

    if mode in ("fwd", "gu_fwd"):

        @kdec
        def kern(
            g: T("bfloat16", lambda p: (p["M"], IN_W)),
            u: T("bfloat16", lambda p: (p["M"], IN_W)),
            q0: T("float8_e4m3fn", QT),
            s0: T("float32", ST),
            c0: T("float8_e4m3fn", QT),
            cs0: T("float32", CST),
            M: txl.i32,
            MB: txl.i32,
            G: txl.i32,
        ):
            body(None, g, u, [(q0, s0, c0, cs0)], MB, G)

    elif mode == "bwd":

        @kdec
        def kern(
            dh: T("bfloat16", lambda p: (p["M"], N)),
            g: T("bfloat16", lambda p: (p["M"], N)),
            u: T("bfloat16", lambda p: (p["M"], N)),
            q0: T("float8_e4m3fn", QT),
            s0: T("float32", ST),
            c0: T("float8_e4m3fn", QT),
            cs0: T("float32", CST),
            q1: T("float8_e4m3fn", QT),
            s1: T("float32", ST),
            c1: T("float8_e4m3fn", QT),
            cs1: T("float32", CST),
            M: txl.i32,
            MB: txl.i32,
            G: txl.i32,
        ):
            body(dh, g, u, [(q0, s0, c0, cs0), (q1, s1, c1, cs1)], MB, G)

    else:

        @kdec
        def kern(
            dh: T("bfloat16", lambda p: (p["M"], N)),
            g: T("bfloat16", lambda p: (p["M"], 2 * N)),
            u: T("bfloat16", lambda p: (p["M"], 2 * N)),
            q0: T("float8_e4m3fn", QT),
            s0: T("float32", ST),
            c0: T("float8_e4m3fn", QT),
            cs0: T("float32", CST),
            M: txl.i32,
            MB: txl.i32,
            G: txl.i32,
        ):
            o = (q0, s0, c0, cs0)
            body(dh, g, u, [o, o] if pair else [o], MB, G)

    ex = kern.compile(tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}))
    return ex, persistent, NT, OCC, (RUN_R if run and RUN_L else 0)


@functools.cache
def num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def grid2(ntiles, occ, sms):
    capacity = occ * sms
    if occ == 1:
        # one resident CTA per SM: equalize per-CTA tile counts instead of a short final wave
        return cdiv(ntiles, cdiv(ntiles, capacity))
    return min(ntiles, capacity)


def launch(mode, N, M, device, *args):
    ex, persistent, nt, occ, run_r = build(mode, N)
    MB = M // BLOCK
    G = grid2(MB * nt, occ, num_sms(device.index)) if persistent else MB * nt
    if run_r:
        G = grid2(MB * run_r, 1, num_sms(device.index))
    elif mode == "gu_bwd_run":
        sms = num_sms(device.index)
        G = MB * (sms // MB) if MB <= sms else G  # whole row-block parts (see body_run)
    with tvm_ffi.use_torch_stream():
        ex(*args, M, MB, G)


def empty_quant(M, K, device):
    return (
        torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M, cdiv(K, BLOCK)), dtype=torch.float32, device=device),
        torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M // BLOCK, K), dtype=torch.float32, device=device),
    )


def fp8_silu_mul_fwd_sm100(g, u, grouped_mm_offs):
    M, N = g.shape
    out = empty_quant(M, N, g.device)
    if M == 0 or N == 0:
        return out
    launch("fwd", N, M, g.device, g, u, *out)
    return out


def fp8_silu_mul_bwd_sm100(dh, g, u, grouped_mm_offs):
    M, N = g.shape
    a, b = empty_quant(M, N, g.device), empty_quant(M, N, g.device)
    if M == 0 or N == 0:
        return a, b
    launch("bwd", N, M, g.device, dh, g, u, *a, *b)
    return a, b


def fp8_silu_mul_gu_fwd_sm100(gu, grouped_mm_offs):
    M, N = gu.shape[0], gu.shape[1] // 2
    out = empty_quant(M, N, gu.device)
    if M == 0 or N == 0:
        return out
    launch("gu_fwd", N, M, gu.device, gu, gu, *out)
    return out


def fp8_silu_mul_gu_bwd_sm100(dh, gu, grouped_mm_offs):
    M, N = gu.shape[0], gu.shape[1] // 2
    out = empty_quant(M, 2 * N, gu.device)
    if M == 0 or N == 0:
        return out
    mode = "gu_bwd_pair" if N % BLOCK == 0 else "gu_bwd_run" if N % 8 == 0 else "gu_bwd_sel"
    launch(mode, N, M, gu.device, dh, gu, gu, *out)
    return out
