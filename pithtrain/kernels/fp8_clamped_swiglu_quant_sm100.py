"""
SM100 fp8_clamped_swiglu_quant: fused TIRx-lite kernels (v2: packed math, 128x128 tiles).

One CTA (8 warps) owns a 128-row x 128-output-column tile. Each thread holds 8 consecutive
output columns of 8 rows (rows rg + 16 i). The clamped SwiGLU (forward) or its gradient
(backward) reproduces pith-train's Triton fp32 sequence bit for bit, using cheaper exact
equivalents: ex2.approx.ftz (only differs where 1 + e rounds to 1 anyway) and
div.full(1, d) == 0.25 * rcp.approx(0.25 * d) for d >= 1 (div.full's own range scaling),
with independent element pairs in f32x2 instructions. Quantization multiplies the bf16
pairs by the exact power-of-two reciprocal in bf16x2 and converts bf16x2 -> e4m3x2.
Backward writes fp32 128-row block sums; a second kernel reduces them per group.

Evolved with TIRx-harness: run clamped, frontier member tma-ring-2group (commit 0830a33, 1704.6 us vs 14038.2 us, 8.236x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import contextlib
import functools
import struct

import torch
import tvm
import tvm_ffi

import tirx_kernels.tirx_lite as txl

__all__ = ["clamped_swiglu_quant_fwd_sm100", "fp8_clamped_swiglu_rowwise_sm100", "fp8_clamped_swiglu_kmajor_sm100", "fp8_clamped_swiglu_bwd_sm100"]

BLOCK = 128
WARPS = 8
NEG_LOG2E = -1.4426950408889634
FP8_MAX_RCP = 1.0 / 448.0
AMAX_FLOOR = 1e-4
FULL = 0xFFFFFFFF


def f(op, *args):
    out = txl.local_scalar("float32")
    txl.ptx[op](out, *args)
    return out


def u2(op, *args):
    out = txl.local_scalar("uint32")
    txl.ptx[op](out, *args)
    return out


def lo_f32(w):
    return txl.reinterpret("float32", txl.shift_left(w, txl.uint32(16)))


def hi_f32(w):
    return txl.reinterpret("float32", txl.bitwise_and(w, txl.uint32(0xFFFF0000)))


def pk(lo, hi):
    out = txl.local_scalar("uint64")
    txl.ptx.mov.b64(out, lo, hi)
    return out


def unpk(p):
    lo = txl.local_scalar("float32")
    hi = txl.local_scalar("float32")
    txl.ptx.mov.b64(lo, hi, p)
    return lo, hi


def op2(op, *args):
    out = txl.local_scalar("uint64")
    txl.ptx[op + ".f32x2"](out, *args)
    return out


def sigmoid2(gc2, alpha2, consts):
    """
    (alpha*gc, sigmoid(alpha*gc)) for a pair, bit-identical to Triton's
    1 / (1 + ex2.approx(alpha*gc*-log2e)) with div.full.
    """
    a2 = op2("mul.rn", alpha2, gc2)
    t2 = op2("mul.rn", a2, consts["nl2"])
    t0, t1 = unpk(t2)
    e2 = pk(f("ex2.approx.ftz.f32", t0), f("ex2.approx.ftz.f32", t1))
    dq2 = op2("fma.rn", e2, consts["q2"], consts["q2"])
    d0, d1 = unpk(dq2)
    r2 = pk(f("rcp.approx.ftz.f32", d0), f("rcp.approx.ftz.f32", d1))
    return a2, op2("mul.rn", r2, consts["q2"])


def clamp_pair(x0, x1, limit, neg_limit):
    g0, g1 = lo_f32(x0), lo_f32(x1)
    u0, u1 = hi_f32(x0), hi_f32(x1)
    gc2 = pk(f("min.f32", g0, limit), f("min.f32", g1, limit))
    uc2 = pk(
        f("min.f32", f("max.f32", u0, neg_limit), limit),
        f("min.f32", f("max.f32", u1, neg_limit), limit),
    )
    return (g0, g1), (u0, u1), gc2, uc2


def fwd_pair(x0, x1, alpha2, limit, neg_limit, consts):
    """
    h for two adjacent channels (biased gate/up words x0, x1) -> bf16x2 word.
    """
    _, _, gc2, uc2 = clamp_pair(x0, x1, limit, neg_limit)
    _, s2 = sigmoid2(gc2, alpha2, consts)
    act2 = op2("mul.rn", gc2, s2)
    h2 = op2("mul.rn", op2("add.rn", uc2, consts["one2"]), act2)
    h0, h1 = unpk(h2)
    return u2("cvt.rn.bf16x2.f32", h1, h0)


def bwd_pair(x0, x1, dhw, alpha2, limit, neg_limit, consts):
    """
    (grad_gate, grad_up) words for two adjacent h channels; dhw = their dh bf16x2.
    """
    (g0, g1), (u0, u1), gc2, uc2 = clamp_pair(x0, x1, limit, neg_limit)
    a2, s2 = sigmoid2(gc2, alpha2, consts)
    dh2 = pk(lo_f32(dhw), hi_f32(dhw))
    act2 = op2("mul.rn", gc2, s2)
    gup2 = op2("mul.rn", act2, dh2)
    t1 = op2("mul.rn", op2("add.rn", uc2, consts["one2"]), dh2)
    t2 = op2("mul.rn", t1, s2)
    oms2 = op2("fma.rn", s2, consts["m12"], consts["one2"])
    fac2 = op2("fma.rn", a2, oms2, consts["one2"])
    gg2 = op2("mul.rn", t2, fac2)
    gg = unpk(gg2)
    gu = unpk(gup2)
    words = []
    for j, (g, u) in enumerate(((g0, u0), (g1, u1))):
        p_g = txl.local_scalar("uint32")
        txl.ptx.setp.ge.f32(p_g, limit, g)
        p_u = txl.local_scalar("uint32")
        txl.ptx.setp.ge.f32(p_u, u, neg_limit)
        p_u2 = txl.local_scalar("uint32")
        txl.ptx.setp.ge.f32(p_u2, limit, u)
        txl.ptx.and_.pred(p_u, txl.ptx.pred(p_u), txl.ptx.pred(p_u2))
        sg = txl.local_scalar("float32")
        txl.ptx.selp.f32(sg, gg[j], txl.float32(0.0), txl.ptx.pred(p_g))
        su = txl.local_scalar("float32")
        txl.ptx.selp.f32(su, gu[j], txl.float32(0.0), txl.ptx.pred(p_u))
        words.append(u2("cvt.rn.bf16x2.f32", su, sg))
    return words


def amax2(a, b):
    """
    max(|a|, |b|) per bf16 lane (sign is garbage; NaN ignored unless both).
    """
    return u2("max.xorsign.abs.bf16x2", a, b)


def bmax2(a, b):
    return u2("max.bf16x2", a, b)


def abs2(w):
    return txl.bitwise_and(w, txl.uint32(0x7FFF7FFF))


def shfl(v, lane_xor):
    peer = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(peer, v, txl.uint32(lane_xor), txl.uint32(31), txl.uint32(FULL))
    return peer


def ratio(amax_f32):
    return f("mul.rn.f32", f("max.f32", amax_f32, txl.float32(AMAX_FLOOR)), txl.float32(FP8_MAX_RCP))


def e8_pair(word):
    """
    bf16x2 amax word -> u32 with the two E8M0 exponents (lo lane in bits 0..7, hi in 8..15).
    amax >= 1e-4 keeps every exponent >= 105, so no 2^-126 floor is needed.
    """
    e16 = txl.local_scalar("uint16")
    txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(e16, ratio(hi_f32(word)), ratio(lo_f32(word)))
    return txl.Cast("uint32", e16)


def scale_of(e):
    return txl.reinterpret("float32", txl.shift_left(e, txl.uint32(23)))


def rcp_bf16(e):
    """
    bf16 bits of 2^(127 - e) (the exact reciprocal of the scale 2^(e - 127)).
    """
    return txl.shift_left(txl.uint32(254) - e, txl.uint32(7))


def fp8x4(w0, w1, r0, r1):
    """
    Quantize two bf16x2 words with bf16x2 reciprocals -> 4 e4m3 bytes.
    """
    b = []
    for w, r in ((w0, r0), (w1, r1)):
        q = u2("mul.rn.bf16x2", w, r)
        pair = txl.local_scalar("uint16")
        txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(pair, q)
        b.append(txl.Cast("uint32", pair))
    return txl.bitwise_or(b[0], txl.shift_left(b[1], txl.uint32(16)))


def halves(w):
    lo = txl.local_scalar("uint16")
    hi = txl.local_scalar("uint16")
    txl.ptx.mov.b32(lo, hi, w)
    return lo, hi


def clamp_bf16(x, lim2, neg2):
    """
    (min(gate, limit), clamp(up, -limit, limit)) on a biased bf16x2 word (limit bf16-exact).
    """
    # max first (NaN up -> -limit); the gate lane's max operand is NaN, the identity of a
    # non-NaN max, so a NaN gate stays NaN and the min turns it into limit, like min.f32.
    return u2("min.bf16x2", u2("max.bf16x2", x, neg2), lim2)


def fwd_pair_bf16(xc0, xc1, alpha2, consts):
    """
    h for two channels from clamped words xc0, xc1 -> bf16x2 word.
    """
    gc2 = pk(lo_f32(xc0), lo_f32(xc1))
    _, s2 = sigmoid2(gc2, alpha2, consts)
    act2 = op2("mul.rn", gc2, s2)
    u1 = []
    for xc in (xc0, xc1):
        _, hi = halves(xc)
        u1.append(f("add.rn.f32.bf16", hi, txl.float32(1.0)))
    h2 = op2("mul.rn", pk(u1[0], u1[1]), act2)
    h0, h1 = unpk(h2)
    return u2("cvt.rn.bf16x2.f32", h1, h0)


def bwd_pair_bf16(x0, x1, xc0, xc1, dhw, alpha2, consts):
    """
    (grad_gate, grad_up) words for two h channels; zero where the clamp was active.
    """
    gc2 = pk(lo_f32(xc0), lo_f32(xc1))
    a2, s2 = sigmoid2(gc2, alpha2, consts)
    dh2 = pk(lo_f32(dhw), hi_f32(dhw))
    u1 = []
    for xc in (xc0, xc1):
        _, hi = halves(xc)
        u1.append(f("add.rn.f32.bf16", hi, txl.float32(1.0)))
    act2 = op2("mul.rn", gc2, s2)
    gup2 = op2("mul.rn", act2, dh2)
    t1 = op2("mul.rn", pk(u1[0], u1[1]), dh2)
    t2 = op2("mul.rn", t1, s2)
    oms2 = op2("fma.rn", s2, consts["m12"], consts["one2"])
    fac2 = op2("fma.rn", a2, oms2, consts["one2"])
    gg2 = op2("mul.rn", t2, fac2)
    gg = unpk(gg2)
    gu = unpk(gup2)
    words = []
    for j, (x, xc) in enumerate(((x0, xc0), (x1, xc1))):
        # gate <= limit  <=>  min(gate, limit) == gate;  |up| <= limit  <=>  clamp(up) == up
        mask = u2("set.eq.u32.bf16x2", x, xc)
        words.append(txl.bitwise_and(u2("cvt.rn.bf16x2.f32", gu[j], gg[j]), mask))
    return words


@functools.cache
def quant_kernel(N: int, bwd: bool, write_row: bool, write_col: bool, bf16_clamp: bool):
    """
    Tile kernel. C output channels (N forward, 2N backward), tiles of 128 rows x 128 cols.
    """
    C = 2 * N if bwd else N
    assert N % 4 == 0, "vector path needs N % 4 == 0"
    n_ct = (C + BLOCK - 1) // BLOCK
    n_sc = n_ct
    has_tail = C % BLOCK != 0

    @txl.kernel(warps=WARPS, arch="sm_100a", grid="num_ctas")
    def kern(
        gu: txl.gptr[txl.bf16],
        bias: txl.gptr[txl.bf16],
        offs: txl.gptr[txl.i32],
        dh: txl.gptr[txl.bf16],
        x_fp8: txl.gptr[txl.u8],
        x_scale: txl.gptr[txl.f32],
        x_ch: txl.gptr[txl.u8],
        x_ch_scale: txl.gptr[txl.f32],
        partial: txl.gptr[txl.f32],
        alpha: txl.f32,
        limit: txl.f32,
        num_groups: txl.i32,
        num_ctas: txl.i32,
    ):
        cta = txl.cta_id()
        tid = txl.thread_id()
        lane = txl.truncmod(tid, txl.int32(32))
        warp = txl.truncdiv(tid, txl.int32(32))
        cg = txl.truncmod(lane, txl.int32(16))
        rg = warp * 2 + txl.truncdiv(lane, txl.int32(16))
        rb = txl.truncdiv(cta, txl.int32(n_ct))
        ct = txl.truncmod(cta, txl.int32(n_ct))
        c0 = ct * BLOCK + cg * 8
        row0 = rb * BLOCK

        smem = txl.smem_pool()
        red = smem.alloc([WARPS * BLOCK], txl.f32)
        redw = smem.alloc([WARPS * 64], txl.u32)

        neg_limit = f("neg.f32", limit)
        consts = {
            "nl2": pk(txl.float32(NEG_LOG2E), txl.float32(NEG_LOG2E)),
            "q2": pk(txl.float32(0.25), txl.float32(0.25)),
            "one2": pk(txl.float32(1.0), txl.float32(1.0)),
            "m12": pk(txl.float32(-1.0), txl.float32(-1.0)),
        }
        alpha2 = pk(alpha, alpha)
        lim_b = txl.shift_right(txl.reinterpret("uint32", limit), txl.uint32(16))
        lim2 = txl.local_scalar("uint32", init=txl.bitwise_or(lim_b, txl.shift_left(lim_b, txl.uint32(16))))
        neg2 = txl.local_scalar("uint32", init=txl.bitwise_or(txl.uint32(0x7FC0), txl.shift_left(txl.bitwise_or(lim_b, txl.uint32(0x8000)), txl.uint32(16))))

        ow = txl.alloc_local([32], "uint32")
        iw = txl.alloc_local([64], "uint32")
        dw = txl.alloc_local([16], "uint32")
        bw = txl.alloc_local([8], "uint32")
        grp = txl.local_scalar("int32", init=txl.int32(0))

        def body(tail):
            v0 = (c0 < C) if tail else None
            v1 = (c0 + 4 < C) if tail else None

            def guarded(cond, fn):
                if cond is None:
                    fn()
                else:
                    with txl.If(cond), txl.Then():
                        fn()

            def zero(regs):
                if tail:
                    for r in regs:
                        txl.ptx.mov.b32(r, txl.uint32(0))

            for i in range(8):
                r64 = txl.Cast("int64", row0 + rg + 16 * i)
                zero([iw[8 * i + k] for k in range(8)] + [dw[2 * i], dw[2 * i + 1]])
                if bwd:
                    def l0(i=i, r64=r64):
                        txl.ptx.ld.global_.nc.v2.b32(iw[8 * i], iw[8 * i + 1], gu.ptr_to([r64 * (2 * N) + txl.Cast("int64", c0)]))
                        txl.ptx.ld.global_.nc.b32(dw[2 * i], dh.ptr_to([r64 * N + txl.Cast("int64", txl.truncdiv(c0, txl.int32(2)))]))

                    def l1(i=i, r64=r64):
                        txl.ptx.ld.global_.nc.v2.b32(iw[8 * i + 2], iw[8 * i + 3], gu.ptr_to([r64 * (2 * N) + txl.Cast("int64", c0 + 4)]))
                        txl.ptx.ld.global_.nc.b32(dw[2 * i + 1], dh.ptr_to([r64 * N + txl.Cast("int64", txl.truncdiv(c0, txl.int32(2)) + 2)]))
                else:
                    def l0(i=i, r64=r64):
                        txl.ptx.ld.global_.nc.v4.b32(iw[8 * i], iw[8 * i + 1], iw[8 * i + 2], iw[8 * i + 3], gu.ptr_to([r64 * (2 * N) + txl.Cast("int64", 2 * c0)]))

                    def l1(i=i, r64=r64):
                        txl.ptx.ld.global_.nc.v4.b32(iw[8 * i + 4], iw[8 * i + 5], iw[8 * i + 6], iw[8 * i + 7], gu.ptr_to([r64 * (2 * N) + txl.Cast("int64", 2 * c0 + 8)]))
                guarded(v0, l0)
                guarded(v1, l1)

            # group of this row block = #(group ends <= row0), clamped to G-1 (warp ballot)
            oi = txl.local_scalar("int32")
            base = txl.local_scalar("int32", init=txl.int32(0))
            bal = txl.local_scalar("uint32")
            with txl.While(base < num_groups):
                txl.assign(oi, row0 + 1)
                with txl.If(base + lane < num_groups), txl.Then():
                    txl.ptx.ld.global_.nc.b32(oi, offs.ptr_to([txl.Cast("int64", base + lane)]))
                txl.ptx.vote_sync.ballot.b32(bal, txl.ptx.pred(oi <= row0), txl.uint32(FULL))
                txl.assign(grp, grp + txl.Cast("int32", txl.popcount(bal)))
                txl.assign(base, base + 32)
            with txl.If(grp > num_groups - 1), txl.Then():
                txl.assign(grp, num_groups - 1)

            zero([bw[k] for k in range(8)])
            g64 = txl.Cast("int64", grp)
            if bwd:
                guarded(v0, lambda: txl.ptx.ld.global_.nc.v2.b32(bw[0], bw[1], bias.ptr_to([g64 * (2 * N) + txl.Cast("int64", c0)])))
                guarded(v1, lambda: txl.ptx.ld.global_.nc.v2.b32(bw[2], bw[3], bias.ptr_to([g64 * (2 * N) + txl.Cast("int64", c0 + 4)])))
            else:
                guarded(v0, lambda: txl.ptx.ld.global_.nc.v4.b32(bw[0], bw[1], bw[2], bw[3], bias.ptr_to([g64 * (2 * N) + txl.Cast("int64", 2 * c0)])))
                guarded(v1, lambda: txl.ptx.ld.global_.nc.v4.b32(bw[4], bw[5], bw[6], bw[7], bias.ptr_to([g64 * (2 * N) + txl.Cast("int64", 2 * c0 + 8)])))

            for i in range(8):
                if bwd:
                    for k in range(2):
                        x0 = u2("add.rn.bf16x2", iw[8 * i + 2 * k], bw[2 * k])
                        x1 = u2("add.rn.bf16x2", iw[8 * i + 2 * k + 1], bw[2 * k + 1])
                        if bf16_clamp:
                            w0, w1 = bwd_pair_bf16(x0, x1, clamp_bf16(x0, lim2, neg2), clamp_bf16(x1, lim2, neg2), dw[2 * i + k], alpha2, consts)
                        else:
                            w0, w1 = bwd_pair(x0, x1, dw[2 * i + k], alpha2, limit, neg_limit, consts)
                        txl.ptx.mov.b32(ow[4 * i + 2 * k], w0)
                        txl.ptx.mov.b32(ow[4 * i + 2 * k + 1], w1)
                else:
                    for k in range(4):
                        x0 = u2("add.rn.bf16x2", iw[8 * i + 2 * k], bw[2 * k])
                        x1 = u2("add.rn.bf16x2", iw[8 * i + 2 * k + 1], bw[2 * k + 1])
                        if bf16_clamp:
                            w = fwd_pair_bf16(clamp_bf16(x0, lim2, neg2), clamp_bf16(x1, lim2, neg2), alpha2, consts)
                        else:
                            w = fwd_pair(x0, x1, alpha2, limit, neg_limit, consts)
                        txl.ptx.mov.b32(ow[4 * i + k], w)
                if tail:
                    with txl.If(c0 >= C), txl.Then():
                        txl.ptx.mov.b32(ow[4 * i + 0], txl.uint32(0))
                        txl.ptx.mov.b32(ow[4 * i + 1], txl.uint32(0))
                    with txl.If(c0 + 4 >= C), txl.Then():
                        txl.ptx.mov.b32(ow[4 * i + 2], txl.uint32(0))
                        txl.ptx.mov.b32(ow[4 * i + 3], txl.uint32(0))

            def store8(buf, row, w0, w1):
                off = txl.Cast("int64", row) * C + txl.Cast("int64", c0)
                if not tail:
                    if C % 8 == 0:
                        txl.ptx.st.global_.v2.b32(buf.ptr_to([off]), w0, w1)
                    else:
                        txl.ptx.st.global_.b32(buf.ptr_to([off]), w0)
                        txl.ptx.st.global_.b32(buf.ptr_to([off + 4]), w1)
                elif C % 8 == 0:
                    with txl.If(v1):
                        with txl.Then():
                            txl.ptx.st.global_.v2.b32(buf.ptr_to([off]), w0, w1)
                        with txl.Else():
                            with txl.If(v0), txl.Then():
                                txl.ptx.st.global_.b32(buf.ptr_to([off]), w0)
                else:
                    with txl.If(v0), txl.Then():
                        txl.ptx.st.global_.b32(buf.ptr_to([off]), w0)
                    with txl.If(v1), txl.Then():
                        txl.ptx.st.global_.b32(buf.ptr_to([off + 4]), w1)

            if write_row:
                for p in range(4):
                    ma = amax2(amax2(ow[8 * p], ow[8 * p + 1]), amax2(ow[8 * p + 2], ow[8 * p + 3]))
                    mb = amax2(amax2(ow[8 * p + 4], ow[8 * p + 5]), amax2(ow[8 * p + 6], ow[8 * p + 7]))
                    m = amax2(u2("prmt.b32", ma, mb, txl.uint32(0x5410)), u2("prmt.b32", ma, mb, txl.uint32(0x7632)))
                    for x in (1, 2, 4, 8):
                        m = amax2(m, shfl(m, x))
                    e = e8_pair(abs2(m))
                    ea = txl.bitwise_and(e, txl.uint32(0xFF))
                    eb = txl.shift_right(e, txl.uint32(8))
                    for half, ee in ((0, ea), (1, eb)):
                        i = 2 * p + half
                        row = row0 + rg + 16 * i
                        with txl.If(cg == 0), txl.Then():
                            txl.ptx.st.global_.f32(x_scale.ptr_to([txl.Cast("int64", row) * n_sc + txl.Cast("int64", ct)]), scale_of(ee))
                        rb16 = rcp_bf16(ee)
                        r2 = txl.bitwise_or(rb16, txl.shift_left(rb16, txl.uint32(16)))
                        store8(x_fp8, row, fp8x4(ow[4 * i], ow[4 * i + 1], r2, r2), fp8x4(ow[4 * i + 2], ow[4 * i + 3], r2, r2))

            if write_col:
                cm = []
                for k in range(4):
                    m = ow[k]
                    for i in range(1, 8):
                        m = amax2(m, ow[4 * i + k])
                    m = amax2(m, shfl(m, 16))
                    cm.append(m)
                with txl.If(lane < 16), txl.Then():
                    txl.ptx.st.shared.v4.b32(redw.ptr_to([warp * 64 + cg * 4]), cm[0], cm[1], cm[2], cm[3])
                txl.ptx.bar.sync(txl.uint32(0), txl.uint32(WARPS * 32))
                cw = txl.alloc_local([4], "uint32")
                tw = txl.alloc_local([4], "uint32")
                txl.ptx.ld.shared.v4.b32(cw[0], cw[1], cw[2], cw[3], redw.ptr_to([cg * 4]))
                for w in range(1, WARPS):
                    txl.ptx.ld.shared.v4.b32(tw[0], tw[1], tw[2], tw[3], redw.ptr_to([w * 64 + cg * 4]))
                    for k in range(4):
                        txl.ptx.mov.b32(cw[k], amax2(cw[k], tw[k]))
                rc = []
                sc = []
                for k in range(4):
                    e = e8_pair(abs2(cw[k]))
                    ea = txl.bitwise_and(e, txl.uint32(0xFF))
                    eb = txl.shift_right(e, txl.uint32(8))
                    sc += [scale_of(ea), scale_of(eb)]
                    rc.append(txl.local_scalar("uint32", init=txl.bitwise_or(rcp_bf16(ea), txl.shift_left(rcp_bf16(eb), txl.uint32(16)))))
                with txl.If(rg == 0), txl.Then():
                    if tail:
                        for j in range(8):
                            with txl.If(c0 + j < C), txl.Then():
                                txl.ptx.st.global_.f32(x_ch_scale.ptr_to([txl.Cast("int64", rb) * C + txl.Cast("int64", c0 + j)]), sc[j])
                    else:
                        for j in range(0, 8, 4):
                            txl.ptx.st.global_.v4.f32(x_ch_scale.ptr_to([txl.Cast("int64", rb) * C + txl.Cast("int64", c0 + j)]), sc[j], sc[j + 1], sc[j + 2], sc[j + 3])
                for i in range(8):
                    row = row0 + rg + 16 * i
                    store8(x_ch, row, fp8x4(ow[4 * i], ow[4 * i + 1], rc[0], rc[1]), fp8x4(ow[4 * i + 2], ow[4 * i + 3], rc[2], rc[3]))

            if bwd:
                cs = txl.alloc_local([8], "float32")
                for j in range(8):
                    txl.ptx.mov.b32(cs[j], txl.float32(0.0))
                for i in range(8):
                    for k in range(4):
                        hl, hh = halves(ow[4 * i + k])
                        txl.ptx["add.rn.f32.bf16"](cs[2 * k], hl, cs[2 * k])
                        txl.ptx["add.rn.f32.bf16"](cs[2 * k + 1], hh, cs[2 * k + 1])
                for j in range(8):
                    txl.ptx.mov.b32(cs[j], f("add.rn.f32", cs[j], txl.reinterpret("float32", shfl(txl.reinterpret("uint32", cs[j]), 16))))
                with txl.If(lane < 16), txl.Then():
                    txl.ptx.st.shared.v4.f32(red.ptr_to([warp * BLOCK + cg * 8]), cs[0], cs[1], cs[2], cs[3])
                    txl.ptx.st.shared.v4.f32(red.ptr_to([warp * BLOCK + cg * 8 + 4]), cs[4], cs[5], cs[6], cs[7])
                txl.ptx.bar.sync(txl.uint32(0), txl.uint32(WARPS * 32))
                tv = txl.alloc_local([8], "float32")
                with txl.If(rg == 0), txl.Then():
                    for w in range(1, WARPS):
                        txl.ptx.ld.shared.v4.f32(tv[0], tv[1], tv[2], tv[3], red.ptr_to([w * BLOCK + cg * 8]))
                        txl.ptx.ld.shared.v4.f32(tv[4], tv[5], tv[6], tv[7], red.ptr_to([w * BLOCK + cg * 8 + 4]))
                        for j in range(8):
                            txl.ptx.mov.b32(cs[j], f("add.rn.f32", cs[j], tv[j]))
                    if tail:
                        for j in range(8):
                            with txl.If(c0 + j < C), txl.Then():
                                txl.ptx.st.global_.f32(partial.ptr_to([txl.Cast("int64", rb) * C + txl.Cast("int64", c0 + j)]), cs[j])
                    else:
                        for j in range(0, 8, 4):
                            txl.ptx.st.global_.v4.f32(partial.ptr_to([txl.Cast("int64", rb) * C + txl.Cast("int64", c0 + j)]), cs[j], cs[j + 1], cs[j + 2], cs[j + 3])

        if has_tail:
            with txl.If(ct == n_ct - 1):
                with txl.Then():
                    body(True)
                with txl.Else():
                    body(False)
        else:
            body(False)

    return compile_kernel(kern)


@functools.cache
def segsum_kernel(C: int):
    @txl.kernel(warps=4, arch="sm_100a", grid="num_ctas")
    def segsum(
        partial: txl.gptr[txl.f32],
        offs: txl.gptr[txl.i32],
        out: txl.gptr[txl.bf16],
        num_groups: txl.i32,
        num_ctas: txl.i32,
    ):
        cta = txl.cta_id()
        tid = txl.thread_id()
        n_cb = (C + 127) // 128
        g = txl.truncdiv(cta, txl.int32(n_cb))
        c = txl.truncmod(cta, txl.int32(n_cb)) * 128 + tid
        start = txl.local_scalar("int32", init=txl.int32(0))
        end = txl.local_scalar("int32")
        with txl.If(g > 0), txl.Then():
            txl.ptx.ld.global_.nc.b32(start, offs.ptr_to([txl.Cast("int64", g - 1)]))
        txl.ptx.ld.global_.nc.b32(end, offs.ptr_to([txl.Cast("int64", g)]))
        acc = txl.local_scalar("float32", init=txl.float32(0.0))
        b = txl.local_scalar("int32", init=txl.truncdiv(start, txl.int32(128)))
        with txl.If(c < C), txl.Then():
            nb = txl.truncdiv(end, txl.int32(128))
            tv = txl.alloc_local([32], "float32")
            with txl.While(b < nb):
                for u in range(32):
                    txl.ptx.mov.b32(tv[u], txl.float32(0.0))
                    with txl.If(b + u < nb), txl.Then():
                        txl.ptx.ld.global_.f32(tv[u], partial.ptr_to([txl.Cast("int64", b + u) * C + txl.Cast("int64", c)]))
                for u in range(32):
                    txl.assign(acc, f("add.rn.f32", acc, tv[u]))
                txl.assign(b, b + 32)
            h = txl.local_scalar("uint16")
            txl.ptx.cvt.rn.bf16.f32(h, acc)
            txl.ptx.st.global_.b16(out.ptr_to([txl.Cast("int64", g) * C + txl.Cast("int64", c)]), h)

    return compile_kernel(segsum)


TMA_2D = "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
PWARPS = 16
MAX_SMEM_GROUPS = 512


@functools.cache
def num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@functools.cache
def pipe_kernel(N: int, bwd: bool, write_row: bool, write_col: bool, bf16_clamp: bool, stages: int):
    """
    Persistent bulk-copy pipelined tile kernel (needs N % 8 == 0, G <= 1024).

    CTA = 16 warps; thread (rg = 0..31, cg = 0..15) owns rows rg + 32 i (i < 4) of the tile and
    8 output channels: forward h channels 4cg..4cg+3 and 64+4cg..64+4cg+3 (16B smem chunks
    at 16cg and 256+16cg of a 512B gate_up row: conflict-free), backward channels 8cg..8cg+7.
    Warp 0 refills a stage with cp.async.bulk (one copy per row, plus the group's bias row)
    once every thread has read it (the CTA barrier after the smem reads).
    """
    C = 2 * N if bwd else N
    assert N % 8 == 0
    n_ct = (C + BLOCK - 1) // BLOCK
    GU_ROW = 512 if not bwd else 256  # smem bytes per row of gate_up per stage
    DH_ROW = 128 if bwd else 0
    BIAS_B = GU_ROW
    STAGE = BLOCK * (GU_ROW + DH_ROW) + BIAS_B
    RPT = 8  # rows per thread (rows rg + 16 i)
    has_tail = C % BLOCK != 0
    # Backward uses two independent 8-warp CTAs per SM.  Each CTA has one
    # consumer group and two ring stages; together the pair has the same
    # 16-warps/four-stages capacity as the frontier's single 16-warp CTA.
    P_WARPS = 8 if bwd else PWARPS
    NGROUPS = 1 if bwd else 2
    GW = P_WARPS // NGROUPS
    NSLOTS = 2 * NGROUPS
    SPLIT = (not bwd) and write_col and not write_row
    DYN = bwd  # dynamic tile schedule (backward)

    GU_BOX = GU_ROW // 2  # bf16 columns per tile row

    def host_prelude(params):
        rows = params["num_rows"]
        gu_map = txl.stack_alloca("tensormap", 1)
        dh_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)

        def encode(desc, data, inner, outer, stride, box0, box1):
            # 2-D bf16, no interleave/swizzle, 128B L2 promotion, zero OOB fill
            txl.call_packed("runtime.cuTensorMapEncodeTiled", desc, "bfloat16", 2, data, inner, outer, stride, box0, box1, 1, 1, 0, 0, 2, 0)

        encode(gu_map, params["gu"].data, 2 * N, rows, 4 * N, GU_BOX, BLOCK)
        encode(b_map, params["bias"].data, 2 * N, params["num_groups"], 4 * N, GU_BOX, 1)
        if bwd:
            encode(dh_map, params["dh"].data, N, rows, 2 * N, DH_ROW // 2, BLOCK)
        else:
            encode(dh_map, params["gu"].data, 2 * N, rows, 4 * N, GU_BOX, BLOCK)
        return gu_map, dh_map, b_map

    @txl.kernel(warps=P_WARPS, arch="sm_100a", grid="num_ctas", min_blocks_per_sm=2 if bwd else 1, host_prelude=host_prelude)
    def kern(
        gu: txl.gptr[txl.bf16],
        bias: txl.gptr[txl.bf16],
        offs: txl.gptr[txl.i32],
        dh: txl.gptr[txl.bf16],
        x_fp8: txl.gptr[txl.u8],
        x_scale: txl.gptr[txl.f32],
        x_ch: txl.gptr[txl.u8],
        x_ch_scale: txl.gptr[txl.f32],
        partial: txl.gptr[txl.f32],
        sched: txl.gptr[txl.u32],
        alpha: txl.f32,
        limit: txl.f32,
        num_groups: txl.i32,
        num_tiles: txl.i32,
        num_ctas: txl.i32,
        num_rows: txl.i32,
        *,
        host,
    ):
        gu_map, dh_map, b_map = host
        cta = txl.cta_id()
        tid = txl.thread_id()
        lane = txl.truncmod(tid, txl.int32(32))
        warp = txl.truncdiv(tid, txl.int32(32))
        cg = txl.truncmod(lane, txl.int32(16))
        gid = txl.truncdiv(warp, txl.int32(GW))  # consumer group (2 groups of GW warps)
        wg = txl.truncmod(warp, txl.int32(GW))
        rg = wg * 2 + txl.truncdiv(lane, txl.int32(16))
        gbar = 1 + gid * 3

        smem = txl.smem_pool()
        ring = smem.alloc([stages * STAGE], txl.u8, align=1024)
        offs_s = smem.alloc([MAX_SMEM_GROUPS], txl.i32, align=16)
        # column-reduction buffers per (group, tile parity): a tile's buffers are rewritten two
        # tiles later, after barriers that order the earlier reads, so no end-of-tile barrier
        redw_all = smem.alloc([NSLOTS * GW * 64], txl.u32, align=16)
        red_all = smem.alloc([NSLOTS * GW * BLOCK if bwd else 4], txl.f32, align=16)
        rcpw_all = smem.alloc([NSLOTS * 64], txl.u32, align=16)
        full = txl.MBarrier(smem, stages)
        full.init(1)
        # empty[s] completes when the group that consumed the stage's tile has read it
        empty = txl.MBarrier(smem, stages)
        empty.init(1)
        if DYN:
            # tile id of each stage's current load (-1 = end of the stream), written by the issuer
            # before its arrive on full[stage]
            tile_s = smem.alloc([stages], txl.i32, align=16)

        with txl.serial(txl.truncdiv(num_groups + (P_WARPS * 32 - 1), txl.int32(P_WARPS * 32))) as it:
            gi = it * (P_WARPS * 32) + tid
            with txl.If(gi < num_groups), txl.Then():
                ov = txl.local_scalar("int32")
                txl.ptx.ld.global_.nc.b32(ov, offs.ptr_to([txl.Cast("int64", gi)]))
                txl.ptx.st.shared.b32(offs_s.ptr_to([gi]), ov)
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(P_WARPS * 32))

        # per-warp monotonic cursor: an issuing warp's tiles have non-decreasing row0, so the
        # count of group ends <= row0 never decreases; scan on from the previous count
        gcur = txl.local_scalar("int32", init=txl.int32(0))

        def group_of(row0):
            """
            #(group ends <= row0) clamped to G-1, by one warp (all lanes return it).
            """
            grp = txl.local_scalar("int32", init=gcur)
            base = txl.local_scalar("int32", init=gcur)
            go = txl.local_scalar("int32", init=txl.int32(1))
            oi = txl.local_scalar("int32")
            bal = txl.local_scalar("uint32")
            with txl.While(go == 1):
                txl.assign(oi, row0 + 1)
                with txl.If(base + lane < num_groups), txl.Then():
                    txl.ptx.ld.shared.b32(oi, offs_s.ptr_to([base + lane]))
                txl.ptx.vote_sync.ballot.b32(bal, txl.ptx.pred(oi <= row0), txl.uint32(FULL))
                cnt = txl.local_scalar("int32", init=txl.Cast("int32", txl.popcount(bal)))
                txl.assign(grp, grp + cnt)
                txl.assign(base, base + 32)
                txl.assign(go, txl.int32(0))
                with txl.If(txl.And(cnt == 32, base < num_groups)), txl.Then():
                    txl.assign(go, txl.int32(1))
            txl.assign(gcur, grp)
            with txl.If(grp > num_groups - 1), txl.Then():
                txl.assign(grp, num_groups - 1)
            return grp

        # Tail split (forward, one quantization): the last partial wave's r tiles are split in
        # halves between 2r CTAs (rows for rowwise, columns for kmajor) when 2r <= num_ctas, so the
        # longest CTA does F + 1/2 tiles instead of F + 1. Both CTAs load the whole tile.
        F = txl.local_scalar("int32", init=txl.truncdiv(num_tiles, num_ctas))
        rem = txl.local_scalar("int32", init=num_tiles - F * num_ctas)
        if SPLIT:
            split_on = txl.local_scalar("int32", init=txl.Select(txl.And(rem > 0, 2 * rem <= num_ctas), txl.int32(1), txl.int32(0)))

        def tile_of(k):
            """
            (tile index, part) of local tile k: part 0 = whole, 1 / 2 = first / second half.
            """
            if not SPLIT:
                return cta + k * num_ctas, None
            tail = txl.And(split_on == 1, k == F)
            t = txl.Select(tail, F * num_ctas + txl.truncdiv(cta, txl.int32(2)), cta + k * num_ctas)
            part = txl.Select(tail, 1 + txl.truncmod(cta, txl.int32(2)), txl.int32(0))
            return t, part

        # dynamic schedule: lane 0 of warp 0 holds the next dynamic tile id, fetched one refill ahead
        nxt = txl.local_scalar("int32", init=txl.int32(0))

        def fetch():
            r = txl.local_scalar("uint32")
            txl.ptx.atom.global_.add.u32(r, sched.ptr_to([0]), txl.uint32(1))
            txl.assign(nxt, 2 * num_ctas + txl.Cast("int32", r))

        def issue(k, first=False):
            """
            Warp 0: bulk-load this CTA's k-th tile into stage k % stages.
            """
            if DYN:
                if first:
                    t = cta + k * num_ctas
                else:
                    t = txl.local_scalar("int32")
                    txl.ptx.shfl_sync.idx.b32(t, nxt, txl.uint32(0), txl.uint32(31), txl.uint32(FULL))
                    with txl.If(txl.And(lane == 0, t < num_tiles)), txl.Then():
                        fetch()
                with txl.If(t >= num_tiles), txl.Then():
                    with txl.If(lane == 0), txl.Then():
                        stg = txl.truncmod(k, txl.int32(stages))
                        txl.ptx.st.shared.b32(tile_s.ptr_to([stg]), txl.int32(-1))
                        txl.ptx.mbarrier.arrive.shared.b64(full.buf.ptr_to([stg]), txl.uint32(1))
            else:
                t, _ = tile_of(k)
            with txl.If(t < num_tiles), txl.Then():
                rb = txl.truncdiv(t, txl.int32(n_ct))
                ct = txl.truncmod(t, txl.int32(n_ct))
                stage = txl.truncmod(k, txl.int32(stages))
                sbase = stage * STAGE
                row0 = rb * BLOCK
                grp = group_of(row0)
                gcol = ct * (BLOCK * 2 if not bwd else BLOCK)  # first gate_up column of the tile
                with txl.If(lane == 0), txl.Then():
                    bar = full.buf.ptr_to([stage])
                    if DYN:
                        txl.ptx.st.shared.b32(tile_s.ptr_to([stage]), t)
                    # the group's generic-proxy reads of this stage precede the async refill
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(bar, txl.uint32(STAGE))
                    txl.ptx[TMA_2D](ring.ptr_to([sbase]), txl.address_of(gu_map), txl.Cast("int32", gcol), txl.Cast("int32", row0), bar)
                    if bwd:
                        txl.ptx[TMA_2D](ring.ptr_to([sbase + BLOCK * GU_ROW]), txl.address_of(dh_map), txl.Cast("int32", ct * (BLOCK // 2)), txl.Cast("int32", row0), bar)
                    txl.ptx[TMA_2D](ring.ptr_to([sbase + BLOCK * (GU_ROW + DH_ROW)]), txl.address_of(b_map), txl.Cast("int32", gcol), txl.Cast("int32", grp), bar)

        with txl.If(warp == 0), txl.Then():
            for k in range(stages):
                issue(k, first=True)
            if DYN:
                with txl.If(lane == 0), txl.Then():
                    fetch()

        neg_limit = f("neg.f32", limit)
        consts = {
            "nl2": pk(txl.float32(NEG_LOG2E), txl.float32(NEG_LOG2E)),
            "q2": pk(txl.float32(0.25), txl.float32(0.25)),
            "one2": pk(txl.float32(1.0), txl.float32(1.0)),
            "m12": pk(txl.float32(-1.0), txl.float32(-1.0)),
        }
        alpha2 = pk(alpha, alpha)
        lim_b = txl.shift_right(txl.reinterpret("uint32", limit), txl.uint32(16))
        lim2 = txl.local_scalar("uint32", init=txl.bitwise_or(lim_b, txl.shift_left(lim_b, txl.uint32(16))))
        neg2 = txl.local_scalar("uint32", init=txl.bitwise_or(txl.uint32(0x7FC0), txl.shift_left(txl.bitwise_or(lim_b, txl.uint32(0x8000)), txl.uint32(16))))

        iw = txl.alloc_local([8], "uint32")
        dw = txl.alloc_local([2], "uint32")
        bw = txl.alloc_local([8], "uint32")
        ow = txl.alloc_local([4 * RPT], "uint32")

        n_my = txl.truncdiv(num_tiles - cta + num_ctas - 1, num_ctas)
        if SPLIT:
            n_my = txl.Select(split_on == 1, F + txl.Select(cta < 2 * rem, txl.int32(1), txl.int32(0)), n_my)
        if DYN:
            kkv = txl.local_scalar("int32", init=txl.int32(0))
            go = txl.local_scalar("int32", init=txl.int32(1))
            loop_cm = txl.While(go != 0)
        else:
            loop_cm = txl.serial(n_my if bwd else txl.truncdiv(n_my - gid + 1, txl.int32(2)))
        with loop_cm as kk2_:
            kk2 = kkv if DYN else kk2_
            k = kk2 if bwd else gid + 2 * kk2
            gp = txl.truncmod(kk2, txl.int32(2)) if bwd else gid * 2 + txl.truncmod(kk2, txl.int32(2))  # (group, tile parity)
            rw0 = gp * (GW * 64)
            rf0 = gp * (GW * BLOCK)
            t0, part0 = (None, None) if DYN else tile_of(k)
            t = txl.local_scalar("int32") if DYN else txl.local_scalar("int32", init=t0)
            if SPLIT:
                part = txl.local_scalar("int32", init=part0)
                actA = part != 2  # rows 0..3 (rowwise) / channels ca (kmajor) are this CTA's
                actB = part != 1
            if bwd:
                # (backward measured faster with these left inline: 50.0 vs 50.6-51.9 us)
                rb = txl.truncdiv(t, txl.int32(n_ct))
                ct = txl.truncmod(t, txl.int32(n_ct))
                row0 = rb * BLOCK
                obase = txl.Cast("int64", row0 + rg) * C
                sbase_row = txl.Cast("int64", ct) * num_rows + txl.Cast("int64", row0 + rg)
            else:
                # per-tile indices computed once in int32 (inlined, every int64 use re-expanded
                # them into a 64-bit division: ~5% of the loop's instructions)
                rb = txl.local_scalar("int32", init=txl.truncdiv(t, txl.int32(n_ct)))
                ct = txl.local_scalar("int32", init=t - rb * n_ct)
                row0 = txl.local_scalar("int32", init=rb * BLOCK)
                # int64 bases of this thread's first output row and of its tile's row-scale run
                obase = txl.local_scalar("int64", init=txl.Cast("int64", row0 + rg) * C)
                sbase_row = txl.local_scalar("int64", init=txl.Cast("int64", ct) * num_rows + txl.Cast("int64", row0 + rg))
            stage = txl.truncmod(k, txl.int32(stages))
            sbase = stage * STAGE
            # Groups alternate tiles, so with an odd stage count the stage's previous tile
            # belongs to the other group; until that tile was consumed the full barrier's
            # parity could alias its still-pending phase. Its release cannot alias.
            with txl.If(k >= stages), txl.Then():
                empty.wait(stage, txl.truncmod(txl.truncdiv(k, txl.int32(stages)) + 1, txl.int32(2)))
            full.wait(stage, txl.truncmod(txl.truncdiv(k, txl.int32(stages)), txl.int32(2)))
            if DYN:
                txl.ptx.ld.shared.b32(t, tile_s.ptr_to([stage]))
            with guard(t >= 0 if DYN else None, go if DYN else None):

                # validity of this thread's two 4-channel granules
                if bwd:
                    ca, cb = ct * BLOCK + cg * 8, ct * BLOCK + cg * 8 + 4
                else:
                    ca, cb = ct * BLOCK + cg * 4, ct * BLOCK + 64 + cg * 4
                va, vb = ca < C, cb < C
                if SPLIT and write_col:
                    va, vb = txl.And(va, actA), txl.And(vb, actB)

                # bias row (same for every row of the tile)
                bb = sbase + BLOCK * (GU_ROW + DH_ROW)
                txl.ptx.ld.shared.v4.b32(bw[0], bw[1], bw[2], bw[3], ring.ptr_to([bb + cg * 16]))
                if not bwd:
                    txl.ptx.ld.shared.v4.b32(bw[4], bw[5], bw[6], bw[7], ring.ptr_to([bb + 256 + cg * 16]))
                if bwd:
                    cs = txl.alloc_local([8], "float32")
                    for j in range(8):
                        txl.ptx.mov.b32(cs[j], txl.float32(0.0))
                rbase = sbase + rg * GU_ROW + cg * 16
                dbase = sbase + BLOCK * GU_ROW + rg * DH_ROW + cg * 8
                for i in range(RPT):
                    # rows are read from the stage just before their math (few live registers)
                    if bwd:
                        txl.ptx.ld.shared.v4.b32(iw[0], iw[1], iw[2], iw[3], ring.ptr_to([rbase + 16 * i * GU_ROW]))
                        txl.ptx.ld.shared.v2.b32(dw[0], dw[1], ring.ptr_to([dbase + 16 * i * DH_ROW]))
                        for kk in range(2):
                            x0 = u2("add.rn.bf16x2", iw[2 * kk], bw[2 * kk])
                            x1 = u2("add.rn.bf16x2", iw[2 * kk + 1], bw[2 * kk + 1])
                            if bf16_clamp:
                                w0, w1 = bwd_pair_bf16(x0, x1, clamp_bf16(x0, lim2, neg2), clamp_bf16(x1, lim2, neg2), dw[kk], alpha2, consts)
                            else:
                                w0, w1 = bwd_pair(x0, x1, dw[kk], alpha2, limit, neg_limit, consts)
                            txl.ptx.mov.b32(ow[4 * i + 2 * kk], w0)
                            txl.ptx.mov.b32(ow[4 * i + 2 * kk + 1], w1)
                            for jj, w in ((2 * kk, w0), (2 * kk + 1, w1)):
                                hl, hh = halves(w)
                                txl.ptx["add.rn.f32.bf16"](cs[2 * jj], hl, cs[2 * jj])
                                txl.ptx["add.rn.f32.bf16"](cs[2 * jj + 1], hh, cs[2 * jj + 1])
                    else:
                        def fwd_math(i=i, kks=range(4)):
                            if 0 in kks:
                                txl.ptx.ld.shared.v4.b32(iw[0], iw[1], iw[2], iw[3], ring.ptr_to([rbase + 16 * i * GU_ROW]))
                            if 2 in kks:
                                txl.ptx.ld.shared.v4.b32(iw[4], iw[5], iw[6], iw[7], ring.ptr_to([rbase + 16 * i * GU_ROW + 256]))
                            for kk in kks:
                                x0 = u2("add.rn.bf16x2", iw[2 * kk], bw[2 * kk])
                                x1 = u2("add.rn.bf16x2", iw[2 * kk + 1], bw[2 * kk + 1])
                                if bf16_clamp:
                                    w = fwd_pair_bf16(clamp_bf16(x0, lim2, neg2), clamp_bf16(x1, lim2, neg2), alpha2, consts)
                                else:
                                    w = fwd_pair(x0, x1, alpha2, limit, neg_limit, consts)
                                txl.ptx.mov.b32(ow[4 * i + kk], w)
                        if not SPLIT:
                            fwd_math()
                        elif write_row:
                            with txl.If(actA if i < RPT // 2 else actB), txl.Then():
                                fwd_math()
                        else:
                            with txl.If(actA), txl.Then():
                                fwd_math(kks=(0, 1))
                            with txl.If(actB), txl.Then():
                                fwd_math(kks=(2, 3))
                    if has_tail:
                        with txl.If(ca >= C), txl.Then():
                            txl.ptx.mov.b32(ow[4 * i + 0], txl.uint32(0))
                            txl.ptx.mov.b32(ow[4 * i + 1], txl.uint32(0))
                        with txl.If(cb >= C), txl.Then():
                            txl.ptx.mov.b32(ow[4 * i + 2], txl.uint32(0))
                            txl.ptx.mov.b32(ow[4 * i + 3], txl.uint32(0))
                tig = wg * 32 + lane  # thread index within the group

                def col_of_word(w):
                    """
                    First of the two output channels held by smem word w (= cg * 4 + kk).
                    """
                    wcg = txl.truncdiv(w, txl.int32(4))
                    wk = txl.truncmod(w, txl.int32(4))
                    if bwd:
                        return ct * BLOCK + wcg * 8 + 2 * wk
                    return ct * BLOCK + wcg * 4 + 2 * wk + txl.truncdiv(wk, txl.int32(2)) * (64 - 4)

                if write_col:
                    cm = []
                    for kk in range(4):
                        m = ow[kk]
                        for i in range(1, RPT):
                            m = amax2(m, ow[4 * i + kk])
                        m = amax2(m, shfl(m, 16))
                        cm.append(m)
                    with txl.If(lane < 16), txl.Then():
                        txl.ptx.st.shared.v4.b32(redw_all.ptr_to([rw0 + wg * 64 + cg * 4]), cm[0], cm[1], cm[2], cm[3])
                if bwd:
                    for j in range(8):
                        txl.ptx.mov.b32(cs[j], f("add.rn.f32", cs[j], txl.reinterpret("float32", shfl(txl.reinterpret("uint32", cs[j]), 16))))
                    with txl.If(lane < 16), txl.Then():
                        txl.ptx.st.shared.v4.f32(red_all.ptr_to([rf0 + wg * BLOCK + cg * 8]), cs[0], cs[1], cs[2], cs[3])
                        txl.ptx.st.shared.v4.f32(red_all.ptr_to([rf0 + wg * BLOCK + cg * 8 + 4]), cs[4], cs[5], cs[6], cs[7])
                # every thread of the group is done with the stage: refill it
                txl.ptx.bar.sync(txl.Cast("uint32", gbar), txl.uint32(GW * 32))
                with txl.If(wg == 0), txl.Then():
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.mbarrier.arrive.shared.b64(empty.buf.ptr_to([stage]), txl.uint32(1))
                    issue(k + stages)

                def store(buf, i, w0, w1):
                    off = obase + 16 * i * C
                    if not has_tail and not (SPLIT and write_col):
                        if bwd:
                            txl.ptx.st.global_.v2.b32(buf.ptr_to([off + txl.Cast("int64", ca)]), w0, w1)
                        else:
                            txl.ptx.st.global_.b32(buf.ptr_to([off + txl.Cast("int64", ca)]), w0)
                            txl.ptx.st.global_.b32(buf.ptr_to([off + txl.Cast("int64", cb)]), w1)
                    elif bwd:
                        with txl.If(vb):
                            with txl.Then():
                                txl.ptx.st.global_.v2.b32(buf.ptr_to([off + txl.Cast("int64", ca)]), w0, w1)
                            with txl.Else():
                                with txl.If(va), txl.Then():
                                    txl.ptx.st.global_.b32(buf.ptr_to([off + txl.Cast("int64", ca)]), w0)
                    else:
                        with txl.If(va), txl.Then():
                            txl.ptx.st.global_.b32(buf.ptr_to([off + txl.Cast("int64", ca)]), w0)
                        with txl.If(vb), txl.Then():
                            txl.ptx.st.global_.b32(buf.ptr_to([off + txl.Cast("int64", cb)]), w1)

                if write_col:
                    # one thread per amax word: 8 warp partials -> E8M0 scales (stored) and the
                    # bf16x2 reciprocals every thread of the group needs (broadcast through smem)
                    with txl.If(tig < 64), txl.Then():
                        mw = txl.local_scalar("uint32")
                        tw = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(mw, redw_all.ptr_to([rw0 + tig]))
                        for w in range(1, GW):
                            txl.ptx.ld.shared.b32(tw, redw_all.ptr_to([rw0 + w * 64 + tig]))
                            txl.ptx.mov.b32(mw, amax2(mw, tw))
                        e = e8_pair(abs2(mw))
                        ea = txl.bitwise_and(e, txl.uint32(0xFF))
                        eb = txl.shift_right(e, txl.uint32(8))
                        txl.ptx.st.shared.b32(rcpw_all.ptr_to([gp * 64 + tig]), txl.bitwise_or(rcp_bf16(ea), txl.shift_left(rcp_bf16(eb), txl.uint32(16))))
                        col = col_of_word(tig)
                        cok = col < C
                        if SPLIT:
                            # words cg*4 + 0..1 hold the first-half channels (ca), 2..3 the second
                            cok = txl.And(cok, txl.Select(txl.truncmod(tig, txl.int32(4)) < 2, txl.Cast("int32", actA), txl.Cast("int32", actB)) == 1)
                        with txl.If(cok), txl.Then():
                            txl.ptx.st.global_.v2.f32(x_ch_scale.ptr_to([txl.Cast("int64", rb) * C + txl.Cast("int64", col)]), scale_of(ea), scale_of(eb))
                if bwd:
                    # one thread per column: the 128-row block sum over the group's 8 warp partials
                    with txl.If(txl.And(tig >= 64, tig < 64 + BLOCK)), txl.Then():
                        j = tig - 64
                        acc = txl.local_scalar("float32")
                        tv = txl.local_scalar("float32")
                        txl.ptx.ld.shared.f32(acc, red_all.ptr_to([rf0 + j]))
                        for w in range(1, GW):
                            txl.ptx.ld.shared.f32(tv, red_all.ptr_to([rf0 + w * BLOCK + j]))
                            txl.assign(acc, f("add.rn.f32", acc, tv))
                        with txl.If(ct * BLOCK + j < C), txl.Then():
                            txl.ptx.st.global_.f32(partial.ptr_to([txl.Cast("int64", rb) * C + txl.Cast("int64", ct * BLOCK + j)]), acc)
                if write_row:
                    for p in range(RPT // 2):
                        def row_quant(p=p):
                            ma = amax2(amax2(ow[8 * p], ow[8 * p + 1]), amax2(ow[8 * p + 2], ow[8 * p + 3]))
                            mb = amax2(amax2(ow[8 * p + 4], ow[8 * p + 5]), amax2(ow[8 * p + 6], ow[8 * p + 7]))
                            m = amax2(u2("prmt.b32", ma, mb, txl.uint32(0x5410)), u2("prmt.b32", ma, mb, txl.uint32(0x7632)))
                            for x in (1, 2, 4, 8):
                                m = amax2(m, shfl(m, x))
                            e = e8_pair(abs2(m))
                            ea = txl.bitwise_and(e, txl.uint32(0xFF))
                            eb = txl.shift_right(e, txl.uint32(8))
                            for half, ee in ((0, ea), (1, eb)):
                                i = 2 * p + half
                                with txl.If(cg == 0), txl.Then():
                                    txl.ptx.st.global_.f32(x_scale.ptr_to([sbase_row + 16 * i]), scale_of(ee))
                                rb16 = rcp_bf16(ee)
                                r2 = txl.bitwise_or(rb16, txl.shift_left(rb16, txl.uint32(16)))
                                store(x_fp8, i, fp8x4(ow[4 * i], ow[4 * i + 1], r2, r2), fp8x4(ow[4 * i + 2], ow[4 * i + 3], r2, r2))
                        if SPLIT:
                            with txl.If(actA if p < RPT // 4 else actB), txl.Then():
                                row_quant()
                        else:
                            row_quant()

                if write_col:
                    txl.ptx.bar.sync(txl.Cast("uint32", gbar + 2), txl.uint32(GW * 32))
                    rc = txl.alloc_local([4], "uint32")
                    txl.ptx.ld.shared.v4.b32(rc[0], rc[1], rc[2], rc[3], rcpw_all.ptr_to([gp * 64 + cg * 4]))
                    for i in range(RPT):
                        store(x_ch, i, fp8x4(ow[4 * i], ow[4 * i + 1], rc[0], rc[1]), fp8x4(ow[4 * i + 2], ow[4 * i + 3], rc[2], rc[3]))
            if DYN:
                txl.assign(kkv, kkv + 1)
        if DYN:
            # the last CTA out returns the counter pair to zero for the next launch
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(P_WARPS * 32))
            with txl.If(tid == 0), txl.Then():
                ex = txl.local_scalar("uint32")
                # release: this CTA's tile-counter atomics; acquire: every other CTA's
                txl.ptx.atom.acq_rel.gpu.global_.add.u32(ex, sched.ptr_to([1]), txl.uint32(1))
                with txl.If(ex == txl.Cast("uint32", num_ctas - 1)), txl.Then():
                    txl.ptx.st.global_.v2.b32(sched.ptr_to([0]), txl.uint32(0), txl.uint32(0))

    return compile_kernel(kern)


@contextlib.contextmanager
def guard(cond, go):
    """
    Body under If(cond) (no-op when cond is None); else clears the loop flag go.
    """
    if cond is None:
        yield
        return
    with txl.If(cond):
        with txl.Then():
            yield
        with txl.Else():
            txl.assign(go, txl.int32(0))


def compile_kernel(kernel):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        return tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")


SCHED = {}


def sched_counter(device):
    """
    [tile counter, CTA exit counter] for the dynamic schedule: zero between launches
    (the last CTA out resets them). One pair per (device, stream), so launches on two streams
    never share a counter.
    """
    key = (device.type, device.index, torch.cuda.current_stream(device).cuda_stream)
    if key not in SCHED:
        SCHED[key] = torch.zeros(2, dtype=torch.int32, device=device).view(torch.uint32)
    return SCHED[key]


def empty_quant(M, C, device):
    return (
        torch.empty((M, C), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M, (C + BLOCK - 1) // BLOCK), dtype=torch.float32, device=device),
        torch.empty((M, C), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M // BLOCK, C), dtype=torch.float32, device=device),
    )


def run(kind, gate_up, bias, offs, alpha, limit, dh=None):
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    bwd = kind == "bwd"
    C = 2 * N if bwd else N
    dev = gate_up.device
    x_fp8, x_scale, x_ch, x_ch_scale = empty_quant(M, C, dev)
    d_bias = None
    if bwd:
        d_bias = torch.zeros(bias.shape, dtype=bias.dtype, device=dev) if M == 0 or N == 0 else torch.empty(bias.shape, dtype=bias.dtype, device=dev)
    if M == 0 or N == 0:
        return (x_fp8, x_scale, x_ch, x_ch_scale), d_bias
    write_row = kind in ("rowwise", "quant_fwd", "bwd")
    write_col = kind in ("kmajor", "quant_fwd", "bwd")
    lim_bits = struct.unpack("<I", struct.pack("<f", float(limit)))[0]
    # the bf16x2 clamp needs limit >= 0 exactly representable in bf16 (gpt-oss: 7.0)
    bf16_clamp = (lim_bits & 0xFFFF) == 0 and lim_bits < 0x7F800000
    G = offs.shape[0]
    n_ct = (C + BLOCK - 1) // BLOCK
    partial = torch.empty((M // BLOCK, C), dtype=torch.float32, device=dev) if bwd else x_ch_scale
    dummy_dh = dh if bwd else gate_up
    tensors = (
        gate_up.view(-1), bias.view(-1), offs, dummy_dh.reshape(-1),
        x_fp8.view(torch.uint8).view(-1), x_scale.view(-1), x_ch.view(torch.uint8).view(-1), x_ch_scale.view(-1),
        partial.view(-1),
    )
    num_tiles = (M // BLOCK) * n_ct
    pipe = N % 8 == 0 and G <= MAX_SMEM_GROUPS
    if pipe:
        # MN-major rowwise scales (scale strides are unchecked): a tile's 128 row scales
        # are one contiguous run; returned as the (M, ceil(C/128)) transposed view
        x_scale = torch.empty((n_ct, M), dtype=torch.float32, device=dev).t()
        tensors = tensors[:5] + (x_scale.t().reshape(-1),) + tensors[6:]
        tensors = tensors + (sched_counter(dev),)
    with tvm_ffi.use_torch_stream():
        if pipe:
            ker = pipe_kernel(N, bwd, write_row, write_col, bf16_clamp, 2 if bwd else 3)
            ker(*tensors, float(alpha), float(limit), G, num_tiles, min(num_tiles, (2 if bwd else 1) * num_sms()), M)
        else:
            ker = quant_kernel(N, bwd, write_row, write_col, bf16_clamp)
            ker(*tensors, float(alpha), float(limit), G, num_tiles)
        if bwd:
            segsum_kernel(C)(partial.view(-1), offs, d_bias.view(-1), G, G * ((C + 127) // 128))
    return (x_fp8, x_scale, x_ch, x_ch_scale), d_bias


def clamped_swiglu_quant_fwd_sm100(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels=None):
    return run("quant_fwd", gate_up, bias, grouped_mm_offs, alpha, limit)[0]


def fp8_clamped_swiglu_rowwise_sm100(gate_up, bias, grouped_mm_offs, alpha, limit):
    return run("rowwise", gate_up, bias, grouped_mm_offs, alpha, limit)[0][:2]


def fp8_clamped_swiglu_kmajor_sm100(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels=None):
    return run("kmajor", gate_up, bias, grouped_mm_offs, alpha, limit)[0][2:]


def fp8_clamped_swiglu_bwd_sm100(dh, gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels=None):
    return run("bwd", gate_up, bias, grouped_mm_offs, alpha, limit, dh=dh)
