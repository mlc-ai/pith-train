"""
SM100 fp8_quant_grad in TIRx-lite: rowwise 1x128 FP8 of x and of x.T from one read of x.

Fast path (bf16, unit column stride, 16B-aligned rows, N % 8 == 0, M % 16 == 0): one CTA per 128x128
tile (non-persistent; the hardware CTA scheduler balances the tail), 8 warps, 4 CTAs/SM. Every thread
issues its eight 16-byte loads up front (8 lanes per row: chunks k and k+8), copies them into an
XOR-swizzled shared tile, quantizes its rows straight from registers (128 B coalesced xq stores per row),
then one barrier and the column phase from shared memory (8 lanes per column pair over 16-row groups,
column abs-max reduced in-warp, 128 B coalesced stores per xtq row). Scales use an exact integer E8M0
formula; quantization is an exact power-of-two bf16x2 multiply plus cvt.rn.satfinite.e4m3x2.bf16x2.
Everything else (fp32/fp16, odd shapes, arbitrary strides) takes a scalar generic kernel.

Evolved with TIRx-harness: run grad, frontier member oneshot-regload-tile (commit f103d83, 217.9 us vs 379.1 us, 1.74x over the
previous SM100 path in the task's benchmark). Compiled once per process by
tvm_tirx.tirx.compile(..., pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc as before; a cubin (sm_100a SASS only, as the old -gencode build), since -arch sm_100a would also emit compute_100 PTX, which rejects sm_100a-only instructions

__all__ = ["fp8_quant_grad_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


BLOCK = 128
NTHREADS = 256


def cdiv(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------- trace-time helpers


def i642(v):
    return txl.Cast("int64", v)


def rc(v, ty):
    return txl.reinterpret(v, ty=ty)


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


def iket_range(name):
    """
    One IKET range per warp; stripped by the ordinary TIRx pipeline.
    """
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
    with txl.if_(txl.lane_id() == 0), txl.then_():
        txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def iket_end(token):
    txl.cuda.iket.range_end(token[0])


# ---------------------------------------------------------------- fast bf16 tile kernel


TILE_WORDS = BLOCK * 64


def build_one():
    """
    Non-persistent: one CTA per 128x128 tile (hardware scheduling balances the tail). Every
    thread issues its eight 16-byte loads up front (registers), writes them to the swizzled smem
    tile, runs the row phase from registers, then one barrier and the column phase from smem.
    """

    @txl.kernel()
    def fp8_quant_grad_one(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), xq: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), xs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), xtq: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), xts: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, ld: txl.int64, off: txl.int64, pf: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=((N + 127) // 128) * ((M + 127) // 128), block=(NTHREADS // 32) * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=4))
        cid = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        pool = txl.smem_pool()
        tile = pool.alloc([TILE_WORDS], "uint32", align=128)
        rsc = pool.alloc([BLOCK], "uint32", align=16)
        nbn = (N + 127) >> 7
        tm = txl.local_scalar("int32", init=txl.truncdiv(cid, nbn))
        tn = txl.local_scalar("int32", init=cid - tm * nbn)
        R0 = tm * 128
        C0 = tn * 128
        M64 = i642(M)
        N64 = i642(N)
        k = lane & 7
        r0 = warp * 4 + (lane >> 3)
        w = txl.alloc_local([32], "uint32")
        tbase = txl.local_scalar("int64", init=off + i642(R0) * ld + i642(C0))
        interior = (R0 + 128 <= M) & (C0 + 128 <= N)
        tok_load = iket_range("global-load")
        with txl.if_(interior):
            with txl.then_():
                for it in range(4):
                    for h in range(2):
                        b = 8 * it + 4 * h
                        txl.ptx.ld.global_.v4.b32(w[b], w[b + 1], w[b + 2], w[b + 3], txl.address_of(x[tbase + i642(r0 + 32 * it) * ld + i642((k + 8 * h) * 8)]))
            with txl.else_():
                for it in range(4):
                    for h in range(2):
                        b = 8 * it + 4 * h
                        for e in range(4):
                            txl.assign(w[b + e], u322(0))
                        with txl.if_((R0 + r0 + 32 * it < M) & (C0 + (k + 8 * h) * 8 < N)), txl.then_():
                            txl.ptx.ld.global_.v4.b32(w[b], w[b + 1], w[b + 2], w[b + 3], txl.address_of(x[tbase + i642(r0 + 32 * it) * ld + i642((k + 8 * h) * 8)]))
        iket_end(tok_load)
        tok_stage = iket_range("shared-stage")
        for it in range(4):
            r = r0 + 32 * it
            g = (r >> 4) & 7
            for h in range(2):
                b = 8 * it + 4 * h
                txl.ptx.st.shared.v4.b32(tile.ptr_to([r * 64 + (((k + 8 * h) ^ h ^ g) << 2)]), w[b], w[b + 1], w[b + 2], w[b + 3])
        iket_end(tok_stage)
        # L2 prefetch of the tile one resident wave ahead (interior tiles only), one 256 B row per thread
        tok_prefetch = iket_range("l2-prefetch")
        fid = cid + pf
        fm = txl.local_scalar("int32", init=txl.truncdiv(fid, nbn))
        fn = fid - fm * nbn
        tq = txl.thread_id()
        with txl.if_((tq < 128) & (fm * 128 + 128 <= M) & (fn * 128 + 128 <= N)), txl.then_():
            txl.ptx.cp.async_.bulk.prefetch.L2.global_(txl.address_of(x[off + i642(fm * 128 + tq) * ld + i642(fn * 128)]), u322(256))
        iket_end(tok_prefetch)
        tok_row = iket_range("row-quant")
        qtile = txl.local_scalar("int64", init=i642(R0) * N64 + i642(C0))
        for it in range(4):
            grow = R0 + r0 + 32 * it
            m = txl.local_scalar("uint32", init=w[8 * it])
            for e in range(1, 8):
                txl.ptx.max.xorsign.abs.bf16x2(m, m, w[8 * it + e])
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
                b = 8 * it + 4 * h
                o.append(pack16(q_bf16x2(w[b], rcp2), q_bf16x2(w[b + 1], rcp2)))
                o.append(pack16(q_bf16x2(w[b + 2], rcp2), q_bf16x2(w[b + 3], rcp2)))
            with txl.if_(grow < M), txl.then_():
                qa = qtile + i642(r0 + 32 * it) * N64 + i642(k * 8)
                with txl.if_(C0 + 128 <= N):
                    with txl.then_():
                        txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa]), o[0], o[1])
                        txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64]), o[2], o[3])
                    with txl.else_():
                        for h in range(2):
                            with txl.if_(C0 + k * 8 + 64 * h < N), txl.then_():
                                txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64 * h]), o[2 * h], o[2 * h + 1])
            with txl.if_(k == 0), txl.then_():
                txl.ptx.st.shared.b32(rsc.ptr_to([r0 + 32 * it]), sbits)
        iket_end(tok_row)
        tok_barrier = iket_range("cta-barrier")
        txl.ptx.bar.sync(u322(0), u322(NTHREADS))
        iket_end(tok_barrier)
        tok_row_scale = iket_range("row-scale-store")
        t = txl.thread_id()
        with txl.if_((t < 128) & (R0 + t < M)), txl.then_():
            rs = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(rs, rsc.ptr_to([t]))
            txl.ptx.st.global_.b32(txl.address_of(xs[i642(tn) * M64 + i642(R0 + t)]), rs)
        iket_end(tok_row_scale)
        tok_column = iket_range("column-quant")
        rg = lane & 7
        pl = lane >> 3
        v = txl.alloc_local([16], "uint32")
        for it2 in range(2):
            p = warp * 4 + pl + 32 * it2
            c = p >> 2
            co = rg * 16 * 64 + ((c ^ ((c >> 3) & 1) ^ rg) << 2) + (p & 3)
            for e in range(16):
                txl.ptx.ld.shared.b32(v[e], tile.ptr_to([co + e * 64]))
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
            with txl.if_((col < N) & (R0 + rg * 16 < M)), txl.then_():
                ta = txl.local_scalar("int64", init=i642(col) * M64 + i642(R0 + rg * 16))
                txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta]), lw[0], lw[1], lw[2], lw[3])
                txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta + M64]), hw[0], hw[1], hw[2], hw[3])
                with txl.if_(rg == 0), txl.then_():
                    sa = txl.local_scalar("int64", init=i642(tm) * N64 + i642(col))
                    txl.ptx.st.global_.v2.b32(txl.address_of(xts[sa]), s_lo, s_hi)
        iket_end(tok_column)

    return fp8_quant_grad_one


@functools.cache
def one_kernel():
    return entry_function(build_one().compile(TARGET))


@functools.cache
def num_sms(dev):
    return torch.cuda.get_device_properties(dev).multi_processor_count



def build_split4(MINB=6):
    """
    Small grids: four CTAs per 128x128 tile. Roles 0/1 quantize rows 0-63 / 64-127 (all 128 columns)
    straight from registers; roles 2/3 quantize columns 0-63 / 64-127 (all 128 rows) through a 16 KB
    XOR-swizzled shared tile. Each tile is read twice (the second read hits L2).
    """

    @txl.kernel()
    def fp8_quant_grad_split4(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), xq: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), xs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), xtq: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), xts: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, ld: txl.int64, off: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=((N + 127) // 128) * ((M + 127) // 128) * 4, block=(NTHREADS // 32) * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=MINB))
        cid = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        pool = txl.smem_pool()
        tile = pool.alloc([BLOCK * 32], "uint32", align=128)
        nbn = (N + 127) >> 7
        role = cid & 3
        tid = cid >> 2
        tm = txl.local_scalar("int32", init=txl.truncdiv(tid, nbn))
        tn = txl.local_scalar("int32", init=tid - tm * nbn)
        R0 = tm * 128
        C0 = tn * 128
        M64 = i642(M)
        N64 = i642(N)
        k = lane & 7
        r0 = warp * 4 + (lane >> 3)
        w = txl.alloc_local([16], "uint32")
        with txl.if_(role < 2):
            with txl.then_():
                # rows R1 .. R1+63, chunks k and k+8 of each row
                R1 = R0 + 64 * role
                tbase = txl.local_scalar("int64", init=off + i642(R1) * ld + i642(C0))
                with txl.if_((R1 + 64 <= M) & (C0 + 128 <= N)):
                    with txl.then_():
                        for it in range(2):
                            for h in range(2):
                                b = 8 * it + 4 * h
                                txl.ptx.ld.global_.v4.b32(w[b], w[b + 1], w[b + 2], w[b + 3], txl.address_of(x[tbase + i642(r0 + 32 * it) * ld + i642((k + 8 * h) * 8)]))
                    with txl.else_():
                        for it in range(2):
                            for h in range(2):
                                b = 8 * it + 4 * h
                                for e in range(4):
                                    txl.assign(w[b + e], u322(0))
                                with txl.if_((R1 + r0 + 32 * it < M) & (C0 + (k + 8 * h) * 8 < N)), txl.then_():
                                    txl.ptx.ld.global_.v4.b32(w[b], w[b + 1], w[b + 2], w[b + 3], txl.address_of(x[tbase + i642(r0 + 32 * it) * ld + i642((k + 8 * h) * 8)]))
                qtile = txl.local_scalar("int64", init=i642(R1) * N64 + i642(C0))
                for it in range(2):
                    grow = R1 + r0 + 32 * it
                    m = txl.local_scalar("uint32", init=w[8 * it])
                    for e in range(1, 8):
                        txl.ptx.max.xorsign.abs.bf16x2(m, m, w[8 * it + e])
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
                        b = 8 * it + 4 * h
                        o.append(pack16(q_bf16x2(w[b], rcp2), q_bf16x2(w[b + 1], rcp2)))
                        o.append(pack16(q_bf16x2(w[b + 2], rcp2), q_bf16x2(w[b + 3], rcp2)))
                    with txl.if_(grow < M), txl.then_():
                        qa = qtile + i642(r0 + 32 * it) * N64 + i642(k * 8)
                        with txl.if_(C0 + 128 <= N):
                            with txl.then_():
                                txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa]), o[0], o[1])
                                txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64]), o[2], o[3])
                            with txl.else_():
                                for h in range(2):
                                    with txl.if_(C0 + k * 8 + 64 * h < N), txl.then_():
                                        txl.ptx.st.global_.v2.b32(txl.address_of(xq[qa + 64 * h]), o[2 * h], o[2 * h + 1])
                        with txl.if_(k == 0), txl.then_():
                            txl.ptx.st.global_.b32(txl.address_of(xs[i642(tn) * M64 + i642(grow)]), sbits)
            with txl.else_():
                # columns C1 .. C1+63 of all 128 rows: lane k loads chunk k (8 columns) of rows r0 + 32 * it
                C1 = C0 + 64 * (role - 2)
                tbase = txl.local_scalar("int64", init=off + i642(R0) * ld + i642(C1))
                with txl.if_((R0 + 128 <= M) & (C1 + 64 <= N)):
                    with txl.then_():
                        for it in range(4):
                            b = 4 * it
                            txl.ptx.ld.global_.v4.b32(w[b], w[b + 1], w[b + 2], w[b + 3], txl.address_of(x[tbase + i642(r0 + 32 * it) * ld + i642(k * 8)]))
                    with txl.else_():
                        for it in range(4):
                            b = 4 * it
                            for e in range(4):
                                txl.assign(w[b + e], u322(0))
                            with txl.if_((R0 + r0 + 32 * it < M) & (C1 + k * 8 < N)), txl.then_():
                                txl.ptx.ld.global_.v4.b32(w[b], w[b + 1], w[b + 2], w[b + 3], txl.address_of(x[tbase + i642(r0 + 32 * it) * ld + i642(k * 8)]))
                for it in range(4):
                    r = r0 + 32 * it
                    b = 4 * it
                    txl.ptx.st.shared.v4.b32(tile.ptr_to([r * 32 + ((k ^ ((r >> 4) & 7)) << 2)]), w[b], w[b + 1], w[b + 2], w[b + 3])
                txl.ptx.bar.sync(u322(0), u322(NTHREADS))
                rg = lane & 7
                p = warp * 4 + (lane >> 3)
                c = p >> 2
                co = rg * 16 * 32 + ((c ^ rg) << 2) + (p & 3)
                v = txl.alloc_local([16], "uint32")
                for e in range(16):
                    txl.ptx.ld.shared.b32(v[e], tile.ptr_to([co + e * 32]))
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
                col = C1 + 2 * p
                with txl.if_((col < N) & (R0 + rg * 16 < M)), txl.then_():
                    ta = txl.local_scalar("int64", init=i642(col) * M64 + i642(R0 + rg * 16))
                    txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta]), lw[0], lw[1], lw[2], lw[3])
                    txl.ptx.st.global_.v4.b32(txl.address_of(xtq[ta + M64]), hw[0], hw[1], hw[2], hw[3])
                    with txl.if_(rg == 0), txl.then_():
                        sa = txl.local_scalar("int64", init=i642(tm) * N64 + i642(col))
                        txl.ptx.st.global_.v2.b32(txl.address_of(xts[sa]), s_lo, s_hi)

    return fp8_quant_grad_split4


@functools.cache
def split4_kernel(MINB=6):
    return entry_function(build_split4(MINB).compile(TARGET))


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

    @txl.kernel()
    def fp8_quant_grad_generic(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), xq: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), xs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), xtq: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), xts: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, s0: txl.int64, s1: txl.int64, off: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[(N + 127) // 128, (M + 127) // 128], block=(NTHREADS // 32) * 32))
        cid = txl.cta_id()
        tn = cid[0]
        tm = cid[1]
        t = txl.thread_id()
        pool = txl.smem_pool()
        tile = pool.alloc([BLOCK * PITCH], "float32", align=16)
        red = pool.alloc([2 * BLOCK], "float32", align=16)
        R0 = tm * 128
        C0 = tn * 128

        with txl.serial(64) as it:
            e = it * NTHREADS + t
            rr = e >> 7
            c = e & 127
            val = txl.local_scalar("float32", init=txl.float32(0.0))
            with txl.if_(R0 + rr < M), txl.then_():
                with txl.if_(C0 + c < N), txl.then_():
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
        with txl.if_(R0 + r < M), txl.then_():
            with txl.serial(64) as i:
                cc = C0 + h * 64 + i
                with txl.if_(cc < N), txl.then_():
                    txl.ptx.ld.shared.f32(tmp, txl.address_of(tile[r * PITCH + h * 64 + i]))
                    b = q_f32(tmp, rcp)
                    txl.ptx.st.global_.b8(txl.address_of(xq[i642(R0 + r) * i642(N) + i642(cc)]), txl.Cast("uint8", b))
            with txl.if_(h == 0), txl.then_():
                txl.ptx.st.global_.b32(txl.address_of(xs[i642(tn) * i642(M) + i642(R0 + r)]), sbits)

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
        with txl.if_(C0 + c < N), txl.then_():
            with txl.serial(64) as i:
                rr = R0 + g * 64 + i
                with txl.if_(rr < M), txl.then_():
                    txl.ptx.ld.shared.f32(tmp, txl.address_of(tile[(g * 64 + i) * PITCH + c]))
                    b = q_f32(tmp, crcp)
                    txl.ptx.st.global_.b8(txl.address_of(xtq[i642(C0 + c) * i642(M) + i642(rr)]), txl.Cast("uint8", b))
            with txl.if_(g == 0), txl.then_():
                txl.ptx.st.global_.b32(txl.address_of(xts[i642(tm) * i642(N) + i642(C0 + c)]), csb)

    return entry_function(fp8_quant_grad_generic.compile(TARGET))


GENERIC_DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def fp8_quant_grad_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    nbn, nbm = cdiv(N, BLOCK), cdiv(M, BLOCK)
    dev = x.device
    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev)
    scale_tok = torch.empty((nbn, M), dtype=torch.float32, device=dev).T
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=dev)
    scale_t = torch.empty((nbm, N), dtype=torch.float32, device=dev).T
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
    if fast:
        outs = (out_tok.view(torch.uint8).view(-1), scale_tok.T.view(-1).view(torch.int32), out_t.view(torch.uint8).view(-1), scale_t.T.view(-1).view(torch.int32))
        sms = num_sms(dev)
        if 2 * nbn * nbm < 3 * sms:
            split4_kernel()(flat, *outs, M, N, s0, off)
        else:
            one_kernel()(flat, *outs, M, N, s0, off, 3 * sms)
    else:
        generic_kernel(GENERIC_DTYPES[x.dtype])(
            flat,
            out_tok.view(torch.uint8).view(-1),
            scale_tok.T.view(-1).view(torch.int32),
            out_t.view(torch.uint8).view(-1),
            scale_t.T.view(-1).view(torch.int32),
            M,
            N,
            s0,
            s1,
            off,
        )
    return out_tok, scale_tok, out_t, scale_t
