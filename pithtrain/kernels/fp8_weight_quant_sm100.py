"""
SM100 fp8_weight_quant in TIRx-lite: 128x128 blockwise FP8 of weights and their transposes,
and the FP8 parameter all-gather kernels (shard amax / cast, E8M0 scales, packed transpose),
including one-launch multi variants of the shard amax and scales.

Every kernel is a @txl.kernel compiled once per process (functools.cache, keyed only on
compile-time constants) and launched through tvm-ffi on torch's current stream.

Evolved with TIRx-harness: run weight, frontier member tile-pdl (commit 24e4366, 4102 us vs 5452 us, 1.329x over the
previous SM100 path in the task's benchmark). Compiled once per process by
tvm_tirx.tirx.compile(..., pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc as before; a cubin (sm_100a SASS only, as the old -gencode build), since -arch sm_100a would also emit compute_100 PTX, which rejects sm_100a-only instructions

__all__ = ["fp8_quant_weight_sm100", "fp8_quant_weight_grouped_sm100", "fp8_shard_amax_sm100", "fp8_shard_cast_sm100", "fp8_scales_sm100", "fp8_pack_transpose_sm100", "fp8_shard_amax_multi_sm100", "fp8_scales_multi_sm100"]

BLOCK = 128
NT = 256  # threads per CTA
NTQ = 128  # small-grid quant: four resident 4-warp CTAs instead of three 8-warp CTAs
FULL = 0xFFFFFFFF
RCP_448 = 1.0 / 448.0
# Streaming fp32 shard loads: no L1 allocation + 256B L2 prefetch (pure reads 5.0 -> 5.8 TB/s).
LDG_V4 = "ld.global.nc.L1::no_allocate.L2::256B.v4.b32"
LDG_B32 = "ld.global.nc.L1::no_allocate.L2::256B.b32"


# Every kernel is launched with programmatic stream serialization (pdl_entry()) and starts with pdl().


def tensor_nd(dtype, nd):
    """
    A rank-nd global tensor parameter, every extent its own symbolic int64.
    """
    return txl.Tensor(tuple(txl.int64() for _ in range(nd)), dtype, data_alignment=1)


def cdiv(a, b):
    return (a + b - 1) // b


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def compile_v(kern):
    """
    Compile once; return the module's entry function itself (calling the module looks the entry up
    on every call, ~4 us of host time). tvm-ffi runs it on torch's current stream, which it takes
    from the torch tensor arguments.
    """
    mod = kern.compile(TARGET)
    EXECUTABLES.append(mod)
    return mod[kern.name]


# ---------------------------------------------------------------- traced helpers


def pdl_entry(grid):
    """
    Device entry of every kernel here: NT threads, programmatic stream serialization.
    """
    txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid, block=NT, programmatic_stream_serialization=True))


def pdl():
    """
    Programmatic dependent launch: let the next kernel in the stream get scheduled now, and
    wait for the previous one (completion + memory flush) before touching global memory.
    """
    txl.ptx.griddepcontrol.wait()
    txl.ptx.griddepcontrol.launch_dependents()


def i642(x):
    return txl.Cast("int64", x)


def scale_bits(amax_f32):
    """
    (scale_bits, rcp_bits) of pith-train's compute_fp8_scale (SM100, native E8M0).
    """
    amc = txl.local_scalar("float32")
    txl.ptx.max.f32(amc, amax_f32, txl.float32(1e-4))
    si = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(si, amc, txl.float32(RCP_448))
    pair = txl.local_scalar("uint16")
    txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(pair, txl.float32(0.0), si)
    sb = txl.local_scalar("uint32")
    txl.ptx.max.u32(sb, (txl.Cast("uint32", pair) & txl.uint32(0xFF)) << txl.uint32(23), txl.uint32(1 << 23))
    rb = txl.local_scalar("uint32", init=txl.uint32(254 << 23) - sb)
    return sb, rb


def e4m3x22(hi, lo, rcp):
    """
    Two f32 values times rcp -> two e4m3 bytes in the low 16 bits of a u32.
    """
    a = txl.local_scalar("float32")
    b = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(a, hi, rcp)
    txl.ptx.mul.rn.f32(b, lo, rcp)
    p = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(p, a, b)
    return txl.Cast("uint32", p)


def bf16_word_to_fp8x2(w, rcp):
    lo = txl.reinterpret(w << txl.uint32(16), ty="float32")
    hi = txl.reinterpret(w & txl.uint32(0xFFFF0000), ty="float32")
    return e4m3x22(hi, lo, rcp)


def cta_max_u32(m, red_s, warp, lane):
    """
    Max of a u32 over the CTA (8 warps); every thread gets the result.
    """
    r = txl.local_scalar("uint32")
    txl.ptx.redux_sync.max.u32(r, m, txl.uint32(FULL))
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.b32(red_s.ptr_to([warp]), r)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NT))
    out = txl.local_scalar("uint32", init=txl.uint32(0))
    for i in range(NT // 32):
        v = txl.local_scalar("uint32")
        txl.ptx.ld.shared.b32(v, red_s.ptr_to([i]))
        txl.ptx.max.u32(out, out, v)
    return out


def cta_max_u32_q128(m, red_s, warp, lane):
    """
    Max of a u32 over the four-warp small-grid quant CTA.
    """
    r = txl.local_scalar("uint32")
    txl.ptx.redux_sync.max.u32(r, m, txl.uint32(FULL))
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.b32(red_s.ptr_to([warp]), r)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTQ))
    out = txl.local_scalar("uint32", init=txl.uint32(0))
    for i in range(NTQ // 32):
        v = txl.local_scalar("uint32")
        txl.ptx.ld.shared.b32(v, red_s.ptr_to([i]))
        txl.ptx.max.u32(out, out, v)
    return out


def prmt2(a, b, sel):
    d = txl.local_scalar("uint32")
    txl.ptx.prmt.b32(d, a, b, txl.uint32(sel))
    return d


def t4x4(a, b, c, d):
    """
    4x4 byte transpose: rows a..d (4 bytes each) -> columns.
    """
    t0 = prmt2(a, b, 0x5140)
    t1 = prmt2(a, b, 0x7362)
    t2 = prmt2(c, d, 0x5140)
    t3 = prmt2(c, d, 0x7362)
    return (prmt2(t0, t2, 0x5410), prmt2(t0, t2, 0x7632), prmt2(t1, t3, 0x5410), prmt2(t1, t3, 0x7632))


def transpose_store(B, t_s, kc, rg):
    """
    B[r][h]: 8 rows (n) x 8 bytes (k) as 2 words per row. Writes the 8x8 block transposed into
    the [128 k][128 n] byte tile t_s (u32 words, 8-byte slots XOR-swizzled by k >> 3).
    """
    cols = [[None, None] for _ in range(8)]  # cols[kk][half of n]
    for h in range(2):
        for rh in range(2):
            outs = t4x4(*[B[4 * rh + i][h] for i in range(4)])
            for j in range(4):
                cols[4 * h + j][rh] = outs[j]
    slot = rg ^ kc
    for kk in range(8):
        row = kc * 8 + kk
        txl.ptx.st.shared.v2.b32(t_s.ptr_to([row * 32 + slot * 2]), cols[kk][0], cols[kk][1])


def transpose_store_q128(B, t_s, kc, rg):
    """
    Transpose one 16x8 byte register block into the same swizzled tile as
    _transpose_store.  A four-warp CTA gives each thread twice as many rows, but
    preserves the exact shared layout consumed by the vectorized flush.
    """
    cols = [[None, None, None, None] for _ in range(8)]
    for h in range(2):
        for rh in range(4):
            outs = t4x4(*[B[4 * rh + i][h] for i in range(4)])
            for j in range(4):
                cols[4 * h + j][rh] = outs[j]
    for kk in range(8):
        row = kc * 8 + kk
        slot0 = (rg * 2) ^ kc
        slot1 = (rg * 2 + 1) ^ kc
        txl.ptx.st.shared.v2.b32(t_s.ptr_to([row * 32 + slot0 * 2]), cols[kk][0], cols[kk][1])
        txl.ptx.st.shared.v2.b32(t_s.ptr_to([row * 32 + slot1 * 2]), cols[kk][2], cols[kk][3])


def transpose_flush(t_s, tid, out_t_ptr, k_base, n_base, Kd, Nd):
    """
    Write the transposed tile from t_s: tile row k (0..127) -> out_t row k_base + k, columns
    n_base .. n_base + 127 in 16-byte chunks (requires Nd % 16 == 0).
    """
    j = tid & 7
    for p in range(BLOCK // 32):
        row = p * 32 + (tid >> 3)
        s = row >> 3
        pb = ((j * 2) ^ s) & 14
        y = txl.alloc_local([4], "uint32")
        txl.ptx.ld.shared.v4.b32(y[0], y[1], y[2], y[3], t_s.ptr_to([row * 32 + pb * 2]))
        odd = (s & 1) == 1
        gk = k_base + row
        gn = n_base + j * 16
        valid = txl.And(gk < Kd, gn < Nd)
        txl.ptx.st.global_.v4.b32(
            out_t_ptr(i642(gk) * i642(Nd) + i642(gn)),
            txl.Select(odd, y[2], y[0]),
            txl.Select(odd, y[3], y[1]),
            txl.Select(odd, y[0], y[2]),
            txl.Select(odd, y[1], y[3]),
            pred=valid,
        )


def transpose_flush_q128(t_s, tid, out_t_ptr, k_base, n_base, Kd, Nd):
    """
    Four-warp distribution of the unchanged 128x128 shared transpose.
    """
    j = tid & 7
    for p in range(BLOCK // 16):
        row = p * 16 + (tid >> 3)
        s = row >> 3
        pb = ((j * 2) ^ s) & 14
        y = txl.alloc_local([4], "uint32")
        txl.ptx.ld.shared.v4.b32(y[0], y[1], y[2], y[3], t_s.ptr_to([row * 32 + pb * 2]))
        odd = (s & 1) == 1
        gk = k_base + row
        gn = n_base + j * 16
        valid = txl.And(gk < Kd, gn < Nd)
        txl.ptx.st.global_.v4.b32(
            out_t_ptr(i642(gk) * i642(Nd) + i642(gn)),
            txl.Select(odd, y[2], y[0]),
            txl.Select(odd, y[3], y[1]),
            txl.Select(odd, y[0], y[2]),
            txl.Select(odd, y[1], y[3]),
            pred=valid,
        )


def tile_coords(bid, R, C):
    tc = txl.local_scalar("int32", init=bid % C)
    t2 = txl.local_scalar("int32", init=bid // C)
    tr = txl.local_scalar("int32", init=t2 % R)
    g = txl.local_scalar("int32", init=t2 // R)
    return g, tr, tc


# ---------------------------------------------------------------- quant_weight (fast path)


@functools.cache
def quant_vec():
    """
    bf16, unit column stride, K % 8 == 0, N % 16 == 0, 16-byte aligned rows.
    """

    @txl.kernel()
    def fp8_quant_vec(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), sc: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), sct: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, sxg: txl.int64, sxn: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        g, tr, tc = tile_coords(txl.cta_id(), R, C)
        smem = txl.smem_pool()
        t_s = smem.alloc([BLOCK * BLOCK // 4], "uint32")
        red_s = smem.alloc([NT // 32], "uint32")

        kc = tid & 15
        rg = tid >> 4
        n0 = tr * BLOCK + rg * 8
        k = tc * BLOCK + kc * 8
        kvalid = k < K
        xb = txl.local_scalar("int64", init=i642(g) * sxg + i642(n0) * sxn + i642(k))
        d = txl.alloc_local([32], "uint32")

        def load(r):
            txl.ptx.ld.global_.nc.v4.b32(d[4 * r], d[4 * r + 1], d[4 * r + 2], d[4 * r + 3], x.ptr_to([xb + sxn * r]))

        with txl.if_(txl.And(tc * BLOCK + BLOCK <= K, tr * BLOCK + BLOCK <= N)):
            with txl.then_():
                for r in range(8):
                    load(r)
            with txl.else_():
                for r in range(8):
                    for i in range(4):
                        txl.ptx.mov.b32(d[4 * r + i], txl.uint32(0))
                    with txl.if_(txl.And(kvalid, n0 + r < N)), txl.then_():
                        load(r)
        m = txl.local_scalar("uint32", init=txl.uint32(0))
        for i in range(32):
            txl.ptx.max.u16x2(m, m, d[i] & txl.uint32(0x7FFF7FFF))
        txl.ptx.max.u32(m, m & txl.uint32(0xFFFF), m >> txl.uint32(16))
        am = cta_max_u32(m, red_s, warp, lane)
        sb, rb = scale_bits(txl.reinterpret(am << txl.uint32(16), ty="float32"))
        rcp = txl.reinterpret(rb, ty="float32")

        ob = txl.local_scalar("int64", init=(i642(g) * i642(N) + i642(n0)) * i642(K) + i642(k))
        B = []
        for r in range(8):
            q = [bf16_word_to_fp8x2(d[4 * r + i], rcp) for i in range(4)]
            w0 = txl.local_scalar("uint32", init=q[0] | (q[1] << txl.uint32(16)))
            w1 = txl.local_scalar("uint32", init=q[2] | (q[3] << txl.uint32(16)))
            B.append((w0, w1))
            txl.ptx.st.global_.v2.b32(
                out.ptr_to([ob + i642(K) * r]), w0, w1, pred=txl.And(kvalid, n0 + r < N)
            )
        transpose_store(B, t_s, kc, rg)
        with txl.if_(tid == 0), txl.then_():
            sf = txl.reinterpret(sb, ty="float32")
            txl.ptx.st.global_.f32(sc.ptr_to([(i642(g) * i642(R) + i642(tr)) * i642(C) + i642(tc)]), sf)
            txl.ptx.st.global_.f32(sct.ptr_to([(i642(g) * i642(C) + i642(tc)) * i642(R) + i642(tr)]), sf)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NT))
        tbase = txl.local_scalar("int64", init=i642(g) * i642(K) * i642(N))
        transpose_flush(
            t_s, tid, lambda off: out_t.ptr_to([tbase + off]), tc * BLOCK, tr * BLOCK, K, N
        )

    return compile_v(fp8_quant_vec)


@functools.cache
def quant_vec_q128():
    """
    Four-warp quant tile for moderate grids.  Each thread owns 16x8 bf16
    values instead of 8x8.  The larger per-thread register tile is intended to
    raise resident CTAs from three to four, reducing wave quantization on the
    dense rows while retaining 16-byte global loads and the existing transpose
    layout.
    """

    @txl.kernel()
    def fp8_quant_vec_q128(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), sc: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), sct: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, sxg: txl.int64, sxn: txl.int64, num_tiles: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_tiles, block=NTQ, programmatic_stream_serialization=True))
        pdl()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        g, tr, tc = tile_coords(txl.cta_id(), R, C)
        smem = txl.smem_pool()
        t_s = smem.alloc([BLOCK * BLOCK // 4], "uint32")
        red_s = smem.alloc([NTQ // 32], "uint32")

        kc = tid & 15
        rg = tid >> 4
        n0 = tr * BLOCK + rg * 16
        k = tc * BLOCK + kc * 8
        kvalid = k < K
        xb = txl.local_scalar("int64", init=i642(g) * sxg + i642(n0) * sxn + i642(k))
        d = txl.alloc_local([64], "uint32")

        def load(r):
            txl.ptx.ld.global_.nc.v4.b32(d[4 * r], d[4 * r + 1], d[4 * r + 2], d[4 * r + 3], x.ptr_to([xb + sxn * r]))

        with txl.if_(txl.And(tc * BLOCK + BLOCK <= K, tr * BLOCK + BLOCK <= N)):
            with txl.then_():
                for r in range(16):
                    load(r)
            with txl.else_():
                for r in range(16):
                    for i in range(4):
                        txl.ptx.mov.b32(d[4 * r + i], txl.uint32(0))
                    with txl.if_(txl.And(kvalid, n0 + r < N)), txl.then_():
                        load(r)
        m = txl.local_scalar("uint32", init=txl.uint32(0))
        for i in range(64):
            txl.ptx.max.u16x2(m, m, d[i] & txl.uint32(0x7FFF7FFF))
        txl.ptx.max.u32(m, m & txl.uint32(0xFFFF), m >> txl.uint32(16))
        am = cta_max_u32_q128(m, red_s, warp, lane)
        sb, rb = scale_bits(txl.reinterpret(am << txl.uint32(16), ty="float32"))
        rcp = txl.reinterpret(rb, ty="float32")

        ob = txl.local_scalar("int64", init=(i642(g) * i642(N) + i642(n0)) * i642(K) + i642(k))
        B = []
        for r in range(16):
            q = [bf16_word_to_fp8x2(d[4 * r + i], rcp) for i in range(4)]
            w0 = txl.local_scalar("uint32", init=q[0] | (q[1] << txl.uint32(16)))
            w1 = txl.local_scalar("uint32", init=q[2] | (q[3] << txl.uint32(16)))
            B.append((w0, w1))
            txl.ptx.st.global_.v2.b32(
                out.ptr_to([ob + i642(K) * r]), w0, w1, pred=txl.And(kvalid, n0 + r < N)
            )
        transpose_store_q128(B, t_s, kc, rg)
        with txl.if_(tid == 0), txl.then_():
            sf = txl.reinterpret(sb, ty="float32")
            txl.ptx.st.global_.f32(sc.ptr_to([(i642(g) * i642(R) + i642(tr)) * i642(C) + i642(tc)]), sf)
            txl.ptx.st.global_.f32(sct.ptr_to([(i642(g) * i642(C) + i642(tc)) * i642(R) + i642(tr)]), sf)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NTQ))
        tbase = txl.local_scalar("int64", init=i642(g) * i642(K) * i642(N))
        transpose_flush_q128(
            t_s, tid, lambda off: out_t.ptr_to([tbase + off]), tc * BLOCK, tr * BLOCK, K, N
        )

    return compile_v(fp8_quant_vec_q128)


# ---------------------------------------------------------------- quant_weight (generic)


@functools.cache
def quant_generic(dtype):
    """
    Any dtype (bf16 / f32), any strides, any N, K: one element per thread step.
    """
    is_bf16 = dtype == "bfloat16"

    @txl.kernel()
    def fp8_quant_generic(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), sc: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), sct: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, sxg: txl.int64, sxn: txl.int64, sxk: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        g, tr, tc = tile_coords(txl.cta_id(), R, C)
        smem = txl.smem_pool()
        red_s = smem.alloc([NT // 32], "uint32")
        c = tid & 127
        k = tc * BLOCK + c
        kvalid = k < K
        r0 = tid >> 7
        xb = txl.local_scalar("int64", init=i642(g) * sxg + i642(tr * BLOCK + r0) * sxn + i642(k) * sxk)
        steps = BLOCK // (NT // BLOCK)
        v = txl.alloc_local([steps], "uint32")
        m = txl.local_scalar("uint32", init=txl.uint32(0))
        for i in range(steps):
            n = tr * BLOCK + r0 + 2 * i
            txl.ptx.mov.b32(v[i], txl.uint32(0))
            with txl.if_(txl.And(kvalid, n < N)), txl.then_():
                if is_bf16:
                    h = txl.local_scalar("uint16")
                    txl.ptx.ld.global_.nc.b16(h, x.ptr_to([xb + sxn * (2 * i)]))
                    txl.ptx.mov.b32(v[i], txl.Cast("uint32", h) << txl.uint32(16))
                else:
                    txl.ptx.ld.global_.nc.b32(v[i], x.ptr_to([xb + sxn * (2 * i)]))
            txl.ptx.max.u32(m, m, v[i] & txl.uint32(0x7FFFFFFF))
        am = cta_max_u32(m, red_s, warp, lane)
        sb, rb = scale_bits(txl.reinterpret(am, ty="float32"))
        rcp = txl.reinterpret(rb, ty="float32")
        for i in range(steps):
            n = tr * BLOCK + r0 + 2 * i
            q = e4m3x22(txl.float32(0.0), txl.reinterpret(v[i], ty="float32"), rcp)
            with txl.if_(txl.And(kvalid, n < N)), txl.then_():
                byte = txl.Cast("uint8", q)
                txl.ptx.st.global_.b8(out.ptr_to([(i642(g) * i642(N) + i642(n)) * i642(K) + i642(k)]), byte)
                txl.ptx.st.global_.b8(out_t.ptr_to([(i642(g) * i642(K) + i642(k)) * i642(N) + i642(n)]), byte)
        with txl.if_(tid == 0), txl.then_():
            sf = txl.reinterpret(sb, ty="float32")
            txl.ptx.st.global_.f32(sc.ptr_to([(i642(g) * i642(R) + i642(tr)) * i642(C) + i642(tc)]), sf)
            txl.ptx.st.global_.f32(sct.ptr_to([(i642(g) * i642(C) + i642(tc)) * i642(R) + i642(tr)]), sf)

    return compile_v(fp8_quant_generic)


def flat_view(x):
    """
    A 1-D view over the storage span x addresses (x itself when contiguous).
    """
    if x.is_contiguous():
        return x.view(-1)
    span = 1 + sum((s - 1) * st for s, st in zip(x.shape, x.stride()))
    return x.as_strided((span,), (1,))


def u82(t):
    return t.view(torch.uint8).view(-1)


def quant(x3, G, N, K, sxg, sxn, sxk, out, scale, out_t, scale_t):
    R, C = cdiv(N, BLOCK), cdiv(K, BLOCK)
    tiles = G * R * C
    if tiles == 0:
        return
    xf = flat_view(x3)
    fast = (
        x3.dtype == torch.bfloat16
        and sxk == 1
        and K % 8 == 0
        and N % 16 == 0
        and sxn % 8 == 0
        and sxg % 8 == 0
        and xf.data_ptr() % 16 == 0
    )
    if fast:
        kern = quant_vec_q128() if G == 1 else quant_vec()
        kern(xf, u82(out), u82(out_t), scale.view(-1), scale_t.view(-1), N, K, R, C, sxg, sxn, tiles)
    else:
        dt = "bfloat16" if x3.dtype == torch.bfloat16 else "float32"
        xf = xf if x3.dtype in (torch.bfloat16, torch.float32) else xf.float()
        quant_generic(dt)(xf, u82(out), u82(out_t), scale.view(-1), scale_t.view(-1), N, K, R, C, sxg, sxn, sxk, tiles)


def fp8_quant_weight_sm100(x: torch.Tensor):
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    N, K = x.shape
    R, C = cdiv(N, BLOCK), cdiv(K, BLOCK)
    dev = x.device
    out = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=dev)
    scale = torch.empty((R, C), dtype=torch.float32, device=dev)
    out_t = torch.empty((K, N), dtype=torch.float8_e4m3fn, device=dev)
    scale_t = torch.empty((C, R), dtype=torch.float32, device=dev)
    if x.numel():
        quant(x, 1, N, K, 0, x.stride(0), x.stride(1), out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t


def fp8_quant_weight_grouped_sm100(x: torch.Tensor):
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    G, N, K = x.shape
    R, C = cdiv(N, BLOCK), cdiv(K, BLOCK)
    dev = x.device
    out = torch.empty((G, N, K), dtype=torch.float8_e4m3fn, device=dev)
    scale = torch.empty((G, R, C), dtype=torch.float32, device=dev)
    out_t = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=dev)
    scale_t = torch.empty((G, C, R), dtype=torch.float32, device=dev)
    if x.numel():
        quant(x, G, N, K, x.stride(0), x.stride(1), x.stride(2), out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t


# ---------------------------------------------------------------- shard amax / cast


def shard_tile_body(tid, warp, lane, red_s, xaddr, oaddr, amax_addr, L, K, row_offset, gl, tr, tc, cast, V, sub=0, RB=BLOCK):
    """
    Rows [sub * RB, sub * RB + RB) of one 128x128 tile of a row shard (global rows
    [row_offset, row_offset + L)). V elements per load (4: K % 4 == 0; 1: any K). cast False
    writes the tile's bf16 amax (RB must be BLOCK); True casts.
    """
    lanes_per_row = BLOCK // V
    rows_per_pass = NT // lanes_per_row
    passes = RB // rows_per_pass
    assert passes >= 1 and (cast or RB == BLOCK)
    cl = tid % lanes_per_row
    rl = tid // lanes_per_row
    k = tc * BLOCK + cl * V
    kvalid = k < K
    lr0 = txl.local_scalar("int32", init=tr * BLOCK + sub * RB + rl - row_offset)
    xb = txl.local_scalar("int64", init=(i642(gl) * i642(L) + i642(lr0)) * i642(K) + i642(k))
    v = txl.alloc_local([passes * V], "uint32")

    def load(p, addr):
        if V == 4:
            txl.ptx[LDG_V4](v[4 * p], v[4 * p + 1], v[4 * p + 2], v[4 * p + 3], addr)
        else:
            txl.ptx[LDG_B32](v[p], addr)

    # CTA-uniform: the whole tile lies inside the shard (most tiles). Unpredicated loads let
    # ptxas keep all of them in flight.
    full = txl.And(
        txl.And(tc * BLOCK + BLOCK <= K, tr * BLOCK + sub * RB - row_offset >= 0), tr * BLOCK + sub * RB + RB - row_offset <= L
    )
    with txl.if_(full):
        with txl.then_():
            for p in range(passes):
                load(p, xaddr(xb + i642(K) * (p * rows_per_pass)))
        with txl.else_():
            for p in range(passes):
                lr = lr0 + p * rows_per_pass
                valid = txl.And(kvalid, txl.And(lr >= 0, lr < L))
                for i in range(V):
                    txl.ptx.mov.b32(v[p * V + i], txl.uint32(0))
                with txl.if_(valid), txl.then_():
                    load(p, xaddr(xb + i642(K) * (p * rows_per_pass)))
    if not cast:
        m = txl.local_scalar("uint32", init=txl.uint32(0))
        for i in range(passes * V):
            txl.ptx.max.u32(m, m, v[i] & txl.uint32(0x7FFFFFFF))
        am = cta_max_u32(m, red_s, warp, lane)
        with txl.if_(tid == 0), txl.then_():
            h = txl.local_scalar("uint16")
            txl.ptx.cvt.rn.bf16.f32(h, txl.reinterpret(am, ty="float32"))
            txl.ptx.st.global_.f32(amax_addr, txl.reinterpret(txl.Cast("uint32", h) << txl.uint32(16), ty="float32"))
        return
    a = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(a, amax_addr)
    _, rb = scale_bits(a)
    rcp = txl.reinterpret(rb, ty="float32")
    ob = txl.local_scalar("int64", init=(i642(gl) * i642(L) + i642(lr0)) * i642(K) + i642(k))
    for p in range(passes):
        lr = lr0 + p * rows_per_pass
        valid = txl.And(kvalid, txl.And(lr >= 0, lr < L))
        addr = oaddr(ob + i642(K) * (p * rows_per_pass))
        if V == 4:
            bw = []
            for i in range(2):
                w = txl.local_scalar("uint32")
                txl.ptx.cvt.rn.bf16x2.f32(
                    w, txl.reinterpret(v[4 * p + 2 * i + 1], ty="float32"), txl.reinterpret(v[4 * p + 2 * i], ty="float32")
                )
                bw.append(bf16_word_to_fp8x2(w, rcp))
            txl.ptx.st.global_.b32(addr, bw[0] | (bw[1] << txl.uint32(16)), pred=valid)
        else:
            h = txl.local_scalar("uint16")
            txl.ptx.cvt.rn.bf16.f32(h, txl.reinterpret(v[p], ty="float32"))
            q = e4m3x22(txl.float32(0.0), txl.reinterpret(txl.Cast("uint32", h) << txl.uint32(16), ty="float32"), rcp)
            txl.ptx.st.global_.b8(addr, txl.Cast("uint8", q), pred=valid)


CAST_SPLIT = 4
CAST_SPLIT_BELOW = 1 << 62  # always split (measured faster at every size)


def flat_addr(buf, nd, esize):
    """
    Element-offset addressing into a contiguous rank-nd buffer.
    """
    base = buf.ptr_to([0] * nd)
    return lambda off: txl.ptx.addr(base, off * esize)


@functools.cache
def shard_kernel(cast, V, amax_nd, split=1):
    """
    Shard tile amax (cast False) or cast (True). shard (g, L, K) and out contiguous; amax
    rank-3 contiguous (amax_nd 3) or a 1-D span with group stride asg (amax_nd 1).
    """

    def body(x, amax, out, L, K, row_offset, tr0, NTR, C, go, asg):
        tid = txl.thread_id()
        smem = txl.smem_pool()
        red_s = smem.alloc([NT // 32], "uint32")
        bid = txl.cta_id()
        gl, trl, tc = tile_coords(bid // split if split > 1 else bid, NTR, C)
        tr = tr0 + trl
        amax_addr = flat_addr(amax, amax_nd, 4)(i642(go + gl) * asg + i642(tr) * i642(C) + i642(tc))
        shard_tile_body(
            tid, txl.warp_id(), txl.lane_id(), red_s,
            flat_addr(x, 3, 4), None if out is None else flat_addr(out, 3, 1),
            amax_addr, L, K, row_offset, gl, tr, tc, cast, V,
            sub=bid % split if split > 1 else 0, RB=BLOCK // split,
        )

    if cast:

        @txl.kernel()
        def fp8_shard_cast(x: tensor_nd(txl.f32, 3), amax: tensor_nd(txl.f32, amax_nd), out: tensor_nd("float8_e4m3fn", 3), L: txl.int32, K: txl.int32, row_offset: txl.int32, tr0: txl.int32, NTR: txl.int32, C: txl.int32, go: txl.int32, asg: txl.int64, num_tiles: txl.int32):
            pdl_entry(num_tiles)
            pdl()
            body(x, amax, out, L, K, row_offset, tr0, NTR, C, go, asg)

        return compile_v(fp8_shard_cast)

    @txl.kernel()
    def fp8_shard_amax(x: tensor_nd(txl.f32, 3), amax: tensor_nd(txl.f32, amax_nd), L: txl.int32, K: txl.int32, row_offset: txl.int32, tr0: txl.int32, NTR: txl.int32, C: txl.int32, go: txl.int32, asg: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        body(x, amax, None, L, K, row_offset, tr0, NTR, C, go, asg)

    return compile_v(fp8_shard_amax)


def shard_geom(shard, amax, row_offset):
    gl, L, K = shard.shape
    tr0 = row_offset // BLOCK
    ntr = (row_offset + L + BLOCK - 1) // BLOCK - tr0
    return gl, L, K, tr0, ntr, amax.shape[2]


def shard_tiles(shard, amax, out, row_offset, group_offset):
    gl, L, K = shard.shape
    if gl == 0 or L == 0 or K == 0:
        return
    tr0 = row_offset // BLOCK
    ntr = (row_offset + L + BLOCK - 1) // BLOCK - tr0
    C = amax.shape[2]
    V = 4 if K % 4 == 0 and shard.data_ptr() % 16 == 0 and (out is None or out.data_ptr() % 4 == 0) else 1
    if amax.is_contiguous():
        a, nd = amax, 3
    else:
        a, nd = flat_view(amax), 1
    if out is None:
        shard_kernel(False, V, nd)(shard, a, L, K, row_offset, tr0, ntr, C, group_offset, amax.stride(0), gl * ntr * C)
    else:
        tiles = gl * ntr * C
        split = CAST_SPLIT if V == 4 and tiles < CAST_SPLIT_BELOW else 1
        shard_kernel(True, V, nd, split)(shard, a, out, L, K, row_offset, tr0, ntr, C, group_offset, amax.stride(0), tiles * split)


def fp8_shard_amax_sm100(shard, amax, row_offset, group_offset):
    shard_tiles(shard, amax, None, row_offset, group_offset)


def fp8_shard_cast_sm100(shard, amax, out, row_offset, group_offset):
    shard_tiles(shard, amax, out, row_offset, group_offset)


# ---------------------------------------------------------------- scales


def scale_elem(i, R, C, amax_addr, sc_addr, sct_addr):
    RC = R * C
    g = i // RC
    rem = i % RC
    r = rem // C
    c = rem % C
    a = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(a, amax_addr(i))
    sb, _ = scale_bits(a)
    sf = txl.reinterpret(sb, ty="float32")
    txl.ptx.st.global_.f32(sc_addr(i), sf)
    txl.ptx.st.global_.f32(sct_addr(g * RC + c * R + r), sf)


@functools.cache
def scales_kernel():
    @txl.kernel()
    def fp8_scales(amax: tensor_nd(txl.f32, 3), sc: tensor_nd(txl.f32, 3), sct: tensor_nd(txl.f32, 3), R: txl.int32, C: txl.int32, total: txl.int32, num_ctas: txl.int32):
        pdl_entry(num_ctas)
        pdl()
        i = txl.cta_id() * NT + txl.thread_id()
        with txl.if_(i < total), txl.then_():
            scale_elem(
                i, R, C,
                flat_addr(amax, 3, 4), flat_addr(sc, 3, 4), flat_addr(sct, 3, 4),
            )

    return compile_v(fp8_scales)


def fp8_scales_sm100(amax, scale, scale_t):
    G, R, C = amax.shape
    total = G * R * C
    if total == 0:
        return
    scales_kernel()(amax, scale, scale_t, R, C, total, (total + NT - 1) // NT)


# ---------------------------------------------------------------- pack transpose


@functools.cache
def pack_vec(dtype):
    """
    K % 8 == 0, N % 16 == 0.
    """

    @txl.kernel()
    def fp8_pack_vec(x: tensor_nd(dtype, 3), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, total: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        tid = txl.thread_id()
        g, tr, tc = tile_coords(txl.cta_id(), R, C)
        smem = txl.smem_pool()
        t_s = smem.alloc([BLOCK * BLOCK // 4], "uint32")
        kc = tid & 15
        rg = tid >> 4
        n0 = tr * BLOCK + rg * 8
        k = tc * BLOCK + kc * 8
        kvalid = k < K
        xb = txl.local_scalar("int64", init=(i642(g) * i642(N) + i642(n0)) * i642(K) + i642(k))
        xa = flat_addr(x, 3, 1)
        B = [(txl.local_scalar("uint32"), txl.local_scalar("uint32")) for _ in range(8)]
        with txl.if_(txl.And(tc * BLOCK + BLOCK <= K, tr * BLOCK + BLOCK <= N)):
            with txl.then_():
                for r in range(8):
                    txl.ptx.ld.global_.nc.v2.b32(B[r][0], B[r][1], xa(xb + i642(K) * r))
            with txl.else_():
                for r in range(8):
                    txl.ptx.mov.b32(B[r][0], txl.uint32(0))
                    txl.ptx.mov.b32(B[r][1], txl.uint32(0))
                    with txl.if_(txl.And(kvalid, n0 + r < N)), txl.then_():
                        txl.ptx.ld.global_.nc.v2.b32(B[r][0], B[r][1], xa(xb + i642(K) * r))
        for r in range(8):
            txl.ptx.st.global_.v2.b32(out.ptr_to([xb + i642(K) * r]), B[r][0], B[r][1], pred=txl.And(kvalid, n0 + r < N))
        transpose_store(B, t_s, kc, rg)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NT))
        tbase = txl.local_scalar("int64", init=total + i642(g) * i642(K) * i642(N))
        transpose_flush(t_s, tid, lambda off: out.ptr_to([tbase + off]), tc * BLOCK, tr * BLOCK, K, N)

    return compile_v(fp8_pack_vec)


@functools.cache
def pack_generic(dtype):
    @txl.kernel()
    def fp8_pack_generic(x: tensor_nd(dtype, 3), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, total: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        tid = txl.thread_id()
        g, tr, tc = tile_coords(txl.cta_id(), R, C)
        c = tid & 127
        k = tc * BLOCK + c
        r0 = tid >> 7
        for i in range(BLOCK // 2):
            n = tr * BLOCK + r0 + 2 * i
            with txl.if_(txl.And(k < K, n < N)), txl.then_():
                b = txl.local_scalar("uint8")
                off = (i642(g) * i642(N) + i642(n)) * i642(K) + i642(k)
                txl.ptx.ld.global_.nc.u8(b, flat_addr(x, 3, 1)(off))
                txl.ptx.st.global_.b8(out.ptr_to([off]), b)
                txl.ptx.st.global_.b8(out.ptr_to([total + (i642(g) * i642(K) + i642(k)) * i642(N) + i642(n)]), b)

    return compile_v(fp8_pack_generic)


ONE_BYTE = {torch.float8_e4m3fn: "float8_e4m3fn", torch.uint8: "uint8", torch.int8: "int8", torch.float8_e5m2: "float8_e5m2"}


def fp8_pack_transpose_sm100(x, out):
    G, N, K = x.shape
    tiles = G * ((N + BLOCK - 1) // BLOCK) * ((K + BLOCK - 1) // BLOCK)
    if tiles == 0:
        return
    if out.dtype != torch.uint8 or out.dim() != 1:
        out = out.view(torch.uint8).view(-1)
    dt = ONE_BYTE.get(x.dtype)
    if dt is None:
        x, dt = x.view(torch.uint8), "uint8"
    fast = K % 8 == 0 and N % 16 == 0 and x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0
    (pack_vec(dt) if fast else pack_generic(dt))(x, out, N, K, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK, G * N * K, tiles)


# ---------------------------------------------------------------- multi launchers

# Descriptor table fields (int64) of the shard-amax multi launch.
SA_FIELDS = 12
(SA_X, SA_AMAX, SA_L, SA_K, SA_RO, SA_GO, SA_TR0, SA_NTR, SA_C, SA_ASG, SA_START, SA_PAD) = range(SA_FIELDS)


NUM_SMS = 148


def load_table(table, nw, fields, cap):
    """
    Stage the int64 descriptor table in smem (once per CTA).
    """
    smem = txl.smem_pool()
    t_s = smem.alloc([cap * fields], "int64")
    red_s = smem.alloc([NT // 32], "uint32")
    tid = txl.thread_id()
    with txl.serial((cap * fields + NT - 1) // NT) as it:
        i = it * NT + tid
        with txl.if_(i < nw * fields), txl.then_():
            v = txl.local_scalar("int64")
            txl.ptx.ld.global_.nc.s64(v, table.ptr_to([i642(i)]))
            txl.ptx.st.shared.s64(t_s.ptr_to([i]), v)
    txl.ptx.bar.sync(txl.uint32(2), txl.uint32(NT))
    return t_s, red_s


def find_entry(t_s, nw, fields, start_field, key):
    """
    Index of the last table entry whose start <= key (entries sorted by start).
    """
    w = txl.local_scalar("int32", init=txl.int32(0))
    with txl.serial(nw) as i:
        s = txl.local_scalar("int64")
        txl.ptx.ld.shared.s64(s, t_s.ptr_to([i * fields + start_field]))
        with txl.if_(s <= key), txl.then_():
            txl.assign(w, i)
    return w


def field(t_s, w, fields, f, dtype="int32"):
    s = txl.local_scalar("int64")
    txl.ptx.ld.shared.s64(s, t_s.ptr_to([w * fields + f]))
    return txl.local_scalar(dtype, init=txl.Cast(dtype, s)) if dtype != "int64" else s


def handle(base, off, elem_bytes):
    return txl.reinterpret(txl.Cast("uint64", base) + txl.Cast("uint64", off * elem_bytes), ty="handle")


def cap2(n):
    c = 8
    while c < n:
        c *= 2
    return c


def gfield(table, w, f, dtype="int32"):
    s = txl.local_scalar("int64")
    txl.ptx.ld.global_.nc.s64(s, table.ptr_to([i642(w * SA_FIELDS + f)]))
    return txl.local_scalar(dtype, init=txl.Cast(dtype, s)) if dtype != "int64" else s


@functools.cache
def shard_amax_multi_kernel(V):
    """
    One CTA per tile over all shards; the CTA finds its shard descriptor in the (L1/L2-resident)
    table with broadcast loads. Beats a persistent tile loop by 3-8 us per precompute row: the
    hardware CTA scheduler keeps every SM busy to the end.
    """

    @txl.kernel()
    def fp8_shard_amax_multi(table: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), nw: txl.int32, total: txl.int32):
        pdl_entry(total)
        pdl()
        tid = txl.thread_id()
        smem = txl.smem_pool()
        red_s = smem.alloc([NT // 32], "uint32")
        t = txl.cta_id()
        w = txl.local_scalar("int32", init=txl.int32(0))
        with txl.serial(nw) as i:
            st = gfield(table, i, SA_START, "int64")
            with txl.if_(st <= i642(t)), txl.then_():
                txl.assign(w, i)
        xp = gfield(table, w, SA_X, "int64")
        ap = gfield(table, w, SA_AMAX, "int64")
        L = gfield(table, w, SA_L)
        K = gfield(table, w, SA_K)
        ro = gfield(table, w, SA_RO)
        go = gfield(table, w, SA_GO)
        tr0 = gfield(table, w, SA_TR0)
        ntr = gfield(table, w, SA_NTR)
        C = gfield(table, w, SA_C)
        asg = gfield(table, w, SA_ASG, "int64")
        start = gfield(table, w, SA_START)
        gl, trl, tc = tile_coords(t - start, ntr, C)
        tr = tr0 + trl
        amax_addr = handle(ap, i642(go + gl) * asg + i642(tr) * i642(C) + i642(tc), 4)
        shard_tile_body(
            tid, txl.warp_id(), txl.lane_id(), red_s,
            lambda off: handle(xp, off, 4), None,
            amax_addr, L, K, ro, gl, tr, tc, False, V,
        )

    return compile_v(fp8_shard_amax_multi)


def fp8_shard_amax_multi_sm100(shards, amaxes, row_offsets, group_offsets):
    rows, start, V = [], 0, 4
    for shard, amax, ro, go in zip(shards, amaxes, row_offsets, group_offsets):
        if shard.numel() == 0:
            continue
        gl, L, K, tr0, ntr, C = shard_geom(shard, amax, ro)
        if K % 4 or shard.data_ptr() % 16:
            V = 1
        rows.append([shard.data_ptr(), amax.data_ptr(), L, K, ro, go, tr0, ntr, C, amax.stride(0), start, 0])
        start += gl * ntr * C
    if start == 0:
        return lambda: None
    dev = shards[0].device
    table = torch.tensor(rows, dtype=torch.int64, device=dev).view(-1)
    kern, nw, tiles = shard_amax_multi_kernel(V), len(rows), start
    keep = (list(shards), list(amaxes))

    def launch():
        _ = keep
        kern(table, nw, tiles)

    return launch


SC_FIELDS = 6
(SC_AMAX, SC_SC, SC_SCT, SC_R, SC_C, SC_START) = range(SC_FIELDS)


@functools.cache
def scales_multi_kernel(cap):
    @txl.kernel()
    def fp8_scales_multi(table: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), nw: txl.int32, total: txl.int32, num_ctas: txl.int32):
        pdl_entry(num_ctas)
        pdl()
        t_s, _ = load_table(table, nw, SC_FIELDS, cap)
        gi = txl.local_scalar("int32", init=txl.cta_id() * NT + txl.thread_id())
        with txl.while_(gi < total):
            w = find_entry(t_s, nw, SC_FIELDS, SC_START, i642(gi))
            ap = field(t_s, w, SC_FIELDS, SC_AMAX, "int64")
            sp = field(t_s, w, SC_FIELDS, SC_SC, "int64")
            tp = field(t_s, w, SC_FIELDS, SC_SCT, "int64")
            R = field(t_s, w, SC_FIELDS, SC_R)
            C = field(t_s, w, SC_FIELDS, SC_C)
            start = field(t_s, w, SC_FIELDS, SC_START)
            i = txl.local_scalar("int32", init=gi - start)
            scale_elem(
                i, R, C,
                lambda j: handle(ap, i642(j), 4), lambda j: handle(sp, i642(j), 4), lambda j: handle(tp, i642(j), 4),
            )
            txl.assign(gi, gi + num_ctas * NT)

    return compile_v(fp8_scales_multi)


def fp8_scales_multi_sm100(amaxes, scales, scales_t):
    rows, start = [], 0
    for amax, scale, scale_t in zip(amaxes, scales, scales_t):
        G, R, C = amax.shape
        if G * R * C == 0:
            continue
        rows.append([amax.data_ptr(), scale.data_ptr(), scale_t.data_ptr(), R, C, start])
        start += G * R * C
    if start == 0:
        return lambda: None
    table = torch.tensor(rows, dtype=torch.int64, device=amaxes[0].device).view(-1)
    kern, nw, total = scales_multi_kernel(cap2(len(rows))), len(rows), start
    ctas = min((total + NT - 1) // NT, NUM_SMS)
    keep = (list(amaxes), list(scales), list(scales_t))

    def launch():
        _ = keep
        kern(table, nw, total, ctas)

    return launch
