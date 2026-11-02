"""
fp8_quant_grouped_sm100: rowwise 1x128 + per-channel 128x1 FP8 quantization from one read.

One CTA per 128x128 tile, the tile held in registers (256 threads, 8 rows x 8 cols each).
Scales are exact powers of two, so x * (1/scale) is exact in bf16 and the bf16x2 -> e4m3x2
satfinite cvt rounds once, matching the fp32 reference bit for bit.

Evolved with TIRx-harness: run grouped, frontier member regtile-v8-l2pf (commit 158986c, 1.758x over the
previous SM100 path in the task's benchmark). Compiled once per process by
tvm_tirx.tirx.compile(..., pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools
import math

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc as before; a cubin (sm_100a SASS only, as the old -gencode build), since -arch sm_100a would also emit compute_100 PTX, which rejects sm_100a-only instructions

__all__ = ["fp8_quant_grouped_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


BLOCK = 128
THREADS = 256
ROWS_PER_THREAD = 8  # 128 rows / (8 warps * 2 half-warps)


def u32(v):
    return txl.uint32(v)


def max_abs(a, b):
    out = txl.local_scalar("uint32")
    txl.ptx.max.xorsign.abs.bf16x2(out, a, b)
    return out


def scale_bits2(packed_amax, lane_hi):
    """
    f32 scale bits of the bf16 amax in one half of a bf16x2 word (sign garbage).
    """
    if lane_hi:
        f = txl.reinterpret(txl.bitwise_and(packed_amax, u32(0x7FFF0000)), ty="float32")
    else:
        f = txl.reinterpret(txl.shift_left(txl.bitwise_and(packed_amax, u32(0x7FFF)), u32(16)), ty="float32")
    clamped = txl.local_scalar("float32")
    txl.ptx.max.f32(clamped, f, txl.float32(1e-4))  # max_num: an all-NaN block clamps to 1e-4
    ratio = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(ratio, clamped, txl.float32(1.0 / 448.0))
    bits = txl.reinterpret(ratio, ty="uint32")
    return txl.local_scalar(
        "uint32", init=txl.bitwise_and(bits + u32(0x7FFFFF), u32(0x7F800000))
    )


def rcp_bf16(scale_bits):
    """
    bf16 bits (low 16) of 1/scale; inf scale -> -inf as the SM90 emulation.
    """
    return txl.shift_right(u32(0x7F000000) - scale_bits, u32(16))


def quant_word(word, rcp2):
    prod = txl.local_scalar("uint32")
    txl.ptx.mul.rn.bf16x2(prod, word, rcp2)
    q = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(q, prod)
    return txl.cast(q, "uint32")


@functools.cache
def build(K: int):
    return entry_function(make(K).compile(TARGET))


def make(K: int):
    nkb = (K + BLOCK - 1) // BLOCK
    vec = math.gcd(K, 8)  # elements per global access
    ragged = K % BLOCK != 0

    @txl.kernel()
    def quant(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), q_row: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), s_row: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), q_col: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), s_col: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), num_rb: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_rb * nkb, block=(THREADS // 32) * 32))
        bid = txl.cta_id()
        tid = txl.thread_id()
        rb = bid // nkb
        kb = bid % nkb
        warp = tid >> 5
        lane = tid & 31
        c = lane & 15
        h = lane >> 4
        col0 = txl.local_scalar("int32", init=kb * BLOCK + c * 8)
        row0 = txl.local_scalar("int32", init=rb * BLOCK + warp * 16 + h)

        smem = txl.smem_pool()
        colmax = smem.alloc([8 * 64], "uint32")  # [warp][c*4 + j]

        d = txl.alloc_local([ROWS_PER_THREAD * 4], "uint32")
        for k in range(ROWS_PER_THREAD * 4):
            txl.assign(d[k], u32(0))

        def gofs(i, sub):
            return txl.Cast("int64", (row0 + 2 * i) * K + col0 + sub * vec)

        # ---- load -------------------------------------------------------------------
        for i in range(ROWS_PER_THREAD):
            if vec == 8:
                def ld():
                    txl.ptx.ld.global_.v4.b32(
                        d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3], x.ptr_to([gofs(i, 0)])
                    )
                if ragged:
                    with txl.if_(col0 < K), txl.then_():
                        ld()
                else:
                    ld()
            else:
                for sub in range(8 // vec):
                    with txl.if_(col0 + sub * vec < K), txl.then_():
                        if vec == 4:
                            w = 4 * i + 2 * sub
                            txl.ptx.ld.global_.v2.b32(d[w], d[w + 1], x.ptr_to([gofs(i, sub)]))
                        elif vec == 2:
                            txl.ptx.ld.global_.b32(d[4 * i + sub], x.ptr_to([gofs(i, sub)]))
                        else:
                            hv = txl.local_scalar("uint16")
                            txl.ptx.ld.global_.b16(hv, x.ptr_to([gofs(i, sub)]))
                            w = 4 * i + sub // 2
                            if sub % 2 == 0:
                                txl.assign(d[w], txl.cast(hv, "uint32"))
                            else:
                                txl.assign(
                                    d[w],
                                    txl.bitwise_or(
                                        d[w], txl.shift_left(txl.cast(hv, "uint32"), u32(16))
                                    ),
                                )

        # ---- column amax: own rows, half-warp partner, then the 8 warps -------------
        cm = txl.alloc_local([4], "uint32")
        for j in range(4):
            txl.assign(cm[j], d[j])
            for i in range(1, ROWS_PER_THREAD):
                txl.assign(cm[j], max_abs(cm[j], d[4 * i + j]))
            peer = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.bfly.b32(peer, cm[j], u32(16), u32(31), u32(0xFFFFFFFF))
            txl.assign(cm[j], max_abs(cm[j], peer))
        with txl.if_(h == 0), txl.then_():
            txl.ptx["st.shared.v4.b32"](
                colmax.ptr_to([warp * 64 + c * 4]), cm[0], cm[1], cm[2], cm[3]
            )

        # ---- row amax (overlaps the barrier wait) -------------------------------------
        rrcp = txl.alloc_local([ROWS_PER_THREAD], "uint32")
        for i in range(ROWS_PER_THREAD):
            rm = max_abs(max_abs(d[4 * i], d[4 * i + 1]), max_abs(d[4 * i + 2], d[4 * i + 3]))
            for delta in (1, 2, 4, 8):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(peer, rm, u32(delta), u32(31), u32(0xFFFFFFFF))
                txl.assign(rm, max_abs(rm, peer))
            # fold the two bf16 halves into the low half
            rm2 = max_abs(rm, txl.shift_right(rm, u32(16)))
            sb = scale_bits2(rm2, False)
            r16 = rcp_bf16(sb)
            txl.assign(rrcp[i], txl.bitwise_or(r16, txl.shift_left(r16, u32(16))))
            with txl.if_(c == 0), txl.then_():
                txl.ptx.st.global_.b32(
                    s_row.ptr_to([txl.Cast("int64", (row0 + 2 * i) * nkb + kb)]), sb
                )

        txl.ptx.bar.sync(u32(0), u32(THREADS))

        for j in range(4):
            txl.assign(cm[j], u32(0))
        for wv in range(8):
            t = txl.alloc_local([4], "uint32")
            txl.ptx["ld.shared.v4.b32"](t[0], t[1], t[2], t[3], colmax.ptr_to([wv * 64 + c * 4]))
            for j in range(4):
                txl.assign(cm[j], max_abs(cm[j], t[j]))
        crcp = txl.alloc_local([4], "uint32")
        csb = txl.alloc_local([8], "uint32")
        for j in range(4):
            lo = scale_bits2(cm[j], False)
            hi = scale_bits2(cm[j], True)
            txl.assign(csb[2 * j], lo)
            txl.assign(csb[2 * j + 1], hi)
            txl.assign(
                crcp[j],
                txl.bitwise_or(rcp_bf16(lo), txl.shift_left(rcp_bf16(hi), u32(16))),
            )
        with txl.if_(tid < 16), txl.then_():
            sbase = rb * K + col0
            for e in range(8):
                if ragged or K % 8 != 0:
                    with txl.if_(col0 + e < K), txl.then_():
                        txl.ptx.st.global_.b32(s_col.ptr_to([txl.Cast("int64", sbase + e)]), csb[e])
                else:
                    txl.ptx.st.global_.b32(s_col.ptr_to([txl.Cast("int64", sbase + e)]), csb[e])

        # ---- quantize + store -------------------------------------------------------
        for which, dst in ((0, q_row), (1, q_col)):
            for i in range(ROWS_PER_THREAD):
                b = [
                    quant_word(d[4 * i + j], rrcp[i] if which == 0 else crcp[j]) for j in range(4)
                ]
                lo = txl.local_scalar("uint32", init=txl.bitwise_or(b[0], txl.shift_left(b[1], u32(16))))
                hi = txl.local_scalar("uint32", init=txl.bitwise_or(b[2], txl.shift_left(b[3], u32(16))))
                if vec == 8:
                    def st():
                        txl.ptx.st.global_.v2.b32(dst.ptr_to([gofs(i, 0)]), lo, hi)
                    if ragged:
                        with txl.if_(col0 < K), txl.then_():
                            st()
                    else:
                        st()
                else:
                    for e in range(8):
                        word = lo if e < 4 else hi
                        byte = txl.cast(
                            txl.bitwise_and(txl.shift_right(word, u32(8 * (e % 4))), u32(0xFF)),
                            "uint8",
                        )
                        with txl.if_(col0 + e < K), txl.then_():
                            txl.ptx.st.global_.b8(
                                dst.ptr_to([txl.Cast("int64", (row0 + 2 * i) * K + col0 + e)]), byte
                            )

    return quant



# ---------------------------------------------------------------------------------------------
# 256-bit variant (K % 16 == 0): a thread holds 4 rows x 16 cols (one ld.global.v8 per row).
# Column scales are reduced by 64 threads (one bf16x2 word each) and broadcast through smem.
# ---------------------------------------------------------------------------------------------
# L2 prefetch distance in tiles (~2 resident CTAs per SM ahead; measured optimum 222-444)
PF_DIST = 148 * 2
W8 = 4  # 128 threads; each holds 8 rows x 16 cols (more loads in flight per SM)
R8 = 128 // W8 // 4
T8 = W8 * 32


@functools.cache
def build_v8(K: int):
    return entry_function(make_v8(K).compile(TARGET))


def make_v8(K: int):
    nkb = (K + BLOCK - 1) // BLOCK
    ragged = K % BLOCK != 0

    @txl.kernel()
    def quant(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), q_row: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), s_row: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), q_col: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), s_col: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), num_rb: txl.int32, pf_dist: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_rb * nkb, block=(W8) * 32))
        bid = txl.cta_id()
        tid = txl.thread_id()
        rb = bid // nkb
        kb = bid % nkb
        warp = tid >> 5
        lane = tid & 31
        c = lane & 7
        h = lane >> 3
        col0 = txl.local_scalar("int32", init=kb * BLOCK + c * 16)
        row0 = txl.local_scalar("int32", init=rb * BLOCK + warp * (4 * R8) + h)

        smem = txl.smem_pool()
        part = smem.alloc([W8 * 64], "uint32")  # [warp][word]
        rcps = smem.alloc([64], "uint32")

        d = txl.alloc_local([R8 * 8], "uint32")

        def gofs(i):
            return txl.Cast("int64", (row0 + 4 * i) * K + col0)

        def guarded(body):
            if ragged:
                with txl.if_(col0 < K), txl.then_():
                    body()
            else:
                body()

        if ragged:
            for k in range(R8 * 8):
                txl.assign(d[k], u32(0))
        for i in range(R8):
            guarded(lambda i=i: txl.ptx.ld.global_.v8.b32(*[d[8 * i + j] for j in range(8)], x.ptr_to([gofs(i)])))

        # column partials: own 4 rows, then the 4 row-lanes (xor 8, 16), then per warp to smem
        # L2 prefetch of the tile a CTA about one resident wave later will load: one row per thread
        t2 = bid + pf_dist
        with txl.if_((pf_dist > 0) & (t2 < num_rb * nkb)), txl.then_():
            rb2 = t2 // nkb
            kb2 = t2 % nkb
            if ragged:
                nbytes = txl.Cast("uint32", txl.min(K - kb2 * BLOCK, BLOCK) * 2)
            else:
                nbytes = u32(BLOCK * 2)
            txl.ptx["cp.async.bulk.prefetch.L2.global"](
                x.ptr_to([txl.Cast("int64", (rb2 * BLOCK + tid) * K + kb2 * BLOCK)]), nbytes
            )

        cm = txl.alloc_local([8], "uint32")
        for j in range(8):
            txl.assign(cm[j], d[j])
            for i in range(1, R8):
                txl.assign(cm[j], max_abs(cm[j], d[8 * i + j]))
            for delta in (8, 16):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(peer, cm[j], u32(delta), u32(31), u32(0xFFFFFFFF))
                txl.assign(cm[j], max_abs(cm[j], peer))
        with txl.if_(h == 0), txl.then_():
            txl.ptx["st.shared.v4.b32"](part.ptr_to([warp * 64 + c * 8]), cm[0], cm[1], cm[2], cm[3])
            txl.ptx["st.shared.v4.b32"](part.ptr_to([warp * 64 + c * 8 + 4]), cm[4], cm[5], cm[6], cm[7])

        # row amax
        rrcp = txl.alloc_local([R8], "uint32")
        for i in range(R8):
            a = [max_abs(d[8 * i + 2 * j], d[8 * i + 2 * j + 1]) for j in range(4)]
            rm = max_abs(max_abs(a[0], a[1]), max_abs(a[2], a[3]))
            for delta in (1, 2, 4):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(peer, rm, u32(delta), u32(31), u32(0xFFFFFFFF))
                txl.assign(rm, max_abs(rm, peer))
            rm2 = max_abs(rm, txl.shift_right(rm, u32(16)))
            sb = scale_bits2(rm2, False)
            r16 = rcp_bf16(sb)
            txl.assign(rrcp[i], txl.bitwise_or(r16, txl.shift_left(r16, u32(16))))
            with txl.if_(c == 0), txl.then_():
                txl.ptx.st.global_.b32(s_row.ptr_to([txl.Cast("int64", (row0 + 4 * i) * nkb + kb)]), sb)

        txl.ptx.bar.sync(u32(0), u32(T8))
        with txl.if_(tid < 64), txl.then_():
            w = txl.local_scalar("uint32", init=u32(0))
            for wv in range(W8):
                t = txl.local_scalar("uint32")
                txl.ptx["ld.shared.b32"](t, part.ptr_to([wv * 64 + tid]))
                txl.assign(w, max_abs(w, t))
            lo = scale_bits2(w, False)
            hi = scale_bits2(w, True)
            txl.ptx["st.shared.b32"](
                rcps.ptr_to([tid]), txl.bitwise_or(rcp_bf16(lo), txl.shift_left(rcp_bf16(hi), u32(16)))
            )
            ccol = kb * BLOCK + tid * 2
            def sst():
                txl.ptx.st.global_.v2.b32(s_col.ptr_to([txl.Cast("int64", rb * K + ccol)]), lo, hi)
            if ragged:
                with txl.if_(ccol < K), txl.then_():
                    sst()
            else:
                sst()
        txl.ptx.bar.sync(u32(0), u32(T8))
        crcp = txl.alloc_local([8], "uint32")
        txl.ptx["ld.shared.v4.b32"](crcp[0], crcp[1], crcp[2], crcp[3], rcps.ptr_to([c * 8]))
        txl.ptx["ld.shared.v4.b32"](crcp[4], crcp[5], crcp[6], crcp[7], rcps.ptr_to([c * 8 + 4]))

        for which, dst in ((0, q_row), (1, q_col)):
            for i in range(R8):
                b = [quant_word(d[8 * i + j], rrcp[i] if which == 0 else crcp[j]) for j in range(8)]
                o = [
                    txl.local_scalar("uint32", init=txl.bitwise_or(b[2 * k], txl.shift_left(b[2 * k + 1], u32(16))))
                    for k in range(4)
                ]
                guarded(lambda i=i, o=o, dst=dst: txl.ptx.st.global_.v4.b32(dst.ptr_to([gofs(i)]), o[0], o[1], o[2], o[3]))

    return quant


def fp8_quant_grouped_sm100(input, grouped_mm_offs):
    M, K = input.shape
    nkb = (K + BLOCK - 1) // BLOCK
    dev = input.device
    x_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    x_scale = torch.empty((M, nkb), dtype=torch.float32, device=dev)
    x_ch_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    x_ch_scale = torch.empty((M // BLOCK, K), dtype=torch.float32, device=dev)
    if M == 0 or K == 0:
        return x_fp8, x_scale, x_ch_fp8, x_ch_scale
    fn = build_v8(K) if K % 16 == 0 else build(K)
    extra = (PF_DIST,) if K % 16 == 0 else ()
    fn(
        input.view(-1),
        x_fp8.view(torch.uint8).view(-1),
        x_scale.view(-1),
        x_ch_fp8.view(torch.uint8).view(-1),
        x_ch_scale.view(-1),
        M // BLOCK,
        *extra,
    )
    return x_fp8, x_scale, x_ch_fp8, x_ch_scale
