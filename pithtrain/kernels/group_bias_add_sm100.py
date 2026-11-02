"""
group_bias_add_sm100: in place y[r] += bias[group_indices[r]] on SM100, TIRx-lite kernels.

Row tile (the common path): one CTA per 4 consecutive rows, each thread moving one W-word vector
(W = 8 -> 32 B, sm_100's 256-bit ld/st) of every row, up to two vectors per row. All four rows' y
loads are issued up front; when the four rows share a group one bias vector serves them all,
otherwise each row loads its own. Each row's add is retired directly into its own store. Half
precision adds use the native packed add.rn.bf16x2 / add.rn.f16x2 (both operands are representable
in y's dtype, so this is the same single round-to-nearest-even as an fp32 add plus cvt.rn); fp32
uses add.rn.f32. The launch is bounded to 5 CTAs per SM (64 registers). The CTAs covering the newest
48 MiB of y run first, walking backwards, reusing the lines its producer left L2-resident.

Flat fallback (M not a multiple of 4, rows wider than two vectors per thread, int32 offset limits):
a flat W-word vector index space, two vectors per thread, also tail-first; rows whose byte length is
not a multiple of 4 take a 16-bit scalar kernel. Offsets are int32 (launches are chunked so every
chunk has < 2^30 elements). Plans are cached per (dtype, index dtype, H, pointer alignment).

Evolved with TIRx-harness: run group_bias_add, frontier member rowtile-packed-direct-retire (commit
4a1608d3, 587.085 us vs 671.33 us, 1.1435x over the previous SM100 path in the task's benchmark).
Compiled once per process by tvm_tirx.tirx.compile(..., pipeline="tirx") for sm_100a with nvcc and
launched through tvm-ffi on torch's current stream.
"""

import functools
import math

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc as before; a cubin (sm_100a SASS only, as the old -gencode build), since -arch sm_100a would also emit compute_100 PTX, which rejects sm_100a-only instructions

__all__ = ["group_bias_add_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


BLOCK = 64
UNROLL = 2
MAXW = 8  # widest vector in 32-bit words (8 -> 256-bit ld/st)
TAIL_BYTES = 96 << 20
RT_ROWS = 4
RT_MAXCH = 2
RT_TAIL = 48 << 20
ESIZE = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}
DNAME = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
INAME = {torch.int32: txl.int32, torch.int64: "int64"}
VSUF = {1: ".b32", 2: ".v2.b32", 4: ".v4.b32", 8: ".v8.b32"}


def add_words(dtype, out, yw, bw, n):
    """
    out[0:n] = round(y + bias) for n 32-bit words of packed y / bias elements (one rounding).
    """
    if dtype == "float32":
        for i in range(n):
            f = txl.local_scalar("float32")
            txl.ptx["add.rn.f32"](f, txl.reinterpret(yw[i], ty="float32"), txl.reinterpret(bw[i], ty="float32"))
            txl.assign(out[i], txl.reinterpret(f, ty="uint32"))
        return
    op = "add.rn.bf16x2" if dtype == "bfloat16" else "add.rn.f16x2"
    for i in range(n):
        # both inputs are representable in the output format, so the native packed add performs the
        # same one round-to-nearest-even as an fp32 add followed by the destination conversion
        txl.ptx[op](out[i], yw[i], bw[i])


class Slice:
    """
    A view buf[off:] of a local array, indexable like the array.
    """

    def __init__(self, buf, off):
        self.buf, self.off = buf, off

    def __getitem__(self, i):
        return self.buf[self.off + i]


@functools.cache
def kernel_flat(dtype: str, itype: str, H: int, W: int):
    """
    Flat fallback: W words per vector (1, 2, 4 or 8); H * esize must be a multiple of 4 * W bytes.
    """
    esize = 4 if dtype == "float32" else 2
    row_words = H * esize // 4
    vpr = row_words // W
    epv = W * 4 // esize
    ld, ldb, st = "ld.global" + VSUF[W], "ld.global.nc" + VSUF[W], "st.global" + VSUF[W]
    ldi = "ld.global.nc.s64" if itype == "int64" else "ld.global.nc.s32"
    per_cta = BLOCK * UNROLL

    var_M, var_G = txl.Var("M", "int32"), txl.Var("G", "int32")

    @txl.kernel()
    def group_bias_add(y: txl.Tensor((var_M, H), dtype, data_alignment=1), bias: txl.Tensor((var_G, H), dtype, data_alignment=1), gidx: txl.Tensor((var_M,), itype, data_alignment=1), M: var_M, G: var_G, tail: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=txl.ceildiv(M * vpr, per_cta), block=(BLOCK // 32) * 32))
        nvec = M * vpr
        tid = txl.thread_id()
        nb = (nvec + per_cta - 1) // per_cta
        c0 = txl.cta_id()
        cta = txl.Select(c0 < tail, nb - 1 - c0, c0 - tail)
        base = cta * per_cta + tid
        yw = txl.alloc_local([UNROLL * W], "uint32")
        bw = txl.alloc_local([UNROLL * W], "uint32")
        g = txl.alloc_local([UNROLL], "int32")
        for u in range(UNROLL):
            v = base + u * BLOCK
            with txl.if_(v < nvec), txl.then_():
                row = v // vpr
                gl = txl.local_scalar(itype)
                txl.ptx[ldi](gl, gidx.ptr_to([row]))
                txl.assign(g[u], txl.cast(gl, "int32"))
                txl.ptx[ld](*[yw[u * W + k] for k in range(W)], y.ptr_to([row, (v - row * vpr) * epv]))
        for u in range(UNROLL):
            v = base + u * BLOCK
            with txl.if_(v < nvec), txl.then_():
                row = v // vpr
                col = v - row * vpr
                txl.ptx[ldb](*[bw[u * W + k] for k in range(W)], bias.ptr_to([g[u], col * epv]))
        for u in range(UNROLL):
            v = base + u * BLOCK
            with txl.if_(v < nvec), txl.then_():
                row = v // vpr
                add_words(dtype, Slice(yw, u * W), Slice(yw, u * W), Slice(bw, u * W), W)
                txl.ptx[st](y.ptr_to([row, (v - row * vpr) * epv]), *[yw[u * W + k] for k in range(W)])

    return entry_function(group_bias_add.compile(TARGET))


@functools.cache
def kernel_h(dtype: str, itype: str, H: int):
    """
    16-bit fallback for half-precision rows whose byte length is not a multiple of 4.
    """
    cvt = "cvt.rn.bf16.f32" if dtype == "bfloat16" else "cvt.rn.f16.f32"
    ldi = "ld.global.nc.s64" if itype == "int64" else "ld.global.nc.s32"

    var_M, var_G = txl.Var("M", "int32"), txl.Var("G", "int32")

    @txl.kernel()
    def group_bias_add_h(y: txl.Tensor((var_M, H), dtype, data_alignment=1), bias: txl.Tensor((var_G, H), dtype, data_alignment=1), gidx: txl.Tensor((var_M,), itype, data_alignment=1), M: var_M, G: var_G):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=txl.ceildiv(M * H, BLOCK), block=(BLOCK // 32) * 32))
        v = txl.cta_id() * BLOCK + txl.thread_id()
        with txl.if_(v < M * H), txl.then_():
            row = v // H
            col = v - row * H
            g = txl.local_scalar(itype)
            yh, bh = txl.local_scalar("uint16"), txl.local_scalar("uint16")
            txl.ptx[ldi](g, gidx.ptr_to([row]))
            txl.ptx["ld.global.b16"](yh, y.ptr_to([row, col]))
            txl.ptx["ld.global.nc.b16"](bh, bias.ptr_to([txl.cast(g, "int32"), col]))
            yf, bf = txl.local_scalar("float32"), txl.local_scalar("float32")
            if dtype == "bfloat16":
                txl.assign(yf, txl.reinterpret(txl.cast(yh, "uint32") << txl.uint32(16), ty="float32"))
                txl.assign(bf, txl.reinterpret(txl.cast(bh, "uint32") << txl.uint32(16), ty="float32"))
            else:
                txl.ptx["cvt.f32.f16"](yf, yh)
                txl.ptx["cvt.f32.f16"](bf, bh)
            s = txl.local_scalar("float32")
            out = txl.local_scalar("uint16")
            txl.ptx["add.rn.f32"](s, yf, bf)
            txl.ptx[cvt](out, s)
            txl.ptx["st.global.b16"](y.ptr_to([row, col]), out)

    return entry_function(group_bias_add_h.compile(TARGET))


def rt_shape(H, W, esize):
    """
    Row tile geometry: (vectors per row, threads per CTA, vectors per thread per row).
    """
    vpr = H * esize // (4 * W)
    block = min(512, -(-vpr // 32) * 32)
    return vpr, block, -(-vpr // block)


@functools.cache
def kernel_rt(dtype: str, itype: str, H: int, W: int):
    """
    Row tile kernel: one CTA per RT_ROWS rows; M must be a multiple of RT_ROWS.
    """
    esize = 4 if dtype == "float32" else 2
    vpr, block, nch = rt_shape(H, W, esize)
    epv = W * 4 // esize
    R = RT_ROWS
    ld, ldb, st = "ld.global" + VSUF[W], "ld.global.nc" + VSUF[W], "st.global" + VSUF[W]
    ldi = "ld.global.nc.s64" if itype == "int64" else "ld.global.nc.s32"

    var_M, var_G = txl.Var("M", "int32"), txl.Var("G", "int32")

    @txl.kernel()
    def group_bias_add_rt(y: txl.Tensor((var_M, H), dtype, data_alignment=1), bias: txl.Tensor((var_G, H), dtype, data_alignment=1), gidx: txl.Tensor((var_M,), itype, data_alignment=1), M: var_M, G: var_G, tail: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=M // R, block=(block // 32) * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=5))
        c0 = txl.cta_id()
        nt = M // R
        row0 = txl.Select(c0 < tail, nt - 1 - c0, c0 - tail) * R
        tid = txl.thread_id()
        g = txl.alloc_local([R], "int32")
        for r in range(R):
            gl = txl.local_scalar(itype)
            txl.ptx[ldi](gl, gidx.ptr_to([row0 + r]))
            txl.assign(g[r], txl.cast(gl, "int32"))
        yw = txl.alloc_local([nch * R * W], "uint32")
        bw = txl.alloc_local([R * W], "uint32")
        for ch in range(nch):
            c = tid + ch * block
            with txl.if_(c < vpr), txl.then_():
                for r in range(R):
                    o = ((ch * R) + r) * W
                    txl.ptx[ld](*[yw[o + k] for k in range(W)], y.ptr_to([row0 + r, c * epv]))
        same = g[0] == g[R - 1]
        for r in range(1, R - 1):
            same = same & (g[0] == g[r])
        for ch in range(nch):
            c = tid + ch * block
            with txl.if_(c < vpr), txl.then_():
                with txl.if_(same):
                    with txl.then_():
                        txl.ptx[ldb](*[bw[k] for k in range(W)], bias.ptr_to([g[0], c * epv]))
                        for r in range(R):
                            o = ((ch * R) + r) * W
                            add_words(dtype, Slice(yw, o), Slice(yw, o), bw, W)
                            txl.ptx[st](y.ptr_to([row0 + r, c * epv]), *[yw[o + k] for k in range(W)])
                    with txl.else_():
                        for r in range(R):
                            o = ((ch * R) + r) * W
                            txl.ptx[ldb](*[bw[k] for k in range(W)], bias.ptr_to([g[r], c * epv]))
                            add_words(dtype, Slice(yw, o), Slice(yw, o), bw, W)
                            txl.ptx[st](y.ptr_to([row0 + r, c * epv]), *[yw[o + k] for k in range(W)])

    return entry_function(group_bias_add_rt.compile(TARGET))


MAX_ELEMS = 1 << 30  # per launch: element offsets (and G * H) stay well inside int32


def general(y, bias, group_indices, M):
    """
    Flat fallback path for any covered input (bias.numel() < MAX_ELEMS, checked by the caller).
    """
    H = y.shape[1]
    dtype, itype = DNAME[y.dtype], INAME[group_indices.dtype]
    row_bytes = H * ESIZE[y.dtype]
    rows_per = max(1, MAX_ELEMS // H)
    if row_bytes % 4 == 0:
        align = math.gcd(4 * MAXW, row_bytes, y.data_ptr() % 32 or 32, bias.data_ptr() % 32 or 32)
        if align >= 4:
            W = align // 4
            fn = kernel_flat(dtype, itype, H, W)
            vpr = row_bytes // (4 * W)
            for r0 in range(0, M, rows_per):
                r1 = min(M, r0 + rows_per)
                nb = -(-((r1 - r0) * vpr) // (BLOCK * UNROLL))
                tail = min(nb, TAIL_BYTES // (BLOCK * UNROLL * W * 4))
                fn(y[r0:r1], bias, group_indices[r0:r1], r1 - r0, bias.shape[0], tail)
            return
    fn = kernel_h(dtype, itype, H)
    for r0 in range(0, M, rows_per):
        r1 = min(M, r0 + rows_per)
        fn(y[r0:r1], bias, group_indices[r0:r1], r1 - r0, bias.shape[0])


def make_plan(tdtype, titype, H, lowbits):
    """
    The launcher for one (dtype, index dtype, H, pointer alignment): the row tile where it applies.
    """
    esize = ESIZE[tdtype]
    row_bytes = H * esize
    if row_bytes % 4:
        return general
    W = math.gcd(4 * MAXW, row_bytes, lowbits or 32) // 4
    if W == 0 or rt_shape(H, W, esize)[2] > RT_MAXCH:
        return general
    fn = kernel_rt(DNAME[tdtype], INAME[titype], H, W)
    tail_tiles = max(1, RT_TAIL // (RT_ROWS * row_bytes))
    max_rows = (MAX_ELEMS - 1) // H

    def run(y, bias, group_indices, M):
        G = bias.shape[0]
        if M % RT_ROWS or M > max_rows or G > max_rows:
            return general(y, bias, group_indices, M)
        fn(y, bias, group_indices, M, G, min(M // RT_ROWS, tail_tiles))

    return run


PLANS = {}


def group_bias_add_sm100(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> None:
    """
    In place y[r] += bias[group_indices[r]], bit-identical to y.copy_(y + bias[group_indices]).
    """
    M, H = y.shape
    if M == 0 or H == 0:
        return
    # inputs the kernels do not cover (mixed dtypes, strided views, int32 offset overflow) take the
    # PyTorch fallback the result is bit-identical to, as the SM90 kernel does
    covered = y.dtype == bias.dtype and y.dtype in DNAME and group_indices.dtype in INAME
    if not covered or not (y.is_contiguous() and bias.is_contiguous() and group_indices.is_contiguous()) or bias.numel() >= MAX_ELEMS:
        y.copy_(y + bias[group_indices])
        return
    key = (y.dtype, group_indices.dtype, H, (y.data_ptr() | bias.data_ptr()) & 31)
    plan = PLANS.get(key)
    if plan is None:
        plan = PLANS[key] = make_plan(*key)
    plan(y, bias, group_indices, M)
