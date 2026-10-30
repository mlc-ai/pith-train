"""
group_bias_add_sm100: in place y[r] += bias[group_indices[r]] on SM100, one TIRx-lite kernel.

Flat vectorized elementwise: the (M, H) problem is cut into W-word vectors (W = 8 -> 32 B when the
row length and base pointers allow, using sm_100's 256-bit ld/st), each thread handles U vectors
strided by the block, issuing all loads before any add/store. The sum is formed in fp32 and rounded
once (cvt.rn) to y's dtype. Offsets are int32 (launches are chunked so every chunk has < 2^31
elements). TAIL > 0 runs the last TAIL CTAs first: they cover the part of y most recently written
(and still L2-resident) by whatever produced it.

Evolved with TIRx-harness: run group_bias_add, frontier member flat-vec-l2tail (commit 0a55a8c, 601 us vs 672 us, 1.118x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools
import math

import torch
import tvm_ffi

import tirx_kernels.tirx_lite as txl

__all__ = ["group_bias_add_sm100"]

BLOCK = 64
UNROLL = 2
MAXW = 8  # widest vector in 32-bit words (8 -> 256-bit ld/st)
TAIL_BYTES = 32 << 20
YLD = "ld.global"
YST = "st.global"
BLD = "ld.global.nc"
ESIZE = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}
DNAME = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
INAME = {torch.int32: "int32", torch.int64: "int64"}
VSUF = {1: ".b32", 2: ".v2.b32", 4: ".v4.b32", 8: ".v8.b32"}


def add_words(dtype, out, yw, bw, n):
    """
    out[0:n] = round(f32(y) + f32(bias)) for n 32-bit words of packed y / bias elements.
    """
    if dtype == "float32":
        for i in range(n):
            f = txl.local_scalar("float32")
            txl.ptx["add.rn.f32"](f, txl.reinterpret("float32", yw[i]), txl.reinterpret("float32", bw[i]))
            txl.assign(out[i], txl.reinterpret("uint32", f))
        return
    for i in range(n):
        if dtype == "bfloat16":
            ylo = txl.reinterpret("float32", yw[i] << txl.uint32(16))
            yhi = txl.reinterpret("float32", yw[i] & txl.uint32(0xFFFF0000))
            blo = txl.reinterpret("float32", bw[i] << txl.uint32(16))
            bhi = txl.reinterpret("float32", bw[i] & txl.uint32(0xFFFF0000))
        else:
            yf = txl.alloc_local([2], "float32")
            bf = txl.alloc_local([2], "float32")
            txl.idioms.cast_f16x2_to_f32x2(yf, 0, yw[i])
            txl.idioms.cast_f16x2_to_f32x2(bf, 0, bw[i])
            ylo, yhi, blo, bhi = yf[0], yf[1], bf[0], bf[1]
        lo = txl.local_scalar("float32")
        hi = txl.local_scalar("float32")
        txl.ptx["add.rn.f32"](lo, ylo, blo)
        txl.ptx["add.rn.f32"](hi, yhi, bhi)
        cvt = "cvt.rn.bf16x2.f32" if dtype == "bfloat16" else "cvt.rn.f16x2.f32"
        txl.ptx[cvt](out[i], hi, lo)


class Slice:
    def __init__(self, buf, off):
        self.buf, self.off = buf, off

    def __getitem__(self, i):
        return self.buf[self.off + i]


@functools.cache
def kernel2(dtype: str, itype: str, H: int, W: int):
    """
    W words per vector (1, 2, 4 or 8); H * esize must be a multiple of 4 * W bytes.
    """
    esize = 4 if dtype == "float32" else 2
    row_words = H * esize // 4
    vpr = row_words // W  # vectors per row
    epv = W * 4 // esize  # elements per vector
    ld = YLD + VSUF[W]
    ldb = BLD + VSUF[W]
    st = YST + VSUF[W]
    ldi = "ld.global.nc.s64" if itype == "int64" else "ld.global.nc.s32"
    per_cta = BLOCK * UNROLL

    @txl.kernel(warps=BLOCK // 32, arch="sm_100a", grid=lambda p: txl.ceildiv(p["nvec"], BLOCK * UNROLL))
    def group_bias_add(y: txl.gptr[dtype], bias: txl.gptr[dtype], gidx: txl.gptr[itype], nvec: txl.i32, tail: txl.i32):
        tid = txl.thread_id()
        if TAIL_BYTES:
            nb = (nvec + per_cta - 1) // per_cta
            c0 = txl.cta_id()
            cta = txl.Select(c0 < tail, nb - tail + c0, c0 - tail)
        else:
            cta = txl.cta_id()
        base = cta * per_cta + tid
        yw = txl.alloc_local([UNROLL * W], "uint32")
        bw = txl.alloc_local([UNROLL * W], "uint32")
        g = txl.alloc_local([UNROLL], "int32")
        for u in range(UNROLL):
            v = base + u * BLOCK
            with txl.If(v < nvec), txl.Then():
                row = v // vpr
                gl = txl.local_scalar(itype)
                txl.ptx[ldi](gl, gidx.ptr_to([row]))
                txl.assign(g[u], txl.cast(gl, "int32"))
                txl.ptx[ld](*[yw[u * W + k] for k in range(W)], y.ptr_to([v * epv]))
        for u in range(UNROLL):
            v = base + u * BLOCK
            with txl.If(v < nvec), txl.Then():
                row = v // vpr
                col = v - row * vpr
                txl.ptx[ldb](*[bw[u * W + k] for k in range(W)], bias.ptr_to([g[u] * H + col * epv]))
        for u in range(UNROLL):
            v = base + u * BLOCK
            with txl.If(v < nvec), txl.Then():
                add_words(dtype, Slice(yw, u * W), Slice(yw, u * W), Slice(bw, u * W), W)
                txl.ptx[st](y.ptr_to([v * epv]), *[yw[u * W + k] for k in range(W)])

    return group_bias_add.compile()


@functools.cache
def kernel_h(dtype: str, itype: str, H: int):
    """
    16-bit fallback for half-precision rows whose byte length is not a multiple of 4.
    """
    vpr = H
    cvt = "cvt.rn.bf16.f32" if dtype == "bfloat16" else "cvt.rn.f16.f32"
    ldi = "ld.global.nc.s64" if itype == "int64" else "ld.global.nc.s32"

    @txl.kernel(warps=BLOCK // 32, arch="sm_100a", grid=lambda p: txl.ceildiv(p["nvec"], BLOCK))
    def group_bias_add_h(y: txl.gptr[dtype], bias: txl.gptr[dtype], gidx: txl.gptr[itype], nvec: txl.i32):
        v = txl.cta_id() * BLOCK + txl.thread_id()
        with txl.If(v < nvec), txl.Then():
            row = v // vpr
            col = v - row * vpr
            g = txl.local_scalar(itype)
            yh = txl.local_scalar("uint16")
            bh = txl.local_scalar("uint16")
            txl.ptx[ldi](g, gidx.ptr_to([row]))
            txl.ptx["ld.global.b16"](yh, y.ptr_to([v]))
            txl.ptx["ld.global.nc.b16"](bh, bias.ptr_to([txl.cast(g, "int32") * H + col]))
            yf = txl.local_scalar("float32")
            bf = txl.local_scalar("float32")
            if dtype == "bfloat16":
                txl.assign(yf, txl.reinterpret("float32", txl.cast(yh, "uint32") << txl.uint32(16)))
                txl.assign(bf, txl.reinterpret("float32", txl.cast(bh, "uint32") << txl.uint32(16)))
            else:
                txl.ptx["cvt.f32.f16"](yf, yh)
                txl.ptx["cvt.f32.f16"](bf, bh)
            s = txl.local_scalar("float32")
            txl.ptx["add.rn.f32"](s, yf, bf)
            o = txl.local_scalar("uint16")
            txl.ptx[cvt](o, s)
            txl.ptx["st.global.b16"](y.ptr_to([v]), o)

    return group_bias_add_h.compile()


MAX_ELEMS = 1 << 30  # per launch: element offsets (and G * H) stay well inside int32


def group_bias_add_sm100(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> None:
    """
    In place y[r] += bias[group_indices[r]], bit-identical to y.copy_(y + bias[group_indices]).
    """
    M, H = y.shape
    if M == 0 or H == 0:
        return
    # inputs the kernel does not cover (mixed dtypes, strided views, int32 offset overflow) take the
    # PyTorch fallback the result is bit-identical to, as the SM90 kernel does
    covered = y.dtype == bias.dtype and y.dtype in DNAME and group_indices.dtype in INAME
    if not covered or not (y.is_contiguous() and bias.is_contiguous() and group_indices.is_contiguous()) or bias.numel() >= MAX_ELEMS:
        y.copy_(y + bias[group_indices])
        return
    dtype, itype = DNAME[y.dtype], INAME[group_indices.dtype]
    esize = ESIZE[y.dtype]
    row_bytes = H * esize
    yv, bv = y.view(-1), bias.view(-1)
    rows_per = max(1, MAX_ELEMS // H)
    with tvm_ffi.use_torch_stream():
        if row_bytes % 4 == 0:
            align = math.gcd(4 * MAXW, row_bytes, y.data_ptr() % 32 or 32, bias.data_ptr() % 32 or 32)
            if align >= 4:
                W = align // 4
                fn = kernel2(dtype, itype, H, W)
                vpr = row_bytes // (4 * W)
                for r0 in range(0, M, rows_per):
                    r1 = min(M, r0 + rows_per)
                    nvec = (r1 - r0) * vpr
                    nb = -(-nvec // (BLOCK * UNROLL))
                    tail = min(nb, TAIL_BYTES // (BLOCK * UNROLL * W * 4))
                    fn(yv[r0 * H:r1 * H], bv, group_indices[r0:r1], nvec, tail)
                return
        fn = kernel_h(dtype, itype, H)
        for r0 in range(0, M, rows_per):
            r1 = min(M, r0 + rows_per)
            fn(yv[r0 * H:r1 * H], bv, group_indices[r0:r1], (r1 - r0) * H)
