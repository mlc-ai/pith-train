"""
silu_mul_sm100: the SwiGLU product silu(gate) * up and its two gradients on SM100, TIRx-lite
kernels bit-exact with the SM90 Triton kernels.

Flat elementwise kernels: each thread moves VPT 16-byte vectors per tensor. Exact-tile BF16
launches use Blackwell packed-f32x2 add/mul/FMA for two independent FP32 lanes at once, while
ex2/rcp remain scalar. The fp32 math reproduces the Triton PTX bit for bit: sigmoid(x) = div.full(1,
ex2.approx(x * -log2e) + 1), computed here with ex2.approx.ftz and rcp.approx.ftz(d / 4) / 4, which
is identical on all 2^32 fp32 inputs and drops div.full's range check; every product is a separate
mul.f32 as in Triton (no product feeds an add, so nothing can contract), and the backward's (1 + g -
silu) is the single fma.rn Triton emits. bf16, fp16 and fp32 run natively; any other float dtype
(or a backward with mixed dtypes) is widened to fp32, run through the fp32 kernels and rounded once
back, the same bits as the Triton kernels (which also compute in fp32 and round once). Bases not
16-byte aligned take a scalar kernel.

Evolved with TIRx-harness: run silu_mul, frontier member packed-f32x2-stream (commit 6ce8ceb4,
487.818 us vs 493.082 us, 1.011x over the previous SM100 path, the Triton kernels, in the task's
benchmark). Compiled once per process by tvm_tirx.tirx.compile(..., pipeline="tirx") and
launched through tvm-ffi on torch's current stream.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc, as TVM_CUDA_COMPILE_MODE=nvcc chose before

__all__ = ["silu_mul_bwd_sm100", "silu_mul_fwd_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


DT = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
NEG_LOG2E = -1.4426950408889634  # rounds to 0xBFB8AA3B, Triton's constant
VBYTES = 16  # bytes per vector load / store
POLICY = {"first": 0x12F0000000000000, "normal": 0x1000000000000000, "last": 0x14F0000000000000}


def f32c(v):
    return txl.float32(v)


def sigmoid(x):
    t, e, d, s = txl.local_scalar("float32"), txl.local_scalar("float32"), txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.mul.f32(t, x, f32c(NEG_LOG2E))
    # ex2 results below 2^-126 only occur where 1 + e rounds to 1, so .ftz is bit-identical.
    txl.ptx.ex2.approx.ftz.f32(e, t)
    txl.ptx.add.f32(d, e, f32c(1.0))
    # div.full(1, d) for d >= 1 is rcp(d) when d <= 2^126 and rcp(d / 4) / 4 above; MUFU.RCP is
    # exponent-invariant, so the scaled form is bit-identical everywhere (checked over all 2^32 x).
    q, r = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.mul.f32(q, d, f32c(0.25))
    txl.ptx.rcp.approx.ftz.f32(r, q)
    txl.ptx.mul.f32(s, r, f32c(0.25))
    return s


def mul(a, b):
    r = txl.local_scalar("float32")
    txl.ptx.mul.f32(r, a, b)
    return r


def fwd_math(g, u):
    return [mul(mul(sigmoid(g), g), u)]


def bwd_math(do, g, u):
    s = sigmoid(g)
    silu = mul(s, g)
    grad_up = mul(do, silu)
    a = mul(mul(do, u), s)
    gp1 = txl.local_scalar("float32")
    txl.ptx.add.f32(gp1, g, f32c(1.0))
    ns = txl.local_scalar("float32")
    txl.ptx.neg.f32(ns, s)
    h = txl.local_scalar("float32")
    txl.ptx.fma.rn.f32(h, ns, g, gp1)
    return [mul(a, h), grad_up]


def pair(a, b):
    return txl.cuda.make_float2(a, b)


def unpair(value):
    lo, hi = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.mov.b64(lo, hi, value)
    return lo, hi


def pair_mul(a, b):
    out = txl.local_scalar("uint64")
    txl.ptx.mul.f32x2(out, a, b)
    return out


def pair_add(a, b):
    out = txl.local_scalar("uint64")
    txl.ptx.add.f32x2(out, a, b)
    return out


def pair_fma(a, b, c):
    out = txl.local_scalar("uint64")
    txl.ptx.fma.rn.f32x2(out, a, b, c)
    return out


def sigmoid_pair(g0, g1):
    g = pair(g0, g1)
    log2e = pair(f32c(NEG_LOG2E), f32c(NEG_LOG2E))
    t0, t1 = unpair(pair_mul(g, log2e))
    e0, e1 = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(e0, t0)
    txl.ptx.ex2.approx.ftz.f32(e1, t1)
    ones = pair(f32c(1.0), f32c(1.0))
    d = pair_add(pair(e0, e1), ones)
    quarter = pair(f32c(0.25), f32c(0.25))
    q0, q1 = unpair(pair_mul(d, quarter))
    r0, r1 = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.rcp.approx.ftz.f32(r0, q0)
    txl.ptx.rcp.approx.ftz.f32(r1, q1)
    return pair_mul(pair(r0, r1), quarter)


def fwd_math_packed(inputs):
    gate, up = inputs
    out = []
    for i in range(0, len(gate), 2):
        g = pair(gate[i], gate[i + 1])
        s = sigmoid_pair(gate[i], gate[i + 1])
        out.extend(unpair(pair_mul(pair_mul(s, g), pair(up[i], up[i + 1]))))
    return [out]


def bwd_math_packed(inputs):
    grad_output, gate, up = inputs
    grad_gate, grad_up = [], []
    for i in range(0, len(gate), 2):
        do, g, u = pair(grad_output[i], grad_output[i + 1]), pair(gate[i], gate[i + 1]), pair(up[i], up[i + 1])
        s = sigmoid_pair(gate[i], gate[i + 1])
        silu = pair_mul(s, g)
        grad_up.extend(unpair(pair_mul(do, silu)))
        a = pair_mul(pair_mul(do, u), s)
        gp1 = pair_add(g, pair(f32c(1.0), f32c(1.0)))
        sign_bits = txl.local_scalar("uint64")
        txl.ptx.mov.b64(sign_bits, txl.uint32(0x80000000), txl.uint32(0x80000000))
        neg_s = txl.bitwise_xor(s, sign_bits)
        h = pair_fma(neg_s, g, gp1)
        grad_gate.extend(unpair(pair_mul(a, h)))
    return [grad_gate, grad_up]


def to_f32(bits16, dtype):
    return txl.cast(txl.reinterpret(bits16, ty=dtype), "float32")


def unpack(word, dtype):
    """
    One b32 word -> list of fp32 values (2 for 16-bit dtypes, 1 for fp32).
    """
    if dtype == "float32":
        return [txl.reinterpret(word, ty="float32")]
    if dtype == "bfloat16":
        # BF16 widening is exact bit placement: the source bits become the high
        # half of FP32 and the low mantissa bits are zero. Avoid two CVTs/word.
        lo, hi = txl.shift_left(word, txl.uint32(16)), txl.bitwise_and(word, txl.uint32(0xFFFF0000))
        return [txl.reinterpret(lo, ty="float32"), txl.reinterpret(hi, ty="float32")]
    lo, hi = to_f32(txl.cast(txl.bitwise_and(word, txl.uint32(0xFFFF)), "uint16"), dtype), to_f32(txl.cast(txl.shift_right(word, txl.uint32(16)), "uint16"), dtype)
    return [lo, hi]


def pack(dst, vals, dtype):
    """
    fp32 values -> b32 words in dst (one RNE rounding).
    """
    if dtype == "float32":
        for i, v in enumerate(vals):
            txl.ptx.mov.b32(dst[i], txl.reinterpret(v, ty="uint32"))
        return
    for p in range(len(vals) // 2):
        if dtype == "float16":
            txl.ptx.cvt.rn.f16x2.f32(dst[p], vals[2 * p + 1], vals[2 * p])
        else:
            txl.ptx.cvt.rn.bf16x2.f32(dst[p], vals[2 * p + 1], vals[2 * p])


def compile_kernel(kernel):
    return entry_function(kernel.compile(TARGET))


SMALL = 1 << 22  # elements; at or below, the whole problem is one latency-bound wave


def pick_kernel(dtype: str, n_in: int, n: int):
    """
    Shape-aware choice over the one kernel family (measured on B200, cold L2).

    Inputs stream with L2 evict_first (read once). Outputs use the default policy: evict_last
    stores survive a cold-L2 flush and pin lines that later kernels (and benchmark iterations)
    then hit, which games cold-L2 timing and pollutes L2 in training.
    """
    if n_in == 2:  # forward: 3 streams; small = more CTAs, large = 32 B in flight per stream
        warps, vpt = (4, 1) if n <= SMALL else (8, 2)
    else:  # backward: 5 streams; one 16 B vector per stream keeps registers / occupancy up
        warps, vpt = 8, 1
    vec = 4 if dtype == "float32" else 8
    full_tiles = n % (warps * 32 * vpt * vec) == 0
    return kernel_cfg(dtype, n_in, warps, vpt, 16, "", "", "first", "", full_tiles)


@functools.cache
def scalar_kernel(dtype: str, n_in: int):
    """
    Unaligned-safe contract fallback; timed allocations take the vector path.
    """
    n_out = 1 if n_in == 2 else 2
    math = fwd_math if n_in == 2 else bwd_math
    threads = 256
    store16 = {"float32": None, "float16": "f16", "bfloat16": "bf16"}[dtype]

    def body(ptrs, n):
        ins, outs = ptrs[:n_in], ptrs[n_in:]
        idx = txl.cast(txl.cta_id(), "int64") * threads + txl.cast(txl.thread_id(), "int64")
        with txl.if_(idx < n):
            with txl.then_():
                vals = []
                for src in ins:
                    if dtype == "float32":
                        w = txl.local_scalar("uint32")
                        txl.ptx.ld.global_.nc.b32(w, txl.address_of(src[idx]))
                        vals.append(txl.reinterpret(w, ty="float32"))
                    else:
                        h = txl.local_scalar("uint16")
                        txl.ptx.ld.global_.nc.b16(h, txl.address_of(src[idx]))
                        if dtype == "bfloat16":
                            vals.append(txl.reinterpret(txl.shift_left(txl.cast(h, "uint32"), txl.uint32(16)), ty="float32"))
                        else:
                            vals.append(to_f32(h, dtype))
                res = math(*vals)
                for o in range(n_out):
                    if dtype == "float32":
                        txl.ptx.st.global_.b32(txl.address_of(outs[o][idx]), txl.reinterpret(res[o], ty="uint32"))
                    else:
                        h = txl.local_scalar("uint16")
                        getattr(txl.ptx.cvt.rn, store16).f32(h, res[o])
                        txl.ptx.st.global_.b16(txl.address_of(outs[o][idx]), h)

    grid = lambda n: txl.cast((n + threads - 1) // threads, "int32")  # noqa: E731
    if n_in == 2:

        @txl.kernel()
        def silu_mul_fwd_scalar(gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid(n), block=256))
            body([gate, up, out], n)

        return compile_kernel(silu_mul_fwd_scalar)

    @txl.kernel()
    def silu_mul_bwd_scalar(do: txl.Tensor((txl.int64(),), dtype, data_alignment=1), gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid(n), block=256))
        body([do, gate, up, grad_gate, grad_up], n)

    return compile_kernel(silu_mul_bwd_scalar)


def hinted(op, hint):
    for h in hint.split(".") if hint else []:
        op = getattr(op, h)
    return op


@functools.cache
def kernel_cfg(dtype: str, n_in: int, WARPS: int, VPT: int, VBYTES: int, LD_HINT: str, ST_HINT: str, LD_POL: str = "", ST_POL: str = "", FULL_TILES: bool = False):
    """
    n_in = 2 (fwd: gate, up -> out) or 3 (bwd: do, gate, up -> grad_gate, grad_up).
    """
    n_out = 1 if n_in == 2 else 2
    math = fwd_math if n_in == 2 else bwd_math
    THREADS = WARPS * 32
    nw = VBYTES // 4  # b32 words per vector
    vec = nw if dtype == "float32" else 2 * nw  # elements per vector
    ld_op, st_op = hinted(txl.ptx.ld.global_.nc, LD_HINT), hinted(txl.ptx.st.global_, ST_HINT)
    if LD_POL:
        ld_raw = getattr(getattr(ld_op, "L2::cache_hint"), f"v{nw}").b32
        ld = lambda *a: ld_raw(*a, txl.uint64(POLICY[LD_POL]))  # noqa: E731
    else:
        ld = getattr(ld_op, f"v{nw}").b32
    if ST_POL:
        st_raw = getattr(getattr(st_op, "L2::cache_hint"), f"v{nw}").b32
        st = lambda *a: st_raw(*a, txl.uint64(POLICY[ST_POL]))  # noqa: E731
    else:
        st = getattr(st_op, f"v{nw}").b32
    per_cta = THREADS * VPT * vec
    store16 = {"float32": None, "float16": "f16", "bfloat16": "bf16"}[dtype]

    def body(ptrs, n):
        ins, outs = ptrs[:n_in], ptrs[n_in:]
        cta, tid = txl.cast(txl.cta_id(), "int64"), txl.cast(txl.thread_id(), "int64")
        base = cta * per_cta
        words = [[txl.alloc_local([nw], "uint32") for _ in range(n_in)] for _ in range(VPT)]
        owords = [txl.alloc_local([nw], "uint32") for _ in range(n_out)]

        def offset(j):
            return base + (j * THREADS + tid) * vec

        def load(j):
            for t in range(n_in):
                w = words[j][t]
                ld(*[w[i] for i in range(nw)], txl.address_of(ins[t][offset(j)]))

        def compute_store(j):
            unpacked = [[x for w in range(nw) for x in unpack(words[j][t][w], dtype)] for t in range(n_in)]
            if FULL_TILES and dtype == "bfloat16":
                res_by_out = fwd_math_packed(unpacked) if n_in == 2 else bwd_math_packed(unpacked)
            else:
                res = [math(*[unpacked[t][e] for t in range(n_in)]) for e in range(vec)]
                res_by_out = [[res[e][o] for e in range(vec)] for o in range(n_out)]
            for o in range(n_out):
                pack(owords[o], res_by_out[o], dtype)
                st(txl.address_of(outs[o][offset(j)]), *[owords[o][i] for i in range(nw)])

        def vector(j):
            load(j)
            compute_store(j)

        def scalar(idx):
            vals = []
            for t in range(n_in):
                if dtype == "float32":
                    w = txl.local_scalar("uint32")
                    txl.ptx.ld.global_.nc.b32(w, txl.address_of(ins[t][idx]))
                    vals.append(txl.reinterpret(w, ty="float32"))
                else:
                    h = txl.local_scalar("uint16")
                    txl.ptx.ld.global_.nc.b16(h, txl.address_of(ins[t][idx]))
                    if dtype == "bfloat16":
                        vals.append(txl.reinterpret(txl.shift_left(txl.cast(h, "uint32"), txl.uint32(16)), ty="float32"))
                    else:
                        vals.append(to_f32(h, dtype))
            res = math(*vals)
            for o in range(n_out):
                if dtype == "float32":
                    txl.ptx.st.global_.b32(txl.address_of(outs[o][idx]), txl.reinterpret(res[o], ty="uint32"))
                else:
                    h = txl.local_scalar("uint16")
                    getattr(txl.ptx.cvt.rn, store16).f32(h, res[o])
                    txl.ptx.st.global_.b16(txl.address_of(outs[o][idx]), h)

        if FULL_TILES:
            for j in range(VPT):
                load(j)
            for j in range(VPT):
                compute_store(j)
        else:
            with txl.if_(base + per_cta <= n):
                with txl.then_():
                    for j in range(VPT):
                        load(j)
                    for j in range(VPT):
                        compute_store(j)
                with txl.else_():
                    for j in range(VPT):
                        off = offset(j)
                        with txl.if_(off + vec <= n):
                            with txl.then_():
                                vector(j)
                            with txl.else_():
                                for e in range(vec):
                                    with txl.if_(off + e < n):
                                        with txl.then_():
                                            scalar(off + e)

    grid = lambda n: txl.cast((n + (per_cta - 1)) // per_cta, "int32")  # noqa: E731
    if n_in == 2:

        @txl.kernel()
        def silu_mul_fwd(gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid(n), block=(WARPS) * 32))
            body([gate, up, out], n)

        return compile_kernel(silu_mul_fwd)

    @txl.kernel()
    def silu_mul_bwd(do: txl.Tensor((txl.int64(),), dtype, data_alignment=1), gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid(n), block=(WARPS) * 32))
        body([do, gate, up, grad_gate, grad_up], n)

    return compile_kernel(silu_mul_bwd)


def aligned(*ts):
    return all(t.data_ptr() % VBYTES == 0 for t in ts)


def silu_mul_fwd_sm100(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    silu(gate) * up of contiguous same-shape, same-dtype gate and up as a new tensor like gate, the
    bits of silu_mul_fwd_sm90.
    """
    assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.dtype == up.dtype, f"dtype mismatch: {gate.dtype} vs {up.dtype}"
    assert gate.is_contiguous(), "gate must be contiguous"
    assert up.is_contiguous(), "up must be contiguous"
    if gate.dtype not in DT:  # other float dtypes: the fp32 kernel, rounded once back
        return silu_mul_fwd_sm100(gate.float(), up.float()).to(gate.dtype)
    out = torch.empty_like(gate)
    n = gate.numel()
    if n == 0:
        return out
    kernel = pick_kernel(DT[gate.dtype], 2, n) if aligned(gate, up, out) else scalar_kernel(DT[gate.dtype], 2)
    kernel(gate.view(-1), up.view(-1), out.view(-1), n)
    return out


def silu_mul_bwd_sm100(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients of silu_mul_fwd_sm100 as new (grad_gate, grad_up) like gate and up, the bits of
    silu_mul_bwd_sm90.
    """
    assert grad_output.shape == gate.shape, f"grad_output shape {grad_output.shape} != gate shape {gate.shape}"
    assert grad_output.dtype == gate.dtype, f"grad_output dtype {grad_output.dtype} != gate dtype {gate.dtype}"
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    if gate.dtype not in DT or up.dtype != gate.dtype:  # the fp32 kernel, each rounded once back
        grad_gate, grad_up = silu_mul_bwd_sm100(grad_output.float(), gate.float(), up.float())
        return grad_gate.to(gate.dtype), grad_up.to(up.dtype)
    gate, up = gate.contiguous(), up.contiguous()
    grad_gate, grad_up = torch.empty_like(gate), torch.empty_like(up)
    n = gate.numel()
    if n == 0:
        return grad_gate, grad_up
    kernel = pick_kernel(DT[gate.dtype], 3, n) if aligned(grad_output, gate, up, grad_gate, grad_up) else scalar_kernel(DT[gate.dtype], 3)
    kernel(grad_output.view(-1), gate.view(-1), up.view(-1), grad_gate.view(-1), grad_up.view(-1), n)
    return grad_gate, grad_up
