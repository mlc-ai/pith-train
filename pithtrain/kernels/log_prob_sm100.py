"""
log_prob_sm100: the fused per-token log-probability and entropy of a tempered softmax (forward) and
its backward overwriting the logits with their gradient, on SM100, as TIRx-lite kernels. Contracts
as in the picker log_prob.

Forward: one CTA per row, each thread streams 16-byte vectors and keeps a base-2 online state (m2,
d, s): u = x * c - m2 with c = log2(e) / T, d = sum 2^u, s = sum 2^u * u. The fast path skips the
per-element running max: each thread fixes its reference m2 from its first vector plus a margin,
which is exact as long as no 2^u overflows and no logit is -inf; a thread whose (d, s) come out
non-finite redoes its own elements with the exact online max (-inf clamped). A thread also redoes
its elements exactly when log2 d > REDO_LOG2 (its reference sits far below its mass), which keeps
the entropy cancellation and the backward's log2 p accurate on spike rows. The warp merge takes the
common maximum with Blackwell's redux.sync.max.f32, rescales each lane once and butterfly-sums d and
s; the CTA then merges the warps the same way. row_max holds m2 and log_sum holds log2(d) (base 2),
which only this file's backward reads.

Backward: one CTA per (row, CHUNK-vector slice), so the grid's tail is a short slice rather than a
whole row. u = (x * c - m2) - log2 d = log2 p, dx = p * (A + B * u) + [i == y] g_logp / T with A =
-(g_logp + g_entropy * H) / T and B = -g_entropy * ln2 / T. The thread that owns the target's vector
adds g_logp / T to that lane before the single rounding.

The kernels read bf16 or fp32 logits and int64 targets. Other float logits run them on an fp32 copy
(the gradient is computed in fp32 and rounded once into inp, as the Triton path does), other integer
targets on an int64 copy.

Evolved with TIRx-harness: run log_prob, frontier member hardware-redux-max (commit c12d48b0,
25669.0 us vs 37896.2 us, 1.476x over the previous SM100 path, the Triton kernels now in
log_prob_sm90, in the task's benchmark). Compiled once per process by tvm_tirx.tirx.compile(...,
pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc, as TVM_CUDA_COMPILE_MODE=nvcc chose before

__all__ = ["log_prob_bwd_sm100", "log_prob_fwd_sm100"]

def old_max(a, b):
    """
    txl.max as apache-tvm 0.27 emitted it: CUDA fmaxf for float32 operands, which drops a NaN operand
    (tvm_tirx emits a NaN-propagating select for txl.max); txl.max otherwise.
    """
    return txl.call_pure_extern("fmaxf", a, b, ty=txl.float32) if is_f32(a, b) else txl.max(a, b)


def old_min(a, b):
    """
    txl.min as apache-tvm 0.27 emitted it: CUDA fminf for float32 operands, txl.min otherwise.
    """
    return txl.call_pure_extern("fminf", a, b, ty=txl.float32) if is_f32(a, b) else txl.min(a, b)


def is_f32(*xs):
    """
    Whether any operand is a float32 expression (the other may be a Python float).
    """
    return any(str(getattr(getattr(x, "ty", None), "dtype", "")) == "float32" for x in xs)



EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
MARGIN = 0.0  # reference = first-vector max; 2^u overflow (u > ~117) falls back
REDO_LOG2 = 20.0
WARPS = 8
NT = WARPS * 32
UNROLL = 4
FULL = 0xFFFFFFFF


def f32(v):
    return txl.float32(v)


def i64(v):
    return txl.Cast("int64", v)


def load(x, dtype, vec, idx, f, off):
    """
    Load `vec` consecutive elements at element index `idx` (int64) into f[off:off+vec] (f32).
    """
    if dtype == "bfloat16":
        if vec == 8:
            w = txl.alloc_local([4], "uint32")
            txl.ptx["ld.global.nc.L1::no_allocate.v4.b32"](w[0], w[1], w[2], w[3], x.ptr_to([idx]))
            for i in range(4):
                txl.assign(f[off + 2 * i], txl.reinterpret(w[i] << txl.uint32(16), ty="float32"))
                txl.assign(f[off + 2 * i + 1], txl.reinterpret(w[i] & txl.uint32(0xFFFF0000), ty="float32"))
        else:
            h = txl.alloc_local([1], "uint16")
            txl.ptx["ld.global.nc.b16"](h[0], x.ptr_to([idx]))
            txl.assign(f[off], txl.reinterpret(txl.Cast("uint32", h[0]) << txl.uint32(16), ty="float32"))
    else:
        if vec == 4:
            txl.ptx["ld.global.nc.L1::no_allocate.v4.f32"](f[off], f[off + 1], f[off + 2], f[off + 3], x.ptr_to([idx]))
        else:
            txl.ptx["ld.global.nc.f32"](f[off], x.ptr_to([idx]))


def store(x, dtype, vec, idx, g, off):
    """
    Round g[off:off+vec] once to x's dtype and store at element index `idx`.
    """
    if dtype == "bfloat16":
        if vec == 8:
            w = txl.alloc_local([4], "uint32")
            for i in range(4):
                txl.ptx["cvt.rn.bf16x2.f32"](w[i], g[off + 2 * i + 1], g[off + 2 * i])
            txl.ptx["st.global.L1::no_allocate.v4.b32"](x.ptr_to([idx]), w[0], w[1], w[2], w[3])
        else:
            h = txl.alloc_local([1], "uint16")
            txl.ptx["cvt.rn.bf16.f32"](h[0], g[off])
            txl.ptx["st.global.b16"](x.ptr_to([idx]), h[0])
    else:
        if vec == 4:
            txl.ptx["st.global.L1::no_allocate.v4.f32"](x.ptr_to([idx]), g[off], g[off + 1], g[off + 2], g[off + 3])
        else:
            txl.ptx["st.global.f32"](x.ptr_to([idx]), g[off])


def finite(v):
    return (txl.reinterpret(v, ty="uint32") & txl.uint32(0x7F800000)) != txl.uint32(0x7F800000)


def ex2(dst, src):
    txl.ptx.ex2.approx.ftz.f32(dst, src)


def shfl(dst, v, delta):
    peer = txl.alloc_local([1], "uint32")
    txl.ptx.shfl_sync.bfly.b32(peer[0], txl.reinterpret(v, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL))
    txl.assign(dst, txl.reinterpret(peer[0], ty="float32"))


def warp_merge(st, rounds):
    # Blackwell provides a native fp32 warp max. Once every lane is expressed relative to the
    # same maximum, d and s are ordinary sums; there is no need to rescale at every tree level.
    m = txl.local_scalar("float32")
    txl.ptx.redux_sync.max.f32(m, st[0], txl.uint32(FULL))
    scale = txl.local_scalar("float32")
    ex2(scale, st[0] - m)
    txl.assign(st[2], (st[2] + (st[0] - m) * st[1]) * scale)
    txl.assign(st[1], st[1] * scale)
    txl.assign(st[0], m)
    o = txl.alloc_local([2], "float32")
    for delta in rounds:
        shfl(o[0], st[1], delta)
        shfl(o[1], st[2], delta)
        txl.assign(st[1], st[1] + o[0])
        txl.assign(st[2], st[2] + o[1])


def sweep(x, dtype, vec, base, nvec, tid, f, body, j0=0, nt=NT):
    """
    Visit this thread's vectors j = j0 + tid + k * nt below j0 + nvec: UNROLL loads in flight,
    then a guarded tail.
    """
    n_main = nvec // (UNROLL * nt)
    tail0 = n_main * (UNROLL * nt)
    with txl.serial(n_main) as it:
        for q in range(UNROLL):
            j = j0 + it * (UNROLL * nt) + q * nt + tid
            load(x, dtype, vec, base + i64(j) * vec, f, q * vec)
        for q in range(UNROLL):
            body(q * vec, j0 + it * (UNROLL * nt) + q * nt + tid)
    with txl.serial((nvec - tail0 + nt - 1) // nt) as it:
        j = tail0 + it * nt + tid
        with txl.if_(j < nvec), txl.then_():
            load(x, dtype, vec, base + i64(j0 + j) * vec, f, 0)
            body(0, j0 + j)


@functools.cache
def fwd_kernel(dtype, vec):
    @txl.kernel()
    def log_prob_fwd(x: txl.Tensor((txl.int64(),), dtype), tgt: txl.Tensor((txl.int64(),), txl.i64), logp: txl.Tensor((txl.int64(),), txl.f32), ent: txl.Tensor((txl.int64(),), txl.f32), rmax: txl.Tensor((txl.int64(),), txl.f32), lsum: txl.Tensor((txl.int64(),), txl.f32), n_rows: txl.int32, n_cols: txl.int32, x_stride: txl.int64, t_stride: txl.int64, inv_t: txl.float32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=n_rows, block=(WARPS) * 32))
        tid = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        # Reverse row order: the rows a preceding pass touched last are still in L2.
        row = txl.local_scalar("int32", init=n_rows - 1 - txl.cta_id())
        base = txl.local_scalar("int64", init=i64(row) * x_stride)
        c = txl.local_scalar("float32", init=inv_t * f32(LOG2E))
        nvec = n_cols // vec

        smem = txl.smem_pool()
        red = smem.alloc([WARPS * 3], txl.f32)

        st = txl.alloc_local([3], "float32")  # m2, d, s
        dd = txl.alloc_local([UNROLL], "float32")
        ss = txl.alloc_local([UNROLL], "float32")
        ev = txl.alloc_local([vec], "float32")
        sv = txl.alloc_local([vec], "float32")
        f = txl.alloc_local([UNROLL * vec], "float32")
        u = txl.alloc_local([1], "float32")

        # Target logit (thread 0), loaded early.
        y = txl.local_scalar("int64", init=txl.int64(-1))
        xy = txl.alloc_local([1], "float32")
        with txl.if_(tid == 0), txl.then_():
            txl.ptx["ld.global.nc.b64"](y, tgt.ptr_to([i64(row) * t_stride]))
            txl.assign(xy[0], f32(0.0))
            with txl.if_(txl.And(y >= txl.int64(0), y < i64(n_cols))), txl.then_():
                load(x, dtype, 1, base + y, xy, 0)

        # ---- fast path: one fixed reference per thread ----
        txl.assign(st[0], f32(-1e30))
        with txl.if_(tid < nvec), txl.then_():
            load(x, dtype, vec, base + i64(tid) * vec, f, 0)
            cm = txl.local_scalar("float32", init=f[0])
            for k in range(1, vec):
                txl.assign(cm, old_max(cm, f[k]))
            txl.assign(st[0], cm * c + f32(MARGIN))
        for q in range(UNROLL):
            txl.assign(dd[q], f32(0.0))
            txl.assign(ss[q], f32(0.0))

        def accumulate(off):
            """
            Tree-sum the vector's (e, e * u) and add it to the slot's accumulators.
            """
            q = off // vec
            w = vec
            while w > 1:
                w //= 2
                for k in range(w):
                    txl.assign(ev[k], ev[k] + ev[k + w])
                    txl.assign(sv[k], sv[k] + sv[k + w])
            txl.assign(dd[q], dd[q] + ev[0])
            txl.assign(ss[q], ss[q] + sv[0])

        def fast(off, j):
            for k in range(vec):
                txl.assign(u[0], f[off + k] * c - st[0])
                ex2(ev[k], u[0])
                txl.assign(sv[k], ev[k] * u[0])
            accumulate(off)

        def slow(off, j):
            cm = txl.local_scalar("float32", init=f[off])
            for k in range(1, vec):
                txl.assign(cm, old_max(cm, f[off + k]))
            nm = txl.local_scalar("float32", init=old_max(st[0], cm * c))
            with txl.if_(nm > st[0]), txl.then_():
                sc = txl.local_scalar("float32")
                ex2(sc, st[0] - nm)
                for q in range(UNROLL):
                    txl.assign(ss[q], (ss[q] + (st[0] - nm) * dd[q]) * sc)
                    txl.assign(dd[q], dd[q] * sc)
                txl.assign(st[0], nm)
            for k in range(vec):
                txl.assign(u[0], old_max(f[off + k] * c - st[0], f32(-3.0e38)))
                ex2(ev[k], u[0])
                txl.assign(sv[k], ev[k] * u[0])
            accumulate(off)

        sweep(x, dtype, vec, base, nvec, tid, f, fast)
        txl.assign(st[1], (dd[0] + dd[1]) + (dd[2] + dd[3]))
        txl.assign(st[2], (ss[0] + ss[1]) + (ss[2] + ss[3]))
        # ---- exact fallback for this thread's elements: overflow, -inf or NaN, or a reference
        # far below the thread's mass (log2 d > REDO_LOG2), which would cost fp32 accuracy in
        # the entropy's log2 d - s / d cancellation and in the backward's log2 p ----
        l2t = txl.local_scalar("float32")
        txl.ptx.lg2.approx.ftz.f32(l2t, st[1])
        bad = txl.Or(txl.Not(txl.And(finite(st[1]), finite(st[2]))), l2t > f32(REDO_LOG2))
        with txl.if_(bad), txl.then_():
            txl.assign(st[0], f32(-1e30))
            for q in range(UNROLL):
                txl.assign(dd[q], f32(0.0))
                txl.assign(ss[q], f32(0.0))
            sweep(x, dtype, vec, base, nvec, tid, f, slow)
            txl.assign(st[1], (dd[0] + dd[1]) + (dd[2] + dd[3]))
            txl.assign(st[2], (ss[0] + ss[1]) + (ss[2] + ss[3]))

        # ---- CTA merge ----
        warp_merge(st, (16, 8, 4, 2, 1))
        with txl.if_(lane == 0), txl.then_():
            for q in range(3):
                txl.ptx["st.shared.f32"](red.ptr_to([warp * 3 + q]), st[q])
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NT))
        with txl.if_(warp == 0), txl.then_():
            with txl.if_(lane < WARPS):
                with txl.then_():
                    for q in range(3):
                        txl.ptx["ld.shared.f32"](st[q], red.ptr_to([lane * 3 + q]))
                with txl.else_():
                    txl.assign(st[0], f32(-1e30))
                    txl.assign(st[1], f32(0.0))
                    txl.assign(st[2], f32(0.0))
            warp_merge(st, tuple(1 << i for i in reversed(range(WARPS.bit_length() - 1))))
            with txl.if_(tid == 0), txl.then_():
                l2d = txl.local_scalar("float32")
                txl.ptx.lg2.approx.ftz.f32(l2d, st[1])
                lp = txl.local_scalar("float32", init=f32(0.0))
                with txl.if_(txl.And(y >= txl.int64(0), y < i64(n_cols))), txl.then_():
                    txl.assign(lp, ((xy[0] * c - st[0]) - l2d) * f32(LN2))
                r = i64(row)
                txl.ptx["st.global.f32"](logp.ptr_to([r]), lp)
                txl.ptx["st.global.f32"](ent.ptr_to([r]), (l2d - st[2] / st[1]) * f32(LN2))
                txl.ptx["st.global.f32"](rmax.ptr_to([r]), st[0])
                txl.ptx["st.global.f32"](lsum.ptr_to([r]), l2d)

    return log_prob_fwd


CHUNK = 2048  # vectors per backward CTA (16384 bf16 elements)
BWARPS = 8
BNT = BWARPS * 32


@functools.cache
def bwd_kernel(dtype, vec):
    @txl.kernel()
    def log_prob_bwd(x: txl.Tensor((txl.int64(),), dtype), tgt: txl.Tensor((txl.int64(),), txl.i64), rmax: txl.Tensor((txl.int64(),), txl.f32), lsum: txl.Tensor((txl.int64(),), txl.f32), ent: txl.Tensor((txl.int64(),), txl.f32), glogp: txl.Tensor((txl.int64(),), txl.f32), gent: txl.Tensor((txl.int64(),), txl.f32), n_ctas: txl.int32, n_chunks: txl.int32, n_cols: txl.int32, x_stride: txl.int64, t_stride: txl.int64, gl_stride: txl.int64, ge_stride: txl.int64, inv_t: txl.float32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=n_ctas, block=(BWARPS) * 32))
        tid = txl.thread_id()
        cta = txl.cta_id()
        row = txl.local_scalar("int32", init=cta // n_chunks)
        j0 = txl.local_scalar("int32", init=(cta % n_chunks) * CHUNK)
        r = i64(row)
        base = txl.local_scalar("int64", init=r * x_stride)
        c = txl.local_scalar("float32", init=inv_t * f32(LOG2E))
        nvec = n_cols // vec
        cnt = txl.local_scalar("int32", init=old_min(nvec - j0, CHUNK))

        y = txl.local_scalar("int64")
        txl.ptx["ld.global.nc.b64"](y, tgt.ptr_to([r * t_stride]))
        sc = txl.alloc_local([5], "float32")  # m2, log2 d, H, g_logp, g_entropy
        txl.ptx["ld.global.nc.f32"](sc[0], rmax.ptr_to([r]))
        txl.ptx["ld.global.nc.f32"](sc[1], lsum.ptr_to([r]))
        txl.ptx["ld.global.nc.f32"](sc[2], ent.ptr_to([r]))
        txl.ptx["ld.global.nc.f32"](sc[3], glogp.ptr_to([r * gl_stride]))
        txl.ptx["ld.global.nc.f32"](sc[4], gent.ptr_to([r * ge_stride]))
        valid = txl.And(y >= txl.int64(0), y < i64(n_cols))
        gl = txl.local_scalar("float32", init=txl.Select(valid, sc[3], f32(0.0)))

        aa = txl.local_scalar("float32", init=-(gl + sc[4] * sc[2]) * inv_t)
        bb = txl.local_scalar("float32", init=-(sc[4] * f32(LN2)) * inv_t)
        ga = txl.local_scalar("float32", init=gl * inv_t)
        yv = txl.local_scalar("int32", init=txl.Select(valid, txl.Cast("int32", y // vec), -1))
        yr = txl.local_scalar("int32", init=txl.Cast("int32", y % vec))

        f = txl.alloc_local([UNROLL * vec], "float32")
        u = txl.alloc_local([1], "float32")
        p = txl.alloc_local([1], "float32")

        def body(off, j):
            for k in range(vec):
                txl.assign(u[0], old_max((f[off + k] * c - sc[0]) - sc[1], f32(-256.0)))
                ex2(p[0], u[0])
                txl.assign(f[off + k], p[0] * (aa + bb * u[0]))
            # The target's own lane takes + g_logp / T before the single rounding.
            with txl.if_(j == yv), txl.then_():
                for k in range(vec):
                    txl.assign(f[off + k], f[off + k] + txl.Select(yr == k, ga, f32(0.0)))
            store(x, dtype, vec, base + i64(j) * vec, f, off)

        sweep(x, dtype, vec, base, cnt, tid, f, body, j0, BNT)

    return log_prob_bwd


def flat(t):
    """
    A 1-D view over the storage that `t`'s (possibly strided or stride-0) elements span.
    """
    if t.dim() == 1 and t.stride(0) == 1:
        return t
    n = t.shape[0]
    stride = t.stride(0) if n > 1 else 1
    span = (n - 1) * stride + 1 if stride > 0 else 1
    return t.as_strided((span,), (1,))


def vec_for(inp):
    vec = 8 if inp.dtype == torch.bfloat16 else 4
    n_cols = inp.shape[1]
    ok = n_cols % vec == 0 and inp.stride(0) % vec == 0 and inp.data_ptr() % 16 == 0
    return vec if ok else 1


def x_view(inp):
    n_rows, n_cols = inp.shape
    if inp.is_contiguous():
        return inp.view(-1)
    return inp.as_strided(((n_rows - 1) * inp.stride(0) + n_cols,), (1,))


@functools.cache
def compiled(kind, dtype, vec):
    k = (fwd_kernel if kind == "fwd" else bwd_kernel)(dtype, vec)
    return entry_function(k.compile(TARGET))


DT = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


def covered(inp, target):
    """
    inp and target in dtypes the kernels read: other float logits as an fp32 copy, other integer
    targets as an int64 copy.
    """
    return (inp if inp.dtype in DT else inp.float()), (target if target.dtype == torch.int64 else target.long())


def log_prob_fwd_sm100(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (logp, entropy, row_max, log_sum), each fp32 (N,); inp is left unchanged. row_max and log_sum
    are the base-2 row statistics log_prob_bwd_sm100 reads.
    """
    n_rows, n_cols = inp.shape
    logp = torch.empty(n_rows, dtype=torch.float32, device=inp.device)
    entropy, row_max, log_sum = torch.empty_like(logp), torch.empty_like(logp), torch.empty_like(logp)
    if n_rows == 0:
        return logp, entropy, row_max, log_sum
    if n_cols == 0:
        # an empty distribution: no target is valid, and the entropy is an empty sum
        return logp.zero_(), entropy.zero_(), row_max.zero_(), log_sum.zero_()
    inp, target = covered(inp, target)
    vec = vec_for(inp)
    fn = compiled("fwd", DT[inp.dtype], vec)
    t_stride = target.stride(0) if n_rows > 1 else 1
    fn(x_view(inp), flat(target), logp, entropy, row_max, log_sum, n_rows, n_cols, inp.stride(0), t_stride, float(inv_temperature))
    return logp, entropy, row_max, log_sum


def log_prob_bwd_sm100(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    """
    Overwrite inp with dL/dx, from the statistics of log_prob_fwd_sm100.
    """
    n_rows, n_cols = inp.shape
    if n_rows == 0 or n_cols == 0:
        return
    out = inp
    inp, target = covered(inp, target)
    vec = vec_for(inp)
    fn = compiled("bwd", DT[inp.dtype], vec)
    t_stride = target.stride(0) if n_rows > 1 else 1
    gl_stride = grad_logp.stride(0) if n_rows > 1 else 1
    ge_stride = grad_entropy.stride(0) if n_rows > 1 else 1
    n_chunks = (n_cols // vec + CHUNK - 1) // CHUNK
    fn(x_view(inp), flat(target), row_max, log_sum, entropy, flat(grad_logp), flat(grad_entropy), n_rows * n_chunks, n_chunks, n_cols, inp.stride(0), t_stride, gl_stride, ge_stride, float(inv_temperature))
    if inp is not out:
        out.copy_(inp)  # the fp32 gradient, rounded once to the logits' dtype
