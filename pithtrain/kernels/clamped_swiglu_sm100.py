"""
clamped_swiglu_sm100: the clamped SwiGLU forward and backward of an interleaved gate_up on SM100,
TIRx-lite kernels, the same bits as clamped_swiglu_sm90.

Flat elementwise kernels over n = M * N outputs. Each thread owns UNROLL groups of VEC consecutive
outputs (groups spaced one CTA-tile apart so every warp access stays coalesced); it issues all of
its wide loads before computing, then writes packed results. The fp32 sequence is sigmoid = 1 /
(1 + ex2.approx(x * -log2 e)) with div.full.f32 semantics, the ex2 / division range fixups that
cannot change the result removed (see sigmoid_parts), and the products
issued two outputs at a time as packed f32x2 (same per-lane rn rounding). Shapes, alpha and limit
are runtime arguments. bf16, fp16 and fp32 run natively; any other float dtype is widened to fp32,
run through the fp32 kernel and rounded once back (the arithmetic is fp32 with one rounding).
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["clamped_swiglu_bwd_sm100", "clamped_swiglu_fwd_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


CFG = {"fwd": {"threads": 256, "vec": 16, "unroll": 1, "ld": "ld.global.v8.b32", "st": "st.global.v8.b32"}, "bwd": {"threads": 64, "vec": 16, "unroll": 1, "ld": "ld.global.v8.b32", "st": "st.global.v8.b32"}}  # per entry: threads, outputs per thread (vec), unroll, load / store instruction
NEG_LOG2E_BITS = 0xBFB8AA3B  # -log2(e) rounded to fp32
DT = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def f32_bits(bits):
    return txl.reinterpret(txl.uint32(bits), ty="float32")


def op(chain, *args, dtype="float32"):
    out = txl.local_scalar(dtype)
    txl.ptx[chain](out, *args)
    return out


def sigmoid_parts(g, alpha):
    """
    x = alpha * g and s = sigmoid(x), in the module's fp32 sequence.
    """
    x = op("mul.rn.f32", alpha, g)
    t = op("mul.rn.f32", x, f32_bits(NEG_LOG2E_BITS))
    # ex2's denormal-range fixup cannot change e + 1 (both sides round to 1.0), so .ftz is exact here.
    e = op("ex2.approx.ftz.f32", t)
    # div.full.f32(1, d) for d = e + 1 >= 1 is rcp(d), or rcp(d / 4) / 4 when d > 2^126; MUFU.RCP is
    # exact under power-of-two scaling, so scaling by 1/4 unconditionally gives the same bits, and
    # fma(e, 1/4, 1/4) == rn(e + 1) / 4 exactly. (Verified over all 2^32 t on the B200.)
    r = op("rcp.approx.ftz.f32", op("fma.rn.f32", e, txl.float32(0.25), txl.float32(0.25)))
    s = op("mul.rn.f32", r, txl.float32(0.25))
    return x, s


def clamp(gate, up, limit, neg_limit):
    g = op("min.f32", gate, limit)
    u = op("min.f32", op("max.f32", up, neg_limit), limit)
    return g, u


def fwd_one(gate, up, alpha, limit, neg_limit):
    g, u = clamp(gate, up, limit, neg_limit)
    _, s = sigmoid_parts(g, alpha)
    ag = op("mul.rn.f32", g, s)
    u1 = op("add.rn.f32", u, txl.float32(1.0))
    return op("mul.rn.f32", u1, ag)


def bwd_one(go, gate, up, alpha, limit, neg_limit):
    g, u = clamp(gate, up, limit, neg_limit)
    x, s = sigmoid_parts(g, alpha)
    ag = op("mul.rn.f32", g, s)
    d_up = op("mul.rn.f32", go, ag)
    u1 = op("add.rn.f32", u, txl.float32(1.0))
    t = op("mul.rn.f32", op("mul.rn.f32", go, u1), s)
    f = op("fma.rn.f32", x, op("sub.rn.f32", txl.float32(1.0), s), txl.float32(1.0))
    d_gate = op("mul.rn.f32", t, f)
    d_gate = txl.Select(gate <= limit, d_gate, txl.float32(0.0))
    d_up = txl.Select(txl.And(up >= neg_limit, up <= limit), d_up, txl.float32(0.0))
    return d_gate, d_up


def pack2(a, b):
    """
    f32 pair (a in the low lane) as one .b64 f32x2 operand.
    """
    lo, hi = txl.cast(txl.reinterpret(a, ty="uint32"), "uint64"), txl.cast(txl.reinterpret(b, ty="uint32"), "uint64")
    return (hi << txl.uint64(32)) | lo


def lanes(p):
    return txl.reinterpret(txl.cast(p, "uint32"), ty="float32"), txl.reinterpret(txl.cast(p >> txl.uint64(32), "uint32"), ty="float32")


def op2(chain, *args):
    return op(chain, *args, dtype="uint64")


def splat_bits(bits):
    return pack2(f32_bits(bits), f32_bits(bits))


ONE_BITS, QUARTER_BITS = 0x3F800000, 0x3E800000


def sigmoid_parts2(G, A2):
    """
    sigmoid_parts on two lanes: the products in packed f32x2 (same rn rounding per lane).
    """
    X = op2("mul.rn.f32x2", A2, G)
    t0, t1 = lanes(op2("mul.rn.f32x2", X, splat_bits(NEG_LOG2E_BITS)))
    E = pack2(op("ex2.approx.ftz.f32", t0), op("ex2.approx.ftz.f32", t1))
    q0, q1 = lanes(op2("fma.rn.f32x2", E, splat_bits(QUARTER_BITS), splat_bits(QUARTER_BITS)))
    R = pack2(op("rcp.approx.ftz.f32", q0), op("rcp.approx.ftz.f32", q1))
    return X, op2("mul.rn.f32x2", R, splat_bits(QUARTER_BITS))


def fwd_two(gates, ups, alpha, limit, neg_limit):
    (g0, u0), (g1, u1) = (clamp(gates[k], ups[k], limit, neg_limit) for k in range(2))
    G = pack2(g0, g1)
    _, S = sigmoid_parts2(G, pack2(alpha, alpha))
    AG = op2("mul.rn.f32x2", G, S)
    U1 = op2("add.rn.f32x2", pack2(u0, u1), splat_bits(ONE_BITS))
    return lanes(op2("mul.rn.f32x2", U1, AG))


def bwd_two(gos, gates, ups, alpha, limit, neg_limit):
    (g0, u0), (g1, u1) = (clamp(gates[k], ups[k], limit, neg_limit) for k in range(2))
    G, GO = pack2(g0, g1), pack2(gos[0], gos[1])
    X, S = sigmoid_parts2(G, pack2(alpha, alpha))
    AG = op2("mul.rn.f32x2", G, S)
    d_up = lanes(op2("mul.rn.f32x2", GO, AG))
    U1 = op2("add.rn.f32x2", pack2(u0, u1), splat_bits(ONE_BITS))
    T = op2("mul.rn.f32x2", op2("mul.rn.f32x2", GO, U1), S)
    F = op2("fma.rn.f32x2", X, op2("sub.rn.f32x2", splat_bits(ONE_BITS), S), splat_bits(ONE_BITS))
    d_gate = lanes(op2("mul.rn.f32x2", T, F))
    res = []
    for k in range(2):
        dg, du = txl.Select(gates[k] <= limit, d_gate[k], txl.float32(0.0)), txl.Select(txl.And(ups[k] >= neg_limit, ups[k] <= limit), d_up[k], txl.float32(0.0))
        res += [dg, du]
    return res


class IO:
    """
    Load/convert helpers for one storage dtype, in 32-bit words.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        self.per_word = 1 if dtype == "float32" else 2
        self.esize = 4 // self.per_word
        self.cvt_in, self.cvt_out, self.cvt_out2 = {"bfloat16": "cvt.f32.bf16", "float16": "cvt.f32.f16"}.get(dtype), {"bfloat16": "cvt.rn.bf16.f32", "float16": "cvt.rn.f16.f32"}.get(dtype), {"bfloat16": "cvt.rn.bf16x2.f32", "float16": "cvt.rn.f16x2.f32"}.get(dtype)

    def unpack(self, words):
        vals = []
        for word in words:
            if self.dtype == "float32":
                vals.append(txl.reinterpret(word, ty="float32"))
            elif self.dtype == "bfloat16":  # bf16 -> f32 is exact: the bits move to the high half
                vals.append(txl.reinterpret(word << txl.uint32(16), ty="float32"))
                vals.append(txl.reinterpret(word & txl.uint32(0xFFFF0000), ty="float32"))
            else:
                vals.append(op(self.cvt_in, txl.cast(word & txl.uint32(0xFFFF), "uint16")))
                vals.append(op(self.cvt_in, txl.cast(word >> txl.uint32(16), "uint16")))
        return vals

    def pack(self, vals):
        if self.dtype == "float32":
            return [txl.reinterpret(v, ty="uint32") for v in vals]
        return [op(self.cvt_out2, vals[2 * i + 1], vals[2 * i], dtype="uint32") for i in range(len(vals) // 2)]

    def scalar_load(self, buf, idx):
        if self.dtype == "float32":
            v = txl.local_scalar("float32")
            txl.ptx.ld.global_.f32(v, buf.ptr_to([idx]))
            return v
        b = txl.local_scalar("uint16")
        txl.ptx.ld.global_.b16(b, buf.ptr_to([idx]))
        return op(self.cvt_in, b)

    def scalar_store(self, buf, idx, v):
        if self.dtype == "float32":
            txl.ptx.st.global_.f32(buf.ptr_to([idx]), v)
        else:
            txl.ptx.st.global_.b16(buf.ptr_to([idx]), op(self.cvt_out, v, dtype="uint16"))

    def load_words(self, buf, idx, nwords, insn):
        words = txl.alloc_local([nwords], "uint32")
        w8 = ".v8." in insn
        step = 8 if w8 and nwords % 8 == 0 else 4
        if step == 4:
            insn = insn.replace(".v8.", ".v4.")
        for w in range(0, nwords, step):
            txl.ptx[insn](*[words[w + j] for j in range(step)], buf.ptr_to([idx + w * 4 // self.esize]))
        return [words[w] for w in range(nwords)]

    def store_words(self, buf, idx, words, insn):
        step = 8 if ".v8." in insn and len(words) % 8 == 0 else 4
        if step == 4:
            insn = insn.replace(".v8.", ".v4.")
        for w in range(0, len(words), step):
            txl.ptx[insn](buf.ptr_to([idx + w * 4 // self.esize]), *words[w : w + step])


def grid_of(threads, vec, unroll, n):
    tile = vec * threads * unroll
    return txl.cast((n + tile - 1) // tile, "int32")


@functools.cache
def fwd_kernel(dtype, threads, vec, unroll):
    io, cfg = IO(dtype), CFG["fwd"]
    tile = vec * threads

    @txl.kernel()
    def clamped_swiglu_fwd(gate_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64, alpha: txl.float32, limit: txl.float32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid_of(threads, vec, unroll, n), block=(threads // 32) * 32))
        neg_limit = op("neg.f32", limit)
        base = txl.cast(txl.cta_id(), "int64") * (tile * unroll) + txl.cast(txl.thread_id(), "int64") * vec
        if vec == 1:  # unaligned fallback: one output per thread, scalar accesses only
            with txl.if_(base < n), txl.then_():
                io.scalar_store(out, base, fwd_one(io.scalar_load(gate_up, 2 * base), io.scalar_load(gate_up, 2 * base + 1), alpha, limit, neg_limit))
            return
        with txl.if_(base + (tile * (unroll - 1) + vec) <= n):
            with txl.then_():
                nw = 2 * vec // io.per_word
                words = [io.load_words(gate_up, (base + u * tile) * 2, nw, cfg["ld"]) for u in range(unroll)]
                for u in range(unroll):
                    vals = io.unpack(words[u])
                    res = []
                    for i in range(0, vec, 2):
                        res += list(fwd_two(vals[2 * i : 2 * i + 4 : 2], vals[2 * i + 1 : 2 * i + 4 : 2], alpha, limit, neg_limit))
                    io.store_words(out, base + u * tile, io.pack(res), cfg["st"])
            with txl.else_():
                with txl.serial(unroll * vec) as k:
                    i = base + txl.cast(k // vec, "int64") * tile + txl.cast(k % vec, "int64")
                    with txl.if_(i < n), txl.then_():
                        r = fwd_one(io.scalar_load(gate_up, 2 * i), io.scalar_load(gate_up, 2 * i + 1), alpha, limit, neg_limit)
                        io.scalar_store(out, i, r)

    return entry_function(clamped_swiglu_fwd.compile(TARGET))


@functools.cache
def bwd_kernel(dtype, threads, vec, unroll):
    io, cfg = IO(dtype), CFG["bwd"]
    tile = vec * threads

    @txl.kernel()
    def clamped_swiglu_bwd(grad_out: txl.Tensor((txl.int64(),), dtype, data_alignment=1), gate_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_gate_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64, alpha: txl.float32, limit: txl.float32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid_of(threads, vec, unroll, n), block=(threads // 32) * 32))
        neg_limit = op("neg.f32", limit)
        base = txl.cast(txl.cta_id(), "int64") * (tile * unroll) + txl.cast(txl.thread_id(), "int64") * vec
        if vec == 1:  # unaligned fallback: one output per thread, scalar accesses only
            with txl.if_(base < n), txl.then_():
                dg, du = bwd_one(io.scalar_load(grad_out, base), io.scalar_load(gate_up, 2 * base), io.scalar_load(gate_up, 2 * base + 1), alpha, limit, neg_limit)
                io.scalar_store(grad_gate_up, 2 * base, dg)
                io.scalar_store(grad_gate_up, 2 * base + 1, du)
            return
        with txl.if_(base + (tile * (unroll - 1) + vec) <= n):
            with txl.then_():
                nw = vec // io.per_word
                gw = [io.load_words(gate_up, (base + u * tile) * 2, 2 * nw, cfg["ld"]) for u in range(unroll)]
                ow = [io.load_words(grad_out, base + u * tile, nw, cfg["ld"]) for u in range(unroll)]
                for u in range(unroll):
                    vals, gos = io.unpack(gw[u]), io.unpack(ow[u])
                    res = []
                    for i in range(0, vec, 2):
                        res += bwd_two(gos[i : i + 2], vals[2 * i : 2 * i + 4 : 2], vals[2 * i + 1 : 2 * i + 4 : 2], alpha, limit, neg_limit)
                    io.store_words(grad_gate_up, (base + u * tile) * 2, io.pack(res), cfg["st"])
            with txl.else_():
                with txl.serial(unroll * vec) as k:
                    i = base + txl.cast(k // vec, "int64") * tile + txl.cast(k % vec, "int64")
                    with txl.if_(i < n), txl.then_():
                        dg, du = bwd_one(io.scalar_load(grad_out, i), io.scalar_load(gate_up, 2 * i), io.scalar_load(gate_up, 2 * i + 1), alpha, limit, neg_limit)
                        io.scalar_store(grad_gate_up, 2 * i, dg)
                        io.scalar_store(grad_gate_up, 2 * i + 1, du)

    return entry_function(clamped_swiglu_bwd.compile(TARGET))


def plan(entry, *tensors):
    """
    The vector paths need every base pointer 32-byte aligned (and whole 4-word vectors per thread);
    otherwise one output per thread with scalar accesses.
    """
    c = CFG[entry]
    if all(t.data_ptr() % 32 == 0 for t in tensors):
        return c["threads"], c["vec"], c["unroll"]
    return 256, 1, 1


def clamped_swiglu_fwd_sm100(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Clamped SwiGLU of the contiguous interleaved (M, 2N) gate_up as a new contiguous (M, N) tensor
    in its dtype, the bits of clamped_swiglu_fwd_sm90.
    """
    assert gate_up.dim() == 2, f"gate_up must be 2-D, got shape {gate_up.shape}"
    assert gate_up.shape[1] % 2 == 0, f"gate_up last dim must be even, got {gate_up.shape[1]}"
    assert gate_up.is_contiguous(), "gate_up must be contiguous"
    if gate_up.dtype not in DT:  # other float dtypes: the fp32 kernel, rounded once back
        return clamped_swiglu_fwd_sm100(gate_up.float(), alpha, limit).to(gate_up.dtype)
    M, two_n = gate_up.shape
    out = torch.empty((M, two_n // 2), device=gate_up.device, dtype=gate_up.dtype)
    n = out.numel()
    if n == 0:
        return out
    kernel = fwd_kernel(DT[gate_up.dtype], *plan("fwd", gate_up, out))
    kernel(gate_up.view(-1), out.view(-1), n, float(alpha), float(limit))
    return out


def clamped_swiglu_bwd_sm100(grad_output: torch.Tensor, gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Gradient of clamped_swiglu_fwd_sm100 with respect to the interleaved gate_up as a new (M, 2N)
    tensor like gate_up, the bits of clamped_swiglu_bwd_sm90.
    """
    assert grad_output.shape[0] == gate_up.shape[0]
    assert grad_output.shape[1] * 2 == gate_up.shape[1]
    assert grad_output.dtype == gate_up.dtype
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    if gate_up.dtype not in DT:  # other float dtypes: the fp32 kernel, rounded once back
        return clamped_swiglu_bwd_sm100(grad_output.float(), gate_up.float(), alpha, limit).to(gate_up.dtype)
    gate_up = gate_up.contiguous()
    grad_gate_up = torch.empty_like(gate_up)
    n = grad_output.numel()
    if n == 0:
        return grad_gate_up
    kernel = bwd_kernel(DT[gate_up.dtype], *plan("bwd", grad_output, gate_up, grad_gate_up))
    kernel(grad_output.view(-1), gate_up.view(-1), grad_gate_up.view(-1), n, float(alpha), float(limit))
    return grad_gate_up
