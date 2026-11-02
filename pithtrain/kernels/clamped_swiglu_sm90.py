"""
gpt-oss's clamped SwiGLU over an interleaved gate_up and its gradient, in CuTe-DSL for SM90.

Flat over the M * N outputs: each thread owns one 16-byte vector of outputs (8 bf16 / fp16, 4 fp32,
2 fp64), reads its gate_up pairs with two 16-byte loads (backward: plus one of grad_output), applies
the Triton clamped_swiglu kernels' PTX in fp32 and writes the vector with one 16-byte store
(backward: two). The thread holding the last partial vector, and every thread when a pointer is not
16-byte aligned, goes element by element. The bf16 outputs are bit-identical to the Triton kernels.

The PTX replay (make_fwd_op, make_bwd_op) is also the activation of the fused FP8 kernels in
fp8_clamped_swiglu_quant_sm90.
"""

from __future__ import annotations

import struct

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm_args_last, bf16_hi, bf16_lo, f32_as_u32, ld_v4, pack_bf16, st_v4_args_last, to_ir, u32_as_f32, unpack
from pithtrain.kernels.fp8_e8m0_scale_sm90 import ptx_f32

__all__ = ["ClampedSwiGLU", "clamp_ptx", "clamped_swiglu_bwd_sm90", "clamped_swiglu_fwd_sm90", "f32", "make_bwd_op", "make_fwd_op", "sigmoid_ptx"]

THREADS = 128
DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16, torch.float32: Float32, torch.float64: cutlass.Float64}


def f32(x: float) -> float:
    """
    x rounded to float32, as Triton receives a Python float scalar argument.
    """
    return struct.unpack("<f", struct.pack("<f", x))[0]


# The clamped SwiGLU math replays the exact PTX the Triton clamped_swiglu kernels compile to (Triton
# 3.7), with L = limit, a = alpha:
#   gc = min.f32(g, L);  uc = min.f32(max.f32(u, 0 - L), L)
#   sigmoid(a * gc):  t = fma(-a, gc, 0);  t = t * log2e;  e = ex2.approx.f32 t;  d = e + 1;
#                     s = div.full.f32(1, d)
# fma(-a, gc, 0) == -(a * gc) rounded, so t * log2e == (a * gc) * -log2e (negation is
# exact). The remaining sigmoid steps are the 5-instruction sequence of silu_mul_quant's
# SIGMOID_PTX, verified there for every fp32 d >= 1 (div.full) and every ex2 input.
#   forward:  out = (uc + 1) * (gc * s)
#   backward: grad_uc = (gc * s) * dh
#             grad_gc = ((uc + 1) * dh * s) * fma(a * gc, 1 - s, 1)      (LLVM contraction)
#             grad_g  = g <= L ? grad_gc : 0;   grad_u = -L <= u <= L ? grad_uc : 0
def sigmoid_ptx(x: str) -> str:
    """
    PTX computing s = sigmoid(x) of the fp32 register x (uses t, s).
    """
    return (
        f"mul.rn.f32 t, {x}, 0fBFB8AA3B;\n"
        "ex2.approx.ftz.f32 t, t;\n"
        "fma.rn.f32 t, t, 0f3E800000, 0f3E800000;\n"
        "rcp.approx.ftz.f32 s, t;\n"
        "mul.rn.f32 s, s, 0f3E800000;\n"
    )


def clamp_ptx(alpha: float, limit: float) -> str:
    """
    gc, uc, ag = alpha * gc and s = sigmoid(ag) from g = $G, u = $U.
    """
    L, neg_L, A = ptx_f32(limit), ptx_f32(-limit), ptx_f32(alpha)
    return (
        f"min.f32 gc, $G, {L};\n"
        f"max.f32 uc, $U, {neg_L};\n"
        f"min.f32 uc, uc, {L};\n"
        f"mul.rn.f32 ag, gc, {A};\n" + sigmoid_ptx("ag")
    )


def make_fwd_op(alpha: float, limit: float):
    clamp_text = clamp_ptx(alpha, limit).replace("$G", "$1").replace("$U", "$2")

    @dsl_user_op
    def fwd_op(gate: Float32, up: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(
            asm_args_last(
                T.f32(),
                "{\n.reg .f32 gc, uc, ag, t, s;\n" + clamp_text + "mul.rn.f32 s, gc, s;\n"  # glu
                "add.rn.f32 uc, uc, 0f3F800000;\n"
                "mul.rn.f32 $0, uc, s;\n}",
                "=f,f,f",
                [to_ir(gate, Float32, loc, ip), to_ir(up, Float32, loc, ip)],
            )
        )

    return fwd_op


def make_bwd_op(alpha: float, limit: float):
    clamp_text = clamp_ptx(alpha, limit).replace("$G", "$3").replace("$U", "$4")
    L, neg_L = ptx_f32(limit), ptx_f32(-limit)

    @dsl_user_op
    def bwd_op(dh: Float32, gate: Float32, up: Float32, *, loc=None, ip=None) -> tuple:
        result_type = ir.Type.parse("!llvm.struct<(f32, f32)>")
        packed = asm_args_last(
            result_type,
            "{\n.reg .f32 gc, uc, ag, t, s, a, b;\n.reg .pred pg, pu, pl;\n"
            + clamp_text
            + "add.rn.f32 uc, uc, 0f3F800000;\n"
            "mul.rn.f32 a, uc, $2;\n"  # (uc + 1) * dh
            "mul.rn.f32 b, gc, s;\n"  # glu
            "mul.rn.f32 b, b, $2;\n"  # grad_uc
            "mul.rn.f32 a, a, s;\n"
            "sub.rn.f32 t, 0f3F800000, s;\n"
            "fma.rn.f32 t, ag, t, 0f3F800000;\n"
            "mul.rn.f32 a, a, t;\n"  # grad_gc
            f"setp.le.f32 pg, $3, {L};\n"
            f"setp.le.f32 pu, $4, {L};\n"
            f"setp.ge.f32 pl, $4, {neg_L};\n"
            "and.pred pu, pu, pl;\n"
            "selp.f32 $0, a, 0f00000000, pg;\n"
            "selp.f32 $1, b, 0f00000000, pu;\n}",
            "=f,=f,f,f,f",
            [to_ir(value, Float32, loc, ip) for value in (dh, gate, up)],
        )
        return unpack(packed, 2, Float32, loc, ip)

    return bwd_op


@dsl_user_op
def f16_lo(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $1;\ncvt.f32.f16 $0, lo;\n}", "=f,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def f16_hi(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $1;\ncvt.f32.f16 $0, hi;\n}", "=f,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def pack_f16(x_lo: Float32, x_hi: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Round two fp32 to FP16 (RN, as Triton's .to(tl.float16)) and pack {x_lo, x_hi}.
    """
    return Uint32(asm_args_last(T.i32(), "cvt.rn.f16x2.f32 $0, $1, $2;", "=r,f,f", [to_ir(x_hi, Float32, loc, ip), to_ir(x_lo, Float32, loc, ip)]))


@dsl_user_op
def f64_to_f32(word_lo: Uint32, word_hi: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "{\n.reg .f64 d;\nmov.b64 d, {$1, $2};\ncvt.rn.f32.f64 $0, d;\n}", "=f,r,r", [to_ir(word_lo, Uint32, loc, ip), to_ir(word_hi, Uint32, loc, ip)]))


@dsl_user_op
def f32_to_f64(x: Float32, *, loc=None, ip=None) -> tuple:
    """
    The fp64 value of x (exact) as its (low, high) words.
    """
    words = asm_args_last(ir.Type.parse("!llvm.struct<(i32, i32)>"), "{\n.reg .f64 d;\ncvt.f64.f32 d, $2;\nmov.b64 {$0, $1}, d;\n}", "=r,=r,f", [to_ir(x, Float32, loc, ip)])
    return unpack(words, 2, Uint32, loc, ip)


@dsl_user_op
def ld_elem(addr: Int64, pred: Int32, num_bytes: int, *, loc=None, ip=None) -> tuple:
    """
    One 2-, 4- or 8-byte global load as 1, 1 or 2 words (a 2-byte value in the low half); zeros
    where pred == 0.
    """
    args = [to_ir(addr, Int64, loc, ip), to_ir(pred, Int32, loc, ip)]
    if num_bytes == 8:
        words = asm_args_last(ir.Type.parse("!llvm.struct<(i32, i32)>"), "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\n@p ld.global.v2.b32 {$0, $1}, [$2];\n}", "=r,=r,l,r", args)
        return unpack(words, 2, Uint32, loc, ip)
    if num_bytes == 4:
        return (Uint32(asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, 0;\n@p ld.global.b32 $0, [$1];\n}", "=r,l,r", args)),)
    return (Uint32(asm_args_last(T.i32(), "{\n.reg .pred p;\n.reg .b16 h;\nsetp.ne.b32 p, $2, 0;\nmov.b16 h, 0;\n@p ld.global.b16 h, [$1];\ncvt.u32.u16 $0, h;\n}", "=r,l,r", args)),)


@dsl_user_op
def st_elem(addr: Int64, words: tuple, pred: Int32, num_bytes: int, *, loc=None, ip=None) -> None:
    """
    One 2-, 4- or 8-byte global store of ld_elem's words where pred != 0.
    """
    args = [to_ir(addr, Int64, loc, ip)] + [to_ir(word, Uint32, loc, ip) for word in words] + [to_ir(pred, Int32, loc, ip)]
    if num_bytes == 8:
        asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\n@p st.global.v2.b32 [$1], {$2, $3};\nmov.b32 $0, 0;\n}", "=r,l,r,r,r", args, side_effects=True)
    elif num_bytes == 4:
        asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\n@p st.global.b32 [$1], $2;\nmov.b32 $0, 0;\n}", "=r,l,r,r", args, side_effects=True)
    else:
        asm_args_last(T.i32(), "{\n.reg .pred p;\n.reg .b16 h;\nsetp.ne.b32 p, $3, 0;\ncvt.u16.u32 h, $2;\n@p st.global.b16 [$1], h;\nmov.b32 $0, 0;\n}", "=r,l,r,r", args, side_effects=True)


def ld_words(addr: Int64, pred: Int32, num_words: int) -> tuple:
    """
    num_words (1, 2 or 4) 32-bit words from addr where pred != 0, else zeros.
    """
    return ld_v4(addr, pred) if num_words == 4 else ld_elem(addr, pred, 4 * num_words)


def st_words(addr: Int64, words: list, pred: Int32) -> None:
    """
    1, 2 or 4 32-bit words to addr where pred != 0.
    """
    if len(words) == 4:
        st_v4_args_last(addr, tuple(words), pred)
    else:
        st_elem(addr, tuple(words), pred, 4 * len(words))


def decode(words: list, num_bytes: int, is_bf16: bool) -> list:
    """
    The fp32 values of the elements packed in words, in memory order.
    """
    if num_bytes == 2:
        lo, hi = (bf16_lo, bf16_hi) if is_bf16 else (f16_lo, f16_hi)
        return [half(word) for word in words for half in (lo, hi)]
    if num_bytes == 4:
        return [u32_as_f32(word) for word in words]
    return [f64_to_f32(words[2 * i], words[2 * i + 1]) for i in range(len(words) // 2)]


def encode(values: list, num_bytes: int, is_bf16: bool) -> list:
    """
    values rounded once to the element type and packed into words, in memory order.
    """
    if num_bytes == 2:
        pack = pack_bf16 if is_bf16 else pack_f16
        return [pack(values[2 * i], values[2 * i + 1]) for i in range(len(values) // 2)]
    if num_bytes == 4:
        return [f32_as_u32(value) for value in values]
    return [word for value in values for word in f32_to_f64(value)]


class ClampedSwiGLU:
    """
    clamped_swiglu_fwd (backward False) or clamped_swiglu_bwd over num_elements outputs (M * N),
    for one element type and alpha / limit. With aligned, every warp but the one holding the end of
    the outputs moves 16-byte vectors; that warp, and every thread without aligned, goes element by
    element.
    """

    def __init__(self, backward: bool, dtype: torch.dtype, alpha: float, limit: float, aligned: bool) -> None:
        self.backward, self.aligned = backward, aligned
        self.num_bytes, self.is_bf16 = dtype.itemsize, dtype == torch.bfloat16
        self.vec = 16 // self.num_bytes  # outputs per thread
        self.fwd_op, self.bwd_op = make_fwd_op(alpha, limit), make_bwd_op(alpha, limit)

    @cute.jit
    def __call__(self, mDh: cute.Tensor, mGU: cute.Tensor, mOut: cute.Tensor, num_elements: Int64, stream):
        per_cta = THREADS * self.vec
        self.kernel(mDh, mGU, mOut, num_elements).launch(grid=[Int32((num_elements + per_cta - 1) // per_cta), 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mDh: cute.Tensor, mGU: cute.Tensor, mOut: cute.Tensor, num_elements: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        first = (Int64(bx) * THREADS + tidx) * self.vec
        gate_up_base, out_base = mGU.iterator.toint(), mOut.iterator.toint()
        dh_base = gate_up_base
        if cutlass.const_expr(self.backward):
            dh_base = mDh.iterator.toint()
        if cutlass.const_expr(self.aligned):
            warp_end = (Int64(bx) * THREADS + tidx - tidx % 32 + 32) * self.vec
            if warp_end <= num_elements:
                self.apply_vector(first, dh_base, gate_up_base, out_base)
            else:
                if first < num_elements:
                    self.apply_partial(first, num_elements, dh_base, gate_up_base, out_base)
        else:
            if first < num_elements:
                self.apply_elements(first, num_elements, dh_base, gate_up_base, out_base)

    @cute.jit
    def apply_vector(self, first: Int64, dh_base, gate_up_base, out_base):
        """
        The thread's vec outputs from first: 16-byte loads and stores only.
        """
        num_bytes, is_bf16 = self.num_bytes, self.is_bf16
        gate_up_addr = gate_up_base + first * (2 * num_bytes)
        gate_up = decode(list(ld_v4(gate_up_addr, Int32(1)) + ld_v4(gate_up_addr + 16, Int32(1))), num_bytes, is_bf16)
        if cutlass.const_expr(self.backward):
            dh = decode(list(ld_v4(dh_base + first * num_bytes, Int32(1))), num_bytes, is_bf16)
            grad = []
            for k in cutlass.range_constexpr(self.vec):
                grad.extend(self.bwd_op(dh[k], gate_up[2 * k], gate_up[2 * k + 1]))
            words = encode(grad, num_bytes, is_bf16)
            st_v4_args_last(out_base + first * (2 * num_bytes), tuple(words[0:4]), Int32(1))
            st_v4_args_last(out_base + first * (2 * num_bytes) + 16, tuple(words[4:8]), Int32(1))
        else:
            h = [self.fwd_op(gate_up[2 * k], gate_up[2 * k + 1]) for k in range(self.vec)]
            st_v4_args_last(out_base + first * num_bytes, tuple(encode(h, num_bytes, is_bf16)), Int32(1))

    @cute.jit
    def apply_partial(self, first: Int64, num_elements: Int64, dh_base, gate_up_base, out_base):
        """
        The warp holding the end of the outputs: each thread's elements below num_elements, one
        gate/up pair (and one dh) load per element. Element 0 is loaded first and finished last; the
        others, only when there are more, load together under predicates before any of their
        (predicated) stores, so no store orders a later load and a 1-element tail runs little code.
        """
        num_bytes, is_bf16, vec = self.num_bytes, self.is_bf16, self.vec
        count = num_elements - first
        gate_up0 = decode(list(ld_words(gate_up_base + first * (2 * num_bytes), Int32(1), num_bytes // 2)), num_bytes, is_bf16)
        dh0 = gate_up0[0]
        if cutlass.const_expr(self.backward):
            dh0 = decode(list(ld_elem(dh_base + first * num_bytes, Int32(1), num_bytes)), num_bytes, is_bf16)[0]
        if count > 1:
            oks = [Int32(count > k) for k in range(vec)]
            gate_up = [decode(list(ld_words(gate_up_base + (first + k) * (2 * num_bytes), oks[k], num_bytes // 2)), num_bytes, is_bf16) for k in range(1, vec)]
            if cutlass.const_expr(self.backward):
                dhs = [decode(list(ld_elem(dh_base + (first + k) * num_bytes, oks[k], num_bytes)), num_bytes, is_bf16)[0] for k in range(1, vec)]
            for k in cutlass.range_constexpr(1, vec):
                if cutlass.const_expr(self.backward):
                    self.store_pair_grad(out_base, first + k, dhs[k - 1], gate_up[k - 1], oks[k])
                else:
                    self.store_output(out_base, first + k, gate_up[k - 1], oks[k])
        if cutlass.const_expr(self.backward):
            self.store_pair_grad(out_base, first, dh0, gate_up0, Int32(1))
        else:
            self.store_output(out_base, first, gate_up0, Int32(1))

    def store_output(self, out_base, elem, gate_up: list, pred: Int32) -> None:
        st_elem(out_base + elem * self.num_bytes, self.encode_one(self.fwd_op(gate_up[0], gate_up[1])), pred, self.num_bytes)

    def store_pair_grad(self, out_base, elem, dh, gate_up: list, pred: Int32) -> None:
        grad_gate, grad_up = self.bwd_op(dh, gate_up[0], gate_up[1])
        st_words(out_base + elem * (2 * self.num_bytes), encode([grad_gate, grad_up], self.num_bytes, self.is_bf16), pred)

    @cute.jit
    def apply_elements(self, first: Int64, num_elements: Int64, dh_base, gate_up_base, out_base):
        """
        The thread's outputs from first below num_elements, one element-sized load or store each;
        every load is issued before the first store (a store may alias a later load, so it would
        otherwise serialize the elements).
        """
        num_bytes, is_bf16, vec = self.num_bytes, self.is_bf16, self.vec
        elems = [first + k for k in range(vec)]
        oks = [Int32(elem < num_elements) for elem in elems]
        gates = [decode(list(ld_elem(gate_up_base + elems[k] * (2 * num_bytes), oks[k], num_bytes)), num_bytes, is_bf16)[0] for k in range(vec)]
        ups = [decode(list(ld_elem(gate_up_base + elems[k] * (2 * num_bytes) + num_bytes, oks[k], num_bytes)), num_bytes, is_bf16)[0] for k in range(vec)]
        if cutlass.const_expr(self.backward):
            dhs = [decode(list(ld_elem(dh_base + elems[k] * num_bytes, oks[k], num_bytes)), num_bytes, is_bf16)[0] for k in range(vec)]
            grads = [self.bwd_op(dhs[k], gates[k], ups[k]) for k in range(vec)]
            for k in cutlass.range_constexpr(vec):
                st_elem(out_base + elems[k] * (2 * num_bytes), self.encode_one(grads[k][0]), oks[k], num_bytes)
                st_elem(out_base + elems[k] * (2 * num_bytes) + num_bytes, self.encode_one(grads[k][1]), oks[k], num_bytes)
        else:
            outs = [self.fwd_op(gates[k], ups[k]) for k in range(vec)]
            for k in cutlass.range_constexpr(vec):
                st_elem(out_base + elems[k] * num_bytes, self.encode_one(outs[k]), oks[k], num_bytes)

    def encode_one(self, value) -> tuple:
        """
        The words of one element, rounded as encode rounds it (a 2-byte value in the low half).
        """
        if self.num_bytes == 2:
            return tuple(encode([value, Float32(0.0)], 2, self.is_bf16))
        return tuple(encode([value], self.num_bytes, self.is_bf16))


compiled: dict = {}


def get_kernel(backward: bool, dtype: torch.dtype, alpha: float, limit: float, aligned: bool, device: torch.device):
    """
    The element count stays a runtime argument, so one compile serves every shape.
    """
    key = ("clamped_swiglu", backward, dtype, alpha, limit, aligned, device.index)
    fn = compiled.get(key)
    if fn is None:
        cute_type, align = DTYPES[dtype], 16 if aligned else dtype.itemsize
        gate_up = fake(cute_type, (cute.sym_int(), cute.sym_int()), (cute.sym_int64(), 1), align)
        out = fake(cute_type, (cute.sym_int(), cute.sym_int()), (cute.sym_int64(), 1), 16)
        dh = fake(cute_type, (cute.sym_int(), cute.sym_int()), (cute.sym_int64(), 1), align) if backward else None
        fn = cute.compile(ClampedSwiGLU(backward, dtype, alpha, limit, aligned), dh, gate_up, out, Int64(0), fake_stream(), options="--enable-tvm-ffi")
        compiled[key] = fn
    return fn


def clamped_swiglu_fwd_sm90(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    clamped_swiglu_fwd on SM90; gate_up bf16, fp16, fp32 or fp64.
    """
    assert gate_up.dim() == 2, f"gate_up must be 2-D, got shape {gate_up.shape}"
    M, two_n = gate_up.shape
    assert two_n % 2 == 0, f"gate_up last dim must be even, got {two_n}"
    assert gate_up.is_contiguous(), "gate_up must be contiguous"
    assert gate_up.dtype in DTYPES, f"gate_up dtype {gate_up.dtype} is not a supported float dtype"
    N = two_n // 2
    out = torch.empty((M, N), device=gate_up.device, dtype=gate_up.dtype)
    num_elements = M * N
    if num_elements == 0:
        return out
    fn = get_kernel(False, gate_up.dtype, f32(alpha), f32(limit), gate_up.data_ptr() % 16 == 0, gate_up.device)
    fn(None, gate_up, out, num_elements)
    return out


def clamped_swiglu_bwd_sm90(grad_output: torch.Tensor, gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    clamped_swiglu_bwd on SM90; gate_up bf16, fp16, fp32 or fp64.
    """
    assert grad_output.shape[0] == gate_up.shape[0]
    assert grad_output.shape[1] * 2 == gate_up.shape[1]
    assert grad_output.dtype == gate_up.dtype
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    assert gate_up.is_contiguous(), "gate_up must be contiguous"
    assert gate_up.dtype in DTYPES, f"gate_up dtype {gate_up.dtype} is not a supported float dtype"
    grad_gate_up = torch.empty_like(gate_up)
    num_elements = grad_output.numel()
    if num_elements == 0:
        return grad_gate_up
    fn = get_kernel(True, gate_up.dtype, f32(alpha), f32(limit), gate_up.data_ptr() % 16 == 0 and grad_output.data_ptr() % 16 == 0, gate_up.device)
    fn(grad_output, gate_up, grad_gate_up, num_elements)
    return grad_gate_up
