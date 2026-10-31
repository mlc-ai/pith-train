"""
SM90 silu_mul: CuTe-DSL kernels computing silu(gate) * up and its two gradients, flat elementwise
over numel. Each thread of a 128-thread CTA owns one 16-byte vector of every tensor (8 bf16 / fp16
or 4 fp32 elements), one CTA per 128 vectors, and CTA 0 computes the numel % vector-size tail.
When any pointer is not 16-byte aligned the same kernel runs one element per thread. The math is
the Triton kernel's fp32 instruction sequence (SIGMOID_PTX, swiglu_fwd, swiglu_bwd): bf16 outputs
equal silu_mul_sm100's bit for bit (the sigmoid sequence is checked over every bf16 gate); fp16 and
fp32 run the same sequence and are verified to one output ulp of the fp32 reference. The fused FP8
kernels of fp8_silu_mul_quant_sm90 share those helpers.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, asm_args_last, bf16_hi, bf16_lo, f32_as_u32, ld_v4, pack_bf16, st_v4, to_ir, u32_as_f32, unpack

__all__ = ["SIGMOID_PTX", "SiluMul", "silu_mul_bwd_sm90", "silu_mul_fwd_sm90", "swiglu_bwd", "swiglu_fwd"]

THREADS = 128
MAX_ELEMENTS = 2**31  # vector indices are Int32 (byte offsets are Int64)


# sigmoid(x) = 1 / (1 + exp(-x)). Triton 3.7 emits, for fp32:
#     sub.f32 t, 0, x;  mul.f32 t, t, log2e;  ex2.approx.f32 e, t;  add.f32 d, e, 1;
#     div.full.f32 s, 1, d
# This 5-instruction sequence gives the same bits for every bf16 gate value (checked exhaustively):
# - x * -log2e == (0 - x) * log2e: negation is exact; ex2(+-0) = 1; NaN is canonical.
# - ex2 non-ftz differs from ftz only when e is subnormal, and then 1 + e == 1 either way.
# - div.full(1, d) == 0.25 * rcp(0.25 * d) for every fp32 d >= 1 (also checked exhaustively),
#   and 0.25 * (1 + e) == fma(e, 0.25, 0.25) because scaling by 0.25 is exact here.
# .rn pins every mul/add so ptxas cannot contract differently from the Triton kernels.
SIGMOID_PTX = (
    "mul.rn.f32 t, $G, 0fBFB8AA3B;\n"
    "ex2.approx.ftz.f32 t, t;\n"
    "fma.rn.f32 t, t, 0f3E800000, 0f3E800000;\n"
    "rcp.approx.ftz.f32 s, t;\n"
    "mul.rn.f32 s, s, 0f3E800000;\n"
)


@dsl_user_op
def swiglu_fwd(gate: Float32, up: Float32, *, loc=None, ip=None) -> Float32:
    """
    silu(gate) * up in fp32 as (sigmoid(gate) * gate) * up, the Triton forward's exact op order.
    """
    return Float32(asm_args_last(T.f32(), "{\n.reg .f32 t, s;\n" + SIGMOID_PTX.replace("$G", "$1") + "mul.rn.f32 s, s, $1;\nmul.rn.f32 $0, s, $2;\n}", "=f,f,f", [to_ir(gate, Float32, loc, ip), to_ir(up, Float32, loc, ip)]))


@dsl_user_op
def swiglu_bwd(grad_h: Float32, gate: Float32, up: Float32, *, loc=None, ip=None) -> tuple:
    """
    (grad_gate, grad_up) in fp32, the Triton backward's ops after LLVM contraction:

    grad_up   = grad_h * (s * gate)
    grad_gate = ((grad_h * up) * s) * fma(-s, gate, 1 + gate)      with s = sigmoid(gate)
    """
    result_type = ir.Type.parse("!llvm.struct<(f32, f32)>")
    packed = asm_args_last(
        result_type,
        "{\n.reg .f32 t, s, a, b;\n"
        + SIGMOID_PTX.replace("$G", "$3")
        + "mul.rn.f32 a, s, $3;\n"  # silu
        "mul.rn.f32 $1, a, $2;\n"  # grad_up
        "mul.rn.f32 b, $2, $4;\n"  # grad_h * up
        "mul.rn.f32 b, b, s;\n"  # * sigmoid
        "add.rn.f32 a, $3, 0f3F800000;\n"  # 1 + gate
        "neg.f32 t, s;\n"
        "fma.rn.f32 a, t, $3, a;\n"  # 1 + gate - s * gate, unrounded product
        "mul.rn.f32 $0, b, a;\n}",
        "=f,=f,f,f,f",
        [to_ir(value, Float32, loc, ip) for value in (grad_h, gate, up)],
    )
    return unpack(packed, 2, Float32, loc, ip)


@dsl_user_op
def ld_elem(addr: Int64, pred: Int32, bits: int, *, loc=None, ip=None) -> Uint32:
    """
    One 16- or 32-bit element, zero-extended into a word; 0 where pred == 0.
    """
    if bits == 16:
        ptx = "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b16 h, 0;\n\t@p ld.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}"
    else:
        ptx = "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, 0;\n\t@p ld.global.b32 $0, [$1];\n\t}"
    return Uint32(asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], ptx, "=r,l,r"))


@dsl_user_op
def st_elem(addr: Int64, word: Uint32, pred: Int32, bits: int, *, loc=None, ip=None) -> None:
    """
    The low 16 or 32 bits of word, stored where pred != 0.
    """
    if bits == 16:
        ptx = "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.b32 p, $2, 0;\n\tcvt.u16.u32 h, $1;\n\t@p st.global.b16 [$0], h;\n\t}"
    else:
        ptx = "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}"
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(word).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], ptx, "l,r,r", side_effects=True)


@dsl_user_op
def f16_lo(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm(T.f32(), [to_ir(x, Uint32, loc, ip)], "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.f16 $0, l;\n\t}", "=f,r"))


@dsl_user_op
def f16_hi(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm(T.f32(), [to_ir(x, Uint32, loc, ip)], "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.f16 $0, h;\n\t}", "=f,r"))


@dsl_user_op
def pack_f16(x_lo: Float32, x_hi: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Round two fp32 to fp16 (RN, as Triton's .to(tl.float16)) and pack {x_lo, x_hi}.
    """
    return Uint32(asm(T.i32(), [to_ir(x_hi, Float32, loc, ip), to_ir(x_lo, Float32, loc, ip)], "cvt.rn.f16x2.f32 $0, $1, $2;", "=r,f,f"))


class SiluMul:
    """
    silu_mul forward (inputs gate, up; output out) or backward (inputs grad_output, gate, up; outputs
    grad_gate, grad_up) over flat tensors. CTA bx covers vectors bx * THREADS .. bx * THREADS +
    THREADS - 1; a vector is 16 bytes when is_vector, else one element. has_tail: the numel %
    vec_elems elements after the last vector are computed by CTA 0, one per thread.
    """

    def __init__(self, dtype: torch.dtype, backward: bool, is_vector: bool, has_tail: bool) -> None:
        self.dtype, self.backward, self.is_vector, self.has_tail = dtype, backward, is_vector, has_tail
        self.elem_bytes = 4 if dtype == torch.float32 else 2
        self.vec_elems = 16 // self.elem_bytes if is_vector else 1
        self.num_inputs, self.num_outputs = (3, 2) if backward else (2, 1)

    @cute.jit
    def __call__(self, mIn0: cute.Tensor, mIn1: cute.Tensor, mIn2: cute.Tensor, mOut0: cute.Tensor, mOut1: cute.Tensor, num_vecs: Int32, num_elements: Int64, num_ctas: Int32, stream):
        self.kernel(mIn0, mIn1, mIn2, mOut0, mOut1, num_vecs, num_elements).launch(grid=[num_ctas, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mIn0: cute.Tensor, mIn1: cute.Tensor, mIn2: cute.Tensor, mOut0: cute.Tensor, mOut1: cute.Tensor, num_vecs: Int32, num_elements: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        in_bases, out_bases = [tensor.iterator.toint() for tensor in [mIn0, mIn1, mIn2][: self.num_inputs]], [tensor.iterator.toint() for tensor in [mOut0, mOut1][: self.num_outputs]]

        # vec_start < num_vecs, so this bound check cannot overflow Int32 (vec_start + tidx can, on masked threads only)
        vec_start = bx * THREADS
        vec_ok = Int32(tidx < num_vecs - vec_start)
        byte_offset = Int64(vec_start + tidx) * (self.vec_elems * self.elem_bytes)
        in_words = self.load_vectors(in_bases, byte_offset, vec_ok, self.is_vector)
        self.store_vectors(out_bases, byte_offset, self.apply_swiglu(in_words, self.is_vector), vec_ok, self.is_vector)

        if cutlass.const_expr(self.has_tail):
            self.process_tail(in_bases, out_bases, num_vecs, num_elements, tidx, bx)

    def load_vectors(self, in_bases: list, byte_offset: Int64, pred: Int32, is_vector: bool) -> list:
        """
        One tuple of 32-bit words per input: the four of a 16-byte vector, or one element
        zero-extended; zeros where pred == 0.
        """
        if cutlass.const_expr(is_vector):
            return [ld_v4(base + byte_offset, pred) for base in in_bases]
        return [(ld_elem(base + byte_offset, pred, self.elem_bytes * 8),) for base in in_bases]

    def apply_swiglu(self, in_words: list, is_vector: bool) -> list:
        """
        One tuple of words per output; output word j holds the results of the elements of word j of
        every input.
        """
        out_words = [[] for _ in range(self.num_outputs)]
        for words in zip(*in_words):
            columns = [self.unpack_word(word, is_vector) for word in words]
            results = [swiglu_bwd(*values) if self.backward else (swiglu_fwd(*values),) for values in zip(*columns)]
            for out_idx in range(self.num_outputs):
                out_words[out_idx].append(self.pack_word([result[out_idx] for result in results]))
        return [tuple(words) for words in out_words]

    def unpack_word(self, word: Uint32, is_vector: bool) -> list:
        """
        The fp32 values of a word, low element first: two 16-bit elements of a vector, or one
        element.
        """
        if self.dtype == torch.float32:
            return [u32_as_f32(word)]
        unpack_lo, unpack_hi = (bf16_lo, bf16_hi) if self.dtype == torch.bfloat16 else (f16_lo, f16_hi)
        return [unpack_lo(word), unpack_hi(word)] if is_vector else [unpack_lo(word)]

    def pack_word(self, values: list) -> Uint32:
        """
        values rounded to the dtype and packed low element first; one 16-bit value fills both
        halves (an element store writes the low one).
        """
        if self.dtype == torch.float32:
            return f32_as_u32(values[0])
        pack_values = pack_bf16 if self.dtype == torch.bfloat16 else pack_f16
        return pack_values(values[0], values[-1])

    def store_vectors(self, out_bases: list, byte_offset: Int64, out_words: list, pred: Int32, is_vector: bool) -> None:
        for base, words in zip(out_bases, out_words):
            if cutlass.const_expr(is_vector):
                st_v4(base + byte_offset, words, pred)
            else:
                st_elem(base + byte_offset, words[0], pred, self.elem_bytes * 8)

    def process_tail(self, in_bases: list, out_bases: list, num_vecs: Int32, num_elements: Int64, tidx: Int32, bx: Int32) -> None:
        """
        The num_elements % vec_elems elements after the last vector, one per thread of CTA 0.
        """
        elem_idx = Int64(num_vecs) * self.vec_elems + tidx
        elem_ok = Int32(bx == 0) & Int32(elem_idx < num_elements)
        byte_offset = elem_idx * self.elem_bytes
        in_words = self.load_vectors(in_bases, byte_offset, elem_ok, False)
        self.store_vectors(out_bases, byte_offset, self.apply_swiglu(in_words, False), elem_ok, False)


COMPILED: dict = {}


def get_kernel(dtype: torch.dtype, backward: bool, is_vector: bool, has_tail: bool, device: torch.device) -> object:
    key = ("silu_mul", dtype, backward, is_vector, has_tail, device.index)
    fn = COMPILED.get(key)
    if fn is None:
        elem_bytes = 4 if dtype == torch.float32 else 2
        flat = fake(TORCH2CUTE[dtype], (cute.sym_int(),), (1,), 16 if is_vector else elem_bytes)
        with torch.cuda.device(device):
            fn = cute.compile(SiluMul(dtype, backward, is_vector, has_tail), flat, flat, flat if backward else None, flat, flat if backward else None, Int32(1), Int64(1), Int32(1), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    return fn


def launch_silu_mul(inputs: tuple, outputs: tuple, backward: bool) -> None:
    """
    Run SiluMul over the flat inputs and outputs (all contiguous, same numel, dtype and CUDA device)
    on that device's current stream; no launch when empty.
    """
    tensors = inputs + outputs
    first = tensors[0]
    numel = first.numel()
    if numel == 0:
        return
    address_bits = 0
    for tensor in tensors:
        address_bits |= tensor.data_ptr()
    dtype, device, is_vector = first.dtype, first.device, address_bits % 16 == 0
    vec_elems = (4 if dtype == torch.float32 else 8) if is_vector else 1
    num_vecs, has_tail = numel // vec_elems, numel % vec_elems != 0
    flat = [tensor.view(-1) for tensor in tensors]
    if not backward:
        flat = [flat[0], flat[1], None, flat[2], None]
    # one CTA even without a full vector, for the tail; computed here so that it cannot overflow Int32
    num_ctas = max((num_vecs + THREADS - 1) // THREADS, 1)
    fn = get_kernel(dtype, backward, is_vector, has_tail, device)
    if device.index == torch.cuda.current_device():
        fn(*flat, num_vecs, numel, num_ctas)
    else:
        with torch.cuda.device(device):
            fn(*flat, num_vecs, numel, num_ctas)


def silu_mul_fwd_sm90(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    The SM90 silu_mul_fwd: silu(gate) * up as a new contiguous tensor like gate (bf16: bit-identical
    to silu_mul_fwd_sm100; fp16 / fp32: within one ulp of the fp32 formula). gate, up: contiguous,
    same shape, dtype (bf16, fp16 or fp32) and CUDA device, fewer than 2^31 elements, any alignment;
    not modified. Runs on that device's current stream.
    """
    assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.dtype == up.dtype, f"dtype mismatch: {gate.dtype} vs {up.dtype}"
    assert gate.dtype in (torch.bfloat16, torch.float16, torch.float32), f"unsupported dtype {gate.dtype}"
    assert gate.is_contiguous() and up.is_contiguous(), "gate and up must be contiguous"
    assert gate.is_cuda and gate.device == up.device, f"device mismatch: {gate.device} vs {up.device}"
    assert gate.numel() < MAX_ELEMENTS, f"{gate.numel()} elements, at most {MAX_ELEMENTS - 1} supported"
    out = torch.empty_like(gate)
    launch_silu_mul((gate, up), (out,), False)
    return out


def silu_mul_bwd_sm90(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The SM90 silu_mul_bwd: (grad_gate, grad_up) as new contiguous tensors (bf16: bit-identical to
    silu_mul_bwd_sm100; fp16 / fp32: within one ulp of the fp32 formulas). grad_output, gate, up:
    contiguous, same shape, dtype (bf16, fp16 or fp32) and CUDA device, fewer than 2^31 elements,
    any alignment; not modified. Runs on that device's current stream.
    """
    assert grad_output.shape == gate.shape == up.shape, f"shape mismatch: {grad_output.shape}, {gate.shape}, {up.shape}"
    assert grad_output.dtype == gate.dtype == up.dtype, f"dtype mismatch: {grad_output.dtype}, {gate.dtype}, {up.dtype}"
    assert gate.dtype in (torch.bfloat16, torch.float16, torch.float32), f"unsupported dtype {gate.dtype}"
    assert grad_output.is_contiguous() and gate.is_contiguous() and up.is_contiguous(), "grad_output, gate and up must be contiguous"
    assert gate.is_cuda and grad_output.device == gate.device == up.device, f"device mismatch: {grad_output.device}, {gate.device}, {up.device}"
    assert gate.numel() < MAX_ELEMENTS, f"{gate.numel()} elements, at most {MAX_ELEMENTS - 1} supported"
    grad_gate, grad_up = torch.empty_like(gate), torch.empty_like(up)
    launch_silu_mul((grad_output, gate, up), (grad_gate, grad_up), True)
    return grad_gate, grad_up
