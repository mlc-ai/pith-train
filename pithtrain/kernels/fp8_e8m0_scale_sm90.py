"""
Inline-PTX E8M0 scale math for the SM90 CuTe-DSL quantization kernels: the SM90 branch of
compute_fp8_scale (fp8_scale.py) bit for bit, and the RN-satfinite e4m3 cast Triton's
.to(tl.float8e4nv) lowers to.
"""

import struct

from cutlass import Float32, Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_ptx_sm90 import asm, asm_args_last, f32_hex, to_ir, unpack

__all__ = ["AMAX_MIN", "FP8_MAX_RCP", "e8m0_scale", "e8m0_scale_bf16x2", "e8m0_scale_f32", "exact_rcp", "ptx_f32", "quant4_e4m3"]


def ptx_f32(x: float) -> str:
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


FP8_MAX_RCP, AMAX_MIN = ptx_f32(1.0 / 448.0), ptx_f32(1e-4)


@dsl_user_op
def e8m0_scale_f32(amax: Float32, *, loc=None, ip=None) -> tuple:
    """
    SM90 branch of compute_fp8_scale: (scale, reciprocal), both exact powers of 2.
    """
    pair_type = ir.Type.parse("!llvm.struct<(f32, f32)>")
    scale_pair = asm_args_last(
        pair_type,
        "{\n.reg .f32 a, s;\n.reg .b32 b, m, e;\n.reg .pred p;\n"
        f"max.f32 a, $2, {AMAX_MIN};\n"
        f"mul.rn.f32 s, a, {FP8_MAX_RCP};\n"
        "mov.b32 b, s;\n"
        "and.b32 m, b, 0x007FFFFF;\n"
        "and.b32 e, b, 0x7F800000;\n"
        "setp.ne.b32 p, m, 0;\n"
        "@p add.s32 e, e, 0x00800000;\n"
        "mov.b32 s, e;\n"
        "max.f32 s, s, 0f00800000;\n"
        "mov.b32 $0, s;\n"
        "mov.b32 e, s;\n"
        "sub.s32 e, 0x7F000000, e;\n"
        "mov.b32 $1, e;\n}",
        "=f,=f,f", [to_ir(amax, Float32, loc, ip)],
    )
    return unpack(scale_pair, 2, Float32, loc, ip)


@dsl_user_op
def quant4_e4m3(x0: Float32, x1: Float32, x2: Float32, x3: Float32, r: tuple, *, loc=None, ip=None) -> Uint32:
    """
    Four cvt.rn.satfinite.e4m3 of x_i * r[i], r holding the four reciprocal scales, packed
    little-endian into one word.
    """
    args = [to_ir(value, Float32, loc, ip) for value in (x0, x1, x2, x3)]
    args += [to_ir(value, Float32, loc, ip) for value in r]
    return Uint32(
        asm_args_last(
            T.i32(),
            "{\n.reg .f32 y0, y1, y2, y3;\n.reg .b16 lo, hi;\n"
            "mul.rn.f32 y0, $1, $5;\nmul.rn.f32 y1, $2, $6;\n"
            "mul.rn.f32 y2, $3, $7;\nmul.rn.f32 y3, $4, $8;\n"
            "cvt.rn.satfinite.e4m3x2.f32 lo, y1, y0;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, y3, y2;\n"
            "mov.b32 $0, {lo, hi};\n}",
            "=r,f,f,f,f,f,f,f,f", args,
        )
    )


AMAX_MIN_BITS = f32_hex(1e-4)
FP8_MAX_RCP_BITS = f32_hex(1.0 / 448.0)


@dsl_user_op
def e8m0_scale_bf16x2(amax2: Uint32, hi: Int32, *, loc=None, ip=None) -> tuple:
    """
    E8M0 scale of the bf16 amax in the low (hi == 0) or high half of amax2, as the fp32 bit patterns
    (scale, reciprocal); matches compute_fp8_scale's SM90 branch bit for bit.
    """
    pair_type = llvm.StructType.get_literal([T.i32(), T.i32()])
    scale_bits_pair = asm(
        pair_type, [Uint32(amax2).ir_value(loc=loc, ip=ip), Int32(hi).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a, b, m, s;\n\t.reg .f32 f;\n\t.reg .pred p, q;\n\t"
        "setp.ne.b32 q, $3, 0;\n\t"
        "and.b32 a, $2, 0xFFFF0000;\n\t"
        "shl.b32 b, $2, 16;\n\t"
        "selp.b32 a, a, b, q;\n\t"
        f"max.u32 a, a, {AMAX_MIN_BITS};\n\t"  # amax >= 0, so integer max == float max
        "mov.b32 f, a;\n\t"
        f"mul.rn.f32 f, f, 0f{FP8_MAX_RCP_BITS[2:]};\n\t"
        "mov.b32 a, f;\n\t"
        "and.b32 m, a, 0x007FFFFF;\n\t"
        "and.b32 s, a, 0x7F800000;\n\t"
        "setp.ne.b32 p, m, 0;\n\t"
        "@p add.u32 s, s, 0x00800000;\n\t"
        "max.u32 s, s, 0x00800000;\n\t"  # max(scale, 2^-126)
        "mov.b32 $0, s;\n\t"
        "sub.u32 $1, 0x7F000000, s;\n\t}",
        "=r,=r,r,r",
    )
    return (Uint32(llvm.extractvalue(T.i32(), scale_bits_pair, [0], loc=loc, ip=ip)), Uint32(llvm.extractvalue(T.i32(), scale_bits_pair, [1], loc=loc, ip=ip)))


@dsl_user_op
def e8m0_scale(amax: Float32, *, loc=None, ip=None) -> Float32:
    """
    SM90 branch of compute_fp8_scale: power-of-2 ceil of max(amax, 1e-4) / 448.

    0f38D1B717 is fp32(1e-4) and 0f3B124925 is fp32(1/448), the constants Triton folds. The final
    max.u32 with 0x00800000 clamps to the smallest normal (2^-126); on non-negative floats an
    unsigned compare of the bits orders like the float compare.
    """
    scale = llvm.inline_asm(
        T.f32(), [Float32(amax).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .f32 t;\n\t"
        ".reg .b32 b, m, e;\n\t"
        ".reg .pred p;\n\t"
        "max.f32 t, $1, 0f38D1B717;\n\t"
        "mul.rn.f32 t, t, 0f3B124925;\n\t"
        "mov.b32 b, t;\n\t"
        "and.b32 m, b, 0x007FFFFF;\n\t"
        "and.b32 e, b, 0x7F800000;\n\t"
        "setp.ne.u32 p, m, 0;\n\t"
        "@p add.u32 e, e, 0x00800000;\n\t"
        "max.u32 e, e, 0x00800000;\n\t"
        "mov.b32 $0, e;\n\t"
        "}",
        "=f,f", has_side_effects=False, is_align_stack=False, loc=loc, ip=ip,
    )
    return Float32(scale)


@dsl_user_op
def exact_rcp(scale: Float32, *, loc=None, ip=None) -> Float32:
    """
    Exact reciprocal of a power of 2 by exponent flip: bits -> (254 << 23) - bits.
    """
    rcp = llvm.inline_asm(
        T.f32(), [Float32(scale).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 b, c;\n\t"
        "mov.b32 b, $1;\n\t"
        "mov.b32 c, 0x7F000000;\n\t"
        "sub.u32 c, c, b;\n\t"
        "mov.b32 $0, c;\n\t"
        "}",
        "=f,f", has_side_effects=False, is_align_stack=False, loc=loc, ip=ip,
    )
    return Float32(rcp)
