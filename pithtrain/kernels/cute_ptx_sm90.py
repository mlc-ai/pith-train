"""
Inline-PTX helpers shared by the SM90 CuTe-DSL kernels, each a dsl_user_op emitting one short PTX
sequence. Two calling conventions: asm(res_type, args, ptx, constraints), whose stores return
nothing, and asm_args_last(res_type, ptx, constraints, args), whose stores return a dummy register.
An asm_args_last helper that shares a name with an asm helper carries a suffix: _args_last when only
the emission differs, otherwise one naming the difference (ld_b32_coherent is a coherent ld.global,
not .nc; lds_b32_pred is predicated; sts_b32_nopred is not).
"""

import struct

import cutlass
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = ["abs2", "asm", "asm_args_last", "atom_add_u64", "bf16_hi", "bf16_lo", "bf16_sel", "bf16x2_to_f32", "cp_async16", "f32_as_u32", "f32_hex", "f32x2_to_bf16x2", "fence_gpu", "lanemask_lt", "ld_b32", "ld_b32_coherent", "ld_relaxed", "ld_v4", "lds_b32", "lds_b32_pred", "lds_v4", "match_any", "max2", "max_u16x2", "pack_bf16", "prmt", "prmt_args_last", "red_add_shared", "red_or_shared", "selp", "selp_args_last", "st_b32", "st_b64", "st_f32", "st_v2", "st_v2_args_last", "st_v4", "st_v4_args_last", "sts_b32", "sts_b32_nopred", "sts_v4", "to_ir", "u32_as_f32", "unpack"]


def f32_hex(value: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', value))[0]:08X}"


def asm(res_type, args, ptx: str, constraints: str, side_effects: bool = False):
    return llvm.inline_asm(res_type, args, ptx, constraints, has_side_effects=side_effects, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_v4(addr: Int64, pred: Int32, *, loc=None, ip=None) -> tuple:
    """
    16-byte global load; zeros where pred == 0. A plain ld.global, which is faster here than the
    .nc read-only path.
    """
    struct_type = llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()])
    words = asm(
        struct_type, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t"
        "mov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ld_b32(addr: Int64, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        asm(
            T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip), Int32(other).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, $3;\n\t"
            "@p ld.global.nc.b32 $0, [$1];\n\t}",
            "=r,l,r,r",
        )
    )


@dsl_user_op
def st_v2(addr: Int64, word_lo: Uint32, word_hi: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(word_lo).ir_value(loc=loc, ip=ip), Uint32(word_hi).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $3, 0;\n\t@p st.global.v2.b32 [$0], {$1, $2};\n\t}", "l,r,r,r", side_effects=True)


@dsl_user_op
def st_b32(addr: Int64, value: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}", "l,r,r", side_effects=True)


@dsl_user_op
def lds_b32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", side_effects=True))


@dsl_user_op
def sts_b32(addr: Int32, value: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.shared.b32 [$0], $1;\n\t}", "r,r,r", side_effects=True)


@dsl_user_op
def max_u16x2(first: Uint32, second: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    Equals the bf16 max for non-negative (abs) values.
    """
    return Uint32(asm(T.i32(), [Uint32(first).ir_value(loc=loc, ip=ip), Uint32(second).ir_value(loc=loc, ip=ip)], "max.u16x2 $0, $1, $2;", "=r,r,r"))


@dsl_user_op
def prmt(word_lo: Uint32, word_hi: Uint32, selector: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [Uint32(word_lo).ir_value(loc=loc, ip=ip), Uint32(word_hi).ir_value(loc=loc, ip=ip)], f"prmt.b32 $0, $1, $2, 0x{selector:04X};", "=r,r,r"))


@dsl_user_op
def selp(pred: Int32, if_true: Uint32, if_false: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [Int32(pred).ir_value(loc=loc, ip=ip), Uint32(if_true).ir_value(loc=loc, ip=ip), Uint32(if_false).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $1, 0;\n\tselp.b32 $0, $2, $3, p;\n\t}", "=r,r,r,r"))


@dsl_user_op
def match_any(value: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [Int32(value).ir_value(loc=loc, ip=ip)], "match.any.sync.b32 $0, $1, -1;", "=r,r"))


@dsl_user_op
def lanemask_lt(*, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [], "mov.u32 $0, %lanemask_lt;", "=r"))


@dsl_user_op
def atom_add_u64(addr: Int64, value: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(asm(T.i64(), [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)], "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;", "=l,l,l", side_effects=True))


@dsl_user_op
def red_add_shared(addr: Int32, value: Int32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p red.shared.add.u32 [$0], $1;\n\t}", "r,r,r", side_effects=True)


@dsl_user_op
def ld_relaxed(addr: Int64, pred: Int32, *, loc=None, ip=None) -> Int32:
    """
    Reads L2, never a stale L1 line; 0 where not pred.
    """
    return Int32(
        asm(
            T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, 0;\n\t"
            "@p ld.relaxed.gpu.global.b32 $0, [$1];\n\t}",
            "=r,l,r", side_effects=True,
        )
    )


@dsl_user_op
def fence_gpu(*, loc=None, ip=None) -> None:
    asm(None, [], "fence.acq_rel.gpu;", "", side_effects=True)


@dsl_user_op
def st_b64(addr: Int64, value: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.global.b64 [$0], $1;\n\t}", "l,l,r", side_effects=True)


@dsl_user_op
def red_or_shared(addr: Int32, value: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p red.shared.or.b32 [$0], $1;\n\t}", "r,r,r", side_effects=True)


@dsl_user_op
def bf16x2_to_f32(x: Uint32, *, loc=None, ip=None) -> tuple:
    x_lo = asm(T.f32(), [Uint32(x).ir_value(loc=loc, ip=ip)], "shl.b32 $0, $1, 16;", "=f,r")
    x_hi = asm(T.f32(), [Uint32(x).ir_value(loc=loc, ip=ip)], "and.b32 $0, $1, 0xFFFF0000;", "=f,r")
    return cutlass.Float32(x_lo), cutlass.Float32(x_hi)


@dsl_user_op
def f32x2_to_bf16x2(x_lo, x_hi, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [cutlass.Float32(x_hi).ir_value(loc=loc, ip=ip), cutlass.Float32(x_lo).ir_value(loc=loc, ip=ip)], "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f"))


@dsl_user_op
def st_v4(addr: Int64, words: tuple, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Uint32(word).ir_value(loc=loc, ip=ip) for word in words] + [Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n\t}", "l,r,r,r,r,r", side_effects=True)


def asm_args_last(res_type, ptx: str, constraints: str, args: list, side_effects: bool = False):
    return asm(res_type, args, ptx, constraints, side_effects)


def to_ir(x, value_type, loc=None, ip=None):
    return value_type(x).ir_value(loc=loc, ip=ip)


def unpack(struct_value, num_values: int, value_type, loc=None, ip=None) -> tuple:
    element_type = T.f32() if value_type is Float32 else T.i32()
    return tuple(value_type(llvm.extractvalue(element_type, struct_value, [i], loc=loc, ip=ip)) for i in range(num_values))


@dsl_user_op
def ld_b32_coherent(addr: Int64, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, $3;\n@p ld.global.b32 $0, [$1];\n}", "=r,l,r,r", [to_ir(addr, Int64, loc, ip), to_ir(pred, Int32, loc, ip), to_ir(other, Int32, loc, ip)]))


@dsl_user_op
def st_v2_args_last(addr: Int64, word_lo: Uint32, word_hi: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm_args_last(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\n@p st.global.v2.b32 [$1], {$2, $3};\n"
        "mov.b32 $0, 0;\n}",
        "=r,l,r,r,r", [to_ir(addr, Int64, loc, ip)] + [to_ir(word, Uint32, loc, ip) for word in (word_lo, word_hi)] + [to_ir(pred, Int32, loc, ip)], side_effects=True,
    )


@dsl_user_op
def st_v4_args_last(addr: Int64, words: tuple, pred: Int32, *, loc=None, ip=None) -> None:
    asm_args_last(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $6, 0;\n@p st.global.v4.b32 [$1], {$2, $3, $4, $5};\n"
        "mov.b32 $0, 0;\n}",
        "=r,l,r,r,r,r,r", [to_ir(addr, Int64, loc, ip)] + [to_ir(word, Uint32, loc, ip) for word in words] + [to_ir(pred, Int32, loc, ip)], side_effects=True,
    )


@dsl_user_op
def st_f32(addr: Int64, value: Float32, pred: Int32, *, loc=None, ip=None) -> None:
    asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\n@p st.global.f32 [$1], $2;\nmov.b32 $0, 0;\n}", "=r,l,f,r", [to_ir(addr, Int64, loc, ip), to_ir(value, Float32, loc, ip), to_ir(pred, Int32, loc, ip)], side_effects=True)


@dsl_user_op
def sts_b32_nopred(addr: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    asm_args_last(T.i32(), "st.shared.b32 [$1], $2;\nmov.b32 $0, 0;", "=r,r,r", [to_ir(addr, Int32, loc, ip), to_ir(value, Uint32, loc, ip)], side_effects=True)


@dsl_user_op
def sts_v4(addr: Int32, words: tuple, *, loc=None, ip=None) -> None:
    asm_args_last(T.i32(), "st.shared.v4.b32 [$1], {$2, $3, $4, $5};\nmov.b32 $0, 0;", "=r,r,r,r,r,r", [to_ir(addr, Int32, loc, ip)] + [to_ir(word, Uint32, loc, ip) for word in words], side_effects=True)


@dsl_user_op
def lds_v4(addr: Int32, *, loc=None, ip=None) -> tuple:
    struct_type = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    words = asm_args_last(struct_type, "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", [to_ir(addr, Int32, loc, ip)], side_effects=True)  # side_effects keeps the load behind the barrier
    return unpack(words, 4, Uint32, loc, ip)


@dsl_user_op
def cp_async16(dst: Int32, src: Int64, src_size: Int32, *, loc=None, ip=None) -> None:
    """
    Zero-fills the 16 destination bytes when src_size is 0.
    """
    asm_args_last(T.i32(), "cp.async.cg.shared.global [$1], [$2], 16, $3;\nmov.b32 $0, 0;", "=r,r,l,r", [to_ir(dst, Int32, loc, ip), to_ir(src, Int64, loc, ip), to_ir(src_size, Int32, loc, ip)], side_effects=True)


@dsl_user_op
def lds_b32_pred(addr: Int32, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, $3;\n@p ld.shared.b32 $0, [$1];\n}", "=r,r,r,r", [to_ir(addr, Int32, loc, ip), to_ir(pred, Int32, loc, ip), to_ir(other, Int32, loc, ip)], side_effects=True))


@dsl_user_op
def selp_args_last(pred: Int32, if_true: Uint32, if_false: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}", "=r,r,r,r", [to_ir(pred, Int32, loc, ip), to_ir(if_true, Uint32, loc, ip), to_ir(if_false, Uint32, loc, ip)]))


@dsl_user_op
def bf16_sel(pred_hi: Int32, x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        asm_args_last(
            T.f32(),
            "{\n.reg .pred p;\n.reg .b32 a, b;\nsetp.ne.b32 p, $1, 0;\n"
            "and.b32 a, $2, 0xFFFF0000;\nshl.b32 b, $2, 16;\nselp.b32 $0, a, b, p;\n}",
            "=f,r,r", [to_ir(pred_hi, Int32, loc, ip), to_ir(x, Uint32, loc, ip)],
        )
    )


@dsl_user_op
def f32_as_u32(x: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm_args_last(T.i32(), "mov.b32 $0, $1;", "=r,f", [to_ir(x, Float32, loc, ip)]))


@dsl_user_op
def u32_as_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "mov.b32 $0, $1;", "=f,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def bf16_lo(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "shl.b32 $0, $1, 16;", "=f,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def bf16_hi(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "and.b32 $0, $1, 0xFFFF0000;", "=f,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def pack_bf16(x_lo: Float32, x_hi: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Round two fp32 to BF16 (RN, as Triton's .to(tl.bfloat16)) and pack {x_lo, x_hi}.
    """
    return Uint32(asm_args_last(T.i32(), "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", [to_ir(x_hi, Float32, loc, ip), to_ir(x_lo, Float32, loc, ip)]))


@dsl_user_op
def abs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm_args_last(T.i32(), "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def max2(first: Uint32, second: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    NaN-ignoring, like Triton's max.f32.
    """
    return Uint32(asm_args_last(T.i32(), "max.bf16x2 $0, $1, $2;", "=r,r,r", [to_ir(first, Uint32, loc, ip), to_ir(second, Uint32, loc, ip)]))


@dsl_user_op
def prmt_args_last(word_lo: Uint32, word_hi: Uint32, selector: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm_args_last(T.i32(), f"prmt.b32 $0, $1, $2, {selector:#x};", "=r,r,r", [to_ir(word_lo, Uint32, loc, ip), to_ir(word_hi, Uint32, loc, ip)]))
