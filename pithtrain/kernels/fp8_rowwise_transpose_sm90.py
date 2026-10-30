"""
CuTe-DSL (SM90) rowwise FP8 quantization of x and of x.T from one read of x, for every input
fp8_quant_grad accepts: bf16, fp16 or fp32, any shape (empty included) and any strides.

One CTA (256 threads) per 128x128 tile; each thread owns an 8x8 block. The transposed tile goes
through bank-swizzled smem so every warp writes whole 128-byte rows of x.T, where a store straight
from registers would scatter each warp instruction over 16 rows. Routes, chosen on the host: a view
with unit row stride runs as its transpose with the outputs swapped; rows with unit column stride
whose 16-byte column groups are all aligned and in or out of their row as a whole are read in place
with 16-byte loads and a runtime row stride; every other view (a non-unit column stride, misaligned
or odd-strided rows, N not a multiple of 16 bytes) is gathered into smem by the CTA first, each
warp reading 64 consecutive columns of a row per instruction pair (coalesced when the column stride
is 1). fp16 and fp32 are Constexpr branches of the same kernel (the amax in the input dtype, the
scale math in fp32).
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import lds_v4, max2, prmt, st_b32, st_v2, st_v4
from pithtrain.kernels.fp8_scale_layout import mn_major_scale

__all__ = ["fp8_quant_grad_sm90"]

TILE, THREADS = 128, 256

# The SM90 E8M0 scale of the recipe, 2^ceil(log2(max(amax, 1e-4) * fp32(1 / 448))) at least 2^-126,
# from the fp32 amax bits: 1e-4 -> 0x38D1B717, 1/448 -> 0x3B124925.
SCALE_PTX = """{
.reg .f32 a, s;
.reg .b32 b, m;
.reg .pred p;
mov.b32 a, $1;
max.f32 a, a, 0f38D1B717;
mul.rn.f32 s, a, 0f3B124925;
mov.b32 b, s;
and.b32 m, b, 0x007FFFFF;
and.b32 b, b, 0x7F800000;
setp.ne.b32 p, m, 0;
@p add.s32 b, b, 0x00800000;
max.s32 b, b, 0x00800000;
mov.b32 $0, b;
}"""

# Rowwise quantize: 4 bf16x2 words (8 consecutive columns) times one reciprocal scale -> 8 fp8
# bytes in two words. bf16 -> f32 is a 16-bit shift; the multiply by a power of 2 is exact and
# cvt.rn.satfinite rounds to nearest even with saturation, as the recipe.
QUANT_ROW_PTX = """{
.reg .f32 a0, a1, a2, a3, a4, a5, a6, a7, r;
.reg .b16 h0, h1, h2, h3;
mov.b32 r, $6;
shl.b32 $0, $2, 16; mov.b32 a0, $0; and.b32 $0, $2, 0xFFFF0000; mov.b32 a1, $0;
shl.b32 $0, $3, 16; mov.b32 a2, $0; and.b32 $0, $3, 0xFFFF0000; mov.b32 a3, $0;
shl.b32 $0, $4, 16; mov.b32 a4, $0; and.b32 $0, $4, 0xFFFF0000; mov.b32 a5, $0;
shl.b32 $0, $5, 16; mov.b32 a6, $0; and.b32 $0, $5, 0xFFFF0000; mov.b32 a7, $0;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
mul.rn.f32 a4, a4, r; mul.rn.f32 a5, a5, r; mul.rn.f32 a6, a6, r; mul.rn.f32 a7, a7, r;
cvt.rn.satfinite.e4m3x2.f32 h0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 h1, a3, a2;
cvt.rn.satfinite.e4m3x2.f32 h2, a5, a4;
cvt.rn.satfinite.e4m3x2.f32 h3, a7, a6;
mov.b32 $0, {h0, h1};
mov.b32 $1, {h2, h3};
}"""

# Column quantize: one half (EXTRACT = low or high bf16) of 4 words (4 consecutive rows of one
# column) -> 4 fp8 bytes in one word.
QUANT_COL_PTX = """{
.reg .f32 a0, a1, a2, a3, r;
.reg .b16 h0, h1;
mov.b32 r, $5;
EXTRACT $0, $1; mov.b32 a0, $0;
EXTRACT $0, $2; mov.b32 a1, $0;
EXTRACT $0, $3; mov.b32 a2, $0;
EXTRACT $0, $4; mov.b32 a3, $0;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
cvt.rn.satfinite.e4m3x2.f32 h0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 h1, a3, a2;
mov.b32 $0, {h0, h1};
}"""
QUANT_COL_EXTRACT = ("shl.b32 {}, {}, 16;", "and.b32 {}, {}, 0xFFFF0000;")

# The same two quantizations for fp16 words (cvt.f32.f16 widens each half exactly) and for fp32
# input (8 words per row, one per column; no widening).
QUANT_ROW_F16_PTX = """{
.reg .f32 a0, a1, a2, a3, a4, a5, a6, a7, r;
.reg .b16 l0, h0, l1, h1, l2, h2, l3, h3;
mov.b32 r, $6;
mov.b32 {l0, h0}, $2; mov.b32 {l1, h1}, $3; mov.b32 {l2, h2}, $4; mov.b32 {l3, h3}, $5;
cvt.f32.f16 a0, l0; cvt.f32.f16 a1, h0; cvt.f32.f16 a2, l1; cvt.f32.f16 a3, h1;
cvt.f32.f16 a4, l2; cvt.f32.f16 a5, h2; cvt.f32.f16 a6, l3; cvt.f32.f16 a7, h3;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
mul.rn.f32 a4, a4, r; mul.rn.f32 a5, a5, r; mul.rn.f32 a6, a6, r; mul.rn.f32 a7, a7, r;
cvt.rn.satfinite.e4m3x2.f32 l0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 l1, a3, a2;
cvt.rn.satfinite.e4m3x2.f32 l2, a5, a4;
cvt.rn.satfinite.e4m3x2.f32 l3, a7, a6;
mov.b32 $0, {l0, l1};
mov.b32 $1, {l2, l3};
}"""

QUANT_ROW_F32_PTX = """{
.reg .f32 a0, a1, a2, a3, a4, a5, a6, a7, r;
.reg .b16 q0, q1, q2, q3;
mov.b32 r, $10;
mov.b32 a0, $2; mov.b32 a1, $3; mov.b32 a2, $4; mov.b32 a3, $5;
mov.b32 a4, $6; mov.b32 a5, $7; mov.b32 a6, $8; mov.b32 a7, $9;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
mul.rn.f32 a4, a4, r; mul.rn.f32 a5, a5, r; mul.rn.f32 a6, a6, r; mul.rn.f32 a7, a7, r;
cvt.rn.satfinite.e4m3x2.f32 q0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 q1, a3, a2;
cvt.rn.satfinite.e4m3x2.f32 q2, a5, a4;
cvt.rn.satfinite.e4m3x2.f32 q3, a7, a6;
mov.b32 $0, {q0, q1};
mov.b32 $1, {q2, q3};
}"""

# HALF is l (low fp16 of each word) or h (high)
QUANT_COL_F16_PTX = """{
.reg .f32 a0, a1, a2, a3, r;
.reg .b16 l0, h0, l1, h1, l2, h2, l3, h3;
mov.b32 r, $5;
mov.b32 {l0, h0}, $1; mov.b32 {l1, h1}, $2; mov.b32 {l2, h2}, $3; mov.b32 {l3, h3}, $4;
cvt.f32.f16 a0, HALF0; cvt.f32.f16 a1, HALF1; cvt.f32.f16 a2, HALF2; cvt.f32.f16 a3, HALF3;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
cvt.rn.satfinite.e4m3x2.f32 l0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 l1, a3, a2;
mov.b32 $0, {l0, l1};
}"""

QUANT_COL_F32_PTX = """{
.reg .f32 a0, a1, a2, a3, r;
.reg .b16 q0, q1;
mov.b32 r, $5;
mov.b32 a0, $1; mov.b32 a1, $2; mov.b32 a2, $3; mov.b32 a3, $4;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
cvt.rn.satfinite.e4m3x2.f32 q0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 q1, a3, a2;
mov.b32 $0, {q0, q1};
}"""

# One fp16 half of a word (HALF = l or h), widened exactly to fp32 bits.
F16_TO_F32_PTX = """{
.reg .b16 l, h;
.reg .f32 f;
mov.b32 {l, h}, $1;
cvt.f32.f16 f, HALF;
mov.b32 $0, f;
}"""

LD_V4_NC_PTX = """{
.reg .pred p;
setp.ne.b32 p, $5, 0;
mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;
@p ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];
}"""

LD_B32_PTX = """{
.reg .pred p;
setp.ne.b32 p, $2, 0;
mov.b32 $0, 0;
@p ld.global.nc.b32 $0, [$1];
}"""

LD_U16_PTX = """{
.reg .pred p;
.reg .b16 h;
setp.ne.b32 p, $2, 0;
mov.b16 h, 0;
@p ld.global.nc.b16 h, [$1];
cvt.u32.u16 $0, h;
}"""

ST_B8_PTX = """{
.reg .pred p;
setp.ne.b32 p, $2, 0;
@p st.global.b8 [$0], $1;
}"""

# 8 fp8 bytes (w0 = bytes 0-3, w1 = bytes 4-7) to an address of any alignment: the widest store
# the runtime alignment allows (8/4/2/1 bytes), mutually exclusive predicates.
ST8_ANY_PTX = """{
.reg .pred p, p8, p4, p2, p1;
.reg .b32 al, t;
setp.ne.b32 p, $3, 0;
cvt.u32.u64 al, $0;
and.b32 t, al, 7;
setp.eq.and.b32 p8, t, 0, p;
setp.eq.and.b32 p4, t, 4, p;
and.b32 t, al, 3;
setp.eq.and.b32 p2, t, 2, p;
and.b32 t, al, 1;
setp.eq.and.b32 p1, t, 1, p;
@p8 st.global.v2.b32 [$0], {$1, $2};
@p4 st.global.b32 [$0], $1;
@p4 st.global.b32 [$0+4], $2;
@p2 st.global.b16 [$0], $1;
@p2 st.global.b16 [$0+4], $2;
@p1 st.global.b8 [$0], $1;
@p1 st.global.b8 [$0+4], $2;
shr.b32 t, $1, 8;
@p1 st.global.b8 [$0+1], t;
shr.b32 t, $1, 16;
@p1 st.global.b8 [$0+2], t;
@p2 st.global.b16 [$0+2], t;
shr.b32 t, $1, 24;
@p1 st.global.b8 [$0+3], t;
shr.b32 t, $2, 8;
@p1 st.global.b8 [$0+5], t;
shr.b32 t, $2, 16;
@p1 st.global.b8 [$0+6], t;
@p2 st.global.b16 [$0+6], t;
shr.b32 t, $2, 24;
@p1 st.global.b8 [$0+7], t;
}"""


def ir_value(value):
    return value.ir_value()


def asm_u32(ptx: str, constraints: str, args: list) -> Uint32:
    return Uint32(llvm.inline_asm(T.i32(), args, ptx, constraints, has_side_effects=False, is_align_stack=False))


def asm_words(ptx: str, constraints: str, args: list, num_words: int) -> list:
    packed = llvm.inline_asm(llvm.StructType.get_literal([T.i32()] * num_words), args, ptx, constraints, has_side_effects=False, is_align_stack=False)
    return [Uint32(llvm.extractvalue(T.i32(), packed, [i])) for i in range(num_words)]


def asm_store(ptx: str, constraints: str, args: list) -> None:
    llvm.inline_asm(None, args, ptx, constraints, has_side_effects=True, is_align_stack=False)


def e8m0_scale_bits(amax_bits: Uint32) -> Uint32:
    return asm_u32(SCALE_PTX, "=r,r", [ir_value(amax_bits)])


def quantize_column(words: list, rcp: Uint32, half: int, kind: str) -> Uint32:
    """
    4 consecutive rows of one column -> 4 fp8 bytes. words hold the column's values: one half
    (half = 0 low, 1 high) of each 16-bit word pair, or the fp32 values themselves.
    """
    if kind == "f32":
        return asm_u32(QUANT_COL_F32_PTX, "=r,r,r,r,r,r", [ir_value(word) for word in words] + [ir_value(rcp)])
    if kind == "f16":
        ptx = QUANT_COL_F16_PTX
        for j in range(4):
            ptx = ptx.replace(f"HALF{j}", f"{'lh'[half]}{j}")
        return asm_u32(ptx, "=r,r,r,r,r,r", [ir_value(word) for word in words] + [ir_value(rcp)])
    extract = QUANT_COL_EXTRACT[half].format("$0", "{}")
    ptx = QUANT_COL_PTX
    for j in range(4):
        ptx = ptx.replace(f"EXTRACT $0, ${j + 1};", extract.format(f"${j + 1}"))
    return asm_u32(ptx, "=&r,r,r,r,r,r", [ir_value(word) for word in words] + [ir_value(rcp)])


def quantize_row(words: list, rcp: Uint32, kind: str) -> list:
    """
    8 consecutive columns of one row (4 16-bit pair words or 8 fp32 words) -> 8 fp8 bytes in two
    words.
    """
    if kind == "f32":
        return asm_words(QUANT_ROW_F32_PTX, "=r,=r,r,r,r,r,r,r,r,r,r", [ir_value(word) for word in words] + [ir_value(rcp)], 2)
    if kind == "f16":
        return asm_words(QUANT_ROW_F16_PTX, "=r,=r,r,r,r,r,r", [ir_value(word) for word in words] + [ir_value(rcp)], 2)
    return asm_words(QUANT_ROW_PTX, "=&r,=&r,r,r,r,r,r", [ir_value(word) for word in words] + [ir_value(rcp)], 2)


def max_words(first: Uint32, second: Uint32, kind: str) -> Uint32:
    """
    NaN-ignoring max of |x| words in the input dtype: two bf16 / fp16 lanes or one fp32.
    """
    if kind == "bf16":
        return max2(first, second)
    return asm_u32("max.f16x2 $0, $1, $2;" if kind == "f16" else "max.f32 $0, $1, $2;", "=r,r,r", [ir_value(first), ir_value(second)])


def widen(word: Uint32, half: int, kind: str) -> Uint32:
    """
    fp32 bits of one element of a word: half (0 low, 1 high) of a bf16 / fp16 pair, or the fp32
    word.
    """
    if kind == "f32":
        return word
    if kind == "f16":
        return asm_u32(F16_TO_F32_PTX.replace("HALF", "lh"[half]), "=r,r", [ir_value(word)])
    return word << 16 if half == 0 else word & Uint32(0xFFFF0000)


def st_b8(addr: Int64, word: Uint32, pred: Int32) -> None:
    asm_store(ST_B8_PTX, "l,r,r", [ir_value(addr), ir_value(word), ir_value(pred)])


def st8_any(addr: Int64, word0: Uint32, word1: Uint32, pred: Int32) -> None:
    asm_store(ST8_ANY_PTX, "l,r,r,r", [ir_value(addr), ir_value(word0), ir_value(word1), ir_value(pred)])


def pred_i32(condition) -> Int32:
    return Int32(cutlass.select_(condition, Int32(1), Int32(0)))


class SubBlock:
    """
    Where one thread's 8 x 8 sub-block sits in the CTA's 128 x 128 tile and in x, with the bounds
    predicates of its rows and columns.
    """

    def __init__(self, tidx: Int32, pid_row: Int32, pid_col: Int32, M: Int32, N: Int32) -> None:
        self.warp = tidx // 32
        self.lane = tidx % 32
        self.col_group_idx = self.lane % 16
        self.row_group_idx = self.warp * 2 + self.lane // 16
        self.pid_row, self.pid_col = pid_row, pid_col
        self.row0 = pid_row * TILE + self.row_group_idx * 8
        self.col0 = pid_col * TILE + self.col_group_idx * 8
        self.rows_ok = [pred_i32(self.row0 + i < M) for i in range(8)]
        self.cols_ok = [pred_i32(self.col0 + j < N) for j in range(8)]
        self.full_cols_ok = pred_i32(self.col0 + 8 <= N)


def load_rows(x_base: Int64, S0: Int64, sub_block: SubBlock, kind: str) -> list:
    """
    The sub-block as 32-bit words, zero out of bounds: rows[i][k] = columns (2k, 2k+1) of row i for
    bf16 / fp16, column k of row i for fp32. Rows of x are S0 elements apart; x, S0 and N are such
    that every 16-byte column group is 16-byte aligned and in or out of its row as a whole.
    """
    row0, col0, rows_ok, cols_ok = sub_block.row0, sub_block.col0, sub_block.rows_ok, sub_block.cols_ok
    rows = []
    for i in range(8):
        elem_idx = Int64(row0 + i) * S0 + Int64(col0)
        if kind == "f32":
            words = []
            for half in range(2):
                load_ok = rows_ok[i] & cols_ok[4 * half]
                byte_offset = cutlass.select_(load_ok != 0, elem_idx * 4 + 16 * half, Int64(0))
                words += asm_words(LD_V4_NC_PTX, "=r,=r,=r,=r,l,r", [ir_value(x_base + byte_offset), ir_value(load_ok)], 4)
            rows.append(words)
        else:
            load_ok = rows_ok[i] & cols_ok[0]
            byte_offset = cutlass.select_(load_ok != 0, elem_idx * 2, Int64(0))
            rows.append(asm_words(LD_V4_NC_PTX, "=r,=r,=r,=r,l,r", [ir_value(x_base + byte_offset), ir_value(load_ok)], 4))
    return rows


def gather_rows(x_base: Int64, S0: Int64, S1: Int64, sub_block: SubBlock, kind: str, s_raw: Int32, M: Int32, N: Int32) -> list:
    """
    load_rows for any other x (a non-unit column stride, or rows that are not 16-byte aligned): the
    CTA first copies its tile into s_raw (row-major, 128 elements per row), each warp instruction
    reading 64 consecutive columns of one row (two per lane), then each thread reads its sub-block
    back with 16-byte shared loads.
    """
    elem_bytes = 4 if kind == "f32" else 2
    lane, warp = sub_block.lane, sub_block.warp
    tile_row0, tile_col0 = sub_block.pid_row * TILE, sub_block.pid_col * TILE
    col_offsets, cols_ok = [], []
    for col in range(4):
        tile_col = tile_col0 + 64 * (col // 2) + 2 * lane + col % 2
        col_offsets.append(Int64(tile_col) * S1)
        cols_ok.append(pred_i32(tile_col < N))
    for r in range(16):
        tile_row = warp * 16 + r
        row_offset = Int64(tile_row0 + tile_row) * S0
        row_ok = pred_i32(tile_row0 + tile_row < M)
        for pass_idx in range(2):
            pair = []
            for half in range(2):
                load_ok = row_ok & cols_ok[2 * pass_idx + half]
                byte_offset = cutlass.select_(load_ok != 0, (row_offset + col_offsets[2 * pass_idx + half]) * elem_bytes, Int64(0))
                pair.append(asm_u32(LD_B32_PTX if kind == "f32" else LD_U16_PTX, "=r,l,r", [ir_value(x_base + byte_offset), ir_value(load_ok)]))
            s_addr = s_raw + (tile_row * TILE + 64 * pass_idx + 2 * lane) * elem_bytes
            if kind == "f32":
                asm_store("st.shared.v2.b32 [$0], {$1, $2};", "r,r,r", [ir_value(s_addr), ir_value(pair[0]), ir_value(pair[1])])
            else:
                asm_store("st.shared.b32 [$0], $1;", "r,r", [ir_value(s_addr), ir_value(pair[0] | (pair[1] << 16))])
    cute.arch.barrier()
    rows = []
    for i in range(8):
        s_addr = s_raw + ((sub_block.row_group_idx * 8 + i) * TILE + sub_block.col_group_idx * 8) * elem_bytes
        rows.append(list(lds_v4(s_addr)) + (list(lds_v4(s_addr + 16)) if kind == "f32" else []))
    return rows


def share_col_amax(abs_rows: list, sub_block: SubBlock, s_red: Int32, kind: str) -> None:
    """
    Amax of the thread's 8 columns over its warp's 16 rows (4 or 8 words), stored to s_red for the
    cross-warp reduction.
    """
    num_words = len(abs_rows[0])
    col_amax = []
    for k in range(num_words):
        acc_amax = abs_rows[0][k]
        for i in range(1, 8):
            acc_amax = max_words(acc_amax, abs_rows[i][k], kind)
        other = Uint32(cute.arch.shuffle_sync_bfly(Int32(acc_amax), offset=16))
        col_amax.append(max_words(acc_amax, other, kind))
    for quad in range(num_words // 4):
        asm_store("{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.shared.v4.b32 [$0], {$1, $2, $3, $4};\n}", "r,r,r,r,r,r", [ir_value(s_red + (sub_block.warp * 16 * num_words + sub_block.col_group_idx * num_words + 4 * quad) * 4)] + [ir_value(amax) for amax in col_amax[4 * quad : 4 * quad + 4]] + [ir_value(pred_i32(sub_block.lane < 16))])


def reduce_row_amax(abs_rows: list, kind: str) -> list:
    """
    fp32 bits of the amax of each of the sub-block's 8 rows over the 16 lanes sharing them; 16-bit
    inputs reduce two rows packed per word.
    """
    if kind == "f32":
        row_amax = []
        for i in range(8):
            acc_amax = abs_rows[i][0]
            for k in range(1, 8):
                acc_amax = max_words(acc_amax, abs_rows[i][k], kind)
            row_amax.append(acc_amax)
        for step in range(4):
            for i in range(8):
                other = Uint32(cute.arch.shuffle_sync_bfly(Int32(row_amax[i]), offset=1 << step))
                row_amax[i] = max_words(row_amax[i], other, kind)
        return row_amax
    row_amax = [max_words(max_words(abs_rows[i][0], abs_rows[i][1], kind), max_words(abs_rows[i][2], abs_rows[i][3], kind), kind) for i in range(8)]
    row_pair_amax = []
    for pair in range(4):
        amax_lo, amax_hi = prmt(row_amax[2 * pair], row_amax[2 * pair + 1], 0x5410), prmt(row_amax[2 * pair], row_amax[2 * pair + 1], 0x7632)
        row_pair_amax.append(max_words(amax_lo, amax_hi, kind))
    for step in range(4):
        for pair in range(4):
            other = Uint32(cute.arch.shuffle_sync_bfly(Int32(row_pair_amax[pair]), offset=1 << step))
            row_pair_amax[pair] = max_words(row_pair_amax[pair], other, kind)
    return [widen(row_pair_amax[i // 2], i % 2, kind) for i in range(8)]


def store_row_scales(row_amax: list, sub_block: SubBlock, tok_scale_base: Int64, M64: Int64) -> list:
    row_scale_bits = [e8m0_scale_bits(amax) for amax in row_amax]
    # MN-major scales: scale[r, j] at j * align(rows, 4) + r.
    m_pad = (M64 + 3) // 4 * 4
    for i in range(8):
        store_ok = sub_block.rows_ok[i] & pred_i32(sub_block.col_group_idx == i)
        st_b32(tok_scale_base + (Int64(sub_block.pid_col) * m_pad + Int64(sub_block.row0 + i)) * 4, row_scale_bits[i], store_ok)
    return row_scale_bits


def store_tok_rows(rows: list, row_scale_bits: list, sub_block: SubBlock, tok_base: Int64, N64: Int64, kind: str, is_tok_aligned: bool) -> None:
    row0, col0, rows_ok, cols_ok, full_cols_ok = sub_block.row0, sub_block.col0, sub_block.rows_ok, sub_block.cols_ok, sub_block.full_cols_ok
    for i in range(8):
        # 0x7F000000 - bits is the exact reciprocal of a power-of-2 scale
        quant_lo, quant_hi = quantize_row(rows[i], Uint32(0x7F000000) - row_scale_bits[i], kind)
        elem_idx = Int64(row0 + i) * N64 + Int64(col0)
        if const_expr(is_tok_aligned):
            st_v2(tok_base + elem_idx, quant_lo, quant_hi, rows_ok[i] & cols_ok[0])
        else:
            st8_any(tok_base + elem_idx, quant_lo, quant_hi, rows_ok[i] & full_cols_ok)
            part_ok = rows_ok[i] & (full_cols_ok ^ 1)
            for byte_idx in range(7):
                word = quant_lo if byte_idx < 4 else quant_hi
                st_b8(tok_base + elem_idx + byte_idx, word >> (8 * (byte_idx % 4)), part_ok & cols_ok[byte_idx])


def reduce_col_amax(sub_block: SubBlock, s_red: Int32, kind: str) -> list:
    """
    fp32 bits of the amax of each of the thread's 8 columns over the whole tile, from the 8 warps'
    partial amaxes in s_red.
    """
    num_words = 8 if kind == "f32" else 4
    warp_amax = [sum((list(lds_v4(s_red + (warp * 16 * num_words + sub_block.col_group_idx * num_words + 4 * quad) * 4)) for quad in range(num_words // 4)), []) for warp in range(8)]
    col_amax = []
    for k in range(num_words):
        acc_amax = warp_amax[0][k]
        for warp in range(1, 8):
            acc_amax = max_words(acc_amax, warp_amax[warp][k], kind)
        col_amax.append(acc_amax)
    return col_amax if kind == "f32" else [widen(col_amax[j // 2], j % 2, kind) for j in range(8)]


def store_col_scales(col_amax: list, sub_block: SubBlock, t_scale_base: Int64, N64: Int64) -> list:
    col_scale_bits = [e8m0_scale_bits(amax) for amax in col_amax]
    n_pad = (N64 + 3) // 4 * 4
    for j in range(8):
        store_ok = sub_block.cols_ok[j] & pred_i32(sub_block.row_group_idx == j)
        st_b32(t_scale_base + (Int64(sub_block.pid_row) * n_pad + Int64(sub_block.col0 + j)) * 4, col_scale_bits[j], store_ok)
    return col_scale_bits


def stage_transposed(rows: list, col_scale_bits: list, sub_block: SubBlock, s_tile: Int32, kind: str) -> None:
    """
    Quantize the sub-block's columns and write them into the tile of x.T in s_tile (row = one column
    of x, 128 B). The 8-byte chunk index is swizzled by the column group so the 32 lanes of a warp
    hit distinct banks.
    """
    col_group_idx, row_group_idx = sub_block.col_group_idx, sub_block.row_group_idx
    for col in range(8):
        word_idx, half = (col, 0) if kind == "f32" else (col // 2, col % 2)
        rcp = Uint32(0x7F000000) - col_scale_bits[col]
        quant_lo, quant_hi = quantize_column([rows[i][word_idx] for i in range(4)], rcp, half, kind), quantize_column([rows[i][word_idx] for i in range(4, 8)], rcp, half, kind)
        asm_store("st.shared.v2.b32 [$0], {$1, $2};", "r,r,r", [ir_value(s_tile + (col_group_idx * 8 + col) * TILE + (row_group_idx ^ col_group_idx) * 8), ir_value(quant_lo), ir_value(quant_hi)])


def store_transposed(sub_block: SubBlock, s_tile: Int32, t_base: Int64, M64: Int64, N64: Int64, m_align: int) -> None:
    """
    Store the staged tile of x.T into out_t; each warp writes 4 contiguous 128-byte rows per pass
    (16 B per lane).
    """
    lane, warp = sub_block.lane, sub_block.warp
    chunk_idx = lane % 8
    x_row_start = Int64(sub_block.pid_row * TILE + chunk_idx * 16)
    for k in range(4):
        tile_row_idx = k * 32 + warp * 4 + lane // 8
        swizzle = (tile_row_idx // 8) % 16
        words = lds_v4(s_tile + tile_row_idx * TILE + (((2 * chunk_idx) ^ swizzle) & 14) * 8)
        is_odd_swizzle = (swizzle % 2) != 0
        words_lo, words_hi = [cutlass.select_(is_odd_swizzle, words[2], words[0]), cutlass.select_(is_odd_swizzle, words[3], words[1])], [cutlass.select_(is_odd_swizzle, words[0], words[2]), cutlass.select_(is_odd_swizzle, words[1], words[3])]
        x_col = sub_block.pid_col * TILE + tile_row_idx
        col_ok = pred_i32(x_col < N64)
        out_addr = t_base + Int64(x_col) * M64 + x_row_start
        if const_expr(m_align == 16):
            st_v4(out_addr, words_lo + words_hi, col_ok & pred_i32(x_row_start < M64))
        else:
            for half, (word0, word1) in enumerate((words_lo, words_hi)):
                half_row_start = x_row_start + 8 * half
                full_ok = col_ok & pred_i32(half_row_start + 8 <= M64)
                if const_expr(m_align == 8):
                    st_v2(out_addr + 8 * half, word0, word1, full_ok)
                else:
                    st8_any(out_addr + 8 * half, word0, word1, full_ok)
                    part_ok = col_ok & pred_i32(half_row_start < M64) & (full_ok ^ 1)
                    for byte_idx in range(7):
                        word = word0 if byte_idx < 4 else word1
                        st_b8(out_addr + 8 * half + byte_idx, word >> (8 * (byte_idx % 4)), part_ok & pred_i32(half_row_start + byte_idx < M64))


@cute.kernel
def rowwise_transpose_kernel(mX: cute.Tensor, mOutTok: cute.Tensor, mScaleTok: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, kind: cutlass.Constexpr[str], is_tok_aligned: cutlass.Constexpr[bool], m_align: cutlass.Constexpr[int], is_gather: cutlass.Constexpr[bool]) -> None:
    M, N = mX.shape[0], mX.shape[1]
    tidx, _, _ = cute.arch.thread_idx()
    pid_col, pid_row, _ = cute.arch.block_idx()
    sub_block = SubBlock(tidx, pid_row, pid_col, M, N)
    smem = cutlass.memory.SmemAllocator()
    num_words = 8 if const_expr(kind == "f32") else 4
    # 2 KB (4 KB for fp32) for the cross-warp column amax, 16 KB for the transposed tile
    s_red = smem.allocate_array(Uint32, 8 * 16 * num_words, byte_alignment=16).toint()
    M64, N64 = Int64(M), Int64(N)
    if const_expr(is_gather):
        # the raw tile is dead once every thread holds its sub-block (before the barrier after
        # share_col_amax), so the transposed tile reuses it
        s_tile = smem.allocate_array(cutlass.Uint8, TILE * TILE * (4 if const_expr(kind == "f32") else 2), byte_alignment=128).toint()
        rows = gather_rows(mX.iterator.toint(), Int64(mX.stride[0]), Int64(mX.stride[1]), sub_block, kind, s_tile, M, N)
    else:
        s_tile = smem.allocate_array(cutlass.Uint8, TILE * TILE, byte_alignment=128).toint()
        rows = load_rows(mX.iterator.toint(), Int64(mX.stride[0]), sub_block, kind)
    tok_base, tok_scale_base, t_base, t_scale_base = mOutTok.iterator.toint(), mScaleTok.iterator.toint(), mOutT.iterator.toint(), mScaleT.iterator.toint()
    abs_mask = Uint32(0x7FFFFFFF) if const_expr(kind == "f32") else Uint32(0x7FFF7FFF)
    abs_rows = [[rows[i][k] & abs_mask for k in range(num_words)] for i in range(8)]
    share_col_amax(abs_rows, sub_block, s_red, kind)
    cute.arch.barrier()
    # needs only registers and warp shuffles; the barrier above is for reduce_col_amax's s_red reads
    row_amax = reduce_row_amax(abs_rows, kind)
    row_scale_bits = store_row_scales(row_amax, sub_block, tok_scale_base, M64)
    store_tok_rows(rows, row_scale_bits, sub_block, tok_base, N64, kind, is_tok_aligned)
    col_amax = reduce_col_amax(sub_block, s_red, kind)
    col_scale_bits = store_col_scales(col_amax, sub_block, t_scale_base, N64)
    stage_transposed(rows, col_scale_bits, sub_block, s_tile, kind)
    cute.arch.barrier()
    store_transposed(sub_block, s_tile, t_base, M64, N64, m_align)


@cute.jit
def launch(mX: cute.Tensor, mOutTok: cute.Tensor, mScaleTok: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, stream, kind: cutlass.Constexpr[str], is_tok_aligned: cutlass.Constexpr[bool], m_align: cutlass.Constexpr[int], is_gather: cutlass.Constexpr[bool]) -> None:
    M, N = mX.shape[0], mX.shape[1]
    rowwise_transpose_kernel(mX, mOutTok, mScaleTok, mOutT, mScaleT, kind, is_tok_aligned, m_align, is_gather).launch(grid=[(N + (TILE - 1)) // TILE, (M + (TILE - 1)) // TILE, 1], block=[THREADS, 1, 1], stream=stream)


INPUT_KINDS = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
COMPILED: dict[tuple[str, bool, int, bool], Callable] = {}


def compile_kernel(kind: str, is_tok_aligned: bool, m_align: int, is_gather: bool) -> Callable:
    M, N, num_tok_scale_cols, num_t_scale_cols = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    dtype = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16, "f32": cutlass.Float32}[kind]
    x = fake(dtype, (M, N), (cute.sym_int64(), cute.sym_int64() if is_gather else 1), dtype.width // 8 if is_gather else 16)
    out_tok, tok_scale = fake(cutlass.Float8E4M3FN, (M, N), (cute.sym_int64(), 1)), fake(cutlass.Float32, (M, num_tok_scale_cols), (1, cute.sym_int64()))
    out_t, t_scale = fake(cutlass.Float8E4M3FN, (N, M), (cute.sym_int64(), 1)), fake(cutlass.Float32, (N, num_t_scale_cols), (1, cute.sym_int64()))
    return cute.compile(launch, x, out_tok, tok_scale, out_t, t_scale, fake_stream(), kind, is_tok_aligned, m_align, is_gather, options="--enable-tvm-ffi")


def quantize(x: torch.Tensor, kind: str, M: int, N: int, S0: int, S1: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The kernel on x, an (M, N) view with strides (S0, S1): 16-byte loads in place when S1 == 1 and
    every 16-byte column group of every row is aligned and in or out of the row as a whole, else the
    gather.
    """
    device = x.device
    out_tok, out_t = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device)
    scale_tok, scale_t = mn_major_scale(M, (N + 127) >> 7, device), mn_major_scale(N, (M + 127) >> 7, device)
    vec = 4 if kind == "f32" else 8
    is_gather = S1 != 1 or x.data_ptr() % 16 != 0 or S0 % vec != 0 or N % vec != 0
    # 8-byte fp8 stores of out_tok need N % 8 == 0
    key = (kind, N % 8 == 0, 16 if M % 16 == 0 else 8 if M % 8 == 0 else 1, is_gather)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_kernel(*key)
    fn(x, out_tok, scale_tok, out_t, scale_t)
    return out_tok, scale_tok, out_t, scale_t


def fp8_quant_grad_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    fp8_quant_grad on SM90: x is 2-D (M, N) bf16, fp16 or fp32 (else TypeError), any M and N (an
    empty x returns empty outputs without a launch), any strides (row-strided, transposed, both
    strides non-unit, misaligned base pointers, size-1 and stride-0 dims); not modified. Returns
    (x_fp8 e4m3 (M, N) contiguous, x_scale fp32 (M, ceil(N / 128)), xt_fp8 e4m3 (N, M) contiguous,
    xt_scale fp32 (N, ceil(M / 128))), both scales MN-major (strides (1, align(rows, 4))), with the
    bytes and scale values of the recipe in the picker's docstring.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    kind = INPUT_KINDS.get(x.dtype)
    if kind is None:
        raise TypeError(f"fp8_quant_grad takes bfloat16, float16 or float32 input, got {x.dtype}")
    (M, N), (S0, S1) = x.shape, x.stride()
    if M == 0 or N == 0:
        device = x.device
        return torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), mn_major_scale(M, (N + 127) >> 7, device), torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device), mn_major_scale(N, (M + 127) >> 7, device)
    if M == 1 or N == 1:
        # the stride of a size-1 dim addresses no element: read it as the dense one
        S0, S1 = S0 if M > 1 else N, S1 if N > 1 else 1
        x = x.as_strided((M, N), (S0, S1))
    if S1 == 1 or S0 != 1:
        return quantize(x, kind, M, N, S0, S1)
    # unit row stride (a transposed view): x.T has unit column stride, and its outputs are x's swapped
    out_t, scale_t, out_tok, scale_tok = quantize(x.T, kind, N, M, S1, 1)
    return out_tok, scale_tok, out_t, scale_t
