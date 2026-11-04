"""
CuTe-DSL (SM90) rowwise FP8 quantization of x and of x.T from one read of x, bit-identical to the
Triton kernel in fp8_rowwise_transpose_sm100.

One CTA (256 threads) per 128x128 tile; each thread owns an 8x8 block. The transposed tile goes
through bank-swizzled smem so every warp writes whole 128-byte rows of x.T, where a store straight
from registers would scatter each warp instruction over 16 rows.
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
from pithtrain.kernels.fp8_rowwise_transpose_sm100 import fp8_quant_grad_sm100
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, to_mn_major

__all__ = ["fp8_quant_grad_sm90", "supports_rowwise_transpose"]

TILE, THREADS = 128, 256

# fp32 bit patterns of the constants compute_fp8_scale uses (Triton materializes the Python
# floats as fp32): 1e-4 -> 0x38D1B717, 1/448 -> 0x3B124925.
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
# cvt.rn.satfinite matches Triton's .to(tl.float8e4nv).
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

LD_V4_NC_PTX = """{
.reg .pred p;
setp.ne.b32 p, $5, 0;
mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;
@p ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];
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

# 8 contiguous bf16 at an address that is only 2-byte aligned: the widest access the runtime
# alignment allows (16/8/4/2 bytes). The four paths are mutually exclusive predicates.
LD8_ANY_PTX = """{
.reg .pred p, p16, p8, p4, p2;
.reg .b32 al, t;
.reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
setp.ne.b32 p, $5, 0;
cvt.u32.u64 al, $4;
and.b32 t, al, 15;
setp.eq.and.b32 p16, t, 0, p;
setp.eq.and.b32 p8, t, 8, p;
and.b32 t, al, 7;
setp.eq.and.b32 p4, t, 4, p;
and.b32 t, al, 3;
setp.eq.and.b32 p2, t, 2, p;
mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;
@p16 ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];
@p8 ld.global.nc.v2.b32 {$0, $1}, [$4];
@p8 ld.global.nc.v2.b32 {$2, $3}, [$4+8];
@p4 ld.global.nc.b32 $0, [$4];
@p4 ld.global.nc.b32 $1, [$4+4];
@p4 ld.global.nc.b32 $2, [$4+8];
@p4 ld.global.nc.b32 $3, [$4+12];
@p2 ld.global.nc.b16 h0, [$4];
@p2 ld.global.nc.b16 h1, [$4+2];
@p2 ld.global.nc.b16 h2, [$4+4];
@p2 ld.global.nc.b16 h3, [$4+6];
@p2 ld.global.nc.b16 h4, [$4+8];
@p2 ld.global.nc.b16 h5, [$4+10];
@p2 ld.global.nc.b16 h6, [$4+12];
@p2 ld.global.nc.b16 h7, [$4+14];
@p2 mov.b32 $0, {h0, h1};
@p2 mov.b32 $1, {h2, h3};
@p2 mov.b32 $2, {h4, h5};
@p2 mov.b32 $3, {h6, h7};
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


def quantize_column(words: list, rcp: Uint32, half: int) -> Uint32:
    extract = QUANT_COL_EXTRACT[half].format("$0", "{}")
    ptx = QUANT_COL_PTX
    for j in range(4):
        ptx = ptx.replace(f"EXTRACT $0, ${j + 1};", extract.format(f"${j + 1}"))
    return asm_u32(ptx, "=&r,r,r,r,r,r", [ir_value(word) for word in words] + [ir_value(rcp)])


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


def load_rows(x_base: Int64, N64: Int64, sub_block: SubBlock, is_row_aligned: bool) -> list:
    """
    The sub-block as bf16x2 words: rows[i][k] = columns (2k, 2k+1) of row i, zero out of bounds.
    """
    row0, col0, rows_ok, cols_ok, full_cols_ok = sub_block.row0, sub_block.col0, sub_block.rows_ok, sub_block.cols_ok, sub_block.full_cols_ok
    rows = []
    if is_row_aligned:
        for i in range(8):
            load_ok = rows_ok[i] & cols_ok[0]
            byte_offset = cutlass.select_(load_ok != 0, (Int64(row0 + i) * N64 + Int64(col0)) * 2, Int64(0))
            rows.append(asm_words(LD_V4_NC_PTX, "=r,=r,=r,=r,l,r", [ir_value(x_base + byte_offset), ir_value(load_ok)], 4))
    else:
        # Rows of x are not 16-byte aligned: full 8-column groups take the widest access the
        # runtime alignment allows; the partial group at the N edge loads element by element.
        for i in range(8):
            full_ok, part_ok = rows_ok[i] & full_cols_ok, rows_ok[i] & (full_cols_ok ^ 1)
            elem_idx = Int64(row0 + i) * N64 + Int64(col0)
            words = asm_words(LD8_ANY_PTX, "=r,=r,=r,=r,l,r", [ir_value(x_base + cutlass.select_(full_ok != 0, elem_idx * 2, Int64(0))), ir_value(full_ok)], 4)
            for k in range(4):
                halves = []
                for half in range(2):
                    elem_ok = part_ok & cols_ok[2 * k + half]
                    byte_offset = cutlass.select_(elem_ok != 0, (elem_idx + (2 * k + half)) * 2, Int64(0))
                    halves.append(asm_u32(LD_U16_PTX, "=r,l,r", [ir_value(x_base + byte_offset), ir_value(elem_ok)]))
                words[k] = words[k] | halves[0] | (halves[1] << 16)
            rows.append(words)
    return rows


def share_col_amax(abs_rows: list, sub_block: SubBlock, s_red: Int32) -> None:
    """
    Amax of the thread's 8 columns over its warp's 16 rows, stored to s_red for
    the cross-warp reduction.
    """
    col_amax = []
    for k in range(4):
        acc_amax = abs_rows[0][k]
        for i in range(1, 8):
            acc_amax = max2(acc_amax, abs_rows[i][k])
        other = Uint32(cute.arch.shuffle_sync_bfly(Int32(acc_amax), offset=16))
        col_amax.append(max2(acc_amax, other))
    asm_store("{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.shared.v4.b32 [$0], {$1, $2, $3, $4};\n}", "r,r,r,r,r,r", [ir_value(s_red + (sub_block.warp * 64 + sub_block.col_group_idx * 4) * 4)] + [ir_value(amax) for amax in col_amax] + [ir_value(pred_i32(sub_block.lane < 16))])


def reduce_row_amax(abs_rows: list) -> list:
    """
    Amax of the sub-block's rows over the 16 lanes sharing them, two rows packed per word.
    """
    row_amax = [max2(max2(abs_rows[i][0], abs_rows[i][1]), max2(abs_rows[i][2], abs_rows[i][3])) for i in range(8)]
    row_pair_amax = []
    for pair in range(4):
        amax_lo, amax_hi = prmt(row_amax[2 * pair], row_amax[2 * pair + 1], 0x5410), prmt(row_amax[2 * pair], row_amax[2 * pair + 1], 0x7632)
        row_pair_amax.append(max2(amax_lo, amax_hi))
    for step in range(4):
        for pair in range(4):
            other = Uint32(cute.arch.shuffle_sync_bfly(Int32(row_pair_amax[pair]), offset=1 << step))
            row_pair_amax[pair] = max2(row_pair_amax[pair], other)
    return row_pair_amax


def store_row_scales(row_pair_amax: list, sub_block: SubBlock, tok_scale_base: Int64, M64: Int64) -> list:
    row_scale_bits = []
    for pair in range(4):
        row_scale_bits.append(e8m0_scale_bits(row_pair_amax[pair] << 16))
        row_scale_bits.append(e8m0_scale_bits(row_pair_amax[pair] & Uint32(0xFFFF0000)))
    # MN-major scales: scale[r, j] at j * align(rows, 4) + r.
    m_pad = (M64 + 3) // 4 * 4
    for i in range(8):
        store_ok = sub_block.rows_ok[i] & pred_i32(sub_block.col_group_idx == i)
        st_b32(tok_scale_base + (Int64(sub_block.pid_col) * m_pad + Int64(sub_block.row0 + i)) * 4, row_scale_bits[i], store_ok)
    return row_scale_bits


def store_tok_rows(rows: list, row_scale_bits: list, sub_block: SubBlock, tok_base: Int64, N64: Int64, is_row_aligned: bool) -> None:
    row0, col0, rows_ok, cols_ok, full_cols_ok = sub_block.row0, sub_block.col0, sub_block.rows_ok, sub_block.cols_ok, sub_block.full_cols_ok
    for i in range(8):
        # 0x7F000000 - bits is the exact reciprocal of a power-of-2 scale
        quant_lo, quant_hi = asm_words(QUANT_ROW_PTX, "=&r,=&r,r,r,r,r,r", [ir_value(word) for word in rows[i]] + [ir_value(Uint32(0x7F000000) - row_scale_bits[i])], 2)
        elem_idx = Int64(row0 + i) * N64 + Int64(col0)
        if const_expr(is_row_aligned):
            st_v2(tok_base + elem_idx, quant_lo, quant_hi, rows_ok[i] & cols_ok[0])
        else:
            st8_any(tok_base + elem_idx, quant_lo, quant_hi, rows_ok[i] & full_cols_ok)
            part_ok = rows_ok[i] & (full_cols_ok ^ 1)
            for byte_idx in range(7):
                word = quant_lo if byte_idx < 4 else quant_hi
                st_b8(tok_base + elem_idx + byte_idx, word >> (8 * (byte_idx % 4)), part_ok & cols_ok[byte_idx])


def reduce_col_amax(sub_block: SubBlock, s_red: Int32) -> list:
    """
    Amax of the thread's 8 columns over the whole tile, from the 8 warps' partial amaxes in s_red.
    """
    warp_amax = [lds_v4(s_red + (warp * 64 + sub_block.col_group_idx * 4) * 4) for warp in range(8)]
    col_amax = []
    for k in range(4):
        acc_amax = warp_amax[0][k]
        for warp in range(1, 8):
            acc_amax = max2(acc_amax, warp_amax[warp][k])
        col_amax.append(acc_amax)
    return col_amax


def store_col_scales(col_amax: list, sub_block: SubBlock, t_scale_base: Int64, N64: Int64) -> list:
    col_scale_bits = []
    for k in range(4):
        col_scale_bits.append(e8m0_scale_bits(col_amax[k] << 16))
        col_scale_bits.append(e8m0_scale_bits(col_amax[k] & Uint32(0xFFFF0000)))
    n_pad = (N64 + 3) // 4 * 4
    for j in range(8):
        store_ok = sub_block.cols_ok[j] & pred_i32(sub_block.row_group_idx == j)
        st_b32(t_scale_base + (Int64(sub_block.pid_row) * n_pad + Int64(sub_block.col0 + j)) * 4, col_scale_bits[j], store_ok)
    return col_scale_bits


def stage_transposed(rows: list, col_scale_bits: list, sub_block: SubBlock, s_tile: Int32) -> None:
    """
    Quantize the sub-block's columns and write them into the tile of x.T in s_tile (row = one column
    of x, 128 B). The 8-byte chunk index is swizzled by the column group so the 32 lanes of a warp
    hit distinct banks.
    """
    col_group_idx, row_group_idx = sub_block.col_group_idx, sub_block.row_group_idx
    for col in range(8):
        word_idx = col // 2
        rcp = Uint32(0x7F000000) - col_scale_bits[col]
        quant_lo, quant_hi = quantize_column([rows[i][word_idx] for i in range(4)], rcp, col % 2), quantize_column([rows[i][word_idx] for i in range(4, 8)], rcp, col % 2)
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
def rowwise_transpose_kernel(mX: cute.Tensor, mOutTok: cute.Tensor, mScaleTok: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, is_row_aligned: cutlass.Constexpr[bool], m_align: cutlass.Constexpr[int]) -> None:
    M, N = mX.shape[0], mX.shape[1]
    tidx, _, _ = cute.arch.thread_idx()
    pid_col, pid_row, _ = cute.arch.block_idx()
    sub_block = SubBlock(tidx, pid_row, pid_col, M, N)
    smem = cutlass.memory.SmemAllocator()
    # 2 KB for the cross-warp column amax, 16 KB for the transposed tile
    s_red = smem.allocate_array(Uint32, 8 * 16 * 4, byte_alignment=16).toint()
    s_tile = smem.allocate_array(cutlass.Uint8, TILE * TILE, byte_alignment=128).toint()
    M64, N64 = Int64(M), Int64(N)
    rows = load_rows(mX.iterator.toint(), N64, sub_block, is_row_aligned)
    tok_base, tok_scale_base, t_base, t_scale_base = mOutTok.iterator.toint(), mScaleTok.iterator.toint(), mOutT.iterator.toint(), mScaleT.iterator.toint()
    abs_rows = [[rows[i][k] & Uint32(0x7FFF7FFF) for k in range(4)] for i in range(8)]
    share_col_amax(abs_rows, sub_block, s_red)
    cute.arch.barrier()
    # needs only registers and warp shuffles; the barrier above is for reduce_col_amax's s_red reads
    row_pair_amax = reduce_row_amax(abs_rows)
    row_scale_bits = store_row_scales(row_pair_amax, sub_block, tok_scale_base, M64)
    store_tok_rows(rows, row_scale_bits, sub_block, tok_base, N64, is_row_aligned)
    col_amax = reduce_col_amax(sub_block, s_red)
    col_scale_bits = store_col_scales(col_amax, sub_block, t_scale_base, N64)
    stage_transposed(rows, col_scale_bits, sub_block, s_tile)
    cute.arch.barrier()
    store_transposed(sub_block, s_tile, t_base, M64, N64, m_align)


@cute.jit
def launch(mX: cute.Tensor, mOutTok: cute.Tensor, mScaleTok: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, stream, is_row_aligned: cutlass.Constexpr[bool], m_align: cutlass.Constexpr[int]) -> None:
    M, N = mX.shape[0], mX.shape[1]
    rowwise_transpose_kernel(mX, mOutTok, mScaleTok, mOutT, mScaleT, is_row_aligned, m_align).launch(grid=[(N + (TILE - 1)) // TILE, (M + (TILE - 1)) // TILE, 1], block=[THREADS, 1, 1], stream=stream)


COMPILED: dict[tuple[bool, int], Callable] = {}


def compile_kernel(is_row_aligned: bool, m_align: int) -> Callable:
    M, N, num_tok_scale_cols, num_t_scale_cols = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    x = fake(cutlass.BFloat16, (M, N), (cute.sym_int64(), 1), 16 if is_row_aligned else 2)
    out_tok, tok_scale = fake(cutlass.Float8E4M3FN, (M, N), (cute.sym_int64(), 1)), fake(cutlass.Float32, (M, num_tok_scale_cols), (1, cute.sym_int64()))
    out_t, t_scale = fake(cutlass.Float8E4M3FN, (N, M), (cute.sym_int64(), 1)), fake(cutlass.Float32, (N, num_t_scale_cols), (1, cute.sym_int64()))
    return cute.compile(launch, x, out_tok, tok_scale, out_t, t_scale, fake_stream(), is_row_aligned, m_align, options="--enable-tvm-ffi")


def supports_rowwise_transpose(x: torch.Tensor) -> bool:
    M, N = x.shape
    return x.dtype is torch.bfloat16 and x.is_contiguous() and bool(M) and bool(N)


def fp8_quant_grad_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    if not supports_rowwise_transpose(x):
        out_tok, scale_tok, out_t, scale_t = fp8_quant_grad_sm100(x)
        return out_tok, to_mn_major(scale_tok), out_t, to_mn_major(scale_t)
    device = x.device
    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device)
    scale_tok = mn_major_scale(M, (N + 127) >> 7, device)
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device)
    scale_t = mn_major_scale(N, (M + 127) >> 7, device)
    key = (N % 8 == 0 and x.data_ptr() % 16 == 0, 16 if M % 16 == 0 else 8 if M % 8 == 0 else 1)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_kernel(*key)
    fn(x, out_tok, scale_tok, out_t, scale_t)
    return out_tok, scale_tok, out_t, scale_t
