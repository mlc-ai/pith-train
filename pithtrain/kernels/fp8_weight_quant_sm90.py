"""
CuTe-DSL (SM90) kernels of the fp8_weight_quant feature: the blockwise transposes, the FP8 parameter
all-gather kernels, and one-launch variants of the shard amax and the scales over a fixed list of
tensors. The blockwise transposes take bf16 or fp32 inputs of any shape and strides without a copy.
Every call goes through opaque torch.library ops whose kernels are C++ (OPS_SOURCE, built with the
host compiler at first import), so eager calls and torch.compile regions launch the CuTe kernel the
same way, without re-entering Python.

Numerics follow the SM90 E8M0 recipe (its SM90 E8M0 scale) exactly. The tile amax is an
integer max over the magnitude bits (|x| of finite non-negative floats orders like its bit pattern),
so a NaN orders above inf: NaN inputs are out of contract.
"""

import fcntl
import hashlib
import os
import socket
import subprocess
import sys
import sysconfig
from typing import Optional, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import tvm_ffi
import tvm_ffi.libinfo
from cutlass import Int32, Int64, Uint32, const_expr
from torch.utils import cpp_extension
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import lds_b32, lds_v4, max_u16x2, prmt_args_last, st_v2, st_v4
from pithtrain.kernels.fp8_e8m0_scale_sm90 import AMAX_MIN, FP8_MAX_RCP

__all__ = ["fp8_quant_weight_sm90", "fp8_quant_weight_grouped_sm90", "fp8_shard_amax_sm90", "fp8_shard_cast_sm90", "fp8_scales_sm90", "fp8_pack_transpose_sm90", "fp8_shard_amax_multi_sm90", "fp8_scales_multi_sm90"]

BLOCK = 128
# Below this many tiles the extra barrier of the shared-memory store path costs more than its
# wider stores save.
SMEM_MIN_TILES = 256
NUM_THREADS = 256
NUM_WARPS = NUM_THREADS // 32


def asm(res_type, ptx: str, constraints: str, args: list, side_effects: bool = False):
    return llvm.inline_asm(res_type, [operand.ir_value() for operand in args], ptx, constraints, has_side_effects=side_effects, is_align_stack=False)


def struct_fields(struct, num_fields: int) -> list:
    return [Uint32(llvm.extractvalue(T.i32(), struct, [i])) for i in range(num_fields)]


@dsl_user_op
def ld_v4_ordered(addr: Int64, pred: Int32, *, loc=None, ip=None) -> list:
    """
    Predicated 16-byte global load (zero when pred is 0), issued as side-effecting asm so it stays
    ahead of the stores; pure asm gets sunk to its use.
    """
    result = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r", [Int64(addr), Int32(pred)], side_effects=True,
    )
    return struct_fields(result, 4)


@dsl_user_op
def ld_v2_ordered(addr: Int64, pred: Int32, *, loc=None, ip=None) -> list:
    """
    Predicated 8-byte global load (zero when pred is 0), side-effecting like ld_v4_ordered.
    """
    result = asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\n"
        "@p ld.global.v2.b32 {$0, $1}, [$2];\n}",
        "=r,=r,l,r", [Int64(addr), Int32(pred)], side_effects=True,
    )
    return struct_fields(result, 2)


@dsl_user_op
def ld_scalar_ordered(addr: Int64, pred: Int32, num_bytes: int, *, loc=None, ip=None) -> Uint32:
    """
    Predicated zero-extended 1/2/4-byte global load, side-effecting like ld_v4_ordered.
    """
    ptx_type = {1: "u8", 2: "u16", 4: "b32"}[num_bytes]
    result = asm(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, 0;\n"
        f"@p ld.global.{ptx_type} $0, [$1];\n}}",
        "=r,l,r", [Int64(addr), Int32(pred)], side_effects=True,
    )
    return Uint32(result)


@dsl_user_op
def st_scalar(addr: Int64, value: Uint32, pred: Int32, num_bytes: int, *, loc=None, ip=None) -> None:
    ptx_type = {1: "b8", 4: "b32"}[num_bytes]
    asm(None, f"{{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\n@p st.global.{ptx_type} [$0], $1;\n}}", "l,r,r", [Int64(addr), Uint32(value), Int32(pred)], side_effects=True)


@dsl_user_op
def max_u32(first: Uint32, second: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), "max.u32 $0, $1, $2;", "=r,r,r", [first, second]))


@dsl_user_op
def f32x2_bits_to_bf16x2(lo_bits: Uint32, hi_bits: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    Two fp32 bit patterns -> packed bf16x2 (round to nearest even), lo_bits in bits [15:0].
    """
    return Uint32(asm(T.i32(), "{\n.reg .f32 a, b;\nmov.b32 a, $1;\nmov.b32 b, $2;\ncvt.rn.bf16x2.f32 $0, b, a;\n}", "=r,r,r", [lo_bits, hi_bits]))


@dsl_user_op
def cast4(x0_bits: Uint32, x1_bits: Uint32, x2_bits: Uint32, x3_bits: Uint32, rcp_bits: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    Four fp32 bit patterns times rcp_bits -> four e4m3 bytes, x0_bits in bits [7:0].
    """
    return Uint32(
        asm(
            T.i32(),
            "{\n.reg .f32 a, b, c, d, r;\n.reg .b16 lo, hi;\n"
            "mov.b32 r, $5;\n"
            "mov.b32 a, $1;\nmov.b32 b, $2;\nmov.b32 c, $3;\nmov.b32 d, $4;\n"
            "mul.rn.f32 a, a, r;\nmul.rn.f32 b, b, r;\nmul.rn.f32 c, c, r;\nmul.rn.f32 d, d, r;\n"
            "cvt.rn.satfinite.e4m3x2.f32 lo, b, a;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, d, c;\n"
            "mov.b32 $0, {lo, hi};\n}",
            "=r,r,r,r,r,r", [x0_bits, x1_bits, x2_bits, x3_bits, rcp_bits],
        )
    )


@dsl_user_op
def compute_scale_bits(amax_bits: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    E8M0 scale bits from amax bits, exactly as the recipe's SM90 E8M0 scale.
    """
    return Uint32(
        asm(
            T.i32(),
            "{\n.reg .f32 a;\n.reg .b32 bits, m, e;\n"
            "mov.b32 a, $1;\n"
            f"max.f32 a, a, {AMAX_MIN};\n"
            f"mul.rn.f32 a, a, {FP8_MAX_RCP};\n"
            "mov.b32 bits, a;\n"
            "and.b32 m, bits, 0x007FFFFF;\n"
            "and.b32 e, bits, 0x7F800000;\n"
            "add.u32 m, m, 0x007FFFFF;\n"
            "shr.u32 m, m, 23;\n"
            "shl.b32 m, m, 23;\n"
            "add.u32 e, e, m;\n"
            "max.u32 $0, e, 0x00800000;\n}",  # max(scale, 2^-126) on non-negative bits
            "=r,r", [amax_bits],
        )
    )


@dsl_user_op
def ld_u64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(asm(T.i64(), "ld.global.nc.u64 $0, [$1];", "=l,l", [Int64(addr)]))


@cute.jit
def thread_coords() -> tuple:
    """
    256 threads per 128x128 tile, each owning the 8x8 block at (block_row, block_col) of the tile.
    """
    tidx, _, _ = cute.arch.thread_idx()
    warp, lane = tidx // 32, tidx % 32
    block_row, block_col = (warp // 2) * 32 + (lane // 8) * 8, (warp % 2) * 64 + (lane % 8) * 8
    return tidx, warp, lane, block_row, block_col


@cute.jit
def load_rows(x_base: Int64, row0: Int32, col0: Int32, num_rows: Int32, K: Int32, row_stride: Int64, in_bytes: cutlass.Constexpr[int], round_bf16: cutlass.Constexpr[bool], is_aligned: cutlass.Constexpr[bool]) -> list:
    """
    Load the thread's 8x8 block, rows row0 + i valid when 0 <= row0 + i < num_rows.

    x_base addresses element (0, 0) of a (rows, K) matrix with unit column stride and row_stride
    elements between rows. Returns 8 rows of words: four
    bf16x2 words per row for 2-byte (or bf16-rounded) data, eight fp32 words for 4-byte data, two
    words of four bytes for 1-byte data.
    """
    rows = []
    for i in cutlass.range_constexpr(8):
        row = row0 + i
        row_ok = Int32(row >= 0) & Int32(row < num_rows)
        row_addr = x_base + (Int64(row) * row_stride + Int64(col0)) * in_bytes
        words = []
        if const_expr(is_aligned):
            vec_ok = row_ok & Int32(col0 < K)
            if const_expr(in_bytes == 2):
                words = ld_v4_ordered(row_addr, vec_ok)
            elif const_expr(in_bytes == 4):
                words = ld_v4_ordered(row_addr, vec_ok) + ld_v4_ordered(row_addr + 16, vec_ok)
            else:
                words = ld_v2_ordered(row_addr, vec_ok)
        else:
            elements = []
            for j in cutlass.range_constexpr(8):
                elem_ok = row_ok & Int32(col0 + j < K)
                elements.append(ld_scalar_ordered(row_addr + j * in_bytes, elem_ok, in_bytes))
            if const_expr(in_bytes == 2):
                words = [elements[2 * k] | (elements[2 * k + 1] << 16) for k in range(4)]
            elif const_expr(in_bytes == 4):
                words = elements
            else:
                words = [elements[4 * k] | (elements[4 * k + 1] << 8) | (elements[4 * k + 2] << 16) | (elements[4 * k + 3] << 24) for k in range(2)]
        if const_expr(in_bytes == 4 and round_bf16):
            words = [f32x2_bits_to_bf16x2(words[2 * k], words[2 * k + 1]) for k in range(4)]
        rows.append(words)
    return rows


@cute.jit
def reduce_thread_amax(rows: list, is_packed_bf16: cutlass.Constexpr[bool]) -> Uint32:
    """
    Max magnitude bit pattern (as fp32 bits) over the thread's block.
    """
    acc_amax = Uint32(0)
    if const_expr(is_packed_bf16):
        for i in cutlass.range_constexpr(8):
            for k in cutlass.range_constexpr(4):
                acc_amax = max_u16x2(acc_amax, rows[i][k] & Uint32(0x7FFF7FFF))
        acc_amax = max_u32(acc_amax & Uint32(0xFFFF), acc_amax >> 16) << 16
    else:
        for i in cutlass.range_constexpr(8):
            for k in cutlass.range_constexpr(8):
                acc_amax = max_u32(acc_amax, rows[i][k] & Uint32(0x7FFFFFFF))
    return acc_amax


@cute.jit
def reduce_block_amax(amax_bits: Uint32, warp: Int32, lane: Int32, s_amax: cute.Tensor) -> Uint32:
    """
    CTA-wide max of amax_bits (every thread gets the result).
    """
    amax_bits = Uint32(asm(T.i32(), "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r", [amax_bits], side_effects=True))
    if lane == 0:
        s_amax[warp] = amax_bits
    cute.arch.sync_threads()
    acc_amax = s_amax[0]
    for i in cutlass.range_constexpr(1, NUM_WARPS):
        acc_amax = max_u32(acc_amax, s_amax[i])
    return acc_amax


@cute.jit
def cast_rows(rows: list, rcp_bits: Uint32, is_packed_bf16: cutlass.Constexpr[bool]) -> list:
    """
    Cast 8 rows to e4m3: two words (cols 0-3, cols 4-7) per row.
    """
    quant = []
    for i in cutlass.range_constexpr(8):
        f32_bits = []
        if const_expr(is_packed_bf16):
            for k in cutlass.range_constexpr(4):
                word = rows[i][k]
                f32_bits.append(word << 16)
                f32_bits.append(word & Uint32(0xFFFF0000))
        else:
            f32_bits = rows[i]
        quant.append([cast4(f32_bits[0], f32_bits[1], f32_bits[2], f32_bits[3], rcp_bits), cast4(f32_bits[4], f32_bits[5], f32_bits[6], f32_bits[7], rcp_bits)])
    return quant


@cute.jit
def transpose4(row0_word: Uint32, row1_word: Uint32, row2_word: Uint32, row3_word: Uint32) -> list:
    """
    4x4 byte transpose: word r holds row r; result word c holds column c.
    """
    rows01_cols01 = prmt_args_last(row0_word, row1_word, 0x5140)
    rows01_cols23 = prmt_args_last(row0_word, row1_word, 0x7362)
    rows23_cols01 = prmt_args_last(row2_word, row3_word, 0x5140)
    rows23_cols23 = prmt_args_last(row2_word, row3_word, 0x7362)
    return [prmt_args_last(rows01_cols01, rows23_cols01, 0x5410), prmt_args_last(rows01_cols01, rows23_cols01, 0x7632), prmt_args_last(rows01_cols23, rows23_cols23, 0x5410), prmt_args_last(rows01_cols23, rows23_cols23, 0x7632)]


@cute.jit
def transpose8(rows: list) -> list:
    """
    8x8 byte transpose of [row][lo, hi] words into [col][rows 0-3, rows 4-7] words.
    """
    top_left = transpose4(rows[0][0], rows[1][0], rows[2][0], rows[3][0])
    bottom_left = transpose4(rows[4][0], rows[5][0], rows[6][0], rows[7][0])
    top_right = transpose4(rows[0][1], rows[1][1], rows[2][1], rows[3][1])
    bottom_right = transpose4(rows[4][1], rows[5][1], rows[6][1], rows[7][1])
    return [[top_left[j], bottom_left[j]] for j in range(4)] + [[top_right[j], bottom_right[j]] for j in range(4)]


@cute.jit
def store_rows(out_base: Int64, quant: list, row0: Int32, col0: Int32, num_rows: Int32, K: Int32, is_aligned: cutlass.Constexpr[bool]) -> None:
    """
    Store [row][lo, hi] fp8 words to a row-major (rows, K) byte matrix at out_base, rows row0 + i
    valid when 0 <= row0 + i < num_rows.
    """
    for i in cutlass.range_constexpr(8):
        row = row0 + i
        row_ok = Int32(row >= 0) & Int32(row < num_rows)
        row_addr = out_base + Int64(row) * Int64(K) + Int64(col0)
        if const_expr(is_aligned):
            st_v2(row_addr, quant[i][0], quant[i][1], row_ok & Int32(col0 < K))
        else:
            for j in cutlass.range_constexpr(8):
                byte = (quant[i][j // 4] >> (8 * (j % 4))) & Uint32(0xFF)
                st_scalar(row_addr + j, byte, row_ok & Int32(col0 + j < K), 1)


# A 128x128 byte tile staged in shared memory as 128 rows of eight 16-byte chunks. The chunk index
# is XOR-swizzled by s(row) = ((row >> 3) & 1) << 2 | ((row >> 3) & 7) >> 1, which makes both the
# row-block writes (8 bytes per lane, rows 8 apart) and the transposed-block writes (rows 8 apart
# the other way) hit the minimum two wavefronts; readback reads whole rows, one wavefront each.
TILE_BYTES = BLOCK * BLOCK


@cute.jit
def swizzle(row: Int32, chunk: Int32) -> Int32:
    """
    Byte offset of 16-byte chunk chunk of tile row row in the staged tile.
    """
    row_block = (row >> 3) & 7
    return row * BLOCK + ((chunk ^ (((row_block & 1) << 2) | (row_block >> 1))) << 4)


@cute.jit
def write_smem_block(s_tile: Int32, quant: list, tile_row0: Int32, tile_col0: Int32) -> None:
    """
    Write [row][lo, hi] words of the 8x8 block at tile (tile_row0, tile_col0) into the staged tile.
    """
    for i in cutlass.range_constexpr(8):
        asm(None, "st.shared.v2.b32 [$0], {$1, $2};", "r,r,r", [Int32(s_tile + swizzle(tile_row0 + i, tile_col0 >> 4) + (tile_col0 & 8)), Uint32(quant[i][0]), Uint32(quant[i][1])], side_effects=True)


@cute.jit
def store_smem_tile(s_tile: Int32, out_tile_addr: Int64, num_rows: Int32, num_cols: Int32, row_stride: Int32, num_threads: cutlass.Constexpr[int] = NUM_THREADS, width: cutlass.Constexpr[int] = 16) -> None:
    """
    Store the staged tile to the row-major byte matrix whose tile row 0, col 0 is at out_tile_addr,
    with width-byte stores (16, 4 or 1; consecutive lanes on consecutive units of a row, row_stride
    and the matrix base a multiple of width); tile row r is valid when r < num_rows, unit c when
    width * c < num_cols.
    """
    tidx, _, _ = cute.arch.thread_idx()
    units_per_row = BLOCK // width
    for i in cutlass.range_constexpr(TILE_BYTES // width // num_threads):
        unit_idx = i * num_threads + tidx
        row, unit = unit_idx // units_per_row, unit_idx % units_per_row
        unit_ok = Int32(row < num_rows) & Int32(unit * width < num_cols)
        out_addr = out_tile_addr + Int64(row) * Int64(row_stride) + Int64(unit * width)
        if const_expr(width == 16):
            st_v4(out_addr, lds_v4(s_tile + swizzle(row, unit)), unit_ok)
        elif const_expr(width == 4):
            st_scalar(out_addr, lds_b32(s_tile + swizzle(row, unit >> 2) + ((unit & 3) << 2)), unit_ok, 4)
        else:
            st_scalar(out_addr, lds_u8(s_tile + swizzle(row, unit >> 4) + (unit & 15)), unit_ok, 1)


@dsl_user_op
def lds_u8(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), "ld.shared.u8 $0, [$1];", "=r,r", [Int32(addr)], side_effects=True))


@dsl_user_op
def sts_scalar(addr: Int32, value: Uint32, num_bytes: int, *, loc=None, ip=None) -> None:
    ptx_type = {2: "b16", 4: "b32"}[num_bytes]
    asm(None, f"st.shared.{ptx_type} [$0], $1;", "r,r", [Int32(addr), Uint32(value)], side_effects=True)


# Element loads a thread keeps in flight before writing them to shared memory.
STAGE_BATCH = 16


@cute.jit
def stage_tile(x_base: Int64, s_tile: Int32, tile_row_start: Int32, tile_col_start: Int32, M: Int32, K: Int32, row_stride: Int64, col_stride: Int64, in_bytes: cutlass.Constexpr[int]) -> None:
    """
    Copy the input tile into shared memory as 128 rows of 128 elements (zero outside the matrix),
    with element loads at any strides: thread t loads column t % 128 of every other row, so
    consecutive lanes read consecutive columns.
    """
    tidx, _, _ = cute.arch.thread_idx()
    col, row_phase = tidx % BLOCK, tidx // BLOCK
    rows_per_pass = NUM_THREADS // BLOCK
    col_ok = Int32(tile_col_start + col < K)
    col_addr = x_base + Int64(tile_col_start + col) * col_stride * in_bytes
    for batch in cutlass.range_constexpr(BLOCK // rows_per_pass // STAGE_BATCH):
        values = []
        for i in cutlass.range_constexpr(STAGE_BATCH):
            row = (batch * STAGE_BATCH + i) * rows_per_pass + row_phase
            values.append(ld_scalar_ordered(col_addr + Int64(tile_row_start + row) * row_stride * in_bytes, col_ok & Int32(tile_row_start + row < M), in_bytes))
        for i in cutlass.range_constexpr(STAGE_BATCH):
            row = (batch * STAGE_BATCH + i) * rows_per_pass + row_phase
            sts_scalar(s_tile + (row * BLOCK + col) * in_bytes, values[i], in_bytes)


@cute.jit
def load_rows_smem(s_tile: Int32, block_row: Int32, block_col: Int32, in_bytes: cutlass.Constexpr[int]) -> list:
    """
    The thread's 8x8 block from the staged input tile, in the words of load_rows.
    """
    rows = []
    for i in cutlass.range_constexpr(8):
        row_addr = s_tile + ((block_row + i) * BLOCK + block_col) * in_bytes
        words = list(lds_v4(row_addr))
        if const_expr(in_bytes == 4):
            words = words + list(lds_v4(row_addr + 16))
        rows.append(words)
    return rows


class BlockwiseTranspose:
    """
    One CTA per 128x128 tile of a (G, M, K) (is_batched) or (M, K) input with any element strides.
    load picks how the tile is read: "vec" loads each thread's 8x8 block with 16-byte loads (unit
    column stride, K % 8 == 0, 16-byte aligned base, row and group strides), "staged" copies the
    tile into shared memory with element loads (anything else). store picks how the FP8 tiles are
    written: "regs" stores each thread's 8-byte rows (M, K % 8 == 0), "staged" stages both tiles in
    shared memory and stores them with coalesced out_width / out_t_width byte stores.
    """

    def __init__(self, in_bytes: int, load: str, store: str, out_width: int, out_t_width: int, is_batched: bool) -> None:
        self.in_bytes, self.load, self.store, self.is_batched = in_bytes, load, store, is_batched
        self.out_width, self.out_t_width = out_width, out_t_width
        self.smem_words = max(TILE_BYTES * in_bytes if load == "staged" else 0, 2 * TILE_BYTES if store == "staged" else 0) // 4

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, mScale: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, group_stride: Int64, row_stride: Int64, col_stride: Int64, stream: cuda.CUstream):
        if const_expr(self.is_batched):
            G, M, K = mOut.shape[0], mOut.shape[1], mOut.shape[2]
        else:
            G, M, K = 1, mOut.shape[0], mOut.shape[1]
        num_row_tiles, num_col_tiles = (M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        self.kernel(mX.iterator.toint(), mOut.iterator.toint(), mScale.iterator.toint(), mOutT.iterator.toint(), mScaleT.iterator.toint(), Int32(M), Int32(K), group_stride, row_stride, col_stride, Int32(num_row_tiles), Int32(num_col_tiles)).launch(grid=[num_row_tiles, num_col_tiles, G], block=[NUM_THREADS, 1, 1], smem=self.smem_words * 4 + 128, stream=stream)

    @cute.kernel
    def kernel(self, x_base: Int64, out_base: Int64, scale_base: Int64, out_t_base: Int64, scale_t_base: Int64, M: Int32, K: Int32, group_stride: Int64, row_stride: Int64, col_stride: Int64, num_row_tiles: Int32, num_col_tiles: Int32):
        bx, by, bz = cute.arch.block_idx()
        tidx, warp, lane, block_row, block_col = thread_coords()
        smem = cutlass.memory.SmemAllocator()
        s_amax = smem.allocate_tensor(Uint32, cute.make_layout(NUM_WARPS), byte_alignment=16)

        tile_row_start, tile_col_start = bx * BLOCK, by * BLOCK
        x_group_base = x_base + Int64(bz) * group_stride * self.in_bytes
        s_tile = Int32(0)
        if const_expr(self.smem_words > 0):
            s_tiles = smem.allocate_tensor(Uint32, cute.make_layout(self.smem_words), byte_alignment=128)
            s_tile = s_tiles.iterator.toint()
        if const_expr(self.load == "staged"):
            stage_tile(x_group_base, s_tile, tile_row_start, tile_col_start, M, K, row_stride, col_stride, self.in_bytes)
            cute.arch.sync_threads()
            rows = load_rows_smem(s_tile, block_row, block_col, self.in_bytes)
        else:
            rows = load_rows(x_group_base, tile_row_start + block_row, tile_col_start + block_col, M, K, row_stride, self.in_bytes, False, True)

        is_packed_bf16 = const_expr(self.in_bytes == 2)
        # The barrier inside also orders the staged input reads before the staged output writes.
        amax_bits = reduce_block_amax(reduce_thread_amax(rows, is_packed_bf16), warp, lane, s_amax)
        scale_bits = compute_scale_bits(amax_bits)
        quant = cast_rows(rows, Uint32(0x7F000000) - scale_bits, is_packed_bf16)
        # out_t is (K, M): the thread's column j is out_t row col0 + j, its rows are columns.
        quant_t = transpose8(quant)
        group_elems = Int64(M) * Int64(K)
        out_group_base, out_t_group_base = out_base + Int64(bz) * group_elems, out_t_base + Int64(bz) * group_elems
        if const_expr(self.store == "staged"):
            write_smem_block(s_tile, quant, block_row, block_col)
            write_smem_block(s_tile + TILE_BYTES, quant_t, block_col, block_row)
            cute.arch.sync_threads()
            out_tile_addr = out_group_base + Int64(tile_row_start) * Int64(K) + Int64(tile_col_start)
            store_smem_tile(s_tile, out_tile_addr, M - tile_row_start, K - tile_col_start, K, NUM_THREADS, self.out_width)
            out_t_tile_addr = out_t_group_base + Int64(tile_col_start) * Int64(M) + Int64(tile_row_start)
            store_smem_tile(s_tile + TILE_BYTES, out_t_tile_addr, K - tile_col_start, M - tile_row_start, M, NUM_THREADS, self.out_t_width)
        else:
            row0, col0 = tile_row_start + block_row, tile_col_start + block_col
            store_rows(out_group_base, quant, row0, col0, M, K, True)
            store_rows(out_t_group_base, quant_t, col0, row0, K, M, True)

        if tidx == 0:
            num_tiles = Int64(num_row_tiles) * Int64(num_col_tiles)
            scale_idx, scale_t_idx = Int64(bz) * num_tiles + Int64(bx) * Int64(num_col_tiles) + Int64(by), Int64(bz) * num_tiles + Int64(by) * Int64(num_row_tiles) + Int64(bx)
            st_scalar(scale_base + scale_idx * 4, scale_bits, Int32(1), 4)
            st_scalar(scale_t_base + scale_t_idx * 4, scale_bits, Int32(1), 4)


COMPILED: dict = {}


def fake_row_major(dtype, num_dims: int, align: int) -> cute.Tensor:
    shape = tuple(cute.sym_int() for _ in range(num_dims))
    stride = tuple(cute.sym_int64() for _ in range(num_dims - 1)) + (1,)
    return fake(dtype, shape, stride, align)


def fake_strided(dtype, num_dims: int, align: int) -> cute.Tensor:
    return fake(dtype, tuple(cute.sym_int() for _ in range(num_dims)), tuple(cute.sym_int64() for _ in range(num_dims)), align)


def compile_blockwise(dtype: torch.dtype, load: str, store: str, out_width: int, out_t_width: int, is_batched: bool):
    key = (dtype, load, store, out_width, out_t_width, is_batched)
    fn = COMPILED.get(key)
    if fn is None:
        in_bytes, num_dims = torch.tensor([], dtype=dtype).element_size(), 3 if is_batched else 2
        fn = cute.compile(BlockwiseTranspose(in_bytes, load, store, out_width, out_t_width, is_batched), fake_strided(TORCH2CUTE[dtype], num_dims, in_bytes), fake_row_major(cutlass.Float8E4M3FN, num_dims, 1), fake_row_major(cutlass.Float32, num_dims, 4), fake_row_major(cutlass.Float8E4M3FN, num_dims, 1), fake_row_major(cutlass.Float32, num_dims, 4), Int64(0), Int64(0), Int64(0), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    return fn


def store_width(row_bytes: int) -> int:
    return 16 if row_bytes % 16 == 0 else 4 if row_bytes % 4 == 0 else 1


def resolve(dtype: torch.dtype, shape: tuple, stride: tuple, x_addr: int) -> tuple:
    """
    (compiled kernel or None for an empty input, whether the outputs are swapped, group / row /
    column element strides) of an x of this dtype, shape, strides and address modulo 16.
    """
    is_batched = len(shape) == 3
    G, N, K = shape[0] if is_batched else 1, shape[-2], shape[-1]
    if G == 0 or N == 0 or K == 0:
        return None, False, ()
    group_stride, row_stride, col_stride = (stride[0] if G > 1 else 0) if is_batched else 0, stride[-2], stride[-1]
    # a view whose transpose has unit column stride runs on that transpose with the outputs swapped
    swap = N > 1 and K > 1 and col_stride != 1 and row_stride == 1
    if swap:
        N, K, row_stride, col_stride = K, N, col_stride, row_stride
    # strides of size-1 dims never move the address: zero them so they do not affect alignment
    row_stride, col_stride = row_stride if N > 1 else 0, col_stride if K > 1 else 1
    in_bytes = dtype.itemsize
    is_vec = col_stride == 1 and K % 8 == 0 and x_addr % 16 == 0 and group_stride * in_bytes % 16 == 0 and row_stride * in_bytes % 16 == 0
    if N % 16 == 0 and K % 16 == 0 and G * ((N + BLOCK - 1) // BLOCK) * ((K + BLOCK - 1) // BLOCK) >= SMEM_MIN_TILES:
        store, out_width, out_t_width = "staged", 16, 16
    elif N % 8 == 0 and K % 8 == 0:
        store, out_width, out_t_width = "regs", 8, 8
    else:
        store, out_width, out_t_width = "staged", store_width(K), store_width(N)
    return compile_blockwise(dtype, "vec" if is_vec else "staged", store, out_width, out_t_width, is_batched), swap, (group_stride, row_stride, col_stride)


# fp8_quant_weight[_grouped] on SM90 as torch.library ops, used for every call (eager and inside
# torch.compile regions, where they are opaque). Their kernels are C++ (OPS_SOURCE, built with the host
# compiler at first import): a CUDA kernel checks the input, allocates the four outputs and launches the
# CuTe kernel through a cache of (compiled kernel's tvm-ffi function, output swap, strides) keyed by
# (dtype, sizes, strides, address mod 16), each entry resolved once by resolve() through a tvm-ffi
# global function, on torch's current stream. The Autograd-key kernel runs that launch directly for
# real tensors outside tracing and dispatch modes (the outputs never carry gradients), which skips the
# autograd-not-implemented fallback, and redispatches below autograd otherwise (fake or functional
# tensors, an active TorchDispatchMode), to the fake or the CUDA kernel.
OPS_SOURCE = r"""
#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cstring>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace {

using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

struct Key {
  int64_t words[8];
  bool operator==(const Key& other) const { return std::memcmp(words, other.words, sizeof(words)) == 0; }
};

struct KeyHash {
  size_t operator()(const Key& key) const {
    size_t hash = 1469598103934665603ull;
    for (int64_t word : key.words) hash = (hash ^ static_cast<size_t>(word)) * 1099511628211ull;
    return hash;
  }
};

struct Entry {
  tvm::ffi::Function fn;
  bool swap = false, empty = true;
  int64_t strides[3] = {0, 0, 0};
};

constexpr size_t max_entries = 4096;
// Leaked on purpose: destroying them after the interpreter finalizes would crash.
auto& cache = *new std::unordered_map<Key, Entry, KeyHash>();
std::mutex cache_mutex;

// Sets tvm-ffi's environment stream for a device and restores the previous one, also on exceptions.
struct EnvStream {
  int32_t device;
  TVMFFIStreamHandle previous = nullptr;
  EnvStream(int32_t device, cudaStream_t stream) : device(device) { TVMFFIEnvSetStream(kDLCUDA, device, stream, &previous); }
  ~EnvStream() { TVMFFIEnvSetStream(kDLCUDA, device, previous, nullptr); }
};

DLTensor as_dl(const at::Tensor& t) {
  DLTensor d;
  d.data = t.data_ptr();
  d.device = DLDevice{kDLCUDA, static_cast<int32_t>(t.device().index())};
  d.ndim = static_cast<int32_t>(t.dim());
  d.dtype = at::getDLDataType(t);
  d.shape = const_cast<int64_t*>(t.sizes().data());
  d.strides = const_cast<int64_t*>(t.strides().data());
  d.byte_offset = 0;
  return d;
}

Entry lookup(const at::Tensor& x) {
  Key key;
  std::memset(key.words, 0, sizeof(key.words));
  int64_t ndim = x.dim();
  key.words[0] = static_cast<int64_t>(x.scalar_type()) * 16 + ndim;
  for (int64_t i = 0; i < ndim; ++i) key.words[1 + i] = x.size(i), key.words[4 + i] = x.stride(i);
  key.words[7] = reinterpret_cast<int64_t>(x.data_ptr()) & 15;
  std::lock_guard<std::mutex> guard(cache_mutex);
  auto found = cache.find(key);
  if (found != cache.end()) return found->second;
  static tvm::ffi::Function& resolve = *new tvm::ffi::Function(tvm::ffi::Function::GetGlobalRequired("pithtrain.fp8_quant_weight_resolve_sm90"));
  tvm::ffi::Array<int64_t> sizes(x.sizes().begin(), x.sizes().end()), strides(x.strides().begin(), x.strides().end());
  // the dispatcher released the GIL, and the resolver is Python
  PyGILState_STATE gil = PyGILState_Ensure();
  auto result = resolve(static_cast<int64_t>(x.scalar_type()), sizes, strides, key.words[7]).cast<tvm::ffi::Array<tvm::ffi::Any>>();
  PyGILState_Release(gil);
  Entry entry;
  // Function is a nullable reference, so an empty input's None is told apart by its type index
  entry.empty = result[0].type_index() == tvm::ffi::TypeIndex::kTVMFFINone;
  if (!entry.empty) {
    entry.fn = result[0].cast<tvm::ffi::Function>();
    entry.swap = result[1].cast<bool>();
    for (int i = 0; i < 3; ++i) entry.strides[i] = result[2 + i].cast<int64_t>();
  }
  if (cache.size() >= max_entries) cache.clear();
  cache.emplace(key, entry);
  return entry;
}

template <int64_t ndim>
Outputs quant(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == ndim, "fp8_quant_weight on SM90 takes a ", ndim, "-D CUDA tensor, got ", x.dim(), "-D on ", x.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "fp8_quant_weight on SM90 takes bf16 or fp32, got ", x.scalar_type());
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t G = ndim == 3 ? x.size(0) : 1, N = x.size(-2), K = x.size(-1), rows = (N + 127) / 128, cols = (K + 127) / 128;
  // the CUDA allocator directly, not at::empty through the dispatcher (host cost per call)
  auto alloc = [&](int64_t a, int64_t b, at::ScalarType dtype) {
    int64_t sizes[3] = {G, a, b};
    return at::Tensor(at::detail::empty_cuda(at::IntArrayRef(sizes + (3 - ndim), ndim), dtype, x.device(), std::nullopt));
  };
  at::Tensor out = alloc(N, K, at::kFloat8_e4m3fn), scale = alloc(rows, cols, at::kFloat), out_t = alloc(K, N, at::kFloat8_e4m3fn), scale_t = alloc(cols, rows, at::kFloat);
  Entry entry = lookup(x);
  if (entry.empty) return {out, scale, out_t, scale_t};
  DLTensor d_x = as_dl(x), d_out = as_dl(out), d_scale = as_dl(scale), d_out_t = as_dl(out_t), d_scale_t = as_dl(scale_t);
  if (entry.swap) std::swap(d_out, d_out_t), std::swap(d_scale, d_scale_t);
  int32_t device = static_cast<int32_t>(x.device().index());
  EnvStream env(device, c10::cuda::getCurrentCUDAStream(device).stream());
  entry.fn(&d_x, &d_out, &d_scale, &d_out_t, &d_scale_t, entry.strides[0], entry.strides[1], entry.strides[2]);
  return {out, scale, out_t, scale_t};
}

template <int64_t ndim, const char* name>
Outputs quant_autograd(const at::Tensor& x) {
  constexpr c10::DispatchKeySet traced({c10::DispatchKey::Python, c10::DispatchKey::PythonTLSSnapshot, c10::DispatchKey::Functionalize});
  if (!x.key_set().has_any(traced) && !c10::impl::tls_local_dispatch_key_set().included_.has_any(traced)) return quant<ndim>(x);
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, "").typed<Outputs(const at::Tensor&)>();
  at::AutoDispatchBelowADInplaceOrView guard;
  return op.call(x);
}

constexpr char quant_name[] = "pithtrain::fp8_quant_weight_sm90";
constexpr char quant_grouped_name[] = "pithtrain::fp8_quant_weight_grouped_sm90";

}  // namespace

TORCH_LIBRARY_FRAGMENT(pithtrain, m) {
  m.def("fp8_quant_weight_sm90(Tensor x) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("fp8_quant_weight_grouped_sm90(Tensor x) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(pithtrain, CUDA, m) {
  m.impl("fp8_quant_weight_sm90", &quant<2>);
  m.impl("fp8_quant_weight_grouped_sm90", &quant<3>);
}

TORCH_LIBRARY_IMPL(pithtrain, Autograd, m) {
  m.impl("fp8_quant_weight_sm90", &quant_autograd<2, quant_name>);
  m.impl("fp8_quant_weight_grouped_sm90", &quant_autograd<3, quant_grouped_name>);
}
"""


@tvm_ffi.register_global_func("pithtrain.fp8_quant_weight_resolve_sm90")
def resolve_for_ops(scalar_type: int, sizes: Sequence[int], strides: Sequence[int], align: int) -> list:
    """
    resolve() for the C++ kernels: [tvm-ffi function of the compiled kernel or None, output swap,
    group / row / column strides] of an input of this torch ScalarType (bf16 or fp32), sizes, strides
    and address modulo 16.
    """
    dtype = {15: torch.bfloat16, 6: torch.float32}[scalar_type]
    fn, swap, launch_strides = resolve(dtype, tuple(sizes), tuple(strides), align)
    return [None, False, 0, 0, 0] if fn is None else [fn._tvm_ffi_function, swap, *launch_strides]


def build_ops() -> str:
    """
    The shared library of OPS_SOURCE, built with the host C++ compiler against this torch and
    tvm-ffi once per source and version, cached in ~/.cache/pithtrain. Each builder writes its own
    source and library under names unique to the host and process and publishes the library with an
    atomic rename, so concurrent builders (ranks, nodes) never see a partial file; the file lock only
    keeps them from building it more than once where it holds (a missed lock costs a duplicate build).
    """
    digest = hashlib.sha256("\n".join([OPS_SOURCE, torch.__version__, tvm_ffi.__version__, sys.version]).encode()).hexdigest()[:16]
    root = os.path.join(os.path.expanduser("~/.cache/pithtrain"), "fp8_weight_quant_ops_sm90", digest)
    library = os.path.join(root, "ops.so")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(library):
            unique = f"{socket.gethostname()}.{os.getpid()}"
            source, partial = os.path.join(root, f"ops.{unique}.cpp"), f"{library}.{unique}.tmp"
            with open(source, "w") as f:
                f.write(OPS_SOURCE)
            includes = cpp_extension.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"], tvm_ffi.libinfo.find_include_path(), tvm_ffi.libinfo.find_dlpack_include_path()]
            tvm_ffi_dir = os.path.dirname(tvm_ffi.libinfo.find_libtvm_ffi())
            command = ["c++", "-O2", "-std=c++17", "-shared", "-fPIC", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", source, "-o", partial]
            command += [f"-I{path}" for path in includes] + [f"-L{path}" for path in cpp_extension.library_paths(device_type="cuda")] + [f"-L{tvm_ffi_dir}", f"-Wl,-rpath,{tvm_ffi_dir}", "-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda", "-ltorch", "-ltvm_ffi"]
            subprocess.run(command, check=True)
            os.replace(partial, library)
            os.remove(source)
    return library


def quant_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The four outputs of the ops for an input x (2-D (N, K) or 3-D (G, N, K)): e4m3 (.., N, K), fp32
    (.., ceil(N / 128), ceil(K / 128)), e4m3 (.., K, N) and fp32 (.., ceil(K / 128), ceil(N / 128)).
    """
    *lead, N, K = x.shape
    rows, cols = (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
    return x.new_empty((*lead, N, K), dtype=torch.float8_e4m3fn), x.new_empty((*lead, rows, cols), dtype=torch.float32), x.new_empty((*lead, K, N), dtype=torch.float8_e4m3fn), x.new_empty((*lead, cols, rows), dtype=torch.float32)


torch.ops.load_library(build_ops())
torch.library.register_fake("pithtrain::fp8_quant_weight_sm90", quant_fake)
torch.library.register_fake("pithtrain::fp8_quant_weight_grouped_sm90", quant_fake)
# fp8_quant_weight on SM90 (x 2-D) and fp8_quant_weight_grouped on SM90 (x 3-D): bf16 or fp32, any
# shape and strides, outputs as the picker documents them.
fp8_quant_weight_sm90, fp8_quant_weight_grouped_sm90 = torch.ops.pithtrain.fp8_quant_weight_sm90.default, torch.ops.pithtrain.fp8_quant_weight_grouped_sm90.default


@cute.jit
def shard_tile(shard_base: Int64, amax_addr: Int64, out_base: Int64, local_rows: Int32, K: Int32, tile_row_idx: Int32, tile_col_idx: Int32, group_idx: Int32, row_offset: Int32, is_cast: cutlass.Constexpr[bool], mode: cutlass.Constexpr[str]):
    """
    One 128x128 tile (global tile row tile_row_idx, column tile tile_col_idx, local group group_idx)
    of an fp32 row shard: its amax into amax_addr, or (is_cast) its e4m3 cast into out_base.
    """
    if const_expr(mode == "row"):
        shard_tile_row(shard_base, amax_addr, out_base, local_rows, K, tile_row_idx, tile_col_idx, group_idx, row_offset, is_cast)
        return
    is_aligned = const_expr(mode != "scalar")
    tidx, warp, lane, block_row, block_col = thread_coords()
    row0 = tile_row_idx * BLOCK + block_row - row_offset
    col0 = tile_col_idx * BLOCK + block_col
    group_elem_start = Int64(group_idx) * Int64(local_rows) * Int64(K)
    rows = load_rows(shard_base + group_elem_start * 4, row0, col0, local_rows, K, Int64(K), 4, True, is_aligned)
    if const_expr(is_cast):
        scale_bits = compute_scale_bits(ld_scalar_ordered(amax_addr, Int32(1), 4))
        quant = cast_rows(rows, Uint32(0x7F000000) - scale_bits, True)
        store_rows(out_base + group_elem_start, quant, row0, col0, local_rows, K, is_aligned)
    else:
        smem = cutlass.memory.SmemAllocator()
        s_amax = smem.allocate_tensor(Uint32, cute.make_layout(NUM_WARPS), byte_alignment=16)
        amax_bits = reduce_block_amax(reduce_thread_amax(rows, True), warp, lane, s_amax)
        if tidx == 0:
            st_scalar(amax_addr, amax_bits, Int32(1), 4)


@cute.jit
def shard_tile_row(shard_base: Int64, amax_addr: Int64, out_base: Int64, local_rows: Int32, K: Int32, tile_row_idx: Int32, tile_col_idx: Int32, group_idx: Int32, row_offset: Int32, is_cast: cutlass.Constexpr[bool]):
    """
    Row-major variant of shard_tile: a warp loads one 128-column tile row (16 bytes per lane), warps
    take rows round-robin, so a shard holding a few rows of a tile still uses all 8 warps. Needs K %
    4 == 0 and a 16-byte aligned shard.
    """
    tidx, _, _ = cute.arch.thread_idx()
    warp, lane = tidx // 32, tidx % 32
    col = tile_col_idx * BLOCK + lane * 4
    group_elem_start = Int64(group_idx) * Int64(local_rows) * Int64(K)
    tile_row_start = tile_row_idx * BLOCK - row_offset
    words, rows_ok = load_shard_rows(shard_base, group_elem_start, tile_row_start, warp, col, local_rows, K)
    if const_expr(is_cast):
        cast_shard_rows(out_base, amax_addr, words, rows_ok, group_elem_start, tile_row_start, warp, col, K)
    else:
        amax_bits = reduce_shard_rows_amax(words, warp, lane)
        if tidx == 0:
            st_scalar(amax_addr, amax_bits, Int32(1), 4)


@cute.jit
def load_shard_rows(shard_base: Int64, group_elem_start: Int64, tile_row_start: Int32, warp: Int32, col: Int32, local_rows: Int32, K: Int32) -> tuple:
    """
    Load the warp's rows of the tile, four fp32 per lane rounded to two bf16x2 words; returns the
    words and the per-row predicates (rows outside the shard read as zero).
    """
    col_ok = Int32(col < K)
    words, rows_ok = [], []
    for i in cutlass.range_constexpr(BLOCK // NUM_WARPS):
        row = tile_row_start + i * NUM_WARPS + warp
        row_ok = Int32(row >= 0) & Int32(row < local_rows) & col_ok
        f32_bits = ld_v4_ordered(shard_base + (group_elem_start + Int64(row) * Int64(K) + Int64(col)) * 4, row_ok)
        words.append([f32x2_bits_to_bf16x2(f32_bits[0], f32_bits[1]), f32x2_bits_to_bf16x2(f32_bits[2], f32_bits[3])])
        rows_ok.append(row_ok)
    return words, rows_ok


@cute.jit
def cast_shard_rows(out_base: Int64, amax_addr: Int64, words: list, rows_ok: list, group_elem_start: Int64, tile_row_start: Int32, warp: Int32, col: Int32, K: Int32) -> None:
    rcp_bits = Uint32(0x7F000000) - compute_scale_bits(ld_scalar_ordered(amax_addr, Int32(1), 4))
    for i in cutlass.range_constexpr(BLOCK // NUM_WARPS):
        row = tile_row_start + i * NUM_WARPS + warp
        cols01, cols23 = words[i]
        quant = cast4(cols01 << 16, cols01 & Uint32(0xFFFF0000), cols23 << 16, cols23 & Uint32(0xFFFF0000), rcp_bits)
        st_scalar(out_base + group_elem_start + Int64(row) * Int64(K) + Int64(col), quant, rows_ok[i], 4)


@cute.jit
def reduce_shard_rows_amax(words: list, warp: Int32, lane: Int32) -> Uint32:
    """
    CTA-wide amax bits (as fp32 bits) of the tile's bf16x2 words.
    """
    acc_amax = Uint32(0)
    for i in cutlass.range_constexpr(BLOCK // NUM_WARPS):
        acc_amax = max_u16x2(acc_amax, words[i][0] & Uint32(0x7FFF7FFF))
        acc_amax = max_u16x2(acc_amax, words[i][1] & Uint32(0x7FFF7FFF))
    acc_amax = max_u32(acc_amax & Uint32(0xFFFF), acc_amax >> 16) << 16
    smem = cutlass.memory.SmemAllocator()
    s_amax = smem.allocate_tensor(Uint32, cute.make_layout(NUM_WARPS), byte_alignment=16)
    return reduce_block_amax(acc_amax, warp, lane, s_amax)


class ShardTile:
    def __init__(self, is_cast: bool, mode: str) -> None:
        self.is_cast = is_cast
        self.mode = mode

    @cute.jit
    def __call__(self, mShard: cute.Tensor, mAmax: cute.Tensor, mOut: cute.Tensor, row_offset: Int32, group_offset: Int32, stream: cuda.CUstream):
        num_local_groups, local_rows, K = mShard.shape[0], mShard.shape[1], mShard.shape[2]
        tile_row_start, tile_row_end, num_col_tiles = row_offset // BLOCK, (row_offset + local_rows + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        self.kernel(mShard.iterator.toint(), mAmax.iterator.toint(), mOut.iterator.toint(), Int32(local_rows), Int32(K), Int32(tile_row_start), Int32(mAmax.shape[1]), Int32(mAmax.shape[2]), row_offset, group_offset).launch(grid=[tile_row_end - tile_row_start, num_col_tiles, num_local_groups], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, shard_base: Int64, amax_base: Int64, out_base: Int64, local_rows: Int32, K: Int32, tile_row_start: Int32, amax_rows: Int32, amax_cols: Int32, row_offset: Int32, group_offset: Int32):
        bx, by, bz = cute.arch.block_idx()
        tile_row_idx = tile_row_start + bx
        amax_idx = (Int64(group_offset + bz) * Int64(amax_rows) + Int64(tile_row_idx)) * Int64(amax_cols) + Int64(by)
        shard_tile(shard_base, amax_base + amax_idx * 4, out_base, local_rows, K, tile_row_idx, by, bz, row_offset, self.is_cast, self.mode)


def shard_tiles(shard: torch.Tensor, amax: torch.Tensor, out: Optional[torch.Tensor], row_offset: int, group_offset: int) -> None:
    if shard.numel() == 0:
        return
    K = shard.shape[2]
    is_cast = out is not None
    if K % 4 == 0 and shard.data_ptr() % 16 == 0 and (not is_cast or out.data_ptr() % 4 == 0):
        mode = "row"
    else:
        mode = "scalar"
    key = ("shard", is_cast, mode, out.dtype if is_cast else None)
    fn = COMPILED.get(key)
    if fn is None:
        fn = cute.compile(ShardTile(is_cast, mode), fake_row_major(cutlass.Float32, 3, 4), fake_row_major(cutlass.Float32, 3, 4), fake_row_major(TORCH2CUTE[out.dtype] if is_cast else cutlass.Float32, 3, 1), Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    fn(shard, amax, out if is_cast else shard, row_offset, group_offset)


def fp8_shard_amax_sm90(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:
    shard_tiles(shard, amax, None, row_offset, group_offset)


def fp8_shard_cast_sm90(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:
    shard_tiles(shard, amax, out, row_offset, group_offset)


class Scales:
    @cute.jit
    def __call__(self, mAmax: cute.Tensor, mScale: cute.Tensor, mScaleT: cute.Tensor, stream: cuda.CUstream):
        G, amax_rows, amax_cols = mAmax.shape[0], mAmax.shape[1], mAmax.shape[2]
        self.kernel(mAmax.iterator.toint(), mScale.iterator.toint(), mScaleT.iterator.toint(), Int32(amax_rows), Int32(amax_cols)).launch(grid=[(amax_rows * amax_cols + NUM_THREADS - 1) // NUM_THREADS, G, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, amax_base: Int64, scale_base: Int64, scale_t_base: Int64, amax_rows: Int32, amax_cols: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, group_idx, _ = cute.arch.block_idx()
        store_scale(amax_base, scale_base, scale_t_base, Int64(group_idx), bx * NUM_THREADS + tidx, amax_rows, amax_cols)


@cute.jit
def store_scale(amax_base: Int64, scale_base: Int64, scale_t_base: Int64, group_idx: Int64, tile_idx: Int32, amax_rows: Int32, amax_cols: Int32):
    num_tiles = amax_rows * amax_cols
    tile_ok = Int32(tile_idx < num_tiles)
    group_start = group_idx * Int64(num_tiles)
    scale_bits = compute_scale_bits(ld_scalar_ordered(amax_base + (group_start + Int64(tile_idx)) * 4, tile_ok, 4))
    tile_row = tile_idx // amax_cols
    tile_col = tile_idx - tile_row * amax_cols
    st_scalar(scale_base + (group_start + Int64(tile_idx)) * 4, scale_bits, tile_ok, 4)
    st_scalar(scale_t_base + (group_start + Int64(tile_col) * Int64(amax_rows) + Int64(tile_row)) * 4, scale_bits, tile_ok, 4)


def fp8_scales_sm90(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    fn = COMPILED.get("scales")
    if fn is None:
        fakes = [fake_row_major(cutlass.Float32, 3, 4) for _ in range(3)]
        fn = cute.compile(Scales(), *fakes, fake_stream(), options="--enable-tvm-ffi")
        COMPILED["scales"] = fn
    fn(amax, scale, scale_t)


PACK_THREADS = 128


class PackTranspose:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.num_threads = PACK_THREADS if mode == "w16" else NUM_THREADS

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        G, N, K = mX.shape[0], mX.shape[1], mX.shape[2]
        self.kernel(mX.iterator.toint(), mOut.iterator.toint(), Int32(N), Int32(K), Int32(G)).launch(grid=[(N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK, G], block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x_base: Int64, out_base: Int64, N: Int32, K: Int32, G: Int32):
        if const_expr(self.mode == "w16"):
            self.kernel_w16(x_base, out_base, N, K, G)
        else:
            self.kernel_8x8(x_base, out_base, N, K, G)

    @cute.jit
    def kernel_w16(self, x_base: Int64, out_base: Int64, N: Int32, K: Int32, G: Int32):
        """
        128 threads, each owning 8 rows x 16 bytes; the transposed tile is staged in shared memory
        for 16-byte stores.
        """
        bx, by, bz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp, lane = tidx // 32, tidx % 32
        block_row, block_col = warp * 32 + (lane // 8) * 8, (lane % 8) * 16
        tile_row_start, tile_col_start = bx * BLOCK, by * BLOCK
        group_elems = Int64(N) * Int64(K)
        group_start = Int64(bz) * group_elems
        col_ok = Int32(tile_col_start + block_col < K)
        words = []
        for i in cutlass.range_constexpr(8):
            row = tile_row_start + block_row + i
            row_addr = x_base + group_start + Int64(row) * Int64(K) + Int64(tile_col_start + block_col)
            words.append(ld_v4_ordered(row_addr, Int32(row < N) & col_ok))
        for i in cutlass.range_constexpr(8):
            row = tile_row_start + block_row + i
            row_addr = out_base + group_start + Int64(row) * Int64(K) + Int64(tile_col_start + block_col)
            st_v4(row_addr, words[i], Int32(row < N) & col_ok)
        left, right = transpose8([[words[i][0], words[i][1]] for i in range(8)]), transpose8([[words[i][2], words[i][3]] for i in range(8)])
        smem = cutlass.memory.SmemAllocator()
        s_tiles = smem.allocate_tensor(Uint32, cute.make_layout(TILE_BYTES // 4), byte_alignment=128)
        s_tile = s_tiles.iterator.toint()
        write_smem_block(s_tile, left, block_col, block_row)
        write_smem_block(s_tile, right, block_col + 8, block_row)
        cute.arch.sync_threads()
        out_t_tile_addr = out_base + Int64(G) * group_elems + group_start + Int64(tile_col_start) * Int64(N) + Int64(tile_row_start)
        store_smem_tile(s_tile, out_t_tile_addr, K - tile_col_start, N - tile_row_start, N, PACK_THREADS)

    @cute.jit
    def kernel_8x8(self, x_base: Int64, out_base: Int64, N: Int32, K: Int32, G: Int32):
        bx, by, bz = cute.arch.block_idx()
        _, _, _, block_row, block_col = thread_coords()
        row0, col0 = bx * BLOCK + block_row, by * BLOCK + block_col
        group_elems = Int64(N) * Int64(K)
        group_start = Int64(bz) * group_elems
        is_aligned = const_expr(self.mode != "scalar")
        words = load_rows(x_base + group_start, row0, col0, N, K, Int64(K), 1, False, is_aligned)
        words_t = transpose8(words)
        out_t_base = out_base + Int64(G) * group_elems + group_start
        store_rows(out_base + group_start, words, row0, col0, N, K, is_aligned)
        store_rows(out_t_base, words_t, col0, row0, K, N, is_aligned)


def fp8_pack_transpose_sm90(x: torch.Tensor, out: torch.Tensor) -> None:
    G, N, K = x.shape
    if x.numel() == 0:
        return
    if N % 16 == 0 and K % 16 == 0 and x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0:
        mode = "w16"
    elif N % 8 == 0 and K % 8 == 0 and x.data_ptr() % 8 == 0 and out.data_ptr() % 8 == 0:
        mode = "v8"
    else:
        mode = "scalar"
    key = ("pack", mode, x.dtype, out.dtype)
    fn = COMPILED.get(key)
    if fn is None:
        fn = cute.compile(PackTranspose(mode), fake_row_major(TORCH2CUTE[x.dtype], 3, 1), fake_row_major(TORCH2CUTE[out.dtype], 1, 1), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    fn(x, out)


DESC_WORDS = 8  # int64 words per descriptor row


@cute.jit
def find_table_row(table_base: Int64, num_rows: Int32, cta_idx: Int32, row_words: cutlass.Constexpr[int]) -> Int32:
    """
    Index of the last descriptor row (row_words int64 words apart) whose first CTA, word 7, is at
    or below cta_idx.
    """
    row_idx = Int32(0)
    for i in cutlass.range(1, num_rows, unroll=1):
        first_cta = Int32(ld_u64(table_base + (Int64(i) * row_words + 7) * 8))
        if first_cta <= cta_idx:
            row_idx = i
    return row_idx


class ShardAmaxMulti:
    """
    Shard amax over n shards. The table holds two rows per shard: (shard, amax, local_rows, K,
    row_offset, first tile row, element offset of group_offset in the grid, first CTA) and (amax
    rows, amax columns, column tiles, tile rows, 0, 0, 0, 0).
    """

    @cute.jit
    def __call__(self, mTable: cute.Tensor, num_shards: Int32, num_ctas: Int32, stream: cuda.CUstream):
        self.kernel(mTable.iterator.toint(), num_shards).launch(grid=[num_ctas, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, table_base: Int64, num_shards: Int32):
        bx, _, _ = cute.arch.block_idx()
        shard_idx = find_table_row(table_base, num_shards, bx, 2 * DESC_WORDS)
        desc_addr = table_base + Int64(shard_idx) * 2 * DESC_WORDS * 8
        shard_base, amax_base = ld_u64(desc_addr), ld_u64(desc_addr + 8)
        local_rows, K = Int32(ld_u64(desc_addr + 16)), Int32(ld_u64(desc_addr + 24))
        row_offset, tile_row_start = Int32(ld_u64(desc_addr + 32)), Int32(ld_u64(desc_addr + 40))
        amax_group_start = ld_u64(desc_addr + 48)
        first_cta = Int32(ld_u64(desc_addr + 56))
        amax_rows, amax_cols, num_col_tiles, num_row_tiles = Int32(ld_u64(desc_addr + 64)), Int32(ld_u64(desc_addr + 72)), Int32(ld_u64(desc_addr + 80)), Int32(ld_u64(desc_addr + 88))
        local_cta = bx - first_cta
        tile_col_idx, row_group_idx = local_cta % num_col_tiles, local_cta // num_col_tiles
        local_tile_row, group_idx = row_group_idx % num_row_tiles, row_group_idx // num_row_tiles
        tile_row_idx = tile_row_start + local_tile_row
        amax_idx = amax_group_start + (Int64(group_idx) * Int64(amax_rows) + Int64(tile_row_idx)) * Int64(amax_cols) + Int64(tile_col_idx)
        shard_tile(shard_base, amax_base + amax_idx * 4, amax_base, local_rows, K, tile_row_idx, tile_col_idx, group_idx, row_offset, False, "row")


class FP8ShardAmaxGroup:
    """
    One launch of fp8_shard_amax over a fixed list of shards. The table holds raw addresses, so it
    is valid only while the tensors are not reallocated. Shards with K % 4 != 0 or not 16-byte
    aligned get one launch each.
    """

    def __init__(self, shards: Sequence[torch.Tensor], amaxes: Sequence[torch.Tensor], row_offsets: Sequence[int], group_offsets: Sequence[int]) -> None:
        desc_rows = []
        first_cta = 0
        self.fallback = []
        for shard, amax, row_offset, group_offset in zip(shards, amaxes, row_offsets, group_offsets):
            num_local_groups, local_rows, K = shard.shape
            if shard.numel() == 0:
                continue
            if K % 4 != 0 or shard.data_ptr() % 16 != 0:
                self.fallback.append((shard, amax, row_offset, group_offset))
                continue
            _, amax_rows, amax_cols = amax.shape
            tile_row_start = row_offset // BLOCK
            num_row_tiles = (row_offset + local_rows + BLOCK - 1) // BLOCK - tile_row_start
            num_col_tiles = (K + BLOCK - 1) // BLOCK
            desc_rows.append([shard.data_ptr(), amax.data_ptr(), local_rows, K, row_offset, tile_row_start, group_offset * amax_rows * amax_cols, first_cta])
            desc_rows.append([amax_rows, amax_cols, num_col_tiles, num_row_tiles, 0, 0, 0, 0])
            first_cta += num_row_tiles * num_col_tiles * num_local_groups
        self.num_ctas = first_cta
        self.num_shards = len(desc_rows) // 2
        device = shards[0].device
        self.table = torch.tensor(desc_rows or [[0] * DESC_WORDS], dtype=torch.int64, device=device)
        fn = COMPILED.get("amax_multi")
        if fn is None:
            fn = cute.compile(ShardAmaxMulti(), fake_row_major(cutlass.Int64, 2, 8), Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")
            COMPILED["amax_multi"] = fn
        self.fn = fn

    def __call__(self) -> None:
        if self.num_ctas:
            self.fn(self.table, self.num_shards, self.num_ctas)
        for shard, amax, row_offset, group_offset in self.fallback:
            fp8_shard_amax_sm90(shard, amax, row_offset, group_offset)


class ScalesMulti:
    """
    Scales over n grids. The table holds one row per grid: (amax, scale, scale_t, amax rows, amax
    columns, CTAs per group, 0, first CTA).
    """

    @cute.jit
    def __call__(self, mTable: cute.Tensor, num_grids: Int32, num_ctas: Int32, stream: cuda.CUstream):
        self.kernel(mTable.iterator.toint(), num_grids).launch(grid=[num_ctas, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, table_base: Int64, num_grids: Int32):
        bx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        grid_idx = find_table_row(table_base, num_grids, bx, DESC_WORDS)
        desc_addr = table_base + Int64(grid_idx) * DESC_WORDS * 8
        amax_base, scale_base, scale_t_base = ld_u64(desc_addr), ld_u64(desc_addr + 8), ld_u64(desc_addr + 16)
        amax_rows = Int32(ld_u64(desc_addr + 24))
        amax_cols, ctas_per_group, first_cta = Int32(ld_u64(desc_addr + 32)), Int32(ld_u64(desc_addr + 40)), Int32(ld_u64(desc_addr + 56))
        local_cta = bx - first_cta
        group_idx = local_cta // ctas_per_group
        group_cta = local_cta - group_idx * ctas_per_group
        store_scale(amax_base, scale_base, scale_t_base, Int64(group_idx), group_cta * NUM_THREADS + tidx, amax_rows, amax_cols)


class FP8ScalesGroup:
    """
    One launch of fp8_scales over a fixed list of (amax, scale, scale_t) grids; same address
    caveat as FP8ShardAmaxGroup.
    """

    def __init__(self, amaxes: Sequence[torch.Tensor], scales: Sequence[torch.Tensor], scales_t: Sequence[torch.Tensor]) -> None:
        desc_rows = []
        first_cta = 0
        for amax, scale, scale_t in zip(amaxes, scales, scales_t):
            G, amax_rows, amax_cols = amax.shape
            ctas_per_group = (amax_rows * amax_cols + NUM_THREADS - 1) // NUM_THREADS
            desc_rows.append([amax.data_ptr(), scale.data_ptr(), scale_t.data_ptr(), amax_rows, amax_cols, ctas_per_group, 0, first_cta])
            first_cta += G * ctas_per_group
        self.num_ctas = first_cta
        self.num_grids = len(desc_rows)
        self.table = torch.tensor(desc_rows, dtype=torch.int64, device=amaxes[0].device)
        fn = COMPILED.get("scales_multi")
        if fn is None:
            fn = cute.compile(ScalesMulti(), fake_row_major(cutlass.Int64, 2, 8), Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")
            COMPILED["scales_multi"] = fn
        self.fn = fn

    def __call__(self) -> None:
        if self.num_ctas:
            self.fn(self.table, self.num_grids, self.num_ctas)


def fp8_shard_amax_multi_sm90(shards: Sequence[torch.Tensor], amaxes: Sequence[torch.Tensor], row_offsets: Sequence[int], group_offsets: Sequence[int]) -> FP8ShardAmaxGroup:
    return FP8ShardAmaxGroup(shards, amaxes, row_offsets, group_offsets)


def fp8_scales_multi_sm90(amaxes: Sequence[torch.Tensor], scales: Sequence[torch.Tensor], scales_t: Sequence[torch.Tensor]) -> FP8ScalesGroup:
    return FP8ScalesGroup(amaxes, scales, scales_t)
