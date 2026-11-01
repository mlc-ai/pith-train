"""
Rowwise and K-major FP8 quantization of a grouped input from one read, both scales returned
MN-major, for every input fp8cast_rowwise_kmajor takes on SM90 (bf16, fp16 or fp32, any strides,
any base alignment, any N; int32 or int64 offsets of any stride).

Two kernels, chosen on the host. RowwiseKmajor reads bf16 rows in place with 16-byte loads (x
contiguous, 16-byte aligned, N % 8 == 0, offsets int32 contiguous), the path training runs.
StagedRowwiseKmajor takes every other input: the CTA copies its 128 x 128 tile into shared memory
with element loads, consecutive lanes along the input's unit-stride axis, and runs the same
arithmetic on the fp32 widening of each element, so it gives RowwiseKmajor's bytes and scales on
every bf16 input. On blocks without NaN both are bit-exact to the recipe (scale = the power of two
at or above max(amax, 1e-4) / 448, at least 2^-126; fp8 = cvt.rn.satfinite e4m3 of x times the
exact reciprocal). The amax is an unsigned max of the |x| bit patterns, where a NaN orders above
every number: a block holding NaN gets the scale bits 0x80000000 and its bytes from the reciprocal
0xFF000000, the same on both kernels and every dtype.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, ld_b32, ld_u16, ld_v4, lds_b32, lds_b32_pred, lds_v4, max_u16x2, prmt, selp, st_b32, st_u8, st_v2, sts_b32, sts_b32_nopred, f32_hex, u32_as_f32
from pithtrain.kernels.fp8_e8m0_scale_sm90 import FP8_MAX_RCP, e8m0_scale_bf16x2, quant4_e4m3
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad

__all__ = ["RowwiseKmajor", "StagedRowwiseKmajor", "THREADS", "TILE", "fp8cast_rowwise_kmajor_sm90", "supports_rowwise_kmajor"]

THREADS, TILE = 256, 128


@dsl_user_op
def quantize_4x4(words: list, row_rcp: list, col_rcp: list, *, loc=None, ip=None) -> tuple:
    """
    Quantize a 4 x 4 bf16 block for both outputs, widening each element to fp32 once.

    words[2 * row + half] holds columns 2 * half and 2 * half + 1 of that row. Returns four rowwise
    words (one row, columns 0..3, times row_rcp of the row) and four K-major words (one column, rows
    0..3, times col_rcp of the column); byte k of a word is its element k.
    cvt.rn.satfinite.e4m3x2.f32 is what Triton's .to(tl.float8e4nv) lowers to on SM90, which keeps
    the bytes identical.
    """
    ptx_lines = ["{", ".reg .b32 x<16>;", ".reg .f32 y<4>;", ".reg .b16 l, h;"]
    # x[row * 4 + col] = element (row, col) as fp32: a bf16 is the high half of its fp32
    for row in range(4):
        for half in range(2):
            word = f"${8 + row * 2 + half}"
            ptx_lines.append(f"shl.b32 x{row * 4 + 2 * half}, {word}, 16;")
            ptx_lines.append(f"and.b32 x{row * 4 + 2 * half + 1}, {word}, 0xFFFF0000;")
    for row in range(4):
        for col in range(4):
            ptx_lines.append(f"mul.rn.f32 y{col}, x{row * 4 + col}, ${16 + row};")
        ptx_lines += ["cvt.rn.satfinite.e4m3x2.f32 l, y1, y0;", "cvt.rn.satfinite.e4m3x2.f32 h, y3, y2;", f"mov.b32 ${row}, {{l, h}};"]
    for col in range(4):
        for row in range(4):
            ptx_lines.append(f"mul.rn.f32 y{row}, x{row * 4 + col}, ${20 + col};")
        ptx_lines += ["cvt.rn.satfinite.e4m3x2.f32 l, y1, y0;", "cvt.rn.satfinite.e4m3x2.f32 h, y3, y2;", f"mov.b32 ${4 + col}, {{l, h}};"]
    ptx_lines.append("}")
    result_type = llvm.StructType.get_literal([T.i32()] * 8)
    operands = [Uint32(value).ir_value(loc=loc, ip=ip) for value in list(words) + list(row_rcp) + list(col_rcp)]
    packed = asm(result_type, operands, "\n\t".join(ptx_lines), ",".join(["=r"] * 8 + ["r"] * 16))
    out = [Uint32(llvm.extractvalue(T.i32(), packed, [i], loc=loc, ip=ip)) for i in range(8)]
    return out[:4], out[4:]


class SubBlock:
    """
    Where one thread's 8 x 8 sub-block sits: in its warp, in the CTA's tile and in the input.
    """

    def __init__(self, thread_coords: tuple, pid_row, pid_col, N) -> None:
        self.lane, self.warp, self.col_vec_idx, self.row_block_idx = thread_coords
        # column vector and row block of the thread within its warp's 8 x 4 sub-blocks
        self.warp_col_vec_idx, self.warp_row_block_idx = self.lane % 8, self.lane // 8
        self.pid_row, self.pid_col = pid_row, pid_col
        self.row_start = Int64(pid_row) * TILE
        self.col0 = pid_col * TILE + self.col_vec_idx * 8
        self.col_ok = Int32(self.col0 < N)
        self.row0 = self.row_start + self.row_block_idx * 8


class RowwiseKmajor:
    """
    One 128 x 128 tile per CTA; 256 threads, each owning an 8 x 8 sub-block, so the K-major bytes of
    eight consecutive rows of a column come out without a transpose.

    Lane l of warp w owns column vector l % 8 + 8 * (w % 2) (8 columns) and row block l // 8 + 4 *
    (w // 2) (8 rows). The grid walks column blocks fastest, so co-resident CTAs cover whole rows of
    the input and outputs (and long runs of each K-major column), which keeps DRAM access local.
    """

    RED_WORDS = 64 * 6
    """
    Shared-memory words quantize_tile reduces through; subclasses (ScatterQuant) size theirs by it.
    """

    def __init__(self, N: int, num_groups: int) -> None:
        self.N = N
        self.num_col_tiles = (N + TILE - 1) // TILE
        self.n_pad = mn_pad(N)  # column stride of the MN-major per-channel scale
        self.num_groups = num_groups

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKmajor: cute.Tensor, mKmajorScale: cute.Tensor, stream) -> None:
        num_row_tiles = mX.shape[0] // TILE
        self.kernel(mX, mOffs, mTok, mTokScale, mKmajor, mKmajorScale).launch(grid=[self.num_col_tiles, num_row_tiles, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKmajor: cute.Tensor, mKmajorScale: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pid_col, pid_row, _ = cute.arch.block_idx()
        bases = (mX.iterator.toint(), mOffs.iterator.toint(), mTok.iterator.toint(), mTokScale.iterator.toint(), mKmajor.iterator.toint(), mKmajorScale.iterator.toint())
        smem = cutlass.memory.SmemAllocator()
        s_red = smem.allocate_tensor(Uint32, cute.make_layout(self.RED_WORDS), 16)
        rows = self.load_rows(bases[0], SubBlock(self.thread_coords(tidx), pid_row, pid_col, self.N))
        self.quantize_tile(rows, pid_row, pid_col, tidx, mX.shape[0], bases, s_red.iterator.toint())

    def thread_coords(self, tidx):
        lane, warp = tidx % 32, tidx // 32
        col_vec_idx, row_block_idx = lane % 8 + 8 * (warp % 2), lane // 8 + 4 * (warp // 2)
        return lane, warp, col_vec_idx, row_block_idx

    def load_rows(self, x_base, sub_block):
        """
        The 8 rows of the sub-block, four words (8 bf16) each, zero where its columns pass N.
        """
        return [ld_v4(x_base + ((sub_block.row0 + i) * self.N + sub_block.col0) * 2, sub_block.col_ok) for i in range(8)]

    def quantize_tile(self, rows, pid_row, pid_col, tidx, M, bases, s_red) -> None:
        """
        M is the column stride of the MN-major rowwise scale; s_red is the address of RED_WORDS
        words of shared memory.
        """
        _, offs_base, tok_base, tok_scale_base, kmajor_base, kmajor_scale_base = bases
        sub_block = SubBlock(self.thread_coords(tidx), pid_row, pid_col, self.N)
        group_start, num_group_rows = self.find_group(offs_base, sub_block)
        row_amax, col_amax = self.reduce_thread_amax(rows)
        row_amax, col_amax = self.reduce_tile_amax(row_amax, col_amax, sub_block, s_red)
        row_rcp, col_rcp = self.store_scales(row_amax, col_amax, tok_scale_base, kmajor_scale_base, M, sub_block)
        self.quantize_and_store(rows, row_rcp, col_rcp, tok_base, kmajor_base, group_start, num_group_rows, sub_block)

    def find_group_idx(self, offs_base, lane, row_start):
        """
        min(searchsorted(offs, row_start, right), G - 1): the group that holds the tile
        starting at row_start.
        """
        G = self.num_groups
        # each round a warp ballot counts the groups among 32 offsets that end at or before the tile
        num_groups_before = Int32(0)
        for i in range((G + 31) // 32):
            probe_idx = lane + i * 32
            probe_end = ld_b32(offs_base + Int64(probe_idx) * 4, Int32(probe_idx < G), Int32(0x7FFFFFFF))
            num_groups_before = num_groups_before + cute.arch.popc(cute.arch.vote_ballot_sync(Int64(probe_end) <= row_start))
        return cutlass.min(num_groups_before, G - 1)

    def find_group(self, offs_base, sub_block):
        group_idx = self.find_group_idx(offs_base, sub_block.lane, sub_block.row_start)
        group_end, group_start = Int64(ld_b32(offs_base + Int64(group_idx) * 4, Int32(1), Int32(0))), Int64(ld_b32(offs_base + Int64(group_idx - 1) * 4, Int32(group_idx > 0), Int32(0)))
        return group_start, group_end - group_start

    def reduce_thread_amax(self, rows):
        """
        Amax of the sub-block's row pairs and column pairs, as packed bf16 bit patterns:
        row_amax[pair] packs rows (2 * pair, 2 * pair + 1), col_amax[k] columns (2k, 2k + 1).
        """
        abs_rows = [[rows[i][k] & Uint32(0x7FFF7FFF) for k in range(4)] for i in range(8)]
        col_amax = []
        for k in range(4):
            acc_amax = abs_rows[0][k]
            for i in range(1, 8):
                acc_amax = max_u16x2(acc_amax, abs_rows[i][k])
            col_amax.append(acc_amax)
        row_amax = []
        for pair in range(4):
            amax_even, amax_odd = (max_u16x2(max_u16x2(abs_rows[i][0], abs_rows[i][1]), max_u16x2(abs_rows[i][2], abs_rows[i][3])) for i in (2 * pair, 2 * pair + 1))
            row_amax.append(max_u16x2(prmt(amax_even, amax_odd, 0x5410), prmt(amax_even, amax_odd, 0x7632)))
        return row_amax, col_amax

    def reduce_tile_amax(self, row_amax, col_amax, sub_block, s_red):
        """
        Reduce the per-thread amaxes over whole tile rows and columns, within the warp by butterfly
        and across warps through shared memory.
        """
        # lane bits 0-2 run along col_vec_idx, bits 3-4 along row_block_idx
        for offset in (1, 2, 4):
            row_amax = [max_u16x2(amax, cute.arch.shuffle_sync_bfly(amax, offset=offset)) for amax in row_amax]
        for offset in (8, 16):
            col_amax = [max_u16x2(amax, cute.arch.shuffle_sync_bfly(amax, offset=offset)) for amax in col_amax]
        # 2 warps share each row (along col_vec_idx), 4 share each column (along row_block_idx)
        s_row_amax = s_red
        s_col_amax = s_red + 64 * 2 * 4
        for pair in range(4):
            sts_b32(s_row_amax + ((sub_block.row_block_idx * 4 + pair) * 2 + sub_block.warp % 2) * 4, row_amax[pair], Int32(sub_block.warp_col_vec_idx == 0))
        for k in range(4):
            sts_b32(s_col_amax + ((sub_block.col_vec_idx * 4 + k) * 4 + sub_block.warp // 2) * 4, col_amax[k], Int32(sub_block.warp_row_block_idx == 0))
        cute.arch.sync_threads()
        row_amax, col_amax = [], []
        for pair in range(4):
            row_addr = s_row_amax + (sub_block.row_block_idx * 4 + pair) * 2 * 4
            row_amax.append(max_u16x2(lds_b32(row_addr), lds_b32(row_addr + 4)))
        for k in range(4):
            col_addr = s_col_amax + (sub_block.col_vec_idx * 4 + k) * 4 * 4
            acc_amax = lds_b32(col_addr)
            for i in range(1, 4):
                acc_amax = max_u16x2(acc_amax, lds_b32(col_addr + i * 4))
            col_amax.append(acc_amax)
        return row_amax, col_amax

    def store_scales(self, row_amax, col_amax, tok_scale_base, kmajor_scale_base, M, sub_block):
        """
        Store the E8M0 scales of the tile's 128 rows and 128 columns, each from one lane, and return
        the reciprocals of the scales of this thread's 8 rows and 8 columns.
        """
        # this lane scales row row_block_idx * 8 + warp_col_vec_idx (half warp_col_vec_idx % 2 of
        # row_amax[warp_col_vec_idx // 2]) and columns warp_row_block_idx and warp_row_block_idx + 4
        # of its column vector; shuffles then hand each thread the reciprocals it needs
        row_amax_pair = selp(Int32(sub_block.warp_col_vec_idx < 4), selp(Int32(sub_block.warp_col_vec_idx < 2), row_amax[0], row_amax[1]), selp(Int32(sub_block.warp_col_vec_idx < 6), row_amax[2], row_amax[3]))
        row_scale, row_rcp_lane = e8m0_scale_bf16x2(row_amax_pair, Int32(sub_block.warp_col_vec_idx % 2))
        col_scale_lo, col_rcp_lo = e8m0_scale_bf16x2(selp(Int32(sub_block.warp_row_block_idx < 2), col_amax[0], col_amax[1]), Int32(sub_block.warp_row_block_idx % 2))
        col_scale_hi, col_rcp_hi = e8m0_scale_bf16x2(selp(Int32(sub_block.warp_row_block_idx < 2), col_amax[2], col_amax[3]), Int32(sub_block.warp_row_block_idx % 2))
        row_rcp = [Uint32(cute.arch.shuffle_sync(Int32(row_rcp_lane), offset=(sub_block.lane & 24) | i)) for i in range(8)]
        col_rcp = [Uint32(cute.arch.shuffle_sync(Int32(col_rcp_lo if j < 4 else col_rcp_hi), offset=(j % 4) * 8 + sub_block.warp_col_vec_idx)) for j in range(8)]
        # MN-major scales: tok_scale[r, j] at j * M + r, kmajor_scale[c, b] at b * align(N, 4) + c
        st_b32(tok_scale_base + (Int64(sub_block.pid_col) * M + sub_block.row0 + sub_block.warp_col_vec_idx) * 4, row_scale, Int32(sub_block.warp % 2 == 0))
        for col, col_scale in ((sub_block.col0 + sub_block.warp_row_block_idx, col_scale_lo), (sub_block.col0 + sub_block.warp_row_block_idx + 4, col_scale_hi)):
            scale_addr = kmajor_scale_base + (Int64(sub_block.pid_row) * self.n_pad + col) * 4
            st_b32(scale_addr, col_scale, Int32(sub_block.warp // 2 == 0) & Int32(col < self.N))
        return row_rcp, col_rcp

    def quantize_and_store(self, rows, row_rcp, col_rcp, tok_base, kmajor_base, group_start, num_group_rows, sub_block) -> None:
        """
        Quantize the sub-block and store it rowwise into tok (M, N) and K-major into kmajor, where
        its group's block is (N, num_group_rows) at group_start * N.
        """
        N = self.N
        kmajor_block_base = kmajor_base + group_start * N + (sub_block.row0 - group_start)
        tok_words, kmajor_words = [[None, None] for _ in range(8)], [[None, None] for _ in range(8)]
        for row_half in range(2):
            for col_half in range(2):
                words = [rows[4 * row_half + row][2 * col_half + half] for row in range(4) for half in range(2)]
                tok_quad, kmajor_quad = quantize_4x4(words, row_rcp[4 * row_half : 4 * row_half + 4], col_rcp[4 * col_half : 4 * col_half + 4])
                for row in range(4):
                    tok_words[4 * row_half + row][col_half] = tok_quad[row]
                for col in range(4):
                    kmajor_words[4 * col_half + col][row_half] = kmajor_quad[col]
            for row in range(4):
                i = 4 * row_half + row
                st_v2(tok_base + (sub_block.row0 + i) * N + sub_block.col0, tok_words[i][0], tok_words[i][1], sub_block.col_ok)
        for j in range(8):
            st_v2(kmajor_block_base + Int64(sub_block.col0 + j) * num_group_rows, kmajor_words[j][0], kmajor_words[j][1], sub_block.col_ok)


@dsl_user_op
def max_u32(first: Uint32, second: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    Unsigned max: the float max of non-negative fp32 bit patterns, a NaN above every number.
    """
    return Uint32(asm(T.i32(), [Uint32(first).ir_value(loc=loc, ip=ip), Uint32(second).ir_value(loc=loc, ip=ip)], "max.u32 $0, $1, $2;", "=r,r,r"))


@dsl_user_op
def f16x2_to_f32(word: Uint32, *, loc=None, ip=None) -> tuple:
    """
    The fp32 bit patterns of the low and high fp16 halves of word (cvt.f32.f16, exact).
    """
    pair_type = llvm.StructType.get_literal([T.i32(), T.i32()])
    pair = asm(pair_type, [Uint32(word).ir_value(loc=loc, ip=ip)], "{\n\t.reg .b16 l, h;\n\t.reg .f32 a, b;\n\tmov.b32 {l, h}, $2;\n\tcvt.f32.f16 a, l;\n\tcvt.f32.f16 b, h;\n\tmov.b32 $0, a;\n\tmov.b32 $1, b;\n\t}", "=r,=r,r")
    return Uint32(llvm.extractvalue(T.i32(), pair, [0], loc=loc, ip=ip)), Uint32(llvm.extractvalue(T.i32(), pair, [1], loc=loc, ip=ip))


@dsl_user_op
def prmt_dynamic(word_lo: Uint32, word_hi: Uint32, selector: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    prmt.b32 with a selector computed at run time.
    """
    return Uint32(asm(T.i32(), [Uint32(word_lo).ir_value(loc=loc, ip=ip), Uint32(word_hi).ir_value(loc=loc, ip=ip), Uint32(selector).ir_value(loc=loc, ip=ip)], "prmt.b32 $0, $1, $2, $3;", "=r,r,r,r"))


@dsl_user_op
def e8m0_scale_bits(amax: Uint32, *, loc=None, ip=None) -> tuple:
    """
    e8m0_scale_bf16x2's arithmetic on the fp32 bit pattern of a non-negative amax: (scale,
    reciprocal) bit patterns. An amax that is a NaN gives (0x80000000, 0xFF000000), as there.
    """
    pair_type = llvm.StructType.get_literal([T.i32(), T.i32()])
    ptx = "{\n\t.reg .b32 a, m, s;\n\t.reg .f32 f;\n\t.reg .pred p;\n\t" f"max.u32 a, $2, {f32_hex(1e-4)};\n\t" "mov.b32 f, a;\n\t" f"mul.rn.f32 f, f, {FP8_MAX_RCP};\n\t" "mov.b32 a, f;\n\tand.b32 m, a, 0x007FFFFF;\n\tand.b32 s, a, 0x7F800000;\n\tsetp.ne.b32 p, m, 0;\n\t@p add.u32 s, s, 0x00800000;\n\tmax.u32 s, s, 0x00800000;\n\tmov.b32 $0, s;\n\tsub.u32 $1, 0x7F000000, s;\n\t}"
    pair = asm(pair_type, [Uint32(amax).ir_value(loc=loc, ip=ip)], ptx, "=r,=r,r")
    return Uint32(llvm.extractvalue(T.i32(), pair, [0], loc=loc, ip=ip)), Uint32(llvm.extractvalue(T.i32(), pair, [1], loc=loc, ip=ip))


class StagedRowwiseKmajor(RowwiseKmajor):
    """
    RowwiseKmajor for every other input: bf16, fp16 or fp32 (elem_bytes 2 or 4), any strides, any
    base alignment, any N. Same grid, threads and 8 x 8 sub-blocks.

    The CTA copies its tile into shared memory first (row-major, or column-major when transposed),
    each warp instruction reading 64 consecutive elements along one tile row (one tile column when
    transposed, for inputs whose unit stride runs along M) with element loads. Each thread then
    reads its sub-block back, widens it to fp32 bit patterns and reduces the amaxes with max_u32,
    which orders the widened bits exactly as max_u16x2 orders bf16 ones; e8m0_scale_bits and
    quant4_e4m3 are RowwiseKmajor's scale and quantization arithmetic. K-major columns and scales
    are stored per column. Rowwise rows are stored from registers in 8- or 4-byte pieces when N
    allows; otherwise (N % 4 != 0) the CTA stages its rowwise fp8 tile in shared memory and each
    warp writes whole row segments as aligned 4-byte words, bytes only at their ends.
    """

    RED_WORDS = 128 * 2 + 128 * 4

    def __init__(self, N: int, num_groups: int, kind: str, transposed: bool) -> None:
        super().__init__(N, num_groups)
        self.kind, self.transposed = kind, transposed
        self.elem_bytes = 4 if kind == "f32" else 2
        self.tok_width = 8 if N % 8 == 0 else 4 if N % 4 == 0 else 2 if N % 2 == 0 else 1

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKmajor: cute.Tensor, mKmajorScale: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pid_col, pid_row, _ = cute.arch.block_idx()
        bases = (mX.iterator.toint(), mOffs.iterator.toint(), mTok.iterator.toint(), mTokScale.iterator.toint(), mKmajor.iterator.toint(), mKmajorScale.iterator.toint())
        smem = cutlass.memory.SmemAllocator()
        s_tile = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(TILE * TILE * self.elem_bytes), 128)
        s_red = smem.allocate_tensor(Uint32, cute.make_layout(self.RED_WORDS), 16)
        sub_block = SubBlock(self.thread_coords(tidx), pid_row, pid_col, self.N)
        self.stage_tile(bases[0], Int64(mX.stride[0]), Int64(mX.stride[1]), sub_block, s_tile.iterator.toint())
        cute.arch.sync_threads()
        values = self.read_sub_block(s_tile.iterator.toint(), sub_block)
        self.quantize_values(values, sub_block, mX.shape[0], bases, s_red.iterator.toint(), s_tile.iterator.toint())

    def stage_tile(self, x_base, S0, S1, sub_block, s_tile) -> None:
        """
        Copy the CTA's tile of x (row r of the tile at r * S0, column c at c * S1 elements) into
        s_tile, zero past column N; warp w copies tile lines 16w..16w+15, two elements per lane per
        pass, consecutive lanes along the line.
        """
        elem_bytes, lane, warp = self.elem_bytes, sub_block.lane, sub_block.warp
        row_start, col_start = sub_block.row_start, sub_block.pid_col * TILE
        for r in range(16):
            line = warp * 16 + r
            for pass_idx in range(2):
                along = 64 * pass_idx + 2 * lane
                words = []
                for half in range(2):
                    if self.transposed:
                        row, col = row_start + along + half, col_start + line
                    else:
                        row, col = row_start + line, col_start + along + half
                    load_ok = Int32(col < self.N)
                    addr = x_base + (row * S0 + Int64(col) * S1) * elem_bytes
                    words.append(Uint32(ld_b32(addr, load_ok, Int32(0))) if self.elem_bytes == 4 else ld_u16(addr, load_ok))
                s_addr = s_tile + (line * TILE + along) * elem_bytes
                if self.elem_bytes == 4:
                    sts_b32_nopred(s_addr, words[0])
                    sts_b32_nopred(s_addr + 4, words[1])
                else:
                    sts_b32_nopred(s_addr, words[0] | (words[1] << 16))

    def read_sub_block(self, s_tile, sub_block) -> list:
        """
        The thread's 8 x 8 sub-block from s_tile as fp32 bit patterns: values[i][j] is row i,
        column j.
        """
        elem_bytes = self.elem_bytes
        lines = []
        for k in range(8):
            if self.transposed:
                s_addr = s_tile + ((sub_block.col_vec_idx * 8 + k) * TILE + sub_block.row_block_idx * 8) * elem_bytes
            else:
                s_addr = s_tile + ((sub_block.row_block_idx * 8 + k) * TILE + sub_block.col_vec_idx * 8) * elem_bytes
            words = list(lds_v4(s_addr)) + (list(lds_v4(s_addr + 16)) if elem_bytes == 4 else [])
            if self.kind == "f32":
                lines.append(words)
            elif self.kind == "bf16":
                lines.append([word << 16 if half == 0 else word & Uint32(0xFFFF0000) for word in words for half in range(2)])
            else:
                lines.append([bits for word in words for bits in f16x2_to_f32(word)])
        if self.transposed:
            return [[lines[j][i] for j in range(8)] for i in range(8)]
        return lines

    def quantize_values(self, values, sub_block, M, bases, s_red, s_tile) -> None:
        """
        s_tile is free once reduce_amax's barrier has passed (every thread holds its values), so the
        staged rowwise store reuses it.
        """
        _, offs_base, tok_base, tok_scale_base, kmajor_base, kmajor_scale_base = bases
        group_start, num_group_rows = self.find_group(offs_base, sub_block)
        abs_values = [[values[i][j] & Uint32(0x7FFFFFFF) for j in range(8)] for i in range(8)]
        row_amax, col_amax = self.reduce_amax(abs_values, sub_block, s_red)
        row_rcp, col_rcp = self.store_scales_per_line(row_amax, col_amax, tok_scale_base, kmajor_scale_base, M, sub_block)
        self.quantize_and_store_values(values, row_rcp, col_rcp, tok_base, kmajor_base, group_start, num_group_rows, sub_block, s_tile)

    def reduce_amax(self, abs_values, sub_block, s_red) -> tuple:
        """
        Amax bit patterns of the thread's 8 rows and 8 columns over the whole tile: within the warp
        by butterfly, across warps through shared memory (2 warps share each row, 4 each column).
        """
        row_amax, col_amax = [], []
        for i in range(8):
            acc_amax = abs_values[i][0]
            for j in range(1, 8):
                acc_amax = max_u32(acc_amax, abs_values[i][j])
            row_amax.append(acc_amax)
        for j in range(8):
            acc_amax = abs_values[0][j]
            for i in range(1, 8):
                acc_amax = max_u32(acc_amax, abs_values[i][j])
            col_amax.append(acc_amax)
        for offset in (1, 2, 4):
            row_amax = [max_u32(amax, Uint32(cute.arch.shuffle_sync_bfly(Int32(amax), offset=offset))) for amax in row_amax]
        for offset in (8, 16):
            col_amax = [max_u32(amax, Uint32(cute.arch.shuffle_sync_bfly(Int32(amax), offset=offset))) for amax in col_amax]
        s_row_amax, s_col_amax = s_red, s_red + 128 * 2 * 4
        for i in range(8):
            sts_b32(s_row_amax + ((sub_block.row_block_idx * 8 + i) * 2 + sub_block.warp % 2) * 4, row_amax[i], Int32(sub_block.warp_col_vec_idx == 0))
        for j in range(8):
            sts_b32(s_col_amax + ((sub_block.col_vec_idx * 8 + j) * 4 + sub_block.warp // 2) * 4, col_amax[j], Int32(sub_block.warp_row_block_idx == 0))
        cute.arch.sync_threads()
        row_amax, col_amax = [], []
        for i in range(8):
            row_addr = s_row_amax + (sub_block.row_block_idx * 8 + i) * 2 * 4
            row_amax.append(max_u32(lds_b32(row_addr), lds_b32(row_addr + 4)))
        for j in range(8):
            col_addr = s_col_amax + (sub_block.col_vec_idx * 8 + j) * 4 * 4
            acc_amax = lds_b32(col_addr)
            for k in range(1, 4):
                acc_amax = max_u32(acc_amax, lds_b32(col_addr + k * 4))
            col_amax.append(acc_amax)
        return row_amax, col_amax

    def store_scales_per_line(self, row_amax, col_amax, tok_scale_base, kmajor_scale_base, M, sub_block) -> tuple:
        """
        Store the E8M0 scales of the thread's rows and columns (row i from the thread whose
        col_vec_idx is i, column j from the one whose row_block_idx is j) and return the
        reciprocals of all 8 of each.
        """
        row_rcp, col_rcp = [], []
        for i in range(8):
            scale, rcp = e8m0_scale_bits(row_amax[i])
            st_b32(tok_scale_base + (Int64(sub_block.pid_col) * M + sub_block.row0 + i) * 4, scale, Int32(sub_block.col_vec_idx == i))
            row_rcp.append(rcp)
        for j in range(8):
            scale, rcp = e8m0_scale_bits(col_amax[j])
            col = sub_block.col0 + j
            st_b32(kmajor_scale_base + (Int64(sub_block.pid_row) * self.n_pad + col) * 4, scale, Int32(sub_block.row_block_idx == j) & Int32(col < self.N))
            col_rcp.append(rcp)
        return row_rcp, col_rcp

    def quantize_and_store_values(self, values, row_rcp, col_rcp, tok_base, kmajor_base, group_start, num_group_rows, sub_block, s_tile) -> None:
        N, width = self.N, self.tok_width
        for i in range(8):
            rcp = (u32_as_f32(row_rcp[i]),) * 4
            words = [quant4_e4m3(*[u32_as_f32(values[i][4 * half + k]) for k in range(4)], rcp) for half in range(2)]
            row_base = tok_base + (sub_block.row0 + i) * N + sub_block.col0
            if width == 8:
                st_v2(row_base, words[0], words[1], sub_block.col_ok)
            elif width == 4:
                for half in range(2):
                    st_b32(row_base + 4 * half, words[half], Int32(sub_block.col0 + 4 * half < N))
            else:
                s_addr = s_tile + (sub_block.row_block_idx * 8 + i) * TILE + sub_block.col_vec_idx * 8
                sts_b32_nopred(s_addr, words[0])
                sts_b32_nopred(s_addr + 4, words[1])
        if width < 4:
            cute.arch.sync_threads()
            self.store_staged_rows(s_tile, tok_base, sub_block)
        kmajor_block_base = kmajor_base + group_start * N + (sub_block.row0 - group_start)
        for j in range(8):
            rcp = (u32_as_f32(col_rcp[j]),) * 4
            words = [quant4_e4m3(*[u32_as_f32(values[4 * half + k][j]) for k in range(4)], rcp) for half in range(2)]
            st_v2(kmajor_block_base + Int64(sub_block.col0 + j) * num_group_rows, words[0], words[1], Int32(sub_block.col0 + j < N))

    def store_staged_rows(self, s_q, tok_base, sub_block) -> None:
        """
        Store the rowwise fp8 tile staged in s_q (128 bytes per tile row): warp w writes tile rows
        16w..16w+15, each row segment (its L = min(128, N - 128 * pid_col) bytes at r * N + 128 *
        pid_col of tok) as the aligned 4-byte words that cover it, word j from lane j (and word 32
        from lane 0), funnel-shifted out of two staged words; a word the segment covers only in part
        is stored byte by byte.
        """
        lane, warp = sub_block.lane, sub_block.warp
        col_start = sub_block.pid_col * TILE
        seg_len = cutlass.min(Int32(self.N) - col_start, Int32(TILE))
        for r in range(16):
            tile_row = warp * 16 + r
            seg_start = tok_base + (sub_block.row_start + tile_row) * self.N + col_start
            shift = Int32(seg_start & 3)
            selector = Uint32(0x7654) - Uint32(shift) * Uint32(0x1111)
            for extra in range(2):
                word_idx = lane + 32 * extra
                word_ok = Int32(word_idx <= 32) if extra else Int32(1)
                first = word_idx * 4 - shift  # segment offset of the word's first byte
                s_row = s_q + tile_row * TILE
                prev = Uint32(lds_b32_pred(s_row + word_idx * 4 - 4, word_ok & Int32(word_idx > 0), Int32(0)))
                cur = Uint32(lds_b32_pred(s_row + word_idx * 4, word_ok & Int32(word_idx < 32), Int32(0)))
                word = prmt_dynamic(prev, cur, selector)
                addr = seg_start - Int64(shift) + word_idx * 4
                full = word_ok & Int32(first >= 0) & Int32(first + 3 < seg_len)
                st_b32(addr, word, full)
                for k in range(4):
                    byte_ok = word_ok & (full ^ 1) & Int32(first + k >= 0) & Int32(first + k < seg_len)
                    st_u8(addr + k, word >> (8 * k), byte_ok)


COMPILED: dict = {}


def compile_kernel(N: int, num_groups: int) -> Callable:
    M, num_kmajor_elems, num_row_tiles = cute.sym_int(), cute.sym_int(), cute.sym_int()
    num_col_tiles = (N + TILE - 1) // TILE
    x, offs, tok = fake(cutlass.BFloat16, (M, N), (N, 1)), fake(Int32, (num_groups,), (1,), 4), fake(cutlass.Float8E4M3FN, (M, N), (N, 1))
    tok_scale = fake(cutlass.Float32, (M, num_col_tiles), (1, cute.sym_int64()))
    kmajor = fake(cutlass.Float8E4M3FN, (num_kmajor_elems,), (1,))
    kmajor_scale = fake(cutlass.Float32, (N, num_row_tiles), (1, mn_pad(N)))
    return cute.compile(RowwiseKmajor(N, num_groups), x, offs, tok, tok_scale, kmajor, kmajor_scale, fake_stream(), options="--enable-tvm-ffi")


def allocate_outputs(x: torch.Tensor, M: int, N: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    return (torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), mn_major_scale(M, (N + TILE - 1) // TILE, device), torch.empty(M * N, dtype=torch.float8_e4m3fn, device=device), mn_major_scale(N, M // TILE, device))


def compile_staged_kernel(N: int, num_groups: int, kind: str, transposed: bool) -> Callable:
    M, num_kmajor_elems, num_row_tiles = cute.sym_int(), cute.sym_int(), cute.sym_int()
    num_col_tiles = (N + TILE - 1) // TILE
    dtype = STAGED_TYPES[kind]
    x, offs, tok = fake(dtype, (M, N), (cute.sym_int64(), cute.sym_int64()), dtype.width // 8), fake(Int32, (num_groups,), (1,), 4), fake(cutlass.Float8E4M3FN, (M, N), (N, 1))
    tok_scale = fake(cutlass.Float32, (M, num_col_tiles), (1, cute.sym_int64()))
    kmajor = fake(cutlass.Float8E4M3FN, (num_kmajor_elems,), (1,))
    kmajor_scale = fake(cutlass.Float32, (N, num_row_tiles), (1, mn_pad(N)))
    return cute.compile(StagedRowwiseKmajor(N, num_groups, kind, transposed), x, offs, tok, tok_scale, kmajor, kmajor_scale, fake_stream(), options="--enable-tvm-ffi")


STAGED_TYPES = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16, "f32": cutlass.Float32}
INPUT_KINDS = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
INT32_MAX = 2**31 - 1


def supports_rowwise_kmajor(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> bool:
    """
    Whether the inputs take RowwiseKmajor, the 16-byte-load kernel training runs; every other input
    takes StagedRowwiseKmajor (after converting the offsets to contiguous int32).
    """
    return not (x.dtype != torch.bfloat16 or x.shape[1] % 8 != 0 or not x.is_contiguous() or x.data_ptr() % 16 != 0 or grouped_mm_offs.dtype != torch.int32 or grouped_mm_offs.shape[0] == 0 or not grouped_mm_offs.is_contiguous())


def fp8cast_rowwise_kmajor_sm90(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    fp8cast_rowwise_kmajor on SM90. x is (M, N) bf16, fp16 or fp32 with M % 128 == 0, any strides
    and base alignment (not modified); grouped_mm_offs is int32 or int64 (G,), any stride, the
    cumulative group ends (last = M, each a multiple of 128). Another dtype raises TypeError. M == 0
    or N == 0 returns empty outputs for any G; otherwise G == 0, or M > 2^31 - 1, raises ValueError.
    Both scales are MN-major. Runs on the current stream.
    """
    assert x.ndim == 2 and x.shape[0] % 128 == 0, f"Expected 2D tensor with M%128==0, got shape {x.shape}"
    M, N = x.shape
    if M == 0 or N == 0:
        if x.dtype not in INPUT_KINDS:
            raise TypeError(f"fp8cast_rowwise_kmajor takes bfloat16, float16 or float32 input, got {x.dtype}")
        return allocate_outputs(x, M, N)
    G = grouped_mm_offs.shape[0]
    if supports_rowwise_kmajor(x, grouped_mm_offs) and M <= INT32_MAX:
        out = allocate_outputs(x, M, N)
        key = (N, G)
        fn = COMPILED.get(key)
        if fn is None:
            fn = COMPILED[key] = compile_kernel(N, G)
        fn(x, grouped_mm_offs, *out)
        return out
    if G == 0:
        raise ValueError(f"fp8cast_rowwise_kmajor: {M} rows need group offsets ending at M, got none")
    if M > INT32_MAX:
        raise ValueError(f"fp8cast_rowwise_kmajor: M = {M} rows do not fit the int32 group offsets")
    kind = INPUT_KINDS.get(x.dtype)
    if kind is None:
        raise TypeError(f"fp8cast_rowwise_kmajor takes bfloat16, float16 or float32 input, got {x.dtype}")
    if grouped_mm_offs.dtype != torch.int32 or not grouped_mm_offs.is_contiguous():
        grouped_mm_offs = grouped_mm_offs.to(torch.int32).contiguous()
        if supports_rowwise_kmajor(x, grouped_mm_offs):
            return fp8cast_rowwise_kmajor_sm90(x, grouped_mm_offs)
    S0, S1 = x.stride()
    # consecutive lanes run along the smaller stride; a single column is read down its rows
    transposed = N == 1 or S0 < S1
    out = allocate_outputs(x, M, N)
    key = (N, G, kind, transposed)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_staged_kernel(N, G, kind, transposed)
    fn(x, grouped_mm_offs, *out)
    return out
