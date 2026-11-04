"""
segment_sum and block_segment_sum in CuTe-DSL for SM90, launched without a host synchronization.

segment_sum runs two kernels from one call. TileSum cuts every group into 128-row tiles from its own
start on the device (each CTA scans the offsets into shared memory) and deals the (tile, column
chunk) items round-robin to persistent CTAs, so no group size serializes the GPU; each of a CTA's
8 warps adds 16 of the tile's rows, one at a time in increasing row order from zero, and a fixed
pairwise tree adds the 8 block sums into the fp32 tile sum. Fold then adds each group's tile sums
in row order, starting from zero, one thread per (group, column), and rounds once to the output
dtype. block_segment_sum is Fold over the given 128-row block partials.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, bf16_lo, bf16x2_to_f32, f32_as_u32, ld_b32, lds_b32, selp, st_b32, st_f32, sts_b32, u32_as_f32

__all__ = ["block_segment_sum_sm90", "segment_sum_sm90"]

TILE = 128  # rows per tile
THREADS, CTAS_PER_SM = 256, 2  # TileSum
WARPS = THREADS // 32
ROWS_PER_WARP = TILE // WARPS
FOLD_THREADS, FOLD_UNROLL, FOLD_DEEP, FOLD_CTAS_PER_SM = 128, 16, 64, 4  # Fold: rows in flight per thread


def pairwise_sum(values: list):
    """
    ((v0 + v1) + (v2 + v3)) + ... over a power-of-two count of values.
    """
    while len(values) > 1:
        values = [values[2 * i] + values[2 * i + 1] for i in range(len(values) // 2)]
    return values[0]


def add_in_order(acc_sum, words: list):
    """
    acc_sum plus the fp32 values of words, one at a time in order.
    """
    for word in words:
        acc_sum = acc_sum + u32_as_f32(Uint32(word))
    return acc_sum


class Fold:
    """
    out[g, c] = fp32 sum of partials[r, c] over the rows r of group g, one at a time in increasing
    row order from zero, rounded once; one thread per (group, column), FOLD_THREADS columns per
    CTA. The rows of group g are [offs[g - 1] / 128, offs[g] / 128) with from_blocks
    (block_segment_sum, short groups), else [tile_start[g], tile_start[g + 1]).
    """

    def __init__(self, out_dtype: type, from_blocks: bool, offs_bytes: int, offs_signed: bool) -> None:
        self.out_dtype, self.from_blocks, self.offs_bytes, self.offs_signed = out_dtype, from_blocks, offs_bytes, offs_signed

    def launch_fold(self, mPartials: cute.Tensor, mRanges: cute.Tensor, mOut: cute.Tensor, stream) -> None:
        G, H = mOut.shape[0], mOut.shape[1]
        # the deep batches need the registers of FOLD_CTAS_PER_SM resident CTAs
        min_ctas = 0 if cutlass.const_expr(self.from_blocks) else FOLD_CTAS_PER_SM
        self.fold_kernel(mPartials, mRanges, mOut).launch(grid=[(H + FOLD_THREADS - 1) // FOLD_THREADS, G, 1], block=[FOLD_THREADS, 1, 1], stream=stream, min_blocks_per_mp=min_ctas)

    @cute.kernel
    def fold_kernel(self, mPartials: cute.Tensor, mRanges: cute.Tensor, mOut: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, group_idx, _ = cute.arch.block_idx()
        H = mOut.shape[1]
        col_idx = bx * FOLD_THREADS + tidx
        col_ok = Int32(col_idx < H)
        row_start, row_end = self.group_rows((mRanges.iterator.toint(), Int64(mRanges.stride[0]) * self.offs_bytes), group_idx, mPartials.shape[0])
        # a lane past H reads column 0 and stores nothing
        acc_sum = self.fold_rows(mPartials.iterator.toint() + Int64(col_idx * col_ok) * 4, Int64(H) * 4, row_start, row_end, col_ok)
        self.store_rounded(mOut.iterator.toint() + (Int64(group_idx) * H + col_idx) * (self.out_dtype.width // 8), acc_sum, col_ok)

    @cute.jit
    def group_rows(self, ranges, group_idx, num_rows):
        """
        The partial rows [row_start, row_end) of group group_idx, within [0, num_rows).
        """
        ranges_base, _ = ranges
        row_start, row_end = Int32(0), Int32(0)
        if cutlass.const_expr(self.from_blocks):
            group_end, prev_end = self.load_offset(ranges, group_idx, Int32(1), num_rows * TILE), self.load_offset(ranges, group_idx - 1, Int32(group_idx > 0), num_rows * TILE)
            row_start, row_end = prev_end // TILE, cutlass.max(group_end, prev_end) // TILE
        else:
            row_start, row_end = ld_b32(ranges_base + Int64(group_idx) * 4, Int32(1), Int32(0)), ld_b32(ranges_base + Int64(group_idx + 1) * 4, Int32(1), Int32(0))
        return row_start, row_end

    def load_offset(self, offs, group_idx, pred, M):
        """
        offs[group_idx] clamped to [0, M] (0 where not pred), compared in 64 bits with the offsets'
        signedness, so that an end past the int32 range clamps instead of wrapping.
        """
        offs_base, offs_stride_bytes = offs
        addr = offs_base + Int64(group_idx) * offs_stride_bytes
        if cutlass.const_expr(self.offs_bytes == 4 and self.offs_signed):
            return cutlass.min(cutlass.max(ld_b32(addr, pred, Int32(0)), Int32(0)), M)
        value = Int64(0)
        if cutlass.const_expr(self.offs_bytes == 4):
            value = Int64(ld_b32(addr, pred, Int32(0))) & Int64(0xFFFFFFFF)
        else:
            value = Int64(asm(T.i64(), [Int64(addr).ir_value(), Int32(pred).ir_value()], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b64 $0, 0;\n\t@p ld.global.nc.b64 $0, [$1];\n\t}", "=l,l,r"))
        clamped = cutlass.min(cutlass.max(value, Int64(0)), Int64(M))
        if cutlass.const_expr(not self.offs_signed and self.offs_bytes == 8):
            # a uint64 at or above 2^63 reads negative: it is past M
            clamped = clamped + (Int64(M) - clamped) * Int64(value < Int64(0))
        return Int32(clamped)

    @cute.jit
    def fold_rows(self, col_base, row_bytes, row_start, row_end, col_ok):
        """
        Without from_blocks, full batches of FOLD_DEEP rows first, with unpredicated loads so that
        ptxas keeps a whole batch in flight (a long group's tile sums); then predicated batches of
        FOLD_UNROLL. The adds stay in row order throughout.
        """
        acc_sum = Float32(0.0)
        deep_end = row_start
        if cutlass.const_expr(not self.from_blocks):
            deep_end = row_start + (row_end - row_start) // FOLD_DEEP * FOLD_DEEP
            for row_idx in cutlass.range(row_start, deep_end, FOLD_DEEP):
                words = [asm(T.i32(), [Int64(col_base + Int64(row_idx + i) * row_bytes).ir_value()], "ld.global.nc.b32 $0, [$1];", "=r,l") for i in range(FOLD_DEEP)]
                acc_sum = add_in_order(acc_sum, words)
        for row_idx in cutlass.range(deep_end, row_end, FOLD_UNROLL):
            values = [ld_b32(col_base + Int64(row_idx + i) * row_bytes, col_ok & Int32(row_idx + i < row_end), Int32(0)) for i in range(FOLD_UNROLL)]
            # masked rows read +0.0, which leaves the sum unchanged: it starts at +0.0, so it is never -0.0
            acc_sum = add_in_order(acc_sum, values)
        return acc_sum

    def store_rounded(self, addr, value, pred) -> None:
        if cutlass.const_expr(self.out_dtype is cutlass.Float32):
            st_f32(addr, value, pred)
        else:
            convert_instr = "cvt.rn.bf16.f32" if cutlass.const_expr(self.out_dtype is cutlass.BFloat16) else "cvt.rn.f16.f32"
            asm(None, [Int64(addr).ir_value(), Float32(value).ir_value(), Int32(pred).ir_value()], "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.b32 p, $2, 0;\n\t" + convert_instr + " h, $1;\n\t@p st.global.b16 [$0], h;\n\t}", "l,f,r", side_effects=True)


class BlockSegmentSum(Fold):
    def __init__(self, out_dtype: type, offs_bytes: int, offs_signed: bool) -> None:
        super().__init__(out_dtype, True, offs_bytes, offs_signed)

    @cute.jit
    def __call__(self, mPartials: cute.Tensor, mOffs: cute.Tensor, mOut: cute.Tensor, stream) -> None:
        self.launch_fold(mPartials, mOffs, mOut, stream)


class SegmentSum(Fold):
    """
    TileSum, then Fold. TileSum runs P = SM_COUNT * CTAS_PER_SM persistent CTAs of WARPS warps;
    CTA b takes items b, b + P, b + 2P, ... of the num_tiles * num_chunks items, item i being
    column chunk i % num_chunks (32 vectors of vec_bytes) of tile i / num_chunks, so that the CTAs
    running together read whole rows. Lane l of warp w adds column vector 32 * chunk + l over rows
    [16 w, 16 w + 16) of the tile one at a time in increasing row order from zero, and the WARPS
    block sums are added by a fixed pairwise tree, ((b0 + b1) + (b2 + b3)) + ((b4 + b5) +
    (b6 + b7)). Tile t's fp32 sums go to row t of tile_sums, and tile_start gets the number of
    tiles before each group (G + 1 entries).
    """

    def __init__(self, x_dtype: type, vec_bytes: int, max_groups: int, out_dtype: type, offs_bytes: int, offs_signed: bool) -> None:
        super().__init__(out_dtype, False, offs_bytes, offs_signed)
        self.x_dtype, self.vec_bytes, self.max_groups = x_dtype, vec_bytes, max_groups
        self.elem_bytes = x_dtype.width // 8
        self.vec_elems = vec_bytes // self.elem_bytes

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOffs: cute.Tensor, mTileSums: cute.Tensor, mTileStart: cute.Tensor, mOut: cute.Tensor, stream) -> None:
        self.tile_sum_kernel(mX, mOffs, mTileSums, mTileStart).launch(grid=[SM_COUNT * CTAS_PER_SM, 1, 1], block=[THREADS, 1, 1], stream=stream, min_blocks_per_mp=CTAS_PER_SM)
        self.launch_fold(mTileSums, mTileStart, mOut, stream)

    @cute.kernel
    def tile_sum_kernel(self, mX: cute.Tensor, mOffs: cute.Tensor, mTileSums: cute.Tensor, mTileStart: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        lane, warp = tidx % 32, tidx // 32
        M, H, G = mX.shape[0], mX.shape[1], mOffs.shape[0]
        smem = cutlass.memory.SmemAllocator()
        s_tile_start, s_row_start, s_row_end = (smem.allocate((self.max_groups + 1) * 4, byte_alignment=16).toint() for _ in range(3))
        s_warp_total = smem.allocate(WARPS * 4, byte_alignment=16).toint()
        s_blocks = smem.allocate(2 * WARPS * 32 * self.vec_elems * 4, byte_alignment=16).toint()  # (buffer, warp, lane, element)

        self.scan_groups((mOffs.iterator.toint(), Int64(mOffs.stride[0]) * self.offs_bytes), G, M, tidx, lane, warp, (s_tile_start, s_row_start, s_row_end, s_warp_total))
        num_tiles = Int32(lds_b32(s_tile_start + G * 4))
        if bx == 0:
            self.store_tile_start(mTileStart.iterator.toint(), G, tidx, s_tile_start)
        self.sum_items(mX.iterator.toint(), mTileSums.iterator.toint(), H, G, num_tiles, tidx, bx, num_ctas, (s_tile_start, s_row_start, s_row_end), s_blocks)

    @cute.jit
    def scan_groups(self, offs, G, M, tidx, lane, warp, s_tables) -> None:
        """
        Cut every group into tiles: s_tile_start[g] = tiles before group g (G + 1 entries), and the
        group's rows [s_row_start[g], s_row_end[g]) with its ends clamped to [0, M].
        """
        s_tile_start, s_row_start, s_row_end, s_warp_total = s_tables
        num_before = Int32(0)
        for round_start in cutlass.range(0, G, THREADS):
            group_idx = round_start + tidx
            group_ok = Int32(group_idx < G)
            group_end, prev_end = self.load_offset(offs, group_idx, group_ok, M), self.load_offset(offs, group_idx - 1, group_ok & Int32(group_idx > 0), M)
            row_start, row_end = prev_end, cutlass.max(group_end, prev_end)
            num_group_tiles = (row_end - row_start + TILE - 1) // TILE
            tiles_through = num_group_tiles
            for level in cutlass.range_constexpr(5):
                tiles_below = cute.arch.shuffle_sync_up(tiles_through, offset=1 << level, mask_and_clamp=0)
                tiles_through = tiles_through + tiles_below * Int32(lane >= (1 << level))
            sts_b32(s_warp_total + warp * 4, Uint32(tiles_through), Int32(lane == 31))
            cute.arch.sync_threads()
            warp_before, round_total = Int32(0), Int32(0)
            for other_warp in cutlass.range_constexpr(WARPS):
                warp_total = Int32(lds_b32(s_warp_total + other_warp * 4))
                warp_before, round_total = warp_before + warp_total * Int32(other_warp < warp), round_total + warp_total
            safe_idx = cutlass.min(group_idx, G - 1)
            sts_b32(s_tile_start + (safe_idx + 1) * 4, Uint32(num_before + warp_before + tiles_through), group_ok)
            sts_b32(s_row_start + safe_idx * 4, Uint32(row_start), group_ok)
            sts_b32(s_row_end + safe_idx * 4, Uint32(row_end), group_ok)
            num_before = num_before + round_total
            cute.arch.sync_threads()
        sts_b32(s_tile_start, Uint32(0), Int32(tidx == 0))
        cute.arch.sync_threads()

    @cute.jit
    def store_tile_start(self, tile_start_base, G, tidx, s_tile_start) -> None:
        for group_idx in cutlass.range(tidx, G + 1, THREADS):
            st_b32(tile_start_base + Int64(group_idx) * 4, lds_b32(s_tile_start + group_idx * 4), Int32(1))

    @cute.jit
    def sum_items(self, x_base, tile_sums_base, H, G, num_tiles, tidx, bx, num_ctas, group_tables, s_blocks) -> None:
        s_tile_start, s_row_start, s_row_end = group_tables
        lane, warp = tidx % 32, tidx // 32
        num_vecs = H * self.elem_bytes // self.vec_bytes
        num_chunks = (num_vecs + 31) // 32
        num_items = num_tiles * num_chunks
        row_bytes = Int64(H) * self.elem_bytes
        for item_idx in cutlass.range(bx, num_items, num_ctas):
            tile_idx = item_idx // num_chunks
            chunk_idx = item_idx - tile_idx * num_chunks
            group_idx = self.find_group(s_tile_start, tile_idx, G)
            row_start = Int32(lds_b32(s_row_start + group_idx * 4)) + (tile_idx - Int32(lds_b32(s_tile_start + group_idx * 4))) * TILE
            num_rows = cutlass.min(Int32(lds_b32(s_row_end + group_idx * 4)) - row_start, Int32(TILE))
            block_start = warp * ROWS_PER_WARP
            num_block_rows = cutlass.max(cutlass.min(num_rows - block_start, Int32(ROWS_PER_WARP)), Int32(0))
            # a lane past the last vector sums the last one again and stores nothing
            vec_idx = cutlass.min(chunk_idx * 32 + lane, num_vecs - 1)
            # an empty block reads (and discards) the tile's last row, which always exists
            block_row = row_start + cutlass.min(block_start, num_rows - 1)
            acc_sum = self.sum_block(x_base + Int64(vec_idx) * self.vec_bytes + Int64(block_row) * row_bytes, row_bytes, num_block_rows)
            s_buffer = s_blocks + ((item_idx // num_ctas) % 2) * (WARPS * 32 * self.vec_elems * 4)
            for k in cutlass.range_constexpr(self.vec_elems):
                sts_b32(s_buffer + ((warp * 32 + lane) * self.vec_elems + k) * 4, f32_as_u32(acc_sum[k]), Int32(1))
            cute.arch.sync_threads()
            self.store_tile_sum(tile_sums_base + (Int64(tile_idx) * H + chunk_idx * 32 * self.vec_elems) * 4, num_vecs * self.vec_elems - chunk_idx * 32 * self.vec_elems, tidx, s_buffer)

    @cute.jit
    def find_group(self, s_tile_start, tile_idx, G):
        """
        The group holding tile tile_idx: the last g < G with s_tile_start[g] <= tile_idx, which is
        never an empty group since an empty group starts where the next one does.
        """
        group_idx = Int32(0)
        step = self.max_groups // 2
        while step > 0:
            candidate = group_idx + step
            candidate_ok = Int32(candidate < G) & Int32(Int32(lds_b32(s_tile_start + cutlass.min(candidate, G - 1) * 4)) <= tile_idx)
            group_idx = group_idx + step * candidate_ok
            step //= 2
        return group_idx

    @cute.jit
    def sum_block(self, row_base, row_bytes, num_rows):
        """
        fp32 sums of the vector's elements over num_rows (0 to ROWS_PER_WARP) rows from row_base,
        one row at a time in increasing row order from zero.
        """
        acc_sum = [Float32(0.0)] * self.vec_elems
        if num_rows == ROWS_PER_WARP:
            acc_sum = self.add_rows(acc_sum, [self.load_vector(row_base + Int64(i) * row_bytes) for i in range(ROWS_PER_WARP)])
        else:
            # rows past the end read the block's last row (row_base itself when the block is empty)
            # and add +0.0, which leaves a sum unchanged: it starts at +0.0, so it is never -0.0
            row_oks = [Int32(i < num_rows) for i in range(ROWS_PER_WARP)]
            words = [self.load_vector(row_base + Int64(cutlass.min(Int32(i), cutlass.max(num_rows - 1, Int32(0)))) * row_bytes) for i in range(ROWS_PER_WARP)]
            acc_sum = self.add_rows(acc_sum, [[selp(row_oks[i], word, Uint32(0)) for word in words[i]] for i in range(ROWS_PER_WARP)])
        return acc_sum

    def add_rows(self, acc_sum: list, rows: list) -> list:
        """
        Add the rows' vectors to acc_sum one row at a time, in order.
        """
        for words in rows:
            elems = self.widen(words)
            acc_sum = [acc_sum[k] + elems[k] for k in range(self.vec_elems)]
        return acc_sum

    @cute.jit
    def store_tile_sum(self, tile_sum_base, num_valid, tidx, s_buffer) -> None:
        """
        Thread t adds element t of the chunk (32 * vec_elems fp32) over the warps' block sums by the
        fixed pairwise tree and stores it unless it lies past the row's last element.
        """
        for round_idx in cutlass.range_constexpr((32 * self.vec_elems + THREADS - 1) // THREADS):
            elem_idx = round_idx * THREADS + tidx
            elem_ok = Int32(elem_idx < 32 * self.vec_elems) & Int32(elem_idx < num_valid)
            block_sums = [u32_as_f32(lds_b32(s_buffer + (warp * 32 * self.vec_elems + cutlass.min(elem_idx, 32 * self.vec_elems - 1)) * 4)) for warp in range(WARPS)]
            st_f32(tile_sum_base + elem_idx * 4, pairwise_sum(block_sums), elem_ok)

    def load_vector(self, addr) -> list:
        """
        The vec_bytes at addr as 32-bit words; a 2-byte vector fills the low half of its word.
        Unpredicated, so that ptxas keeps every load of a block in flight (predicated loads with a
        zero default get interleaved with the adds); L1::no_allocate, since x is read once.
        """
        if cutlass.const_expr(self.vec_bytes == 2):
            return [Uint32(asm(T.i32(), [Int64(addr).ir_value()], "ld.global.L1::no_allocate.u16 $0, [$1];", "=r,l"))]
        num_words = self.vec_bytes // 4
        if cutlass.const_expr(num_words == 1):
            return [Uint32(asm(T.i32(), [Int64(addr).ir_value()], "ld.global.L1::no_allocate.b32 $0, [$1];", "=r,l"))]
        targets = ", ".join(f"${i}" for i in range(num_words))
        words = asm(llvm.StructType.get_literal([T.i32()] * num_words), [Int64(addr).ir_value()], f"ld.global.L1::no_allocate.v{num_words}.b32 {{{targets}}}, [${num_words}];", ",".join(["=r"] * num_words + ["l"]))
        return [Uint32(llvm.extractvalue(T.i32(), words, [i])) for i in range(num_words)]

    def widen(self, words: list) -> list:
        """
        The vector's vec_elems elements as fp32, in column order.
        """
        if cutlass.const_expr(self.x_dtype is cutlass.Float32):
            return [u32_as_f32(word) for word in words]
        if cutlass.const_expr(self.vec_bytes == 2):
            return [bf16_lo(words[0])] if cutlass.const_expr(self.x_dtype is cutlass.BFloat16) else [self.f16_to_f32(words[0], 0)]
        if cutlass.const_expr(self.x_dtype is cutlass.BFloat16):
            return [elem for word in words for elem in bf16x2_to_f32(word)]
        return [self.f16_to_f32(word, half) for word in words for half in range(2)]

    def f16_to_f32(self, word, half: int):
        """
        The low (half 0) or high (half 1) fp16 of word as fp32.
        """
        return Float32(asm(T.f32(), [Uint32(word).ir_value()], "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $1;\n\tcvt.f32.f16 $0, " + ("hi" if half else "lo") + ";\n\t}", "=f,r"))


COMPILED: dict = {}


def vector_bytes(x: torch.Tensor) -> int:
    """
    The widest load (16, 8, 4 or 2 bytes, at least one element) that divides a row and the base.
    """
    row_bytes = x.shape[1] * x.element_size()
    return next(vec_bytes for vec_bytes in (16, 8, 4, 2) if vec_bytes <= x.element_size() or (row_bytes % vec_bytes == 0 and x.data_ptr() % vec_bytes == 0))


def compile_segment_sum(x_dtype: torch.dtype, vec_bytes: int, max_groups: int, out_dtype: torch.dtype, offs_dtype: torch.dtype) -> Callable:
    matrix = lambda dtype, align: fake(TORCH2CUTE[dtype], (cute.sym_int(), cute.sym_int()), (cute.sym_int64(), 1), align)
    offs_type, offs_bytes, offs_signed = OFFSET_DTYPES[offs_dtype]
    kernel = SegmentSum(TORCH2CUTE[x_dtype], vec_bytes, max_groups, TORCH2CUTE[out_dtype], offs_bytes, offs_signed)
    return cute.compile(kernel, matrix(x_dtype, vec_bytes), fake(offs_type, (cute.sym_int(),), (cute.sym_int64(),), offs_bytes), matrix(torch.float32, 4), fake(Int32, (cute.sym_int(),), (1,), 4), matrix(out_dtype, 2), fake_stream(), options="--enable-tvm-ffi")


def compile_block_segment_sum(out_dtype: torch.dtype, offs_dtype: torch.dtype) -> Callable:
    matrix = lambda dtype, align: fake(TORCH2CUTE[dtype], (cute.sym_int(), cute.sym_int()), (cute.sym_int64(), 1), align)
    offs_type, offs_bytes, offs_signed = OFFSET_DTYPES[offs_dtype]
    return cute.compile(BlockSegmentSum(TORCH2CUTE[out_dtype], offs_bytes, offs_signed), matrix(torch.float32, 4), fake(offs_type, (cute.sym_int(),), (cute.sym_int64(),), offs_bytes), matrix(out_dtype, 2), fake_stream(), options="--enable-tvm-ffi")


FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
OFFSET_DTYPES = {torch.int32: (Int32, 4, True), torch.int64: (Int64, 8, True), torch.uint32: (cutlass.Uint32, 4, False), torch.uint64: (cutlass.Uint64, 8, False)}


def segment_sum_sm90(x: torch.Tensor, grouped_mm_offs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    assert x.ndim == 2 and x.dtype in FLOAT_DTYPES and out_dtype in FLOAT_DTYPES, (x.shape, x.dtype, out_dtype)
    assert x.is_contiguous(), "x must be contiguous"
    M, H = x.shape
    G = grouped_mm_offs.shape[0]
    if G == 0 or H == 0:
        return torch.empty((G, H), dtype=out_dtype, device=x.device)
    if M == 0:
        return torch.zeros((G, H), dtype=out_dtype, device=x.device)
    offs = grouped_mm_offs
    if offs.dtype not in OFFSET_DTYPES:
        # the kernel reads 32- and 64-bit integers of any stride; narrower ones convert exactly in one launch
        offs = offs.to(torch.int32)
    vec_bytes = vector_bytes(x)
    max_groups = max(THREADS, 1 << (G - 1).bit_length())
    tile_sums = torch.empty((-(-M // TILE) + G, H), dtype=torch.float32, device=x.device)
    tile_start = torch.empty(G + 1, dtype=torch.int32, device=x.device)
    out = torch.empty((G, H), dtype=out_dtype, device=x.device)
    key = ("segment_sum", x.dtype, vec_bytes, max_groups, out_dtype, offs.dtype)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_segment_sum(x.dtype, vec_bytes, max_groups, out_dtype, offs.dtype)
    fn(x, offs, tile_sums, tile_start, out)
    return out


def block_segment_sum_sm90(partials: torch.Tensor, grouped_mm_offs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert partials.ndim == 2 and partials.dtype == torch.float32 and partials.is_contiguous() and dtype in FLOAT_DTYPES, (partials.shape, partials.dtype, dtype)
    G, H = grouped_mm_offs.shape[0], partials.shape[1]
    out = torch.empty((G, H), dtype=dtype, device=partials.device)
    if G == 0 or H == 0:
        return out
    offs = grouped_mm_offs
    if offs.dtype not in OFFSET_DTYPES:
        # 32- and 64-bit integers are read in place at any stride; narrower ones convert exactly
        offs = offs.to(torch.int32)
    key = ("block_segment_sum", dtype, offs.dtype)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_block_segment_sum(dtype, offs.dtype)
    fn(partials, offs, out)
    return out
