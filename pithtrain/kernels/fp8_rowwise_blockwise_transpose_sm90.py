"""
CuTe-DSL (SM90) rowwise FP8 quantization of x and 128x128 blockwise FP8 quantization of x.T from one
read of x, bit-identical to the Triton kernel in fp8_rowwise_blockwise_transpose_sm100. Covers every
input, so there is no fallback.

One CTA of 1024 / R threads per 128x128 tile, R = 8 (large grids) or 4 (small grids); each thread
owns R consecutive rows of 16 columns, so the transposed store needs no smem transpose: it writes
each column's R rows as vectors of up to R bytes.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake_stream
from pithtrain.kernels.fp8_e8m0_scale_sm90 import e8m0_scale, exact_rcp
from pithtrain.kernels.fp8_scale_layout import mn_major_scale

__all__ = ["fp8_quant_act_sm90"]

BLOCK = 128

SMALL_GRID_TILES = 200


def asm_u32(operands: list, ptx: str, constraints: str, loc=None, ip=None) -> Uint32:
    result = llvm.inline_asm(T.i32(), [operand.ir_value(loc=loc, ip=ip) for operand in operands], ptx, constraints, has_side_effects=False, is_align_stack=False, loc=loc, ip=ip)
    return Uint32(result)


# The register block is kept as packed 32-bit words of the input dtype (bf16 / f16: two elements
# per word, low half first; f32: one). Every helper widens to fp32 only inside its own asm block,
# so LLVM cannot hoist the widening and keep an fp32 copy of the block live, which would double the
# register footprint, or make ptxas spill under a register cap.


@dsl_user_op
def accumulate_absmax(acc: Uint32, word: Uint32, kind: str, *, loc=None, ip=None) -> Uint32:
    """
    acc = max(acc, |word|) lane-wise in the input dtype (maxNum semantics, as tl.max).
    """
    max_op = {"bf16": "max.bf16x2", "f16": "max.f16x2", "f32": "max.f32"}[kind]
    mask = "0x7FFFFFFF" if kind == "f32" else "0x7FFF7FFF"
    ptx = f"{{\n\t.reg .b32 t;\n\tand.b32 t, $2, {mask};\n\t{max_op} $0, $1, t;\n\t}}"
    return asm_u32([Uint32(acc), Uint32(word)], ptx, "=r,r,r", loc, ip)


@dsl_user_op
def widen_amax(acc: Uint32, kind: str, *, loc=None, ip=None) -> Float32:
    """
    Reduce a packed |max| word to one fp32 value (exact widening).
    """
    if kind == "f32":
        ptx_body = "mov.b32 $0, $1;"
    else:
        ptx_type = {"bf16": "bf16", "f16": "f16"}[kind]
        ptx_body = (
            ".reg .b16 l, h, m;\n\tmov.b32 {l, h}, $1;\n\t"
            f"max.{ptx_type} m, l, h;\n\tcvt.f32.{ptx_type} $0, m;"
        )
    result = llvm.inline_asm(T.f32(), [Uint32(acc).ir_value(loc=loc, ip=ip)], "{\n\t" + ptx_body + "\n\t}", "=f,r", has_side_effects=False, is_align_stack=False, loc=loc, ip=ip)
    return Float32(result)


@dsl_user_op
def quantize4_words(words: list, elem_picks: list, kind: str, rcp: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Quantize 4 elements to 4 packed e4m3 bytes (element 0 in the lowest byte). elem_picks[k] =
    (j, half) picks element k from words[j] (half 0 = low 16 bits); rcp is an exact power of 2.
    The cvt is the one Triton emits for .to(tl.float8e4nv) on SM90; its first source lands in
    the upper byte.
    """
    num_words = len(words)
    rcp_operand = f"${num_words + 1}"
    ptx_lines = [".reg .b32 f0, f1, f2, f3;", ".reg .b16 l, h, q0, q1;"]
    for k, (j, half) in enumerate(elem_picks):
        word_operand = f"${j + 1}"
        if kind == "bf16":
            if half == 0:
                ptx_lines.append(f"shl.b32 f{k}, {word_operand}, 16;")
            else:
                ptx_lines.append(f"and.b32 f{k}, {word_operand}, 0xFFFF0000;")
        elif kind == "f16":
            ptx_lines.append(f"mov.b32 {{l, h}}, {word_operand};")
            ptx_lines.append(f"cvt.f32.f16 f{k}, {'l' if half == 0 else 'h'};")
        else:
            ptx_lines.append(f"mov.b32 f{k}, {word_operand};")
        ptx_lines.append(f"mul.rn.f32 f{k}, f{k}, {rcp_operand};")
    ptx_lines += ["cvt.rn.satfinite.e4m3x2.f32 q0, f1, f0;", "cvt.rn.satfinite.e4m3x2.f32 q1, f3, f2;", "mov.b32 $0, {q0, q1};"]
    ptx, constraints, operands = "{\n\t" + "\n\t".join(ptx_lines) + "\n\t}", "=r," + ",".join(["r"] * num_words) + ",f", [Uint32(word) for word in words] + [Float32(rcp)]
    return asm_u32(operands, ptx, constraints, loc, ip)


def global_ptr(base: cute.Pointer, offset: Int64, elem_bytes: int, dtype: type, align: int) -> cute.Pointer:
    addr = base.toint() + offset * elem_bytes
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


class SubBlock:
    """
    Where one thread's rows x 16 register block sits in its CTA's tile and in x.
    """

    def __init__(self, tidx, m_tile_idx, k_tile_idx, M, K, rows: int) -> None:
        self.tidx, self.m_tile_idx, self.k_tile_idx, self.M, self.K = tidx, m_tile_idx, k_tile_idx, M, K
        self.warp, self.lane = tidx // 32, tidx % 32
        # each warp covers 4 x rows rows and 8 x 16 columns
        self.row_group_idx, self.col_group_idx = self.lane // 8, self.lane % 8
        self.row0, self.col0 = m_tile_idx * BLOCK + self.warp * (4 * rows) + self.row_group_idx * rows, k_tile_idx * BLOCK + self.col_group_idx * 8


class RowwiseBlockwiseTranspose:
    """
    vec_k / vec_m: elements per global vector access along K (x and tok rows) / along M (blk_t
    rows), chosen on the host so no vector straddles the edge.
    """

    def __init__(self, dtype: type, vec_k: int, vec_m: int, rows: int = 8) -> None:
        self.dtype = dtype
        self.rows = rows
        self.threads = 1024 // rows
        self.num_warps = 32 // rows
        self.kind = {cutlass.BFloat16: "bf16", cutlass.Float16: "f16", Float32: "f32"}[dtype]
        self.elem_bytes = dtype.width // 8
        self.elems_per_word = 32 // dtype.width
        self.vec_k = vec_k
        self.vec_m = vec_m

    @cute.jit
    def __call__(self, mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor, mQT: cute.Tensor, mST: cute.Tensor, stream) -> None:
        M, K = mX.shape[0], mX.shape[1]
        grid_m, grid_k = (M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        self.kernel(mX, mQ, mS, mQT, mST).launch(grid=[grid_m, grid_k, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor, mQT: cute.Tensor, mST: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        m_tile_idx, k_tile_idx, _ = cute.arch.block_idx()
        sub_block = SubBlock(tidx, m_tile_idx, k_tile_idx, Int32(mX.shape[0]), Int32(mX.shape[1]), self.rows)
        x_words = self.load_block(mX, sub_block)
        row_amax = self.reduce_row_amax(x_words)
        q_bytes = cute.make_rmem_tensor(cute.make_layout(8), Uint8)
        self.store_rowwise(x_words, row_amax, q_bytes, mQ, mS, sub_block)
        tile_rcp = self.reduce_tile_amax(row_amax, mST, sub_block)
        self.store_transposed(x_words, tile_rcp, q_bytes, mQT, sub_block)

    @cute.jit
    def load_block(self, mX: cute.Tensor, sub_block: SubBlock) -> cute.Tensor:
        """
        The thread's register block as packed words, zero outside (M, K): element (row0 + i, col0 +
        half * 64 + e) at i * 16 + half * 8 + e.
        """
        row0, col0, M, K = sub_block.row0, sub_block.col0, sub_block.M, sub_block.K
        VEC_K, elem_bytes = self.vec_k, self.elem_bytes
        x_row_stride = Int64(mX.stride[0])
        x_block = cute.make_rmem_tensor(cute.make_layout(16 * self.rows), self.dtype)
        x_block.fill(0.0)
        for i in cutlass.range_constexpr(self.rows):
            row = row0 + i
            for half in cutlass.range_constexpr(2):
                for vec_idx in cutlass.range_constexpr(8 // VEC_K):
                    col = col0 + half * 64 + vec_idx * VEC_K
                    if row < M:
                        if col < K:
                            offset = Int64(row) * x_row_stride + Int64(col)
                            src = cute.make_tensor(global_ptr(mX.iterator, offset, elem_bytes, self.dtype, VEC_K * elem_bytes), cute.make_layout(VEC_K))
                            dst = cute.make_tensor(x_block.iterator + (i * 16 + half * 8 + vec_idx * VEC_K), cute.make_layout(VEC_K))
                            cute.autovec_copy(src, dst)
        return cute.recast_tensor(x_block, Uint32)

    @cute.jit
    def reduce_row_amax(self, x_words: cute.Tensor) -> cute.Tensor:
        """
        Amax of each of the thread's rows over the whole 128-column tile row (all 8 column groups).
        """
        row_amax = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for i in cutlass.range_constexpr(self.rows):
            acc_amax = Uint32(0)
            for elem_idx in cutlass.range_constexpr(i * 16, i * 16 + 16, self.elems_per_word):
                acc_amax = accumulate_absmax(acc_amax, x_words[elem_idx // self.elems_per_word], self.kind)
            amax = widen_amax(acc_amax, self.kind)
            for step in cutlass.range_constexpr(3):
                amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << step))
            row_amax[i] = amax
        return row_amax

    @cute.jit
    def store_rowwise(self, x_words: cute.Tensor, row_amax: cute.Tensor, q_bytes: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor, sub_block: SubBlock) -> None:
        """
        Quantize the register block by its row scales into tok and store the row scales.
        """
        row0, col0, col_group_idx, k_tile_idx, M, K = sub_block.row0, sub_block.col0, sub_block.col_group_idx, sub_block.k_tile_idx, sub_block.M, sub_block.K
        VEC_K = self.vec_k
        # mS is the MN-major tok_scale: scale[row, k_tile_idx] at k_tile_idx * tok_scale_stride + row.
        tok_scale_stride = Int64(mS.stride[1])
        q_words = cute.recast_tensor(q_bytes, Uint32)
        for i in cutlass.range_constexpr(self.rows):
            row = row0 + i
            scale = e8m0_scale(row_amax[i])
            rcp = exact_rcp(scale)
            if row < M:
                if col_group_idx == i:
                    scale_tile = cute.make_tensor(global_ptr(mS.iterator, Int64(k_tile_idx) * tok_scale_stride + Int64(row), 4, Float32, 4), cute.make_layout(1))
                    scale_tile[0] = scale
                for half in cutlass.range_constexpr(2):
                    for quad in cutlass.range_constexpr(2):
                        elems = [i * 16 + half * 8 + 4 * quad + k for k in range(4)]
                        q_words[quad] = self.quantize4(x_words, elems, rcp)
                    for vec_idx in cutlass.range_constexpr(8 // VEC_K):
                        col = col0 + half * 64 + vec_idx * VEC_K
                        if col < K:
                            dst = cute.make_tensor(global_ptr(mQ.iterator, Int64(row) * Int64(K) + Int64(col), 1, Uint8, VEC_K), cute.make_layout(VEC_K))
                            src = cute.make_tensor(q_bytes.iterator + vec_idx * VEC_K, cute.make_layout(VEC_K))
                            cute.autovec_copy(src, dst)

    @cute.jit
    def reduce_tile_amax(self, row_amax: cute.Tensor, mST: cute.Tensor, sub_block: SubBlock) -> Float32:
        """
        Reduce the row amaxes over the CTA's tile, store the tile's blk_t scale (thread 0) and
        return its reciprocal.
        """
        tidx, warp, lane, m_tile_idx, k_tile_idx = sub_block.tidx, sub_block.warp, sub_block.lane, sub_block.m_tile_idx, sub_block.k_tile_idx
        num_m_tiles = Int32(mST.shape[1])
        tile_amax = row_amax[0]
        for i in cutlass.range_constexpr(1, self.rows):
            tile_amax = cute.arch.fmax(tile_amax, row_amax[i])
        tile_amax = cute.arch.fmax(tile_amax, cute.arch.shuffle_sync_bfly(tile_amax, offset=8))
        tile_amax = cute.arch.fmax(tile_amax, cute.arch.shuffle_sync_bfly(tile_amax, offset=16))
        smem = cutlass.memory.SmemAllocator()
        s_amax = smem.allocate_tensor(Float32, cute.make_layout(self.num_warps), byte_alignment=16)
        if lane == 0:
            s_amax[warp] = tile_amax
        cute.arch.barrier()
        tile_amax = s_amax[0]
        for warp_idx in cutlass.range_constexpr(1, self.num_warps):
            tile_amax = cute.arch.fmax(tile_amax, s_amax[warp_idx])
        tile_scale = e8m0_scale(tile_amax)
        tile_rcp = exact_rcp(tile_scale)
        if tidx == 0:
            scale_t_tile = cute.make_tensor(global_ptr(mST.iterator, Int64(k_tile_idx) * Int64(num_m_tiles) + Int64(m_tile_idx), 4, Float32, 4), cute.make_layout(1))
            scale_t_tile[0] = tile_scale
        return tile_rcp

    @cute.jit
    def store_transposed(self, x_words: cute.Tensor, tile_rcp: Float32, q_bytes: cute.Tensor, mQT: cute.Tensor, sub_block: SubBlock) -> None:
        """
        Quantize the register block by the tile scale into blk_t, each column's rows as one vector.
        """
        row0, col0, M, K = sub_block.row0, sub_block.col0, sub_block.M, sub_block.K
        VEC_M = self.vec_m
        q_words = cute.recast_tensor(q_bytes, Uint32)
        if row0 < M:
            for half in cutlass.range_constexpr(2):
                for col_in_half in cutlass.range_constexpr(8):
                    col = col0 + half * 64 + col_in_half
                    if col < K:
                        for quad in cutlass.range_constexpr(self.rows // 4):
                            q_words[quad] = self.quantize4(x_words, [(4 * quad + k) * 16 + half * 8 + col_in_half for k in range(4)], tile_rcp)
                        for vec_idx in cutlass.range_constexpr(self.rows // VEC_M):
                            row = row0 + vec_idx * VEC_M
                            if row < M:
                                offset = Int64(col) * Int64(M) + Int64(row)
                                dst = cute.make_tensor(global_ptr(mQT.iterator, offset, 1, Uint8, VEC_M), cute.make_layout(VEC_M))
                                src = cute.make_tensor(q_bytes.iterator + vec_idx * VEC_M, cute.make_layout(VEC_M))
                                cute.autovec_copy(src, dst)

    def quantize4(self, x_words: cute.Tensor, elems: list, rcp: Float32) -> Uint32:
        word_idxs: list = []
        elem_picks: list = []
        for elem_idx in elems:
            word_idx = elem_idx // self.elems_per_word
            if word_idx not in word_idxs:
                word_idxs.append(word_idx)
            elem_picks.append((word_idxs.index(word_idx), elem_idx % self.elems_per_word))
        return quantize4_words([x_words[word_idx] for word_idx in word_idxs], elem_picks, self.kind, rcp)


COMPILED: dict[tuple, Callable] = {}


def vector_width(size: int, max_width: int) -> int:
    """
    The largest power-of-2 width up to max_width (a power of 2) that divides size.
    """
    width = max_width
    while size % width:
        width //= 2
    return width


def compile_kernel(dtype: type, vec_k: int, vec_m: int, rows: int) -> Callable:
    m_sym, k_sym, num_k_tiles_sym, num_m_tiles_sym = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    elem_bytes = dtype.width // 8
    make_fake = cute.runtime.make_fake_tensor
    x = make_fake(dtype, (m_sym, k_sym), stride=(cute.sym_int64(divisibility=vec_k), 1), assumed_align=vec_k * elem_bytes)
    tok = make_fake(cutlass.Float8E4M3FN, (m_sym, k_sym), stride=(k_sym, 1), assumed_align=16)
    tok_scale = make_fake(Float32, (m_sym, num_k_tiles_sym), stride=(1, cute.sym_int64()), assumed_align=16)
    blk_t = make_fake(cutlass.Float8E4M3FN, (k_sym, m_sym), stride=(m_sym, 1), assumed_align=16)
    blk_t_scale = make_fake(Float32, (num_k_tiles_sym, num_m_tiles_sym), stride=(num_m_tiles_sym, 1), assumed_align=16)
    return cute.compile(RowwiseBlockwiseTranspose(dtype, vec_k, vec_m, rows), x, tok, tok_scale, blk_t, blk_t_scale, fake_stream(), options="--enable-tvm-ffi")


DISPATCH: dict[tuple, tuple] = {}


def resolve(dtype: torch.dtype, M: int, K: int, x_row_stride: int, x_addr: int) -> tuple:
    """
    (compiled kernel or None for an empty input, ceil(K/128), ceil(M/128)).
    """
    num_k_tiles, num_m_tiles = (K + BLOCK - 1) // BLOCK, (M + BLOCK - 1) // BLOCK
    if M == 0 or K == 0:
        return None, num_k_tiles, num_m_tiles
    elem_bytes = dtype.itemsize
    vec_k = vector_width(K, min(16 // elem_bytes, 8))
    vec_k = vector_width(x_row_stride, vec_k)
    vec_k = vector_width(x_addr // elem_bytes, vec_k)
    # 128 threads x 8 rows wins once the grid fills the GPU (fewer, fuller stores); a small grid
    # is latency-bound, so split each tile over 256 threads x 4 rows.
    rows = 4 if num_m_tiles * num_k_tiles <= SMALL_GRID_TILES else 8
    vec_m = vector_width(M, rows)
    key = (dtype, vec_k, vec_m, rows)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_kernel(TORCH2CUTE[dtype], vec_k, vec_m, rows)
    if len(DISPATCH) > 4096:
        DISPATCH.clear()
    return fn, num_k_tiles, num_m_tiles


def fp8_quant_act_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    x_row_stride, x_col_stride = x.stride()
    if x_col_stride != 1:
        # as_strided pins the unit inner stride that torch leaves arbitrary when K == 1.
        x = x.contiguous().as_strided((M, K), (K, 1))
        x_row_stride = K
    dtype = x.dtype
    # Host cost matters (hundreds of calls per step), so resolve once per key.
    key = (dtype, M, K, x_row_stride, x.data_ptr() & 15)
    entry = DISPATCH.get(key)
    if entry is None:
        entry = DISPATCH[key] = resolve(dtype, M, K, x_row_stride, x.data_ptr())
    fn, num_k_tiles, num_m_tiles = entry
    device = x.device
    out_tok = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device)
    scale_tok = mn_major_scale(M, num_k_tiles, device)
    out_blk_t, scale_blk_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=device), torch.empty((num_k_tiles, num_m_tiles), dtype=torch.float32, device=device)
    if fn is not None:
        fn(x, out_tok, scale_tok, out_blk_t, scale_blk_t)
    return out_tok, scale_tok, out_blk_t, scale_blk_t
