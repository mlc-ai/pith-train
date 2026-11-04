"""
gpt-oss's biased clamped SwiGLU fused with the FP8 rowwise + K-major quantization of its consumers,
in CuTe-DSL for SM90.

The forward reads the raw gate_up GEMM output and the bias and writes the four outputs of
fp8cast_rowwise_kmajor of h; the backward reads dh, gate_up and the bias and writes the four outputs
for d(gate_up) plus fp32 per-128-row-block column sums that block_segment_sum turns into the bias
gradient. Neither materializes the bf16 biased gate_up, h or d(gate_up). The FP8 bytes and scales
are bit-identical to the unfused sequence: the bias add is one fp32 add rounded to bf16, and the
activation replays the Triton clamped_swiglu kernels' PTX. Both scales come back MN-major.

Only N % 8 == 0 with 1 to 1024 groups and a bf16 bias runs fused; anything else takes the unfused
sequence (the fp8_clamped_swiglu_quant_sm100 helpers, clamped_swiglu_fwd and
fp8cast_rowwise_kmajor_sm90).
"""

from __future__ import annotations

import struct

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.clamped_swiglu import clamped_swiglu_fwd
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import abs2, asm_args_last, bf16_hi, bf16_lo, bf16_sel, cp_async16, f32_as_u32, ld_b32_coherent, ld_v4, lds_b32_pred, lds_v4, max2, pack_bf16, prmt_args_last, selp_args_last, st_f32, st_v2_args_last, st_v4_args_last, sts_b32_nopred, sts_v4, to_ir, u32_as_f32, unpack
from pithtrain.kernels.fp8_clamped_swiglu_quant_sm100 import biased_gate_up, clamped_swiglu_bwd_biased
from pithtrain.kernels.fp8_e8m0_scale_sm90 import e8m0_scale_f32, ptx_f32, quant4_e4m3
from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import fp8cast_rowwise_kmajor_sm90
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad
from pithtrain.kernels.segment_sum import block_segment_sum

__all__ = ["ClampedSwiGLUQuant", "fp8_clamped_swiglu_bwd_sm90", "clamped_swiglu_quant_fusable", "clamped_swiglu_quant_fwd_sm90", "fp8_clamped_swiglu_kmajor_sm90", "fp8_clamped_swiglu_rowwise_sm90"]

BLOCK = 128  # rows per quantization group and output columns per CTA tile
THREADS = 512
VEC = 8  # output columns per thread
G_CAP = 1024  # expert groups the shared-memory offset table holds
STAGES = {False: 3, True: 2}  # cp.async tiles in flight per CTA, keyed by backward


def f32(x: float) -> float:
    """
    x rounded to float32, as Triton receives a Python float scalar argument.
    """
    return struct.unpack("<f", struct.pack("<f", x))[0]


@dsl_user_op
def add_bias_bf16x2(gate_up_word: Uint32, bias_word: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    Two packed BF16 plus two packed BF16, each an fp32 add rounded to BF16 (RN).
    """
    return Uint32(
        asm_args_last(
            T.i32(),
            "{\n.reg .f32 a0, a1, b0, b1;\n"
            "shl.b32 a0, $1, 16;\nand.b32 a1, $1, 0xFFFF0000;\n"
            "shl.b32 b0, $2, 16;\nand.b32 b1, $2, 0xFFFF0000;\n"
            "add.rn.f32 a0, a0, b0;\nadd.rn.f32 a1, a1, b1;\n"
            "cvt.rn.bf16x2.f32 $0, a1, a0;\n}",
            "=r,r,r",
            [to_ir(gate_up_word, Uint32, loc, ip), to_ir(bias_word, Uint32, loc, ip)],
        )
    )


@dsl_user_op
def lds_v2(addr: Int32, *, loc=None, ip=None) -> tuple:
    struct_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    words = asm_args_last(struct_type, "ld.shared.v2.b32 {$0, $1}, [$2];", "=r,=r,r", [to_ir(addr, Int32, loc, ip)], side_effects=True)  # side_effects keeps the load behind the barrier
    return unpack(words, 2, Uint32, loc, ip)


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


class TileThread:
    """
    Where one thread works in one 128 x 128 tile: its lane and warp, its column vector (8 columns
    from col0) and row group (4 rows), and the tile's place in the output.
    """

    def __init__(self, coords: tuple, row_block_idx, col_block_idx, num_out_cols: int, has_col_tail: bool) -> None:
        self.tidx, self.lane, self.warp, self.col_vec_idx, self.row_group_idx = coords
        self.row_block_idx, self.col_block_idx = row_block_idx, col_block_idx
        self.row_start, self.col_start = row_block_idx * BLOCK, col_block_idx * BLOCK
        self.col0 = self.col_start + self.col_vec_idx * VEC
        self.col_ok = Int32(1)
        if cutlass.const_expr(has_col_tail):
            self.col_ok = Int32(self.col0 < num_out_cols)


class ClampedSwiGLUQuant:
    """
    Persistent kernel (one 512-thread CTA per SM over 128 x 128 output tiles), compiled per (N,
    backward, alpha, limit, KP, rowwise, kmajor).

    N is the SwiGLU output width (gate_up has 2N columns). The quantized output is h (M, N) in
    forward and d(gate_up) (M, 2N) in backward. Thread t owns output columns col_vec_idx * 8 ..
    col_vec_idx * 8 + 7 (col_vec_idx = t % 16) and rows row_group_idx * 4 .. row_group_idx * 4 + 3
    (row_group_idx = t // 16) of the tile.

    Stage layout (bytes, per stage): forward, two 16-byte slots per (row, thread) holding the
    interleaved gate_up columns of output columns +0..3 and +4..7; backward, an 8-byte dh slot then
    a 16-byte gate_up slot per (row, thread). Every thread reads back only the bytes it copied, so
    the staging needs cp.async.wait_group and no barrier.
    """

    def __init__(self, N: int, backward: bool, alpha: float, limit: float, num_ctas: int, KP: int, rowwise: bool = True, kmajor: bool = True):
        self.N = N
        self.backward = backward
        self.num_out_cols = 2 * N if backward else N
        self.KP = KP  # channels of the K-major output (>= num_out_cols; the rest is left unwritten)
        self.rowwise = rowwise  # False: only the K-major pair (backward recompute of h)
        self.kmajor = kmajor  # False: only the rowwise pair (forward when h is recomputed)
        assert rowwise or kmajor
        assert kmajor or not backward
        self.num_col_blocks = (self.num_out_cols + BLOCK - 1) // BLOCK
        self.has_col_tail = self.num_out_cols % BLOCK != 0
        self.stages = STAGES[backward]
        self.num_ctas = num_ctas
        self.stage_bytes = 4 * THREADS * (24 if backward else 32)
        self.fwd_op = make_fwd_op(alpha, limit)
        self.bwd_op = make_bwd_op(alpha, limit)

    @cute.jit
    def __call__(self, mDh: cute.Tensor, mGU: cute.Tensor, mBias: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKmajor: cute.Tensor, mKmajorScale: cute.Tensor, mPartials: cute.Tensor, stream):
        num_tiles = (mGU.shape[0] // BLOCK) * self.num_col_blocks
        self.kernel(mDh, mGU, mBias, mOffs, mTok, mTokScale, mKmajor, mKmajorScale, mPartials).launch(grid=[cutlass.min(num_tiles, self.num_ctas), 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mDh: cute.Tensor, mGU: cute.Tensor, mBias: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKmajor: cute.Tensor, mKmajorScale: cute.Tensor, mPartials: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        coords = self.coords(tidx)
        M = mGU.shape[0]
        num_tiles = (M // BLOCK) * self.num_col_blocks
        G = mOffs.shape[0]
        s_stages, s_tile_t, s_col_amax, s_col_rcp, s_offs, s_col_sum = self.allocate_smem()
        gate_up_base = mGU.iterator.toint()
        dh_base = gate_up_base
        if cutlass.const_expr(self.backward):
            dh_base = mDh.iterator.toint()
        bias_base = mBias.iterator.toint()

        for stage in cutlass.range_constexpr(self.stages):
            self.load_stage(bx + stage * num_ctas, s_stages + stage * self.stage_bytes, num_tiles, tidx, dh_base, gate_up_base)
        self.load_offsets(mOffs, s_offs, G, tidx)

        num_iters = (num_tiles - bx + num_ctas - 1) // num_ctas
        for iter_idx in cutlass.range(num_iters, unroll=1):
            tile_idx = bx + iter_idx * num_ctas
            row_block_idx, col_block_idx = self.locate_tile(tile_idx)
            tile = TileThread(coords, row_block_idx, col_block_idx, self.num_out_cols, self.has_col_tail)
            stage_addr = s_stages + (iter_idx % self.stages) * self.stage_bytes
            group_idx, group_start, num_group_rows = self.find_group(s_offs, G, tile)
            bias_words = self.load_bias(bias_base, group_idx, tile)
            cute.arch.cp_async_wait_group(self.stages - 1)
            out_words = self.apply_activation(stage_addr, bias_words, tidx)
            # the stage is consumed: refill it with the tile self.stages iterations ahead
            self.load_stage(tile_idx + self.stages * num_ctas, stage_addr, num_tiles, tidx, dh_base, gate_up_base)
            abs_words = [[abs2(out_words[k][j]) for j in range(4)] for k in range(4)]
            if cutlass.const_expr(self.kmajor):
                self.reduce_col_amax(abs_words, tile, s_col_amax)
            if cutlass.const_expr(self.backward):
                self.reduce_col_sum(out_words, tile, s_col_sum)
            if cutlass.const_expr(self.rowwise):
                self.quantize_rows(out_words, self.reduce_row_amax(abs_words), tile, mTok, mTokScale, M)
            if cutlass.const_expr(self.kmajor):
                cute.arch.sync_threads()
                self.store_col_scales(tile, s_col_amax, s_col_sum, s_col_rcp, mKmajorScale, mPartials)
                cute.arch.sync_threads()
                self.quantize_cols(out_words, tile, s_col_rcp, s_tile_t)
                cute.arch.sync_threads()
                self.store_kmajor(tile, s_tile_t, mKmajor, group_start, num_group_rows)

        cute.arch.cp_async_wait_group(0)

    def coords(self, tidx):
        return tidx, tidx % 32, tidx // 32, tidx % 16, tidx // 16

    def allocate_smem(self):
        smem = cutlass.memory.SmemAllocator()
        s_stages = smem.allocate(self.stages * self.stage_bytes, byte_alignment=128).toint()
        s_tile_t = smem.allocate(BLOCK * BLOCK, byte_alignment=128).toint()
        s_col_amax = smem.allocate((THREADS // 32) * 16 * 16, byte_alignment=16).toint()
        s_col_rcp = smem.allocate(BLOCK * 4, byte_alignment=16).toint()
        s_offs = smem.allocate(G_CAP * 4, byte_alignment=16).toint()
        s_col_sum = s_col_amax
        if cutlass.const_expr(self.backward):
            s_col_sum = smem.allocate((THREADS // 32) * BLOCK * 4, byte_alignment=16).toint()
        return s_stages, s_tile_t, s_col_amax, s_col_rcp, s_offs, s_col_sum

    def locate_tile(self, tile_idx: Int32) -> tuple:
        return tile_idx // self.num_col_blocks, tile_idx % self.num_col_blocks

    def load_stage(self, tile_idx: Int32, stage_addr: Int32, num_tiles: Int32, tidx: Int32, dh_base, gate_up_base):
        """
        cp.async tile tile_idx (zero-filled when out of range) into a stage.
        """
        N = self.N
        copy_ok = Int32(tile_idx < num_tiles)
        row_block_idx, col_block_idx = self.locate_tile(tile_idx)
        row0 = row_block_idx * BLOCK + (tidx // 16) * 4
        col0 = col_block_idx * BLOCK + (tidx % 16) * VEC  # first output column
        if cutlass.const_expr(self.has_col_tail):
            copy_ok = copy_ok & Int32(col0 < self.num_out_cols)
        if cutlass.const_expr(self.backward):
            # dh columns col0/2 .. col0/2+3 (8 B); gate_up columns col0 .. col0+7 (16 B).
            dh_offset, gate_up_offset = (Int64(row0) * N + col0 // 2) * 2 * Int64(copy_ok), (Int64(row0) * (2 * N) + col0) * 2 * Int64(copy_ok)
            for k in range(4):
                # 8-byte cp.async into the stage; zero-fills where copy_ok is 0
                asm_args_last(T.i32(), "cp.async.ca.shared.global [$1], [$2], 8, $3;\nmov.b32 $0, 0;", "=r,r,l,r", [to_ir(stage_addr + (k * THREADS + tidx) * 8, Int32), to_ir(dh_base + dh_offset + k * N * 2, Int64), to_ir(copy_ok * 8, Int32)], side_effects=True)
            for k in range(4):
                cp_async16(stage_addr + 4 * THREADS * 8 + (k * THREADS + tidx) * 16, gate_up_base + gate_up_offset + k * (2 * N) * 2, copy_ok * 16)
        else:
            # gate_up columns 2 * col0 .. 2 * col0 + 15 as two 16-byte chunks.
            gate_up_offset = (Int64(row0) * (2 * N) + 2 * col0) * 2 * Int64(copy_ok)
            for half in range(2):
                for k in range(4):
                    cp_async16(stage_addr + ((half * 4 + k) * THREADS + tidx) * 16, gate_up_base + gate_up_offset + k * (2 * N) * 2 + half * 16, copy_ok * 16)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def load_offsets(self, mOffs: cute.Tensor, s_offs, G, tidx):
        """
        Copy the G group end offsets into shared memory, where every tile searches them.
        """
        for chunk_idx in cutlass.range((G + THREADS - 1) // THREADS, unroll=1):
            offs_idx = chunk_idx * THREADS + tidx
            if offs_idx < G:
                group_end = ld_b32_coherent(mOffs.iterator.toint() + Int64(offs_idx) * 4, Int32(1), Int32(0))
                sts_b32_nopred(s_offs + offs_idx * 4, Uint32(group_end))
        cute.arch.sync_threads()

    @cute.jit
    def find_group(self, s_offs, G, tile):
        """
        Index, start row and row count of group min(searchsorted(offs, row_start, right), G - 1),
        the group that holds the tile.
        """
        lane, row_start = tile.lane, tile.row_start
        # each round a warp ballot counts the groups among 32 offsets that end at or before the tile
        num_groups_before = Int32(0)
        for chunk_idx in cutlass.range((G + 31) // 32, unroll=1):
            probe_idx = chunk_idx * 32 + lane
            probe_end = lds_b32_pred(s_offs + probe_idx * 4, Int32(probe_idx < G), Int32(0x7FFFFFFF))
            num_groups_before += cute.arch.popc(cute.arch.vote_ballot_sync(probe_end <= row_start))
        group_idx = cutlass.min(num_groups_before, G - 1)
        group_end, group_start = lds_b32_pred(s_offs + group_idx * 4, Int32(1), Int32(0)), lds_b32_pred(s_offs + (group_idx - 1) * 4, Int32(group_idx > 0), Int32(0))
        return group_idx, group_start, Int64(group_end - group_start)

    def load_bias(self, bias_base, group_idx, tile):
        """
        The group's bias at the thread's gate_up columns: 4 words in backward, 8 in forward (where
        each output column reads an interleaved gate / up pair).
        """
        bias_row_base = bias_base + Int64(group_idx) * (2 * self.N) * 2
        if cutlass.const_expr(self.backward):
            return ld_v4(bias_row_base + Int64(tile.col0) * 2, tile.col_ok)
        return ld_v4(bias_row_base + Int64(tile.col0) * 4, tile.col_ok) + ld_v4(bias_row_base + Int64(tile.col0) * 4 + 16, tile.col_ok)

    def apply_activation(self, stage_addr, bias_words, tidx):
        """
        The thread's outputs from its staged inputs (h forward, d(gate_up) backward), two BF16
        packed per word: out_words[k][j] holds output columns 2j, 2j + 1 of row k.
        """
        out_words = [[None] * 4 for _ in range(4)]
        if cutlass.const_expr(self.backward):
            for k in range(4):
                dh_words = lds_v2(stage_addr + (k * THREADS + tidx) * 8)
                gate_up_words = lds_v4(stage_addr + 4 * THREADS * 8 + (k * THREADS + tidx) * 16)
                for j in range(4):
                    biased = add_bias_bf16x2(gate_up_words[j], bias_words[j])
                    dh_half = bf16_lo if j % 2 == 0 else bf16_hi
                    grad_gate, grad_up = self.bwd_op(dh_half(dh_words[j // 2]), bf16_lo(biased), bf16_hi(biased))
                    out_words[k][j] = pack_bf16(grad_gate, grad_up)
        else:
            for k in range(4):
                words_lo, words_hi = lds_v4(stage_addr + (k * THREADS + tidx) * 16), lds_v4(stage_addr + ((4 + k) * THREADS + tidx) * 16)
                gate_up_words = words_lo + words_hi
                h_f32 = [None] * 8
                for j in range(8):
                    biased = add_bias_bf16x2(gate_up_words[j], bias_words[j])
                    h_f32[j] = self.fwd_op(bf16_lo(biased), bf16_hi(biased))
                for j in range(4):
                    out_words[k][j] = pack_bf16(h_f32[2 * j], h_f32[2 * j + 1])
        return out_words

    @cute.jit
    def reduce_col_amax(self, abs_words, tile, s_col_amax):
        """
        Stage one column amax partial per warp in shared memory: the thread's 4 rows in registers,
        then the other row group of the warp by shuffle.
        """
        lane, warp, col_vec_idx = tile.lane, tile.warp, tile.col_vec_idx
        col_amax = [None] * 4
        for j in cutlass.range_constexpr(4):
            amax = max2(max2(abs_words[0][j], abs_words[1][j]), max2(abs_words[2][j], abs_words[3][j]))
            amax_peer = Uint32(cute.arch.shuffle_sync_bfly(Int32(amax), offset=16))
            col_amax[j] = max2(amax, amax_peer)
        if lane < 16:
            sts_v4(s_col_amax + (warp * 16 + col_vec_idx) * 16, tuple(col_amax))

    @cute.jit
    def reduce_col_sum(self, out_words, tile, s_col_sum):
        """
        Stage one fp32 column-sum partial of the BF16 gradient per warp in shared memory (for the
        bias gradient), reduced like reduce_col_amax.
        """
        lane, warp, col_vec_idx = tile.lane, tile.warp, tile.col_vec_idx
        sum_bits = [None] * 8
        for j in cutlass.range_constexpr(4):
            sum_lo = ((bf16_lo(out_words[0][j]) + bf16_lo(out_words[1][j])) + bf16_lo(out_words[2][j])) + bf16_lo(out_words[3][j])
            sum_hi = ((bf16_hi(out_words[0][j]) + bf16_hi(out_words[1][j])) + bf16_hi(out_words[2][j])) + bf16_hi(out_words[3][j])
            sum_lo, sum_hi = sum_lo + cute.arch.shuffle_sync_bfly(sum_lo, offset=16), sum_hi + cute.arch.shuffle_sync_bfly(sum_hi, offset=16)
            sum_bits[2 * j] = f32_as_u32(sum_lo)
            sum_bits[2 * j + 1] = f32_as_u32(sum_hi)
        if lane < 16:
            sum_addr = s_col_sum + (warp * BLOCK + col_vec_idx * VEC) * 4
            sts_v4(sum_addr, tuple(sum_bits[0:4]))
            sts_v4(sum_addr + 16, tuple(sum_bits[4:8]))

    def reduce_row_amax(self, abs_words):
        """
        Amax of the thread's rows over the whole tile row, packed per pair: row_pair_amax[pair]
        holds rows 2 * pair and 2 * pair + 1, reduced over the 16 column vectors by butterfly.
        """
        row_half_amax = [max2(max2(abs_words[k][0], abs_words[k][1]), max2(abs_words[k][2], abs_words[k][3])) for k in range(4)]
        row_pair_amax = [None] * 2
        for pair in range(2):
            amax_even, amax_odd = row_half_amax[2 * pair], row_half_amax[2 * pair + 1]
            amax = max2(prmt_args_last(amax_even, amax_odd, 0x5410), prmt_args_last(amax_even, amax_odd, 0x7632))
            for shift in range(4):
                amax_peer = Uint32(cute.arch.shuffle_sync_bfly(Int32(amax), offset=1 << shift))
                amax = max2(amax, amax_peer)
            row_pair_amax[pair] = amax
        return row_pair_amax

    def quantize_rows(self, out_words, row_pair_amax, tile, mTok: cute.Tensor, mTokScale: cute.Tensor, M):
        """
        Store the rowwise scales and the thread's 4 rows quantized by them; the thread with
        col_vec_idx = c < 4 stores the scale of row row_group_idx * 4 + c.
        """
        lane, col_vec_idx, row_group_idx, row_start = tile.lane, tile.col_vec_idx, tile.row_group_idx, tile.row_start
        row_amax = bf16_sel(Int32(col_vec_idx % 2), selp_args_last(Int32(col_vec_idx < 2), row_pair_amax[0], row_pair_amax[1]))
        row_scale, row_rcp_lane = e8m0_scale_f32(row_amax)
        st_f32(mTokScale.iterator.toint() + (Int64(tile.col_block_idx) * M + row_start + row_group_idx * 4 + col_vec_idx) * 4, row_scale, Int32(col_vec_idx < 4))
        for k in range(4):
            row_rcp = cute.arch.shuffle_sync(row_rcp_lane, offset=(lane & 16) | k)
            row_rcp4 = (row_rcp, row_rcp, row_rcp, row_rcp)
            row_words = out_words[k]
            quant_lo = quant4_e4m3(bf16_lo(row_words[0]), bf16_hi(row_words[0]), bf16_lo(row_words[1]), bf16_hi(row_words[1]), row_rcp4)
            quant_hi = quant4_e4m3(bf16_lo(row_words[2]), bf16_hi(row_words[2]), bf16_lo(row_words[3]), bf16_hi(row_words[3]), row_rcp4)
            row_idx = row_start + row_group_idx * 4 + k
            st_v2_args_last(mTok.iterator.toint() + Int64(row_idx) * self.num_out_cols + tile.col0, quant_lo, quant_hi, tile.col_ok)

    @cute.jit
    def store_col_scales(self, tile, s_col_amax, s_col_sum, s_col_rcp, mKmajorScale: cute.Tensor, mPartials: cute.Tensor):
        """
        Thread c < 128 reduces column c of the tile over the warps' partials, stores its E8M0 scale
        (and in backward its bias-gradient partial sum) and leaves the reciprocal in s_col_rcp.
        """
        tidx = tile.tidx
        if tidx < BLOCK:
            acc_amax = Uint32(0)
            for warp_idx in cutlass.range_constexpr(THREADS // 32):
                amax_bits = lds_b32_pred(s_col_amax + (warp_idx * 16 + tidx // 8) * 16 + ((tidx % 8) // 2) * 4, Int32(1), Int32(0))
                acc_amax = max2(acc_amax, Uint32(amax_bits))
            col_scale, col_rcp = e8m0_scale_f32(bf16_sel(Int32(tidx % 2), acc_amax))
            col = tile.col_start + tidx
            col_ok = Int32(1)
            if cutlass.const_expr(self.has_col_tail):
                col_ok = Int32(col < self.num_out_cols)
            st_f32(mKmajorScale.iterator.toint() + (Int64(tile.row_block_idx) * mn_pad(self.KP) + col) * 4, col_scale, col_ok)
            sts_b32_nopred(s_col_rcp + tidx * 4, Uint32(f32_as_u32(col_rcp)))
            if cutlass.const_expr(self.backward):
                acc_sum = Float32(0.0)
                for warp_idx in cutlass.range_constexpr(THREADS // 32):
                    sum_bits = lds_b32_pred(s_col_sum + (warp_idx * BLOCK + tidx) * 4, Int32(1), Int32(0))
                    acc_sum = acc_sum + u32_as_f32(Uint32(sum_bits))
                st_f32(mPartials.iterator.toint() + (Int64(tile.row_block_idx) * self.num_out_cols + col) * 4, acc_sum, col_ok)

    def quantize_cols(self, out_words, tile, s_col_rcp, s_tile_t):
        """
        Quantize the thread's 8 columns by their column scales into the transposed tile, whose
        16-byte chunks are swizzled by col_vec_idx % 8.
        """
        col_vec_idx, row_group_idx = tile.col_vec_idx, tile.row_group_idx
        col_rcp_words = lds_v4(s_col_rcp + col_vec_idx * 32) + lds_v4(s_col_rcp + col_vec_idx * 32 + 16)
        swizzle = (col_vec_idx & 7) << 4
        for j in range(8):
            word_idx = j // 2
            col_half = bf16_lo if j % 2 == 0 else bf16_hi
            col_rcp = u32_as_f32(col_rcp_words[j])
            quant = quant4_e4m3(col_half(out_words[0][word_idx]), col_half(out_words[1][word_idx]), col_half(out_words[2][word_idx]), col_half(out_words[3][word_idx]), (col_rcp,) * 4)
            tile_col = col_vec_idx * VEC + j
            sts_b32_nopred(s_tile_t + tile_col * BLOCK + (((row_group_idx // 4) << 4) ^ swizzle) + (row_group_idx % 4) * 4, quant)

    def store_kmajor(self, tile, s_tile_t, mKmajor: cute.Tensor, group_start, num_group_rows):
        """
        Copy the transposed tile into the K-major output, where its group's block is (KP,
        num_group_rows) at group_start * KP; 8 threads per column store its 128 contiguous bytes.
        """
        tidx, row_start, col_start = tile.tidx, tile.row_start, tile.col_start
        kmajor_block_base = mKmajor.iterator.toint() + Int64(group_start) * self.KP + Int64(row_start - group_start)
        for i in range(BLOCK * BLOCK // 16 // THREADS):
            vec_idx = i * THREADS + tidx
            tile_col, chunk_idx = vec_idx // 8, vec_idx % 8
            chunk = lds_v4(s_tile_t + tile_col * BLOCK + ((chunk_idx ^ ((tile_col // 8) & 7)) << 4))
            col = col_start + tile_col
            col_ok = Int32(1)
            if cutlass.const_expr(self.has_col_tail):
                col_ok = Int32(col < self.num_out_cols)
            st_v4_args_last(kmajor_block_base + Int64(col) * num_group_rows + chunk_idx * 16, chunk, col_ok)


compiled: dict = {}


def get_kernel(N: int, backward: bool, alpha: float, limit: float, device: torch.device, KP: int, rowwise: bool = True, kmajor: bool = True):
    """
    M and the group count stay dynamic (symbolic in the compile), so one compile serves every batch.
    """
    key = ("clamped_swiglu_quant", N, backward, alpha, limit, device.index, STAGES[backward], KP, rowwise, kmajor)
    fn = compiled.get(key)
    if fn is None:
        M = cute.sym_int()
        num_out_cols = 2 * N if backward else N
        num_col_blocks = (num_out_cols + BLOCK - 1) // BLOCK
        bf16_type, fp8_type = cutlass.BFloat16, cutlass.Float8E4M3FN
        gate_up = fake(bf16_type, (M, 2 * N), (2 * N, 1))
        dh = fake(bf16_type, (M, N), (N, 1)) if backward else None
        bias = fake(bf16_type, (cute.sym_int(), 2 * N), (2 * N, 1))
        outputs = (fake(fp8_type, (M, num_out_cols), (num_out_cols, 1)), fake(Float32, (M, num_col_blocks), (1, cute.sym_int64()), 16), fake(fp8_type, (cute.sym_int(),), (1,)), fake(Float32, (KP, cute.sym_int()), (1, mn_pad(KP)), 16))
        if not rowwise:
            outputs = (None, None) + outputs[2:]
        if not kmajor:
            outputs = outputs[:2] + (None, None)
        partials = fake(Float32, (cute.sym_int(), num_out_cols), (num_out_cols, 1)) if backward else None
        offs = fake(Int32, (cute.sym_int(),), (1,), 4)
        fn = cute.compile(ClampedSwiGLUQuant(N, backward, alpha, limit, SM_COUNT, KP, rowwise, kmajor), dh, gate_up, bias, offs, *outputs, partials, fake_stream(), options="--enable-tvm-ffi")
        compiled[key] = fn
    return fn


def allocate_outputs(M: int, N: int, device: torch.device, KP: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The four outputs of fp8cast_rowwise_kmajor for an (M, N) input, the K-major pair with KP
    >= N channels.
    """
    num_col_blocks = (N + BLOCK - 1) // BLOCK
    KP = N if KP is None else KP
    return (torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), mn_major_scale(M, num_col_blocks, device), torch.empty(M * KP, dtype=torch.float8_e4m3fn, device=device), mn_major_scale(KP, M // BLOCK, device))


def clamped_swiglu_quant_fusable(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor) -> bool:
    N = gate_up.shape[1] // 2
    return (gate_up.dtype == torch.bfloat16 and gate_up.is_contiguous() and gate_up.data_ptr() % 16 == 0 and bias.dtype == torch.bfloat16 and bias.is_contiguous() and bias.data_ptr() % 16 == 0 and N % VEC == 0 and grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.is_contiguous() and 0 < grouped_mm_offs.shape[0] <= G_CAP)


def clamped_swiglu_quant_fwd_sm90(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    grouped_mm_offs is int32 (G,) cumulative row counts, multiples of 128, last entry M. If set
    (fused path only), kmajor_channels >= N lays the K-major pair out for that many channels: group
    g's block is (kmajor_channels, rows of g) and the scale (kmajor_channels, M / 128); channels >=
    N are left unwritten.
    """
    assert gate_up.ndim == 2 and gate_up.shape[1] % 2 == 0 and gate_up.shape[0] % BLOCK == 0
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0:
        return allocate_outputs(M, N, gate_up.device)
    if not clamped_swiglu_quant_fusable(gate_up, bias, grouped_mm_offs):
        h = clamped_swiglu_fwd(biased_gate_up(gate_up, bias, grouped_mm_offs), alpha, limit)
        return fp8cast_rowwise_kmajor_sm90(h, grouped_mm_offs)
    KP = N if kmajor_channels is None else kmajor_channels
    assert KP >= N
    out = allocate_outputs(M, N, gate_up.device, KP)
    fn = get_kernel(N, False, f32(alpha), f32(limit), gate_up.device, KP)
    fn(None, gate_up, bias, grouped_mm_offs, *out, None)
    return out


def fp8_clamped_swiglu_rowwise_sm90(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0 or not clamped_swiglu_quant_fusable(gate_up, bias, grouped_mm_offs):
        return clamped_swiglu_quant_fwd_sm90(gate_up, bias, grouped_mm_offs, alpha, limit)[:2]
    tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=gate_up.device)
    tok_scale = mn_major_scale(M, (N + BLOCK - 1) // BLOCK, gate_up.device)
    fn = get_kernel(N, False, f32(alpha), f32(limit), gate_up.device, N, kmajor=False)
    fn(None, gate_up, bias, grouped_mm_offs, tok, tok_scale, None, None, None)
    return tok, tok_scale


def fp8_clamped_swiglu_kmajor_sm90(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0 or not clamped_swiglu_quant_fusable(gate_up, bias, grouped_mm_offs):
        return clamped_swiglu_quant_fwd_sm90(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels)[2:]
    KP = N if kmajor_channels is None else kmajor_channels
    assert KP >= N
    kmajor = torch.empty(M * KP, dtype=torch.float8_e4m3fn, device=gate_up.device)
    kmajor_scale = mn_major_scale(KP, M // BLOCK, gate_up.device)
    fn = get_kernel(N, False, f32(alpha), f32(limit), gate_up.device, KP, rowwise=False)
    fn(None, gate_up, bias, grouped_mm_offs, None, None, kmajor, kmajor_scale, None)
    return kmajor, kmajor_scale


def fp8_clamped_swiglu_bwd_sm90(dh: torch.Tensor, gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    assert dh.shape == (M, N), (dh.shape, gate_up.shape)
    G = grouped_mm_offs.shape[0]
    if M == 0 or N == 0:
        return allocate_outputs(M, 2 * N, gate_up.device), bias.new_zeros(bias.shape)
    if not (clamped_swiglu_quant_fusable(gate_up, bias, grouped_mm_offs) and dh.dtype == torch.bfloat16 and dh.is_contiguous()):
        grad_gate_up = clamped_swiglu_bwd_biased(dh, gate_up, bias, grouped_mm_offs, alpha, limit)
        partials = grad_gate_up.float().view(M // BLOCK, BLOCK, 2 * N).sum(1)
        return fp8cast_rowwise_kmajor_sm90(grad_gate_up, grouped_mm_offs), block_segment_sum(partials, grouped_mm_offs, bias.dtype)
    KP = 2 * N if kmajor_channels is None else kmajor_channels
    assert KP >= 2 * N
    out = allocate_outputs(M, 2 * N, gate_up.device, KP)
    partials = torch.empty((M // BLOCK, 2 * N), dtype=torch.float32, device=gate_up.device)
    fn = get_kernel(N, True, f32(alpha), f32(limit), gate_up.device, KP)
    fn(dh, gate_up, bias, grouped_mm_offs, *out, partials)
    assert G > 0
    return out, block_segment_sum(partials, grouped_mm_offs, bias.dtype)
