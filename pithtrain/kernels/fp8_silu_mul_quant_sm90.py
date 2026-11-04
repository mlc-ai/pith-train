"""
SM90 fp8_silu_mul_quant: CuTe-DSL kernels computing silu(g) * u (or its two gradients) and writing
only its rowwise + K-major FP8 quantization, never the bf16 intermediate. Bit-identical to
silu_mul_fwd / silu_mul_bwd followed by fp8cast_rowwise_kmajor; both scales MN-major. The
SiluMulQuantGU variant reads g and u as the halves of a fused (M, 2N) gate_up output and writes the
backward as one (M, 2N) operand. Shapes the fused kernels do not take (see silu_mul_quant_fusable,
silu_mul_quant_gu_fusable) run the unfused sequence.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import abs2, asm_args_last, bf16_hi, bf16_lo, bf16_sel, cp_async16, f32_as_u32, ld_b32_coherent, lds_b32_pred, lds_v4, max2, pack_bf16, prmt_args_last, selp_args_last, st_f32, st_v2_args_last, st_v4_args_last, sts_b32_nopred, sts_v4, to_ir, u32_as_f32, unpack
from pithtrain.kernels.fp8_e8m0_scale_sm90 import e8m0_scale_f32, quant4_e4m3
from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import fp8cast_rowwise_kmajor_sm90
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad
from pithtrain.kernels.silu_mul import silu_mul_bwd, silu_mul_fwd

__all__ = ["SiluMulQuant", "SiluMulQuantGU", "fp8_silu_mul_gu_bwd_sm90", "fp8_silu_mul_bwd_sm90", "silu_mul_quant_fusable", "fp8_silu_mul_fwd_sm90", "silu_mul_quant_gu_fusable", "fp8_silu_mul_gu_fwd_sm90"]

BLOCK = 128  # rows per quantization group and columns per CTA tile
THREADS = 512
VEC = 8  # BF16 columns per thread (one 16-byte load)


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


G_CAP = 1024  # expert groups the shared-memory offset table holds
STAGES = 2  # cp.async tiles in flight per CTA (backward cannot fit 3 in shared memory)


class ThreadCoords:
    """
    Lane and warp of thread tidx, and its place in a tile: column vector col_vec_idx (columns
    col_vec_idx * 8 .. col_vec_idx * 8 + 7) and row group row_group_idx (rows row_group_idx * 4 ..
    row_group_idx * 4 + 3).
    """

    def __init__(self, tidx) -> None:
        self.tidx = tidx
        self.lane, self.warp, self.col_vec_idx, self.row_group_idx = tidx % 32, tidx // 32, tidx % 16, tidx // 16


class Tile:
    def __init__(self, row_block_idx, col_block_idx, col_vec_idx, N: int) -> None:
        self.row_block_idx, self.col_block_idx = row_block_idx, col_block_idx
        self.row_start, self.col_start = row_block_idx * BLOCK, col_block_idx * BLOCK
        self.col = self.col_start + col_vec_idx * VEC
        self.col_ok = Int32(1)
        if cutlass.const_expr(N % BLOCK != 0):
            self.col_ok = Int32(self.col < N)


class Output:
    """
    Where one quantized output goes: tok (num_rows, num_channels) with its MN-major rowwise scale,
    and the K-major copy with its MN-major per-channel scale. The output's own channels start at
    channel_offset and its column blocks at col_block_offset (non-zero only for du in the
    fused gate_up gradient).
    """

    def __init__(self, tensors: tuple, num_rows, num_channels: int, channel_offset: int, col_block_offset: int) -> None:
        self.tok_base, self.tok_scale_base, self.kmajor_base, self.kmajor_scale_base = (tensor.iterator.toint() for tensor in tensors)
        self.num_rows, self.num_channels = num_rows, num_channels
        self.channel_offset, self.col_block_offset = channel_offset, col_block_offset


class SharedBuffers:
    """
    The CTA's shared memory: the cp.async stages, the transposed FP8 tile, one column-amax partial
    per warp, the column reciprocals and the expert-group offsets.
    """

    def __init__(self, num_stage_bytes: int) -> None:
        smem = cutlass.memory.SmemAllocator()
        self.s_stages = smem.allocate(num_stage_bytes, byte_alignment=128).toint()
        self.s_kmajor = smem.allocate(BLOCK * BLOCK, byte_alignment=128).toint()
        self.s_col_amax = smem.allocate((THREADS // 32) * 16 * 16, byte_alignment=16).toint()
        self.s_col_rcp = smem.allocate(BLOCK * 4, byte_alignment=16).toint()
        self.s_offs = smem.allocate(G_CAP * 4, byte_alignment=16).toint()


class SiluMulQuant:
    """
    Persistent kernel (one 512-thread CTA per SM over 128 x 128 tiles), compiled per (N, backward,
    device). Inputs are staged into shared memory with cp.async, STAGES tiles ahead. Every thread
    reads back only the 16-byte chunks it copied itself, so the staging needs no barrier, only
    cp.async.wait_group. Each thread owns a 4-row x 8-column block of the tile (ThreadCoords). A
    tile's 128 rows lie inside one expert group.
    """

    def __init__(self, N: int, backward: bool, num_ctas: int):
        self.N = N
        self.backward = backward
        self.num_col_blocks = (N + BLOCK - 1) // BLOCK
        self.num_inputs = 3 if backward else 2
        self.num_outputs = 2 if backward else 1
        self.stages = STAGES
        self.num_ctas = num_ctas
        self.stage_bytes = self.num_inputs * 4 * THREADS * 16

    @cute.jit
    def __call__(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor, mOffs: cute.Tensor, mTok0: cute.Tensor, mTokS0: cute.Tensor, mKm0: cute.Tensor, mKmS0: cute.Tensor, mTok1: cute.Tensor, mTokS1: cute.Tensor, mKm1: cute.Tensor, mKmS1: cute.Tensor, stream):
        num_tiles = (mG.shape[0] // BLOCK) * self.num_col_blocks
        self.kernel(mDh, mG, mU, mOffs, mTok0, mTokS0, mKm0, mKmS0, mTok1, mTokS1, mKm1, mKmS1).launch(grid=[cutlass.min(num_tiles, self.num_ctas), 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor, mOffs: cute.Tensor, mTok0: cute.Tensor, mTokS0: cute.Tensor, mKm0: cute.Tensor, mKmS0: cute.Tensor, mTok1: cute.Tensor, mTokS1: cute.Tensor, mKm1: cute.Tensor, mKmS1: cute.Tensor):
        M = mG.shape[0]
        out_tensors = [(mTok0, mTokS0, mKm0, mKmS0), (mTok1, mTokS1, mKm1, mKmS1)]
        outputs = [Output(out_tensors[i], M, self.N, 0, 0) for i in range(self.num_outputs)]
        self.process_tiles(self.input_bases(mDh, mG, mU), mOffs.iterator.toint(), outputs, M, mOffs.shape[0])

    def input_bases(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor) -> list:
        inputs = [mG, mU]
        if cutlass.const_expr(self.backward):
            inputs = [mDh, mG, mU]
        return [tensor.iterator.toint() for tensor in inputs]

    def tile_coords(self, tile_idx: Int32) -> tuple:
        """
        The column blocks of one row block get adjacent tile indices, for read locality.
        """
        return tile_idx // self.num_col_blocks, tile_idx % self.num_col_blocks

    def issue_stage(self, tile_idx: Int32, stage_addr: Int32, num_tiles: Int32, tidx: Int32, input_bases: list):
        """
        cp.async tile tile_idx (zero-filled when out of range) into a stage.
        """
        N = self.N
        copy_ok = Int32(tile_idx < num_tiles)
        row_block_idx, col_block_idx = self.tile_coords(tile_idx)
        row_idx, col_idx = row_block_idx * BLOCK + (tidx // 16) * 4, col_block_idx * BLOCK + (tidx % 16) * VEC
        if cutlass.const_expr(N % BLOCK != 0):
            copy_ok = copy_ok & Int32(col_idx < N)
        byte_offset = (Int64(row_idx) * N + col_idx) * 2 * Int64(copy_ok)
        for input_idx in range(self.num_inputs):
            for k in range(4):
                cp_async16(stage_addr + ((input_idx * 4 + k) * THREADS + tidx) * 16, input_bases[input_idx] + byte_offset + k * N * 2, copy_ok * 16)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def process_tiles(self, input_bases: list, offs_base: Int64, outputs: list, M: Int32, G: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        grid_size, _, _ = cute.arch.grid_dim()
        thread = ThreadCoords(tidx)
        num_row_blocks = M // BLOCK
        num_tiles = num_row_blocks * self.num_col_blocks
        smem = SharedBuffers(self.stages * self.stage_bytes)
        for i in cutlass.range_constexpr(self.stages):
            self.issue_stage(bx + i * grid_size, smem.s_stages + i * self.stage_bytes, num_tiles, tidx, input_bases)
        self.load_group_offsets(offs_base, smem.s_offs, tidx, G)
        num_iters = (num_tiles - bx + grid_size - 1) // grid_size
        for iter_idx in cutlass.range(num_iters, unroll=1):
            tile_idx = bx + iter_idx * grid_size
            tile = Tile(*self.tile_coords(tile_idx), thread.col_vec_idx, self.N)
            stage_addr = smem.s_stages + (iter_idx % self.stages) * self.stage_bytes
            cute.arch.cp_async_wait_group(self.stages - 1)
            out_words = self.apply_swiglu(self.load_stage(stage_addr, tidx))
            # the stage is consumed: refill it with the tile self.stages iterations ahead
            self.issue_stage(tile_idx + self.stages * grid_size, stage_addr, num_tiles, tidx, input_bases)
            group_start, num_group_rows = self.find_group(smem.s_offs, tile.row_start, thread.lane, G)
            for out_idx in cutlass.range_constexpr(self.num_outputs):
                self.quantize_output(out_words[out_idx], outputs[out_idx], tile, thread, group_start, num_group_rows, smem)
        cute.arch.cp_async_wait_group(0)

    @cute.jit
    def load_group_offsets(self, offs_base: Int64, s_offs: Int32, tidx: Int32, G: Int32):
        for i in cutlass.range((G + THREADS - 1) // THREADS, unroll=1):
            group_idx = i * THREADS + tidx
            if group_idx < G:
                group_end = ld_b32_coherent(offs_base + Int64(group_idx) * 4, Int32(1), Int32(0))
                sts_b32_nopred(s_offs + group_idx * 4, Uint32(group_end))
        cute.arch.sync_threads()

    @cute.jit
    def load_stage(self, stage_addr: Int32, tidx: Int32) -> list:
        """
        The thread's words of a staged tile: stage_words[input_idx][k] holds the 8 bf16 values of
        its row k.
        """
        stage_words = [[None] * 4 for _ in range(self.num_inputs)]
        for input_idx in cutlass.range_constexpr(self.num_inputs):
            for k in cutlass.range_constexpr(4):
                stage_words[input_idx][k] = lds_v4(stage_addr + ((input_idx * 4 + k) * THREADS + tidx) * 16)
        return stage_words

    @cute.jit
    def apply_swiglu(self, stage_words: list) -> list:
        """
        SwiGLU (backward: its two gradients) in fp32, rounded to bf16 and packed in pairs:
        out_words[out_idx][k][j] holds columns 2j and 2j + 1 of the thread's row k.
        """
        out_words = [[[None] * 4 for _ in range(4)] for _ in range(self.num_outputs)]
        for k in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.backward):
                    grad_h, gate, up = stage_words[0][k][j], stage_words[1][k][j], stage_words[2][k][j]
                    grad_gate_lo, grad_up_lo = swiglu_bwd(bf16_lo(grad_h), bf16_lo(gate), bf16_lo(up))
                    grad_gate_hi, grad_up_hi = swiglu_bwd(bf16_hi(grad_h), bf16_hi(gate), bf16_hi(up))
                    out_words[0][k][j] = pack_bf16(grad_gate_lo, grad_gate_hi)
                    out_words[1][k][j] = pack_bf16(grad_up_lo, grad_up_hi)
                else:
                    gate, up = stage_words[0][k][j], stage_words[1][k][j]
                    h_lo, h_hi = swiglu_fwd(bf16_lo(gate), bf16_lo(up)), swiglu_fwd(bf16_hi(gate), bf16_hi(up))
                    out_words[0][k][j] = pack_bf16(h_lo, h_hi)
        return out_words

    @cute.jit
    def find_group(self, s_offs: Int32, row_start: Int32, lane: Int32, G: Int32) -> tuple:
        """
        Start row and row count of group min(searchsorted(offs, row_start, right), G - 1), the group
        that holds the tile.
        """
        # each round a warp ballot counts the groups among 32 offsets that end at or before the tile
        num_groups_before = Int32(0)
        for i in cutlass.range((G + 31) // 32, unroll=1):
            probe_idx = i * 32 + lane
            probe_end = lds_b32_pred(s_offs + probe_idx * 4, Int32(probe_idx < G), Int32(0x7FFFFFFF))
            num_groups_before += cute.arch.popc(cute.arch.vote_ballot_sync(probe_end <= row_start))
        group_idx = cutlass.min(num_groups_before, G - 1)
        group_end, group_start = lds_b32_pred(s_offs + group_idx * 4, Int32(1), Int32(0)), lds_b32_pred(s_offs + (group_idx - 1) * 4, Int32(group_idx > 0), Int32(0))
        return group_start, Int64(group_end - group_start)

    @cute.jit
    def quantize_output(self, words: list, output: Output, tile: Tile, thread: ThreadCoords, group_start: Int32, num_group_rows: Int64, smem: SharedBuffers):
        abs_words = [[abs2(words[k][j]) for j in range(4)] for k in range(4)]
        self.reduce_col_amax(abs_words, thread, smem.s_col_amax)
        row_amax = self.reduce_row_amax(abs_words)
        self.quantize_rows(words, row_amax, output, tile, thread)
        cute.arch.sync_threads()
        self.store_col_scales(output, tile, thread.tidx, smem.s_col_amax, smem.s_col_rcp)
        cute.arch.sync_threads()
        self.quantize_cols(words, thread, smem.s_col_rcp, smem.s_kmajor)
        cute.arch.sync_threads()
        self.store_kmajor(output, tile, thread.tidx, group_start, num_group_rows, smem.s_kmajor)

    @cute.jit
    def reduce_col_amax(self, abs_words: list, thread: ThreadCoords, s_col_amax: Int32):
        """
        Column amax over the thread's 4 rows, then the warp's other row group, written as one
        partial per warp into s_col_amax.
        """
        lane, warp, col_vec_idx = thread.lane, thread.warp, thread.col_vec_idx
        col_amax = [None] * 4
        for j in cutlass.range_constexpr(4):
            acc_amax = max2(max2(abs_words[0][j], abs_words[1][j]), max2(abs_words[2][j], abs_words[3][j]))
            peer_amax = Uint32(cute.arch.shuffle_sync_bfly(Int32(acc_amax), offset=16))
            col_amax[j] = max2(acc_amax, peer_amax)
        if lane < 16:
            sts_v4(s_col_amax + (warp * 16 + col_vec_idx) * 16, tuple(col_amax))

    @cute.jit
    def reduce_row_amax(self, abs_words: list) -> list:
        """
        Amax of the thread's 4 rows over the whole tile as packed bf16 bit patterns, reduced by
        butterfly over the 16 lanes of the row group: row_amax[pair] packs rows 2 * pair and 2 *
        pair + 1.
        """
        row_amax_halves = [max2(max2(abs_words[k][0], abs_words[k][1]), max2(abs_words[k][2], abs_words[k][3])) for k in range(4)]
        row_amax = [None] * 2
        for pair in cutlass.range_constexpr(2):
            amax_even, amax_odd = row_amax_halves[2 * pair], row_amax_halves[2 * pair + 1]
            acc_amax = max2(prmt_args_last(amax_even, amax_odd, 0x5410), prmt_args_last(amax_even, amax_odd, 0x7632))
            for i in cutlass.range_constexpr(4):
                peer_amax = Uint32(cute.arch.shuffle_sync_bfly(Int32(acc_amax), offset=1 << i))
                acc_amax = max2(acc_amax, peer_amax)
            row_amax[pair] = acc_amax
        return row_amax

    @cute.jit
    def quantize_rows(self, words: list, row_amax: list, output: Output, tile: Tile, thread: ThreadCoords):
        lane, col_vec_idx, row_group_idx = thread.lane, thread.col_vec_idx, thread.row_group_idx
        # lane col_vec_idx < 4 of each row group scales row row_group_idx * 4 + col_vec_idx
        own_row_amax = bf16_sel(Int32(col_vec_idx % 2), selp_args_last(Int32(col_vec_idx < 2), row_amax[0], row_amax[1]))
        row_scale, row_rcp_lane = e8m0_scale_f32(own_row_amax)
        st_f32(output.tok_scale_base + (Int64(tile.col_block_idx + output.col_block_offset) * output.num_rows + tile.row_start + row_group_idx * 4 + col_vec_idx) * 4, row_scale, Int32(col_vec_idx < 4))
        for k in cutlass.range_constexpr(4):
            row_rcp = cute.arch.shuffle_sync(row_rcp_lane, offset=(lane & 16) | k)
            rcp4 = (row_rcp, row_rcp, row_rcp, row_rcp)
            row_words = words[k]
            quant_lo = quant4_e4m3(bf16_lo(row_words[0]), bf16_hi(row_words[0]), bf16_lo(row_words[1]), bf16_hi(row_words[1]), rcp4)
            quant_hi = quant4_e4m3(bf16_lo(row_words[2]), bf16_hi(row_words[2]), bf16_lo(row_words[3]), bf16_hi(row_words[3]), rcp4)
            row_idx = tile.row_start + row_group_idx * 4 + k
            st_v2_args_last(output.tok_base + Int64(row_idx) * output.num_channels + output.channel_offset + tile.col, quant_lo, quant_hi, tile.col_ok)

    @cute.jit
    def store_col_scales(self, output: Output, tile: Tile, tidx: Int32, s_col_amax: Int32, s_col_rcp: Int32):
        """
        Thread tidx < 128 reduces the per-warp partials of tile column tidx, stores its E8M0 scale
        and leaves the reciprocal in s_col_rcp.
        """
        if tidx < BLOCK:
            acc_amax = Uint32(0)
            for i in cutlass.range_constexpr(THREADS // 32):
                warp_amax = lds_b32_pred(s_col_amax + (i * 16 + tidx // 8) * 16 + ((tidx % 8) // 2) * 4, Int32(1), Int32(0))
                acc_amax = max2(acc_amax, Uint32(warp_amax))
            col_scale, col_rcp = e8m0_scale_f32(bf16_sel(Int32(tidx % 2), acc_amax))
            col_idx = tile.col_start + tidx
            col_ok = Int32(1)
            if cutlass.const_expr(self.N % BLOCK != 0):
                col_ok = Int32(col_idx < self.N)
            st_f32(output.kmajor_scale_base + (Int64(tile.row_block_idx) * mn_pad(output.num_channels) + output.channel_offset + col_idx) * 4, col_scale, col_ok)
            sts_b32_nopred(s_col_rcp + tidx * 4, Uint32(f32_as_u32(col_rcp)))

    @cute.jit
    def quantize_cols(self, words: list, thread: ThreadCoords, s_col_rcp: Int32, s_kmajor: Int32):
        """
        Quantize the thread's 4 x 8 block columnwise into the transposed tile s_kmajor, its 16-byte
        chunks swizzled by col_vec_idx % 8.
        """
        col_vec_idx, row_group_idx = thread.col_vec_idx, thread.row_group_idx
        col_rcp_words = lds_v4(s_col_rcp + col_vec_idx * 32) + lds_v4(s_col_rcp + col_vec_idx * 32 + 16)
        swizzle = (col_vec_idx & 7) << 4
        for i in cutlass.range_constexpr(8):
            j = i // 2
            bf16_half = bf16_lo if i % 2 == 0 else bf16_hi
            col_rcp = u32_as_f32(col_rcp_words[i])
            quant = quant4_e4m3(bf16_half(words[0][j]), bf16_half(words[1][j]), bf16_half(words[2][j]), bf16_half(words[3][j]), (col_rcp,) * 4)
            local_col = col_vec_idx * VEC + i
            sts_b32_nopred(s_kmajor + local_col * BLOCK + (((row_group_idx // 4) << 4) ^ swizzle) + (row_group_idx % 4) * 4, quant)

    @cute.jit
    def store_kmajor(self, output: Output, tile: Tile, tidx: Int32, group_start: Int32, num_group_rows: Int64, s_kmajor: Int32):
        """
        Store the transposed tile into the K-major output, whose group block is (num_channels,
        num_group_rows) at group_start * num_channels, 8 threads writing the 128 contiguous bytes
        of each column.
        """
        kmajor_tile_base = output.kmajor_base + Int64(group_start) * output.num_channels + Int64(tile.row_start - group_start)
        for i in cutlass.range_constexpr(BLOCK * BLOCK // 16 // THREADS):
            flat_idx = i * THREADS + tidx
            local_col, chunk_idx = flat_idx // 8, flat_idx % 8
            chunk = lds_v4(s_kmajor + local_col * BLOCK + ((chunk_idx ^ ((local_col // 8) & 7)) << 4))
            col_idx = tile.col_start + local_col
            col_ok = Int32(1)
            if cutlass.const_expr(self.N % BLOCK != 0):
                col_ok = Int32(col_idx < self.N)
            st_v4_args_last(kmajor_tile_base + Int64(output.channel_offset + col_idx) * num_group_rows + chunk_idx * 16, chunk, col_ok)


class SiluMulQuantGU(SiluMulQuant):
    """
    SiluMulQuant reading g / u as the column halves of one (M, 2N) buffer and, in backward, writing
    dg and du as the column halves of one quantized (M, 2N) operand: output out_idx (0 = dg or h, 1
    = du) lands at channel offset out_idx * N of a num_channels = num_outputs * N channel layout.
    Requires N % 128 == 0 (every tile inside one half).
    """

    def __init__(self, N: int, backward: bool, num_ctas: int):
        super().__init__(N, backward, num_ctas)
        assert N % BLOCK == 0
        self.gu_row_stride = 2 * N
        self.num_channels = self.num_outputs * N

    @cute.jit
    def __call__(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokS: cute.Tensor, mKm: cute.Tensor, mKmS: cute.Tensor, stream):
        num_tiles = (mG.shape[0] // BLOCK) * self.num_col_blocks
        self.kernel(mDh, mG, mU, mOffs, mTok, mTokS, mKm, mKmS).launch(grid=[cutlass.min(num_tiles, self.num_ctas), 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokS: cute.Tensor, mKm: cute.Tensor, mKmS: cute.Tensor):
        M = mG.shape[0]
        outputs = [Output((mTok, mTokS, mKm, mKmS), M, self.num_channels, i * self.N, i * self.num_col_blocks) for i in range(self.num_outputs)]
        self.process_tiles(self.input_bases(mDh, mG, mU), mOffs.iterator.toint(), outputs, M, mOffs.shape[0])

    def issue_stage(self, tile_idx: Int32, stage_addr: Int32, num_tiles: Int32, tidx: Int32, input_bases: list):
        N = self.N
        copy_ok = Int32(tile_idx < num_tiles)
        row_block_idx, col_block_idx = self.tile_coords(tile_idx)
        row_idx, col_idx = row_block_idx * BLOCK + (tidx // 16) * 4, col_block_idx * BLOCK + (tidx % 16) * VEC
        for input_idx in range(self.num_inputs):
            row_stride = N if (self.backward and input_idx == 0) else self.gu_row_stride
            byte_offset = (Int64(row_idx) * row_stride + col_idx) * 2 * Int64(copy_ok)
            for k in range(4):
                cp_async16(stage_addr + ((input_idx * 4 + k) * THREADS + tidx) * 16, input_bases[input_idx] + byte_offset + k * row_stride * 2, copy_ok * 16)
        cute.arch.cp_async_commit_group()


COMPILED: dict = {}


def get_kernel(N: int, backward: bool, device: torch.device) -> object:
    key = ("silu_mul_quant", N, backward, device.index)
    fn = COMPILED.get(key)
    if fn is None:
        M = cute.sym_int()
        num_col_blocks = (N + BLOCK - 1) // BLOCK
        x = fake(cutlass.BFloat16, (M, N), (N, 1))
        fp8_type = cutlass.Float8E4M3FN
        out = (fake(fp8_type, (M, N), (N, 1)), fake(Float32, (M, num_col_blocks), (1, cute.sym_int64()), 16), fake(fp8_type, (cute.sym_int(),), (1,)), fake(Float32, (N, cute.sym_int()), (1, mn_pad(N)), 16))
        offs = fake(Int32, (cute.sym_int(),), (1,), 4)
        fn = cute.compile(SiluMulQuant(N, backward, SM_COUNT), x if backward else None, x, x, offs, *out, *(out if backward else (None,) * 4), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    return fn


def get_kernel_gu(N: int, backward: bool, device: torch.device) -> object:
    key = ("silu_mul_quant_gu", N, backward, device.index)
    fn = COMPILED.get(key)
    if fn is None:
        M = cute.sym_int()
        num_channels = 2 * N if backward else N
        gu_half = fake(cutlass.BFloat16, (M, N), (2 * N, 1))
        fp8_type = cutlass.Float8E4M3FN
        out = (fake(fp8_type, (M, num_channels), (num_channels, 1)), fake(cutlass.Float32, (M, num_channels // BLOCK), (1, cute.sym_int64()), 16), fake(fp8_type, (cute.sym_int(),), (1,)), fake(cutlass.Float32, (num_channels, cute.sym_int()), (1, mn_pad(num_channels)), 16))
        offs = fake(Int32, (cute.sym_int(),), (1,), 4)
        fn = cute.compile(SiluMulQuantGU(N, backward, SM_COUNT), fake(cutlass.BFloat16, (M, N), (N, 1)) if backward else None, gu_half, gu_half, offs, *out, fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    return fn


def alloc_rowwise_kmajor_outputs(M: int, N: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The four outputs of fp8cast_rowwise_kmajor of an (M, N) input, both scales MN-major, each its
    own allocation (different lifetimes: the rowwise pair feeds the next GEMM, the K-major pair
    waits for wgrad).
    """
    num_col_blocks = (N + BLOCK - 1) // BLOCK
    return (torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), mn_major_scale(M, num_col_blocks, device), torch.empty(M * N, dtype=torch.float8_e4m3fn, device=device), mn_major_scale(N, M // BLOCK, device))


def silu_mul_quant_fusable(N: int, grouped_mm_offs: torch.Tensor) -> bool:
    """
    Shapes the fused kernel handles; others take the bit-identical unfused path.
    """
    return N % VEC == 0 and 0 < grouped_mm_offs.shape[0] <= G_CAP


def silu_mul_quant_gu_fusable(N: int, M: int, grouped_mm_offs: torch.Tensor) -> bool:
    """
    Shapes the fused gate_up kernel handles; others take the bit-identical unfused path.
    """
    return N % BLOCK == 0 and M % BLOCK == 0 and 0 < grouped_mm_offs.shape[0] <= G_CAP


def fp8_silu_mul_fwd_sm90(g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = g.shape
    if M != 0 and N != 0 and not silu_mul_quant_fusable(N, grouped_mm_offs):
        return fp8cast_rowwise_kmajor_sm90(silu_mul_fwd(g, u), grouped_mm_offs)
    out = alloc_rowwise_kmajor_outputs(M, N, g.device)
    if M == 0 or N == 0:
        return out
    get_kernel(N, False, g.device)(None, g, u, grouped_mm_offs, *out, None, None, None, None)
    return out


def fp8_silu_mul_bwd_sm90(dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    M, N = g.shape
    if M == 0 or N == 0:
        return (alloc_rowwise_kmajor_outputs(M, N, g.device), alloc_rowwise_kmajor_outputs(M, N, g.device))
    if not silu_mul_quant_fusable(N, grouped_mm_offs):
        grad_gate, grad_up = silu_mul_bwd(dh, g, u)
        return fp8cast_rowwise_kmajor_sm90(grad_gate, grouped_mm_offs), fp8cast_rowwise_kmajor_sm90(grad_up, grouped_mm_offs)
    grad_gate_out, grad_up_out = alloc_rowwise_kmajor_outputs(M, N, g.device), alloc_rowwise_kmajor_outputs(M, N, g.device)
    get_kernel(N, True, g.device)(dh, g, u, grouped_mm_offs, *grad_gate_out, *grad_up_out)
    return grad_gate_out, grad_up_out


def fp8_silu_mul_gu_fwd_sm90(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return alloc_rowwise_kmajor_outputs(0, N, gu.device)
    if not silu_mul_quant_gu_fusable(N, M, grouped_mm_offs):
        gate, up = gu[:, :N].contiguous(), gu[:, N:].contiguous()
        return fp8cast_rowwise_kmajor_sm90(silu_mul_fwd(gate, up), grouped_mm_offs)
    out = alloc_rowwise_kmajor_outputs(M, N, gu.device)
    get_kernel_gu(N, False, gu.device)(None, gu[:, :N], gu[:, N:], grouped_mm_offs, *out)
    return out


def fp8_silu_mul_gu_bwd_sm90(dh: torch.Tensor, gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return alloc_rowwise_kmajor_outputs(0, 2 * N, gu.device)
    if not silu_mul_quant_gu_fusable(N, M, grouped_mm_offs):
        gate, up = gu[:, :N].contiguous(), gu[:, N:].contiguous()
        grad_gate, grad_up = silu_mul_bwd(dh, gate, up)
        return fp8cast_rowwise_kmajor_sm90(torch.cat([grad_gate, grad_up], 1), grouped_mm_offs)
    out = alloc_rowwise_kmajor_outputs(M, 2 * N, gu.device)
    get_kernel_gu(N, True, gu.device)(dh, gu[:, :N], gu[:, N:], grouped_mm_offs, *out)
    return out
