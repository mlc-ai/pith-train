"""
Causal depthwise conv1d on SM90. Contract, the PyTorch recipe the verifier checks: each y and dx
element accumulates its K taps in order k = 0 .. K - 1 as fp32 fma (ATen's depthwise order; w read
as fp32, a missing tap of the causal padding adds w * 0), then rounds once to bf16 (round to nearest
even), so y and dx are bit-identical to ATen's conv1d and its backward. Backward reduces the
g * x terms in FP64 in a fixed order before converting dw to w's dtype. K is at most MAX_TAPS
(ValueError beyond).

The TMA kernels use contiguous, 16-byte aligned BF16 inputs with a channel count divisible by 64
and 2 <= K <= 4. Forward uses one sequence and S divisible by eight. Backward uses one sequence,
or two sequences at the target shape, with channels-last g. Each CTA processes a 64-channel tile
over a sequence or a row span. A producer loads each stage into shared memory with the necessary
halo. The tensor descriptor supplies zeros at its bounds. Backward reduces in FP64 within each CTA
and merges span partials in FP64 when a sequence uses multiple CTAs for a channel tile.

Direct kernels: each thread owns TOKENS consecutive rows x 8 channels: lane % 4 picks the channel
group, lane // 4 the token block, so a warp reads eight 64-byte row segments of the channels-last
tensors and writes 8 x 16 bytes per channel of the channels-first ones, with no transpose through
shared memory.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Float64, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, atom_add_u64, bf16_hi, bf16_lo, f32_as_u32, fence_gpu, ld_relaxed, ld_v4, lds_b32, lds_v4, pack_bf16, red_or_shared, st_b64, st_f32, st_v2, st_v4, sts_b32, to_ir, u32_as_f32

__all__ = ["causal_conv1d_bwd_sm90", "causal_conv1d_fwd_sm90"]

GROUPS_PER_WARP, BLOCKS_PER_WARP = 4, 8  # channel groups of 8 and token blocks of TOKENS per warp
FWD_TOKENS, FWD_WARPS = 8, 4
BWD_TOKENS, BWD_WARPS = 8, 4
CHANNELS_PER_CTA = GROUPS_PER_WARP * 8
TARGET_CTAS = 8  # per SM, for the backward's token split
TMA_CHANNELS = 64  # channels per CTA of the TMA kernels: one 128-byte box row
TMA_CONSUMERS = 128  # consumer threads per TMA group
TMA_MAX_TAPS = 4  # K of the TMA kernels: 2 .. 4 (box rows and registers checked for these)
MAX_TAPS = 8  # K of any input: the direct kernels keep K taps per channel in registers and K partials per channel in smem
BWD_TMA_ROWS, BWD_TMA_STAGES = 64, 4
BWD_TMA_THREAD_CHANNELS = 4
FWD_TMA_ROWS, FWD_TMA_STAGES, FWD_TMA_GROUPS = 128, 8, 2
SMEM_LIMIT = 227 * 1024


@dsl_user_op
def fma_f32(first: Float32, second: Float32, acc: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(asm(T.f32(), [to_ir(first, Float32, loc, ip), to_ir(second, Float32, loc, ip), to_ir(acc, Float32, loc, ip)], "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f"))


@dsl_user_op
def fma_f64(first: Float64, second: Float64, acc: Float64, *, loc=None, ip=None) -> Float64:
    return Float64(asm(T.f64(), [to_ir(first, Float64, loc, ip), to_ir(second, Float64, loc, ip), to_ir(acc, Float64, loc, ip)], "fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d"))


@dsl_user_op
def max_abs_operand(prior: Float32, value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(asm(
        T.f32(), [to_ir(prior, Float32, loc, ip), to_ir(value, Float32, loc, ip)],
        "{ .reg .f32 mag; abs.f32 mag, $2; max.f32 $0, $1, mag; }", "=f,f,f",
    ))


@dsl_user_op
def warp_max_magnitude(value: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(
        T.i32(), [to_ir(value, Uint32, loc, ip)],
        "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r",
    ))


@dsl_user_op
def round_dw_bf16(value: Float64, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(
        T.i32(), [to_ir(value, Float64, loc, ip)],
        "{ .reg .b16 bits; cvt.rn.bf16.f64 bits, $1; cvt.u32.u16 $0, bits; }", "=r,d",
    ))


@dsl_user_op
def lds_f64(addr: Int32, *, loc=None, ip=None) -> Float64:
    return Float64(asm(T.f64(), [to_ir(addr, Int32, loc, ip)], "ld.shared.f64 $0, [$1];", "=d,r", side_effects=True))


@dsl_user_op
def sts_f64(addr: Int32, value: Float64, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [to_ir(addr, Int32, loc, ip), to_ir(value, Float64, loc, ip), to_ir(pred, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.shared.f64 [$0], $1;\n\t}", "r,d,r", side_effects=True)


@dsl_user_op
def ld_relaxed_f64(addr: Int64, pred: Int32, *, loc=None, ip=None) -> Float64:
    return Float64(asm(T.f64(), [to_ir(addr, Int64, loc, ip), to_ir(pred, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b64 $0, 0;\n\t@p ld.relaxed.gpu.global.f64 $0, [$1];\n\t}",
        "=d,l,r", side_effects=True))


@dsl_user_op
def st_dw_16(addr: Int64, value: Float64, pred: Int32, dtype: cutlass.Constexpr, *, loc=None, ip=None) -> None:
    ptx_dtype = "f16" if dtype is cutlass.Float16 else "bf16"
    asm(
        None,
        [to_ir(addr, Int64, loc, ip), to_ir(value, Float64, loc, ip), to_ir(pred, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\t.reg .b16 rounded;\n\tsetp.ne.b32 p, $2, 0;\n\t"
        f"cvt.rn.{ptx_dtype}.f64 rounded, $1;\n\t"
        "@p st.global.b16 [$0], rounded;\n\t}",
        "l,d,r",
        side_effects=True,
    )


def store_dw_value(base: Int64, index: Int64, value: Float64, pred: Int32, dtype: cutlass.Constexpr) -> None:
    """Round the completed FP64 sum directly to the output type."""
    if dtype is Float64:
        st_b64(base + index * 8, value.bitcast(Int64), pred)
    elif dtype is Float32:
        st_f32(base + index * 4, value.to(Float32), pred)
    else:
        st_dw_16(base + index * 2, value, pred, dtype)


@dsl_user_op
def ld_b16(addr: Int64, pred: Int32, *, loc=None, ip=None) -> Uint32:
    """
    2-byte global load, zero-extended; 0 where pred == 0.
    """
    return Uint32(asm(T.i32(), [to_ir(addr, Int64, loc, ip), to_ir(pred, Int32, loc, ip)], "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b16 h, 0;\n\t@p ld.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}", "=r,l,r"))


@dsl_user_op
def st_b16(addr: Int64, value: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    """
    Store the low 2 bytes of value.
    """
    asm(None, [to_ir(addr, Int64, loc, ip), to_ir(value, Uint32, loc, ip), to_ir(pred, Int32, loc, ip)], "{\n\t.reg .pred p;\n\t.reg .b16 l, h;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 {l, h}, $1;\n\t@p st.global.b16 [$0], l;\n\t}", "l,r,r", side_effects=True)


@dsl_user_op
def mbar_wait(addr: Int32, parity: Int32, *, loc=None, ip=None) -> None:
    """
    Spin until the phase of the mbarrier at addr with the given parity has completed.
    """
    asm(None, [to_ir(addr, Int32, loc, ip), to_ir(parity, Int32, loc, ip)], "{\n\t.reg .pred p;\n\tLAB_WAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n\t@!p bra LAB_WAIT;\n\t}", "r,r", side_effects=True)


@dsl_user_op
def expect_slot_bytes(addr: Int32, num_bytes: int, *, loc=None, ip=None) -> None:
    """
    The producer's arrival on a slot's full barrier at addr, expecting num_bytes of TMA data.
    """
    asm(None, [to_ir(addr, Int32, loc, ip)], f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {num_bytes};", "r", side_effects=True)


@dsl_user_op
def release_slot(addr: Int32, *, loc=None, ip=None) -> None:
    """
    A consumer is done reading a ring slot: fence its generic-proxy reads against the next TMA
    (async-proxy) write into the slot, then arrive on the slot's empty barrier at addr.
    """
    asm(None, [to_ir(addr, Int32, loc, ip)], "fence.proxy.async.shared::cta;\n\tmbarrier.arrive.shared::cta.b64 _, [$0];", "r", side_effects=True)


@dsl_user_op
def tma_load_rows(dst: Int32, desc: Int64, chan0: Int32, row0: Int32, full: Int32, *, loc=None, ip=None) -> None:
    """
    TMA tile load of the box at (channel chan0, row row0) of the tensor desc describes into smem
    dst, completing on the mbarrier full; rows outside the tensor (row0 may be negative) are zeros.
    """
    asm(None, [to_ir(dst, Int32, loc, ip), to_ir(desc, Int64, loc, ip), to_ir(chan0, Int32, loc, ip), to_ir(row0, Int32, loc, ip), to_ir(full, Int32, loc, ip)], "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];", "r,l,r,r,r", side_effects=True)


@dsl_user_op
def tma_store_rows(desc: Int64, chan0: Int32, row0: Int32, src: Int32, *, loc=None, ip=None) -> None:
    asm(
        None, [to_ir(desc, Int64, loc, ip), to_ir(chan0, Int32, loc, ip),
               to_ir(row0, Int32, loc, ip), to_ir(src, Int32, loc, ip)],
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [$0, {$1, $2}], [$3];",
        "l,r,r,r", side_effects=True,
    )
    asm(None, [], "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 1;", "", side_effects=True)


@dsl_user_op
def fence_dx_stage(*, loc=None, ip=None) -> None:
    asm(None, [], "fence.proxy.async.shared::cta;", "", side_effects=True)


@dsl_user_op
def finish_dx_stores(*, loc=None, ip=None) -> None:
    asm(None, [], "cp.async.bulk.wait_group 0;", "", side_effects=True)


@dsl_user_op
def tma_desc_addr(tma_atom, *, loc=None, ip=None) -> Int64:
    """
    Generic address of a TMA atom's descriptor (a kernel parameter), for tma_load_rows.
    """
    exec_atom = cute_nvgpu.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    ptr_type = ir.Type.parse("!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>")
    return cute_nvgpu.get_tma_desc_addr(ptr_type, exec_atom, loc=loc, ip=ip).toint(loc=loc, ip=ip)


@dsl_user_op
def init_ring(full0: Int32, empty0: Int32, num_slots: int, *, loc=None, ip=None) -> None:
    """
    mbarriers of a ring of num_slots slots: full (one producer arrival plus the TMA bytes) and empty
    (one arrival per consumer thread), then the fence that publishes them to the async proxy.
    """
    for slot in range(num_slots):
        asm(None, [to_ir(full0 + 8 * slot, Int32, loc, ip)], "mbarrier.init.shared::cta.b64 [$0], 1;", "r", side_effects=True)
        asm(None, [to_ir(empty0 + 8 * slot, Int32, loc, ip)], f"mbarrier.init.shared::cta.b64 [$0], {TMA_CONSUMERS};", "r", side_effects=True)
    asm(None, [], "fence.mbarrier_init.release.cluster;", "", side_effects=True)


def element(words: list, idx: int) -> Float32:
    """
    Element idx of bf16 values packed two per word, as fp32.
    """
    return bf16_hi(words[idx // 2]) if idx % 2 else bf16_lo(words[idx // 2])


def load_chunk(addr: Int64, row_ok: Int32, chan0: Int32, C: cutlass.Constexpr, is_vector: cutlass.Constexpr) -> list:
    """
    The 8 bf16 channels chan0 .. chan0 + 7 of one channels-last row at addr, as 4 words; zeros where
    the row or a channel is out of range.
    """
    words = []
    if cutlass.const_expr(is_vector):
        words = list(ld_v4(addr, row_ok & Int32(chan0 < C)))
    else:
        for word_idx in range(4):
            word_lo = ld_b16(addr + 4 * word_idx, row_ok & Int32(chan0 + 2 * word_idx < C))
            word_hi = ld_b16(addr + 4 * word_idx + 2, row_ok & Int32(chan0 + 2 * word_idx + 1 < C))
            words.append(word_lo | (word_hi << 16))
    return words


def store_chunk(addr: Int64, words: list, row_ok: Int32, chan0: Int32, C: cutlass.Constexpr, is_vector: cutlass.Constexpr) -> None:
    """
    Store 4 words as the 8 bf16 channels chan0 .. chan0 + 7 of one channels-last row.
    """
    if cutlass.const_expr(is_vector):
        st_v4(addr, tuple(words), row_ok & Int32(chan0 < C))
    else:
        for word_idx in range(4):
            st_b16(addr + 4 * word_idx, words[word_idx], row_ok & Int32(chan0 + 2 * word_idx < C))
            st_b16(addr + 4 * word_idx + 2, words[word_idx] >> 16, row_ok & Int32(chan0 + 2 * word_idx + 1 < C))


def load_weights(w_base: Int64, chan0: Int32, C: cutlass.Constexpr, K: cutlass.Constexpr, is_w_fp32: cutlass.Constexpr, channels: int = 8) -> list:
    """
    w[chan, k] of the thread's channels as fp32, weights[chan][k]; zeros past C.
    """
    weights = []
    for chan in range(channels):
        chan_ok = Int32(chan0 + chan < C)
        taps = []
        for k in range(K):
            elem_addr = w_base + (Int64(chan0 + chan) * K + k) * (4 if is_w_fp32 else 2)
            if cutlass.const_expr(is_w_fp32):
                taps.append(u32_as_f32(Uint32(ld_relaxed(elem_addr, chan_ok))))
            else:
                taps.append(bf16_lo(ld_b16(elem_addr, chan_ok)))
        weights.append(taps)
    return weights


def sum_over_token_blocks(value: Float64) -> Float64:
    """
    Butterfly sum over the lanes of a warp that share a channel group.
    """
    lane_mask = GROUPS_PER_WARP
    while lane_mask < 32:
        value = value + cute.arch.shuffle_sync_bfly(value, offset=lane_mask)
        lane_mask *= 2
    return value


class ThreadTile:
    """
    The thread's channel group and token block: channels chan0 .. chan0 + 7 of channel tile by, and
    token block block_idx of tokens rows, starting block_offset rows into the CTA's chunk.
    """

    def __init__(self, tidx, by, tokens: int) -> None:
        lane, warp = tidx % 32, tidx // 32
        self.chan0 = by * CHANNELS_PER_CTA + (lane % GROUPS_PER_WARP) * 8
        self.block_idx = warp * BLOCKS_PER_WARP + lane // GROUPS_PER_WARP
        self.block_offset = self.block_idx * tokens


class ConvFwd:
    """
    One CTA per (FWD_WARPS * 8 * FWD_TOKENS)-token chunk, channel tile of 32 and batch entry.
    """

    def __init__(self, C: int, K: int, is_w_fp32: bool, is_row_vector: bool, is_col_vector: bool) -> None:
        self.C, self.K, self.is_w_fp32 = C, K, is_w_fp32
        self.is_row_vector, self.is_col_vector = is_row_vector, is_col_vector
        self.chunk_tokens = FWD_WARPS * BLOCKS_PER_WARP * FWD_TOKENS

    @cute.jit
    def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, B: Int32, L: Int32, S: Int32, offset: Int32, stream) -> None:
        num_chunks = (S + self.chunk_tokens - 1) // self.chunk_tokens
        num_chan_tiles = (self.C + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
        self.kernel(mX, mW, mY, L, S, offset).launch(grid=[num_chunks, num_chan_tiles, B], block=[FWD_WARPS * 32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, L: Int32, S: Int32, offset: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        tile = ThreadTile(tidx, by, FWD_TOKENS)
        out_start = bx * self.chunk_tokens + tile.block_offset
        x_base = mX.iterator.toint() + Int64(bz) * L * self.C * 2
        y_base = mY.iterator.toint() + Int64(bz) * self.C * S * 2
        weights = load_weights(mW.iterator.toint(), tile.chan0, self.C, self.K, self.is_w_fp32)
        rows = self.load_window(x_base, out_start + offset - (self.K - 1), L, tile.chan0)
        out_words = self.convolve(rows, weights)
        self.store_channels_first(y_base, out_words, out_start, S, tile.chan0)

    def load_window(self, x_base: Int64, row_start: Int32, L: Int32, chan0: Int32) -> list:
        """
        The FWD_TOKENS + K - 1 input rows the thread's outputs read; zeros outside [0, L).
        """
        rows = []
        for i in range(FWD_TOKENS + self.K - 1):
            row_idx = row_start + i
            row_ok = Int32(row_idx >= 0) & Int32(row_idx < L)
            rows.append(load_chunk(x_base + (Int64(row_idx) * self.C + chan0) * 2, row_ok, chan0, self.C, self.is_row_vector))
        return rows

    def convolve(self, rows: list, weights: list) -> list:
        """
        out_words[chan][j // 2] packs outputs j, j + 1 of channel chan (the channels-first order).
        """
        out_words = []
        for chan in range(8):
            outputs = []
            for j in range(FWD_TOKENS):
                acc = Float32(0.0)
                for k in range(self.K):
                    acc = fma_f32(weights[chan][k], element(rows[j + k], chan), acc)
                outputs.append(acc)
            out_words.append([pack_bf16(outputs[2 * i], outputs[2 * i + 1]) for i in range(FWD_TOKENS // 2)])
        return out_words

    def store_channels_first(self, y_base: Int64, out_words: list, out_start: Int32, S: Int32, chan0: Int32) -> None:
        for chan in range(8):
            chan_ok = Int32(chan0 + chan < self.C)
            chan_addr = y_base + (Int64(chan0 + chan) * S + out_start) * 2
            if cutlass.const_expr(self.is_col_vector):
                for i in range(FWD_TOKENS // 8):
                    st_v4(chan_addr + 16 * i, tuple(out_words[chan][4 * i : 4 * i + 4]), chan_ok & Int32(out_start + 8 * i < S))
            else:
                for j in range(FWD_TOKENS):
                    word = out_words[chan][j // 2]
                    st_b16(chan_addr + 2 * j, (word >> 16) if j % 2 else word, chan_ok & Int32(out_start + j < S))


class ConvBwd:
    """
    dx of input rows and the partial dw of the outputs t = row - offset, over num_iters chunks of
    BWD_WARPS * 8 * BWD_TOKENS rows per CTA (grid: channel tile, split, batch entry). Each CTA
    writes one FP64 partial of its 32 channels; the last CTA of a channel tile to finish sums the
    partials in order, writes dw and resets the tile's counter.

    The thread's window of g is the BWD_TOKENS + K - 1 output rows from row0 - offset: dx[row0 + j]
    reads g window entries j .. j + K - 1, dw reads entries 0 .. BWD_TOKENS - 1 against x rows
    row0 - (K - 1) .. row0 + BWD_TOKENS - 1.
    """

    def __init__(self, C: int, K: int, is_w_fp32: bool, is_row_vector: bool, is_g_channels_first: bool, is_g_vector: bool, g_shift: int, dw_dtype: type, certify: bool) -> None:
        self.C, self.K, self.is_w_fp32, self.is_row_vector = C, K, is_w_fp32, is_row_vector
        self.is_g_channels_first, self.is_g_vector, self.g_shift = is_g_channels_first, is_g_vector, g_shift
        self.dw_dtype, self.certify = dw_dtype, certify
        self.dw_bytes = BWD_WARPS * CHANNELS_PER_CTA * K * 8
        self.chunk_tokens = BWD_WARPS * BLOCKS_PER_WARP * BWD_TOKENS
        self.window = BWD_TOKENS + K - 1

    @cute.jit
    def __call__(self, mG: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, mPartials: cute.Tensor, mCounters: cute.Tensor, B: Int32, L: Int32, S: Int32, offset: Int32, g_stride_b: Int64, g_stride: Int64, num_splits: Int32, num_iters: Int32, stream) -> None:
        num_chan_tiles = (self.C + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
        self.kernel(mG, mX, mW, mDx, mDw, mPartials, mCounters, B, L, S, offset, g_stride_b, g_stride, num_splits, num_iters).launch(grid=[num_chan_tiles, num_splits, B], block=[BWD_WARPS * 32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mG: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, mPartials: cute.Tensor, mCounters: cute.Tensor, B: Int32, L: Int32, S: Int32, offset: Int32, g_stride_b: Int64, g_stride: Int64, num_splits: Int32, num_iters: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        tile = ThreadTile(tidx, bx, BWD_TOKENS)
        smem = cutlass.memory.SmemAllocator()
        if cutlass.const_expr(self.certify):
            s_dw = smem.allocate(self.dw_bytes + BWD_WARPS * 4, byte_alignment=16).toint()
            s_is_last = Int32(0)
        else:
            s_dw = smem.allocate(self.dw_bytes, byte_alignment=16).toint()
            s_is_last = smem.allocate(16, byte_alignment=16).toint()
        g_base = mG.iterator.toint() + Int64(bz) * g_stride_b * 2
        x_base = mX.iterator.toint() + Int64(bz) * L * self.C * 2
        dx_base = mDx.iterator.toint() + Int64(bz) * L * self.C * 2
        weights = load_weights(mW.iterator.toint(), tile.chan0, self.C, self.K, self.is_w_fp32)
        acc_dw = cute.make_rmem_tensor((8 * self.K,), Float64)
        for i in cutlass.range_constexpr(8 * self.K):
            acc_dw[i] = Float64(0.0)

        max_operand = Float32(0.0)
        for iter_idx in cutlass.range(num_iters, unroll=1):
            row0 = (by * num_iters + iter_idx) * self.chunk_tokens + tile.block_offset
            g_window = self.load_g_window(g_base, row0 - offset, S, g_stride, tile.chan0)
            x_rows = self.load_x_window(x_base, row0 - (self.K - 1), L, tile.chan0)
            self.store_dx(dx_base, g_window, weights, row0, L, tile.chan0)
            self.accumulate_dw(acc_dw, g_window, x_rows)
            if cutlass.const_expr(self.certify):
                for j in cutlass.range_constexpr(BWD_TOKENS):
                    for chan in cutlass.range_constexpr(8):
                        max_operand = max_abs_operand(max_operand, g_window[j][chan])
                for j in cutlass.range_constexpr(self.window):
                    for chan in cutlass.range_constexpr(8):
                        max_operand = max_abs_operand(max_operand, element(x_rows[j], chan))

        self.reduce_dw_in_cta(acc_dw, tidx, s_dw, max_operand)
        partial_idx = bz * num_splits + by
        self.store_partial(mPartials.iterator.toint(), s_dw, partial_idx, bx, tidx)
        if cutlass.const_expr(self.certify):
            if tidx == 0:
                tile_max = Float32(0.0)
                for warp in cutlass.range_constexpr(BWD_WARPS):
                    warp_magnitude = u32_as_f32(lds_b32(s_dw + self.dw_bytes + warp * 4))
                    tile_max = max_abs_operand(tile_max, warp_magnitude)
                tiles = (self.C + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
                index = Int64(B) * num_splits * self.C * self.K + Int64(partial_idx) * tiles + bx
                st_b64(mPartials.iterator.toint() + index * 8, tile_max.to(Float64).bitcast(Int64), Int32(1))
        else:
            counter_addr = mCounters.iterator.toint() + Int64(bx) * 8
            is_last = self.take_counter(counter_addr, B * num_splits, tidx, s_is_last)
            if is_last != 0:
                self.store_dw(mPartials.iterator.toint(), mDw.iterator.toint(), counter_addr, B * num_splits, bx, tidx)

    def load_g_window(self, g_base: Int64, out_start: Int32, S: Int32, g_stride: Int64, chan0: Int32) -> list:
        """
        g_window[i][chan] = g[out_start + i, chan0 + chan] as fp32, zero outside [0, S) and past C.
        g_stride is the stride between rows (channels-last) or between channels (channels-first).
        """
        C, window = self.C, self.window
        g_window = [[None] * 8 for _ in range(window)]
        if cutlass.const_expr(not self.is_g_channels_first):
            for i in range(window):
                out_idx = out_start + i
                row_ok = Int32(out_idx >= 0) & Int32(out_idx < S)
                words = load_chunk(g_base + (Int64(out_idx) * g_stride + chan0) * 2, row_ok, chan0, C, self.is_g_vector)
                for chan in range(8):
                    g_window[i][chan] = element(words, chan)
        elif cutlass.const_expr(self.is_g_vector):
            # out_start - g_shift and S are multiples of 8: each 8-token vector is in or out whole
            vec_start = out_start - self.g_shift
            num_vecs = (self.g_shift + window + 7) // 8
            for chan in range(8):
                chan_ok = Int32(chan0 + chan < C)
                chan_addr = g_base + Int64(chan0 + chan) * g_stride * 2
                words = []
                for vec_idx in range(num_vecs):
                    tok_idx = vec_start + 8 * vec_idx
                    words += list(ld_v4(chan_addr + Int64(tok_idx) * 2, chan_ok & Int32(tok_idx >= 0) & Int32(tok_idx < S)))
                for i in range(window):
                    g_window[i][chan] = element(words, self.g_shift + i)
        else:
            for chan in range(8):
                chan_ok = Int32(chan0 + chan < C)
                chan_addr = g_base + Int64(chan0 + chan) * g_stride * 2
                for i in range(window):
                    out_idx = out_start + i
                    g_window[i][chan] = bf16_lo(ld_b16(chan_addr + Int64(out_idx) * 2, chan_ok & Int32(out_idx >= 0) & Int32(out_idx < S)))
        return g_window

    def load_x_window(self, x_base: Int64, row_start: Int32, L: Int32, chan0: Int32) -> list:
        rows = []
        for i in range(self.window):
            row_idx = row_start + i
            row_ok = Int32(row_idx >= 0) & Int32(row_idx < L)
            rows.append(load_chunk(x_base + (Int64(row_idx) * self.C + chan0) * 2, row_ok, chan0, self.C, self.is_row_vector))
        return rows

    def store_dx(self, dx_base: Int64, g_window: list, weights: list, row0: Int32, L: Int32, chan0: Int32) -> None:
        """
        dx[row0 + j] = sum over k = 0 .. K - 1, in order, of w[k] * g_window[j + K - 1 - k].
        """
        K = self.K
        for j in range(BWD_TOKENS):
            words = []
            for pair in range(4):
                accs = []
                for chan in (2 * pair, 2 * pair + 1):
                    acc = Float32(0.0)
                    for k in range(K):
                        acc = fma_f32(weights[chan][k], g_window[j + K - 1 - k][chan], acc)
                    accs.append(acc)
                words.append(pack_bf16(accs[0], accs[1]))
            row_idx = row0 + j
            store_chunk(dx_base + (Int64(row_idx) * self.C + chan0) * 2, words, Int32(row_idx < L), chan0, self.C, self.is_row_vector)

    def accumulate_dw(self, acc_dw, g_window: list, x_rows: list) -> None:
        """
        acc_dw[chan * K + k] += g_window[j][chan] * x_rows[j + k][chan], over the thread's
        outputs j in order.
        """
        K = self.K
        for chan in range(8):
            for k in range(K):
                acc = acc_dw[chan * K + k]
                for j in range(BWD_TOKENS):
                    acc = fma_f64(g_window[j][chan].to(Float64), element(x_rows[j + k], chan).to(Float64), acc)
                acc_dw[chan * K + k] = acc

    @cute.jit
    def reduce_dw_in_cta(self, acc_dw, tidx, s_dw: Int32, max_operand: Float32) -> None:
        """
        Sum acc_dw over the token blocks of each warp (butterfly), then lanes 0 .. 3 of warp w store
        their channel group's sums to s_dw[w][channel][k].
        """
        K = self.K
        lane, warp = tidx % 32, tidx // 32
        for i in cutlass.range_constexpr(8 * K):
            acc_dw[i] = sum_over_token_blocks(acc_dw[i])
        for i in cutlass.range_constexpr(8 * K):
            entry = (warp * CHANNELS_PER_CTA + lane * 8) * K + i
            sts_f64(s_dw + entry * 8, acc_dw[i], Int32(lane < GROUPS_PER_WARP))
        if cutlass.const_expr(self.certify):
            warp_max = warp_max_magnitude(f32_as_u32(max_operand))
            sts_b32(s_dw + self.dw_bytes + warp * 4, warp_max, Int32(lane == 0))
        cute.arch.sync_threads()

    @cute.jit
    def store_partial(self, partials_base: Int64, s_dw: Int32, partial_idx: Int32, chan_tile_idx: Int32, tidx) -> None:
        """
        partials[partial_idx, chan, k] = the sum over the warps, in order, of s_dw[w][chan][k].
        """
        K, C = self.K, self.C
        for i in cutlass.range_constexpr((CHANNELS_PER_CTA * K + BWD_WARPS * 32 - 1) // (BWD_WARPS * 32)):
            entry = i * BWD_WARPS * 32 + tidx
            entry_ok = Int32(entry < CHANNELS_PER_CTA * K)
            acc = Float64(0.0)
            for warp in cutlass.range_constexpr(BWD_WARPS):
                acc = acc + lds_f64(s_dw + (warp * CHANNELS_PER_CTA * K + entry * entry_ok) * 8)
            chan_idx = chan_tile_idx * CHANNELS_PER_CTA + entry // K
            st_b64(partials_base + ((Int64(partial_idx) * C + chan_idx) * K + entry % K) * 8, acc.bitcast(Int64), entry_ok & Int32(chan_idx < C))

    @cute.jit
    def take_counter(self, counter_addr: Int64, num_partials: Int32, tidx, s_is_last: Int32) -> Int32:
        """
        Count this CTA's partial as written; 1 in every thread of the CTA that counts the last one.
        """
        fence_gpu()  # each thread's partial stores before the barrier and the count
        cute.arch.sync_threads()
        if tidx == 0:
            num_done = atom_add_u64(counter_addr, Int64(1))
            sts_b32(s_is_last, Uint32(Int32(num_done == Int64(num_partials) - 1)), Int32(1))
        cute.arch.sync_threads()
        return Int32(lds_b32(s_is_last))

    @cute.jit
    def store_dw(self, partials_base: Int64, dw_base: Int64, counter_addr: Int64, num_partials: Int32, chan_tile_idx: Int32, tidx) -> None:
        """
        dw[chan, k] = the sum over the partials, in order, cast to w's dtype; resets the counter.
        """
        K, C = self.K, self.C
        fence_gpu()  # the other CTAs' partials, published before their count
        for i in cutlass.range_constexpr((CHANNELS_PER_CTA * K + BWD_WARPS * 32 - 1) // (BWD_WARPS * 32)):
            entry = i * BWD_WARPS * 32 + tidx
            chan_idx = chan_tile_idx * CHANNELS_PER_CTA + entry // K
            entry_ok = Int32(entry < CHANNELS_PER_CTA * K) & Int32(chan_idx < C)
            acc = Float64(0.0)
            for partial_idx in cutlass.range(num_partials, unroll=1):
                acc = acc + ld_relaxed_f64(partials_base + ((Int64(partial_idx) * C + chan_idx) * K + entry % K) * 8, entry_ok)
            elem_idx = Int64(chan_idx) * K + entry % K
            store_dw_value(dw_base, elem_idx, acc, entry_ok, self.dw_dtype)
        st_b64(counter_addr, Int64(0), Int32(tidx == 0))


class TmaRing:
    """
    Shared-memory ring of the TMA kernels: num_slots slots of slot_bytes, a full and an empty
    mbarrier per slot. Stage it uses slot it % num_slots in phase (it // num_slots) % 2; the
    producer waits on empty with the opposite parity, so its first pass does not block. Consumer
    groups take stages it = group, group + groups, ...; num_slots % groups == 0 keeps each slot with
    one group, so no group waits on a slot more than one phase ahead (a parity wait cannot tell
    phases two apart).
    """

    def __init__(self, num_slots: int, slot_bytes: int, groups: int) -> None:
        assert num_slots % groups == 0
        self.num_slots, self.slot_bytes = num_slots, slot_bytes

    def allocate(self, smem) -> tuple:
        slots = Int32(smem.allocate(self.num_slots * self.slot_bytes, byte_alignment=128).toint())
        bars = Int32(smem.allocate(2 * self.num_slots * 8, byte_alignment=8).toint())
        return slots, bars, bars + 8 * self.num_slots

    def slot(self, stage_idx: Int32) -> tuple:
        """
        (slot index, parity) of a stage.
        """
        return stage_idx % self.num_slots, (stage_idx // self.num_slots) % 2


class ConvBwdTma:
    """
    Backward on the TMA ring, one CTA per 64-channel tile and row span of one sequence. At each
    stage's first row r, the producer loads stage_rows + K - 1 rows of g and x starting at
    r - offset and r - (K - 1), respectively. Each consumer owns four channels and eight rows.
    Per row j in order it loads box rows j + K - 1 of g and x (a sliding window of K
    rows), writes dx[j] = sum over k in order of w[k] * g[j + K - 1 - k] (fp32 fma, ATen's order)
    with one eight-byte store, and adds g[j] * x[j + k] to its dw sums.

    dw order (deterministic, FP64): (1) each thread's sum[chan][k] over its rows
    in increasing order across all stages; (2) butterfly over the two token blocks of a warp sharing
    the channel group (shuffle xor 16); (3) the four warps' sums in warp order from 0.0
    through shared memory; (4) a store to the span scratch tensor, or a final cast for one span.
    """

    def __init__(self, C: int, K: int, is_w_fp32: bool, sequence_splits: int, dw_dtype: type, batched: bool = False, certify_single: bool = False) -> None:
        self.C, self.K, self.is_w_fp32 = C, K, is_w_fp32
        self.sequence_splits = sequence_splits
        self.dw_dtype = dw_dtype
        self.batched = batched
        self.sequences = 2 if batched else 1
        self.certify = batched or certify_single
        self.thread_channels = 2 if batched else BWD_TMA_THREAD_CHANNELS
        self.channel_groups = TMA_CHANNELS // self.thread_channels
        self.stage_rows = 32 if batched else BWD_TMA_ROWS
        self.tokens = self.stage_rows // (TMA_CONSUMERS // self.channel_groups)
        self.box_rows = self.stage_rows + K - 1
        self.box_bytes = self.box_rows * TMA_CHANNELS * 2
        self.box_stride = (self.box_bytes + 127) // 128 * 128
        self.ring = TmaRing(BWD_TMA_STAGES, 2 * self.box_stride, 1)
        self.dx_stage_bytes = self.stage_rows * TMA_CHANNELS * 2
        self.dx_slots = 3 if batched else 0
        self.producer_warp = 0
        self.dw_bytes = 4 * TMA_CHANNELS * K * 8
        self.reduction_bytes = self.dw_bytes + (TMA_CONSUMERS // 32 * 4 if self.certify else 0)
        assert (self.ring.num_slots * self.ring.slot_bytes + self.reduction_bytes
                + self.dx_slots * self.dx_stage_bytes + 1024 <= SMEM_LIMIT)

    @cute.jit
    def __call__(self, mG: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, L: Int32, offset: Int32, stream) -> None:
        box = cute.make_layout((self.box_rows, TMA_CHANNELS), stride=(TMA_CHANNELS, 1))
        atom_g, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mG, box, (self.box_rows, TMA_CHANNELS))
        atom_x, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mX, box, (self.box_rows, TMA_CHANNELS))
        self.kernel(atom_g, atom_x, atom_g, atom_x, atom_g, mW, mDx, mDw, L, offset).launch(grid=[self.C // TMA_CHANNELS, self.sequence_splits, 1], block=[32 + TMA_CONSUMERS, 1, 1], stream=stream)

    @cute.jit
    def batch_two(self, g0: cute.Tensor, x0: cute.Tensor, g1: cute.Tensor, x1: cute.Tensor,
                  mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, L: Int32, offset: Int32, stream) -> None:
        box = cute.make_layout((self.box_rows, TMA_CHANNELS), stride=(TMA_CHANNELS, 1))
        tile = (self.box_rows, TMA_CHANNELS)
        atom_g0, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), g0, box, tile)
        atom_x0, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), x0, box, tile)
        atom_g1, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), g1, box, tile)
        atom_x1, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), x1, box, tile)
        dx_tile = (self.stage_rows, TMA_CHANNELS)
        dx_box = cute.make_layout(dx_tile, stride=(TMA_CHANNELS, 1))
        atom_dx, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mDx, dx_box, dx_tile)
        self.kernel(atom_g0, atom_x0, atom_g1, atom_x1, atom_dx, mW, mDx, mDw, L, offset).launch(
            grid=[self.C // TMA_CHANNELS, self.sequence_splits, 2],
            block=[32 + TMA_CONSUMERS, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(self, atom_g: cute.CopyAtom, atom_x: cute.CopyAtom, atom_g1: cute.CopyAtom, atom_x1: cute.CopyAtom,
               atom_dx: cute.CopyAtom, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, L: Int32, offset: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, batch = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        smem = cutlass.memory.SmemAllocator()
        slots, full0, empty0 = self.ring.allocate(smem)
        s_dw = Int32(smem.allocate(self.reduction_bytes, byte_alignment=16).toint())
        s_dx = Int32(0)
        if cutlass.const_expr(self.batched):
            s_dx = Int32(smem.allocate(self.dx_slots * self.dx_stage_bytes, byte_alignment=128).toint())
        if tidx == 0:
            init_ring(full0, empty0, self.ring.num_slots)
        cute.arch.sync_threads()
        rows_per_stage_across_spans = self.stage_rows * self.sequence_splits
        num_stages = (L + rows_per_stage_across_spans - 1) // rows_per_stage_across_spans
        row_base = Int32(0)
        if cutlass.const_expr(self.sequence_splits > 1):
            row_base = by * num_stages * self.stage_rows
        tile_chan0 = bx * TMA_CHANNELS
        if warp == self.producer_warp:
            if tidx == 0:
                g_desc, x_desc = tma_desc_addr(atom_g), tma_desc_addr(atom_x)
                if cutlass.const_expr(self.batched):
                    if batch == 1:
                        cpasync.prefetch_descriptor(atom_g1)
                        cpasync.prefetch_descriptor(atom_x1)
                        g_desc, x_desc = tma_desc_addr(atom_g1), tma_desc_addr(atom_x1)
                    else:
                        cpasync.prefetch_descriptor(atom_g)
                        cpasync.prefetch_descriptor(atom_x)
                else:
                    cpasync.prefetch_descriptor(atom_g)
                    cpasync.prefetch_descriptor(atom_x)
                self.load_loop(g_desc, x_desc, slots, full0, empty0, num_stages, tile_chan0, row_base, offset)
        else:
            dx_base = mDx.iterator.toint()
            dx_desc = Int64(0)
            span = by
            if cutlass.const_expr(self.batched):
                dx_desc = tma_desc_addr(atom_dx)
                span += batch * self.sequence_splits
            self.compute_loop(mW, dx_base, mDw, slots, full0, empty0, s_dw, num_stages,
                              tile_chan0, row_base, span, L, tidx - 32, s_dx, dx_desc, batch * L)

    @cute.jit
    def load_loop(self, g_desc: Int64, x_desc: Int64, slots: Int32, full0: Int32, empty0: Int32, num_stages: Int32, tile_chan0: Int32, row_base: Int32, offset: Int32) -> None:
        for stage_idx in cutlass.range(num_stages, unroll=1):
            slot_idx, parity = self.ring.slot(stage_idx)
            mbar_wait(empty0 + 8 * slot_idx, parity ^ 1)
            full = full0 + 8 * slot_idx
            expect_slot_bytes(full, 2 * self.box_bytes)
            slot_addr = slots + slot_idx * self.ring.slot_bytes
            row0 = row_base + stage_idx * self.stage_rows
            tma_load_rows(slot_addr, g_desc, tile_chan0, row0 - offset, full)
            tma_load_rows(slot_addr + self.box_stride, x_desc, tile_chan0, row0 - (self.K - 1), full)

    @cute.jit
    def compute_loop(self, mW: cute.Tensor, dx_base: Int64, mDw: cute.Tensor, slots: Int32, full0: Int32, empty0: Int32, s_dw: Int32, num_stages: Int32, tile_chan0: Int32, row_base: Int32, span: Int32, L: Int32, consumer_idx: Int32, s_dx: Int32,
                     dx_desc: Int64, batch_row: Int32) -> None:
        group_idx, block_idx = consumer_idx % self.channel_groups, consumer_idx // self.channel_groups
        chan0 = tile_chan0 + group_idx * self.thread_channels
        weights = load_weights(mW.iterator.toint(), chan0, self.C, self.K, self.is_w_fp32, self.thread_channels)
        acc_dw = cute.make_rmem_tensor((self.thread_channels * self.K,), Float64)
        for i in cutlass.range_constexpr(self.thread_channels * self.K):
            acc_dw[i] = Float64(0.0)
        max_operand = Float32(0.0)
        local0 = block_idx * self.tokens
        for stage_idx in cutlass.range(num_stages, unroll=1):
            slot_idx, parity = self.ring.slot(stage_idx)
            mbar_wait(full0 + 8 * slot_idx, parity)
            g_slot = slots + slot_idx * self.ring.slot_bytes
            stage_row = row_base + stage_idx * self.stage_rows
            dx_slot = Int32(0)
            if cutlass.const_expr(self.batched):
                dx_slot = s_dx + (stage_idx % self.dx_slots) * self.dx_stage_bytes
            max_operand = self.compute_stage(
                g_slot, g_slot + self.box_stride, empty0 + 8 * slot_idx, local0, group_idx,
                weights, acc_dw, max_operand, dx_base, stage_row + local0, L, chan0, dx_slot,
            )
            if cutlass.const_expr(self.batched):
                fence_dx_stage()
                cute.arch.barrier(barrier_id=2, number_of_threads=TMA_CONSUMERS)
                if consumer_idx == 0:
                    # The next stage's barrier waits for this thread. A slot is
                    # reused only after the previous store's shared read retires.
                    tma_store_rows(dx_desc, tile_chan0, batch_row + stage_row, dx_slot)
        if cutlass.const_expr(self.batched):
            if consumer_idx == 0:
                finish_dx_stores()
        self.reduce_dw(acc_dw, max_operand, consumer_idx, s_dw, mDw.iterator.toint(), tile_chan0, span)

    def load_box_row(self, slot_addr: Int32, row_idx: Int32, group_idx: Int32) -> list:
        """
        Load the thread's consecutive BF16 channels from the box as FP32.
        """
        addr = slot_addr + (row_idx * TMA_CHANNELS + group_idx * self.thread_channels) * 2
        if cutlass.const_expr(self.thread_channels == 2):
            words = (lds_b32(addr),)
        else:
            bits = lds_f64(addr).bitcast(Int64)
            words = (bits.to(Uint32), (bits >> 32).to(Uint32))
        return [element(words, chan) for chan in range(self.thread_channels)]

    def compute_stage(self, g_slot: Int32, x_slot: Int32, empty: Int32, local0: Int32, group_idx: Int32, weights: list, acc_dw, max_operand: Float32, dx_base: Int64, row0: Int32, L: Int32, chan0: Int32, dx_slot: Int32) -> Float32:
        """
        The thread's rows row0 .. row0 + tokens - 1 of a stage: dx and the dw sums over a
        sliding window of K box rows of g and x; the slot is released after its last load.
        """
        K = self.K
        g_rows = [self.load_box_row(g_slot, local0 + i, group_idx) for i in range(K - 1)]
        x_rows = [self.load_box_row(x_slot, local0 + i, group_idx) for i in range(K - 1)]
        if cutlass.const_expr(self.certify):
            for row in x_rows:
                for elem in row:
                    max_operand = max_abs_operand(max_operand, elem)
        for j in range(self.tokens):
            g_rows.append(self.load_box_row(g_slot, local0 + j + K - 1, group_idx))
            x_rows.append(self.load_box_row(x_slot, local0 + j + K - 1, group_idx))
            if j == self.tokens - 1:
                release_slot(empty)
            words = []
            for pair in range(self.thread_channels // 2):
                accs = []
                for chan in (2 * pair, 2 * pair + 1):
                    acc = Float32(0.0)
                    for k in range(K):
                        acc = fma_f32(weights[chan][k], g_rows[j + K - 1 - k][chan], acc)
                    accs.append(acc)
                words.append(pack_bf16(accs[0], accs[1]))
            row_idx = row0 + j
            addr = dx_base + (Int64(row_idx) * self.C + chan0) * 2
            if cutlass.const_expr(self.batched):
                smem_elem = (local0 + j) * TMA_CHANNELS + group_idx * self.thread_channels
                sts_b32(dx_slot + smem_elem * 2, words[0], Int32(1))
            else:
                st_v2(addr, words[0], words[1], Int32(row_idx < L))
            for chan in range(self.thread_channels):
                if cutlass.const_expr(self.certify):
                    max_operand = max_abs_operand(max_operand, g_rows[j][chan])
                    max_operand = max_abs_operand(max_operand, x_rows[j + K - 1][chan])
                grad = g_rows[j][chan].to(Float64)
                for k in range(K):
                    acc_dw[chan * K + k] = fma_f64(grad, x_rows[j + k][chan].to(Float64), acc_dw[chan * K + k])
        return max_operand

    @cute.jit
    def reduce_dw(self, acc_dw, max_operand: Float32, consumer_idx: Int32, s_dw: Int32, dw_base: Int64, tile_chan0: Int32, span: Int32) -> None:
        """
        Steps (2) to (4) of the dw order: reduce same-channel lanes, then the first channel groups store
        s_dw[w][channel][k], then after a barrier of the consumers thread e < 64 * K writes
        dw[tile_chan0 * K + e] = sum over w in order of s_dw[w][e], rounded once.
        """
        K = self.K
        lane, consumer_warp = consumer_idx % 32, consumer_idx // 32
        if cutlass.const_expr(self.channel_groups < 32):
            for i in cutlass.range_constexpr(self.thread_channels * K):
                value = acc_dw[i]
                acc_dw[i] = value + cute.arch.shuffle_sync_bfly(value, offset=self.channel_groups)
        for i in cutlass.range_constexpr(self.thread_channels * K):
            entry = (consumer_warp * TMA_CHANNELS + lane * self.thread_channels) * K + i
            sts_f64(s_dw + entry * 8, acc_dw[i], Int32(lane < self.channel_groups))
        if cutlass.const_expr(self.certify):
            warp_max = warp_max_magnitude(f32_as_u32(max_operand))
            sts_b32(s_dw + self.dw_bytes + consumer_warp * 4, warp_max, Int32(lane == 0))
        cute.arch.barrier(barrier_id=1, number_of_threads=TMA_CONSUMERS)
        if cutlass.const_expr(self.certify):
            if consumer_idx == 0:
                tile_max = Float32(0.0)
                for warp_idx in cutlass.range_constexpr(TMA_CONSUMERS // 32):
                    warp_magnitude = u32_as_f32(lds_b32(s_dw + self.dw_bytes + warp_idx * 4))
                    tile_max = max_abs_operand(tile_max, warp_magnitude)
                tiles = self.C // TMA_CHANNELS
                max_index = self.sequences * self.sequence_splits * self.C * K + span * tiles + tile_chan0 // TMA_CHANNELS
                st_b64(dw_base + Int64(max_index) * 8, tile_max.to(Float64).bitcast(Int64), Int32(1))
        for entry0 in cutlass.range_constexpr(0, TMA_CHANNELS * K, TMA_CONSUMERS):
            entry = entry0 + consumer_idx
            entry_ok = Int32(entry < TMA_CHANNELS * K)
            acc = Float64(0.0)
            for warp_idx in cutlass.range_constexpr(TMA_CONSUMERS // 32):
                acc = acc + lds_f64(s_dw + (warp_idx * TMA_CHANNELS * K + entry * entry_ok) * 8)
            elem_idx = Int64(tile_chan0) * K + entry
            if cutlass.const_expr(self.sequence_splits > 1 or self.certify):
                elem_idx += Int64(span) * self.C * K
                st_b64(dw_base + elem_idx * 8, acc.bitcast(Int64), entry_ok)
            else:
                store_dw_value(dw_base, elem_idx, acc, entry_ok, self.dw_dtype)


class ConvFwdTma:
    """
    Forward on the TMA ring for one sequence (S % 8 == 0). Each CTA owns a 64-channel
    tile and a span of output rows. Split dispatch requires equal spans of whole stages.
    Stage s covers outputs [s * stage_rows, + stage_rows) and loads x rows
    [s * stage_rows + offset - (K - 1), + stage_rows + K - 1). Two consumer groups take alternate
    stages. Consumer thread: channel group (warp % 2) * 4 + lane % 4, token block
    (warp // 2) * 8 + lane // 4 of its group (stage_rows / 16 = 8 outputs), so one 16-byte store of
    a warp writes 4 channels x 128 contiguous bytes of the channels-first y.
    """

    def __init__(self, C: int, K: int, is_w_fp32: bool, sequence_splits: int) -> None:
        self.C, self.K, self.is_w_fp32 = C, K, is_w_fp32
        self.sequence_splits = sequence_splits
        self.stage_rows, self.tokens = FWD_TMA_ROWS, FWD_TMA_ROWS // 16
        self.box_rows = self.stage_rows + K - 1
        self.box_bytes = self.box_rows * TMA_CHANNELS * 2
        self.ring = TmaRing(FWD_TMA_STAGES, (self.box_bytes + 127) // 128 * 128, FWD_TMA_GROUPS)
        self.producer_warp = 0
        assert self.tokens == 8 and self.ring.num_slots * self.ring.slot_bytes + 1024 <= SMEM_LIMIT

    @cute.jit
    def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, S: Int32, offset: Int32, stream) -> None:
        box = cute.make_layout((self.box_rows, TMA_CHANNELS), stride=(TMA_CHANNELS, 1))
        atom_x, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mX, box, (self.box_rows, TMA_CHANNELS))
        self.kernel(atom_x, mW, mY, S, offset).launch(grid=[self.C // TMA_CHANNELS, self.sequence_splits, 1], block=[32 + TMA_CONSUMERS * FWD_TMA_GROUPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, atom_x: cute.CopyAtom, mW: cute.Tensor, mY: cute.Tensor, S: Int32, offset: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        smem = cutlass.memory.SmemAllocator()
        slots, full0, empty0 = self.ring.allocate(smem)
        if tidx == 0:
            init_ring(full0, empty0, self.ring.num_slots)
        cute.arch.sync_threads()
        rows_per_stage_across_spans = self.stage_rows * self.sequence_splits
        num_stages = (S + rows_per_stage_across_spans - 1) // rows_per_stage_across_spans
        row_base = Int32(0)
        if cutlass.const_expr(self.sequence_splits > 1):
            row_base = by * num_stages * self.stage_rows
        tile_chan0 = bx * TMA_CHANNELS
        if warp == self.producer_warp:
            if tidx == 0:
                self.load_loop(atom_x, slots, full0, empty0, num_stages, tile_chan0, offset + row_base)
        else:
            self.compute_loop(mW, mY, slots, full0, empty0, num_stages, tile_chan0, S, row_base, tidx - 32)

    @cute.jit
    def load_loop(self, atom_x: cute.CopyAtom, slots: Int32, full0: Int32, empty0: Int32, num_stages: Int32, tile_chan0: Int32, offset: Int32) -> None:
        cpasync.prefetch_descriptor(atom_x)
        x_desc = tma_desc_addr(atom_x)
        for stage_idx in cutlass.range(num_stages, unroll=1):
            slot_idx, parity = self.ring.slot(stage_idx)
            mbar_wait(empty0 + 8 * slot_idx, parity ^ 1)
            full = full0 + 8 * slot_idx
            expect_slot_bytes(full, self.box_bytes)
            tma_load_rows(slots + slot_idx * self.ring.slot_bytes, x_desc, tile_chan0, stage_idx * self.stage_rows + offset - (self.K - 1), full)

    @cute.jit
    def compute_loop(self, mW: cute.Tensor, mY: cute.Tensor, slots: Int32, full0: Int32, empty0: Int32, num_stages: Int32, tile_chan0: Int32, S: Int32, row_base: Int32, consumer_idx: Int32) -> None:
        group, group_thread = consumer_idx // TMA_CONSUMERS, consumer_idx % TMA_CONSUMERS
        lane, group_warp = group_thread % 32, group_thread // 32
        group_idx, block_idx = (group_warp % 2) * 4 + lane % 4, (group_warp // 2) * 8 + lane // 4
        chan0 = tile_chan0 + group_idx * 8
        weights = load_weights(mW.iterator.toint(), chan0, self.C, self.K, self.is_w_fp32)
        local0 = block_idx * self.tokens
        for stage_idx in cutlass.range(group, num_stages, FWD_TMA_GROUPS, unroll=1):
            slot_idx, parity = self.ring.slot(stage_idx)
            mbar_wait(full0 + 8 * slot_idx, parity)
            slot_addr = slots + slot_idx * self.ring.slot_bytes
            rows = []
            for i in cutlass.range_constexpr(self.tokens + self.K - 1):
                rows.append(lds_v4(slot_addr + ((local0 + i) * TMA_CHANNELS + group_idx * 8) * 2))
            release_slot(empty0 + 8 * slot_idx)
            self.store_y(mY.iterator.toint(), rows, weights, row_base + stage_idx * self.stage_rows + local0, S, chan0)

    def store_y(self, y_base: Int64, rows: list, weights: list, out_start: Int32, S: Int32, chan0: Int32) -> None:
        """
        y[chan, out_start + j] = sum over k in order of w[chan][k] * rows[j + k][chan], with one
        store per 8 outputs.
        """
        for chan in range(8):
            outputs = []
            for j in range(self.tokens):
                acc = Float32(0.0)
                for k in range(self.K):
                    acc = fma_f32(weights[chan][k], element(rows[j + k], chan), acc)
                outputs.append(acc)
            words = [pack_bf16(outputs[2 * i], outputs[2 * i + 1]) for i in range(self.tokens // 2)]
            st_v4(y_base + (Int64(chan0 + chan) * S + out_start) * 2, tuple(words), Int32(out_start < S))


COMPILED: dict = {}
COUNTERS: dict = {}
DW_TYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: Float32,
    torch.float64: Float64,
}


@cute.kernel
def merge_dw_spans_kernel(partials: cute.Tensor, dw: cute.Tensor, entries: Int32, spans: cutlass.Constexpr[int]) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    entry = bx * 128 + tidx
    if entry < entries:
        acc = Float64(0.0)
        for span in cutlass.range_constexpr(spans):
            acc = acc + partials[span * entries + entry]
        store_dw_value(dw.iterator.toint(), Int64(entry), acc, Int32(1), dw.element_type)


@cute.jit
def merge_dw_spans(partials: cute.Tensor, dw: cute.Tensor, entries: Int32, stream, spans: cutlass.Constexpr[int]) -> None:
    merge_dw_spans_kernel(partials, dw, entries, spans).launch(
        grid=[(entries + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream,
    )


# Finite BF16 products are signed integers times 2**-266. An individual
# product needs 522 bits, and 37 base-2**16 digits also cover the sum over
# every row admitted by the existing batch and sequence dimensions.
DW_EXACT_DIGITS = 37
DW_EXACT_THREADS = 128
# The comparator scales the 2**-266 product bins by another 2**23.
dw_bound_slack_numerator, dw_bound_slack_denominator = (1e-30).as_integer_ratio()
dw_bound_scaled_slack = (dw_bound_slack_numerator << 289) // dw_bound_slack_denominator
DW_BOUND_SLACK_DIGITS = tuple(
    (i, (dw_bound_scaled_slack >> (16 * i)) & 65535)
    for i in range(DW_EXACT_DIGITS)
    if (dw_bound_scaled_slack >> (16 * i)) & 65535
)


@dsl_user_op
def add_dw_digit(addr: Int32, value: Int64, *, loc=None, ip=None) -> None:
    asm(
        None, [to_ir(addr, Int32, loc, ip), to_ir(value, Int64, loc, ip)],
        "{ .reg .b64 prior; atom.shared.add.u64 prior, [$0], $1; }", "r,l", side_effects=True,
    )


@dsl_user_op
def highest_set_bit(value: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(asm(
        T.i32(), [to_ir(value, Uint32, loc, ip)],
        "{ .reg .b32 zeros; clz.b32 zeros, $1; sub.s32 $0, 31, zeros; }", "=r,r",
    ))


@cute.jit
def normalize_dw_digits(digits: Int32) -> tuple[Int32, Int32]:
    """Carry the signed bins, then leave the magnitude in base 2**16."""
    carry = Int64(0)
    for i in cutlass.range(DW_EXACT_DIGITS, unroll=1):
        addr = digits + i * 8
        total = lds_f64(addr).bitcast(Int64) + carry
        sts_f64(addr, (total & Int64(65535)).bitcast(Float64), Int32(1))
        carry = total >> 16
    negative = Int32(carry < 0)
    inverse = Int64(negative) * 65535
    carry = Int64(negative)
    highest = Int32(-1)
    for i in cutlass.range(DW_EXACT_DIGITS, unroll=1):
        addr = digits + i * 8
        total = (lds_f64(addr).bitcast(Int64) ^ inverse) + carry
        digit = total & Int64(65535)
        sts_f64(addr, digit.bitcast(Float64), Int32(1))
        carry = total >> 16
        if digit != 0:
            highest = i
    top = Int32(-1)
    if highest >= 0:
        top = highest * 16 + highest_set_bit(lds_f64(digits + highest * 8).bitcast(Int64).to(Uint32))
    return negative, top


@cute.jit
def dw_rounding_bits(digits: Int32, shift: Int32, fraction_bits: cutlass.Constexpr[int]) -> tuple[Int64, Int64, Int64]:
    """Read the fraction, guard bit, and whether any lower bits are set."""
    base, intra = (shift - 1) // 16, (shift - 1) % 16
    low = lds_f64(digits + base * 8).bitcast(Int64)
    window = low >> intra
    sticky = Int64((low & ((Int64(1) << intra) - 1)) != 0)
    for i in cutlass.range_constexpr(1, (fraction_bits + 16) // 16 + 1):
        left = i * 16 - intra
        if left < fraction_bits + 1:
            window = window | (lds_f64(digits + (base + i) * 8).bitcast(Int64) << left)
    for i in cutlass.range(base, unroll=1):
        sticky = sticky | Int64(lds_f64(digits + i * 8).bitcast(Int64) != 0)
    return (window >> 1) & Int64((1 << fraction_bits) - 1), window & Int64(1), sticky


@cute.jit
def exact_dw_to_float(digits: Int32, negative: Int32, top: Int32,
                      fraction_bits: cutlass.Constexpr[int], round_to_odd: cutlass.Constexpr[bool]) -> Float32:
    """Round the normalized sum at FP32 or BF16 precision and encode it in FP32."""
    magnitude = Uint32(0)
    if top >= 0:
        exponent = top - 266
        if exponent > 127:
            magnitude = Uint32(255 << fraction_bits)
        elif exponent < -127 - fraction_bits:
            if cutlass.const_expr(round_to_odd):
                magnitude = Uint32(1)
        else:
            shift = top - fraction_bits
            exponent_bits = exponent + 127
            if exponent_bits <= 0:
                shift = Int32(140 - fraction_bits)
                exponent_bits = Int32(0)
            fraction, guard, sticky = dw_rounding_bits(digits, shift, fraction_bits)
            magnitude = (Uint32(exponent_bits) << fraction_bits) | Uint32(fraction)
            if cutlass.const_expr(round_to_odd):
                magnitude = magnitude | Uint32(guard | sticky)
            else:
                magnitude = magnitude + Uint32(guard & (sticky | (fraction & Int64(1))))
    return u32_as_f32((Uint32(negative) << 31) | (magnitude << (23 - fraction_bits)))


@cute.jit
def initialize_dw_bound(digits: Int32, endpoint: Float32, epsilon: Float32) -> None:
    """Start the exact scaled difference (1 - epsilon) * abs(sum) - endpoint - slack."""
    scale = Int64(2**23) - (epsilon * Float32(2**23)).to(Int64)
    for i in cutlass.range(DW_EXACT_DIGITS, unroll=1):
        addr = digits + i * 8
        sts_f64(addr, (lds_f64(addr).bitcast(Int64) * scale).bitcast(Float64), Int32(1))
    bits = f32_as_u32(endpoint) & Uint32(0x7fffffff)
    exponent = Int32(bits >> 23)
    mantissa = Int64(bits & Uint32(0x7fffff)) + Int64(exponent != 0) * 2**23
    # The FP32 mantissa is scaled to integer units of 2**-289.
    shift = exponent + Int32(exponent == 0) + 139
    start = shift // 16
    shifted = mantissa << (shift % 16)
    for i in cutlass.range_constexpr(3):
        add_dw_digit(digits + (start + i) * 8, -((shifted >> (16 * i)) & Int64(65535)))
    for i in cutlass.range_constexpr(len(DW_BOUND_SLACK_DIGITS)):
        index, digit = DW_BOUND_SLACK_DIGITS[i]
        add_dw_digit(digits + index * 8, Int64(-digit))


@cute.jit
def dw_bound_difference_positive(digits: Int32) -> Int32:
    carry = Int64(0)
    nonzero = Int32(0)
    for i in cutlass.range(DW_EXACT_DIGITS, unroll=1):
        total = lds_f64(digits + i * 8).bitcast(Int64) + carry
        nonzero = nonzero | Int32((total & Int64(65535)) != 0)
        carry = total >> 16
    return Int32(carry > 0) | (Int32(carry == 0) & nonzero)


@cute.jit
def accumulate_exact_dw(g: cute.Tensor, x: cute.Tensor, entry: Int32, tidx: Int32,
                        digits: Int32, nonfinite: Int32, B: Int32, L: Int32, S: Int32,
                        C: Int32, K: Int32, offset: Int32, g_batch_stride: Int64,
                        g_row_stride: Int64, g_channel_stride: Int64,
                        subtract_bound: cutlass.Constexpr[bool]) -> None:
    channel, tap = entry // K, entry % K
    if cutlass.const_expr(not subtract_bound):
        if tidx < DW_EXACT_DIGITS:
            sts_f64(digits + tidx * 8, Float64(0.0), Int32(1))
        if tidx == 0:
            sts_b32(nonfinite, Uint32(0), Int32(1))
    cute.arch.sync_threads()

    for index in cutlass.range(Int64(tidx), Int64(B) * S, DW_EXACT_THREADS, unroll=1):
        batch, token = index // S, index % S
        input_row = token + offset - (K - 1) + tap
        grad_index = batch * g_batch_stride + token * g_row_stride + channel * g_channel_stride
        input_index = (batch * L + input_row) * C + channel
        g_bits = ld_b16(g.iterator.toint() + grad_index * 2, Int32(1))
        x_bits = ld_b16(x.iterator.toint() + input_index * 2, Int32(input_row >= 0) & Int32(input_row < L))
        gx = Int32((g_bits >> 7) & 255)
        xx = Int32((x_bits >> 7) & 255)
        if gx == 255 or xx == 255:
            red_or_shared(nonfinite, Uint32(1), Int32(1))
        else:
            g_mantissa = Int64(g_bits & 127) + Int64(Int32(gx != 0)) * 128
            x_mantissa = Int64(x_bits & 127) + Int64(Int32(xx != 0)) * 128
            product = g_mantissa * x_mantissa
            if product != 0:
                # Subnormals share the scale of exponent field 1 but have no hidden bit.
                shift = gx + xx - Int32(gx != 0) - Int32(xx != 0)
                shifted = product << (shift % 16)
                sign = Int64(-4)
                if cutlass.const_expr(not subtract_bound):
                    sign = Int64(1) - Int64(((g_bits ^ x_bits) >> 15) & 1) * 2
                addr = digits + (shift // 16) * 8
                add_dw_digit(addr, sign * (shifted & 65535))
                add_dw_digit(addr + 8, sign * (shifted >> 16))
    cute.arch.sync_threads()


@cute.jit
def finish_finite_dw(g: cute.Tensor, x: cute.Tensor, entry: Int32, tidx: Int32,
                     digits: Int32, nonfinite: Int32, overflow: Int32, B: Int32, L: Int32, S: Int32,
                     C: Int32, K: Int32, offset: Int32, g_batch_stride: Int64,
                     g_row_stride: Int64, g_channel_stride: Int64,
                     lowest: Float32, highest: Float32, epsilon: Float32,
                     fraction_bits: cutlass.Constexpr[int], round_to_odd: cutlass.Constexpr[bool]) -> tuple[Float32, Int32]:
    value = Float32(0.0)
    if tidx == 0:
        negative, top = normalize_dw_digits(digits)
        value = exact_dw_to_float(digits, negative, top, fraction_bits, round_to_odd)
        clamped = Int32(value > highest) | (Int32(lowest < Float32(0.0)) & Int32(value < lowest))
        if value < lowest:
            value = lowest
        elif value > highest:
            value = highest
        if clamped != 0:
            initialize_dw_bound(digits, value, epsilon)
        sts_b32(overflow, Uint32(clamped), Int32(1))
    cute.arch.sync_threads()

    replay = Int32(0)
    if lds_b32(overflow) != 0:
        accumulate_exact_dw(g, x, entry, tidx, digits, nonfinite, B, L, S, C, K, offset,
                            g_batch_stride, g_row_stride, g_channel_stride, True)
        if tidx == 0:
            sts_b32(overflow, Uint32(dw_bound_difference_positive(digits)), Int32(1))
        cute.arch.sync_threads()
        replay = Int32(lds_b32(overflow))
    return value, replay


@cute.jit
def replay_pinned_dw(g: cute.Tensor, x: cute.Tensor, entry: Int32, lane: Int32,
                     B: Int32, L: Int32, S: Int32, C: Int32, K: Int32, offset: Int32,
                     g_batch_stride: Int64, g_row_stride: Int64, g_channel_stride: Int64,
                     num_splits: Int32, num_iters: Int32, pinned_tma: cutlass.Constexpr[bool]) -> Float32:
    """One warp replays the pinned FP32 FMA and reduction order for an entry."""
    tokens = BWD_TMA_ROWS // 16 if pinned_tma else BWD_TOKENS
    blocks_per_warp = 4 if pinned_tma else BLOCKS_PER_WARP
    rows_per_iter = BWD_TMA_ROWS if pinned_tma else BWD_WARPS * BLOCKS_PER_WARP * BWD_TOKENS
    channel, tap = entry // K, entry % K
    total = Float32(0.0)
    for partial in cutlass.range(B * num_splits, unroll=1):
        batch, split = partial // num_splits, partial % num_splits
        x_base = x.iterator.toint() + (Int64(batch) * L * C + channel) * 2
        g_base = g.iterator.toint() + (Int64(batch) * g_batch_stride + channel * g_channel_stride) * 2
        acc = Float32(0.0)
        if lane * tokens < rows_per_iter:
            for i in cutlass.range(num_iters, unroll=1):
                row0 = (split * num_iters + i) * rows_per_iter + lane * tokens
                for j in cutlass.range_constexpr(tokens):
                    row = row0 + j
                    token, x_row = row - offset, row - (K - 1) + tap
                    grad = ld_b16(g_base + Int64(token) * g_row_stride * 2,
                                  Int32(token >= 0) & Int32(token < S))
                    inp = ld_b16(x_base + Int64(x_row) * C * 2, Int32(x_row >= 0) & Int32(x_row < L))
                    acc = fma_f32(bf16_lo(grad), bf16_lo(inp), acc)
        acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=1)
        acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=2)
        if cutlass.const_expr(not pinned_tma):
            acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=4)
        cta_sum = Float32(0.0)
        for warp in cutlass.range_constexpr(BWD_WARPS):
            cta_sum = cta_sum + cute.arch.shuffle_sync(acc, offset=warp * blocks_per_warp)
        total = total + cta_sum
    return total


@cute.jit
def exact_dw_to_fp64(digits: Int32, negative: Int32, top: Int32) -> Float64:
    magnitude = Float64(0.0)
    if top >= 53:
        fraction, guard, sticky = dw_rounding_bits(digits, top - 52, 52)
        bits = (Int64(top - 266 + 1023) << 52) | fraction
        magnitude = (bits + (guard & (sticky | (fraction & Int64(1))))).bitcast(Float64)
    elif top >= 0:
        raw = Int64(0)
        for i in cutlass.range_constexpr(4):
            raw = raw | (lds_f64(digits + 8 * i).bitcast(Int64) << (16 * i))
        magnitude = raw.to(Float64) * Float64(2.0**-266)
    return -magnitude if negative != 0 else magnitude


@cute.kernel
def exact_narrow_dw_kernel(g: cute.Tensor, x: cute.Tensor, approx: cute.Tensor, out: cute.Tensor,
                           B: Int32, L: Int32, S: Int32, C: Int32, K: Int32, offset: Int32,
                           g_batch_stride: Int64, g_row_stride: Int64, g_channel_stride: Int64,
                           num_splits: Int32, num_iters: Int32, lowest: Float32, highest: Float32, epsilon: Float32,
                           pinned_tma: cutlass.Constexpr[bool]) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    entry, _, _ = cute.arch.block_idx()
    smem = cutlass.memory.SmemAllocator()
    digits = Int32(smem.allocate(DW_EXACT_DIGITS * 8, byte_alignment=8).toint())
    nonfinite = Int32(smem.allocate(4, byte_alignment=4).toint())
    overflow = Int32(smem.allocate(8, byte_alignment=8).toint())
    accumulate_exact_dw(g, x, entry, tidx, digits, nonfinite, B, L, S, C, K, offset,
                        g_batch_stride, g_row_stride, g_channel_stride, False)
    prior = approx[entry]
    replay = Int32(lds_b32(nonfinite) != 0) | Int32(prior != prior)
    if cutlass.const_expr(out.element_type is Float64):
        if replay == 0:
            if tidx == 0:
                negative, top = normalize_dw_digits(digits)
                out[entry] = exact_dw_to_fp64(digits, negative, top)
        elif tidx < 32:
            value = replay_pinned_dw(g, x, entry, tidx, B, L, S, C, K, offset,
                                     g_batch_stride, g_row_stride, g_channel_stride, num_splits, num_iters, pinned_tma)
            if tidx == 0:
                out[entry] = value.to(Float64)
    else:
        value = Float32(0.0)
        if replay == 0:
            value, replay = finish_finite_dw(g, x, entry, tidx, digits, nonfinite, overflow, B, L, S, C, K, offset,
                                             g_batch_stride, g_row_stride, g_channel_stride, lowest, highest, epsilon, 23, True)
        if replay != 0 and tidx < 32:
            value = replay_pinned_dw(g, x, entry, tidx, B, L, S, C, K, offset,
                                     g_batch_stride, g_row_stride, g_channel_stride, num_splits, num_iters, pinned_tma)
        if tidx == 0:
            out[entry] = value


@cute.jit
def exact_narrow_dw(g: cute.Tensor, x: cute.Tensor, approx: cute.Tensor, out: cute.Tensor,
                    B: Int32, L: Int32, S: Int32, C: Int32, K: Int32, offset: Int32,
                    g_batch_stride: Int64, g_row_stride: Int64, g_channel_stride: Int64,
                    num_splits: Int32, num_iters: Int32, lowest: Float32, highest: Float32, epsilon: Float32, stream,
                    pinned_tma: cutlass.Constexpr[bool]) -> None:
    exact_narrow_dw_kernel(g, x, approx, out, B, L, S, C, K, offset,
                           g_batch_stride, g_row_stride, g_channel_stride,
                           num_splits, num_iters, lowest, highest, epsilon, pinned_tma).launch(
        grid=[C * K, 1, 1], block=[DW_EXACT_THREADS, 1, 1], stream=stream,
    )


@dsl_user_op
def reserve_dw_repair(count: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(asm(T.i32(), [to_ir(count, Int32, loc, ip)],
                     "atom.shared.add.u32 $0, [$1], 1;", "=r,r", side_effects=True))


@cute.kernel
def repair_wide_dw_kernel(g: cute.Tensor, x: cute.Tensor, dw: cute.Tensor, partials: cute.Tensor,
                          B: Int32, L: Int32, S: Int32, C: Int32, K: Int32, offset: Int32,
                          g_batch_stride: Int64, g_row_stride: Int64, g_channel_stride: Int64,
                          num_splits: Int32, num_iters: Int32, pinned_tma: cutlass.Constexpr[bool],
                          spans: cutlass.Constexpr[int], certificate_tile: cutlass.Constexpr[int],
                          generic_certificate: cutlass.Constexpr[bool]) -> None:
    """Optionally merge and certify FP64 spans before repairing exceptional entries."""
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    entry = bx * DW_EXACT_THREADS + tidx
    smem = cutlass.memory.SmemAllocator()
    count = Int32(smem.allocate(4, byte_alignment=4).toint())
    entries = Int32(smem.allocate(DW_EXACT_THREADS * 4, byte_alignment=4).toint())
    digits = Int32(smem.allocate(DW_EXACT_DIGITS * 8, byte_alignment=8).toint())
    nonfinite = Int32(smem.allocate(4, byte_alignment=4).toint())
    overflow = Int32(smem.allocate(8, byte_alignment=8).toint())
    is_bf16 = dw.element_type is cutlass.BFloat16
    is_fp64 = dw.element_type is Float64
    limit = 0x7f7f0000 if is_bf16 else 0x7f7fffff
    fraction_bits = 7 if is_bf16 else 23

    total_spans = Int32(spans)
    if cutlass.const_expr(generic_certificate and not pinned_tma):
        total_spans = B * num_splits
    if tidx == 0:
        sts_b32(count, Uint32(0), Int32(1))
        if cutlass.const_expr(spans > 0 and not generic_certificate):
            tile = (bx * DW_EXACT_THREADS) // (certificate_tile * K)
            tiles = C // certificate_tile
            max_index = spans * C * K + tile
            tile_max = Float32(0.0)
            for span in cutlass.range_constexpr(spans):
                maximum = partials[max_index + span * tiles].to(Float32)
                tile_max = max_abs_operand(tile_max, maximum)
            magnitude = tile_max.to(Float64)
            # At the production shape, depth <1024 and at most 8192 products give an error
            # below 2**-30 * max_operand**2. The extra factor covers
            # rounding in the interval endpoints.
            sts_f64(overflow, Float64(2.0**-29) * magnitude * magnitude, Int32(1))
    cute.arch.sync_threads()
    if entry < C * K:
        if cutlass.const_expr(generic_certificate):
            acc = Float64(0.0)
            entry_max = Float32(0.0)
            tiles = (C + certificate_tile - 1) // certificate_tile
            max_index = Int64(total_spans) * C * K + entry // K // certificate_tile
            for span in cutlass.range(total_spans, unroll=1):
                acc = acc + partials[Int64(span) * C * K + entry]
                entry_max = max_abs_operand(entry_max, partials[max_index + Int64(span) * tiles].to(Float32))
            # The depth covers the per-thread FMAs, warp and CTA sums, and
            # ordered span merge. The factor covers roundoff at both endpoints.
            terms = (Int64(B) * S).to(Float64)
            depth = (Int64(num_iters) * BWD_TOKENS + total_spans + 8).to(Float64)
            magnitude = entry_max.to(Float64)
            radius = Float64(2.0**-51) * terms * depth * magnitude * magnitude
            if cutlass.const_expr(is_bf16):
                bits = round_dw_bf16(acc)
                st_b16(dw.iterator.toint() + Int64(entry) * 2, bits, Int32(1))
                needs_repair = (bits & Uint32(0x7fff)) >= Uint32(0x7f7f)
                needs_repair = needs_repair | (round_dw_bf16(acc - radius) != round_dw_bf16(acc + radius))
            else:
                rounded = acc.to(Float32)
                dw[entry] = rounded
                rounded_bits = f32_as_u32(rounded)
                lower = f32_as_u32((acc - radius).to(Float32))
                upper = f32_as_u32((acc + radius).to(Float32))
                needs_repair = ((rounded_bits & Uint32(0x7fffffff)) >= Uint32(limit)) | (lower != upper)
        elif cutlass.const_expr(spans > 0):
            acc = Float64(0.0)
            for span in cutlass.range_constexpr(spans):
                acc = acc + partials[span * C * K + entry]
            bits = round_dw_bf16(acc)
            st_b16(dw.iterator.toint() + Int64(entry) * 2, bits, Int32(1))
            radius = lds_f64(overflow)
            needs_repair = (bits & Uint32(0x7fff)) >= Uint32(0x7f7f)
            needs_repair = needs_repair | (round_dw_bf16(acc - radius) != round_dw_bf16(acc + radius))
        elif cutlass.const_expr(is_fp64):
            magnitude = dw[entry].bitcast(Int64) & Int64(0x7fffffffffffffff)
            needs_repair = magnitude >= Int64(0x7ff0000000000000)
        else:
            magnitude = f32_as_u32(dw[entry].to(Float32)) & Uint32(0x7fffffff)
            needs_repair = magnitude >= Uint32(limit)
        if needs_repair:
            slot = reserve_dw_repair(count)
            sts_b32(entries + slot * 4, Uint32(entry), Int32(1))
    cute.arch.sync_threads()

    for slot in cutlass.range(Int32(lds_b32(count)), unroll=1):
        repair_entry = Int32(lds_b32(entries + slot * 4))
        replay = Int32(1)
        value = Float32(0.0)
        if cutlass.const_expr(not is_fp64):
            prior = dw[repair_entry]
            accumulate_exact_dw(g, x, repair_entry, tidx, digits, nonfinite, B, L, S, C, K, offset,
                                g_batch_stride, g_row_stride, g_channel_stride, False)
            replay = Int32(lds_b32(nonfinite) != 0) | Int32(prior != prior)
            if replay == 0:
                highest = u32_as_f32(Uint32(limit))
                value, replay = finish_finite_dw(
                    g, x, repair_entry, tidx, digits, nonfinite, overflow, B, L, S, C, K, offset,
                    g_batch_stride, g_row_stride, g_channel_stride, -highest, highest,
                    Float32(2.0 ** -fraction_bits), fraction_bits, False,
                )
        if replay != 0 and tidx < 32:
            value = replay_pinned_dw(g, x, repair_entry, tidx, B, L, S, C, K, offset,
                                     g_batch_stride, g_row_stride, g_channel_stride, num_splits, num_iters, pinned_tma)
        if tidx == 0:
            dw[repair_entry] = value.to(dw.element_type)
        cute.arch.sync_threads()


@cute.jit
def repair_wide_dw(g: cute.Tensor, x: cute.Tensor, dw: cute.Tensor, partials: cute.Tensor,
                   B: Int32, L: Int32, S: Int32, C: Int32, K: Int32, offset: Int32,
                   g_batch_stride: Int64, g_row_stride: Int64, g_channel_stride: Int64,
                   num_splits: Int32, num_iters: Int32, stream, pinned_tma: cutlass.Constexpr[bool],
                   spans: cutlass.Constexpr[int], certificate_tile: cutlass.Constexpr[int],
                   generic_certificate: cutlass.Constexpr[bool]) -> None:
    repair_wide_dw_kernel(g, x, dw, partials, B, L, S, C, K, offset,
                          g_batch_stride, g_row_stride, g_channel_stride, num_splits, num_iters,
                          pinned_tma, spans, certificate_tile, generic_certificate).launch(
        grid=[(C * K + DW_EXACT_THREADS - 1) // DW_EXACT_THREADS, 1, 1],
        block=[DW_EXACT_THREADS, 1, 1], stream=stream,
    )


def flat(dtype: type) -> object:
    return fake(dtype, (cute.sym_int(),), (1,), dtype.width // 8)


def finish_exact_dw(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, approx: torch.Tensor, offset: int,
                     pinned_tma: bool, num_splits: int, num_iters: int) -> torch.Tensor:
    """
    Round FP64 outputs directly and use an FP32 carrier for narrower outputs.
    """
    B, L, C = x.shape
    K = w.shape[-1]
    fp64 = w.dtype == torch.float64
    carrier_type = Float64 if fp64 else Float32
    key = ("conv_bwd_exact_dw", pinned_tma, carrier_type)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = cute.compile(
            exact_narrow_dw, flat(cutlass.BFloat16), flat(cutlass.BFloat16), flat(Float64), flat(carrier_type),
            *(Int32(1) for _ in range(6)), Int64(1), Int64(1), Int64(1),
            Int32(1), Int32(1), Float32(1), Float32(1), Float32(1), fake_stream(), pinned_tma,
            options="--enable-tvm-ffi",
        )
    carrier = torch.empty(w.shape, dtype=torch.float64 if fp64 else torch.float32, device=w.device)
    info = torch.finfo(w.dtype)
    if fp64:
        limits = (0.0, 0.0, 0.0)
    else:
        lowest = info.tiny if w.dtype == getattr(torch, "float8_e8m0fnu", None) else info.min
        limits = (lowest, info.max, info.eps)
    fn(g.as_strided((1,), (1,)), x.view(-1), approx.view(-1), carrier.view(-1),
       B, L, L - offset, C, K, offset, *g.stride(), num_splits, num_iters, *limits)
    return carrier


def final_dw(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, dw: torch.Tensor, offset: int,
             exact_output: bool, pinned_tma: bool, num_splits: int, num_iters: int,
             partials: torch.Tensor | None = None, spans: int = 0,
             certificate_tile: int = TMA_CHANNELS, generic_certificate: bool = False) -> torch.Tensor:
    if exact_output or w.dtype == torch.float64:
        dw = finish_exact_dw(g, x, w, dw, offset, pinned_tma, num_splits, num_iters)
    elif w.dtype in DW_TYPES:
        B, L, C = x.shape
        K = w.shape[-1]
        key = ("conv_bwd_repair_wide_dw", w.dtype, pinned_tma, spans, certificate_tile, generic_certificate)
        fn = COMPILED.get(key)
        if fn is None:
            fn = COMPILED[key] = cute.compile(
                repair_wide_dw, flat(cutlass.BFloat16), flat(cutlass.BFloat16),
                flat(DW_TYPES[w.dtype]), flat(Float64 if spans or generic_certificate else DW_TYPES[w.dtype]),
                *(Int32(1) for _ in range(6)), Int64(1), Int64(1), Int64(1),
                Int32(1), Int32(1), fake_stream(), pinned_tma, spans, certificate_tile, generic_certificate,
                options="--enable-tvm-ffi",
            )
        partial_data = dw if partials is None else partials
        fn(g.as_strided((1,), (1,)), x.view(-1), dw.view(-1), partial_data.view(-1),
           B, L, L - offset, C, K, offset, *g.stride(), num_splits, num_iters)
    return dw.to(w.dtype).view(w.shape)


def compile_fwd(C: int, K: int, is_w_fp32: bool, is_row_vector: bool, is_col_vector: bool) -> Callable:
    w = flat(Float32 if is_w_fp32 else cutlass.BFloat16)
    return cute.compile(ConvFwd(C, K, is_w_fp32, is_row_vector, is_col_vector), flat(cutlass.BFloat16), w, flat(cutlass.BFloat16), Int32(1), Int32(1), Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def compile_bwd(C: int, K: int, is_w_fp32: bool, is_row_vector: bool, is_g_channels_first: bool, is_g_vector: bool, g_shift: int, dw_dtype: type, certify: bool) -> Callable:
    w = flat(Float32 if is_w_fp32 else cutlass.BFloat16)
    partials = fake(Float64, (cute.sym_int(),), (1,), 8)
    tensors = (flat(cutlass.BFloat16), flat(cutlass.BFloat16), w, flat(cutlass.BFloat16), flat(dw_dtype), partials, fake(Int64, (cute.sym_int(),), (1,), 8))
    scalars = (Int32(1), Int32(1), Int32(1), Int32(0), Int64(0), Int64(0), Int32(1), Int32(1))
    return cute.compile(ConvBwd(C, K, is_w_fp32, is_row_vector, is_g_channels_first, is_g_vector, g_shift, dw_dtype, certify), *tensors, *scalars, fake_stream(), options="--enable-tvm-ffi")


def rows_of(dtype: type, C: int) -> object:
    """
    A (rows, C) row-major tensor with a dynamic row count, as the TMA descriptors read it.
    """
    return fake(dtype, (cute.sym_int(), C), (C, 1))


def compile_fwd_tma(C: int, K: int, is_w_fp32: bool, sequence_splits: int) -> Callable:
    w = flat(Float32 if is_w_fp32 else cutlass.BFloat16)
    return cute.compile(ConvFwdTma(C, K, is_w_fp32, sequence_splits), rows_of(cutlass.BFloat16, C), w, flat(cutlass.BFloat16), Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def compile_bwd_tma(C: int, K: int, is_w_fp32: bool, sequence_splits: int, dw_dtype: type, batched: bool, certify_single: bool) -> Callable:
    w = flat(Float32 if is_w_fp32 else cutlass.BFloat16)
    dw = flat(Float64 if sequence_splits > 1 or certify_single else dw_dtype)
    conv = ConvBwdTma(C, K, is_w_fp32, sequence_splits, dw_dtype, batched, certify_single)
    entry = conv.batch_two if batched else conv
    inputs = [rows_of(cutlass.BFloat16, C) for _ in range(4 if batched else 2)]
    dx = rows_of(cutlass.BFloat16, C) if batched else flat(cutlass.BFloat16)
    return cute.compile(entry, *inputs, w, dx, dw, Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def is_tma_input(x: torch.Tensor, K: int) -> bool:
    """
    x and the weight width the TMA kernels take: one sequence of bf16 x, contiguous and 16-byte
    aligned (the TMA descriptor's base), C a multiple of the 64-channel tile, 2 <= K <= 4.
    """
    B, _, C = x.shape
    return B == 1 and C % TMA_CHANNELS == 0 and 2 <= K <= TMA_MAX_TAPS and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0


def is_tma_fwd(x: torch.Tensor, K: int, S: int) -> bool:
    """
    Inputs of causal_conv1d_fwd_sm90 that run ConvFwdTma: is_tma_input, and S % 8 == 0 (8 outputs
    of a channel per 16-byte store of the channels-first y). All others run ConvFwd.
    """
    return is_tma_input(x, K) and S % 8 == 0


def is_tma_bwd(g: torch.Tensor, x: torch.Tensor, K: int) -> bool:
    """
    Inputs of causal_conv1d_bwd_sm90 that run ConvBwdTma: is_tma_input, and bf16 g channels-last
    contiguous (strides (S * C, C, 1)) and 16-byte aligned. All others run ConvBwd.
    """
    C = x.shape[2]
    return is_tma_input(x, K) and g.dtype == torch.bfloat16 and g.stride(2) == 1 and g.stride(1) == C and g.data_ptr() % 16 == 0


def kernel_weights(w: torch.Tensor) -> torch.Tensor:
    """
    w as the kernels read it: contiguous, bf16 or fp32.
    """
    return w.contiguous() if w.dtype in (torch.bfloat16, torch.float32) else w.float()


def tile_counters(device: torch.device, num_chan_tiles: int) -> torch.Tensor:
    """
    Zeroed per-channel-tile counters of the current stream; each launch leaves them zero again.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    counters = COUNTERS.get(key)
    if counters is None or counters.numel() < num_chan_tiles:
        counters = COUNTERS[key] = torch.zeros(max(num_chan_tiles, 1024), dtype=torch.int64, device=device)
    return counters


def causal_conv1d_fwd_sm90(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    B, L, C = x.shape
    K, S = w.shape[-1], L - offset
    if K > MAX_TAPS:
        raise ValueError(f"causal_conv1d on SM90 supports kernel widths up to {MAX_TAPS}, got {K}")
    y = x.new_empty((B, C, S)).transpose(1, 2)
    if y.numel() == 0:
        return y
    w = kernel_weights(w)
    if is_tma_fwd(x, K, S):
        sequence_splits = 4 if S == 4096 and offset == 0 and w.shape == (10240, 1, 4) and w.dtype == torch.bfloat16 else 1
        key = ("conv_fwd_tma", C, K, w.dtype == torch.float32, sequence_splits)
        fn = COMPILED.get(key)
        if fn is None:
            fn = COMPILED[key] = compile_fwd_tma(*key[1:])
        fn(x[0], w.view(-1), y.transpose(1, 2).view(-1), S, offset)
        return y
    is_row_vector, is_col_vector = C % 8 == 0 and x.data_ptr() % 16 == 0, S % 8 == 0
    key = ("conv_fwd", C, K, w.dtype == torch.float32, is_row_vector, is_col_vector)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_fwd(*key[1:])
    fn(x.view(-1), w.view(-1), y.transpose(1, 2).view(-1), B, L, S, offset)
    return y


def direct_bwd_splits(B: int, L: int, num_chan_tiles: int) -> tuple[int, int]:
    chunk_rows = BWD_WARPS * BLOCKS_PER_WARP * BWD_TOKENS
    num_chunks = (L + chunk_rows - 1) // chunk_rows
    num_splits = min(num_chunks, max(1, round(SM_COUNT * TARGET_CTAS / (B * num_chan_tiles))))
    num_iters = (num_chunks + num_splits - 1) // num_splits
    return (num_chunks + num_iters - 1) // num_iters, num_iters


def causal_conv1d_bwd_sm90(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, L, C = x.shape
    K, S = w.shape[-1], L - offset
    if K > MAX_TAPS:
        raise ValueError(f"causal_conv1d on SM90 supports kernel widths up to {MAX_TAPS}, got {K}")
    if B * S * C == 0:
        return torch.zeros_like(x), torch.zeros_like(w)
    w_kernel = kernel_weights(w)
    exact_output = w.dtype == torch.float16 or (w.dtype not in DW_TYPES and w.is_floating_point())
    storage_dtype = torch.float64 if exact_output else w.dtype if w.dtype in DW_TYPES else torch.float32
    dw_dtype = DW_TYPES[storage_dtype]
    dx = torch.empty_like(x)
    dw = torch.empty_like(w_kernel, dtype=storage_dtype)
    target_shape = (L, C, K, offset) == (4096, 10240, 4, 0)
    certify_wide = w.dtype in (torch.bfloat16, torch.float32)
    generic_certificate = not (target_shape and B in (1, 2) and w.dtype == torch.bfloat16)
    num_chan_tiles = (C + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
    pinned_tma = is_tma_bwd(g, x, K)
    if pinned_tma:
        num_splits, num_iters = 1, (L + BWD_TMA_ROWS - 1) // BWD_TMA_ROWS
    else:
        num_splits, num_iters = direct_bwd_splits(B, L, num_chan_tiles)
    batched_tma = (
        B == 2 and target_shape and x.is_contiguous() and g.is_contiguous()
        and is_tma_bwd(g[:1], x[:1], K)
    )
    batch_grid = batched_tma and w.dtype == torch.bfloat16
    if batch_grid or pinned_tma or (batched_tma and not certify_wide):
        sequence_splits = 8 if batch_grid else (4 if target_shape else 1)
        certify_single = certify_wide and pinned_tma
        key = ("conv_bwd_tma", C, K, w_kernel.dtype == torch.float32, sequence_splits, dw_dtype,
               batch_grid, certify_single)
        fn = COMPILED.get(key)
        if fn is None:
            fn = COMPILED[key] = compile_bwd_tma(*key[1:])
        if batch_grid:
            spans = B * sequence_splits
            partials = torch.empty(spans * (C * K + C // TMA_CHANNELS), dtype=torch.float64, device=x.device)
            fn(g[0], x[0], g[1], x[1], w_kernel.view(-1), dx.view(-1, C), partials, L, offset)
            return dx, final_dw(g, x, w, dw, offset, exact_output, pinned_tma, num_splits, num_iters, partials, spans)
        if certify_single:
            partials = torch.empty(sequence_splits * (C * K + C // TMA_CHANNELS),
                                   dtype=torch.float64, device=x.device)
            fn(g[0], x[0], w_kernel.view(-1), dx.view(-1), partials, L, offset)
            return dx, final_dw(g, x, w, dw, offset, exact_output, pinned_tma,
                                num_splits, num_iters, partials, sequence_splits, generic_certificate=generic_certificate)
        if sequence_splits > 1:
            partials = torch.empty((B, sequence_splits, C * K), dtype=torch.float64, device=x.device)
            for seq in range(B):
                fn(g[seq], x[seq], w_kernel.view(-1), dx[seq].view(-1), partials[seq].view(-1), L, offset)
            merge_key = ("conv_bwd_merge_spans", B * sequence_splits, dw_dtype)
            merge = COMPILED.get(merge_key)
            if merge is None:
                merge = COMPILED[merge_key] = cute.compile(
                    merge_dw_spans, flat(Float64), flat(dw_dtype), Int32(1), fake_stream(),
                    B * sequence_splits, options="--enable-tvm-ffi",
                )
            merge(partials.view(-1), dw.view(-1), C * K)
        else:
            fn(g[0], x[0], w_kernel.view(-1), dx.view(-1), dw.view(-1), L, offset)
        return dx, final_dw(g, x, w, dw, offset, exact_output, pinned_tma, num_splits, num_iters)
    if g.stride(2) != 1 and g.stride(1) != 1:
        g = g.contiguous()
    is_g_channels_first = g.stride(2) != 1
    g_stride_b, g_stride = g.stride(0), g.stride(2) if is_g_channels_first else g.stride(1)
    is_aligned = g.data_ptr() % 16 == 0 and g_stride_b % 8 == 0 and g_stride % 8 == 0
    is_g_vector = is_aligned and (S % 8 == 0 if is_g_channels_first else C % 8 == 0)
    g_shift = -offset % 8 if is_g_channels_first and is_g_vector else 0
    spans = B * num_splits
    entries_per_span = C * K + (num_chan_tiles if certify_wide else 0)
    partials = torch.empty(spans * entries_per_span, dtype=torch.float64, device=x.device)
    counters = tile_counters(x.device, num_chan_tiles)
    is_row_vector = C % 8 == 0 and x.data_ptr() % 16 == 0
    key = ("conv_bwd", C, K, w_kernel.dtype == torch.float32, is_row_vector, is_g_channels_first, is_g_vector, g_shift, dw_dtype, certify_wide)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_bwd(*key[1:])
    fn(g.as_strided((1,), (1,)), x.view(-1), w_kernel.view(-1), dx.view(-1), dw.view(-1), partials, counters, B, L, S, offset, g_stride_b, g_stride, num_splits, num_iters)
    if certify_wide:
        return dx, final_dw(g, x, w, dw, offset, exact_output, pinned_tma,
                            num_splits, num_iters, partials, 0 if generic_certificate else spans, CHANNELS_PER_CTA,
                            generic_certificate=generic_certificate)
    return dx, final_dw(g, x, w, dw, offset, exact_output, pinned_tma, num_splits, num_iters)
