"""
Causal depthwise conv1d on SM90. Contract, the PyTorch recipe the verifier checks: each y and dx
element accumulates its K taps in order k = 0 .. K - 1 as fp32 fma (ATen's depthwise order; w read
as fp32, a missing tap of the causal padding adds w * 0), then rounds once to bf16 (round to nearest
even), so y and dx are bit-identical to ATen's conv1d and its backward. dw is an fp32 sum of the
g * x terms reduced inside the backward kernel in a fixed order (deterministic, equal to the exact
sum to fp32 rounding), rounded once to w's dtype. K is at most MAX_TAPS (ValueError beyond).

Two implementations. The TMA kernels (ConvFwdTma, ConvBwdTma) take the inputs of a training step,
selected by is_tma_fwd / is_tma_bwd: one sequence (B == 1), bf16 x (and g) with x contiguous and
16-byte aligned, g channels-last contiguous, C % 64 == 0, 2 <= K <= 4, w bf16 or fp32 as the
kernels read it, and for the forward S % 8 == 0. Every other input runs the direct kernels (ConvFwd,
ConvBwd), whose dw is a fixed-order sum of per-split partials.

TMA kernels: one CTA per 64-channel tile walks the whole sequence in stages of rows. One producer
thread loads each stage's rows of the tile (with the K - 1 rows before them) with TMA into a ring of
shared-memory slots; rows outside the tensor arrive as zeros (the causal padding and the end of the
sequence). Four consumer warps compute the stage and release the slot. The backward's dw needs no
partials: each thread accumulates its rows over all stages, then the CTA reduces once (see
ConvBwdTma for the order).

Direct kernels: each thread owns TOKENS consecutive rows x 8 channels: lane % 4 picks the channel
group, lane // 4 the token block, so a warp reads eight 64-byte row segments of the channels-last
tensors and writes 8 x 16 bytes per channel of the channels-first ones, with no transpose through
shared memory.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, atom_add_u64, bf16_hi, bf16_lo, f32_as_u32, fence_gpu, ld_relaxed, ld_v4, lds_b32, lds_v4, pack_bf16, st_b64, st_f32, st_v4, sts_b32, to_ir, u32_as_f32

__all__ = ["causal_conv1d_bwd_sm90", "causal_conv1d_fwd_sm90"]

GROUPS_PER_WARP, BLOCKS_PER_WARP = 4, 8  # channel groups of 8 and token blocks of TOKENS per warp
FWD_TOKENS, FWD_WARPS = 8, 4
BWD_TOKENS, BWD_WARPS = 8, 4
CHANNELS_PER_CTA = GROUPS_PER_WARP * 8
TARGET_CTAS = 8  # per SM, for the backward's token split
TMA_CHANNELS = 64  # channels per CTA of the TMA kernels: one 128-byte box row
TMA_CONSUMERS = 128  # consumer threads of the TMA kernels: 8 channel groups x 16 token blocks
TMA_MAX_TAPS = 4  # K of the TMA kernels: 2 .. 4 (box rows and registers checked for these)
MAX_TAPS = 8  # K of any input: the direct kernels keep K taps per channel in registers and K partials per channel in smem
BWD_TMA_ROWS, BWD_TMA_STAGES = 64, 4
FWD_TMA_ROWS, FWD_TMA_STAGES, FWD_TMA_GROUPS = 128, 8, 2
SMEM_LIMIT = 227 * 1024


@dsl_user_op
def fma_f32(first: Float32, second: Float32, acc: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(asm(T.f32(), [to_ir(first, Float32, loc, ip), to_ir(second, Float32, loc, ip), to_ir(acc, Float32, loc, ip)], "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f"))


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


def load_weights(w_base: Int64, chan0: Int32, C: cutlass.Constexpr, K: cutlass.Constexpr, is_w_fp32: cutlass.Constexpr) -> list:
    """
    w[chan, k] of the thread's 8 channels as fp32, weights[chan][k]; zeros past C.
    """
    weights = []
    for chan in range(8):
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


def sum_over_token_blocks(value: Float32) -> Float32:
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
    writes one fp32 partial of its 32 channels; the last CTA of a channel tile to finish sums the
    partials in order, writes dw and resets the tile's counter.

    The thread's window of g is the BWD_TOKENS + K - 1 output rows from row0 - offset: dx[row0 + j]
    reads g window entries j .. j + K - 1, dw reads entries 0 .. BWD_TOKENS - 1 against x rows
    row0 - (K - 1) .. row0 + BWD_TOKENS - 1.
    """

    def __init__(self, C: int, K: int, is_w_fp32: bool, is_row_vector: bool, is_g_channels_first: bool, is_g_vector: bool, g_shift: int) -> None:
        self.C, self.K, self.is_w_fp32, self.is_row_vector = C, K, is_w_fp32, is_row_vector
        self.is_g_channels_first, self.is_g_vector, self.g_shift = is_g_channels_first, is_g_vector, g_shift
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
        s_dw = smem.allocate(BWD_WARPS * CHANNELS_PER_CTA * self.K * 4, byte_alignment=16).toint()
        s_is_last = smem.allocate(16, byte_alignment=16).toint()
        g_base = mG.iterator.toint() + Int64(bz) * g_stride_b * 2
        x_base = mX.iterator.toint() + Int64(bz) * L * self.C * 2
        dx_base = mDx.iterator.toint() + Int64(bz) * L * self.C * 2
        weights = load_weights(mW.iterator.toint(), tile.chan0, self.C, self.K, self.is_w_fp32)
        acc_dw = cute.make_rmem_tensor((8 * self.K,), Float32)
        for i in cutlass.range_constexpr(8 * self.K):
            acc_dw[i] = Float32(0.0)

        for iter_idx in cutlass.range(num_iters, unroll=1):
            row0 = (by * num_iters + iter_idx) * self.chunk_tokens + tile.block_offset
            g_window = self.load_g_window(g_base, row0 - offset, S, g_stride, tile.chan0)
            x_rows = self.load_x_window(x_base, row0 - (self.K - 1), L, tile.chan0)
            self.store_dx(dx_base, g_window, weights, row0, L, tile.chan0)
            self.accumulate_dw(acc_dw, g_window, x_rows)

        self.reduce_dw_in_cta(acc_dw, tidx, s_dw)
        partial_idx = bz * num_splits + by
        self.store_partial(mPartials.iterator.toint(), s_dw, partial_idx, bx, tidx)
        is_last = self.take_counter(mCounters.iterator.toint() + Int64(bx) * 8, B * num_splits, tidx, s_is_last)
        if is_last != 0:
            self.store_dw(mPartials.iterator.toint(), mDw.iterator.toint(), mCounters.iterator.toint() + Int64(bx) * 8, B * num_splits, bx, tidx)

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
                    acc = fma_f32(g_window[j][chan], element(x_rows[j + k], chan), acc)
                acc_dw[chan * K + k] = acc

    @cute.jit
    def reduce_dw_in_cta(self, acc_dw, tidx, s_dw: Int32) -> None:
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
            sts_b32(s_dw + entry * 4, f32_as_u32(acc_dw[i]), Int32(lane < GROUPS_PER_WARP))
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
            acc = Float32(0.0)
            for warp in cutlass.range_constexpr(BWD_WARPS):
                acc = acc + u32_as_f32(lds_b32(s_dw + (warp * CHANNELS_PER_CTA * K + entry * entry_ok) * 4))
            chan_idx = chan_tile_idx * CHANNELS_PER_CTA + entry // K
            st_f32(partials_base + ((Int64(partial_idx) * C + chan_idx) * K + entry % K) * 4, acc, entry_ok & Int32(chan_idx < C))

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
            acc = Float32(0.0)
            for partial_idx in cutlass.range(num_partials, unroll=1):
                acc = acc + u32_as_f32(Uint32(ld_relaxed(partials_base + ((Int64(partial_idx) * C + chan_idx) * K + entry % K) * 4, entry_ok)))
            elem_idx = Int64(chan_idx) * K + entry % K
            if cutlass.const_expr(self.is_w_fp32):
                st_f32(dw_base + elem_idx * 4, acc, entry_ok)
            else:
                st_b16(dw_base + elem_idx * 2, pack_bf16(acc, acc), entry_ok)
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
    Backward on the TMA ring, one CTA per 64-channel tile over all L rows (B == 1). Stage s covers
    dx rows [s * stage_rows, + stage_rows): the producer loads g rows [s * stage_rows - offset, +
    stage_rows + K - 1) and x rows [s * stage_rows - (K - 1), + stage_rows + K - 1) of the tile into
    one slot. Consumer thread: channel group tid % 8 (8 channels), token block tid // 8 (stage_rows
    / 16 rows); per row j in order it loads box rows j + K - 1 of g and x (a sliding window of K
    rows), writes dx[j] = sum over k in order of w[k] * g[j + K - 1 - k] (fp32 fma, ATen's order)
    with one 16-byte store, and adds g[j] * x[j + k] to its dw sums.

    dw order (deterministic, no atomics or partials): (1) each thread's sum[chan][k] over its rows
    in increasing order across all stages; (2) butterfly over the 4 token blocks of a warp sharing
    the channel group (shuffle xor 8, then xor 16); (3) the 4 warps' sums in warp order from 0.0
    through shared memory; (4) one rounding to w's dtype.
    """

    def __init__(self, C: int, K: int, is_w_fp32: bool) -> None:
        self.C, self.K, self.is_w_fp32 = C, K, is_w_fp32
        self.stage_rows, self.tokens = BWD_TMA_ROWS, BWD_TMA_ROWS // 16
        self.box_rows = self.stage_rows + K - 1
        self.box_bytes = self.box_rows * TMA_CHANNELS * 2
        self.box_stride = (self.box_bytes + 127) // 128 * 128
        self.ring = TmaRing(BWD_TMA_STAGES, 2 * self.box_stride, 1)
        self.producer_warp = 0
        self.dw_bytes = 4 * TMA_CHANNELS * K * 4
        assert self.ring.num_slots * self.ring.slot_bytes + self.dw_bytes + 1024 <= SMEM_LIMIT

    @cute.jit
    def __call__(self, mG: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, L: Int32, offset: Int32, stream) -> None:
        box = cute.make_layout((self.box_rows, TMA_CHANNELS), stride=(TMA_CHANNELS, 1))
        atom_g, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mG, box, (self.box_rows, TMA_CHANNELS))
        atom_x, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mX, box, (self.box_rows, TMA_CHANNELS))
        self.kernel(atom_g, atom_x, mW, mDx, mDw, L, offset).launch(grid=[self.C // TMA_CHANNELS, 1, 1], block=[32 + TMA_CONSUMERS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, atom_g: cute.CopyAtom, atom_x: cute.CopyAtom, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, L: Int32, offset: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        smem = cutlass.memory.SmemAllocator()
        slots, full0, empty0 = self.ring.allocate(smem)
        s_dw = Int32(smem.allocate(self.dw_bytes, byte_alignment=16).toint())
        if tidx == 0:
            init_ring(full0, empty0, self.ring.num_slots)
        cute.arch.sync_threads()
        num_stages = (L + self.stage_rows - 1) // self.stage_rows
        tile_chan0 = bx * TMA_CHANNELS
        if warp == self.producer_warp:
            if tidx == 0:
                self.load_loop(atom_g, atom_x, slots, full0, empty0, num_stages, tile_chan0, offset)
        else:
            self.compute_loop(mW, mDx, mDw, slots, full0, empty0, s_dw, num_stages, tile_chan0, L, tidx - 32)

    @cute.jit
    def load_loop(self, atom_g: cute.CopyAtom, atom_x: cute.CopyAtom, slots: Int32, full0: Int32, empty0: Int32, num_stages: Int32, tile_chan0: Int32, offset: Int32) -> None:
        cpasync.prefetch_descriptor(atom_g)
        cpasync.prefetch_descriptor(atom_x)
        g_desc, x_desc = tma_desc_addr(atom_g), tma_desc_addr(atom_x)
        for stage_idx in cutlass.range(num_stages, unroll=1):
            slot_idx, parity = self.ring.slot(stage_idx)
            mbar_wait(empty0 + 8 * slot_idx, parity ^ 1)
            full = full0 + 8 * slot_idx
            expect_slot_bytes(full, 2 * self.box_bytes)
            slot_addr = slots + slot_idx * self.ring.slot_bytes
            row0 = stage_idx * self.stage_rows
            tma_load_rows(slot_addr, g_desc, tile_chan0, row0 - offset, full)
            tma_load_rows(slot_addr + self.box_stride, x_desc, tile_chan0, row0 - (self.K - 1), full)

    @cute.jit
    def compute_loop(self, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, slots: Int32, full0: Int32, empty0: Int32, s_dw: Int32, num_stages: Int32, tile_chan0: Int32, L: Int32, consumer_idx: Int32) -> None:
        group_idx, block_idx = consumer_idx % 8, consumer_idx // 8
        chan0 = tile_chan0 + group_idx * 8
        weights = load_weights(mW.iterator.toint(), chan0, self.C, self.K, self.is_w_fp32)
        acc_dw = cute.make_rmem_tensor((8 * self.K,), Float32)
        for i in cutlass.range_constexpr(8 * self.K):
            acc_dw[i] = Float32(0.0)
        local0 = block_idx * self.tokens
        for stage_idx in cutlass.range(num_stages, unroll=1):
            slot_idx, parity = self.ring.slot(stage_idx)
            mbar_wait(full0 + 8 * slot_idx, parity)
            g_slot = slots + slot_idx * self.ring.slot_bytes
            self.compute_stage(g_slot, g_slot + self.box_stride, empty0 + 8 * slot_idx, local0, group_idx, weights, acc_dw, mDx.iterator.toint(), stage_idx * self.stage_rows + local0, L, chan0)
        self.reduce_dw(acc_dw, consumer_idx, s_dw, mDw.iterator.toint(), tile_chan0)

    def load_box_row(self, slot_addr: Int32, row_idx: Int32, group_idx: Int32) -> list:
        """
        Box row row_idx, channels group_idx * 8 .. + 7, as fp32.
        """
        words = lds_v4(slot_addr + (row_idx * TMA_CHANNELS + group_idx * 8) * 2)
        return [element(words, chan) for chan in range(8)]

    def compute_stage(self, g_slot: Int32, x_slot: Int32, empty: Int32, local0: Int32, group_idx: Int32, weights: list, acc_dw, dx_base: Int64, row0: Int32, L: Int32, chan0: Int32) -> None:
        """
        The thread's rows row0 .. row0 + stage_rows / 16 - 1 of a stage: dx and the dw sums over a
        sliding window of K box rows of g and x; the slot is released after its last load.
        """
        K = self.K
        g_rows = [self.load_box_row(g_slot, local0 + i, group_idx) for i in range(K - 1)]
        x_rows = [self.load_box_row(x_slot, local0 + i, group_idx) for i in range(K - 1)]
        for j in range(self.tokens):
            g_rows.append(self.load_box_row(g_slot, local0 + j + K - 1, group_idx))
            x_rows.append(self.load_box_row(x_slot, local0 + j + K - 1, group_idx))
            if j == self.tokens - 1:
                release_slot(empty)
            words = []
            for pair in range(4):
                accs = []
                for chan in (2 * pair, 2 * pair + 1):
                    acc = Float32(0.0)
                    for k in range(K):
                        acc = fma_f32(weights[chan][k], g_rows[j + K - 1 - k][chan], acc)
                    accs.append(acc)
                words.append(pack_bf16(accs[0], accs[1]))
            row_idx = row0 + j
            st_v4(dx_base + (Int64(row_idx) * self.C + chan0) * 2, tuple(words), Int32(row_idx < L))
            for chan in range(8):
                for k in range(K):
                    acc_dw[chan * K + k] = fma_f32(g_rows[j][chan], x_rows[j + k][chan], acc_dw[chan * K + k])

    @cute.jit
    def reduce_dw(self, acc_dw, consumer_idx: Int32, s_dw: Int32, dw_base: Int64, tile_chan0: Int32) -> None:
        """
        Steps (2) to (4) of the dw order: butterfly within each warp, lanes 0 .. 7 of warp w store
        s_dw[w][channel][k], then after a barrier of the consumers thread e < 64 * K writes
        dw[tile_chan0 * K + e] = sum over w in order of s_dw[w][e], rounded once.
        """
        K = self.K
        lane, consumer_warp = consumer_idx % 32, consumer_idx // 32
        for i in cutlass.range_constexpr(8 * K):
            value = acc_dw[i]
            value = value + cute.arch.shuffle_sync_bfly(value, offset=8)
            value = value + cute.arch.shuffle_sync_bfly(value, offset=16)
            acc_dw[i] = value
        for i in cutlass.range_constexpr(8 * K):
            entry = (consumer_warp * TMA_CHANNELS + lane * 8) * K + i
            sts_b32(s_dw + entry * 4, f32_as_u32(acc_dw[i]), Int32(lane < 8))
        cute.arch.barrier(barrier_id=1, number_of_threads=TMA_CONSUMERS)
        for entry0 in cutlass.range_constexpr(0, TMA_CHANNELS * K, TMA_CONSUMERS):
            entry = entry0 + consumer_idx
            entry_ok = Int32(entry < TMA_CHANNELS * K)
            acc = Float32(0.0)
            for warp_idx in cutlass.range_constexpr(TMA_CONSUMERS // 32):
                acc = acc + u32_as_f32(lds_b32(s_dw + (warp_idx * TMA_CHANNELS * K + entry * entry_ok) * 4))
            elem_idx = Int64(tile_chan0) * K + entry
            if cutlass.const_expr(self.is_w_fp32):
                st_f32(dw_base + elem_idx * 4, acc, entry_ok)
            else:
                st_b16(dw_base + elem_idx * 2, pack_bf16(acc, acc), entry_ok)


class ConvFwdTma:
    """
    Forward on the TMA ring, one CTA per 64-channel tile over all S outputs (B == 1, S % 8 == 0).
    Stage s covers outputs [s * stage_rows, + stage_rows) and loads x rows
    [s * stage_rows + offset - (K - 1), + stage_rows + K - 1). Two consumer groups take alternate
    stages. Consumer thread: channel group (warp % 2) * 4 + lane % 4, token block
    (warp // 2) * 8 + lane // 4 of its group (stage_rows / 16 = 8 outputs), so one 16-byte store of
    a warp writes 4 channels x 128 contiguous bytes of the channels-first y.
    """

    def __init__(self, C: int, K: int, is_w_fp32: bool) -> None:
        self.C, self.K, self.is_w_fp32 = C, K, is_w_fp32
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
        self.kernel(atom_x, mW, mY, S, offset).launch(grid=[self.C // TMA_CHANNELS, 1, 1], block=[32 + TMA_CONSUMERS * FWD_TMA_GROUPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, atom_x: cute.CopyAtom, mW: cute.Tensor, mY: cute.Tensor, S: Int32, offset: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        smem = cutlass.memory.SmemAllocator()
        slots, full0, empty0 = self.ring.allocate(smem)
        if tidx == 0:
            init_ring(full0, empty0, self.ring.num_slots)
        cute.arch.sync_threads()
        num_stages = (S + self.stage_rows - 1) // self.stage_rows
        tile_chan0 = bx * TMA_CHANNELS
        if warp == self.producer_warp:
            if tidx == 0:
                self.load_loop(atom_x, slots, full0, empty0, num_stages, tile_chan0, offset)
        else:
            self.compute_loop(mW, mY, slots, full0, empty0, num_stages, tile_chan0, S, tidx - 32)

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
    def compute_loop(self, mW: cute.Tensor, mY: cute.Tensor, slots: Int32, full0: Int32, empty0: Int32, num_stages: Int32, tile_chan0: Int32, S: Int32, consumer_idx: Int32) -> None:
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
            self.store_y(mY.iterator.toint(), rows, weights, stage_idx * self.stage_rows + local0, S, chan0)

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


def flat(dtype: type) -> object:
    return fake(dtype, (cute.sym_int(),), (1,), 2 if dtype is cutlass.BFloat16 else 4)


def compile_fwd(C: int, K: int, is_w_fp32: bool, is_row_vector: bool, is_col_vector: bool) -> Callable:
    w = flat(Float32 if is_w_fp32 else cutlass.BFloat16)
    return cute.compile(ConvFwd(C, K, is_w_fp32, is_row_vector, is_col_vector), flat(cutlass.BFloat16), w, flat(cutlass.BFloat16), Int32(1), Int32(1), Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def compile_bwd(C: int, K: int, is_w_fp32: bool, is_row_vector: bool, is_g_channels_first: bool, is_g_vector: bool, g_shift: int) -> Callable:
    w = flat(Float32 if is_w_fp32 else cutlass.BFloat16)
    tensors = (flat(cutlass.BFloat16), flat(cutlass.BFloat16), w, flat(cutlass.BFloat16), w, flat(Float32), fake(Int64, (cute.sym_int(),), (1,), 8))
    scalars = (Int32(1), Int32(1), Int32(1), Int32(0), Int64(0), Int64(0), Int32(1), Int32(1))
    return cute.compile(ConvBwd(C, K, is_w_fp32, is_row_vector, is_g_channels_first, is_g_vector, g_shift), *tensors, *scalars, fake_stream(), options="--enable-tvm-ffi")


def rows_of(dtype: type, C: int) -> object:
    """
    A (rows, C) row-major tensor with a dynamic row count, as the TMA descriptors read it.
    """
    return fake(dtype, (cute.sym_int(), C), (C, 1))


def compile_fwd_tma(C: int, K: int, is_w_fp32: bool) -> Callable:
    w = flat(Float32 if is_w_fp32 else cutlass.BFloat16)
    return cute.compile(ConvFwdTma(C, K, is_w_fp32), rows_of(cutlass.BFloat16, C), w, flat(cutlass.BFloat16), Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def compile_bwd_tma(C: int, K: int, is_w_fp32: bool) -> Callable:
    w = flat(Float32 if is_w_fp32 else cutlass.BFloat16)
    return cute.compile(ConvBwdTma(C, K, is_w_fp32), rows_of(cutlass.BFloat16, C), rows_of(cutlass.BFloat16, C), w, flat(cutlass.BFloat16), w, Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


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
        key = ("conv_fwd_tma", C, K, w.dtype == torch.float32)
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


def causal_conv1d_bwd_sm90(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, L, C = x.shape
    K, S = w.shape[-1], L - offset
    if K > MAX_TAPS:
        raise ValueError(f"causal_conv1d on SM90 supports kernel widths up to {MAX_TAPS}, got {K}")
    if B * S * C == 0:
        return torch.zeros_like(x), torch.zeros_like(w)
    if is_tma_bwd(g, x, K):
        w_kernel = kernel_weights(w)
        dx, dw = torch.empty_like(x), torch.empty_like(w_kernel)
        key = ("conv_bwd_tma", C, K, w_kernel.dtype == torch.float32)
        fn = COMPILED.get(key)
        if fn is None:
            fn = COMPILED[key] = compile_bwd_tma(*key[1:])
        fn(g[0], x[0], w_kernel.view(-1), dx.view(-1), dw.view(-1), L, offset)
        return dx, dw.to(w.dtype).view(w.shape)
    if g.stride(2) != 1 and g.stride(1) != 1:
        g = g.contiguous()
    is_g_channels_first = g.stride(2) != 1
    g_stride_b, g_stride = g.stride(0), g.stride(2) if is_g_channels_first else g.stride(1)
    is_aligned = g.data_ptr() % 16 == 0 and g_stride_b % 8 == 0 and g_stride % 8 == 0
    is_g_vector = is_aligned and (S % 8 == 0 if is_g_channels_first else C % 8 == 0)
    g_shift = -offset % 8 if is_g_channels_first and is_g_vector else 0
    w_kernel = kernel_weights(w)
    dx, dw = torch.empty_like(x), torch.empty_like(w_kernel)
    num_chan_tiles = (C + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
    num_chunks = (L + BWD_WARPS * BLOCKS_PER_WARP * BWD_TOKENS - 1) // (BWD_WARPS * BLOCKS_PER_WARP * BWD_TOKENS)
    num_splits = min(num_chunks, max(1, round(SM_COUNT * TARGET_CTAS / (B * num_chan_tiles))))
    num_iters = (num_chunks + num_splits - 1) // num_splits
    num_splits = (num_chunks + num_iters - 1) // num_iters
    partials = torch.empty(B * num_splits * C * K, dtype=torch.float32, device=x.device)
    counters = tile_counters(x.device, num_chan_tiles)
    is_row_vector = C % 8 == 0 and x.data_ptr() % 16 == 0
    key = ("conv_bwd", C, K, w_kernel.dtype == torch.float32, is_row_vector, is_g_channels_first, is_g_vector, g_shift)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_bwd(*key[1:])
    fn(g.as_strided((1,), (1,)), x.view(-1), w_kernel.view(-1), dx.view(-1), dw.view(-1), partials, counters, B, L, S, offset, g_stride_b, g_stride, num_splits, num_iters)
    return dx, dw.to(w.dtype).view(w.shape)
