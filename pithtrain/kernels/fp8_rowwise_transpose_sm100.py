"""
SM100 fp8_quant_grad in TIRx-lite: rowwise 1x128 FP8 of x and of x.T from one read of x.

Fast path (bf16, unit column stride, 16B-aligned rows, N % 8 == 0, M % 16 == 0): one CTA per
128x128 tile (non-persistent; the hardware CTA scheduler balances the tail), 8 warps, 4 CTAs/SM.
Every thread issues its eight 16-byte loads up front (8 lanes per row: chunks k and k+8), copies
them into an XOR-swizzled shared tile, quantizes its rows straight from registers (128 B coalesced
stores per row of x_fp8), then one barrier and the column phase from shared memory (8 lanes per
column pair over 16-row groups, column abs-max reduced in-warp, 128 B coalesced stores per row of
xt_fp8). Grids too small to fill the GPU run four CTAs per tile instead (two for rows, two for
columns). Scales use an exact integer E8M0 formula; quantization is an exact power-of-two bf16x2
multiply plus cvt.rn.satfinite.e4m3x2.bf16x2. Everything else (fp32/fp16, odd shapes, arbitrary
strides) takes a scalar generic kernel.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["fp8_quant_grad_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


BLOCK = 128
THREADS = 256


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def i64(value):
    return txl.Cast("int64", value)


def u32(value):
    return txl.uint32(value)


def e8m0_scale_bits(amax):
    """
    E8M0 scale of one block as (scale f32 bits, reciprocal f32 bits), both uint32.

    scale = 2^ceil(log2(max(amax, 1e-4) / 448)). With 448 = 7 * 2^6 and c = m * 2^E
    (1 <= m < 2), c / 448 rounds up to 2^(E-8) when m <= 1.75 and to 2^(E-7) otherwise; adding
    0x1FFFFF carries into the exponent exactly when the mantissa field exceeds 0x600000.
    """
    clamped = txl.local_scalar("float32")
    txl.ptx.max.f32(clamped, amax, txl.float32(1e-4))
    scale_bits = txl.local_scalar("uint32", init=txl.bitwise_and(txl.reinterpret(clamped, ty="uint32") + u32(0x1FFFFF), u32(0x7F800000)) - u32(8 << 23))
    rcp_bits = txl.local_scalar("uint32", init=u32(254 << 23) - scale_bits)
    return scale_bits, rcp_bits


def broadcast_rcp_bf16x2(rcp_bits):
    """
    Reciprocal (an exact power of two) broadcast into both bf16 halves.
    """
    rcp_hi = txl.shift_right(rcp_bits, u32(16))
    return txl.local_scalar("uint32", init=txl.bitwise_or(rcp_hi, txl.shift_left(rcp_hi, u32(16))))


def quantize_bf16x2(word, rcp_bf16x2):
    """
    Two bf16 -> two e4m3 bytes (low element in the low byte) as uint16.
    """
    scaled = txl.local_scalar("uint32")
    txl.ptx.mul.rn.bf16x2(scaled, word, rcp_bf16x2)
    out = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(out, scaled)
    return out


def pack_u16x2(lo, hi):
    out = txl.local_scalar("uint32")
    txl.ptx.mov.b32(out, lo, hi)
    return out


def start_iket_range(name):
    """
    One IKET range per warp; stripped by the ordinary TIRx pipeline.
    """
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
    with txl.if_(txl.lane_id() == 0), txl.then_():
        txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def end_iket_range(token):
    txl.cuda.iket.range_end(token[0])


TILE_WORDS = BLOCK * 64


class TileThread:
    """
    A thread's place in its 128x128 tile and the matrix bounds. Lane chunk (lane % 8) loads the
    16-byte chunks chunk and chunk + 8 of rows thread_row + 32 * i.
    """

    def __init__(self, warp, lane, chunk, thread_row, row_start, col_start, tile_row_idx, tile_col_idx, M, N, M_i64, N_i64):
        self.warp, self.lane, self.chunk, self.thread_row = warp, lane, chunk, thread_row
        self.row_start, self.col_start, self.tile_row_idx, self.tile_col_idx = row_start, col_start, tile_row_idx, tile_col_idx
        self.M, self.N, self.M_i64, self.N_i64 = M, N, M_i64, N_i64


def load_rows(x, words, tile_base, is_interior, row_start, num_row_groups, row_stride, tile):
    """
    Load chunks chunk and chunk + 8 of the thread's num_row_groups rows from row_start into words;
    out-of-bounds chunks read as zero.
    """
    chunk, thread_row, col_start = tile.chunk, tile.thread_row, tile.col_start
    M, N = tile.M, tile.N
    with txl.if_(is_interior):
        with txl.then_():
            for row_group in range(num_row_groups):
                for half in range(2):
                    first_word = 8 * row_group + 4 * half
                    txl.ptx.ld.global_.v4.b32(words[first_word], words[first_word + 1], words[first_word + 2], words[first_word + 3], txl.address_of(x[tile_base + i64(thread_row + 32 * row_group) * row_stride + i64((chunk + 8 * half) * 8)]))
        with txl.else_():
            for row_group in range(num_row_groups):
                for half in range(2):
                    first_word = 8 * row_group + 4 * half
                    for j in range(4):
                        txl.assign(words[first_word + j], u32(0))
                    with txl.if_((row_start + thread_row + 32 * row_group < M) & (col_start + (chunk + 8 * half) * 8 < N)), txl.then_():
                        txl.ptx.ld.global_.v4.b32(words[first_word], words[first_word + 1], words[first_word + 2], words[first_word + 3], txl.address_of(x[tile_base + i64(thread_row + 32 * row_group) * row_stride + i64((chunk + 8 * half) * 8)]))


def quantize_row(words, row_group):
    """
    E8M0 scale bits and the four FP8 words of one row, its 128 elements held by 8 lanes.
    """
    acc_amax = txl.local_scalar("uint32", init=words[8 * row_group])
    for j in range(1, 8):
        txl.ptx.max.xorsign.abs.bf16x2(acc_amax, acc_amax, words[8 * row_group + j])
    amax_bits = txl.local_scalar("uint32", init=txl.bitwise_and(acc_amax, u32(0x7FFF7FFF)))
    txl.ptx.max.u32(amax_bits, txl.bitwise_and(amax_bits, u32(0xFFFF)), txl.shift_right(amax_bits, u32(16)))
    peer_amax = txl.local_scalar("uint32")
    for xor_mask in (1, 2, 4):
        txl.ptx.shfl_sync.bfly.b32(peer_amax, amax_bits, u32(xor_mask), u32(31), u32(0xFFFFFFFF))
        txl.ptx.max.u32(amax_bits, amax_bits, peer_amax)
    scale_bits, rcp_bits = e8m0_scale_bits(txl.reinterpret(txl.shift_left(amax_bits, u32(16)), ty="float32"))
    rcp_bf16x2 = broadcast_rcp_bf16x2(rcp_bits)
    quant_words = []
    for half in range(2):
        first_word = 8 * row_group + 4 * half
        quant_words.append(pack_u16x2(quantize_bf16x2(words[first_word], rcp_bf16x2), quantize_bf16x2(words[first_word + 1], rcp_bf16x2)))
        quant_words.append(pack_u16x2(quantize_bf16x2(words[first_word + 2], rcp_bf16x2), quantize_bf16x2(words[first_word + 3], rcp_bf16x2)))
    return scale_bits, quant_words


def store_quant_row(out_tok, out_tile_base, row_group, quant_words, tile):
    """
    Store the thread's 16 FP8 bytes of one row; chunks past N are skipped.
    """
    chunk, thread_row, col_start = tile.chunk, tile.thread_row, tile.col_start
    N, N_i64 = tile.N, tile.N_i64
    out_idx = out_tile_base + i64(thread_row + 32 * row_group) * N_i64 + i64(chunk * 8)
    with txl.if_(col_start + 128 <= N):
        with txl.then_():
            txl.ptx.st.global_.v2.b32(txl.address_of(out_tok[out_idx]), quant_words[0], quant_words[1])
            txl.ptx.st.global_.v2.b32(txl.address_of(out_tok[out_idx + 64]), quant_words[2], quant_words[3])
        with txl.else_():
            for half in range(2):
                with txl.if_(col_start + chunk * 8 + 64 * half < N), txl.then_():
                    txl.ptx.st.global_.v2.b32(txl.address_of(out_tok[out_idx + 64 * half]), quant_words[2 * half], quant_words[2 * half + 1])


def quantize_column_pair(col_words):
    """
    E8M0 scale bits and FP8 words of two adjacent 16-row column segments, from 16 bf16x2 words
    (one per row); the amax of each column is reduced over the 8 lanes holding its 128 rows.
    """
    acc_amax = txl.local_scalar("uint32", init=col_words[0])
    for j in range(1, 16):
        txl.ptx.max.xorsign.abs.bf16x2(acc_amax, acc_amax, col_words[j])
    txl.assign(acc_amax, txl.bitwise_and(acc_amax, u32(0x7FFF7FFF)))
    peer_amax = txl.local_scalar("uint32")
    for xor_mask in (1, 2, 4):
        txl.ptx.shfl_sync.bfly.b32(peer_amax, acc_amax, u32(xor_mask), u32(31), u32(0xFFFFFFFF))
        txl.ptx.max.bf16x2(acc_amax, acc_amax, peer_amax)
    scale_bits_lo, rcp_bits_lo = e8m0_scale_bits(txl.reinterpret(txl.shift_left(acc_amax, u32(16)), ty="float32"))
    scale_bits_hi, rcp_bits_hi = e8m0_scale_bits(txl.reinterpret(txl.bitwise_and(acc_amax, u32(0xFFFF0000)), ty="float32"))
    rcp_bf16x2_lo, rcp_bf16x2_hi = broadcast_rcp_bf16x2(rcp_bits_lo), broadcast_rcp_bf16x2(rcp_bits_hi)
    quant_lo, quant_hi = [], []
    for j in range(0, 16, 2):
        packed_lo, packed_hi = txl.local_scalar("uint32"), txl.local_scalar("uint32")
        txl.ptx.prmt.b32(packed_lo, col_words[j], col_words[j + 1], u32(0x5410))
        txl.ptx.prmt.b32(packed_hi, col_words[j], col_words[j + 1], u32(0x7632))
        quant_lo.append(quantize_bf16x2(packed_lo, rcp_bf16x2_lo))
        quant_hi.append(quantize_bf16x2(packed_hi, rcp_bf16x2_hi))
    out_words_lo = [pack_u16x2(quant_lo[2 * j], quant_lo[2 * j + 1]) for j in range(4)]
    out_words_hi = [pack_u16x2(quant_hi[2 * j], quant_hi[2 * j + 1]) for j in range(4)]
    return scale_bits_lo, scale_bits_hi, out_words_lo, out_words_hi


def store_column_pair(out_t, scale_t, col_idx, row_block, column_quant, tile):
    """
    Store the FP8 words of columns col_idx and col_idx + 1 over the 16 rows of row_block; lanes of
    row block 0 also store the two scales.
    """
    scale_bits_lo, scale_bits_hi, out_words_lo, out_words_hi = column_quant
    row_start, tile_row_idx, M, N = tile.row_start, tile.tile_row_idx, tile.M, tile.N
    M_i64, N_i64 = tile.M_i64, tile.N_i64
    with txl.if_((col_idx < N) & (row_start + row_block * 16 < M)), txl.then_():
        out_t_idx = txl.local_scalar("int64", init=i64(col_idx) * M_i64 + i64(row_start + row_block * 16))
        txl.ptx.st.global_.v4.b32(txl.address_of(out_t[out_t_idx]), out_words_lo[0], out_words_lo[1], out_words_lo[2], out_words_lo[3])
        txl.ptx.st.global_.v4.b32(txl.address_of(out_t[out_t_idx + M_i64]), out_words_hi[0], out_words_hi[1], out_words_hi[2], out_words_hi[3])
        with txl.if_(row_block == 0), txl.then_():
            scale_t_idx = txl.local_scalar("int64", init=i64(tile_row_idx) * N_i64 + i64(col_idx))
            txl.ptx.st.global_.v2.b32(txl.address_of(scale_t[scale_t_idx]), scale_bits_lo, scale_bits_hi)


def stage_tile(s_tile, words, tile):
    """
    Copy the thread's 32 words into the XOR-swizzled shared tile (64 words per row).
    """
    chunk, thread_row = tile.chunk, tile.thread_row
    for row_group in range(4):
        row = thread_row + 32 * row_group
        swizzle = (row >> 4) & 7
        for half in range(2):
            first_word = 8 * row_group + 4 * half
            txl.ptx.st.shared.v4.b32(s_tile.ptr_to([row * 64 + (((chunk + 8 * half) ^ half ^ swizzle) << 2)]), words[first_word], words[first_word + 1], words[first_word + 2], words[first_word + 3])


def prefetch_next_wave(x, cta_idx, prefetch_distance, num_col_tiles, x_offset, row_stride, tile):
    """
    L2 prefetch of the tile one resident wave ahead (interior tiles only), one 256 B row per thread.
    """
    M, N = tile.M, tile.N
    prefetch_cta_idx = cta_idx + prefetch_distance
    prefetch_row_tile = txl.local_scalar("int32", init=txl.truncdiv(prefetch_cta_idx, num_col_tiles))
    prefetch_col_tile = prefetch_cta_idx - prefetch_row_tile * num_col_tiles
    tidx = txl.thread_id()
    with txl.if_((tidx < 128) & (prefetch_row_tile * 128 + 128 <= M) & (prefetch_col_tile * 128 + 128 <= N)), txl.then_():
        txl.ptx.cp.async_.bulk.prefetch.L2.global_(txl.address_of(x[x_offset + i64(prefetch_row_tile * 128 + tidx) * row_stride + i64(prefetch_col_tile * 128)]), u32(256))


def quantize_tile_rows(out_tok, s_row_scale, words, tile):
    """
    Quantize the thread's four rows from registers; the row scales go to shared memory.
    """
    chunk, thread_row, row_start, col_start = tile.chunk, tile.thread_row, tile.row_start, tile.col_start
    M, N_i64 = tile.M, tile.N_i64
    out_tile_base = txl.local_scalar("int64", init=i64(row_start) * N_i64 + i64(col_start))
    for row_group in range(4):
        row_idx = row_start + thread_row + 32 * row_group
        scale_bits, quant_words = quantize_row(words, row_group)
        with txl.if_(row_idx < M), txl.then_():
            store_quant_row(out_tok, out_tile_base, row_group, quant_words, tile)
        with txl.if_(chunk == 0), txl.then_():
            txl.ptx.st.shared.b32(s_row_scale.ptr_to([thread_row + 32 * row_group]), scale_bits)


def store_row_scales(scale_tok, s_row_scale, tile):
    """
    Threads 0-127 copy the tile's row scales from shared memory to scale_tok.
    """
    row_start, tile_col_idx, M, M_i64 = tile.row_start, tile.tile_col_idx, tile.M, tile.M_i64
    tidx = txl.thread_id()
    with txl.if_((tidx < 128) & (row_start + tidx < M)), txl.then_():
        row_scale_bits = txl.local_scalar("uint32")
        txl.ptx.ld.shared.b32(row_scale_bits, s_row_scale.ptr_to([tidx]))
        txl.ptx.st.global_.b32(txl.address_of(scale_tok[i64(tile_col_idx) * M_i64 + i64(row_start + tidx)]), row_scale_bits)


def quantize_tile_columns(out_t, scale_t, s_tile, tile):
    """
    Quantize the tile's columns from shared memory: lane % 8 picks a 16-row block, each thread
    covers two column pairs.
    """
    warp, lane, col_start = tile.warp, tile.lane, tile.col_start
    row_block, pair_lane = lane & 7, lane >> 3
    col_words = txl.alloc_local([16], "uint32")
    for pair_group in range(2):
        pair_idx = warp * 4 + pair_lane + 32 * pair_group
        col_chunk = pair_idx >> 2
        col_offset = row_block * 16 * 64 + ((col_chunk ^ ((col_chunk >> 3) & 1) ^ row_block) << 2) + (pair_idx & 3)
        for j in range(16):
            txl.ptx.ld.shared.b32(col_words[j], s_tile.ptr_to([col_offset + j * 64]))
        column_quant = quantize_column_pair(col_words)
        col_idx = col_start + 2 * pair_idx
        store_column_pair(out_t, scale_t, col_idx, row_block, column_quant, tile)


def build_tile_kernel():
    """
    Non-persistent: one CTA per 128x128 tile (hardware scheduling balances the tail). Every
    thread issues its eight 16-byte loads up front (registers), writes them to the swizzled smem
    tile, runs the row phase from registers, then one barrier and the column phase from smem.
    """

    @txl.kernel()
    def fp8_quant_grad_tile(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out_tok: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_tok: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_t: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, row_stride: txl.int64, x_offset: txl.int64, prefetch_distance: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=((N + 127) // 128) * ((M + 127) // 128), block=(THREADS // 32) * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=4))
        cta_idx = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        pool = txl.smem_pool()
        s_tile = pool.alloc([TILE_WORDS], "uint32", align=128)
        s_row_scale = pool.alloc([BLOCK], "uint32", align=16)
        num_col_tiles = (N + 127) >> 7
        tile_row_idx = txl.local_scalar("int32", init=txl.truncdiv(cta_idx, num_col_tiles))
        tile_col_idx = txl.local_scalar("int32", init=cta_idx - tile_row_idx * num_col_tiles)
        row_start, col_start = tile_row_idx * 128, tile_col_idx * 128
        M_i64, N_i64 = i64(M), i64(N)
        chunk = lane & 7
        thread_row = warp * 4 + (lane >> 3)
        words = txl.alloc_local([32], "uint32")
        tile = TileThread(warp, lane, chunk, thread_row, row_start, col_start, tile_row_idx, tile_col_idx, M, N, M_i64, N_i64)

        tile_base = txl.local_scalar("int64", init=x_offset + i64(row_start) * row_stride + i64(col_start))
        is_interior = (row_start + 128 <= M) & (col_start + 128 <= N)
        range_token = start_iket_range("global-load")
        load_rows(x, words, tile_base, is_interior, row_start, 4, row_stride, tile)
        end_iket_range(range_token)

        range_token = start_iket_range("shared-stage")
        stage_tile(s_tile, words, tile)
        end_iket_range(range_token)

        range_token = start_iket_range("l2-prefetch")
        prefetch_next_wave(x, cta_idx, prefetch_distance, num_col_tiles, x_offset, row_stride, tile)
        end_iket_range(range_token)

        range_token = start_iket_range("row-quant")
        quantize_tile_rows(out_tok, s_row_scale, words, tile)
        end_iket_range(range_token)

        range_token = start_iket_range("cta-barrier")
        txl.ptx.bar.sync(u32(0), u32(THREADS))
        end_iket_range(range_token)

        range_token = start_iket_range("row-scale-store")
        store_row_scales(scale_tok, s_row_scale, tile)
        end_iket_range(range_token)

        range_token = start_iket_range("column-quant")
        quantize_tile_columns(out_t, scale_t, s_tile, tile)
        end_iket_range(range_token)

    return fp8_quant_grad_tile


@functools.cache
def tile_kernel():
    return entry_function(build_tile_kernel().compile(TARGET))


@functools.cache
def count_sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def quantize_row_half(x, out_tok, scale_tok, words, role, x_offset, row_stride, tile):
    """
    Roles 0 / 1: quantize rows 0-63 / 64-127 of the tile (all 128 columns) from registers.
    """
    chunk, thread_row, row_start, col_start = tile.chunk, tile.thread_row, tile.row_start, tile.col_start
    M, N = tile.M, tile.N
    tile_col_idx, M_i64, N_i64 = tile.tile_col_idx, tile.M_i64, tile.N_i64
    half_row_start = row_start + 64 * role
    tile_base = txl.local_scalar("int64", init=x_offset + i64(half_row_start) * row_stride + i64(col_start))
    load_rows(x, words, tile_base, (half_row_start + 64 <= M) & (col_start + 128 <= N), half_row_start, 2, row_stride, tile)
    out_tile_base = txl.local_scalar("int64", init=i64(half_row_start) * N_i64 + i64(col_start))
    for row_group in range(2):
        row_idx = half_row_start + thread_row + 32 * row_group
        scale_bits, quant_words = quantize_row(words, row_group)
        with txl.if_(row_idx < M), txl.then_():
            store_quant_row(out_tok, out_tile_base, row_group, quant_words, tile)
            with txl.if_(chunk == 0), txl.then_():
                txl.ptx.st.global_.b32(txl.address_of(scale_tok[i64(tile_col_idx) * M_i64 + i64(row_idx)]), scale_bits)


def quantize_column_half(x, out_t, scale_t, s_tile, words, role, x_offset, row_stride, tile):
    """
    Roles 2 / 3: quantize columns 0-63 / 64-127 of the tile (all 128 rows) through a 16 KB
    XOR-swizzled shared tile; each lane loads one 8-column chunk of rows thread_row + 32 * i.
    """
    warp, lane, chunk, thread_row = tile.warp, tile.lane, tile.chunk, tile.thread_row
    row_start, col_start, M, N = tile.row_start, tile.col_start, tile.M, tile.N
    half_col_start = col_start + 64 * (role - 2)
    tile_base = txl.local_scalar("int64", init=x_offset + i64(row_start) * row_stride + i64(half_col_start))
    with txl.if_((row_start + 128 <= M) & (half_col_start + 64 <= N)):
        with txl.then_():
            for row_group in range(4):
                first_word = 4 * row_group
                txl.ptx.ld.global_.v4.b32(words[first_word], words[first_word + 1], words[first_word + 2], words[first_word + 3], txl.address_of(x[tile_base + i64(thread_row + 32 * row_group) * row_stride + i64(chunk * 8)]))
        with txl.else_():
            for row_group in range(4):
                first_word = 4 * row_group
                for j in range(4):
                    txl.assign(words[first_word + j], u32(0))
                with txl.if_((row_start + thread_row + 32 * row_group < M) & (half_col_start + chunk * 8 < N)), txl.then_():
                    txl.ptx.ld.global_.v4.b32(words[first_word], words[first_word + 1], words[first_word + 2], words[first_word + 3], txl.address_of(x[tile_base + i64(thread_row + 32 * row_group) * row_stride + i64(chunk * 8)]))
    for row_group in range(4):
        row = thread_row + 32 * row_group
        first_word = 4 * row_group
        txl.ptx.st.shared.v4.b32(s_tile.ptr_to([row * 32 + ((chunk ^ ((row >> 4) & 7)) << 2)]), words[first_word], words[first_word + 1], words[first_word + 2], words[first_word + 3])
    txl.ptx.bar.sync(u32(0), u32(THREADS))
    row_block = lane & 7
    pair_idx = warp * 4 + (lane >> 3)
    col_chunk = pair_idx >> 2
    col_offset = row_block * 16 * 32 + ((col_chunk ^ row_block) << 2) + (pair_idx & 3)
    col_words = txl.alloc_local([16], "uint32")
    for j in range(16):
        txl.ptx.ld.shared.b32(col_words[j], s_tile.ptr_to([col_offset + j * 32]))
    column_quant = quantize_column_pair(col_words)
    col_idx = half_col_start + 2 * pair_idx
    store_column_pair(out_t, scale_t, col_idx, row_block, column_quant, tile)


def build_split4_kernel(min_blocks_per_sm=6):
    """
    Small grids: four CTAs per 128x128 tile. Roles 0/1 quantize rows 0-63 / 64-127 (all 128
    columns) straight from registers; roles 2/3 quantize columns 0-63 / 64-127 (all 128 rows)
    through a 16 KB XOR-swizzled shared tile. Each tile is read twice (the second read hits L2).
    """

    @txl.kernel()
    def fp8_quant_grad_split4(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out_tok: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_tok: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_t: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, row_stride: txl.int64, x_offset: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=((N + 127) // 128) * ((M + 127) // 128) * 4, block=(THREADS // 32) * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=min_blocks_per_sm))
        cta_idx = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        pool = txl.smem_pool()
        s_tile = pool.alloc([BLOCK * 32], "uint32", align=128)
        num_col_tiles = (N + 127) >> 7
        role, tile_idx = cta_idx & 3, cta_idx >> 2
        tile_row_idx = txl.local_scalar("int32", init=txl.truncdiv(tile_idx, num_col_tiles))
        tile_col_idx = txl.local_scalar("int32", init=tile_idx - tile_row_idx * num_col_tiles)
        row_start, col_start = tile_row_idx * 128, tile_col_idx * 128
        M_i64, N_i64 = i64(M), i64(N)
        chunk = lane & 7
        thread_row = warp * 4 + (lane >> 3)
        words = txl.alloc_local([16], "uint32")
        tile = TileThread(warp, lane, chunk, thread_row, row_start, col_start, tile_row_idx, tile_col_idx, M, N, M_i64, N_i64)

        with txl.if_(role < 2):
            with txl.then_():
                quantize_row_half(x, out_tok, scale_tok, words, role, x_offset, row_stride, tile)
            with txl.else_():
                quantize_column_half(x, out_t, scale_t, s_tile, words, role, x_offset, row_stride, tile)

    return fp8_quant_grad_split4


@functools.cache
def split4_kernel(min_blocks_per_sm=6):
    return entry_function(build_split4_kernel(min_blocks_per_sm).compile(TARGET))


def load_f32(x, dtype, elem_idx):
    """
    One element of x widened to f32.
    """
    if dtype == "float32":
        value = txl.local_scalar("float32")
        txl.ptx.ld.global_.f32(value, txl.address_of(x[elem_idx]))
        return value
    raw_bits = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(raw_bits, txl.address_of(x[elem_idx]))
    if dtype == "bfloat16":
        return txl.reinterpret(txl.shift_left(txl.Cast("uint32", raw_bits), u32(16)), ty="float32")
    value = txl.local_scalar("float32")
    txl.ptx.cvt.f32.f16(value, raw_bits)
    return value


def quantize_f32(value, rcp):
    scaled = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(scaled, value, rcp)
    out = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(out, txl.float32(0.0), scaled)
    return out


@functools.cache
def generic_kernel(dtype):
    PITCH = BLOCK + 1

    @txl.kernel()
    def fp8_quant_grad_generic(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out_tok: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_tok: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_t: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, row_stride: txl.int64, col_stride: txl.int64, x_offset: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[(N + 127) // 128, (M + 127) // 128], block=(THREADS // 32) * 32))
        cta_idx = txl.cta_id()
        tile_col_idx, tile_row_idx = cta_idx[0], cta_idx[1]
        tidx = txl.thread_id()
        pool = txl.smem_pool()
        s_tile = pool.alloc([BLOCK * PITCH], "float32", align=16)
        s_col_amax = pool.alloc([2 * BLOCK], "float32", align=16)
        row_start, col_start = tile_row_idx * 128, tile_col_idx * 128

        with txl.serial(64) as step:
            elem = step * THREADS + tidx
            row, col = elem >> 7, elem & 127
            value = txl.local_scalar("float32", init=txl.float32(0.0))
            with txl.if_(row_start + row < M), txl.then_():
                with txl.if_(col_start + col < N), txl.then_():
                    txl.assign(value, load_f32(x, dtype, x_offset + i64(row_start + row) * row_stride + i64(col_start + col) * col_stride))
            txl.ptx.st.shared.f32(txl.address_of(s_tile[row * PITCH + col]), value)
        txl.ptx.bar.sync(u32(0), u32(THREADS))

        # rowwise: thread (row, half) covers 64 columns
        row, half = tidx >> 1, tidx & 1
        amax = txl.local_scalar("float32", init=txl.float32(0.0))
        elem_value = txl.local_scalar("float32")
        with txl.serial(64) as i:
            txl.ptx.ld.shared.f32(elem_value, txl.address_of(s_tile[row * PITCH + half * 64 + i]))
            txl.ptx.max.f32(amax, amax, txl.reinterpret(txl.bitwise_and(txl.reinterpret(elem_value, ty="uint32"), u32(0x7FFFFFFF)), ty="float32"))
        peer_amax = txl.local_scalar("uint32")
        txl.ptx.shfl_sync.bfly.b32(peer_amax, txl.reinterpret(amax, ty="uint32"), u32(1), u32(31), u32(0xFFFFFFFF))
        txl.ptx.max.f32(amax, amax, txl.reinterpret(peer_amax, ty="float32"))
        scale_bits, rcp_bits = e8m0_scale_bits(amax)
        rcp = txl.reinterpret(rcp_bits, ty="float32")
        with txl.if_(row_start + row < M), txl.then_():
            with txl.serial(64) as i:
                col_idx = col_start + half * 64 + i
                with txl.if_(col_idx < N), txl.then_():
                    txl.ptx.ld.shared.f32(elem_value, txl.address_of(s_tile[row * PITCH + half * 64 + i]))
                    quant = quantize_f32(elem_value, rcp)
                    txl.ptx.st.global_.b8(txl.address_of(out_tok[i64(row_start + row) * i64(N) + i64(col_idx)]), txl.Cast("uint8", quant))
            with txl.if_(half == 0), txl.then_():
                txl.ptx.st.global_.b32(txl.address_of(scale_tok[i64(tile_col_idx) * i64(M) + i64(row_start + row)]), scale_bits)

        # columnwise: thread (col, half) covers 64 rows; the two halves meet through s_col_amax
        col, half = tidx & 127, tidx >> 7
        col_amax = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.serial(64) as i:
            txl.ptx.ld.shared.f32(elem_value, txl.address_of(s_tile[(half * 64 + i) * PITCH + col]))
            txl.ptx.max.f32(col_amax, col_amax, txl.reinterpret(txl.bitwise_and(txl.reinterpret(elem_value, ty="uint32"), u32(0x7FFFFFFF)), ty="float32"))
        txl.ptx.st.shared.f32(txl.address_of(s_col_amax[half * 128 + col]), col_amax)
        txl.ptx.bar.sync(u32(0), u32(THREADS))
        txl.ptx.ld.shared.f32(elem_value, txl.address_of(s_col_amax[(1 - half) * 128 + col]))
        txl.ptx.max.f32(col_amax, col_amax, elem_value)
        col_scale_bits, col_rcp_bits = e8m0_scale_bits(col_amax)
        col_rcp = txl.reinterpret(col_rcp_bits, ty="float32")
        with txl.if_(col_start + col < N), txl.then_():
            with txl.serial(64) as i:
                row_idx = row_start + half * 64 + i
                with txl.if_(row_idx < M), txl.then_():
                    txl.ptx.ld.shared.f32(elem_value, txl.address_of(s_tile[(half * 64 + i) * PITCH + col]))
                    quant = quantize_f32(elem_value, col_rcp)
                    txl.ptx.st.global_.b8(txl.address_of(out_t[i64(col_start + col) * i64(M) + i64(row_idx)]), txl.Cast("uint8", quant))
            with txl.if_(half == 0), txl.then_():
                txl.ptx.st.global_.b32(txl.address_of(scale_t[i64(tile_row_idx) * i64(N) + i64(col_start + col)]), col_scale_bits)

    return entry_function(fp8_quant_grad_generic.compile(TARGET))


GENERIC_DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def fp8_quant_grad_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SM100 fp8_quant_grad; both scales are column-major.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    num_col_blocks, num_row_blocks = ceil_div(N, BLOCK), ceil_div(M, BLOCK)
    device = x.device
    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device)
    scale_tok = torch.empty((num_col_blocks, M), dtype=torch.float32, device=device).T
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device)
    scale_t = torch.empty((num_row_blocks, N), dtype=torch.float32, device=device).T
    if M == 0 or N == 0:
        return out_tok, scale_tok, out_t, scale_t

    row_stride, col_stride = x.stride()
    x_offset = x.storage_offset()
    x_flat = x.as_strided(((M - 1) * row_stride + (N - 1) * col_stride + 1 + x_offset,), (1,), 0)
    use_fast_path = x.dtype == torch.bfloat16 and col_stride == 1 and row_stride % 8 == 0 and x_offset % 8 == 0 and N % 8 == 0 and M % 16 == 0 and x_flat.data_ptr() % 16 == 0
    if use_fast_path:
        outputs = (out_tok.view(torch.uint8).view(-1), scale_tok.T.view(-1).view(torch.int32), out_t.view(torch.uint8).view(-1), scale_t.T.view(-1).view(torch.int32))
        num_sms = count_sms(device)
        if 2 * num_col_blocks * num_row_blocks < 3 * num_sms:
            split4_kernel()(x_flat, *outputs, M, N, row_stride, x_offset)
        else:
            tile_kernel()(x_flat, *outputs, M, N, row_stride, x_offset, 3 * num_sms)
    else:
        generic_kernel(GENERIC_DTYPES[x.dtype])(x_flat, out_tok.view(torch.uint8).view(-1), scale_tok.T.view(-1).view(torch.int32), out_t.view(torch.uint8).view(-1), scale_t.T.view(-1).view(torch.int32), M, N, row_stride, col_stride, x_offset)
    return out_tok, scale_tok, out_t, scale_t
