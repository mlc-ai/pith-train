"""
fp8_quant_grouped_sm100: rowwise 1x128 + per-channel 128x1 FP8 quantization from one read.

One CTA per 128x128 tile, the tile held in registers: 256 threads of 8 rows x 8 cols each, or, when
K % 16 == 0, 128 threads of 8 rows x 16 cols each (one ld.global.v8 per row). Scales are exact
powers of two, so x * (1/scale) is exact in bf16 and the bf16x2 -> e4m3x2 satfinite cvt rounds
once, matching the fp32 reference bit for bit.
"""

import functools
import math

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["fp8_quant_grouped_sm100"]


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
ROWS_PER_THREAD = 8  # 128 rows / (8 warps * 2 half-warps)


def u32(value):
    return txl.uint32(value)


def max_abs(first, second):
    """
    Per-half max(|first|, |second|) of two bf16x2 words, with the xor of the inputs' signs.
    """
    out = txl.local_scalar("uint32")
    txl.ptx.max.xorsign.abs.bf16x2(out, first, second)
    return out


def scale_bits_of_half(packed_amax, is_high_half):
    """
    f32 scale bits of the bf16 amax in one half of a bf16x2 word (sign garbage).
    """
    if is_high_half:
        amax_f32 = txl.reinterpret(txl.bitwise_and(packed_amax, u32(0x7FFF0000)), ty="float32")
    else:
        amax_f32 = txl.reinterpret(txl.shift_left(txl.bitwise_and(packed_amax, u32(0x7FFF)), u32(16)), ty="float32")
    clamped = txl.local_scalar("float32")
    txl.ptx.max.f32(clamped, amax_f32, txl.float32(1e-4))  # max_num: an all-NaN block clamps to 1e-4
    ratio = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(ratio, clamped, txl.float32(1.0 / 448.0))
    ratio_bits = txl.reinterpret(ratio, ty="uint32")
    return txl.local_scalar("uint32", init=txl.bitwise_and(ratio_bits + u32(0x7FFFFF), u32(0x7F800000)))


def rcp_bf16(scale_bits):
    """
    bf16 bits (low 16) of 1/scale; inf scale -> -inf as the SM90 emulation.
    """
    return txl.shift_right(u32(0x7F000000) - scale_bits, u32(16))


def quantize_word(word, rcp_pair):
    """
    Two e4m3 bytes (low 16 bits) of a bf16x2 word times a bf16x2 pair of reciprocal scales.
    """
    product = txl.local_scalar("uint32")
    txl.ptx.mul.rn.bf16x2(product, word, rcp_pair)
    fp8_pair = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(fp8_pair, product)
    return txl.cast(fp8_pair, "uint32")


@functools.cache
def compile_quant(K: int):
    return entry_function(make_quant(K).compile(TARGET))


def make_quant(K: int):
    num_col_blocks = (K + BLOCK - 1) // BLOCK
    vec_width = math.gcd(K, 8)  # elements per global access
    is_ragged = K % BLOCK != 0

    def x_offset(row0, col0, i, sub):
        return txl.Cast("int64", (row0 + 2 * i) * K + col0 + sub * vec_width)

    def load_tile(x, row0, col0):
        """
        Load the thread's 8 rows x 8 cols as 32 bf16x2 words; columns at or past K read as zero.
        """
        words = txl.alloc_local([ROWS_PER_THREAD * 4], "uint32")
        for k in range(ROWS_PER_THREAD * 4):
            txl.assign(words[k], u32(0))
        for i in range(ROWS_PER_THREAD):
            if vec_width == 8:
                def load_row():
                    txl.ptx.ld.global_.v4.b32(words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3], x.ptr_to([x_offset(row0, col0, i, 0)]))
                if is_ragged:
                    with txl.if_(col0 < K), txl.then_():
                        load_row()
                else:
                    load_row()
            else:
                for sub in range(8 // vec_width):
                    with txl.if_(col0 + sub * vec_width < K), txl.then_():
                        if vec_width == 4:
                            word_idx = 4 * i + 2 * sub
                            txl.ptx.ld.global_.v2.b32(words[word_idx], words[word_idx + 1], x.ptr_to([x_offset(row0, col0, i, sub)]))
                        elif vec_width == 2:
                            txl.ptx.ld.global_.b32(words[4 * i + sub], x.ptr_to([x_offset(row0, col0, i, sub)]))
                        else:
                            half_bits = txl.local_scalar("uint16")
                            txl.ptx.ld.global_.b16(half_bits, x.ptr_to([x_offset(row0, col0, i, sub)]))
                            word_idx = 4 * i + sub // 2
                            if sub % 2 == 0:
                                txl.assign(words[word_idx], txl.cast(half_bits, "uint32"))
                            else:
                                txl.assign(words[word_idx], txl.bitwise_or(words[word_idx], txl.shift_left(txl.cast(half_bits, "uint32"), u32(16))))
        return words

    def reduce_warp_col_amax(words, s_col_amax, warp, col_slot, row_slot):
        """
        Column amax over the thread's own rows and its half-warp partner, stored per warp to smem.
        """
        col_amax = txl.alloc_local([4], "uint32")
        for j in range(4):
            txl.assign(col_amax[j], words[j])
            for i in range(1, ROWS_PER_THREAD):
                txl.assign(col_amax[j], max_abs(col_amax[j], words[4 * i + j]))
            peer = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.bfly.b32(peer, col_amax[j], u32(16), u32(31), u32(0xFFFFFFFF))
            txl.assign(col_amax[j], max_abs(col_amax[j], peer))
        with txl.if_(row_slot == 0), txl.then_():
            txl.ptx["st.shared.v4.b32"](s_col_amax.ptr_to([warp * 64 + col_slot * 4]), col_amax[0], col_amax[1], col_amax[2], col_amax[3])
        return col_amax

    def compute_row_scales(words, scale_rowwise, row0, col_block, col_slot):
        """
        Row scales, stored to global; returns the bf16x2 reciprocal of each of the thread's rows.
        """
        row_rcp = txl.alloc_local([ROWS_PER_THREAD], "uint32")
        for i in range(ROWS_PER_THREAD):
            row_amax = max_abs(max_abs(words[4 * i], words[4 * i + 1]), max_abs(words[4 * i + 2], words[4 * i + 3]))
            for delta in (1, 2, 4, 8):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(peer, row_amax, u32(delta), u32(31), u32(0xFFFFFFFF))
                txl.assign(row_amax, max_abs(row_amax, peer))
            # fold the two bf16 halves into the low half
            row_amax_folded = max_abs(row_amax, txl.shift_right(row_amax, u32(16)))
            row_scale_bits = scale_bits_of_half(row_amax_folded, False)
            row_rcp_bf16 = rcp_bf16(row_scale_bits)
            txl.assign(row_rcp[i], txl.bitwise_or(row_rcp_bf16, txl.shift_left(row_rcp_bf16, u32(16))))
            with txl.if_(col_slot == 0), txl.then_():
                txl.ptx.st.global_.b32(scale_rowwise.ptr_to([txl.Cast("int64", (row0 + 2 * i) * num_col_blocks + col_block)]), row_scale_bits)
        return row_rcp

    def compute_col_scales(col_amax, s_col_amax, scale_colwise, tidx, row_block, col0, col_slot):
        """
        Column amax across the 8 warps into col_amax, column scales stored to global by the first
        16 threads; returns the bf16x2 reciprocals of the thread's 8 columns.
        """
        for j in range(4):
            txl.assign(col_amax[j], u32(0))
        for warp_idx in range(8):
            partial = txl.alloc_local([4], "uint32")
            txl.ptx["ld.shared.v4.b32"](partial[0], partial[1], partial[2], partial[3], s_col_amax.ptr_to([warp_idx * 64 + col_slot * 4]))
            for j in range(4):
                txl.assign(col_amax[j], max_abs(col_amax[j], partial[j]))
        col_rcp = txl.alloc_local([4], "uint32")
        col_scale_bits = txl.alloc_local([8], "uint32")
        for j in range(4):
            lo_bits = scale_bits_of_half(col_amax[j], False)
            hi_bits = scale_bits_of_half(col_amax[j], True)
            txl.assign(col_scale_bits[2 * j], lo_bits)
            txl.assign(col_scale_bits[2 * j + 1], hi_bits)
            txl.assign(col_rcp[j], txl.bitwise_or(rcp_bf16(lo_bits), txl.shift_left(rcp_bf16(hi_bits), u32(16))))
        with txl.if_(tidx < 16), txl.then_():
            scale_start = row_block * K + col0
            for elem in range(8):
                if is_ragged or K % 8 != 0:
                    with txl.if_(col0 + elem < K), txl.then_():
                        txl.ptx.st.global_.b32(scale_colwise.ptr_to([txl.Cast("int64", scale_start + elem)]), col_scale_bits[elem])
                else:
                    txl.ptx.st.global_.b32(scale_colwise.ptr_to([txl.Cast("int64", scale_start + elem)]), col_scale_bits[elem])
        return col_rcp

    def quantize_and_store(words, row_rcp, col_rcp, out_rowwise, out_colwise, row0, col0):
        """
        Quantize the tile with the row scales into out_rowwise, then with the column scales
        into out_colwise.
        """
        for scale_axis, out in ((0, out_rowwise), (1, out_colwise)):
            for i in range(ROWS_PER_THREAD):
                fp8_pairs = [quantize_word(words[4 * i + j], row_rcp[i] if scale_axis == 0 else col_rcp[j]) for j in range(4)]
                out_lo = txl.local_scalar("uint32", init=txl.bitwise_or(fp8_pairs[0], txl.shift_left(fp8_pairs[1], u32(16))))
                out_hi = txl.local_scalar("uint32", init=txl.bitwise_or(fp8_pairs[2], txl.shift_left(fp8_pairs[3], u32(16))))
                if vec_width == 8:
                    def store_row():
                        txl.ptx.st.global_.v2.b32(out.ptr_to([x_offset(row0, col0, i, 0)]), out_lo, out_hi)
                    if is_ragged:
                        with txl.if_(col0 < K), txl.then_():
                            store_row()
                    else:
                        store_row()
                else:
                    for elem in range(8):
                        out_word = out_lo if elem < 4 else out_hi
                        out_byte = txl.cast(txl.bitwise_and(txl.shift_right(out_word, u32(8 * (elem % 4))), u32(0xFF)), "uint8")
                        with txl.if_(col0 + elem < K), txl.then_():
                            txl.ptx.st.global_.b8(out.ptr_to([txl.Cast("int64", (row0 + 2 * i) * K + col0 + elem)]), out_byte)

    @txl.kernel()
    def quant(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out_rowwise: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_rowwise: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), out_colwise: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_colwise: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), num_row_blocks: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_row_blocks * num_col_blocks, block=(THREADS // 32) * 32))
        bx, tidx = txl.cta_id(), txl.thread_id()
        row_block, col_block = bx // num_col_blocks, bx % num_col_blocks
        warp, lane = tidx >> 5, tidx & 31
        col_slot, row_slot = lane & 15, lane >> 4
        col0 = txl.local_scalar("int32", init=col_block * BLOCK + col_slot * 8)
        row0 = txl.local_scalar("int32", init=row_block * BLOCK + warp * 16 + row_slot)
        smem = txl.smem_pool()
        s_col_amax = smem.alloc([8 * 64], "uint32")  # [warp][col_slot*4 + j]

        words = load_tile(x, row0, col0)

        col_amax = reduce_warp_col_amax(words, s_col_amax, warp, col_slot, row_slot)

        # before the barrier, so the row reduction overlaps its wait
        row_rcp = compute_row_scales(words, scale_rowwise, row0, col_block, col_slot)

        txl.ptx.bar.sync(u32(0), u32(THREADS))

        col_rcp = compute_col_scales(col_amax, s_col_amax, scale_colwise, tidx, row_block, col0, col_slot)

        quantize_and_store(words, row_rcp, col_rcp, out_rowwise, out_colwise, row0, col0)

    return quant


# 256-bit variant (K % 16 == 0): a thread holds 8 rows x 16 cols (one ld.global.v8 per row).
# Column scales are reduced by 64 threads (one bf16x2 word each) and broadcast through smem.


PREFETCH_DIST = 148 * 2  # L2 prefetch distance in tiles: about 2 resident CTAs per SM ahead
WARPS_V8 = 4  # 128 threads; each holds 8 rows x 16 cols (more loads in flight per SM)
ROWS_PER_THREAD_V8 = 128 // WARPS_V8 // 4
THREADS_V8 = WARPS_V8 * 32


@functools.cache
def compile_quant_v8(K: int):
    return entry_function(make_quant_v8(K).compile(TARGET))


def make_quant_v8(K: int):
    num_col_blocks = (K + BLOCK - 1) // BLOCK
    is_ragged = K % BLOCK != 0

    def x_offset(row0, col0, i):
        return txl.Cast("int64", (row0 + 4 * i) * K + col0)

    def guard_cols(col0, body):
        if is_ragged:
            with txl.if_(col0 < K), txl.then_():
                body()
        else:
            body()

    def load_tile(x, row0, col0):
        """
        Load the thread's 8 rows x 16 cols as 64 bf16x2 words; columns at or past K read as zero.
        """
        words = txl.alloc_local([ROWS_PER_THREAD_V8 * 8], "uint32")
        if is_ragged:
            for k in range(ROWS_PER_THREAD_V8 * 8):
                txl.assign(words[k], u32(0))
        for i in range(ROWS_PER_THREAD_V8):
            guard_cols(col0, lambda i=i: txl.ptx.ld.global_.v8.b32(*[words[8 * i + j] for j in range(8)], x.ptr_to([x_offset(row0, col0, i)])))
        return words

    def prefetch_later_tile(x, bx, tidx, num_row_blocks, prefetch_dist):
        """
        L2 prefetch of the tile a CTA about one resident wave later will load: one row per thread.
        """
        prefetch_tile = bx + prefetch_dist
        with txl.if_((prefetch_dist > 0) & (prefetch_tile < num_row_blocks * num_col_blocks)), txl.then_():
            prefetch_row_block, prefetch_col_block = prefetch_tile // num_col_blocks, prefetch_tile % num_col_blocks
            if is_ragged:
                num_bytes = txl.Cast("uint32", txl.min(K - prefetch_col_block * BLOCK, BLOCK) * 2)
            else:
                num_bytes = u32(BLOCK * 2)
            txl.ptx["cp.async.bulk.prefetch.L2.global"](x.ptr_to([txl.Cast("int64", (prefetch_row_block * BLOCK + tidx) * K + prefetch_col_block * BLOCK)]), num_bytes)

    def reduce_warp_col_amax(words, s_col_partial, warp, col_slot, row_slot):
        """
        Column amax over the thread's own rows, then the 4 row lanes (xor 8, 16), stored per warp
        to smem.
        """
        col_amax = txl.alloc_local([8], "uint32")
        for j in range(8):
            txl.assign(col_amax[j], words[j])
            for i in range(1, ROWS_PER_THREAD_V8):
                txl.assign(col_amax[j], max_abs(col_amax[j], words[8 * i + j]))
            for delta in (8, 16):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(peer, col_amax[j], u32(delta), u32(31), u32(0xFFFFFFFF))
                txl.assign(col_amax[j], max_abs(col_amax[j], peer))
        with txl.if_(row_slot == 0), txl.then_():
            txl.ptx["st.shared.v4.b32"](s_col_partial.ptr_to([warp * 64 + col_slot * 8]), col_amax[0], col_amax[1], col_amax[2], col_amax[3])
            txl.ptx["st.shared.v4.b32"](s_col_partial.ptr_to([warp * 64 + col_slot * 8 + 4]), col_amax[4], col_amax[5], col_amax[6], col_amax[7])

    def compute_row_scales(words, scale_rowwise, row0, col_block, col_slot):
        """
        Row scales, stored to global; returns the bf16x2 reciprocal of each of the thread's rows.
        """
        row_rcp = txl.alloc_local([ROWS_PER_THREAD_V8], "uint32")
        for i in range(ROWS_PER_THREAD_V8):
            pair_amax = [max_abs(words[8 * i + 2 * j], words[8 * i + 2 * j + 1]) for j in range(4)]
            row_amax = max_abs(max_abs(pair_amax[0], pair_amax[1]), max_abs(pair_amax[2], pair_amax[3]))
            for delta in (1, 2, 4):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(peer, row_amax, u32(delta), u32(31), u32(0xFFFFFFFF))
                txl.assign(row_amax, max_abs(row_amax, peer))
            row_amax_folded = max_abs(row_amax, txl.shift_right(row_amax, u32(16)))
            row_scale_bits = scale_bits_of_half(row_amax_folded, False)
            row_rcp_bf16 = rcp_bf16(row_scale_bits)
            txl.assign(row_rcp[i], txl.bitwise_or(row_rcp_bf16, txl.shift_left(row_rcp_bf16, u32(16))))
            with txl.if_(col_slot == 0), txl.then_():
                txl.ptx.st.global_.b32(scale_rowwise.ptr_to([txl.Cast("int64", (row0 + 4 * i) * num_col_blocks + col_block)]), row_scale_bits)
        return row_rcp

    def compute_col_scales(s_col_partial, s_col_rcp, scale_colwise, tidx, row_block, col_block):
        """
        The first 64 threads reduce one bf16x2 column word each across the warps, store its two
        column scales to global and its reciprocals to smem.
        """
        with txl.if_(tidx < 64), txl.then_():
            col_amax = txl.local_scalar("uint32", init=u32(0))
            for warp_idx in range(WARPS_V8):
                partial = txl.local_scalar("uint32")
                txl.ptx["ld.shared.b32"](partial, s_col_partial.ptr_to([warp_idx * 64 + tidx]))
                txl.assign(col_amax, max_abs(col_amax, partial))
            lo_bits = scale_bits_of_half(col_amax, False)
            hi_bits = scale_bits_of_half(col_amax, True)
            txl.ptx["st.shared.b32"](s_col_rcp.ptr_to([tidx]), txl.bitwise_or(rcp_bf16(lo_bits), txl.shift_left(rcp_bf16(hi_bits), u32(16))))
            col_idx = col_block * BLOCK + tidx * 2
            def store_col_scales():
                txl.ptx.st.global_.v2.b32(scale_colwise.ptr_to([txl.Cast("int64", row_block * K + col_idx)]), lo_bits, hi_bits)
            if is_ragged:
                with txl.if_(col_idx < K), txl.then_():
                    store_col_scales()
            else:
                store_col_scales()

    def load_col_rcp(s_col_rcp, col_slot):
        """
        The bf16x2 reciprocals of the thread's 16 columns, from smem.
        """
        col_rcp = txl.alloc_local([8], "uint32")
        txl.ptx["ld.shared.v4.b32"](col_rcp[0], col_rcp[1], col_rcp[2], col_rcp[3], s_col_rcp.ptr_to([col_slot * 8]))
        txl.ptx["ld.shared.v4.b32"](col_rcp[4], col_rcp[5], col_rcp[6], col_rcp[7], s_col_rcp.ptr_to([col_slot * 8 + 4]))
        return col_rcp

    def quantize_and_store(words, row_rcp, col_rcp, out_rowwise, out_colwise, row0, col0):
        """
        Quantize the tile with the row scales into out_rowwise, then with the column scales
        into out_colwise.
        """
        for scale_axis, out in ((0, out_rowwise), (1, out_colwise)):
            for i in range(ROWS_PER_THREAD_V8):
                fp8_pairs = [quantize_word(words[8 * i + j], row_rcp[i] if scale_axis == 0 else col_rcp[j]) for j in range(8)]
                out_words = [txl.local_scalar("uint32", init=txl.bitwise_or(fp8_pairs[2 * k], txl.shift_left(fp8_pairs[2 * k + 1], u32(16)))) for k in range(4)]
                guard_cols(col0, lambda i=i, out_words=out_words, out=out: txl.ptx.st.global_.v4.b32(out.ptr_to([x_offset(row0, col0, i)]), out_words[0], out_words[1], out_words[2], out_words[3]))

    @txl.kernel()
    def quant(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out_rowwise: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_rowwise: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), out_colwise: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), scale_colwise: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), num_row_blocks: txl.int32, prefetch_dist: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_row_blocks * num_col_blocks, block=WARPS_V8 * 32))
        bx, tidx = txl.cta_id(), txl.thread_id()
        row_block, col_block = bx // num_col_blocks, bx % num_col_blocks
        warp, lane = tidx >> 5, tidx & 31
        col_slot, row_slot = lane & 7, lane >> 3
        col0 = txl.local_scalar("int32", init=col_block * BLOCK + col_slot * 16)
        row0 = txl.local_scalar("int32", init=row_block * BLOCK + warp * (4 * ROWS_PER_THREAD_V8) + row_slot)
        smem = txl.smem_pool()
        s_col_partial = smem.alloc([WARPS_V8 * 64], "uint32")  # [warp][word]
        s_col_rcp = smem.alloc([64], "uint32")

        words = load_tile(x, row0, col0)

        prefetch_later_tile(x, bx, tidx, num_row_blocks, prefetch_dist)

        reduce_warp_col_amax(words, s_col_partial, warp, col_slot, row_slot)

        row_rcp = compute_row_scales(words, scale_rowwise, row0, col_block, col_slot)

        txl.ptx.bar.sync(u32(0), u32(THREADS_V8))
        compute_col_scales(s_col_partial, s_col_rcp, scale_colwise, tidx, row_block, col_block)
        txl.ptx.bar.sync(u32(0), u32(THREADS_V8))
        col_rcp = load_col_rcp(s_col_rcp, col_slot)

        quantize_and_store(words, row_rcp, col_rcp, out_rowwise, out_colwise, row0, col0)

    return quant


def fp8_quant_grouped_sm100(input, grouped_mm_offs):
    """
    SM100 fp8_quant_grouped: the per-channel pair is (M, K) with scale (M / 128, K), all row-major.
    """
    M, K = input.shape
    num_col_blocks = (K + BLOCK - 1) // BLOCK
    device = input.device
    x_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device)
    x_scale = torch.empty((M, num_col_blocks), dtype=torch.float32, device=device)
    x_ch_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device)
    x_ch_scale = torch.empty((M // BLOCK, K), dtype=torch.float32, device=device)
    if M == 0 or K == 0:
        return x_fp8, x_scale, x_ch_fp8, x_ch_scale
    fn = compile_quant_v8(K) if K % 16 == 0 else compile_quant(K)
    prefetch_args = (PREFETCH_DIST,) if K % 16 == 0 else ()
    fn(input.view(-1), x_fp8.view(torch.uint8).view(-1), x_scale.view(-1), x_ch_fp8.view(torch.uint8).view(-1), x_ch_scale.view(-1), M // BLOCK, *prefetch_args)
    return x_fp8, x_scale, x_ch_fp8, x_ch_scale
