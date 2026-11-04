"""
SM100 fp8_weight_quant in TIRx-lite: 128x128 blockwise FP8 of weights and their transposes,
and the FP8 parameter all-gather kernels (shard amax / cast, E8M0 scales, packed transpose),
including one-launch multi variants of the shard amax and scales.

Every kernel is a @txl.kernel compiled once per process (functools.cache, keyed only on
compile-time constants) and launched through tvm-ffi on torch's current stream. Kernel parameters
R and C are the numbers of 128-row and 128-column tiles of one group; sxg, sxn and sxk are the
element strides of x over groups, rows and columns.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["fp8_quant_weight_sm100", "fp8_quant_weight_grouped_sm100", "fp8_shard_amax_sm100", "fp8_shard_cast_sm100", "fp8_scales_sm100", "fp8_pack_transpose_sm100", "fp8_shard_amax_multi_sm100", "fp8_scales_multi_sm100"]

BLOCK = 128
THREADS = 256  # threads per CTA
THREADS_Q128 = 128  # small-grid quant (q128): four resident 4-warp CTAs instead of three 8-warp CTAs
FULL_MASK = 0xFFFFFFFF
RCP_448 = 1.0 / 448.0
# Streaming fp32 shard loads: no L1 allocation + 256B L2 prefetch, faster for pure reads.
LDG_V4 = "ld.global.nc.L1::no_allocate.L2::256B.v4.b32"
LDG_B32 = "ld.global.nc.L1::no_allocate.L2::256B.b32"


# Every kernel is launched with programmatic stream serialization (pdl_entry()) and starts with pdl().


def tensor_nd(dtype, rank):
    """
    A rank-dimensional global tensor parameter, every extent its own symbolic int64.
    """
    return txl.Tensor(tuple(txl.int64() for _ in range(rank)), dtype, data_alignment=1)


def cdiv(numerator, denominator):
    return (numerator + denominator - 1) // denominator


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def compile_kernel(kernel):
    """
    Compile once; return the module's entry function itself (calling the module looks the entry up
    on every call, which costs host time). tvm-ffi runs it on torch's current stream, which it
    takes from the torch tensor arguments.
    """
    module = kernel.compile(TARGET)
    EXECUTABLES.append(module)
    return module[kernel.name]


def pdl_entry(grid, threads=THREADS):
    """
    Device entry of every kernel here: threads per CTA, programmatic stream serialization.
    """
    txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid, block=threads, programmatic_stream_serialization=True))


def pdl():
    """
    Programmatic dependent launch: let the next kernel in the stream get scheduled now, and
    wait for the previous one (completion + memory flush) before touching global memory.
    """
    txl.ptx.griddepcontrol.wait()
    txl.ptx.griddepcontrol.launch_dependents()


def as_i64(value):
    return txl.Cast("int64", value)


def e8m0_scale_bits(amax):
    """
    (scale_bits, rcp_bits) of the recipe's E8M0 scale (SM100, native E8M0).
    """
    amax_clamped = txl.local_scalar("float32")
    txl.ptx.max.f32(amax_clamped, amax, txl.float32(1e-4))
    scale_f32 = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(scale_f32, amax_clamped, txl.float32(RCP_448))
    scale_pair = txl.local_scalar("uint16")
    txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(scale_pair, txl.float32(0.0), scale_f32)
    # The E8M0 byte becomes an fp32 exponent, at least 2^-126; the reciprocal negates the exponent.
    scale_bits = txl.local_scalar("uint32")
    txl.ptx.max.u32(scale_bits, (txl.Cast("uint32", scale_pair) & txl.uint32(0xFF)) << txl.uint32(23), txl.uint32(1 << 23))
    rcp_bits = txl.local_scalar("uint32", init=txl.uint32(254 << 23) - scale_bits)
    return scale_bits, rcp_bits


def cast_e4m3x2(hi, lo, rcp):
    """
    Two f32 values times rcp -> two e4m3 bytes in the low 16 bits of a u32.
    """
    hi_scaled, lo_scaled = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(hi_scaled, hi, rcp)
    txl.ptx.mul.rn.f32(lo_scaled, lo, rcp)
    packed = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(packed, hi_scaled, lo_scaled)
    return txl.Cast("uint32", packed)


def bf16_word_to_fp8x2(word, rcp):
    lo, hi = txl.reinterpret(word << txl.uint32(16), ty="float32"), txl.reinterpret(word & txl.uint32(0xFFFF0000), ty="float32")
    return cast_e4m3x2(hi, lo, rcp)


def cta_max_u32(value, s_reduce, warp, lane, threads):
    """
    Max of a u32 over a CTA of the given number of threads; every thread gets the result.
    """
    warp_max = txl.local_scalar("uint32")
    txl.ptx.redux_sync.max.u32(warp_max, value, txl.uint32(FULL_MASK))
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.b32(s_reduce.ptr_to([warp]), warp_max)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(threads))
    out = txl.local_scalar("uint32", init=txl.uint32(0))
    for i in range(threads // 32):
        partial = txl.local_scalar("uint32")
        txl.ptx.ld.shared.b32(partial, s_reduce.ptr_to([i]))
        txl.ptx.max.u32(out, out, partial)
    return out


def permute_bytes(first, second, selector):
    out = txl.local_scalar("uint32")
    txl.ptx.prmt.b32(out, first, second, txl.uint32(selector))
    return out


def transpose4x4_bytes(row0, row1, row2, row3):
    """
    4x4 byte transpose: four rows of 4 bytes each -> four columns.
    """
    mix01_lo = permute_bytes(row0, row1, 0x5140)
    mix01_hi = permute_bytes(row0, row1, 0x7362)
    mix23_lo = permute_bytes(row2, row3, 0x5140)
    mix23_hi = permute_bytes(row2, row3, 0x7362)
    return (permute_bytes(mix01_lo, mix23_lo, 0x5410), permute_bytes(mix01_lo, mix23_lo, 0x7632), permute_bytes(mix01_hi, mix23_hi, 0x5410), permute_bytes(mix01_hi, mix23_hi, 0x7632))


def store_block_transposed(block, s_tile, k_chunk, row_group):
    """
    block[row][half]: 8 rows (n) x 8 bytes (k) as 2 words per row. Writes the 8x8 block transposed
    into the [128 k][128 n] byte tile s_tile (u32 words, 8-byte slots XOR-swizzled by k >> 3).
    """
    columns = [[None, None] for _ in range(8)]  # columns[k][half of n]
    for half in range(2):
        for row_half in range(2):
            outs = transpose4x4_bytes(*[block[4 * row_half + i][half] for i in range(4)])
            for j in range(4):
                columns[4 * half + j][row_half] = outs[j]
    slot = row_group ^ k_chunk
    for k in range(8):
        row = k_chunk * 8 + k
        txl.ptx.st.shared.v2.b32(s_tile.ptr_to([row * 32 + slot * 2]), columns[k][0], columns[k][1])


def store_block_transposed_q128(block, s_tile, k_chunk, row_group):
    """
    store_block_transposed for the 16 rows (n) x 8 bytes (k) block of a q128 thread: the same
    swizzled shared tile, so the flush reads it unchanged.
    """
    columns = [[None, None, None, None] for _ in range(8)]
    for half in range(2):
        for row_half in range(4):
            outs = transpose4x4_bytes(*[block[4 * row_half + i][half] for i in range(4)])
            for j in range(4):
                columns[4 * half + j][row_half] = outs[j]
    for k in range(8):
        row = k_chunk * 8 + k
        slot0 = (row_group * 2) ^ k_chunk
        slot1 = (row_group * 2 + 1) ^ k_chunk
        txl.ptx.st.shared.v2.b32(s_tile.ptr_to([row * 32 + slot0 * 2]), columns[k][0], columns[k][1])
        txl.ptx.st.shared.v2.b32(s_tile.ptr_to([row * 32 + slot1 * 2]), columns[k][2], columns[k][3])


def flush_transposed_tile(s_tile, tid, out_t_addr, k_start, n_start, K, N, threads):
    """
    Write the transposed tile from s_tile: tile row k (0..127) -> out_t row k_start + k, columns
    n_start .. n_start + 127 in 16-byte chunks (requires N % 16 == 0), by a CTA of threads threads.
    """
    chunk = tid & 7
    rows_per_pass = threads // 8
    for pass_idx in range(BLOCK // rows_per_pass):
        row = pass_idx * rows_per_pass + (tid >> 3)
        swizzle = row >> 3
        slot = ((chunk * 2) ^ swizzle) & 14
        words = txl.alloc_local([4], "uint32")
        txl.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], s_tile.ptr_to([row * 32 + slot * 2]))
        # An odd swizzle lands the chunk's two 8-byte halves swapped within the 16-byte slot pair.
        is_odd = (swizzle & 1) == 1
        k_idx = k_start + row
        n_idx = n_start + chunk * 16
        store_ok = txl.And(k_idx < K, n_idx < N)
        txl.ptx.st.global_.v4.b32(out_t_addr(as_i64(k_idx) * as_i64(N) + as_i64(n_idx)), txl.Select(is_odd, words[2], words[0]), txl.Select(is_odd, words[3], words[1]), txl.Select(is_odd, words[0], words[2]), txl.Select(is_odd, words[1], words[3]), pred=store_ok)


def tile_coords(cta, R, C):
    """
    (group_idx, tile_row, tile_col) of a CTA over the G x R x C tiles, tile columns fastest.
    """
    tile_col = txl.local_scalar("int32", init=cta % C)
    group_tile = txl.local_scalar("int32", init=cta // C)
    tile_row = txl.local_scalar("int32", init=group_tile % R)
    group_idx = txl.local_scalar("int32", init=group_tile // R)
    return group_idx, tile_row, tile_col


def store_tile_scale(tid, sc, sct, scale_bits, group_idx, tile_row, tile_col, R, C):
    """
    Thread 0 writes the tile's scale into the (G, R, C) scale and the (G, C, R) transposed scale.
    """
    with txl.if_(tid == 0), txl.then_():
        scale_f32 = txl.reinterpret(scale_bits, ty="float32")
        txl.ptx.st.global_.f32(sc.ptr_to([(as_i64(group_idx) * as_i64(R) + as_i64(tile_row)) * as_i64(C) + as_i64(tile_col)]), scale_f32)
        txl.ptx.st.global_.f32(sct.ptr_to([(as_i64(group_idx) * as_i64(C) + as_i64(tile_col)) * as_i64(R) + as_i64(tile_row)]), scale_f32)


def load_bf16_block(x, x_offset, sxn, num_rows, tile_row, tile_col, row0, k_ok, N, K):
    """
    The thread's num_rows x 8 bf16 block, 4 words per row; elements outside N x K read as zero.
    """
    words = txl.alloc_local([4 * num_rows], "uint32")

    def load_row(row):
        txl.ptx.ld.global_.nc.v4.b32(words[4 * row], words[4 * row + 1], words[4 * row + 2], words[4 * row + 3], x.ptr_to([x_offset + sxn * row]))

    with txl.if_(txl.And(tile_col * BLOCK + BLOCK <= K, tile_row * BLOCK + BLOCK <= N)):
        with txl.then_():
            for row in range(num_rows):
                load_row(row)
        with txl.else_():
            for row in range(num_rows):
                for i in range(4):
                    txl.ptx.mov.b32(words[4 * row + i], txl.uint32(0))
                with txl.if_(txl.And(k_ok, row0 + row < N)), txl.then_():
                    load_row(row)
    return words


def reduce_bf16_scale(words, num_rows, s_reduce, warp, lane, threads):
    """
    (scale_bits, rcp) of the tile from every thread's bf16 block.
    """
    # |bf16| bit patterns order like their values, so a u16 max of the masked halves is the amax.
    acc_amax = txl.local_scalar("uint32", init=txl.uint32(0))
    for i in range(4 * num_rows):
        txl.ptx.max.u16x2(acc_amax, acc_amax, words[i] & txl.uint32(0x7FFF7FFF))
    txl.ptx.max.u32(acc_amax, acc_amax & txl.uint32(0xFFFF), acc_amax >> txl.uint32(16))
    amax_bits = cta_max_u32(acc_amax, s_reduce, warp, lane, threads)
    scale_bits, rcp_bits = e8m0_scale_bits(txl.reinterpret(amax_bits << txl.uint32(16), ty="float32"))
    rcp = txl.reinterpret(rcp_bits, ty="float32")
    return scale_bits, rcp


def quantize_store_rows(out, words, rcp, num_rows, group_idx, row0, k_idx, k_ok, N, K):
    """
    Cast the block to e4m3 and store it row-major; returns it as (word0, word1) per row.
    """
    out_offset = txl.local_scalar("int64", init=(as_i64(group_idx) * as_i64(N) + as_i64(row0)) * as_i64(K) + as_i64(k_idx))
    block = []
    for row in range(num_rows):
        pairs = [bf16_word_to_fp8x2(words[4 * row + i], rcp) for i in range(4)]
        word0 = txl.local_scalar("uint32", init=pairs[0] | (pairs[1] << txl.uint32(16)))
        word1 = txl.local_scalar("uint32", init=pairs[2] | (pairs[3] << txl.uint32(16)))
        block.append((word0, word1))
        txl.ptx.st.global_.v2.b32(out.ptr_to([out_offset + as_i64(K) * row]), word0, word1, pred=txl.And(k_ok, row0 + row < N))
    return block


@functools.cache
def compile_quant_vec():
    """
    bf16, unit column stride, K % 8 == 0, N % 16 == 0, 16-byte aligned rows.
    """

    @txl.kernel()
    def fp8_quant_vec(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), sc: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), sct: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, sxg: txl.int64, sxn: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        group_idx, tile_row, tile_col = tile_coords(txl.cta_id(), R, C)
        smem = txl.smem_pool()
        s_tile = smem.alloc([BLOCK * BLOCK // 4], "uint32")
        s_reduce = smem.alloc([THREADS // 32], "uint32")

        k_chunk, row_group = tid & 15, tid >> 4
        row0 = tile_row * BLOCK + row_group * 8
        k_idx = tile_col * BLOCK + k_chunk * 8
        k_ok = k_idx < K
        x_offset = txl.local_scalar("int64", init=as_i64(group_idx) * sxg + as_i64(row0) * sxn + as_i64(k_idx))
        words = load_bf16_block(x, x_offset, sxn, 8, tile_row, tile_col, row0, k_ok, N, K)
        scale_bits, rcp = reduce_bf16_scale(words, 8, s_reduce, warp, lane, THREADS)

        block = quantize_store_rows(out, words, rcp, 8, group_idx, row0, k_idx, k_ok, N, K)
        store_block_transposed(block, s_tile, k_chunk, row_group)
        store_tile_scale(tid, sc, sct, scale_bits, group_idx, tile_row, tile_col, R, C)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS))
        out_t_offset = txl.local_scalar("int64", init=as_i64(group_idx) * as_i64(K) * as_i64(N))
        flush_transposed_tile(s_tile, tid, lambda offset: out_t.ptr_to([out_t_offset + offset]), tile_col * BLOCK, tile_row * BLOCK, K, N, THREADS)

    return compile_kernel(fp8_quant_vec)


@functools.cache
def compile_quant_vec_q128():
    """
    compile_quant_vec for moderate grids: four-warp CTAs, each thread owning 16x8 bf16 values
    instead of 8x8. The larger per-thread register tile raises resident CTAs from three to four,
    reducing wave quantization on the dense rows while keeping 16-byte global loads and the same
    transpose layout.
    """

    @txl.kernel()
    def fp8_quant_vec_q128(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), sc: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), sct: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, sxg: txl.int64, sxn: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles, THREADS_Q128)
        pdl()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        group_idx, tile_row, tile_col = tile_coords(txl.cta_id(), R, C)
        smem = txl.smem_pool()
        s_tile = smem.alloc([BLOCK * BLOCK // 4], "uint32")
        s_reduce = smem.alloc([THREADS_Q128 // 32], "uint32")

        k_chunk, row_group = tid & 15, tid >> 4
        row0 = tile_row * BLOCK + row_group * 16
        k_idx = tile_col * BLOCK + k_chunk * 8
        k_ok = k_idx < K
        x_offset = txl.local_scalar("int64", init=as_i64(group_idx) * sxg + as_i64(row0) * sxn + as_i64(k_idx))
        words = load_bf16_block(x, x_offset, sxn, 16, tile_row, tile_col, row0, k_ok, N, K)
        scale_bits, rcp = reduce_bf16_scale(words, 16, s_reduce, warp, lane, THREADS_Q128)

        block = quantize_store_rows(out, words, rcp, 16, group_idx, row0, k_idx, k_ok, N, K)
        store_block_transposed_q128(block, s_tile, k_chunk, row_group)
        store_tile_scale(tid, sc, sct, scale_bits, group_idx, tile_row, tile_col, R, C)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS_Q128))
        out_t_offset = txl.local_scalar("int64", init=as_i64(group_idx) * as_i64(K) * as_i64(N))
        flush_transposed_tile(s_tile, tid, lambda offset: out_t.ptr_to([out_t_offset + offset]), tile_col * BLOCK, tile_row * BLOCK, K, N, THREADS_Q128)

    return compile_kernel(fp8_quant_vec_q128)


@functools.cache
def compile_quant_generic(dtype):
    """
    Any dtype (bf16 / f32), any strides, any N, K: one element per thread step.
    """
    is_bf16 = dtype == "bfloat16"

    @txl.kernel()
    def fp8_quant_generic(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), sc: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), sct: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, sxg: txl.int64, sxn: txl.int64, sxk: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        group_idx, tile_row, tile_col = tile_coords(txl.cta_id(), R, C)
        smem = txl.smem_pool()
        s_reduce = smem.alloc([THREADS // 32], "uint32")
        col = tid & 127
        k_idx = tile_col * BLOCK + col
        k_ok = k_idx < K
        row_parity = tid >> 7  # the two halves of the CTA take alternate rows
        x_offset = txl.local_scalar("int64", init=as_i64(group_idx) * sxg + as_i64(tile_row * BLOCK + row_parity) * sxn + as_i64(k_idx) * sxk)
        num_steps = BLOCK // (THREADS // BLOCK)
        values = txl.alloc_local([num_steps], "uint32")
        acc_amax = txl.local_scalar("uint32", init=txl.uint32(0))
        for i in range(num_steps):
            n_idx = tile_row * BLOCK + row_parity + 2 * i
            txl.ptx.mov.b32(values[i], txl.uint32(0))
            with txl.if_(txl.And(k_ok, n_idx < N)), txl.then_():
                if is_bf16:
                    value_bf16 = txl.local_scalar("uint16")
                    txl.ptx.ld.global_.nc.b16(value_bf16, x.ptr_to([x_offset + sxn * (2 * i)]))
                    txl.ptx.mov.b32(values[i], txl.Cast("uint32", value_bf16) << txl.uint32(16))
                else:
                    txl.ptx.ld.global_.nc.b32(values[i], x.ptr_to([x_offset + sxn * (2 * i)]))
            txl.ptx.max.u32(acc_amax, acc_amax, values[i] & txl.uint32(0x7FFFFFFF))
        amax_bits = cta_max_u32(acc_amax, s_reduce, warp, lane, THREADS)
        scale_bits, rcp_bits = e8m0_scale_bits(txl.reinterpret(amax_bits, ty="float32"))
        rcp = txl.reinterpret(rcp_bits, ty="float32")

        for i in range(num_steps):
            n_idx = tile_row * BLOCK + row_parity + 2 * i
            fp8_pair = cast_e4m3x2(txl.float32(0.0), txl.reinterpret(values[i], ty="float32"), rcp)
            with txl.if_(txl.And(k_ok, n_idx < N)), txl.then_():
                byte = txl.Cast("uint8", fp8_pair)
                txl.ptx.st.global_.b8(out.ptr_to([(as_i64(group_idx) * as_i64(N) + as_i64(n_idx)) * as_i64(K) + as_i64(k_idx)]), byte)
                txl.ptx.st.global_.b8(out_t.ptr_to([(as_i64(group_idx) * as_i64(K) + as_i64(k_idx)) * as_i64(N) + as_i64(n_idx)]), byte)
        store_tile_scale(tid, sc, sct, scale_bits, group_idx, tile_row, tile_col, R, C)

    return compile_kernel(fp8_quant_generic)


def flat_view(x):
    """
    A 1-D view over the storage span x addresses (x itself when contiguous).
    """
    if x.is_contiguous():
        return x.view(-1)
    span = 1 + sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))
    return x.as_strided((span,), (1,))


def as_bytes(tensor):
    return tensor.view(torch.uint8).view(-1)


def launch_quant(x, G, N, K, sxg, sxn, sxk, out, scale, out_t, scale_t):
    """
    Quantize the (G, N, K) view of x with element strides (sxg, sxn, sxk) into the outputs.
    """
    R, C = cdiv(N, BLOCK), cdiv(K, BLOCK)
    num_tiles = G * R * C
    if num_tiles == 0:
        return
    x_flat = flat_view(x)
    use_vec = x.dtype == torch.bfloat16 and sxk == 1 and K % 8 == 0 and N % 16 == 0 and sxn % 8 == 0 and sxg % 8 == 0 and x_flat.data_ptr() % 16 == 0
    if use_vec:
        fn = compile_quant_vec_q128() if G == 1 else compile_quant_vec()
        fn(x_flat, as_bytes(out), as_bytes(out_t), scale.view(-1), scale_t.view(-1), N, K, R, C, sxg, sxn, num_tiles)
    else:
        dtype_name = "bfloat16" if x.dtype == torch.bfloat16 else "float32"
        x_flat = x_flat if x.dtype in (torch.bfloat16, torch.float32) else x_flat.float()
        compile_quant_generic(dtype_name)(x_flat, as_bytes(out), as_bytes(out_t), scale.view(-1), scale_t.view(-1), N, K, R, C, sxg, sxn, sxk, num_tiles)


def fp8_quant_weight_sm100(x: torch.Tensor):
    """
    The SM100 fp8_quant_weight.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    N, K = x.shape
    R, C = cdiv(N, BLOCK), cdiv(K, BLOCK)
    device = x.device
    out = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty((R, C), dtype=torch.float32, device=device)
    out_t = torch.empty((K, N), dtype=torch.float8_e4m3fn, device=device)
    scale_t = torch.empty((C, R), dtype=torch.float32, device=device)
    if x.numel():
        launch_quant(x, 1, N, K, 0, x.stride(0), x.stride(1), out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t


def fp8_quant_weight_grouped_sm100(x: torch.Tensor):
    """
    The SM100 fp8_quant_weight_grouped.
    """
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    G, N, K = x.shape
    R, C = cdiv(N, BLOCK), cdiv(K, BLOCK)
    device = x.device
    out = torch.empty((G, N, K), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty((G, R, C), dtype=torch.float32, device=device)
    out_t = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=device)
    scale_t = torch.empty((G, C, R), dtype=torch.float32, device=device)
    if x.numel():
        launch_quant(x, G, N, K, x.stride(0), x.stride(1), x.stride(2), out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t


def amax_or_cast_shard_tile(tid, warp, lane, s_reduce, x_addr, out_addr, amax_addr, L, K, row_offset, group_idx, tile_row, tile_col, is_cast, VEC, sub_block=0, BLOCK_ROWS=BLOCK):
    """
    Rows [sub_block * BLOCK_ROWS, (sub_block + 1) * BLOCK_ROWS) of one 128x128 tile of a row shard
    (global rows [row_offset, row_offset + L)). VEC elements per load (4: K % 4 == 0; 1: any K).
    is_cast False writes the tile's bf16 amax (BLOCK_ROWS must be BLOCK); True casts.
    """
    lanes_per_row = BLOCK // VEC
    rows_per_pass = THREADS // lanes_per_row
    num_passes = BLOCK_ROWS // rows_per_pass
    assert num_passes >= 1 and (is_cast or BLOCK_ROWS == BLOCK)
    col_lane = tid % lanes_per_row
    row_lane = tid // lanes_per_row
    k_idx = tile_col * BLOCK + col_lane * VEC
    k_ok = k_idx < K
    local_row0 = txl.local_scalar("int32", init=tile_row * BLOCK + sub_block * BLOCK_ROWS + row_lane - row_offset)
    x_offset = txl.local_scalar("int64", init=(as_i64(group_idx) * as_i64(L) + as_i64(local_row0)) * as_i64(K) + as_i64(k_idx))
    values = txl.alloc_local([num_passes * VEC], "uint32")

    def load_pass(pass_idx, addr):
        if VEC == 4:
            txl.ptx[LDG_V4](values[4 * pass_idx], values[4 * pass_idx + 1], values[4 * pass_idx + 2], values[4 * pass_idx + 3], addr)
        else:
            txl.ptx[LDG_B32](values[pass_idx], addr)

    # CTA-uniform: the whole tile lies inside the shard (most tiles). Unpredicated loads let
    # ptxas keep all of them in flight.
    is_inside = txl.And(txl.And(tile_col * BLOCK + BLOCK <= K, tile_row * BLOCK + sub_block * BLOCK_ROWS - row_offset >= 0), tile_row * BLOCK + sub_block * BLOCK_ROWS + BLOCK_ROWS - row_offset <= L)
    with txl.if_(is_inside):
        with txl.then_():
            for pass_idx in range(num_passes):
                load_pass(pass_idx, x_addr(x_offset + as_i64(K) * (pass_idx * rows_per_pass)))
        with txl.else_():
            for pass_idx in range(num_passes):
                local_row = local_row0 + pass_idx * rows_per_pass
                row_ok = txl.And(k_ok, txl.And(local_row >= 0, local_row < L))
                for i in range(VEC):
                    txl.ptx.mov.b32(values[pass_idx * VEC + i], txl.uint32(0))
                with txl.if_(row_ok), txl.then_():
                    load_pass(pass_idx, x_addr(x_offset + as_i64(K) * (pass_idx * rows_per_pass)))
    if not is_cast:
        acc_amax = txl.local_scalar("uint32", init=txl.uint32(0))
        for i in range(num_passes * VEC):
            txl.ptx.max.u32(acc_amax, acc_amax, values[i] & txl.uint32(0x7FFFFFFF))
        amax_bits = cta_max_u32(acc_amax, s_reduce, warp, lane, THREADS)
        with txl.if_(tid == 0), txl.then_():
            amax_bf16 = txl.local_scalar("uint16")
            txl.ptx.cvt.rn.bf16.f32(amax_bf16, txl.reinterpret(amax_bits, ty="float32"))
            txl.ptx.st.global_.f32(amax_addr, txl.reinterpret(txl.Cast("uint32", amax_bf16) << txl.uint32(16), ty="float32"))
        return

    amax = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(amax, amax_addr)
    _, rcp_bits = e8m0_scale_bits(amax)
    rcp = txl.reinterpret(rcp_bits, ty="float32")
    out_offset = txl.local_scalar("int64", init=(as_i64(group_idx) * as_i64(L) + as_i64(local_row0)) * as_i64(K) + as_i64(k_idx))
    for pass_idx in range(num_passes):
        local_row = local_row0 + pass_idx * rows_per_pass
        row_ok = txl.And(k_ok, txl.And(local_row >= 0, local_row < L))
        addr = out_addr(out_offset + as_i64(K) * (pass_idx * rows_per_pass))
        if VEC == 4:
            fp8_pairs = []
            for i in range(2):
                bf16_pair = txl.local_scalar("uint32")
                txl.ptx.cvt.rn.bf16x2.f32(bf16_pair, txl.reinterpret(values[4 * pass_idx + 2 * i + 1], ty="float32"), txl.reinterpret(values[4 * pass_idx + 2 * i], ty="float32"))
                fp8_pairs.append(bf16_word_to_fp8x2(bf16_pair, rcp))
            txl.ptx.st.global_.b32(addr, fp8_pairs[0] | (fp8_pairs[1] << txl.uint32(16)), pred=row_ok)
        else:
            value_bf16 = txl.local_scalar("uint16")
            txl.ptx.cvt.rn.bf16.f32(value_bf16, txl.reinterpret(values[pass_idx], ty="float32"))
            fp8_pair = cast_e4m3x2(txl.float32(0.0), txl.reinterpret(txl.Cast("uint32", value_bf16) << txl.uint32(16), ty="float32"), rcp)
            txl.ptx.st.global_.b8(addr, txl.Cast("uint8", fp8_pair), pred=row_ok)


CAST_SPLIT = 4
CAST_SPLIT_BELOW = 1 << 62  # always split: the split cast is faster at every size


def flat_addr(buf, rank, elem_bytes):
    """
    Element-offset addressing into a contiguous rank-dimensional buffer.
    """
    base = buf.ptr_to([0] * rank)
    return lambda offset: txl.ptx.addr(base, offset * elem_bytes)


@functools.cache
def compile_shard_kernel(is_cast, VEC, amax_rank, split=1):
    """
    Shard tile amax (is_cast False) or cast (True). shard (g, L, K) and out contiguous; amax
    rank-3 contiguous (amax_rank 3) or a 1-D span with group stride asg (amax_rank 1).
    """

    def shard_tile(x, amax, out, L, K, row_offset, tr0, NTR, C, go, asg):
        tid = txl.thread_id()
        smem = txl.smem_pool()
        s_reduce = smem.alloc([THREADS // 32], "uint32")
        cta = txl.cta_id()
        group_idx, local_tile_row, tile_col = tile_coords(cta // split if split > 1 else cta, NTR, C)
        tile_row = tr0 + local_tile_row
        amax_addr = flat_addr(amax, amax_rank, 4)(as_i64(go + group_idx) * asg + as_i64(tile_row) * as_i64(C) + as_i64(tile_col))
        amax_or_cast_shard_tile(tid, txl.warp_id(), txl.lane_id(), s_reduce, flat_addr(x, 3, 4), None if out is None else flat_addr(out, 3, 1), amax_addr, L, K, row_offset, group_idx, tile_row, tile_col, is_cast, VEC, sub_block=cta % split if split > 1 else 0, BLOCK_ROWS=BLOCK // split)

    if is_cast:

        @txl.kernel()
        def fp8_shard_cast(x: tensor_nd(txl.f32, 3), amax: tensor_nd(txl.f32, amax_rank), out: tensor_nd("float8_e4m3fn", 3), L: txl.int32, K: txl.int32, row_offset: txl.int32, tr0: txl.int32, NTR: txl.int32, C: txl.int32, go: txl.int32, asg: txl.int64, num_tiles: txl.int32):
            pdl_entry(num_tiles)
            pdl()
            shard_tile(x, amax, out, L, K, row_offset, tr0, NTR, C, go, asg)

        return compile_kernel(fp8_shard_cast)

    @txl.kernel()
    def fp8_shard_amax(x: tensor_nd(txl.f32, 3), amax: tensor_nd(txl.f32, amax_rank), L: txl.int32, K: txl.int32, row_offset: txl.int32, tr0: txl.int32, NTR: txl.int32, C: txl.int32, go: txl.int32, asg: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        shard_tile(x, amax, None, L, K, row_offset, tr0, NTR, C, go, asg)

    return compile_kernel(fp8_shard_amax)


def shard_geometry(shard, amax, row_offset):
    """
    (G, L, K, tr0, NTR, C) of a shard: its shape, its first tile row and number of tile rows, and
    the tile columns of the amax grid.
    """
    G, L, K = shard.shape
    tr0 = row_offset // BLOCK
    NTR = (row_offset + L + BLOCK - 1) // BLOCK - tr0
    return G, L, K, tr0, NTR, amax.shape[2]


def launch_shard_tiles(shard, amax, out, row_offset, group_offset):
    """
    fp8_shard_amax_sm100 (out None) or fp8_shard_cast_sm100.
    """
    G, L, K, tr0, NTR, C = shard_geometry(shard, amax, row_offset)
    if G == 0 or L == 0 or K == 0:
        return
    VEC = 4 if K % 4 == 0 and shard.data_ptr() % 16 == 0 and (out is None or out.data_ptr() % 4 == 0) else 1
    if amax.is_contiguous():
        amax_arg, amax_rank = amax, 3
    else:
        amax_arg, amax_rank = flat_view(amax), 1
    if out is None:
        compile_shard_kernel(False, VEC, amax_rank)(shard, amax_arg, L, K, row_offset, tr0, NTR, C, group_offset, amax.stride(0), G * NTR * C)
    else:
        num_tiles = G * NTR * C
        split = CAST_SPLIT if VEC == 4 and num_tiles < CAST_SPLIT_BELOW else 1
        compile_shard_kernel(True, VEC, amax_rank, split)(shard, amax_arg, out, L, K, row_offset, tr0, NTR, C, group_offset, amax.stride(0), num_tiles * split)


def fp8_shard_amax_sm100(shard, amax, row_offset, group_offset):
    """
    The SM100 fp8_shard_amax.
    """
    launch_shard_tiles(shard, amax, None, row_offset, group_offset)


def fp8_shard_cast_sm100(shard, amax, out, row_offset, group_offset):
    """
    The SM100 fp8_shard_cast.
    """
    launch_shard_tiles(shard, amax, out, row_offset, group_offset)


def store_scale_element(elem_idx, R, C, amax_addr, sc_addr, sct_addr):
    """
    The E8M0 scale of one amax element, written to the scale and the per-group transposed scale.
    """
    tiles_per_group = R * C
    group_idx, tile_in_group = elem_idx // tiles_per_group, elem_idx % tiles_per_group
    tile_row, tile_col = tile_in_group // C, tile_in_group % C
    amax = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(amax, amax_addr(elem_idx))
    scale_bits, _ = e8m0_scale_bits(amax)
    scale_f32 = txl.reinterpret(scale_bits, ty="float32")
    txl.ptx.st.global_.f32(sc_addr(elem_idx), scale_f32)
    txl.ptx.st.global_.f32(sct_addr(group_idx * tiles_per_group + tile_col * R + tile_row), scale_f32)


@functools.cache
def compile_scales():
    @txl.kernel()
    def fp8_scales(amax: tensor_nd(txl.f32, 3), sc: tensor_nd(txl.f32, 3), sct: tensor_nd(txl.f32, 3), R: txl.int32, C: txl.int32, total: txl.int32, num_ctas: txl.int32):
        pdl_entry(num_ctas)
        pdl()
        elem_idx = txl.cta_id() * THREADS + txl.thread_id()
        with txl.if_(elem_idx < total), txl.then_():
            store_scale_element(elem_idx, R, C, flat_addr(amax, 3, 4), flat_addr(sc, 3, 4), flat_addr(sct, 3, 4))

    return compile_kernel(fp8_scales)


def fp8_scales_sm100(amax, scale, scale_t):
    """
    The SM100 fp8_scales.
    """
    G, R, C = amax.shape
    total = G * R * C
    if total == 0:
        return
    compile_scales()(amax, scale, scale_t, R, C, total, (total + THREADS - 1) // THREADS)


@functools.cache
def compile_pack_vec(dtype):
    """
    K % 8 == 0, N % 16 == 0.
    """

    @txl.kernel()
    def fp8_pack_vec(x: tensor_nd(dtype, 3), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, total: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        tid = txl.thread_id()
        group_idx, tile_row, tile_col = tile_coords(txl.cta_id(), R, C)
        smem = txl.smem_pool()
        s_tile = smem.alloc([BLOCK * BLOCK // 4], "uint32")
        k_chunk, row_group = tid & 15, tid >> 4
        row0 = tile_row * BLOCK + row_group * 8
        k_idx = tile_col * BLOCK + k_chunk * 8
        k_ok = k_idx < K
        x_offset = txl.local_scalar("int64", init=(as_i64(group_idx) * as_i64(N) + as_i64(row0)) * as_i64(K) + as_i64(k_idx))
        x_addr = flat_addr(x, 3, 1)
        block = [(txl.local_scalar("uint32"), txl.local_scalar("uint32")) for _ in range(8)]
        with txl.if_(txl.And(tile_col * BLOCK + BLOCK <= K, tile_row * BLOCK + BLOCK <= N)):
            with txl.then_():
                for row in range(8):
                    txl.ptx.ld.global_.nc.v2.b32(block[row][0], block[row][1], x_addr(x_offset + as_i64(K) * row))
            with txl.else_():
                for row in range(8):
                    txl.ptx.mov.b32(block[row][0], txl.uint32(0))
                    txl.ptx.mov.b32(block[row][1], txl.uint32(0))
                    with txl.if_(txl.And(k_ok, row0 + row < N)), txl.then_():
                        txl.ptx.ld.global_.nc.v2.b32(block[row][0], block[row][1], x_addr(x_offset + as_i64(K) * row))

        for row in range(8):
            txl.ptx.st.global_.v2.b32(out.ptr_to([x_offset + as_i64(K) * row]), block[row][0], block[row][1], pred=txl.And(k_ok, row0 + row < N))
        store_block_transposed(block, s_tile, k_chunk, row_group)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS))
        out_t_offset = txl.local_scalar("int64", init=total + as_i64(group_idx) * as_i64(K) * as_i64(N))
        flush_transposed_tile(s_tile, tid, lambda offset: out.ptr_to([out_t_offset + offset]), tile_col * BLOCK, tile_row * BLOCK, K, N, THREADS)

    return compile_kernel(fp8_pack_vec)


@functools.cache
def compile_pack_generic(dtype):
    @txl.kernel()
    def fp8_pack_generic(x: tensor_nd(dtype, 3), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), N: txl.int32, K: txl.int32, R: txl.int32, C: txl.int32, total: txl.int64, num_tiles: txl.int32):
        pdl_entry(num_tiles)
        pdl()
        tid = txl.thread_id()
        group_idx, tile_row, tile_col = tile_coords(txl.cta_id(), R, C)
        col = tid & 127
        k_idx = tile_col * BLOCK + col
        row_parity = tid >> 7  # the two halves of the CTA take alternate rows
        for i in range(BLOCK // 2):
            n_idx = tile_row * BLOCK + row_parity + 2 * i
            with txl.if_(txl.And(k_idx < K, n_idx < N)), txl.then_():
                byte = txl.local_scalar("uint8")
                offset = (as_i64(group_idx) * as_i64(N) + as_i64(n_idx)) * as_i64(K) + as_i64(k_idx)
                txl.ptx.ld.global_.nc.u8(byte, flat_addr(x, 3, 1)(offset))
                txl.ptx.st.global_.b8(out.ptr_to([offset]), byte)
                txl.ptx.st.global_.b8(out.ptr_to([total + (as_i64(group_idx) * as_i64(K) + as_i64(k_idx)) * as_i64(N) + as_i64(n_idx)]), byte)

    return compile_kernel(fp8_pack_generic)


ONE_BYTE_DTYPES = {torch.float8_e4m3fn: "float8_e4m3fn", torch.uint8: "uint8", torch.int8: "int8", torch.float8_e5m2: "float8_e5m2"}


def fp8_pack_transpose_sm100(x, out):
    """
    The SM100 fp8_pack_transpose.
    """
    G, N, K = x.shape
    num_tiles = G * ((N + BLOCK - 1) // BLOCK) * ((K + BLOCK - 1) // BLOCK)
    if num_tiles == 0:
        return
    if out.dtype != torch.uint8 or out.dim() != 1:
        out = out.view(torch.uint8).view(-1)
    dtype_name = ONE_BYTE_DTYPES.get(x.dtype)
    if dtype_name is None:
        x, dtype_name = x.view(torch.uint8), "uint8"
    use_vec = K % 8 == 0 and N % 16 == 0 and x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0
    (compile_pack_vec(dtype_name) if use_vec else compile_pack_generic(dtype_name))(x, out, N, K, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK, G * N * K, num_tiles)


# Descriptor table fields (int64) of the shard-amax multi launch: the arguments of fp8_shard_amax
# (RO row_offset, GO go, ASG asg), the entry's first tile (START) and padding.
SA_FIELDS = 12
(SA_X, SA_AMAX, SA_L, SA_K, SA_RO, SA_GO, SA_TR0, SA_NTR, SA_C, SA_ASG, SA_START, SA_PAD) = range(SA_FIELDS)


NUM_SMS = 148


def load_table(table, num_entries, fields, capacity):
    """
    Stage the int64 descriptor table in shared memory (once per CTA).
    """
    smem = txl.smem_pool()
    s_table = smem.alloc([capacity * fields], "int64")
    s_reduce = smem.alloc([THREADS // 32], "uint32")
    tid = txl.thread_id()
    with txl.serial((capacity * fields + THREADS - 1) // THREADS) as chunk:
        i = chunk * THREADS + tid
        with txl.if_(i < num_entries * fields), txl.then_():
            value = txl.local_scalar("int64")
            txl.ptx.ld.global_.nc.s64(value, table.ptr_to([as_i64(i)]))
            txl.ptx.st.shared.s64(s_table.ptr_to([i]), value)
    txl.ptx.bar.sync(txl.uint32(2), txl.uint32(THREADS))
    return s_table, s_reduce


def find_entry(s_table, num_entries, fields, start_field, key):
    """
    Index of the last table entry whose start <= key (entries sorted by start).
    """
    entry_idx = txl.local_scalar("int32", init=txl.int32(0))
    with txl.serial(num_entries) as i:
        entry_start = txl.local_scalar("int64")
        txl.ptx.ld.shared.s64(entry_start, s_table.ptr_to([i * fields + start_field]))
        with txl.if_(entry_start <= key), txl.then_():
            txl.assign(entry_idx, i)
    return entry_idx


def read_shared_field(s_table, entry_idx, fields, field_idx, dtype="int32"):
    value = txl.local_scalar("int64")
    txl.ptx.ld.shared.s64(value, s_table.ptr_to([entry_idx * fields + field_idx]))
    return txl.local_scalar(dtype, init=txl.Cast(dtype, value)) if dtype != "int64" else value


def offset_handle(base, offset, elem_bytes):
    return txl.reinterpret(txl.Cast("uint64", base) + txl.Cast("uint64", offset * elem_bytes), ty="handle")


def table_capacity(num_entries):
    capacity = 8
    while capacity < num_entries:
        capacity *= 2
    return capacity


def read_global_field(table, entry_idx, field_idx, dtype="int32"):
    value = txl.local_scalar("int64")
    txl.ptx.ld.global_.nc.s64(value, table.ptr_to([as_i64(entry_idx * SA_FIELDS + field_idx)]))
    return txl.local_scalar(dtype, init=txl.Cast(dtype, value)) if dtype != "int64" else value


@functools.cache
def compile_shard_amax_multi(VEC):
    """
    One CTA per tile over all shards; the CTA finds its shard descriptor in the (L1/L2-resident)
    table with broadcast loads. Faster than a persistent tile loop: the hardware CTA scheduler
    keeps every SM busy to the end.
    """

    @txl.kernel()
    def fp8_shard_amax_multi(table: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), nw: txl.int32, total: txl.int32):
        pdl_entry(total)
        pdl()
        tid = txl.thread_id()
        smem = txl.smem_pool()
        s_reduce = smem.alloc([THREADS // 32], "uint32")
        cta = txl.cta_id()
        entry_idx = txl.local_scalar("int32", init=txl.int32(0))
        with txl.serial(nw) as i:
            entry_start = read_global_field(table, i, SA_START, "int64")
            with txl.if_(entry_start <= as_i64(cta)), txl.then_():
                txl.assign(entry_idx, i)
        x_ptr = read_global_field(table, entry_idx, SA_X, "int64")
        amax_ptr = read_global_field(table, entry_idx, SA_AMAX, "int64")
        L = read_global_field(table, entry_idx, SA_L)
        K = read_global_field(table, entry_idx, SA_K)
        row_offset = read_global_field(table, entry_idx, SA_RO)
        go = read_global_field(table, entry_idx, SA_GO)
        tr0 = read_global_field(table, entry_idx, SA_TR0)
        NTR = read_global_field(table, entry_idx, SA_NTR)
        C = read_global_field(table, entry_idx, SA_C)
        asg = read_global_field(table, entry_idx, SA_ASG, "int64")
        tile_start = read_global_field(table, entry_idx, SA_START)
        group_idx, local_tile_row, tile_col = tile_coords(cta - tile_start, NTR, C)
        tile_row = tr0 + local_tile_row
        amax_addr = offset_handle(amax_ptr, as_i64(go + group_idx) * asg + as_i64(tile_row) * as_i64(C) + as_i64(tile_col), 4)
        amax_or_cast_shard_tile(tid, txl.warp_id(), txl.lane_id(), s_reduce, lambda offset: offset_handle(x_ptr, offset, 4), None, amax_addr, L, K, row_offset, group_idx, tile_row, tile_col, False, VEC)

    return compile_kernel(fp8_shard_amax_multi)


def fp8_shard_amax_multi_sm100(shards, amaxes, row_offsets, group_offsets):
    """
    The SM100 fp8_shard_amax_multi.
    """
    entries, start, VEC = [], 0, 4
    for shard, amax, row_offset, group_offset in zip(shards, amaxes, row_offsets, group_offsets):
        if shard.numel() == 0:
            continue
        G, L, K, tr0, NTR, C = shard_geometry(shard, amax, row_offset)
        if K % 4 or shard.data_ptr() % 16:
            VEC = 1
        entries.append([shard.data_ptr(), amax.data_ptr(), L, K, row_offset, group_offset, tr0, NTR, C, amax.stride(0), start, 0])
        start += G * NTR * C
    if start == 0:
        return lambda: None
    device = shards[0].device
    table = torch.tensor(entries, dtype=torch.int64, device=device).view(-1)
    fn, num_entries, num_tiles = compile_shard_amax_multi(VEC), len(entries), start
    keep_alive = (list(shards), list(amaxes))

    def launch():
        _ = keep_alive  # the table holds raw addresses: the tensors must outlive the launch
        fn(table, num_entries, num_tiles)

    return launch


SC_FIELDS = 6
(SC_AMAX, SC_SC, SC_SCT, SC_R, SC_C, SC_START) = range(SC_FIELDS)


@functools.cache
def compile_scales_multi(capacity):
    @txl.kernel()
    def fp8_scales_multi(table: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), nw: txl.int32, total: txl.int32, num_ctas: txl.int32):
        pdl_entry(num_ctas)
        pdl()
        s_table, _ = load_table(table, nw, SC_FIELDS, capacity)
        global_idx = txl.local_scalar("int32", init=txl.cta_id() * THREADS + txl.thread_id())
        with txl.while_(global_idx < total):
            entry_idx = find_entry(s_table, nw, SC_FIELDS, SC_START, as_i64(global_idx))
            amax_ptr = read_shared_field(s_table, entry_idx, SC_FIELDS, SC_AMAX, "int64")
            scale_ptr = read_shared_field(s_table, entry_idx, SC_FIELDS, SC_SC, "int64")
            scale_t_ptr = read_shared_field(s_table, entry_idx, SC_FIELDS, SC_SCT, "int64")
            R = read_shared_field(s_table, entry_idx, SC_FIELDS, SC_R)
            C = read_shared_field(s_table, entry_idx, SC_FIELDS, SC_C)
            entry_start = read_shared_field(s_table, entry_idx, SC_FIELDS, SC_START)
            elem_idx = txl.local_scalar("int32", init=global_idx - entry_start)
            store_scale_element(elem_idx, R, C, lambda idx: offset_handle(amax_ptr, as_i64(idx), 4), lambda idx: offset_handle(scale_ptr, as_i64(idx), 4), lambda idx: offset_handle(scale_t_ptr, as_i64(idx), 4))
            txl.assign(global_idx, global_idx + num_ctas * THREADS)

    return compile_kernel(fp8_scales_multi)


def fp8_scales_multi_sm100(amaxes, scales, scales_t):
    """
    The SM100 fp8_scales_multi.
    """
    entries, start = [], 0
    for amax, scale, scale_t in zip(amaxes, scales, scales_t):
        G, R, C = amax.shape
        if G * R * C == 0:
            continue
        entries.append([amax.data_ptr(), scale.data_ptr(), scale_t.data_ptr(), R, C, start])
        start += G * R * C
    if start == 0:
        return lambda: None
    table = torch.tensor(entries, dtype=torch.int64, device=amaxes[0].device).view(-1)
    fn, num_entries, total = compile_scales_multi(table_capacity(len(entries))), len(entries), start
    num_ctas = min((total + THREADS - 1) // THREADS, NUM_SMS)
    keep_alive = (list(amaxes), list(scales), list(scales_t))

    def launch():
        _ = keep_alive  # the table holds raw addresses: the tensors must outlive the launch
        fn(table, num_entries, total, num_ctas)

    return launch
