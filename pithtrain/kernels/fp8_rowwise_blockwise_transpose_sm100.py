"""
fp8_quant_act on SM100 in TIRx-lite.

Fast path (bf16, unit column stride, 16 B aligned base and rows, K % 16 == 0, M % 16 == 0):
8-warp CTAs, 3 per SM. One-wave grids load coalesced 128-bit chunks straight into registers;
multi-wave grids remain persistent and load 128x128 tiles with TMA as 8 row parts, with each warp
reloading its own next row part after reading the current one. Each tile uses packed bf16x2
quantization, TMA stores for both FP8 layouts, and column-major row scales. Everything else runs a
generic scalar kernel.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["fp8_quant_act_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


BLOCK = 128
TIRX_DTYPE = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
RCP_448 = 1.0 / 448.0
FULL_MASK = 0xFFFFFFFF


def e8m0_scale(amax):
    """
    (scale, rcp) of an f32 amax: 2^ceil(log2(max(amax, 1e-4) * fp32(1/448))), saturated at 2^127.
    """
    ratio = txl.local_scalar("float32")
    txl.ptx.max.f32(ratio, amax, txl.float32(1e-4))
    txl.ptx.mul.f32(ratio, ratio, txl.float32(RCP_448))
    ratio_bits = txl.reinterpret(ratio, ty="uint32")
    scale_bits = txl.local_scalar("uint32", init=txl.bitwise_and(ratio_bits, txl.uint32(0x7F800000)))
    with txl.if_(txl.bitwise_and(ratio_bits, txl.uint32(0x007FFFFF)) != txl.uint32(0)), txl.then_():
        txl.assign(scale_bits, scale_bits + txl.uint32(0x00800000))
    txl.assign(scale_bits, txl.min(scale_bits, txl.uint32(254 << 23)))
    scale = txl.reinterpret(scale_bits, ty="float32")
    rcp = txl.reinterpret(txl.uint32(254 << 23) - scale_bits, ty="float32")
    return scale, rcp


def to_e4m3_byte(value):
    """
    The e4m3 byte of an f32, rounded to nearest even with saturation.
    """
    byte_pair = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(byte_pair, txl.float32(0.0), value)
    return txl.cast(txl.bitwise_and(txl.cast(byte_pair, "uint32"), txl.uint32(0xFF)), "uint8")


def half_to_f32(half_bits, dtype):
    """
    f32 of the 16-bit value in the low half of a u32.
    """
    if dtype == "bfloat16":
        return txl.reinterpret(txl.shift_left(half_bits, txl.uint32(16)), ty="float32")
    return txl.cast(txl.reinterpret(txl.cast(half_bits, "uint16"), ty="float16"), "float32")


def low_half_f32(word, dtype):
    """
    f32 of the 16-bit value in the low half of a u32.
    """
    return half_to_f32(txl.bitwise_and(word, txl.uint32(0xFFFF)), dtype)


def high_half_f32(word, dtype):
    """
    f32 of the 16-bit value in the high half of a u32.
    """
    if dtype == "bfloat16":
        return txl.reinterpret(txl.bitwise_and(word, txl.uint32(0xFFFF0000)), ty="float32")
    return half_to_f32(txl.shift_right(word, txl.uint32(16)), dtype)


@functools.cache
def generic_kernel(dtype: str):
    @txl.kernel()
    def fp8_quant_act_generic(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_scale: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_t_scale: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), M: txl.int32, K: txl.int32, num_row_blocks: txl.int32, num_col_blocks: txl.int32, row_stride: txl.int64, col_stride: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_row_blocks * num_col_blocks, block=128))
        bx = txl.cta_id()
        tidx = txl.thread_id()
        smem = txl.smem_pool()
        s_amax = smem.alloc([4], txl.f32)
        row_block_idx = txl.local_scalar("int32", init=txl.truncmod(bx, num_row_blocks))
        col_block_idx = txl.local_scalar("int32", init=txl.truncdiv(bx, num_row_blocks))
        row = txl.local_scalar("int32", init=row_block_idx * BLOCK + tidx)
        row_ok = row < M
        amax = txl.local_scalar("float32", init=txl.float32(0.0))
        value = txl.local_scalar("float32")
        half_bits = txl.local_scalar("uint16")

        def load_element(col_offset):
            """
            Load x[row, col] into value, zero outside x; returns col.
            """
            col = col_block_idx * BLOCK + col_offset
            txl.assign(value, txl.float32(0.0))
            with txl.if_(row_ok & (col < K)), txl.then_():
                x_addr = x.ptr_to([txl.cast(row, "int64") * row_stride + txl.cast(col, "int64") * col_stride])
                if dtype == "float32":
                    txl.ptx["ld.global.nc.f32"](value, x_addr)
                else:
                    txl.ptx["ld.global.nc.b16"](half_bits, x_addr)
                    txl.assign(value, half_to_f32(txl.cast(half_bits, "uint32"), dtype))
            return col

        with txl.serial(BLOCK) as col_offset:
            load_element(col_offset)
            txl.ptx.max.f32(amax, amax, txl.abs(value))
        row_scale_expr, row_rcp_expr = e8m0_scale(amax)
        row_scale = txl.local_scalar("float32", init=row_scale_expr)
        row_rcp = txl.local_scalar("float32", init=row_rcp_expr)
        with txl.if_(row_ok), txl.then_():
            txl.ptx.st.global_.f32(out_scale.ptr_to([txl.cast(row, "int64") * num_col_blocks + col_block_idx]), row_scale)
        with txl.serial(BLOCK) as col_offset:
            col = load_element(col_offset)
            with txl.if_(row_ok & (col < K)), txl.then_():
                txl.ptx.st.global_.b8(out.ptr_to([txl.cast(row, "int64") * K + col]), to_e4m3_byte(value * row_rcp))

        peer = txl.local_scalar("uint32")
        for offset in (16, 8, 4, 2, 1):
            txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(amax, ty="uint32"), txl.uint32(offset), txl.uint32(31), txl.uint32(FULL_MASK))
            txl.ptx.max.f32(amax, amax, txl.reinterpret(peer, ty="float32"))
        with txl.if_(txl.bitwise_and(tidx, 31) == 0), txl.then_():
            txl.ptx.st.shared.f32(s_amax.ptr_to([tidx >> 5]), amax)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(128))
        block_amax = txl.local_scalar("float32")
        warp_amax = txl.local_scalar("float32")
        txl.ptx.ld.shared.f32(block_amax, s_amax.ptr_to([0]))
        for warp_idx in range(1, 4):
            txl.ptx.ld.shared.f32(warp_amax, s_amax.ptr_to([warp_idx]))
            txl.ptx.max.f32(block_amax, block_amax, warp_amax)
        block_scale, block_rcp_expr = e8m0_scale(block_amax)
        block_rcp = txl.local_scalar("float32", init=block_rcp_expr)
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.st.global_.f32(out_t_scale.ptr_to([col_block_idx * num_row_blocks + row_block_idx]), block_scale)
        with txl.serial(BLOCK) as col_offset:
            col = load_element(col_offset)
            with txl.if_(row_ok & (col < K)), txl.then_():
                txl.ptx.st.global_.b8(out_t.ptr_to([txl.cast(col, "int64") * M + row]), to_e4m3_byte(value * block_rcp))

    return entry_function(fp8_quant_act_generic.compile(TARGET))


# Fast path: bf16 128x128 tiles, loaded by TMA (persistent grids) or straight into registers.


TMA_LOAD_2D = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
TMA_STORE_2D = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
FAST_CTAS_PER_SM = 3
FAST_THREADS = 256
X_TILE_BYTES = BLOCK * BLOCK * 2


def begin_range(name):
    """
    Open the named iket range; returns the token end_range closes.
    """
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def end_range(token):
    txl.cuda.iket.range_end(token[0])


def mul_bf16x2(dst, src, factor):
    txl.ptx["mul.rn.bf16x2"](dst, src, factor)


def cvt_e4m3x2(dst_pair, src):
    """
    Two e4m3 bytes (in a u16) of the two bf16 halves of src.
    """
    txl.ptx["cvt.rn.satfinite.e4m3x2.bf16x2"](dst_pair, src)


def pack_halves(bits_lo, bits_hi):
    return txl.bitwise_or(txl.cast(bits_lo, "uint32"), txl.shift_left(txl.cast(bits_hi, "uint32"), txl.uint32(16)))


def broadcast_bf16x2(value):
    """
    Both halves = the bf16 of an f32 power of two (exact).
    """
    bits = txl.reinterpret(value, ty="uint32")
    return txl.bitwise_or(txl.shift_right(bits, txl.uint32(16)), txl.bitwise_and(bits, txl.uint32(0xFFFF0000)))


@functools.cache
def fast_kernel(ctas_per_sm: int, is_persistent: bool, use_direct_load: bool = False):
    return entry_function(trace_fast_kernel(ctas_per_sm, is_persistent, use_direct_load).compile(TARGET))


def trace_fast_kernel(ctas_per_sm: int, is_persistent: bool = True, use_direct_load: bool = False):
    def encode_tensor_maps(x, out, out_t, M, K, row_stride):
        """
        TMA descriptors of x (unless loaded directly), out and out_t (128B swizzle).
        """
        if not use_direct_load:
            x_map = txl.stack_alloca("tensormap", 1)
            txl.cu_tensor_map_encode_tiled(x_map, "bfloat16", 2, x.data, K, M, row_stride * 2, BLOCK, BLOCK // 8, 1, 1, 0, 0, 2, 0)
        out_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(out_map, "uint8", 2, out.data, K, M, K, BLOCK, BLOCK, 1, 1, 0, 0, 2, 0)
        out_t_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(out_t_map, "uint8", 2, out_t.data, M, K, M, BLOCK, BLOCK, 1, 1, 0, 3, 2, 0)
        return (out_map, out_t_map) if use_direct_load else (x_map, out_map, out_t_map)

    @txl.kernel()
    def fp8_quant_act_tiled(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_scale: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), out_t: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), out_t_scale: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), M: txl.int32, K: txl.int32, num_row_blocks: txl.int32, num_col_blocks: txl.int32, row_stride: txl.int64, num_ctas: txl.int32):
        tensor_maps = encode_tensor_maps(x, out, out_t, M, K, row_stride)
        launch = txl.cuda.LaunchConfig(grid=num_ctas, block=8 * 32)
        txl.device_entry(launch=launch, kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=ctas_per_sm))
        if use_direct_load:
            out_map, out_t_map = tensor_maps
        else:
            x_map, out_map, out_t_map = tensor_maps
        bx = txl.cta_id()
        tidx = txl.thread_id()
        lane = tidx & 31
        warp = txl.warp_id()
        smem = txl.smem_pool()
        if not use_direct_load:
            s_x = smem.alloc([BLOCK * 64], txl.u32, align=1024)  # [row][64 words] bf16 tile
        s_rows = smem.alloc([BLOCK * 32], txl.u32, align=1024)  # [row][32 words] rowwise FP8
        s_tile_t = smem.alloc([BLOCK * 32], txl.u32, align=1024)  # [k][128 B of m] FP8, TMA 128B swizzle
        s_amax = smem.alloc([8], txl.f32, align=16)
        if not use_direct_load:
            full_barriers = smem.alloc([8], "uint64", align=8)
            load_first_tile(x_map, s_x, full_barriers, bx, tidx, num_col_blocks)
        tile_idx = txl.local_scalar("int32", init=bx)
        phase = txl.local_scalar("int32", init=txl.int32(0))

        def advance_tile():
            txl.assign(tile_idx, tile_idx + num_ctas)
            if not use_direct_load:
                txl.assign(phase, phase ^ 1)

        def reload_row_part():
            """
            Load this warp's 16 rows of its next tile into the row part it has just read.
            """
            next_tile_idx = tile_idx + num_ctas
            with txl.if_(next_tile_idx < num_row_blocks * num_col_blocks), txl.then_():
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full_barriers.ptr_to([warp]), txl.uint32(X_TILE_BYTES // 8))
                txl.ptx[TMA_LOAD_2D](s_x.ptr_to([warp * 16 * 64]), txl.address_of(x_map), txl.Cast("int32", txl.truncmod(next_tile_idx, num_col_blocks) * BLOCK), txl.Cast("int32", txl.truncdiv(next_tile_idx, num_col_blocks) * BLOCK + warp * 16), txl.cuda.cvta_generic_to_shared(full_barriers.ptr_to([warp])))

        if is_persistent:
            quantize_tiles(x, row_stride, s_x if not use_direct_load else None, s_rows, s_tile_t, s_amax, full_barriers if not use_direct_load else None, txl.int32(0), phase, tile_idx, advance_tile, out_map, out_t_map, out_scale, out_t_scale, M, K, num_row_blocks, num_col_blocks, num_row_blocks * num_col_blocks, tidx, lane, warp, use_direct_load=use_direct_load, reload_next=None if use_direct_load else reload_row_part, has_next=lambda: tile_idx + num_ctas < num_row_blocks * num_col_blocks)
        else:  # num_ctas == number of tiles: one tile per CTA
            quantize_tiles(x, row_stride, s_x if not use_direct_load else None, s_rows, s_tile_t, s_amax, full_barriers if not use_direct_load else None, txl.int32(0), txl.int32(0), tile_idx, None, out_map, out_t_map, out_scale, out_t_scale, M, K, num_row_blocks, num_col_blocks, bx + 1, tidx, lane, warp, is_oneshot=True, use_direct_load=use_direct_load)

    return fp8_quant_act_tiled


def load_first_tile(x_map, s_x, full_barriers, bx, tidx, num_col_blocks):
    """
    Initialise the 8 row-part barriers and issue the TMA loads of the CTA's first tile.
    """
    row_block_idx = txl.truncdiv(bx, num_col_blocks)
    col_block_idx = txl.truncmod(bx, num_col_blocks)
    with txl.if_(tidx == 0), txl.then_():
        for part in range(8):
            txl.ptx.mbarrier.init.shared.b64(full_barriers.ptr_to([part]), txl.uint32(1))
        txl.ptx.fence.mbarrier_init.release.cluster()
        for part in range(8):
            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full_barriers.ptr_to([part]), txl.uint32(X_TILE_BYTES // 8))
            txl.ptx[TMA_LOAD_2D](s_x.ptr_to([part * 16 * 64]), txl.address_of(x_map), txl.Cast("int32", col_block_idx * BLOCK), txl.Cast("int32", row_block_idx * BLOCK + part * 16), txl.cuda.cvta_generic_to_shared(full_barriers.ptr_to([part])))
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(FAST_THREADS))


class FastThread:
    """
    A thread's place in a fast-path CTA: it owns rows 4 * row_quad .. 4 * row_quad + 3 of each tile,
    columns 8 * col_chunk .. and 64 + 8 * col_chunk ..
    """

    def __init__(self, tidx, lane, warp):
        self.tidx, self.lane, self.warp = tidx, lane, warp
        self.row_group = lane >> 3  # row group within the warp
        self.col_chunk = lane & 7  # 16 B column chunk within each 64-column half
        self.row_quad = warp * 4 + self.row_group


class TileRegisters:
    """
    The fast path's per-thread registers, allocated once and reused by every tile.
    """

    def __init__(self):
        self.data = txl.alloc_local([32], "uint32")  # [half][row][word], word = 2 adjacent columns
        self.row_amax = txl.alloc_local([4], "float32")
        self.amax_bits = txl.local_scalar("uint32")
        self.row_amax_pairs = txl.alloc_local([2], "uint32")  # bf16x2 amax of rows (0, 1) and (2, 3)
        self.peer = txl.local_scalar("uint32")
        self.row_block_idx = txl.local_scalar("int32")
        self.col_block_idx = txl.local_scalar("int32")
        self.scale_base = txl.local_scalar("int64")
        self.row_start = txl.local_scalar("int32")
        self.rows_offset = txl.local_scalar("int32")
        self.scaled = txl.alloc_local([4], "uint32")
        self.rcp_bf16x2 = txl.local_scalar("uint32")
        self.fp8_pairs = txl.alloc_local([4], "uint16")
        self.t_words = txl.alloc_local([8], "uint32")
        self.rotated = txl.alloc_local([8], "uint32")
        self.block_amax = txl.local_scalar("float32")
        self.warp_amax = txl.local_scalar("float32")


def quantize_tiles(x, row_stride, s_x, s_rows, s_tile_t, s_amax, full_barriers, stage, phase, tile_idx, advance, out_map, out_t_map, out_scale, out_t_scale, M, K, num_row_blocks, num_col_blocks, tile_end, tidx, lane, warp, is_oneshot=False, use_direct_load=False, reload_next=None, has_next=None):
    """
    Quantize the CTA's tiles [tile_idx, tile_end), stepping with advance (one tile when is_oneshot).
    """
    thread = FastThread(tidx, lane, warp)
    regs = TileRegisters()
    tile_loop = txl.if_(tile_idx < tile_end) if is_oneshot else txl.while_(tile_idx < tile_end)
    with tile_loop:
        if is_oneshot:
            then_scope = txl.then_()
            then_scope.__enter__()
        txl.assign(regs.row_block_idx, txl.truncdiv(tile_idx, num_col_blocks))
        txl.assign(regs.col_block_idx, txl.truncmod(tile_idx, num_col_blocks))
        txl.assign(regs.row_start, regs.row_block_idx * BLOCK)
        txl.assign(regs.rows_offset, 0)  # s_rows holds one tile
        # x_scale stored column-major: element (m, kb) at kb * M + m
        txl.assign(regs.scale_base, txl.cast(regs.col_block_idx, "int64") * txl.cast(M, "int64") + txl.cast(regs.row_start, "int64"))

        range_rowwise = begin_range("rowwise")
        if use_direct_load:
            load_tile_direct(regs, thread, x, row_stride, M, K)
        else:
            load_tile_smem(regs, thread, s_x, full_barriers, stage, phase, reload_next)
        reduce_row_amax(regs)
        quantize_rows(regs, thread, s_rows, out_scale, M)
        end_range(range_rowwise)

        range_block_amax = begin_range("blk-amax")
        reduce_warp_amax(regs, thread, s_amax)
        store_rows(regs, thread, s_rows, out_map, is_oneshot)
        store_block_scale(regs, thread, s_amax, out_t_scale, num_row_blocks)
        end_range(range_block_amax)

        range_transpose = begin_range("t-smem")
        transpose_to_smem(regs, thread, s_tile_t)
        store_transposed(regs, thread, s_tile_t, out_t_map, reload_next, has_next)
        end_range(range_transpose)
        if is_oneshot:
            then_scope.__exit__(None, None, None)
        else:
            advance()
    with txl.if_(tidx == 0), txl.then_():
        txl.ptx.cp.async_.bulk.wait_group.read(0)


def load_tile_direct(regs, thread, x, row_stride, M, K):
    """
    Load the thread's 4 rows x 2 chunks of 8 columns from global memory; chunks outside x read 0.
    """
    data, row_start, col_block_idx = regs.data, regs.row_start, regs.col_block_idx
    col_chunk, row_quad = thread.col_chunk, thread.row_quad
    for half in range(2):
        for quad_row in range(4):
            base = half * 16 + quad_row * 4
            for word in range(4):
                txl.assign(data[base + word], txl.uint32(0))
            col = col_block_idx * BLOCK + half * 64 + col_chunk * 8
            row = row_start + row_quad * 4 + quad_row
            with txl.if_((row < M) & (col < K)), txl.then_():
                txl.ptx["ld.global.nc.L1::no_allocate.v4.b32"](data[base], data[base + 1], data[base + 2], data[base + 3], x.ptr_to([txl.cast(row, "int64") * row_stride + txl.cast(col, "int64")]))


def load_tile_smem(regs, thread, s_x, full_barriers, stage, phase, reload_next):
    """
    Wait for the warp's TMA row part, read the thread's words from it, then let lane 0 reload it.
    """
    data = regs.data
    lane, warp, col_chunk, row_quad = thread.lane, thread.warp, thread.col_chunk, thread.row_quad
    range_wait = begin_range("wait-full")
    txl.cuda.mbarrier_wait(full_barriers.ptr_to([stage * 8 + warp]), phase)
    end_range(range_wait)
    for half in range(2):
        for quad_row in range(4):
            base = half * 16 + quad_row * 4
            txl.ptx["ld.shared.v4.b32"](data[base], data[base + 1], data[base + 2], data[base + 3], s_x.ptr_to([stage * (BLOCK * 64) + (row_quad * 4 + quad_row) * 64 + half * 32 + col_chunk * 4]))
    if reload_next is not None:
        # order this thread's generic-proxy reads before the next TMA write into the part
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx["bar.warp.sync"](txl.uint32(FULL_MASK))
        with txl.if_(lane == 0), txl.then_():
            reload_next()


def reduce_row_amax(regs):
    """
    The amax of each of the thread's 4 rows over the 8 lanes of its row group, into regs.row_amax.
    """
    data, row_amax, amax_bits = regs.data, regs.row_amax, regs.amax_bits
    row_amax_pairs, peer = regs.row_amax_pairs, regs.peer
    for quad_row in range(4):
        acc_amax = txl.local_scalar("uint32")
        pair_amax = txl.local_scalar("uint32")
        base_lo, base_hi = quad_row * 4, 16 + quad_row * 4
        txl.ptx["max.xorsign.abs.bf16x2"](acc_amax, data[base_lo], data[base_lo + 1])
        txl.ptx["max.xorsign.abs.bf16x2"](pair_amax, data[base_lo + 2], data[base_lo + 3])
        txl.ptx["max.xorsign.abs.bf16x2"](acc_amax, acc_amax, pair_amax)
        txl.ptx["max.xorsign.abs.bf16x2"](pair_amax, data[base_hi], data[base_hi + 1])
        txl.ptx["max.xorsign.abs.bf16x2"](acc_amax, acc_amax, pair_amax)
        txl.ptx["max.xorsign.abs.bf16x2"](pair_amax, data[base_hi + 2], data[base_hi + 3])
        txl.ptx["max.xorsign.abs.bf16x2"](acc_amax, acc_amax, pair_amax)
        txl.ptx["max.xorsign.abs.bf16x2"](amax_bits, acc_amax, txl.shift_right(acc_amax, txl.uint32(16)))
        # xorsign gives the result the xor of the inputs' signs: 0x7FFF keeps the bf16 magnitude
        if quad_row % 2 == 0:
            txl.assign(row_amax_pairs[quad_row // 2], txl.bitwise_and(amax_bits, txl.uint32(0x7FFF)))
        else:
            txl.assign(row_amax_pairs[quad_row // 2], txl.bitwise_or(row_amax_pairs[quad_row // 2], txl.shift_left(txl.bitwise_and(amax_bits, txl.uint32(0x7FFF)), txl.uint32(16))))
    for offset in (1, 2, 4):
        for pair in range(2):
            txl.ptx.shfl_sync.bfly.b32(peer, row_amax_pairs[pair], txl.uint32(offset), txl.uint32(31), txl.uint32(FULL_MASK))
            txl.ptx.max.bf16x2(row_amax_pairs[pair], row_amax_pairs[pair], peer)
    for quad_row in range(4):
        txl.assign(row_amax[quad_row], low_half_f32(row_amax_pairs[quad_row // 2], "bfloat16") if quad_row % 2 == 0 else high_half_f32(row_amax_pairs[quad_row // 2], "bfloat16"))


def quantize_rows(regs, thread, s_rows, out_scale, M):
    """
    Store each row's scale and write the rowwise FP8 of the thread's words into s_rows.
    """
    data, row_amax, rcp_bf16x2 = regs.data, regs.row_amax, regs.rcp_bf16x2
    scaled, fp8_pairs = regs.scaled, regs.fp8_pairs
    row_start, rows_offset, scale_base = regs.row_start, regs.rows_offset, regs.scale_base
    col_chunk, row_quad = thread.col_chunk, thread.row_quad
    for quad_row in range(4):
        tile_row = row_quad * 4 + quad_row
        scale, rcp = e8m0_scale(row_amax[quad_row])
        txl.assign(rcp_bf16x2, broadcast_bf16x2(rcp))
        with txl.if_((col_chunk == 0) & (row_start + tile_row < M)), txl.then_():
            txl.ptx.st.global_.f32(out_scale.ptr_to([scale_base + txl.cast(tile_row, "int64")]), scale)
        for half in range(2):
            base = half * 16 + quad_row * 4
            for word in range(4):
                mul_bf16x2(scaled[word], data[base + word], rcp_bf16x2)
                cvt_e4m3x2(fp8_pairs[word], scaled[word])
            txl.ptx["st.shared.v2.b32"](s_rows.ptr_to([rows_offset + tile_row * 32 + half * 16 + col_chunk * 2]), pack_halves(fp8_pairs[0], fp8_pairs[1]), pack_halves(fp8_pairs[2], fp8_pairs[3]))


def reduce_warp_amax(regs, thread, s_amax):
    """
    The amax of the warp's 16 rows, written to s_amax[warp] by lane 0.
    """
    row_amax, block_amax, peer = regs.row_amax, regs.block_amax, regs.peer
    lane, warp = thread.lane, thread.warp
    txl.assign(block_amax, row_amax[0])
    for quad_row in range(1, 4):
        txl.ptx.max.f32(block_amax, block_amax, row_amax[quad_row])
    for offset in (8, 16):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(block_amax, ty="uint32"), txl.uint32(offset), txl.uint32(31), txl.uint32(FULL_MASK))
        txl.ptx.max.f32(block_amax, block_amax, txl.reinterpret(peer, ty="float32"))
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.f32(s_amax.ptr_to([warp]), block_amax)


def store_rows(regs, thread, s_rows, out_map, is_oneshot):
    """
    Once every warp has written s_rows, store the tile's rowwise FP8 with TMA.
    """
    tidx = thread.tidx
    if not is_oneshot:
        # the previous tile's TMA stores must have read s_rows / s_tile_t before they're rewritten
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.cp.async_.bulk.wait_group.read(0)
    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(FAST_THREADS))
    # every warp's rowwise FP8 is in s_rows now: store it while the transposed half is built
    with txl.if_(tidx == 0), txl.then_():
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx[TMA_STORE_2D](txl.address_of(out_map), txl.Cast("int32", regs.col_block_idx * BLOCK), txl.Cast("int32", regs.row_start), s_rows.ptr_to([regs.rows_offset]))
        txl.ptx.cp.async_.bulk.commit_group()


def store_block_scale(regs, thread, s_amax, out_t_scale, num_row_blocks):
    """
    Reduce the 8 warp amaxes to the tile's scale, store it, and put its reciprocal in rcp_bf16x2.
    """
    block_amax, warp_amax = regs.block_amax, regs.warp_amax
    txl.ptx.ld.shared.f32(block_amax, s_amax.ptr_to([0]))
    for warp_idx in range(1, 8):
        txl.ptx.ld.shared.f32(warp_amax, s_amax.ptr_to([warp_idx]))
        txl.ptx.max.f32(block_amax, block_amax, warp_amax)
    block_scale, block_rcp = e8m0_scale(block_amax)
    txl.assign(regs.rcp_bf16x2, broadcast_bf16x2(block_rcp))
    with txl.if_(thread.tidx == 0), txl.then_():
        txl.ptx.st.global_.f32(out_t_scale.ptr_to([regs.col_block_idx * num_row_blocks + regs.row_block_idx]), block_scale)


def transpose_to_smem(regs, thread, s_tile_t):
    """
    Scale the thread's words by the tile's reciprocal and write their FP8 transposed into s_tile_t.
    """
    data, rcp_bf16x2, fp8_pairs = regs.data, regs.rcp_bf16x2, regs.fp8_pairs
    t_words, rotated = regs.t_words, regs.rotated
    warp, row_group, col_chunk = thread.warp, thread.row_group, thread.col_chunk
    # transposed FP8 in TMA 128B-swizzle layout: byte (k, m) at k*128 + ((m/16) ^ (k%8))*16 + m%16.
    # Word T[j] = rows 4g..4g+3 of column 64h + 8cc + j (g = row_quad, h = half, cc = col_chunk).
    # Lane cc stores T[(step + cc) % 8] at each step, so k % 8 differs across the 8 lanes of a row
    # group: 32 distinct banks per store.
    for half in range(2):
        for quad_row in range(4):
            for word in range(4):
                mul_bf16x2(data[half * 16 + quad_row * 4 + word], data[half * 16 + quad_row * 4 + word], rcp_bf16x2)
        for word in range(4):
            word0, word1, word2, word3 = (data[half * 16 + quad_row * 4 + word] for quad_row in range(4))
            for half_word, selector in ((0, 0x5410), (1, 0x7632)):
                rows01 = txl.local_scalar("uint32")
                rows23 = txl.local_scalar("uint32")
                txl.ptx["prmt.b32"](rows01, word0, word1, txl.uint32(selector))
                txl.ptx["prmt.b32"](rows23, word2, word3, txl.uint32(selector))
                cvt_e4m3x2(fp8_pairs[0], rows01)
                cvt_e4m3x2(fp8_pairs[1], rows23)
                txl.assign(t_words[2 * word + half_word], pack_halves(fp8_pairs[0], fp8_pairs[1]))
        # rotate left by col_chunk: rotated[i] = t_words[(i + col_chunk) % 8]
        for bit in (1, 2, 4):
            for i in range(8):
                txl.assign(rotated[i], txl.Select((col_chunk & bit) != 0, t_words[(i + bit) % 8], t_words[i]))
            for i in range(8):
                txl.assign(t_words[i], rotated[i])
        for step in range(8):
            chunk_col = (step + col_chunk) & 7
            k_idx = half * 64 + col_chunk * 8 + chunk_col
            txl.ptx["st.shared.b32"](s_tile_t.ptr_to([k_idx * 32 + txl.bitwise_xor(warp, chunk_col) * 4 + row_group]), t_words[step])


def store_transposed(regs, thread, s_tile_t, out_t_map, reload_next, has_next):
    """
    Once every warp has written s_tile_t, store the tile's transposed FP8 with TMA.
    """
    tidx = thread.tidx
    if reload_next is not None:
        # single s_rows buffer: the rowwise store must have read it before the next tile rewrites it
        with txl.if_((tidx == 0) & has_next()), txl.then_():
            txl.ptx.cp.async_.bulk.wait_group.read(0)
    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(FAST_THREADS))
    with txl.if_(tidx == 0), txl.then_():
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx[TMA_STORE_2D](txl.address_of(out_t_map), txl.Cast("int32", regs.row_start), txl.Cast("int32", regs.col_block_idx * BLOCK), s_tile_t.ptr_to([0]))
        txl.ptx.cp.async_.bulk.commit_group()


NUM_SMS = None


def num_sms():
    global NUM_SMS
    if NUM_SMS is None:
        NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return NUM_SMS


def is_fast_path(x):
    M, K = x.shape
    return x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.data_ptr() % 16 == 0 and x.stride(0) % 8 == 0 and K % 16 == 0 and M % 16 == 0


def fp8_quant_act_sm100(x: torch.Tensor):
    """
    fp8_quant_act on SM100; x_scale is column-major.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    num_col_blocks, num_row_blocks = (K + BLOCK - 1) // BLOCK, (M + BLOCK - 1) // BLOCK
    device = x.device
    use_fast = M > 0 and K > 0 and is_fast_path(x)
    out = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device)
    # x_scale is returned column-major (act_scale_like's SM100 layout): the fast path stores it so,
    # each tile's 128 row scales one contiguous run; the generic kernel's row-major store is copied.
    scale_store = torch.empty((num_col_blocks, M) if use_fast else (M, num_col_blocks), dtype=torch.float32, device=device)
    out_scale = scale_store.T if use_fast else torch.empty((num_col_blocks, M), dtype=torch.float32, device=device).T
    out_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=device)
    out_t_scale = torch.empty((num_col_blocks, num_row_blocks), dtype=torch.float32, device=device)
    if M == 0 or K == 0:
        return out, out_scale, out_t, out_t_scale
    row_stride, col_stride = x.stride()
    span = (M - 1) * row_stride + (K - 1) * col_stride + 1
    x_flat = x.as_strided((span,), (1,))
    outputs = (out.view(torch.uint8).view(-1), scale_store.view(-1), out_t.view(torch.uint8).view(-1), out_t_scale.view(-1))
    if use_fast:
        num_tiles = num_row_blocks * num_col_blocks
        # one CTA per tile when the grid fits one resident wave, else persistent CTAs
        num_ctas = min(num_tiles, FAST_CTAS_PER_SM * num_sms())
        is_persistent = num_ctas < num_tiles
        # Direct register loads win consistently when every tile fits in one resident wave;
        # the self-producing TMA path remains faster once CTAs iterate persistently.
        fast_kernel(FAST_CTAS_PER_SM, is_persistent, not is_persistent)(x_flat, *outputs, M, K, num_row_blocks, num_col_blocks, row_stride, num_ctas)
    else:
        generic_kernel(TIRX_DTYPE[x.dtype])(x_flat, *outputs, M, K, num_row_blocks, num_col_blocks, row_stride, col_stride)
    if not use_fast:
        out_scale.copy_(scale_store)
    return out, out_scale, out_t, out_t_scale
