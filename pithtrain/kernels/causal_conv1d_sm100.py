"""
causal_conv1d_sm100: the causal depthwise conv1d of the Qwen3.5 Gated DeltaNet (forward y, backward
dx and dw) on SM100, as TIRx-lite kernels. Contracts as in the picker causal_conv1d.

Tiles are T = 128 tokens x TILE_CHANNELS = 64 channels. Every output runs the K-tap fp32 fma chain
in tap order from 0.0, rounded once (bit-identical to ATen).

TMA path (C % 8 == 0, K <= 8, 16-byte aligned tensors, and S % 8 == 0 where a channels-first tensor
is involved): persistent CTAs, one producer warp streams tiles into a shared-memory ring with TMA
(OOB rows read as zero, which is exactly the causal padding / history clipping), compute warps read
the ring and write the results to a double-buffered staging tile that one thread stores with TMA.
- forward: x tile rows [t0 + offset - (K - 1), +256 + K - 1) channels-last (a 256-row head box and a
  K - 1 row halo box, evict-last L2 hints), lane = channel pair with f32x2 FMAs; y staging is two
  128B-swizzled [64 ch][64 tok] boxes, stored into y viewed as (B, C, S).
- backward: 2-CTA clusters per channel slice, g rows [s0 - offset, +T + K - 1) (channels-last, or
  [ch][CHANNELS_FIRST_PITCH tok] for g in the output's channels-first layout) and x rows
  [s0 - (K - 1), +T + K - 1); dx staging [T][slice]. BF16 w on channels-last g computes with
  fma.rn.f32.bf16 straight from the staged halves. dw: per thread a sequential fma over its tokens,
  the token groups summed in order, then the cluster's ranks summed in order through DSMEM. Same
  order in every path, so dx / dw bits do not depend on g's layout.
Plain path (any C, S, K, strides of g): the same tiles and arithmetic with plain loads / stores, dw
partials summed in the TMA path's order by a second kernel.
Inputs outside the contract the kernels read directly (w of another float dtype, non-contiguous x
or w) are converted first: w to fp32, dw rounded back to the dtype of w.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["causal_conv1d_bwd_sm100", "causal_conv1d_fwd_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


T = 128  # tokens (rows) per tile
TILE_CHANNELS = 64  # channels per tile
TOKENS_PER_THREAD = 32  # tokens per compute thread (scalar kernels)
PACKED_TOKENS_PER_THREAD = 16  # tokens per compute thread (packed TMA kernels: 2 channels x 16 tokens)
NUM_TOKEN_GROUPS = 8  # dw token groups per tile (T // PACKED_TOKENS_PER_THREAD), the same in every path
NUM_COMPUTE_WARPS = 8
NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32
PRODUCER_WARP = NUM_COMPUTE_WARPS  # the TMA kernels' producer warp follows the compute warps
NUM_STAGES_BWD = 2
NUM_DX_BUFS = 2  # dx staging buffers of the TMA backward
CTAS_PER_SM = 2
NUM_SMS = 148
USE_WARP_ARRIVE = False  # one empty-barrier arrival per warp after warp_sync instead of one per thread
EMPTY_ARRIVALS = NUM_COMPUTE_WARPS if USE_WARP_ARRIVE else NUM_COMPUTE_THREADS
USE_BF16_FMA_BWD = True  # packed channels-last backward with native BF16-input FP32 FMAs (fma.rn.f32.bf16)
USE_BF16_FMA_FWD = False  # forward with native BF16-input FP32 FMAs instead of the f32x2 path
CONVERT_BEFORE_STAGING = True  # forward: convert (and shuffle) before the staging barrier
# Channels-first g tile pitch (tokens): >= 3 * TOKENS_PER_THREAD + 48, and 76 words = 12 mod 32
# banks, so the 16B reads of 8 lanes (8 channels) along tokens hit distinct bank quads.
CHANNELS_FIRST_PITCH = 152
CLUSTER_SIZE = 2  # CTAs per backward cluster; also the dw rank order of every backward path
USE_PACKED_CF = False  # channels-first backward with the packed channel-pair layout instead of one channel per thread

WEIGHT_DTYPE_NAME = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def cdiv(a, b):
    return (a + b - 1) // b


def round_up(a, b):
    return cdiv(a, b) * b


def as_int64(value):
    return txl.Cast("int64", value)


def vector_width(C):
    """
    Widest bf16 vector (8, 4, 2 or 1 elements) that divides C.
    """
    for width in (8, 4, 2, 1):
        if C % width == 0:
            return width
    return 1


# Building blocks shared by the kernels.


def load_w_f32(dst, w, idx, w_dtype, valid):
    """
    dst = float32(w[idx]) if valid else 0.
    """
    txl.assign(dst, txl.float32(0.0))
    with txl.if_(valid), txl.then_():
        if w_dtype == "float32":
            txl.ptx.ld.global_.f32(dst, w.ptr_to([idx]))
        else:
            w_bits = txl.local_scalar("uint16")
            txl.ptx.ld.global_.b16(w_bits, w.ptr_to([idx]))
            if w_dtype == "bfloat16":
                txl.ptx.cvt.f32.bf16(dst, w_bits)
            else:
                txl.ptx.cvt.f32.f16(dst, w_bits)


def load_w_bf16(dst, w, idx, valid):
    """
    dst = the bf16 bits of w[idx] if valid else 0.
    """
    txl.assign(dst, txl.uint16(0))
    with txl.if_(valid), txl.then_():
        txl.ptx.ld.global_.b16(dst, w.ptr_to([idx]))


def load_shared_bf16(dst, s_ptr):
    """
    dst = float32 of the bf16 at s_ptr in shared memory.
    """
    bits = txl.local_scalar("uint16")
    txl.ptx["ld.shared.b16"](bits, s_ptr)
    txl.ptx.cvt.f32.bf16(dst, bits)


def shared_addr(ptr):
    return txl.cuda.cvta_generic_to_shared(ptr)


def tensor_map_ptr(tensor_map):
    return txl.reinterpret(txl.address_of(tensor_map), ty=txl.handle().ty)


def evict_last_policy():
    policy = txl.local_scalar("uint64")
    txl.ptx.createpolicy.fractional.L2__evict_last.b64(policy, txl.float32(1.0))
    return policy


def tma_load(dst, tensor_map, coord0, coord1, coord2, mbar, policy):
    name = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
    txl.ptx[name + ".L2::cache_hint"](dst, tensor_map, coord0, coord1, coord2, mbar, policy)


def tma_store(tensor_map, coord0, coord1, coord2, src):
    txl.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](tensor_map, coord0, coord1, coord2, src)


def named_barrier(barrier_idx, num_threads):
    txl.ptx.bar.sync(txl.uint32(barrier_idx), txl.uint32(num_threads))


def compute_y_taps(out, w_taps, x_vals, K):
    """
    out[j] = sum over k in order of w_taps[k] * x_vals[j + k], fp32 fma from 0.
    """
    acc = txl.local_scalar("float32")
    for j in range(TOKENS_PER_THREAD):
        txl.assign(acc, txl.float32(0.0))
        for k in range(K):
            txl.ptx.fma.rn.f32(acc, w_taps[k], x_vals[j + k], acc)
        txl.assign(out[j], acc)


def compute_dx_taps(out, w_taps, g_vals, K):
    """
    out[i] = sum over k in order of w_taps[k] * g_vals[i + K - 1 - k], fp32 fma from 0.
    """
    acc = txl.local_scalar("float32")
    for i in range(TOKENS_PER_THREAD):
        txl.assign(acc, txl.float32(0.0))
        for k in range(K):
            txl.ptx.fma.rn.f32(acc, w_taps[k], g_vals[i + K - 1 - k], acc)
        txl.assign(out[i], acc)


class LocalWindow:
    """
    array[start + k] window of a local array.
    """

    def __init__(self, array, start):
        self.array, self.start = array, start

    def __getitem__(self, k):
        return self.array[self.start + k]


def compute_dw_partials(dw_part, g_vals, x_vals, K, token_start, num_tokens):
    """
    dw_part[k] = sum over the tokens in order of g_vals[i] * x_vals[i + k], fp32 fma from 0.
    """
    for k in range(K):
        txl.assign(dw_part[k], txl.float32(0.0))
        for i in range(token_start, token_start + num_tokens):
            txl.ptx.fma.rn.f32(dw_part[k], g_vals[i], x_vals[i + k], dw_part[k])


def compute_dx_taps_bf16(out, w_taps, g_vals, K):
    """
    compute_dx_taps on bf16 w and g bits with fma.rn.f32.bf16.
    """
    acc = txl.local_scalar("float32")
    for i in range(TOKENS_PER_THREAD):
        txl.assign(acc, txl.float32(0.0))
        for k in range(K):
            txl.ptx["fma.rn.f32.bf16"](acc, w_taps[k], g_vals[i + K - 1 - k], acc)
        txl.assign(out[i], acc)


def compute_dw_partials_bf16(dw_part, g_vals, x_vals, K, token_start, num_tokens):
    """
    compute_dw_partials on bf16 g and x bits with fma.rn.f32.bf16.
    """
    for k in range(K):
        txl.assign(dw_part[k], txl.float32(0.0))
        for i in range(token_start, token_start + num_tokens):
            txl.ptx["fma.rn.f32.bf16"](dw_part[k], g_vals[i], x_vals[i + k], dw_part[k])


# Packed (f32x2) building blocks: one thread owns channels (ch, ch + 1), elements .x / .y.


def bf16x2_to_f32x2(word):
    lo = txl.reinterpret(word << txl.uint32(16), ty="float32")
    hi = txl.reinterpret(word & txl.uint32(0xFFFF0000), ty="float32")
    return txl.cuda.make_float2(lo, hi)


def float2_elem(value, elem):
    return txl.cuda.float2_x(value) if elem == 0 else txl.cuda.float2_y(value)


def load_w_pair(w_taps, w_pair, w, ch, C, K, w_dtype):
    """
    w_taps[k] = float2(w[ch][k], w[ch + 1][k]), channels past C read as zero.
    """
    for k in range(K):
        for elem in range(2):
            load_w_f32(w_pair[elem], w, (ch + elem) * K + k, w_dtype, ch + elem < C)
        txl.assign(w_taps[k], txl.cuda.make_float2(w_pair[0], w_pair[1]))


def compute_taps_packed(out, w_taps, vals, K, reverse):
    """
    out[i] = sum over k in order of w_taps[k] * vals[i + k] (reverse: vals[i + K - 1 - k]), fp32 fma
    from 0.
    """
    for i in range(PACKED_TOKENS_PER_THREAD):
        txl.assign(out[i], txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))
        for k in range(K):
            txl.ptx["fma.rn.f32x2"](out[i], w_taps[k], vals[i + (K - 1 - k if reverse else k)], out[i])


def release_stage(s_empty, stage, lane):
    """
    Consumer release of a ring stage: one arrival per thread, or one per warp after warp_sync.
    """
    if USE_WARP_ARRIVE:
        txl.cuda.warp_sync()
        with txl.if_(lane == 0), txl.then_():
            txl.ptx["mbarrier.arrive.shared.b64"](shared_addr(s_empty.ptr_to([stage])), txl.uint32(1))
    else:
        txl.ptx["mbarrier.arrive.shared.b64"](shared_addr(s_empty.ptr_to([stage])), txl.uint32(1))


def shuffle_idx_u32(dst, src, src_lane):
    txl.ptx.shfl_sync.idx.b32(dst, src, txl.Cast("uint32", src_lane), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))


def compute_dw_partials_packed(dw_part, g_vals, x_vals, K):
    """
    compute_dw_partials over the packed thread's PACKED_TOKENS_PER_THREAD tokens, f32x2.
    """
    for k in range(K):
        txl.assign(dw_part[k], txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))
        for i in range(PACKED_TOKENS_PER_THREAD):
            txl.ptx["fma.rn.f32x2"](dw_part[k], g_vals[i], x_vals[i + k], dw_part[k])


@functools.cache
def build_fwd_tma(C, K, w_dtype, shuffle_store):
    """
    TMA forward: 256-token tiles of a TILE_CHANNELS slice, y stored channels-first.
    """
    FWD_TILE_TOKENS = 256
    STAGE_ROWS = FWD_TILE_TOKENS + K - 1
    NUM_CH_TILES = cdiv(C, TILE_CHANNELS)
    # Two 259x64 input stages plus the 32 KiB output ring fit two CTAs per SM.
    NUM_STAGES = 2
    X_STAGE_BYTES = STAGE_ROWS * TILE_CHANNELS * 2
    NUM_CTAS = NUM_SMS * CTAS_PER_SM

    def encode_tensor_maps(args):
        x_head_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(x_head_map, "bfloat16", 3, txl.handle_add_byte_offset(args["x"].data, 0), C, args["L"], args["B"], C * 2, as_int64(args["L"]) * (C * 2), TILE_CHANNELS, FWD_TILE_TOKENS, 1, 1, 1, 1, 0, 0, 3, 0)
        x_halo_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(x_halo_map, "bfloat16", 3, txl.handle_add_byte_offset(args["x"].data, 0), C, args["L"], args["B"], C * 2, as_int64(args["L"]) * (C * 2), TILE_CHANNELS, K - 1, 1, 1, 1, 1, 0, 0, 3, 0)
        y_map = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(y_map, "bfloat16", 3, txl.handle_add_byte_offset(args["y"].data, 0), args["S"], C, args["B"], as_int64(args["S"]) * 2, as_int64(args["S"]) * (C * 2), 64, TILE_CHANNELS, 1, 1, 1, 1, 0, 3, 3, 0)
        return (x_head_map, x_halo_map, y_map)

    @txl.kernel()
    def conv_fwd_tma(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w: txl.Tensor((txl.int64(),), w_dtype, data_alignment=1), y: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), L: txl.int32, S: txl.int32, offset: txl.int32, B: txl.int32):
        tensor_maps = encode_tensor_maps(dict(x=x, y=y, L=L, S=S, B=B))
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_CTAS, block=(NUM_COMPUTE_WARPS + 1) * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=CTAS_PER_SM))
        x_head_map, x_halo_map, y_map = tensor_maps
        cta_idx = txl.cta_id()
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        num_token_tiles = (S + FWD_TILE_TOKENS - 1) // FWD_TILE_TOKENS
        num_tiles = num_token_tiles * NUM_CH_TILES * B

        smem = txl.smem_pool()
        s_y = smem.alloc([2 * 2 * TILE_CHANNELS * 64], "bfloat16", align=1024)
        s_x = smem.alloc([NUM_STAGES * STAGE_ROWS * TILE_CHANNELS], "bfloat16", align=128)
        s_full = smem.alloc([NUM_STAGES], "uint64", align=8)
        s_empty = smem.alloc([NUM_STAGES], "uint64", align=8)

        with txl.if_(tidx == 0), txl.then_():
            for stage_idx in range(NUM_STAGES):
                txl.ptx["mbarrier.init.shared.b64"](shared_addr(s_full.ptr_to([stage_idx])), txl.uint32(1))
                txl.ptx["mbarrier.init.shared.b64"](shared_addr(s_empty.ptr_to([stage_idx])), txl.uint32(EMPTY_ARRIVALS))
            txl.ptx["fence.mbarrier_init.release.cluster"]()
            txl.ptx["fence.proxy.async.shared::cta"]()
        named_barrier(0, (NUM_COMPUTE_WARPS + 1) * 32)

        stage = txl.local_scalar("int32", init=txl.int32(0))
        phase = txl.local_scalar("int32", init=txl.int32(0))
        tile_idx = txl.local_scalar("int32", init=cta_idx)

        def advance_tile():
            txl.assign(stage, stage + 1)
            with txl.if_(stage == NUM_STAGES), txl.then_():
                txl.assign(stage, txl.int32(0))
                txl.assign(phase, phase ^ 1)
            txl.assign(tile_idx, tile_idx + NUM_CTAS)

        def decode_tile(tile):
            # Token tile fastest: adjacent CTAs write adjacent token ranges of
            # the same channels-first output rows.
            token_tile_idx = tile % num_token_tiles
            rest = tile // num_token_tiles
            return rest % NUM_CH_TILES, token_tile_idx, rest // NUM_CH_TILES

        def load_loop():
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(x_head_map))
                txl.ptx.prefetch.tensormap(txl.address_of(x_halo_map))
                txl.ptx.prefetch.tensormap(txl.address_of(y_map))
                policy = evict_last_policy()
                with txl.while_(tile_idx < num_tiles):
                    ch_tile_idx, token_tile_idx, batch_idx = decode_tile(tile_idx)
                    txl.cuda.mbarrier_wait(s_empty.ptr_to([stage]), phase ^ 1)
                    # order the compute warps' generic reads of this stage before the TMA rewrite
                    txl.ptx["fence.proxy.async.shared::cta"]()
                    txl.ptx["mbarrier.arrive.expect_tx.shared.b64"](shared_addr(s_full.ptr_to([stage])), txl.uint32(X_STAGE_BYTES))
                    x_row = token_tile_idx * FWD_TILE_TOKENS + offset - (K - 1)
                    full_addr = shared_addr(s_full.ptr_to([stage]))
                    tma_load(shared_addr(s_x.ptr_to([stage * (STAGE_ROWS * TILE_CHANNELS)])), tensor_map_ptr(x_head_map), ch_tile_idx * TILE_CHANNELS, x_row, batch_idx, full_addr, policy)
                    tma_load(shared_addr(s_x.ptr_to([stage * (STAGE_ROWS * TILE_CHANNELS) + FWD_TILE_TOKENS * TILE_CHANNELS])), tensor_map_ptr(x_halo_map), ch_tile_idx * TILE_CHANNELS, x_row + FWD_TILE_TOKENS, batch_idx, full_addr, policy)
                    advance_tile()

        def compute_loop():
            # lane = channel pair (2 * lane, 2 * lane + 1), warp = token group of PACKED_TOKENS_PER_THREAD tokens
            pair_ch = 2 * lane
            token_group = warp
            w_taps = txl.alloc_local([K], "uint64")
            w_pair = txl.alloc_local([2], "float32")
            x_vals = txl.alloc_local([PACKED_TOKENS_PER_THREAD + K - 1], "uint64")
            out = txl.alloc_local([PACKED_TOKENS_PER_THREAD], "uint64")
            x_word = txl.local_scalar("uint32")
            if shuffle_store:
                packed_ch0 = txl.alloc_local([4], "uint32")
                packed_ch1 = txl.alloc_local([4], "uint32")
                shuffled = txl.alloc_local([4], "uint32")
                from_ch0 = txl.local_scalar("uint32")
                from_ch1 = txl.local_scalar("uint32")
            else:
                packed = txl.alloc_local([4], "uint32")
            if shuffle_store:
                shuffled_all = txl.alloc_local([16], "uint32")
            else:
                packed_all = txl.alloc_local([16], "uint32")
            use_bf16_fma = USE_BF16_FMA_FWD and w_dtype == "bfloat16"
            if use_bf16_fma:
                w_taps_bf16 = [txl.alloc_local([K], "uint16") for _ in range(2)]
                x_halves = [txl.alloc_local([PACKED_TOKENS_PER_THREAD + K - 1], "uint16") for _ in range(2)]
                acc_ch0 = txl.local_scalar("float32")
                acc_ch1 = txl.local_scalar("float32")
            tile_count = txl.local_scalar("int32", init=txl.int32(0))

            def load_weights(ch_tile_idx):
                if use_bf16_fma:
                    for elem in range(2):
                        for k in range(K):
                            load_w_bf16(w_taps_bf16[elem][k], w, (ch_tile_idx * TILE_CHANNELS + pair_ch + elem) * K + k, ch_tile_idx * TILE_CHANNELS + pair_ch + elem < C)
                else:
                    load_w_pair(w_taps, w_pair, w, ch_tile_idx * TILE_CHANNELS + pair_ch, C, K, w_dtype)

            def load_x_rows(x_stage_base):
                for j in range(PACKED_TOKENS_PER_THREAD + K - 1):
                    txl.ptx["ld.shared.b32"](x_word, s_x.ptr_to([x_stage_base + (token_group * PACKED_TOKENS_PER_THREAD + j) * TILE_CHANNELS + pair_ch]))
                    if use_bf16_fma:
                        txl.assign(x_halves[0][j], txl.Cast("uint16", x_word))
                        txl.assign(x_halves[1][j], txl.Cast("uint16", x_word >> txl.uint32(16)))
                    else:
                        txl.assign(x_vals[j], bf16x2_to_f32x2(x_word))

            def compute_y():
                if use_bf16_fma:
                    for i in range(PACKED_TOKENS_PER_THREAD):
                        txl.assign(acc_ch0, txl.float32(0.0))
                        txl.assign(acc_ch1, txl.float32(0.0))
                        for k in range(K):
                            txl.ptx["fma.rn.f32.bf16"](acc_ch0, w_taps_bf16[0][k], x_halves[0][i + k], acc_ch0)
                            txl.ptx["fma.rn.f32.bf16"](acc_ch1, w_taps_bf16[1][k], x_halves[1][i + k], acc_ch1)
                        txl.assign(out[i], txl.cuda.make_float2(acc_ch0, acc_ch1))
                else:
                    compute_taps_packed(out, w_taps, x_vals, K, False)

            def stage_y(y_buf, token_half):
                """
                Convert out to bf16 and write it to staging buffer y_buf, token chunk chunk_idx (8
                tokens) of half token_half at physical chunk chunk_idx ^ (ch % 8) (128B swizzle).
                """
                if CONVERT_BEFORE_STAGING:
                    # convert (and shuffle) while thread 0 waits for the old bulk store to drain
                    if shuffle_store:
                        for chunk in range(PACKED_TOKENS_PER_THREAD // 8):
                            for word in range(4):
                                txl.ptx["cvt.rn.bf16x2.f32"](packed_ch0[word], float2_elem(out[chunk * 8 + 2 * word + 1], 0), float2_elem(out[chunk * 8 + 2 * word], 0))
                                txl.ptx["cvt.rn.bf16x2.f32"](packed_ch1[word], float2_elem(out[chunk * 8 + 2 * word + 1], 1), float2_elem(out[chunk * 8 + 2 * word], 1))
                            for ch_group in range(2):
                                src_lane = ch_group * 16 + (lane >> 1)
                                for word in range(4):
                                    shuffle_idx_u32(from_ch0, packed_ch0[word], src_lane)
                                    shuffle_idx_u32(from_ch1, packed_ch1[word], src_lane)
                                    txl.assign(shuffled_all[(chunk * 2 + ch_group) * 4 + word], txl.Select((lane & 1) == 0, from_ch0, from_ch1))
                        named_barrier(1, NUM_COMPUTE_THREADS)
                        for chunk in range(PACKED_TOKENS_PER_THREAD // 8):
                            for ch_group in range(2):
                                ch = ch_group * 32 + lane
                                chunk_idx = (token_group % 4) * (PACKED_TOKENS_PER_THREAD // 8) + chunk
                                y_off = ((y_buf * 2 + token_half) * TILE_CHANNELS + ch) * 64 + ((chunk_idx ^ (ch % 8)) * 8)
                                word_start = (chunk * 2 + ch_group) * 4
                                txl.ptx["st.shared.v4.b32"](s_y.ptr_to([y_off]), shuffled_all[word_start], shuffled_all[word_start + 1], shuffled_all[word_start + 2], shuffled_all[word_start + 3])
                    else:
                        for elem in range(2):
                            for chunk in range(PACKED_TOKENS_PER_THREAD // 8):
                                for word in range(4):
                                    txl.ptx["cvt.rn.bf16x2.f32"](packed_all[(elem * 2 + chunk) * 4 + word], float2_elem(out[chunk * 8 + 2 * word + 1], elem), float2_elem(out[chunk * 8 + 2 * word], elem))
                        named_barrier(1, NUM_COMPUTE_THREADS)
                        for elem in range(2):
                            ch = pair_ch + elem
                            for chunk in range(PACKED_TOKENS_PER_THREAD // 8):
                                chunk_idx = (token_group % 4) * (PACKED_TOKENS_PER_THREAD // 8) + chunk
                                y_off = ((y_buf * 2 + token_half) * TILE_CHANNELS + ch) * 64 + ((chunk_idx ^ (ch % 8)) * 8)
                                word_start = (elem * 2 + chunk) * 4
                                txl.ptx["st.shared.v4.b32"](s_y.ptr_to([y_off]), packed_all[word_start], packed_all[word_start + 1], packed_all[word_start + 2], packed_all[word_start + 3])
                else:
                    named_barrier(1, NUM_COMPUTE_THREADS)
                    if shuffle_store:
                        for chunk in range(PACKED_TOKENS_PER_THREAD // 8):
                            for word in range(4):
                                txl.ptx["cvt.rn.bf16x2.f32"](packed_ch0[word], float2_elem(out[chunk * 8 + 2 * word + 1], 0), float2_elem(out[chunk * 8 + 2 * word], 0))
                                txl.ptx["cvt.rn.bf16x2.f32"](packed_ch1[word], float2_elem(out[chunk * 8 + 2 * word + 1], 1), float2_elem(out[chunk * 8 + 2 * word], 1))
                            for ch_group in range(2):
                                src_lane = ch_group * 16 + (lane >> 1)
                                for word in range(4):
                                    shuffle_idx_u32(from_ch0, packed_ch0[word], src_lane)
                                    shuffle_idx_u32(from_ch1, packed_ch1[word], src_lane)
                                    txl.assign(shuffled[word], txl.Select((lane & 1) == 0, from_ch0, from_ch1))
                                ch = ch_group * 32 + lane
                                chunk_idx = (token_group % 4) * (PACKED_TOKENS_PER_THREAD // 8) + chunk
                                y_off = ((y_buf * 2 + token_half) * TILE_CHANNELS + ch) * 64 + ((chunk_idx ^ (ch % 8)) * 8)
                                txl.ptx["st.shared.v4.b32"](s_y.ptr_to([y_off]), shuffled[0], shuffled[1], shuffled[2], shuffled[3])
                    else:
                        for elem in range(2):
                            ch = pair_ch + elem
                            for chunk in range(PACKED_TOKENS_PER_THREAD // 8):
                                chunk_idx = (token_group % 4) * (PACKED_TOKENS_PER_THREAD // 8) + chunk
                                for word in range(4):
                                    txl.ptx["cvt.rn.bf16x2.f32"](packed[word], float2_elem(out[chunk * 8 + 2 * word + 1], elem), float2_elem(out[chunk * 8 + 2 * word], elem))
                                y_off = ((y_buf * 2 + token_half) * TILE_CHANNELS + ch) * 64 + ((chunk_idx ^ (ch % 8)) * 8)
                                txl.ptx["st.shared.v4.b32"](s_y.ptr_to([y_off]), packed[0], packed[1], packed[2], packed[3])

            def store_y(sub_tile, y_buf, ch_tile_idx, token_tile_idx, batch_idx):
                txl.ptx["fence.proxy.async.shared::cta"]()
                named_barrier(1, NUM_COMPUTE_THREADS)
                with txl.if_(tidx == 0), txl.then_():
                    for store_half in range(2):
                        tma_store(tensor_map_ptr(y_map), token_tile_idx * FWD_TILE_TOKENS + sub_tile * T + store_half * 64, ch_tile_idx * TILE_CHANNELS, batch_idx, shared_addr(s_y.ptr_to([(y_buf * 2 + store_half) * TILE_CHANNELS * 64])))
                    txl.ptx["cp.async.bulk.commit_group"]()

            with txl.while_(tile_idx < num_tiles):
                ch_tile_idx, token_tile_idx, batch_idx = decode_tile(tile_idx)
                load_weights(ch_tile_idx)
                txl.cuda.mbarrier_wait(s_full.ptr_to([stage]), phase)
                for sub_tile in range(2):
                    x_stage_base = stage * (STAGE_ROWS * TILE_CHANNELS) + sub_tile * T * TILE_CHANNELS
                    load_x_rows(x_stage_base)
                    if sub_tile == 1:
                        release_stage(s_empty, stage, lane)
                    compute_y()
                    y_buf = tile_count % 2
                    token_half = token_group // 4
                    with txl.if_(tidx == 0), txl.then_():
                        txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(1))
                    stage_y(y_buf, token_half)
                    store_y(sub_tile, y_buf, ch_tile_idx, token_tile_idx, batch_idx)
                    txl.assign(tile_count, tile_count + 1)
                advance_tile()
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx["cp.async.bulk.wait_group"](txl.int32(0))

        with txl.if_(warp == PRODUCER_WARP):
            with txl.then_():
                load_loop()
            with txl.else_():
                compute_loop()

    return entry_function(conv_fwd_tma.compile(TARGET))


def cluster_sync():
    txl.ptx.barrier.cluster.arrive.release()
    txl.ptx.barrier.cluster.wait.acquire()


def plan_bwd_slices(C, channels_first):
    """
    Channel slices of the TMA backward, one per CLUSTER_SIZE-CTA cluster: (num_clusters,
    base_units, step_units, num_wide), the first num_wide clusters (base_units + step_units)
    8-channel units wide and the rest base_units. For C=8192 the plan uses 64/48-channel slices:
    both row widths are 32-byte-sector aligned, unlike the nearly even 56/48 split. Other valid
    channel counts retain the nearly even plan.
    """
    num_units = C // 8
    # Contiguous g keeps fixed 64-channel blocks; only channels-first g uses the 296-CTA balanced
    # partition.
    if C == 8192 and not channels_first:
        return 128, 8, 1, 0
    num_clusters = max(min(NUM_SMS * CTAS_PER_SM // CLUSTER_SIZE, num_units), cdiv(num_units, TILE_CHANNELS // 8))
    if C == 8192 and num_clusters == NUM_SMS:
        return num_clusters, 6, 2, 68
    if C == 512:
        return 10, 6, 2, 2
    base_units, num_wide = divmod(num_units, num_clusters)
    return num_clusters, base_units, 1, num_wide


class BwdContext:
    """
    What the TMA backward's compute loops share with the kernel entry: the shared-memory buffers
    and ring barriers, the ring and row-tile counters, the CTA's channel slice and its thread
    coordinates, and the entry's store_dx / advance_tile steps.
    """

    def __init__(self, s_g, s_x, s_dx, s_dw_groups, w, w_dtype, C, K, ch_start, num_ch, g_shift, tidx, warp, lane, s_full, s_empty, stage, phase, row_tile_idx, num_row_tiles, num_token_tiles, g_stage_size, x_stage_size, width_max, store_dx, advance_tile):
        self.s_g, self.s_x, self.s_dx, self.s_dw_groups = s_g, s_x, s_dx, s_dw_groups
        self.w, self.w_dtype, self.C, self.K = w, w_dtype, C, K
        self.ch_start, self.num_ch, self.g_shift = ch_start, num_ch, g_shift
        self.tidx, self.warp, self.lane = tidx, warp, lane
        self.s_full, self.s_empty, self.stage, self.phase = s_full, s_empty, stage, phase
        self.row_tile_idx, self.num_row_tiles, self.num_token_tiles = row_tile_idx, num_row_tiles, num_token_tiles
        self.g_stage_size, self.x_stage_size, self.width_max = g_stage_size, x_stage_size, width_max
        self.store_dx, self.advance_tile = store_dx, advance_tile


@functools.cache
def build_bwd_tma(C, K, w_dtype, channels_first):
    """
    TMA backward: 2-CTA clusters over channel slices, dx stored with TMA, dw summed in order.
    """
    STAGE_ROWS = T + K - 1
    NUM_CLUSTERS, base_units, step_units, num_wide = plan_bwd_slices(C, channels_first)
    NUM_CTAS = NUM_CLUSTERS * CLUSTER_SIZE
    WIDTH_WIDE, WIDTH_NARROW = (base_units + step_units) * 8, base_units * 8  # slice widths: clusters < num_wide, the others
    WIDTHS = [WIDTH_WIDE, WIDTH_NARROW] if num_wide else [WIDTH_NARROW]
    WIDTH_MAX = WIDTHS[0]
    NUM_STAGES = NUM_STAGES_BWD
    G_STAGE_SIZE = round_up(WIDTH_MAX * CHANNELS_FIRST_PITCH if channels_first else STAGE_ROWS * WIDTH_MAX, 64)  # elements of one g stage (128B-aligned stride)
    X_STAGE_SIZE = round_up(STAGE_ROWS * WIDTH_MAX, 64)
    assert 3 * TOKENS_PER_THREAD + 48 <= CHANNELS_FIRST_PITCH and TOKENS_PER_THREAD + K - 1 + 7 <= 48 and WIDTH_MAX <= TILE_CHANNELS

    def stage_tx_bytes(width):
        return ((width * CHANNELS_FIRST_PITCH if channels_first else STAGE_ROWS * width) + STAGE_ROWS * width) * 2

    def encode_tensor_maps(args):
        tensor_maps = []
        for width in WIDTHS:
            g_map = txl.stack_alloca("tensormap", 1)
            if channels_first:
                txl.cu_tensor_map_encode_tiled(g_map, "bfloat16", 3, txl.handle_add_byte_offset(args["g"].data, 0), args["S"], C, args["B"], as_int64(args["S"]) * 2, as_int64(args["S"]) * (C * 2), CHANNELS_FIRST_PITCH, width, 1, 1, 1, 1, 0, 0, 3, 0)
            else:
                txl.cu_tensor_map_encode_tiled(g_map, "bfloat16", 3, txl.handle_add_byte_offset(args["g"].data, 0), C, args["S"], args["B"], C * 2, as_int64(args["S"]) * (C * 2), width, STAGE_ROWS, 1, 1, 1, 1, 0, 0, 3, 0)
            x_map = txl.stack_alloca("tensormap", 1)
            txl.cu_tensor_map_encode_tiled(x_map, "bfloat16", 3, txl.handle_add_byte_offset(args["x"].data, 0), C, args["L"], args["B"], C * 2, as_int64(args["L"]) * (C * 2), width, STAGE_ROWS, 1, 1, 1, 1, 0, 0, 3, 0)
            dx_map = txl.stack_alloca("tensormap", 1)
            txl.cu_tensor_map_encode_tiled(dx_map, "bfloat16", 3, txl.handle_add_byte_offset(args["dx"].data, 0), C, args["L"], args["B"], C * 2, as_int64(args["L"]) * (C * 2), width, T, 1, 1, 1, 1, 0, 0, 3, 0)
            tensor_maps += [g_map, x_map, dx_map]
        return tuple(tensor_maps)

    @txl.kernel()
    def conv_bwd_tma(g: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w: txl.Tensor((txl.int64(),), w_dtype, data_alignment=1), dx: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dw: txl.Tensor((txl.int64(),), w_dtype, data_alignment=1), L: txl.int32, S: txl.int32, offset: txl.int32, B: txl.int32):
        tensor_maps = encode_tensor_maps(dict(g=g, x=x, dx=dx, L=L, S=S, B=B))
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_CTAS, block=(NUM_COMPUTE_WARPS + 1) * 32, cluster=(CLUSTER_SIZE,), preferred_cluster=[CLUSTER_SIZE]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=CTAS_PER_SM))
        maps_by_width = [tensor_maps[3 * i:3 * i + 3] for i in range(len(WIDTHS))]
        cta_idx = txl.cta_id()
        cluster_rank = txl.Cast("int32", txl.cuda.mov_sreg(32, "cluster_ctarank"))
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        cluster_idx = cta_idx // CLUSTER_SIZE
        if num_wide:
            is_wide = cluster_idx < num_wide
            ch_start = 8 * (cluster_idx * base_units + txl.min(cluster_idx, txl.int32(num_wide)) * step_units)
            num_ch = txl.Select(is_wide, txl.int32(WIDTH_WIDE), txl.int32(WIDTH_NARROW))
        else:
            ch_start = 8 * base_units * cluster_idx
            num_ch = txl.int32(WIDTH_NARROW)

        def for_slice_width(body):
            """
            body(width, maps of that width) under the slice-width branch of this CTA.
            """
            if num_wide:
                with txl.if_(is_wide):
                    with txl.then_():
                        body(WIDTH_WIDE, maps_by_width[0])
                    with txl.else_():
                        body(WIDTH_NARROW, maps_by_width[1])
            else:
                body(WIDTH_NARROW, maps_by_width[0])

        num_token_tiles = (L + T - 1) // T
        num_row_tiles = num_token_tiles * B  # row tiles of this channel slice, row_tile_idx = batch_idx * num_token_tiles + token_tile_idx
        # TMA needs a 16B-aligned inner coordinate: (token_tile_idx * T - offset) minus its 8-aligned floor
        g_shift = (8 - (offset & 7)) & 7

        smem = txl.smem_pool()
        s_dx = smem.alloc([NUM_DX_BUFS * T * WIDTH_MAX], "bfloat16", align=1024)
        s_g = smem.alloc([NUM_STAGES * G_STAGE_SIZE], "bfloat16", align=128)
        s_x = smem.alloc([NUM_STAGES * X_STAGE_SIZE], "bfloat16", align=128)
        s_dw_groups = smem.alloc([NUM_TOKEN_GROUPS * K * TILE_CHANNELS], "float32", align=16)
        s_dw = smem.alloc([K * TILE_CHANNELS], "float32", align=16)
        s_full = smem.alloc([NUM_STAGES], "uint64", align=8)
        s_empty = smem.alloc([NUM_STAGES], "uint64", align=8)

        with txl.if_(tidx == 0), txl.then_():
            for stage_idx in range(NUM_STAGES):
                txl.ptx["mbarrier.init.shared.b64"](shared_addr(s_full.ptr_to([stage_idx])), txl.uint32(1))
                txl.ptx["mbarrier.init.shared.b64"](shared_addr(s_empty.ptr_to([stage_idx])), txl.uint32(EMPTY_ARRIVALS))
            txl.ptx["fence.mbarrier_init.release.cluster"]()
            txl.ptx["fence.proxy.async.shared::cta"]()
        named_barrier(0, (NUM_COMPUTE_WARPS + 1) * 32)

        stage = txl.local_scalar("int32", init=txl.int32(0))
        phase = txl.local_scalar("int32", init=txl.int32(0))
        row_tile_idx = txl.local_scalar("int32", init=cluster_rank)

        def advance_tile():
            txl.assign(stage, stage + 1)
            with txl.if_(stage == NUM_STAGES), txl.then_():
                txl.assign(stage, txl.int32(0))
                txl.assign(phase, phase ^ 1)
            txl.assign(row_tile_idx, row_tile_idx + CLUSTER_SIZE)

        def load_loop():
            with txl.if_(lane == 0), txl.then_():
                for maps in maps_by_width:
                    for tensor_map in maps:
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map))
                policy = evict_last_policy()
                with txl.while_(row_tile_idx < num_row_tiles):
                    token_tile_idx = row_tile_idx % num_token_tiles
                    batch_idx = row_tile_idx // num_token_tiles
                    txl.cuda.mbarrier_wait(s_empty.ptr_to([stage]), phase ^ 1)
                    # order the compute warps' generic reads of this stage before the TMA rewrite
                    txl.ptx["fence.proxy.async.shared::cta"]()
                    full_addr = shared_addr(s_full.ptr_to([stage]))

                    def issue_loads(width, maps):
                        g_map, x_map, _ = maps
                        txl.ptx["mbarrier.arrive.expect_tx.shared.b64"](full_addr, txl.uint32(stage_tx_bytes(width)))
                        if channels_first:
                            # [width ch][CHANNELS_FIRST_PITCH tok] box from the 16B-aligned token below
                            # token_tile_idx * T - offset: tile token j + g_shift = g row token_tile_idx * T - offset + j
                            g_coords = (token_tile_idx * T - offset - g_shift, ch_start, batch_idx)
                        else:
                            g_coords = (ch_start, token_tile_idx * T - offset, batch_idx)
                        tma_load(shared_addr(s_g.ptr_to([stage * G_STAGE_SIZE])), tensor_map_ptr(g_map), g_coords[0], g_coords[1], g_coords[2], full_addr, policy)
                        tma_load(shared_addr(s_x.ptr_to([stage * X_STAGE_SIZE])), tensor_map_ptr(x_map), ch_start, token_tile_idx * T - (K - 1), batch_idx, full_addr, policy)

                    for_slice_width(issue_loads)
                    advance_tile()

        def store_dx(dx_buf, token_tile_idx, batch_idx):
            with txl.if_(tidx == 0), txl.then_():
                for_slice_width(lambda width, maps: tma_store(tensor_map_ptr(maps[2]), ch_start, token_tile_idx * T, batch_idx, shared_addr(s_dx.ptr_to([dx_buf * T * WIDTH_MAX]))))
                txl.ptx["cp.async.bulk.commit_group"]()

        def reduce_dw_token_groups():
            """
            s_dw = the token groups' dw partials summed in order.
            """
            named_barrier(1, NUM_COMPUTE_THREADS)
            acc_dw = txl.local_scalar("float32")
            partial = txl.local_scalar("float32")
            for i in range(cdiv(K * TILE_CHANNELS, NUM_COMPUTE_THREADS)):
                idx = i * NUM_COMPUTE_THREADS + tidx
                with txl.if_(idx < K * TILE_CHANNELS), txl.then_():
                    txl.assign(acc_dw, txl.float32(0.0))
                    for group_idx in range(NUM_TOKEN_GROUPS):
                        txl.ptx["ld.shared.f32"](partial, s_dw_groups.ptr_to([group_idx * K * TILE_CHANNELS + idx]))
                        txl.assign(acc_dw, acc_dw + partial)
                    txl.ptx["st.shared.f32"](s_dw.ptr_to([idx]), acc_dw)

        def compute_loop():
            ctx = BwdContext(s_g, s_x, s_dx, s_dw_groups, w, w_dtype, C, K, ch_start, num_ch, g_shift, tidx, warp, lane, s_full, s_empty, stage, phase, row_tile_idx, num_row_tiles, num_token_tiles, G_STAGE_SIZE, X_STAGE_SIZE, WIDTH_MAX, store_dx, advance_tile)
            if channels_first:
                (bwd_compute_loop_cf_packed if USE_PACKED_CF else bwd_compute_loop_cf)(ctx)
            else:
                bwd_compute_loop_cl(ctx)
            reduce_dw_token_groups()
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx["cp.async.bulk.wait_group"](txl.int32(0))

        def reduce_dw_cluster():
            """
            dw of the slice = the cluster's ranks' s_dw summed in rank order, by rank 0.
            """
            with txl.if_((cluster_rank == 0) & (tidx < NUM_COMPUTE_THREADS)), txl.then_():
                acc_dw = txl.local_scalar("float32")
                partial = txl.local_scalar("float32")
                remote_addr = txl.local_scalar("uint32")
                for i in range(cdiv(K * TILE_CHANNELS, NUM_COMPUTE_THREADS)):
                    idx = i * NUM_COMPUTE_THREADS + tidx
                    tap = idx // TILE_CHANNELS
                    ch = idx % TILE_CHANNELS
                    with txl.if_((idx < K * TILE_CHANNELS) & (ch < num_ch)), txl.then_():
                        txl.assign(acc_dw, txl.float32(0.0))
                        for rank_idx in range(CLUSTER_SIZE):
                            txl.ptx.mapa.shared__cluster.u32(remote_addr, shared_addr(s_dw.ptr_to([idx])), txl.uint32(rank_idx))
                            txl.ptx["ld.shared::cluster.f32"](partial, remote_addr)
                            txl.assign(acc_dw, acc_dw + partial)
                        store_dw_elem(dw, (ch_start + ch) * K + tap, acc_dw, w_dtype)

        with txl.if_(warp == PRODUCER_WARP):
            with txl.then_():
                load_loop()
            with txl.else_():
                compute_loop()
        cluster_sync()
        reduce_dw_cluster()
        cluster_sync()

    return entry_function(conv_bwd_tma.compile(TARGET))


def bwd_compute_loop_cl(ctx):
    """
    Compute warps for channels-last g: lane = channel pair pair_ch, warp = token group of
    PACKED_TOKENS_PER_THREAD tokens, packed f32x2 arithmetic. Tiles are [rows][num_ch] (pitch
    num_ch).
    """
    if USE_BF16_FMA_BWD and ctx.w_dtype == "bfloat16":
        return bwd_compute_loop_cl_bf16(ctx)
    s_g, s_x, s_dx, s_dw_groups = ctx.s_g, ctx.s_x, ctx.s_dx, ctx.s_dw_groups
    w, w_dtype, C, K = ctx.w, ctx.w_dtype, ctx.C, ctx.K
    ch_start, num_ch, tidx, warp, lane = ctx.ch_start, ctx.num_ch, ctx.tidx, ctx.warp, ctx.lane
    s_full, s_empty, stage, phase = ctx.s_full, ctx.s_empty, ctx.stage, ctx.phase
    row_tile_idx, num_row_tiles, num_token_tiles = ctx.row_tile_idx, ctx.num_row_tiles, ctx.num_token_tiles
    g_stage_size, x_stage_size, width_max = ctx.g_stage_size, ctx.x_stage_size, ctx.width_max
    pair_ch = 2 * lane
    token_group = warp
    ch_ok = pair_ch < num_ch
    w_taps = txl.alloc_local([K], "uint64")
    w_pair = txl.alloc_local([2], "float32")
    g_vals = txl.alloc_local([PACKED_TOKENS_PER_THREAD + K - 1], "uint64")
    x_vals = txl.alloc_local([PACKED_TOKENS_PER_THREAD + K - 1], "uint64")
    out = txl.alloc_local([PACKED_TOKENS_PER_THREAD], "uint64")
    dw_part = txl.alloc_local([K], "uint64")
    acc_dw = txl.alloc_local([K], "uint64")
    dx_words = txl.alloc_local([PACKED_TOKENS_PER_THREAD], "uint32")
    g_word = txl.local_scalar("uint32")
    x_word = txl.local_scalar("uint32")
    load_w_pair(w_taps, w_pair, w, ch_start + pair_ch, C, K, w_dtype)
    for k in range(K):
        txl.assign(acc_dw[k], txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))
    tile_count = txl.local_scalar("int32", init=txl.int32(0))
    with txl.while_(row_tile_idx < num_row_tiles):
        token_tile_idx = row_tile_idx % num_token_tiles
        batch_idx = row_tile_idx // num_token_tiles
        txl.cuda.mbarrier_wait(s_full.ptr_to([stage]), phase)
        # lanes past the slice width read (and discard) the slice's last pair: no reads past the tile
        read_ch = txl.min(pair_ch, num_ch - 2)
        g_base = stage * g_stage_size + read_ch
        x_base = stage * x_stage_size + read_ch
        for j in range(PACKED_TOKENS_PER_THREAD + K - 1):
            txl.ptx["ld.shared.b32"](g_word, s_g.ptr_to([g_base + (token_group * PACKED_TOKENS_PER_THREAD + j) * num_ch]))
            txl.assign(g_vals[j], bf16x2_to_f32x2(g_word))
            txl.ptx["ld.shared.b32"](x_word, s_x.ptr_to([x_base + (token_group * PACKED_TOKENS_PER_THREAD + j) * num_ch]))
            txl.assign(x_vals[j], bf16x2_to_f32x2(x_word))
        release_stage(s_empty, stage, lane)
        compute_taps_packed(out, w_taps, g_vals, K, True)
        compute_dw_partials_packed(dw_part, g_vals, x_vals, K)
        for k in range(K):
            txl.ptx["add.rn.f32x2"](acc_dw[k], acc_dw[k], dw_part[k])
        dx_buf = tile_count % NUM_DX_BUFS
        # thread 0 owns the bulk-store group: its read-completion wait overlaps the others' conversion
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(NUM_DX_BUFS - 1))
        for i in range(PACKED_TOKENS_PER_THREAD):
            txl.ptx["cvt.rn.bf16x2.f32"](dx_words[i], float2_elem(out[i], 1), float2_elem(out[i], 0))
        named_barrier(1, NUM_COMPUTE_THREADS)
        with txl.if_(ch_ok), txl.then_():
            for i in range(PACKED_TOKENS_PER_THREAD):
                txl.ptx["st.shared.b32"](s_dx.ptr_to([dx_buf * T * width_max + (token_group * PACKED_TOKENS_PER_THREAD + i) * num_ch + pair_ch]), dx_words[i])
        txl.ptx["fence.proxy.async.shared::cta"]()
        named_barrier(1, NUM_COMPUTE_THREADS)
        ctx.store_dx(dx_buf, token_tile_idx, batch_idx)
        txl.assign(tile_count, tile_count + 1)
        ctx.advance_tile()
    # per (token group, tap, channel); summed over the groups in order by the caller
    for k in range(K):
        for elem in range(2):
            txl.ptx["st.shared.f32"](s_dw_groups.ptr_to([(token_group * K + k) * TILE_CHANNELS + pair_ch + elem]), float2_elem(acc_dw[k], elem))


def bwd_compute_loop_cl_bf16(ctx):
    """
    Packed channels-last layout (lane = channel pair, warp = PACKED_TOKENS_PER_THREAD-token group)
    with scalar fma.rn.f32.bf16 on the staged BF16 halves: no BF16->FP32 conversions, same
    per-channel order.
    """
    s_g, s_x, s_dx, s_dw_groups = ctx.s_g, ctx.s_x, ctx.s_dx, ctx.s_dw_groups
    w, C, K = ctx.w, ctx.C, ctx.K
    ch_start, num_ch, tidx, warp, lane = ctx.ch_start, ctx.num_ch, ctx.tidx, ctx.warp, ctx.lane
    s_full, s_empty, stage, phase = ctx.s_full, ctx.s_empty, ctx.stage, ctx.phase
    row_tile_idx, num_row_tiles, num_token_tiles = ctx.row_tile_idx, ctx.num_row_tiles, ctx.num_token_tiles
    g_stage_size, x_stage_size, width_max = ctx.g_stage_size, ctx.x_stage_size, ctx.width_max
    pair_ch = 2 * lane
    token_group = warp
    ch_ok = pair_ch < num_ch
    num_vals = PACKED_TOKENS_PER_THREAD + K - 1
    w_taps = [txl.alloc_local([K], "uint16") for _ in range(2)]
    g_vals = [txl.alloc_local([num_vals], "uint16") for _ in range(2)]
    x_vals = [txl.alloc_local([num_vals], "uint16") for _ in range(2)]
    out = [txl.alloc_local([PACKED_TOKENS_PER_THREAD], "float32") for _ in range(2)]
    dw_part = [txl.alloc_local([K], "float32") for _ in range(2)]
    acc_dw = [txl.alloc_local([K], "float32") for _ in range(2)]
    dx_words = txl.alloc_local([PACKED_TOKENS_PER_THREAD], "uint32")
    g_word = txl.local_scalar("uint32")
    x_word = txl.local_scalar("uint32")
    for elem in range(2):
        for k in range(K):
            load_w_bf16(w_taps[elem][k], w, (ch_start + pair_ch + elem) * K + k, ch_start + pair_ch + elem < C)
            txl.assign(acc_dw[elem][k], txl.float32(0.0))
    tile_count = txl.local_scalar("int32", init=txl.int32(0))
    with txl.while_(row_tile_idx < num_row_tiles):
        token_tile_idx = row_tile_idx % num_token_tiles
        batch_idx = row_tile_idx // num_token_tiles
        txl.cuda.mbarrier_wait(s_full.ptr_to([stage]), phase)
        read_ch = txl.min(pair_ch, num_ch - 2)
        g_base = stage * g_stage_size + read_ch
        x_base = stage * x_stage_size + read_ch
        for j in range(num_vals):
            txl.ptx["ld.shared.b32"](g_word, s_g.ptr_to([g_base + (token_group * PACKED_TOKENS_PER_THREAD + j) * num_ch]))
            txl.assign(g_vals[0][j], txl.Cast("uint16", g_word))
            txl.assign(g_vals[1][j], txl.Cast("uint16", g_word >> txl.uint32(16)))
            txl.ptx["ld.shared.b32"](x_word, s_x.ptr_to([x_base + (token_group * PACKED_TOKENS_PER_THREAD + j) * num_ch]))
            txl.assign(x_vals[0][j], txl.Cast("uint16", x_word))
            txl.assign(x_vals[1][j], txl.Cast("uint16", x_word >> txl.uint32(16)))
        release_stage(s_empty, stage, lane)
        for elem in range(2):
            for i in range(PACKED_TOKENS_PER_THREAD):
                txl.assign(out[elem][i], txl.float32(0.0))
                for k in range(K):
                    txl.ptx["fma.rn.f32.bf16"](out[elem][i], w_taps[elem][k], g_vals[elem][i + K - 1 - k], out[elem][i])
        for elem in range(2):
            for k in range(K):
                txl.assign(dw_part[elem][k], txl.float32(0.0))
                for i in range(PACKED_TOKENS_PER_THREAD):
                    txl.ptx["fma.rn.f32.bf16"](dw_part[elem][k], g_vals[elem][i], x_vals[elem][i + k], dw_part[elem][k])
                txl.assign(acc_dw[elem][k], acc_dw[elem][k] + dw_part[elem][k])
        dx_buf = tile_count % NUM_DX_BUFS
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(NUM_DX_BUFS - 1))
        for i in range(PACKED_TOKENS_PER_THREAD):
            txl.ptx["cvt.rn.bf16x2.f32"](dx_words[i], out[1][i], out[0][i])
        named_barrier(1, NUM_COMPUTE_THREADS)
        with txl.if_(ch_ok), txl.then_():
            for i in range(PACKED_TOKENS_PER_THREAD):
                txl.ptx["st.shared.b32"](s_dx.ptr_to([dx_buf * T * width_max + (token_group * PACKED_TOKENS_PER_THREAD + i) * num_ch + pair_ch]), dx_words[i])
        txl.ptx["fence.proxy.async.shared::cta"]()
        named_barrier(1, NUM_COMPUTE_THREADS)
        ctx.store_dx(dx_buf, token_tile_idx, batch_idx)
        txl.assign(tile_count, tile_count + 1)
        ctx.advance_tile()
    for k in range(K):
        for elem in range(2):
            txl.ptx["st.shared.f32"](s_dw_groups.ptr_to([(token_group * K + k) * TILE_CHANNELS + pair_ch + elem]), acc_dw[elem][k])


def bwd_compute_loop_cf(ctx):
    """
    Compute warps for g in the output's channels-first layout, read in place (no transpose).

    Thread = channel ch, tokens [token_group * TOKENS_PER_THREAD, +TOKENS_PER_THREAD) = dw token
    groups 2 token_group and 2 token_group + 1; g is read along tokens with 16B loads from the
    [num_ch][CHANNELS_FIRST_PITCH] tile (48 tokens from the aligned token_group *
    TOKENS_PER_THREAD, then shifted by the runtime g_shift in registers: two word selects and a
    byte permute), x per token from the channels-last tile. Same per-channel arithmetic order as
    the packed channels-last path.
    """
    s_g, s_x, s_dx, s_dw_groups = ctx.s_g, ctx.s_x, ctx.s_dx, ctx.s_dw_groups
    w, w_dtype, C, K = ctx.w, ctx.w_dtype, ctx.C, ctx.K
    ch_start, num_ch, g_shift = ctx.ch_start, ctx.num_ch, ctx.g_shift
    tidx, warp, lane = ctx.tidx, ctx.warp, ctx.lane
    s_full, s_empty, stage, phase = ctx.s_full, ctx.s_empty, ctx.stage, ctx.phase
    row_tile_idx, num_row_tiles, num_token_tiles = ctx.row_tile_idx, ctx.num_row_tiles, ctx.num_token_tiles
    g_stage_size, x_stage_size, width_max = ctx.g_stage_size, ctx.x_stage_size, ctx.width_max
    ch = lane + 32 * (warp % 2)
    token_group = warp // 2
    ch_ok = ch < num_ch
    NUM_G_WORDS = 24  # g words per thread: tokens [token_group * TOKENS_PER_THREAD, +48)
    NUM_SHIFTED_WORDS = cdiv(TOKENS_PER_THREAD + K - 1, 2)  # words after the shift
    use_bf16_fma = w_dtype == "bfloat16"
    w_taps = txl.alloc_local([K], "uint16" if use_bf16_fma else "float32")
    g_words = txl.alloc_local([NUM_G_WORDS], "uint32")
    g_words_shift4 = txl.alloc_local([NUM_SHIFTED_WORDS + 2], "uint32")
    g_words_shift2 = txl.alloc_local([NUM_SHIFTED_WORDS + 1], "uint32")
    is_shift4 = (g_shift & 4) != 0
    is_shift2 = (g_shift & 2) != 0
    prmt_selector = txl.Select((g_shift & 1) != 0, txl.uint32(0x5432), txl.uint32(0x3210))
    g_vals = txl.alloc_local([TOKENS_PER_THREAD + K - 1], "uint16" if use_bf16_fma else "float32")
    x_vals = txl.alloc_local([TOKENS_PER_THREAD + K - 1], "uint16" if use_bf16_fma else "float32")
    out = txl.alloc_local([TOKENS_PER_THREAD], "float32")
    dw_part = txl.alloc_local([K], "float32")
    acc_dw = txl.alloc_local([2 * K], "float32")
    dx_bits = txl.alloc_local([TOKENS_PER_THREAD], "uint16")
    for k in range(K):
        if use_bf16_fma:
            load_w_bf16(w_taps[k], w, (ch_start + ch) * K + k, ch_start + ch < C)
        else:
            load_w_f32(w_taps[k], w, (ch_start + ch) * K + k, w_dtype, ch_start + ch < C)
    for i in range(2 * K):
        txl.assign(acc_dw[i], txl.float32(0.0))
    tile_count = txl.local_scalar("int32", init=txl.int32(0))
    with txl.while_(row_tile_idx < num_row_tiles):
        token_tile_idx = row_tile_idx % num_token_tiles
        batch_idx = row_tile_idx // num_token_tiles
        txl.cuda.mbarrier_wait(s_full.ptr_to([stage]), phase)
        # channel slots past the slice width read (and discard) the slice's last channel
        read_ch = txl.min(ch, num_ch - 1)
        g_base = stage * g_stage_size + read_ch * CHANNELS_FIRST_PITCH + token_group * TOKENS_PER_THREAD
        x_base = stage * x_stage_size + read_ch
        for word_idx in range(NUM_G_WORDS // 4):
            txl.ptx["ld.shared.v4.b32"](g_words[4 * word_idx], g_words[4 * word_idx + 1], g_words[4 * word_idx + 2], g_words[4 * word_idx + 3], s_g.ptr_to([g_base + 8 * word_idx]))
        for word_idx in range(NUM_SHIFTED_WORDS + 2):
            txl.assign(g_words_shift4[word_idx], txl.Select(is_shift4, g_words[word_idx + 2], g_words[word_idx]))
        for word_idx in range(NUM_SHIFTED_WORDS + 1):
            txl.assign(g_words_shift2[word_idx], txl.Select(is_shift2, g_words_shift4[word_idx + 1], g_words_shift4[word_idx]))
        for word_idx in range(NUM_SHIFTED_WORDS):
            txl.ptx["prmt.b32"](g_words[word_idx], g_words_shift2[word_idx], g_words_shift2[word_idx + 1], prmt_selector)
        for j in range(TOKENS_PER_THREAD + K - 1):
            g_word = g_words[j // 2]
            if use_bf16_fma:
                txl.assign(g_vals[j], txl.Cast("uint16", g_word if j % 2 == 0 else g_word >> txl.uint32(16)))
                txl.ptx["ld.shared.b16"](x_vals[j], s_x.ptr_to([x_base + (token_group * TOKENS_PER_THREAD + j) * num_ch]))
            else:
                if j % 2 == 0:
                    txl.assign(g_vals[j], txl.reinterpret(g_word << txl.uint32(16), ty="float32"))
                else:
                    txl.assign(g_vals[j], txl.reinterpret(g_word & txl.uint32(0xFFFF0000), ty="float32"))
                load_shared_bf16(x_vals[j], s_x.ptr_to([x_base + (token_group * TOKENS_PER_THREAD + j) * num_ch]))
        release_stage(s_empty, stage, lane)
        if use_bf16_fma:
            compute_dx_taps_bf16(out, w_taps, g_vals, K)
        else:
            compute_dx_taps(out, w_taps, g_vals, K)
        for group_half in range(2):
            if use_bf16_fma:
                compute_dw_partials_bf16(dw_part, g_vals, x_vals, K, group_half * PACKED_TOKENS_PER_THREAD, PACKED_TOKENS_PER_THREAD)
            else:
                compute_dw_partials(dw_part, g_vals, x_vals, K, group_half * PACKED_TOKENS_PER_THREAD, PACKED_TOKENS_PER_THREAD)
            for k in range(K):
                txl.assign(acc_dw[group_half * K + k], acc_dw[group_half * K + k] + dw_part[k])
        dx_buf = tile_count % NUM_DX_BUFS
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(NUM_DX_BUFS - 1))
        for i in range(TOKENS_PER_THREAD):
            txl.ptx["cvt.rn.bf16.f32"](dx_bits[i], out[i])
        named_barrier(1, NUM_COMPUTE_THREADS)
        with txl.if_(ch_ok), txl.then_():
            for i in range(TOKENS_PER_THREAD):
                txl.ptx["st.shared.b16"](s_dx.ptr_to([dx_buf * T * width_max + (token_group * TOKENS_PER_THREAD + i) * num_ch + ch]), dx_bits[i])
        txl.ptx["fence.proxy.async.shared::cta"]()
        named_barrier(1, NUM_COMPUTE_THREADS)
        ctx.store_dx(dx_buf, token_tile_idx, batch_idx)
        txl.assign(tile_count, tile_count + 1)
        ctx.advance_tile()
    for group_half in range(2):
        for k in range(K):
            txl.ptx["st.shared.f32"](s_dw_groups.ptr_to([((2 * token_group + group_half) * K + k) * TILE_CHANNELS + ch]), acc_dw[group_half * K + k])


def bwd_compute_loop_cf_packed(ctx):
    """
    Channels-first g with the packed channels-last thread layout: lane = channel pair pair_ch, warp
    = token group token_group of PACKED_TOKENS_PER_THREAD tokens, f32x2 arithmetic. g is read along
    tokens from the [num_ch][CHANNELS_FIRST_PITCH] tile (32 tokens from the aligned token_group *
    PACKED_TOKENS_PER_THREAD per channel, shifted by the runtime g_shift in registers), x per token
    pair from the channels-last tile. Per-channel arithmetic order identical to every other path.
    """
    s_g, s_x, s_dx, s_dw_groups = ctx.s_g, ctx.s_x, ctx.s_dx, ctx.s_dw_groups
    w, w_dtype, C, K = ctx.w, ctx.w_dtype, ctx.C, ctx.K
    ch_start, num_ch, g_shift = ctx.ch_start, ctx.num_ch, ctx.g_shift
    tidx, warp, lane = ctx.tidx, ctx.warp, ctx.lane
    s_full, s_empty, stage, phase = ctx.s_full, ctx.s_empty, ctx.stage, ctx.phase
    row_tile_idx, num_row_tiles, num_token_tiles = ctx.row_tile_idx, ctx.num_row_tiles, ctx.num_token_tiles
    g_stage_size, x_stage_size, width_max = ctx.g_stage_size, ctx.x_stage_size, ctx.width_max
    pair_ch = 2 * lane
    token_group = warp
    ch_ok = pair_ch < num_ch
    NUM_G_WORDS = 16  # g words per channel: tokens [token_group * PACKED_TOKENS_PER_THREAD, +32)
    num_vals = PACKED_TOKENS_PER_THREAD + K - 1
    NUM_SHIFTED_WORDS = cdiv(num_vals, 2)
    w_taps = txl.alloc_local([K], "uint64")
    w_pair = txl.alloc_local([2], "float32")
    g_words = [txl.alloc_local([NUM_G_WORDS], "uint32") for _ in range(2)]
    g_words_shift4 = txl.alloc_local([NUM_SHIFTED_WORDS + 2], "uint32")
    g_words_shift2 = txl.alloc_local([NUM_SHIFTED_WORDS + 1], "uint32")
    is_shift4 = (g_shift & 4) != 0
    is_shift2 = (g_shift & 2) != 0
    prmt_selector = txl.Select((g_shift & 1) != 0, txl.uint32(0x5432), txl.uint32(0x3210))
    g_vals = txl.alloc_local([num_vals], "uint64")
    x_vals = txl.alloc_local([num_vals], "uint64")
    out = txl.alloc_local([PACKED_TOKENS_PER_THREAD], "uint64")
    dw_part = txl.alloc_local([K], "uint64")
    acc_dw = txl.alloc_local([K], "uint64")
    dx_words = txl.alloc_local([PACKED_TOKENS_PER_THREAD], "uint32")
    x_word = txl.local_scalar("uint32")
    load_w_pair(w_taps, w_pair, w, ch_start + pair_ch, C, K, w_dtype)
    for k in range(K):
        txl.assign(acc_dw[k], txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))
    tile_count = txl.local_scalar("int32", init=txl.int32(0))
    with txl.while_(row_tile_idx < num_row_tiles):
        token_tile_idx = row_tile_idx % num_token_tiles
        batch_idx = row_tile_idx // num_token_tiles
        txl.cuda.mbarrier_wait(s_full.ptr_to([stage]), phase)
        # lanes past the slice width read (and discard) the slice's last pair: no reads past the tile
        read_ch = txl.min(pair_ch, num_ch - 2)
        for elem in range(2):
            g_base = stage * g_stage_size + (read_ch + elem) * CHANNELS_FIRST_PITCH + token_group * PACKED_TOKENS_PER_THREAD
            for word_idx in range(NUM_G_WORDS // 4):
                txl.ptx["ld.shared.v4.b32"](g_words[elem][4 * word_idx], g_words[elem][4 * word_idx + 1], g_words[elem][4 * word_idx + 2], g_words[elem][4 * word_idx + 3], s_g.ptr_to([g_base + 8 * word_idx]))
            for word_idx in range(NUM_SHIFTED_WORDS + 2):
                txl.assign(g_words_shift4[word_idx], txl.Select(is_shift4, g_words[elem][word_idx + 2], g_words[elem][word_idx]))
            for word_idx in range(NUM_SHIFTED_WORDS + 1):
                txl.assign(g_words_shift2[word_idx], txl.Select(is_shift2, g_words_shift4[word_idx + 1], g_words_shift4[word_idx]))
            for word_idx in range(NUM_SHIFTED_WORDS):
                txl.ptx["prmt.b32"](g_words[elem][word_idx], g_words_shift2[word_idx], g_words_shift2[word_idx + 1], prmt_selector)
        x_base = stage * x_stage_size + read_ch
        for j in range(num_vals):
            word_ch0 = g_words[0][j // 2]
            word_ch1 = g_words[1][j // 2]
            if j % 2 == 0:
                lo = txl.reinterpret(word_ch0 << txl.uint32(16), ty="float32")
                hi = txl.reinterpret(word_ch1 << txl.uint32(16), ty="float32")
            else:
                lo = txl.reinterpret(word_ch0 & txl.uint32(0xFFFF0000), ty="float32")
                hi = txl.reinterpret(word_ch1 & txl.uint32(0xFFFF0000), ty="float32")
            txl.assign(g_vals[j], txl.cuda.make_float2(lo, hi))
            txl.ptx["ld.shared.b32"](x_word, s_x.ptr_to([x_base + (token_group * PACKED_TOKENS_PER_THREAD + j) * num_ch]))
            txl.assign(x_vals[j], bf16x2_to_f32x2(x_word))
        release_stage(s_empty, stage, lane)
        compute_taps_packed(out, w_taps, g_vals, K, True)
        compute_dw_partials_packed(dw_part, g_vals, x_vals, K)
        for k in range(K):
            txl.ptx["add.rn.f32x2"](acc_dw[k], acc_dw[k], dw_part[k])
        dx_buf = tile_count % NUM_DX_BUFS
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(NUM_DX_BUFS - 1))
        for i in range(PACKED_TOKENS_PER_THREAD):
            txl.ptx["cvt.rn.bf16x2.f32"](dx_words[i], float2_elem(out[i], 1), float2_elem(out[i], 0))
        named_barrier(1, NUM_COMPUTE_THREADS)
        with txl.if_(ch_ok), txl.then_():
            for i in range(PACKED_TOKENS_PER_THREAD):
                txl.ptx["st.shared.b32"](s_dx.ptr_to([dx_buf * T * width_max + (token_group * PACKED_TOKENS_PER_THREAD + i) * num_ch + pair_ch]), dx_words[i])
        txl.ptx["fence.proxy.async.shared::cta"]()
        named_barrier(1, NUM_COMPUTE_THREADS)
        ctx.store_dx(dx_buf, token_tile_idx, batch_idx)
        txl.assign(tile_count, tile_count + 1)
        ctx.advance_tile()
    for k in range(K):
        for elem in range(2):
            txl.ptx["st.shared.f32"](s_dw_groups.ptr_to([(token_group * K + k) * TILE_CHANNELS + pair_ch + elem]), float2_elem(acc_dw[k], elem))


def store_dw_elem(dw, idx, value, w_dtype):
    """
    dw[idx] = value rounded to w_dtype.
    """
    if w_dtype == "float32":
        txl.ptx["st.global.f32"](dw.ptr_to([idx]), value)
    else:
        bits = txl.local_scalar("uint16")
        txl.ptx["cvt.rn.bf16.f32" if w_dtype == "bfloat16" else "cvt.rn.f16.f32"](bits, value)
        txl.ptx["st.global.b16"](dw.ptr_to([idx]), bits)


# Plain path (any C, S, strides): one tile per CTA, plain loads / stores, same arithmetic.


def copy_chunk_to_shared(width, words, g_ptr, s_ptr, valid):
    """
    Copy width contiguous bf16 from global to shared (zeros when not valid).
    """
    num_words = max(width // 2, 1)
    word_type = "uint32" if width >= 2 else "uint16"
    for i in range(num_words):
        txl.assign(words[i], txl.Cast(word_type, 0))
    with txl.if_(valid), txl.then_():
        if width == 8:
            txl.ptx["ld.global.nc.v4.b32"](words[0], words[1], words[2], words[3], g_ptr)
        elif width == 4:
            txl.ptx["ld.global.nc.v2.b32"](words[0], words[1], g_ptr)
        elif width == 2:
            txl.ptx["ld.global.nc.b32"](words[0], g_ptr)
        else:
            txl.ptx["ld.global.nc.b16"](words[0], g_ptr)
    if width == 8:
        txl.ptx["st.shared.v4.b32"](s_ptr, words[0], words[1], words[2], words[3])
    elif width == 4:
        txl.ptx["st.shared.v2.b32"](s_ptr, words[0], words[1])
    elif width == 2:
        txl.ptx["st.shared.b32"](s_ptr, words[0])
    else:
        txl.ptx["st.shared.b16"](s_ptr, words[0])


def stage_rows(width, words, src, s_buf, tidx, first_row, num_rows_valid, num_rows, C, ch_start, src_base):
    """
    s_buf[j][ch] = src[src_base + (first_row + j) * C + ch_start + ch] for j < num_rows, zero
    outside.
    """
    chunks_per_row = TILE_CHANNELS // width
    num_chunks = num_rows * chunks_per_row
    for i in range(cdiv(num_chunks, NUM_COMPUTE_THREADS)):
        idx = i * NUM_COMPUTE_THREADS + tidx
        row = idx // chunks_per_row
        chunk_idx = idx % chunks_per_row
        src_row = first_row + row
        ch = ch_start + chunk_idx * width
        valid = (src_row >= 0) & (src_row < num_rows_valid) & (ch < C) & (idx < num_chunks)
        with txl.if_(idx < num_chunks), txl.then_():
            copy_chunk_to_shared(width, words, src.ptr_to([src_base + as_int64(src_row * C + ch)]), s_buf.ptr_to([row * TILE_CHANNELS + chunk_idx * width]), valid)


@functools.cache
def build_fwd_plain(C, K, w_dtype):
    """
    Plain forward: one (T tokens, TILE_CHANNELS channels) tile per CTA.
    """
    STAGE_ROWS = T + K - 1
    NUM_CH_TILES = cdiv(C, TILE_CHANNELS)
    WIDTH = vector_width(C)
    Y_PITCH = T + 2

    @txl.kernel()
    def conv_fwd_plain(x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w: txl.Tensor((txl.int64(),), w_dtype, data_alignment=1), y: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), L: txl.int32, S: txl.int32, offset: txl.int32, B: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[(S + T - 1) // T, NUM_CH_TILES, B], block=NUM_COMPUTE_WARPS * 32))
        token_tile_idx, ch_tile_idx, batch_idx = txl.cta_id()
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        token_start = token_tile_idx * T
        ch_start = ch_tile_idx * TILE_CHANNELS
        smem = txl.smem_pool()
        s_x = smem.alloc([STAGE_ROWS * TILE_CHANNELS], "bfloat16", align=16)
        s_y = smem.alloc([TILE_CHANNELS * Y_PITCH], "bfloat16", align=16)
        words = txl.alloc_local([4], "uint32" if WIDTH >= 2 else "uint16")
        stage_rows(WIDTH, words, x, s_x, tidx, token_start + offset - (K - 1), L, STAGE_ROWS, C, ch_start, as_int64(batch_idx) * as_int64(L) * C)
        ch = lane + 32 * (warp % 2)
        token_group = warp // 2
        w_taps = txl.alloc_local([K], "float32")
        for k in range(K):
            load_w_f32(w_taps[k], w, (ch_start + ch) * K + k, w_dtype, ch_start + ch < C)
        named_barrier(0, NUM_COMPUTE_THREADS)
        x_vals = txl.alloc_local([TOKENS_PER_THREAD + K - 1], "float32")
        for j in range(TOKENS_PER_THREAD + K - 1):
            load_shared_bf16(x_vals[j], s_x.ptr_to([(token_group * TOKENS_PER_THREAD + j) * TILE_CHANNELS + ch]))
        out = txl.alloc_local([TOKENS_PER_THREAD], "float32")
        compute_y_taps(out, w_taps, x_vals, K)
        y_bits = txl.local_scalar("uint16")
        for j in range(TOKENS_PER_THREAD):
            txl.ptx["cvt.rn.bf16.f32"](y_bits, out[j])
            txl.ptx["st.shared.b16"](s_y.ptr_to([ch * Y_PITCH + token_group * TOKENS_PER_THREAD + j]), y_bits)
        named_barrier(0, NUM_COMPUTE_THREADS)
        y_batch_base = as_int64(batch_idx) * C * as_int64(S)
        for i in range(cdiv(TILE_CHANNELS * T, NUM_COMPUTE_THREADS)):
            idx = i * NUM_COMPUTE_THREADS + tidx
            tile_ch = idx // T
            tile_token = idx % T
            with txl.if_((ch_start + tile_ch < C) & (token_start + tile_token < S)), txl.then_():
                txl.ptx["ld.shared.b16"](y_bits, s_y.ptr_to([tile_ch * Y_PITCH + tile_token]))
                txl.ptx["st.global.b16"](y.ptr_to([y_batch_base + as_int64((ch_start + tile_ch) * S + token_start + tile_token)]), y_bits)

    return entry_function(conv_fwd_plain.compile(TARGET))


@functools.cache
def build_bwd_plain(C, K, w_dtype, mode):
    """
    mode: "cl" (g contiguous) or "gen" (runtime strides). Per-thread dw partials go to
    dwp[row_tile_idx][token group][k][c] (row_tile_idx = batch_idx * num_token_tiles +
    token_tile_idx) for build_dw_reduce, which sums them in the TMA order.
    """
    STAGE_ROWS = T + K - 1
    NUM_CH_TILES = cdiv(C, TILE_CHANNELS)
    WIDTH = vector_width(C)
    G_PITCH = TILE_CHANNELS + 2 if mode == "gen" else TILE_CHANNELS

    @txl.kernel()
    def conv_bwd_plain(g: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), x: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), w: txl.Tensor((txl.int64(),), w_dtype, data_alignment=1), dx: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), dwp: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), L: txl.int32, S: txl.int32, offset: txl.int32, B: txl.int32, gsb: txl.int64, gst: txl.int64, gsc: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=[(L + T - 1) // T, NUM_CH_TILES, B], block=NUM_COMPUTE_WARPS * 32))
        token_tile_idx, ch_tile_idx, batch_idx = txl.cta_id()
        tidx = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        token_start = token_tile_idx * T
        ch_start = ch_tile_idx * TILE_CHANNELS
        num_token_tiles = (L + T - 1) // T
        smem = txl.smem_pool()
        s_g = smem.alloc([STAGE_ROWS * G_PITCH], "bfloat16", align=16)
        s_x = smem.alloc([STAGE_ROWS * TILE_CHANNELS], "bfloat16", align=16)
        words = txl.alloc_local([4], "uint32" if WIDTH >= 2 else "uint16")
        x_batch_base = as_int64(batch_idx) * as_int64(L) * C
        stage_rows(WIDTH, words, x, s_x, tidx, token_start - (K - 1), L, STAGE_ROWS, C, ch_start, x_batch_base)
        if mode == "cl":
            stage_rows(WIDTH, words, g, s_g, tidx, token_start - offset, S, STAGE_ROWS, C, ch_start, as_int64(batch_idx) * gsb)
        else:
            g_bits = txl.local_scalar("uint16")
            for i in range(cdiv(STAGE_ROWS * TILE_CHANNELS, NUM_COMPUTE_THREADS)):
                idx = i * NUM_COMPUTE_THREADS + tidx
                row = idx // TILE_CHANNELS
                tile_ch = idx % TILE_CHANNELS
                token = token_start - offset + row
                txl.assign(g_bits, txl.uint16(0))
                with txl.if_((token >= 0) & (token < S) & (ch_start + tile_ch < C) & (idx < STAGE_ROWS * TILE_CHANNELS)), txl.then_():
                    txl.ptx["ld.global.nc.b16"](g_bits, g.ptr_to([as_int64(batch_idx) * gsb + as_int64(token) * gst + as_int64(ch_start + tile_ch) * gsc]))
                with txl.if_(idx < STAGE_ROWS * TILE_CHANNELS), txl.then_():
                    txl.ptx["st.shared.b16"](s_g.ptr_to([row * G_PITCH + tile_ch]), g_bits)
        ch = lane + 32 * (warp % 2)
        token_group = warp // 2
        w_taps = txl.alloc_local([K], "float32")
        for k in range(K):
            load_w_f32(w_taps[k], w, (ch_start + ch) * K + k, w_dtype, ch_start + ch < C)
        named_barrier(0, NUM_COMPUTE_THREADS)
        g_vals = txl.alloc_local([TOKENS_PER_THREAD + K - 1], "float32")
        x_vals = txl.alloc_local([TOKENS_PER_THREAD + K - 1], "float32")
        for j in range(TOKENS_PER_THREAD + K - 1):
            load_shared_bf16(g_vals[j], s_g.ptr_to([(token_group * TOKENS_PER_THREAD + j) * G_PITCH + ch]))
            load_shared_bf16(x_vals[j], s_x.ptr_to([(token_group * TOKENS_PER_THREAD + j) * TILE_CHANNELS + ch]))
        out = txl.alloc_local([TOKENS_PER_THREAD], "float32")
        compute_dx_taps(out, w_taps, g_vals, K)
        dw_part = txl.alloc_local([2 * K], "float32")
        # dw partials per PACKED_TOKENS_PER_THREAD-token group (the packed TMA kernels' grouping)
        for group_half in range(2):
            compute_dw_partials(LocalWindow(dw_part, group_half * K), g_vals, x_vals, K, group_half * PACKED_TOKENS_PER_THREAD, PACKED_TOKENS_PER_THREAD)
        dx_bits = txl.local_scalar("uint16")
        for i in range(TOKENS_PER_THREAD):
            token = token_start + token_group * TOKENS_PER_THREAD + i
            with txl.if_((token < L) & (ch_start + ch < C)), txl.then_():
                txl.ptx["cvt.rn.bf16.f32"](dx_bits, out[i])
                txl.ptx["st.global.b16"](dx.ptr_to([x_batch_base + as_int64(token * C + ch_start + ch)]), dx_bits)
        row_tile_idx = batch_idx * num_token_tiles + token_tile_idx
        with txl.if_(ch_start + ch < C), txl.then_():
            for group_half in range(2):
                for k in range(K):
                    txl.ptx["st.global.f32"](dwp.ptr_to([((as_int64(row_tile_idx) * NUM_TOKEN_GROUPS + as_int64(2 * token_group + group_half)) * K + k) * C + as_int64(ch_start + ch)]), dw_part[group_half * K + k])

    return entry_function(conv_bwd_plain.compile(TARGET))


@functools.cache
def build_dw_reduce(C, K, w_dtype):
    """
    dw[c][k] in the TMA kernel's order: ranks rank_idx < CLUSTER_SIZE, token groups, row tiles
    row_tile_idx = rank_idx mod CLUSTER_SIZE.
    """
    NUM_W = C * K

    @txl.kernel()
    def dw_reduce(dwp: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), dw: txl.Tensor((txl.int64(),), w_dtype, data_alignment=1), nrt: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=(NUM_W + 255) // 256, block=256))
        idx = txl.cta_id() * 256 + txl.thread_id()
        tap = idx // C
        ch = idx % C
        acc_dw = txl.local_scalar("float32")
        acc_rank = txl.local_scalar("float32")
        acc_group = txl.local_scalar("float32")
        partial = txl.local_scalar("float32")
        row_tile_idx = txl.local_scalar("int32")
        with txl.if_(idx < NUM_W), txl.then_():
            txl.assign(acc_dw, txl.float32(0.0))
            for rank_idx in range(CLUSTER_SIZE):
                txl.assign(acc_rank, txl.float32(0.0))
                for group_idx in range(NUM_TOKEN_GROUPS):
                    txl.assign(acc_group, txl.float32(0.0))
                    txl.assign(row_tile_idx, txl.int32(rank_idx))
                    with txl.while_(row_tile_idx < nrt):
                        txl.ptx["ld.global.f32"](partial, dwp.ptr_to([((as_int64(row_tile_idx) * NUM_TOKEN_GROUPS + group_idx) * K + as_int64(tap)) * C + as_int64(ch)]))
                        txl.assign(acc_group, acc_group + partial)
                        txl.assign(row_tile_idx, row_tile_idx + CLUSTER_SIZE)
                    txl.assign(acc_rank, acc_rank + acc_group)
                txl.assign(acc_dw, acc_dw + acc_rank)
            store_dw_elem(dw, ch * K + tap, acc_dw, w_dtype)

    return entry_function(dw_reduce.compile(TARGET))


def is_aligned16(*tensors):
    return all(tensor.data_ptr() % 16 == 0 for tensor in tensors)


def realigned(tensor):
    """
    tensor itself if its data starts on a 16-byte boundary, else a fresh copy in the same layout
    (every path reads 16-byte vectors from the base, so a misaligned base would fault).
    """
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def flatten_weight(w):
    """
    w flattened in a dtype the kernels read: other float dtypes are read as fp32.
    """
    w_flat = w.reshape(-1)
    return w_flat if w.dtype in WEIGHT_DTYPE_NAME else w_flat.float()


def causal_conv1d_fwd_sm100(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    """
    The forward of the picker's causal_conv1d_fwd: a fresh bf16 (B, S, C) in the layout of
    causal_conv1d_empty, bit-identical to ATen's depthwise conv1d.
    """
    x = realigned(x.contiguous())
    B, L, C = x.shape
    K = w.shape[-1]
    S = L - offset
    y = torch.empty((B, C, S), dtype=x.dtype, device=x.device)
    if S > 0 and B > 0 and C > 0:
        w_flat = flatten_weight(w)
        w_dtype = WEIGHT_DTYPE_NAME[w_flat.dtype]
        if C % 8 == 0 and S % 8 == 0 and 2 <= K <= 8 and is_aligned16(x, y):
            build_fwd_tma(C, K, w_dtype, S >= 8192)(x.view(-1), w_flat, y.view(-1), L, S, offset, B)
        else:
            build_fwd_plain(C, K, w_dtype)(x.view(-1), w_flat, y.view(-1), L, S, offset, B)
    return y.transpose(1, 2)


def causal_conv1d_bwd_sm100(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The backward of the picker's causal_conv1d_bwd: dx bit-identical to ATen's depthwise conv
    backward, dw in one fixed fp32 order whatever g's strides.
    """
    x, g = realigned(x.contiguous()), realigned(g)
    B, L, C = x.shape
    K = w.shape[-1]
    S = L - offset
    dx = torch.empty_like(x)
    if C == 0:
        return dx.zero_(), torch.zeros(w.shape, dtype=w.dtype, device=w.device)
    w_flat = flatten_weight(w)
    w_dtype = WEIGHT_DTYPE_NAME[w_flat.dtype]
    dw_flat = torch.empty((C * K,), dtype=w_flat.dtype, device=w.device)
    if g.numel() == 0:
        g = g.contiguous()  # nothing to read: the plain path reads it as channels-last
    num_row_tiles = B * cdiv(L, T)
    g_channels_last = g.is_contiguous()
    g_channels_first = (not g_channels_last) and S > 0 and g.stride(1) == 1 and g.stride(2) == S and g.stride(0) == C * S
    can_use_tma = C % 8 == 0 and 2 <= K <= 8 and S > 0 and num_row_tiles > 0 and is_aligned16(g, x, dx)
    if can_use_tma and (g_channels_last or (g_channels_first and S % 8 == 0)):
        g_flat = g.view(-1) if g_channels_last else g.as_strided((B * C * S,), (1,))
        build_bwd_tma(C, K, w_dtype, not g_channels_last)(g_flat, x.view(-1), w_flat, dx.view(-1), dw_flat, L, S, offset, B)
    else:
        dw_partials = torch.empty((max(num_row_tiles, 1), NUM_TOKEN_GROUPS, K, C), dtype=torch.float32, device=x.device)
        if num_row_tiles > 0:
            if g_channels_last:
                build_bwd_plain(C, K, w_dtype, "cl")(g.view(-1), x.view(-1), w_flat, dx.view(-1), dw_partials.view(-1), L, S, offset, B, S * C, C, 1)
            else:
                g_stride_b, g_stride_s, g_stride_c = g.stride()
                g_span = 1 + (B - 1) * g_stride_b + max(S - 1, 0) * g_stride_s + (C - 1) * g_stride_c
                build_bwd_plain(C, K, w_dtype, "gen")(g.as_strided((g_span,), (1,)), x.view(-1), w_flat, dx.view(-1), dw_partials.view(-1), L, S, offset, B, g_stride_b, g_stride_s, g_stride_c)
        build_dw_reduce(C, K, w_dtype)(dw_partials.view(-1), dw_flat, num_row_tiles)
    return dx, dw_flat.to(w.dtype).view(w.shape)
