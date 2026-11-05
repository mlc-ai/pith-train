"""
SM100 fused SwiGLU (+ gradients) -> grouped FP8 quantization (SM100 layout), TIRx-lite.

The output is cut into 128-row x 128-column tiles: one tile holds whole 1x128 row blocks
(x_fp8 / x_scale) and whole 128x1 column blocks (x_ch_fp8 / x_ch_scale), because groups are
128-row aligned (grouped_mm_offs is therefore not needed for the SM100 layout).

Persistent schedule: 512-thread CTAs (one per SM) walk tiles round-robin. Each thread owns the same
4 rows x 8 columns of every tile and prefetches them with cp.async into its own slots of a
shared-memory ring, so the HBM stream overlaps the arithmetic without any producer/consumer
barrier (a thread only ever reads bytes it copied itself). A tile load is split into row parts
(halves or quarters) that form a unit ring of per-mode depth: the forward keeps 1.5 tiles in
flight (three half-tile units), N=512 backward a quarter-tile ring one tile deep, N=1408 backward a
half-tile ring. The one CTA-wide exchange is the column amax: per-warp partials, 64 reducer
threads, an rcp broadcast.

Arithmetic runs on packed pairs (f32x2 / bf16x2). The sigmoid of silu_mul, 1 / (1 + ex2.approx(x))
with div.full.f32, is computed as 0.25 * rcp.approx.ftz(fma(ex2.approx.ftz(x), 0.25, 0.25)): this
is bit-identical for every bf16 input (checked exhaustively, including inf and NaN), since
1 + e absorbs any subnormal e and the 0.25 pre-scale reproduces div.full's large-divisor path.

gu_bwd with N % 128 != 0 (dg and du blocks offset by N % 128 columns) keeps one load and one
sigmoid per element by walking short ranges of consecutive 128-column windows
(body_interleaved_ranges): CTA c takes ranges c, c + G, ... so concurrent CTAs read adjacent
columns of the same rows, carries du of the previous window in thread-private shared memory and
recomputes it (halo) at range starts.

N not a multiple of 8 falls back to a non-persistent kernel with narrower predicated loads.
"""

import functools
import math

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["fp8_silu_mul_fwd_sm100", "fp8_silu_mul_bwd_sm100", "fp8_silu_mul_gu_fwd_sm100", "fp8_silu_mul_gu_bwd_sm100"]


def max_fmaxf(a, b):
    """
    txl.max as apache-tvm 0.27 emitted it: CUDA fmaxf for float32 operands, which drops a NaN
    operand (tvm_tirx emits a NaN-propagating select for txl.max); txl.max otherwise.
    """
    return txl.call_pure_extern("fmaxf", a, b, ty=txl.float32) if is_f32(a, b) else txl.max(a, b)


def min_fminf(a, b):
    """
    txl.min as apache-tvm 0.27 emitted it: CUDA fminf for float32 operands, txl.min otherwise.
    """
    return txl.call_pure_extern("fminf", a, b, ty=txl.float32) if is_f32(a, b) else txl.min(a, b)


def is_f32(*operands):
    """
    Whether any operand is a float32 expression (the other may be a Python float).
    """
    return any(str(getattr(getattr(operand, "ty", None), "dtype", "")) == "float32" for operand in operands)


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


BLOCK = 128
NEG_LOG2E = -1.4426950408889634
FP8_MAX_RCP = 1.0 / 448.0
FULL = 0xFFFFFFFF
ABS2 = 0x7FFF7FFF
# Persistent schedule: 512-thread CTAs, one resident per SM. Modes without a unit ring ("hu") use a
# one-tile cp.async lookahead: the tile is copied to registers before the next one is issued, so
# one ring slot suffices.
CFG = {"threads": 512, "stages_fwd": 1, "occ_fwd": 1, "threads_bwd": 512, "stages_bwd": 1, "occ_bwd": 1,
       # cp.async unit ring per mode[:N]: a tile load is split into hp row parts, hu units in flight
       "hu": {"fwd": 3, "gu_fwd": 3, "bwd:1408": 2, "gu_bwd_pair:1408": 2, "bwd:512": 4, "gu_bwd_pair:512": 4},
       "hp": {"bwd:512": 4, "gu_bwd_pair:512": 4},
       # modes whose tile coordinates keep the signed division (see tile_coords)
       "signed": ["bwd:1408", "gu_bwd_pair:1408", "bwd:512", "gu_bwd_pair:512"],
       # gu_bwd with N % 128 != 0: windows per interleaved range
       "run_l": 3}


def cdiv(a, b):
    return (a + b - 1) // b


def lo_f32(word):
    return txl.reinterpret(txl.shift_left(word, txl.uint32(16)), ty="float32")


def hi_f32(word):
    return txl.reinterpret(txl.bitwise_and(word, txl.uint32(0xFFFF0000)), ty="float32")


def f32x2(a, b):
    return txl.cuda.make_float2(a, b)


class F32Pair:
    """
    A packed f32 pair living in one 64-bit register.
    """

    def __init__(self):
        self.reg = txl.local_scalar("uint64")

    @property
    def x(self):
        return txl.cuda.float2_x(self.reg)

    @property
    def y(self):
        return txl.cuda.float2_y(self.reg)


def unpack_bf16x2(word):
    pair = F32Pair()
    txl.ptx.mov.b64(pair.reg, f32x2(lo_f32(word), hi_f32(word)))
    return pair


def mul_f32x2(a, b):
    pair = F32Pair()
    txl.ptx.mul.rn.f32x2(pair.reg, a, b)
    return pair


def fma_f32x2(a, b, c):
    pair = F32Pair()
    txl.ptx.fma.rn.f32x2(pair.reg, a, b, c)
    return pair


def add_f32x2(a, b):
    pair = F32Pair()
    txl.ptx.add.rn.f32x2(pair.reg, a, b)
    return pair


def splat_f32x2(value):
    return f32x2(txl.float32(value), txl.float32(value))


def sigmoid_x4(gate):
    """
    Packed sigmoid of the pair gate, times 4 (see module docstring).
    """
    scaled = mul_f32x2(gate.reg, splat_f32x2(NEG_LOG2E))
    exp_lo, exp_hi = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(exp_lo, scaled.x)
    txl.ptx.ex2.approx.ftz.f32(exp_hi, scaled.y)
    denom = fma_f32x2(f32x2(exp_lo, exp_hi), splat_f32x2(0.25), splat_f32x2(0.25))
    rcp_lo, rcp_hi = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.rcp.approx.ftz.f32(rcp_lo, denom.x)
    txl.ptx.rcp.approx.ftz.f32(rcp_hi, denom.y)
    return f32x2(rcp_lo, rcp_hi)


def scale_rcp(amax):
    """
    E8M0 scale and its exact reciprocal (the recipe's E8M0 scale, native cvt).
    """
    scale_e8m0x2 = txl.local_scalar("uint16")
    scaled_amax = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32(scaled_amax, max_fmaxf(amax, txl.float32(1e-4)), txl.float32(FP8_MAX_RCP))
    txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(scale_e8m0x2, txl.float32(0.0), scaled_amax)
    exponent = txl.bitwise_and(txl.cast(scale_e8m0x2, "uint32"), txl.uint32(0xFF))
    scale_bits = txl.local_scalar("uint32")
    txl.assign(scale_bits, txl.Select(exponent == txl.uint32(0), txl.uint32(1 << 23), txl.shift_left(exponent, txl.uint32(23))))
    return txl.reinterpret(scale_bits, ty="float32"), txl.reinterpret(txl.uint32(254 << 23) - scale_bits, ty="float32")


def quantize_bf16x2(dst, word, rcp_bf16x2):
    """
    e4m3x2 of a bf16x2 word times a bf16x2 pair of powers of two.

    The bf16 product is exact wherever it is a normal number; where it is not, |x * rcp| is
    below 2^-126 and both this and the fp32 reference round it to a signed zero byte.
    """
    product = txl.local_scalar("uint32")
    txl.ptx["mul.rn.bf16x2"](product, word, rcp_bf16x2)
    txl.ptx["cvt.rn.satfinite.e4m3x2.bf16x2"](dst, product)


def local_value(expr, dtype="int32"):
    """
    expr held in a fresh local register of dtype.
    """
    value = txl.local_scalar(dtype)
    txl.assign(value, expr)
    return value


def grad_up_word(word_idx, gate_words, grad_words):
    """
    bf16x2 word of du = silu(g) * dh for word word_idx (the du half of swiglu's backward).
    """
    gate = unpack_bf16x2(gate_words[word_idx])
    grad_h = unpack_bf16x2(grad_words[word_idx])
    sig = mul_f32x2(sigmoid_x4(gate), splat_f32x2(0.25))
    grad_up = mul_f32x2(mul_f32x2(sig.reg, gate.reg).reg, grad_h.reg)
    word = txl.local_scalar("uint32")
    txl.ptx.cvt.rn.bf16x2.f32(word, grad_up.y, grad_up.x)
    return word


def swiglu(is_bwd, is_select, N, out_col, gate_words, up_words, grad_words, out_words, WORDS):
    """
    bf16 SwiGLU (or its gradients dg, du) of the packed words, written to out_words[out][word].
    """
    for word_idx in range(WORDS):
        gate = unpack_bf16x2(gate_words[word_idx])
        up = unpack_bf16x2(up_words[word_idx])
        sig_x4 = sigmoid_x4(gate)
        sig = mul_f32x2(sig_x4, splat_f32x2(0.25))
        silu = mul_f32x2(sig.reg, gate.reg)
        if not is_bwd:
            h = mul_f32x2(silu.reg, up.reg)
            txl.ptx.cvt.rn.bf16x2.f32(out_words[0][word_idx], h.y, h.x)
            continue
        grad_h = unpack_bf16x2(grad_words[word_idx])
        neg_sig = mul_f32x2(sig_x4, splat_f32x2(-0.25))
        grad_silu_sig = mul_f32x2(mul_f32x2(grad_h.reg, up.reg).reg, sig.reg)
        gate_plus_one = add_f32x2(gate.reg, splat_f32x2(1.0))
        dsilu_factor = fma_f32x2(neg_sig.reg, gate.reg, gate_plus_one.reg)
        grad_gate = mul_f32x2(grad_silu_sig.reg, dsilu_factor.reg)
        grad_up = mul_f32x2(silu.reg, grad_h.reg)
        if not is_select:
            txl.ptx.cvt.rn.bf16x2.f32(out_words[0][word_idx], grad_gate.y, grad_gate.x)
            txl.ptx.cvt.rn.bf16x2.f32(out_words[1][word_idx], grad_up.y, grad_up.x)
        else:
            col = out_col + 2 * (word_idx % 4)
            lo = txl.Select(col < N, grad_gate.x, grad_up.x)
            hi = txl.Select(col + 1 < N, grad_gate.y, grad_up.y)
            txl.ptx.cvt.rn.bf16x2.f32(out_words[0][word_idx], hi, lo)


@functools.cache
def build_kernel(mode, N):
    """
    The compiled kernel of mode (fwd | gu_fwd | bwd | gu_bwd_pair | gu_bwd_run | gu_bwd_sel) for
    width N, with its launch facts (persistent, column tiles, occupancy, ranges per row block).
    """
    vec_width = math.gcd(8, N)
    is_persistent = vec_width == 8
    is_run = mode == "gu_bwd_run"
    THREADS = CFG["threads_bwd" if mode in ("bwd", "gu_bwd_pair", "gu_bwd_run", "gu_bwd_sel") else "threads"] if is_persistent else 256
    ROWS_PER_THREAD = BLOCK // (THREADS // 16)
    ROW_STRIDE, WARPS = THREADS // 16, THREADS // 32
    is_gu = mode in ("gu_fwd", "gu_bwd_pair", "gu_bwd_run", "gu_bwd_sel")
    is_bwd = mode in ("bwd", "gu_bwd_pair", "gu_bwd_run", "gu_bwd_sel")
    is_select = mode == "gu_bwd_sel"
    is_pair = mode in ("bwd", "gu_bwd_pair", "gu_bwd_run")
    IN_WIDTH = 2 * N if is_gu else N
    UP_OFFSET = N if is_gu else 0
    K = 2 * N if mode in ("gu_bwd_pair", "gu_bwd_run", "gu_bwd_sel") else N  # output width
    NUM_OUT_BLOCKS = cdiv(K, BLOCK)
    COL_LIMIT = K if is_select else N  # column limit of this kernel's tiles (run: windows walk the N input columns)
    NUM_COL_TILES = cdiv(COL_LIMIT, BLOCK)  # column tiles per row block
    is_full = COL_LIMIT % BLOCK == 0
    # run (gu_bwd, N % 128 = HR != 0): input windows of 128 columns; du blocks start HR columns into a
    # window, HS = 128 - HR columns of each du block come from the previous window
    HQ, HR = N // BLOCK, N % BLOCK
    HS = BLOCK - HR
    num_out = 2 if is_pair else 1
    num_in = 3 if is_bwd else 2
    out_offsets = [(0, 0), (N, N // BLOCK)] if mode == "gu_bwd_pair" else [(0, 0), (0, 0)]
    STAGES, OCCUPANCY = CFG["stages_bwd" if is_bwd else "stages_fwd"], CFG["occ_bwd" if is_bwd else "occ_fwd"]
    WORDS = ROWS_PER_THREAD * 4

    def default_geometry(out_idx, col_block, out_col, tidx):
        """
        (store column, row-scale block, reducer column, row-scale guard) of output out_idx.
        """
        col_offset, block_offset = out_offsets[out_idx]
        return out_col + col_offset, col_block + block_offset, col_block * BLOCK + 2 * tidx + col_offset, None

    def quant_store(out_words, outs, row_block, col_block, out_col0, row_base, smem_bufs, tidx, lane, warp, col_group, buf_idx, geometry=None):
        """
        Quantize and store the thread's words of each output, row-wise and column-wise.
        geometry: optional [(words, out tensors, geometry dict)] overriding out_words, outs and
        default_geometry.
        """
        s_col_amax, s_col_rcp = smem_bufs
        peer = txl.local_scalar("uint32")
        items = geometry if geometry is not None else [(out_words[out_idx], outs[out_idx], None) for out_idx in range(num_out)]
        for out_idx, (words, out_tensors, geom) in enumerate(items):
            row_fp8, row_scale, col_fp8, col_scale = out_tensors
            overrides = {}
            if isinstance(geom, dict):
                overrides = geom
                out_col, scale_block, reducer_col, scale_ok = geom["ocol"], geom["sblk"], geom["ccol"], geom["sguard"]
            else:
                out_col, scale_block, reducer_col, scale_ok = geom if geom is not None else default_geometry(out_idx, col_block, out_col0, tidx)
            out_limit = overrides.get("olim", COL_LIMIT + out_offsets[out_idx][0])
            is_out_full = is_full and "olim" not in overrides
            scale_lane = overrides.get("slane", 0)
            # |x| maxima on bf16x2 pairs; accumulators start at 0 so NaN is ignored
            row_amax, col_amax = txl.alloc_local([ROWS_PER_THREAD], "uint32"), txl.alloc_local([4], "uint32")
            for k in range(4):
                txl.assign(col_amax[k], txl.uint32(0))
            for i in range(ROWS_PER_THREAD):
                txl.assign(row_amax[i], txl.uint32(0))
                for k in range(4):
                    txl.ptx["max.xorsign.abs.bf16x2"](row_amax[i], row_amax[i], words[i * 4 + k])
                    txl.ptx["max.xorsign.abs.bf16x2"](col_amax[k], col_amax[k], words[i * 4 + k])
            for i in range(ROWS_PER_THREAD):
                txl.assign(row_amax[i], txl.bitwise_and(row_amax[i], txl.uint32(ABS2)))
            for k in range(4):
                txl.assign(col_amax[k], txl.bitwise_and(col_amax[k], txl.uint32(ABS2)))
            # column partials of this warp (rows row0 and row0 ^ 1 of each 32-row band)
            for k in range(4):
                txl.ptx.shfl_sync.bfly.b32(peer, col_amax[k], txl.uint32(16), txl.uint32(31), txl.uint32(FULL))
                txl.ptx.max.bf16x2(col_amax[k], col_amax[k], peer)
            with txl.if_(lane < 16), txl.then_():
                txl.ptx.st.shared.v4.b32(s_col_amax.ptr_to([buf_idx, out_idx, warp, col_group * 4]), col_amax[0], col_amax[1], col_amax[2], col_amax[3])
            # row maxima: fold the two column parities, two rows per bf16x2 word
            row_amax_pairs = txl.alloc_local([ROWS_PER_THREAD // 2], "uint32")
            for pair_idx in (range(ROWS_PER_THREAD // 2) if "rw" not in overrides else []):
                amax_lo, amax_hi = txl.local_scalar("uint32"), txl.local_scalar("uint32")
                txl.ptx["prmt.b32"](amax_lo, row_amax[2 * pair_idx], row_amax[2 * pair_idx + 1], txl.uint32(0x5410))
                txl.ptx["prmt.b32"](amax_hi, row_amax[2 * pair_idx], row_amax[2 * pair_idx + 1], txl.uint32(0x7632))
                txl.ptx.max.bf16x2(row_amax_pairs[pair_idx], amax_lo, amax_hi)
            for pair_idx in (range(ROWS_PER_THREAD // 2) if "rw" not in overrides else []):
                for lane_mask in (1, 2, 4, 8):
                    txl.ptx.shfl_sync.bfly.b32(peer, row_amax_pairs[pair_idx], txl.uint32(lane_mask), txl.uint32(31), txl.uint32(FULL))
                    txl.ptx.max.bf16x2(row_amax_pairs[pair_idx], row_amax_pairs[pair_idx], peer)
            if "rw" in overrides:
                for pair_idx in range(ROWS_PER_THREAD // 2):
                    txl.assign(row_amax_pairs[pair_idx], overrides["rw"][pair_idx])
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))

            # 64 reducer threads: one bf16x2 column pair each
            with txl.if_(tidx < BLOCK // 2), txl.then_():
                acc_amax = txl.local_scalar("uint32")
                warp_amax = txl.alloc_local([WARPS], "uint32")
                for warp_idx in range(WARPS):
                    txl.ptx.ld.shared.b32(warp_amax[warp_idx], s_col_amax.ptr_to([buf_idx, out_idx, warp_idx, tidx]))
                txl.assign(acc_amax, warp_amax[0])
                for warp_idx in range(1, WARPS):
                    txl.ptx.max.bf16x2(acc_amax, acc_amax, warp_amax[warp_idx])
                scale_lo, rcp_lo = scale_rcp(lo_f32(acc_amax))
                scale_hi, rcp_hi = scale_rcp(hi_f32(acc_amax))
                scale_lo_reg, scale_hi_reg = txl.local_scalar("float32"), txl.local_scalar("float32")
                txl.assign(scale_lo_reg, scale_lo)
                txl.assign(scale_hi_reg, scale_hi)
                txl.ptx.st.shared.v2.f32(s_col_rcp.ptr_to([buf_idx, out_idx, 2 * tidx]), rcp_lo, rcp_hi)
                if out_limit % 2 == 0:
                    with txl.if_(reducer_col < out_limit), txl.then_():
                        txl.ptx.st.global_.v2.f32(col_scale.ptr_to([row_block, reducer_col]), scale_lo_reg, scale_hi_reg)
                else:
                    with txl.if_(reducer_col < out_limit), txl.then_():
                        txl.ptx.st.global_.f32(col_scale.ptr_to([row_block, reducer_col]), scale_lo_reg)
                    with txl.if_(reducer_col + 1 < out_limit), txl.then_():
                        txl.ptx.st.global_.f32(col_scale.ptr_to([row_block, reducer_col + 1]), scale_hi_reg)

            def pack_u16x2(quant, lo_idx, hi_idx):
                return txl.bitwise_or(txl.cast(quant[lo_idx], "uint32"), txl.shift_left(txl.cast(quant[hi_idx], "uint32"), txl.uint32(16)))

            def store_fp8x8(dst, row, quant):
                if vec_width == 8:
                    if is_out_full:
                        txl.ptx.st.global_.v2.b32(dst.ptr_to([row, out_col]), pack_u16x2(quant, 0, 1), pack_u16x2(quant, 2, 3))
                    else:
                        with txl.if_(out_col < out_limit), txl.then_():
                            txl.ptx.st.global_.v2.b32(dst.ptr_to([row, out_col]), pack_u16x2(quant, 0, 1), pack_u16x2(quant, 2, 3))
                elif vec_width == 4:
                    for sub in range(2):
                        with txl.if_(out_col + 4 * sub < out_limit), txl.then_():
                            txl.ptx.st.global_.b32(dst.ptr_to([row, out_col + 4 * sub]), pack_u16x2(quant, 2 * sub, 2 * sub + 1))
                elif vec_width == 2:
                    for sub in range(4):
                        with txl.if_(out_col + 2 * sub < out_limit), txl.then_():
                            txl.ptx.st.global_.b16(dst.ptr_to([row, out_col + 2 * sub]), quant[sub])
                else:
                    for sub in range(8):
                        byte = txl.bitwise_and(txl.shift_right(txl.cast(quant[sub // 2], "uint32"), txl.uint32(8 * (sub % 2))), txl.uint32(0xFF))
                        with txl.if_(out_col + sub < out_limit), txl.then_():
                            txl.ptx.st.global_.b8(dst.ptr_to([row, out_col + sub]), txl.cast(byte, "uint8"))

            # row blocks, while the reducers finish the column scales
            for i in range(ROWS_PER_THREAD):
                row = row_base + ROW_STRIDE * i
                amax_word = row_amax_pairs[i // 2]
                row_scale_value, row_rcp = scale_rcp(lo_f32(amax_word) if i % 2 == 0 else hi_f32(amax_word))
                row_rcp_bf16x2 = txl.local_scalar("uint32")
                txl.ptx["prmt.b32"](row_rcp_bf16x2, txl.reinterpret(row_rcp, ty="uint32"), txl.reinterpret(row_rcp, ty="uint32"), txl.uint32(0x7632))
                with txl.if_(col_group == scale_lane if scale_ok is None else txl.And(col_group == scale_lane, scale_ok)), txl.then_():
                    txl.ptx.st.global_.f32(row_scale.ptr_to([row, scale_block]), row_scale_value)
                row_quant = txl.alloc_local([4], "uint16")
                for k in range(4):
                    quantize_bf16x2(row_quant[k], words[i * 4 + k], row_rcp_bf16x2)
                store_fp8x8(row_fp8, row, row_quant)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))

            col_rcp = txl.alloc_local([8], "uint32")
            txl.ptx.ld.shared.v4.b32(col_rcp[0], col_rcp[1], col_rcp[2], col_rcp[3], s_col_rcp.ptr_to([buf_idx, out_idx, col_group * 8]))
            txl.ptx.ld.shared.v4.b32(col_rcp[4], col_rcp[5], col_rcp[6], col_rcp[7], s_col_rcp.ptr_to([buf_idx, out_idx, col_group * 8 + 4]))
            col_rcp_bf16x2 = txl.alloc_local([4], "uint32")
            for k in range(4):
                txl.ptx["prmt.b32"](col_rcp_bf16x2[k], col_rcp[2 * k], col_rcp[2 * k + 1], txl.uint32(0x7632))
            for i in range(ROWS_PER_THREAD):
                row = row_base + ROW_STRIDE * i
                col_quant = txl.alloc_local([4], "uint16")
                for k in range(4):
                    quantize_bf16x2(col_quant[k], words[i * 4 + k], col_rcp_bf16x2[k])
                store_fp8x8(col_fp8, row, col_quant)

    def tile_coords(tile, col_group, row0):
        if f"{mode}:{N}" in CFG["signed"] or mode in CFG["signed"]:
            # The signed coordinate expression emits more address instructions, but schedules
            # registers better for these pipelines.
            row_block, col_block = tile // NUM_COL_TILES, tile % NUM_COL_TILES
            out_col = col_block * BLOCK + col_group * 8
            return row_block, col_block, out_col, row_block * BLOCK + row0
        # tile indices are non-negative: unsigned division, 64-bit row so per-row offsets fold into addresses
        tile_u32 = txl.cast(tile, "uint32")
        row_block = txl.cast(tile_u32 // txl.uint32(NUM_COL_TILES), "int32")
        col_block = txl.cast(tile_u32 % txl.uint32(NUM_COL_TILES), "int32")
        out_col = col_block * BLOCK + col_group * 8
        return row_block, col_block, out_col, txl.cast(row_block, "int64") * BLOCK + txl.cast(row0, "int64")

    def body_persistent(dh, gate, up, outs, MB, G):
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        col_group, row0 = tidx & 15, tidx >> 4
        cta = txl.cta_id()
        num_tiles = MB * NUM_COL_TILES
        smem = txl.smem_pool()
        # A unit ring (CFG["hu"], CFG["hp"]) splits each tile load into row parts pipelined
        # independently, so a part's load overlaps the copy of the previous part to registers,
        # without changing tile ownership or the quantization protocol. Widths not listed keep the
        # whole-tile schedule, which the split slows down.
        RING_UNITS = CFG["hu"].get(f"{mode}:{N}", CFG["hu"].get(mode, 0))
        ROW_PARTS = CFG["hp"].get(f"{mode}:{N}", CFG["hp"].get(mode, 2))  # row parts per tile load
        ROWS_PER_PART = ROWS_PER_THREAD // ROW_PARTS
        if RING_UNITS:
            s_ring = smem.alloc([RING_UNITS, num_in, BLOCK // ROW_PARTS, BLOCK], "bfloat16", align=128)
        else:
            s_ring = smem.alloc([STAGES, num_in, BLOCK, BLOCK], "bfloat16", align=128)
        smem_bufs = (smem.alloc([2, num_out, WARPS, BLOCK // 2], "uint32"), smem.alloc([2, num_out, BLOCK], "float32", align=16))
        sources = [gate, up] + ([dh] if is_bwd else [])

        def issue_unit(unit):
            """
            Issue ring unit unit = ROW_PARTS * tile_step + row_part.
            """
            tile = cta + (unit // ROW_PARTS) * G
            part, slot = unit % ROW_PARTS, unit % RING_UNITS
            with txl.if_(tile < num_tiles), txl.then_():
                _, _, out_col, row_base = tile_coords(tile, col_group, row0)
                col_ok = None if is_full else out_col < COL_LIMIT
                in_col = txl.Select(out_col < N, out_col, out_col - N) if is_select else out_col
                if not is_full:
                    in_col = txl.Select(col_ok, in_col, 0)
                for i in range(ROWS_PER_PART):
                    local_row = row0 + ROW_STRIDE * i
                    row = row_base + txl.cast(ROW_STRIDE * ROWS_PER_PART * part, "int64") + ROW_STRIDE * i
                    for src_idx, src in enumerate(sources):
                        col = in_col + UP_OFFSET if src_idx == 1 else in_col
                        dst = s_ring.ptr_to([slot, src_idx, local_row, col_group * 8])
                        if col_ok is None:
                            txl.ptx["cp.async.cg.shared.global"](dst, src.ptr_to([row, col]), 16, 16)
                        else:
                            txl.ptx["cp.async.cg.shared.global"](dst, src.ptr_to([row, col]), 16, txl.ptx.pred(out_col >= COL_LIMIT))
            txl.ptx["cp.async.commit_group"]()

        def issue_tile(step):
            tile = cta + step * G
            slot = step % STAGES
            with txl.if_(tile < num_tiles), txl.then_():
                _, _, out_col, row_base = tile_coords(tile, col_group, row0)
                col_ok = None if is_full else out_col < COL_LIMIT
                in_col = txl.Select(out_col < N, out_col, out_col - N) if is_select else out_col
                if not is_full:
                    in_col = txl.Select(col_ok, in_col, 0)
                for i in range(ROWS_PER_THREAD):
                    local_row = row0 + ROW_STRIDE * i
                    row = row_base + ROW_STRIDE * i
                    for src_idx, src in enumerate(sources):
                        col = in_col + UP_OFFSET if src_idx == 1 else in_col
                        dst = s_ring.ptr_to([slot, src_idx, local_row, col_group * 8])
                        if col_ok is None:
                            txl.ptx["cp.async.cg.shared.global"](dst, src.ptr_to([row, col]), 16, 16)
                        else:
                            txl.ptx["cp.async.cg.shared.global"](dst, src.ptr_to([row, col]), 16, txl.ptx.pred(out_col >= COL_LIMIT))
            txl.ptx["cp.async.commit_group"]()

        for prologue_idx in range(RING_UNITS if RING_UNITS else STAGES):
            (issue_unit if RING_UNITS else issue_tile)(prologue_idx)
        step = txl.local_scalar("int32")
        txl.assign(step, 0)
        out_words = [txl.alloc_local([WORDS], "uint32") for _ in range(num_out)]
        gate_words, up_words = txl.alloc_local([WORDS], "uint32"), txl.alloc_local([WORDS], "uint32")
        grad_words = txl.alloc_local([WORDS], "uint32") if is_bwd else None
        word_bufs = [gate_words, up_words] + ([grad_words] if is_bwd else [])
        with txl.while_(cta + step * G < num_tiles):
            tile = cta + step * G
            if RING_UNITS:
                for part in range(ROW_PARTS):
                    unit = ROW_PARTS * step + part
                    txl.ptx["cp.async.wait_group"](RING_UNITS - 1)
                    for i2 in range(ROWS_PER_PART):
                        i = part * ROWS_PER_PART + i2
                        local_row = row0 + ROW_STRIDE * i2
                        for src_idx in range(num_in):
                            dst = word_bufs[src_idx]
                            txl.ptx.ld.shared.v4.b32(dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], s_ring.ptr_to([unit % RING_UNITS, src_idx, local_row, col_group * 8]))
                    issue_unit(unit + RING_UNITS)
            else:
                slot = step % STAGES
                txl.ptx["cp.async.wait_group"](STAGES - 1)
                for i in range(ROWS_PER_THREAD):
                    local_row = row0 + ROW_STRIDE * i
                    for src_idx in range(num_in):
                        dst = word_bufs[src_idx]
                        txl.ptx.ld.shared.v4.b32(dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], s_ring.ptr_to([slot, src_idx, local_row, col_group * 8]))
                issue_tile(step + STAGES)
            row_block, col_block, out_col, row_base = tile_coords(tile, col_group, row0)
            swiglu(is_bwd, is_select, N, out_col, gate_words, up_words, grad_words, out_words, WORDS)
            quant_store(out_words, outs, row_block, col_block, out_col, row_base, smem_bufs, tidx, lane, warp, col_group, step % 2)
            txl.assign(step, step + 1)
        txl.ptx["cp.async.wait_group"](0)

    def body_simple(dh, gate, up, outs, MB, G):
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        col_group, row0 = tidx & 15, tidx >> 4
        row_block, col_block, out_col, row_base = tile_coords(txl.cta_id(), col_group, row0)
        smem = txl.smem_pool()
        smem_bufs = (smem.alloc([1, num_out, WARPS, BLOCK // 2], "uint32"), smem.alloc([1, num_out, BLOCK], "float32", align=16))
        out_words = [txl.alloc_local([WORDS], "uint32") for _ in range(num_out)]
        gate_words, up_words = txl.alloc_local([WORDS], "uint32"), txl.alloc_local([WORDS], "uint32")
        grad_words = txl.alloc_local([WORDS], "uint32") if is_bwd else None

        def load_words(dst, tensor, row, col, word0, pred, sub):
            addr = tensor.ptr_to([row, col])
            pred_kwargs = {} if pred is None else {"pred": pred}
            if vec_width == 1:
                half = txl.local_scalar("uint16")
                txl.ptx.ld.global_.b16(half, addr, **pred_kwargs)
                value = txl.cast(half, "uint32") if pred is None else txl.Select(pred, txl.cast(half, "uint32"), txl.uint32(0))
                if sub % 2 == 0:
                    txl.assign(dst[word0], value)
                else:
                    txl.assign(dst[word0], txl.bitwise_or(dst[word0], txl.shift_left(value, txl.uint32(16))))
                return
            if vec_width == 4:
                txl.ptx.ld.global_.v2.b32(dst[word0], dst[word0 + 1], addr, **pred_kwargs)
            else:
                txl.ptx.ld.global_.b32(dst[word0], addr, **pred_kwargs)
            if pred is not None:
                # a skipped predicated load leaves its destination undefined
                for word_idx in range(word0, word0 + vec_width // 2):
                    txl.assign(dst[word_idx], txl.Select(pred, dst[word_idx], txl.uint32(0)))

        for i in range(ROWS_PER_THREAD):
            row = row_base + ROW_STRIDE * i
            for sub in range(8 // vec_width):
                col = out_col + sub * vec_width
                word0 = i * 4 + (sub * vec_width) // 2
                col_ok = None if is_full else col < COL_LIMIT
                in_col = txl.Select(col < N, col, col - N) if is_select else col
                load_words(gate_words, gate, row, in_col, word0, col_ok, sub)
                load_words(up_words, up, row, in_col + UP_OFFSET, word0, col_ok, sub)
                if is_bwd:
                    load_words(grad_words, dh, row, in_col, word0, col_ok, sub)
        swiglu(is_bwd, is_select, N, out_col, gate_words, up_words, grad_words, out_words, WORDS)
        quant_store(out_words, outs, row_block, col_block, out_col, row_base, smem_bufs, tidx, lane, warp, col_group, 0)

    def body_window_range(dh, gate, up, outs, MB, G):
        """
        gu_bwd, N % 128 = HR != 0. Each CTA walks one contiguous range of input windows k (a part
        of a row block). Window k loads g, u, dh once and computes dg and du (one sigmoid). dg
        block k is complete. du block HQ + k is du columns >= HS of window k - 1 followed by
        columns < HS of window k: each thread keeps its du chunk of window k - 1 (in shared
        memory, private to the thread) and takes Select(thread_col < HS, du_k, du_{k-1}), so the
        block is assembled without moving data between threads. The first window of a range
        starting at k > 0 recomputes du of window k - 1 (columns >= HS) from a g/dh halo load;
        window HQ also loads du columns < HS of window 0 (aux) for the block where dg ends and du
        starts, and outputs the partial last du block.
        """
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        col_group, row0 = tidx & 15, tidx >> 4
        cta = txl.cta_id()
        num_windows = MB * NUM_COL_TILES
        smem = txl.smem_pool()
        s_ring = smem.alloc([5, BLOCK, BLOCK], "bfloat16", align=128)
        smem_bufs = (smem.alloc([2, 2, WARPS, BLOCK // 2], "uint32"), smem.alloc([2, 2, BLOCK], "float32", align=16))
        out = outs[0]
        thread_col = col_group * 8
        is_low_col = thread_col < HS

        # CTA cta owns the flattened windows flat = row_block * NUM_COL_TILES + window in
        # [window_start, window_end): one contiguous range, at most a few row-block segments; a
        # segment starting at window > 0 recomputes du of window - 1 (halo).
        # G >= MB: each row block is cut into num_parts = G // MB parts of part_len windows (the host
        # launches MB * num_parts CTAs); otherwise flat contiguous ranges of ceil(num_windows / G)
        # windows. No range starts at window HQ (it would need both a halo and the window-0 aux load).
        num_parts = local_value(G // MB)
        part_len = local_value((NUM_COL_TILES + num_parts - 1) // max_fmaxf(num_parts, 1))
        range_len = local_value((num_windows + G - 1) // G)

        def part_cut(part):
            window = min_fminf(part * part_len, NUM_COL_TILES)
            if HQ > 0:
                window = window + txl.Select(window == HQ, 1, 0)
            return window

        def range_start(cta_idx):
            flat = cta_idx * range_len
            if HQ > 0:
                flat = flat + txl.Select(flat % NUM_COL_TILES == HQ, 1, 0)
            return min_fminf(flat, num_windows)

        is_aligned = num_parts >= 1
        cta_row_block, cta_part = cta // max_fmaxf(num_parts, 1), cta % max_fmaxf(num_parts, 1)
        window_start = local_value(txl.Select(is_aligned, cta_row_block * NUM_COL_TILES + part_cut(cta_part), range_start(cta)))
        window_end = local_value(txl.Select(is_aligned, cta_row_block * NUM_COL_TILES + part_cut(cta_part + 1), range_start(cta + 1)))

        def split_window(flat):
            flat_u32 = local_value(txl.cast(flat, "uint32"), "uint32")
            return local_value(txl.cast(flat_u32 // txl.uint32(NUM_COL_TILES), "int32")), local_value(txl.cast(flat_u32 % txl.uint32(NUM_COL_TILES), "int32"))

        def issue_main(flat, window_ok):
            with txl.if_(window_ok), txl.then_():
                row_block, window = split_window(flat)
                row_base = txl.cast(row_block, "int64") * BLOCK + txl.cast(row0, "int64")
                out_col = window * BLOCK + thread_col
                col_ok = out_col < N
                col = txl.Select(col_ok, out_col, 0)
                for i in range(ROWS_PER_THREAD):
                    local_row = row0 + ROW_STRIDE * i
                    row = row_base + ROW_STRIDE * i
                    for slot, src, src_col in ((0, gate, col), (1, up, col + UP_OFFSET), (2, dh, col)):
                        txl.ptx["cp.async.cg.shared.global"](s_ring.ptr_to([slot, local_row, thread_col]), src.ptr_to([row, src_col]), 16, txl.ptx.pred(txl.Not(col_ok)))
            txl.ptx["cp.async.commit_group"]()

        def aux_coords(flat, window):
            # halo (first window of a segment, window > 0): window - 1, columns >= HS; window HQ: window 0, columns < HS
            is_halo = txl.And(flat == window_start, window > 0)
            is_last = window == HQ
            col = txl.Select(is_halo, (window - 1) * BLOCK + thread_col, thread_col - HR)
            col_ok = txl.Select(is_halo, txl.Not(is_low_col), txl.And(is_last, thread_col >= HR))
            col_ok = txl.And(col_ok, col < N)
            return txl.Select(col_ok, col, 0), col_ok, txl.Or(is_halo, is_last)

        def issue_aux(flat, window_ok):
            with txl.if_(window_ok), txl.then_():
                row_block, window = split_window(flat)
                row_base = txl.cast(row_block, "int64") * BLOCK + txl.cast(row0, "int64")
                col, col_ok, needs_aux = aux_coords(flat, window)
                col = local_value(col)
                col_ok_int = local_value(txl.Select(col_ok, 1, 0))
                col_ok = col_ok_int == 1
                with txl.if_(needs_aux), txl.then_():
                    for i in range(ROWS_PER_THREAD):
                        local_row = row0 + ROW_STRIDE * i
                        row = row_base + ROW_STRIDE * i
                        for slot, src in ((3, gate), (4, dh)):
                            txl.ptx["cp.async.cg.shared.global"](s_ring.ptr_to([slot, local_row, thread_col]), src.ptr_to([row, col]), 16, txl.ptx.pred(txl.Not(col_ok)))
            txl.ptx["cp.async.commit_group"]()

        def load_slots(word_bufs, slots):
            for i in range(ROWS_PER_THREAD):
                local_row = row0 + ROW_STRIDE * i
                for dst, slot in zip(word_bufs, slots):
                    txl.ptx.ld.shared.v4.b32(dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], s_ring.ptr_to([slot, local_row, thread_col]))

        def geometry_linear(window):  # dg block window, or (window == HQ) the block where dg ends and du starts: linear columns
            return {"ocol": window * BLOCK + thread_col, "sblk": window, "ccol": window * BLOCK + 2 * tidx, "sguard": None, "olim": K}

        def geometry_grad_up(block):  # du block (> HQ): its columns start HR into a window
            col_in_block = lambda col: txl.bitwise_and(col + HR, BLOCK - 1)
            return {"ocol": block * BLOCK + col_in_block(thread_col), "sblk": block, "ccol": block * BLOCK + col_in_block(2 * tidx), "sguard": block < NUM_OUT_BLOCKS, "olim": K}

        issue_main(window_start, window_start < window_end)
        issue_aux(window_start, window_start < window_end)
        flat = txl.local_scalar("int32")
        txl.assign(flat, window_start)
        out_words = [txl.alloc_local([WORDS], "uint32") for _ in range(2)]  # dg / du_k
        # du_{k-1} of this thread, kept in shared memory (only its owner reads or writes it)
        s_prev_grad_up = smem.alloc([WORDS // 4, THREADS, 4], "uint32", align=16)

        def store_prev(words):
            for quad in range(WORDS // 4):
                txl.ptx.st.shared.v4.b32(s_prev_grad_up.ptr_to([quad, tidx, 0]), words[4 * quad], words[4 * quad + 1], words[4 * quad + 2], words[4 * quad + 3])

        def load_prev(words):
            for quad in range(WORDS // 4):
                txl.ptx.ld.shared.v4.b32(words[4 * quad], words[4 * quad + 1], words[4 * quad + 2], words[4 * quad + 3], s_prev_grad_up.ptr_to([quad, tidx, 0]))

        gate_words, up_words, grad_words = txl.alloc_local([WORDS], "uint32"), txl.alloc_local([WORDS], "uint32"), txl.alloc_local([WORDS], "uint32")
        with txl.while_(flat < window_end):
            row_block, window = split_window(flat)
            row_base = txl.cast(row_block, "int64") * BLOCK + txl.cast(row0, "int64")
            next_flat = local_value(flat + 1)
            txl.ptx["cp.async.wait_group"](1)
            load_slots([gate_words, up_words, grad_words], [0, 1, 2])
            issue_main(next_flat, next_flat < window_end)
            swiglu(True, False, N, 0, gate_words, up_words, grad_words, [out_words[0], out_words[1]], WORDS)
            # du block HQ + k: du_k on lanes < HS, du_{k-1} (shared memory) on the others; du_k becomes
            # the next window's du_{k-1}
            load_prev(up_words)
            for word_idx in range(WORDS):
                txl.assign(up_words[word_idx], txl.Select(is_low_col, out_words[1][word_idx], up_words[word_idx]))
            store_prev(out_words[1])
            txl.ptx["cp.async.wait_group"](1)
            _, _, needs_aux = aux_coords(flat, window)
            load_slots([gate_words, grad_words], [3, 4])
            issue_aux(next_flat, next_flat < window_end)
            is_last = window == HQ
            is_halo = txl.And(flat == window_start, window > 0)
            with txl.if_(needs_aux), txl.then_():
                # halo: du_{k-1} for lanes >= HS; last window: du of window 0 (columns < HS) for the dg|du block
                for word_idx in range(WORDS):
                    word = grad_up_word(word_idx, gate_words, grad_words)
                    txl.assign(up_words[word_idx], txl.Select(txl.And(is_halo, txl.Not(is_low_col)), word, up_words[word_idx]))
                    txl.assign(out_words[0][word_idx], txl.Select(txl.And(is_last, thread_col >= HR), word, out_words[0][word_idx]))
            # (window 0 has no du block: its stores are masked off)
            geom_du = geometry_grad_up(window + HQ)
            has_du = window > 0
            geom_du = dict(geom_du, ocol=txl.Select(has_du, geom_du["ocol"], K), ccol=txl.Select(has_du, geom_du["ccol"], K), sguard=txl.And(geom_du["sguard"], has_du))
            quant_store(None, None, row_block, window, 0, row_base, smem_bufs, tidx, lane, warp, col_group, 0, geometry=[(out_words[0], out, geometry_linear(window)), (up_words, out, geom_du)])
            if HR > 64:
                # window HQ: du columns >= HS (valid < HR) form the partial last block 2HQ + 1
                with txl.if_(is_last), txl.then_():
                    load_prev(gate_words)
                    for word_idx in range(WORDS):
                        txl.assign(gate_words[word_idx], txl.Select(is_low_col, txl.uint32(0), gate_words[word_idx]))
                    quant_store(None, None, row_block, window, 0, row_base, smem_bufs, tidx, lane, warp, col_group, 1, geometry=[(gate_words, out, geometry_grad_up(2 * HQ + 1))])
            txl.assign(flat, next_flat)
        txl.ptx["cp.async.wait_group"](0)

    def body_interleaved_ranges(dh, gate, up, outs, MB, G):
        """
        gu_bwd, N % 128 = HR != 0, short interleaved window ranges. Each row block's windows are
        cut into RUN_R ranges of about RUN_L windows; CTA c walks ranges c, c + G, ...
        (consecutive CTAs take adjacent ranges of the same rows, so concurrent loads share DRAM
        pages) with one continuous cp.async pipeline across range boundaries. Per window as
        body_window_range: dg block k, and du block HQ + k from du_k (lanes < HS) and the thread's
        carried du_{k-1} (lanes >= HS); the first window of a range with k > 0 recomputes
        du_{k-1} from a g/dh halo load of window k - 1.
        """
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        col_group, row0 = tidx & 15, tidx >> 4
        cta = txl.cta_id()
        num_ranges = MB * RUN_R
        smem = txl.smem_pool()
        s_ring = smem.alloc([5, BLOCK, BLOCK], "bfloat16", align=128)
        smem_bufs = (smem.alloc([2, 2, WARPS, BLOCK // 2], "uint32"), smem.alloc([2, 2, BLOCK], "float32", align=16))
        out = outs[0]
        thread_col = col_group * 8
        is_low_col = thread_col < HS

        def range_cut(part):
            window = min_fminf(part * RUN_L, NUM_COL_TILES)
            if HQ > 0:
                window = window + txl.Select(window == HQ, 1, 0)
            return window

        def split_range(range_idx):
            range_u32 = local_value(txl.cast(range_idx, "uint32"), "uint32")
            row_block = local_value(txl.cast(range_u32 // txl.uint32(RUN_R), "int32"))
            part = local_value(txl.cast(range_u32 % txl.uint32(RUN_R), "int32"))
            return row_block, part

        def issue_main(row_block, window, window_ok):
            with txl.if_(window_ok), txl.then_():
                row_base = txl.cast(row_block, "int64") * BLOCK + txl.cast(row0, "int64")
                out_col = window * BLOCK + thread_col
                col_ok = out_col < N
                col = txl.Select(col_ok, out_col, 0)
                for i in range(ROWS_PER_THREAD):
                    local_row = row0 + ROW_STRIDE * i
                    row = row_base + ROW_STRIDE * i
                    for slot, src, src_col in ((0, gate, col), (1, up, col + UP_OFFSET), (2, dh, col)):
                        txl.ptx["cp.async.cg.shared.global"](s_ring.ptr_to([slot, local_row, thread_col]), src.ptr_to([row, src_col]), 16, txl.ptx.pred(txl.Not(col_ok)))
            txl.ptx["cp.async.commit_group"]()

        def aux_coords(is_first, window):
            # halo (first window of a range, window > 0): window - 1, columns >= HS; window HQ: window 0, columns < HS
            is_halo = txl.And(is_first == 1, window > 0)
            is_last = window == HQ
            col = txl.Select(is_halo, (window - 1) * BLOCK + thread_col, thread_col - HR)
            col_ok = txl.Select(is_halo, txl.Not(is_low_col), txl.And(is_last, thread_col >= HR))
            col_ok = txl.And(col_ok, col < N)
            return txl.Select(col_ok, col, 0), col_ok, txl.Or(is_halo, is_last)

        def issue_aux(row_block, window, is_first, window_ok):
            with txl.if_(window_ok), txl.then_():
                row_base = txl.cast(row_block, "int64") * BLOCK + txl.cast(row0, "int64")
                col, col_ok, needs_aux = aux_coords(is_first, window)
                col = local_value(col)
                col_ok_int = local_value(txl.Select(col_ok, 1, 0))
                col_ok = col_ok_int == 1
                with txl.if_(needs_aux), txl.then_():
                    for i in range(ROWS_PER_THREAD):
                        local_row = row0 + ROW_STRIDE * i
                        row = row_base + ROW_STRIDE * i
                        for slot, src in ((3, gate), (4, dh)):
                            txl.ptx["cp.async.cg.shared.global"](s_ring.ptr_to([slot, local_row, thread_col]), src.ptr_to([row, col]), 16, txl.ptx.pred(txl.Not(col_ok)))
            txl.ptx["cp.async.commit_group"]()

        def load_slots(word_bufs, slots):
            for i in range(ROWS_PER_THREAD):
                local_row = row0 + ROW_STRIDE * i
                for dst, slot in zip(word_bufs, slots):
                    txl.ptx.ld.shared.v4.b32(dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], s_ring.ptr_to([slot, local_row, thread_col]))

        def geometry_linear(window):
            return {"ocol": window * BLOCK + thread_col, "sblk": window, "ccol": window * BLOCK + 2 * tidx, "sguard": None, "olim": K}

        def geometry_grad_up(block):
            col_in_block = lambda col: txl.bitwise_and(col + HR, BLOCK - 1)
            return {"ocol": block * BLOCK + col_in_block(thread_col), "sblk": block, "ccol": block * BLOCK + col_in_block(2 * tidx), "sguard": block < NUM_OUT_BLOCKS, "olim": K}

        # walk state: range, window, range end, first window of its range, row block
        range_idx, window, window_end = txl.local_scalar("int32"), txl.local_scalar("int32"), txl.local_scalar("int32")
        is_first, cur_row_block = txl.local_scalar("int32"), txl.local_scalar("int32")
        txl.assign(range_idx, cta)
        row_block0, part0 = split_range(cta)
        txl.assign(cur_row_block, row_block0)
        txl.assign(window, range_cut(part0))
        txl.assign(window_end, range_cut(part0 + 1))
        txl.assign(is_first, 1)
        issue_main(cur_row_block, window, range_idx < num_ranges)
        issue_aux(cur_row_block, window, is_first, range_idx < num_ranges)
        out_words = [txl.alloc_local([WORDS], "uint32") for _ in range(2)]
        s_prev_grad_up = smem.alloc([WORDS // 4, THREADS, 4], "uint32", align=16)

        def store_prev(words):
            for quad in range(WORDS // 4):
                txl.ptx.st.shared.v4.b32(s_prev_grad_up.ptr_to([quad, tidx, 0]), words[4 * quad], words[4 * quad + 1], words[4 * quad + 2], words[4 * quad + 3])

        def load_prev(words):
            for quad in range(WORDS // 4):
                txl.ptx.ld.shared.v4.b32(words[4 * quad], words[4 * quad + 1], words[4 * quad + 2], words[4 * quad + 3], s_prev_grad_up.ptr_to([quad, tidx, 0]))

        gate_words, up_words, grad_words = txl.alloc_local([WORDS], "uint32"), txl.alloc_local([WORDS], "uint32"), txl.alloc_local([WORDS], "uint32")
        with txl.while_(range_idx < num_ranges):
            row_block = cur_row_block
            row_base = txl.cast(row_block, "int64") * BLOCK + txl.cast(row0, "int64")
            # next window of this CTA's walk
            is_same_range = local_value(txl.Select(window + 1 < window_end, 1, 0))
            next_range = local_value(txl.Select(is_same_range == 1, range_idx, range_idx + G))
            next_row_block, next_part = split_range(next_range)
            next_row_block = local_value(txl.Select(is_same_range == 1, row_block, next_row_block))
            next_window = local_value(txl.Select(is_same_range == 1, window + 1, range_cut(next_part)))
            next_window_end = local_value(txl.Select(is_same_range == 1, window_end, range_cut(next_part + 1)))
            next_is_first = local_value(1 - is_same_range)
            txl.ptx["cp.async.wait_group"](1)
            load_slots([gate_words, up_words, grad_words], [0, 1, 2])
            issue_main(next_row_block, next_window, next_range < num_ranges)
            swiglu(True, False, N, 0, gate_words, up_words, grad_words, [out_words[0], out_words[1]], WORDS)
            load_prev(up_words)
            for word_idx in range(WORDS):
                txl.assign(up_words[word_idx], txl.Select(is_low_col, out_words[1][word_idx], up_words[word_idx]))
            store_prev(out_words[1])
            txl.ptx["cp.async.wait_group"](1)
            _, _, needs_aux = aux_coords(is_first, window)
            load_slots([gate_words, grad_words], [3, 4])
            issue_aux(next_row_block, next_window, next_is_first, next_range < num_ranges)
            is_last = window == HQ
            is_halo = txl.And(is_first == 1, window > 0)
            with txl.if_(needs_aux), txl.then_():
                for word_idx in range(WORDS):
                    word = grad_up_word(word_idx, gate_words, grad_words)
                    txl.assign(up_words[word_idx], txl.Select(txl.And(is_halo, txl.Not(is_low_col)), word, up_words[word_idx]))
                    txl.assign(out_words[0][word_idx], txl.Select(txl.And(is_last, thread_col >= HR), word, out_words[0][word_idx]))
            geom_du = geometry_grad_up(window + HQ)
            has_du = window > 0
            geom_du = dict(geom_du, ocol=txl.Select(has_du, geom_du["ocol"], K), ccol=txl.Select(has_du, geom_du["ccol"], K), sguard=txl.And(geom_du["sguard"], has_du))
            quant_store(None, None, row_block, window, 0, row_base, smem_bufs, tidx, lane, warp, col_group, 0, geometry=[(out_words[0], out, geometry_linear(window)), (up_words, out, geom_du)])
            if HR > 64:
                with txl.if_(is_last), txl.then_():
                    load_prev(gate_words)
                    for word_idx in range(WORDS):
                        txl.assign(gate_words[word_idx], txl.Select(is_low_col, txl.uint32(0), gate_words[word_idx]))
                    quant_store(None, None, row_block, window, 0, row_base, smem_bufs, tidx, lane, warp, col_group, 1, geometry=[(gate_words, out, geometry_grad_up(2 * HQ + 1))])
            txl.assign(range_idx, next_range)
            txl.assign(cur_row_block, next_row_block)
            txl.assign(window, next_window)
            txl.assign(window_end, next_window_end)
            txl.assign(is_first, next_is_first)
        txl.ptx["cp.async.wait_group"](0)

    RUN_L = CFG["run_l"]
    if is_run and RUN_L:
        cuts = sorted({min(part * RUN_L, NUM_COL_TILES) + (1 if HQ > 0 and min(part * RUN_L, NUM_COL_TILES) == HQ else 0) for part in range(NUM_COL_TILES // RUN_L + 2)})
        RUN_R = len(cuts) - 1  # non-empty ranges per row block (the formula is monotone; empty ones only at the end)
        assert all(cuts[i] == min(i * RUN_L, NUM_COL_TILES) + (1 if HQ > 0 and min(i * RUN_L, NUM_COL_TILES) == HQ else 0) for i in range(RUN_R + 1))
    body = ((body_interleaved_ranges if RUN_L else body_window_range) if is_run else body_persistent) if is_persistent else body_simple

    def tensor_arg(dtype, shape):
        return txl.Tensor(shape, dtype, data_alignment=1)

    var_M, var_MB = txl.Var("M", "int32"), txl.Var("MB", "int32")
    FP8_SHAPE, SCALE_SHAPE, COL_SCALE_SHAPE = (var_M, K), (var_M, NUM_OUT_BLOCKS), (var_MB, K)
    kernel_attrs = txl.cuda.KernelAttributes(min_blocks_per_sm=OCCUPANCY) if is_persistent else None

    if mode in ("fwd", "gu_fwd"):

        @txl.kernel()
        def kern(
            g: tensor_arg("bfloat16", (var_M, IN_WIDTH)),
            u: tensor_arg("bfloat16", (var_M, IN_WIDTH)),
            q0: tensor_arg("float8_e4m3fn", FP8_SHAPE),
            s0: tensor_arg("float32", SCALE_SHAPE),
            c0: tensor_arg("float8_e4m3fn", FP8_SHAPE),
            cs0: tensor_arg("float32", COL_SCALE_SHAPE),
            M: var_M,
            MB: var_MB,
            G: txl.int32,
        ):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G, block=WARPS * 32), kernel_attrs=kernel_attrs)
            body(None, g, u, [(q0, s0, c0, cs0)], MB, G)

    elif mode == "bwd":

        @txl.kernel()
        def kern(
            dh: tensor_arg("bfloat16", (var_M, N)),
            g: tensor_arg("bfloat16", (var_M, N)),
            u: tensor_arg("bfloat16", (var_M, N)),
            q0: tensor_arg("float8_e4m3fn", FP8_SHAPE),
            s0: tensor_arg("float32", SCALE_SHAPE),
            c0: tensor_arg("float8_e4m3fn", FP8_SHAPE),
            cs0: tensor_arg("float32", COL_SCALE_SHAPE),
            q1: tensor_arg("float8_e4m3fn", FP8_SHAPE),
            s1: tensor_arg("float32", SCALE_SHAPE),
            c1: tensor_arg("float8_e4m3fn", FP8_SHAPE),
            cs1: tensor_arg("float32", COL_SCALE_SHAPE),
            M: var_M,
            MB: var_MB,
            G: txl.int32,
        ):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G, block=WARPS * 32), kernel_attrs=kernel_attrs)
            body(dh, g, u, [(q0, s0, c0, cs0), (q1, s1, c1, cs1)], MB, G)

    else:

        @txl.kernel()
        def kern(
            dh: tensor_arg("bfloat16", (var_M, N)),
            g: tensor_arg("bfloat16", (var_M, 2 * N)),
            u: tensor_arg("bfloat16", (var_M, 2 * N)),
            q0: tensor_arg("float8_e4m3fn", FP8_SHAPE),
            s0: tensor_arg("float32", SCALE_SHAPE),
            c0: tensor_arg("float8_e4m3fn", FP8_SHAPE),
            cs0: tensor_arg("float32", COL_SCALE_SHAPE),
            M: var_M,
            MB: var_MB,
            G: txl.int32,
        ):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G, block=WARPS * 32), kernel_attrs=kernel_attrs)
            outs = (q0, s0, c0, cs0)
            body(dh, g, u, [outs, outs] if is_pair else [outs], MB, G)

    fn = entry_function(kern.compile(TARGET))
    return fn, is_persistent, NUM_COL_TILES, OCCUPANCY, (RUN_R if is_run and RUN_L else 0)


@functools.cache
def device_sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def persistent_grid(num_tiles, occupancy, num_sms):
    capacity = occupancy * num_sms
    if occupancy == 1:
        # one resident CTA per SM: equalize per-CTA tile counts instead of a short final wave
        return cdiv(num_tiles, cdiv(num_tiles, capacity))
    return min(num_tiles, capacity)


def launch(mode, N, M, device, *args):
    fn, is_persistent, num_col_tiles, occupancy, num_run_ranges = build_kernel(mode, N)
    MB = M // BLOCK
    G = persistent_grid(MB * num_col_tiles, occupancy, device_sm_count(device.index)) if is_persistent else MB * num_col_tiles
    if num_run_ranges:
        G = persistent_grid(MB * num_run_ranges, 1, device_sm_count(device.index))
    elif mode == "gu_bwd_run":
        num_sms = device_sm_count(device.index)
        G = MB * (num_sms // MB) if MB <= num_sms else G  # whole row-block parts (see body_window_range)
    fn(*args, M, MB, G)


def empty_quant(M, K, device):
    return (
        torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M, cdiv(K, BLOCK)), dtype=torch.float32, device=device),
        torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M // BLOCK, K), dtype=torch.float32, device=device),
    )


def fp8_silu_mul_fwd_sm100(g, u, grouped_mm_offs):
    """
    SM100 fp8_silu_mul_fwd; grouped_mm_offs is unused (groups are 128-row aligned).
    """
    M, N = g.shape
    out = empty_quant(M, N, g.device)
    if M == 0 or N == 0:
        return out
    launch("fwd", N, M, g.device, g, u, *out)
    return out


def fp8_silu_mul_bwd_sm100(dh, g, u, grouped_mm_offs):
    """
    SM100 fp8_silu_mul_bwd; grouped_mm_offs is unused (groups are 128-row aligned).
    """
    M, N = g.shape
    grad_gate_out, grad_up_out = empty_quant(M, N, g.device), empty_quant(M, N, g.device)
    if M == 0 or N == 0:
        return grad_gate_out, grad_up_out
    launch("bwd", N, M, g.device, dh, g, u, *grad_gate_out, *grad_up_out)
    return grad_gate_out, grad_up_out


def fp8_silu_mul_gu_fwd_sm100(gu, grouped_mm_offs):
    """
    SM100 fp8_silu_mul_gu_fwd; grouped_mm_offs is unused (groups are 128-row aligned).
    """
    M, N = gu.shape[0], gu.shape[1] // 2
    out = empty_quant(M, N, gu.device)
    if M == 0 or N == 0:
        return out
    launch("gu_fwd", N, M, gu.device, gu, gu, *out)
    return out


def fp8_silu_mul_gu_bwd_sm100(dh, gu, grouped_mm_offs):
    """
    SM100 fp8_silu_mul_gu_bwd; grouped_mm_offs is unused (groups are 128-row aligned).
    """
    M, N = gu.shape[0], gu.shape[1] // 2
    out = empty_quant(M, 2 * N, gu.device)
    if M == 0 or N == 0:
        return out
    mode = "gu_bwd_pair" if N % BLOCK == 0 else "gu_bwd_run" if N % 8 == 0 else "gu_bwd_sel"
    launch(mode, N, M, gu.device, dh, gu, gu, *out)
    return out
