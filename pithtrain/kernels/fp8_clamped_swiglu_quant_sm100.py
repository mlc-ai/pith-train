"""
SM100 fp8_clamped_swiglu_quant: fused TIRx-lite kernels, packed math on 128x128 tiles.

One CTA (8 warps) owns a 128-row x 128-output-column tile. Each thread holds 8 consecutive
output columns of 8 rows (rows row_group + 16 i). The clamped SwiGLU (forward) or its gradient
(backward) is clamped_swiglu's fp32 sequence bit for bit, using cheaper exact equivalents:
ex2.approx.ftz (only differs where 1 + e rounds to 1 anyway) and div.full(1, d) ==
0.25 * rcp.approx(0.25 * d) for d >= 1 (div.full's own range scaling), with independent element
pairs in f32x2 instructions. Quantization multiplies the bf16 pairs by the exact power-of-two
reciprocal in bf16x2 and converts bf16x2 -> e4m3x2.

Backward writes fp32 128-row block sums. The persistent TMA-ring kernel (N % 8 == 0, at most 512
groups) folds them into d_bias per group inside the same launch: each CTA publishes its partials
through a GPU-scope exit word once every persistent CTA has started, and late-exiting CTAs reduce
them in block order (the last CTA out is the fallback). The register-direct fallback kernel
reduces them with a second kernel.
"""

import contextlib
import functools
import struct

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["clamped_swiglu_quant_fwd_sm100", "fp8_clamped_swiglu_rowwise_sm100", "fp8_clamped_swiglu_kmajor_sm100", "fp8_clamped_swiglu_bwd_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


BLOCK = 128
BWD_SPLIT_TILES_PER_CTA = 30  # backward split arrival below this many tiles per CTA
WARPS = 8
NEG_LOG2E = -1.4426950408889634
FP8_MAX_RCP = 1.0 / 448.0
AMAX_FLOOR = 1e-4
FULL_MASK = 0xFFFFFFFF


def ptx_f32(op, *args):
    """
    Emit PTX instruction op into a fresh f32 register and return it.
    """
    out = txl.local_scalar("float32")
    txl.ptx[op](out, *args)
    return out


def ptx_u32(op, *args):
    """
    Emit PTX instruction op into a fresh u32 register and return it.
    """
    out = txl.local_scalar("uint32")
    txl.ptx[op](out, *args)
    return out


def lo_f32(word):
    return txl.reinterpret(txl.shift_left(word, txl.uint32(16)), ty="float32")


def hi_f32(word):
    return txl.reinterpret(txl.bitwise_and(word, txl.uint32(0xFFFF0000)), ty="float32")


def pack_f32x2(lo, hi):
    out = txl.local_scalar("uint64")
    txl.ptx.mov.b64(out, lo, hi)
    return out


def unpack_f32x2(pair):
    lo = txl.local_scalar("float32")
    hi = txl.local_scalar("float32")
    txl.ptx.mov.b64(lo, hi, pair)
    return lo, hi


def ptx_f32x2(op, *args):
    """
    Emit the f32x2 form of PTX instruction op into a fresh packed register and return it.
    """
    out = txl.local_scalar("uint64")
    txl.ptx[op + ".f32x2"](out, *args)
    return out


class SwigluConstants:
    """
    Registers every clamped SwiGLU pair reads: the f32x2 constants, alpha as a pair, the f32 limit
    and its negation, and the bf16x2 clamp bounds (upper: limit in both lanes; lower: NaN for the
    gate lane, -limit for the up lane).
    """

    def __init__(self, alpha, limit):
        self.limit = limit
        self.neg_limit = ptx_f32("neg.f32", limit)
        self.neg_log2e_x2 = pack_f32x2(txl.float32(NEG_LOG2E), txl.float32(NEG_LOG2E))
        self.quarter_x2 = pack_f32x2(txl.float32(0.25), txl.float32(0.25))
        self.one_x2 = pack_f32x2(txl.float32(1.0), txl.float32(1.0))
        self.minus_one_x2 = pack_f32x2(txl.float32(-1.0), txl.float32(-1.0))
        self.alpha_x2 = pack_f32x2(alpha, alpha)
        limit_bf16 = txl.shift_right(txl.reinterpret(limit, ty="uint32"), txl.uint32(16))
        self.upper_bf16x2 = txl.local_scalar("uint32", init=txl.bitwise_or(limit_bf16, txl.shift_left(limit_bf16, txl.uint32(16))))
        self.lower_bf16x2 = txl.local_scalar("uint32", init=txl.bitwise_or(txl.uint32(0x7FC0), txl.shift_left(txl.bitwise_or(limit_bf16, txl.uint32(0x8000)), txl.uint32(16))))


def compute_sigmoid_x2(clamped_gate_x2, swiglu):
    """
    (alpha * gate, sigmoid(alpha * gate)) for a pair, bit-identical to clamped_swiglu's
    1 / (1 + ex2.approx(alpha * gate * -log2e)) with div.full.
    """
    scaled_gate_x2 = ptx_f32x2("mul.rn", swiglu.alpha_x2, clamped_gate_x2)
    exponent_x2 = ptx_f32x2("mul.rn", scaled_gate_x2, swiglu.neg_log2e_x2)
    exponent_lo, exponent_hi = unpack_f32x2(exponent_x2)
    exp2_x2 = pack_f32x2(ptx_f32("ex2.approx.ftz.f32", exponent_lo), ptx_f32("ex2.approx.ftz.f32", exponent_hi))
    quarter_denom_x2 = ptx_f32x2("fma.rn", exp2_x2, swiglu.quarter_x2, swiglu.quarter_x2)
    quarter_denom_lo, quarter_denom_hi = unpack_f32x2(quarter_denom_x2)
    rcp_x2 = pack_f32x2(ptx_f32("rcp.approx.ftz.f32", quarter_denom_lo), ptx_f32("rcp.approx.ftz.f32", quarter_denom_hi))
    return scaled_gate_x2, ptx_f32x2("mul.rn", rcp_x2, swiglu.quarter_x2)


def clamp_gate_up_pair(word0, word1, swiglu):
    """
    Unpack two biased gate/up words to f32 and clamp them in f32: (gates, ups, clamped gate pair,
    clamped up pair).
    """
    gate0, gate1 = lo_f32(word0), lo_f32(word1)
    up0, up1 = hi_f32(word0), hi_f32(word1)
    clamped_gate_x2 = pack_f32x2(ptx_f32("min.f32", gate0, swiglu.limit), ptx_f32("min.f32", gate1, swiglu.limit))
    clamped_up_x2 = pack_f32x2(ptx_f32("min.f32", ptx_f32("max.f32", up0, swiglu.neg_limit), swiglu.limit), ptx_f32("min.f32", ptx_f32("max.f32", up1, swiglu.neg_limit), swiglu.limit))
    return (gate0, gate1), (up0, up1), clamped_gate_x2, clamped_up_x2


def swiglu_fwd_pair(word0, word1, swiglu):
    """
    h for two adjacent channels (biased gate/up words word0, word1) -> bf16x2 word.
    """
    _, _, clamped_gate_x2, clamped_up_x2 = clamp_gate_up_pair(word0, word1, swiglu)
    _, sigmoid_x2 = compute_sigmoid_x2(clamped_gate_x2, swiglu)
    glu_x2 = ptx_f32x2("mul.rn", clamped_gate_x2, sigmoid_x2)
    h_x2 = ptx_f32x2("mul.rn", ptx_f32x2("add.rn", clamped_up_x2, swiglu.one_x2), glu_x2)
    h_lo, h_hi = unpack_f32x2(h_x2)
    return ptx_u32("cvt.rn.bf16x2.f32", h_hi, h_lo)


def swiglu_bwd_pair(word0, word1, grad_h_word, swiglu):
    """
    (grad_gate, grad_up) words for two adjacent h channels; grad_h_word = their dh bf16x2.
    """
    (gate0, gate1), (up0, up1), clamped_gate_x2, clamped_up_x2 = clamp_gate_up_pair(word0, word1, swiglu)
    scaled_gate_x2, sigmoid_x2 = compute_sigmoid_x2(clamped_gate_x2, swiglu)
    grad_h_x2 = pack_f32x2(lo_f32(grad_h_word), hi_f32(grad_h_word))
    glu_x2 = ptx_f32x2("mul.rn", clamped_gate_x2, sigmoid_x2)
    grad_up_x2 = ptx_f32x2("mul.rn", glu_x2, grad_h_x2)
    grad_glu_x2 = ptx_f32x2("mul.rn", ptx_f32x2("add.rn", clamped_up_x2, swiglu.one_x2), grad_h_x2)
    grad_sigmoid_x2 = ptx_f32x2("mul.rn", grad_glu_x2, sigmoid_x2)
    one_minus_sigmoid_x2 = ptx_f32x2("fma.rn", sigmoid_x2, swiglu.minus_one_x2, swiglu.one_x2)
    silu_factor_x2 = ptx_f32x2("fma.rn", scaled_gate_x2, one_minus_sigmoid_x2, swiglu.one_x2)
    grad_gate_x2 = ptx_f32x2("mul.rn", grad_sigmoid_x2, silu_factor_x2)
    grad_gate, grad_up = unpack_f32x2(grad_gate_x2), unpack_f32x2(grad_up_x2)
    words = []
    for j, (gate, up) in enumerate(((gate0, up0), (gate1, up1))):
        gate_in_range = txl.local_scalar("uint32")
        txl.ptx.setp.ge.f32(gate_in_range, swiglu.limit, gate)
        up_in_range = txl.local_scalar("uint32")
        txl.ptx.setp.ge.f32(up_in_range, up, swiglu.neg_limit)
        up_below_limit = txl.local_scalar("uint32")
        txl.ptx.setp.ge.f32(up_below_limit, swiglu.limit, up)
        txl.ptx.and_.pred(up_in_range, txl.ptx.pred(up_in_range), txl.ptx.pred(up_below_limit))
        masked_grad_gate = txl.local_scalar("float32")
        txl.ptx.selp.f32(masked_grad_gate, grad_gate[j], txl.float32(0.0), txl.ptx.pred(gate_in_range))
        masked_grad_up = txl.local_scalar("float32")
        txl.ptx.selp.f32(masked_grad_up, grad_up[j], txl.float32(0.0), txl.ptx.pred(up_in_range))
        words.append(ptx_u32("cvt.rn.bf16x2.f32", masked_grad_up, masked_grad_gate))
    return words


def amax_bf16x2(a, b):
    """
    max(|a|, |b|) per bf16 lane (sign is garbage; NaN ignored unless both).
    """
    return ptx_u32("max.xorsign.abs.bf16x2", a, b)


def abs_bf16x2(word):
    return txl.bitwise_and(word, txl.uint32(0x7FFF7FFF))


def shfl_xor(value, lane_xor):
    peer = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(peer, value, txl.uint32(lane_xor), txl.uint32(31), txl.uint32(FULL_MASK))
    return peer


def scale_ratio(amax_f32):
    """
    max(amax, 1e-4) / 448, the value the E8M0 scale rounds up.
    """
    return ptx_f32("mul.rn.f32", ptx_f32("max.f32", amax_f32, txl.float32(AMAX_FLOOR)), txl.float32(FP8_MAX_RCP))


def e8m0_pair(word):
    """
    bf16x2 amax word -> u32 with the two E8M0 exponents (lo lane in bits 0..7, hi in 8..15).
    amax >= 1e-4 keeps every exponent >= 105, so no 2^-126 floor is needed.
    """
    e8m0_x2 = txl.local_scalar("uint16")
    txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(e8m0_x2, scale_ratio(hi_f32(word)), scale_ratio(lo_f32(word)))
    return txl.Cast("uint32", e8m0_x2)


def e8m0_to_scale(e8m0):
    return txl.reinterpret(txl.shift_left(e8m0, txl.uint32(23)), ty="float32")


def e8m0_to_rcp_bf16(e8m0):
    """
    bf16 bits of 2^(127 - e8m0) (the exact reciprocal of the scale 2^(e8m0 - 127)).
    """
    return txl.shift_left(txl.uint32(254) - e8m0, txl.uint32(7))


def quantize_fp8x4(word0, word1, rcp0, rcp1):
    """
    Quantize two bf16x2 words with bf16x2 reciprocals -> 4 e4m3 bytes.
    """
    fp8_pairs = []
    for word, rcp in ((word0, rcp0), (word1, rcp1)):
        scaled = ptx_u32("mul.rn.bf16x2", word, rcp)
        pair = txl.local_scalar("uint16")
        txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(pair, scaled)
        fp8_pairs.append(txl.Cast("uint32", pair))
    return txl.bitwise_or(fp8_pairs[0], txl.shift_left(fp8_pairs[1], txl.uint32(16)))


def split_bf16x2(word):
    lo = txl.local_scalar("uint16")
    hi = txl.local_scalar("uint16")
    txl.ptx.mov.b32(lo, hi, word)
    return lo, hi


def clamp_gate_up_bf16(word, swiglu):
    """
    (min(gate, limit), clamp(up, -limit, limit)) on a biased bf16x2 word (limit bf16-exact).
    """
    # max first (NaN up -> -limit); the gate lane's max operand is NaN, the identity of a
    # non-NaN max, so a NaN gate stays NaN and the min turns it into limit, like min.f32.
    return ptx_u32("min.bf16x2", ptx_u32("max.bf16x2", word, swiglu.lower_bf16x2), swiglu.upper_bf16x2)


def swiglu_fwd_pair_bf16(clamped0, clamped1, swiglu):
    """
    h for two channels from clamped words clamped0, clamped1 -> bf16x2 word.
    """
    clamped_gate_x2 = pack_f32x2(lo_f32(clamped0), lo_f32(clamped1))
    _, sigmoid_x2 = compute_sigmoid_x2(clamped_gate_x2, swiglu)
    glu_x2 = ptx_f32x2("mul.rn", clamped_gate_x2, sigmoid_x2)
    up_plus_one = []
    for clamped in (clamped0, clamped1):
        _, up_bits = split_bf16x2(clamped)
        up_plus_one.append(ptx_f32("add.rn.f32.bf16", up_bits, txl.float32(1.0)))
    h_x2 = ptx_f32x2("mul.rn", pack_f32x2(up_plus_one[0], up_plus_one[1]), glu_x2)
    h_lo, h_hi = unpack_f32x2(h_x2)
    return ptx_u32("cvt.rn.bf16x2.f32", h_hi, h_lo)


def swiglu_bwd_pair_bf16(word0, word1, clamped0, clamped1, grad_h_word, swiglu):
    """
    (grad_gate, grad_up) words for two h channels; zero where the clamp was active.
    """
    clamped_gate_x2 = pack_f32x2(lo_f32(clamped0), lo_f32(clamped1))
    scaled_gate_x2, sigmoid_x2 = compute_sigmoid_x2(clamped_gate_x2, swiglu)
    grad_h_x2 = pack_f32x2(lo_f32(grad_h_word), hi_f32(grad_h_word))
    up_plus_one = []
    for clamped in (clamped0, clamped1):
        _, up_bits = split_bf16x2(clamped)
        up_plus_one.append(ptx_f32("add.rn.f32.bf16", up_bits, txl.float32(1.0)))
    glu_x2 = ptx_f32x2("mul.rn", clamped_gate_x2, sigmoid_x2)
    grad_up_x2 = ptx_f32x2("mul.rn", glu_x2, grad_h_x2)
    grad_glu_x2 = ptx_f32x2("mul.rn", pack_f32x2(up_plus_one[0], up_plus_one[1]), grad_h_x2)
    grad_sigmoid_x2 = ptx_f32x2("mul.rn", grad_glu_x2, sigmoid_x2)
    one_minus_sigmoid_x2 = ptx_f32x2("fma.rn", sigmoid_x2, swiglu.minus_one_x2, swiglu.one_x2)
    silu_factor_x2 = ptx_f32x2("fma.rn", scaled_gate_x2, one_minus_sigmoid_x2, swiglu.one_x2)
    grad_gate_x2 = ptx_f32x2("mul.rn", grad_sigmoid_x2, silu_factor_x2)
    grad_gate, grad_up = unpack_f32x2(grad_gate_x2), unpack_f32x2(grad_up_x2)
    words = []
    for j, (word, clamped) in enumerate(((word0, clamped0), (word1, clamped1))):
        # gate <= limit  <=>  min(gate, limit) == gate;  |up| <= limit  <=>  clamp(up) == up
        unclamped_mask = ptx_u32("set.eq.u32.bf16x2", word, clamped)
        words.append(txl.bitwise_and(ptx_u32("cvt.rn.bf16x2.f32", grad_up[j], grad_gate[j]), unclamped_mask))
    return words


def swiglu_fwd_word(in0, in1, bias0, bias1, swiglu, use_bf16_clamp):
    """
    Add the bias words to two gate/up words and return the bf16x2 h word of their two channels.
    """
    biased0 = ptx_u32("add.rn.bf16x2", in0, bias0)
    biased1 = ptx_u32("add.rn.bf16x2", in1, bias1)
    if use_bf16_clamp:
        return swiglu_fwd_pair_bf16(clamp_gate_up_bf16(biased0, swiglu), clamp_gate_up_bf16(biased1, swiglu), swiglu)
    return swiglu_fwd_pair(biased0, biased1, swiglu)


def swiglu_bwd_words(in0, in1, bias0, bias1, grad_h_word, swiglu, use_bf16_clamp):
    """
    Add the bias words to two gate/up words and return their (grad_gate, grad_up) words.
    """
    biased0 = ptx_u32("add.rn.bf16x2", in0, bias0)
    biased1 = ptx_u32("add.rn.bf16x2", in1, bias1)
    if use_bf16_clamp:
        return swiglu_bwd_pair_bf16(biased0, biased1, clamp_gate_up_bf16(biased0, swiglu), clamp_gate_up_bf16(biased1, swiglu), grad_h_word, swiglu)
    return swiglu_bwd_pair(biased0, biased1, grad_h_word, swiglu)


@functools.cache
def compile_tile_kernel(N: int, is_bwd: bool, has_rowwise: bool, has_kmajor: bool, use_bf16_clamp: bool):
    """
    Register-direct tile kernel: C output channels (N forward, 2N backward), 128 x 128 tiles.
    """
    C = 2 * N if is_bwd else N
    assert N % 4 == 0, "vector path needs N % 4 == 0"
    num_col_tiles = (C + BLOCK - 1) // BLOCK
    has_tail = C % BLOCK != 0

    @txl.kernel()
    def kern(gu: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), bias: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), dh: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), x_fp8: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), x_scale: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), x_ch: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), x_ch_scale: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), partial: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), alpha: txl.float32, limit: txl.float32, num_groups: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=(WARPS) * 32))
        cta = txl.cta_id()
        tidx = txl.thread_id()
        lane, warp = txl.truncmod(tidx, txl.int32(32)), txl.truncdiv(tidx, txl.int32(32))
        col_group = txl.truncmod(lane, txl.int32(16))
        row_group = warp * 2 + txl.truncdiv(lane, txl.int32(16))
        row_block, col_tile = txl.truncdiv(cta, txl.int32(num_col_tiles)), txl.truncmod(cta, txl.int32(num_col_tiles))
        col0 = col_tile * BLOCK + col_group * 8
        row0 = row_block * BLOCK

        smem = txl.smem_pool()
        s_colsum = smem.alloc([WARPS * BLOCK], txl.f32)
        s_col_amax = smem.alloc([WARPS * 64], txl.u32)

        swiglu = SwigluConstants(alpha, limit)

        out_words = txl.alloc_local([32], "uint32")
        in_words = txl.alloc_local([64], "uint32")
        grad_words = txl.alloc_local([16], "uint32")
        bias_words = txl.alloc_local([8], "uint32")
        group_idx = txl.local_scalar("int32", init=txl.int32(0))

        def process_tile(is_tail):
            lo_ok, hi_ok = (col0 < C) if is_tail else None, (col0 + 4 < C) if is_tail else None

            def run_guarded(cond, emit):
                if cond is None:
                    emit()
                else:
                    with txl.if_(cond), txl.then_():
                        emit()

            def zero_words(words):
                if is_tail:
                    for word in words:
                        txl.ptx.mov.b32(word, txl.uint32(0))

            def load_rows():
                for i in range(8):
                    row64 = txl.Cast("int64", row0 + row_group + 16 * i)
                    zero_words([in_words[8 * i + k] for k in range(8)] + [grad_words[2 * i], grad_words[2 * i + 1]])
                    if is_bwd:
                        def load_lo(i=i, row64=row64):
                            txl.ptx.ld.global_.nc.v2.b32(in_words[8 * i], in_words[8 * i + 1], gu.ptr_to([row64 * (2 * N) + txl.Cast("int64", col0)]))
                            txl.ptx.ld.global_.nc.b32(grad_words[2 * i], dh.ptr_to([row64 * N + txl.Cast("int64", txl.truncdiv(col0, txl.int32(2)))]))

                        def load_hi(i=i, row64=row64):
                            txl.ptx.ld.global_.nc.v2.b32(in_words[8 * i + 2], in_words[8 * i + 3], gu.ptr_to([row64 * (2 * N) + txl.Cast("int64", col0 + 4)]))
                            txl.ptx.ld.global_.nc.b32(grad_words[2 * i + 1], dh.ptr_to([row64 * N + txl.Cast("int64", txl.truncdiv(col0, txl.int32(2)) + 2)]))
                    else:
                        def load_lo(i=i, row64=row64):
                            txl.ptx.ld.global_.nc.v4.b32(in_words[8 * i], in_words[8 * i + 1], in_words[8 * i + 2], in_words[8 * i + 3], gu.ptr_to([row64 * (2 * N) + txl.Cast("int64", 2 * col0)]))

                        def load_hi(i=i, row64=row64):
                            txl.ptx.ld.global_.nc.v4.b32(in_words[8 * i + 4], in_words[8 * i + 5], in_words[8 * i + 6], in_words[8 * i + 7], gu.ptr_to([row64 * (2 * N) + txl.Cast("int64", 2 * col0 + 8)]))
                    run_guarded(lo_ok, load_lo)
                    run_guarded(hi_ok, load_hi)

            def find_group():
                """
                Group of this row block = #(group ends <= row0), clamped to G-1 (warp ballot).
                """
                group_end = txl.local_scalar("int32")
                group_base = txl.local_scalar("int32", init=txl.int32(0))
                ballot = txl.local_scalar("uint32")
                with txl.while_(group_base < num_groups):
                    txl.assign(group_end, row0 + 1)
                    with txl.if_(group_base + lane < num_groups), txl.then_():
                        txl.ptx.ld.global_.nc.b32(group_end, offs.ptr_to([txl.Cast("int64", group_base + lane)]))
                    txl.ptx.vote_sync.ballot.b32(ballot, txl.ptx.pred(group_end <= row0), txl.uint32(FULL_MASK))
                    txl.assign(group_idx, group_idx + txl.Cast("int32", txl.popcount(ballot)))
                    txl.assign(group_base, group_base + 32)
                with txl.if_(group_idx > num_groups - 1), txl.then_():
                    txl.assign(group_idx, num_groups - 1)

            def load_bias():
                zero_words([bias_words[k] for k in range(8)])
                group64 = txl.Cast("int64", group_idx)
                if is_bwd:
                    run_guarded(lo_ok, lambda: txl.ptx.ld.global_.nc.v2.b32(bias_words[0], bias_words[1], bias.ptr_to([group64 * (2 * N) + txl.Cast("int64", col0)])))
                    run_guarded(hi_ok, lambda: txl.ptx.ld.global_.nc.v2.b32(bias_words[2], bias_words[3], bias.ptr_to([group64 * (2 * N) + txl.Cast("int64", col0 + 4)])))
                else:
                    run_guarded(lo_ok, lambda: txl.ptx.ld.global_.nc.v4.b32(bias_words[0], bias_words[1], bias_words[2], bias_words[3], bias.ptr_to([group64 * (2 * N) + txl.Cast("int64", 2 * col0)])))
                    run_guarded(hi_ok, lambda: txl.ptx.ld.global_.nc.v4.b32(bias_words[4], bias_words[5], bias_words[6], bias_words[7], bias.ptr_to([group64 * (2 * N) + txl.Cast("int64", 2 * col0 + 8)])))

            def compute_rows():
                for i in range(8):
                    if is_bwd:
                        for pair_idx in range(2):
                            word0, word1 = swiglu_bwd_words(in_words[8 * i + 2 * pair_idx], in_words[8 * i + 2 * pair_idx + 1], bias_words[2 * pair_idx], bias_words[2 * pair_idx + 1], grad_words[2 * i + pair_idx], swiglu, use_bf16_clamp)
                            txl.ptx.mov.b32(out_words[4 * i + 2 * pair_idx], word0)
                            txl.ptx.mov.b32(out_words[4 * i + 2 * pair_idx + 1], word1)
                    else:
                        for pair_idx in range(4):
                            word = swiglu_fwd_word(in_words[8 * i + 2 * pair_idx], in_words[8 * i + 2 * pair_idx + 1], bias_words[2 * pair_idx], bias_words[2 * pair_idx + 1], swiglu, use_bf16_clamp)
                            txl.ptx.mov.b32(out_words[4 * i + pair_idx], word)
                    if is_tail:
                        with txl.if_(col0 >= C), txl.then_():
                            txl.ptx.mov.b32(out_words[4 * i + 0], txl.uint32(0))
                            txl.ptx.mov.b32(out_words[4 * i + 1], txl.uint32(0))
                        with txl.if_(col0 + 4 >= C), txl.then_():
                            txl.ptx.mov.b32(out_words[4 * i + 2], txl.uint32(0))
                            txl.ptx.mov.b32(out_words[4 * i + 3], txl.uint32(0))

            def store_row8(dst, row, word0, word1):
                offset = txl.Cast("int64", row) * C + txl.Cast("int64", col0)
                if not is_tail:
                    if C % 8 == 0:
                        txl.ptx.st.global_.v2.b32(dst.ptr_to([offset]), word0, word1)
                    else:
                        txl.ptx.st.global_.b32(dst.ptr_to([offset]), word0)
                        txl.ptx.st.global_.b32(dst.ptr_to([offset + 4]), word1)
                elif C % 8 == 0:
                    with txl.if_(hi_ok):
                        with txl.then_():
                            txl.ptx.st.global_.v2.b32(dst.ptr_to([offset]), word0, word1)
                        with txl.else_():
                            with txl.if_(lo_ok), txl.then_():
                                txl.ptx.st.global_.b32(dst.ptr_to([offset]), word0)
                else:
                    with txl.if_(lo_ok), txl.then_():
                        txl.ptx.st.global_.b32(dst.ptr_to([offset]), word0)
                    with txl.if_(hi_ok), txl.then_():
                        txl.ptx.st.global_.b32(dst.ptr_to([offset + 4]), word1)

            def quantize_rows():
                for row_pair in range(4):
                    amax_first = amax_bf16x2(amax_bf16x2(out_words[8 * row_pair], out_words[8 * row_pair + 1]), amax_bf16x2(out_words[8 * row_pair + 2], out_words[8 * row_pair + 3]))
                    amax_second = amax_bf16x2(amax_bf16x2(out_words[8 * row_pair + 4], out_words[8 * row_pair + 5]), amax_bf16x2(out_words[8 * row_pair + 6], out_words[8 * row_pair + 7]))
                    amax = amax_bf16x2(ptx_u32("prmt.b32", amax_first, amax_second, txl.uint32(0x5410)), ptx_u32("prmt.b32", amax_first, amax_second, txl.uint32(0x7632)))
                    for lane_xor in (1, 2, 4, 8):
                        amax = amax_bf16x2(amax, shfl_xor(amax, lane_xor))
                    e8m0 = e8m0_pair(abs_bf16x2(amax))
                    e8m0_lo, e8m0_hi = txl.bitwise_and(e8m0, txl.uint32(0xFF)), txl.shift_right(e8m0, txl.uint32(8))
                    for half, row_e8m0 in ((0, e8m0_lo), (1, e8m0_hi)):
                        i = 2 * row_pair + half
                        row = row0 + row_group + 16 * i
                        with txl.if_(col_group == 0), txl.then_():
                            txl.ptx.st.global_.f32(x_scale.ptr_to([txl.Cast("int64", row) * num_col_tiles + txl.Cast("int64", col_tile)]), e8m0_to_scale(row_e8m0))
                        rcp = e8m0_to_rcp_bf16(row_e8m0)
                        rcp_x2 = txl.bitwise_or(rcp, txl.shift_left(rcp, txl.uint32(16)))
                        store_row8(x_fp8, row, quantize_fp8x4(out_words[4 * i], out_words[4 * i + 1], rcp_x2, rcp_x2), quantize_fp8x4(out_words[4 * i + 2], out_words[4 * i + 3], rcp_x2, rcp_x2))

            def quantize_cols():
                col_amax = []
                for pair_idx in range(4):
                    amax = out_words[pair_idx]
                    for i in range(1, 8):
                        amax = amax_bf16x2(amax, out_words[4 * i + pair_idx])
                    amax = amax_bf16x2(amax, shfl_xor(amax, 16))
                    col_amax.append(amax)
                with txl.if_(lane < 16), txl.then_():
                    txl.ptx.st.shared.v4.b32(s_col_amax.ptr_to([warp * 64 + col_group * 4]), col_amax[0], col_amax[1], col_amax[2], col_amax[3])
                txl.ptx.bar.sync(txl.uint32(0), txl.uint32(WARPS * 32))
                acc_amax, peer_amax = txl.alloc_local([4], "uint32"), txl.alloc_local([4], "uint32")
                txl.ptx.ld.shared.v4.b32(acc_amax[0], acc_amax[1], acc_amax[2], acc_amax[3], s_col_amax.ptr_to([col_group * 4]))
                for peer_warp in range(1, WARPS):
                    txl.ptx.ld.shared.v4.b32(peer_amax[0], peer_amax[1], peer_amax[2], peer_amax[3], s_col_amax.ptr_to([peer_warp * 64 + col_group * 4]))
                    for pair_idx in range(4):
                        txl.ptx.mov.b32(acc_amax[pair_idx], amax_bf16x2(acc_amax[pair_idx], peer_amax[pair_idx]))
                col_rcp = []
                col_scale = []
                for pair_idx in range(4):
                    e8m0 = e8m0_pair(abs_bf16x2(acc_amax[pair_idx]))
                    e8m0_lo, e8m0_hi = txl.bitwise_and(e8m0, txl.uint32(0xFF)), txl.shift_right(e8m0, txl.uint32(8))
                    col_scale += [e8m0_to_scale(e8m0_lo), e8m0_to_scale(e8m0_hi)]
                    col_rcp.append(txl.local_scalar("uint32", init=txl.bitwise_or(e8m0_to_rcp_bf16(e8m0_lo), txl.shift_left(e8m0_to_rcp_bf16(e8m0_hi), txl.uint32(16)))))
                with txl.if_(row_group == 0), txl.then_():
                    if is_tail:
                        for j in range(8):
                            with txl.if_(col0 + j < C), txl.then_():
                                txl.ptx.st.global_.f32(x_ch_scale.ptr_to([txl.Cast("int64", row_block) * C + txl.Cast("int64", col0 + j)]), col_scale[j])
                    else:
                        for j in range(0, 8, 4):
                            txl.ptx.st.global_.v4.f32(x_ch_scale.ptr_to([txl.Cast("int64", row_block) * C + txl.Cast("int64", col0 + j)]), col_scale[j], col_scale[j + 1], col_scale[j + 2], col_scale[j + 3])
                for i in range(8):
                    row = row0 + row_group + 16 * i
                    store_row8(x_ch, row, quantize_fp8x4(out_words[4 * i], out_words[4 * i + 1], col_rcp[0], col_rcp[1]), quantize_fp8x4(out_words[4 * i + 2], out_words[4 * i + 3], col_rcp[2], col_rcp[3]))

            def store_block_sums():
                """
                Column sums of the tile's grad words (the d_bias partials of this row block).
                """
                acc_colsum = txl.alloc_local([8], "float32")
                for j in range(8):
                    txl.ptx.mov.b32(acc_colsum[j], txl.float32(0.0))
                for i in range(8):
                    for pair_idx in range(4):
                        lo_bits, hi_bits = split_bf16x2(out_words[4 * i + pair_idx])
                        txl.ptx["add.rn.f32.bf16"](acc_colsum[2 * pair_idx], lo_bits, acc_colsum[2 * pair_idx])
                        txl.ptx["add.rn.f32.bf16"](acc_colsum[2 * pair_idx + 1], hi_bits, acc_colsum[2 * pair_idx + 1])
                for j in range(8):
                    txl.ptx.mov.b32(acc_colsum[j], ptx_f32("add.rn.f32", acc_colsum[j], txl.reinterpret(shfl_xor(txl.reinterpret(acc_colsum[j], ty="uint32"), 16), ty="float32")))
                with txl.if_(lane < 16), txl.then_():
                    txl.ptx.st.shared.v4.f32(s_colsum.ptr_to([warp * BLOCK + col_group * 8]), acc_colsum[0], acc_colsum[1], acc_colsum[2], acc_colsum[3])
                    txl.ptx.st.shared.v4.f32(s_colsum.ptr_to([warp * BLOCK + col_group * 8 + 4]), acc_colsum[4], acc_colsum[5], acc_colsum[6], acc_colsum[7])
                txl.ptx.bar.sync(txl.uint32(0), txl.uint32(WARPS * 32))
                peer_colsum = txl.alloc_local([8], "float32")
                with txl.if_(row_group == 0), txl.then_():
                    for peer_warp in range(1, WARPS):
                        txl.ptx.ld.shared.v4.f32(peer_colsum[0], peer_colsum[1], peer_colsum[2], peer_colsum[3], s_colsum.ptr_to([peer_warp * BLOCK + col_group * 8]))
                        txl.ptx.ld.shared.v4.f32(peer_colsum[4], peer_colsum[5], peer_colsum[6], peer_colsum[7], s_colsum.ptr_to([peer_warp * BLOCK + col_group * 8 + 4]))
                        for j in range(8):
                            txl.ptx.mov.b32(acc_colsum[j], ptx_f32("add.rn.f32", acc_colsum[j], peer_colsum[j]))
                    if is_tail:
                        for j in range(8):
                            with txl.if_(col0 + j < C), txl.then_():
                                txl.ptx.st.global_.f32(partial.ptr_to([txl.Cast("int64", row_block) * C + txl.Cast("int64", col0 + j)]), acc_colsum[j])
                    else:
                        for j in range(0, 8, 4):
                            txl.ptx.st.global_.v4.f32(partial.ptr_to([txl.Cast("int64", row_block) * C + txl.Cast("int64", col0 + j)]), acc_colsum[j], acc_colsum[j + 1], acc_colsum[j + 2], acc_colsum[j + 3])

            load_rows()
            find_group()
            load_bias()
            compute_rows()
            if has_rowwise:
                quantize_rows()
            if has_kmajor:
                quantize_cols()
            if is_bwd:
                store_block_sums()

        if has_tail:
            with txl.if_(col_tile == num_col_tiles - 1):
                with txl.then_():
                    process_tile(True)
                with txl.else_():
                    process_tile(False)
        else:
            process_tile(False)

    return compile_kernel(kern)


@functools.cache
def compile_segsum_kernel(C: int):
    """
    d_bias with one thread per column, each group's block partials summed in block order.
    """
    @txl.kernel()
    def segsum(partial: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), num_groups: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=128))
        cta = txl.cta_id()
        tidx = txl.thread_id()
        num_col_blocks = (C + 127) // 128
        group_idx = txl.truncdiv(cta, txl.int32(num_col_blocks))
        col = txl.truncmod(cta, txl.int32(num_col_blocks)) * 128 + tidx
        group_start, group_end = txl.local_scalar("int32", init=txl.int32(0)), txl.local_scalar("int32")
        with txl.if_(group_idx > 0), txl.then_():
            txl.ptx.ld.global_.nc.b32(group_start, offs.ptr_to([txl.Cast("int64", group_idx - 1)]))
        txl.ptx.ld.global_.nc.b32(group_end, offs.ptr_to([txl.Cast("int64", group_idx)]))
        acc_sum = txl.local_scalar("float32", init=txl.float32(0.0))
        block_idx = txl.local_scalar("int32", init=txl.truncdiv(group_start, txl.int32(128)))
        with txl.if_(col < C), txl.then_():
            block_end = txl.truncdiv(group_end, txl.int32(128))
            block_sums = txl.alloc_local([32], "float32")
            with txl.while_(block_idx < block_end):
                for unroll_idx in range(32):
                    txl.ptx.mov.b32(block_sums[unroll_idx], txl.float32(0.0))
                    with txl.if_(block_idx + unroll_idx < block_end), txl.then_():
                        txl.ptx.ld.global_.f32(block_sums[unroll_idx], partial.ptr_to([txl.Cast("int64", block_idx + unroll_idx) * C + txl.Cast("int64", col)]))
                for unroll_idx in range(32):
                    txl.assign(acc_sum, ptx_f32("add.rn.f32", acc_sum, block_sums[unroll_idx]))
                txl.assign(block_idx, block_idx + 32)
            sum_bf16 = txl.local_scalar("uint16")
            txl.ptx.cvt.rn.bf16.f32(sum_bf16, acc_sum)
            txl.ptx.st.global_.b16(out.ptr_to([txl.Cast("int64", group_idx) * C + txl.Cast("int64", col)]), sum_bf16)

    return compile_kernel(segsum)


SEG_UNROLL = 12  # block partial loads in flight per thread (one 16B load each)


@functools.cache
def compile_segsum_split_kernel(C: int):
    """
    d_bias[g, col_block * 128 .. +128] = sum of group g's 128-row block partials (row-major
    [M / 128, C], C % 4 == 0). 8 warps split the group's blocks into 8 contiguous runs; lane = 4
    columns, so each warp load is one contiguous 512B row segment. Each run is summed in block
    order, then the 8 run sums in warp order (a fixed order: deterministic).
    """
    assert C % 4 == 0

    @txl.kernel()
    def segsum4(partial: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), num_groups: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=256))
        cta = txl.cta_id()
        tidx = txl.thread_id()
        lane, warp = txl.truncmod(tidx, txl.int32(32)), txl.truncdiv(tidx, txl.int32(32))
        num_col_blocks = (C + 127) // 128
        group_idx = txl.truncdiv(cta, txl.int32(num_col_blocks))
        col = txl.truncmod(cta, txl.int32(num_col_blocks)) * 128 + lane * 4
        smem = txl.smem_pool()
        s_run_sums = smem.alloc([8 * 128], txl.f32, align=16)
        group_start, group_end = txl.local_scalar("int32", init=txl.int32(0)), txl.local_scalar("int32")
        with txl.if_(group_idx > 0), txl.then_():
            txl.ptx.ld.global_.nc.b32(group_start, offs.ptr_to([txl.Cast("int64", group_idx - 1)]))
        txl.ptx.ld.global_.nc.b32(group_end, offs.ptr_to([txl.Cast("int64", group_idx)]))
        block_start = txl.truncdiv(group_start, txl.int32(128))
        num_blocks = txl.truncdiv(group_end, txl.int32(128)) - block_start
        run_start = txl.local_scalar("int32", init=block_start + txl.truncdiv(num_blocks * warp, txl.int32(8)))
        run_end = txl.local_scalar("int32", init=block_start + txl.truncdiv(num_blocks * (warp + 1), txl.int32(8)))
        acc_sum = txl.alloc_local([4], "float32")
        block_sums = txl.alloc_local([4 * SEG_UNROLL], "float32")
        for j in range(4):
            txl.ptx.mov.b32(acc_sum[j], txl.float32(0.0))
        with txl.if_(col < C), txl.then_():
            with txl.while_(run_start < run_end):
                for unroll_idx in range(SEG_UNROLL):
                    for j in range(4):
                        txl.ptx.mov.b32(block_sums[4 * unroll_idx + j], txl.float32(0.0))
                    with txl.if_(run_start + unroll_idx < run_end), txl.then_():
                        txl.ptx.ld.global_.nc.v4.f32(block_sums[4 * unroll_idx], block_sums[4 * unroll_idx + 1], block_sums[4 * unroll_idx + 2], block_sums[4 * unroll_idx + 3], partial.ptr_to([txl.Cast("int64", run_start + unroll_idx) * C + txl.Cast("int64", col)]))
                for unroll_idx in range(SEG_UNROLL):
                    for j in range(4):
                        txl.assign(acc_sum[j], ptx_f32("add.rn.f32", acc_sum[j], block_sums[4 * unroll_idx + j]))
                txl.assign(run_start, run_start + SEG_UNROLL)
        txl.ptx.st.shared.v4.f32(s_run_sums.ptr_to([warp * 128 + lane * 4]), acc_sum[0], acc_sum[1], acc_sum[2], acc_sum[3])
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(256))
        with txl.if_(tidx < 128), txl.then_():
            out_col = txl.truncmod(cta, txl.int32(num_col_blocks)) * 128 + tidx
            with txl.if_(out_col < C), txl.then_():
                col_sum, run_sum = txl.local_scalar("float32"), txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(col_sum, s_run_sums.ptr_to([tidx]))
                for run_idx in range(1, 8):
                    txl.ptx.ld.shared.f32(run_sum, s_run_sums.ptr_to([run_idx * 128 + tidx]))
                    txl.assign(col_sum, ptx_f32("add.rn.f32", col_sum, run_sum))
                sum_bf16 = txl.local_scalar("uint16")
                txl.ptx.cvt.rn.bf16.f32(sum_bf16, col_sum)
                txl.ptx.st.global_.b16(out.ptr_to([txl.Cast("int64", group_idx) * C + txl.Cast("int64", out_col)]), sum_bf16)

    return compile_kernel(segsum4)


@functools.cache
def compile_segsum_warp_kernel(C: int):
    """
    Small groups: one warp per (group, 128 columns), lane = 4 columns (16B loads, one 512B row
    segment per warp load), the group's blocks summed in block order. Many tiny CTAs fit one wave.
    """
    assert C % 4 == 0

    @txl.kernel()
    def segsum1(partial: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1), num_groups: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=32))
        cta = txl.cta_id()
        lane = txl.thread_id()
        num_col_blocks = (C + 127) // 128
        group_idx = txl.truncdiv(cta, txl.int32(num_col_blocks))
        col = txl.truncmod(cta, txl.int32(num_col_blocks)) * 128 + lane * 4
        group_start, group_end = txl.local_scalar("int32", init=txl.int32(0)), txl.local_scalar("int32")
        with txl.if_(group_idx > 0), txl.then_():
            txl.ptx.ld.global_.nc.b32(group_start, offs.ptr_to([txl.Cast("int64", group_idx - 1)]))
        txl.ptx.ld.global_.nc.b32(group_end, offs.ptr_to([txl.Cast("int64", group_idx)]))
        block_idx = txl.local_scalar("int32", init=txl.truncdiv(group_start, txl.int32(128)))
        block_end = txl.truncdiv(group_end, txl.int32(128))
        acc_sum = txl.alloc_local([4], "float32")
        block_sums = txl.alloc_local([4 * SEG_UNROLL], "float32")
        for j in range(4):
            txl.ptx.mov.b32(acc_sum[j], txl.float32(0.0))
        with txl.if_(col < C), txl.then_():
            with txl.while_(block_idx < block_end):
                for unroll_idx in range(SEG_UNROLL):
                    for j in range(4):
                        txl.ptx.mov.b32(block_sums[4 * unroll_idx + j], txl.float32(0.0))
                    with txl.if_(block_idx + unroll_idx < block_end), txl.then_():
                        txl.ptx.ld.global_.nc.v4.f32(block_sums[4 * unroll_idx], block_sums[4 * unroll_idx + 1], block_sums[4 * unroll_idx + 2], block_sums[4 * unroll_idx + 3], partial.ptr_to([txl.Cast("int64", block_idx + unroll_idx) * C + txl.Cast("int64", col)]))
                for unroll_idx in range(SEG_UNROLL):
                    for j in range(4):
                        txl.assign(acc_sum[j], ptx_f32("add.rn.f32", acc_sum[j], block_sums[4 * unroll_idx + j]))
                txl.assign(block_idx, block_idx + SEG_UNROLL)
            sums_bf16 = txl.alloc_local([4], "uint16")
            for j in range(4):
                txl.ptx.cvt.rn.bf16.f32(sums_bf16[j], acc_sum[j])
            txl.ptx.st.global_.v4.b16(out.ptr_to([txl.Cast("int64", group_idx) * C + txl.Cast("int64", col)]), sums_bf16[0], sums_bf16[1], sums_bf16[2], sums_bf16[3])

    return compile_kernel(segsum1)


TMA_2D = "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
TMA_2D_HINT = TMA_2D + ".L2::cache_hint"
TMA_PREFETCH_2D = "cp.async.bulk.prefetch.tensor.2d.L2.global.tile"
POLICY_EVICT_FIRST = 0x12F0000000000000
POLICY_EVICT_LAST = 0x14F0000000000000
PIPE_WARPS = 16
MAX_SMEM_GROUPS = 512


@functools.cache
def count_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def fold_dbias(partial, out, s_offs, s_run_sums, num_groups, seg_mode, part_idx, num_parts, tidx, lane, warp, C):
    """
    d_bias[g, col_block * 128 .. +128] from the fp32 block partials, by participant part_idx of
    num_parts (8 warps). seg_mode 1 (large groups): one CTA per (group, 128 columns), the 8 warps
    split the group's blocks into 8 contiguous runs summed in block order, the run sums added in
    warp order (as compile_segsum_split_kernel). seg_mode 0: one warp per (group, 128 columns),
    blocks in block order (as compile_segsum_warp_kernel / compile_segsum_kernel). Lane = 4
    columns (16B loads).
    """
    num_col_blocks = (C + 127) // 128
    num_chunks = num_groups * num_col_blocks
    acc_sum = txl.alloc_local([4], "float32")
    block_sums = txl.alloc_local([4 * SEG_UNROLL], "float32")

    def sum_blocks(block_idx, block_end, col):
        for j in range(4):
            txl.ptx.mov.b32(acc_sum[j], txl.float32(0.0))
        with txl.if_(col < C), txl.then_():
            with txl.while_(block_idx < block_end):
                for unroll_idx in range(SEG_UNROLL):
                    for j in range(4):
                        txl.ptx.mov.b32(block_sums[4 * unroll_idx + j], txl.float32(0.0))
                    with txl.if_(block_idx + unroll_idx < block_end), txl.then_():
                        txl.ptx["ld.relaxed.gpu.global.v4.f32"](block_sums[4 * unroll_idx], block_sums[4 * unroll_idx + 1], block_sums[4 * unroll_idx + 2], block_sums[4 * unroll_idx + 3], partial.ptr_to([txl.Cast("int64", block_idx + unroll_idx) * C + txl.Cast("int64", col)]))
                for unroll_idx in range(SEG_UNROLL):
                    for j in range(4):
                        txl.assign(acc_sum[j], ptx_f32("add.rn.f32", acc_sum[j], block_sums[4 * unroll_idx + j]))
                txl.assign(block_idx, block_idx + SEG_UNROLL)

    def group_blocks(group_idx):
        """
        (first block, number of blocks) of group group_idx.
        """
        group_start, group_end = txl.local_scalar("int32", init=txl.int32(0)), txl.local_scalar("int32")
        with txl.if_(group_idx > 0), txl.then_():
            txl.ptx.ld.shared.b32(group_start, s_offs.ptr_to([group_idx - 1]))
        txl.ptx.ld.shared.b32(group_end, s_offs.ptr_to([group_idx]))
        block_start = txl.truncdiv(group_start, txl.int32(128))
        return block_start, txl.truncdiv(group_end, txl.int32(128)) - block_start

    with txl.if_(seg_mode == 1):
        with txl.then_():
            chunk_idx = txl.local_scalar("int32", init=part_idx)
            is_first_chunk = txl.local_scalar("int32", init=txl.int32(1))
            with txl.while_(chunk_idx < num_chunks):
                group_idx = txl.local_scalar("int32", init=txl.truncdiv(chunk_idx, txl.int32(num_col_blocks)))
                col_block = txl.local_scalar("int32", init=chunk_idx - group_idx * num_col_blocks)
                block_start, num_blocks = group_blocks(group_idx)
                run_start = txl.local_scalar("int32", init=block_start + txl.truncdiv(num_blocks * warp, txl.int32(8)))
                run_end = txl.local_scalar("int32", init=block_start + txl.truncdiv(num_blocks * (warp + 1), txl.int32(8)))
                sum_blocks(run_start, run_end, col_block * 128 + lane * 4)
                # The first chunk has no previous shared payload to protect.
                with txl.if_(is_first_chunk == 0), txl.then_():
                    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(256))
                txl.ptx.st.shared.v4.f32(s_run_sums.ptr_to([warp * 128 + lane * 4]), acc_sum[0], acc_sum[1], acc_sum[2], acc_sum[3])
                txl.ptx.bar.sync(txl.uint32(0), txl.uint32(256))
                with txl.if_(tidx < 128), txl.then_():
                    out_col = col_block * 128 + tidx
                    with txl.if_(out_col < C), txl.then_():
                        col_sum, run_sum = txl.local_scalar("float32"), txl.local_scalar("float32")
                        txl.ptx.ld.shared.f32(col_sum, s_run_sums.ptr_to([tidx]))
                        for run_idx in range(1, 8):
                            txl.ptx.ld.shared.f32(run_sum, s_run_sums.ptr_to([run_idx * 128 + tidx]))
                            txl.assign(col_sum, ptx_f32("add.rn.f32", col_sum, run_sum))
                        sum_bf16 = txl.local_scalar("uint16")
                        txl.ptx.cvt.rn.bf16.f32(sum_bf16, col_sum)
                        txl.ptx.st.global_.b16(out.ptr_to([txl.Cast("int64", group_idx) * C + txl.Cast("int64", out_col)]), sum_bf16)
                txl.assign(is_first_chunk, txl.int32(0))
                txl.assign(chunk_idx, chunk_idx + num_parts)
        with txl.else_():
            chunk_idx = txl.local_scalar("int32", init=part_idx * 8 + warp)
            with txl.while_(chunk_idx < num_chunks):
                group_idx = txl.local_scalar("int32", init=txl.truncdiv(chunk_idx, txl.int32(num_col_blocks)))
                col_block = txl.local_scalar("int32", init=chunk_idx - group_idx * num_col_blocks)
                block_start, num_blocks = group_blocks(group_idx)
                block_idx = txl.local_scalar("int32", init=block_start)
                col = col_block * 128 + lane * 4
                sum_blocks(block_idx, block_start + num_blocks, col)
                with txl.if_(col < C), txl.then_():
                    sums_bf16 = txl.alloc_local([4], "uint16")
                    for j in range(4):
                        txl.ptx.cvt.rn.bf16.f32(sums_bf16[j], acc_sum[j])
                    txl.ptx.st.global_.v4.b16(out.ptr_to([txl.Cast("int64", group_idx) * C + txl.Cast("int64", col)]), sums_bf16[0], sums_bf16[1], sums_bf16[2], sums_bf16[3])
                txl.assign(chunk_idx, chunk_idx + num_parts * 8)


@functools.cache
def compile_pipe_kernel(N: int, is_bwd: bool, has_rowwise: bool, has_kmajor: bool, use_bf16_clamp: bool, stages: int, use_bwd_split: bool = False):
    """
    Persistent bulk-copy pipelined tile kernel (needs N % 8 == 0, G <= MAX_SMEM_GROUPS).

    The CTA's consumer groups of 8 warps (two forward, one backward) take alternate tiles; thread
    (row_group = 0..15, col_group = 0..15) of a group owns rows row_group + 16 i (i < 8) of the
    tile and 8 output channels: forward h channels 4 col_group .. +3 and 64 + 4 col_group .. +3
    (16B smem chunks at 16 col_group and 256 + 16 col_group of a 512B gate_up row: conflict-free),
    backward channels 8 col_group .. +7. Warp 0 of each group refills a stage with bulk tensor
    copies (the tile's rows, plus the group's bias row) once the group has read it (the group
    barrier after the smem reads).
    """
    C = 2 * N if is_bwd else N
    assert N % 8 == 0
    num_col_tiles = (C + BLOCK - 1) // BLOCK
    GU_ROW = 512 if not is_bwd else 256  # smem bytes per row of gate_up per stage
    DH_ROW = 128 if is_bwd else 0
    BIAS_ROW = GU_ROW
    GU_STAGE = BLOCK * GU_ROW + BIAS_ROW
    DH_STAGE = BLOCK * DH_ROW
    STAGE = GU_STAGE + DH_STAGE
    ROWS_PER_THREAD = 8  # rows row_group + 16 i
    has_tail = C % BLOCK != 0
    # Backward uses two independent 8-warp CTAs per SM. Each CTA has one consumer group and two
    # ring stages; together the pair has the same 16-warp / four-stage capacity as one 16-warp CTA.
    CTA_WARPS = 8 if is_bwd else PIPE_WARPS
    NUM_CONSUMERS = 1 if is_bwd else 2
    CONSUMER_WARPS = CTA_WARPS // NUM_CONSUMERS
    NUM_SLOTS = 2 * NUM_CONSUMERS
    USE_TAIL_SPLIT = (not is_bwd) and has_kmajor and not has_rowwise
    USE_DYNAMIC = is_bwd  # dynamic tile schedule (backward)
    # split arrival: a stage's rows land as two 64-row halves with their own full barriers
    USE_SPLIT_ARRIVAL = use_bwd_split if is_bwd else has_kmajor
    BOX_ROWS = BLOCK // 2 if USE_SPLIT_ARRIVAL else BLOCK

    GU_BOX_COLS = GU_ROW // 2  # bf16 columns per tile row

    def host_prelude(params):
        num_rows = params["num_rows"]
        gu_map, dh_map, bias_map = txl.stack_alloca("tensormap", 1), txl.stack_alloca("tensormap", 1), txl.stack_alloca("tensormap", 1)

        def encode_tensor_map(desc, data, inner, outer, outer_stride, box_inner, box_outer, swizzle=0):
            # 2-D bf16, optional 128B XOR swizzle, 128B L2 promotion, zero OOB fill
            txl.call_packed("runtime.cuTensorMapEncodeTiled", desc, "bfloat16", 2, data, inner, outer, outer_stride, box_inner, box_outer, 1, 1, 0, swizzle, 2, 0)

        encode_tensor_map(gu_map, params["gu"].data, 2 * N, num_rows, 4 * N, GU_BOX_COLS, BOX_ROWS)
        encode_tensor_map(bias_map, params["bias"].data, 2 * N, params["num_groups"], 4 * N, GU_BOX_COLS, 1)
        if is_bwd:
            encode_tensor_map(dh_map, params["dh"].data, N, num_rows, 2 * N, DH_ROW // 2, BOX_ROWS, 3)
        else:
            encode_tensor_map(dh_map, params["gu"].data, 2 * N, num_rows, 4 * N, GU_BOX_COLS, BLOCK)
        return gu_map, dh_map, bias_map

    @txl.kernel()
    def kern(
        gu: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1),
        bias: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1),
        offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1),
        dh: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1),
        x_fp8: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1),
        x_scale: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1),
        x_ch: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1),
        x_ch_scale: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1),
        partial: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1),
        sched: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1),
        dbias: txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1),
        alpha: txl.float32,
        limit: txl.float32,
        num_groups: txl.int32,
        num_tiles: txl.int32,
        num_ctas: txl.int32,
        num_rows: txl.int32,
        seg_mode: txl.int32,
    ):
        host_maps = host_prelude({"gu": gu, "bias": bias, "dh": dh, "num_groups": num_groups, "num_rows": num_rows})
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=CTA_WARPS * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=2 if is_bwd else 1))
        gu_map, dh_map, bias_map = host_maps
        cta = txl.cta_id()
        tidx = txl.thread_id()
        lane, warp = txl.truncmod(tidx, txl.int32(32)), txl.truncdiv(tidx, txl.int32(32))
        col_group = txl.truncmod(lane, txl.int32(16))
        consumer_idx = txl.truncdiv(warp, txl.int32(CONSUMER_WARPS))
        consumer_warp = txl.truncmod(warp, txl.int32(CONSUMER_WARPS))
        row_group = consumer_warp * 2 + txl.truncdiv(lane, txl.int32(16))
        consumer_bar = 1 + consumer_idx * 3

        smem = txl.smem_pool()
        # gate_up+bias remain row-major. dh uses TMA's 128B XOR swizzle so the
        # two half-warps' 8B vector loads do not alias the same bank pattern.
        s_ring = smem.alloc([stages * GU_STAGE], txl.u8, align=1024)
        s_dh_ring = smem.alloc((stages, BLOCK, DH_ROW // 2), "bfloat16", swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM) if is_bwd else None
        s_offs = smem.alloc([MAX_SMEM_GROUPS], txl.i32, align=16)
        # column-reduction buffers per (group, tile parity): a tile's buffers are rewritten two
        # tiles later, after barriers that order the earlier reads, so no end-of-tile barrier
        s_col_amax = smem.alloc([NUM_SLOTS * CONSUMER_WARPS * 64], txl.u32, align=16)
        s_colsum = smem.alloc([NUM_SLOTS * CONSUMER_WARPS * BLOCK if is_bwd else 4], txl.f32, align=16)
        s_col_rcp = smem.alloc([NUM_SLOTS * 64], txl.u32, align=16)
        full_bar = txl.MBarrier(smem, 2 * stages if USE_SPLIT_ARRIVAL else stages)
        full_bar.init(1)
        # empty_bar[s] completes when the group that consumed the stage's tile has read it
        empty_bar = txl.MBarrier(smem, stages)
        empty_bar.init(1)
        if USE_DYNAMIC:
            # tile id of each stage's current load (-1 = end of the stream), written by the issuer
            # before its arrive on full_bar[stage]
            s_tile_ids = smem.alloc([stages], txl.i32, align=16)
            s_role = smem.alloc([2], txl.i32, align=8)
            # started-CTA count: a CTA may wait for the others at its exit only once all have started
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx.red.global_.add.u32(sched.ptr_to([2]), txl.uint32(1))

        def copy_group_offsets():
            """
            Copy the group end offsets into shared memory, visible to the async proxy.
            """
            with txl.serial(txl.truncdiv(num_groups + (CTA_WARPS * 32 - 1), txl.int32(CTA_WARPS * 32))) as step_idx:
                group_idx = step_idx * (CTA_WARPS * 32) + tidx
                with txl.if_(group_idx < num_groups), txl.then_():
                    group_end = txl.local_scalar("int32")
                    txl.ptx.ld.global_.nc.b32(group_end, offs.ptr_to([txl.Cast("int64", group_idx)]))
                    txl.ptx.st.shared.b32(s_offs.ptr_to([group_idx]), group_end)
            txl.ptx.fence.proxy.async_.shared__cta()
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(CTA_WARPS * 32))

        copy_group_offsets()

        # per-warp monotonic cursor: an issuing warp's tiles have non-decreasing row0, so the
        # count of group ends <= row0 never decreases; scan on from the previous count
        group_cursor = txl.local_scalar("int32", init=txl.int32(0))

        def find_group(row0):
            """
            #(group ends <= row0) clamped to G-1, by one warp (all lanes return it).
            """
            group_idx = txl.local_scalar("int32", init=group_cursor)
            group_base = txl.local_scalar("int32", init=group_cursor)
            is_scanning = txl.local_scalar("int32", init=txl.int32(1))
            group_end = txl.local_scalar("int32")
            ballot = txl.local_scalar("uint32")
            with txl.while_(is_scanning == 1):
                txl.assign(group_end, row0 + 1)
                with txl.if_(group_base + lane < num_groups), txl.then_():
                    txl.ptx.ld.shared.b32(group_end, s_offs.ptr_to([group_base + lane]))
                txl.ptx.vote_sync.ballot.b32(ballot, txl.ptx.pred(group_end <= row0), txl.uint32(FULL_MASK))
                num_ended = txl.local_scalar("int32", init=txl.Cast("int32", txl.popcount(ballot)))
                txl.assign(group_idx, group_idx + num_ended)
                txl.assign(group_base, group_base + 32)
                txl.assign(is_scanning, txl.int32(0))
                with txl.if_(txl.And(num_ended == 32, group_base < num_groups)), txl.then_():
                    txl.assign(is_scanning, txl.int32(1))
            txl.assign(group_cursor, group_idx)
            with txl.if_(group_idx > num_groups - 1), txl.then_():
                txl.assign(group_idx, num_groups - 1)
            return group_idx

        # Tail split (forward, one quantization): the last partial wave's r tiles are split in
        # halves between 2r CTAs (rows for rowwise, columns for kmajor) when 2r <= num_ctas, so the
        # longest CTA does F + 1/2 tiles instead of F + 1. Both CTAs load the whole tile.
        num_full_waves = txl.local_scalar("int32", init=txl.truncdiv(num_tiles, num_ctas))
        num_tail_tiles = txl.local_scalar("int32", init=num_tiles - num_full_waves * num_ctas)
        if USE_TAIL_SPLIT:
            is_tail_split = txl.local_scalar("int32", init=txl.Select(txl.And(num_tail_tiles > 0, 2 * num_tail_tiles <= num_ctas), txl.int32(1), txl.int32(0)))

        def locate_tile(local_idx):
            """
            (tile index, part) of local tile local_idx: part 0 = whole, 1 / 2 = first / second half.
            """
            if not USE_TAIL_SPLIT:
                return cta + local_idx * num_ctas, None
            is_tail = txl.And(is_tail_split == 1, local_idx == num_full_waves)
            tile_idx = txl.Select(is_tail, num_full_waves * num_ctas + txl.truncdiv(cta, txl.int32(2)), cta + local_idx * num_ctas)
            part = txl.Select(is_tail, 1 + txl.truncmod(cta, txl.int32(2)), txl.int32(0))
            return tile_idx, part

        # dynamic schedule: lane 0 of warp 0 holds the next dynamic tile id, fetched one refill ahead
        next_tile = txl.local_scalar("int32", init=txl.int32(0))

        def fetch_next_tile():
            ticket = txl.local_scalar("uint32")
            txl.ptx.atom.global_.add.u32(ticket, sched.ptr_to([0]), txl.uint32(1))
            txl.assign(next_tile, 2 * num_ctas + txl.Cast("int32", ticket))

        def issue_tile(local_idx, is_first=False):
            """
            Warp 0: bulk-load this CTA's local_idx-th tile into stage local_idx % stages.
            """
            if USE_DYNAMIC:
                if is_first:
                    tile_idx = cta + local_idx * num_ctas
                else:
                    tile_idx = txl.local_scalar("int32")
                    txl.ptx.shfl_sync.idx.b32(tile_idx, next_tile, txl.uint32(0), txl.uint32(31), txl.uint32(FULL_MASK))
                    with txl.if_(txl.And(lane == 0, tile_idx < num_tiles)), txl.then_():
                        fetch_next_tile()
                with txl.if_(tile_idx >= num_tiles), txl.then_():
                    with txl.if_(lane == 0), txl.then_():
                        stage = txl.truncmod(local_idx, txl.int32(stages))
                        txl.ptx.st.shared.b32(s_tile_ids.ptr_to([stage]), txl.int32(-1))
                        txl.ptx.mbarrier.arrive.shared.b64(full_bar.buf.ptr_to([stage]), txl.uint32(1))
            else:
                tile_idx, _ = locate_tile(local_idx)
            with txl.if_(tile_idx < num_tiles), txl.then_():
                row_block, col_tile = txl.truncdiv(tile_idx, txl.int32(num_col_tiles)), txl.truncmod(tile_idx, txl.int32(num_col_tiles))
                stage = txl.truncmod(local_idx, txl.int32(stages))
                stage_base = stage * GU_STAGE
                dh_stage = s_dh_ring[stage] if is_bwd else None
                row0 = row_block * BLOCK
                group_idx = find_group(row0)
                gu_col = col_tile * (BLOCK * 2 if not is_bwd else BLOCK)  # first gate_up column of the tile
                with txl.if_(lane == 0), txl.then_():
                    full_addr = full_bar.buf.ptr_to([stage])
                    if USE_DYNAMIC:
                        txl.ptx.st.shared.b32(s_tile_ids.ptr_to([stage]), tile_idx)
                    # the group's generic-proxy reads of this stage precede the async refill
                    txl.ptx.fence.proxy.async_.shared__cta()
                    if not USE_SPLIT_ARRIVAL:
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full_addr, txl.uint32(STAGE))
                        if is_bwd:
                            txl.ptx[TMA_2D_HINT](s_ring.ptr_to([stage_base]), txl.address_of(gu_map), txl.Cast("int32", gu_col), txl.Cast("int32", row0), full_addr, txl.uint64(POLICY_EVICT_LAST))
                        else:
                            txl.ptx[TMA_2D_HINT](s_ring.ptr_to([stage_base]), txl.address_of(gu_map), txl.Cast("int32", gu_col), txl.Cast("int32", row0), full_addr, txl.uint64(POLICY_EVICT_FIRST))
                        if is_bwd:
                            txl.ptx[TMA_2D_HINT](dh_stage.ptr_to(0, 0), txl.address_of(dh_map), txl.Cast("int32", col_tile * (BLOCK // 2)), txl.Cast("int32", row0), full_addr, txl.uint64(POLICY_EVICT_LAST))
                        txl.ptx[TMA_2D](s_ring.ptr_to([stage_base + BLOCK * GU_ROW]), txl.address_of(bias_map), txl.Cast("int32", gu_col), txl.Cast("int32", group_idx), full_addr)
                    else:
                        full_bottom_addr = full_bar.buf.ptr_to([stages + stage])
                        HALF_ROWS = BLOCK // 2
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full_addr, txl.uint32(HALF_ROWS * (GU_ROW + DH_ROW) + BIAS_ROW))
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full_bottom_addr, txl.uint32(HALF_ROWS * (GU_ROW + DH_ROW)))
                        policy = txl.uint64(POLICY_EVICT_LAST if is_bwd else POLICY_EVICT_FIRST)
                        txl.ptx[TMA_2D_HINT](s_ring.ptr_to([stage_base]), txl.address_of(gu_map), txl.Cast("int32", gu_col), txl.Cast("int32", row0), full_addr, policy)
                        if is_bwd:
                            txl.ptx[TMA_2D_HINT](dh_stage.ptr_to(0, 0), txl.address_of(dh_map), txl.Cast("int32", col_tile * (BLOCK // 2)), txl.Cast("int32", row0), full_addr, txl.uint64(POLICY_EVICT_LAST))
                        txl.ptx[TMA_2D](s_ring.ptr_to([stage_base + BLOCK * GU_ROW]), txl.address_of(bias_map), txl.Cast("int32", gu_col), txl.Cast("int32", group_idx), full_addr)
                        txl.ptx[TMA_2D_HINT](s_ring.ptr_to([stage_base + HALF_ROWS * GU_ROW]), txl.address_of(gu_map), txl.Cast("int32", gu_col), txl.Cast("int32", row0 + HALF_ROWS), full_bottom_addr, policy)
                        if is_bwd:
                            txl.ptx[TMA_2D_HINT](dh_stage.ptr_to(HALF_ROWS, 0), txl.address_of(dh_map), txl.Cast("int32", col_tile * (BLOCK // 2)), txl.Cast("int32", row0 + HALF_ROWS), full_bottom_addr, txl.uint64(POLICY_EVICT_LAST))
            if has_rowwise and not has_kmajor and not is_bwd and not (is_first and local_idx < stages - 1):
                next_tile_idx, _ = locate_tile(local_idx + 1)
                with txl.if_(txl.And(lane == 0, next_tile_idx < num_tiles)), txl.then_():
                    next_row_block = txl.truncdiv(next_tile_idx, txl.int32(num_col_tiles))
                    next_col_tile = next_tile_idx - next_row_block * num_col_tiles
                    txl.ptx[TMA_PREFETCH_2D](txl.address_of(gu_map), txl.Cast("int32", next_col_tile * (BLOCK * 2)), txl.Cast("int32", next_row_block * BLOCK))

        with txl.if_(warp == 0), txl.then_():
            for local_idx in range(stages):
                issue_tile(local_idx, is_first=True)
            if USE_DYNAMIC:
                with txl.if_(lane == 0), txl.then_():
                    fetch_next_tile()

        swiglu = SwigluConstants(alpha, limit)

        in_words = txl.alloc_local([8], "uint32")
        grad_words = txl.alloc_local([2], "uint32")
        bias_words = txl.alloc_local([8], "uint32")
        out_words = txl.alloc_local([4 * ROWS_PER_THREAD], "uint32")

        num_my_tiles = txl.truncdiv(num_tiles - cta + num_ctas - 1, num_ctas)
        if USE_TAIL_SPLIT:
            num_my_tiles = txl.Select(is_tail_split == 1, num_full_waves + txl.Select(cta < 2 * num_tail_tiles, txl.int32(1), txl.int32(0)), num_my_tiles)
        if USE_DYNAMIC:
            step_counter = txl.local_scalar("int32", init=txl.int32(0))
            is_running = txl.local_scalar("int32", init=txl.int32(1))
            tile_loop = txl.while_(is_running != 0)
        else:
            tile_loop = txl.serial(num_my_tiles if is_bwd else txl.truncdiv(num_my_tiles - consumer_idx + 1, txl.int32(2)))
        with tile_loop as loop_idx:
            step = step_counter if USE_DYNAMIC else loop_idx  # this group's tile count so far
            local_idx = step if is_bwd else consumer_idx + 2 * step
            slot_idx = txl.truncmod(step, txl.int32(2)) if is_bwd else consumer_idx * 2 + txl.truncmod(step, txl.int32(2))  # (group, tile parity)
            amax_slot_base, colsum_slot_base = slot_idx * (CONSUMER_WARPS * 64), slot_idx * (CONSUMER_WARPS * BLOCK)
            first_tile_idx, first_part = (None, None) if USE_DYNAMIC else locate_tile(local_idx)
            tile_idx = txl.local_scalar("int32") if USE_DYNAMIC else txl.local_scalar("int32", init=first_tile_idx)
            if USE_TAIL_SPLIT:
                part = txl.local_scalar("int32", init=first_part)
                owns_first = part != 2  # rows 0..3 (rowwise) / channels col_first (kmajor) are this CTA's
                owns_second = part != 1
            if is_bwd:
                # left as inline expressions: backward measured faster this way
                row_block, col_tile = txl.truncdiv(tile_idx, txl.int32(num_col_tiles)), txl.truncmod(tile_idx, txl.int32(num_col_tiles))
                row0 = row_block * BLOCK
                out_row_base = txl.Cast("int64", row0 + row_group) * C
                row_scale_base = txl.Cast("int64", col_tile) * num_rows + txl.Cast("int64", row0 + row_group)
            else:
                # per-tile indices computed once in int32 (inlined, every int64 use re-expanded
                # them into a 64-bit division)
                row_block = txl.local_scalar("int32", init=txl.truncdiv(tile_idx, txl.int32(num_col_tiles)))
                col_tile = txl.local_scalar("int32", init=tile_idx - row_block * num_col_tiles)
                row0 = txl.local_scalar("int32", init=row_block * BLOCK)
                # int64 bases of this thread's first output row and of its tile's row-scale run
                out_row_base = txl.local_scalar("int64", init=txl.Cast("int64", row0 + row_group) * C)
                row_scale_base = txl.local_scalar("int64", init=txl.Cast("int64", col_tile) * num_rows + txl.Cast("int64", row0 + row_group))
            stage = txl.truncmod(local_idx, txl.int32(stages))
            stage_base = stage * GU_STAGE
            dh_stage = s_dh_ring[stage] if is_bwd else None
            # Groups alternate tiles, so with an odd stage count the stage's previous tile
            # belongs to the other group; until that tile was consumed the full barrier's
            # parity could alias its still-pending phase. Its release cannot alias.
            with txl.if_(local_idx >= stages), txl.then_():
                empty_bar.wait(stage, txl.truncmod(txl.truncdiv(local_idx, txl.int32(stages)) + 1, txl.int32(2)))
            full_bar.wait(stage, txl.truncmod(txl.truncdiv(local_idx, txl.int32(stages)), txl.int32(2)))
            if USE_DYNAMIC:
                txl.ptx.ld.shared.b32(tile_idx, s_tile_ids.ptr_to([stage]))
            with guard_tile(tile_idx >= 0 if USE_DYNAMIC else None, is_running if USE_DYNAMIC else None):
                # validity of this thread's two 4-channel granules
                if is_bwd:
                    col_first, col_second = col_tile * BLOCK + col_group * 8, col_tile * BLOCK + col_group * 8 + 4
                else:
                    col_first, col_second = col_tile * BLOCK + col_group * 4, col_tile * BLOCK + 64 + col_group * 4
                first_ok, second_ok = col_first < C, col_second < C
                if USE_TAIL_SPLIT and has_kmajor:
                    first_ok, second_ok = txl.And(first_ok, owns_first), txl.And(second_ok, owns_second)

                def compute_rows():
                    """
                    Read the bias row and the thread's rows from the stage and compute out_words;
                    backward also returns the thread's per-channel sums of them.
                    """
                    # bias row (same for every row of the tile)
                    bias_base = stage_base + BLOCK * GU_ROW
                    txl.ptx.ld.shared.v4.b32(bias_words[0], bias_words[1], bias_words[2], bias_words[3], s_ring.ptr_to([bias_base + col_group * 16]))
                    if not is_bwd:
                        txl.ptx.ld.shared.v4.b32(bias_words[4], bias_words[5], bias_words[6], bias_words[7], s_ring.ptr_to([bias_base + 256 + col_group * 16]))
                    acc_colsum = None
                    if is_bwd:
                        acc_colsum = txl.alloc_local([8], "float32")
                        for j in range(8):
                            txl.ptx.mov.b32(acc_colsum[j], txl.float32(0.0))
                    row_addr = stage_base + row_group * GU_ROW + col_group * 16
                    for i in range(ROWS_PER_THREAD):
                        if USE_SPLIT_ARRIVAL and i == ROWS_PER_THREAD // 2:
                            full_bar.wait(stages + stage, txl.truncmod(txl.truncdiv(local_idx, txl.int32(stages)), txl.int32(2)))
                        # rows are read from the stage just before their math (few live registers)
                        if is_bwd:
                            txl.ptx.ld.shared.v4.b32(in_words[0], in_words[1], in_words[2], in_words[3], s_ring.ptr_to([row_addr + 16 * i * GU_ROW]))
                            txl.ptx.ld.shared.v2.b32(grad_words[0], grad_words[1], dh_stage.ptr_to(row_group + 16 * i, col_group * 4))
                            for pair_idx in range(2):
                                word0, word1 = swiglu_bwd_words(in_words[2 * pair_idx], in_words[2 * pair_idx + 1], bias_words[2 * pair_idx], bias_words[2 * pair_idx + 1], grad_words[pair_idx], swiglu, use_bf16_clamp)
                                txl.ptx.mov.b32(out_words[4 * i + 2 * pair_idx], word0)
                                txl.ptx.mov.b32(out_words[4 * i + 2 * pair_idx + 1], word1)
                                for word_idx, word in ((2 * pair_idx, word0), (2 * pair_idx + 1, word1)):
                                    lo_bits, hi_bits = split_bf16x2(word)
                                    txl.ptx["add.rn.f32.bf16"](acc_colsum[2 * word_idx], lo_bits, acc_colsum[2 * word_idx])
                                    txl.ptx["add.rn.f32.bf16"](acc_colsum[2 * word_idx + 1], hi_bits, acc_colsum[2 * word_idx + 1])
                        else:
                            def compute_fwd_row(i=i, pair_idxs=range(4)):
                                if 0 in pair_idxs:
                                    txl.ptx.ld.shared.v4.b32(in_words[0], in_words[1], in_words[2], in_words[3], s_ring.ptr_to([row_addr + 16 * i * GU_ROW]))
                                if 2 in pair_idxs:
                                    txl.ptx.ld.shared.v4.b32(in_words[4], in_words[5], in_words[6], in_words[7], s_ring.ptr_to([row_addr + 16 * i * GU_ROW + 256]))
                                for pair_idx in pair_idxs:
                                    word = swiglu_fwd_word(in_words[2 * pair_idx], in_words[2 * pair_idx + 1], bias_words[2 * pair_idx], bias_words[2 * pair_idx + 1], swiglu, use_bf16_clamp)
                                    txl.ptx.mov.b32(out_words[4 * i + pair_idx], word)
                            if not USE_TAIL_SPLIT:
                                compute_fwd_row()
                            elif has_rowwise:
                                with txl.if_(owns_first if i < ROWS_PER_THREAD // 2 else owns_second), txl.then_():
                                    compute_fwd_row()
                            else:
                                with txl.if_(owns_first), txl.then_():
                                    compute_fwd_row(pair_idxs=(0, 1))
                                with txl.if_(owns_second), txl.then_():
                                    compute_fwd_row(pair_idxs=(2, 3))
                        if has_tail:
                            with txl.if_(col_first >= C), txl.then_():
                                txl.ptx.mov.b32(out_words[4 * i + 0], txl.uint32(0))
                                txl.ptx.mov.b32(out_words[4 * i + 1], txl.uint32(0))
                            with txl.if_(col_second >= C), txl.then_():
                                txl.ptx.mov.b32(out_words[4 * i + 2], txl.uint32(0))
                                txl.ptx.mov.b32(out_words[4 * i + 3], txl.uint32(0))
                    return acc_colsum

                acc_colsum = compute_rows()
                consumer_tidx = consumer_warp * 32 + lane

                def col_of_amax_word(word_idx):
                    """
                    First of the two output channels in smem word word_idx (= col_group * 4 + pair).
                    """
                    word_col_group = txl.truncdiv(word_idx, txl.int32(4))
                    word_pair = txl.truncmod(word_idx, txl.int32(4))
                    if is_bwd:
                        return col_tile * BLOCK + word_col_group * 8 + 2 * word_pair
                    return col_tile * BLOCK + word_col_group * 4 + 2 * word_pair + txl.truncdiv(word_pair, txl.int32(2)) * (64 - 4)

                def stage_warp_partials():
                    """
                    Per-warp column amax (kmajor) and column sums (backward) into this tile's slot.
                    """
                    if has_kmajor:
                        col_amax = []
                        for pair_idx in range(4):
                            amax = out_words[pair_idx]
                            for i in range(1, ROWS_PER_THREAD):
                                amax = amax_bf16x2(amax, out_words[4 * i + pair_idx])
                            amax = amax_bf16x2(amax, shfl_xor(amax, 16))
                            col_amax.append(amax)
                        with txl.if_(lane < 16), txl.then_():
                            txl.ptx.st.shared.v4.b32(s_col_amax.ptr_to([amax_slot_base + consumer_warp * 64 + col_group * 4]), col_amax[0], col_amax[1], col_amax[2], col_amax[3])
                    if is_bwd:
                        for j in range(8):
                            txl.ptx.mov.b32(acc_colsum[j], ptx_f32("add.rn.f32", acc_colsum[j], txl.reinterpret(shfl_xor(txl.reinterpret(acc_colsum[j], ty="uint32"), 16), ty="float32")))
                        with txl.if_(lane < 16), txl.then_():
                            txl.ptx.st.shared.v4.f32(s_colsum.ptr_to([colsum_slot_base + consumer_warp * BLOCK + col_group * 8]), acc_colsum[0], acc_colsum[1], acc_colsum[2], acc_colsum[3])
                            txl.ptx.st.shared.v4.f32(s_colsum.ptr_to([colsum_slot_base + consumer_warp * BLOCK + col_group * 8 + 4]), acc_colsum[4], acc_colsum[5], acc_colsum[6], acc_colsum[7])

                def release_stage():
                    """
                    Group barrier after the stage reads, then warp 0 frees the stage and refills it.
                    """
                    # Backward can overlap its independent rowwise quantization with warp 0's
                    # refill. The non-issuing warps only arrive here; a second named barrier
                    # below keeps every shared reduction read behind completion of this barrier.
                    if is_bwd:
                        with txl.if_(consumer_warp == 0):
                            with txl.then_():
                                txl.ptx.bar.sync(txl.Cast("uint32", consumer_bar), txl.uint32(CONSUMER_WARPS * 32))
                            with txl.else_():
                                txl.ptx.bar.arrive(txl.Cast("uint32", consumer_bar), txl.uint32(CONSUMER_WARPS * 32))
                    else:
                        txl.ptx.bar.sync(txl.Cast("uint32", consumer_bar), txl.uint32(CONSUMER_WARPS * 32))
                    with txl.if_(consumer_warp == 0), txl.then_():
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.mbarrier.arrive.shared.b64(empty_bar.buf.ptr_to([stage]), txl.uint32(1))
                        issue_tile(local_idx + stages)

                def store_row_words(dst, i, word0, word1):
                    offset = out_row_base + 16 * i * C
                    if not has_tail and not (USE_TAIL_SPLIT and has_kmajor):
                        if is_bwd:
                            txl.ptx.st.global_.v2.b32(dst.ptr_to([offset + txl.Cast("int64", col_first)]), word0, word1)
                        else:
                            txl.ptx.st.global_.b32(dst.ptr_to([offset + txl.Cast("int64", col_first)]), word0)
                            txl.ptx.st.global_.b32(dst.ptr_to([offset + txl.Cast("int64", col_second)]), word1)
                    elif is_bwd:
                        with txl.if_(second_ok):
                            with txl.then_():
                                txl.ptx.st.global_.v2.b32(dst.ptr_to([offset + txl.Cast("int64", col_first)]), word0, word1)
                            with txl.else_():
                                with txl.if_(first_ok), txl.then_():
                                    txl.ptx.st.global_.b32(dst.ptr_to([offset + txl.Cast("int64", col_first)]), word0)
                    else:
                        with txl.if_(first_ok), txl.then_():
                            txl.ptx.st.global_.b32(dst.ptr_to([offset + txl.Cast("int64", col_first)]), word0)
                        with txl.if_(second_ok), txl.then_():
                            txl.ptx.st.global_.b32(dst.ptr_to([offset + txl.Cast("int64", col_second)]), word1)

                def quantize_rows():
                    for row_pair in range(ROWS_PER_THREAD // 2):
                        def quantize_row_pair(row_pair=row_pair):
                            amax_first = amax_bf16x2(amax_bf16x2(out_words[8 * row_pair], out_words[8 * row_pair + 1]), amax_bf16x2(out_words[8 * row_pair + 2], out_words[8 * row_pair + 3]))
                            amax_second = amax_bf16x2(amax_bf16x2(out_words[8 * row_pair + 4], out_words[8 * row_pair + 5]), amax_bf16x2(out_words[8 * row_pair + 6], out_words[8 * row_pair + 7]))
                            amax = amax_bf16x2(ptx_u32("prmt.b32", amax_first, amax_second, txl.uint32(0x5410)), ptx_u32("prmt.b32", amax_first, amax_second, txl.uint32(0x7632)))
                            for lane_xor in (1, 2, 4, 8):
                                amax = amax_bf16x2(amax, shfl_xor(amax, lane_xor))
                            e8m0 = e8m0_pair(abs_bf16x2(amax))
                            e8m0_lo, e8m0_hi = txl.bitwise_and(e8m0, txl.uint32(0xFF)), txl.shift_right(e8m0, txl.uint32(8))
                            for half, row_e8m0 in ((0, e8m0_lo), (1, e8m0_hi)):
                                i = 2 * row_pair + half
                                with txl.if_(col_group == 0), txl.then_():
                                    txl.ptx.st.global_.f32(x_scale.ptr_to([row_scale_base + 16 * i]), e8m0_to_scale(row_e8m0))
                                rcp = e8m0_to_rcp_bf16(row_e8m0)
                                rcp_x2 = txl.bitwise_or(rcp, txl.shift_left(rcp, txl.uint32(16)))
                                store_row_words(x_fp8, i, quantize_fp8x4(out_words[4 * i], out_words[4 * i + 1], rcp_x2, rcp_x2), quantize_fp8x4(out_words[4 * i + 2], out_words[4 * i + 3], rcp_x2, rcp_x2))
                        if USE_TAIL_SPLIT:
                            with txl.if_(owns_first if row_pair < ROWS_PER_THREAD // 4 else owns_second), txl.then_():
                                quantize_row_pair()
                        else:
                            quantize_row_pair()

                def reduce_col_scales():
                    """
                    One thread per amax word: the group's warp partials -> E8M0 scales (stored) and
                    the bf16x2 reciprocals every thread of the group needs (broadcast through smem).
                    """
                    with txl.if_(consumer_tidx < 64), txl.then_():
                        acc_amax, peer_amax = txl.local_scalar("uint32"), txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(acc_amax, s_col_amax.ptr_to([amax_slot_base + consumer_tidx]))
                        for peer_warp in range(1, CONSUMER_WARPS):
                            txl.ptx.ld.shared.b32(peer_amax, s_col_amax.ptr_to([amax_slot_base + peer_warp * 64 + consumer_tidx]))
                            txl.ptx.mov.b32(acc_amax, amax_bf16x2(acc_amax, peer_amax))
                        e8m0 = e8m0_pair(abs_bf16x2(acc_amax))
                        e8m0_lo, e8m0_hi = txl.bitwise_and(e8m0, txl.uint32(0xFF)), txl.shift_right(e8m0, txl.uint32(8))
                        txl.ptx.st.shared.b32(s_col_rcp.ptr_to([slot_idx * 64 + consumer_tidx]), txl.bitwise_or(e8m0_to_rcp_bf16(e8m0_lo), txl.shift_left(e8m0_to_rcp_bf16(e8m0_hi), txl.uint32(16))))
                        col = col_of_amax_word(consumer_tidx)
                        col_ok = col < C
                        if USE_TAIL_SPLIT:
                            # words col_group*4 + 0..1 hold the first-half channels (col_first), 2..3 the second
                            col_ok = txl.And(col_ok, txl.Select(txl.truncmod(consumer_tidx, txl.int32(4)) < 2, txl.Cast("int32", owns_first), txl.Cast("int32", owns_second)) == 1)
                        with txl.if_(col_ok), txl.then_():
                            txl.ptx.st.global_.v2.f32(x_ch_scale.ptr_to([txl.Cast("int64", row_block) * C + txl.Cast("int64", col)]), e8m0_to_scale(e8m0_lo), e8m0_to_scale(e8m0_hi))

                def store_block_sums():
                    """
                    One thread per column: the 128-row block sum over the group's warp partials.
                    """
                    with txl.if_(txl.And(consumer_tidx >= 64, consumer_tidx < 64 + BLOCK)), txl.then_():
                        col_idx = consumer_tidx - 64
                        acc_sum, peer_sum = txl.local_scalar("float32"), txl.local_scalar("float32")
                        txl.ptx.ld.shared.f32(acc_sum, s_colsum.ptr_to([colsum_slot_base + col_idx]))
                        for peer_warp in range(1, CONSUMER_WARPS):
                            txl.ptx.ld.shared.f32(peer_sum, s_colsum.ptr_to([colsum_slot_base + peer_warp * BLOCK + col_idx]))
                            txl.assign(acc_sum, ptx_f32("add.rn.f32", acc_sum, peer_sum))
                        with txl.if_(col_tile * BLOCK + col_idx < C), txl.then_():
                            txl.ptx.st.global_.f32(partial.ptr_to([txl.Cast("int64", row_block) * C + txl.Cast("int64", col_tile * BLOCK + col_idx)]), acc_sum)

                def quantize_cols():
                    txl.ptx.bar.sync(txl.Cast("uint32", consumer_bar + 2), txl.uint32(CONSUMER_WARPS * 32))
                    col_rcp = txl.alloc_local([4], "uint32")
                    txl.ptx.ld.shared.v4.b32(col_rcp[0], col_rcp[1], col_rcp[2], col_rcp[3], s_col_rcp.ptr_to([slot_idx * 64 + col_group * 4]))
                    for i in range(ROWS_PER_THREAD):
                        store_row_words(x_ch, i, quantize_fp8x4(out_words[4 * i], out_words[4 * i + 1], col_rcp[0], col_rcp[1]), quantize_fp8x4(out_words[4 * i + 2], out_words[4 * i + 3], col_rcp[2], col_rcp[3]))

                stage_warp_partials()
                release_stage()
                if is_bwd and has_rowwise:
                    quantize_rows()
                    # Orders the arrive-only warps' shared reduction stores before the
                    # column-amax and d_bias readers, and prevents barrier-generation overrun.
                    txl.ptx.bar.sync(txl.Cast("uint32", consumer_bar + 1), txl.uint32(CONSUMER_WARPS * 32))
                if has_kmajor:
                    reduce_col_scales()
                if is_bwd:
                    store_block_sums()
                if has_rowwise and not is_bwd:
                    quantize_rows()
                if has_kmajor:
                    quantize_cols()
            if USE_DYNAMIC:
                txl.assign(step_counter, step_counter + 1)

        def exit_and_fold_dbias():
            """
            Exit protocol and in-kernel d_bias (backward). sched = [tile counter, exit word (low 16
            bits: CTAs out, high 16 bits: participants), started CTAs, participants done]. A CTA
            that exits after every CTA has started registers as a participant (waiting on the
            others cannot deadlock then). The exit RMW releases this CTA's partial stores (ordered
            by the CTA barrier); a participant acquires every CTA's once the low half reaches
            num_ctas. If no CTA registered, the last CTA out reduces everything alone. The last
            participant to read the exit word returns all counters to zero.
            """
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(CTA_WARPS * 32))
            with txl.if_(tidx == 0), txl.then_():
                num_started = txl.local_scalar("uint32")
                txl.ptx.ld.relaxed.gpu.global_.u32(num_started, sched.ptr_to([2]))
                # Register only CTAs that can own a d_bias chunk. Mode 1 assigns one
                # (group, 128-column) chunk per participant; mode 0 assigns eight, one
                # per warp. The last CTA still falls back to one participant if all of
                # the selected CTAs happened to exit before the full grid had started.
                num_chunks = num_groups * ((C + 127) // 128)
                num_wanted = txl.local_scalar("int32", init=txl.Select(seg_mode == 1, num_chunks, txl.ceildiv(num_chunks, txl.int32(8))))
                txl.assign(num_wanted, txl.Select(num_wanted < num_ctas, num_wanted, num_ctas))
                is_participant = txl.local_scalar("uint32", init=txl.Select(txl.And(num_started == txl.Cast("uint32", num_ctas), cta < num_wanted), txl.uint32(1), txl.uint32(0)))
                exit_word = txl.local_scalar("uint32")
                txl.ptx.atom.acq_rel.gpu.global_.add.u32(exit_word, sched.ptr_to([1]), txl.uint32(1) + txl.shift_left(is_participant, txl.uint32(16)))
                part_idx = txl.local_scalar("int32", init=txl.int32(-1))
                num_parts = txl.local_scalar("int32", init=txl.int32(0))
                if is_bwd:
                    with txl.if_(is_participant == txl.uint32(1)):
                        with txl.then_():
                            txl.assign(part_idx, txl.Cast("int32", txl.shift_right(exit_word, txl.uint32(16))))
                            final_word = txl.local_scalar("uint32")
                            txl.cuda.wait_until(final_word, sched.ptr_to([1]), txl.bitwise_and(final_word, txl.uint32(0xFFFF)) == txl.Cast("uint32", num_ctas), scope="gpu", ptx_type="b32", ty=txl.void)  # ty: tvm-tirx b0 types the Call Missing, which drops it unemitted
                            txl.assign(num_parts, txl.Cast("int32", txl.shift_right(final_word, txl.uint32(16))))
                        with txl.else_():
                            with txl.if_(txl.And(txl.bitwise_and(exit_word, txl.uint32(0xFFFF)) == txl.Cast("uint32", num_ctas - 1), txl.shift_right(exit_word, txl.uint32(16)) == txl.uint32(0))), txl.then_():
                                txl.assign(part_idx, txl.int32(0))
                                txl.assign(num_parts, txl.int32(1))
                    with txl.if_(part_idx >= 0), txl.then_():
                        num_done = txl.local_scalar("uint32")
                        txl.ptx.atom.acq_rel.gpu.global_.add.u32(num_done, sched.ptr_to([3]), txl.uint32(1))
                        with txl.if_(num_done == txl.Cast("uint32", num_parts - 1)), txl.then_():
                            txl.ptx.st.global_.v4.b32(sched.ptr_to([0]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0))
                    txl.ptx.st.shared.v2.b32(s_role.ptr_to([0]), part_idx, num_parts)
                else:
                    with txl.if_(exit_word == txl.Cast("uint32", num_ctas - 1)), txl.then_():
                        txl.ptx.st.global_.v4.b32(sched.ptr_to([0]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0))
            if is_bwd:
                txl.ptx.bar.sync(txl.uint32(0), txl.uint32(CTA_WARPS * 32))
                part_idx = txl.local_scalar("int32")
                num_parts = txl.local_scalar("int32")
                txl.ptx.ld.shared.v2.b32(part_idx, num_parts, s_role.ptr_to([0]))
                with txl.if_(part_idx >= 0), txl.then_():
                    fold_dbias(partial, dbias, s_offs, s_colsum, num_groups, seg_mode, part_idx, num_parts, tidx, lane, warp, C)

        if USE_DYNAMIC:
            exit_and_fold_dbias()

    return compile_kernel(kern)


@contextlib.contextmanager
def guard_tile(cond, is_running):
    """
    Body under If(cond) (no-op when cond is None); else clears the loop flag is_running.
    """
    if cond is None:
        yield
        return
    with txl.if_(cond):
        with txl.then_():
            yield
        with txl.else_():
            txl.assign(is_running, txl.int32(0))


def compile_kernel(kernel):
    target = TARGET
    with target:
        return entry_function(tvm_tirx.tirx.compile(kernel.mod, target=target, pipeline="tirx"))


SCHED_COUNTERS = {}


def get_sched_counters(device):
    """
    [tile counter, exit word, started CTAs, participants done] for the dynamic backward schedule
    and its in-kernel d_bias: zero between launches (reset at the end of each launch). One set per
    (device, stream), so launches on two streams never share a counter.
    """
    key = (device.type, device.index, torch.cuda.current_stream(device).cuda_stream)
    if key not in SCHED_COUNTERS:
        SCHED_COUNTERS[key] = torch.zeros(4, dtype=torch.int32, device=device).view(torch.uint32)
    return SCHED_COUNTERS[key]


def alloc_quant_outputs(M, C, device):
    """
    Uninitialized (x_fp8, x_scale, x_ch, x_ch_scale) for M rows of C channels.
    """
    return (
        torch.empty((M, C), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M, (C + BLOCK - 1) // BLOCK), dtype=torch.float32, device=device),
        torch.empty((M, C), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M // BLOCK, C), dtype=torch.float32, device=device),
    )


def launch_clamped_swiglu_quant(kind, gate_up, bias, offs, alpha, limit, dh=None):
    """
    Run one of the fused kernels; kind is "quant_fwd", "rowwise", "kmajor" or "bwd". Returns the
    four quantized outputs and d_bias (None unless bwd).
    """
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    is_bwd = kind == "bwd"
    C = 2 * N if is_bwd else N
    device = gate_up.device
    x_fp8, x_scale, x_ch, x_ch_scale = alloc_quant_outputs(M, C, device)
    d_bias = None
    if is_bwd:
        d_bias = torch.zeros(bias.shape, dtype=bias.dtype, device=device) if M == 0 or N == 0 else torch.empty(bias.shape, dtype=bias.dtype, device=device)
    if M == 0 or N == 0:
        return (x_fp8, x_scale, x_ch, x_ch_scale), d_bias
    has_rowwise = kind in ("rowwise", "quant_fwd", "bwd")
    has_kmajor = kind in ("kmajor", "quant_fwd", "bwd")
    limit_bits = struct.unpack("<I", struct.pack("<f", float(limit)))[0]
    # the bf16x2 clamp needs limit >= 0 exactly representable in bf16 (gpt-oss: 7.0)
    use_bf16_clamp = (limit_bits & 0xFFFF) == 0 and limit_bits < 0x7F800000
    G = offs.shape[0]
    num_col_tiles = (C + BLOCK - 1) // BLOCK
    partial = torch.empty((M // BLOCK, C), dtype=torch.float32, device=device) if is_bwd else x_ch_scale
    dh_arg = dh if is_bwd else gate_up
    tensors = (gate_up.view(-1), bias.view(-1), offs, dh_arg.reshape(-1), x_fp8.view(torch.uint8).view(-1), x_scale.view(-1), x_ch.view(torch.uint8).view(-1), x_ch_scale.view(-1), partial.view(-1))
    num_tiles = (M // BLOCK) * num_col_tiles
    use_pipe = N % 8 == 0 and G <= MAX_SMEM_GROUPS
    if use_pipe:
        # MN-major rowwise scales (scale strides are unchecked): a tile's 128 row scales
        # are one contiguous run; returned as the (M, ceil(C/128)) transposed view
        x_scale = torch.empty((num_col_tiles, M), dtype=torch.float32, device=device).t()
        tensors = tensors[:5] + (x_scale.t().reshape(-1),) + tensors[6:]
        tensors = tensors + (get_sched_counters(device), d_bias.view(-1) if is_bwd else bias.view(-1))
        num_ctas = min(num_tiles, (2 if is_bwd else 1) * count_sms())
        # backward: split-arrival stages only when each CTA gets few tiles (ramp/tail dominated)
        fn = compile_pipe_kernel(N, is_bwd, has_rowwise, has_kmajor, use_bf16_clamp, 2 if is_bwd else 3, is_bwd and num_tiles < BWD_SPLIT_TILES_PER_CTA * num_ctas)
        # d_bias is folded into the backward kernel: 8-warp runs for large groups, else one warp per chunk
        seg_mode = 1 if (M // BLOCK) >= 16 * G else 0
        fn(*tensors, float(alpha), float(limit), G, num_tiles, num_ctas, M, seg_mode)
    else:
        fn = compile_tile_kernel(N, is_bwd, has_rowwise, has_kmajor, use_bf16_clamp)
        fn(*tensors, float(alpha), float(limit), G, num_tiles)
    if is_bwd and not use_pipe:
        # by the mean blocks per group: large groups -> 8 warps split each group's blocks;
        # tiny groups -> one warp per (group, 128 columns) so the many CTAs fit one wave;
        # otherwise one thread per column (measured best near 8 blocks per group)
        num_blocks = M // BLOCK
        if C % 4 != 0 or 4 * G <= num_blocks < 16 * G:
            segsum_fn = compile_segsum_kernel(C)
        elif num_blocks >= 16 * G:
            segsum_fn = compile_segsum_split_kernel(C)
        else:
            segsum_fn = compile_segsum_warp_kernel(C)
        segsum_fn(partial.view(-1), offs, d_bias.view(-1), G, G * ((C + 127) // 128))
    return (x_fp8, x_scale, x_ch, x_ch_scale), d_bias


def clamped_swiglu_quant_fwd_sm100(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels=None):
    """
    SM100 clamped_swiglu_quant_fwd; kmajor_channels is ignored (N channels).
    """
    return launch_clamped_swiglu_quant("quant_fwd", gate_up, bias, grouped_mm_offs, alpha, limit)[0]


def fp8_clamped_swiglu_rowwise_sm100(gate_up, bias, grouped_mm_offs, alpha, limit):
    """
    SM100 fp8_clamped_swiglu_rowwise.
    """
    return launch_clamped_swiglu_quant("rowwise", gate_up, bias, grouped_mm_offs, alpha, limit)[0][:2]


def fp8_clamped_swiglu_kmajor_sm100(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels=None):
    """
    SM100 fp8_clamped_swiglu_kmajor; kmajor_channels is ignored (N channels).
    """
    return launch_clamped_swiglu_quant("kmajor", gate_up, bias, grouped_mm_offs, alpha, limit)[0][2:]


def fp8_clamped_swiglu_bwd_sm100(dh, gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels=None):
    """
    SM100 fp8_clamped_swiglu_bwd; kmajor_channels is ignored (2N channels).
    """
    return launch_clamped_swiglu_quant("bwd", gate_up, bias, grouped_mm_offs, alpha, limit, dh=dh)
