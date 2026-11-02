"""
causal_conv1d_sm100: the causal depthwise conv1d of the Qwen3.5 Gated DeltaNet (forward y, backward
dx and dw) on SM100, as TIRx-lite kernels. Contracts as in the picker causal_conv1d.

Tiles are T = 128 tokens x CB = 64 channels. Every output runs the K-tap fp32 fma chain in tap order
from 0.0, rounded once (bit-identical to ATen).

TMA path (C % 8 == 0, K <= 8, 16-byte aligned tensors, and S % 8 == 0 where a channels-first tensor
is involved): persistent CTAs, one producer warp streams tiles into a shared-memory ring with TMA
(OOB rows read as zero, which is exactly the causal padding / history clipping), compute warps read
the ring and write the results to a double-buffered staging tile that one thread stores with TMA.
- forward: x tile rows [t0 + offset - (K - 1), +256 + K - 1) channels-last (a 256-row head box and a
  K - 1 row halo box, evict-last L2 hints), lane = channel pair with f32x2 FMAs; y staging is two
  128B-swizzled [64 ch][64 tok] boxes, stored into y viewed as (B, C, S).
- backward: 2-CTA clusters per channel slice, g rows [s0 - offset, +T + K - 1) (channels-last, or
  [ch][CF_P tok] for g in the output's channels-first layout) and x rows [s0 - (K - 1), +T + K - 1);
  dx staging [T][slice]. BF16 w on channels-last g computes with fma.rn.f32.bf16 straight from the
  staged halves. dw: per thread a sequential fma over its tokens, the token groups summed in order,
  then the cluster's ranks summed in order through DSMEM. Same order in every path, so dx / dw bits
  do not depend on g's layout.
Plain path (any C, S, K, strides of g): the same tiles and arithmetic with plain loads / stores, dw
partials summed in the TMA path's order by a second kernel.
Inputs outside the contract the kernels read directly (w of another float dtype, non-contiguous x
or w) are converted first: w to fp32 (as the Triton path reads it), dw rounded back to w's dtype.

Evolved with TIRx-harness: run causal_conv1d, frontier member tma-ring-cluster-dw (commit f48a3abe,
305.2 us vs 793.6 us, 2.6x over the previous SM100 path, the Triton kernels now in
causal_conv1d_sm90, in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools

import torch

import tirx_kernels.tirx_lite as txl

__all__ = ["causal_conv1d_bwd_sm100", "causal_conv1d_fwd_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


T = 128  # tokens (rows) per tile
CB = 64  # channels per tile
TG = 32  # tokens per compute thread (scalar fallback kernels)
TG2 = 16  # tokens per compute thread (packed TMA kernels: 2 channels x TG2 tokens)
NTG = 8  # dw token groups per tile (T // TG2), the same in every path
NCW = 8  # compute warps
NC = NCW * 32  # compute threads
NS_FWD = 3  # ring stages
NS_BWD = 2
CTAS_PER_SM = 2
NUM_SMS = 148
WARP_ARRIVE = False  # one empty arrival per thread (per-warp arrivals measured noise-level)
EMPTY_COUNT = NCW if WARP_ARRIVE else NC
CLM = True  # packed channels-last backward with native BF16-input FP32 FMAs (fma.rn.f32.bf16)
FWM = False  # forward mixed FMAs measured noise / slower at S=8192
FWO = True  # forward: convert (and shuffle) before the staging barrier
CF_P = 152  # channels-first g tile pitch (tokens): >= 3 * TG + 48, and 76 words = 12 mod 32 banks, so the
# 16B reads of 8 lanes (8 channels) along tokens hit distinct bank quads

WDT = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def cdiv(a, b):
    return (a + b - 1) // b


def i64(e):
    return txl.Cast("int64", e)


def vec_width(C):
    for v in (8, 4, 2, 1):
        if C % v == 0:
            return v
    return 1


# ------------------------------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------------------------------


def load_w_f32(dst, w, idx, wdt, valid):
    """
    dst = float32(w[idx]) if valid else 0.
    """
    txl.assign(dst, txl.float32(0.0))
    with txl.If(valid), txl.Then():
        if wdt == "float32":
            txl.ptx.ld.global_.f32(dst, w.ptr_to([idx]))
        else:
            h = txl.local_scalar("uint16")
            txl.ptx.ld.global_.b16(h, w.ptr_to([idx]))
            if wdt == "bfloat16":
                txl.ptx.cvt.f32.bf16(dst, h)
            else:
                txl.ptx.cvt.f32.f16(dst, h)


def load_w_bf16(dst, w, idx, valid):
    txl.assign(dst, txl.uint16(0))
    with txl.If(valid), txl.Then():
        txl.ptx.ld.global_.b16(dst, w.ptr_to([idx]))


def ld_s_bf16(dst, sptr):
    h = txl.local_scalar("uint16")
    txl.ptx["ld.shared.b16"](h, sptr)
    txl.ptx.cvt.f32.bf16(dst, h)


def cvta(ptr):
    return txl.cuda.cvta_generic_to_shared(ptr)


def tmap_ptr(m):
    return txl.reinterpret(txl.handle().ty, txl.address_of(m))


def load_policy():
    pol = txl.local_scalar("uint64")
    txl.ptx.createpolicy.fractional.L2__evict_last.b64(pol, txl.float32(1.0))
    return pol


def tma_load(dst, tmap, c0, c1, c2, mbar, pol):
    name = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
    txl.ptx[name + ".L2::cache_hint"](dst, tmap, c0, c1, c2, mbar, pol)


def tma_store(tmap, c0, c1, c2, src):
    txl.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](tmap, c0, c1, c2, src)


def bar_named(i, n):
    txl.ptx.bar.sync(txl.uint32(i), txl.uint32(n))


def taps_fwd(out, wk, xv, K):
    acc = txl.local_scalar("float32")
    for j in range(TG):
        txl.assign(acc, txl.float32(0.0))
        for k in range(K):
            txl.ptx.fma.rn.f32(acc, wk[k], xv[j + k], acc)
        txl.assign(out[j], acc)


def taps_dx(out, wk, gv, K):
    acc = txl.local_scalar("float32")
    for i in range(TG):
        txl.assign(acc, txl.float32(0.0))
        for k in range(K):
            txl.ptx.fma.rn.f32(acc, wk[k], gv[i + K - 1 - k], acc)
        txl.assign(out[i], acc)


class Sub:
    """
    pw[base + k] window of a local array.
    """

    def __init__(self, a, base):
        self.a, self.base = a, base

    def __getitem__(self, k):
        return self.a[self.base + k]


def partial_dw(pw, gv, xv, K, i0=0, n=TG):
    for k in range(K):
        txl.assign(pw[k], txl.float32(0.0))
        for i in range(i0, i0 + n):
            txl.ptx.fma.rn.f32(pw[k], gv[i], xv[i + k], pw[k])


def taps_dx_mixed(out, wk, gv, K):
    acc = txl.local_scalar("float32")
    for i in range(TG):
        txl.assign(acc, txl.float32(0.0))
        for k in range(K):
            txl.ptx["fma.rn.f32.bf16"](acc, wk[k], gv[i + K - 1 - k], acc)
        txl.assign(out[i], acc)


def partial_dw_mixed(pw, gv, xv, K, i0=0, n=TG):
    for k in range(K):
        txl.assign(pw[k], txl.float32(0.0))
        for i in range(i0, i0 + n):
            txl.ptx["fma.rn.f32.bf16"](pw[k], gv[i], xv[i + k], pw[k])


# packed (f32x2) helpers: one thread owns channels (c, c + 1), lanes .x / .y


def bf16x2_to_f32x2(word):
    lo = txl.reinterpret("float32", word << txl.uint32(16))
    hi = txl.reinterpret("float32", word & txl.uint32(0xFFFF0000))
    return txl.cuda.make_float2(lo, hi)


def f2(v, e):
    return txl.cuda.float2_x(v) if e == 0 else txl.cuda.float2_y(v)


def load_w_pair(wk, wf, w, c, C, K, wdt):
    for k in range(K):
        for e in range(2):
            load_w_f32(wf[e], w, (c + e) * K + k, wdt, c + e < C)
        txl.assign(wk[k], txl.cuda.make_float2(wf[0], wf[1]))


def taps2(out, wk, v, K, rev):
    """
    out[i] = sum over k in order of wk[k] * v[i + k] (rev: v[i + K - 1 - k]), fp32 fma from 0.
    """
    for i in range(TG2):
        txl.assign(out[i], txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))
        for k in range(K):
            txl.ptx["fma.rn.f32x2"](out[i], wk[k], v[i + (K - 1 - k if rev else k)], out[i])


def release(empty, stage, lane):
    """
    Consumer release of a ring stage: one arrival per thread, or one per warp after warp_sync.
    """
    if WARP_ARRIVE:
        txl.cuda.warp_sync()
        with txl.If(lane == 0), txl.Then():
            txl.ptx["mbarrier.arrive.shared.b64"](cvta(empty.ptr_to([stage])), txl.uint32(1))
    else:
        txl.ptx["mbarrier.arrive.shared.b64"](cvta(empty.ptr_to([stage])), txl.uint32(1))


def shfl_idx_u32(dst, src, source_lane):
    txl.ptx.shfl_sync.idx.b32(
        dst, src, txl.Cast("uint32", source_lane), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
    )


def partial_dw2(pw, gv, xv, K):
    for k in range(K):
        txl.assign(pw[k], txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))
        for i in range(TG2):
            txl.ptx["fma.rn.f32x2"](pw[k], gv[i], xv[i + k], pw[k])


# ------------------------------------------------------------------------------------------
# TMA forward
# ------------------------------------------------------------------------------------------


@functools.cache
def fwd_tma(C, K, wdt, shuffle_store):
    FT = 256
    R = FT + K - 1
    NCB = cdiv(C, CB)
    # Two 259x64 input stages plus the 32 KiB output ring fit two CTAs per SM.
    NS = 2
    XBYTES = R * CB * 2
    G = NUM_SMS * CTAS_PER_SM

    def host_prelude(p):
        xmh = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            xmh, "bfloat16", 3, txl.handle_add_byte_offset(p["x"].data, 0),
            C, p["L"], p["B"],
            C * 2, i64(p["L"]) * (C * 2),
            CB, FT, 1,
            1, 1, 1,
            0, 0, 3, 0,
        )
        xmt = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            xmt, "bfloat16", 3, txl.handle_add_byte_offset(p["x"].data, 0),
            C, p["L"], p["B"],
            C * 2, i64(p["L"]) * (C * 2),
            CB, K - 1, 1,
            1, 1, 1,
            0, 0, 3, 0,
        )
        ym = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            ym, "bfloat16", 3, txl.handle_add_byte_offset(p["y"].data, 0),
            p["S"], C, p["B"],
            i64(p["S"]) * 2, i64(p["S"]) * (C * 2),
            64, CB, 1,
            1, 1, 1,
            0, 3, 3, 0,
        )
        return (xmh, xmt, ym)

    @txl.kernel(warps=NCW + 1, grid=G, min_blocks_per_sm=CTAS_PER_SM, host_prelude=host_prelude)
    def conv_fwd_tma(
        x: txl.gptr[txl.bf16],
        w: txl.gptr[wdt],
        y: txl.gptr[txl.bf16],
        L: txl.i32,
        S: txl.i32,
        offset: txl.i32,
        B: txl.i32,
        *,
        host,
    ):
        xmh, xmt, ym = host
        cta = txl.cta_id()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        nbt = (S + FT - 1) // FT
        ntiles = nbt * NCB * B

        smem = txl.smem_pool()
        ys = smem.alloc([2 * 2 * CB * 64], "bfloat16", align=1024)
        xs = smem.alloc([NS * R * CB], "bfloat16", align=128)
        full = smem.alloc([NS], "uint64", align=8)
        empty = smem.alloc([NS], "uint64", align=8)

        with txl.If(tid == 0), txl.Then():
            for s in range(NS):
                txl.ptx["mbarrier.init.shared.b64"](cvta(full.ptr_to([s])), txl.uint32(1))
                txl.ptx["mbarrier.init.shared.b64"](cvta(empty.ptr_to([s])), txl.uint32(EMPTY_COUNT))
            txl.ptx["fence.mbarrier_init.release.cluster"]()
            txl.ptx["fence.proxy.async.shared::cta"]()
        bar_named(0, (NCW + 1) * 32)

        stage = txl.local_scalar("int32", init=txl.int32(0))
        phase = txl.local_scalar("int32", init=txl.int32(0))
        tile = txl.local_scalar("int32", init=cta)

        def advance():
            txl.assign(stage, stage + 1)
            with txl.If(stage == NS), txl.Then():
                txl.assign(stage, txl.int32(0))
                txl.assign(phase, phase ^ 1)
            txl.assign(tile, tile + G)

        def decode(tl):
            # Token tile fastest: adjacent CTAs write adjacent token ranges of
            # the same channels-first output rows.
            bt = tl % nbt
            rest = tl // nbt
            return rest % NCB, bt, rest // NCB

        with txl.If(warp == NCW):
            with txl.Then():
                with txl.If(lane == 0), txl.Then():
                    txl.ptx.prefetch.tensormap(txl.address_of(xmh))
                    txl.ptx.prefetch.tensormap(txl.address_of(xmt))
                    txl.ptx.prefetch.tensormap(txl.address_of(ym))
                    lpol = load_policy()
                    with txl.While(tile < ntiles):
                        cb, bt, bb = decode(tile)
                        txl.cuda.mbarrier_wait(empty.ptr_to([stage]), phase ^ 1)
                        # order the compute warps' generic reads of this stage before the TMA rewrite
                        txl.ptx["fence.proxy.async.shared::cta"]()
                        txl.ptx["mbarrier.arrive.expect_tx.shared.b64"](
                            cvta(full.ptr_to([stage])), txl.uint32(XBYTES)
                        )
                        xrow = bt * FT + offset - (K - 1)
                        fb = cvta(full.ptr_to([stage]))
                        tma_load(cvta(xs.ptr_to([stage * (R * CB)])), tmap_ptr(xmh),
                                  cb * CB, xrow, bb, fb, lpol)
                        tma_load(cvta(xs.ptr_to([stage * (R * CB) + FT * CB])), tmap_ptr(xmt),
                                  cb * CB, xrow + FT, bb, fb, lpol)
                        advance()
            with txl.Else():
                # lane = channel pair (2 * lane, 2 * lane + 1), warp = token group of TG2 tokens
                c2 = 2 * lane
                tg = warp
                wk = txl.alloc_local([K], "uint64")
                wf = txl.alloc_local([2], "float32")
                xv = txl.alloc_local([TG2 + K - 1], "uint64")
                out = txl.alloc_local([TG2], "uint64")
                xw = txl.local_scalar("uint32")
                if shuffle_store:
                    pka = txl.alloc_local([4], "uint32")
                    pkb = txl.alloc_local([4], "uint32")
                    sp = txl.alloc_local([4], "uint32")
                    sa = txl.local_scalar("uint32")
                    sb = txl.local_scalar("uint32")
                else:
                    pk = txl.alloc_local([4], "uint32")
                if shuffle_store:
                    spa = txl.alloc_local([16], "uint32")
                else:
                    pka2 = txl.alloc_local([16], "uint32")
                fwm = FWM and wdt == "bfloat16"
                if fwm:
                    wkb = [txl.alloc_local([K], "uint16") for _ in range(2)]
                    xh = [txl.alloc_local([TG2 + K - 1], "uint16") for _ in range(2)]
                    oa = txl.local_scalar("float32")
                    ob = txl.local_scalar("float32")
                it = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(tile < ntiles):
                    cb, bt, bb = decode(tile)
                    if fwm:
                        for e in range(2):
                            for k in range(K):
                                load_w_bf16(wkb[e][k], w, (cb * CB + c2 + e) * K + k, cb * CB + c2 + e < C)
                    else:
                        load_w_pair(wk, wf, w, cb * CB + c2, C, K, wdt)
                    txl.cuda.mbarrier_wait(full.ptr_to([stage]), phase)
                    for ft in range(2):
                        base = stage * (R * CB) + ft * T * CB
                        for j in range(TG2 + K - 1):
                            txl.ptx["ld.shared.b32"](xw, xs.ptr_to([base + (tg * TG2 + j) * CB + c2]))
                            if fwm:
                                txl.assign(xh[0][j], txl.Cast("uint16", xw))
                                txl.assign(xh[1][j], txl.Cast("uint16", xw >> txl.uint32(16)))
                            else:
                                txl.assign(xv[j], bf16x2_to_f32x2(xw))
                        if ft == 1:
                            release(empty, stage, lane)
                        if fwm:
                            for i in range(TG2):
                                txl.assign(oa, txl.float32(0.0))
                                txl.assign(ob, txl.float32(0.0))
                                for k in range(K):
                                    txl.ptx["fma.rn.f32.bf16"](oa, wkb[0][k], xh[0][i + k], oa)
                                    txl.ptx["fma.rn.f32.bf16"](ob, wkb[1][k], xh[1][i + k], ob)
                                txl.assign(out[i], txl.cuda.make_float2(oa, ob))
                        else:
                            taps2(out, wk, xv, K, False)
                        b = it % 2
                        h = tg // 4
                        with txl.If(tid == 0), txl.Then():
                            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(1))
                        if FWO:
                            # convert (and shuffle) while thread 0 waits for the old bulk store to drain
                            if shuffle_store:
                                for qq in range(TG2 // 8):
                                    for p in range(4):
                                        txl.ptx["cvt.rn.bf16x2.f32"](
                                            pka[p], f2(out[qq * 8 + 2 * p + 1], 0), f2(out[qq * 8 + 2 * p], 0)
                                        )
                                        txl.ptx["cvt.rn.bf16x2.f32"](
                                            pkb[p], f2(out[qq * 8 + 2 * p + 1], 1), f2(out[qq * 8 + 2 * p], 1)
                                        )
                                    for cg in range(2):
                                        src_l = cg * 16 + (lane >> 1)
                                        for p in range(4):
                                            shfl_idx_u32(sa, pka[p], src_l)
                                            shfl_idx_u32(sb, pkb[p], src_l)
                                            txl.assign(spa[(qq * 2 + cg) * 4 + p], txl.Select((lane & 1) == 0, sa, sb))
                                bar_named(1, NC)
                                for qq in range(TG2 // 8):
                                    for cg in range(2):
                                        ch = cg * 32 + lane
                                        q = (tg % 4) * (TG2 // 8) + qq
                                        off = ((b * 2 + h) * CB + ch) * 64 + ((q ^ (ch % 8)) * 8)
                                        o4 = (qq * 2 + cg) * 4
                                        txl.ptx["st.shared.v4.b32"](ys.ptr_to([off]), spa[o4], spa[o4 + 1], spa[o4 + 2], spa[o4 + 3])
                            else:
                                for e in range(2):
                                    for qq in range(TG2 // 8):
                                        for p in range(4):
                                            txl.ptx["cvt.rn.bf16x2.f32"](
                                                pka2[(e * 2 + qq) * 4 + p], f2(out[qq * 8 + 2 * p + 1], e), f2(out[qq * 8 + 2 * p], e)
                                            )
                                bar_named(1, NC)
                                for e in range(2):
                                    ch = c2 + e
                                    for qq in range(TG2 // 8):
                                        q = (tg % 4) * (TG2 // 8) + qq
                                        off = ((b * 2 + h) * CB + ch) * 64 + ((q ^ (ch % 8)) * 8)
                                        o4 = (e * 2 + qq) * 4
                                        txl.ptx["st.shared.v4.b32"](ys.ptr_to([off]), pka2[o4], pka2[o4 + 1], pka2[o4 + 2], pka2[o4 + 3])
                        else:
                            bar_named(1, NC)
                            # token chunk q (8 tokens) of half h: physical chunk q ^ (ch % 8) (128B swizzle)
                            if shuffle_store:
                                for qq in range(TG2 // 8):
                                    for p in range(4):
                                        txl.ptx["cvt.rn.bf16x2.f32"](
                                            pka[p], f2(out[qq * 8 + 2 * p + 1], 0), f2(out[qq * 8 + 2 * p], 0)
                                        )
                                        txl.ptx["cvt.rn.bf16x2.f32"](
                                            pkb[p], f2(out[qq * 8 + 2 * p + 1], 1), f2(out[qq * 8 + 2 * p], 1)
                                        )
                                    for cg in range(2):
                                        src = cg * 16 + (lane >> 1)
                                        for p in range(4):
                                            shfl_idx_u32(sa, pka[p], src)
                                            shfl_idx_u32(sb, pkb[p], src)
                                            txl.assign(sp[p], txl.Select((lane & 1) == 0, sa, sb))
                                        ch = cg * 32 + lane
                                        q = (tg % 4) * (TG2 // 8) + qq
                                        off = ((b * 2 + h) * CB + ch) * 64 + ((q ^ (ch % 8)) * 8)
                                        txl.ptx["st.shared.v4.b32"](ys.ptr_to([off]), sp[0], sp[1], sp[2], sp[3])
                            else:
                                for e in range(2):
                                    ch = c2 + e
                                    for qq in range(TG2 // 8):
                                        q = (tg % 4) * (TG2 // 8) + qq
                                        for p in range(4):
                                            txl.ptx["cvt.rn.bf16x2.f32"](
                                                pk[p], f2(out[qq * 8 + 2 * p + 1], e), f2(out[qq * 8 + 2 * p], e)
                                            )
                                        off = ((b * 2 + h) * CB + ch) * 64 + ((q ^ (ch % 8)) * 8)
                                        txl.ptx["st.shared.v4.b32"](ys.ptr_to([off]), pk[0], pk[1], pk[2], pk[3])
                        txl.ptx["fence.proxy.async.shared::cta"]()
                        bar_named(1, NC)
                        with txl.If(tid == 0), txl.Then():
                            for hh in range(2):
                                tma_store(
                                    tmap_ptr(ym), bt * FT + ft * T + hh * 64, cb * CB, bb,
                                    cvta(ys.ptr_to([(b * 2 + hh) * CB * 64])),
                                )
                            txl.ptx["cp.async.bulk.commit_group"]()
                        txl.assign(it, it + 1)
                    advance()
                with txl.If(tid == 0), txl.Then():
                    txl.ptx["cp.async.bulk.wait_group"](txl.int32(0))

    return entry_function(conv_fwd_tma.compile())


# ------------------------------------------------------------------------------------------
# TMA backward
# ------------------------------------------------------------------------------------------


KF = 2  # CTAs per backward cluster; also the dw rank order of every backward path


def kfac(NCB):
    return KF


def cluster_sync():
    txl.ptx.barrier.cluster.arrive.release()
    txl.ptx.barrier.cluster.wait.acquire()


def bwd_plan(C, cf):
    """
    Channel slices of the TMA backward, one per KF-CTA cluster: NCL clusters, the first `extra`
    (base + step) 8-channel units wide and the rest `base` units. For the timed C=8192 case, use
    64/48-channel slices: both row widths are 32-byte-sector aligned, unlike the nearly even
    56/48 split. Other valid channel counts retain the nearly even plan.
    """
    units = C // 8
    # Timed contiguous-g rows keep the best frontier's fixed 64-channel blocks.
    # Only channels-first g uses the 296-CTA balanced partition that won that row.
    if C == 8192 and not cf:
        return 128, 8, 1, 0
    ncl = max(min(NUM_SMS * CTAS_PER_SM // KF, units), cdiv(units, CB // 8))
    if C == 8192 and ncl == NUM_SMS:
        return ncl, 6, 2, 68
    if C == 512:
        return 10, 6, 2, 2
    base, extra = divmod(units, ncl)
    return ncl, base, 1, extra


def rup(a, b):
    return cdiv(a, b) * b


@functools.cache
def bwd_tma(C, K, wdt, cf):
    R = T + K - 1
    NCL, base, step, extra = bwd_plan(C, cf)
    G = NCL * KF
    WA, WB = (base + step) * 8, base * 8  # slice widths: clusters < extra, the others
    WS = [WA, WB] if extra else [WB]
    WMAX = WS[0]
    NS = NS_BWD
    GSZ = rup(WMAX * CF_P if cf else R * WMAX, 64)  # elements of one g stage (128B-aligned stride)
    XSZ = rup(R * WMAX, 64)
    assert 3 * TG + 48 <= CF_P and TG + K - 1 + 7 <= 48 and WMAX <= CB

    def txbytes(W):
        return ((W * CF_P if cf else R * W) + R * W) * 2

    def host_prelude(p):
        maps = []
        for W in WS:
            gm = txl.stack_alloca("tensormap", 1)
            if cf:
                txl.cu_tensor_map_encode_tiled(
                    gm, "bfloat16", 3, txl.handle_add_byte_offset(p["g"].data, 0),
                    p["S"], C, p["B"],
                    i64(p["S"]) * 2, i64(p["S"]) * (C * 2),
                    CF_P, W, 1,
                    1, 1, 1,
                    0, 0, 3, 0,
                )
            else:
                txl.cu_tensor_map_encode_tiled(
                    gm, "bfloat16", 3, txl.handle_add_byte_offset(p["g"].data, 0),
                    C, p["S"], p["B"],
                    C * 2, i64(p["S"]) * (C * 2),
                    W, R, 1,
                    1, 1, 1,
                    0, 0, 3, 0,
                )
            xm = txl.stack_alloca("tensormap", 1)
            txl.cu_tensor_map_encode_tiled(
                xm, "bfloat16", 3, txl.handle_add_byte_offset(p["x"].data, 0),
                C, p["L"], p["B"],
                C * 2, i64(p["L"]) * (C * 2),
                W, R, 1,
                1, 1, 1,
                0, 0, 3, 0,
            )
            dm = txl.stack_alloca("tensormap", 1)
            txl.cu_tensor_map_encode_tiled(
                dm, "bfloat16", 3, txl.handle_add_byte_offset(p["dx"].data, 0),
                C, p["L"], p["B"],
                C * 2, i64(p["L"]) * (C * 2),
                W, T, 1,
                1, 1, 1,
                0, 0, 3, 0,
            )
            maps += [gm, xm, dm]
        return tuple(maps)

    @txl.kernel(warps=NCW + 1, grid=False, min_blocks_per_sm=CTAS_PER_SM, host_prelude=host_prelude)
    def conv_bwd_tma(
        g: txl.gptr[txl.bf16],
        x: txl.gptr[txl.bf16],
        w: txl.gptr[wdt],
        dx: txl.gptr[txl.bf16],
        dw: txl.gptr[wdt],
        L: txl.i32,
        S: txl.i32,
        offset: txl.i32,
        B: txl.i32,
        *,
        host,
    ):
        maps = [host[3 * i:3 * i + 3] for i in range(len(WS))]
        cta = txl.cta_id([G])
        rank = txl.Cast("int32", txl.cta_id_in_cluster([KF], preferred=[KF]))
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        ci = cta // KF
        if extra:
            wide = ci < extra
            c0 = 8 * (ci * base + txl.min(ci, txl.int32(extra)) * step)
            nch = txl.Select(wide, txl.int32(WA), txl.int32(WB))
        else:
            c0 = 8 * base * ci
            nch = txl.int32(WB)

        def by_width(fn):
            """
            fn(W, maps of width W) under the slice-width branch of this CTA.
            """
            if extra:
                with txl.If(wide):
                    with txl.Then():
                        fn(WA, maps[0])
                    with txl.Else():
                        fn(WB, maps[1])
            else:
                fn(WB, maps[0])

        nbs = (L + T - 1) // T
        nrt = nbs * B  # row tiles of this channel slice, u = bb * nbs + bs
        # TMA needs a 16B-aligned inner coordinate: (bs * T - offset) minus its 8-aligned floor
        dlt = (8 - (offset & 7)) & 7

        smem = txl.smem_pool()
        NDX = 2  # dx staging buffers
        dxs = smem.alloc([NDX * T * WMAX], "bfloat16", align=1024)
        gs = smem.alloc([NS * GSZ], "bfloat16", align=128)
        xs = smem.alloc([NS * XSZ], "bfloat16", align=128)
        red = smem.alloc([NTG * K * CB], "float32", align=16)
        red2 = smem.alloc([K * CB], "float32", align=16)
        full = smem.alloc([NS], "uint64", align=8)
        empty = smem.alloc([NS], "uint64", align=8)

        with txl.If(tid == 0), txl.Then():
            for s in range(NS):
                txl.ptx["mbarrier.init.shared.b64"](cvta(full.ptr_to([s])), txl.uint32(1))
                txl.ptx["mbarrier.init.shared.b64"](cvta(empty.ptr_to([s])), txl.uint32(EMPTY_COUNT))
            txl.ptx["fence.mbarrier_init.release.cluster"]()
            txl.ptx["fence.proxy.async.shared::cta"]()
        bar_named(0, (NCW + 1) * 32)

        stage = txl.local_scalar("int32", init=txl.int32(0))
        phase = txl.local_scalar("int32", init=txl.int32(0))
        u = txl.local_scalar("int32", init=rank)

        def advance():
            txl.assign(stage, stage + 1)
            with txl.If(stage == NS), txl.Then():
                txl.assign(stage, txl.int32(0))
                txl.assign(phase, phase ^ 1)
            txl.assign(u, u + KF)

        with txl.If(warp == NCW):
            with txl.Then():
                with txl.If(lane == 0), txl.Then():
                    for mp in maps:
                        for m in mp:
                            txl.ptx.prefetch.tensormap(txl.address_of(m))
                    lpol = load_policy()
                    with txl.While(u < nrt):
                        bs = u % nbs
                        bb = u // nbs
                        txl.cuda.mbarrier_wait(empty.ptr_to([stage]), phase ^ 1)
                        # order the compute warps' generic reads of this stage before the TMA rewrite
                        txl.ptx["fence.proxy.async.shared::cta"]()
                        fb = cvta(full.ptr_to([stage]))

                        def issue(W, mp):
                            gm, xm, _ = mp
                            txl.ptx["mbarrier.arrive.expect_tx.shared.b64"](fb, txl.uint32(txbytes(W)))
                            if cf:
                                # [W ch][CF_P tok] box from the 16B-aligned token below bs * T - offset:
                                # tile token j + dlt = g row bs * T - offset + j
                                gc = (bs * T - offset - dlt, c0, bb)
                            else:
                                gc = (c0, bs * T - offset, bb)
                            tma_load(cvta(gs.ptr_to([stage * GSZ])), tmap_ptr(gm), gc[0], gc[1], gc[2], fb, lpol)
                            tma_load(cvta(xs.ptr_to([stage * XSZ])), tmap_ptr(xm), c0, bs * T - (K - 1), bb, fb, lpol)

                        by_width(issue)
                        advance()
            with txl.Else():

                def store_dx(b, bs, bb):
                    with txl.If(tid == 0), txl.Then():
                        by_width(lambda W, mp: tma_store(
                            tmap_ptr(mp[2]), c0, bs * T, bb, cvta(dxs.ptr_to([b * T * WMAX]))))
                        txl.ptx["cp.async.bulk.commit_group"]()

                if cf:
                    (bwd_cf2_compute if CF2 else bwd_cf_compute)(gs, xs, dxs, red, w, wdt, C, K, c0, nch, dlt, tid, warp, lane, full, empty,
                                    stage, phase, u, nrt, nbs, NDX, GSZ, XSZ, WMAX, store_dx, advance)
                else:
                    bwd_cl_compute(gs, xs, dxs, red, w, wdt, C, K, c0, nch, tid, warp, lane, full, empty,
                                    stage, phase, u, nrt, nbs, NDX, GSZ, XSZ, WMAX, store_dx, advance)
                bar_named(1, NC)
                r = txl.local_scalar("float32")
                rv = txl.local_scalar("float32")
                for i in range(cdiv(K * CB, NC)):
                    idx = i * NC + tid
                    with txl.If(idx < K * CB), txl.Then():
                        txl.assign(r, txl.float32(0.0))
                        for q in range(NTG):
                            txl.ptx["ld.shared.f32"](rv, red.ptr_to([q * K * CB + idx]))
                            txl.assign(r, r + rv)
                        txl.ptx["st.shared.f32"](red2.ptr_to([idx]), r)
                with txl.If(tid == 0), txl.Then():
                    txl.ptx["cp.async.bulk.wait_group"](txl.int32(0))
        # the cluster's ranks in order, by rank 0
        cluster_sync()
        with txl.If((rank == 0) & (tid < NC)), txl.Then():
            r = txl.local_scalar("float32")
            rv = txl.local_scalar("float32")
            pa = txl.local_scalar("uint32")
            for i in range(cdiv(K * CB, NC)):
                idx = i * NC + tid
                k = idx // CB
                c = idx % CB
                with txl.If((idx < K * CB) & (c < nch)), txl.Then():
                    txl.assign(r, txl.float32(0.0))
                    for q in range(KF):
                        txl.ptx.mapa.shared__cluster.u32(pa, cvta(red2.ptr_to([idx])), txl.uint32(q))
                        txl.ptx["ld.shared::cluster.f32"](rv, pa)
                        txl.assign(r, r + rv)
                    store_w(dw, (c0 + c) * K + k, r, wdt)
        cluster_sync()

    return entry_function(conv_bwd_tma.compile())


def bwd_cl_compute(gs, xs, dxs, red, w, wdt, C, K, c0, nch, tid, warp, lane, full, empty, stage, phase,
                    u, nrt, nbs, NDX, GSZ, XSZ, WMAX, store_dx, advance):
    """
    Compute warps for channels-last g: lane = channel pair c2, warp = token group of TG2 tokens,
    packed f32x2 arithmetic. Tiles are [rows][nch] (pitch nch).
    """
    if CLM and wdt == "bfloat16":
        return bwd_clm_compute(gs, xs, dxs, red, w, C, K, c0, nch, tid, warp, lane, full, empty, stage, phase,
                                u, nrt, nbs, NDX, GSZ, XSZ, WMAX, store_dx, advance)
    c2 = 2 * lane
    tg = warp
    act = c2 < nch
    wk = txl.alloc_local([K], "uint64")
    wf = txl.alloc_local([2], "float32")
    gv = txl.alloc_local([TG2 + K - 1], "uint64")
    xv = txl.alloc_local([TG2 + K - 1], "uint64")
    out = txl.alloc_local([TG2], "uint64")
    pw = txl.alloc_local([K], "uint64")
    pacc = txl.alloc_local([K], "uint64")
    hv = txl.alloc_local([TG2], "uint32")
    wa = txl.local_scalar("uint32")
    wb = txl.local_scalar("uint32")
    load_w_pair(wk, wf, w, c0 + c2, C, K, wdt)
    for k in range(K):
        txl.assign(pacc[k], txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))
    it = txl.local_scalar("int32", init=txl.int32(0))
    with txl.While(u < nrt):
        bs = u % nbs
        bb = u // nbs
        txl.cuda.mbarrier_wait(full.ptr_to([stage]), phase)
        # lanes past the slice width read (and discard) the slice's last pair: no reads past the tile
        cl = txl.min(c2, nch - 2)
        gb = stage * GSZ + cl
        xb = stage * XSZ + cl
        for j in range(TG2 + K - 1):
            txl.ptx["ld.shared.b32"](wa, gs.ptr_to([gb + (tg * TG2 + j) * nch]))
            txl.assign(gv[j], bf16x2_to_f32x2(wa))
            txl.ptx["ld.shared.b32"](wb, xs.ptr_to([xb + (tg * TG2 + j) * nch]))
            txl.assign(xv[j], bf16x2_to_f32x2(wb))
        release(empty, stage, lane)
        taps2(out, wk, gv, K, True)
        partial_dw2(pw, gv, xv, K)
        for k in range(K):
            txl.ptx["add.rn.f32x2"](pacc[k], pacc[k], pw[k])
        b = it % NDX
        # thread 0 owns the bulk-store group: its read-completion wait overlaps the others' conversion
        with txl.If(tid == 0), txl.Then():
            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(NDX - 1))
        for i in range(TG2):
            txl.ptx["cvt.rn.bf16x2.f32"](hv[i], f2(out[i], 1), f2(out[i], 0))
        bar_named(1, NC)
        with txl.If(act), txl.Then():
            for i in range(TG2):
                txl.ptx["st.shared.b32"](dxs.ptr_to([b * T * WMAX + (tg * TG2 + i) * nch + c2]), hv[i])
        txl.ptx["fence.proxy.async.shared::cta"]()
        bar_named(1, NC)
        store_dx(b, bs, bb)
        txl.assign(it, it + 1)
        advance()
    # per (token group, tap, channel); summed over the groups in order by the caller
    for k in range(K):
        for e in range(2):
            txl.ptx["st.shared.f32"](red.ptr_to([(tg * K + k) * CB + c2 + e]), f2(pacc[k], e))



def bwd_clm_compute(gs, xs, dxs, red, w, C, K, c0, nch, tid, warp, lane, full, empty, stage, phase,
                     u, nrt, nbs, NDX, GSZ, XSZ, WMAX, store_dx, advance):
    """
    Packed channels-last layout (lane = channel pair, warp = TG2-token group) with scalar
    fma.rn.f32.bf16 on the staged BF16 halves: no BF16->FP32 conversions, same per-channel
    order.
    """
    c2 = 2 * lane
    tg = warp
    act = c2 < nch
    NR = TG2 + K - 1
    wk = [txl.alloc_local([K], "uint16") for _ in range(2)]
    gv = [txl.alloc_local([NR], "uint16") for _ in range(2)]
    xv = [txl.alloc_local([NR], "uint16") for _ in range(2)]
    out = [txl.alloc_local([TG2], "float32") for _ in range(2)]
    pw = [txl.alloc_local([K], "float32") for _ in range(2)]
    pacc = [txl.alloc_local([K], "float32") for _ in range(2)]
    hv = txl.alloc_local([TG2], "uint32")
    wa = txl.local_scalar("uint32")
    wb = txl.local_scalar("uint32")
    for e in range(2):
        for k in range(K):
            load_w_bf16(wk[e][k], w, (c0 + c2 + e) * K + k, c0 + c2 + e < C)
            txl.assign(pacc[e][k], txl.float32(0.0))
    it = txl.local_scalar("int32", init=txl.int32(0))
    with txl.While(u < nrt):
        bs = u % nbs
        bb = u // nbs
        txl.cuda.mbarrier_wait(full.ptr_to([stage]), phase)
        cl = txl.min(c2, nch - 2)
        gb = stage * GSZ + cl
        xb = stage * XSZ + cl
        for j in range(NR):
            txl.ptx["ld.shared.b32"](wa, gs.ptr_to([gb + (tg * TG2 + j) * nch]))
            txl.assign(gv[0][j], txl.Cast("uint16", wa))
            txl.assign(gv[1][j], txl.Cast("uint16", wa >> txl.uint32(16)))
            txl.ptx["ld.shared.b32"](wb, xs.ptr_to([xb + (tg * TG2 + j) * nch]))
            txl.assign(xv[0][j], txl.Cast("uint16", wb))
            txl.assign(xv[1][j], txl.Cast("uint16", wb >> txl.uint32(16)))
        release(empty, stage, lane)
        for e in range(2):
            for i in range(TG2):
                txl.assign(out[e][i], txl.float32(0.0))
                for k in range(K):
                    txl.ptx["fma.rn.f32.bf16"](out[e][i], wk[e][k], gv[e][i + K - 1 - k], out[e][i])
        for e in range(2):
            for k in range(K):
                txl.assign(pw[e][k], txl.float32(0.0))
                for i in range(TG2):
                    txl.ptx["fma.rn.f32.bf16"](pw[e][k], gv[e][i], xv[e][i + k], pw[e][k])
                txl.assign(pacc[e][k], pacc[e][k] + pw[e][k])
        b = it % NDX
        with txl.If(tid == 0), txl.Then():
            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(NDX - 1))
        for i in range(TG2):
            txl.ptx["cvt.rn.bf16x2.f32"](hv[i], out[1][i], out[0][i])
        bar_named(1, NC)
        with txl.If(act), txl.Then():
            for i in range(TG2):
                txl.ptx["st.shared.b32"](dxs.ptr_to([b * T * WMAX + (tg * TG2 + i) * nch + c2]), hv[i])
        txl.ptx["fence.proxy.async.shared::cta"]()
        bar_named(1, NC)
        store_dx(b, bs, bb)
        txl.assign(it, it + 1)
        advance()
    for k in range(K):
        for e in range(2):
            txl.ptx["st.shared.f32"](red.ptr_to([(tg * K + k) * CB + c2 + e]), pacc[e][k])

def bwd_cf_compute(gs, xs, dxs, red, w, wdt, C, K, c0, nch, dlt, tid, warp, lane, full, empty, stage, phase,
                    u, nrt, nbs, NDX, GSZ, XSZ, WMAX, store_dx, advance):
    """
    Compute warps for g in the output's channels-first layout, read in place (no transpose).

    Thread = channel ch, tokens [tq * TG, +TG) = dw token groups 2 tq and 2 tq + 1; g is read
    along tokens with 16B loads from the [nch][CF_P] tile (48 tokens from the aligned tq * TG,
    then shifted by the runtime dlt in registers: two word selects and a byte permute), x per
    token from the channels-last tile. Same per-channel arithmetic order as the packed
    channels-last path.
    """
    ch = lane + 32 * (warp % 2)
    tq = warp // 2
    act = ch < nch
    NW = 24  # g words per thread: tokens [tq * TG, +48)
    NV = cdiv(TG + K - 1, 2)  # words after the shift
    mixed = wdt == "bfloat16"
    wk = txl.alloc_local([K], "uint16" if mixed else "float32")
    gw = txl.alloc_local([NW], "uint32")
    ga = txl.alloc_local([NV + 2], "uint32")
    gq = txl.alloc_local([NV + 1], "uint32")
    s4 = (dlt & 4) != 0
    s2 = (dlt & 2) != 0
    sel = txl.Select((dlt & 1) != 0, txl.uint32(0x5432), txl.uint32(0x3210))
    gv = txl.alloc_local([TG + K - 1], "uint16" if mixed else "float32")
    xv = txl.alloc_local([TG + K - 1], "uint16" if mixed else "float32")
    out = txl.alloc_local([TG], "float32")
    pw = txl.alloc_local([K], "float32")
    pacc = txl.alloc_local([2 * K], "float32")
    hv = txl.alloc_local([TG], "uint16")
    for k in range(K):
        if mixed:
            load_w_bf16(wk[k], w, (c0 + ch) * K + k, c0 + ch < C)
        else:
            load_w_f32(wk[k], w, (c0 + ch) * K + k, wdt, c0 + ch < C)
    for i in range(2 * K):
        txl.assign(pacc[i], txl.float32(0.0))
    it = txl.local_scalar("int32", init=txl.int32(0))
    with txl.While(u < nrt):
        bs = u % nbs
        bb = u // nbs
        txl.cuda.mbarrier_wait(full.ptr_to([stage]), phase)
        # channel slots past the slice width read (and discard) the slice's last channel
        cr = txl.min(ch, nch - 1)
        gb = stage * GSZ + cr * CF_P + tq * TG
        xb = stage * XSZ + cr
        for m in range(NW // 4):
            txl.ptx["ld.shared.v4.b32"](gw[4 * m], gw[4 * m + 1], gw[4 * m + 2], gw[4 * m + 3],
                                        gs.ptr_to([gb + 8 * m]))
        for m in range(NV + 2):
            txl.assign(ga[m], txl.Select(s4, gw[m + 2], gw[m]))
        for m in range(NV + 1):
            txl.assign(gq[m], txl.Select(s2, ga[m + 1], ga[m]))
        for m in range(NV):
            txl.ptx["prmt.b32"](gw[m], gq[m], gq[m + 1], sel)
        for j in range(TG + K - 1):
            wv = gw[j // 2]
            if mixed:
                txl.assign(gv[j], txl.Cast("uint16", wv if j % 2 == 0 else wv >> txl.uint32(16)))
                txl.ptx["ld.shared.b16"](xv[j], xs.ptr_to([xb + (tq * TG + j) * nch]))
            else:
                if j % 2 == 0:
                    txl.assign(gv[j], txl.reinterpret("float32", wv << txl.uint32(16)))
                else:
                    txl.assign(gv[j], txl.reinterpret("float32", wv & txl.uint32(0xFFFF0000)))
                ld_s_bf16(xv[j], xs.ptr_to([xb + (tq * TG + j) * nch]))
        release(empty, stage, lane)
        if mixed:
            taps_dx_mixed(out, wk, gv, K)
        else:
            taps_dx(out, wk, gv, K)
        for h in range(2):
            if mixed:
                partial_dw_mixed(pw, gv, xv, K, h * TG2, TG2)
            else:
                partial_dw(pw, gv, xv, K, h * TG2, TG2)
            for k in range(K):
                txl.assign(pacc[h * K + k], pacc[h * K + k] + pw[k])
        b = it % NDX
        with txl.If(tid == 0), txl.Then():
            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(NDX - 1))
        for i in range(TG):
            txl.ptx["cvt.rn.bf16.f32"](hv[i], out[i])
        bar_named(1, NC)
        with txl.If(act), txl.Then():
            for i in range(TG):
                txl.ptx["st.shared.b16"](dxs.ptr_to([b * T * WMAX + (tq * TG + i) * nch + ch]), hv[i])
        txl.ptx["fence.proxy.async.shared::cta"]()
        bar_named(1, NC)
        store_dx(b, bs, bb)
        txl.assign(it, it + 1)
        advance()
    for h in range(2):
        for k in range(K):
            txl.ptx["st.shared.f32"](red.ptr_to([((2 * tq + h) * K + k) * CB + ch]), pacc[h * K + k])


CF2 = False  # scalar channel ownership uses native BF16-input FP32 FMAs on the timed BF16 path


def bwd_cf2_compute(gs, xs, dxs, red, w, wdt, C, K, c0, nch, dlt, tid, warp, lane, full, empty, stage, phase,
                     u, nrt, nbs, NDX, GSZ, XSZ, WMAX, store_dx, advance):
    """
    Channels-first g with the packed channels-last thread layout: lane = channel pair c2, warp =
    token group tg of TG2 tokens, f32x2 arithmetic. g is read along tokens from the [nch][CF_P]
    tile (32 tokens from the aligned tg * TG2 per channel, shifted by the runtime dlt in
    registers), x per token pair from the channels-last tile. Per-channel arithmetic order
    identical to every other path.
    """
    c2 = 2 * lane
    tg = warp
    act = c2 < nch
    NW = 16  # g words per channel: tokens [tg * TG2, +32)
    NR = TG2 + K - 1
    NV = cdiv(NR, 2)
    wk = txl.alloc_local([K], "uint64")
    wf = txl.alloc_local([2], "float32")
    gw = [txl.alloc_local([NW], "uint32") for _ in range(2)]
    ga = txl.alloc_local([NV + 2], "uint32")
    gq = txl.alloc_local([NV + 1], "uint32")
    s4 = (dlt & 4) != 0
    s2 = (dlt & 2) != 0
    sel = txl.Select((dlt & 1) != 0, txl.uint32(0x5432), txl.uint32(0x3210))
    gv = txl.alloc_local([NR], "uint64")
    xv = txl.alloc_local([NR], "uint64")
    out = txl.alloc_local([TG2], "uint64")
    pw = txl.alloc_local([K], "uint64")
    pacc = txl.alloc_local([K], "uint64")
    hv = txl.alloc_local([TG2], "uint32")
    wa = txl.local_scalar("uint32")
    load_w_pair(wk, wf, w, c0 + c2, C, K, wdt)
    for k in range(K):
        txl.assign(pacc[k], txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))
    it = txl.local_scalar("int32", init=txl.int32(0))
    with txl.While(u < nrt):
        bs = u % nbs
        bb = u // nbs
        txl.cuda.mbarrier_wait(full.ptr_to([stage]), phase)
        # lanes past the slice width read (and discard) the slice's last pair: no reads past the tile
        cl = txl.min(c2, nch - 2)
        for e in range(2):
            gb = stage * GSZ + (cl + e) * CF_P + tg * TG2
            for m in range(NW // 4):
                txl.ptx["ld.shared.v4.b32"](gw[e][4 * m], gw[e][4 * m + 1], gw[e][4 * m + 2], gw[e][4 * m + 3],
                                            gs.ptr_to([gb + 8 * m]))
            for m in range(NV + 2):
                txl.assign(ga[m], txl.Select(s4, gw[e][m + 2], gw[e][m]))
            for m in range(NV + 1):
                txl.assign(gq[m], txl.Select(s2, ga[m + 1], ga[m]))
            for m in range(NV):
                txl.ptx["prmt.b32"](gw[e][m], gq[m], gq[m + 1], sel)
        xb = stage * XSZ + cl
        for j in range(NR):
            g0 = gw[0][j // 2]
            g1 = gw[1][j // 2]
            if j % 2 == 0:
                lo = txl.reinterpret("float32", g0 << txl.uint32(16))
                hi = txl.reinterpret("float32", g1 << txl.uint32(16))
            else:
                lo = txl.reinterpret("float32", g0 & txl.uint32(0xFFFF0000))
                hi = txl.reinterpret("float32", g1 & txl.uint32(0xFFFF0000))
            txl.assign(gv[j], txl.cuda.make_float2(lo, hi))
            txl.ptx["ld.shared.b32"](wa, xs.ptr_to([xb + (tg * TG2 + j) * nch]))
            txl.assign(xv[j], bf16x2_to_f32x2(wa))
        release(empty, stage, lane)
        taps2(out, wk, gv, K, True)
        partial_dw2(pw, gv, xv, K)
        for k in range(K):
            txl.ptx["add.rn.f32x2"](pacc[k], pacc[k], pw[k])
        b = it % NDX
        with txl.If(tid == 0), txl.Then():
            txl.ptx["cp.async.bulk.wait_group.read"](txl.int32(NDX - 1))
        for i in range(TG2):
            txl.ptx["cvt.rn.bf16x2.f32"](hv[i], f2(out[i], 1), f2(out[i], 0))
        bar_named(1, NC)
        with txl.If(act), txl.Then():
            for i in range(TG2):
                txl.ptx["st.shared.b32"](dxs.ptr_to([b * T * WMAX + (tg * TG2 + i) * nch + c2]), hv[i])
        txl.ptx["fence.proxy.async.shared::cta"]()
        bar_named(1, NC)
        store_dx(b, bs, bb)
        txl.assign(it, it + 1)
        advance()
    for k in range(K):
        for e in range(2):
            txl.ptx["st.shared.f32"](red.ptr_to([(tg * K + k) * CB + c2 + e]), f2(pacc[k], e))


def store_w(dw, idx, r, wdt):
    if wdt == "float32":
        txl.ptx["st.global.f32"](dw.ptr_to([idx]), r)
    else:
        h = txl.local_scalar("uint16")
        txl.ptx["cvt.rn.bf16.f32" if wdt == "bfloat16" else "cvt.rn.f16.f32"](h, r)
        txl.ptx["st.global.b16"](dw.ptr_to([idx]), h)


# ------------------------------------------------------------------------------------------
# fallback (any C, S, strides): one tile per CTA, plain loads / stores, same arithmetic
# ------------------------------------------------------------------------------------------


def g2s_chunk(V, words, gptr_, sptr, valid):
    """
    Copy V contiguous bf16 from global to shared (zeros when not valid).
    """
    nw = max(V // 2, 1)
    dt = "uint32" if V >= 2 else "uint16"
    for i in range(nw):
        txl.assign(words[i], txl.Cast(dt, 0))
    with txl.If(valid), txl.Then():
        if V == 8:
            txl.ptx["ld.global.nc.v4.b32"](words[0], words[1], words[2], words[3], gptr_)
        elif V == 4:
            txl.ptx["ld.global.nc.v2.b32"](words[0], words[1], gptr_)
        elif V == 2:
            txl.ptx["ld.global.nc.b32"](words[0], gptr_)
        else:
            txl.ptx["ld.global.nc.b16"](words[0], gptr_)
    if V == 8:
        txl.ptx["st.shared.v4.b32"](sptr, words[0], words[1], words[2], words[3])
    elif V == 4:
        txl.ptx["st.shared.v2.b32"](sptr, words[0], words[1])
    elif V == 2:
        txl.ptx["st.shared.b32"](sptr, words[0])
    else:
        txl.ptx["st.shared.b16"](sptr, words[0])


def stage_rows(V, words, src, sbuf, tid, first_row, nrows_valid, R, C, c0, rowbase):
    """
    sbuf[j][c] = src[rowbase + (first_row + j) * C + c0 + c] for j < R, zero outside.
    """
    CPR = CB // V
    total = R * CPR
    for i in range(cdiv(total, NC)):
        idx = i * NC + tid
        row = idx // CPR
        chn = idx % CPR
        gr = first_row + row
        c = c0 + chn * V
        valid = (gr >= 0) & (gr < nrows_valid) & (c < C) & (idx < total)
        with txl.If(idx < total), txl.Then():
            g2s_chunk(V, words, src.ptr_to([rowbase + i64(gr * C + c)]), sbuf.ptr_to([row * CB + chn * V]), valid)


@functools.cache
def fwd_plain(C, K, wdt):
    R = T + K - 1
    NCB = cdiv(C, CB)
    V = vec_width(C)
    YP = T + 2

    @txl.kernel(warps=NCW, grid=lambda p: [(p["S"] + T - 1) // T, NCB, p["B"]])
    def conv_fwd_plain(
        x: txl.gptr[txl.bf16],
        w: txl.gptr[wdt],
        y: txl.gptr[txl.bf16],
        L: txl.i32,
        S: txl.i32,
        offset: txl.i32,
        B: txl.i32,
    ):
        bt, bc, bb = txl.cta_id()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        t0 = bt * T
        c0 = bc * CB
        smem = txl.smem_pool()
        xs = smem.alloc([R * CB], "bfloat16", align=16)
        ys = smem.alloc([CB * YP], "bfloat16", align=16)
        words = txl.alloc_local([4], "uint32" if V >= 2 else "uint16")
        stage_rows(V, words, x, xs, tid, t0 + offset - (K - 1), L, R, C, c0, i64(bb) * i64(L) * C)
        ch = lane + 32 * (warp % 2)
        tg = warp // 2
        wk = txl.alloc_local([K], "float32")
        for k in range(K):
            load_w_f32(wk[k], w, (c0 + ch) * K + k, wdt, c0 + ch < C)
        bar_named(0, NC)
        xv = txl.alloc_local([TG + K - 1], "float32")
        for j in range(TG + K - 1):
            ld_s_bf16(xv[j], xs.ptr_to([(tg * TG + j) * CB + ch]))
        out = txl.alloc_local([TG], "float32")
        taps_fwd(out, wk, xv, K)
        h = txl.local_scalar("uint16")
        for j in range(TG):
            txl.ptx["cvt.rn.bf16.f32"](h, out[j])
            txl.ptx["st.shared.b16"](ys.ptr_to([ch * YP + tg * TG + j]), h)
        bar_named(0, NC)
        ybase = i64(bb) * C * i64(S)
        for i in range(cdiv(CB * T, NC)):
            idx = i * NC + tid
            c = idx // T
            t = idx % T
            with txl.If((c0 + c < C) & (t0 + t < S)), txl.Then():
                txl.ptx["ld.shared.b16"](h, ys.ptr_to([c * YP + t]))
                txl.ptx["st.global.b16"](y.ptr_to([ybase + i64((c0 + c) * S + t0 + t)]), h)

    return entry_function(conv_fwd_plain.compile())


@functools.cache
def bwd_plain(C, K, wdt, mode):
    """
    mode: "cl" (g contiguous) or "gen" (runtime strides). Per-thread dw partials go to
    dwp[u][tg][k][c] (u = bb * nbs + bs) for dw_kernel, which sums them in the TMA order.
    """
    R = T + K - 1
    NCB = cdiv(C, CB)
    V = vec_width(C)
    GP = CB + 2 if mode == "gen" else CB

    @txl.kernel(warps=NCW, grid=lambda p: [(p["L"] + T - 1) // T, NCB, p["B"]])
    def conv_bwd_plain(
        g: txl.gptr[txl.bf16],
        x: txl.gptr[txl.bf16],
        w: txl.gptr[wdt],
        dx: txl.gptr[txl.bf16],
        dwp: txl.gptr[txl.f32],
        L: txl.i32,
        S: txl.i32,
        offset: txl.i32,
        B: txl.i32,
        gsb: txl.i64,
        gst: txl.i64,
        gsc: txl.i64,
    ):
        bs, bc, bb = txl.cta_id()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        s0 = bs * T
        c0 = bc * CB
        nbs = (L + T - 1) // T
        smem = txl.smem_pool()
        gs = smem.alloc([R * GP], "bfloat16", align=16)
        xs = smem.alloc([R * CB], "bfloat16", align=16)
        words = txl.alloc_local([4], "uint32" if V >= 2 else "uint16")
        xbase = i64(bb) * i64(L) * C
        stage_rows(V, words, x, xs, tid, s0 - (K - 1), L, R, C, c0, xbase)
        if mode == "cl":
            stage_rows(V, words, g, gs, tid, s0 - offset, S, R, C, c0, i64(bb) * gsb)
        else:
            h = txl.local_scalar("uint16")
            for i in range(cdiv(R * CB, NC)):
                idx = i * NC + tid
                row = idx // CB
                c = idx % CB
                t = s0 - offset + row
                txl.assign(h, txl.uint16(0))
                with txl.If((t >= 0) & (t < S) & (c0 + c < C) & (idx < R * CB)), txl.Then():
                    txl.ptx["ld.global.nc.b16"](h, g.ptr_to([i64(bb) * gsb + i64(t) * gst + i64(c0 + c) * gsc]))
                with txl.If(idx < R * CB), txl.Then():
                    txl.ptx["st.shared.b16"](gs.ptr_to([row * GP + c]), h)
        ch = lane + 32 * (warp % 2)
        tg = warp // 2
        wk = txl.alloc_local([K], "float32")
        for k in range(K):
            load_w_f32(wk[k], w, (c0 + ch) * K + k, wdt, c0 + ch < C)
        bar_named(0, NC)
        gv = txl.alloc_local([TG + K - 1], "float32")
        xv = txl.alloc_local([TG + K - 1], "float32")
        for j in range(TG + K - 1):
            ld_s_bf16(gv[j], gs.ptr_to([(tg * TG + j) * GP + ch]))
            ld_s_bf16(xv[j], xs.ptr_to([(tg * TG + j) * CB + ch]))
        out = txl.alloc_local([TG], "float32")
        taps_dx(out, wk, gv, K)
        pw = txl.alloc_local([2 * K], "float32")
        # dw partials per TG2-token group (the packed TMA kernels' grouping)
        for hf in range(2):
            partial_dw(Sub(pw, hf * K), gv, xv, K, hf * TG2, TG2)
        h2 = txl.local_scalar("uint16")
        for i in range(TG):
            s = s0 + tg * TG + i
            with txl.If((s < L) & (c0 + ch < C)), txl.Then():
                txl.ptx["cvt.rn.bf16.f32"](h2, out[i])
                txl.ptx["st.global.b16"](dx.ptr_to([xbase + i64(s * C + c0 + ch)]), h2)
        u = bb * nbs + bs
        with txl.If(c0 + ch < C), txl.Then():
            for hf in range(2):
                for k in range(K):
                    txl.ptx["st.global.f32"](
                        dwp.ptr_to([((i64(u) * NTG + i64(2 * tg + hf)) * K + k) * C + i64(c0 + ch)]),
                        pw[hf * K + k],
                    )

    return entry_function(conv_bwd_plain.compile())


@functools.cache
def dw_kernel(C, K, wdt):
    """
    dw[c][k] in the TMA kernel's order: ranks q < KF, token groups, row tiles u = q mod KF.
    """
    N = C * K
    KF = kfac(cdiv(C, CB))

    @txl.kernel(warps=8, grid=lambda p: (N + 255) // 256)
    def dw_reduce(dwp: txl.gptr[txl.f32], dw: txl.gptr[wdt], nrt: txl.i32):
        idx = txl.cta_id() * 256 + txl.thread_id()
        k = idx // C
        c = idx % C
        out = txl.local_scalar("float32")
        r = txl.local_scalar("float32")
        a = txl.local_scalar("float32")
        rv = txl.local_scalar("float32")
        uu = txl.local_scalar("int32")
        with txl.If(idx < N), txl.Then():
            txl.assign(out, txl.float32(0.0))
            for q in range(KF):
                txl.assign(r, txl.float32(0.0))
                for tg in range(NTG):
                    txl.assign(a, txl.float32(0.0))
                    txl.assign(uu, txl.int32(q))
                    with txl.While(uu < nrt):
                        txl.ptx["ld.global.f32"](
                            rv, dwp.ptr_to([((i64(uu) * NTG + tg) * K + i64(k)) * C + i64(c)])
                        )
                        txl.assign(a, a + rv)
                        txl.assign(uu, uu + KF)
                    txl.assign(r, r + a)
                txl.assign(out, out + r)
            store_w(dw, c * K + k, out, wdt)

    return entry_function(dw_reduce.compile())


# ------------------------------------------------------------------------------------------
# entry points
# ------------------------------------------------------------------------------------------


def aligned(*ts):
    return all(t.data_ptr() % 16 == 0 for t in ts)


def covered_weight(w):
    """
    w flattened in a dtype the kernels read: other float dtypes are read as fp32, as the Triton path
    does.
    """
    wf = w.reshape(-1)
    return wf if w.dtype in WDT else wf.float()


def causal_conv1d_fwd_sm100(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    """
    The forward of the picker's causal_conv1d_fwd: a fresh bf16 (B, S, C) in the layout of
    causal_conv1d_empty, bit-identical to ATen's depthwise conv1d.
    """
    x = x.contiguous()
    B, L, C = x.shape
    K = w.shape[-1]
    S = L - offset
    y = torch.empty((B, C, S), dtype=x.dtype, device=x.device)
    if S > 0 and B > 0 and C > 0:
        wf = covered_weight(w)
        wdt = WDT[wf.dtype]
        if C % 8 == 0 and S % 8 == 0 and K <= 8 and aligned(x, y):
            fwd_tma(C, K, wdt, S >= 8192)(x.view(-1), wf, y.view(-1), L, S, offset, B)
        else:
            fwd_plain(C, K, wdt)(x.view(-1), wf, y.view(-1), L, S, offset, B)
    return y.transpose(1, 2)


def causal_conv1d_bwd_sm100(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The backward of the picker's causal_conv1d_bwd: dx bit-identical to ATen's depthwise conv
    backward, dw in one fixed fp32 order whatever g's strides.
    """
    x = x.contiguous()
    B, L, C = x.shape
    K = w.shape[-1]
    S = L - offset
    dx = torch.empty_like(x)
    if C == 0:
        return dx.zero_(), torch.zeros(w.shape, dtype=w.dtype, device=w.device)
    wf = covered_weight(w)
    wdt = WDT[wf.dtype]
    dwf = torch.empty((C * K,), dtype=wf.dtype, device=w.device)
    if g.numel() == 0:
        g = g.contiguous()  # nothing to read: the plain path reads it as channels-last
    nrt = B * cdiv(L, T)
    cl = g.is_contiguous()
    cf = (not cl) and S > 0 and g.stride(1) == 1 and g.stride(2) == S and g.stride(0) == C * S
    tma_ok = C % 8 == 0 and K <= 8 and S > 0 and nrt > 0 and aligned(g, x, dx)
    if tma_ok and (cl or (cf and S % 8 == 0)):
        gv = g.view(-1) if cl else g.as_strided((B * C * S,), (1,))
        bwd_tma(C, K, wdt, not cl)(gv, x.view(-1), wf, dx.view(-1), dwf, L, S, offset, B)
    else:
        dwp = torch.empty((max(nrt, 1), NTG, K, C), dtype=torch.float32, device=x.device)
        if nrt > 0:
            if cl:
                bwd_plain(C, K, wdt, "cl")(g.view(-1), x.view(-1), wf, dx.view(-1), dwp.view(-1), L, S, offset, B, S * C, C, 1)
            else:
                sb, st, sc = g.stride()
                span = 1 + (B - 1) * sb + max(S - 1, 0) * st + (C - 1) * sc
                bwd_plain(C, K, wdt, "gen")(g.as_strided((span,), (1,)), x.view(-1), wf, dx.view(-1), dwp.view(-1), L, S, offset, B, sb, st, sc)
        dw_kernel(C, K, wdt)(dwp.view(-1), dwf, nrt)
    return dx, dwf.to(w.dtype).view(w.shape)
