"""
segment_sum_sm100: per-group column sums of rows sorted by group, in TIRx-lite (B200).

Two kernels:
1. tile_sum: one CTA per (tile slot, column chunk). Groups are cut into 128-row tiles from their own
   start; tile j of group g lives at slot f(g) + j with f(g) = g + start_g // 128, which is strictly
   increasing and leaves no two tiles on one slot, so a slot is found from the offsets alone (no
   scan). The CTA writes the tile's fp32 column sums (rows in a fixed per-tile order) to partials.
2. tile_reduce: one CTA per (group, column chunk) adds the group's tile sums in row order starting
   from zero and rounds once to out_dtype.

Evolved with TIRx-harness: run segment_sum, frontier member tma2d-ring (commit e356c72, 207 us vs 647 us, 3.121x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools
import math

import torch
import tvm

import tirx_kernels.tirx_lite as txl

__all__ = ["segment_sum_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


TILE = 128
DT = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
IT = {torch.int32: "int32", torch.int64: "int64"}


def cdiv(a, b):
    return (a + b - 1) // b


def vec2(dtype, H):
    esz = 4 if dtype == "float32" else 2
    return math.gcd(H, 16 // esz), esz


def load_vec(x, idx, dtype, vec, esz, dst, pred):
    """
    dst[0:vec] = fp32 of x[idx : idx + vec] if pred else 0.
    """
    nbytes = vec * esz
    nw = max(1, nbytes // 4)
    if nbytes >= 4:
        w = txl.alloc_local([nw], "uint32")
        for i in range(nw):
            txl.assign(w[i], txl.uint32(0))
        with txl.If(pred), txl.Then():
            if nw == 4:
                txl.ptx.ld.global_.v4.b32(w[0], w[1], w[2], w[3], x.ptr_to([idx]))
            elif nw == 2:
                txl.ptx.ld.global_.v2.b32(w[0], w[1], x.ptr_to([idx]))
            else:
                txl.ptx.ld.global_.b32(w[0], x.ptr_to([idx]))
        for i in range(nw):
            if dtype == "float32":
                txl.assign(dst[i], txl.reinterpret("float32", w[i]))
            elif dtype == "bfloat16":
                txl.assign(dst[2 * i], txl.reinterpret("float32", w[i] << txl.uint32(16)))
                txl.assign(dst[2 * i + 1], txl.reinterpret("float32", w[i] & txl.uint32(0xFFFF0000)))
            else:
                txl.idioms.cast_f16x2_to_f32x2(dst, i, w[i])
    else:  # one 16-bit element
        h = txl.alloc_local([1], "uint16")
        txl.assign(h[0], txl.uint16(0))
        with txl.If(pred), txl.Then():
            txl.ptx.ld.global_.b16(h[0], x.ptr_to([idx]))
        if dtype == "bfloat16":
            txl.assign(dst[0], txl.reinterpret("float32", txl.cast(h[0], "uint32") << txl.uint32(16)))
        else:
            txl.ptx.cvt.f32.f16(dst[0], h[0])


def store_vec(out, idx, dtype, vec, src):
    """
    out[idx : idx + vec] = src[0:vec] rounded to dtype.
    """
    if dtype == "float32":
        w = [txl.reinterpret("uint32", src[i]) for i in range(vec)]
    else:
        if vec == 1:
            h = txl.alloc_local([1], "uint16")
            if dtype == "bfloat16":
                txl.ptx.cvt.rn.bf16.f32(h[0], src[0])
            else:
                txl.ptx.cvt.rn.f16.f32(h[0], src[0])
            txl.ptx.st.global_.b16(out.ptr_to([idx]), h[0])
            return
        wl = txl.alloc_local([vec // 2], "uint32")
        for i in range(vec // 2):
            if dtype == "bfloat16":
                txl.ptx.cvt.rn.bf16x2.f32(wl[i], src[2 * i + 1], src[2 * i])
            else:
                txl.ptx.cvt.rn.f16x2.f32(wl[i], src[2 * i + 1], src[2 * i])
        w = [wl[i] for i in range(vec // 2)]
    n = len(w)
    while n > 0:
        k = 4 if n >= 4 else (2 if n >= 2 else 1)
        base = len(w) - n
        off = base * (vec // len(w))
        p = out.ptr_to([idx + off])
        if k == 4:
            txl.ptx.st.global_.v4.b32(p, w[base], w[base + 1], w[base + 2], w[base + 3])
        elif k == 2:
            txl.ptx.st.global_.v2.b32(p, w[base], w[base + 1])
        else:
            txl.ptx.st.global_.b32(p, w[base])
        n -= k


def load_off(offs, g, idtype):
    """
    offs[g] as int32 (one global load).
    """
    v = txl.alloc_local([1], idtype)
    if idtype == "int64":
        txl.ptx.ld.global_.s64(v[0], offs.ptr_to([g]))
    else:
        txl.ptx.ld.global_.s32(v[0], offs.ptr_to([g]))
    return txl.cast(v[0], "int32") if idtype == "int64" else v[0]


def bounds(offs, g, M, idtype):
    """
    (start, end) of group g, ends clamped to M, start 0 for g = 0, end >= start.
    """
    e = txl.min(load_off(offs, g, idtype), M)
    s = txl.Select(g > 0, txl.min(load_off(offs, txl.max(g - 1, 0), idtype), M), txl.int32(0))
    return s, txl.max(e, s)


def layout(H, vec):
    NV = H // vec
    NC = cdiv(NV, 128)
    CV = cdiv(NV, NC)
    TW = cdiv(CV, 32) * 32
    return NV, NC, CV, TW


@functools.cache
def tile_sum2(dtype, idtype, H):
    vec, esz = vec2(dtype, H)
    NV, NC, CV, TW = layout(H, vec)
    RG = max(1, 256 // TW)
    while TILE % RG:
        RG -= 1
    RPG = TILE // RG
    U = min(RPG, 16 if vec * esz >= 8 else 32)
    NLOOP = RPG // U
    NT = TW * RG

    @txl.kernel(warps=NT // 32, grid="n_units")
    def tile_sum(
        x: txl.gptr[dtype],
        offs: txl.gptr[idtype],
        part: txl.gptr[txl.f32],
        M: txl.i32,
        G: txl.i32,
        n_units: txl.i32,
    ):
        u = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        txl.ptx.griddepcontrol.launch_dependents()
        sigma = u // NC
        c = u % NC
        smem = txl.smem_pool()
        red = smem.alloc([max(RG - 1, 1) * TW * vec], "float32")

        # find the last group with f(g) <= sigma (f is strictly increasing, f(0) = 0)
        cnt = txl.local_scalar("int32")
        txl.assign(cnt, txl.int32(0))
        with txl.serial((G + 31) // 32) as it:
            g = it * 32 + lane
            gs = txl.Select(g < G, g, txl.int32(0))
            s, _ = bounds(offs, gs, M, idtype)
            ok = txl.And(g < G, g + s // TILE <= sigma)
            bal = txl.alloc_local([1], "uint32")
            txl.ptx.vote_sync.ballot.b32(bal[0], ok, txl.uint32(0xFFFFFFFF))
            txl.assign(cnt, cnt + txl.cast(txl.popcount(bal[0]), "int32"))
        grp = cnt - 1
        s, e = bounds(offs, grp, M, idtype)
        r0 = s + (sigma - grp - s // TILE) * TILE
        nvalid = e - r0
        with txl.If(nvalid > 0), txl.Then():
            rg = tid // TW
            lc = tid % TW
            v = c * CV + lc
            act = txl.And(lc < CV, v < NV)
            acc = txl.alloc_local([vec], "float32")
            for k in range(vec):
                txl.assign(acc[k], txl.float32(0))
            vals = txl.alloc_local([U * vec], "float32")
            with txl.serial(NLOOP) as li:
                rbase = rg * RPG + li * U
                for i in range(U):
                    row = rbase + i
                    idx = txl.cast(r0 + row, "int64") * H + txl.cast(v * vec, "int64")
                    load_vec(x, idx, dtype, vec, esz, View(vals, i * vec), txl.And(act, row < nvalid))
                for i in range(U):
                    for k in range(vec):
                        txl.assign(acc[k], acc[k] + vals[i * vec + k])
            if RG > 1:
                with txl.If(rg > 0), txl.Then():
                    for k in range(vec):
                        txl.ptx.st.shared.f32(red.ptr_to([((rg - 1) * TW + lc) * vec + k]), acc[k])
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NT))
            with txl.If(txl.And(rg == 0, act)), txl.Then():
                tmp = txl.alloc_local([1], "float32")
                for r in range(1, RG):
                    for k in range(vec):
                        txl.ptx.ld.shared.f32(tmp[0], red.ptr_to([((r - 1) * TW + lc) * vec + k]))
                        txl.assign(acc[k], acc[k] + tmp[0])
                store_vec(part, txl.cast(sigma, "int64") * H + txl.cast(v * vec, "int64"), "float32", vec, acc)

    return tile_sum


class View:
    """
    dst[i] -> buf[base + i].
    """

    def __init__(self, buf, base):
        self.buf, self.base = buf, base

    def __getitem__(self, i):
        return self.buf[self.base + i]


SR = 32  # rows per ring stage (one bulk copy per producer lane)
NWC = 8  # consumer warps
BULK_G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
EVICT_FIRST = 0x12F0000000000000
SMEM_BUDGET = 100000
LONG_TILES = 16  # average tiles per group from which the chunked reduce is used
CPS = 1  # persistent CTAs per SM
CONSUME = 1  # diagnosis knob: 0 skips the consumer smem reads (pure TMA streaming)
CVT = 48  # target vectors per column chunk


TMA_G2S = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
BOXMAX = 256  # TMA box extent limit (elements)


def ring_layout(dtype, H):
    """
    None unless rows are 16-byte aligned (TMA strides); else (vec, NV, NC, CV, NS).

    Column chunks are TMA boxes of BC = CV * vec columns: the widest divisor of H that is a
    multiple of vec and at most 256, else 256 with the tail chunk's box running past H (zero fill).
    """
    vec, esz = vec2(dtype, H)
    if vec * esz != 16:
        return None
    NV = H // vec
    if H <= BOXMAX:
        BC = H
    else:
        BC = next((d for d in range(BOXMAX, 63, -vec) if H % d == 0), BOXMAX)
    CV = BC // vec
    NC = cdiv(H, BC)
    NS = max(2, min(16, SMEM_BUDGET // (SR * BC * esz + 8)))
    return vec, NV, NC, CV, NS


def search(offs, sigma, M, G, idtype, lane):
    """
    (group, first row, valid rows) of tile slot sigma, by a warp ballot over f(g).
    """
    cnt = txl.local_scalar("int32")
    txl.assign(cnt, txl.int32(0))
    with txl.serial((G + 31) // 32) as it:
        g = it * 32 + lane
        gs = txl.Select(g < G, g, txl.int32(0))
        s, _ = bounds(offs, gs, M, idtype)
        ok = txl.And(g < G, g + s // TILE <= sigma)
        bal = txl.alloc_local([1], "uint32")
        txl.ptx.vote_sync.ballot.b32(bal[0], ok, txl.uint32(0xFFFFFFFF))
        txl.assign(cnt, cnt + txl.cast(txl.popcount(bal[0]), "int32"))
    grp = cnt - 1
    s, e = bounds(offs, grp, M, idtype)
    r0 = s + (sigma - grp - s // TILE) * TILE
    return r0, txl.min(e - r0, txl.int32(TILE))


GS = 2048  # groups whose clamped ends are staged in shared memory (more: global ballot search)


def search_smem(ends, sigma):
    """
    _search over the staged ends: binary search for the last g with f(g) <= sigma.
    """
    lo = txl.local_scalar("int32")
    hi = txl.local_scalar("int32")
    mid = txl.local_scalar("int32")
    sv = txl.local_scalar("int32")
    txl.assign(lo, txl.int32(0))
    txl.assign(hi, txl.int32(GS))  # caller clamps to G - 1 below
    return lo, hi, mid, sv


def start_of(ends, g, sv):
    with txl.If(g > 0):
        with txl.Then():
            txl.ptx.ld.shared.s32(sv, ends.ptr_to([txl.max(g - 1, 0)]))
        with txl.Else():
            txl.assign(sv, txl.int32(0))
    return sv


@functools.cache
def tile_sum_ring2(dtype, idtype, H):
    vec, NV, NC, CV, NS = ring_layout(dtype, H)
    RG = (NWC * 32) // CV
    RB = CV * 16  # stage row stride in bytes
    KPU = TILE // SR  # stages per unit
    NPR = cdiv(SR, RG)  # rows per consumer thread per stage (max)
    NT = (NWC + 1) * 32

    esz = 16 // vec

    def host_prelude(params):
        xmap = txl.stack_alloca("tensormap", 1)
        txl.call_packed(
            "runtime.cuTensorMapEncodeTiled", xmap, dtype, 2, params["x"].data,
            H, params["M"], H * esz, CV * vec, SR, 1, 1, 0, 0, 3, 0,
        )
        return xmap

    @txl.kernel(warps=NWC + 1, grid="n_ctas", host_prelude=host_prelude)
    def tile_sum_ring(
        x: txl.gptr[dtype],
        offs: txl.gptr[idtype],
        part: txl.gptr[txl.f32],
        M: txl.i32,
        G: txl.i32,
        n_units: txl.i32,
        n_ctas: txl.i32,
        *,
        host,
    ):
        xmap = host
        cta = txl.cta_id()
        tid = txl.thread_id()
        warp = tid >> 5
        lane = tid & 31
        txl.ptx.griddepcontrol.launch_dependents()
        smem = txl.smem_pool()
        full = smem.alloc([NS], "uint64", align=8)
        empty = smem.alloc([NS], "uint64", align=8)
        meta = smem.alloc([NS], "int32", align=4)
        ends = smem.alloc([GS], "int32", align=4)
        red = smem.alloc([2 * max(RG - 1, 1) * CV * vec], "float32", align=16)
        ring = smem.alloc([NS * SR * RB // 4], "uint32", align=128)
        with txl.If(tid == 0), txl.Then():
            for i in range(NS):
                txl.ptx.mbarrier.init.shared.b64(full.ptr_to([i]), txl.uint32(1))
                txl.ptx.mbarrier.init.shared.b64(empty.ptr_to([i]), txl.uint32(NWC))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.If(G <= GS), txl.Then():
            with txl.serial((G + NT - 1) // NT) as gi:
                g = gi * NT + tid
                with txl.If(g < G), txl.Then():
                    txl.ptx.st.shared.s32(ends.ptr_to([g]), txl.min(load_off(offs, g, idtype), M))
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NT))
        n_mine = (n_units - cta + n_ctas - 1) // n_ctas

        with txl.If(warp == NWC):
            with txl.Then():  # producer warp
                with txl.serial(n_mine) as it:
                    u = cta + it * n_ctas
                    sigma = u // NC
                    c = u % NC
                    r0 = txl.local_scalar("int32")
                    nvalid = txl.local_scalar("int32")
                    with txl.If(G <= GS):
                        with txl.Then():
                            lo, hi, mid, sv = search_smem(ends, sigma)
                            txl.assign(hi, G - 1)
                            with txl.While(lo < hi):
                                txl.assign(mid, (lo + hi + 1) >> 1)
                                st = start_of(ends, mid, sv)
                                with txl.If(mid + st // TILE <= sigma):
                                    with txl.Then():
                                        txl.assign(lo, mid)
                                    with txl.Else():
                                        txl.assign(hi, mid - 1)
                            st = start_of(ends, lo, sv)
                            en = txl.local_scalar("int32")
                            txl.ptx.ld.shared.s32(en, ends.ptr_to([lo]))
                            txl.assign(r0, st + (sigma - lo - st // TILE) * TILE)
                            txl.assign(nvalid, txl.min(txl.max(en, st) - r0, txl.int32(TILE)))
                        with txl.Else():
                            a, b = search(offs, sigma, M, G, idtype, lane)
                            txl.assign(r0, a)
                            txl.assign(nvalid, b)
                    for k in range(KPU):
                        q = it * KPU + k
                        s = q % NS
                        ph = (q // NS) & 1
                        txl.cuda.mbarrier_wait(empty.ptr_to([s]), ph ^ 1)
                        with txl.If(lane == 0), txl.Then():
                            txl.ptx.st.shared.s32(meta.ptr_to([s]), nvalid)
                            with txl.If(nvalid > k * SR):
                                with txl.Then():
                                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full.ptr_to([s]), txl.uint32(SR * CV * 16))
                                    txl.ptx[TMA_G2S](
                                        ring.ptr_to([s * SR * (RB // 4)]),
                                        txl.address_of(xmap),
                                        c * (CV * vec),
                                        r0 + k * SR,
                                        txl.cuda.cvta_generic_to_shared(full.ptr_to([s])),
                                        txl.uint64(EVICT_FIRST),
                                    )
                                with txl.Else():
                                    txl.ptx.mbarrier.arrive.shared.b64(full.ptr_to([s]), txl.uint32(1))
            with txl.Else():  # consumer warps
                lc = tid % CV
                rg = tid // CV
                act = rg < RG
                acc = txl.alloc_local([vec], "float32")
                vals = txl.alloc_local([NPR * vec], "float32")
                w = txl.alloc_local([NPR * 4], "uint32")
                nv = txl.local_scalar("int32")
                with txl.serial(n_mine) as it:
                    u = cta + it * n_ctas
                    sigma = u // NC
                    c = u % NC
                    cv_c = txl.min(txl.int32(CV), NV - c * CV)
                    for kk in range(vec):
                        txl.assign(acc[kk], txl.float32(0))
                    for k in range(KPU):
                        q = it * KPU + k
                        s = q % NS
                        ph = (q // NS) & 1
                        txl.cuda.mbarrier_wait(full.ptr_to([s]), ph)
                        txl.ptx.ld.shared.s32(nv, meta.ptr_to([s]))
                        # rows i of this stage with (k * SR + i) % RG == rg, in increasing order
                        i0 = (rg + RG - (k * SR) % RG) % RG
                        for j in range(NPR):
                            i = i0 + j * RG
                            okr = txl.And(txl.And(act, lc < cv_c), txl.And(i < SR, k * SR + i < nv * CONSUME))
                            for kk in range(4):
                                txl.assign(w[j * 4 + kk], txl.uint32(0))
                            with txl.If(okr), txl.Then():
                                txl.ptx.ld.shared.v4.b32(w[j * 4], w[j * 4 + 1], w[j * 4 + 2], w[j * 4 + 3], ring.ptr_to([(s * SR + i) * (RB // 4) + lc * 4]))
                        txl.cuda.warp_sync()
                        with txl.If(lane == 0), txl.Then():
                            txl.ptx.mbarrier.arrive.shared.b64(empty.ptr_to([s]), txl.uint32(1))
                        for j in range(NPR):
                            for kk in range(4):
                                wd = w[j * 4 + kk]
                                if dtype == "float32":
                                    txl.assign(vals[j * vec + kk], txl.reinterpret("float32", wd))
                                elif dtype == "bfloat16":
                                    txl.assign(vals[j * vec + 2 * kk], txl.reinterpret("float32", wd << txl.uint32(16)))
                                    txl.assign(vals[j * vec + 2 * kk + 1], txl.reinterpret("float32", wd & txl.uint32(0xFFFF0000)))
                                else:
                                    txl.idioms.cast_f16x2_to_f32x2(View(vals, j * vec), kk, wd)
                            for kk in range(vec):
                                txl.assign(acc[kk], acc[kk] + vals[j * vec + kk])
                    rb = (it & 1) * (max(RG - 1, 1) * CV * vec)
                    if RG > 1:
                        with txl.If(txl.And(rg > 0, act)), txl.Then():
                            for kk in range(vec):
                                txl.ptx.st.shared.f32(red.ptr_to([rb + ((rg - 1) * CV + lc) * vec + kk]), acc[kk])
                        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NWC * 32))
                    with txl.If(txl.And(txl.And(rg == 0, lc < cv_c), nv > 0)), txl.Then():
                        tmp = txl.alloc_local([1], "float32")
                        for r in range(1, RG):
                            for kk in range(vec):
                                txl.ptx.ld.shared.f32(tmp[0], red.ptr_to([rb + ((r - 1) * CV + lc) * vec + kk]))
                                txl.assign(acc[kk], acc[kk] + tmp[0])
                        store_vec(part, txl.cast(sigma, "int64") * H + txl.cast((c * CV + lc) * vec, "int64"), "float32", vec, acc)

    return tile_sum_ring


@functools.cache
def sm_count():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


RCOLS, RLANES, RU = 32, 8, 8  # reduce: columns per CTA, tile lanes, tiles per lane per chunk


@functools.cache
def tile_reduce2(idtype, odtype, H):
    """
    out[g] = (((0 + t0) + t1) + ...) over group g's tile sums: per CTA 32 columns; each chunk of
    RLANES * RU tiles is loaded by all 8 warps at once into smem, then warp 0 adds it in row order.
    """
    NCB = cdiv(H, RCOLS)
    TB = RLANES * RU

    @txl.kernel(warps=RLANES, grid=lambda p: p["G"] * NCB)
    def tile_reduce(
        part: txl.gptr[txl.f32],
        offs: txl.gptr[idtype],
        out: txl.gptr[odtype],
        M: txl.i32,
        G: txl.i32,
    ):
        u = txl.cta_id()
        g = u // NCB
        cb = u % NCB
        tid = txl.thread_id()
        tl = tid >> 5
        col = cb * RCOLS + (tid & 31)
        smem = txl.smem_pool()
        buf = smem.alloc([TB * RCOLS], "float32", align=16)
        txl.ptx.griddepcontrol.wait()
        s, e = bounds(offs, g, M, idtype)
        ntiles = (e - s + (TILE - 1)) // TILE
        f = g + s // TILE
        acc = txl.alloc_local([1], "float32")
        txl.assign(acc[0], txl.float32(0))
        vals = txl.alloc_local([RU], "float32")
        tmp = txl.alloc_local([1], "float32")
        with txl.serial((ntiles + TB - 1) // TB) as ch:
            base = ch * TB
            for i in range(RU):
                j = base + tl * RU + i
                txl.assign(vals[i], txl.float32(0))
                with txl.If(txl.And(j < ntiles, col < H)), txl.Then():
                    txl.ptx.ld.global_.f32(vals[i], part.ptr_to([txl.cast(f + j, "int64") * H + txl.cast(col, "int64")]))
            for i in range(RU):
                txl.ptx.st.shared.f32(buf.ptr_to([(tl * RU + i) * RCOLS + (tid & 31)]), vals[i])
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(RLANES * 32))
            with txl.If(tl == 0), txl.Then():
                for i in range(TB):
                    with txl.If(base + i < ntiles), txl.Then():
                        txl.ptx.ld.shared.f32(tmp[0], buf.ptr_to([i * RCOLS + (tid & 31)]))
                        txl.assign(acc[0], acc[0] + tmp[0])
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(RLANES * 32))
        with txl.If(txl.And(tl == 0, col < H)), txl.Then():
            store_vec(out, txl.cast(g, "int64") * H + txl.cast(col, "int64"), odtype, 1, acc)

    return tile_reduce


@functools.cache
def tile_reduce_wide(idtype, odtype, H):
    vec = math.gcd(H, 2)
    NV = H // vec
    TW = 128
    NC = cdiv(NV, TW)
    U = 32

    @txl.kernel(warps=TW // 32, grid=lambda p: p["G"] * NC)
    def tile_reduce(
        part: txl.gptr[txl.f32],
        offs: txl.gptr[idtype],
        out: txl.gptr[odtype],
        M: txl.i32,
        G: txl.i32,
    ):
        u = txl.cta_id()
        g = u // NC
        c = u % NC
        tid = txl.thread_id()
        v = c * TW + tid
        txl.ptx.griddepcontrol.wait()
        with txl.If(v < NV), txl.Then():
            s, e = bounds(offs, g, M, idtype)
            ntiles = (e - s + (TILE - 1)) // TILE
            f = g + s // TILE
            acc = txl.alloc_local([vec], "float32")
            for k in range(vec):
                txl.assign(acc[k], txl.float32(0))
            vals = txl.alloc_local([U * vec], "float32")
            with txl.serial((ntiles + U - 1) // U) as li:
                for i in range(U):
                    j = li * U + i
                    idx = txl.cast(f + j, "int64") * H + txl.cast(v * vec, "int64")
                    load_vec(part, idx, "float32", vec, 4, View(vals, i * vec), j < ntiles)
                for i in range(U):
                    j = li * U + i
                    with txl.If(j < ntiles), txl.Then():
                        for k in range(vec):
                            txl.assign(acc[k], acc[k] + vals[i * vec + k])
            store_vec(out, txl.cast(g, "int64") * H + txl.cast(v * vec, "int64"), odtype, vec, acc)

    return tile_reduce


@functools.cache
def compiled(which, *key):
    kern = {"sum": tile_sum2, "ring": tile_sum_ring2, "reduce": tile_reduce2, "reduce_wide": tile_reduce_wide}[which](*key)
    func = kern.func
    if which in ("reduce", "reduce_wide"):  # programmatic dependent launch: overlaps tile_sum's tail
        func = func.with_attr("tirx.kernel_launch_params", ["blockIdx.x", "threadIdx.x", "tirx.use_programtic_dependent_launch", "tirx.use_dyn_shared_memory"])
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        return entry_function(tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx"))


def segment_sum_sm100(x: torch.Tensor, grouped_mm_offs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    assert x.is_contiguous(), "x must be contiguous"
    M, H = x.shape
    G = grouped_mm_offs.shape[0]
    out = torch.empty((G, H), dtype=out_dtype, device=x.device)
    if G == 0 or H == 0:
        return out
    dt, it, ot = DT[x.dtype], IT[grouped_mm_offs.dtype], DT[out_dtype]
    slots = G + cdiv(M, TILE)
    part = torch.empty((max(slots, 1), H), dtype=torch.float32, device=x.device)
    # long groups: smem-staged chunked reduce; short groups: one thread per 2 columns
    k_red = compiled("reduce" if M >= LONG_TILES * TILE * G else "reduce_wide", it, ot, H)
    ring = ring_layout(dt, H)
    xf = x.view(-1) if M > 0 else torch.empty(1, dtype=x.dtype, device=x.device)
    if M > 0:
        if ring is not None:
            n_units = slots * ring[2]
            compiled("ring", dt, it, H)(xf, grouped_mm_offs, part.view(-1), M, G, n_units, min(n_units, CPS * sm_count()))
        else:
            vec, _ = vec2(dt, H)
            compiled("sum", dt, it, H)(xf, grouped_mm_offs, part.view(-1), M, G, slots * layout(H, vec)[1])
    k_red(part.view(-1), grouped_mm_offs, out.view(-1), M, G)
    return out
