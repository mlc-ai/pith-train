"""
segment_sum_sm100: per-group column sums of rows sorted by group, in TIRx-lite (B200).

Groups are cut into 128-row tiles from their own start; tile j of group g lives at slot f(g) + j
with f(g) = g + start_g // 128 (strictly increasing, collision-free, no scan). A tile's fp32 column
sums (rows in a fixed per-tile order) are computed once, then every group adds its tile sums in row
order starting from zero and rounds once to out_dtype. Dispatch:
- many groups (at least DMIN) whose (group, chunk) CTAs fit one wave or are short: one CTA per
  (group, chunk) folds the group's tiles in registers (no second kernel);
- otherwise (rows 16-byte aligned, at most GS groups): a compact persistent TMA ring (only real
  tiles, interleaved over the fewest CTAs that give every one the same unit count, each CTA's last
  ~TAILK units claimed from a shared tag-validated counter so slow SMs shed tail work; for bf16
  H=2880 128B-swizzled 256-column chunks) publishes each tile sum as tagged 64-bit words, and a
  PDL-launched reduce detects each chain's last word by its tag (no griddepcontrol.wait) and folds
  in order;
- rows not 16-byte aligned or more than GS groups: plain-LDG tile sums + ordered reduce.
Every TMA kernel sums a tile column by column with rows in classes (tile-relative row mod 4)
accumulated in row order and the classes folded 0, 1, 2, 3, so a group's bits do not depend on the
kernel its call is dispatched to.

The tagged scratch is a buffer this module owns per (device, stream), zero-filled when allocated: a
word counts as written only when its high half carries this call's tag, a 16-bit per-call epoch of
that buffer spread into an inf/NaN pattern (tag_of), so consecutive calls always differ. A zero word
carries no tag, and only this module's earlier calls on the same stream ever wrote the buffer; when
the epoch would wrap (every 65535 calls) the buffer is zeroed again, in stream order before the
call, so a stale word can never carry the current tag. The tag is a host argument fixed per launch,
so this path must not be captured in a CUDA graph (replays would reuse one tag over the previous
replay's words).

Evolved with TIRx-harness: run segment_sum, frontier member compact-swizzle-tagreduce (commit
3bc9af80, 181.894 us vs 640.179 us, 3.52x over the previous SM100 path in the task's benchmark).
Compiled once per process by tvm_tirx.tirx.compile(..., pipeline="tirx") and launched through
tvm-ffi on torch's current stream.
"""

import functools
import math

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc, as TVM_CUDA_COMPILE_MODE=nvcc chose before

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


TAGGED_SCRATCH = {}  # (device type, index, stream) -> [int64 buffer (uint64 words), epoch of its last call]


def tag_of(e):
    """
    The 32-bit word tag of epoch e (1 .. 0xFFFF): its 16 bits spread under a mask that makes the tag
    half an inf/NaN pattern as fp32, as both bf16 / fp16 halves and as FP8 bytes, never zero.
    """
    t = 0x7F807F80 | (e & 0x7F) | ((e & 0x80) << 8) | ((e & 0x7F00) << 8) | ((e & 0x8000) << 16)
    return t - (1 << 32) if t >= 1 << 31 else t


def tagged_scratch(n, device):
    """
    (the first n words of this stream's tagged scratch, this call's tag). The buffer is zero-filled
    when allocated or grown, and zeroed again (stream-ordered, before this call's kernels) when its
    epoch would wrap, so no word in it can carry the returned tag before this call writes it. Kept
    per (device, stream): launches on two streams never share words or epochs.
    """
    key = (device.type, device.index, torch.cuda.current_stream(device).cuda_stream)
    ent = TAGGED_SCRATCH.get(key)
    if ent is None or ent[0].numel() < n:
        ent = TAGGED_SCRATCH[key] = [torch.zeros(max(n, 1 << 16), dtype=torch.int64, device=device), 0]
    ent[1] += 1
    if ent[1] > 0xFFFF:
        ent[0].zero_()
        ent[1] = 1
    return ent[0][:n].view(torch.uint64), tag_of(ent[1])


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
        with txl.if_(pred), txl.then_():
            if nw == 4:
                txl.ptx.ld.global_.v4.b32(w[0], w[1], w[2], w[3], x.ptr_to([idx]))
            elif nw == 2:
                txl.ptx.ld.global_.v2.b32(w[0], w[1], x.ptr_to([idx]))
            else:
                txl.ptx.ld.global_.b32(w[0], x.ptr_to([idx]))
        for i in range(nw):
            if dtype == "float32":
                txl.assign(dst[i], txl.reinterpret(w[i], ty="float32"))
            elif dtype == "bfloat16":
                txl.assign(dst[2 * i], txl.reinterpret(w[i] << txl.uint32(16), ty="float32"))
                txl.assign(dst[2 * i + 1], txl.reinterpret(w[i] & txl.uint32(0xFFFF0000), ty="float32"))
            else:
                txl.idioms.cast_f16x2_to_f32x2(dst, i, w[i])
    else:  # one 16-bit element
        h = txl.alloc_local([1], "uint16")
        txl.assign(h[0], txl.uint16(0))
        with txl.if_(pred), txl.then_():
            txl.ptx.ld.global_.b16(h[0], x.ptr_to([idx]))
        if dtype == "bfloat16":
            txl.assign(dst[0], txl.reinterpret(txl.cast(h[0], "uint32") << txl.uint32(16), ty="float32"))
        else:
            txl.ptx.cvt.f32.f16(dst[0], h[0])


def store_vec(out, idx, dtype, vec, src):
    """
    out[idx : idx + vec] = src[0:vec] rounded to dtype.
    """
    if dtype == "float32":
        w = [txl.reinterpret(src[i], ty="uint32") for i in range(vec)]
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


def store_tagged(part, idx, vec, src, tag):
    """
    part[idx : idx + vec] = (tag << 32) | bits(src[i]): one 64-bit word per column, written with
    relaxed gpu-scope stores. A 64-bit scalar access is single-copy atomic, so a reader that sees
    this call's tag in a word also sees that word's value: every word validates itself and no fence
    or flag is needed.
    """
    hi = txl.cast(txl.cast(tag, "uint32"), "uint64") << txl.uint64(32)
    for i in range(0, vec, 2):
        w0 = hi | txl.cast(txl.reinterpret(src[i], ty="uint32"), "uint64")
        w1 = hi | txl.cast(txl.reinterpret(src[i + 1], ty="uint32"), "uint64")
        txl.ptx["st.relaxed.gpu.global.L2::cache_hint.v2.b64"](part.ptr_to([idx + i]), w0, w1, txl.uint64(EVICT_LAST))


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

    @txl.kernel()
    def tile_sum(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), part: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), M: txl.int32, G: txl.int32, n_units: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=n_units, block=(NT // 32) * 32))
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
        with txl.if_(nvalid > 0), txl.then_():
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
                with txl.if_(rg > 0), txl.then_():
                    for k in range(vec):
                        txl.ptx.st.shared.f32(red.ptr_to([((rg - 1) * TW + lc) * vec + k]), acc[k])
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NT))
            with txl.if_(txl.And(rg == 0, act)), txl.then_():
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
NWC = 4  # consumer warps; each owns two of the frontier's row classes
BULK_G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
EVICT_FIRST = 0x12F0000000000000
EVICT_LAST = 0x14F0000000000000
SMEM_BUDGET = 100000
LONG_TILES = 16  # average tiles per group from which the chunked reduce is used
CPS = 1  # persistent CTAs per SM
DMIN = 16  # groups from which the direct kernel may be used (fewer: compact ring)
SHORT_ROWS = 512  # average rows per group up to which the direct kernel is used regardless of waves
CVT = 48  # target vectors per column chunk
TAILK = 6  # ring: last ~TAILK units per CTA claimed dynamically from a shared pool (0 = fully static)
LEAD = 2  # static units before the end at which the first claim is issued
IDLE_JOIN = 0  # CTAs beyond n_eff also claim pool units


TMA_G2S = "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
TMA_G2S_3D = "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
BOXMAX = 256  # TMA box extent limit (elements)


def ring_layout(dtype, H):
    """
    None unless rows are 16-byte aligned (TMA strides); else (vec, NV, NC, CV, NS).

    Column chunks are TMA boxes of BC = CV * vec columns: the widest divisor of H that is a multiple
    of vec and at most 256, else 256 with the tail chunk's box running past H (zero fill).
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


def group_ring_layout(dtype, H):
    """
    The direct kernel uses the ring's column chunks, so both share one per-column tile arithmetic
    (rows by tile-relative index mod RG = 256 // CV, classes folded in order) for every H: a group's
    bits do not depend on which kernel a call is dispatched to.
    """
    ring = ring_layout(dtype, H)
    if ring is None:
        return None
    vec, NV, NC, CV, _ = ring
    NS = max(2, min(16, DIRECT_SMEM // (SR * CV * 16 + 8)))
    return vec, NV, NC, CV, NS


DIRECT_SMEM = 66000  # direct kernel ring budget: 4 stages of 16 KB, three CTAs per SM


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
    search over the staged ends: binary search for the last g with f(g) <= sigma.
    """
    lo = txl.local_scalar("int32")
    hi = txl.local_scalar("int32")
    mid = txl.local_scalar("int32")
    sv = txl.local_scalar("int32")
    txl.assign(lo, txl.int32(0))
    txl.assign(hi, txl.int32(GS))  # caller clamps to G - 1 below
    return lo, hi, mid, sv


def start_of(ends, g, sv):
    with txl.if_(g > 0):
        with txl.then_():
            txl.ptx.ld.shared.s32(sv, ends.ptr_to([txl.max(g - 1, 0)]))
        with txl.else_():
            txl.assign(sv, txl.int32(0))
    return sv


@functools.cache
def tile_sum_ring2(dtype, idtype, H, tagged=False):
    """
    Compact persistent TMA ring: units (real tile, column chunk) interleaved over the CTAs; each
    tile sum goes to its collision-free slot f(g) + j, as fp32 or (tagged) as tag-validated 64-bit
    words for tile_reduce_tag2.
    """
    vec, NV, NC, CV, NS = ring_layout(dtype, H)
    SWIZZLE = dtype == "bfloat16" and H == 2880  # 128B-swizzled 256-column chunks (conflict-free smem reads)
    if SWIZZLE:
        BC = 256
        CV = BC // vec
        NC = cdiv(H, BC)
        NS = max(2, min(16, SMEM_BUDGET // (SR * BC * 2 + 8)))
    ATOM = 64
    NBLOCK = (CV * vec) // ATOM if SWIZZLE else 0
    RG = (NWC * 32) // CV
    RB = CV * 16  # stage row stride in bytes
    KPU = TILE // SR  # stages per unit
    NPR = cdiv(SR, RG)  # rows per consumer thread per stage (max)
    NT = (NWC + 1) * 32

    esz = 16 // vec

    def prepare_host(params):
        xmap = txl.stack_alloca("tensormap", 1)
        if SWIZZLE:
            txl.call_packed(
                "runtime.cuTensorMapEncodeTiled", xmap, dtype, 3, params["x"].data,
                ATOM, params["M"], H // ATOM, H * esz, ATOM * esz,
                ATOM, SR, NBLOCK, 1, 1, 1,
                0, txl.SwizzleMode.SWIZZLE_128B_ATOM.value, 3, 0,
            )
        else:
            txl.call_packed(
                "runtime.cuTensorMapEncodeTiled", xmap, dtype, 2, params["x"].data,
                H, params["M"], H * esz, CV * vec, SR, 1, 1, 0, 0, 3, 0,
            )
        return xmap

    @txl.kernel()
    def tile_sum_ring(
        x: txl.Tensor((txl.int64(),), dtype, data_alignment=1),
        offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1),
        part: txl.Tensor((txl.int64(),), txl.u64 if tagged else txl.f32, data_alignment=1),
        M: txl.int32,
        G: txl.int32,
        n_units: txl.int32,
        n_ctas: txl.int32,
        tag: txl.int32,
    ):
        xmap = prepare_host({"x": x, "M": M})
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=n_ctas, block=(NWC + 1) * 32))
        cta = txl.cta_id()
        tid = txl.thread_id()
        warp = tid >> 5
        lane = tid & 31

        def iket_range(name):
            token = txl.alloc_local([1], "uint32")
            txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        txl.ptx.griddepcontrol.launch_dependents()
        smem = txl.smem_pool()
        full = smem.alloc([NS], "uint64", align=8)
        empty = smem.alloc([NS], "uint64", align=8)
        meta = smem.alloc([NS], "int32", align=4)
        msig = smem.alloc([NS], "int32", align=4)
        munit = smem.alloc([NS], "int32", align=4)
        ends = smem.alloc([GS], "int32", align=4)
        fpre = smem.alloc([GS + 1], "int32", align=4)
        red = smem.alloc([2 * max(RG - 1, 1) * CV * vec], "float32", align=16)
        if SWIZZLE:
            ring = smem.alloc((NS, NBLOCK, SR, ATOM), dtype, swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM, align=1024)
        else:
            ring = smem.alloc([NS * SR * RB // 4], "uint32", align=128)
        ctr = part.ptr_to([txl.cast(G + (M + (TILE - 1)) // TILE, "int64") * H])
        thi = txl.cast(txl.cast(tag, "uint32"), "uint64") << txl.uint64(32)
        if TAILK > 0:
            with txl.if_(txl.And(cta == 0, tid == NWC * 32)), txl.then_():
                # pool counter, (tag << 32) | claims: a claim that sees another tag (stale scratch, or this store not
                # yet performed) is retried, so the counter needs no zeroed scratch
                txl.ptx["st.relaxed.gpu.global.b64"](ctr, thi)
        with txl.if_(tid == NWC * 32), txl.then_():
            txl.ptx.prefetch.tensormap(txl.address_of(xmap))
            # L2 prefetch of the first unit, guessing compact tile sigma at row sigma * TILE (exact when every group
            # before it is a multiple of 128 rows); a wrong guess only costs bandwidth. One unit measured best (two or
            # more, or only part of one, start the stream slower).
            for k2 in range(1):
                uu = cta + k2 * n_ctas
                with txl.if_((uu // NC) * TILE < M), txl.then_():
                    for k in range(KPU):
                        if SWIZZLE:
                            txl.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"](
                                txl.address_of(xmap),
                                txl.int32(0), (uu // NC) * TILE + k * SR, (uu % NC) * NBLOCK,
                                txl.uint64(EVICT_FIRST),
                            )
                        else:
                            txl.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"](
                                txl.address_of(xmap),
                                (uu % NC) * (CV * vec), (uu // NC) * TILE + k * SR,
                                txl.uint64(EVICT_FIRST),
                            )
        with txl.if_(tid == 0), txl.then_():
            for i in range(NS):
                txl.ptx.mbarrier.init.shared.b64(full.ptr_to([i]), txl.uint32(1))
                txl.ptx.mbarrier.init.shared.b64(empty.ptr_to([i]), txl.uint32(NWC))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.serial((G + NT - 1) // NT) as gi:
            g = gi * NT + tid
            with txl.if_(g < G), txl.then_():
                txl.ptx.st.shared.s32(ends.ptr_to([g]), txl.min(load_off(offs, g, idtype), M))
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NT))
        # compact tile numbering: fpre[g] = real tiles of groups < g (no units for the empty slots)
        with txl.if_(warp == 0), txl.then_():
            run = txl.local_scalar("int32")
            en = txl.local_scalar("int32")
            sv0 = txl.local_scalar("int32")
            incl = txl.local_scalar("int32")
            peer = txl.local_scalar("int32")
            txl.assign(run, txl.int32(0))
            with txl.serial((G + 31) // 32) as gi:
                g = gi * 32 + lane
                txl.assign(en, txl.int32(0))
                txl.assign(sv0, txl.int32(0))
                with txl.if_(g < G), txl.then_():
                    txl.ptx.ld.shared.s32(en, ends.ptr_to([g]))
                    with txl.if_(g > 0), txl.then_():
                        txl.ptx.ld.shared.s32(sv0, ends.ptr_to([txl.max(g - 1, 0)]))
                nt = (txl.max(en, sv0) - sv0 + (TILE - 1)) // TILE
                txl.assign(incl, nt)
                for d in (1, 2, 4, 8, 16):
                    txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(d), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                    txl.assign(incl, incl + txl.Select(lane >= d, peer, txl.int32(0)))
                with txl.if_(g < G), txl.then_():
                    txl.ptx.st.shared.s32(fpre.ptr_to([g]), run + incl - nt)
                txl.ptx.shfl_sync.idx.b32(peer, incl, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                txl.assign(run, run + peer)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.s32(fpre.ptr_to([G]), run)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NT))
        tot = txl.local_scalar("int32")
        txl.ptx.ld.shared.s32(tot, fpre.ptr_to([G]))
        # Equal unit counts: only the fewest CTAs that still take ceil(units / n_ctas) units each stream, so no
        # partial last wave of single units runs alone after the others finish (e.g. 3264 units on 148 SMs: 142).
        n_all = tot * NC
        per = txl.max((n_all + n_ctas - 1) // n_ctas, txl.int32(1))
        n_eff = txl.max((n_all + per - 1) // per, txl.int32(1))
        n_mine = txl.Select(cta < n_eff, (n_all - cta + n_eff - 1) // n_eff, txl.int32(0))
        if TAILK > 0:
            # static units cta + i * n_eff for i < n_stat, then the pool [S0, n_all) is claimed one unit at a time
            n_stat = txl.max(per - TAILK, txl.int32(0))
            S0 = n_eff * n_stat
            my_stat = txl.Select(cta < n_eff, n_stat, txl.int32(0))
            joins = txl.Or(cta < n_eff, txl.int32(IDLE_JOIN) == 1)

        with txl.if_(warp == NWC):
            with txl.then_():  # producer warp
                role_token = iket_range("ring-producer")
                if TAILK > 0:
                    it = txl.local_scalar("int32")
                    u = txl.local_scalar("int32")
                    pend = txl.local_scalar("uint64")
                    polls = txl.local_scalar("int32")
                    bhi = txl.local_scalar("uint32")
                    blo = txl.local_scalar("uint32")

                    def claim():
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx["atom.relaxed.gpu.global.add.u64"](pend, ctr, txl.uint64(1))

                    def bcast():
                        txl.ptx.shfl_sync.idx.b32(bhi, txl.cast(pend >> txl.uint64(32), "uint32"), txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                        txl.ptx.shfl_sync.idx.b32(blo, txl.cast(pend & txl.uint64(0xFFFFFFFF), "uint32"), txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF))

                    def take():  # the pending claim's unit (warp-uniform), retrying claims that saw a stale tag
                        txl.assign(polls, txl.int32(0))
                        bcast()
                        with txl.while_(txl.And(bhi != txl.cast(tag, "uint32"), polls < POLL_LIMIT)):
                            txl.cuda.nano_sleep(txl.uint32(TR_SLEEP))
                            claim()
                            bcast()
                            txl.assign(polls, polls + 1)
                        txl.assign(u, txl.Select(polls < POLL_LIMIT, S0 + txl.cast(blo, "int32"), n_all))
                        with txl.if_(u < n_all), txl.then_():
                            claim()  # one claim ahead

                    txl.assign(it, txl.int32(0))
                    txl.assign(u, n_all)
                    with txl.if_(my_stat > 0):
                        with txl.then_():
                            txl.assign(u, cta)
                        with txl.else_():
                            with txl.if_(joins), txl.then_():
                                claim()
                                take()
                    loop = txl.while_(u < n_all)
                else:
                    loop = txl.serial(n_mine)
                with loop as it_:
                    if TAILK > 0:
                        with txl.if_(txl.And(my_stat > 0, it == txl.max(my_stat - LEAD, txl.int32(0)))), txl.then_():
                            claim()
                    else:
                        it = it_
                        u = cta + it * n_eff
                    sigma = u // NC
                    c = u % NC
                    r0 = txl.local_scalar("int32")
                    nvalid = txl.local_scalar("int32")
                    lo, hi, mid, sv = search_smem(ends, sigma)
                    fv = txl.local_scalar("int32")
                    grp = txl.local_scalar("int32")
                    # Above the small-G crossover, load one warp of the prefix table in parallel and derive the
                    # upper-bound index with one ballot.  Binary search is cheaper for G < 16 and handles G > 32.
                    with txl.if_(txl.And(G >= 16, G <= 32)):
                        with txl.then_():
                            fg = txl.local_scalar("int32")
                            txl.assign(fg, txl.int32(0))
                            with txl.if_(lane < G), txl.then_():
                                txl.ptx.ld.shared.s32(fg, fpre.ptr_to([lane]))
                            bal = txl.alloc_local([1], "uint32")
                            txl.ptx.vote_sync.ballot.b32(
                                bal[0], txl.And(lane < G, fg <= sigma), txl.uint32(0xFFFFFFFF)
                            )
                            txl.assign(grp, txl.cast(txl.popcount(bal[0]), "int32") - 1)
                        with txl.else_():
                            txl.assign(hi, G - 1)
                            with txl.while_(lo < hi):
                                txl.assign(mid, (lo + hi + 1) >> 1)
                                txl.ptx.ld.shared.s32(fv, fpre.ptr_to([mid]))
                                with txl.if_(fv <= sigma):
                                    with txl.then_():
                                        txl.assign(lo, mid)
                                    with txl.else_():
                                        txl.assign(hi, mid - 1)
                            txl.assign(grp, lo)
                    st = start_of(ends, grp, sv)
                    en = txl.local_scalar("int32")
                    txl.ptx.ld.shared.s32(en, ends.ptr_to([grp]))
                    txl.ptx.ld.shared.s32(fv, fpre.ptr_to([grp]))
                    txl.assign(r0, st + (sigma - fv) * TILE)
                    txl.assign(nvalid, txl.min(en - r0, txl.int32(TILE)))
                    slot = grp + st // TILE + (sigma - fv)  # the tile's partial slot f(g) + j
                    for k in range(KPU):
                        q = it * KPU + k
                        s = q % NS
                        ph = (q // NS) & 1
                        txl.cuda.mbarrier_wait(empty.ptr_to([s]), ph ^ 1)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.st.shared.s32(meta.ptr_to([s]), nvalid)
                            txl.ptx.st.shared.s32(msig.ptr_to([s]), slot)
                            txl.ptx.st.shared.s32(munit.ptr_to([s]), u)
                            with txl.if_(nvalid > k * SR):
                                with txl.then_():
                                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(full.ptr_to([s]), txl.uint32(SR * CV * 16))
                                    if SWIZZLE:
                                        txl.ptx[TMA_G2S_3D](
                                            ring.ptr_to([s, 0, 0, 0]),
                                            txl.address_of(xmap),
                                            txl.int32(0), r0 + k * SR, c * NBLOCK,
                                            txl.cuda.cvta_generic_to_shared(full.ptr_to([s])),
                                            txl.uint64(EVICT_FIRST),
                                        )
                                    else:
                                        txl.ptx[TMA_G2S](
                                            ring.ptr_to([s * SR * (RB // 4)]),
                                            txl.address_of(xmap),
                                            c * (CV * vec),
                                            r0 + k * SR,
                                            txl.cuda.cvta_generic_to_shared(full.ptr_to([s])),
                                            txl.uint64(EVICT_FIRST),
                                        )
                                with txl.else_():
                                    txl.ptx.mbarrier.arrive.shared.b64(full.ptr_to([s]), txl.uint32(1))
                    if TAILK > 0:
                        txl.assign(it, it + 1)
                        with txl.if_(it < my_stat):
                            with txl.then_():
                                txl.assign(u, cta + it * n_eff)
                            with txl.else_():
                                take()
                if TAILK > 0:  # end marker for the consumers
                    q = it * KPU
                    s = q % NS
                    txl.cuda.mbarrier_wait(empty.ptr_to([s]), ((q // NS) & 1) ^ 1)
                    with txl.if_(lane == 0), txl.then_():
                        txl.ptx.st.shared.s32(munit.ptr_to([s]), txl.int32(-1))
                        txl.ptx.mbarrier.arrive.shared.b64(full.ptr_to([s]), txl.uint32(1))
                txl.cuda.iket.range_end(role_token[0])
            with txl.else_():  # consumer warps
                role_token = iket_range("ring-consumer")
                if RG <= NWC and CV <= 32:
                    lc = lane
                    rg = warp
                    act = txl.And(rg < RG, lc < CV)
                else:
                    lc = tid % CV
                    rg = tid // CV
                    act = rg < RG
                acc = txl.alloc_local([vec], "float32")
                vals = txl.alloc_local([NPR * vec], "float32")
                w = txl.alloc_local([NPR * 4], "uint32")
                nv = txl.local_scalar("int32")
                slot_c = txl.local_scalar("int32")
                packed_add = txl.local_scalar("uint64")
                if TAILK > 0:
                    it = txl.local_scalar("int32")
                    u = txl.local_scalar("int32")
                    txl.assign(it, txl.int32(0))
                    txl.cuda.mbarrier_wait(full.ptr_to([0]), 0)
                    txl.ptx.ld.shared.s32(u, munit.ptr_to([0]))
                    loop = txl.while_(u >= 0)
                else:
                    loop = txl.serial(n_mine)
                with loop as it_:
                    unit_token = iket_range("ring-unit")
                    if TAILK == 0:
                        it = it_
                        u = cta + it * n_eff
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
                        if k == 0:
                            txl.ptx.ld.shared.s32(slot_c, msig.ptr_to([s]))
                        # rows i of this stage with (k * SR + i) % RG == rg, in increasing order
                        i0 = (rg + RG - (k * SR) % RG) % RG
                        for j in range(NPR):
                            i = i0 + j * RG
                            okr = txl.And(txl.And(act, lc < cv_c), txl.And(i < SR, k * SR + i < nv))
                            for kk in range(4):
                                txl.assign(w[j * 4 + kk], txl.uint32(0))
                            with txl.if_(okr), txl.then_():
                                if SWIZZLE:
                                    txl.ptx.ld.shared.v4.b32(
                                        w[j * 4], w[j * 4 + 1], w[j * 4 + 2], w[j * 4 + 3],
                                        ring.ptr_to([s, (lc * vec) // ATOM, i, (lc * vec) % ATOM]),
                                    )
                                else:
                                    txl.ptx.ld.shared.v4.b32(w[j * 4], w[j * 4 + 1], w[j * 4 + 2], w[j * 4 + 3], ring.ptr_to([(s * SR + i) * (RB // 4) + lc * 4]))
                        txl.cuda.warp_sync()
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.mbarrier.arrive.shared.b64(empty.ptr_to([s]), txl.uint32(1))
                        for j in range(NPR):
                            for kk in range(4):
                                wd = w[j * 4 + kk]
                                if dtype == "float32":
                                    txl.assign(vals[j * vec + kk], txl.reinterpret(wd, ty="float32"))
                                elif dtype == "bfloat16":
                                    txl.assign(vals[j * vec + 2 * kk], txl.reinterpret(wd << txl.uint32(16), ty="float32"))
                                    txl.assign(vals[j * vec + 2 * kk + 1], txl.reinterpret(wd & txl.uint32(0xFFFF0000), ty="float32"))
                                else:
                                    txl.idioms.cast_f16x2_to_f32x2(View(vals, j * vec), kk, wd)
                            for kk in range(0, vec, 2):
                                txl.ptx.add.rn.f32x2(
                                    packed_add,
                                    txl.cuda.make_float2(acc[kk], acc[kk + 1]),
                                    txl.cuda.make_float2(vals[j * vec + kk], vals[j * vec + kk + 1]),
                                )
                                txl.ptx.mov.b64(acc[kk], acc[kk + 1], packed_add)
                    rb = (it & 1) * (max(RG - 1, 1) * CV * vec)
                    if RG > 1:
                        with txl.if_(txl.And(rg > 0, act)), txl.then_():
                            for kk in range(vec):
                                txl.ptx.st.shared.f32(red.ptr_to([rb + ((rg - 1) * CV + lc) * vec + kk]), acc[kk])
                        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NWC * 32))
                    with txl.if_(txl.And(txl.And(rg == 0, lc < cv_c), nv > 0)), txl.then_():
                        pair = txl.alloc_local([2], "float32")
                        for r in range(1, RG):
                            for kk in range(0, vec, 2):
                                txl.ptx.ld.shared.v2.b32(
                                    pair[0], pair[1],
                                    red.ptr_to([rb + ((r - 1) * CV + lc) * vec + kk]),
                                )
                                txl.ptx.add.rn.f32x2(
                                    packed_add,
                                    txl.cuda.make_float2(acc[kk], acc[kk + 1]),
                                    txl.cuda.make_float2(pair[0], pair[1]),
                                )
                                txl.ptx.mov.b64(acc[kk], acc[kk + 1], packed_add)
                        if tagged:
                            store_tagged(part, txl.cast(slot_c, "int64") * H + txl.cast((c * CV + lc) * vec, "int64"), vec, acc, tag)
                        else:
                            store_vec(part, txl.cast(slot_c, "int64") * H + txl.cast((c * CV + lc) * vec, "int64"), "float32", vec, acc)
                    txl.cuda.iket.range_end(unit_token[0])
                    if TAILK > 0:
                        txl.assign(it, it + 1)
                        qn = it * KPU
                        txl.cuda.mbarrier_wait(full.ptr_to([qn % NS]), (qn // NS) & 1)
                        txl.ptx.ld.shared.s32(u, munit.ptr_to([qn % NS]))
                txl.cuda.iket.range_end(role_token[0])

    return tile_sum_ring


@functools.cache
def group_sum_ring2(dtype, idtype, odtype, H):
    """
    One CTA per (group, column chunk), accumulating 128-row tile sums in order.
    """
    vec, NV, NC, CV, NS = group_ring_layout(dtype, H)
    SWIZZLE = dtype == "bfloat16" and H == 2880
    if SWIZZLE:
        BC = 256
        CV = BC // vec
        NC = cdiv(H, BC)
        NS = max(2, min(16, DIRECT_SMEM // (SR * BC * 2 + 8)))
    ATOM = 64
    NBLOCK = (CV * vec) // ATOM if SWIZZLE else 0
    RG = (NWC * 32) // CV
    RB = CV * 16
    KPU = TILE // SR
    NPR = cdiv(SR, RG)
    NT = (NWC + 1) * 32
    BC = CV * vec
    esz = 16 // vec

    def prepare_host(params):
        xmap = txl.stack_alloca("tensormap", 1)
        if SWIZZLE:
            txl.call_packed(
                "runtime.cuTensorMapEncodeTiled", xmap, dtype, 3, params["x"].data,
                ATOM, params["M"], H // ATOM, H * esz, ATOM * esz,
                ATOM, SR, NBLOCK, 1, 1, 1,
                0, txl.SwizzleMode.SWIZZLE_128B_ATOM.value, 3, 0,
            )
        else:
            txl.call_packed(
                "runtime.cuTensorMapEncodeTiled", xmap, dtype, 2, params["x"].data,
                H, params["M"], H * esz, BC, SR, 1, 1, 0, 0, 3, 0,
            )
        return xmap

    @txl.kernel()
    def group_sum_ring(
        x: txl.Tensor((txl.int64(),), dtype, data_alignment=1),
        offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1),
        out: txl.Tensor((txl.int64(),), odtype, data_alignment=1),
        M: txl.int32,
        G: txl.int32,
    ):
        xmap = prepare_host({"x": x, "M": M})
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NC, block=(NWC + 1) * 32))
        u = txl.cta_id()
        g = u // NC
        c = u % NC
        tid = txl.thread_id()
        warp = tid >> 5
        lane = tid & 31

        def iket_range(name):
            token = txl.alloc_local([1], "uint32")
            txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        smem = txl.smem_pool()
        full = smem.alloc([NS], "uint64", align=8)
        empty = smem.alloc([NS], "uint64", align=8)
        meta = smem.alloc([NS], "int32", align=4)
        red = smem.alloc([2 * max(RG - 1, 1) * CV * vec], "float32", align=16)
        if SWIZZLE:
            ring = smem.alloc((NS, NBLOCK, SR, ATOM), dtype, swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM, align=1024)
        else:
            ring = smem.alloc([NS * SR * RB // 4], "uint32", align=128)
        start, end = bounds(offs, g, M, idtype)  # issued before the barrier setup: overlaps its latency
        with txl.if_(tid == NWC * 32), txl.then_():
            txl.ptx.prefetch.tensormap(txl.address_of(xmap))
        with txl.if_(tid == 0), txl.then_():
            for i in range(NS):
                txl.ptx.mbarrier.init.shared.b64(full.ptr_to([i]), txl.uint32(1))
                txl.ptx.mbarrier.init.shared.b64(empty.ptr_to([i]), txl.uint32(NWC))
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NT))
        ntiles = (end - start + (TILE - 1)) // TILE

        with txl.if_(warp == NWC):
            with txl.then_():
                role_token = iket_range("group-producer")
                with txl.serial(ntiles) as it:
                    r0 = start + it * TILE
                    nvalid = txl.min(end - r0, txl.int32(TILE))
                    for k in range(KPU):
                        q = it * KPU + k
                        sl = q % NS
                        ph = (q // NS) & 1
                        txl.cuda.mbarrier_wait(empty.ptr_to([sl]), ph ^ 1)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.st.shared.s32(meta.ptr_to([sl]), nvalid)
                            with txl.if_(nvalid > k * SR):
                                with txl.then_():
                                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        full.ptr_to([sl]), txl.uint32(SR * CV * 16)
                                    )
                                    if SWIZZLE:
                                        txl.ptx[TMA_G2S_3D](
                                            ring.ptr_to([sl, 0, 0, 0]),
                                            txl.address_of(xmap),
                                            txl.int32(0), r0 + k * SR, c * NBLOCK,
                                            txl.cuda.cvta_generic_to_shared(full.ptr_to([sl])),
                                            txl.uint64(EVICT_FIRST),
                                        )
                                    else:
                                        txl.ptx[TMA_G2S](
                                            ring.ptr_to([sl * SR * (RB // 4)]),
                                            txl.address_of(xmap),
                                            c * BC,
                                            r0 + k * SR,
                                            txl.cuda.cvta_generic_to_shared(full.ptr_to([sl])),
                                            txl.uint64(EVICT_FIRST),
                                        )
                                with txl.else_():
                                    txl.ptx.mbarrier.arrive.shared.b64(full.ptr_to([sl]), txl.uint32(1))
                txl.cuda.iket.range_end(role_token[0])
            with txl.else_():
                role_token = iket_range("group-consumer")
                if RG <= NWC and CV <= 32:
                    lc = lane
                    rg = warp
                    act = txl.And(rg < RG, lc < CV)
                else:
                    lc = tid % CV
                    rg = tid // CV
                    act = rg < RG
                cv_c = txl.min(txl.int32(CV), NV - c * CV)
                acc = txl.alloc_local([vec], "float32")
                gacc = txl.alloc_local([vec], "float32")
                vals = txl.alloc_local([NPR * vec], "float32")
                words = txl.alloc_local([NPR * 4], "uint32")
                nv = txl.local_scalar("int32")
                packed_add = txl.local_scalar("uint64")
                for kk in range(vec):
                    txl.assign(gacc[kk], txl.float32(0))
                with txl.serial(ntiles) as it:
                    tile_token = iket_range("group-tile")
                    for kk in range(vec):
                        txl.assign(acc[kk], txl.float32(0))
                    for k in range(KPU):
                        q = it * KPU + k
                        sl = q % NS
                        ph = (q // NS) & 1
                        txl.cuda.mbarrier_wait(full.ptr_to([sl]), ph)
                        txl.ptx.ld.shared.s32(nv, meta.ptr_to([sl]))
                        i0 = (rg + RG - (k * SR) % RG) % RG
                        for j in range(NPR):
                            i = i0 + j * RG
                            okr = txl.And(
                                txl.And(act, lc < cv_c),
                                txl.And(i < SR, k * SR + i < nv),
                            )
                            for kk in range(4):
                                txl.assign(words[j * 4 + kk], txl.uint32(0))
                            with txl.if_(okr), txl.then_():
                                if SWIZZLE:
                                    txl.ptx.ld.shared.v4.b32(
                                        words[j * 4], words[j * 4 + 1],
                                        words[j * 4 + 2], words[j * 4 + 3],
                                        ring.ptr_to([sl, (lc * vec) // ATOM, i, (lc * vec) % ATOM]),
                                    )
                                else:
                                    txl.ptx.ld.shared.v4.b32(
                                        words[j * 4], words[j * 4 + 1],
                                        words[j * 4 + 2], words[j * 4 + 3],
                                        ring.ptr_to([(sl * SR + i) * (RB // 4) + lc * 4]),
                                    )
                        txl.cuda.warp_sync()
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.mbarrier.arrive.shared.b64(empty.ptr_to([sl]), txl.uint32(1))
                        for j in range(NPR):
                            for kk in range(4):
                                wd = words[j * 4 + kk]
                                if dtype == "float32":
                                    txl.assign(vals[j * vec + kk], txl.reinterpret(wd, ty="float32"))
                                elif dtype == "bfloat16":
                                    txl.assign(vals[j * vec + 2 * kk], txl.reinterpret(wd << txl.uint32(16), ty="float32"))
                                    txl.assign(vals[j * vec + 2 * kk + 1], txl.reinterpret(wd & txl.uint32(0xFFFF0000), ty="float32"))
                                else:
                                    txl.idioms.cast_f16x2_to_f32x2(View(vals, j * vec), kk, wd)
                            for kk in range(0, vec, 2):
                                txl.ptx.add.rn.f32x2(
                                    packed_add,
                                    txl.cuda.make_float2(acc[kk], acc[kk + 1]),
                                    txl.cuda.make_float2(vals[j * vec + kk], vals[j * vec + kk + 1]),
                                )
                                txl.ptx.mov.b64(acc[kk], acc[kk + 1], packed_add)
                    rb = (it & 1) * (max(RG - 1, 1) * CV * vec)
                    if RG > 1:
                        with txl.if_(txl.And(rg > 0, act)), txl.then_():
                            for kk in range(vec):
                                txl.ptx.st.shared.f32(
                                    red.ptr_to([rb + ((rg - 1) * CV + lc) * vec + kk]), acc[kk]
                                )
                        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NWC * 32))
                    with txl.if_(txl.And(rg == 0, lc < cv_c)), txl.then_():
                        pair = txl.alloc_local([2], "float32")
                        for r in range(1, RG):
                            for kk in range(0, vec, 2):
                                txl.ptx.ld.shared.v2.b32(
                                    pair[0], pair[1],
                                    red.ptr_to([rb + ((r - 1) * CV + lc) * vec + kk]),
                                )
                                txl.ptx.add.rn.f32x2(
                                    packed_add,
                                    txl.cuda.make_float2(acc[kk], acc[kk + 1]),
                                    txl.cuda.make_float2(pair[0], pair[1]),
                                )
                                txl.ptx.mov.b64(acc[kk], acc[kk + 1], packed_add)
                        for kk in range(0, vec, 2):
                            txl.ptx.add.rn.f32x2(
                                packed_add,
                                txl.cuda.make_float2(gacc[kk], gacc[kk + 1]),
                                txl.cuda.make_float2(acc[kk], acc[kk + 1]),
                            )
                            txl.ptx.mov.b64(gacc[kk], gacc[kk + 1], packed_add)
                    txl.cuda.iket.range_end(tile_token[0])
                with txl.if_(txl.And(rg == 0, lc < cv_c)), txl.then_():
                    store_vec(
                        out,
                        txl.cast(g, "int64") * H + txl.cast((c * CV + lc) * vec, "int64"),
                        odtype, vec, gacc,
                    )
                txl.cuda.iket.range_end(role_token[0])

    return group_sum_ring


@functools.cache
def sm_count():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


RCOLS, RLANES, RU = 32, 8, 8  # reduce: columns per CTA, tile lanes, tiles per lane per chunk


@functools.cache
def tile_reduce2(idtype, odtype, H):
    """
    Ordered tile reduction; one column/lane is faster for the long low-G rows.
    """
    PC = 1
    CTA_COLS = RCOLS * PC
    NCB = cdiv(H, CTA_COLS)
    TB = RLANES * RU

    @txl.kernel()
    def tile_reduce(part: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), out: txl.Tensor((txl.int64(),), odtype, data_alignment=1), M: txl.int32, G: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NCB, block=(RLANES) * 32, programmatic_stream_serialization=True))
        u = txl.cta_id()
        g = u // NCB
        cb = u % NCB
        tid = txl.thread_id()
        tl = tid >> 5
        col = cb * CTA_COLS + (tid & 31) * PC
        smem = txl.smem_pool()
        buf = smem.alloc([TB * CTA_COLS], "float32", align=16)
        txl.ptx.griddepcontrol.wait()
        s, e = bounds(offs, g, M, idtype)
        ntiles = (e - s + (TILE - 1)) // TILE
        f = g + s // TILE
        acc = txl.alloc_local([PC], "float32")
        for k in range(PC):
            txl.assign(acc[k], txl.float32(0))
        vals = txl.alloc_local([RU * PC], "float32")
        tmp = txl.alloc_local([PC], "float32")
        packed_add = txl.local_scalar("uint64")
        with txl.serial((ntiles + TB - 1) // TB) as ch:
            base = ch * TB
            for i in range(RU):
                j = base + tl * RU + i
                load_vec(
                    part,
                    txl.cast(f + j, "int64") * H + txl.cast(col, "int64"),
                    "float32", PC, 4, View(vals, i * PC),
                    txl.And(j < ntiles, col < H),
                )
            for i in range(RU):
                bp = buf.ptr_to([(tl * RU + i) * CTA_COLS + (tid & 31) * PC])
                if PC == 2:
                    txl.ptx.st.shared.v2.b32(bp, txl.reinterpret(vals[i * PC], ty="uint32"), txl.reinterpret(vals[i * PC + 1], ty="uint32"))
                else:
                    txl.ptx.st.shared.f32(bp, vals[i])
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(RLANES * 32))
            with txl.if_(tl == 0), txl.then_():
                for i in range(TB):
                    with txl.if_(base + i < ntiles), txl.then_():
                        bp = buf.ptr_to([i * CTA_COLS + (tid & 31) * PC])
                        if PC == 2:
                            txl.ptx.ld.shared.v2.b32(tmp[0], tmp[1], bp)
                            txl.ptx.add.rn.f32x2(
                                packed_add,
                                txl.cuda.make_float2(acc[0], acc[1]),
                                txl.cuda.make_float2(tmp[0], tmp[1]),
                            )
                            txl.ptx.mov.b64(acc[0], acc[1], packed_add)
                        else:
                            txl.ptx.ld.shared.f32(tmp[0], bp)
                            txl.assign(acc[0], acc[0] + tmp[0])
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(RLANES * 32))
        with txl.if_(txl.And(tl == 0, col < H)), txl.then_():
            store_vec(out, txl.cast(g, "int64") * H + txl.cast(col, "int64"), odtype, PC, acc)

    return tile_reduce


@functools.cache
def tile_reduce_wide2(idtype, odtype, H):
    vec = math.gcd(H, 2)
    NV = H // vec
    TW = 128
    NC = cdiv(NV, TW)
    U = 32

    @txl.kernel()
    def tile_reduce(part: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), out: txl.Tensor((txl.int64(),), odtype, data_alignment=1), M: txl.int32, G: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NC, block=(TW // 32) * 32, programmatic_stream_serialization=True))
        u = txl.cta_id()
        g = u // NC
        c = u % NC
        tid = txl.thread_id()
        v = c * TW + tid
        txl.ptx.griddepcontrol.wait()
        with txl.if_(v < NV), txl.then_():
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
                    with txl.if_(j < ntiles), txl.then_():
                        for k in range(vec):
                            txl.assign(acc[k], acc[k] + vals[i * vec + k])
            store_vec(out, txl.cast(g, "int64") * H + txl.cast(v * vec, "int64"), odtype, vec, acc)

    return tile_reduce


TR_LONG = (1, 8, 8)  # tag reduce for long groups: (column warps, tile lanes, tiles per lane per round)
TR_SHORT = (4, 2, 8)  # short groups: wider column blocks, fewer tile lanes
TR_SLEEP = 64  # ns between tag polls
POLL_LIMIT = 1 << 20  # poll rounds before a word is given up on (NaN result, never a hang)


@functools.cache
def tile_reduce_tag2(idtype, odtype, H, CW, LANES, U):
    """
    Ordered reduce over tagged tile words, gated by tag detection instead of griddepcontrol.wait:
    one thread polls the chain's last tile word (the ring writes tiles in row order); then the CTA
    loads up to LANES * U tiles at once (one column per lane, every word's tag checked, a rare late
    word re-polled), stages them in shared memory and tile lane 0 adds them in row order from zero.
    Absent tiles stage +0.0, which leaves the running sum unchanged (it starts at +0 and can never
    be -0), so the fold needs no per-tile guard.
    """
    COLS = 32 * CW
    NCB = cdiv(H, COLS)
    TB = LANES * U
    NTH = COLS * LANES

    @txl.kernel()
    def tile_reduce_tag(part: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), out: txl.Tensor((txl.int64(),), odtype, data_alignment=1), M: txl.int32, G: txl.int32, tag: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NCB, block=(CW * LANES) * 32, programmatic_stream_serialization=True))
        u = txl.cta_id()
        g = u // NCB
        cb = u % NCB
        tid = txl.thread_id()
        cl = tid % COLS
        tl = tid // COLS
        col = cb * COLS + cl
        hi = txl.cast(txl.cast(tag, "uint32"), "uint64")
        smem = txl.smem_pool()
        buf = smem.alloc([TB * COLS], "float32", align=16)
        s, e = bounds(offs, g, M, idtype)
        ntiles = (e - s + (TILE - 1)) // TILE
        f = g + s // TILE
        polls = txl.local_scalar("int32")
        pend = txl.local_scalar("int32")
        with txl.if_(txl.And(tid == 0, ntiles > 0)), txl.then_():
            lw = txl.local_scalar("uint64")
            lp = txl.cast(f + ntiles - 1, "int64") * H + txl.cast(cb * COLS, "int64")
            txl.ptx["ld.relaxed.gpu.global.b64"](lw, part.ptr_to([lp]))
            txl.assign(polls, txl.int32(0))
            with txl.while_(txl.And((lw >> txl.uint64(32)) != hi, polls < POLL_LIMIT)):
                txl.cuda.nano_sleep(txl.uint32(TR_SLEEP))
                txl.ptx["ld.relaxed.gpu.global.b64"](lw, part.ptr_to([lp]))
                txl.assign(polls, polls + 1)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NTH))
        acc = txl.alloc_local([1], "float32")
        txl.assign(acc[0], txl.float32(0))
        wv = txl.alloc_local([U], "uint64")
        tmp = txl.alloc_local([TB], "float32")
        with txl.serial((ntiles + TB - 1) // TB) as ch:
            base = ch * TB
            for i in range(U):
                j = base + tl * U + i
                txl.assign(wv[i], hi << txl.uint64(32))  # absent tiles / columns: +0.0 with a valid tag
                with txl.if_(txl.And(j < ntiles, col < H)), txl.then_():
                    txl.ptx["ld.relaxed.gpu.global.b64"](wv[i], part.ptr_to([txl.cast(f + j, "int64") * H + txl.cast(col, "int64")]))
            # words whose tag is not this call's yet are re-loaded together, one round trip per round
            txl.assign(polls, txl.int32(0))
            txl.assign(pend, txl.int32(1))
            with txl.while_(txl.And(pend == 1, polls < POLL_LIMIT)):
                txl.assign(pend, txl.int32(0))
                for i in range(U):
                    with txl.if_((wv[i] >> txl.uint64(32)) != hi), txl.then_():
                        txl.assign(pend, txl.int32(1))
                with txl.if_(pend == 1), txl.then_():
                    txl.cuda.nano_sleep(txl.uint32(TR_SLEEP))
                    for i in range(U):
                        j = base + tl * U + i
                        with txl.if_((wv[i] >> txl.uint64(32)) != hi), txl.then_():
                            txl.ptx["ld.relaxed.gpu.global.b64"](wv[i], part.ptr_to([txl.cast(f + j, "int64") * H + txl.cast(col, "int64")]))
                    txl.assign(polls, polls + 1)
            for i in range(U):
                v = txl.reinterpret(txl.cast(wv[i] & txl.uint64(0xFFFFFFFF), "uint32"), ty="float32")
                v = txl.Select(polls < POLL_LIMIT, v, txl.reinterpret(txl.uint32(0x7FC00000), ty="float32"))
                if LANES > 1:
                    txl.ptx.st.shared.f32(buf.ptr_to([(tl * U + i) * COLS + cl]), v)
                else:
                    txl.assign(tmp[i], v)
            if LANES > 1:
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NTH))
                with txl.if_(tl == 0), txl.then_():
                    for i in range(TB):
                        txl.ptx.ld.shared.f32(tmp[i], buf.ptr_to([i * COLS + cl]))
                    for i in range(TB):
                        txl.assign(acc[0], acc[0] + tmp[i])
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NTH))
            else:
                for i in range(TB):
                    txl.assign(acc[0], acc[0] + tmp[i])
        with txl.if_(txl.And(tl == 0, col < H)), txl.then_():
            store_vec(out, txl.cast(g, "int64") * H + txl.cast(col, "int64"), odtype, 1, acc)

    return tile_reduce_tag


@functools.cache
def compiled(which, *key):
    kern = {
        "sum": tile_sum2,
        "ring": tile_sum_ring2,
        "group": group_sum_ring2,
        "reduce": tile_reduce2,
        "reduce_wide": tile_reduce_wide2,
        "reduce_tag": tile_reduce_tag2,
    }[which](*key)
    # the reduce kernels ("reduce", "reduce_wide", "reduce_tag") declare programmatic dependent launch in
    # their LaunchConfig: resident while the ring streams
    return entry_function(kern.compile(TARGET))


def segment_sum_sm100(x: torch.Tensor, grouped_mm_offs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Per-group column sums of the group-sorted rows of x (see the picker's contract), as a new (G, H)
    tensor of out_dtype.
    """
    assert x.is_contiguous(), "x must be contiguous"
    M, H = x.shape
    G = grouped_mm_offs.shape[0]
    out = torch.empty((G, H), dtype=out_dtype, device=x.device)
    if G == 0 or H == 0:
        return out
    dt, it, ot = DT[x.dtype], IT[grouped_mm_offs.dtype], DT[out_dtype]
    ring = ring_layout(dt, H)
    xf = x.view(-1) if M > 0 else torch.empty(1, dtype=x.dtype, device=x.device)
    # Many groups: one CTA per (group, chunk) accumulates the group's tiles in order (no second kernel) when those
    # CTAs fit one wave (two per SM) or the groups are short; otherwise long, uneven groups would leave a tail
    # and the compact tile ring + ordered reduce balances better.
    if M > 0 and ring is not None and G >= DMIN:
        nc_direct = group_ring_layout(dt, H)[2]
        if G * nc_direct <= 2 * sm_count() or M <= SHORT_ROWS * G:
            compiled("group", dt, it, ot, H)(xf, grouped_mm_offs, out.view(-1), M, G)
            return out
    slots = G + cdiv(M, TILE)
    if M > 0 and ring is not None and G <= GS:
        # compact ring publishes tagged tile words; the reduce (PDL, resident early) starts on each chain's last word
        part, tag = tagged_scratch(slots * H + 1, x.device)  # + the pool counter
        cfg = TR_LONG if M >= LONG_TILES * TILE * G else TR_SHORT
        n_units = slots * ring[2]  # upper bound (the kernel schedules only real tiles)
        compiled("ring", dt, it, H, True)(xf, grouped_mm_offs, part.view(-1), M, G, n_units, min(n_units, CPS * sm_count()), tag)
        compiled("reduce_tag", it, ot, H, *cfg)(part.view(-1), grouped_mm_offs, out.view(-1), M, G, tag)
        return out
    part = torch.empty((max(slots, 1), H), dtype=torch.float32, device=x.device)
    # long groups: smem-staged chunked reduce; short groups: one thread per 2 columns
    k_red = compiled("reduce" if M >= LONG_TILES * TILE * G else "reduce_wide", it, ot, H)
    if M > 0:
        vec, _ = vec2(dt, H)
        compiled("sum", dt, it, H)(xf, grouped_mm_offs, part.view(-1), M, G, slots * layout(H, vec)[1])
    k_red(part.view(-1), grouped_mm_offs, out.view(-1), M, G)
    return out
