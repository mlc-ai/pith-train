"""
cross_entropy_sm100: the fused mean cross-entropy forward and backward on SM100 (the logits
overwritten in place with the gradient of the mean), TIRx-lite kernels.

Design (cluster-resident rows):
- ce_count: one CTA counts the non-ignored targets, writes inv_n = 1 / max(n, 1) and zeroes the
  row-completion counter and the loss.
- ce_cluster: a cluster of C CTAs owns one row; CTA r bulk-copies (TMA) its slice of the row into
  shared memory once, computes a per-thread (max, sum exp) pair, reduces it over the CTA and then
  over the cluster through DSMEM, and writes (softmax - onehot) * inv_n straight from shared memory.
  Every logit is read from HBM once and written once. The CTA that records the last row loss sums
  all row losses in a fixed order (deterministic) and writes the mean.
- ce_stream: generic fallback (row bytes not a multiple of 16, a base not 16-byte aligned, or no
  cluster size that fits): one CTA per row, two streaming passes, then ce_final sums the per-row
  losses in a fixed order and scales by inv_n.
bf16, fp16 and fp32 logits with int32 or int64 targets run the kernels; other integer targets are
widened to int64 first, and other logit dtypes take a PyTorch path with the same contract.

Evolved with TIRx-harness: run cross_entropy, frontier member cluster-resident-tma (commit fce2df20,
10880 us vs 19448 us, 1.788x over the previous SM100 path, the Triton kernel adapted from
TransformerEngine, in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools

import torch
import torch.nn.functional as F
import tvm

import tirx_kernels.tirx_lite as txl

__all__ = ["cross_entropy_fwd_bwd_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
FULL = 0xFFFFFFFF
NEG_BIG = -3.0e38

TDT = {torch.int64: txl.i64, torch.int32: txl.i32}
LDT = {torch.bfloat16: txl.bf16, torch.float16: txl.f16, torch.float32: txl.f32}
ESZ = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}

CLUSTER_WARPS = 8
MAX_CLUSTER = 16
SMEM_PER_SM = 233472
CTA_OVERHEAD = 1024 + 512  # driver reserve + this kernel's barriers/slots


def blocks_per_sm(cap_bytes):
    return min(SMEM_PER_SM // (cap_bytes + CTA_OVERHEAD), 8)


def ld_target(t, target, row, tstride, tdt):
    if tdt == txl.i64:
        txl.ptx.ld.global_.s64(t, target.ptr_to([txl.Cast("int64", row) * tstride]))
    else:
        t32 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(t32, target.ptr_to([txl.Cast("int64", row) * tstride]))
        txl.assign(t, txl.Cast("int64", t32))


def exp2(x):
    d = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(d, x)
    return d


def lg2(x):
    d = txl.local_scalar("float32")
    txl.ptx.lg2.approx.ftz.f32(d, x)
    return d


def shfl_bfly_f32(v, delta):
    peer = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret("uint32", v), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL))
    return txl.reinterpret("float32", peer)


def shfl_idx_f32(v, lane):
    peer = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(peer, txl.reinterpret("uint32", v), txl.uint32(lane), txl.uint32(31), txl.uint32(FULL))
    return txl.reinterpret("float32", peer)


def merge(m, s, m2, s2):
    """
    (m, s) <- (m, s) (+) (m2, s2) for running (max, sum exp(x - max)).
    """
    mn = txl.local_scalar("float32", init=txl.max(m, m2))
    txl.assign(s, s * exp2((m - mn) * LOG2E) + s2 * exp2((m2 - mn) * LOG2E))
    txl.assign(m, mn)


def warp_merge(m, s):
    """
    Butterfly (max, sum) over the warp; every lane then takes lane 0's pair (deterministic).
    """
    for delta in (16, 8, 4, 2, 1):
        m2, s2 = txl.local_scalar("float32", init=shfl_bfly_f32(m, delta)), txl.local_scalar("float32", init=shfl_bfly_f32(s, delta))
        merge(m, s, m2, s2)
    txl.assign(m, shfl_idx_f32(m, 0))
    txl.assign(s, shfl_idx_f32(s, 0))


def block_merge(m, s, red, warps, lane, warp):
    """
    (max, sum) over the CTA through shared slots red[0:2*warps]; result in every thread.
    """
    warp_merge(m, s)
    if warps == 1:
        return
    with txl.If(lane == 0), txl.Then():
        txl.ptx.st.shared.f32(red.ptr_to([2 * warp]), m)
        txl.ptx.st.shared.f32(red.ptr_to([2 * warp + 1]), s)
    txl.cuda.cta_sync()
    with txl.If(lane < warps):
        with txl.Then():
            txl.ptx.ld.shared.f32(m, red.ptr_to([2 * lane]))
            txl.ptx.ld.shared.f32(s, red.ptr_to([2 * lane + 1]))
        with txl.Else():
            txl.assign(m, txl.float32(NEG_BIG))
            txl.assign(s, txl.float32(0.0))
    warp_merge(m, s)


def to_f32(dst, words, nwords, ldt):
    """
    Unpack nwords 32-bit words of ldt logits into dst[0 : elems].
    """
    if ldt == txl.f32:
        for k in range(nwords):
            txl.assign(dst[k], txl.reinterpret("float32", words[k]))
    elif ldt == txl.bf16:
        for k in range(nwords):
            txl.assign(dst[2 * k], txl.reinterpret("float32", words[k] << txl.uint32(16)))
            txl.assign(dst[2 * k + 1], txl.reinterpret("float32", words[k] & txl.uint32(0xFFFF0000)))
    else:
        for k in range(nwords):
            txl.idioms.cast_f16x2_to_f32x2(dst, k, words[k])


def from_f32(words, src, nwords, ldt):
    if ldt == txl.f32:
        for k in range(nwords):
            txl.assign(words[k], txl.reinterpret("uint32", src[k]))
    else:
        cvt = "cvt.rn.bf16x2.f32" if ldt == txl.bf16 else "cvt.rn.f16x2.f32"
        for k in range(nwords):
            txl.ptx[cvt](words[k], src[2 * k + 1], src[2 * k])


@functools.cache
def count_kernel(tdt):
    U = 4  # target loads in flight per thread

    @txl.kernel(warps=32, grid=1)
    def ce_count(target: txl.gptr[tdt], tstride: txl.i64, N: txl.i32, ii: txl.i64, stats: txl.gptr[txl.f32], done: txl.gptr[txl.i32], loss: txl.gptr[txl.f32]):
        tid = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        smem = txl.smem_pool()
        red = smem.alloc([32], txl.i32)
        cnt = txl.local_scalar("int32", init=txl.int32(0))
        ts = [txl.local_scalar("int64", init=txl.Cast("int64", ii)) for _ in range(U)]
        with txl.serial((N + 1024 * U - 1) // (1024 * U)) as it:
            for u in range(U):
                r = (it * U + u) * 1024 + tid
                with txl.If(r < N), txl.Then():
                    ld_target(ts[u], target, r, tstride, tdt)
            for u in range(U):
                with txl.If(ts[u] != ii), txl.Then():
                    txl.assign(cnt, cnt + 1)
                txl.assign(ts[u], txl.Cast("int64", ii))
        w = txl.local_scalar("uint32")
        txl.ptx.redux_sync.add.u32(w, txl.reinterpret("uint32", cnt), txl.uint32(FULL))
        with txl.If(lane == 0), txl.Then():
            txl.ptx.st.shared.s32(red.ptr_to([warp]), txl.reinterpret("int32", w))
        txl.cuda.cta_sync()
        with txl.If(warp == 0), txl.Then():
            v = txl.local_scalar("int32")
            txl.ptx.ld.shared.s32(v, red.ptr_to([lane]))
            txl.ptx.redux_sync.add.u32(w, txl.reinterpret("uint32", v), txl.uint32(FULL))
            with txl.If(lane == 0), txl.Then():
                nf = txl.Cast("float32", txl.max(txl.reinterpret("int32", w), txl.int32(1)))
                txl.ptx.st.global_.f32(stats.ptr_to([0]), txl.float32(1.0) / nf)
                # The main kernel's last row to finish sums the row losses into loss (0 if N == 0).
                txl.ptx.st.global_.s32(done.ptr_to([0]), txl.int32(0))
                txl.ptx.st.global_.f32(loss.ptr_to([0]), txl.float32(0.0))

    return entry_function(ce_count.compile())


@functools.cache
def final_kernel():
    @txl.kernel(warps=32, grid=1)
    def ce_final(row_loss: txl.gptr[txl.f32], stats: txl.gptr[txl.f32], loss: txl.gptr[txl.f32], N: txl.i32):
        tid = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        smem = txl.smem_pool()
        red = smem.alloc([32], txl.f32)
        acc, v = txl.local_scalar("float32", init=txl.float32(0.0)), txl.local_scalar("float32")
        with txl.serial((N + 1023) // 1024) as it:
            r = it * 1024 + tid
            with txl.If(r < N), txl.Then():
                txl.ptx.ld.global_.f32(v, row_loss.ptr_to([r]))
                txl.assign(acc, acc + v)
        for delta in (16, 8, 4, 2, 1):
            txl.assign(acc, acc + shfl_bfly_f32(acc, delta))
        with txl.If(lane == 0), txl.Then():
            txl.ptx.st.shared.f32(red.ptr_to([warp]), acc)
        txl.cuda.cta_sync()
        with txl.If(warp == 0), txl.Then():
            txl.ptx.ld.shared.f32(acc, red.ptr_to([lane]))
            for delta in (16, 8, 4, 2, 1):
                txl.assign(acc, acc + shfl_bfly_f32(acc, delta))
            with txl.If(lane == 0), txl.Then():
                txl.ptx.ld.global_.f32(v, stats.ptr_to([0]))
                txl.ptx.st.global_.f32(loss.ptr_to([0]), acc * v)

    return entry_function(ce_final.compile())


@functools.cache
def stream_kernel(ldt, tdt):
    """
    Generic fallback: one CTA per row, scalar loads, two passes over global memory.
    """
    warps = 16
    nt = warps * 32
    BITS = {txl.bf16: 16, txl.f16: 16, txl.f32: 32}[ldt]

    def load_f32(inp, idx):
        if BITS == 32:
            x = txl.local_scalar("float32")
            txl.ptx.ld.global_.f32(x, inp.ptr_to([idx]))
            return x
        u = txl.local_scalar("uint16")
        txl.ptx.ld.global_.b16(u, inp.ptr_to([idx]))
        if ldt == txl.bf16:
            return txl.reinterpret("float32", txl.Cast("uint32", u) << txl.uint32(16))
        return txl.Cast("float32", txl.reinterpret("float16", u))

    def store(inp, idx, g):
        if BITS == 32:
            txl.ptx.st.global_.f32(inp.ptr_to([idx]), g)
        else:
            txl.ptx.st.global_.b16(inp.ptr_to([idx]), txl.reinterpret("uint16", txl.Cast(ldt, g)))

    @txl.kernel(warps=warps, grid="N")
    def ce_stream(inp: txl.gptr[ldt], target: txl.gptr[tdt], tstride: txl.i64, stats: txl.gptr[txl.f32], row_loss: txl.gptr[txl.f32], N: txl.i32, V: txl.i32, ii: txl.i64):
        row = txl.cta_id()
        tid = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        smem = txl.smem_pool()
        red = smem.alloc([2 * warps], txl.f32)
        t = txl.local_scalar("int64")
        ld_target(t, target, row, tstride, tdt)
        base = txl.Cast("int64", row) * txl.Cast("int64", V)
        niter = (V + nt - 1) // nt
        with txl.If(t == ii):
            with txl.Then():
                with txl.serial(niter) as it:
                    c = it * nt + tid
                    with txl.If(c < V), txl.Then():
                        store(inp, base + txl.Cast("int64", c), txl.float32(0.0))
                with txl.If(tid == 0), txl.Then():
                    txl.ptx.st.global_.f32(row_loss.ptr_to([row]), txl.float32(0.0))
            with txl.Else():
                m, s = txl.local_scalar("float32", init=txl.float32(NEG_BIG)), txl.local_scalar("float32", init=txl.float32(0.0))
                with txl.serial(niter) as it:
                    c = it * nt + tid
                    with txl.If(c < V), txl.Then():
                        x = txl.local_scalar("float32", init=load_f32(inp, base + txl.Cast("int64", c)))
                        merge(m, s, x, txl.float32(1.0))
                block_merge(m, s, red, warps, lane, warp)
                inv_n = txl.local_scalar("float32")
                txl.ptx.ld.global_.f32(inv_n, stats.ptr_to([0]))
                scale = txl.local_scalar("float32", init=inv_n / s)
                with txl.serial(niter) as it:
                    c = it * nt + tid
                    with txl.If(c < V), txl.Then():
                        x = txl.local_scalar("float32", init=load_f32(inp, base + txl.Cast("int64", c)))
                        g = txl.local_scalar("float32", init=exp2((x - m) * LOG2E) * scale)
                        with txl.If(txl.Cast("int64", c) == t), txl.Then():
                            txl.assign(g, g - inv_n)
                            lg = lg2(s)
                            txl.ptx.st.global_.f32(row_loss.ptr_to([row]), m + lg * LN2 - x)
                        store(inp, base + txl.Cast("int64", c), g)

    return entry_function(ce_stream.compile())


def f2(op, a, b):
    """
    One packed f32x2 instruction (sub/mul/add/max), result as a b64 pair.
    """
    d = txl.local_scalar("uint64")
    txl.ptx[f"{op}.rn.ftz.f32x2"](d, a, b)
    return d


def pairs(words, ldt):
    """
    The float2 pairs of one 4-word vector: 4 pairs for 16-bit logits, 2 for fp32.
    """
    if ldt == txl.f32:
        return [txl.cuda.make_float2(txl.reinterpret("float32", words[2 * k]), txl.reinterpret("float32", words[2 * k + 1])) for k in range(2)]
    if ldt == txl.bf16:
        return [txl.cuda.make_float2(txl.reinterpret("float32", words[k] << txl.uint32(16)), txl.reinterpret("float32", words[k] & txl.uint32(0xFFFF0000))) for k in range(4)]
    out = []
    for k in range(4):
        tmp = txl.alloc_local([2], "float32")
        txl.idioms.cast_f16x2_to_f32x2(tmp, 0, words[k])
        out.append(txl.cuda.make_float2(tmp[0], tmp[1]))
    return out


def vec_max(words, nw, ldt):
    """
    Exact max over nw words of packed logits, as f32.
    """
    if ldt == txl.f32:
        acc = txl.local_scalar("float32", init=txl.reinterpret("float32", words[0]))
        for k in range(1, nw):
            txl.assign(acc, txl.max(acc, txl.reinterpret("float32", words[k])))
        return acc
    op = "max.bf16x2" if ldt == txl.bf16 else "max.f16x2"
    acc = txl.local_scalar("uint32", init=words[0])
    for k in range(1, nw):
        txl.ptx[op](acc, acc, words[k])
    tmp = txl.alloc_local([2], "float32")
    if ldt == txl.bf16:
        txl.assign(tmp[0], txl.reinterpret("float32", acc << txl.uint32(16)))
        txl.assign(tmp[1], txl.reinterpret("float32", acc & txl.uint32(0xFFFF0000)))
    else:
        txl.idioms.cast_f16x2_to_f32x2(tmp, 0, acc)
    return txl.max(tmp[0], tmp[1])


PAD = {txl.bf16: 0xFF80FF80, txl.f16: 0xFC00FC00, txl.f32: 0xFF800000}  # -inf lanes


def ld_logit(inp, idx, ldt):
    """
    One logit from global memory as f32.
    """
    x = txl.local_scalar("float32")
    if ldt == txl.f32:
        txl.ptx.ld.global_.f32(x, inp.ptr_to([idx]))
        return x
    u = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(u, inp.ptr_to([idx]))
    if ldt == txl.bf16:
        txl.assign(x, txl.reinterpret("float32", txl.Cast("uint32", u) << txl.uint32(16)))
    else:
        txl.ptx.cvt.f32.f16(x, u)
    return x


def cluster_func(ldt, tdt, C, CAP_BYTES, warps=CLUSTER_WARPS):
    """
    C CTAs per row; each holds up to CAP bytes of the row in shared memory, loaded in chunks.
    """
    nt = warps * 32
    esz = 4 if ldt == txl.f32 else 2
    cap = CAP_BYTES // esz  # elements per CTA
    epv = 16 // esz  # elements per 16-byte vector
    VPT = 2  # vectors per thread per chunk
    chunk_vec = nt * VPT
    chunk_el = chunk_vec * epv
    chunk_bytes = chunk_vec * 16
    nchunk = -(-CAP_BYTES // chunk_bytes)  # the last chunk may be partial
    assert CAP_BYTES % 16 == 0
    viters = nchunk * VPT
    L2 = txl.cuda.make_float2(txl.float32(LOG2E), txl.float32(LOG2E))

    @txl.kernel(warps=warps, grid=lambda p: p["N"] * C, min_blocks_per_sm=blocks_per_sm(CAP_BYTES))
    def ce_cluster(inp: txl.gptr[ldt], target: txl.gptr[tdt], tstride: txl.i64, stats: txl.gptr[txl.f32], row_loss: txl.gptr[txl.f32], N: txl.i32, V: txl.i32, ii: txl.i64, done: txl.gptr[txl.i32], loss: txl.gptr[txl.f32]):
        cta = txl.cta_id()
        rank = txl.cta_id_in_cluster([C]) if C > 1 else txl.int32(0)
        # Rows run last-to-first: the restore copy (or producer) wrote the tail last, so it is
        # the part most likely still resident in L2.
        row = N - 1 - cta // C
        tid = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        smem = txl.smem_pool()
        buf, red, slots, last = smem.alloc([cap], ldt, align=128), smem.alloc([2 * warps], txl.f32, align=16), smem.alloc([2 * MAX_CLUSTER], txl.f32, align=16), smem.alloc([1], txl.i32, align=4)
        bar, sbar = txl.MBarrier(smem, nchunk), txl.MBarrier(smem, 1)
        bar.init(1)
        sbar.init(1)
        with txl.If(tid == 0), txl.Then():
            txl.ptx.st.shared.s32(last.ptr_to([0]), txl.int32(0))
        if C > 1:
            with txl.If(tid == 0), txl.Then():
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(sbar.ptr_to([0]), txl.uint32(8 * C))
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.cuda.cta_sync()
        if C > 1:
            # Peers push their (max, sum) into our slots; their barriers must be initialized first.
            txl.ptx.barrier.cluster.arrive.relaxed()

        per = txl.local_scalar("int32", init=((V + C * 8 - 1) // (C * 8)) * 8)
        lo = txl.local_scalar("int32", init=rank * per)
        cnt = txl.local_scalar("int32", init=txl.max(txl.min(V - lo, per), txl.int32(0)))
        nvec = cnt // epv
        base = txl.local_scalar("int64", init=txl.Cast("int64", row) * txl.Cast("int64", V) + txl.Cast("int64", lo))
        t = txl.local_scalar("int64")
        ld_target(t, target, row, tstride, tdt)
        tl = txl.local_scalar("int32", init=txl.Cast("int32", t - txl.Cast("int64", lo)))

        words = txl.alloc_local([4 * VPT], "uint32")

        with txl.If(t == ii):
            with txl.Then():
                for k in range(4):
                    txl.assign(words[k], txl.uint32(0))
                with txl.serial(viters) as it:
                    v = it * nt + tid
                    with txl.If(v < nvec), txl.Then():
                        txl.ptx["st.global.v4.b32"](inp.ptr_to([base + txl.Cast("int64", v * epv)]), words[0], words[1], words[2], words[3])
                with txl.If(txl.And(tid == 0, rank == 0)), txl.Then():
                    txl.ptx.st.global_.f32(row_loss.ptr_to([row]), txl.float32(0.0))
                    old = txl.local_scalar("int32")
                    txl.ptx.atom.acq_rel.gpu.global_.add.s32(old, done.ptr_to([0]), txl.int32(1))
                    txl.ptx.st.shared.s32(last.ptr_to([0]), txl.Select(old == N - 1, txl.int32(1), txl.int32(0)))
                if C > 1:
                    txl.ptx.barrier.cluster.wait()
            with txl.Else():
                with txl.If(tid == 0), txl.Then():
                    with txl.serial(nchunk) as k:
                        rem = cnt * esz - k * chunk_bytes
                        with txl.If(rem > 0), txl.Then():
                            sz = txl.Cast("uint32", txl.min(rem, txl.int32(chunk_bytes)))
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(bar.ptr_to([k]), sz)
                            txl.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](buf.ptr_to([k * chunk_el]), inp.ptr_to([base + txl.Cast("int64", k * chunk_el)]), sz, bar.ptr_to([k]))

                # Per-thread, per-chunk: chunk max -> running max mref[k]; e = exp(x - mref[k]) is summed
                # into s and written back in place (f16 for 16-bit logits, f32 for fp32), so the grad pass
                # needs no second exponential: grad = e * exp(mref[k] - M) * inv_n / S.
                m, s, xt = txl.local_scalar("float32", init=txl.float32(NEG_BIG)), txl.local_scalar("float32", init=txl.float32(0.0)), txl.local_scalar("float32", init=txl.float32(0.0))
                owner = txl.And(tl >= 0, tl < cnt)
                with txl.If(txl.And(tid == 0, owner)), txl.Then():
                    txl.assign(xt, ld_logit(inp, base + txl.Cast("int64", tl), ldt))
                ow = txl.alloc_local([4], "uint32")
                mref = []
                for k in range(nchunk):
                    mk = txl.local_scalar("float32", init=txl.float32(NEG_BIG))
                    mref.append(mk)
                    with txl.If(k * chunk_el < cnt), txl.Then():
                        bar.wait(k, 0)
                        for j in range(VPT):
                            v = k * chunk_vec + j * nt + tid
                            with txl.If(v < nvec):
                                with txl.Then():
                                    txl.ptx["ld.shared.v4.b32"](words[4 * j], words[4 * j + 1], words[4 * j + 2], words[4 * j + 3], buf.ptr_to([v * epv]))
                                with txl.Else():
                                    for q in range(4):
                                        txl.assign(words[4 * j + q], txl.uint32(PAD[ldt]))
                        cm = vec_max(words, 4 * VPT, ldt)
                        mn = txl.local_scalar("float32", init=txl.max(m, cm))
                        txl.assign(s, s * exp2((m - mn) * LOG2E))
                        txl.assign(m, mn)
                        txl.assign(mk, mn)
                        mm = txl.cuda.make_float2(mn, mn)
                        acc = txl.local_scalar("uint64", init=txl.reinterpret("uint64", txl.uint64(0)))
                        for j in range(VPT):
                            prs = pairs([words[4 * j + q] for q in range(4)], ldt)
                            for q, pr in enumerate(prs):
                                d = f2("mul", f2("sub", pr, mm), L2)
                                ex, ey = txl.local_scalar("float32", init=exp2(txl.cuda.float2_x(d))), txl.local_scalar("float32", init=exp2(txl.cuda.float2_y(d)))
                                txl.ptx["add.rn.ftz.f32x2"](acc, acc, txl.cuda.make_float2(ex, ey))
                                if ldt == txl.f32:
                                    txl.assign(ow[2 * q], txl.reinterpret("uint32", ex))
                                    txl.assign(ow[2 * q + 1], txl.reinterpret("uint32", ey))
                                else:
                                    txl.ptx["cvt.rn.f16x2.f32"](ow[q], ey, ex)
                            v = k * chunk_vec + j * nt + tid
                            with txl.If(v < nvec), txl.Then():
                                txl.ptx["st.shared.v4.b32"](buf.ptr_to([v * epv]), ow[0], ow[1], ow[2], ow[3])
                        txl.assign(s, s + txl.cuda.float2_x(acc) + txl.cuda.float2_y(acc))
                block_merge(m, s, red, warps, lane, warp)

                if C > 1:
                    txl.ptx.barrier.cluster.wait()
                    # Lane r of warp 0 pushes this CTA's pair into slot [rank] of peer r.
                    with txl.If(txl.And(warp == 0, lane < C)), txl.Then():
                        ra, rb = txl.local_scalar("uint32"), txl.local_scalar("uint32")
                        txl.ptx.mapa.shared__cluster.u32(ra, txl.cuda.cvta_generic_to_shared(slots.ptr_to([2 * rank])), txl.Cast("uint32", lane))
                        txl.ptx.mapa.shared__cluster.u32(rb, txl.cuda.cvta_generic_to_shared(sbar.ptr_to([0])), txl.Cast("uint32", lane))
                        txl.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v2.f32(ra, m, s, rb)
                    txl.cuda.mbarrier_wait_acquire_cluster(sbar.ptr_to([0]), 0)
                    # Every thread merges the C pairs in rank order: identical in all CTAs of the row.
                    txl.ptx.ld.shared.f32(m, slots.ptr_to([0]))
                    txl.ptx.ld.shared.f32(s, slots.ptr_to([1]))
                    for r in range(1, C):
                        m2, s2 = txl.local_scalar("float32"), txl.local_scalar("float32")
                        txl.ptx.ld.shared.f32(m2, slots.ptr_to([2 * r]))
                        txl.ptx.ld.shared.f32(s2, slots.ptr_to([2 * r + 1]))
                        merge(m, s, m2, s2)

                inv_n = txl.local_scalar("float32")
                txl.ptx.ld.global_.f32(inv_n, stats.ptr_to([0]))
                scale = txl.local_scalar("float32", init=inv_n / s)
                with txl.If(txl.And(tid == 0, owner)), txl.Then():
                    txl.ptx.st.global_.f32(row_loss.ptr_to([row]), m + lg2(s) * LN2 - xt)
                    old = txl.local_scalar("int32")
                    txl.ptx.atom.acq_rel.gpu.global_.add.s32(old, done.ptr_to([0]), txl.int32(1))
                    txl.ptx.st.shared.s32(last.ptr_to([0]), txl.Select(old == N - 1, txl.int32(1), txl.int32(0)))
                vt = txl.local_scalar("int32", init=txl.Select(owner, tl // epv, txl.int32(-1)))
                g = txl.alloc_local([epv], "float32")
                for k in range(nchunk):
                    with txl.If(k * chunk_el < cnt), txl.Then():
                        f = txl.local_scalar("float32", init=exp2((mref[k] - m) * LOG2E) * scale)
                        ff = txl.cuda.make_float2(f, f)
                        for j in range(VPT):
                            v = k * chunk_vec + j * nt + tid
                            with txl.If(v < nvec), txl.Then():
                                txl.ptx["ld.shared.v4.b32"](words[0], words[1], words[2], words[3], buf.ptr_to([v * epv]))
                                for q, pr in enumerate(pairs(words, txl.f32 if ldt == txl.f32 else txl.f16)):
                                    r2 = f2("mul", pr, ff)
                                    txl.assign(g[2 * q], txl.cuda.float2_x(r2))
                                    txl.assign(g[2 * q + 1], txl.cuda.float2_y(r2))
                                with txl.If(v == vt), txl.Then():
                                    rel = tl - v * epv
                                    for q in range(epv):
                                        with txl.If(rel == q), txl.Then():
                                            txl.assign(g[q], g[q] - inv_n)
                                from_f32(words, g, 4, ldt)
                                txl.ptx["st.global.v4.b32"](inp.ptr_to([base + txl.Cast("int64", v * epv)]), words[0], words[1], words[2], words[3])
        # The CTA that recorded the last row loss sums them all in a fixed order (deterministic).
        txl.cuda.cta_sync()
        flag = txl.local_scalar("int32")
        txl.ptx.ld.shared.s32(flag, last.ptr_to([0]))
        with txl.If(flag != 0), txl.Then():
            txl.ptx.fence.acq_rel.gpu()
            acc, lv = txl.local_scalar("float32", init=txl.float32(0.0)), txl.local_scalar("float32")
            with txl.serial((N + nt - 1) // nt) as it:
                r = it * nt + tid
                with txl.If(r < N), txl.Then():
                    txl.ptx.ld.relaxed.gpu.global_.f32(lv, row_loss.ptr_to([r]))
                    txl.assign(acc, acc + lv)
            for delta in (16, 8, 4, 2, 1):
                txl.assign(acc, acc + shfl_bfly_f32(acc, delta))
            with txl.If(lane == 0), txl.Then():
                txl.ptx.st.shared.f32(red.ptr_to([warp]), acc)
            txl.cuda.cta_sync()
            with txl.If(warp == 0), txl.Then():
                with txl.If(lane < warps):
                    with txl.Then():
                        txl.ptx.ld.shared.f32(acc, red.ptr_to([lane]))
                    with txl.Else():
                        txl.assign(acc, txl.float32(0.0))
                for delta in (16, 8, 4, 2, 1):
                    txl.assign(acc, acc + shfl_bfly_f32(acc, delta))
                with txl.If(lane == 0), txl.Then():
                    sc = txl.local_scalar("float32")
                    txl.ptx.ld.global_.f32(sc, stats.ptr_to([0]))
                    txl.ptx.st.global_.f32(loss.ptr_to([0]), acc * sc)

    func = ce_cluster.func
    if C > 1:
        func = func.with_attr("tirx.launch_tags", ["blockIdx.x", "clusterCtaIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"])
    return func


@functools.cache
def cluster_kernel(ldt, tdt, C, cap, warps=CLUSTER_WARPS):
    func = cluster_func(ldt, tdt, C, cap, warps)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    return entry_function(tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx"))


def pick_cluster(V, esz):
    """
    (C, cap bytes): the smallest cluster whose exact slices fit 5 CTAs per SM if that needs <= 10
    CTAs, else the smallest fitting 4 per SM, else 3.

    Measured on B200 (N=4096, bf16, kernel time vs the D2D copy time): V=248320 C=9 at 4/SM 1.03x
    copy (C=8 at 3/SM 1.18x, C=11 at 5/SM 1.07x, C>=12 worse); V=151936 C=7 at 5/SM 1.02x; V=102400
    C=5 at 5/SM 0.98x; V=201088 C=9/10 at 5/SM 1.00x. Clusters of 11+ CTAs and 6-7 CTAs/SM lose.
    """
    if (V * esz) % 16 != 0:
        return None
    for want, cmax in ((5, 10), (4, MAX_CLUSTER), (3, MAX_CLUSTER)):
        for C in range(1, cmax + 1):
            cap = -(-V // (C * 8)) * 8 * esz
            if blocks_per_sm(cap) >= want:
                return C, cap
    return None


def cross_entropy_torch(inp: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """
    The contract in PyTorch for logit dtypes the kernels do not take: the fp32 mean loss, the
    gradient of that mean written into the row-contiguous inp.
    """
    x, keep = inp.float(), target != ignore_index
    n = keep.sum().clamp(min=1).float()
    loss = F.cross_entropy(x, target.long(), ignore_index=ignore_index, reduction="sum") / n
    grad = torch.softmax(x, dim=-1)
    grad.scatter_add_(1, torch.where(keep, target, 0).long().unsqueeze(1), -keep.float().unsqueeze(1))
    inp.copy_(grad.mul_(keep.unsqueeze(1) / n))
    return loss


def cross_entropy_fwd_bwd_sm100(inp: torch.Tensor, target: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cross-entropy of the (N, V) logits inp against target, overwriting the logits (or the
    contiguous copy a non-row-contiguous inp is replaced by) with the gradient of that mean. Returns
    (loss, grad), the same contract as cross_entropy_fwd_bwd_sm90.
    """
    N, V = inp.shape
    if inp.stride(-1) != 1 or inp.stride(-2) != V:
        inp = inp.contiguous()
    if inp.dtype not in LDT:
        return cross_entropy_torch(inp, target, ignore_index), inp
    if target.dtype not in TDT:  # other integer targets: the same values as int64
        target = target.long()
    dev = inp.device
    tstride = target.stride(0) if target.dim() == 1 else 1
    # A 1-D view over target's storage (metadata only) so any stride reaches the kernel.
    tflat = target.as_strided(((N - 1) * tstride + 1 if N > 0 else 1,), (1,))
    tdt, ldt = TDT[target.dtype], LDT[inp.dtype]
    stats, done, row_loss, loss = torch.empty(2, dtype=torch.float32, device=dev), torch.empty(1, dtype=torch.int32, device=dev), torch.empty(max(N, 1), dtype=torch.float32, device=dev), torch.empty((), dtype=torch.float32, device=dev)
    flat = inp.view(-1)
    count_kernel(tdt)(tflat, tstride, N, ignore_index, stats, done, loss.view(1))
    if N > 0:
        # TMA bulk copies and 16-byte vector stores need a 16-byte aligned base; otherwise ce_stream
        pick = pick_cluster(V, ESZ[inp.dtype]) if inp.data_ptr() % 16 == 0 else None
        if pick is None:
            stream_kernel(ldt, tdt)(flat, tflat, tstride, stats, row_loss, N, V, ignore_index)
            final_kernel()(row_loss, stats, loss.view(1), N)
        else:
            cluster_kernel(ldt, tdt, *pick)(flat, tflat, tstride, stats, row_loss, N, V, ignore_index, done, loss.view(1))
    return loss, inp
