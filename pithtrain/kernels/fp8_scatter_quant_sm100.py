"""
scatter_quant_sm100 in TIRx-lite: one cooperative persistent kernel (global route map + streaming quant).

Grid = all co-resident CTAs (2 per SM), launched cooperatively; three grid syncs separate the phases:
1. count: CTA c histograms its slice of the slots (match.any per warp, red.shared into per-warp histograms;
   (expert, source row, match mask) cached in the idle stage ring) and writes its per-group totals.
2. prefix: the CTA owning group g scans the per-CTA counts of g (exclusive) and its total.
3. rank: every CTA scans the padded group sizes (offs/ks from CTA 0), ranks its cached slots
   (row = start(g) + earlier CTAs + earlier warps + earlier steps + popc), writes reverse_shuffle_idxs and
   src_of_row[row]; group owners write -1 into their padding rows.
4. stream: each CTA owns an equal contiguous range of 128x128 tiles (<= KMAX row blocks), loads their
   src_of_row into smem and streams them through a 3-stage cp.async ring, takes 1x128 and 128x1 abs-maxima in
   bf16x2 and quantizes both layouts with exact power-of-two bf16x2 products and cvt.rn.satfinite.e4m3x2.bf16x2.
Without host counts, a small counting kernel and the one allowed device-to-host copy size the outputs.

Evolved with TIRx-harness: run scatter, frontier member coop-route-colorder (commit 3925531, 1178 us vs 6203 us, 5.267x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools
import inspect

import torch
import tvm
import tvm_ffi

import tirx_kernels.tirx_lite as txl

__all__ = ["scatter_quant_sm100"]

BLOCK = 128
FULL = 0xFFFFFFFF
MAXABS = "max.xorsign.abs.bf16x2"
WARPS = 8
NTHREADS = WARPS * 32
UNROLL = 4  # slot loads in flight per lane in the count pass
STAGES = 3
SM_COUNT = 148
CTAS_PER_SM = 2
KMAX = 8  # row blocks a CTA's static tile range may touch
DIST = 5  # dynamic tiles claimed ahead (claim, store, load: one iteration each)
RING = 8  # claimed dynamic tiles in flight: (row block | column << 20) and their slot lists
MAX_CACHE_STEPS = 16  # count-pass steps cached in the idle stage ring: warps x steps x 32 lanes x 3 words
COUNT_WARPS = 32
PF_UNITS = 2048  # 16-byte units per L2 prefetch (32 KiB)
PF_CAP = 0 << 20  # bytes of `gathered` warmed into L2 at entry (more only evicts itself)
SMEM_LIMIT = 113 << 10  # per CTA, for 2 CTAs per SM


def pad(x):
    """
    x rounded up to a multiple of 128, as int32.
    """
    return txl.shift_left(txl.shift_right(txl.Cast("int32", x) + 127, 7), 7)


@functools.cache
def count_kernel(gb):
    """
    Single CTA: per-expert counts, grouped_mm_offs and ks_tensor (G > 32 path).
    """
    nthreads = COUNT_WARPS * 32
    gpl = gb // 32

    @txl.kernel(warps=COUNT_WARPS, grid=1)
    def count(
        expert_idxs: txl.gptr[txl.i32],
        counts: txl.gptr[txl.i32],
        offs: txl.gptr[txl.i32],
        ks_t: txl.gptr[txl.i32],
        m: txl.i32,
        num_groups: txl.i32,
    ):
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        smem = txl.smem_pool()
        cnt = smem.alloc([gb], txl.u32)
        for k in range((gb + nthreads - 1) // nthreads):
            g = k * nthreads + tid
            with txl.If(g < gb), txl.Then():
                txl.ptx.st.shared.u32(cnt.ptr_to([g]), txl.uint32(0))
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(nthreads))
        e = txl.local_scalar("uint32")
        mask = txl.local_scalar("uint32")
        total = txl.local_scalar("uint32")
        leader = txl.local_scalar("uint32")
        old = txl.local_scalar("uint32")
        i = txl.local_scalar("int32", init=tid)
        with txl.While(i - lane < m):
            valid = i < m
            txl.assign(e, txl.uint32(FULL))
            with txl.If(valid), txl.Then():
                txl.ptx.ld.global_.nc.b32(e, expert_idxs.ptr_to([i * 2]))
            txl.ptx["match.any.sync.b32"](mask, e, txl.uint32(FULL))
            txl.ptx.popc.b32(total, mask)
            txl.ptx.bfind.u32(leader, mask)
            with txl.If(txl.And(valid, txl.Cast("uint32", lane) == leader)), txl.Then():
                txl.ptx.atom.shared.add.u32(old, cnt.ptr_to([e]), total)
            txl.assign(i, i + nthreads)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(nthreads))
        with txl.If(warp == 0), txl.Then():
            run = txl.local_scalar("int32", init=0)
            c = txl.local_scalar("uint32")
            for k in range(gpl):
                txl.ptx.ld.shared.u32(c, cnt.ptr_to([lane * gpl + k]))
                txl.assign(run, run + pad(c))
            incl = txl.local_scalar("int32", init=run)
            peer = txl.local_scalar("int32")
            for d in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
                with txl.If(lane >= d), txl.Then():
                    txl.assign(incl, incl + peer)
            acc = txl.local_scalar("int32", init=incl - run)
            for k in range(gpl):
                g = lane * gpl + k
                txl.ptx.ld.shared.u32(c, cnt.ptr_to([g]))
                with txl.If(g < num_groups), txl.Then():
                    txl.ptx.st.global_.s32(counts.ptr_to([g]), txl.Cast("int32", c))
                    txl.ptx.st.global_.s32(offs.ptr_to([g]), acc + pad(c))
                    txl.ptx.st.global_.s32(ks_t.ptr_to([g]), pad(c))
                txl.assign(acc, acc + pad(c))

    return count.compile()


@functools.cache
def fused_kernel(n, gb):
    func = fused_traced(n, gb).func.with_attr(
        "tirx.kernel_launch_params", ["blockIdx.x", "threadIdx.x", "tirx.use_cooperative_launch", "tirx.use_dyn_shared_memory"]
    )
    return tvm.compile(tvm.IRModule({"main": func}), target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}), tir_pipeline="tirx")


def smem_bytes(n, gb):
    return STAGES * BLOCK * BLOCK * 2 + (KMAX + RING) * BLOCK * 4 + 2 * WARPS * BLOCK * 2 + BLOCK * 4 + 3 * gb * 4 + 512


@functools.cache
def fused_traced(n, gb):
    vec = 8 if n % 8 == 0 else 4
    assert n % vec == 0, f"N={n} must be a multiple of 4"
    wpl = vec // 2  # bf16x2 words per lane per row
    lpr = BLOCK // vec  # lanes per row
    rpi = 32 // lpr  # rows per warp instruction
    rows_per_warp = BLOCK // WARPS
    iters = rows_per_warp // rpi
    nct = (n + BLOCK - 1) // BLOCK
    nd = max(1, (nct + 1) // 3)  # last column tiles handed out dynamically (tail balancing across SMs)
    ncs = nct - nd  # first column tiles assigned statically
    nw = n // 2  # words per bf16 row
    gpl = gb // 32
    stage_words = iters * NTHREADS * wpl
    # during the sort the idle stage ring holds the per-warp histograms (at its tail) and the step cache
    hist_off = STAGES * stage_words - WARPS * gb
    CACHE_STEPS = min(MAX_CACHE_STEPS, hist_off // (WARPS * 32 * 3))
    cp_insn = "cp.async.cg.shared.global" if vec == 8 else "cp.async.ca.shared.global"
    cp_bytes = vec * 2

    def body(gathered, experts, expand, offs, ks_t, reverse, x_fp8, x_scale, x_ch, x_ch_scale, ws, src_of_row,
             m, num_groups, ntiles, identity, num_ctas, gunits):
        cid = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        num_rb = txl.local_scalar("int32", init=ntiles // nct)
        nstat = num_rb * ncs  # static tiles (columns < ncs), split into equal contiguous ranges

        def tile_begin(p):
            return txl.Cast("int32", txl.Cast("int64", p) * nstat // num_ctas)

        tbeg = txl.local_scalar("int32", init=tile_begin(cid))
        tend = txl.local_scalar("int32", init=tile_begin(cid + 1))
        rbf = txl.local_scalar("int32", init=tbeg // max(ncs, 1))
        rbl = txl.local_scalar("int32", init=(tend - 1) // max(ncs, 1))

        smem = txl.smem_pool()
        stage = smem.alloc([STAGES * stage_words], txl.u32)
        tcnt = smem.alloc([gb], txl.i32)
        tstart = smem.alloc([gb], txl.i32)
        wsum = smem.alloc([gb], txl.i32)
        sl = smem.alloc([KMAX * BLOCK], txl.i32)
        dsl = smem.alloc([RING * BLOCK], txl.i32)
        qbuf = smem.alloc([RING], txl.i32)
        cm = smem.alloc([2 * WARPS * BLOCK // 2], txl.u32)
        cmf = smem.alloc([2 * BLOCK // 2], txl.u32, align=16)

        class Hist:  # per-warp group histograms, [warp * gb + g], at the tail of the idle stage ring
            @staticmethod
            def ptr_to(idx):
                return stage.ptr_to([hist_off + idx[0]])

        hist = Hist
        cache = stage

        def cidx(step, f):
            return ((warp * CACHE_STEPS + step) * 32 + lane) * 3 + f

        def cta_sync():
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))

        def iket_range(name):
            token = txl.alloc_local([1], "uint32")
            txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
            with txl.If(tid == 0), txl.Then():
                txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        def mark(name):
            with txl.If(tid == 0), txl.Then():
                txl.cuda.iket.mark(name)

        # workspace: cnt[g * num_ctas + c], pre[gb * num_ctas + g * num_ctas + c], tot[2 * gb * num_ctas + g]
        def ws_cnt(g, c):
            return ws.ptr_to([g * num_ctas + c])

        def ws_pre(g, c):
            return ws.ptr_to([gb * num_ctas + g * num_ctas + c])

        def ws_tot(g):
            return ws.ptr_to([2 * gb * num_ctas + g])

        tk_so = iket_range("sort")
        mark("entry")
        # ---- 1. count this CTA's slice of the slots
        chunk = txl.local_scalar("int32", init=(m + num_ctas * 32 - 1) // (num_ctas * 32) * 32)
        s0 = txl.local_scalar("int32", init=cid * chunk)
        nsteps = txl.local_scalar("int32", init=(chunk + NTHREADS - 1) // NTHREADS)
        ev = txl.alloc_local([UNROLL], "int32")
        sv = txl.alloc_local([UNROLL], "int32")
        pm = txl.alloc_local([UNROLL], "uint32")
        pmask = txl.local_scalar("uint32")
        total = txl.local_scalar("uint32")
        leader = txl.local_scalar("uint32")
        before = txl.local_scalar("uint32")
        lt = txl.shift_left(txl.uint32(1), txl.Cast("uint32", lane)) - txl.uint32(1)
        c = txl.local_scalar("int32")
        v = txl.local_scalar("int32")
        cval = txl.local_scalar("int32")

        def slot(t):
            return s0 + (t * WARPS + warp) * 32 + lane

        def in_range(t):
            i = slot(t)
            return txl.And(txl.And(t < nsteps, i < m), i < s0 + chunk)

        def load_steps(t0):
            for u in range(UNROLL):
                i = slot(t0 + u)
                txl.assign(ev[u], -1)
                txl.assign(sv[u], i)
                with txl.If(in_range(t0 + u)), txl.Then():
                    txl.ptx.ld.global_.nc.b32(ev[u], experts.ptr_to([i * 2]))
                    with txl.If(identity == 0), txl.Then():
                        txl.ptx.ld.global_.nc.b32(sv[u], expand.ptr_to([i * 2]))

        load_steps(0)
        # warm L2 with this CTA's share of `gathered` while the sort keeps DRAM idle
        with txl.If(warp == 0), txl.Then():
            pchunk = txl.local_scalar("int32", init=(gunits + num_ctas - 1) // num_ctas)
            pend = txl.local_scalar("int32", init=txl.min(cid * pchunk + pchunk, gunits))
            pu = txl.local_scalar("int32", init=cid * pchunk + lane * PF_UNITS)
            with txl.While(pu < pend):
                txl.ptx["cp.async.bulk.prefetch.L2.global"](
                    gathered.ptr_to([txl.Cast("int64", pu) * 4]),
                    txl.Cast("uint32", txl.min(pend - pu, PF_UNITS) * 16),
                )
                txl.assign(pu, pu + 32 * PF_UNITS)
        txl.assign(c, tid)
        with txl.While(c < WARPS * gb):
            txl.ptx.st.shared.s32(hist.ptr_to([c]), txl.int32(0))
            txl.assign(c, c + NTHREADS)
        cta_sync()
        t = txl.local_scalar("int32", init=0)
        with txl.While(t < nsteps):
            with txl.If(t > 0), txl.Then():
                load_steps(t)
            for u in range(UNROLL):
                txl.ptx["match.any.sync.b32"](pmask, ev[u], txl.uint32(FULL))
                txl.assign(pm[u], pmask)
                txl.ptx.popc.b32(total, pmask)
                txl.ptx.bfind.u32(leader, pmask)
                with txl.If(txl.And(ev[u] >= 0, txl.Cast("uint32", lane) == leader)), txl.Then():
                    txl.ptx.red.shared.add.u32(hist.ptr_to([warp * gb + ev[u]]), total)
                with txl.If(t + u < CACHE_STEPS), txl.Then():
                    txl.ptx.st.shared.s32(cache.ptr_to([cidx(t + u, 0)]), ev[u])
                    txl.ptx.st.shared.s32(cache.ptr_to([cidx(t + u, 1)]), sv[u])
                    txl.ptx.st.shared.u32(cache.ptr_to([cidx(t + u, 2)]), pm[u])
            txl.assign(t, t + UNROLL)
        cta_sync()
        txl.assign(c, tid)
        with txl.While(c < gb):
            txl.assign(v, 0)
            for wi in range(WARPS):
                txl.ptx.ld.shared.s32(before, hist.ptr_to([wi * gb + c]))
                txl.assign(v, v + txl.Cast("int32", before))
            txl.ptx.st.global_.s32(ws_cnt(c, cid), v)
            txl.assign(c, c + NTHREADS)
        mark("count-done")
        txl.cuda.grid_sync()
        mark("sync1")
        # ---- 2. per-group exclusive prefix over the CTAs (group g owned by CTA g mod num_ctas)
        x0 = txl.local_scalar("int32")
        x1 = txl.local_scalar("int32")
        incl = txl.local_scalar("int32")
        peer = txl.local_scalar("int32")
        g = txl.local_scalar("int32", init=cid)
        with txl.While(g < num_groups):
            txl.assign(x0, 0)
            txl.assign(x1, 0)
            with txl.If(tid * 2 < num_ctas), txl.Then():
                txl.ptx.ld.global_.b32(x0, ws_cnt(g, tid * 2))
            with txl.If(tid * 2 + 1 < num_ctas), txl.Then():
                txl.ptx.ld.global_.b32(x1, ws_cnt(g, tid * 2 + 1))
            txl.assign(incl, x0 + x1)
            for d in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
                with txl.If(lane >= d), txl.Then():
                    txl.assign(incl, incl + peer)
            with txl.If(lane == 31), txl.Then():
                txl.ptx.st.shared.s32(wsum.ptr_to([warp]), incl)
            cta_sync()
            txl.assign(v, 0)
            for wi in range(WARPS):
                txl.ptx.ld.shared.s32(cval, wsum.ptr_to([wi]))
                with txl.If(wi < warp), txl.Then():
                    txl.assign(v, v + cval)
                if wi == WARPS - 1:
                    pass
            # exclusive prefix of this thread's pair
            exc = txl.local_scalar("int32", init=v + incl - x0 - x1)
            with txl.If(tid * 2 < num_ctas), txl.Then():
                txl.ptx.st.global_.s32(ws_pre(g, tid * 2), exc)
            with txl.If(tid * 2 + 1 < num_ctas), txl.Then():
                txl.ptx.st.global_.s32(ws_pre(g, tid * 2 + 1), exc + x0)
            with txl.If(tid == NTHREADS - 1), txl.Then():
                txl.ptx.st.global_.s32(ws_tot(g), v + incl)
            cta_sync()
            txl.assign(g, g + num_ctas)
        mark("prefix-done")
        txl.cuda.grid_sync()
        mark("sync2")
        # ---- 3. group sizes, scan of the padded sizes, ranks
        txl.assign(c, tid)
        with txl.While(c < gb):
            txl.assign(cval, 0)
            with txl.If(c < num_groups), txl.Then():
                txl.ptx.ld.global_.b32(cval, ws_tot(c))
            txl.ptx.st.shared.s32(tcnt.ptr_to([c]), cval)
            txl.assign(c, c + NTHREADS)
        cta_sync()
        with txl.If(warp == 0), txl.Then():
            run0 = txl.local_scalar("int32", init=0)
            for k in range(gpl):
                txl.ptx.ld.shared.s32(cval, tcnt.ptr_to([lane * gpl + k]))
                txl.assign(run0, run0 + pad(cval))
            txl.assign(incl, run0)
            for d in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
                with txl.If(lane >= d), txl.Then():
                    txl.assign(incl, incl + peer)
            acc = txl.local_scalar("int32", init=incl - run0)
            for k in range(gpl):
                gg = lane * gpl + k
                txl.ptx.st.shared.s32(tstart.ptr_to([gg]), acc)
                txl.ptx.ld.shared.s32(cval, tcnt.ptr_to([gg]))
                with txl.If(txl.And(gg < num_groups, cid == 0)), txl.Then():
                    txl.ptx.st.global_.s32(offs.ptr_to([gg]), acc + pad(cval))
                    txl.ptx.st.global_.s32(ks_t.ptr_to([gg]), pad(cval))
                txl.assign(acc, acc + pad(cval))
        cta_sync()
        # padding rows of the owned groups -> -1
        txl.assign(g, cid)
        with txl.While(g < num_groups):
            txl.ptx.ld.shared.s32(cval, tcnt.ptr_to([g]))
            txl.ptx.ld.shared.s32(v, tstart.ptr_to([g]))
            with txl.If(tid < pad(cval) - cval), txl.Then():
                txl.ptx.st.global_.s32(src_of_row.ptr_to([v + cval + tid]), txl.int32(-1))
            txl.assign(g, g + num_ctas)
        # run[w][g] = start(g) + earlier CTAs + earlier warps
        txl.assign(c, tid)
        with txl.While(c < gb):
            txl.ptx.ld.shared.s32(v, tstart.ptr_to([c]))
            with txl.If(c < num_groups), txl.Then():
                txl.ptx.ld.global_.b32(cval, ws_pre(c, cid))
                txl.assign(v, v + cval)
            for wi in range(WARPS):
                txl.ptx.ld.shared.s32(before, hist.ptr_to([wi * gb + c]))
                txl.ptx.st.shared.s32(hist.ptr_to([wi * gb + c]), v)
                txl.assign(v, v + txl.Cast("int32", before))
            txl.assign(c, c + NTHREADS)
        cta_sync()
        e1 = txl.local_scalar("int32")
        s1 = txl.local_scalar("int32")
        run = txl.local_scalar("int32")
        row = txl.local_scalar("int32")
        row64 = txl.local_scalar("int64")
        txl.assign(t, 0)
        with txl.While(t < nsteps):
            with txl.If(t < CACHE_STEPS):
                with txl.Then():
                    txl.ptx.ld.shared.s32(e1, cache.ptr_to([cidx(t, 0)]))
                    txl.ptx.ld.shared.s32(s1, cache.ptr_to([cidx(t, 1)]))
                    txl.ptx.ld.shared.u32(pmask, cache.ptr_to([cidx(t, 2)]))
                with txl.Else():
                    i = slot(t)
                    txl.assign(e1, -1)
                    txl.assign(s1, i)
                    with txl.If(in_range(t)), txl.Then():
                        txl.ptx.ld.global_.nc.b32(e1, experts.ptr_to([i * 2]))
                        with txl.If(identity == 0), txl.Then():
                            txl.ptx.ld.global_.nc.b32(s1, expand.ptr_to([i * 2]))
                    txl.ptx["match.any.sync.b32"](pmask, e1, txl.uint32(FULL))
            txl.ptx.popc.b32(total, pmask)
            txl.ptx.bfind.u32(leader, pmask)
            txl.ptx.popc.b32(before, txl.bitwise_and(pmask, lt))
            with txl.If(e1 >= 0), txl.Then():
                txl.ptx.ld.shared.s32(run, hist.ptr_to([warp * gb + e1]))
            txl.ptx.bar.warp.sync(txl.uint32(FULL))
            with txl.If(e1 >= 0), txl.Then():
                txl.assign(row, run + txl.Cast("int32", before))
                with txl.If(txl.Cast("uint32", lane) == leader), txl.Then():
                    txl.ptx.st.shared.s32(hist.ptr_to([warp * gb + e1]), run + txl.Cast("int32", total))
                txl.assign(row64, txl.Cast("int64", row))
                txl.ptx.st.global_.s64(reverse.ptr_to([slot(t)]), row64)
                txl.ptx.st.global_.s32(src_of_row.ptr_to([row]), s1)
            txl.ptx.bar.warp.sync(txl.uint32(FULL))
            txl.assign(t, t + 1)
        # the dynamic tile queue's counter, published by the grid sync
        with txl.If(txl.And(cid == 0, tid == 0)), txl.Then():
            txl.ptx.st.global_.s32(ws.ptr_to([2 * gb * num_ctas + gb]), txl.int32(0))
        mark("rank-done")
        txl.cuda.grid_sync()
        mark("sync3")
        # this CTA's static slot lists
        for q in range(KMAX * BLOCK // NTHREADS):
            e = q * NTHREADS + tid
            with txl.If(txl.And(tend > tbeg, rbf + (e >> 7) <= rbl)), txl.Then():
                txl.ptx.ld.global_.b32(cval, src_of_row.ptr_to([(rbf + (e >> 7)) * BLOCK + (e & 127)]))
                txl.ptx.st.shared.s32(sl.ptr_to([e]), cval)
        # dynamic tiles: q -> column ncs + q // num_rb, row block q % num_rb (column-major: the whole GPU sweeps
        # the last columns in order)
        nst = txl.local_scalar("int32", init=tend - tbeg)
        ndyn = num_rb * nd
        qold = txl.local_scalar("int32")

        def claim_issue():  # thread 0; the atomic's latency is hidden until claim_store
            txl.ptx.atom.global_.add.s32(qold, ws.ptr_to([2 * gb * num_ctas + gb]), txl.int32(1))

        def claim_store(slot):  # thread 0
            cq_ = txl.local_scalar("int32", init=qold // num_rb)
            txl.ptx.st.shared.s32(
                qbuf.ptr_to([slot]),
                txl.Select(qold < ndyn, (qold - cq_ * num_rb) | txl.shift_left(cq_ + ncs, 20), txl.int32(-1)),
            )

        def load_slots(slot):  # 128 threads
            with txl.If(tid < BLOCK), txl.Then():
                txl.ptx.ld.shared.s32(v, qbuf.ptr_to([slot]))
                with txl.If(v >= 0), txl.Then():
                    txl.ptx.ld.global_.b32(cval, src_of_row.ptr_to([(v & 0xFFFFF) * BLOCK + tid]))
                    txl.ptx.st.shared.s32(dsl.ptr_to([slot * BLOCK + tid]), cval)

        # dynamic ordinals the in-loop pipeline cannot reach in time (short static ranges) are claimed up front
        npre = txl.local_scalar("int32", init=txl.max(DIST - nst, 0))
        with txl.If(tid == 0), txl.Then():
            for k in range(DIST):
                with txl.If(k < npre), txl.Then():
                    claim_issue()
                    claim_store(k)
        cta_sync()
        for k in range(DIST):
            with txl.If(k < npre), txl.Then():
                load_slots(k)
        cta_sync()
        txl.cuda.iket.range_end(tk_so[0])
        tk_st = iket_range("stream")
        # ---- 4. stream the column tiles
        cq = lane % lpr
        sub = lane // lpr
        srcs = txl.alloc_local([iters], "int32")

        def lrow(j):
            return warp * rows_per_warp + j * rpi + sub


        def stage_ptr(s, j):
            return stage.ptr_to([((s * iters + j) * NTHREADS + tid) * wpl])

        def issue(slbuf, base, ct, s):
            col = ct * BLOCK + cq * vec
            for j in range(iters):
                txl.ptx.ld.shared.s32(srcs[j], slbuf.ptr_to([base + lrow(j)]))
            for j in range(iters):
                ok = txl.And(srcs[j] >= 0, col < n)
                woff = txl.Cast("int64", txl.max(srcs[j], 0)) * nw + txl.Cast("int64", txl.min(col, n - vec) // 2)
                txl.ptx[cp_insn](
                    stage_ptr(s, j), gathered.ptr_to([woff]), cp_bytes,
                    txl.Select(ok, txl.uint32(cp_bytes), txl.uint32(0)),
                )

        # column-outer order over the static tiles: every CTA sweeps the column tiles at about the same pace, so
        # the slots of a token (same gathered row, different row blocks) read each 128-column piece of it
        # within a short window and the repeat reads hit L2; then the dynamic tiles
        def bmin(c_):
            return (tbeg - c_ + ncs - 1) // ncs

        def bmax(c_):
            return (tend - 1 - c_ + ncs) // ncs - 1

        def it_init(b_, c_):
            txl.assign(c_, 0)
            txl.assign(b_, bmin(0))
            with txl.While(txl.And(b_ > bmax(c_), c_ < ncs - 1)):
                txl.assign(c_, c_ + 1)
                txl.assign(b_, bmin(c_))

        def it_next(b_, c_):
            txl.assign(b_, b_ + 1)
            with txl.While(txl.And(b_ > bmax(c_), c_ < ncs - 1)):
                txl.assign(c_, c_ + 1)
                txl.assign(b_, bmin(c_))

        ib = txl.local_scalar("int32")
        ic = txl.local_scalar("int32")
        pb = txl.local_scalar("int32")
        pc = txl.local_scalar("int32")
        if ncs > 0:
            it_init(ib, ic)
            it_init(pb, pc)
        dq = txl.local_scalar("int32")
        pq = txl.local_scalar("int32")
        pslot = txl.local_scalar("int32", init=-1)  # claimed (atomic in flight), id not stored yet
        lslot = txl.local_scalar("int32", init=-1)  # id stored (published by the last bar 2), slot list not loaded

        def issue_seq(x, s):
            """
            issue the CTA's x-th tile (static first, then dynamic) into stage s
            """
            def dyn():
                txl.ptx.ld.shared.s32(dq, qbuf.ptr_to([(x - nst) & (RING - 1)]))
                with txl.If(dq >= 0), txl.Then():
                    issue(dsl, ((x - nst) & (RING - 1)) * BLOCK, dq >> 20, s)

            if ncs == 0:
                dyn()
                return
            with txl.If(x < nst):
                with txl.Then():
                    issue(sl, (ib - rbf) * BLOCK, ic, s)
                    it_next(ib, ic)
                with txl.Else():
                    dyn()

        for s in range(STAGES - 1):
            issue_seq(txl.int32(s), s)
            txl.ptx.cp.async_.commit_group()

        def proc_valid(x):  # 1 if the CTA's x-th tile exists (dynamic ones: claimed id >= 0)
            txl.assign(pq, 1)
            with txl.If(x >= nst), txl.Then():
                txl.ptx.ld.shared.s32(pq, qbuf.ptr_to([(x - nst) & (RING - 1)]))

        proc_valid(txl.int32(0))

        w = txl.alloc_local([iters * wpl], "uint32")
        cmax = txl.alloc_local([wpl], "uint32")
        rmax = txl.alloc_local([iters], "uint32")
        upeer = txl.local_scalar("uint32")
        crcp = txl.alloc_local([wpl], "uint32")
        cexp = txl.alloc_local([vec], "uint32")
        e16 = txl.local_scalar("uint16")
        rr = txl.local_scalar("uint32")
        y = txl.local_scalar("uint32")
        q16 = txl.alloc_local([wpl], "uint16")
        o = txl.alloc_local([max(wpl // 2, 1)], "uint32")

        def e8m0_pair(dst16, hi_amax, lo_amax):
            hs = txl.max(hi_amax, txl.float32(1e-4)) * txl.float32(1.0 / 448.0)
            ls = txl.max(lo_amax, txl.float32(1e-4)) * txl.float32(1.0 / 448.0)
            txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(dst16, hs, ls)

        def bf_lo(word):
            return txl.reinterpret("float32", txl.shift_left(word, txl.uint32(16)))

        def bf_hi(word):
            return txl.reinterpret("float32", txl.bitwise_and(word, txl.uint32(0xFFFF0000)))

        sendv = txl.local_scalar("uint32")
        rexp = txl.local_scalar("uint32")
        obase = txl.local_scalar("int64")
        kk = txl.local_scalar("int32", init=0)
        with txl.While(pq >= 0):
            rb = txl.local_scalar("int32")
            ct = txl.local_scalar("int32")
            if ncs > 0:
                with txl.If(kk < nst):
                    with txl.Then():
                        txl.assign(rb, pb)
                        txl.assign(ct, pc)
                        it_next(pb, pc)
                    with txl.Else():
                        txl.assign(rb, pq & 0xFFFFF)
                        txl.assign(ct, pq >> 20)
            else:
                txl.assign(rb, pq & 0xFFFFF)
                txl.assign(ct, pq >> 20)
            # dynamic claim pipeline, one step per iteration: atomic -> id into qbuf -> slot list into dsl
            with txl.If(lslot >= 0), txl.Then():
                load_slots(lslot)
                txl.assign(lslot, -1)
            with txl.If(pslot >= 0), txl.Then():
                with txl.If(tid == 0), txl.Then():
                    claim_store(pslot)
                txl.assign(lslot, pslot)
                txl.assign(pslot, -1)
            with txl.If(kk + DIST - nst >= npre), txl.Then():
                txl.assign(pslot, (kk + DIST - nst) & (RING - 1))
                with txl.If(tid == 0), txl.Then():
                    claim_issue()
            row0 = rb * BLOCK
            issue_seq(kk + STAGES - 1, (kk + STAGES - 1) % STAGES)
            txl.ptx.cp.async_.commit_group()
            txl.ptx.cp.async_.wait_group(STAGES - 1)
            s = kk % STAGES
            for j in range(iters):
                if wpl == 4:
                    txl.ptx.ld.shared.v4.b32(w[j * 4], w[j * 4 + 1], w[j * 4 + 2], w[j * 4 + 3], stage_ptr(s, j))
                else:
                    txl.ptx.ld.shared.v2.b32(w[j * 2], w[j * 2 + 1], stage_ptr(s, j))
            col = txl.local_scalar("int32", init=ct * BLOCK + cq * vec)
            col_ok = col < n
            # |max| over this thread's rows (columns) and over each row's 8/4 elements; the
            # xorsign.abs max keeps the magnitude max, its sign bit is dropped at the end
            for k in range(wpl):
                txl.assign(cmax[k], w[k])
            for j in range(1, iters):
                for k in range(wpl):
                    txl.ptx[MAXABS](cmax[k], cmax[k], w[j * wpl + k])
            for j in range(iters):
                txl.assign(rmax[j], w[j * wpl])
                for k in range(1, wpl):
                    txl.ptx[MAXABS](rmax[j], rmax[j], w[j * wpl + k])
            # transposed butterfly over the lpr lanes of a row: lane ends with row rsel's max
            cur = [rmax[j] for j in range(iters)]
            xor = lpr // 2
            while len(cur) > 1:
                half = len(cur) // 2
                bit = (lane & xor) != 0
                nxt = []
                for i in range(half):
                    txl.assign(sendv, txl.Select(bit, cur[i], cur[i + half]))
                    txl.ptx.shfl_sync.bfly.b32(upeer, sendv, txl.uint32(xor), txl.uint32(31), txl.uint32(FULL))
                    txl.assign(rmax[i], txl.Select(bit, cur[i + half], cur[i]))
                    txl.ptx[MAXABS](rmax[i], rmax[i], upeer)
                    nxt.append(rmax[i])
                cur = nxt
                xor //= 2
            while xor >= 1:
                txl.ptx.shfl_sync.bfly.b32(upeer, rmax[0], txl.uint32(xor), txl.uint32(31), txl.uint32(FULL))
                txl.ptx[MAXABS](rmax[0], rmax[0], upeer)
                xor //= 2
            # this lane's row: bits of the lane select it
            rsel = 0
            span = iters
            xb = lpr // 2
            while span > 1:
                span //= 2
                rsel = rsel + txl.Select((lane & xb) != 0, span, 0)
                xb //= 2
            amaxw = txl.bitwise_and(rmax[0], txl.uint32(0x7FFF7FFF))
            amax = txl.max(bf_lo(amaxw), bf_hi(amaxw))
            e8m0_pair(e16, txl.float32(0.0), amax)
            txl.assign(rexp, txl.bitwise_and(txl.Cast("uint32", e16), txl.uint32(0xFF)))
            rowbase = row0 + warp * rows_per_warp + sub
            with txl.If((lane & (lpr // iters - 1)) == 0), txl.Then():
                txl.ptx.st.global_.f32(
                    x_scale.ptr_to([(rowbase + rsel * rpi) * nct + ct]),
                    txl.reinterpret("float32", txl.shift_left(rexp, txl.uint32(23))),
                )
            if rpi == 2:
                for k in range(wpl):
                    txl.ptx.shfl_sync.bfly.b32(upeer, cmax[k], txl.uint32(16), txl.uint32(31), txl.uint32(FULL))
                    txl.ptx[MAXABS](cmax[k], cmax[k], upeer)
            cmb = (kk & 1) * (WARPS * BLOCK // 2)
            with txl.If(sub == 0), txl.Then():
                for k in range(wpl):
                    txl.ptx.st.shared.u32(cm.ptr_to([cmb + warp * (BLOCK // 2) + cq * wpl + k]), cmax[k])
            # named barriers: warps 2.. hand their column partials to warps 0-1 (bar 1) and wait only for
            # their column scales (bar 2); every warp quantizes the row-scaled x_fp8 in between
            with txl.If(warp >= 2), txl.Then():
                txl.ptx.bar.arrive(txl.uint32(1), txl.uint32(NTHREADS))
            txl.assign(obase, txl.Cast("int64", rowbase) * (n // 4) + txl.Cast("int64", col // 4))

            def quant(dst, rcp_of):
                for j in range(iters):
                    if dst is x_fp8:
                        # row j's exponent from the lane that reduced it
                        lg = iters.bit_length() - 1
                        src_lane = (lane & ~(lpr - 1)) + sum(((j >> (lg - 1 - q)) & 1) * (lpr >> (q + 1)) for q in range(lg))
                        txl.ptx.shfl_sync.idx.b32(upeer, rexp, txl.uint32(src_lane) if isinstance(src_lane, int) else txl.Cast("uint32", src_lane), txl.uint32(31), txl.uint32(FULL))
                        rh = txl.shift_left(txl.uint32(254) - upeer, txl.uint32(7))
                        txl.assign(rr, rh | txl.shift_left(rh, txl.uint32(16)))
                    with txl.If(col_ok), txl.Then():
                        for k in range(wpl):
                            txl.ptx.mul.rn.bf16x2(y, w[j * wpl + k], rcp_of(k))
                            txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(q16[k], y)
                        for p in range(wpl // 2):
                            txl.ptx.mov.b32(o[p], q16[2 * p], q16[2 * p + 1])
                        woff = obase + j * rpi * (n // 4)
                        if wpl == 4:
                            txl.ptx.st.global_.v2.b32(dst.ptr_to([woff]), o[0], o[1])
                        else:
                            txl.ptx.st.global_.b32(dst.ptr_to([woff]), o[0])

            quant(x_fp8, lambda k: rr)
            with txl.If(warp < 2), txl.Then():
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(NTHREADS))
                # one thread per column word folds the warps' partial maxima
                txl.ptx.ld.shared.u32(sendv, cm.ptr_to([cmb + tid]))
                for wi in range(1, WARPS):
                    txl.ptx.ld.shared.u32(upeer, cm.ptr_to([cmb + wi * (BLOCK // 2) + tid]))
                    txl.ptx[MAXABS](sendv, sendv, upeer)
                # the column pair's E8M0 scales and bf16x2 power-of-two reciprocal, once per CTA
                cw = txl.bitwise_and(sendv, txl.uint32(0x7FFF7FFF))
                e8m0_pair(e16, bf_hi(cw), bf_lo(cw))
                e32 = txl.Cast("uint32", e16)
                txl.assign(cexp[0], txl.bitwise_and(e32, txl.uint32(0xFF)))
                txl.assign(cexp[1], txl.shift_right(e32, txl.uint32(8)))
                txl.ptx.st.shared.u32(
                    cmf.ptr_to([(kk & 1) * (BLOCK // 2) + tid]),
                    txl.shift_left(txl.uint32(254) - cexp[0], txl.uint32(7))
                    | txl.shift_left(txl.uint32(254) - cexp[1], txl.uint32(23)),
                )
                with txl.If(ct * BLOCK + tid * 2 < n), txl.Then():
                    txl.ptx.st.global_.v2.f32(
                        x_ch_scale.ptr_to([rb * n + ct * BLOCK + tid * 2]),
                        txl.reinterpret("float32", txl.shift_left(cexp[0], txl.uint32(23))),
                        txl.reinterpret("float32", txl.shift_left(cexp[1], txl.uint32(23))),
                    )
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(NTHREADS))
            if wpl == 4:
                txl.ptx.ld.shared.v4.b32(crcp[0], crcp[1], crcp[2], crcp[3], cmf.ptr_to([(kk & 1) * (BLOCK // 2) + cq * 4]))
            else:
                txl.ptx.ld.shared.v2.b32(crcp[0], crcp[1], cmf.ptr_to([(kk & 1) * (BLOCK // 2) + cq * 2]))
            quant(x_ch, lambda k: crcp[k])
            txl.assign(kk, kk + 1)
            proc_valid(kk)
        txl.cuda.iket.range_end(tk_st[0])

    params = [
        ("gathered", txl.gptr[txl.i32]), ("experts", txl.gptr[txl.i32]), ("expand", txl.gptr[txl.i32]),
        ("offs", txl.gptr[txl.i32]), ("ks_t", txl.gptr[txl.i32]),
        ("reverse", txl.gptr[txl.i64]), ("x_fp8", txl.gptr[txl.i32]), ("x_scale", txl.gptr[txl.f32]),
        ("x_ch", txl.gptr[txl.i32]), ("x_ch_scale", txl.gptr[txl.f32]), ("ws", txl.gptr[txl.i32]),
        ("src_of_row", txl.gptr[txl.i32]),
        ("m", txl.i32), ("num_groups", txl.i32), ("ntiles", txl.i32), ("identity", txl.i32), ("num_ctas", txl.i32),
        ("gunits", txl.i32),
    ]

    def fused(*args):
        body(*args)

    fused.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ann) for name, ann in params]
    )
    fused.__name__ = "fused"
    return txl.kernel(warps=WARPS, grid="num_ctas", min_blocks_per_sm=CTAS_PER_SM)(fused)


def bound_groups(num_groups):
    gb = 32
    while gb < num_groups:
        gb *= 2
    return gb


def empty_outputs(n, dev):
    nsc = (n + BLOCK - 1) // BLOCK
    return (
        (
            torch.empty((0, n), dtype=torch.float8_e4m3fn, device=dev),
            torch.empty((0, nsc), dtype=torch.float32, device=dev),
            torch.empty((0, n), dtype=torch.float8_e4m3fn, device=dev),
            torch.empty((0, n), dtype=torch.float32, device=dev),
        ),
        torch.empty((0,), dtype=torch.int64, device=dev),
        torch.empty((0,), dtype=torch.int32, device=dev),
        [],
        torch.empty((0,), dtype=torch.int32, device=dev),
    )


def scatter_quant_sm100(gathered, expert_idxs, num_groups, expand_idx=None, padding_alignment=128, group_counts=None):
    assert padding_alignment == 128, "the per-channel FP8 layout needs 128-row groups"
    dev = gathered.device
    m = expert_idxs.shape[0]
    n = gathered.shape[1]
    if m == 0:
        return empty_outputs(n, dev)
    g = num_groups
    gb = bound_groups(g)
    nct = (n + BLOCK - 1) // BLOCK
    experts = expert_idxs.view(torch.int32)
    expand = experts if expand_idx is None else expand_idx.view(torch.int32)
    offs = torch.empty((g,), dtype=torch.int32, device=dev)
    ks_t = torch.empty((g,), dtype=torch.int32, device=dev)
    if group_counts is None:
        # sizes needed on the host to allocate: one counting kernel + the one allowed device-to-host copy
        counts_dev = torch.empty((g,), dtype=torch.int32, device=dev)
        with tvm_ffi.use_torch_stream():
            count_kernel(gb)(experts, counts_dev, offs, ks_t, m, g)
        group_counts = counts_dev.tolist()
    ks = [(c + BLOCK - 1) // BLOCK * BLOCK for c in group_counts]
    rows = sum(ks)
    num_rb = rows // BLOCK
    # one cooperative wave of co-resident CTAs; contiguous equal tile ranges (<= KMAX row blocks each)
    ntiles = num_rb * nct
    resident = SM_COUNT * (CTAS_PER_SM if smem_bytes(n, gb) <= SMEM_LIMIT else 1)
    num_ctas = min(resident, ntiles)
    nd = max(1, (nct + 1) // 3)
    assert -(-num_rb * (nct - nd) // num_ctas) <= (KMAX - 1) * max(nct - nd, 1), "static ranges would exceed KMAX row blocks"
    reverse = torch.empty((m,), dtype=torch.int64, device=dev)
    x_fp8 = torch.empty((rows, n), dtype=torch.float8_e4m3fn, device=dev)
    x_scale = torch.empty((rows, nct), dtype=torch.float32, device=dev)
    x_ch = torch.empty((rows, n), dtype=torch.float8_e4m3fn, device=dev)
    x_ch_scale = torch.empty((num_rb, n), dtype=torch.float32, device=dev)
    ws = torch.empty((2 * gb * num_ctas + gb + 1,), dtype=torch.int32, device=dev)
    src_of_row = torch.empty((rows,), dtype=torch.int32, device=dev)
    args = [
        gathered.view(-1).view(torch.int32), experts, expand, offs, ks_t, reverse,
        x_fp8.view(-1).view(torch.int32), x_scale.view(-1), x_ch.view(-1).view(torch.int32), x_ch_scale.view(-1),
        ws, src_of_row, m, g, ntiles, int(expand_idx is None), num_ctas, min(gathered.numel() * 2, PF_CAP) // 16,
    ]
    with tvm_ffi.use_torch_stream():
        fused_kernel(n, gb)(*args)
    return (x_fp8, x_scale, x_ch, x_ch_scale), reverse, offs, ks, ks_t

