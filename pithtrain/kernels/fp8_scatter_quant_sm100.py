"""
scatter_quant_sm100 in TIRx-lite: one fused kernel, no grid-wide synchronization.

Placement is the stable order: the slot of expert g with rank r (r-th occurrence of g in
expert_idxs) goes to grouped row start(g) + r. Every CTA owns one 128-row block (and a split
of its column tiles), so it finds its own slots: pass 1 counts the occurrences of its group per
256-slot chunk (ballot-free, 8 slots per lane + redux), a block scan turns the chunk counts into
ranks, and pass 2 revisits only the chunks that hold ranks [lo, lo + 128). The CTA then streams
its column tiles through a cp.async ring (each thread consumes what it copied), takes the 1x128
and 128x1 abs-max in bf16x2, and quantizes both layouts with exact power-of-two bf16x2 products
and cvt.rn.satfinite.e4m3x2.bf16x2.

The group sizes come in as kernel arguments when there are at most 32 groups (host counts, or
the one device-to-host copy of the device counts), else from a small counting kernel.

Evolved with TIRx-harness: run scatter, frontier member cluster-sort-balanced (commit 4f06168, 5.725x over the
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
ARG_GROUPS = 32
WARPS = 8
NTHREADS = WARPS * 32
CHUNK = 256  # slots per scan chunk: 8 per lane
UNROLL = 8  # slot loads in flight per lane in the sort
STAGES = 3
SM_COUNT = 148
CLUSTER = 8
CTA_BUDGET = 264  # resident CTAs (2 per SM) at cluster granularity
KMAX = 4  # row blocks a CTA's tile range may touch
CACHE_STEPS = 32  # sort steps cached in the (still idle) stage ring: 8 warps x 32 x 32 lanes x 3 words
COUNT_WARPS = 32


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
def fused_kernel(n, gb, from_args, cluster):
    func = fused_traced(n, gb, from_args, cluster).func
    if cluster:
        func = func.with_attr(
            "tirx.launch_tags", ["blockIdx.x", "clusterCtaIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]
        )
    return tvm.compile(tvm.IRModule({"main": func}), target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}), tir_pipeline="tirx")


@functools.cache
def fused_traced(n, gb, from_args, cluster):
    vec = 8 if n % 8 == 0 else 4
    assert n % vec == 0, f"N={n} must be a multiple of 4"
    wpl = vec // 2  # bf16x2 words per lane per row
    lpr = BLOCK // vec  # lanes per row
    rpi = 32 // lpr  # rows per warp instruction
    rows_per_warp = BLOCK // WARPS
    iters = rows_per_warp // rpi
    nct = (n + BLOCK - 1) // BLOCK
    nw = n // 2  # words per bf16 row
    gpl = gb // 32
    stage_words = iters * NTHREADS * wpl
    cp_insn = "cp.async.cg.shared.global" if vec == 8 else "cp.async.ca.shared.global"
    cp_bytes = vec * 2

    def body(gathered, experts, expand, counts_in, offs, ks_t, reverse, x_fp8, x_scale, x_ch, x_ch_scale,
             m, num_groups, ntiles, identity, num_ctas, *cargs):
        cid = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        def tile_begin(p):
            return txl.Cast("int32", txl.Cast("int64", p) * ntiles // num_ctas)

        tbeg = txl.local_scalar("int32", init=tile_begin(cid))
        tend = txl.local_scalar("int32", init=tile_begin(cid + 1))
        rbf = txl.local_scalar("int32", init=tbeg // nct)
        rbl = txl.local_scalar("int32", init=(tend - 1) // nct)

        smem = txl.smem_pool()
        stage = smem.alloc([STAGES * stage_words], txl.u32)
        tcnt = smem.alloc([gb], txl.i32)
        tstart = smem.alloc([gb], txl.i32)
        ctot = smem.alloc([gb], txl.i32)
        hist = smem.alloc([WARPS * gb], txl.i32)
        sl = smem.alloc([KMAX * BLOCK], txl.i32)
        cm = smem.alloc([2 * WARPS * BLOCK // 2], txl.u32)
        misc = smem.alloc([4 * KMAX], txl.i32)
        rbq = smem.alloc([CLUSTER * KMAX], txl.i32)
        tbs = smem.alloc([CLUSTER + 1], txl.i32)

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

        tk_pro = iket_range("prologue")
        mark("entry")
        # the first sort round's loads go out before anything else
        rank = txl.cta_id_in_cluster([CLUSTER])
        cid0 = cid - rank
        sw = txl.local_scalar("int32", init=(m + CLUSTER * WARPS * 32 - 1) // (CLUSTER * WARPS * 32) * 32)
        gw = rank * WARPS + warp
        nsteps = txl.local_scalar("int32", init=sw // 32)
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

        def load_steps(t0):
            """
            expert id and source row of this lane's slot in steps t0 .. t0 + UNROLL.
            """
            for u in range(UNROLL):
                i = gw * sw + (t0 + u) * 32 + lane
                txl.assign(ev[u], -1)
                txl.assign(sv[u], i)
                with txl.If(txl.And(t0 + u < nsteps, i < m)), txl.Then():
                    txl.ptx.ld.global_.nc.b32(ev[u], experts.ptr_to([i * 2]))
                    with txl.If(identity == 0), txl.Then():
                        txl.ptx.ld.global_.nc.b32(sv[u], expand.ptr_to([i * 2]))

        load_steps(0)
        with txl.If(tid < KMAX), txl.Then():
            txl.ptx.st.shared.s32(misc.ptr_to([tid * 4]), txl.int32(0))
            txl.ptx.st.shared.s32(misc.ptr_to([tid * 4 + 1]), txl.int32(0))
            txl.ptx.st.shared.s32(misc.ptr_to([tid * 4 + 2]), txl.int32(0))
        # ---- group sizes -> smem; scan of the padded sizes; this block's group
        cval = txl.local_scalar("int32")
        if from_args:
            with txl.If(tid == 0), txl.Then():
                for k in range(ARG_GROUPS):
                    txl.ptx.st.shared.s32(tcnt.ptr_to([k]), cargs[k])
        else:
            for k in range((gb + NTHREADS - 1) // NTHREADS):
                g = k * NTHREADS + tid
                with txl.If(g < gb), txl.Then():
                    txl.assign(cval, 0)
                    with txl.If(g < num_groups), txl.Then():
                        txl.ptx.ld.global_.b32(cval, counts_in.ptr_to([g]))
                    txl.ptx.st.shared.s32(tcnt.ptr_to([g]), cval)
        mark("pre-sync0")
        cta_sync()
        mark("pro-sync0")
        with txl.If(warp == 0), txl.Then():
            run = txl.local_scalar("int32", init=0)
            for k in range(gpl):
                txl.ptx.ld.shared.s32(cval, tcnt.ptr_to([lane * gpl + k]))
                with txl.If(lane * gpl + k < num_groups), txl.Then():
                    txl.assign(run, run + pad(cval))
            incl = txl.local_scalar("int32", init=run)
            peer = txl.local_scalar("int32")
            for d in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
                with txl.If(lane >= d), txl.Then():
                    txl.assign(incl, incl + peer)
            acc = txl.local_scalar("int32", init=incl - run)
            for k in range(gpl):
                g = lane * gpl + k
                with txl.If(g < num_groups), txl.Then():
                    txl.ptx.ld.shared.s32(cval, tcnt.ptr_to([g]))
                    padded = pad(cval)
                    txl.ptx.st.shared.s32(tstart.ptr_to([g]), acc)
                    if from_args:
                        with txl.If(cid == 0), txl.Then():
                            txl.ptx.st.global_.s32(offs.ptr_to([g]), acc + padded)
                            txl.ptx.st.global_.s32(ks_t.ptr_to([g]), padded)
                    for kq in range(KMAX):
                        row0 = (rbf + kq) * BLOCK
                        with txl.If(txl.And(txl.And(acc <= row0, row0 < acc + padded), rbf + kq <= rbl)), txl.Then():
                            # nreal of the block: real rows of its group past its first row
                            txl.ptx.st.shared.s32(misc.ptr_to([kq * 4]), txl.min(cval - (row0 - acc), BLOCK))
                    txl.assign(acc, acc + padded)
        cta_sync()
        mark("pro-sync1")

        txl.cuda.iket.range_end(tk_pro[0])
        tk_p1 = iket_range("sort")
        # ---- stable counting sort inside the cluster
        # Cluster CTA `rank`, warp `warp` owns the slot run [gw * sw, gw * sw + sw), gw = rank * 8 + warp.
        # pass A: per-warp group histograms; cluster barrier; prefix over (CTA, warp); pass B ranks
        # the slots in order and pushes each slot of this cluster's row blocks into the owner's sl.
        txl.assign(c, tid)
        with txl.While(c < WARPS * gb):
            txl.ptx.st.shared.s32(hist.ptr_to([c]), txl.int32(0))
            txl.assign(c, c + NTHREADS)
        for q in range(KMAX * BLOCK // NTHREADS):
            e = q * NTHREADS + tid
            nr = txl.local_scalar("int32")
            txl.ptx.ld.shared.s32(nr, misc.ptr_to([(e >> 7) * 4]))
            with txl.If((e & 127) >= nr), txl.Then():
                txl.ptx.st.shared.s32(sl.ptr_to([e]), txl.int32(-1))
        cta_sync()

        def match(u, cached=False):
            if cached:
                txl.assign(pmask, pm[u])
            else:
                txl.ptx["match.any.sync.b32"](pmask, ev[u], txl.uint32(FULL))
                txl.assign(pm[u], pmask)
            txl.ptx.popc.b32(total, pmask)
            txl.ptx.bfind.u32(leader, pmask)

        # per-warp cache of (expert, source row, match mask) for the first CACHE_STEPS steps
        cache = stage

        def cidx(step, f):
            return ((warp * CACHE_STEPS + step) * 32 + lane) * 3 + f

        t = txl.local_scalar("int32", init=0)
        with txl.While(t < nsteps):
            with txl.If(t > 0), txl.Then():
                load_steps(t)
            for u in range(UNROLL):
                match(u)
                with txl.If(txl.And(ev[u] >= 0, txl.Cast("uint32", lane) == leader)), txl.Then():
                    txl.ptx.red.shared.add.u32(hist.ptr_to([warp * gb + ev[u]]), total)
                with txl.If(t + u < CACHE_STEPS), txl.Then():
                    txl.ptx.st.shared.s32(cache.ptr_to([cidx(t + u, 0)]), ev[u])
                    txl.ptx.st.shared.s32(cache.ptr_to([cidx(t + u, 1)]), sv[u])
                    txl.ptx.st.shared.u32(cache.ptr_to([cidx(t + u, 2)]), pm[u])
            txl.assign(t, t + UNROLL)
        cta_sync()
        # CTA totals for the peers
        txl.assign(c, tid)
        with txl.While(c < gb):
            txl.assign(v, 0)
            for wi in range(WARPS):
                txl.ptx.ld.shared.s32(before, hist.ptr_to([wi * gb + c]))
                txl.assign(v, v + txl.Cast("int32", before))
            txl.ptx.st.shared.s32(ctot.ptr_to([c]), v)
            txl.assign(c, c + NTHREADS)
        mark("passA-done")
        txl.ptx.barrier.cluster.arrive()
        txl.ptx.barrier.cluster.wait()
        mark("barrier1-done")
        # run[w][g] = start(g) + earlier CTAs' totals + earlier warps' counts
        paddr = txl.local_scalar("uint32")
        row64 = txl.local_scalar("int64")
        got = txl.local_scalar("uint32")
        txl.assign(c, tid)
        with txl.While(c < gb):
            txl.ptx.ld.shared.s32(v, tstart.ptr_to([c]))
            for rp in range(CLUSTER - 1):
                with txl.If(rp < rank), txl.Then():
                    txl.ptx.mapa.shared__cluster.u32(
                        paddr, txl.cuda.cvta_generic_to_shared(ctot.ptr_to([c])), txl.uint32(rp)
                    )
                    txl.ptx.ld.shared__cluster.b32(got, paddr)
                    txl.assign(v, v + txl.Cast("int32", got))
            for wi in range(WARPS):
                txl.ptx.ld.shared.s32(before, hist.ptr_to([wi * gb + c]))
                txl.ptx.st.shared.s32(hist.ptr_to([wi * gb + c]), v)
                txl.assign(v, v + txl.Cast("int32", before))
            txl.assign(c, c + NTHREADS)
        cta_sync()
        mark("prefix-done")
        # pass B
        run = txl.local_scalar("int32")
        row = txl.local_scalar("int32")
        with txl.If(tid <= CLUSTER), txl.Then():
            txl.ptx.st.shared.s32(tbs.ptr_to([tid]), tile_begin(cid0 + tid))
        cta_sync()
        # ranks covering each row block of the cluster: lo | hi << 8
        with txl.If(tid < CLUSTER * KMAX), txl.Then():
            tb_lo = txl.local_scalar("int32")
            tb_hi = txl.local_scalar("int32")
            txl.ptx.ld.shared.s32(tb_lo, tbs.ptr_to([0]))
            rbx = txl.local_scalar("int32", init=tb_lo // nct + tid)
            qlo = txl.local_scalar("int32", init=CLUSTER)
            qhi = txl.local_scalar("int32", init=-1)
            for qq in range(CLUSTER):
                txl.ptx.ld.shared.s32(tb_lo, tbs.ptr_to([qq]))
                txl.ptx.ld.shared.s32(tb_hi, tbs.ptr_to([qq + 1]))
                with txl.If(txl.And(txl.And(tb_lo < (rbx + 1) * nct, tb_hi > rbx * nct), tb_hi > tb_lo)), txl.Then():
                    txl.assign(qlo, txl.min(qlo, qq))
                    txl.assign(qhi, qq)
            txl.ptx.st.shared.s32(rbq.ptr_to([tid]), qlo | txl.shift_left(qhi + 1, 8))
        cta_sync()
        tb0 = txl.local_scalar("int32")
        tb8 = txl.local_scalar("int32")
        txl.ptx.ld.shared.s32(tb0, tbs.ptr_to([0]))
        txl.ptx.ld.shared.s32(tb8, tbs.ptr_to([CLUSTER]))
        rba = txl.local_scalar("int32", init=tb0 // nct)
        rbb = txl.local_scalar("int32", init=(tb8 - 1) // nct)
        e1 = txl.local_scalar("int32")
        s1 = txl.local_scalar("int32")
        tq = txl.local_scalar("int32")
        tq1 = txl.local_scalar("int32")
        q = txl.local_scalar("int32")
        txl.assign(t, 0)
        with txl.While(t < nsteps):
            with txl.If(t < CACHE_STEPS):
                with txl.Then():
                    txl.ptx.ld.shared.s32(e1, cache.ptr_to([cidx(t, 0)]))
                    txl.ptx.ld.shared.s32(s1, cache.ptr_to([cidx(t, 1)]))
                    txl.ptx.ld.shared.u32(pmask, cache.ptr_to([cidx(t, 2)]))
                with txl.Else():
                    i = gw * sw + t * 32 + lane
                    txl.assign(e1, -1)
                    txl.assign(s1, i)
                    with txl.If(i < m), txl.Then():
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
                rbr = row >> 7
                with txl.If(txl.And(rbr >= rba, rbr <= rbb)), txl.Then():
                    with txl.If(rbr * nct >= tb0), txl.Then():
                        txl.assign(row64, txl.Cast("int64", row))
                        txl.ptx.st.global_.s64(reverse.ptr_to([gw * sw + t * 32 + lane]), row64)
                    txl.ptx.ld.shared.s32(tq1, rbq.ptr_to([rbr - rba]))
                    txl.assign(q, tq1 & 255)
                    with txl.While(q < (tq1 >> 8)):
                        txl.ptx.ld.shared.s32(tq, tbs.ptr_to([q]))
                        txl.ptx.mapa.shared__cluster.u32(
                            paddr,
                            txl.cuda.cvta_generic_to_shared(sl.ptr_to([(rbr - tq // nct) * BLOCK + (row & 127)])),
                            txl.Cast("uint32", q),
                        )
                        txl.ptx.st.shared__cluster.b32(paddr, txl.Cast("uint32", s1))
                        txl.assign(q, q + 1)
            txl.ptx.bar.warp.sync(txl.uint32(FULL))
            txl.assign(t, t + 1)
        mark("passB-done")
        txl.ptx.barrier.cluster.arrive()
        txl.ptx.barrier.cluster.wait()
        with txl.If(tend <= tbeg), txl.Then():
            txl.Return(txl.int32(0))

        txl.cuda.iket.range_end(tk_p1[0])
        tk_st = iket_range("stream")
        # ---- stream the column tiles
        cq = lane % lpr
        sub = lane // lpr
        srcs = txl.alloc_local([iters], "int32")

        def lrow(j):
            return warp * rows_per_warp + j * rpi + sub


        def stage_ptr(s, j):
            return stage.ptr_to([((s * iters + j) * NTHREADS + tid) * wpl])

        def issue(tile, s):
            trb = tile // nct
            ct = tile - trb * nct
            col = ct * BLOCK + cq * vec
            for j in range(iters):
                txl.ptx.ld.shared.s32(srcs[j], sl.ptr_to([(trb - rbf) * BLOCK + lrow(j)]))
            for j in range(iters):
                ok = txl.And(srcs[j] >= 0, col < n)
                woff = txl.Cast("int64", txl.max(srcs[j], 0)) * nw + txl.Cast("int64", txl.min(col, n - vec) // 2)
                txl.ptx[cp_insn](
                    stage_ptr(s, j), gathered.ptr_to([woff]), cp_bytes,
                    txl.Select(ok, txl.uint32(cp_bytes), txl.uint32(0)),
                )

        ntile = tend - tbeg
        for s in range(STAGES - 1):
            with txl.If(s < ntile), txl.Then():
                issue(tbeg + s, s)
            txl.ptx.cp.async_.commit_group()

        w = txl.alloc_local([iters * wpl], "uint32")
        a = txl.alloc_local([wpl], "uint32")
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

        kk = txl.local_scalar("int32", init=0)
        with txl.While(kk < ntile):
            tile = txl.local_scalar("int32", init=tbeg + kk)
            rb = txl.local_scalar("int32", init=tile // nct)
            ct = txl.local_scalar("int32", init=tile - rb * nct)
            row0 = rb * BLOCK
            with txl.If(kk + STAGES - 1 < ntile), txl.Then():
                issue(tile + STAGES - 1, (kk + STAGES - 1) % STAGES)
            txl.ptx.cp.async_.commit_group()
            txl.ptx.cp.async_.wait_group(STAGES - 1)
            s = kk % STAGES
            for j in range(iters):
                if wpl == 4:
                    txl.ptx.ld.shared.v4.b32(w[j * 4], w[j * 4 + 1], w[j * 4 + 2], w[j * 4 + 3], stage_ptr(s, j))
                else:
                    txl.ptx.ld.shared.v2.b32(w[j * 2], w[j * 2 + 1], stage_ptr(s, j))
            col = ct * BLOCK + cq * vec
            col_ok = col < n
            for k in range(wpl):
                txl.assign(cmax[k], txl.uint32(0))
            for j in range(iters):
                for k in range(wpl):
                    txl.assign(a[k], txl.bitwise_and(w[j * wpl + k], txl.uint32(0x7FFF7FFF)))
                    txl.ptx.max.bf16x2(cmax[k], cmax[k], a[k])
                txl.assign(rmax[j], a[0])
                for k in range(1, wpl):
                    txl.ptx.max.bf16x2(rmax[j], rmax[j], a[k])
                xor = lpr // 2
                while xor >= 1:
                    txl.ptx.shfl_sync.bfly.b32(upeer, rmax[j], txl.uint32(xor), txl.uint32(31), txl.uint32(FULL))
                    txl.ptx.max.bf16x2(rmax[j], rmax[j], upeer)
                    xor //= 2
            if rpi == 2:
                for k in range(wpl):
                    txl.ptx.shfl_sync.bfly.b32(upeer, cmax[k], txl.uint32(16), txl.uint32(31), txl.uint32(FULL))
                    txl.ptx.max.bf16x2(cmax[k], cmax[k], upeer)
            cmb = (kk & 1) * (WARPS * BLOCK // 2)
            with txl.If(sub == 0), txl.Then():
                for k in range(wpl):
                    txl.ptx.st.shared.u32(cm.ptr_to([cmb + warp * (BLOCK // 2) + cq * wpl + k]), cmax[k])
            cta_sync()
            for wi in range(WARPS):
                for k in range(wpl):
                    txl.ptx.ld.shared.u32(upeer, cm.ptr_to([cmb + wi * (BLOCK // 2) + cq * wpl + k]))
                    if wi == 0:
                        txl.assign(cmax[k], upeer)
                    else:
                        txl.ptx.max.bf16x2(cmax[k], cmax[k], upeer)
            for k in range(wpl):
                e8m0_pair(e16, bf_hi(cmax[k]), bf_lo(cmax[k]))
                e32 = txl.Cast("uint32", e16)
                txl.assign(cexp[2 * k], txl.bitwise_and(e32, txl.uint32(0xFF)))
                txl.assign(cexp[2 * k + 1], txl.shift_right(e32, txl.uint32(8)))
                txl.assign(
                    crcp[k],
                    txl.shift_left(txl.uint32(254) - cexp[2 * k], txl.uint32(7))
                    | txl.shift_left(txl.uint32(254) - cexp[2 * k + 1], txl.uint32(23)),
                )
            with txl.If(txl.And(txl.And(warp == 0, sub == 0), col_ok)), txl.Then():
                sbase = rb * n + col
                for q in range(vec // 4):
                    txl.ptx.st.global_.v4.f32(
                        x_ch_scale.ptr_to([sbase + q * 4]),
                        *[txl.reinterpret("float32", txl.shift_left(cexp[q * 4 + t], txl.uint32(23))) for t in range(4)],
                    )
            for j in range(iters):
                row = row0 + lrow(j)
                amax = txl.max(bf_lo(rmax[j]), bf_hi(rmax[j]))
                e8m0_pair(e16, txl.float32(0.0), amax)
                re = txl.bitwise_and(txl.Cast("uint32", e16), txl.uint32(0xFF))
                rh = txl.shift_left(txl.uint32(254) - re, txl.uint32(7))
                txl.assign(rr, rh | txl.shift_left(rh, txl.uint32(16)))
                with txl.If(cq == 0), txl.Then():
                    txl.ptx.st.global_.f32(
                        x_scale.ptr_to([row * nct + ct]),
                        txl.reinterpret("float32", txl.shift_left(re, txl.uint32(23))),
                    )
                with txl.If(col_ok), txl.Then():
                    for dst, rcp_of in ((x_fp8, lambda k: rr), (x_ch, lambda k: crcp[k])):
                        for k in range(wpl):
                            txl.ptx.mul.rn.bf16x2(y, w[j * wpl + k], rcp_of(k))
                            txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(q16[k], y)
                        for p in range(wpl // 2):
                            txl.ptx.mov.b32(o[p], q16[2 * p], q16[2 * p + 1])
                        woff = txl.Cast("int64", row) * (n // 4) + txl.Cast("int64", col // 4)
                        if wpl == 4:
                            txl.ptx.st.global_.v2.b32(dst.ptr_to([woff]), o[0], o[1])
                        else:
                            txl.ptx.st.global_.b32(dst.ptr_to([woff]), o[0])
            txl.assign(kk, kk + 1)
        txl.cuda.iket.range_end(tk_st[0])

    params = [
        ("gathered", txl.gptr[txl.i32]), ("experts", txl.gptr[txl.i32]), ("expand", txl.gptr[txl.i32]),
        ("counts_in", txl.gptr[txl.i32]), ("offs", txl.gptr[txl.i32]), ("ks_t", txl.gptr[txl.i32]),
        ("reverse", txl.gptr[txl.i64]), ("x_fp8", txl.gptr[txl.i32]), ("x_scale", txl.gptr[txl.f32]),
        ("x_ch", txl.gptr[txl.i32]), ("x_ch_scale", txl.gptr[txl.f32]),
        ("m", txl.i32), ("num_groups", txl.i32), ("ntiles", txl.i32), ("identity", txl.i32), ("num_ctas", txl.i32),
    ]
    if from_args:
        params += [(f"c{k}", txl.i32) for k in range(ARG_GROUPS)]

    def fused(*args):
        body(*args)

    fused.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ann) for name, ann in params]
    )
    fused.__name__ = "fused"
    return txl.kernel(warps=WARPS, grid="num_ctas", min_blocks_per_sm=2)(fused)


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
    from_args = g <= ARG_GROUPS
    counts_dev = offs
    if group_counts is None or not from_args:
        counts_dev = torch.empty((g,), dtype=torch.int32, device=dev)
        with tvm_ffi.use_torch_stream():
            count_kernel(gb)(experts, counts_dev, offs, ks_t, m, g)
    if group_counts is None:
        group_counts = counts_dev.tolist()
    ks = [(c + BLOCK - 1) // BLOCK * BLOCK for c in group_counts]
    rows = sum(ks)
    num_rb = rows // BLOCK
    # contiguous equal tile ranges; a range touches at most KMAX row blocks
    ntiles = num_rb * nct
    num_ctas = min(CTA_BUDGET, ntiles)
    num_ctas = max(num_ctas, -(-ntiles // ((KMAX - 1) * nct)))
    num_ctas = (num_ctas + CLUSTER - 1) // CLUSTER * CLUSTER
    cluster = True
    reverse = torch.empty((m,), dtype=torch.int64, device=dev)
    x_fp8 = torch.empty((rows, n), dtype=torch.float8_e4m3fn, device=dev)
    x_scale = torch.empty((rows, nct), dtype=torch.float32, device=dev)
    x_ch = torch.empty((rows, n), dtype=torch.float8_e4m3fn, device=dev)
    x_ch_scale = torch.empty((num_rb, n), dtype=torch.float32, device=dev)
    args = [
        gathered.view(-1).view(torch.int32), experts, expand, counts_dev, offs, ks_t, reverse,
        x_fp8.view(-1).view(torch.int32), x_scale.view(-1), x_ch.view(-1).view(torch.int32), x_ch_scale.view(-1),
        m, g, ntiles, int(expand_idx is None), num_ctas,
    ]
    if from_args:
        args += list(group_counts) + [0] * (ARG_GROUPS - g)
    with tvm_ffi.use_torch_stream():
        fused_kernel(n, gb, from_args, cluster)(*args)
    return (x_fp8, x_scale, x_ch, x_ch_scale), reverse, offs, ks, ks_t
