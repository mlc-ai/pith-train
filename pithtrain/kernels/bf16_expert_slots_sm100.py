"""
SM100 bf16_expert_slots in TIRx-lite: one launch per call (two for indexed segment_sum), no memsets.

group_slots: one cluster of 8 CTAs, each owning a slice of the slots held in registers: per-(tile,
warp) match.any counts and a slice histogram that every CTA pushes into every peer's shared memory
before one cluster barrier; the mask is zeroed by its owners before that barrier and OR-ed with
global reductions after it, then each slice is ranked with no further barrier (a slice too large
for registers takes the rescan kernel, where every CTA rescans all slots). gather_sum is persistent:
in each CTA warp 0 turns its tokens' mask bits into units of up to 4 grouped rows (looked up in the
per-(expert, token) table), one bulk copy per row into a 2-stage mbarrier ring, and every thread adds
a unit's rows in ascending expert order.
gather_rows is a 16-byte vector copy. segment_sum without an index gives each warp one 256-row
chunk, streamed through a per-thread cp.async ring, and adds the chunk sums in order through shared
memory; with an index it takes two launches: one warp per (chunk, 512-column block) writes the
chunk's fp32 sum, then a second kernel adds each segment's chunk sums in order.

Evolved with TIRx-harness: run slots, frontier member tma-gather (commit 3a959b6, 986 us vs 10312 us, 10.455x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools
from typing import NamedTuple, Optional

import torch

import tirx_kernels.tirx_lite as txl

__all__ = ["bf16_expert_group_slots_sm100", "bf16_expert_gather_sum_sm100", "bf16_expert_gather_rows_sm100", "bf16_expert_segment_sum_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


FULL = 0xFFFFFFFF


class GroupSlots(NamedTuple):
    cu_seqlens: torch.Tensor
    a_idx: torch.Tensor
    row_of: torch.Tensor
    slot_of: torch.Tensor
    mask: Optional[torch.Tensor]
    table: Optional[torch.Tensor]


def u322(x):
    return txl.Cast("uint32", x)


def i322(x):
    return txl.Cast("int32", x)


# ------------------------------------------------------------------------------------- group_slots

TILE = 1024  # slots ranked per CTA (one per thread)
MAXT = 4096  # tokens per mask window in shared memory


@functools.cache
def group_slots_kernel(G: int, has_x: bool):
    NW = TILE // 32

    @txl.kernel(warps=NW, arch="sm_100a", grid=lambda p: txl.max(1, (p["S"] + TILE - 1) // TILE))
    def group_slots(
        e: txl.gptr[txl.i64],
        x: txl.gptr[txl.i64],
        cu: txl.gptr[txl.i32],
        a_idx: txl.gptr[txl.i32],
        row_of: txl.gptr[txl.i32],
        slot_of: txl.gptr[txl.i32],
        mask: txl.gptr[txl.u64],
        table: txl.gptr[txl.i32],
        S: txl.i32,
        T: txl.i32,
    ):
        b = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        nb = txl.max(1, (S + TILE - 1) // TILE)
        smem = txl.smem_pool()
        tot = smem.alloc([64], txl.i32)
        bef = smem.alloc([64], txl.i32)
        cus = smem.alloc([64], txl.i32)
        wc = smem.alloc([64 * NW], txl.i32)
        msk = smem.alloc([MAXT], txl.u64)
        lt = txl.shift_left(txl.uint32(1), u322(lane)) - txl.uint32(1)

        with txl.If(tid < 64), txl.Then():
            txl.ptx.st.shared.s32(tot.ptr_to([tid]), txl.int32(0))
            txl.ptx.st.shared.s32(bef.ptr_to([tid]), txl.int32(0))
        with txl.serial(0, (64 * NW) // TILE) as i:
            txl.ptx.st.shared.s32(wc.ptr_to([i * TILE + tid]), txl.int32(0))

        tpb = (T + nb - 1) // nb  # tokens owned by this CTA: [t0, t0 + tpb)
        t0 = b * tpb
        nwin = txl.max(1, (tpb + MAXT - 1) // MAXT)
        before = b * TILE
        U = 4
        ev = txl.alloc_local([U], txl.i64)
        xv = txl.alloc_local([U], txl.i64)
        peers = txl.alloc_local([U], txl.u32)
        old = txl.local_scalar(txl.u64)
        with txl.serial(0, nwin) as w_:
            wt0 = t0 + w_ * MAXT
            wt1 = txl.min(txl.min(t0 + tpb, T), wt0 + MAXT)
            with txl.serial(0, MAXT // TILE) as i:
                txl.ptx.st.shared.u64(msk.ptr_to([i * TILE + tid]), txl.uint64(0))
            txl.cuda.cta_sync()
            with txl.serial(0, (S + TILE * U - 1) // (TILE * U)) as k:
                for u in range(U):
                    j = (k * U + u) * TILE + tid
                    txl.assign(ev[u], txl.int64(64))
                    with txl.If(j < S), txl.Then():
                        txl.ptx.ld.global_.nc.s64(ev[u], e.ptr_to([j]))
                        if has_x:
                            txl.ptx.ld.global_.nc.s64(xv[u], x.ptr_to([j]))
                        else:
                            txl.assign(xv[u], txl.Cast("int64", j))
                for u in range(U):
                    j = (k * U + u) * TILE + tid
                    ei = i322(ev[u])
                    if has_x:
                        xi = i322(xv[u])
                        with txl.If(txl.And(ei < 64, txl.And(xi >= wt0, xi < wt1))), txl.Then():
                            txl.ptx.atom.shared.or_.b64(
                                old, msk.ptr_to([xi - wt0]), txl.shift_left(txl.uint64(1), txl.Cast("uint64", ei))
                            )
                    with txl.If(w_ == 0), txl.Then():
                        txl.ptx["match.any.sync.b32"](peers[u], ei, txl.uint32(FULL))
                        lead = (peers[u] & lt) == txl.uint32(0)
                        with txl.If(txl.And(lead, ei < 64)), txl.Then():
                            cnt = i322(txl.popcount(peers[u]))
                            txl.ptx.red.shared.add.s32(tot.ptr_to([ei]), cnt)
                            # a 1024-slot step lies wholly before this CTA's tile or not at all
                            with txl.If(j < before), txl.Then():
                                txl.ptx.red.shared.add.s32(bef.ptr_to([ei]), cnt)
            txl.cuda.cta_sync()
            if has_x:
                with txl.serial(0, MAXT // TILE) as i:
                    t = wt0 + i * TILE + tid
                    with txl.If(t < wt1), txl.Then():
                        txl.ptx.ld.shared.u64(old, msk.ptr_to([i * TILE + tid]))
                        txl.ptx.st.global_.u64(mask.ptr_to([t]), old)
            txl.cuda.cta_sync()

        # cu: exclusive scan of the totals by warp 0 (two experts a lane)
        with txl.If(warp == 0), txl.Then():
            a0 = txl.local_scalar(txl.i32)
            a1 = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(a0, tot.ptr_to([2 * lane]))
            txl.ptx.ld.shared.s32(a1, tot.ptr_to([2 * lane + 1]))
            s = txl.local_scalar(txl.i32, init=a0 + a1)
            incl = txl.local_scalar(txl.i32, init=s)
            o = txl.local_scalar(txl.i32)
            for d in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(o, incl, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
                txl.assign(incl, incl + txl.Select(lane >= d, o, txl.int32(0)))
            ex = incl - s
            txl.ptx.st.shared.s32(cus.ptr_to([2 * lane]), ex)
            txl.ptx.st.shared.s32(cus.ptr_to([2 * lane + 1]), ex + a0)
            with txl.If(b == 0), txl.Then():
                with txl.If(2 * lane <= G), txl.Then():
                    txl.ptx.st.global_.s32(cu.ptr_to([2 * lane]), ex)
                with txl.If(2 * lane + 1 <= G), txl.Then():
                    txl.ptx.st.global_.s32(cu.ptr_to([2 * lane + 1]), ex + a0)
                if G == 64:
                    with txl.If(lane == 31), txl.Then():
                        txl.ptx.st.global_.s32(cu.ptr_to([64]), incl)

        # rank this CTA's tile
        j = before + tid
        e0 = txl.local_scalar(txl.i64, init=txl.int64(64))
        x0 = txl.local_scalar(txl.i64, init=txl.int64(0))
        with txl.If(j < S), txl.Then():
            txl.ptx.ld.global_.nc.s64(e0, e.ptr_to([j]))
            if has_x:
                txl.ptx.ld.global_.nc.s64(x0, x.ptr_to([j]))
        ei = i322(e0)
        pr = txl.local_scalar(txl.u32)
        txl.ptx["match.any.sync.b32"](pr, ei, txl.uint32(FULL))
        lrank = i322(txl.popcount(pr & lt))
        with txl.If(txl.And(lrank == 0, ei < 64)), txl.Then():
            txl.ptx.st.shared.s32(wc.ptr_to([ei * NW + warp]), i322(txl.popcount(pr)))
        txl.cuda.cta_sync()
        # per-expert exclusive scan over the warps: warp w scans experts w, w + 32
        for half in range(2):
            ee = warp + 32 * half
            v = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(v, wc.ptr_to([ee * NW + lane]))
            inc = txl.local_scalar(txl.i32, init=v)
            o2 = txl.local_scalar(txl.i32)
            for d in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(o2, inc, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
                txl.assign(inc, inc + txl.Select(lane >= d, o2, txl.int32(0)))
            txl.ptx.st.shared.s32(wc.ptr_to([ee * NW + lane]), inc - v)
        txl.cuda.cta_sync()
        with txl.If(j < S), txl.Then():
            c0 = txl.local_scalar(txl.i32)
            c1 = txl.local_scalar(txl.i32)
            c2 = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(c0, cus.ptr_to([ei]))
            txl.ptx.ld.shared.s32(c1, bef.ptr_to([ei]))
            txl.ptx.ld.shared.s32(c2, wc.ptr_to([ei * NW + warp]))
            row = txl.local_scalar(txl.i32, init=c0 + c1 + c2 + lrank)
            txl.ptx.st.global_.s32(row_of.ptr_to([j]), row)
            txl.ptx.st.global_.s32(slot_of.ptr_to([row]), j)
            if has_x:
                xi = i322(x0)
                txl.ptx.st.global_.s32(a_idx.ptr_to([row]), xi)
                txl.ptx.st.global_.s32(table.ptr_to([ei * T + xi]), row)
            else:
                txl.ptx.st.global_.s32(a_idx.ptr_to([row]), j)

    return entry_function(group_slots.compile())


CCS = 16  # (non-portable cluster size, launches on B200) cluster group_slots: the whole call is one cluster of CCS CTAs, CTA r owns slots [r * SP, r * SP + SP)
NCL = 4  # clusters: each computes the histograms/scan redundantly and stores 1/NCL of the slots
CSP = (1024, 2048, 4096, 8192)  # slots per CTA held in registers: the smaller fits if it can (fewer registers); larger S rescans


@functools.cache
def group_slots_cluster_kernel(G: int, has_x: bool, csp: int):
    NW = TILE // 32
    U = csp // TILE  # 1024-slot tiles per CTA, held in registers

    @txl.kernel(warps=NW, arch="sm_100a", grid=CCS * NCL)
    def group_slots_cluster(
        e: txl.gptr[txl.i32],  # int64 ids viewed as int32 pairs: the low word at 2 * j
        x: txl.gptr[txl.i32],
        cu: txl.gptr[txl.i32],
        a_idx: txl.gptr[txl.i32],
        row_of: txl.gptr[txl.i32],
        slot_of: txl.gptr[txl.i32],
        mask: txl.gptr[txl.u64],
        table: txl.gptr[txl.i32],
        S: txl.i32,
        T: txl.i32,
    ):
        r = txl.cta_id_in_cluster([CCS])
        # peers' shared memory may be written only once every CTA of the cluster has started: arrive now,
        # wait just before the first remote store (by then the wait is free)
        txl.ptx.barrier.cluster.arrive.relaxed()
        cl = txl.cta_id() // CCS  # this cluster stores warp-tiles (u, warp) with (u * NW + warp) % NCL == cl
        # and owns the mask tokens of 32-token blocks b with b % NCL == cl (interleaved: every CTA's slots
        # spread their mask bits over all clusters, whatever the token order of the slots)
        TT = ((T + 32 * NCL - 1) // (32 * NCL)) * 32  # owned tokens per cluster (upper bound)
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        smem = txl.smem_pool()
        hist = smem.alloc([64], txl.i32)  # this CTA's slice histogram
        allh = smem.alloc([CCS * 64], txl.i32)  # every CTA's histogram, pushed here before the cluster barrier
        base = smem.alloc([64], txl.i32)  # cu[e] + slots of e before this CTA's slice
        wc = smem.alloc([U * 64 * NW], txl.i32)  # (tile, expert, warp) counts, then their exclusive scan
        lt = txl.shift_left(txl.uint32(1), u322(lane)) - txl.uint32(1)
        SP = (S + CCS - 1) // CCS
        s0 = r * SP
        n = txl.max(0, txl.min(S, s0 + SP) - s0)
        TPC = (TT + CCS - 1) // CCS
        raddr = txl.local_scalar(txl.u32)

        def csync():
            txl.ptx.barrier.cluster.arrive.release()
            txl.ptx.barrier.cluster.wait.acquire()

        def iket_range(name):
            token = txl.alloc_local([1], "uint32")
            txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
            with txl.If(lane == 0), txl.Then():
                txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        tk = iket_range("stage")
        ev = txl.alloc_local([U], txl.i32)
        xv = txl.alloc_local([U], txl.i32)
        prs = txl.alloc_local([U], txl.u32)
        for u in range(U):
            jl = u * TILE + tid
            txl.assign(ev[u], txl.int32(64))
            with txl.If(jl < n), txl.Then():
                txl.ptx.ld.global_.nc.s32(ev[u], e.ptr_to([2 * (s0 + jl)]))
                if has_x:
                    txl.ptx.ld.global_.nc.s32(xv[u], x.ptr_to([2 * (s0 + jl)]))
        with txl.If(tid < 64), txl.Then():
            txl.ptx.st.shared.s32(hist.ptr_to([tid]), txl.int32(0))
        for u in range(U):
            with txl.If(u * TILE < n), txl.Then():
                for i in range((64 * NW) // TILE):
                    txl.ptx.st.shared.s32(wc.ptr_to([u * 64 * NW + i * TILE + tid]), txl.int32(0))
        if has_x:
            with txl.serial(0, (TPC + TILE - 1) // TILE) as i:
                o = r * TPC + i * TILE + tid  # owned-token ordinal
                t = ((o >> 5) * NCL + cl) * 32 + (o & 31)
                with txl.If(txl.And(t < T, i * TILE + tid < TPC)), txl.Then():
                    txl.ptx.st.global_.u64(mask.ptr_to([t]), txl.uint64(0))
        txl.cuda.cta_sync()
        for u in range(U):
            with txl.If(u * TILE < n), txl.Then():
                txl.ptx["match.any.sync.b32"](prs[u], ev[u], txl.uint32(FULL))
                with txl.If(txl.And((prs[u] & lt) == txl.uint32(0), ev[u] < 64)), txl.Then():
                    cnt = i322(txl.popcount(prs[u]))
                    txl.ptx.red.shared.add.s32(hist.ptr_to([ev[u]]), cnt)
                    txl.ptx.st.shared.s32(wc.ptr_to([(u * 64 + ev[u]) * NW + warp]), cnt)
        txl.cuda.cta_sync()
        # the slice histogram is complete: push it to every peer now, under the scan below
        txl.ptx.barrier.cluster.wait()
        with txl.If(tid < 64), txl.Then():
            hv = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(hv, hist.ptr_to([tid]))
            for q in range(CCS):
                txl.ptx.mapa.shared__cluster.u32(raddr, txl.cuda.cvta_generic_to_shared(allh.ptr_to([r * 64 + tid])), txl.uint32(q))
                txl.ptx["st.shared::cluster.s32"](raddr, hv)
        # per-expert exclusive scan over (tile, warp): warp w scans experts w and w + 32
        v = txl.local_scalar(txl.i32)
        inc = txl.local_scalar(txl.i32)
        o2 = txl.local_scalar(txl.i32)
        carry = txl.local_scalar(txl.i32)
        tl_ = txl.local_scalar(txl.i32)
        for half in range(2):
            ee = warp + 32 * half
            txl.assign(carry, txl.int32(0))
            for u in range(U):
                with txl.If(u * TILE < n), txl.Then():
                    txl.ptx.ld.shared.s32(v, wc.ptr_to([(u * 64 + ee) * NW + lane]))
                    txl.assign(inc, v)
                    for d in (1, 2, 4, 8, 16):
                        txl.ptx.shfl_sync.up.b32(o2, inc, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
                        txl.assign(inc, inc + txl.Select(lane >= d, o2, txl.int32(0)))
                    txl.ptx.st.shared.s32(wc.ptr_to([(u * 64 + ee) * NW + lane]), carry + inc - v)
                    txl.ptx.shfl_sync.idx.b32(tl_, inc, txl.uint32(31), txl.uint32(31), txl.uint32(FULL))
                    txl.assign(carry, carry + tl_)
        txl.cuda.iket.range_end(tk[0])
        tk2 = iket_range("csync1")
        csync()  # all histograms pushed; the mask zeros visible cluster-wide
        txl.cuda.iket.range_end(tk2[0])
        tk3 = iket_range("mask-dsmem")
        if has_x:
            for u in range(U):
                with txl.If(txl.And(u * TILE + tid < n, ((xv[u] >> 5) % NCL) == cl)), txl.Then():
                    txl.ptx.red.global_.or_.b64(mask.ptr_to([xv[u]]), txl.shift_left(txl.uint64(1), txl.Cast("uint64", ev[u])))
        # warp 0: totals and this slice's offsets for experts 2 * lane and 2 * lane + 1, then the cu scan
        with txl.If(warp == 0), txl.Then():
            h0 = txl.alloc_local([CCS], txl.i32)
            h1 = txl.alloc_local([CCS], txl.i32)
            for q in range(CCS):
                txl.ptx.ld.shared.v2.b32(h0[q], h1[q], allh.ptr_to([q * 64 + 2 * lane]))
            a0 = txl.local_scalar(txl.i32, init=txl.int32(0))
            a1 = txl.local_scalar(txl.i32, init=txl.int32(0))
            b0 = txl.local_scalar(txl.i32, init=txl.int32(0))
            b1 = txl.local_scalar(txl.i32, init=txl.int32(0))
            for q in range(CCS):
                txl.assign(a0, a0 + h0[q])
                txl.assign(a1, a1 + h1[q])
                txl.assign(b0, b0 + txl.Select(r > q, h0[q], txl.int32(0)))
                txl.assign(b1, b1 + txl.Select(r > q, h1[q], txl.int32(0)))
            s = txl.local_scalar(txl.i32, init=a0 + a1)
            incl = txl.local_scalar(txl.i32, init=s)
            o = txl.local_scalar(txl.i32)
            for d in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(o, incl, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
                txl.assign(incl, incl + txl.Select(lane >= d, o, txl.int32(0)))
            ex = txl.local_scalar(txl.i32, init=incl - s)
            txl.ptx.st.shared.s32(base.ptr_to([2 * lane]), ex + b0)
            txl.ptx.st.shared.s32(base.ptr_to([2 * lane + 1]), ex + a0 + b1)
            with txl.If(txl.And(r == 0, cl == 0)), txl.Then():
                with txl.If(2 * lane <= G), txl.Then():
                    txl.ptx.st.global_.s32(cu.ptr_to([2 * lane]), ex)
                with txl.If(2 * lane + 1 <= G), txl.Then():
                    txl.ptx.st.global_.s32(cu.ptr_to([2 * lane + 1]), ex + a0)
                if G == 64:
                    with txl.If(lane == 31), txl.Then():
                        txl.ptx.st.global_.s32(cu.ptr_to([64]), incl)
        txl.cuda.cta_sync()
        txl.cuda.iket.range_end(tk3[0])
        tk5 = iket_range("rank")
        c0 = txl.local_scalar(txl.i32)
        c2 = txl.local_scalar(txl.i32)
        for u in range(U):
            jl = u * TILE + tid
            with txl.If(txl.And(jl < n, (u * NW + warp) % NCL == cl)), txl.Then():
                j = s0 + jl
                txl.ptx.ld.shared.s32(c0, base.ptr_to([ev[u]]))
                txl.ptx.ld.shared.s32(c2, wc.ptr_to([(u * 64 + ev[u]) * NW + warp]))
                row = txl.local_scalar(txl.i32, init=c0 + c2 + i322(txl.popcount(prs[u] & lt)))
                txl.ptx.st.global_.s32(row_of.ptr_to([j]), row)
                txl.ptx.st.global_.s32(slot_of.ptr_to([row]), j)
                if has_x:
                    txl.ptx.st.global_.s32(a_idx.ptr_to([row]), xv[u])
                    txl.ptx.st.global_.s32(table.ptr_to([ev[u] * T + xv[u]]), row)
                else:
                    txl.ptx.st.global_.s32(a_idx.ptr_to([row]), j)
        txl.cuda.iket.range_end(tk5[0])

    return entry_function(group_slots_cluster.compile())


def bf16_expert_group_slots_sm100(expert_idxs, expand_idx, num_groups, num_tokens):
    assert 0 < num_groups <= 64
    S, dev = expert_idxs.shape[0], expert_idxs.device
    cu = torch.empty(num_groups + 1, dtype=torch.int32, device=dev)
    a_idx = torch.empty(S, dtype=torch.int32, device=dev)
    row_of = torch.empty(S, dtype=torch.int32, device=dev)
    slot_of = torch.empty(S, dtype=torch.int32, device=dev)
    has_x = expand_idx is not None
    if has_x:
        mask = torch.empty(num_tokens, dtype=torch.int64, device=dev)
        table = torch.empty(num_tokens * num_groups, dtype=torch.int32, device=dev)
    else:
        mask = torch.empty(1, dtype=torch.int64, device=dev)
        table = torch.empty(1, dtype=torch.int32, device=dev)
    csp = next((c for c in CSP if S <= CCS * c), None)
    if csp is not None:
        k = group_slots_cluster_kernel(num_groups, has_x, csp)
        k(expert_idxs.view(torch.int32), (expand_idx if has_x else expert_idxs).view(torch.int32), cu, a_idx, row_of,
          slot_of, mask.view(torch.uint64), table, S, num_tokens if has_x else 0)
        if not has_x:
            return GroupSlots(cu, a_idx, row_of, slot_of, None, None)
        return GroupSlots(cu, a_idx, row_of, slot_of, mask, table)
    k = group_slots_kernel(num_groups, has_x)
    k(expert_idxs, expand_idx if has_x else expert_idxs, cu, a_idx, row_of, slot_of, mask.view(torch.uint64),
      table, S, num_tokens if has_x else 0)
    if not has_x:
        return GroupSlots(cu, a_idx, row_of, slot_of, None, None)
    return GroupSlots(cu, a_idx, row_of, slot_of, mask, table)


# -------------------------------------------------------------------------------------- gather_sum


GS_BATCH = 8  # experts per prefetched batch
GS_CTAS_PER_SM = {2880: 3}  # by H; default 4
NUM_SMS = 148


@functools.cache
def gather_sum_kernel(G: int, H: int):
    V = H // 8
    NW = (V + 31) // 32
    KB = min(GS_BATCH, G)
    GRID = NUM_SMS * GS_CTAS_PER_SM.get(H, 4)

    @txl.kernel(warps=NW, arch="sm_100a", grid=lambda p: txl.min(p["T"], GRID))
    def gather_sum(
        grad: txl.gptr[txl.u32],
        mask: txl.gptr[txl.u64],
        table: txl.gptr[txl.i32],
        out: txl.gptr[txl.u32],
        T: txl.i32,
    ):
        tid = txl.thread_id()
        grid = txl.min(T, GRID)
        active = tid < V
        t = txl.local_scalar(txl.i32, init=txl.cta_id())
        low = txl.local_scalar(txl.u64)
        acc = txl.alloc_local([8], txl.f32)
        w = txl.alloc_local([4 * KB], txl.u32)
        rc = txl.alloc_local([KB], txl.i32)  # current token's grouped rows (-1: none)
        rn = txl.alloc_local([KB], txl.i32)  # next token's
        ee = txl.alloc_local([KB], txl.i32)
        rem_c = txl.local_scalar(txl.u64)
        rem_n = txl.local_scalar(txl.u64)
        m_n = txl.local_scalar(txl.u64, init=txl.uint64(0))
        m_n2 = txl.local_scalar(txl.u64)
        row = txl.local_scalar(txl.i32)

        def take_batch(m, tt, rows):
            # pop the next KB experts of bitmask m (ascending) and load their grouped rows
            for k in range(KB):
                txl.assign(low, m & (txl.uint64(0) - m))
                txl.assign(ee[k], txl.Select(low != txl.uint64(0), i322(txl.popcount(low - txl.uint64(1))), txl.int32(-1)))
                txl.assign(m, m ^ low)
            for k in range(KB):
                txl.assign(rows[k], txl.int32(-1))
                with txl.If(ee[k] >= 0), txl.Then():
                    txl.ptx.ld.global_.nc.s32(rows[k], table.ptr_to([ee[k] * T + tt]))

        def add_rows(rows):
            for k in range(KB):
                with txl.If(txl.And(rows[k] >= 0, active)), txl.Then():
                    txl.ptx.ld.global_.nc.v4.b32(w[4 * k], w[4 * k + 1], w[4 * k + 2], w[4 * k + 3],
                                                 grad.ptr_to([txl.Cast("int64", rows[k]) * (H // 2) + tid * 4]))

        def accumulate(rows):
            for k in range(KB):
                with txl.If(rows[k] >= 0), txl.Then():
                    for p in range(4):
                        lo = txl.reinterpret("float32", txl.shift_left(w[4 * k + p], txl.uint32(16)))
                        hi = txl.reinterpret("float32", w[4 * k + p] & txl.uint32(0xFFFF0000))
                        txl.assign(acc[2 * p], acc[2 * p] + lo)
                        txl.assign(acc[2 * p + 1], acc[2 * p + 1] + hi)

        # prologue: rows of the first token, mask of the second
        txl.assign(rem_c, txl.uint64(0))
        with txl.If(t < T), txl.Then():
            txl.ptx.ld.global_.nc.u64(rem_c, mask.ptr_to([t]))
        take_batch(rem_c, t, rc)
        with txl.If(t + grid < T), txl.Then():
            txl.ptx.ld.global_.nc.u64(m_n, mask.ptr_to([t + grid]))
        with txl.While(t < T):
            add_rows(rc)
            txl.assign(m_n2, txl.uint64(0))
            with txl.If(t + 2 * grid < T), txl.Then():
                txl.ptx.ld.global_.nc.u64(m_n2, mask.ptr_to([t + 2 * grid]))
            txl.assign(rem_n, m_n)
            take_batch(rem_n, t + grid, rn)
            for i in range(8):
                txl.assign(acc[i], txl.float32(0))
            accumulate(rc)
            # experts past the first KB (rare): one at a time, still ascending
            with txl.While(rem_c != txl.uint64(0)):
                txl.assign(low, rem_c & (txl.uint64(0) - rem_c))
                txl.assign(rem_c, rem_c ^ low)
                txl.ptx.ld.global_.nc.s32(row, table.ptr_to([i322(txl.popcount(low - txl.uint64(1))) * T + t]))
                with txl.If(active), txl.Then():
                    txl.ptx.ld.global_.nc.v4.b32(w[0], w[1], w[2], w[3],
                                                 grad.ptr_to([txl.Cast("int64", row) * (H // 2) + tid * 4]))
                    for p in range(4):
                        lo = txl.reinterpret("float32", txl.shift_left(w[p], txl.uint32(16)))
                        hi = txl.reinterpret("float32", w[p] & txl.uint32(0xFFFF0000))
                        txl.assign(acc[2 * p], acc[2 * p] + lo)
                        txl.assign(acc[2 * p + 1], acc[2 * p + 1] + hi)
            with txl.If(active), txl.Then():
                for p in range(4):
                    txl.ptx["cvt.rn.bf16x2.f32"](w[p], acc[2 * p + 1], acc[2 * p])
                txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", t) * (H // 2) + tid * 4]), w[0], w[1], w[2], w[3])
            for k in range(KB):
                txl.assign(rc[k], rn[k])
            txl.assign(rem_c, rem_n)
            txl.assign(m_n, m_n2)
            txl.assign(t, t + grid)

    return entry_function(gather_sum.compile())


GS_TMA = True
GS2_CFG = {2048: (2, 6), 2880: (2, 4)}  # by H: (pipeline stages, CTAs per SM); default (2, 2)
GS2_BATCH = 4  # grouped rows per unit
BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
EVF = {"gs": False, "rows": False, "seg": True}  # stream-once reads marked L2 evict_first: helps segment_sum, hurts gather_rows, neutral for gather_sum


@functools.cache
def gather_sum_tma_kernel(G: int, H: int):
    """
    Persistent: warp 0 turns each token's mask bits into units of up to KB grouped rows, one bulk copy
    per row into an ST-deep shared-memory ring; every thread adds a unit's rows in ascending expert order.
    """
    V = H // 8
    NW = (V + 31) // 32
    KB = min(GS2_BATCH, G)
    ST, CPS = GS2_CFG.get(H, (2, 2))
    GRID = NUM_SMS * CPS
    RW = H // 2  # u32 words per row
    RB = H * 2  # bytes per row
    assert RB % 16 == 0 and KB <= 32

    @txl.kernel(warps=NW, arch="sm_100a", grid=lambda p: txl.min(p["T"], GRID))
    def gather_sum_tma(
        grad: txl.gptr[txl.u32],
        mask: txl.gptr[txl.u64],
        table: txl.gptr[txl.i32],
        out: txl.gptr[txl.u32],
        T: txl.i32,
    ):
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        grid = txl.min(T, GRID)
        active = tid < V
        smem = txl.smem_pool()
        stage = smem.alloc([ST * KB * RW], txl.u32, align=128)
        bars = smem.alloc([ST], txl.u64, align=8)
        meta = smem.alloc([2 * ST], txl.i32)  # per slot: token, rows | last << 8 (-1: no more units)
        with txl.If(tid < ST), txl.Then():
            txl.ptx.mbarrier.init.shared.b64(bars.ptr_to([tid]), txl.uint32(1))
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        # producer state (warp 0, uniform across its lanes)
        tp = txl.local_scalar(txl.i32, init=txl.cta_id())
        mp = txl.local_scalar(txl.u64, init=txl.uint64(0))
        mn = txl.local_scalar(txl.u64, init=txl.uint64(0))
        low = txl.local_scalar(txl.u64)
        ee = txl.alloc_local([KB], txl.i32)
        em = txl.local_scalar(txl.i32)
        nv = txl.local_scalar(txl.i32)
        rr = txl.local_scalar(txl.i32)
        with txl.If(warp == 0), txl.Then():
            with txl.If(tp < T), txl.Then():
                txl.ptx.ld.global_.nc.u64(mp, mask.ptr_to([tp]))
            with txl.If(tp + grid < T), txl.Then():
                txl.ptx.ld.global_.nc.u64(mn, mask.ptr_to([tp + grid]))

        def produce(slot):
            with txl.If(tp < T):
                with txl.Then():
                    txl.assign(nv, txl.int32(0))
                    txl.assign(em, txl.int32(-1))
                    for k in range(KB):
                        txl.assign(low, mp & (txl.uint64(0) - mp))
                        txl.assign(ee[k], txl.Select(low != txl.uint64(0), i322(txl.popcount(low - txl.uint64(1))), txl.int32(-1)))
                        txl.assign(mp, mp ^ low)
                        txl.assign(nv, nv + txl.Select(ee[k] >= 0, txl.int32(1), txl.int32(0)))
                        txl.assign(em, txl.Select(lane == k, ee[k], em))
                    last = txl.Select(mp == txl.uint64(0), txl.int32(1), txl.int32(0))
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.shared.s32(meta.ptr_to([2 * slot]), tp)
                        txl.ptx.st.shared.s32(meta.ptr_to([2 * slot + 1]), nv + last * 256)
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(bars.ptr_to([slot]), u322(nv * RB))
                    txl.ptx["bar.warp.sync"](txl.uint32(FULL))
                    with txl.If(em >= 0), txl.Then():
                        txl.ptx.ld.global_.nc.s32(rr, table.ptr_to([em * T + tp]))
                        txl.ptx[BULK_G2S](
                            stage.ptr_to([(slot * KB + lane) * RW]),
                            grad.ptr_to([txl.Cast("int64", rr) * RW]),
                            txl.uint32(RB),
                            bars.ptr_to([slot]),
                        )
                    with txl.If(mp == txl.uint64(0)), txl.Then():
                        txl.assign(tp, tp + grid)
                        txl.assign(mp, mn)
                        txl.assign(mn, txl.uint64(0))
                        with txl.If(tp + grid < T), txl.Then():
                            txl.ptx.ld.global_.nc.u64(mn, mask.ptr_to([tp + grid]))
                with txl.Else():
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.shared.s32(meta.ptr_to([2 * slot + 1]), txl.int32(-1))
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(bars.ptr_to([slot]), txl.uint32(0))

        with txl.If(warp == 0), txl.Then():
            for q in range(ST):
                produce(q)

        acc = txl.alloc_local([8], txl.f32)
        w = txl.alloc_local([4], txl.u32)
        i = txl.local_scalar(txl.i32, init=txl.int32(0))
        go = txl.local_scalar(txl.i32, init=txl.int32(1))
        tt = txl.local_scalar(txl.i32)
        info = txl.local_scalar(txl.i32)
        sl = txl.local_scalar(txl.i32)
        for p in range(8):
            txl.assign(acc[p], txl.float32(0))
        with txl.While(go > 0):
            txl.assign(sl, i % ST)
            txl.cuda.mbarrier_wait(bars.ptr_to([sl]), u322((i // ST) & 1))
            txl.ptx.ld.shared.s32(tt, meta.ptr_to([2 * sl]))
            txl.ptx.ld.shared.s32(info, meta.ptr_to([2 * sl + 1]))
            with txl.If(info < 0):
                with txl.Then():
                    txl.assign(go, txl.int32(0))
                with txl.Else():
                    with txl.If(active), txl.Then():
                        for k in range(KB):
                            with txl.If(k < (info & 255)), txl.Then():
                                txl.ptx.ld.shared.v4.b32(w[0], w[1], w[2], w[3], stage.ptr_to([(sl * KB + k) * RW + tid * 4]))
                                for p in range(4):
                                    lo = txl.reinterpret("float32", txl.shift_left(w[p], txl.uint32(16)))
                                    hi = txl.reinterpret("float32", w[p] & txl.uint32(0xFFFF0000))
                                    txl.assign(acc[2 * p], acc[2 * p] + lo)
                                    txl.assign(acc[2 * p + 1], acc[2 * p + 1] + hi)
                    with txl.If(info >= 256), txl.Then():
                        with txl.If(active), txl.Then():
                            for p in range(4):
                                txl.ptx["cvt.rn.bf16x2.f32"](w[p], acc[2 * p + 1], acc[2 * p])
                            txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", tt) * RW + tid * 4]), w[0], w[1], w[2], w[3])
                        for p in range(8):
                            txl.assign(acc[p], txl.float32(0))
            txl.cuda.cta_sync()  # every thread is done with slot sl
            with txl.If(txl.And(go > 0, warp == 0)), txl.Then():
                produce(sl)
            txl.assign(i, i + 1)

    return entry_function(gather_sum_tma.compile())


GS_PRODUCER_WARP = True  # tma3: a dedicated producer warp and per-slot empty barriers (no CTA barrier per unit)
GS_SPIN = True  # tma3 consumers wait with a hint-free try_wait spin
GS3 = True  # multi-token units (below) instead of one token per unit
GS3_CFG = {2048: (4, 3, 4), 2880: (4, 2, 4)}  # by H: (pipeline stages, CTAs per SM, grouped rows per unit); producer-warp sweep
LAST_BIT = 1 << 30
NOROW_BIT = 1 << 29


@functools.cache
def gather_sum_tma3_kernel(G: int, H: int):
    """
    Persistent: warp 0 packs the grouped rows of consecutive tokens (ascending expert order within a
    token) into units of KB entries, one bulk copy per row into an ST-deep ring; an entry is (token,
    last-of-token, no-row). Every thread adds a unit's rows; a token's last entry rounds and stores it.
    Token masks are read 32 tokens at a time (one per lane), the next window prefetched.
    """
    V = H // 8
    NW = (V + 31) // 32
    ST, CPS, KB = GS3_CFG.get(H, (2, 2, 4))
    GRID = NUM_SMS * CPS
    RW = H // 2  # u32 words per row
    RB = H * 2  # bytes per row
    assert RB % 16 == 0 and KB <= 32

    PW = NW if GS_PRODUCER_WARP else 0  # the producer warp (a dedicated extra warp, or consumer warp 0)

    @txl.kernel(warps=NW + (1 if GS_PRODUCER_WARP else 0), arch="sm_100a", grid=lambda p: txl.min(p["T"], GRID))
    def gather_sum_tma3(
        grad: txl.gptr[txl.u32],
        mask: txl.gptr[txl.u64],
        table: txl.gptr[txl.i32],
        out: txl.gptr[txl.u32],
        T: txl.i32,
    ):
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        cta = txl.cta_id()
        grid = txl.min(T, GRID)
        active = tid < V
        smem = txl.smem_pool()
        stage = smem.alloc([ST * KB * RW], txl.u32, align=128)
        bars = smem.alloc([ST], txl.u64, align=8)
        meta_n = smem.alloc([ST], txl.i32)  # entries in the unit (-1: no more units)
        meta_e = smem.alloc([ST * KB], txl.i32)  # token | LAST_BIT | NOROW_BIT
        empt = smem.alloc([ST], txl.u64, align=8)  # slot consumed: one arrival per consumer thread
        with txl.If(tid < ST), txl.Then():
            txl.ptx.mbarrier.init.shared.b64(bars.ptr_to([tid]), txl.uint32(1))
            txl.ptx.mbarrier.init.shared.b64(empt.ptr_to([tid]), txl.uint32(NW * 32))
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        # producer state (warp 0; uniform across its lanes except win / winn / per-entry registers)
        win = txl.local_scalar(txl.u64, init=txl.uint64(0))  # mask of token cta + (wbase + lane) * grid
        winn = txl.local_scalar(txl.u64, init=txl.uint64(0))  # the next window
        wbase = txl.local_scalar(txl.i32, init=txl.int32(0))
        cur = txl.local_scalar(txl.i32, init=txl.int32(0))
        tp = txl.local_scalar(txl.i32, init=cta)
        mp = txl.local_scalar(txl.u64, init=txl.uint64(0))
        low = txl.local_scalar(txl.u64)
        ek = txl.local_scalar(txl.i32)
        em = txl.local_scalar(txl.i32)  # this lane's entry: expert (-1: none)
        tk = txl.local_scalar(txl.i32)  # this lane's entry: token word
        nv = txl.local_scalar(txl.i32)
        nr = txl.local_scalar(txl.i32)
        rr = txl.local_scalar(txl.i32)
        lo32 = txl.local_scalar(txl.u32)
        hi32 = txl.local_scalar(txl.u32)

        pol = txl.local_scalar(txl.u64)
        txl.ptx.createpolicy.fractional.L2__evict_first.b64(pol, txl.float32(1.0))

        def load_window(dst, base):
            t_ = cta + (base + lane) * grid
            txl.assign(dst, txl.uint64(0))
            with txl.If(t_ < T), txl.Then():
                txl.ptx.ld.global_.nc.u64(dst, mask.ptr_to([t_]))

        def fetch_mask():  # mp = mask of window entry cur (all lanes)
            lw = txl.Cast("uint32", win & txl.uint64(0xFFFFFFFF))
            hw = txl.Cast("uint32", win >> txl.uint64(32))
            txl.ptx.shfl_sync.idx.b32(lo32, lw, u322(cur), txl.uint32(31), txl.uint32(FULL))
            txl.ptx.shfl_sync.idx.b32(hi32, hw, u322(cur), txl.uint32(31), txl.uint32(FULL))
            txl.assign(mp, (txl.Cast("uint64", hi32) << txl.uint64(32)) | txl.Cast("uint64", lo32))

        with txl.If(warp == PW), txl.Then():
            load_window(win, txl.int32(0))
            load_window(winn, txl.int32(32))
            fetch_mask()

        def produce(slot):
            txl.assign(nv, txl.int32(0))
            txl.assign(nr, txl.int32(0))
            txl.assign(em, txl.int32(-1))
            for k in range(KB):
                with txl.If(tp < T), txl.Then():
                    txl.assign(low, mp & (txl.uint64(0) - mp))
                    txl.assign(ek, txl.Select(low != txl.uint64(0), i322(txl.popcount(low - txl.uint64(1))), txl.int32(-1)))
                    txl.assign(mp, mp ^ low)
                    word = tp + txl.Select(mp == txl.uint64(0), txl.int32(LAST_BIT), txl.int32(0)) + txl.Select(ek < 0, txl.int32(NOROW_BIT), txl.int32(0))
                    with txl.If(lane == k), txl.Then():
                        txl.assign(em, ek)
                        txl.assign(tk, word)
                    txl.assign(nv, nv + 1)
                    txl.assign(nr, nr + txl.Select(ek >= 0, txl.int32(1), txl.int32(0)))
                    with txl.If(mp == txl.uint64(0)), txl.Then():  # token done: next token of this CTA
                        txl.assign(cur, cur + 1)
                        with txl.If(cur == 32), txl.Then():
                            txl.assign(win, winn)
                            txl.assign(wbase, wbase + 32)
                            txl.assign(cur, txl.int32(0))
                            load_window(winn, wbase + 32)
                        txl.assign(tp, cta + (wbase + cur) * grid)
                        fetch_mask()
            with txl.If(lane == 0), txl.Then():
                txl.ptx.st.shared.s32(meta_n.ptr_to([slot]), txl.Select(nv > 0, nv, txl.int32(-1)))
            with txl.If(lane < nv), txl.Then():
                txl.ptx.st.shared.s32(meta_e.ptr_to([slot * KB + lane]), tk)
            txl.ptx["bar.warp.sync"](txl.uint32(FULL))  # every lane's meta store before the (releasing) arrive
            with txl.If(lane == 0), txl.Then():
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(bars.ptr_to([slot]), u322(nr * RB))
            with txl.If(txl.And(lane < nv, em >= 0)), txl.Then():
                txl.ptx.ld.global_.nc.s32(rr, table.ptr_to([em * T + (tk & txl.int32(NOROW_BIT - 1))]))
                if EVF["gs"]:
                    txl.ptx[BULK_G2S + ".L2::cache_hint"](
                        stage.ptr_to([(slot * KB + lane) * RW]),
                        grad.ptr_to([txl.Cast("int64", rr) * RW]),
                        txl.uint32(RB),
                        bars.ptr_to([slot]),
                        pol,
                    )
                else:
                    txl.ptx[BULK_G2S](
                        stage.ptr_to([(slot * KB + lane) * RW]),
                        grad.ptr_to([txl.Cast("int64", rr) * RW]),
                        txl.uint32(RB),
                        bars.ptr_to([slot]),
                    )

        def spin(bar, parity):
            rdy = txl.local_scalar(txl.u32, init=txl.uint32(0))
            with txl.While(rdy == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.shared.b64(rdy, bar, parity)

        with txl.If(warp == PW), txl.Then():
            for q in range(ST):
                produce(q)
            if GS_PRODUCER_WARP:  # refill each slot once its consumers release it, until the end marker
                pi = txl.local_scalar(txl.i32, init=txl.int32(ST))
                fin = txl.local_scalar(txl.i32, init=txl.Select(nv == 0, txl.int32(1), txl.int32(0)))
                ps = txl.local_scalar(txl.i32)
                with txl.While(fin == 0):
                    txl.assign(ps, pi % ST)
                    spin(empt.ptr_to([ps]), u322((pi // ST - 1) & 1))
                    produce(ps)
                    txl.assign(fin, txl.Select(nv == 0, txl.int32(1), txl.int32(0)))
                    txl.assign(pi, pi + 1)

        acc = txl.alloc_local([8], txl.f32)
        w = txl.alloc_local([4], txl.u32)
        i = txl.local_scalar(txl.i32, init=txl.int32(0))
        go = txl.local_scalar(txl.i32, init=txl.int32(1))
        nn = txl.local_scalar(txl.i32)
        wd = txl.local_scalar(txl.i32)
        sl = txl.local_scalar(txl.i32)
        for p in range(8):
            txl.assign(acc[p], txl.float32(0))
        txl.assign(go, txl.Select(warp < NW, txl.int32(1), txl.int32(0)))
        with txl.While(go > 0):
            txl.assign(sl, i % ST)
            if GS_SPIN:  # try_wait without a suspend-time hint, in a spin loop
                rdy = txl.local_scalar(txl.u32, init=txl.uint32(0))
                with txl.While(rdy == txl.uint32(0)):
                    txl.ptx.mbarrier.try_wait.parity.shared.b64(rdy, bars.ptr_to([sl]), u322((i // ST) & 1))
            else:
                txl.cuda.mbarrier_wait(bars.ptr_to([sl]), u322((i // ST) & 1))
            txl.ptx.ld.shared.s32(nn, meta_n.ptr_to([sl]))
            with txl.If(nn < 0):
                with txl.Then():
                    txl.assign(go, txl.int32(0))
                with txl.Else():
                    with txl.If(active), txl.Then():
                        for k in range(KB):
                            with txl.If(k < nn), txl.Then():
                                txl.ptx.ld.shared.s32(wd, meta_e.ptr_to([sl * KB + k]))
                                with txl.If((wd & txl.int32(NOROW_BIT)) == 0), txl.Then():
                                    txl.ptx.ld.shared.v4.b32(w[0], w[1], w[2], w[3], stage.ptr_to([(sl * KB + k) * RW + tid * 4]))
                                    for p in range(4):
                                        lo = txl.reinterpret("float32", txl.shift_left(w[p], txl.uint32(16)))
                                        hi = txl.reinterpret("float32", w[p] & txl.uint32(0xFFFF0000))
                                        txl.assign(acc[2 * p], acc[2 * p] + lo)
                                        txl.assign(acc[2 * p + 1], acc[2 * p + 1] + hi)
                                with txl.If((wd & txl.int32(LAST_BIT)) != 0), txl.Then():
                                    for p in range(4):
                                        txl.ptx["cvt.rn.bf16x2.f32"](w[p], acc[2 * p + 1], acc[2 * p])
                                    tt = wd & txl.int32(NOROW_BIT - 1)
                                    txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", tt) * RW + tid * 4]), w[0], w[1], w[2], w[3])
                                    for p in range(8):
                                        txl.assign(acc[p], txl.float32(0))
            if GS_PRODUCER_WARP:
                with txl.If(go > 0), txl.Then():  # each consumer thread releases the slot after its own reads
                    txl.ptx.mbarrier.arrive.shared.b64(empt.ptr_to([sl]))
            else:
                txl.cuda.cta_sync()  # every thread is done with slot sl
                with txl.If(txl.And(go > 0, warp == 0)), txl.Then():
                    produce(sl)
            txl.assign(i, i + 1)

    return entry_function(gather_sum_tma3.compile())


def bf16_expert_gather_sum_sm100(grad, maps, num_groups, max_slots):
    T, H = maps.mask.shape[0], grad.shape[1]
    out = torch.empty(T, H, dtype=torch.bfloat16, device=grad.device)
    if T:
        k = (gather_sum_tma3_kernel if GS3 else gather_sum_tma_kernel if GS_TMA else gather_sum_kernel)(num_groups, H)
        k(grad.view(-1).view(torch.uint32), maps.mask.view(torch.uint64), maps.table, out.view(-1).view(torch.uint32), T)
    return out


# ------------------------------------------------------------------------------------- gather_rows

ROWS_PER_CTA = 4


@functools.cache
def gather_rows_kernel(N: int):
    V = N // 8
    NW = 8

    @txl.kernel(warps=NW, arch="sm_100a", grid=lambda p: (p["R"] + ROWS_PER_CTA - 1) // ROWS_PER_CTA)
    def gather_rows(x: txl.gptr[txl.u32], idx: txl.gptr[txl.i32], out: txl.gptr[txl.u32], R: txl.i32):
        b = txl.cta_id()
        tid = txl.thread_id()
        NI = (V * ROWS_PER_CTA + NW * 32 - 1) // (NW * 32)
        w = txl.alloc_local([4 * NI], txl.u32)
        src = txl.alloc_local([NI], txl.i32)
        pol = txl.local_scalar(txl.u64)
        txl.ptx.createpolicy.fractional.L2__evict_first.b64(pol, txl.float32(1.0))
        for i in range(NI):
            f = i * NW * 32 + tid
            r = b * ROWS_PER_CTA + f // V
            with txl.If(txl.And(f < V * ROWS_PER_CTA, r < R)), txl.Then():
                txl.ptx.ld.global_.nc.s32(src[i], idx.ptr_to([r]))
        for i in range(NI):
            f = i * NW * 32 + tid
            r = b * ROWS_PER_CTA + f // V
            with txl.If(txl.And(f < V * ROWS_PER_CTA, r < R)), txl.Then():
                if EVF["rows"]:
                    txl.ptx["ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32"](
                        w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3],
                        x.ptr_to([txl.Cast("int64", src[i]) * (N // 2) + (f % V) * 4]), pol)
                else:
                    txl.ptx.ld.global_.nc.v4.b32(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3],
                                                 x.ptr_to([txl.Cast("int64", src[i]) * (N // 2) + (f % V) * 4]))
        for i in range(NI):
            f = i * NW * 32 + tid
            r = b * ROWS_PER_CTA + f // V
            with txl.If(txl.And(f < V * ROWS_PER_CTA, r < R)), txl.Then():
                txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", r) * (N // 2) + (f % V) * 4]),
                                          w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3])

    return entry_function(gather_rows.compile())


def bf16_expert_gather_rows_sm100(x, idx):
    R, N = idx.shape[0], x.shape[1]
    out = torch.empty(R, N, dtype=torch.bfloat16, device=x.device)
    if R:
        k = gather_rows_kernel(N)
        k(x.view(-1).view(torch.uint32), idx, out.view(-1).view(torch.uint32), R)
    return out


# ------------------------------------------------------------------------------------- segment_sum

# (warps = chunk workers per CTA, bytes per lane per row, cp.async rows in flight per thread)
SEG_CFG = {False: (16, 16, 16), True: (16, 8, 16)}
SEG_F32X2 = False


@functools.cache
def segment_sum_kernel(N: int, has_idx: bool):
    W, VB, D = SEG_CFG[has_idx]
    VE = VB // 2  # bf16 per lane
    VW = VB // 4  # u32 words per lane
    COLS = 32 * VE
    NCB = (N + COLS - 1) // COLS
    NT = W * 32
    assert 256 % D == 0

    @txl.kernel(warps=W, arch="sm_100a", grid=lambda p: p["G"] * NCB)
    def segment_sum(
        x: txl.gptr[txl.u32],
        idx: txl.gptr[txl.i32],
        cu: txl.gptr[txl.i32],
        out: txl.gptr[txl.f32],
        G: txl.i32,
    ):
        b = txl.cta_id()
        g = b // NCB
        cb = b % NCB
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        smem = txl.smem_pool()
        ring = smem.alloc([D * NT * VW], txl.u32)  # slot s of thread t at (s * NT + t) * VB
        part = smem.alloc([W * COLS], txl.f32)
        rows_s = smem.alloc([W * 256], txl.i32, align=16)
        start = txl.local_scalar(txl.i32)
        end = txl.local_scalar(txl.i32)
        txl.ptx.ld.global_.nc.s32(start, cu.ptr_to([g]))
        txl.ptx.ld.global_.nc.s32(end, cu.ptr_to([g + 1]))
        nch = (end - start + 255) // 256
        col = cb * COLS + lane * VE
        colok = col < N
        ccol = txl.Select(colok, col, 0)
        acc = txl.alloc_local([VE], txl.f32)
        acc2 = txl.alloc_local([VW], txl.u64)
        tot = txl.local_scalar(txl.f32, init=txl.float32(0))
        w = txl.alloc_local([VW], txl.u32)
        rv = txl.local_scalar(txl.i32)
        cpy = "cp.async.cg.shared.global" if VB == 16 else "cp.async.ca.shared.global"

        ivr = txl.alloc_local([D], txl.i32)  # grouped rows of the next D chunk rows (has_idx)

        def load_idx(i0):
            # rows i0 .. i0 + D - 1 of this warp's chunk, read ahead of their issues
            for q in range(D // 4):
                txl.ptx.ld.shared.v4.b32(ivr[4 * q], ivr[4 * q + 1], ivr[4 * q + 2], ivr[4 * q + 3],
                                         rows_s.ptr_to([warp * 256 + i0 + 4 * q]))

        def issue(i, slot, cs, ce):
            r = cs + i
            ok = txl.And(r < ce, colok)
            if has_idx:
                txl.assign(rv, ivr[slot])
            else:
                txl.assign(rv, r)
            src_row = txl.Select(ok, rv, 0)
            txl.ptx[cpy](
                ring.ptr_to([(slot * NT + tid) * VW]),
                x.ptr_to([txl.Cast("int64", src_row) * (N // 2) + ccol // 2]),
                VB,
                txl.ptx.pred(txl.Not(ok)),
            )

        def consume(slot):
            if VW == 4:
                txl.ptx.ld.shared.v4.b32(w[0], w[1], w[2], w[3], ring.ptr_to([(slot * NT + tid) * VW]))
            else:
                txl.ptx.ld.shared.v2.b32(w[0], w[1], ring.ptr_to([(slot * NT + tid) * VW]))
            for p in range(VW):
                lo = txl.reinterpret("float32", txl.shift_left(w[p], txl.uint32(16)))
                hi = txl.reinterpret("float32", w[p] & txl.uint32(0xFFFF0000))
                if SEG_F32X2:
                    txl.ptx["add.rn.f32x2"](acc2[p], acc2[p], txl.cuda.make_float2(lo, hi))
                else:
                    txl.assign(acc[2 * p], acc[2 * p] + lo)
                    txl.assign(acc[2 * p + 1], acc[2 * p + 1] + hi)

        with txl.serial(0, (nch + W - 1) // W) as rnd:
            c = rnd * W + warp
            cs = start + c * 256
            ce = txl.min(cs + 256, end)
            for i in range(VE):
                txl.assign(acc[i], txl.float32(0))
            for p in range(VW):
                txl.assign(acc2[p], txl.uint64(0))
            with txl.If(c < nch), txl.Then():
                if has_idx:
                    for q in range(8):
                        r = cs + q * 32 + lane
                        iv = txl.local_scalar(txl.i32, init=txl.int32(0))
                        with txl.If(r < ce), txl.Then():
                            txl.ptx.ld.global_.nc.s32(iv, idx.ptr_to([r]))
                        txl.ptx.st.shared.s32(rows_s.ptr_to([warp * 256 + q * 32 + lane]), iv)
                    txl.ptx["bar.warp.sync"](txl.uint32(FULL))
                if has_idx:
                    load_idx(0)
                for j in range(D):
                    issue(j, j, cs, ce)
                    txl.ptx["cp.async.commit_group"]()
                # rows blk * D + j live in ring slot j: the slot index is static
                with txl.serial(0, 256 // D - 1) as blk:
                    if has_idx:
                        load_idx((blk + 1) * D)
                    for j in range(D):
                        txl.ptx["cp.async.wait_group"](D - 1)
                        consume(j)
                        issue((blk + 1) * D + j, j, cs, ce)
                        txl.ptx["cp.async.commit_group"]()
                txl.ptx["cp.async.wait_group"](0)
                for j in range(D):
                    consume(j)
            if SEG_F32X2:
                for p in range(VW):
                    txl.assign(acc[2 * p], txl.cuda.float2_x(acc2[p]))
                    txl.assign(acc[2 * p + 1], txl.cuda.float2_y(acc2[p]))
            for i in range(VE):
                txl.ptx.st.shared.f32(part.ptr_to([warp * COLS + lane * VE + i]), acc[i])
            txl.cuda.cta_sync()
            # thread tid < COLS owns column cb * COLS + tid: add this round's chunk sums in order
            with txl.If(tid < COLS), txl.Then():
                for q in range(W):
                    pv = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.f32(pv, part.ptr_to([q * COLS + tid]))
                    with txl.If(rnd * W + q < nch), txl.Then():
                        txl.assign(tot, tot + pv)
            txl.cuda.cta_sync()
        oc = cb * COLS + tid
        with txl.If(txl.And(tid < COLS, oc < N)), txl.Then():
            txl.ptx.st.global_.f32(out.ptr_to([txl.Cast("int64", g) * N + oc]), tot)

    return entry_function(segment_sum.compile())


# two-launch split for these has_idx values: (warps per CTA, bytes per lane per row, rows in flight)
SEG2_CFG = {True: (1, 16, 32), False: (1, 16, 32)}
SEG2_MIN_BYTES = {True: 0, False: 32 << 20}  # no-idx: below this one launch (no second launch to hide)


def chunk_of(cu, G, c, cs, ce, lane):
    """
    Warp-cooperative: rows [cs, ce) of global chunk c (chunks numbered segment-major, 256 rows each).
    """
    base = txl.local_scalar(txl.i32, init=txl.int32(0))
    s_ = txl.local_scalar(txl.i32)
    e_ = txl.local_scalar(txl.i32)
    n_ = txl.local_scalar(txl.i32)
    inc = txl.local_scalar(txl.i32)
    o = txl.local_scalar(txl.i32)
    hit = txl.local_scalar(txl.u32)
    with txl.serial(0, (G + 31) // 32) as rd:
        gg = rd * 32 + lane
        txl.assign(n_, txl.int32(0))
        txl.assign(s_, txl.int32(0))
        with txl.If(gg < G), txl.Then():
            txl.ptx.ld.global_.nc.s32(s_, cu.ptr_to([gg]))
            txl.ptx.ld.global_.nc.s32(e_, cu.ptr_to([gg + 1]))
            txl.assign(n_, (e_ - s_ + 255) // 256)
        txl.assign(inc, n_)
        for d in (1, 2, 4, 8, 16):
            txl.ptx.shfl_sync.up.b32(o, inc, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
            txl.assign(inc, inc + txl.Select(lane >= d, o, txl.int32(0)))
        first = base + inc - n_  # this lane's segment's first chunk
        mine = txl.And(c >= first, c < base + inc)
        txl.ptx.vote_sync.ballot.b32(hit, txl.ptx.pred(mine), txl.uint32(FULL))
        with txl.If(hit != txl.uint32(0)), txl.Then():
            src = i322(txl.popcount((hit & (txl.uint32(0) - hit)) - txl.uint32(1)))
            fs = txl.local_scalar(txl.i32)
            ss = txl.local_scalar(txl.i32)
            es = txl.local_scalar(txl.i32)
            txl.ptx.shfl_sync.idx.b32(fs, first, u322(src), txl.uint32(31), txl.uint32(FULL))
            txl.ptx.shfl_sync.idx.b32(ss, s_, u322(src), txl.uint32(31), txl.uint32(FULL))
            txl.ptx.shfl_sync.idx.b32(es, e_, u322(src), txl.uint32(31), txl.uint32(FULL))
            txl.assign(cs, ss + (c - fs) * 256)
            txl.assign(ce, txl.min(ss + (c - fs) * 256 + 256, es))
        tl = txl.local_scalar(txl.i32)
        txl.ptx.shfl_sync.idx.b32(tl, inc, txl.uint32(31), txl.uint32(31), txl.uint32(FULL))
        txl.assign(base, base + tl)


@functools.cache
def segment_chunks_kernel(N: int, has_idx: bool):
    """
    Kernel 1 of 2: CTA (chunk, column group) writes the chunk's fp32 row sum (rows added in order).
    """
    W, VB, D = SEG2_CFG[has_idx]
    VE = VB // 2
    VW = VB // 4
    COLS = 32 * VE  # per warp
    NCB = (N + W * COLS - 1) // (W * COLS)
    NT = W * 32
    assert 256 % D == 0

    @txl.kernel(warps=W, arch="sm_100a", grid=lambda p: p["NC"] * NCB)
    def segment_chunks(
        x: txl.gptr[txl.u32],
        idx: txl.gptr[txl.i32],
        cu: txl.gptr[txl.i32],
        part: txl.gptr[txl.f32],
        G: txl.i32,
        NC: txl.i32,
    ):
        b = txl.cta_id()
        c = b // NCB
        cb = b % NCB
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        smem = txl.smem_pool()
        ring = smem.alloc([D * NT * VW], txl.u32)
        rows_s = smem.alloc([256], txl.i32, align=16)
        cs = txl.local_scalar(txl.i32, init=txl.int32(0))
        ce = txl.local_scalar(txl.i32, init=txl.int32(0))
        chunk_of(cu, G, c, cs, ce, lane)
        if has_idx:
            with txl.serial(0, (256 + NT - 1) // NT) as q:
                k_ = q * NT + tid
                iv = txl.local_scalar(txl.i32, init=txl.int32(0))
                with txl.If(txl.And(k_ < 256, cs + k_ < ce)), txl.Then():
                    txl.ptx.ld.global_.nc.s32(iv, idx.ptr_to([cs + k_]))
                with txl.If(k_ < 256), txl.Then():
                    txl.ptx.st.shared.s32(rows_s.ptr_to([k_]), iv)
            txl.cuda.cta_sync()
        col = (cb * W + warp) * COLS + lane * VE
        colok = col < N
        ccol = txl.Select(colok, col, 0)
        acc = txl.alloc_local([VE], txl.f32)
        w = txl.alloc_local([VW], txl.u32)
        rv = txl.local_scalar(txl.i32)
        cpy = "cp.async.cg.shared.global" if VB == 16 else "cp.async.ca.shared.global"
        pol = txl.local_scalar(txl.u64)
        txl.ptx.createpolicy.fractional.L2__evict_first.b64(pol, txl.float32(1.0))
        ivr = txl.alloc_local([D], txl.i32)

        def load_idx(i0):
            for q in range(D // 4):
                txl.ptx.ld.shared.v4.b32(ivr[4 * q], ivr[4 * q + 1], ivr[4 * q + 2], ivr[4 * q + 3], rows_s.ptr_to([i0 + 4 * q]))

        def issue(i, slot):
            r = cs + i
            ok = txl.And(r < ce, colok)
            if has_idx:
                txl.assign(rv, ivr[slot])
            else:
                txl.assign(rv, r)
            src_row = txl.Select(ok, rv, 0)
            if EVF["seg"]:
                txl.ptx[cpy + ".L2::cache_hint"](
                    ring.ptr_to([(slot * NT + tid) * VW]),
                    x.ptr_to([txl.Cast("int64", src_row) * (N // 2) + ccol // 2]),
                    VB,
                    txl.ptx.pred(txl.Not(ok)),
                    pol,
                )
            else:
                txl.ptx[cpy](
                    ring.ptr_to([(slot * NT + tid) * VW]),
                    x.ptr_to([txl.Cast("int64", src_row) * (N // 2) + ccol // 2]),
                    VB,
                    txl.ptx.pred(txl.Not(ok)),
                )

        def consume(slot):
            if VW == 4:
                txl.ptx.ld.shared.v4.b32(w[0], w[1], w[2], w[3], ring.ptr_to([(slot * NT + tid) * VW]))
            elif VW == 2:
                txl.ptx.ld.shared.v2.b32(w[0], w[1], ring.ptr_to([(slot * NT + tid) * VW]))
            else:
                txl.ptx.ld.shared.b32(w[0], ring.ptr_to([slot * NT + tid]))
            for p in range(VW):
                lo = txl.reinterpret("float32", txl.shift_left(w[p], txl.uint32(16)))
                hi = txl.reinterpret("float32", w[p] & txl.uint32(0xFFFF0000))
                txl.assign(acc[2 * p], acc[2 * p] + lo)
                txl.assign(acc[2 * p + 1], acc[2 * p + 1] + hi)

        for i in range(VE):
            txl.assign(acc[i], txl.float32(0))
        with txl.If((cb * W + warp) * COLS < N), txl.Then():
            if has_idx:
                load_idx(0)
            for j in range(D):
                issue(j, j)
                txl.ptx["cp.async.commit_group"]()
            with txl.serial(0, (ce - cs + D - 1) // D - 1) as blk:  # only the chunk's real rows
                if has_idx:
                    load_idx((blk + 1) * D)
                for j in range(D):
                    txl.ptx["cp.async.wait_group"](D - 1)
                    consume(j)
                    issue((blk + 1) * D + j, j)
                    txl.ptx["cp.async.commit_group"]()
            txl.ptx["cp.async.wait_group"](0)
            for j in range(D):
                consume(j)
            with txl.If(colok), txl.Then():
                for i in range(VE):
                    txl.ptx.st.global_.f32(part.ptr_to([txl.Cast("int64", c) * N + col + i]), acc[i])

    return entry_function(segment_chunks.compile())


@functools.cache
def segment_combine_kernel(N: int):
    """
    Kernel 2 of 2: out[g, col] = chunk sums of segment g added in order (0 for an empty segment).
    """
    NCB = (N + 255) // 256

    @txl.kernel(warps=8, arch="sm_100a", grid=lambda p: p["G"] * NCB)
    def segment_combine(
        cu: txl.gptr[txl.i32],
        part: txl.gptr[txl.f32],
        out: txl.gptr[txl.f32],
        G: txl.i32,
    ):
        b = txl.cta_id()
        g = b // NCB
        col = (b % NCB) * 256 + txl.thread_id()
        base = txl.local_scalar(txl.i32, init=txl.int32(0))
        s_ = txl.local_scalar(txl.i32)
        e_ = txl.local_scalar(txl.i32)
        with txl.serial(0, g) as q:
            txl.ptx.ld.global_.nc.s32(s_, cu.ptr_to([q]))
            txl.ptx.ld.global_.nc.s32(e_, cu.ptr_to([q + 1]))
            txl.assign(base, base + (e_ - s_ + 255) // 256)
        txl.ptx.ld.global_.nc.s32(s_, cu.ptr_to([g]))
        txl.ptx.ld.global_.nc.s32(e_, cu.ptr_to([g + 1]))
        tot = txl.local_scalar(txl.f32, init=txl.float32(0))
        pv = txl.alloc_local([8], txl.f32)
        nc_ = (e_ - s_ + 255) // 256
        with txl.If(col < N), txl.Then():
            with txl.serial(0, (nc_ + 7) // 8) as k8:
                for q in range(8):
                    with txl.If(k8 * 8 + q < nc_), txl.Then():
                        txl.ptx.ld.global_.f32(pv[q], part.ptr_to([txl.Cast("int64", base + k8 * 8 + q) * N + col]))
                for q in range(8):
                    with txl.If(k8 * 8 + q < nc_), txl.Then():
                        txl.assign(tot, tot + pv[q])
            txl.ptx.st.global_.f32(out.ptr_to([txl.Cast("int64", g) * N + col]), tot)

    return entry_function(segment_combine.compile())


def bf16_expert_segment_sum_sm100(x, cu_host, cu, idx=None):
    G, N = len(cu_host) - 1, x.shape[1]
    out = torch.empty(G, N, dtype=torch.float32, device=x.device)
    has_idx = idx is not None
    if G and has_idx in SEG2_CFG and cu_host[-1] * N * 2 >= SEG2_MIN_BYTES[has_idx]:
        NC = sum((cu_host[g + 1] - cu_host[g] + 255) // 256 for g in range(G))
        part = torch.empty(max(NC, 1) * N, dtype=torch.float32, device=x.device)
        if NC:
            segment_chunks_kernel(N, has_idx)(x.view(-1).view(torch.uint32), idx if has_idx else cu, cu, part, G, NC)
        segment_combine_kernel(N)(cu, part, out.view(-1), G)
        return out
    if G:
        k = segment_sum_kernel(N, idx is not None)
        k(x.view(-1).view(torch.uint32), idx if idx is not None else cu, cu, out.view(-1), G)
    return out
