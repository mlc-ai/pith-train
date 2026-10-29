"""
SM100 bf16_expert_slots in TIRx-lite: one launch per call, no memsets.

group_slots: every CTA scans all slots once (warp-aggregated shared histograms of the slots before
its tile and of all slots, plus the token bitmask of the tokens it owns), then ranks its own
1024-slot tile with match.any + per-warp counts. gather_sum is persistent: each CTA walks its
tokens' mask bits in ascending expert order through the per-(token, expert) grouped-row table,
prefetching the next token's mask and table rows under the current token's gradient loads.
gather_rows is a 16-byte vector copy. segment_sum gives each warp one 256-row chunk, streamed
through a per-thread cp.async ring, and adds the chunk sums in order through shared memory.

Evolved with TIRx-harness: run slots, frontier member cp-async-ring (commit d51189b, 9.784x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools
from typing import NamedTuple, Optional

import torch
import tvm_ffi

import tirx_kernels.tirx_lite as txl

__all__ = ["bf16_expert_group_slots_sm100", "bf16_expert_gather_sum_sm100", "bf16_expert_gather_rows_sm100", "bf16_expert_segment_sum_sm100"]

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
                txl.ptx.st.global_.s32(table.ptr_to([xi * G + ei]), row)
            else:
                txl.ptx.st.global_.s32(a_idx.ptr_to([row]), j)

    return group_slots.compile()


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
    k = group_slots_kernel(num_groups, has_x)
    with tvm_ffi.use_torch_stream():
        k(expert_idxs, expand_idx if has_x else expert_idxs, cu, a_idx, row_of, slot_of, mask.view(torch.uint64),
          table, S, num_tokens if has_x else 0)
    if not has_x:
        return GroupSlots(cu, a_idx, row_of, slot_of, None, None)
    return GroupSlots(cu, a_idx, row_of, slot_of, mask, table)


# -------------------------------------------------------------------------------------- gather_sum


GS_BATCH = 8  # experts per prefetched batch
GS_CTAS_PER_SM = 4
NUM_SMS = 148


@functools.cache
def gather_sum_kernel(G: int, H: int):
    V = H // 8
    NW = (V + 31) // 32
    KB = min(GS_BATCH, G)
    GRID = NUM_SMS * GS_CTAS_PER_SM

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
                    txl.ptx.ld.global_.nc.s32(rows[k], table.ptr_to([tt * G + ee[k]]))

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
                txl.ptx.ld.global_.nc.s32(row, table.ptr_to([t * G + i322(txl.popcount(low - txl.uint64(1)))]))
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

    return gather_sum.compile()


def bf16_expert_gather_sum_sm100(grad, maps, num_groups, max_slots):
    T, H = maps.mask.shape[0], grad.shape[1]
    out = torch.empty(T, H, dtype=torch.bfloat16, device=grad.device)
    if T:
        k = gather_sum_kernel(num_groups, H)
        with tvm_ffi.use_torch_stream():
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
        for i in range(NI):
            f = i * NW * 32 + tid
            r = b * ROWS_PER_CTA + f // V
            with txl.If(txl.And(f < V * ROWS_PER_CTA, r < R)), txl.Then():
                txl.ptx.ld.global_.nc.s32(src[i], idx.ptr_to([r]))
        for i in range(NI):
            f = i * NW * 32 + tid
            r = b * ROWS_PER_CTA + f // V
            with txl.If(txl.And(f < V * ROWS_PER_CTA, r < R)), txl.Then():
                txl.ptx.ld.global_.nc.v4.b32(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3],
                                             x.ptr_to([txl.Cast("int64", src[i]) * (N // 2) + (f % V) * 4]))
        for i in range(NI):
            f = i * NW * 32 + tid
            r = b * ROWS_PER_CTA + f // V
            with txl.If(txl.And(f < V * ROWS_PER_CTA, r < R)), txl.Then():
                txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", r) * (N // 2) + (f % V) * 4]),
                                          w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3])

    return gather_rows.compile()


def bf16_expert_gather_rows_sm100(x, idx):
    R, N = idx.shape[0], x.shape[1]
    out = torch.empty(R, N, dtype=torch.bfloat16, device=x.device)
    if R:
        k = gather_rows_kernel(N)
        with tvm_ffi.use_torch_stream():
            k(x.view(-1).view(torch.uint32), idx, out.view(-1).view(torch.uint32), R)
    return out


# ------------------------------------------------------------------------------------- segment_sum

# (warps = chunk workers per CTA, bytes per lane per row, cp.async rows in flight per thread)
SEG_CFG = {False: (16, 16, 16), True: (16, 8, 32)}
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
        rows_s = smem.alloc([W * 256], txl.i32)
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

        def issue(i, slot, cs, ce):
            r = cs + i
            ok = txl.And(r < ce, colok)
            if has_idx:
                txl.ptx.ld.shared.s32(rv, rows_s.ptr_to([warp * 256 + i]))
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
                for j in range(D):
                    issue(j, j, cs, ce)
                    txl.ptx["cp.async.commit_group"]()
                # rows blk * D + j live in ring slot j: the slot index is static
                with txl.serial(0, 256 // D - 1) as blk:
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

    return segment_sum.compile()


def bf16_expert_segment_sum_sm100(x, cu_host, cu, idx=None):
    G, N = len(cu_host) - 1, x.shape[1]
    out = torch.empty(G, N, dtype=torch.float32, device=x.device)
    if G:
        k = segment_sum_kernel(N, idx is not None)
        with tvm_ffi.use_torch_stream():
            k(x.view(-1).view(torch.uint32), idx if idx is not None else cu, cu, out.view(-1), G)
    return out
