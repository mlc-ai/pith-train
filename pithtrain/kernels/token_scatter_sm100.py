"""
SM100 scatter_by_expert as TIRx-lite kernels: a host-count plan kernel when the group counts are
known on the host, else one cooperative kernel with a dynamic row queue.

Host-count path (group_counts given, at most 256 groups, contiguous 16-byte-aligned rows): the counts
ride in as 128 packed int64 kernel parameters, so every CTA block-scans the group starts and pad
rows locally (no histogram, no grid sync, no reverse-index round trip, no cooperative launch); CTA 0
writes grouped_mm_offs / ks_tensor. P = 3 pipeline warps pull tasks of R = 8 rows from a global
queue (a static first task per ring slot, then queue grabs consumed one refill later) in two-stream
order (queue item q takes copy task q / 2 of the first half of the input when q is even, of the
second half when odd; zero-row tasks keep their order). Lane 0 moves a task's R source rows with one
grouped cp.async.bulk G2S into a D = 2 shared-memory ring; each copy lane claims its row's rank
inside its group with one warp-aggregated global atomic per distinct expert (match.any; the two
static slots share one batch), written as the row's reverse index, and stores the row with an
independent hint-free cp.async.bulk S2G. Zero rows (the group padding and the tail up to a multiple
of 1024 rows) are stored from an immutable zeroed shared buffer in one terminal bulk group per warp.
Each (device, stream) keeps two counter banks used alternately: a call clears the bank the next
call uses, so no CTA resets counters at exit. Expert ids are read as their low 32-bit word on this
path (the contract bounds them to [0, num_groups)).

Cooperative fallback (no host counts, more than 256 groups, strided or unaligned rows), one launch,
grid = co-resident CTAs (cooperative launch, txl.cuda.grid_sync):
  1. CTA c histograms its token slice [c*tpc, (c+1)*tpc) in shared memory; the shared atomic's
     return is the token's rank within (CTA, expert). Each nonzero CTA count claims its base inside
     the expert with one global atomic on a persistent counter array, and a second pass parks
     (e << 32 | base + rank) in reverse_shuffle_idxs.
  2. grid sync.
  3. every CTA reads the expert totals and block-scans padded sizes (group starts) and pad rows;
     CTA 0 writes grouped_mm_offs / ks_tensor.
  4. warps pull tasks of R rows from a global queue (copy tasks, then zero-row tasks), so CTAs that
     get more bandwidth do more rows. A copy row's destination is start[e] + (low word of rev).
     Bulk path (contiguous, 16-byte aligned rows): lane 0 moves all R source rows with one grouped
     cp.async.bulk G2S and one mbarrier; lanes j < R scatter the rows with independent S2G copies.
     Each warp uses a D-deep shared-memory task ring and issues the first D - 1 loads before the
     histogram. Arbitrary row strides and unaligned rows retain the register path.
  5. the last CTA to finish re-zeroes the persistent counters and the queue for the next call.
Without host counts, ks is read back with one device-to-host copy and stream sync (as before).
Rows are raw bytes: kernels are keyed on (row bytes, vector width) only, shared by every dtype.

Evolved with TIRx-harness: run token_scatter, frontier member host-plan-tma-ring (commit 00207e06,
179.456 us vs 561.641 us summed over the task's 3 rows, 3.13x over the previous SM100 path in the
task's benchmark). Compiled once per process by tvm_tirx.tirx.compile(..., pipeline="tirx") and
launched through tvm-ffi on torch's current stream. The fallback launch is cooperative: every CTA
must be resident at once, so it runs only when the whole grid fits beside whatever other streams
are running.
"""

import functools
import inspect
import math
import os
from typing import Optional

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})  # nvcc, as TVM_CUDA_COMPILE_MODE=nvcc chose before

__all__ = ["scatter_by_expert_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


GEMM_ALLOC_ALIGNMENT = 1024
WARPS = int(os.environ.get("TS_WARPS", 8))
NT = WARPS * 32
GMAX = 1024  # largest supported num_groups
IPT = GMAX // NT  # groups per thread in the block scan
KREG = 2  # tokens per thread whose (expert, rank) stay in registers through phase 1
CHUNK_BATCH = 32  # register path: vector chunks per lane loaded before storing
TASK_ROWS = 4  # register path: rows per copy task
ZERO_ROWS = 1  # register path: rows per zero task (fine-grained tail)
BULK_R = int(os.environ.get("TS_R", 8))  # bulk path: rows per task (lanes per warp)
BULK_D = int(os.environ.get("TS_D", 2))  # bulk path: ring depth in tasks
BULK_P = int(os.environ.get("TS_P", 3))  # bulk path: pipeline warps
BULK_EARLY = int(os.environ.get("TS_EARLY", 1))  # issue the first loads before the histogram
BULK = int(os.environ.get("TS_BULK", 1))  # 0 forces the register path
VMAX = int(os.environ.get("TS_VMAX", 16))  # widest register access (32: 256-bit ld/st)
RING_BUDGET = 192 * 1024
ZB = 4096  # zero source bytes in shared memory
POLICY = {0: 0x1000000000000000, 1: 0x12F0000000000000, 2: 0x14F0000000000000}  # L2 normal / evict_first / evict_last
LD_POL = int(os.environ.get("TS_LDPOL", 0))  # source rows: evict_normal (hints screened neutral-to-worse)
ST_POL = int(os.environ.get("TS_STPOL", 0))  # output rows: evict_normal
G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
S2G = "cp.async.bulk.global.shared::cta.bulk_group"
# persistent scratch words: [0, GMAX) expert counters, GMAX queue head, GMAX + 1 done counter
STATE_WORDS = GMAX + 2

pinned: dict = {}
state2: dict = {}


def pinned_buffer(numel):
    buf = pinned.get(numel)
    if buf is None:
        buf = torch.empty(numel, dtype=torch.int32, device="cpu", pin_memory=True)
        pinned[numel] = buf
    return buf


def ld_vec(vec, regs, ptr, pol=0):
    if vec == 16 and pol:
        txl.ptx["ld.global.L2::cache_hint.v4.b32"](regs[0], regs[1], regs[2], regs[3], ptr, txl.uint64(POLICY[pol]))
    elif vec == 32:
        txl.ptx["ld.global.v8.b32"](*[regs[i] for i in range(8)], ptr)
    elif vec == 16:
        txl.ptx["ld.global.v4.b32"](regs[0], regs[1], regs[2], regs[3], ptr)
    elif vec == 8:
        txl.ptx["ld.global.v2.b32"](regs[0], regs[1], ptr)
    elif vec == 4:
        txl.ptx["ld.global.b32"](regs[0], ptr)
    elif vec == 2:
        txl.ptx["ld.global.b16"](regs[0], ptr)
    else:
        txl.ptx["ld.global.b8"](regs[0], ptr)


def st_vec(vec, regs, ptr, pol=0):
    if vec == 16 and pol:
        txl.ptx["st.global.L2::cache_hint.v4.b32"](ptr, regs[0], regs[1], regs[2], regs[3], txl.uint64(POLICY[pol]))
    elif vec == 32:
        txl.ptx["st.global.v8.b32"](ptr, *[regs[i] for i in range(8)])
    elif vec == 16:
        txl.ptx["st.global.v4.b32"](ptr, regs[0], regs[1], regs[2], regs[3])
    elif vec == 8:
        txl.ptx["st.global.v2.b32"](ptr, regs[0], regs[1])
    elif vec == 4:
        txl.ptx["st.global.b32"](ptr, regs[0])
    elif vec == 2:
        txl.ptx["st.global.b16"](ptr, regs[0])
    else:
        txl.ptx["st.global.b8"](ptr, regs[0])


def reg_dtype(vec):
    return {32: "uint32", 16: "uint32", 8: "uint32", 4: "uint32", 2: "uint16", 1: "uint8"}[vec]


def nregs(vec):
    return {32: 8, 16: 4, 8: 2}.get(vec, 1)


def bulk_cfg(row_bytes, vec, contiguous):
    """
    (pipeline warps, rows per task, ring depth) for the bulk path, or None for the register path.
    """
    if vec % 16 or not BULK or not contiguous:
        return None
    p, r, d = min(BULK_P, WARPS), BULK_R, BULK_D
    if p * r * d * row_bytes <= RING_BUDGET:
        return (p, r, d)
    for p, r, d in ((p, 1, d), (WARPS, 1, 2), (4, 1, 2), (2, 1, 2), (1, 1, 2)):
        if p * r * d * row_bytes <= RING_BUDGET:
            return (p, r, d)
    return None


@functools.cache
def init_kernel():
    @txl.kernel()
    def token_scatter_state_init(state: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=128))
        with txl.serial(txl.thread_id(), STATE_WORDS, step=128) as i:
            txl.ptx["st.global.s32"](state.ptr_to([txl.cast(i, "int64")]), txl.int32(0))

    return entry_function(token_scatter_state_init.compile(TARGET))


hstate: dict = {}


def get_host_state(device):
    """
    Two zeroed counter banks for the host-count kernel and the index of the bank the next call uses.
    """
    # one pair of banks per (device, stream): the bank alternation relies on stream order, so
    # launches on two streams never share them
    key = (device.type, device.index, torch.cuda.current_stream(device).cuda_stream)
    hs = hstate.get(key)
    if hs is None:
        banks = [torch.empty((STATE_WORDS,), device=device, dtype=torch.int32) for _ in range(2)]
        for b in banks:
            init_kernel()(b)
        hs = hstate[key] = [banks, 0]
    return hs


def get_state(device):
    # one scratch block per (device, stream): launches on two streams never share the counters
    key = (device.type, device.index, torch.cuda.current_stream(device).cuda_stream)
    st = state2.get(key)
    if st is None:
        st = torch.empty((STATE_WORDS,), device=device, dtype=torch.int32)
        init_kernel()(st)
        state2[key] = st
    return st


@functools.cache
def build(row_bytes: int, vec: int, cfg, instr: bool = False):
    bulk = cfg is not None
    nchunks = row_bytes // vec
    lane_iters = (nchunks + 31) // 32
    full = nchunks % 32 == 0
    nreg = nregs(vec)
    rdt = reg_dtype(vec)
    QHEAD = GMAX
    DONE = GMAX + 1
    if bulk:
        P, R, D = cfg
        zb = min(row_bytes, ZB)
    else:
        R = TASK_ROWS

    @txl.kernel()
    def token_scatter(x: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), eidx: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), rev: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), ks_t: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), state: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), dbg: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), m: txl.int32, G: txl.int32, align: txl.int32, src_stride: txl.int64, tpc: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=(WARPS) * 32, cooperative=True), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        c = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        pool = txl.smem_pool()
        s_cnt = pool.alloc((GMAX,), "int32")
        s_tot = pool.alloc((GMAX,), "int32")
        s_start = pool.alloc((GMAX,), "int32")
        s_zp = pool.alloc((GMAX,), "int32")
        s_wa = pool.alloc((WARPS,), "int32")
        s_wb = pool.alloc((WARPS,), "int32")
        if bulk:
            s_mbar = pool.alloc((P * D,), "uint64", align=8)
            s_zero = pool.alloc((zb,), "uint8", align=128)
            s_ring = pool.alloc((P * D * R * row_bytes,), "uint8", align=128)

        stamp_n = [0]

        def stamp():
            if instr:
                with txl.if_(tid == 0), txl.then_():
                    ts = txl.local_scalar("uint64")
                    txl.ptx["mov.u64"](ts, txl.reinterpret(txl.cuda.mov_sreg(64, "globaltimer"), ty="uint64"))
                    txl.ptx["st.global.u64"](dbg.ptr_to([txl.cast(c, "int64") * 8 + stamp_n[0]]), ts)
            stamp_n[0] += 1

        def bar():
            txl.ptx.bar.sync(txl.uint32(0))

        def lds(buf, i):
            v = txl.local_scalar("int32")
            txl.ptx["ld.shared.s32"](v, buf.ptr_to([i]))
            return v

        def sts(buf, i, v):
            txl.ptx["st.shared.s32"](buf.ptr_to([i]), v)

        def shfl0(v):
            out_v = txl.local_scalar("int32")
            txl.ptx.shfl_sync.idx.b32(out_v, v, txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            return out_v

        def warp_incl(v):
            acc = txl.local_scalar("int32", init=v)
            for step in (1, 2, 4, 8, 16):
                peer = txl.local_scalar("int32")
                txl.ptx.shfl_sync.up.b32(peer, acc, txl.uint32(step), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                txl.assign(acc, txl.Select(lane >= step, acc + peer, acc))
            return acc

        # bulk: the first D - 1 tasks of every pipeline warp are static, the queue hands out the rest
        n_static = num_ctas * P * (D - 1) if bulk else 0

        def grab():
            v = txl.local_scalar("int32", init=txl.int32(0))
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.atom.global_.add.s32(v, state.ptr_to([QHEAD]), txl.int32(1))
                txl.assign(v, v + n_static)
            return v

        n_copy = txl.local_scalar("int32", init=(m + R - 1) // R)

        stamp()
        # ---- phase 0: issue the expert-id loads first (the front's critical path), clear the shared
        # counters; bulk: barriers, zero source, then the first row loads behind the id loads -------
        t0 = c * tpc
        t1 = txl.local_scalar("int32", init=txl.min(m, t0 + tpc))
        # the first KREG tokens of each thread keep (e, rank) in registers; the rest round-trip rev
        k_e64 = []
        for k in range(KREG):
            e64 = txl.local_scalar("int64", init=txl.int64(0))
            with txl.if_(t0 + tid + k * NT < t1), txl.then_():
                txl.ptx["ld.global.nc.s64"](e64, eidx.ptr_to([txl.cast(t0 + tid + k * NT, "int64")]))
            k_e64.append(e64)
        with txl.serial(tid, GMAX, step=NT) as g:
            sts(s_cnt, g, txl.int32(0))

        if bulk:
            pw = txl.local_scalar("int32", init=warp)
            is_pipe = txl.And(warp < P, lane < R)
            is_loader = txl.And(warp < P, lane == 0)
            with txl.serial(tid, zb // 16, step=NT) as q:
                txl.ptx["st.shared.v4.b32"](
                    s_zero.ptr_to([q * 16]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0)
                )
            txl.ptx.fence.proxy.async_.shared__cta()

            def slot(b):
                return pw * D + b

            cur = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
            ph = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(D)]
            pk = [txl.local_scalar("int64", init=txl.int64(0)) for _ in range(D)]

            def issue_load(b):
                t = cur[b] * R
                with txl.if_(txl.And(is_loader, txl.And(cur[b] < n_copy, t < m))), txl.then_():
                    rows = txl.min(txl.int32(R), m - t)
                    copy_bytes = rows * row_bytes
                    mb = s_mbar.ptr_to([slot(b)])
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(mb, txl.cast(copy_bytes, "uint32"))
                    txl.ptx[G2S](
                        s_ring.ptr_to([slot(b) * R * row_bytes]),
                        x.ptr_to([txl.cast(t, "int64") * src_stride]),
                        txl.cast(copy_bytes, "uint32"),
                        mb,
                        txl.uint64(POLICY[LD_POL]),
                    )

            def issue_rev(b):
                t = cur[b] * R + lane
                with txl.if_(txl.And(is_pipe, txl.And(cur[b] < n_copy, t < m))), txl.then_():
                    txl.ptx["ld.global.cg.s64"](pk[b], rev.ptr_to([txl.cast(t, "int64")]))

            with txl.if_(is_loader), txl.then_():
                for b in range(D):
                    txl.ptx.mbarrier.init.shared.b64(s_mbar.ptr_to([slot(b)]), txl.uint32(1))
                txl.ptx.fence.mbarrier_init.release.cluster()
            with txl.if_(warp < P), txl.then_():
                for b in range(D - 1):
                    txl.assign(cur[b], (c * P + warp) * (D - 1) + b)
                if BULK_EARLY:
                    for b in range(D - 1):
                        issue_load(b)
        bar()

        # ---- phase 1: per-CTA histogram, rank within (CTA, expert), claim CTA bases -------
        k_e = [txl.local_scalar("int32", init=txl.cast(k_e64[k], "int32")) for k in range(KREG)]
        k_r = []
        for k in range(KREG):
            r = txl.local_scalar("int32", init=txl.int32(0))
            with txl.if_(t0 + tid + k * NT < t1), txl.then_():
                txl.ptx.atom.shared.add.s32(r, s_cnt.ptr_to([k_e[k]]), txl.int32(1))
            k_r.append(r)
        with txl.serial(t0 + tid + KREG * NT, t1, step=NT) as t:
            e64 = txl.local_scalar("int64")
            txl.ptx["ld.global.nc.s64"](e64, eidx.ptr_to([txl.cast(t, "int64")]))
            e = txl.cast(e64, "int32")
            r = txl.local_scalar("int32")
            txl.ptx.atom.shared.add.s32(r, s_cnt.ptr_to([e]), txl.int32(1))
            packed = txl.bitwise_or(txl.shift_left(e64, txl.int64(32)), txl.cast(r, "int64"))
            txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(t, "int64")]), packed)
        bar()
        with txl.serial(tid, G, step=NT) as g:
            cnt = lds(s_cnt, g)
            with txl.if_(cnt > 0), txl.then_():
                base = txl.local_scalar("int32")
                txl.ptx.atom.global_.add.s32(base, state.ptr_to([txl.cast(g, "int64")]), cnt)
                sts(s_tot, g, base)
        bar()
        for k in range(KREG):
            t = t0 + tid + k * NT
            with txl.if_(t < t1), txl.then_():
                packed = txl.bitwise_or(
                    txl.shift_left(txl.cast(k_e[k], "int64"), txl.int64(32)),
                    txl.cast(k_r[k] + lds(s_tot, k_e[k]), "int64"),
                )
                txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(t, "int64")]), packed)
        # same thread -> same tokens as the first pass, so its own rev stores are visible
        with txl.serial(t0 + tid + KREG * NT, t1, step=NT) as t:
            packed = txl.local_scalar("int64")
            txl.ptx["ld.global.s64"](packed, rev.ptr_to([txl.cast(t, "int64")]))
            e = txl.cast(txl.shift_right(packed, txl.int64(32)), "int32")
            txl.ptx["st.global.s64"](
                rev.ptr_to([txl.cast(t, "int64")]), packed + txl.cast(lds(s_tot, e), "int64")
            )
        stamp()
        txl.cuda.grid_sync()
        stamp()

        # ---- phase 3: expert totals, block scan of padded sizes and pad rows ----------------
        ks_l = []
        pad_l = []
        for j in range(IPT):
            g = tid * IPT + j
            cnt = txl.local_scalar("int32", init=txl.int32(0))
            with txl.if_(g < G), txl.then_():
                txl.ptx["ld.global.cg.s32"](cnt, state.ptr_to([txl.cast(g, "int64")]))
            ks = txl.local_scalar("int32", init=(cnt + align - 1) // align * align)
            ks_l.append(ks)
            pad_l.append(ks - cnt)
        bar()  # s_tot held the CTA bases until here
        for j in range(IPT):
            sts(s_tot, tid * IPT + j, ks_l[j] - pad_l[j])
        a = ks_l[0]
        b_ = pad_l[0]
        for j in range(1, IPT):
            a = a + ks_l[j]
            b_ = b_ + pad_l[j]
        a_s = txl.local_scalar("int32", init=a)
        b_s = txl.local_scalar("int32", init=b_)
        ia = warp_incl(a_s)
        ib = warp_incl(b_s)
        with txl.if_(lane == 31), txl.then_():
            sts(s_wa, warp, ia)
            sts(s_wb, warp, ib)
        bar()
        ea = txl.local_scalar("int32", init=ia - a_s)
        eb = txl.local_scalar("int32", init=ib - b_s)
        M = txl.local_scalar("int32", init=txl.int32(0))
        ZP = txl.local_scalar("int32", init=txl.int32(0))
        for w in range(WARPS):
            wa = lds(s_wa, w)
            wb = lds(s_wb, w)
            txl.assign(ea, ea + txl.Select(warp > w, wa, txl.int32(0)))
            txl.assign(eb, eb + txl.Select(warp > w, wb, txl.int32(0)))
            txl.assign(M, M + wa)
            txl.assign(ZP, ZP + wb)
        for j in range(IPT):
            g = tid * IPT + j
            sts(s_start, g, ea)
            sts(s_zp, g, eb)
            with txl.if_(txl.And(c == 0, g < G)), txl.then_():
                txl.ptx["st.global.s32"](offs.ptr_to([txl.cast(g, "int64")]), ea + ks_l[j])
                txl.ptx["st.global.s32"](ks_t.ptr_to([txl.cast(g, "int64")]), ks_l[j])
            txl.assign(ea, ea + ks_l[j])
            txl.assign(eb, eb + pad_l[j])
        bar()
        stamp()

        Mr = (M + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT
        Z = txl.local_scalar("int32", init=ZP + (Mr - M))
        n_task = txl.local_scalar("int32", init=n_copy + (Z + R - 1) // R)

        def zero_row(z):
            zrow = txl.local_scalar("int32")
            with txl.if_(z < ZP):
                with txl.then_():
                    lo = txl.local_scalar("int32", init=txl.int32(0))
                    hi = txl.local_scalar("int32", init=G)
                    # largest g with zp[g] <= z
                    with txl.while_(hi - lo > 1):
                        mid = (lo + hi) >> 1
                        with txl.if_(lds(s_zp, mid) <= z):
                            with txl.then_():
                                txl.assign(lo, mid)
                            with txl.else_():
                                txl.assign(hi, mid)
                    txl.assign(zrow, lds(s_start, lo) + lds(s_tot, lo) + (z - lds(s_zp, lo)))
                with txl.else_():
                    txl.assign(zrow, M + (z - ZP))
            return zrow

        def dest_of(packed):
            e = txl.cast(txl.shift_right(packed, txl.int64(32)), "int32")
            w = txl.cast(txl.bitwise_and(packed, txl.int64(0xFFFFFFFF)), "int32")
            return txl.local_scalar("int64", init=txl.cast(lds(s_start, e) + w, "int64"))

        if bulk:
            # ---- phase 4 (bulk): D-deep ring of row slots per lane, tasks from the queue ------
            def process(b):
                task = cur[b]
                with txl.if_(is_pipe), txl.then_():
                    with txl.if_(task < n_copy):
                        with txl.then_():
                            t = task * R + lane
                            with txl.if_(t < m), txl.then_():
                                dest = dest_of(pk[b])
                                txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(t, "int64")]), dest)
                                txl.cuda.mbarrier_wait(s_mbar.ptr_to([slot(b)]), ph[b])
                                txl.assign(ph[b], ph[b] ^ txl.uint32(1))
                                txl.ptx[S2G](
                                    out.ptr_to([dest * txl.int64(row_bytes)]),
                                    s_ring.ptr_to([(slot(b) * R + lane) * row_bytes]),
                                    txl.uint32(row_bytes),
                                )
                        with txl.else_():
                            z = (task - n_copy) * R + lane
                            with txl.if_(z < Z), txl.then_():
                                dst = txl.local_scalar(
                                    "int64", init=txl.cast(zero_row(z), "int64") * txl.int64(row_bytes)
                                )
                                for q in range(0, row_bytes, zb):
                                    txl.ptx[S2G](
                                        out.ptr_to([dst + txl.int64(q)]),
                                        s_zero.ptr_to([0]),
                                        txl.uint32(min(zb, row_bytes - q)),
                                    )
                    txl.ptx.cp.async_.bulk.commit_group()

            with txl.if_(warp < P), txl.then_():
                if not BULK_EARLY:
                    for b in range(D - 1):
                        issue_load(b)
                for b in range(D - 1):
                    issue_rev(b)
                with txl.while_(cur[0] < n_task):
                    for b in range(D):
                        nxt = grab()
                        process(b)
                        nb = (b + D - 1) % D
                        txl.assign(cur[nb], shfl0(nxt))
                        with txl.if_(is_pipe), txl.then_():
                            txl.ptx.cp.async_.bulk.wait_group.read(1)
                        issue_load(nb)
                        issue_rev(nb)
                with txl.if_(is_pipe), txl.then_():
                    txl.ptx.cp.async_.bulk.wait_group(0)
        else:
            # ---- phase 4 (registers): warp per row, loads before stores --------------------
            regs = txl.alloc_local((CHUNK_BATCH * nreg,), rdt)
            zregs = txl.alloc_local((nreg,), rdt)
            for q in range(nreg):
                txl.assign(zregs[q], txl.Cast(rdt, 0))

            def copy_row(src_base, dst_base):
                for b0 in range(0, lane_iters, CHUNK_BATCH):
                    nb = min(CHUNK_BATCH, lane_iters - b0)
                    for j in range(nb):
                        k = b0 + j
                        off = txl.cast((k * 32 + lane) * vec, "int64")
                        r = [regs[j * nreg + q] for q in range(nreg)]
                        if full or (k + 1) * 32 <= nchunks:
                            ld_vec(vec, r, x.ptr_to([src_base + off]), LD_POL)
                        else:
                            with txl.if_(k * 32 + lane < nchunks), txl.then_():
                                ld_vec(vec, r, x.ptr_to([src_base + off]), LD_POL)
                    for j in range(nb):
                        k = b0 + j
                        off = txl.cast((k * 32 + lane) * vec, "int64")
                        r = [regs[j * nreg + q] for q in range(nreg)]
                        if full or (k + 1) * 32 <= nchunks:
                            st_vec(vec, r, out.ptr_to([dst_base + off]), ST_POL)
                        else:
                            with txl.if_(k * 32 + lane < nchunks), txl.then_():
                                st_vec(vec, r, out.ptr_to([dst_base + off]), ST_POL)

            rpb = max(1, min(TASK_ROWS, CHUNK_BATCH // lane_iters))  # rows per register batch

            def copy_rows(srcs, dsts, valids):
                """
                Load every row of the batch, then store them: rpb rows of loads in flight per warp.
                """
                n = len(srcs)
                for i in range(n):
                    with txl.if_(valids[i]), txl.then_():
                        for k in range(lane_iters):
                            off = txl.cast((k * 32 + lane) * vec, "int64")
                            r = [regs[(i * lane_iters + k) * nreg + q] for q in range(nreg)]
                            if full or (k + 1) * 32 <= nchunks:
                                ld_vec(vec, r, x.ptr_to([srcs[i] + off]), LD_POL)
                            else:
                                with txl.if_(k * 32 + lane < nchunks), txl.then_():
                                    ld_vec(vec, r, x.ptr_to([srcs[i] + off]), LD_POL)
                for i in range(n):
                    with txl.if_(valids[i]), txl.then_():
                        for k in range(lane_iters):
                            off = txl.cast((k * 32 + lane) * vec, "int64")
                            r = [regs[(i * lane_iters + k) * nreg + q] for q in range(nreg)]
                            if full or (k + 1) * 32 <= nchunks:
                                st_vec(vec, r, out.ptr_to([dsts[i] + off]), ST_POL)
                            else:
                                with txl.if_(k * 32 + lane < nchunks), txl.then_():
                                    st_vec(vec, r, out.ptr_to([dsts[i] + off]), ST_POL)

            def zero_store(dst_base):
                for k in range(lane_iters):
                    off = txl.cast((k * 32 + lane) * vec, "int64")
                    if full or (k + 1) * 32 <= nchunks:
                        st_vec(vec, zregs, out.ptr_to([dst_base + off]), ST_POL)
                    else:
                        with txl.if_(k * 32 + lane < nchunks), txl.then_():
                            st_vec(vec, zregs, out.ptr_to([dst_base + off]), ST_POL)

            n_rtask = txl.local_scalar("int32", init=n_copy + (Z + ZERO_ROWS - 1) // ZERO_ROWS)
            task = txl.local_scalar("int32", init=shfl0(grab()))
            with txl.while_(task < n_rtask):
                nxt = grab()
                with txl.if_(task < n_copy):
                    with txl.then_():
                        rows = [txl.local_scalar("int32", init=task * TASK_ROWS + j) for j in range(TASK_ROWS)]
                        dests = []
                        for j in range(TASK_ROWS):
                            packed = txl.local_scalar("int64")
                            txl.ptx["ld.global.cg.s64"](
                                packed, rev.ptr_to([txl.cast(txl.min(rows[j], m - 1), "int64")])
                            )
                            dests.append(packed)
                        dests = [dest_of(p) for p in dests]
                        # every lane read rev[rows] before lane 0 overwrites it
                        txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
                        for j in range(TASK_ROWS):
                            with txl.if_(txl.And(rows[j] < m, lane == 0)), txl.then_():
                                txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(rows[j], "int64")]), dests[j])
                        if rpb == 1:
                            for j in range(TASK_ROWS):
                                with txl.if_(rows[j] < m), txl.then_():
                                    copy_row(
                                        txl.cast(rows[j], "int64") * src_stride, dests[j] * txl.int64(row_bytes)
                                    )
                        else:
                            for j0 in range(0, TASK_ROWS, rpb):
                                js = list(range(j0, min(TASK_ROWS, j0 + rpb)))
                                copy_rows(
                                    [txl.cast(rows[j], "int64") * src_stride for j in js],
                                    [dests[j] * txl.int64(row_bytes) for j in js],
                                    [rows[j] < m for j in js],
                                )
                    with txl.else_():
                        zt = task - n_copy
                        for j in range(ZERO_ROWS):
                            z = txl.local_scalar("int32", init=zt * ZERO_ROWS + j)
                            with txl.if_(z < Z), txl.then_():
                                zero_store(txl.cast(zero_row(z), "int64") * txl.int64(row_bytes))
                txl.assign(task, shfl0(nxt))
        bar()
        stamp()

        # ---- phase 5: the last CTA out resets the persistent counters ---------------------
        with txl.if_(tid == 0), txl.then_():
            old = txl.local_scalar("int32")
            txl.ptx.atom.acq_rel.gpu.global_.add.s32(
                old, state.ptr_to([DONE]), txl.int32(1)
            )
            sts(s_wa, 0, old)
        bar()
        with txl.if_(lds(s_wa, 0) == num_ctas - 1), txl.then_():
            with txl.serial(tid, G, step=NT) as g:
                txl.ptx["st.global.s32"](state.ptr_to([txl.cast(g, "int64")]), txl.int32(0))
            with txl.if_(tid == 0), txl.then_():
                txl.ptx["st.global.s32"](state.ptr_to([QHEAD]), txl.int32(0))
                txl.ptx["st.global.s32"](state.ptr_to([DONE]), txl.int32(0))

    return entry_function(token_scatter.compile(TARGET))


HOST_PARAMS = 128  # fast host-count path: 256 experts packed two per int64 parameter
HOST_GMAX = 2 * HOST_PARAMS
HOST = int(os.environ.get("TS_HOST", 1))


@functools.cache
def build_host(row_bytes: int, cfg):
    """
    Host-count path. Group starts follow from the counts passed as kernel parameters, so there is no
    histogram, no grid sync and no rev round trip: each copy row claims its rank inside its group
    with one global atomic issued inside the TMA pipeline, a ring step before the row is stored.
    """
    P, R, D = cfg
    assert D == 2 and 2 * R <= 32
    zb = min(row_bytes, ZB)
    QHEAD = GMAX
    IPH = HOST_GMAX // NT  # groups per thread in the block scan
    store_warps = list(range(P, WARPS)) if P < WARPS else list(range(WARPS))

    def body(x, eidx, out, rev, offs, ks_t, state, state_nx, m, G, align, num_ctas, *hc):
        c = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        pool = txl.smem_pool()
        s_hc = pool.alloc((HOST_PARAMS,), "int64")
        s_tot = pool.alloc((HOST_GMAX,), "int32")
        s_start = pool.alloc((HOST_GMAX,), "int32")
        s_zp = pool.alloc((HOST_GMAX,), "int32")
        s_wa = pool.alloc((WARPS,), "int32")
        s_wb = pool.alloc((WARPS,), "int32")
        s_mbar = pool.alloc((P * D,), "uint64", align=8)
        s_zero = pool.alloc((zb,), "uint8", align=128)
        s_ring = pool.alloc((P * D * R * row_bytes,), "uint8", align=128)

        def bar():
            txl.ptx.bar.sync(txl.uint32(0))

        def lds(buf, i):
            v = txl.local_scalar("int32")
            txl.ptx["ld.shared.s32"](v, buf.ptr_to([i]))
            return v

        def sts(buf, i, v):
            txl.ptx["st.shared.s32"](buf.ptr_to([i]), v)

        def shfl0(v):
            out_v = txl.local_scalar("int32")
            txl.ptx.shfl_sync.idx.b32(out_v, v, txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            return out_v

        def warp_incl(v):
            acc = txl.local_scalar("int32", init=v)
            for step in (1, 2, 4, 8, 16):
                peer = txl.local_scalar("int32")
                txl.ptx.shfl_sync.up.b32(peer, acc, txl.uint32(step), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                txl.assign(acc, txl.Select(lane >= step, acc + peer, acc))
            return acc

        n_static = num_ctas * P * D  # every ring slot starts with a static task
        nxt = txl.local_scalar("int32", init=txl.int32(0))  # raw queue head, consumed one refill later

        def grab_into(v):
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.atom.global_.add.s32(v, state.ptr_to([QHEAD]), txl.int32(1))

        n_copy = txl.local_scalar("int32", init=(m + R - 1) // R)
        pw = txl.local_scalar("int32", init=warp)
        is_pipe = txl.And(warp < P, lane < R)
        is_loader = txl.And(warp < P, lane == 0)

        def slot(b):
            return pw * D + b

        # Two-stream copy order: queue item q takes copy task q/2 from the first half of the
        # input (q even) or from the second half (q odd).  ceil-half makes this a bijection on
        # [0, n_copy) for any n_copy; zero tasks (q >= n_copy) keep their order.
        il_half = txl.local_scalar("int32", init=(n_copy + 1) >> 1)

        def remap(raw):
            return txl.Select(
                raw < n_copy,
                txl.Select((raw & 1) == 1, il_half + (raw >> 1), raw >> 1),
                raw,
            )

        cur = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
        ph = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(D)]
        # Expert IDs are contractually nonnegative and below HOST_GMAX, so the
        # low word of their int64 storage is the complete value on this path.
        pe = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
        pr = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
        lb = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
        mst = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(D)]
        bef = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(D)]

        def copy_lane(b):
            t = cur[b] * R + lane
            return txl.And(is_pipe, txl.And(cur[b] < n_copy, t < m))

        def issue_load(b):
            t = cur[b] * R
            with txl.if_(txl.And(is_loader, txl.And(cur[b] < n_copy, t < m))), txl.then_():
                rows = txl.min(txl.int32(R), m - t)
                copy_bytes = rows * row_bytes
                mb = s_mbar.ptr_to([slot(b)])
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(mb, txl.cast(copy_bytes, "uint32"))
                txl.ptx[G2S](
                    s_ring.ptr_to([slot(b) * R * row_bytes]),
                    x.ptr_to([txl.cast(t, "int64") * row_bytes]),
                    txl.cast(copy_bytes, "uint32"),
                    mb,
                    txl.uint64(POLICY[LD_POL]),
                )

        def issue_eidx(b):
            with txl.if_(copy_lane(b)), txl.then_():
                txl.ptx["ld.global.nc.u32"](pe[b], eidx.ptr_to([txl.cast(cur[b] * R + lane, "int64")]))

        def issue_atom(b):
            # Adjacent dispatched rows are commonly expert-minor.  Collapse equal
            # experts within the warp to one global atomic while retaining a
            # unique rank for every participating lane.  Invalid lanes use -1,
            # which is outside the expert-id contract, but still participate in
            # the warp-wide match and shuffle instructions.
            valid = copy_lane(b)
            key = txl.local_scalar(
                "int32", init=txl.Select(valid, pe[b], txl.int32(-1))
            )
            peers = txl.local_scalar("uint32")
            txl.ptx.match.any.sync.b32(peers, key, txl.uint32(0xFFFFFFFF))
            master = txl.local_scalar("uint32")
            txl.ptx.bfind.u32(master, peers)
            count = txl.local_scalar("uint32")
            txl.ptx.popc.b32(count, peers)
            lane_mask = txl.shift_left(txl.uint32(1), txl.cast(lane, "uint32")) - txl.uint32(1)
            txl.ptx.popc.b32(bef[b], txl.bitwise_and(peers, lane_mask))
            txl.assign(mst[b], master)
            with txl.if_(txl.And(valid, lane == txl.cast(master, "int32"))), txl.then_():
                txl.ptx.atom.global_.add.s32(
                    lb[b], state.ptr_to([key]), txl.cast(count, "int32")
                )

        def issue_static_pair_atom():
            # The two static slots are adjacent tasks.  Present both slots as one keyed
            # 2R-lane batch to match.any, halving rank atomics at startup and allowing an
            # expert run that crosses the slot boundary to share a single claim.
            in0 = lane < R
            in1 = txl.And(lane >= R, lane < 2 * R)
            t0 = cur[0] * R + lane
            t1 = cur[1] * R + (lane - R)
            valid0 = txl.And(in0, txl.And(cur[0] < n_copy, t0 < m))
            valid1 = txl.And(in1, txl.And(cur[1] < n_copy, t1 < m))
            e1 = txl.local_scalar("int32")
            src1 = txl.Select(lane < R, lane, lane - R)
            txl.ptx.shfl_sync.idx.b32(
                e1, pe[1], txl.cast(src1, "uint32"),
                txl.uint32(31), txl.uint32(0xFFFFFFFF)
            )
            key = txl.local_scalar(
                "int32",
                init=txl.Select(
                    valid0,
                    pe[0],
                    txl.Select(valid1, e1, txl.int32(-1)),
                ),
            )
            peers = txl.local_scalar("uint32")
            txl.ptx.match.any.sync.b32(peers, key, txl.uint32(0xFFFFFFFF))
            master = txl.local_scalar("uint32")
            txl.ptx.bfind.u32(master, peers)
            count = txl.local_scalar("uint32")
            txl.ptx.popc.b32(count, peers)
            lane_mask = txl.shift_left(txl.uint32(1), txl.cast(lane, "uint32")) - txl.uint32(1)
            before = txl.local_scalar("uint32")
            txl.ptx.popc.b32(before, txl.bitwise_and(peers, lane_mask))
            pair_lb = txl.local_scalar("int32", init=txl.int32(0))
            with txl.if_(txl.And(txl.Or(valid0, valid1), lane == txl.cast(master, "int32"))), txl.then_():
                txl.ptx.atom.global_.add.s32(
                    pair_lb, state.ptr_to([key]), txl.cast(count, "int32")
                )
            # finish_atom consumes these per-slot views later, after the count scan.
            txl.assign(lb[0], pair_lb)
            txl.assign(lb[1], pair_lb)
            txl.assign(mst[0], master)
            txl.assign(bef[0], before)
            master1 = txl.local_scalar("uint32")
            before1 = txl.local_scalar("uint32")
            src_hi = txl.Select(lane < R, lane + R, lane)
            txl.ptx.shfl_sync.idx.b32(
                master1, master, txl.cast(src_hi, "uint32"),
                txl.uint32(31), txl.uint32(0xFFFFFFFF)
            )
            txl.ptx.shfl_sync.idx.b32(
                before1, before, txl.cast(src_hi, "uint32"),
                txl.uint32(31), txl.uint32(0xFFFFFFFF)
            )
            txl.assign(mst[1], master1)
            txl.assign(bef[1], before1)

        def finish_atom(b):
            # warp-converged: the leader's atomic return is consumed only here, just before the store
            base = txl.local_scalar("int32")
            txl.ptx.shfl_sync.idx.b32(
                base, lb[b], mst[b], txl.uint32(31), txl.uint32(0xFFFFFFFF)
            )
            txl.assign(pr[b], txl.Select(copy_lane(b), base + txl.cast(bef[b], "int32"), txl.int32(0)))

        # ---- phase 0: first loads and expert ids, then the host counts into shared memory --------
        with txl.if_(is_loader), txl.then_():
            for b in range(D):
                txl.ptx.mbarrier.init.shared.b64(s_mbar.ptr_to([slot(b)]), txl.uint32(1))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.if_(warp < P), txl.then_():
            for b in range(D):
                txl.assign(cur[b], remap((c * P + warp) * D + b))
            for b in range(D):
                issue_eidx(b)
            grab_into(nxt)
            for b in range(D):
                issue_load(b)
        with txl.serial(tid, zb // 16, step=NT) as q:
            txl.ptx["st.shared.v4.b32"](
                s_zero.ptr_to([q * 16]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0)
            )
        txl.ptx.fence.proxy.async_.shared__cta()
        # The next call uses the other counter bank: clear it now (the previous call that used it has
        # completed in stream order, and nothing in this call touches it), so no CTA resets at exit.
        with txl.if_(c == num_ctas - 1), txl.then_():
            with txl.serial(tid, HOST_GMAX, step=NT) as g:
                txl.ptx["st.global.s32"](state_nx.ptr_to([txl.cast(g, "int64")]), txl.int32(0))
            with txl.if_(tid == 0), txl.then_():
                txl.ptx["st.global.s32"](state_nx.ptr_to([QHEAD]), txl.int32(0))
        for i, w in enumerate(store_warps):
            with txl.if_(txl.And(warp == w, lane == 0)), txl.then_():
                for p in range(i, HOST_PARAMS, len(store_warps)):
                    txl.ptx["st.shared.s64"](s_hc.ptr_to([p]), hc[p])
        with txl.if_(warp < P), txl.then_():
            issue_static_pair_atom()  # rank claims of both static slots overlap the scan
        bar()

        # ---- phase 1: block scan of padded sizes and pad rows -------------------------------
        ks_l = []
        pad_l = []
        for j in range(IPH):
            g = tid * IPH + j
            w64 = txl.local_scalar("int64")
            txl.ptx["ld.shared.s64"](w64, s_hc.ptr_to([g >> 1]))
            half = txl.Select((g & 1) == 1, txl.shift_right(w64, txl.int64(32)), w64)
            cnt = txl.local_scalar("int32", init=txl.Select(g < G, txl.cast(txl.bitwise_and(half, txl.int64(0xFFFFFFFF)), "int32"), txl.int32(0)))
            ks = txl.local_scalar("int32", init=(cnt + align - 1) // align * align)
            ks_l.append(ks)
            pad_l.append(ks - cnt)
            sts(s_tot, g, cnt)
        a = ks_l[0]
        b_ = pad_l[0]
        for j in range(1, IPH):
            a = a + ks_l[j]
            b_ = b_ + pad_l[j]
        a_s = txl.local_scalar("int32", init=a)
        b_s = txl.local_scalar("int32", init=b_)
        ia = warp_incl(a_s)
        ib = warp_incl(b_s)
        with txl.if_(lane == 31), txl.then_():
            sts(s_wa, warp, ia)
            sts(s_wb, warp, ib)
        bar()
        ea = txl.local_scalar("int32", init=ia - a_s)
        eb = txl.local_scalar("int32", init=ib - b_s)
        M = txl.local_scalar("int32", init=txl.int32(0))
        ZP = txl.local_scalar("int32", init=txl.int32(0))
        for w in range(WARPS):
            wa = lds(s_wa, w)
            wb = lds(s_wb, w)
            txl.assign(ea, ea + txl.Select(warp > w, wa, txl.int32(0)))
            txl.assign(eb, eb + txl.Select(warp > w, wb, txl.int32(0)))
            txl.assign(M, M + wa)
            txl.assign(ZP, ZP + wb)
        for j in range(IPH):
            g = tid * IPH + j
            sts(s_start, g, ea)
            sts(s_zp, g, eb)
            with txl.if_(txl.And(c == 0, g < G)), txl.then_():
                txl.ptx["st.global.s32"](offs.ptr_to([txl.cast(g, "int64")]), ea + ks_l[j])
                txl.ptx["st.global.s32"](ks_t.ptr_to([txl.cast(g, "int64")]), ks_l[j])
            txl.assign(ea, ea + ks_l[j])
            txl.assign(eb, eb + pad_l[j])
        bar()

        Mr = (M + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT
        Z = txl.local_scalar("int32", init=ZP + (Mr - M))
        n_task = txl.local_scalar("int32", init=n_copy + (Z + R - 1) // R)

        def zero_row(z):
            zrow = txl.local_scalar("int32")
            with txl.if_(z < ZP):
                with txl.then_():
                    lo = txl.local_scalar("int32", init=txl.int32(0))
                    hi = txl.local_scalar("int32", init=G)
                    with txl.while_(hi - lo > 1):
                        mid = (lo + hi) >> 1
                        with txl.if_(lds(s_zp, mid) <= z):
                            with txl.then_():
                                txl.assign(lo, mid)
                            with txl.else_():
                                txl.assign(hi, mid)
                    txl.assign(zrow, lds(s_start, lo) + lds(s_tot, lo) + (z - lds(s_zp, lo)))
                with txl.else_():
                    txl.assign(zrow, M + (z - ZP))
            return zrow

        # ---- phase 2: D-deep ring of row slots per lane, tasks from the queue ----------------
        def process(b):
            task = cur[b]
            with txl.if_(is_pipe), txl.then_():
                with txl.if_(task < n_copy):
                    with txl.then_():
                        t = task * R + lane
                        with txl.if_(t < m), txl.then_():
                            e = pe[b]
                            dest = txl.local_scalar("int64", init=txl.cast(lds(s_start, e) + pr[b], "int64"))
                            txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(t, "int64")]), dest)
                            txl.cuda.mbarrier_wait(s_mbar.ptr_to([slot(b)]), ph[b])
                            txl.assign(ph[b], ph[b] ^ txl.uint32(1))
                            txl.ptx[S2G](
                                out.ptr_to([dest * txl.int64(row_bytes)]),
                                s_ring.ptr_to([(slot(b) * R + lane) * row_bytes]),
                                txl.uint32(row_bytes),
                            )
                            # Copy groups are committed individually because the
                            # following wait.read protects reuse of this ring slot.
                            txl.ptx.cp.async_.bulk.commit_group()
                    with txl.else_():
                        z = (task - n_copy) * R + lane
                        with txl.if_(z < Z), txl.then_():
                            dst = txl.local_scalar(
                                "int64", init=txl.cast(zero_row(z), "int64") * txl.int64(row_bytes)
                            )
                            for q in range(0, row_bytes, zb):
                                txl.ptx[S2G](
                                    out.ptr_to([dst + txl.int64(q)]),
                                    s_zero.ptr_to([0]),
                                    txl.uint32(min(zb, row_bytes - q)),
                                )
                        # s_zero is immutable, so all terminal zero stores can remain
                        # uncommitted and share one group drained at warp exit.

        with txl.if_(warp < P), txl.then_():
            first = txl.local_scalar("int32", init=txl.int32(1))
            with txl.while_(cur[0] < n_task):
                for b in range(D):
                    with txl.if_(cur[b] < n_copy), txl.then_():
                        finish_atom(b)
                    process(b)
                    nb = (b + D - 1) % D  # the slot processed one step ago
                    with txl.if_(first == 0), txl.then_():
                        txl.assign(cur[nb], remap(shfl0(nxt) + n_static))
                        grab_into(nxt)
                        # Zero refills neither reuse the ring nor need expert/rank
                        # metadata.  Avoid the otherwise unconditional match.any,
                        # bfind, popc, and rank shuffle on the terminal suffix.
                        with txl.if_(cur[nb] < n_copy), txl.then_():
                            with txl.if_(is_pipe), txl.then_():
                                txl.ptx.cp.async_.bulk.wait_group.read(1)
                            issue_load(nb)
                            issue_eidx(nb)
                            issue_atom(nb)
                    txl.assign(first, txl.int32(0))
            with txl.if_(is_pipe), txl.then_():
                txl.ptx.cp.async_.bulk.commit_group()
                txl.ptx.cp.async_.bulk.wait_group(0)
        # No CTA barrier is needed at exit: every issuing lane has drained its
        # own bulk group, and no thread accesses shared state after this point.


    def token_scatter_host(*args):
        num_ctas = args[11]
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=WARPS * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        return body(*args)

    K = inspect.Parameter.POSITIONAL_OR_KEYWORD
    params = [
        inspect.Parameter("x", K, annotation=txl.Tensor((txl.int64(),), txl.u8, data_alignment=1)),
        inspect.Parameter("eidx", K, annotation=txl.Tensor((txl.int64(),), txl.i64, data_alignment=1)),
        inspect.Parameter("out", K, annotation=txl.Tensor((txl.int64(),), txl.u8, data_alignment=1)),
        inspect.Parameter("rev", K, annotation=txl.Tensor((txl.int64(),), txl.i64, data_alignment=1)),
        inspect.Parameter("offs", K, annotation=txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        inspect.Parameter("ks_t", K, annotation=txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        inspect.Parameter("state", K, annotation=txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        inspect.Parameter("state_nx", K, annotation=txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        inspect.Parameter("m", K, annotation=txl.i32),
        inspect.Parameter("G", K, annotation=txl.i32),
        inspect.Parameter("align", K, annotation=txl.i32),
        inspect.Parameter("num_ctas", K, annotation=txl.i32),
    ] + [inspect.Parameter(f"hc{p}", K, annotation=txl.i64) for p in range(HOST_PARAMS)]
    token_scatter_host.__signature__ = inspect.Signature(params)
    kern = txl.kernel()(token_scatter_host)
    return entry_function(kern.compile(TARGET))


@functools.cache
def num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


INSTR = False
DBG = None


def scatter_by_expert_sm100(
    input_tokens: torch.Tensor,
    expert_idxs: torch.Tensor,
    num_groups: int,
    padding_alignment: int = 128,
    group_counts: Optional[list[int]] = None,
):
    """
    scatter_by_expert on SM100 (see the picker's contract).
    """
    global DBG
    num_tokens, hidden = input_tokens.shape[0], input_tokens.shape[1]
    device = input_tokens.device
    if num_tokens == 0:
        return (
            torch.empty((0, hidden), device=device, dtype=input_tokens.dtype),
            torch.empty((0,), device=device, dtype=torch.int64),
            torch.empty((0,), device=device, dtype=torch.int32),
            torch.empty((0,), device=device, dtype=torch.int32),
            [],
        )
    if expert_idxs.shape[0] != num_tokens:
        raise ValueError(f"{expert_idxs.shape[0]=} must equal the input row count {num_tokens=}")
    if num_groups > GMAX:
        raise ValueError(f"num_groups={num_groups} exceeds the supported {GMAX}")
    if input_tokens.stride(1) != 1 and hidden > 1:
        raise ValueError("input_tokens must have unit inner stride")

    esize = input_tokens.element_size()
    row_bytes = hidden * esize
    src_stride = input_tokens.stride(0) * esize
    vec = math.gcd(VMAX, row_bytes, src_stride if num_tokens > 1 else row_bytes, input_tokens.data_ptr() % VMAX or VMAX)

    if group_counts is not None:
        ks = [(n + padding_alignment - 1) // padding_alignment * padding_alignment for n in group_counts]
        max_rows = sum(ks)
    else:
        ks = None
        max_rows = num_tokens + num_groups * (padding_alignment - 1)
    max_rows = (max_rows + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT

    out = torch.empty((max(max_rows, 1), hidden), device=device, dtype=input_tokens.dtype)
    rev = torch.empty((num_tokens,), device=device, dtype=torch.int64)
    offs = torch.empty((num_groups,), device=device, dtype=torch.int32)
    ks_tensor = torch.empty((num_groups,), device=device, dtype=torch.int32)

    sms = num_sms(device.index if device.index is not None else torch.cuda.current_device())
    work_rows = num_tokens + max_rows
    num_ctas = max(1, min(sms, (work_rows + 63) // 64))
    tpc = (num_tokens + num_ctas - 1) // num_ctas
    state = get_state(device)
    if INSTR:
        DBG = torch.zeros((num_ctas * 8,), device=device, dtype=torch.uint64)
        dbg = DBG
    else:
        dbg = state.view(torch.uint64)[:1]

    span = (num_tokens - 1) * input_tokens.stride(0) + hidden
    x_bytes = torch.as_strided(input_tokens, (span,), (1,)).view(torch.uint8)
    cfg = bulk_cfg(row_bytes, vec, src_stride == row_bytes)
    if HOST and not INSTR and cfg is not None and group_counts is not None and num_groups <= HOST_GMAX:
        gc = [int(n) for n in group_counts] + [0] * (HOST_GMAX - num_groups)
        packed = []
        for p in range(HOST_PARAMS):
            word = gc[2 * p] | (gc[2 * p + 1] << 32)
            packed.append(word if word < (1 << 63) else word - (1 << 64))
        hs = get_host_state(device)
        banks, bank = hs
        hs[1] = bank ^ 1
        build_host(row_bytes, cfg)(x_bytes, expert_idxs, out.view(-1).view(torch.uint8), rev, offs, ks_tensor, banks[bank], banks[bank ^ 1], num_tokens, num_groups, padding_alignment, num_ctas, *packed)
        return out[: sum(ks)], rev, offs, ks_tensor, ks
    kern = build(row_bytes, vec, cfg, INSTR)
    kern(x_bytes, expert_idxs, out.view(-1).view(torch.uint8), rev, offs, ks_tensor, state, dbg, num_tokens, num_groups, padding_alignment, src_stride, tpc, num_ctas)
    if ks is None:
        ks_host = pinned_buffer(num_groups)
        ks_host.copy_(ks_tensor, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        ks = ks_host.tolist()
    return out[: sum(ks)], rev, offs, ks_tensor, ks
