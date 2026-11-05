"""
SM100 scatter_by_expert as TIRx-lite kernels: a host-count plan kernel when the group counts are
known on the host, else one cooperative kernel with a dynamic row queue.

Host-count path (group_counts given, at most 256 groups, contiguous 16-byte-aligned rows): the
counts ride in as 128 packed int64 kernel parameters, so every CTA block-scans the group starts and
pad rows locally (no histogram, no grid sync, no reverse-index round trip, no cooperative launch);
CTA 0 writes grouped_mm_offs / ks_tensor. P = 3 pipeline warps pull tasks of R = 8 rows from a
global queue (a static first task per ring slot, then queue grabs consumed one refill later) in
two-stream order (queue item q takes copy task q / 2 of the first half of the input when q is even,
of the second half when odd; zero-row tasks keep their order). Lane 0 moves a task's R source rows
with one grouped cp.async.bulk G2S into a D = 2 shared-memory ring; each copy lane claims its row's
rank inside its group with one warp-aggregated global atomic per distinct expert (match.any; the
two static slots share one batch), written as the row's reverse index, and stores the row with an
independent hint-free cp.async.bulk S2G. Zero rows (the group padding and the tail up to a multiple
of 1024 rows) are stored from an immutable zeroed shared buffer in one terminal bulk group per
warp. Each (device, stream) keeps two counter banks used alternately: a call clears the bank the
next call uses, so no CTA resets counters at exit. Expert ids are read as their low 32-bit word on
this path (the contract bounds them to [0, num_groups)).

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
Without host counts, ks is read back with one device-to-host copy and stream sync.
Rows are raw bytes: kernels are keyed on (row bytes, vector width) only, shared by every dtype.

The fallback launch is cooperative: every CTA must be resident at once, so it runs only when the
whole grid fits beside whatever other streams are running.
"""

import functools
import inspect
import math
import os
from typing import Optional

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["scatter_by_expert_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


GEMM_ALLOC_ALIGNMENT = 1024
WARPS = int(os.environ.get("TS_WARPS", 8))
THREADS = WARPS * 32
MAX_GROUPS = 1024  # largest supported num_groups
GROUPS_PER_THREAD = MAX_GROUPS // THREADS  # groups per thread in the block scan
REG_TOKENS = 2  # tokens per thread whose (expert, rank) stay in registers through phase 1
CHUNK_BATCH = 32  # register path: vector chunks per lane loaded before storing
TASK_ROWS = 4  # register path: rows per copy task
ZERO_ROWS = 1  # register path: rows per zero task (fine-grained tail)
BULK_R = int(os.environ.get("TS_R", 8))  # bulk path: rows per task (lanes per warp)
BULK_D = int(os.environ.get("TS_D", 2))  # bulk path: ring depth in tasks
BULK_P = int(os.environ.get("TS_P", 3))  # bulk path: pipeline warps
BULK_EARLY = int(os.environ.get("TS_EARLY", 1))  # issue the first loads before the histogram
BULK = int(os.environ.get("TS_BULK", 1))  # 0 forces the register path
MAX_VEC_BYTES = int(os.environ.get("TS_VMAX", 16))  # widest register access (32: 256-bit ld/st)
RING_BUDGET = 192 * 1024
ZERO_BYTES = 4096  # zero source bytes in shared memory
L2_POLICY = {0: 0x1000000000000000, 1: 0x12F0000000000000, 2: 0x14F0000000000000}  # L2 normal / evict_first / evict_last
LOAD_POLICY = int(os.environ.get("TS_LDPOL", 0))  # source rows: evict_normal (the other hints are no faster)
STORE_POLICY = int(os.environ.get("TS_STPOL", 0))  # output rows: evict_normal
G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
S2G = "cp.async.bulk.global.shared::cta.bulk_group"
# persistent scratch words: [0, MAX_GROUPS) expert counters, MAX_GROUPS queue head, MAX_GROUPS + 1
# done counter
STATE_WORDS = MAX_GROUPS + 2

pinned_buffers: dict = {}
state_cache: dict = {}


def pinned_buffer(numel):
    buf = pinned_buffers.get(numel)
    if buf is None:
        buf = torch.empty(numel, dtype=torch.int32, device="cpu", pin_memory=True)
        pinned_buffers[numel] = buf
    return buf


def load_vec(vec, regs, ptr, policy=0):
    if vec == 16 and policy:
        txl.ptx["ld.global.L2::cache_hint.v4.b32"](regs[0], regs[1], regs[2], regs[3], ptr, txl.uint64(L2_POLICY[policy]))
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


def store_vec(vec, regs, ptr, policy=0):
    if vec == 16 and policy:
        txl.ptx["st.global.L2::cache_hint.v4.b32"](ptr, regs[0], regs[1], regs[2], regs[3], txl.uint64(L2_POLICY[policy]))
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


def regs_per_vec(vec):
    return {32: 8, 16: 4, 8: 2}.get(vec, 1)


def bulk_cfg(row_bytes, vec, contiguous):
    """
    (pipeline warps, rows per task, ring depth) for the bulk path, or None for the register path.
    """
    if vec % 16 or not BULK or not contiguous:
        return None
    pipe_warps, task_rows, depth = min(BULK_P, WARPS), BULK_R, BULK_D
    if pipe_warps * task_rows * depth * row_bytes <= RING_BUDGET:
        return (pipe_warps, task_rows, depth)
    for pipe_warps, task_rows, depth in ((pipe_warps, 1, depth), (WARPS, 1, 2), (4, 1, 2), (2, 1, 2), (1, 1, 2)):
        if pipe_warps * task_rows * depth * row_bytes <= RING_BUDGET:
            return (pipe_warps, task_rows, depth)
    return None


def sync_block():
    txl.ptx.bar.sync(txl.uint32(0))


def load_shared(buf, idx):
    value = txl.local_scalar("int32")
    txl.ptx["ld.shared.s32"](value, buf.ptr_to([idx]))
    return value


def store_shared(buf, idx, value):
    txl.ptx["st.shared.s32"](buf.ptr_to([idx]), value)


def broadcast_lane0(value):
    """
    Lane 0's value of an int32 in every lane.
    """
    out_value = txl.local_scalar("int32")
    txl.ptx.shfl_sync.idx.b32(out_value, value, txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF))
    return out_value


def warp_inclusive_scan(value, lane):
    acc = txl.local_scalar("int32", init=value)
    for step in (1, 2, 4, 8, 16):
        peer = txl.local_scalar("int32")
        txl.ptx.shfl_sync.up.b32(peer, acc, txl.uint32(step), txl.uint32(0), txl.uint32(0xFFFFFFFF))
        txl.assign(acc, txl.Select(lane >= step, acc + peer, acc))
    return acc


def zero_row_dest(zero_idx, G, M, num_pad_rows, s_start, s_total, s_pad_start):
    """
    Output row of zero row zero_idx: the group padding rows in group order, then the tail from M.
    """
    row = txl.local_scalar("int32")
    with txl.if_(zero_idx < num_pad_rows):
        with txl.then_():
            lo = txl.local_scalar("int32", init=txl.int32(0))
            hi = txl.local_scalar("int32", init=G)
            # largest group with pad_start[group] <= zero_idx
            with txl.while_(hi - lo > 1):
                mid = (lo + hi) >> 1
                with txl.if_(load_shared(s_pad_start, mid) <= zero_idx):
                    with txl.then_():
                        txl.assign(lo, mid)
                    with txl.else_():
                        txl.assign(hi, mid)
            txl.assign(row, load_shared(s_start, lo) + load_shared(s_total, lo) + (zero_idx - load_shared(s_pad_start, lo)))
        with txl.else_():
            txl.assign(row, M + (zero_idx - num_pad_rows))
    return row


@functools.cache
def init_kernel():
    @txl.kernel()
    def token_scatter_state_init(state: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=128))
        with txl.serial(txl.thread_id(), STATE_WORDS, step=128) as i:
            txl.ptx["st.global.s32"](state.ptr_to([txl.cast(i, "int64")]), txl.int32(0))

    return entry_function(token_scatter_state_init.compile(TARGET))


host_state_cache: dict = {}


def get_host_state(device):
    """
    Two zeroed counter banks for the host-count kernel and the index of the bank the next call uses.
    """
    # one pair of banks per (device, stream): the bank alternation relies on stream order, so
    # launches on two streams never share them
    key = (device.type, device.index, torch.cuda.current_stream(device).cuda_stream)
    host_state = host_state_cache.get(key)
    if host_state is None:
        banks = [torch.empty((STATE_WORDS,), device=device, dtype=torch.int32) for _ in range(2)]
        for bank in banks:
            init_kernel()(bank)
        host_state = host_state_cache[key] = [banks, 0]
    return host_state


def get_state(device):
    # one scratch block per (device, stream): launches on two streams never share the counters
    key = (device.type, device.index, torch.cuda.current_stream(device).cuda_stream)
    state = state_cache.get(key)
    if state is None:
        state = torch.empty((STATE_WORDS,), device=device, dtype=torch.int32)
        init_kernel()(state)
        state_cache[key] = state
    return state


@functools.cache
def build(row_bytes: int, vec: int, cfg, instr: bool = False):
    use_bulk = cfg is not None
    num_chunks = row_bytes // vec
    lane_iters = (num_chunks + 31) // 32
    has_no_tail = num_chunks % 32 == 0
    num_regs = regs_per_vec(vec)
    reg_type = reg_dtype(vec)
    QUEUE_HEAD, DONE_COUNTER = MAX_GROUPS, MAX_GROUPS + 1
    if use_bulk:
        P, R, D = cfg
        zero_bytes = min(row_bytes, ZERO_BYTES)
    else:
        R = TASK_ROWS

    @txl.kernel()
    def token_scatter(x: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), eidx: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u8, data_alignment=1), rev: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), ks_t: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), state: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), dbg: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), m: txl.int32, G: txl.int32, align: txl.int32, src_stride: txl.int64, tpc: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=(WARPS) * 32, cooperative=True), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        cta = txl.cta_id()
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5

        pool = txl.smem_pool()
        s_count = pool.alloc((MAX_GROUPS,), "int32")
        s_total = pool.alloc((MAX_GROUPS,), "int32")
        s_start = pool.alloc((MAX_GROUPS,), "int32")
        s_pad_start = pool.alloc((MAX_GROUPS,), "int32")
        s_warp_padded = pool.alloc((WARPS,), "int32")
        s_warp_pad = pool.alloc((WARPS,), "int32")
        if use_bulk:
            s_mbar = pool.alloc((P * D,), "uint64", align=8)
            s_zero = pool.alloc((zero_bytes,), "uint8", align=128)
            s_ring = pool.alloc((P * D * R * row_bytes,), "uint8", align=128)

        num_stamps = [0]

        def stamp():
            """
            With instr, thread 0 of each CTA records the global timer in dbg.
            """
            if instr:
                with txl.if_(tidx == 0), txl.then_():
                    timestamp = txl.local_scalar("uint64")
                    txl.ptx["mov.u64"](timestamp, txl.reinterpret(txl.cuda.mov_sreg(64, "globaltimer"), ty="uint64"))
                    txl.ptx["st.global.u64"](dbg.ptr_to([txl.cast(cta, "int64") * 8 + num_stamps[0]]), timestamp)
            num_stamps[0] += 1

        # bulk: the first D - 1 tasks of every pipeline warp are static, the queue hands out the rest
        num_static = num_ctas * P * (D - 1) if use_bulk else 0

        def grab_task():
            task = txl.local_scalar("int32", init=txl.int32(0))
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.atom.global_.add.s32(task, state.ptr_to([QUEUE_HEAD]), txl.int32(1))
                txl.assign(task, task + num_static)
            return task

        num_copy_tasks = txl.local_scalar("int32", init=(m + R - 1) // R)

        stamp()
        # phase 0: issue the expert-id loads first (the front's critical path), clear the shared
        # counters; bulk: barriers, zero source, then the first row loads behind the id loads
        token_start = cta * tpc
        token_end = txl.local_scalar("int32", init=txl.min(m, token_start + tpc))
        # the first REG_TOKENS tokens of each thread keep (expert, rank) in registers; the rest
        # round-trip rev
        reg_expert64 = []
        for k in range(REG_TOKENS):
            expert64 = txl.local_scalar("int64", init=txl.int64(0))
            with txl.if_(token_start + tidx + k * THREADS < token_end), txl.then_():
                txl.ptx["ld.global.nc.s64"](expert64, eidx.ptr_to([txl.cast(token_start + tidx + k * THREADS, "int64")]))
            reg_expert64.append(expert64)
        with txl.serial(tidx, MAX_GROUPS, step=THREADS) as group_idx:
            store_shared(s_count, group_idx, txl.int32(0))

        if use_bulk:
            pipe_warp = txl.local_scalar("int32", init=warp)
            is_pipe = txl.And(warp < P, lane < R)
            is_loader = txl.And(warp < P, lane == 0)
            with txl.serial(tidx, zero_bytes // 16, step=THREADS) as vec_idx:
                txl.ptx["st.shared.v4.b32"](s_zero.ptr_to([vec_idx * 16]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0))
            txl.ptx.fence.proxy.async_.shared__cta()

            def ring_slot(stage):
                return pipe_warp * D + stage

            cur_task = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
            mbar_phase = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(D)]
            packed_rev = [txl.local_scalar("int64", init=txl.int64(0)) for _ in range(D)]

            def issue_load(stage):
                token = cur_task[stage] * R
                with txl.if_(txl.And(is_loader, txl.And(cur_task[stage] < num_copy_tasks, token < m))), txl.then_():
                    rows = txl.min(txl.int32(R), m - token)
                    copy_bytes = rows * row_bytes
                    mbar = s_mbar.ptr_to([ring_slot(stage)])
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar, txl.cast(copy_bytes, "uint32"))
                    txl.ptx[G2S](s_ring.ptr_to([ring_slot(stage) * R * row_bytes]), x.ptr_to([txl.cast(token, "int64") * src_stride]), txl.cast(copy_bytes, "uint32"), mbar, txl.uint64(L2_POLICY[LOAD_POLICY]))

            def issue_rev(stage):
                token = cur_task[stage] * R + lane
                with txl.if_(txl.And(is_pipe, txl.And(cur_task[stage] < num_copy_tasks, token < m))), txl.then_():
                    txl.ptx["ld.global.cg.s64"](packed_rev[stage], rev.ptr_to([txl.cast(token, "int64")]))

            with txl.if_(is_loader), txl.then_():
                for stage in range(D):
                    txl.ptx.mbarrier.init.shared.b64(s_mbar.ptr_to([ring_slot(stage)]), txl.uint32(1))
                txl.ptx.fence.mbarrier_init.release.cluster()
            with txl.if_(warp < P), txl.then_():
                for stage in range(D - 1):
                    txl.assign(cur_task[stage], (cta * P + warp) * (D - 1) + stage)
                if BULK_EARLY:
                    for stage in range(D - 1):
                        issue_load(stage)
        sync_block()

        # phase 1: per-CTA histogram, rank within (CTA, expert), claim CTA bases
        reg_expert = [txl.local_scalar("int32", init=txl.cast(reg_expert64[k], "int32")) for k in range(REG_TOKENS)]
        reg_rank = []
        for k in range(REG_TOKENS):
            rank = txl.local_scalar("int32", init=txl.int32(0))
            with txl.if_(token_start + tidx + k * THREADS < token_end), txl.then_():
                txl.ptx.atom.shared.add.s32(rank, s_count.ptr_to([reg_expert[k]]), txl.int32(1))
            reg_rank.append(rank)
        with txl.serial(token_start + tidx + REG_TOKENS * THREADS, token_end, step=THREADS) as token:
            expert64 = txl.local_scalar("int64")
            txl.ptx["ld.global.nc.s64"](expert64, eidx.ptr_to([txl.cast(token, "int64")]))
            expert = txl.cast(expert64, "int32")
            rank = txl.local_scalar("int32")
            txl.ptx.atom.shared.add.s32(rank, s_count.ptr_to([expert]), txl.int32(1))
            packed = txl.bitwise_or(txl.shift_left(expert64, txl.int64(32)), txl.cast(rank, "int64"))
            txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(token, "int64")]), packed)
        sync_block()
        with txl.serial(tidx, G, step=THREADS) as group_idx:
            count = load_shared(s_count, group_idx)
            with txl.if_(count > 0), txl.then_():
                base = txl.local_scalar("int32")
                txl.ptx.atom.global_.add.s32(base, state.ptr_to([txl.cast(group_idx, "int64")]), count)
                store_shared(s_total, group_idx, base)
        sync_block()
        for k in range(REG_TOKENS):
            token = token_start + tidx + k * THREADS
            with txl.if_(token < token_end), txl.then_():
                packed = txl.bitwise_or(txl.shift_left(txl.cast(reg_expert[k], "int64"), txl.int64(32)), txl.cast(reg_rank[k] + load_shared(s_total, reg_expert[k]), "int64"))
                txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(token, "int64")]), packed)
        # same thread -> same tokens as the first pass, so its own rev stores are visible
        with txl.serial(token_start + tidx + REG_TOKENS * THREADS, token_end, step=THREADS) as token:
            packed = txl.local_scalar("int64")
            txl.ptx["ld.global.s64"](packed, rev.ptr_to([txl.cast(token, "int64")]))
            expert = txl.cast(txl.shift_right(packed, txl.int64(32)), "int32")
            txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(token, "int64")]), packed + txl.cast(load_shared(s_total, expert), "int64"))
        stamp()
        txl.cuda.grid_sync()
        stamp()

        # phase 3: expert totals, block scan of padded sizes and pad rows
        padded_sizes = []
        pad_rows = []
        for j in range(GROUPS_PER_THREAD):
            group_idx = tidx * GROUPS_PER_THREAD + j
            count = txl.local_scalar("int32", init=txl.int32(0))
            with txl.if_(group_idx < G), txl.then_():
                txl.ptx["ld.global.cg.s32"](count, state.ptr_to([txl.cast(group_idx, "int64")]))
            padded = txl.local_scalar("int32", init=(count + align - 1) // align * align)
            padded_sizes.append(padded)
            pad_rows.append(padded - count)
        sync_block()  # s_total held the CTA bases until here
        for j in range(GROUPS_PER_THREAD):
            store_shared(s_total, tidx * GROUPS_PER_THREAD + j, padded_sizes[j] - pad_rows[j])
        sum_padded, sum_pad = padded_sizes[0], pad_rows[0]
        for j in range(1, GROUPS_PER_THREAD):
            sum_padded = sum_padded + padded_sizes[j]
            sum_pad = sum_pad + pad_rows[j]
        thread_padded = txl.local_scalar("int32", init=sum_padded)
        thread_pad = txl.local_scalar("int32", init=sum_pad)
        incl_padded = warp_inclusive_scan(thread_padded, lane)
        incl_pad = warp_inclusive_scan(thread_pad, lane)
        with txl.if_(lane == 31), txl.then_():
            store_shared(s_warp_padded, warp, incl_padded)
            store_shared(s_warp_pad, warp, incl_pad)
        sync_block()
        group_start = txl.local_scalar("int32", init=incl_padded - thread_padded)
        pad_start = txl.local_scalar("int32", init=incl_pad - thread_pad)
        M = txl.local_scalar("int32", init=txl.int32(0))
        num_pad_rows = txl.local_scalar("int32", init=txl.int32(0))
        for other_warp in range(WARPS):
            warp_padded = load_shared(s_warp_padded, other_warp)
            warp_pad = load_shared(s_warp_pad, other_warp)
            txl.assign(group_start, group_start + txl.Select(warp > other_warp, warp_padded, txl.int32(0)))
            txl.assign(pad_start, pad_start + txl.Select(warp > other_warp, warp_pad, txl.int32(0)))
            txl.assign(M, M + warp_padded)
            txl.assign(num_pad_rows, num_pad_rows + warp_pad)
        for j in range(GROUPS_PER_THREAD):
            group_idx = tidx * GROUPS_PER_THREAD + j
            store_shared(s_start, group_idx, group_start)
            store_shared(s_pad_start, group_idx, pad_start)
            with txl.if_(txl.And(cta == 0, group_idx < G)), txl.then_():
                txl.ptx["st.global.s32"](offs.ptr_to([txl.cast(group_idx, "int64")]), group_start + padded_sizes[j])
                txl.ptx["st.global.s32"](ks_t.ptr_to([txl.cast(group_idx, "int64")]), padded_sizes[j])
            txl.assign(group_start, group_start + padded_sizes[j])
            txl.assign(pad_start, pad_start + pad_rows[j])
        sync_block()
        stamp()

        alloc_rows = (M + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT
        num_zero_rows = txl.local_scalar("int32", init=num_pad_rows + (alloc_rows - M))
        num_tasks = txl.local_scalar("int32", init=num_copy_tasks + (num_zero_rows + R - 1) // R)

        def zero_row(zero_idx):
            return zero_row_dest(zero_idx, G, M, num_pad_rows, s_start, s_total, s_pad_start)

        def dest_row(packed):
            expert = txl.cast(txl.shift_right(packed, txl.int64(32)), "int32")
            rank = txl.cast(txl.bitwise_and(packed, txl.int64(0xFFFFFFFF)), "int32")
            return txl.local_scalar("int64", init=txl.cast(load_shared(s_start, expert) + rank, "int64"))

        if use_bulk:
            # phase 4 (bulk): D-deep ring of row slots per lane, tasks from the queue
            def store_task(stage):
                task = cur_task[stage]
                with txl.if_(is_pipe), txl.then_():
                    with txl.if_(task < num_copy_tasks):
                        with txl.then_():
                            token = task * R + lane
                            with txl.if_(token < m), txl.then_():
                                dest = dest_row(packed_rev[stage])
                                txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(token, "int64")]), dest)
                                txl.cuda.mbarrier_wait(s_mbar.ptr_to([ring_slot(stage)]), mbar_phase[stage])
                                txl.assign(mbar_phase[stage], mbar_phase[stage] ^ txl.uint32(1))
                                txl.ptx[S2G](out.ptr_to([dest * txl.int64(row_bytes)]), s_ring.ptr_to([(ring_slot(stage) * R + lane) * row_bytes]), txl.uint32(row_bytes))
                        with txl.else_():
                            zero_idx = (task - num_copy_tasks) * R + lane
                            with txl.if_(zero_idx < num_zero_rows), txl.then_():
                                zero_dest = txl.local_scalar("int64", init=txl.cast(zero_row(zero_idx), "int64") * txl.int64(row_bytes))
                                for byte_start in range(0, row_bytes, zero_bytes):
                                    txl.ptx[S2G](out.ptr_to([zero_dest + txl.int64(byte_start)]), s_zero.ptr_to([0]), txl.uint32(min(zero_bytes, row_bytes - byte_start)))
                    txl.ptx.cp.async_.bulk.commit_group()

            with txl.if_(warp < P), txl.then_():
                if not BULK_EARLY:
                    for stage in range(D - 1):
                        issue_load(stage)
                for stage in range(D - 1):
                    issue_rev(stage)
                with txl.while_(cur_task[0] < num_tasks):
                    for stage in range(D):
                        next_task = grab_task()
                        store_task(stage)
                        refill_stage = (stage + D - 1) % D
                        txl.assign(cur_task[refill_stage], broadcast_lane0(next_task))
                        with txl.if_(is_pipe), txl.then_():
                            txl.ptx.cp.async_.bulk.wait_group.read(1)
                        issue_load(refill_stage)
                        issue_rev(refill_stage)
                with txl.if_(is_pipe), txl.then_():
                    txl.ptx.cp.async_.bulk.wait_group(0)
        else:
            # phase 4 (registers): warp per row, loads before stores
            regs = txl.alloc_local((CHUNK_BATCH * num_regs,), reg_type)
            zero_regs = txl.alloc_local((num_regs,), reg_type)
            for reg_idx in range(num_regs):
                txl.assign(zero_regs[reg_idx], txl.Cast(reg_type, 0))

            def copy_row(src_base, dst_base):
                for batch_start in range(0, lane_iters, CHUNK_BATCH):
                    batch_len = min(CHUNK_BATCH, lane_iters - batch_start)
                    for j in range(batch_len):
                        chunk = batch_start + j
                        offset = txl.cast((chunk * 32 + lane) * vec, "int64")
                        chunk_regs = [regs[j * num_regs + reg_idx] for reg_idx in range(num_regs)]
                        if has_no_tail or (chunk + 1) * 32 <= num_chunks:
                            load_vec(vec, chunk_regs, x.ptr_to([src_base + offset]), LOAD_POLICY)
                        else:
                            with txl.if_(chunk * 32 + lane < num_chunks), txl.then_():
                                load_vec(vec, chunk_regs, x.ptr_to([src_base + offset]), LOAD_POLICY)
                    for j in range(batch_len):
                        chunk = batch_start + j
                        offset = txl.cast((chunk * 32 + lane) * vec, "int64")
                        chunk_regs = [regs[j * num_regs + reg_idx] for reg_idx in range(num_regs)]
                        if has_no_tail or (chunk + 1) * 32 <= num_chunks:
                            store_vec(vec, chunk_regs, out.ptr_to([dst_base + offset]), STORE_POLICY)
                        else:
                            with txl.if_(chunk * 32 + lane < num_chunks), txl.then_():
                                store_vec(vec, chunk_regs, out.ptr_to([dst_base + offset]), STORE_POLICY)

            rows_per_batch = max(1, min(TASK_ROWS, CHUNK_BATCH // lane_iters))

            def copy_rows(src_bases, dst_bases, row_oks):
                """
                Load every row of the batch, then store them: rows_per_batch rows of loads in flight
                per warp.
                """
                num_rows = len(src_bases)
                for i in range(num_rows):
                    with txl.if_(row_oks[i]), txl.then_():
                        for chunk in range(lane_iters):
                            offset = txl.cast((chunk * 32 + lane) * vec, "int64")
                            chunk_regs = [regs[(i * lane_iters + chunk) * num_regs + reg_idx] for reg_idx in range(num_regs)]
                            if has_no_tail or (chunk + 1) * 32 <= num_chunks:
                                load_vec(vec, chunk_regs, x.ptr_to([src_bases[i] + offset]), LOAD_POLICY)
                            else:
                                with txl.if_(chunk * 32 + lane < num_chunks), txl.then_():
                                    load_vec(vec, chunk_regs, x.ptr_to([src_bases[i] + offset]), LOAD_POLICY)
                for i in range(num_rows):
                    with txl.if_(row_oks[i]), txl.then_():
                        for chunk in range(lane_iters):
                            offset = txl.cast((chunk * 32 + lane) * vec, "int64")
                            chunk_regs = [regs[(i * lane_iters + chunk) * num_regs + reg_idx] for reg_idx in range(num_regs)]
                            if has_no_tail or (chunk + 1) * 32 <= num_chunks:
                                store_vec(vec, chunk_regs, out.ptr_to([dst_bases[i] + offset]), STORE_POLICY)
                            else:
                                with txl.if_(chunk * 32 + lane < num_chunks), txl.then_():
                                    store_vec(vec, chunk_regs, out.ptr_to([dst_bases[i] + offset]), STORE_POLICY)

            def store_zero_row(dst_base):
                for chunk in range(lane_iters):
                    offset = txl.cast((chunk * 32 + lane) * vec, "int64")
                    if has_no_tail or (chunk + 1) * 32 <= num_chunks:
                        store_vec(vec, zero_regs, out.ptr_to([dst_base + offset]), STORE_POLICY)
                    else:
                        with txl.if_(chunk * 32 + lane < num_chunks), txl.then_():
                            store_vec(vec, zero_regs, out.ptr_to([dst_base + offset]), STORE_POLICY)

            num_reg_tasks = txl.local_scalar("int32", init=num_copy_tasks + (num_zero_rows + ZERO_ROWS - 1) // ZERO_ROWS)
            task = txl.local_scalar("int32", init=broadcast_lane0(grab_task()))
            with txl.while_(task < num_reg_tasks):
                next_task = grab_task()
                with txl.if_(task < num_copy_tasks):
                    with txl.then_():
                        rows = [txl.local_scalar("int32", init=task * TASK_ROWS + j) for j in range(TASK_ROWS)]
                        dests = []
                        for j in range(TASK_ROWS):
                            packed = txl.local_scalar("int64")
                            txl.ptx["ld.global.cg.s64"](packed, rev.ptr_to([txl.cast(txl.min(rows[j], m - 1), "int64")]))
                            dests.append(packed)
                        dests = [dest_row(packed) for packed in dests]
                        # every lane read rev[rows] before lane 0 overwrites it
                        txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
                        for j in range(TASK_ROWS):
                            with txl.if_(txl.And(rows[j] < m, lane == 0)), txl.then_():
                                txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(rows[j], "int64")]), dests[j])
                        if rows_per_batch == 1:
                            for j in range(TASK_ROWS):
                                with txl.if_(rows[j] < m), txl.then_():
                                    copy_row(txl.cast(rows[j], "int64") * src_stride, dests[j] * txl.int64(row_bytes))
                        else:
                            for batch_start in range(0, TASK_ROWS, rows_per_batch):
                                batch_rows = list(range(batch_start, min(TASK_ROWS, batch_start + rows_per_batch)))
                                copy_rows([txl.cast(rows[j], "int64") * src_stride for j in batch_rows], [dests[j] * txl.int64(row_bytes) for j in batch_rows], [rows[j] < m for j in batch_rows])
                    with txl.else_():
                        zero_task = task - num_copy_tasks
                        for j in range(ZERO_ROWS):
                            zero_idx = txl.local_scalar("int32", init=zero_task * ZERO_ROWS + j)
                            with txl.if_(zero_idx < num_zero_rows), txl.then_():
                                store_zero_row(txl.cast(zero_row(zero_idx), "int64") * txl.int64(row_bytes))
                txl.assign(task, broadcast_lane0(next_task))
        sync_block()
        stamp()

        # phase 5: the last CTA out resets the persistent counters
        with txl.if_(tidx == 0), txl.then_():
            num_done = txl.local_scalar("int32")
            txl.ptx.atom.acq_rel.gpu.global_.add.s32(num_done, state.ptr_to([DONE_COUNTER]), txl.int32(1))
            store_shared(s_warp_padded, 0, num_done)
        sync_block()
        with txl.if_(load_shared(s_warp_padded, 0) == num_ctas - 1), txl.then_():
            with txl.serial(tidx, G, step=THREADS) as group_idx:
                txl.ptx["st.global.s32"](state.ptr_to([txl.cast(group_idx, "int64")]), txl.int32(0))
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx["st.global.s32"](state.ptr_to([QUEUE_HEAD]), txl.int32(0))
                txl.ptx["st.global.s32"](state.ptr_to([DONE_COUNTER]), txl.int32(0))

    return entry_function(token_scatter.compile(TARGET))


HOST_PARAMS = 128  # fast host-count path: 256 experts packed two per int64 parameter
HOST_MAX_GROUPS = 2 * HOST_PARAMS
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
    zero_bytes = min(row_bytes, ZERO_BYTES)
    QUEUE_HEAD = MAX_GROUPS
    HOST_GROUPS_PER_THREAD = HOST_MAX_GROUPS // THREADS  # groups per thread in the block scan
    store_warps = list(range(P, WARPS)) if P < WARPS else list(range(WARPS))

    def scatter_host_body(x, eidx, out, rev, offs, ks_t, state, state_nx, m, G, align, num_ctas, *host_counts):
        cta = txl.cta_id()
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5

        pool = txl.smem_pool()
        s_host_counts = pool.alloc((HOST_PARAMS,), "int64")
        s_total = pool.alloc((HOST_MAX_GROUPS,), "int32")
        s_start = pool.alloc((HOST_MAX_GROUPS,), "int32")
        s_pad_start = pool.alloc((HOST_MAX_GROUPS,), "int32")
        s_warp_padded = pool.alloc((WARPS,), "int32")
        s_warp_pad = pool.alloc((WARPS,), "int32")
        s_mbar = pool.alloc((P * D,), "uint64", align=8)
        s_zero = pool.alloc((zero_bytes,), "uint8", align=128)
        s_ring = pool.alloc((P * D * R * row_bytes,), "uint8", align=128)

        num_static = num_ctas * P * D  # every ring slot starts with a static task
        queue_head = txl.local_scalar("int32", init=txl.int32(0))  # raw queue head, consumed one refill later

        def grab_task_into(task):
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.atom.global_.add.s32(task, state.ptr_to([QUEUE_HEAD]), txl.int32(1))

        num_copy_tasks = txl.local_scalar("int32", init=(m + R - 1) // R)
        pipe_warp = txl.local_scalar("int32", init=warp)
        is_pipe = txl.And(warp < P, lane < R)
        is_loader = txl.And(warp < P, lane == 0)

        def ring_slot(stage):
            return pipe_warp * D + stage

        # Two-stream copy order: queue item q takes copy task q/2 from the first half of the
        # input (q even) or from the second half (q odd). ceil-half makes this a bijection on
        # [0, num_copy_tasks) for any num_copy_tasks; zero tasks (q >= num_copy_tasks) keep their
        # order.
        second_half_start = txl.local_scalar("int32", init=(num_copy_tasks + 1) >> 1)

        def interleave_task(raw):
            return txl.Select(raw < num_copy_tasks, txl.Select((raw & 1) == 1, second_half_start + (raw >> 1), raw >> 1), raw)

        cur_task = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
        mbar_phase = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(D)]
        # Expert IDs are contractually nonnegative and below HOST_MAX_GROUPS, so the
        # low word of their int64 storage is the complete value on this path.
        stage_expert = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
        stage_rank = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
        stage_leader_base = [txl.local_scalar("int32", init=txl.int32(0)) for _ in range(D)]
        stage_leader = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(D)]
        stage_before = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(D)]

        def is_copy_lane(stage):
            token = cur_task[stage] * R + lane
            return txl.And(is_pipe, txl.And(cur_task[stage] < num_copy_tasks, token < m))

        def issue_load(stage):
            token = cur_task[stage] * R
            with txl.if_(txl.And(is_loader, txl.And(cur_task[stage] < num_copy_tasks, token < m))), txl.then_():
                rows = txl.min(txl.int32(R), m - token)
                copy_bytes = rows * row_bytes
                mbar = s_mbar.ptr_to([ring_slot(stage)])
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar, txl.cast(copy_bytes, "uint32"))
                txl.ptx[G2S](s_ring.ptr_to([ring_slot(stage) * R * row_bytes]), x.ptr_to([txl.cast(token, "int64") * row_bytes]), txl.cast(copy_bytes, "uint32"), mbar, txl.uint64(L2_POLICY[LOAD_POLICY]))

        def issue_expert_load(stage):
            with txl.if_(is_copy_lane(stage)), txl.then_():
                txl.ptx["ld.global.nc.u32"](stage_expert[stage], eidx.ptr_to([txl.cast(cur_task[stage] * R + lane, "int64")]))

        def claim_ranks(stage):
            # Adjacent dispatched rows are commonly expert-minor. Collapse equal
            # experts within the warp to one global atomic while retaining a
            # unique rank for every participating lane. Invalid lanes use -1,
            # which is outside the expert-id contract, but still participate in
            # the warp-wide match and shuffle instructions.
            valid = is_copy_lane(stage)
            key = txl.local_scalar("int32", init=txl.Select(valid, stage_expert[stage], txl.int32(-1)))
            peers = txl.local_scalar("uint32")
            txl.ptx.match.any.sync.b32(peers, key, txl.uint32(0xFFFFFFFF))
            master = txl.local_scalar("uint32")
            txl.ptx.bfind.u32(master, peers)
            count = txl.local_scalar("uint32")
            txl.ptx.popc.b32(count, peers)
            lane_mask = txl.shift_left(txl.uint32(1), txl.cast(lane, "uint32")) - txl.uint32(1)
            txl.ptx.popc.b32(stage_before[stage], txl.bitwise_and(peers, lane_mask))
            txl.assign(stage_leader[stage], master)
            with txl.if_(txl.And(valid, lane == txl.cast(master, "int32"))), txl.then_():
                txl.ptx.atom.global_.add.s32(stage_leader_base[stage], state.ptr_to([key]), txl.cast(count, "int32"))

        def claim_static_pair_ranks():
            # The two static slots are adjacent tasks. Present both slots as one keyed
            # 2R-lane batch to match.any, halving rank atomics at startup and allowing an
            # expert run that crosses the slot boundary to share a single claim.
            in_slot0 = lane < R
            in_slot1 = txl.And(lane >= R, lane < 2 * R)
            token0 = cur_task[0] * R + lane
            token1 = cur_task[1] * R + (lane - R)
            valid0 = txl.And(in_slot0, txl.And(cur_task[0] < num_copy_tasks, token0 < m))
            valid1 = txl.And(in_slot1, txl.And(cur_task[1] < num_copy_tasks, token1 < m))
            expert1 = txl.local_scalar("int32")
            src_lane1 = txl.Select(lane < R, lane, lane - R)
            txl.ptx.shfl_sync.idx.b32(expert1, stage_expert[1], txl.cast(src_lane1, "uint32"), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            key = txl.local_scalar("int32", init=txl.Select(valid0, stage_expert[0], txl.Select(valid1, expert1, txl.int32(-1))))
            peers = txl.local_scalar("uint32")
            txl.ptx.match.any.sync.b32(peers, key, txl.uint32(0xFFFFFFFF))
            master = txl.local_scalar("uint32")
            txl.ptx.bfind.u32(master, peers)
            count = txl.local_scalar("uint32")
            txl.ptx.popc.b32(count, peers)
            lane_mask = txl.shift_left(txl.uint32(1), txl.cast(lane, "uint32")) - txl.uint32(1)
            before = txl.local_scalar("uint32")
            txl.ptx.popc.b32(before, txl.bitwise_and(peers, lane_mask))
            pair_base = txl.local_scalar("int32", init=txl.int32(0))
            with txl.if_(txl.And(txl.Or(valid0, valid1), lane == txl.cast(master, "int32"))), txl.then_():
                txl.ptx.atom.global_.add.s32(pair_base, state.ptr_to([key]), txl.cast(count, "int32"))
            # finish_rank consumes these per-slot views later, after the count scan.
            txl.assign(stage_leader_base[0], pair_base)
            txl.assign(stage_leader_base[1], pair_base)
            txl.assign(stage_leader[0], master)
            txl.assign(stage_before[0], before)
            master1 = txl.local_scalar("uint32")
            before1 = txl.local_scalar("uint32")
            src_lane_hi = txl.Select(lane < R, lane + R, lane)
            txl.ptx.shfl_sync.idx.b32(master1, master, txl.cast(src_lane_hi, "uint32"), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.ptx.shfl_sync.idx.b32(before1, before, txl.cast(src_lane_hi, "uint32"), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(stage_leader[1], master1)
            txl.assign(stage_before[1], before1)

        def finish_rank(stage):
            # warp-converged: the leader's atomic return is consumed only here, just before the store
            base = txl.local_scalar("int32")
            txl.ptx.shfl_sync.idx.b32(base, stage_leader_base[stage], stage_leader[stage], txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(stage_rank[stage], txl.Select(is_copy_lane(stage), base + txl.cast(stage_before[stage], "int32"), txl.int32(0)))

        # phase 0: first loads and expert ids, then the host counts into shared memory
        with txl.if_(is_loader), txl.then_():
            for stage in range(D):
                txl.ptx.mbarrier.init.shared.b64(s_mbar.ptr_to([ring_slot(stage)]), txl.uint32(1))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.if_(warp < P), txl.then_():
            for stage in range(D):
                txl.assign(cur_task[stage], interleave_task((cta * P + warp) * D + stage))
            for stage in range(D):
                issue_expert_load(stage)
            grab_task_into(queue_head)
            for stage in range(D):
                issue_load(stage)
        with txl.serial(tidx, zero_bytes // 16, step=THREADS) as vec_idx:
            txl.ptx["st.shared.v4.b32"](s_zero.ptr_to([vec_idx * 16]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0))
        txl.ptx.fence.proxy.async_.shared__cta()
        # The next call uses the other counter bank: clear it now (the previous call that used it has
        # completed in stream order, and nothing in this call touches it), so no CTA resets at exit.
        with txl.if_(cta == num_ctas - 1), txl.then_():
            with txl.serial(tidx, HOST_MAX_GROUPS, step=THREADS) as group_idx:
                txl.ptx["st.global.s32"](state_nx.ptr_to([txl.cast(group_idx, "int64")]), txl.int32(0))
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx["st.global.s32"](state_nx.ptr_to([QUEUE_HEAD]), txl.int32(0))
        for i, store_warp in enumerate(store_warps):
            with txl.if_(txl.And(warp == store_warp, lane == 0)), txl.then_():
                for param_idx in range(i, HOST_PARAMS, len(store_warps)):
                    txl.ptx["st.shared.s64"](s_host_counts.ptr_to([param_idx]), host_counts[param_idx])
        with txl.if_(warp < P), txl.then_():
            claim_static_pair_ranks()  # rank claims of both static slots overlap the scan
        sync_block()

        # phase 1: block scan of padded sizes and pad rows
        padded_sizes = []
        pad_rows = []
        for j in range(HOST_GROUPS_PER_THREAD):
            group_idx = tidx * HOST_GROUPS_PER_THREAD + j
            count_pair = txl.local_scalar("int64")
            txl.ptx["ld.shared.s64"](count_pair, s_host_counts.ptr_to([group_idx >> 1]))
            half = txl.Select((group_idx & 1) == 1, txl.shift_right(count_pair, txl.int64(32)), count_pair)
            count = txl.local_scalar("int32", init=txl.Select(group_idx < G, txl.cast(txl.bitwise_and(half, txl.int64(0xFFFFFFFF)), "int32"), txl.int32(0)))
            padded = txl.local_scalar("int32", init=(count + align - 1) // align * align)
            padded_sizes.append(padded)
            pad_rows.append(padded - count)
            store_shared(s_total, group_idx, count)
        sum_padded, sum_pad = padded_sizes[0], pad_rows[0]
        for j in range(1, HOST_GROUPS_PER_THREAD):
            sum_padded = sum_padded + padded_sizes[j]
            sum_pad = sum_pad + pad_rows[j]
        thread_padded = txl.local_scalar("int32", init=sum_padded)
        thread_pad = txl.local_scalar("int32", init=sum_pad)
        incl_padded = warp_inclusive_scan(thread_padded, lane)
        incl_pad = warp_inclusive_scan(thread_pad, lane)
        with txl.if_(lane == 31), txl.then_():
            store_shared(s_warp_padded, warp, incl_padded)
            store_shared(s_warp_pad, warp, incl_pad)
        sync_block()
        group_start = txl.local_scalar("int32", init=incl_padded - thread_padded)
        pad_start = txl.local_scalar("int32", init=incl_pad - thread_pad)
        M = txl.local_scalar("int32", init=txl.int32(0))
        num_pad_rows = txl.local_scalar("int32", init=txl.int32(0))
        for other_warp in range(WARPS):
            warp_padded = load_shared(s_warp_padded, other_warp)
            warp_pad = load_shared(s_warp_pad, other_warp)
            txl.assign(group_start, group_start + txl.Select(warp > other_warp, warp_padded, txl.int32(0)))
            txl.assign(pad_start, pad_start + txl.Select(warp > other_warp, warp_pad, txl.int32(0)))
            txl.assign(M, M + warp_padded)
            txl.assign(num_pad_rows, num_pad_rows + warp_pad)
        for j in range(HOST_GROUPS_PER_THREAD):
            group_idx = tidx * HOST_GROUPS_PER_THREAD + j
            store_shared(s_start, group_idx, group_start)
            store_shared(s_pad_start, group_idx, pad_start)
            with txl.if_(txl.And(cta == 0, group_idx < G)), txl.then_():
                txl.ptx["st.global.s32"](offs.ptr_to([txl.cast(group_idx, "int64")]), group_start + padded_sizes[j])
                txl.ptx["st.global.s32"](ks_t.ptr_to([txl.cast(group_idx, "int64")]), padded_sizes[j])
            txl.assign(group_start, group_start + padded_sizes[j])
            txl.assign(pad_start, pad_start + pad_rows[j])
        sync_block()

        alloc_rows = (M + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT
        num_zero_rows = txl.local_scalar("int32", init=num_pad_rows + (alloc_rows - M))
        num_tasks = txl.local_scalar("int32", init=num_copy_tasks + (num_zero_rows + R - 1) // R)

        def zero_row(zero_idx):
            return zero_row_dest(zero_idx, G, M, num_pad_rows, s_start, s_total, s_pad_start)

        # phase 2: D-deep ring of row slots per lane, tasks from the queue
        def store_task(stage):
            task = cur_task[stage]
            with txl.if_(is_pipe), txl.then_():
                with txl.if_(task < num_copy_tasks):
                    with txl.then_():
                        token = task * R + lane
                        with txl.if_(token < m), txl.then_():
                            expert = stage_expert[stage]
                            dest = txl.local_scalar("int64", init=txl.cast(load_shared(s_start, expert) + stage_rank[stage], "int64"))
                            txl.ptx["st.global.s64"](rev.ptr_to([txl.cast(token, "int64")]), dest)
                            txl.cuda.mbarrier_wait(s_mbar.ptr_to([ring_slot(stage)]), mbar_phase[stage])
                            txl.assign(mbar_phase[stage], mbar_phase[stage] ^ txl.uint32(1))
                            txl.ptx[S2G](out.ptr_to([dest * txl.int64(row_bytes)]), s_ring.ptr_to([(ring_slot(stage) * R + lane) * row_bytes]), txl.uint32(row_bytes))
                            # Copy groups are committed individually because the
                            # following wait.read protects reuse of this ring slot.
                            txl.ptx.cp.async_.bulk.commit_group()
                    with txl.else_():
                        zero_idx = (task - num_copy_tasks) * R + lane
                        with txl.if_(zero_idx < num_zero_rows), txl.then_():
                            zero_dest = txl.local_scalar("int64", init=txl.cast(zero_row(zero_idx), "int64") * txl.int64(row_bytes))
                            for byte_start in range(0, row_bytes, zero_bytes):
                                txl.ptx[S2G](out.ptr_to([zero_dest + txl.int64(byte_start)]), s_zero.ptr_to([0]), txl.uint32(min(zero_bytes, row_bytes - byte_start)))
                        # s_zero is immutable, so all terminal zero stores can remain
                        # uncommitted and share one group drained at warp exit.

        with txl.if_(warp < P), txl.then_():
            is_first = txl.local_scalar("int32", init=txl.int32(1))
            with txl.while_(cur_task[0] < num_tasks):
                for stage in range(D):
                    with txl.if_(cur_task[stage] < num_copy_tasks), txl.then_():
                        finish_rank(stage)
                    store_task(stage)
                    refill_stage = (stage + D - 1) % D  # the slot processed one step ago
                    with txl.if_(is_first == 0), txl.then_():
                        txl.assign(cur_task[refill_stage], interleave_task(broadcast_lane0(queue_head) + num_static))
                        grab_task_into(queue_head)
                        # Zero refills neither reuse the ring nor need expert/rank
                        # metadata. Avoid the otherwise unconditional match.any,
                        # bfind, popc, and rank shuffle on the terminal suffix.
                        with txl.if_(cur_task[refill_stage] < num_copy_tasks), txl.then_():
                            with txl.if_(is_pipe), txl.then_():
                                txl.ptx.cp.async_.bulk.wait_group.read(1)
                            issue_load(refill_stage)
                            issue_expert_load(refill_stage)
                            claim_ranks(refill_stage)
                    txl.assign(is_first, txl.int32(0))
            with txl.if_(is_pipe), txl.then_():
                txl.ptx.cp.async_.bulk.commit_group()
                txl.ptx.cp.async_.bulk.wait_group(0)
        # No CTA barrier is needed at exit: every issuing lane has drained its
        # own bulk group, and no thread accesses shared state after this point.

    def token_scatter_host(*args):
        num_ctas = args[11]
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=WARPS * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        return scatter_host_body(*args)

    # the 128 count parameters are generated, so the kernel's signature is built here
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    params = [
        inspect.Parameter("x", kind, annotation=txl.Tensor((txl.int64(),), txl.u8, data_alignment=1)),
        inspect.Parameter("eidx", kind, annotation=txl.Tensor((txl.int64(),), txl.i64, data_alignment=1)),
        inspect.Parameter("out", kind, annotation=txl.Tensor((txl.int64(),), txl.u8, data_alignment=1)),
        inspect.Parameter("rev", kind, annotation=txl.Tensor((txl.int64(),), txl.i64, data_alignment=1)),
        inspect.Parameter("offs", kind, annotation=txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        inspect.Parameter("ks_t", kind, annotation=txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        inspect.Parameter("state", kind, annotation=txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        inspect.Parameter("state_nx", kind, annotation=txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        inspect.Parameter("m", kind, annotation=txl.i32),
        inspect.Parameter("G", kind, annotation=txl.i32),
        inspect.Parameter("align", kind, annotation=txl.i32),
        inspect.Parameter("num_ctas", kind, annotation=txl.i32),
    ] + [inspect.Parameter(f"hc{param_idx}", kind, annotation=txl.i64) for param_idx in range(HOST_PARAMS)]
    token_scatter_host.__signature__ = inspect.Signature(params)
    kernel = txl.kernel()(token_scatter_host)
    return entry_function(kernel.compile(TARGET))


@functools.cache
def num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


INSTR = False  # record per-phase timestamps in DBG (the cooperative kernel only)
DBG = None


def scatter_by_expert_torch(input_tokens: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, padding_alignment: int, group_counts) -> tuple:
    """
    scatter_by_expert in PyTorch, for deterministic placement or more than MAX_GROUPS: same outputs,
    tokens in token order within each group.
    """
    num_tokens, hidden = input_tokens.shape
    device = input_tokens.device
    counts = torch.bincount(expert_idxs, minlength=num_groups)[:num_groups]
    ks_tensor = ((counts + padding_alignment - 1) // padding_alignment * padding_alignment).to(torch.int32)
    offs = ks_tensor.cumsum(0).to(torch.int32)
    if group_counts is not None:
        ks = [(n + padding_alignment - 1) // padding_alignment * padding_alignment for n in group_counts]
    else:
        ks = ks_tensor.tolist()
    order = torch.argsort(expert_idxs, stable=True)
    ordered = expert_idxs[order]
    first = torch.searchsorted(ordered, torch.arange(num_groups, device=device, dtype=ordered.dtype))
    rank = torch.empty_like(order)
    rank[order] = torch.arange(num_tokens, device=device) - first[ordered]
    rev = (offs - ks_tensor).long()[expert_idxs] + rank
    total = sum(ks)
    rows = (total + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT
    out = torch.zeros((max(rows, 1), hidden), device=device, dtype=input_tokens.dtype)
    out.index_copy_(0, rev, input_tokens)
    return out[:total], rev, offs, ks_tensor, ks


def scatter_by_expert_sm100(input_tokens: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, padding_alignment: int = 128, group_counts: Optional[list[int]] = None, *, deterministic: bool = False):
    """
    scatter_by_expert on SM100 (see the picker's contract).
    """
    global DBG
    num_tokens, hidden = input_tokens.shape[0], input_tokens.shape[1]
    device = input_tokens.device
    if expert_idxs.shape[0] != num_tokens:
        raise ValueError(f"{expert_idxs.shape[0]=} must equal the input row count {num_tokens=}")
    if num_tokens == 0:
        return (torch.empty((0, hidden), device=device, dtype=input_tokens.dtype), torch.empty((0,), device=device, dtype=torch.int64), torch.empty((0,), device=device, dtype=torch.int32), torch.empty((0,), device=device, dtype=torch.int32), [])
    if deterministic or num_groups > MAX_GROUPS:
        return scatter_by_expert_torch(input_tokens, expert_idxs, num_groups, padding_alignment, group_counts)
    if input_tokens.stride(1) != 1 and hidden > 1:
        raise ValueError("input_tokens must have unit inner stride")

    elem_bytes = input_tokens.element_size()
    row_bytes = hidden * elem_bytes
    src_stride = input_tokens.stride(0) * elem_bytes
    vec = math.gcd(MAX_VEC_BYTES, row_bytes, src_stride if num_tokens > 1 else row_bytes, input_tokens.data_ptr() % MAX_VEC_BYTES or MAX_VEC_BYTES)

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

    num_sm = num_sms(device.index if device.index is not None else torch.cuda.current_device())
    work_rows = num_tokens + max_rows
    num_ctas = max(1, min(num_sm, (work_rows + 63) // 64))
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
    if HOST and not INSTR and cfg is not None and group_counts is not None and num_groups <= HOST_MAX_GROUPS:
        counts = [int(n) for n in group_counts] + [0] * (HOST_MAX_GROUPS - num_groups)
        packed = []
        for param_idx in range(HOST_PARAMS):
            word = counts[2 * param_idx] | (counts[2 * param_idx + 1] << 32)
            packed.append(word if word < (1 << 63) else word - (1 << 64))
        host_state = get_host_state(device)
        banks, bank = host_state
        host_state[1] = bank ^ 1
        build_host(row_bytes, cfg)(x_bytes, expert_idxs, out.view(-1).view(torch.uint8), rev, offs, ks_tensor, banks[bank], banks[bank ^ 1], num_tokens, num_groups, padding_alignment, num_ctas, *packed)
        return out[: sum(ks)], rev, offs, ks_tensor, ks
    kernel = build(row_bytes, vec, cfg, INSTR)
    kernel(x_bytes, expert_idxs, out.view(-1).view(torch.uint8), rev, offs, ks_tensor, state, dbg, num_tokens, num_groups, padding_alignment, src_stride, tpc, num_ctas)
    if ks is None:
        ks_host = pinned_buffer(num_groups)
        ks_host.copy_(ks_tensor, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        ks = ks_host.tolist()
    return out[: sum(ks)], rev, offs, ks_tensor, ks
