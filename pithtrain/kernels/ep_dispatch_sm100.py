"""
TIRx-lite kernels of the ep_dispatch feature (SM100), one launch per side.

Sender side: one launch with no unconditional grid barrier. Each CTA counts its token chunk in
shared memory and claims its base in every expert / rank bin with one global atomic per bin. A CTA
spins on the others only after it has seen every CTA of the grid start, so the spin always ends even
when comm kernels on other streams hold SMs; otherwise it exits and leaves its placement to a shared
chunk queue (see send_def). Placement within a bucket uses shared-memory atomics, so the order
within a bucket is launch-dependent, as on the Triton path it replaces. The workspace (bin counters,
start / arrive / place / exit words and per-CTA bases) is int32, kept per (device, stream) and
re-zeroed by the last CTA out, so it is zeroed once on first use and never again from the host.

Receiver side: one launch. Every CTA scans the received metadata in shared memory, then
binary-searches the segment of each of its slots.

Inputs the kernels do not cover (more than SEND_MAX_BINS bins or RECV_MAX_SEGMENTS segments, more
than 32 ranks on the receiver side, 2^31 or more slots or dedup entries) take a PyTorch path with
the same outputs; topk_ids of another integer dtype or with a non-unit inner stride are converted to
contiguous int64 first. Neither side, nor the PyTorch path, synchronizes the device with the host.

Evolved with TIRx-harness: run ep_dispatch, frontier member single-launch-start-gated (commit
1be128ef, 788.6 us vs 2384.8 us, 3.024x over the previous SM100 path in the task's benchmark).
Compiled once per process by tvm.compile(..., tir_pipeline="tirx") and launched through tvm-ffi on
torch's current stream.
"""

import functools

import torch

import tirx_kernels.tirx_lite as txl

__all__ = ["ep_dispatch_recv_idx_sm100", "ep_dispatch_send_idx_sm100"]

FULL = 0xFFFFFFFF

EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


def bar():
    txl.ptx.bar.sync(txl.uint32(0))


def ld_s32(buf, i):
    out = txl.local_scalar("int32")
    txl.ptx.ld.shared.b32(out, buf.ptr_to([i]))
    return out


def st_s32(buf, i, v):
    txl.ptx.st.shared.b32(buf.ptr_to([i]), v)


def ld_g64(buf, i):
    out = txl.local_scalar("int64")
    txl.ptx.ld.global_.b64(out, buf.ptr_to([i]))
    return out


def ld_g32(buf, i):
    out = txl.local_scalar("int32")
    txl.ptx.ld.global_.b32(out, buf.ptr_to([i]))
    return out


def st_g64(buf, i, v):
    tmp = txl.local_scalar("int64", init=v)
    txl.ptx.st.global_.b64(buf.ptr_to([i]), tmp)


def shfl_up(v, d):
    out = txl.local_scalar("int32")
    txl.ptx.shfl_sync.up.b32(out, v, txl.uint32(d), txl.uint32(0), txl.uint32(FULL))
    return out


def warp_incl_scan(v, lane):
    acc = txl.local_scalar("int32", init=v)
    for s in (1, 2, 4, 8, 16):
        peer = shfl_up(acc, s)
        txl.assign(acc, txl.Select(lane >= s, acc + peer, acc))
    return acc


def block_excl_scan(src, dst, n, tid, lane, warp, s_warp, nthreads, add=None):
    """
    dst[i] = sum(src[:i]) (+ add[i] if given) over shared int32 arrays of trace-time length n.

    Each thread owns a contiguous run of ceil(n / nthreads) items; one warp scans the run totals.
    Ends with a CTA barrier. Returns nothing; dst may alias src only when add is None.
    """
    per = (n + nthreads - 1) // nthreads
    nwarps = nthreads // 32
    run = txl.local_scalar("int32", init=txl.int32(0))
    for q in range(per):
        i = tid * per + q
        with txl.If(i < n), txl.Then():
            txl.assign(run, run + ld_s32(src, i))
    inc = warp_incl_scan(run, lane)
    with txl.If(lane == 31), txl.Then():
        st_s32(s_warp, warp, inc)
    bar()
    with txl.If(warp == 0), txl.Then():
        w = txl.local_scalar("int32", init=txl.int32(0))
        with txl.If(lane < nwarps), txl.Then():
            txl.assign(w, ld_s32(s_warp, lane))
        wi = warp_incl_scan(w, lane)
        with txl.If(lane < nwarps), txl.Then():
            st_s32(s_warp, lane, wi - w)
    bar()
    acc = txl.local_scalar("int32", init=ld_s32(s_warp, warp) + inc - run)
    vals = []
    for q in range(per):
        i = tid * per + q
        v = txl.local_scalar("int32", init=txl.int32(0))
        with txl.If(i < n), txl.Then():
            txl.assign(v, ld_s32(src, i))
        vals.append(v)
    bar()
    for q in range(per):
        i = tid * per + q
        with txl.If(i < n), txl.Then():
            if add is None:
                st_s32(dst, i, acc)
            else:
                st_s32(dst, i, acc + ld_s32(add, i))
        txl.assign(acc, acc + vals[q])
    bar()


# --------------------------------------------------------------------------------------------
# recv
# --------------------------------------------------------------------------------------------
RECV_WARPS = 8
RECV_PER_THREAD = 1
RECV_MAX_SEGMENTS = 2048  # ep_size * experts_per_rank the kernel covers (two int32 smem arrays)


@functools.cache
def recv_kernel(ep, epr):
    return entry_function(recv_def(ep, epr).compile())


def recv_def(ep, epr):
    nseg = ep * epr
    nthreads = RECV_WARPS * 32
    tile = nthreads * RECV_PER_THREAD
    steps = max(1, nseg.bit_length())

    @txl.kernel(warps=RECV_WARPS, arch="sm_100a", grid="num_ctas")
    def ep_recv_idx(
        meta: txl.gptr[txl.i64],
        received: txl.gptr[txl.i64],
        expert_out: txl.gptr[txl.i64],
        expand_out: txl.gptr[txl.i64],
        recv_stride: txl.i64,
        total: txl.i32,
        num_ctas: txl.i32,
    ):
        tid = txl.thread_id()
        lane, warp = tid & 31, tid >> 5
        cta = txl.cta_id()
        pool = txl.smem_pool()
        s_cnt, s_end, s_dd, s_warp = pool.alloc([nseg], "int32"), pool.alloc([nseg], "int32"), pool.alloc([32], "int32"), pool.alloc([32], "int32")

        # issue this thread's received loads first: their latency overlaps the metadata scan
        rv = []
        for q in range(RECV_PER_THREAD):
            slot = cta * tile + q * nthreads + tid
            r = txl.local_scalar("int64", init=txl.int64(0))
            with txl.If(slot < total), txl.Then():
                txl.ptx.ld.global_.b64(r, received.ptr_to([txl.Cast("int64", slot) * recv_stride]))
            rv.append(r)

        def seg_count(i):
            s = i // epr
            return txl.Cast("int32", ld_g64(meta, txl.Cast("int64", s * (epr + 1) + (i - s * epr))))

        if nseg <= nthreads:
            cnt = txl.local_scalar("int32", init=txl.int32(0))
            with txl.If(tid < nseg), txl.Then():
                txl.assign(cnt, seg_count(tid))
        else:
            with txl.serial((nseg + nthreads - 1) // nthreads) as it:
                i = it * nthreads + tid
                with txl.If(i < nseg), txl.Then():
                    st_s32(s_cnt, i, seg_count(i))
        # sender dedup exclusive prefix (ep <= 32) in warp 0
        with txl.If(warp == 0), txl.Then():
            d = txl.local_scalar("int32", init=txl.int32(0))
            with txl.If(lane < ep), txl.Then():
                txl.assign(d, txl.Cast("int32", ld_g64(meta, txl.Cast("int64", lane * (epr + 1) + epr))))
            di = warp_incl_scan(d, lane)
            st_s32(s_dd, lane, di - d)
        if nseg <= nthreads:
            inc = warp_incl_scan(cnt, lane)
            with txl.If(lane == 31), txl.Then():
                st_s32(s_warp, warp, inc)
            bar()
            pre = txl.local_scalar("int32", init=txl.int32(0))
            for w in range(RECV_WARPS - 1):
                txl.assign(pre, pre + txl.Select(warp > w, ld_s32(s_warp, w), txl.int32(0)))
            with txl.If(tid < nseg), txl.Then():
                st_s32(s_end, tid, pre + inc)
            bar()
        else:
            bar()
            # inclusive end of every segment: exclusive scan plus own count
            block_excl_scan(s_cnt, s_end, nseg, tid, lane, warp, s_warp, nthreads, add=s_cnt)

        for q in range(RECV_PER_THREAD):
            slot = cta * tile + q * nthreads + tid
            with txl.If(slot < total), txl.Then():
                lo, hi = txl.local_scalar("int32", init=txl.int32(0)), txl.local_scalar("int32", init=txl.int32(nseg))
                for step in range(steps):
                    mid = txl.local_scalar("int32", init=(lo + hi) >> 1)
                    with txl.If(lo < hi), txl.Then():
                        above = ld_s32(s_end, txl.Min(mid, txl.int32(nseg - 1))) > slot
                        txl.assign(hi, txl.Select(above, mid, hi))
                        txl.assign(lo, txl.Select(above, lo, mid + 1))
                seg = txl.Min(lo, txl.int32(nseg - 1))
                sender = seg // epr
                expert = seg - sender * epr
                st_g64(expert_out, txl.Cast("int64", slot), txl.Cast("int64", expert))
                st_g64(expand_out, txl.Cast("int64", slot), rv[q] + txl.Cast("int64", ld_s32(s_dd, sender)))

    return ep_recv_idx


def recv_idx_torch(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The receiver side in PyTorch (no host sync), for inputs the kernel does not cover.
    """
    rows = recv_meta.reshape(ep_size, experts_per_rank + 1)
    seg_end = rows[:, :experts_per_rank].reshape(-1).cumsum(0)
    slot = torch.arange(received_expand_idx.shape[0], dtype=torch.int64, device=received_expand_idx.device)
    seg = torch.searchsorted(seg_end, slot, right=True).clamp_(max=ep_size * experts_per_rank - 1)
    dedup = rows[:, experts_per_rank]
    dedup_start = dedup.cumsum(0) - dedup
    return seg % experts_per_rank, received_expand_idx + dedup_start[seg // experts_per_rank]


def ep_dispatch_recv_idx_sm100(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    # the kernel reads recv_meta with unit stride; received_expand_idx with its own (non-negative)
    recv_meta = recv_meta.contiguous()
    if received_expand_idx.stride(0) < 0:
        received_expand_idx = received_expand_idx.contiguous()
    total = received_expand_idx.shape[0]
    dev = received_expand_idx.device
    if total == 0:
        return torch.empty(0, dtype=torch.int64, device=dev), torch.empty(0, dtype=torch.int64, device=dev)
    if not (ep_size <= 32 and 0 < ep_size * experts_per_rank <= RECV_MAX_SEGMENTS and total < 2**31 and received_expand_idx.dtype == torch.int64):
        return recv_idx_torch(recv_meta, received_expand_idx, ep_size, experts_per_rank)
    expert_idxs, expand_idx = torch.empty(total, dtype=torch.int64, device=dev), torch.empty(total, dtype=torch.int64, device=dev)
    fn = recv_kernel(ep_size, experts_per_rank)
    tile = RECV_WARPS * 32 * RECV_PER_THREAD
    stride = received_expand_idx.stride(0)
    flat = received_expand_idx.as_strided(((total - 1) * stride + 1,), (1,))
    fn(recv_meta, flat, expert_idxs, expand_idx, stride, total, (total + tile - 1) // tile)
    return expert_idxs, expand_idx


# --------------------------------------------------------------------------------------------
# send: one launch; a CTA waits for others only once every CTA of the grid has started
# --------------------------------------------------------------------------------------------
SEND_WARPS = 8
SEND_MIN_TOKENS_PER_CTA = 128
SEND_MAX_CTAS = 128
SEND_MAX_BINS = 2048  # ep_size * (experts_per_rank + 1) the kernel covers (shared memory < 48 KB)


def atom_s_add(buf, i, v):
    out = txl.local_scalar("uint32")
    txl.ptx.atom.shared.add.u32(out, buf.ptr_to([i]), txl.uint32(v) if isinstance(v, int) else v)
    return txl.reinterpret("int32", out)


@functools.cache
def zero_kernel(n):
    @txl.kernel(warps=4, arch="sm_100a", grid=1)
    def ep_ws_zero(ws: txl.gptr[txl.i32]):
        tid = txl.thread_id()
        with txl.serial((n + 127) // 128) as it:
            i = it * 128 + tid
            with txl.If(i < n), txl.Then():
                txl.ptx.st.global_.b32(ws.ptr_to([i]), txl.int32(0))

    return entry_function(ep_ws_zero.compile())


def words(ne, ep):
    """
    Workspace word indices: counters [0, ne + ep), then start / arrive / place / exit, then bases.
    """
    nb = ne + ep
    w = {"start": nb, "arrive": nb + 1, "place": nb + 2, "exit": nb + 3}
    w["cb_off"] = (nb + 4 + 3) // 4 * 4
    return w


@functools.cache
def send_kernel(k, ep, epr, ids_dtype):
    return entry_function(send_def(k, ep, epr, ids_dtype).compile())


def send_def(k, ep, epr, ids_dtype):
    """
    Deadlock-free single launch.

    Phase 1 (never waits): CTA c counts its token chunk c in shared memory, claims its base in every
    expert / rank with one global atomic per bin and stores the bases at cbase[c]; it then arrives.
    Every CTA bumped `start` on entry. A CTA spins on `arrive` only if `start` already equals the
    grid size: then every CTA is resident and will arrive without waiting on anything, so the spin
    ends. A CTA that sees an unstarted CTA exits instead. Placement is a shared queue of chunks
    claimed with `place`; the last CTA to start always sees start == grid size, waits and drains
    the queue, so every chunk is placed. In the common case all CTAs stay and each places about
    one chunk. The last CTA to exit re-zeroes the counters and words for the next launch.
    """
    ne = ep * epr
    nb = ne + ep
    nt = SEND_WARPS * 32
    w = words(ne, ep)
    cb_off = w["cb_off"]

    @txl.kernel(warps=SEND_WARPS, arch="sm_100a", grid="num_ctas")
    def ep_send_idx(
        topk: txl.gptr[ids_dtype],
        meta_out: txl.gptr[txl.i64],
        out: txl.gptr[txl.i64],
        ws: txl.gptr[txl.i32],
        m: txl.i32,
        row_stride: txl.i64,
        tpc: txl.i32,
        num_ctas: txl.i32,
    ):
        tid = txl.thread_id()
        lane, warp = tid & 31, tid >> 5
        cta = txl.cta_id()
        pool = txl.smem_pool()
        s_hist, s_ex, s_start, s_cur = pool.alloc([nb], "int32"), pool.alloc([ne], "int32"), pool.alloc([ne], "int32"), pool.alloc([ne], "int32")
        s_rh, s_rcur, s_rbase, s_rstart, s_warp, s_flag = pool.alloc([32], "int32"), pool.alloc([32], "int32"), pool.alloc([32], "int32"), pool.alloc([32], "int32"), pool.alloc([32], "int32"), pool.alloc([4], "int32")

        m64 = txl.Cast("int64", m)
        o_idx = m64 * ep
        o_exp = o_idx + m64 * k
        o_sm = o_exp + m64 * k

        def load_id(tok, j):
            off = txl.Cast("int64", tok) * row_stride + j
            if ids_dtype == "int64":
                return txl.Cast("int32", ld_g64(topk, off))
            return ld_g32(topk, off)

        with txl.If(tid == 0), txl.Then():
            txl.ptx.red.relaxed.gpu.global_.add.s32(ws.ptr_to([w["start"]]), txl.int32(1))
        with txl.serial((nb + nt - 1) // nt) as it:
            i = it * nt + tid
            with txl.If(i < nb), txl.Then():
                st_s32(s_hist, i, txl.int32(0))
        bar()

        # ---- phase 1: count chunk `cta`, claim its bases --------------------------------
        t0 = cta * tpc
        t1 = txl.Min(m, t0 + tpc)
        with txl.serial((tpc + nt - 1) // nt) as it:
            tok = t0 + it * nt + tid
            with txl.If(tok < t1), txl.Then():
                mask = txl.local_scalar("uint32", init=txl.uint32(0))
                ids = [txl.local_scalar("int32", init=load_id(tok, j)) for j in range(k)]
                for j in range(k):
                    e = ids[j]
                    atom_s_add(s_hist, e, 1)
                    r = e // epr
                    bit = txl.shift_left(txl.uint32(1), txl.Cast("uint32", r))
                    with txl.If(txl.bitwise_and(mask, bit) == txl.uint32(0)), txl.Then():
                        atom_s_add(s_hist, ne + r, 1)
                    txl.assign(mask, txl.bitwise_or(mask, bit))
        bar()
        with txl.serial((nb + nt - 1) // nt) as it:
            i = it * nt + tid
            with txl.If(i < nb), txl.Then():
                v = ld_s32(s_hist, i)
                base = txl.local_scalar("uint32", init=txl.uint32(0))
                with txl.If(v > 0), txl.Then():
                    txl.ptx.atom.global_.add.u32(base, ws.ptr_to([i]), txl.reinterpret("uint32", v))
                txl.ptx.st.global_.b32(ws.ptr_to([cb_off + txl.Cast("int64", cta) * nb + i]), txl.reinterpret("int32", base))
        bar()

        # ---- arrive; stay only if the whole grid has started -----------------------------
        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.acq_rel.gpu()
            txl.ptx.red.relaxed.gpu.global_.add.s32(ws.ptr_to([w["arrive"]]), txl.int32(1))
            started = txl.local_scalar("int32")
            txl.ptx.ld.acquire.gpu.global_.b32(started, ws.ptr_to([w["start"]]))
            stay = txl.Select(started >= num_ctas, txl.int32(1), txl.int32(0))
            st_s32(s_flag, 0, stay)
            with txl.If(stay == 1), txl.Then():
                spin = txl.local_scalar("int32")
                txl.cuda.wait_until(spin, ws.ptr_to([w["arrive"]]), spin >= num_ctas, scope="gpu", ptx_type="b32")
        bar()

        with txl.If(ld_s32(s_flag, 0) == 1), txl.Then():
            # totals, expert exclusive starts, rank starts
            with txl.serial((ne + nt - 1) // nt) as it:
                i = it * nt + tid
                with txl.If(i < ne), txl.Then():
                    tot = txl.local_scalar("int32")
                    txl.ptx.ld.relaxed.gpu.global_.b32(tot, ws.ptr_to([i]))
                    st_s32(s_hist, i, tot)
            with txl.If(tid < 32), txl.Then():
                tot = txl.local_scalar("int32", init=txl.int32(0))
                with txl.If(tid < ep), txl.Then():
                    txl.ptx.ld.relaxed.gpu.global_.b32(tot, ws.ptr_to([ne + tid]))
                st_s32(s_rh, tid, tot)
            bar()
            block_excl_scan(s_hist, s_ex, ne, tid, lane, warp, s_warp, nt)
            with txl.If(warp == 0), txl.Then():
                d = ld_s32(s_rh, lane)
                di = warp_incl_scan(d, lane)
                st_s32(s_rstart, lane, di - d)

            ch = txl.local_scalar("int32")

            def claim():
                bar()
                with txl.If(tid == 0), txl.Then():
                    c = txl.local_scalar("int32")
                    txl.ptx.atom.relaxed.gpu.global_.add.s32(c, ws.ptr_to([w["place"]]), txl.int32(1))
                    st_s32(s_flag, 1, c)
                bar()
                txl.assign(ch, ld_s32(s_flag, 1))

            claim()
            with txl.While(ch < num_ctas):
                cbo = cb_off + txl.Cast("int64", ch) * nb
                with txl.serial((ne + nt - 1) // nt) as it:
                    i = it * nt + tid
                    with txl.If(i < ne), txl.Then():
                        st_s32(s_start, i, ld_s32(s_ex, i) + ld_g32(ws, cbo + i))
                        st_s32(s_cur, i, txl.int32(0))
                with txl.If(tid < 32), txl.Then():
                    cb = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.If(tid < ep), txl.Then():
                        txl.assign(cb, ld_g32(ws, cbo + ne + tid))
                    st_s32(s_rbase, tid, cb)
                    st_s32(s_rcur, tid, txl.int32(0))
                bar()

                c0 = ch * tpc
                c1 = txl.Min(m, c0 + tpc)
                with txl.serial((tpc + nt - 1) // nt) as it:
                    tok = c0 + it * nt + tid
                    with txl.If(tok < c1), txl.Then():
                        mask = txl.local_scalar("uint32", init=txl.uint32(0))
                        tok64 = txl.Cast("int64", tok)
                        ids = [txl.local_scalar("int32", init=load_id(tok, j)) for j in range(k)]
                        ranks, poss = [], []
                        for j in range(k):
                            e = ids[j]
                            r = txl.local_scalar("int32", init=e // epr)
                            bit = txl.shift_left(txl.uint32(1), txl.Cast("uint32", r))
                            p = txl.local_scalar("int32", init=txl.int32(0))
                            with txl.If(txl.bitwise_and(mask, bit) == txl.uint32(0)):
                                with txl.Then():
                                    q = atom_s_add(s_rcur, r, 1)
                                    txl.assign(p, ld_s32(s_rbase, r) + q)
                                    st_g64(out, txl.Cast("int64", ld_s32(s_rstart, r) + p), tok64)
                                with txl.Else():
                                    for jj in range(j):
                                        txl.assign(p, txl.Select(ranks[jj] == r, poss[jj], p))
                            txl.assign(mask, txl.bitwise_or(mask, bit))
                            ranks.append(r)
                            poss.append(p)
                            s = atom_s_add(s_cur, e, 1)
                            sidx = txl.Cast("int64", ld_s32(s_start, e) + s)
                            st_g64(out, o_idx + sidx, tok64 * k + j)
                            st_g64(out, o_exp + sidx, txl.Cast("int64", p))

                with txl.If(ch == 0), txl.Then():
                    with txl.serial((ne + nt - 1) // nt) as it:
                        i = it * nt + tid
                        with txl.If(i < ne), txl.Then():
                            r = i // epr
                            st_g64(out, o_sm + (i + r), txl.Cast("int64", ld_s32(s_hist, i)))
                    with txl.If(tid < ep), txl.Then():
                        tsum = txl.local_scalar("int32", init=txl.int32(0))
                        with txl.serial(epr) as le:
                            txl.assign(tsum, tsum + ld_s32(s_hist, tid * epr + le))
                        dd = txl.Cast("int64", ld_s32(s_rh, tid))
                        st_g64(meta_out, txl.Cast("int64", tid), txl.Cast("int64", tsum))
                        st_g64(meta_out, txl.Cast("int64", tid + ep), dd)
                        st_g64(out, o_sm + (tid * (epr + 1) + epr), dd)
                claim()

        # ---- exit; the last CTA out re-zeroes counters and words ----------------------------
        bar()
        with txl.If(tid == 0), txl.Then():
            old = txl.local_scalar("int32")
            txl.ptx.atom.acq_rel.gpu.global_.add.s32(old, ws.ptr_to([w["exit"]]), txl.int32(1))
            st_s32(s_flag, 2, txl.Select(old == num_ctas - 1, txl.int32(1), txl.int32(0)))
        bar()
        with txl.If(ld_s32(s_flag, 2) == 1), txl.Then():
            txl.ptx.fence.acq_rel.gpu()
            with txl.serial((nb + 4 + nt - 1) // nt) as it:
                i = it * nt + tid
                with txl.If(i < nb + 4), txl.Then():
                    txl.ptx.st.relaxed.gpu.global_.b32(ws.ptr_to([i]), txl.int32(0))

    return ep_send_idx


# Self-resetting int32 workspace of the send kernel, one per (device, stream, ne, ep), so launches
# on two streams never share it.
WORKSPACES: dict = {}
WORKSPACE_ENTRIES = 8


def workspace(device, ne, ep):
    """
    int32: counters and words, zeroed once and kept zero by the kernel's last CTA; then the per-CTA
    bases (SEND_MAX_CTAS * (ne + ep)), rewritten every launch.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream, ne, ep)
    ws = WORKSPACES.get(key)
    if ws is None:
        off = words(ne, ep)["cb_off"]
        ws = torch.empty(off + SEND_MAX_CTAS * (ne + ep), dtype=torch.int32, device=device)
        zero_kernel(off)(ws)
        if len(WORKSPACES) >= WORKSPACE_ENTRIES:
            WORKSPACES.clear()
        WORKSPACES[key] = ws
    return ws


def send_idx_torch(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    The sender side in PyTorch (no host sync, token order within each bucket), for inputs the kernel
    does not cover. m > 0.
    """
    m, k = topk_ids.shape
    dev = topk_ids.device
    ids = topk_ids.to(torch.int64)
    flat = ids.reshape(-1)
    idxs = torch.argsort(flat, stable=True)
    visited = torch.zeros(m, ep_size, dtype=torch.bool, device=dev).scatter_(1, ids // experts_per_rank, True).t()
    pos = visited.long().cumsum(1) - 1
    rank_of = torch.arange(ep_size, device=dev)[:, None].expand(ep_size, m)
    dispatch_token_idxs = torch.argsort(torch.where(visited, rank_of, ep_size).reshape(-1), stable=True) % m
    expand_idx = pos[flat[idxs] // experts_per_rank, idxs // k]
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=dev).scatter_add_(0, flat, torch.ones_like(flat)).view(ep_size, experts_per_rank)
    dedup = visited.sum(1)
    meta_out[:ep_size].copy_(expert_counts.sum(1))
    meta_out[ep_size : 2 * ep_size].copy_(dedup)
    send_meta = torch.cat([expert_counts, dedup[:, None]], 1).reshape(-1)
    return meta_out[:ep_size], meta_out[ep_size : 2 * ep_size], dispatch_token_idxs, idxs, expand_idx, send_meta


def ep_dispatch_send_idx_sm100(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = topk_ids.shape
    dev = topk_ids.device
    assert ep_size <= 32, f"ep_size={ep_size} exceeds the 32-rank bitmask"
    if meta_out is None:
        meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=dev)
    tpr, dedup = meta_out[:ep_size], meta_out[ep_size : 2 * ep_size]
    nsm = ep_size * (experts_per_rank + 1)
    if m == 0:
        send_meta = torch.zeros(nsm, dtype=torch.int64, device=dev)
        meta_out[: 2 * ep_size].zero_()
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        return tpr, dedup, empty, empty.clone(), empty.clone(), send_meta
    # the kernel reads int32 or int64 ids with unit inner stride
    if topk_ids.dtype not in (torch.int32, torch.int64) or (k > 1 and topk_ids.stride(1) != 1) or topk_ids.stride(0) < 0:
        topk_ids = topk_ids.to(torch.int64).contiguous()
    if nsm > SEND_MAX_BINS or m * max(k, ep_size) >= 2**31:
        return send_idx_torch(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)
    ws = workspace(dev, num_experts, ep_size)
    out = torch.empty(m * ep_size + 2 * m * k + nsm, dtype=torch.int64, device=dev)
    dti, idxs, expand, send_meta = out.split((m * ep_size, m * k, m * k, nsm))
    num_ctas = min(SEND_MAX_CTAS, (m + SEND_MIN_TOKENS_PER_CTA - 1) // SEND_MIN_TOKENS_PER_CTA)
    tpc = (m + num_ctas - 1) // num_ctas
    num_ctas = (m + tpc - 1) // tpc
    fn = send_kernel(k, ep_size, experts_per_rank, "int64" if topk_ids.dtype == torch.int64 else "int32")
    stride = topk_ids.stride(0)
    flat_ids = topk_ids.as_strided(((m - 1) * stride + k,), (1,))
    fn(flat_ids, meta_out, out, ws, m, stride, tpc, num_ctas)
    return tpr, dedup, dti, idxs, expand, send_meta
