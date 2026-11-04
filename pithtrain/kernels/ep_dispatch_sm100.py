"""
TIRx-lite kernels of the ep_dispatch feature (SM100), one launch per side.

Sender side: one launch with no unconditional grid barrier. Each CTA counts its token chunk in
shared memory and claims its base in every expert / rank bin with one global atomic per bin. A CTA
spins on the others only after it has seen every CTA of the grid start, so the spin always ends even
when comm kernels on other streams hold SMs; otherwise it exits and leaves its placement to a shared
chunk queue (see define_send_kernel). Placement within a bucket uses shared-memory atomics, so the
order within a bucket is launch-dependent. The workspace (bin counters, start / arrive / place /
exit words and per-CTA bases) is int32, kept per (device, stream) and re-zeroed by the last CTA
out, so it is zeroed once on first use and never again from the host.

Receiver side: one launch. Every CTA scans the received metadata in shared memory, then
binary-searches the segment of each of its slots.

Inputs the kernels do not cover (more than SEND_MAX_BINS bins or RECV_MAX_SEGMENTS segments, more
than 32 ranks on the receiver side, 2^31 or more slots or dedup entries) take a PyTorch path with
the same outputs; topk_ids of another integer dtype or with a non-unit inner stride are converted to
contiguous int64 first. Neither side, nor the PyTorch path, synchronizes the device with the host.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["ep_dispatch_recv_idx_sm100", "ep_dispatch_send_idx_sm100"]

FULL_MASK = 0xFFFFFFFF

EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


def bar_sync():
    txl.ptx.bar.sync(txl.uint32(0))


def ld_s32(buf, idx):
    out = txl.local_scalar("int32")
    txl.ptx.ld.shared.b32(out, buf.ptr_to([idx]))
    return out


def st_s32(buf, idx, value):
    txl.ptx.st.shared.b32(buf.ptr_to([idx]), value)


def ld_g64(buf, idx):
    out = txl.local_scalar("int64")
    txl.ptx.ld.global_.b64(out, buf.ptr_to([idx]))
    return out


def ld_g32(buf, idx):
    out = txl.local_scalar("int32")
    txl.ptx.ld.global_.b32(out, buf.ptr_to([idx]))
    return out


def st_g64(buf, idx, value):
    tmp = txl.local_scalar("int64", init=value)
    txl.ptx.st.global_.b64(buf.ptr_to([idx]), tmp)


def atom_s_add(buf, idx, value):
    out = txl.local_scalar("uint32")
    txl.ptx.atom.shared.add.u32(out, buf.ptr_to([idx]), txl.uint32(value) if isinstance(value, int) else value)
    return txl.reinterpret(out, ty="int32")


def warp_inclusive_scan(value, lane):
    acc_sum = txl.local_scalar("int32", init=value)
    for offset in (1, 2, 4, 8, 16):
        peer = txl.local_scalar("int32")
        txl.ptx.shfl_sync.up.b32(peer, acc_sum, txl.uint32(offset), txl.uint32(0), txl.uint32(FULL_MASK))
        txl.assign(acc_sum, txl.Select(lane >= offset, acc_sum + peer, acc_sum))
    return acc_sum


def block_exclusive_scan(src, dst, num_items, tidx, lane, warp, s_warp, num_threads, add=None):
    """
    dst[i] = sum(src[:i]) (+ add[i] if given) over shared int32 arrays of num_items (trace-time).

    Each thread owns a contiguous run of ceil(num_items / num_threads) items; one warp scans the run
    totals. Ends with a CTA barrier. Returns nothing; dst may alias src only when add is None.
    """
    items_per_thread = (num_items + num_threads - 1) // num_threads
    num_warps = num_threads // 32
    run_total = txl.local_scalar("int32", init=txl.int32(0))
    for i in range(items_per_thread):
        idx = tidx * items_per_thread + i
        with txl.if_(idx < num_items), txl.then_():
            txl.assign(run_total, run_total + ld_s32(src, idx))
    run_end = warp_inclusive_scan(run_total, lane)
    with txl.if_(lane == 31), txl.then_():
        st_s32(s_warp, warp, run_end)
    bar_sync()
    with txl.if_(warp == 0), txl.then_():
        warp_total = txl.local_scalar("int32", init=txl.int32(0))
        with txl.if_(lane < num_warps), txl.then_():
            txl.assign(warp_total, ld_s32(s_warp, lane))
        warp_end = warp_inclusive_scan(warp_total, lane)
        with txl.if_(lane < num_warps), txl.then_():
            st_s32(s_warp, lane, warp_end - warp_total)
    bar_sync()
    acc_sum = txl.local_scalar("int32", init=ld_s32(s_warp, warp) + run_end - run_total)
    run_values = []
    for i in range(items_per_thread):
        idx = tidx * items_per_thread + i
        value = txl.local_scalar("int32", init=txl.int32(0))
        with txl.if_(idx < num_items), txl.then_():
            txl.assign(value, ld_s32(src, idx))
        run_values.append(value)
    bar_sync()
    for i in range(items_per_thread):
        idx = tidx * items_per_thread + i
        with txl.if_(idx < num_items), txl.then_():
            if add is None:
                st_s32(dst, idx, acc_sum)
            else:
                st_s32(dst, idx, acc_sum + ld_s32(add, idx))
        txl.assign(acc_sum, acc_sum + run_values[i])
    bar_sync()


RECV_WARPS = 8
RECV_PER_THREAD = 1
RECV_MAX_SEGMENTS = 2048  # ep_size * experts_per_rank the kernel covers (two int32 smem arrays)


@functools.cache
def compile_recv_kernel(ep_size, experts_per_rank):
    return entry_function(define_recv_kernel(ep_size, experts_per_rank).compile(TARGET))


def define_recv_kernel(ep_size, experts_per_rank):
    num_segments = ep_size * experts_per_rank
    num_threads = RECV_WARPS * 32
    slots_per_cta = num_threads * RECV_PER_THREAD
    num_search_steps = max(1, num_segments.bit_length())

    @txl.kernel()
    def ep_recv_idx(meta: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), received: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), expert_out: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), expand_out: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), recv_stride: txl.int64, total: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=RECV_WARPS * 32))
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        cta_idx = txl.cta_id()
        pool = txl.smem_pool()
        s_seg_count, s_seg_end, s_dedup_start, s_warp = pool.alloc([num_segments], "int32"), pool.alloc([num_segments], "int32"), pool.alloc([32], "int32"), pool.alloc([32], "int32")

        def load_received_values():
            """
            Issue this thread's received loads first: their latency overlaps the metadata scan.
            """
            received_values = []
            for i in range(RECV_PER_THREAD):
                slot = cta_idx * slots_per_cta + i * num_threads + tidx
                received_value = txl.local_scalar("int64", init=txl.int64(0))
                with txl.if_(slot < total), txl.then_():
                    txl.ptx.ld.global_.b64(received_value, received.ptr_to([txl.Cast("int64", slot) * recv_stride]))
                received_values.append(received_value)
            return received_values

        def load_segment_count(seg_idx):
            sender_idx = seg_idx // experts_per_rank
            return txl.Cast("int32", ld_g64(meta, txl.Cast("int64", sender_idx * (experts_per_rank + 1) + (seg_idx - sender_idx * experts_per_rank))))

        def load_segment_counts():
            """
            One segment per thread in a register when they fit, else all of them into s_seg_count.
            """
            if num_segments <= num_threads:
                count = txl.local_scalar("int32", init=txl.int32(0))
                with txl.if_(tidx < num_segments), txl.then_():
                    txl.assign(count, load_segment_count(tidx))
                return count
            with txl.serial((num_segments + num_threads - 1) // num_threads) as iteration:
                seg_idx = iteration * num_threads + tidx
                with txl.if_(seg_idx < num_segments), txl.then_():
                    st_s32(s_seg_count, seg_idx, load_segment_count(seg_idx))
            return None

        def scan_dedup_starts():
            """
            Exclusive prefix of the senders' dedup counts (ep_size <= 32) in warp 0.
            """
            with txl.if_(warp == 0), txl.then_():
                dedup = txl.local_scalar("int32", init=txl.int32(0))
                with txl.if_(lane < ep_size), txl.then_():
                    txl.assign(dedup, txl.Cast("int32", ld_g64(meta, txl.Cast("int64", lane * (experts_per_rank + 1) + experts_per_rank))))
                dedup_end = warp_inclusive_scan(dedup, lane)
                st_s32(s_dedup_start, lane, dedup_end - dedup)

        def scan_segment_ends(count):
            """
            Inclusive end of every segment into s_seg_end; ends with a CTA barrier.
            """
            if num_segments <= num_threads:
                count_end = warp_inclusive_scan(count, lane)
                with txl.if_(lane == 31), txl.then_():
                    st_s32(s_warp, warp, count_end)
                bar_sync()
                warp_prefix = txl.local_scalar("int32", init=txl.int32(0))
                for prev_warp in range(RECV_WARPS - 1):
                    txl.assign(warp_prefix, warp_prefix + txl.Select(warp > prev_warp, ld_s32(s_warp, prev_warp), txl.int32(0)))
                with txl.if_(tidx < num_segments), txl.then_():
                    st_s32(s_seg_end, tidx, warp_prefix + count_end)
                bar_sync()
            else:
                bar_sync()
                # exclusive scan plus own count
                block_exclusive_scan(s_seg_count, s_seg_end, num_segments, tidx, lane, warp, s_warp, num_threads, add=s_seg_count)

        def store_slot_indices(received_values):
            """
            Binary-search each slot's segment; store its local expert and its deduplicated row.
            """
            for i in range(RECV_PER_THREAD):
                slot = cta_idx * slots_per_cta + i * num_threads + tidx
                with txl.if_(slot < total), txl.then_():
                    search_start, search_end = txl.local_scalar("int32", init=txl.int32(0)), txl.local_scalar("int32", init=txl.int32(num_segments))
                    for _ in range(num_search_steps):
                        mid = txl.local_scalar("int32", init=(search_start + search_end) >> 1)
                        with txl.if_(search_start < search_end), txl.then_():
                            above = ld_s32(s_seg_end, txl.Min(mid, txl.int32(num_segments - 1))) > slot
                            txl.assign(search_end, txl.Select(above, mid, search_end))
                            txl.assign(search_start, txl.Select(above, search_start, mid + 1))
                    seg_idx = txl.Min(search_start, txl.int32(num_segments - 1))
                    sender_idx = seg_idx // experts_per_rank
                    expert_idx = seg_idx - sender_idx * experts_per_rank
                    st_g64(expert_out, txl.Cast("int64", slot), txl.Cast("int64", expert_idx))
                    st_g64(expand_out, txl.Cast("int64", slot), received_values[i] + txl.Cast("int64", ld_s32(s_dedup_start, sender_idx)))

        received_values = load_received_values()
        count = load_segment_counts()
        scan_dedup_starts()
        scan_segment_ends(count)
        store_slot_indices(received_values)

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
    """
    SM100 implementation of ep_dispatch_recv_idx.
    """
    # the kernel reads recv_meta with unit stride; received_expand_idx with its own (non-negative)
    recv_meta = recv_meta.contiguous()
    if received_expand_idx.stride(0) < 0:
        received_expand_idx = received_expand_idx.contiguous()
    total = received_expand_idx.shape[0]
    device = received_expand_idx.device
    if total == 0:
        return torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device)
    if not (ep_size <= 32 and 0 < ep_size * experts_per_rank <= RECV_MAX_SEGMENTS and total < 2**31 and received_expand_idx.dtype == torch.int64):
        return recv_idx_torch(recv_meta, received_expand_idx, ep_size, experts_per_rank)
    expert_idxs, expand_idx = torch.empty(total, dtype=torch.int64, device=device), torch.empty(total, dtype=torch.int64, device=device)
    fn = compile_recv_kernel(ep_size, experts_per_rank)
    slots_per_cta = RECV_WARPS * 32 * RECV_PER_THREAD
    stride = received_expand_idx.stride(0)
    received_flat = received_expand_idx.as_strided(((total - 1) * stride + 1,), (1,))
    fn(recv_meta, received_flat, expert_idxs, expand_idx, stride, total, (total + slots_per_cta - 1) // slots_per_cta)
    return expert_idxs, expand_idx


SEND_WARPS = 8
SEND_MIN_TOKENS_PER_CTA = 128
SEND_MAX_CTAS = 128
SEND_MAX_BINS = 2048  # ep_size * (experts_per_rank + 1) the kernel covers (shared memory < 48 KB)


@functools.cache
def compile_zero_kernel(num_words):
    @txl.kernel()
    def ep_ws_zero(workspace: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=128))
        tidx = txl.thread_id()
        with txl.serial((num_words + 127) // 128) as iteration:
            word_idx = iteration * 128 + tidx
            with txl.if_(word_idx < num_words), txl.then_():
                txl.ptx.st.global_.b32(workspace.ptr_to([word_idx]), txl.int32(0))

    return entry_function(ep_ws_zero.compile(TARGET))


def workspace_words(num_experts, ep_size):
    """
    Workspace word indices: counters [0, num_experts + ep_size), then start / arrive / place / exit,
    then the per-CTA bases from bases_start.
    """
    num_bins = num_experts + ep_size
    words = {"start": num_bins, "arrive": num_bins + 1, "place": num_bins + 2, "exit": num_bins + 3}
    words["bases_start"] = (num_bins + 4 + 3) // 4 * 4
    return words


@functools.cache
def compile_send_kernel(top_k, ep_size, experts_per_rank, ids_dtype):
    return entry_function(define_send_kernel(top_k, ep_size, experts_per_rank, ids_dtype).compile(TARGET))


def define_send_kernel(top_k, ep_size, experts_per_rank, ids_dtype):
    """
    Deadlock-free single launch.

    Phase 1 (never waits): CTA c counts its token chunk c in shared memory, claims its base in every
    expert / rank with one global atomic per bin and stores the bases at its row of the per-CTA
    bases; it then arrives. Every CTA bumped the start word on entry. A CTA spins on the arrive word
    only if start already equals the grid size: then every CTA is resident and will arrive without
    waiting on anything, so the spin ends. A CTA that sees an unstarted CTA exits instead. Placement
    is a shared queue of chunks claimed with the place word; the last CTA to start always sees
    start == grid size, waits and drains the queue, so every chunk is placed. In the common case
    all CTAs stay and each places about one chunk. The last CTA to exit re-zeroes the counters and
    words for the next launch.
    """
    num_experts = ep_size * experts_per_rank
    num_bins = num_experts + ep_size
    num_threads = SEND_WARPS * 32
    words = workspace_words(num_experts, ep_size)
    bases_start = words["bases_start"]

    @txl.kernel()
    def ep_send_idx(topk_ids: txl.Tensor((txl.int64(),), ids_dtype, data_alignment=1), meta_out: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), workspace: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), num_tokens: txl.int32, row_stride: txl.int64, tokens_per_cta: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=SEND_WARPS * 32))
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        cta_idx = txl.cta_id()
        pool = txl.smem_pool()
        s_bin_count, s_expert_start, s_slot_start, s_expert_fill = pool.alloc([num_bins], "int32"), pool.alloc([num_experts], "int32"), pool.alloc([num_experts], "int32"), pool.alloc([num_experts], "int32")
        s_rank_count, s_rank_fill, s_rank_base, s_rank_start, s_warp, s_flag = pool.alloc([32], "int32"), pool.alloc([32], "int32"), pool.alloc([32], "int32"), pool.alloc([32], "int32"), pool.alloc([32], "int32"), pool.alloc([4], "int32")

        # starts of the idxs, expand_idx and send_meta regions of out, after dispatch_token_idxs
        num_tokens64 = txl.Cast("int64", num_tokens)
        idxs_start = num_tokens64 * ep_size
        expand_start = idxs_start + num_tokens64 * top_k
        send_meta_start = expand_start + num_tokens64 * top_k

        def load_expert_id(token, j):
            id_offset = txl.Cast("int64", token) * row_stride + j
            if ids_dtype == "int64":
                return txl.Cast("int32", ld_g64(topk_ids, id_offset))
            return ld_g32(topk_ids, id_offset)

        def count_chunk():
            """
            Histogram of this CTA's chunk: per expert, and per rank the tokens sent there.
            """
            with txl.serial((num_bins + num_threads - 1) // num_threads) as iteration:
                bin_idx = iteration * num_threads + tidx
                with txl.if_(bin_idx < num_bins), txl.then_():
                    st_s32(s_bin_count, bin_idx, txl.int32(0))
            bar_sync()
            token_start = cta_idx * tokens_per_cta
            token_end = txl.Min(num_tokens, token_start + tokens_per_cta)
            with txl.serial((tokens_per_cta + num_threads - 1) // num_threads) as iteration:
                token = token_start + iteration * num_threads + tidx
                with txl.if_(token < token_end), txl.then_():
                    rank_mask = txl.local_scalar("uint32", init=txl.uint32(0))
                    expert_ids = [txl.local_scalar("int32", init=load_expert_id(token, j)) for j in range(top_k)]
                    for j in range(top_k):
                        expert_idx = expert_ids[j]
                        atom_s_add(s_bin_count, expert_idx, 1)
                        rank_idx = expert_idx // experts_per_rank
                        rank_bit = txl.shift_left(txl.uint32(1), txl.Cast("uint32", rank_idx))
                        with txl.if_(txl.bitwise_and(rank_mask, rank_bit) == txl.uint32(0)), txl.then_():
                            atom_s_add(s_bin_count, num_experts + rank_idx, 1)
                        txl.assign(rank_mask, txl.bitwise_or(rank_mask, rank_bit))
            bar_sync()

        def claim_bin_bases():
            """
            Claim this CTA's base in every bin with one global atomic and store it in its bases row.
            """
            with txl.serial((num_bins + num_threads - 1) // num_threads) as iteration:
                bin_idx = iteration * num_threads + tidx
                with txl.if_(bin_idx < num_bins), txl.then_():
                    count = ld_s32(s_bin_count, bin_idx)
                    bin_base = txl.local_scalar("uint32", init=txl.uint32(0))
                    with txl.if_(count > 0), txl.then_():
                        txl.ptx.atom.global_.add.u32(bin_base, workspace.ptr_to([bin_idx]), txl.reinterpret(count, ty="uint32"))
                    txl.ptx.st.global_.b32(workspace.ptr_to([bases_start + txl.Cast("int64", cta_idx) * num_bins + bin_idx]), txl.reinterpret(bin_base, ty="int32"))
            bar_sync()

        def arrive():
            """
            Arrive; stay (s_flag[0] = 1) and wait for all arrivals only if the whole grid started.
            """
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx.fence.acq_rel.gpu()
                txl.ptx.red.relaxed.gpu.global_.add.s32(workspace.ptr_to([words["arrive"]]), txl.int32(1))
                started = txl.local_scalar("int32")
                txl.ptx.ld.acquire.gpu.global_.b32(started, workspace.ptr_to([words["start"]]))
                stay = txl.Select(started >= num_ctas, txl.int32(1), txl.int32(0))
                st_s32(s_flag, 0, stay)
                with txl.if_(stay == 1), txl.then_():
                    # ty=void: tvm_tirx registers no result type for wait_until and silently drops
                    # a non-void call
                    spin = txl.local_scalar("int32", init=txl.int32(0))
                    txl.cuda.wait_until(spin, workspace.ptr_to([words["arrive"]]), spin >= num_ctas, scope="gpu", ptx_type="b32", ty=txl.void)
            bar_sync()

        def scan_bin_totals():
            """
            Load the grid's bin totals; exclusive starts of every expert and every rank.
            """
            with txl.serial((num_experts + num_threads - 1) // num_threads) as iteration:
                expert_idx = iteration * num_threads + tidx
                with txl.if_(expert_idx < num_experts), txl.then_():
                    total = txl.local_scalar("int32")
                    txl.ptx.ld.relaxed.gpu.global_.b32(total, workspace.ptr_to([expert_idx]))
                    st_s32(s_bin_count, expert_idx, total)
            with txl.if_(tidx < 32), txl.then_():
                total = txl.local_scalar("int32", init=txl.int32(0))
                with txl.if_(tidx < ep_size), txl.then_():
                    txl.ptx.ld.relaxed.gpu.global_.b32(total, workspace.ptr_to([num_experts + tidx]))
                st_s32(s_rank_count, tidx, total)
            bar_sync()
            block_exclusive_scan(s_bin_count, s_expert_start, num_experts, tidx, lane, warp, s_warp, num_threads)
            with txl.if_(warp == 0), txl.then_():
                rank_count = ld_s32(s_rank_count, lane)
                rank_end = warp_inclusive_scan(rank_count, lane)
                st_s32(s_rank_start, lane, rank_end - rank_count)

        def claim_chunk(chunk_idx):
            bar_sync()
            with txl.if_(tidx == 0), txl.then_():
                claimed = txl.local_scalar("int32")
                txl.ptx.atom.relaxed.gpu.global_.add.s32(claimed, workspace.ptr_to([words["place"]]), txl.int32(1))
                st_s32(s_flag, 1, claimed)
            bar_sync()
            txl.assign(chunk_idx, ld_s32(s_flag, 1))

        def load_chunk_bases(chunk_idx):
            """
            Slot starts of the chunk in every expert and rank bin; zero the chunk's fill counters.
            """
            chunk_bases = bases_start + txl.Cast("int64", chunk_idx) * num_bins
            with txl.serial((num_experts + num_threads - 1) // num_threads) as iteration:
                expert_idx = iteration * num_threads + tidx
                with txl.if_(expert_idx < num_experts), txl.then_():
                    st_s32(s_slot_start, expert_idx, ld_s32(s_expert_start, expert_idx) + ld_g32(workspace, chunk_bases + expert_idx))
                    st_s32(s_expert_fill, expert_idx, txl.int32(0))
            with txl.if_(tidx < 32), txl.then_():
                rank_base = txl.local_scalar("int32", init=txl.int32(0))
                with txl.if_(tidx < ep_size), txl.then_():
                    txl.assign(rank_base, ld_g32(workspace, chunk_bases + num_experts + tidx))
                st_s32(s_rank_base, tidx, rank_base)
                st_s32(s_rank_fill, tidx, txl.int32(0))
            bar_sync()

        def place_chunk(chunk_idx):
            """
            Store each token of the chunk once per rank it goes to and once per expert slot.
            """
            chunk_start = chunk_idx * tokens_per_cta
            chunk_end = txl.Min(num_tokens, chunk_start + tokens_per_cta)
            with txl.serial((tokens_per_cta + num_threads - 1) // num_threads) as iteration:
                token = chunk_start + iteration * num_threads + tidx
                with txl.if_(token < chunk_end), txl.then_():
                    rank_mask = txl.local_scalar("uint32", init=txl.uint32(0))
                    token64 = txl.Cast("int64", token)
                    expert_ids = [txl.local_scalar("int32", init=load_expert_id(token, j)) for j in range(top_k)]
                    rank_idxs, rank_positions = [], []
                    for j in range(top_k):
                        expert_idx = expert_ids[j]
                        rank_idx = txl.local_scalar("int32", init=expert_idx // experts_per_rank)
                        rank_bit = txl.shift_left(txl.uint32(1), txl.Cast("uint32", rank_idx))
                        rank_pos = txl.local_scalar("int32", init=txl.int32(0))
                        with txl.if_(txl.bitwise_and(rank_mask, rank_bit) == txl.uint32(0)):
                            with txl.then_():
                                rank_slot = atom_s_add(s_rank_fill, rank_idx, 1)
                                txl.assign(rank_pos, ld_s32(s_rank_base, rank_idx) + rank_slot)
                                st_g64(out, txl.Cast("int64", ld_s32(s_rank_start, rank_idx) + rank_pos), token64)
                            with txl.else_():
                                for prev in range(j):
                                    txl.assign(rank_pos, txl.Select(rank_idxs[prev] == rank_idx, rank_positions[prev], rank_pos))
                        txl.assign(rank_mask, txl.bitwise_or(rank_mask, rank_bit))
                        rank_idxs.append(rank_idx)
                        rank_positions.append(rank_pos)
                        expert_slot = atom_s_add(s_expert_fill, expert_idx, 1)
                        slot_idx = txl.Cast("int64", ld_s32(s_slot_start, expert_idx) + expert_slot)
                        st_g64(out, idxs_start + slot_idx, token64 * top_k + j)
                        st_g64(out, expand_start + slot_idx, txl.Cast("int64", rank_pos))

        def store_send_meta():
            """
            Per-rank token and dedup counts into meta_out and the send_meta rows of out.
            """
            with txl.serial((num_experts + num_threads - 1) // num_threads) as iteration:
                expert_idx = iteration * num_threads + tidx
                with txl.if_(expert_idx < num_experts), txl.then_():
                    rank_idx = expert_idx // experts_per_rank
                    st_g64(out, send_meta_start + (expert_idx + rank_idx), txl.Cast("int64", ld_s32(s_bin_count, expert_idx)))
            with txl.if_(tidx < ep_size), txl.then_():
                rank_tokens = txl.local_scalar("int32", init=txl.int32(0))
                with txl.serial(experts_per_rank) as local_expert:
                    txl.assign(rank_tokens, rank_tokens + ld_s32(s_bin_count, tidx * experts_per_rank + local_expert))
                rank_dedup = txl.Cast("int64", ld_s32(s_rank_count, tidx))
                st_g64(meta_out, txl.Cast("int64", tidx), txl.Cast("int64", rank_tokens))
                st_g64(meta_out, txl.Cast("int64", tidx + ep_size), rank_dedup)
                st_g64(out, send_meta_start + (tidx * (experts_per_rank + 1) + experts_per_rank), rank_dedup)

        def exit_grid():
            """
            Count this CTA out; the last CTA out re-zeroes the counters and words.
            """
            bar_sync()
            with txl.if_(tidx == 0), txl.then_():
                exited = txl.local_scalar("int32")
                txl.ptx.atom.acq_rel.gpu.global_.add.s32(exited, workspace.ptr_to([words["exit"]]), txl.int32(1))
                st_s32(s_flag, 2, txl.Select(exited == num_ctas - 1, txl.int32(1), txl.int32(0)))
            bar_sync()
            with txl.if_(ld_s32(s_flag, 2) == 1), txl.then_():
                txl.ptx.fence.acq_rel.gpu()
                with txl.serial((num_bins + 4 + num_threads - 1) // num_threads) as iteration:
                    word_idx = iteration * num_threads + tidx
                    with txl.if_(word_idx < num_bins + 4), txl.then_():
                        txl.ptx.st.relaxed.gpu.global_.b32(workspace.ptr_to([word_idx]), txl.int32(0))

        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.red.relaxed.gpu.global_.add.s32(workspace.ptr_to([words["start"]]), txl.int32(1))
        count_chunk()
        claim_bin_bases()
        arrive()

        with txl.if_(ld_s32(s_flag, 0) == 1), txl.then_():
            scan_bin_totals()
            chunk_idx = txl.local_scalar("int32")
            claim_chunk(chunk_idx)
            with txl.while_(chunk_idx < num_ctas):
                load_chunk_bases(chunk_idx)
                place_chunk(chunk_idx)
                with txl.if_(chunk_idx == 0), txl.then_():
                    store_send_meta()
                claim_chunk(chunk_idx)

        exit_grid()

    return ep_send_idx


# Self-resetting int32 workspace of the send kernel, one per (device, stream, num_experts, ep_size),
# so launches on two streams never share it. At most WORKSPACE_ENTRIES are cached (least recently
# used evicted); an evicted workspace is kept alive in RETIRED until an event recorded on its stream
# at eviction has completed, so its memory is never released while a launch on that stream may
# still use it.
WORKSPACES: dict = {}
WORKSPACE_ENTRIES = 8
RETIRED: list = []  # (event, workspace) of evicted workspaces whose stream may still be using them


def get_workspace(device, num_experts, ep_size):
    """
    int32: counters and words, zeroed once and kept zero by the kernel's last CTA; then the per-CTA
    bases (SEND_MAX_CTAS * (num_experts + ep_size)), rewritten every launch.
    """
    stream = torch.cuda.current_stream(device)
    key = (device, stream.cuda_stream, num_experts, ep_size)
    entry = WORKSPACES.pop(key, None)
    if entry is None:
        bases_start = workspace_words(num_experts, ep_size)["bases_start"]
        workspace = torch.empty(bases_start + SEND_MAX_CTAS * (num_experts + ep_size), dtype=torch.int32, device=device)
        compile_zero_kernel(bases_start)(workspace)
        RETIRED[:] = [(event, old) for event, old in RETIRED if not event.query()]
        while len(WORKSPACES) >= WORKSPACE_ENTRIES:
            old, old_stream = WORKSPACES.pop(next(iter(WORKSPACES)))
            event = torch.cuda.Event()
            event.record(old_stream)
            RETIRED.append((event, old))
        entry = (workspace, stream)
    WORKSPACES[key] = entry  # most recently used last
    return entry[0]


def send_idx_torch(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    The sender side in PyTorch (no host sync, token order within each bucket), for inputs the kernel
    does not cover. num_tokens > 0.
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    ids = topk_ids.to(torch.int64)
    flat = ids.reshape(-1)
    idxs = torch.argsort(flat, stable=True)
    visited = torch.zeros(num_tokens, ep_size, dtype=torch.bool, device=device).scatter_(1, ids // experts_per_rank, True).t()
    pos = visited.long().cumsum(1) - 1
    rank_of = torch.arange(ep_size, device=device)[:, None].expand(ep_size, num_tokens)
    dispatch_token_idxs = torch.argsort(torch.where(visited, rank_of, ep_size).reshape(-1), stable=True) % num_tokens
    expand_idx = pos[flat[idxs] // experts_per_rank, idxs // top_k]
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=device).scatter_add_(0, flat, torch.ones_like(flat)).view(ep_size, experts_per_rank)
    dedup = visited.sum(1)
    meta_out[:ep_size].copy_(expert_counts.sum(1))
    meta_out[ep_size : 2 * ep_size].copy_(dedup)
    send_meta = torch.cat([expert_counts, dedup[:, None]], 1).reshape(-1)
    return meta_out[:ep_size], meta_out[ep_size : 2 * ep_size], dispatch_token_idxs, idxs, expand_idx, send_meta


def ep_dispatch_send_idx_sm100(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SM100 implementation of ep_dispatch_send_idx; order within a bucket is atomic order.
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    assert ep_size <= 32, f"ep_size={ep_size} exceeds the 32-rank bitmask"
    if meta_out is None:
        meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=device)
    tokens_per_rank, dedup = meta_out[:ep_size], meta_out[ep_size : 2 * ep_size]
    num_bins = ep_size * (experts_per_rank + 1)
    if num_tokens == 0:
        send_meta = torch.zeros(num_bins, dtype=torch.int64, device=device)
        meta_out[: 2 * ep_size].zero_()
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return tokens_per_rank, dedup, empty, empty.clone(), empty.clone(), send_meta
    # the kernel reads int32 or int64 ids with unit inner stride
    if topk_ids.dtype not in (torch.int32, torch.int64) or (top_k > 1 and topk_ids.stride(1) != 1) or topk_ids.stride(0) < 0:
        topk_ids = topk_ids.to(torch.int64).contiguous()
    if num_bins > SEND_MAX_BINS or num_tokens * max(top_k, ep_size) >= 2**31:
        return send_idx_torch(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)
    workspace = get_workspace(device, num_experts, ep_size)
    out = torch.empty(num_tokens * ep_size + 2 * num_tokens * top_k + num_bins, dtype=torch.int64, device=device)
    dispatch_token_idxs, idxs, expand_idx, send_meta = out.split((num_tokens * ep_size, num_tokens * top_k, num_tokens * top_k, num_bins))
    num_ctas = min(SEND_MAX_CTAS, (num_tokens + SEND_MIN_TOKENS_PER_CTA - 1) // SEND_MIN_TOKENS_PER_CTA)
    tokens_per_cta = (num_tokens + num_ctas - 1) // num_ctas
    num_ctas = (num_tokens + tokens_per_cta - 1) // tokens_per_cta
    fn = compile_send_kernel(top_k, ep_size, experts_per_rank, "int64" if topk_ids.dtype == torch.int64 else "int32")
    stride = topk_ids.stride(0)
    flat_ids = topk_ids.as_strided(((num_tokens - 1) * stride + top_k,), (1,))
    fn(flat_ids, meta_out, out, workspace, num_tokens, stride, tokens_per_cta, num_ctas)
    return tokens_per_rank, dedup, dispatch_token_idxs, idxs, expand_idx, send_meta
