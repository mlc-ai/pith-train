"""
CuTe-DSL kernels of the ep_dispatch feature (SM90), launched through TVM-FFI: dispatch_index, the
sender side in one cooperative launch with stable (token-order) placement, and expand_index, the
receiver side. Inputs they do not cover (supports_index / supports_expand) take the Triton kernels
below (ep_dispatch_send_idx_triton / ep_dispatch_recv_idx_triton), which were the SM100 path before
its TIRx kernels. The Triton sender side is three launches: a per-CTA bincount (tl.histogram, no
atomics), a single-CTA reduce + prefix sum that also builds send_meta, and a dedup scatter +
counting sort that places rows with atomic_add, so the order within a bucket is launch-dependent.
The Triton receiver side is two: a single-CTA prefix sum of the received segment counts, then a
binary search per slot.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, atom_add_u64, fence_gpu, lanemask_lt, ld_b32, ld_relaxed, lds_b32, red_add_shared, red_or_shared, st_b32, st_b64, sts_b32

__all__ = ["dispatch_index", "expand_index", "ep_dispatch_recv_idx_sm90", "ep_dispatch_send_idx_sm90", "supports_expand", "supports_index"]

THREADS = 256
WARPS = THREADS // 32
ROUNDS = 2
CHUNK = ROUNDS * THREADS  # tokens per CTA of the index kernels
MAX_BINS = 1024  # experts + EP ranks the kernel's lane bitmasks cover (shared memory < 48 KB)
SLOTS_PER_THREAD = 4
MAX_SEGMENTS = 8 * THREADS


def exclusive_scan(s_src: Int32, s_dst: Int32, num_values: int, s_scan: Int32, tidx: Int32) -> None:
    """
    Shared memory, run by the whole CTA; s_dst may alias s_src, s_scan holds 2 * THREADS words. Ends
    with a barrier.
    """
    values_per_thread = (num_values + THREADS - 1) // THREADS
    values = []
    thread_sum = Int32(0)
    for i in range(values_per_thread):
        value_idx = tidx * values_per_thread + i
        value_ok = Int32(value_idx < num_values)
        value = Int32(lds_b32(s_src + cutlass.min(value_idx, num_values - 1) * 4)) * value_ok
        values.append(value)
        thread_sum = thread_sum + value
    sts_b32(s_scan + tidx * 4, Uint32(thread_sum), Int32(1))
    cute.arch.sync_threads()
    # Hillis-Steele scan of the thread sums, ping-ponging between the two halves of s_scan
    src_half = 0
    for step in range(THREADS.bit_length() - 1):
        offset = 1 << step
        has_left, own_sum = Int32(tidx >= offset), Int32(lds_b32(s_scan + (src_half * THREADS + tidx) * 4))
        left_sum = Int32(lds_b32(s_scan + (src_half * THREADS + tidx - offset * has_left) * 4))
        sts_b32(s_scan + ((1 - src_half) * THREADS + tidx) * 4, Uint32(own_sum + left_sum * has_left), Int32(1))
        cute.arch.sync_threads()
        src_half = 1 - src_half
    acc_prefix = Int32(lds_b32(s_scan + (src_half * THREADS + tidx) * 4)) - thread_sum
    for i in range(values_per_thread):
        value_idx = tidx * values_per_thread + i
        sts_b32(s_dst + value_idx * 4, Uint32(acc_prefix), Int32(value_idx < num_values))
        acc_prefix = acc_prefix + values[i]
    cute.arch.sync_threads()


class IndexBuffers:
    def __init__(self, out_base: Int64, num_slots: Int64, hist_base: Int64, s_counts: Int32, s_starts: Int32, s_next_pos: Int32, s_lane_masks: Int32, s_scan: Int32) -> None:
        self.out_base, self.num_slots = out_base, num_slots
        self.idxs_base, self.expand_base, self.inv_base = out_base, out_base + num_slots * 8, out_base + 2 * num_slots * 8
        self.dispatch_token_base = out_base + 3 * num_slots * 8
        self.hist_base = hist_base
        self.s_counts, self.s_starts, self.s_next_pos = s_counts, s_starts, s_next_pos
        self.s_lane_masks, self.s_scan = s_lane_masks, s_scan


class DispatchIndex:
    """
    Bins [0, num_experts) count slots per expert, bins [num_experts, num_bins) first visits
    (distinct tokens) per EP rank. Each CTA owns chunks cta_idx, cta_idx + num_ctas, ... of CHUNK
    tokens. Phase 1 publishes every owned chunk's histogram; after a grid barrier, phase 2 scans the
    totals into the starts, adds the lower chunks' histograms to get each owned chunk's bases, and
    ranks its tokens stably with per-warp lane bitmasks (so a token's k experts must be distinct).

    The grid barrier needs every CTA resident: the launch is cooperative and the grid capped at the
    SM count. Its counter is epoch << 32 | arrivals; the last arrival moves it to the next epoch, so
    it never needs resetting.

    Output layout (one int64 tensor, num_slots = M * K): idxs [0, num_slots), expand_idx [num_slots,
    2 * num_slots), inv_idxs [2 * num_slots, 3 * num_slots), dispatch_token_idxs [3 * num_slots, 3 *
    num_slots + M * ep_size), send_meta after it. Scratch: hist (num_chunks * num_bins int32), the
    barrier counter (int64).
    """

    def __init__(self, k: int, ep_size: int, experts_per_rank: int, max_ctas: int) -> None:
        self.K, self.ep_size, self.experts_per_rank = k, ep_size, experts_per_rank
        self.num_experts = ep_size * experts_per_rank
        self.num_bins = self.num_experts + ep_size
        self.max_ctas = max_ctas
        assert self.num_bins <= MAX_BINS and ep_size <= 32

    @cute.jit
    def __call__(self, mTopk: cute.Tensor, mOut: cute.Tensor, mCounts: cute.Tensor, mHist: cute.Tensor, mCounter: cute.Tensor, stream) -> None:
        num_chunks = (mTopk.shape[0] + CHUNK - 1) // CHUNK
        num_ctas = cutlass.min(num_chunks, self.max_ctas)
        self.kernel(mTopk, mOut, mCounts, mHist, mCounter).launch(grid=[num_ctas, 1, 1], block=[THREADS, 1, 1], stream=stream, cooperative=True)

    @cute.kernel
    def kernel(self, mTopk: cute.Tensor, mOut: cute.Tensor, mCounts: cute.Tensor, mHist: cute.Tensor, mCounter: cute.Tensor) -> None:
        num_bins = self.num_bins
        tidx, _, _ = cute.arch.thread_idx()
        cta_idx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        warp = tidx // 32
        lane = tidx - warp * 32
        M = mTopk.shape[0]
        num_chunks = (M + CHUNK - 1) // CHUNK
        num_slots = Int64(M) * self.K
        counter_base = mCounter.iterator.toint()

        smem = cutlass.memory.SmemAllocator()
        s_counts = smem.allocate(num_bins * 4, byte_alignment=16).toint()  # chunk histogram, then totals
        s_starts = smem.allocate(num_bins * 4, byte_alignment=16).toint()  # exclusive starts of the totals
        s_next_pos = smem.allocate(num_bins * 4, byte_alignment=16).toint()
        s_lane_masks = smem.allocate(num_bins * WARPS * 4, byte_alignment=16).toint()
        s_scan = smem.allocate(2 * THREADS * 4, byte_alignment=16).toint()
        buffers = IndexBuffers(mOut.iterator.toint(), num_slots, mHist.iterator.toint(), s_counts, s_starts, s_next_pos, s_lane_masks, s_scan)

        self.publish_histograms(mTopk, buffers, tidx, cta_idx, num_ctas, num_chunks)
        self.grid_barrier(counter_base, tidx, num_ctas)
        self.scan_totals(buffers, tidx, num_chunks)
        if cta_idx == 0:
            self.store_counts(mCounts, buffers, tidx, M)
        lt_mask = lanemask_lt()
        lane_bit = Uint32(1) << Uint32(lane)
        rank_bins_start = Int32(lds_b32(s_starts + self.num_experts * 4))
        for chunk_idx in cutlass.range(cta_idx, num_chunks, num_ctas, unroll=1):
            self.load_chunk_bases(buffers, tidx, chunk_idx)
            expert_ids, first_visits, tokens_ok, token_idxs = self.load_ids(mTopk, chunk_idx, tidx)
            for round_idx in cutlass.range_constexpr(ROUNDS):
                self.mark_lanes(expert_ids[round_idx], first_visits[round_idx], tokens_ok[round_idx], buffers, warp, lane_bit)
                cute.arch.sync_threads()
                self.place_round(expert_ids[round_idx], first_visits[round_idx], tokens_ok[round_idx], token_idxs[round_idx], buffers, warp, lt_mask, rank_bins_start)
                cute.arch.sync_threads()
                self.advance_positions(buffers, tidx)
                cute.arch.sync_threads()

    def load_ids(self, mTopk: cute.Tensor, chunk_idx: Int32, tidx: Int32) -> tuple:
        M = mTopk.shape[0]
        row_stride = Int64(mTopk.stride[0])
        topk_base = mTopk.iterator.toint()
        expert_ids, first_visits, tokens_ok, token_idxs = [], [], [], []
        for round_idx in range(ROUNDS):
            token_idx = chunk_idx * CHUNK + round_idx * THREADS + tidx
            token_ok = Int32(token_idx < M)
            token_experts = []
            for j in range(self.K):
                token_experts.append(ld_b32(topk_base + (Int64(token_idx) * row_stride + j) * 8, token_ok, Int32(0)))
            visited_ranks = Int32(0)
            is_first = []
            for j in range(self.K):
                rank_bit = Int32(1) << (token_experts[j] // self.experts_per_rank)
                is_first.append(token_ok & Int32((visited_ranks & rank_bit) == 0))
                visited_ranks = visited_ranks | rank_bit
            expert_ids.append(token_experts)
            first_visits.append(is_first)
            tokens_ok.append(token_ok)
            token_idxs.append(token_idx)
        return expert_ids, first_visits, tokens_ok, token_idxs

    @cute.jit
    def publish_histograms(self, mTopk: cute.Tensor, buffers, tidx, cta_idx, num_ctas, num_chunks) -> None:
        num_experts, num_bins, experts_per_rank = self.num_experts, self.num_bins, self.experts_per_rank
        s_counts, hist_base = buffers.s_counts, buffers.hist_base
        for chunk_idx in cutlass.range(cta_idx, num_chunks, num_ctas, unroll=1):
            for j in cutlass.range_constexpr((num_bins + THREADS - 1) // THREADS):
                bin_idx = j * THREADS + tidx
                sts_b32(s_counts + bin_idx * 4, Uint32(0), Int32(bin_idx < num_bins))
            cute.arch.sync_threads()
            expert_ids, first_visits, tokens_ok, _ = self.load_ids(mTopk, chunk_idx, tidx)
            for round_idx in cutlass.range_constexpr(ROUNDS):
                for j in cutlass.range_constexpr(self.K):
                    expert_idx = expert_ids[round_idx][j]
                    red_add_shared(s_counts + expert_idx * 4, Int32(1), tokens_ok[round_idx])
                    red_add_shared(s_counts + (num_experts + expert_idx // experts_per_rank) * 4, Int32(1), first_visits[round_idx][j])
            cute.arch.sync_threads()
            for j in cutlass.range_constexpr((num_bins + THREADS - 1) // THREADS):
                bin_idx = j * THREADS + tidx
                count = lds_b32(s_counts + cutlass.min(bin_idx, num_bins - 1) * 4)
                st_b32(hist_base + (Int64(chunk_idx) * num_bins + bin_idx) * 4, count, Int32(bin_idx < num_bins))
            cute.arch.sync_threads()

    @cute.jit
    def grid_barrier(self, counter_base, tidx, num_ctas) -> None:
        """
        Also makes each CTA's hist stores visible to every CTA.
        """
        fence_gpu()
        cute.arch.sync_threads()
        if tidx == 0:
            prev_counter = atom_add_u64(counter_base, Int64(1))
            num_arrived, epoch = Int32(prev_counter & 0xFFFFFFFF), Int32(prev_counter >> 32)
            is_last = Int32(num_arrived == num_ctas - 1)
            # the last arrival starts the next epoch and zeroes the arrivals; the others wait for it
            asm(
                None, [Int64(counter_base).ir_value(), Int64((Int64(1) << 32) - Int64(num_ctas)).ir_value(), Int32(is_last).ir_value()],
                "{\n\t.reg .pred p;\n\t.reg .b64 t;\n\tsetp.ne.b32 p, $2, 0;\n\t"
                "@p atom.relaxed.gpu.global.add.u64 t, [$0], $1;\n\t}",
                "l,l,r", side_effects=True,
            )
            # all but the last arrival: spin with acquire loads until the epoch (high word of the counter) moves on
            asm(
                None, [Int64(counter_base).ir_value(), Int32(epoch).ir_value(), Int32(Int32(1) - is_last).ir_value()],
                "{\n\t.reg .pred p, q;\n\t.reg .b32 lo, hi;\n\t"
                "setp.eq.b32 q, $2, 0;\n\t"
                "@q bra EP_DONE_${:uid};\n"
                "EP_LOOP_${:uid}:\n\t"
                "ld.acquire.gpu.global.v2.b32 {lo, hi}, [$0];\n\t"
                "setp.eq.b32 p, hi, $1;\n\t"
                "@p bra EP_LOOP_${:uid};\n"
                "EP_DONE_${:uid}:\n\t}",
                "l,r,r", side_effects=True,
            )
        cute.arch.sync_threads()
        fence_gpu()

    @cute.jit
    def scan_totals(self, buffers, tidx, num_chunks) -> None:
        """
        Every CTA computes the totals and starts itself.
        """
        num_bins, s_counts, hist_base = self.num_bins, buffers.s_counts, buffers.hist_base
        for j in cutlass.range_constexpr((num_bins + THREADS - 1) // THREADS):
            bin_idx = j * THREADS + tidx
            bin_ok, acc_total = Int32(bin_idx < num_bins), Int32(0)
            for chunk_idx in cutlass.range(num_chunks, unroll=4):
                acc_total = acc_total + ld_relaxed(hist_base + (Int64(chunk_idx) * num_bins + bin_idx) * 4, bin_ok)
            sts_b32(s_counts + cutlass.min(bin_idx, num_bins - 1) * 4, Uint32(acc_total), bin_ok)
        cute.arch.sync_threads()
        exclusive_scan(s_counts, buffers.s_starts, num_bins, buffers.s_scan, tidx)

    @cute.jit
    def store_counts(self, mCounts: cute.Tensor, buffers, tidx, M) -> None:
        """
        Store tokens_per_ep_rank, dedup_tokens_per_gpu and send_meta from the totals (CTA 0 only).
        """
        ep_size, experts_per_rank, num_experts, num_bins = self.ep_size, self.experts_per_rank, self.num_experts, self.num_bins
        s_counts, s_starts = buffers.s_counts, buffers.s_starts
        counts_base = mCounts.iterator.toint()
        if tidx < ep_size:
            rank_start, rank_end = Int32(lds_b32(s_starts + tidx * experts_per_rank * 4)), Int32(lds_b32(s_starts + (tidx + 1) * experts_per_rank * 4))
            st_b64(counts_base + Int64(tidx) * 8, Int64(rank_end - rank_start), Int32(1))
            num_dedup = Int32(lds_b32(s_counts + (num_experts + tidx) * 4))
            st_b64(counts_base + Int64(ep_size + tidx) * 8, Int64(num_dedup), Int32(1))
        send_meta_base = buffers.out_base + (3 * buffers.num_slots + Int64(M) * ep_size) * 8
        # send_meta row of rank r: its experts' slot counts, then its dedup count (bin num_experts + r)
        for j in cutlass.range_constexpr((ep_size * (experts_per_rank + 1) + THREADS - 1) // THREADS):
            meta_idx = j * THREADS + tidx
            meta_ok = Int32(meta_idx < ep_size * (experts_per_rank + 1))
            rank_idx = meta_idx // (experts_per_rank + 1)
            field_idx = meta_idx - rank_idx * (experts_per_rank + 1)
            bin_idx = rank_idx * experts_per_rank + field_idx + (num_experts + rank_idx - rank_idx * experts_per_rank - field_idx) * Int32(field_idx == experts_per_rank)
            count = Int32(lds_b32(s_counts + cutlass.min(bin_idx, num_bins - 1) * 4))
            st_b64(send_meta_base + Int64(meta_idx) * 8, Int64(count), meta_ok)

    @cute.jit
    def load_chunk_bases(self, buffers, tidx, chunk_idx) -> None:
        """
        Set each bin's next position to the chunk's base and clear the lane masks. The base is the
        expert's global start (EP-rank bins: 0, expand_idx is local to the rank's dedup chunk) plus
        everything the lower chunks put in the bin.
        """
        num_experts, num_bins, hist_base = self.num_experts, self.num_bins, buffers.hist_base
        for j in cutlass.range_constexpr((num_bins + THREADS - 1) // THREADS):
            bin_idx = j * THREADS + tidx
            bin_ok = Int32(bin_idx < num_bins)
            bin_clamped = cutlass.min(bin_idx, num_bins - 1)
            acc_pos = Int32(lds_b32(buffers.s_starts + bin_clamped * 4)) * Int32(bin_idx < num_experts)
            for lower_chunk_idx in cutlass.range(chunk_idx, unroll=4):
                acc_pos = acc_pos + ld_relaxed(hist_base + (Int64(lower_chunk_idx) * num_bins + bin_idx) * 4, bin_ok)
            sts_b32(buffers.s_next_pos + bin_clamped * 4, Uint32(acc_pos), bin_ok)
        for j in cutlass.range_constexpr((num_bins * WARPS + THREADS - 1) // THREADS):
            mask_idx = j * THREADS + tidx
            sts_b32(buffers.s_lane_masks + mask_idx * 4, Uint32(0), Int32(mask_idx < num_bins * WARPS))
        cute.arch.sync_threads()

    def mark_lanes(self, token_experts: list, is_first: list, token_ok: Int32, buffers: IndexBuffers, warp: Int32, lane_bit: Uint32) -> None:
        num_experts, experts_per_rank, s_lane_masks = self.num_experts, self.experts_per_rank, buffers.s_lane_masks
        for j in range(self.K):
            expert_idx = token_experts[j]
            red_or_shared(s_lane_masks + (expert_idx * WARPS + warp) * 4, lane_bit, token_ok)
            red_or_shared(s_lane_masks + ((num_experts + expert_idx // experts_per_rank) * WARPS + warp) * 4, lane_bit, is_first[j])

    def place_round(self, token_experts: list, is_first: list, token_ok: Int32, token_idx: Int32, buffers: IndexBuffers, warp: Int32, lt_mask: Uint32, rank_bins_start: Int32) -> None:
        """
        Rank the thread's slots among this round's lower lanes (lower warps, then lower lanes of its
        own warp) in each bin.
        """
        K, num_experts, experts_per_rank = self.K, self.num_experts, self.experts_per_rank
        s_lane_masks, s_next_pos, s_starts = buffers.s_lane_masks, buffers.s_next_pos, buffers.s_starts
        for j in range(K):
            expert_idx = token_experts[j]
            expert_masks, rank_masks = expert_idx * WARPS, (num_experts + expert_idx // experts_per_rank) * WARPS
            expert_offset, dedup_offset = Int32(0), Int32(0)
            for i in range(WARPS):
                is_lower_warp = Int32(i < warp)
                expert_offset = expert_offset + Int32(cute.arch.popc(lds_b32(s_lane_masks + (expert_masks + i) * 4))) * is_lower_warp
                dedup_offset = dedup_offset + Int32(cute.arch.popc(lds_b32(s_lane_masks + (rank_masks + i) * 4))) * is_lower_warp
            expert_offset = expert_offset + Int32(cute.arch.popc(lds_b32(s_lane_masks + (expert_masks + warp) * 4) & lt_mask))
            dedup_offset = dedup_offset + Int32(cute.arch.popc(lds_b32(s_lane_masks + (rank_masks + warp) * 4) & lt_mask))
            slot_pos, dedup_pos = Int32(lds_b32(s_next_pos + expert_idx * 4)) + expert_offset, Int32(lds_b32(s_next_pos + (num_experts + expert_idx // experts_per_rank) * 4)) + dedup_offset
            rank_dispatch_start = Int32(lds_b32(s_starts + (num_experts + expert_idx // experts_per_rank) * 4)) - rank_bins_start
            flat_idx = Int64(token_idx) * K + j
            st_b64(buffers.idxs_base + Int64(slot_pos) * 8, flat_idx, token_ok)
            st_b64(buffers.expand_base + Int64(slot_pos) * 8, Int64(dedup_pos), token_ok)
            st_b64(buffers.inv_base + flat_idx * 8, Int64(slot_pos), token_ok)
            st_b64(buffers.dispatch_token_base + Int64(rank_dispatch_start + dedup_pos) * 8, Int64(token_idx), is_first[j])

    @cute.jit
    def advance_positions(self, buffers, tidx) -> None:
        num_bins, s_lane_masks, s_next_pos = self.num_bins, buffers.s_lane_masks, buffers.s_next_pos
        for j in cutlass.range_constexpr((num_bins + THREADS - 1) // THREADS):
            bin_idx = j * THREADS + tidx
            if bin_idx < num_bins:
                acc_count = Int32(0)
                for i in cutlass.range_constexpr(WARPS):
                    acc_count = acc_count + Int32(cute.arch.popc(lds_b32(s_lane_masks + (bin_idx * WARPS + i) * 4)))
                    sts_b32(s_lane_masks + (bin_idx * WARPS + i) * 4, Uint32(0), Int32(1))
                sts_b32(s_next_pos + bin_idx * 4, Uint32(Int32(lds_b32(s_next_pos + bin_idx * 4)) + acc_count), Int32(1))


class ExpertAndExpandIdx:
    """
    One CTA per THREADS * SLOTS_PER_THREAD received slots; each slot finds its (sender, expert)
    segment by a binary search over the segment starts scanned in shared memory.
    """

    def __init__(self, ep_size: int, experts_per_rank: int) -> None:
        self.ep_size = ep_size
        self.experts_per_rank = experts_per_rank
        self.num_segments = ep_size * experts_per_rank
        assert 0 < self.num_segments <= MAX_SEGMENTS and ep_size <= THREADS

    @cute.jit
    def __call__(self, mMeta: cute.Tensor, mRecv: cute.Tensor, mOut: cute.Tensor, stream) -> None:
        slots_per_cta = THREADS * SLOTS_PER_THREAD
        num_ctas = (mRecv.shape[0] + slots_per_cta - 1) // slots_per_cta
        self.kernel(mMeta, mRecv, mOut).launch(grid=[num_ctas, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mMeta: cute.Tensor, mRecv: cute.Tensor, mOut: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        cta_idx, _, _ = cute.arch.block_idx()
        smem = cutlass.memory.SmemAllocator()
        s_seg_starts = smem.allocate(self.num_segments * 4, byte_alignment=16).toint()  # segment sizes, then their exclusive starts
        s_dedup_starts = smem.allocate(self.ep_size * 4, byte_alignment=16).toint()  # exclusive dedup starts per sender
        s_scan = smem.allocate(2 * THREADS * 4, byte_alignment=16).toint()
        self.scan_segments(mMeta.iterator.toint(), s_seg_starts, s_dedup_starts, s_scan, tidx)
        self.expand_slots(mRecv, mOut.iterator.toint(), s_seg_starts, s_dedup_starts, tidx, cta_idx)

    @cute.jit
    def scan_segments(self, meta_base, s_seg_starts, s_dedup_starts, s_scan, tidx) -> None:
        ep_size, experts_per_rank, num_segments = self.ep_size, self.experts_per_rank, self.num_segments
        for j in cutlass.range_constexpr((num_segments + THREADS - 1) // THREADS):
            seg_idx = j * THREADS + tidx
            seg_ok = Int32(seg_idx < num_segments)
            sender_idx = seg_idx // experts_per_rank
            seg_size = ld_b32(meta_base + Int64(seg_idx + sender_idx) * 8, seg_ok, Int32(0))  # (experts_per_rank + 1) per sender
            sts_b32(s_seg_starts + cutlass.min(seg_idx, num_segments - 1) * 4, Uint32(seg_size), seg_ok)
        if tidx < ep_size:
            acc_dedup = Int32(0)
            for sender_idx in cutlass.range(tidx, unroll=1):
                acc_dedup = acc_dedup + ld_b32(meta_base + Int64(sender_idx * (experts_per_rank + 1) + experts_per_rank) * 8, Int32(1), Int32(0))
            sts_b32(s_dedup_starts + tidx * 4, Uint32(acc_dedup), Int32(1))
        cute.arch.sync_threads()
        exclusive_scan(s_seg_starts, s_seg_starts, num_segments, s_scan, tidx)

    @cute.jit
    def expand_slots(self, mRecv: cute.Tensor, out_base, s_seg_starts, s_dedup_starts, tidx, cta_idx) -> None:
        experts_per_rank, num_segments = self.experts_per_rank, self.num_segments
        num_slots = mRecv.shape[0]
        recv_base = mRecv.iterator.toint()
        top_step = 1 << (num_segments.bit_length() - 1)
        for i in cutlass.range_constexpr(SLOTS_PER_THREAD):
            slot_idx = (cta_idx * SLOTS_PER_THREAD + i) * THREADS + tidx
            slot_ok = Int32(slot_idx < num_slots)
            seg_idx = Int32(0)  # last segment whose start is <= slot_idx (empty segments share starts)
            step = top_step
            while step > 0:
                probe_idx = seg_idx + step
                probe_ok, probe_start = Int32(probe_idx < num_segments), Int32(lds_b32(s_seg_starts + cutlass.min(probe_idx, num_segments - 1) * 4))
                seg_idx = seg_idx + step * (probe_ok & Int32(probe_start <= slot_idx))
                step //= 2
            sender_idx = seg_idx // experts_per_rank
            expert_idx = seg_idx - sender_idx * experts_per_rank
            local_idx = ld_b32(recv_base + Int64(slot_idx) * 8, slot_ok, Int32(0))
            global_idx = local_idx + Int32(lds_b32(s_dedup_starts + sender_idx * 4))
            st_b64(out_base + Int64(slot_idx) * 8, Int64(expert_idx), slot_ok)
            st_b64(out_base + (Int64(num_slots) + Int64(slot_idx)) * 8, Int64(global_idx), slot_ok)


INDEX: dict = {}
EXPAND: dict = {}


class Scratch:
    def __init__(self, device: torch.device, num_chunks: int, num_bins: int) -> None:
        self.num_chunks, self.num_bins = num_chunks, num_bins
        self.hist = torch.empty(num_chunks * num_bins, dtype=torch.int32, device=device)
        self.counter = torch.zeros(1, dtype=torch.int64, device=device)


SCRATCH: dict = {}


def get_scratch(device: torch.device, num_chunks: int, num_bins: int) -> Scratch:
    cached = SCRATCH.get(device.index)
    if cached is None or cached.num_chunks < num_chunks or cached.num_bins < num_bins:
        num_chunks, num_bins = max(num_chunks, cached.num_chunks if cached is not None else 0, 64), max(num_bins, cached.num_bins if cached is not None else 0)
        cached = SCRATCH[device.index] = Scratch(device, num_chunks, num_bins)
    return cached


def compile_index(k: int, ep_size: int, experts_per_rank: int, max_ctas: int) -> Callable:
    return cute.compile(DispatchIndex(k, ep_size, experts_per_rank, max_ctas), fake(Int64, (cute.sym_int(), k), (cute.sym_int64(), 1), 8), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int64, (2 * ep_size,), (1,), 8), fake(Int32, (cute.sym_int(),), (1,), 4), fake(Int64, (1,), (1,), 8), fake_stream(), options="--enable-tvm-ffi")


def supports_index(topk_ids: torch.Tensor, ep_size: int, experts_per_rank: int) -> bool:
    return (topk_ids.dtype == torch.int64 and topk_ids.dim() == 2 and topk_ids.stride(1) == 1 and ep_size <= 32 and ep_size * (experts_per_rank + 1) <= MAX_BINS and topk_ids.shape[0] * max(topk_ids.shape[1], ep_size) < 2**31)


def dispatch_index(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """
    The outputs of ep_dispatch_send_idx with stable placement, plus inv_idxs (m * k,)
    (inv_idxs[idxs[p]] == p) as a seventh. All index outputs share one int64 allocation.
    """
    M, K = topk_ids.shape
    device = topk_ids.device
    assert num_experts == ep_size * experts_per_rank
    if meta_out is None:
        meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=device)
    counts = meta_out[: 2 * ep_size]
    num_meta, num_slots = ep_size * (experts_per_rank + 1), M * K
    out = torch.empty(3 * num_slots + M * ep_size + num_meta, dtype=torch.int64, device=device)
    send_meta = out[3 * num_slots + M * ep_size :]
    if M == 0:
        counts.zero_()
        send_meta.zero_()
    else:
        key = (K, ep_size, experts_per_rank, SM_COUNT)
        fn = INDEX.get(key)
        if fn is None:
            fn = INDEX[key] = compile_index(*key)
        scratch = get_scratch(device, (M + CHUNK - 1) // CHUNK, num_experts + ep_size)
        fn(topk_ids, out, counts, scratch.hist, scratch.counter)
    return (counts[:ep_size], counts[ep_size:], out[3 * num_slots : 3 * num_slots + M * ep_size], out[:num_slots], out[num_slots : 2 * num_slots], send_meta, out[2 * num_slots : 3 * num_slots])


def compile_expand(ep_size: int, experts_per_rank: int) -> Callable:
    return cute.compile(ExpertAndExpandIdx(ep_size, experts_per_rank), fake(Int64, (ep_size * (experts_per_rank + 1),), (1,), 8), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int64, (cute.sym_int(),), (1,), 8), fake_stream(), options="--enable-tvm-ffi")


def supports_expand(received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> bool:
    return (0 < ep_size * experts_per_rank <= MAX_SEGMENTS and ep_size <= 32 and received_expand_idx.shape[0] < 2**31 and received_expand_idx.dtype == torch.int64 and received_expand_idx.is_contiguous())


def expand_index(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The outputs of ep_dispatch_recv_idx, sharing one int64 allocation.
    """
    total = received_expand_idx.shape[0]
    out = received_expand_idx.new_empty(2 * total)
    if total == 0:
        return out[:0], out[0:0]
    key = (ep_size, experts_per_rank)
    fn = EXPAND.get(key)
    if fn is None:
        fn = EXPAND[key] = compile_expand(ep_size, experts_per_rank)
    fn(recv_meta, received_expand_idx, out)
    return out[:total], out[total:]


@triton.jit
def dedup_bincount_kernel(
    topk_ids_ptr,
    per_cta_expert_hist_ptr,  # [num_ctas * NE_PADDED] int64, output
    per_cta_rank_hist_ptr,  # [num_ctas * EP_PADDED] int64, output
    M,
    topk_row_stride,
    TOPK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NE_PADDED: tl.constexpr,  # next_power_of_2(NUM_EXPERTS + 1)
    EP_PADDED: tl.constexpr,  # next_power_of_2(EP_SIZE + 1)
    BLOCK: tl.constexpr,
):
    """
    Per-CTA histograms of slots per expert and of distinct tokens per EP rank.
    """
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    acc_expert_hist, acc_rank_hist = tl.zeros([NE_PADDED], dtype=tl.int64), tl.zeros([EP_PADDED], dtype=tl.int64)

    for tok_start in range(pid * BLOCK, M, num_ctas * BLOCK):
        tok_offs = tok_start + tl.arange(0, BLOCK)
        tok_mask = tok_offs < M

        # per-token visited-rank bitmask: int32 limits ep_size to 32
        visited_ranks = tl.zeros([BLOCK], dtype=tl.int32)

        for j in tl.static_range(TOPK):
            expert_idx = tl.load(topk_ids_ptr + tok_offs * topk_row_stride + j, mask=tok_mask, other=0)

            # masked lanes count into the sentinel bin NUM_EXPERTS
            safe_expert_idx = tl.where(tok_mask, expert_idx, NUM_EXPERTS).to(tl.int32)
            acc_expert_hist += tl.histogram(safe_expert_idx, NE_PADDED).to(tl.int64)

            rank_idx = expert_idx // EXPERTS_PER_RANK
            rank_bit = (1 << rank_idx).to(tl.int32)
            is_visited = (visited_ranks & rank_bit) != 0
            is_first_visit = tok_mask & ~is_visited
            visited_ranks = visited_ranks | rank_bit

            safe_rank_idx = tl.where(is_first_visit, rank_idx, EP_SIZE).to(tl.int32)
            acc_rank_hist += tl.histogram(safe_rank_idx, EP_PADDED).to(tl.int64)

    expert_offs = tl.arange(0, NE_PADDED)
    tl.store(per_cta_expert_hist_ptr + pid * NE_PADDED + expert_offs, acc_expert_hist)
    rank_offs = tl.arange(0, EP_PADDED)
    tl.store(per_cta_rank_hist_ptr + pid * EP_PADDED + rank_offs, acc_rank_hist)


@triton.jit
def reduce_and_prefix_sum_kernel(
    per_cta_expert_hist_ptr,  # [num_ctas * NE_PADDED] int64
    per_cta_rank_hist_ptr,  # [num_ctas * EP_PADDED] int64
    dedup_tokens_per_rank_ptr,  # [EP_SIZE] int64
    tokens_per_ep_rank_ptr,  # [EP_SIZE] int64
    expert_starts_ptr,  # [NUM_EXPERTS] int64
    rank_starts_ptr,  # [EP_SIZE] int64
    dedup_counters_ptr,  # [EP_SIZE] int64, output, zeroed here for kernel 3
    sort_counters_ptr,  # [NUM_EXPERTS] int64, output, zeroed here for kernel 3
    send_meta_ptr,  # [EP_SIZE * (EXPERTS_PER_RANK + 1)] int64
    NUM_CTAS: tl.constexpr,
    NE_PADDED: tl.constexpr,
    EP_PADDED: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """
    Single CTA: reduce the per-CTA histograms into the counts, their exclusive starts and send_meta,
    and zero kernel 3's atomic counters.
    """
    expert_prefix, rank_prefix = tl.zeros([], dtype=tl.int64), tl.zeros([], dtype=tl.int64)
    for rank_idx in tl.static_range(EP_SIZE):
        rank_slot_count = tl.zeros([], dtype=tl.int64)
        for local_expert_idx in tl.static_range(EXPERTS_PER_RANK):
            expert_count = tl.zeros([], dtype=tl.int64)
            for cta_idx in range(NUM_CTAS):
                expert_count += tl.load(per_cta_expert_hist_ptr + cta_idx * NE_PADDED + rank_idx * EXPERTS_PER_RANK + local_expert_idx)
            tl.store(expert_starts_ptr + rank_idx * EXPERTS_PER_RANK + local_expert_idx, expert_prefix)
            expert_prefix += expert_count
            rank_slot_count += expert_count
            tl.store(send_meta_ptr + rank_idx * (EXPERTS_PER_RANK + 1) + local_expert_idx, expert_count)

        tl.store(tokens_per_ep_rank_ptr + rank_idx, rank_slot_count)

        rank_dedup_count = tl.zeros([], dtype=tl.int64)
        for cta_idx in range(NUM_CTAS):
            rank_dedup_count += tl.load(per_cta_rank_hist_ptr + cta_idx * EP_PADDED + rank_idx)
        tl.store(dedup_tokens_per_rank_ptr + rank_idx, rank_dedup_count)
        tl.store(rank_starts_ptr + rank_idx, rank_prefix)
        rank_prefix += rank_dedup_count
        tl.store(send_meta_ptr + rank_idx * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK, rank_dedup_count)

        tl.store(dedup_counters_ptr + rank_idx, tl.zeros([], dtype=tl.int64))

    for expert_idx in tl.static_range(NUM_EXPERTS):
        tl.store(sort_counters_ptr + expert_idx, tl.zeros([], dtype=tl.int64))


@triton.jit
def dedup_scatter_expand_kernel(
    topk_ids_ptr,
    expert_starts_ptr,  # [num_experts] int64 (from kernel 2)
    rank_starts_ptr,  # [ep_size] int64 (from kernel 2)
    dedup_counters_ptr,  # [ep_size] int64, pre-zeroed (atomics)
    sort_counters_ptr,  # [num_experts] int64, pre-zeroed (atomics)
    dispatch_token_idxs_ptr,  # [total_dedup] int64, output
    dedup_local_pos_ptr,  # [M * ep_size] int64, scratch
    idxs_ptr,  # [M * TOPK] int64, output
    expand_idx_ptr,  # [M * TOPK] int64, output
    M,
    topk_row_stride,
    TOPK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Pass 1 places each token once per visited rank in dispatch_token_idxs and records its position
    in dedup_local_pos; pass 2 counting-sorts the slots by expert into idxs and looks up expand_idx
    from dedup_local_pos.
    """
    pid = tl.program_id(0)
    tok_start = pid * BLOCK_M
    tok_offs = tok_start + tl.arange(0, BLOCK_M)
    tok_mask = tok_offs < M

    visited_ranks = tl.zeros([BLOCK_M], dtype=tl.int32)

    for j in tl.static_range(TOPK):
        expert_idx = tl.load(topk_ids_ptr + tok_offs * topk_row_stride + j, mask=tok_mask, other=0)
        rank_idx = expert_idx // EXPERTS_PER_RANK

        rank_bit = (1 << rank_idx).to(tl.int32)
        is_visited = (visited_ranks & rank_bit) != 0
        is_first_visit = tok_mask & ~is_visited
        visited_ranks = visited_ranks | rank_bit

        dedup_pos = tl.atomic_add(dedup_counters_ptr + rank_idx, 1, mask=is_first_visit)
        rank_start = tl.load(rank_starts_ptr + rank_idx, mask=is_first_visit, other=0)
        tl.store(dispatch_token_idxs_ptr + rank_start + dedup_pos, tok_offs, mask=is_first_visit)
        tl.store(dedup_local_pos_ptr + tok_offs * EP_SIZE + rank_idx, dedup_pos, mask=is_first_visit)

    for j in tl.static_range(TOPK):
        expert_idx = tl.load(topk_ids_ptr + tok_offs * topk_row_stride + j, mask=tok_mask, other=0)
        rank_idx = expert_idx // EXPERTS_PER_RANK

        expert_start = tl.load(expert_starts_ptr + expert_idx, mask=tok_mask, other=0)
        sort_pos = tl.atomic_add(sort_counters_ptr + expert_idx, 1, mask=tok_mask)
        sorted_idx = expert_start + sort_pos

        flat_idx = tok_offs * TOPK + j
        tl.store(idxs_ptr + sorted_idx, flat_idx, mask=tok_mask)

        local_pos = tl.load(dedup_local_pos_ptr + tok_offs * EP_SIZE + rank_idx, mask=tok_mask, other=0)
        tl.store(expand_idx_ptr + sorted_idx, local_pos, mask=tok_mask)


# Scratch of ep_dispatch_send_idx_triton, one int64 allocation per shape and stream. Only
# temporaries consumed in stream order before the next call on the same stream live here (and
# send_meta, documented as valid until then); outputs that outlive the call are fresh.
TRITON_SCRATCH: dict[tuple, tuple[torch.Tensor, ...]] = {}
TRITON_SCRATCH_ENTRIES = 8


def triton_scratch(device: torch.device, M: int, num_ctas: int, ne_padded: int, ep_padded: int, num_experts: int, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, ...]:
    key = (device, torch.cuda.current_stream(device).cuda_stream, M, num_experts, ep_size)
    scratch_views = TRITON_SCRATCH.get(key)
    if scratch_views is None:
        sizes = (
            num_ctas * ne_padded,  # per_cta_expert_hist
            num_ctas * ep_padded,  # per_cta_rank_hist
            num_experts,  # expert_starts
            ep_size,  # rank_starts
            ep_size,  # dedup_counters
            num_experts,  # sort_counters
            ep_size * (experts_per_rank + 1),  # send_meta
            M * ep_size,  # dedup_local_pos
        )
        scratch_buffer = torch.empty(sum(sizes), dtype=torch.int64, device=device)
        scratch_views = tuple(scratch_buffer.split(sizes))
        if len(TRITON_SCRATCH) >= TRITON_SCRATCH_ENTRIES:
            TRITON_SCRATCH.clear()
        TRITON_SCRATCH[key] = scratch_views
    return scratch_views


def ep_dispatch_send_idx_triton(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, num_topk = topk_ids.shape
    device = topk_ids.device
    assert ep_size <= 32, f"ep_size={ep_size} exceeds int32 bitmask limit of 32"

    if M == 0:
        send_meta = torch.zeros(ep_size * (experts_per_rank + 1), dtype=torch.int64, device=device)
        if meta_out is None:
            meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=device)
        meta_out[: 2 * ep_size].zero_()
        return (meta_out[:ep_size], meta_out[ep_size : 2 * ep_size], torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device), send_meta)

    # at most 32 CTAs: kernel 2 reduces the per-CTA histograms in one CTA with NUM_CTAS constexpr
    BLOCK = 1024
    num_ctas = min(triton.cdiv(M, BLOCK), 32)
    # tl.histogram needs a power-of-2 bin count; +1 for the sentinel bin of masked-out lanes
    ne_padded, ep_padded = triton.next_power_of_2(num_experts + 1), triton.next_power_of_2(ep_size + 1)

    (per_cta_expert_hist, per_cta_rank_hist, expert_starts, rank_starts, dedup_counters, sort_counters, send_meta, dedup_local_pos) = triton_scratch(device, M, num_ctas, ne_padded, ep_padded, num_experts, ep_size, experts_per_rank)

    dedup_bincount_kernel[(num_ctas,)](topk_ids, per_cta_expert_hist, per_cta_rank_hist, M, topk_ids.stride(0), TOPK=num_topk, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, NUM_EXPERTS=num_experts, NE_PADDED=ne_padded, EP_PADDED=ep_padded, BLOCK=BLOCK)

    if meta_out is None:
        meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=device)
    tokens_per_ep_rank, dedup_tokens_per_gpu = meta_out[:ep_size], meta_out[ep_size : 2 * ep_size]

    reduce_and_prefix_sum_kernel[(1,)](per_cta_expert_hist, per_cta_rank_hist, dedup_tokens_per_gpu, tokens_per_ep_rank, expert_starts, rank_starts, dedup_counters, sort_counters, send_meta, NUM_CTAS=num_ctas, NE_PADDED=ne_padded, EP_PADDED=ep_padded, NUM_EXPERTS=num_experts, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank)

    dispatch_token_idxs = torch.empty(M * ep_size, dtype=torch.int64, device=device)
    idxs = torch.empty(M * num_topk, dtype=torch.int64, device=device)
    expand_idx = torch.empty(M * num_topk, dtype=torch.int64, device=device)

    BLOCK_M = 128
    grid = (triton.cdiv(M, BLOCK_M),)
    dedup_scatter_expand_kernel[grid](topk_ids, expert_starts, rank_starts, dedup_counters, sort_counters, dispatch_token_idxs, dedup_local_pos, idxs, expand_idx, M, topk_ids.stride(0), TOPK=num_topk, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, BLOCK_M=BLOCK_M)

    return (tokens_per_ep_rank, dedup_tokens_per_gpu, dispatch_token_idxs, idxs, expand_idx, send_meta)


@triton.jit
def recv_segment_ends_kernel(
    recv_meta_ptr,  # [EP_SIZE * (EXPERTS_PER_RANK + 1)] int64, input
    seg_end_ptr,  # [EP_SIZE * EXPERTS_PER_RANK] int64, output
    dedup_start_ptr,  # [EP_SIZE] int64, output
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    SEG_PADDED: tl.constexpr,  # next_power_of_2(EP_SIZE * EXPERTS_PER_RANK)
    EP_PADDED: tl.constexpr,  # next_power_of_2(EP_SIZE)
):
    """
    One CTA: the inclusive cumulative slot count of every (sender, expert) segment, sender-major,
    and the exclusive cumulative dedup count of every sender.
    """
    seg = tl.arange(0, SEG_PADDED)
    seg_mask = seg < EP_SIZE * EXPERTS_PER_RANK
    seg_count = tl.load(recv_meta_ptr + seg // EXPERTS_PER_RANK * (EXPERTS_PER_RANK + 1) + seg % EXPERTS_PER_RANK, mask=seg_mask, other=0)
    tl.store(seg_end_ptr + seg, tl.cumsum(seg_count, 0), mask=seg_mask)
    sender = tl.arange(0, EP_PADDED)
    sender_mask = sender < EP_SIZE
    dedup = tl.load(recv_meta_ptr + sender * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK, mask=sender_mask, other=0)
    tl.store(dedup_start_ptr + sender, tl.cumsum(dedup, 0) - dedup, mask=sender_mask)


@triton.jit
def expert_and_expand_idx_kernel(
    seg_end_ptr,  # [NUM_SEGS] int64, input
    dedup_start_ptr,  # [EP_SIZE] int64, input
    received_expand_idx_ptr,  # [total] int64, input
    expert_idxs_ptr,  # [total] int64, output
    expand_idx_ptr,  # [total] int64, output
    total,
    NUM_SEGS: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,  # ceil(log2(NUM_SEGS + 1))
    BLOCK: tl.constexpr,
):
    """
    Per received slot: its segment, the first whose cumulative end exceeds the slot (a binary
    search over seg_end, so the cost and the compile time grow with log2 of the segment count),
    hence its local expert, and its expand_idx plus the dedup counts of the preceding senders.

    total is a kernel argument, not a device-side sum, so the store mask and the caller's allocation
    are the same number (a mismatch would write out of bounds).
    """
    pid = tl.program_id(0)
    slot_offs = pid * BLOCK + tl.arange(0, BLOCK)
    slot_mask = slot_offs < total
    lo, hi = tl.zeros([BLOCK], dtype=tl.int64), tl.full([BLOCK], NUM_SEGS, dtype=tl.int64)
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        seg_end = tl.load(seg_end_ptr + mid, mask=slot_mask & (mid < NUM_SEGS), other=0)
        above = seg_end > slot_offs
        lo, hi = tl.where(above, lo, mid + 1), tl.where(above, mid, hi)
    seg = tl.minimum(lo, NUM_SEGS - 1)
    slot_dedup_start = tl.load(dedup_start_ptr + seg // EXPERTS_PER_RANK, mask=slot_mask, other=0)
    received_idx = tl.load(received_expand_idx_ptr + slot_offs, mask=slot_mask)
    tl.store(expert_idxs_ptr + slot_offs, seg % EXPERTS_PER_RANK, mask=slot_mask)
    tl.store(expand_idx_ptr + slot_offs, received_idx + slot_dedup_start, mask=slot_mask)


def ep_dispatch_recv_idx_triton(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    # the kernels index both inputs with unit stride
    recv_meta, received_expand_idx = recv_meta.contiguous(), received_expand_idx.contiguous()
    total = received_expand_idx.shape[0]
    expert_idxs, expand_idx = torch.empty_like(received_expand_idx), torch.empty_like(received_expand_idx)
    if total == 0:
        return expert_idxs, expand_idx
    num_segs = ep_size * experts_per_rank
    seg_end, dedup_start = torch.empty(num_segs, dtype=torch.int64, device=recv_meta.device), torch.empty(ep_size, dtype=torch.int64, device=recv_meta.device)
    recv_segment_ends_kernel[(1,)](recv_meta, seg_end, dedup_start, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, SEG_PADDED=triton.next_power_of_2(num_segs), EP_PADDED=triton.next_power_of_2(ep_size))
    BLOCK = 1024
    expert_and_expand_idx_kernel[(triton.cdiv(total, BLOCK),)](seg_end, dedup_start, received_expand_idx, expert_idxs, expand_idx, total, NUM_SEGS=num_segs, EXPERTS_PER_RANK=experts_per_rank, SEARCH_STEPS=num_segs.bit_length(), BLOCK=BLOCK)
    return expert_idxs, expand_idx


def ep_dispatch_send_idx_sm90(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    if supports_index(topk_ids, ep_size, experts_per_rank):
        return dispatch_index(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)[:6]
    return ep_dispatch_send_idx_triton(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)


def ep_dispatch_recv_idx_sm90(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if supports_expand(received_expand_idx, ep_size, experts_per_rank):
        return expand_index(recv_meta, received_expand_idx, ep_size, experts_per_rank)
    return ep_dispatch_recv_idx_triton(recv_meta, received_expand_idx, ep_size, experts_per_rank)
