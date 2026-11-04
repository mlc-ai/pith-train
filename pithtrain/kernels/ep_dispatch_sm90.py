"""
CuTe-DSL kernels of the ep_dispatch feature (SM90), launched through TVM-FFI: dispatch_index, the
sender side in one cooperative launch with stable (token-order) placement, and expand_index, the
receiver side. Inputs they do not cover (supports_index / supports_expand) take the Triton
kernels of ep_dispatch_sm100.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, atom_add_u64, fence_gpu, lanemask_lt, ld_b32, ld_relaxed, lds_b32, red_add_shared, red_or_shared, st_b32, st_b64, sts_b32
from pithtrain.kernels.ep_dispatch_sm100 import ep_dispatch_recv_idx_sm100, ep_dispatch_send_idx_sm100

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


def ep_dispatch_send_idx_sm90(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    if supports_index(topk_ids, ep_size, experts_per_rank):
        return dispatch_index(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)[:6]
    return ep_dispatch_send_idx_sm100(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)


def ep_dispatch_recv_idx_sm90(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if supports_expand(received_expand_idx, ep_size, experts_per_rank):
        return expand_index(recv_meta, received_expand_idx, ep_size, experts_per_rank)
    return ep_dispatch_recv_idx_sm100(recv_meta, received_expand_idx, ep_size, experts_per_rank)
