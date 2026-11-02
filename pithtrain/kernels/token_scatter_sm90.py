"""
SM90 CuTe-DSL scatter_by_expert in two launches:

1. GroupRanks (one CTA per 2048 tokens) ranks each token stably within its expert (token order) and,
   in the last chunk, writes the per-group counts, the padded sizes, grouped_mm_offs and the start of
   each group's padding in the list of zero rows. Chunks are ordered without a grid barrier: each CTA
   takes a ticket (chunk index in dispatch order), publishes its chunk histogram behind a release
   flag, and waits only for lower chunks, which already hold their tickets and never wait on it, so
   forward progress needs no co-residency. Any number of groups: the per-chunk counts live in shared
   memory up to COUNTS_SMEM_BYTES, else in a per-CTA slice of global scratch, read the same way
   through generic addresses.
2. RowScatter, launched as a programmatic dependent (PDL) of GroupRanks, moves one row per warp:
   each warp takes a contiguous range of the token rows, fixed before the prep results are known so
   that its first row's loads are issued while GroupRanks still runs, and a contiguous range of the
   zero rows (every group's padding, then the allocation tail up to a multiple of 1024 rows), so
   every output row has exactly one writer. Rows are moved as bytes with the widest vector (16 / 8 /
   4 / 2 / 1 bytes) that divides the row bytes, the input row stride and the input address, so one
   compiled kernel per width serves every dtype.

Placement is stable, so the output is deterministic.
"""

from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, atom_add_u64, byte_sum, cvta_shared, fence_gpu, lanemask_lt, ld_b32, ld_generic_b32, ld_generic_v4, ld_relaxed, ld_u8, ld_u16, ld_v2, ld_v4, lds_b32, match_any, red_add_generic, st_b32, st_b64, st_generic_b32, st_generic_u8, st_generic_v4, st_u8, st_u16, st_v2, st_v4, sts_b32
from pithtrain.kernels.host_transfer import get_pinned_buffer

__all__ = ["GEMM_ALLOC_ALIGNMENT_SM90", "GroupRanks", "RowScatter", "scatter_by_expert_sm90"]

GEMM_ALLOC_ALIGNMENT_SM90 = 1024
PREP_THREADS = 512
PREP_WARPS = PREP_THREADS // 32
PREP_CHUNK = 2048  # tokens per GroupRanks CTA
SLOTS_PER_THREAD = PREP_CHUNK // PREP_THREADS
COUNT_BYTES_PER_GROUP = 24  # per-warp byte counts of one round (16), running count (4), lower chunks' count (4)
COUNTS_SMEM_BYTES = 32 * 1024  # up to 1024 groups; static shared memory is capped at 48 KiB
ROW_THREADS = 256
ROW_WARPS = ROW_THREADS // 32
ROW_CTAS_PER_SM = 8
ROW_UNROLL = 16  # vectors in flight per lane


def meta_layout(G: int) -> tuple:
    """
    Element offsets in the int32 meta allocation: grouped_mm_offs (G) | ks (G) | counts (G) | zero
    starts (G + 1: exclusive prefix sum of the padding rows per group, then their total) | ranks.
    """
    return 0, G, 2 * G, 3 * G, 4 * G + 1


class GroupRanks:
    """
    The prep kernel. group_cap: compile-time capacity of the shared-memory counts (0 when they live
    in global scratch); the group count and the padding alignment are runtime arguments.
    """

    def __init__(self, group_cap: int) -> None:
        self.group_cap = group_cap

    @cute.jit
    def __call__(self, mE: cute.Tensor, mMeta: cute.Tensor, mFlags: cute.Tensor, mHist: cute.Tensor, mCtr: cute.Tensor, mCounts: cute.Tensor, num_groups: Int32, alignment: Int32, stream) -> None:
        T = mE.shape[0]
        num_chunks = (T + PREP_CHUNK - 1) // PREP_CHUNK
        self.kernel(mE.iterator.toint(), mMeta.iterator.toint(), mFlags.iterator.toint(), mHist.iterator.toint(), mCtr.iterator.toint(), mCounts.iterator.toint(), Int32(T), num_groups, alignment).launch(grid=[num_chunks, 1, 1], block=[PREP_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, experts_base: Int64, meta_base: Int64, flags_base: Int64, hist_base: Int64, counter_base: Int64, counts_global_base: Int64, T: Int32, G: Int32, alignment: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp, lane = tidx // 32, tidx % 32
        num_chunks = (T + PREP_CHUNK - 1) // PREP_CHUNK
        group_stride = (G + 3) // 4 * 4
        smem = cutlass.memory.SmemAllocator()
        s_ticket = smem.allocate(16, byte_alignment=16).toint()
        s_scan = smem.allocate(2 * PREP_WARPS * 4, byte_alignment=16).toint()
        counts_base = counts_global_base + Int64(block_idx) * (COUNT_BYTES_PER_GROUP * group_stride)
        if cutlass.const_expr(self.group_cap > 0):
            counts_base = cvta_shared(smem.allocate(COUNT_BYTES_PER_GROUP * self.group_cap, byte_alignment=16).toint())
        warp_counts_base, run_base, lower_base = counts_base, counts_base + 16 * group_stride, counts_base + 20 * group_stride
        cute.arch.griddepcontrol_launch_dependents()  # let the row kernel's CTAs get resident

        self.clear_counts(counts_base, group_stride, tidx)
        chunk_idx, epoch = self.take_ticket(tidx, counter_base, s_ticket)
        expert_idxs, token_oks = self.load_chunk(experts_base, chunk_idx, tidx, T)
        safe_expert_idxs, chunk_ranks = self.rank_in_chunk(expert_idxs, token_oks, warp, warp_counts_base, run_base)
        self.publish_histogram(hist_base, flags_base, chunk_idx, epoch, tidx, G, run_base)
        self.wait_lower_chunks(flags_base, chunk_idx, epoch, tidx)
        self.sum_lower_chunks(hist_base, chunk_idx, tidx, G, lower_base)
        rank_base = meta_base + Int64(4 * G + 1) * 4
        for i in cutlass.range_constexpr(SLOTS_PER_THREAD):
            token_idx = chunk_idx * PREP_CHUNK + i * PREP_THREADS + tidx
            rank = ld_generic_b32(lower_base + Int64(safe_expert_idxs[i]) * 4) + chunk_ranks[i]
            st_b32(rank_base + Int64(token_idx) * 4, Uint32(rank), token_oks[i])
        if chunk_idx == num_chunks - 1:
            self.store_group_offsets(meta_base, counter_base, num_chunks, tidx, warp, lane, G, alignment, run_base, lower_base, s_scan)

    @cute.jit
    def clear_counts(self, counts_base, group_stride, tidx) -> None:
        num_vectors = COUNT_BYTES_PER_GROUP * group_stride // 16
        zeros = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        for i in cutlass.range((num_vectors + PREP_THREADS - 1) // PREP_THREADS, unroll=1):
            vector_idx = i * PREP_THREADS + tidx
            st_generic_v4(counts_base + Int64(vector_idx) * 16, zeros, Int32(vector_idx < num_vectors))

    @cute.jit
    def take_ticket(self, tidx, counter_base, s_ticket):
        if tidx == 0:
            # the counter packs (epoch << 32) | tickets taken; the old value gives this CTA both
            counter = atom_add_u64(counter_base, Int64(1))
            sts_b32(s_ticket, Uint32(Int32(counter & 0xFFFFFFFF)), Int32(1))
            sts_b32(s_ticket + 4, Uint32(Int32(counter >> 32)), Int32(1))
        cute.arch.sync_threads()
        chunk_idx, epoch = Int32(lds_b32(s_ticket)), Int32(lds_b32(s_ticket + 4))
        return chunk_idx, epoch

    @cute.jit
    def load_chunk(self, experts_base, chunk_idx, tidx, T):
        expert_idxs, token_oks = [], []
        for i in cutlass.range_constexpr(SLOTS_PER_THREAD):
            token_idx = chunk_idx * PREP_CHUNK + i * PREP_THREADS + tidx
            token_ok = Int32(token_idx < T)
            expert_idxs.append(ld_b32(experts_base + Int64(token_idx) * 8, token_ok, Int32(-1)))
            token_oks.append(token_ok)
        return expert_idxs, token_oks

    @cute.jit
    def rank_in_chunk(self, expert_idxs, token_oks, warp, warp_counts_base, run_base):
        """
        Stable rank of each token within its expert inside this chunk, and its expert index (0 for an
        invalid slot); leaves the chunk's per-expert counts in the running counts. Each round of 512
        tokens, the leader lane of each expert in each warp writes the warp's count of that expert as
        one byte of the expert's 16-byte row (one byte per warp); a token's rank is the running count
        plus the bytes of the lower warps plus its lower peers, and the highest warp holding the expert
        then folds the row into the running count and clears it.
        """
        lower_lanes_mask = lanemask_lt()
        safe_expert_idxs, chunk_ranks = [], []
        for i in cutlass.range_constexpr(SLOTS_PER_THREAD):
            expert_idx, token_ok = expert_idxs[i], token_oks[i]
            peers = match_any(expert_idx)
            num_lower_peers, num_peers = cute.arch.popc(peers & lower_lanes_mask), cute.arch.popc(peers)
            safe_expert_idx = expert_idx * token_ok  # safe index for invalid lanes (their stores are masked)
            row_addr = warp_counts_base + Int64(safe_expert_idx) * 16
            is_leader = token_ok & Int32(num_lower_peers == 0)
            st_generic_u8(row_addr + warp, Uint32(num_peers), is_leader)
            cute.arch.sync_threads()
            words = ld_generic_v4(row_addr)
            running = ld_generic_b32(run_base + Int64(safe_expert_idx) * 4)
            num_lower_warps, num_total = Int32(0), Int32(0)
            for k in cutlass.range_constexpr(4):
                # bytes of warps 4k .. 4k + 3; keep those below this warp
                num_kept = cutlass.max(cutlass.min(warp - 4 * k, 4), 0)
                mask = Uint32((Int64(1) << Int64(num_kept * 8)) - 1)
                num_lower_warps, num_total = num_lower_warps + byte_sum(words[k] & mask), num_total + byte_sum(words[k])
            chunk_ranks.append(running + num_lower_warps + num_lower_peers)
            safe_expert_idxs.append(safe_expert_idx)
            cute.arch.sync_threads()
            is_owner = is_leader & Int32(num_lower_warps + num_peers == num_total)
            st_generic_b32(run_base + Int64(safe_expert_idx) * 4, Uint32(running + num_total), is_owner)
            st_generic_v4(row_addr, (Uint32(0), Uint32(0), Uint32(0), Uint32(0)), is_owner)
            cute.arch.sync_threads()
        return safe_expert_idxs, chunk_ranks

    @cute.jit
    def publish_histogram(self, hist_base, flags_base, chunk_idx, epoch, tidx, G, run_base) -> None:
        for i in cutlass.range((G + PREP_THREADS - 1) // PREP_THREADS, unroll=1):
            group_idx = i * PREP_THREADS + tidx
            group_ok = Int32(group_idx < G)
            count = ld_generic_b32(run_base + Int64(group_idx * group_ok) * 4)
            st_b32(hist_base + (Int64(chunk_idx) * G + group_idx) * 4, Uint32(count), group_ok)
        # every thread's histogram store is ordered before thread 0's release of the flag
        fence_gpu()
        cute.arch.sync_threads()
        if tidx == 0:
            asm(None, [Int64(flags_base + Int64(chunk_idx) * 4).ir_value(), Int32(epoch).ir_value()], "st.release.gpu.global.b32 [$0], $1;", "l,r", side_effects=True)

    @cute.jit
    def wait_lower_chunks(self, flags_base, chunk_idx, epoch, tidx) -> None:
        for round_idx in cutlass.range((chunk_idx + PREP_THREADS - 1) // PREP_THREADS, unroll=1):
            lower_idx = round_idx * PREP_THREADS + tidx
            # for lower_idx < chunk_idx: spin with acquire loads until that chunk's flag reaches this epoch
            asm(
                None, [Int64(flags_base + Int64(lower_idx) * 4).ir_value(), Int32(epoch).ir_value(), Int32(lower_idx < chunk_idx).ir_value()],
                "{\n\t.reg .pred p, q;\n\t.reg .b32 t;\n\t"
                "setp.eq.b32 q, $2, 0;\n\t"
                "@q bra TS_DONE_${:uid};\n"
                "TS_LOOP_${:uid}:\n\t"
                "ld.acquire.gpu.global.b32 t, [$0];\n\t"
                "setp.ne.b32 p, t, $1;\n\t"
                "@p bra TS_LOOP_${:uid};\n"
                "TS_DONE_${:uid}:\n\t}",
                "l,r,r", side_effects=True,
            )
        cute.arch.sync_threads()
        fence_gpu()

    @cute.jit
    def sum_lower_chunks(self, hist_base, chunk_idx, tidx, G, lower_base) -> None:
        num_lower_entries = Int64(chunk_idx) * G
        for round_idx in cutlass.range((num_lower_entries + PREP_THREADS - 1) // PREP_THREADS, unroll=1):
            entry_idx = round_idx * PREP_THREADS + tidx
            entry_ok = Int32(entry_idx < num_lower_entries)
            count = ld_relaxed(hist_base + entry_idx * 4, entry_ok)
            red_add_generic(lower_base + (entry_idx % G) * 4, count, entry_ok)
        cute.arch.sync_threads()

    @cute.jit
    def warp_inclusive_sum(self, value, lane):
        for i in cutlass.range_constexpr(5):
            peer = cute.arch.shuffle_sync_up(value, offset=1 << i)
            value = value + peer * Int32(lane >= (1 << i))
        return value

    @cute.jit
    def store_group_offsets(self, meta_base, counter_base, num_chunks, tidx, warp, lane, G, alignment, run_base, lower_base, s_scan) -> None:
        """
        Run by the last chunk, which has waited for all others: a block-wide scan over the groups in
        tiles of PREP_THREADS, carrying the padded rows and the padding rows of the tiles before.
        """
        offs_base, ks_base, counts_base, zero_starts_base = meta_base, meta_base + Int64(G) * 4, meta_base + Int64(G) * 8, meta_base + Int64(G) * 12
        rows_carry, pad_carry = Int32(0), Int32(0)
        for tile_idx in cutlass.range((G + PREP_THREADS - 1) // PREP_THREADS, unroll=1):
            group_idx = tile_idx * PREP_THREADS + tidx
            group_ok = Int32(group_idx < G)
            safe_group_idx = Int64(group_idx * group_ok)
            count = (ld_generic_b32(lower_base + safe_group_idx * 4) + ld_generic_b32(run_base + safe_group_idx * 4)) * group_ok
            padded = (count + alignment - 1) // alignment * alignment
            pad = padded - count
            rows_incl, pad_incl = self.warp_inclusive_sum(padded, lane), self.warp_inclusive_sum(pad, lane)
            sts_b32(s_scan + warp * 4, Uint32(rows_incl), Int32(lane == 31))
            sts_b32(s_scan + (PREP_WARPS + warp) * 4, Uint32(pad_incl), Int32(lane == 31))
            cute.arch.sync_threads()
            rows_lower, pad_lower, rows_total, pad_total = Int32(0), Int32(0), Int32(0), Int32(0)
            for w in cutlass.range_constexpr(PREP_WARPS):
                warp_rows, warp_pad = Int32(lds_b32(s_scan + w * 4)), Int32(lds_b32(s_scan + (PREP_WARPS + w) * 4))
                is_lower = Int32(w < warp)
                rows_lower, pad_lower, rows_total, pad_total = rows_lower + warp_rows * is_lower, pad_lower + warp_pad * is_lower, rows_total + warp_rows, pad_total + warp_pad
            st_b32(offs_base + Int64(group_idx) * 4, Uint32(rows_carry + rows_lower + rows_incl), group_ok)
            st_b32(ks_base + Int64(group_idx) * 4, Uint32(padded), group_ok)
            st_b32(counts_base + Int64(group_idx) * 4, Uint32(count), group_ok)
            st_b32(zero_starts_base + Int64(group_idx) * 4, Uint32(pad_carry + pad_lower + pad_incl - pad), group_ok)
            rows_carry, pad_carry = rows_carry + rows_total, pad_carry + pad_total
            cute.arch.sync_threads()
        if tidx == 0:
            st_b32(zero_starts_base + Int64(G) * 4, Uint32(pad_carry), Int32(1))
            # every CTA holds its ticket (this one took the last): next epoch, zero tickets
            atom_add_u64(counter_base, (Int64(1) << 32) - Int64(num_chunks))


class RowScatter:
    """
    The row kernel for one vector width (bytes). With PDL its CTAs become resident while GroupRanks
    runs and wait in griddepcontrol.wait for its results.
    """

    def __init__(self, vec: int) -> None:
        self.vec = vec

    @cute.jit
    def __call__(self, mX: cute.Tensor, mE: cute.Tensor, mMeta: cute.Tensor, mRsi: cute.Tensor, mOut: cute.Tensor, num_groups: Int32, num_ctas: Int32, search_steps: Int32, stream) -> None:
        args = (mX.iterator.toint(), Int64(mX.stride[0]), Int32(mX.shape[1]), mE.iterator.toint(), mMeta.iterator.toint(), mRsi.iterator.toint(), mOut.iterator.toint(), Int32(mE.shape[0]), num_groups, search_steps)
        self.kernel(*args).launch(grid=[num_ctas, 1, 1], block=[ROW_THREADS, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, x_base: Int64, x_row_stride: Int64, row_bytes: Int32, experts_base: Int64, meta_base: Int64, rsi_base: Int64, out_base: Int64, T: Int32, G: Int32, search_steps: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        lane = tidx % 32
        warp_idx, num_warps = block_idx * ROW_WARPS + tidx // 32, num_ctas * ROW_WARPS
        offs_base, ks_base, counts_base, zero_starts_base = meta_base, meta_base + Int64(G) * 4, meta_base + Int64(G) * 8, meta_base + Int64(G) * 12
        rank_base = meta_base + Int64(4 * G + 1) * 4
        # token rows split before the prep results are known, so the first row's loads overlap it
        tokens_per_warp = (T + num_warps - 1) // num_warps
        token_start = cutlass.min(warp_idx * tokens_per_warp, T)
        token_end = cutlass.min(token_start + tokens_per_warp, T)
        first_vectors = self.load_segment(x_base + Int64(token_start) * x_row_stride, Int32(0), row_bytes, lane, Int32(token_start < token_end))
        cute.arch.griddepcontrol_wait()
        rows_end, num_pad_rows = ld_relaxed(offs_base + Int64(G - 1) * 4, Int32(1)), ld_relaxed(zero_starts_base + Int64(G) * 4, Int32(1))
        num_tail_rows = (rows_end + GEMM_ALLOC_ALIGNMENT_SM90 - 1) // GEMM_ALLOC_ALIGNMENT_SM90 * GEMM_ALLOC_ALIGNMENT_SM90 - rows_end
        num_zero_rows = num_pad_rows + num_tail_rows
        zeros_per_warp = (num_zero_rows + num_warps - 1) // num_warps
        zero_start = cutlass.min(warp_idx * zeros_per_warp, num_zero_rows)
        zero_end = cutlass.min(zero_start + zeros_per_warp, num_zero_rows)
        if token_start < token_end:
            self.copy_row(x_base, x_row_stride, row_bytes, experts_base, offs_base, rank_base, rsi_base, out_base, token_start, lane, first_vectors)
        for token_idx in cutlass.range(token_start + 1, token_end, unroll=1):
            self.copy_row(x_base, x_row_stride, row_bytes, experts_base, offs_base, rank_base, rsi_base, out_base, token_idx, lane, None)
        for zero_idx in cutlass.range(zero_start, zero_end, unroll=1):
            out_row = rows_end + (zero_idx - num_pad_rows)
            if zero_idx < num_pad_rows:
                group_idx = self.find_pad_group(zero_starts_base, zero_idx, G, search_steps)
                group_start = ld_relaxed(offs_base + Int64(group_idx) * 4, Int32(1)) - ld_relaxed(ks_base + Int64(group_idx) * 4, Int32(1))
                out_row = group_start + ld_relaxed(counts_base + Int64(group_idx) * 4, Int32(1)) + (zero_idx - ld_relaxed(zero_starts_base + Int64(group_idx) * 4, Int32(1)))
            self.zero_row(out_base + Int64(out_row) * row_bytes, row_bytes, lane)

    @cute.jit
    def find_pad_group(self, zero_starts_base, zero_idx, G, search_steps):
        """
        The largest group index g with zero_starts[g] <= zero_idx (zero_starts is non-decreasing and
        zero_idx is below its last entry, so g < G and zero_idx is in group g's padding).
        """
        group_idx = Int32(0)
        for i in cutlass.range(search_steps, unroll=1):
            probe_idx = group_idx + (Int32(1) << (search_steps - 1 - i))
            probe_ok = Int32(probe_idx <= G)
            probe_start = ld_relaxed(zero_starts_base + Int64(probe_idx * probe_ok) * 4, probe_ok)
            group_idx = group_idx + (probe_idx - group_idx) * (probe_ok & Int32(probe_start <= zero_idx))
        return group_idx

    @cute.jit
    def load_vec(self, addr, pred):
        vec = self.vec
        if cutlass.const_expr(vec == 16):
            return ld_v4(addr, pred)
        elif cutlass.const_expr(vec == 8):
            return ld_v2(addr, pred)
        elif cutlass.const_expr(vec == 4):
            return (Uint32(ld_b32(addr, pred, Int32(0))),)
        elif cutlass.const_expr(vec == 2):
            return (ld_u16(addr, pred),)
        else:
            return (ld_u8(addr, pred),)

    @cute.jit
    def store_vec(self, addr, words, pred) -> None:
        vec = self.vec
        if cutlass.const_expr(vec == 16):
            st_v4(addr, words, pred)
        elif cutlass.const_expr(vec == 8):
            st_v2(addr, words[0], words[1], pred)
        elif cutlass.const_expr(vec == 4):
            st_b32(addr, words[0], pred)
        elif cutlass.const_expr(vec == 2):
            st_u16(addr, words[0], pred)
        else:
            st_u8(addr, words[0], pred)

    @cute.jit
    def load_segment(self, src_base, segment_idx, row_bytes, lane, pred):
        vec = self.vec
        segment_bytes = 32 * ROW_UNROLL * vec
        vectors = []
        for u in cutlass.range_constexpr(ROW_UNROLL):
            byte_idx = segment_idx * segment_bytes + (u * 32 + lane) * vec
            vectors.append(self.load_vec(src_base + byte_idx, pred & Int32(byte_idx < row_bytes)))
        return vectors

    @cute.jit
    def store_segment(self, dst_base, segment_idx, row_bytes, lane, vectors) -> None:
        vec = self.vec
        segment_bytes = 32 * ROW_UNROLL * vec
        for u in cutlass.range_constexpr(ROW_UNROLL):
            byte_idx = segment_idx * segment_bytes + (u * 32 + lane) * vec
            self.store_vec(dst_base + byte_idx, vectors[u], Int32(byte_idx < row_bytes))

    @cute.jit
    def copy_row(self, x_base, x_row_stride, row_bytes, experts_base, offs_base, rank_base, rsi_base, out_base, token_idx, lane, first_vectors) -> None:
        """
        first_vectors: the row's first segment if already loaded, else None.
        """
        segment_bytes = 32 * ROW_UNROLL * self.vec
        expert_idx = ld_b32(experts_base + Int64(token_idx) * 8, Int32(1), Int32(0))
        group_start = ld_relaxed(offs_base + Int64(expert_idx - 1) * 4, Int32(expert_idx > 0))
        out_row = Int64(group_start + ld_relaxed(rank_base + Int64(token_idx) * 4, Int32(1)))
        st_b64(rsi_base + Int64(token_idx) * 8, out_row, Int32(lane == 0))
        src_base, dst_base = x_base + Int64(token_idx) * x_row_stride, out_base + out_row * row_bytes
        first_segment = 0
        if cutlass.const_expr(first_vectors is not None):
            self.store_segment(dst_base, Int32(0), row_bytes, lane, first_vectors)
            first_segment = 1
        for segment_idx in cutlass.range(first_segment, (row_bytes + segment_bytes - 1) // segment_bytes, unroll=1):
            self.store_segment(dst_base, segment_idx, row_bytes, lane, self.load_segment(src_base, segment_idx, row_bytes, lane, Int32(1)))

    @cute.jit
    def zero_row(self, dst_base, row_bytes, lane) -> None:
        vec = self.vec
        segment_bytes = 32 * ROW_UNROLL * vec
        zeros = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        for segment_idx in cutlass.range((row_bytes + segment_bytes - 1) // segment_bytes, unroll=1):
            for u in cutlass.range_constexpr(ROW_UNROLL):
                byte_idx = segment_idx * segment_bytes + (u * 32 + lane) * vec
                self.store_vec(dst_base + byte_idx, zeros, Int32(byte_idx < row_bytes))


class Scratch:
    """
    GroupRanks scratch of one (device, stream): a ticket counter, epoch flags, chunk histograms and,
    for group counts beyond shared memory, per-CTA count slices. The counter packs (epoch << 32) |
    tickets; the last chunk of each launch advances the epoch and zeroes the tickets, and a chunk's
    flag holds the epoch it was published in. Launches on one stream are ordered, and each stream
    has its own scratch, so concurrent streams never share it.
    """

    def __init__(self, device: torch.device, num_chunks: int, num_groups: int, count_words: int) -> None:
        self.num_chunks, self.num_groups, self.count_words = num_chunks, num_groups, count_words
        self.counter = torch.full((1,), 1 << 32, dtype=torch.int64, device=device)  # epoch 1
        self.flags = torch.zeros(num_chunks, dtype=torch.int32, device=device)
        self.hist = torch.empty(num_chunks * num_groups, dtype=torch.int32, device=device)
        self.counts = torch.empty(max(count_words, 4), dtype=torch.int32, device=device)


SCRATCH: dict = {}
PREP: dict = {}
ROWS: dict = {}
copy_stream: Optional[torch.cuda.Stream] = None
copy_event: Optional[torch.cuda.Event] = None


def get_scratch(device: torch.device, num_chunks: int, num_groups: int, count_words: int) -> Scratch:
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    scratch = SCRATCH.get(key)
    if scratch is None or scratch.num_chunks < num_chunks or scratch.num_groups < num_groups or scratch.count_words < count_words:
        if scratch is not None:
            num_chunks, num_groups, count_words = max(num_chunks, scratch.num_chunks), max(num_groups, scratch.num_groups), max(count_words, scratch.count_words)
        scratch = SCRATCH[key] = Scratch(device, max(num_chunks, 64), num_groups, count_words)
    return scratch


def compile_prep(group_cap: int) -> Callable:
    int32_vector = lambda: fake(Int32, (cute.sym_int(),), (1,), 4)
    return cute.compile(GroupRanks(group_cap), fake(Int64, (cute.sym_int(),), (1,), 8), int32_vector(), int32_vector(), int32_vector(), fake(Int64, (1,), (1,), 8), fake(Int32, (cute.sym_int(),), (1,), 16), Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def compile_rows(vec: int) -> Callable:
    x = fake(cutlass.Uint8, (cute.sym_int(), cute.sym_int()), (cute.sym_int64(divisibility=vec), 1), vec)
    out = fake(cutlass.Uint8, (cute.sym_int(), cute.sym_int()), (cute.sym_int64(divisibility=vec), 1), 16)
    return cute.compile(RowScatter(vec), x, fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int32, (cute.sym_int(),), (1,), 4), fake(Int64, (cute.sym_int(),), (1,), 8), out, Int32(0), Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def vector_width(*sizes: int) -> int:
    """
    The widest of 16 / 8 / 4 / 2 / 1 bytes that divides every size.
    """
    width = 16
    while any(size % width for size in sizes):
        width //= 2
    return width


def scatter_by_expert_sm90(input_tokens: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, padding_alignment: int = 128, group_counts: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Placement within a group is stable (token order). The allocation holds exactly
    roundup(sum(ks), 1024) rows when group_counts is given, else an upper bound. Row offsets are
    int32: num_tokens + num_groups * (padding_alignment - 1), rounded up to 1024 rows, at or
    above 2^31 raises ValueError.
    """
    global copy_stream, copy_event
    num_tokens, hidden_size = input_tokens.shape[0], input_tokens.shape[1]
    device = input_tokens.device
    if expert_idxs.shape[0] != num_tokens:
        raise ValueError(f"{expert_idxs.shape[0]=} must equal the input row count {num_tokens=}")
    if num_tokens == 0:
        output_tokens = torch.empty((0, hidden_size), device=device, dtype=input_tokens.dtype)
        reverse_shuffle_idxs = torch.empty((0,), device=device, dtype=torch.int64)
        grouped_mm_offs = torch.empty((0,), device=device, dtype=torch.int32)
        ks_tensor = torch.empty((0,), device=device, dtype=torch.int32)
        return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, []

    G = num_groups
    # row offsets are int32 on the device, up to the allocation rounded to 1024 rows: its bound must fit
    rows_bound = (num_tokens + G * (padding_alignment - 1) + GEMM_ALLOC_ALIGNMENT_SM90 - 1) // GEMM_ALLOC_ALIGNMENT_SM90 * GEMM_ALLOC_ALIGNMENT_SM90
    if rows_bound >= 1 << 31:
        raise ValueError(f"{num_tokens=} rows with {G=} groups padded to {padding_alignment=} may exceed int32 row offsets")
    if group_counts is not None:
        ks = [(count + padding_alignment - 1) // padding_alignment * padding_alignment for count in group_counts]
        max_rows = (sum(ks) + GEMM_ALLOC_ALIGNMENT_SM90 - 1) // GEMM_ALLOC_ALIGNMENT_SM90 * GEMM_ALLOC_ALIGNMENT_SM90
    else:
        ks = None
        max_rows = num_tokens + G * (padding_alignment - 1)
        max_rows = (max_rows + GEMM_ALLOC_ALIGNMENT_SM90 - 1) // GEMM_ALLOC_ALIGNMENT_SM90 * GEMM_ALLOC_ALIGNMENT_SM90
    output_tokens = torch.empty((max_rows, hidden_size), device=device, dtype=input_tokens.dtype)
    reverse_shuffle_idxs = torch.empty((num_tokens,), device=device, dtype=torch.int64)
    offs_at, ks_at, _, _, rank_at = meta_layout(G)
    meta = torch.empty(rank_at + num_tokens, dtype=torch.int32, device=device)
    grouped_mm_offs, ks_tensor = meta[offs_at : offs_at + G], meta[ks_at : ks_at + G]

    num_chunks = (num_tokens + PREP_CHUNK - 1) // PREP_CHUNK
    group_stride = (G + 3) // 4 * 4
    # shared-memory capacity rounded up to a power of 2 (fewer compiled variants), 0 for global counts
    group_cap = 1 << (group_stride - 1).bit_length()
    in_smem = COUNT_BYTES_PER_GROUP * group_cap <= COUNTS_SMEM_BYTES
    group_cap = group_cap if in_smem else 0
    scratch = get_scratch(device, num_chunks, G, 0 if in_smem else num_chunks * COUNT_BYTES_PER_GROUP * group_stride // 4)
    prep_fn = PREP.get(group_cap)
    if prep_fn is None:
        prep_fn = PREP[group_cap] = compile_prep(group_cap)
    if ks is None and copy_stream is None:
        copy_stream, copy_event = torch.cuda.Stream(device=device), torch.cuda.Event()
    # everything the row launch needs is ready before the prep launch, so the two launches go back to back
    elem_bytes = input_tokens.element_size()
    x_bytes = input_tokens.view(torch.uint8)
    vec = vector_width(hidden_size * elem_bytes, input_tokens.stride(0) * elem_bytes, input_tokens.data_ptr())
    rows_fn = ROWS.get(vec)
    if rows_fn is None:
        rows_fn = ROWS[vec] = compile_rows(vec)
    num_ctas = max(1, min(SM_COUNT * ROW_CTAS_PER_SM, (max_rows + ROW_WARPS - 1) // ROW_WARPS))
    out_bytes = output_tokens.view(torch.uint8)
    prep_fn(expert_idxs, meta, scratch.flags, scratch.hist, scratch.counter, scratch.counts, G, padding_alignment)
    if ks is None:
        copy_event.record()  # ks_tensor is final after the prep
    rows_fn(x_bytes, expert_idxs, meta, reverse_shuffle_idxs, out_bytes, G, num_ctas, G.bit_length())

    if ks is None:
        ks_host = get_pinned_buffer("scatter_by_expert_ks", G, torch.int32)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(copy_event)
            ks_host.copy_(ks_tensor, non_blocking=True)
        copy_stream.synchronize()
        ks = ks_host.tolist()
    # trimmed to sum(ks) exactly: DeepGEMM's k_grouped_fp8_gemm_tn_contiguous asserts it
    output_tokens = output_tokens[: sum(ks)]
    return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks
