"""
The grouping is two launches with no host sync: a per-chunk expert histogram, then one CTA per chunk
of slots that ranks its slots stably within their expert (warp match_any votes) and derives each
expert's base row from all the chunk histograms, so CTAs never wait for each other. With expand_idx
it also fills, per token, a bitmask of its local experts and its grouped row in each, which the
gather-sum walks in expert order.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cutlass_dsl import T

from pithtrain.kernels.bf16_expert_support import MAX_GROUPS
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, bf16x2_to_f32, f32x2_to_bf16x2, lanemask_lt, ld_b32, ld_v4, lds_b32, match_any, red_add_shared, st_b32, st_b64, st_v4, sts_b32

__all__ = ["SlotMaps", "bf16_expert_group_slots_sm90", "bf16_expert_gather_rows_sm90", "bf16_expert_gather_sum_sm90", "bf16_expert_segment_sum_sm90"]

THREADS = 512
WARPS = THREADS // 32
CHUNK = 2048  # slots per grouping CTA

SUM_THREADS = 128
SUM_TOKENS = 4  # tokens per gather-sum CTA


class ChunkHistogram:
    """
    Expert histogram of each CHUNK-slot chunk into hist[chunk * G + expert]; also zeroes the token
    bitmask the grouping launch ORs into.
    """

    def __init__(self, num_groups: int) -> None:
        self.G = num_groups

    @cute.jit
    def __call__(self, mExpertIdx, mHist, mMask, stream) -> None:
        num_chunks = (mExpertIdx.shape[0] + CHUNK - 1) // CHUNK
        self.kernel(mExpertIdx, mHist, mMask).launch(grid=[num_chunks, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mExpertIdx, mHist, mMask) -> None:
        G = self.G
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, _, _ = cute.arch.block_idx()
        num_slots = mExpertIdx.shape[0]
        num_chunks = (num_slots + CHUNK - 1) // CHUNK
        num_tokens = mMask.shape[0]
        for round_idx in cutlass.range((num_tokens + num_chunks * THREADS - 1) // (num_chunks * THREADS), unroll=1):
            token_idx = (round_idx * num_chunks + chunk_idx) * THREADS + tidx
            st_b64(mMask.iterator.toint() + Int64(token_idx) * 8, Int64(0), Int32(token_idx < num_tokens))
        smem = cutlass.memory.SmemAllocator()
        s_hist = smem.allocate(G * 4, byte_alignment=16).toint()
        sts_b32(s_hist + tidx * 4, Uint32(0), Int32(tidx < G))
        cute.arch.sync_threads()
        # one shared add per distinct expert in a warp: lanes of the same expert vote first
        lt_mask = lanemask_lt()
        for i in cutlass.range_constexpr(CHUNK // THREADS):
            slot_idx = chunk_idx * CHUNK + i * THREADS + tidx
            slot_ok = Int32(slot_idx < num_slots)
            expert_idx = ld_b32(mExpertIdx.iterator.toint() + Int64(slot_idx) * 8, slot_ok, Int32(-1))
            peers = match_any(expert_idx)
            is_first_peer = slot_ok & Int32(cute.arch.popc(peers & lt_mask) == 0)
            red_add_shared(s_hist + expert_idx * 4, Int32(cute.arch.popc(peers)), is_first_peer)
        cute.arch.sync_threads()
        hist_addr = mHist.iterator.toint() + (Int64(chunk_idx) * G + Int64(tidx)) * 4
        st_b32(hist_addr, lds_b32(s_hist + tidx * 4), Int32(tidx < G))


class ThreadSlots:
    """
    The CHUNK // THREADS slots one thread of the grouping launch handles, one list entry per slot.
    """

    def __init__(self, expert_idxs: list, src_idxs: list, slot_idxs: list, slot_oks: list) -> None:
        self.expert_idxs, self.src_idxs, self.slot_idxs, self.slot_oks = expert_idxs, src_idxs, slot_idxs, slot_oks


class GroupSlots:
    """
    Grouping launch, one CTA per CHUNK slots. Each CTA sums the chunk histograms: all of them give
    every expert's first row, those below its chunk the chunk's base within each expert.
    """

    def __init__(self, num_groups: int, has_expand: bool) -> None:
        assert 0 < num_groups <= MAX_GROUPS
        self.G = num_groups
        self.has_expand = has_expand

    @cute.jit
    def __call__(self, mExpertIdx, mExpandIdx, mHist, mCuSeqlens, mAIdx, mRowOf, mSlotOf, mMask, mTable, stream) -> None:
        num_chunks = (mExpertIdx.shape[0] + CHUNK - 1) // CHUNK
        args = (mExpertIdx, mExpandIdx, mHist, mCuSeqlens, mAIdx, mRowOf, mSlotOf, mMask, mTable)
        self.kernel(*args).launch(grid=[num_chunks, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mExpertIdx, mExpandIdx, mHist, mCuSeqlens, mAIdx, mRowOf, mSlotOf, mMask, mTable) -> None:
        G = self.G
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, _, _ = cute.arch.block_idx()
        num_slots = mExpertIdx.shape[0]
        smem = cutlass.memory.SmemAllocator()
        s_total = smem.allocate(G * 4, byte_alignment=16).toint()  # slots per expert
        s_before = smem.allocate(G * 4, byte_alignment=16).toint()  # slots per expert in the chunks before this one
        s_run = smem.allocate(G * 4, byte_alignment=16).toint()  # running count within the chunk, then the expert's base row
        s_warp_count = smem.allocate(WARPS * G * 4, byte_alignment=16).toint()
        s_warp_base = smem.allocate(WARPS * G * 4, byte_alignment=16).toint()  # exclusive scan of s_warp_count over warps
        self.zero_counts(tidx, s_total, s_before, s_run, s_warp_count)
        cute.arch.sync_threads()
        self.sum_histograms(mHist, tidx, chunk_idx, num_slots, s_total, s_before)
        lt_mask = lanemask_lt()
        slots = self.load_slots(mExpertIdx, mExpandIdx, tidx, chunk_idx, num_slots)
        safe_expert_idxs, ranks = self.rank_slots(slots, tidx, lt_mask, s_run, s_warp_count, s_warp_base)
        self.store_expert_bases(mCuSeqlens, tidx, chunk_idx, num_slots, s_total, s_before, s_run)
        cute.arch.sync_threads()
        self.store_maps(mAIdx, mSlotOf, mRowOf, mMask, mTable, slots, safe_expert_idxs, ranks, s_run)

    @cute.jit
    def zero_counts(self, tidx, s_total, s_before, s_run, s_warp_count) -> None:
        G = self.G
        for i in cutlass.range_constexpr((WARPS * G + THREADS - 1) // THREADS):
            word_idx = i * THREADS + tidx
            sts_b32(s_warp_count + word_idx * 4, Uint32(0), Int32(word_idx < WARPS * G))
        sts_b32(s_total + tidx * 4, Uint32(0), Int32(tidx < G))
        sts_b32(s_before + tidx * 4, Uint32(0), Int32(tidx < G))
        sts_b32(s_run + tidx * 4, Uint32(0), Int32(tidx < G))

    @cute.jit
    def sum_histograms(self, mHist, tidx, chunk_idx, num_slots, s_total, s_before) -> None:
        """
        Slots of expert tidx in all chunks and in the chunks before this one, from the histograms.
        """
        G = self.G
        num_chunks = (num_slots + CHUNK - 1) // CHUNK
        if tidx < G:
            acc_total, acc_before = Int32(0), Int32(0)
            for hist_chunk_idx in cutlass.range(num_chunks, unroll=1):
                count = ld_b32(mHist.iterator.toint() + (Int64(hist_chunk_idx) * G + Int64(tidx)) * 4, Int32(1), Int32(0))
                acc_total, acc_before = acc_total + count, acc_before + count * Int32(hist_chunk_idx < chunk_idx)
            sts_b32(s_total + tidx * 4, Uint32(acc_total), Int32(1))
            sts_b32(s_before + tidx * 4, Uint32(acc_before), Int32(1))

    @cute.jit
    def load_slots(self, mExpertIdx, mExpandIdx, tidx, chunk_idx, num_slots) -> ThreadSlots:
        """
        Expert and source row of each of the thread's slots, with every load in flight at once.
        """
        chunk_start = chunk_idx * CHUNK
        expert_base = mExpertIdx.iterator.toint()
        expert_idxs, src_idxs, slot_idxs, slot_oks = [], [], [], []
        for i in cutlass.range_constexpr(CHUNK // THREADS):
            slot_idx = chunk_start + i * THREADS + tidx
            slot_ok = Int32(slot_idx < num_slots)
            expert_idx = ld_b32(expert_base + Int64(slot_idx) * 8, slot_ok, Int32(-1))
            src_idx = slot_idx
            if cutlass.const_expr(self.has_expand):
                src_idx = ld_b32(mExpandIdx.iterator.toint() + Int64(slot_idx) * 8, slot_ok, Int32(0))
            expert_idxs.append(expert_idx)
            src_idxs.append(src_idx)
            slot_idxs.append(slot_idx)
            slot_oks.append(slot_ok)
        return ThreadSlots(expert_idxs, src_idxs, slot_idxs, slot_oks)

    @cute.jit
    def rank_slots(self, slots, tidx, lt_mask, s_run, s_warp_count, s_warp_base):
        """
        Stable rank of each slot within its expert inside the chunk, and its expert index with
        invalid slots mapped to expert 0.
        """
        G = self.G
        warp = tidx // 32
        safe_expert_idxs, ranks = [], []
        for i in cutlass.range_constexpr(CHUNK // THREADS):
            expert_idx, slot_ok = slots.expert_idxs[i], slots.slot_oks[i]
            peers = match_any(expert_idx)
            num_lower_peers = cute.arch.popc(peers & lt_mask)
            num_peers = cute.arch.popc(peers)
            safe_expert_idx = expert_idx * slot_ok  # invalid lanes hold -1; 0 keeps their unmasked shared loads in bounds
            sts_b32(s_warp_count + (warp * G + safe_expert_idx) * 4, Uint32(num_peers), slot_ok & Int32(num_lower_peers == 0))
            cute.arch.sync_threads()
            self.scan_warp_counts(tidx, s_run, s_warp_count, s_warp_base)
            cute.arch.sync_threads()
            rank = Int32(lds_b32(s_warp_base + (warp * G + safe_expert_idx) * 4)) + num_lower_peers
            safe_expert_idxs.append(safe_expert_idx)
            ranks.append(rank)
        return safe_expert_idxs, ranks

    @cute.jit
    def scan_warp_counts(self, tidx, s_run, s_warp_count, s_warp_base) -> None:
        """
        Exclusive scan over warps of expert tidx's counts, continuing its running count; clears the
        per-warp counts for the next round.
        """
        G = self.G
        if tidx < G:
            acc_run = Int32(lds_b32(s_run + tidx * 4))
            for warp_idx in cutlass.range_constexpr(WARPS):
                count_addr = s_warp_count + (warp_idx * G + tidx) * 4
                count = Int32(lds_b32(count_addr))
                sts_b32(count_addr, Uint32(0), Int32(1))
                sts_b32(s_warp_base + (warp_idx * G + tidx) * 4, Uint32(acc_run), Int32(1))
                acc_run = acc_run + count
            sts_b32(s_run + tidx * 4, Uint32(acc_run), Int32(1))

    @cute.jit
    def store_expert_bases(self, mCuSeqlens, tidx, chunk_idx, num_slots, s_total, s_before, s_run) -> None:
        """
        Base row of expert tidx for this chunk into s_run: its first row plus its slots in the lower
        chunks. CTA 0 also writes cu_seqlens.
        """
        G = self.G
        if tidx < G:
            first_row = Int32(0)
            for group_idx in cutlass.range(tidx, unroll=1):
                first_row = first_row + Int32(lds_b32(s_total + group_idx * 4))
            sts_b32(s_run + tidx * 4, Uint32(first_row + Int32(lds_b32(s_before + tidx * 4))), Int32(1))
            if chunk_idx == 0:
                st_b32(mCuSeqlens.iterator.toint() + Int64(tidx) * 4, Uint32(first_row), Int32(1))
                if tidx == G - 1:
                    st_b32(mCuSeqlens.iterator.toint() + Int64(G) * 4, Uint32(Int32(num_slots)), Int32(1))

    @cute.jit
    def store_maps(self, mAIdx, mSlotOf, mRowOf, mMask, mTable, slots, safe_expert_idxs, ranks, s_run) -> None:
        """
        Store each slot's grouped row into the index maps and, with expand_idx, its expert bit and
        row into the token's mask and table.
        """
        G = self.G
        for i in cutlass.range_constexpr(CHUNK // THREADS):
            row = Int32(lds_b32(s_run + safe_expert_idxs[i] * 4)) + ranks[i]
            slot_ok = slots.slot_oks[i]
            st_b32(mAIdx.iterator.toint() + Int64(row) * 4, Uint32(slots.src_idxs[i]), slot_ok)
            st_b32(mSlotOf.iterator.toint() + Int64(row) * 4, Uint32(slots.slot_idxs[i]), slot_ok)
            st_b32(mRowOf.iterator.toint() + Int64(slots.slot_idxs[i]) * 4, Uint32(row), slot_ok)
            if cutlass.const_expr(self.has_expand):
                src_idx = Int64(slots.src_idxs[i])
                # atomic OR: the slots of one token, in any thread or CTA, set bits of the same mask word
                asm(None, [Int64(mMask.iterator.toint() + src_idx * 8).ir_value(), Int64(Int64(1) << Int64(safe_expert_idxs[i])).ir_value(), Int32(slot_ok).ir_value()], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p red.relaxed.gpu.global.or.b64 [$0], $1;\n\t}", "l,l,r", side_effects=True)
                st_b32(mTable.iterator.toint() + (src_idx * G + Int64(safe_expert_idxs[i])) * 4, Uint32(row), slot_ok)


class GatherSum:
    """
    out[t] = sum over the token's local experts e (ascending) of grad[table[t, e]], in fp32.
    """

    def __init__(self, hidden: int, num_groups: int, max_slots: int) -> None:
        assert hidden % 8 == 0
        self.H, self.G, self.K = hidden, num_groups, max_slots

    @cute.jit
    def __call__(self, mGrad, mMask, mTable, mOut, stream) -> None:
        num_ctas = (mOut.shape[0] + SUM_TOKENS - 1) // SUM_TOKENS
        self.kernel(mGrad, mMask, mTable, mOut).launch(grid=[num_ctas, 1, 1], block=[SUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mGrad, mMask, mTable, mOut) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        num_tokens = mOut.shape[0]
        for i in cutlass.range_constexpr(SUM_TOKENS):
            token_idx = bx * SUM_TOKENS + i
            token_ok = Int32(token_idx < num_tokens)
            rows, row_oks = self.find_expert_rows(mMask, mTable, token_idx, token_ok)
            self.sum_rows(mGrad, mOut, rows, row_oks, token_idx, token_ok, tidx)

    @cute.jit
    def find_expert_rows(self, mMask, mTable, token_idx, token_ok):
        """
        Grouped rows of the token's first K local experts in ascending order, and their predicates.
        """
        G, K = self.G, self.K
        # the token's expert bitmask; 0 past the last token
        expert_mask = Int64(
            asm(
                T.i64(), [Int64(mMask.iterator.toint() + Int64(token_idx) * 8).ir_value(), Int32(token_ok).ir_value()],
                "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b64 $0, 0;\n\t"
                "@p ld.global.nc.b64 $0, [$1];\n\t}",
                "=l,l,r",
            )
        )
        rows, row_oks = [], []
        for j in cutlass.range_constexpr(K):
            row_ok = Int32(expert_mask != Int64(0))
            # index of the lowest set bit (64 for 0): the next expert of the token
            expert_idx = Int32(asm(T.i32(), [Int64(expert_mask).ir_value()], "{\n\t.reg .b64 r;\n\tbrev.b64 r, $1;\n\tclz.b64 $0, r;\n\t}", "=r,l"))
            rows.append(ld_b32(mTable.iterator.toint() + (Int64(token_idx) * G + Int64(expert_idx)) * 4, row_ok, Int32(0)))
            row_oks.append(row_ok)
            expert_mask = expert_mask & (expert_mask - Int64(1))
        return rows, row_oks

    @cute.jit
    def sum_rows(self, mGrad, mOut, rows, row_oks, token_idx, token_ok, tidx) -> None:
        H, K = self.H, self.K
        num_vecs = H // 8
        grad_base, out_base = mGrad.iterator.toint(), mOut.iterator.toint()
        for pass_idx in cutlass.range_constexpr((num_vecs + SUM_THREADS - 1) // SUM_THREADS):
            vec_idx = pass_idx * SUM_THREADS + tidx
            vec_ok = token_ok
            if cutlass.const_expr(num_vecs % SUM_THREADS != 0):
                vec_ok = vec_ok & Int32(vec_idx < num_vecs)
            acc_sum = [cutlass.Float32(0.0)] * 8
            for j in cutlass.range_constexpr(K):
                words = ld_v4(grad_base + (Int64(rows[j]) * H + Int64(vec_idx) * 8) * 2, row_oks[j] & vec_ok)
                for k in cutlass.range_constexpr(4):
                    elem_lo, elem_hi = bf16x2_to_f32(words[k])
                    acc_sum[2 * k] = acc_sum[2 * k] + elem_lo
                    acc_sum[2 * k + 1] = acc_sum[2 * k + 1] + elem_hi
            out = tuple(f32x2_to_bf16x2(acc_sum[2 * k], acc_sum[2 * k + 1]) for k in range(4))
            st_v4(out_base + (Int64(token_idx) * H + Int64(vec_idx) * 8) * 2, out, vec_ok)


class SlotMaps(NamedTuple):
    cu_seqlens: torch.Tensor  # (G + 1,) int32
    a_idx: torch.Tensor  # (S,) int32
    row_of: torch.Tensor  # (S,) int32
    slot_of: torch.Tensor  # (S,) int32
    mask: torch.Tensor | None  # (T,) int64, None without expand_idx
    table: torch.Tensor | None  # (T * G,) int32, entries only where mask has the bit


COMPILED: dict = {}


def fake_vector(dtype: type, align: int) -> object:
    return fake(dtype, (cute.sym_int(),), (1,), align)


def bf16_expert_group_slots_sm90(expert_idxs: torch.Tensor, expand_idx: torch.Tensor | None, num_groups: int, num_tokens: int) -> SlotMaps:
    G = num_groups
    num_slots = expert_idxs.shape[0]
    device = expert_idxs.device
    assert 0 < G <= MAX_GROUPS
    # one allocation: the token bitmask (int64) first, then each int32 map 16-byte aligned
    has_expand = expand_idx is not None
    num_mask_words = (num_tokens * 2 + 3) // 4 * 4 if has_expand else 0
    slots_padded, cu_padded = (num_slots + 3) // 4 * 4, (G + 4) // 4 * 4
    num_chunks = (num_slots + CHUNK - 1) // CHUNK
    num_table_words = (num_tokens * G + 3) // 4 * 4 if has_expand else 0
    storage = torch.empty(num_mask_words + cu_padded + 3 * slots_padded + num_table_words + num_chunks * G, dtype=torch.int32, device=device)
    int32_maps = storage[num_mask_words:]
    cu_seqlens = int32_maps[: G + 1]
    a_idx, row_of, slot_of = (int32_maps[cu_padded + j * slots_padded : cu_padded + j * slots_padded + num_slots] for j in range(3))
    table = int32_maps[cu_padded + 3 * slots_padded : cu_padded + 3 * slots_padded + num_tokens * G] if has_expand else None
    hist = int32_maps[cu_padded + 3 * slots_padded + num_table_words :]
    mask = storage[: 2 * num_tokens].view(torch.int64) if has_expand else None
    if num_slots == 0:
        cu_seqlens.zero_()
        if mask is not None:
            mask.zero_()
        return SlotMaps(cu_seqlens, a_idx, row_of, slot_of, mask, table)
    key = ("group_slots", G, has_expand)
    compiled = COMPILED.get(key)
    if compiled is None:
        i32_vector = lambda: fake_vector(Int32, 4)
        compiled = COMPILED[key] = (cute.compile(ChunkHistogram(G), fake_vector(Int64, 8), i32_vector(), fake_vector(Int64, 8), fake_stream(), options="--enable-tvm-ffi"), cute.compile(GroupSlots(G, has_expand), fake_vector(Int64, 8), fake_vector(Int64, 8), i32_vector(), i32_vector(), i32_vector(), i32_vector(), i32_vector(), fake_vector(Int64, 8), i32_vector(), fake_stream(), options="--enable-tvm-ffi"))
    hist_fn, group_fn = compiled
    hist_fn(expert_idxs, hist, mask if has_expand else expert_idxs[:0])
    if has_expand:
        group_fn(expert_idxs, expand_idx, hist, cu_seqlens, a_idx, row_of, slot_of, mask, table)
    else:
        group_fn(expert_idxs, expert_idxs, hist, cu_seqlens, a_idx, row_of, slot_of, expert_idxs, a_idx)
    return SlotMaps(cu_seqlens, a_idx, row_of, slot_of, mask, table)


def bf16_expert_gather_sum_sm90(grad: torch.Tensor, maps: SlotMaps, num_groups: int, max_slots: int) -> torch.Tensor:
    num_tokens, H = maps.mask.shape[0], grad.shape[1]
    out = grad.new_empty(num_tokens, H)
    if num_tokens == 0:
        return out
    key = ("gather_sum", H, num_groups, max_slots)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = cute.compile(GatherSum(H, num_groups, max_slots), fake(cutlass.BFloat16, (cute.sym_int(), H), (H, 1)), fake_vector(Int64, 8), fake_vector(Int32, 4), fake(cutlass.BFloat16, (cute.sym_int(), H), (H, 1)), fake_stream(), options="--enable-tvm-ffi")
    fn(grad, maps.mask, maps.table, out)
    return out


SEG_THREADS = 128
SEG_ROWS = 256  # rows per partial sum


class SegmentSum:
    """
    fp32 row-order partial sums of SEG_ROWS-row chunks of each segment of x[idx[r]] (or x[r]); one
    CTA per (chunk, column block).
    """

    def __init__(self, num_cols: int, use_gather: bool) -> None:
        assert num_cols % 8 == 0
        self.N, self.use_gather = num_cols, use_gather

    @cute.jit
    def __call__(self, mX, mIdx, mCu, mChunks, mPartial, stream) -> None:
        num_chunks = mPartial.shape[0]
        num_col_blocks = (self.N // 8 + SEG_THREADS - 1) // SEG_THREADS
        self.kernel(mX, mIdx, mCu, mChunks, mPartial).launch(grid=[num_chunks, num_col_blocks, 1], block=[SEG_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX, mIdx, mCu, mChunks, mPartial) -> None:
        N = self.N
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, col_block_idx, _ = cute.arch.block_idx()
        vec_idx = col_block_idx * SEG_THREADS + tidx
        vec_ok = Int32(vec_idx < N // 8)
        # chunk chunk_idx covers rows [row_start, segment_end) of its segment, at most SEG_ROWS of them
        group_idx, group_chunk_idx = ld_b32(mChunks.iterator.toint() + Int64(chunk_idx) * 8, Int32(1), Int32(0)), ld_b32(mChunks.iterator.toint() + Int64(chunk_idx) * 8 + 4, Int32(1), Int32(0))
        segment_end = ld_b32(mCu.iterator.toint() + Int64(group_idx + 1) * 4, Int32(1), Int32(0))
        row_start = ld_b32(mCu.iterator.toint() + Int64(group_idx) * 4, Int32(1), Int32(0)) + group_chunk_idx * SEG_ROWS
        num_rows = cutlass.min(segment_end - row_start, Int32(SEG_ROWS))
        acc_sum = [cutlass.Float32(0.0)] * 8
        for row_offset in cutlass.range(num_rows, unroll=4):
            row_idx = row_start + row_offset
            if cutlass.const_expr(self.use_gather):
                row_idx = ld_b32(mIdx.iterator.toint() + Int64(row_idx) * 4, Int32(1), Int32(0))
            words = ld_v4(mX.iterator.toint() + (Int64(row_idx) * N + Int64(vec_idx) * 8) * 2, vec_ok)
            for k in cutlass.range_constexpr(4):
                elem_lo, elem_hi = bf16x2_to_f32(words[k])
                acc_sum[2 * k] = acc_sum[2 * k] + elem_lo
                acc_sum[2 * k + 1] = acc_sum[2 * k + 1] + elem_hi
        partial_addr = mPartial.iterator.toint() + (Int64(chunk_idx) * N + Int64(vec_idx) * 8) * 4
        for k in cutlass.range_constexpr(8):
            st_b32(partial_addr + k * 4, Uint32(asm(T.i32(), [cutlass.Float32(acc_sum[k]).ir_value()], "mov.b32 $0, $1;", "=r,f")), vec_ok)


def bf16_expert_segment_sum_sm90(x: torch.Tensor, cu_host: list[int], cu: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
    G, N = len(cu_host) - 1, x.shape[1]
    num_group_chunks = [-(-(cu_host[group_idx + 1] - cu_host[group_idx]) // SEG_ROWS) for group_idx in range(G)]
    chunks = [(group_idx, group_chunk_idx) for group_idx in range(G) for group_chunk_idx in range(num_group_chunks[group_idx])]
    if not chunks:
        return torch.zeros(G, N, dtype=torch.float32, device=x.device)
    chunk_meta = torch.tensor([value for chunk in chunks for value in chunk] + num_group_chunks, dtype=torch.int32)
    chunk_meta = chunk_meta.to(x.device, non_blocking=True)
    chunk_table, num_group_chunks_dev = chunk_meta[: 2 * len(chunks)].view(-1, 2), chunk_meta[2 * len(chunks) :]
    partial = torch.empty(len(chunks), N, dtype=torch.float32, device=x.device)
    key = ("segment_sum", N, idx is not None)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = cute.compile(SegmentSum(N, idx is not None), fake(cutlass.BFloat16, (cute.sym_int(), N), (N, 1)), fake_vector(Int32, 4), fake_vector(Int32, 4), fake(Int32, (cute.sym_int(), 2), (2, 1), 8), fake(cutlass.Float32, (cute.sym_int(), N), (N, 1)), fake_stream(), options="--enable-tvm-ffi")
    fn(x, idx if idx is not None else cu, cu, chunk_table, partial)
    # the chunks of a segment are consecutive; sum them in order (an empty segment sums to 0)
    return torch.segment_reduce(partial, "sum", lengths=num_group_chunks_dev, initial=0.0)


ROW_THREADS, ROWS_PER_CTA = 128, 4


class GatherRows:
    def __init__(self, num_cols: int) -> None:
        assert num_cols % 8 == 0
        self.N = num_cols

    @cute.jit
    def __call__(self, mX, mIdx, mOut, stream) -> None:
        num_ctas = (mOut.shape[0] + ROWS_PER_CTA - 1) // ROWS_PER_CTA
        self.kernel(mX, mIdx, mOut).launch(grid=[num_ctas, 1, 1], block=[ROW_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX, mIdx, mOut) -> None:
        N, num_vecs = self.N, self.N // 8
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        num_rows = mOut.shape[0]
        for i in cutlass.range_constexpr(ROWS_PER_CTA):
            row_idx = bx * ROWS_PER_CTA + i
            row_ok = Int32(row_idx < num_rows)
            src_row = ld_b32(mIdx.iterator.toint() + Int64(row_idx) * 4, row_ok, Int32(0))
            for pass_idx in cutlass.range_constexpr((num_vecs + ROW_THREADS - 1) // ROW_THREADS):
                vec_idx = pass_idx * ROW_THREADS + tidx
                vec_ok = row_ok
                if cutlass.const_expr(num_vecs % ROW_THREADS != 0):
                    vec_ok = vec_ok & Int32(vec_idx < num_vecs)
                words = ld_v4(mX.iterator.toint() + (Int64(src_row) * N + Int64(vec_idx) * 8) * 2, vec_ok)
                st_v4(mOut.iterator.toint() + (Int64(row_idx) * N + Int64(vec_idx) * 8) * 2, words, vec_ok)


def bf16_expert_gather_rows_sm90(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    N = x.shape[1]
    out = x.new_empty(idx.shape[0], N)
    if idx.shape[0] == 0:
        return out
    key = ("gather_rows", N)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = cute.compile(GatherRows(N), fake(cutlass.BFloat16, (cute.sym_int(), N), (N, 1)), fake_vector(Int32, 4), fake(cutlass.BFloat16, (cute.sym_int(), N), (N, 1)), fake_stream(), options="--enable-tvm-ffi")
    fn(x, idx, out)
    return out
