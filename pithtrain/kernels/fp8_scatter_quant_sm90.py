"""
SM90 CuTe-DSL scatter_quant: dedup expansion + scatter by expert + K-major FP8 quantization in two
launches and one read of the received tokens; the grouped bf16 input never exists.

1. Prep (one CTA per 2048 token slots) ranks each slot stably within its expert, writes
   grouped_mm_offs and ks_tensor, each slot's rank, and the inverse map inv[g * T + k] = the source
   row of row k of group g (T token slots).
2. ScatterQuant, the rowwise_kmajor tile kernel whose row loads go through inv (padding rows load
   zeros), quantizes and writes both scales MN-major; its column-block-0 CTAs also write
   reverse_shuffle_idxs. It is launched as a programmatic dependent (PDL) of the prep kernel.

Placement is stable (slot order within each group), so the output is deterministic and bit-identical
to the unfused path run with that placement.

The prep kernel orders chunks without a grid barrier: each CTA takes a ticket (chunk index in
dispatch order), publishes its chunk histogram behind a release flag, and waits only for lower
chunks, which already hold their tickets and never wait on it, so forward progress needs no
co-residency. The highest chunk, having waited for all others, writes the group offsets.
"""

from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, atom_add_u64, fence_gpu, lanemask_lt, ld_b32, ld_relaxed, ld_v4, lds_b32, match_any, red_add_shared, st_b32, st_b64, sts_b32
from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import THREADS, TILE, RowwiseKmajor, fp8cast_rowwise_kmajor_sm90
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad
from pithtrain.kernels.host_transfer import get_pinned_buffer
from pithtrain.kernels.token_scatter import scatter_by_expert

__all__ = ["Prep", "ScatterQuant", "scatter_quant_sm90", "supports_scatter_quant"]

PREP_THREADS = 512
PREP_WARPS = PREP_THREADS // 32
PREP_CHUNK = 2048  # token slots per prep CTA
MAX_GROUPS = PREP_THREADS
ALIGN = 128


class Prep:
    def __init__(self, num_groups: int, has_expand: bool) -> None:
        assert 0 < num_groups <= MAX_GROUPS
        self.G = num_groups
        self.has_expand = has_expand

    @cute.jit
    def __call__(self, mE: cute.Tensor, mExp: cute.Tensor, mInv: cute.Tensor, mRank: cute.Tensor, mOffs: cute.Tensor, mKs: cute.Tensor, mCnt: cute.Tensor, mFlags: cute.Tensor, mHist: cute.Tensor, mCtr: cute.Tensor, stream) -> None:
        num_chunks = (mE.shape[0] + PREP_CHUNK - 1) // PREP_CHUNK
        args = (mE, mExp, mInv, mRank, mOffs, mKs, mCnt, mFlags, mHist, mCtr)
        self.kernel(*args).launch(grid=[num_chunks, 1, 1], block=[PREP_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mE: cute.Tensor, mExp: cute.Tensor, mInv: cute.Tensor, mRank: cute.Tensor, mOffs: cute.Tensor, mKs: cute.Tensor, mCnt: cute.Tensor, mFlags: cute.Tensor, mHist: cute.Tensor, mCtr: cute.Tensor) -> None:
        G = self.G
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        T = mE.shape[0]
        num_chunks = (T + PREP_CHUNK - 1) // PREP_CHUNK
        experts_base, inv_base, flags_base = mE.iterator.toint(), mInv.iterator.toint(), mFlags.iterator.toint()
        hist_base, counter_base = mHist.iterator.toint(), mCtr.iterator.toint()

        smem = cutlass.memory.SmemAllocator()
        s_chunk_count = smem.allocate(G * 4, byte_alignment=16).toint()  # per-expert count in this chunk, running while ranking
        s_lower_count = smem.allocate(G * 4, byte_alignment=16).toint()
        s_warp_count = smem.allocate(PREP_WARPS * G * 4, byte_alignment=16).toint()  # per-warp counts of one round
        s_warp_base = smem.allocate(PREP_WARPS * G * 4, byte_alignment=16).toint()  # per-warp exclusive bases of one round
        s_ticket = smem.allocate(16, byte_alignment=16).toint()
        cute.arch.griddepcontrol_launch_dependents()  # let the main kernel's CTAs get resident

        self.clear_counts(tidx, s_chunk_count, s_lower_count, s_warp_count)
        chunk_idx, epoch = self.take_ticket(tidx, counter_base, s_ticket)
        expert_idxs, src_rows, token_idxs, token_oks = self.load_chunk(experts_base, mExp, chunk_idx, tidx, T)
        safe_expert_idxs, chunk_ranks = self.rank_in_chunk(expert_idxs, token_oks, tidx, warp, s_chunk_count, s_warp_count, s_warp_base)
        self.publish_histogram(hist_base, flags_base, chunk_idx, epoch, tidx, s_chunk_count)
        self.wait_lower_chunks(flags_base, chunk_idx, epoch, tidx)
        self.sum_lower_chunks(hist_base, chunk_idx, tidx, s_lower_count)
        self.store_ranks(inv_base, mRank.iterator.toint(), T, src_rows, token_idxs, token_oks, safe_expert_idxs, chunk_ranks, s_lower_count)
        if chunk_idx == num_chunks - 1:
            self.store_group_offsets(mOffs.iterator.toint(), mKs.iterator.toint(), mCnt.iterator.toint(), counter_base, num_chunks, tidx, s_chunk_count, s_lower_count)

    @cute.jit
    def clear_counts(self, tidx, s_chunk_count, s_lower_count, s_warp_count) -> None:
        G = self.G
        for j in cutlass.range_constexpr((PREP_WARPS * G + PREP_THREADS - 1) // PREP_THREADS):
            entry_idx = j * PREP_THREADS + tidx
            sts_b32(s_warp_count + entry_idx * 4, Uint32(0), Int32(entry_idx < PREP_WARPS * G))
        sts_b32(s_chunk_count + tidx * 4, Uint32(0), Int32(tidx < G))
        sts_b32(s_lower_count + tidx * 4, Uint32(0), Int32(tidx < G))

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
    def load_chunk(self, experts_base, mExp, chunk_idx, tidx, T):
        """
        All loads are issued before any is used.
        """
        expert_idxs, src_rows, token_idxs, token_oks = [], [], [], []
        for i in cutlass.range_constexpr(PREP_CHUNK // PREP_THREADS):
            token_idx = chunk_idx * PREP_CHUNK + i * PREP_THREADS + tidx
            token_ok = Int32(token_idx < T)
            expert_idx = ld_b32(experts_base + Int64(token_idx) * 8, token_ok, Int32(-1))
            src_row = token_idx
            if cutlass.const_expr(self.has_expand):
                src_row = ld_b32(mExp.iterator.toint() + Int64(token_idx) * 8, token_ok, Int32(0))
            expert_idxs.append(expert_idx)
            src_rows.append(src_row)
            token_idxs.append(token_idx)
            token_oks.append(token_ok)
        return expert_idxs, src_rows, token_idxs, token_oks

    @cute.jit
    def rank_in_chunk(self, expert_idxs, token_oks, tidx, warp, s_chunk_count, s_warp_count, s_warp_base):
        """
        Stable rank of each token slot within its expert inside this chunk, and its expert index (0
        for an invalid slot); leaves the chunk's per-expert counts in s_chunk_count.
        """
        G = self.G
        lower_lanes_mask = lanemask_lt()
        safe_expert_idxs, chunk_ranks = [], []
        for i in cutlass.range_constexpr(PREP_CHUNK // PREP_THREADS):
            expert_idx, token_ok = expert_idxs[i], token_oks[i]
            peers = match_any(expert_idx)
            num_lower_peers = cute.arch.popc(peers & lower_lanes_mask)
            num_peers = cute.arch.popc(peers)
            safe_expert_idx = expert_idx * token_ok  # safe index for invalid lanes (their stores/loads are masked)
            sts_b32(s_warp_count + (warp * G + safe_expert_idx) * 4, Uint32(num_peers), token_ok & Int32(num_lower_peers == 0))
            cute.arch.sync_threads()
            self.scan_warp_counts(tidx, s_chunk_count, s_warp_count, s_warp_base)
            cute.arch.sync_threads()
            chunk_rank = Int32(lds_b32(s_warp_base + (warp * G + safe_expert_idx) * 4)) + num_lower_peers
            safe_expert_idxs.append(safe_expert_idx)
            chunk_ranks.append(chunk_rank)
        return safe_expert_idxs, chunk_ranks

    @cute.jit
    def scan_warp_counts(self, tidx, s_chunk_count, s_warp_count, s_warp_base) -> None:
        """
        Exclusive scan over warps of one round's per-expert counts, continuing the chunk's running
        count; clears the per-warp counts for the next round.
        """
        G = self.G
        if tidx < G:
            acc_count = Int32(lds_b32(s_chunk_count + tidx * 4))
            for warp_idx in cutlass.range_constexpr(PREP_WARPS):
                count_addr = s_warp_count + (warp_idx * G + tidx) * 4
                warp_count = Int32(lds_b32(count_addr))
                sts_b32(count_addr, Uint32(0), Int32(1))
                sts_b32(s_warp_base + (warp_idx * G + tidx) * 4, Uint32(acc_count), Int32(1))
                acc_count = acc_count + warp_count
            sts_b32(s_chunk_count + tidx * 4, Uint32(acc_count), Int32(1))

    @cute.jit
    def publish_histogram(self, hist_base, flags_base, chunk_idx, epoch, tidx, s_chunk_count) -> None:
        G = self.G
        if tidx < G:
            st_b32(hist_base + (Int64(chunk_idx) * G + tidx) * 4, lds_b32(s_chunk_count + tidx * 4), Int32(1))
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
                "@q bra SQ_DONE_${:uid};\n"
                "SQ_LOOP_${:uid}:\n\t"
                "ld.acquire.gpu.global.b32 t, [$0];\n\t"
                "setp.ne.b32 p, t, $1;\n\t"
                "@p bra SQ_LOOP_${:uid};\n"
                "SQ_DONE_${:uid}:\n\t}",
                "l,r,r", side_effects=True,
            )
        cute.arch.sync_threads()
        fence_gpu()

    @cute.jit
    def sum_lower_chunks(self, hist_base, chunk_idx, tidx, s_lower_count) -> None:
        G = self.G
        num_lower_entries = chunk_idx * G
        for round_idx in cutlass.range((num_lower_entries + PREP_THREADS - 1) // PREP_THREADS, unroll=1):
            entry_idx = round_idx * PREP_THREADS + tidx
            entry_ok = Int32(entry_idx < num_lower_entries)
            count = ld_relaxed(hist_base + Int64(entry_idx) * 4, entry_ok)
            red_add_shared(s_lower_count + (entry_idx % G) * 4, count, entry_ok)
        cute.arch.sync_threads()

    @cute.jit
    def store_ranks(self, inv_base, rank_base, T, src_rows, token_idxs, token_oks, safe_expert_idxs, chunk_ranks, s_lower_count) -> None:
        for i in cutlass.range_constexpr(PREP_CHUNK // PREP_THREADS):
            rank = Int32(lds_b32(s_lower_count + safe_expert_idxs[i] * 4)) + chunk_ranks[i]
            inv_addr = inv_base + (Int64(safe_expert_idxs[i]) * T + rank) * 4
            st_b32(inv_addr, Uint32(src_rows[i]), token_oks[i])
            st_b32(rank_base + Int64(token_idxs[i]) * 4, Uint32(rank), token_oks[i])

    @cute.jit
    def store_group_offsets(self, offs_base, ks_base, counts_base, counter_base, num_chunks, tidx, s_chunk_count, s_lower_count) -> None:
        """
        Run by the last chunk, which has waited for all others.
        """
        G = self.G
        if tidx < G:
            group_end = Int32(0)
            for group_idx in cutlass.range(tidx + 1, unroll=1):
                group_count = Int32(lds_b32(s_lower_count + group_idx * 4)) + Int32(lds_b32(s_chunk_count + group_idx * 4))
                group_end = group_end + (group_count + (ALIGN - 1)) // ALIGN * ALIGN
            count = Int32(lds_b32(s_lower_count + tidx * 4)) + Int32(lds_b32(s_chunk_count + tidx * 4))
            padded_count = (count + (ALIGN - 1)) // ALIGN * ALIGN
            st_b32(offs_base + Int64(tidx) * 4, Uint32(group_end), Int32(1))
            st_b32(ks_base + Int64(tidx) * 4, Uint32(padded_count), Int32(1))
            st_b32(counts_base + Int64(tidx) * 4, Uint32(count), Int32(1))
        if tidx == 0:
            # every CTA holds its ticket (this one took the last): next epoch, zero tickets
            atom_add_u64(counter_base, (Int64(1) << 32) - Int64(num_chunks))


class ScatterQuant(RowwiseKmajor):
    """
    RowwiseKmajor whose row k of group g is gathered[inv[g * T + k]]. With pdl its CTAs become
    resident while the prep kernel runs and wait in griddepcontrol.wait for its results.
    """

    def __init__(self, N: int, num_groups: int, pdl: bool) -> None:
        super().__init__(N, num_groups)
        self.pdl = pdl

    @cute.jit
    def __call__(self, mX: cute.Tensor, mE: cute.Tensor, mRank: cute.Tensor, mInv: cute.Tensor, mCnt: cute.Tensor, mOffs: cute.Tensor, mRsi: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKm: cute.Tensor, mKmScale: cute.Tensor, stream) -> None:
        num_row_tiles = mTok.shape[0] // TILE
        args = (mX, mE, mRank, mInv, mCnt, mOffs, mRsi, mTok, mTokScale, mKm, mKmScale)
        self.kernel(*args).launch(grid=[self.num_col_tiles, num_row_tiles, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=self.pdl)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mE: cute.Tensor, mRank: cute.Tensor, mInv: cute.Tensor, mCnt: cute.Tensor, mOffs: cute.Tensor, mRsi: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKm: cute.Tensor, mKmScale: cute.Tensor) -> None:
        G = self.num_groups
        tidx, _, _ = cute.arch.thread_idx()
        pid_col, pid_row, _ = cute.arch.block_idx()
        T = mRsi.shape[0]
        x_base, offs_base = mX.iterator.toint(), mOffs.iterator.toint()
        bases = (x_base, offs_base, mTok.iterator.toint(), mTokScale.iterator.toint(), mKm.iterator.toint(), mKmScale.iterator.toint())
        smem = cutlass.memory.SmemAllocator()
        s_red = smem.allocate_tensor(Uint32, cute.make_layout(self.RED_WORDS), 16)
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()

        # M = total padded rows: the column stride of the MN-major rowwise scale (M % 128 == 0)
        M = Int64(ld_b32(offs_base + (G - 1) * 4, Int32(1), Int32(0)))
        row_start = Int64(pid_row) * TILE
        if row_start < M:
            rows = self.gather_rows(x_base, offs_base, mInv.iterator.toint(), mCnt.iterator.toint(), T, row_start, pid_col, tidx)
            self.store_reverse_shuffle(mE.iterator.toint(), mRank.iterator.toint(), offs_base, mRsi.iterator.toint(), T, pid_row, pid_col, tidx)
            self.quantize_tile(rows, pid_row, pid_col, tidx, M, bases, s_red.iterator.toint())

    @cute.jit
    def gather_rows(self, x_base, offs_base, inv_base, counts_base, T, row_start, pid_col, tidx):
        """
        The thread's 8 rows of the tile, four words (8 bf16) each, read through the inverse map of
        the tile's group; padding rows and columns past N read as zero.
        """
        N = self.N
        lane, _, col_vec_idx, row_block_idx = self.thread_coords(tidx)
        group_idx = self.find_group_idx(offs_base, lane, row_start)
        group_start = Int64(ld_b32(offs_base + Int64(group_idx - 1) * 4, Int32(group_idx > 0), Int32(0)))
        num_group_tokens = Int64(ld_b32(counts_base + Int64(group_idx) * 4, Int32(1), Int32(0)))
        col0 = pid_col * TILE + col_vec_idx * 8
        col_ok = Int32(col0 < N)
        group_inv_base = inv_base + Int64(group_idx) * T * 4
        rows = []
        for i in cutlass.range_constexpr(8):
            group_row = row_start + row_block_idx * 8 + i - group_start
            row_ok = Int32(group_row < num_group_tokens)
            src_row = ld_b32(group_inv_base + group_row * 4, row_ok, Int32(0))
            rows.append(ld_v4(x_base + (Int64(src_row) * N + col0) * 2, row_ok & col_ok))
        return rows

    @cute.jit
    def store_reverse_shuffle(self, experts_base, rank_base, offs_base, rsi_base, T, pid_row, pid_col, tidx) -> None:
        """
        reverse_shuffle_idxs[t] = start of expert(t) + rank(t) for the tile's token slots.
        """
        token_idx = Int64(pid_row) * TILE + tidx
        token_ok = Int32(pid_col == 0) & Int32(tidx < TILE) & Int32(token_idx < T)
        expert_idx, rank = ld_b32(experts_base + token_idx * 8, token_ok, Int32(0)), ld_b32(rank_base + token_idx * 4, token_ok, Int32(0))
        expert_start = ld_b32(offs_base + Int64(expert_idx - 1) * 4, token_ok & Int32(expert_idx > 0), Int32(0))
        st_b64(rsi_base + token_idx * 8, Int64(expert_start) + Int64(rank), token_ok)


PREP: dict = {}
MAIN: dict = {}


class Scratch:
    """
    Per-device prep scratch: a ticket counter, epoch flags and chunk histograms. The counter packs
    (epoch << 32) | tickets; the last chunk of each launch advances the epoch and zeroes the
    tickets, and a chunk's flag holds the epoch it was published in. The state lives on the device,
    so launches may be captured in a CUDA graph; they must not run concurrently on two streams.
    """

    def __init__(self, device: torch.device, num_chunks: int, num_groups: int) -> None:
        self.num_chunks, self.num_groups = num_chunks, num_groups
        self.counter = torch.full((1,), 1 << 32, dtype=torch.int64, device=device)  # epoch 1
        self.flags = torch.zeros(num_chunks, dtype=torch.int32, device=device)
        self.hist = torch.empty(num_chunks * num_groups, dtype=torch.int32, device=device)


SCRATCH: dict = {}
copy_stream: Optional[torch.cuda.Stream] = None
copy_event: Optional[torch.cuda.Event] = None


def get_scratch(device: torch.device, num_chunks: int, num_groups: int) -> Scratch:
    scratch = SCRATCH.get(device.index)
    if scratch is None or scratch.num_chunks < num_chunks or scratch.num_groups < num_groups:
        num_chunks, num_groups = max(num_chunks, scratch.num_chunks if scratch is not None else 0, 64), max(num_groups, scratch.num_groups if scratch is not None else 0)
        scratch = SCRATCH[device.index] = Scratch(device, num_chunks, num_groups)
    return scratch


def compile_prep(G: int, has_expand: bool) -> Callable:
    fake_int32_vector = lambda: fake(Int32, (cute.sym_int(),), (1,), 4)
    return cute.compile(Prep(G, has_expand), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int64, (cute.sym_int(),), (1,), 8), fake_int32_vector(), fake_int32_vector(), fake(Int32, (G,), (1,), 4), fake(Int32, (G,), (1,), 4), fake(Int32, (G,), (1,), 4), fake_int32_vector(), fake_int32_vector(), fake(Int64, (1,), (1,), 8), fake_stream(), options="--enable-tvm-ffi")


def compile_main(N: int, G: int, pdl: bool) -> Callable:
    M = cute.sym_int()
    num_col_tiles = (N + TILE - 1) // TILE
    return cute.compile(ScatterQuant(N, G, pdl), fake(cutlass.BFloat16, (cute.sym_int(), N), (N, 1)), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int32, (cute.sym_int(),), (1,), 4), fake(Int32, (cute.sym_int(),), (1,), 4), fake(Int32, (G,), (1,), 4), fake(Int32, (G,), (1,), 4), fake(Int64, (cute.sym_int(),), (1,), 8), fake(cutlass.Float8E4M3FN, (M, N), (N, 1)), fake(cutlass.Float32, (cute.sym_int(), num_col_tiles), (1, cute.sym_int64())), fake(cutlass.Float8E4M3FN, (cute.sym_int(),), (1,)), fake(cutlass.Float32, (N, cute.sym_int()), (1, mn_pad(N))), fake_stream(), options="--enable-tvm-ffi")


def run_unfused(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor], group_counts: Optional[list[int]]) -> tuple:
    """
    The unfused path for inputs the fused kernels do not cover (rows in atomic_add order).
    """
    x = gathered if expand_idx is None else gathered[expand_idx]
    out, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = scatter_by_expert(x, expert_idxs, num_groups, group_counts=group_counts)
    return fp8cast_rowwise_kmajor_sm90(out, grouped_mm_offs), reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor


def supports_scatter_quant(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor] = None) -> bool:
    T, N = expert_idxs.shape[0], gathered.shape[1]
    return not (T == 0 or N == 0 or not 0 < num_groups <= MAX_GROUPS or gathered.dtype != torch.bfloat16 or N % 8 != 0 or not gathered.is_contiguous() or gathered.data_ptr() % 16 != 0 or expert_idxs.dtype != torch.int64 or not expert_idxs.is_contiguous() or (expand_idx is not None and (expand_idx.dtype != torch.int64 or not expand_idx.is_contiguous())))


def scatter_quant_sm90(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor] = None, padding_alignment: int = 128, group_counts: Optional[list[int]] = None, pdl: bool = True) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """
    Scales are MN-major.
    """
    global copy_stream, copy_event
    assert padding_alignment == ALIGN, "the K-major FP8 layout needs 128-row groups"
    G = num_groups
    T, N = expert_idxs.shape[0], gathered.shape[1]
    device = gathered.device
    if expand_idx is None and gathered.shape[0] != T:
        raise ValueError(f"{gathered.shape[0]=} must equal {T=} without expand_idx")
    if expand_idx is not None and expand_idx.shape[0] != T:
        raise ValueError(f"{expand_idx.shape[0]=} must equal {T=}")
    if not supports_scatter_quant(gathered, expert_idxs, G, expand_idx):
        return run_unfused(gathered, expert_idxs, G, expand_idx, group_counts)

    # outputs: exact when the host knows the counts, else an upper bound
    if group_counts is not None:
        ks = [(count + ALIGN - 1) // ALIGN * ALIGN for count in group_counts]
        M_alloc = sum(ks)
    else:
        ks = None
        M_alloc = (T + G * (ALIGN - 1) + ALIGN - 1) // ALIGN * ALIGN
    num_col_tiles = (N + TILE - 1) // TILE
    # one int32 allocation: grouped_mm_offs | ks_tensor | counts | rank (T) | inv (G * T)
    meta = torch.empty(3 * G + T + G * T, dtype=torch.int32, device=device)
    grouped_mm_offs, ks_tensor, group_token_counts = meta[:G], meta[G : 2 * G], meta[2 * G : 3 * G]
    rank, inv = meta[3 * G : 3 * G + T], meta[3 * G + T :]
    reverse_shuffle_idxs, x_fp8 = torch.empty(T, dtype=torch.int64, device=device), torch.empty((M_alloc, N), dtype=torch.float8_e4m3fn, device=device)
    # the kernel strides this scale by the device-side M <= M_alloc, so the M_alloc buffer holds it
    x_scale = mn_major_scale(M_alloc, num_col_tiles, device)
    x_ch_fp8 = torch.empty(M_alloc * N, dtype=torch.float8_e4m3fn, device=device)
    x_ch_scale = mn_major_scale(N, M_alloc // TILE, device)

    num_chunks = (T + PREP_CHUNK - 1) // PREP_CHUNK
    scratch = get_scratch(device, num_chunks, G)
    has_expand = expand_idx is not None
    prep_fn = PREP.get((G, has_expand))
    if prep_fn is None:
        prep_fn = PREP[(G, has_expand)] = compile_prep(G, has_expand)
    # without expand_idx the kernel never reads mExp, so any int64 vector fills the slot
    expand_arg = expand_idx if has_expand else expert_idxs
    prep_fn(expert_idxs, expand_arg, inv, rank, grouped_mm_offs, ks_tensor, group_token_counts, scratch.flags, scratch.hist, scratch.counter)

    if ks is None:
        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=device)
            copy_event = torch.cuda.Event()
        copy_event.record()

    key = (N, G, pdl)
    fn = MAIN.get(key)
    if fn is None:
        fn = MAIN[key] = compile_main(N, G, pdl)
    fn(gathered, expert_idxs, rank, inv, group_token_counts, grouped_mm_offs, reverse_shuffle_idxs, x_fp8, x_scale, x_ch_fp8, x_ch_scale)

    if ks is None:
        ks_cpu = get_pinned_buffer("scatter_quant_ks", G, torch.int32)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(copy_event)
            ks_cpu.copy_(ks_tensor, non_blocking=True)
        copy_stream.synchronize()
        ks = ks_cpu.tolist()
        M = sum(ks)
        if M != M_alloc:
            x_fp8, x_ch_fp8 = x_fp8[:M], x_ch_fp8[: M * N]
            x_scale = x_scale.as_strided((M, num_col_tiles), (1, M))
            x_ch_scale = x_ch_scale[:, : M // TILE]
    return (x_fp8, x_scale, x_ch_fp8, x_ch_scale), reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor
