"""
Per-token log-probability and entropy of a tempered softmax on SM90 (CuTe-DSL). The forward reduces
one row per cluster of CTAs, each CTA bulk-copying its slice of the row into shared memory; the
backward splits each row over several CTAs and overwrites it with its gradient as its slice lands.
Eager calls launch the compiled kernels directly; under torch.compile both passes go through
torch.library custom ops.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cutlass_dsl import T

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, barrier_expect_tx, barrier_wait, lds_b32, lds_v4, packed_op, st_v4, to_ir
from pithtrain.kernels.row_slice_sm90 import RowAccess, RowSlice, lds_f32, pack_unit, sts_f32, unpack_unit

__all__ = ["log_prob_bwd_sm90", "log_prob_fwd_sm90"]

FLT_MAX, STATS_BYTES = 3.4028234663852886e38, 16
# Forward: one CTA per row up to FWD_SLICE_BYTES, else a cluster of 2, each CTA's slice streamed through
# a two-slot shared-memory ring in chunks of about FWD_CHUNK_BYTES, so a CTA needs two chunks of shared
# memory whatever the row's length and many CTAs share an SM. Backward: slices of about BWD_SLICE_BYTES,
# so many CTAs per SM keep their loads in flight.
FWD_SLICE_BYTES, FWD_CHUNK_BYTES, BWD_SLICE_BYTES = 80 * 1024, 16 * 1024, 12 * 1024
# With few rows the GPU has far more CTA slots than the grid fills, so a row is cut finer: up to
# SMALL_FWD_ROWS rows the forward uses slices of at most TINY_FWD_SLICE_BYTES (up to TINY_FWD_ROWS rows
# landing whole in one copy), up to TINY_BWD_ROWS rows the backward slices of about
# TINY_BWD_SLICE_BYTES with 128 threads.
TINY_FWD_ROWS, SMALL_FWD_ROWS, TINY_BWD_ROWS = 8, 64, 16
TINY_FWD_SLICE_BYTES, TINY_BWD_SLICE_BYTES = 40 * 1024, 2 * 1024


def exp_approx(value: Float32, use_ftz: bool) -> Float32:
    """
    exp(value) as multiply by log2(e), then ex2.approx. The ftz form differs
    from the full-range ex2 only for results below 2^-126.
    """
    ex2_op = "ex2.approx.ftz.f32" if use_ftz else "ex2.approx.f32"
    return Float32(asm(T.f32(), [to_ir(value, Float32)], "{\n.reg .f32 t;\nmul.rn.f32 t, $1, 0f3FB8AA3B;\n" + ex2_op + " $0, t;\n}", "=f,f"))


def mul_rounded(first: Float32, second: Float32) -> Float32:
    """
    first * second rounded once by mul.rn, which ptxas never fuses into a following add.
    """
    return Float32(asm(T.f32(), [to_ir(first, Float32), to_ir(second, Float32)], "mul.rn.f32 $0, $1, $2;", "=f,f,f"))


def centered_value(value: Float32, inv_temperature: Float32, shift: Float32) -> Float32:
    """
    value * inv_temperature - shift rounded once (fma.rn): z - max with one rounding, never more
    than z = x * inv_temperature, then z - max rounded twice.
    """
    return Float32(asm(T.f32(), [to_ir(value, Float32), to_ir(inv_temperature, Float32), to_ir(shift, Float32)], "{\n.reg .f32 t;\nneg.f32 t, $3;\nfma.rn.f32 $0, $1, $2, t;\n}", "=f,f,f,f"))


def pairwise_sum(values: list) -> Float32:
    """
    The sum of values as a balanced tree of adds.
    """
    while len(values) > 1:
        values = [values[i] + values[i + 1] if i + 1 < len(values) else values[i] for i in range(0, len(values), 2)]
    return values[0]


def combine_row_stats(max_a: Float32, sum_a: Float32, ent_a: Float32, max_b: Float32, sum_b: Float32, ent_b: Float32) -> tuple:
    """
    Merge two (max, d, s) triples, d = sum e and s = sum e (z - max) with e = exp(z - max). Each
    side is rescaled to the merged max, shifting s by (max_side - max) d; the shift is clamped to
    -FLT_MAX so an all -inf side, whose d and s are 0, adds 0 instead of NaN.
    """
    row_max = cute.arch.fmax(max_a, max_b)
    shift = cutlass.select_(row_max == -Float32.inf, Float32(0.0), row_max)
    delta_a, delta_b = cute.arch.fmax(max_a - shift, Float32(-FLT_MAX)), cute.arch.fmax(max_b - shift, Float32(-FLT_MAX))
    scale_a = cutlass.select_(max_a == row_max, Float32(1.0), exp_approx(delta_a, True))
    scale_b = cutlass.select_(max_b == row_max, Float32(1.0), exp_approx(delta_b, True))
    return row_max, sum_a * scale_a + sum_b * scale_b, (ent_a + delta_a * sum_a) * scale_a + (ent_b + delta_b * sum_b) * scale_b


def checked_target(target_idx: Int64, num_cols: int) -> tuple:
    """
    (index, is_valid): a target outside [0, num_cols) is replaced by num_cols before any address is
    formed, so RowSlice puts it at the row's end, which no element and no unit holds, and rank 0
    owns it.
    """
    is_valid = (target_idx >= Int64(0)) & (target_idx < Int64(num_cols))
    return cutlass.select_(is_valid, target_idx, Int64(num_cols)), is_valid


def line_aligned_units(num_units: int, num_slices: int, chunks: int) -> tuple:
    """
    (slice_units, chunk_units): a row of num_units 16-byte units cut into num_slices slices of
    chunks chunks, every slice and chunk a whole number of 128-byte lines (8 units), so the bulk
    copies of a line-aligned row start on line boundaries and no line is split between two CTAs.
    """
    slice_units = (num_units + num_slices - 1) // num_slices
    slice_units = (slice_units + 7) // 8 * 8
    chunk_units = ((slice_units + chunks - 1) // chunks + 7) // 8 * 8
    return slice_units, chunk_units


class LogProbFwd:
    """
    grid [cluster_size * N], cluster [cluster_size]: row bx // cluster_size, CTA rank bx %
    cluster_size. Each CTA folds its slice into per-thread online (max, d, s) of z = x *
    inv_temperature, merges them over the CTA and then, through shared memory of the cluster, over
    the row in rank order, so every CTA holds the same bits. Rank 0 stores entropy, row_max and
    log_sum, the CTA owning x[target] stores logp.
    """

    def __init__(self, dtype, num_cols: int, cluster_size: int, chunks: int, threads: int) -> None:
        self.dtype, self.num_cols, self.cluster_size = dtype, num_cols, cluster_size
        self.chunks, self.threads, self.warps = chunks, threads, threads // 32
        self.element_bytes = dtype.width // 8
        self.unit_elements = 16 // self.element_bytes
        self.slice_units, self.chunk_units = line_aligned_units(num_cols * self.element_bytes // 16, cluster_size, chunks)
        self.slots = min(chunks, 2)
        self.access = RowAccess(dtype, chunks, self.chunk_units)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mTarget: cute.Tensor, mStats: cute.Tensor, inv_temperature: Float32, stream) -> None:
        N = mTarget.shape[0]
        self.kernel(mX, mTarget, mStats, inv_temperature).launch(grid=[N * self.cluster_size, 1, 1], block=[self.threads, 1, 1], cluster=[self.cluster_size, 1, 1] if const_expr(self.cluster_size > 1) else None, stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mTarget: cute.Tensor, mStats: cute.Tensor, inv_temperature: Float32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row_idx, cta_rank = bx // self.cluster_size, bx % self.cluster_size
        lane, warp = tidx % 32, tidx // 32
        s_slice, s_stats, s_warp, s_target, s_barriers = self.allocate_buffers()
        self.access.init_barriers(s_barriers, tidx, self.slots + 1, self.cluster_size > 1)

        # The slice's geometry does not depend on the target: start its copy before the target's load.
        geometry = RowSlice(mX.iterator.toint(), row_idx, cta_rank, Int64(self.num_cols), self.num_cols, self.element_bytes, self.slice_units)
        for k in cutlass.range_constexpr(self.slots):
            self.issue_chunk(geometry, s_slice, s_barriers, Int32(k), tidx)
        target_idx, is_target_valid = checked_target(Int64(mTarget[row_idx]), self.num_cols)
        row = RowSlice(mX.iterator.toint(), row_idx, cta_rank, target_idx, self.num_cols, self.element_bytes, self.slice_units)
        acc_max, acc_sum, acc_ent = self.reduce_slice(row, s_slice, s_barriers, s_target, tidx, cta_rank, inv_temperature)
        cta_max, cta_sum, cta_ent = self.reduce_block(acc_max, acc_sum, acc_ent, s_warp, lane, warp)
        row_max, row_sum, row_ent = self.exchange_row_stats(cta_max, cta_sum, cta_ent, s_stats, s_barriers, tidx, lane, warp, cta_rank)
        self.store_row_outputs(row, is_target_valid, s_slice, s_target, row_max, row_sum, row_ent, mStats, mTarget.shape[0], row_idx, tidx, cta_rank, inv_temperature)
        self.sync_cluster_exit()

    def allocate_buffers(self) -> tuple:
        smem = cutlass.memory.SmemAllocator()
        ring_bytes = max(self.slots * self.chunk_units * 16, 16)
        s_slice = smem.allocate(ring_bytes, byte_alignment=128).toint()
        s_stats = smem.allocate(self.cluster_size * STATS_BYTES, byte_alignment=16).toint()
        s_warp = smem.allocate(self.warps * 16, byte_alignment=16).toint()
        s_target = smem.allocate(16, byte_alignment=16).toint()
        s_barriers = smem.allocate((self.slots + 1) * 8, byte_alignment=8).toint()
        return s_slice, s_stats, s_warp, s_target, s_barriers

    @cute.jit
    def reduce_slice(self, row: RowSlice, s_slice: Int32, s_barriers: Int32, s_target: Int32, tidx: Int32, cta_rank: Int32, inv_temperature: Float32) -> tuple:
        """
        Per-thread (max, d, s) over the thread's units, chunk by chunk through the ring, plus rank
        0's head and tail elements; the edge thread holding x[target], if any, copies it to
        s_target. Each landed chunk is read twice from shared memory: its max first, then its terms
        against that fixed shift, so the running sums are rescaled once per chunk, not per unit. A
        slot is refilled only after every thread has finished it.
        """
        acc_max, acc_sum, acc_ent = -Float32.inf, Float32(0.0), Float32(0.0)
        for k in cutlass.range(self.chunks, unroll=1):
            slot = k % self.slots
            chunk_start = k * self.chunk_units
            chunk_units = cutlass.max(cutlass.min(row.cta_units - chunk_start, Int32(self.chunk_units)), Int32(0))
            s_chunk = s_slice + slot * (self.chunk_units * 16)
            barrier_wait(s_barriers + slot * 8, (k // self.slots) % 2)
            chunk_max = self.thread_chunk_max(s_chunk, Int32(0), chunk_units, tidx)
            acc_max, shift, acc_sum, acc_ent = self.rescale(chunk_max, acc_max, acc_sum, acc_ent, inv_temperature)
            # The chunk's sums start from 0 and join the running sums once, so a thread's rounding error
            # grows with its chunks plus its units per chunk, not with all its units.
            chunk_sum, chunk_ent = Float32(0.0), Float32(0.0)
            for unit_idx in cutlass.range(tidx, chunk_units, self.threads, unroll=1):
                unit_sum, unit_ent = self.sum_terms(unpack_unit(lds_v4(s_chunk + unit_idx * 16), self.dtype), shift, inv_temperature)
                chunk_sum, chunk_ent = chunk_sum + unit_sum, chunk_ent + unit_ent
            acc_sum, acc_ent = acc_sum + chunk_sum, acc_ent + chunk_ent
            self.keep_target(row, s_chunk, s_target, chunk_start, chunk_units, tidx)
            if k + self.slots < self.chunks:
                cute.arch.sync_threads()
                self.issue_chunk(row, s_slice, s_barriers, k + self.slots, tidx)
        if cta_rank == 0:
            edge_addr, edge_ok = self.access.edge_element(row, tidx)
            edge_value = self.access.load_element(edge_addr, edge_ok)
            acc_max, acc_sum, acc_ent = self.accumulate([edge_value], edge_value, acc_max, acc_sum, acc_ent, inv_temperature)
            if edge_ok & (edge_addr == row.target_addr):
                sts_f32(s_target, edge_value)
        return acc_max, acc_sum, acc_ent

    @cute.jit
    def issue_chunk(self, row: RowSlice, s_slice: Int32, s_barriers: Int32, chunk_idx: Int32, tidx: Int32) -> None:
        """
        One thread starts the bulk copy of chunk chunk_idx of the slice into ring slot chunk_idx %
        slots. The fence orders the threads' earlier shared-memory reads of that slot (made visible
        by the caller's barrier) before the copy's async-proxy writes.
        """
        if tidx == 0:
            slot = chunk_idx % self.slots
            chunk_units = cutlass.max(cutlass.min(row.cta_units - chunk_idx * self.chunk_units, Int32(self.chunk_units)), Int32(0))
            asm(None, [], "fence.proxy.async.shared::cta;", "", side_effects=True)
            barrier_expect_tx(s_barriers + slot * 8, chunk_units * 16)
            if chunk_units > 0:
                asm(None, [to_ir(s_slice + slot * (self.chunk_units * 16), Int32), to_ir(row.slice_start + Int64(chunk_idx) * Int64(self.chunk_units * 16), Int64), to_ir(chunk_units * 16, Int32), to_ir(s_barriers + slot * 8, Int32)], "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];", "r,l,r,r", side_effects=True)

    @cute.jit
    def keep_target(self, row: RowSlice, s_chunk: Int32, s_target: Int32, chunk_start: Int32, chunk_units: Int32, tidx: Int32) -> None:
        """
        Thread 0 copies x[target] to s_target while its chunk is in the ring, before the slot is
        reused.
        """
        if (tidx == 0) & (row.target_unit >= chunk_start) & (row.target_unit < chunk_start + chunk_units):
            elements_per_word = 4 // self.element_bytes
            word = lds_b32(s_chunk + (row.target_unit - chunk_start) * 16 + (row.target_element // elements_per_word) * 4)
            halves = unpack_unit((word,), self.dtype)
            value = halves[0]
            if const_expr(elements_per_word == 2):
                value = cutlass.select_(row.target_element % 2 == 1, halves[1], halves[0])
            sts_f32(s_target, value)

    @cute.jit
    def thread_chunk_max(self, s_slice: Int32, chunk_start: Int32, chunk_units: Int32, tidx: Int32) -> Float32:
        """
        The largest logit of the thread's units of a landed chunk (-inf for none); 16-bit pairs stay
        packed while scanning, so a unit costs its load and three packed max.
        """
        best = -Float32.inf
        if const_expr(self.dtype is cutlass.Float32):
            for unit_idx in cutlass.range(chunk_start + tidx, chunk_start + chunk_units, self.threads, unroll=1):
                for value in unpack_unit(lds_v4(s_slice + unit_idx * 16), self.dtype):
                    best = cute.arch.fmax(best, value)
        else:
            max_op = "max.bf16x2" if const_expr(self.dtype is cutlass.BFloat16) else "max.f16x2"
            packed = Uint32(0xFF80FF80 if const_expr(self.dtype is cutlass.BFloat16) else 0xFC00FC00)
            for unit_idx in cutlass.range(chunk_start + tidx, chunk_start + chunk_units, self.threads, unroll=1):
                words = lds_v4(s_slice + unit_idx * 16)
                packed = packed_op(max_op, packed, packed_op(max_op, packed_op(max_op, words[0], words[1]), packed_op(max_op, words[2], words[3])))
            pair = unpack_unit((packed,), self.dtype)
            best = cute.arch.fmax(pair[0], pair[1])
        return best

    @cute.jit
    def rescale(self, values_max: Float32, acc_max: Float32, acc_sum: Float32, acc_ent: Float32, inv_temperature: Float32) -> tuple:
        """
        Move the running (max, d, s) to the max that includes values_max (raw logits; x -> x *
        inv_temperature is monotonic, so the max is taken before tempering). Returns (max, shift, d,
        s): shift is 0 while the max is -inf, so a slice of only -inf so far keeps d = s = 0.
        """
        new_max = cute.arch.fmax(acc_max, mul_rounded(values_max, inv_temperature))
        shift = cutlass.select_(new_max == -Float32.inf, Float32(0.0), new_max)
        delta = cute.arch.fmax(acc_max - shift, Float32(-FLT_MAX))
        scale = exp_approx(delta, True)
        return new_max, shift, acc_sum * scale, (acc_ent + delta * acc_sum) * scale

    def sum_terms(self, values: list, shift: Float32, inv_temperature: Float32) -> tuple:
        """
        (sum e, sum e (z - shift)) over values, e = exp(z - shift), each a balanced tree, so a
        thread's rounding error grows with its units rather than its elements. -inf logits give z -
        shift = -inf, clamped to -FLT_MAX, so they add e = 0 and e (z - shift) = 0 instead of NaN.
        """
        centered = [cute.arch.fmax(centered_value(value, inv_temperature, shift), Float32(-FLT_MAX)) for value in values]
        terms = [exp_approx(value, True) for value in centered]
        return pairwise_sum(terms), pairwise_sum([term * value for term, value in zip(terms, centered)])

    @cute.jit
    def accumulate(self, values: list, values_max: Float32, acc_max: Float32, acc_sum: Float32, acc_ent: Float32, inv_temperature: Float32) -> tuple:
        """
        Fold values (raw logits, values_max their max) into the running (max, d, s), rescaling once.
        """
        acc_max, shift, acc_sum, acc_ent = self.rescale(values_max, acc_max, acc_sum, acc_ent, inv_temperature)
        unit_sum, unit_ent = self.sum_terms(values, shift, inv_temperature)
        return acc_max, acc_sum + unit_sum, acc_ent + unit_ent

    @cute.jit
    def reduce_block(self, acc_max: Float32, acc_sum: Float32, acc_ent: Float32, s_warp: Int32, lane: Int32, warp: Int32) -> tuple:
        """
        Butterfly within each warp with lane 0's triple broadcast, then every thread merges the
        warps' triples in warp order, so all threads of the CTA hold the same bits.
        """
        for i in cutlass.range_constexpr(5):
            peer_max, peer_sum, peer_ent = cute.arch.shuffle_sync_bfly(acc_max, offset=16 >> i), cute.arch.shuffle_sync_bfly(acc_sum, offset=16 >> i), cute.arch.shuffle_sync_bfly(acc_ent, offset=16 >> i)
            acc_max, acc_sum, acc_ent = combine_row_stats(acc_max, acc_sum, acc_ent, peer_max, peer_sum, peer_ent)
        acc_max, acc_sum, acc_ent = cute.arch.shuffle_sync(acc_max, offset=0), cute.arch.shuffle_sync(acc_sum, offset=0), cute.arch.shuffle_sync(acc_ent, offset=0)
        if lane == 0:
            asm(None, [to_ir(s_warp + warp * 16, Int32), to_ir(acc_max, Float32), to_ir(acc_sum, Float32), to_ir(acc_ent, Float32)], "st.shared.v4.f32 [$0], {$1, $2, $3, $3};", "r,f,f,f", side_effects=True)
        cute.arch.sync_threads()
        cta_max, cta_sum, cta_ent = -Float32.inf, Float32(0.0), Float32(0.0)
        for i in cutlass.range_constexpr(self.warps):
            cta_max, cta_sum, cta_ent = combine_row_stats(cta_max, cta_sum, cta_ent, lds_f32(s_warp + i * 16), lds_f32(s_warp + i * 16 + 4), lds_f32(s_warp + i * 16 + 8))
        return cta_max, cta_sum, cta_ent

    @cute.jit
    def exchange_row_stats(self, cta_max: Float32, cta_sum: Float32, cta_ent: Float32, s_stats: Int32, s_barriers: Int32, tidx: Int32, lane: Int32, warp: Int32, cta_rank: Int32) -> tuple:
        """
        Every CTA sends its triple to slot cta_rank of every peer's s_stats with st.async (which may
        not target the issuing CTA), after the cluster wait that guarantees the peers' barriers are
        initialized, and merges all slots in rank order, its own from registers, so the cluster
        agrees on the row's (max, d, s). A peer's complete_tx may land before this CTA's expect_tx:
        the tx-count is signed and the phase completes only after that arrive.
        """
        row_max, row_sum, row_ent = cta_max, cta_sum, cta_ent
        if const_expr(self.cluster_size > 1):
            stats_barrier = s_barriers + self.slots * 8
            cute.arch.cluster_wait()
            if tidx == 0:
                barrier_expect_tx(stats_barrier, Int32((self.cluster_size - 1) * STATS_BYTES))
            if (warp == 0) & (lane < self.cluster_size) & (lane != cta_rank):
                asm(None, [to_ir(s_stats + cta_rank * STATS_BYTES, Int32), to_ir(stats_barrier, Int32), to_ir(lane, Int32), to_ir(cta_max, Float32), to_ir(cta_sum, Float32), to_ir(cta_ent, Float32)], "{\n.reg .b32 remote_slot, remote_barrier;\nmapa.shared::cluster.u32 remote_slot, $0, $2;\nmapa.shared::cluster.u32 remote_barrier, $1, $2;\nst.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [remote_slot], {$3, $4, $5, $5}, [remote_barrier];\n}", "r,r,r,f,f,f", side_effects=True)
            barrier_wait(stats_barrier, Int32(0))
            row_max, row_sum, row_ent = -Float32.inf, Float32(0.0), Float32(0.0)
            for i in cutlass.range_constexpr(self.cluster_size):
                is_own = cta_rank == i
                slot_max = cutlass.select_(is_own, cta_max, lds_f32(s_stats + i * STATS_BYTES))
                slot_sum = cutlass.select_(is_own, cta_sum, lds_f32(s_stats + i * STATS_BYTES + 4))
                slot_ent = cutlass.select_(is_own, cta_ent, lds_f32(s_stats + i * STATS_BYTES + 8))
                row_max, row_sum, row_ent = combine_row_stats(row_max, row_sum, row_ent, slot_max, slot_sum, slot_ent)
            cute.arch.cluster_arrive_relaxed()
        return row_max, row_sum, row_ent

    @cute.jit
    def store_row_outputs(self, row: RowSlice, is_target_valid, s_slice: Int32, s_target: Int32, row_max: Float32, row_sum: Float32, row_ent: Float32, mStats: cute.Tensor, num_rows: Int32, row_idx: Int32, tidx: Int32, cta_rank: Int32, inv_temperature: Float32) -> None:
        """
        log_sum = log(d) by the full-precision log (once per row; lg2.approx would cost about 1e-7
        near d = 1), entropy = log_sum - s / d, logp = (z[target] - max) - log_sum, so a dominant
        target gives 0; 0 for an invalid target.
        """
        if tidx == 0:
            log_sum = cute.math.log(row_sum, fastmath=False)
            if cta_rank == 0:
                mStats[num_rows + row_idx] = log_sum - Float32(asm(T.f32(), [to_ir(row_ent, Float32), to_ir(row_sum, Float32)], "div.rn.f32 $0, $1, $2;", "=f,f,f"))
                mStats[2 * num_rows + row_idx] = row_max
                mStats[3 * num_rows + row_idx] = log_sum
            if row.owns_target:
                logp = Float32(0.0)
                if is_target_valid:
                    logp = centered_value(self.load_target(row, s_slice, s_target), inv_temperature, row_max) - log_sum
                mStats[row_idx] = logp

    @cute.jit
    def load_target(self, row: RowSlice, s_slice: Int32, s_target: Int32) -> Float32:
        """
        x[target], read once after the reduction instead of being picked out of every unit: from
        s_target, where keep_target or the head / tail thread put it.
        """
        return lds_f32(s_target)

    @cute.jit
    def sync_cluster_exit(self) -> None:
        """
        No CTA exits while a peer may still write to its shared memory.
        """
        if const_expr(self.cluster_size > 1):
            cute.arch.cluster_wait()


class GradTerms:
    """
    The row's constants of dx = p (coef_log log p + coef_const), plus target_add at the target,
    with log p = (x * inv_temperature - row_max) - log_sum.
    """

    def __init__(self, row_max: Float32, log_sum: Float32, coef_log: Float32, coef_const: Float32, target_add: Float32, inv_temperature: Float32) -> None:
        self.row_max, self.log_sum, self.coef_log, self.coef_const = row_max, log_sum, coef_log, coef_const
        self.target_add, self.inv_temperature = target_add, inv_temperature


class LogProbBwd:
    """
    grid [split * N], no cluster: CTA bx overwrites slice bx % split of row bx // split. Each CTA
    reads only its own slice (and rank 0 the head and tail), all of it before it stores, and the
    thread storing the target's unit rewrites the target element from its own registers, so the
    in-place update has no race.
    """

    def __init__(self, dtype, num_cols: int, split: int, chunks: int, threads: int) -> None:
        self.dtype, self.num_cols, self.split, self.chunks, self.threads = dtype, num_cols, split, chunks, threads
        self.element_bytes = dtype.width // 8
        self.unit_elements = 16 // self.element_bytes
        self.slice_units, self.chunk_units = line_aligned_units(num_cols * self.element_bytes // 16, split, chunks)
        self.access = RowAccess(dtype, chunks, self.chunk_units)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mTarget: cute.Tensor, mRowMax: cute.Tensor, mLogSum: cute.Tensor, mEntropy: cute.Tensor, mGradLogp: cute.Tensor, mGradEntropy: cute.Tensor, inv_temperature: Float32, stream) -> None:
        N = mTarget.shape[0]
        self.kernel(mX, mTarget, mRowMax, mLogSum, mEntropy, mGradLogp, mGradEntropy, inv_temperature).launch(grid=[N * self.split, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mTarget: cute.Tensor, mRowMax: cute.Tensor, mLogSum: cute.Tensor, mEntropy: cute.Tensor, mGradLogp: cute.Tensor, mGradEntropy: cute.Tensor, inv_temperature: Float32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row_idx, cta_rank = bx // self.split, bx % self.split
        smem = cutlass.memory.SmemAllocator()
        s_slice = smem.allocate(max(self.slice_units * 16, 16), byte_alignment=128).toint()
        s_barriers = smem.allocate(self.chunks * 8, byte_alignment=8).toint()
        self.access.init_barriers(s_barriers, tidx, self.chunks, False)

        target_idx, is_target_valid = checked_target(Int64(mTarget[row_idx]), self.num_cols)
        row = RowSlice(mX.iterator.toint(), row_idx, cta_rank, target_idx, self.num_cols, self.element_bytes, self.slice_units)
        self.access.issue_slice_loads(row, s_slice, s_barriers, tidx)
        terms = self.load_grad_terms(is_target_valid, mRowMax, mLogSum, mEntropy, mGradLogp, mGradEntropy, row_idx, inv_temperature)
        self.store_slice_grad(row, s_slice, s_barriers, terms, tidx)
        self.store_edge_grad(row, terms, tidx, cta_rank)

    @cute.jit
    def load_grad_terms(self, is_target_valid, mRowMax: cute.Tensor, mLogSum: cute.Tensor, mEntropy: cute.Tensor, mGradLogp: cute.Tensor, mGradEntropy: cute.Tensor, row_idx: Int32, inv_temperature: Float32) -> GradTerms:
        """
        An invalid target has constant log p = 0, so its grad_logp is dropped.
        """
        grad_logp = cutlass.select_(is_target_valid, mGradLogp[row_idx], Float32(0.0))
        grad_entropy = mGradEntropy[row_idx]
        coef_log = -grad_entropy * inv_temperature
        coef_const = -(grad_logp + grad_entropy * mEntropy[row_idx]) * inv_temperature
        # Rounded on its own, so the target's p (...) + target_add cancels exactly when p = 1.
        return GradTerms(mRowMax[row_idx], mLogSum[row_idx], coef_log, coef_const, mul_rounded(grad_logp, inv_temperature), inv_temperature)

    @cute.jit
    def grad_of(self, value: Float32, terms: GradTerms) -> Float32:
        """
        log p is clamped below at -1e4 only where p = 0 already, so a -inf logit gets 0, not NaN.
        """
        logp = centered_value(value, terms.inv_temperature, terms.row_max) - terms.log_sum
        prob = exp_approx(logp, False)
        return prob * (terms.coef_log * cute.arch.fmax(logp, Float32(-1e4)) + terms.coef_const)

    @cute.jit
    def store_slice_grad(self, row: RowSlice, s_slice: Int32, s_barriers: Int32, terms: GradTerms, tidx: Int32) -> None:
        """
        As each chunk lands, 16-byte gradient stores over the thread's units; the thread holding the
        target's unit then rewrites the target element with the target_add term.
        """
        for k in cutlass.range_constexpr(self.chunks):
            chunk_start = k * self.chunk_units
            chunk_units = cutlass.max(cutlass.min(row.cta_units - chunk_start, Int32(self.chunk_units)), Int32(0))
            barrier_wait(s_barriers + k * 8, Int32(0))
            for unit_idx in cutlass.range(chunk_start + tidx, chunk_start + chunk_units, self.threads, unroll=1):
                values = unpack_unit(lds_v4(s_slice + unit_idx * 16), self.dtype)
                grads = [self.grad_of(value, terms) for value in values]
                st_v4(row.slice_start + Int64(unit_idx) * Int64(16), pack_unit(grads, self.dtype), Int32(1))
                if unit_idx == row.target_unit:
                    target_value = values[0]
                    for i in cutlass.range_constexpr(1, self.unit_elements):
                        target_value = cutlass.select_(row.target_element == i, values[i], target_value)
                    self.access.store_element(row.target_addr, self.grad_of(target_value, terms) + terms.target_add, True)

    @cute.jit
    def store_edge_grad(self, row: RowSlice, terms: GradTerms, tidx: Int32, cta_rank: Int32) -> None:
        if cta_rank == 0:
            edge_addr, edge_ok = self.access.edge_element(row, tidx)
            edge_grad = self.grad_of(self.access.load_element(edge_addr, edge_ok), terms)
            self.access.store_element(edge_addr, cutlass.select_(edge_addr == row.target_addr, edge_grad + terms.target_add, edge_grad), edge_ok)


TARGET_TYPES = {torch.int32: Int32, torch.int64: Int64}
COMPILED: dict = {}


def pick_fwd_layout(num_cols: int, element_bytes: int, num_rows: int) -> tuple:
    """
    (cluster_size, chunks, threads) of the forward for num_rows rows of num_cols elements: each
    CTA's slice streams through the two-slot ring in chunks of about FWD_CHUNK_BYTES, or, up to
    TINY_FWD_ROWS rows, lands whole in one copy.
    """
    row_bytes = num_cols * element_bytes
    if num_rows <= SMALL_FWD_ROWS:
        cluster_size = next((size for size in (1, 2, 4, 8) if row_bytes <= size * TINY_FWD_SLICE_BYTES), 8)
        if num_rows <= TINY_FWD_ROWS:
            return cluster_size, 1, 256
        return cluster_size, max(1, -(-row_bytes // (cluster_size * FWD_CHUNK_BYTES))), 128
    cluster_size = 1 if row_bytes <= FWD_SLICE_BYTES else 2
    return cluster_size, max(1, -(-row_bytes // (cluster_size * FWD_CHUNK_BYTES))), 128


def pick_bwd_layout(num_cols: int, element_bytes: int, num_rows: int) -> tuple:
    """
    (split, chunks, threads) of the backward for num_rows rows of num_cols elements.
    """
    if num_rows <= TINY_BWD_ROWS:
        return max(1, -(-num_cols * element_bytes // TINY_BWD_SLICE_BYTES)), 1, 128
    return max(1, -(-num_cols * element_bytes // BWD_SLICE_BYTES)), 1, 256


def compile_fwd(dtype: torch.dtype, target_dtype: torch.dtype, num_cols: int, cluster_size: int, chunks: int, threads: int) -> object:
    x = fake(TORCH2CUTE[dtype], (cute.sym_int(), num_cols), (num_cols, 1), dtype.itemsize)
    target = fake(TARGET_TYPES[target_dtype], (cute.sym_int(),), (1,), target_dtype.itemsize)
    stats = fake(Float32, (cute.sym_int(),), (1,), 4)
    return cute.compile(LogProbFwd(TORCH2CUTE[dtype], num_cols, cluster_size, chunks, threads), x, target, stats, Float32(1.0), fake_stream(), options="--enable-tvm-ffi")


def compile_bwd(dtype: torch.dtype, target_dtype: torch.dtype, num_cols: int, split: int, chunks: int, threads: int) -> object:
    x = fake(TORCH2CUTE[dtype], (cute.sym_int(), num_cols), (num_cols, 1), dtype.itemsize)
    target = fake(TARGET_TYPES[target_dtype], (cute.sym_int(),), (1,), target_dtype.itemsize)
    row_inputs = [fake(Float32, (cute.sym_int(),), (1,), 4) for _ in range(5)]
    return cute.compile(LogProbBwd(TORCH2CUTE[dtype], num_cols, split, chunks, threads), x, target, *row_inputs, Float32(1.0), fake_stream(), options="--enable-tvm-ffi")


def compiled_fwd(inp: torch.Tensor, target: torch.Tensor) -> object:
    """
    The compiled forward for these inputs, built on first use; one dict lookup afterwards.
    """
    num_rows, num_cols = inp.shape
    key = ("log_prob_fwd", inp.dtype, target.dtype, num_cols, (num_rows > TINY_FWD_ROWS) + (num_rows > SMALL_FWD_ROWS))
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_fwd(inp.dtype, target.dtype, num_cols, *pick_fwd_layout(num_cols, inp.element_size(), num_rows))
    return fn


def compiled_bwd(inp: torch.Tensor, target: torch.Tensor) -> object:
    num_rows, num_cols = inp.shape
    key = ("log_prob_bwd", inp.dtype, target.dtype, num_cols, num_rows <= TINY_BWD_ROWS)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_bwd(inp.dtype, target.dtype, num_cols, *pick_bwd_layout(num_cols, inp.element_size(), num_rows))
    return fn


def as_target(target: torch.Tensor) -> torch.Tensor:
    """
    The target as an int32 or int64 tensor of stride 1, converting other integer dtypes to int64.
    """
    if target.dtype not in TARGET_TYPES:
        target = target.to(torch.int64)
    return target if target.is_contiguous() else target.contiguous()


def as_row_vector(value: torch.Tensor) -> torch.Tensor:
    """
    An (N,) fp32 tensor of stride 1 (stride-0 upstream gradients included).
    """
    if value.dtype != torch.float32:
        value = value.float()
    return value if value.is_contiguous() else value.contiguous()


def launch_fwd(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> torch.Tensor:
    """
    One (4, N) fp32 tensor holding logp, entropy, row_max and log_sum in its rows.
    """
    stats = torch.empty((4, inp.shape[0]), dtype=torch.float32, device=inp.device)
    if inp.shape[0] > 0:
        target = as_target(target)
        compiled_fwd(inp, target)(inp, target, stats.view(-1), inv_temperature)
    return stats


def launch_bwd(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    if inp.shape[0] > 0:
        target = as_target(target)
        compiled_bwd(inp, target)(inp, target, as_row_vector(row_max), as_row_vector(log_sum), as_row_vector(entropy), as_row_vector(grad_logp), as_row_vector(grad_entropy), inv_temperature)


@torch.library.custom_op("pithtrain::log_prob_fwd", mutates_args=())
def log_prob_stats_op(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> torch.Tensor:
    """
    The forward as a custom op, for torch.compile graphs.
    """
    return launch_fwd(inp, target, inv_temperature)


@log_prob_stats_op.register_fake
def fake_log_prob_stats_op(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> torch.Tensor:
    return inp.new_empty((4, inp.shape[0]), dtype=torch.float32)


@torch.library.custom_op("pithtrain::log_prob_bwd", mutates_args=("inp",))
def log_prob_grad_op(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    """
    The backward as a custom op, for torch.compile graphs; the declared mutation of inp bumps its
    version counter once.
    """
    launch_bwd(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)


@log_prob_grad_op.register_fake
def fake_log_prob_grad_op(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    return None


def log_prob_fwd_sm90(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The picker's contract on SM90, every dtype and shape through the CuTe kernels. Eager calls
    launch the cached kernel directly; only a torch.compile trace goes through the custom op, whose
    dispatch costs host time in eager.
    """
    if torch.compiler.is_compiling():
        return log_prob_stats_op(inp, target, inv_temperature).unbind(0)
    return launch_fwd(inp, target, inv_temperature).unbind(0)


def log_prob_bwd_sm90(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    """
    The picker's contract on SM90. The version counter of inp is bumped once: by the custom op's
    declared mutation under torch.compile, explicitly after the direct launch in eager.
    """
    if torch.compiler.is_compiling():
        log_prob_grad_op(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)
        return
    launch_bwd(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)
    torch.autograd.graph.increment_version(inp)
