"""
Fused cross-entropy forward and backward on SM90 (CuTe-DSL), reading and writing the logits once:
one row per cluster of CTAs, each CTA's slice of the row kept in shared memory between the reduction
and the gradient pass.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, barrier_expect_tx, barrier_wait, f32_as_u32, ld_v4, lds_b32, lds_v4, st_v4, sts_b32, to_ir
from pithtrain.kernels.row_slice_sm90 import RowAccess, RowSlice, lds_f32, pack_unit, sts_f32, unit_extremes, unpack_unit

__all__ = ["cross_entropy_fwd_bwd_sm90"]

COUNT_THREADS = 1024
STATS_BYTES = 8
LN2 = 0.6931471805599453
# Largest row slice one CTA keeps in shared memory, and the slice size up to which a smaller cluster
# is preferred: about 80 KB leaves room for 2-3 CTAs per SM.
MAX_SLICE_BYTES, PREFERRED_SLICE_BYTES = 200 * 1024, 80 * 1024


@dsl_user_op
def exp_shifted(x: Float32, shift: Float32, use_ftz: bool, *, loc=None, ip=None) -> Float32:
    """
    exp(x - shift) as sub, multiply by log2(e), then ex2.approx. The ftz form differs from the
    full-range ex2 only for results below 2^-126.
    """
    ex2_op = "ex2.approx.ftz.f32" if use_ftz else "ex2.approx.f32"
    return Float32(asm(T.f32(), [to_ir(x, Float32, loc, ip), to_ir(shift, Float32, loc, ip)], "{\n.reg .f32 t;\nsub.rn.f32 t, $1, $2;\nmul.rn.f32 t, t, 0f3FB8AA3B;\n" + ex2_op + " $0, t;\n}", "=f,f,f"))


def combine_stats(max_a: Float32, sum_a: Float32, max_b: Float32, sum_b: Float32) -> tuple:
    """
    Merge two (max, sum) pairs, each sum of exp(x - max); a factor below 2^-126 scales a sum that is
    negligible next to the other, which holds the max's term of 1.
    """
    row_max = cute.arch.fmax(max_a, max_b)
    scale_a = cutlass.select_(max_a == row_max, Float32(1.0), exp_shifted(max_a, row_max, True))
    scale_b = cutlass.select_(max_b == row_max, Float32(1.0), exp_shifted(max_b, row_max, True))
    return row_max, sum_a * scale_a + sum_b * scale_b


@cute.jit
def block_sum(value, s_partial: Int32, lane: Int32, warp: Int32, num_warps: cutlass.Constexpr[int]):
    """
    The CTA's sum of value (Int32 or Float32) in a fixed order, valid in thread 0: a butterfly
    within each warp, then thread 0 adds the warps' sums in warp order.
    """
    is_float = isinstance(value, Float32)
    for i in cutlass.range_constexpr(5):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=16 >> i)
    if lane == 0:
        sts_b32(s_partial + warp * 4, f32_as_u32(value) if const_expr(is_float) else Uint32(value), Int32(1))
    cute.arch.sync_threads()
    total = Float32(0.0) if const_expr(is_float) else Int32(0)
    if warp == 0:
        for i in cutlass.range_constexpr(num_warps):
            total = total + (lds_f32(s_partial + i * 4) if const_expr(is_float) else Int32(lds_b32(s_partial + i * 4)))
    return total


class RowStats:
    """
    The row's max and sum of exp(x - max), the same bits in every CTA of the cluster, and the
    smallest element of this CTA's part.
    """

    def __init__(self, row_max: Float32, row_sum: Float32, cta_min: Float32) -> None:
        self.row_max, self.row_sum, self.cta_min = row_max, row_sum, cta_min


class CrossEntropy:
    """
    grid [cluster_size * N], cluster [cluster_size]: row bx // cluster_size, CTA rank bx %
    cluster_size. Every element of a row has exactly one owning CTA, and each CTA stores only after
    the cluster has exchanged the row's statistics, which each CTA sends only after it has read its
    slice, so x[target] is always read before any store.
    """

    def __init__(self, dtype, num_cols: int, cluster_size: int, chunks: int, is_resident: bool, threads: int, mean: bool) -> None:
        self.dtype, self.num_cols, self.cluster_size = dtype, num_cols, cluster_size
        self.chunks, self.is_resident, self.threads = chunks, is_resident, threads
        self.mean = mean
        self.warps = threads // 32
        self.element_bytes = dtype.width // 8
        self.unit_elements = 16 // self.element_bytes
        self.slice_units = (num_cols * self.element_bytes // 16 + cluster_size - 1) // cluster_size
        self.chunk_units = (self.slice_units + chunks - 1) // chunks
        self.access = RowAccess(dtype, chunks, self.chunk_units)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mTarget: cute.Tensor, mLoss: cute.Tensor, mCount: cute.Tensor, mMeanLoss: cute.Tensor, ignore_index: Int64, stream) -> None:
        N = mTarget.shape[0]
        CountNonIgnored(self.mean).kernel(mTarget, mCount, ignore_index).launch(grid=[1, 1, 1], block=[COUNT_THREADS, 1, 1], stream=stream)
        self.kernel(mX, mTarget, mLoss, mCount, ignore_index).launch(grid=[N * self.cluster_size, 1, 1], block=[self.threads, 1, 1], cluster=[self.cluster_size, 1, 1] if const_expr(self.cluster_size > 1) else None, stream=stream, use_pdl=True)
        MeanLoss().kernel(mLoss, mCount, mMeanLoss).launch(grid=[1, 1, 1], block=[COUNT_THREADS, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mTarget: cute.Tensor, mLoss: cute.Tensor, mCount: cute.Tensor, ignore_index: Int64) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        cute.arch.griddepcontrol_launch_dependents()
        bx, _, _ = cute.arch.block_idx()
        row_idx, cta_rank = bx // self.cluster_size, bx % self.cluster_size
        lane, warp = tidx % 32, tidx // 32
        s_slice, s_stats, s_warp, s_target, s_barriers = self.allocate_buffers()
        self.access.init_barriers(s_barriers, tidx, self.chunks + 1, self.cluster_size > 1)

        target_idx = Int64(mTarget[row_idx])
        row = RowSlice(mX.iterator.toint(), row_idx, cta_rank, target_idx, self.num_cols, self.element_bytes, self.slice_units)
        if target_idx == ignore_index:
            self.store_zero_row(row, mLoss, row_idx, tidx, cta_rank)
            self.sync_cluster_exit()
        else:
            if const_expr(self.is_resident):
                self.access.issue_slice_loads(row, s_slice, s_barriers, tidx)
            acc_max, acc_sum, acc_min, edge_value = self.reduce_slice(row, s_slice, s_barriers, s_target, tidx, cta_rank)
            cta_max, cta_sum, cta_min = self.reduce_block(acc_max, acc_sum, acc_min, s_warp, lane, warp)
            row_max, row_sum = self.exchange_row_stats(cta_max, cta_sum, s_stats, s_barriers, tidx, lane, warp, cta_rank)
            stats = RowStats(row_max, row_sum, cta_min)
            cute.arch.griddepcontrol_wait()
            num_counted = Float32(Int64(mCount[0]))
            self.store_grad(row, s_slice, s_target, edge_value, stats, num_counted, tidx, cta_rank)
            self.store_row_loss(row, s_target, stats, mLoss, row_idx, tidx)
            self.sync_cluster_exit()

    def allocate_buffers(self) -> tuple:
        smem = cutlass.memory.SmemAllocator()
        slice_bytes = max(self.slice_units * 16, 16) if const_expr(self.is_resident) else 16
        s_slice = smem.allocate(slice_bytes, byte_alignment=128).toint()
        s_stats = smem.allocate(self.cluster_size * STATS_BYTES, byte_alignment=16).toint()
        s_warp = smem.allocate(self.warps * 16, byte_alignment=16).toint()
        s_target = smem.allocate(16, byte_alignment=16).toint()
        s_barriers = smem.allocate((self.chunks + 1) * 8, byte_alignment=8).toint()
        return s_slice, s_stats, s_warp, s_target, s_barriers

    @cute.jit
    def load_unit(self, row: RowSlice, s_slice: Int32, unit_idx: Int32) -> tuple:
        if const_expr(self.is_resident):
            return lds_v4(s_slice + unit_idx * 16)
        return ld_v4(row.slice_start + Int64(unit_idx) * Int64(16), Int32(1))

    @cute.jit
    def reduce_slice(self, row: RowSlice, s_slice: Int32, s_barriers: Int32, s_target: Int32, tidx: Int32, cta_rank: Int32) -> tuple:
        """
        Per-thread online (max, sum) and min over the thread's units as each chunk lands, plus rank
        0's head and tail elements; whoever holds x[target] copies it to s_target.
        """
        acc_max, acc_sum, acc_min = -Float32.inf, Float32(0.0), Float32.inf
        for chunk_idx in cutlass.range_constexpr(self.chunks):
            chunk_start = chunk_idx * self.chunk_units
            num_units = cutlass.max(cutlass.min(row.cta_units - chunk_start, Int32(self.chunk_units)), Int32(0))
            if const_expr(self.is_resident):
                barrier_wait(s_barriers + chunk_idx * 8, Int32(0))
            for unit_idx in cutlass.range(chunk_start + tidx, chunk_start + num_units, self.threads, unroll=1):
                words = self.load_unit(row, s_slice, unit_idx)
                values = unpack_unit(words, self.dtype)
                acc_max, acc_sum, acc_min = self.accumulate_unit(words, values, acc_max, acc_sum, acc_min)
                if unit_idx == row.target_unit:
                    target_value = values[0]
                    for i in cutlass.range_constexpr(1, self.unit_elements):
                        target_value = cutlass.select_(row.target_element == i, values[i], target_value)
                    sts_f32(s_target, target_value)
        edge_value = -Float32.inf
        if cta_rank == 0:
            edge_addr, edge_ok = self.access.edge_element(row, tidx)
            edge_value = self.access.load_element(edge_addr, edge_ok)
            acc_max, acc_sum = combine_stats(acc_max, acc_sum, edge_value, cutlass.select_(edge_ok, Float32(1.0), Float32(0.0)))
            acc_min = cutlass.select_(edge_ok, cute.arch.fmin(acc_min, edge_value), acc_min)
            if edge_ok & (edge_addr == row.target_addr):
                sts_f32(s_target, edge_value)
        return acc_max, acc_sum, acc_min, edge_value

    @cute.jit
    def accumulate_unit(self, words: tuple, values: list, acc_max: Float32, acc_sum: Float32, acc_min: Float32) -> tuple:
        """
        The running sum is rescaled once per unit, only when the max moves. The exponentials use the
        ftz ex2: a term below 2^-126 cannot change a sum that holds its max's term of 1.
        """
        unit_max, unit_min = unit_extremes(words, self.dtype)
        new_max = cute.arch.fmax(acc_max, unit_max)
        shift = cutlass.select_(new_max == -Float32.inf, Float32(0.0), new_max)
        acc_sum = acc_sum * cutlass.select_(new_max == acc_max, Float32(1.0), exp_shifted(acc_max, shift, True))
        for value in values:
            acc_sum = acc_sum + exp_shifted(value, shift, True)
        return new_max, acc_sum, cute.arch.fmin(acc_min, unit_min)

    @cute.jit
    def reduce_block(self, acc_max: Float32, acc_sum: Float32, acc_min: Float32, s_warp: Int32, lane: Int32, warp: Int32) -> tuple:
        """
        Butterfly within each warp with lane 0's pair broadcast, then every thread merges the warps'
        pairs in warp order, so all threads of the CTA hold the same bits.
        """
        for i in cutlass.range_constexpr(5):
            peer_max, peer_sum = cute.arch.shuffle_sync_bfly(acc_max, offset=16 >> i), cute.arch.shuffle_sync_bfly(acc_sum, offset=16 >> i)
            acc_max, acc_sum = combine_stats(acc_max, acc_sum, peer_max, peer_sum)
            acc_min = cute.arch.fmin(acc_min, cute.arch.shuffle_sync_bfly(acc_min, offset=16 >> i))
        acc_max, acc_sum = cute.arch.shuffle_sync(acc_max, offset=0), cute.arch.shuffle_sync(acc_sum, offset=0)
        if lane == 0:
            asm(None, [to_ir(s_warp + warp * 16, Int32), to_ir(acc_max, Float32), to_ir(acc_sum, Float32), to_ir(acc_min, Float32)], "st.shared.v4.f32 [$0], {$1, $2, $3, $3};", "r,f,f,f", side_effects=True)
        cute.arch.sync_threads()
        cta_max, cta_sum, cta_min = -Float32.inf, Float32(0.0), Float32.inf
        for i in cutlass.range_constexpr(self.warps):
            cta_max, cta_sum = combine_stats(cta_max, cta_sum, lds_f32(s_warp + i * 16), lds_f32(s_warp + i * 16 + 4))
            cta_min = cute.arch.fmin(cta_min, lds_f32(s_warp + i * 16 + 8))
        return cta_max, cta_sum, cta_min

    @cute.jit
    def exchange_row_stats(self, cta_max: Float32, cta_sum: Float32, s_stats: Int32, s_barriers: Int32, tidx: Int32, lane: Int32, warp: Int32, cta_rank: Int32) -> tuple:
        """
        Every CTA sends its pair to slot cta_rank of every CTA's s_stats (itself included) with
        st.async, and merges the slots in rank order, so the cluster agrees on the row's (max, sum).
        """
        row_max, row_sum = cta_max, cta_sum
        if const_expr(self.cluster_size > 1):
            stats_barrier = s_barriers + self.chunks * 8
            cute.arch.cluster_wait()
            if tidx == 0:
                barrier_expect_tx(stats_barrier, Int32(self.cluster_size * STATS_BYTES))
            if (warp == 0) & (lane < self.cluster_size):
                asm(None, [to_ir(s_stats + cta_rank * STATS_BYTES, Int32), to_ir(stats_barrier, Int32), to_ir(lane, Int32), to_ir(cta_max, Float32), to_ir(cta_sum, Float32)], "{\n.reg .b32 remote_slot, remote_barrier;\nmapa.shared::cluster.u32 remote_slot, $0, $2;\nmapa.shared::cluster.u32 remote_barrier, $1, $2;\nst.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [remote_slot], {$3, $4}, [remote_barrier];\n}", "r,r,r,f,f", side_effects=True)
            barrier_wait(stats_barrier, Int32(0))
            row_max, row_sum = -Float32.inf, Float32(0.0)
            for i in cutlass.range_constexpr(self.cluster_size):
                row_max, row_sum = combine_stats(row_max, row_sum, lds_f32(s_stats + i * STATS_BYTES), lds_f32(s_stats + i * STATS_BYTES + 4))
            cute.arch.cluster_arrive_relaxed()
        return row_max, row_sum

    @cute.jit
    def grad_of(self, value: Float32, row_max: Float32, rcp_sum: Float32, rcp_counted: Float32, use_ftz: bool) -> Float32:
        """
        (e / sum) / n in that order, each division a multiply by the correctly rounded reciprocal of
        the row's divisor, taken once per row.
        """
        return Float32(asm(T.f32(), [to_ir(exp_shifted(value, row_max, use_ftz), Float32), to_ir(rcp_sum, Float32), to_ir(rcp_counted, Float32)], "{\n.reg .f32 t;\nmul.rn.f32 t, $1, $2;\nmul.rn.f32 $0, t, $3;\n}", "=f,f,f,f"))

    @cute.jit
    def store_grad(self, row: RowSlice, s_slice: Int32, s_target: Int32, edge_value: Float32, stats: RowStats, num_counted: Float32, tidx: Int32, cta_rank: Int32) -> None:
        """
        (softmax(x) - onehot) / n. The ftz ex2 is used only when no exponent of the slice is below
        -126, where it equals the full-range ex2.
        """
        rcp_sum, rcp_counted = (Float32(asm(T.f32(), [to_ir(divisor, Float32)], "rcp.rn.f32 $0, $1;", "=f,f")) for divisor in (stats.row_sum, num_counted))
        min_exponent = Float32(asm(T.f32(), [to_ir(stats.cta_min, Float32), to_ir(stats.row_max, Float32)], "{\n.reg .f32 t;\nsub.rn.f32 t, $1, $2;\nmul.rn.f32 $0, t, 0f3FB8AA3B;\n}", "=f,f,f"))
        if min_exponent >= Float32(-126.0):
            self.store_slice_grad(row, s_slice, s_target, stats.row_max, rcp_sum, rcp_counted, tidx, True)
        else:
            self.store_slice_grad(row, s_slice, s_target, stats.row_max, rcp_sum, rcp_counted, tidx, False)
        if cta_rank == 0:
            edge_addr, edge_ok = self.access.edge_element(row, tidx)
            edge_grad = self.grad_of(edge_value, stats.row_max, rcp_sum, rcp_counted, False)
            self.access.store_element(edge_addr, cutlass.select_(edge_addr == row.target_addr, edge_grad - rcp_counted, edge_grad), edge_ok)

    @cute.jit
    def store_slice_grad(self, row: RowSlice, s_slice: Int32, s_target: Int32, row_max: Float32, rcp_sum: Float32, rcp_counted: Float32, tidx: Int32, use_ftz: cutlass.Constexpr[bool]) -> None:
        """
        The thread that stores the target's unit then rewrites the target element with the -1 / n
        term, from the copy of x[target] taken before any store.
        """
        for unit_idx in cutlass.range(tidx, row.cta_units, self.threads, unroll=1):
            grads = [self.grad_of(value, row_max, rcp_sum, rcp_counted, use_ftz) for value in unpack_unit(self.load_unit(row, s_slice, unit_idx), self.dtype)]
            st_v4(row.slice_start + Int64(unit_idx) * Int64(16), pack_unit(grads, self.dtype), Int32(1))
            if unit_idx == row.target_unit:
                self.access.store_element(row.target_addr, self.grad_of(lds_f32(s_target), row_max, rcp_sum, rcp_counted, False) - rcp_counted, True)

    @cute.jit
    def store_row_loss(self, row: RowSlice, s_target: Int32, stats: RowStats, mLoss: cute.Tensor, row_idx: Int32, tidx: Int32) -> None:
        """
        logsumexp - x[target] as (max - x[target]) + log(sum), so a dominant target gives 0.
        """
        if row.owns_target & (tidx == 0):
            log2_sum = Float32(asm(T.f32(), [to_ir(stats.row_sum, Float32)], "lg2.approx.f32 $0, $1;", "=f,f"))
            mLoss[row_idx] = (stats.row_max - lds_f32(s_target)) + log2_sum * LN2

    @cute.jit
    def store_zero_row(self, row: RowSlice, mLoss: cute.Tensor, row_idx: Int32, tidx: Int32, cta_rank: Int32) -> None:
        zeros = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        for unit_idx in cutlass.range(tidx, row.cta_units, self.threads, unroll=1):
            st_v4(row.slice_start + Int64(unit_idx) * Int64(16), zeros, Int32(1))
        if cta_rank == 0:
            edge_addr, edge_ok = self.access.edge_element(row, tidx)
            self.access.store_element(edge_addr, Float32(0.0), edge_ok)
            if tidx == 0:
                mLoss[row_idx] = Float32(0.0)

    @cute.jit
    def sync_cluster_exit(self) -> None:
        """
        No CTA exits while a peer may still write to its shared memory.
        """
        if const_expr(self.cluster_size > 1):
            cute.arch.cluster_wait()


class CountNonIgnored:
    """
    n = max(#(target != ignore_index), 1) for mean, or 1 for sum, into count[0]. The main kernel,
    its programmatic dependent, waits for it only before it first reads n.
    """

    def __init__(self, mean: bool) -> None:
        self.mean = mean

    @cute.kernel
    def kernel(self, mTarget: cute.Tensor, mCount: cute.Tensor, ignore_index: Int64) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        lane, warp = tidx % 32, tidx // 32
        cute.arch.griddepcontrol_launch_dependents()
        s_partial = cutlass.memory.SmemAllocator().allocate(COUNT_THREADS // 32 * 4, byte_alignment=16).toint()
        num_counted = Int32(0)
        if const_expr(self.mean):
            num_counted = self.count_rows(mTarget, ignore_index, tidx)
        num_counted = block_sum(num_counted, s_partial, lane, warp, COUNT_THREADS // 32)
        if tidx == 0:
            mCount[0] = Int64(cutlass.max(num_counted, Int32(1)))

    @cute.jit
    def count_rows(self, mTarget: cute.Tensor, ignore_index: Int64, tidx: Int32) -> Int32:
        num_counted = Int32(0)
        for row_idx in cutlass.range(tidx, mTarget.shape[0], COUNT_THREADS, unroll=1):
            num_counted = num_counted + Int32(Int64(mTarget[row_idx]) != ignore_index)
        return num_counted


class MeanLoss:
    """
    The fp32 reduction of the row losses over n (1 for sum), summed in a fixed order, once the
    main kernel is done.
    """

    @cute.kernel
    def kernel(self, mLoss: cute.Tensor, mCount: cute.Tensor, mMeanLoss: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        lane, warp = tidx % 32, tidx // 32
        s_partial = cutlass.memory.SmemAllocator().allocate(COUNT_THREADS // 32 * 4, byte_alignment=16).toint()
        cute.arch.griddepcontrol_wait()
        loss_sum = self.sum_row_losses(mLoss, tidx)
        loss_sum = block_sum(loss_sum, s_partial, lane, warp, COUNT_THREADS // 32)
        if tidx == 0:
            mMeanLoss[0] = Float32(asm(T.f32(), [to_ir(loss_sum, Float32), to_ir(Float32(Int64(mCount[0])), Float32)], "div.rn.f32 $0, $1, $2;", "=f,f,f"))

    @cute.jit
    def sum_row_losses(self, mLoss: cute.Tensor, tidx: Int32) -> Float32:
        loss_sum = Float32(0.0)
        for row_idx in cutlass.range(tidx, mLoss.shape[0], COUNT_THREADS, unroll=1):
            loss_sum = loss_sum + mLoss[row_idx]
        return loss_sum


TARGET_TYPES = {torch.int32: Int32, torch.int64: Int64}
COMPILED: dict = {}


def pick_layout(num_cols: int, element_bytes: int) -> tuple:
    """
    (cluster_size, chunks, is_resident, threads) for a row of num_cols elements: the smallest
    portable cluster whose slices fit PREFERRED_SLICE_BYTES, else 8 CTAs with slices up to
    MAX_SLICE_BYTES, else 8 CTAs reading the row twice from global memory.
    """
    row_bytes = num_cols * element_bytes
    for cluster_size in (1, 2, 4, 8):
        if row_bytes <= cluster_size * PREFERRED_SLICE_BYTES:
            return cluster_size, 2, True, 256
    return 8, 2, row_bytes <= 8 * MAX_SLICE_BYTES, 256


def compile_kernel(dtype: torch.dtype, target_dtype: torch.dtype, num_cols: int, cluster_size: int, chunks: int, is_resident: bool, threads: int, reduction: str) -> Callable:
    x = fake(TORCH2CUTE[dtype], (cute.sym_int(),), (1,), dtype.itemsize)
    target = fake(TARGET_TYPES[target_dtype], (cute.sym_int(),), (1,), target_dtype.itemsize)
    row_loss, count, mean_loss = fake(Float32, (cute.sym_int(),), (1,), 4), fake(Int64, (1,), (1,), 8), fake(Float32, (1,), (1,), 4)
    return cute.compile(CrossEntropy(TORCH2CUTE[dtype], num_cols, cluster_size, chunks, is_resident, threads, reduction == "mean"), x, target, row_loss, count, mean_loss, Int64(0), fake_stream(), options="--enable-tvm-ffi")


def cross_entropy_fwd_bwd_sm90(inp: torch.Tensor, target: torch.Tensor, ignore_index: int, reduction: str = "mean") -> tuple[torch.Tensor, torch.Tensor]:
    """
    The picker's contract on SM90, every dtype, shape and stride through the CuTe kernels.
    """
    num_rows, num_cols = inp.shape
    if num_cols == 0:
        raise ValueError("cross_entropy_fwd_bwd needs at least one class (inp.shape[1] >= 1)")
    if inp.stride(-1) != 1 or inp.stride(-2) != num_cols:
        inp = inp.contiguous()
    if target.dtype not in TARGET_TYPES:
        target = target.to(torch.int64)
    if target.stride(-1) != 1:
        target = target.contiguous()
    if num_rows == 0:
        return torch.zeros((), dtype=torch.float32, device=inp.device), inp
    loss = torch.empty((), dtype=torch.float32, device=inp.device)
    row_loss = torch.empty(num_rows, dtype=torch.float32, device=inp.device)
    count = torch.empty(1, dtype=torch.int64, device=inp.device)
    layout = pick_layout(num_cols, inp.element_size())
    key = ("cross_entropy", inp.dtype, target.dtype, num_cols, *layout, reduction)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_kernel(inp.dtype, target.dtype, num_cols, *layout, reduction)
    fn(inp.view(-1), target, row_loss, count, loss.view(1), ignore_index)
    return loss, inp
