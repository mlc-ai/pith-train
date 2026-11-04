"""
scatter_quant_sm100 in TIRx-lite: one cooperative persistent kernel that builds the global route
map and then streams the quantization.

The grid is every co-resident CTA (one or two per SM by stream shape), launched cooperatively. The
host group counts arrive as group_bound scalar kernel parameters, so the group starts are known at
entry and one grid sync (before the stream) is enough:
1. count: each CTA histograms its slice of the slots (match.any per warp, red.shared into per-warp
   histograms; expert, source row and match mask are cached in the idle stage ring), then claims
   its base inside each group with one atomic add of its count on the group total (CTA order
   inside a group is arbitrary but consistent).
2. rank (no grid sync before it): every CTA scans the padded host group sizes (offs and ks_t from
   CTA 0), ranks its cached slots (row = start(g) + CTA base + earlier warps + earlier steps +
   popc), writes reverse_shuffle_idxs and src_of_row[row]; group owners write -1 into their
   padding rows. After the grid sync CTA 0 re-zeros the totals, so the persistent workspace is zero
   again for the next launch on that stream.
3. stream: each CTA owns an equal contiguous range of 128-row tiles (128 or 256 columns by
   dispatch), loads their src_of_row into shared memory and streams them through a 3-stage
   cp.async ring, takes 1x128 and 128x1 abs-maxima in bf16x2 and quantizes both layouts with exact
   power-of-two bf16x2 products and cvt.rn.satfinite.e4m3x2.bf16x2. For N=2880 the complete
   256-column tiles run wide, then the first CTAs to finish claim 512-row x 64-column tail tasks
   from a second persistent queue; this removes the mostly predicated last wide-tile wave.
Without host counts, a small counting kernel and the one allowed device-to-host copy size the
outputs.
"""

import functools
import inspect

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["scatter_quant_sm100"]


def max_fmaxf(a, b):
    """
    Maximum as CUDA fmaxf for float32 operands, which drops a NaN operand (tvm_tirx emits a
    NaN-propagating select for txl.max; the kernel keeps the fmaxf result); txl.max otherwise.
    """
    return txl.call_pure_extern("fmaxf", a, b, ty=txl.float32) if is_float32(a, b) else txl.max(a, b)


def min_fminf(a, b):
    """
    Minimum as CUDA fminf for float32 operands, txl.min otherwise.
    """
    return txl.call_pure_extern("fminf", a, b, ty=txl.float32) if is_float32(a, b) else txl.min(a, b)


def is_float32(*operands):
    """
    Whether any operand is a float32 expression (the other may be a Python float).
    """
    return any(str(getattr(getattr(operand, "ty", None), "dtype", "")) == "float32" for operand in operands)


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


BLOCK = 128
FULL_MASK = 0xFFFFFFFF
MAX_ABS = "max.xorsign.abs.bf16x2"
WIDE_MIN_ROW_BLOCKS = 250  # wide from 250 row blocks: Qwen EP2 (285+) and GPT-OSS 8k (257-259); DeepSeek EP2 (209) stays narrow
UNROLL = 4  # slot loads in flight per lane in the count pass
STAGES = 3
SM_COUNT = 148
MAX_STATIC_ROW_BLOCKS = 8  # row blocks a CTA's static tile range may touch
CLAIM_AHEAD = 5  # dynamic tiles claimed ahead (claim, store, load: one iteration each)
CLAIM_RING = 8  # claimed dynamic tiles in flight: (row block | column << 20) and their slot lists
MAX_CACHE_STEPS = 16  # count-pass steps cached in the idle stage ring: warps x steps x 32 lanes x 3 words
COUNT_WARPS = 32
PREFETCH_UNITS = 2048  # 16-byte units per L2 prefetch (32 KiB)
PREFETCH_CAP_BYTES = 0 << 20  # bytes of `gathered` warmed into L2 at entry (more only evicts itself)


def pad_to_block(x):
    """
    x rounded up to a multiple of 128, as int32.
    """
    return txl.shift_left(txl.shift_right(txl.Cast("int32", x) + 127, 7), 7)


@functools.cache
def count_kernel(group_bound):
    """
    Single CTA: per-expert counts, grouped_mm_offs and ks_tensor (G > 32 path).
    """
    THREADS = COUNT_WARPS * 32
    groups_per_lane = group_bound // 32

    @txl.kernel()
    def count(expert_idxs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), counts: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), ks_t: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), m: txl.int32, num_groups: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=COUNT_WARPS * 32))
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        smem = txl.smem_pool()
        s_counts = smem.alloc([group_bound], txl.u32)
        for k in range((group_bound + THREADS - 1) // THREADS):
            group_idx = k * THREADS + tidx
            with txl.if_(group_idx < group_bound), txl.then_():
                txl.ptx.st.shared.u32(s_counts.ptr_to([group_idx]), txl.uint32(0))
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))

        expert = txl.local_scalar("uint32")
        match_mask = txl.local_scalar("uint32")
        match_count = txl.local_scalar("uint32")
        leader = txl.local_scalar("uint32")
        prior_count = txl.local_scalar("uint32")
        slot_idx = txl.local_scalar("int32", init=tidx)
        with txl.while_(slot_idx - lane < m):
            is_valid = slot_idx < m
            txl.assign(expert, txl.uint32(FULL_MASK))
            with txl.if_(is_valid), txl.then_():
                txl.ptx.ld.global_.nc.b32(expert, expert_idxs.ptr_to([slot_idx * 2]))
            txl.ptx["match.any.sync.b32"](match_mask, expert, txl.uint32(FULL_MASK))
            txl.ptx.popc.b32(match_count, match_mask)
            txl.ptx.bfind.u32(leader, match_mask)
            with txl.if_(txl.And(is_valid, txl.Cast("uint32", lane) == leader)), txl.then_():
                txl.ptx.atom.shared.add.u32(prior_count, s_counts.ptr_to([expert]), match_count)
            txl.assign(slot_idx, slot_idx + THREADS)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))

        with txl.if_(warp == 0), txl.then_():
            run_sum = txl.local_scalar("int32", init=0)
            group_count = txl.local_scalar("uint32")
            for k in range(groups_per_lane):
                txl.ptx.ld.shared.u32(group_count, s_counts.ptr_to([lane * groups_per_lane + k]))
                txl.assign(run_sum, run_sum + pad_to_block(group_count))
            scan_incl = txl.local_scalar("int32", init=run_sum)
            scan_peer = txl.local_scalar("int32")
            for shift in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(scan_peer, scan_incl, txl.uint32(shift), txl.uint32(0), txl.uint32(FULL_MASK))
                with txl.if_(lane >= shift), txl.then_():
                    txl.assign(scan_incl, scan_incl + scan_peer)
            acc_start = txl.local_scalar("int32", init=scan_incl - run_sum)
            for k in range(groups_per_lane):
                group_idx = lane * groups_per_lane + k
                txl.ptx.ld.shared.u32(group_count, s_counts.ptr_to([group_idx]))
                with txl.if_(group_idx < num_groups), txl.then_():
                    txl.ptx.st.global_.s32(counts.ptr_to([group_idx]), txl.Cast("int32", group_count))
                    txl.ptx.st.global_.s32(offs.ptr_to([group_idx]), acc_start + pad_to_block(group_count))
                    txl.ptx.st.global_.s32(ks_t.ptr_to([group_idx]), pad_to_block(group_count))
                txl.assign(acc_start, acc_start + pad_to_block(group_count))

    return entry_function(count.compile(TARGET))


@functools.cache
def fused_kernel(N, group_bound, wide, is_large=False):
    """
    The compiled cooperative kernel for one compile-time configuration.
    """
    prim_func = fused_traced(N, group_bound, wide, is_large).func  # cooperative launch: LaunchConfig(cooperative=True)
    return entry_function(tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": prim_func}), target=TARGET, pipeline="tirx"))


def stream_config(N, wide):
    """
    (warps, CTAs per SM, tile width): wide = 16 warps, 1 CTA/SM, two 128-column blocks per row
    instruction (256B contiguous stores; needs N % 8 == 0); narrow = 8 warps, 2 CTAs/SM, 128-column
    tiles.
    """
    if wide and N % 8 == 0:
        return 16, 1, 2 * BLOCK
    return 8, 2, BLOCK


def smem_bytes(N, group_bound, wide):
    """
    Shared memory one CTA of the fused kernel uses.
    """
    warps, _, TILE_N = stream_config(N, wide)
    return STAGES * BLOCK * TILE_N * 2 + (MAX_STATIC_ROW_BLOCKS + CLAIM_RING) * BLOCK * 4 + 2 * warps * TILE_N * 2 + TILE_N * 4 + 3 * group_bound * 4 + 512


class TailTask:
    """
    A thread's place in a 512-row x 64-column tail task: four row blocks, four warps per row block,
    eight lanes per row.
    """

    def __init__(self, tail_id, warp, lane, N):
        self.row_block = tail_id * 4
        self.block_offset = warp >> 2
        self.warp_in_block = warp & 3
        self.col_lane = lane & 7
        self.row_sub = lane >> 3
        self.col_start = N - 64


@functools.cache
def fused_traced(N, group_bound, wide, is_large=False):
    """
    The traced cooperative kernel: count, rank, then stream (module docstring).
    """
    WARPS, ctas_per_sm, TILE_N = stream_config(N, wide)
    THREADS = WARPS * 32
    lane_elems = 8 if N % 8 == 0 else 4
    assert N % lane_elems == 0, f"N={N} must be a multiple of 4"
    words_per_lane = lane_elems // 2  # bf16x2 words per lane per row
    lanes_per_row = BLOCK // lane_elems
    blocks_per_insn = TILE_N // BLOCK  # 128-column blocks per warp instruction (wide: one row, adjacent blocks: 256B contiguous stores)
    rows_per_insn = 32 // lanes_per_row // blocks_per_insn  # rows per warp instruction
    rows_per_warp = BLOCK // WARPS
    rows_per_lane = rows_per_warp // rows_per_insn
    num_scale_cols = (N + BLOCK - 1) // BLOCK  # 128-column scale blocks
    has_tail64 = blocks_per_insn > 1 and N > TILE_N and N % TILE_N == 64
    # N=2880: stream the eleven complete 256-column tiles here. The 64-column remainder is a
    # 512-row task after the ordinary loop, so it does not pay a mostly-predicated 128x256 tile for
    # every row block.
    num_tile_cols = N // TILE_N if has_tail64 else (num_scale_cols + blocks_per_insn - 1) // blocks_per_insn
    num_dynamic_cols = max(1, num_tile_cols // 2)  # last tile columns handed out dynamically (tail balancing across SMs)
    num_static_cols = num_tile_cols - num_dynamic_cols  # first tile columns assigned statically
    row_words = N // 2  # words per bf16 row
    groups_per_lane = group_bound // 32
    stage_words = rows_per_lane * THREADS * words_per_lane
    # during the sort the idle stage ring holds the per-warp histograms (at its tail) and the step cache
    hist_offset = STAGES * stage_words - WARPS * group_bound
    CACHE_STEPS = min(MAX_CACHE_STEPS, hist_offset // (WARPS * 32 * 3))
    copy_insn = "cp.async.cg.shared.global" if lane_elems == 8 else "cp.async.ca.shared.global"
    # src_of_row holds each row's word offset into `gathered` (src * N/2; -1 = padding row), so the
    # stream issues each row's copy from one wide multiply-add and an ignore-src predicate;
    # is_large (gathered >= 2^31 words) keeps row ids
    src_stride = 1 if is_large else row_words
    copy_bytes = lane_elems * 2

    def body(gathered, experts, expand, offs, ks_t, reverse, x_fp8, x_scale, x_ch, x_ch_scale, ws, src_of_row, m, num_groups, ntiles, identity, num_ctas, gunits, *host_counts):
        cta_idx = txl.cta_id()
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5

        num_row_blocks = txl.local_scalar("int32", init=ntiles // num_tile_cols)
        num_static_tiles = num_row_blocks * num_static_cols  # static tiles (columns < num_static_cols), split into equal contiguous ranges

        def tile_start_of(cta):
            return txl.Cast("int32", txl.Cast("int64", cta) * num_static_tiles // num_ctas)

        tile_start, tile_end = txl.local_scalar("int32", init=tile_start_of(cta_idx)), txl.local_scalar("int32", init=tile_start_of(cta_idx + 1))
        first_row_block = txl.local_scalar("int32", init=tile_start // max(num_static_cols, 1))
        last_row_block = txl.local_scalar("int32", init=(tile_end - 1) // max(num_static_cols, 1))

        smem = txl.smem_pool()
        s_stage = smem.alloc([STAGES * stage_words], txl.u32)
        s_group_count = smem.alloc([group_bound], txl.i32)
        s_group_start = smem.alloc([group_bound], txl.i32)
        s_cta_base = smem.alloc([group_bound], txl.i32)
        s_static_slots = smem.alloc([MAX_STATIC_ROW_BLOCKS * BLOCK], txl.i32)
        s_dynamic_slots = smem.alloc([CLAIM_RING * BLOCK], txl.i32)
        s_claimed = smem.alloc([CLAIM_RING], txl.i32)
        s_col_partials = smem.alloc([2 * WARPS * TILE_N // 2], txl.u32)
        s_col_rcp = smem.alloc([2 * TILE_N // 2], txl.u32, align=16)

        class Histograms:  # per-warp group histograms, [warp * group_bound + g], at the tail of the idle stage ring
            @staticmethod
            def ptr_to(idx):
                return s_stage.ptr_to([hist_offset + idx[0]])

        hist = Histograms
        cache = s_stage

        def cache_idx(step, field):
            return ((warp * CACHE_STEPS + step) * 32 + lane) * 3 + field

        def cta_sync():
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))

        def start_range(name):
            token = txl.alloc_local([1], "uint32")
            txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
            with txl.if_(tidx == 0), txl.then_():
                txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        def mark(name):
            with txl.if_(tidx == 0), txl.then_():
                txl.cuda.iket.mark(name)

        # workspace (persistent, zero between launches): tot[g] (atomic group totals), queue counters at [group_bound] and [group_bound + 1]
        def group_total_addr(group):
            return ws.ptr_to([group])

        def slot_of_step(step_idx):
            return slot_start + (step_idx * WARPS + warp) * 32 + lane

        def step_in_range(step_idx):
            slot_idx = slot_of_step(step_idx)
            return txl.And(txl.And(step_idx < num_steps, slot_idx < m), slot_idx < slot_start + slots_per_cta)

        def load_step_slots(first_step):
            for unroll_idx in range(UNROLL):
                slot_idx = slot_of_step(first_step + unroll_idx)
                txl.assign(step_experts[unroll_idx], -1)
                txl.assign(step_sources[unroll_idx], slot_idx)
                with txl.if_(step_in_range(first_step + unroll_idx)), txl.then_():
                    txl.ptx.ld.global_.nc.b32(step_experts[unroll_idx], experts.ptr_to([slot_idx * 2]))
                    with txl.if_(identity == 0), txl.then_():
                        txl.ptx.ld.global_.nc.b32(step_sources[unroll_idx], expand.ptr_to([slot_idx * 2]))

        def prefetch_gathered():
            """
            Warm L2 with this CTA's share of gathered while the sort keeps DRAM idle (warp 0).
            """
            with txl.if_(warp == 0), txl.then_():
                prefetch_chunk = txl.local_scalar("int32", init=(gunits + num_ctas - 1) // num_ctas)
                prefetch_end = txl.local_scalar("int32", init=min_fminf(cta_idx * prefetch_chunk + prefetch_chunk, gunits))
                prefetch_unit = txl.local_scalar("int32", init=cta_idx * prefetch_chunk + lane * PREFETCH_UNITS)
                with txl.while_(prefetch_unit < prefetch_end):
                    txl.ptx["cp.async.bulk.prefetch.L2.global"](
                        gathered.ptr_to([txl.Cast("int64", prefetch_unit) * 4]),
                        txl.Cast("uint32", min_fminf(prefetch_end - prefetch_unit, PREFETCH_UNITS) * 16),
                    )
                    txl.assign(prefetch_unit, prefetch_unit + 32 * PREFETCH_UNITS)

        def zero_histograms_and_store_host_counts():
            txl.assign(stride_idx, tidx)
            with txl.while_(stride_idx < WARPS * group_bound):
                txl.ptx.st.shared.s32(hist.ptr_to([stride_idx]), txl.int32(0))
                txl.assign(stride_idx, stride_idx + THREADS)
            # host group counts are kernel parameters, zero past num_groups
            with txl.if_(tidx == 0), txl.then_():
                for host_group in range(group_bound):
                    txl.ptx.st.shared.s32(s_group_count.ptr_to([host_group]), host_counts[host_group])

        def scan_group_sizes():
            """
            Exclusive scan of the padded host sizes into s_group_start; CTA 0 also writes offs and
            ks_t. Warp 0 runs it off the count's critical path; s_group_start is read after the next
            sync.
            """
            with txl.if_(warp == 0), txl.then_():
                run_sum = txl.local_scalar("int32", init=0)
                for k in range(groups_per_lane):
                    txl.ptx.ld.shared.s32(scratch_b, s_group_count.ptr_to([lane * groups_per_lane + k]))
                    txl.assign(run_sum, run_sum + pad_to_block(scratch_b))
                txl.assign(scan_incl, run_sum)
                for shift in (1, 2, 4, 8, 16):
                    txl.ptx.shfl_sync.up.b32(scan_peer, scan_incl, txl.uint32(shift), txl.uint32(0), txl.uint32(FULL_MASK))
                    with txl.if_(lane >= shift), txl.then_():
                        txl.assign(scan_incl, scan_incl + scan_peer)
                acc_start = txl.local_scalar("int32", init=scan_incl - run_sum)
                for k in range(groups_per_lane):
                    group_idx = lane * groups_per_lane + k
                    txl.ptx.st.shared.s32(s_group_start.ptr_to([group_idx]), acc_start)
                    txl.ptx.ld.shared.s32(scratch_b, s_group_count.ptr_to([group_idx]))
                    with txl.if_(txl.And(group_idx < num_groups, cta_idx == 0)), txl.then_():
                        txl.ptx.st.global_.s32(offs.ptr_to([group_idx]), acc_start + pad_to_block(scratch_b))
                        txl.ptx.st.global_.s32(ks_t.ptr_to([group_idx]), pad_to_block(scratch_b))
                    txl.assign(acc_start, acc_start + pad_to_block(scratch_b))

        def count_slots():
            """
            Histogram this CTA's slots per warp and cache the first CACHE_STEPS steps' expert,
            source and match mask.
            """
            with txl.while_(step_idx < num_steps):
                with txl.if_(step_idx > 0), txl.then_():
                    load_step_slots(step_idx)
                for unroll_idx in range(UNROLL):
                    txl.ptx["match.any.sync.b32"](match_mask, step_experts[unroll_idx], txl.uint32(FULL_MASK))
                    txl.assign(step_masks[unroll_idx], match_mask)
                    txl.ptx.popc.b32(match_count, match_mask)
                    txl.ptx.bfind.u32(leader, match_mask)
                    with txl.if_(txl.And(step_experts[unroll_idx] >= 0, txl.Cast("uint32", lane) == leader)), txl.then_():
                        txl.ptx.red.shared.add.u32(hist.ptr_to([warp * group_bound + step_experts[unroll_idx]]), match_count)
                    with txl.if_(step_idx + unroll_idx < CACHE_STEPS), txl.then_():
                        txl.ptx.st.shared.s32(cache.ptr_to([cache_idx(step_idx + unroll_idx, 0)]), step_experts[unroll_idx])
                        txl.ptx.st.shared.s32(cache.ptr_to([cache_idx(step_idx + unroll_idx, 1)]), step_sources[unroll_idx])
                        txl.ptx.st.shared.u32(cache.ptr_to([cache_idx(step_idx + unroll_idx, 2)]), step_masks[unroll_idx])
                txl.assign(step_idx, step_idx + UNROLL)

        def write_padding_rows():
            """
            src_of_row = -1 on the padding rows of the groups this CTA owns.
            """
            txl.assign(owned_group, cta_idx)
            with txl.while_(owned_group < num_groups):
                txl.ptx.ld.shared.s32(scratch_b, s_group_count.ptr_to([owned_group]))
                txl.ptx.ld.shared.s32(scratch_a, s_group_start.ptr_to([owned_group]))
                with txl.if_(tidx < pad_to_block(scratch_b) - scratch_b), txl.then_():
                    txl.ptx.st.global_.s32(src_of_row.ptr_to([scratch_a + scratch_b + tidx]), txl.int32(-1))
                txl.assign(owned_group, owned_group + num_ctas)

        def claim_cta_bases():
            """
            The CTA's base inside each group: one atomic add of its count on the group total.
            """
            txl.assign(stride_idx, tidx)
            with txl.while_(stride_idx < group_bound):
                txl.assign(scratch_a, 0)
                for warp_idx in range(WARPS):
                    txl.ptx.ld.shared.s32(before, hist.ptr_to([warp_idx * group_bound + stride_idx]))
                    txl.assign(scratch_a, scratch_a + txl.Cast("int32", before))
                txl.assign(scratch_b, 0)
                with txl.if_(scratch_a > 0), txl.then_():
                    txl.ptx.atom.global_.add.s32(scratch_b, group_total_addr(stride_idx), scratch_a)
                txl.ptx.st.shared.s32(s_cta_base.ptr_to([stride_idx]), scratch_b)
                txl.assign(stride_idx, stride_idx + THREADS)

        def compute_run_starts():
            """
            Turn the histograms into each warp's first row: run[w][g] = start(g) + earlier CTAs +
            earlier warps.
            """
            txl.assign(stride_idx, tidx)
            with txl.while_(stride_idx < group_bound):
                txl.ptx.ld.shared.s32(scratch_a, s_group_start.ptr_to([stride_idx]))
                with txl.if_(stride_idx < num_groups), txl.then_():
                    txl.ptx.ld.shared.s32(scratch_b, s_cta_base.ptr_to([stride_idx]))
                    txl.assign(scratch_a, scratch_a + scratch_b)
                for warp_idx in range(WARPS):
                    txl.ptx.ld.shared.s32(before, hist.ptr_to([warp_idx * group_bound + stride_idx]))
                    txl.ptx.st.shared.s32(hist.ptr_to([warp_idx * group_bound + stride_idx]), scratch_a)
                    txl.assign(scratch_a, scratch_a + txl.Cast("int32", before))
                txl.assign(stride_idx, stride_idx + THREADS)

        def rank_slots():
            """
            Give every slot its row (from the step cache, or reloaded past CACHE_STEPS) and write
            reverse and src_of_row.
            """
            expert = txl.local_scalar("int32")
            source = txl.local_scalar("int32")
            run_row = txl.local_scalar("int32")
            row = txl.local_scalar("int32")
            row64 = txl.local_scalar("int64")
            txl.assign(step_idx, 0)
            with txl.while_(step_idx < num_steps):
                with txl.if_(step_idx < CACHE_STEPS):
                    with txl.then_():
                        txl.ptx.ld.shared.s32(expert, cache.ptr_to([cache_idx(step_idx, 0)]))
                        txl.ptx.ld.shared.s32(source, cache.ptr_to([cache_idx(step_idx, 1)]))
                        txl.ptx.ld.shared.u32(match_mask, cache.ptr_to([cache_idx(step_idx, 2)]))
                    with txl.else_():
                        slot_idx = slot_of_step(step_idx)
                        txl.assign(expert, -1)
                        txl.assign(source, slot_idx)
                        with txl.if_(step_in_range(step_idx)), txl.then_():
                            txl.ptx.ld.global_.nc.b32(expert, experts.ptr_to([slot_idx * 2]))
                            with txl.if_(identity == 0), txl.then_():
                                txl.ptx.ld.global_.nc.b32(source, expand.ptr_to([slot_idx * 2]))
                        txl.ptx["match.any.sync.b32"](match_mask, expert, txl.uint32(FULL_MASK))
                txl.ptx.popc.b32(match_count, match_mask)
                txl.ptx.bfind.u32(leader, match_mask)
                txl.ptx.popc.b32(before, txl.bitwise_and(match_mask, lanes_below))
                with txl.if_(expert >= 0), txl.then_():
                    txl.ptx.ld.shared.s32(run_row, hist.ptr_to([warp * group_bound + expert]))
                txl.ptx.bar.warp.sync(txl.uint32(FULL_MASK))
                with txl.if_(expert >= 0), txl.then_():
                    txl.assign(row, run_row + txl.Cast("int32", before))
                    with txl.if_(txl.Cast("uint32", lane) == leader), txl.then_():
                        txl.ptx.st.shared.s32(hist.ptr_to([warp * group_bound + expert]), run_row + txl.Cast("int32", match_count))
                    txl.assign(row64, txl.Cast("int64", row))
                    txl.ptx.st.global_.s64(reverse.ptr_to([slot_of_step(step_idx)]), row64)
                    txl.ptx.st.global_.s32(src_of_row.ptr_to([row]), source * src_stride)
                txl.ptx.bar.warp.sync(txl.uint32(FULL_MASK))
                txl.assign(step_idx, step_idx + 1)

        def load_static_slot_lists():
            for piece in range(MAX_STATIC_ROW_BLOCKS * BLOCK // THREADS):
                slot_pos = piece * THREADS + tidx
                with txl.if_(txl.And(tile_end > tile_start, first_row_block + (slot_pos >> 7) <= last_row_block)), txl.then_():
                    txl.ptx.ld.global_.b32(scratch_b, src_of_row.ptr_to([(first_row_block + (slot_pos >> 7)) * BLOCK + (slot_pos & 127)]))
                    txl.ptx.st.shared.s32(s_static_slots.ptr_to([slot_pos]), scratch_b)

        def issue_claim():  # thread THREADS - 1; the atomic's latency is hidden until store_claim
            txl.ptx.atom.global_.add.s32(claimed_ordinal, ws.ptr_to([group_bound]), txl.int32(1))

        def store_claim(ring_slot):  # thread THREADS - 1
            claim_col = txl.local_scalar("int32", init=claimed_ordinal // num_row_blocks)
            txl.ptx.st.shared.s32(
                s_claimed.ptr_to([ring_slot]),
                txl.Select(claimed_ordinal < num_dynamic, (claimed_ordinal - claim_col * num_row_blocks) | txl.shift_left(claim_col + num_static_cols, 20), txl.int32(-1)),
            )

        def load_dynamic_slots(ring_slot):  # 128 threads
            # the last four warps: never fold warps, so the synchronous slot-list load stays off the column-scale chain
            with txl.if_(tidx >= THREADS - BLOCK), txl.then_():
                txl.ptx.ld.shared.s32(scratch_a, s_claimed.ptr_to([ring_slot]))
                with txl.if_(scratch_a >= 0), txl.then_():
                    txl.ptx.ld.global_.b32(scratch_b, src_of_row.ptr_to([(scratch_a & 0xFFFFF) * BLOCK + tidx - (THREADS - BLOCK)]))
                    txl.ptx.st.shared.s32(s_dynamic_slots.ptr_to([ring_slot * BLOCK + tidx - (THREADS - BLOCK)]), scratch_b)

        def preclaim_dynamic_tiles():
            """
            Claim up front the dynamic ordinals the in-loop pipeline cannot reach in time (short
            static ranges); returns their number.
            """
            num_preclaimed = txl.local_scalar("int32", init=max_fmaxf(CLAIM_AHEAD - num_static, 0))
            with txl.if_(tidx == THREADS - 1), txl.then_():
                for k in range(CLAIM_AHEAD):
                    with txl.if_(k < num_preclaimed), txl.then_():
                        issue_claim()
                        store_claim(k)
            cta_sync()
            for k in range(CLAIM_AHEAD):
                with txl.if_(k < num_preclaimed), txl.then_():
                    load_dynamic_slots(k)
            return num_preclaimed

        def local_row(j):
            return warp * rows_per_warp + j * rows_per_insn + row_sub

        def stage_ptr(stage_idx, j):
            return s_stage.ptr_to([((stage_idx * rows_per_lane + j) * THREADS + tidx) * words_per_lane])

        use_col_check = not (N % TILE_N == 0 or has_tail64)  # else every stream tile column lies inside N

        def issue_tile(slot_list, list_start, tile_col, stage_idx):
            col = tile_col * TILE_N + col_sub * BLOCK + col_lane * lane_elems
            for j in range(rows_per_lane):
                txl.ptx.ld.shared.s32(row_sources[j], slot_list.ptr_to([list_start + local_row(j)]))
            if is_large:
                col_word = txl.local_scalar("int64", init=txl.Cast("int64", col // 2))
            else:  # word index fits int32: padding rows (-1) stay negative, as 0 <= col/2 < N/2
                col_word = txl.local_scalar("int32", init=col // 2)
            for j in range(rows_per_lane):
                if is_large:
                    skip = row_sources[j] < 0 if not use_col_check else txl.Or(row_sources[j] < 0, col >= N)
                    word_offset = col_word + txl.Cast("int64", row_sources[j]) * row_words
                else:
                    word_idx = txl.local_scalar("int32", init=row_sources[j] + col_word)
                    skip = row_sources[j] < 0 if not use_col_check else txl.Or(row_sources[j] < 0, col >= N)
                    word_offset = txl.Cast("int64", word_idx)
                txl.ptx[copy_insn](stage_ptr(stage_idx, j), gathered.ptr_to([word_offset]), copy_bytes, txl.ptx.pred(skip))

        # column-outer order over the static tiles: every CTA sweeps the column tiles at about the
        # same pace, so the slots of a token (same gathered row, different row blocks) read each
        # 128-column piece of it within a short window and the repeat reads hit L2; then the
        # dynamic tiles
        def first_block(col_):
            return (tile_start - col_ + num_static_cols - 1) // num_static_cols

        def last_block(col_):
            return (tile_end - 1 - col_ + num_static_cols) // num_static_cols - 1

        def iter_start(block_, col_):
            txl.assign(col_, 0)
            txl.assign(block_, first_block(0))
            with txl.while_(txl.And(block_ > last_block(col_), col_ < num_static_cols - 1)):
                txl.assign(col_, col_ + 1)
                txl.assign(block_, first_block(col_))

        def iter_next(block_, col_):
            txl.assign(block_, block_ + 1)
            with txl.while_(txl.And(block_ > last_block(col_), col_ < num_static_cols - 1)):
                txl.assign(col_, col_ + 1)
                txl.assign(block_, first_block(col_))

        def issue_nth(ordinal, stage_idx):
            """
            Issue the CTA's ordinal-th tile (static first, then dynamic) into stage stage_idx.
            """
            def issue_dynamic():
                txl.ptx.ld.shared.s32(issue_id, s_claimed.ptr_to([(ordinal - num_static) & (CLAIM_RING - 1)]))
                with txl.if_(issue_id >= 0), txl.then_():
                    issue_tile(s_dynamic_slots, ((ordinal - num_static) & (CLAIM_RING - 1)) * BLOCK, issue_id >> 20, stage_idx)

            if num_static_cols == 0:
                issue_dynamic()
                return
            with txl.if_(ordinal < num_static):
                with txl.then_():
                    issue_tile(s_static_slots, (issue_block - first_row_block) * BLOCK, issue_col, stage_idx)
                    iter_next(issue_block, issue_col)
                with txl.else_():
                    issue_dynamic()

        def load_proc_id(ordinal):  # 1 if the CTA's ordinal-th tile exists (dynamic ones: claimed id >= 0)
            txl.assign(proc_id, 1)
            with txl.if_(ordinal >= num_static), txl.then_():
                txl.ptx.ld.shared.s32(proc_id, s_claimed.ptr_to([(ordinal - num_static) & (CLAIM_RING - 1)]))

        def cvt_e8m0_pair(dst, hi_amax, lo_amax):
            hi_scale = max_fmaxf(hi_amax, txl.float32(1e-4)) * txl.float32(1.0 / 448.0)
            lo_scale = max_fmaxf(lo_amax, txl.float32(1e-4)) * txl.float32(1.0 / 448.0)
            txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(dst, hi_scale, lo_scale)

        def bf16_lo(word):
            return txl.reinterpret(txl.shift_left(word, txl.uint32(16)), ty="float32")

        def bf16_hi(word):
            return txl.reinterpret(txl.bitwise_and(word, txl.uint32(0xFFFF0000)), ty="float32")

        def next_tile_coords():
            """
            The row block and tile column of the tile this iteration processes.
            """
            row_block = txl.local_scalar("int32")
            tile_col = txl.local_scalar("int32")
            if num_static_cols > 0:
                with txl.if_(tile_ordinal < num_static):
                    with txl.then_():
                        txl.assign(row_block, proc_block)
                        txl.assign(tile_col, proc_col)
                        iter_next(proc_block, proc_col)
                    with txl.else_():
                        txl.assign(row_block, proc_id & 0xFFFFF)
                        txl.assign(tile_col, proc_id >> 20)
            else:
                txl.assign(row_block, proc_id & 0xFFFFF)
                txl.assign(tile_col, proc_id >> 20)
            return row_block, tile_col

        def advance_claims():
            """
            One step of the dynamic claim pipeline: atomic -> id into s_claimed -> slot list into
            s_dynamic_slots. The slot-list load is split: the global load issues here, its shared
            store waits until just before this warp's bar 1 arrival (after the maxima), so the L2
            round trip hides behind the tile's data wait.
            """
            txl.assign(store_slot, -1)
            with txl.if_(load_slot >= 0), txl.then_():
                with txl.if_(tidx >= THREADS - BLOCK), txl.then_():
                    txl.ptx.ld.shared.s32(scratch_a, s_claimed.ptr_to([load_slot]))
                    with txl.if_(scratch_a >= 0), txl.then_():
                        txl.ptx.ld.global_.b32(slot_value, src_of_row.ptr_to([(scratch_a & 0xFFFFF) * BLOCK + tidx - (THREADS - BLOCK)]))
                        txl.assign(store_slot, load_slot)
                txl.assign(load_slot, -1)
            with txl.if_(claim_slot >= 0), txl.then_():
                with txl.if_(tidx == THREADS - 1), txl.then_():
                    store_claim(claim_slot)
                txl.assign(load_slot, claim_slot)
                txl.assign(claim_slot, -1)
            with txl.if_(tile_ordinal + CLAIM_AHEAD - num_static >= num_preclaimed), txl.then_():
                txl.assign(claim_slot, (tile_ordinal + CLAIM_AHEAD - num_static) & (CLAIM_RING - 1))
                with txl.if_(tidx == THREADS - 1), txl.then_():
                    issue_claim()

        def load_tile_words(stage_idx):
            for j in range(rows_per_lane):
                if words_per_lane == 4:
                    txl.ptx.ld.shared.v4.b32(words[j * 4], words[j * 4 + 1], words[j * 4 + 2], words[j * 4 + 3], stage_ptr(stage_idx, j))
                else:
                    txl.ptx.ld.shared.v2.b32(words[j * 2], words[j * 2 + 1], stage_ptr(stage_idx, j))

        def reduce_lane_maxima():
            """
            |max| over this thread's rows (columns) and over each row's 8/4 elements; the
            xorsign.abs max keeps the magnitude max, its sign bit is dropped at the end.
            """
            for k in range(words_per_lane):
                txl.assign(col_max[k], words[k])
            for j in range(1, rows_per_lane):
                for k in range(words_per_lane):
                    txl.ptx[MAX_ABS](col_max[k], col_max[k], words[j * words_per_lane + k])
            for j in range(rows_per_lane):
                txl.assign(row_max[j], words[j * words_per_lane])
                for k in range(1, words_per_lane):
                    txl.ptx[MAX_ABS](row_max[j], row_max[j], words[j * words_per_lane + k])

        def reduce_row_amax():
            """
            Transposed butterfly over the lanes_per_row lanes of a row: the lane ends with the max
            of row row_sel, which the bits of the lane select; returns row_sel.
            """
            pending = [row_max[j] for j in range(rows_per_lane)]
            xor_mask = lanes_per_row // 2
            while len(pending) > 1:
                half = len(pending) // 2
                is_upper = (lane & xor_mask) != 0
                next_pending = []
                for i in range(half):
                    txl.assign(shfl_send, txl.Select(is_upper, pending[i], pending[i + half]))
                    txl.ptx.shfl_sync.bfly.b32(shfl_peer, shfl_send, txl.uint32(xor_mask), txl.uint32(31), txl.uint32(FULL_MASK))
                    txl.assign(row_max[i], txl.Select(is_upper, pending[i + half], pending[i]))
                    txl.ptx[MAX_ABS](row_max[i], row_max[i], shfl_peer)
                    next_pending.append(row_max[i])
                pending = next_pending
                xor_mask //= 2
            while xor_mask >= 1:
                txl.ptx.shfl_sync.bfly.b32(shfl_peer, row_max[0], txl.uint32(xor_mask), txl.uint32(31), txl.uint32(FULL_MASK))
                txl.ptx[MAX_ABS](row_max[0], row_max[0], shfl_peer)
                xor_mask //= 2
            row_sel = 0
            span = rows_per_lane
            sel_mask = lanes_per_row // 2
            while span > 1:
                span //= 2
                row_sel = row_sel + txl.Select((lane & sel_mask) != 0, span, 0)
                sel_mask //= 2
            return row_sel

        def store_col_partials():
            """
            Join the half-warps' column maxima (narrow) and store this warp's column partials into
            this tile's half of s_col_partials; returns that half's start.
            """
            if rows_per_insn == 2:
                for k in range(words_per_lane):
                    txl.ptx.shfl_sync.bfly.b32(shfl_peer, col_max[k], txl.uint32(16), txl.uint32(31), txl.uint32(FULL_MASK))
                    txl.ptx[MAX_ABS](col_max[k], col_max[k], shfl_peer)
            partial_base = (tile_ordinal & 1) * (WARPS * TILE_N // 2)
            with txl.if_(row_sub == 0), txl.then_():
                for k in range(words_per_lane):
                    txl.ptx.st.shared.u32(s_col_partials.ptr_to([partial_base + warp * (TILE_N // 2) + col_sub * (BLOCK // 2) + col_lane * words_per_lane + k]), col_max[k])
            return partial_base

        def fold_column_scales(partial_base, row_block, tile_col):
            """
            One thread per column word folds the warps' partial maxima (balanced tree over the
            leaves), then writes the column pair's E8M0 scales and bf16x2 power-of-two reciprocal,
            once per CTA.
            """
            for warp_idx in range(WARPS):
                txl.ptx.ld.shared.u32(leaves[warp_idx], s_col_partials.ptr_to([partial_base + warp_idx * (TILE_N // 2) + tidx]))
            width = WARPS
            while width > 1:
                width //= 2
                for warp_idx in range(width):
                    txl.ptx[MAX_ABS](leaves[warp_idx], leaves[warp_idx], leaves[warp_idx + width])
            txl.assign(shfl_send, leaves[0])
            col_amax_bits = txl.bitwise_and(shfl_send, txl.uint32(0x7FFF7FFF))
            cvt_e8m0_pair(scale_pair, bf16_hi(col_amax_bits), bf16_lo(col_amax_bits))
            scale_pair_u32 = txl.Cast("uint32", scale_pair)
            txl.assign(col_exp[0], txl.bitwise_and(scale_pair_u32, txl.uint32(0xFF)))
            txl.assign(col_exp[1], txl.shift_right(scale_pair_u32, txl.uint32(8)))
            txl.ptx.st.shared.u32(
                s_col_rcp.ptr_to([(tile_ordinal & 1) * (TILE_N // 2) + tidx]),
                txl.shift_left(txl.uint32(254) - col_exp[0], txl.uint32(7))
                | txl.shift_left(txl.uint32(254) - col_exp[1], txl.uint32(23)),
            )
            with txl.if_(tile_col * TILE_N + tidx * 2 < N), txl.then_():
                txl.ptx.st.global_.v2.f32(
                    x_ch_scale.ptr_to([row_block * N + tile_col * TILE_N + tidx * 2]),
                    txl.reinterpret(txl.shift_left(col_exp[0], txl.uint32(23)), ty="float32"),
                    txl.reinterpret(txl.shift_left(col_exp[1], txl.uint32(23)), ty="float32"),
                )

        def quantize(out, rcp_of, tile_col, col_ok):
            # N not a multiple of the tile width: only the last tile column needs per-row column
            # checks; full tiles take the unchecked copy of the pass (tile-uniform branch)
            if N % TILE_N == 0:
                quantize_pass(out, rcp_of, False, col_ok)
            else:
                with txl.if_(tile_col < num_tile_cols - 1):
                    with txl.then_():
                        quantize_pass(out, rcp_of, False, col_ok)
                    with txl.else_():
                        quantize_pass(out, rcp_of, True, col_ok)

        def quantize_pass(out, rcp_of, is_checked, col_ok):
            for j in range(rows_per_lane):
                if out is x_fp8:
                    # row j's exponent from the lane that reduced it
                    num_sel_bits = rows_per_lane.bit_length() - 1
                    src_lane = (lane & ~(lanes_per_row - 1)) + sum(((j >> (num_sel_bits - 1 - bit_idx)) & 1) * (lanes_per_row >> (bit_idx + 1)) for bit_idx in range(num_sel_bits))
                    txl.ptx.shfl_sync.idx.b32(shfl_peer, row_exp, txl.uint32(src_lane) if isinstance(src_lane, int) else txl.Cast("uint32", src_lane), txl.uint32(31), txl.uint32(FULL_MASK))
                    rcp_half = txl.shift_left(txl.uint32(254) - shfl_peer, txl.uint32(7))
                    txl.assign(row_rcp, rcp_half | txl.shift_left(rcp_half, txl.uint32(16)))

                def store_row():
                    for k in range(words_per_lane):
                        txl.ptx.mul.rn.bf16x2(scaled, words[j * words_per_lane + k], rcp_of(k))
                        txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(fp8_pairs[k], scaled)
                    for pair_idx in range(words_per_lane // 2):
                        txl.ptx.mov.b32(packed[pair_idx], fp8_pairs[2 * pair_idx], fp8_pairs[2 * pair_idx + 1])
                    word_offset = out_word_base + j * rows_per_insn * (N // 4)
                    if words_per_lane == 4:
                        txl.ptx.st.global_.v2.b32(out.ptr_to([word_offset]), packed[0], packed[1])
                    else:
                        txl.ptx.st.global_.b32(out.ptr_to([word_offset]), packed[0])

                if not is_checked:
                    store_row()
                else:
                    with txl.if_(col_ok), txl.then_():
                        store_row()

        def stream_tail64():
            """
            The 64-column remainder of N=2880 in 512-row tasks from the second queue. One task is
            four row blocks x 64 columns, the same 64 KiB input footprint as a full 128x256 tile;
            four warps own each row block and a warp instruction covers four rows (eight lanes per
            row), so every lane does useful work instead of issuing a 75%-predicated wide tile.
            """
            mark("main-end")
            # drain the ordinary stream before reusing stage 0
            txl.ptx.cp.async_.wait_group(0)
            cta_sync()  # every warp's last s_claimed read (load_proc_id) precedes the tail id's store into s_claimed[0]
            num_tail_tasks = (num_row_blocks + 3) // 4
            tail_id = txl.local_scalar("int32")
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx.atom.global_.add.s32(tail_id, ws.ptr_to([group_bound + 1]), txl.int32(1))
                txl.ptx.st.shared.s32(s_claimed.ptr_to([0]), tail_id)
            cta_sync()
            txl.ptx.ld.shared.s32(tail_id, s_claimed.ptr_to([0]))
            tail_range = txl.alloc_local([1], "uint32")
            txl.assign(tail_range[0], txl.cuda.iket.sentinel_token("tail64"))
            with txl.if_(tail_id < num_tail_tasks), txl.then_():
                with txl.if_(tidx == 0), txl.then_():
                    txl.assign(tail_range[0], txl.cuda.iket.range_start("tail64"))
                tail = TailTask(tail_id, warp, lane, N)
                load_tail_tile(tail)
                quantize_tail_rows(tail)
                fold_tail_col_scales(tail)
                quantize_tail_cols(tail)
            txl.cuda.iket.range_end(tail_range[0])

        def load_tail_tile(tail):
            """
            Each thread owns eight columns of eight rows; its 32 packed bf16 words retain the tile
            for both output layouts.
            """
            row_block, block_offset, warp_in_block, col_lane, row_sub, col_start = tail.row_block, tail.block_offset, tail.warp_in_block, tail.col_lane, tail.row_sub, tail.col_start
            tail_source = txl.local_scalar("int32")
            for j in range(8):
                tail_row = (row_block + block_offset) * BLOCK + warp_in_block * 32 + j * 4 + row_sub
                txl.assign(tail_source, -1)
                with txl.if_(row_block + block_offset < num_row_blocks), txl.then_():
                    txl.ptx.ld.global_.s32(tail_source, src_of_row.ptr_to([tail_row]))
                tail_word_offset = txl.Cast("int64", tail_source) * (row_words // src_stride) + (col_start // 2) + col_lane * 4
                txl.ptx[copy_insn](s_stage.ptr_to([(j * THREADS + tidx) * 4]), gathered.ptr_to([tail_word_offset]), copy_bytes, txl.ptx.pred(tail_source < 0))
            txl.ptx.cp.async_.commit_group()
            txl.ptx.cp.async_.wait_group(0)
            for j in range(8):
                txl.ptx.ld.shared.v4.b32(words[j * 4], words[j * 4 + 1], words[j * 4 + 2], words[j * 4 + 3], s_stage.ptr_to([(j * THREADS + tidx) * 4]))

        def quantize_tail_rows(tail):
            """
            Rowwise block scale and FP8 output; XOR 4/2/1 reduces each independent eight-lane row
            group.
            """
            row_block, block_offset, warp_in_block, col_lane, row_sub, col_start = tail.row_block, tail.block_offset, tail.warp_in_block, tail.col_lane, tail.row_sub, tail.col_start
            for j in range(8):
                txl.assign(row_max[0], words[j * 4])
                for k in range(1, 4):
                    txl.ptx[MAX_ABS](row_max[0], row_max[0], words[j * 4 + k])
                for shift in (4, 2, 1):
                    txl.ptx.shfl_sync.bfly.b32(shfl_peer, row_max[0], txl.uint32(shift), txl.uint32(31), txl.uint32(FULL_MASK))
                    txl.ptx[MAX_ABS](row_max[0], row_max[0], shfl_peer)
                amax_bits = txl.bitwise_and(row_max[0], txl.uint32(0x7FFF7FFF))
                amax = max_fmaxf(bf16_lo(amax_bits), bf16_hi(amax_bits))
                cvt_e8m0_pair(scale_pair, txl.float32(0.0), amax)
                txl.assign(row_exp, txl.bitwise_and(txl.Cast("uint32", scale_pair), txl.uint32(0xFF)))
                rcp_half = txl.shift_left(txl.uint32(254) - row_exp, txl.uint32(7))
                txl.assign(row_rcp, rcp_half | txl.shift_left(rcp_half, txl.uint32(16)))
                tail_row = (row_block + block_offset) * BLOCK + warp_in_block * 32 + j * 4 + row_sub
                with txl.if_(row_block + block_offset < num_row_blocks), txl.then_():
                    with txl.if_(col_lane == 0), txl.then_():
                        txl.ptx.st.global_.f32(
                            x_scale.ptr_to([(num_scale_cols - 1) * (num_row_blocks * BLOCK) + tail_row]),
                            txl.reinterpret(txl.shift_left(row_exp, txl.uint32(23)), ty="float32"),
                        )
                    for k in range(4):
                        txl.ptx.mul.rn.bf16x2(scaled, words[j * 4 + k], row_rcp)
                        txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(fp8_pairs[k], scaled)
                    txl.ptx.mov.b32(packed[0], fp8_pairs[0], fp8_pairs[1])
                    txl.ptx.mov.b32(packed[1], fp8_pairs[2], fp8_pairs[3])
                    txl.ptx.st.global_.v2.b32(x_fp8.ptr_to([txl.Cast("int64", tail_row) * (N // 4) + col_start // 4 + col_lane * 2]), packed[0], packed[1])

        def fold_tail_col_scales(tail):
            """
            Per-column maxima: each lane folds its eight rows, XOR 8/16 joins the four row quarters
            of the warp, then one warp per row block folds its four warp partials and publishes 32
            packed column reciprocals and scales.
            """
            row_block, block_offset, warp_in_block, col_lane, row_sub, col_start = tail.row_block, tail.block_offset, tail.warp_in_block, tail.col_lane, tail.row_sub, tail.col_start
            for k in range(4):
                txl.assign(col_max[k], words[k])
                for j in range(1, 8):
                    txl.ptx[MAX_ABS](col_max[k], col_max[k], words[j * 4 + k])
                for shift in (8, 16):
                    txl.ptx.shfl_sync.bfly.b32(shfl_peer, col_max[k], txl.uint32(shift), txl.uint32(31), txl.uint32(FULL_MASK))
                    txl.ptx[MAX_ABS](col_max[k], col_max[k], shfl_peer)
            with txl.if_(row_sub == 0), txl.then_():
                for k in range(4):
                    txl.ptx.st.shared.u32(s_col_partials.ptr_to([warp * 32 + col_lane * 4 + k]), col_max[k])
            cta_sync()

            with txl.if_(warp_in_block == 0), txl.then_():
                for warp_idx in range(4):
                    txl.ptx.ld.shared.u32(leaves[warp_idx], s_col_partials.ptr_to([(block_offset * 4 + warp_idx) * 32 + lane]))
                txl.ptx[MAX_ABS](leaves[0], leaves[0], leaves[2])
                txl.ptx[MAX_ABS](leaves[1], leaves[1], leaves[3])
                txl.ptx[MAX_ABS](leaves[0], leaves[0], leaves[1])
                col_amax_bits = txl.bitwise_and(leaves[0], txl.uint32(0x7FFF7FFF))
                cvt_e8m0_pair(scale_pair, bf16_hi(col_amax_bits), bf16_lo(col_amax_bits))
                scale_pair_u32 = txl.Cast("uint32", scale_pair)
                txl.assign(col_exp[0], txl.bitwise_and(scale_pair_u32, txl.uint32(0xFF)))
                txl.assign(col_exp[1], txl.shift_right(scale_pair_u32, txl.uint32(8)))
                txl.ptx.st.shared.u32(
                    s_col_rcp.ptr_to([block_offset * 32 + lane]),
                    txl.shift_left(txl.uint32(254) - col_exp[0], txl.uint32(7))
                    | txl.shift_left(txl.uint32(254) - col_exp[1], txl.uint32(23)),
                )
                with txl.if_(row_block + block_offset < num_row_blocks), txl.then_():
                    txl.ptx.st.global_.v2.f32(
                        x_ch_scale.ptr_to([(row_block + block_offset) * N + col_start + lane * 2]),
                        txl.reinterpret(txl.shift_left(col_exp[0], txl.uint32(23)), ty="float32"),
                        txl.reinterpret(txl.shift_left(col_exp[1], txl.uint32(23)), ty="float32"),
                    )
            cta_sync()

        def quantize_tail_cols(tail):
            row_block, block_offset, warp_in_block, col_lane, row_sub, col_start = tail.row_block, tail.block_offset, tail.warp_in_block, tail.col_lane, tail.row_sub, tail.col_start
            txl.ptx.ld.shared.v4.b32(col_rcp[0], col_rcp[1], col_rcp[2], col_rcp[3], s_col_rcp.ptr_to([block_offset * 32 + col_lane * 4]))

            for j in range(8):
                tail_row = (row_block + block_offset) * BLOCK + warp_in_block * 32 + j * 4 + row_sub
                with txl.if_(row_block + block_offset < num_row_blocks), txl.then_():
                    for k in range(4):
                        txl.ptx.mul.rn.bf16x2(scaled, words[j * 4 + k], col_rcp[k])
                        txl.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(fp8_pairs[k], scaled)
                    txl.ptx.mov.b32(packed[0], fp8_pairs[0], fp8_pairs[1])
                    txl.ptx.mov.b32(packed[1], fp8_pairs[2], fp8_pairs[3])
                    txl.ptx.st.global_.v2.b32(x_ch.ptr_to([txl.Cast("int64", tail_row) * (N // 4) + col_start // 4 + col_lane * 2]), packed[0], packed[1])

        sort_range = start_range("sort")
        mark("entry")
        slots_per_cta = txl.local_scalar("int32", init=(m + num_ctas * 32 - 1) // (num_ctas * 32) * 32)
        slot_start = txl.local_scalar("int32", init=cta_idx * slots_per_cta)
        num_steps = txl.local_scalar("int32", init=(slots_per_cta + THREADS - 1) // THREADS)
        step_experts = txl.alloc_local([UNROLL], "int32")
        step_sources = txl.alloc_local([UNROLL], "int32")
        step_masks = txl.alloc_local([UNROLL], "uint32")
        match_mask = txl.local_scalar("uint32")
        match_count = txl.local_scalar("uint32")
        leader = txl.local_scalar("uint32")
        before = txl.local_scalar("uint32")  # lanes of the match before this one; also a histogram load
        lanes_below = txl.shift_left(txl.uint32(1), txl.Cast("uint32", lane)) - txl.uint32(1)
        # loop counter and two scratch registers, reused by the sort phases
        stride_idx = txl.local_scalar("int32")
        scratch_a = txl.local_scalar("int32")
        scratch_b = txl.local_scalar("int32")

        load_step_slots(0)
        prefetch_gathered()
        zero_histograms_and_store_host_counts()
        cta_sync()

        scan_incl = txl.local_scalar("int32")
        scan_peer = txl.local_scalar("int32")
        owned_group = txl.local_scalar("int32")
        scan_group_sizes()
        step_idx = txl.local_scalar("int32", init=0)
        count_slots()
        cta_sync()

        write_padding_rows()
        claim_cta_bases()
        mark("count-done")
        cta_sync()

        compute_run_starts()
        cta_sync()

        rank_slots()
        # the queue counters, published by the grid sync
        with txl.if_(txl.And(cta_idx == 0, tidx == 0)), txl.then_():
            txl.ptx.st.global_.s32(ws.ptr_to([group_bound]), txl.int32(0))
            txl.ptx.st.global_.s32(ws.ptr_to([group_bound + 1]), txl.int32(0))
        mark("rank-done")
        txl.cuda.grid_sync()
        mark("sync3")

        # every tot[] read precedes sync3: re-zero for the next launch
        with txl.if_(cta_idx == 0), txl.then_():
            txl.assign(stride_idx, tidx)
            with txl.while_(stride_idx < group_bound):
                txl.ptx.st.global_.s32(group_total_addr(stride_idx), txl.int32(0))
                txl.assign(stride_idx, stride_idx + THREADS)
        load_static_slot_lists()

        # dynamic tiles: ordinal q -> column num_static_cols + q // num_row_blocks, row block
        # q % num_row_blocks (column-major: the whole GPU sweeps the last columns in order)
        num_static = txl.local_scalar("int32", init=tile_end - tile_start)
        num_dynamic = num_row_blocks * num_dynamic_cols
        claimed_ordinal = txl.local_scalar("int32")
        num_preclaimed = preclaim_dynamic_tiles()
        cta_sync()
        txl.cuda.iket.range_end(sort_range[0])

        stream_range = start_range("stream")
        col_lane = lane % lanes_per_row
        lane_sub = lane // lanes_per_row
        row_sub = lane_sub if rows_per_insn > 1 else 0  # row offset of this lane's half-warp (narrow)
        col_sub = lane_sub if blocks_per_insn > 1 else 0  # column block of this lane's half-warp (wide)
        row_sources = txl.alloc_local([rows_per_lane], "int32")
        issue_block = txl.local_scalar("int32")
        issue_col = txl.local_scalar("int32")
        proc_block = txl.local_scalar("int32")
        proc_col = txl.local_scalar("int32")
        if num_static_cols > 0:
            iter_start(issue_block, issue_col)
            iter_start(proc_block, proc_col)
        issue_id = txl.local_scalar("int32")
        proc_id = txl.local_scalar("int32")
        claim_slot = txl.local_scalar("int32", init=-1)  # claimed (atomic in flight), id not stored yet
        store_slot = txl.local_scalar("int32", init=-1)  # slot list loaded into slot_value this iteration, stored before bar 1
        slot_value = txl.local_scalar("int32")
        load_slot = txl.local_scalar("int32", init=-1)  # id stored (published by the last bar 2), slot list not loaded
        for stage_idx in range(STAGES - 1):
            issue_nth(txl.int32(stage_idx), stage_idx)
            txl.ptx.cp.async_.commit_group()
        load_proc_id(txl.int32(0))

        words = txl.alloc_local([rows_per_lane * words_per_lane], "uint32")
        col_max = txl.alloc_local([words_per_lane], "uint32")
        row_max = txl.alloc_local([rows_per_lane], "uint32")
        shfl_peer = txl.local_scalar("uint32")
        col_rcp = txl.alloc_local([words_per_lane], "uint32")
        col_exp = txl.alloc_local([lane_elems], "uint32")
        scale_pair = txl.local_scalar("uint16")
        row_rcp = txl.local_scalar("uint32")
        scaled = txl.local_scalar("uint32")
        fp8_pairs = txl.alloc_local([words_per_lane], "uint16")
        packed = txl.alloc_local([max(words_per_lane // 2, 1)], "uint32")
        shfl_send = txl.local_scalar("uint32")
        leaves = txl.alloc_local([WARPS], "uint32")
        num_fold_warps = TILE_N // 64  # fold warps: one thread per column word
        row_exp = txl.local_scalar("uint32")
        out_word_base = txl.local_scalar("int64")
        tile_ordinal = txl.local_scalar("int32", init=0)
        with txl.while_(proc_id >= 0):
            row_block, tile_col = next_tile_coords()
            advance_claims()
            row_start = row_block * BLOCK
            issue_nth(tile_ordinal + STAGES - 1, (tile_ordinal + STAGES - 1) % STAGES)
            txl.ptx.cp.async_.commit_group()
            txl.ptx.cp.async_.wait_group(STAGES - 1)
            load_tile_words(tile_ordinal % STAGES)

            col = txl.local_scalar("int32", init=tile_col * TILE_N + col_sub * BLOCK + col_lane * lane_elems)
            scale_col = tile_col * blocks_per_insn + col_sub  # this lane's 128-column scale block
            col_ok = col < N
            reduce_lane_maxima()
            row_sel = reduce_row_amax()
            amax_bits = txl.bitwise_and(row_max[0], txl.uint32(0x7FFF7FFF))
            amax = max_fmaxf(bf16_lo(amax_bits), bf16_hi(amax_bits))
            cvt_e8m0_pair(scale_pair, txl.float32(0.0), amax)
            txl.assign(row_exp, txl.bitwise_and(txl.Cast("uint32", scale_pair), txl.uint32(0xFF)))
            lane_row_start = row_start + warp * rows_per_warp + row_sub
            with txl.if_(txl.And((lane & (lanes_per_row // rows_per_lane - 1)) == 0, scale_col < num_scale_cols)), txl.then_():
                txl.ptx.st.global_.f32(
                    x_scale.ptr_to([scale_col * (num_row_blocks * BLOCK) + lane_row_start + row_sel * rows_per_insn]),
                    txl.reinterpret(txl.shift_left(row_exp, txl.uint32(23)), ty="float32"),
                )
            partial_base = store_col_partials()

            # named barriers: warps num_fold_warps.. hand their column partials to the fold warps
            # (bar 1) and wait only for their column scales (bar 2); every warp quantizes the
            # row-scaled x_fp8 in between
            with txl.if_(store_slot >= 0), txl.then_():
                txl.ptx.st.shared.s32(s_dynamic_slots.ptr_to([store_slot * BLOCK + tidx - (THREADS - BLOCK)]), slot_value)
            with txl.if_(warp >= num_fold_warps), txl.then_():
                txl.ptx.bar.arrive(txl.uint32(1), txl.uint32(THREADS))
            txl.assign(out_word_base, txl.Cast("int64", lane_row_start) * (N // 4) + txl.Cast("int64", col // 4))
            if blocks_per_insn > 1:
                # wide (one CTA per SM, no sibling CTA to hide the fold): the fold warps fold first
                # and publish the column scales with bar.arrive(2), then quantize their x_fp8 rows;
                # bar 3 orders the scales among them
                with txl.if_(warp >= num_fold_warps), txl.then_():
                    quantize(x_fp8, lambda k: row_rcp, tile_col, col_ok)
                with txl.if_(warp < num_fold_warps):
                    with txl.then_():
                        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS))
                        fold_column_scales(partial_base, row_block, tile_col)
                        txl.ptx.bar.arrive(txl.uint32(2), txl.uint32(THREADS))
                        quantize(x_fp8, lambda k: row_rcp, tile_col, col_ok)
                        txl.ptx.bar.sync(txl.uint32(3), txl.uint32(num_fold_warps * 32))
                    with txl.else_():
                        txl.ptx.bar.sync(txl.uint32(2), txl.uint32(THREADS))
            else:
                quantize(x_fp8, lambda k: row_rcp, tile_col, col_ok)
                with txl.if_(warp < num_fold_warps), txl.then_():
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS))
                    fold_column_scales(partial_base, row_block, tile_col)
                txl.ptx.bar.sync(txl.uint32(2), txl.uint32(THREADS))

            if words_per_lane == 4:
                txl.ptx.ld.shared.v4.b32(col_rcp[0], col_rcp[1], col_rcp[2], col_rcp[3], s_col_rcp.ptr_to([(tile_ordinal & 1) * (TILE_N // 2) + col_sub * (BLOCK // 2) + col_lane * 4]))
            else:
                txl.ptx.ld.shared.v2.b32(col_rcp[0], col_rcp[1], s_col_rcp.ptr_to([(tile_ordinal & 1) * (TILE_N // 2) + col_sub * (BLOCK // 2) + col_lane * 2]))
            quantize(x_ch, lambda k: col_rcp[k], tile_col, col_ok)
            txl.assign(tile_ordinal, tile_ordinal + 1)
            load_proc_id(tile_ordinal)

        if has_tail64:
            stream_tail64()
        txl.cuda.iket.range_end(stream_range[0])

    params = [
        ("gathered", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("experts", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("expand", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        ("offs", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("ks_t", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        ("reverse", txl.Tensor((txl.int64(),), txl.i64, data_alignment=1)), ("x_fp8", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("x_scale", txl.Tensor((txl.int64(),), txl.f32, data_alignment=1)),
        ("x_ch", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("x_ch_scale", txl.Tensor((txl.int64(),), txl.f32, data_alignment=1)), ("ws", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        ("src_of_row", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
        ("m", txl.i32), ("num_groups", txl.i32), ("ntiles", txl.i32), ("identity", txl.i32), ("num_ctas", txl.i32),
        ("gunits", txl.i32),
    ] + [(f"hc{host_group}", txl.i32) for host_group in range(group_bound)]

    names = [name for name, _ in params]

    def fused(*args):
        launch = txl.cuda.LaunchConfig(grid=args[names.index("num_ctas")], block=WARPS * 32, cooperative=True)
        txl.device_entry(launch=launch, kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=ctas_per_sm))
        body(*args)

    fused.__signature__ = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation_type) for name, annotation_type in params])
    fused.__name__ = "fused"
    return txl.kernel()(fused)


WORKSPACES = {}


def workspace(device, group_bound):
    """
    The group totals tot[group_bound] and the stream and tail queue counters, per device and stream;
    the totals are re-zeroed after, and the queues reset before, the cooperative grid sync.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream, group_bound)
    counters = WORKSPACES.get(key)
    if counters is None:
        counters = WORKSPACES[key] = torch.zeros((group_bound + 2,), dtype=torch.int32, device=device)
    return counters


def bound_groups(num_groups):
    """
    num_groups rounded up to a power of two, at least 32.
    """
    group_bound = 32
    while group_bound < num_groups:
        group_bound *= 2
    return group_bound


def empty_outputs(N, device):
    num_scale_cols = (N + BLOCK - 1) // BLOCK
    return (
        (
            torch.empty((0, N), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((0, num_scale_cols), dtype=torch.float32, device=device),
            torch.empty((0, N), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((0, N), dtype=torch.float32, device=device),
        ),
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.int32, device=device),
        [],
        torch.empty((0,), dtype=torch.int32, device=device),
    )


def scatter_quant_sm100(gathered, expert_idxs, num_groups, expand_idx=None, padding_alignment=128, group_counts=None):
    """
    SM100 implementation of scatter_quant: one cooperative launch, plus a counting kernel and a
    device-to-host copy when group_counts is None.
    """
    assert padding_alignment == 128, "the per-channel FP8 layout needs 128-row groups"
    device = gathered.device
    m = expert_idxs.shape[0]
    N = gathered.shape[1]
    if m == 0:
        return empty_outputs(N, device)
    group_bound = bound_groups(num_groups)
    num_scale_cols = (N + BLOCK - 1) // BLOCK
    experts = expert_idxs.view(torch.int32)
    expand = experts if expand_idx is None else expand_idx.view(torch.int32)
    offs = torch.empty((num_groups,), dtype=torch.int32, device=device)
    ks_t = torch.empty((num_groups,), dtype=torch.int32, device=device)
    if group_counts is None:
        # sizes needed on the host to allocate: one counting kernel + the one allowed device-to-host copy
        device_counts = torch.empty((num_groups,), dtype=torch.int32, device=device)
        count_kernel(group_bound)(experts, device_counts, offs, ks_t, m, num_groups)
        group_counts = device_counts.tolist()
    group_counts = [int(count) for count in group_counts]
    ks = [(count + BLOCK - 1) // BLOCK * BLOCK for count in group_counts]
    rows = sum(ks)
    num_row_blocks = rows // BLOCK
    # one cooperative wave of co-resident CTAs; contiguous equal tile ranges (<= MAX_STATIC_ROW_BLOCKS row blocks each)
    wide = num_row_blocks >= WIDE_MIN_ROW_BLOCKS
    _, ctas_per_sm, TILE_N = stream_config(N, wide)
    blocks_per_insn = TILE_N // BLOCK
    num_tile_cols = N // TILE_N if blocks_per_insn > 1 and N > TILE_N and N % TILE_N == 64 else (num_scale_cols + blocks_per_insn - 1) // blocks_per_insn
    ntiles = num_row_blocks * num_tile_cols
    # 228 KiB of shared memory per SM, 1 KiB of it reserved per CTA
    num_resident = SM_COUNT * (ctas_per_sm if smem_bytes(N, group_bound, wide) * ctas_per_sm <= (228 - ctas_per_sm) << 10 else 1)
    num_ctas = min(num_resident, ntiles)
    num_dynamic_cols = max(1, num_tile_cols // 2)
    assert -(-num_row_blocks * (num_tile_cols - num_dynamic_cols) // num_ctas) <= (MAX_STATIC_ROW_BLOCKS - 1) * max(num_tile_cols - num_dynamic_cols, 1), "static ranges would exceed MAX_STATIC_ROW_BLOCKS row blocks"
    reverse = torch.empty((m,), dtype=torch.int64, device=device)
    x_fp8 = torch.empty((rows, N), dtype=torch.float8_e4m3fn, device=device)
    # MN-major scale storage: logical (M, ceil(N/128)), physical strides (1, M).
    x_scale_t = torch.empty((num_scale_cols, rows), dtype=torch.float32, device=device)
    x_scale = x_scale_t.t()
    x_ch = torch.empty((rows, N), dtype=torch.float8_e4m3fn, device=device)
    x_ch_scale = torch.empty((num_row_blocks, N), dtype=torch.float32, device=device)
    ws = workspace(device, group_bound)
    src_of_row = torch.empty((rows,), dtype=torch.int32, device=device)
    args = [
        gathered.view(-1).view(torch.int32), experts, expand, offs, ks_t, reverse,
        x_fp8.view(-1).view(torch.int32), x_scale_t.view(-1), x_ch.view(-1).view(torch.int32), x_ch_scale.view(-1),
        ws, src_of_row, m, num_groups, ntiles, int(expand_idx is None), num_ctas, min(gathered.numel() * 2, PREFETCH_CAP_BYTES) // 16,
    ] + group_counts + [0] * (group_bound - num_groups)
    fused_kernel(N, group_bound, wide, gathered.numel() // 2 >= 2**31 - 1)(*args)
    return (x_fp8, x_scale, x_ch, x_ch_scale), reverse, offs, ks, ks_t
