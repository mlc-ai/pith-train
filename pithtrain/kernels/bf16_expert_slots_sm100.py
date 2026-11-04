"""
SM100 bf16_expert_slots in TIRx-lite: one launch per call (two for a split segment_sum), and no
memsets at all.

group_slots: five clusters of 16 CTAs, each cluster computing the histograms and scan redundantly
and storing a fifth of the slots, each CTA owning a slice of the slots held in registers: per-(tile,
warp) match.any counts and a slice histogram that every CTA pushes into every peer's shared memory
before one cluster barrier; the mask is zeroed by its owners before that barrier and OR-ed with
global reductions after it, then each slice is ranked with no further barrier (a slice too large
for registers takes the rescan kernel, where every CTA rescans all slots). gather_sum is
persistent: in each CTA a producer warp packs the grouped rows of consecutive tokens (looked up in
the per-(expert, token) table) into units, one bulk copy per row into an mbarrier ring, and every
consumer thread adds a unit's rows in ascending expert order.

gather_rows is a 16-byte vector copy. segment_sum without an index gives each warp one 256-row
chunk, streamed through a per-thread cp.async ring, and adds the chunk sums in order through shared
memory; with an index (or a large input without one) it takes two launches: one warp per (chunk,
512-column block) writes the chunk's fp32 sum, then a second kernel, launched with programmatic
dependent launch so its launch and prologue overlap the first one's tail, adds each segment's chunk
sums in order.
"""

import functools
from typing import NamedTuple, Optional

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["bf16_expert_group_slots_sm100", "bf16_expert_gather_sum_sm100", "bf16_expert_gather_rows_sm100", "bf16_expert_segment_sum_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


FULL_MASK = 0xFFFFFFFF


class GroupSlots(NamedTuple):
    cu_seqlens: torch.Tensor
    a_idx: torch.Tensor
    row_of: torch.Tensor
    slot_of: torch.Tensor
    mask: Optional[torch.Tensor]
    table: Optional[torch.Tensor]


def as_u32(value):
    return txl.Cast("uint32", value)


def as_i32(value):
    return txl.Cast("int32", value)


def warp_inclusive_scan(value, peer, lane):
    """
    Turn each lane's value into the inclusive prefix sum over lanes 0 .. lane; peer is scratch.
    """
    for offset in (1, 2, 4, 8, 16):
        txl.ptx.shfl_sync.up.b32(peer, value, txl.uint32(offset), txl.uint32(0), txl.uint32(FULL_MASK))
        txl.assign(value, value + txl.Select(lane >= offset, peer, txl.int32(0)))


def store_expert_offsets(cu, G, lane, pair_excl, total_even, pair_incl):
    """
    Store cu[2 * lane] and cu[2 * lane + 1] from warp 0's scan of expert pairs, and cu[64] when
    G is 64.
    """
    with txl.if_(2 * lane <= G), txl.then_():
        txl.ptx.st.global_.s32(cu.ptr_to([2 * lane]), pair_excl)
    with txl.if_(2 * lane + 1 <= G), txl.then_():
        txl.ptx.st.global_.s32(cu.ptr_to([2 * lane + 1]), pair_excl + total_even)
    if G == 64:
        with txl.if_(lane == 31), txl.then_():
            txl.ptx.st.global_.s32(cu.ptr_to([64]), pair_incl)


def add_bf16x2(acc, words, word_idx, pair):
    """
    Add the low and high bf16 of words[word_idx] to acc[2 * pair] and acc[2 * pair + 1].
    """
    lo = txl.reinterpret(txl.shift_left(words[word_idx], txl.uint32(16)), ty="float32")
    hi = txl.reinterpret(words[word_idx] & txl.uint32(0xFFFF0000), ty="float32")
    txl.assign(acc[2 * pair], acc[2 * pair] + lo)
    txl.assign(acc[2 * pair + 1], acc[2 * pair + 1] + hi)


def pack_bf16x2(words, acc):
    """
    Round the 8 fp32 sums in acc to bf16 pairs in words[0 .. 4).
    """
    for pair in range(4):
        txl.ptx["cvt.rn.bf16x2.f32"](words[pair], acc[2 * pair + 1], acc[2 * pair])


TILE = 1024  # slots ranked per CTA (one per thread)
MASK_WINDOW = 4096  # tokens per mask window in shared memory


@functools.cache
def compile_group_slots(G: int, has_expand: bool):
    """
    The rescan kernel: one CTA per 1024-slot tile, every CTA counting all slots itself.
    """
    NUM_WARPS = TILE // 32

    @txl.kernel()
    def group_slots(e: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), x: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), a_idx: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), row_of: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), slot_of: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), mask: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), table: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), S: txl.int32, T: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=txl.max(1, (S + TILE - 1) // TILE), block=NUM_WARPS * 32))
        cta_idx = txl.cta_id()
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        num_ctas = txl.max(1, (S + TILE - 1) // TILE)
        smem = txl.smem_pool()
        s_expert_total = smem.alloc([64], txl.i32)
        s_expert_before = smem.alloc([64], txl.i32)  # slots of each expert before this CTA's tile
        s_expert_start = smem.alloc([64], txl.i32)
        s_warp_count = smem.alloc([64 * NUM_WARPS], txl.i32)
        s_mask = smem.alloc([MASK_WINDOW], txl.u64)
        lane_mask_lt = txl.shift_left(txl.uint32(1), as_u32(lane)) - txl.uint32(1)

        def zero_counters():
            with txl.if_(tidx < 64), txl.then_():
                txl.ptx.st.shared.s32(s_expert_total.ptr_to([tidx]), txl.int32(0))
                txl.ptx.st.shared.s32(s_expert_before.ptr_to([tidx]), txl.int32(0))
            with txl.serial(0, (64 * NUM_WARPS) // TILE) as i:
                txl.ptx.st.shared.s32(s_warp_count.ptr_to([i * TILE + tidx]), txl.int32(0))

        def count_and_mark_slots(window_idx, window_start, window_end):
            """
            Set the mask bits of the window's tokens; in the first window also count every expert.
            """
            with txl.serial(0, (S + TILE * UNROLL - 1) // (TILE * UNROLL)) as step:
                for i in range(UNROLL):
                    slot_idx = (step * UNROLL + i) * TILE + tidx
                    txl.assign(slot_experts[i], txl.int64(64))
                    with txl.if_(slot_idx < S), txl.then_():
                        txl.ptx.ld.global_.nc.s64(slot_experts[i], e.ptr_to([slot_idx]))
                        if has_expand:
                            txl.ptx.ld.global_.nc.s64(slot_tokens[i], x.ptr_to([slot_idx]))
                        else:
                            txl.assign(slot_tokens[i], txl.Cast("int64", slot_idx))
                for i in range(UNROLL):
                    slot_idx = (step * UNROLL + i) * TILE + tidx
                    expert = as_i32(slot_experts[i])
                    if has_expand:
                        token = as_i32(slot_tokens[i])
                        with txl.if_(txl.And(expert < 64, txl.And(token >= window_start, token < window_end))), txl.then_():
                            txl.ptx.atom.shared.or_.b64(mask_word, s_mask.ptr_to([token - window_start]), txl.shift_left(txl.uint64(1), txl.Cast("uint64", expert)))
                    with txl.if_(window_idx == 0), txl.then_():
                        txl.ptx["match.any.sync.b32"](peer_masks[i], expert, txl.uint32(FULL_MASK))
                        is_leader = (peer_masks[i] & lane_mask_lt) == txl.uint32(0)
                        with txl.if_(txl.And(is_leader, expert < 64)), txl.then_():
                            peer_count = as_i32(txl.popcount(peer_masks[i]))
                            txl.ptx.red.shared.add.s32(s_expert_total.ptr_to([expert]), peer_count)
                            # a 1024-slot step lies wholly before this CTA's tile or not at all
                            with txl.if_(slot_idx < tile_start), txl.then_():
                                txl.ptx.red.shared.add.s32(s_expert_before.ptr_to([expert]), peer_count)

        def scan_expert_starts():
            """
            Warp 0: exclusive scan of the expert totals, two experts a lane, stored to
            s_expert_start and cu.
            """
            total_even = txl.local_scalar(txl.i32)
            total_odd = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(total_even, s_expert_total.ptr_to([2 * lane]))
            txl.ptx.ld.shared.s32(total_odd, s_expert_total.ptr_to([2 * lane + 1]))
            pair_total = txl.local_scalar(txl.i32, init=total_even + total_odd)
            pair_incl = txl.local_scalar(txl.i32, init=pair_total)
            peer = txl.local_scalar(txl.i32)
            warp_inclusive_scan(pair_incl, peer, lane)
            pair_excl = pair_incl - pair_total
            txl.ptx.st.shared.s32(s_expert_start.ptr_to([2 * lane]), pair_excl)
            txl.ptx.st.shared.s32(s_expert_start.ptr_to([2 * lane + 1]), pair_excl + total_even)
            with txl.if_(cta_idx == 0), txl.then_():
                store_expert_offsets(cu, G, lane, pair_excl, total_even, pair_incl)

        def count_warp_slots():
            """
            Load this thread's slot of the tile, rank it in its warp and store the warp's counts.
            """
            slot_idx = tile_start + tidx
            slot_expert = txl.local_scalar(txl.i64, init=txl.int64(64))
            slot_token = txl.local_scalar(txl.i64, init=txl.int64(0))
            with txl.if_(slot_idx < S), txl.then_():
                txl.ptx.ld.global_.nc.s64(slot_expert, e.ptr_to([slot_idx]))
                if has_expand:
                    txl.ptx.ld.global_.nc.s64(slot_token, x.ptr_to([slot_idx]))
            expert = as_i32(slot_expert)
            peer_mask = txl.local_scalar(txl.u32)
            txl.ptx["match.any.sync.b32"](peer_mask, expert, txl.uint32(FULL_MASK))
            lane_rank = as_i32(txl.popcount(peer_mask & lane_mask_lt))
            with txl.if_(txl.And(lane_rank == 0, expert < 64)), txl.then_():
                txl.ptx.st.shared.s32(s_warp_count.ptr_to([expert * NUM_WARPS + warp]), as_i32(txl.popcount(peer_mask)))
            return slot_idx, expert, slot_token, lane_rank

        def scan_warp_counts():
            """
            Per-expert exclusive scan over the warps: warp w scans experts w and w + 32.
            """
            for half in range(2):
                scan_expert = warp + 32 * half
                count = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(count, s_warp_count.ptr_to([scan_expert * NUM_WARPS + lane]))
                count_incl = txl.local_scalar(txl.i32, init=count)
                peer = txl.local_scalar(txl.i32)
                warp_inclusive_scan(count_incl, peer, lane)
                txl.ptx.st.shared.s32(s_warp_count.ptr_to([scan_expert * NUM_WARPS + lane]), count_incl - count)

        def store_slot_maps(slot_idx, expert, slot_token, lane_rank):
            with txl.if_(slot_idx < S), txl.then_():
                expert_start = txl.local_scalar(txl.i32)
                tile_before = txl.local_scalar(txl.i32)
                warp_before = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(expert_start, s_expert_start.ptr_to([expert]))
                txl.ptx.ld.shared.s32(tile_before, s_expert_before.ptr_to([expert]))
                txl.ptx.ld.shared.s32(warp_before, s_warp_count.ptr_to([expert * NUM_WARPS + warp]))
                row_idx = txl.local_scalar(txl.i32, init=expert_start + tile_before + warp_before + lane_rank)
                txl.ptx.st.global_.s32(row_of.ptr_to([slot_idx]), row_idx)
                txl.ptx.st.global_.s32(slot_of.ptr_to([row_idx]), slot_idx)
                if has_expand:
                    token = as_i32(slot_token)
                    txl.ptx.st.global_.s32(a_idx.ptr_to([row_idx]), token)
                    txl.ptx.st.global_.s32(table.ptr_to([expert * T + token]), row_idx)
                else:
                    txl.ptx.st.global_.s32(a_idx.ptr_to([row_idx]), slot_idx)

        zero_counters()

        tokens_per_cta = (T + num_ctas - 1) // num_ctas  # tokens owned by this CTA: [token_start, token_start + tokens_per_cta)
        token_start = cta_idx * tokens_per_cta
        num_windows = txl.max(1, (tokens_per_cta + MASK_WINDOW - 1) // MASK_WINDOW)
        tile_start = cta_idx * TILE
        UNROLL = 4
        slot_experts = txl.alloc_local([UNROLL], txl.i64)
        slot_tokens = txl.alloc_local([UNROLL], txl.i64)
        peer_masks = txl.alloc_local([UNROLL], txl.u32)
        mask_word = txl.local_scalar(txl.u64)
        with txl.serial(0, num_windows) as window_idx:
            window_start = token_start + window_idx * MASK_WINDOW
            window_end = txl.min(txl.min(token_start + tokens_per_cta, T), window_start + MASK_WINDOW)
            with txl.serial(0, MASK_WINDOW // TILE) as i:
                txl.ptx.st.shared.u64(s_mask.ptr_to([i * TILE + tidx]), txl.uint64(0))
            txl.cuda.cta_sync()
            count_and_mark_slots(window_idx, window_start, window_end)
            txl.cuda.cta_sync()
            if has_expand:
                with txl.serial(0, MASK_WINDOW // TILE) as i:
                    token = window_start + i * TILE + tidx
                    with txl.if_(token < window_end), txl.then_():
                        txl.ptx.ld.shared.u64(mask_word, s_mask.ptr_to([i * TILE + tidx]))
                        txl.ptx.st.global_.u64(mask.ptr_to([token]), mask_word)
            txl.cuda.cta_sync()

        with txl.if_(warp == 0), txl.then_():
            scan_expert_starts()

        slot_idx, expert, slot_token, lane_rank = count_warp_slots()
        txl.cuda.cta_sync()
        scan_warp_counts()
        txl.cuda.cta_sync()
        store_slot_maps(slot_idx, expert, slot_token, lane_rank)

    return entry_function(group_slots.compile(TARGET))


CLUSTER_SIZE = 16  # CTAs per cluster (a non-portable size, which B200 launches); CTA r of a cluster owns slots [r * slice_len, r * slice_len + slice_len)
NUM_CLUSTERS = 5  # each cluster computes the histograms and scan redundantly and stores 1/NUM_CLUSTERS of the slots (five shorten the large-row rank and store tail)
CLUSTER_SLOTS_PER_CTA = (1024, 2048, 4096, 8192)  # slots per CTA held in registers: the smallest that fits (fewer registers); larger S rescans


@functools.cache
def compile_group_slots_cluster(G: int, has_expand: bool, slots_per_cta: int):
    """
    The cluster kernel: NUM_CLUSTERS clusters of CLUSTER_SIZE CTAs, each CTA holding up to
    slots_per_cta slots in registers.
    """
    NUM_WARPS = TILE // 32
    NUM_TILES = slots_per_cta // TILE  # 1024-slot tiles per CTA, held in registers

    # e and x are the int64 ids viewed as int32 pairs: the low word of slot j at 2 * j
    @txl.kernel()
    def group_slots_cluster(e: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), x: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), a_idx: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), row_of: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), slot_of: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), mask: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), table: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), S: txl.int32, T: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=CLUSTER_SIZE * NUM_CLUSTERS, block=NUM_WARPS * 32, cluster=CLUSTER_SIZE))
        cluster_rank = txl.cuda.cluster_cta_id("x")
        # peers' shared memory may be written only once every CTA of the cluster has started: arrive
        # now, wait just before the first remote store (by then the wait is free)
        txl.ptx.barrier.cluster.arrive.relaxed()
        cluster_idx = txl.cta_id() // CLUSTER_SIZE  # this cluster stores warp-tiles (tile, warp) with (tile * NUM_WARPS + warp) % NUM_CLUSTERS == cluster_idx
        # and owns the mask tokens of 32-token blocks b with b % NUM_CLUSTERS == cluster_idx
        # (interleaved: every CTA's slots spread their mask bits over all clusters, whatever the token
        # order of the slots)
        cluster_tokens = ((T + 32 * NUM_CLUSTERS - 1) // (32 * NUM_CLUSTERS)) * 32  # owned tokens per cluster (upper bound)
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        smem = txl.smem_pool()
        s_hist = smem.alloc([64], txl.i32)  # this CTA's slice histogram
        s_all_hist = smem.alloc([CLUSTER_SIZE * 64], txl.i32)  # every CTA's histogram, pushed here before the cluster barrier
        s_slice_base = smem.alloc([64], txl.i32)  # cu[e] + slots of e before this CTA's slice
        s_tile_count = smem.alloc([NUM_TILES * 64 * NUM_WARPS], txl.i32)  # (tile, expert, warp) counts, then their exclusive scan
        lane_mask_lt = txl.shift_left(txl.uint32(1), as_u32(lane)) - txl.uint32(1)
        slice_len = (S + CLUSTER_SIZE - 1) // CLUSTER_SIZE
        slice_start = cluster_rank * slice_len
        num_slots = txl.max(0, txl.min(S, slice_start + slice_len) - slice_start)
        tokens_per_cta = (cluster_tokens + CLUSTER_SIZE - 1) // CLUSTER_SIZE
        remote_addr = txl.local_scalar(txl.u32)

        def cluster_sync():
            txl.ptx.barrier.cluster.arrive.release()
            txl.ptx.barrier.cluster.wait.acquire()

        def start_profile_range(name):
            range_token = txl.alloc_local([1], "uint32")
            txl.assign(range_token[0], txl.cuda.iket.sentinel_token(name))
            with txl.if_(lane == 0), txl.then_():
                txl.assign(range_token[0], txl.cuda.iket.range_start(name))
            return range_token

        def load_slots():
            """
            Load the expert (64 past the end) and token of each slot of the slice into registers.
            """
            slot_experts = txl.alloc_local([NUM_TILES], txl.i32)
            slot_tokens = txl.alloc_local([NUM_TILES], txl.i32)
            peer_masks = txl.alloc_local([NUM_TILES], txl.u32)
            for tile in range(NUM_TILES):
                local_idx = tile * TILE + tidx
                txl.assign(slot_experts[tile], txl.int32(64))
                with txl.if_(local_idx < num_slots), txl.then_():
                    txl.ptx.ld.global_.nc.s32(slot_experts[tile], e.ptr_to([2 * (slice_start + local_idx)]))
                    if has_expand:
                        txl.ptx.ld.global_.nc.s32(slot_tokens[tile], x.ptr_to([2 * (slice_start + local_idx)]))
            return slot_experts, slot_tokens, peer_masks

        def zero_counts():
            with txl.if_(tidx < 64), txl.then_():
                txl.ptx.st.shared.s32(s_hist.ptr_to([tidx]), txl.int32(0))
            for tile in range(NUM_TILES):
                with txl.if_(tile * TILE < num_slots), txl.then_():
                    for i in range((64 * NUM_WARPS) // TILE):
                        txl.ptx.st.shared.s32(s_tile_count.ptr_to([tile * 64 * NUM_WARPS + i * TILE + tidx]), txl.int32(0))

        def zero_owned_mask():
            with txl.serial(0, (tokens_per_cta + TILE - 1) // TILE) as i:
                owned_idx = cluster_rank * tokens_per_cta + i * TILE + tidx  # owned-token ordinal
                token = ((owned_idx >> 5) * NUM_CLUSTERS + cluster_idx) * 32 + (owned_idx & 31)
                with txl.if_(txl.And(token < T, i * TILE + tidx < tokens_per_cta)), txl.then_():
                    txl.ptx.st.global_.u64(mask.ptr_to([token]), txl.uint64(0))

        def count_slots():
            """
            Per (tile, warp) match.any counts, and the slice histogram.
            """
            for tile in range(NUM_TILES):
                with txl.if_(tile * TILE < num_slots), txl.then_():
                    txl.ptx["match.any.sync.b32"](peer_masks[tile], slot_experts[tile], txl.uint32(FULL_MASK))
                    with txl.if_(txl.And((peer_masks[tile] & lane_mask_lt) == txl.uint32(0), slot_experts[tile] < 64)), txl.then_():
                        peer_count = as_i32(txl.popcount(peer_masks[tile]))
                        txl.ptx.red.shared.add.s32(s_hist.ptr_to([slot_experts[tile]]), peer_count)
                        txl.ptx.st.shared.s32(s_tile_count.ptr_to([(tile * 64 + slot_experts[tile]) * NUM_WARPS + warp]), peer_count)

        def push_histogram():
            """
            The slice histogram is complete: push it to every peer now, under the scan that follows.
            """
            txl.ptx.barrier.cluster.wait()
            with txl.if_(tidx < 64), txl.then_():
                count = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(count, s_hist.ptr_to([tidx]))
                for peer_rank in range(CLUSTER_SIZE):
                    txl.ptx.mapa.shared__cluster.u32(remote_addr, txl.cuda.cvta_generic_to_shared(s_all_hist.ptr_to([cluster_rank * 64 + tidx])), txl.uint32(peer_rank))
                    txl.ptx["st.shared::cluster.s32"](remote_addr, count)

        def scan_tile_counts():
            """
            Per-expert exclusive scan over (tile, warp): warp w scans experts w and w + 32.
            """
            count = txl.local_scalar(txl.i32)
            count_incl = txl.local_scalar(txl.i32)
            peer = txl.local_scalar(txl.i32)
            carry = txl.local_scalar(txl.i32)
            tile_total = txl.local_scalar(txl.i32)
            for half in range(2):
                scan_expert = warp + 32 * half
                txl.assign(carry, txl.int32(0))
                for tile in range(NUM_TILES):
                    with txl.if_(tile * TILE < num_slots), txl.then_():
                        txl.ptx.ld.shared.s32(count, s_tile_count.ptr_to([(tile * 64 + scan_expert) * NUM_WARPS + lane]))
                        txl.assign(count_incl, count)
                        warp_inclusive_scan(count_incl, peer, lane)
                        txl.ptx.st.shared.s32(s_tile_count.ptr_to([(tile * 64 + scan_expert) * NUM_WARPS + lane]), carry + count_incl - count)
                        txl.ptx.shfl_sync.idx.b32(tile_total, count_incl, txl.uint32(31), txl.uint32(31), txl.uint32(FULL_MASK))
                        txl.assign(carry, carry + tile_total)

        def or_mask_bits():
            for tile in range(NUM_TILES):
                with txl.if_(txl.And(tile * TILE + tidx < num_slots, ((slot_tokens[tile] >> 5) % NUM_CLUSTERS) == cluster_idx)), txl.then_():
                    txl.ptx.red.global_.or_.b64(mask.ptr_to([slot_tokens[tile]]), txl.shift_left(txl.uint64(1), txl.Cast("uint64", slot_experts[tile])))

        def scan_slice_bases():
            """
            Warp 0: the totals and this slice's offsets for experts 2 * lane and 2 * lane + 1,
            then the cu scan.
            """
            hist_even = txl.alloc_local([CLUSTER_SIZE], txl.i32)
            hist_odd = txl.alloc_local([CLUSTER_SIZE], txl.i32)
            for peer_rank in range(CLUSTER_SIZE):
                txl.ptx.ld.shared.v2.b32(hist_even[peer_rank], hist_odd[peer_rank], s_all_hist.ptr_to([peer_rank * 64 + 2 * lane]))
            total_even = txl.local_scalar(txl.i32, init=txl.int32(0))
            total_odd = txl.local_scalar(txl.i32, init=txl.int32(0))
            before_even = txl.local_scalar(txl.i32, init=txl.int32(0))
            before_odd = txl.local_scalar(txl.i32, init=txl.int32(0))
            for peer_rank in range(CLUSTER_SIZE):
                txl.assign(total_even, total_even + hist_even[peer_rank])
                txl.assign(total_odd, total_odd + hist_odd[peer_rank])
                txl.assign(before_even, before_even + txl.Select(cluster_rank > peer_rank, hist_even[peer_rank], txl.int32(0)))
                txl.assign(before_odd, before_odd + txl.Select(cluster_rank > peer_rank, hist_odd[peer_rank], txl.int32(0)))
            pair_total = txl.local_scalar(txl.i32, init=total_even + total_odd)
            pair_incl = txl.local_scalar(txl.i32, init=pair_total)
            peer = txl.local_scalar(txl.i32)
            warp_inclusive_scan(pair_incl, peer, lane)
            pair_excl = txl.local_scalar(txl.i32, init=pair_incl - pair_total)
            txl.ptx.st.shared.s32(s_slice_base.ptr_to([2 * lane]), pair_excl + before_even)
            txl.ptx.st.shared.s32(s_slice_base.ptr_to([2 * lane + 1]), pair_excl + total_even + before_odd)
            with txl.if_(txl.And(cluster_rank == 0, cluster_idx == 0)), txl.then_():
                store_expert_offsets(cu, G, lane, pair_excl, total_even, pair_incl)

        def store_slot_maps():
            """
            Rank and store the slots of the warp-tiles this cluster owns.
            """
            slice_base = txl.local_scalar(txl.i32)
            warp_base = txl.local_scalar(txl.i32)
            for tile in range(NUM_TILES):
                local_idx = tile * TILE + tidx
                with txl.if_(txl.And(local_idx < num_slots, (tile * NUM_WARPS + warp) % NUM_CLUSTERS == cluster_idx)), txl.then_():
                    slot_idx = slice_start + local_idx
                    txl.ptx.ld.shared.s32(slice_base, s_slice_base.ptr_to([slot_experts[tile]]))
                    txl.ptx.ld.shared.s32(warp_base, s_tile_count.ptr_to([(tile * 64 + slot_experts[tile]) * NUM_WARPS + warp]))
                    row_idx = txl.local_scalar(txl.i32, init=slice_base + warp_base + as_i32(txl.popcount(peer_masks[tile] & lane_mask_lt)))
                    txl.ptx.st.global_.s32(row_of.ptr_to([slot_idx]), row_idx)
                    txl.ptx.st.global_.s32(slot_of.ptr_to([row_idx]), slot_idx)
                    if has_expand:
                        txl.ptx.st.global_.s32(a_idx.ptr_to([row_idx]), slot_tokens[tile])
                        txl.ptx.st.global_.s32(table.ptr_to([slot_experts[tile] * T + slot_tokens[tile]]), row_idx)
                    else:
                        txl.ptx.st.global_.s32(a_idx.ptr_to([row_idx]), slot_idx)

        stage_range = start_profile_range("stage")
        slot_experts, slot_tokens, peer_masks = load_slots()
        zero_counts()
        if has_expand:
            zero_owned_mask()
        txl.cuda.cta_sync()
        count_slots()
        txl.cuda.cta_sync()
        push_histogram()
        scan_tile_counts()
        txl.cuda.iket.range_end(stage_range[0])

        sync_range = start_profile_range("csync1")
        cluster_sync()  # all histograms pushed; the mask zeros visible cluster-wide
        txl.cuda.iket.range_end(sync_range[0])

        mask_range = start_profile_range("mask-dsmem")
        if has_expand:
            or_mask_bits()
        with txl.if_(warp == 0), txl.then_():
            scan_slice_bases()
        txl.cuda.cta_sync()
        txl.cuda.iket.range_end(mask_range[0])

        rank_range = start_profile_range("rank")
        store_slot_maps()
        txl.cuda.iket.range_end(rank_range[0])

    return entry_function(group_slots_cluster.compile(TARGET))


def bf16_expert_group_slots_sm100(expert_idxs, expand_idx, num_groups, num_tokens):
    """
    bf16_expert_group_slots on SM100: the cluster kernel up to CLUSTER_SIZE * 8192 slots, else
    the rescan kernel.
    """
    assert 0 < num_groups <= 64
    S, device = expert_idxs.shape[0], expert_idxs.device
    cu = torch.empty(num_groups + 1, dtype=torch.int32, device=device)
    a_idx = torch.empty(S, dtype=torch.int32, device=device)
    row_of = torch.empty(S, dtype=torch.int32, device=device)
    slot_of = torch.empty(S, dtype=torch.int32, device=device)
    has_expand = expand_idx is not None
    if has_expand:
        mask = torch.empty(num_tokens, dtype=torch.int64, device=device)
        table = torch.empty(num_tokens * num_groups, dtype=torch.int32, device=device)
    else:
        mask = torch.empty(1, dtype=torch.int64, device=device)
        table = torch.empty(1, dtype=torch.int32, device=device)
    slots_per_cta = next((slots for slots in CLUSTER_SLOTS_PER_CTA if S <= CLUSTER_SIZE * slots), None)
    if slots_per_cta is not None:
        fn = compile_group_slots_cluster(num_groups, has_expand, slots_per_cta)
        fn(expert_idxs.view(torch.int32), (expand_idx if has_expand else expert_idxs).view(torch.int32), cu, a_idx, row_of, slot_of, mask.view(torch.uint64), table, S, num_tokens if has_expand else 0)
        if not has_expand:
            return GroupSlots(cu, a_idx, row_of, slot_of, None, None)
        return GroupSlots(cu, a_idx, row_of, slot_of, mask, table)
    fn = compile_group_slots(num_groups, has_expand)
    fn(expert_idxs, expand_idx if has_expand else expert_idxs, cu, a_idx, row_of, slot_of, mask.view(torch.uint64), table, S, num_tokens if has_expand else 0)
    if not has_expand:
        return GroupSlots(cu, a_idx, row_of, slot_of, None, None)
    return GroupSlots(cu, a_idx, row_of, slot_of, mask, table)


GATHER_SUM_BATCH = 8  # experts per prefetched batch
GATHER_SUM_CTAS_PER_SM = {2880: 3}  # by H; default 4
NUM_SMS = 148


@functools.cache
def compile_gather_sum(G: int, H: int):
    """
    Register-only gather_sum (used when USE_GATHER_SUM_TMA and USE_MULTI_TOKEN_UNITS are off): one
    token per CTA step, the next token's rows prefetched in batches of GATHER_SUM_BATCH experts.
    """
    NUM_VECS = H // 8
    NUM_WARPS = (NUM_VECS + 31) // 32
    BATCH = min(GATHER_SUM_BATCH, G)
    GRID = NUM_SMS * GATHER_SUM_CTAS_PER_SM.get(H, 4)

    @txl.kernel()
    def gather_sum(grad: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), mask: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), table: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), T: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=txl.min(T, GRID), block=NUM_WARPS * 32))
        tidx = txl.thread_id()
        grid_size = txl.min(T, GRID)
        vec_ok = tidx < NUM_VECS
        token_idx = txl.local_scalar(txl.i32, init=txl.cta_id())
        lowest_bit = txl.local_scalar(txl.u64)
        acc = txl.alloc_local([8], txl.f32)
        words = txl.alloc_local([4 * BATCH], txl.u32)
        rows_cur = txl.alloc_local([BATCH], txl.i32)  # current token's grouped rows (-1: none)
        rows_next = txl.alloc_local([BATCH], txl.i32)  # next token's
        batch_experts = txl.alloc_local([BATCH], txl.i32)
        remaining_cur = txl.local_scalar(txl.u64)
        remaining_next = txl.local_scalar(txl.u64)
        next_mask = txl.local_scalar(txl.u64, init=txl.uint64(0))
        next_next_mask = txl.local_scalar(txl.u64)
        row_idx = txl.local_scalar(txl.i32)

        def take_batch(expert_bits, token, rows):
            # pop the next BATCH experts of expert_bits (ascending) and load their grouped rows
            for k in range(BATCH):
                txl.assign(lowest_bit, expert_bits & (txl.uint64(0) - expert_bits))
                txl.assign(batch_experts[k], txl.Select(lowest_bit != txl.uint64(0), as_i32(txl.popcount(lowest_bit - txl.uint64(1))), txl.int32(-1)))
                txl.assign(expert_bits, expert_bits ^ lowest_bit)
            for k in range(BATCH):
                txl.assign(rows[k], txl.int32(-1))
                with txl.if_(batch_experts[k] >= 0), txl.then_():
                    txl.ptx.ld.global_.nc.s32(rows[k], table.ptr_to([batch_experts[k] * T + token]))

        def load_rows(rows):
            for k in range(BATCH):
                with txl.if_(txl.And(rows[k] >= 0, vec_ok)), txl.then_():
                    txl.ptx.ld.global_.nc.v4.b32(words[4 * k], words[4 * k + 1], words[4 * k + 2], words[4 * k + 3], grad.ptr_to([txl.Cast("int64", rows[k]) * (H // 2) + tidx * 4]))

        def add_loaded_rows(rows):
            for k in range(BATCH):
                with txl.if_(rows[k] >= 0), txl.then_():
                    for pair in range(4):
                        add_bf16x2(acc, words, 4 * k + pair, pair)

        # prologue: rows of the first token, mask of the second
        txl.assign(remaining_cur, txl.uint64(0))
        with txl.if_(token_idx < T), txl.then_():
            txl.ptx.ld.global_.nc.u64(remaining_cur, mask.ptr_to([token_idx]))
        take_batch(remaining_cur, token_idx, rows_cur)
        with txl.if_(token_idx + grid_size < T), txl.then_():
            txl.ptx.ld.global_.nc.u64(next_mask, mask.ptr_to([token_idx + grid_size]))
        with txl.while_(token_idx < T):
            load_rows(rows_cur)
            txl.assign(next_next_mask, txl.uint64(0))
            with txl.if_(token_idx + 2 * grid_size < T), txl.then_():
                txl.ptx.ld.global_.nc.u64(next_next_mask, mask.ptr_to([token_idx + 2 * grid_size]))
            txl.assign(remaining_next, next_mask)
            take_batch(remaining_next, token_idx + grid_size, rows_next)
            for i in range(8):
                txl.assign(acc[i], txl.float32(0))
            add_loaded_rows(rows_cur)
            # experts past the first BATCH (rare): one at a time, still ascending
            with txl.while_(remaining_cur != txl.uint64(0)):
                txl.assign(lowest_bit, remaining_cur & (txl.uint64(0) - remaining_cur))
                txl.assign(remaining_cur, remaining_cur ^ lowest_bit)
                txl.ptx.ld.global_.nc.s32(row_idx, table.ptr_to([as_i32(txl.popcount(lowest_bit - txl.uint64(1))) * T + token_idx]))
                with txl.if_(vec_ok), txl.then_():
                    txl.ptx.ld.global_.nc.v4.b32(words[0], words[1], words[2], words[3], grad.ptr_to([txl.Cast("int64", row_idx) * (H // 2) + tidx * 4]))
                    for pair in range(4):
                        add_bf16x2(acc, words, pair, pair)
            with txl.if_(vec_ok), txl.then_():
                pack_bf16x2(words, acc)
                txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", token_idx) * (H // 2) + tidx * 4]), words[0], words[1], words[2], words[3])
            for k in range(BATCH):
                txl.assign(rows_cur[k], rows_next[k])
            txl.assign(remaining_cur, remaining_next)
            txl.assign(next_mask, next_next_mask)
            txl.assign(token_idx, token_idx + grid_size)

    return entry_function(gather_sum.compile(TARGET))


USE_GATHER_SUM_TMA = True
GATHER_SUM_TMA_CFG = {2048: (2, 6), 2880: (2, 4)}  # by H: (pipeline stages, CTAs per SM); default (2, 2)
GATHER_SUM_TMA_BATCH = 4  # grouped rows per unit
BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
EVICT_FIRST = {"gs": False, "rows": False, "seg": True}  # mark stream-once reads L2 evict_first: it helps segment_sum, not gather_rows or gather_sum


@functools.cache
def compile_gather_sum_tma(G: int, H: int):
    """
    Persistent: warp 0 turns each token's mask bits into units of up to UNIT_ROWS grouped rows, one
    bulk copy per row into a STAGES-deep shared-memory ring; every thread adds a unit's rows in
    ascending expert order.
    """
    NUM_VECS = H // 8
    NUM_WARPS = (NUM_VECS + 31) // 32
    UNIT_ROWS = min(GATHER_SUM_TMA_BATCH, G)
    STAGES, CTAS_PER_SM = GATHER_SUM_TMA_CFG.get(H, (2, 2))
    GRID = NUM_SMS * CTAS_PER_SM
    ROW_WORDS = H // 2  # u32 words per row
    ROW_BYTES = H * 2
    assert ROW_BYTES % 16 == 0 and UNIT_ROWS <= 32

    @txl.kernel()
    def gather_sum_tma(grad: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), mask: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), table: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), T: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=txl.min(T, GRID), block=NUM_WARPS * 32))
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        grid_size = txl.min(T, GRID)
        vec_ok = tidx < NUM_VECS
        smem = txl.smem_pool()
        s_stage = smem.alloc([STAGES * UNIT_ROWS * ROW_WORDS], txl.u32, align=128)
        s_full = smem.alloc([STAGES], txl.u64, align=8)
        s_unit_meta = smem.alloc([2 * STAGES], txl.i32)  # per slot: token, rows | last << 8 (-1: no more units)
        with txl.if_(tidx < STAGES), txl.then_():
            txl.ptx.mbarrier.init.shared.b64(s_full.ptr_to([tidx]), txl.uint32(1))
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        # producer state (warp 0, uniform across its lanes)
        token_idx = txl.local_scalar(txl.i32, init=txl.cta_id())
        token_mask = txl.local_scalar(txl.u64, init=txl.uint64(0))
        next_mask = txl.local_scalar(txl.u64, init=txl.uint64(0))
        lowest_bit = txl.local_scalar(txl.u64)
        unit_experts = txl.alloc_local([UNIT_ROWS], txl.i32)
        lane_expert = txl.local_scalar(txl.i32)
        num_rows = txl.local_scalar(txl.i32)
        row_idx = txl.local_scalar(txl.i32)
        with txl.if_(warp == 0), txl.then_():
            with txl.if_(token_idx < T), txl.then_():
                txl.ptx.ld.global_.nc.u64(token_mask, mask.ptr_to([token_idx]))
            with txl.if_(token_idx + grid_size < T), txl.then_():
                txl.ptx.ld.global_.nc.u64(next_mask, mask.ptr_to([token_idx + grid_size]))

        def produce(slot):
            with txl.if_(token_idx < T):
                with txl.then_():
                    txl.assign(num_rows, txl.int32(0))
                    txl.assign(lane_expert, txl.int32(-1))
                    for k in range(UNIT_ROWS):
                        txl.assign(lowest_bit, token_mask & (txl.uint64(0) - token_mask))
                        txl.assign(unit_experts[k], txl.Select(lowest_bit != txl.uint64(0), as_i32(txl.popcount(lowest_bit - txl.uint64(1))), txl.int32(-1)))
                        txl.assign(token_mask, token_mask ^ lowest_bit)
                        txl.assign(num_rows, num_rows + txl.Select(unit_experts[k] >= 0, txl.int32(1), txl.int32(0)))
                        txl.assign(lane_expert, txl.Select(lane == k, unit_experts[k], lane_expert))
                    is_last = txl.Select(token_mask == txl.uint64(0), txl.int32(1), txl.int32(0))
                    with txl.if_(lane == 0), txl.then_():
                        txl.ptx.st.shared.s32(s_unit_meta.ptr_to([2 * slot]), token_idx)
                        txl.ptx.st.shared.s32(s_unit_meta.ptr_to([2 * slot + 1]), num_rows + is_last * 256)
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(s_full.ptr_to([slot]), as_u32(num_rows * ROW_BYTES))
                    txl.ptx["bar.warp.sync"](txl.uint32(FULL_MASK))
                    with txl.if_(lane_expert >= 0), txl.then_():
                        txl.ptx.ld.global_.nc.s32(row_idx, table.ptr_to([lane_expert * T + token_idx]))
                        txl.ptx[BULK_G2S](s_stage.ptr_to([(slot * UNIT_ROWS + lane) * ROW_WORDS]), grad.ptr_to([txl.Cast("int64", row_idx) * ROW_WORDS]), txl.uint32(ROW_BYTES), s_full.ptr_to([slot]))
                    with txl.if_(token_mask == txl.uint64(0)), txl.then_():
                        txl.assign(token_idx, token_idx + grid_size)
                        txl.assign(token_mask, next_mask)
                        txl.assign(next_mask, txl.uint64(0))
                        with txl.if_(token_idx + grid_size < T), txl.then_():
                            txl.ptx.ld.global_.nc.u64(next_mask, mask.ptr_to([token_idx + grid_size]))
                with txl.else_():
                    with txl.if_(lane == 0), txl.then_():
                        txl.ptx.st.shared.s32(s_unit_meta.ptr_to([2 * slot + 1]), txl.int32(-1))
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(s_full.ptr_to([slot]), txl.uint32(0))

        with txl.if_(warp == 0), txl.then_():
            for stage in range(STAGES):
                produce(stage)

        acc = txl.alloc_local([8], txl.f32)
        words = txl.alloc_local([4], txl.u32)
        unit_idx = txl.local_scalar(txl.i32, init=txl.int32(0))
        is_running = txl.local_scalar(txl.i32, init=txl.int32(1))
        token = txl.local_scalar(txl.i32)
        unit_info = txl.local_scalar(txl.i32)
        slot = txl.local_scalar(txl.i32)
        for i in range(8):
            txl.assign(acc[i], txl.float32(0))
        with txl.while_(is_running > 0):
            txl.assign(slot, unit_idx % STAGES)
            txl.cuda.mbarrier_wait(s_full.ptr_to([slot]), as_u32((unit_idx // STAGES) & 1))
            txl.ptx.ld.shared.s32(token, s_unit_meta.ptr_to([2 * slot]))
            txl.ptx.ld.shared.s32(unit_info, s_unit_meta.ptr_to([2 * slot + 1]))
            with txl.if_(unit_info < 0):
                with txl.then_():
                    txl.assign(is_running, txl.int32(0))
                with txl.else_():
                    with txl.if_(vec_ok), txl.then_():
                        for k in range(UNIT_ROWS):
                            with txl.if_(k < (unit_info & 255)), txl.then_():
                                txl.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], s_stage.ptr_to([(slot * UNIT_ROWS + k) * ROW_WORDS + tidx * 4]))
                                for pair in range(4):
                                    add_bf16x2(acc, words, pair, pair)
                    with txl.if_(unit_info >= 256), txl.then_():
                        with txl.if_(vec_ok), txl.then_():
                            pack_bf16x2(words, acc)
                            txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", token) * ROW_WORDS + tidx * 4]), words[0], words[1], words[2], words[3])
                        for i in range(8):
                            txl.assign(acc[i], txl.float32(0))
            txl.cuda.cta_sync()  # every thread is done with the slot
            with txl.if_(txl.And(is_running > 0, warp == 0)), txl.then_():
                produce(slot)
            txl.assign(unit_idx, unit_idx + 1)

    return entry_function(gather_sum_tma.compile(TARGET))


USE_PRODUCER_WARP = True  # a dedicated producer warp and per-slot empty barriers (no CTA barrier per unit)
USE_SPIN_WAIT = True  # consumers wait with a hint-free try_wait spin
USE_MULTI_TOKEN_UNITS = True  # multi-token units (below) instead of one token per unit
MULTI_TOKEN_CFG = {2048: (4, 3, 4), 2880: (4, 2, 4)}  # by H: (pipeline stages, CTAs per SM, grouped rows per unit)
MULTI_TOKEN_CFG_BY_G = {(64, 2048): (3, 3, 5), (8, 2048): (6, 3, 3)}  # by (G, H), ahead of MULTI_TOKEN_CFG: sparse EP4 packs more consecutive tokens per unit, dense local masks trade unit width for a deeper ring
LAST_BIT = 1 << 30
NOROW_BIT = 1 << 29


@functools.cache
def compile_gather_sum_multi_token(G: int, H: int):
    """
    Persistent: the producer warp packs the grouped rows of consecutive tokens (ascending expert
    order within a token) into units of UNIT_ROWS entries, one bulk copy per row into a STAGES-deep
    ring; an entry is (token, last-of-token, no-row). Every consumer thread adds a unit's rows; a
    token's last entry rounds and stores it. Token masks are read 32 tokens at a time (one per
    lane), the next window prefetched.
    """
    NUM_VECS = H // 8
    NUM_WARPS = (NUM_VECS + 31) // 32
    STAGES, CTAS_PER_SM, UNIT_ROWS = MULTI_TOKEN_CFG_BY_G.get((G, H), MULTI_TOKEN_CFG.get(H, (2, 2, 4)))
    GRID = NUM_SMS * CTAS_PER_SM
    ROW_WORDS = H // 2  # u32 words per row
    ROW_BYTES = H * 2
    assert ROW_BYTES % 16 == 0 and UNIT_ROWS <= 32

    PRODUCER_WARP = NUM_WARPS if USE_PRODUCER_WARP else 0  # a dedicated extra warp, or consumer warp 0

    @txl.kernel()
    def gather_sum_tma3(grad: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), mask: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), table: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), T: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=txl.min(T, GRID), block=(NUM_WARPS + (1 if USE_PRODUCER_WARP else 0)) * 32))
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        cta_idx = txl.cta_id()
        grid_size = txl.min(T, GRID)
        vec_ok = tidx < NUM_VECS
        smem = txl.smem_pool()
        s_stage = smem.alloc([STAGES * UNIT_ROWS * ROW_WORDS], txl.u32, align=128)
        s_full = smem.alloc([STAGES], txl.u64, align=8)
        s_unit_len = smem.alloc([STAGES], txl.i32)  # entries in the unit (-1: no more units)
        s_unit_entries = smem.alloc([STAGES * UNIT_ROWS], txl.i32)  # token | LAST_BIT | NOROW_BIT
        s_empty = smem.alloc([STAGES], txl.u64, align=8)  # slot consumed: one arrival per consumer thread
        with txl.if_(tidx < STAGES), txl.then_():
            txl.ptx.mbarrier.init.shared.b64(s_full.ptr_to([tidx]), txl.uint32(1))
            txl.ptx.mbarrier.init.shared.b64(s_empty.ptr_to([tidx]), txl.uint32(NUM_WARPS * 32))
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        # producer state (uniform across the producer warp's lanes except the window masks and the
        # per-entry registers)
        window_masks = txl.local_scalar(txl.u64, init=txl.uint64(0))  # mask of token cta_idx + (window_start + lane) * grid_size
        next_window_masks = txl.local_scalar(txl.u64, init=txl.uint64(0))
        window_start = txl.local_scalar(txl.i32, init=txl.int32(0))
        window_pos = txl.local_scalar(txl.i32, init=txl.int32(0))
        token_idx = txl.local_scalar(txl.i32, init=cta_idx)
        token_mask = txl.local_scalar(txl.u64, init=txl.uint64(0))
        lowest_bit = txl.local_scalar(txl.u64)
        expert = txl.local_scalar(txl.i32)
        lane_expert = txl.local_scalar(txl.i32)  # this lane's entry: expert (-1: none)
        lane_entry = txl.local_scalar(txl.i32)  # this lane's entry: token word
        num_entries = txl.local_scalar(txl.i32)
        num_rows = txl.local_scalar(txl.i32)
        row_idx = txl.local_scalar(txl.i32)
        mask_lo = txl.local_scalar(txl.u32)
        mask_hi = txl.local_scalar(txl.u32)

        evict_policy = txl.local_scalar(txl.u64)
        txl.ptx.createpolicy.fractional.L2__evict_first.b64(evict_policy, txl.float32(1.0))

        def load_window(dst, start):
            token = cta_idx + (start + lane) * grid_size
            txl.assign(dst, txl.uint64(0))
            with txl.if_(token < T), txl.then_():
                txl.ptx.ld.global_.nc.u64(dst, mask.ptr_to([token]))

        def fetch_mask():
            """
            token_mask = the mask of window entry window_pos, in every lane.
            """
            window_lo = txl.Cast("uint32", window_masks & txl.uint64(0xFFFFFFFF))
            window_hi = txl.Cast("uint32", window_masks >> txl.uint64(32))
            txl.ptx.shfl_sync.idx.b32(mask_lo, window_lo, as_u32(window_pos), txl.uint32(31), txl.uint32(FULL_MASK))
            txl.ptx.shfl_sync.idx.b32(mask_hi, window_hi, as_u32(window_pos), txl.uint32(31), txl.uint32(FULL_MASK))
            txl.assign(token_mask, (txl.Cast("uint64", mask_hi) << txl.uint64(32)) | txl.Cast("uint64", mask_lo))

        def produce(slot):
            """
            Fill ring slot with the next unit: its entries, then one bulk copy per grouped row.
            """
            txl.assign(num_entries, txl.int32(0))
            txl.assign(num_rows, txl.int32(0))
            txl.assign(lane_expert, txl.int32(-1))
            for k in range(UNIT_ROWS):
                with txl.if_(token_idx < T), txl.then_():
                    txl.assign(lowest_bit, token_mask & (txl.uint64(0) - token_mask))
                    txl.assign(expert, txl.Select(lowest_bit != txl.uint64(0), as_i32(txl.popcount(lowest_bit - txl.uint64(1))), txl.int32(-1)))
                    txl.assign(token_mask, token_mask ^ lowest_bit)
                    entry_word = token_idx + txl.Select(token_mask == txl.uint64(0), txl.int32(LAST_BIT), txl.int32(0)) + txl.Select(expert < 0, txl.int32(NOROW_BIT), txl.int32(0))
                    with txl.if_(lane == k), txl.then_():
                        txl.assign(lane_expert, expert)
                        txl.assign(lane_entry, entry_word)
                    txl.assign(num_entries, num_entries + 1)
                    txl.assign(num_rows, num_rows + txl.Select(expert >= 0, txl.int32(1), txl.int32(0)))
                    with txl.if_(token_mask == txl.uint64(0)), txl.then_():  # token done: next token of this CTA
                        txl.assign(window_pos, window_pos + 1)
                        with txl.if_(window_pos == 32), txl.then_():
                            txl.assign(window_masks, next_window_masks)
                            txl.assign(window_start, window_start + 32)
                            txl.assign(window_pos, txl.int32(0))
                            load_window(next_window_masks, window_start + 32)
                        txl.assign(token_idx, cta_idx + (window_start + window_pos) * grid_size)
                        fetch_mask()
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.s32(s_unit_len.ptr_to([slot]), txl.Select(num_entries > 0, num_entries, txl.int32(-1)))
            with txl.if_(lane < num_entries), txl.then_():
                txl.ptx.st.shared.s32(s_unit_entries.ptr_to([slot * UNIT_ROWS + lane]), lane_entry)
            txl.ptx["bar.warp.sync"](txl.uint32(FULL_MASK))  # every lane's meta store before the (releasing) arrive
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(s_full.ptr_to([slot]), as_u32(num_rows * ROW_BYTES))
            with txl.if_(txl.And(lane < num_entries, lane_expert >= 0)), txl.then_():
                txl.ptx.ld.global_.nc.s32(row_idx, table.ptr_to([lane_expert * T + (lane_entry & txl.int32(NOROW_BIT - 1))]))
                if EVICT_FIRST["gs"]:
                    txl.ptx[BULK_G2S + ".L2::cache_hint"](s_stage.ptr_to([(slot * UNIT_ROWS + lane) * ROW_WORDS]), grad.ptr_to([txl.Cast("int64", row_idx) * ROW_WORDS]), txl.uint32(ROW_BYTES), s_full.ptr_to([slot]), evict_policy)
                else:
                    txl.ptx[BULK_G2S](s_stage.ptr_to([(slot * UNIT_ROWS + lane) * ROW_WORDS]), grad.ptr_to([txl.Cast("int64", row_idx) * ROW_WORDS]), txl.uint32(ROW_BYTES), s_full.ptr_to([slot]))

        def spin_wait(barrier, parity):
            is_ready = txl.local_scalar(txl.u32, init=txl.uint32(0))
            with txl.while_(is_ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.shared.b64(is_ready, barrier, parity)

        def producer_loop():
            """
            Fill every slot once, then (with a dedicated producer warp) refill each slot once its
            consumers release it, until the end marker.
            """
            for slot in range(STAGES):
                produce(slot)
            if USE_PRODUCER_WARP:
                unit_idx = txl.local_scalar(txl.i32, init=txl.int32(STAGES))
                is_done = txl.local_scalar(txl.i32, init=txl.Select(num_entries == 0, txl.int32(1), txl.int32(0)))
                slot = txl.local_scalar(txl.i32)
                with txl.while_(is_done == 0):
                    txl.assign(slot, unit_idx % STAGES)
                    spin_wait(s_empty.ptr_to([slot]), as_u32((unit_idx // STAGES - 1) & 1))
                    produce(slot)
                    txl.assign(is_done, txl.Select(num_entries == 0, txl.int32(1), txl.int32(0)))
                    txl.assign(unit_idx, unit_idx + 1)

        def add_unit_rows(slot, unit_len):
            """
            Add the unit's rows to acc; a token's last entry rounds, stores and resets acc.
            """
            for k in range(UNIT_ROWS):
                with txl.if_(k < unit_len), txl.then_():
                    txl.ptx.ld.shared.s32(entry_word, s_unit_entries.ptr_to([slot * UNIT_ROWS + k]))
                    with txl.if_((entry_word & txl.int32(NOROW_BIT)) == 0), txl.then_():
                        txl.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], s_stage.ptr_to([(slot * UNIT_ROWS + k) * ROW_WORDS + tidx * 4]))
                        for pair in range(4):
                            add_bf16x2(acc, words, pair, pair)
                    with txl.if_((entry_word & txl.int32(LAST_BIT)) != 0), txl.then_():
                        pack_bf16x2(words, acc)
                        token = entry_word & txl.int32(NOROW_BIT - 1)
                        txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", token) * ROW_WORDS + tidx * 4]), words[0], words[1], words[2], words[3])
                        for i in range(8):
                            txl.assign(acc[i], txl.float32(0))

        with txl.if_(warp == PRODUCER_WARP), txl.then_():
            load_window(window_masks, txl.int32(0))
            load_window(next_window_masks, txl.int32(32))
            fetch_mask()

        with txl.if_(warp == PRODUCER_WARP), txl.then_():
            producer_loop()

        acc = txl.alloc_local([8], txl.f32)
        words = txl.alloc_local([4], txl.u32)
        unit_idx = txl.local_scalar(txl.i32, init=txl.int32(0))
        is_running = txl.local_scalar(txl.i32, init=txl.int32(1))
        unit_len = txl.local_scalar(txl.i32)
        entry_word = txl.local_scalar(txl.i32)
        slot = txl.local_scalar(txl.i32)
        for i in range(8):
            txl.assign(acc[i], txl.float32(0))
        txl.assign(is_running, txl.Select(warp < NUM_WARPS, txl.int32(1), txl.int32(0)))
        with txl.while_(is_running > 0):
            txl.assign(slot, unit_idx % STAGES)
            if USE_SPIN_WAIT:  # try_wait without a suspend-time hint, in a spin loop
                is_ready = txl.local_scalar(txl.u32, init=txl.uint32(0))
                with txl.while_(is_ready == txl.uint32(0)):
                    txl.ptx.mbarrier.try_wait.parity.shared.b64(is_ready, s_full.ptr_to([slot]), as_u32((unit_idx // STAGES) & 1))
            else:
                txl.cuda.mbarrier_wait(s_full.ptr_to([slot]), as_u32((unit_idx // STAGES) & 1))
            txl.ptx.ld.shared.s32(unit_len, s_unit_len.ptr_to([slot]))
            with txl.if_(unit_len < 0):
                with txl.then_():
                    txl.assign(is_running, txl.int32(0))
                with txl.else_():
                    with txl.if_(vec_ok), txl.then_():
                        add_unit_rows(slot, unit_len)
            if USE_PRODUCER_WARP:
                with txl.if_(is_running > 0), txl.then_():  # each consumer thread releases the slot after its own reads
                    txl.ptx.mbarrier.arrive.shared.b64(s_empty.ptr_to([slot]))
            else:
                txl.cuda.cta_sync()  # every thread is done with the slot
                with txl.if_(txl.And(is_running > 0, warp == 0)), txl.then_():
                    produce(slot)
            txl.assign(unit_idx, unit_idx + 1)

    return entry_function(gather_sum_tma3.compile(TARGET))


def bf16_expert_gather_sum_sm100(grad, maps, num_groups, max_slots):
    """
    bf16_expert_gather_sum on SM100; max_slots is not needed here.
    """
    T, H = maps.mask.shape[0], grad.shape[1]
    out = torch.empty(T, H, dtype=torch.bfloat16, device=grad.device)
    if T:
        fn = (compile_gather_sum_multi_token if USE_MULTI_TOKEN_UNITS else compile_gather_sum_tma if USE_GATHER_SUM_TMA else compile_gather_sum)(num_groups, H)
        fn(grad.view(-1).view(torch.uint32), maps.mask.view(torch.uint64), maps.table, out.view(-1).view(torch.uint32), T)
    return out


ROWS_PER_CTA = 4


@functools.cache
def compile_gather_rows(N: int):
    """
    ROWS_PER_CTA rows per CTA, each thread loading all its 16-byte vectors before storing any.
    """
    NUM_VECS = N // 8
    NUM_WARPS = 8

    @txl.kernel()
    def gather_rows(x: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), idx: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), R: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=(R + ROWS_PER_CTA - 1) // ROWS_PER_CTA, block=NUM_WARPS * 32))
        cta_idx = txl.cta_id()
        tidx = txl.thread_id()
        VECS_PER_THREAD = (NUM_VECS * ROWS_PER_CTA + NUM_WARPS * 32 - 1) // (NUM_WARPS * 32)
        words = txl.alloc_local([4 * VECS_PER_THREAD], txl.u32)
        src_rows = txl.alloc_local([VECS_PER_THREAD], txl.i32)
        evict_policy = txl.local_scalar(txl.u64)
        txl.ptx.createpolicy.fractional.L2__evict_first.b64(evict_policy, txl.float32(1.0))
        for i in range(VECS_PER_THREAD):
            vec_idx = i * NUM_WARPS * 32 + tidx
            row_idx = cta_idx * ROWS_PER_CTA + vec_idx // NUM_VECS
            with txl.if_(txl.And(vec_idx < NUM_VECS * ROWS_PER_CTA, row_idx < R)), txl.then_():
                txl.ptx.ld.global_.nc.s32(src_rows[i], idx.ptr_to([row_idx]))
        for i in range(VECS_PER_THREAD):
            vec_idx = i * NUM_WARPS * 32 + tidx
            row_idx = cta_idx * ROWS_PER_CTA + vec_idx // NUM_VECS
            with txl.if_(txl.And(vec_idx < NUM_VECS * ROWS_PER_CTA, row_idx < R)), txl.then_():
                if EVICT_FIRST["rows"]:
                    txl.ptx["ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32"](words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3], x.ptr_to([txl.Cast("int64", src_rows[i]) * (N // 2) + (vec_idx % NUM_VECS) * 4]), evict_policy)
                else:
                    txl.ptx.ld.global_.nc.v4.b32(words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3], x.ptr_to([txl.Cast("int64", src_rows[i]) * (N // 2) + (vec_idx % NUM_VECS) * 4]))
        for i in range(VECS_PER_THREAD):
            vec_idx = i * NUM_WARPS * 32 + tidx
            row_idx = cta_idx * ROWS_PER_CTA + vec_idx // NUM_VECS
            with txl.if_(txl.And(vec_idx < NUM_VECS * ROWS_PER_CTA, row_idx < R)), txl.then_():
                txl.ptx.st.global_.v4.b32(out.ptr_to([txl.Cast("int64", row_idx) * (N // 2) + (vec_idx % NUM_VECS) * 4]), words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3])

    return entry_function(gather_rows.compile(TARGET))


def bf16_expert_gather_rows_sm100(x, idx):
    """
    bf16_expert_gather_rows on SM100.
    """
    R, N = idx.shape[0], x.shape[1]
    out = torch.empty(R, N, dtype=torch.bfloat16, device=x.device)
    if R:
        fn = compile_gather_rows(N)
        fn(x.view(-1).view(torch.uint32), idx, out.view(-1).view(torch.uint32), R)
    return out


# (warps = chunk workers per CTA, bytes per lane per row, cp.async rows in flight per thread)
SEGMENT_SUM_CFG = {False: (16, 16, 16), True: (16, 8, 16)}
USE_F32X2_ADD = False
# two-launch split for these has_idx values: (warps per CTA, bytes per lane per row, rows in flight)
SPLIT_SEGMENT_CFG = {True: (1, 16, 32), False: (1, 16, 32)}
SPLIT_SEGMENT_MIN_BYTES = {True: 0, False: 32 << 20}  # no-idx: below this one launch (no second launch to hide)
USE_PDL = True  # combine launched with programmatic dependent launch: its launch and cu prologue overlap the chunk tail


@functools.cache
def compile_segment_sum(N: int, has_idx: bool):
    """
    One launch: CTA (segment, column block) gives each warp one 256-row chunk per round and adds the
    chunk sums in order through shared memory.
    """
    NUM_WARPS, LANE_BYTES, RING_DEPTH = SEGMENT_SUM_CFG[has_idx]
    LANE_ELEMS = LANE_BYTES // 2  # bf16 per lane
    LANE_WORDS = LANE_BYTES // 4  # u32 words per lane
    COLS = 32 * LANE_ELEMS
    NUM_COL_BLOCKS = (N + COLS - 1) // COLS
    THREADS = NUM_WARPS * 32
    assert 256 % RING_DEPTH == 0

    @txl.kernel()
    def segment_sum(x: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), idx: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), G: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NUM_COL_BLOCKS, block=NUM_WARPS * 32))
        cta_idx = txl.cta_id()
        group_idx, col_block = cta_idx // NUM_COL_BLOCKS, cta_idx % NUM_COL_BLOCKS
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        smem = txl.smem_pool()
        s_ring = smem.alloc([RING_DEPTH * THREADS * LANE_WORDS], txl.u32)  # slot s of thread t at (s * THREADS + t) * LANE_BYTES
        s_partial = smem.alloc([NUM_WARPS * COLS], txl.f32)
        s_rows = smem.alloc([NUM_WARPS * 256], txl.i32, align=16)
        seg_start = txl.local_scalar(txl.i32)
        seg_end = txl.local_scalar(txl.i32)
        txl.ptx.ld.global_.nc.s32(seg_start, cu.ptr_to([group_idx]))
        txl.ptx.ld.global_.nc.s32(seg_end, cu.ptr_to([group_idx + 1]))
        num_chunks = (seg_end - seg_start + 255) // 256
        col = col_block * COLS + lane * LANE_ELEMS
        col_ok = col < N
        safe_col = txl.Select(col_ok, col, 0)
        acc = txl.alloc_local([LANE_ELEMS], txl.f32)
        acc_pairs = txl.alloc_local([LANE_WORDS], txl.u64)
        acc_total = txl.local_scalar(txl.f32, init=txl.float32(0))
        words = txl.alloc_local([LANE_WORDS], txl.u32)
        src_idx = txl.local_scalar(txl.i32)
        copy_op = "cp.async.cg.shared.global" if LANE_BYTES == 16 else "cp.async.ca.shared.global"

        chunk_rows = txl.alloc_local([RING_DEPTH], txl.i32)  # grouped rows of the next RING_DEPTH chunk rows (has_idx)

        def load_chunk_rows(row_start):
            # rows row_start .. row_start + RING_DEPTH - 1 of this warp's chunk, read ahead of their issues
            for quad in range(RING_DEPTH // 4):
                txl.ptx.ld.shared.v4.b32(chunk_rows[4 * quad], chunk_rows[4 * quad + 1], chunk_rows[4 * quad + 2], chunk_rows[4 * quad + 3], s_rows.ptr_to([warp * 256 + row_start + 4 * quad]))

        def issue_row(i, slot, chunk_start, chunk_end):
            row_idx = chunk_start + i
            row_ok = txl.And(row_idx < chunk_end, col_ok)
            if has_idx:
                txl.assign(src_idx, chunk_rows[slot])
            else:
                txl.assign(src_idx, row_idx)
            src_row = txl.Select(row_ok, src_idx, 0)
            txl.ptx[copy_op](s_ring.ptr_to([(slot * THREADS + tidx) * LANE_WORDS]), x.ptr_to([txl.Cast("int64", src_row) * (N // 2) + safe_col // 2]), LANE_BYTES, txl.ptx.pred(txl.Not(row_ok)))

        def add_ring_slot(slot):
            if LANE_WORDS == 4:
                txl.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], s_ring.ptr_to([(slot * THREADS + tidx) * LANE_WORDS]))
            else:
                txl.ptx.ld.shared.v2.b32(words[0], words[1], s_ring.ptr_to([(slot * THREADS + tidx) * LANE_WORDS]))
            for pair in range(LANE_WORDS):
                lo = txl.reinterpret(txl.shift_left(words[pair], txl.uint32(16)), ty="float32")
                hi = txl.reinterpret(words[pair] & txl.uint32(0xFFFF0000), ty="float32")
                if USE_F32X2_ADD:
                    txl.ptx["add.rn.f32x2"](acc_pairs[pair], acc_pairs[pair], txl.cuda.make_float2(lo, hi))
                else:
                    txl.assign(acc[2 * pair], acc[2 * pair] + lo)
                    txl.assign(acc[2 * pair + 1], acc[2 * pair + 1] + hi)

        def sum_chunk(chunk_start, chunk_end):
            """
            Add the chunk's rows into acc in order, through the cp.async ring.
            """
            if has_idx:
                for k in range(8):
                    row_idx = chunk_start + k * 32 + lane
                    grouped_row = txl.local_scalar(txl.i32, init=txl.int32(0))
                    with txl.if_(row_idx < chunk_end), txl.then_():
                        txl.ptx.ld.global_.nc.s32(grouped_row, idx.ptr_to([row_idx]))
                    txl.ptx.st.shared.s32(s_rows.ptr_to([warp * 256 + k * 32 + lane]), grouped_row)
                txl.ptx["bar.warp.sync"](txl.uint32(FULL_MASK))
            if has_idx:
                load_chunk_rows(0)
            for j in range(RING_DEPTH):
                issue_row(j, j, chunk_start, chunk_end)
                txl.ptx["cp.async.commit_group"]()
            # rows block * RING_DEPTH + j live in ring slot j: the slot index is static
            with txl.serial(0, 256 // RING_DEPTH - 1) as block:
                if has_idx:
                    load_chunk_rows((block + 1) * RING_DEPTH)
                for j in range(RING_DEPTH):
                    txl.ptx["cp.async.wait_group"](RING_DEPTH - 1)
                    add_ring_slot(j)
                    issue_row((block + 1) * RING_DEPTH + j, j, chunk_start, chunk_end)
                    txl.ptx["cp.async.commit_group"]()
            txl.ptx["cp.async.wait_group"](0)
            for j in range(RING_DEPTH):
                add_ring_slot(j)

        def add_round_sums(round_idx):
            """
            Thread tidx < COLS owns column col_block * COLS + tidx and adds this round's chunk
            sums in order.
            """
            if USE_F32X2_ADD:
                for pair in range(LANE_WORDS):
                    txl.assign(acc[2 * pair], txl.cuda.float2_x(acc_pairs[pair]))
                    txl.assign(acc[2 * pair + 1], txl.cuda.float2_y(acc_pairs[pair]))
            for i in range(LANE_ELEMS):
                txl.ptx.st.shared.f32(s_partial.ptr_to([warp * COLS + lane * LANE_ELEMS + i]), acc[i])
            txl.cuda.cta_sync()
            with txl.if_(tidx < COLS), txl.then_():
                for warp_idx in range(NUM_WARPS):
                    chunk_sum = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.f32(chunk_sum, s_partial.ptr_to([warp_idx * COLS + tidx]))
                    with txl.if_(round_idx * NUM_WARPS + warp_idx < num_chunks), txl.then_():
                        txl.assign(acc_total, acc_total + chunk_sum)
            txl.cuda.cta_sync()

        with txl.serial(0, (num_chunks + NUM_WARPS - 1) // NUM_WARPS) as round_idx:
            chunk_idx = round_idx * NUM_WARPS + warp
            chunk_start = seg_start + chunk_idx * 256
            chunk_end = txl.min(chunk_start + 256, seg_end)
            for i in range(LANE_ELEMS):
                txl.assign(acc[i], txl.float32(0))
            for pair in range(LANE_WORDS):
                txl.assign(acc_pairs[pair], txl.uint64(0))
            with txl.if_(chunk_idx < num_chunks), txl.then_():
                sum_chunk(chunk_start, chunk_end)
            add_round_sums(round_idx)
        out_col = col_block * COLS + tidx
        with txl.if_(txl.And(tidx < COLS, out_col < N)), txl.then_():
            txl.ptx.st.global_.f32(out.ptr_to([txl.Cast("int64", group_idx) * N + out_col]), acc_total)

    return entry_function(segment_sum.compile(TARGET))


def find_chunk_rows(cu, G, chunk_idx, chunk_start, chunk_end, lane):
    """
    Warp-cooperative: rows [chunk_start, chunk_end) of global chunk chunk_idx (chunks numbered
    segment-major, 256 rows each).
    """
    chunks_before = txl.local_scalar(txl.i32, init=txl.int32(0))
    seg_start = txl.local_scalar(txl.i32)
    seg_end = txl.local_scalar(txl.i32)
    seg_chunks = txl.local_scalar(txl.i32)
    chunks_incl = txl.local_scalar(txl.i32)
    peer = txl.local_scalar(txl.i32)
    owner_lanes = txl.local_scalar(txl.u32)
    with txl.serial(0, (G + 31) // 32) as round_idx:
        group_idx = round_idx * 32 + lane
        txl.assign(seg_chunks, txl.int32(0))
        txl.assign(seg_start, txl.int32(0))
        with txl.if_(group_idx < G), txl.then_():
            txl.ptx.ld.global_.nc.s32(seg_start, cu.ptr_to([group_idx]))
            txl.ptx.ld.global_.nc.s32(seg_end, cu.ptr_to([group_idx + 1]))
            txl.assign(seg_chunks, (seg_end - seg_start + 255) // 256)
        txl.assign(chunks_incl, seg_chunks)
        warp_inclusive_scan(chunks_incl, peer, lane)
        first_chunk = chunks_before + chunks_incl - seg_chunks  # this lane's segment's first chunk
        is_owner = txl.And(chunk_idx >= first_chunk, chunk_idx < chunks_before + chunks_incl)
        txl.ptx.vote_sync.ballot.b32(owner_lanes, txl.ptx.pred(is_owner), txl.uint32(FULL_MASK))
        with txl.if_(owner_lanes != txl.uint32(0)), txl.then_():
            owner_lane = as_i32(txl.popcount((owner_lanes & (txl.uint32(0) - owner_lanes)) - txl.uint32(1)))
            owner_first = txl.local_scalar(txl.i32)
            owner_start = txl.local_scalar(txl.i32)
            owner_end = txl.local_scalar(txl.i32)
            txl.ptx.shfl_sync.idx.b32(owner_first, first_chunk, as_u32(owner_lane), txl.uint32(31), txl.uint32(FULL_MASK))
            txl.ptx.shfl_sync.idx.b32(owner_start, seg_start, as_u32(owner_lane), txl.uint32(31), txl.uint32(FULL_MASK))
            txl.ptx.shfl_sync.idx.b32(owner_end, seg_end, as_u32(owner_lane), txl.uint32(31), txl.uint32(FULL_MASK))
            txl.assign(chunk_start, owner_start + (chunk_idx - owner_first) * 256)
            txl.assign(chunk_end, txl.min(owner_start + (chunk_idx - owner_first) * 256 + 256, owner_end))
        round_chunks = txl.local_scalar(txl.i32)
        txl.ptx.shfl_sync.idx.b32(round_chunks, chunks_incl, txl.uint32(31), txl.uint32(31), txl.uint32(FULL_MASK))
        txl.assign(chunks_before, chunks_before + round_chunks)


@functools.cache
def compile_segment_chunks(N: int, has_idx: bool):
    """
    Kernel 1 of 2: CTA (chunk, column group) writes the chunk's fp32 row sum (rows added in order).
    """
    NUM_WARPS, LANE_BYTES, RING_DEPTH = SPLIT_SEGMENT_CFG[has_idx]
    LANE_ELEMS = LANE_BYTES // 2
    LANE_WORDS = LANE_BYTES // 4
    COLS = 32 * LANE_ELEMS  # per warp
    NUM_COL_BLOCKS = (N + NUM_WARPS * COLS - 1) // (NUM_WARPS * COLS)
    THREADS = NUM_WARPS * 32
    assert 256 % RING_DEPTH == 0

    @txl.kernel()
    def segment_chunks(x: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), idx: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), part: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), G: txl.int32, NC: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NC * NUM_COL_BLOCKS, block=NUM_WARPS * 32))
        if USE_PDL:
            txl.ptx.griddepcontrol.launch_dependents()  # the combine may launch now; it waits for our writes
        cta_idx = txl.cta_id()
        chunk_idx, col_block = cta_idx // NUM_COL_BLOCKS, cta_idx % NUM_COL_BLOCKS
        tidx = txl.thread_id()
        lane, warp = tidx & 31, tidx >> 5
        smem = txl.smem_pool()
        s_ring = smem.alloc([RING_DEPTH * THREADS * LANE_WORDS], txl.u32)
        s_rows = smem.alloc([256], txl.i32, align=16)
        chunk_start = txl.local_scalar(txl.i32, init=txl.int32(0))
        chunk_end = txl.local_scalar(txl.i32, init=txl.int32(0))
        find_chunk_rows(cu, G, chunk_idx, chunk_start, chunk_end, lane)
        if has_idx:
            with txl.serial(0, (256 + THREADS - 1) // THREADS) as step:
                row_offset = step * THREADS + tidx
                grouped_row = txl.local_scalar(txl.i32, init=txl.int32(0))
                with txl.if_(txl.And(row_offset < 256, chunk_start + row_offset < chunk_end)), txl.then_():
                    txl.ptx.ld.global_.nc.s32(grouped_row, idx.ptr_to([chunk_start + row_offset]))
                with txl.if_(row_offset < 256), txl.then_():
                    txl.ptx.st.shared.s32(s_rows.ptr_to([row_offset]), grouped_row)
            txl.cuda.cta_sync()
        col = (col_block * NUM_WARPS + warp) * COLS + lane * LANE_ELEMS
        col_ok = col < N
        safe_col = txl.Select(col_ok, col, 0)
        acc = txl.alloc_local([LANE_ELEMS], txl.f32)
        words = txl.alloc_local([LANE_WORDS], txl.u32)
        src_idx = txl.local_scalar(txl.i32)
        copy_op = "cp.async.cg.shared.global" if LANE_BYTES == 16 else "cp.async.ca.shared.global"
        evict_policy = txl.local_scalar(txl.u64)
        txl.ptx.createpolicy.fractional.L2__evict_first.b64(evict_policy, txl.float32(1.0))
        chunk_rows = txl.alloc_local([RING_DEPTH], txl.i32)

        def load_chunk_rows(row_start):
            for quad in range(RING_DEPTH // 4):
                txl.ptx.ld.shared.v4.b32(chunk_rows[4 * quad], chunk_rows[4 * quad + 1], chunk_rows[4 * quad + 2], chunk_rows[4 * quad + 3], s_rows.ptr_to([row_start + 4 * quad]))

        def issue_row(i, slot):
            row_idx = chunk_start + i
            row_ok = txl.And(row_idx < chunk_end, col_ok)
            if has_idx:
                txl.assign(src_idx, chunk_rows[slot])
            else:
                txl.assign(src_idx, row_idx)
            src_row = txl.Select(row_ok, src_idx, 0)
            if EVICT_FIRST["seg"]:
                txl.ptx[copy_op + ".L2::cache_hint"](s_ring.ptr_to([(slot * THREADS + tidx) * LANE_WORDS]), x.ptr_to([txl.Cast("int64", src_row) * (N // 2) + safe_col // 2]), LANE_BYTES, txl.ptx.pred(txl.Not(row_ok)), evict_policy)
            else:
                txl.ptx[copy_op](s_ring.ptr_to([(slot * THREADS + tidx) * LANE_WORDS]), x.ptr_to([txl.Cast("int64", src_row) * (N // 2) + safe_col // 2]), LANE_BYTES, txl.ptx.pred(txl.Not(row_ok)))

        def add_ring_slot(slot):
            if LANE_WORDS == 4:
                txl.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], s_ring.ptr_to([(slot * THREADS + tidx) * LANE_WORDS]))
            elif LANE_WORDS == 2:
                txl.ptx.ld.shared.v2.b32(words[0], words[1], s_ring.ptr_to([(slot * THREADS + tidx) * LANE_WORDS]))
            else:
                txl.ptx.ld.shared.b32(words[0], s_ring.ptr_to([slot * THREADS + tidx]))
            for pair in range(LANE_WORDS):
                add_bf16x2(acc, words, pair, pair)

        def sum_chunk():
            """
            Add the chunk's real rows into acc in order through the cp.async ring; store the sums.
            """
            if has_idx:
                load_chunk_rows(0)
            for j in range(RING_DEPTH):
                issue_row(j, j)
                txl.ptx["cp.async.commit_group"]()
            with txl.serial(0, (chunk_end - chunk_start + RING_DEPTH - 1) // RING_DEPTH - 1) as block:
                if has_idx:
                    load_chunk_rows((block + 1) * RING_DEPTH)
                for j in range(RING_DEPTH):
                    txl.ptx["cp.async.wait_group"](RING_DEPTH - 1)
                    add_ring_slot(j)
                    issue_row((block + 1) * RING_DEPTH + j, j)
                    txl.ptx["cp.async.commit_group"]()
            txl.ptx["cp.async.wait_group"](0)
            for j in range(RING_DEPTH):
                add_ring_slot(j)
            with txl.if_(col_ok), txl.then_():
                for i in range(LANE_ELEMS):
                    txl.ptx.st.global_.f32(part.ptr_to([txl.Cast("int64", chunk_idx) * N + col + i]), acc[i])

        for i in range(LANE_ELEMS):
            txl.assign(acc[i], txl.float32(0))
        with txl.if_((col_block * NUM_WARPS + warp) * COLS < N), txl.then_():
            sum_chunk()

    return entry_function(segment_chunks.compile(TARGET))


@functools.cache
def compile_segment_combine(N: int, use_pdl: bool):
    """
    Kernel 2 of 2: out[g, col] = chunk sums of segment g added in order (0 for an empty segment).
    """
    NUM_COL_BLOCKS = (N + 255) // 256

    @txl.kernel()
    def segment_combine(cu: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), part: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), G: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NUM_COL_BLOCKS, block=256, programmatic_stream_serialization=True if use_pdl else None))
        cta_idx = txl.cta_id()
        group_idx = cta_idx // NUM_COL_BLOCKS
        col = (cta_idx % NUM_COL_BLOCKS) * 256 + txl.thread_id()
        chunk_base = txl.local_scalar(txl.i32, init=txl.int32(0))
        seg_start = txl.local_scalar(txl.i32)
        seg_end = txl.local_scalar(txl.i32)
        with txl.serial(0, group_idx) as prev_group:
            txl.ptx.ld.global_.nc.s32(seg_start, cu.ptr_to([prev_group]))
            txl.ptx.ld.global_.nc.s32(seg_end, cu.ptr_to([prev_group + 1]))
            txl.assign(chunk_base, chunk_base + (seg_end - seg_start + 255) // 256)
        txl.ptx.ld.global_.nc.s32(seg_start, cu.ptr_to([group_idx]))
        txl.ptx.ld.global_.nc.s32(seg_end, cu.ptr_to([group_idx + 1]))
        if use_pdl:
            txl.ptx.griddepcontrol.wait()  # the chunk kernel's partial sums are complete and visible
        acc_total = txl.local_scalar(txl.f32, init=txl.float32(0))
        chunk_sums = txl.alloc_local([8], txl.f32)
        num_chunks = (seg_end - seg_start + 255) // 256
        with txl.if_(col < N), txl.then_():
            with txl.serial(0, (num_chunks + 7) // 8) as block:
                for k in range(8):
                    with txl.if_(block * 8 + k < num_chunks), txl.then_():
                        txl.ptx.ld.global_.f32(chunk_sums[k], part.ptr_to([txl.Cast("int64", chunk_base + block * 8 + k) * N + col]))
                for k in range(8):
                    with txl.if_(block * 8 + k < num_chunks), txl.then_():
                        txl.assign(acc_total, acc_total + chunk_sums[k])
            txl.ptx.st.global_.f32(out.ptr_to([txl.Cast("int64", group_idx) * N + col]), acc_total)

    return entry_function(segment_combine.compile(TARGET))


def bf16_expert_segment_sum_sm100(x, cu_host, cu, idx=None):
    """
    bf16_expert_segment_sum on SM100: two launches (chunk sums, then their combine) with an index or
    past SPLIT_SEGMENT_MIN_BYTES, else one.
    """
    G, N = len(cu_host) - 1, x.shape[1]
    out = torch.empty(G, N, dtype=torch.float32, device=x.device)
    has_idx = idx is not None
    if G and has_idx in SPLIT_SEGMENT_CFG and cu_host[-1] * N * 2 >= SPLIT_SEGMENT_MIN_BYTES[has_idx]:
        num_chunks = sum((cu_host[g + 1] - cu_host[g] + 255) // 256 for g in range(G))
        part = torch.empty(max(num_chunks, 1) * N, dtype=torch.float32, device=x.device)
        if num_chunks:
            compile_segment_chunks(N, has_idx)(x.view(-1).view(torch.uint32), idx if has_idx else cu, cu, part, G, num_chunks)
        compile_segment_combine(N, USE_PDL and num_chunks > 0)(cu, part, out.view(-1), G)
        return out
    if G:
        fn = compile_segment_sum(N, idx is not None)
        fn(x.view(-1).view(torch.uint32), idx if idx is not None else cu, cu, out.view(-1), G)
    return out
