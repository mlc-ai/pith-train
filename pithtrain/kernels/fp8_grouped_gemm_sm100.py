"""
pith-train SM100 FP8 expert GEMMs as fused TIRx-lite tcgen05 block-scaled kernels.

m-grouped (fwd / dgrad): out[rows of g] = a_g @ b[g].T, bf16.
k-grouped (wgrad):       d[g] += a_g^T @ b_g, fp32, accumulated in place by TMA reduce-add.

Both kernels are persistent, warp specialized (TMA / MMA / scale-factor / epilogue warps) and
read pith-train's fp32 E8M0 scales directly: a scale warp converts them to packed UE8M0 bytes in
shared memory, the MMA warp copies them to tensor memory with tcgen05.cp and issues
tcgen05.mma.kind::mxf8f6f4.block_scale (one scale per 128-wide K block, selected by sf_id).

The m-grouped GEMM picks a route by shape: 256 x 320 pair tiles as two N=160 sub-MMAs over
rotating TMEM slots (n a multiple of 320 but not of 256, gpt-oss), a 1-CTA 128 x 256 kernel for
small groups, swap-AB pair tiles (tokens on the MMA N side) for other small-group shapes, 256-wide
tiles as two N=128 sub-MMAs over rotating slots for long qwen rows, or 2-CTA BN=192 / BN=256
tiles. The k-grouped GEMM claims 256 x 256 pair tiles dynamically from a per-(device, stream)
counter and is launched with programmatic dependent launch (its prologue overlaps the previous
kernel's tail; every global access follows griddepcontrol.wait).
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

from tvm_tirx.backend.cuda.cpp.descriptors import encode_instr_descriptor_block_scaled_uint32, encode_smem_descriptor_base_uint64

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["fp8_grouped_gemm_sm100", "k_grouped_fp8_gemm_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


NUM_SMS = 148
BM = 128
BK = 128
UMMA_K = 32
EPI_BARRIER = 8
EVICT_NORMAL = 0x1000000000000000
EVICT_LAST = 0x14F0000000000000
FP8 = "float8_e4m3fn"
L2_PROMOTION = 3  # TMA L2 sector promotion for m-grouped operands: 256 B (a miss also fetches the next K block)


def uint64_const(value):
    """
    The Python int value as a uint64 expression, assembled from its two 32-bit halves.
    """
    return txl.bitwise_or(txl.shift_left(txl.uint64((value >> 32) & 0xFFFFFFFF), txl.uint64(32)), txl.uint64(value & 0xFFFFFFFF))


def to_u32(x):
    return txl.cast(x, "uint32")


def unsigned_div(x, d):
    return txl.cast(to_u32(x) // to_u32(d), "int32")


def unsigned_mod(x, d):
    return txl.cast(to_u32(x) % to_u32(d), "int32")


def smem_descriptor(base_const, addr_u32):
    """
    Matrix/SF descriptor: constant fields | (addr >> 4) in bits [0, 14).
    """
    return txl.bitwise_or(uint64_const(base_const), txl.cast(txl.bitwise_and(txl.shift_right(addr_u32, txl.uint32(4)), txl.uint32(0x3FFF)), "uint64"))


def with_sf_id(desc, sf_id):
    """
    The instruction descriptor desc with its A and B scale-factor id fields set to sf_id.
    """
    out = txl.bitwise_and(desc, txl.uint32(0x9FFFFFCF))
    out = txl.bitwise_or(out, txl.shift_left(to_u32(sf_id), txl.uint32(29)))
    return txl.bitwise_or(out, txl.shift_left(to_u32(sf_id), txl.uint32(4)))


def pack_scales4(scale_words, start):
    """
    Pack the four fp32 E8M0 scales scale_words[start:start + 4] into one word of UE8M0 bytes.
    """
    lo, hi, out = txl.local_scalar("uint16"), txl.local_scalar("uint16"), txl.local_scalar("uint32")
    txl.ptx.cvt.rz.satfinite.ue8m0x2.f32(lo, txl.reinterpret(scale_words[start + 1], ty="float32"), txl.reinterpret(scale_words[start], ty="float32"))
    txl.ptx.cvt.rz.satfinite.ue8m0x2.f32(hi, txl.reinterpret(scale_words[start + 3], ty="float32"), txl.reinterpret(scale_words[start + 2], ty="float32"))
    txl.ptx.mov.b32(out, lo, hi)
    return out


SF_DESC_BASE = encode_smem_descriptor_base_uint64(0, 8, 0)
AB_DESC_BASE = encode_smem_descriptor_base_uint64(0, 64, 3)  # SW128, 8 rows x 128 B per group
MMA_CHAIN = "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
UTCCP_CHAIN = "tcgen05.cp.cta_group::1.32x128b.warpx4"
LOAD2D = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
LOAD3D = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
STORE2D = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
PREFETCH3D = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"
REDUCE3D = "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"


def offset_descriptor(desc, byte_off):
    """
    Descriptor of the operand at +byte_off (16-B units in the start-address field; no carry out).
    """
    return desc if byte_off == 0 else desc + txl.uint64(byte_off >> 4)


def wait_stage(barrier, stage, phase):
    txl.cuda.mbarrier_wait(barrier.ptr_to([stage]), phase)


def range_start(name):
    """
    Open the named iket profiling range; returns the token range_end closes it with.
    """
    range_token = txl.alloc_local([1], "uint32")
    txl.assign(range_token[0], txl.cuda.iket.range_start(name))
    return range_token


def range_end(range_token):
    txl.cuda.iket.range_end(range_token[0])


def timed_wait(name, barrier, stage, phase):
    """
    Wait for the given stage of barrier inside the named profiling range.
    """
    wait_range = range_start(name)
    wait_stage(barrier, stage, phase)
    range_end(wait_range)


# m-grouped: out (M, N) bf16, A (M, K) K-major, B (G, N, K) K-major.


def build_mgrouped(STAGES, BN, STORE_BUFFERS=1, USE_SCALE_TAIL=False, USE_VECTOR_SCALES=False):
    """
    2-CTA (cta_group::2) m-grouped kernel: a CTA pair computes 256 rows x BN cols.

    Both 128-row halves of a pair belong to one group (pairs are formed per group); the odd last
    block of a group is computed by the pair's second CTA on rows it then does not store.
    """
    A_BYTES, B_HALF = BM * BK, (BN // 2) * BK
    ACC_STAGES = 2 if BN <= 192 else 1  # two accumulators + 12 scale columns fit in 512 TMEM columns
    # USE_SCALE_TAIL (BN=256): two 256-column accumulators anyway; a tile's scale columns live in cols
    # 240-251 of the OTHER accumulator, freed when the previous tile's epilogue has read that tail chunk
    # (drained first).
    USE_SCALE_TAIL = USE_SCALE_TAIL and BN == 256
    if USE_SCALE_TAIL:
        ACC_STAGES = 2
    N_SF_CHUNKS = (BN + 127) // 128  # 128-row SFB chunks per tile
    SFA_COL = ACC_STAGES * BN
    EPI_WARPGROUPS = 2 if BN >= 192 else 1
    EPI_WARPS = 4 * EPI_WARPGROUPS
    COLS_PER_WG = BN // EPI_WARPGROUPS
    assert COLS_PER_WG % 32 == 0 and (COLS_PER_WG // 32) % STORE_BUFFERS == 0
    INSTR_DESC = encode_instr_descriptor_block_scaled_uint32(M=2 * BM, N=BN, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2)
    # Final-wave split (BN=256): the last partial wave's tiles run as two 128-column halves when
    # those halves fit in one wave, halving the makespan tail.
    USE_FINAL_WAVE_SPLIT = BN == 256
    INSTR_DESC_HALF = encode_instr_descriptor_block_scaled_uint32(M=2 * BM, N=BN // 2, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2)
    MMA_PAIR = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP_PAIR = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LOAD2D_PAIR = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
    LOAD3D_PAIR = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def encode_tensor_maps(params):
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        b_half_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = params["M"], params["N"], params["K"], params["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, params["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(b_map, FP8, 3, params["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 2, 1, 1, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(b_half_map, FP8, 3, params["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 4, 1, 1, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(d_map, "bfloat16", 2, params["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0)
        return a_map, b_map, b_half_map, d_map

    @txl.kernel()
    def mgrouped(a: txl.Tensor((txl.int64(), txl.int64()), FP8, data_alignment=1), sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32, data_alignment=1), b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8, data_alignment=1), sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32, data_alignment=1), d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, K: txl.int32, G: txl.int32):
        tensor_maps = encode_tensor_maps(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=(4 + EPI_WARPS) * 32, cluster=[2]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, b_half_map, d_map = tensor_maps
        rank = txl.cuda.cluster_cta_id("x")
        cluster_idx = txl.cta_id() // 2
        warp = txl.warp_id()
        lane = txl.lane_id()
        tidx = txl.thread_id()

        roles = txl.specialize()
        tma_role = roles.role("tma", warps=[0])
        mma_role = roles.role("mma", warps=[1])
        sf_role = roles.role("sf", warps=[2])
        idle_role = roles.role("idle", warps=[3])
        epi_role = roles.role("epi", warps=list(range(4, 4 + EPI_WARPS)))
        # Issue the first 32 group ends' loads now; their latency overlaps the setup below.
        chunk0_group_end, chunk0_group_start = txl.local_scalar("int32"), txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(chunk0_group_end, offs.ptr_to([txl.min(lane, G - 1)]))
        txl.ptx.ld.global_.s32(chunk0_group_start, offs.ptr_to([txl.max(txl.min(lane, G - 1) - 1, 0)]))
        setup_range = range_start("setup")

        smem = txl.smem_pool()
        s_a = smem.alloc((STAGES * A_BYTES,), "uint8", align=1024)
        s_b = smem.alloc((STAGES * B_HALF,), "uint8", align=1024)
        s_store = smem.alloc((EPI_WARPGROUPS * STORE_BUFFERS * BM * 16,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * N_SF_CHUNKS * 128,), "uint32", align=16)
        load_full = txl.TMABar(smem, STAGES)
        load_empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        acc_full = txl.TCGen05Bar(smem, ACC_STAGES)
        acc_empty = txl.MBarrier(smem, ACC_STAGES)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        load_full.init(1)
        load_empty.init(1)
        sf_full.init(64)
        acc_full.init(1)
        acc_empty.init(2 * 32 * EPI_WARPS)
        tail_free = txl.MBarrier(smem, 2)
        tail_free.init(2 * 32 * 4)  # warpgroup 1 (columns 128-255) of both CTAs
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with idle_role:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](txl.address_of(tmem_slot[0]), txl.uint32(512))
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        leader_load_full = load_full.remote_view(0)
        leader_sf_full = sf_full.remote_view(0)
        range_end(setup_range)

        num_n_blocks, num_k_blocks = (N + (BN - 1)) // BN, (K + (BK - 1)) // BK
        # Lanes take 32 groups at a time: pairs per group (ceil(rows / 256)), their running sum
        # (shfl.up scan + carry) and, by warp-wide reductions, the total and the group holding this
        # cluster's first pair, so for_each_tile() starts there instead of walking from group 0.
        walk_range = range_start("walk")
        carry = txl.local_scalar("uint32", init=txl.uint32(0))
        lane_group_end = txl.local_scalar("int32")
        lane_group_start = txl.local_scalar("int32")
        first_pair = txl.local_scalar("int32", init=unsigned_div(cluster_idx, num_n_blocks))
        walk_start = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(6)]  # has_first_pair, group_idx, group_start, group_end, pair_start, pair_end
        peer_value = txl.local_scalar("uint32")

        def warp_sum(value):
            # one redux.sync per value: the six walk reductions are independent (no 5-deep shuffle chains)
            txl.ptx.redux_sync.add.u32(value, value, txl.uint32(0xFFFFFFFF))

        with txl.serial((G + 31) // 32) as chunk_idx:
            lane_group_idx = txl.local_scalar("int32", init=chunk_idx * 32 + lane)
            with txl.if_(chunk_idx == 0):
                with txl.then_():
                    txl.assign(lane_group_end, chunk0_group_end)
                    txl.assign(lane_group_start, chunk0_group_start)
                with txl.else_():
                    txl.ptx.ld.global_.s32(lane_group_end, offs.ptr_to([txl.min(lane_group_idx, G - 1)]))
                    txl.ptx.ld.global_.s32(lane_group_start, offs.ptr_to([txl.max(txl.min(lane_group_idx, G - 1) - 1, 0)]))
            with txl.if_(lane_group_idx == 0), txl.then_():
                txl.assign(lane_group_start, 0)
            lane_pairs = txl.local_scalar("uint32", init=txl.if_then_else(lane_group_idx < G, to_u32(unsigned_div(lane_group_end - lane_group_start + 255, 256)), txl.uint32(0)))
            pair_scan = txl.local_scalar("uint32", init=lane_pairs)
            for scan_offset in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer_value, pair_scan, txl.uint32(scan_offset), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.if_(lane >= scan_offset), txl.then_():
                    txl.assign(pair_scan, pair_scan + peer_value)
            lane_pair_end = txl.local_scalar("int32", init=txl.cast(carry + pair_scan, "int32"))
            lane_pair_start = txl.local_scalar("int32", init=lane_pair_end - txl.cast(lane_pairs, "int32"))
            has_first_pair = txl.local_scalar("uint32", init=txl.if_then_else(txl.And(lane_pair_start <= first_pair, first_pair < lane_pair_end), txl.uint32(1), txl.uint32(0)))
            walk_terms = [has_first_pair, has_first_pair * to_u32(lane_group_idx), has_first_pair * to_u32(lane_group_start), has_first_pair * to_u32(lane_group_end), has_first_pair * to_u32(lane_pair_start), has_first_pair * to_u32(lane_pair_end)]
            for j in range(6):
                value = txl.local_scalar("uint32", init=walk_terms[j])
                warp_sum(value)
                txl.assign(walk_start[j], walk_start[j] + value)
            chunk_pairs = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.idx.b32(chunk_pairs, pair_scan, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(carry, carry + chunk_pairs)
        total = txl.local_scalar("int32", init=txl.cast(carry, "int32"))
        has_start = walk_start[0] != txl.uint32(0)
        range_end(walk_range)
        # the group walk (registers + global loads only) overlaps the cluster barrier latency
        txl.ptx.barrier.cluster.wait.acquire.aligned()

        def for_each_tile(process_tile):
            """
            Pair walk: ticket -> (pair_idx, n_block); pairs are counted per group (ceil(rows/256)).
            """
            ticket = txl.local_scalar("int32", init=cluster_idx)
            group_idx = txl.local_scalar("int32", init=txl.if_then_else(has_start, txl.cast(walk_start[1], "int32"), -1))
            group_start = txl.local_scalar("int32", init=txl.cast(walk_start[2], "int32"))  # first row of group group_idx
            group_end = txl.local_scalar("int32", init=txl.cast(walk_start[3], "int32"))  # one past its last row
            pair_start = txl.local_scalar("int32", init=txl.cast(walk_start[4], "int32"))  # first pair of group group_idx
            pair_end = txl.local_scalar("int32", init=txl.cast(walk_start[5], "int32"))  # one past its last pair
            n_tiles = total * num_n_blocks
            tail_tiles = unsigned_mod(n_tiles, NUM_SMS // 2) if USE_FINAL_WAVE_SPLIT else 0
            tail_tiles = txl.if_then_else(2 * tail_tiles <= NUM_SMS // 2, tail_tiles, 0)
            full_tiles_end = n_tiles - tail_tiles  # tiles [full_tiles_end, n_tiles) run as two column halves each
            with txl.while_(ticket < full_tiles_end + 2 * tail_tiles):
                is_half = txl.local_scalar("int32", init=txl.if_then_else(ticket >= full_tiles_end, 1, 0))
                tile_idx = txl.local_scalar("int32", init=txl.if_then_else(is_half == 1, full_tiles_end + unsigned_div(ticket - full_tiles_end, 2), ticket))
                pair_idx = txl.local_scalar("int32", init=unsigned_div(tile_idx, num_n_blocks))
                n_block = txl.local_scalar("int32", init=unsigned_mod(tile_idx, num_n_blocks))
                col0 = txl.local_scalar("int32", init=n_block * BN + txl.if_then_else(is_half == 1, unsigned_mod(ticket - full_tiles_end, 2) * (BN // 2), 0))
                # The walk's precomputed start assumes the first ticket is cluster_idx; a final-wave split
                # covering the whole problem (< one wave of tiles) starts earlier: restart from group 0.
                with txl.if_(pair_idx < pair_start), txl.then_():
                    txl.assign(group_idx, -1)
                    txl.assign(group_start, 0)
                    txl.assign(group_end, 0)
                    txl.assign(pair_start, 0)
                    txl.assign(pair_end, 0)
                with txl.while_(pair_idx >= pair_end):
                    txl.assign(group_idx, group_idx + 1)
                    txl.assign(group_start, group_end)
                    txl.ptx.ld.global_.s32(group_end, offs.ptr_to([group_idx]))
                    txl.assign(pair_start, pair_end)
                    txl.assign(pair_end, pair_end + unsigned_div(group_end - group_start + 255, 256))
                m0 = txl.local_scalar("int32", init=group_start + (pair_idx - pair_start) * 256 + rank * BM)
                process_tile(m0, col0, group_idx, m0 < group_end, is_half)
                txl.assign(ticket, ticket + NUM_SMS // 2)

        with tma_role:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                data_state = txl.PipelineState(STAGES, 1)

                def process_tile(m0, col0, group_idx, has_rows, is_half):
                    with txl.serial(num_k_blocks) as k:
                        timed_wait("tma-wait-empty", load_empty, data_state.stage, data_state.phase)
                        full_addr = txl.cuda.cvta_generic_to_shared(leader_load_full.ptr_to([data_state.stage]))
                        with txl.if_(rank == 0), txl.then_():
                            leader_load_full.arrive(data_state.stage, txl.if_then_else(is_half == 1, 2 * (A_BYTES + B_HALF // 2), 2 * (A_BYTES + B_HALF)))
                        txl.ptx[LOAD2D_PAIR](s_a.ptr_to([data_state.stage * A_BYTES]), txl.address_of(a_map), txl.cast(k * BK, "int32"), m0, full_addr)
                        with txl.if_(is_half == 0):
                            with txl.then_():
                                txl.ptx[LOAD3D_PAIR](s_b.ptr_to([data_state.stage * B_HALF]), txl.address_of(b_map), txl.cast(k * BK, "int32"), col0 + rank * (BN // 2), group_idx, full_addr)
                            with txl.else_():
                                txl.ptx[LOAD3D_PAIR](s_b.ptr_to([data_state.stage * B_HALF]), txl.address_of(b_half_map), txl.cast(k * BK, "int32"), col0 + rank * (BN // 4), group_idx, full_addr)
                        data_state.advance()

                for_each_tile(process_tile)

        with sf_role:
            data_state, sf_state = txl.PipelineState(STAGES, 1), txl.PipelineState(STAGES, 1)
            sfa_words = txl.alloc_local((16,), "uint32")
            # The tile's B rows [n_block*BN, n_block*BN + BN) span at most 2 (BN <= 128: 1, 192: 2, 256: 2)
            # 128-row weight-scale blocks; 32-row groups never straddle a block (BN % 64 == 0).
            NUM_SCALE_BLOCKS = 1 if BN <= 128 else 2
            sfb_words = txl.alloc_local((4 * NUM_SCALE_BLOCKS,), "uint32")
            scale_f32 = txl.local_scalar("float32")
            num_sfb_blocks = (N + 127) // 128

            def load_scale_group(m0, col0, group_idx, k_block):
                if USE_VECTOR_SCALES:
                    for i in range(4):
                        row = txl.min(m0 + lane + 32 * i, M - 1)
                        txl.ptx.ld.global_.nc.v4.b32(sfa_words[i * 4], sfa_words[i * 4 + 1], sfa_words[i * 4 + 2], sfa_words[i * 4 + 3], sfa.ptr_to([row, k_block]))
                    for chunk in range(NUM_SCALE_BLOCKS):
                        sfb_block = txl.min(unsigned_div(col0, 128) + chunk, num_sfb_blocks - 1)
                        txl.ptx.ld.global_.nc.v4.b32(sfb_words[chunk * 4], sfb_words[chunk * 4 + 1], sfb_words[chunk * 4 + 2], sfb_words[chunk * 4 + 3], sfb.ptr_to([group_idx, sfb_block, k_block]))
                else:
                    for j in range(4):
                        scale_k = txl.min(k_block + j, num_k_blocks - 1)
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfa.ptr_to([txl.min(m0 + lane + 32 * i, M - 1), scale_k]))
                            txl.assign(sfa_words[i * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))
                        for chunk in range(NUM_SCALE_BLOCKS):
                            sfb_block = txl.min(unsigned_div(col0, 128) + chunk, num_sfb_blocks - 1)
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfb.ptr_to([group_idx, sfb_block, scale_k]))
                            txl.assign(sfb_words[chunk * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))

            def process_tile(m0, col0, group_idx, has_rows, is_half):
                load_scale_group(m0, col0, group_idx, 0)
                with txl.serial(num_k_blocks) as k:
                    with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                        # Scale slots advance only when scales change. Reuse is protected by the data
                        # stage at k: after STAGES scale groups, k advanced 4*STAGES and is back on the
                        # same data stage, whose empty barrier includes the prior tcgen05.cp.
                        timed_wait("sf-wait-empty", load_empty, data_state.stage, data_state.phase)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.st.shared.v4.u32(s_sfa.ptr_to([sf_state.stage * 128 + lane * 4]), pack_scales4(sfa_words, 0), pack_scales4(sfa_words, 4), pack_scales4(sfa_words, 8), pack_scales4(sfa_words, 12))
                        sfb_word0 = txl.local_scalar("uint32", init=pack_scales4(sfb_words, 0))
                        sfb_word1 = txl.local_scalar("uint32", init=pack_scales4(sfb_words, 4 * (NUM_SCALE_BLOCKS - 1)))
                        block_offset = unsigned_mod(col0, 128)  # tile start within its first scale block
                        for chunk in range(N_SF_CHUNKS):
                            # word i of chunk holds B rows chunk*128 + 32*i + lane of the tile
                            chunk_words = [txl.if_then_else(block_offset + (chunk * 128 + 32 * i) >= 128, sfb_word1, sfb_word0) for i in range(4)]
                            txl.ptx.st.shared.v4.u32(s_sfb.ptr_to([(sf_state.stage * N_SF_CHUNKS + chunk) * 128 + lane * 4]), *chunk_words)
                        with txl.if_(k + 4 < num_k_blocks), txl.then_():
                            load_scale_group(m0, col0, group_idx, k + 4)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        leader_sf_full.arrive(sf_state.stage)
                        sf_state.advance()
                    data_state.advance()

            for_each_tile(process_tile)

        with mma_role:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    data_state, sf_state, acc_state = txl.PipelineState(STAGES, 0), txl.PipelineState(STAGES, 0), txl.PipelineState(ACC_STAGES, 0)
                    tiles_issued = txl.local_scalar("int32", init=0)
                    a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                    b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                    sfa_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0])))
                    sfb_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0])))

                    def process_tile(m0, col0, group_idx, has_rows, is_half):
                        timed_wait("mma-wait-acc", acc_empty, acc_state.stage, acc_state.phase ^ 1)
                        if USE_SCALE_TAIL:
                            sfa_col = txl.local_scalar("uint32", init=to_u32((1 - acc_state.stage) * 256 + 240))
                            with txl.if_(tiles_issued > 0), txl.then_():
                                # the previous tile (other accumulator) has had its tail chunk read
                                timed_wait("mma-wait-tail", tail_free, 1 - acc_state.stage, to_u32(unsigned_mod(unsigned_div(tiles_issued - 1, 2), 2)))
                        else:
                            sfa_col = txl.uint32(SFA_COL)
                        tile_range = range_start("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(num_k_blocks) as k:
                            a_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, a_base + to_u32(data_state.stage * A_BYTES)))
                            b_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, b_base + to_u32(data_state.stage * B_HALF)))
                            instr_desc = txl.local_scalar("uint32", init=with_sf_id(txl.if_then_else(is_half == 1, txl.uint32(INSTR_DESC_HALF), txl.uint32(INSTR_DESC)), unsigned_mod(k, 4)))
                            sfa_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfa_base + to_u32(sf_state.stage * 512)))
                            sfb_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfb_base + to_u32(sf_state.stage * N_SF_CHUNKS * 512)))
                            timed_wait("mma-wait-tma", load_full, data_state.stage, data_state.phase)
                            with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                                timed_wait("mma-wait-sf", sf_full, sf_state.stage, sf_state.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                                txl.ptx[UTCCP_PAIR](sfa_col, sfa_desc)
                                for chunk in range(N_SF_CHUNKS):
                                    txl.ptx[UTCCP_PAIR](sfa_col + txl.uint32(4 + 4 * chunk), offset_descriptor(sfb_desc, chunk * 512))
                                sf_state.advance()
                            for k_step in range(BK // UMMA_K):
                                txl.ptx[MMA_PAIR](to_u32(acc_state.stage * BN), offset_descriptor(a_desc, k_step * UMMA_K), offset_descriptor(b_desc, k_step * UMMA_K), instr_desc, sfa_col, sfa_col + txl.uint32(4), txl.Or(k > 0, k_step > 0) if k_step == 0 else txl.bool(True))
                            load_empty.arrive(data_state.stage, cta_group=2, cta_mask=3)
                            data_state.advance()
                        acc_full.arrive(acc_state.stage, cta_group=2, cta_mask=3)
                        range_end(tile_range)
                        acc_state.advance()
                        txl.assign(tiles_issued, tiles_issued + 1)

                    for_each_tile(process_tile)

        with epi_role:
            epi_warp = warp - 4
            quad = warp % 4
            warpgroup = epi_warp // 4
            wg_tidx = (epi_warp % 4) * 32 + lane  # thread id within this warpgroup
            row = quad * 32 + lane
            acc_state = txl.PipelineState(ACC_STAGES, 0)
            acc_words, packed = txl.alloc_local((32,), "uint32"), txl.alloc_local((COLS_PER_WG // 2,), "uint32")

            def process_tile(m0, col0, group_idx, has_rows, is_half):
                timed_wait("epi-wait-acc", acc_full, acc_state.stage, acc_state.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                drain_range = range_start("epi-drain")
                # Drain this warpgroup's accumulator columns into packed bf16 registers (USE_SCALE_TAIL: the
                # tail chunk first; warpgroup 1 then frees it for the next tile's scale columns).
                num_chunks = COLS_PER_WG // 32
                order = [num_chunks - 1] + list(range(num_chunks - 1)) if USE_SCALE_TAIL else list(range(num_chunks))
                for i, chunk in enumerate(order):
                    tmem_addr = txl.bitwise_or(txl.shift_left(to_u32(quad * 32), txl.uint32(16)), to_u32(acc_state.stage * BN + warpgroup * COLS_PER_WG + chunk * 32))
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[acc_words[j] for j in range(32)], tmem_addr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    if USE_SCALE_TAIL and i == 0:
                        with txl.if_(warpgroup == 1), txl.then_():
                            txl.ptx.tcgen05.fence__before_thread_sync()
                            tail_free.arrive(acc_state.stage, remote=0)
                    for word in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(packed[chunk * 16 + word], txl.reinterpret(acc_words[2 * word + 1], ty="float32"), txl.reinterpret(acc_words[2 * word], ty="float32"))
                txl.ptx.tcgen05.fence__before_thread_sync()
                acc_empty.arrive(acc_state.stage, remote=0)
                range_end(drain_range)
                store_range = range_start("epi-store")
                for chunk in range(COLS_PER_WG // 32):
                    buf_offset = (warpgroup * STORE_BUFFERS + chunk % STORE_BUFFERS) * (BM * 16)
                    with txl.if_(wg_tidx == 0), txl.then_():
                        txl.ptx.cp.async_.bulk.wait_group.read(STORE_BUFFERS - 1)
                    txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                    for unit in range(4):
                        word_idx = chunk * 16 + unit * 4
                        txl.ptx.st.shared.v4.u32(s_store.ptr_to([buf_offset + row * 16 + txl.bitwise_xor(unit, (row // 2) % 4) * 4]), packed[word_idx], packed[word_idx + 1], packed[word_idx + 2], packed[word_idx + 3])
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                    # A half tile's 128 columns all belong to warpgroup 0.
                    with txl.if_(txl.And(txl.And(wg_tidx == 0, has_rows), txl.Or(is_half == 0, warpgroup == 0))), txl.then_():
                        txl.ptx[STORE2D](txl.address_of(d_map), col0 + warpgroup * COLS_PER_WG + chunk * 32, m0, s_store.ptr_to([buf_offset]))
                    with txl.if_(wg_tidx == 0), txl.then_():
                        txl.ptx.cp.async_.bulk.commit_group()
                range_end(store_range)
                acc_state.advance()

            for_each_tile(process_tile)
            with txl.if_(wg_tidx == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with idle_role:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped


def build_mgrouped_w320(STAGES, USE_VECTOR_SCALES=False):
    """
    BN=320 variant of the pair kernel (n % 320 == 0, e.g. gpt-oss 2880 = 9 x 320): each tile is two
    N=160 sub-MMAs (D cols 0 and 160). CTA r loads B rows [80 r, +80) and [160 + 80 r, +80) of the
    tile so both sub-MMAs land in natural column order; sub-MMA j reads its scales from its own SFB
    chunks. 320 accumulator + 20 scale columns: a single accumulator.

    Both 128-row halves of a pair belong to one group (pairs are formed per group); the odd last
    block of a group is computed by the pair's second CTA on rows it then does not store.
    """
    BN = 320
    STORE_BUFFERS = 1
    A_BYTES, B_HALF = BM * BK, (BN // 2) * BK
    B_SUB = 80 * BK  # one 80-row B box
    ACC_STAGES = 1
    N_SF_CHUNKS = 4  # tile rows [0,128), [128,256) for sub-MMA 0; [160,288), [288,416) for sub-MMA 1
    CHUNK_BASE = (0, 128, 160, 288)
    SFA_COL = 320
    EPI_WARPGROUPS = 2 if BN >= 192 else 1
    EPI_WARPS = 4 * EPI_WARPGROUPS
    COLS_PER_WG = BN // EPI_WARPGROUPS
    assert COLS_PER_WG % 32 == 0 and (COLS_PER_WG // 32) % STORE_BUFFERS == 0
    INSTR_DESC = encode_instr_descriptor_block_scaled_uint32(M=2 * BM, N=160, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2)
    INSTR_DESC_HALF = encode_instr_descriptor_block_scaled_uint32(M=2 * BM, N=BN // 2, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2)
    MMA_PAIR = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP_PAIR = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LOAD2D_PAIR = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
    LOAD3D_PAIR = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def encode_tensor_maps(params):
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        b_half_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = params["M"], params["N"], params["K"], params["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, params["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(b_map, FP8, 3, params["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, 80, 1, 1, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(b_half_map, FP8, 3, params["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 4, 1, 1, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(d_map, "bfloat16", 2, params["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0)
        return a_map, b_map, b_half_map, d_map

    @txl.kernel()
    def mgrouped_w320(a: txl.Tensor((txl.int64(), txl.int64()), FP8, data_alignment=1), sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32, data_alignment=1), b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8, data_alignment=1), sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32, data_alignment=1), d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, K: txl.int32, G: txl.int32):
        tensor_maps = encode_tensor_maps(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=(4 + EPI_WARPS) * 32, cluster=[2]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, b_half_map, d_map = tensor_maps
        rank = txl.cuda.cluster_cta_id("x")
        cluster_idx = txl.cta_id() // 2
        warp = txl.warp_id()
        lane = txl.lane_id()
        tidx = txl.thread_id()

        roles = txl.specialize()
        tma_role = roles.role("tma", warps=[0])
        mma_role = roles.role("mma", warps=[1])
        sf_role = roles.role("sf", warps=[2])
        idle_role = roles.role("idle", warps=[3])
        epi_role = roles.role("epi", warps=list(range(4, 4 + EPI_WARPS)))
        # Issue the first 32 group ends' loads now; their latency overlaps the setup below.
        chunk0_group_end, chunk0_group_start = txl.local_scalar("int32"), txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(chunk0_group_end, offs.ptr_to([txl.min(lane, G - 1)]))
        txl.ptx.ld.global_.s32(chunk0_group_start, offs.ptr_to([txl.max(txl.min(lane, G - 1) - 1, 0)]))
        setup_range = range_start("setup")

        smem = txl.smem_pool()
        s_a = smem.alloc((STAGES * A_BYTES,), "uint8", align=1024)
        s_b = smem.alloc((STAGES * B_HALF,), "uint8", align=1024)
        s_store = smem.alloc((EPI_WARPGROUPS * STORE_BUFFERS * BM * 16,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * N_SF_CHUNKS * 128,), "uint32", align=16)
        load_full = txl.TMABar(smem, STAGES)
        load_empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        acc_full = txl.TCGen05Bar(smem, ACC_STAGES)
        acc_empty = txl.MBarrier(smem, ACC_STAGES)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        load_full.init(1)
        load_empty.init(1)
        sf_full.init(64)
        acc_full.init(1)
        acc_empty.init(2 * 32 * EPI_WARPS)
        tail_free = txl.MBarrier(smem, 2)
        tail_free.init(2 * 32 * 4)  # warpgroup 1 (columns 128-255) of both CTAs
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with idle_role:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](txl.address_of(tmem_slot[0]), txl.uint32(512))
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        leader_load_full = load_full.remote_view(0)
        leader_sf_full = sf_full.remote_view(0)
        range_end(setup_range)

        num_n_blocks, num_k_blocks = (N + (BN - 1)) // BN, (K + (BK - 1)) // BK
        # Lanes take 32 groups at a time: pairs per group (ceil(rows / 256)), their running sum
        # (shfl.up scan + carry) and, by warp-wide reductions, the total and the group holding this
        # cluster's first pair, so for_each_tile() starts there instead of walking from group 0.
        walk_range = range_start("walk")
        carry = txl.local_scalar("uint32", init=txl.uint32(0))
        lane_group_end = txl.local_scalar("int32")
        lane_group_start = txl.local_scalar("int32")
        first_pair = txl.local_scalar("int32", init=unsigned_div(cluster_idx, num_n_blocks))
        walk_start = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(6)]  # has_first_pair, group_idx, group_start, group_end, pair_start, pair_end
        peer_value = txl.local_scalar("uint32")

        def warp_sum(value):
            # one redux.sync per value: the six walk reductions are independent (no 5-deep shuffle chains)
            txl.ptx.redux_sync.add.u32(value, value, txl.uint32(0xFFFFFFFF))

        with txl.serial((G + 31) // 32) as chunk_idx:
            lane_group_idx = txl.local_scalar("int32", init=chunk_idx * 32 + lane)
            with txl.if_(chunk_idx == 0):
                with txl.then_():
                    txl.assign(lane_group_end, chunk0_group_end)
                    txl.assign(lane_group_start, chunk0_group_start)
                with txl.else_():
                    txl.ptx.ld.global_.s32(lane_group_end, offs.ptr_to([txl.min(lane_group_idx, G - 1)]))
                    txl.ptx.ld.global_.s32(lane_group_start, offs.ptr_to([txl.max(txl.min(lane_group_idx, G - 1) - 1, 0)]))
            with txl.if_(lane_group_idx == 0), txl.then_():
                txl.assign(lane_group_start, 0)
            lane_pairs = txl.local_scalar("uint32", init=txl.if_then_else(lane_group_idx < G, to_u32(unsigned_div(lane_group_end - lane_group_start + 255, 256)), txl.uint32(0)))
            pair_scan = txl.local_scalar("uint32", init=lane_pairs)
            for scan_offset in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer_value, pair_scan, txl.uint32(scan_offset), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.if_(lane >= scan_offset), txl.then_():
                    txl.assign(pair_scan, pair_scan + peer_value)
            lane_pair_end = txl.local_scalar("int32", init=txl.cast(carry + pair_scan, "int32"))
            lane_pair_start = txl.local_scalar("int32", init=lane_pair_end - txl.cast(lane_pairs, "int32"))
            has_first_pair = txl.local_scalar("uint32", init=txl.if_then_else(txl.And(lane_pair_start <= first_pair, first_pair < lane_pair_end), txl.uint32(1), txl.uint32(0)))
            walk_terms = [has_first_pair, has_first_pair * to_u32(lane_group_idx), has_first_pair * to_u32(lane_group_start), has_first_pair * to_u32(lane_group_end), has_first_pair * to_u32(lane_pair_start), has_first_pair * to_u32(lane_pair_end)]
            for j in range(6):
                value = txl.local_scalar("uint32", init=walk_terms[j])
                warp_sum(value)
                txl.assign(walk_start[j], walk_start[j] + value)
            chunk_pairs = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.idx.b32(chunk_pairs, pair_scan, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(carry, carry + chunk_pairs)
        total = txl.local_scalar("int32", init=txl.cast(carry, "int32"))
        has_start = walk_start[0] != txl.uint32(0)
        range_end(walk_range)
        # the group walk (registers + global loads only) overlaps the cluster barrier latency
        txl.ptx.barrier.cluster.wait.acquire.aligned()

        def for_each_tile(process_tile):
            """
            Pair walk: ticket -> (pair_idx, n_block); pairs are counted per group (ceil(rows/256)).
            """
            ticket = txl.local_scalar("int32", init=cluster_idx)
            group_idx = txl.local_scalar("int32", init=txl.if_then_else(has_start, txl.cast(walk_start[1], "int32"), -1))
            group_start = txl.local_scalar("int32", init=txl.cast(walk_start[2], "int32"))  # first row of group group_idx
            group_end = txl.local_scalar("int32", init=txl.cast(walk_start[3], "int32"))  # one past its last row
            pair_start = txl.local_scalar("int32", init=txl.cast(walk_start[4], "int32"))  # first pair of group group_idx
            pair_end = txl.local_scalar("int32", init=txl.cast(walk_start[5], "int32"))  # one past its last pair
            n_tiles = total * num_n_blocks
            tail_tiles = 0  # no final-wave split
            tail_tiles = txl.if_then_else(2 * tail_tiles <= NUM_SMS // 2, tail_tiles, 0)
            full_tiles_end = n_tiles - tail_tiles  # tiles [full_tiles_end, n_tiles) run as two column halves each
            with txl.while_(ticket < full_tiles_end + 2 * tail_tiles):
                is_half = txl.local_scalar("int32", init=txl.if_then_else(ticket >= full_tiles_end, 1, 0))
                tile_idx = txl.local_scalar("int32", init=txl.if_then_else(is_half == 1, full_tiles_end + unsigned_div(ticket - full_tiles_end, 2), ticket))
                pair_idx = txl.local_scalar("int32", init=unsigned_div(tile_idx, num_n_blocks))
                n_block = txl.local_scalar("int32", init=unsigned_mod(tile_idx, num_n_blocks))
                col0 = txl.local_scalar("int32", init=n_block * BN + txl.if_then_else(is_half == 1, unsigned_mod(ticket - full_tiles_end, 2) * (BN // 2), 0))
                # The walk's precomputed start assumes the first ticket is cluster_idx; a final-wave split
                # covering the whole problem (< one wave of tiles) starts earlier: restart from group 0.
                with txl.if_(pair_idx < pair_start), txl.then_():
                    txl.assign(group_idx, -1)
                    txl.assign(group_start, 0)
                    txl.assign(group_end, 0)
                    txl.assign(pair_start, 0)
                    txl.assign(pair_end, 0)
                with txl.while_(pair_idx >= pair_end):
                    txl.assign(group_idx, group_idx + 1)
                    txl.assign(group_start, group_end)
                    txl.ptx.ld.global_.s32(group_end, offs.ptr_to([group_idx]))
                    txl.assign(pair_start, pair_end)
                    txl.assign(pair_end, pair_end + unsigned_div(group_end - group_start + 255, 256))
                m0 = txl.local_scalar("int32", init=group_start + (pair_idx - pair_start) * 256 + rank * BM)
                process_tile(m0, col0, group_idx, m0 < group_end, is_half)
                txl.assign(ticket, ticket + NUM_SMS // 2)

        with tma_role:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                data_state = txl.PipelineState(STAGES, 1)

                def process_tile(m0, col0, group_idx, has_rows, is_half):
                    with txl.serial(num_k_blocks) as k:
                        timed_wait("tma-wait-empty", load_empty, data_state.stage, data_state.phase)
                        full_addr = txl.cuda.cvta_generic_to_shared(leader_load_full.ptr_to([data_state.stage]))
                        with txl.if_(rank == 0), txl.then_():
                            leader_load_full.arrive(data_state.stage, txl.if_then_else(is_half == 1, 2 * (A_BYTES + B_HALF // 2), 2 * (A_BYTES + B_HALF)))
                        txl.ptx[LOAD2D_PAIR](s_a.ptr_to([data_state.stage * A_BYTES]), txl.address_of(a_map), txl.cast(k * BK, "int32"), m0, full_addr)
                        for j in range(2):
                            txl.ptx[LOAD3D_PAIR](s_b.ptr_to([data_state.stage * B_HALF + j * B_SUB]), txl.address_of(b_map), txl.cast(k * BK, "int32"), col0 + j * 160 + rank * 80, group_idx, full_addr)
                        data_state.advance()

                for_each_tile(process_tile)

        with sf_role:
            data_state, sf_state = txl.PipelineState(STAGES, 1), txl.PipelineState(STAGES, 1)
            sfa_words = txl.alloc_local((16,), "uint32")
            NUM_SCALE_BLOCKS = 3  # a 320-row tile starting at a multiple of 64 spans <= 3 weight-scale blocks
            sfb_words = txl.alloc_local((4 * NUM_SCALE_BLOCKS,), "uint32")
            scale_f32 = txl.local_scalar("float32")
            num_sfb_blocks = (N + 127) // 128

            def load_scale_group(m0, col0, group_idx, k_block):
                if USE_VECTOR_SCALES:
                    for i in range(4):
                        row = txl.min(m0 + lane + 32 * i, M - 1)
                        txl.ptx.ld.global_.nc.v4.b32(sfa_words[i * 4], sfa_words[i * 4 + 1], sfa_words[i * 4 + 2], sfa_words[i * 4 + 3], sfa.ptr_to([row, k_block]))
                    for chunk in range(NUM_SCALE_BLOCKS):
                        sfb_block = txl.min(unsigned_div(col0, 128) + chunk, num_sfb_blocks - 1)
                        txl.ptx.ld.global_.nc.v4.b32(sfb_words[chunk * 4], sfb_words[chunk * 4 + 1], sfb_words[chunk * 4 + 2], sfb_words[chunk * 4 + 3], sfb.ptr_to([group_idx, sfb_block, k_block]))
                else:
                    for j in range(4):
                        scale_k = txl.min(k_block + j, num_k_blocks - 1)
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfa.ptr_to([txl.min(m0 + lane + 32 * i, M - 1), scale_k]))
                            txl.assign(sfa_words[i * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))
                        for chunk in range(NUM_SCALE_BLOCKS):
                            sfb_block = txl.min(unsigned_div(col0, 128) + chunk, num_sfb_blocks - 1)
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfb.ptr_to([group_idx, sfb_block, scale_k]))
                            txl.assign(sfb_words[chunk * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))

            def process_tile(m0, col0, group_idx, has_rows, is_half):
                load_scale_group(m0, col0, group_idx, 0)
                with txl.serial(num_k_blocks) as k:
                    with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                        # Scale slots advance only when scales change. Reuse is protected by the data
                        # stage at k: after STAGES scale groups, k advanced 4*STAGES and is back on the
                        # same data stage, whose empty barrier includes the prior tcgen05.cp.
                        timed_wait("sf-wait-empty", load_empty, data_state.stage, data_state.phase)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.st.shared.v4.u32(s_sfa.ptr_to([sf_state.stage * 128 + lane * 4]), pack_scales4(sfa_words, 0), pack_scales4(sfa_words, 4), pack_scales4(sfa_words, 8), pack_scales4(sfa_words, 12))
                        sfb_packed = [txl.local_scalar("uint32", init=pack_scales4(sfb_words, 4 * j)) for j in range(NUM_SCALE_BLOCKS)]
                        block_offset = unsigned_mod(col0, 128)  # tile start within its first scale block
                        for chunk in range(N_SF_CHUNKS):
                            # word i of chunk holds B rows CHUNK_BASE[chunk] + 32*i + lane of the tile
                            chunk_words = []
                            for i in range(4):
                                block_row = block_offset + (CHUNK_BASE[chunk] + 32 * i)
                                chunk_words.append(txl.if_then_else(block_row >= 256, sfb_packed[2], txl.if_then_else(block_row >= 128, sfb_packed[1], sfb_packed[0])))
                            txl.ptx.st.shared.v4.u32(s_sfb.ptr_to([(sf_state.stage * N_SF_CHUNKS + chunk) * 128 + lane * 4]), *chunk_words)
                        with txl.if_(k + 4 < num_k_blocks), txl.then_():
                            load_scale_group(m0, col0, group_idx, k + 4)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        leader_sf_full.arrive(sf_state.stage)
                        sf_state.advance()
                    data_state.advance()

            for_each_tile(process_tile)

        with mma_role:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    data_state, sf_state, acc_state = txl.PipelineState(STAGES, 0), txl.PipelineState(STAGES, 0), txl.PipelineState(ACC_STAGES, 0)
                    tiles_issued = txl.local_scalar("int32", init=0)
                    a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                    b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                    sfa_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0])))
                    sfb_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0])))

                    def process_tile(m0, col0, group_idx, has_rows, is_half):
                        timed_wait("mma-wait-acc", acc_empty, acc_state.stage, acc_state.phase ^ 1)
                        sfa_col = txl.uint32(SFA_COL)
                        tile_range = range_start("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(num_k_blocks) as k:
                            a_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, a_base + to_u32(data_state.stage * A_BYTES)))
                            b_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, b_base + to_u32(data_state.stage * B_HALF)))
                            instr_desc = txl.local_scalar("uint32", init=with_sf_id(txl.if_then_else(is_half == 1, txl.uint32(INSTR_DESC_HALF), txl.uint32(INSTR_DESC)), unsigned_mod(k, 4)))
                            sfa_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfa_base + to_u32(sf_state.stage * 512)))
                            sfb_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfb_base + to_u32(sf_state.stage * N_SF_CHUNKS * 512)))
                            timed_wait("mma-wait-tma", load_full, data_state.stage, data_state.phase)
                            with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                                timed_wait("mma-wait-sf", sf_full, sf_state.stage, sf_state.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                                txl.ptx[UTCCP_PAIR](sfa_col, sfa_desc)
                                for chunk in range(N_SF_CHUNKS):
                                    txl.ptx[UTCCP_PAIR](sfa_col + txl.uint32(4 + 4 * chunk), offset_descriptor(sfb_desc, chunk * 512))
                                sf_state.advance()
                            for k_step in range(BK // UMMA_K):
                                for j in range(2):
                                    txl.ptx[MMA_PAIR](txl.uint32(160 * j), offset_descriptor(a_desc, k_step * UMMA_K), offset_descriptor(b_desc, j * B_SUB + k_step * UMMA_K), instr_desc, sfa_col, sfa_col + txl.uint32(4 + 8 * j), txl.Or(k > 0, k_step > 0) if k_step == 0 else txl.bool(True))
                            load_empty.arrive(data_state.stage, cta_group=2, cta_mask=3)
                            data_state.advance()
                        acc_full.arrive(acc_state.stage, cta_group=2, cta_mask=3)
                        range_end(tile_range)
                        acc_state.advance()
                        txl.assign(tiles_issued, tiles_issued + 1)

                    for_each_tile(process_tile)

        with epi_role:
            epi_warp = warp - 4
            quad = warp % 4
            warpgroup = epi_warp // 4
            wg_tidx = (epi_warp % 4) * 32 + lane  # thread id within this warpgroup
            row = quad * 32 + lane
            acc_state = txl.PipelineState(ACC_STAGES, 0)
            acc_words, packed = txl.alloc_local((32,), "uint32"), txl.alloc_local((COLS_PER_WG // 2,), "uint32")

            def process_tile(m0, col0, group_idx, has_rows, is_half):
                timed_wait("epi-wait-acc", acc_full, acc_state.stage, acc_state.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                drain_range = range_start("epi-drain")
                # Drain this warpgroup's accumulator columns into packed bf16 registers.
                for chunk in range(COLS_PER_WG // 32):
                    tmem_addr = txl.bitwise_or(txl.shift_left(to_u32(quad * 32), txl.uint32(16)), to_u32(acc_state.stage * BN + warpgroup * COLS_PER_WG + chunk * 32))
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[acc_words[j] for j in range(32)], tmem_addr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    for word in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(packed[chunk * 16 + word], txl.reinterpret(acc_words[2 * word + 1], ty="float32"), txl.reinterpret(acc_words[2 * word], ty="float32"))
                txl.ptx.tcgen05.fence__before_thread_sync()
                acc_empty.arrive(acc_state.stage, remote=0)
                range_end(drain_range)
                store_range = range_start("epi-store")
                for chunk in range(COLS_PER_WG // 32):
                    buf_offset = (warpgroup * STORE_BUFFERS + chunk % STORE_BUFFERS) * (BM * 16)
                    with txl.if_(wg_tidx == 0), txl.then_():
                        txl.ptx.cp.async_.bulk.wait_group.read(STORE_BUFFERS - 1)
                    txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                    for unit in range(4):
                        word_idx = chunk * 16 + unit * 4
                        txl.ptx.st.shared.v4.u32(s_store.ptr_to([buf_offset + row * 16 + txl.bitwise_xor(unit, (row // 2) % 4) * 4]), packed[word_idx], packed[word_idx + 1], packed[word_idx + 2], packed[word_idx + 3])
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                    # A half tile's 128 columns all belong to warpgroup 0.
                    with txl.if_(txl.And(txl.And(wg_tidx == 0, has_rows), txl.Or(is_half == 0, warpgroup == 0))), txl.then_():
                        txl.ptx[STORE2D](txl.address_of(d_map), col0 + warpgroup * COLS_PER_WG + chunk * 32, m0, s_store.ptr_to([buf_offset]))
                    with txl.if_(wg_tidx == 0), txl.then_():
                        txl.ptx.cp.async_.bulk.commit_group()
                range_end(store_range)
                acc_state.advance()

            for_each_tile(process_tile)
            with txl.if_(wg_tidx == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with idle_role:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped_w320


def build_mgrouped_w320r(STAGES, USE_VECTOR_SCALES=False, DEFER_STEPS=2, SUB_N=160, STORE_BUFFERS=1, USE_LAST_TILE_BUFFERS=False):
    """
    Pair kernel whose tiles are two SUB_N-wide sub-MMAs over rotating accumulator slots: SUB_N=160
    for n % 320 == 0 (gpt-oss 2880 = 9 x 320), SUB_N=128 for 256-wide tiles. CTA r loads B rows
    [SUB_N/2 r, +SUB_N/2) and [SUB_N + SUB_N/2 r, +SUB_N/2) of the tile so both sub-MMAs land in
    natural column order; sub-MMA j reads its scales from its own SFB chunks.

    TMEM holds three SUB_N-column slots plus the scale columns. A tile uses (slot0, slot1); the next
    tile uses (free slot, slot0). Its sub-MMA 0 starts at once in the free slot; the sub-MMA 1 work
    of its first DEFER_STEPS K steps is deferred (those stages stay held) until the epilogue frees
    slot0, so the tensor pipe keeps working through the previous tile's commit -> drain round trip.

    Both 128-row halves of a pair belong to one group (pairs are formed per group); the odd last
    block of a group is computed by the pair's second CTA on rows it then does not store.
    """
    assert 1 <= DEFER_STEPS <= 4  # deferred sub-MMA 1 steps must use the scale group copied at k = 0
    BN = 2 * SUB_N  # two sub-MMAs of width SUB_N
    SUB_SF_CHUNKS = (SUB_N + 127) // 128  # SFB chunks per sub-MMA
    A_BYTES, B_HALF = BM * BK, (BN // 2) * BK
    B_SUB = (SUB_N // 2) * BK  # one B box (SUB_N / 2 rows per CTA per sub-MMA)
    N_SF_CHUNKS = 2 * SUB_SF_CHUNKS  # chunk c covers tile rows [CHUNK_BASE[c], +128)
    CHUNK_BASE = [SUB_N * (chunk // SUB_SF_CHUNKS) + 128 * (chunk % SUB_SF_CHUNKS) for chunk in range(N_SF_CHUNKS)]
    SFA_COL = 3 * SUB_N
    assert SFA_COL + 4 + 4 * N_SF_CHUNKS <= 512
    EPI_WARPGROUPS = 2 if BN >= 192 else 1
    EPI_WARPS = 4 * EPI_WARPGROUPS
    COLS_PER_WG = BN // EPI_WARPGROUPS
    assert COLS_PER_WG % 32 == 0  # store buffers rotate on a running chunk count (any STORE_BUFFERS)
    INSTR_DESC = encode_instr_descriptor_block_scaled_uint32(M=2 * BM, N=SUB_N, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2)
    MMA_PAIR = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP_PAIR = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LOAD2D_PAIR = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
    LOAD3D_PAIR = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def encode_tensor_maps(params):
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        b_half_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = params["M"], params["N"], params["K"], params["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, params["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(b_map, FP8, 3, params["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, SUB_N // 2, 1, 1, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(b_half_map, FP8, 3, params["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN // 4, 1, 1, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(d_map, "bfloat16", 2, params["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0)
        return a_map, b_map, b_half_map, d_map

    @txl.kernel()
    def mgrouped_w320r(a: txl.Tensor((txl.int64(), txl.int64()), FP8, data_alignment=1), sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32, data_alignment=1), b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8, data_alignment=1), sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32, data_alignment=1), d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, K: txl.int32, G: txl.int32):
        tensor_maps = encode_tensor_maps(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=(4 + EPI_WARPS) * 32, cluster=[2]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, b_half_map, d_map = tensor_maps
        rank = txl.cuda.cluster_cta_id("x")
        cluster_idx = txl.cta_id() // 2
        warp = txl.warp_id()
        lane = txl.lane_id()
        tidx = txl.thread_id()

        roles = txl.specialize()
        tma_role = roles.role("tma", warps=[0])
        mma_role = roles.role("mma", warps=[1])
        sf_role = roles.role("sf", warps=[2])
        idle_role = roles.role("idle", warps=[3])
        epi_role = roles.role("epi", warps=list(range(4, 4 + EPI_WARPS)))
        # Issue the first 32 group ends' loads now; their latency overlaps the setup below.
        chunk0_group_end, chunk0_group_start = txl.local_scalar("int32"), txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(chunk0_group_end, offs.ptr_to([txl.min(lane, G - 1)]))
        txl.ptx.ld.global_.s32(chunk0_group_start, offs.ptr_to([txl.max(txl.min(lane, G - 1) - 1, 0)]))
        setup_range = range_start("setup")

        smem = txl.smem_pool()
        s_a = smem.alloc((STAGES * A_BYTES,), "uint8", align=1024)
        s_b = smem.alloc((STAGES * B_HALF,), "uint8", align=1024)
        s_store = smem.alloc((EPI_WARPGROUPS * (2 if USE_LAST_TILE_BUFFERS else STORE_BUFFERS) * BM * 16,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * N_SF_CHUNKS * 128,), "uint32", align=16)
        load_full = txl.TMABar(smem, STAGES)
        load_empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        acc_full = txl.TCGen05Bar(smem, 2)
        slot_free = txl.MBarrier(smem, 3)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        load_full.init(1)
        load_empty.init(1)
        sf_full.init(64)
        acc_full.init(1)
        slot_free.init(2 * 128)  # one epilogue warpgroup of each CTA drains a slot
        tail_free = txl.MBarrier(smem, 2)
        tail_free.init(2 * 32 * 4)  # warpgroup 1 (columns 128-255) of both CTAs
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with idle_role:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](txl.address_of(tmem_slot[0]), txl.uint32(512))
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        leader_load_full = load_full.remote_view(0)
        leader_sf_full = sf_full.remote_view(0)
        range_end(setup_range)

        num_n_blocks, num_k_blocks = (N + (BN - 1)) // BN, (K + (BK - 1)) // BK
        # Lanes take 32 groups at a time: pairs per group (ceil(rows / 256)), their running sum
        # (shfl.up scan + carry) and, by warp-wide reductions, the total and the group holding this
        # cluster's first pair, so for_each_tile() starts there instead of walking from group 0.
        walk_range = range_start("walk")
        carry = txl.local_scalar("uint32", init=txl.uint32(0))
        lane_group_end = txl.local_scalar("int32")
        lane_group_start = txl.local_scalar("int32")
        first_pair = txl.local_scalar("int32", init=unsigned_div(cluster_idx, num_n_blocks))
        walk_start = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(6)]  # has_first_pair, group_idx, group_start, group_end, pair_start, pair_end
        peer_value = txl.local_scalar("uint32")

        def warp_sum(value):
            # one redux.sync per value: the six walk reductions are independent (no 5-deep shuffle chains)
            txl.ptx.redux_sync.add.u32(value, value, txl.uint32(0xFFFFFFFF))

        with txl.serial((G + 31) // 32) as chunk_idx:
            lane_group_idx = txl.local_scalar("int32", init=chunk_idx * 32 + lane)
            with txl.if_(chunk_idx == 0):
                with txl.then_():
                    txl.assign(lane_group_end, chunk0_group_end)
                    txl.assign(lane_group_start, chunk0_group_start)
                with txl.else_():
                    txl.ptx.ld.global_.s32(lane_group_end, offs.ptr_to([txl.min(lane_group_idx, G - 1)]))
                    txl.ptx.ld.global_.s32(lane_group_start, offs.ptr_to([txl.max(txl.min(lane_group_idx, G - 1) - 1, 0)]))
            with txl.if_(lane_group_idx == 0), txl.then_():
                txl.assign(lane_group_start, 0)
            lane_pairs = txl.local_scalar("uint32", init=txl.if_then_else(lane_group_idx < G, to_u32(unsigned_div(lane_group_end - lane_group_start + 255, 256)), txl.uint32(0)))
            pair_scan = txl.local_scalar("uint32", init=lane_pairs)
            for scan_offset in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer_value, pair_scan, txl.uint32(scan_offset), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.if_(lane >= scan_offset), txl.then_():
                    txl.assign(pair_scan, pair_scan + peer_value)
            lane_pair_end = txl.local_scalar("int32", init=txl.cast(carry + pair_scan, "int32"))
            lane_pair_start = txl.local_scalar("int32", init=lane_pair_end - txl.cast(lane_pairs, "int32"))
            has_first_pair = txl.local_scalar("uint32", init=txl.if_then_else(txl.And(lane_pair_start <= first_pair, first_pair < lane_pair_end), txl.uint32(1), txl.uint32(0)))
            walk_terms = [has_first_pair, has_first_pair * to_u32(lane_group_idx), has_first_pair * to_u32(lane_group_start), has_first_pair * to_u32(lane_group_end), has_first_pair * to_u32(lane_pair_start), has_first_pair * to_u32(lane_pair_end)]
            for j in range(6):
                value = txl.local_scalar("uint32", init=walk_terms[j])
                warp_sum(value)
                txl.assign(walk_start[j], walk_start[j] + value)
            chunk_pairs = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.idx.b32(chunk_pairs, pair_scan, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(carry, carry + chunk_pairs)
        total = txl.local_scalar("int32", init=txl.cast(carry, "int32"))
        has_start = walk_start[0] != txl.uint32(0)
        range_end(walk_range)
        # the group walk (registers + global loads only) overlaps the cluster barrier latency
        txl.ptx.barrier.cluster.wait.acquire.aligned()

        def for_each_tile(process_tile):
            """
            Pair walk: ticket -> (pair_idx, n_block); pairs are counted per group (ceil(rows/256)).
            """
            ticket = txl.local_scalar("int32", init=cluster_idx)
            group_idx = txl.local_scalar("int32", init=txl.if_then_else(has_start, txl.cast(walk_start[1], "int32"), -1))
            group_start = txl.local_scalar("int32", init=txl.cast(walk_start[2], "int32"))  # first row of group group_idx
            group_end = txl.local_scalar("int32", init=txl.cast(walk_start[3], "int32"))  # one past its last row
            pair_start = txl.local_scalar("int32", init=txl.cast(walk_start[4], "int32"))  # first pair of group group_idx
            pair_end = txl.local_scalar("int32", init=txl.cast(walk_start[5], "int32"))  # one past its last pair
            n_tiles = total * num_n_blocks
            tail_tiles = 0  # no final-wave split
            tail_tiles = txl.if_then_else(2 * tail_tiles <= NUM_SMS // 2, tail_tiles, 0)
            full_tiles_end = n_tiles - tail_tiles  # tiles [full_tiles_end, n_tiles) run as two column halves each
            with txl.while_(ticket < full_tiles_end + 2 * tail_tiles):
                is_half = txl.local_scalar("int32", init=txl.if_then_else(ticket >= full_tiles_end, 1, 0))
                tile_idx = txl.local_scalar("int32", init=txl.if_then_else(is_half == 1, full_tiles_end + unsigned_div(ticket - full_tiles_end, 2), ticket))
                pair_idx = txl.local_scalar("int32", init=unsigned_div(tile_idx, num_n_blocks))
                n_block = txl.local_scalar("int32", init=unsigned_mod(tile_idx, num_n_blocks))
                col0 = txl.local_scalar("int32", init=n_block * BN + txl.if_then_else(is_half == 1, unsigned_mod(ticket - full_tiles_end, 2) * (BN // 2), 0))
                # The walk's precomputed start assumes the first ticket is cluster_idx; a final-wave split
                # covering the whole problem (< one wave of tiles) starts earlier: restart from group 0.
                with txl.if_(pair_idx < pair_start), txl.then_():
                    txl.assign(group_idx, -1)
                    txl.assign(group_start, 0)
                    txl.assign(group_end, 0)
                    txl.assign(pair_start, 0)
                    txl.assign(pair_end, 0)
                with txl.while_(pair_idx >= pair_end):
                    txl.assign(group_idx, group_idx + 1)
                    txl.assign(group_start, group_end)
                    txl.ptx.ld.global_.s32(group_end, offs.ptr_to([group_idx]))
                    txl.assign(pair_start, pair_end)
                    txl.assign(pair_end, pair_end + unsigned_div(group_end - group_start + 255, 256))
                m0 = txl.local_scalar("int32", init=group_start + (pair_idx - pair_start) * 256 + rank * BM)
                process_tile(m0, col0, group_idx, m0 < group_end, is_half, ticket + NUM_SMS // 2 >= full_tiles_end + 2 * tail_tiles)
                txl.assign(ticket, ticket + NUM_SMS // 2)

        with tma_role:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                data_state = txl.PipelineState(STAGES, 1)

                def process_tile(m0, col0, group_idx, has_rows, is_half, is_last=None):
                    with txl.serial(num_k_blocks) as k:
                        timed_wait("tma-wait-empty", load_empty, data_state.stage, data_state.phase)
                        full_addr = txl.cuda.cvta_generic_to_shared(leader_load_full.ptr_to([data_state.stage]))
                        with txl.if_(rank == 0), txl.then_():
                            leader_load_full.arrive(data_state.stage, txl.if_then_else(is_half == 1, 2 * (A_BYTES + B_HALF // 2), 2 * (A_BYTES + B_HALF)))
                        txl.ptx[LOAD2D_PAIR](s_a.ptr_to([data_state.stage * A_BYTES]), txl.address_of(a_map), txl.cast(k * BK, "int32"), m0, full_addr)
                        for j in range(2):
                            txl.ptx[LOAD3D_PAIR](s_b.ptr_to([data_state.stage * B_HALF + j * B_SUB]), txl.address_of(b_map), txl.cast(k * BK, "int32"), col0 + j * SUB_N + rank * (SUB_N // 2), group_idx, full_addr)
                        data_state.advance()

                for_each_tile(process_tile)

        with sf_role:
            data_state, sf_state = txl.PipelineState(STAGES, 1), txl.PipelineState(STAGES, 1)
            sfa_words = txl.alloc_local((16,), "uint32")
            NUM_SCALE_BLOCKS = (BN + 64 + 127) // 128  # weight-scale blocks a tile starting at a multiple of 64 spans
            sfb_words = txl.alloc_local((4 * NUM_SCALE_BLOCKS,), "uint32")
            scale_f32 = txl.local_scalar("float32")
            num_sfb_blocks = (N + 127) // 128

            def load_scale_group(m0, col0, group_idx, k_block):
                if USE_VECTOR_SCALES:
                    for i in range(4):
                        row = txl.min(m0 + lane + 32 * i, M - 1)
                        txl.ptx.ld.global_.nc.v4.b32(sfa_words[i * 4], sfa_words[i * 4 + 1], sfa_words[i * 4 + 2], sfa_words[i * 4 + 3], sfa.ptr_to([row, k_block]))
                    for chunk in range(NUM_SCALE_BLOCKS):
                        sfb_block = txl.min(unsigned_div(col0, 128) + chunk, num_sfb_blocks - 1)
                        txl.ptx.ld.global_.nc.v4.b32(sfb_words[chunk * 4], sfb_words[chunk * 4 + 1], sfb_words[chunk * 4 + 2], sfb_words[chunk * 4 + 3], sfb.ptr_to([group_idx, sfb_block, k_block]))
                else:
                    for j in range(4):
                        scale_k = txl.min(k_block + j, num_k_blocks - 1)
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfa.ptr_to([txl.min(m0 + lane + 32 * i, M - 1), scale_k]))
                            txl.assign(sfa_words[i * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))
                        for chunk in range(NUM_SCALE_BLOCKS):
                            sfb_block = txl.min(unsigned_div(col0, 128) + chunk, num_sfb_blocks - 1)
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfb.ptr_to([group_idx, sfb_block, scale_k]))
                            txl.assign(sfb_words[chunk * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))

            def process_tile(m0, col0, group_idx, has_rows, is_half, is_last=None):
                load_scale_group(m0, col0, group_idx, 0)
                with txl.serial(num_k_blocks) as k:
                    with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                        # Scale slots advance only when scales change. Reuse is protected by the data
                        # stage at k: after STAGES scale groups, k advanced 4*STAGES and is back on the
                        # same data stage, whose empty barrier includes the prior tcgen05.cp.
                        timed_wait("sf-wait-empty", load_empty, data_state.stage, data_state.phase)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.st.shared.v4.u32(s_sfa.ptr_to([sf_state.stage * 128 + lane * 4]), pack_scales4(sfa_words, 0), pack_scales4(sfa_words, 4), pack_scales4(sfa_words, 8), pack_scales4(sfa_words, 12))
                        sfb_packed = [txl.local_scalar("uint32", init=pack_scales4(sfb_words, 4 * j)) for j in range(NUM_SCALE_BLOCKS)]
                        block_offset = unsigned_mod(col0, 128)  # tile start within its first scale block
                        for chunk in range(N_SF_CHUNKS):
                            # word i of chunk holds B rows CHUNK_BASE[chunk] + 32*i + lane of the tile
                            chunk_words = []
                            for i in range(4):
                                block_row = block_offset + (CHUNK_BASE[chunk] + 32 * i)
                                chunk_words.append(txl.if_then_else(block_row >= 256, sfb_packed[2], txl.if_then_else(block_row >= 128, sfb_packed[1], sfb_packed[0])))
                            txl.ptx.st.shared.v4.u32(s_sfb.ptr_to([(sf_state.stage * N_SF_CHUNKS + chunk) * 128 + lane * 4]), *chunk_words)
                        with txl.if_(k + 4 < num_k_blocks), txl.then_():
                            load_scale_group(m0, col0, group_idx, k + 4)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        leader_sf_full.arrive(sf_state.stage)
                        sf_state.advance()
                    data_state.advance()

            for_each_tile(process_tile)

        with mma_role:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    data_state, sf_state, acc_state = txl.PipelineState(STAGES, 0), txl.PipelineState(STAGES, 0), txl.PipelineState(2, 0)
                    a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                    b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                    sfa_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0])))
                    sfb_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0])))

                    slot0, slot1 = txl.local_scalar("int32", init=0), txl.local_scalar("int32", init=1)
                    slot_phases = txl.local_scalar("int32", init=0)  # bit i: phase of slot_free[i]
                    sfa_col = txl.uint32(SFA_COL)

                    def make_descriptors(stage, k):
                        a_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, a_base + to_u32(stage * A_BYTES)))
                        b_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, b_base + to_u32(stage * B_HALF)))
                        instr_desc = txl.local_scalar("uint32", init=with_sf_id(txl.uint32(INSTR_DESC), unsigned_mod(k, 4)))
                        return a_desc, b_desc, instr_desc

                    def sub_mma(j, col, a_desc, b_desc, instr_desc, k):
                        for k_step in range(BK // UMMA_K):
                            txl.ptx[MMA_PAIR](col, offset_descriptor(a_desc, k_step * UMMA_K), offset_descriptor(b_desc, j * B_SUB + k_step * UMMA_K), instr_desc, sfa_col, sfa_col + txl.uint32(4 + 4 * SUB_SF_CHUNKS * j), txl.Or(k > 0, k_step > 0) if k_step == 0 else txl.bool(True))

                    def wait_slot(slot):
                        timed_wait("mma-wait-acc", slot_free, slot, txl.bitwise_and(txl.shift_right(slot_phases, slot), 1) ^ 1)

                    def load_step(k):
                        """
                        Wait for data stage data_state (K step k); copy the scale group at k % 4 == 0.
                        """
                        sfa_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfa_base + to_u32(sf_state.stage * 512)))
                        sfb_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfb_base + to_u32(sf_state.stage * N_SF_CHUNKS * 512)))
                        timed_wait("mma-wait-tma", load_full, data_state.stage, data_state.phase)
                        with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                            timed_wait("mma-wait-sf", sf_full, sf_state.stage, sf_state.phase)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                            txl.ptx[UTCCP_PAIR](sfa_col, sfa_desc)
                            for chunk in range(N_SF_CHUNKS):
                                txl.ptx[UTCCP_PAIR](sfa_col + txl.uint32(4 + 4 * chunk), offset_descriptor(sfb_desc, chunk * 512))
                            sf_state.advance()

                    def process_tile(m0, col0, group_idx, has_rows, is_half, is_last=None):
                        slot0_col, slot1_col = txl.local_scalar("uint32", init=to_u32(slot0 * SUB_N)), txl.local_scalar("uint32", init=to_u32(slot1 * SUB_N))
                        wait_slot(slot0)  # the free slot: drained a tile ago
                        tile_range = range_start("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        num_deferred = txl.local_scalar("int32", init=txl.min(num_k_blocks, DEFER_STEPS))
                        first_stage = txl.local_scalar("int32", init=data_state.stage)
                        # first num_deferred K steps: sub-MMA 0 only (stages stay held)
                        with txl.serial(num_deferred) as k:
                            a_desc, b_desc, instr_desc = make_descriptors(data_state.stage, k)
                            load_step(k)
                            sub_mma(0, slot0_col, a_desc, b_desc, instr_desc, k)
                            data_state.advance()
                        # the previous tile's first slot is drained: catch up sub-MMA 1, release those stages
                        wait_slot(slot1)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(num_deferred) as k:
                            stage = txl.local_scalar("int32", init=unsigned_mod(first_stage + k, STAGES))
                            a_desc, b_desc, instr_desc = make_descriptors(stage, k)
                            sub_mma(1, slot1_col, a_desc, b_desc, instr_desc, k)
                            load_empty.arrive(stage, cta_group=2, cta_mask=3)
                        with txl.serial(num_k_blocks - num_deferred) as rest_idx:
                            k = rest_idx + num_deferred
                            a_desc, b_desc, instr_desc = make_descriptors(data_state.stage, k)
                            load_step(k)
                            sub_mma(0, slot0_col, a_desc, b_desc, instr_desc, k)
                            sub_mma(1, slot1_col, a_desc, b_desc, instr_desc, k)
                            load_empty.arrive(data_state.stage, cta_group=2, cta_mask=3)
                            data_state.advance()
                        acc_full.arrive(acc_state.stage, cta_group=2, cta_mask=3)
                        range_end(tile_range)
                        acc_state.advance()
                        txl.assign(slot_phases, txl.bitwise_xor(slot_phases, txl.bitwise_or(txl.shift_left(txl.int32(1), slot0), txl.shift_left(txl.int32(1), slot1))))
                        next_slot = txl.local_scalar("int32", init=3 - slot0 - slot1)
                        txl.assign(slot1, slot0)
                        txl.assign(slot0, next_slot)

                    for_each_tile(process_tile)

        with epi_role:
            epi_warp = warp - 4
            quad = warp % 4
            warpgroup = epi_warp // 4
            wg_tidx = (epi_warp % 4) * 32 + lane  # thread id within this warpgroup
            row = quad * 32 + lane
            acc_state = txl.PipelineState(2, 0)
            epi_slot0 = txl.local_scalar("int32", init=0)
            chunks_issued = txl.local_scalar("int32", init=0)  # store chunks issued by this warpgroup (buffer rotation)
            epi_slot1 = txl.local_scalar("int32", init=1)
            acc_words, packed = txl.alloc_local((32,), "uint32"), txl.alloc_local((COLS_PER_WG // 2,), "uint32")

            def process_tile(m0, col0, group_idx, has_rows, is_half, is_last=None):
                timed_wait("epi-wait-acc", acc_full, acc_state.stage, acc_state.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                drain_range = range_start("epi-drain")
                my_slot = txl.local_scalar("int32", init=txl.Select(warpgroup == 0, epi_slot0, epi_slot1))
                # Drain this warpgroup's accumulator columns into packed bf16 registers.
                for chunk in range(COLS_PER_WG // 32):
                    tmem_addr = txl.bitwise_or(txl.shift_left(to_u32(quad * 32), txl.uint32(16)), to_u32(my_slot * SUB_N + chunk * 32))
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[acc_words[j] for j in range(32)], tmem_addr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    for word in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(packed[chunk * 16 + word], txl.reinterpret(acc_words[2 * word + 1], ty="float32"), txl.reinterpret(acc_words[2 * word], ty="float32"))
                txl.ptx.tcgen05.fence__before_thread_sync()
                slot_free.arrive(my_slot, remote=0)
                range_end(drain_range)
                next_epi_slot = txl.local_scalar("int32", init=3 - epi_slot0 - epi_slot1)
                txl.assign(epi_slot1, epi_slot0)
                txl.assign(epi_slot0, next_epi_slot)
                store_range = range_start("epi-store")
                for chunk in range(COLS_PER_WG // 32):
                    if USE_LAST_TILE_BUFFERS:
                        # this CTA's last tile: no next tile's loads to compete with, so alternate two
                        # buffers (chunk -> (chunk + 1) % 2: the previous tile's final store used buffer 0)
                        buf_offset = (warpgroup * 2 + txl.if_then_else(is_last, (chunk + 1) % 2, 0)) * (BM * 16)
                        with txl.if_(wg_tidx == 0), txl.then_():
                            with txl.if_(is_last):
                                with txl.then_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(1)
                                with txl.else_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                    else:
                        buf_offset = (warpgroup * STORE_BUFFERS + (unsigned_mod(chunks_issued + chunk, STORE_BUFFERS) if STORE_BUFFERS > 1 else 0)) * (BM * 16)
                        with txl.if_(wg_tidx == 0), txl.then_():
                            txl.ptx.cp.async_.bulk.wait_group.read(STORE_BUFFERS - 1)
                    txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                    for unit in range(4):
                        word_idx = chunk * 16 + unit * 4
                        txl.ptx.st.shared.v4.u32(s_store.ptr_to([buf_offset + row * 16 + txl.bitwise_xor(unit, (row // 2) % 4) * 4]), packed[word_idx], packed[word_idx + 1], packed[word_idx + 2], packed[word_idx + 3])
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                    # A half tile's 128 columns all belong to warpgroup 0.
                    with txl.if_(txl.And(txl.And(wg_tidx == 0, has_rows), txl.Or(is_half == 0, warpgroup == 0))), txl.then_():
                        txl.ptx[STORE2D](txl.address_of(d_map), col0 + warpgroup * COLS_PER_WG + chunk * 32, m0, s_store.ptr_to([buf_offset]))
                    with txl.if_(wg_tidx == 0), txl.then_():
                        txl.ptx.cp.async_.bulk.commit_group()
                range_end(store_range)
                txl.assign(chunks_issued, chunks_issued + COLS_PER_WG // 32)
                acc_state.advance()

            for_each_tile(process_tile)
            with txl.if_(wg_tidx == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with idle_role:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped_w320r


def build_mgrouped_1cta(STAGES, USE_VECTOR_SCALES=False):
    """
    1-CTA m-grouped kernel: a CTA computes 128 rows (one 128-row block of one group) x 256
    columns.

    No pair padding: 128-row blocks tile the 128-padded groups exactly. One 256-column accumulator
    (+12 scale columns); two epilogue warpgroups each drain 128 columns to registers and release
    TMEM before packing and storing.
    """
    BN = 256
    A_BYTES, B_BYTES = BM * BK, BN * BK
    N_SF_CHUNKS = BN // 128
    SFA_COL = BN
    SFB_COL = SFA_COL + 4
    INSTR_DESC = encode_instr_descriptor_block_scaled_uint32(M=BM, N=BN, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=1)
    # Final-wave split: a last partial wave of <= half the CTAs runs its tiles as two 128-column halves.
    INSTR_DESC_HALF = encode_instr_descriptor_block_scaled_uint32(M=BM, N=BN // 2, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=1)

    def encode_tensor_maps(params):
        a_map, b_map, d_map = txl.stack_alloca("tensormap", 1), txl.stack_alloca("tensormap", 1), txl.stack_alloca("tensormap", 1)
        M, N, K, G = params["M"], params["N"], params["K"], params["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, params["a"].data, K, M, K, BK, BM, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(b_map, FP8, 3, params["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, BN, 1, 1, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(d_map, "bfloat16", 2, params["d"].data, N, M, N * 2, 32, BM, 1, 1, 0, 2, 2, 0)
        return a_map, b_map, d_map

    @txl.kernel()
    def mgrouped_1cta(a: txl.Tensor((txl.int64(), txl.int64()), FP8, data_alignment=1), sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32, data_alignment=1), b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8, data_alignment=1), sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32, data_alignment=1), d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, K: txl.int32, G: txl.int32):
        tensor_maps = encode_tensor_maps(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=384), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, d_map = tensor_maps
        cta_idx = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        tidx = txl.thread_id()

        roles = txl.specialize()
        tma_role = roles.role("tma", warps=[0])
        mma_role = roles.role("mma", warps=[1])
        sf_role = roles.role("sf", warps=[2])
        idle_role = roles.role("idle", warps=[3])
        epi_role = roles.role("epi", warps=list(range(4, 12)))

        smem = txl.smem_pool()
        s_a = smem.alloc((STAGES * A_BYTES,), "uint8", align=1024)
        s_b = smem.alloc((STAGES * B_BYTES,), "uint8", align=1024)
        s_store = smem.alloc((2 * BM * 16,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * BN,), "uint32", align=16)
        load_full = txl.TMABar(smem, STAGES)
        load_empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        acc_full = txl.TCGen05Bar(smem, 1)
        acc_empty = txl.MBarrier(smem, 1)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        load_full.init(1)
        load_empty.init(1)
        sf_full.init(32)
        acc_full.init(1)
        acc_empty.init(256)
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))
        with idle_role:
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](txl.address_of(tmem_slot[0]), txl.uint32(512))
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
        txl.ptx.bar.sync(txl.uint32(0))

        num_n_blocks = (N + (BN - 1)) // BN
        total = unsigned_div(M, BM) * num_n_blocks
        num_k_blocks = (K + (BK - 1)) // BK
        def for_each_tile(process_tile):
            """
            Persistent walk: ticket -> (128-row block, n block); group_idx follows the row block
            monotonically.
            """
            ticket = txl.local_scalar("int32", init=cta_idx)
            group_idx = txl.local_scalar("int32", init=0)
            group_start = txl.local_scalar("int32", init=0)  # first row of group group_idx
            group_end = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(group_end, offs.ptr_to([0]))
            tail_tiles = unsigned_mod(total, NUM_SMS)
            tail_tiles = txl.if_then_else(2 * tail_tiles <= NUM_SMS, tail_tiles, 0)
            full_tiles_end = total - tail_tiles  # tiles [full_tiles_end, total) run as two column halves each
            with txl.while_(ticket < full_tiles_end + 2 * tail_tiles):
                is_half = txl.local_scalar("int32", init=txl.if_then_else(ticket >= full_tiles_end, 1, 0))
                tile_idx = txl.local_scalar("int32", init=txl.if_then_else(is_half == 1, full_tiles_end + unsigned_div(ticket - full_tiles_end, 2), ticket))
                m0 = txl.local_scalar("int32", init=unsigned_div(tile_idx, num_n_blocks) * BM)
                col0 = txl.local_scalar("int32", init=unsigned_mod(tile_idx, num_n_blocks) * BN + txl.if_then_else(is_half == 1, unsigned_mod(ticket - full_tiles_end, 2) * (BN // 2), 0))
                # half-tile ids can map below this CTA's previous tile (problem under one wave): restart the walk
                with txl.if_(m0 < group_start), txl.then_():
                    txl.assign(group_idx, 0)
                    txl.assign(group_start, 0)
                    txl.ptx.ld.global_.s32(group_end, offs.ptr_to([0]))
                with txl.while_(m0 >= group_end):
                    txl.assign(group_idx, group_idx + 1)
                    txl.assign(group_start, group_end)
                    txl.ptx.ld.global_.s32(group_end, offs.ptr_to([group_idx]))
                process_tile(m0, col0, group_idx, is_half)
                txl.assign(ticket, ticket + NUM_SMS)

        def for_each_tile_no_groups(process_tile):
            """
            Same tile sequence without the group walk (for roles that only need the half-tile flag).
            """
            ticket = txl.local_scalar("int32", init=cta_idx)
            tail_tiles = unsigned_mod(total, NUM_SMS)
            tail_tiles = txl.if_then_else(2 * tail_tiles <= NUM_SMS, tail_tiles, 0)
            full_tiles_end = total - tail_tiles
            with txl.while_(ticket < full_tiles_end + 2 * tail_tiles):
                is_half = txl.local_scalar("int32", init=txl.if_then_else(ticket >= full_tiles_end, 1, 0))
                process_tile(0, 0, 0, is_half)
                txl.assign(ticket, ticket + NUM_SMS)

        with tma_role:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                data_state = txl.PipelineState(STAGES, 1)

                def process_tile(m0, col0, group_idx, is_half):
                    with txl.serial(num_k_blocks) as k:
                        timed_wait("tma-wait-empty", load_empty, data_state.stage, data_state.phase)
                        load_full.arrive(data_state.stage, A_BYTES + B_BYTES)
                        txl.ptx[LOAD2D](s_a.ptr_to([data_state.stage * A_BYTES]), txl.address_of(a_map), txl.cast(k * BK, "int32"), m0, load_full.ptr_to([data_state.stage]), txl.uint64(EVICT_NORMAL))
                        txl.ptx[LOAD3D](s_b.ptr_to([data_state.stage * B_BYTES]), txl.address_of(b_map), txl.cast(k * BK, "int32"), col0, group_idx, load_full.ptr_to([data_state.stage]), txl.uint64(EVICT_NORMAL))
                        data_state.advance()

                for_each_tile(process_tile)

        with sf_role:
            data_state = txl.PipelineState(STAGES, 1)
            sfa_words, sfb_words = txl.alloc_local((16,), "uint32"), txl.alloc_local((4 * N_SF_CHUNKS,), "uint32")
            scale_f32 = txl.local_scalar("float32")
            num_sfb_blocks = (N + 127) // 128

            def load_scale_group(m0, col0, group_idx, k_block):
                if USE_VECTOR_SCALES:
                    for i in range(4):
                        txl.ptx.ld.global_.nc.v4.b32(sfa_words[i * 4], sfa_words[i * 4 + 1], sfa_words[i * 4 + 2], sfa_words[i * 4 + 3], sfa.ptr_to([m0 + lane + 32 * i, k_block]))
                    for chunk in range(N_SF_CHUNKS):
                        sfb_block = txl.min(unsigned_div(col0, 128) + chunk, num_sfb_blocks - 1)
                        txl.ptx.ld.global_.nc.v4.b32(sfb_words[chunk * 4], sfb_words[chunk * 4 + 1], sfb_words[chunk * 4 + 2], sfb_words[chunk * 4 + 3], sfb.ptr_to([group_idx, sfb_block, k_block]))
                else:
                    for j in range(4):
                        scale_k = txl.min(k_block + j, num_k_blocks - 1)
                        for i in range(4):
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfa.ptr_to([m0 + lane + 32 * i, scale_k]))
                            txl.assign(sfa_words[i * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))
                        for chunk in range(N_SF_CHUNKS):
                            sfb_block = txl.min(unsigned_div(col0, 128) + chunk, num_sfb_blocks - 1)
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfb.ptr_to([group_idx, sfb_block, scale_k]))
                            txl.assign(sfb_words[chunk * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))

            def process_tile(m0, col0, group_idx, is_half):
                load_scale_group(m0, col0, group_idx, 0)
                with txl.serial(num_k_blocks) as k:
                    timed_wait("sf-wait-empty", load_empty, data_state.stage, data_state.phase)
                    # the previous tcgen05.cp of this stage (async proxy) completed: order our writes after it
                    txl.ptx.fence.proxy.async_.shared__cta()
                    with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                        txl.ptx.st.shared.v4.u32(s_sfa.ptr_to([data_state.stage * 128 + lane * 4]), pack_scales4(sfa_words, 0), pack_scales4(sfa_words, 4), pack_scales4(sfa_words, 8), pack_scales4(sfa_words, 12))
                        for chunk in range(N_SF_CHUNKS):
                            sfb_packed = txl.local_scalar("uint32", init=pack_scales4(sfb_words, chunk * 4))
                            txl.ptx.st.shared.v4.u32(s_sfb.ptr_to([data_state.stage * BN + chunk * 128 + lane * 4]), sfb_packed, sfb_packed, sfb_packed, sfb_packed)
                        with txl.if_(k + 4 < num_k_blocks), txl.then_():
                            load_scale_group(m0, col0, group_idx, k + 4)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    sf_full.arrive(data_state.stage)
                    data_state.advance()

            for_each_tile(process_tile)

        with mma_role:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                data_state, acc_state = txl.PipelineState(STAGES, 0), txl.PipelineState(1, 0)
                a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                sfa_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0])))
                sfb_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0])))

                def process_tile(m0, col0, group_idx, is_half):
                    timed_wait("mma-wait-acc", acc_empty, acc_state.stage, acc_state.phase ^ 1)
                    tile_range = range_start("mma-tile")
                    txl.ptx.tcgen05.fence__after_thread_sync()
                    with txl.serial(num_k_blocks) as k:
                        a_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, a_base + to_u32(data_state.stage * A_BYTES)))
                        b_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, b_base + to_u32(data_state.stage * B_BYTES)))
                        instr_desc = txl.local_scalar("uint32", init=with_sf_id(txl.if_then_else(is_half == 1, txl.uint32(INSTR_DESC_HALF), txl.uint32(INSTR_DESC)), unsigned_mod(k, 4)))
                        sfa_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfa_base + to_u32(data_state.stage * 512)))
                        sfb_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfb_base + to_u32(data_state.stage * (BN * 4))))
                        timed_wait("mma-wait-tma", load_full, data_state.stage, data_state.phase)
                        timed_wait("mma-wait-sf", sf_full, data_state.stage, data_state.phase)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                            txl.ptx[UTCCP_CHAIN](txl.uint32(SFA_COL), sfa_desc)
                            for chunk in range(N_SF_CHUNKS):
                                txl.ptx[UTCCP_CHAIN](txl.uint32(SFB_COL + 4 * chunk), offset_descriptor(sfb_desc, chunk * 512))
                        for k_step in range(BK // UMMA_K):
                            txl.ptx[MMA_CHAIN](txl.uint32(0), offset_descriptor(a_desc, k_step * UMMA_K), offset_descriptor(b_desc, k_step * UMMA_K), instr_desc, txl.uint32(SFA_COL), txl.uint32(SFB_COL), txl.Or(k > 0, k_step > 0) if k_step == 0 else txl.bool(True))
                        load_empty.arrive(data_state.stage)
                        data_state.advance()
                    acc_full.arrive(acc_state.stage)
                    range_end(tile_range)
                    acc_state.advance()

                for_each_tile_no_groups(process_tile)

        with epi_role:
            epi_warp = warp - 4
            quad = warp % 4
            warpgroup = epi_warp // 4
            wg_tidx, row = (epi_warp % 4) * 32 + lane, quad * 32 + lane
            acc_state = txl.PipelineState(1, 0)
            acc_words, packed = txl.alloc_local((128,), "uint32"), txl.alloc_local((64,), "uint32")

            def process_tile(m0, col0, group_idx, is_half):
                is_live = txl.Or(is_half == 0, warpgroup == 0)  # a half tile's 128 columns belong to warpgroup 0
                timed_wait("epi-wait-acc", acc_full, acc_state.stage, acc_state.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                with txl.if_(is_live), txl.then_():
                    for chunk in range(4):
                        tmem_addr = txl.bitwise_or(txl.shift_left(to_u32(quad * 32), txl.uint32(16)), to_u32(warpgroup * 128 + chunk * 32))
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[acc_words[chunk * 32 + j] for j in range(32)], tmem_addr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx.tcgen05.fence__before_thread_sync()
                acc_empty.arrive(acc_state.stage)
                with txl.if_(is_live), txl.then_():
                    for word in range(64):
                        txl.ptx.cvt.rn.bf16x2.f32(packed[word], txl.reinterpret(acc_words[2 * word + 1], ty="float32"), txl.reinterpret(acc_words[2 * word], ty="float32"))
                    buf_offset = warpgroup * (BM * 16)
                    for chunk in range(4):
                        with txl.if_(wg_tidx == 0), txl.then_():
                            txl.ptx.cp.async_.bulk.wait_group.read(0)
                        txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                        for unit in range(4):
                            word_idx = chunk * 16 + unit * 4
                            txl.ptx.st.shared.v4.u32(s_store.ptr_to([buf_offset + row * 16 + txl.bitwise_xor(unit, (row // 2) % 4) * 4]), packed[word_idx], packed[word_idx + 1], packed[word_idx + 2], packed[word_idx + 3])
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                        with txl.if_(wg_tidx == 0), txl.then_():
                            txl.ptx[STORE2D](txl.address_of(d_map), col0 + warpgroup * 128 + chunk * 32, m0, s_store.ptr_to([buf_offset]))
                            txl.ptx.cp.async_.bulk.commit_group()
                acc_state.advance()

            for_each_tile(process_tile)
            with txl.if_(wg_tidx == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.bar.sync(txl.uint32(0))
        with idle_role:
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped_1cta


def build_mgrouped_swap(STAGES, USE_VECTOR_SCALES=False, TAIL_STEPS=4):
    """
    Swap-AB 2-CTA m-grouped kernel: MMA M = 256 weight rows (128 per CTA), MMA N = tokens.

    A pair tile covers 256 tokens of one group, or 128 for a group's odd tail (MMA N = 128, half
    the work), times 256 output columns; CTA r stores columns [r*128, r*128+128) of the tile.
    """
    assert 1 <= TAIL_STEPS <= BK // UMMA_K
    W_BYTES = 128 * BK  # weight rows per CTA
    T_BYTES = 128 * BK  # token rows per CTA (full tile; a tail tile uses half)
    SFA_COL = 256
    SFB_COL = SFA_COL + 4
    EPI_WARPS = 8
    INSTR_DESC_FULL = encode_instr_descriptor_block_scaled_uint32(M=256, N=256, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2)
    INSTR_DESC_HALF = encode_instr_descriptor_block_scaled_uint32(M=256, N=128, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=False, trans_b=False, cta_group=2)
    MMA_PAIR = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP_PAIR = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LOAD2D_PAIR = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
    LOAD3D_PAIR = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def encode_tensor_maps(params):
        a_map = txl.stack_alloca("tensormap", 1)
        a64_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        d_map = txl.stack_alloca("tensormap", 1)
        M, N, K, G = params["M"], params["N"], params["K"], params["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, params["a"].data, K, M, K, BK, 128, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(a64_map, FP8, 2, params["a"].data, K, M, K, BK, 64, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(b_map, FP8, 3, params["b"].data, K, N, G, K, txl.cast(N, "int64") * K, BK, 128, 1, 1, 1, 1, 0, 3, L2_PROMOTION, 0)
        txl.cu_tensor_map_encode_tiled(d_map, "bfloat16", 2, params["d"].data, N, M, N * 2, 128, 32, 1, 1, 0, 0, 2, 0)
        return a_map, a64_map, b_map, d_map

    @txl.kernel()
    def mgrouped_swap(a: txl.Tensor((txl.int64(), txl.int64()), FP8, data_alignment=1), sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32, data_alignment=1), b: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), FP8, data_alignment=1), sfb: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32, data_alignment=1), d: txl.Tensor((txl.int64(), txl.int64()), txl.bf16, data_alignment=1), offs: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, K: txl.int32, G: txl.int32):
        tensor_maps = encode_tensor_maps(locals())
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=(4 + EPI_WARPS) * 32, cluster=[2]), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, a64_map, b_map, d_map = tensor_maps
        rank = txl.cuda.cluster_cta_id("x")
        cluster_idx = txl.cta_id() // 2
        warp = txl.warp_id()
        lane = txl.lane_id()
        tidx = txl.thread_id()

        roles = txl.specialize()
        tma_role = roles.role("tma", warps=[0])
        mma_role = roles.role("mma", warps=[1])
        sf_role = roles.role("sf", warps=[2])
        idle_role = roles.role("idle", warps=[3])
        epi_role = roles.role("epi", warps=list(range(4, 4 + EPI_WARPS)))
        # Issue the first 32 group ends' loads now; their latency overlaps the setup below.
        chunk0_group_end, chunk0_group_start = txl.local_scalar("int32"), txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(chunk0_group_end, offs.ptr_to([txl.min(lane, G - 1)]))
        txl.ptx.ld.global_.s32(chunk0_group_start, offs.ptr_to([txl.max(txl.min(lane, G - 1) - 1, 0)]))

        smem = txl.smem_pool()
        s_w, s_t = smem.alloc((STAGES * W_BYTES,), "uint8", align=1024), smem.alloc((STAGES * T_BYTES,), "uint8", align=1024)
        s_store = smem.alloc((2 * 32 * 64,), "uint32", align=1024)  # per WG: 32 tokens x 128 cols bf16
        s_sfa, s_sfb = smem.alloc((STAGES * 128,), "uint32", align=16), smem.alloc((STAGES * 256,), "uint32", align=16)
        load_full = txl.TMABar(smem, STAGES)
        load_empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        acc_full = txl.TCGen05Bar(smem, 1)
        acc_empty = txl.MBarrier(smem, 1)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        load_full.init(1)
        load_empty.init(1)
        sf_full.init(64)
        acc_full.init(1)
        acc_empty.init(2 * 32 * EPI_WARPS)
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with idle_role:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](txl.address_of(tmem_slot[0]), txl.uint32(512))
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        leader_load_full = load_full.remote_view(0)
        leader_sf_full = sf_full.remote_view(0)

        num_n_blocks, num_k_blocks = (N + 255) // 256, (K + (BK - 1)) // BK
        # Lanes take 32 groups at a time: pairs per group (ceil(rows / 256)), their running sum
        # (shfl.up scan + carry) and, by warp-wide reductions, the total and the group holding this
        # cluster's first pair, so for_each_tile() starts there instead of walking from group 0.
        walk_range = range_start("walk")
        carry = txl.local_scalar("uint32", init=txl.uint32(0))
        lane_group_end = txl.local_scalar("int32")
        lane_group_start = txl.local_scalar("int32")
        first_pair = txl.local_scalar("int32", init=unsigned_div(cluster_idx, num_n_blocks))
        walk_start = [txl.local_scalar("uint32", init=txl.uint32(0)) for _ in range(6)]  # has_first_pair, group_idx, group_start, group_end, pair_start, pair_end
        peer_value = txl.local_scalar("uint32")

        def warp_sum(value):
            # one redux.sync per value: the six walk reductions are independent (no 5-deep shuffle chains)
            txl.ptx.redux_sync.add.u32(value, value, txl.uint32(0xFFFFFFFF))

        with txl.serial((G + 31) // 32) as chunk_idx:
            lane_group_idx = txl.local_scalar("int32", init=chunk_idx * 32 + lane)
            with txl.if_(chunk_idx == 0):
                with txl.then_():
                    txl.assign(lane_group_end, chunk0_group_end)
                    txl.assign(lane_group_start, chunk0_group_start)
                with txl.else_():
                    txl.ptx.ld.global_.s32(lane_group_end, offs.ptr_to([txl.min(lane_group_idx, G - 1)]))
                    txl.ptx.ld.global_.s32(lane_group_start, offs.ptr_to([txl.max(txl.min(lane_group_idx, G - 1) - 1, 0)]))
            with txl.if_(lane_group_idx == 0), txl.then_():
                txl.assign(lane_group_start, 0)
            lane_pairs = txl.local_scalar("uint32", init=txl.if_then_else(lane_group_idx < G, to_u32(unsigned_div(lane_group_end - lane_group_start + 255, 256)), txl.uint32(0)))
            pair_scan = txl.local_scalar("uint32", init=lane_pairs)
            for scan_offset in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer_value, pair_scan, txl.uint32(scan_offset), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                with txl.if_(lane >= scan_offset), txl.then_():
                    txl.assign(pair_scan, pair_scan + peer_value)
            lane_pair_end = txl.local_scalar("int32", init=txl.cast(carry + pair_scan, "int32"))
            lane_pair_start = txl.local_scalar("int32", init=lane_pair_end - txl.cast(lane_pairs, "int32"))
            has_first_pair = txl.local_scalar("uint32", init=txl.if_then_else(txl.And(lane_pair_start <= first_pair, first_pair < lane_pair_end), txl.uint32(1), txl.uint32(0)))
            walk_terms = [has_first_pair, has_first_pair * to_u32(lane_group_idx), has_first_pair * to_u32(lane_group_start), has_first_pair * to_u32(lane_group_end), has_first_pair * to_u32(lane_pair_start), has_first_pair * to_u32(lane_pair_end)]
            for j in range(6):
                value = txl.local_scalar("uint32", init=walk_terms[j])
                warp_sum(value)
                txl.assign(walk_start[j], walk_start[j] + value)
            chunk_pairs = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.idx.b32(chunk_pairs, pair_scan, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(carry, carry + chunk_pairs)
        total = txl.local_scalar("int32", init=txl.cast(carry, "int32"))
        has_start = walk_start[0] != txl.uint32(0)
        range_end(walk_range)
        # the group walk (registers + global loads only) overlaps the cluster barrier latency
        txl.ptx.barrier.cluster.wait.acquire.aligned()

        def for_each_tile(process_tile):
            """
            Pair walk: ticket -> (token tile pair_idx, n_block); a group has ceil(rows/256) token tiles.
            """
            ticket = txl.local_scalar("int32", init=cluster_idx)
            group_idx = txl.local_scalar("int32", init=txl.if_then_else(has_start, txl.cast(walk_start[1], "int32"), -1))
            group_start = txl.local_scalar("int32", init=txl.cast(walk_start[2], "int32"))
            group_end = txl.local_scalar("int32", init=txl.cast(walk_start[3], "int32"))
            pair_start = txl.local_scalar("int32", init=txl.cast(walk_start[4], "int32"))
            pair_end = txl.local_scalar("int32", init=txl.cast(walk_start[5], "int32"))
            with txl.while_(ticket < total * num_n_blocks):
                pair_idx = txl.local_scalar("int32", init=unsigned_div(ticket, num_n_blocks))
                n_block = txl.local_scalar("int32", init=unsigned_mod(ticket, num_n_blocks))
                with txl.while_(pair_idx >= pair_end):
                    txl.assign(group_idx, group_idx + 1)
                    txl.assign(group_start, group_end)
                    txl.ptx.ld.global_.s32(group_end, offs.ptr_to([group_idx]))
                    txl.assign(pair_start, pair_end)
                    txl.assign(pair_end, pair_end + unsigned_div(group_end - group_start + 255, 256))
                m0 = txl.local_scalar("int32", init=group_start + (pair_idx - pair_start) * 256)
                process_tile(m0, n_block, group_idx, group_end - m0 >= 256)
                txl.assign(ticket, ticket + NUM_SMS // 2)

        with tma_role:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(a64_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                data_state = txl.PipelineState(STAGES, 1)

                def process_tile(m0, n_block, group_idx, is_full_tile):
                    tx_bytes = txl.local_scalar("int32", init=txl.if_then_else(is_full_tile, 2 * (W_BYTES + T_BYTES), 2 * (W_BYTES + T_BYTES // 2)))
                    with txl.serial(num_k_blocks) as k:
                        timed_wait("tma-wait-empty", load_empty, data_state.stage, data_state.phase)
                        full_addr = txl.cuda.cvta_generic_to_shared(leader_load_full.ptr_to([data_state.stage]))
                        with txl.if_(rank == 0), txl.then_():
                            leader_load_full.arrive(data_state.stage, tx_bytes)
                        txl.ptx[LOAD3D_PAIR](s_w.ptr_to([data_state.stage * W_BYTES]), txl.address_of(b_map), txl.cast(k * BK, "int32"), n_block * 256 + rank * 128, group_idx, full_addr)
                        with txl.if_(is_full_tile):
                            with txl.then_():
                                txl.ptx[LOAD2D_PAIR](s_t.ptr_to([data_state.stage * T_BYTES]), txl.address_of(a_map), txl.cast(k * BK, "int32"), m0 + rank * 128, full_addr)
                            with txl.else_():
                                txl.ptx[LOAD2D_PAIR](s_t.ptr_to([data_state.stage * T_BYTES]), txl.address_of(a64_map), txl.cast(k * BK, "int32"), m0 + rank * 64, full_addr)
                        data_state.advance()

                for_each_tile(process_tile)

        with sf_role:
            data_state = txl.PipelineState(STAGES, 1)
            sfa_words, sfb_words = txl.alloc_local((4,), "uint32"), txl.alloc_local((32,), "uint32")
            scale_f32 = txl.local_scalar("float32")
            num_sfb_blocks = (N + 127) // 128

            def load_scale_group(m0, n_block, group_idx, k_block):
                sfb_block = txl.min(n_block * 2 + rank, num_sfb_blocks - 1)
                if USE_VECTOR_SCALES:
                    txl.ptx.ld.global_.nc.v4.b32(sfa_words[0], sfa_words[1], sfa_words[2], sfa_words[3], sfb.ptr_to([group_idx, sfb_block, k_block]))
                    for chunk in range(2):
                        for i in range(4):
                            row = txl.min(m0 + chunk * 128 + lane + 32 * i, M - 1)
                            word_idx = chunk * 16 + i * 4
                            txl.ptx.ld.global_.nc.v4.b32(sfb_words[word_idx], sfb_words[word_idx + 1], sfb_words[word_idx + 2], sfb_words[word_idx + 3], sfa.ptr_to([row, k_block]))
                else:
                    for j in range(4):
                        scale_k = txl.min(k_block + j, num_k_blocks - 1)
                        txl.ptx.ld.global_.nc.f32(scale_f32, sfb.ptr_to([group_idx, sfb_block, scale_k]))
                        txl.assign(sfa_words[j], txl.reinterpret(scale_f32, ty="uint32"))
                        for chunk in range(2):
                            for i in range(4):
                                txl.ptx.ld.global_.nc.f32(scale_f32, sfa.ptr_to([txl.min(m0 + chunk * 128 + lane + 32 * i, M - 1), scale_k]))
                                txl.assign(sfb_words[chunk * 16 + i * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))

            def process_tile(m0, n_block, group_idx, is_full_tile):
                load_scale_group(m0, n_block, group_idx, 0)
                with txl.serial(num_k_blocks) as k:
                    timed_wait("sf-wait-empty", load_empty, data_state.stage, data_state.phase)
                    # The previous tcgen05.cp of this stage (async proxy) completed; order it before our writes.
                    txl.ptx.fence.proxy.async_.shared__cta()
                    with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                        sfa_packed = txl.local_scalar("uint32", init=pack_scales4(sfa_words, 0))
                        txl.ptx.st.shared.v4.u32(s_sfa.ptr_to([data_state.stage * 128 + lane * 4]), sfa_packed, sfa_packed, sfa_packed, sfa_packed)
                        for chunk in range(2):
                            txl.ptx.st.shared.v4.u32(s_sfb.ptr_to([data_state.stage * 256 + chunk * 128 + lane * 4]), pack_scales4(sfb_words, chunk * 16), pack_scales4(sfb_words, chunk * 16 + 4), pack_scales4(sfb_words, chunk * 16 + 8), pack_scales4(sfb_words, chunk * 16 + 12))
                        with txl.if_(k + 4 < num_k_blocks), txl.then_():
                            load_scale_group(m0, n_block, group_idx, k + 4)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    leader_sf_full.arrive(data_state.stage)
                    data_state.advance()

            for_each_tile(process_tile)

        with mma_role:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    data_state, acc_state = txl.PipelineState(STAGES, 0), txl.PipelineState(1, 0)
                    w_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_w.ptr_to([0])))
                    t_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_t.ptr_to([0])))
                    sfa_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0])))
                    sfb_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0])))

                    def process_tile(m0, n_block, group_idx, is_full_tile):
                        tile_instr_desc = txl.local_scalar("uint32", init=txl.if_then_else(is_full_tile, txl.uint32(INSTR_DESC_FULL), txl.uint32(INSTR_DESC_HALF)))
                        timed_wait("mma-wait-acc", acc_empty, acc_state.stage, acc_state.phase ^ 1)
                        tile_range = range_start("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        def issue_stage(k, mma_steps):
                            a_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, w_base + to_u32(data_state.stage * W_BYTES)))
                            b_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, t_base + to_u32(data_state.stage * T_BYTES)))
                            instr_desc = txl.local_scalar("uint32", init=with_sf_id(tile_instr_desc, unsigned_mod(k, 4)))
                            sfa_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfa_base + to_u32(data_state.stage * 512)))
                            sfb_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfb_base + to_u32(data_state.stage * 1024)))
                            timed_wait("mma-wait-tma", load_full, data_state.stage, data_state.phase)
                            timed_wait("mma-wait-sf", sf_full, data_state.stage, data_state.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                                txl.ptx[UTCCP_PAIR](txl.uint32(SFA_COL), sfa_desc)
                                for chunk in range(2):
                                    txl.ptx[UTCCP_PAIR](txl.uint32(SFB_COL + 4 * chunk), offset_descriptor(sfb_desc, chunk * 512))
                            for k_step in range(mma_steps):
                                txl.ptx[MMA_PAIR](txl.uint32(0), offset_descriptor(a_desc, k_step * UMMA_K), offset_descriptor(b_desc, k_step * UMMA_K), instr_desc, txl.uint32(SFA_COL), txl.uint32(SFB_COL), txl.Or(k > 0, k_step > 0) if k_step == 0 else txl.bool(True))
                            load_empty.arrive(data_state.stage, cta_group=2, cta_mask=3)
                            data_state.advance()
                        if TAIL_STEPS < BK // UMMA_K:
                            # Specialize by K: full stages have no tail predicates, and the peeled last
                            # stage omits the zero-filled 32-wide MMA steps.
                            with txl.serial(num_k_blocks - 1) as k:
                                issue_stage(k, BK // UMMA_K)
                            issue_stage(num_k_blocks - 1, TAIL_STEPS)
                        else:
                            with txl.serial(num_k_blocks) as k:
                                issue_stage(k, BK // UMMA_K)
                        acc_full.arrive(acc_state.stage, cta_group=2, cta_mask=3)
                        range_end(tile_range)
                        acc_state.advance()

                    for_each_tile(process_tile)

        with epi_role:
            epi_warp = warp - 4
            quad = warp % 4
            warpgroup = epi_warp // 4
            wg_tidx = (epi_warp % 4) * 32 + lane
            lane_parity = lane % 2
            col = quad * 32 + lane - lane_parity  # even output column of this lane's pair
            acc_state = txl.PipelineState(1, 0)
            acc_words, packed = txl.alloc_local((128,), "uint32"), txl.alloc_local((16,), "uint32")
            peer_value = txl.local_scalar("uint32")
            buf_offset = warpgroup * (32 * 64)

            def process_tile(m0, n_block, group_idx, is_full_tile):
                is_live = txl.Or(is_full_tile, warpgroup == 0)
                timed_wait("epi-wait-acc", acc_full, acc_state.stage, acc_state.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                with txl.if_(is_live), txl.then_():
                    for chunk in range(4):
                        tmem_addr = txl.bitwise_or(txl.shift_left(to_u32(quad * 32), txl.uint32(16)), to_u32(warpgroup * 128 + chunk * 32))
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[acc_words[chunk * 32 + j] for j in range(32)], tmem_addr)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                # The raw accumulator is in registers: release TMEM before packing and storing.
                txl.ptx.tcgen05.fence__before_thread_sync()
                acc_empty.arrive(acc_state.stage, remote=0)
                for chunk in range(4):
                    with txl.if_(is_live), txl.then_():
                        # Lane pairs swap halves so each lane owns two adjacent columns of one token.
                        for word in range(16):
                            txl.ptx.shfl_sync.bfly.b32(peer_value, txl.if_then_else(lane_parity == 0, acc_words[chunk * 32 + 2 * word + 1], acc_words[chunk * 32 + 2 * word]), txl.uint32(1), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                            lo = txl.if_then_else(lane_parity == 0, acc_words[chunk * 32 + 2 * word], peer_value)
                            hi = txl.if_then_else(lane_parity == 0, peer_value, acc_words[chunk * 32 + 2 * word + 1])
                            txl.ptx.cvt.rn.bf16x2.f32(packed[word], txl.reinterpret(hi, ty="float32"), txl.reinterpret(lo, ty="float32"))
                        with txl.if_(wg_tidx == 0), txl.then_():
                            txl.ptx.cp.async_.bulk.wait_group.read(0)
                        txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                        for word in range(16):
                            txl.ptx.st.shared.u32(s_store.ptr_to([buf_offset + (2 * word + lane_parity) * 64 + col // 2]), packed[word])
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                        with txl.if_(wg_tidx == 0), txl.then_():
                            txl.ptx[STORE2D](txl.address_of(d_map), n_block * 256 + rank * 128, m0 + warpgroup * 128 + chunk * 32, s_store.ptr_to([buf_offset]))
                            txl.ptx.cp.async_.bulk.commit_group()
                acc_state.advance()

            for_each_tile(process_tile)
            with txl.if_(wg_tidx == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with idle_role:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return mgrouped_swap


# k-grouped: d (G, N, K) fp32 += A_g^T B_g; A (rows, N), B (rows, K), both MN-major operands.


SCHED_DEPTH = 3  # tile ids the scheduler may claim ahead of the epilogue
SCHED_CONSUMERS = 21  # per tile id: CTA0 tma + sf + mma + 8 epi warps, CTA1 tma + sf + 8 epi warps


def build_kgrouped(STAGES, STORE_BUFFERS=1, USE_REVERSE_WALK=False, BN=256, USE_D_PREFETCH=False, USE_EPILOGUE_D_PREFETCH=False, REDUCE_COLS=32, USE_SHARED_STORE_BOX=False):
    """
    2-CTA k-grouped kernel: a CTA pair accumulates a 256 (N) x 256 (K) fp32 tile of d[g].

    Tiles are claimed dynamically: the leader CTA's idle warp takes tile ids from a global atomic
    counter (ctr[0]) and hands them to every role of both CTAs through a shared-memory ring. The
    last pair to drain the counter resets ctr to zero for the next launch.
    """
    A_BYTES = BM * BK
    B_COLS_PER_CTA = BN // 2  # B columns held by each CTA of the pair
    B_HALF = B_COLS_PER_CTA * BK
    B_SWIZZLE = 3 if B_COLS_PER_CTA == 128 else 2
    B_DESC_BASE = encode_smem_descriptor_base_uint64(0, 8 * B_COLS_PER_CTA // 16, B_SWIZZLE)
    B_K_STEP = UMMA_K * B_COLS_PER_CTA
    N_SF_CHUNKS = (BN + 127) // 128
    ACC_STAGES = 1 if BN == 256 else 2
    SFA_COL = ACC_STAGES * BN
    SFB_COL = SFA_COL + 4
    COLS_PER_WG = BN // 2  # accumulator columns per epilogue warpgroup
    REDUCE_SWIZZLE = 3 if REDUCE_COLS == 32 else 2  # reduce box: REDUCE_COLS fp32 columns = 128 B (SW128) or 64 B (SW64) rows
    if USE_SHARED_STORE_BOX:
        # One 16 KiB SW128 staging box is time-shared by the two epilogue warpgroups. This keeps
        # the six-stage operand ring while halving the number of reduce-add TMA instructions.
        assert REDUCE_COLS == 32 and STORE_BUFFERS == 1 and BN == 256
    INSTR_DESC = encode_instr_descriptor_block_scaled_uint32(M=2 * BM, N=BN, K=UMMA_K, d_dtype="float32", a_dtype=FP8, b_dtype=FP8, sf_dtype="float8_e8m0fnu", trans_a=True, trans_b=True, cta_group=2)
    K_STEP = UMMA_K * 128  # bytes between UMMA K steps of an MN-major SW128 tile
    MMA_PAIR = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
    UTCCP_PAIR = "tcgen05.cp.cta_group::2.32x128b.warpx4"
    LOAD2D_PAIR = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"

    def encode_tensor_maps(params):
        a_map, b_map, d_map = txl.stack_alloca("tensormap", 1), txl.stack_alloca("tensormap", 1), txl.stack_alloca("tensormap", 1)
        M, N, K, G = params["M"], params["N"], params["K"], params["G"]
        txl.cu_tensor_map_encode_tiled(a_map, FP8, 2, params["a"].data, N, M, N, 128, BK, 1, 1, 0, 3, 2, 0)
        txl.cu_tensor_map_encode_tiled(b_map, FP8, 2, params["b"].data, K, M, K, B_COLS_PER_CTA, BK, 1, 1, 0, B_SWIZZLE, 2, 0)
        txl.cu_tensor_map_encode_tiled(d_map, "float32", 3, params["d"].data, K, N, G, K * 4, txl.cast(N, "int64") * K * 4, REDUCE_COLS, BM, 1, 1, 1, 1, 0, REDUCE_SWIZZLE, 2, 0)
        return a_map, b_map, d_map

    @txl.kernel()
    def kgrouped(a: txl.Tensor((txl.int64(), txl.int64()), FP8, data_alignment=1), sfa: txl.Tensor((txl.int64(), txl.int64()), txl.f32, data_alignment=1), b: txl.Tensor((txl.int64(), txl.int64()), FP8, data_alignment=1), sfb: txl.Tensor((txl.int64(), txl.int64()), txl.f32, data_alignment=1), d: txl.Tensor((txl.int64(), txl.int64(), txl.int64()), txl.f32, data_alignment=1), ks: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), ctr: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), M: txl.int32, N: txl.int32, K: txl.int32, G: txl.int32):
        tensor_maps = encode_tensor_maps(locals())
        # Programmatic dependent launch: the prologue (barrier init, TMEM allocation, cluster sync)
        # overlaps the tail of the preceding kernel; all global accesses follow griddepcontrol.wait.
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=NUM_SMS, block=384, cluster=[2], programmatic_stream_serialization=True), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        a_map, b_map, d_map = tensor_maps
        rank = txl.cuda.cluster_cta_id("x")
        warp = txl.warp_id()
        lane = txl.lane_id()
        tidx = txl.thread_id()

        roles = txl.specialize()
        tma_role = roles.role("tma", warps=[0])
        mma_role = roles.role("mma", warps=[1])
        sf_role = roles.role("sf", warps=[2])
        idle_role = roles.role("idle", warps=[3])
        epi_role = roles.role("epi", warps=list(range(4, 12)))

        smem = txl.smem_pool()
        s_a = smem.alloc((STAGES * A_BYTES,), "uint8", align=1024)
        s_b = smem.alloc((STAGES * B_HALF,), "uint8", align=1024)
        s_store = smem.alloc(((1 if USE_SHARED_STORE_BOX else 2) * STORE_BUFFERS * BM * REDUCE_COLS,), "uint32", align=1024)
        s_sfa = smem.alloc((STAGES * 128,), "uint32", align=16)
        s_sfb = smem.alloc((STAGES * BN,), "uint32", align=16)
        load_full = txl.TMABar(smem, STAGES)
        load_empty = txl.TCGen05Bar(smem, STAGES)
        sf_full = txl.MBarrier(smem, STAGES)
        acc_full = txl.TCGen05Bar(smem, ACC_STAGES)
        acc_empty = txl.MBarrier(smem, ACC_STAGES)
        tmem_slot = smem.alloc((1,), "uint32", align=4)
        load_full.init(1)
        load_empty.init(1)
        sf_full.init(64)
        acc_full.init(1)
        acc_empty.init(512)
        s_tile = smem.alloc((SCHED_DEPTH,), "int32", align=16)
        sched_full, sched_empty = txl.MBarrier(smem, SCHED_DEPTH), txl.MBarrier(smem, SCHED_DEPTH)
        sched_full.init(1)
        sched_empty.init(SCHED_CONSUMERS)
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))  # CTA-wide order for barrier init (also what compute-sanitizer models)
        with idle_role:
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](txl.address_of(tmem_slot[0]), txl.uint32(512))
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        # PDL: the prologue above overlaps the preceding grid; every global access (operands, scales,
        # ks, the tile counter, d) follows its completion.
        txl.ptx.griddepcontrol.wait()
        leader_load_full = load_full.remote_view(0)
        leader_sf_full = sf_full.remote_view(0)
        leader_sched_empty = sched_empty.remote_view(0)

        num_m_tiles, num_n_tiles = (N + 255) // 256, (K + (BN - 1)) // BN
        tiles_per_group = num_m_tiles * num_n_tiles
        total = G * tiles_per_group

        def for_each_tile(process_tile, is_one_lane=True):
            """
            Pair walk over (group, 256-row tile of d, 256-col tile of d); empty groups skipped.

            Tile ids arrive in increasing order from the scheduler ring; -1 ends the walk.
            is_one_lane: called by a single elected thread (else by a full warp, lane 0 releases).
            """
            sched_state = txl.PipelineState(SCHED_DEPTH, 0)
            ticket = txl.local_scalar("int32", init=0)

            def next_tile():
                txl.cuda.mbarrier_wait_acquire_cluster(sched_full.ptr_to([sched_state.stage]), sched_state.phase)
                txl.ptx.ld.shared.s32(ticket, s_tile.ptr_to([sched_state.stage]))
                if not is_one_lane:
                    txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
                with txl.if_(txl.Or(txl.bool(is_one_lane), lane == 0)), txl.then_():
                    leader_sched_empty.arrive(sched_state.stage)
                sched_state.advance()

            next_tile()
            group_rows = txl.local_scalar("int32")
            if USE_REVERSE_WALK:
                # Walk groups from the last one: d's tail is the most recently written memory.
                group_idx, row0 = txl.local_scalar("int32", init=G - 1), txl.local_scalar("int32", init=M)
                txl.ptx.ld.global_.s32(group_rows, ks.ptr_to([G - 1]))
                txl.assign(row0, row0 - group_rows)
            else:
                group_idx, row0 = txl.local_scalar("int32", init=0), txl.local_scalar("int32", init=0)
                txl.ptx.ld.global_.s32(group_rows, ks.ptr_to([0]))
            with txl.while_(ticket >= 0):
                if USE_REVERSE_WALK:
                    tile_group = txl.local_scalar("int32", init=G - 1 - unsigned_div(ticket, tiles_per_group))
                    with txl.while_(group_idx > tile_group):
                        txl.assign(group_idx, group_idx - 1)
                        txl.ptx.ld.global_.s32(group_rows, ks.ptr_to([group_idx]))
                        txl.assign(row0, row0 - group_rows)
                    group_tile = txl.local_scalar("int32", init=tiles_per_group - 1 - (ticket - (G - 1 - tile_group) * tiles_per_group))
                else:
                    tile_group = txl.local_scalar("int32", init=unsigned_div(ticket, tiles_per_group))
                    with txl.while_(group_idx < tile_group):
                        txl.assign(row0, row0 + group_rows)
                        txl.assign(group_idx, group_idx + 1)
                        txl.ptx.ld.global_.s32(group_rows, ks.ptr_to([group_idx]))
                    group_tile = txl.local_scalar("int32", init=ticket - tile_group * tiles_per_group)
                m0 = txl.local_scalar("int32", init=unsigned_div(group_tile, num_n_tiles) * 256 + rank * BM)
                n0 = txl.local_scalar("int32", init=unsigned_mod(group_tile, num_n_tiles) * BN)
                with txl.if_(group_rows > 0), txl.then_():
                    process_tile(m0, n0, group_idx, row0, unsigned_div(group_rows, BK))
                next_tile()

        with tma_role:
            with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))
                data_state = txl.PipelineState(STAGES, 1)

                txl.ptx.prefetch.tensormap(txl.address_of(d_map))

                def process_tile(m0, n0, group_idx, row0, num_k_blocks):
                    # USE_D_PREFETCH: pull this CTA's d slice into L2 ahead of the epilogue's reduce-add. A reduce that
                    # misses L2 waits on its own DRAM read, exposed when tiles are short (small groups).
                    if USE_D_PREFETCH:
                        for chunk in range(BN // REDUCE_COLS):
                            txl.ptx[PREFETCH3D](txl.address_of(d_map), n0 + chunk * REDUCE_COLS, m0, group_idx, uint64_const(EVICT_LAST))
                    with txl.serial(num_k_blocks) as k:
                        timed_wait("tma-wait-empty", load_empty, data_state.stage, data_state.phase)
                        full_addr = txl.cuda.cvta_generic_to_shared(leader_load_full.ptr_to([data_state.stage]))
                        with txl.if_(rank == 0), txl.then_():
                            leader_load_full.arrive(data_state.stage, 2 * (A_BYTES + B_HALF))
                        k_row = row0 + txl.cast(k * BK, "int32")
                        txl.ptx[LOAD2D_PAIR](s_a.ptr_to([data_state.stage * A_BYTES]), txl.address_of(a_map), m0, k_row, full_addr)
                        txl.ptx[LOAD2D_PAIR](s_b.ptr_to([data_state.stage * B_HALF]), txl.address_of(b_map), n0 + rank * (BN // 2), k_row, full_addr)
                        data_state.advance()

                for_each_tile(process_tile)

        with sf_role:
            data_state = txl.PipelineState(STAGES, 1)
            sfa_words, sfb_words = txl.alloc_local((16,), "uint32"), txl.alloc_local((16 * N_SF_CHUNKS,), "uint32")
            scale_f32 = txl.local_scalar("float32")

            def load_scale_group(m0, n0, row0, num_k_blocks, k_block):
                for j in range(4):
                    scale_k = unsigned_div(row0, BK) + txl.min(k_block + j, num_k_blocks - 1)
                    for i in range(4):
                        txl.ptx.ld.global_.nc.f32(scale_f32, sfa.ptr_to([scale_k, txl.min(m0 + lane + 32 * i, N - 1)]))
                        txl.assign(sfa_words[i * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))
                        for chunk in range(N_SF_CHUNKS):
                            txl.ptx.ld.global_.nc.f32(scale_f32, sfb.ptr_to([scale_k, txl.min(n0 + chunk * 128 + lane + 32 * i, K - 1)]))
                            txl.assign(sfb_words[chunk * 16 + i * 4 + j], txl.reinterpret(scale_f32, ty="uint32"))

            def process_tile(m0, n0, group_idx, row0, num_k_blocks):
                load_scale_group(m0, n0, row0, num_k_blocks, 0)
                with txl.serial(num_k_blocks) as k:
                    timed_wait("sf-wait-empty", load_empty, data_state.stage, data_state.phase)
                    # The previous tcgen05.cp of this stage (async proxy) completed; order it before our writes.
                    txl.ptx.fence.proxy.async_.shared__cta()
                    with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                        txl.ptx.st.shared.v4.u32(s_sfa.ptr_to([data_state.stage * 128 + lane * 4]), pack_scales4(sfa_words, 0), pack_scales4(sfa_words, 4), pack_scales4(sfa_words, 8), pack_scales4(sfa_words, 12))
                        for chunk in range(N_SF_CHUNKS):
                            txl.ptx.st.shared.v4.u32(s_sfb.ptr_to([data_state.stage * BN + chunk * 128 + lane * 4]), pack_scales4(sfb_words, chunk * 16), pack_scales4(sfb_words, chunk * 16 + 4), pack_scales4(sfb_words, chunk * 16 + 8), pack_scales4(sfb_words, chunk * 16 + 12))
                        with txl.if_(k + 4 < num_k_blocks), txl.then_():
                            load_scale_group(m0, n0, row0, num_k_blocks, k + 4)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    leader_sf_full.arrive(data_state.stage)
                    data_state.advance()

            for_each_tile(process_tile, is_one_lane=False)

        with idle_role:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    sched_state = txl.PipelineState(SCHED_DEPTH, 1)
                    claimed = txl.local_scalar("int32", init=0)
                    peer_tile_addr = txl.local_scalar("uint32")
                    peer_full_addr = txl.local_scalar("uint32")
                    with txl.while_(claimed >= 0):
                        txl.cuda.mbarrier_wait(sched_empty.ptr_to([sched_state.stage]), sched_state.phase)
                        txl.ptx.atom.global_.add.s32(claimed, ctr.ptr_to([0]), txl.int32(1))
                        with txl.if_(claimed >= total), txl.then_():
                            txl.assign(claimed, -1)
                        txl.ptx.st.shared.s32(s_tile.ptr_to([sched_state.stage]), claimed)
                        txl.ptx["mapa.shared::cluster.u32"](peer_tile_addr, txl.cuda.cvta_generic_to_shared(s_tile.ptr_to([sched_state.stage])), txl.uint32(1))
                        txl.ptx["st.shared::cluster.s32"](peer_tile_addr, claimed)
                        txl.ptx["mapa.shared::cluster.u32"](peer_full_addr, txl.cuda.cvta_generic_to_shared(sched_full.ptr_to([sched_state.stage])), txl.uint32(1))
                        txl.ptx["mbarrier.arrive.release.cluster.shared::cluster.b64"](peer_full_addr)
                        txl.ptx["mbarrier.arrive.release.cluster.shared::cta.b64"](txl.cuda.cvta_generic_to_shared(sched_full.ptr_to([sched_state.stage])))
                        sched_state.advance()
                    # The last pair to see the counter drained resets it for the next launch.
                    pairs_done = txl.local_scalar("int32")
                    # acq_rel: each pair releases its last ticket fetch; the last pair acquires them all
                    # before resetting the counter.
                    txl.ptx["atom.acq_rel.gpu.global.add.s32"](pairs_done, ctr.ptr_to([1]), txl.int32(1))
                    with txl.if_(pairs_done == NUM_SMS // 2 - 1), txl.then_():
                        txl.ptx.st.global_.s32(ctr.ptr_to([0]), txl.int32(0))
                        txl.ptx.st.global_.s32(ctr.ptr_to([1]), txl.int32(0))

        with mma_role:
            with txl.if_(rank == 0), txl.then_():
                with txl.if_(txl.cuda.elect_sync() != txl.uint32(0)), txl.then_():
                    data_state, acc_state = txl.PipelineState(STAGES, 0), txl.PipelineState(ACC_STAGES, 0)
                    a_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_a.ptr_to([0])))
                    b_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_b.ptr_to([0])))
                    sfa_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfa.ptr_to([0])))
                    sfb_base = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(s_sfb.ptr_to([0])))

                    def process_tile(m0, n0, group_idx, row0, num_k_blocks):
                        timed_wait("mma-wait-acc", acc_empty, acc_state.stage, acc_state.phase ^ 1)
                        tile_range = range_start("mma-tile")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(num_k_blocks) as k:
                            a_desc = txl.local_scalar("uint64", init=smem_descriptor(AB_DESC_BASE, a_base + to_u32(data_state.stage * A_BYTES)))
                            b_desc = txl.local_scalar("uint64", init=smem_descriptor(B_DESC_BASE, b_base + to_u32(data_state.stage * B_HALF)))
                            instr_desc = txl.local_scalar("uint32", init=with_sf_id(txl.uint32(INSTR_DESC), unsigned_mod(k, 4)))
                            sfa_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfa_base + to_u32(data_state.stage * 512)))
                            sfb_desc = txl.local_scalar("uint64", init=smem_descriptor(SF_DESC_BASE, sfb_base + to_u32(data_state.stage * (BN * 4))))
                            timed_wait("mma-wait-tma", load_full, data_state.stage, data_state.phase)
                            timed_wait("mma-wait-sf", sf_full, data_state.stage, data_state.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.if_(unsigned_mod(k, 4) == 0), txl.then_():
                                txl.ptx[UTCCP_PAIR](txl.uint32(SFA_COL), sfa_desc)
                                for chunk in range(N_SF_CHUNKS):
                                    txl.ptx[UTCCP_PAIR](txl.uint32(SFB_COL + 4 * chunk), offset_descriptor(sfb_desc, chunk * 512))
                            for k_step in range(BK // UMMA_K):
                                txl.ptx[MMA_PAIR](to_u32(acc_state.stage * BN), offset_descriptor(a_desc, k_step * K_STEP), offset_descriptor(b_desc, k_step * B_K_STEP), instr_desc, txl.uint32(SFA_COL), txl.uint32(SFB_COL), txl.Or(k > 0, k_step > 0) if k_step == 0 else txl.bool(True))
                            load_empty.arrive(data_state.stage, cta_group=2, cta_mask=3)
                            data_state.advance()
                        acc_full.arrive(acc_state.stage, cta_group=2, cta_mask=3)
                        range_end(tile_range)
                        acc_state.advance()

                    for_each_tile(process_tile)

        with epi_role:
            epi_warp = warp - 4
            quad = warp % 4
            warpgroup = epi_warp // 4
            wg_tidx, row = (epi_warp % 4) * 32 + lane, quad * 32 + lane
            acc_state = txl.PipelineState(ACC_STAGES, 0)
            acc_words = txl.alloc_local((COLS_PER_WG,), "uint32")

            def process_tile(m0, n0, group_idx, row0, num_k_blocks):
                if USE_EPILOGUE_D_PREFETCH:
                    # Long tiles: prefetch this warpgroup's d slice while the MMA accumulates it.
                    with txl.if_(wg_tidx == 0), txl.then_():
                        for chunk in range(COLS_PER_WG // REDUCE_COLS):
                            txl.ptx[PREFETCH3D](txl.address_of(d_map), n0 + warpgroup * COLS_PER_WG + chunk * REDUCE_COLS, m0, group_idx, uint64_const(EVICT_LAST))
                timed_wait("epi-wait-acc", acc_full, acc_state.stage, acc_state.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                drain_range = range_start("epi-drain")
                for chunk in range(COLS_PER_WG // 32):
                    tmem_addr = txl.bitwise_or(txl.shift_left(to_u32(quad * 32), txl.uint32(16)), to_u32(acc_state.stage * BN + warpgroup * COLS_PER_WG + chunk * 32))
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](*[acc_words[chunk * 32 + j] for j in range(32)], tmem_addr)
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx.tcgen05.fence__before_thread_sync()
                acc_empty.arrive(acc_state.stage, remote=0)
                range_end(drain_range)
                store_range = range_start("epi-store")
                if USE_SHARED_STORE_BOX:
                    # Batch all four chunks for one owner before handing the box to the other WG. The
                    # final common barrier of a tile protects the first write of the next one, so only
                    # two cross-WG barriers are needed per tile instead of one per reduce.
                    for owner in range(2):
                        with txl.if_(warpgroup == owner), txl.then_():
                            for chunk in range(COLS_PER_WG // REDUCE_COLS):
                                with txl.if_(wg_tidx == 0), txl.then_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                                txl.ptx.bar.sync(to_u32(EPI_BARRIER + owner), txl.uint32(128))
                                for unit in range(REDUCE_COLS // 4):
                                    swizzled = txl.bitwise_xor(unit, row % 8)
                                    txl.ptx.st.shared.v4.u32(s_store.ptr_to([row * REDUCE_COLS + swizzled * 4]), acc_words[chunk * REDUCE_COLS + unit * 4], acc_words[chunk * REDUCE_COLS + unit * 4 + 1], acc_words[chunk * REDUCE_COLS + unit * 4 + 2], acc_words[chunk * REDUCE_COLS + unit * 4 + 3])
                                txl.ptx.fence.proxy.async_.shared__cta()
                                txl.ptx.bar.sync(to_u32(EPI_BARRIER + owner), txl.uint32(128))
                                with txl.if_(wg_tidx == 0), txl.then_():
                                    txl.ptx[REDUCE3D + ".L2::cache_hint"](txl.address_of(d_map), n0 + owner * COLS_PER_WG + chunk * REDUCE_COLS, m0, group_idx, s_store.ptr_to([0]), uint64_const(EVICT_LAST))
                                    txl.ptx.cp.async_.bulk.commit_group()
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                        txl.ptx.bar.sync(to_u32(EPI_BARRIER + 2), txl.uint32(256))
                else:
                    for chunk in range(COLS_PER_WG // REDUCE_COLS):
                        buf_offset = (warpgroup * STORE_BUFFERS + chunk % STORE_BUFFERS) * (BM * REDUCE_COLS)
                        with txl.if_(wg_tidx == 0), txl.then_():
                            txl.ptx.cp.async_.bulk.wait_group.read(STORE_BUFFERS - 1)
                        txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                        for unit in range(REDUCE_COLS // 4):
                            swizzled = txl.bitwise_xor(unit, row % 8) if REDUCE_COLS == 32 else txl.bitwise_xor(unit, (row // 2) % 4)
                            txl.ptx.st.shared.v4.u32(s_store.ptr_to([buf_offset + row * REDUCE_COLS + swizzled * 4]), acc_words[chunk * REDUCE_COLS + unit * 4], acc_words[chunk * REDUCE_COLS + unit * 4 + 1], acc_words[chunk * REDUCE_COLS + unit * 4 + 2], acc_words[chunk * REDUCE_COLS + unit * 4 + 3])
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(to_u32(EPI_BARRIER + warpgroup), txl.uint32(128))
                        with txl.if_(wg_tidx == 0), txl.then_():
                            # evict_last keeps the reduced d lines resident in L2 while the stream runs
                            txl.ptx[REDUCE3D + ".L2::cache_hint"](txl.address_of(d_map), n0 + warpgroup * COLS_PER_WG + chunk * REDUCE_COLS, m0, group_idx, s_store.ptr_to([buf_offset]), uint64_const(EVICT_LAST))
                            txl.ptx.cp.async_.bulk.commit_group()
                range_end(store_range)
                acc_state.advance()

            for_each_tile(process_tile, is_one_lane=False)
            with txl.if_(wg_tidx == 0), txl.then_():
                txl.ptx.cp.async_.bulk.wait_group(0)

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with idle_role:
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

    return kgrouped


def compile_entry(kernel):
    return entry_function(kernel.compile(TARGET))


@functools.cache
def mgrouped_swap_exec(STAGES, USE_VECTOR_SCALES=False, TAIL_STEPS=4):
    return compile_entry(build_mgrouped_swap(STAGES, USE_VECTOR_SCALES, TAIL_STEPS))


@functools.cache
def mgrouped_1cta_exec(STAGES, USE_VECTOR_SCALES=False):
    return compile_entry(build_mgrouped_1cta(STAGES, USE_VECTOR_SCALES))


@functools.cache
def mgrouped_exec(STAGES, BN, STORE_BUFFERS=1, USE_SCALE_TAIL=False, USE_VECTOR_SCALES=False):
    return compile_entry(build_mgrouped(STAGES, BN, STORE_BUFFERS, USE_SCALE_TAIL, USE_VECTOR_SCALES))


@functools.cache
def mgrouped_w320_exec(STAGES, USE_VECTOR_SCALES=False):
    return compile_entry(build_mgrouped_w320(STAGES, USE_VECTOR_SCALES))


@functools.cache
def mgrouped_w320r_exec(STAGES, USE_VECTOR_SCALES=False, DEFER_STEPS=2, SUB_N=160, STORE_BUFFERS=1, USE_LAST_TILE_BUFFERS=False):
    return compile_entry(build_mgrouped_w320r(STAGES, USE_VECTOR_SCALES, DEFER_STEPS, SUB_N, STORE_BUFFERS, USE_LAST_TILE_BUFFERS))


def mgrouped_config(K, N):
    """
    build_mgrouped's (STAGES, BN, STORE_BUFFERS[, USE_SCALE_TAIL]) per shape: BN=192
    (double-buffered accumulator, 7 stages) when it tiles N exactly and 256 does not; otherwise
    256-wide scale-tail tiles, with 2 store buffers when K is short.
    """
    if N % 256 != 0 and N % 192 == 0:
        return (7, 192, 1)
    return (5, 256, 2, True) if K <= 512 else (6, 256, 1, True)


USE_ONE_CTA = True
W320_STAGES = 5
W320_DEFER = 3
USE_ROTATING_256 = True
ROTATING_256_DEFER = 2


def fp8_grouped_gemm_sm100(a, b, n, grouped_mm_offs):
    """
    fp8_grouped_gemm on SM100; the 1x128 scale of a must be row-major.
    """
    x, x_scale = a
    weight, weight_scale = b
    M, K = x.shape
    G = weight.shape[0]
    out = torch.empty((M, n), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return out
    # Small groups (< 512 rows on average) waste most on odd 128-row tails: swap A/B so tokens are
    # the MMA N dimension and a tail costs half a tile. Weight-side padding (n % 256) costs far less.
    # With short K the swap epilogue (lane-pair transpose, 32-token store boxes) costs more than the
    # tail it saves.
    # BN=192 tiles (exact for n % 192 == 0) load 14% more operand bytes per FLOP than the swap kernel's
    # 256x256 tiles; when padding n up to 256 wastes <= 5% (n = 5760: 2.2%), the swap kernel wins.
    n_padded = (n + 255) // 256 * 256
    use_swap_for_padding = n % 256 != 0 and n % 192 == 0 and n_padded * 20 <= n * 21
    use_vector_scales = ((K + 127) // 128) % 4 == 0 and x_scale.data_ptr() % 16 == 0 and weight_scale.data_ptr() % 16 == 0
    if n % 320 == 0 and n % 256 != 0 and M >= 512 * G:
        fn = mgrouped_w320r_exec(W320_STAGES, use_vector_scales, W320_DEFER, 160, 1, True)
    elif M < 512 * G and n % 256 == 0 and USE_ONE_CTA:
        # small groups, 256-multiple n: 1-CTA 128-row tiles (no pair padding)
        fn = mgrouped_1cta_exec(4, use_vector_scales)
    elif (M < 512 * G or use_swap_for_padding) and K > 512:
        tail_steps = (K % BK + UMMA_K - 1) // UMMA_K if K % BK else BK // UMMA_K
        fn = mgrouped_swap_exec(6, use_vector_scales, tail_steps)
    elif n % 320 == 0 and n % 256 != 0:
        # n = 2880: 320-wide tiles (two N=160 sub-MMAs) move 23% fewer operand bytes per FLOP than BN=192
        fn = mgrouped_w320_exec(W320_STAGES, use_vector_scales)
    elif n % 256 == 0 and K <= 512 and n >= 1024 and USE_ROTATING_256:
        fn = mgrouped_w320r_exec(5, use_vector_scales, ROTATING_256_DEFER, 128, 2)
    elif n % 256 == 0 and K >= 1024 and n >= 1024 and USE_ROTATING_256:
        # 256-wide tiles as two N=128 sub-MMAs over three rotating 128-col TMEM slots (scales in dedicated
        # columns): no scale-tail round trip between tiles. Short K (512: store-bound) and n = 512 rows stay
        # on the BN256 scale-tail kernel, which is faster there.
        fn = mgrouped_w320r_exec(6, use_vector_scales, ROTATING_256_DEFER, 128)
    else:
        fn = mgrouped_exec(*mgrouped_config(K, n), USE_VECTOR_SCALES=use_vector_scales)
    fn(x, x_scale, weight, weight_scale, out, grouped_mm_offs, M, n, K, G)
    return out


@functools.cache
def kgrouped_exec(STAGES, STORE_BUFFERS=1, USE_REVERSE_WALK=False, BN=256, USE_D_PREFETCH=False, USE_EPILOGUE_D_PREFETCH=False, REDUCE_COLS=32, USE_SHARED_STORE_BOX=False):
    return compile_entry(build_kgrouped(STAGES, STORE_BUFFERS, USE_REVERSE_WALK, BN, USE_D_PREFETCH, USE_EPILOGUE_D_PREFETCH, REDUCE_COLS, USE_SHARED_STORE_BOX))


@txl.kernel()
def zero2(ctr: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)):
    txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
    with txl.if_(txl.thread_id() < 2), txl.then_():
        txl.ptx.st.global_.s32(ctr.ptr_to([txl.thread_id()]), txl.int32(0))


COUNTERS = {}


def tile_counter(device):
    """
    Per (device, stream) tile counter, zeroed once; each k-grouped launch leaves it zero.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    counter = COUNTERS.get(key)
    if counter is None:
        counter = torch.empty(2, dtype=torch.int32, device=device)
        zero_exec()(counter)
        COUNTERS[key] = counter
    return counter


@functools.cache
def zero_exec():
    """
    The compiled kernel that zeroes a fresh tile counter.
    """
    return compile_entry(zero2)


def k_grouped_fp8_gemm_sm100(a, b, d, ks, ks_tensor):
    """
    k_grouped_fp8_gemm on SM100: d (G, N, K) fp32 += a_g^T @ b_g in place; scales row-major.
    """
    a_data, a_scale = a
    b_data, b_scale = b
    M, N = a_data.shape
    K = b_data.shape[1]
    G = d.shape[0]
    if M == 0 or G == 0:
        return
    # 256x256 pair tiles: half the A re-reads of 256x128 tiles; dynamic tile claiming balances
    # the per-SM reduce throughput.
    # d prefetch into L2 when groups average < 2048 rows: the epilogue prefetches its tile's d slice at
    # tile start (about one MMA tile of lead), so the reduce-add hits L2 instead of stalling on its own
    # DRAM miss. With >= 2048 rows per group the tiles are long enough that the prefetched lines only
    # add traffic.
    if M < 2048 * G:
        fn = kgrouped_exec(5, 1, True, 256, False, True)
    else:
        # long tiles (TC-bound): a sixth ring stage, funded by 16-column (8 KB per warpgroup) reduce chunks
        fn = kgrouped_exec(6, 1, True, 256, False, False, 32, True)
    fn(a_data, a_scale, b_data, b_scale, d, ks_tensor, tile_counter(d.device), M, N, K, G)
