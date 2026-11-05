"""
SM100 TIRx-lite kernels for the fused BF16 expert grouped GEMMs.

One persistent, warp-specialized tcgen05 grouped-GEMM template covers the six entry points. The
M-grouped kinds (rows of expert g are cu[g] <= r < cu[g + 1], fixed K) are gate_up_fwd, down_fwd,
down_dgrad and gate_up_dgrad; the K-grouped kinds (expert g reduces over its rows, fixed output)
are gate_up_wgrad and down_wgrad.

Persistent 2-CTA clusters (tcgen05.mma.cta_group::2, 256 x 256 x 64 tiles, two TMEM
accumulators). Warpgroup 0 holds the TMA producer, the MMA issuer (CTA0) or gather relay (CTA1)
and the tile-prefix warp; then come 4 or 8 epilogue warps, then 4, 8 or 16 cp.async row-gather
warps for the a_idx / slot_of operands. Small expert groups (mean rows <= SWAP_ROWS) run the
M-grouped GEMMs swap-AB: weights on the MMA M side, the group's tokens on a runtime N (<= 256,
steps of 32), so a group tail wastes < 32 rows. The wgrads gather their rows in-kernel and
accumulate the fp32 main grads by TMA reduce-add.

The two gate / up weights are read through one rank-4 tensor map (H, 2, I, G) whose second mode
strides between the two weights in address order, so the MMA sees the interleaved rows
[w_lo[0]; w_hi[0]; w_lo[1]; ...]: the accumulator columns are the interleaved preact pairs, and
the gate_up dgrad contracts the interleaved dpre directly. TMA strides stop below 2^40 bytes, so
two independent allocations farther apart than that (separate parameters in training) take the
far-pair variants instead: the gate_up fwd / dgrad weight rows come in through the cp.async
gather warps with 64-bit addresses (same interleaved smem rows, same epilogues), and the gate_up
wgrad reduces each main grad through its own tensor map.

Kernel overlap: every launch uses programmatic dependent launch. The wgrads, when the host finds
them independent of every launch that may still run on the stream (byte extents), take an
overlap variant: cluster-launch-control dynamic tile scheduling and no griddepcontrol.wait before
the main loop, so their clusters fill the SMs the previous kernel frees in its tail; they wait
for the previous grid at exit, so completion order is unchanged.
"""
import functools
import inspect

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["bf16_expert_gate_up_fwd_sm100", "bf16_expert_down_fwd_sm100", "bf16_expert_down_dgrad_sm100", "bf16_expert_gate_up_dgrad_sm100", "bf16_expert_gate_up_wgrad_sm100", "bf16_expert_down_wgrad_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


CTA_M = 128  # tile rows per CTA
CLUSTER_M = 256  # tile rows per 2-CTA cluster
BLOCK_K = 64  # K per pipeline stage
CLUSTER_N = 256  # widest cluster tile N, one TMEM accumulator of 256 fp32 columns
CTA_N = 128  # B columns each CTA of the pair stages
STAGES = 6
MAX_GROUPS = 1024  # bounded by the shared-memory tile prefix
# warp roles: warpgroup 0 holds the TMA producer, the MMA issuer (CTA0) or gather relay (CTA1),
# the tile-prefix warp and the CLC scheduler (overlap variant only); then the epilogue warps,
# then the cp.async gather warps
TMA_WARP = 0
MMA_WARP = 1
PREFIX_WARP = 2
SCHEDULER_WARP = 3
FIRST_EPILOGUE_WARP = 4
USE_IKET = False  # compile through the iket profiler, which reads the profile ranges
TILE_SLOTS = 2  # tile-id ring depth

MMA_F16 = "tcgen05.mma.cta_group::2.kind::f16"
TMEM_LD_32X32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMA_LOAD = "cp.async.bulk.tensor.{}d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
CP_ASYNC_16B = "cp.async.cg.shared.global.L2::128B"
TMA_STORE_2D = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
LOG2E = 1.4426950408889634

# kind -> (k_grouped, a_major, b_major, a_gather)
KINDS = {
    "gu_fwd": (False, "k", "k", True),
    "down_fwd": (False, "k", "k", False),
    "down_dgrad": (False, "k", "mn", True),
    "gu_dgrad": (False, "k", "mn", False),
    "gu_wgrad": (True, "mn", "mn", False),  # B is the gathered x rows
    "down_wgrad": (True, "mn", "mn", True),  # A is the gathered dy rows
}


def profile_range_start(name):
    """
    Open a named iket profile range; returns the token profile_range_end closes.
    """
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def profile_range_end(token):
    txl.cuda.iket.range_end(token[0])


class OffsetView:
    """
    Read-only view array[offset + i] of a register array.
    """

    def __init__(self, array, offset):
        self.array, self.offset = array, offset

    def __getitem__(self, i):
        return self.array[self.offset + i]


def sigmoid(x):
    """
    1 / (1 + 2^(-x log2 e)) in fp32.
    """
    exp_neg = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(exp_neg, x * txl.float32(-LOG2E))
    rcp = txl.local_scalar("float32")
    txl.ptx.rcp.approx.ftz.f32(rcp, txl.float32(1.0) + exp_neg)
    return rcp


def bf16_lo(word):
    """
    The low bf16 of a packed word, as fp32.
    """
    return txl.reinterpret(word << txl.uint32(16), ty="float32")


def bf16_hi(word):
    """
    The high bf16 of a packed word, as fp32.
    """
    return txl.reinterpret(word & txl.uint32(0xFFFF0000), ty="float32")


def pack_bf16x2(dst, lo, hi):
    """
    dst = (bf16(lo), bf16(hi)) as one word, lo in the low half; round to nearest even.
    """
    txl.ptx["cvt.rn.bf16x2.f32"](dst, hi, lo)


def swiglu_act(gate, up, clamped, alpha, limit):
    """
    h of the bf16-rounded gate / up (fp32 values).
    """
    if not clamped:
        return gate * sigmoid(gate) * up
    gate_clamped, up_clamped = txl.Select(gate < limit, gate, limit), txl.Select(up < limit, txl.Select(up > -limit, up, -limit), limit)
    return (up_clamped + txl.float32(1.0)) * gate_clamped * sigmoid(alpha * gate_clamped)


def kernel_params(kind, has_bias, clamped, far=False):
    """
    The traced kernel's (name, type) parameters for one kind.
    """
    i32, i64, f32 = "int32", "int64", "float32"

    # a fresh Tensor type (own symbolic extent) per parameter: a shared one would make the host
    # check require equal lengths
    def bf16_tensor():
        return txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1)

    def u32_tensor():
        return txl.Tensor((txl.int64(),), txl.u32, data_alignment=1)

    params = [("cu", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("G", i32), ("M", i32), ("num_ctas", i32)]
    if kind == "gu_fwd":
        params += [("x", bf16_tensor()), ("T", i32), ("a_idx", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("w", bf16_tensor()), ("wdiff", i64), ("ws0", i64), ("ws1", i64), ("preact", u32_tensor()), ("h", u32_tensor()), ("swapped", i32)]
        if has_bias:
            params += [("bias", u32_tensor())]
    elif kind == "down_fwd":
        params += [("hin", bf16_tensor()), ("wd", bf16_tensor()), ("slot_of", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("out", u32_tensor())]
        if has_bias:
            params += [("bias", u32_tensor())]
    elif kind == "down_dgrad":
        params += [("dy", bf16_tensor()), ("T", i32), ("slot_of", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("wd", bf16_tensor()), ("preact", u32_tensor()), ("dpre", u32_tensor()), ("h", u32_tensor()), ("swapped", i32)]
    elif kind == "gu_dgrad":
        params += [("dpre", bf16_tensor()), ("w", bf16_tensor()), ("wdiff", i64), ("ws0", i64), ("ws1", i64), ("dx", u32_tensor())]
    elif kind == "gu_wgrad":
        params += [("dpre", bf16_tensor()), ("x", bf16_tensor()), ("T", i32), ("a_idx", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("mg", txl.Tensor((txl.int64(),), txl.f32, data_alignment=1))] + ([("mg2", txl.Tensor((txl.int64(),), txl.f32, data_alignment=1))] if far else []) + [("mgdiff", i64), ("ms0", i64), ("ms1", i64), ("flip", i32)]
    elif kind == "down_wgrad":
        params += [("dy", bf16_tensor()), ("T", i32), ("slot_of", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("hin", bf16_tensor()), ("mg", txl.Tensor((txl.int64(),), txl.f32, data_alignment=1))]
    if clamped and kind in ("gu_fwd", "down_dgrad"):
        params += [("alpha", f32), ("limit", f32)]
    return params


SWAP_ROWS = 512  # swap-AB when the mean rows per expert is at most this (tail waste dominates)


def has_swap_ab(kind, inter):
    """
    Kinds whose swap-AB variant is faster for small groups: the gate_up GEMMs, and down_fwd when
    its K = I is long enough to hide the row-scatter epilogue.
    """
    return kind in ("gu_fwd", "gu_dgrad") or (kind == "down_fwd" and inter >= 2048)


def cluster_tile_n(kind, N, swap, far):
    """
    Cluster tile N: 240 where N is a multiple of 240 but not of 256 (gpt-oss 2880). down_fwd only
    (K-major B): the MN-major-B kinds (gu_dgrad, down_dgrad) are slower with 120-column B
    halves per CTA.
    """
    if kind == "down_fwd" and not swap and not far and N % 256 and N % 240 == 0:
        return 240
    return CLUSTER_N


# gate_up fwd: the non-swap kernel is already faster at 512 rows per expert
SWAP_ROWS_GU_FWD = 448


def has_small_groups(M, G, rows=SWAP_ROWS):
    """
    The mean rows per expert is at most rows (the swap-AB condition).
    """
    return M <= rows * G


def use_materialized_gu_fwd(M, G, H, inter):
    """
    The materialized gather plus pure-TMA GEMM is faster only for the GPT-OSS shape (H = I = 2880)
    at a mean of 1024 or more rows per expert.
    """
    return H == 2880 and inter == 2880 and M >= 1024 * G


@functools.cache
def build_expert_gemm(kind, H, inter, has_bias, clamped, swap=False, materialized=False, far=False, dyn=False):
    """
    Compile the grouped GEMM of one kind (KINDS) for H and I = inter; the remaining arguments pick
    its variant: bias, clamped SwiGLU, swap-AB, materialized gather, far pair and overlap (dyn).
    """
    k_grouped, a_major, b_major, a_gather = KINDS[kind]
    # problem geometry per kind: (K, N) for M-grouped, (OUT_M, N) for K-grouped
    if kind == "gu_fwd":
        K, N = H, 2 * inter
    elif kind == "down_fwd":
        K, N = inter, H
    elif kind == "down_dgrad":
        K, N = H, inter
    elif kind == "gu_dgrad":
        K, N = 2 * inter, H
    elif kind == "gu_wgrad":
        OUT_M, N = 2 * inter, H
    else:
        OUT_M, N = H, inter
    # gpt-oss N = 2880 = 11.25 x 256: uniform 240-column tiles (12 x 240) instead of 11 full
    # tiles and a quarter tile that costs a full one (non-swap, fast-path plain epilogues only)
    TILE_N = cluster_tile_n(kind, N, swap and has_swap_ab(kind, inter), far)
    CTA_TILE_N = TILE_N // 2
    NUM_N_TILES = -(-N // TILE_N)
    NUM_K_TILES = None if k_grouped else -(-K // BLOCK_K)
    NUM_M_TILES = -(-OUT_M // CLUSTER_M) if k_grouped else None
    A_STAGE_BYTES = CTA_M * BLOCK_K * 2
    B_STAGE_BYTES = (CTA_TILE_N if b_major == "k" else CTA_N) * BLOCK_K * 2
    # For large GPT-OSS forward rows, a separate coalesced gather followed by a pure-TMA
    # gate/up GEMM is faster than issuing the indexed cp.async gather inside every K stage.
    # Keep the indexed cp.async path for every other kind and shape.
    USE_MATERIALIZED = materialized and kind == "gu_fwd"
    if USE_MATERIALIZED:
        a_gather = False
    # far-pair gate / up weights (more than 2^40 bytes apart): no tensor map spans both, so the
    # gather warps load the interleaved weight rows (B) with plain 64-bit addresses
    HAS_FAR_WEIGHTS = far and kind in ("gu_fwd", "gu_dgrad")
    HAS_FAR_MAIN_GRADS = far and kind == "gu_wgrad"  # far-pair main grads: one reduce tensor map per buffer
    # the gathered operand(s): A for gu_fwd / down_dgrad / down_wgrad, B for gu_wgrad and the
    # far-pair weights
    GATHER_A = kind in ("gu_fwd", "down_dgrad", "down_wgrad") and not USE_MATERIALIZED
    GATHER_B = kind == "gu_wgrad" or HAS_FAR_WEIGHTS
    HAS_GATHER = GATHER_A or GATHER_B
    TMA_STAGE_BYTES = (0 if GATHER_A else A_STAGE_BYTES) + (0 if GATHER_B else B_STAGE_BYTES)
    # the SwiGLU-heavy epilogues get two warps per TMEM lane quadrant (column halves)
    NUM_EPILOGUE_WARPS = 8 if kind in ("gu_fwd", "down_dgrad") else 4
    # longest-group-first tile order (helps down_wgrad's tail; not gu_wgrad's)
    USE_LPT_ORDER = kind == "down_wgrad"
    CHUNKS_PER_WARP = (CLUSTER_N // 32) // (NUM_EPILOGUE_WARPS // 4)  # 32-column chunks per epilogue warp
    FIRST_GATHER_WARP = FIRST_EPILOGUE_WARP + NUM_EPILOGUE_WARPS
    # gather warps: more warps keep more cp.async row segments in flight. gu_fwd: 8; wgrads: 16
    # for small I (qwen-like), 8 for I = 2880, where 16 is slower; down_dgrad and the far-pair
    # weight gathers keep 4 (register budget)
    NUM_GATHER_WARPS = 4
    if HAS_GATHER and not HAS_FAR_WEIGHTS and kind in ("gu_fwd", "gu_wgrad", "down_wgrad"):
        NUM_GATHER_WARPS = 16 if (k_grouped and inter <= 1024) else 8
    ROWS_PER_GATHER_WARP = CTA_M // NUM_GATHER_WARPS  # K-major A: tile rows per gather warp
    K_ROWS_PER_GATHER_WARP = BLOCK_K // NUM_GATHER_WARPS  # MN-major: K rows per gather warp
    NUM_WARPS = FIRST_GATHER_WARP + (NUM_GATHER_WARPS if HAS_GATHER else 0)
    # setmaxnreg per role (0: warpgroup 0, 1: gather warps, 2: epilogue warps): down_dgrad's
    # SwiGLU-backward epilogue takes the registers the other roles give up
    MAX_REGS = (56, 80, 184) if kind == "down_dgrad" else None
    # swap-AB (weights on the MMA M side, tokens on a runtime N <= 256 in steps of 32): the
    # gate_up GEMMs always; down_fwd when its K = I is long enough to hide the scatter epilogue
    USE_SWAP_AB = swap and has_swap_ab(kind, inter)
    USE_STAGING = not k_grouped and not USE_SWAP_AB
    # per-warp transpose tile (u32 words); the plain-SwiGLU down_dgrad stores dpre in two 64-byte
    # halves through a 2 KB tile to afford a 5th stage (the clamped one spills with that split)
    USE_SPLIT_DPRE = kind == "down_dgrad" and not clamped
    STAGING_WORDS = 1024 if (kind == "down_dgrad" and not USE_SPLIT_DPRE) else 512
    # the wgrads keep 32 KB of smem for the fp32 reduce staging, down_dgrad its transpose and
    # preact buffers: a stage fewer (two for the clamped down_dgrad)
    NUM_STAGES = STAGES - 1 if k_grouped else ((STAGES - 1 if USE_SPLIT_DPRE else STAGES - 2) if (kind == "down_dgrad" and not USE_SWAP_AB) else STAGES)
    # down_dgrad prefetches the next preact chunk by cp.async into a per-warp smem tile
    USE_PREACT_PREFETCH = kind == "down_dgrad" and not USE_SWAP_AB
    # M-grouped bf16 outputs (down_dgrad dpre / h, gu_fwd preact / h, gu_dgrad dx): full 32-row
    # chunks leave by TMA store from the swizzled staging tile (the epilogue warps skip the
    # ld.shared + st.global copy-out); group-tail chunks keep the masked stores
    USE_TMA_STORE = kind in ("down_dgrad", "gu_fwd", "gu_dgrad") and not USE_SWAP_AB
    USE_SCATTER_STAGING = kind == "down_fwd" and USE_SWAP_AB  # swap-AB scatter: stage token rows across the 4 lane-quadrant warps
    a_shape = (NUM_STAGES, CTA_M, BLOCK_K) if a_major == "k" else (NUM_STAGES, BLOCK_K, CTA_M)
    b_shape = (NUM_STAGES, CTA_TILE_N, BLOCK_K) if b_major == "k" else (NUM_STAGES, BLOCK_K, CTA_N)
    a_is_mn_major = a_major == "mn"
    b_is_mn_major = b_major == "mn"
    # overlap variant (dyn): dynamic tile scheduling by cluster launch control (the grid holds one
    # cluster per tile, upper bound; resident clusters cancel pending ones and take their tile
    # ids) and no griddepcontrol.wait before the main loop: clusters land on the SMs the previous
    # (independent, host-checked) kernel frees in its tail and take more tiles than late ones.
    # Every CTA waits for the previous grid before it exits, so this grid's completion still
    # implies its predecessor's.
    USE_DYNAMIC_TILES = dyn
    # ring consumers per entry (arrivals on CTA0's slot-empty barrier): the TMA producer(s),
    # the MMA issuer, CTA1's gather relay, every epilogue warp and every gather warp
    NUM_SLOT_CONSUMERS = (1 if TMA_STAGE_BYTES == 0 else 2) + 1 + (1 if HAS_GATHER else 0) + 2 * NUM_EPILOGUE_WARPS + (2 * NUM_GATHER_WARPS if HAS_GATHER else 0)
    param_specs = kernel_params(kind, has_bias, clamped, HAS_FAR_MAIN_GRADS)
    param_names = [name for name, _ in param_specs]

    def encode_tensor_map(tensor_map, rank, data, *dims_strides_box, dtype="bfloat16"):
        txl.cu_tensor_map_encode_tiled(tensor_map, dtype, rank, data, *dims_strides_box)

    def host_prelude(params):
        """
        Encode the kind's tensor maps on the host: a and b for the operands, c (and c2) for the
        main-grad reduce, d and e for the TMA-stored outputs.
        """
        maps = {}
        if kind == "gu_fwd":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["a"], 2, params["x"].data, H, params["T"], H * 2, 64, CTA_M if USE_MATERIALIZED else 1, 1, 1, 0, 3, 2, 0)
            if HAS_FAR_WEIGHTS:
                maps["b"] = maps["a"]  # unused: the gather warps load the weights
            else:
                maps["b"] = txl.stack_alloca("tensormap", 1)
                encode_tensor_map(maps["b"], 4, params["w"].data, H, 2, inter, params["G"], params["wdiff"] * 2, params["ws1"] * 2, params["ws0"] * 2, 64, 2, CTA_N // 2, 1, 1, 1, 1, 1, 0, 3, 2, 0)
            if USE_TMA_STORE:
                # preact (M, 2I): 32 x 32 boxes (64B swizzle); h (M, I): 32 rows x 16 (32B)
                maps["d"] = txl.stack_alloca("tensormap", 1)
                encode_tensor_map(maps["d"], 2, params["preact"].data, 2 * inter, params["M"], 4 * inter, 32, 32, 1, 1, 0, 2, 2, 0)
                maps["e"] = txl.stack_alloca("tensormap", 1)
                encode_tensor_map(maps["e"], 2, params["h"].data, inter, params["M"], 2 * inter, 16, 32, 1, 1, 0, 1, 2, 0)
        elif kind == "down_fwd":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["a"], 2, params["hin"].data, inter, params["M"], inter * 2, 64, CTA_M, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["b"], 3, params["wd"].data, inter, H, params["G"], inter * 2, H * inter * 2, 64, CTA_TILE_N, 1, 1, 1, 1, 0, 3, 2, 0)
        elif kind == "down_dgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["a"], 2, params["dy"].data, H, params["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["b"], 3, params["wd"].data, inter, H, params["G"], inter * 2, H * inter * 2, 64, BLOCK_K, 1, 1, 1, 1, 0, 3, 2, 0)
            if USE_TMA_STORE:
                # epilogue TMA stores of the swizzled staging tiles: dpre (M, 2I) in 32-row boxes
                # of 64 columns (128B swizzle) or 32 (split staging, 64B swizzle), h (M, I) in
                # 32 x 32 (64B swizzle)
                maps["d"] = txl.stack_alloca("tensormap", 1)
                encode_tensor_map(maps["d"], 2, params["dpre"].data, 2 * inter, params["M"], 4 * inter, 32 if USE_SPLIT_DPRE else 64, 32, 1, 1, 0, 2 if USE_SPLIT_DPRE else 3, 2, 0)
                maps["e"] = txl.stack_alloca("tensormap", 1)
                encode_tensor_map(maps["e"], 2, params["h"].data, inter, params["M"], 2 * inter, 32, 32, 1, 1, 0, 2, 2, 0)
        elif kind == "gu_dgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["a"], 2, params["dpre"].data, 2 * inter, params["M"], 4 * inter, 64, CTA_M, 1, 1, 0, 3, 2, 0)
            if HAS_FAR_WEIGHTS:
                maps["b"] = maps["a"]  # unused: the gather warps load the weights
            else:
                maps["b"] = txl.stack_alloca("tensormap", 1)
                encode_tensor_map(maps["b"], 4, params["w"].data, H, 2, inter, params["G"], params["wdiff"] * 2, params["ws1"] * 2, params["ws0"] * 2, 64, 2, BLOCK_K // 2, 1, 1, 1, 1, 1, 0, 3, 2, 0)
            if USE_TMA_STORE:
                # dx (M, H): 32 x 32 boxes (64B swizzle)
                maps["d"] = txl.stack_alloca("tensormap", 1)
                encode_tensor_map(maps["d"], 2, params["dx"].data, H, params["M"], 2 * H, 32, 32, 1, 1, 0, 2, 2, 0)
        elif kind == "gu_wgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["a"], 2, params["dpre"].data, 2 * inter, params["M"], 4 * inter, 64, BLOCK_K, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["b"], 2, params["x"].data, H, params["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            # main grads as (H, 2, I, G) over the lower buffer: rows (i, address lane)
            maps["c"] = txl.stack_alloca("tensormap", 1)
            if HAS_FAR_MAIN_GRADS:
                # far pair: each buffer as (H, I, G), the staging tile's lane-0 rows reduce into
                # the lower buffer, its lane-1 rows into the upper one
                encode_tensor_map(maps["c"], 3, params["mg"].data, H, inter, params["G"], params["ms1"] * 4, params["ms0"] * 4, 32, 16, 1, 1, 1, 1, 0, 3, 2, 0, dtype="float32")
                maps["c2"] = txl.stack_alloca("tensormap", 1)
                encode_tensor_map(maps["c2"], 3, params["mg2"].data, H, inter, params["G"], params["ms1"] * 4, params["ms0"] * 4, 32, 16, 1, 1, 1, 1, 0, 3, 2, 0, dtype="float32")
            else:
                encode_tensor_map(maps["c"], 4, params["mg"].data, H, 2, inter, params["G"], params["mgdiff"] * 4, params["ms1"] * 4, params["ms0"] * 4, 32, 2, 16, 1, 1, 1, 1, 1, 0, 3, 2, 0, dtype="float32")
        else:  # down_wgrad
            maps["a"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["a"], 2, params["dy"].data, H, params["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["b"], 2, params["hin"].data, inter, params["M"], inter * 2, 64, BLOCK_K, 1, 1, 0, 3, 2, 0)
            maps["c"] = txl.stack_alloca("tensormap", 1)
            encode_tensor_map(maps["c"], 3, params["mg"].data, inter, H, params["G"], inter * 4, H * inter * 4, 32, 32, 1, 1, 1, 1, 0, 3, 2, 0, dtype="float32")
        return maps

    def body(params, maps):
        """
        The kernel: shared setup (shared memory, barriers, the tile prefix, TMEM), one loop per
        warp role, then the TMEM release. The role loops and the helpers they share come first.
        """

        def set_max_regs(role):
            """
            Warpgroup-collective register handoff to the SwiGLU-backward epilogue; emitted at the
            head of each role so it dominates that role's code.
            """
            if MAX_REGS is None:
                return
            num_regs = MAX_REGS[role]
            txl.ptx[f"setmaxnreg.{'inc' if role == 2 else 'dec'}.sync.aligned.u32"](num_regs)

        class TileWalk:
            """
            Per-role cursor over this cluster's tiles: tile -> (group, m_tile, n_tile) and the
            group's rows; the group position only moves forward (static schedule).
            """

            def __init__(self):
                self.tile = txl.local_scalar("int32", init=cluster_idx)
                self.group_pos = txl.local_scalar("int32", init=0)
                self.group = txl.local_scalar("int32", init=0)
                self.m_tile = txl.local_scalar("int32", init=0)
                self.n_tile = txl.local_scalar("int32", init=0)
                self.row_start = txl.local_scalar("int32", init=0)
                self.row_end = txl.local_scalar("int32", init=0)
                self.tile_rows = txl.local_scalar("int32", init=CLUSTER_N)
                if USE_DYNAMIC_TILES:
                    self.next_tile = txl.local_scalar("int32", init=0)
                    self.slot_state = txl.PipelineState(TILE_SLOTS, 0)

            def locate(self):
                """
                Find the current tile's group, its (m_tile, n_tile) and the group's row range.
                """
                if USE_DYNAMIC_TILES:
                    # CLC ids normally grow per cluster; restart the group search otherwise
                    pos_first_tile = txl.local_scalar("int32")
                    txl.ptx.ld.shared.s32(pos_first_tile, s_prefix.ptr_to([self.group_pos]))
                    with txl.if_(self.tile < pos_first_tile), txl.then_():
                        txl.assign(self.group_pos, 0)
                next_first_tile = txl.local_scalar("int32")
                txl.ptx.ld.shared.s32(next_first_tile, s_prefix.ptr_to([self.group_pos + 1]))
                with txl.while_(self.tile >= next_first_tile):
                    txl.assign(self.group_pos, self.group_pos + 1)
                    txl.ptx.ld.shared.s32(next_first_tile, s_prefix.ptr_to([self.group_pos + 1]))
                first_tile = txl.local_scalar("int32")
                txl.ptx.ld.shared.s32(first_tile, s_prefix.ptr_to([self.group_pos]))
                if USE_LPT_ORDER:
                    txl.ptx.ld.shared.s32(self.group, s_order.ptr_to([self.group_pos]))
                else:
                    txl.assign(self.group, self.group_pos)
                local_tile = self.tile - first_tile
                txl.assign(self.m_tile, local_tile // NUM_N_TILES)
                txl.assign(self.n_tile, local_tile % NUM_N_TILES)
                txl.ptx.ld.global_.s32(self.row_start, cu.ptr_to([self.group]))
                txl.ptx.ld.global_.s32(self.row_end, cu.ptr_to([self.group + 1]))
                if USE_SWAP_AB:
                    # this token chunk's MMA N: the rows left, rounded up to 32 (both CTAs
                    # then hold a multiple of 16), at most 256
                    rows_left = self.row_end - self.row_start - self.m_tile * CLUSTER_M
                    txl.assign(self.tile_rows, txl.Select(rows_left >= CLUSTER_N, txl.int32(CLUSTER_N), (rows_left + 31) // 32 * 32))

            def fetch_next(self):
                """
                Warp-collective: take the next ring entry (this cluster's next tile, or past
                total_tiles when the scheduler stopped) into next_tile and release the slot.
                """
                if USE_DYNAMIC_TILES:
                    slot_full.wait(self.slot_state.stage, self.slot_state.phase)
                    clc_ctaid = txl.local_scalar("uint32")
                    txl.query_cancel_first_ctaid_x(clc_ctaid, s_clc_resp.ptr_to([self.slot_state.stage * 4]))
                    txl.assign(self.next_tile, txl.Select(clc_ctaid == txl.uint32(0xFFFFFFFF), txl.int32(0x7FFFFFFF), txl.Cast("int32", clc_ctaid >> txl.uint32(1))))
                    txl.cuda.warp_sync()
                    with txl.if_(txl.lane_id() == 0), txl.then_():
                        slot_empty.arrive(self.slot_state.stage, remote=0)
                    self.slot_state.advance()

            def advance(self):
                """
                Move to this cluster's next tile.
                """
                if USE_DYNAMIC_TILES:
                    txl.assign(self.tile, self.next_tile)
                else:
                    txl.assign(self.tile, self.tile + params["num_ctas"] // 2)

        def cta0_full_addr(stage):
            """
            Shared-memory address of CTA0's full barrier of a stage (the TMA loads of both CTAs
            complete on it).
            """
            return txl.cuda.cvta_generic_to_shared(cta0_full.ptr_to([stage]))

        def compute_tile_prefix():
            """
            Warp-collective: s_prefix[p] = first tile of the group at position p (tiles numbered
            group by group) and s_order the group order (longest first with USE_LPT_ORDER).
            """
            running_total, group_base = txl.local_scalar("int32", init=0), txl.local_scalar("int32", init=0)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.s32(s_prefix.ptr_to([0]), txl.int32(0))
            if USE_LPT_ORDER:
                # longest-first group order: rank sort over register-resident counts (warp
                # shuffles); identity order past 128 groups
                with txl.if_(G <= 128):
                    with txl.then_():
                        counts = txl.alloc_local((4,), "int32")
                        for quarter in range(4):
                            group_idx = quarter * 32 + lane
                            txl.assign(counts[quarter], -1)
                            with txl.if_(group_idx < G), txl.then_():
                                row_start, row_end = txl.local_scalar("int32"), txl.local_scalar("int32")
                                txl.ptx.ld.global_.s32(row_start, cu.ptr_to([group_idx]))
                                txl.ptx.ld.global_.s32(row_end, cu.ptr_to([group_idx + 1]))
                                txl.assign(counts[quarter], row_end - row_start)
                        ranks = txl.alloc_local((4,), "int32")
                        for quarter in range(4):
                            txl.assign(ranks[quarter], 0)
                        other_count = txl.local_scalar("int32")
                        for other_quarter in range(4):
                            with txl.if_(other_quarter * 32 < G), txl.then_():
                                with txl.serial(32) as other_lane:
                                    txl.ptx.shfl_sync.idx.b32(other_count, counts[other_quarter], txl.Cast("uint32", other_lane), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                                    other_group = other_quarter * 32 + other_lane
                                    for quarter in range(4):
                                        group_idx = quarter * 32 + lane
                                        txl.assign(ranks[quarter], ranks[quarter] + txl.Select((other_count > counts[quarter]) | ((other_count == counts[quarter]) & (other_group < group_idx)), 1, 0))
                        for quarter in range(4):
                            group_idx = quarter * 32 + lane
                            with txl.if_(group_idx < G), txl.then_():
                                txl.ptx.st.shared.s32(s_order.ptr_to([ranks[quarter]]), group_idx)
                    with txl.else_():
                        with txl.while_(group_base < G):
                            with txl.if_(group_base + lane < G), txl.then_():
                                txl.ptx.st.shared.s32(s_order.ptr_to([group_base + lane]), group_base + lane)
                            txl.assign(group_base, group_base + 32)
                txl.cuda.warp_sync()
                txl.assign(group_base, 0)
            with txl.while_(group_base < G):
                group_idx = txl.local_scalar("int32", init=group_base + lane)
                num_group_tiles = txl.local_scalar("int32", init=0)
                if USE_LPT_ORDER:
                    with txl.if_(group_idx < G), txl.then_():
                        txl.ptx.ld.shared.s32(group_idx, s_order.ptr_to([group_base + lane]))
                with txl.if_(group_base + lane < G), txl.then_():
                    row_start, row_end = txl.local_scalar("int32"), txl.local_scalar("int32")
                    txl.ptx.ld.global_.s32(row_start, cu.ptr_to([group_idx]))
                    txl.ptx.ld.global_.s32(row_end, cu.ptr_to([group_idx + 1]))
                    if k_grouped:
                        txl.assign(num_group_tiles, txl.Select(row_end > row_start, txl.int32(NUM_M_TILES * NUM_N_TILES), txl.int32(0)))
                    else:
                        txl.assign(num_group_tiles, ((row_end - row_start + (CLUSTER_M - 1)) // CLUSTER_M) * NUM_N_TILES)
                # inclusive warp scan
                for delta in (1, 2, 4, 8, 16):
                    other_count = txl.local_scalar("int32")
                    txl.ptx.shfl_sync.up.b32(other_count, num_group_tiles, txl.uint32(delta), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                    with txl.if_(lane >= delta), txl.then_():
                        txl.assign(num_group_tiles, num_group_tiles + other_count)
                with txl.if_(group_base + lane < G), txl.then_():
                    txl.ptx.st.shared.s32(s_prefix.ptr_to([group_base + lane + 1]), running_total + num_group_tiles)
                warp_total = txl.local_scalar("int32")
                txl.ptx.shfl_sync.idx.b32(warp_total, num_group_tiles, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                txl.assign(running_total, running_total + warp_total)
                txl.assign(group_base, group_base + 32)

        def clc_scheduler_loop():
            """
            One lane: fill the tile-id ring from cluster launch control until it runs out of
            tiles, then cancel the remaining pending clusters.
            """
            full_state, empty_state = txl.PipelineState(TILE_SLOTS, 0), txl.PipelineState(TILE_SLOTS, 1)
            clc_ctaid = txl.local_scalar("uint32", init=txl.uint32(0))
            is_done = txl.local_scalar("int32", init=txl.Select(cluster_idx >= total_tiles, 1, 0))
            with txl.while_(is_done == 0):
                slot = full_state.stage
                with txl.if_(cta_rank == 0), txl.then_():
                    slot_empty.wait(empty_state.stage, empty_state.phase)
                    txl.ptx[
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                    ](s_clc_resp.ptr_to([slot * 4]), slot_full.ptr_to([slot]))
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(slot_full.ptr_to([slot]), txl.uint32(16))
                slot_full.wait(slot, full_state.phase)
                txl.query_cancel_first_ctaid_x(clc_ctaid, s_clc_resp.ptr_to([slot * 4]))
                with txl.if_((clc_ctaid == txl.uint32(0xFFFFFFFF)) | (txl.Cast("int32", clc_ctaid >> txl.uint32(1)) >= total_tiles)), txl.then_():
                    txl.assign(is_done, 1)
                full_state.advance()
                empty_state.advance()
            # drain: cancel the remaining (tile-less) pending clusters so they never launch
            with txl.if_((cta_rank == 0) & (clc_ctaid != txl.uint32(0xFFFFFFFF))), txl.then_():
                drain_phase = txl.local_scalar("int32", init=0)
                with txl.while_(clc_ctaid != txl.uint32(0xFFFFFFFF)):
                    txl.ptx[
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.b128"
                    ](s_drain_resp.ptr_to([0]), drain_full.ptr_to([0]))
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(drain_full.ptr_to([0]), txl.uint32(16))
                    drain_full.wait(0, drain_phase)
                    txl.query_cancel_first_ctaid_x(clc_ctaid, s_drain_resp.ptr_to([0]))
                    txl.assign(drain_phase, drain_phase ^ 1)

        def tma_producer_loop():
            """
            One lane: per k-step, wait for a free stage and TMA-load the operand(s) the gather
            warps do not load.
            """
            set_max_regs(0)
            walk = TileWalk()
            stage_state = txl.PipelineState(NUM_STAGES, 1)
            with txl.while_(walk.tile < total_tiles):
                walk.locate()
                walk.fetch_next()
                num_k_iters = txl.local_scalar("int32")
                if k_grouped:
                    txl.assign(num_k_iters, (walk.row_end - walk.row_start + (BLOCK_K - 1)) // BLOCK_K)
                else:
                    txl.assign(num_k_iters, NUM_K_TILES)
                with txl.if_(lane == 0), txl.then_():
                    with txl.serial(num_k_iters) as k_iter:
                        range_token = profile_range_start("p-wait-empty")
                        ab_pipe.empty.wait(stage_state.stage, stage_state.phase)
                        profile_range_end(range_token)
                        stage = stage_state.stage
                        with txl.if_(cta_rank == 0), txl.then_():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(ab_pipe.full.ptr_to([stage]), txl.uint32(2 * TMA_STAGE_BYTES))
                        a_stage = s_a[stage].ptr_to(0, 0)
                        b_stage = s_b[stage].ptr_to(0, 0)
                        as_i32 = lambda value: txl.Cast("int32", value)  # noqa: E731
                        if not k_grouped:
                            k_start = k_iter * BLOCK_K
                            a_row = walk.row_start + walk.m_tile * CLUSTER_M + cta_rank * (walk.tile_rows // 2)
                            if not a_gather:
                                txl.ptx[TMA_LOAD.format(2)](a_stage, txl.address_of(a_map), as_i32(k_start), as_i32(a_row), cta0_full_addr(stage))
                            n_start = walk.n_tile * TILE_N + cta_rank * CTA_TILE_N
                            if GATHER_B:
                                pass  # the gather warps load B
                            elif kind == "gu_fwd":
                                txl.ptx[TMA_LOAD.format(4)](b_stage, txl.address_of(b_map), as_i32(k_start), as_i32(0), as_i32(n_start // 2), as_i32(walk.group), cta0_full_addr(stage))
                            elif kind == "down_fwd":
                                txl.ptx[TMA_LOAD.format(3)](b_stage, txl.address_of(b_map), as_i32(k_start), as_i32(n_start), as_i32(walk.group), cta0_full_addr(stage))
                            elif kind == "down_dgrad":
                                for part in range(CTA_N // 64):
                                    txl.ptx[TMA_LOAD.format(3)](txl.ptx.addr(b_stage, part * BLOCK_K * 128), txl.address_of(b_map), as_i32(n_start + part * 64), as_i32(k_start), as_i32(walk.group), cta0_full_addr(stage))
                            else:  # gu_dgrad
                                for part in range(CTA_N // 64):
                                    txl.ptx[TMA_LOAD.format(4)](txl.ptx.addr(b_stage, part * BLOCK_K * 128), txl.address_of(b_map), as_i32(n_start + part * 64), as_i32(0), as_i32(k_start // 2), as_i32(walk.group), cta0_full_addr(stage))
                        else:
                            k_start = walk.row_start + k_iter * BLOCK_K
                            if kind == "gu_wgrad":
                                m_start = walk.m_tile * CLUSTER_M + cta_rank * CTA_M
                                for part in range(CTA_M // 64):
                                    txl.ptx[TMA_LOAD.format(2)](txl.ptx.addr(a_stage, part * BLOCK_K * 128), txl.address_of(a_map), as_i32(m_start + part * 64), as_i32(k_start), cta0_full_addr(stage))
                            else:
                                n_start = walk.n_tile * CLUSTER_N + cta_rank * CTA_N
                                for part in range(CTA_N // 64):
                                    txl.ptx[TMA_LOAD.format(2)](txl.ptx.addr(b_stage, part * BLOCK_K * 128), txl.address_of(b_map), as_i32(n_start + part * 64), as_i32(k_start), cta0_full_addr(stage))
                        stage_state.advance()
                txl.cuda.warp_sync()
                walk.advance()

        def gather_loop():
            """
            Per k-step: cp.async the gathered operand rows (token rows by index, or far-pair weight
            rows) into the stage, then arrive on gather_full.
            """
            set_max_regs(1)
            gather_warp = warp - FIRST_GATHER_WARP
            walk = TileWalk()
            stage_state = txl.PipelineState(NUM_STAGES, 1)
            if kind != "gu_dgrad":
                src_rows = params["x"] if kind in ("gu_fwd", "gu_wgrad") else params["dy"]
                row_ids = params["a_idx"] if kind in ("gu_fwd", "gu_wgrad") else params["slot_of"]
            my_row = txl.local_scalar("int32")  # this lane's row index (-1: zero row)
            cur_row = txl.local_scalar("int32")
            src_row = txl.local_scalar("int32")
            vec, lane_row = lane % 8, lane // 8
            FULL_MASK = txl.uint32(0xFFFFFFFF)

            def load_row_index(row_pos, row_end):
                """
                my_row = the source row of operand row row_pos, or -1 (a zero row) past row_end.
                """
                txl.assign(my_row, -1)
                with txl.if_(row_pos < row_end), txl.then_():
                    txl.ptx.ld.global_.s32(my_row, row_ids.ptr_to([row_pos]))

            def copy_row_vec(dst, src_col, col_ok):
                """
                cp.async 16 B of the row held in src_row; zero-fill rows < 0 and columns past H.
                """
                load_ok = (src_row >= 0) & col_ok
                txl.ptx[CP_ASYNC_16B](dst, src_rows.ptr_to([txl.Select(load_ok, src_row * H + src_col, 0)]), 16, txl.Select(load_ok, txl.uint32(16), txl.uint32(0)))

            def copy_weight_vec(dst, weight_row, col, load_ok):
                """
                cp.async 16 B at column col of interleaved weight row weight_row = (i, address
                lane): lane 1 is the upper buffer, wdiff elements above (64-bit addresses: no
                TMA stride limit).
                """
                offset = txl.Cast("int64", weight_row % 2) * params["wdiff"] + txl.Cast("int64", walk.group) * params["ws0"] + txl.Cast("int64", weight_row // 2) * params["ws1"] + txl.Cast("int64", col)
                txl.ptx[CP_ASYNC_16B](dst, params["w"].ptr_to([txl.Select(load_ok, offset, txl.Cast("int64", 0))]), 16, txl.Select(load_ok, txl.uint32(16), txl.uint32(0)))

            with txl.while_(walk.tile < total_tiles):
                walk.locate()
                walk.fetch_next()
                if kind == "gu_dgrad":
                    # far-pair MN-major weights: gather warp w owns K rows 16 w .. 16 w + 15 of the stage
                    num_segs = CTA_N // 64
                    weight_col_start = walk.n_tile * CLUSTER_N + cta_rank * CTA_N
                    with txl.serial(NUM_K_TILES) as k_iter:
                        range_token = profile_range_start("g-wait-empty")
                        ab_pipe.empty.wait(stage_state.stage, stage_state.phase)
                        profile_range_end(range_token)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        stage = stage_state.stage
                        for i in range(4 * num_segs):
                            pair_idx = i * 4 + lane_row
                            k_row = pair_idx // num_segs
                            seg = pair_idx % num_segs
                            col = seg * 64 + vec * 8
                            copy_weight_vec(s_b[stage].ptr_to(gather_warp * 16 + k_row, col), k_iter * BLOCK_K + gather_warp * 16 + k_row, weight_col_start + col, weight_col_start + col < H)
                        txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](gather_full.ptr_to([stage]))
                        stage_state.advance()
                elif not k_grouped:
                    # K-major A: gather warp w owns its ROWS_PER_GATHER_WARP tile rows; 8 lanes per row
                    # (far pair: likewise B's weight rows)
                    if GATHER_A:
                        load_row_index(walk.row_start + walk.m_tile * CLUSTER_M + cta_rank * (walk.tile_rows // 2) + gather_warp * ROWS_PER_GATHER_WARP + lane % ROWS_PER_GATHER_WARP, walk.row_end)
                    weight_row_start = walk.n_tile * CLUSTER_N + cta_rank * CTA_N
                    with txl.serial(NUM_K_TILES) as k_iter:
                        range_token = profile_range_start("g-wait-empty")
                        ab_pipe.empty.wait(stage_state.stage, stage_state.phase)
                        profile_range_end(range_token)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        stage = stage_state.stage
                        if GATHER_A:
                            for i in range(ROWS_PER_GATHER_WARP // 4):
                                txl.ptx.shfl_sync.idx.b32(src_row, my_row, txl.uint32(i * 4) + txl.Cast("uint32", lane_row), txl.uint32(31), FULL_MASK)
                                tile_row = gather_warp * ROWS_PER_GATHER_WARP + i * 4 + lane_row
                                copy_row_vec(s_a[stage].ptr_to(tile_row, vec * 8), k_iter * BLOCK_K + vec * 8, True)
                        if HAS_FAR_WEIGHTS:
                            for i in range(8):
                                tile_row = gather_warp * 32 + i * 4 + lane_row
                                copy_weight_vec(s_b[stage].ptr_to(tile_row, vec * 8), weight_row_start + tile_row, k_iter * BLOCK_K + vec * 8, weight_row_start + tile_row < 2 * inter)
                        txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](gather_full.ptr_to([stage]))
                        stage_state.advance()
                else:
                    # MN-major operand: gather warp w owns K_ROWS_PER_GATHER_WARP K rows of the stage
                    if kind == "gu_wgrad":
                        s_tile, num_cols, col_start = s_b, CTA_N, walk.n_tile * CLUSTER_N + cta_rank * CTA_N
                    else:
                        s_tile, num_cols, col_start = s_a, CTA_M, walk.m_tile * CLUSTER_M + cta_rank * CTA_M
                    num_segs = num_cols // 64  # 128-byte row segments per K row
                    num_k_iters = txl.local_scalar("int32", init=(walk.row_end - walk.row_start + (BLOCK_K - 1)) // BLOCK_K)
                    # the row index of k-step k_iter + 1 is loaded one step ahead (off the cp.async
                    # critical path)
                    load_row_index(walk.row_start + gather_warp * K_ROWS_PER_GATHER_WARP + lane % K_ROWS_PER_GATHER_WARP, walk.row_end)
                    with txl.serial(num_k_iters) as k_iter:
                        txl.assign(cur_row, my_row)
                        next_row_pos = walk.row_start + (k_iter + 1) * BLOCK_K + gather_warp * K_ROWS_PER_GATHER_WARP + lane % K_ROWS_PER_GATHER_WARP
                        load_row_index(next_row_pos, walk.row_end)
                        range_token = profile_range_start("g-wait-empty")
                        ab_pipe.empty.wait(stage_state.stage, stage_state.phase)
                        profile_range_end(range_token)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        stage = stage_state.stage
                        for i in range(K_ROWS_PER_GATHER_WARP * num_segs // 4):
                            # 4 (row, segment) pairs per instruction, 8 lanes each
                            pair_idx = i * 4 + lane_row
                            k_row = pair_idx // num_segs
                            seg = pair_idx % num_segs
                            txl.ptx.shfl_sync.idx.b32(src_row, cur_row, txl.Cast("uint32", k_row), txl.uint32(31), FULL_MASK)
                            col = seg * 64 + vec * 8
                            copy_row_vec(s_tile[stage].ptr_to(gather_warp * K_ROWS_PER_GATHER_WARP + k_row, col), col_start + col, col_start + col < H)
                        txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](gather_full.ptr_to([stage]))
                        stage_state.advance()
                walk.advance()

        def gather_relay_loop():
            """
            CTA1, one lane: once this CTA's gather warps filled a stage, arrive on CTA0's full
            barrier for it.
            """
            set_max_regs(0)
            walk = TileWalk()
            stage_state = txl.PipelineState(NUM_STAGES, 0)
            with txl.while_(walk.tile < total_tiles):
                walk.locate()
                walk.fetch_next()
                num_k_iters = txl.local_scalar("int32")
                if k_grouped:
                    txl.assign(num_k_iters, (walk.row_end - walk.row_start + (BLOCK_K - 1)) // BLOCK_K)
                else:
                    txl.assign(num_k_iters, NUM_K_TILES)
                with txl.if_(lane == 0), txl.then_():
                    with txl.serial(num_k_iters):
                        range_token = profile_range_start("r-wait-gfull")
                        gather_full.wait(stage_state.stage, stage_state.phase)
                        profile_range_end(range_token)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        ab_pipe.full.arrive(stage_state.stage, remote=0)
                        stage_state.advance()
                txl.cuda.warp_sync()
                walk.advance()

        def mma_loop():
            """
            CTA0: per tile, wait for a free accumulator, then per k-step wait for the stage and
            issue the 2-CTA MMAs into it.
            """
            set_max_regs(0)
            walk = TileWalk()
            stage_state = txl.PipelineState(NUM_STAGES, 0)
            acc_state = txl.PipelineState(2, 1)
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tmem_base, s_tmem_addr.ptr_to([0]))
            instr_desc = txl.alloc_local((1,), "uint32")
            txl.cuda.tcgen05.encode_instr_descriptor(txl.address_of(instr_desc[0]), d_dtype="float32", a_dtype="bfloat16", b_dtype="bfloat16", M=CLUSTER_M, N=CLUSTER_N, K=16, trans_a=b_is_mn_major if USE_SWAP_AB else a_is_mn_major, trans_b=a_is_mn_major if USE_SWAP_AB else b_is_mn_major, n_cta_groups=2)
            if TILE_N != CLUSTER_N:
                # PTX allows N in steps of 16 for cta_group::2 kind::f16 (the encoder helper
                # checks steps of 32): patch the N field (bits 17-22 = N >> 3)
                txl.assign(instr_desc[0], (instr_desc[0] & txl.uint32(0xFF81FFFF)) | txl.uint32((TILE_N >> 3) << 17))
            a_desc, a_k_offset = s_a[0].encode(major=a_major, mma_k=16)
            b_desc, b_k_offset = s_b[0].encode(major=b_major, mma_k=16)
            is_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.while_(walk.tile < total_tiles):
                walk.locate()
                walk.fetch_next()
                num_k_iters = txl.local_scalar("int32")
                if k_grouped:
                    txl.assign(num_k_iters, (walk.row_end - walk.row_start + (BLOCK_K - 1)) // BLOCK_K)
                else:
                    txl.assign(num_k_iters, NUM_K_TILES)
                range_token = profile_range_start("m-wait-acc")
                acc_pipe.empty.wait(acc_state.stage, acc_state.phase)
                profile_range_end(range_token)
                txl.ptx.tcgen05.fence__after_thread_sync()
                acc_addr = tmem_base + txl.Cast("uint32", acc_state.stage * CLUSTER_N)
                tile_instr_desc = txl.local_scalar("uint32", init=instr_desc[0])
                if USE_SWAP_AB:
                    # instruction-descriptor N field (bits 17-22 = N >> 3) for this tile
                    txl.assign(tile_instr_desc, (instr_desc[0] & txl.uint32(0xFF81FFFF)) | (txl.Cast("uint32", walk.tile_rows >> 3) << txl.uint32(17)))
                with txl.serial(num_k_iters) as k_iter:
                    range_token = profile_range_start("m-wait-full")
                    ab_pipe.full.wait(stage_state.stage, stage_state.phase)
                    profile_range_end(range_token)
                    if HAS_GATHER:
                        # this CTA's cp.async half wrote the stage through the generic proxy
                        range_token = profile_range_start("m-wait-gfull")
                        gather_full.wait(stage_state.stage, stage_state.phase)
                        profile_range_end(range_token)
                        txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.tcgen05.fence__after_thread_sync()
                    stage = stage_state.stage
                    with txl.if_(is_leader == txl.uint32(1)), txl.then_():
                        for k_step in range(BLOCK_K // 16):
                            a_mma_desc = a_desc.value + txl.Cast("uint64", stage * (A_STAGE_BYTES // 16) + a_k_offset(k_step))
                            b_mma_desc = b_desc.value + txl.Cast("uint64", stage * (B_STAGE_BYTES // 16) + b_k_offset(k_step))
                            accumulate = txl.cast(k_iter > 0, "bool") if k_step == 0 else txl.cast(True, "bool")
                            if USE_SWAP_AB:
                                # A = the weight tile (128 features per CTA), B = the token tile
                                txl.ptx[MMA_F16](acc_addr, b_mma_desc, a_mma_desc, tile_instr_desc, *([txl.uint32(0)] * 8), accumulate)
                            else:
                                txl.ptx[MMA_F16](acc_addr, a_mma_desc, b_mma_desc, instr_desc[0], *([txl.uint32(0)] * 8), accumulate)
                        ab_pipe.empty.arrive(stage, cta_group=2, cta_mask=3)
                    txl.cuda.warp_sync()
                    stage_state.advance()
                with txl.if_(is_leader == txl.uint32(1)), txl.then_():
                    acc_pipe.full.arrive(acc_state.stage, cta_group=2, cta_mask=3)
                txl.cuda.warp_sync()
                acc_state.advance()
                walk.advance()

        def epilogue_loop():
            """
            Per tile: read the accumulator from TMEM, apply the kind's epilogue math and store it;
            then release the accumulator.
            """
            set_max_regs(2)
            walk = TileWalk()
            acc_state = txl.PipelineState(2, 0)
            quadrant = warp % 4
            col_half = (warp - FIRST_EPILOGUE_WARP) // 4  # column half (NUM_EPILOGUE_WARPS == 8) or 0
            acc_row = quadrant * 32 + lane
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tmem_base, s_tmem_addr.ptr_to([0]))
            acc_vals = txl.alloc_local((32,), "float32")
            words16 = txl.alloc_local((16,), "uint32")
            words8 = txl.alloc_local((8,), "uint32")
            in_words = txl.alloc_local((32,), "uint32")
            dst_slot = txl.local_scalar("int32")

            staging_base = (warp - FIRST_EPILOGUE_WARP) * STAGING_WORDS

            def staging_addr(row, vec, num_vecs):
                """
                Address of 16-byte vector vec of row row in this warp's staging tile (num_vecs
                vectors per row).
                """
                if USE_TMA_STORE and num_vecs == 4:
                    # 64-byte rows in the TMA 64B-swizzle pattern (16-byte chunk ^= (row / 2) % 4)
                    return s_staging.ptr_to([staging_base + row * 16 + ((vec ^ ((row // 2) % 4)) * 4)])
                if USE_TMA_STORE and num_vecs == 2:
                    # 32-byte rows in the TMA 32B-swizzle pattern (16-byte chunk ^= (row / 4) % 2)
                    return s_staging.ptr_to([staging_base + row * 8 + ((vec ^ ((row // 4) % 2)) * 4)])
                # rows of num_vecs 16-byte vectors packed into 128-byte lines, 128B-swizzled
                rows_per_line = 8 // num_vecs
                line = row // rows_per_line
                line_vec = (row % rows_per_line) * num_vecs + vec
                return s_staging.ptr_to([staging_base + line * 32 + ((line_vec ^ (line % 8)) * 4)])

            preact_base = (warp - FIRST_EPILOGUE_WARP) * 1024

            def preact_addr(row, vec):
                """
                Address of 16-byte vector vec of row row in this warp's 128B-swizzled preact tile.
                """
                line = row
                return s_preact.ptr_to([preact_base + line * 32 + ((vec ^ (line % 8)) * 4)])

            def prefetch_preact(tile_walk, chunk):
                """
                cp.async this warp's 32 rows x 32 (gate, up) words of preact for the tile of
                tile_walk, chunk chunk, into s_preact (rows past the group zero-filled; chunks
                past N skipped).
                """
                n_start = tile_walk.n_tile * TILE_N + col_half * (CHUNKS_PER_WARP * 32) + chunk * 32
                row_base = tile_walk.row_start + tile_walk.m_tile * CLUSTER_M + cta_rank * CTA_M + quadrant * 32
                with txl.if_(n_start < N), txl.then_():
                    for i in range(8):
                        row = i * 4 + lane // 8
                        vec = lane % 8
                        load_ok = row_base + row < tile_walk.row_end
                        txl.ptx[CP_ASYNC_16B](preact_addr(row, vec), params["preact"].ptr_to([txl.Select(load_ok, (row_base + row) * inter + n_start + vec * 4, 0)]), 16, txl.Select(load_ok, txl.uint32(16), txl.uint32(0)))
                txl.ptx["cp.async.commit_group"]()

            def store_staged(src_words, num_vecs, global_addr, row_ok, col_ok=None, tma_store=None, is_full=None):
                """
                This lane's row src_words[4 num_vecs] -> smem -> coalesced global rows (or, with
                tma_store and a full 32-row chunk, one TMA store of the staging tile).
                """
                words = txl.alloc_local((4,), "uint32")
                for i in range(num_vecs):
                    txl.ptx.st.shared.v4.b32(staging_addr(lane, i, num_vecs), src_words[4 * i], src_words[4 * i + 1], src_words[4 * i + 2], src_words[4 * i + 3])
                txl.cuda.warp_sync()
                rows_per_iter = 32 // num_vecs
                vec = lane % num_vecs

                def copy_out():
                    for i in range(num_vecs):
                        row = i * rows_per_iter + lane // num_vecs
                        txl.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], staging_addr(row, vec, num_vecs))
                        dst_ptr = global_addr(row, vec)  # warp-convergent (may shuffle)
                        with txl.if_(row_ok(row) if col_ok is None else (row_ok(row) & col_ok(vec))), txl.then_():
                            txl.ptx.st.global_.v4.b32(dst_ptr, words[0], words[1], words[2], words[3])
                if tma_store is None:
                    copy_out()
                else:
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.cuda.warp_sync()
                    with txl.if_(is_full):
                        with txl.then_():
                            with txl.if_(lane == 0), txl.then_():
                                tma_store(s_staging.ptr_to([staging_base]))
                                txl.ptx.cp.async_.bulk.commit_group()
                                # the staging tile is rewritten next: its source read must be done
                                txl.ptx.cp.async_.bulk.wait_group.read(0)
                        with txl.else_():
                            copy_out()
                txl.cuda.warp_sync()

            dpre_words = txl.alloc_local((32,), "uint32") if USE_STAGING else None
            alpha, limit = params.get("alpha"), params.get("limit")
            FULL_MASK = txl.uint32(0xFFFFFFFF)

            def shuffle_xor(val, lane_mask):
                """
                val of lane ^ lane_mask (fp32).
                """
                result = txl.local_scalar("float32")
                txl.ptx.shfl_sync.bfly.b32(result, val, txl.uint32(lane_mask), txl.uint32(31), FULL_MASK)
                return result

            next_in_words = txl.alloc_local((32,), "uint32") if (kind == "down_dgrad" and USE_SWAP_AB) else None

            def load_swap_preact(buf, tile_walk, chunk):
                """
                buf[j] = this lane's (gate, up) preact word of token chunk * 32 + j (0 past the
                group or past N).
                """
                feature = tile_walk.n_tile * CLUSTER_N + cta_rank * CTA_N + quadrant * 32 + lane
                row_base = tile_walk.row_start + tile_walk.m_tile * CLUSTER_M + chunk * 32
                for j in range(32):
                    txl.assign(buf[j], txl.uint32(0))
                    with txl.if_((feature < N) & (row_base + j < tile_walk.row_end) & (chunk * 32 < tile_walk.tile_rows)), txl.then_():
                        txl.ptx.ld.global_.u32(buf[j], params["preact"].ptr_to([(row_base + j) * inter + feature]))

            def swap_epilogue(walk, acc_state):
                """
                Swap-AB tile: TMEM lane = output feature, column = token. Every register column
                j is one token row; a warp's lanes cover 32 consecutive features, so each store
                is one coalesced row segment.
                """
                feature = walk.n_tile * CLUSTER_N + cta_rank * CTA_N + quadrant * 32 + lane
                feature_ok = feature < N
                tile_row0 = walk.row_start + walk.m_tile * CLUSTER_M
                is_even = (lane % 2) == 0
                bias_word = txl.local_scalar("uint32", init=txl.uint32(0))
                if has_bias and kind == "gu_fwd":
                    with txl.if_(feature_ok & is_even), txl.then_():
                        txl.ptx.ld.global_.u32(bias_word, params["bias"].ptr_to([walk.group * inter + feature // 2]))
                bias_val = txl.local_scalar("float32", init=txl.float32(0.0))
                if has_bias and kind == "down_fwd":
                    with txl.if_(feature_ok), txl.then_():
                        txl.ptx.ld.global_.u32(bias_word, params["bias"].ptr_to([(walk.group * N + feature) // 2]))
                    txl.assign(bias_val, txl.Select(is_even, bf16_lo(bias_word), bf16_hi(bias_word)))
                is_swapped = params["swapped"] != 0 if kind in ("gu_fwd", "down_dgrad") else None
                NUM_SWAP_CHUNKS = 8 // (NUM_EPILOGUE_WARPS // 4)
                if kind == "down_dgrad":
                    load_swap_preact(in_words, walk, col_half)
                with txl.serial(NUM_SWAP_CHUNKS) as chunk_iter:
                    chunk = chunk_iter * (NUM_EPILOGUE_WARPS // 4) + col_half
                    with txl.if_(chunk * 32 < walk.tile_rows), txl.then_():
                        if USE_SCATTER_STAGING:
                            scatter_base = (col_half * 2 + chunk_iter % 2) * 2048
                        tmem_lane_addr = tmem_base + txl.Cast("uint32", (quadrant * 32) << 16) + txl.Cast("uint32", acc_state.stage * CLUSTER_N + chunk * 32)
                        txl.ptx[TMEM_LD_32X32](*[acc_vals[i] for i in range(32)], tmem_lane_addr)
                        txl.ptx.tcgen05.wait__ld.sync.aligned()
                        row_base = tile_row0 + chunk * 32
                        if kind == "down_fwd":
                            txl.assign(dst_slot, 0)
                            with txl.if_(row_base + lane < walk.row_end), txl.then_():
                                txl.ptx.ld.global_.s32(dst_slot, params["slot_of"].ptr_to([row_base + lane]))
                        if kind == "down_dgrad":
                            # prefetch the next chunk's preact words while this one computes
                            load_swap_preact(next_in_words, walk, chunk + NUM_EPILOGUE_WARPS // 4)
                        for j in range(32):
                            row = row_base + j
                            store_ok = feature_ok & (row < walk.row_end)
                            if kind in ("gu_dgrad", "down_fwd"):
                                val_lo = txl.local_scalar("float32")
                                txl.assign(val_lo, acc_vals[j] + bias_val)
                                val_hi = shuffle_xor(val_lo, 1)
                                word = txl.local_scalar("uint32")
                                pack_bf16x2(word, val_lo, val_hi)
                                if kind == "gu_dgrad":
                                    out_ptr = params["dx"].ptr_to([row * (N // 2) + feature // 2])
                                else:
                                    dst_row = txl.local_scalar("int32")
                                    txl.ptx.shfl_sync.idx.b32(dst_row, dst_slot, txl.uint32(j), txl.uint32(31), FULL_MASK)
                                    out_ptr = params["out"].ptr_to([dst_row * (N // 2) + feature // 2])
                                if USE_SCATTER_STAGING:
                                    with txl.if_(is_even), txl.then_():
                                        txl.ptx.st.shared.u32(s_scatter.ptr_to([scatter_base + j * 64 + quadrant * 16 + lane // 2]), word)
                                else:
                                    with txl.if_(store_ok & is_even), txl.then_():
                                        txl.ptx.st.global_.u32(out_ptr, word)
                            elif kind == "gu_fwd":
                                lo_v = txl.local_scalar("float32")
                                txl.assign(lo_v, acc_vals[j])
                                hi_v = shuffle_xor(lo_v, 1)
                                gate, up = txl.local_scalar("float32"), txl.local_scalar("float32")
                                txl.assign(gate, txl.Select(is_swapped, hi_v, lo_v))
                                txl.assign(up, txl.Select(is_swapped, lo_v, hi_v))
                                if has_bias:
                                    txl.assign(gate, gate + bf16_lo(bias_word))
                                    txl.assign(up, up + bf16_hi(bias_word))
                                preact_word = txl.local_scalar("uint32")
                                pack_bf16x2(preact_word, gate, up)
                                h_val = txl.local_scalar("float32")
                                txl.assign(h_val, swiglu_act(bf16_lo(preact_word), bf16_hi(preact_word), clamped, alpha, limit))
                                h_pair = shuffle_xor(h_val, 2)
                                h_word = txl.local_scalar("uint32")
                                pack_bf16x2(h_word, h_val, h_pair)
                                with txl.if_(store_ok & is_even), txl.then_():
                                    txl.ptx.st.global_.u32(params["preact"].ptr_to([row * inter + feature // 2]), preact_word)
                                with txl.if_(store_ok & ((lane % 4) == 0)), txl.then_():
                                    txl.ptx.st.global_.u32(params["h"].ptr_to([row * (inter // 2) + feature // 4]), h_word)
                            else:  # down_dgrad: feature = i, dh = acc_vals[j]
                                gate, up = txl.local_scalar("float32"), txl.local_scalar("float32")
                                txl.assign(gate, bf16_lo(in_words[j]))
                                txl.assign(up, bf16_hi(in_words[j]))
                                dh = acc_vals[j]
                                one = txl.float32(1.0)
                                sig = txl.local_scalar("float32")
                                h_val = txl.local_scalar("float32")
                                if not clamped:
                                    txl.assign(sig, sigmoid(gate))
                                    grad_gate = dh * up * sig * (one + gate * (one - sig))
                                    grad_up = dh * gate * sig
                                    txl.assign(h_val, gate * sig * up)
                                else:
                                    gate_clamped, up_clamped = txl.local_scalar("float32"), txl.local_scalar("float32")
                                    txl.assign(gate_clamped, txl.Select(gate < limit, gate, limit))
                                    txl.assign(up_clamped, txl.Select(up < limit, txl.Select(up > -limit, up, -limit), limit))
                                    txl.assign(sig, sigmoid(alpha * gate_clamped))
                                    grad_gate = dh * (up_clamped + one) * (sig + alpha * gate_clamped * sig * (one - sig))
                                    grad_gate = txl.Select(gate <= limit, grad_gate, txl.float32(0.0))
                                    grad_up = txl.Select((up <= limit) & (up >= -limit), dh * gate_clamped * sig, txl.float32(0.0))
                                    txl.assign(h_val, (up_clamped + one) * gate_clamped * sig)
                                grad_gate_val, grad_up_val = txl.local_scalar("float32"), txl.local_scalar("float32")
                                txl.assign(grad_gate_val, grad_gate)
                                txl.assign(grad_up_val, grad_up)
                                dpre_word = txl.local_scalar("uint32")
                                pack_bf16x2(dpre_word, txl.Select(is_swapped, grad_up_val, grad_gate_val), txl.Select(is_swapped, grad_gate_val, grad_up_val))
                                h_pair = shuffle_xor(h_val, 1)
                                h_word = txl.local_scalar("uint32")
                                pack_bf16x2(h_word, h_val, h_pair)
                                with txl.if_(store_ok), txl.then_():
                                    txl.ptx.st.global_.u32(params["dpre"].ptr_to([row * inter + feature]), dpre_word)
                                with txl.if_(store_ok & is_even), txl.then_():
                                    txl.ptx.st.global_.u32(params["h"].ptr_to([row * (inter // 2) + feature // 2]), h_word)
                        if kind == "down_dgrad":
                            for j in range(32):
                                txl.assign(in_words[j], next_in_words[j])
                        if USE_SCATTER_STAGING:
                            # the 4 quadrant warps' features of each token are now one 256-byte
                            # row segment: each warp scatters 8 whole token rows
                            txl.ptx.bar.sync(txl.uint32(1 + col_half), txl.uint32(128))
                            feature0 = walk.n_tile * CLUSTER_N + cta_rank * CTA_N
                            for token_iter in range(8):
                                token = quadrant * 8 + token_iter
                                dst_row = txl.local_scalar("int32")
                                txl.ptx.shfl_sync.idx.b32(dst_row, dst_slot, txl.Cast("uint32", token), txl.uint32(31), FULL_MASK)
                                word_lo, word_hi = txl.local_scalar("uint32"), txl.local_scalar("uint32")
                                txl.ptx.ld.shared.v2.b32(word_lo, word_hi, s_scatter.ptr_to([scatter_base + token * 64 + 2 * lane]))
                                with txl.if_((row_base + token < walk.row_end) & (feature0 + 4 * lane < N)), txl.then_():
                                    txl.ptx.st.global_.v2.b32(params["out"].ptr_to([dst_row * (N // 2) + feature0 // 2 + 2 * lane]), word_lo, word_hi)

            if USE_PREACT_PREFETCH:
                # zero this warp's preact tile once (reads of chunks past N see initialized smem)
                for i in range(8):
                    txl.ptx.st.shared.v4.b32(s_preact.ptr_to([preact_base + i * 128 + lane * 4]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0))
                txl.cuda.warp_sync()
                prefetch_walk = TileWalk()
                with txl.if_(prefetch_walk.tile < total_tiles), txl.then_():
                    prefetch_walk.locate()
                    prefetch_preact(prefetch_walk, 0)
            with txl.while_(walk.tile < total_tiles):
                walk.locate()
                walk.fetch_next()
                range_token = profile_range_start("e-wait-acc")
                acc_pipe.full.wait(acc_state.stage, acc_state.phase)
                profile_range_end(range_token)
                body_token = profile_range_start("e-body")
                txl.ptx.tcgen05.fence__after_thread_sync()
                if USE_SWAP_AB:
                    swap_epilogue(walk, acc_state)
                else:
                    if not k_grouped:
                        out_row = walk.row_start + walk.m_tile * CLUSTER_M + cta_rank * CTA_M + acc_row
                        row_ok = out_row < walk.row_end
                        if kind == "down_fwd":
                            txl.assign(dst_slot, 0)
                            with txl.if_(row_ok), txl.then_():
                                txl.ptx.ld.global_.s32(dst_slot, params["slot_of"].ptr_to([out_row]))
                    else:
                        if HAS_FAR_MAIN_GRADS:
                            # address lane a = dpre lane ^ flip: rows 0-15 lane 0, 16-31 lane 1
                            stage_row = ((lane % 2) ^ params["flip"]) * 16 + lane // 2
                        else:
                            stage_row = (lane ^ params["flip"]) if kind == "gu_wgrad" else lane
                        out_row0 = walk.m_tile * CLUSTER_M + cta_rank * CTA_M + quadrant * 32
                    # runtime chunk loop: one copy of the chunk body keeps the epilogue in the I-cache
                    with txl.serial(CHUNKS_PER_WARP) as chunk:
                        col = col_half * (CHUNKS_PER_WARP * 32) + chunk * 32
                        n_start = walk.n_tile * TILE_N + col
                        if USE_PREACT_PREFETCH:
                            # this chunk's preact landed in s_preact: registers, then
                            # prefetch the next chunk (or the next tile's first)
                            txl.ptx["cp.async.wait_group"](0)
                            txl.cuda.warp_sync()
                            # (chunks past N were never prefetched: they read the zero-initialized
                            # tile, values unused; an n_start < N guard here makes ptxas spill)
                            for i in range(8):
                                txl.ptx.ld.shared.v4.b32(in_words[4 * i], in_words[4 * i + 1], in_words[4 * i + 2], in_words[4 * i + 3], preact_addr(lane, i))
                            txl.cuda.warp_sync()
                            with txl.if_(chunk + 1 < CHUNKS_PER_WARP):
                                with txl.then_():
                                    prefetch_preact(walk, chunk + 1)
                                with txl.else_():
                                    if USE_DYNAMIC_TILES:
                                        txl.assign(prefetch_walk.tile, walk.next_tile)
                                    else:
                                        prefetch_walk.advance()
                                    with txl.if_(prefetch_walk.tile < total_tiles), txl.then_():
                                        prefetch_walk.locate()
                                        prefetch_preact(prefetch_walk, 0)
                        with txl.if_((n_start < N) if TILE_N == CLUSTER_N else ((n_start < N) & (col < TILE_N))), txl.then_():
                            tmem_lane_addr = tmem_base + txl.Cast("uint32", (quadrant * 32) << 16) + txl.Cast("uint32", acc_state.stage * CLUSTER_N + col)
                            if TILE_N % 32 == 0:
                                txl.ptx[TMEM_LD_32X32](*[acc_vals[i] for i in range(32)], tmem_lane_addr)
                            else:
                                # a 240-column tile's last chunk: only 16 accumulator columns
                                with txl.if_(col + 32 <= TILE_N):
                                    with txl.then_():
                                        txl.ptx[TMEM_LD_32X32](*[acc_vals[i] for i in range(32)], tmem_lane_addr)
                                    with txl.else_():
                                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](*[acc_vals[i] for i in range(16)], tmem_lane_addr)
                                        for i in range(16, 32):
                                            txl.assign(acc_vals[i], txl.float32(0.0))
                            txl.ptx.tcgen05.wait__ld.sync.aligned()
                            if k_grouped:
                                # fp32 main grads: stage the 32x32 block, TMA-reduce-add it (OOB
                                # rows / columns are clipped by the tensor map)
                                stage_slot = quadrant * 2 + (chunk % 2)
                                with txl.if_(lane == 0), txl.then_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(1)
                                txl.cuda.warp_sync()
                                for i in range(8):
                                    txl.ptx.st.shared.v4.f32(s_reduce.ptr_to([stage_slot * 1024 + stage_row * 32 + ((i ^ (stage_row % 8)) * 4)]), acc_vals[4 * i], acc_vals[4 * i + 1], acc_vals[4 * i + 2], acc_vals[4 * i + 3])
                                txl.ptx.fence.proxy.async_.shared__cta()
                                txl.cuda.warp_sync()
                                with txl.if_(lane == 0), txl.then_():
                                    if HAS_FAR_MAIN_GRADS:
                                        for lane_half, reduce_map in ((0, maps["c"]), (1, maps["c2"])):
                                            txl.ptx["cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"](txl.address_of(reduce_map), txl.Cast("int32", n_start), txl.Cast("int32", out_row0 // 2), txl.Cast("int32", walk.group), s_reduce.ptr_to([stage_slot * 1024 + lane_half * 512]))
                                    elif kind == "gu_wgrad":
                                        txl.ptx["cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"](txl.address_of(maps["c"]), txl.Cast("int32", n_start), txl.int32(0), txl.Cast("int32", out_row0 // 2), txl.Cast("int32", walk.group), s_reduce.ptr_to([stage_slot * 1024]))
                                    else:
                                        txl.ptx["cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"](txl.address_of(maps["c"]), txl.Cast("int32", n_start), txl.Cast("int32", out_row0), txl.Cast("int32", walk.group), s_reduce.ptr_to([stage_slot * 1024]))
                                    txl.ptx.cp.async_.bulk.commit_group()
                                txl.cuda.warp_sync()
                            else:
                                if USE_STAGING:
                                    row_base = walk.row_start + walk.m_tile * CLUSTER_M + cta_rank * CTA_M + quadrant * 32
                                    row_in_group = lambda row: row_base + row < walk.row_end  # noqa: E731
                                    is_full_chunk = (row_base + 32 <= walk.row_end) if USE_TMA_STORE else None
                                    as_i32 = lambda value: txl.Cast("int32", value)  # noqa: E731

                                    def tma_store_fn(map_key, col0):
                                        """
                                        TMA store of the staging tile through maps[map_key] at
                                        column col0, or None without TMA stores.
                                        """
                                        if not USE_TMA_STORE:
                                            return None
                                        return lambda s_src: txl.ptx[TMA_STORE_2D](txl.address_of(maps[map_key]), as_i32(col0), as_i32(row_base), s_src)
                                    if kind == "down_dgrad":
                                        apply_swiglu_dgrad(params, in_words, acc_vals, dpre_words, words16, clamped, alpha, limit)
                                        # 240-column tiles: the last chunk holds 16 columns (i)
                                        is_partial = TILE_N % 32 != 0
                                        if USE_SPLIT_DPRE:
                                            for dpre_half in range(2):
                                                store_staged(OffsetView(dpre_words, 16 * dpre_half), 4, lambda row, vec, dpre_half=dpre_half: params["dpre"].ptr_to([(row_base + row) * inter + n_start + 16 * dpre_half + vec * 4]), row_in_group, (lambda vec, dpre_half=dpre_half: col + 16 * dpre_half + vec * 4 < TILE_N) if is_partial else None, tma_store_fn("d", 2 * (n_start + 16 * dpre_half)), is_full_chunk)
                                        else:
                                            store_staged(dpre_words, 8, lambda row, vec: params["dpre"].ptr_to([(row_base + row) * inter + n_start + vec * 4]), row_in_group, (lambda vec: col + vec * 4 < TILE_N) if is_partial else None, tma_store_fn("d", 2 * n_start), is_full_chunk)
                                        store_staged(words16, 4, lambda row, vec: params["h"].ptr_to([(row_base + row) * (inter // 2) + n_start // 2 + vec * 4]), row_in_group, (lambda vec: col + vec * 8 < TILE_N) if is_partial else None, tma_store_fn("e", n_start), is_full_chunk)
                                    elif kind == "gu_fwd":
                                        apply_gate_up_fwd(params, walk, n_start, acc_vals, words16, words8, in_words, has_bias, clamped, alpha, limit)
                                        store_staged(words16, 4, lambda row, vec: params["preact"].ptr_to([(row_base + row) * inter + n_start // 2 + vec * 4]), row_in_group, None, tma_store_fn("d", n_start), is_full_chunk)
                                        store_staged(words8, 2, lambda row, vec: params["h"].ptr_to([(row_base + row) * (inter // 2) + n_start // 4 + vec * 4]), row_in_group, None, tma_store_fn("e", n_start // 2), is_full_chunk)
                                    else:
                                        apply_bias_round(params, walk, n_start, acc_vals, words16, in_words, has_bias)
                                        # 240-column tiles: the last chunk holds 16 columns
                                        col_mask = None if TILE_N % 32 == 0 else (lambda vec: col + vec * 8 < TILE_N)
                                        if kind == "gu_dgrad":
                                            store_staged(words16, 4, lambda row, vec: params["dx"].ptr_to([(row_base + row) * (N // 2) + n_start // 2 + vec * 4]), row_in_group, col_mask, tma_store_fn("d", n_start), is_full_chunk)
                                        else:
                                            def out_addr(row, vec):
                                                """
                                                Output address of vector vec of row row: rows go to
                                                their slot_of slots.
                                                """
                                                dst_row = txl.local_scalar("int32")
                                                txl.ptx.shfl_sync.idx.b32(dst_row, dst_slot, txl.Cast("uint32", row), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                                                return params["out"].ptr_to([dst_row * (N // 2) + n_start // 2 + vec * 4])
                                            store_staged(words16, 4, out_addr, row_in_group, col_mask)

                txl.ptx.tcgen05.fence__before_thread_sync()
                txl.cuda.warp_sync()
                with txl.if_(lane == 0), txl.then_():
                    acc_pipe.empty.arrive(acc_state.stage, remote=0)
                profile_range_end(body_token)
                acc_state.advance()
                walk.advance()
            if k_grouped or USE_TMA_STORE:
                with txl.if_(lane == 0), txl.then_():
                    txl.ptx.cp.async_.bulk.wait_group(0)
                txl.cuda.warp_sync()

        cu, G = params["cu"], params["G"]
        a_map, b_map = maps["a"], maps["b"]
        cta = txl.cta_id()
        cta_rank = txl.cuda.cluster_cta_id("x")
        cluster_idx = cta // 2
        warp = txl.warp_id()
        lane = txl.lane_id()

        with txl.if_(warp == TMA_WARP), txl.then_():
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))

        smem = txl.smem_pool()
        s_tmem_addr = smem.alloc((1,), "uint32", align=8)
        s_prefix = smem.alloc((MAX_GROUPS + 1,), "int32", align=16)
        # full: CTA0's barrier collects both CTAs' TMA bytes (+ CTA1's gather relay);
        # empty / acc_full: per CTA, signaled by the multicast commit; acc_empty: CTA0's,
        # arrived by every epilogue warp of the pair
        ab_pipe = txl.Pipeline(smem, NUM_STAGES, full="tma", empty="tcgen05", init_full=1 + (1 if HAS_GATHER else 0))
        acc_pipe = txl.Pipeline(smem, 2, full="tcgen05", empty="mbar", init_empty=2 * NUM_EPILOGUE_WARPS)
        gather_full = txl.MBarrier(smem, NUM_STAGES)
        gather_full.init(32 * NUM_GATHER_WARPS)
        s_order = smem.alloc((MAX_GROUPS,), "int32", align=16)
        if USE_DYNAMIC_TILES:
            # tile-id ring: per CTA a full barrier (the local scheduler's expect_tx arrive + the
            # multicast CLC response bytes) and the 16-byte responses; CTA0's empty barrier
            # collects every consumer warp of the pair; drain_full / s_drain_resp: CTA0's private
            # drain slot
            slot_full = txl.MBarrier(smem, TILE_SLOTS)
            slot_full.init(1)
            slot_empty = txl.MBarrier(smem, TILE_SLOTS)
            slot_empty.init(NUM_SLOT_CONSUMERS)
            drain_full = txl.MBarrier(smem, 1)
            drain_full.init(1)
            s_clc_resp, s_drain_resp = smem.alloc((TILE_SLOTS * 4,), "uint32", align=16), smem.alloc((4,), "uint32", align=16)
        smem.pool.move_base_to(8192 + 4 * (MAX_GROUPS + 1))
        s_a = smem.alloc(a_shape, "bfloat16", swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM)
        s_b = smem.alloc(b_shape, "bfloat16", swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM)
        if k_grouped:
            # per epilogue warp, two 32x32 fp32 staging tiles (128-byte swizzled rows)
            s_reduce = smem.alloc((4 * 2 * 32 * 32,), "float32", align=1024)
        if USE_STAGING:
            # per epilogue warp: one 32-row x 128-byte transpose tile (16-byte chunks swizzled)
            s_staging = smem.alloc((NUM_EPILOGUE_WARPS * STAGING_WORDS,), "uint32", align=1024)
        if USE_PREACT_PREFETCH:
            s_preact = smem.alloc((NUM_EPILOGUE_WARPS * 1024,), "uint32", align=1024)
        if USE_SCATTER_STAGING:
            # per epilogue half, two 32-token x 128-feature bf16 tiles (64 u32 per token row)
            s_scatter = smem.alloc(((NUM_EPILOGUE_WARPS // 4) * 2 * 32 * 64,), "uint32", align=1024)

        # programmatic dependent launch: the smem-only prologue above overlaps the previous
        # kernel's tail; every global access of this grid follows its completion, and the next
        # kernel may start its own prologue once all of this grid's CTAs are resident
        if not USE_DYNAMIC_TILES:
            txl.ptx.griddepcontrol.wait()
        txl.ptx.griddepcontrol.launch_dependents()

        # the tile prefix over the groups, then TMEM for both accumulators (512 columns)
        with txl.if_(warp == PREFIX_WARP), txl.then_():
            compute_tile_prefix()
        with txl.if_(warp == MMA_WARP), txl.then_():
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](txl.address_of(s_tmem_addr[0]), txl.uint32(512))
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        cta0_full = ab_pipe.full.remote_view(0)
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.tcgen05.fence__before_thread_sync()
        txl.cuda.cluster_sync()
        txl.ptx.tcgen05.fence__after_thread_sync()

        with txl.if_((warp == PREFIX_WARP) | (warp == SCHEDULER_WARP)), txl.then_():
            set_max_regs(0)

        total_tiles = txl.local_scalar("int32")
        txl.ptx.ld.shared.s32(total_tiles, s_prefix.ptr_to([G]))

        if USE_DYNAMIC_TILES:
            with txl.if_((warp == SCHEDULER_WARP) & (lane == 0)), txl.then_():
                clc_scheduler_loop()
        # the TMA producer loads the operand(s) not gathered. With both gathered (far-pair
        # gate_up fwd) only CTA0's runs: its zero-byte arrive on the full barrier keeps the stage
        # protocol; CTA1's would wait on empty phases nothing gates.
        with txl.if_((warp == TMA_WARP) & (cta_rank == 0) if TMA_STAGE_BYTES == 0 else (warp == TMA_WARP)), txl.then_():
            tma_producer_loop()
        if HAS_GATHER:
            with txl.if_(warp >= FIRST_GATHER_WARP), txl.then_():
                gather_loop()
        if HAS_GATHER:
            with txl.if_((warp == MMA_WARP) & (cta_rank == 1)), txl.then_():
                gather_relay_loop()
        with txl.if_((warp == MMA_WARP) & (cta_rank == 0)), txl.then_():
            mma_loop()
        with txl.if_((warp >= FIRST_EPILOGUE_WARP) & (warp < FIRST_EPILOGUE_WARP + NUM_EPILOGUE_WARPS)), txl.then_():
            epilogue_loop()

        txl.ptx.tcgen05.fence__before_thread_sync()
        txl.cuda.cluster_sync()
        with txl.if_(warp == MMA_WARP), txl.then_():
            txl.ptx.tcgen05.fence__after_thread_sync()
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tmem_base, s_tmem_addr.ptr_to([0]))
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](tmem_base, txl.uint32(512))
        if USE_DYNAMIC_TILES:
            # completion order: this grid ends only after the previous one
            txl.ptx.griddepcontrol.wait()

    def apply_gate_up_fwd(params, walk, n_start, acc_vals, words16, words8, in_words, has_bias, clamped, alpha, limit):
        """
        gate_up fwd chunk: the preact words words16[16] ((gate, up) order, + bias) and the h
        words words8[8] of it.
        """
        is_swapped = params["swapped"] != 0
        if has_bias:
            for i in range(4):
                txl.ptx.ld.global_.v4.b32(in_words[4 * i], in_words[4 * i + 1], in_words[4 * i + 2], in_words[4 * i + 3], params["bias"].ptr_to([walk.group * inter + n_start // 2 + 4 * i]))
        for i in range(16):
            gate, up = txl.local_scalar("float32"), txl.local_scalar("float32")
            txl.assign(gate, txl.Select(is_swapped, acc_vals[2 * i + 1], acc_vals[2 * i]))
            txl.assign(up, txl.Select(is_swapped, acc_vals[2 * i], acc_vals[2 * i + 1]))
            if has_bias:
                txl.assign(gate, gate + bf16_lo(in_words[i]))
                txl.assign(up, up + bf16_hi(in_words[i]))
            pack_bf16x2(words16[i], gate, up)
        for i in range(8):
            h_lo, h_hi = txl.local_scalar("float32"), txl.local_scalar("float32")
            txl.assign(h_lo, swiglu_act(bf16_lo(words16[2 * i]), bf16_hi(words16[2 * i]), clamped, alpha, limit))
            txl.assign(h_hi, swiglu_act(bf16_lo(words16[2 * i + 1]), bf16_hi(words16[2 * i + 1]), clamped, alpha, limit))
            pack_bf16x2(words8[i], h_lo, h_hi)

    def apply_bias_round(params, walk, n_start, acc_vals, words16, in_words, has_bias):
        """
        down fwd / gate_up dgrad chunk: acc_vals (+ bias) rounded to bf16 words words16[16].
        """
        if has_bias:
            for i in range(4):
                # 8 columns per load; a 240-column tile's last chunk may end past N
                bias_ok = n_start + 8 * i < N
                txl.ptx.ld.global_.v4.b32(in_words[4 * i], in_words[4 * i + 1], in_words[4 * i + 2], in_words[4 * i + 3], params["bias"].ptr_to([txl.Select(bias_ok, (walk.group * N + n_start) // 2 + 4 * i, 0)]))
        for i in range(16):
            val_lo, val_hi = txl.local_scalar("float32"), txl.local_scalar("float32")
            txl.assign(val_lo, acc_vals[2 * i])
            txl.assign(val_hi, acc_vals[2 * i + 1])
            if has_bias:
                txl.assign(val_lo, val_lo + bf16_lo(in_words[i]))
                txl.assign(val_hi, val_hi + bf16_hi(in_words[i]))
            pack_bf16x2(words16[i], val_lo, val_hi)

    def apply_swiglu_dgrad(params, in_words, acc_vals, dpre_words, words16, clamped, alpha, limit):
        """
        dpre words dpre_words[32] (weight address order) and h words words16[16] of one 32-column
        chunk: in_words = 32 (gate, up) preact words, acc_vals = 32 fp32 dh. One sigmoid per
        element serves both the gradient and h (h keeps the forward's exact expression order).
        """
        is_swapped = params["swapped"] != 0
        h_vals = txl.alloc_local((32,), "float32")
        one = txl.float32(1.0)
        for i in range(32):
            gate, up = txl.local_scalar("float32"), txl.local_scalar("float32")
            txl.assign(gate, bf16_lo(in_words[i]))
            txl.assign(up, bf16_hi(in_words[i]))
            dh = acc_vals[i]
            sig = txl.local_scalar("float32")
            if not clamped:
                txl.assign(sig, sigmoid(gate))
                grad_gate = dh * up * sig * (one + gate * (one - sig))
                grad_up = dh * gate * sig
                txl.assign(h_vals[i], gate * sig * up)
            else:
                gate_clamped, up_clamped = txl.local_scalar("float32"), txl.local_scalar("float32")
                txl.assign(gate_clamped, txl.Select(gate < limit, gate, limit))
                txl.assign(up_clamped, txl.Select(up < limit, txl.Select(up > -limit, up, -limit), limit))
                txl.assign(sig, sigmoid(alpha * gate_clamped))
                grad_gate = dh * (up_clamped + one) * (sig + alpha * gate_clamped * sig * (one - sig))
                grad_gate = txl.Select(gate <= limit, grad_gate, txl.float32(0.0))
                grad_up = txl.Select((up <= limit) & (up >= -limit), dh * gate_clamped * sig, txl.float32(0.0))
                txl.assign(h_vals[i], (up_clamped + one) * gate_clamped * sig)
            grad_gate_val, grad_up_val = txl.local_scalar("float32"), txl.local_scalar("float32")
            txl.assign(grad_gate_val, grad_gate)
            txl.assign(grad_up_val, grad_up)
            pack_bf16x2(dpre_words[i], txl.Select(is_swapped, grad_up_val, grad_gate_val), txl.Select(is_swapped, grad_gate_val, grad_up_val))
        for i in range(16):
            pack_bf16x2(words16[i], h_vals[2 * i], h_vals[2 * i + 1])

    def entry(*args):
        params = dict(zip(param_names, args, strict=True))
        maps = host_prelude(params)
        # 2-CTA clusters; programmatic stream serialization (PDL) on every launch
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=params["num_ctas"], block=NUM_WARPS * 32, cluster=[2, 1], preferred_cluster=[2, 1], programmatic_stream_serialization=True), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        body(params, maps)

    entry.__name__ = f"bf16_expert_{kind}"
    entry.__signature__ = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation) for name, annotation in param_specs])
    traced = txl.kernel()(entry)
    if USE_IKET:
        from tvm_tirx.backend.cuda.iket import IketProfiler
        with TARGET:
            return IketProfiler().compile(traced.mod, target=TARGET)
    return entry_function(traced.compile(TARGET))


@functools.cache
def build_row_gather(H):
    """
    out[r, :] = src[idx[r], :] for r < M: one warp per row, 16 bytes per lane.
    """
    NUM_VECS = H // 8  # 16-byte vectors per row
    WARPS_PER_CTA = 8

    def gather_rows(src: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), idx: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), M: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=WARPS_PER_CTA * 32, programmatic_stream_serialization=True))
        warp = txl.warp_id()
        lane = txl.lane_id()
        row_idx = txl.local_scalar("int32", init=txl.cta_id() * WARPS_PER_CTA + warp)
        src_row = txl.local_scalar("int32")
        buf = txl.alloc_local((4 * (-(-NUM_VECS // 32)),), "uint32")
        txl.ptx.griddepcontrol.wait()
        txl.ptx.griddepcontrol.launch_dependents()
        with txl.while_(row_idx < M):
            txl.ptx.ld.global_.s32(src_row, idx.ptr_to([row_idx]))
            for i in range(-(-NUM_VECS // 32)):
                vec = i * 32 + lane
                with txl.if_(vec < NUM_VECS), txl.then_():
                    txl.ptx.ld.global_.nc.v4.b32(buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3], src.ptr_to([src_row * (H // 2) + vec * 4]))
            for i in range(-(-NUM_VECS // 32)):
                vec = i * 32 + lane
                with txl.if_(vec < NUM_VECS), txl.then_():
                    txl.ptx.st.global_.v4.b32(out.ptr_to([row_idx * (H // 2) + vec * 4]), buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3])
            txl.assign(row_idx, row_idx + num_ctas * WARPS_PER_CTA)

    return entry_function(txl.kernel()(gather_rows).compile(TARGET))


def gather_expert_rows(src, idx):
    """
    Contiguous src[idx] (M, H) through the TIRx-lite row-gather kernel.
    """
    M, H = idx.shape[0], src.shape[1]
    out = src.new_empty(M, H)
    num_ctas = max(1, min(4 * num_sms(), -(-M // 8)))
    build_row_gather(H)(src.reshape(-1).view(torch.uint32), idx, out.reshape(-1).view(torch.uint32), M, num_ctas)
    return out


# Host side: grid sizing, overlap tracking and the launchers.


@functools.cache
def num_sms():
    """
    SM count of the current device.
    """
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def flat_view(tensor):
    """
    A contiguous 1-D view over tensor's storage from its first element to its last.
    """
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return tensor.as_strided((span,), (1,))


TMA_STRIDE_LIMIT = 1 << 40  # cuTensorMapEncodeTiled: every global stride below 2^40 bytes


def order_pair(gate, up):
    """
    (lo, hi, diff_elems, swapped, far) of a gate / up pair: swapped when up is the lower
    address; far when the two buffers (independent allocations) are too far apart for one tensor
    map to stride between them.
    """
    if gate.stride() != up.stride() or gate.stride(-1) != 1:
        raise NotImplementedError("the gate / up pair must share strides, unit-stride rows")
    swapped = up.data_ptr() < gate.data_ptr()
    lo, hi = (up, gate) if swapped else (gate, up)
    nbytes = hi.data_ptr() - lo.data_ptr()
    if nbytes % 16 or lo.data_ptr() % 16:
        raise NotImplementedError("the gate / up pair must be 16-byte aligned, a multiple of 16 bytes apart")
    return lo, hi, nbytes // hi.element_size(), swapped, nbytes >= TMA_STRIDE_LIMIT


def grid_ctas(M, G, k_grouped, per_group_tiles, num_n_tiles, dyn=False):
    """
    CTA count: two per cluster, at most one cluster per SM pair or per tile.
    """
    if G > MAX_GROUPS:
        raise NotImplementedError(f"at most {MAX_GROUPS} experts per call (smem tile prefix), got {G}")
    if k_grouped:
        max_clusters = G * per_group_tiles
    else:
        max_clusters = (-(-M // CLUSTER_M) + G) * num_n_tiles
    if dyn:
        return 2 * max(1, max_clusters)
    return 2 * max(1, min(num_sms() // 2, max_clusters))


# Kernel overlap (the wgrads): every launch here uses programmatic dependent launch and triggers
# its dependents right after its prologue, and every launch that waits (griddepcontrol.wait) does
# so before its first global access. The overlap variants skip that wait and wait at exit
# instead, so they may run alongside every launch since (and including) the last waiting one on
# the stream: OVERLAP_WINDOW holds those launches' byte extents (reads, writes), and an overlap variant
# is chosen only if it neither reads what they write nor writes what they touch. Ordinary
# (non-PDL) stream work in between only adds ordering. Assumes no foreign kernel that triggers
# its dependents early sits right before an overlap launch.
OVERLAP_WINDOW = {}


def byte_extents(*tensors):
    """
    Byte extents [lo, hi) of tensors' storage spans (empty tensors skipped).
    """
    extents = []
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        extents.append((tensor.data_ptr(), tensor.data_ptr() + span * tensor.element_size()))
    return extents


def extents_overlap(first, second):
    """
    Some extent of first intersects some extent of second.
    """
    return any(first_lo < second_hi and second_lo < first_hi for first_lo, first_hi in first for second_lo, second_hi in second)


def stream_key():
    """
    OVERLAP_WINDOW key of the current stream.
    """
    stream = torch.cuda.current_stream()
    return stream.device_index, stream.cuda_stream  # per (device, stream): stream handles may repeat across devices


def may_overlap(reads, writes):
    """
    An overlap launch with these extents is independent of every launch that may still run.
    """
    window = OVERLAP_WINDOW.get(stream_key())
    if not window or len(window) >= 4:
        return False
    return not any(extents_overlap(writes, window_reads) or extents_overlap(writes, window_writes) or extents_overlap(reads, window_writes) for window_reads, window_writes in window)


def track_launch(reads, writes, overlapped):
    """
    Record a launch's extents: an overlap launch joins the window, any other starts a new one.
    """
    key = stream_key()
    if overlapped:
        OVERLAP_WINDOW[key].append((reads, writes))
    else:
        OVERLAP_WINDOW[key] = [(reads, writes)]


def as_u32(tensor):
    """
    bf16 tensor -> flat uint32 view of the same bytes.
    """
    return tensor.reshape(-1).view(torch.uint32)


def check_shapes(H, inter):
    """
    The tile shapes need H a multiple of 64 and I a multiple of 32.
    """
    assert H % 64 == 0 and inter % 32 == 0, (H, inter)


def bf16_expert_gate_up_fwd_sm100(x, w_gate, w_up, cu, a_idx, bias=None, act=()):
    """
    SM100 implementation of bf16_expert_gate_up_fwd.
    """
    M, G, inter, H = a_idx.shape[0], w_gate.shape[0], w_gate.shape[1], w_gate.shape[2]
    preact, h = x.new_empty(M, 2 * inter), x.new_empty(M, inter)
    if M == 0:
        return preact, h
    check_shapes(H, inter)
    lo, hi, pair_diff, swapped, far = order_pair(w_gate, w_up)
    clamped = bool(act)
    materialized = use_materialized_gu_fwd(M, G, H, inter)
    fn = build_expert_gemm("gu_fwd", H, inter, bias is not None, clamped, has_small_groups(M, G, SWAP_ROWS_GU_FWD) and has_swap_ab("gu_fwd", inter), materialized, far)
    num_n_tiles = -(-2 * inter // CLUSTER_N)
    x_rows = gather_expert_rows(x, a_idx) if materialized else x
    args = [cu, G, M, grid_ctas(M, G, False, 0, num_n_tiles), x_rows.reshape(-1), x_rows.shape[0], a_idx, flat_view(lo), pair_diff, lo.stride(0), lo.stride(1), as_u32(preact), as_u32(h), int(swapped)]
    if bias is not None:
        args.append(as_u32(bias.contiguous()))
    if clamped:
        args += [float(act[0]), float(act[1])]
    fn(*args)
    track_launch(byte_extents(x, x_rows, lo, hi, cu, a_idx, bias), byte_extents(preact, h), False)
    return preact, h


def bf16_expert_down_fwd_sm100(h, w_down, cu, slot_of, bias=None):
    """
    SM100 implementation of bf16_expert_down_fwd.
    """
    M, G, H, inter = h.shape[0], w_down.shape[0], w_down.shape[1], w_down.shape[2]
    out = h.new_empty(M, H)
    if M == 0:
        return out
    check_shapes(H, inter)
    use_swap_ab = has_small_groups(M, G) and has_swap_ab("down_fwd", inter)
    fn = build_expert_gemm("down_fwd", H, inter, bias is not None, False, use_swap_ab)
    num_n_tiles = -(-H // cluster_tile_n("down_fwd", H, use_swap_ab, False))
    args = [cu, G, M, grid_ctas(M, G, False, 0, num_n_tiles), h.reshape(-1), w_down.reshape(-1), slot_of, as_u32(out)]
    if bias is not None:
        args.append(as_u32(bias.contiguous()))
    fn(*args)
    track_launch(byte_extents(h, w_down, cu, slot_of, bias), byte_extents(out), False)
    return out


def bf16_expert_down_dgrad_sm100(dy, w_down, preact, cu, slot_of, w_gate, w_up, act=()):
    """
    SM100 implementation of bf16_expert_down_dgrad.
    """
    M, inter = preact.shape[0], preact.shape[1] // 2
    G, H = w_down.shape[0], w_down.shape[1]
    swapped = w_up.data_ptr() < w_gate.data_ptr()
    dpre, h = preact.new_empty(M, 2 * inter), preact.new_empty(M, inter)
    if M == 0:
        return dpre, h, swapped
    check_shapes(H, inter)
    clamped = bool(act)
    fn = build_expert_gemm("down_dgrad", H, inter, False, clamped)
    num_n_tiles = -(-inter // cluster_tile_n("down_dgrad", inter, False, False))
    args = [cu, G, M, grid_ctas(M, G, False, 0, num_n_tiles), dy.reshape(-1), dy.shape[0], slot_of, w_down.reshape(-1), as_u32(preact), as_u32(dpre), as_u32(h), int(swapped)]
    if clamped:
        args += [float(act[0]), float(act[1])]
    fn(*args)
    track_launch(byte_extents(dy, w_down, preact, cu, slot_of), byte_extents(dpre, h), False)
    return dpre, h, swapped


def bf16_expert_gate_up_dgrad_sm100(dpre, w_gate, w_up, cu, dpre_swapped):
    """
    SM100 implementation of bf16_expert_gate_up_dgrad.
    """
    M, inter = dpre.shape[0], dpre.shape[1] // 2
    G, H = w_gate.shape[0], w_gate.shape[2]
    dx = dpre.new_empty(M, H)
    if M == 0:
        return dx
    check_shapes(H, inter)
    lo, hi, pair_diff, swapped, far = order_pair(w_gate, w_up)
    if bool(swapped) != bool(dpre_swapped):
        raise NotImplementedError("dpre pair order must follow the weights' address order")
    use_swap_ab = has_small_groups(M, G) and has_swap_ab("gu_dgrad", inter)
    fn = build_expert_gemm("gu_dgrad", H, inter, False, False, use_swap_ab, False, far)
    num_n_tiles = -(-H // cluster_tile_n("gu_dgrad", H, use_swap_ab, far))
    args = [cu, G, M, grid_ctas(M, G, False, 0, num_n_tiles), dpre.reshape(-1), flat_view(lo), pair_diff, lo.stride(0), lo.stride(1), as_u32(dx)]
    fn(*args)
    track_launch(byte_extents(dpre, lo, hi, cu), byte_extents(dx), False)
    return dx


def bf16_expert_gate_up_wgrad_sm100(x, dpre, dpre_swapped, mg_gate, mg_up, cu, a_idx):
    """
    SM100 implementation of bf16_expert_gate_up_wgrad.
    """
    M = a_idx.shape[0]
    if M == 0:
        return None
    G, inter, H = mg_gate.shape
    check_shapes(H, inter)
    lo, hi, pair_diff, mg_swapped, far = order_pair(mg_gate, mg_up)
    reads, writes = byte_extents(x, dpre, cu, a_idx), byte_extents(lo, hi)
    overlap = may_overlap(reads, writes)
    fn = build_expert_gemm("gu_wgrad", H, inter, False, False, False, False, far, overlap)
    num_m_tiles, num_n_tiles = -(-2 * inter // CLUSTER_M), -(-H // CLUSTER_N)
    # dpre lane l is the gate iff l == dpre_swapped; main-grad address lane a is the gate iff
    # a == mg_swapped, so a = l ^ flip
    flip = int(bool(dpre_swapped)) ^ int(mg_swapped)
    args = [cu, G, M, grid_ctas(M, G, True, num_m_tiles * num_n_tiles, num_n_tiles, overlap), dpre.reshape(-1), x.reshape(-1), x.shape[0], a_idx, flat_view(lo)] + ([flat_view(hi)] if far else []) + [pair_diff, lo.stride(0), lo.stride(1), flip]
    fn(*args)
    track_launch(reads, writes, overlap)
    return None


def bf16_expert_down_wgrad_sm100(dy, h, mg_down, cu, slot_of):
    """
    SM100 implementation of bf16_expert_down_wgrad.
    """
    M = slot_of.shape[0]
    if M == 0:
        return None
    G, H, inter = mg_down.shape
    assert mg_down.is_contiguous()
    check_shapes(H, inter)
    reads, writes = byte_extents(dy, h, cu, slot_of), byte_extents(mg_down)
    overlap = may_overlap(reads, writes)
    fn = build_expert_gemm("down_wgrad", H, inter, False, False, False, False, False, overlap)
    num_m_tiles, num_n_tiles = -(-H // CLUSTER_M), -(-inter // CLUSTER_N)
    args = [cu, G, M, grid_ctas(M, G, True, num_m_tiles * num_n_tiles, num_n_tiles, overlap), dy.reshape(-1), dy.shape[0], slot_of, h.reshape(-1), mg_down.reshape(-1)]
    fn(*args)
    track_launch(reads, writes, overlap)
    return None
