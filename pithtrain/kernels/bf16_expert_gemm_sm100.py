"""
SM100 TIRx-lite kernels for pith-train's fused BF16 expert grouped GEMMs.

One persistent, warp-specialized tcgen05 grouped-GEMM template covers the six entry points:

* M-grouped (rows of expert g are cu[g] <= r < cu[g+1], fixed K): gate_up_fwd, down_fwd,
  down_dgrad, gate_up_dgrad.
* K-grouped (expert g reduces over its rows, fixed output): gate_up_wgrad, down_wgrad.

Persistent 2-CTA clusters (tcgen05.mma.cta_group::2, 256 x 256 x 64 tiles, two TMEM
accumulators). Warpgroup 0: TMA producer, MMA issuer (CTA0) / gather relay (CTA1), tile-prefix
warp; then 4 or 8 epilogue warps; then 4, 8 or 16 cp.async row-gather warps for a_idx / slot_of
operands.
Small expert groups (mean rows <= SWAP_ROWS) run the M-grouped GEMMs swap-AB: weights on the MMA
M side, the group's tokens on a runtime N (<= 256, steps of 32), so a group tail wastes < 32 rows.
The wgrads gather their rows in-kernel and accumulate the fp32 main grads by TMA reduce-add.

The two gate / up weights are read through one rank-4 tensor map (H, 2, I, G) whose second mode
strides between the two weights in address order, so the MMA sees the interleaved
[w_lo[0]; w_hi[0]; w_lo[1]; ...] rows: the accumulator columns are the interleaved preact
pairs, and the gate_up dgrad contracts the interleaved dpre directly. TMA strides stop below
2^40 bytes, so two independent allocations farther apart than that (separate parameters in
training) take the far-pair variants instead: the gate_up fwd / dgrad weight rows come in through
the cp.async gather warps with 64-bit addresses (same interleaved smem rows, same epilogues), and
the gate_up wgrad reduces each main grad through its own tensor map.

Kernel overlap: every launch uses programmatic dependent launch. The wgrads, when the host finds
them independent of every launch that may still run on the stream (byte extents), take an
overlap variant: cluster-launch-control dynamic tile scheduling and no griddepcontrol.wait before
the main loop, so their clusters fill the SMs the previous kernel frees in its tail; they wait for
the previous grid at exit, so completion order is unchanged.
"""

import functools
import inspect

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["bf16_expert_gate_up_fwd_sm100", "bf16_expert_down_fwd_sm100", "bf16_expert_down_dgrad_sm100", "bf16_expert_gate_up_dgrad_sm100", "bf16_expert_gate_up_wgrad_sm100", "bf16_expert_down_wgrad_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(ex):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(ex)
    return ex["main"]


BM = 128  # rows per CTA; a 2-CTA cluster covers BMC = 256
BMC = 256
BK = 64
BN = 256  # cluster N; each CTA stages HN = 128 of B
HN = 128
STAGES = 6
GMAX = 1024
# warpgroup 0: TMA producer, MMA issuer (CTA1: gather relay), tile-prefix warp, idle;
# then the epilogue warpgroup(s), then the cp.async gather warpgroup
EPI_WARP0 = 4
PREFIX_WARP = 2
IKET = False
SLOTS = 2  # tile-id ring depth

MMA = "tcgen05.mma.cta_group::2.kind::f16"
TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMA = "cp.async.bulk.tensor.{}d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
CP16 = "cp.async.cg.shared.global.L2::128B"
TMA_ST2 = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
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


def range_v(name):
    tok = txl.alloc_local([1], "uint32")
    txl.assign(tok[0], txl.cuda.iket.range_start(name))
    return tok


def end(tok):
    txl.cuda.iket.range_end(tok[0])


class Off:
    """
    Read-only view src[off + i] of a register array.
    """

    def __init__(self, arr, off):
        self.arr, self.off = arr, off

    def __getitem__(self, i):
        return self.arr[self.off + i]


def sigmoid(x):
    """
    1 / (1 + 2^(-x log2 e)) in fp32.
    """
    e = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(e, x * txl.float32(-LOG2E))
    r = txl.local_scalar("float32")
    txl.ptx.rcp.approx.ftz.f32(r, txl.float32(1.0) + e)
    return r


def bf16_lo(word):
    return txl.reinterpret(word << txl.uint32(16), ty="float32")


def bf16_hi(word):
    return txl.reinterpret(word & txl.uint32(0xFFFF0000), ty="float32")


def pack(dst, lo, hi):
    txl.ptx["cvt.rn.bf16x2.f32"](dst, hi, lo)


def act2(g, u, clamped, alpha, limit):
    """
    h of the bf16-rounded gate / up (fp32 values).
    """
    if not clamped:
        return g * sigmoid(g) * u
    gc = txl.Select(g < limit, g, limit)
    uc = txl.Select(u < limit, txl.Select(u > -limit, u, -limit), limit)
    return (uc + txl.float32(1.0)) * gc * sigmoid(alpha * gc)


def params2(kind, has_bias, clamped, far=False):
    i32, i64, f32 = "int32", "int64", "float32"
    # a fresh Tensor type (own symbolic extent) per parameter: a shared one would make the host
    # check require equal lengths
    def bf():
        return txl.Tensor((txl.int64(),), txl.bf16, data_alignment=1)

    def u32():
        return txl.Tensor((txl.int64(),), txl.u32, data_alignment=1)

    p = [("cu", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("G", i32), ("M", i32), ("num_ctas", i32)]
    if kind == "gu_fwd":
        p += [("x", bf()), ("T", i32), ("a_idx", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("w", bf()), ("wdiff", i64),
              ("ws0", i64), ("ws1", i64), ("preact", u32()), ("h", u32()), ("swapped", i32)]
        if has_bias:
            p += [("bias", u32())]
    elif kind == "down_fwd":
        p += [("hin", bf()), ("wd", bf()), ("slot_of", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("out", u32())]
        if has_bias:
            p += [("bias", u32())]
    elif kind == "down_dgrad":
        p += [("dy", bf()), ("T", i32), ("slot_of", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("wd", bf()),
              ("preact", u32()), ("dpre", u32()), ("h", u32()), ("swapped", i32)]
    elif kind == "gu_dgrad":
        p += [("dpre", bf()), ("w", bf()), ("wdiff", i64), ("ws0", i64), ("ws1", i64), ("dx", u32())]
    elif kind == "gu_wgrad":
        p += [("dpre", bf()), ("x", bf()), ("T", i32), ("a_idx", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)),
              ("mg", txl.Tensor((txl.int64(),), txl.f32, data_alignment=1))] + ([("mg2", txl.Tensor((txl.int64(),), txl.f32, data_alignment=1))] if far else []) + [
              ("mgdiff", i64), ("ms0", i64), ("ms1", i64), ("flip", i32)]
    elif kind == "down_wgrad":
        p += [("dy", bf()), ("T", i32), ("slot_of", txl.Tensor((txl.int64(),), txl.i32, data_alignment=1)), ("hin", bf()),
              ("mg", txl.Tensor((txl.int64(),), txl.f32, data_alignment=1))]
    if clamped and kind in ("gu_fwd", "down_dgrad"):
        p += [("alpha", f32), ("limit", f32)]
    return p


SWAP_ROWS = 512  # swap-AB when the mean rows per expert is at most this (tail waste dominates)


def swap_ok(kind, inter):
    """
    Kinds whose swap-AB variant measured faster for small groups: the gate_up GEMMs, and
    down_fwd when its K = I is long enough to hide the row-scatter epilogue.
    """
    return kind in ("gu_fwd", "gu_dgrad") or (kind == "down_fwd" and inter >= 2048)


def bnt(kind, N, swap, far):
    """
    Cluster tile N: 240 where N is a multiple of 240 but not of 256 (gpt-oss 2880). down_fwd
    only (K-major B): the MN-major-B kinds (gu_dgrad, down_dgrad) measured 6-11% slower with
    120-column B halves per CTA.
    """
    if kind == "down_fwd" and not swap and not far and N % 256 and N % 240 == 0:
        return 240
    return BN


# gate_up fwd: non-swap measured faster at 512 rows/expert (q35 EP4/4k 134 -> 129 us)
SWAP_ROWS_GU_FWD = 448


def use_swap(M, G, rows=SWAP_ROWS):
    return M <= rows * G


def use_materialized_gu_fwd(M, G, H, inter):
    """
    The materialized gather plus pure-TMA GEMM is faster only for GPT-OSS forward rows at >= 4096 rows
    per expert.
    """
    return H == 2880 and inter == 2880 and M >= 1024 * G


@functools.cache
def build(kind, H, inter, has_bias, clamped, swap=False, materialized=False, far=False, dyn=False):
    k_grouped, a_major, b_major, a_gather = KINDS[kind]
    # Problem geometry per kind: (K, N) for M-grouped; (Mo, No) for K-grouped.
    if kind == "gu_fwd":
        K, N = H, 2 * inter
    elif kind == "down_fwd":
        K, N = inter, H
    elif kind == "down_dgrad":
        K, N = H, inter
    elif kind == "gu_dgrad":
        K, N = 2 * inter, H
    elif kind == "gu_wgrad":
        MO, N = 2 * inter, H
    else:
        MO, N = H, inter
    # gpt-oss N = 2880 = 11.25 x 256: uniform 240-column tiles (12 x 240) instead of 11 full
    # tiles and a quarter tile that costs a full one (non-swap, fast-path plain epilogues only)
    BNT = bnt(kind, N, swap and swap_ok(kind, inter), far)
    HNT = BNT // 2
    NT = -(-N // BNT)
    KT = None if k_grouped else -(-K // BK)
    MT = -(-MO // BMC) if k_grouped else None
    A_BYTES = BM * BK * 2
    B_BYTES = (HNT if b_major == "k" else HN) * BK * 2
    # For large GPT-OSS forward rows, a separate coalesced gather followed by a pure-TMA
    # gate/up GEMM is faster than issuing the indexed cp.async gather inside every K stage.
    # Keep the indexed cp.async path for every other kind and shape.
    MAT_GU = materialized and kind == "gu_fwd"
    if MAT_GU:
        a_gather = False
    # far-pair gate / up weights (more than 2^40 bytes apart): no tensor map spans both, so the
    # gather warps load the interleaved weight rows (B) with plain 64-bit addresses
    WFAR = far and kind in ("gu_fwd", "gu_dgrad")
    MG2 = far and kind == "gu_wgrad"  # far-pair main grads: one reduce tensor map per buffer
    # the gathered operand(s): A for gu_fwd / down_dgrad / down_wgrad, B for gu_wgrad and WFAR
    A_GATH = kind in ("gu_fwd", "down_dgrad", "down_wgrad") and not MAT_GU
    B_GATH = kind == "gu_wgrad" or WFAR
    GATHER = A_GATH or B_GATH
    TMA_BYTES = (0 if A_GATH else A_BYTES) + (0 if B_GATH else B_BYTES)
    # the SwiGLU-heavy epilogues get two warps per TMEM lane quadrant (column halves)
    NEPI = 8 if kind in ("gu_fwd", "down_dgrad") else 4
    # longest-group-first tile order (helps down_wgrad's tail; measured worse for gu_wgrad)
    LPT = kind == "down_wgrad"
    # the wgrads read rows a gather kernel already laid out contiguously (row r is row r)
    PREGATHERED = False  # wgrads gather x[a_idx] / dy[slot_of] rows in-kernel (no gather pass)
    CH = (BN // 32) // (NEPI // 4)  # 32-column chunks per epilogue warp
    GW0 = EPI_WARP0 + NEPI
    # gather warps: more warps keep more cp.async row segments in flight. gu_fwd: 8; wgrads: 16
    # for small I (qwen-like, measured -8..-13 us per wgrad vs 8), 8 for I = 2880 (16 measured
    # +5..15 us there); down_dgrad and the far-pair weight gathers keep 4 (register budget)
    NGW = 4
    if GATHER and not WFAR and kind in ("gu_fwd", "gu_wgrad", "down_wgrad"):
        NGW = 16 if (k_grouped and inter <= 1024) else 8
    RPW = BM // NGW  # K-major A: tile rows per gather warp
    KPW = BK // NGW  # MN-major: K rows per gather warp
    NWARPS = GW0 + (NGW if GATHER else 0)
    REGS = (56, 80, 184) if kind == "down_dgrad" else None
    # wgrads keep 32 KB of smem for the fp32 TMA-reduce staging of the epilogue
    # wgrads keep 32 KB for the fp32 reduce staging, down_dgrad 32 KB for its coalescing
    # transpose buffers
    # swap-AB (weights on the MMA M side, tokens on a runtime N <= 256 in steps of 32): the
    # gate_up GEMMs always; down_fwd when its K = I is long enough to hide the scatter epilogue
    SWAP = swap and swap_ok(kind, inter)
    XSTAGE = not k_grouped and not SWAP
    # per-warp transpose tile (u32 words); the plain-SwiGLU down_dgrad stores dpre in two 64-byte
    # halves through a 2 KB tile to afford a 5th stage (the clamped one spills with that split)
    SPLIT_DPRE = kind == "down_dgrad" and not clamped
    XWORDS = 1024 if (kind == "down_dgrad" and not SPLIT_DPRE) else 512
    NST = STAGES - 1 if k_grouped else ((STAGES - 1 if SPLIT_DPRE else STAGES - 2) if (kind == "down_dgrad" and not SWAP) else STAGES)
    # down_dgrad prefetches the next preact chunk by cp.async into a per-warp smem tile
    PF = kind == "down_dgrad" and not SWAP
    # M-grouped bf16 outputs (down_dgrad dpre / h, gu_fwd preact / h, gu_dgrad dx): full 32-row
    # chunks leave by TMA store from the swizzled staging tile (the epilogue warps skip the
    # ld.shared + st.global copy-out); group-tail chunks keep the masked stores
    TMA_ST = kind in ("down_dgrad", "gu_fwd", "gu_dgrad") and not SWAP
    STG = kind == "down_fwd" and SWAP  # swap-AB scatter: stage token rows across the 4 lane-quadrant warps
    a_shape = (NST, BM, BK) if a_major == "k" else (NST, BK, BM)
    b_shape = (NST, HNT, BK) if b_major == "k" else (NST, BK, HN)
    a_idesc_trans = a_major == "mn"
    b_idesc_trans = b_major == "mn"
    # overlap variant (dyn): dynamic tile scheduling by cluster launch control (the grid holds one
    # cluster per tile, upper bound; resident clusters cancel pending ones and take their tile
    # ids) and no griddepcontrol.wait before the main loop: clusters land on the SMs the previous
    # (independent, host-checked) kernel frees in its tail and take more tiles than late ones.
    # Every CTA waits for the previous grid before it exits, so this grid's completion still
    # implies its predecessor's.
    DYN = dyn
    # ring consumers per entry (arrivals on CTA0's slot-empty barrier): the TMA producer(s),
    # the MMA issuer, CTA1's gather relay, every epilogue warp and every gather warp
    NCONS = ((1 if TMA_BYTES == 0 else 2) + 1 + (1 if GATHER else 0) + 2 * NEPI
             + (2 * NGW if GATHER else 0))
    params = params2(kind, has_bias, clamped, MG2)
    names = [n for n, _ in params]

    def enc(m, rank, data, *ops, dtype="bfloat16"):
        txl.cu_tensor_map_encode_tiled(m, dtype, rank, data, *ops)

    def host_prelude(p):
        maps = {}
        if kind == "gu_fwd":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["x"].data, H, p["T"], H * 2, 64,
                BM if MAT_GU else 1, 1, 1, 0, 3, 2, 0)
            if WFAR:
                maps["b"] = maps["a"]  # unused: the gather warps load the weights
            else:
                maps["b"] = txl.stack_alloca("tensormap", 1)
                enc(maps["b"], 4, p["w"].data, H, 2, inter, p["G"], p["wdiff"] * 2, p["ws1"] * 2,
                    p["ws0"] * 2, 64, 2, HN // 2, 1, 1, 1, 1, 1, 0, 3, 2, 0)
            if TMA_ST:
                # preact (M, 2I): 32 x 32 boxes (64B swizzle); h (M, I): 32 rows x 16 (32B)
                maps["d"] = txl.stack_alloca("tensormap", 1)
                enc(maps["d"], 2, p["preact"].data, 2 * inter, p["M"], 4 * inter, 32, 32, 1, 1, 0, 2, 2, 0)
                maps["e"] = txl.stack_alloca("tensormap", 1)
                enc(maps["e"], 2, p["h"].data, inter, p["M"], 2 * inter, 16, 32, 1, 1, 0, 1, 2, 0)
        elif kind == "down_fwd":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["hin"].data, inter, p["M"], inter * 2, 64, BM, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 3, p["wd"].data, inter, H, p["G"], inter * 2, H * inter * 2, 64, HNT, 1,
                1, 1, 1, 0, 3, 2, 0)
        elif kind == "down_dgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["dy"].data, H, p["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 3, p["wd"].data, inter, H, p["G"], inter * 2, H * inter * 2, 64, BK, 1,
                1, 1, 1, 0, 3, 2, 0)
            if TMA_ST:
                # epilogue TMA stores of the swizzled staging tiles: dpre (M, 2I) in 32-row boxes
                # of 64 columns (128B swizzle) or 32 (split staging, 64B swizzle), h (M, I) in
                # 32 x 32 (64B swizzle)
                maps["d"] = txl.stack_alloca("tensormap", 1)
                enc(maps["d"], 2, p["dpre"].data, 2 * inter, p["M"], 4 * inter, 32 if SPLIT_DPRE else 64, 32,
                    1, 1, 0, 2 if SPLIT_DPRE else 3, 2, 0)
                maps["e"] = txl.stack_alloca("tensormap", 1)
                enc(maps["e"], 2, p["h"].data, inter, p["M"], 2 * inter, 32, 32, 1, 1, 0, 2, 2, 0)
        elif kind == "gu_dgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["dpre"].data, 2 * inter, p["M"], 4 * inter, 64, BM, 1, 1, 0, 3, 2, 0)
            if WFAR:
                maps["b"] = maps["a"]  # unused: the gather warps load the weights
            else:
                maps["b"] = txl.stack_alloca("tensormap", 1)
                enc(maps["b"], 4, p["w"].data, H, 2, inter, p["G"], p["wdiff"] * 2, p["ws1"] * 2,
                    p["ws0"] * 2, 64, 2, BK // 2, 1, 1, 1, 1, 1, 0, 3, 2, 0)
            if TMA_ST:
                # dx (M, H): 32 x 32 boxes (64B swizzle)
                maps["d"] = txl.stack_alloca("tensormap", 1)
                enc(maps["d"], 2, p["dx"].data, H, p["M"], 2 * H, 32, 32, 1, 1, 0, 2, 2, 0)
        elif kind == "gu_wgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["dpre"].data, 2 * inter, p["M"], 4 * inter, 64, BK, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 2, p["x"].data, H, p["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            # main grads as (H, 2, I, G) over the lower buffer: rows (i, address lane)
            maps["c"] = txl.stack_alloca("tensormap", 1)
            if MG2:
                # far pair: each buffer as (H, I, G), the staging tile's lane-0 rows reduce into
                # the lower buffer, its lane-1 rows into the upper one
                enc(maps["c"], 3, p["mg"].data, H, inter, p["G"], p["ms1"] * 4, p["ms0"] * 4,
                    32, 16, 1, 1, 1, 1, 0, 3, 2, 0, dtype="float32")
                maps["c2"] = txl.stack_alloca("tensormap", 1)
                enc(maps["c2"], 3, p["mg2"].data, H, inter, p["G"], p["ms1"] * 4, p["ms0"] * 4,
                    32, 16, 1, 1, 1, 1, 0, 3, 2, 0, dtype="float32")
            else:
                enc(maps["c"], 4, p["mg"].data, H, 2, inter, p["G"], p["mgdiff"] * 4, p["ms1"] * 4,
                    p["ms0"] * 4, 32, 2, 16, 1, 1, 1, 1, 1, 0, 3, 2, 0, dtype="float32")
        else:  # down_wgrad
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["dy"].data, H, p["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 2, p["hin"].data, inter, p["M"], inter * 2, 64, BK, 1, 1, 0, 3, 2, 0)
            maps["c"] = txl.stack_alloca("tensormap", 1)
            enc(maps["c"], 3, p["mg"].data, inter, H, p["G"], inter * 4, H * inter * 4, 32, 32, 1, 1, 1, 1,
                0, 3, 2, 0, dtype="float32")
        return maps

    def body(p, maps):
        cu, G = p["cu"], p["G"]
        a_map, b_map = maps["a"], maps["b"]
        cta = txl.cta_id()
        cbx = txl.cuda.cluster_cta_id("x")
        cluster = cta // 2
        warp = txl.warp_id()
        lane = txl.lane_id()

        with txl.if_(warp == 0), txl.then_():
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(a_map))
                txl.ptx.prefetch.tensormap(txl.address_of(b_map))

        smem = txl.smem_pool()
        tmem_addr = smem.alloc((1,), "uint32", align=8)
        prefix = smem.alloc((GMAX + 1,), "int32", align=16)
        # full: CTA0's barrier collects both CTAs' TMA bytes (+ CTA1's gather relay);
        # empty / acc_full: per CTA, signaled by the multicast commit; acc_empty: CTA0's,
        # arrived by all eight epilogue warps of the pair.
        ab_pipe = txl.Pipeline(smem, NST, full="tma", empty="tcgen05",
                               init_full=1 + (1 if GATHER else 0))
        acc_pipe = txl.Pipeline(smem, 2, full="tcgen05", empty="mbar", init_empty=2 * NEPI)
        gfull = txl.MBarrier(smem, NST)
        gfull.init(32 * NGW)
        order = smem.alloc((GMAX,), "int32", align=16)
        if DYN:
            # tile-id ring: per CTA a full barrier (the local scheduler's expect_tx arrive + the
            # multicast CLC response bytes) and the 16-byte responses; CTA0's empty barrier
            # collects every consumer warp of the pair. dfull / dresp: CTA0's private drain slot.
            sfull = txl.MBarrier(smem, SLOTS)
            sfull.init(1)
            sempty = txl.MBarrier(smem, SLOTS)
            sempty.init(NCONS)
            dfull = txl.MBarrier(smem, 1)
            dfull.init(1)
            resp = smem.alloc((SLOTS * 4,), "uint32", align=16)
            dresp = smem.alloc((4,), "uint32", align=16)
        smem.pool.move_base_to(8192 + 4 * (GMAX + 1))
        a_tile = smem.alloc(a_shape, "bfloat16", swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM)
        b_tile = smem.alloc(b_shape, "bfloat16", swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM)
        if k_grouped:
            # per epilogue warp, two 32x32 fp32 staging tiles (128-byte swizzled rows)
            cbuf = smem.alloc((4 * 2 * 32 * 32,), "float32", align=1024)
        if XSTAGE:
            # per epilogue warp: one 32-row x 128-byte transpose tile (16-byte chunks swizzled)
            xbuf = smem.alloc((NEPI * XWORDS,), "uint32", align=1024)
        if PF:
            pbuf = smem.alloc((NEPI * 1024,), "uint32", align=1024)
        if STG:
            # per epilogue half, two 32-token x 128-feature bf16 tiles (64 u32 per token row)
            sbuf = smem.alloc(((NEPI // 4) * 2 * 32 * 64,), "uint32", align=1024)

        # programmatic dependent launch: the smem-only prologue above overlaps the previous
        # kernel's tail; every global access of this grid follows its completion, and the next
        # kernel may start its own prologue once all of this grid's CTAs are resident
        if not DYN:
            txl.ptx.griddepcontrol.wait()
        txl.ptx.griddepcontrol.launch_dependents()

        # ---- tile prefix over the groups (warp 2), then TMEM allocation (warp 1).
        with txl.if_(warp == PREFIX_WARP), txl.then_():
            run = txl.local_scalar("int32", init=0)
            base = txl.local_scalar("int32", init=0)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.s32(prefix.ptr_to([0]), txl.int32(0))
            if LPT:
                # longest-first group order: rank sort over register-resident counts (warp
                # shuffles); identity order past 128 groups
                with txl.if_(G <= 128):
                    with txl.then_():
                        cnts = txl.alloc_local((4,), "int32")
                        for q in range(4):
                            g = q * 32 + lane
                            txl.assign(cnts[q], -1)
                            with txl.if_(g < G), txl.then_():
                                lo = txl.local_scalar("int32")
                                hi = txl.local_scalar("int32")
                                txl.ptx.ld.global_.s32(lo, cu.ptr_to([g]))
                                txl.ptx.ld.global_.s32(hi, cu.ptr_to([g + 1]))
                                txl.assign(cnts[q], hi - lo)
                        ranks = txl.alloc_local((4,), "int32")
                        for q in range(4):
                            txl.assign(ranks[q], 0)
                        other = txl.local_scalar("int32")
                        for qh in range(4):
                            with txl.if_(qh * 32 < G), txl.then_():
                                with txl.serial(32) as hl:
                                    txl.ptx.shfl_sync.idx.b32(other, cnts[qh], txl.Cast("uint32", hl),
                                                              txl.uint32(31), txl.uint32(0xFFFFFFFF))
                                    h = qh * 32 + hl
                                    for q in range(4):
                                        g = q * 32 + lane
                                        txl.assign(ranks[q], ranks[q] + txl.Select(
                                            (other > cnts[q]) | ((other == cnts[q]) & (h < g)), 1, 0))
                        for q in range(4):
                            g = q * 32 + lane
                            with txl.if_(g < G), txl.then_():
                                txl.ptx.st.shared.s32(order.ptr_to([ranks[q]]), g)
                    with txl.else_():
                        with txl.while_(base < G):
                            with txl.if_(base + lane < G), txl.then_():
                                txl.ptx.st.shared.s32(order.ptr_to([base + lane]), base + lane)
                            txl.assign(base, base + 32)
                txl.cuda.warp_sync()
                txl.assign(base, 0)
            with txl.while_(base < G):
                g = txl.local_scalar("int32", init=base + lane)
                cnt = txl.local_scalar("int32", init=0)
                if LPT:
                    with txl.if_(g < G), txl.then_():
                        txl.ptx.ld.shared.s32(g, order.ptr_to([base + lane]))
                with txl.if_(base + lane < G), txl.then_():
                    lo = txl.local_scalar("int32")
                    hi = txl.local_scalar("int32")
                    txl.ptx.ld.global_.s32(lo, cu.ptr_to([g]))
                    txl.ptx.ld.global_.s32(hi, cu.ptr_to([g + 1]))
                    if k_grouped:
                        txl.assign(cnt, txl.Select(hi > lo, txl.int32(MT * NT), txl.int32(0)))
                    else:
                        txl.assign(cnt, ((hi - lo + (BMC - 1)) // BMC) * NT)
                # inclusive warp scan
                for off in (1, 2, 4, 8, 16):
                    other = txl.local_scalar("int32")
                    txl.ptx.shfl_sync.up.b32(other, cnt, txl.uint32(off), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                    with txl.if_(lane >= off), txl.then_():
                        txl.assign(cnt, cnt + other)
                with txl.if_(base + lane < G), txl.then_():
                    txl.ptx.st.shared.s32(prefix.ptr_to([base + lane + 1]), run + cnt)
                last = txl.local_scalar("int32")
                txl.ptx.shfl_sync.idx.b32(last, cnt, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                txl.assign(run, run + last)
                txl.assign(base, base + 32)
        with txl.if_(warp == 1), txl.then_():
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_addr[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        full_cta0 = ab_pipe.full.remote_view(0)
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.tcgen05.fence__before_thread_sync()
        txl.cuda.cluster_sync()
        txl.ptx.tcgen05.fence__after_thread_sync()

        def set_regs(role):
            """
            Warpgroup-collective register handoff to the SwiGLU-backward epilogue; emitted
            at the head of each role so it dominates that role's code.
            """
            if REGS is None:
                return
            n = REGS[role]
            txl.ptx[f"setmaxnreg.{'inc' if role == 2 else 'dec'}.sync.aligned.u32"](n)

        with txl.if_((warp == 2) | (warp == 3)), txl.then_():
            set_regs(0)

        total = txl.local_scalar("int32")
        txl.ptx.ld.shared.s32(total, prefix.ptr_to([G]))

        class TileWalk:
            """
            Per-role cursor: tile t -> (g, mt, nt), the group index only moves forward.
            """

            def __init__(self):
                self.t = txl.local_scalar("int32", init=cluster)
                self.k = txl.local_scalar("int32", init=0)
                self.g = txl.local_scalar("int32", init=0)
                self.mt = txl.local_scalar("int32", init=0)
                self.nt = txl.local_scalar("int32", init=0)
                self.lo = txl.local_scalar("int32", init=0)
                self.hi = txl.local_scalar("int32", init=0)
                self.nn = txl.local_scalar("int32", init=BN)
                if DYN:
                    self.tn = txl.local_scalar("int32", init=0)
                    self.ss = txl.PipelineState(SLOTS, 0)

            def locate(self):
                if DYN:
                    # CLC ids normally grow per cluster; restart the group search otherwise
                    first = txl.local_scalar("int32")
                    txl.ptx.ld.shared.s32(first, prefix.ptr_to([self.k]))
                    with txl.if_(self.t < first), txl.then_():
                        txl.assign(self.k, 0)
                nxt = txl.local_scalar("int32")
                txl.ptx.ld.shared.s32(nxt, prefix.ptr_to([self.k + 1]))
                with txl.while_(self.t >= nxt):
                    txl.assign(self.k, self.k + 1)
                    txl.ptx.ld.shared.s32(nxt, prefix.ptr_to([self.k + 1]))
                start = txl.local_scalar("int32")
                txl.ptx.ld.shared.s32(start, prefix.ptr_to([self.k]))
                if LPT:
                    txl.ptx.ld.shared.s32(self.g, order.ptr_to([self.k]))
                else:
                    txl.assign(self.g, self.k)
                local = self.t - start
                txl.assign(self.mt, local // NT)
                txl.assign(self.nt, local % NT)
                txl.ptx.ld.global_.s32(self.lo, cu.ptr_to([self.g]))
                txl.ptx.ld.global_.s32(self.hi, cu.ptr_to([self.g + 1]))
                if SWAP:
                    # this token chunk's MMA N: the rows left, rounded up to 32 (both CTAs
                    # then hold a multiple of 16), at most 256
                    rem = self.hi - self.lo - self.mt * BMC
                    txl.assign(self.nn, txl.Select(rem >= BN, txl.int32(BN), (rem + 31) // 32 * 32))

            def fetch_next(self):
                """
                Warp-collective: take the next ring entry (this cluster's next tile, or past
                total when the scheduler stopped) into self.tn and release the slot.
                """
                if DYN:
                    sfull.wait(self.ss.stage, self.ss.phase)
                    cid = txl.local_scalar("uint32")
                    txl.query_cancel_first_ctaid_x(cid, resp.ptr_to([self.ss.stage * 4]))
                    txl.assign(self.tn, txl.Select(cid == txl.uint32(0xFFFFFFFF),
                                                   txl.int32(0x7FFFFFFF),
                                                   txl.Cast("int32", cid >> txl.uint32(1))))
                    txl.cuda.warp_sync()
                    with txl.if_(txl.lane_id() == 0), txl.then_():
                        sempty.arrive(self.ss.stage, remote=0)
                    self.ss.advance()

            def advance(self):
                if DYN:
                    txl.assign(self.t, self.tn)
                else:
                    txl.assign(self.t, self.t + p["num_ctas"] // 2)

        def full_bar(stage):
            return txl.cuda.cvta_generic_to_shared(full_cta0.ptr_to([stage]))

        # ------------------------------------------------------------ CLC scheduler (warp 3)
        if DYN:
            with txl.if_((warp == 3) & (lane == 0)), txl.then_():
                ph_f = txl.PipelineState(SLOTS, 0)
                ph_e = txl.PipelineState(SLOTS, 1)
                cid = txl.local_scalar("uint32", init=txl.uint32(0))
                stop = txl.local_scalar("int32", init=txl.Select(cluster >= total, 1, 0))
                with txl.while_(stop == 0):
                    s = ph_f.stage
                    with txl.if_(cbx == 0), txl.then_():
                        sempty.wait(ph_e.stage, ph_e.phase)
                        txl.ptx[
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        ](resp.ptr_to([s * 4]), sfull.ptr_to([s]))
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(sfull.ptr_to([s]), txl.uint32(16))
                    sfull.wait(s, ph_f.phase)
                    txl.query_cancel_first_ctaid_x(cid, resp.ptr_to([s * 4]))
                    with txl.if_((cid == txl.uint32(0xFFFFFFFF))
                                | (txl.Cast("int32", cid >> txl.uint32(1)) >= total)), txl.then_():
                        txl.assign(stop, 1)
                    ph_f.advance()
                    ph_e.advance()
                # drain: cancel the remaining (tile-less) pending clusters so they never launch
                with txl.if_((cbx == 0) & (cid != txl.uint32(0xFFFFFFFF))), txl.then_():
                    dph = txl.local_scalar("int32", init=0)
                    with txl.while_(cid != txl.uint32(0xFFFFFFFF)):
                        txl.ptx[
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                        ](dresp.ptr_to([0]), dfull.ptr_to([0]))
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(dfull.ptr_to([0]), txl.uint32(16))
                        dfull.wait(0, dph)
                        txl.query_cancel_first_ctaid_x(cid, dresp.ptr_to([0]))
                        txl.assign(dph, dph ^ 1)

        # ------------------------------------------------------------ TMA producer
        # (the non-gathered operand(s); the gathered one is the cp.async role below). With both
        # gathered (far-pair gate_up fwd) only CTA0's runs: its zero-byte arrive on the full
        # barrier keeps the stage protocol; CTA1's would wait on empty phases nothing gates.
        with txl.if_((warp == 0) & (cbx == 0) if TMA_BYTES == 0 else (warp == 0)), txl.then_():
            set_regs(0)
            walk = TileWalk()
            st = txl.PipelineState(NST, 1)
            with txl.while_(walk.t < total):
                walk.locate()
                walk.fetch_next()
                nkt = txl.local_scalar("int32")
                if k_grouped:
                    txl.assign(nkt, (walk.hi - walk.lo + (BK - 1)) // BK)
                else:
                    txl.assign(nkt, KT)
                with txl.if_(lane == 0), txl.then_():
                    with txl.serial(nkt) as kt:
                        tk = range_v("p-wait-empty")
                        ab_pipe.empty.wait(st.stage, st.phase)
                        end(tk)
                        s = st.stage
                        with txl.if_(cbx == 0), txl.then_():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([s]), txl.uint32(2 * TMA_BYTES)
                            )
                        a_base = a_tile[s].ptr_to(0, 0)
                        b_base = b_tile[s].ptr_to(0, 0)
                        i32 = lambda v: txl.Cast("int32", v)  # noqa: E731
                        if not k_grouped:
                            k = kt * BK
                            mrow = walk.lo + walk.mt * BMC + cbx * (walk.nn // 2)
                            if not a_gather:
                                txl.ptx[TMA.format(2)](
                                    a_base, txl.address_of(a_map), i32(k), i32(mrow), full_bar(s)
                                )
                            n0 = walk.nt * BNT + cbx * HNT
                            if B_GATH:
                                pass
                            elif kind == "gu_fwd":
                                txl.ptx[TMA.format(4)](
                                    b_base, txl.address_of(b_map), i32(k), i32(0), i32(n0 // 2),
                                    i32(walk.g), full_bar(s),
                                )
                            elif kind == "down_fwd":
                                txl.ptx[TMA.format(3)](
                                    b_base, txl.address_of(b_map), i32(k), i32(n0), i32(walk.g),
                                    full_bar(s),
                                )
                            elif kind == "down_dgrad":
                                for c in range(HN // 64):
                                    txl.ptx[TMA.format(3)](
                                        txl.ptx.addr(b_base, c * BK * 128), txl.address_of(b_map),
                                        i32(n0 + c * 64), i32(k), i32(walk.g), full_bar(s),
                                    )
                            else:  # gu_dgrad
                                for c in range(HN // 64):
                                    txl.ptx[TMA.format(4)](
                                        txl.ptx.addr(b_base, c * BK * 128), txl.address_of(b_map),
                                        i32(n0 + c * 64), i32(0), i32(k // 2), i32(walk.g),
                                        full_bar(s),
                                    )
                        else:
                            k0 = walk.lo + kt * BK
                            if kind == "gu_wgrad":
                                m0 = walk.mt * BMC + cbx * BM
                                for c in range(BM // 64):
                                    txl.ptx[TMA.format(2)](
                                        txl.ptx.addr(a_base, c * BK * 128), txl.address_of(a_map),
                                        i32(m0 + c * 64), i32(k0), full_bar(s),
                                    )
                            else:
                                n0 = walk.nt * BN + cbx * HN
                                for c in range(HN // 64):
                                    txl.ptx[TMA.format(2)](
                                        txl.ptx.addr(b_base, c * BK * 128), txl.address_of(b_map),
                                        i32(n0 + c * 64), i32(k0), full_bar(s),
                                    )
                        st.advance()
                txl.cuda.warp_sync()
                walk.advance()

        # ------------------------------------------------------------ cp.async gather
        if GATHER:
            with txl.if_(warp >= GW0), txl.then_():
                set_regs(1)
                gw = warp - GW0
                walk = TileWalk()
                st = txl.PipelineState(NST, 1)
                if kind != "gu_dgrad":
                    src = p["x"] if kind in ("gu_fwd", "gu_wgrad") else p["dy"]
                    ids = p["a_idx"] if kind in ("gu_fwd", "gu_wgrad") else p["slot_of"]
                my = txl.local_scalar("int32")  # this lane's row index (-1: zero row)
                cur_row = txl.local_scalar("int32")
                row = txl.local_scalar("int32")
                chunk = lane % 8
                sub = lane // 8
                FULLM = txl.uint32(0xFFFFFFFF)

                def load_index(r, hi):
                    txl.assign(my, -1)
                    with txl.if_(r < hi), txl.then_():
                        if PREGATHERED and k_grouped:
                            txl.assign(my, r)
                        else:
                            txl.ptx.ld.global_.s32(my, ids.ptr_to([r]))

                def copy(dst, src_col, col_ok):
                    # 16 B of the row held in `row`; zero-fill rows < 0 and columns past H
                    ok = (row >= 0) & col_ok
                    txl.ptx[CP16](
                        dst, src.ptr_to([txl.Select(ok, row * H + src_col, 0)]), 16,
                        txl.Select(ok, txl.uint32(16), txl.uint32(0)),
                    )

                def wcopy(dst, r, col, ok):
                    """
                    16 B at column col of interleaved weight row r = (i, address lane): lane 1
                    is the upper buffer, wdiff elements above (64-bit, no TMA stride limit).
                    """
                    off = (txl.Cast("int64", r % 2) * p["wdiff"] + txl.Cast("int64", walk.g) * p["ws0"]
                           + txl.Cast("int64", r // 2) * p["ws1"] + txl.Cast("int64", col))
                    txl.ptx[CP16](
                        dst, p["w"].ptr_to([txl.Select(ok, off, txl.Cast("int64", 0))]), 16,
                        txl.Select(ok, txl.uint32(16), txl.uint32(0)),
                    )

                with txl.while_(walk.t < total):
                    walk.locate()
                    walk.fetch_next()
                    if kind == "gu_dgrad":
                        # far-pair MN-major weights: warp gw owns K rows 16 gw .. + 15 of the stage
                        nseg = HN // 64
                        wcol0 = walk.nt * BN + cbx * HN
                        with txl.serial(KT) as kt:
                            tk = range_v("g-wait-empty")
                            ab_pipe.empty.wait(st.stage, st.phase)
                            end(tk)
                            txl.ptx.fence.proxy.async_.shared__cta()
                            s = st.stage
                            for i in range(4 * nseg):
                                pair = i * 4 + sub
                                rr = pair // nseg
                                seg = pair % nseg
                                col = seg * 64 + chunk * 8
                                wcopy(b_tile[s].ptr_to(gw * 16 + rr, col), kt * BK + gw * 16 + rr,
                                      wcol0 + col, wcol0 + col < H)
                            txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](gfull.ptr_to([s]))
                            st.advance()
                    elif not k_grouped:
                        # K-major A: warp gw owns tile rows 32 gw .. + 31; 8 lanes per row
                        # (far pair: likewise B's weight rows)
                        if A_GATH:
                            load_index(walk.lo + walk.mt * BMC + cbx * (walk.nn // 2) + gw * RPW + lane % RPW, walk.hi)
                        wn0 = walk.nt * BN + cbx * HN
                        with txl.serial(KT) as kt:
                            tk = range_v("g-wait-empty")
                            ab_pipe.empty.wait(st.stage, st.phase)
                            end(tk)
                            txl.ptx.fence.proxy.async_.shared__cta()
                            s = st.stage
                            if A_GATH:
                                for i in range(RPW // 4):
                                    txl.ptx.shfl_sync.idx.b32(row, my, txl.uint32(i * 4) + txl.Cast("uint32", sub), txl.uint32(31), FULLM)
                                    trow = gw * RPW + i * 4 + sub
                                    copy(a_tile[s].ptr_to(trow, chunk * 8), kt * BK + chunk * 8, True)
                            if WFAR:
                                for i in range(8):
                                    trow = gw * 32 + i * 4 + sub
                                    wcopy(b_tile[s].ptr_to(trow, chunk * 8), wn0 + trow,
                                          kt * BK + chunk * 8, wn0 + trow < 2 * inter)
                            txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](gfull.ptr_to([s]))
                            st.advance()
                    else:
                        # MN-major operand: warp gw owns K rows 16 gw .. + 15 of the stage
                        if kind == "gu_wgrad":
                            tile, ncols, col0 = b_tile, HN, walk.nt * BN + cbx * HN
                        else:
                            tile, ncols, col0 = a_tile, BM, walk.mt * BMC + cbx * BM
                        nseg = ncols // 64  # 128-byte row segments per K row
                        nkt = txl.local_scalar("int32", init=(walk.hi - walk.lo + (BK - 1)) // BK)
                        # the row index of stage kt + 1 is loaded one stage ahead (off the
                        # cp.async critical path)
                        load_index(walk.lo + gw * KPW + lane % KPW, walk.hi)
                        with txl.serial(nkt) as kt:
                            txl.assign(cur_row, my)
                            rn = walk.lo + (kt + 1) * BK + gw * KPW + lane % KPW
                            load_index(rn, walk.hi)
                            tk = range_v("g-wait-empty")
                            ab_pipe.empty.wait(st.stage, st.phase)
                            end(tk)
                            txl.ptx.fence.proxy.async_.shared__cta()
                            s = st.stage
                            for i in range(KPW * nseg // 4):
                                # 4 (row, segment) pairs per instruction, 8 lanes each
                                pair = i * 4 + sub
                                rr = pair // nseg
                                seg = pair % nseg
                                txl.ptx.shfl_sync.idx.b32(row, cur_row, txl.Cast("uint32", rr), txl.uint32(31), FULLM)
                                col = seg * 64 + chunk * 8
                                copy(tile[s].ptr_to(gw * KPW + rr, col), col0 + col, col0 + col < H)
                            txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](gfull.ptr_to([s]))
                            st.advance()
                    walk.advance()

        # ------------------------------------------------------------ gather relay (CTA1)
        if GATHER:
            with txl.if_((warp == 1) & (cbx == 1)), txl.then_():
                set_regs(0)
                walk = TileWalk()
                st = txl.PipelineState(NST, 0)
                with txl.while_(walk.t < total):
                    walk.locate()
                    walk.fetch_next()
                    nkt = txl.local_scalar("int32")
                    if k_grouped:
                        txl.assign(nkt, (walk.hi - walk.lo + (BK - 1)) // BK)
                    else:
                        txl.assign(nkt, KT)
                    with txl.if_(lane == 0), txl.then_():
                        with txl.serial(nkt):
                            tk = range_v("r-wait-gfull")
                            gfull.wait(st.stage, st.phase)
                            end(tk)
                            txl.ptx.fence.proxy.async_.shared__cta()
                            ab_pipe.full.arrive(st.stage, remote=0)
                            st.advance()
                    txl.cuda.warp_sync()
                    walk.advance()

        # ------------------------------------------------------------ MMA issuer (CTA0)
        with txl.if_((warp == 1) & (cbx == 0)), txl.then_():
            set_regs(0)
            walk = TileWalk()
            st = txl.PipelineState(NST, 0)
            acc = txl.PipelineState(2, 1)
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tmem_base, tmem_addr.ptr_to([0]))
            idesc = txl.alloc_local((1,), "uint32")
            txl.cuda.tcgen05.encode_instr_descriptor(
                txl.address_of(idesc[0]), d_dtype="float32", a_dtype="bfloat16",
                b_dtype="bfloat16", M=BMC, N=BN, K=16,
                trans_a=b_idesc_trans if SWAP else a_idesc_trans,
                trans_b=a_idesc_trans if SWAP else b_idesc_trans, n_cta_groups=2,
            )
            if BNT != BN:
                # PTX allows N in steps of 16 for cta_group::2 kind::f16 (the encoder helper
                # checks steps of 32): patch the N field (bits 17-22 = N >> 3)
                txl.assign(idesc[0], (idesc[0] & txl.uint32(0xFF81FFFF)) | txl.uint32((BNT >> 3) << 17))
            a_desc, a_off = a_tile[0].encode(major=a_major, mma_k=16)
            b_desc, b_off = b_tile[0].encode(major=b_major, mma_k=16)
            leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.while_(walk.t < total):
                walk.locate()
                walk.fetch_next()
                nkt = txl.local_scalar("int32")
                if k_grouped:
                    txl.assign(nkt, (walk.hi - walk.lo + (BK - 1)) // BK)
                else:
                    txl.assign(nkt, KT)
                tk = range_v("m-wait-acc")
                acc_pipe.empty.wait(acc.stage, acc.phase)
                end(tk)
                txl.ptx.tcgen05.fence__after_thread_sync()
                d_addr = tmem_base + txl.Cast("uint32", acc.stage * BN)
                idesc_t = txl.local_scalar("uint32", init=idesc[0])
                if SWAP:
                    # instruction-descriptor N field (bits 17-22 = N >> 3) for this tile
                    txl.assign(idesc_t, (idesc[0] & txl.uint32(0xFF81FFFF))
                               | (txl.Cast("uint32", walk.nn >> 3) << txl.uint32(17)))
                with txl.serial(nkt) as kt:
                    tk = range_v("m-wait-full")
                    ab_pipe.full.wait(st.stage, st.phase)
                    end(tk)
                    if GATHER:
                        # this CTA's cp.async half wrote the stage through the generic proxy
                        tk = range_v("m-wait-gfull")
                        gfull.wait(st.stage, st.phase)
                        end(tk)
                        txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.tcgen05.fence__after_thread_sync()
                    s = st.stage
                    with txl.if_(leader == txl.uint32(1)), txl.then_():
                        for kp in range(BK // 16):
                            ad = a_desc.value + txl.Cast("uint64", s * (A_BYTES // 16) + a_off(kp))
                            bd = b_desc.value + txl.Cast("uint64", s * (B_BYTES // 16) + b_off(kp))
                            accum = txl.cast(kt > 0, "bool") if kp == 0 else txl.cast(True, "bool")
                            if SWAP:
                                # A = the weight tile (128 features per CTA), B = the token tile
                                txl.ptx[MMA](
                                    d_addr, bd, ad, idesc_t, *([txl.uint32(0)] * 8), accum,
                                )
                            else:
                                txl.ptx[MMA](
                                    d_addr, ad, bd, idesc[0], *([txl.uint32(0)] * 8), accum,
                                )
                        ab_pipe.empty.arrive(s, cta_group=2, cta_mask=3)
                    txl.cuda.warp_sync()
                    st.advance()
                with txl.if_(leader == txl.uint32(1)), txl.then_():
                    acc_pipe.full.arrive(acc.stage, cta_group=2, cta_mask=3)
                txl.cuda.warp_sync()
                acc.advance()
                walk.advance()

        # ------------------------------------------------------------ epilogue
        with txl.if_((warp >= EPI_WARP0) & (warp < EPI_WARP0 + NEPI)), txl.then_():
            set_regs(2)
            walk = TileWalk()
            acc = txl.PipelineState(2, 0)
            ew = warp % 4
            half = (warp - EPI_WARP0) // 4  # column half (NEPI == 8) or 0
            m = ew * 32 + lane
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tmem_base, tmem_addr.ptr_to([0]))
            v = txl.alloc_local((32,), "float32")
            w16 = txl.alloc_local((16,), "uint32")
            w8 = txl.alloc_local((8,), "uint32")
            ld = txl.alloc_local((32,), "uint32")
            dst = txl.local_scalar("int32")

            xb = (warp - EPI_WARP0) * XWORDS

            def xaddr(row, ch, nch):
                if TMA_ST and nch == 4:
                    # 64-byte rows in the TMA 64B-swizzle pattern (16-byte chunk ^= (row / 2) % 4)
                    return xbuf.ptr_to([xb + row * 16 + ((ch ^ ((row // 2) % 4)) * 4)])
                if TMA_ST and nch == 2:
                    # 32-byte rows in the TMA 32B-swizzle pattern (16-byte chunk ^= (row / 4) % 2)
                    return xbuf.ptr_to([xb + row * 8 + ((ch ^ ((row // 4) % 2)) * 4)])
                # rows of nch 16-byte chunks packed into 128-byte lines, 128B-swizzled
                rpl = 8 // nch
                line = row // rpl
                c8 = (row % rpl) * nch + ch
                return xbuf.ptr_to([xb + line * 32 + ((c8 ^ (line % 8)) * 4)])

            pb = (warp - EPI_WARP0) * 1024

            def paddr(row, ch):
                line = row
                return pbuf.ptr_to([pb + line * 32 + ((ch ^ (line % 8)) * 4)])

            def prefetch(w, c):
                """
                cp.async this warp's 32 rows x 32 (gate, up) words of preact for tile w,
                chunk c into pbuf (rows past the group zero-filled; chunks past N skipped).
                """
                n0 = w.nt * BNT + half * (CH * 32) + c * 32
                rb = w.lo + w.mt * BMC + cbx * BM + ew * 32
                with txl.if_(n0 < N), txl.then_():
                    for it in range(8):
                        row = it * 4 + lane // 8
                        ch = lane % 8
                        ok = rb + row < w.hi
                        txl.ptx[CP16](
                            paddr(row, ch),
                            p["preact"].ptr_to([txl.Select(ok, (rb + row) * inter + n0 + ch * 4, 0)]), 16,
                            txl.Select(ok, txl.uint32(16), txl.uint32(0)),
                        )
                txl.ptx["cp.async.commit_group"]()

            def stage_load(dst, nch, gaddr, ok):
                """
                dst[4 nch] = this lane's row: coalesced global rows -> smem -> registers.
                """
                tmp = txl.alloc_local((4,), "uint32")
                rpi = 32 // nch
                ch = lane % nch
                for it in range(nch):
                    row = it * rpi + lane // nch
                    for k4 in range(4):
                        txl.assign(tmp[k4], txl.uint32(0))  # rows past the group read as zero
                    with txl.if_(ok(row)), txl.then_():
                        txl.ptx.ld.global_.v4.b32(tmp[0], tmp[1], tmp[2], tmp[3], gaddr(row, ch))
                    txl.ptx.st.shared.v4.b32(xaddr(row, ch, nch), tmp[0], tmp[1], tmp[2], tmp[3])
                txl.cuda.warp_sync()
                for c2 in range(nch):
                    txl.ptx.ld.shared.v4.b32(dst[4 * c2], dst[4 * c2 + 1], dst[4 * c2 + 2],
                                             dst[4 * c2 + 3], xaddr(lane, c2, nch))
                txl.cuda.warp_sync()

            def stage_store(src, nch, gaddr, ok, cmask=None, tma=None, full=None):
                """
                This lane's row src[4 nch] -> smem -> coalesced global rows (or, with tma and a
                full 32-row chunk, one TMA store of the staging tile).
                """
                tmp = txl.alloc_local((4,), "uint32")
                for c2 in range(nch):
                    txl.ptx.st.shared.v4.b32(xaddr(lane, c2, nch), src[4 * c2], src[4 * c2 + 1],
                                             src[4 * c2 + 2], src[4 * c2 + 3])
                txl.cuda.warp_sync()
                rpi = 32 // nch
                ch = lane % nch

                def copy_out():
                    for it in range(nch):
                        row = it * rpi + lane // nch
                        txl.ptx.ld.shared.v4.b32(tmp[0], tmp[1], tmp[2], tmp[3], xaddr(row, ch, nch))
                        dst_ptr = gaddr(row, ch)  # warp-convergent (may shuffle)
                        with txl.if_(ok(row) if cmask is None else (ok(row) & cmask(ch))), txl.then_():
                            txl.ptx.st.global_.v4.b32(dst_ptr, tmp[0], tmp[1], tmp[2], tmp[3])
                if tma is None:
                    copy_out()
                else:
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.cuda.warp_sync()
                    with txl.if_(full):
                        with txl.then_():
                            with txl.if_(lane == 0), txl.then_():
                                tma(xbuf.ptr_to([xb]))
                                txl.ptx.cp.async_.bulk.commit_group()
                                # the staging tile is rewritten next: its source read must be done
                                txl.ptx.cp.async_.bulk.wait_group.read(0)
                        with txl.else_():
                            copy_out()
                txl.cuda.warp_sync()

            dw = txl.alloc_local((32,), "uint32") if XSTAGE else None

            def load_preact(buf, row, n0):
                for q in range(8):
                    txl.ptx.ld.global_.v4.b32(
                        buf[4 * q], buf[4 * q + 1], buf[4 * q + 2], buf[4 * q + 3],
                        p["preact"].ptr_to([row * inter + n0 + 4 * q]),
                    )
            alpha = p.get("alpha")
            limit = p.get("limit")
            FULL = txl.uint32(0xFFFFFFFF)

            def shx(val, m):
                """
                val of lane ^ m (fp32).
                """
                o = txl.local_scalar("float32")
                txl.ptx.shfl_sync.bfly.b32(o, val, txl.uint32(m), txl.uint32(31), FULL)
                return o

            ldn = txl.alloc_local((32,), "uint32") if (kind == "down_dgrad" and SWAP) else None

            def load_pre(buf, w, c):
                """
                buf[j] = this lane's (gate, up) preact word of token c * 32 + j (0 past the
                group or past N).
                """
                fq = w.nt * BN + cbx * HN + ew * 32 + lane
                rq = w.lo + w.mt * BMC + c * 32
                for j in range(32):
                    txl.assign(buf[j], txl.uint32(0))
                    with txl.if_((fq < N) & (rq + j < w.hi) & (c * 32 < w.nn)), txl.then_():
                        txl.ptx.ld.global_.u32(buf[j], p["preact"].ptr_to([(rq + j) * inter + fq]))

            def swap_epilogue(walk, acc):
                """
                Swap-AB tile: TMEM lane = output feature f, column = token. Every register
                column j is one token row; a warp's lanes cover 32 consecutive features, so each
                store is one coalesced row segment.
                """
                f = walk.nt * BN + cbx * HN + ew * 32 + lane
                fok = f < N
                r0 = walk.lo + walk.mt * BMC
                even = (lane % 2) == 0
                bw = txl.local_scalar("uint32", init=txl.uint32(0))
                if has_bias and kind == "gu_fwd":
                    with txl.if_(fok & even), txl.then_():
                        txl.ptx.ld.global_.u32(bw, p["bias"].ptr_to([walk.g * inter + f // 2]))
                bf = txl.local_scalar("float32", init=txl.float32(0.0))
                if has_bias and kind == "down_fwd":
                    with txl.if_(fok), txl.then_():
                        txl.ptx.ld.global_.u32(bw, p["bias"].ptr_to([(walk.g * N + f) // 2]))
                    txl.assign(bf, txl.Select(even, bf16_lo(bw), bf16_hi(bw)))
                sw = p["swapped"] != 0 if kind in ("gu_fwd", "down_dgrad") else None
                NCH = 8 // (NEPI // 4)
                if kind == "down_dgrad":
                    load_pre(ld, walk, half)
                with txl.serial(NCH) as cc:
                    c = cc * (NEPI // 4) + half
                    with txl.if_(c * 32 < walk.nn), txl.then_():
                        if STG:
                            sb = (half * 2 + cc % 2) * 2048
                        taddr = tmem_base + txl.Cast("uint32", (ew * 32) << 16) + txl.Cast(
                            "uint32", acc.stage * BN + c * 32)
                        txl.ptx[TMEM_LD32](*[v[i] for i in range(32)], taddr)
                        txl.ptx.tcgen05.wait__ld.sync.aligned()
                        rb = r0 + c * 32
                        if kind == "down_fwd":
                            txl.assign(dst, 0)
                            with txl.if_(rb + lane < walk.hi), txl.then_():
                                txl.ptx.ld.global_.s32(dst, p["slot_of"].ptr_to([rb + lane]))
                        if kind == "down_dgrad":
                            # prefetch the next chunk's preact words while this one computes
                            load_pre(ldn, walk, c + NEPI // 4)
                        for j in range(32):
                            r = rb + j
                            ok = fok & (r < walk.hi)
                            if kind in ("gu_dgrad", "down_fwd"):
                                a0 = txl.local_scalar("float32")
                                txl.assign(a0, v[j] + bf)
                                a1 = shx(a0, 1)
                                wd = txl.local_scalar("uint32")
                                pack(wd, a0, a1)
                                if kind == "gu_dgrad":
                                    optr = p["dx"].ptr_to([r * (N // 2) + f // 2])
                                else:
                                    d = txl.local_scalar("int32")
                                    txl.ptx.shfl_sync.idx.b32(d, dst, txl.uint32(j), txl.uint32(31), FULL)
                                    optr = p["out"].ptr_to([d * (N // 2) + f // 2])
                                if STG:
                                    with txl.if_(even), txl.then_():
                                        txl.ptx.st.shared.u32(sbuf.ptr_to([sb + j * 64 + ew * 16 + lane // 2]), wd)
                                else:
                                    with txl.if_(ok & even), txl.then_():
                                        txl.ptx.st.global_.u32(optr, wd)
                            elif kind == "gu_fwd":
                                lo_v = txl.local_scalar("float32")
                                txl.assign(lo_v, v[j])
                                hi_v = shx(lo_v, 1)
                                gate = txl.local_scalar("float32")
                                up = txl.local_scalar("float32")
                                txl.assign(gate, txl.Select(sw, hi_v, lo_v))
                                txl.assign(up, txl.Select(sw, lo_v, hi_v))
                                if has_bias:
                                    txl.assign(gate, gate + bf16_lo(bw))
                                    txl.assign(up, up + bf16_hi(bw))
                                pw = txl.local_scalar("uint32")
                                pack(pw, gate, up)
                                hv = txl.local_scalar("float32")
                                txl.assign(hv, act2(bf16_lo(pw), bf16_hi(pw), clamped, alpha, limit))
                                hv2 = shx(hv, 2)
                                hw = txl.local_scalar("uint32")
                                pack(hw, hv, hv2)
                                with txl.if_(ok & even), txl.then_():
                                    txl.ptx.st.global_.u32(p["preact"].ptr_to([r * inter + f // 2]), pw)
                                with txl.if_(ok & ((lane % 4) == 0)), txl.then_():
                                    txl.ptx.st.global_.u32(p["h"].ptr_to([r * (inter // 2) + f // 4]), hw)
                            else:  # down_dgrad: f = i, dh = v[j]
                                g = txl.local_scalar("float32")
                                u = txl.local_scalar("float32")
                                txl.assign(g, bf16_lo(ld[j]))
                                txl.assign(u, bf16_hi(ld[j]))
                                dh = v[j]
                                one = txl.float32(1.0)
                                sig = txl.local_scalar("float32")
                                hv = txl.local_scalar("float32")
                                if not clamped:
                                    txl.assign(sig, sigmoid(g))
                                    dg = dh * u * sig * (one + g * (one - sig))
                                    du = dh * g * sig
                                    txl.assign(hv, g * sig * u)
                                else:
                                    gc = txl.local_scalar("float32")
                                    uc = txl.local_scalar("float32")
                                    txl.assign(gc, txl.Select(g < limit, g, limit))
                                    txl.assign(uc, txl.Select(u < limit, txl.Select(u > -limit, u, -limit), limit))
                                    txl.assign(sig, sigmoid(alpha * gc))
                                    dg = dh * (uc + one) * (sig + alpha * gc * sig * (one - sig))
                                    dg = txl.Select(g <= limit, dg, txl.float32(0.0))
                                    du = txl.Select((u <= limit) & (u >= -limit), dh * gc * sig, txl.float32(0.0))
                                    txl.assign(hv, (uc + one) * gc * sig)
                                dgv = txl.local_scalar("float32")
                                duv = txl.local_scalar("float32")
                                txl.assign(dgv, dg)
                                txl.assign(duv, du)
                                dw1 = txl.local_scalar("uint32")
                                pack(dw1, txl.Select(sw, duv, dgv), txl.Select(sw, dgv, duv))
                                hv2 = shx(hv, 1)
                                hw = txl.local_scalar("uint32")
                                pack(hw, hv, hv2)
                                with txl.if_(ok), txl.then_():
                                    txl.ptx.st.global_.u32(p["dpre"].ptr_to([r * inter + f]), dw1)
                                with txl.if_(ok & even), txl.then_():
                                    txl.ptx.st.global_.u32(p["h"].ptr_to([r * (inter // 2) + f // 2]), hw)
                        if kind == "down_dgrad":
                            for j in range(32):
                                txl.assign(ld[j], ldn[j])
                        if STG:
                            # the 4 quadrant warps' features of each token are now one 256-byte
                            # row segment: each warp scatters 8 whole token rows
                            txl.ptx.bar.sync(txl.uint32(1 + half), txl.uint32(128))
                            f0 = walk.nt * BN + cbx * HN
                            for t in range(8):
                                jj = ew * 8 + t
                                d = txl.local_scalar("int32")
                                txl.ptx.shfl_sync.idx.b32(d, dst, txl.Cast("uint32", jj), txl.uint32(31), FULL)
                                q0 = txl.local_scalar("uint32")
                                q1 = txl.local_scalar("uint32")
                                txl.ptx.ld.shared.v2.b32(q0, q1, sbuf.ptr_to([sb + jj * 64 + 2 * lane]))
                                with txl.if_((rb + jj < walk.hi) & (f0 + 4 * lane < N)), txl.then_():
                                    txl.ptx.st.global_.v2.b32(
                                        p["out"].ptr_to([d * (N // 2) + f0 // 2 + 2 * lane]), q0, q1)

            if PF:
                # zero this warp's preact tile once (reads of chunks past N see initialized smem)
                for z in range(8):
                    txl.ptx.st.shared.v4.b32(pbuf.ptr_to([pb + z * 128 + lane * 4]), txl.uint32(0),
                                             txl.uint32(0), txl.uint32(0), txl.uint32(0))
                txl.cuda.warp_sync()
                pw = TileWalk()
                with txl.if_(pw.t < total), txl.then_():
                    pw.locate()
                    prefetch(pw, 0)
            with txl.while_(walk.t < total):
                walk.locate()
                walk.fetch_next()
                tk = range_v("e-wait-acc")
                acc_pipe.full.wait(acc.stage, acc.phase)
                end(tk)
                tk2 = range_v("e-body")
                txl.ptx.tcgen05.fence__after_thread_sync()
                if SWAP:
                    swap_epilogue(walk, acc)
                else:
                    if not k_grouped:
                        r = walk.lo + walk.mt * BMC + cbx * BM + m
                        valid = r < walk.hi
                        if kind == "down_fwd":
                            txl.assign(dst, 0)
                            with txl.if_(valid), txl.then_():
                                txl.ptx.ld.global_.s32(dst, p["slot_of"].ptr_to([r]))

                    else:
                        j = walk.mt * BMC + cbx * BM + m
                        valid = j < MO
                        if MG2:
                            # address lane a = dpre lane ^ flip: rows 0-15 lane 0, 16-31 lane 1
                            srow = ((lane % 2) ^ p["flip"]) * 16 + lane // 2
                        else:
                            srow = (lane ^ p["flip"]) if kind == "gu_wgrad" else lane
                        j0 = walk.mt * BMC + cbx * BM + ew * 32
                    # runtime chunk loop: one copy of the chunk body keeps the epilogue in the I-cache
                    with txl.serial(CH) as c:
                        col = half * (CH * 32) + c * 32
                        n0 = walk.nt * BNT + col
                        if PF:
                            # this chunk's preact landed in pbuf: registers, then
                            # prefetch the next chunk (or the next tile's first)
                            txl.ptx["cp.async.wait_group"](0)
                            txl.cuda.warp_sync()
                            # (chunks past N were never prefetched: they read the zero-initialized
                            # tile, values unused; an n0 < N guard here makes ptxas spill)
                            for c2 in range(8):
                                txl.ptx.ld.shared.v4.b32(ld[4 * c2], ld[4 * c2 + 1], ld[4 * c2 + 2],
                                                         ld[4 * c2 + 3], paddr(lane, c2))
                            txl.cuda.warp_sync()
                            with txl.if_(c + 1 < CH):
                                with txl.then_():
                                    prefetch(walk, c + 1)
                                with txl.else_():
                                    if DYN:
                                        txl.assign(pw.t, walk.tn)
                                    else:
                                        pw.advance()
                                    with txl.if_(pw.t < total), txl.then_():
                                        pw.locate()
                                        prefetch(pw, 0)
                        with txl.if_((n0 < N) if BNT == BN else ((n0 < N) & (col < BNT))), txl.then_():
                            taddr = tmem_base + txl.Cast("uint32", (ew * 32) << 16) + txl.Cast(
                                "uint32", acc.stage * BN + col
                            )
                            if BNT % 32 == 0:
                                txl.ptx[TMEM_LD32](*[v[i] for i in range(32)], taddr)
                            else:
                                # a 240-column tile's last chunk: only 16 accumulator columns
                                with txl.if_(col + 32 <= BNT):
                                    with txl.then_():
                                        txl.ptx[TMEM_LD32](*[v[i] for i in range(32)], taddr)
                                    with txl.else_():
                                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                                            *[v[i] for i in range(16)], taddr)
                                        for i in range(16, 32):
                                            txl.assign(v[i], txl.float32(0.0))
                            txl.ptx.tcgen05.wait__ld.sync.aligned()
                            if k_grouped:
                                # fp32 main grads: stage the 32x32 block, TMA-reduce-add it (OOB
                                # rows / columns are clipped by the tensor map)
                                slot = ew * 2 + (c % 2)
                                with txl.if_(lane == 0), txl.then_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(1)
                                txl.cuda.warp_sync()
                                for q in range(8):
                                    txl.ptx.st.shared.v4.f32(
                                        cbuf.ptr_to([slot * 1024 + srow * 32 + ((q ^ (srow % 8)) * 4)]),
                                        v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3],
                                    )
                                txl.ptx.fence.proxy.async_.shared__cta()
                                txl.cuda.warp_sync()
                                with txl.if_(lane == 0), txl.then_():
                                    if MG2:
                                        for half, cm in ((0, maps["c"]), (1, maps["c2"])):
                                            txl.ptx["cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"](
                                                txl.address_of(cm), txl.Cast("int32", n0),
                                                txl.Cast("int32", j0 // 2), txl.Cast("int32", walk.g),
                                                cbuf.ptr_to([slot * 1024 + half * 512]),
                                            )
                                    elif kind == "gu_wgrad":
                                        txl.ptx["cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"](
                                            txl.address_of(maps["c"]), txl.Cast("int32", n0), txl.int32(0),
                                            txl.Cast("int32", j0 // 2), txl.Cast("int32", walk.g),
                                            cbuf.ptr_to([slot * 1024]),
                                        )
                                    else:
                                        txl.ptx["cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"](
                                            txl.address_of(maps["c"]), txl.Cast("int32", n0),
                                            txl.Cast("int32", j0), txl.Cast("int32", walk.g),
                                            cbuf.ptr_to([slot * 1024]),
                                        )
                                    txl.ptx.cp.async_.bulk.commit_group()
                                txl.cuda.warp_sync()
                            else:
                                if XSTAGE:
                                    rb = walk.lo + walk.mt * BMC + cbx * BM + ew * 32
                                    ok = lambda row: rb + row < walk.hi  # noqa: E731
                                    full = (rb + 32 <= walk.hi) if TMA_ST else None
                                    i32c = lambda e: txl.Cast("int32", e)  # noqa: E731

                                    def tma_st(m, c0):
                                        if not TMA_ST:
                                            return None
                                        return lambda sp: txl.ptx[TMA_ST2](txl.address_of(maps[m]), i32c(c0), i32c(rb), sp)
                                    if kind == "down_dgrad":
                                        dgrad_math(p, ld, v, dw, w16, clamped, alpha, limit)
                                        # 240-column tiles: the last chunk holds 16 columns (i)
                                        part = BNT % 32 != 0
                                        if SPLIT_DPRE:
                                            for hh in range(2):
                                                stage_store(Off(dw, 16 * hh), 4, lambda row, ch, hh=hh: p["dpre"].ptr_to([(rb + row) * inter + n0 + 16 * hh + ch * 4]), ok,
                                                            (lambda ch, hh=hh: col + 16 * hh + ch * 4 < BNT) if part else None,
                                                            tma_st("d", 2 * (n0 + 16 * hh)), full)
                                        else:
                                            stage_store(dw, 8, lambda row, ch: p["dpre"].ptr_to([(rb + row) * inter + n0 + ch * 4]), ok,
                                                        (lambda ch: col + ch * 4 < BNT) if part else None,
                                                        tma_st("d", 2 * n0), full)
                                        stage_store(w16, 4, lambda row, ch: p["h"].ptr_to([(rb + row) * (inter // 2) + n0 // 2 + ch * 4]), ok,
                                                    (lambda ch: col + ch * 8 < BNT) if part else None,
                                                    tma_st("e", n0), full)
                                    elif kind == "gu_fwd":
                                        fwd_math(p, walk, n0, v, w16, w8, ld, has_bias, clamped, alpha, limit)
                                        stage_store(w16, 4, lambda row, ch: p["preact"].ptr_to([(rb + row) * inter + n0 // 2 + ch * 4]), ok,
                                                    None, tma_st("d", n0), full)
                                        stage_store(w8, 2, lambda row, ch: p["h"].ptr_to([(rb + row) * (inter // 2) + n0 // 4 + ch * 4]), ok,
                                                    None, tma_st("e", n0 // 2), full)
                                    else:
                                        plain_math(p, walk, n0, v, w16, ld, has_bias)
                                        # 240-column tiles: the last chunk holds 16 columns
                                        cm = None if BNT % 32 == 0 else (lambda ch: col + ch * 8 < BNT)
                                        if kind == "gu_dgrad":
                                            stage_store(w16, 4, lambda row, ch: p["dx"].ptr_to([(rb + row) * (N // 2) + n0 // 2 + ch * 4]), ok, cm,
                                                        tma_st("d", n0), full)
                                        else:
                                            def out_addr(row, ch):
                                                d = txl.local_scalar("int32")
                                                txl.ptx.shfl_sync.idx.b32(d, dst, txl.Cast("uint32", row), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                                                return p["out"].ptr_to([d * (N // 2) + n0 // 2 + ch * 4])
                                            stage_store(w16, 4, out_addr, ok, cm)

                txl.ptx.tcgen05.fence__before_thread_sync()
                txl.cuda.warp_sync()
                with txl.if_(lane == 0), txl.then_():
                    acc_pipe.empty.arrive(acc.stage, remote=0)
                end(tk2)
                acc.advance()
                walk.advance()
            if k_grouped or TMA_ST:
                with txl.if_(lane == 0), txl.then_():
                    txl.ptx.cp.async_.bulk.wait_group(0)
                txl.cuda.warp_sync()

        txl.ptx.tcgen05.fence__before_thread_sync()
        txl.cuda.cluster_sync()
        with txl.if_(warp == 1), txl.then_():
            txl.ptx.tcgen05.fence__after_thread_sync()
            dealloc = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](dealloc, txl.uint32(512))
        if DYN:
            # completion order: this grid ends only after the previous one
            txl.ptx.griddepcontrol.wait()

    def fwd_math(p, walk, n0, v, w16, w8, ld, has_bias, clamped, alpha, limit):
        """
        gate_up fwd chunk: preact words w16[16] ((gate, up) order) and h words w8[8].
        """
        sw = p["swapped"] != 0
        if has_bias:
            for q in range(4):
                txl.ptx.ld.global_.v4.b32(
                    ld[4 * q], ld[4 * q + 1], ld[4 * q + 2], ld[4 * q + 3],
                    p["bias"].ptr_to([walk.g * inter + n0 // 2 + 4 * q]),
                )
        for q in range(16):
            gate = txl.local_scalar("float32")
            up = txl.local_scalar("float32")
            txl.assign(gate, txl.Select(sw, v[2 * q + 1], v[2 * q]))
            txl.assign(up, txl.Select(sw, v[2 * q], v[2 * q + 1]))
            if has_bias:
                txl.assign(gate, gate + bf16_lo(ld[q]))
                txl.assign(up, up + bf16_hi(ld[q]))
            pack(w16[q], gate, up)
        for q in range(8):
            hv0 = txl.local_scalar("float32")
            hv1 = txl.local_scalar("float32")
            txl.assign(hv0, act2(bf16_lo(w16[2 * q]), bf16_hi(w16[2 * q]), clamped, alpha, limit))
            txl.assign(hv1, act2(bf16_lo(w16[2 * q + 1]), bf16_hi(w16[2 * q + 1]), clamped, alpha,
                                 limit))
            pack(w8[q], hv0, hv1)

    def plain_math(p, walk, n0, v, w16, ld, has_bias):
        """
        down fwd / gate_up dgrad chunk: (+ bias) rounded to bf16 words w16[16].
        """
        if has_bias:
            for q in range(4):
                # 8 columns per load; a 240-column tile's last chunk may end past N
                ok = n0 + 8 * q < N
                txl.ptx.ld.global_.v4.b32(
                    ld[4 * q], ld[4 * q + 1], ld[4 * q + 2], ld[4 * q + 3],
                    p["bias"].ptr_to([txl.Select(ok, (walk.g * N + n0) // 2 + 4 * q, 0)]),
                )
        for q in range(16):
            a0 = txl.local_scalar("float32")
            a1 = txl.local_scalar("float32")
            txl.assign(a0, v[2 * q])
            txl.assign(a1, v[2 * q + 1])
            if has_bias:
                txl.assign(a0, a0 + bf16_lo(ld[q]))
                txl.assign(a1, a1 + bf16_hi(ld[q]))
            pack(w16[q], a0, a1)

    def dgrad_math(p, ld, v, dw, w16, clamped, alpha, limit):
        """
        dpre words dw[32] (weight address order) and h words w16[16] of one 32-column chunk:
        ld = 32 (gate, up) preact words, v = 32 fp32 dh. One sigmoid per element serves both
        the gradient and h (h keeps the forward's exact expression order).
        """
        sw = p["swapped"] != 0
        hv = txl.alloc_local((32,), "float32")
        one = txl.float32(1.0)
        for e in range(32):
            g = txl.local_scalar("float32")
            u = txl.local_scalar("float32")
            txl.assign(g, bf16_lo(ld[e]))
            txl.assign(u, bf16_hi(ld[e]))
            dh = v[e]
            sig = txl.local_scalar("float32")
            if not clamped:
                txl.assign(sig, sigmoid(g))
                dg = dh * u * sig * (one + g * (one - sig))
                du = dh * g * sig
                txl.assign(hv[e], g * sig * u)
            else:
                gc = txl.local_scalar("float32")
                uc = txl.local_scalar("float32")
                txl.assign(gc, txl.Select(g < limit, g, limit))
                txl.assign(uc, txl.Select(u < limit, txl.Select(u > -limit, u, -limit), limit))
                txl.assign(sig, sigmoid(alpha * gc))
                dg = dh * (uc + one) * (sig + alpha * gc * sig * (one - sig))
                dg = txl.Select(g <= limit, dg, txl.float32(0.0))
                du = txl.Select((u <= limit) & (u >= -limit), dh * gc * sig, txl.float32(0.0))
                txl.assign(hv[e], (uc + one) * gc * sig)
            dgv = txl.local_scalar("float32")
            duv = txl.local_scalar("float32")
            txl.assign(dgv, dg)
            txl.assign(duv, du)
            pack(dw[e], txl.Select(sw, duv, dgv), txl.Select(sw, dgv, duv))
        for q in range(16):
            pack(w16[q], hv[2 * q], hv[2 * q + 1])

    def entry(*args):
        p = dict(zip(names, args, strict=True))
        maps = host_prelude(p)
        # 2-CTA clusters; programmatic stream serialization (PDL) on every launch
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=p["num_ctas"], block=NWARPS * 32, cluster=[2, 1], preferred_cluster=[2, 1], programmatic_stream_serialization=True),
                         kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1))
        body(p, maps)

    entry.__name__ = f"bf16_expert_{kind}"
    entry.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=a) for n, a in params]
    )
    kern = txl.kernel()(entry)
    if IKET:
        from tvm_tirx.backend.cuda.iket import IketProfiler
        with TARGET:
            return IketProfiler().compile(kern.mod, target=TARGET)
    return entry_function(kern.compile(TARGET))


@functools.cache
def build_gather(H):
    """
    out[r, :] = src[idx[r], :] for r < M: one warp per row, 16 bytes per lane.
    """
    CH = H // 8  # 16-byte chunks per row
    WPC = 8

    def gather_rows(src: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1), idx: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), out: txl.Tensor((txl.int64(),), txl.u32, data_alignment=1),
                    M: txl.int32, num_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=WPC * 32, programmatic_stream_serialization=True))
        warp = txl.warp_id()
        lane = txl.lane_id()
        r = txl.local_scalar("int32", init=txl.cta_id() * WPC + warp)
        row = txl.local_scalar("int32")
        buf = txl.alloc_local((4 * (-(-CH // 32)),), "uint32")
        txl.ptx.griddepcontrol.wait()
        txl.ptx.griddepcontrol.launch_dependents()
        with txl.while_(r < M):
            txl.ptx.ld.global_.s32(row, idx.ptr_to([r]))
            for i in range(-(-CH // 32)):
                c = i * 32 + lane
                with txl.if_(c < CH), txl.then_():
                    txl.ptx.ld.global_.nc.v4.b32(
                        buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3],
                        src.ptr_to([row * (H // 2) + c * 4]),
                    )
            for i in range(-(-CH // 32)):
                c = i * 32 + lane
                with txl.if_(c < CH), txl.then_():
                    txl.ptx.st.global_.v4.b32(
                        out.ptr_to([r * (H // 2) + c * 4]),
                        buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3],
                    )
            txl.assign(r, r + num_ctas * WPC)

    return entry_function(txl.kernel()(gather_rows).compile(TARGET))


def gather(src, idx):
    """
    Contiguous src[idx] (M, H) through the TIRx-lite row-gather kernel.
    """
    M, H = idx.shape[0], src.shape[1]
    out = src.new_empty(M, H)
    ctas = max(1, min(4 * num_sms(), -(-M // 8)))
    build_gather(H)(src.reshape(-1).view(torch.uint32), idx, out.reshape(-1).view(torch.uint32),
                     M, ctas)
    return out


# ---------------------------------------------------------------------------- host side

@functools.cache
def num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def flat(t):
    """
    A contiguous 1-D view over t's storage from t's first element to its last.
    """
    span = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    return t.as_strided((span,), (1,))


TMA_STRIDE_LIMIT = 1 << 40  # cuTensorMapEncodeTiled: every global stride below 2^40 bytes


def pair2(a, b):
    """
    (lo, hi, diff_elems, swapped, far): b below a -> swapped; far when the two buffers are too
    far apart for one tensor map to stride between them (independent allocations).
    """
    if a.stride() != b.stride() or a.stride(-1) != 1:
        raise NotImplementedError("the gate / up pair must share strides, unit-stride rows")
    swapped = b.data_ptr() < a.data_ptr()
    lo, hi = (b, a) if swapped else (a, b)
    nbytes = hi.data_ptr() - lo.data_ptr()
    if nbytes % 16 or lo.data_ptr() % 16:
        raise NotImplementedError("the gate / up pair must be 16-byte aligned, a multiple of 16 bytes apart")
    return lo, hi, nbytes // hi.element_size(), swapped, nbytes >= TMA_STRIDE_LIMIT


def grid2(M, G, k_grouped, per_group_tiles, NT, dyn=False):
    """
    CTA count: two per cluster, at most one cluster per SM pair or per tile.
    """
    if G > GMAX:
        raise NotImplementedError(f"at most {GMAX} experts per call (smem tile prefix), got {G}")
    if k_grouped:
        ub = G * per_group_tiles
    else:
        ub = (-(-M // BMC) + G) * NT
    if dyn:
        return 2 * max(1, ub)
    return 2 * max(1, min(num_sms() // 2, ub))


# Kernel overlap (the wgrads): every launch here uses programmatic dependent launch and triggers
# its dependents right after its prologue, and every launch that waits (griddepcontrol.wait) does
# so before its first global access. The overlap variants skip that wait and wait at exit
# instead, so they may run alongside every launch since (and including) the last waiting one on
# the stream: WINDOW holds those launches' byte extents (reads, writes), and an overlap variant
# is chosen only if it neither reads what they write nor writes what they touch. Ordinary
# (non-PDL) stream work in between only adds ordering. Assumes no foreign kernel that triggers
# its dependents early sits right before an overlap launch.
WINDOW = {}


def ext(*ts):
    """
    Byte extents [lo, hi) of tensors' storage spans (empty tensors skipped).
    """
    out = []
    for t in ts:
        if t is None or t.numel() == 0:
            continue
        span = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
        out.append((t.data_ptr(), t.data_ptr() + span * t.element_size()))
    return out


def hit(xs, ys):
    return any(a < d and c < b for a, b in xs for c, d in ys)


def stream_key():
    stream = torch.cuda.current_stream()
    return stream.device_index, stream.cuda_stream  # per (device, stream): stream handles may repeat across devices


def may_overlap(reads, writes):
    """
    An overlap launch with these extents is independent of every launch that may still run.
    """
    win = WINDOW.get(stream_key())
    if not win or len(win) >= 4:
        return False
    return not any(hit(writes, r) or hit(writes, w) or hit(reads, w) for r, w in win)


def track(reads, writes, overlapped):
    key = stream_key()
    if overlapped:
        WINDOW[key].append((reads, writes))
    else:
        WINDOW[key] = [(reads, writes)]


def u322(t):
    """
    bf16 tensor -> flat uint32 view of the same bytes.
    """
    return t.reshape(-1).view(torch.uint32)


def check_shapes(H, inter):
    assert H % 64 == 0 and inter % 32 == 0, (H, inter)


def bf16_expert_gate_up_fwd_sm100(x, w_gate, w_up, cu, a_idx, bias=None, act=()):
    M, G, inter, H = a_idx.shape[0], w_gate.shape[0], w_gate.shape[1], w_gate.shape[2]
    preact = x.new_empty(M, 2 * inter)
    h = x.new_empty(M, inter)
    if M == 0:
        return preact, h
    check_shapes(H, inter)
    lo, hi, diff, swapped, far = pair2(w_gate, w_up)
    clamped = bool(act)
    materialized = use_materialized_gu_fwd(M, G, H, inter)
    fn = build("gu_fwd", H, inter, bias is not None, clamped,
                use_swap(M, G, SWAP_ROWS_GU_FWD) and swap_ok("gu_fwd", inter), materialized, far)
    NT = -(-2 * inter // BN)
    xsrc = gather(x, a_idx) if materialized else x
    args = [cu, G, M, grid2(M, G, False, 0, NT), xsrc.reshape(-1), xsrc.shape[0], a_idx, flat(lo),
            diff, lo.stride(0), lo.stride(1), u322(preact), u322(h), int(swapped)]
    if bias is not None:
        args.append(u322(bias.contiguous()))
    if clamped:
        args += [float(act[0]), float(act[1])]
    fn(*args)
    track(ext(x, xsrc, lo, hi, cu, a_idx, bias), ext(preact, h), False)
    return preact, h


def bf16_expert_down_fwd_sm100(h, w_down, cu, slot_of, bias=None):
    M, G, H, inter = h.shape[0], w_down.shape[0], w_down.shape[1], w_down.shape[2]
    out = h.new_empty(M, H)
    if M == 0:
        return out
    check_shapes(H, inter)
    sw = use_swap(M, G) and swap_ok("down_fwd", inter)
    fn = build("down_fwd", H, inter, bias is not None, False, sw)
    NT = -(-H // bnt("down_fwd", H, sw, False))
    args = [cu, G, M, grid2(M, G, False, 0, NT), h.reshape(-1), w_down.reshape(-1), slot_of,
            u322(out)]
    if bias is not None:
        args.append(u322(bias.contiguous()))
    fn(*args)
    track(ext(h, w_down, cu, slot_of, bias), ext(out), False)
    return out


def bf16_expert_down_dgrad_sm100(dy, w_down, preact, cu, slot_of, w_gate, w_up, act=()):
    M, inter = preact.shape[0], preact.shape[1] // 2
    G, H = w_down.shape[0], w_down.shape[1]
    swapped = w_up.data_ptr() < w_gate.data_ptr()
    dpre = preact.new_empty(M, 2 * inter)
    h = preact.new_empty(M, inter)
    if M == 0:
        return dpre, h, swapped
    check_shapes(H, inter)
    clamped = bool(act)
    fn = build("down_dgrad", H, inter, False, clamped)
    NT = -(-inter // bnt("down_dgrad", inter, False, False))
    args = [cu, G, M, grid2(M, G, False, 0, NT), dy.reshape(-1), dy.shape[0], slot_of,
            w_down.reshape(-1), u322(preact), u322(dpre), u322(h), int(swapped)]
    if clamped:
        args += [float(act[0]), float(act[1])]
    fn(*args)
    track(ext(dy, w_down, preact, cu, slot_of), ext(dpre, h), False)
    return dpre, h, swapped


def bf16_expert_gate_up_dgrad_sm100(dpre, w_gate, w_up, cu, dpre_swapped):
    M, inter = dpre.shape[0], dpre.shape[1] // 2
    G, H = w_gate.shape[0], w_gate.shape[2]
    dx = dpre.new_empty(M, H)
    if M == 0:
        return dx
    check_shapes(H, inter)
    lo, hi, diff, swapped, far = pair2(w_gate, w_up)
    if bool(swapped) != bool(dpre_swapped):
        raise NotImplementedError("dpre pair order must follow the weights' address order")
    sw = use_swap(M, G) and swap_ok("gu_dgrad", inter)
    fn = build("gu_dgrad", H, inter, False, False, sw, False, far)
    NT = -(-H // bnt("gu_dgrad", H, sw, far))
    args = [cu, G, M, grid2(M, G, False, 0, NT), dpre.reshape(-1), flat(lo), diff,
            lo.stride(0), lo.stride(1), u322(dx)]
    fn(*args)
    track(ext(dpre, lo, hi, cu), ext(dx), False)
    return dx


def bf16_expert_gate_up_wgrad_sm100(x, dpre, dpre_swapped, mg_gate, mg_up, cu, a_idx):
    M = a_idx.shape[0]
    if M == 0:
        return None
    G, inter, H = mg_gate.shape
    check_shapes(H, inter)
    lo, hi, diff, mg_swapped, far = pair2(mg_gate, mg_up)
    reads, writes = ext(x, dpre, cu, a_idx), ext(lo, hi)
    ovl = may_overlap(reads, writes)
    fn = build("gu_wgrad", H, inter, False, False, False, False, far, ovl)
    MT, NT = -(-2 * inter // BMC), -(-H // BN)
    # dpre lane l is the gate iff l == dpre_swapped; main-grad address lane a is the gate iff
    # a == mg_swapped, so a = l ^ flip
    flip = int(bool(dpre_swapped)) ^ int(mg_swapped)
    args = [cu, G, M, grid2(M, G, True, MT * NT, NT, ovl), dpre.reshape(-1), x.reshape(-1),
            x.shape[0], a_idx, flat(lo)] + ([flat(hi)] if far else []) + [diff, lo.stride(0),
                                                                             lo.stride(1), flip]
    fn(*args)
    track(reads, writes, ovl)
    return None


def bf16_expert_down_wgrad_sm100(dy, h, mg_down, cu, slot_of):
    M = slot_of.shape[0]
    if M == 0:
        return None
    G, H, inter = mg_down.shape
    assert mg_down.is_contiguous()
    check_shapes(H, inter)
    reads, writes = ext(dy, h, cu, slot_of), ext(mg_down)
    ovl = may_overlap(reads, writes)
    fn = build("down_wgrad", H, inter, False, False, False, False, False, ovl)
    MT, NT = -(-H // BMC), -(-inter // BN)
    args = [cu, G, M, grid2(M, G, True, MT * NT, NT, ovl), dy.reshape(-1), dy.shape[0], slot_of,
            h.reshape(-1), mg_down.reshape(-1)]
    fn(*args)
    track(reads, writes, ovl)
    return None
