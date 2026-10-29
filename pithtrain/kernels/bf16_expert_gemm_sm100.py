"""
SM100 TIRx-lite kernels for pith-train's fused BF16 expert grouped GEMMs.

One persistent, warp-specialized tcgen05 grouped-GEMM template covers the six entry points:

* M-grouped (rows of expert g are cu[g] <= r < cu[g+1], fixed K): gate_up_fwd, down_fwd,
  down_dgrad, gate_up_dgrad.
* K-grouped (expert g reduces over its rows, fixed output): gate_up_wgrad, down_wgrad.

Warp 0 loads with TMA (tile / tile::gather4 for the a_idx / slot_of row gathers), warp 1 issues
tcgen05.mma (M=128, cta_group::1) into a double-buffered TMEM accumulator, warps 2-5 run the
fused epilogue (bias, SwiGLU / its backward, scatter, fp32 main-grad accumulation).

The two gate / up weights are read through one rank-4 tensor map (H, 2, I, G) whose second mode
strides between the two weights in address order, so the MMA sees the interleaved
[w_lo[0]; w_hi[0]; w_lo[1]; ...] rows: the accumulator columns are the interleaved preact
pairs, and the gate_up dgrad contracts the interleaved dpre directly.

Evolved with TIRx-harness: run bf16gemm, frontier member cluster2-cpasync-tmareduce (commit 3e76cbd, 6.777x over the
previous SM100 path in the task's benchmark). Compiled once per process by tvm.compile(...,
tir_pipeline="tirx") and launched through tvm-ffi on torch's current stream.
"""

import functools
import inspect

import torch
import tvm
import tvm_ffi

import tirx_kernels.tirx_lite as txl

__all__ = ["bf16_expert_gate_up_fwd_sm100", "bf16_expert_down_fwd_sm100", "bf16_expert_down_dgrad_sm100", "bf16_expert_gate_up_dgrad_sm100", "bf16_expert_gate_up_wgrad_sm100", "bf16_expert_down_wgrad_sm100"]

BM = 128  # rows per CTA; a 2-CTA cluster covers BMC = 256
BMC = 256
BK = 64
BN = 256  # cluster N; each CTA stages HN = 128 of B
HN = 128
STAGES = 6
GMAX = 1024
WARPS = 6
EPI_WARP0 = 2
IKET = False

MMA = "tcgen05.mma.cta_group::2.kind::f16"
TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMA = "cp.async.bulk.tensor.{}d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
CP16 = "cp.async.cg.shared.global.L2::128B"
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
    return txl.reinterpret("float32", word << txl.uint32(16))


def bf16_hi(word):
    return txl.reinterpret("float32", word & txl.uint32(0xFFFF0000))


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


def dact(g, u, dh, clamped, alpha, limit):
    """
    (dg, du) of the saved bf16 gate / up and the fp32 dh.
    """
    if not clamped:
        sig = sigmoid(g)
        dg = dh * u * sig * (txl.float32(1.0) + g * (txl.float32(1.0) - sig))
        du = dh * g * sig
        return dg, du
    gc = txl.Select(g < limit, g, limit)
    uc = txl.Select(u < limit, txl.Select(u > -limit, u, -limit), limit)
    sig = txl.local_scalar("float32")
    txl.assign(sig, sigmoid(alpha * gc))
    dg = dh * (uc + txl.float32(1.0)) * (sig + alpha * gc * sig * (txl.float32(1.0) - sig))
    dg = txl.Select(g <= limit, dg, txl.float32(0.0))
    du = txl.Select((u <= limit) & (u >= -limit), dh * gc * sig, txl.float32(0.0))
    return dg, du


def params2(kind, has_bias, clamped):
    i32, i64, f32 = "int32", "int64", "float32"
    bf = txl.gptr["bfloat16"]
    u32 = txl.gptr["uint32"]
    p = [("cu", txl.gptr["int32"]), ("G", i32), ("M", i32), ("num_ctas", i32)]
    if kind == "gu_fwd":
        p += [("x", bf), ("T", i32), ("a_idx", txl.gptr["int32"]), ("w", bf), ("wdiff", i64),
              ("ws0", i64), ("ws1", i64), ("preact", u32), ("h", u32), ("swapped", i32)]
        if has_bias:
            p += [("bias", u32)]
    elif kind == "down_fwd":
        p += [("hin", bf), ("wd", bf), ("slot_of", txl.gptr["int32"]), ("out", u32)]
        if has_bias:
            p += [("bias", u32)]
    elif kind == "down_dgrad":
        p += [("dy", bf), ("T", i32), ("slot_of", txl.gptr["int32"]), ("wd", bf),
              ("preact", u32), ("dpre", u32), ("h", u32), ("swapped", i32)]
    elif kind == "gu_dgrad":
        p += [("dpre", bf), ("w", bf), ("wdiff", i64), ("ws0", i64), ("ws1", i64), ("dx", u32)]
    elif kind == "gu_wgrad":
        p += [("dpre", bf), ("x", bf), ("T", i32), ("a_idx", txl.gptr["int32"]),
              ("mg", txl.gptr["float32"]), ("mgdiff", i64), ("ms0", i64), ("ms1", i64),
              ("flip", i32)]
    elif kind == "down_wgrad":
        p += [("dy", bf), ("T", i32), ("slot_of", txl.gptr["int32"]), ("hin", bf),
              ("mg", txl.gptr["float32"])]
    if clamped and kind in ("gu_fwd", "down_dgrad"):
        p += [("alpha", f32), ("limit", f32)]
    return p


@functools.cache
def build(kind, H, inter, has_bias, clamped):
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
    NT = -(-N // BN)
    KT = None if k_grouped else -(-K // BK)
    MT = -(-MO // BMC) if k_grouped else None
    A_BYTES = BM * BK * 2
    B_BYTES = HN * BK * 2
    GATHER = kind in ("gu_fwd", "down_dgrad", "gu_wgrad", "down_wgrad")
    # the gathered operand: A for gu_fwd / down_dgrad / down_wgrad, B for gu_wgrad
    TMA_BYTES = (A_BYTES + B_BYTES) - ((B_BYTES if kind == "gu_wgrad" else A_BYTES) if GATHER else 0)
    # the SwiGLU-heavy epilogues get two warps per TMEM lane quadrant (column halves)
    NEPI = 8 if kind in ("gu_fwd", "down_dgrad") else 4
    # longest-group-first tile order (helps down_wgrad's tail; measured worse for gu_wgrad)
    LPT = kind == "down_wgrad"
    # the wgrads read rows a gather kernel already laid out contiguously (row r is row r)
    PREGATHERED = k_grouped
    CH = (BN // 32) // (NEPI // 4)  # 32-column chunks per epilogue warp
    GW0 = EPI_WARP0 + NEPI
    NWARPS = GW0 + (4 if GATHER else 0)
    # wgrads keep 32 KB of smem for the fp32 TMA-reduce staging of the epilogue
    NST = STAGES - 1 if k_grouped else STAGES
    a_shape = (NST, BM, BK) if a_major == "k" else (NST, BK, BM)
    b_shape = (NST, HN, BK) if b_major == "k" else (NST, BK, HN)
    a_idesc_trans = a_major == "mn"
    b_idesc_trans = b_major == "mn"
    params = params2(kind, has_bias, clamped)
    names = [n for n, _ in params]

    def enc(m, rank, data, *ops, dtype="bfloat16"):
        txl.cu_tensor_map_encode_tiled(m, dtype, rank, data, *ops)

    def host_prelude(p):
        maps = {}
        if kind == "gu_fwd":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["x"].data, H, p["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 4, p["w"].data, H, 2, inter, p["G"], p["wdiff"] * 2, p["ws1"] * 2,
                p["ws0"] * 2, 64, 2, HN // 2, 1, 1, 1, 1, 1, 0, 3, 2, 0)
        elif kind == "down_fwd":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["hin"].data, inter, p["M"], inter * 2, 64, BM, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 3, p["wd"].data, inter, H, p["G"], inter * 2, H * inter * 2, 64, HN, 1,
                1, 1, 1, 0, 3, 2, 0)
        elif kind == "down_dgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["dy"].data, H, p["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 3, p["wd"].data, inter, H, p["G"], inter * 2, H * inter * 2, 64, BK, 1,
                1, 1, 1, 0, 3, 2, 0)
        elif kind == "gu_dgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["dpre"].data, 2 * inter, p["M"], 4 * inter, 64, BM, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 4, p["w"].data, H, 2, inter, p["G"], p["wdiff"] * 2, p["ws1"] * 2,
                p["ws0"] * 2, 64, 2, BK // 2, 1, 1, 1, 1, 1, 0, 3, 2, 0)
        elif kind == "gu_wgrad":
            maps["a"] = txl.stack_alloca("tensormap", 1)
            enc(maps["a"], 2, p["dpre"].data, 2 * inter, p["M"], 4 * inter, 64, BK, 1, 1, 0, 3, 2, 0)
            maps["b"] = txl.stack_alloca("tensormap", 1)
            enc(maps["b"], 2, p["x"].data, H, p["T"], H * 2, 64, 1, 1, 1, 0, 3, 2, 0)
            # main grads as (H, 2, I, G) over the lower buffer: rows (i, address lane)
            maps["c"] = txl.stack_alloca("tensormap", 1)
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
        cbx, cby = txl.cta_id_in_cluster([2, 1], preferred=[2, 1])
        cluster = cta // 2
        warp = txl.warp_id()
        lane = txl.lane_id()

        with txl.If(warp == 0), txl.Then():
            with txl.If(lane == 0), txl.Then():
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
        gfull.init(128)
        order = smem.alloc((GMAX,), "int32", align=16)
        smem.pool.move_base_to(8192 + 4 * (GMAX + 1))
        a_tile = smem.alloc(a_shape, "bfloat16", swizzle=txl.SW128B)
        b_tile = smem.alloc(b_shape, "bfloat16", swizzle=txl.SW128B)
        if k_grouped:
            # per epilogue warp, two 32x32 fp32 staging tiles (128-byte swizzled rows)
            cbuf = smem.alloc((4 * 2 * 32 * 32,), "float32", align=1024)

        # ---- tile prefix over the groups (warp 2), then TMEM allocation (warp 1).
        with txl.If(warp == EPI_WARP0), txl.Then():
            run = txl.local_scalar("int32", init=0)
            base = txl.local_scalar("int32", init=0)
            with txl.If(lane == 0), txl.Then():
                txl.ptx.st.shared.s32(prefix.ptr_to([0]), txl.int32(0))
            if LPT:
                # longest-first group order: rank sort over register-resident counts (warp
                # shuffles); identity order past 128 groups
                with txl.If(G <= 128):
                    with txl.Then():
                        cnts = txl.alloc_local((4,), "int32")
                        for q in range(4):
                            g = q * 32 + lane
                            txl.assign(cnts[q], -1)
                            with txl.If(g < G), txl.Then():
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
                            with txl.If(qh * 32 < G), txl.Then():
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
                            with txl.If(g < G), txl.Then():
                                txl.ptx.st.shared.s32(order.ptr_to([ranks[q]]), g)
                    with txl.Else():
                        with txl.While(base < G):
                            with txl.If(base + lane < G), txl.Then():
                                txl.ptx.st.shared.s32(order.ptr_to([base + lane]), base + lane)
                            txl.assign(base, base + 32)
                txl.cuda.warp_sync()
                txl.assign(base, 0)
            with txl.While(base < G):
                g = txl.local_scalar("int32", init=base + lane)
                cnt = txl.local_scalar("int32", init=0)
                if LPT:
                    with txl.If(g < G), txl.Then():
                        txl.ptx.ld.shared.s32(g, order.ptr_to([base + lane]))
                with txl.If(base + lane < G), txl.Then():
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
                    with txl.If(lane >= off), txl.Then():
                        txl.assign(cnt, cnt + other)
                with txl.If(base + lane < G), txl.Then():
                    txl.ptx.st.shared.s32(prefix.ptr_to([base + lane + 1]), run + cnt)
                last = txl.local_scalar("int32")
                txl.ptx.shfl_sync.idx.b32(last, cnt, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                txl.assign(run, run + last)
                txl.assign(base, base + 32)
        with txl.If(warp == 1), txl.Then():
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

            def locate(self):
                nxt = txl.local_scalar("int32")
                txl.ptx.ld.shared.s32(nxt, prefix.ptr_to([self.k + 1]))
                with txl.While(self.t >= nxt):
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

            def advance(self):
                txl.assign(self.t, self.t + p["num_ctas"] // 2)

        def full_bar(stage):
            return txl.cuda.cvta_generic_to_shared(full_cta0.ptr_to([stage]))

        # ------------------------------------------------------------ TMA producer
        # (the non-gathered operand(s); the gathered one is the cp.async role below)
        with txl.If(warp == 0), txl.Then():
            walk = TileWalk()
            st = txl.PipelineState(NST, 1)
            with txl.While(walk.t < total):
                walk.locate()
                nkt = txl.local_scalar("int32")
                if k_grouped:
                    txl.assign(nkt, (walk.hi - walk.lo + (BK - 1)) // BK)
                else:
                    txl.assign(nkt, KT)
                with txl.If(lane == 0), txl.Then():
                    with txl.serial(nkt) as kt:
                        tk = range_v("p-wait-empty")
                        ab_pipe.empty.wait(st.stage, st.phase)
                        end(tk)
                        s = st.stage
                        with txl.If(cbx == 0), txl.Then():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([s]), txl.uint32(2 * TMA_BYTES)
                            )
                        a_base = a_tile[s].ptr_to(0, 0)
                        b_base = b_tile[s].ptr_to(0, 0)
                        i32 = lambda v: txl.Cast("int32", v)  # noqa: E731
                        if not k_grouped:
                            k = kt * BK
                            mrow = walk.lo + walk.mt * BMC + cbx * BM
                            if not a_gather:
                                txl.ptx[TMA.format(2)](
                                    a_base, txl.address_of(a_map), i32(k), i32(mrow), full_bar(s)
                                )
                            n0 = walk.nt * BN + cbx * HN
                            if kind == "gu_fwd":
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
            with txl.If(warp >= GW0), txl.Then():
                gw = warp - GW0
                walk = TileWalk()
                st = txl.PipelineState(NST, 1)
                src = p["x"] if kind in ("gu_fwd", "gu_wgrad") else p["dy"]
                ids = p["a_idx"] if kind in ("gu_fwd", "gu_wgrad") else p["slot_of"]
                my = txl.local_scalar("int32")  # this lane's row index (-1: zero row)
                row = txl.local_scalar("int32")
                chunk = lane % 8
                sub = lane // 8
                FULLM = txl.uint32(0xFFFFFFFF)

                def load_index(r, hi):
                    txl.assign(my, -1)
                    with txl.If(r < hi), txl.Then():
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

                with txl.While(walk.t < total):
                    walk.locate()
                    if not k_grouped:
                        # K-major A: warp gw owns tile rows 32 gw .. + 31; 8 lanes per row
                        load_index(walk.lo + walk.mt * BMC + cbx * BM + gw * 32 + lane, walk.hi)
                        with txl.serial(KT) as kt:
                            tk = range_v("g-wait-empty")
                            ab_pipe.empty.wait(st.stage, st.phase)
                            end(tk)
                            txl.ptx.fence.proxy.async_.shared__cta()
                            s = st.stage
                            for i in range(8):
                                txl.ptx.shfl_sync.idx.b32(row, my, txl.uint32(i * 4) + txl.Cast("uint32", sub), txl.uint32(31), FULLM)
                                trow = gw * 32 + i * 4 + sub
                                copy(a_tile[s].ptr_to(trow, chunk * 8), kt * BK + chunk * 8, True)
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
                        with txl.serial(nkt) as kt:
                            load_index(walk.lo + kt * BK + gw * 16 + lane % 16, walk.hi)
                            tk = range_v("g-wait-empty")
                            ab_pipe.empty.wait(st.stage, st.phase)
                            end(tk)
                            txl.ptx.fence.proxy.async_.shared__cta()
                            s = st.stage
                            for i in range(4 * nseg):
                                # 4 (row, segment) pairs per instruction, 8 lanes each
                                pair = i * 4 + sub
                                rr = pair // nseg
                                seg = pair % nseg
                                txl.ptx.shfl_sync.idx.b32(row, my, txl.Cast("uint32", rr), txl.uint32(31), FULLM)
                                col = seg * 64 + chunk * 8
                                copy(tile[s].ptr_to(gw * 16 + rr, col), col0 + col, col0 + col < H)
                            txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](gfull.ptr_to([s]))
                            st.advance()
                    walk.advance()

        # ------------------------------------------------------------ gather relay (CTA1)
        if GATHER:
            with txl.If((warp == 1) & (cbx == 1)), txl.Then():
                walk = TileWalk()
                st = txl.PipelineState(NST, 0)
                with txl.While(walk.t < total):
                    walk.locate()
                    nkt = txl.local_scalar("int32")
                    if k_grouped:
                        txl.assign(nkt, (walk.hi - walk.lo + (BK - 1)) // BK)
                    else:
                        txl.assign(nkt, KT)
                    with txl.If(lane == 0), txl.Then():
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
        with txl.If((warp == 1) & (cbx == 0)), txl.Then():
            walk = TileWalk()
            st = txl.PipelineState(NST, 0)
            acc = txl.PipelineState(2, 1)
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tmem_base, tmem_addr.ptr_to([0]))
            idesc = txl.alloc_local((1,), "uint32")
            txl.cuda.tcgen05.encode_instr_descriptor(
                txl.address_of(idesc[0]), d_dtype="float32", a_dtype="bfloat16",
                b_dtype="bfloat16", M=BMC, N=BN, K=16, trans_a=a_idesc_trans,
                trans_b=b_idesc_trans, n_cta_groups=2,
            )
            a_desc, a_off = a_tile[0].encode(major=a_major, mma_k=16)
            b_desc, b_off = b_tile[0].encode(major=b_major, mma_k=16)
            leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.While(walk.t < total):
                walk.locate()
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
                    with txl.If(leader == txl.uint32(1)), txl.Then():
                        for kp in range(BK // 16):
                            ad = a_desc.value + txl.Cast("uint64", s * (A_BYTES // 16) + a_off(kp))
                            bd = b_desc.value + txl.Cast("uint64", s * (B_BYTES // 16) + b_off(kp))
                            accum = txl.cast(kt > 0, "bool") if kp == 0 else txl.cast(True, "bool")
                            txl.ptx[MMA](
                                d_addr, ad, bd, idesc[0], *([txl.uint32(0)] * 8), accum,
                            )
                        ab_pipe.empty.arrive(s, cta_group=2, cta_mask=3)
                    txl.cuda.warp_sync()
                    st.advance()
                with txl.If(leader == txl.uint32(1)), txl.Then():
                    acc_pipe.full.arrive(acc.stage, cta_group=2, cta_mask=3)
                txl.cuda.warp_sync()
                acc.advance()
                walk.advance()

        # ------------------------------------------------------------ epilogue
        with txl.If((warp >= EPI_WARP0) & (warp < EPI_WARP0 + NEPI)), txl.Then():
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
            ld2 = txl.alloc_local((32,), "uint32") if kind == "down_dgrad" else None
            dst = txl.local_scalar("int32")

            def load_preact(buf, row, n0):
                for q in range(8):
                    txl.ptx.ld.global_.v4.b32(
                        buf[4 * q], buf[4 * q + 1], buf[4 * q + 2], buf[4 * q + 3],
                        p["preact"].ptr_to([row * inter + n0 + 4 * q]),
                    )
            alpha = p.get("alpha")
            limit = p.get("limit")
            with txl.While(walk.t < total):
                walk.locate()
                tk = range_v("e-wait-acc")
                acc_pipe.full.wait(acc.stage, acc.phase)
                end(tk)
                tk2 = range_v("e-body")
                txl.ptx.tcgen05.fence__after_thread_sync()
                if not k_grouped:
                    r = walk.lo + walk.mt * BMC + cbx * BM + m
                    valid = r < walk.hi
                    if kind == "down_fwd":
                        txl.assign(dst, 0)
                        with txl.If(valid), txl.Then():
                            txl.ptx.ld.global_.s32(dst, p["slot_of"].ptr_to([r]))
                    if kind == "down_dgrad":
                        with txl.If(valid & (walk.nt * BN + half * (CH * 32) < N)), txl.Then():
                            load_preact(ld, r, walk.nt * BN + half * (CH * 32))
                else:
                    j = walk.mt * BMC + cbx * BM + m
                    valid = j < MO
                    srow = (lane ^ p["flip"]) if kind == "gu_wgrad" else lane
                    j0 = walk.mt * BMC + cbx * BM + ew * 32
                for c in range(CH):
                    col = half * (CH * 32) + c * 32
                    n0 = walk.nt * BN + col
                    with txl.If(n0 < N), txl.Then():
                        taddr = tmem_base + txl.Cast("uint32", (ew * 32) << 16) + txl.Cast(
                            "uint32", acc.stage * BN + col
                        )
                        txl.ptx[TMEM_LD32](*[v[i] for i in range(32)], taddr)
                        txl.ptx.tcgen05.wait__ld.sync.aligned()
                        if k_grouped:
                            # fp32 main grads: stage the 32x32 block, TMA-reduce-add it (OOB
                            # rows / columns are clipped by the tensor map)
                            slot = ew * 2 + (c % 2)
                            with txl.If(lane == 0), txl.Then():
                                txl.ptx.cp.async_.bulk.wait_group.read(1)
                            txl.cuda.warp_sync()
                            for q in range(8):
                                txl.ptx.st.shared.v4.f32(
                                    cbuf.ptr_to([slot * 1024 + srow * 32 + ((q ^ (srow % 8)) * 4)]),
                                    v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3],
                                )
                            txl.ptx.fence.proxy.async_.shared__cta()
                            txl.cuda.warp_sync()
                            with txl.If(lane == 0), txl.Then():
                                if kind == "gu_wgrad":
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
                            cur = ld
                            if kind == "down_dgrad":
                                cur, nxt = (ld, ld2) if c % 2 == 0 else (ld2, ld)
                                if c + 1 < CH:
                                    with txl.If(valid & (n0 + 32 < N)), txl.Then():
                                        load_preact(nxt, r, n0 + 32)
                            with txl.If(valid), txl.Then():
                                epilogue_chunk(kind, p, walk, r, n0, v, w16, w8,
                                                cur, dst, has_bias, clamped, alpha, limit)
                txl.ptx.tcgen05.fence__before_thread_sync()
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    acc_pipe.empty.arrive(acc.stage, remote=0)
                end(tk2)
                acc.advance()
                walk.advance()
            if k_grouped:
                with txl.If(lane == 0), txl.Then():
                    txl.ptx.cp.async_.bulk.wait_group(0)
                txl.cuda.warp_sync()

        txl.ptx.tcgen05.fence__before_thread_sync()
        txl.cuda.cluster_sync()
        with txl.If(warp == 1), txl.Then():
            txl.ptx.tcgen05.fence__after_thread_sync()
            dealloc = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](dealloc, txl.uint32(512))

    def epilogue_chunk(kind, p, walk, row, n0, v, w16, w8, ld, dst, has_bias, clamped, alpha,
                        limit):
        if kind == "gu_fwd":
            # pairs (lo, hi) in weight address order -> (gate, up)
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
            for q in range(4):
                txl.ptx.st.global_.v4.b32(
                    p["preact"].ptr_to([row * inter + n0 // 2 + 4 * q]),
                    w16[4 * q], w16[4 * q + 1], w16[4 * q + 2], w16[4 * q + 3],
                )
            for q in range(8):
                h0 = act2(bf16_lo(w16[2 * q]), bf16_hi(w16[2 * q]), clamped, alpha, limit)
                hv0 = txl.local_scalar("float32")
                txl.assign(hv0, h0)
                h1 = act2(bf16_lo(w16[2 * q + 1]), bf16_hi(w16[2 * q + 1]), clamped, alpha, limit)
                hv1 = txl.local_scalar("float32")
                txl.assign(hv1, h1)
                pack(w8[q], hv0, hv1)
            for q in range(2):
                txl.ptx.st.global_.v4.b32(
                    p["h"].ptr_to([row * (inter // 2) + n0 // 4 + 4 * q]),
                    w8[4 * q], w8[4 * q + 1], w8[4 * q + 2], w8[4 * q + 3],
                )
        elif kind in ("down_fwd", "gu_dgrad"):
            if has_bias:
                for q in range(4):
                    txl.ptx.ld.global_.v4.b32(
                        ld[4 * q], ld[4 * q + 1], ld[4 * q + 2], ld[4 * q + 3],
                        p["bias"].ptr_to([(walk.g * N + n0) // 2 + 4 * q]),
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
            out = p["out"] if kind == "down_fwd" else p["dx"]
            orow = dst if kind == "down_fwd" else row
            for q in range(4):
                txl.ptx.st.global_.v4.b32(
                    out.ptr_to([orow * (N // 2) + n0 // 2 + 4 * q]),
                    w16[4 * q], w16[4 * q + 1], w16[4 * q + 2], w16[4 * q + 3],
                )
        elif kind == "down_dgrad":
            # ld holds preact[row, 2 n0 .. 2 n0 + 63]: 32 (gate, up) words (prefetched)
            sw = p["swapped"] != 0
            for half in range(2):
                for q in range(16):
                    e = half * 16 + q
                    g = bf16_lo(ld[e])
                    u = bf16_hi(ld[e])
                    gv = txl.local_scalar("float32")
                    uv = txl.local_scalar("float32")
                    txl.assign(gv, g)
                    txl.assign(uv, u)
                    dg, du = dact(gv, uv, v[e], clamped, alpha, limit)
                    dgv = txl.local_scalar("float32")
                    duv = txl.local_scalar("float32")
                    txl.assign(dgv, dg)
                    txl.assign(duv, du)
                    pack(w16[q], txl.Select(sw, duv, dgv), txl.Select(sw, dgv, duv))
                for q in range(4):
                    txl.ptx.st.global_.v4.b32(
                        p["dpre"].ptr_to([row * inter + n0 + half * 16 + 4 * q]),
                        w16[4 * q], w16[4 * q + 1], w16[4 * q + 2], w16[4 * q + 3],
                    )
            for q in range(16):
                h0 = txl.local_scalar("float32")
                h1 = txl.local_scalar("float32")
                txl.assign(h0, act2(bf16_lo(ld[2 * q]), bf16_hi(ld[2 * q]), clamped, alpha, limit))
                txl.assign(h1, act2(bf16_lo(ld[2 * q + 1]), bf16_hi(ld[2 * q + 1]), clamped, alpha,
                                    limit))
                pack(w16[q], h0, h1)
            for q in range(4):
                txl.ptx.st.global_.v4.b32(
                    p["h"].ptr_to([row * (inter // 2) + n0 // 2 + 4 * q]),
                    w16[4 * q], w16[4 * q + 1], w16[4 * q + 2], w16[4 * q + 3],
                )

    def entry(*args, host):
        body(dict(zip(names, args, strict=True)), host)

    entry.__name__ = f"bf16_expert_{kind}"
    entry.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=a) for n, a in params]
        + [inspect.Parameter("host", inspect.Parameter.KEYWORD_ONLY)]
    )
    kern = txl.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid="num_ctas",
                      host_prelude=host_prelude)(entry)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        if IKET:
            from tvm.tirx.cuda.iket import IketProfiler
            return IketProfiler().compile(kern.mod, target=target, tir_pipeline="tirx")
        return tvm.compile(kern.mod, target=target, tir_pipeline="tirx")


@functools.cache
def build_gather(H):
    """
    out[r, :] = src[idx[r], :] for r < M: one warp per row, 16 bytes per lane.
    """
    CH = H // 8  # 16-byte chunks per row
    WPC = 8

    def gather_rows(src: txl.gptr[txl.u32], idx: txl.gptr[txl.i32], out: txl.gptr[txl.u32],
                    M: txl.i32, num_ctas: txl.i32):
        warp = txl.warp_id()
        lane = txl.lane_id()
        r = txl.local_scalar("int32", init=txl.cta_id() * WPC + warp)
        row = txl.local_scalar("int32")
        buf = txl.alloc_local((4 * (-(-CH // 32)),), "uint32")
        with txl.While(r < M):
            txl.ptx.ld.global_.s32(row, idx.ptr_to([r]))
            for i in range(-(-CH // 32)):
                c = i * 32 + lane
                with txl.If(c < CH), txl.Then():
                    txl.ptx.ld.global_.nc.v4.b32(
                        buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3],
                        src.ptr_to([row * (H // 2) + c * 4]),
                    )
            for i in range(-(-CH // 32)):
                c = i * 32 + lane
                with txl.If(c < CH), txl.Then():
                    txl.ptx.st.global_.v4.b32(
                        out.ptr_to([r * (H // 2) + c * 4]),
                        buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3],
                    )
            txl.assign(r, r + num_ctas * WPC)

    kern = txl.kernel(warps=WPC, arch="sm_100a", grid="num_ctas")(gather_rows)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        return tvm.compile(kern.mod, target=target, tir_pipeline="tirx")


def gather(src, idx):
    """
    Contiguous src[idx] (M, H) through the TIRx-lite row-gather kernel.
    """
    M, H = idx.shape[0], src.shape[1]
    out = src.new_empty(M, H)
    ctas = max(1, min(4 * num_sms(), -(-M // 8)))
    with tvm_ffi.use_torch_stream():
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


def pair2(a, b):
    """
    (lo, hi, diff_elems, swapped): b below a -> swapped.
    """
    swapped = b.data_ptr() < a.data_ptr()
    lo, hi = (b, a) if swapped else (a, b)
    diff = (hi.data_ptr() - lo.data_ptr()) // hi.element_size()
    return lo, hi, diff, swapped


def grid2(M, G, k_grouped, per_group_tiles, NT):
    """
    CTA count: two per cluster, at most one cluster per SM pair or per tile.
    """
    if k_grouped:
        ub = G * per_group_tiles
    else:
        ub = (-(-M // BMC) + G) * NT
    return 2 * max(1, min(num_sms() // 2, ub))


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
    lo, hi2, diff, swapped = pair2(w_gate, w_up)
    clamped = bool(act)
    fn = build("gu_fwd", H, inter, bias is not None, clamped)
    NT = -(-2 * inter // BN)
    args = [cu, G, M, grid2(M, G, False, 0, NT), x.reshape(-1), x.shape[0], a_idx, flat(lo),
            diff, lo.stride(0), lo.stride(1), u322(preact), u322(h), int(swapped)]
    if bias is not None:
        args.append(u322(bias.contiguous()))
    if clamped:
        args += [float(act[0]), float(act[1])]
    with tvm_ffi.use_torch_stream():
        fn(*args)
    return preact, h


def bf16_expert_down_fwd_sm100(h, w_down, cu, slot_of, bias=None):
    M, G, H, inter = h.shape[0], w_down.shape[0], w_down.shape[1], w_down.shape[2]
    out = h.new_empty(M, H)
    if M == 0:
        return out
    check_shapes(H, inter)
    fn = build("down_fwd", H, inter, bias is not None, False)
    NT = -(-H // BN)
    args = [cu, G, M, grid2(M, G, False, 0, NT), h.reshape(-1), w_down.reshape(-1), slot_of,
            u322(out)]
    if bias is not None:
        args.append(u322(bias.contiguous()))
    with tvm_ffi.use_torch_stream():
        fn(*args)
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
    NT = -(-inter // BN)
    args = [cu, G, M, grid2(M, G, False, 0, NT), dy.reshape(-1), dy.shape[0], slot_of,
            w_down.reshape(-1), u322(preact), u322(dpre), u322(h), int(swapped)]
    if clamped:
        args += [float(act[0]), float(act[1])]
    with tvm_ffi.use_torch_stream():
        fn(*args)
    return dpre, h, swapped


def bf16_expert_gate_up_dgrad_sm100(dpre, w_gate, w_up, cu, dpre_swapped):
    M, inter = dpre.shape[0], dpre.shape[1] // 2
    G, H = w_gate.shape[0], w_gate.shape[2]
    dx = dpre.new_empty(M, H)
    if M == 0:
        return dx
    check_shapes(H, inter)
    lo, hi2, diff, swapped = pair2(w_gate, w_up)
    if bool(swapped) != bool(dpre_swapped):
        raise NotImplementedError("dpre pair order must follow the weights' address order")
    fn = build("gu_dgrad", H, inter, False, False)
    NT = -(-H // BN)
    args = [cu, G, M, grid2(M, G, False, 0, NT), dpre.reshape(-1), flat(lo), diff,
            lo.stride(0), lo.stride(1), u322(dx)]
    with tvm_ffi.use_torch_stream():
        fn(*args)
    return dx


def bf16_expert_gate_up_wgrad_sm100(x, dpre, dpre_swapped, mg_gate, mg_up, cu, a_idx):
    M = a_idx.shape[0]
    if M == 0:
        return None
    G, inter, H = mg_gate.shape
    check_shapes(H, inter)
    fn = build("gu_wgrad", H, inter, False, False)
    MT, NT = -(-2 * inter // BMC), -(-H // BN)
    lo, hi2, diff, mg_swapped = pair2(mg_gate, mg_up)
    # dpre lane l is the gate iff l == dpre_swapped; main-grad address lane a is the gate iff
    # a == mg_swapped, so a = l ^ flip
    flip = int(bool(dpre_swapped)) ^ int(mg_swapped)
    xg = gather(x, a_idx)
    args = [cu, G, M, grid2(M, G, True, MT * NT, NT), dpre.reshape(-1), xg.reshape(-1), M,
            a_idx, flat(lo), diff, lo.stride(0), lo.stride(1), flip]
    with tvm_ffi.use_torch_stream():
        fn(*args)
    return None


def bf16_expert_down_wgrad_sm100(dy, h, mg_down, cu, slot_of):
    M = slot_of.shape[0]
    if M == 0:
        return None
    G, H, inter = mg_down.shape
    assert mg_down.is_contiguous()
    check_shapes(H, inter)
    fn = build("down_wgrad", H, inter, False, False)
    MT, NT = -(-H // BMC), -(-inter // BN)
    dys = gather(dy, slot_of)
    args = [cu, G, M, grid2(M, G, True, MT * NT, NT), dys.reshape(-1), M, slot_of,
            h.reshape(-1), mg_down.reshape(-1)]
    with tvm_ffi.use_torch_stream():
        fn(*args)
    return None
