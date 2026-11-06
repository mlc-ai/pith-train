"""
cross_entropy_sm100: the fused mean cross-entropy forward and backward on SM100 (the logits
overwritten in place with the gradient of the mean), TIRx-lite kernels.

Design (cluster-resident rows):
- ce_count: one CTA counts the non-ignored targets, writes inv_n = 1 / max(n, 1) and zeroes the
  row-completion counter and the loss.
- ce_cluster: a cluster of CLUSTER CTAs owns one row; CTA r bulk-copies (TMA) its slice of the row
  into shared memory once, computes a per-thread (max, sum exp) pair, reduces it over the CTA and
  then over the cluster through DSMEM, and writes (softmax - onehot) * inv_n straight from shared
  memory. Every logit is read from HBM once and written once. The CTA that records the last row
  loss sums all row losses in a fixed order (deterministic) and writes the mean.
- ce_stream: generic fallback (row bytes not a multiple of 16, a base not 16-byte aligned, or no
  cluster size that fits): one CTA per row, two streaming passes, then ce_final sums the per-row
  losses in a fixed order and scales by inv_n.
bf16, fp16 and fp32 logits with int32 or int64 targets run the kernels; other integer targets are
widened to int64 first, and other logit dtypes take a PyTorch path with the same contract.

For 16-bit logits the cluster path keeps the logits in shared memory and recomputes exp(x - max)
in fp32 in the gradient pass, since an f16 copy of the exponentials would flush terms more than
~16.6 below the chunk max to zero.
"""

import functools

import torch
import torch.nn.functional as F
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["cross_entropy_fwd_bwd_sm100"]


def max_drop_nan(a, b):
    """
    CUDA fmaxf for float32 operands, which drops a NaN operand (txl.max emits a NaN-propagating
    select); txl.max otherwise.
    """
    return txl.call_pure_extern("fmaxf", a, b, ty=txl.float32) if is_any_f32(a, b) else txl.max(a, b)


def min_drop_nan(a, b):
    """
    CUDA fminf for float32 operands, txl.min otherwise.
    """
    return txl.call_pure_extern("fminf", a, b, ty=txl.float32) if is_any_f32(a, b) else txl.min(a, b)


def is_any_f32(*operands):
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


LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
FULL_MASK = 0xFFFFFFFF
NEG_BIG = -3.0e38

TARGET_DTYPE_NAME = {torch.int64: "int64", torch.int32: "int32"}
LOGIT_DTYPE_NAME = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
ELEMENT_BYTES = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}

CLUSTER_WARPS = 8
MAX_CLUSTER = 16
SMEM_PER_SM = 233472
CTA_OVERHEAD = 1024 + 512  # driver reserve + this kernel's barriers/slots


def count_ctas_per_sm(cap_bytes):
    """
    How many cluster CTAs holding cap_bytes of a row each fit on one SM (at most 8).
    """
    return min(SMEM_PER_SM // (cap_bytes + CTA_OVERHEAD), 8)


def load_target(target_idx, target, row, tstride, target_dtype):
    """
    target_idx = target[row * tstride] as int64.
    """
    if target_dtype == "int64":
        txl.ptx.ld.global_.s64(target_idx, target.ptr_to([txl.Cast("int64", row) * tstride]))
    else:
        target_i32 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(target_i32, target.ptr_to([txl.Cast("int64", row) * tstride]))
        txl.assign(target_idx, txl.Cast("int64", target_i32))


def exp2(x):
    result = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(result, x)
    return result


def lg2(x):
    result = txl.local_scalar("float32")
    txl.ptx.lg2.approx.ftz.f32(result, x)
    return result


def shfl_bfly_f32(value, delta):
    peer = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(value, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
    return txl.reinterpret(peer, ty="float32")


def shfl_idx_f32(value, src_lane):
    peer = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(peer, txl.reinterpret(value, ty="uint32"), txl.uint32(src_lane), txl.uint32(31), txl.uint32(FULL_MASK))
    return txl.reinterpret(peer, ty="float32")


def reduce_warp_sum(acc_sum):
    """
    Butterfly sum of acc_sum over the warp, in place; every lane gets the total.
    """
    for delta in (16, 8, 4, 2, 1):
        txl.assign(acc_sum, acc_sum + shfl_bfly_f32(acc_sum, delta))


def merge_max_sum(acc_max, acc_sum, peer_max, peer_sum):
    """
    (acc_max, acc_sum) <- (acc_max, acc_sum) (+) (peer_max, peer_sum) for running (max, sum
    exp(x - max)).
    """
    new_max = txl.local_scalar("float32", init=max_drop_nan(acc_max, peer_max))
    txl.assign(acc_sum, acc_sum * exp2((acc_max - new_max) * LOG2E) + peer_sum * exp2((peer_max - new_max) * LOG2E))
    txl.assign(acc_max, new_max)


def merge_warp_max_sum(acc_max, acc_sum):
    """
    Butterfly (max, sum) over the warp; every lane then takes lane 0's pair (deterministic).
    """
    for delta in (16, 8, 4, 2, 1):
        peer_max, peer_sum = txl.local_scalar("float32", init=shfl_bfly_f32(acc_max, delta)), txl.local_scalar("float32", init=shfl_bfly_f32(acc_sum, delta))
        merge_max_sum(acc_max, acc_sum, peer_max, peer_sum)
    txl.assign(acc_max, shfl_idx_f32(acc_max, 0))
    txl.assign(acc_sum, shfl_idx_f32(acc_sum, 0))


def merge_block_max_sum(acc_max, acc_sum, s_partials, num_warps, lane, warp):
    """
    (max, sum) over the CTA through shared slots s_partials[0:2*num_warps]; result in every thread.
    """
    merge_warp_max_sum(acc_max, acc_sum)
    if num_warps == 1:
        return
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.f32(s_partials.ptr_to([2 * warp]), acc_max)
        txl.ptx.st.shared.f32(s_partials.ptr_to([2 * warp + 1]), acc_sum)
    txl.cuda.cta_sync()
    with txl.if_(lane < num_warps):
        with txl.then_():
            txl.ptx.ld.shared.f32(acc_max, s_partials.ptr_to([2 * lane]))
            txl.ptx.ld.shared.f32(acc_sum, s_partials.ptr_to([2 * lane + 1]))
        with txl.else_():
            txl.assign(acc_max, txl.float32(NEG_BIG))
            txl.assign(acc_sum, txl.float32(0.0))
    merge_warp_max_sum(acc_max, acc_sum)


def pack_from_f32(words, src, num_words, logit_dtype):
    """
    Round the f32 values src into num_words 32-bit words of packed logits.
    """
    if logit_dtype == "float32":
        for k in range(num_words):
            txl.assign(words[k], txl.reinterpret(src[k], ty="uint32"))
    else:
        cvt_op = "cvt.rn.bf16x2.f32" if logit_dtype == "bfloat16" else "cvt.rn.f16x2.f32"
        for k in range(num_words):
            txl.ptx[cvt_op](words[k], src[2 * k + 1], src[2 * k])


@functools.cache
def build_count_kernel(target_dtype, reduction: str = "mean"):
    LOADS_IN_FLIGHT = 4  # target loads in flight per thread

    @txl.kernel()
    def ce_count(target: txl.Tensor((txl.int64(),), target_dtype, data_alignment=1), tstride: txl.int64, N: txl.int32, ii: txl.int64, stats: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), done: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), loss: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1)):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=1024))
        tidx = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        smem = txl.smem_pool()
        s_counts = smem.alloc([32], txl.i32)
        acc_count = txl.local_scalar("int32", init=txl.int32(0))
        target_vals = [txl.local_scalar("int64", init=txl.Cast("int64", ii)) for _ in range(LOADS_IN_FLIGHT)]
        with txl.serial((N + 1024 * LOADS_IN_FLIGHT - 1) // (1024 * LOADS_IN_FLIGHT)) as step:
            for i in range(LOADS_IN_FLIGHT):
                row = (step * LOADS_IN_FLIGHT + i) * 1024 + tidx
                with txl.if_(row < N), txl.then_():
                    load_target(target_vals[i], target, row, tstride, target_dtype)
            for i in range(LOADS_IN_FLIGHT):
                with txl.if_(target_vals[i] != ii), txl.then_():
                    txl.assign(acc_count, acc_count + 1)
                txl.assign(target_vals[i], txl.Cast("int64", ii))
        count_sum = txl.local_scalar("uint32")
        txl.ptx.redux_sync.add.u32(count_sum, txl.reinterpret(acc_count, ty="uint32"), txl.uint32(FULL_MASK))
        with txl.if_(lane == 0), txl.then_():
            txl.ptx.st.shared.s32(s_counts.ptr_to([warp]), txl.reinterpret(count_sum, ty="int32"))
        txl.cuda.cta_sync()
        with txl.if_(warp == 0), txl.then_():
            warp_count = txl.local_scalar("int32")
            txl.ptx.ld.shared.s32(warp_count, s_counts.ptr_to([lane]))
            txl.ptx.redux_sync.add.u32(count_sum, txl.reinterpret(warp_count, ty="uint32"), txl.uint32(FULL_MASK))
            with txl.if_(lane == 0), txl.then_():
                num_valid = txl.Cast("float32", max_drop_nan(txl.reinterpret(count_sum, ty="int32"), txl.int32(1))) if reduction == "mean" else txl.float32(1.0)
                txl.ptx.st.global_.f32(stats.ptr_to([0]), txl.float32(1.0) / num_valid)
                # The main kernel's last row to finish sums the row losses into loss (0 if N == 0).
                txl.ptx.st.global_.s32(done.ptr_to([0]), txl.int32(0))
                txl.ptx.st.global_.f32(loss.ptr_to([0]), txl.float32(0.0))

    return entry_function(ce_count.compile(TARGET))


@functools.cache
def build_final_kernel():
    @txl.kernel()
    def ce_final(row_loss: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), stats: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), loss: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=1024))
        tidx = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        smem = txl.smem_pool()
        s_partials = smem.alloc([32], txl.f32)
        # One load register serves the row losses and then inv_n.
        acc_sum, loaded = txl.local_scalar("float32", init=txl.float32(0.0)), txl.local_scalar("float32")
        with txl.serial((N + 1023) // 1024) as step:
            row = step * 1024 + tidx
            with txl.if_(row < N), txl.then_():
                txl.ptx.ld.global_.f32(loaded, row_loss.ptr_to([row]))
                txl.assign(acc_sum, acc_sum + loaded)
        reduce_warp_sum(acc_sum)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx.st.shared.f32(s_partials.ptr_to([warp]), acc_sum)
        txl.cuda.cta_sync()
        with txl.if_(warp == 0), txl.then_():
            txl.ptx.ld.shared.f32(acc_sum, s_partials.ptr_to([lane]))
            reduce_warp_sum(acc_sum)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.ld.global_.f32(loaded, stats.ptr_to([0]))
                txl.ptx.st.global_.f32(loss.ptr_to([0]), acc_sum * loaded)

    return entry_function(ce_final.compile(TARGET))


@functools.cache
def build_stream_kernel(logit_dtype, target_dtype):
    """
    Generic fallback: one CTA per row, scalar loads, two passes over global memory.
    """
    WARPS = 16
    THREADS = WARPS * 32
    BITS = {"bfloat16": 16, "float16": 16, "float32": 32}[logit_dtype]

    def load_logit(inp, idx):
        if BITS == 32:
            logit = txl.local_scalar("float32")
            txl.ptx.ld.global_.f32(logit, inp.ptr_to([idx]))
            return logit
        logit_bits = txl.local_scalar("uint16")
        txl.ptx.ld.global_.b16(logit_bits, inp.ptr_to([idx]))
        if logit_dtype == "bfloat16":
            return txl.reinterpret(txl.Cast("uint32", logit_bits) << txl.uint32(16), ty="float32")
        return txl.Cast("float32", txl.reinterpret(logit_bits, ty="float16"))

    def store_logit(inp, idx, grad):
        if BITS == 32:
            txl.ptx.st.global_.f32(inp.ptr_to([idx]), grad)
        else:
            txl.ptx.st.global_.b16(inp.ptr_to([idx]), txl.reinterpret(txl.Cast(logit_dtype, grad), ty="uint16"))

    @txl.kernel()
    def ce_stream(inp: txl.Tensor((txl.int64(),), logit_dtype, data_alignment=1), target: txl.Tensor((txl.int64(),), target_dtype, data_alignment=1), tstride: txl.int64, stats: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), row_loss: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32, V: txl.int32, ii: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=N, block=WARPS * 32))
        row = txl.cta_id()
        tidx = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        smem = txl.smem_pool()
        s_partials = smem.alloc([2 * WARPS], txl.f32)
        target_idx = txl.local_scalar("int64")
        load_target(target_idx, target, row, tstride, target_dtype)
        row_base = txl.Cast("int64", row) * txl.Cast("int64", V)
        num_iters = (V + THREADS - 1) // THREADS
        with txl.if_(target_idx == ii):
            with txl.then_():
                with txl.serial(num_iters) as step:
                    col = step * THREADS + tidx
                    with txl.if_(col < V), txl.then_():
                        store_logit(inp, row_base + txl.Cast("int64", col), txl.float32(0.0))
                with txl.if_(tidx == 0), txl.then_():
                    txl.ptx.st.global_.f32(row_loss.ptr_to([row]), txl.float32(0.0))
            with txl.else_():
                acc_max, acc_sum = txl.local_scalar("float32", init=txl.float32(NEG_BIG)), txl.local_scalar("float32", init=txl.float32(0.0))
                with txl.serial(num_iters) as step:
                    col = step * THREADS + tidx
                    with txl.if_(col < V), txl.then_():
                        logit = txl.local_scalar("float32", init=load_logit(inp, row_base + txl.Cast("int64", col)))
                        merge_max_sum(acc_max, acc_sum, logit, txl.float32(1.0))
                merge_block_max_sum(acc_max, acc_sum, s_partials, WARPS, lane, warp)
                inv_n = txl.local_scalar("float32")
                txl.ptx.ld.global_.f32(inv_n, stats.ptr_to([0]))
                scale = txl.local_scalar("float32", init=inv_n / acc_sum)
                with txl.serial(num_iters) as step:
                    col = step * THREADS + tidx
                    with txl.if_(col < V), txl.then_():
                        logit = txl.local_scalar("float32", init=load_logit(inp, row_base + txl.Cast("int64", col)))
                        grad = txl.local_scalar("float32", init=exp2((logit - acc_max) * LOG2E) * scale)
                        with txl.if_(txl.Cast("int64", col) == target_idx), txl.then_():
                            txl.assign(grad, grad - inv_n)
                            log2_sum = lg2(acc_sum)
                            txl.ptx.st.global_.f32(row_loss.ptr_to([row]), acc_max + log2_sum * LN2 - logit)
                        store_logit(inp, row_base + txl.Cast("int64", col), grad)

    return entry_function(ce_stream.compile(TARGET))


def apply_f32x2(op, a, b):
    """
    One packed f32x2 instruction (sub/mul/add/max), result as a b64 pair.
    """
    result = txl.local_scalar("uint64")
    txl.ptx[f"{op}.rn.ftz.f32x2"](result, a, b)
    return result


def unpack_pairs(words, logit_dtype):
    """
    The float2 pairs of one 4-word vector: 4 pairs for 16-bit logits, 2 for fp32.
    """
    if logit_dtype == "float32":
        return [txl.cuda.make_float2(txl.reinterpret(words[2 * k], ty="float32"), txl.reinterpret(words[2 * k + 1], ty="float32")) for k in range(2)]
    if logit_dtype == "bfloat16":
        return [txl.cuda.make_float2(txl.reinterpret(words[k] << txl.uint32(16), ty="float32"), txl.reinterpret(words[k] & txl.uint32(0xFFFF0000), ty="float32")) for k in range(4)]
    out = []
    for k in range(4):
        halves = txl.alloc_local([2], "float32")
        txl.idioms.cast_f16x2_to_f32x2(halves, 0, words[k])
        out.append(txl.cuda.make_float2(halves[0], halves[1]))
    return out


def reduce_words_max(words, num_words, logit_dtype):
    """
    Exact max over num_words words of packed logits, as f32.
    """
    if logit_dtype == "float32":
        acc_max = txl.local_scalar("float32", init=txl.reinterpret(words[0], ty="float32"))
        for k in range(1, num_words):
            txl.assign(acc_max, max_drop_nan(acc_max, txl.reinterpret(words[k], ty="float32")))
        return acc_max
    max_op = "max.bf16x2" if logit_dtype == "bfloat16" else "max.f16x2"
    acc_max = txl.local_scalar("uint32", init=words[0])
    for k in range(1, num_words):
        txl.ptx[max_op](acc_max, acc_max, words[k])
    halves = txl.alloc_local([2], "float32")
    if logit_dtype == "bfloat16":
        txl.assign(halves[0], txl.reinterpret(acc_max << txl.uint32(16), ty="float32"))
        txl.assign(halves[1], txl.reinterpret(acc_max & txl.uint32(0xFFFF0000), ty="float32"))
    else:
        txl.idioms.cast_f16x2_to_f32x2(halves, 0, acc_max)
    return max_drop_nan(halves[0], halves[1])


NEG_INF_WORDS = {"bfloat16": 0xFF80FF80, "float16": 0xFC00FC00, "float32": 0xFF800000}  # -inf lanes


def load_target_logit(inp, idx, logit_dtype):
    """
    One logit from global memory as f32.
    """
    logit = txl.local_scalar("float32")
    if logit_dtype == "float32":
        txl.ptx.ld.global_.f32(logit, inp.ptr_to([idx]))
        return logit
    logit_bits = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(logit_bits, inp.ptr_to([idx]))
    if logit_dtype == "bfloat16":
        txl.assign(logit, txl.reinterpret(txl.Cast("uint32", logit_bits) << txl.uint32(16), ty="float32"))
    else:
        txl.ptx.cvt.f32.f16(logit, logit_bits)
    return logit


class SliceCoords:
    """
    A cluster CTA's coordinates: its thread indices, cluster rank and row, and its slice of the row
    (the slice's first element, element and 16-byte vector counts, the target's offset in it).
    """

    def __init__(self, tidx, lane, warp, cluster_rank, row, slice_base, slice_count, num_vecs, target_col):
        self.tidx, self.lane, self.warp, self.cluster_rank = tidx, lane, warp, cluster_rank
        self.row, self.slice_base, self.slice_count, self.num_vecs = row, slice_base, slice_count, num_vecs
        self.target_col = target_col


def trace_cluster_kernel(logit_dtype, target_dtype, CLUSTER, CAP_BYTES, WARPS=CLUSTER_WARPS):
    """
    CLUSTER CTAs per row; each holds up to CAP_BYTES of the row in shared memory, loaded in chunks.
    """
    THREADS = WARPS * 32
    elem_bytes = 4 if logit_dtype == "float32" else 2
    cap_elems = CAP_BYTES // elem_bytes  # elements per CTA
    elems_per_vec = 16 // elem_bytes  # elements per 16-byte vector
    VECS_PER_THREAD = 2  # vectors per thread per chunk
    chunk_vecs = THREADS * VECS_PER_THREAD
    chunk_elems = chunk_vecs * elems_per_vec
    chunk_bytes = chunk_vecs * 16
    num_chunks = -(-CAP_BYTES // chunk_bytes)  # the last chunk may be partial
    assert CAP_BYTES % 16 == 0
    num_vec_iters = num_chunks * VECS_PER_THREAD
    log2e_pair = txl.cuda.make_float2(txl.float32(LOG2E), txl.float32(LOG2E))

    def exp_pair(pair, max_pair):
        """
        (exp(x - max), exp(y - max)) of a float2 pair, in fp32.
        """
        scaled = apply_f32x2("mul", apply_f32x2("sub", pair, max_pair), log2e_pair)
        exp_x, exp_y = txl.local_scalar("float32", init=exp2(txl.cuda.float2_x(scaled))), txl.local_scalar("float32", init=exp2(txl.cuda.float2_y(scaled)))
        return exp_x, exp_y

    def init_barriers(chunk_bar, peer_bar, s_is_last, tidx):
        """
        Initialize the chunk and peer barriers and the last-row flag, then arrive on the cluster
        barrier.
        """
        chunk_bar.init(1)
        peer_bar.init(1)
        with txl.if_(tidx == 0), txl.then_():
            txl.ptx.st.shared.s32(s_is_last.ptr_to([0]), txl.int32(0))
        if CLUSTER > 1:
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(peer_bar.ptr_to([0]), txl.uint32(8 * CLUSTER))
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.cuda.cta_sync()
        if CLUSTER > 1:
            # Peers push their (max, sum) into our slots; their barriers must be initialized first.
            txl.ptx.barrier.cluster.arrive.relaxed()

    def zero_ignored_row(inp, row_loss, done, s_is_last, words, coords, N):
        """
        An ignored row: zero this CTA's slice of the gradient; rank 0 records a zero row loss.
        """
        tidx, cluster_rank, row, slice_base, num_vecs = coords.tidx, coords.cluster_rank, coords.row, coords.slice_base, coords.num_vecs
        for i in range(4):
            txl.assign(words[i], txl.uint32(0))
        with txl.serial(num_vec_iters) as step:
            vec_idx = step * THREADS + tidx
            with txl.if_(vec_idx < num_vecs), txl.then_():
                txl.ptx["st.global.v4.b32"](inp.ptr_to([slice_base + txl.Cast("int64", vec_idx * elems_per_vec)]), words[0], words[1], words[2], words[3])
        with txl.if_(txl.And(tidx == 0, cluster_rank == 0)), txl.then_():
            txl.ptx.st.global_.f32(row_loss.ptr_to([row]), txl.float32(0.0))
            prev_done = txl.local_scalar("int32")
            txl.ptx.atom.acq_rel.gpu.global_.add.s32(prev_done, done.ptr_to([0]), txl.int32(1))
            txl.ptx.st.shared.s32(s_is_last.ptr_to([0]), txl.Select(prev_done == N - 1, txl.int32(1), txl.int32(0)))

    def issue_slice_copies(inp, s_row, chunk_bar, coords):
        """
        Thread 0 issues one bulk copy per chunk of the slice, each completing on its own barrier.
        """
        tidx, slice_base, slice_count = coords.tidx, coords.slice_base, coords.slice_count
        with txl.if_(tidx == 0), txl.then_():
            with txl.serial(num_chunks) as k:
                remaining = slice_count * elem_bytes - k * chunk_bytes
                with txl.if_(remaining > 0), txl.then_():
                    copy_bytes = txl.Cast("uint32", min_drop_nan(remaining, txl.int32(chunk_bytes)))
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(chunk_bar.ptr_to([k]), copy_bytes)
                    txl.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](s_row.ptr_to([k * chunk_elems]), inp.ptr_to([slice_base + txl.Cast("int64", k * chunk_elems)]), copy_bytes, chunk_bar.ptr_to([k]))

    def reduce_slice(inp, s_row, s_partials, chunk_bar, words, coords):
        """
        (max, sum exp) over the CTA's slice, chunk by chunk as the copies land, merged over the CTA;
        thread 0 of the CTA holding the target loads its logit. Returns (acc_max, acc_sum,
        target_logit, is_owner, chunk_max), chunk_max[k] the thread's running max after chunk k.
        """
        tidx, lane, warp, slice_base, slice_count, num_vecs, target_col = coords.tidx, coords.lane, coords.warp, coords.slice_base, coords.slice_count, coords.num_vecs, coords.target_col
        # Per-thread, per-chunk: chunk max -> running max chunk_max[k]; e = exp(x - chunk_max[k]) is
        # summed into acc_sum. fp32 logits get e written back in place, so their grad pass needs no
        # second exponential: grad = e * exp(chunk_max[k] - M) * inv_n / S. 16-bit logits stay in
        # shared memory as loaded and the grad pass recomputes exp(x - M) in fp32: an f16 copy of e
        # would flush terms more than ~16.6 below the chunk max to zero (and lose precision from ~9.7
        # below, in f16 denormals), where the contract is fp32 until the final round.
        acc_max, acc_sum, target_logit = txl.local_scalar("float32", init=txl.float32(NEG_BIG)), txl.local_scalar("float32", init=txl.float32(0.0)), txl.local_scalar("float32", init=txl.float32(0.0))
        is_owner = txl.And(target_col >= 0, target_col < slice_count)
        with txl.if_(txl.And(tidx == 0, is_owner)), txl.then_():
            txl.assign(target_logit, load_target_logit(inp, slice_base + txl.Cast("int64", target_col), logit_dtype))
        exp_words = txl.alloc_local([4], "uint32")
        chunk_max = []
        for k in range(num_chunks):
            max_k = txl.local_scalar("float32", init=txl.float32(NEG_BIG))
            chunk_max.append(max_k)
            with txl.if_(k * chunk_elems < slice_count), txl.then_():
                chunk_bar.wait(k, 0)
                for j in range(VECS_PER_THREAD):
                    vec_idx = k * chunk_vecs + j * THREADS + tidx
                    with txl.if_(vec_idx < num_vecs):
                        with txl.then_():
                            txl.ptx["ld.shared.v4.b32"](words[4 * j], words[4 * j + 1], words[4 * j + 2], words[4 * j + 3], s_row.ptr_to([vec_idx * elems_per_vec]))
                        with txl.else_():
                            for i in range(4):
                                txl.assign(words[4 * j + i], txl.uint32(NEG_INF_WORDS[logit_dtype]))
                words_max = reduce_words_max(words, 4 * VECS_PER_THREAD, logit_dtype)
                new_max = txl.local_scalar("float32", init=max_drop_nan(acc_max, words_max))
                txl.assign(acc_sum, acc_sum * exp2((acc_max - new_max) * LOG2E))
                txl.assign(acc_max, new_max)
                txl.assign(max_k, new_max)
                max_pair = txl.cuda.make_float2(new_max, new_max)
                acc_pair = txl.local_scalar("uint64", init=txl.reinterpret(txl.uint64(0), ty="uint64"))
                for j in range(VECS_PER_THREAD):
                    pairs = unpack_pairs([words[4 * j + i] for i in range(4)], logit_dtype)
                    for i, pair in enumerate(pairs):
                        exp_x, exp_y = exp_pair(pair, max_pair)
                        txl.ptx["add.rn.ftz.f32x2"](acc_pair, acc_pair, txl.cuda.make_float2(exp_x, exp_y))
                        if logit_dtype == "float32":
                            txl.assign(exp_words[2 * i], txl.reinterpret(exp_x, ty="uint32"))
                            txl.assign(exp_words[2 * i + 1], txl.reinterpret(exp_y, ty="uint32"))
                    if logit_dtype == "float32":
                        vec_idx = k * chunk_vecs + j * THREADS + tidx
                        with txl.if_(vec_idx < num_vecs), txl.then_():
                            txl.ptx["st.shared.v4.b32"](s_row.ptr_to([vec_idx * elems_per_vec]), exp_words[0], exp_words[1], exp_words[2], exp_words[3])
                txl.assign(acc_sum, acc_sum + txl.cuda.float2_x(acc_pair) + txl.cuda.float2_y(acc_pair))
        merge_block_max_sum(acc_max, acc_sum, s_partials, WARPS, lane, warp)
        return acc_max, acc_sum, target_logit, is_owner, chunk_max

    def merge_cluster_max_sum(acc_max, acc_sum, s_peer_pairs, peer_bar, coords):
        """
        Exchange the CTA's (max, sum) with every CTA of the cluster through DSMEM and merge them.
        """
        lane, warp, cluster_rank = coords.lane, coords.warp, coords.cluster_rank
        # Lane r of warp 0 pushes this CTA's pair into slot [cluster_rank] of peer r.
        with txl.if_(txl.And(warp == 0, lane < CLUSTER)), txl.then_():
            slot_addr, bar_addr = txl.local_scalar("uint32"), txl.local_scalar("uint32")
            txl.ptx.mapa.shared__cluster.u32(slot_addr, txl.cuda.cvta_generic_to_shared(s_peer_pairs.ptr_to([2 * cluster_rank])), txl.Cast("uint32", lane))
            txl.ptx.mapa.shared__cluster.u32(bar_addr, txl.cuda.cvta_generic_to_shared(peer_bar.ptr_to([0])), txl.Cast("uint32", lane))
            txl.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v2.f32(slot_addr, acc_max, acc_sum, bar_addr)
        txl.cuda.mbarrier_wait_acquire_cluster(peer_bar.ptr_to([0]), 0)
        # Every thread merges the CLUSTER pairs in rank order: identical in all CTAs of the row.
        txl.ptx.ld.shared.f32(acc_max, s_peer_pairs.ptr_to([0]))
        txl.ptx.ld.shared.f32(acc_sum, s_peer_pairs.ptr_to([1]))
        for peer_rank in range(1, CLUSTER):
            peer_max, peer_sum = txl.local_scalar("float32"), txl.local_scalar("float32")
            txl.ptx.ld.shared.f32(peer_max, s_peer_pairs.ptr_to([2 * peer_rank]))
            txl.ptx.ld.shared.f32(peer_sum, s_peer_pairs.ptr_to([2 * peer_rank + 1]))
            merge_max_sum(acc_max, acc_sum, peer_max, peer_sum)

    def store_slice_grad(inp, s_row, words, acc_max, scale, inv_n, chunk_max, is_owner, coords):
        """
        Write (softmax - onehot) * inv_n over the slice from shared memory, one rounding per value.
        """
        tidx, slice_base, slice_count, num_vecs, target_col = coords.tidx, coords.slice_base, coords.slice_count, coords.num_vecs, coords.target_col
        target_vec = txl.local_scalar("int32", init=txl.Select(is_owner, target_col // elems_per_vec, txl.int32(-1)))
        grad = txl.alloc_local([elems_per_vec], "float32")
        max_pair, scale_pair = txl.cuda.make_float2(acc_max, acc_max), txl.cuda.make_float2(scale, scale)
        for k in range(num_chunks):
            with txl.if_(k * chunk_elems < slice_count), txl.then_():
                if logit_dtype == "float32":
                    chunk_scale = txl.local_scalar("float32", init=exp2((chunk_max[k] - acc_max) * LOG2E) * scale)
                    chunk_scale_pair = txl.cuda.make_float2(chunk_scale, chunk_scale)
                for j in range(VECS_PER_THREAD):
                    vec_idx = k * chunk_vecs + j * THREADS + tidx
                    with txl.if_(vec_idx < num_vecs), txl.then_():
                        txl.ptx["ld.shared.v4.b32"](words[0], words[1], words[2], words[3], s_row.ptr_to([vec_idx * elems_per_vec]))
                        for i, pair in enumerate(unpack_pairs(words, logit_dtype)):
                            if logit_dtype == "float32":
                                grad_pair = apply_f32x2("mul", pair, chunk_scale_pair)
                                txl.assign(grad[2 * i], txl.cuda.float2_x(grad_pair))
                                txl.assign(grad[2 * i + 1], txl.cuda.float2_y(grad_pair))
                            else:
                                # exp(x - M) * inv_n / S in fp32, as ce_stream computes it
                                exp_x, exp_y = exp_pair(pair, max_pair)
                                grad_pair = apply_f32x2("mul", txl.cuda.make_float2(exp_x, exp_y), scale_pair)
                                txl.assign(grad[2 * i], txl.cuda.float2_x(grad_pair))
                                txl.assign(grad[2 * i + 1], txl.cuda.float2_y(grad_pair))
                        with txl.if_(vec_idx == target_vec), txl.then_():
                            target_elem = target_col - vec_idx * elems_per_vec
                            for i in range(elems_per_vec):
                                with txl.if_(target_elem == i), txl.then_():
                                    txl.assign(grad[i], grad[i] - inv_n)
                        pack_from_f32(words, grad, 4, logit_dtype)
                        txl.ptx["st.global.v4.b32"](inp.ptr_to([slice_base + txl.Cast("int64", vec_idx * elems_per_vec)]), words[0], words[1], words[2], words[3])

    def sum_row_losses(stats, row_loss, loss, s_partials, s_is_last, coords, N):
        """
        The CTA that recorded the last row loss sums them all in a fixed order (deterministic) and
        writes the mean.
        """
        tidx, lane, warp = coords.tidx, coords.lane, coords.warp
        txl.cuda.cta_sync()
        is_last = txl.local_scalar("int32")
        txl.ptx.ld.shared.s32(is_last, s_is_last.ptr_to([0]))
        with txl.if_(is_last != 0), txl.then_():
            txl.ptx.fence.acq_rel.gpu()
            acc_sum, row_loss_val = txl.local_scalar("float32", init=txl.float32(0.0)), txl.local_scalar("float32")
            with txl.serial((N + THREADS - 1) // THREADS) as step:
                row = step * THREADS + tidx
                with txl.if_(row < N), txl.then_():
                    txl.ptx.ld.relaxed.gpu.global_.f32(row_loss_val, row_loss.ptr_to([row]))
                    txl.assign(acc_sum, acc_sum + row_loss_val)
            reduce_warp_sum(acc_sum)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.f32(s_partials.ptr_to([warp]), acc_sum)
            txl.cuda.cta_sync()
            with txl.if_(warp == 0), txl.then_():
                with txl.if_(lane < WARPS):
                    with txl.then_():
                        txl.ptx.ld.shared.f32(acc_sum, s_partials.ptr_to([lane]))
                    with txl.else_():
                        txl.assign(acc_sum, txl.float32(0.0))
                reduce_warp_sum(acc_sum)
                with txl.if_(lane == 0), txl.then_():
                    inv_n = txl.local_scalar("float32")
                    txl.ptx.ld.global_.f32(inv_n, stats.ptr_to([0]))
                    txl.ptx.st.global_.f32(loss.ptr_to([0]), acc_sum * inv_n)

    @txl.kernel()
    def ce_cluster(inp: txl.Tensor((txl.int64(),), logit_dtype, data_alignment=1), target: txl.Tensor((txl.int64(),), target_dtype, data_alignment=1), tstride: txl.int64, stats: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), row_loss: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), N: txl.int32, V: txl.int32, ii: txl.int64, done: txl.Tensor((txl.int64(),), txl.i32, data_alignment=1), loss: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1)):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=N * CLUSTER, block=WARPS * 32, cluster=(CLUSTER,) if CLUSTER > 1 else None), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=count_ctas_per_sm(CAP_BYTES)))
        cta_idx = txl.cta_id()
        cluster_rank = txl.cuda.mov_sreg(32, "cluster_ctarank") if CLUSTER > 1 else txl.int32(0)  # %cluster_ctarank: %cluster_ctaid.x costs a division by the cluster size
        # Rows run last-to-first: the restore copy (or producer) wrote the tail last, so it is
        # the part most likely still resident in L2.
        row = N - 1 - cta_idx // CLUSTER
        tidx = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        smem = txl.smem_pool()
        s_row, s_partials, s_peer_pairs, s_is_last = smem.alloc([cap_elems], logit_dtype, align=128), smem.alloc([2 * WARPS], txl.f32, align=16), smem.alloc([2 * MAX_CLUSTER], txl.f32, align=16), smem.alloc([1], txl.i32, align=4)
        chunk_bar, peer_bar = txl.MBarrier(smem, num_chunks), txl.MBarrier(smem, 1)
        init_barriers(chunk_bar, peer_bar, s_is_last, tidx)

        slice_len = txl.local_scalar("int32", init=((V + CLUSTER * 8 - 1) // (CLUSTER * 8)) * 8)
        slice_start = txl.local_scalar("int32", init=cluster_rank * slice_len)
        slice_count = txl.local_scalar("int32", init=max_drop_nan(min_drop_nan(V - slice_start, slice_len), txl.int32(0)))
        num_vecs = slice_count // elems_per_vec
        slice_base = txl.local_scalar("int64", init=txl.Cast("int64", row) * txl.Cast("int64", V) + txl.Cast("int64", slice_start))
        target_idx = txl.local_scalar("int64")
        load_target(target_idx, target, row, tstride, target_dtype)
        target_col = txl.local_scalar("int32", init=txl.Cast("int32", target_idx - txl.Cast("int64", slice_start)))
        coords = SliceCoords(tidx, lane, warp, cluster_rank, row, slice_base, slice_count, num_vecs, target_col)

        words = txl.alloc_local([4 * VECS_PER_THREAD], "uint32")

        with txl.if_(target_idx == ii):
            with txl.then_():
                zero_ignored_row(inp, row_loss, done, s_is_last, words, coords, N)
                if CLUSTER > 1:
                    txl.ptx.barrier.cluster.wait()
            with txl.else_():
                issue_slice_copies(inp, s_row, chunk_bar, coords)
                acc_max, acc_sum, target_logit, is_owner, chunk_max = reduce_slice(inp, s_row, s_partials, chunk_bar, words, coords)
                if CLUSTER > 1:
                    txl.ptx.barrier.cluster.wait()
                    merge_cluster_max_sum(acc_max, acc_sum, s_peer_pairs, peer_bar, coords)
                inv_n = txl.local_scalar("float32")
                txl.ptx.ld.global_.f32(inv_n, stats.ptr_to([0]))
                scale = txl.local_scalar("float32", init=inv_n / acc_sum)
                with txl.if_(txl.And(tidx == 0, is_owner)), txl.then_():
                    txl.ptx.st.global_.f32(row_loss.ptr_to([row]), acc_max + lg2(acc_sum) * LN2 - target_logit)
                    prev_done = txl.local_scalar("int32")
                    txl.ptx.atom.acq_rel.gpu.global_.add.s32(prev_done, done.ptr_to([0]), txl.int32(1))
                    txl.ptx.st.shared.s32(s_is_last.ptr_to([0]), txl.Select(prev_done == N - 1, txl.int32(1), txl.int32(0)))
                store_slice_grad(inp, s_row, words, acc_max, scale, inv_n, chunk_max, is_owner, coords)
        sum_row_losses(stats, row_loss, loss, s_partials, s_is_last, coords, N)

    return ce_cluster.func


@functools.cache
def build_cluster_kernel(logit_dtype, target_dtype, CLUSTER, cap_bytes, WARPS=CLUSTER_WARPS):
    func = trace_cluster_kernel(logit_dtype, target_dtype, CLUSTER, cap_bytes, WARPS)
    return entry_function(tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": func}), target=TARGET, pipeline="tirx"))


def pick_cluster(V, elem_bytes):
    """
    (CLUSTER, cap bytes): the smallest cluster whose exact slices fit 5 CTAs per SM if that needs
    <= 10 CTAs, else the smallest fitting 4 per SM, else 3; None if the row bytes are not a
    multiple of 16. Clusters of 11+ CTAs and 6-7 CTAs per SM run slower.
    """
    if (V * elem_bytes) % 16 != 0:
        return None
    for want, max_cluster in ((5, 10), (4, MAX_CLUSTER), (3, MAX_CLUSTER)):
        for CLUSTER in range(1, max_cluster + 1):
            cap_bytes = -(-V // (CLUSTER * 8)) * 8 * elem_bytes
            if count_ctas_per_sm(cap_bytes) >= want:
                return CLUSTER, cap_bytes
    return None


def cross_entropy_fwd_bwd_torch(inp: torch.Tensor, target: torch.Tensor, ignore_index: int, reduction: str = "mean") -> torch.Tensor:
    """
    The contract in PyTorch for logit dtypes the kernels do not take: the fp32 loss and the
    gradient of the requested reduction written into the row-contiguous inp.
    """
    x, keep = inp.float(), target != ignore_index
    num_valid = keep.sum().clamp(min=1).float() if reduction == "mean" else 1
    loss = F.cross_entropy(x, target.long(), ignore_index=ignore_index, reduction="sum") / num_valid
    grad = torch.softmax(x, dim=-1)
    grad.scatter_add_(1, torch.where(keep, target, 0).long().unsqueeze(1), -keep.float().unsqueeze(1))
    inp.copy_(grad.mul_(keep.unsqueeze(1) / num_valid))
    return loss


def cross_entropy_fwd_bwd_sm100(inp: torch.Tensor, target: torch.Tensor, ignore_index: int, reduction: str = "mean") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean or summed cross-entropy of the (N, V) logits inp against target, overwriting the logits
    (or the contiguous copy) with the gradient of the requested reduction.
    Returns (loss, grad), the same contract as cross_entropy_fwd_bwd_sm90.
    """
    N, V = inp.shape
    if inp.stride(-1) != 1 or inp.stride(-2) != V:
        inp = inp.contiguous()
    if N == 0:  # no rows: loss 0 and the empty gradient, as cross_entropy_fwd_bwd_sm90 returns
        return torch.zeros((), dtype=torch.float32, device=inp.device), inp
    if inp.dtype not in LOGIT_DTYPE_NAME:
        return cross_entropy_fwd_bwd_torch(inp, target, ignore_index, reduction), inp
    if target.dtype not in TARGET_DTYPE_NAME:  # other integer targets: the same values as int64
        target = target.long()
    device = inp.device
    tstride = target.stride(0) if target.dim() == 1 else 1
    # A 1-D view over target's storage (metadata only) so any stride reaches the kernel.
    target_flat = target.as_strided(((N - 1) * tstride + 1 if N > 0 else 1,), (1,))
    target_dtype, logit_dtype = TARGET_DTYPE_NAME[target.dtype], LOGIT_DTYPE_NAME[inp.dtype]
    stats, done, row_loss, loss = torch.empty(2, dtype=torch.float32, device=device), torch.empty(1, dtype=torch.int32, device=device), torch.empty(max(N, 1), dtype=torch.float32, device=device), torch.empty((), dtype=torch.float32, device=device)
    logits_flat = inp.view(-1)
    build_count_kernel(target_dtype, reduction)(target_flat, tstride, N, ignore_index, stats, done, loss.view(1))
    if N > 0:
        # TMA bulk copies and 16-byte vector stores need a 16-byte aligned base; otherwise ce_stream
        cluster_config = pick_cluster(V, ELEMENT_BYTES[inp.dtype]) if inp.data_ptr() % 16 == 0 else None
        if cluster_config is None:
            build_stream_kernel(logit_dtype, target_dtype)(logits_flat, target_flat, tstride, stats, row_loss, N, V, ignore_index)
            build_final_kernel()(row_loss, stats, loss.view(1), N)
        else:
            build_cluster_kernel(logit_dtype, target_dtype, *cluster_config)(logits_flat, target_flat, tstride, stats, row_loss, N, V, ignore_index, done, loss.view(1))
    return loss, inp
