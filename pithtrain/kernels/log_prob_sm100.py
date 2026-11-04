"""
log_prob_sm100: the fused per-token log-probability and entropy of a tempered softmax (forward) and
its backward overwriting the logits with their gradient, on SM100, as TIRx-lite kernels. Contracts
as in the picker log_prob.

Forward: one CTA per row, each thread streams 16-byte vectors and keeps a base-2 online state (m2,
d, s): u = x * c - m2 with c = log2(e) / T, d = sum 2^u, s = sum 2^u * u. The fast path skips the
per-element running max: each thread fixes its reference m2 from its first vector plus a margin,
which is exact as long as no 2^u overflows and no logit is -inf; a thread whose (d, s) come out
non-finite redoes its own elements with the exact online max (-inf clamped). A thread also redoes
its elements exactly when log2 d > REDO_LOG2 (its reference sits far below its mass), which keeps
the entropy cancellation and the backward's log2 p accurate on spike rows. The warp merge takes the
common maximum with Blackwell's redux.sync.max.f32, rescales each lane once and butterfly-sums d and
s; the CTA then merges the warps the same way. row_max holds m2 and log_sum holds log2(d) (base 2),
which only this file's backward reads.

Backward: one CTA per (row, CHUNK-vector slice), so the grid's tail is a short slice rather than a
whole row. u = (x * c - m2) - log2 d = log2 p, dx = p * (A + B * u) + [i == y] g_logp / T with A =
-(g_logp + g_entropy * H) / T and B = -g_entropy * ln2 / T. The thread that owns the target's vector
adds g_logp / T to that lane before the single rounding.

The kernels read bf16 or fp32 logits and int64 targets. Other float logits run them on an fp32 copy
(the gradient is computed in fp32 and rounded once into inp), other integer targets on an int64
copy.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["log_prob_bwd_sm100", "log_prob_fwd_sm100"]


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
MARGIN = 0.0  # reference = first-vector max; 2^u overflow (u > ~117) falls back
REDO_LOG2 = 20.0
WARPS = 8
THREADS = WARPS * 32
UNROLL = 4
FULL_MASK = 0xFFFFFFFF


def f32(value):
    return txl.float32(value)


def i64(value):
    return txl.Cast("int64", value)


def load_vector(x, dtype, vec, elem_idx, dst, dst_offset):
    """
    Load vec consecutive elements at element index elem_idx (int64) as f32 into
    dst[dst_offset:][:vec].
    """
    if dtype == "bfloat16":
        if vec == 8:
            words = txl.alloc_local([4], "uint32")
            txl.ptx["ld.global.nc.L1::no_allocate.v4.b32"](words[0], words[1], words[2], words[3], x.ptr_to([elem_idx]))
            for i in range(4):
                txl.assign(dst[dst_offset + 2 * i], txl.reinterpret(words[i] << txl.uint32(16), ty="float32"))
                txl.assign(dst[dst_offset + 2 * i + 1], txl.reinterpret(words[i] & txl.uint32(0xFFFF0000), ty="float32"))
        else:
            half = txl.alloc_local([1], "uint16")
            txl.ptx["ld.global.nc.b16"](half[0], x.ptr_to([elem_idx]))
            txl.assign(dst[dst_offset], txl.reinterpret(txl.Cast("uint32", half[0]) << txl.uint32(16), ty="float32"))
    else:
        if vec == 4:
            txl.ptx["ld.global.nc.L1::no_allocate.v4.f32"](dst[dst_offset], dst[dst_offset + 1], dst[dst_offset + 2], dst[dst_offset + 3], x.ptr_to([elem_idx]))
        else:
            txl.ptx["ld.global.nc.f32"](dst[dst_offset], x.ptr_to([elem_idx]))


def store_vector(x, dtype, vec, elem_idx, src, src_offset):
    """
    Round src[src_offset:][:vec] once to x's dtype and store it at element index elem_idx.
    """
    if dtype == "bfloat16":
        if vec == 8:
            words = txl.alloc_local([4], "uint32")
            for i in range(4):
                txl.ptx["cvt.rn.bf16x2.f32"](words[i], src[src_offset + 2 * i + 1], src[src_offset + 2 * i])
            txl.ptx["st.global.L1::no_allocate.v4.b32"](x.ptr_to([elem_idx]), words[0], words[1], words[2], words[3])
        else:
            half = txl.alloc_local([1], "uint16")
            txl.ptx["cvt.rn.bf16.f32"](half[0], src[src_offset])
            txl.ptx["st.global.b16"](x.ptr_to([elem_idx]), half[0])
    else:
        if vec == 4:
            txl.ptx["st.global.L1::no_allocate.v4.f32"](x.ptr_to([elem_idx]), src[src_offset], src[src_offset + 1], src[src_offset + 2], src[src_offset + 3])
        else:
            txl.ptx["st.global.f32"](x.ptr_to([elem_idx]), src[src_offset])


def is_finite(value):
    return (txl.reinterpret(value, ty="uint32") & txl.uint32(0x7F800000)) != txl.uint32(0x7F800000)


def ex2(dst, src):
    txl.ptx.ex2.approx.ftz.f32(dst, src)


def shfl_bfly(dst, value, delta):
    peer = txl.alloc_local([1], "uint32")
    txl.ptx.shfl_sync.bfly.b32(peer[0], txl.reinterpret(value, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
    txl.assign(dst, txl.reinterpret(peer[0], ty="float32"))


def merge_warp_state(state, rounds):
    """
    Merge the lanes' (m2, d, s) into every lane, butterfly-summing over the given shuffle deltas.
    """
    # Blackwell provides a native fp32 warp max. Once every lane is expressed relative to the
    # same maximum, d and s are ordinary sums; there is no need to rescale at every tree level.
    warp_max = txl.local_scalar("float32")
    txl.ptx.redux_sync.max.f32(warp_max, state[0], txl.uint32(FULL_MASK))
    scale = txl.local_scalar("float32")
    ex2(scale, state[0] - warp_max)
    txl.assign(state[2], (state[2] + (state[0] - warp_max) * state[1]) * scale)
    txl.assign(state[1], state[1] * scale)
    txl.assign(state[0], warp_max)
    peer_sums = txl.alloc_local([2], "float32")
    for delta in rounds:
        shfl_bfly(peer_sums[0], state[1], delta)
        shfl_bfly(peer_sums[1], state[2], delta)
        txl.assign(state[1], state[1] + peer_sums[0])
        txl.assign(state[2], state[2] + peer_sums[1])


def sweep(x, dtype, vec, row_offset, num_vecs, tidx, vals, visit, vec_start=0, num_threads=THREADS):
    """
    Visit this thread's vectors vec_idx = vec_start + tidx + k * num_threads below vec_start +
    num_vecs: UNROLL loads in flight, then a guarded tail.
    """
    num_main_iters = num_vecs // (UNROLL * num_threads)
    tail_start = num_main_iters * (UNROLL * num_threads)
    with txl.serial(num_main_iters) as iter_idx:
        for slot in range(UNROLL):
            vec_idx = vec_start + iter_idx * (UNROLL * num_threads) + slot * num_threads + tidx
            load_vector(x, dtype, vec, row_offset + i64(vec_idx) * vec, vals, slot * vec)
        for slot in range(UNROLL):
            visit(slot * vec, vec_start + iter_idx * (UNROLL * num_threads) + slot * num_threads + tidx)
    with txl.serial((num_vecs - tail_start + num_threads - 1) // num_threads) as iter_idx:
        vec_idx = tail_start + iter_idx * num_threads + tidx
        with txl.if_(vec_idx < num_vecs), txl.then_():
            load_vector(x, dtype, vec, row_offset + i64(vec_start + vec_idx) * vec, vals, 0)
            visit(0, vec_start + vec_idx)


@functools.cache
def build_log_prob_fwd(dtype, vec):
    @txl.kernel()
    def log_prob_fwd(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), target: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), logp: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), entropy: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), row_max: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), log_sum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), num_rows: txl.int32, num_cols: txl.int32, x_stride: txl.int64, target_stride: txl.int64, inv_temperature: txl.float32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_rows, block=WARPS * 32))
        tidx = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        # Reverse row order: the rows a preceding pass touched last are still in L2.
        row_idx = txl.local_scalar("int32", init=num_rows - 1 - txl.cta_id())
        row_offset = txl.local_scalar("int64", init=i64(row_idx) * x_stride)
        log2e_over_t = txl.local_scalar("float32", init=inv_temperature * f32(LOG2E))
        num_vecs = num_cols // vec

        smem = txl.smem_pool()
        s_warp_states = smem.alloc([WARPS * 3], txl.f32)

        state = txl.alloc_local([3], "float32")  # m2, d, s
        acc_d, acc_s = txl.alloc_local([UNROLL], "float32"), txl.alloc_local([UNROLL], "float32")
        exp_vals, exp_u_vals = txl.alloc_local([vec], "float32"), txl.alloc_local([vec], "float32")
        vals, u = txl.alloc_local([UNROLL * vec], "float32"), txl.alloc_local([1], "float32")

        def load_target_logit():
            """
            Thread 0 loads the target index and, when it is in range, the target logit, early.
            """
            target_idx = txl.local_scalar("int64", init=txl.int64(-1))
            target_logit = txl.alloc_local([1], "float32")
            with txl.if_(tidx == 0), txl.then_():
                txl.ptx["ld.global.nc.b64"](target_idx, target.ptr_to([i64(row_idx) * target_stride]))
                txl.assign(target_logit[0], f32(0.0))
                with txl.if_(txl.And(target_idx >= txl.int64(0), target_idx < i64(num_cols))), txl.then_():
                    load_vector(x, dtype, 1, row_offset + target_idx, target_logit, 0)
            return target_idx, target_logit

        def set_thread_reference():
            """
            Fix the fast path's reference m2 from the thread's first vector.
            """
            txl.assign(state[0], f32(-1e30))
            with txl.if_(tidx < num_vecs), txl.then_():
                load_vector(x, dtype, vec, row_offset + i64(tidx) * vec, vals, 0)
                vec_max = txl.local_scalar("float32", init=vals[0])
                for k in range(1, vec):
                    txl.assign(vec_max, max_drop_nan(vec_max, vals[k]))
                txl.assign(state[0], vec_max * log2e_over_t + f32(MARGIN))

        def reset_accumulators():
            for slot in range(UNROLL):
                txl.assign(acc_d[slot], f32(0.0))
                txl.assign(acc_s[slot], f32(0.0))

        def sum_slots():
            txl.assign(state[1], (acc_d[0] + acc_d[1]) + (acc_d[2] + acc_d[3]))
            txl.assign(state[2], (acc_s[0] + acc_s[1]) + (acc_s[2] + acc_s[3]))

        def accumulate_vector(slot_offset):
            """
            Tree-sum the vector's (e, e * u) and add it to the slot's accumulators.
            """
            slot = slot_offset // vec
            width = vec
            while width > 1:
                width //= 2
                for k in range(width):
                    txl.assign(exp_vals[k], exp_vals[k] + exp_vals[k + width])
                    txl.assign(exp_u_vals[k], exp_u_vals[k] + exp_u_vals[k + width])
            txl.assign(acc_d[slot], acc_d[slot] + exp_vals[0])
            txl.assign(acc_s[slot], acc_s[slot] + exp_u_vals[0])

        def visit_fast(slot_offset, vec_idx):
            for k in range(vec):
                txl.assign(u[0], vals[slot_offset + k] * log2e_over_t - state[0])
                ex2(exp_vals[k], u[0])
                txl.assign(exp_u_vals[k], exp_vals[k] * u[0])
            accumulate_vector(slot_offset)

        def visit_exact(slot_offset, vec_idx):
            vec_max = txl.local_scalar("float32", init=vals[slot_offset])
            for k in range(1, vec):
                txl.assign(vec_max, max_drop_nan(vec_max, vals[slot_offset + k]))
            new_max = txl.local_scalar("float32", init=max_drop_nan(state[0], vec_max * log2e_over_t))
            with txl.if_(new_max > state[0]), txl.then_():
                rescale = txl.local_scalar("float32")
                ex2(rescale, state[0] - new_max)
                for slot in range(UNROLL):
                    txl.assign(acc_s[slot], (acc_s[slot] + (state[0] - new_max) * acc_d[slot]) * rescale)
                    txl.assign(acc_d[slot], acc_d[slot] * rescale)
                txl.assign(state[0], new_max)
            for k in range(vec):
                txl.assign(u[0], max_drop_nan(vals[slot_offset + k] * log2e_over_t - state[0], f32(-3.0e38)))
                ex2(exp_vals[k], u[0])
                txl.assign(exp_u_vals[k], exp_vals[k] * u[0])
            accumulate_vector(slot_offset)

        def redo_exact_if_needed():
            """
            Redo the thread's elements with the exact online max on overflow, -inf or NaN, or when
            the reference sits far below the thread's mass (log2 d > REDO_LOG2), which would cost
            fp32 accuracy in the entropy's log2 d - s / d cancellation and in the backward's log2 p.
            """
            thread_log2_d = txl.local_scalar("float32")
            txl.ptx.lg2.approx.ftz.f32(thread_log2_d, state[1])
            needs_redo = txl.Or(txl.Not(txl.And(is_finite(state[1]), is_finite(state[2]))), thread_log2_d > f32(REDO_LOG2))
            with txl.if_(needs_redo), txl.then_():
                txl.assign(state[0], f32(-1e30))
                reset_accumulators()
                sweep(x, dtype, vec, row_offset, num_vecs, tidx, vals, visit_exact)
                sum_slots()

        def share_warp_states():
            """
            Merge each warp, then publish its state to shared memory for warp 0.
            """
            merge_warp_state(state, (16, 8, 4, 2, 1))
            with txl.if_(lane == 0), txl.then_():
                for k in range(3):
                    txl.ptx["st.shared.f32"](s_warp_states.ptr_to([warp * 3 + k]), state[k])
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))

        def merge_warp_states():
            """
            Warp 0: merge the warps' states published in shared memory.
            """
            with txl.if_(lane < WARPS):
                with txl.then_():
                    for k in range(3):
                        txl.ptx["ld.shared.f32"](state[k], s_warp_states.ptr_to([lane * 3 + k]))
                with txl.else_():
                    txl.assign(state[0], f32(-1e30))
                    txl.assign(state[1], f32(0.0))
                    txl.assign(state[2], f32(0.0))
            merge_warp_state(state, tuple(1 << i for i in reversed(range(WARPS.bit_length() - 1))))

        def store_row_stats(target_idx, target_logit):
            log2_d = txl.local_scalar("float32")
            txl.ptx.lg2.approx.ftz.f32(log2_d, state[1])
            target_logp = txl.local_scalar("float32", init=f32(0.0))
            with txl.if_(txl.And(target_idx >= txl.int64(0), target_idx < i64(num_cols))), txl.then_():
                txl.assign(target_logp, ((target_logit[0] * log2e_over_t - state[0]) - log2_d) * f32(LN2))
            row_idx64 = i64(row_idx)
            txl.ptx["st.global.f32"](logp.ptr_to([row_idx64]), target_logp)
            txl.ptx["st.global.f32"](entropy.ptr_to([row_idx64]), (log2_d - state[2] / state[1]) * f32(LN2))
            txl.ptx["st.global.f32"](row_max.ptr_to([row_idx64]), state[0])
            txl.ptx["st.global.f32"](log_sum.ptr_to([row_idx64]), log2_d)

        target_idx, target_logit = load_target_logit()

        set_thread_reference()
        reset_accumulators()
        sweep(x, dtype, vec, row_offset, num_vecs, tidx, vals, visit_fast)
        sum_slots()

        redo_exact_if_needed()

        share_warp_states()
        with txl.if_(warp == 0), txl.then_():
            merge_warp_states()
            with txl.if_(tidx == 0), txl.then_():
                store_row_stats(target_idx, target_logit)

    return log_prob_fwd


CHUNK = 2048  # vectors per backward CTA (16384 bf16 elements)
BWD_WARPS = 8
BWD_THREADS = BWD_WARPS * 32


@functools.cache
def build_log_prob_bwd(dtype, vec):
    @txl.kernel()
    def log_prob_bwd(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), target: txl.Tensor((txl.int64(),), txl.i64, data_alignment=1), row_max: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), log_sum: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), entropy: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), grad_logp: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), grad_entropy: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), num_ctas: txl.int32, num_chunks: txl.int32, num_cols: txl.int32, x_stride: txl.int64, target_stride: txl.int64, grad_logp_stride: txl.int64, grad_entropy_stride: txl.int64, inv_temperature: txl.float32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=num_ctas, block=BWD_WARPS * 32))
        tidx = txl.thread_id()
        cta_idx = txl.cta_id()
        row_idx = txl.local_scalar("int32", init=cta_idx // num_chunks)
        vec_start = txl.local_scalar("int32", init=(cta_idx % num_chunks) * CHUNK)
        row_idx64 = i64(row_idx)
        row_offset = txl.local_scalar("int64", init=row_idx64 * x_stride)
        log2e_over_t = txl.local_scalar("float32", init=inv_temperature * f32(LOG2E))
        num_vecs = num_cols // vec
        num_chunk_vecs = txl.local_scalar("int32", init=min_drop_nan(num_vecs - vec_start, CHUNK))

        target_idx = txl.local_scalar("int64")
        txl.ptx["ld.global.nc.b64"](target_idx, target.ptr_to([row_idx64 * target_stride]))
        row_stats = txl.alloc_local([5], "float32")  # m2, log2 d, H, g_logp, g_entropy
        txl.ptx["ld.global.nc.f32"](row_stats[0], row_max.ptr_to([row_idx64]))
        txl.ptx["ld.global.nc.f32"](row_stats[1], log_sum.ptr_to([row_idx64]))
        txl.ptx["ld.global.nc.f32"](row_stats[2], entropy.ptr_to([row_idx64]))
        txl.ptx["ld.global.nc.f32"](row_stats[3], grad_logp.ptr_to([row_idx64 * grad_logp_stride]))
        txl.ptx["ld.global.nc.f32"](row_stats[4], grad_entropy.ptr_to([row_idx64 * grad_entropy_stride]))
        has_target = txl.And(target_idx >= txl.int64(0), target_idx < i64(num_cols))
        target_grad_logp = txl.local_scalar("float32", init=txl.Select(has_target, row_stats[3], f32(0.0)))

        coef_a = txl.local_scalar("float32", init=-(target_grad_logp + row_stats[4] * row_stats[2]) * inv_temperature)
        coef_b = txl.local_scalar("float32", init=-(row_stats[4] * f32(LN2)) * inv_temperature)
        target_grad = txl.local_scalar("float32", init=target_grad_logp * inv_temperature)
        target_vec = txl.local_scalar("int32", init=txl.Select(has_target, txl.Cast("int32", target_idx // vec), -1))
        target_elem = txl.local_scalar("int32", init=txl.Cast("int32", target_idx % vec))

        vals, u, p = txl.alloc_local([UNROLL * vec], "float32"), txl.alloc_local([1], "float32"), txl.alloc_local([1], "float32")

        def store_grad_vector(slot_offset, vec_idx):
            for k in range(vec):
                txl.assign(u[0], max_drop_nan((vals[slot_offset + k] * log2e_over_t - row_stats[0]) - row_stats[1], f32(-256.0)))
                ex2(p[0], u[0])
                txl.assign(vals[slot_offset + k], p[0] * (coef_a + coef_b * u[0]))
            # The target's own lane takes + g_logp / T before the single rounding.
            with txl.if_(vec_idx == target_vec), txl.then_():
                for k in range(vec):
                    txl.assign(vals[slot_offset + k], vals[slot_offset + k] + txl.Select(target_elem == k, target_grad, f32(0.0)))
            store_vector(x, dtype, vec, row_offset + i64(vec_idx) * vec, vals, slot_offset)

        sweep(x, dtype, vec, row_offset, num_chunk_vecs, tidx, vals, store_grad_vector, vec_start, BWD_THREADS)

    return log_prob_bwd


def flat_view(tensor):
    """
    A 1-D view over the storage that tensor's (possibly strided or stride-0) elements span.
    """
    if tensor.dim() == 1 and tensor.stride(0) == 1:
        return tensor
    length = tensor.shape[0]
    stride = tensor.stride(0) if length > 1 else 1
    span = (length - 1) * stride + 1 if stride > 0 else 1
    return tensor.as_strided((span,), (1,))


def pick_vec(inp):
    """
    Elements per 16-byte vector, or 1 when the rows are not 16-byte aligned.
    """
    vec = 8 if inp.dtype == torch.bfloat16 else 4
    num_cols = inp.shape[1]
    is_aligned = num_cols % vec == 0 and inp.stride(0) % vec == 0 and inp.data_ptr() % 16 == 0
    return vec if is_aligned else 1


def flat_logits(inp):
    """
    A 1-D view over the storage the rows of inp span.
    """
    num_rows, num_cols = inp.shape
    if inp.is_contiguous():
        return inp.view(-1)
    return inp.as_strided(((num_rows - 1) * inp.stride(0) + num_cols,), (1,))


@functools.cache
def compile_kernel(kind, dtype, vec):
    kernel = (build_log_prob_fwd if kind == "fwd" else build_log_prob_bwd)(dtype, vec)
    return entry_function(kernel.compile(TARGET))


KERNEL_DTYPE = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


def cast_to_kernel_dtypes(inp, target):
    """
    inp and target in dtypes the kernels read: other float logits as an fp32 copy, other integer
    targets as an int64 copy.
    """
    return (inp if inp.dtype in KERNEL_DTYPE else inp.float()), (target if target.dtype == torch.int64 else target.long())


def log_prob_fwd_sm100(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (logp, entropy, row_max, log_sum), each fp32 (N,); inp is left unchanged. row_max and log_sum
    are the base-2 row statistics log_prob_bwd_sm100 reads.
    """
    num_rows, num_cols = inp.shape
    logp = torch.empty(num_rows, dtype=torch.float32, device=inp.device)
    entropy, row_max, log_sum = torch.empty_like(logp), torch.empty_like(logp), torch.empty_like(logp)
    if num_rows == 0:
        return logp, entropy, row_max, log_sum
    if num_cols == 0:
        # an empty distribution: no target is valid, and the entropy is an empty sum
        return logp.zero_(), entropy.zero_(), row_max.zero_(), log_sum.zero_()
    inp, target = cast_to_kernel_dtypes(inp, target)
    vec = pick_vec(inp)
    fn = compile_kernel("fwd", KERNEL_DTYPE[inp.dtype], vec)
    target_stride = target.stride(0) if num_rows > 1 else 1
    fn(flat_logits(inp), flat_view(target), logp, entropy, row_max, log_sum, num_rows, num_cols, inp.stride(0), target_stride, float(inv_temperature))
    return logp, entropy, row_max, log_sum


def log_prob_bwd_sm100(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    """
    Overwrite inp with dL/dx, from the statistics of log_prob_fwd_sm100.
    """
    num_rows, num_cols = inp.shape
    if num_rows == 0 or num_cols == 0:
        return
    out = inp
    inp, target = cast_to_kernel_dtypes(inp, target)
    vec = pick_vec(inp)
    fn = compile_kernel("bwd", KERNEL_DTYPE[inp.dtype], vec)
    target_stride = target.stride(0) if num_rows > 1 else 1
    grad_logp_stride = grad_logp.stride(0) if num_rows > 1 else 1
    grad_entropy_stride = grad_entropy.stride(0) if num_rows > 1 else 1
    num_chunks = (num_cols // vec + CHUNK - 1) // CHUNK
    fn(flat_logits(inp), flat_view(target), row_max, log_sum, entropy, flat_view(grad_logp), flat_view(grad_entropy), num_rows * num_chunks, num_chunks, num_cols, inp.stride(0), target_stride, grad_logp_stride, grad_entropy_stride, float(inv_temperature))
    if inp is not out:
        out.copy_(inp)  # the fp32 gradient, rounded once to the logits' dtype
