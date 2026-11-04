"""
silu_mul_sm100: the SwiGLU product silu(gate) * up and its two gradients on SM100, TIRx-lite
kernels, the same bits as silu_mul_sm90.

Flat elementwise kernels: each thread moves VECTORS_PER_THREAD 16-byte vectors per tensor.
Exact-tile BF16 launches use Blackwell packed-f32x2 add/mul/FMA for two independent FP32 lanes at
once, while ex2/rcp remain scalar. The fp32 math: sigmoid(x) = div.full(1, ex2.approx(x * -log2e)
+ 1), computed with ex2.approx.ftz and rcp.approx.ftz(d / 4) / 4, which is identical on all 2^32
fp32 inputs and drops div.full's range check; every product is a separate mul.rn.f32 (no product
feeds an add, so nothing can contract), and the backward's (1 + gate - silu) is a single fma.rn.
bf16, fp16 and fp32 run natively; any other float dtype (or a backward with mixed dtypes) is
widened to fp32, run through the fp32 kernels and rounded once back. Bases not 16-byte aligned
take a scalar kernel.
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["silu_mul_bwd_sm100", "silu_mul_fwd_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


DTYPE_NAME = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
NEG_LOG2E = -1.4426950408889634  # rounds to 0xBFB8AA3B
VECTOR_BYTES = 16  # bytes per vector load / store
L2_POLICY = {"first": 0x12F0000000000000, "normal": 0x1000000000000000, "last": 0x14F0000000000000}


def f32_const(value):
    return txl.float32(value)


def sigmoid(x):
    scaled, exp2, denom, sig = txl.local_scalar("float32"), txl.local_scalar("float32"), txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.mul.f32(scaled, x, f32_const(NEG_LOG2E))
    # ex2 results below 2^-126 only occur where 1 + exp2 rounds to 1, so .ftz is bit-identical.
    txl.ptx.ex2.approx.ftz.f32(exp2, scaled)
    txl.ptx.add.f32(denom, exp2, f32_const(1.0))
    # div.full(1, d) for d >= 1 is rcp(d) when d <= 2^126 and rcp(d / 4) / 4 above; MUFU.RCP is
    # exponent-invariant, so the scaled form is bit-identical everywhere (checked over all 2^32 x).
    denom_quarter, rcp_quarter = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.mul.f32(denom_quarter, denom, f32_const(0.25))
    txl.ptx.rcp.approx.ftz.f32(rcp_quarter, denom_quarter)
    txl.ptx.mul.f32(sig, rcp_quarter, f32_const(0.25))
    return sig


def mul(lhs, rhs):
    product = txl.local_scalar("float32")
    txl.ptx.mul.f32(product, lhs, rhs)
    return product


def fwd_math(gate, up):
    return [mul(mul(sigmoid(gate), gate), up)]


def bwd_math(grad_output, gate, up):
    sig = sigmoid(gate)
    silu = mul(sig, gate)
    grad_up = mul(grad_output, silu)
    grad_silu_sig = mul(mul(grad_output, up), sig)
    gate_plus_one = txl.local_scalar("float32")
    txl.ptx.add.f32(gate_plus_one, gate, f32_const(1.0))
    neg_sig = txl.local_scalar("float32")
    txl.ptx.neg.f32(neg_sig, sig)
    one_plus_gate_minus_silu = txl.local_scalar("float32")
    txl.ptx.fma.rn.f32(one_plus_gate_minus_silu, neg_sig, gate, gate_plus_one)
    return [mul(grad_silu_sig, one_plus_gate_minus_silu), grad_up]


def pair(lo, hi):
    return txl.cuda.make_float2(lo, hi)


def unpair(value):
    lo, hi = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.mov.b64(lo, hi, value)
    return lo, hi


def pair_mul(lhs, rhs):
    out = txl.local_scalar("uint64")
    txl.ptx.mul.f32x2(out, lhs, rhs)
    return out


def pair_add(lhs, rhs):
    out = txl.local_scalar("uint64")
    txl.ptx.add.f32x2(out, lhs, rhs)
    return out


def pair_fma(lhs, rhs, addend):
    out = txl.local_scalar("uint64")
    txl.ptx.fma.rn.f32x2(out, lhs, rhs, addend)
    return out


def sigmoid_pair(gate_lo, gate_hi):
    gate_pair = pair(gate_lo, gate_hi)
    neg_log2e = pair(f32_const(NEG_LOG2E), f32_const(NEG_LOG2E))
    scaled_lo, scaled_hi = unpair(pair_mul(gate_pair, neg_log2e))
    exp2_lo, exp2_hi = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(exp2_lo, scaled_lo)
    txl.ptx.ex2.approx.ftz.f32(exp2_hi, scaled_hi)
    ones = pair(f32_const(1.0), f32_const(1.0))
    denom = pair_add(pair(exp2_lo, exp2_hi), ones)
    quarter = pair(f32_const(0.25), f32_const(0.25))
    denom_quarter_lo, denom_quarter_hi = unpair(pair_mul(denom, quarter))
    rcp_lo, rcp_hi = txl.local_scalar("float32"), txl.local_scalar("float32")
    txl.ptx.rcp.approx.ftz.f32(rcp_lo, denom_quarter_lo)
    txl.ptx.rcp.approx.ftz.f32(rcp_hi, denom_quarter_hi)
    return pair_mul(pair(rcp_lo, rcp_hi), quarter)


def fwd_math_packed(inputs):
    gate, up = inputs
    out = []
    for i in range(0, len(gate), 2):
        gate_pair = pair(gate[i], gate[i + 1])
        sig = sigmoid_pair(gate[i], gate[i + 1])
        out.extend(unpair(pair_mul(pair_mul(sig, gate_pair), pair(up[i], up[i + 1]))))
    return [out]


def bwd_math_packed(inputs):
    grad_output, gate, up = inputs
    grad_gate, grad_up = [], []
    for i in range(0, len(gate), 2):
        grad_output_pair, gate_pair, up_pair = pair(grad_output[i], grad_output[i + 1]), pair(gate[i], gate[i + 1]), pair(up[i], up[i + 1])
        sig = sigmoid_pair(gate[i], gate[i + 1])
        silu = pair_mul(sig, gate_pair)
        grad_up.extend(unpair(pair_mul(grad_output_pair, silu)))
        grad_silu_sig = pair_mul(pair_mul(grad_output_pair, up_pair), sig)
        gate_plus_one = pair_add(gate_pair, pair(f32_const(1.0), f32_const(1.0)))
        sign_bits = txl.local_scalar("uint64")
        txl.ptx.mov.b64(sign_bits, txl.uint32(0x80000000), txl.uint32(0x80000000))
        neg_sig = txl.bitwise_xor(sig, sign_bits)
        one_plus_gate_minus_silu = pair_fma(neg_sig, gate_pair, gate_plus_one)
        grad_gate.extend(unpair(pair_mul(grad_silu_sig, one_plus_gate_minus_silu)))
    return [grad_gate, grad_up]


def to_f32(bits16, dtype):
    return txl.cast(txl.reinterpret(bits16, ty=dtype), "float32")


def unpack(word, dtype):
    """
    One b32 word -> list of fp32 values (2 for 16-bit dtypes, 1 for fp32).
    """
    if dtype == "float32":
        return [txl.reinterpret(word, ty="float32")]
    if dtype == "bfloat16":
        # BF16 widening is exact bit placement: the source bits become the high
        # half of FP32 and the low mantissa bits are zero. Avoid two CVTs/word.
        lo, hi = txl.shift_left(word, txl.uint32(16)), txl.bitwise_and(word, txl.uint32(0xFFFF0000))
        return [txl.reinterpret(lo, ty="float32"), txl.reinterpret(hi, ty="float32")]
    lo, hi = to_f32(txl.cast(txl.bitwise_and(word, txl.uint32(0xFFFF)), "uint16"), dtype), to_f32(txl.cast(txl.shift_right(word, txl.uint32(16)), "uint16"), dtype)
    return [lo, hi]


def pack(dst, values, dtype):
    """
    fp32 values -> b32 words in dst (one RNE rounding).
    """
    if dtype == "float32":
        for i, value in enumerate(values):
            txl.ptx.mov.b32(dst[i], txl.reinterpret(value, ty="uint32"))
        return
    for pair_idx in range(len(values) // 2):
        if dtype == "float16":
            txl.ptx.cvt.rn.f16x2.f32(dst[pair_idx], values[2 * pair_idx + 1], values[2 * pair_idx])
        else:
            txl.ptx.cvt.rn.bf16x2.f32(dst[pair_idx], values[2 * pair_idx + 1], values[2 * pair_idx])


def process_element(inputs, outputs, idx, dtype, scalar_math, num_outputs, half_type):
    """
    Load element idx of every input with scalar loads, apply scalar_math and store every output.
    """
    values = []
    for src in inputs:
        if dtype == "float32":
            word = txl.local_scalar("uint32")
            txl.ptx.ld.global_.nc.b32(word, txl.address_of(src[idx]))
            values.append(txl.reinterpret(word, ty="float32"))
        else:
            half = txl.local_scalar("uint16")
            txl.ptx.ld.global_.nc.b16(half, txl.address_of(src[idx]))
            if dtype == "bfloat16":
                values.append(txl.reinterpret(txl.shift_left(txl.cast(half, "uint32"), txl.uint32(16)), ty="float32"))
            else:
                values.append(to_f32(half, dtype))
    results = scalar_math(*values)
    for output_idx in range(num_outputs):
        if dtype == "float32":
            txl.ptx.st.global_.b32(txl.address_of(outputs[output_idx][idx]), txl.reinterpret(results[output_idx], ty="uint32"))
        else:
            half = txl.local_scalar("uint16")
            getattr(txl.ptx.cvt.rn, half_type).f32(half, results[output_idx])
            txl.ptx.st.global_.b16(txl.address_of(outputs[output_idx][idx]), half)


def compile_kernel(kernel):
    return entry_function(kernel.compile(TARGET))


SMALL_NUMEL = 1 << 22  # elements; at or below, the whole problem is one latency-bound wave


def pick_kernel(dtype: str, num_inputs: int, numel: int):
    """
    Shape-aware launch config of the vector kernel.

    Inputs stream with L2 evict_first (read once). Outputs use the default policy: evict_last
    stores survive a cold-L2 flush and pin lines that later kernels (and benchmark iterations)
    then hit, which games cold-L2 timing and pollutes L2 in training.
    """
    if num_inputs == 2:  # forward: 3 streams; small = more CTAs, large = 32 B in flight per stream
        warps, vectors_per_thread = (4, 1) if numel <= SMALL_NUMEL else (8, 2)
    else:  # backward: 5 streams; one 16 B vector per stream keeps registers / occupancy up
        warps, vectors_per_thread = 8, 1
    num_elems = 4 if dtype == "float32" else 8
    full_tiles = numel % (warps * 32 * vectors_per_thread * num_elems) == 0
    return build_vector_kernel(dtype, num_inputs, warps, vectors_per_thread, 16, "", "", "first", "", full_tiles)


@functools.cache
def build_scalar_kernel(dtype: str, num_inputs: int):
    """
    Unaligned-safe fallback, one element per thread.
    """
    num_outputs = 1 if num_inputs == 2 else 2
    scalar_math = fwd_math if num_inputs == 2 else bwd_math
    THREADS = 256
    half_type = {"float32": None, "float16": "f16", "bfloat16": "bf16"}[dtype]

    def body(ptrs, numel):
        inputs, outputs = ptrs[:num_inputs], ptrs[num_inputs:]
        idx = txl.cast(txl.cta_id(), "int64") * THREADS + txl.cast(txl.thread_id(), "int64")
        with txl.if_(idx < numel):
            with txl.then_():
                process_element(inputs, outputs, idx, dtype, scalar_math, num_outputs, half_type)

    grid_size = lambda numel: txl.cast((numel + THREADS - 1) // THREADS, "int32")
    if num_inputs == 2:
        @txl.kernel()
        def silu_mul_fwd_scalar(gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid_size(n), block=256))
            body([gate, up, out], n)

        return compile_kernel(silu_mul_fwd_scalar)

    @txl.kernel()
    def silu_mul_bwd_scalar(do: txl.Tensor((txl.int64(),), dtype, data_alignment=1), gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid_size(n), block=256))
        body([do, gate, up, grad_gate, grad_up], n)

    return compile_kernel(silu_mul_bwd_scalar)


def apply_hints(op, hints):
    """
    op.<part> for each part of the dotted hints string, op itself when hints is empty.
    """
    for part in hints.split(".") if hints else []:
        op = getattr(op, part)
    return op


@functools.cache
def build_vector_kernel(dtype: str, num_inputs: int, WARPS: int, VECTORS_PER_THREAD: int, VECTOR_BYTES: int, LOAD_HINT: str, STORE_HINT: str, LOAD_POLICY: str = "", STORE_POLICY: str = "", FULL_TILES: bool = False):
    """
    num_inputs = 2 (fwd: gate, up -> out) or 3 (bwd: grad_output, gate, up -> grad_gate, grad_up).
    """
    num_outputs = 1 if num_inputs == 2 else 2
    scalar_math = fwd_math if num_inputs == 2 else bwd_math
    THREADS = WARPS * 32
    num_words = VECTOR_BYTES // 4  # b32 words per vector
    num_elems = num_words if dtype == "float32" else 2 * num_words  # elements per vector
    load_op, store_op = apply_hints(txl.ptx.ld.global_.nc, LOAD_HINT), apply_hints(txl.ptx.st.global_, STORE_HINT)
    if LOAD_POLICY:
        load_with_policy = getattr(getattr(load_op, "L2::cache_hint"), f"v{num_words}").b32
        load_vector = lambda *operands: load_with_policy(*operands, txl.uint64(L2_POLICY[LOAD_POLICY]))
    else:
        load_vector = getattr(load_op, f"v{num_words}").b32
    if STORE_POLICY:
        store_with_policy = getattr(getattr(store_op, "L2::cache_hint"), f"v{num_words}").b32
        store_vector = lambda *operands: store_with_policy(*operands, txl.uint64(L2_POLICY[STORE_POLICY]))
    else:
        store_vector = getattr(store_op, f"v{num_words}").b32
    elems_per_cta = THREADS * VECTORS_PER_THREAD * num_elems
    half_type = {"float32": None, "float16": "f16", "bfloat16": "bf16"}[dtype]

    def body(ptrs, numel):
        inputs, outputs = ptrs[:num_inputs], ptrs[num_inputs:]
        cta_idx, tidx = txl.cast(txl.cta_id(), "int64"), txl.cast(txl.thread_id(), "int64")
        cta_start = cta_idx * elems_per_cta
        in_words = [[txl.alloc_local([num_words], "uint32") for _ in range(num_inputs)] for _ in range(VECTORS_PER_THREAD)]
        out_words = [txl.alloc_local([num_words], "uint32") for _ in range(num_outputs)]

        def vector_start(vector_idx):
            return cta_start + (vector_idx * THREADS + tidx) * num_elems

        def load_inputs(vector_idx):
            for input_idx in range(num_inputs):
                words = in_words[vector_idx][input_idx]
                load_vector(*[words[i] for i in range(num_words)], txl.address_of(inputs[input_idx][vector_start(vector_idx)]))

        def compute_store(vector_idx):
            unpacked = [[value for word_idx in range(num_words) for value in unpack(in_words[vector_idx][input_idx][word_idx], dtype)] for input_idx in range(num_inputs)]
            if FULL_TILES and dtype == "bfloat16":
                results_by_output = fwd_math_packed(unpacked) if num_inputs == 2 else bwd_math_packed(unpacked)
            else:
                results = [scalar_math(*[unpacked[input_idx][elem_idx] for input_idx in range(num_inputs)]) for elem_idx in range(num_elems)]
                results_by_output = [[results[elem_idx][output_idx] for elem_idx in range(num_elems)] for output_idx in range(num_outputs)]
            for output_idx in range(num_outputs):
                pack(out_words[output_idx], results_by_output[output_idx], dtype)
                store_vector(txl.address_of(outputs[output_idx][vector_start(vector_idx)]), *[out_words[output_idx][i] for i in range(num_words)])

        def process_vector(vector_idx):
            load_inputs(vector_idx)
            compute_store(vector_idx)

        # All loads are issued before any math so the vectors of a thread are in flight together.
        if FULL_TILES:
            for vector_idx in range(VECTORS_PER_THREAD):
                load_inputs(vector_idx)
            for vector_idx in range(VECTORS_PER_THREAD):
                compute_store(vector_idx)
        else:
            with txl.if_(cta_start + elems_per_cta <= numel):
                with txl.then_():
                    for vector_idx in range(VECTORS_PER_THREAD):
                        load_inputs(vector_idx)
                    for vector_idx in range(VECTORS_PER_THREAD):
                        compute_store(vector_idx)
                with txl.else_():
                    for vector_idx in range(VECTORS_PER_THREAD):
                        elem_start = vector_start(vector_idx)
                        with txl.if_(elem_start + num_elems <= numel):
                            with txl.then_():
                                process_vector(vector_idx)
                            with txl.else_():
                                for elem_idx in range(num_elems):
                                    with txl.if_(elem_start + elem_idx < numel):
                                        with txl.then_():
                                            process_element(inputs, outputs, elem_start + elem_idx, dtype, scalar_math, num_outputs, half_type)

    grid_size = lambda numel: txl.cast((numel + (elems_per_cta - 1)) // elems_per_cta, "int32")
    if num_inputs == 2:
        @txl.kernel()
        def silu_mul_fwd(gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid_size(n), block=WARPS * 32))
            body([gate, up, out], n)

        return compile_kernel(silu_mul_fwd)

    @txl.kernel()
    def silu_mul_bwd(do: txl.Tensor((txl.int64(),), dtype, data_alignment=1), gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_gate: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), n: txl.int64):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid_size(n), block=WARPS * 32))
        body([do, gate, up, grad_gate, grad_up], n)

    return compile_kernel(silu_mul_bwd)


def is_vector_aligned(*tensors):
    return all(tensor.data_ptr() % VECTOR_BYTES == 0 for tensor in tensors)


def silu_mul_fwd_sm100(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    silu(gate) * up of contiguous same-shape, same-dtype gate and up as a new tensor like gate, the
    bits of silu_mul_fwd_sm90.
    """
    assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.dtype == up.dtype, f"dtype mismatch: {gate.dtype} vs {up.dtype}"
    assert gate.is_contiguous(), "gate must be contiguous"
    assert up.is_contiguous(), "up must be contiguous"
    if gate.dtype not in DTYPE_NAME:  # other float dtypes: the fp32 kernel, rounded once back
        return silu_mul_fwd_sm100(gate.float(), up.float()).to(gate.dtype)
    out = torch.empty_like(gate)
    numel = gate.numel()
    if numel == 0:
        return out
    kernel = pick_kernel(DTYPE_NAME[gate.dtype], 2, numel) if is_vector_aligned(gate, up, out) else build_scalar_kernel(DTYPE_NAME[gate.dtype], 2)
    kernel(gate.view(-1), up.view(-1), out.view(-1), numel)
    return out


def silu_mul_bwd_sm100(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients of silu_mul_fwd_sm100 as new (grad_gate, grad_up) like gate and up, the bits of
    silu_mul_bwd_sm90.
    """
    assert grad_output.shape == gate.shape, f"grad_output shape {grad_output.shape} != gate shape {gate.shape}"
    assert grad_output.dtype == gate.dtype, f"grad_output dtype {grad_output.dtype} != gate dtype {gate.dtype}"
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    if gate.dtype not in DTYPE_NAME or up.dtype != gate.dtype:  # the fp32 kernel, each rounded once back
        grad_gate, grad_up = silu_mul_bwd_sm100(grad_output.float(), gate.float(), up.float())
        return grad_gate.to(gate.dtype), grad_up.to(up.dtype)
    gate, up = gate.contiguous(), up.contiguous()
    grad_gate, grad_up = torch.empty_like(gate), torch.empty_like(up)
    numel = gate.numel()
    if numel == 0:
        return grad_gate, grad_up
    kernel = pick_kernel(DTYPE_NAME[gate.dtype], 3, numel) if is_vector_aligned(grad_output, gate, up, grad_gate, grad_up) else build_scalar_kernel(DTYPE_NAME[gate.dtype], 3)
    kernel(grad_output.view(-1), gate.view(-1), up.view(-1), grad_gate.view(-1), grad_up.view(-1), numel)
    return grad_gate, grad_up
