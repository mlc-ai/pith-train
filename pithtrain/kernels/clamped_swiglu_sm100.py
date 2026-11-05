"""
clamped_swiglu_sm100: the clamped SwiGLU forward and backward of an interleaved gate_up on SM100,
TIRx-lite kernels, the same bits as clamped_swiglu_sm90.

Flat elementwise kernels over M * N outputs. Each thread owns UNROLL groups of VEC consecutive
outputs (groups spaced one CTA tile apart so every warp access stays coalesced); it issues all of
its wide loads before computing, then writes packed results. The fp32 sequence is sigmoid = 1 /
(1 + ex2.approx(x * -log2 e)) with div.full.f32 semantics, the ex2 / division range fixups that
cannot change the result removed (see sigmoid_parts), and the products issued two outputs at a
time as packed f32x2 (same per-lane rn rounding). Shapes, alpha and limit are runtime arguments.
bf16, fp16 and fp32 run natively; any other float dtype is widened to fp32, run through the fp32
kernel and rounded once back (the arithmetic is fp32 with one rounding).
"""

import functools

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["clamped_swiglu_bwd_sm100", "clamped_swiglu_fwd_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on every
    call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


LAUNCH_CONFIG = {"fwd": {"threads": 256, "vec": 16, "unroll": 1, "ld": "ld.global.v8.b32", "st": "st.global.v8.b32"}, "bwd": {"threads": 64, "vec": 16, "unroll": 1, "ld": "ld.global.v8.b32", "st": "st.global.v8.b32"}}  # per entry: threads, outputs per thread (vec), unroll, load / store instruction
NEG_LOG2E_BITS = 0xBFB8AA3B  # -log2(e) rounded to fp32
KERNEL_DTYPE = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def f32_from_bits(bits):
    """
    The fp32 constant with bit pattern bits.
    """
    return txl.reinterpret(txl.uint32(bits), ty="float32")


def emit_ptx(instruction, *args, dtype="float32"):
    """
    One PTX instruction into a new register of dtype, which is returned.
    """
    out = txl.local_scalar(dtype)
    txl.ptx[instruction](out, *args)
    return out


def sigmoid_parts(gate, alpha):
    """
    x = alpha * gate and sigmoid(x), in the module's fp32 sequence.
    """
    x = emit_ptx("mul.rn.f32", alpha, gate)
    exponent = emit_ptx("mul.rn.f32", x, f32_from_bits(NEG_LOG2E_BITS))
    # ex2's denormal-range fixup cannot change e + 1 (both sides round to 1.0), so .ftz is exact here.
    exp2 = emit_ptx("ex2.approx.ftz.f32", exponent)
    # div.full.f32(1, d) for d = e + 1 >= 1 is rcp(d), or rcp(d / 4) / 4 when d > 2^126; MUFU.RCP is
    # exact under power-of-two scaling, so scaling by 1/4 unconditionally gives the same bits, and
    # fma(e, 1/4, 1/4) == rn(e + 1) / 4 exactly. (Verified over all 2^32 exponents on the B200.)
    rcp_quarter = emit_ptx("rcp.approx.ftz.f32", emit_ptx("fma.rn.f32", exp2, txl.float32(0.25), txl.float32(0.25)))
    sigmoid = emit_ptx("mul.rn.f32", rcp_quarter, txl.float32(0.25))
    return x, sigmoid


def clamp_gate_up(gate, up, limit, neg_limit):
    """
    gate clamped to at most limit, up to [-limit, limit].
    """
    clamped_gate = emit_ptx("min.f32", gate, limit)
    clamped_up = emit_ptx("min.f32", emit_ptx("max.f32", up, neg_limit), limit)
    return clamped_gate, clamped_up


def swiglu_fwd_one(gate, up, alpha, limit, neg_limit):
    """
    One output: (up + 1) * gate * sigmoid(alpha * gate) on the clamped gate and up.
    """
    clamped_gate, clamped_up = clamp_gate_up(gate, up, limit, neg_limit)
    _, sigmoid = sigmoid_parts(clamped_gate, alpha)
    glu = emit_ptx("mul.rn.f32", clamped_gate, sigmoid)
    up_plus_one = emit_ptx("add.rn.f32", clamped_up, txl.float32(1.0))
    return emit_ptx("mul.rn.f32", up_plus_one, glu)


def swiglu_bwd_one(grad_out, gate, up, alpha, limit, neg_limit):
    """
    One output's (grad_gate, grad_up), zero where the clamp cut the input.
    """
    clamped_gate, clamped_up = clamp_gate_up(gate, up, limit, neg_limit)
    x, sigmoid = sigmoid_parts(clamped_gate, alpha)
    glu = emit_ptx("mul.rn.f32", clamped_gate, sigmoid)
    grad_up = emit_ptx("mul.rn.f32", grad_out, glu)
    up_plus_one = emit_ptx("add.rn.f32", clamped_up, txl.float32(1.0))
    grad_sigmoid = emit_ptx("mul.rn.f32", emit_ptx("mul.rn.f32", grad_out, up_plus_one), sigmoid)
    silu_grad_factor = emit_ptx("fma.rn.f32", x, emit_ptx("sub.rn.f32", txl.float32(1.0), sigmoid), txl.float32(1.0))
    grad_gate = emit_ptx("mul.rn.f32", grad_sigmoid, silu_grad_factor)
    grad_gate = txl.Select(gate <= limit, grad_gate, txl.float32(0.0))
    grad_up = txl.Select(txl.And(up >= neg_limit, up <= limit), grad_up, txl.float32(0.0))
    return grad_gate, grad_up


def pack_f32x2(value_lo, value_hi):
    """
    f32 pair (value_lo in the low lane) as one .b64 f32x2 operand.
    """
    bits_lo, bits_hi = txl.cast(txl.reinterpret(value_lo, ty="uint32"), "uint64"), txl.cast(txl.reinterpret(value_hi, ty="uint32"), "uint64")
    return (bits_hi << txl.uint64(32)) | bits_lo


def unpack_f32x2(packed):
    """
    The (low, high) f32 lanes of a .b64 f32x2 value.
    """
    return txl.reinterpret(txl.cast(packed, "uint32"), ty="float32"), txl.reinterpret(txl.cast(packed >> txl.uint64(32), "uint32"), ty="float32")


def emit_ptx_x2(instruction, *args):
    """
    emit_ptx for an f32x2 instruction, into a .b64 register.
    """
    return emit_ptx(instruction, *args, dtype="uint64")


def splat_bits_x2(bits):
    """
    The fp32 constant with bit pattern bits in both f32x2 lanes.
    """
    return pack_f32x2(f32_from_bits(bits), f32_from_bits(bits))


ONE_BITS, QUARTER_BITS = 0x3F800000, 0x3E800000


def sigmoid_parts_x2(gate_x2, alpha_x2):
    """
    sigmoid_parts on two lanes: the products in packed f32x2 (same rn rounding per lane).
    """
    x_x2 = emit_ptx_x2("mul.rn.f32x2", alpha_x2, gate_x2)
    exponent_lo, exponent_hi = unpack_f32x2(emit_ptx_x2("mul.rn.f32x2", x_x2, splat_bits_x2(NEG_LOG2E_BITS)))
    exp2_x2 = pack_f32x2(emit_ptx("ex2.approx.ftz.f32", exponent_lo), emit_ptx("ex2.approx.ftz.f32", exponent_hi))
    quarter_denom_lo, quarter_denom_hi = unpack_f32x2(emit_ptx_x2("fma.rn.f32x2", exp2_x2, splat_bits_x2(QUARTER_BITS), splat_bits_x2(QUARTER_BITS)))
    rcp_quarter_x2 = pack_f32x2(emit_ptx("rcp.approx.ftz.f32", quarter_denom_lo), emit_ptx("rcp.approx.ftz.f32", quarter_denom_hi))
    return x_x2, emit_ptx_x2("mul.rn.f32x2", rcp_quarter_x2, splat_bits_x2(QUARTER_BITS))


def swiglu_fwd_two(gates, ups, alpha, limit, neg_limit):
    """
    swiglu_fwd_one on two outputs, the products in packed f32x2.
    """
    (gate_lo, up_lo), (gate_hi, up_hi) = (clamp_gate_up(gates[k], ups[k], limit, neg_limit) for k in range(2))
    gate_x2 = pack_f32x2(gate_lo, gate_hi)
    _, sigmoid_x2 = sigmoid_parts_x2(gate_x2, pack_f32x2(alpha, alpha))
    glu_x2 = emit_ptx_x2("mul.rn.f32x2", gate_x2, sigmoid_x2)
    up_plus_one_x2 = emit_ptx_x2("add.rn.f32x2", pack_f32x2(up_lo, up_hi), splat_bits_x2(ONE_BITS))
    return unpack_f32x2(emit_ptx_x2("mul.rn.f32x2", up_plus_one_x2, glu_x2))


def swiglu_bwd_two(grad_outs, gates, ups, alpha, limit, neg_limit):
    """
    swiglu_bwd_one on two outputs, the products in packed f32x2, as [grad_gate, grad_up] for each
    output in turn.
    """
    (gate_lo, up_lo), (gate_hi, up_hi) = (clamp_gate_up(gates[k], ups[k], limit, neg_limit) for k in range(2))
    gate_x2, grad_out_x2 = pack_f32x2(gate_lo, gate_hi), pack_f32x2(grad_outs[0], grad_outs[1])
    x_x2, sigmoid_x2 = sigmoid_parts_x2(gate_x2, pack_f32x2(alpha, alpha))
    glu_x2 = emit_ptx_x2("mul.rn.f32x2", gate_x2, sigmoid_x2)
    grad_up = unpack_f32x2(emit_ptx_x2("mul.rn.f32x2", grad_out_x2, glu_x2))
    up_plus_one_x2 = emit_ptx_x2("add.rn.f32x2", pack_f32x2(up_lo, up_hi), splat_bits_x2(ONE_BITS))
    grad_sigmoid_x2 = emit_ptx_x2("mul.rn.f32x2", emit_ptx_x2("mul.rn.f32x2", grad_out_x2, up_plus_one_x2), sigmoid_x2)
    silu_grad_factor_x2 = emit_ptx_x2("fma.rn.f32x2", x_x2, emit_ptx_x2("sub.rn.f32x2", splat_bits_x2(ONE_BITS), sigmoid_x2), splat_bits_x2(ONE_BITS))
    grad_gate = unpack_f32x2(emit_ptx_x2("mul.rn.f32x2", grad_sigmoid_x2, silu_grad_factor_x2))
    grads = []
    for k in range(2):
        masked_grad_gate, masked_grad_up = txl.Select(gates[k] <= limit, grad_gate[k], txl.float32(0.0)), txl.Select(txl.And(ups[k] >= neg_limit, ups[k] <= limit), grad_up[k], txl.float32(0.0))
        grads += [masked_grad_gate, masked_grad_up]
    return grads


class StorageIO:
    """
    Load/convert helpers for one storage dtype, in 32-bit words.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        self.elems_per_word = 1 if dtype == "float32" else 2
        self.elem_bytes = 4 // self.elems_per_word
        self.cvt_to_f32, self.cvt_from_f32, self.cvt_from_f32x2 = {"bfloat16": "cvt.f32.bf16", "float16": "cvt.f32.f16"}.get(dtype), {"bfloat16": "cvt.rn.bf16.f32", "float16": "cvt.rn.f16.f32"}.get(dtype), {"bfloat16": "cvt.rn.bf16x2.f32", "float16": "cvt.rn.f16x2.f32"}.get(dtype)

    def unpack_words(self, words):
        values = []
        for word in words:
            if self.dtype == "float32":
                values.append(txl.reinterpret(word, ty="float32"))
            elif self.dtype == "bfloat16":  # bf16 -> f32 is exact: the bits move to the high half
                values.append(txl.reinterpret(word << txl.uint32(16), ty="float32"))
                values.append(txl.reinterpret(word & txl.uint32(0xFFFF0000), ty="float32"))
            else:
                values.append(emit_ptx(self.cvt_to_f32, txl.cast(word & txl.uint32(0xFFFF), "uint16")))
                values.append(emit_ptx(self.cvt_to_f32, txl.cast(word >> txl.uint32(16), "uint16")))
        return values

    def pack_words(self, values):
        if self.dtype == "float32":
            return [txl.reinterpret(value, ty="uint32") for value in values]
        return [emit_ptx(self.cvt_from_f32x2, values[2 * i + 1], values[2 * i], dtype="uint32") for i in range(len(values) // 2)]

    def load_scalar(self, tensor, elem_idx):
        if self.dtype == "float32":
            value = txl.local_scalar("float32")
            txl.ptx.ld.global_.f32(value, tensor.ptr_to([elem_idx]))
            return value
        value_bits = txl.local_scalar("uint16")
        txl.ptx.ld.global_.b16(value_bits, tensor.ptr_to([elem_idx]))
        return emit_ptx(self.cvt_to_f32, value_bits)

    def store_scalar(self, tensor, elem_idx, value):
        if self.dtype == "float32":
            txl.ptx.st.global_.f32(tensor.ptr_to([elem_idx]), value)
        else:
            txl.ptx.st.global_.b16(tensor.ptr_to([elem_idx]), emit_ptx(self.cvt_from_f32, value, dtype="uint16"))

    def load_words(self, tensor, elem_idx, num_words, instruction):
        words = txl.alloc_local([num_words], "uint32")
        is_v8 = ".v8." in instruction
        words_per_access = 8 if is_v8 and num_words % 8 == 0 else 4
        if words_per_access == 4:
            instruction = instruction.replace(".v8.", ".v4.")
        for word_idx in range(0, num_words, words_per_access):
            txl.ptx[instruction](*[words[word_idx + j] for j in range(words_per_access)], tensor.ptr_to([elem_idx + word_idx * 4 // self.elem_bytes]))
        return [words[word_idx] for word_idx in range(num_words)]

    def store_words(self, tensor, elem_idx, words, instruction):
        words_per_access = 8 if ".v8." in instruction and len(words) % 8 == 0 else 4
        if words_per_access == 4:
            instruction = instruction.replace(".v8.", ".v4.")
        for word_idx in range(0, len(words), words_per_access):
            txl.ptx[instruction](tensor.ptr_to([elem_idx + word_idx * 4 // self.elem_bytes]), *words[word_idx : word_idx + words_per_access])


def compute_grid(THREADS, VEC, UNROLL, num_outputs):
    """
    CTAs covering num_outputs outputs, VEC * THREADS * UNROLL per CTA.
    """
    TILE = VEC * THREADS * UNROLL
    return txl.cast((num_outputs + TILE - 1) // TILE, "int32")


@functools.cache
def compile_fwd_kernel(dtype, THREADS, VEC, UNROLL):
    storage, config = StorageIO(dtype), LAUNCH_CONFIG["fwd"]
    TILE = VEC * THREADS

    @txl.kernel()
    def clamped_swiglu_fwd(gate_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), out: txl.Tensor((txl.int64(),), dtype, data_alignment=1), num_outputs: txl.int64, alpha: txl.float32, limit: txl.float32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=compute_grid(THREADS, VEC, UNROLL, num_outputs), block=(THREADS // 32) * 32))
        neg_limit = emit_ptx("neg.f32", limit)
        thread_start = txl.cast(txl.cta_id(), "int64") * (TILE * UNROLL) + txl.cast(txl.thread_id(), "int64") * VEC
        if VEC == 1:  # unaligned fallback: one output per thread, scalar accesses only
            with txl.if_(thread_start < num_outputs), txl.then_():
                storage.store_scalar(out, thread_start, swiglu_fwd_one(storage.load_scalar(gate_up, 2 * thread_start), storage.load_scalar(gate_up, 2 * thread_start + 1), alpha, limit, neg_limit))
            return
        with txl.if_(thread_start + (TILE * (UNROLL - 1) + VEC) <= num_outputs):
            with txl.then_():
                num_words = 2 * VEC // storage.elems_per_word
                gate_up_words = [storage.load_words(gate_up, (thread_start + unroll_idx * TILE) * 2, num_words, config["ld"]) for unroll_idx in range(UNROLL)]
                for unroll_idx in range(UNROLL):
                    values = storage.unpack_words(gate_up_words[unroll_idx])
                    outs = []
                    for out_idx in range(0, VEC, 2):
                        outs += list(swiglu_fwd_two(values[2 * out_idx : 2 * out_idx + 4 : 2], values[2 * out_idx + 1 : 2 * out_idx + 4 : 2], alpha, limit, neg_limit))
                    storage.store_words(out, thread_start + unroll_idx * TILE, storage.pack_words(outs), config["st"])
            with txl.else_():
                with txl.serial(UNROLL * VEC) as step_idx:
                    elem_idx = thread_start + txl.cast(step_idx // VEC, "int64") * TILE + txl.cast(step_idx % VEC, "int64")
                    with txl.if_(elem_idx < num_outputs), txl.then_():
                        out_value = swiglu_fwd_one(storage.load_scalar(gate_up, 2 * elem_idx), storage.load_scalar(gate_up, 2 * elem_idx + 1), alpha, limit, neg_limit)
                        storage.store_scalar(out, elem_idx, out_value)

    return entry_function(clamped_swiglu_fwd.compile(TARGET))


@functools.cache
def compile_bwd_kernel(dtype, THREADS, VEC, UNROLL):
    storage, config = StorageIO(dtype), LAUNCH_CONFIG["bwd"]
    TILE = VEC * THREADS

    @txl.kernel()
    def clamped_swiglu_bwd(grad_out: txl.Tensor((txl.int64(),), dtype, data_alignment=1), gate_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), grad_gate_up: txl.Tensor((txl.int64(),), dtype, data_alignment=1), num_outputs: txl.int64, alpha: txl.float32, limit: txl.float32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=compute_grid(THREADS, VEC, UNROLL, num_outputs), block=(THREADS // 32) * 32))
        neg_limit = emit_ptx("neg.f32", limit)
        thread_start = txl.cast(txl.cta_id(), "int64") * (TILE * UNROLL) + txl.cast(txl.thread_id(), "int64") * VEC
        if VEC == 1:  # unaligned fallback: one output per thread, scalar accesses only
            with txl.if_(thread_start < num_outputs), txl.then_():
                grad_gate, grad_up = swiglu_bwd_one(storage.load_scalar(grad_out, thread_start), storage.load_scalar(gate_up, 2 * thread_start), storage.load_scalar(gate_up, 2 * thread_start + 1), alpha, limit, neg_limit)
                storage.store_scalar(grad_gate_up, 2 * thread_start, grad_gate)
                storage.store_scalar(grad_gate_up, 2 * thread_start + 1, grad_up)
            return
        with txl.if_(thread_start + (TILE * (UNROLL - 1) + VEC) <= num_outputs):
            with txl.then_():
                num_words = VEC // storage.elems_per_word
                gate_up_words = [storage.load_words(gate_up, (thread_start + unroll_idx * TILE) * 2, 2 * num_words, config["ld"]) for unroll_idx in range(UNROLL)]
                grad_out_words = [storage.load_words(grad_out, thread_start + unroll_idx * TILE, num_words, config["ld"]) for unroll_idx in range(UNROLL)]
                for unroll_idx in range(UNROLL):
                    values, grad_outs = storage.unpack_words(gate_up_words[unroll_idx]), storage.unpack_words(grad_out_words[unroll_idx])
                    grads = []
                    for out_idx in range(0, VEC, 2):
                        grads += swiglu_bwd_two(grad_outs[out_idx : out_idx + 2], values[2 * out_idx : 2 * out_idx + 4 : 2], values[2 * out_idx + 1 : 2 * out_idx + 4 : 2], alpha, limit, neg_limit)
                    storage.store_words(grad_gate_up, (thread_start + unroll_idx * TILE) * 2, storage.pack_words(grads), config["st"])
            with txl.else_():
                with txl.serial(UNROLL * VEC) as step_idx:
                    elem_idx = thread_start + txl.cast(step_idx // VEC, "int64") * TILE + txl.cast(step_idx % VEC, "int64")
                    with txl.if_(elem_idx < num_outputs), txl.then_():
                        grad_gate, grad_up = swiglu_bwd_one(storage.load_scalar(grad_out, elem_idx), storage.load_scalar(gate_up, 2 * elem_idx), storage.load_scalar(gate_up, 2 * elem_idx + 1), alpha, limit, neg_limit)
                        storage.store_scalar(grad_gate_up, 2 * elem_idx, grad_gate)
                        storage.store_scalar(grad_gate_up, 2 * elem_idx + 1, grad_up)

    return entry_function(clamped_swiglu_bwd.compile(TARGET))


def pick_launch_config(entry, *tensors):
    """
    The vector paths need every base pointer 32-byte aligned (and whole 4-word vectors per thread);
    otherwise one output per thread with scalar accesses.
    """
    config = LAUNCH_CONFIG[entry]
    if all(tensor.data_ptr() % 32 == 0 for tensor in tensors):
        return config["threads"], config["vec"], config["unroll"]
    return 256, 1, 1


def clamped_swiglu_fwd_sm100(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Clamped SwiGLU of the contiguous interleaved (M, 2N) gate_up as a new contiguous (M, N) tensor
    in its dtype, the bits of clamped_swiglu_fwd_sm90.
    """
    assert gate_up.dim() == 2, f"gate_up must be 2-D, got shape {gate_up.shape}"
    assert gate_up.shape[1] % 2 == 0, f"gate_up last dim must be even, got {gate_up.shape[1]}"
    assert gate_up.is_contiguous(), "gate_up must be contiguous"
    if gate_up.dtype not in KERNEL_DTYPE:  # other float dtypes: the fp32 kernel, rounded once back
        return clamped_swiglu_fwd_sm100(gate_up.float(), alpha, limit).to(gate_up.dtype)
    M, gate_up_cols = gate_up.shape
    out = torch.empty((M, gate_up_cols // 2), device=gate_up.device, dtype=gate_up.dtype)
    num_outputs = out.numel()
    if num_outputs == 0:
        return out
    fn = compile_fwd_kernel(KERNEL_DTYPE[gate_up.dtype], *pick_launch_config("fwd", gate_up, out))
    fn(gate_up.view(-1), out.view(-1), num_outputs, float(alpha), float(limit))
    return out


def clamped_swiglu_bwd_sm100(grad_output: torch.Tensor, gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Gradient of clamped_swiglu_fwd_sm100 with respect to the interleaved gate_up as a new (M, 2N)
    tensor like gate_up, the bits of clamped_swiglu_bwd_sm90.
    """
    assert grad_output.shape[0] == gate_up.shape[0]
    assert grad_output.shape[1] * 2 == gate_up.shape[1]
    assert grad_output.dtype == gate_up.dtype
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    if gate_up.dtype not in KERNEL_DTYPE:  # other float dtypes: the fp32 kernel, rounded once back
        return clamped_swiglu_bwd_sm100(grad_output.float(), gate_up.float(), alpha, limit).to(gate_up.dtype)
    gate_up = gate_up.contiguous()
    grad_gate_up = torch.empty_like(gate_up)
    num_outputs = grad_output.numel()
    if num_outputs == 0:
        return grad_gate_up
    fn = compile_bwd_kernel(KERNEL_DTYPE[gate_up.dtype], *pick_launch_config("bwd", grad_output, gate_up, grad_gate_up))
    fn(grad_output.view(-1), gate_up.view(-1), grad_gate_up.view(-1), num_outputs, float(alpha), float(limit))
    return grad_gate_up
