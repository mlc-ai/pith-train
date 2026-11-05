"""
group_bias_add_sm100: in place y[r] += bias[group_indices[r]] on SM100, TIRx-lite kernels.

Row tile (the common path): one CTA per 4 consecutive rows, each thread moving one W-word vector
(W = 8 -> 32 B, sm_100's 256-bit ld/st) of every row, up to two vectors per row. All four rows' y
loads are issued up front; when the four rows share a group one bias vector serves them all,
otherwise each row loads its own. Each row's add is retired directly into its own store. Half
precision adds use the native packed add.rn.bf16x2 / add.rn.f16x2 (both operands are representable
in y's dtype, so this is the same single round-to-nearest-even as an fp32 add plus cvt.rn); fp32
uses add.rn.f32. The launch is bounded to 5 CTAs per SM (64 registers). The CTAs covering the newest
48 MiB of y run first, walking backwards, reusing the lines its producer left L2-resident.

Flat fallback (M not a multiple of 4, rows wider than two vectors per thread, int32 offset limits):
a flat W-word vector index space, two vectors per thread, also tail-first; rows whose byte length is
not a multiple of 4 take a 16-bit scalar kernel. Offsets are int32 (launches are chunked so every
chunk has < 2^30 elements). Plans are cached per (dtype, index dtype, H, pointer alignment).
"""

import functools
import math

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["group_bias_add_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


BLOCK = 64
UNROLL = 2
MAX_WORDS = 8  # widest vector in 32-bit words (8 -> 256-bit ld/st)
TAIL_BYTES = 96 << 20
TILE_ROWS = 4
TILE_MAX_CHUNKS = 2
TILE_TAIL_BYTES = 48 << 20
ELEMENT_BYTES = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}
DTYPE_NAME = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
INDEX_DTYPE_NAME = {torch.int32: txl.int32, torch.int64: "int64"}
VECTOR_SUFFIX = {1: ".b32", 2: ".v2.b32", 4: ".v4.b32", 8: ".v8.b32"}


def add_words(dtype, out, y_words, bias_words, num_words):
    """
    out[0:num_words] = round(y + bias) for 32-bit words of packed y / bias elements (one rounding).
    """
    if dtype == "float32":
        for i in range(num_words):
            sum_f32 = txl.local_scalar("float32")
            txl.ptx["add.rn.f32"](sum_f32, txl.reinterpret(y_words[i], ty="float32"), txl.reinterpret(bias_words[i], ty="float32"))
            txl.assign(out[i], txl.reinterpret(sum_f32, ty="uint32"))
        return
    add_op = "add.rn.bf16x2" if dtype == "bfloat16" else "add.rn.f16x2"
    for i in range(num_words):
        # both inputs are representable in the output format, so the native packed add performs the
        # same one round-to-nearest-even as an fp32 add followed by the destination conversion
        txl.ptx[add_op](out[i], y_words[i], bias_words[i])


class Slice:
    """
    A view buf[offset:] of a local array, indexable like the array.
    """

    def __init__(self, buf, offset):
        self.buf, self.offset = buf, offset

    def __getitem__(self, i):
        return self.buf[self.offset + i]


@functools.cache
def build_flat_kernel(dtype: str, index_dtype: str, H: int, W: int):
    """
    Flat fallback: W words per vector (1, 2, 4 or 8); H * element bytes must be a multiple of 4 * W.
    """
    elem_bytes = 4 if dtype == "float32" else 2
    row_words = H * elem_bytes // 4
    vecs_per_row = row_words // W
    elems_per_vec = W * 4 // elem_bytes
    ld_y, ld_bias, st_y = "ld.global" + VECTOR_SUFFIX[W], "ld.global.nc" + VECTOR_SUFFIX[W], "st.global" + VECTOR_SUFFIX[W]
    ld_index = "ld.global.nc.s64" if index_dtype == "int64" else "ld.global.nc.s32"
    vecs_per_cta = BLOCK * UNROLL

    var_M, var_G = txl.Var("M", "int32"), txl.Var("G", "int32")

    @txl.kernel()
    def group_bias_add_flat(y: txl.Tensor((var_M, H), dtype, data_alignment=1), bias: txl.Tensor((var_G, H), dtype, data_alignment=1), group_indices: txl.Tensor((var_M,), index_dtype, data_alignment=1), M: var_M, G: var_G, num_tail_ctas: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=txl.ceildiv(M * vecs_per_row, vecs_per_cta), block=(BLOCK // 32) * 32))
        num_vecs = M * vecs_per_row
        tidx = txl.thread_id()
        num_ctas = (num_vecs + vecs_per_cta - 1) // vecs_per_cta
        launch_idx = txl.cta_id()
        # the last num_tail_ctas CTAs (the newest, L2-resident part of y) run first, walking back
        cta_idx = txl.Select(launch_idx < num_tail_ctas, num_ctas - 1 - launch_idx, launch_idx - num_tail_ctas)
        vec_start = cta_idx * vecs_per_cta + tidx
        y_words, bias_words, group_idx = txl.alloc_local([UNROLL * W], "uint32"), txl.alloc_local([UNROLL * W], "uint32"), txl.alloc_local([UNROLL], "int32")

        for unroll_idx in range(UNROLL):
            vec_idx = vec_start + unroll_idx * BLOCK
            with txl.if_(vec_idx < num_vecs), txl.then_():
                row = vec_idx // vecs_per_row
                index_value = txl.local_scalar(index_dtype)
                txl.ptx[ld_index](index_value, group_indices.ptr_to([row]))
                txl.assign(group_idx[unroll_idx], txl.cast(index_value, "int32"))
                txl.ptx[ld_y](*[y_words[unroll_idx * W + k] for k in range(W)], y.ptr_to([row, (vec_idx - row * vecs_per_row) * elems_per_vec]))

        for unroll_idx in range(UNROLL):
            vec_idx = vec_start + unroll_idx * BLOCK
            with txl.if_(vec_idx < num_vecs), txl.then_():
                row = vec_idx // vecs_per_row
                col_vec = vec_idx - row * vecs_per_row
                txl.ptx[ld_bias](*[bias_words[unroll_idx * W + k] for k in range(W)], bias.ptr_to([group_idx[unroll_idx], col_vec * elems_per_vec]))

        for unroll_idx in range(UNROLL):
            vec_idx = vec_start + unroll_idx * BLOCK
            with txl.if_(vec_idx < num_vecs), txl.then_():
                row = vec_idx // vecs_per_row
                add_words(dtype, Slice(y_words, unroll_idx * W), Slice(y_words, unroll_idx * W), Slice(bias_words, unroll_idx * W), W)
                txl.ptx[st_y](y.ptr_to([row, (vec_idx - row * vecs_per_row) * elems_per_vec]), *[y_words[unroll_idx * W + k] for k in range(W)])

    return entry_function(group_bias_add_flat.compile(TARGET))


@functools.cache
def build_b16_kernel(dtype: str, index_dtype: str, H: int):
    """
    16-bit fallback for half-precision rows whose byte length is not a multiple of 4.
    """
    cvt_out = "cvt.rn.bf16.f32" if dtype == "bfloat16" else "cvt.rn.f16.f32"
    ld_index = "ld.global.nc.s64" if index_dtype == "int64" else "ld.global.nc.s32"

    var_M, var_G = txl.Var("M", "int32"), txl.Var("G", "int32")

    @txl.kernel()
    def group_bias_add_b16(y: txl.Tensor((var_M, H), dtype, data_alignment=1), bias: txl.Tensor((var_G, H), dtype, data_alignment=1), group_indices: txl.Tensor((var_M,), index_dtype, data_alignment=1), M: var_M, G: var_G):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=txl.ceildiv(M * H, BLOCK), block=(BLOCK // 32) * 32))
        elem_idx = txl.cta_id() * BLOCK + txl.thread_id()
        with txl.if_(elem_idx < M * H), txl.then_():
            row = elem_idx // H
            col = elem_idx - row * H
            group_idx = txl.local_scalar(index_dtype)
            y_bits, bias_bits = txl.local_scalar("uint16"), txl.local_scalar("uint16")
            txl.ptx[ld_index](group_idx, group_indices.ptr_to([row]))
            txl.ptx["ld.global.b16"](y_bits, y.ptr_to([row, col]))
            txl.ptx["ld.global.nc.b16"](bias_bits, bias.ptr_to([txl.cast(group_idx, "int32"), col]))
            y_f32, bias_f32 = txl.local_scalar("float32"), txl.local_scalar("float32")
            if dtype == "bfloat16":
                txl.assign(y_f32, txl.reinterpret(txl.cast(y_bits, "uint32") << txl.uint32(16), ty="float32"))
                txl.assign(bias_f32, txl.reinterpret(txl.cast(bias_bits, "uint32") << txl.uint32(16), ty="float32"))
            else:
                txl.ptx["cvt.f32.f16"](y_f32, y_bits)
                txl.ptx["cvt.f32.f16"](bias_f32, bias_bits)
            sum_f32, out = txl.local_scalar("float32"), txl.local_scalar("uint16")
            txl.ptx["add.rn.f32"](sum_f32, y_f32, bias_f32)
            txl.ptx[cvt_out](out, sum_f32)
            txl.ptx["st.global.b16"](y.ptr_to([row, col]), out)

    return entry_function(group_bias_add_b16.compile(TARGET))


def row_tile_shape(H, W, elem_bytes):
    """
    Row tile geometry: (vectors per row, threads per CTA, vectors per thread per row).
    """
    vecs_per_row = H * elem_bytes // (4 * W)
    num_threads = min(512, -(-vecs_per_row // 32) * 32)
    return vecs_per_row, num_threads, -(-vecs_per_row // num_threads)


@functools.cache
def build_row_tile_kernel(dtype: str, index_dtype: str, H: int, W: int):
    """
    Row tile kernel: one CTA per TILE_ROWS rows; M must be a multiple of TILE_ROWS.
    """
    elem_bytes = 4 if dtype == "float32" else 2
    vecs_per_row, num_threads, num_chunks = row_tile_shape(H, W, elem_bytes)
    elems_per_vec = W * 4 // elem_bytes
    ROWS = TILE_ROWS
    ld_y, ld_bias, st_y = "ld.global" + VECTOR_SUFFIX[W], "ld.global.nc" + VECTOR_SUFFIX[W], "st.global" + VECTOR_SUFFIX[W]
    ld_index = "ld.global.nc.s64" if index_dtype == "int64" else "ld.global.nc.s32"

    var_M, var_G = txl.Var("M", "int32"), txl.Var("G", "int32")

    @txl.kernel()
    def group_bias_add_row_tile(y: txl.Tensor((var_M, H), dtype, data_alignment=1), bias: txl.Tensor((var_G, H), dtype, data_alignment=1), group_indices: txl.Tensor((var_M,), index_dtype, data_alignment=1), M: var_M, G: var_G, num_tail_tiles: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=M // ROWS, block=(num_threads // 32) * 32), kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=5))
        launch_idx = txl.cta_id()
        num_tiles = M // ROWS
        # the last num_tail_tiles tiles (the newest, L2-resident part of y) run first, walking back
        row0 = txl.Select(launch_idx < num_tail_tiles, num_tiles - 1 - launch_idx, launch_idx - num_tail_tiles) * ROWS
        tidx = txl.thread_id()
        group_idx = txl.alloc_local([ROWS], "int32")
        for row in range(ROWS):
            index_value = txl.local_scalar(index_dtype)
            txl.ptx[ld_index](index_value, group_indices.ptr_to([row0 + row]))
            txl.assign(group_idx[row], txl.cast(index_value, "int32"))

        y_words, bias_words = txl.alloc_local([num_chunks * ROWS * W], "uint32"), txl.alloc_local([ROWS * W], "uint32")
        for chunk in range(num_chunks):
            vec_idx = tidx + chunk * num_threads
            with txl.if_(vec_idx < vecs_per_row), txl.then_():
                for row in range(ROWS):
                    word_start = ((chunk * ROWS) + row) * W
                    txl.ptx[ld_y](*[y_words[word_start + k] for k in range(W)], y.ptr_to([row0 + row, vec_idx * elems_per_vec]))

        is_same_group = group_idx[0] == group_idx[ROWS - 1]
        for row in range(1, ROWS - 1):
            is_same_group = is_same_group & (group_idx[0] == group_idx[row])

        for chunk in range(num_chunks):
            vec_idx = tidx + chunk * num_threads
            with txl.if_(vec_idx < vecs_per_row), txl.then_():
                with txl.if_(is_same_group):
                    with txl.then_():
                        txl.ptx[ld_bias](*[bias_words[k] for k in range(W)], bias.ptr_to([group_idx[0], vec_idx * elems_per_vec]))
                        for row in range(ROWS):
                            word_start = ((chunk * ROWS) + row) * W
                            add_words(dtype, Slice(y_words, word_start), Slice(y_words, word_start), bias_words, W)
                            txl.ptx[st_y](y.ptr_to([row0 + row, vec_idx * elems_per_vec]), *[y_words[word_start + k] for k in range(W)])
                    with txl.else_():
                        for row in range(ROWS):
                            word_start = ((chunk * ROWS) + row) * W
                            txl.ptx[ld_bias](*[bias_words[k] for k in range(W)], bias.ptr_to([group_idx[row], vec_idx * elems_per_vec]))
                            add_words(dtype, Slice(y_words, word_start), Slice(y_words, word_start), bias_words, W)
                            txl.ptx[st_y](y.ptr_to([row0 + row, vec_idx * elems_per_vec]), *[y_words[word_start + k] for k in range(W)])

    return entry_function(group_bias_add_row_tile.compile(TARGET))


MAX_ELEMS = 1 << 30  # per launch: element offsets (and G * H) stay well inside int32


def launch_flat(y, bias, group_indices, M):
    """
    Flat fallback path for any covered input (bias.numel() < MAX_ELEMS, checked by the caller).
    """
    H = y.shape[1]
    dtype, index_dtype = DTYPE_NAME[y.dtype], INDEX_DTYPE_NAME[group_indices.dtype]
    row_bytes = H * ELEMENT_BYTES[y.dtype]
    rows_per_launch = max(1, MAX_ELEMS // H)
    if row_bytes % 4 == 0:
        align_bytes = math.gcd(4 * MAX_WORDS, row_bytes, y.data_ptr() % 32 or 32, bias.data_ptr() % 32 or 32)
        if align_bytes >= 4:
            W = align_bytes // 4
            fn = build_flat_kernel(dtype, index_dtype, H, W)
            vecs_per_row = row_bytes // (4 * W)
            for row_start in range(0, M, rows_per_launch):
                row_end = min(M, row_start + rows_per_launch)
                num_ctas = -(-((row_end - row_start) * vecs_per_row) // (BLOCK * UNROLL))
                num_tail_ctas = min(num_ctas, TAIL_BYTES // (BLOCK * UNROLL * W * 4))
                fn(y[row_start:row_end], bias, group_indices[row_start:row_end], row_end - row_start, bias.shape[0], num_tail_ctas)
            return
    fn = build_b16_kernel(dtype, index_dtype, H)
    for row_start in range(0, M, rows_per_launch):
        row_end = min(M, row_start + rows_per_launch)
        fn(y[row_start:row_end], bias, group_indices[row_start:row_end], row_end - row_start, bias.shape[0])


def make_plan(torch_dtype, torch_index_dtype, H, ptr_low_bits):
    """
    The launcher for one (dtype, index dtype, H, pointer alignment): the row tile where it applies.
    """
    elem_bytes = ELEMENT_BYTES[torch_dtype]
    row_bytes = H * elem_bytes
    if row_bytes % 4:
        return launch_flat
    W = math.gcd(4 * MAX_WORDS, row_bytes, ptr_low_bits or 32) // 4
    if W == 0 or row_tile_shape(H, W, elem_bytes)[2] > TILE_MAX_CHUNKS:
        return launch_flat
    fn = build_row_tile_kernel(DTYPE_NAME[torch_dtype], INDEX_DTYPE_NAME[torch_index_dtype], H, W)
    num_tail_tiles = max(1, TILE_TAIL_BYTES // (TILE_ROWS * row_bytes))
    max_rows = (MAX_ELEMS - 1) // H

    def launch_row_tile(y, bias, group_indices, M):
        G = bias.shape[0]
        if M % TILE_ROWS or M > max_rows or G > max_rows:
            return launch_flat(y, bias, group_indices, M)
        fn(y, bias, group_indices, M, G, min(M // TILE_ROWS, num_tail_tiles))

    return launch_row_tile


PLANS = {}


def group_bias_add_sm100(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> None:
    """
    In place y[r] += bias[group_indices[r]], bit-identical to y.copy_(y + bias[group_indices]).
    """
    M, H = y.shape
    if M == 0 or H == 0:
        return
    # inputs the kernels do not cover (mixed dtypes, strided views, int32 offset overflow) take the
    # PyTorch fallback the result is bit-identical to, as the SM90 kernel does
    covered = y.dtype == bias.dtype and y.dtype in DTYPE_NAME and group_indices.dtype in INDEX_DTYPE_NAME
    if not covered or not (y.is_contiguous() and bias.is_contiguous() and group_indices.is_contiguous()) or bias.numel() >= MAX_ELEMS:
        y.copy_(y + bias[group_indices])
        return
    key = (y.dtype, group_indices.dtype, H, (y.data_ptr() | bias.data_ptr()) & 31)
    plan = PLANS.get(key)
    if plan is None:
        plan = PLANS[key] = make_plan(*key)
    plan(y, bias, group_indices, M)
