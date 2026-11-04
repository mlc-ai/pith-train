"""
SM90 group_bias_add: a CuTe-DSL kernel over the 16-byte vectors of y for the inputs kernel_covers()
accepts, the PyTorch expression for the rest.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, bf16x2_to_f32, f32_as_u32, f32x2_to_bf16x2, ld_b32, ld_v4, st_v4, u32_as_f32

__all__ = ["group_bias_add_sm90"]

THREADS, VECS_PER_THREAD, VEC_BYTES = 256, 2, 16
# Vector indices stay Int32: the last CTA's indices run up to THREADS * VECS_PER_THREAD past the total.
MAX_VECTORS = 2**31 - THREADS * VECS_PER_THREAD
VECTOR_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
INDEX2CUTE = {torch.int32: Int32, torch.int64: Int64}


@dsl_user_op
def f16x2_to_f32(x: Uint32, *, loc=None, ip=None) -> tuple:
    x_lo = asm(T.f32(), [Uint32(x).ir_value(loc=loc, ip=ip)], "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $1;\n\tcvt.f32.f16 $0, lo;\n\t}", "=f,r")
    x_hi = asm(T.f32(), [Uint32(x).ir_value(loc=loc, ip=ip)], "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $1;\n\tcvt.f32.f16 $0, hi;\n\t}", "=f,r")
    return cutlass.Float32(x_lo), cutlass.Float32(x_hi)


class GroupBiasAdd:
    """
    Each thread handles VECS_PER_THREAD vectors spaced THREADS apart, so a warp touches 512
    contiguous bytes per step; the bias rows (G of them) stay in L2.
    """

    def __init__(self, H: int, dtype: type, index_dtype: type) -> None:
        self.dtype, self.index_dtype = dtype, index_dtype
        self.vecs_per_row = H * dtype.width // 8 // VEC_BYTES

    @cute.jit
    def __call__(self, mY: cute.Tensor, mBias: cute.Tensor, mIdx: cute.Tensor, stream) -> None:
        vecs_per_cta = THREADS * VECS_PER_THREAD
        num_ctas = (mIdx.shape[0] * self.vecs_per_row + vecs_per_cta - 1) // vecs_per_cta
        self.kernel(mY, mBias, mIdx).launch(grid=[num_ctas, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mY: cute.Tensor, mBias: cute.Tensor, mIdx: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        y_base, bias_base, idx_base = mY.iterator.toint(), mBias.iterator.toint(), mIdx.iterator.toint()
        vec_idx, vec_ok = self.vector_coords(bx, tidx, Int32(mIdx.shape[0]) * self.vecs_per_row)
        group_idx = self.load_group_idx(idx_base, vec_idx, vec_ok)
        y_words = self.load_rows(y_base, vec_idx, vec_ok)
        bias_words = self.load_bias(bias_base, group_idx, vec_idx, vec_ok)
        out_words = self.add_bias(y_words, bias_words)
        self.store_rows(y_base, out_words, vec_idx, vec_ok)

    def vector_coords(self, bx, tidx, num_vecs) -> tuple:
        """
        Indices of the thread's vectors in the flattened y, and whether each is inside it.
        """
        vec_idx = [(bx * VECS_PER_THREAD + i) * THREADS + tidx for i in range(VECS_PER_THREAD)]
        return vec_idx, [Int32(vec < num_vecs) for vec in vec_idx]

    def load_group_idx(self, idx_base, vec_idx, vec_ok) -> list:
        group_idx = []
        for i in range(VECS_PER_THREAD):
            row_addr = idx_base + Int64(vec_idx[i] // self.vecs_per_row) * (self.index_dtype.width // 8)
            if cutlass.const_expr(self.index_dtype is Int64):
                # ld.global.nc.b64: the full id, so the bias row offset is computed in 64 bits.
                group_idx.append(Int64(asm(T.i64(), [Int64(row_addr).ir_value(), Int32(vec_ok[i]).ir_value()], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b64 $0, 0;\n\t@p ld.global.nc.b64 $0, [$1];\n\t}", "=l,l,r")))
            else:
                group_idx.append(Int64(ld_b32(row_addr, vec_ok[i], Int32(0))))
        return group_idx

    def load_rows(self, y_base, vec_idx, vec_ok) -> list:
        return [ld_v4(y_base + Int64(vec_idx[i]) * VEC_BYTES, vec_ok[i]) for i in range(VECS_PER_THREAD)]

    def load_bias(self, bias_base, group_idx, vec_idx, vec_ok) -> list:
        """
        The bias vector at the same column of each vector's group row.
        """
        bias_words = []
        for i in range(VECS_PER_THREAD):
            col_idx = vec_idx[i] % self.vecs_per_row
            bias_words.append(ld_v4(bias_base + (group_idx[i] * self.vecs_per_row + Int64(col_idx)) * VEC_BYTES, vec_ok[i]))
        return bias_words

    def add_bias(self, y_words, bias_words) -> list:
        return [tuple(self.add_word(y_words[i][j], bias_words[i][j]) for j in range(4)) for i in range(VECS_PER_THREAD)]

    def add_word(self, y_word, bias_word) -> Uint32:
        """
        Sum of the elements packed in two 32-bit words, in fp32, rounded once (cvt.rn) to the dtype.
        """
        if cutlass.const_expr(self.dtype is cutlass.Float32):
            return f32_as_u32(u32_as_f32(y_word) + u32_as_f32(bias_word))
        if cutlass.const_expr(self.dtype is cutlass.BFloat16):
            (y_lo, y_hi), (bias_lo, bias_hi) = bf16x2_to_f32(y_word), bf16x2_to_f32(bias_word)
            return f32x2_to_bf16x2(y_lo + bias_lo, y_hi + bias_hi)
        (y_lo, y_hi), (bias_lo, bias_hi) = f16x2_to_f32(y_word), f16x2_to_f32(bias_word)
        return Uint32(asm(T.i32(), [cutlass.Float32(y_hi + bias_hi).ir_value(), cutlass.Float32(y_lo + bias_lo).ir_value()], "cvt.rn.f16x2.f32 $0, $1, $2;", "=r,f,f"))

    def store_rows(self, y_base, out_words, vec_idx, vec_ok) -> None:
        for i in range(VECS_PER_THREAD):
            st_v4(y_base + Int64(vec_idx[i]) * VEC_BYTES, out_words[i], vec_ok[i])


COMPILED: dict[tuple, Callable] = {}


def compile_kernel(H: int, dtype: type, index_dtype: type) -> Callable:
    m_sym, g_sym = cute.sym_int(), cute.sym_int()
    index_bytes = index_dtype.width // 8
    y, bias = fake(dtype, (m_sym, H), (H, 1), VEC_BYTES), fake(dtype, (g_sym, H), (H, 1), VEC_BYTES)
    return cute.compile(GroupBiasAdd(H, dtype, index_dtype), y, bias, fake(index_dtype, (m_sym,), (1,), index_bytes), fake_stream(), options="--enable-tvm-ffi")


def kernel_covers(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> bool:
    """
    Whether the kernel takes these inputs: same-dtype 16-byte vectors of y and bias, every row
    starting on a 16-byte boundary, int32 / int64 ids, contiguous, one CUDA device, Int32 vector
    indices, and not while torch.compile traces (checked first, before any data_ptr()).
    """
    M, H = y.shape
    row_bytes = H * y.element_size()
    return (not torch.compiler.is_compiling() and y.dtype == bias.dtype and y.dtype in VECTOR_DTYPES and group_indices.dtype in INDEX2CUTE and bias.ndim == 2 and bias.shape[1] == H and group_indices.shape == (M,) and y.is_cuda and y.device == bias.device == group_indices.device and y.is_contiguous() and bias.is_contiguous() and group_indices.is_contiguous() and row_bytes % VEC_BYTES == 0 and y.data_ptr() % VEC_BYTES == 0 and bias.data_ptr() % VEC_BYTES == 0 and M * (row_bytes // VEC_BYTES) <= MAX_VECTORS)


def group_bias_add_sm90(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> None:
    """
    SM90 group_bias_add: the kernel where kernel_covers(), the PyTorch expression elsewhere.
    """
    M, H = y.shape
    if M == 0 or H == 0:
        return
    if not kernel_covers(y, bias, group_indices):
        y.copy_(y + bias[group_indices])
        return
    key = ("group_bias_add", H, y.dtype, group_indices.dtype)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_kernel(H, TORCH2CUTE[y.dtype], INDEX2CUTE[group_indices.dtype])
    fn(y, bias, group_indices)
