"""
Per-row group ids of a grouped GEMM input in one launch (SM90): row r gets min(#{g : offs[g] <= r},
G - 1), the same values as arange + searchsorted(right=True) + clamp.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import ld_b32, lds_b32, st_b32, sts_b32

__all__ = ["group_indices", "supports_group_indices"]

THREADS, ROWS_PER_THREAD, MAX_GROUPS = 256, 8, 1024


class GroupIndices:
    def __init__(self, num_groups: int) -> None:
        assert 0 < num_groups <= MAX_GROUPS
        self.G = num_groups

    @cute.jit
    def __call__(self, mOffs: cute.Tensor, mOut: cute.Tensor, stream) -> None:
        rows_per_cta = THREADS * ROWS_PER_THREAD
        num_ctas = (mOut.shape[0] + rows_per_cta - 1) // rows_per_cta
        self.kernel(mOffs, mOut).launch(grid=[num_ctas, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mOffs: cute.Tensor, mOut: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        M = mOut.shape[0]
        offs_base, out_base = mOffs.iterator.toint(), mOut.iterator.toint()
        smem = cutlass.memory.SmemAllocator()
        s_offs = smem.allocate(self.G * 4, byte_alignment=16).toint()
        self.load_group_ends(offs_base, s_offs, tidx)
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            row_idx = (bx * ROWS_PER_THREAD + i) * THREADS + tidx
            group_idx = self.find_group(s_offs, row_idx)
            st_b32(out_base + Int64(row_idx) * 4, Uint32(group_idx), Int32(row_idx < M))

    @cute.jit
    def load_group_ends(self, offs_base, s_offs, tidx) -> None:
        G = self.G
        for j in cutlass.range_constexpr((G + THREADS - 1) // THREADS):
            group_idx = j * THREADS + tidx
            group_end = ld_b32(offs_base + Int64(cutlass.min(group_idx, G - 1)) * 4, Int32(group_idx < G), Int32(0))
            sts_b32(s_offs + cutlass.min(group_idx, G - 1) * 4, Uint32(group_end), Int32(group_idx < G))

    @cute.jit
    def find_group(self, s_offs, row_idx):
        """
        min(number of group ends <= row_idx, G - 1), by binary search for the first end > row_idx.
        """
        G = self.G
        num_ended = Int32(0)
        step = 1 << (G.bit_length() - 1)
        while step > 0:
            candidate = num_ended + step
            candidate_ok, group_end = Int32(candidate <= G), Int32(lds_b32(s_offs + cutlass.min(candidate - 1, G - 1) * 4))
            num_ended = num_ended + step * (candidate_ok & Int32(group_end <= row_idx))
            step //= 2
        return cutlass.min(num_ended, G - 1)


COMPILED: dict = {}


def compile_kernel(G: int) -> Callable:
    return cute.compile(GroupIndices(G), fake(Int32, (G,), (1,), 4), fake(Int32, (cute.sym_int(),), (1,), 4), fake_stream(), options="--enable-tvm-ffi")


def supports_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> bool:
    return (grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.dim() == 1 and 0 < grouped_mm_offs.shape[0] <= MAX_GROUPS and grouped_mm_offs.is_contiguous() and 0 <= M < 2**31 and not torch.compiler.is_compiling())


def group_indices(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor:
    G = grouped_mm_offs.shape[0]
    out = torch.empty(M, dtype=torch.int32, device=grouped_mm_offs.device)
    if M == 0:
        return out
    key = ("group_indices", G)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_kernel(G)
    fn(grouped_mm_offs, out)
    return out
