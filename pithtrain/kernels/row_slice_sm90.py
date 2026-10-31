"""
Row machinery shared by the SM90 row kernels (cross-entropy, log-probability): where a CTA's slice
of a row sits, the bulk copy of that slice into shared memory, the 16-byte unit unpack and pack, and
the row's head and tail elements.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T

from pithtrain.kernels.cute_ptx_sm90 import asm, barrier_expect_tx, bf16_hi, bf16_lo, f32_as_u32, lds_b32, pack_bf16, packed_op, sts_b32, to_ir, u32_as_f32, unpack

__all__ = ["RowAccess", "RowSlice", "lds_f32", "pack_unit", "sts_f32", "unit_extremes", "unpack_unit"]


def lds_f32(addr: Int32) -> Float32:
    return u32_as_f32(lds_b32(addr))


def sts_f32(addr: Int32, value: Float32) -> None:
    sts_b32(addr, f32_as_u32(value), Int32(1))


def unpack_unit(words: tuple, dtype) -> list:
    """
    The elements of 16 bytes (or of fewer words) as fp32, in address order.
    """
    if const_expr(dtype is cutlass.BFloat16):
        return [half for word in words for half in (bf16_lo(word), bf16_hi(word))]
    if const_expr(dtype is cutlass.Float16):
        f32_pair = ir.Type.parse("!llvm.struct<(f32, f32)>")
        return [half for word in words for half in unpack(asm(f32_pair, [to_ir(word, Uint32)], "{\n.reg .b16 l, h;\nmov.b32 {l, h}, $2;\ncvt.f32.f16 $0, l;\ncvt.f32.f16 $1, h;\n}", "=f,=f,r"), 2, Float32)]
    return [u32_as_f32(word) for word in words]


def unit_extremes(words: tuple, dtype) -> tuple:
    """
    (max, min) of a unit; for 16-bit types the packed max / min select an input value exactly before
    it is widened.
    """
    if const_expr(dtype is not cutlass.Float32):
        suffix = ".bf16x2" if const_expr(dtype is cutlass.BFloat16) else ".f16x2"
        packed_max = packed_op("max" + suffix, packed_op("max" + suffix, words[0], words[1]), packed_op("max" + suffix, words[2], words[3]))
        packed_min = packed_op("min" + suffix, packed_op("min" + suffix, words[0], words[1]), packed_op("min" + suffix, words[2], words[3]))
        (max_lo, max_hi), (min_lo, min_hi) = unpack_unit((packed_max,), dtype), unpack_unit((packed_min,), dtype)
        return cute.arch.fmax(max_lo, max_hi), cute.arch.fmin(min_lo, min_hi)
    values = unpack_unit(words, dtype)
    unit_max, unit_min = values[0], values[0]
    for value in values[1:]:
        unit_max, unit_min = cute.arch.fmax(unit_max, value), cute.arch.fmin(unit_min, value)
    return unit_max, unit_min


def pack_unit(values: list, dtype) -> tuple:
    """
    Round the fp32 values once to dtype and pack them into four words.
    """
    if const_expr(dtype is cutlass.BFloat16):
        return tuple(pack_bf16(values[2 * i], values[2 * i + 1]) for i in range(4))
    if const_expr(dtype is cutlass.Float16):
        return tuple(Uint32(asm(T.i32(), [to_ir(values[2 * i + 1], Float32), to_ir(values[2 * i], Float32)], "cvt.rn.f16x2.f32 $0, $1, $2;", "=r,f,f")) for i in range(4))
    return tuple(f32_as_u32(value) for value in values)


class RowSlice:
    """
    Where a CTA's part of one row sits. The row splits into a 16-byte aligned interior, cut into
    slices of slice_units 16-byte units, one per CTA of the cluster, and a head and a tail of fewer
    than 16 bytes each, which rank 0 handles element by element. target_unit is the CTA-local unit
    holding x[target], -1 when another CTA or the head / tail holds it.
    """

    def __init__(self, x_base: Int64, row_idx: Int32, cta_rank: Int32, target_idx: Int64, num_cols: int, element_bytes: int, slice_units: int) -> None:
        row_bytes = Int64(num_cols * element_bytes)
        self.row_start = x_base + Int64(row_idx) * row_bytes
        self.row_end = self.row_start + row_bytes
        self.interior_start = cutlass.min((self.row_start + Int64(15)) & Int64(-16), self.row_end)
        self.interior_end = cutlass.max(self.row_end & Int64(-16), self.interior_start)
        num_units = Int32((self.interior_end - self.interior_start) >> Int64(4))
        unit_start = cta_rank * slice_units
        self.cta_units = cutlass.max(cutlass.min(num_units - unit_start, Int32(slice_units)), Int32(0))
        self.slice_start = self.interior_start + Int64(unit_start) * Int64(16)
        self.target_addr = self.row_start + target_idx * Int64(element_bytes)
        is_interior_target = (self.target_addr >= self.interior_start) & (self.target_addr < self.interior_end)
        target_unit_in_row = Int32((self.target_addr - self.interior_start) >> Int64(4))
        owner_rank = cutlass.select_(is_interior_target, target_unit_in_row // max(slice_units, 1), Int32(0))
        self.owns_target = owner_rank == cta_rank
        self.target_unit = cutlass.select_(is_interior_target & self.owns_target, target_unit_in_row - unit_start, Int32(-1))
        self.target_element = Int32((self.target_addr - self.interior_start) & Int64(15)) // element_bytes


class RowAccess:
    """
    How a row kernel moves its row for one element type: the bulk copy of a CTA's slice into shared
    memory in chunks of chunk_units 16-byte units, one mbarrier per chunk, and the head and tail
    elements one at a time.
    """

    def __init__(self, dtype, chunks: int, chunk_units: int) -> None:
        self.dtype, self.chunks, self.chunk_units = dtype, chunks, chunk_units
        self.element_bytes = dtype.width // 8
        self.unit_elements = 16 // self.element_bytes

    @cute.jit
    def init_barriers(self, s_barriers: Int32, tidx: Int32, num_barriers: cutlass.Constexpr[int], is_cluster: cutlass.Constexpr[bool]) -> None:
        """
        num_barriers mbarriers of one arrival each. In a cluster, peers signal a CTA's barriers only
        after the cluster wait that follows this arrive.
        """
        if tidx == 0:
            for k in cutlass.range_constexpr(num_barriers):
                asm(None, [to_ir(s_barriers + k * 8, Int32)], "mbarrier.init.shared::cta.b64 [$0], 1;", "r", side_effects=True)
            asm(None, [], "fence.mbarrier_init.release.cluster;", "", side_effects=True)
        cute.arch.sync_threads()
        if const_expr(is_cluster):
            cute.arch.cluster_arrive_relaxed()

    @cute.jit
    def issue_slice_loads(self, row: RowSlice, s_slice: Int32, s_barriers: Int32, tidx: Int32) -> None:
        """
        One thread starts a bulk copy of each chunk of the slice into shared memory.
        """
        if tidx == 0:
            for k in cutlass.range_constexpr(self.chunks):
                chunk_units = cutlass.max(cutlass.min(row.cta_units - k * self.chunk_units, Int32(self.chunk_units)), Int32(0))
                barrier_expect_tx(s_barriers + k * 8, chunk_units * 16)
                if chunk_units > 0:
                    asm(None, [to_ir(s_slice + k * self.chunk_units * 16, Int32), to_ir(row.slice_start + Int64(k * self.chunk_units * 16), Int64), to_ir(chunk_units * 16, Int32), to_ir(s_barriers + k * 8, Int32)], "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];", "r,l,r,r", side_effects=True)

    @cute.jit
    def edge_element(self, row: RowSlice, tidx: Int32) -> tuple:
        """
        Thread i < unit_elements owns head element i, thread unit_elements + i owns tail element i.
        """
        is_head = tidx < self.unit_elements
        edge_addr = cutlass.select_(is_head, row.row_start + Int64(tidx * self.element_bytes), row.interior_end + Int64((tidx - self.unit_elements) * self.element_bytes))
        edge_ok = cutlass.select_(is_head, edge_addr < row.interior_start, (tidx < 2 * self.unit_elements) & (edge_addr < row.row_end))
        return edge_addr, edge_ok

    @cute.jit
    def load_element(self, addr: Int64, ok) -> Float32:
        value = -Float32.inf
        if ok:
            if const_expr(self.element_bytes == 4):
                value = Float32(asm(T.f32(), [to_ir(addr, Int64)], "ld.global.f32 $0, [$1];", "=f,l", side_effects=True))
            else:
                bits = Uint32(asm(T.i32(), [to_ir(addr, Int64)], "{\n.reg .b16 h;\nld.global.b16 h, [$1];\ncvt.u32.u16 $0, h;\n}", "=r,l", side_effects=True))
                value = unpack_unit((bits,), self.dtype)[0]
        return value

    @cute.jit
    def store_element(self, addr: Int64, value: Float32, ok) -> None:
        if ok:
            if const_expr(self.element_bytes == 4):
                asm(None, [to_ir(addr, Int64), to_ir(value, Float32)], "st.global.f32 [$0], $1;", "l,f", side_effects=True)
            else:
                convert = "cvt.rn.bf16.f32" if const_expr(self.dtype is cutlass.BFloat16) else "cvt.rn.f16.f32"
                asm(None, [to_ir(addr, Int64), to_ir(value, Float32)], "{\n.reg .b16 h;\n" + convert + " h, $1;\nst.global.b16 [$0], h;\n}", "l,f", side_effects=True)
