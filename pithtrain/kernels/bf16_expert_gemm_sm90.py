"""
SM90 implementations of the bf16_expert_gemm functions: QuACK's SM90 GEMM class with our own
epilogues, compiled once per (kind, lane order, tile config, device, activation) through TVM-FFI.
Every launch runs DEFAULT_CONFIG until tune_kernels() picks, per kind, shape and bucket of rows per
expert, one of CANDIDATES; QuACK's own autotuner is not used.

Gate and up are two tensors read as one operand: a tensor over the lower address whose size-2 pair
mode strides to the higher one, so they act as one (G, 2I, H) weight with gate and up interleaved
(TMA takes any positive 16-byte-multiple stride). Lane 0 is the lower address, gate or up; the
epilogue variant is picked per call from the pointer order.
"""

from functools import lru_cache
from typing import Callable, NamedTuple

import cutlass
import cutlass.cute as cute
import quack.sm90_utils as quack_sm90_utils
import torch
from cutlass import BFloat16, Float32, Int32
from quack.activation import dswiglu, dswiglu_oai, swiglu, swiglu_oai
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_max_active_clusters
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import pack, unpack
from quack.epilogue.ops import EpiOp, RowVecLoad, TileStore
from quack.rounding import RoundingMode
from quack.tile_scheduler import TileSchedulerOptions
from quack.varlen_utils import VarlenArguments

from pithtrain.kernels.tuning import Tunable
from pithtrain.kernels.cute_compile_sm90 import fake_stream


def round_bf16(value):
    if isinstance(value, tuple):
        return type(value)(*(round_bf16(x) for x in value))
    return value.to(BFloat16).to(Float32)


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def gated(acc):
    acc_lo, acc_hi = unpack(acc)
    gate, up = round_bf16(acc_lo), round_bf16(acc_hi)
    return {"D": pack(gate, up), "postact": swiglu(gate, up)}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def gated_swapped(acc):
    acc_lo, acc_hi = unpack(acc)
    gate, up = round_bf16(acc_hi), round_bf16(acc_lo)
    return {"D": pack(gate, up), "postact": swiglu(gate, up)}


@gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
def dgated(acc, c):
    gate, up = unpack(c)
    grad_gate, grad_up, h = dswiglu(gate, up, acc)
    return {"D": pack(grad_gate, grad_up), "postact": h}


@gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
def dgated_swapped(acc, c):
    gate, up = unpack(c)
    grad_gate, grad_up, h = dswiglu(gate, up, acc)
    return {"D": pack(grad_up, grad_gate), "postact": h}


@gemm_epilogue()
def plain(acc):
    return {"D": acc}


@gemm_epilogue(mode="acc_pair")
def pair_swapped(acc):
    acc_lo, acc_hi = unpack(acc)
    return {"D": pack(acc_hi, acc_lo)}


# gpt-oss: the gate / up rows interleave, so gate is always the lower address and there is no
# swapped variant.
@lru_cache(maxsize=None)
def make_gated_oai(alpha: float, limit: float):
    @gemm_epilogue(outputs=("postact",), mode="acc_pair", ops={"bias": RowVecLoad("bias")})
    def gated_oai(acc, bias):
        gate, up = unpack(acc + bias)
        gate, up = round_bf16(gate), round_bf16(up)
        return {"D": pack(gate, up), "postact": swiglu_oai(gate, up, alpha, limit)}

    return gated_oai


@lru_cache(maxsize=None)
def make_dgated_oai(alpha: float, limit: float):
    @gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
    def dgated_oai(acc, c):
        gate, up = unpack(c)
        grad_gate, grad_up, h = dswiglu_oai(gate, up, acc, alpha, limit)
        return {"D": pack(grad_gate, grad_up), "postact": h}

    return dgated_oai


class IndexArg(EpiOp):
    """
    A tensor handed through to the epilogue params (read by ScatterRowStore).
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}


class ScatterRowStore(TileStore):
    """
    TileStore of out (slots, N) writing row r of the grouped output to out[idx[r]]: the tile is
    staged in smem as for a TMA store, then all epilogue threads copy its rows with 16-byte stores.
    """

    def __init__(self, name: str, idx_name: str):
        super().__init__(name)
        self.idx_name = idx_name

    def config_key(self):
        return (*super().config_key(), self.idx_name)

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
        assert layout.is_n_major_c()
        setattr(gemm, self._layout_gemm_attr(), layout)
        setattr(gemm, self._dtype_gemm_attr(), tensor.element_type)
        smem_layout = quack_sm90_utils.make_smem_layout_epi(tensor.element_type, layout, gemm.epi_tile, gemm.epi_stage)
        return {self._tma_atom_key(): None, self.name: tensor, self._smem_layout_key(): smem_layout, self._epi_tile_key(): gemm.epi_tile, self._dtype_field(): tensor.element_type}

    def tma_atoms(self, gemm, params):
        return []

    def store_setup(self, gemm, params, smem_tensor, tiled_copy_r2s, tiled_copy_t2r, tile_coord_mnkl, varlen_manager, tidx):
        tiled_copy = self._make_tiled_copy_r2s(gemm, params, tiled_copy_r2s, tiled_copy_t2r)
        expert_idx = tile_coord_mnkl[3]
        self.store_state = dict(s_epi=smem_tensor, out=getattr(params, self.name), slot_of=getattr(params, self.idx_name), group_start=varlen_manager.params.cu_seqlens_m[expert_idx], num_rows=varlen_manager.len_m(expert_idx), row_start=tile_coord_mnkl[0] * gemm.cta_tile_shape_mnk[0], col_start=tile_coord_mnkl[1] * gemm.cta_tile_shape_mnk[1], subtile_layout=cute.make_ordered_layout(cute.ceil_div(gemm.cta_tile_shape_mnk[:2], gemm.epi_tile), order=(0, 1) if gemm.epi_m_major else (1, 0)), num_stored=0)
        tRS_s = tiled_copy.get_slice(tidx).partition_D(smem_tensor)  # (CPY, CPY_M, CPY_N, STAGE)
        return (tiled_copy, tRS_s, lambda src_idx, dst_idx: None, None)

    @cute.jit
    def store_r2s(self, gemm, tiled_copy, frag_out, tRS_s_stage, tidx):
        super().store_r2s(gemm, tiled_copy, frag_out, tRS_s_stage, tidx)
        state = self.store_state
        # the driver stores the subtiles in epi_idx order within one traced tile, each in stage
        # (tiles before + epi_idx) % epi_stage like its TMA stores
        epi_coord = state["subtile_layout"].get_hier_coord(state["num_stored"] % cute.size(state["subtile_layout"]))
        state["num_stored"] += 1
        s_epi = state["s_epi"]  # (EPI_M, EPI_N, STAGE)
        thr_s = tiled_copy.get_slice(tidx).partition_D(s_epi)  # (CPY, CPY_M, CPY_N, STAGE)
        stage_bytes = cute.size(s_epi[None, None, 0]) * (s_epi.element_type.width // 8)
        stage_idx = (tRS_s_stage.iterator.toint() - thr_s[None, None, None, 0].iterator.toint()) // stage_bytes
        s_stage = s_epi[None, None, stage_idx]
        gemm.epilogue_barrier.arrive_and_wait()
        epi_m, epi_n = cute.size(gemm.epi_tile, mode=[0]), cute.size(gemm.epi_tile, mode=[1])
        out, slot_of = state["out"], state["slot_of"]
        N = cute.size(out, mode=[1])
        num_threads = gemm.num_epi_warps * cute.arch.WARP_SIZE
        vecs_per_row = epi_n // 8
        epi_tidx = tidx % num_threads
        for pass_idx in cutlass.range_constexpr((epi_m * vecs_per_row + num_threads - 1) // num_threads):
            vec_idx = pass_idx * num_threads + epi_tidx
            sub_row, sub_col = vec_idx // vecs_per_row, (vec_idx % vecs_per_row) * 8
            row, col = state["row_start"] + epi_coord[0] * epi_m + sub_row, state["col_start"] + epi_coord[1] * epi_n + sub_col
            # rows past the expert's end and columns past N (a partial last tile) stay unwritten
            if (row < state["num_rows"]) & (col < N) & (vec_idx < epi_m * vecs_per_row):
                fragment = cute.make_rmem_tensor(cute.make_layout(8), s_stage.element_type)
                cute.autovec_copy(cute.local_tile(s_stage, (1, 8), (sub_row, sub_col // 8))[0, None], fragment)
                dst = cute.local_tile(out, (1, 8), (slot_of[state["group_start"] + row], col // 8))[0, None]
                cute.autovec_copy(fragment, dst)


@gemm_epilogue(outputs=(ScatterRowStore("out", "slot"),), extra_ops=(IndexArg("slot"),))
def scatter(acc):
    return {"out": acc}


@gemm_epilogue(outputs=(ScatterRowStore("out", "slot"),), ops={"bias": RowVecLoad("bias")}, extra_ops=(IndexArg("slot"),))
def scatter_bias(acc, bias):
    return {"out": acc + bias}


def mint_gemm_class(epilogue, *, has_c=False, paired_acc=False, packed_c=False, add_to_output=False, has_bias=False) -> type:
    kind_sig = (("c", "c"),) if has_c else (("bias", "row"),) if has_bias else ()
    return epilogue._mint(kind_sig, 9, paired_acc, packed_c, (), RoundingMode.RN, (), add_to_output)


def paired_tensor(lower: cute.Tensor, higher: cute.Tensor, make_shape: Callable, make_stride: Callable) -> cute.Tensor:
    """
    A tensor over lower whose size-2 pair mode strides to higher (same element type); make_shape and
    make_stride take that pair stride in elements.
    """
    pair_stride = (higher.iterator.toint() - lower.iterator.toint()) // (lower.element_type.width // 8)
    return cute.make_tensor(lower.iterator, cute.make_layout(make_shape(pair_stride), stride=make_stride(pair_stride)))


def transpose_view(tensor: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(tensor.iterator, cute.select(tensor.layout, mode=[1, 0]))


class GemmEntry:
    def __init__(self, gemm, cluster_size: int):
        self.gemm = gemm
        self.max_active_clusters = get_max_active_clusters(cluster_size)

    def scheduler_options(self) -> TileSchedulerOptions:
        return TileSchedulerOptions(max_active_clusters=Int32(self.max_active_clusters))


class GatedFwd(GemmEntry):
    """
    preact (M, 2I), h (M, I) = gated(gathered[a_idx] @ [W0|W1]^T), per expert.
    """

    @cute.jit
    def __call__(self, mX, mW0, mW1, mPreact, mH, mCuSeqlens, mAIdx, stream):
        num_experts, inter_size, hidden_size = mW0.shape
        expert_stride, inter_stride, hidden_stride = mW0.stride
        mB = paired_tensor(mW0, mW1, lambda _: (num_experts, (2, inter_size), hidden_size), lambda pair_stride: (expert_stride, (pair_stride, inter_stride), hidden_stride))
        epilogue_args = self.gemm.EpilogueArguments(postact=mH)
        varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlens, mAIdx=mAIdx)
        self.gemm(mX, mB, mPreact, None, epilogue_args, self.scheduler_options(), varlen_args, stream)


class GatedFwdBias(GemmEntry):
    """
    As GatedFwd plus a (G, 2I) bias row (interleaved like the weight rows) before the gate.
    """

    @cute.jit
    def __call__(self, mX, mW0, mW1, mBias, mPreact, mH, mCuSeqlens, mAIdx, stream):
        num_experts, inter_size, hidden_size = mW0.shape
        expert_stride, inter_stride, hidden_stride = mW0.stride
        mB = paired_tensor(mW0, mW1, lambda _: (num_experts, (2, inter_size), hidden_size), lambda pair_stride: (expert_stride, (pair_stride, inter_stride), hidden_stride))
        epilogue_args = self.gemm.EpilogueArguments(bias=mBias, postact=mH)
        varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlens, mAIdx=mAIdx)
        self.gemm(mX, mB, mPreact, None, epilogue_args, self.scheduler_options(), varlen_args, stream)


class DownScatter(GemmEntry):
    """
    out[slot_of[r]] = (h @ W_down^T)[r], per expert: rows land in received-slot order.
    """

    @cute.jit
    def __call__(self, mH, mW, mOut, mSlot, mCuSeqlens, stream):
        varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlens)
        epilogue_args = self.gemm.EpilogueArguments(out=mOut, slot=mSlot)
        self.gemm(mH, mW, None, None, epilogue_args, self.scheduler_options(), varlen_args, stream)


class DownBiasScatter(GemmEntry):
    """
    out[slot_of[r]] = (h @ W_down^T + bias)[r], per expert.
    """

    @cute.jit
    def __call__(self, mH, mW, mBias, mOut, mSlot, mCuSeqlens, stream):
        varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlens)
        epilogue_args = self.gemm.EpilogueArguments(bias=mBias, out=mOut, slot=mSlot)
        self.gemm(mH, mW, None, None, epilogue_args, self.scheduler_options(), varlen_args, stream)


class Dgated(GemmEntry):
    """
    dpre (M, 2I), h (M, I) = dgated(dy[slot_of] @ W_down, preact), per expert.
    """

    @cute.jit
    def __call__(self, mDy, mW, mPreact, mDpre, mH, mCuSeqlens, mAIdx, stream):
        epilogue_args = self.gemm.EpilogueArguments(postact=mH)
        varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlens, mAIdx=mAIdx)
        self.gemm(mDy, mW, mDpre, mPreact, epilogue_args, self.scheduler_options(), varlen_args, stream)


class Dgrad(GemmEntry):
    """
    dx (M, H) = dpre @ [W0; W1], one GEMM over K = 2I, per expert.
    """

    @cute.jit
    def __call__(self, mDpre, mW0, mW1, mDx, mCuSeqlens, stream):
        num_experts, inter_size, hidden_size = mW0.shape
        expert_stride, inter_stride, hidden_stride = mW0.stride
        mB = paired_tensor(mW0, mW1, lambda _: (num_experts, hidden_size, (2, inter_size)), lambda pair_stride: (expert_stride, hidden_stride, (pair_stride, inter_stride)))
        varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlens)
        self.gemm(mDpre, mB, mDx, None, self.gemm.EpilogueArguments(), self.scheduler_options(), varlen_args, stream)


class WgradPair(GemmEntry):
    """
    [G0 | G1] (G, I, H) fp32 += dpre^T @ gathered[a_idx], per expert, as D = (H, (2, I)).
    """

    @cute.jit
    def __call__(self, mX, mDpre, mG0, mG1, mCuSeqlens, mAIdx, stream):
        num_experts, inter_size, hidden_size = mG0.shape
        expert_stride, inter_stride, hidden_stride = mG0.stride
        mD = paired_tensor(mG0, mG1, lambda _: (num_experts, hidden_size, (2, inter_size)), lambda pair_stride: (expert_stride, hidden_stride, (pair_stride, inter_stride)))
        varlen_args = VarlenArguments(mCuSeqlensK=mCuSeqlens, mAIdx=mAIdx)
        epilogue_args = self.gemm.EpilogueArguments()
        self.gemm(transpose_view(mX), transpose_view(mDpre), mD, None, epilogue_args, self.scheduler_options(), varlen_args, stream)


class WgradDown(GemmEntry):
    """
    G (G, H, I) fp32 += dy[slot_of]^T @ h, per expert.
    """

    @cute.jit
    def __call__(self, mDy, mH, mG, mCuSeqlens, mAIdx, stream):
        varlen_args = VarlenArguments(mCuSeqlensK=mCuSeqlens, mAIdx=mAIdx)
        epilogue_args = self.gemm.EpilogueArguments()
        self.gemm(transpose_view(mDy), transpose_view(mH), mG, None, epilogue_args, self.scheduler_options(), varlen_args, stream)


class Config(NamedTuple):
    tile_m: int
    tile_n: int
    cluster_m: int
    pingpong: bool = False


DEFAULT_CONFIG = Config(128, 192, 2, True)
"""
The candidate that is fastest most often across kinds and sizes; every launch runs it until tuned.
"""

CANDIDATES = (Config(128, 128, 2, True), Config(128, 192, 1, True), Config(128, 192, 2, False), Config(128, 192, 2, True), Config(128, 256, 1, False), Config(128, 256, 2, False), Config(256, 192, 1, False), Config(256, 192, 2, False))
"""
The configs tune_kernels() chooses among, for every kind and shape; each compiles and computes every
kind correctly (tests/kernels/test_bf16_expert_gemm.py loads each in turn).
"""


def fake_matrix(dtype, divisibility: int = 8):
    return make_fake_tensor(dtype, (cute.sym_int(), cute.sym_int()), divisibility=divisibility, leading_dim=1)


def fake_expert_tensor(dtype, divisibility: int = 8):
    return make_fake_tensor(dtype, (cute.sym_int(), cute.sym_int(), cute.sym_int()), divisibility=divisibility, leading_dim=2)


def fake_index_vector():
    return make_fake_tensor(Int32, (cute.sym_int(),), divisibility=4, leading_dim=0)


class GemmKind(NamedTuple):
    """
    How one kind of GEMM is built. epilogues is (lane order kept, swapped), None where there is no
    swapped variant; an epilogue that is a function takes the activation parameters alpha, limit.
    """

    entry: type
    epilogues: tuple
    mint_kwargs: dict
    gather_a: bool
    gemm_attrs: dict
    fake_args: Callable


KINDS = {
    "gated": GemmKind(GatedFwd, (gated, gated_swapped), dict(paired_acc=True), True, {}, lambda: (fake_matrix(BFloat16), fake_expert_tensor(BFloat16), fake_expert_tensor(BFloat16), fake_matrix(BFloat16), fake_matrix(BFloat16), fake_index_vector(), fake_index_vector())),
    "dgated": GemmKind(Dgated, (dgated, dgated_swapped), dict(has_c=True, packed_c=True), True, dict(cd_packed="n", implicit_dtype=BFloat16, b_transposed=True), lambda: (fake_matrix(BFloat16), fake_expert_tensor(BFloat16), fake_matrix(BFloat16), fake_matrix(BFloat16), fake_matrix(BFloat16), fake_index_vector(), fake_index_vector())),
    "dgrad": GemmKind(Dgrad, (plain, None), {}, False, {}, lambda: (fake_matrix(BFloat16), fake_expert_tensor(BFloat16), fake_expert_tensor(BFloat16), fake_matrix(BFloat16), fake_index_vector())),
    "wgrad_pair": GemmKind(WgradPair, (plain, pair_swapped), dict(add_to_output=True), True, {}, lambda: (fake_matrix(BFloat16), fake_matrix(BFloat16), fake_expert_tensor(Float32, 4), fake_expert_tensor(Float32, 4), fake_index_vector(), fake_index_vector())),
    "wgrad_down": GemmKind(WgradDown, (plain, None), dict(add_to_output=True), True, {}, lambda: (fake_matrix(BFloat16), fake_matrix(BFloat16), fake_expert_tensor(Float32, 4), fake_index_vector(), fake_index_vector())),
    "down_scatter": GemmKind(DownScatter, (scatter, None), {}, False, {}, lambda: (fake_matrix(BFloat16), fake_expert_tensor(BFloat16), fake_matrix(BFloat16), fake_index_vector(), fake_index_vector())),
    "down_bias_scatter": GemmKind(DownBiasScatter, (scatter_bias, None), dict(has_bias=True), False, {}, lambda: (fake_matrix(BFloat16), fake_expert_tensor(BFloat16), fake_matrix(BFloat16, 4), fake_matrix(BFloat16), fake_index_vector(), fake_index_vector())),
    "gated_oai": GemmKind(GatedFwdBias, (make_gated_oai, None), dict(paired_acc=True, has_bias=True), True, {}, lambda: (fake_matrix(BFloat16), fake_expert_tensor(BFloat16), fake_expert_tensor(BFloat16), fake_matrix(BFloat16, 4), fake_matrix(BFloat16), fake_matrix(BFloat16), fake_index_vector(), fake_index_vector())),
    "dgated_oai": GemmKind(Dgated, (make_dgated_oai, None), dict(has_c=True, packed_c=True), True, dict(cd_packed="n", implicit_dtype=BFloat16, b_transposed=True), lambda: (fake_matrix(BFloat16), fake_expert_tensor(BFloat16), fake_matrix(BFloat16), fake_matrix(BFloat16), fake_matrix(BFloat16), fake_index_vector(), fake_index_vector())),
}

compiled: dict = {}


def compile_gemm(kind: str, swapped: bool, config: Config, device: torch.device, act: tuple = ()) -> Callable:
    key = (kind, swapped, config, device.index, act)
    fn = compiled.get(key)
    if fn is None:
        gemm_kind = KINDS[kind]
        epilogue = gemm_kind.epilogues[1] if swapped else gemm_kind.epilogues[0]
        if act:
            epilogue = epilogue(*act)
        mint_kwargs = gemm_kind.mint_kwargs
        if kind == "wgrad_pair" and swapped:
            mint_kwargs = dict(mint_kwargs, paired_acc=True)
        gemm_class = mint_gemm_class(epilogue, **mint_kwargs)
        gemm = gemm_class(Float32, BFloat16, (config.tile_m, config.tile_n), (config.cluster_m, 1, 1), pingpong=config.pingpong, is_persistent=True, gather_A=gemm_kind.gather_a)
        for name, value in gemm_kind.gemm_attrs.items():
            setattr(gemm, name, value)
        with torch.cuda.device(device):
            fn = cute.compile(gemm_kind.entry(gemm, config.cluster_m), *gemm_kind.fake_args(), fake_stream(), options="--enable-tvm-ffi")
        compiled[key] = fn
    return fn


MAX_SYNTHETIC_BYTES = 4 << 30


def default_config(key: tuple) -> Config:
    return DEFAULT_CONFIG


def gemm_config(kind: str, swapped: bool, N: int, K: int, G: int, act: tuple, rows: int) -> Config:
    return TUNING.pick((kind, swapped, N, K, G, act), rows // max(G, 1))


def prepare_launch(key: tuple, rows_per_expert: int) -> Callable | None:
    """
    launch(config) for synthetic inputs of this kind with rows_per_expert rows in every expert, laid
    out as the exported functions pass them; launch zeroes the accumulated weight gradients first so
    each launch gives them afresh.
    """
    kind, swapped, N, K, G, act = key
    M, device = rows_per_expert * G, torch.device("cuda", torch.cuda.current_device())
    if max(M * max(N, K), G * N * K) * 4 > MAX_SYNTHETIC_BYTES:
        return None

    def randn(*shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, device=device) * shape[-1] ** -0.5).to(dtype)

    cu, rows_idx = (torch.arange(G + 1, device=device) * rows_per_expert).to(torch.int32), torch.arange(M, device=device, dtype=torch.int32)
    if kind == "gated":
        pair = randn(2, G, N // 2, K)
        args, outputs = (randn(M, K), pair[0], pair[1], randn(M, N), randn(M, N // 2), cu, rows_idx), (3, 4)
    elif kind == "gated_oai":
        pair = randn(G, N, K)
        args, outputs = (randn(M, K), pair[:, 0::2], pair[:, 1::2], randn(G, N), randn(M, N), randn(M, N // 2), cu, rows_idx), (4, 5)
    elif kind == "down_scatter":
        args, outputs = (randn(M, K), randn(G, N, K), randn(M, N), rows_idx, cu), (2,)
    elif kind == "down_bias_scatter":
        args, outputs = (randn(M, K), randn(G, N, K), randn(G, N), randn(M, N), rows_idx, cu), (3,)
    elif kind in ("dgated", "dgated_oai"):
        args, outputs = (randn(M, K), randn(G, K, N), randn(M, 2 * N), randn(M, 2 * N), randn(M, N), cu, rows_idx), (3, 4)
    elif kind == "dgrad":
        pair = randn(2, G, K // 2, N)
        args, outputs = (randn(M, K), pair[0], pair[1], randn(M, N), cu), (3,)
    elif kind == "wgrad_pair":
        pair = randn(2, G, N // 2, K, dtype=torch.float32)
        args, outputs = (randn(M, K), randn(M, N), pair[0], pair[1], cu, rows_idx), (2, 3)
    else:
        assert kind == "wgrad_down"
        args, outputs = (randn(M, K), randn(M, N), randn(G, K, N, dtype=torch.float32), cu, rows_idx), (2,)
    accumulated = kind in ("wgrad_pair", "wgrad_down")

    def launch(config: Config) -> tuple:
        if accumulated:
            for idx in outputs:
                args[idx].zero_()
        compile_gemm(kind, swapped, config, device, act)(*args)
        return tuple(args[idx] for idx in outputs)

    return launch


TUNING = Tunable("bf16_expert_gemm", CANDIDATES, default_config, prepare_launch)


def order_by_address(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Returns (lower, higher, whether first is the higher) by address.
    """
    assert first.dtype == second.dtype and first.shape == second.shape and first.stride() == second.stride()
    assert first.data_ptr() % 16 == 0 and second.data_ptr() % 16 == 0
    byte_gap = second.data_ptr() - first.data_ptr()
    assert byte_gap != 0 and byte_gap % 16 == 0, "paired operands must be distinct and 16-byte apart"
    return (first, second, False) if byte_gap > 0 else (second, first, True)


def bf16_expert_gate_up_fwd_sm90(x, w_gate, w_up, cu, a_idx, bias=None, act=()):
    M, inter = a_idx.shape[0], w_gate.shape[1]
    w_lower, w_higher, swapped = order_by_address(w_gate, w_up)
    preact, h = x.new_empty(M, 2 * inter), x.new_empty(M, inter)
    if M == 0:
        return preact, h
    if act:
        assert not swapped and bias is not None
        fn = compile_gemm("gated_oai", False, gemm_config("gated_oai", False, 2 * inter, x.shape[1], w_gate.shape[0], act, M), x.device, act)
        fn(x, w_lower, w_higher, bias, preact, h, cu, a_idx)
    else:
        compile_gemm("gated", swapped, gemm_config("gated", swapped, 2 * inter, x.shape[1], w_gate.shape[0], (), M), x.device)(x, w_lower, w_higher, preact, h, cu, a_idx)
    return preact, h


def bf16_expert_down_fwd_sm90(h, w_down, cu, slot_of, bias=None):
    M, H = h.shape[0], w_down.shape[1]
    out = h.new_empty(M, H)
    if M == 0:
        return out
    if bias is not None:
        fn = compile_gemm("down_bias_scatter", False, gemm_config("down_bias_scatter", False, H, h.shape[1], w_down.shape[0], (), M), h.device)
        fn(h, w_down, bias, out, slot_of, cu)
    else:
        compile_gemm("down_scatter", False, gemm_config("down_scatter", False, H, h.shape[1], w_down.shape[0], (), M), h.device)(h, w_down, out, slot_of, cu)
    return out


def bf16_expert_down_dgrad_sm90(dy, w_down, preact, cu, slot_of, w_gate, w_up, act=()):
    M, inter = preact.shape[0], preact.shape[1] // 2
    swapped = w_up.data_ptr() < w_gate.data_ptr()
    dpre, h = preact.new_empty(M, 2 * inter), preact.new_empty(M, inter)
    if M:
        if act:
            assert not swapped
            fn = compile_gemm("dgated_oai", False, gemm_config("dgated_oai", False, inter, dy.shape[1], w_down.shape[0], act, M), dy.device, act)
        else:
            fn = compile_gemm("dgated", swapped, gemm_config("dgated", swapped, inter, dy.shape[1], w_down.shape[0], (), M), dy.device)
        fn(dy, w_down, preact, dpre, h, cu, slot_of)
    return dpre, h, swapped


def bf16_expert_gate_up_dgrad_sm90(dpre, w_gate, w_up, cu, dpre_swapped):
    M, H = dpre.shape[0], w_gate.shape[2]
    w_lower, w_higher, swapped = order_by_address(w_gate, w_up)
    assert swapped == dpre_swapped, "the weights moved between the dgated backward and the dgrad"
    dx = dpre.new_empty(M, H)
    if M:
        compile_gemm("dgrad", False, gemm_config("dgrad", False, H, dpre.shape[1], w_gate.shape[0], (), M), dpre.device)(dpre, w_lower, w_higher, dx, cu)
    return dx


def bf16_expert_gate_up_wgrad_sm90(x, dpre, dpre_swapped, mg_gate, mg_up, cu, a_idx):
    if a_idx.shape[0] == 0:
        return
    mg_lower, mg_higher, mg_swapped = order_by_address(mg_gate, mg_up)
    swapped = mg_swapped != dpre_swapped
    fn = compile_gemm("wgrad_pair", swapped, gemm_config("wgrad_pair", swapped, dpre.shape[1], x.shape[1], mg_gate.shape[0], (), a_idx.shape[0]), x.device)
    fn(x, dpre, mg_lower, mg_higher, cu, a_idx)


def bf16_expert_down_wgrad_sm90(dy, h, mg_down, cu, slot_of):
    if slot_of.shape[0] == 0:
        return
    compile_gemm("wgrad_down", False, gemm_config("wgrad_down", False, h.shape[1], dy.shape[1], mg_down.shape[0], (), slot_of.shape[0]), dy.device)(dy, h, mg_down, cu, slot_of)


__all__ = ["bf16_expert_gate_up_fwd_sm90", "bf16_expert_down_fwd_sm90", "bf16_expert_down_dgrad_sm90", "bf16_expert_gate_up_dgrad_sm90", "bf16_expert_gate_up_wgrad_sm90", "bf16_expert_down_wgrad_sm90"]
