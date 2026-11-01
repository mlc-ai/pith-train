"""
SM90 dense chunked gated delta rule (chunk size 64), forward and the pieces both directions share.
With T a multiple of the chunk size the forward runs two CuTe-DSL kernels: WyRepresentation (the
gate cumsum, the solved WY matrix A and w, u) and ChunkStateOutput (the chunk-state recurrence fused
with the output, h and v_new never leaving the SM). When T is not a multiple of the chunk size it
launches FLA's Triton kernels directly with the configs FLA's autotuner picks on H200, skipping
FLA's per-launch Python wrappers; the launch machinery here (launch, CONFIGS,
pinned_launch_supported) serves both directions.

The FLA launches keep the launch order, allocations, in-place accumulations and dtypes of FLA
0.5.2's chunk_gated_delta_rule_fwd on the dense path (no cu_seqlens, no initial or final state), so
they are bit-identical to FLA while the pinned configs are FLA's picks (pinned_launch_supported
checks the FLA and Triton versions and the kernels' signatures, not the picks). The CuTe forward
follows FLA's chunk algorithm and rounding points (bf16 A, w, u, h, v_new, fp32 accumulation, exp2
of the log2 gate, TF32 block merges of A with truncated operands, as FLA's tl.dot on Hopper) and
matches it to bf16 rounding. The ops stay opaque to torch.compile. Grouped value heads run on
unrepeated q / k, which the kernels index per group. The backward is
gated_delta_rule_pinned_bwd_sm90.
"""

import cutlass
import cutlass.cute as cute
import fla
import torch
import triton
from cutlass import BFloat16, Float32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.ops.common.chunk_o import chunk_bwd_kernel_dqkwg, chunk_bwd_kernel_dv_local, chunk_fwd_kernel_o
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd_kernel, recompute_w_u_fwd_kernel
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel
from flash_attn.cute.ampere_helpers import gemm, gemm_rs, get_smem_layout_atom
from quack.layout_utils import reshape_acc_to_frgA
from triton import knobs

from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream

__all__ = ["BT", "CONFIGS", "ChunkStateOutput", "HEAD_DIM", "STATE_BV", "STATE_STAGES", "STATE_THREADS", "WyRepresentation", "chunk_copies", "chunk_cumsum", "chunk_gdr_fwd_pinned_sm90", "compute_chunk_states", "launch", "multiply_a", "operand_copies", "pinned_launch_supported", "recompute_w_u", "scale_kv"]

BT = 64  # chunk size
BC = 16  # sub-chunk of the fused kkt + triangular solve, fixed by FLA at BT = 64

# Launch configs FLA's autotuner selects on H200 at H = HV = 32, K = V = 128 (bf16 q/k/v/beta,
# fp32 gate). The BK / BV the launches below pass explicitly are FLA's own constants on Hopper.
VERSIONS = {"fla": "0.5.2", "triton": "3.7.1"}
CONFIGS = {"cumsum_fwd": dict(num_warps=1, num_stages=3), "cumsum_rev": dict(num_warps=1, num_stages=3), "kkt_solve": dict(BK=64, num_warps=1, num_stages=3), "w_u": dict(num_warps=4, num_stages=4), "fwd_h": dict(BV=32, num_warps=4, num_stages=3), "fwd_o": dict(BK=64, BV=64, num_warps=4, num_stages=3), "dv_local": dict(num_warps=2, num_stages=3), "bwd_dhu": dict(BV=32, num_warps=4, num_stages=3), "dqkwg": dict(num_warps=4, num_stages=3), "wy_bwd": dict(num_warps=4, num_stages=4)}

KERNELS = {"cumsum": chunk_local_cumsum_scalar_kernel, "kkt_solve": chunk_gated_delta_rule_fwd_kkt_solve_kernel, "w_u": recompute_w_u_fwd_kernel, "fwd_h": chunk_gated_delta_rule_fwd_kernel_h_blockdim64, "fwd_o": chunk_fwd_kernel_o, "dv_local": chunk_bwd_kernel_dv_local, "bwd_dhu": chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64, "dqkwg": chunk_bwd_kernel_dqkwg, "wy_bwd": prepare_wy_repr_bwd_kernel}

# The parameter lists the launches below are written against.
SIGNATURES = {"cumsum": "s o scale cu_seqlens chunk_indices T B H BT REVERSE HAS_SCALE IS_VARLEN HEAD_FIRST", "kkt_solve": "k g beta A cu_seqlens chunk_indices T H HV K BT BC BK USE_G IS_VARLEN", "w_u": "k v beta w u A g cu_seqlens chunk_indices T H HV K V BT BK BV USE_G IS_VARLEN", "fwd_h": "k v w v_new g gk h h0 ht cu_seqlens chunk_offsets T H HV K V BT BV USE_G USE_GK USE_INITIAL_STATE STORE_FINAL_STATE SAVE_NEW_VALUE STATE_V_FIRST IS_VARLEN", "fwd_o": "q k v h g g_gamma o cu_seqlens chunk_indices scale T H HV K V BT BK BV USE_G USE_G_GAMMA STATE_V_FIRST IS_VARLEN", "dv_local": "q k g g_gamma A do dv cu_seqlens chunk_indices scale T H HV K V BT BK BV USE_G USE_G_GAMMA USE_A IS_VARLEN", "bwd_dhu": "q k w g gk dht dh0 do dh dv dv2 cu_seqlens chunk_offsets scale T H HV K V BT BV USE_G USE_GK USE_INITIAL_STATE USE_FINAL_STATE_GRADIENT STATE_V_FIRST IS_VARLEN", "dqkwg": "q k v g g_gamma h do dh dq dk dw dv dg cu_seqlens chunk_indices scale B T H HV K V BT BK BV USE_G USE_G_GAMMA USE_DW STATE_V_FIRST IS_VARLEN", "wy_bwd": "k v beta g A dw du dk dv db dg cu_seqlens chunk_indices T H HV K V BT BK BV USE_G IS_VARLEN"}


def unwrap_kernel(name: str) -> triton.JITFunction:
    """
    The raw JITFunction under FLA's Heuristics(CachedAutotuner(...)) wrappers.
    """
    return KERNELS[name].fn.fn


compiled: dict[tuple, object] = {}
key_positions: dict[str, tuple[list[int], list[int]]] = {}


def launch(name: str, grid: tuple, **kwargs) -> None:
    """
    jit_fn[grid](**kwargs), with the compiled kernel cached and launched straight through its C
    launcher on the device's current stream. The cache key is what Triton specializes these kernels
    on: each tensor's dtype and 16-byte alignment, the value of every other argument except T
    (do_not_specialize) and float scalars (never specialized), plus the device and launch config.
    """
    jit_fn = unwrap_kernel(name)
    args = [kwargs[arg_name] for arg_name in jit_fn.arg_names]
    if name not in key_positions:
        tensor_positions = [i for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)]
        value_positions = [i for i, arg_name in enumerate(jit_fn.arg_names) if i not in tensor_positions and arg_name != "T"]
        key_positions[name] = tensor_positions, value_positions
    tensor_positions, value_positions = key_positions[name]
    device = args[tensor_positions[0]].device.index
    # None (e.g. an absent scale) is a constexpr, so it keys on its value.
    key = (name, device, kwargs["num_warps"], kwargs["num_stages"], *(float if args[i].__class__ is float else args[i] for i in value_positions), *((args[i].dtype, args[i].data_ptr() & 15) for i in tensor_positions))
    fn = compiled.get(key)
    if fn is None:
        compiled[key] = jit_fn[grid](**kwargs)
        return
    grid = (*grid, 1, 1)
    stream = torch._C._cuda_getCurrentRawStream(device)
    enter_hook, exit_hook = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
    metadata = None if enter_hook is None else fn.launch_metadata(grid, stream, *args)
    fn.run(grid[0], grid[1], grid[2], stream, fn.function, fn.packed_metadata, metadata, enter_hook, exit_hook, *args)


supported: list[bool] = []


@torch.compiler.assume_constant_result
def pinned_launch_supported() -> bool:
    """
    Whether the pinned configs apply in this process: the recorded FLA and Triton versions, SM90,
    and FLA kernels whose wrappers and parameter lists match the launches here.
    """
    if not supported:
        supported.append(check_supported())
    return supported[0]


def check_supported() -> bool:
    if fla.__version__ != VERSIONS["fla"] or triton.__version__ != VERSIONS["triton"]:
        return False
    if ARCH_MAJOR != 9:
        return False
    for name, kernel in KERNELS.items():
        inner = getattr(getattr(kernel, "fn", None), "fn", None)
        if not isinstance(inner, triton.JITFunction) or inner.arg_names != SIGNATURES[name].split():
            return False
    return True


def chunk_cumsum(x: torch.Tensor, reverse: bool) -> torch.Tensor:
    B, T, H = x.shape
    out = torch.empty_like(x, dtype=torch.float)
    config = CONFIGS["cumsum_rev" if reverse else "cumsum_fwd"]
    launch("cumsum", (triton.cdiv(T, BT), B * H), s=x, o=out, scale=None if reverse else RCP_LN2, cu_seqlens=None, chunk_indices=None, T=T, B=B, H=H, BT=BT, REVERSE=reverse, HAS_SCALE=not reverse, IS_VARLEN=False, HEAD_FIRST=False, **config)
    return out


def recompute_w_u(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, HV, V = *k.shape, *v.shape[2:]
    w, u = k.new_empty(B, T, HV, K), torch.empty_like(v)
    launch("w_u", (triton.cdiv(T, BT), B * HV), k=k, v=v, beta=beta, w=w, u=u, A=A, g=g, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=64, BV=64, USE_G=True, IS_VARLEN=False, **CONFIGS["w_u"])
    return w, u


def compute_chunk_states(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, HV, V = *k.shape, *u.shape[2:]
    h, v_new = k.new_empty(B, triton.cdiv(T, BT), HV, K, V), torch.empty_like(u)
    config = CONFIGS["fwd_h"]
    launch("fwd_h", (triton.cdiv(V, config["BV"]), B * HV), k=k, v=u, w=w, v_new=v_new, g=g, gk=None, h=h, h0=None, ht=None, cu_seqlens=None, chunk_offsets=None, T=T, H=H, HV=HV, K=K, V=V, BT=BT, USE_G=True, USE_GK=False, USE_INITIAL_STATE=False, STORE_FINAL_STATE=False, SAVE_NEW_VALUE=True, STATE_V_FIRST=False, IS_VARLEN=False, **config)
    return h, v_new


HEAD_DIM = 128  # K == V, fixed by pinned_shapes
TF32_STRIDE = 68  # fp32 row stride of the 64 x 64 solve buffers: conflict-free TF32 A-fragment loads
WY_THREADS = 128
STATE_THREADS = 128
STATE_STAGES = 3  # chunks of k, w, u, g in flight
STATE_BV = 32  # state columns per CTA of the recurrence
BAR_RECURRENCE, BAR_OUTPUT, BAR_FULL, BAR_EMPTY = 1, 2, 3, 5  # FULL and EMPTY use ids +0 / +1 by chunk parity


@cute.jit
def block_product(thr: cute.ThrMma, mma_tf32: cute.TiledMma, acc: cute.Tensor, s_left: cute.Tensor, s_right: cute.Tensor, even_odd_k: cutlass.Constexpr[bool]):
    """
    acc += s_left @ s_right for 16 x 16 fp32 shared-memory blocks, on one warp's TF32 MMA (two
    m16n8k8 per 8 columns). The fp32 values go to the MMA as they are (recast, not converted), so it
    reads them truncated to TF32, as FLA's tl.dot does. The two k8 instructions take k = 0 .. 7 and
    8 .. 15, or with even_odd_k the even and the odd k (lane slot t + 4 j of instruction i reads
    k = 2 t + 8 j + i), the split of Triton's kWidth = 2 operand layout, which FLA's inner merge
    products use. The order changes the fp32 accumulation, not the products.
    """
    stride = TF32_STRIDE
    k_left = cute.make_layout(((4, 2), 2), stride=((2, 8), 1)) if const_expr(even_odd_k) else cute.make_layout(16, stride=1)
    k_right = cute.make_layout(((4, 2), 2), stride=((2 * stride, 8 * stride), stride)) if const_expr(even_odd_k) else cute.make_layout(16, stride=stride)
    s_left_k = cute.make_tensor(s_left.iterator, cute.make_layout((16, k_left.shape), stride=(stride, k_left.stride)))
    s_right_t = cute.make_tensor(s_right.iterator, cute.make_layout((16, k_right.shape), stride=(1, k_right.stride)))
    t_left, t_right = thr.partition_A(s_left_k), thr.partition_B(s_right_t)
    r_left, r_right = cute.make_rmem_tensor(thr.partition_shape_A((16, 16)), Float32), cute.make_rmem_tensor(thr.partition_shape_B((16, 16)), Float32)
    cute.autovec_copy(t_left, r_left)
    cute.autovec_copy(t_right, r_right)
    tf32 = lambda r: cute.make_tensor(cute.recast_ptr(r.iterator, dtype=cutlass.TFloat32), r.layout)
    cute.gemm(mma_tf32, acc, tf32(r_left), tf32(r_right), acc)


def chunk_copies(threads: int, tidx) -> tuple:
    """
    The 16-byte cp.async load and the 16-byte stores of bf16 (rows, 128) and (rows, 64) tiles.
    """
    atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
    atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
    load = cute.make_tiled_copy_tv(atom_load, cute.make_ordered_layout((threads // 16, 16), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
    store_128 = cute.make_tiled_copy_tv(atom_store, cute.make_ordered_layout((threads // 16, 16), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
    store_64 = cute.make_tiled_copy_tv(atom_store, cute.make_ordered_layout((threads // 8, 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
    return load, store_128, store_64


def operand_copies(mma: cute.TiledMma, tidx) -> tuple:
    """
    ldmatrix copies of the A / B operands (plain and transposed) and stmatrix / ldmatrix of C tiles.
    """
    plain = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)
    transposed = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16)
    ld_A, ld_B = cute.make_tiled_copy_A(plain, mma).get_slice(tidx), cute.make_tiled_copy_B(plain, mma).get_slice(tidx)
    ldt_A, ldt_B = cute.make_tiled_copy_A(transposed, mma).get_slice(tidx), cute.make_tiled_copy_B(transposed, mma).get_slice(tidx)
    st_C = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16), mma).get_slice(tidx)
    st_C_t = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16), mma).get_slice(tidx)
    ld_C = cute.make_tiled_copy_C(plain, mma).get_slice(tidx)
    return ld_A, ld_B, ldt_A, ldt_B, st_C, st_C_t, ld_C


@cute.jit
def scale_kv(load: cute.TiledCopy, s_k: cute.Tensor, s_v: cute.Tensor, s_gcum: cute.Tensor, s_beta: cute.Tensor, key_rounding: cutlass.Constexpr[bool]):
    """
    In place, 8 columns per access: v <- bf16(v beta), and k <- bf16(bf16(k beta) 2^g) with
    key_rounding (FLA's recompute_w_u: its k * beta is a bf16 x bf16 product, rounded to bf16) or
    bf16(k beta 2^g) without it (one rounding; k beta is exact in fp32).
    """
    t_k, t_v = load.partition_D(s_k), load.partition_D(s_v)
    coords = load.partition_D(cute.make_identity_tensor((BT, HEAD_DIM)))
    for i in cutlass.range_constexpr(cute.size(t_k, mode=[1])):
        for j in cutlass.range_constexpr(cute.size(t_k, mode=[2])):
            row = coords[0, i, j][0]
            beta, decay = s_beta[row], cute.math.exp2(s_gcum[row], fastmath=True)
            key_beta = t_k[None, i, j].load().to(Float32) * beta
            if const_expr(key_rounding):
                key_beta = key_beta.to(BFloat16).to(Float32)
            t_k[None, i, j].store((key_beta * decay).to(BFloat16))
            t_v[None, i, j].store((t_v[None, i, j].load().to(Float32) * beta).to(BFloat16))
    cute.arch.barrier()


@cute.jit
def multiply_a(mma: cute.TiledMma, thr: cute.ThrMma, ld_A: cute.TiledCopy, ldt_B: cute.TiledCopy, s_a: cute.Tensor, s_x: cute.Tensor):
    """
    s_x <- bf16(A @ s_x), the (HEAD_DIM, BT) transposed view of s_x as the B operand.
    """
    s_x_t = cute.composition(s_x, cute.make_ordered_layout((HEAD_DIM, BT), order=(1, 0)))
    acc = cute.make_rmem_tensor(thr.partition_shape_C((BT, HEAD_DIM)), Float32)
    acc.fill(0.0)
    gemm(mma, acc, thr.make_fragment_A(thr.partition_A(s_a)), thr.make_fragment_B(thr.partition_B(s_x_t)), ld_A.partition_S(s_a), ldt_B.partition_S(s_x_t), ld_A, ldt_B)
    cute.arch.barrier()
    t_x = thr.partition_C(s_x)
    for elem in cutlass.range_constexpr(cute.size(acc)):
        t_x[elem] = BFloat16(acc[elem])
    cute.arch.barrier()


class WyRepresentation:
    """
    One CTA per (chunk, value head), four warps: the chunk-local log2 cumsum of g, A = (I + L)^-1
    with L the strictly lower beta_i k_i.k_j 2^(g_i - g_j), w = A bf16(bf16(k beta) 2^g) and u = A
    bf16(v beta). The inverse: the four 16 x 16 diagonal blocks by fp32 forward substitution, then
    block rows 1..3 as X_ij = -X_ii sum_k L_ik X_kj, each 16 x 16 product one TF32 warp MMA (FLA's
    merge order and its precision on Hopper).
    """

    def __init__(self, H: int, HV: int):
        self.H, self.HV = H, HV

    @cute.jit
    def __call__(self, mK: cute.Tensor, mV: cute.Tensor, mG: cute.Tensor, mBeta: cute.Tensor, mGcum: cute.Tensor, mA: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, stream):
        mma_bf16 = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        mma_tf32 = cute.make_tiled_mma(warp.MmaTF32Op((16, 8, 8)), (1, 1, 1), permutation_mnk=(16, 16, 8))
        chunk_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (BT, HEAD_DIM), (0, 1))
        a_layout = cute.tile_to_shape(get_smem_layout_atom(BFloat16, BT), (BT, BT), (0, 1))
        self.kernel(mK, mV, mG, mBeta, mGcum, mA, mW, mU, mma_bf16, mma_tf32, chunk_layout, a_layout).launch(grid=[mG.shape[0] // BT, self.HV, 1], block=[WY_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mK: cute.Tensor, mV: cute.Tensor, mG: cute.Tensor, mBeta: cute.Tensor, mGcum: cute.Tensor, mA: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mma_bf16: cute.TiledMma, mma_tf32: cute.TiledMma, chunk_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, _ = cute.arch.block_idx()
        key_head_idx = head_idx // (self.HV // self.H)
        fp32_layout = cute.make_layout((BT, BT), stride=(TF32_STRIDE, 1))
        smem = cutlass.utils.SmemAllocator()
        s_inverse = smem.allocate_tensor(Float32, fp32_layout, 1024)  # holds k first
        s_v_region = smem.allocate_tensor(Float32, cute.make_layout(BT * TF32_STRIDE), 1024)
        s_l = smem.allocate_tensor(Float32, fp32_layout, 1024)  # holds k again after the inverse
        s_gcum, s_beta = smem.allocate_tensor(Float32, cute.make_layout(BT), 16), smem.allocate_tensor(Float32, cute.make_layout(BT), 16)
        s_a = smem.allocate_tensor(BFloat16, a_layout, 1024)
        s_k = cute.make_tensor(cute.recast_ptr(s_inverse.iterator, dtype=BFloat16), chunk_layout)
        s_k_again = cute.make_tensor(cute.recast_ptr(s_l.iterator, dtype=BFloat16), chunk_layout)
        s_v = cute.make_tensor(cute.recast_ptr(s_v_region.iterator, dtype=BFloat16), chunk_layout)
        load, store_128, store_64 = chunk_copies(WY_THREADS, tidx)
        ld_A, ld_B, _, ldt_B, _, _, _ = operand_copies(mma_bf16, tidx)
        thr = mma_bf16.get_slice(tidx)
        gK = cute.local_tile(mK[None, key_head_idx, None], (BT, HEAD_DIM), (chunk_idx, 0))
        gV = cute.local_tile(mV[None, head_idx, None], (BT, HEAD_DIM), (chunk_idx, 0))
        gA = cute.local_tile(mA[None, head_idx, None], (BT, BT), (chunk_idx, 0))

        cute.copy(load, load.partition_S(gK), load.partition_D(s_k))
        cute.arch.cp_async_commit_group()
        cute.copy(load, load.partition_S(gV), load.partition_D(s_v))
        cute.arch.cp_async_commit_group()
        self.cumsum_gate(mG, mBeta, mGcum, s_gcum, s_beta, tidx, chunk_idx * BT, head_idx)
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()
        self.compute_l(mma_bf16, thr, ld_A, ld_B, s_k, s_l, s_gcum, s_beta)
        self.invert_diagonal_blocks(s_l, s_inverse, tidx)
        self.merge_blocks(mma_tf32, s_l, s_inverse, tidx)
        self.convert_inverse(s_inverse, s_a, tidx)
        # k again from L2 into L's buffer, which the inverse no longer reads
        cute.copy(load, load.partition_S(gK), load.partition_D(s_k_again))
        cute.arch.cp_async_commit_group()
        cute.arch.barrier()
        cute.copy(store_64, store_64.partition_S(s_a), store_64.partition_D(gA))
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        scale_kv(load, s_k_again, s_v, s_gcum, s_beta, True)
        for which in cutlass.range_constexpr(2):
            s_x = s_v if const_expr(which == 0) else s_k_again
            mOut = mU if const_expr(which == 0) else mW
            multiply_a(mma_bf16, thr, ld_A, ldt_B, s_a, s_x)
            cute.copy(store_128, store_128.partition_S(s_x), store_128.partition_D(cute.local_tile(mOut[None, head_idx, None], (BT, HEAD_DIM), (chunk_idx, 0))))

    @cute.jit
    def cumsum_gate(self, mG: cute.Tensor, mBeta: cute.Tensor, mGcum: cute.Tensor, s_gcum: cute.Tensor, s_beta: cute.Tensor, tidx, row_start, head_idx):
        """
        Warp 0: the chunk-local cumsum of g in log2 units (written to g_cum) and beta, in FLA's
        tl.cumsum order: lane l owns rows l and l + 32, each half is scanned over the lanes
        (shuffles up by 1 .. 16), the second half adds the first half's total, then the scale.
        """
        if tidx < 32:
            row = row_start + tidx
            first, second = Float32(mG[row, head_idx]), Float32(mG[row + 32, head_idx])
            for i in cutlass.range_constexpr(5):
                from_first, from_second = cute.arch.shuffle_sync_up(first, offset=1 << i, mask_and_clamp=0), cute.arch.shuffle_sync_up(second, offset=1 << i, mask_and_clamp=0)
                if tidx >= (1 << i):
                    first, second = first + from_first, second + from_second
            second = second + cute.arch.shuffle_sync(first, offset=31)
            cum_first, cum_second = first * RCP_LN2, second * RCP_LN2
            s_gcum[tidx], s_gcum[tidx + 32] = cum_first, cum_second
            mGcum[row, head_idx], mGcum[row + 32, head_idx] = cum_first, cum_second
            s_beta[tidx], s_beta[tidx + 32] = Float32(mBeta[row, head_idx]), Float32(mBeta[row + 32, head_idx])

    @cute.jit
    def compute_l(self, mma: cute.TiledMma, thr: cute.ThrMma, ld_A: cute.TiledCopy, ld_B: cute.TiledCopy, s_k: cute.Tensor, s_l: cute.Tensor, s_gcum: cute.Tensor, s_beta: cute.Tensor):
        """
        L = strictly lower beta_i k_i.k_j 2^(g_i - g_j), from K K^T on bf16 tensor cores.
        """
        acc_s = cute.make_rmem_tensor(thr.partition_shape_C((BT, BT)), Float32)
        acc_s.fill(0.0)
        gemm(mma, acc_s, thr.make_fragment_A(thr.partition_A(s_k)), thr.make_fragment_B(thr.partition_B(s_k)), ld_A.partition_S(s_k), ld_B.partition_S(s_k), ld_A, ld_B)
        coords, t_l = thr.partition_C(cute.make_identity_tensor((BT, BT))), thr.partition_C(s_l)
        for elem in cutlass.range_constexpr(cute.size(acc_s)):
            row, col = coords[elem]
            value = Float32(0.0)
            if row > col:
                value = acc_s[elem] * cute.math.exp2(s_gcum[row] - s_gcum[col], fastmath=True) * s_beta[row]
            t_l[elem] = value
        cute.arch.barrier()

    @cute.jit
    def invert_diagonal_blocks(self, s_l: cute.Tensor, s_inverse: cute.Tensor, tidx):
        """
        Warp w inverts diagonal block w of I + L, lane c < 16 owning column c, in FLA's order: X
        starts as -L; for rows i = 2 .. 15, X_ic = a_c + sum_k a_k X_kc with a the row i of -L (k <
        i), the sum over the 16 rows k as FLA's reduction on the MMA layout runs it (rows k and k +
        8 first, then partners by row-group bits 4, 2, 1); then X += I.
        """
        warp_idx, lane = tidx // 32, tidx % 32
        base = warp_idx * 16
        if lane < 16:
            column = [-s_l[base + r, base + lane] for r in range(16)]
            for i in cutlass.range_constexpr(2, 16):
                row_i = [-s_l[base + i, base + k] if k < i else Float32(0.0) for k in range(16)]
                pairs = [cute.math.fma(row_i[g], column[g], row_i[g + 8] * column[g + 8]) for g in range(8)]
                total = ((pairs[0] + pairs[4]) + (pairs[2] + pairs[6])) + ((pairs[1] + pairs[5]) + (pairs[3] + pairs[7]))
                own = Float32(0.0)
                if lane < i:
                    own = -s_l[base + i, base + lane]
                column[i] = own + total
            for r in cutlass.range_constexpr(16):
                identity = Float32(0.0)
                if lane == r:
                    identity = Float32(1.0)
                s_inverse[base + r, base + lane] = column[r] + identity
        cute.arch.barrier()

    @cute.jit
    def merge_blocks(self, mma_tf32: cute.TiledMma, s_l: cute.Tensor, s_inverse: cute.Tensor, tidx):
        """
        Block rows 1..3, warp j computing block (i, j), in FLA's association: adjacent blocks as
        X_ij = -(X_ii L_ij) X_jj, the others as X_ij = -X_ii (sum_k L_ik X_kj) with the sum
        accumulated in k order (FLA's dots fuse into one accumulator); the inner product is staged
        in L's unused upper block (j, i).
        """
        warp_idx, lane = tidx // 32, tidx % 32
        thr = mma_tf32.get_slice(lane)
        block = lambda s, i, j: cute.local_tile(s, (16, 16), (i, j))
        for block_row in cutlass.range_constexpr(1, 4):
            for block_col in cutlass.range_constexpr(block_row):
                if warp_idx == block_col:
                    acc = cute.make_rmem_tensor(thr.partition_shape_C((16, 16)), Float32)
                    acc.fill(0.0)
                    if const_expr(block_col == block_row - 1):
                        block_product(thr, mma_tf32, acc, block(s_inverse, block_row, block_row), block(s_l, block_row, block_col), True)
                    else:
                        for block_mid in cutlass.range_constexpr(block_col, block_row):
                            block_product(thr, mma_tf32, acc, block(s_l, block_row, block_mid), block(s_inverse, block_mid, block_col), True)
                    t_inner = thr.partition_C(block(s_l, block_col, block_row))
                    for elem in cutlass.range_constexpr(cute.size(acc)):
                        t_inner[elem] = acc[elem]
                    cute.arch.sync_warp()
                    acc.fill(0.0)
                    if const_expr(block_col == block_row - 1):
                        block_product(thr, mma_tf32, acc, block(s_l, block_col, block_row), block(s_inverse, block_col, block_col), False)
                    else:
                        block_product(thr, mma_tf32, acc, block(s_inverse, block_row, block_row), block(s_l, block_col, block_row), False)
                    t_block = thr.partition_C(block(s_inverse, block_row, block_col))
                    for elem in cutlass.range_constexpr(cute.size(acc)):
                        t_block[elem] = -acc[elem]
            cute.arch.barrier()

    @cute.jit
    def convert_inverse(self, s_inverse: cute.Tensor, s_a: cute.Tensor, tidx):
        """
        A = bf16(X), zero above the diagonal blocks (which the merge never writes).
        """
        for i in cutlass.range_constexpr(BT * BT // WY_THREADS):
            idx = i * WY_THREADS + tidx
            row, col = idx // BT, idx % BT
            value = Float32(0.0)
            if row // 16 >= col // 16:
                value = s_inverse[row, col]
            s_a[row, col] = BFloat16(value)


class ChunkStateOutput:
    """
    The chunk-state recurrence fused with the output, one CTA per (STATE_BV state columns, value
    head). The recurrence warpgroup, chunk by chunk, with h the HEAD_DIM x STATE_BV fp32 state in
    registers (h0 or zero at the start):
        h_n = bf16(h);  v = u_n - w_n h_n;  v_new = bf16(v)
        h = h 2^(g_last) + k_n^T bf16(v 2^(g_last - g))
    and issues the loads of k, w, u, g (STATE_STAGES chunks in flight). The output warpgroup trails
    it by one chunk: o_n = bf16((q_n h_n) 2^g scale + bf16(mask(q_n k_n^T 2^(g_i - g_j))) v_new
    scale). h_n and v_new pass through double-buffered shared memory, stored transposed (STATE_BV
    rows) so every B operand is a plain ldmatrix; FULL[n % 2] says chunk n is ready, EMPTY[n % 2]
    that the output warpgroup is done with it. Neither h nor v_new is written to global memory.
    """

    def __init__(self, H: int, HV: int, has_h0: bool, has_ht: bool):
        self.H, self.HV, self.has_h0, self.has_ht = H, HV, has_h0, has_ht
        self.scale = HEAD_DIM ** -0.5
        self.recurrence_threads, self.output_threads = STATE_THREADS, STATE_THREADS

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGcum: cute.Tensor, mO: cute.Tensor, mH0: cute.Tensor, mHt: cute.Tensor, stream):
        mma = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16))
        layouts = (
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (BT, HEAD_DIM, STATE_STAGES), (0, 1, 2)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (BT, HEAD_DIM, 2), (0, 1, 2)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_BV), (BT, STATE_BV, STATE_STAGES), (0, 1, 2)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, HEAD_DIM), (STATE_BV, HEAD_DIM, 2), (0, 1, 2)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, BT), (STATE_BV, BT, 2), (0, 1, 2)),
            cute.tile_to_shape(get_smem_layout_atom(BFloat16, STATE_BV), (BT, STATE_BV), (0, 1)),
        )
        self.kernel(mQ, mK, mW, mU, mGcum, mO, mH0, mHt, mma, *layouts).launch(grid=[HEAD_DIM // STATE_BV, self.HV, 1], block=[self.recurrence_threads + self.output_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGcum: cute.Tensor, mO: cute.Tensor, mH0: cute.Tensor, mHt: cute.Tensor, mma: cute.TiledMma, ring_layout: cute.ComposedLayout, q_layout: cute.ComposedLayout, u_layout: cute.ComposedLayout, h_layout: cute.ComposedLayout, v_layout: cute.ComposedLayout, o_layout: cute.ComposedLayout):
        tid, _, _ = cute.arch.thread_idx()
        col_tile, head_idx, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s_k, s_w = smem.allocate_tensor(BFloat16, ring_layout, 1024), smem.allocate_tensor(BFloat16, ring_layout, 1024)
        s_q, s_u = smem.allocate_tensor(BFloat16, q_layout, 1024), smem.allocate_tensor(BFloat16, u_layout, 1024)
        s_h, s_vnew = smem.allocate_tensor(BFloat16, h_layout, 1024), smem.allocate_tensor(BFloat16, v_layout, 1024)
        s_vscaled, s_o = smem.allocate_tensor(BFloat16, v_layout, 1024), smem.allocate_tensor(BFloat16, o_layout, 1024)
        s_gcum = smem.allocate_tensor(Float32, cute.make_layout((BT, STATE_STAGES), stride=(1, BT)), 16)
        buffers = (s_k, s_w, s_q, s_u, s_h, s_vnew, s_vscaled, s_o, s_gcum)
        if tid < self.recurrence_threads:
            self.recurrence_loop(mK, mW, mU, mGcum, mH0, mHt, mma, buffers, tid, col_tile, head_idx)
        else:
            self.output_loop(mQ, mO, mma, buffers, tid - self.recurrence_threads, col_tile, head_idx, mGcum.shape[0] // BT)

    @cute.jit
    def issue_chunk(self, load: cute.TiledCopy, load_u: cute.TiledCopy, load_gate: cute.TiledCopy, sources: tuple, targets: tuple, chunk: int, stage: int, tidx):
        """
        cp.async of chunk's k, w, u and g into ring stage, one commit group.
        """
        t_k_src, t_w_src, t_u_src, t_g_src = sources
        t_k_dst, t_w_dst, t_u_dst, t_g_dst = targets
        cute.copy(load, t_k_src[None, None, None, chunk], t_k_dst[None, None, None, stage])
        cute.copy(load, t_w_src[None, None, None, chunk], t_w_dst[None, None, None, stage])
        cute.copy(load_u, t_u_src[None, None, None, chunk], t_u_dst[None, None, None, stage])
        if tidx < BT:
            cute.copy(load_gate, t_g_src[None, None, chunk], t_g_dst[None, None, stage])
        cute.arch.cp_async_commit_group()

    @cute.jit
    def recurrence_loop(self, mK: cute.Tensor, mW: cute.Tensor, mU: cute.Tensor, mGcum: cute.Tensor, mH0: cute.Tensor, mHt: cute.Tensor, mma: cute.TiledMma, buffers: tuple, tidx, col_tile, head_idx):
        s_k, s_w, _, s_u, s_h, s_vnew, s_vscaled, _, s_gcum = buffers
        num_chunks = mGcum.shape[0] // BT
        key_head_idx = head_idx // (self.HV // self.H)
        threads = self.recurrence_threads
        load, _, _ = chunk_copies(threads, tidx)
        atom_load = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
        load_u = cute.make_tiled_copy_tv(atom_load, cute.make_ordered_layout((threads // (STATE_BV // 8), STATE_BV // 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        load_gate = cute.make_tiled_copy_tv(cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32), cute.make_layout(BT), cute.make_layout(1)).get_slice(tidx % BT)
        gK = cute.local_tile(mK[None, key_head_idx, None], (BT, HEAD_DIM), (None, 0))
        gW = cute.local_tile(mW[None, head_idx, None], (BT, HEAD_DIM), (None, 0))
        gU = cute.local_tile(mU[None, head_idx, None], (BT, STATE_BV), (None, col_tile))
        gGcum = cute.local_tile(mGcum[None, head_idx], (BT,), (None,))
        sources = (load.partition_S(gK), load.partition_S(gW), load_u.partition_S(gU), load_gate.partition_S(gGcum))
        targets = (load.partition_D(s_k), load.partition_D(s_w), load_u.partition_D(s_u), load_gate.partition_D(s_gcum))
        # an empty commit group stands in for a missing chunk, so the group count stays fixed
        for stage in cutlass.range_constexpr(STATE_STAGES - 1):
            if stage < num_chunks:
                self.issue_chunk(load, load_u, load_gate, sources, targets, stage, stage, tidx)
            else:
                cute.arch.cp_async_commit_group()
        thr = mma.get_slice(tidx)
        ld_A, ld_B, ldt_A, _, _, st_C_t, ld_C = operand_copies(mma, tidx)
        acc_h = cute.make_rmem_tensor(thr.partition_shape_C((HEAD_DIM, STATE_BV)), Float32)  # h, (K rows, BV columns)
        acc_h.fill(0.0)
        h_coords = thr.partition_C(cute.make_identity_tensor((HEAD_DIM, STATE_BV)))
        if const_expr(self.has_h0):
            for elem in cutlass.range_constexpr(cute.size(acc_h)):
                acc_h[elem] = mH0[head_idx, h_coords[elem][0], col_tile * STATE_BV + h_coords[elem][1]]
        v_coords = thr.partition_C(cute.make_identity_tensor((BT, STATE_BV)))
        for chunk in cutlass.range(num_chunks, unroll=1):
            stage, parity = chunk % STATE_STAGES, chunk % 2
            s_k_n, s_w_n, s_u_n = s_k[None, None, stage], s_w[None, None, stage], s_u[None, None, stage]
            s_h_n, s_vnew_n, s_vscaled_n = s_h[None, None, parity], s_vnew[None, None, parity], s_vscaled[None, None, 0]
            # h_n into s_h[parity]: the output warpgroup released it at chunk - 2
            r_h = cute.make_fragment_like(acc_h, BFloat16)
            r_h.store(acc_h.load().to(BFloat16))
            cute.copy(st_C_t, st_C_t.retile(r_h), st_C_t.partition_D(cute.composition(s_h_n, cute.make_ordered_layout((HEAD_DIM, STATE_BV), order=(1, 0)))))
            cute.arch.cp_async_wait_group(STATE_STAGES - 2)
            cute.arch.barrier(barrier_id=BAR_RECURRENCE, number_of_threads=threads)
            acc_v = cute.make_rmem_tensor(thr.partition_shape_C((BT, STATE_BV)), Float32)
            acc_v.fill(0.0)
            gemm(mma, acc_v, thr.make_fragment_A(thr.partition_A(s_w_n)), thr.make_fragment_B(thr.partition_B(s_h_n)), ld_A.partition_S(s_w_n), ld_B.partition_S(s_h_n), ld_A, ld_B)
            gate_last = s_gcum[BT - 1, stage]
            r_u = cute.make_fragment_like(acc_v, BFloat16)
            cute.copy(ld_C, ld_C.partition_S(s_u_n), ld_C.retile(r_u))
            v_value = r_u.load().to(Float32) - acc_v.load()
            r_vnew = cute.make_fragment_like(acc_v, BFloat16)
            r_vnew.store(v_value.to(BFloat16))
            acc_v.store(v_value)
            for elem in cutlass.range_constexpr(cute.size(acc_v)):
                acc_v[elem] = acc_v[elem] * cute.math.exp2(gate_last - s_gcum[v_coords[elem][0], stage], fastmath=True)
            r_vscaled = cute.make_fragment_like(acc_v, BFloat16)
            r_vscaled.store(acc_v.load().to(BFloat16))
            cute.copy(st_C_t, st_C_t.retile(r_vnew), st_C_t.partition_D(cute.composition(s_vnew_n, cute.make_ordered_layout((BT, STATE_BV), order=(1, 0)))))
            cute.copy(st_C_t, st_C_t.retile(r_vscaled), st_C_t.partition_D(cute.composition(s_vscaled_n, cute.make_ordered_layout((BT, STATE_BV), order=(1, 0)))))
            decay = cute.math.exp2(gate_last, fastmath=True)
            for elem in cutlass.range_constexpr(cute.size(acc_h)):
                acc_h[elem] = acc_h[elem] * decay
            cute.arch.barrier(barrier_id=BAR_RECURRENCE, number_of_threads=threads)
            cute.arch.barrier_arrive(barrier_id=BAR_FULL + parity, number_of_threads=threads + self.output_threads)
            s_k_n_t = cute.composition(s_k_n, cute.make_ordered_layout((HEAD_DIM, BT), order=(1, 0)))
            gemm(mma, acc_h, thr.make_fragment_A(thr.partition_A(s_k_n_t)), thr.make_fragment_B(thr.partition_B(s_vscaled_n)), ldt_A.partition_S(s_k_n_t), ld_B.partition_S(s_vscaled_n), ldt_A, ld_B)
            # the output warpgroup is done with chunk - 1: its ring stage and buffers are free
            if chunk > 0:
                cute.arch.barrier(barrier_id=BAR_EMPTY + 1 - parity, number_of_threads=threads + self.output_threads)
            upcoming = chunk + STATE_STAGES - 1
            if upcoming < num_chunks:
                self.issue_chunk(load, load_u, load_gate, sources, targets, upcoming, upcoming % STATE_STAGES, tidx)
            else:
                cute.arch.cp_async_commit_group()
            cute.arch.barrier(barrier_id=BAR_RECURRENCE, number_of_threads=threads)
        if const_expr(self.has_ht):
            for elem in cutlass.range_constexpr(cute.size(acc_h)):
                mHt[head_idx, h_coords[elem][0], col_tile * STATE_BV + h_coords[elem][1]] = acc_h[elem]

    @cute.jit
    def output_loop(self, mQ: cute.Tensor, mO: cute.Tensor, mma: cute.TiledMma, buffers: tuple, tidx, col_tile, head_idx, num_chunks):
        s_k, _, s_q, _, s_h, s_vnew, _, s_o, s_gcum = buffers
        key_head_idx = head_idx // (self.HV // self.H)
        threads = self.output_threads
        load, _, _ = chunk_copies(threads, tidx)
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        store_o = cute.make_tiled_copy_tv(atom_store, cute.make_ordered_layout((threads // (STATE_BV // 8), STATE_BV // 8), order=(1, 0)), cute.make_layout((1, 8))).get_slice(tidx)
        thr = mma.get_slice(tidx)
        ld_A, ld_B, _, _, st_C, _, _ = operand_copies(mma, tidx)
        gQ = cute.local_tile(mQ[None, key_head_idx, None], (BT, HEAD_DIM), (None, 0))
        gO = cute.local_tile(mO[None, head_idx, None], (BT, STATE_BV), (None, col_tile))
        t_q_src, t_q_dst = load.partition_S(gQ), load.partition_D(s_q)
        s_coords = thr.partition_C(cute.make_identity_tensor((BT, BT)))
        o_coords = thr.partition_C(cute.make_identity_tensor((BT, STATE_BV)))
        cute.copy(load, t_q_src[None, None, None, 0], t_q_dst[None, None, None, 0])
        cute.arch.cp_async_commit_group()
        for chunk in cutlass.range(num_chunks, unroll=1):
            stage, parity = chunk % STATE_STAGES, chunk % 2
            if chunk + 1 < num_chunks:
                cute.copy(load, t_q_src[None, None, None, chunk + 1], t_q_dst[None, None, None, 1 - parity])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier(barrier_id=BAR_FULL + parity, number_of_threads=self.recurrence_threads + threads)
            s_q_n, s_k_n, s_h_n, s_vnew_n = s_q[None, None, parity], s_k[None, None, stage], s_h[None, None, parity], s_vnew[None, None, parity]
            acc_o = cute.make_rmem_tensor(thr.partition_shape_C((BT, STATE_BV)), Float32)
            acc_o.fill(0.0)
            r_q = thr.make_fragment_A(thr.partition_A(s_q_n))
            gemm(mma, acc_o, r_q, thr.make_fragment_B(thr.partition_B(s_h_n)), ld_A.partition_S(s_q_n), ld_B.partition_S(s_h_n), ld_A, ld_B)
            acc_s = cute.make_rmem_tensor(thr.partition_shape_C((BT, BT)), Float32)
            acc_s.fill(0.0)
            gemm(mma, acc_s, r_q, thr.make_fragment_B(thr.partition_B(s_k_n)), ld_A.partition_S(s_q_n), ld_B.partition_S(s_k_n), ld_A, ld_B)
            for elem in cutlass.range_constexpr(cute.size(acc_s)):
                row, col = s_coords[elem]
                value = Float32(0.0)
                if row >= col:
                    value = acc_s[elem] * cute.math.exp2(s_gcum[row, stage] - s_gcum[col, stage], fastmath=True)
                acc_s[elem] = value
            r_p = cute.make_fragment_like(acc_s, BFloat16)  # P stays in registers as the A operand
            r_p.store(acc_s.load().to(BFloat16))
            acc_intra = cute.make_rmem_tensor(thr.partition_shape_C((BT, STATE_BV)), Float32)
            acc_intra.fill(0.0)
            gemm_rs(mma, acc_intra, reshape_acc_to_frgA(r_p), thr.make_fragment_B(thr.partition_B(s_vnew_n)), ld_B.partition_S(s_vnew_n), ld_B)
            for elem in cutlass.range_constexpr(cute.size(acc_o)):
                acc_o[elem] = acc_o[elem] * cute.math.exp2(s_gcum[o_coords[elem][0], stage], fastmath=True)
            cute.arch.barrier_arrive(barrier_id=BAR_EMPTY + parity, number_of_threads=self.recurrence_threads + threads)
            r_o = cute.make_fragment_like(acc_o, BFloat16)
            r_o.store((acc_o.load() * self.scale + acc_intra.load() * self.scale).to(BFloat16))
            cute.copy(st_C, st_C.retile(r_o), st_C.partition_D(s_o))
            cute.arch.barrier(barrier_id=BAR_OUTPUT, number_of_threads=threads)
            cute.copy(store_o, store_o.partition_S(s_o), store_o.partition_D(gO[None, None, chunk]))
            cute.arch.barrier(barrier_id=BAR_OUTPUT, number_of_threads=threads)


COMPILED: dict[tuple, object] = {}


def compile_wy(H: int, HV: int) -> object:
    T = cute.sym_int()
    key_shape, value_shape, gate_shape = (T, H, HEAD_DIM), (T, HV, HEAD_DIM), (T, HV)
    key_stride, value_stride, gate_stride = (H * HEAD_DIM, HEAD_DIM, 1), (HV * HEAD_DIM, HEAD_DIM, 1), (HV, 1)
    keys, values = fake(BFloat16, key_shape, key_stride), fake(BFloat16, value_shape, value_stride)
    gate, beta = fake(Float32, gate_shape, gate_stride, 4), fake(BFloat16, gate_shape, gate_stride, 2)
    a = fake(BFloat16, (T, HV, BT), (HV * BT, BT, 1))
    return cute.compile(WyRepresentation(H, HV), keys, values, gate, beta, gate, a, values, values, fake_stream(), options="--enable-tvm-ffi")


def compile_state_output(H: int, HV: int, has_h0: bool, has_ht: bool) -> object:
    T = cute.sym_int()
    keys = fake(BFloat16, (T, H, HEAD_DIM), (H * HEAD_DIM, HEAD_DIM, 1))
    values = fake(BFloat16, (T, HV, HEAD_DIM), (HV * HEAD_DIM, HEAD_DIM, 1))
    gate = fake(Float32, (T, HV), (HV, 1), 4)
    state = fake(Float32, (HV, HEAD_DIM, HEAD_DIM), (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1))
    return cute.compile(ChunkStateOutput(H, HV, has_h0, has_ht), keys, keys, values, values, gate, values, state if has_h0 else None, state if has_ht else None, fake_stream(), options="--enable-tvm-ffi")


def chunk_gdr_fwd_cute(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    chunk_gdr_fwd_pinned_sm90 on the CuTe kernels: B == 1 and T a multiple of BT.
    """
    _, T, H, _ = k.shape
    HV = v.shape[2]
    g_cum = torch.empty(1, T, HV, device=k.device, dtype=torch.float32)
    A = torch.empty(1, T, HV, BT, device=k.device, dtype=k.dtype)
    w, u, o = k.new_empty(1, T, HV, HEAD_DIM), torch.empty_like(v), torch.empty_like(v)
    key = ("wy_representation", H, HV, k.device.index)
    if key not in COMPILED:
        COMPILED[key] = compile_wy(H, HV)
    COMPILED[key](k[0], v[0], g[0], beta[0], g_cum[0], A[0], w[0], u[0])
    key = ("chunk_state_output", H, HV, k.device.index)
    if key not in COMPILED:
        COMPILED[key] = compile_state_output(H, HV, False, False)
    COMPILED[key](q[0], k[0], w[0], u[0], g_cum[0], o[0], None, None)
    return o, g_cum, A


def chunk_gdr_fwd_pinned_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dense chunked gated delta rule forward (chunk size 64, scale K ** -0.5, no initial or final
    state): fresh (o, g_cum, A).

    q, k: bf16 (B, T, H, K) contiguous, unrepeated key heads. v: bf16 (B, T, HV, V). g: fp32 (B, T,
    HV), the raw log gate. beta: bf16 (B, T, HV). HV is a multiple of H; value head j reads key head
    j // (HV / H). Inputs are not modified.

    o: bf16 (B, T, HV, V). g_cum: fp32 (B, T, HV), the chunk-local cumulative gate. A: bf16 (B, T,
    HV, 64), the solved WY matrix. g_cum and A are inputs of chunk_gdr_bwd_pinned_sm90. With T a
    multiple of 64 the CuTe kernels, matching FLA 0.5.2's chunk_gated_delta_rule_fwd on q / k
    repeat-interleaved to HV heads to bf16 rounding; otherwise FLA's kernels on the recorded launch
    configs, bit-identical to it while those are its autotuner's picks (CONFIGS, FLA's on H200).
    Deterministic. Only where gated_delta_rule_pinned_supported is True.
    """
    B, T, H, K, HV, V = *k.shape, *v.shape[2:]
    if T % BT == 0:
        return chunk_gdr_fwd_cute(q, k, v, g, beta)
    num_chunks = triton.cdiv(T, BT)
    g_cum = chunk_cumsum(g, reverse=False)
    # The fused kkt + solve stores only the triangular blocks: A must start zeroed.
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    launch("kkt_solve", (num_chunks, B * HV), k=k, g=g_cum, beta=beta, A=A, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, USE_G=True, IS_VARLEN=False, **CONFIGS["kkt_solve"])
    w, u = recompute_w_u(k, v, beta, A, g_cum)
    h, v_new = compute_chunk_states(k, w, u, g_cum)
    o = torch.empty_like(v_new)
    config = CONFIGS["fwd_o"]
    launch("fwd_o", (triton.cdiv(V, config["BV"]), num_chunks, B * HV), q=q, k=k, v=v_new, h=h, g=g_cum, g_gamma=None, o=o, cu_seqlens=None, chunk_indices=None, scale=K**-0.5, T=T, H=H, HV=HV, K=K, V=V, BT=BT, USE_G=True, USE_G_GAMMA=False, STATE_V_FIRST=False, IS_VARLEN=False, **config)
    return o, g_cum, A
