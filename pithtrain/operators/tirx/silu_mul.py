"""
TIRx implementation of the fused SwiGLU-style ``silu(gate) * up`` autograd
function (a port of the Triton kernels in ``silu_mul.py``).

The numerics mirror the Triton version exactly:

- forward fuses ``silu(gate) * up`` into one kernel (two loads, one store), and
- backward fuses ``grad_gate`` / ``grad_up`` into one kernel that recomputes
  ``silu`` / ``sigmoid`` on the fly (three loads, two stores).

Only ``gate`` and ``up`` are saved for backward - ``silu(gate)`` is not stored.

Launch / data mapping
---------------------
The op is pure element-wise, so the inputs are flattened to a **1-D array of
``n`` elements** and processed by a **flat 1-D grid**. Each CTA owns a fixed tile
of ``ELEMS_PER_CTA`` contiguous elements, split across ``BLOCK`` threads in
128-bit-wide vectors (``VECTOR_SIZE = 16 / elem_bytes`` elements, e.g. 8 for
bf16). Each thread processes ``STEPS = ELEMS_PER_CTA / (BLOCK * VECTOR_SIZE)``
vectors with an interleaved load -> compute -> store per step (no prefetch;
register buffers are one vector wide). The grid is ``ceil(n / ELEMS_PER_CTA)``;
the **last CTA may own fewer than ``ELEMS_PER_CTA`` elements**, guarded by
``col < n``. Because ``n`` is a multiple of ``VECTOR_SIZE`` (the flattened tensor
is row-major with a vector-divisible last dim), a guarded 128-bit vector never
overruns the buffer.

``(BLOCK, ELEMS_PER_CTA)`` is chosen by total element count ``n`` (see
``_select_config``): small problems favor wide CTAs, large problems favor small
tiles. Nothing depends on ``hidden`` or ``n_rows``, so a single compiled kernel
per ``(dtype, BLOCK, ELEMS_PER_CTA)`` serves every shape that maps to it.
"""

from __future__ import annotations

import torch
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

# Torch dtype -> (TIRx buffer dtype string, bytes per element).
_DTYPE_MAP = {
    torch.float16: ("float16", 2),
    torch.bfloat16: ("bfloat16", 2),
    torch.float32: ("float32", 4),
}

_VEC_BYTES = 16  # 128-bit vectorized global access

# Flat-grid configs as (BLOCK threads, ELEMS_PER_CTA), selected by element count.
_CONFIG_C = (256, 2048)  # small problems
_CONFIG_MID = (128, 2048)  # medium problems ("current")
_CONFIG_B = (128, 1024)  # large problems


def _select_config(n: int) -> tuple[int, int]:
    """Pick (BLOCK, ELEMS_PER_CTA) for a flat array of ``n`` elements."""
    if n < 4194304:
        return _CONFIG_C
    if n < 16777216:
        return _CONFIG_MID
    return _CONFIG_B


def _launch_params(block: int, tile: int, elem_bytes: int) -> tuple[int, int, int]:
    """Return (VECTOR_SIZE, BLOCK_SIZE, STEPS) for a ``tile``-element CTA."""
    vec = _VEC_BYTES // elem_bytes
    assert tile % (block * vec) == 0, "tile must divide evenly into thread vectors"
    return vec, block, tile // (block * vec)


def _build_fwd_kernel(dtype: str, block: int, tile: int, elem_bytes: int):
    """TIRx prim_func for ``out = silu(gate) * up``, flat 1-D grid."""
    VEC, BLOCK, STEPS = _launch_params(block, tile, elem_bytes)
    TILE = BLOCK * VEC * STEPS

    @T.prim_func
    def silu_mul_fwd(gate_ptr: T.handle, up_ptr: T.handle, out_ptr: T.handle):
        n = T.int32()
        gate_g = T.match_buffer(gate_ptr, [n], dtype, scope="global")
        up_g = T.match_buffer(up_ptr, [n], dtype, scope="global")
        out_g = T.match_buffer(out_ptr, [n], dtype, scope="global")
        T.device_entry()
        cta = T.cta_id([(n + TILE - 1) // TILE])
        tx = T.thread_id([BLOCK])

        gate_v = T.alloc_buffer((VEC,), dtype, scope="local")
        up_v = T.alloc_buffer((VEC,), dtype, scope="local")
        out_v = T.alloc_buffer((VEC,), dtype, scope="local")
        gate_f = T.alloc_buffer((VEC,), "float32", scope="local")
        up_f = T.alloc_buffer((VEC,), "float32", scope="local")

        for s in T.unroll(STEPS):
            col = T.meta_var(cta * TILE + (s * BLOCK + tx) * VEC)
            if col < n:
                Tx.copy(gate_v[:], gate_g[col : col + VEC])
                Tx.copy(up_v[:], up_g[col : col + VEC])
                Tx.cast(gate_f[:], gate_v[:])
                Tx.cast(up_f[:], up_v[:])
                for v in T.unroll(VEC):
                    sig = T.float32(1.0) / (T.float32(1.0) + T.exp(-gate_f[v]))
                    out_v[v] = T.cast(gate_f[v] * sig * up_f[v], dtype)
                Tx.copy(out_g[col : col + VEC], out_v[:])

    return silu_mul_fwd


def _build_bwd_kernel(dtype: str, block: int, tile: int, elem_bytes: int):
    """TIRx prim_func for ``grad_gate`` / ``grad_up``, flat 1-D grid.

    With ``s = silu(gate)`` and ``sigma = sigmoid(gate)``:
        grad_up   = grad_out * s
        grad_gate = grad_out * up * sigma * (1 + gate - s)
    """
    VEC, BLOCK, STEPS = _launch_params(block, tile, elem_bytes)
    TILE = BLOCK * VEC * STEPS

    @T.prim_func
    def silu_mul_bwd(
        grad_out_ptr: T.handle,
        gate_ptr: T.handle,
        up_ptr: T.handle,
        grad_gate_ptr: T.handle,
        grad_up_ptr: T.handle,
    ):
        n = T.int32()
        grad_out_g = T.match_buffer(grad_out_ptr, [n], dtype, scope="global")
        gate_g = T.match_buffer(gate_ptr, [n], dtype, scope="global")
        up_g = T.match_buffer(up_ptr, [n], dtype, scope="global")
        grad_gate_g = T.match_buffer(grad_gate_ptr, [n], dtype, scope="global")
        grad_up_g = T.match_buffer(grad_up_ptr, [n], dtype, scope="global")
        T.device_entry()
        cta = T.cta_id([(n + TILE - 1) // TILE])
        tx = T.thread_id([BLOCK])

        grad_out_v = T.alloc_buffer((VEC,), dtype, scope="local")
        gate_v = T.alloc_buffer((VEC,), dtype, scope="local")
        up_v = T.alloc_buffer((VEC,), dtype, scope="local")
        grad_gate_v = T.alloc_buffer((VEC,), dtype, scope="local")
        grad_up_v = T.alloc_buffer((VEC,), dtype, scope="local")
        grad_out_f = T.alloc_buffer((VEC,), "float32", scope="local")
        gate_f = T.alloc_buffer((VEC,), "float32", scope="local")
        up_f = T.alloc_buffer((VEC,), "float32", scope="local")

        for s in T.unroll(STEPS):
            col = T.meta_var(cta * TILE + (s * BLOCK + tx) * VEC)
            if col < n:
                Tx.copy(grad_out_v[:], grad_out_g[col : col + VEC])
                Tx.copy(gate_v[:], gate_g[col : col + VEC])
                Tx.copy(up_v[:], up_g[col : col + VEC])
                Tx.cast(grad_out_f[:], grad_out_v[:])
                Tx.cast(gate_f[:], gate_v[:])
                Tx.cast(up_f[:], up_v[:])
                for v in T.unroll(VEC):
                    sig = T.float32(1.0) / (T.float32(1.0) + T.exp(-gate_f[v]))
                    silu = gate_f[v] * sig
                    grad_up_v[v] = T.cast(grad_out_f[v] * silu, dtype)
                    grad_gate_v[v] = T.cast(
                        grad_out_f[v] * up_f[v] * sig * (T.float32(1.0) + gate_f[v] - silu),
                        dtype,
                    )
                Tx.copy(grad_gate_g[col : col + VEC], grad_gate_v[:])
                Tx.copy(grad_up_g[col : col + VEC], grad_up_v[:])

    return silu_mul_bwd


def _compile(func):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    return tvm.compile(mod, target=target, tir_pipeline="tirx")


# Compiled-executable caches, keyed by (TIRx dtype string, BLOCK, ELEMS_PER_CTA).
_FWD_CACHE: dict[tuple[str, int, int], object] = {}
_BWD_CACHE: dict[tuple[str, int, int], object] = {}


def _fwd_exec(dtype: str, block: int, tile: int, elem_bytes: int):
    key = (dtype, block, tile)
    ex = _FWD_CACHE.get(key)
    if ex is None:
        ex = _compile(_build_fwd_kernel(dtype, block, tile, elem_bytes))
        _FWD_CACHE[key] = ex
    return ex


def _bwd_exec(dtype: str, block: int, tile: int, elem_bytes: int):
    key = (dtype, block, tile)
    ex = _BWD_CACHE.get(key)
    if ex is None:
        ex = _compile(_build_bwd_kernel(dtype, block, tile, elem_bytes))
        _BWD_CACHE[key] = ex
    return ex


class _SiLUMulTirx(torch.autograd.Function):
    """Fused SwiGLU activation (TIRx backend) that saves only ``gate`` and ``up``."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
        assert gate.dtype == up.dtype, f"dtype mismatch: {gate.dtype} vs {up.dtype}"
        assert gate.is_contiguous(), "gate must be contiguous"
        assert up.is_contiguous(), "up must be contiguous"
        assert gate.dtype in _DTYPE_MAP, f"unsupported dtype: {gate.dtype}"
        ctx.save_for_backward(gate, up)
        out = torch.empty_like(gate)
        dtype, elem_bytes = _DTYPE_MAP[gate.dtype]
        n = gate.numel()
        assert n % (_VEC_BYTES // elem_bytes) == 0, "numel must be vector-divisible"
        block, tile = _select_config(n)
        _fwd_exec(dtype, block, tile, elem_bytes)(
            gate.reshape(-1),
            up.reshape(-1),
            out.reshape(-1),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        assert grad_output.shape == gate.shape, (
            f"grad_output shape {grad_output.shape} != gate shape {gate.shape}"
        )
        assert grad_output.dtype == gate.dtype, (
            f"grad_output dtype {grad_output.dtype} != gate dtype {gate.dtype}"
        )
        grad_output = grad_output.contiguous()
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        dtype, elem_bytes = _DTYPE_MAP[gate.dtype]
        n = gate.numel()
        block, tile = _select_config(n)
        _bwd_exec(dtype, block, tile, elem_bytes)(
            grad_output.reshape(-1),
            gate.reshape(-1),
            up.reshape(-1),
            grad_gate.reshape(-1),
            grad_up.reshape(-1),
        )
        return grad_gate, grad_up


def silu_mul_tirx(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up`` with a single-kernel backward (TIRx backend).

    Parameters
    ----------
    gate, up : torch.Tensor
        Same-shape, same-dtype tensors produced by the gate and up projections
        of a SwiGLU-style MLP.

    Returns
    -------
    torch.Tensor
        Element-wise ``silu(gate) * up`` in the same dtype as the inputs.
    """
    return _SiLUMulTirx.apply(gate, up)
