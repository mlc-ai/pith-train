"""Test the correctness of ring attention under context parallelism."""

from dataclasses import dataclass, fields

import pytest
import torch

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx
from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.ring_attention import ring_attention_func
from tests.utilities import cosine_error, launch


@dataclass
class Request:
    B: int
    S: int
    HQ: int
    HK: int
    D: int
    atol: float = 1e-5


@dataclass
class Result:
    out: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


def extract_zigzag(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """
    Extract this rank's zigzag-local slice along the sequence dim (dim=1).

    The global sequence is split into 2*cp_size equal chunks; rank r holds
    chunk r (front) concatenated with chunk 2*cp_size - r - 1 (mirror back).
    """
    chunks = x.chunk(2 * cp_size, dim=1)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]], dim=1).contiguous()


def record(ctx: DistributedCtx, req: Request) -> tuple[Result, Result]:
    """
    Record the forward output and the input gradients dQ, dK, dV for both the
    baseline and the implementation. The baseline is flash_attn_func run on the
    full sequence with no CP communication; the implementation is the zigzag
    ring_attention_func run on this rank's zigzag-local slice of Q/K/V with K/V
    rotated around the CP ring during the forward and backward passes.
    """
    cp_group = ctx.device_mesh.get_group("cp")
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()
    softmax_scale = req.D**-0.5

    torch.manual_seed(42)
    q_full = torch.randn(req.B, req.S, req.HQ, req.D, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(req.B, req.S, req.HK, req.D, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(req.B, req.S, req.HK, req.D, device=device, dtype=torch.bfloat16)

    q_ref = q_full.clone().requires_grad_(True)
    k_ref = k_full.clone().requires_grad_(True)
    v_ref = v_full.clone().requires_grad_(True)
    out_ref = flash_attn_func(q_ref, k_ref, v_ref, softmax_scale, causal=True)
    out_ref.sum().backward()
    out_ref = extract_zigzag(out_ref, cp_rank, cp_size)
    dq_ref = extract_zigzag(q_ref.grad, cp_rank, cp_size)
    dk_ref = extract_zigzag(k_ref.grad, cp_rank, cp_size)
    dv_ref = extract_zigzag(v_ref.grad, cp_rank, cp_size)
    ref = Result(out_ref, dq_ref, dk_ref, dv_ref)

    q_imp = extract_zigzag(q_full, cp_rank, cp_size).clone().requires_grad_(True)
    k_imp = extract_zigzag(k_full, cp_rank, cp_size).clone().requires_grad_(True)
    v_imp = extract_zigzag(v_full, cp_rank, cp_size).clone().requires_grad_(True)
    out_imp = ring_attention_func(q_imp, k_imp, v_imp, softmax_scale, cp_group)
    out_imp.sum().backward()
    imp = Result(out_imp, q_imp.grad, k_imp.grad, v_imp.grad)

    return ref, imp


def verify(ctx: DistributedCtx, req: Request) -> None:
    ref, imp = record(ctx, req)
    for f in fields(ref):
        error = cosine_error(getattr(ref, f.name), getattr(imp, f.name))
        if error >= req.atol:
            raise AssertionError(f"{f.name} diverged: {error=:.2e} >= {req.atol=}")


REQUESTS = []
REQUESTS.append(pytest.param(2, Request(B=1, S=2048, HQ=4, HK=4, D=64), id="CP2-MHA-S2048"))
REQUESTS.append(pytest.param(2, Request(B=2, S=2048, HQ=8, HK=2, D=64), id="CP2-GQA-S2048"))
REQUESTS.append(pytest.param(4, Request(B=1, S=4096, HQ=12, HK=4, D=128), id="CP4-GQA-S4096"))


@pytest.mark.parametrize("cp_size,req", REQUESTS)
def test_ring_attention_vs_dense(cp_size: int, req: Request) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify, req)
