"""
The context-parallel ring attention operators end to end, at every CP mesh of the spec (family
ring_attention of tests/kernels/cases.py): cp ranks, one per GPU (CP 16 on two nodes), each holding
its zigzag share (blocks r and 2 cp - 1 - r of 2 cp) of B sequences of seq = cp x S_local tokens.

- GQA / MHA models: ring_attention_func(q, k, v), as qwen3 and qwen3.5 call it.
- DeepSeek-V2 (MLA): mla_ring_attention_func(q_nope, q_pe, normed_kv, k_pe, kv_b weight) with the
  bf16 kv_b (training.fp8 off), as deepseek_v2.py calls it.

Reference, independent of FlashAttention and of the ring: every rank draws the whole sequence from
the same seed and computes fp32 causal attention (tests/kernels/attention_reference.py) of its own
query rows over all keys, with a random output gradient; its partial key / value gradients (and,
for MLA, the decompression through kv_b in fp32) are summed over the CP group with all_reduce, so
each rank checks its local output and the gradients of every input it holds, and the kv_b weight
gradient summed over the ranks.

Checked per output on every rank: finite, and relative L2 error within BOUNDS (about 3x the
largest error measured on the H200 over all 24 cases, CP 2 .. 16, every rank, listed next to
each bound: the floor of the bf16 outputs plus, for MLA, the bf16 rounding of the decompressed
k / v and of the combined ring partials;
PITHTRAIN_SPEC_ERROR_LOG=<path> appends each rank's errors as JSON lines). Every case is marked
multigpu: it needs cp GPUs, and too few fails rather than skips (tests/conftest.py).
"""

import json
import os

import pytest
import torch
import torch.distributed as dist

from pithtrain.contexts import distributed
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.ring_attention import mla_ring_attention_func, ring_attention_func
from tests.kernels.attention_reference import attention_reference
from tests.kernels.cases import params, seed_of
from tests.kernels.references import rel_l2
from tests.utilities import launch

BOUNDS = {
    "out": 9e-3,  # measured 2.84e-3 (ring_attention/qwen3.5-27b/cp8-B1-S8k)
    "dq": 9e-3,  # measured 2.97e-3 (ring_attention/qwen3-30b-a3b/cp2-B1-S8k)
    "dk": 9e-3,  # measured 2.97e-3 (ring_attention/qwen3.5-27b/cp2-B1-S8k)
    "dv": 9e-3,  # measured 2.95e-3 (ring_attention/qwen3.5-27b/cp2-B1-S8k)
    "dq_nope": 1e-2,  # measured 3.11e-3 (ring_attention/deepseek-v2-lite/cp4-B1-S4k)
    "dq_pe": 1.15e-2,  # measured 3.69e-3 (ring_attention/deepseek-v2-lite/cp2-B1-S8k)
    "dnormed_kv": 1.05e-2,  # measured 3.40e-3 (ring_attention/deepseek-v2-lite/cp4-B1-S8k)
    "dk_pe": 1.25e-2,  # measured 4.06e-3 (ring_attention/deepseek-v2-lite/cp4-B1-S8k)
    "dkv_b": 8e-3,  # measured 2.56e-3 (ring_attention/deepseek-v2-lite/cp2-B1-S8k)
}
NOPE, ROPE, VDIM, LORA = 128, 64, 128, 512  # DeepSeek-V2-Lite's MLA dims (config.json)


def zigzag_rows(rank: int, cp: int, seq: int, device) -> torch.Tensor:
    """
    The global indices of rank's tokens: block rank, then block 2 cp - 1 - rank, of seq / (2 cp).
    """
    block = seq // (2 * cp)
    front = torch.arange(rank * block, (rank + 1) * block, device=device)
    back = torch.arange((2 * cp - 1 - rank) * block, (2 * cp - rank) * block, device=device)
    return torch.cat([front, back])


def randn(gen: torch.Generator, *shape: int, scale: float = 1.0) -> torch.Tensor:
    x = torch.randn(*shape, generator=gen, device=gen.device, dtype=torch.float32)
    return (x * scale).to(torch.bfloat16)


def check(case: dict, rank: int, got: dict, want: dict) -> None:
    errors = {}
    for name, value in got.items():
        value = value.detach()
        assert torch.isfinite(value).all(), f"rank {rank}: {name} has non-finite values"
        errors[name] = rel_l2(value, want[name])
    log = os.environ.get("PITHTRAIN_SPEC_ERROR_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(dict(id=case["id"], rank=rank, **errors)) + "\n")
    failed = {name: err for name, err in errors.items() if err > BOUNDS[name]}
    assert not failed, f"rank {rank}: errors above BOUNDS: {failed} (all: {errors})"


def gqa_worker(case: dict) -> None:
    a = case["axes"]
    group = distributed.cp_group
    rank, cp = group.rank(), group.size()
    B, seq, Hq, Hkv, D, Dv = a["B"], a["seq"], a["Hq"], a["Hkv"], a["D"], a["Dv"]
    scale = D**-0.5
    gen = torch.Generator(device=f"cuda:{torch.cuda.current_device()}").manual_seed(seed_of(case))
    q, k, v = randn(gen, B, seq, Hq, D), randn(gen, B, seq, Hkv, D), randn(gen, B, seq, Hkv, Dv)
    dout = randn(gen, B, seq, Hq, Dv)
    rows = zigzag_rows(rank, cp, seq, q.device)

    q_loc = q[:, rows].contiguous().requires_grad_(True)
    k_loc = k[:, rows].contiguous().requires_grad_(True)
    v_loc = v[:, rows].contiguous().requires_grad_(True)
    out = ring_attention_func(q_loc, k_loc, v_loc, scale, group)
    out.backward(dout[:, rows])
    got = dict(out=out, dq=q_loc.grad, dk=k_loc.grad, dv=v_loc.grad)

    want = {name: [] for name in got}
    for b in range(B):
        ref = attention_reference(q[b, rows], k[b], v[b], scale, True, dout=dout[b, rows], q_positions=rows)  # fmt: skip
        dk, dv = ref["dk"].contiguous(), ref["dv"].contiguous()
        dist.all_reduce(dk, group=group)
        dist.all_reduce(dv, group=group)
        want["out"].append(ref["out"])
        want["dq"].append(ref["dq"])
        want["dk"].append(dk[rows])
        want["dv"].append(dv[rows])
        del ref, dk, dv
    check(case, rank, got, {name: torch.stack(value) for name, value in want.items()})


def mla_worker(case: dict) -> None:
    a = case["axes"]
    group = distributed.cp_group
    rank, cp = group.rank(), group.size()
    B, seq, H = a["B"], a["seq"], a["Hq"]
    assert (a["D"], a["Dv"]) == (NOPE + ROPE, VDIM)
    scale = a["D"] ** -0.5
    gen = torch.Generator(device=f"cuda:{torch.cuda.current_device()}").manual_seed(seed_of(case))
    q_nope, q_pe = randn(gen, B, seq, H, NOPE), randn(gen, B, seq, H, ROPE)
    normed_kv, k_pe = randn(gen, B, seq, LORA), randn(gen, B, seq, 1, ROPE)
    weight = randn(gen, H * (NOPE + VDIM), LORA, scale=LORA**-0.5)
    dout = randn(gen, B, seq, H, VDIM)
    rows = zigzag_rows(rank, cp, seq, q_nope.device)

    leaves = [x[:, rows].contiguous().requires_grad_(True) for x in (q_nope, q_pe, normed_kv, k_pe)]  # fmt: skip
    w = weight.clone().requires_grad_(True)
    out = mla_ring_attention_func(*leaves, w, sm_scale=scale, qk_nope_head_dim=NOPE, v_head_dim=VDIM, cp_group=group)  # fmt: skip
    out.backward(dout[:, rows])
    dw = w.grad.float()
    dist.all_reduce(dw, group=group)
    got = dict(out=out, dq_nope=leaves[0].grad, dq_pe=leaves[1].grad, dnormed_kv=leaves[2].grad, dk_pe=leaves[3].grad, dkv_b=dw)  # fmt: skip

    w32 = weight.float()
    want = {name: [] for name in got if name != "dkv_b"}
    dw_ref = torch.zeros_like(w32)
    for b in range(B):
        kv = (normed_kv[b].float() @ w32.T).view(seq, H, NOPE + VDIM)
        k_full = torch.cat([kv[..., :NOPE], k_pe[b].float().expand(seq, H, ROPE)], dim=-1)
        q_rows = torch.cat([q_nope[b, rows], q_pe[b, rows]], dim=-1)
        ref = attention_reference(q_rows, k_full, kv[..., NOPE:], scale, True, dout=dout[b, rows], q_positions=rows)  # fmt: skip
        del kv, k_full
        dk, dv = ref["dk"].contiguous(), ref["dv"].contiguous()
        dist.all_reduce(dk, group=group)
        dist.all_reduce(dv, group=group)
        dkv = torch.cat([dk[..., :NOPE], dv], dim=-1).view(seq, H * (NOPE + VDIM))
        want["out"].append(ref["out"])
        want["dq_nope"].append(ref["dq"][..., :NOPE])
        want["dq_pe"].append(ref["dq"][..., NOPE:])
        want["dnormed_kv"].append((dkv[rows] @ w32))
        want["dk_pe"].append(dk[rows, :, NOPE:].sum(1, keepdim=True))
        dw_ref += dkv.T @ normed_kv[b].float()
        del ref, dk, dv, dkv
    want = {name: torch.stack(value) for name, value in want.items()}
    want["dkv_b"] = dw_ref
    check(case, rank, got, want)


def worker(case: dict) -> None:
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        (mla_worker if case["axes"]["mla"] else gqa_worker)(case)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def run(case: dict) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = case["axes"]["cp"]
    launch(cfg, worker, case)


# One function per GPU count, so that -k selects what a run has: CP 2 and 4 fit in 4 GPUs, CP 8 needs
# a node of 8, CP 16 two nodes (see cases.py).
@pytest.mark.parametrize("case", params("ring_attention", keep=lambda c: c["axes"]["cp"] <= 4, multigpu=True))  # fmt: skip
def test_ring_attention(case: dict) -> None:
    run(case)


@pytest.mark.parametrize("case", params("ring_attention", keep=lambda c: c["axes"]["cp"] == 8, multigpu=True))  # fmt: skip
def test_ring_attention_cp8(case: dict) -> None:
    run(case)


@pytest.mark.parametrize("case", params("ring_attention", keep=lambda c: c["axes"]["cp"] == 16, multigpu=True))  # fmt: skip
def test_ring_attention_cp16(case: dict) -> None:
    run(case)
