"""
Correctness test for ring attention (context parallelism).

Compares the output and gradients of ring_attention_func (split across CP ranks)
against a single flash_attn_func call on the full un-split sequence.

Launch with:
    torchrun --nproc-per-node=2 tests/test_ring_attention.py
"""

import os
import sys

import torch
import torch.distributed as dist

from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.ring_attention import ring_attention_func


def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def _cosine_error(a, b):
    """1 - cosine similarity: 0 = identical, 2 = opposite."""
    a, b = a.double().flatten(), b.double().flatten()
    return 1.0 - torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)


def _rel_error(a, b):
    """Max element-wise relative error."""
    a, b = a.float(), b.float()
    return ((a - b).abs() / (b.abs().clamp(min=1e-6))).max().item()


def test_forward_backward(B, S, H, D, num_kv_heads, cp_group, dtype=torch.bfloat16):
    """Compare ring attention on split sequence vs standard attention on full sequence."""
    rank = cp_group.rank()
    cp_size = cp_group.size()
    device = torch.cuda.current_device()
    scale = D**-0.5
    S_local = S // cp_size

    torch.manual_seed(42)
    q_full = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k_full = torch.randn(B, S, num_kv_heads, D, device=device, dtype=dtype)
    v_full = torch.randn(B, S, num_kv_heads, D, device=device, dtype=dtype)

    # --- Reference: standard causal flash attention on full sequence ---
    q_ref = q_full.clone().requires_grad_(True)
    k_ref = k_full.clone().requires_grad_(True)
    v_ref = v_full.clone().requires_grad_(True)
    out_ref = flash_attn_func(q_ref, k_ref, v_ref, softmax_scale=scale, causal=True)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # --- Ring attention: each rank gets its chunk ---
    q_chunk = q_full[:, rank * S_local : (rank + 1) * S_local].clone().requires_grad_(True)
    k_chunk = k_full[:, rank * S_local : (rank + 1) * S_local].clone().requires_grad_(True)
    v_chunk = v_full[:, rank * S_local : (rank + 1) * S_local].clone().requires_grad_(True)
    out_ring = ring_attention_func(
        q_chunk, k_chunk, v_chunk, softmax_scale=scale, cp_group=cp_group
    )
    loss_ring = out_ring.sum()
    loss_ring.backward()

    # --- Compare forward output ---
    out_ref_chunk = out_ref[:, rank * S_local : (rank + 1) * S_local]
    fwd_cos = _cosine_error(out_ring, out_ref_chunk).item()

    # --- Compare gradients (cosine similarity) ---
    dq_ref_chunk = q_ref.grad[:, rank * S_local : (rank + 1) * S_local]
    dk_ref_chunk = k_ref.grad[:, rank * S_local : (rank + 1) * S_local]
    dv_ref_chunk = v_ref.grad[:, rank * S_local : (rank + 1) * S_local]

    dq_cos = _cosine_error(q_chunk.grad, dq_ref_chunk).item()
    dk_cos = _cosine_error(k_chunk.grad, dk_ref_chunk).item()
    dv_cos = _cosine_error(v_chunk.grad, dv_ref_chunk).item()

    return fwd_cos, dq_cos, dk_cos, dv_cos


def main():
    rank, world_size = setup()
    assert world_size >= 2, "Need at least 2 GPUs"

    cp_group = dist.new_group(list(range(world_size)))
    atol = 1e-5

    configs = [
        {"B": 1, "S": 128, "H": 4, "D": 64, "num_kv_heads": 4, "label": "MHA S=128"},
        {"B": 2, "S": 256, "H": 8, "D": 64, "num_kv_heads": 2, "label": "GQA S=256"},
        {"B": 1, "S": 512, "H": 12, "D": 128, "num_kv_heads": 4, "label": "GQA S=512 D=128"},
    ]

    all_passed = True
    for cfg in configs:
        label = cfg.pop("label")
        fwd, dq, dk, dv = test_forward_backward(**cfg, cp_group=cp_group)
        worst = max(fwd, dq, dk, dv)
        passed = worst < atol
        all_passed &= passed
        if rank == 0:
            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] {label}: fwd_cos={fwd:.2e}  dQ_cos={dq:.2e}  dK_cos={dk:.2e}  dV_cos={dv:.2e}"
            )

    dist.destroy_process_group()
    if rank == 0:
        print("\nAll tests passed." if all_passed else "\nSome tests FAILED.")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
