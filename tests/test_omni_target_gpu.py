"""Actual fused CE + FSDP sum reduction with unequal/empty rank-local labels.

torchrun --standalone --nproc-per-node=4 tests/test_omni_target_gpu.py
Two independent PP-stage groups each contain two DP/CP ranks. Every gradient
and SGD update is checked against a single-device global-token reference.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def main():
    import os
    from datetime import timedelta

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=2))
    from pithtrain.modules.microbatch import Microbatch
    from pithtrain.modules.training_data import global_loss_mean, global_target_count
    from pithtrain.tasks.pretrain_lm import objective

    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    mesh = DeviceMesh("cuda", torch.arange(4).reshape(2, 2), mesh_dim_names=("pp", "dp_cp"))
    group = mesh["dp_cp"].get_group()
    local = rank % 2
    device = torch.device("cuda", torch.cuda.current_device())
    for empty_rank, dtype in (
        (False, torch.float32),
        (True, torch.float32),
        (False, torch.bfloat16),
        (True, torch.bfloat16),
    ):
        model = torch.nn.Linear(8, 32, bias=False, device=device)
        with torch.no_grad():
            model.weight.copy_(torch.linspace(-0.2, 0.2, 256, device=device).reshape(32, 8))
        initial = model.weight.detach().clone()
        reference = torch.nn.Linear(8, 32, bias=False, device=device)
        reference.load_state_dict(model.state_dict())
        reference.to(dtype=dtype)
        fully_shard(
            model,
            mesh=mesh["dp_cp"],
            mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
        )
        model.set_gradient_divide_factor(1.0)
        inputs = torch.arange(64, device=device).reshape(8, 8).float() / 64
        labels = torch.tensor([0, -100, -100, -100, 4, 5, 6, -100], device=device)
        if empty_rank:
            labels[:4] = -100
        x, y = inputs[local * 4 : (local + 1) * 4], labels[local * 4 : (local + 1) * 4]
        mb = Microbatch(model_inputs=(x,), objective_inputs=(y,), cu_seqlens=None)
        count = global_target_count([mb], group)
        assert count.item() == (3 if empty_rank else 4), count
        loss, _ = objective((model(x),), (y,))
        if empty_rank and local == 0:
            assert loss.item() == 0, loss
        loss.backward()
        model.weight.grad.div_(count)
        expected = F.cross_entropy(reference(inputs.to(dtype)).float(), labels)
        expected.backward()
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-4)
        torch.testing.assert_close(
            global_loss_mean(loss, count, group), expected.detach(), rtol=rtol, atol=atol
        )
        torch.testing.assert_close(
            model.weight.grad.full_tensor(), reference.weight.grad.float(), rtol=rtol, atol=atol
        )
        reference.float()
        with torch.no_grad():
            reference.weight.copy_(initial)
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        torch.optim.SGD(reference.parameters(), lr=0.1).step()
        torch.testing.assert_close(
            model.weight.full_tensor(), reference.weight, rtol=rtol, atol=atol
        )
        # A globally empty batch must fail consistently on every PP-stage rank.
        mb.objective_inputs = (torch.full_like(y, -100),)
        try:
            global_target_count([mb], group)
        except ValueError as error:
            assert "no valid" in str(error)
        else:
            raise AssertionError("Globally empty targets were accepted")
    if rank == 0:
        print("PASSED: fused CE/FSDP loss, gradients, updates and empty-target guards", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
