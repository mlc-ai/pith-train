"""
The index kernels of prepare_dispatch are verified in tests/kernels/test_ep_dispatch.py; this file
runs the operator across EP ranks.
"""

import pytest
import torch


def ragged_dispatch_worker(rank, world_size, tokens_per_rank, out):
    import os

    import torch.distributed as dist

    from pithtrain.operators.all_to_all import direct_all_to_all
    from pithtrain.operators.ep_dispatch import prepare_dispatch
    from pithtrain.operators.token_scatter import padded_index_gather, scatter_for_grouped_gemm

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    ep_size, experts_per_rank, top_k, hidden = world_size, 4, 4, 64
    num_experts = ep_size * experts_per_rank
    device = torch.device("cuda", rank)
    torch.manual_seed(1234 + rank)

    num_tokens = tokens_per_rank[rank]
    hidden_states = torch.randn(1, num_tokens, hidden, device=device, dtype=torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)])  # fmt: skip
    topk_weight = torch.rand(num_tokens, top_k, device=device)

    dispatch_tokens, routing = prepare_dispatch(
        hidden_states,
        topk_ids,
        topk_weight,
        num_experts,
        ep_size,
        experts_per_rank,
        dist.group.WORLD,
    )
    gathered = direct_all_to_all(
        dispatch_tokens.detach(),
        routing.dispatch_splits.output_splits,
        routing.dispatch_splits.input_splits,
        dist.group.WORLD,
    )
    work = getattr(gathered, "comm_work", None)
    if work is not None:
        work.wait()

    # The two lines of forward_stage3 that consume the routing info.
    expanded = padded_index_gather(gathered, routing.expand_idx)
    scatter_for_grouped_gemm(expanded, routing.expert_idxs, experts_per_rank)

    out.append(
        (
            rank,
            num_tokens,
            expanded.shape[0],
            routing.expert_idxs.numel(),
            sum(routing.combine_splits.output_splits),
        )
    )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_prepare_dispatch_ragged_token_counts():
    """
    A rank holding a short micro-batch still receives every slot routed to it. The senders decide
    the received row count, so expert_idxs must have one entry per received row, as many as
    expand_idx, whatever this rank's own token count (8 here against 4096 on the other rank).
    """
    import torch.multiprocessing as mp

    world_size = 2
    manager = mp.Manager()
    out = manager.list()
    mp.spawn(ragged_dispatch_worker, args=(world_size, [8, 4096], out), nprocs=world_size)

    assert len(out) == world_size
    for rank, m, rows, idxs, expected in sorted(out):
        assert rows == expected, f"rank {rank}: gathered {rows} rows, expected {expected}"
        assert idxs == expected, (
            f"rank {rank} (m={m}): expert_idxs has {idxs} entries but {expected} rows were "
            f"received; the old code capped it at m * k * ep_size = {m * 4 * world_size}"
        )
