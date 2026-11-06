"""
prepare_dispatch with deduplicate=False, the Qwen3-Omni Talker's expert-parallel dispatch, on one
process: each EP rank is one call, the count all-to-all inside it is answered from every rank's
routing, and the token all-to-alls are replayed from the returned splits as pipeline/execution.py
runs them. Every already-expanded (tokens, K, hidden) slot must be sent once, in expert order with
ties in slot order (stable_dedup_prepare_dispatch's order), reach the expert it was routed to, come
back to its own slot through moe_local_idxs, and receive exactly its own gradient row.
"""

import pytest
import torch

from pithtrain.operators import ep_dispatch
from pithtrain.operators.stable_scatter import stable_dedup_prepare_dispatch

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    ),
]
EP_GROUP = object()


def ascending_routes(tokens, topk, experts, generator, device):
    """Distinct experts per token in ascending order, as the Talker sorts them before dispatch."""
    scores = torch.rand(tokens, experts, generator=generator, device=device)
    return scores.topk(topk, dim=-1).indices.sort(dim=-1).values


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "tokens,topk,experts,ep", [(1, 2, 4, 2), (257, 4, 8, 2), (1025, 2, 8, 4), (64, 8, 16, 4)]
)
def test_independent_dispatch_preserves_routes_and_original_slot_order(
    device, deterministic, tokens, topk, experts, ep, monkeypatch
):
    epr, hidden = experts // ep, 16
    generator = torch.Generator(device=device).manual_seed(1731)
    ranks = []
    for rank in range(ep):
        rows = 0 if rank == 1 else tokens  # rank 1 sends nothing
        ids = ascending_routes(rows, topk, experts, generator, device)
        slots = torch.randn(rows, topk, hidden, generator=generator, device=device)
        weights = torch.rand(rows, topk, generator=generator, device=device)
        ranks.append((ids, slots.to(torch.bfloat16).requires_grad_(), weights))
    counts = [torch.bincount(ids.view(-1), minlength=experts) for ids, _, _ in ranks]
    caller = []

    def count_all_to_all(output, x, output_split_sizes=None, input_split_sizes=None, group=None):
        # The only collective inside: every rank sends rank r its counts for r's experts.
        (rank,) = caller
        assert group is EP_GROUP and output_split_sizes is None and input_split_sizes is None
        assert torch.equal(x, counts[rank])
        output.copy_(torch.cat([sent.view(ep, epr)[rank] for sent in counts]))

    monkeypatch.setattr(ep_dispatch, "all_to_all_sync", count_all_to_all)
    dispatched = []
    for rank, (ids, slots, weights) in enumerate(ranks):
        caller[:] = [rank]
        order = torch.argsort(ids.view(-1), stable=True)
        reference = stable_dedup_prepare_dispatch(ids, experts, ep, epr)
        assert torch.equal(order, reference[3])
        for _ in range(3):
            tokens_out, routing = ep_dispatch.prepare_dispatch(
                slots, ids, weights, experts, ep, epr, EP_GROUP,
                deterministic=deterministic, deduplicate=False,
            )  # fmt: skip
            assert torch.equal(routing.moe_local_idxs, order)
            assert torch.equal(tokens_out, slots.detach().view(-1, hidden)[order])
        assert routing.dispatch_splits.input_splits == reference[0].tolist()
        assert torch.equal(routing.topk_weight, weights)
        dispatched.append((tokens_out, routing))

    # Dispatch all-to-all (stage 2): rank r receives every sender's r-th chunk, in sender order.
    returned = []
    for rank, (_, routing) in enumerate(dispatched):
        chunks, routed = [], []
        for (ids, _, _), (tokens_out, sender) in zip(ranks, dispatched):
            splits = sender.dispatch_splits.input_splits
            chunks.append(tokens_out.split(splits)[rank])
            routed.append(ids.view(-1)[sender.moe_local_idxs].split(splits)[rank])
        received, routed = torch.cat(chunks), torch.cat(routed)
        assert routing.dispatch_splits.output_splits == [chunk.shape[0] for chunk in chunks]
        assert torch.equal(routing.expert_idxs, routed - rank * epr)
        assert routing.expert_counts == torch.bincount(routed - rank * epr, minlength=epr).tolist()
        assert torch.equal(routing.expand_idx, torch.arange(received.shape[0], device=device))
        # Emulated experts scale each row by its global expert id + 1; combine (stage 4) sends
        # combine_splits.output_splits rows back to each sender.
        scale = (routing.expert_idxs + rank * epr + 1).to(received.dtype)
        returned.append(
            (received.detach() * scale[:, None]).split(routing.combine_splits.output_splits)
        )

    for rank, ((ids, slots, _), (tokens_out, routing)) in enumerate(zip(ranks, dispatched)):
        back = [returned[receiver][rank] for receiver in range(ep)]
        assert routing.combine_splits.input_splits == [chunk.shape[0] for chunk in back]
        back = torch.cat(back)
        restored = back.new_zeros(ids.numel(), hidden).index_copy(0, routing.moe_local_idxs, back)
        scale = (ids.view(-1) + 1).to(slots.dtype)
        assert torch.equal(restored, slots.detach().view(-1, hidden) * scale[:, None])
        # Each slot is sent once, so its backward lane receives exactly its own gradient row.
        grad = torch.randn(tokens_out.shape, generator=generator, device=device)
        tokens_out.backward(grad.to(tokens_out.dtype))
        assert torch.equal(
            slots.grad.view(-1, hidden)[routing.moe_local_idxs], grad.to(slots.dtype)
        )
