"""
The Gated DeltaNet operator under CP at the spec's cases, every rank against the fp64 recurrence
over the whole sequence.
"""

import pytest
import torch
import torch.distributed as dist

from pithtrain.contexts import distributed
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.gated_delta_rule import gated_delta_rule
from tests.kernels import gdn_reference
from tests.kernels.cases import params, seed_of
from tests.utilities import launch


def cp_worker(case: dict) -> None:
    group = distributed.cp_group
    rank, W = group.rank(), group.size()
    axes = case["axes"]
    b, T = axes["mbs"], axes["T_local"]
    H, HV, S = axes["H"] // b, axes["HV"] // b, W * T
    # the whole sequence, identical on every rank
    q, k, v, g, beta, do = gdn_reference.make_inputs(S, H, HV, B=b, seed=seed_of(case))  # fmt: skip
    mine = slice(rank * T, (rank + 1) * T)
    leaves = [x[:, mine].contiguous().requires_grad_() for x in (q, k, v, g, beta)]
    o = gated_delta_rule(*leaves, cp_group=group)
    o.backward(do[:, mine].contiguous())
    got = [o.detach()] + [leaf.grad for leaf in leaves]
    del o, leaves

    def fold(t: torch.Tensor) -> torch.Tensor:  # [b, S, heads, ...] -> [1, S, b * heads, ...]
        return t.transpose(0, 1).reshape(1, S, -1, *t.shape[3:]).contiguous()

    ref = gdn_reference.reference_by_heads(
        *(fold(t) for t in (q, k, v, g, beta, do)), keep=mine, key_grads=True
    )
    # [1, T, b * heads, ...] -> [b, T, heads, ...]
    ref = [x.reshape(T, b, -1, *x.shape[3:]).transpose(0, 1) for x in ref]
    failures = gdn_reference.failures_of(f"{case['id']} rank {rank}", got, ref)
    gathered = [None] * W
    dist.all_gather_object(gathered, failures, group=group)
    failures = [f for rank_failures in gathered for f in rank_failures]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("case", params("gated_delta_rule", keep=lambda c: c["axes"]["cp"] > 1))
def test_context_parallel_at_training_shapes(case: dict) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = case["axes"]["cp"]
    launch(cfg, cp_worker, case)
