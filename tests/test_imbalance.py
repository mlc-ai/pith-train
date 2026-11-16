"""
Benchmark routing (pithtrain.modules.imbalance) imposes the expert-load imbalance of a real training
run: at SIGMA its log-normal shares give the per-EP-rank max/mean the Qwen3-30B-A3B run measured at
EP 4 to 32, and the indices it replays give every token distinct experts whose counts match the
drawn shares.
"""

import numpy as np
import pytest
import torch

from pithtrain.contexts import training
from pithtrain.modules import imbalance


def rank_imbalance(num_experts, ep_size, sigma, draws=2000):
    shares = np.stack(
        [imbalance.expert_shares(num_experts, sigma, layer, 0) for layer in range(draws)]
    )
    per_rank = shares.reshape(draws, ep_size, -1).sum(axis=2)
    return float((per_rank.max(axis=1) * ep_size).mean())


def test_sigma_reproduces_the_measured_run():
    assert imbalance.SIGMA == 0.2
    # 128 experts: the run's per-rank max/mean, averaged over layers, at EP 4, 8, 16, 32.
    measured = {4: 1.037, 8: 1.073, 16: 1.130, 32: 1.219}
    for ep_size, real in measured.items():
        assert abs(rank_imbalance(128, ep_size, imbalance.SIGMA) - real) < 0.015


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replay builds indices on the GPU")
@pytest.mark.parametrize("num_experts, top_k", [(128, 8), (64, 6), (32, 4)])
def test_replay_matches_the_drawn_shares(num_experts, top_k, monkeypatch):
    num_tokens = 4096
    route = imbalance.replay(num_experts, layer=3, sigma=imbalance.SIGMA)
    monkeypatch.setattr(training, "current_microbatch", 0)
    indices = route(num_tokens, top_k)
    assert indices.shape == (num_tokens, top_k)
    # Every token's experts are distinct.
    assert (indices.sort(dim=1).values.diff(dim=1) > 0).all()
    counts = torch.bincount(indices.flatten(), minlength=num_experts).cpu().numpy()
    shares = imbalance.expert_shares(num_experts, imbalance.SIGMA, 3, 0)
    # Expert counts match the drawn shares up to rounding.
    assert np.abs(counts - shares * num_tokens * top_k).max() <= 1
    # Later micro-batches of the step reuse its routing; micro-batch 0 again starts a new step.
    monkeypatch.setattr(training, "current_microbatch", 1)
    assert route(num_tokens, top_k) is indices
    monkeypatch.setattr(training, "current_microbatch", 0)
    assert not torch.equal(route(num_tokens, top_k), indices)
