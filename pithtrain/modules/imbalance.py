"""
Benchmark routing with a realistic expert-load imbalance, which TrainingCfg.benchmark turns on.

Per layer and step each expert's token share is a log-normal draw with spread SIGMA, normalized and
seeded by (layer, step) so every rank routes to the same loads; each token gets top_k distinct
experts whose counts match those shares. The per-EP-rank imbalance follows from the contiguous
expert-to-rank placement, so one SIGMA carries over to every EP degree. SIGMA = 0 routes
force-balanced (load_balance.force_balance).
"""

from typing import Callable

import numpy as np
import torch

from pithtrain.contexts import training

SIGMA = 0.2  # measured 0.212 on qwen3-30b-a3b, rounded


def expert_shares(num_experts: int, sigma: float, layer: int, step: int) -> np.ndarray:
    """
    One layer's per-expert load shares at one step: independent log-normal draws with spread sigma,
    normalized to sum 1, seeded by (layer, step) so every rank draws the same shares.
    """
    loads = np.random.default_rng((layer, step)).lognormal(0.0, sigma, num_experts)
    return loads / loads.sum()


def replay(num_experts: int, layer: int, sigma: float) -> Callable[[int, int], torch.Tensor]:
    """
    A router replay for one gate (see load_balance.replay_indices): maps (num_tokens, top_k) to the
    (num_tokens, top_k) expert indices whose counts match this layer's expert_shares at the current
    step. The step advances when the micro-batch index (training.current_microbatch) wraps around;
    every micro-batch of a step routes the same way. Indices are built on the GPU without a host
    sync from counts drawn on the host.
    """
    state = {"step": -1, "last": None, "cache": {}}

    def route(num_tokens: int, top_k: int) -> torch.Tensor:
        mb = training.current_microbatch
        if state["last"] is None or mb is None or mb <= state["last"]:
            state["step"] += 1
            state["cache"] = {}
        state["last"] = mb
        indices = state["cache"].get((num_tokens, top_k))
        if indices is None:
            shares = expert_shares(num_experts, sigma, layer, state["step"])
            slots = num_tokens * top_k
            # Largest-remainder rounding to exactly `slots`, each expert capped at one slot per
            # token, so that every token's top_k experts are distinct.
            want = np.minimum(shares * slots, num_tokens)
            counts = np.floor(want).astype(np.int64)
            short = slots - counts.sum()
            for e in np.argsort(-(want - counts), kind="stable"):
                if short <= 0:
                    break
                if counts[e] < num_tokens:
                    counts[e] += 1
                    short -= 1
            device = torch.cuda.current_device()
            counts_t = torch.from_numpy(counts).to(device, non_blocking=True)
            ids = torch.repeat_interleave(
                torch.arange(num_experts, device=device), counts_t, output_size=slots
            )
            # Token t takes slots t, t + N, t + 2N, ...: an expert's slots are contiguous and at
            # most N long, so no token gets an expert twice.
            indices = ids.view(top_k, num_tokens).t().contiguous()
            state["cache"][(num_tokens, top_k)] = indices
        return indices

    return route
