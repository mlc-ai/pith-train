"""
Benchmark routing with a realistic expert-load imbalance, used by TrainingCfg.benchmark.
"""

from typing import Callable

import numpy as np
import torch

from pithtrain.contexts import training

SIGMA = 0.2
"""
Spread of the log-normal per-expert token shares, measured as 0.212 on Qwen3-30B-A3B trained from
scratch on DCLM at 4K sequence length, with real routing and a global-batch load-balancing loss of
1e-3, after routing stabilized (from step 200 on), rounded.
"""


def expert_shares(num_experts: int, sigma: float, layer: int, step: int) -> np.ndarray:
    """
    One layer's per-expert token shares at one step: log-normal with spread sigma, normalized,
    seeded by (layer, step).
    """
    loads = np.random.default_rng((layer, step)).lognormal(0.0, sigma, num_experts)
    return loads / loads.sum()


def replay(num_experts: int, layer: int) -> Callable[[int, int], torch.Tensor]:
    """
    A router replay for one gate: maps (num_tokens, top_k) to expert indices whose counts follow
    this layer's expert_shares at the current step. The step advances when the micro-batch index
    wraps around.
    """
    step, last_microbatch, cache = -1, None, {}

    def route(num_tokens: int, top_k: int) -> torch.Tensor:
        nonlocal step, last_microbatch, cache
        microbatch = training.current_microbatch
        if last_microbatch is None or microbatch is None or microbatch <= last_microbatch:
            step, cache = step + 1, {}
        last_microbatch = microbatch
        if (num_tokens, top_k) in cache:
            return cache[(num_tokens, top_k)]

        # Largest-remainder rounding to exactly num_tokens * top_k slots, each expert capped at one
        # slot per token, so that every token's top_k experts are distinct.
        slots = num_tokens * top_k
        exact = np.minimum(expert_shares(num_experts, SIGMA, layer, step) * slots, num_tokens)
        counts = np.floor(exact).astype(np.int64)
        order = np.argsort(-(exact - counts), kind="stable")
        has_room = order[counts[order] < num_tokens]
        counts[has_room[: slots - counts.sum()]] += 1

        # Token t takes slots t, t + N, t + 2N, ...: an expert's slots are contiguous and at most N
        # long, so no token gets an expert twice.
        device = torch.cuda.current_device()
        counts = torch.from_numpy(counts).to(device, non_blocking=True)
        experts = torch.arange(num_experts, device=device)
        expert_of_slot = torch.repeat_interleave(experts, counts, output_size=slots)
        cache[(num_tokens, top_k)] = expert_of_slot.view(top_k, num_tokens).t().contiguous()
        return cache[(num_tokens, top_k)]

    return route
