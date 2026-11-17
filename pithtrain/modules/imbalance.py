"""
Benchmark routing with a realistic expert-load imbalance, used by TrainingCfg.benchmark.
"""

from typing import Callable

import numpy as np
import torch

from pithtrain.contexts import training

SIGMA = 0.2
"""
Spread of the log-normal per-expert token shares; 0 routes force-balanced. Measured as 0.212 on
Qwen3-30B-A3B trained from scratch on DCLM at 4K sequence length, with real routing and a
global-batch load-balancing loss of 1e-3, after routing stabilized (from step 200 on), rounded.
"""


def expert_shares(num_experts: int, sigma: float, layer: int, step: int) -> np.ndarray:
    """
    One layer's per-expert token shares at one step: log-normal with spread sigma, normalized,
    seeded by (layer, step).
    """
    loads = np.random.default_rng((layer, step)).lognormal(0.0, sigma, num_experts)
    return loads / loads.sum()


def replay(num_experts: int, layer: int, sigma: float) -> Callable[[int, int], torch.Tensor]:
    """
    A router replay for one gate: maps (num_tokens, top_k) to expert indices whose counts follow
    this layer's expert_shares at the current step.
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
