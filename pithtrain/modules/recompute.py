"""
Selective recompute of the decoder layer's two hidden-size RMSNorm outputs in compiled stage 1.

The input_layernorm and post_attention_layernorm outputs are saved by their consumers for the
weight gradients, 2 x B*S*hidden bf16 per layer and micro-batch. norm_recompute drops them and
runs the norm again in backward, as Megatron-LM's recompute_modules=["layernorm"] does. Only
aten._fused_rms_norm on hidden-size inputs is recomputed, so the q/k head norms and DeepSeek's
latent norms stay saved; GEMMs, attention and the pithtrain custom ops are always saved. BF16 only.
"""

from functools import partial

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from pithtrain.contexts import training

__all__ = ["norm_recompute"]

GEMM_OPS = frozenset(
    (torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default)
)


def recompute_policy(hidden_size: int, ctx, op, *args, **kwargs) -> CheckpointPolicy:
    x = args[0] if args and isinstance(args[0], torch.Tensor) else None
    if (
        op == torch.ops.aten._fused_rms_norm.default
        and x is not None
        and x.shape[-1] == hidden_size
    ):
        return CheckpointPolicy.PREFER_RECOMPUTE
    name = op.name() if hasattr(op, "name") else str(op)
    if op in GEMM_OPS or getattr(op, "namespace", None) == "pithtrain" or "flash" in name:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_SAVE


def recompute_context(hidden_size: int):
    return create_selective_checkpoint_contexts(partial(recompute_policy, hidden_size))


def norm_recompute(fn, hidden_size: int, *args):
    """
    Return fn(*args), recomputing its hidden-size RMSNorm outputs in backward instead of saving
    them. Call it inside the compiled region.
    """
    if training.fp8:
        return fn(*args)
    return checkpoint(fn, *args, use_reentrant=False, context_fn=partial(recompute_context, hidden_size))  # fmt: skip
