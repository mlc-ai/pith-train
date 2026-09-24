"""CPU-importable batch contract shared by training data and the GPU pipeline."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


@dataclass(slots=True, kw_only=True)
class Microbatch:
    """
    One micro-batch of work for DualPipeV.step.

    The caller partitions the global batch into these and hands the same list to every pipeline
    rank within each data/context coordinate, so each derives receive-buffer shapes locally.
    The first PP rank consumes model/objective inputs; all PP ranks may read model_context.

    Attributes:
        model_inputs: Inputs to the model, for instance the token ids, handed to its first
            stage positionally. The batch and sequence dimensions of the first one size the
            activation buffers every pipeline stage receives, so it must lead with those two.
        cu_seqlens: Document boundaries when this micro-batch packs several sequences, or None
            when each row holds a single sequence.
        objective_inputs: Whatever the objective needs for this micro-batch, passed through
            untouched. The engine never inspects it.
    """

    model_inputs: Tuple[torch.Tensor, ...]
    cu_seqlens: Optional[torch.Tensor]
    objective_inputs: Any

    model_context: Optional[dict[str, torch.Tensor]] = None
    """Per-sample media/position inputs, present on every PP rank.

    Models opting into media data must accept model_context in forward,
    forward_prolog and forward_posemb. Decoder-stage context is not a P2P
    activation: each PP rank receives it from the deterministic data stream.
    """
    sample_ids: tuple[str, ...] = ()
