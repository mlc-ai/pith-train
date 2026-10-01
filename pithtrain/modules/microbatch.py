"""CPU-importable batch contract shared by training data and the GPU pipeline."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


@dataclass(slots=True, kw_only=True)
class Microbatch:
    """
    One micro-batch of work for DualPipeV.step.

    The caller supplies the same sample order and sequence shapes to every pipeline rank
    within each data/context coordinate. The first PP rank consumes model/objective inputs;
    all PP ranks may read model_context. Only the rank hosting stage 0 holds media_inputs.

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
    """Token, mask and position/layout inputs shared by all decoder stages.

    Models opting into media data accept model_context in forward,
    forward_prolog and forward_posemb. The pipeline adds media_inputs only for
    stage 0; later stages receive this shared context without media payloads.
    """
    media_inputs: Optional[dict[str, torch.Tensor]] = None
    """Encoder inputs, such as image pixels or audio features; stage 0 only."""
    sample_ids: tuple[str, ...] = ()

    def context_for_stage(self, stage_index: int) -> Optional[dict[str, torch.Tensor]]:
        if stage_index == 0 and self.media_inputs:
            context = self.model_context or {}
            if context.keys() & self.media_inputs.keys():
                raise ValueError("Media inputs must not overwrite shared model context")
            return dict(context, **self.media_inputs)
        return self.model_context
