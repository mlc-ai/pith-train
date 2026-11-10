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
    all PP ranks may read media_context. Only the rank hosting stage 0 holds media_inputs.

    Attributes:
        model_inputs: Inputs to the model, for instance the token ids, handed to its first
            stage positionally. The batch and sequence dimensions of the first one size the
            activation buffers every pipeline stage receives, so it must lead with those two.
        cu_seqlens: Document boundaries when this micro-batch packs several sequences, or None
            when each row holds a single sequence. Under context parallelism they stay those of
            the whole packed sample, while the tensors hold this rank's share of every document
            (cp_sequence.zigzag_varlen_index).
        objective_inputs: Whatever the objective needs for this micro-batch, passed through
            untouched. The engine never inspects it.
    """

    model_inputs: Tuple[torch.Tensor, ...]
    cu_seqlens: Optional[torch.Tensor]
    objective_inputs: Any

    media_context: Optional[dict[str, torch.Tensor]] = None
    """Inputs needed to reconstruct multimodal rotary positions on every decoder stage.

    With the Qwen3-Omni processor these are input_ids, attention_mask,
    image_grid_thw / video_grid_thw, video_second_per_grid and audio
    feature_attention_mask when present. They describe where media tokens lie
    and their temporal/spatial extent; hidden activations alone cannot recover it.
    Text-only batches leave this None and continue to use S and cu_seqlens.

    Media models accept this dictionary in forward, forward_prolog and
    forward_posemb. context_for_stage adds encoder tensors only at stage 0.
    """
    media_inputs: Optional[dict[str, torch.Tensor]] = None
    """Encoder inputs, such as image pixels or audio features; stage 0 only."""
    sample_ids: tuple[str, ...] = ()

    def context_for_stage(self, stage_index: int) -> Optional[dict[str, torch.Tensor]]:
        if stage_index == 0 and self.media_inputs:
            context = self.media_context or {}
            if context.keys() & self.media_inputs.keys():
                raise ValueError("Media inputs must not overwrite shared media context")
            return dict(context, **self.media_inputs)
        return self.media_context
