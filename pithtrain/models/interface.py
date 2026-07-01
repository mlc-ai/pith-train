from typing import Dict, List, NamedTuple, Optional, Protocol

import torch
import torch.nn as nn


class ForwardAttnOutput(NamedTuple):
    """Output from the forward_attn method of a decoder layer."""

    sorted_tokens: torch.Tensor
    moe_local_idxs: torch.Tensor
    topk_weight: torch.Tensor
    output_splits: List[int]
    input_splits: List[int]
    expert_idxs: torch.Tensor
    residual: torch.Tensor
    expand_idx: Optional[torch.Tensor] = None
    dedup_input_splits: Optional[List[int]] = None
    dedup_output_splits: Optional[List[int]] = None


class DecoderLayerMlpProtocol(Protocol):
    """
    Protocol for the MLP component of a decoder layer in DualPipeV.

    If the experts attribute is present, we treat the MLP as a MoE layer. The EP process
    group is read from pithtrain.contexts.distributed, not carried on the layer.
    """

    ep_size: int


class DecoderLayerProtocol(Protocol):
    """Protocol for a decoder layer in DualPipeV."""

    idx: int
    mlp: DecoderLayerMlpProtocol

    def reference_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Reference forward implementation for correctness validation."""

    def forward_attn(
        self,
        hidden_states: torch.Tensor,
    ) -> ForwardAttnOutput:
        """LN + Attn + LN + Expert selection."""

    def forward_mlp(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MLP forward."""

    def forward_aggregate(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted expert output + residual connection."""


class ModelProtocol(Protocol):
    """Protocol for the DualPipeV model."""

    embed_tokens: Optional[nn.Module]
    norm: Optional[nn.Module]
    lm_head: Optional[nn.Module]
    layers: Dict[str, DecoderLayerProtocol]
