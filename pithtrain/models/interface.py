from typing import Dict, List, NamedTuple, Optional, Protocol, Tuple

import torch
import torch.nn as nn


class AllToAllSplits(NamedTuple):
    input_splits: List[int]
    output_splits: List[int]


class MoERouting(NamedTuple):
    topk_weight: torch.Tensor
    expert_idxs: torch.Tensor
    moe_local_idxs: Optional[torch.Tensor] = None
    expand_idx: Optional[torch.Tensor] = None
    dispatch_splits: Optional[AllToAllSplits] = None
    combine_splits: Optional[AllToAllSplits] = None


class MlpProtocol(Protocol):
    """
    Protocol for the MLP component of a DualPipeV-compatible decoder layer.
    """


class LayerProtocol(Protocol):
    """
    Protocol for a DualPipeV-compatible decoder layer.

    Each layer is split into five stages so the pipeline scheduler can interleave different
    micro-batches and overlap the compute of one with the communication of another.

    - Stage 1: pre-dispatch compute.
    - Stage 2: dispatch all-to-all.
    - Stage 3: expert compute.
    - Stage 4: combine all-to-all.
    - Stage 5: post-combine compute.
    """

    idx: int
    mlp: MlpProtocol

    def reference_forward(self, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Reference forward implementation for correctness validation.
        """

    def forward_stage1(self, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[MoERouting]]:
        """
        Stage 1, the pre-dispatch compute (runs before the stage-2 dispatch).
        Run the attention sublayer and shared experts, then route tokens to experts and prepare the dispatch (MoE layers).
        """

    def forward_stage3(self, gathered_tokens: torch.Tensor, expert_idxs: Optional[torch.Tensor] = None, expand_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Stage 3, the expert compute (runs after the stage-2 dispatch and before the stage-4 combine).
        Run the experts (or dense MLP) on the dispatched tokens.
        """

    def forward_stage5(self, moe_outs: torch.Tensor, moe_local_idxs: Optional[torch.Tensor], topk_weight: Optional[torch.Tensor], residual: torch.Tensor) -> torch.Tensor:
        """
        Stage 5, the post-combine compute (runs after the stage-4 combine).
        Aggregate the expert outputs by router weights and add the residual from stage 1.
        """


class ModelProtocol(Protocol):
    """
    Protocol for a DualPipeV-compatible transformer language model.
    """

    embed_tokens: Optional[nn.Module]
    norm: Optional[nn.Module]
    lm_head: Optional[nn.Module]
    layers: Dict[str, LayerProtocol]

    def rotary_posemb(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the (cos, sin) rotary embeddings for the tokens in this micro-batch.
        """
