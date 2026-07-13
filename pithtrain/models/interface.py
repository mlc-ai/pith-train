from typing import Dict, List, NamedTuple, Optional, Protocol, Tuple

import torch


class AllToAllSplits(NamedTuple):
    input_splits: List[int]
    output_splits: List[int]


class RoutingInfo(NamedTuple):
    topk_weight: torch.Tensor
    expert_idxs: torch.Tensor
    moe_local_idxs: Optional[torch.Tensor] = None
    expand_idx: Optional[torch.Tensor] = None
    dispatch_splits: Optional[AllToAllSplits] = None
    combine_splits: Optional[AllToAllSplits] = None


class MLPProtocol(Protocol):
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
    mlp: MLPProtocol

    def reference_forward(
        self, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Reference forward implementation for correctness validation.
        """

    def forward_stage1(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[RoutingInfo]]:
        """
        Stage 1, the pre-dispatch compute (runs before the stage-2 dispatch).
        Run the attention sublayer and shared experts, then route tokens to experts and prepare the dispatch (MoE layers).
        """

    def forward_stage3(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stage 3, the expert compute (runs after the stage-2 dispatch and before the stage-4 combine).
        Run the experts (or dense MLP) on the dispatched tokens.
        """

    def forward_stage5(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 5, the post-combine compute (runs after the stage-4 combine).
        Aggregate the expert outputs by router weights and add the residual from stage 1.
        """


class ModelProtocol(Protocol):
    """
    Protocol for a DualPipeV-compatible transformer language model.
    """

    stage_index: int
    stage_count: int
    layers: Dict[str, LayerProtocol]

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Reference forward implementation for correctness validation.
        """

    def forward_posemb(
        self, S: int, cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the (cos, sin) rotary embeddings for a sequence of length S. When
        ``cu_seqlens`` is set the sequence is packed, so positions restart at zero
        at each document boundary.
        """

    def forward_prolog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Prolog compute (first stage only): embed the input token ids.
        """

    def forward_epilog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Epilog compute (last stage only): final norm + lm_head projection to logits.
        """
