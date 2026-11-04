from typing import Dict, List, NamedTuple, Optional, Protocol, Tuple

import torch

__all__ = ["AllToAllSplits", "LayerProtocol", "MLPProtocol", "ModelProtocol", "RoutingInfo"]


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
    expert_counts: Optional[list[int]] = None
    """
    Rows routed to each local expert, on the host, when dispatch already knows them; lets stage 3
    size its grouped GEMMs without another device-to-host sync.
    """


class MLPProtocol(Protocol): ...


class LayerProtocol(Protocol):
    """
    A DualPipeV decoder layer, split into five stages so the pipeline scheduler can overlap the
    compute of one micro-batch with the communication of another.

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
    ) -> torch.Tensor: ...

    def forward_stage1(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[RoutingInfo]]:
        """
        Stage 1, before the dispatch: run the attention sublayer and shared experts, then route the
        tokens to experts and prepare the dispatch (MoE layers).
        """

    def forward_stage3(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
        expert_counts: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Stage 3, between the dispatch and the combine: run the experts (or the dense MLP) on the
        tokens this rank received.
        """

    def forward_stage5(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 5, after the combine: weight the expert outputs by the router weights, sum them per
        token and add the residual from stage 1.
        """


class ModelProtocol(Protocol):
    stage_index: int
    stage_count: int
    hidden_size: int
    layers: Dict[str, LayerProtocol]

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def forward_posemb(
        self, S: int, cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The (cos, sin) rotary embeddings for this rank's S tokens. When cu_seqlens is set the
        sequence is packed and positions restart at zero at each document boundary; cu_seqlens
        holds the boundaries of the whole packed sample, also under context parallelism.
        """

    def forward_prolog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Prolog compute (first stage only): embed the input token ids.
        """

    def forward_epilog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Epilog compute (last stage only): final norm and lm_head projection to logits.
        """
