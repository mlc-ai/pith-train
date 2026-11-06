from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Protocol, Tuple

import torch

__all__ = [
    "AllToAllSplits",
    "LayerProtocol",
    "MLPProtocol",
    "MetadataStage5Protocol",
    "ModelProtocol",
    "RoutingInfo",
    "Stage5RouteMetadata",
    "Stage5RoutingInfo",
]


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


class Stage5RouteMetadata(NamedTuple):
    """Owned global router IDs and per-token expert argsort; never mutate tensors."""

    global_expert_ids: torch.Tensor
    expert_order: torch.Tensor

    @classmethod
    def from_expert_ids(cls, expert_ids: torch.Tensor):
        if expert_ids.ndim != 2 or expert_ids.dtype != torch.int64:
            raise ValueError("Stage5 expert IDs must be int64 router slots [N,K]")
        ids = expert_ids.detach().clone()
        return cls(ids, ids.argsort(dim=-1))


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Stage5RoutingInfo:
    """Optional-only metadata owner; the ordinary RoutingInfo tuple is unchanged.

    Delegate legacy attributes and tuple adapters to the original route. After
    aggregation, gather autograd owns the integer permutation required by VJPs;
    no extra Stage5Args lane or mutable layer route cache is needed.
    """

    routing: RoutingInfo
    stage5_metadata: Stage5RouteMetadata

    def __getattr__(self, name):
        return getattr(self.routing, name)

    def __iter__(self):
        return iter(self.routing)

    def __len__(self):
        return len(self.routing)

    def __getitem__(self, key):
        return self.routing[key]

    def _replace(self, **kwargs):
        return type(self)(self.routing._replace(**kwargs), self.stage5_metadata)


class MetadataStage5Protocol(Protocol):
    """Optional capability selected only for metadata-bearing Thinker routes."""

    def forward_stage5_with_metadata(
        self,
        moe_outs,
        moe_local_idxs,
        topk_weight,
        residual,
        *,
        stage5_metadata: Stage5RouteMetadata,
    ) -> torch.Tensor: ...


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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[RoutingInfo | Stage5RoutingInfo]]:
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
    # Text-only models may omit this. Media models declare their implemented inputs
    # and forward the stage-selected model_context into model_forward(...).
    # Only stage 0 receives encoder payloads; other stages get tokens/masks/layout.
    input_modalities: frozenset[str]

    stage_index: int
    stage_count: int
    hidden_size: int
    layers: Dict[str, LayerProtocol]

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def forward_posemb(
        self, S: int, cu_seqlens: Optional[torch.Tensor] = None, *, model_context=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The (cos, sin) rotary embeddings for this rank's S tokens. When cu_seqlens is set the
        sequence is packed and positions restart at zero at each document boundary; cu_seqlens
        holds the boundaries of the whole packed sample, also under context parallelism.
        """

    def forward_prolog(self, hidden_states: torch.Tensor, *, model_context=None) -> torch.Tensor:
        """
        Prolog compute (first stage only): embed the input token ids.
        """

    def forward_epilog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Epilog compute (last stage only): final norm and lm_head projection to logits.
        """
