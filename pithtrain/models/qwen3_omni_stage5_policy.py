"""Thinker aggregation semantics, independent of expert projection backends."""

from typing import Literal

OmniStage5Aggregation = Literal["grouped_sum", "hf_eager_expert_id_serial"]


def validate_omni_stage5_aggregation(value: OmniStage5Aggregation) -> OmniStage5Aggregation:
    if type(value) is not str or value not in ("grouped_sum", "hf_eager_expert_id_serial"):
        raise ValueError(
            "omni_stage5_aggregation must be 'grouped_sum' or 'hf_eager_expert_id_serial'"
        )
    return value
