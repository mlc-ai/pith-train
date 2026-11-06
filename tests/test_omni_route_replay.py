"""CPU checks for route-conditioned parity and the analytic-zero audio key-bias criterion.

The MoE arm uses the real native router/reference method with the CPU kernel
substitutes below and the production replay_indices hook, so router_replay is
exercised exactly as the full-size gates use it. A benign attention change (SDPA
instead of eager) stands in for FA4.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel as HFModel,
)

from pithtrain.contexts import training as training_context
from pithtrain.models.qwen3_omni_stage5_policy import validate_omni_stage5_aggregation
from pithtrain.modules.load_balance import replay_indices
from tests.acceptance import omni_route_replay as routing
from tests.acceptance.omni_full_size import gradient_diagnostics, metric
from tests.cpu_helpers import load_definitions
from tests.test_qwen3_omni_audio import audio_config, paired


class CPUGroupedLinear:
    @staticmethod
    def apply(x, weight, offsets):
        return F.grouped_mm(x, weight.transpose(-2, -1), offs=offsets)


def cpu_scatter(tokens, expert_ids, num_groups, *, deterministic):
    """Unpadded substitute for the stable grouped-GEMM scatter, same row permutation."""
    assert deterministic
    order = expert_ids.argsort(stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    offsets = torch.bincount(expert_ids, minlength=num_groups).cumsum(0).to(torch.int32)
    return tokens[order], inverse, offsets, None, None


NS = load_definitions(
    "pithtrain/models/qwen3_omni_moe.py",
    (
        "Qwen3OmniMoeThinkerTextExperts",
        "Qwen3OmniMoeThinkerTextTopKRouter",
        "_combine_expert_sum",
        "Qwen3OmniMoeThinkerTextSparseMoeBlock",
    ),
    dict(
        torch=torch,
        nn=nn,
        F=F,
        validate_omni_stage5_aggregation=validate_omni_stage5_aggregation,
        training=SimpleNamespace(
            deterministic_token_layout=True,
            omni_stage5_aggregation=training_context.omni_stage5_aggregation,
        ),
        distributed=SimpleNamespace(ep_size=1),
        GroupedLinearFunc=CPUGroupedLinear,
        scatter_for_grouped_gemm=cpu_scatter,
        padded_index_gather=lambda x, index: x[index],
        replay_indices=replay_indices,
    ),
)


class NativeMoE(NS["Qwen3OmniMoeThinkerTextSparseMoeBlock"]):
    forward = NS["Qwen3OmniMoeThinkerTextSparseMoeBlock"].reference_forward


class NativeText(nn.Module):
    """HF attention/norm stack with native MoE blocks, called like the native model."""

    def __init__(self, reference, config):
        super().__init__()
        self.inner = copy.deepcopy(reference)
        for layer in self.inner["model"].layers:
            moe = NativeMoE(config).to(layer.mlp.gate.weight.dtype)
            moe.load_state_dict(layer.mlp.state_dict())
            layer.mlp = moe
        self.inner["model"].set_attn_implementation("sdpa")

    def forward(self, values):
        return self.inner["lm_head"](
            self.inner["model"](inputs_embeds=values, use_cache=False).last_hidden_state
        )


def official_like(layers=24):
    # Official top-8 of 128 experts; router std scaled so router-logit std matches the
    # official width (0.02 * sqrt(2048)), i.e. the full-size near-tie regime.
    config = Qwen3OmniMoeTextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=16,
        max_position_embeddings=512,
        attention_dropout=0.0,
        norm_topk_prob=True,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=10000.0,
            mrope_section=[2, 3, 3],
            interleaved=True,
            mrope_interleaved=True,
        ),
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "grouped_mm"
    return config


def build(config, seed):
    torch.manual_seed(seed)
    reference = nn.ModuleDict({"model": HFModel(config), "lm_head": nn.Linear(64, 64, bias=False)})
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if parameter.ndim == 1:
                parameter.fill_(1.0)
            else:
                std = 0.02 * (2048 / 64) ** 0.5 if name.endswith("mlp.gate.weight") else 0.02
                parameter.normal_(std=std)
    return reference.to(torch.bfloat16)


def run(model, native, values, labels, replay=None):
    model.zero_grad(set_to_none=True)
    with routing.capture_routes(model, native) as routes:
        if replay is None:
            logits = model(values) if native else hf_logits(model, values)
        else:
            with routing.replay_routes(model, replay):
                logits = model(values)
    F.cross_entropy(logits.float().flatten(0, 1), labels.flatten()).backward()
    gradients = {
        name.removeprefix("inner."): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None  # inputs_embeds bypasses the embedding table
    }
    return logits.detach(), gradients, routes


def hf_logits(model, values):
    return model["lm_head"](model["model"](inputs_embeds=values, use_cache=False).last_hidden_state)


@pytest.fixture(scope="module")
def flipped_case():
    config = official_like()
    reference = build(config, 739)
    generator = torch.Generator().manual_seed(753)
    values = (torch.randn(1, 14, 64, generator=generator) * 0.02).to(torch.bfloat16)
    labels = torch.randint(0, 64, (1, 14), generator=generator)
    labels[:, :4] = -100
    native = NativeText(reference, config)
    return reference, native, values, labels


def errors(actual, expected):
    assert actual.keys() == expected.keys()
    return {name: metric(actual[name], expected[name]) for name in expected}


def test_free_running_flips_are_near_ties_and_replay_removes_every_gradient_failure(
    flipped_case,
):
    reference, native, values, labels = flipped_case
    hf_out, hf_grads, hf_routes = run(reference, False, values, labels)
    free_out, free_grads, free_routes = run(native, True, values, labels)
    free = routing.route_statistics(hf_routes, free_routes)
    free_errors = errors(free_grads, hf_grads)
    # A benign attention change alone flips near-tie routes and fails the original gate.
    assert free["flipped_token_layers"] > 0
    assert free["flipped_reference_margin"]["median"] < free["unflipped_reference_margin"]["median"]
    failing = {name for name, value in free_errors.items() if value >= 1e-2}
    assert failing
    assert {routing.parameter_group(name) for name in failing} <= {
        "router",
        "experts",
        "post_attention_norm",
    }
    replay_out, replay_grads, replay_routes = run(native, True, values, labels, hf_routes)
    replayed = routing.route_statistics(hf_routes, replay_routes)
    assert replayed["flipped_token_layers"] == 0
    replay_errors = errors(replay_grads, hf_grads)
    assert max(replay_errors.values()) < 1e-2
    assert metric(replay_out, hf_out) < 1e-3
    # Replay leaves the native routers in their production state.
    assert all(router.router_replay is None for router in routing.moe_routers(native).values())


def test_replay_rejects_routes_for_a_different_token_count(flipped_case):
    reference, native, values, labels = flipped_case
    _, _, hf_routes = run(reference, False, values, labels)
    with pytest.raises(AssertionError, match="do not match the router input"):
        run(native, True, values[:, :-1], labels[:, :-1], hf_routes)
    assert all(router.router_replay is None for router in routing.moe_routers(native).values())


def test_route_statistics_count_set_changes_and_use_reference_margins():
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.4, 0.31, 0.29, 0.0]])
    reference = {0: {"indices": torch.tensor([[0, 1], [0, 1]]), "probabilities": probabilities}}
    # Token 0 keeps its set in another order; token 1 swaps expert 1 for expert 2.
    actual = {0: {"indices": torch.tensor([[1, 0], [0, 2]]), "probabilities": probabilities}}
    stats = routing.route_statistics(reference, actual)
    assert stats["flipped_token_layers"] == 1 and stats["flipped_tokens_per_layer"] == [1]
    assert stats["flipped_reference_margin"]["max"] == pytest.approx(0.02)
    assert stats["unflipped_reference_margin"]["max"] == pytest.approx(0.15)


def key_bias_rows(k_actual, k_reference, q=1.0, status="FAIL"):
    rows = []
    for name, actual, reference in (
        ("audio_tower.layers.0.self_attn.k_proj.bias", k_actual, k_reference),
        ("audio_tower.layers.0.self_attn.q_proj.bias", q, q),
    ):
        rows.append(
            {
                "name": name,
                "status": status,
                "finite": True,
                "actual_l2": actual,
                "reference_l2": reference,
                "difference": 1.0,
            }
        )
    return rows


@pytest.mark.parametrize("status", ["PASS", "FAIL"])
@pytest.mark.parametrize(
    "k_actual,k_reference,q,ratios,states",
    [
        (4e-7, 5.6e-6, 1.0, (4e-7, 5.6e-6), ("DEFINED", "DEFINED")),
        (0.05, 0.05, 1.0, (0.05, 0.05), ("DEFINED", "DEFINED")),
        (0.5, 1e-6, 1.0, (0.5, 1e-6), ("DEFINED", "DEFINED")),
        (None, 1e-6, 1.0, (None, 1e-6), ("MISSING", "DEFINED")),
        (0.0, 0.0, 0.0, (None, None), ("UNDEFINED_ZERO_QUERY", "UNDEFINED_ZERO_QUERY")),
        (float("nan"), 1e-6, 1.0, (None, 1e-6), ("NONFINITE", "DEFINED")),
    ],
)
def test_key_bias_ratios_are_reported_without_changing_the_nse_verdict(
    k_actual, k_reference, q, ratios, states, status
):
    rows = routing.annotate_key_bias_ratios(key_bias_rows(k_actual, k_reference, q, status))
    key, query = rows
    assert key["status"] == query["status"] == status and "reason" not in key
    assert key["difference"] == 1.0
    got = key["key_to_query_gradient_ratio"]
    for side, expected in zip(("actual", "reference"), ratios, strict=True):
        assert got[side] is None if expected is None else got[side] == pytest.approx(expected)
    assert key["key_to_query_ratio_state"] == dict(actual=states[0], reference=states[1])
    assert "key_to_query_gradient_ratio" not in query


def test_key_bias_ratio_without_its_query_row_is_missing():
    (row,) = routing.annotate_key_bias_ratios(key_bias_rows(1e-6, 1e-6, status="FAIL")[:1])
    assert row["status"] == "FAIL"
    assert row["key_to_query_ratio_state"] == dict(actual="MISSING", reference="MISSING")


def test_audio_key_bias_gradient_is_analytically_zero_but_its_nse_is_ill_posed():
    """Guards the analytic-zero rationale: RoPE or a key norm would break invariance."""
    config = audio_config(encoder_layers=2)
    native, reference = paired(config, dtype=torch.float64)
    lengths = torch.tensor([201, 17])
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(16, int(lengths.sum()), generator=generator, dtype=torch.float64)
    with torch.no_grad():
        base = native(features, lengths).last_hidden_state
        changes = {}
        for target in ("k", "q"):
            shifted = copy.deepcopy(native)
            for index in range(config.encoder_layers):
                bias = shifted.get_submodule(f"layers.{index}.self_attn.{target}_proj").bias
                delta = torch.randn(bias.shape, generator=generator, dtype=torch.float64)
                bias.add_(delta / delta.norm())
            output = shifted(features, lengths).last_hidden_state
            changes[target] = ((output - base).norm() / base.norm()).item()
    assert changes["k"] < 1e-9 < 1e-5 < changes["q"]
    # Both implementations' computed key-bias gradients are rounding residue, and
    # switching only the HF attention backend makes their NSE meaningless (~1).
    reference.config._attn_implementation = "sdpa"
    gradients = {}
    for name, model, call in (
        ("native", native, lambda m: m(features, lengths)),
        ("hf", reference, lambda m: m(features, feature_lens=lengths)),
    ):
        model.zero_grad(set_to_none=True)
        out = call(model).last_hidden_state
        probe = torch.randn(out.shape, generator=torch.Generator().manual_seed(5), dtype=out.dtype)
        (out * probe).sum().backward()
        gradients[name] = {
            target: model.get_submodule(f"layers.0.self_attn.{target}_proj").bias.grad
            for target in ("k", "q")
        }
    for name in gradients:
        assert gradients[name]["k"].norm() < 1e-4 * gradients[name]["q"].norm()
    rows = [
        {
            "name": f"audio_tower.layers.0.self_attn.{target}_proj.bias",
            "status": "PASS",
            "finite": True,
            **gradient_diagnostics(gradients["native"][target], gradients["hf"][target]),
        }
        for target in ("k", "q")
    ]
    assert rows[0]["nse"] > 0.1 and rows[1]["nse"] < 1e-9
    routing.annotate_key_bias_ratios(rows)
    # The ratio is reported, not judged: the rows keep the status they came with.
    assert rows[0]["status"] == rows[1]["status"] == "PASS"
    assert rows[0]["key_to_query_gradient_ratio"]["actual"] < 1e-4
