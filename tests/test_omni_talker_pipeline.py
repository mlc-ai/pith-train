"""CPU execution of production Talker stage bodies and state-layout contracts.

CUDA imports are excluded by extracting pure class methods, following the
existing stage5 tests. The real fullgraph decorators and arithmetic are retained.
These tests do not establish FA4, collectives or FSDP GPU acceptance.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.cpu_helpers import load_definitions

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "pithtrain/models/qwen3_omni_talker_pipeline.py"


def method_namespace(ep=1):
    namespace = load_definitions(
        "pithtrain/models/qwen3_omni_moe.py",
        ("_combine_expert_order",),
        dict(torch=torch, nn=nn, distributed=SimpleNamespace(ep_size=ep)),
    )
    namespace.update(
        load_definitions(
            "pithtrain/models/qwen3_omni_talker_pipeline.py", ("_TalkerMoeInputs",), namespace
        )
    )
    return namespace


def extract_class_methods(name, methods, namespace):
    cls = type(name, (nn.Module,), {})
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    for method in methods:
        node = next(
            n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == method
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        setattr(cls, method, namespace[method])
    return cls


@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_talker_compiled_stage5_preserves_shared_expert_order_and_all_gradients(ep, dtype):
    torch.manual_seed(941)
    cls = extract_class_methods(
        "Qwen3OmniMoeTalkerPipelineDecoderLayer", ("forward_stage5",), method_namespace(ep)
    )
    layer = cls()
    residual = torch.randn(2, 3, 16, dtype=dtype, requires_grad=True)
    weights = torch.randn(6, 2, dtype=dtype, requires_grad=True)
    indexes = torch.arange(12) if ep == 1 else torch.tensor([9, 2, 5, 4, 10, 1])
    output = torch.randn(len(indexes), 8, dtype=dtype, requires_grad=True)
    # Independent local token aggregation, including sparse nonordered EP lanes.
    pieces = []
    for token in range(6):
        take = indexes // 2 == token
        pieces.append((output[take] * weights.flatten()[indexes[take], None]).sum(0))
    routed = torch.stack(pieces).reshape(2, 3, 8)
    expected = residual[..., :8] + (routed + residual[..., 8:])
    actual = layer.forward_stage5(output, indexes, weights, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(
        actual.float().square().sum(), (output, weights, residual), retain_graph=True
    )
    expected_grads = torch.autograd.grad(
        expected.float().square().sum(), (output, weights, residual)
    )
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        # Same FP32 fullgraph tolerance as existing test_omni_stage5_compile:
        # Inductor's reduction scheduling may differ by one FP32 ULP.
        torch.testing.assert_close(observed, reference, rtol=1e-5, atol=2e-6)


def test_talker_stage1_computes_shared_expert_before_dispatch_without_folding_residual():
    from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerRMSNorm

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, hidden, rotary, cu):
            return self.proj(hidden)

    class SharedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(8, 12)
            self.up = nn.Linear(8, 12)
            self.down = nn.Linear(12, 8)

        def forward(self, value, up_value=None):
            other = value if up_value is None else up_value
            return self.down(torch.nn.functional.silu(self.gate(value)) * self.up(other))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts_per_tok = 2
            self.shared_expert = SharedMLP()
            self.shared_expert_gate = nn.Linear(8, 1, bias=False)
            self.router = nn.Linear(8, 3, bias=False)

        def shared_forward(self, hidden):
            return self.shared_expert(hidden) * self.shared_expert_gate(hidden).sigmoid()

        def gate(self, hidden, replay):
            probabilities = self.router(hidden.reshape(-1, 8)).softmax(-1)
            weights, indexes = probabilities.topk(2, -1)
            return indexes, weights, None

    cls = extract_class_methods(
        "Qwen3OmniMoeTalkerPipelineDecoderLayer",
        ("_forward_shared_expert", "forward_stage1_compute"),
        method_namespace(),
    )
    actual = cls()
    actual.self_attn = Attention()
    actual.mlp = MLP()
    actual.input_layernorm = Qwen3OmniMoeTalkerRMSNorm(8, 1e-6)
    actual.post_attention_layernorm = Qwen3OmniMoeTalkerRMSNorm(8, 1e-6)
    reference = copy.deepcopy(actual)
    left = torch.randn(2, 3, 8, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    output, packed, indexes, weights, _ = actual.forward_stage1_compute(left, None)
    residual = right + reference.self_attn(reference.input_layernorm(right), None, None)
    normalized = reference.post_attention_layernorm(residual)
    shared = reference.mlp.shared_forward(normalized)
    idx, probs, _ = reference.mlp.gate(normalized, None)
    expected_slots = normalized.reshape(-1, 1, 8).expand(-1, 2, -1)
    torch.testing.assert_close(output, expected_slots, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(packed, torch.cat((residual, shared), -1), rtol=1e-5, atol=1e-6)
    assert torch.equal(indexes, idx)
    (output.square().sum() + packed.square().sum() + weights.square().sum()).backward()
    (
        expected_slots.square().sum()
        + residual.square().sum()
        + shared.square().sum()
        + probs.square().sum()
    ).backward()
    for (name, observed), (_, expected) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert observed.grad is not None and expected.grad is not None, name
        torch.testing.assert_close(observed.grad, expected.grad, rtol=1e-5, atol=2e-6, msg=name)
    torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=2e-6)


def test_pipeline_parameter_names_and_fsdp_edges_are_not_duplicate_modules():
    from tests.test_omni_talker_native import tiny_config

    cfg = tiny_config().text_config

    class Layer(nn.Module):
        def __init__(self, config, index):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(config.hidden_size))

    namespace = method_namespace()
    namespace.update(
        training=SimpleNamespace(fp8=False),
        BF16Linear=nn.Linear,
        Qwen3OmniMoeThinkerTextRotaryEmbedding=lambda config: nn.Identity(),
        Qwen3OmniMoeThinkerTextRMSNorm=lambda h, eps: nn.LayerNorm(h, eps=eps),
        Qwen3OmniMoeTalkerPipelineDecoderLayer=Layer,
        layer_partition=lambda n, count, stage: range(n),
    )
    cls = extract_class_methods(
        "Qwen3OmniMoeTalkerTextModel",
        ("__init__", "layers", "embed_tokens", "norm", "lm_head", "fsdp_edge_modules"),
        namespace,
    )
    model = cls(cfg, -1)
    names = set(model.state_dict())
    assert "model.codec_embedding.weight" in names
    assert "codec_head.weight" in names
    assert "model.layers.0.weight" in names
    assert not any(n.startswith(("layers.", "embed_tokens.", "lm_head.", "norm.")) for n in names)
    assert set(model.fsdp_edge_modules()) == {
        model.model.codec_embedding,
        model.model.norm,
        model.codec_head,
    }
    assert model.model not in model.fsdp_edge_modules()


def test_registry_and_muon_codec_exclusions_execute_actual_functions():
    path = ROOT / "pithtrain/modules/training.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = ("model_class_for_config", "is_muon_param")
    marker = object()
    namespace = dict(
        torch=torch, resolve_text_model_config=lambda cfg: cfg, Qwen3OmniMoeTalkerTextModel=marker
    )
    for name in names:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    assert (
        namespace["model_class_for_config"](
            SimpleNamespace(model_type="qwen3_omni_moe_talker_text")
        )
        is marker
    )
    classify = namespace["is_muon_param"]
    value = torch.zeros(4, 8)
    for name in (
        "model.codec_embedding.weight",
        "codec_head.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ):
        assert not classify(name, value)
    assert classify("model.layers.0.mlp.shared_expert.gate_proj.weight", value)
    assert classify("layers.0.mlp.shared_expert_gate.weight", value)  # Existing Qwen3.5 policy.


def test_actual_fsdp_edge_selection_preserves_talker_container_and_shards_children_first():
    from dataclasses import dataclass, replace

    @dataclass
    class Policy:
        param_dtype: object
        reduce_dtype: object
        output_dtype: object
        cast_forward_inputs: bool

    class Mesh:
        def __getitem__(self, key):
            return self

        def _flatten(self, *args):
            return self

    class Stage(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.codec_embedding = nn.Embedding(4, 8)
            self.model.norm = nn.LayerNorm(8)
            self.codec_head = nn.Linear(8, 4, bias=False)
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = nn.Linear(8, 8)
            self.model.layers = nn.ModuleDict({"0": layer})

        @property
        def layers(self):
            return self.model.layers

        def fsdp_edge_modules(self):
            return (self.model.codec_embedding, self.model.norm, self.codec_head)

    calls = []
    path = ROOT / "pithtrain/modules/training.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "apply_fsdp"
    )
    fake_torch = SimpleNamespace(
        float32=torch.float32,
        distributed=SimpleNamespace(
            fsdp=SimpleNamespace(register_fsdp_forward_method=lambda *args: None)
        ),
    )
    namespace = dict(
        torch=fake_torch,
        replace=replace,
        MixedPrecisionPolicy=Policy,
        distributed=SimpleNamespace(expt_mesh=Mesh(), attn_mesh=Mesh()),
        training=SimpleNamespace(PARAM_DTYPE=torch.bfloat16),
        fully_shard=lambda child, **kwargs: calls.append(child),
        PIPELINE_STAGE_MODELS=(Stage,),
        Qwen3OmniMoeTalkerTextModel=Stage,
        FSDPModule=type("FSDPModule", (), {}),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    model = nn.Sequential(Stage(), Stage())
    namespace["apply_fsdp"](model)
    for stage in model:
        assert stage.model not in calls
        layer = stage.layers["0"]
        assert calls.index(layer.mlp.experts) < calls.index(layer) < calls.index(stage)
        assert all(calls.index(edge) < calls.index(layer) for edge in stage.fsdp_edge_modules())


@pytest.mark.parametrize(
    "relative",
    [
        "tests/configs/qwen3_omni_talker_text/tiny.json",
        "examples/pretrain_lm/qwen3-omni-talker-text/config.json",
    ],
)
def test_talker_pipeline_configs_resolve_offline(relative):
    from transformers import AutoConfig

    from pithtrain.modules.model_config import resolve_text_model_config

    cfg = AutoConfig.from_pretrained(ROOT / relative, local_files_only=True)
    assert resolve_text_model_config(cfg) is cfg
    assert cfg.model_type == "qwen3_omni_moe_talker_text"
    assert cfg.shared_expert_intermediate_size > 0
    assert cfg.num_experts % 2 == 0
