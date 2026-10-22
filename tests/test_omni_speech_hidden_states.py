"""CPU HF parity for the actual native DeepStack/speech hidden-state boundary.

Only GPU-specific decoder arithmetic is substituted with a tiny independent HF
layer. The native image class's actual methods and speech extraction function are
executed; this does not claim validation of native GPU kernels or DualPipeV.
"""

import ast
import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTextConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextDecoderLayer as HFLayer,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel as HFModel,
)

from pithtrain.models.qwen3_omni_speech import thinker_speech_forward

ROOT = Path(__file__).resolve().parents[1]


class HFBase(nn.Module):
    """Independent CPU decoder math under the native payload adapter."""

    def __init__(self, config, layer_id):
        super().__init__()
        self.idx = layer_id
        self.layer = HFLayer(config, layer_id)

    @property
    def mlp(self):
        return self.layer.mlp

    def reference_forward(self, hidden, rotary, cu_seqlens=None):
        assert cu_seqlens is None
        return self.layer(hidden, attention_mask=rotary[0], position_embeddings=rotary[1])


def actual_native_layer_class():
    path = ROOT / "pithtrain/models/qwen3_omni_image.py"
    tree = ast.parse(path.read_text())
    layer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3OmniMoeThinkerImageDecoderLayer"
    )
    wanted = {"__init__", "reference_forward", "reference_forward_with_hidden"}
    layer.body = [
        node for node in layer.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in layer.body} == wanted
    helper_path = ROOT / "pithtrain/models/qwen3_omni_image_inputs.py"
    helper = next(
        node
        for node in ast.parse(helper_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "inject_deepstack"
    )
    namespace = {"torch": torch, "Qwen3OmniMoeThinkerTextDecoderLayer": HFBase}
    module = ast.Module(body=[helper, layer], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[layer.name]


def config():
    cfg = Qwen3OmniMoeTextConfig(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=12,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        pad_token_id=None,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rope_parameters=dict(
            rope_type="default", rope_theta=1000000, mrope_section=[2, 1, 1], interleaved=True
        ),
    )
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    return cfg


class NativePayloadThinker(nn.Module):
    def __init__(self, hf, head, deepstack, visual_mask):
        super().__init__()
        self.stage_count, self.stage_index = 1, 0
        self.text_hidden_size = hf.config.hidden_size
        self.hidden_size = hf.config.hidden_size * (1 + len(deepstack))
        self.embed_tokens = copy.deepcopy(hf.embed_tokens)
        self.norm = copy.deepcopy(hf.norm)
        self.lm_head = copy.deepcopy(head)
        self.rotary = copy.deepcopy(hf.rotary_emb)
        self.deepstack = nn.ParameterList(
            [nn.Parameter(item.detach().clone()) for item in deepstack]
        )
        self.register_buffer("visual_mask", visual_mask)
        layer_class = actual_native_layer_class()
        self.layers = nn.ModuleDict(
            {
                str(index): layer_class(hf.config, index, len(deepstack))
                for index in range(len(hf.layers))
            }
        )
        for index, layer in enumerate(hf.layers):
            self.layers[str(index)].layer.load_state_dict(layer.state_dict(), strict=True)

    def forward_prolog(self, input_ids, model_context=None):
        embed = self.embed_tokens(input_ids)
        lanes = []
        for feature in self.deepstack:
            lane = torch.zeros_like(embed)
            lane[self.visual_mask] = feature
            lanes.append(lane)
        return torch.cat((embed, *lanes), -1)

    def forward_posemb(self, length, model_context=None):
        positions = torch.arange(length)[None, None].expand(3, 1, -1)
        dummy = self.embed_tokens.weight.new_zeros(1, length, self.text_hidden_size)
        mask = torch.full((1, 1, length, length), torch.finfo(dummy.dtype).min).triu(1)
        return mask, self.rotary(dummy, positions)

    def forward_epilog(self, payload):
        return self.lm_head(self.norm(payload[..., : self.text_hidden_size]))


def pair():
    torch.set_num_threads(1)
    torch.manual_seed(184)
    hf = HFModel(config()).float()
    with torch.no_grad():
        for name, value in hf.named_parameters():
            if ".experts." in name:
                value.normal_(std=0.02)
    head = nn.Linear(24, 64, bias=False)
    deepstack = nn.ParameterList([nn.Parameter(torch.randn(1, 24) * 0.4) for _ in range(2)])
    mask = torch.tensor([[False, True, False, False]])
    native = NativePayloadThinker(hf, head, deepstack, mask)
    return native, hf, head, deepstack, mask


def assert_gradients(actual, expected, name):
    assert actual.grad is not None and expected.grad is not None, name
    assert torch.isfinite(actual.grad).all() and torch.isfinite(expected.grad).all(), name
    numerator = (actual.grad.double() - expected.grad.double()).square().sum()
    denominator = actual.grad.double().square().sum() + expected.grad.double().square().sum()
    assert numerator / denominator.clamp_min(1e-30) < 1e-8, name


@pytest.mark.parametrize("selected", [0, 1, 2, 3])
def test_actual_speech_extraction_hf_hidden_tuple_and_all_gradients(selected):
    native, hf, head, deepstack, mask = pair()
    ids = torch.tensor([[1, 2, 3, 4]])
    actual = thinker_speech_forward(native, ids, {}, selected)
    expected = hf(
        input_ids=ids,
        visual_pos_masks=mask,
        deepstack_visual_embeds=list(deepstack),
        output_hidden_states=True,
        use_cache=False,
    )
    logits = head(expected.last_hidden_state)
    torch.testing.assert_close(actual.embeddings, expected.hidden_states[0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        actual.hidden_states, expected.hidden_states[selected], rtol=1e-5, atol=1e-7
    )
    torch.testing.assert_close(actual.logits, logits, rtol=1e-5, atol=1e-7)
    # The original post-injection selection is observably different, not a
    # vacuous all-zero lane fixture; final norm is checked separately above.
    if selected in (1, 2):
        payload = native.forward_prolog(ids)
        rotary = native.forward_posemb(ids.shape[1])
        for index in range(selected):
            payload = native.layers[str(index)].reference_forward(payload, rotary)
        assert not torch.allclose(payload[..., :24], expected.hidden_states[selected])
    actual_loss = (
        actual.logits.square().sum()
        + 0.17 * actual.hidden_states.square().sum()
        + 0.13 * actual.embeddings.square().sum()
    )
    expected_loss = (
        logits.square().sum()
        + 0.17 * expected.hidden_states[selected].square().sum()
        + 0.13 * expected.hidden_states[0].square().sum()
    )
    actual_loss.backward()
    expected_loss.backward()
    for name, parameter in hf.named_parameters():
        target = (
            "layers." + name.split(".", 2)[1] + ".layer." + name.split(".", 2)[2]
            if name.startswith("layers.")
            else name
        )
        assert_gradients(native.get_parameter(target), parameter, name)
    assert_gradients(native.lm_head.weight, head.weight, "lm_head")
    for index, feature in enumerate(deepstack):
        assert_gradients(native.deepstack[index], feature, f"deepstack.{index}")


@pytest.mark.parametrize("index", [0, 1, 2])
def test_existing_payload_forward_and_gradients_are_unchanged(index):
    native, _, _, _, _ = pair()
    layer = native.layers[str(index)]
    payload = torch.randn(1, 4, 72, requires_grad=True)
    rotary = native.forward_posemb(4)
    actual, pre = layer.reference_forward_with_hidden(payload, rotary)
    legacy = layer.reference_forward(payload, rotary)
    hidden, lanes = payload[..., :24], payload[..., 24:]
    raw = HFBase.reference_forward(layer, hidden, rotary)
    expected_hidden = raw + lanes[..., index * 24 : (index + 1) * 24] if index < 2 else raw
    expected = torch.cat((expected_hidden, lanes), -1)
    torch.testing.assert_close(pre, raw, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(legacy, expected, rtol=0, atol=0)
    variables = (payload, *layer.parameters())
    actual_grads = torch.autograd.grad(legacy.square().sum(), variables)
    expected_grads = torch.autograd.grad(expected.square().sum(), variables)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-7)
