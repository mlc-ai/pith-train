import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration as HFTalker,
)

from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration


def tiny_config(*, norm_topk=True, attention_bias=False, sliding=False):
    text = dict(
        vocab_size=41,
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=norm_topk,
        attention_bias=attention_bias,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000,
            "mrope_section": [2, 1, 1],
            "interleaved": True,
        },
    )
    predictor = dict(
        vocab_size=23,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_code_groups=16,
        attention_bias=attention_bias,
        attention_dropout=0.0,
        max_position_embeddings=128,
        sliding_window=4 if sliding else None,
        layer_types=["sliding_attention" if sliding else "full_attention"] * 2,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000},
    )
    config = Qwen3OmniMoeTalkerConfig(
        text_config=text,
        code_predictor_config=predictor,
        thinker_hidden_size=20,
        num_code_groups=16,
        spatial_merge_size=2,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.text_config._experts_implementation = "eager"
    config.code_predictor_config._attn_implementation = "eager"
    return config


def initialized_hf(config):
    model = HFTalker(config).eval()
    # HF 5.17's inherited initializer covers ThinkerTextExperts, not the
    # distinct TalkerTextExperts class. Supply bounded checkpoint-like raw
    # expert tensors before copying the exact same state into native.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".mlp.experts." in name or name.endswith(".mlp.gate.weight"):
                parameter.normal_(mean=0, std=0.02)
    return model


def reference_teacher_force(predictor, hidden, first, codes):
    embedded = [
        embedding(codes[..., i]) for i, embedding in enumerate(predictor.model.codec_embedding)
    ]
    sequence = torch.stack([hidden, first, *embedded[:-1]], 2)
    b, t, s, h = sequence.shape
    flat = sequence.reshape(b * t, s, h)
    masks = None
    if "sliding_attention" in predictor.config.layer_types:
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        kwargs = dict(
            config=predictor.config,
            inputs_embeds=flat,
            attention_mask=None,
            past_key_values=None,
            position_ids=torch.arange(s)[None],
        )
        # The HF model accepts an explicit mapping; its automatic path builds
        # only full_attention even when the config requests sliding layers.
        masks = {
            "full_attention": create_causal_mask(**kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**kwargs),
        }
    final = predictor.model(
        inputs_embeds=flat, attention_mask=masks, use_cache=False
    ).last_hidden_state
    logits = torch.stack([head(final[:, i + 1]) for i, head in enumerate(predictor.lm_head)], 1)
    return logits.reshape(b, t, len(embedded), -1), torch.stack([first, *embedded], 2).sum(2)


@pytest.mark.parametrize(
    "norm_topk,attention_bias,sliding", [(True, False, False), (False, True, True)]
)
def test_talker_full_tree_forward_all_gradients_and_update(norm_topk, attention_bias, sliding):
    torch.manual_seed(902)
    torch.set_num_threads(1)
    cfg = tiny_config(norm_topk=norm_topk, attention_bias=attention_bias, sliding=sliding)
    reference = initialized_hf(copy.deepcopy(cfg))
    native = Qwen3OmniMoeTalkerForConditionalGeneration(copy.deepcopy(cfg)).eval()
    assert native.state_dict().keys() == reference.state_dict().keys()
    native.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.randint(0, cfg.text_config.vocab_size, (2, 4))
    codes = torch.randint(0, cfg.code_predictor_config.vocab_size, (2, 4, 15))
    text = torch.randn(2, 4, cfg.thinker_hidden_size)
    condition = torch.randn_like(text)
    positions = torch.arange(4).view(1, 1, -1).expand(3, 2, -1).clone()
    positions[1] *= 2
    positions[2] *= 3
    outputs, inputs, opts = [], [], []
    for model, is_native in ((reference, False), (native, True)):
        a, b = text.clone().requires_grad_(), condition.clone().requires_grad_()
        inputs.append((a, b))
        # Both released projection branches and the first-code embedding are
        # actual conditioning, not zero-gradient anchors.
        embed = model.model.codec_embedding(ids)
        initial = embed + model.text_projection(a) + model.hidden_projection(b)
        if is_native:
            first = model(inputs_embeds=initial, position_ids=positions)
            residual = model.code_predictor.teacher_force(first.last_hidden_state, embed, codes)
            hidden, logits, summed = (
                first.last_hidden_state,
                residual.logits,
                residual.codec_embeddings,
            )
        else:
            hidden = model.model(
                inputs_embeds=initial, position_ids=positions, use_cache=False
            ).last_hidden_state
            logits, summed = reference_teacher_force(model.code_predictor, hidden, embed, codes)
        # A subsequent Talker input uses all sixteen codec embeddings. This is
        # the real graph that trains the final residual embedding table.
        following = model.model(
            inputs_embeds=summed + model.text_projection(a),
            position_ids=positions,
            **({} if is_native else {"use_cache": False}),
        )
        following = following if is_native else following.last_hidden_state
        first_logits = model.codec_head(hidden)
        next_logits = model.codec_head(following)
        outputs.append((hidden, first_logits, logits, summed, next_logits))
        loss = F.cross_entropy(first_logits.flatten(0, 1), ids.flatten(), reduction="sum")
        loss = loss + F.cross_entropy(logits.flatten(0, 2), codes.flatten(), reduction="sum")
        loss = loss + F.cross_entropy(next_logits.flatten(0, 1), ids.flatten(), reduction="sum")
        loss.backward()
        opts.append(torch.optim.SGD(model.parameters(), lr=0.01))
    for left, right in zip(*outputs):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    ref_params, params = dict(reference.named_parameters()), dict(native.named_parameters())
    assert ref_params.keys() == params.keys()
    for name in ref_params:
        assert ref_params[name].grad is not None, name
        assert params[name].grad is not None, name
        assert torch.isfinite(params[name].grad).all(), name
        torch.testing.assert_close(
            params[name].grad, ref_params[name].grad, rtol=2e-5, atol=2e-6, msg=name
        )
    for pair in zip(*inputs):
        torch.testing.assert_close(pair[0].grad, pair[1].grad, rtol=2e-5, atol=2e-6)
    for opt in opts:
        opt.step()
    for name in ref_params:
        torch.testing.assert_close(params[name], ref_params[name], rtol=2e-5, atol=2e-6, msg=name)


def test_teacher_force_matches_original_autoregressive_prefixes_without_target_leakage():
    torch.manual_seed(914)
    cfg = tiny_config()
    hf = initialized_hf(cfg)
    native = Qwen3OmniMoeTalkerForConditionalGeneration(copy.deepcopy(cfg)).eval()
    native.load_state_dict(hf.state_dict(), strict=True)
    hidden, first = torch.randn(1, 2, 24), torch.randn(1, 2, 24)
    codes = torch.randint(0, 23, (1, 2, 15))
    output = native.code_predictor.teacher_force(hidden, first, codes)
    sequence = [hidden.reshape(2, 24), first.reshape(2, 24)]
    for i in range(15):
        prefix = torch.stack(sequence, 1)
        expected = hf.code_predictor(
            inputs_embeds=prefix, generation_steps=i, use_cache=False
        ).logits[:, -1]
        torch.testing.assert_close(output.logits[0, :, i], expected, rtol=1e-5, atol=1e-6)
        sequence.append(hf.code_predictor.model.codec_embedding[i](codes[0, :, i]))
    changed = codes.clone()
    changed[..., -1] = (changed[..., -1] + 1) % 23
    other = native.code_predictor.teacher_force(hidden, first, changed)
    torch.testing.assert_close(other.logits, output.logits, rtol=0, atol=0)
    assert not torch.equal(other.codec_embeddings, output.codec_embeddings)


def test_talker_mask_and_explicit_positions_match_hf():
    torch.manual_seed(929)
    cfg = tiny_config()
    reference = initialized_hf(cfg)
    native = Qwen3OmniMoeTalkerForConditionalGeneration(copy.deepcopy(cfg)).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.randint(0, 41, (2, 5))
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    ref = reference.model(
        inputs_embeds=reference.model.codec_embedding(ids),
        position_ids=positions,
        attention_mask=mask,
        use_cache=False,
    ).last_hidden_state
    out = native(ids, attention_mask=mask, position_ids=positions)
    torch.testing.assert_close(out.last_hidden_state, ref, rtol=1e-5, atol=1e-6)


def test_released_talker_complete_parameter_shapes_without_allocating_payloads():
    # Source-controlled config is copied from the pinned public config in the
    # fixture below; meta construction reads no released tensor payloads.
    path = Path(__file__).parent / "fixtures/qwen3_omni_talker_config.json"
    config = Qwen3OmniMoeTalkerConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    config._attn_implementation = "eager"
    with torch.device("meta"):
        reference = HFTalker(copy.deepcopy(config))
        native = Qwen3OmniMoeTalkerForConditionalGeneration(copy.deepcopy(config))
    expected = {name: tuple(value.shape) for name, value in reference.state_dict().items()}
    observed = {name: tuple(value.shape) for name, value in native.state_dict().items()}
    assert expected == observed
    assert len(observed) == 397
    assert "model.codec_embedding.weight" in observed
    assert "model.embed_tokens.weight" not in observed


@pytest.mark.parametrize("bad", [-1, 15, True])
def test_predictor_rejects_invalid_codebook(bad):
    model = Qwen3OmniMoeTalkerForConditionalGeneration(tiny_config())
    with pytest.raises(ValueError, match="generation_steps"):
        model.code_predictor(inputs_embeds=torch.randn(1, 2, 24), generation_steps=bad)


@pytest.mark.parametrize("kind", ["text", "audio", "image", "video", "av", "mixed"])
def test_prompt_position_builder_matches_hf_with_padding_and_media(kind):
    from types import SimpleNamespace

    from pithtrain.models.qwen3_omni_talker_positions import TalkerPositionBuilder

    cfg = tiny_config()
    cfg.audio_config = SimpleNamespace(n_window=50)
    cfg.audio_end_token_id = 151670
    cfg.position_id_per_seconds = 13
    builder = TalkerPositionBuilder(cfg)
    hf = initialized_hf(cfg)
    vs, ve = cfg.vision_start_token_id, 151653
    audio_start, audio_end, audio = (
        cfg.audio_start_token_id,
        cfg.audio_end_token_id,
        cfg.audio_token_id,
    )
    image, video = cfg.image_token_id, cfg.video_token_id
    prefix, suffix = [151644, 123, 777], [151645, 151644, 789, 100, 101, 102, 103, 104, 105]
    data = dict(image_grid_thw=None, video_grid_thw=None, audio_seqlens=None, second_per_grids=None)
    if kind == "text":
        media = []
    elif kind == "audio":
        media = [audio_start, audio, audio, audio_end]
        data["audio_seqlens"] = torch.tensor([16])
    elif kind == "image":
        media = [vs, image, image, ve]
        data["image_grid_thw"] = torch.tensor([[1, 2, 4]])
    elif kind == "video":
        media = [vs, video, video, ve]
        data["video_grid_thw"] = torch.tensor([[2, 2, 2]])
        data["second_per_grids"] = torch.tensor([0.5])
    elif kind == "av":
        media = [vs, audio_start, video, audio, audio, video, audio_end, ve]
        data["video_grid_thw"] = torch.tensor([[2, 2, 2]])
        data["second_per_grids"] = torch.tensor([0.5])
        data["audio_seqlens"] = torch.tensor([16])
    else:
        media = [vs, image, ve, 121, audio_start, audio, audio, audio_end]
        data["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        data["audio_seqlens"] = torch.tensor([16])
    row = prefix + media + suffix
    ids = torch.tensor([[0, 0, *row], [0, 0, *row]])
    mask = torch.ones_like(ids)
    mask[:, :2] = 0
    # Media arrays are indexed across rows, as in the public processor contract.
    data = {k: None if v is None else torch.cat((v, v)) for k, v in data.items()}
    kwargs = dict(attention_mask=mask, use_audio_in_video=kind == "av", **data)
    expected = hf.get_rope_index(ids, **kwargs)
    observed = builder.get_rope_index(ids, **kwargs)
    for left, right in zip(expected, observed):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_native_forward_gradients_match_fp32_master_bfloat16_autocast_contract():
    torch.manual_seed(937)
    cfg = tiny_config()
    reference = initialized_hf(copy.deepcopy(cfg))
    native = Qwen3OmniMoeTalkerForConditionalGeneration(cfg).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.randint(0, 41, (1, 5))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        left = reference.model(
            inputs_embeds=reference.model.codec_embedding(ids), use_cache=False
        ).last_hidden_state
        right = native(ids).last_hidden_state
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        ref_loss = reference.codec_head(left).float().square().sum()
        native_loss = native.codec_head(right).float().square().sum()
    ref_loss.backward()
    native_loss.backward()
    for name, param in native.named_parameters():
        if name.startswith(("model.", "codec_head.")):
            expected = dict(reference.named_parameters())[name]
            assert param.grad is not None and expected.grad is not None, name
            torch.testing.assert_close(param.grad, expected.grad, rtol=0, atol=0, msg=name)


def test_direct_bfloat16_hf_rotary_buffer_quantization_is_a_distinct_contract():
    # This is an explicit diagnostic distinction, not a parity acceptance gate.
    cfg = tiny_config()
    hf = initialized_hf(copy.deepcopy(cfg))
    original = hf.model.rotary_emb.inv_freq.clone()
    hf.to(torch.bfloat16)
    assert not torch.equal(original, hf.model.rotary_emb.inv_freq.float())
    native = Qwen3OmniMoeTalkerForConditionalGeneration(cfg).to(torch.bfloat16)
    position = torch.arange(128).view(1, 1, -1).expand(3, 1, -1)
    hidden = torch.zeros(1, 128, 24, dtype=torch.bfloat16)
    hf_cos, hf_sin = hf.model.rotary_emb(hidden, position)
    native_cos, native_sin = native.model.rotary_emb(hidden, position)
    assert not (torch.equal(hf_cos, native_cos) and torch.equal(hf_sin, native_sin))


def test_complete_talker_all_parameter_gradients_under_bfloat16_autocast():
    # Same full-tree target, all 15 heads, both projections and next-time
    # embedding use as FP32; no gradients are omitted from this contract.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        test_talker_full_tree_forward_all_gradients_and_update(True, False, False)


def test_talker_packed_position_reset_preserves_document_isolation():
    torch.manual_seed(953)
    cfg = tiny_config()
    reference = initialized_hf(copy.deepcopy(cfg))
    native = Qwen3OmniMoeTalkerForConditionalGeneration(cfg).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.randint(0, 41, (1, 6))
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    expected = reference.model(
        inputs_embeds=reference.model.codec_embedding(ids), position_ids=positions, use_cache=False
    ).last_hidden_state
    observed = native(ids, position_ids=positions).last_hidden_state
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    embeddings = native.get_input_embeddings()(ids).detach().requires_grad_()
    independent = native(inputs_embeds=embeddings, position_ids=positions).last_hidden_state
    derivative = torch.autograd.grad(independent[:, 3:].square().sum(), embeddings)[0]
    # Exact cross-document Jacobian zero establishes isolation. Changing the
    # first document can change the expert GEMM batch shape and FP32 last bits.
    torch.testing.assert_close(
        derivative[:, :3], torch.zeros_like(derivative[:, :3]), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_projection_strided_payload_matches_contiguous_hf(dtype):
    # Pipeline speech features are slices of a [embedding, hidden, logits]
    # payload. Test learned (nonzero) biases, not just initialization.
    torch.manual_seed(841)
    torch.set_num_threads(1)
    cfg = tiny_config()
    hf = initialized_hf(copy.deepcopy(cfg)).text_projection
    native = Qwen3OmniMoeTalkerForConditionalGeneration(cfg).text_projection
    with torch.no_grad():
        for parameter in hf.parameters():
            parameter.normal_(std=0.02)
    native.load_state_dict(hf.state_dict())
    payload = torch.randn(2, 13, 3 * cfg.thinker_hidden_size, dtype=dtype, requires_grad=True)
    strided = payload[..., : cfg.thinker_hidden_size]
    compact = strided.detach().contiguous().requires_grad_()
    assert not strided.is_contiguous()
    outputs = []
    for model, inputs in ((native, strided), (hf, compact)):
        parameters = {name: value.to(dtype) for name, value in model.named_parameters()}
        outputs.append(torch.func.functional_call(model, parameters, (inputs,)))
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    cotangent = torch.randn_like(outputs[0])
    for output in outputs:
        output.backward(cotangent)
    torch.testing.assert_close(
        payload.grad[..., : cfg.thinker_hidden_size], compact.grad, rtol=0, atol=0
    )
    assert torch.count_nonzero(payload.grad[..., cfg.thinker_hidden_size :]) == 0
    for (name, parameter), (ref_name, reference) in zip(
        native.named_parameters(), hf.named_parameters()
    ):
        assert name == ref_name and parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0, msg=name)


def test_headwise_boundary_preserves_frozen_weights_and_single_forward():
    from pithtrain.models.qwen3_omni_talker import _headwise_backward

    torch.manual_seed(194)
    module = torch.nn.Linear(8, 8).bfloat16()
    module.bias.requires_grad_(False)
    initial = {name: value.clone() for name, value in module.state_dict().items()}
    x = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    directions = torch.randn(4, 3, 8, dtype=torch.bfloat16)
    calls = []

    def forward(value):
        calls.append(value.shape)
        return module(value)

    heads = x.unsqueeze(0).expand(4, *x.shape)
    y = _headwise_backward(module, forward, heads)
    (y * directions).sum().backward()
    assert calls == [x.shape]
    assert module.bias.grad is None
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
    assert initial.keys() == module.state_dict().keys()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    # Independent per-head graph; retain reverse BF16 accumulation exactly.
    reference = torch.nn.Linear(8, 8).bfloat16()
    reference.load_state_dict(initial)
    reference.bias.requires_grad_(False)
    weights, inputs = [], []
    for direction in directions:
        leaf = x.detach().clone().requires_grad_()
        gradient = torch.autograd.grad(reference(leaf), (leaf, reference.weight), direction)
        inputs.append(gradient[0])
        weights.append(gradient[1])
    expected = weights[-1]
    for gradient in reversed(weights[:-1]):
        expected = expected + gradient
    torch.testing.assert_close(module.weight.grad, expected, rtol=0, atol=0)
    # The outer expand uses its ordinary sum; this boundary keeps all lanes.
    torch.testing.assert_close(x.grad, torch.stack(inputs).sum(0), rtol=0, atol=0)
