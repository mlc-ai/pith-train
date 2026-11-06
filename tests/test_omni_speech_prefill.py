"""Official assistant-prefill boundaries and native generation control flow.

The HF helper comparison uses real tiny Talker modules. Generation control-flow
tests use an explicit scripted Thinker and replace only the codec sampling loop;
they do not claim native GPU Thinker forward or whole-chain numerical acceptance.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig, Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration as HFComposite,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration as HFTalker,
)

import pithtrain.models.qwen3_omni_speech as speech
from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration


@pytest.fixture(scope="module")
def official():
    options = dict(revision="26291f793822fb6be9555850f06dfe95f2d7e695", local_files_only=True)
    name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    try:
        return Qwen3OmniMoeProcessor.from_pretrained(name, **options), AutoConfig.from_pretrained(
            name, **options
        )
    except OSError:
        pytest.skip("Pinned official processor metadata must be cached locally")


def pair(official):
    torch.set_num_threads(1)
    torch.manual_seed(187)
    processor, original = official
    top = copy.deepcopy(original)
    cfg = top.talker_config
    cfg.thinker_hidden_size = 20
    cfg.accept_hidden_layer = 0
    cfg.codec_eos_token_id = 31
    cfg.codec_nothink_id, cfg.codec_think_bos_id, cfg.codec_think_eos_id = 32, 33, 34
    cfg.codec_pad_id, cfg.codec_bos_id, cfg.speaker_id = 35, 36, {"ethan": 37}
    for name in ("text_config", "code_predictor_config"):
        child = getattr(cfg, name)
        child.hidden_size = 24
        child.intermediate_size = 32
        child.num_hidden_layers = 1
        child.num_attention_heads = 4
        child.num_key_value_heads = 2
        child.head_dim = 8
        child.vocab_size = 41 if name == "text_config" else 23
        child._attn_implementation = "eager"
        child.rope_parameters = dict(rope_type="default", rope_theta=1000000)
    cfg.text_config.moe_intermediate_size = 12
    cfg.text_config.shared_expert_intermediate_size = 16
    cfg.text_config.num_experts, cfg.text_config.num_experts_per_tok = 4, 2
    cfg.text_config.rope_parameters.update(mrope_section=[2, 1, 1], interleaved=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._experts_implementation = "eager"
    hf = HFTalker(copy.deepcopy(cfg)).float().eval()
    native = Qwen3OmniMoeTalkerForConditionalGeneration(copy.deepcopy(cfg)).float().eval()
    native.load_state_dict(hf.state_dict(), strict=True)
    table = nn.Embedding(top.thinker_config.text_config.vocab_size, 20)
    return processor, top, native, hf, table


def rendered_prompt(processor, prefill):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if prefill:
        kwargs["enable_thinking"] = False
    rendered = processor.apply_chat_template([dict(role="user", content="Hello")], **kwargs)
    ids = processor.tokenizer.encode(rendered, add_special_tokens=False)
    header_text = rendered.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
    header_end = len(processor.tokenizer.encode(header_text, add_special_tokens=False))
    return torch.tensor([ids]), header_end


@pytest.mark.parametrize("prefill", [False, True])
def test_prefill_is_retained_exactly_like_hf_assistant_helper(official, prefill):
    processor, top, native, hf, table = pair(official)
    prompt, header_end = rendered_prompt(processor, prefill)
    generated = processor.tokenizer.encode("A reply.", add_special_tokens=False)
    ids = torch.cat((prompt, torch.tensor([generated])), 1)
    features = table(ids)
    actual = speech.prepare_talker_condition(
        native,
        features,
        features,
        ids,
        config=top,
        text_embedding=table,
        prompt_lengths=[header_end],
    )
    start = header_end - 3
    special = hf.text_projection(
        table(torch.tensor([[top.tts_bos_token_id, top.tts_eos_token_id, top.tts_pad_token_id]]))
    )
    bos, eos, pad = special.chunk(3, 1)
    proxy = SimpleNamespace(config=top, talker=hf)
    expected_user = HFComposite._get_talker_user_parts(
        proxy, 0, start, torch.zeros_like(ids, dtype=torch.bool), features, features
    )
    expected_assistant, expected_ids, expected_tail = HFComposite._get_talker_assistant_parts(
        proxy, start, ids.shape[1], 37, features, pad, bos, eos
    )
    torch.testing.assert_close(
        actual.inputs_embeds,
        torch.cat((expected_user, expected_assistant), 1),
        rtol=1e-5,
        atol=1e-7,
    )
    torch.testing.assert_close(
        actual.input_ids, torch.cat((ids[:, :start], expected_ids), 1), rtol=0, atol=0
    )
    torch.testing.assert_close(actual.trailing_text_hidden, expected_tail, rtol=1e-5, atol=1e-7)
    assert expected_tail.shape[1] == ids.shape[1] - header_end
    if prefill:
        assert prompt.shape[1] > header_end
        # No helper-level thinking filter: the first prefill token is the first text token.
        first = hf.text_projection(features[:, header_end : header_end + 1])
        torch.testing.assert_close(
            expected_assistant[:, -1:], first + hf.get_input_embeddings()(torch.tensor([[36]]))
        )


class ScriptedThinker(nn.Module):
    def __init__(self, table, output_tokens):
        super().__init__()
        self.embed_tokens = table
        self.stage_count, self.stage_index = 1, 0
        self.hidden_size = 20
        self.layers = nn.ModuleDict()
        self.norm = nn.Identity()
        self.output_tokens, self.calls = output_tokens, 0

    def forward_prolog(self, ids, model_context=None):
        return self.embed_tokens(ids)

    def forward_posemb(self, length, model_context=None):
        return None

    def forward_epilog(self, payload):
        logits = payload.new_full((*payload.shape[:2], self.embed_tokens.num_embeddings), -100.0)
        logits[..., self.output_tokens[min(self.calls, len(self.output_tokens) - 1)]] = 100.0
        self.calls += 1
        return logits


@pytest.mark.parametrize("prefill", [False, True])
def test_generate_distinguishes_original_prompt_from_assistant_header(
    official, monkeypatch, prefill
):
    processor, top, native, _, table = pair(official)
    prompt, header_end = rendered_prompt(processor, prefill)
    token = processor.tokenizer.encode("Hello", add_special_tokens=False)[0]
    assert top.thinker_config.text_config.eos_token_id is None
    thinker = ScriptedThinker(table, [token, top.im_end_token_id])
    model = speech.Qwen3OmniMoeSpeechModel(top, thinker, native, nn.Identity())
    observed = []

    def capture(talker, codec, condition, **kwargs):
        observed.append(condition)
        return "captured"

    monkeypatch.setattr(speech, "generate_speech_from_condition", capture)
    result = model.generate(
        prompt, dict(input_ids=prompt, attention_mask=torch.ones_like(prompt)), max_text_tokens=3
    )
    assert result.text_reached_eos
    assert result.sequences.tolist() == [prompt[0].tolist() + [token, top.im_end_token_id]]
    assert result.speech == "captured" and len(observed) == 1
    assert observed[0].trailing_text_hidden.shape[1] == prompt.shape[1] + 1 - header_end


def test_default_eos_immediate_response_produces_no_speech(official):
    processor, top, native, _, table = pair(official)
    prompt, _ = rendered_prompt(processor, False)
    thinker = ScriptedThinker(table, [top.im_end_token_id])
    model = speech.Qwen3OmniMoeSpeechModel(top, thinker, native, nn.Identity())
    result = model.generate(
        prompt, dict(input_ids=prompt, attention_mask=torch.ones_like(prompt)), max_text_tokens=1
    )
    assert result.text_reached_eos and result.speech is None
    assert result.sequences[0, -1] == top.im_end_token_id


@pytest.mark.parametrize("component", ["talker", "code2wav"])
def test_actual_fsdp_generation_rejected_before_forward_or_mode_change(
    official, tmp_path, component
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    processor, top, native, _, table = pair(official)
    prompt, _ = rendered_prompt(processor, False)
    thinker = ScriptedThinker(table, [top.im_end_token_id])
    model = speech.Qwen3OmniMoeSpeechModel(top, thinker, native, nn.Linear(4, 4))
    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        fully_shard(getattr(model, component), mesh=mesh)
        modes = [module.training for module in model.modules()]
        with pytest.raises(ValueError, match="FSDP is training-only"):
            model.generate(prompt, dict(attention_mask=torch.ones_like(prompt)))
        with pytest.raises(ValueError, match="FSDP is training-only"):
            speech.generate_speech_from_condition(model.talker, model.code2wav, None)
        assert thinker.calls == 0
        assert [module.training for module in model.modules()] == modes
    finally:
        dist.destroy_process_group()
