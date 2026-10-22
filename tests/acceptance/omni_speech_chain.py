"""Source-bound complete Qwen3-Omni numerical and fresh-process resume gates.

No downloads, job submission, or GPU allocation. ``prepare`` is CPU-only.
``reference`` compares four actual AdamW updates with independent HF 5.17
modules. ``produce`` saves step two and records uninterrupted steps three/four;
``restore`` must be a different process and checks every restored tensor exactly.
The codec-to-waveform objective uses supplied integer labels: sampling is not a
differentiable connection from Talker to code2wav. This is tiny numerical
acceptance, not a released-weight quality or joint DualPipeV claim.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

REPO = Path(__file__).resolve().parents[2]
SEED = 739
MEDIA = ("text", "image", "audio", "video", "mixed")
FLOAT_INPUTS = ("pixel_values", "pixel_values_videos", "input_features")
GRADIENT_NSE = 1e-2
OUTPUT_NSE = 1e-3
STEPS = 4
SCHEMA = "qwen3_omni_complete_speech_acceptance_v1"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
RELEASED_CONFIG_SHA256 = "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"


class PreconditionError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def source_identity(expected):
    actual = git("rev-parse", "HEAD")
    if len(expected) != 40 or actual != expected or git("status", "--porcelain"):
        raise PreconditionError("Exact clean, committed source required; no inherited evidence")
    return dict(commit=actual, tree=git("rev-parse", "HEAD^{tree}"))


def nse(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("Tensor shape/dtype differs")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Nonfinite tensor")
    a, b = actual.detach().double(), expected.detach().double()
    denominator = (a.square() + b.square()).sum()
    return 0.0 if denominator == 0 else float((a - b).square().sum() / denominator)


def compare_named_gradients(actual, expected):
    """No magnitude floor, omitted parameter, or missing-gradient exemption."""
    if actual.keys() != expected.keys():
        raise AssertionError("Parameter/input gradient identities differ")
    report, failures = {}, []
    for name in actual:
        left, right = actual[name].grad, expected[name].grad
        if left is None or right is None:
            report[name] = dict(
                status="FAIL",
                reason="missing gradient",
                native_present=left is not None,
                hf_present=right is not None,
            )
            failures.append(name)
            continue
        error = nse(left, right)
        report[name] = dict(
            status="PASS" if error < GRADIENT_NSE else "FAIL",
            nse=error,
            native_norm=float(left.double().norm()),
            hf_norm=float(right.double().norm()),
            max_abs=float((left.double() - right.double()).abs().max()),
        )
        if error >= GRADIENT_NSE:
            failures.append(name)
    return report, failures


def compare_update_deltas(actual, expected, actual_before, expected_before):
    """Whole-weight agreement cannot substitute for actual optimizer deltas."""
    if not actual.keys() == expected.keys() == actual_before.keys() == expected_before.keys():
        raise AssertionError("Optimizer update parameter identities differ")
    report, failures = {}, []
    for name in actual:
        left = actual[name].detach() - actual_before[name]
        right = expected[name].detach() - expected_before[name]
        try:
            error = nse(left, right)
            report[name] = dict(
                status="PASS" if error < GRADIENT_NSE else "FAIL",
                nse=error,
                native_delta_norm=float(left.double().norm()),
                hf_delta_norm=float(right.double().norm()),
            )
        except AssertionError as exc:
            report[name] = dict(status="FAIL", reason=str(exc))
        if report[name]["status"] != "PASS":
            failures.append(name)
    return report, failures


def exact(left, right, name="state"):
    if type(left) is not type(right):
        raise AssertionError(f"{name}: type differs")
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise AssertionError(f"{name}: tensor differs")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{name}: keys differ")
        for key in left:
            exact(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"{name}: length differs")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            exact(a, b, f"{name}.{index}")
    elif left != right:
        raise AssertionError(f"{name}: value differs")


def canonical(name):
    # Complete native Thinker stores text layers at its root; HF uses model.
    if name.startswith("thinker.") and not name.startswith(
        ("thinker.visual.", "thinker.audio_tower.", "thinker.lm_head.")
    ):
        return "thinker.model." + name[len("thinker.") :]
    return name


def tiny_config(thinker, released):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeCode2WavConfig,
        Qwen3OmniMoeConfig,
        Qwen3OmniMoeTalkerConfig,
    )

    text = dict(released["talker_config"]["text_config"])
    text.update(
        vocab_size=41,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=48,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        attention_dropout=0.0,
        max_position_embeddings=4096,
        rope_parameters=dict(
            rope_type="default", rope_theta=1000000, mrope_section=[4, 2, 2], interleaved=True
        ),
    )
    predictor = dict(released["talker_config"]["code_predictor_config"])
    predictor.update(
        vocab_size=23,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_code_groups=16,
        attention_dropout=0.0,
        rope_parameters=dict(rope_type="default", rope_theta=1000000),
    )
    for options in (text, predictor):
        # HF 5.17 migrates legacy rope_scaling over rope_parameters. The
        # released head_dim=128 sections must not override this tiny fixture.
        options.pop("rope_scaling", None)
        if options.get("layer_types") is not None:
            options["layer_types"] = options["layer_types"][: options["num_hidden_layers"]]
    talker_options = copy.deepcopy(released["talker_config"])
    talker_options.update(
        text_config=text,
        code_predictor_config=predictor,
        num_code_groups=16,
        thinker_hidden_size=thinker.text_config.hidden_size,
        accept_hidden_layer=2,
        spatial_merge_size=thinker.vision_config.spatial_merge_size,
        codec_eos_token_id=31,
        codec_nothink_id=32,
        codec_think_bos_id=33,
        codec_think_eos_id=34,
        codec_pad_id=35,
        codec_bos_id=36,
        speaker_id={"ethan": 37},
    )
    talker = Qwen3OmniMoeTalkerConfig(**talker_options)
    wav_options = copy.deepcopy(released["code2wav_config"])
    wav_options.update(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=48,
        num_hidden_layers=2,
        num_quantizers=16,
        codebook_size=23,
        decoder_dim=32,
        upsample_rates=[2, 2],
        upsampling_ratios=[2, 2],
        sliding_window=3,
    )
    if wav_options.get("layer_types") is not None:
        wav_options["layer_types"] = wav_options["layer_types"][: wav_options["num_hidden_layers"]]
    wav = Qwen3OmniMoeCode2WavConfig(**wav_options)
    complete = copy.deepcopy(released)
    complete.update(
        thinker_config=thinker.to_dict(),
        talker_config=talker.to_dict(),
        code2wav_config=wav.to_dict(),
    )
    return Qwen3OmniMoeConfig(**complete)


def make_sample(batch, config, index):
    inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.model_inputs.items()
    }
    ids = inputs["input_ids"]
    if ids.shape[0] != 1 or not (inputs["attention_mask"] == 1).all():
        raise PreconditionError("Fixture requires a single unpadded native media row")
    if (ids == config.im_start_token_id).any():
        raise PreconditionError("Prepared source must contain content, not nested chat turns")
    # Preserve all original media placeholder IDs and grids. Use official role IDs.
    user = ids.new_tensor([[config.im_start_token_id, config.user_token_id, 198]])
    assistant = ids.new_tensor([[config.im_start_token_id, config.assistant_token_id, 198]])
    response = ids.new_tensor([[40, 41, 42, 43]])
    ids = torch.cat((user, ids, assistant, response), 1)
    prompt = ids.shape[1] - response.shape[1]
    inputs.update(input_ids=ids, attention_mask=torch.ones_like(ids))
    inputs.pop("position_ids", None)
    labels = torch.full_like(ids, -100)
    labels[:, prompt - 1 : -1] = response
    generator = torch.Generator().manual_seed(SEED + index)
    codes = torch.randint(23, (1, 4, 16), generator=generator)
    position_kwargs = {
        key: inputs[key] for key in ("image_grid_thw", "video_grid_thw") if key in inputs
    }
    if "feature_attention_mask" in inputs:
        position_kwargs["audio_seqlens"] = inputs["feature_attention_mask"].sum(-1)
    if "video_second_per_grid" in inputs:
        position_kwargs["second_per_grids"] = inputs["video_second_per_grid"]
    position_kwargs["use_audio_in_video"] = bool(inputs.get("use_audio_in_video", False))
    return dict(
        input_ids=ids,
        model_context=inputs,
        labels=labels,
        codes=codes,
        code_lengths=torch.tensor([4]),
        prompt_lengths=[prompt],
        position_kwargs=position_kwargs,
        media_kind=MEDIA[index],
    )


def prepare(args):
    from tests.acceptance.omni_multimodal_training import source_inputs

    if torch.cuda.is_initialized():
        raise PreconditionError("Preparation must not initialize CUDA")
    bundle, _, batches, thinker = source_inputs(args.dataset)
    if (
        bundle["recipe"]["revision"] != REVISION
        or sha(args.released_config) != RELEASED_CONFIG_SHA256
    ):
        raise PreconditionError("Pinned released config/revision mismatch")
    released = json.loads(args.released_config.read_text(encoding="utf-8"))
    config = tiny_config(thinker, released)
    samples = [
        make_sample(batches["all" if kind == "mixed" else kind], config, i)
        for i, kind in enumerate(MEDIA)
    ]
    config_path, data_path = args.output / "config.json", args.output / "samples.pt"
    write_json(config_path, config.to_dict())
    torch.save(samples, data_path)
    report = dict(
        schema=SCHEMA,
        status="PREPARED_NOT_NUMERICALLY_VALIDATED",
        source=args.source,
        seed=SEED,
        media=list(MEDIA),
        code_groups=16,
        dataset_checksum=sha(args.dataset / "checksums.sha256"),
        processor_revision=bundle["recipe"]["revision"],
        released_config_revision=REVISION,
        released_config_sha256=RELEASED_CONFIG_SHA256,
        config_sha256=sha(config_path),
        samples_sha256=sha(data_path),
        codec_provenance="fixed synthetic RVQ labels; not audio-encoder ground truth",
        waveform_objective="deterministic sine-target squared-error sum/count",
        numerical_acceptance="NOT_RUN",
        cuda_initialized=torch.cuda.is_initialized(),
    )
    write_json(args.output / "fixture.json", report)
    return report


def load_fixture(path, commit):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    report = json.loads((path / "fixture.json").read_text(encoding="utf-8"))
    if report["source"]["commit"] != commit or report["schema"] != SCHEMA:
        raise PreconditionError("Prepared fixture belongs to a different source/contract")
    if (
        report["released_config_revision"] != REVISION
        or report["released_config_sha256"] != RELEASED_CONFIG_SHA256
    ):
        raise PreconditionError("Fixture released source identity differs")
    for filename, field in (("config.json", "config_sha256"), ("samples.pt", "samples_sha256")):
        if sha(path / filename) != report[field]:
            raise PreconditionError(f"Fixture hash differs: {filename}")
    config = Qwen3OmniMoeConfig(**json.loads((path / "config.json").read_text(encoding="utf-8")))
    return config, torch.load(path / "samples.pt", map_location="cpu", weights_only=True), report


def move(value, device, *, gradients=False, _name=None):
    if isinstance(value, torch.Tensor):
        result = value.detach().clone().to(device)
        if result.is_floating_point() and _name in FLOAT_INPUTS:
            result = result.to(torch.bfloat16)
            if gradients:
                result.requires_grad_()
        return result
    if isinstance(value, dict):
        return {
            key: move(item, device, gradients=gradients, _name=key) for key, item in value.items()
        }
    if isinstance(value, list):
        return [move(item, device, gradients=gradients) for item in value]
    return value


class ComputeModel(nn.Module):
    """Real FP32 optimizer leaves; each functional forward casts to BF16 once.

    The owned native/HF module retains its original FP32 RoPE buffers. This does
    not claim equivalence to loading a whole HF model with .to(bfloat16).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.float()
        self.config = model.config

    @property
    def thinker(self):
        return self.model.thinker

    @property
    def talker(self):
        return self.model.talker

    @property
    def code2wav(self):
        return self.model.code2wav

    def forward(self, *args, **kwargs):
        parameters = {name: p.to(torch.bfloat16) for name, p in self.model.named_parameters()}
        return functional_call(
            self.model, (parameters, dict(self.model.named_buffers())), args, kwargs, strict=True
        )

    def decode_training_codes(self, codes):
        parameters = {name: p.to(torch.bfloat16) for name, p in self.code2wav.named_parameters()}
        return functional_call(
            self.code2wav, (parameters, dict(self.code2wav.named_buffers())), (codes,), strict=True
        )


class HFChain(nn.Module):
    """Independent official HF component forwards and conditioning expressions."""

    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeCode2Wav,
            Qwen3OmniMoeTalkerForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        self.config = copy.deepcopy(config)
        for cfg in (self.config.thinker_config.text_config, self.config.talker_config.text_config):
            cfg._experts_implementation = "eager"
        self.config.thinker_config._attn_implementation = "eager"
        self.config.talker_config._attn_implementation = "eager"
        self.config.code2wav_config._attn_implementation = "eager"
        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration(self.config.thinker_config)
        self.talker = Qwen3OmniMoeTalkerForConditionalGeneration(self.config.talker_config)
        self.code2wav = Qwen3OmniMoeCode2Wav(self.config.code2wav_config)
        # HF 5.17 does not initialize raw Talker expert storage through post_init.
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if ".mlp.experts." in name or name.endswith(".mlp.gate.weight"):
                    parameter.normal_(0, 0.02)

    def forward(
        self,
        input_ids,
        model_context,
        codes,
        code_lengths,
        *,
        prompt_lengths,
        speaker="ethan",
        position_kwargs=None,
        codebook_mask=None,
    ):
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration as HFComposite,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFPositions,
        )

        from tests.acceptance.omni_multimodal_training import positions
        from tests.acceptance.omni_reference_attention import reference_attention_kwargs

        if input_ids.shape[0] != 1 or codebook_mask is not None:
            raise PreconditionError(
                "The fixed reference fixture is unpadded B=1 with all codebooks"
            )
        inputs = dict(model_context)
        inputs.update(
            position_ids=positions(self.config.thinker_config, inputs),
            use_cache=False,
            output_hidden_states=True,
            **reference_attention_kwargs(self.thinker),
        )
        output = self.thinker(**inputs)
        embed, hidden = (
            output.hidden_states[0],
            output.hidden_states[self.config.talker_config.accept_hidden_layer],
        )
        cfg = self.config
        mm = (
            (input_ids == cfg.thinker_config.image_token_id)
            | (input_ids == cfg.thinker_config.video_token_id)
            | (input_ids == cfg.thinker_config.audio_token_id)
        )
        specials = input_ids.new_tensor(
            [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]]
        )
        bos, eos, pad = self.talker.text_projection(
            self.thinker.get_input_embeddings()(specials)
        ).chunk(3, 1)
        start = prompt_lengths[0] - 3
        user = HFComposite._get_talker_user_parts(self, 0, start, mm, hidden, embed)
        assistant, assistant_ids, tail = HFComposite._get_talker_assistant_parts(
            self,
            start,
            input_ids.shape[1],
            cfg.talker_config.speaker_id[speaker],
            embed,
            pad,
            bos,
            eos,
        )
        prefix = torch.cat((user, assistant), 1)
        prefix_ids = torch.cat((input_ids[:, :start], assistant_ids), 1)
        # Upstream Talker config lacks audio_config; supply only official position metadata.
        position_cfg = copy.deepcopy(cfg.talker_config)
        position_cfg.audio_config = cfg.thinker_config.audio_config
        owner = SimpleNamespace(
            config=position_cfg, spatial_merge_size=cfg.talker_config.spatial_merge_size
        )
        owner.get_llm_pos_ids_for_vision = HFPositions.get_llm_pos_ids_for_vision.__get__(owner)
        pos, _ = HFPositions.get_rope_index(
            owner, prefix_ids, attention_mask=torch.ones_like(prefix_ids), **(position_kwargs or {})
        )
        frames, groups = codes.shape[1:]
        if not (code_lengths == frames).all():
            raise PreconditionError("Fixed full-chain fixture uses four complete RVQ frames")
        embeddings = [self.talker.get_input_embeddings()(codes[..., 0])]
        embeddings += [
            table(codes[..., i + 1])
            for i, table in enumerate(self.talker.code_predictor.model.codec_embedding)
        ]
        # Official Talker.prepare_inputs_for_generation concatenates all
        # codebook rows and reduces once. Sequential BF16 additions round at
        # every group and alter the upstream Thinker/audio cotangent.
        previous = torch.stack(embeddings, dim=-2).sum(dim=-2)
        trailing = torch.cat((tail, pad.expand(-1, frames, -1)), 1)[:, :frames]
        combined = torch.cat((prefix, previous + trailing), 1)
        extra = pos.amax().to(pos.dtype) + torch.arange(
            1, frames + 1, device=pos.device, dtype=pos.dtype
        )
        pos = torch.cat((pos, extra.view(1, 1, -1).expand(3, 1, -1)), -1)
        trunk = self.talker.model(
            inputs_embeds=combined,
            attention_mask=torch.ones(combined.shape[:2], device=combined.device),
            position_ids=pos,
            use_cache=False,
        ).last_hidden_state
        begin = prefix.shape[1] - 1
        first = self.talker.codec_head(trunk[:, begin:])
        residual = []
        for group in range(groups - 1):
            sequence = torch.stack([trunk[:, begin : begin + frames], *embeddings[: group + 1]], 2)
            values = self.talker.code_predictor.model(
                inputs_embeds=sequence.flatten(0, 1), use_cache=False
            ).last_hidden_state
            residual.append(
                self.talker.code_predictor.lm_head[group](values[:, -1]).view(1, frames, -1)
            )
        first_labels = torch.cat(
            (codes[..., 0], codes.new_full((1, 1), cfg.talker_config.codec_eos_token_id)), 1
        )
        return SimpleNamespace(
            first_logits=first,
            residual_logits=torch.stack(residual, 2),
            first_labels=first_labels,
            residual_labels=codes[..., 1:],
            thinker_logits=output.logits,
        )


def build_pair(config, device):
    from pithtrain.contexts import distributed, training
    from pithtrain.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
    from pithtrain.models.qwen3_omni_multimodal import Qwen3OmniMoeThinkerMultimodalModel
    from pithtrain.models.qwen3_omni_speech import Qwen3OmniMoeSpeechModel
    from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration
    from pithtrain.operators.grouped_linear import GroupedLinear
    from tests.acceptance.omni_reference_attention import configure_reference_attention

    distributed.device = device
    distributed.pp_size = distributed.cp_size = distributed.ep_size = 1
    distributed.pp_rank = distributed.cp_rank = distributed.ep_rank = 0
    distributed.pp_group = distributed.cp_group = distributed.ep_group = None
    training.fp8, training.Linear, training.GroupedLinear = False, nn.Linear, GroupedLinear
    training.deterministic_attention = True
    torch.manual_seed(SEED)
    hf = HFChain(config).to(device)
    attention = configure_reference_attention(
        hf.thinker, config.thinker_config, "native-matched-v2"
    )
    # Dense Talker, predictor and code2wav native implementations use eager math.
    for component in (hf.talker, hf.code2wav):
        component.set_attn_implementation("eager", allow_all_kernels=False)
    with torch.device(device):
        native = Qwen3OmniMoeSpeechModel(
            config,
            Qwen3OmniMoeThinkerMultimodalModel(config.thinker_config, -1),
            Qwen3OmniMoeTalkerForConditionalGeneration(config.talker_config),
            Qwen3OmniMoeCode2Wav(config.code2wav_config),
        )
    expected = dict(hf.named_parameters())
    actual = {canonical(name): p for name, p in native.named_parameters()}
    if actual.keys() != expected.keys():
        raise PreconditionError(
            f"Full component parameter keys differ: {set(actual) ^ set(expected)}"
        )
    with torch.no_grad():
        for name, p in actual.items():
            p.copy_(expected[name])
    return ComputeModel(native), ComputeModel(hf), attention


def speech_args(batch):
    return (
        (batch["input_ids"], batch["model_context"], batch["codes"], batch["code_lengths"]),
        dict(prompt_lengths=batch["prompt_lengths"], position_kwargs=batch["position_kwargs"]),
    )


def loss_terms(output, labels):
    from pithtrain.tasks.omni_objectives import speech_loss_sums, text_loss_sum

    codec = speech_loss_sums(
        output.first_logits, output.first_labels, output.residual_logits, output.residual_labels
    )
    text = text_loss_sum(output.thinker_logits, labels)
    return codec.loss_sum.sum(), codec.target_count.sum(), text.loss_sum, text.target_count


def waveform_loss(output, batch):
    from pithtrain.tasks.omni_objectives import TargetLoss

    target = (
        torch.sin(torch.arange(output.shape[-1], device=output.device, dtype=torch.float32) * 0.1)
        * 0.1
    )
    return TargetLoss(
        (output.float() - target).square().sum(),
        torch.tensor(output.numel(), device=output.device, dtype=torch.int64),
    )


def parameter_steps(model, optimizer, expected):
    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        state = optimizer.state.get(parameter)
        if state is None or "step" not in state or float(state["step"]) != expected:
            raise AssertionError(
                f"Incomplete optimizer update counter: {name}, expected {expected}"
            )
        result[name] = int(state["step"])
    return result


def reference(args, config, samples, native, hf, attention):
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, foreach=False)
        for model in (native, hf)
    ]
    reports = []
    for step in range(STEPS):
        actual_inputs, expected_inputs, output_errors = {}, {}, {}
        objectives = []
        # One combined objective reaches every component; no artificial zero anchors.
        for model_index, model in enumerate((native, hf)):
            sums, counts, text_sums, text_counts, waves = [], [], [], [], []
            outputs, aligned_labels = [], []
            for index, sample in enumerate(samples):
                batch = move(sample, args.device, gradients=True)
                leaves = {
                    f"{MEDIA[index]}.{key}": batch["model_context"][key]
                    for key in FLOAT_INPUTS
                    if key in batch["model_context"]
                }
                (actual_inputs if model_index == 0 else expected_inputs).update(leaves)
                positional, kwargs = speech_args(batch)
                output = model(*positional, **kwargs)
                value, count, text, text_count = loss_terms(output, batch["labels"])
                wave = model.decode_training_codes(batch["codes"].transpose(1, 2))
                sums.append(value)
                counts.append(count)
                text_sums.append(text)
                text_counts.append(text_count)
                waves.append(waveform_loss(wave, batch))
                outputs.append(
                    (output.thinker_logits, output.first_logits, output.residual_logits, wave)
                )
                aligned_labels.append((output.first_labels, output.residual_labels))
            if model_index == 0:
                native_outputs = outputs
                native_labels = aligned_labels
            else:
                for index, (left, right) in enumerate(
                    zip(native_labels, aligned_labels, strict=True)
                ):
                    exact(left, right, f"{MEDIA[index]}.aligned-labels")
                for index, (a, b) in enumerate(zip(native_outputs, outputs, strict=True)):
                    for name, left, right in zip(
                        ("text", "first", "residual", "waveform"), a, b, strict=True
                    ):
                        output_errors[f"{MEDIA[index]}.{name}"] = nse(left, right)
            objectives.append(
                sum(sums) / sum(counts)
                + sum(text_sums) / sum(text_counts)
                + sum(x.loss_sum for x in waves) / sum(x.target_count for x in waves)
            )
        for objective in objectives:
            objective.backward()
        actual = {canonical(n): p for n, p in native.model.named_parameters()}
        expected = dict(hf.model.named_parameters())
        parameter_errors, bad_parameters = compare_named_gradients(actual, expected)
        input_errors, bad_inputs = compare_named_gradients(actual_inputs, expected_inputs)
        report = dict(
            step=step + 1,
            outputs=output_errors,
            parameters=parameter_errors,
            floating_inputs=input_errors,
            integer_inputs="NONDIFFERENTIABLE_BY_TYPE",
            native_loss=float(objectives[0].detach()),
            hf_loss=float(objectives[1].detach()),
            loss_close=bool(
                torch.isclose(objectives[0].detach(), objectives[1].detach(), rtol=1e-3, atol=1e-3)
            ),
            status="FAIL"
            if bad_parameters or bad_inputs or any(e >= OUTPUT_NSE for e in output_errors.values())
            else "PASS",
        )
        if not report["loss_close"]:
            report["status"] = "FAIL"
        write_json(args.output / f"step-{step + 1}.json", report)
        if report["status"] != "PASS":
            raise AssertionError(
                f"Unmodified numerical gate failed at step {step + 1}; inspect all-name report"
            )
        before = {name: p.detach().clone() for name, p in actual.items()}
        hf_before = {name: p.detach().clone() for name, p in expected.items()}
        for model, optimizer in zip((native, hf), optimizers, strict=True):
            optimizer.step()
            parameter_steps(model, optimizer, step + 1)
            optimizer.zero_grad(set_to_none=True)
        # Report every post-update parameter; FP32 master states must stay close too.
        report["post_update_parameters"] = {n: nse(p, expected[n]) for n, p in actual.items()}
        report["updated_parameters"] = [
            name for name, p in actual.items() if not torch.equal(before[name], p)
        ]
        missing_update = any(
            not any(name.startswith(component + ".") for name in report["updated_parameters"])
            for component in ("thinker", "talker", "code2wav")
        )
        report["update_deltas"], bad_deltas = compare_update_deltas(
            actual, expected, before, hf_before
        )
        bad_update = bool(bad_deltas) or any(
            value >= GRADIENT_NSE for value in report["post_update_parameters"].values()
        )
        report["post_update_status"] = "FAIL" if bad_update or missing_update else "PASS"
        report["status"] = report["post_update_status"]
        write_json(args.output / f"step-{step + 1}.json", report)
        if bad_update or missing_update:
            raise AssertionError("Post-AdamW parity or actual component update failed")
        reports.append(report)
        write_json(args.output / f"step-{step + 1}.json", report)
    return dict(
        status="PASS",
        steps=reports,
        reference_attention=attention,
        precision="FP32 leaves/AdamW state, BF16 compute weights, FP32 rotary buffers",
        full_parameter_count=len(actual),
        reference="independent official HF 5.17",
        waveform_bridge="integer labels; no gradient through codec sampling",
    )


class FixedData:
    def __init__(self, samples, fingerprint):
        self.samples, self.fingerprint = samples, fingerprint
        self.scanned_batches = self.completed_updates = 0
        self.last_update_scan = -1
        self.observed = {}

    def get_batch(self, scan_index, device):
        if scan_index != self.scanned_batches or not 0 <= scan_index < STEPS:
            raise ValueError("Unexpected fixed acceptance data cursor")
        self.scanned_batches += 1
        result = [move(sample, device) for sample in self.samples]
        self.observed[scan_index] = tensor_tree(result)
        return result

    def commit_update(self, *, scan_index, update_index):
        if scan_index != self.scanned_batches - 1 or update_index != self.completed_updates:
            raise ValueError("Unexpected acceptance update cursor")
        self.completed_updates += 1
        self.last_update_scan = scan_index

    def state_dict(self):
        return dict(
            fingerprint=self.fingerprint,
            scanned_batches=self.scanned_batches,
            completed_updates=self.completed_updates,
            last_update_scan=self.last_update_scan,
        )

    def validate_state_dict(self, state):
        if set(state) != set(self.state_dict()) or state["fingerprint"] != self.fingerprint:
            raise ValueError("Fixture identity differs")
        scan, update, last = (
            state[key] for key in ("scanned_batches", "completed_updates", "last_update_scan")
        )
        if any(type(x) is not int for x in (scan, update, last)) or not (
            0 <= update <= scan <= STEPS
        ):
            raise ValueError("Invalid acceptance cursor")
        if last != update - 1 or scan != update:
            raise ValueError("This fixed nonempty fixture requires one update per scan")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.scanned_batches, self.completed_updates, self.last_update_scan = (
            state[key] for key in ("scanned_batches", "completed_updates", "last_update_scan")
        )


def tensor_tree(value):
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(
            shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest()
        )
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_tree(item) for item in value]
    return value


def checkpoint_gate(args, samples, native, fixture):
    from pithtrain.tasks import omni_speech as task
    from pithtrain.tasks.sft_lm import SFTState

    cfg = task.OmniSpeechCfg(
        mode=args.training_mode,
        max_updates=4,
        max_batches=4,
        max_grad_norm=None,
        waveform_objective_id="speech-acceptance-fixed-sine-mse-v1",
    )
    active = task.configure_trainable_components(native, cfg.mode)
    optimizer = torch.optim.AdamW(active, lr=1e-4, weight_decay=0, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    data = FixedData(samples, fixture["samples_sha256"])
    state = SFTState()
    options = dict(
        cfg=cfg,
        model=native,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        state=state,
        token_group=None,
        gradient_reduction="sum",
    )
    if args.mode == "restore":
        producer = json.loads((args.producer / "result.json").read_text(encoding="utf-8"))
        if (
            producer["status"] != "PASS"
            or producer["pid"] == os.getpid()
            or producer["source"] != args.source
            or producer["training_mode"] != cfg.mode
            or producer["fixture_sha256"] != sha(args.prepared / "fixture.json")
        ):
            raise PreconditionError("Fresh producer identity/source/fixture/mode mismatch")
        task.load_checkpoint(args.producer / "step2.pt", **options)
        restored = args.output / "restored-step2.pt"
        task.save_checkpoint(restored, **options)
        exact(
            task._read_verified_checkpoint(restored),
            task._read_verified_checkpoint(args.producer / "step2.pt"),
        )
        parameter_steps(native, optimizer, 2)
        start = 3
    else:
        start = 1
    results = []
    for step in range(start, 5):
        cfg.max_updates = cfg.max_batches = step
        task.launch(
            cfg,
            data,
            model=native,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            state=state,
            waveform_objective=waveform_loss,
        )
        counters = parameter_steps(native, optimizer, step)
        path = args.output / f"step{step}.pt"
        task.save_checkpoint(path, **options)
        record = dict(step=step, parameter_steps=counters, inputs=data.observed[step - 1])
        if args.mode == "restore":
            expected = task._read_verified_checkpoint(args.producer / f"step{step}.pt")
            exact(task._read_verified_checkpoint(path), expected)
            original = json.loads(
                (args.producer / f"step{step}-record.json").read_text(encoding="utf-8")
            )
            exact(record["inputs"], original["inputs"], "next-batch")
        write_json(args.output / f"step{step}-record.json", record)
        results.append(record)
    return dict(
        status="PASS",
        training_mode=cfg.mode,
        steps=results,
        fixture_sha256=sha(args.prepared / "fixture.json"),
        checkpoint_contract="production omni_speech integrity-checked local per-rank; not DualPipeV/FSDP",
        fresh_process=args.mode == "restore",
        restored_boundary="EXACT" if args.mode == "restore" else "NOT_RUN",
    )


def verify_runtime_lock(path, observed):
    expected = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {"torch", "cuda", "transformers", "python", "packages"}
    if set(expected) != fields or set(observed) != fields:
        raise PreconditionError(
            "Runtime lock must identify exact Python/CUDA/packages/torch/transformers"
        )
    if expected != observed:
        raise PreconditionError("Installed dependencies differ from the reviewed runtime lock")
    return sha(path)


def failure_report(phase, exc):
    started = phase != "source_precondition"
    return dict(
        status="FAIL" if started else "NOT_RUN_PRECONDITION",
        failure_phase=phase,
        error_type=type(exc).__name__,
        error=str(exc),
        gpu_acceptance="FAIL_THIS_GATE" if started else "NOT_RUN",
    )


def runtime_precondition(args):
    if importlib.metadata.version("transformers") != "5.17.0":
        raise PreconditionError("Independent reference is pinned to transformers 5.17.0")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise PreconditionError(
            "This full-chain gate is single rank; distributed claims are separate"
        )
    if not torch.cuda.is_available():
        raise PreconditionError("CUDA hardware required for the actual native Thinker")
    if torch.cuda.get_device_name(0) != args.hardware:
        raise PreconditionError("Hardware differs from explicit --hardware lock")
    if torch.cuda.get_device_capability(0)[0] not in (9, 10):
        raise PreconditionError("Hopper/Blackwell required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise PreconditionError("Set deterministic CUBLAS_WORKSPACE_CONFIG before Python starts")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.device = torch.device("cuda", 0)
    observed = dict(
        torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        cuda=torch.version.cuda,
        python=platform.python_version(),
        packages=sorted(
            subprocess.check_output(
                [os.sys.executable, "-m", "pip", "freeze"], text=True
            ).splitlines()
        ),
    )
    lock = verify_runtime_lock(args.runtime_lock, observed)
    return dict(
        **observed,
        runtime_lock_sha256=lock,
        hardware=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "reference", "produce", "restore"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--released-config", type=Path, help="Pinned official config.json; metadata only"
    )
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hardware", help="Exact torch.cuda.get_device_name(0) value")
    parser.add_argument("--runtime-lock", type=Path, help="Reviewed exact dependency lock JSON")
    parser.add_argument(
        "--training-mode", choices=("text_and_codec", "waveform"), default="text_and_codec"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(
        schema=SCHEMA,
        mode=args.mode,
        status="NOT_RUN",
        pid=os.getpid(),
        seed=SEED,
        thresholds=dict(gradient_nse=GRADIENT_NSE, output_nse=OUTPUT_NSE),
        joint_cp_ep_pp="UNSUPPORTED_BY_COMPLETE_TASK",
        gpu_acceptance="NOT_RUN",
    )
    phase = "source_precondition"
    try:
        args.source = source_identity(args.expected_commit)
        result["source"] = args.source
        if importlib.metadata.version("transformers") != "5.17.0":
            raise PreconditionError("Preparation and validation both require transformers 5.17.0")
        if args.mode == "prepare":
            if args.dataset is None or args.released_config is None:
                raise PreconditionError("--dataset and --released-config are required")
            result.update(prepare(args))
        else:
            if (
                args.prepared is None
                or not args.hardware
                or args.runtime_lock is None
                or (args.mode == "restore" and args.producer is None)
            ):
                raise PreconditionError(
                    "--prepared/--hardware/--runtime-lock and restore --producer required"
                )
            result["runtime"] = runtime_precondition(args)
            config, samples, fixture = load_fixture(args.prepared, args.source["commit"])
            phase = "model_setup"
            native, hf, attention = build_pair(config, args.device)
            result["reference_attention"] = attention
            phase = "numerical" if args.mode == "reference" else "checkpoint_training"
            result.update(
                reference(args, config, samples, native, hf, attention)
                if args.mode == "reference"
                else checkpoint_gate(args, samples, native, fixture)
            )
            result["gpu_acceptance"] = "PASS_THIS_GATE_ONLY"
    except Exception as exc:
        result.update(failure_report(phase, exc))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
