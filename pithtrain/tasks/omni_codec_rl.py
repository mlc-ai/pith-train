"""Local categorical Talker/RVQ policy GRPO with a fixed Thinker boundary.

Reuses the GRPO transaction and conditional-Adam speech checkpoint protocols.
No service, audio judge, ASR reward, download, or distributed transport is implicit.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from pithtrain.models.qwen3_omni_speech import (
    _append_positions,
    codec_embeddings,
    prepare_talker_condition,
    speech_teacher_force,
    thinker_speech_forward,
)
from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration
from pithtrain.modules.resume_semantics import implementation_contract, model_contract, seal
from pithtrain.tasks import omni_rl, omni_speech
from pithtrain.tasks.sft_lm import SFTState

PROMPT_KEYS = {
    "input_ids",
    "thinker_embeddings",
    "thinker_hidden",
    "special_embeddings",
    "prompt_lengths",
}
POSITION_KEYS = {
    "image_grid_thw",
    "video_grid_thw",
    "audio_seqlens",
    "second_per_grids",
    "use_audio_in_video",
}


def _tensor_identity(values):
    return omni_rl._weights_digest(omni_rl._cpu_tree(values))


def _processed_identity(value):
    if isinstance(value, torch.Tensor):
        return {"tensor_sha256": _tensor_identity({"value": value})}
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _processed_identity(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_processed_identity(v) for v in value]
    if value is None or type(value) in (bool, str, int, float):
        return value
    raise ValueError("Unverifiable processed Thinker input identity")


def _local(model):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise NotImplementedError("Codec policy currently requires one device, DP/CP/EP/PP1")
    if any(isinstance(m, FSDPModule) for m in model.modules()) or any(
        isinstance(p, DTensor) for p in model.parameters()
    ):
        raise NotImplementedError("Codec policy does not support sharded rollout/training")


@torch.no_grad()
def freeze_thinker_prompt(speech_model, batch):
    """Extract actual native frozen features once; keep Talker projections trainable.

    Supply an already processed/shift-aligned speech batch from the existing
    reader. The returned CPU tensors, source model/config and inputs are bound
    into the immutable rollout prompt; generation does not resample Thinker text.
    """
    _local(speech_model)
    modes = [(m, m.training) for m in speech_model.modules()]
    rng = omni_speech._rng_state()
    try:
        speech_model.eval()
        features = thinker_speech_forward(
            speech_model.thinker,
            batch["input_ids"],
            batch["model_context"],
            speech_model.config.talker_config.accept_hidden_layer,
        )
        cfg = speech_model.config
        ids = batch["input_ids"]
        special_ids = ids.new_tensor(
            [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]]
        )
        payload = dict(
            input_ids=ids,
            thinker_embeddings=features.embeddings,
            thinker_hidden=features.hidden_states,
            special_embeddings=speech_model.thinker.embed_tokens(special_ids),
            prompt_lengths=torch.as_tensor(batch["prompt_lengths"], device=ids.device),
        )
        for key, value in batch.get("position_kwargs", {}).items():
            if key not in POSITION_KEYS:
                raise ValueError(f"Unsupported codec prompt position metadata: {key}")
            payload[key] = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        source = seal(
            dict(
                thinker=model_contract(speech_model.thinker),
                thinker_weights=_tensor_identity(speech_model.thinker.state_dict()),
                processed_inputs=_processed_identity(
                    {
                        key: batch.get(key)
                        for key in (
                            "input_ids",
                            "model_context",
                            "prompt_lengths",
                            "position_kwargs",
                        )
                    }
                ),
                extraction=implementation_contract(freeze_thinker_prompt),
            )
        )
        return omni_rl._cpu_tree(payload), source["sha256"]
    finally:
        for module, mode in modes:
            module.training = mode
        omni_speech._set_rng(rng)


def action_layout(codes, lengths, reached_eos, *, max_frames, vocab):
    """Frame-major 16-code actions; EOS occupies only the next first-code slot."""
    if codes.ndim != 3 or codes.shape[-1] != 16 or codes.dtype != torch.long:
        raise ValueError("Expected integer B,T,16 RVQ actions")
    batch, frames, _ = codes.shape
    if frames > max_frames or lengths.shape != (batch,) or lengths.dtype != torch.long:
        raise ValueError("Invalid codec lengths/cap")
    if reached_eos.shape != (batch,) or reached_eos.dtype != torch.bool:
        raise ValueError("Explicit EOS versus cap status required")
    if (lengths < 0).any() or (lengths > frames).any():
        raise ValueError("Codec lengths exceed stored frames")
    if ((~reached_eos) & (lengths != max_frames)).any() or (
        reached_eos & (lengths >= max_frames)
    ).any():
        raise ValueError("EOS must precede the cap; truncation must reach it")
    active = torch.arange(frames, device=codes.device)[None] < lengths[:, None]
    if ((codes[active] < 0) | (codes[active] >= vocab)).any() or (codes[~active] != 0).any():
        raise ValueError("Invalid active code or nonzero padded codec action")
    actions = codes.new_zeros(batch, max_frames * 16 + 1)
    actions[:, : frames * 16] = codes.flatten(1)
    mask = torch.arange(actions.shape[1], device=codes.device)[None] < lengths[:, None] * 16
    rows = reached_eos.nonzero().flatten()
    actions[rows, lengths[rows] * 16] = vocab  # compact first-head EOS class
    mask[rows, lengths[rows] * 16] = True
    return actions, mask


class CodecPolicy(nn.Module):
    """Actual native Talker/projectors/RVQ predictor over frozen Thinker features."""

    def __init__(self, talker, config, *, max_frames=4, speaker="ethan", max_prompt_tokens=1024):
        super().__init__()
        _local(talker)
        if not isinstance(talker, Qwen3OmniMoeTalkerForConditionalGeneration):
            raise TypeError("Codec policy requires the native Qwen3-Omni Talker")
        if (
            type(max_frames) is not int
            or max_frames < 1
            or type(max_prompt_tokens) is not int
            or max_prompt_tokens < 1
        ):
            raise ValueError("Positive finite codec/prompt bounds are required")
        if config.model_type != "qwen3_omni_moe" or config.talker_config.num_code_groups != 16:
            raise ValueError("Only Qwen3-Omni 16-codebook policy is supported")
        if talker.config.to_dict() != config.talker_config.to_dict():
            raise ValueError("Talker runtime/config differ")
        self.talker, self.config = talker, copy.deepcopy(config)
        self.max_frames, self.speaker, self.max_prompt_tokens = (
            max_frames,
            speaker,
            max_prompt_tokens,
        )
        self.vocab = config.talker_config.code_predictor_config.vocab_size
        self.eos_id = config.talker_config.codec_eos_token_id
        if not self.vocab <= self.eos_id < config.talker_config.text_config.vocab_size:
            raise ValueError(
                "Codec EOS must be distinct from RVQ IDs and inside first-head vocabulary"
            )
        if (
            config.talker_config.text_config.attention_dropout
            or config.talker_config.code_predictor_config.attention_dropout
            or any(getattr(m, "attention_dropout", 0) for m in talker.modules())
        ):
            raise ValueError("On-policy codec reference requires zero attention dropout")

    def semantics(self):
        return seal(
            dict(
                model=model_contract(self),
                config=self.config.to_dict(),
                max_frames=self.max_frames,
                max_prompt_tokens=self.max_prompt_tokens,
                speaker=self.speaker,
                vocab=self.vocab,
                eos=self.eos_id,
                actions="frame-major-16-plus-optional-eos-v1",
                sampling="full-categorical-v1",
                implementation=dict(
                    recurrence=implementation_contract(speech_teacher_force),
                    objective=implementation_contract(omni_rl.grpo_loss_sum),
                    transaction=implementation_contract(omni_rl.OmniRLTrainer),
                    checkpoint=implementation_contract(omni_speech.save_checkpoint),
                ),
                thinker="frozen-detached-features",
                code2wav="outside-discrete-policy-gradient",
            )
        )

    def validate_prompt(self, prompt):
        if not PROMPT_KEYS <= prompt.keys() or set(prompt) - PROMPT_KEYS - POSITION_KEYS:
            raise ValueError("Incomplete/unknown frozen Thinker prompt fields")
        if any(not isinstance(v, torch.Tensor) or v.requires_grad for v in prompt.values()):
            raise ValueError("Prompt fields must be detached tensors")
        ids = prompt["input_ids"]
        if (
            ids.ndim != 2
            or ids.shape[0] != 1
            or not 0 < ids.shape[1] <= self.max_prompt_tokens
            or ids.dtype != torch.long
        ):
            raise ValueError("Codec policy requires one bounded unpadded prompt per microbatch")
        hidden = self.config.talker_config.thinker_hidden_size
        for key in ("thinker_embeddings", "thinker_hidden"):
            value = prompt[key]
            if (
                value.shape != (*ids.shape, hidden)
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid frozen Thinker feature geometry/value")
        specials = prompt["special_embeddings"]
        if (
            specials.shape != (1, 3, hidden)
            or not specials.is_floating_point()
            or not torch.isfinite(specials).all()
        ):
            raise ValueError("Invalid frozen TTS special embeddings")
        if prompt["prompt_lengths"].shape != (1,) or prompt["prompt_lengths"].dtype != torch.long:
            raise ValueError("One explicit assistant prompt boundary is required")

    def condition(self, prompt):
        self.validate_prompt(prompt)
        # Frozen encoder outputs retain their source dtype/identity. Project
        # using the admitted Talker dtype without mutating the sealed prompt.
        dtype = next(self.talker.text_projection.parameters()).dtype

        def special_embeddings(ids):
            expected = ids.new_tensor(
                [
                    [
                        self.config.tts_bos_token_id,
                        self.config.tts_eos_token_id,
                        self.config.tts_pad_token_id,
                    ]
                ]
            )
            if not torch.equal(ids, expected):
                raise ValueError("Unexpected special embedding request")
            return prompt["special_embeddings"].to(dtype=dtype)

        return prepare_talker_condition(
            self.talker,
            prompt["thinker_embeddings"].to(dtype=dtype),
            prompt["thinker_hidden"].to(dtype=dtype),
            prompt["input_ids"],
            config=self.config,
            text_embedding=special_embeddings,
            prompt_lengths=prompt["prompt_lengths"],
            speaker=self.speaker,
            position_kwargs={key: prompt[key] for key in POSITION_KEYS if key in prompt},
        )

    def decode_actions(self, actions, lengths, reached_eos):
        if actions.shape != (1, self.max_frames * 16 + 1) or actions.dtype != torch.long:
            raise ValueError("Fixed frame-major action geometry required")
        if (
            lengths.shape != (1,)
            or lengths.dtype != torch.long
            or not 0 <= int(lengths[0]) <= self.max_frames
        ):
            raise ValueError("Invalid codec action length")
        frames = int(lengths[0])
        codes = actions[:, : frames * 16].reshape(1, frames, 16)
        encoded, mask = action_layout(
            codes, lengths, reached_eos, max_frames=self.max_frames, vocab=self.vocab
        )
        if not torch.equal(encoded, actions):
            raise ValueError("Action padding/EOS/codec sequence is not canonical")
        return codes, mask

    def forward(self, actions, *, lengths, reached_eos, **prompt):
        codes, _ = self.decode_actions(actions, lengths, reached_eos)
        output = speech_teacher_force(self.talker, self.condition(prompt), codes, lengths)
        allowed = torch.cat(
            (torch.arange(self.vocab, device=actions.device), actions.new_tensor([self.eos_id]))
        )
        first = output.first_logits.float().index_select(-1, allowed).log_softmax(-1)
        residual = output.residual_logits.float().log_softmax(-1)
        if not torch.isfinite(first).all() or not torch.isfinite(residual).all():
            raise ValueError("Nonfinite codec policy distribution")
        # Log probabilities are valid categorical scores for the shared GRPO
        # objective. A finite -1e9 padded class has exactly zero FP32 probability;
        # residual heads never gain a nonexistent EOS action.
        residual = torch.nn.functional.pad(residual, (0, 1), value=-1e9)
        frames = codes.shape[1]
        rows = torch.cat((first[:, :frames, None], residual), 2).flatten(1, 2)
        rows = torch.cat((rows, first[:, frames : frames + 1]), 1)
        return torch.nn.functional.pad(
            rows, (0, 0, 0, self.max_frames * 16 + 1 - rows.shape[1]), value=-1e9
        )

    @torch.no_grad()
    def sample(self, prompt, generator):
        modes = [(m, m.training) for m in self.modules()]
        self.eval()
        try:
            condition = self.condition(prompt)
            embeds, attention = condition.inputs_embeds, condition.attention_mask
            history, selected, stopped = [], [], False
            for frame in range(self.max_frames):
                out = self.talker(
                    inputs_embeds=embeds,
                    attention_mask=attention,
                    position_ids=_append_positions(condition, frame),
                )
                raw = out.logits[:, -1].float()
                allowed = torch.cat(
                    (
                        torch.arange(self.vocab, device=raw.device),
                        torch.tensor([self.eos_id], device=raw.device),
                    )
                )
                logp = raw.index_select(-1, allowed).log_softmax(-1)
                token = torch.multinomial(logp.exp(), 1, generator=generator)
                selected.append(logp.gather(-1, token).squeeze())
                if int(token) == self.vocab:
                    stopped = True
                    break
                first = token[:, 0]
                predictor_input = torch.stack(
                    (out.last_hidden_state[:, -1], self.talker.get_input_embeddings()(first)), 1
                )
                groups = [first]
                for group, table in enumerate(self.talker.code_predictor.get_input_embeddings()):
                    prediction = self.talker.code_predictor(
                        inputs_embeds=predictor_input, generation_steps=group
                    )
                    logp = prediction.logits[:, -1].float().log_softmax(-1)
                    token = torch.multinomial(logp.exp(), 1, generator=generator)
                    selected.append(logp.gather(-1, token).squeeze())
                    groups.append(token[:, 0])
                    predictor_input = torch.cat((predictor_input, table(token)), 1)
                frame_codes = torch.stack(groups, -1)
                history.append(frame_codes)
                tail = (
                    condition.trailing_text_hidden[:, frame : frame + 1]
                    if frame < condition.trailing_text_hidden.shape[1]
                    else condition.tts_pad_embed
                )
                embeds = torch.cat(
                    (embeds, codec_embeddings(self.talker, frame_codes[:, None]) + tail), 1
                )
                attention = torch.cat((attention, attention.new_ones(1, 1)), 1)
            device = embeds.device
            codes = (
                torch.stack(history, 1)
                if history
                else torch.empty(1, 0, 16, device=device, dtype=torch.long)
            )
            lengths = torch.tensor([len(history)], device=device)
            eos = torch.tensor([stopped], device=device)
            actions, mask = action_layout(
                codes, lengths, eos, max_frames=self.max_frames, vocab=self.vocab
            )
            old = torch.zeros(actions.shape, dtype=torch.float32, device=device)
            old[mask] = torch.stack(selected)
            return actions, mask, old, lengths, eos
        finally:
            for module, mode in modes:
                module.training = mode


class CodecPolicyAdapter(omni_rl.TorchPolicyAdapter):
    """Single-device GRPO adapter; conditional Adam state uses speech's strict format."""

    def __init__(self, model, optimizer, *, scheduler=None):
        _local(model)
        if not isinstance(model, CodecPolicy) or type(optimizer) is not torch.optim.AdamW:
            raise ValueError("Codec adapter requires CodecPolicy and explicit AdamW")
        super().__init__(model, optimizer, scheduler=scheduler)
        self._residual_used = False

    def model_semantics(self):
        return self.model.semantics()

    def validate_batch_layout(self, batch, cfg, rewards):
        if (
            cfg.micro_batch_size != 1
            or batch.model_context is None
            or batch.media_inputs is not None
        ):
            raise ValueError("Codec policy requires explicit B1 frozen-feature contexts")
        for i, context in enumerate(batch.model_context):
            context = dict(context)
            lengths, eos = context.pop("lengths"), context.pop("reached_eos")
            self.model.validate_prompt(context)
            _, mask = self.model.decode_actions(batch.input_ids[i : i + 1], lengths, eos)
            if not torch.equal(mask, batch.token_mask[i : i + 1]) or not torch.equal(
                batch.input_ids, batch.sampled_token_ids
            ):
                raise ValueError(
                    "Codec action mask/target differs; no additional label shift permitted"
                )

    def backward(self, microbatches, objective):
        self._residual_used = any(int(mb.model_context["lengths"][0]) > 0 for mb in microbatches)
        return super().backward(microbatches, objective)

    def update(self, count, max_grad_norm):
        active = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                if self._residual_used or not name.startswith("talker.code_predictor."):
                    raise ValueError(f"Unexpected missing codec gradient: {name}")
                continue
            if not torch.isfinite(p.grad).all():
                raise ValueError("Nonfinite codec gradient; refusing update")
            active.append(p)
        if not active or int(count) <= 0:
            raise ValueError("No valid codec policy gradient/target")
        for p in active:
            p.grad.div_(count)
        norm = nn.utils.clip_grad_norm_(active, max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Nonfinite updated codec policy; restore a committed checkpoint")
        return float(norm)

    def _checkpoint(self, operation, path, trainer):
        cfg = omni_speech.OmniSpeechCfg()
        cfg.mode = "codec_only"
        cfg.max_updates = trainer.cfg.max_updates
        cfg.max_batches = trainer.cfg.max_requests
        cfg.max_grad_norm = trainer.cfg.max_grad_norm
        cfg.speaker = self.model.speaker
        operation(
            path,
            cfg=cfg,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            data=_CheckpointData(trainer),
            state=SFTState(trainer.requests, trainer.updates),
            token_group=None,
            gradient_reduction="sum",
        )

    def save(self, path, state):
        self._checkpoint(omni_speech.save_checkpoint, path, state)

    def load(self, path, state):
        self._checkpoint(omni_speech.load_checkpoint, path, state)


class _CheckpointData:
    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self):
        return dict(
            scanned_batches=self.trainer.requests,
            completed_updates=self.trainer.updates,
            rl_state=self.trainer.state_dict(),
        )

    def validate_state_dict(self, state):
        if set(state) != {"scanned_batches", "completed_updates", "rl_state"}:
            raise ValueError("Incomplete codec RL checkpoint transaction")
        self.trainer.validate_state_dict(state["rl_state"])
        if (
            state["scanned_batches"] != state["rl_state"]["requests"]
            or state["completed_updates"] != state["rl_state"]["updates"]
            or state["rl_state"]["rollout"]["calls"] != state["scanned_batches"]
            or state["rl_state"]["reward"]["calls"] != state["scanned_batches"]
            or state["rl_state"]["data_cursor"]
            != state["scanned_batches"] * self.trainer.cfg.group_size
        ):
            raise ValueError("Codec RL request/update cursors differ")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.trainer.load_state_dict(state["rl_state"])


class CodecPrompt:
    """Copied immutable feature/target record with explicit upstream provenance."""

    def __init__(self, identity, payload, source_sha256, target_codes, target_eos=True):
        if (
            not isinstance(identity, str)
            or not identity
            or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)
        ):
            raise ValueError("Explicit prompt ID and upstream source digest required")
        if (
            type(target_eos) is not bool
            or target_codes.ndim != 2
            or target_codes.shape[-1] != 16
            or target_codes.dtype != torch.long
        ):
            raise ValueError(
                "Exact-codec target must be T,16 integer codes plus explicit EOS status"
            )
        self.identity, self.source_sha256, self.target_eos = identity, source_sha256, target_eos
        self.payload = omni_rl._cpu_tree(payload)
        self.target_codes = target_codes.detach().cpu().clone()
        self.sha256 = self._digest()

    def _digest(self):
        return omni_rl.fingerprint(
            dict(
                id=self.identity,
                source=self.source_sha256,
                eos=self.target_eos,
                payload=_tensor_identity(self.payload),
                target=_tensor_identity({"codes": self.target_codes}),
            )
        )

    def validate(self):
        if self._digest() != self.sha256:
            raise ValueError("Frozen codec prompt/target changed")


class LocalCodecRollout:
    """Bounded in-process actual native sampling; caller owns outer time/memory admission."""

    def __init__(self, actor, reference, prompts, *, max_weight_bytes=64 << 20):
        for model in (actor, reference):
            _local(model)
            if not isinstance(model, CodecPolicy):
                raise TypeError("Local codec provider requires actual CodecPolicy models")
        if actor.semantics() != reference.semantics():
            raise ValueError("Actor and fixed reference policy semantics differ")
        if {p.data_ptr() for p in actor.parameters()} & {
            p.data_ptr() for p in reference.parameters()
        }:
            raise ValueError("Actor and fixed reference must not share parameter storage")
        if type(max_weight_bytes) is not int or max_weight_bytes < 1:
            raise ValueError("Explicit positive weight bound required")
        if any(
            sum(t.numel() * t.element_size() for t in m.state_dict().values()) > max_weight_bytes
            for m in (actor, reference)
        ):
            raise ValueError("Local codec policy exceeds admitted per-copy weight bytes")
        self.actor, self.reference, self.prompts = actor, reference, tuple(prompts)
        if not self.prompts or len({p.identity for p in self.prompts}) != len(self.prompts):
            raise ValueError("Unique explicit local codec prompts required")
        for prompt in self.prompts:
            prompt.validate()
            actor.validate_prompt(prompt.payload)
            action_layout(
                prompt.target_codes[None],
                torch.tensor([len(prompt.target_codes)]),
                torch.tensor([prompt.target_eos]),
                max_frames=actor.max_frames,
                vocab=actor.vocab,
            )
        self.reference_sha256 = _tensor_identity(reference.state_dict())
        self.config = dict(
            schema=1,
            semantics=actor.semantics(),
            reference_sha256=self.reference_sha256,
            prompts=[p.sha256 for p in self.prompts],
            max_weight_bytes=max_weight_bytes,
            implementation=implementation_contract(type(self)),
        )
        self.calls, self.installed = 0, None

    def _validate(self):
        for prompt in self.prompts:
            prompt.validate()
        if (
            [p.sha256 for p in self.prompts] != self.config["prompts"]
            or self.actor.semantics() != self.config["semantics"]
            or self.reference.semantics() != self.config["semantics"]
            or _tensor_identity(self.reference.state_dict()) != self.reference_sha256
        ):
            raise ValueError("Codec actor/reference semantics or frozen reference changed")

    def synchronize(self, snapshot):
        self._validate()
        snapshot.identity.validate()
        if (
            snapshot.semantics != self.actor.semantics()
            or snapshot.identity.semantics_sha256 != snapshot.semantics["sha256"]
            or snapshot.identity.contract.reference_policy_sha256 != self.reference_sha256
        ):
            raise ValueError("Published codec policy/reference semantics differ")
        current = self.actor.state_dict()
        if current.keys() != snapshot.payload.keys() or any(
            current[k].shape != v.shape
            or current[k].dtype != v.dtype
            or not torch.isfinite(v).all()
            for k, v in snapshot.payload.items()
        ):
            raise ValueError("Published codec weight layout/value mismatch")
        if _tensor_identity(snapshot.payload) != snapshot.identity.weights_sha256:
            raise ValueError("Published codec weights do not match identity")
        self.actor.load_state_dict(snapshot.payload, strict=True)
        if _tensor_identity(self.actor.state_dict()) != snapshot.identity.weights_sha256:
            raise ValueError("Codec actor installation readback differs")
        self.installed = snapshot.identity
        return self.installed

    @torch.no_grad()
    def generate(self, request):
        self._validate()
        if (
            self.installed is None
            or _tensor_identity(self.actor.state_dict()) != self.installed.weights_sha256
        ):
            raise ValueError("Installed codec actor weights changed")
        if (
            request.identity != self.installed
            or request.request_index != self.calls
            or self.calls >= len(self.prompts)
            or not request.rollout_leader
            or request.dp_rank != 0
            or request.pp_rank != 0
        ):
            raise ValueError("Codec rollout identity/cursor/topology mismatch")
        if (
            request.group_size < 2
            or request.data_cursor != self.calls * request.group_size
            or request.group_size > request.max_samples
            or self.actor.max_frames * 16 + 1 > request.max_sequence_length
        ):
            raise ValueError("Codec rollout exceeds request action/sample bounds")
        prompt = self.prompts[self.calls]
        device = next(self.actor.parameters()).device
        if next(self.reference.parameters()).device != device:
            raise ValueError("Local codec actor/reference must use the same explicit device")
        values = {k: v.to(device) for k, v in prompt.payload.items()}
        generator = torch.Generator(device=device).manual_seed(request.seed)
        samples, masks, olds, refs, contexts = [], [], [], [], []
        for _ in range(request.group_size):
            actions, mask, old, lengths, eos = self.actor.sample(values, generator)
            context = dict(values, lengths=lengths, reached_eos=eos)
            recomputed = (
                self.actor(actions, **context)
                .float()
                .log_softmax(-1)
                .gather(-1, actions[..., None])
                .squeeze(-1)
            )
            if not torch.allclose(recomputed[mask], old[mask], rtol=1e-5, atol=1e-6):
                raise ValueError("Autoregressive/teacher-forced codec logprobs differ")
            ref = (
                self.reference(actions, **context)
                .float()
                .log_softmax(-1)
                .gather(-1, actions[..., None])
                .squeeze(-1)
            )
            samples.append(actions.cpu())
            masks.append(mask.cpu())
            olds.append(old.cpu())
            refs.append(ref.cpu())
            contexts.append(omni_rl._cpu_tree(context))
        self.calls += 1
        return omni_rl.RolloutBatch(
            request.identity,
            request.request_index,
            torch.cat(samples),
            torch.cat(samples),
            torch.cat(masks),
            torch.cat(olds),
            torch.cat(refs),
            (prompt.identity,) * request.group_size,
            tuple(
                f"{prompt.identity}:{request.request_index}:{i}" for i in range(request.group_size)
            ),
            tuple(contexts),
            None,
        )

    def state_dict(self):
        self._validate()
        return dict(schema=1, config=copy.deepcopy(self.config), calls=self.calls)

    def validate_state_dict(self, state):
        self._validate()
        if (
            set(state) != {"schema", "config", "calls"}
            or state["schema"] != 1
            or state["config"] != self.config
            or type(state["calls"]) is not int
            or not 0 <= state["calls"] <= len(self.prompts)
        ):
            raise ValueError("Codec rollout checkpoint config/cursor differs")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.calls, self.installed = state["calls"], None


class ExactCodecReward:
    """Exact supplied RVQ sequence/EOS reward, not perceptual speech quality."""

    def __init__(self, prompts, policy):
        self.prompts, self.policy = tuple(prompts), policy
        self.config = dict(
            kind="exact_codec_sequence_v1",
            prompts=[p.sha256 for p in prompts],
            policy=policy.semantics(),
            implementation=implementation_contract(type(self)),
        )
        self.calls = 0

    def _validate(self):
        for prompt in self.prompts:
            prompt.validate()
        if [p.sha256 for p in self.prompts] != self.config[
            "prompts"
        ] or self.policy.semantics() != self.config["policy"]:
            raise ValueError("Exact codec reward prompt/policy configuration changed")

    def score(self, request, batch):
        self._validate()
        if (
            self.calls >= len(self.prompts)
            or request.request_index != self.calls
            or batch.identity != request.identity
            or batch.request_index != self.calls
            or request.identity.contract.reward_config_sha256 != omni_rl.fingerprint(self.config)
        ):
            raise ValueError("Codec reward identity/config/cursor differs")
        prompt = self.prompts[self.calls]
        prompt.validate()
        target, mask = action_layout(
            prompt.target_codes[None],
            torch.tensor([len(prompt.target_codes)]),
            torch.tensor([prompt.target_eos]),
            max_frames=self.policy.max_frames,
            vocab=self.policy.vocab,
        )
        result = torch.tensor(
            [
                float(torch.equal(a, target[0]) and torch.equal(m, mask[0]))
                for a, m in zip(batch.sampled_token_ids, batch.token_mask, strict=True)
            ]
        )
        self.calls += 1
        return result

    def state_dict(self):
        self._validate()
        return dict(schema=1, config=copy.deepcopy(self.config), calls=self.calls)

    def validate_state_dict(self, state):
        self._validate()
        if (
            set(state) != {"schema", "config", "calls"}
            or state["schema"] != 1
            or state["config"] != self.config
            or type(state["calls"]) is not int
            or not 0 <= state["calls"] <= len(self.prompts)
        ):
            raise ValueError("Codec reward checkpoint config/cursor differs")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.calls = state["calls"]


def _validate_providers(cfg, backend, rollout, reward):
    if not isinstance(backend, CodecPolicyAdapter) or not isinstance(rollout, LocalCodecRollout):
        raise TypeError("Codec task requires the explicit local codec adapter/provider")
    if type(reward) is not ExactCodecReward:
        raise TypeError("Codec task requires ExactCodecReward with its checkpoint cursor contract")
    if cfg.micro_batch_size != 1:
        raise ValueError("Codec policy supports micro_batch_size=1")
    if cfg.max_requests > len(rollout.prompts):
        raise ValueError("The finite local prompt set must cover all admitted requests")
    rollout._validate()
    if backend.model.semantics() != rollout.actor.semantics():
        raise ValueError("Training and rollout codec policy semantics differ")
    owned = {p.data_ptr() for p in backend.model.parameters()}
    external = {
        p.data_ptr() for model in (rollout.actor, rollout.reference) for p in model.parameters()
    }
    if owned & external:
        raise ValueError(
            "Training, rollout actor and fixed reference require independent parameter storage"
        )


def create_trainer(cfg, *, contract, backend, rollout, reward):
    """Validate codec ownership, then construct the existing GRPO transaction engine."""
    _validate_providers(cfg, backend, rollout, reward)
    return omni_rl.OmniRLTrainer(
        cfg, contract=contract, backend=backend, rollout=rollout, reward=reward
    )


def launch(cfg, *, contract, backend, rollout, reward, trainer=None):
    """Run the existing GRPO trainer with an actual codec policy adapter/provider."""
    _validate_providers(cfg, backend, rollout, reward)
    return omni_rl.launch(
        cfg, contract=contract, backend=backend, rollout=rollout, reward=reward, trainer=trainer
    )
