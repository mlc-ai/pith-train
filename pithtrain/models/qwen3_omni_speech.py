# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license terms.
"""Native Qwen3-Omni speech conditioning, teacher forcing and bounded decoding.

The conditioning and codec recurrence follow the Transformers 5.17 Qwen3-Omni
implementation. Codec labels must come from an explicitly identified external
encoder or prepared dataset; discrete sampling is not a differentiable bridge.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ThinkerSpeechOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class TalkerCondition:
    inputs_embeds: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    trailing_text_hidden: torch.Tensor
    tts_pad_embed: torch.Tensor


@dataclass
class SpeechTrainingOutput:
    first_logits: torch.Tensor
    residual_logits: torch.Tensor
    first_labels: torch.Tensor
    residual_labels: torch.Tensor
    thinker_logits: torch.Tensor | None = None


@dataclass
class SpeechGenerationOutput:
    codes: torch.Tensor
    code_lengths: torch.Tensor
    waveforms: list[torch.Tensor]
    reached_eos: torch.Tensor


@dataclass
class OmniGenerationOutput:
    sequences: torch.Tensor
    text_reached_eos: bool
    speech: SpeechGenerationOutput | None


def _require_unsharded_generation(*roots):
    # EOS and per-row chunk lengths change call counts. The training FSDP plan
    # does not provide a synchronized generation protocol for those branches.
    from torch.distributed.fsdp import FSDPModule

    if any(isinstance(module, FSDPModule) for root in roots for module in root.modules()):
        raise ValueError("Speech generation requires unsharded modules; FSDP is training-only")


def _sample(logits, *, do_sample, temperature, top_k, top_p, generator):
    if not torch.isfinite(logits).any(dim=-1).all() or torch.isnan(logits).any():
        raise ValueError("Every generation row must have finite candidate logits")
    if not do_sample:
        return logits.argmax(-1)
    if (
        not math.isfinite(temperature)
        or temperature <= 0
        or not 0 < top_p <= 1
        or type(top_k) is not int
        or top_k < 0
    ):
        raise ValueError("Invalid sampling temperature, top_k or top_p")
    logits = logits.float() / temperature
    if top_k:
        cutoff = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    if top_p < 1:
        sorted_logits, indices = logits.sort(dim=-1, descending=True)
        reject = sorted_logits.softmax(-1).cumsum(-1) > top_p
        reject[:, 1:] = reject[:, :-1].clone()
        reject[:, 0] = False
        logits = logits.scatter(1, indices, sorted_logits.masked_fill(reject, -torch.inf))
    return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate_speech_from_condition(
    talker,
    code2wav,
    condition,
    *,
    max_frames=1024,
    do_sample=False,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.05,
    generator=None,
    chunk_size=300,
    left_context_size=25,
):
    """Bounded native codec generation followed by causal waveform decoding.

    This implementation recomputes the prefix instead of maintaining a KV
    cache. Greedy mode is greedy for *every* codebook. EOS is never passed to the
    codec decoder; individual finished batch rows produce no additional frames.
    ``reached_eos=False`` explicitly identifies truncation at the supplied cap.
    """
    _require_unsharded_generation(talker, code2wav)
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("max_frames must be a positive finite integer")
    if not torch.isfinite(torch.tensor(repetition_penalty)) or repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be finite and positive")
    modes = [(module, module.training) for root in (talker, code2wav) for module in root.modules()]
    talker.eval()
    code2wav.eval()
    try:
        batch = condition.inputs_embeds.shape[0]
        device = condition.inputs_embeds.device
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch, dtype=torch.long, device=device)
        group_count = talker.config.num_code_groups
        vocab = talker.config.code_predictor_config.vocab_size
        eos_id = talker.config.codec_eos_token_id
        history, frame_masks = [], []
        embeds, attention = condition.inputs_embeds, condition.attention_mask
        sample_options = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        for step in range(max_frames):
            output = talker(
                inputs_embeds=embeds,
                attention_mask=attention,
                position_ids=_append_positions(condition, step),
            )
            logits = output.logits[:, -1].float().clone()
            if not 0 <= eos_id < logits.shape[-1]:
                raise ValueError("Codec EOS is outside the first-head vocabulary")
            allowed = torch.arange(logits.shape[-1], device=device) < vocab
            allowed[eos_id] = True
            logits.masked_fill_(~allowed, -torch.inf)
            if history and repetition_penalty != 1:
                previous = torch.stack(history, dim=1)[..., 0]
                previous_valid = torch.stack(frame_masks, dim=1)
                for row in range(batch):
                    seen = previous[row, previous_valid[row]].unique()
                    values = logits[row, seen]
                    logits[row, seen] = torch.where(
                        values < 0, values * repetition_penalty, values / repetition_penalty
                    )
            first = _sample(logits, **sample_options)
            finished = finished | (first == eos_id)
            live = ~finished
            if not live.any():
                break
            first = first.masked_fill(~live, 0)
            first_embed = talker.get_input_embeddings()(first)
            predictor_inputs = torch.stack((output.last_hidden_state[:, -1], first_embed), 1)
            groups = [first]
            embeddings = talker.code_predictor.get_input_embeddings()
            for index in range(group_count - 1):
                prediction = talker.code_predictor(
                    inputs_embeds=predictor_inputs, generation_steps=index
                )
                # HF residual sampling has its own fixed top-p/top-k policy.
                residual = _sample(
                    prediction.logits[:, -1],
                    do_sample=do_sample,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.8,
                    generator=generator,
                ).masked_fill(~live, 0)
                groups.append(residual)
                predictor_inputs = torch.cat(
                    (predictor_inputs, embeddings[index](residual).unsqueeze(1)), 1
                )
            codes = torch.stack(groups, dim=-1)
            history.append(codes)
            frame_masks.append(live)
            lengths += live
            next_embed = codec_embeddings(talker, codes[:, None])
            tail = (
                condition.trailing_text_hidden[:, step : step + 1]
                if step < condition.trailing_text_hidden.shape[1]
                else condition.tts_pad_embed
            )
            embeds = torch.cat((embeds, next_embed + tail), dim=1)
            attention = torch.cat((attention, live[:, None].to(attention.dtype)), dim=1)
        codes = (
            torch.stack(history, dim=1)
            if history
            else torch.empty(batch, 0, group_count, dtype=torch.long, device=device)
        )
        waves = code2wav.decode_batch(
            codes.transpose(1, 2),
            lengths,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
        )
        return SpeechGenerationOutput(codes, lengths, [w.float() for w in waves], finished)
    finally:
        for module, was_training in modes:
            module.training = was_training


def thinker_speech_forward(thinker, input_ids, model_context, accept_hidden_layer):
    """Expose native Thinker embeddings and the selected layer without hooks.

    This composition requires a complete native module (phase=-1, CP=EP=1).
    A partitioned pipeline must transport these activations explicitly instead
    of silently reading a layer that is absent from this rank.
    """
    if thinker.stage_count != 1 or thinker.stage_index != 0:
        raise ValueError("Speech feature extraction requires the complete Thinker module")
    layers = list(thinker.layers.values())
    if not 0 <= accept_hidden_layer <= len(layers):
        raise ValueError("accept_hidden_layer is outside the Thinker hidden-state tuple")
    if [layer.idx for layer in layers] != list(range(len(layers))):
        raise ValueError("Thinker layers must be complete and ordered")
    for layer in layers:
        mlp = layer.mlp
        if hasattr(mlp, "experts_per_rank") and mlp.experts_per_rank != mlp.num_experts:
            raise ValueError("Speech feature extraction cannot bypass expert-parallel dispatch")
    width = getattr(thinker, "text_hidden_size", thinker.hidden_size)
    payload = thinker.forward_prolog(input_ids, model_context)
    embeddings = payload[..., :width]
    current_hidden = embeddings
    context = model_context or {}
    has_vision = any(context.get(key) is not None for key in ("image_grid_thw", "video_grid_thw"))
    hidden = embeddings if accept_hidden_layer == 0 else None
    rotary = thinker.forward_posemb(input_ids.shape[1], model_context=model_context)
    for index, layer in enumerate(layers, 1):
        if hasattr(layer, "reference_forward_with_hidden"):
            payload, layer_hidden = layer.reference_forward_with_hidden(
                payload, rotary, hidden_states=current_hidden
            )
            # A real DeepStack injection creates a new tensor in HF too. For
            # absent/exhausted vision lanes keep the alias, including the
            # selected hidden state shared with the speech projection.
            injected = has_vision and layer.idx < layer.deepstack_count
            current_hidden = payload[..., :width] if injected else layer_hidden
        else:
            payload = layer.reference_forward(current_hidden, rotary)
            layer_hidden = current_hidden = payload[..., :width]
        if index == accept_hidden_layer:
            hidden = layer_hidden
    # HF's last recorded hidden state is after the final norm.
    if accept_hidden_layer == len(layers):
        hidden = thinker.norm(current_hidden)
    return ThinkerSpeechOutput(thinker.forward_epilog(current_hidden), embeddings, hidden)


def prepare_talker_condition(
    talker,
    thinker_embeddings,
    thinker_hidden,
    input_ids,
    *,
    config,
    text_embedding,
    prompt_lengths,
    speaker="ethan",
    attention_mask=None,
    position_builder=None,
    position_kwargs=None,
):
    """Build the official user + nine-token assistant speech prefix.

    ``input_ids`` and the two feature tensors describe exactly the same tokens:
    the completed assistant response excludes its terminal text EOS. The final
    assistant header is three tokens (im_start, assistant, newline) and must be
    followed by at least one response token. ``prompt_lengths`` locates the end
    of each original prompt, so earlier assistant turns cannot be mistaken for
    the response to speak. Rows may have different left/right padding.
    """
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be a batch of integer token ids")
    if thinker_embeddings.shape[:2] != input_ids.shape:
        raise ValueError("Thinker embeddings and tokens must align")
    if thinker_hidden.shape != thinker_embeddings.shape:
        raise ValueError("Thinker selected hidden states and embeddings must align")
    batch, length = input_ids.shape
    mask = (
        torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask
    )
    if mask.shape != input_ids.shape or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("attention_mask must be a binary token mask")
    mask = mask.bool()
    if len(prompt_lengths) != batch:
        raise ValueError("One prompt length is required per row")
    speaker_id = config.talker_config.speaker_id.get(speaker.lower())
    if speaker_id is None:
        raise ValueError(f"Unknown Qwen3-Omni speaker: {speaker}")
    specials = input_ids.new_tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]]
    )
    bos, eos, pad = talker.text_projection(text_embedding(specials)).chunk(3, dim=1)
    codec_ids = input_ids.new_tensor(
        [
            [
                config.talker_config.codec_nothink_id,
                config.talker_config.codec_think_bos_id,
                config.talker_config.codec_think_eos_id,
                speaker_id,
                config.talker_config.codec_pad_id,
                config.talker_config.codec_bos_id,
            ]
        ]
    )
    codec_prefix = talker.get_input_embeddings()(codec_ids)
    media_config = config.thinker_config
    mm_mask = (
        (input_ids == media_config.image_token_id)
        | (input_ids == media_config.audio_token_id)
        | (input_ids == media_config.video_token_id)
    )
    # Both projections stay in the graph, including modality-absent batches.
    projected = torch.where(
        mm_mask.unsqueeze(-1),
        talker.hidden_projection(thinker_hidden),
        talker.text_projection(thinker_embeddings),
    )
    prefixes, prefix_ids, trailing = [], [], []
    for row in range(batch):
        end = int(prompt_lengths[row])
        if not 0 < end <= length:
            raise ValueError("Prompt length is outside the input")
        ids = input_ids[row]
        starts = torch.nonzero(
            (ids[:-1] == config.im_start_token_id)
            & (ids[1:] == config.assistant_token_id)
            & mask[row, :-1]
            & mask[row, 1:],
            as_tuple=False,
        ).flatten()
        starts = starts[starts < end]
        if starts.numel() == 0:
            raise ValueError("The prompt must contain an assistant generation header")
        start = int(starts[-1])
        if end != start + 3 or not mask[row, start:end].all():
            raise ValueError("The prompt must end after the complete assistant header")
        response_mask = mask[row, end:]
        if not response_mask.any() or response_mask.tolist() != sorted(
            response_mask.tolist(), reverse=True
        ):
            raise ValueError("Assistant response must be nonempty and contiguous")
        response_end = end + int(response_mask.sum())
        user_indices, role = [], None
        for index in range(start):
            if not mask[row, index]:
                continue
            if ids[index] == config.im_start_token_id and index + 1 < start:
                role = int(ids[index + 1])
            if role == config.user_token_id:
                user_indices.append(index)
        if not user_indices:
            raise ValueError("Speech conditioning requires at least one user turn")
        user_idx = ids.new_tensor(user_indices)
        user = projected[row : row + 1, user_idx]
        assistant = talker.text_projection(thinker_embeddings[row : row + 1, start:response_end])
        text_prefix = torch.cat(
            (assistant[:, :3], pad.expand(1, 4, -1), bos, assistant[:, 3:4]), dim=1
        )
        codec = torch.cat((codec_prefix.new_zeros(1, 3, codec_prefix.shape[-1]), codec_prefix), 1)
        prefixes.append(torch.cat((user, text_prefix + codec), dim=1))
        prefix_ids.append(torch.cat((ids[user_idx], ids.new_full((9,), config.tts_pad_token_id))))
        trailing.append(torch.cat((assistant[:, 4:], eos), dim=1))
    prefix_length = max(item.shape[1] for item in prefixes)
    trailing_length = max(item.shape[1] for item in trailing)
    hidden_width = prefixes[0].shape[-1]
    # Left padding preserves the final common next-code prediction position.
    padded_embeds, padded_ids, padded_masks, padded_trailing = [], [], [], []
    for prefix, ids, tail in zip(prefixes, prefix_ids, trailing):
        missing = prefix_length - prefix.shape[1]
        padded_embeds.append(torch.cat((prefix.new_zeros(1, missing, hidden_width), prefix), dim=1))
        padded_ids.append(torch.cat((ids.new_full((missing,), config.tts_pad_token_id), ids)))
        padded_masks.append(torch.cat((ids.new_zeros(missing), ids.new_ones(ids.shape[0]))))
        padded_trailing.append(
            torch.cat((tail, pad.expand(1, trailing_length - tail.shape[1], -1)), dim=1)
        )
    embeds = torch.cat(padded_embeds)
    ids = torch.stack(padded_ids)
    active = torch.stack(padded_masks)
    builder = talker.get_rope_index if position_builder is None else position_builder
    positions, _ = builder(input_ids=ids, attention_mask=active, **(position_kwargs or {}))
    return TalkerCondition(
        embeds, ids, active, positions, torch.cat(padded_trailing), pad.expand(batch, -1, -1)
    )


def _codec_embedding_values(talker, codes):
    """Keep shared BF16 outputs for the next-frame and residual-head branches."""
    embeddings = talker.code_predictor.get_input_embeddings()
    if codes.shape[-1] != len(embeddings) + 1:
        raise ValueError("Codec group count differs from the residual predictor")
    values = [talker.get_input_embeddings()(codes[..., 0])]
    values.extend(embedding(codes[..., index + 1]) for index, embedding in enumerate(embeddings))
    return values


def codec_embeddings(talker, codes):
    """Sum the first and all residual codebook embeddings, as the Talker does."""
    values = _codec_embedding_values(talker, codes)
    # Match HF's single reduction over all codebooks. Sequential BF16 additions
    # round after every group and change the next frame's autoregressive input.
    return torch.stack(values, dim=-2).sum(dim=-2)


def _append_positions(condition, count):
    if count == 0:
        return condition.position_ids
    if condition.position_ids.shape[0] not in (3, 4):
        raise ValueError("Talker positions need T/H/W axes, with optional text axis")
    # Each sample continues after its own maximum multimodal position.
    axes = condition.position_ids
    valid = condition.attention_mask.bool().unsqueeze(0)
    maxima = axes.masked_fill(~valid, -1).amax(dim=(0, 2))
    extra = maxima[:, None] + torch.arange(1, count + 1, dtype=axes.dtype, device=axes.device)
    extra = extra.unsqueeze(0).expand(axes.shape[0], -1, -1)
    return torch.cat((axes, extra), dim=-1)


def _trailing(condition, count):
    tail = condition.trailing_text_hidden[:, :count]
    if tail.shape[1] < count:
        tail = torch.cat(
            (tail, condition.tts_pad_embed.expand(-1, count - tail.shape[1], -1)), dim=1
        )
    return tail


def speech_teacher_force(talker, condition, codes, code_lengths, *, codebook_mask=None):
    """Predict codec frames and EOS without leaking the target first code.

    ``codes`` is [B,T,G], without EOS, containing externally prepared RVQ labels.
    A position sees only earlier frames; each residual head additionally sees
    the current first code and earlier residual codes. The terminal EOS has a
    first-head target only. Padding and unavailable supervision become -100.
    """
    if codes.ndim != 3 or codes.dtype not in (torch.int32, torch.int64):
        raise ValueError("codes must be integer [batch,time,codebook] labels")
    batch, frames, groups = codes.shape
    if batch != condition.inputs_embeds.shape[0] or groups != talker.config.num_code_groups:
        raise ValueError("Codec batch/groups differ from the Talker")
    if code_lengths.shape != (batch,) or code_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("code_lengths must contain one integer per sample")
    if (code_lengths < 0).any() or (code_lengths > frames).any():
        raise ValueError("Codec lengths are outside the padded frame range")
    valid = torch.arange(frames, device=codes.device)[None] < code_lengths[:, None]
    if codebook_mask is not None:
        if codebook_mask.shape != codes.shape or codebook_mask.dtype != torch.bool:
            raise ValueError("codebook_mask must be boolean and match codes")
        if (codebook_mask & ~valid.unsqueeze(-1)).any():
            raise ValueError("Codec supervision extends beyond a sample's valid frames")
    codec_vocab = talker.config.code_predictor_config.vocab_size
    if not 0 <= talker.config.codec_eos_token_id < talker.config.text_config.vocab_size:
        raise ValueError("Codec EOS is outside the first-head vocabulary")
    # Every input frame needs all groups even when only some heads are supervised.
    if ((codes[valid] < 0) | (codes[valid] >= codec_vocab)).any():
        raise ValueError("Active codec ids are outside the RVQ codebook")
    safe_codes = codes.masked_fill(~valid.unsqueeze(-1), 0)
    embedded = _codec_embedding_values(talker, safe_codes)
    previous = torch.stack(embedded, dim=-2).sum(dim=-2) + _trailing(condition, frames)
    inputs = torch.cat((condition.inputs_embeds, previous), dim=1)
    # EOS positions are active; subsequent padded inputs are never used as targets.
    attention = torch.cat((condition.attention_mask, valid.to(torch.long)), dim=1)
    result = talker(
        inputs_embeds=inputs,
        attention_mask=attention,
        position_ids=_append_positions(condition, frames),
    )
    start = condition.inputs_embeds.shape[1] - 1
    first_logits = result.logits[:, start:]
    first_labels = codes.new_full((batch, frames + 1), -100)
    first_labels[:, :frames] = safe_codes[..., 0].masked_fill(~valid, -100)
    if codebook_mask is not None:
        first_labels[:, :frames].masked_fill_(~codebook_mask[..., 0], -100)
    first_labels.scatter_(1, code_lengths[:, None], talker.config.codec_eos_token_id)
    residual_labels = safe_codes[..., 1:].masked_fill(~valid.unsqueeze(-1), -100)
    if codebook_mask is not None:
        residual_labels = residual_labels.masked_fill(~codebook_mask[..., 1:], -100)
    if frames:
        residual = talker.code_predictor.teacher_force(
            result.last_hidden_state[:, start : start + frames],
            embedded[0],
            safe_codes[..., 1:],
            residual_embeddings=embedded[1:],
        ).logits
    else:
        # Keep the shape contract for all-EOS batches without calling a zero-batch transformer.
        residual = first_logits.new_empty(batch, 0, groups - 1, codec_vocab)
    return SpeechTrainingOutput(first_logits, residual, first_labels, residual_labels)


class Qwen3OmniMoeSpeechModel(nn.Module):
    """Named native components with explicit text/codec/waveform training paths."""

    def __init__(self, config, thinker, talker, code2wav):
        super().__init__()
        if config.model_type != "qwen3_omni_moe":
            raise ValueError("This task supports Qwen3-Omni only")
        self.config = config
        self.thinker, self.talker, self.code2wav = thinker, talker, code2wav

    def forward(
        self,
        input_ids,
        model_context,
        codes,
        code_lengths,
        *,
        prompt_lengths,
        speaker="ethan",
        codebook_mask=None,
        position_kwargs=None,
    ):
        features = thinker_speech_forward(
            self.thinker, input_ids, model_context, self.config.talker_config.accept_hidden_layer
        )
        condition = prepare_talker_condition(
            self.talker,
            features.embeddings,
            features.hidden_states,
            input_ids,
            config=self.config,
            text_embedding=self.thinker.embed_tokens,
            prompt_lengths=prompt_lengths,
            speaker=speaker,
            attention_mask=model_context.get("attention_mask"),
            position_kwargs=position_kwargs,
        )
        outputs = speech_teacher_force(
            self.talker, condition, codes, code_lengths, codebook_mask=codebook_mask
        )
        outputs.thinker_logits = features.logits
        return outputs

    def decode_training_codes(self, codes):
        """Differentiable codec-to-waveform path; callers supply their waveform objective."""
        return self.code2wav(codes)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        model_context,
        *,
        max_text_tokens=128,
        max_codec_frames=1024,
        text_eos_token_id=None,
        return_audio=True,
        speaker="ethan",
        do_sample=False,
        generator=None,
        position_kwargs=None,
    ):
        """Generate native Thinker text and optional speech with finite token caps.

        The current complete native media path accepts one unpadded sample.
        This is an eager reference integration, not a KV-cache or DualPipeV
        speech-generation claim. Component methods remain independently usable.
        """
        _require_unsharded_generation(self)
        if type(max_text_tokens) is not int or max_text_tokens < 1:
            raise ValueError("max_text_tokens must be a positive finite integer")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Native complete Thinker generation requires one nonempty row")
        mask = model_context.get("attention_mask")
        if mask is not None and (mask.shape != input_ids.shape or not (mask == 1).all()):
            raise ValueError("Native complete Thinker generation requires an unpadded prompt")
        # The released Thinker text config leaves eos_token_id unset.
        # Its composite generation contract uses the ChatML end-of-turn token.
        eos = self.config.im_end_token_id if text_eos_token_id is None else text_eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else eos
        if not eos_ids or any(type(token) is not int or token < 0 for token in eos_ids):
            raise ValueError("Explicit valid Thinker EOS token ids are required")
        assistant_header_end = None
        if return_audio:
            starts = torch.nonzero(
                (input_ids[0, :-1] == self.config.im_start_token_id)
                & (input_ids[0, 1:] == self.config.assistant_token_id),
                as_tuple=False,
            ).flatten()
            if not starts.numel():
                raise ValueError("Speech generation requires an assistant header")
            assistant_header_end = int(starts[-1]) + 3
            if assistant_header_end > input_ids.shape[1]:
                raise ValueError("Speech generation requires a complete assistant header")
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            sequence, prompt_length, reached_eos = input_ids.clone(), input_ids.shape[1], False
            context = dict(model_context)
            context.pop("position_ids", None)  # Recompute positions for the growing token sequence.
            for _ in range(max_text_tokens):
                context.update(input_ids=sequence, attention_mask=torch.ones_like(sequence))
                features = thinker_speech_forward(
                    self.thinker,
                    sequence,
                    context,
                    self.config.talker_config.accept_hidden_layer,
                )
                token = _sample(
                    features.logits[:, -1],
                    do_sample=do_sample,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    generator=generator,
                )
                if int(token[0]) in eos_ids:
                    reached_eos = True
                    break
                sequence = torch.cat((sequence, token[:, None]), dim=1)
            speech = None
            # HF composite generation conditions on actual decode forwards:
            # the last sampled token has no hidden state until a later forward.
            # Reuse the final decode features instead of forwarding that token
            # once more and silently extending Talker's trailing text sequence.
            condition_length = features.embeddings.shape[1]
            if (
                return_audio
                and sequence.shape[1] > prompt_length
                and condition_length > assistant_header_end
            ):
                conditioning_ids = sequence[:, :condition_length]
                condition = prepare_talker_condition(
                    self.talker,
                    features.embeddings,
                    features.hidden_states,
                    conditioning_ids,
                    config=self.config,
                    text_embedding=self.thinker.embed_tokens,
                    prompt_lengths=[assistant_header_end],
                    speaker=speaker,
                    position_kwargs=position_kwargs,
                )
                speech = generate_speech_from_condition(
                    self.talker,
                    self.code2wav,
                    condition,
                    max_frames=max_codec_frames,
                    do_sample=do_sample,
                    generator=generator,
                )
            if reached_eos:
                sequence = torch.cat((sequence, token[:, None]), dim=1)
            return OmniGenerationOutput(sequence, reached_eos, speech)
        finally:
            for module, was_training in modes:
                module.training = was_training
