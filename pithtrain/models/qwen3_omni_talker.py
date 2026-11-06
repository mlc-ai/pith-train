"""Native Qwen3-Omni Talker and residual-code predictor.

The parameter tree and eager math follow Transformers 5.17 (Apache-2.0).
This module is CPU importable and does not execute Transformers modeling code.
Speech alignment, target shifting, losses and sampling belong to the caller.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from pithtrain.models.qwen3_omni_talker_positions import TalkerPositionBuilder


class TalkerOutput(NamedTuple):
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


class CodePredictorOutput(NamedTuple):
    logits: torch.Tensor
    last_hidden_state: torch.Tensor
    generation_steps: int


class TeacherForcedCodeOutput(NamedTuple):
    logits: torch.Tensor
    codec_embeddings: torch.Tensor


def _sum_head_gradients(gradients):
    result = gradients[-1].clone()
    for index in range(len(gradients) - 2, -1, -1):
        result = result + gradients[index]
    return result


class _HeadInputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, heads):
        return hidden_states.unsqueeze(0).expand(heads, *hidden_states.shape)

    @staticmethod
    def backward(ctx, gradients):
        return _sum_head_gradients(gradients), None


def _headwise_backward(module, forward, hidden_states, *arguments):
    """One forward per FSDP unit, independent first-order codec-head VJPs.

    The leading axis carries derivatives, not repeated forward computation.
    Keep BF16 rounding at each head's derivative before the shared parameter
    sum, as in the independent causal-prefix graph. Parameters remain owned by
    their existing FSDP units and keep their ordinary checkpoint names.
    """
    names, parameters = zip(*module.named_parameters())

    class ForwardMethod(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = module

        def forward(self, value):
            return forward(value, *arguments)

    proxy = ForwardMethod()

    class HeadwiseVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, *weights):
            with torch.enable_grad():
                leaf = value[0].detach().requires_grad_()
                leaves = [p.detach().requires_grad_(p.requires_grad) for p in weights]
                replacements = {"inner." + name: p for name, p in zip(names, leaves)}
                output = torch.func.functional_call(
                    proxy, (replacements, dict(proxy.named_buffers())), (leaf,), strict=True
                )
            ctx.leaf, ctx.weights, ctx.output = leaf, leaves, output
            return output.unsqueeze(0).expand(value.shape[0], *output.shape)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, gradients):
            active = [p for p in ctx.weights if p.requires_grad]
            derivatives = torch.autograd.grad(
                ctx.output, (ctx.leaf, *active), gradients, is_grads_batched=True
            )
            iterator = iter(derivatives[1:])
            parameter_grads = [
                _sum_head_gradients(next(iterator)) if p.requires_grad else None
                for p in ctx.weights
            ]
            return derivatives[0], *parameter_grads

    return HeadwiseVJP.apply(hidden_states, *parameters)


class Qwen3OmniMoeTalkerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, value):
        dtype = value.dtype
        value = value.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(dtype)

    def _forward_per_head(self, hidden_states):
        return _headwise_backward(self, self.forward, hidden_states)


class Qwen3OmniMoeTalkerRotaryEmbedding(nn.Module):
    def __init__(self, config, *, multimodal=True):
        super().__init__()
        rope = config.rope_parameters
        if rope["rope_type"] != "default":
            raise NotImplementedError("Native Talker currently supports default RoPE only")
        self.base = rope["rope_theta"]
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.sections = tuple(rope.get("mrope_section", (24, 20, 20))) if multimodal else None
        if self.head_dim % 2:
            raise ValueError("Talker head_dim must be even")
        if multimodal and (len(self.sections) != 3 or sum(self.sections) != self.head_dim // 2):
            raise ValueError("Talker mrope_section must partition half of head_dim")

    @torch.no_grad()
    def forward(self, hidden_states, position_ids):
        inv = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2, device=hidden_states.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        # Recompute in FP32 after dtype/device moves; inverse frequencies must
        # never be rounded to the model's BF16 parameter dtype.
        freq = position_ids.float().unsqueeze(-1) * inv
        outputs = []
        for value in (freq.cos(), freq.sin()):
            if self.sections is not None:
                mixed = value[0].clone()
                for axis in (1, 2):
                    section = slice(axis, self.sections[axis] * 3, 3)
                    mixed[..., section] = value[axis, ..., section]
                value = mixed
            outputs.append(torch.cat((value, value), -1).to(hidden_states.dtype))
        return tuple(outputs)


def _rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _causal_mask(hidden_states, attention_mask, sliding_window=None):
    batch, length, _ = hidden_states.shape
    row = torch.arange(length, device=hidden_states.device)
    allowed = row[:, None] >= row[None, :]
    if sliding_window is not None:
        allowed = allowed & (row[:, None] - row[None, :] < sliding_window)
    allowed = allowed[None, None].expand(batch, 1, -1, -1)
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            if attention_mask.shape != (batch, 1, length, length):
                raise ValueError("Additive mask must be [batch, 1, length, length]")
            if not attention_mask.is_floating_point():
                raise ValueError("A 4D attention mask must be additive floating point")
            # A supplied 4D mask is an explicit caller-owned causal/packed mask.
            return attention_mask.to(hidden_states.dtype)
        if attention_mask.shape != (batch, length):
            raise ValueError("Padding mask must be [batch, length]")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("Padding mask must be binary")
        allowed = allowed & attention_mask[:, None, None, :].bool()
    return torch.where(allowed, 0.0, torch.finfo(hidden_states.dtype).min).to(hidden_states.dtype)


class Qwen3OmniMoeTalkerAttention(nn.Module):
    def __init__(self, config, layer_idx, *, predictor=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Attention heads must be divisible by KV heads")
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.sliding_window = None
        if predictor:
            layer_type = config.layer_types[layer_idx]
            if layer_type not in ("full_attention", "sliding_attention"):
                raise ValueError(f"Unsupported predictor layer type: {layer_type}")
            if layer_type == "sliding_attention":
                if not config.sliding_window or config.sliding_window < 1:
                    raise ValueError("Sliding attention requires a positive window")
                self.sliding_window = config.sliding_window
        elif config.sliding_window is not None:
            raise NotImplementedError("The released Talker trunk uses full attention")
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3OmniMoeTalkerRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3OmniMoeTalkerRMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = (part[:, None] for part in position_embeddings)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        scores = scores + _causal_mask(hidden_states, attention_mask, self.sliding_window)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
        output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        return self.o_proj(output.reshape(*hidden_states.shape[:-1], -1))


class Qwen3OmniMoeTalkerTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen3OmniMoeTalkerTextTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("Invalid Talker expert/top-k configuration")

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, self.weight.shape[1]), self.weight)
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.to(logits.dtype), indices


class Qwen3OmniMoeTalkerTextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            active = (mask.sum(dim=(-1, -2)) > 0).nonzero().flatten()
        for expert in active:
            lane, token = torch.where(mask[expert])
            gate, up = F.linear(hidden_states[token], self.gate_up_proj[expert]).chunk(2, dim=-1)
            contribution = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            contribution = contribution * top_k_weights[token, lane, None]
            output.index_add_(0, token, contribution.to(output.dtype))
        return output


class Qwen3OmniMoeTalkerTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen3OmniMoeTalkerTextTopKRouter(config)
        self.experts = Qwen3OmniMoeTalkerTextExperts(config)
        self.shared_expert = Qwen3OmniMoeTalkerTextMLP(
            config, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        shared = self.shared_expert(flat)
        _, weights, indices = self.gate(flat)
        routed = self.experts(flat, indices, weights)
        shared = self.shared_expert_gate(flat).sigmoid() * shared
        return (routed + shared).reshape(hidden_states.shape)


class Qwen3OmniMoeTalkerDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, *, predictor=False):
        super().__init__()
        self.self_attn = Qwen3OmniMoeTalkerAttention(config, layer_idx, predictor=predictor)
        self.mlp = (
            Qwen3OmniMoeTalkerTextMLP(config)
            if predictor
            else Qwen3OmniMoeTalkerTextSparseMoeBlock(config)
        )
        self.input_layernorm = Qwen3OmniMoeTalkerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3OmniMoeTalkerRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def _forward_per_head(self, hidden_states, position_embeddings):
        return _headwise_backward(self, self.forward, hidden_states, position_embeddings)


class Qwen3OmniMoeTalkerModel(nn.Module):
    def __init__(self, config, *, predictor=False):
        super().__init__()
        self.config = config
        self.predictor = predictor
        self.layers = nn.ModuleList(
            [
                Qwen3OmniMoeTalkerDecoderLayer(config, i, predictor=predictor)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3OmniMoeTalkerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3OmniMoeTalkerRotaryEmbedding(config, multimodal=not predictor)
        if predictor:
            self.codec_embedding = nn.ModuleList(
                [
                    nn.Embedding(config.vocab_size, config.hidden_size)
                    for _ in range(config.num_code_groups - 1)
                ]
            )
        else:
            if config.mlp_only_layers or config.decoder_sparse_step != 1:
                raise NotImplementedError("Talker uses an MoE plus shared expert in every layer")
            self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def get_input_embeddings(self):
        return self.codec_embedding

    def forward(
        self, input_ids=None, *, inputs_embeds=None, position_ids=None, attention_mask=None
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if input_ids is not None:
            if self.predictor:
                raise ValueError("Predictor model needs explicit codebook-selected inputs_embeds")
            inputs_embeds = self.codec_embedding(input_ids)
        batch, length, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=inputs_embeds.device)[None].expand(batch, -1)
        text_positions = (
            position_ids
            if position_ids.ndim == 2
            else (
                position_ids[0] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else None
            )
        )
        if not self.predictor and attention_mask is None and text_positions is not None:
            difference = torch.diff(text_positions, prepend=text_positions[:, :1] - 1, dim=-1)
            documents = (difference != 1).cumsum(-1)
            base_mask = _causal_mask(inputs_embeds, None)
            attention_mask = base_mask.masked_fill(
                documents[:, None, :, None] != documents[:, None, None, :],
                torch.finfo(inputs_embeds.dtype).min,
            )
        if not self.predictor:
            if position_ids.ndim == 2:
                position_ids = position_ids[None].expand(3, -1, -1)
            elif position_ids.shape[0] == 4:
                position_ids = position_ids[1:]
            if position_ids.shape != (3, batch, length):
                raise ValueError("Talker positions must be [batch,time] or [3/4,batch,time]")
        elif position_ids.shape != (batch, length):
            raise ValueError("Predictor positions must be [batch, codebook_sequence]")
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        return self.norm(hidden_states)

    def _forward_per_head(self, inputs_embeds, heads):
        if not self.predictor:
            raise ValueError("Per-head derivatives are only used by the codec predictor")
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        positions = positions[None].expand(inputs_embeds.shape[0], -1)
        embeddings = self.rotary_emb(inputs_embeds, positions)
        hidden_states = _HeadInputs.apply(inputs_embeds, heads)
        for layer in self.layers:
            hidden_states = layer._forward_per_head(hidden_states, embeddings)
        return self.norm._forward_per_head(hidden_states)


class Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.num_code_groups < 2:
            raise ValueError("Residual prediction needs at least two code groups")
        self.model = Qwen3OmniMoeTalkerModel(config, predictor=True)
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.num_code_groups - 1)
            ]
        )
        if config.tie_word_embeddings:
            raise NotImplementedError("Released residual codebook heads are untied")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, *, inputs_embeds, generation_steps, position_ids=None, attention_mask=None):
        if type(generation_steps) is not int or not 0 <= generation_steps < len(self.lm_head):
            raise ValueError("generation_steps selects one residual codebook head")
        hidden = self.model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, attention_mask=attention_mask
        )
        return CodePredictorOutput(
            self.lm_head[generation_steps](hidden), hidden, generation_steps + 1
        )

    def teacher_force(
        self, hidden_states, first_code_embeddings, residual_codes, *, residual_embeddings=None
    ):
        """Predict all residual groups without shifting the time axis.

        Inputs: hidden/first embedding [B,T,H], integer residual targets [B,T,G-1].
        Outputs: logits [B,T,G-1,Vr], embedding sum [B,T,H]. Residual target j
        becomes the input for j+1 only; the last embedding contributes to the
        next-time codec sum, preserving its genuine differentiable use. The
        speech recurrence passes its existing embeddings so both branches join
        before the BF16 embedding backward, rather than at FP32 master weights.
        """
        groups = len(self.lm_head)
        if hidden_states.ndim != 3 or first_code_embeddings.shape != hidden_states.shape:
            raise ValueError("Talker hidden and first-code embeddings must share [B,T,H]")
        if residual_codes.shape != (*hidden_states.shape[:2], groups):
            raise ValueError("Residual targets must be [B,T,num_code_groups-1]")
        if residual_codes.dtype not in (torch.int32, torch.int64):
            raise ValueError("Residual targets must be integer code IDs")
        if residual_codes.numel() and (
            residual_codes.min() < 0 or residual_codes.max() >= self.config.vocab_size
        ):
            raise ValueError("Residual target outside predictor vocabulary")
        embedded = residual_embeddings
        if embedded is None:
            embedded = [
                embedding(residual_codes[..., i])
                for i, embedding in enumerate(self.model.codec_embedding)
            ]
        elif len(embedded) != groups or any(
            value.shape != hidden_states.shape
            or value.dtype != first_code_embeddings.dtype
            or value.device != first_code_embeddings.device
            for value in embedded
        ):
            raise ValueError("Residual embeddings must share the first-code [B,T,H] layout")
        sequence = torch.stack([hidden_states, first_code_embeddings, *embedded[:-1]], dim=2)
        batch, time, length, width = sequence.shape
        flat = sequence.reshape(batch * time, length, width)
        # Preserve the existing FP32 activation graph, including autocast.
        # Separate head derivatives only when activations themselves are BF16.
        if torch.is_grad_enabled() and flat.dtype == torch.bfloat16:
            final = self.model._forward_per_head(flat, groups)
            logits = torch.stack(
                [head(final[i, :, i + 1]) for i, head in enumerate(self.lm_head)], dim=1
            )
        else:
            final = self.model(inputs_embeds=flat)
            logits = torch.stack(
                [head(final[:, i + 1]) for i, head in enumerate(self.lm_head)], dim=1
            )
        summed = torch.stack([first_code_embeddings, *embedded], dim=2).sum(dim=2)
        return TeacherForcedCodeOutput(logits.reshape(batch, time, groups, -1), summed)


class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.thinker_hidden_size, config.text_config.intermediate_size
        )
        self.linear_fc2 = nn.Linear(
            config.text_config.intermediate_size, config.text_config.hidden_size
        )
        self.act_fn = ACT2FN[config.text_config.hidden_act]

    def forward(self, hidden_states):
        # Payload slices may be strided. Keep the biased linear on the same
        # fused path as compact HF conditioning, without an extra BF16 round.
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states.contiguous())))


class Qwen3OmniMoeTalkerForConditionalGeneration(nn.Module):
    """Full Talker parameter tree; no implicit loss, shift, cache or sampling."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.num_code_groups != config.code_predictor_config.num_code_groups:
            raise ValueError("Talker and predictor code-group counts differ")
        if config.text_config.hidden_size != config.code_predictor_config.hidden_size:
            raise ValueError("Talker and predictor hidden widths must agree")
        self.model = Qwen3OmniMoeTalkerModel(config.text_config)
        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(config)
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(config)
        self.codec_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(
            config.code_predictor_config
        )
        self.apply(self._init_weights)
        self.code_predictor.apply(self.code_predictor._init_weights)
        for module in self.modules():
            if isinstance(module, Qwen3OmniMoeTalkerTextExperts):
                nn.init.normal_(module.gate_up_proj, std=config.text_config.initializer_range)
                nn.init.normal_(module.down_proj, std=config.text_config.initializer_range)
            elif isinstance(module, Qwen3OmniMoeTalkerTextTopKRouter):
                nn.init.normal_(module.weight, std=config.text_config.initializer_range)
        if config.text_config.tie_word_embeddings:
            self.codec_head.weight = self.model.codec_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.text_config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.model.codec_embedding

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        use_audio_in_video=False,
        audio_seqlens=None,
        second_per_grids=None,
        *,
        audio_n_window=50,
    ):
        return TalkerPositionBuilder(self.config, audio_n_window=audio_n_window).get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            use_audio_in_video,
            audio_seqlens,
            second_per_grids,
        )

    def forward(
        self, input_ids=None, *, inputs_embeds=None, position_ids=None, attention_mask=None
    ):
        hidden = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return TalkerOutput(self.codec_head(hidden), hidden)
