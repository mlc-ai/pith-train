"""
Static memory profiling: model parameters, FSDP sharding, and optimizer states.

Replicates the layer distribution logic from Qwen3MoeModel and the FSDP sharding
topology from apply_fsdp, without importing any PithTrain code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .tensor_spec import MemoryBucket, TensorSpec


@dataclass(slots=True)
class ModelConfig:
    """Parsed model configuration (from HuggingFace config JSON)."""

    model_type: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    decoder_sparse_step: int
    mlp_only_layers: list[int]

    @staticmethod
    def from_json(path: str | Path) -> "ModelConfig":
        with open(path) as f:
            cfg = json.load(f)
        hidden_size = cfg["hidden_size"]
        num_attention_heads = cfg["num_attention_heads"]
        return ModelConfig(
            model_type=cfg.get("model_type", "qwen3_moe"),
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=cfg.get("num_key_value_heads", num_attention_heads),
            head_dim=cfg.get("head_dim", hidden_size // num_attention_heads),
            intermediate_size=cfg.get("intermediate_size", 4 * hidden_size),
            num_experts=cfg.get("num_experts", cfg.get("n_routed_experts", 0)),
            num_experts_per_tok=cfg.get("num_experts_per_tok", 0),
            moe_intermediate_size=cfg.get("moe_intermediate_size", 0),
            num_hidden_layers=cfg["num_hidden_layers"],
            vocab_size=cfg["vocab_size"],
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            decoder_sparse_step=cfg.get("decoder_sparse_step", 1),
            mlp_only_layers=cfg.get("mlp_only_layers", []),
        )


@dataclass(slots=True)
class ParallelismConfig:
    """Parallelism and training configuration."""

    pp_size: int
    ep_size: int
    dp_size: int
    cp_size: int
    micro_batch_size: int
    global_batch_size: int
    sequence_length: int
    fp8_training: str = "disabled"


def compute_layer_distribution(num_hidden_layers: int, num_stages: int) -> list[int]:
    """
    Distribute decoder layers across pipeline stages.

    Matches the algorithm in pithtrain/dualpipe/dualpipev.py:
    even base allocation, remainder layers go to inner stages first
    (edges stay light to compensate for embed/head overhead).
    """
    assert num_hidden_layers >= num_stages >= 1
    base, remainder = divmod(num_hidden_layers, num_stages)
    layers = [base] * num_stages
    for _ in range(remainder):
        min_val = min(layers)
        best = None
        for i in range(1, num_stages - 1):
            if layers[i] == min_val:
                best = i
                break
        if best is None:
            for i in range(num_stages):
                if layers[i] == min_val:
                    best = i
                    break
        layers[best] += 1
    return layers


def layer_is_moe(
    layer_idx: int,
    num_experts: int,
    decoder_sparse_step: int,
    mlp_only_layers: list[int],
) -> bool:
    """Whether a given layer uses MoE (vs dense MLP)."""
    if num_experts <= 0:
        return False
    if (layer_idx + 1) % decoder_sparse_step != 0:
        return False
    if layer_idx in mlp_only_layers:
        return False
    return True


class ModelMemoryProfile:
    """Computes static parameter and optimizer memory for a given rank."""

    def __init__(self, model_cfg: ModelConfig, parallel_cfg: ParallelismConfig, pp_rank: int):
        self.model_cfg = model_cfg
        self.parallel_cfg = parallel_cfg
        self.pp_rank = pp_rank

        pp_size = parallel_cfg.pp_size
        num_stages = pp_size * 2
        self.num_stages = num_stages

        # Layer distribution
        self.layer_dist = compute_layer_distribution(model_cfg.num_hidden_layers, num_stages)

        # Module assignments (V-shape)
        self.module0_stage = pp_rank
        self.module1_stage = num_stages - 1 - pp_rank

        # Layer ranges for each module
        self.module0_layers = self._layer_range(self.module0_stage)
        self.module1_layers = self._layer_range(self.module1_stage)

        # Derived
        self.experts_per_rank = (
            model_cfg.num_experts // parallel_cfg.ep_size if model_cfg.num_experts > 0 else 0
        )
        self.is_first_pp_rank = pp_rank == 0
        self.is_last_pp_rank = pp_rank == pp_size - 1

        # module[0] has embed_tokens if stage_id == 0
        # module[1] has embed_tokens if its stage_id == 0
        # module[0] has norm/lm_head if stage_id == num_stages - 1
        # module[1] has norm/lm_head if its stage_id == num_stages - 1
        self.module0_has_embed = self.module0_stage == 0
        self.module1_has_embed = self.module1_stage == 0
        self.module0_has_head = self.module0_stage == num_stages - 1
        self.module1_has_head = self.module1_stage == num_stages - 1

    def _layer_range(self, stage_id: int) -> list[int]:
        """Return global layer indices for a given stage."""
        start = sum(self.layer_dist[:stage_id])
        count = self.layer_dist[stage_id]
        return list(range(start, start + count))

    def _layer_params(self, layer_idx: int, prefix: str) -> list[TensorSpec]:
        """Enumerate all parameters for one decoder layer."""
        cfg = self.model_cfg
        H = cfg.hidden_size
        nheads = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        hd = cfg.head_dim

        specs: list[TensorSpec] = []

        # Attention parameters
        specs.append(TensorSpec(f"{prefix}.self_attn.q_proj.weight", (nheads * hd, H), "bf16"))
        specs.append(TensorSpec(f"{prefix}.self_attn.k_proj.weight", (nkv * hd, H), "bf16"))
        specs.append(TensorSpec(f"{prefix}.self_attn.v_proj.weight", (nkv * hd, H), "bf16"))
        specs.append(TensorSpec(f"{prefix}.self_attn.o_proj.weight", (H, nheads * hd), "bf16"))
        specs.append(TensorSpec(f"{prefix}.self_attn.q_norm.weight", (hd,), "bf16"))
        specs.append(TensorSpec(f"{prefix}.self_attn.k_norm.weight", (hd,), "bf16"))

        # Layer norms
        specs.append(TensorSpec(f"{prefix}.input_layernorm.weight", (H,), "bf16"))
        specs.append(TensorSpec(f"{prefix}.post_attention_layernorm.weight", (H,), "bf16"))

        # MLP / MoE
        is_moe = layer_is_moe(
            layer_idx,
            cfg.num_experts,
            cfg.decoder_sparse_step,
            cfg.mlp_only_layers,
        )
        if is_moe:
            E_local = self.experts_per_rank
            moe_inter = cfg.moe_intermediate_size
            # Gate
            specs.append(TensorSpec(f"{prefix}.mlp.gate.weight", (cfg.num_experts, H), "bf16"))
            # Experts (stacked per rank)
            specs.append(
                TensorSpec(
                    f"{prefix}.mlp.experts.gate_proj.weight",
                    (E_local, moe_inter, H),
                    "bf16",
                )
            )
            specs.append(
                TensorSpec(
                    f"{prefix}.mlp.experts.up_proj.weight",
                    (E_local, moe_inter, H),
                    "bf16",
                )
            )
            specs.append(
                TensorSpec(
                    f"{prefix}.mlp.experts.down_proj.weight",
                    (E_local, H, moe_inter),
                    "bf16",
                )
            )
        else:
            inter = cfg.intermediate_size
            specs.append(TensorSpec(f"{prefix}.mlp.gate_proj.weight", (inter, H), "bf16"))
            specs.append(TensorSpec(f"{prefix}.mlp.up_proj.weight", (inter, H), "bf16"))
            specs.append(TensorSpec(f"{prefix}.mlp.down_proj.weight", (H, inter), "bf16"))

        return specs

    def _is_expert_param(self, spec: TensorSpec) -> bool:
        return ".mlp.experts." in spec.name

    def _fsdp_shard_world(self, spec: TensorSpec) -> int:
        """FSDP shard world size for a parameter."""
        pcfg = self.parallel_cfg
        if self._is_expert_param(spec):
            return pcfg.dp_size * pcfg.cp_size
        else:
            return pcfg.dp_size * pcfg.cp_size * pcfg.ep_size

    def _fsdp_shard_size(self, spec: TensorSpec) -> int:
        """Number of FSDP-sharded elements on this rank."""
        shard_world = self._fsdp_shard_world(spec)
        # FSDP shards the flattened parameter; each rank holds ceil(numel / world)
        return (spec.numel + shard_world - 1) // shard_world

    def compute_module_params(self, module_idx: int) -> MemoryBucket:
        """
        Compute unsharded parameter memory for one module.

        During the DualPipeV pipeline step, FSDP layers have reshard_after_forward=False,
        so the full unsharded parameters are in memory.
        """
        layers = self.module0_layers if module_idx == 0 else self.module1_layers
        stage_id = self.module0_stage if module_idx == 0 else self.module1_stage
        has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
        has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head

        bucket = MemoryBucket(f"module[{module_idx}] params (stage {stage_id})")

        cfg = self.model_cfg
        H = cfg.hidden_size

        if has_embed:
            bucket.add(f"module[{module_idx}].embed_tokens.weight", (cfg.vocab_size, H), "bf16")

        for layer_idx in layers:
            prefix = f"module[{module_idx}].layer.{layer_idx}"
            for spec in self._layer_params(layer_idx, prefix):
                bucket.specs.append(spec)

        if has_head:
            bucket.add(f"module[{module_idx}].norm.weight", (H,), "bf16")
            bucket.add(f"module[{module_idx}].lm_head.weight", (cfg.vocab_size, H), "bf16")

        # Rotary embedding buffers (not parameters but in memory)
        # cos_cached, sin_cached: [max_pos, head_dim] - relatively small
        max_pos = (
            getattr(cfg, "max_position_embeddings", 40960)
            if hasattr(cfg, "max_position_embeddings")
            else 40960
        )
        bucket.add(
            f"module[{module_idx}].rotary_emb.cos_cached",
            (max_pos, cfg.head_dim),
            "bf16",
        )
        bucket.add(
            f"module[{module_idx}].rotary_emb.sin_cached",
            (max_pos, cfg.head_dim),
            "bf16",
        )

        return bucket

    def compute_fsdp_sharded_params(self) -> MemoryBucket:
        """
        Compute the persistent FSDP-sharded parameter memory.

        FSDP keeps sharded copies alongside unsharded. During the pipeline step,
        both exist in memory for layers with reshard_after_forward=False.

        When FSDP mesh_size=1 (e.g., expert params with dp=1), the sharded copy
        IS the unsharded copy - same tensor, no duplication. We skip these to
        avoid double-counting with params_unsharded.
        """
        bucket = MemoryBucket("FSDP sharded params (persistent)")
        for module_idx in range(2):
            layers = self.module0_layers if module_idx == 0 else self.module1_layers
            has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
            has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head

            cfg = self.model_cfg
            H = cfg.hidden_size

            all_specs: list[TensorSpec] = []
            if has_embed:
                all_specs.append(
                    TensorSpec(f"m[{module_idx}].embed_tokens.weight", (cfg.vocab_size, H), "bf16")
                )
            for layer_idx in layers:
                prefix = f"m[{module_idx}].layer.{layer_idx}"
                all_specs.extend(self._layer_params(layer_idx, prefix))
            if has_head:
                all_specs.append(TensorSpec(f"m[{module_idx}].norm.weight", (H,), "bf16"))
                all_specs.append(
                    TensorSpec(f"m[{module_idx}].lm_head.weight", (cfg.vocab_size, H), "bf16")
                )

            for spec in all_specs:
                shard_world = self._fsdp_shard_world(spec)
                if shard_world <= 1:
                    # mesh_size=1: sharded = full param, already counted in params_unsharded
                    continue
                shard_numel = self._fsdp_shard_size(spec)
                bucket.add(
                    f"shard({spec.name})",
                    (shard_numel,),
                    "bf16",
                )
        return bucket

    def compute_optimizer_states(self) -> MemoryBucket:
        """
        Compute Adam optimizer state memory (exp_avg + exp_avg_sq in fp32).

        Optimizer operates on FSDP-sharded parameters.
        """
        bucket = MemoryBucket("Optimizer states (Adam m+v, fp32)")
        for module_idx in range(2):
            layers = self.module0_layers if module_idx == 0 else self.module1_layers
            has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
            has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head

            cfg = self.model_cfg
            H = cfg.hidden_size

            all_specs: list[TensorSpec] = []
            if has_embed:
                all_specs.append(
                    TensorSpec(f"m[{module_idx}].embed_tokens.weight", (cfg.vocab_size, H), "bf16")
                )
            for layer_idx in layers:
                prefix = f"m[{module_idx}].layer.{layer_idx}"
                all_specs.extend(self._layer_params(layer_idx, prefix))
            if has_head:
                all_specs.append(TensorSpec(f"m[{module_idx}].norm.weight", (H,), "bf16"))
                all_specs.append(
                    TensorSpec(f"m[{module_idx}].lm_head.weight", (cfg.vocab_size, H), "bf16")
                )

            for spec in all_specs:
                shard_numel = self._fsdp_shard_size(spec)
                # exp_avg (m) in fp32
                bucket.add(f"adam_m({spec.name})", (shard_numel,), "fp32")
                # exp_avg_sq (v) in fp32
                bucket.add(f"adam_v({spec.name})", (shard_numel,), "fp32")
        return bucket

    def compute_gradient_bucket(self, module_idx: int) -> MemoryBucket:
        """
        Compute the gradient memory for one module (all layers).

        During the pipeline step with FSDP's set_requires_gradient_sync(False),
        gradients accumulate on the full unsharded parameter in bf16.
        """
        layers = self.module0_layers if module_idx == 0 else self.module1_layers
        has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
        has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head

        bucket = MemoryBucket(f"module[{module_idx}] gradients")

        cfg = self.model_cfg
        H = cfg.hidden_size

        if has_embed:
            bucket.add(f"grad(m[{module_idx}].embed_tokens.weight)", (cfg.vocab_size, H), "bf16")

        for layer_idx in layers:
            prefix = f"grad(m[{module_idx}].layer.{layer_idx})"
            for spec in self._layer_params(layer_idx, prefix):
                bucket.add(spec.name, spec.shape, spec.dtype)

        if has_head:
            bucket.add(f"grad(m[{module_idx}].norm.weight)", (H,), "bf16")
            bucket.add(f"grad(m[{module_idx}].lm_head.weight)", (cfg.vocab_size, H), "bf16")

        return bucket

    def get_module_layer_count(self, module_idx: int) -> int:
        layers = self.module0_layers if module_idx == 0 else self.module1_layers
        return len(layers)

    def get_module_layer_indices(self, module_idx: int) -> list[int]:
        return self.module0_layers if module_idx == 0 else self.module1_layers

    def module_has_embed(self, module_idx: int) -> bool:
        return self.module0_has_embed if module_idx == 0 else self.module1_has_embed

    def module_has_head(self, module_idx: int) -> bool:
        return self.module0_has_head if module_idx == 0 else self.module1_has_head

    def get_total_param_count(self) -> int:
        """Total parameter count across both modules (unsharded, for display)."""
        total = 0
        for module_idx in range(2):
            bucket = self.compute_module_params(module_idx)
            for spec in bucket.specs:
                if "rotary_emb" not in spec.name:
                    total += spec.numel
        return total
