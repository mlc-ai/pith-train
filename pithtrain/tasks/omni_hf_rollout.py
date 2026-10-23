"""Bounded local HF Thinker rollout and exact-answer reward for native GRPO.

No hub access, service, retry, or distributed transport is implicit. The provider
owns one spawned HF worker. Native training remains in the caller's process.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from pithtrain.modules.resume_semantics import implementation_contract, model_contract, seal
from pithtrain.tasks.omni_rl import (
    PithTrainPolicyAdapter,
    PolicyIdentity,
    PolicySnapshot,
    RolloutBatch,
    RolloutContract,
    TorchPolicyAdapter,
    _cpu_tree,
    fingerprint,
)


@dataclass(frozen=True)
class LocalHFRolloutCfg:
    scope: str = "text"
    device: str = "cpu"
    compute_dtype: str = "float32"
    max_requests: int = 4
    group_size: int = 2
    max_sequence_length: int = 256
    max_new_tokens: int = 32
    seed: int = 1234
    timeout_seconds: float = 120.0
    max_weight_bytes: int = 512 << 20
    max_result_bytes: int = 256 << 20
    # Explicit provider weight residency budgets, not whole-training RAM/VRAM.
    # None preserves the small-model default; models above 512 MiB require them.
    max_host_weight_bytes: int | None = None
    max_device_weight_bytes: int | None = None
    cpu_threads: int = 1
    sequential_models: bool = False

    def validate(self):
        if type(self.sequential_models) is not bool:
            raise ValueError("sequential_models must be a boolean")
        if (
            self.scope not in {"text", "thinker"}
            or self.device not in {"cpu", "cuda:0"}
            or self.compute_dtype not in {"float32", "bfloat16"}
        ):
            raise ValueError("Unsupported rollout scope/device/compute dtype")
        for name in (
            "max_requests",
            "group_size",
            "max_sequence_length",
            "max_new_tokens",
            "max_weight_bytes",
            "max_result_bytes",
            "cpu_threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_host_weight_bytes", "max_device_weight_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be an explicit positive integer")
        if self.group_size < 2 or self.max_new_tokens >= self.max_sequence_length:
            raise ValueError("Complete GRPO group and prompt space required")
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or not 0 < self.timeout_seconds <= 3600
        ):
            raise ValueError("Invalid seed or finite request deadline")


def _converter(scope):
    if scope == "text":
        from pithtrain.tasks.convert_checkpoint import qwen3_omni_moe

        return qwen3_omni_moe
    from pithtrain.tasks.convert_checkpoint import qwen3_omni_thinker

    return qwen3_omni_thinker


def _config(raw, scope):
    converter = _converter(scope)
    return converter.text_config(raw) if scope == "text" else converter.thinker_config(raw)


def _canonical(hf, config, scope):
    return _converter(scope).hf_to_canonical(hf, config, prefix="")[0]


def weight_memory_plan(config, cfg, *, local_reference_copy=False):
    """Worst live provider weight phases; no native optimizer/activation estimate.

    The native producer owns reference + publication (2W). During synchronous
    installation its worker owns two FP32 models + one incoming payload (3W).
    The local convenience provider additionally owns a defensive reference copy.
    Sequential execution parks the inactive model on CPU; host admission remains
    conservative. BF16 autocast is not BF16 parameter storage.
    """
    cfg.validate()
    shapes = _converter(cfg.scope).parameter_shapes(config)
    weight = sum(torch.Size(shape).numel() * 4 for shape in shapes.values())
    if weight > cfg.max_weight_bytes:
        raise ValueError("Canonical FP32 weight byte limit exceeded before allocation")
    host = (6 if local_reference_copy else 5) * weight
    device_models = 1 if cfg.sequential_models else 2
    device = device_models * weight if cfg.device.startswith("cuda") else 0
    if weight > 512 << 20 and (
        cfg.max_host_weight_bytes is None or (device and cfg.max_device_weight_bytes is None)
    ):
        raise ValueError(
            "Full-size provider requires explicit host/device weight residency budgets"
        )
    if cfg.max_host_weight_bytes is not None and host > cfg.max_host_weight_bytes:
        raise ValueError("Provider peak host weight residency exceeds explicit budget")
    if cfg.max_device_weight_bytes is not None and device > cfg.max_device_weight_bytes:
        raise ValueError("FP32 HF model residency exceeds explicit device weight budget")
    return dict(
        canonical_fp32_bytes=weight,
        host_peak_weight_bytes=host,
        device_resident_weight_bytes=device,
        worker_models=2,
        device_models=device_models if device else 0,
        sequential_models=cfg.sequential_models,
        host_phase="parent reference/publication + worker models/incoming payload",
        local_reference_copy=local_reference_copy,
        transient_copy_chunk_bytes=4 << 20,
        temporary_weight_files=1,
        additional_memory_required="native training/checkpoint, activations, buffers, allocator and OS",
        feasibility="WEIGHT_BUDGET_ONLY_NOT_FULLSIZE_VALIDATION",
    )


def _stream_weights_digest(values):
    """Exact existing digest encoding, without a tensor-sized Python bytes copy."""
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        flat = value.detach().reshape(-1)
        elements = max(1, (4 << 20) // value.element_size())
        for begin in range(0, flat.numel(), elements):
            piece = flat[begin : begin + elements].contiguous().cpu()
            digest.update(memoryview(piece.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _install_canonical(model, values, config, cfg):
    """Write through the strict converter's HF canonical views, not fused stacks."""
    _check_weights(values, config, cfg)
    stored = model.state_dict()
    target = _canonical(stored, config, cfg.scope)
    storages = {(v.device, v.untyped_storage().data_ptr()) for v in stored.values()}
    if any((v.device, v.untyped_storage().data_ptr()) not in storages for v in target.values()):
        raise ValueError("HF canonical installation must use live parameter storage views")
    # Fused HF expert slices are contiguous live views, as are media parameters.
    # No model parameters or buffers are rebound and no reference dtype changes.
    with torch.no_grad():
        for name, value in values.items():
            destination, source = target[name].view(-1), value.reshape(-1)
            for begin in range(0, source.numel(), (4 << 20) // 4):
                destination[begin : begin + (4 << 20) // 4].copy_(
                    source[begin : begin + (4 << 20) // 4]
                )
    digest = _stream_weights_digest(_canonical(model.state_dict(), config, cfg.scope))
    if digest != _stream_weights_digest(values):
        raise ValueError("Installed policy readback differs from published weights")
    return digest


def _check_weights(values, config, cfg):
    from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import validate_tensors

    weight_memory_plan(config, cfg)
    shapes = _converter(cfg.scope).parameter_shapes(config)
    validate_tensors(values, shapes, "Local rollout complete canonical weights")
    if any(
        v.device.type != "cpu"
        or v.dtype != torch.float32
        or v.requires_grad
        or any(not torch.isfinite(chunk).all() for chunk in v.reshape(-1).split((4 << 20) // 4))
        for v in values.values()
    ):
        raise ValueError("Canonical rollout weights must be finite detached FP32 CPU tensors")


def _new_hf(raw, scope):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerTextModel,
    )

    config = _config(raw, scope)
    config._attn_implementation = "eager"
    if scope == "thinker":
        config.text_config._attn_implementation = "eager"
        config.audio_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    else:
        # This exact HF text trunk plus its ordinary untied output head matches
        # the existing text converter's model.* / lm_head.* namespace.
        model = HFTextPolicy(config, Qwen3OmniMoeThinkerTextModel(config))
    return model.float().eval().requires_grad_(False), config


def _new_hf_from_canonical(raw, cfg, values):
    """Construct a complete text policy for loading, never for random initialization.

    Real CPU constructors retain nonpersistent RoPE buffers. Only initialization
    of parameters that the strict installer replaces is skipped; no meta/to_empty
    conversion can leave computed buffers uninitialized.
    """
    from transformers.initialization import no_init_weights

    if cfg.scope != "text":
        raise ValueError("Load-only HF construction supports the text scope only")
    config = _config(raw, cfg.scope)
    _check_weights(values, config, cfg)
    with no_init_weights():
        model, config = _new_hf(raw, cfg.scope)
    canonical = _canonical(model.state_dict(), config, cfg.scope)
    storage = lambda tensor: (tensor.device, tensor.untyped_storage().data_ptr())
    if {storage(value) for value in canonical.values()} != {
        storage(value) for value in model.parameters()
    } or sum(value.numel() for value in canonical.values()) != sum(
        value.numel() for value in model.parameters()
    ):
        raise ValueError("Load-only canonical weights must cover every HF parameter")
    digest = _install_canonical(model, values, config, cfg)
    return model, config, digest


def _hf_implementation():
    from transformers import masking_utils, modeling_rope_utils
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextModel,
    )

    from pithtrain.modules.omni_rollout_data import file_sha
    from pithtrain.tasks.convert_checkpoint import qwen3_omni_moe, qwen3_omni_thinker

    return dict(
        model=implementation_contract(Qwen3OmniMoeThinkerTextModel),
        masks=file_sha(masking_utils.__file__),
        rope=file_sha(modeling_rope_utils.__file__),
        text_converter=file_sha(qwen3_omni_moe.__file__),
        thinker_converter=file_sha(qwen3_omni_thinker.__file__),
        tokenizers=importlib.metadata.version("tokenizers"),
    )


def execution_identity():
    """Bind worker behavior independently of provider/reward source co-location."""
    return dict(
        schema=1,
        torch=str(torch.__version__),
        transformers=importlib.metadata.version("transformers"),
        hf=_hf_implementation(),
        engine=implementation_contract(_Engine),
        worker=implementation_contract(_WorkerProcess),
    )


def worker_contract(raw, cfg):
    return dict(
        config=_config(raw, cfg.scope).to_dict(),
        scope=cfg.scope,
        compute_dtype=cfg.compute_dtype,
        device=cfg.device,
        attention="eager",
        fp32_masters=True,
        sequential_models=cfg.sequential_models,
        execution=execution_identity(),
    )


def validate_native_runtime(model, raw, cfg):
    """Read actual native stored configs/context, including image_config omitted
    by generic model.config inspection. Pure metadata: no gather or forward.
    """
    from pithtrain.contexts import training

    expected = _config(raw, cfg.scope)
    text = expected if cfg.scope == "text" else expected.text_config
    if cfg.compute_dtype != "bfloat16" or training.fp8 or training.PARAM_DTYPE != torch.bfloat16:
        raise ValueError("Native runtime compute policy differs from HF BF16 contract")
    result = model_contract(model)
    if (
        result["identity"]["modules"][""]["implementation"]
        != "pithtrain.pipeline.dualpipev.DualPipeV"
    ):
        raise ValueError("Unsupported native runtime root")
    for name, module in model.named_modules():
        for attribute in ("config", "image_config"):
            actual = getattr(module, attribute, None)
            if actual is None:
                continue
            if attribute == "image_config":
                wanted = expected
            elif ".visual" in name:
                wanted = getattr(expected, "vision_config", None)
            elif ".audio_tower" in name:
                wanted = getattr(expected, "audio_config", None)
            else:
                wanted = text
            if wanted is None or configuration_identity(actual) != configuration_identity(wanted):
                raise ValueError(f"Native stored {attribute} differs from HF config: {name}")
    return result


def configuration_identity(config):
    """Exclude only HF loading metadata; retain every numerical/config field."""

    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in {"_name_or_path", "_commit_hash", "transformers_version"}
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    return clean(config.to_dict())


def _hf_module_name(name):
    """DualPipeV chunk names retain global layer IDs, never reindex locally."""
    if name.startswith("module."):
        parts = name.split(".", 2)
        if len(parts) == 2:
            return "model"
        if not parts[1].isdigit():
            raise ValueError("Invalid native pipeline chunk name")
        name = parts[2]
    return (
        name
        if name.startswith(("visual.", "audio_tower.", "lm_head."))
        or name in {"visual", "audio_tower", "lm_head"}
        else "model." + name
    )


def _validate_stored_config(actual, wanted, name):
    from pithtrain.modules.resume_semantics import _canonical_model_value

    # model_contract encodes mapping key types. Strip only the same non-numerical
    # HF loader metadata as configuration_identity, preserving all other fields.
    def clean(value):
        if isinstance(value, dict):
            return {
                k: clean(v)
                for k, v in value.items()
                if k not in {"s:_name_or_path", "s:_commit_hash", "s:transformers_version"}
            }
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    if clean(actual) != _canonical_model_value(configuration_identity(wanted)):
        raise ValueError(f"Native stored module config differs from HF config: {name}")


def validate_policy_semantics(semantics, raw, cfg):
    """Explicit supported native/HF correspondence, independent of supplied hashes.

    Equal shapes and a self-consistent seal are insufficient. Custom forward
    callbacks, changed runtime norms/RoPE and unsupported precision fail before
    the worker sees any weights. Different implementations are not asserted to
    be numerically exact; that remains the GPU parity acceptance gate.
    """
    if seal(semantics["identity"]) != semantics:
        raise ValueError("Invalid policy semantic seal")
    identity = semantics["identity"]
    if identity.get("kind") != "native_policy_ranks":
        if cfg.scope != "text" or cfg.compute_dtype != "float32":
            raise ValueError("CPU HF adapter requires explicit text FP32 compute semantics")
        with torch.device("meta"):
            expected, _ = _new_hf(raw, "text")
        expected_forward = dict(
            default="model(*inputs, **context)",
            adapter=implementation_contract(HFCanonicalPolicyAdapter),
        )
        if identity != dict(schema=1, model=model_contract(expected), forward=expected_forward):
            raise ValueError("HF policy config/runtime/forward adapter differs from rollout")
        return worker_contract(raw, cfg)
    if cfg.compute_dtype != "bfloat16" or not identity.get("ranks"):
        raise ValueError("Native publication requires explicit BF16 compute and rank semantics")
    from pithtrain.modules.omni_rollout_data import file_sha

    root = Path(__file__).resolve().parents[1]
    expected_forward = dict(
        adapter=implementation_contract(PithTrainPolicyAdapter),
        step=dict(
            name="pithtrain.pipeline.dualpipev.DualPipeV.step",
            source_sha256=file_sha(root / "pipeline/dualpipev.py"),
        ),
    )
    expected_config = _config(raw, cfg.scope)
    text = expected_config if cfg.scope == "text" else expected_config.text_config
    with torch.device("meta"):
        expected_model, _ = _new_hf(raw, cfg.scope)
    expected_modules = model_contract(expected_model)["identity"]["modules"]
    for rank in identity["ranks"]:
        if seal(rank["identity"]) != rank:
            raise ValueError("Native rank semantic seal differs")
        value = rank["identity"]
        training = value.get("training") or {}
        if (
            value.get("forward") != expected_forward
            or training.get("fp8") is not False
            or training.get("benchmark") is not False
            or training.get("moe_load_balance_coef") != 0
            or training.get("native_multimodal", False) != (cfg.scope == "thinker")
        ):
            raise ValueError("Unsupported native forward/precision/routing semantics")
        path = Path(training["model"]) / "config.json"
        actual_config = _config(json.loads(path.read_text()), cfg.scope)
        if configuration_identity(actual_config) != configuration_identity(expected_config):
            raise ValueError("Native/HF model configuration differs")
        contract = value["model"]
        if seal(contract["identity"]) != contract:
            raise ValueError("Native model semantic seal differs")
        modules = contract["identity"]["modules"]
        root_type = modules.get("", {}).get("implementation")
        if root_type != "pithtrain.pipeline.dualpipev.DualPipeV":
            raise ValueError("Unsupported native policy root implementation")
        for name, module in modules.items():
            implementation = module["implementation"]
            if implementation.startswith("pithtrain."):
                supported = (
                    "pithtrain.pipeline.dualpipev.",
                    "pithtrain.models.qwen3_omni_moe.",
                    "pithtrain.models.qwen3_omni_multimodal.",
                    "pithtrain.models.qwen3_omni_image.",
                    "pithtrain.models.qwen3_omni_vision.",
                    "pithtrain.models.qwen3_omni_audio.",
                    "pithtrain.operators.linear.",
                    "pithtrain.operators.grouped_linear.",
                )
                if not implementation.startswith(supported):
                    raise ValueError("Unreviewed native submodule implementation")
                module_name = implementation.rsplit(".", 1)[0]
                source = root.parent / (module_name.replace(".", "/") + ".py")
                if not source.is_file() or file_sha(source) != module["implementation_sha256"]:
                    raise ValueError(
                        "Native policy implementation differs from reviewed local source"
                    )
            attributes = module["numerical_attributes"]
            hf_name = _hf_module_name(name)
            if module.get("config") is not None:
                wanted = (
                    getattr(expected_config, "vision_config", None)
                    if hf_name.startswith("visual")
                    else getattr(expected_config, "audio_config", None)
                    if hf_name.startswith("audio_tower")
                    else text
                )
                if wanted is None:
                    raise ValueError("Native module config has unsupported component scope")
                _validate_stored_config(module["config"], wanted, name)
            hf_attributes = expected_modules.get(hf_name, {}).get("numerical_attributes", {})
            numerical = {
                "variance_epsilon",
                "eps",
                "theta",
                "base",
                "norm_topk_prob",
                "scaling",
                "attention_dropout",
                "dropout",
                "hidden_act",
                "rope_theta",
                "partial_rotary_factor",
                "top_k",
                "mrope_section",
                "head_dim",
                "num_heads",
                "num_key_value_groups",
                "rope_type",
            }
            if any(
                attributes[key] != hf_attributes[key]
                for key in attributes.keys() & hf_attributes.keys() & numerical
            ):
                raise ValueError(f"Native/HF runtime numerical attributes differ: {name}")
            is_text = ".visual" not in name and ".audio_tower" not in name
            if (
                is_text
                and "variance_epsilon" in attributes
                and attributes["variance_epsilon"] != text.rms_norm_eps
            ):
                raise ValueError("Native runtime RMS epsilon differs from HF config")
            if is_text:
                for key, wanted in (
                    ("top_k", text.num_experts_per_tok),
                    ("num_experts_per_tok", text.num_experts_per_tok),
                    ("head_dim", text.head_dim),
                    ("num_heads", text.num_attention_heads),
                    ("num_kv_heads", text.num_key_value_heads),
                    ("mrope_section", list(text.rope_parameters["mrope_section"])),
                ):
                    if key in attributes and attributes[key] != wanted:
                        raise ValueError(f"Native runtime {key} differs from HF config")
            if is_text and "RotaryEmbedding" in implementation:
                theta = attributes.get("theta", attributes.get("base"))
                if theta != text.rope_parameters["rope_theta"]:
                    raise ValueError("Native runtime RoPE differs from HF config")
    return worker_contract(raw, cfg)


class HFTextPolicy(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config, self.model = config, model
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        hidden = self.model(input_ids=input_ids, use_cache=False, **kwargs).last_hidden_state
        return self.lm_head(hidden)


class HFCanonicalPolicyAdapter(TorchPolicyAdapter):
    """Local actual-HF training adapter for CPU integration, with canonical export."""

    def __init__(self, model, optimizer, *, scheduler=None):
        if type(model) is not HFTextPolicy:
            raise ValueError("Only the explicit HF text policy forward is supported")
        super().__init__(model, optimizer, scheduler=scheduler)

    def snapshot(self, version, contract):
        values = _canonical(_cpu_tree(self.model.state_dict()), self.model.config, "text")
        semantics = self.model_semantics()
        return PolicySnapshot(
            PolicyIdentity(version, _stream_weights_digest(values), contract, semantics["sha256"]),
            values,
            semantics,
        )


class _Engine:
    def __init__(self, raw, cfg, reference):
        if importlib.metadata.version("transformers") != "5.17.0":
            raise ValueError("HF worker requires Transformers 5.17.0")
        self.cfg = cfg
        self.config = _config(raw, cfg.scope)
        _check_weights(reference, self.config, cfg)
        self.raw = copy.deepcopy(raw)
        if cfg.scope == "text":
            self.reference, _, self.reference_digest = _new_hf_from_canonical(raw, cfg, reference)
            # The policy's complete snapshot arrives in synchronize. Do not copy
            # disposable initialized weights or transfer them to the device.
            self.policy = None
        else:
            self.policy, _ = _new_hf(raw, cfg.scope)
            self.reference = copy.deepcopy(self.policy)
            self.reference_digest = _install_canonical(self.reference, reference, self.config, cfg)
        self.reference.to("cpu" if cfg.sequential_models else cfg.device)
        if self.policy is not None:
            self.policy.to(cfg.device)
        self.contract = worker_contract(self.config.to_dict(), cfg)

    def synchronize(self, values):
        if self.policy is None:
            policy, _, digest = _new_hf_from_canonical(self.raw, self.cfg, values)
            policy.to(self.cfg.device)
            # Keep the exact installed-device readback after the first transfer.
            if (
                _stream_weights_digest(_canonical(policy.state_dict(), self.config, self.cfg.scope))
                != digest
            ):
                raise ValueError("HF installed policy changed during device transfer")
            self.policy = policy
        else:
            digest = _install_canonical(self.policy, values, self.config, self.cfg)
        return dict(
            installed_weights_sha256=digest,
            reference_weights_sha256=self.reference_digest,
            worker_contract=self.contract,
        )

    def _logits(self, model, ids, media):
        kwargs = {
            k: v.to(self.cfg.device)
            for k, v in media.items()
            if k not in {"input_ids", "attention_mask"}
        }
        if "use_audio_in_video" in kwargs:
            kwargs["use_audio_in_video"] = bool(kwargs["use_audio_in_video"].item())
        if self.cfg.scope == "thinker":
            kwargs["use_cache"] = False
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.cfg.device.split(":")[0],
                dtype=torch.bfloat16,
                enabled=self.cfg.compute_dtype == "bfloat16",
            ),
        ):
            result = model(input_ids=ids.to(self.cfg.device), **kwargs)
        logits = result if isinstance(result, torch.Tensor) else result.logits
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite HF policy logits")
        return logits.float().cpu()

    def _selected_logprobs(self, model, inputs, targets, prompt):
        return (
            self._logits(model, inputs, prompt)
            .log_softmax(-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )

    def generate(self, prompt, rng, eos_token_id):
        if self.policy is None:
            raise ValueError("HF policy requires a complete synchronized snapshot")
        ids = prompt["input_ids"]
        length = ids.shape[1]
        if length + self.cfg.max_new_tokens > self.cfg.max_sequence_length:
            raise ValueError("Expanded prompt plus response exceeds fixed sequence bound")
        if self.cfg.scope == "text" and set(prompt) - {"input_ids", "attention_mask"}:
            raise ValueError("Text rollout cannot discard media context")
        generator = torch.Generator().set_state(rng)
        all_ids, all_targets, all_masks, old_values, ref_values, contexts, medias = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(self.cfg.group_size):
            sequence = ids.clone()
            active, mask = True, []
            for _ in range(self.cfg.max_new_tokens):
                if active:
                    logits = self._logits(self.policy, sequence, prompt)[:, -1]
                    token = torch.multinomial(logits.softmax(-1), 1, generator=generator)
                else:
                    token = torch.tensor([[eos_token_id]], dtype=torch.long)
                mask.append(active)
                sequence = torch.cat((sequence, token), dim=1)
                if int(token.item()) == eos_token_id:
                    active = False
            # EOS tail is outside the objective. It is still real causal input,
            # avoiding unsupported padded native media microbatches.
            sequence = torch.cat(
                (
                    sequence,
                    torch.full(
                        (1, self.cfg.max_sequence_length + 1 - sequence.shape[1]),
                        eos_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=1,
            )
            inputs, targets = sequence[:, :-1], sequence[:, 1:]
            valid = torch.zeros_like(inputs, dtype=torch.bool)
            valid[:, length - 1 : length - 1 + len(mask)] = torch.tensor(mask)
            if self.cfg.scope == "thinker":
                from pithtrain.models.qwen3_omni_multimodal_inputs import multimodal_position_ids

                # Reject generated unpaired media delimiters; do not change the
                # categorical policy or silently mask its probability support.
                multimodal_position_ids(
                    inputs,
                    {**prompt, "input_ids": inputs, "attention_mask": torch.ones_like(inputs)},
                    self.config,
                )
            old = self._selected_logprobs(self.policy, inputs, targets, prompt)
            if not self.cfg.sequential_models:
                ref_values.append(self._selected_logprobs(self.reference, inputs, targets, prompt))
            all_ids.append(inputs)
            all_targets.append(targets)
            all_masks.append(valid)
            old_values.append(old)
            context = {
                k: v.clone()
                for k, v in prompt.items()
                if k
                not in {
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "pixel_values_videos",
                    "input_features",
                }
            }
            context.update(input_ids=inputs.clone(), attention_mask=torch.ones_like(inputs))
            contexts.append(context if self.cfg.scope == "thinker" else None)
            medias.append(
                {
                    k: v.clone()
                    for k, v in prompt.items()
                    if k in {"pixel_values", "pixel_values_videos", "input_features"}
                }
                or None
            )
        if self.cfg.sequential_models:
            # Finish every policy sample before moving either model. Transfers
            # preserve FP32 parameters/buffers and do not touch the CPU RNG.
            self.policy.to("cpu")
            try:
                self.reference.to(self.cfg.device)
                for inputs, targets in zip(all_ids, all_targets, strict=True):
                    ref_values.append(
                        self._selected_logprobs(self.reference, inputs, targets, prompt)
                    )
            finally:
                # Restore policy placement even if reference scoring fails.
                # The owned worker still fails closed on any request error.
                self.reference.to("cpu")
                self.policy.to(self.cfg.device)
        return dict(
            input_ids=torch.cat(all_ids),
            sampled_token_ids=torch.cat(all_targets),
            token_mask=torch.cat(all_masks),
            old_logprobs=torch.cat(old_values),
            reference_logprobs=torch.cat(ref_values),
            model_context=tuple(contexts) if self.cfg.scope == "thinker" else None,
            media_inputs=tuple(medias) if self.cfg.scope == "thinker" else None,
            rng=generator.get_state(),
        )


def _worker(connection, directory, raw, cfg):
    """Small Pipe control messages; bounded tensor payloads stay in owned local files."""
    try:
        torch.set_num_threads(cfg.cpu_threads)
        reference = torch.load(
            Path(directory) / "reference.pt", map_location="cpu", weights_only=True
        )
        engine = _Engine(raw, cfg, reference)
        del reference
        (Path(directory) / "reference.pt").unlink()
        connection.send(("ready", None))
        while True:
            command = connection.recv()
            if command == "close":
                return
            payload = torch.load(
                Path(directory) / "request.pt", map_location="cpu", weights_only=True
            )
            if command == "synchronize":
                result = engine.synchronize(payload)
            elif command == "generate":
                result = engine.generate(**payload)
            else:
                raise ValueError("Unknown HF worker operation")
            del payload
            (Path(directory) / "request.pt").unlink()
            output = Path(directory) / "result.pt"
            torch.save(result, output)
            if output.stat().st_size > cfg.max_result_bytes:
                raise ValueError("HF response exceeds explicit result byte limit")
            connection.send(("ok", None))
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


class _WorkerProcess:
    def __init__(self, raw, cfg, reference):
        deadline = time.monotonic() + cfg.timeout_seconds
        self.cfg = cfg
        self.directory = tempfile.TemporaryDirectory(prefix="pithtrain-hf-rollout-")
        self.process = None
        self.connection = None
        try:
            torch.save(reference, Path(self.directory.name) / "reference.pt")
            self._remaining(deadline)
            context = mp.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(
                target=_worker, args=(child, self.directory.name, raw, cfg), daemon=True
            )
            self.process.start()
            child.close()
            self._wait("ready", deadline)
        except BaseException:
            self.close()
            raise

    def _remaining(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Owned HF worker exceeded request deadline; no retry")
        return remaining

    def _wait(self, expected, deadline):
        try:
            if not self.connection.poll(self._remaining(deadline)):
                raise TimeoutError("Owned HF worker exceeded request deadline; no retry")
            status, error = self.connection.recv()
            self._remaining(deadline)
            if status != expected:
                raise ValueError(f"Owned HF worker failed: {error}")
        except BaseException:
            self.close()
            raise

    def call(self, command, payload):
        if self.process is None or not self.process.is_alive():
            raise ValueError("HF worker is closed/failed; construct fresh and restore")
        # Serialization, IPC and result loading share one elapsed-time budget.
        deadline = time.monotonic() + self.cfg.timeout_seconds
        try:
            torch.save(payload, Path(self.directory.name) / "request.pt")
            self._remaining(deadline)
            self.connection.send(command)
            self._wait("ok", deadline)
            path = Path(self.directory.name) / "result.pt"
            if path.stat().st_size > self.cfg.max_result_bytes:
                raise ValueError("HF result byte limit exceeded")
            result = torch.load(path, map_location="cpu", weights_only=True)
            path.unlink()
            self._remaining(deadline)
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            # Reaping a large CUDA process can outlast the signal itself. Nested
            # error handlers share this deadline instead of restarting the wait.
            if not hasattr(self, "_close_deadline"):
                self._close_deadline = time.monotonic() + 30
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=min(2, max(0, self._close_deadline - time.monotonic())))
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=max(0, self._close_deadline - time.monotonic()))
            if self.process.is_alive():
                raise RuntimeError("Owned HF process cleanup unverified")
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.directory.cleanup()


class LocalHFThinkerRollout:
    """Single-rank local provider; CPU reference/policy is the default placement.

    FP32 policy/reference masters preserve exact weight installation. Optional
    BF16 autocast controls compute. This intentionally does not claim a 30B model
    fits alongside native training; limits must be chosen after a memory review.
    """

    def __init__(self, cfg, *, model_config, prompts, reference, eos_token_id, reward_config):
        cfg.validate()
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Local HF bridge supports DP1/PP1/EP1 only")
        if importlib.metadata.version("transformers") != "5.17.0":
            raise ValueError("This provider requires the reviewed Transformers 5.17.0 API")
        self.cfg, self.raw, self.prompts = cfg, copy.deepcopy(model_config), prompts
        self.config = _config(self.raw, cfg.scope)
        weight_memory_plan(self.config, cfg, local_reference_copy=True)
        _check_weights(reference, self.config, cfg)
        self.reference = _cpu_tree(reference)
        vocab = (
            self.config.vocab_size if cfg.scope == "text" else self.config.text_config.vocab_size
        )
        if type(eos_token_id) is not int or not 0 <= eos_token_id < vocab:
            raise ValueError("Explicit tokenizer EOS must fit the policy vocabulary")
        if any(
            p.inputs["input_ids"].min() < 0 or p.inputs["input_ids"].max() >= vocab
            for p in prompts.prompts
        ):
            raise ValueError("Prepared prompt IDs exceed the model vocabulary")
        self.eos_token_id = eos_token_id
        self.contract = RolloutContract(
            fingerprint(prompts.identity["provenance"]),
            prompts.sha256,
            fingerprint(
                dict(config=self.raw, input_format="document-prefix-v1", hf_attention="eager")
            ),
            _stream_weights_digest(self.reference),
            fingerprint(reward_config),
        )
        self.semantic = dict(
            cfg=asdict(cfg),
            model=self.raw,
            contract=asdict(self.contract),
            eos=eos_token_id,
            execution=execution_identity(),
            implementation=implementation_contract(type(self)),
        )
        self.requests = self.cursor = 0
        self.rng = torch.Generator().manual_seed(cfg.seed).get_state()
        self.identity = self.worker = self.installation_receipt = None
        self.failed = False

    def synchronize(self, snapshot):
        if self.failed:
            raise ValueError("Failed provider requires fresh construction and restore")
        self.prompts.validate()
        if _stream_weights_digest(self.reference) != self.contract.reference_policy_sha256:
            raise ValueError("Fixed reference policy changed")
        snapshot.identity.validate()
        if (
            snapshot.identity.contract != self.contract
            or seal(snapshot.semantics["identity"]) != snapshot.semantics
            or snapshot.identity.semantics_sha256 != snapshot.semantics["sha256"]
        ):
            raise ValueError("Snapshot contract/model semantics mismatch")
        expected_worker = validate_policy_semantics(snapshot.semantics, self.raw, self.cfg)
        _check_weights(snapshot.payload, self.config, self.cfg)
        if _stream_weights_digest(snapshot.payload) != snapshot.identity.weights_sha256:
            raise ValueError("Published canonical payload digest mismatch")
        try:
            if self.worker is None:
                self.worker = _WorkerProcess(self.raw, self.cfg, self.reference)
            receipt = self.worker.call("synchronize", snapshot.payload)
            if receipt != dict(
                installed_weights_sha256=snapshot.identity.weights_sha256,
                reference_weights_sha256=self.contract.reference_policy_sha256,
                worker_contract=expected_worker,
            ):
                raise ValueError("HF worker did not acknowledge installed policy/reference")
            self.identity, self.installation_receipt = snapshot.identity, receipt
            return self.identity
        except BaseException:
            self.failed = True
            self.close()
            raise

    def generate(self, request):
        if (
            self.failed
            or request.identity != self.identity
            or not request.rollout_leader
            or request.dp_rank != 0
            or request.pp_rank != 0
        ):
            raise ValueError("No matching installed policy or unsupported distributed request")
        if (
            request.request_index != self.requests
            or request.data_cursor != self.cursor
            or request.seed != self.cfg.seed + self.requests
            or request.group_size != self.cfg.group_size
            or request.max_samples < self.cfg.group_size
            or request.max_sequence_length != self.cfg.max_sequence_length
            or self.requests >= self.cfg.max_requests
        ):
            raise ValueError("Rollout request cursor/seed/bounds differ")
        try:
            prompt = self.prompts.payload(self.requests)
            value = self.worker.call(
                "generate", dict(prompt=prompt, rng=self.rng, eos_token_id=self.eos_token_id)
            )
            self.rng = value.pop("rng")
            torch.Generator().set_state(self.rng)
            sample = self.prompts.prompts[self.requests].sample_id
            result = RolloutBatch(
                request.identity,
                request.request_index,
                group_ids=(sample,) * self.cfg.group_size,
                sample_ids=tuple(
                    f"{request.request_index}:{sample}:{i}" for i in range(self.cfg.group_size)
                ),
                **value,
            )
            self.requests += 1
            self.cursor += self.cfg.group_size
            return result
        except BaseException:
            self.failed = True
            self.close()
            raise

    def state_dict(self):
        if self.failed:
            raise ValueError("Cannot checkpoint failed HF provider")
        self.prompts.validate()
        return dict(
            schema=1,
            semantic=copy.deepcopy(self.semantic),
            requests=self.requests,
            cursor=self.cursor,
            rng=self.rng.clone(),
        )

    def validate_state_dict(self, state):
        self.prompts.validate()
        if _stream_weights_digest(self.reference) != self.contract.reference_policy_sha256:
            raise ValueError("Fixed reference policy changed")
        if (
            set(state) != {"schema", "semantic", "requests", "cursor", "rng"}
            or state["schema"] != 1
            or state["semantic"] != self.semantic
        ):
            raise ValueError("HF rollout resume data/processor/reference/reward/semantics changed")
        if (
            type(state["requests"]) is not int
            or not 0 <= state["requests"] <= min(self.cfg.max_requests, len(self.prompts.prompts))
            or type(state["cursor"]) is not int
            or state["cursor"] != state["requests"] * self.cfg.group_size
        ):
            raise ValueError("HF rollout resume cursor mismatch")
        if (
            not isinstance(state["rng"], torch.Tensor)
            or state["rng"].dtype != torch.uint8
            or state["rng"].device.type != "cpu"
        ):
            raise ValueError("Invalid provider RNG state")
        torch.Generator().set_state(state["rng"])

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.close()
        self.requests, self.cursor, self.rng = (
            state["requests"],
            state["cursor"],
            state["rng"].clone(),
        )
        self.identity = self.installation_receipt = None
        self.failed = False

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None


class ExactAnswerReward:
    """Deterministic local text reward. No executable callback or remote judge.

    Normalization is configurable and fingerprinted; only response targets under
    the aligned mask are decoded. Other rewards implement the existing protocol.
    """

    def __init__(self, prompts, tokenizer, *, strip=True, casefold=False, dp_rank=0, dp_size=1):
        if type(strip) is not bool or type(casefold) is not bool:
            raise ValueError("Reward normalization flags must be booleans")
        self.prompts, self.tokenizer = prompts, tokenizer
        if type(dp_rank) is not int or type(dp_size) is not int or not 0 <= dp_rank < dp_size:
            raise ValueError("Invalid reward data-parallel coordinate")
        self.dp_rank, self.dp_size = dp_rank, dp_size
        self.config = dict(
            kind="exact_answer_v1",
            strip=strip,
            casefold=casefold,
            dp_size=dp_size,
            prompts=prompts.sha256,
            tokenizer=tokenizer.backend_tokenizer.to_str(),
            special_tokens=tokenizer.special_tokens_map,
            implementation=implementation_contract(type(self)),
        )
        self.calls = 0
        self.last_scores = None

    def _normalize(self, value):
        if self.config["strip"]:
            value = value.strip()
        return value.casefold() if self.config["casefold"] else value

    def score(self, request, batch):
        self.prompts.validate()
        self._validate_tokenizer()
        if (
            request.request_index != self.calls
            or batch.request_index != self.calls
            or batch.identity != request.identity
            or request.identity.contract.reward_config_sha256 != fingerprint(self.config)
        ):
            raise ValueError("Reward request identity/config/cursor mismatch")
        if request.dp_rank != self.dp_rank:
            raise ValueError("Reward data-parallel coordinate differs")
        target = self._normalize(
            self.prompts.prompts[self.calls * self.dp_size + self.dp_rank].answer
        )
        rewards = []
        for tokens, mask in zip(batch.sampled_token_ids, batch.token_mask, strict=True):
            if tokens.numel() > request.max_sequence_length:
                raise ValueError("Reward decode exceeds bounded request length")
            text = self.tokenizer.decode(
                tokens[mask].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            rewards.append(float(self._normalize(text) == target))
        self.calls += 1
        self.last_scores = torch.tensor(rewards, dtype=torch.float32)
        return self.last_scores.clone()

    def state_dict(self):
        self._validate_tokenizer()
        return dict(
            schema=1, config=copy.deepcopy(self.config), calls=self.calls, dp_rank=self.dp_rank
        )

    def validate_state_dict(self, state):
        self.prompts.validate()
        self._validate_tokenizer()
        if (
            set(state) != {"schema", "config", "calls", "dp_rank"}
            or state["schema"] != 1
            or state["config"] != self.config
            or state["dp_rank"] != self.dp_rank
            or type(state["calls"]) is not int
            or not 0 <= state["calls"] <= len(self.prompts.prompts) // self.dp_size
        ):
            raise ValueError("Reward resume config/cursor differs")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.calls = state["calls"]
        self.last_scores = None

    def _validate_tokenizer(self):
        if (
            self.tokenizer.backend_tokenizer.to_str() != self.config["tokenizer"]
            or self.tokenizer.special_tokens_map != self.config["special_tokens"]
        ):
            raise ValueError("Reward tokenizer changed")


def publish_native_single_rank(model, config, cfg, version, contract, semantics):
    """Explicit real canonical export. No hidden full-model gather across ranks."""
    from torch.distributed.tensor import DTensor

    from pithtrain.modules.checkpoint import to_canonical_model

    if not dist.is_initialized() or dist.get_world_size() != 1:
        raise ValueError("Native local HF publisher requires initialized world_size=1")
    weight_memory_plan(config, cfg)
    shapes = _converter(cfg.scope).parameter_shapes(config)
    if sum(torch.Size(shape).numel() * 4 for shape in shapes.values()) > cfg.max_weight_bytes:
        raise ValueError("Native publication exceeds reviewed weight byte bound")
    localized = {}
    for name, value in model.named_parameters():
        value = value.detach()
        if isinstance(value, DTensor):
            value = value.full_tensor()
        localized[name] = value.to(device="cpu", copy=True)
    canonical = to_canonical_model(localized, model)
    _check_weights(canonical, config, cfg)
    return PolicySnapshot(
        PolicyIdentity(version, _stream_weights_digest(canonical), contract, semantics["sha256"]),
        canonical,
        semantics,
    )
