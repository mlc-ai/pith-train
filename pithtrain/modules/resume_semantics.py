"""Versioned numerical contracts for strict resume and rollout publication.

Tensor state belongs to the checkpoint. These contracts bind configuration,
audited numerical attributes, implementation, and verifiable LR/forward callables.
Callable objects with opaque/mutable state are deliberately unsupported.
"""

import builtins
import dis
import functools
import hashlib
import inspect
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel


def seal(identity):
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return dict(identity=identity, sha256=hashlib.sha256(raw).hexdigest())


def implementation_contract(value):
    source = inspect.getsourcefile(value)
    if source is None:
        raise ValueError("Strict contract requires inspectable implementation source")
    return dict(
        name=value.__module__ + "." + value.__qualname__,
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
    )


def callable_contract(function, *, bound_objects=()):
    """Capture code, defaults, closures and referenced globals, without executing it.

    Source paths/line numbers are not semantics. Functions recreated by an
    identical factory agree; changed code or captured values cannot silently
    inherit a checkpoint. Opaque callables and dynamic namespace access fail closed.
    """
    seen = set()
    permitted_builtins = {
        "abs",
        "all",
        "any",
        "bool",
        "enumerate",
        "float",
        "int",
        "len",
        "max",
        "min",
        "range",
        "round",
        "sum",
        "tuple",
        "zip",
    }

    def encode(value):
        for obj, contract in bound_objects:
            if value is obj:
                return {"bound_object": contract}
        if isinstance(value, types.CodeType):
            return dict(
                bytecode=value.co_code.hex(),
                constants=[encode(v) for v in value.co_consts],
                names=value.co_names,
                varnames=value.co_varnames,
                freevars=value.co_freevars,
                cellvars=value.co_cellvars,
                argcount=value.co_argcount,
                posonlyargcount=value.co_posonlyargcount,
                kwonlyargcount=value.co_kwonlyargcount,
                flags=value.co_flags,
                exceptiontable=value.co_exceptiontable.hex(),
            )
        if isinstance(value, types.ModuleType):
            if value.__name__ != "math" and not value.__name__.startswith("torch"):
                raise ValueError("Unverifiable callable module dependency")
            path = getattr(value, "__file__", None)
            return dict(
                module=value.__name__,
                source_sha256=None
                if path is None
                else hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                torch=str(torch.__version__),
            )
        if inspect.isbuiltin(value):
            if value.__module__ != "builtins" or value.__name__ not in permitted_builtins:
                raise ValueError("Unverifiable callable builtin dependency")
            return {"builtin": value.__name__}
        if isinstance(value, type) and value in (int, float, bool, tuple):
            return {"builtin": value.__name__}
        if isinstance(value, (types.FunctionType, functools.partial, types.MethodType)):
            if id(value) in seen:
                raise ValueError("Recursive callable contracts are unsupported")
            seen.add(id(value))
            try:
                if isinstance(value, functools.partial):
                    return dict(
                        partial=encode(value.func),
                        args=encode(value.args),
                        keywords=encode(value.keywords),
                    )
                if isinstance(value, types.MethodType):
                    return dict(method=encode(value.__func__), self=encode(value.__self__))

                def global_names(code):
                    names = {
                        op.argval
                        for op in dis.get_instructions(code)
                        if op.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
                    }
                    for constant in code.co_consts:
                        if isinstance(constant, types.CodeType):
                            names.update(global_names(constant))
                    return names

                names = global_names(value.__code__)
                globals_contract = {}
                for name in sorted(names):
                    if name in value.__globals__:
                        globals_contract[name] = encode(value.__globals__[name])
                    elif hasattr(builtins, name):
                        globals_contract[name] = encode(getattr(builtins, name))
                    else:
                        raise ValueError(f"Unresolved callable global: {name}")
                return dict(
                    code=encode(value.__code__),
                    defaults=encode(value.__defaults__),
                    kwdefaults=encode(value.__kwdefaults__),
                    globals=globals_contract,
                    closure=[encode(c.cell_contents) for c in value.__closure__ or ()],
                )
            finally:
                seen.remove(id(value))
        if callable(value):
            raise ValueError("Unverifiable callable object in strict resume contract")
        if isinstance(value, (tuple, list)):
            return {"tuple" if isinstance(value, tuple) else "list": [encode(v) for v in value]}
        if isinstance(value, dict):
            if any(type(key) not in (str, int) for key in value):
                raise ValueError("Unverifiable callable mapping key type")
            tagged = {
                ("s:" if type(key) is str else "i:") + str(key): item for key, item in value.items()
            }
            return {"mapping": {key: encode(tagged[key]) for key in sorted(tagged)}}
        return _canonical_model_value(value)

    return seal(
        dict(
            schema=1,
            kind="python_callable",
            python=list(sys.version_info[:2]),
            callable=encode(function),
        )
    )


def scheduler_contract(scheduler):
    """Bind stable policy separately from mutable scheduler counters/state."""
    if scheduler is None:
        return None
    if type(scheduler) not in (torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.LambdaLR):
        raise ValueError("Strict scheduler semantics support exact StepLR/LambdaLR only")
    mutable = {"last_epoch", "_step_count", "_last_lr", "_get_lr_called_within_step"}
    config = {
        k: v
        for k, v in scheduler.state_dict().items()
        if k not in mutable | {"lr_lambdas", "_pith_factory_contract"}
    }
    lambdas = None
    if type(scheduler) is torch.optim.lr_scheduler.LambdaLR:
        lambdas = [callable_contract(fn) for fn in scheduler.lr_lambdas]
    return seal(
        dict(
            schema=1,
            kind="scheduler_policy",
            torch=str(torch.__version__),
            implementation=implementation_contract(type(scheduler)),
            configuration=_canonical_model_value(config),
            callables=lambdas,
            factory=getattr(scheduler, "_pith_factory_contract", None),
        )
    )


def bind_scheduler_factory(schedulers, factory, arguments):
    contract = seal(
        dict(
            schema=1,
            implementation=implementation_contract(factory),
            arguments=_canonical_model_value(arguments),
        )
    )
    for scheduler in schedulers:
        scheduler._pith_factory_contract = contract
    return schedulers


_NUMERICAL_ATTRIBUTES = (
    "base",
    "theta",
    "rope_theta",
    "rope_type",
    "head_dim",
    "dim",
    "sections",
    "mrope_section",
    "interleaved",
    "variance_epsilon",
    "eps",
    "scaling",
    "scale",
    "attention_scaling",
    "attention_dropout",
    "dropout",
    "p",
    "sliding_window",
    "activation",
    "hidden_act",
    "attn_implementation",
    "num_heads",
    "num_kv_heads",
    "num_key_value_groups",
    "top_k",
    "num_experts_per_tok",
    "norm_topk_prob",
    "num_experts",
    "num_code_groups",
    "num_quantizers",
    "patch_size",
    "temporal_patch_size",
    "spatial_merge_size",
    "deepstack_visual_indexes",
    "use_postshuffle_norm",
    "n_window",
    "n_window_infer",
    "upsample_factor",
    "upsample_rates",
    "upsampling_ratios",
    "stride",
    "padding",
    "dilation",
    "groups",
    "padding_mode",
    "ceil_mode",
    "count_include_pad",
    "normalized_shape",
)


def _canonical_model_value(value):
    """Stable JSON values only: no object repr, addresses, tensors or runtime caches."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite speech model configuration")
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_canonical_model_value(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError("Speech config mapping keys must be strings or integers")
        # HF id2label uses integer keys. Type tags avoid int/string collisions.
        tagged = {
            ("s:" if type(key) is str else "i:") + str(key): item for key, item in value.items()
        }
        return {key: _canonical_model_value(tagged[key]) for key in sorted(tagged)}
    if hasattr(value, "to_json_dict"):
        return _canonical_model_value(value.to_json_dict())
    if hasattr(value, "to_dict"):
        return _canonical_model_value(value.to_dict())
    if isinstance(value, SimpleNamespace):
        return _canonical_model_value(vars(value))
    raise ValueError(f"Unsupported speech model contract value type: {type(value).__name__}")


def model_contract(model):
    """Bind configuration AND actual non-tensor model semantics before any restore.

    FSDP's generated wrapper class is normalized to its original module class.
    No parameters/buffers are read or gathered. Dynamic caches and training mode
    are excluded; the checkpoint separately owns mutable tensor/mode/RNG state.
    """
    base = model.module if isinstance(model, DistributedDataParallel) else model
    config = getattr(base, "config", None)
    composite = _canonical_model_value(config)
    for name in ("thinker", "talker", "code2wav"):
        expected = getattr(config, name + "_config", None)
        actual = getattr(getattr(base, name, None), "config", None)
        if expected is not None and actual is not None:
            if _canonical_model_value(expected) != _canonical_model_value(actual):
                raise ValueError(f"Speech component config differs from composite: {name}")

    def semantic_config(name):
        component = name.split(".", 1)[0]
        cfg = getattr(config, component + "_config", None)
        if cfg is None:
            return None
        if component == "thinker":
            field = (
                "vision_config"
                if name.startswith("thinker.visual")
                else "audio_config"
                if name.startswith("thinker.audio_tower")
                else "text_config"
            )
            return getattr(cfg, field, cfg)
        if component == "talker":
            field = (
                "code_predictor_config"
                if name.startswith("talker.code_predictor")
                else "text_config"
            )
            return getattr(cfg, field, cfg)
        return cfg

    files, modules = {}, {}
    for name, module in base.named_modules():
        cls = type(module)
        if isinstance(module, FSDPModule):
            cls = next(
                candidate
                for candidate in cls.__mro__[1:]
                if issubclass(candidate, torch.nn.Module) and not issubclass(candidate, FSDPModule)
            )
        identifier = cls.__module__ + "." + cls.__qualname__
        source = inspect.getsourcefile(cls)
        if source is None:
            raise ValueError(f"Speech model implementation source unavailable: {identifier}")
        if source not in files:
            files[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        attributes = {
            key: _canonical_model_value(getattr(module, key))
            for key in _NUMERICAL_ATTRIBUTES
            if hasattr(module, key)
            # Child modules have their own entry; tensor state is checkpointed separately.
            and not isinstance(getattr(module, key), (torch.nn.Module, torch.Tensor))
        }
        cfg = semantic_config(name)
        if cfg is not None and cls.__module__.startswith("pithtrain.models.qwen3_omni"):
            if hasattr(module, "variance_epsilon") and hasattr(cfg, "rms_norm_eps"):
                if module.variance_epsilon != cfg.rms_norm_eps:
                    raise ValueError(f"Runtime RMS epsilon differs from config: {name}")
            if "RotaryEmbedding" in cls.__name__ or (
                cls.__name__ == "Qwen3OmniMoeCode2WavTransformerModel"
            ):
                rope = getattr(cfg, "rope_parameters", None) or {}
                expected_theta = rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
                actual_theta = getattr(module, "theta", getattr(module, "base", None))
                if actual_theta != expected_theta:
                    raise ValueError(f"Runtime RoPE theta differs from config: {name}")
        modules[name] = dict(
            implementation=identifier,
            implementation_sha256=files[source],
            numerical_attributes=attributes,
            config=_canonical_model_value(module.config) if hasattr(module, "config") else None,
        )
    identity = dict(
        schema=3,
        kind="model_config_and_runtime",
        torch=str(torch.__version__),
        config=composite,
        modules=modules,
    )
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return dict(identity=identity, sha256=hashlib.sha256(encoded).hexdigest())
