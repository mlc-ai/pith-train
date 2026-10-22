"""Official-size Thinker text execution, with explicit synthetic weights.

This does not allocate cloud resources. --plan is CPU only. --execute runs on
an existing dedicated H200. Native mode checks a real full-size forward/backward,
not HF parity, distributed training, optimizer state, or released weights.
HF parity mode runs the HF model first, retains gradients on CPU, releases the
HF GPU model, then reconstructs identical native weights and compares every
parameter without reducing the full dimensions or numerical thresholds.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
from pathlib import Path

from tests.acceptance.omni_reference_attention import (
    LEGACY,
    add_reference_attention_argument,
    configure_reference_attention,
    reference_attention_kwargs,
    reference_contract,
)
from tests.acceptance.omni_topology_contract import verify_source, write_json

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
OFFICIAL_CONFIG_SHA256 = "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"
PARAMETER_COUNT = 30_532_646_912
PARAMETER_KEYS = 531
SEQUENCE_LENGTH = 32
INIT_CHUNK = 1 << 24
METRIC_CHUNK = 1 << 20
GRAD_THRESHOLD = 1e-2
LOGIT_THRESHOLD = 1e-3
LOSS_RTOL = LOSS_ATOL = 1e-3


def full_shapes(config):
    """Independent inventory, including every fused expert and both vocabulary matrices."""
    h, v = config["hidden_size"], config["vocab_size"]
    e, m, d = config["num_experts"], config["moe_intermediate_size"], config["head_dim"]
    shapes = {
        "model.embed_tokens.weight": (v, h),
        "model.norm.weight": (h,),
        "lm_head.weight": (v, h),
    }
    for index in range(config["num_hidden_layers"]):
        layer = {
            "input_layernorm.weight": (h,),
            "post_attention_layernorm.weight": (h,),
            "self_attn.q_proj.weight": (config["num_attention_heads"] * d, h),
            "self_attn.k_proj.weight": (config["num_key_value_heads"] * d, h),
            "self_attn.v_proj.weight": (config["num_key_value_heads"] * d, h),
            "self_attn.o_proj.weight": (h, config["num_attention_heads"] * d),
            "self_attn.q_norm.weight": (d,),
            "self_attn.k_norm.weight": (d,),
            "mlp.gate.weight": (e, h),
            "mlp.experts.gate_up_proj": (e, 2 * m, h),
            "mlp.experts.down_proj": (e, h, m),
        }
        shapes.update({f"model.layers.{index}.{name}": shape for name, shape in layer.items()})
    return shapes


def load_config(official_path, repo):
    payload = Path(official_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != OFFICIAL_CONFIG_SHA256:
        raise ValueError("Official configuration differs from the pinned revision artifact")
    official = json.loads(payload)["thinker_config"]["text_config"]
    config = json.loads((Path(repo) / "tests/configs/qwen3_omni_text/full.json").read_text())
    for key, value in config.items():
        if key == "rope_parameters":
            rope = dict(official["rope_scaling"], rope_theta=official["rope_theta"])
            rope.pop("type")
            if value != rope:
                raise ValueError("Full RoPE parameters differ from the official configuration")
        elif official[key] != value:
            raise ValueError(f"Full model configuration differs at {key}")
    shapes = full_shapes(config)
    if (
        len(shapes) != PARAMETER_KEYS
        or sum(math.prod(s) for s in shapes.values()) != PARAMETER_COUNT
    ):
        raise ValueError("Full model dimensions or parameter inventory changed")
    return config, shapes


def resource_estimate():
    size = PARAMETER_COUNT * 2
    return {
        "parameters": PARAMETER_COUNT,
        "parameter_keys": PARAMETER_KEYS,
        "batch": 1,
        "sequence_length": SEQUENCE_LENGTH,
        "bf16_weights_bytes": size,
        "bf16_weights_and_gradients_bytes": 2 * size,
        "bf16_weights_and_fp32_gradients_bytes": 3 * size,
        "additional_gpu_headroom_required_bytes": 12 * 2**30,
        "reference_gradient_host_bytes": size,
        "recommended_host_memory_bytes": 256 * 2**30,
        "suggested_outer_timeout_seconds": {"native": 180, "hf-parity": 600},
        "supports": "single full-size BF16 autograd step with all gradients",
        "does_not_cover": [
            "released weights",
            "DualPipeV FP32 main_grad",
            "optimizer update",
            "distributed topology",
            "checkpoint restore",
            "multimodal encoders",
        ],
        "estimate_only": True,
    }


def canonical(name):
    return name if name.startswith("lm_head.") else "model." + name


def parameter_map(model, native):
    return {
        canonical(name) if native else name: parameter
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }


def check_inventory(parameters, shapes):
    if parameters.keys() != shapes.keys():
        raise AssertionError("Missing or extra full-size parameter names")
    for name, parameter in parameters.items():
        if tuple(parameter.shape) != shapes[name]:
            raise AssertionError(f"Wrong full-size shape: {name}")


def parameter_seed(name, seed=0):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % 2**63


def initialize(parameters, std, seed=0):
    """Name-seeded CPU BF16 chunks give HF/native identical bits independent of module order."""
    import torch

    receipts = {}
    with torch.no_grad():
        for name in sorted(parameters):
            parameter = parameters[name]
            generator = torch.Generator(device="cpu").manual_seed(parameter_seed(name, seed))
            flat = parameter.view(-1)
            digest = hashlib.sha256()
            for start in range(0, flat.numel(), INIT_CHUNK):
                count = min(INIT_CHUNK, flat.numel() - start)
                chunk = torch.empty(count, device="cpu", dtype=torch.bfloat16)
                if parameter.ndim > 1:
                    chunk.normal_(std=std, generator=generator)
                else:
                    chunk.fill_(0 if name.endswith("bias") else 1)
                digest.update(chunk.view(torch.uint8).numpy().tobytes())
                flat[start : start + count].copy_(chunk)
            receipts[name] = {"seed": parameter_seed(name, seed), "sha256": digest.hexdigest()}
    return receipts


def metric(actual, reference):
    """Streaming FP64 normalized squared error, fixed original strict thresholds."""

    if actual.shape != reference.shape:
        raise AssertionError("Numerical comparison shape mismatch")
    x, y = actual.reshape(-1), reference.reshape(-1)
    numerator = denominator = 0.0
    for start in range(0, x.numel(), METRIC_CHUNK):
        a, b = x[start : start + METRIC_CHUNK].double(), y[start : start + METRIC_CHUNK].double()
        numerator += (a - b).square().sum().item()
        denominator += (a.square().sum() + b.square().sum()).item()
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise AssertionError("Nonfinite numerical comparison")
    return 0.0 if denominator == 0 else numerator / denominator


def gradient_diagnostics(actual, reference=None):
    """Chunked CPU FP64 diagnostics, including tiny/zero gradients without omissions."""
    import torch

    tensors = {"actual": actual.detach().cpu().reshape(-1)}
    if reference is not None:
        tensors["reference"] = reference.detach().cpu().reshape(-1)
    result = {}
    for side, values in tensors.items():
        finite, nonzero, square_sum, maximum = True, False, 0.0, 0.0
        for start in range(0, values.numel(), METRIC_CHUNK):
            value = values[start : start + METRIC_CHUNK].double()
            valid = bool(torch.isfinite(value).all())
            finite &= valid
            nonzero |= bool(torch.count_nonzero(value))
            if valid:
                square_sum += value.square().sum().item()
                maximum = max(maximum, value.abs().max().item())
        result.update(
            {
                side + "_finite": finite,
                side + "_nonzero": nonzero,
                side + "_l2": math.sqrt(square_sum) if finite else None,
                side + "_max_abs": maximum if finite else None,
            }
        )
    if reference is not None:
        if result["actual_finite"] and result["reference_finite"]:
            a, b = tensors["actual"], tensors["reference"]
            maximum = 0.0
            for start in range(0, a.numel(), METRIC_CHUNK):
                difference = (
                    a[start : start + METRIC_CHUNK].double()
                    - b[start : start + METRIC_CHUNK].double()
                )
                maximum = max(maximum, difference.abs().max().item())
            result["difference_max_abs"] = maximum
            result["difference"] = metric(tensors["actual"], tensors["reference"])
            result["nse"] = result["difference"]
        else:
            result.update(difference_max_abs=None, difference=None, nse=None)
    return result


def gradient_rows(parameters, shapes, reference=None):
    check_inventory(parameters, shapes)
    rows = []
    for name in sorted(shapes):
        gradient = parameters[name].grad
        row = {"name": name, "shape": list(shapes[name]), "status": "FAIL"}
        if gradient is None:
            row["reason"] = "missing gradient"
        elif tuple(gradient.shape) != shapes[name]:
            row["reason"] = "wrong gradient shape"
        else:
            diagnostics = gradient_diagnostics(
                gradient, None if reference is None else reference[name]
            )
            row.update(
                diagnostics,
                dtype=str(gradient.dtype),
                finite=diagnostics["actual_finite"],
                nonzero=diagnostics["actual_nonzero"],
            )
            if row["finite"] and row["nonzero"]:
                row["status"] = "PASS"
            if reference is not None:
                row["threshold"] = GRAD_THRESHOLD
                if row["difference"] is None or row["difference"] >= GRAD_THRESHOLD:
                    row["status"] = "FAIL"
        rows.append(row)
    return rows


def configure_native(device):
    import torch
    from torch import nn

    from pithtrain.contexts import distributed, training
    from pithtrain.operators.grouped_linear import GroupedLinear

    distributed.pp_size = distributed.ep_size = distributed.cp_size = 1
    distributed.pp_rank = distributed.ep_rank = distributed.cp_rank = 0
    distributed.pp_group = distributed.ep_group = distributed.cp_group = None
    distributed.device = device
    training.fp8, training.Linear, training.GroupedLinear = False, nn.Linear, GroupedLinear
    training.PARAM_DTYPE = torch.bfloat16


def build_model(config, native, device):
    import torch
    from torch import nn

    prior_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            if native:
                from pithtrain.models.qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel

                model = Qwen3OmniMoeThinkerTextModel(config, -1)
            else:
                from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                    Qwen3OmniMoeThinkerTextModel as HFModel,
                )

                config = copy.deepcopy(config)
                config._attn_implementation = "eager"
                model = nn.ModuleDict(
                    {
                        "model": HFModel(config),
                        "lm_head": nn.Linear(config.hidden_size, config.vocab_size, bias=False),
                    }
                )
        model.to_empty(device=device)
        if not native:
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                Qwen3OmniMoeThinkerTextRotaryEmbedding,
            )

            # to_empty does not preserve meta buffers; explicitly reconstruct all RoPE buffers
            # in FP32, exactly as HF's normal constructor does.
            torch.set_default_dtype(torch.float32)
            with torch.device(device):
                model["model"].rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config)
    finally:
        torch.set_default_dtype(prior_dtype)
    return model


def phase(model, native, inputs, labels, shapes, reference=None):
    import torch
    import torch.nn.functional as F

    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if native:
        logits = model.reference_forward(inputs)
    else:
        logits = model["lm_head"](
            model["model"](
                input_ids=inputs, use_cache=False, **reference_attention_kwargs(model)
            ).last_hidden_state
        )
    loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
        raise AssertionError("Nonfinite full-size logits or cross entropy")
    loss.backward()
    torch.cuda.synchronize()
    parameters = parameter_map(model, native)
    result = {
        "mode": "native"
        if native
        else getattr(model, "reference_attention", {"contract": LEGACY})["contract"],
        "cross_entropy": loss.item(),
        "logits_shape": list(logits.shape),
        "logits_finite": True,
        "gradient_parameters": gradient_rows(parameters, shapes, reference),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - start,
    }
    result["status"] = (
        "PASS" if all(row["status"] == "PASS" for row in result["gradient_parameters"]) else "FAIL"
    )
    cpu_logits, cpu_loss = logits.detach().cpu(), loss.detach().cpu()
    del logits, loss
    return result, cpu_logits, cpu_loss


def execute(args, config_data, shapes, source):
    import torch
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTextConfig,
    )

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "source": source,
        "model": MODEL_ID,
        "revision": REVISION,
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "weight_source": args.weight_source,
        "mode": args.mode,
        "reference_attention": {"contract": reference_contract(args)},
        "status": "RUNNING",
        "resource_estimate": resource_estimate(),
        "dtype": "bfloat16",
        "gradients": "autograd BF16; no FP32 main_grad installation",
        "released_weight_validation": "NOT_RUN",
        "dependencies": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "phases": [],
    }

    def save():
        write_json(args.out / "summary.json", report)

    save()
    started = time.monotonic()
    try:
        if args.weight_source != "random":
            report["status"] = "UNSUPPORTED"
            raise NotImplementedError("Released-weight loading is not implemented by this gate")
        if (
            torch.__version__ != "2.13.0+cu130"
            or importlib.metadata.version("transformers") != "5.17.0"
        ):
            raise RuntimeError(
                "This gate requires the reviewed torch2.13.0+cu130 / Transformers5.17.0"
            )
        snapshot = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
        (args.out / "gpu.csv").write_text(snapshot)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("The full-size gate requires exactly one visible dedicated H200")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        required = resource_estimate()["bf16_weights_and_gradients_bytes"] + 12 * 2**30
        if "H200" not in properties.name or free < required:
            raise RuntimeError(f"Dedicated H200 has insufficient free memory: {free} < {required}")
        report["hardware"] = {"name": properties.name, "free_bytes": free, "total_bytes": total}
        save()
        configure_native(device)
        config = Qwen3OmniMoeTextConfig(**config_data)
        generator = torch.Generator(device="cpu").manual_seed(7)
        tokens = torch.randint(0, config.vocab_size, (1, SEQUENCE_LENGTH + 1), generator=generator)
        inputs, labels = (
            tokens[:, :-1].contiguous().to(device),
            tokens[:, 1:].contiguous().to(device),
        )
        reference_gradients = reference_init = reference_logits = reference_loss = None
        if args.mode == "hf-parity":
            model = build_model(config, False, device)
            report["reference_attention"] = configure_reference_attention(
                model, config, reference_contract(args)
            )
            save()
            parameters = parameter_map(model, False)
            check_inventory(parameters, shapes)
            reference_init = initialize(parameters, config.initializer_range)
            result, reference_logits, reference_loss = phase(model, False, inputs, labels, shapes)
            report["phases"].append(result)
            report["weight_fingerprints"] = reference_init
            save()
            if result["status"] != "PASS":
                raise AssertionError(
                    "HF full-size forward/backward failed; native phase not started"
                )
            reference_gradients = {name: p.grad.detach().cpu() for name, p in parameters.items()}
            del parameters, model
            gc.collect()
            torch.cuda.empty_cache()
        model = build_model(config, True, device)
        parameters = parameter_map(model, True)
        check_inventory(parameters, shapes)
        native_init = initialize(parameters, config.initializer_range)
        if reference_init is not None and native_init != reference_init:
            raise AssertionError("Native and HF initial weights differ")
        report["weight_fingerprints"] = native_init
        result, logits, loss = phase(model, True, inputs, labels, shapes, reference_gradients)
        report["phases"].append(result)
        if reference_logits is not None:
            report["parity"] = {
                "logits_error": metric(logits, reference_logits),
                "logits_threshold": LOGIT_THRESHOLD,
                "loss_rtol": LOSS_RTOL,
                "loss_atol": LOSS_ATOL,
                "loss_close": bool(
                    torch.isclose(loss, reference_loss, rtol=LOSS_RTOL, atol=LOSS_ATOL)
                ),
                "all_initial_weights_identical": True,
            }
            if (
                report["parity"]["logits_error"] >= LOGIT_THRESHOLD
                or not report["parity"]["loss_close"]
            ):
                raise AssertionError("Full-size logits or cross entropy parity failed")
        if result["status"] != "PASS":
            raise AssertionError("Full-size native gradient verification failed")
        report["source_after"] = verify_source(
            Path(__file__).resolve().parents[2], args.expected_commit
        )
        report["status"] = (
            "PASS_FULL_SIZE_NATIVE" if args.mode == "native" else "PASS_FULL_SIZE_HF_PARITY"
        )
    except Exception as error:
        if report["status"] != "UNSUPPORTED":
            report["status"] = "FAIL"
        report["error"] = repr(error)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()
    return 0 if report["status"].startswith("PASS_") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_attention_argument(parser)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--weight-source", choices=("random", "released"), required=True)
    parser.add_argument("--mode", choices=("native", "hf-parity"), required=True)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = verify_source(repo, args.expected_commit)
    config, shapes = load_config(args.official_config, repo)
    if args.plan:
        print(
            json.dumps(
                {
                    "source": source,
                    "model": MODEL_ID,
                    "revision": REVISION,
                    "weight_source": args.weight_source,
                    "mode": args.mode,
                    "status": "NOT_RUN" if args.weight_source == "random" else "UNSUPPORTED",
                    "weight_source_supported": args.weight_source == "random",
                    "estimate": resource_estimate(),
                },
                indent=2,
            )
        )
        return 0
    if not args.out:
        parser.error("--execute requires --out")
    return execute(args, config, shapes, source)


if __name__ == "__main__":
    raise SystemExit(main())
