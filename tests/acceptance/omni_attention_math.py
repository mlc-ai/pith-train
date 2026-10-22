"""Independent FP64 attention gate; not whole-model training acceptance.

An external owner must enforce the unchanged 180-second command timeout. This
entrypoint allocates no cloud resources and downloads nothing. Actual native
Q/K/V and cotangents are observed at the registered backward operation; a
separate unobserved repeat must agree exactly on logits/loss/every parameter
gradient. Tiny/full GQA geometry cases use the selected real fixture lengths.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from tests.acceptance.omni_reference_attention import verify_local_fa4
from tests.acceptance.omni_topology_contract import verify_source

OUTPUT_NSE = 1e-3
GRADIENT_NSE = 1e-2
REPO = Path(__file__).resolve().parents[2]
BWD_NAMES = {
    "pithtrain.flash_attn4_mha_bwd.default": "dense",
    "pithtrain.flash_attn4_varlen_bwd.default": "varlen",
}


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def nse(actual, expected):
    if actual.shape != expected.shape:
        raise AssertionError("Attention tensor shape mismatch")
    a, b = actual.detach().double(), expected.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Nonfinite independent attention comparison")
    denominator = (a.square() + b.square()).sum()
    return 0.0 if denominator == 0 else ((a - b).square().sum() / denominator).item()


def mathematical_attention(q, k, v, scale, lengths=None):
    """Independent QK-softmax-V; never calls SDPA, HF, FA4 or a native helper."""
    if lengths is not None:
        if sum(lengths) != q.shape[0] or any(length <= 0 for length in lengths):
            raise ValueError("Invalid packed lengths")
        return torch.cat(
            [
                mathematical_attention(a[None], b[None], c[None], scale)[0]
                for a, b, c in zip(
                    q.split(lengths), k.split(lengths), v.split(lengths), strict=True
                )
            ]
        )
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2]:
        raise ValueError("Expected equal-length BSHD self-attention")
    if q.shape[2] % k.shape[2]:
        raise ValueError("Q heads must be divisible by KV heads")
    queries = q.transpose(1, 2)
    keys = k.repeat_interleave(q.shape[2] // k.shape[2], 2).transpose(1, 2)
    values = v.repeat_interleave(q.shape[2] // v.shape[2], 2).transpose(1, 2)
    scores = (queries @ keys.transpose(-2, -1)) * scale
    mask = torch.ones(q.shape[1], q.shape[1], dtype=torch.bool, device=q.device).triu(1)
    probabilities = scores.masked_fill(mask, -torch.inf).softmax(-1)
    return (probabilities @ values).transpose(1, 2).contiguous()


class ActualBackwardCapture(TorchDispatchMode):
    """Observe an original custom-op invocation and return its exact result."""

    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        kind = BWD_NAMES.get(str(function))
        if kind is not None:
            if kwargs:
                raise RuntimeError("Unexpected FA4 backward kwargs; review capture schema")
            q, k, v, output, _lse, cotangent = args[:6]
            if kind == "dense":
                scale, causal, left, right, deterministic = args[6:]
                lengths = None
            else:
                cu_q, cu_k, max_q, max_k, scale, causal, left, right, deterministic = args[6:]
                if not torch.equal(cu_q, cu_k) or max_q != max_k:
                    raise RuntimeError("Only the declared self-attention contract is supported")
                lengths = (cu_q[1:] - cu_q[:-1]).tolist()
            if not causal or left is not None or right is not None:
                raise RuntimeError(
                    "Independent gate requires original causal/full-window attention"
                )
            self.records.append(
                {
                    "kind": kind,
                    "lengths": lengths,
                    "scale": scale,
                    "deterministic": deterministic,
                    **{
                        name: value.detach().clone()
                        for name, value in zip(
                            ("q", "k", "v", "output", "cotangent", "dq", "dk", "dv"),
                            (q, k, v, output, cotangent, *result),
                            strict=True,
                        )
                    },
                }
            )
        return result


def check_record(record):
    q, k, v = (record[name].detach().double().clone().requires_grad_() for name in ("q", "k", "v"))
    expected = mathematical_attention(q, k, v, record["scale"], record["lengths"])
    gradients = torch.autograd.grad(expected, (q, k, v), record["cotangent"].double())
    values = (expected, *gradients)
    errors = {
        name: nse(record[name], value)
        for name, value in zip(("output", "dq", "dk", "dv"), values, strict=True)
    }
    report = {
        "status": "FAILED",
        "kind": record["kind"],
        "lengths": record["lengths"],
        "shape": {name: list(record[name].shape) for name in ("q", "k", "v")},
        "dtype": str(record["q"].dtype),
        "deterministic": record["deterministic"],
        "softmax_scale": record["scale"],
        "causal": True,
        "input_sha256": {name: tensor_hash(record[name]) for name in ("q", "k", "v", "cotangent")},
        "nse": errors,
        "thresholds": {
            "output": OUTPUT_NSE,
            "dq": GRADIENT_NSE,
            "dk": GRADIENT_NSE,
            "dv": GRADIENT_NSE,
        },
    }
    if errors["output"] < OUTPUT_NSE and all(
        errors[name] < GRADIENT_NSE for name in ("dq", "dk", "dv")
    ):
        report["status"] = "PASSED"
    return report


def fixtures(component, dataset):
    if component == "audio":
        from tests.acceptance import omni_audio_reference, omni_audio_training

        config, original = omni_audio_reference.fixture()
        batches = [
            SimpleNamespace(model_inputs=inputs, labels=labels) for inputs, labels in original
        ]
        real_batches = omni_audio_training.source_inputs(dataset)[2]
        return (
            omni_audio_reference.HFReference,
            config,
            batches,
            real_batches,
            "original_synthetic_audio",
        )
    module = importlib.import_module(f"tests.acceptance.omni_{component}_training")
    _, _, batches, config = module.source_inputs(dataset)
    if isinstance(batches, dict):
        batches = [batches[name] for name in ("text", "image", "audio", "video", "all", "av")]
    return module.HFReference, config, batches, batches, "original_prepared_fixture"


def prepare_native(component, config, reference_class, device):
    from pithtrain.contexts import distributed, training
    from pithtrain.operators.grouped_linear import GroupedLinear

    distributed.device = device
    distributed.pp_size = distributed.cp_size = distributed.ep_size = 1
    distributed.pp_rank = distributed.cp_rank = distributed.ep_rank = 0
    distributed.pp_group = distributed.cp_group = distributed.ep_group = None
    training.fp8, training.Linear, training.GroupedLinear = False, torch.nn.Linear, GroupedLinear
    training.deterministic_attention = False
    suffix = "audio_model" if component == "audio" else component
    module = importlib.import_module(f"pithtrain.models.qwen3_omni_{suffix}")
    classes = {
        "image": "Qwen3OmniMoeThinkerImageModel",
        "audio": "Qwen3OmniMoeThinkerAudioModel",
        "video": "Qwen3OmniMoeThinkerVideoModel",
        "multimodal": "Qwen3OmniMoeThinkerMultimodalModel",
    }
    reference = reference_class(config, torch.bfloat16)
    with torch.device(device):
        native = getattr(module, classes[component])(config, -1).to(torch.bfloat16)
    actual = {
        name if name.startswith(("visual.", "audio_tower.", "lm_head.")) else "model." + name: value
        for name, value in native.named_parameters()
    }
    expected = dict(reference.named_parameters())
    if actual.keys() != expected.keys():
        raise AssertionError("Complete original parameter mapping is required")
    with torch.no_grad():
        for name, value in actual.items():
            value.copy_(expected[name])
    if any(layer.mlp.gate.router_replay is not None for layer in native.layers.values()):
        raise AssertionError("Free original routing is required")
    return native


def execute_original(native, batches, device, observer=None, *, audio_input_gradients=False):
    native.zero_grad(set_to_none=True)
    losses, logits, feature_inputs = [], [], []
    count = sum(int((batch.labels != -100).sum()) for batch in batches)
    for batch in batches:
        inputs = {name: value.to(device) for name, value in batch.model_inputs.items()}
        if audio_input_gradients:
            inputs["input_features"] = inputs["input_features"].detach().requires_grad_()
            feature_inputs.append(inputs["input_features"])
        output = native.reference_forward(inputs["input_ids"], model_context=inputs)
        logits.append(output.detach().cpu())
        losses.append(
            F.cross_entropy(
                output.float().flatten(0, 1), batch.labels.to(device).flatten(), reduction="sum"
            )
            / count
        )
    loss = sum(losses)
    with observer if observer is not None else contextlib.nullcontext():
        loss.backward()
    gradients = {}
    for name, parameter in native.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Missing/nonfinite native control gradient: {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    feature_gradients = []
    for value in feature_inputs:
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise AssertionError("Missing/nonfinite original audio input gradient")
        feature_gradients.append(value.grad.detach().cpu().clone())
    return logits, loss.detach().cpu(), gradients, feature_gradients


class CaptureTransparencyPreconditionError(AssertionError):
    """Repeatability was not established; this is not a mathematical failure."""


def exact_observer_control(before, after):
    left_logits, left_loss, left_gradients, left_features = before
    right_logits, right_loss, right_gradients, right_features = after
    if left_gradients.keys() != right_gradients.keys():
        raise CaptureTransparencyPreconditionError("Observer control parameter coverage differs")
    if not torch.equal(left_loss, right_loss) or any(
        not torch.equal(left, right) for left, right in zip(left_logits, right_logits, strict=True)
    ):
        raise CaptureTransparencyPreconditionError("Observer/unobserved logits or objective differ")
    for name, value in left_gradients.items():
        if not torch.equal(value, right_gradients[name]):
            raise CaptureTransparencyPreconditionError(
                f"Observer/unobserved parameter gradient differs: {name}"
            )
    for left, right in zip(left_features, right_features, strict=True):
        if not torch.equal(left, right):
            raise CaptureTransparencyPreconditionError(
                "Observer/unobserved audio input gradient differs"
            )
    return {
        "status": "EXACT",
        "parameter_gradients": len(left_gradients),
        "audio_input_gradients": len(left_features),
        "logits": len(left_logits),
        "objective": "EXACT",
    }


def validate_geometry_lengths(lengths):
    # The unchanged prepared video/mixed recipe admits up to 2048 tokens.
    # Real CPU-prepared video samples have 386/1631; mixed includes 1260.
    # Do not truncate those samples or substitute unrelated long sequences.
    if len(lengths) != 2 or min(lengths) < 1 or max(lengths) > 2048:
        raise ValueError("Use two actual prepared fixture lengths (1..2048)")


def geometry_records(lengths, device):
    from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func

    validate_geometry_lengths(lengths)
    for size in ("tiny", "full"):
        config = json.loads((REPO / f"tests/configs/qwen3_omni_text/{size}.json").read_text())
        hq, hk, dim = (
            config[name] for name in ("num_attention_heads", "num_key_value_heads", "head_dim")
        )
        for kind, sequence_lengths in (
            ("dense", [lengths[0]]),
            ("dense", [lengths[1]]),
            ("varlen", lengths),
        ):
            for deterministic in (False, True):
                generator = torch.Generator(device=device).manual_seed(739)
                prefix = (1, sequence_lengths[0]) if kind == "dense" else (sum(sequence_lengths),)
                q, k, v = (
                    torch.randn(
                        *prefix,
                        heads,
                        dim,
                        generator=generator,
                        device=device,
                        dtype=torch.bfloat16,
                    ).requires_grad_()
                    for heads in (hq, hk, hk)
                )
                cotangent = torch.randn(q.shape, generator=generator, device=device, dtype=q.dtype)
                scale = dim**-0.5
                if kind == "dense":
                    output = flash_attn_func(
                        q, k, v, scale, causal=True, deterministic=deterministic
                    )
                else:
                    cu = torch.tensor(
                        [0, sequence_lengths[0], sum(sequence_lengths)],
                        device=device,
                        dtype=torch.int32,
                    )
                    output = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu,
                        max(sequence_lengths),
                        scale,
                        causal=True,
                        deterministic=deterministic,
                    )
                gradients = torch.autograd.grad(output, (q, k, v), cotangent)
                yield {
                    "size": size,
                    "kind": kind,
                    "lengths": None if kind == "dense" else sequence_lengths,
                    "scale": scale,
                    "deterministic": deterministic,
                    **dict(
                        zip(
                            ("q", "k", "v", "output", "cotangent", "dq", "dk", "dv"),
                            (q, k, v, output, cotangent, *gradients),
                            strict=True,
                        )
                    ),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component", choices=("image", "audio", "video", "multimodal"), required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new evidence file")
    if os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") != "1":
        raise RuntimeError("Explicit existing GPU gate authorization is required")
    source = verify_source(REPO, args.expected_commit)
    runtime = verify_local_fa4()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    started = time.monotonic()
    report = {
        "status": "RUNNING",
        "acceptance_scope": "independent_attention_math_v1",
        "model_training_acceptance": "NOT_ESTABLISHED",
        "mathematical_comparison": "NOT_RUN",
        "component": args.component,
        "source": source,
        "runtime": runtime,
        "actual_captures": [],
        "geometry_cases": [],
        "outer_timeout_seconds": 180,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        reference_class, config, batches, real_batches, fixture_kind = fixtures(
            args.component, args.dataset
        )
        report["fixture_kind"] = fixture_kind
        report["fixture_sha256"] = [
            {
                **{name: tensor_hash(value) for name, value in batch.model_inputs.items()},
                "labels": tensor_hash(batch.labels),
            }
            for batch in batches
        ]
        report["dataset_checksums_sha256"] = hashlib.sha256(
            (args.dataset / "checksums.sha256").read_bytes()
        ).hexdigest()
        native = prepare_native(args.component, config, reference_class, device)
        before = execute_original(
            native, batches, device, audio_input_gradients=args.component == "audio"
        )
        observer = ActualBackwardCapture()
        after = execute_original(
            native, batches, device, observer, audio_input_gradients=args.component == "audio"
        )
        report["observer_control"] = exact_observer_control(before, after)
        del before, after, native
        expected_count = config.text_config.num_hidden_layers * len(batches)
        if len(observer.records) != expected_count:
            raise AssertionError(
                f"Actual attention capture count: {len(observer.records)} != {expected_count}"
            )
        report["expected_actual_captures"] = expected_count
        report["mathematical_comparison"] = "RUNNING"
        for record in observer.records:
            report["actual_captures"].append(check_record(record))
            save()
        lengths = [int(batch.model_inputs["input_ids"].shape[1]) for batch in real_batches[:2]]
        report["geometry_fixture_lengths"] = lengths
        report["full_geometry_fp64_scores_bytes"] = {
            "largest_dense": 32 * max(lengths) ** 2 * 8,
            "packed_documents_total": 32 * sum(length**2 for length in lengths) * 8,
            "scope": "score tensors only; autograd intermediates additionally required",
        }
        for record in geometry_records(lengths, device):
            row = check_record(record)
            row.update(size=record["size"], input_kind="synthetic_qkv_at_actual_fixture_geometry")
            report["geometry_cases"].append(row)
            save()
        if len(report["geometry_cases"]) != 12:
            raise AssertionError(
                "All tiny/full dense/varlen and both backward policies are required"
            )
        if any(
            row["status"] != "PASSED"
            for row in report["actual_captures"] + report["geometry_cases"]
        ):
            raise AssertionError("Independent FP64 mathematical attention gate failed")
        report["source_after"] = verify_source(REPO, args.expected_commit)
        report["status"] = "PASSED"
        report["mathematical_comparison"] = "PASSED"
    except CaptureTransparencyPreconditionError as error:
        report.update(
            status="NOT_RUN_CAPTURE_TRANSPARENCY_PRECONDITION",
            mathematical_comparison="NOT_RUN",
            failure_phase="capture_transparency_precondition",
            error=f"{type(error).__name__}: {error}",
            interpretation=(
                "Native repeatability was not established. Natural nondeterminism and "
                "observer interference are not distinguished by this result. This does "
                "not establish a mathematical attention failure or change model criteria."
            ),
        )
        raise
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()


if __name__ == "__main__":
    main()
