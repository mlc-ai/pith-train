"""Full official-size image/audio/video numerical gates on an existing H200.

Uses the existing real prepared-bundle fixtures and HF forward adapters, while
replacing their returned tiny configs with the pinned official full config.
No allocation, download, optimizer update, or released-weight loading occurs.
Run each modality in a fresh process under an external 600-second hard timeout.
"""

import argparse
import copy
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import subprocess
import time
from pathlib import Path

from tests.acceptance import omni_full_size as text_gate
from tests.acceptance.omni_reference_attention import (
    add_reference_attention_argument,
    configure_reference_attention,
    reference_contract,
)
from tests.acceptance.omni_topology_contract import verify_source, write_json

MODALITIES = ("image", "audio", "video")
ENCODER_COUNTS = {
    "image": (538_631_408, 351),
    "video": (538_631_408, 351),
    "audio": (647_927_168, 525),
}
MAX_INPUT_TOKENS = 512


def official_config(path, repo):
    # Also verifies every field of the full text config against the pinned official bytes.
    text_gate.load_config(path, repo)
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerConfig,
    )

    config = Qwen3OmniMoeThinkerConfig(**json.loads(Path(path).read_text())["thinker_config"])
    for component in (config.text_config, config.audio_config, config.vision_config):
        component._attn_implementation = "eager"
    return config


def estimate(modality):
    encoder, tensors = ENCODER_COUNTS[modality]
    parameters = text_gate.PARAMETER_COUNT + encoder
    return {
        "parameters": parameters,
        "parameter_keys": text_gate.PARAMETER_KEYS + tensors,
        "bf16_weights_and_gradients_bytes": parameters * 4,
        "reference_gradient_host_bytes": parameters * 2,
        "minimum_free_gpu_bytes": parameters * 4 + 12 * 2**30,
        "recommended_host_memory_bytes": 256 * 2**30,
        "maximum_input_tokens": MAX_INPUT_TOKENS,
        "outer_timeout_seconds": 600,
        "optimizer_update": "NOT_RUN",
        "fp32_main_grad": "NOT_RUN",
        "released_weights": "NOT_RUN",
        "estimate_only": True,
        "headroom_is_an_admission_minimum_not_a_measured_peak": True,
        "fp32_adam_update": "NOT_RUN; full resident state exceeds a single H200",
    }


def reference_type(modality):
    name = "omni_audio_reference" if modality == "audio" else f"omni_{modality}_training"
    return importlib.import_module(f"tests.acceptance.{name}").HFReference


def build_reference(config, modality):
    """Reuse the established HF forward method without its eager CPU weight allocation."""
    import torch
    from torch import nn
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeThinkerTextModel,
        Qwen3OmniMoeVisionEncoder,
    )

    class FullReference(nn.Module):
        forward = reference_type(modality).forward

        def __init__(self):
            super().__init__()
            self.config = copy.deepcopy(config)
            self.model = Qwen3OmniMoeThinkerTextModel(self.config.text_config)
            self.lm_head = nn.Linear(
                config.text_config.hidden_size, config.text_config.vocab_size, bias=False
            )
            if modality == "audio":
                self.audio_tower = Qwen3OmniMoeAudioEncoder(self.config.audio_config)
            else:
                self.visual = Qwen3OmniMoeVisionEncoder(self.config.vision_config)

    with torch.device("meta"):
        return FullReference()


def full_inventory(config, modality):
    reference = build_reference(config, modality)
    shapes = {name: tuple(value.shape) for name, value in reference.named_parameters()}
    expected = estimate(modality)
    if (
        len(shapes) != expected["parameter_keys"]
        or sum(math.prod(shape) for shape in shapes.values()) != expected["parameters"]
    ):
        raise AssertionError("Official full media inventory count changed")
    text_shapes = text_gate.full_shapes(
        json.loads(
            (
                Path(__file__).resolve().parents[2] / "tests/configs/qwen3_omni_text/full.json"
            ).read_text()
        )
    )
    if any(shapes.get(name) != shape for name, shape in text_shapes.items()):
        raise AssertionError("Media reference downscaled the full text decoder")
    return shapes


def parameter_map(model, native):
    return {
        name
        if not native or name.startswith(("visual.", "audio_tower.", "lm_head."))
        else "model." + name: value
        for name, value in model.named_parameters(remove_duplicate=False)
    }


def restore_reference_buffers(model, config, modality, device):
    import torch
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextRotaryEmbedding,
        Qwen3OmniMoeVisionRotaryEmbedding,
    )

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device(device):
            model.model.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config.text_config)
            if modality == "audio":
                cls = type(model.audio_tower.positional_embedding)
                model.audio_tower.positional_embedding = cls(
                    config.audio_config.max_source_positions, config.audio_config.d_model
                )
            else:
                model.visual.rotary_pos_emb = Qwen3OmniMoeVisionRotaryEmbedding(
                    config.vision_config
                )
    finally:
        torch.set_default_dtype(previous)
    if any(value.is_meta or not bool(torch.isfinite(value).all()) for value in model.buffers()):
        raise AssertionError("Uninitialized HF position/rotary buffer")


def build_model(config, modality, native, device):
    import torch

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        if native:
            classes = {
                "image": ("qwen3_omni_image", "Qwen3OmniMoeThinkerImageModel"),
                "video": ("qwen3_omni_video", "Qwen3OmniMoeThinkerVideoModel"),
                "audio": ("qwen3_omni_audio_model", "Qwen3OmniMoeThinkerAudioModel"),
            }
            module, name = classes[modality]
            cls = getattr(importlib.import_module(f"pithtrain.models.{module}"), name)
            with torch.device("meta"):
                model = cls(config, -1)
        else:
            model = build_reference(config, modality)
        model.to_empty(device=device)
        if not native:
            restore_reference_buffers(model, config, modality, device)
        return model
    finally:
        torch.set_default_dtype(previous)


def select_fixture(dataset, modality, config):
    """Real samples are processed before the existing helper mutates its own tiny config."""
    module = importlib.import_module(f"tests.acceptance.omni_{modality}_training")
    bundle, samples, batches, _tiny_config = module.source_inputs(Path(dataset))
    if bundle["recipe"]["revision"] != text_gate.REVISION:
        raise ValueError("Fixture processor revision differs from official config")
    # Fixed existing first sample: never select by observed loss/gradient or retry on another.
    index, batch = 0, batches[0]
    if batch.model_inputs["input_ids"].shape[-1] > MAX_INPUT_TOKENS:
        raise ValueError("The fixed first real sample exceeds the declared 512-token bound")
    validators = {
        "image": ("qwen3_omni_image_inputs", "validate_image_inputs"),
        "audio": ("qwen3_omni_audio_inputs", "validate_audio_inputs"),
        "video": ("qwen3_omni_video_inputs", "validate_video_inputs"),
    }
    source, symbol = validators[modality]
    validator = getattr(importlib.import_module(f"pithtrain.models.{source}"), symbol)
    positions = validator(batch.model_inputs["input_ids"], batch.model_inputs, config)
    if not bool((batch.labels != -100).any()):
        raise AssertionError("The representative batch has no prediction targets")
    report = {
        "sample_id": samples[index]["id"],
        "selection": "fixed first existing sample; no crop, replacement, or numeric selection",
        "processor_revision": bundle["recipe"]["revision"],
        "input_tokens": batch.model_inputs["input_ids"].shape[-1],
        "valid_targets": int((batch.labels != -100).sum()),
        "input_shapes": {
            name: list(value.shape)
            for name, value in batch.model_inputs.items()
            if hasattr(value, "shape")
        },
        "position_ids": positions.tolist(),
        "bundle_checksums_sha256": hashlib.sha256(
            (Path(dataset) / "checksums.sha256").read_bytes()
        ).hexdigest(),
    }
    return batch, report


def move_inputs(batch, device):
    import torch

    inputs = {}
    for name, value in batch.model_inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().clone().to(device)
            if name in ("pixel_values", "pixel_values_videos", "input_features"):
                value.requires_grad_()
        inputs[name] = value
    return inputs, batch.labels.to(device)


def compare_gradients(parameters, shapes, expected=None):
    text_gate.check_inventory(parameters, shapes)
    rows = []
    for name in sorted(shapes):
        gradient = parameters[name].grad
        row = {
            "name": name,
            "shape": list(shapes[name]),
            "status": "FAIL",
            "threshold": text_gate.GRAD_THRESHOLD,
            "compared": False,
        }
        if gradient is None:
            row["reason"] = "missing gradient"
        elif tuple(gradient.shape) != shapes[name]:
            row["reason"] = "wrong gradient shape"
        else:
            diagnostics = text_gate.gradient_diagnostics(
                gradient, None if expected is None else expected[name]
            )
            row.update(
                diagnostics,
                dtype=str(gradient.dtype),
                finite=diagnostics["actual_finite"],
                nonzero=diagnostics["actual_nonzero"],
            )
            if row["finite"]:
                row["status"] = "PASS"
                if expected is not None:
                    row["compared"] = row["difference"] is not None
                    if row["difference"] is None or row["difference"] >= text_gate.GRAD_THRESHOLD:
                        row["status"] = "FAIL"
        rows.append(row)
    # Individual mathematically-zero gradients are still compared (including 0 vs 0).
    # Each full trainable component must actually participate in the objective.
    for prefix in ("model.", "lm_head.", "audio_tower.", "visual."):
        members = [row for row in rows if row["name"].startswith(prefix)]
        if members and not any(row.get("nonzero") for row in members):
            for row in members:
                row.update(
                    status="FAIL", reason=f"No nonzero gradient in trainable component {prefix}"
                )
    return rows


def run_phase(model, native, inputs, labels, shapes, phase_report, save, expected=None):
    import torch
    import torch.nn.functional as F

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    phase_report["forward"] = "RUNNING"
    save()
    logits = (
        model.reference_forward(inputs["input_ids"], model_context=inputs)
        if native
        else model(inputs)
    )
    loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
    forward_finite = bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss))
    torch.cuda.synchronize()
    phase_report.update(
        forward="PASS" if forward_finite else "FAIL",
        backward="NOT_RUN",
        update="NOT_RUN",
        cross_entropy=loss.item() if forward_finite else None,
        logits_shape=list(logits.shape),
        forward_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        forward_seconds=time.monotonic() - started,
    )
    save()
    phase_report["backward"] = "RUNNING"
    save()
    loss.backward()
    torch.cuda.synchronize()
    rows = compare_gradients(parameter_map(model, native), shapes, expected)
    phase_report.update(
        backward="PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        gradient_parameters=rows,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.monotonic() - started,
    )
    save()
    feature_gradients = {}
    for name in ("pixel_values", "pixel_values_videos", "input_features"):
        if name in inputs:
            gradient = inputs[name].grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise AssertionError(f"Missing or nonfinite input-feature gradient: {name}")
            feature_gradients[name] = gradient.detach().cpu()
    return logits.detach().cpu(), loss.detach().cpu(), feature_gradients


def execute(args, config, shapes, source, batch, fixture):
    import torch

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "source": source,
        "model": text_gate.MODEL_ID,
        "revision": text_gate.REVISION,
        "official_config_sha256": text_gate.OFFICIAL_CONFIG_SHA256,
        "modality": args.modality,
        "weight_source": args.weight_source,
        "status": "RUNNING",
        "dtype": "bfloat16",
        "fixture": fixture,
        "estimate": estimate(args.modality),
        "configuration": config.to_dict(),
        "reference_attention": {"contract": reference_contract(args)},
        "phases": [],
        "update": "NOT_RUN",
        "fp32_main_grad": "NOT_RUN",
        "released_weight_validation": "NOT_RUN",
        "dependencies": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
    }

    def save():
        write_json(args.out / "summary.json", report)

    started = time.monotonic()
    save()
    try:
        if (
            torch.__version__ != "2.13.0+cu130"
            or importlib.metadata.version("transformers") != "5.17.0"
        ):
            raise RuntimeError("Requires reviewed torch2.13.0+cu130 / Transformers5.17")
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
            raise RuntimeError("Requires exactly one visible dedicated H200")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        report["hardware"] = {"name": properties.name, "free_bytes": free, "total_bytes": total}
        save()
        if (
            "H200" not in properties.name
            or free < estimate(args.modality)["minimum_free_gpu_bytes"]
        ):
            raise RuntimeError(
                "Dedicated H200 free memory is below declared model+gradient+headroom"
            )
        text_gate.configure_native(device)
        model = build_model(config, args.modality, False, device)
        report["reference_attention"] = configure_reference_attention(
            model, config, reference_contract(args)
        )
        save()
        parameters = parameter_map(model, False)
        text_gate.check_inventory(parameters, shapes)
        reference_init = text_gate.initialize(parameters, config.text_config.initializer_range)
        inputs, labels = move_inputs(batch, device)
        phase_report = {
            "implementation": "HF",
            "forward": "NOT_RUN",
            "backward": "NOT_RUN",
            "update": "NOT_RUN",
        }
        report["phases"].append(phase_report)
        reference_logits, reference_loss, reference_inputs = run_phase(
            model, False, inputs, labels, shapes, phase_report, save
        )
        if phase_report["forward"] != "PASS" or phase_report["backward"] != "PASS":
            raise AssertionError(
                "Full-size HF media forward/backward failed; all gradient rows saved"
            )
        reference_gradients = {name: p.grad.detach().cpu() for name, p in parameters.items()}
        del parameters, model, inputs, labels
        gc.collect()
        torch.cuda.empty_cache()
        model = build_model(config, args.modality, True, device)
        parameters = parameter_map(model, True)
        text_gate.check_inventory(parameters, shapes)
        native_init = text_gate.initialize(parameters, config.text_config.initializer_range)
        if native_init != reference_init:
            raise AssertionError("HF/native full-size media weight bytes differ")
        report["weight_fingerprints"] = native_init
        inputs, labels = move_inputs(batch, device)
        phase_report = {
            "implementation": "native",
            "forward": "NOT_RUN",
            "backward": "NOT_RUN",
            "update": "NOT_RUN",
        }
        report["phases"].append(phase_report)
        logits, loss, input_gradients = run_phase(
            model, True, inputs, labels, shapes, phase_report, save, reference_gradients
        )
        report["parity"] = {
            "logit_error": text_gate.metric(logits, reference_logits),
            "logit_threshold": text_gate.LOGIT_THRESHOLD,
            "loss_close": bool(
                torch.isclose(
                    loss, reference_loss, rtol=text_gate.LOSS_RTOL, atol=text_gate.LOSS_ATOL
                )
            ),
            "loss_rtol": text_gate.LOSS_RTOL,
            "loss_atol": text_gate.LOSS_ATOL,
            "input_gradient_errors": {
                name: text_gate.metric(value, reference_inputs[name])
                for name, value in input_gradients.items()
            },
            "input_gradient_threshold": text_gate.GRAD_THRESHOLD,
        }
        parity = report["parity"]
        if (
            parity["logit_error"] >= text_gate.LOGIT_THRESHOLD
            or not parity["loss_close"]
            or any(
                value >= text_gate.GRAD_THRESHOLD
                for value in parity["input_gradient_errors"].values()
            )
            or phase_report["forward"] != "PASS"
            or phase_report["backward"] != "PASS"
        ):
            raise AssertionError("Full-size media logits/loss/gradient parity failed")
        report["source_after"] = verify_source(
            Path(__file__).resolve().parents[2], args.expected_commit
        )
        report["status"] = "PASS_FULL_SIZE_FORWARD_BACKWARD"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        for phase_report in report["phases"]:
            for stage in ("forward", "backward"):
                if phase_report[stage] == "RUNNING":
                    phase_report[stage] = "FAIL"
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()
    return 0 if report["status"] == "PASS_FULL_SIZE_FORWARD_BACKWARD" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_attention_argument(parser)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--weight-source", choices=("random",), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = verify_source(repo, args.expected_commit)
    config = official_config(args.official_config, repo)
    shapes = full_inventory(config, args.modality)
    batch = fixture = None
    if args.dataset:
        batch, fixture = select_fixture(args.dataset, args.modality, config)
    if args.plan:
        print(
            json.dumps(
                {
                    "source": source,
                    "modality": args.modality,
                    "status": "NOT_RUN",
                    "weight_source": args.weight_source,
                    "estimate": estimate(args.modality),
                    "fixture": fixture,
                },
                indent=2,
            )
        )
        return 0
    if not args.out or batch is None:
        parser.error("--execute requires dataset and a new output directory")
    return execute(args, config, shapes, source, batch, fixture)


if __name__ == "__main__":
    raise SystemExit(main())
