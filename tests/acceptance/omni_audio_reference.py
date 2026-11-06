"""Opt-in H100 native audio/text reference parity, using tiny synthetic inputs.

No model downloads, scheduler submission, or dataset preparation occur here.
This is the HF reference gate; distributed DualPipeV and full checkpoint gates
must additionally pass before audio training is considered accepted.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HFAudio,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFPositions,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel as HFText,
)

from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from tests.acceptance.omni_image_training import FP32MasterAdamW
from tests.acceptance.omni_reference_attention import (
    LEGACY,
    add_reference_attention_argument,
    configure_reference_attention,
    reference_attention_kwargs,
    reference_contract,
)

REPO = Path(__file__).resolve().parents[2]


def fixture():
    raw = json.loads((REPO / "tests/configs/qwen3_omni_text/tiny.json").read_text())
    text = Qwen3OmniMoeTextConfig(**raw)
    config = Qwen3OmniMoeThinkerConfig(
        text_config=text,
        audio_config=Qwen3OmniMoeAudioEncoderConfig(
            num_mel_bins=128,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            d_model=32,
            output_dim=text.hidden_size,
            downsample_hidden_size=4,
            n_window=50,
            n_window_infer=200,
            conv_chunksize=2,
            max_source_positions=13,
        ),
        audio_token_id=151675,
        audio_start_token_id=151669,
        audio_end_token_id=151670,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
    )
    generator = torch.Generator().manual_seed(825)
    batches = []
    for lengths in ([101, 17], [201]):
        feature_lens = torch.tensor(lengths)
        width = max(lengths)
        tokens = [7, 8]
        for index, length in enumerate(lengths):
            count = length // 100 * 13 + (length % 100 + 7) // 8
            tokens += [config.audio_start_token_id, *([config.audio_token_id] * count)]
            tokens += [config.audio_end_token_id, 9 + index]
        tokens += [21, 22, 23, 24]
        ids = torch.tensor([tokens])
        labels = torch.full_like(ids, -100)
        labels[:, -4:-1] = ids[:, -3:]
        batches.append(
            (
                dict(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    input_features=torch.randn(len(lengths), 128, width, generator=generator),
                    feature_attention_mask=(
                        torch.arange(width)[None] < feature_lens[:, None]
                    ).long(),
                ),
                labels,
            )
        )
    return config, batches


def error_ratio(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    actual, expected = actual.double(), expected.double()
    denominator = (actual.square() + expected.square()).sum()
    return 0.0 if denominator == 0 else ((actual - expected).square().sum() / denominator).item()


class GradientParityError(AssertionError):
    def __init__(self, diagnostics, context):
        self.diagnostics, self.context = diagnostics, context
        super().__init__(
            {name: row for name, row in diagnostics.items() if row["status"] != "PASSED"}
        )


def check_gradients(actual, expected, context=None):
    assert actual.keys() == expected.keys()
    diagnostics = {}
    for name, parameter in actual.items():
        left, right = parameter.grad, expected[name].grad
        row = {"status": "FAILED", "threshold": 1e-2}
        diagnostics[name] = row
        if left is None or right is None:
            row.update(actual_missing=left is None, reference_missing=right is None)
            continue
        row.update(actual_shape=list(left.shape), reference_shape=list(right.shape))
        if (
            left.shape != right.shape
            or not torch.isfinite(left).all()
            or not torch.isfinite(right).all()
        ):
            row["reason"] = "shape mismatch or nonfinite gradients"
            continue
        left, right = left.double(), right.double()
        row.update(
            nse=error_ratio(left, right),
            actual_l2=left.norm().item(),
            reference_l2=right.norm().item(),
            actual_max_abs=left.abs().max().item() if left.numel() else 0.0,
            reference_max_abs=right.abs().max().item() if right.numel() else 0.0,
            difference_max_abs=(left - right).abs().max().item() if left.numel() else 0.0,
        )
        if row["nse"] < 1e-2:
            row["status"] = "PASSED"
    if any(row["status"] != "PASSED" for row in diagnostics.values()):
        raise GradientParityError(diagnostics, context)
    return {name: row["nse"] for name, row in diagnostics.items()}


def encoder_gate(config, device):
    config = copy.deepcopy(config.audio_config)
    config._attn_implementation = "eager"
    torch.manual_seed(739)
    reference = HFAudio(config).to(device=device, dtype=torch.bfloat16)
    native = Qwen3OmniMoeAudioEncoder(config).to(device=device, dtype=torch.bfloat16)
    native.load_state_dict(reference.state_dict(), strict=True)
    reports = []
    for values in ([1], [99], [100], [101], [201, 17], [300, 100]):
        lengths = torch.tensor(values, device=device)
        features = torch.randn(128, sum(values), device=device, requires_grad=True)
        expected_features = features.detach().clone().requires_grad_()
        output = native(features, lengths).last_hidden_state
        expected = reference(expected_features, feature_lens=lengths).last_hidden_state
        output_error = error_ratio(output, expected)
        assert output_error < 1e-3, (values, output_error)
        probe = torch.randn_like(output)
        (output.float() * probe).sum().backward()
        (expected.float() * probe).sum().backward()
        gradients = check_gradients(
            dict(native.named_parameters()), dict(reference.named_parameters())
        )
        input_error = error_ratio(features.grad, expected_features.grad)
        assert input_error < 1e-2, (values, input_error)
        reports.append(
            dict(
                lengths=values,
                output_error=output_error,
                gradient_errors=gradients,
                input_error=input_error,
            )
        )
        native.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    return reports


class HFReference(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.config.text_config._attn_implementation = "eager"
        self.config.audio_config._attn_implementation = "eager"
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with torch.device("cpu"), torch.random.fork_rng(devices=[]):
                torch.manual_seed(739)
                self.audio_tower = HFAudio(self.config.audio_config)
                self.model = HFText(self.config.text_config)
                self.lm_head = nn.Linear(
                    config.text_config.hidden_size, config.text_config.vocab_size, bias=False
                )
        finally:
            torch.set_default_dtype(previous)

    def forward(self, inputs):
        ids = inputs["input_ids"]
        mask = inputs["feature_attention_mask"].bool()
        flat = inputs["input_features"].permute(0, 2, 1)[mask].T
        audio = self.audio_tower(flat, feature_lens=mask.long().sum(-1)).last_hidden_state
        embeddings = self.model.embed_tokens(ids)
        embeddings = embeddings.masked_scatter(
            (ids == self.config.audio_token_id)[..., None], audio.to(embeddings.dtype)
        )
        owner = SimpleNamespace(
            config=self.config, spatial_merge_size=self.config.vision_config.spatial_merge_size
        )
        positions, _ = HFPositions.get_rope_index(
            owner, ids, attention_mask=inputs["attention_mask"]
        )
        hidden = self.model(
            inputs_embeds=embeddings,
            attention_mask=inputs["attention_mask"],
            position_ids=positions,
            use_cache=False,
            **reference_attention_kwargs(self),
        ).last_hidden_state
        return self.lm_head(hidden)


def text_gate(config, batches, device, *, reference_attention=LEGACY, attention_report=None):
    # Optional CUDA kernels are loaded only after the explicit GPU gate.
    from pithtrain.contexts import distributed, training
    from pithtrain.models.qwen3_omni_audio_model import Qwen3OmniMoeThinkerAudioModel
    from pithtrain.operators.grouped_linear import GroupedLinear
    from tests.acceptance.omni_audio_training import training_groups

    distributed.device = device
    distributed.pp_size = distributed.cp_size = distributed.ep_size = 1
    distributed.pp_rank = distributed.cp_rank = distributed.ep_rank = 0
    distributed.pp_group = distributed.cp_group = distributed.ep_group = None
    training.fp8, training.Linear, training.GroupedLinear = False, nn.Linear, GroupedLinear
    reference = HFReference(config).to(device)
    attention = configure_reference_attention(reference, config, reference_attention)
    if attention_report is not None:
        attention_report.update(attention)
    with torch.device(device):
        native = Qwen3OmniMoeThinkerAudioModel(config, -1).to(torch.bfloat16)
    expected = dict(reference.named_parameters())
    actual = {
        name if name.startswith(("audio_tower.", "lm_head.")) else "model." + name: parameter
        for name, parameter in native.named_parameters()
    }
    assert actual.keys() == expected.keys()
    with torch.no_grad():
        for name, parameter in actual.items():
            parameter.copy_(expected[name])
    optimizer = FP32MasterAdamW(actual, lr=1e-4, weight_decay=0)
    expected_optimizer = FP32MasterAdamW(expected, lr=1e-4, weight_decay=0)
    groups = training_groups(config)
    reports = []
    for step in range(4):
        losses, expected_losses, logits_errors, feature_pairs = [], [], [], []
        count = sum(int((labels != -100).sum()) for _, labels in batches)
        for inputs, labels in batches:
            inputs = {name: value.to(device) for name, value in inputs.items()}
            inputs["input_features"].requires_grad_()
            ref_inputs = {name: value.detach().clone() for name, value in inputs.items()}
            ref_inputs["input_features"].requires_grad_()
            feature_pairs.append((inputs["input_features"], ref_inputs["input_features"]))
            labels = labels.to(device)
            logits = native.reference_forward(inputs["input_ids"], model_context=inputs)
            target = reference(ref_inputs)
            logits_errors.append(error_ratio(logits, target))
            assert logits_errors[-1] < 1e-3, logits_errors[-1]
            losses.append(
                F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
                / count
            )
            expected_losses.append(
                F.cross_entropy(target.float().flatten(0, 1), labels.flatten(), reduction="sum")
                / count
            )
        loss, expected_loss = sum(losses), sum(expected_losses)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)
        loss.backward()
        expected_loss.backward()
        gradients = check_gradients(
            actual,
            expected,
            context=dict(
                component="audio_text",
                step=step,
                loss=loss.item(),
                expected_loss=expected_loss.item(),
                logits_errors=logits_errors,
            ),
        )
        feature_errors = [error_ratio(a.grad, b.grad) for a, b in feature_pairs]
        assert max(feature_errors) < 1e-2, feature_errors
        before = {name: p.detach().clone() for name, p in optimizer.master_parameters.items()}
        compute_before = {name: p.detach().clone() for name, p in actual.items()}
        optimizer.step()
        expected_optimizer.step()
        updated = [
            name
            for name, parameter in optimizer.master_parameters.items()
            if not torch.equal(before[name], parameter)
        ]
        compute_updated = [
            name for name, p in actual.items() if not torch.equal(compute_before[name], p)
        ]
        modules = optimizer.update_evidence(before, compute_before, groups)
        reports.append(
            dict(
                step=step,
                loss=loss.item(),
                expected_loss=expected_loss.item(),
                target_count=count,
                logits_errors=logits_errors,
                gradient_errors=gradients,
                input_gradient_errors=feature_errors,
                updated_parameters=updated,
                updated_parameter_dtype="torch.float32",
                updated_compute_parameters=compute_updated,
                compute_parameter_dtype="torch.bfloat16",
                training_modules=modules,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        expected_optimizer.zero_grad(set_to_none=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_attention_argument(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", "GPU execution is opt-in"
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(), "Requires H100"
    assert not args.output.exists(), "Use a new report path"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    config, batches = fixture()
    report = dict(
        status="RUNNING",
        scope="Synthetic tiny BF16 native audio encoder and audio/text HF reference parity",
        distributed_pipeline="NOT_RUN",
        checkpoint_resume="NOT_RUN",
        real_processor="NOT_RUN",
        commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        cuda=torch.version.cuda,
        hardware=torch.cuda.get_device_name(),
        config=config.to_dict(),
        fixture_sha256=[
            hashlib.sha256(inputs["input_features"].numpy().tobytes()).hexdigest()
            for inputs, _ in batches
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["encoder"] = encoder_gate(config, device)
        report["reference_attention"] = {"contract": reference_contract(args)}
        report["audio_text"] = text_gate(
            config,
            batches,
            device,
            reference_attention=reference_contract(args),
            attention_report=report["reference_attention"],
        )
        report["status"] = "PASSED"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = repr(error)
        if isinstance(error, GradientParityError):
            report["gradient_diagnostics"] = error.diagnostics
            report["failure_context"] = error.context
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(dict(status=report["status"], output=str(args.output), commit=report["commit"]))
    )


if __name__ == "__main__":
    main()
