"""Offline CPU real-audio processor and native encoder parity with evidence."""

import argparse
import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HFAudio,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFPositions,
)

from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from pithtrain.models.qwen3_omni_audio_inputs import (
    flatten_audio_features,
    insert_audio_features,
    validate_audio_inputs,
)
from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset
from pithtrain.tasks.prepare_omni_data import processor_for, verify_bundle
from tests.acceptance.omni_audio_reference import error_ratio

REPO = Path(__file__).resolve().parents[2]


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def run(dataset, report):
    bundle = verify_bundle(dataset)
    processor, config = processor_for(bundle["recipe"], offline=True)
    reader = Qwen3OmniDataset(
        [dataset / name for name in bundle["manifests"]["train"]["audio"]],
        media_root=dataset,
        allowed_modalities={"audio"},
    )
    assert len(reader) >= 2
    collator = Qwen3OmniCollator(processor, config, max_length=2048)
    audio = config.audio_config
    audio.encoder_layers, audio.encoder_attention_heads = 2, 4
    audio.encoder_ffn_dim, audio.d_model, audio.downsample_hidden_size = 64, 32, 4
    audio.output_dim = 24
    config.text_config.hidden_size = 24
    # Keep the released 128-mel processor, convolution windows and inference windows.
    audio.conv_chunksize = 2
    torch.manual_seed(739)
    expected_config = copy.deepcopy(audio)
    expected_config._attn_implementation = "eager"
    reference, native = HFAudio(expected_config), Qwen3OmniMoeAudioEncoder(audio)
    native.load_state_dict(reference.state_dict(), strict=True)
    losses, expected_losses, audits, input_pairs = [], [], [], []
    for index in range(2):
        sample = reader[index]
        batch = collator([sample])
        inputs = batch.model_inputs
        positions = validate_audio_inputs(inputs["input_ids"], inputs, config)
        owner = SimpleNamespace(
            config=config, spatial_merge_size=config.vision_config.spatial_merge_size
        )
        expected_positions, _ = HFPositions.get_rope_index(
            owner, inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        torch.testing.assert_close(positions.float(), expected_positions, rtol=0, atol=0)
        inputs["input_features"].requires_grad_()
        expected_input = inputs["input_features"].detach().clone().requires_grad_()
        input_pairs.append((inputs["input_features"], expected_input))
        lengths = inputs["feature_attention_mask"].sum(-1)
        actual = native(flatten_audio_features(inputs), lengths).last_hidden_state
        expected = reference(
            expected_input.permute(0, 2, 1)[inputs["feature_attention_mask"].bool()].T,
            feature_lens=lengths,
        ).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        ids = inputs["input_ids"]
        embeddings = torch.randn(*ids.shape, 24, requires_grad=True)
        expected_embeddings = embeddings.detach().clone().requires_grad_()
        combined = insert_audio_features(embeddings, ids, config.audio_token_id, actual)
        combined_expected = expected_embeddings.masked_scatter(
            (ids == config.audio_token_id)[..., None], expected
        )
        probe = torch.randn_like(combined)
        losses.append((combined * probe).sum())
        expected_losses.append((combined_expected * probe).sum())
        audits.append(
            dict(
                sample_id=sample["id"],
                text=sample["text"],
                audio=[
                    dict(
                        path=str(Path(media["path"]).relative_to(dataset)),
                        sha256=hashlib.sha256(Path(media["path"]).read_bytes()).hexdigest(),
                    )
                    for media in sample["media"]
                ],
                input_features=dict(
                    shape=list(inputs["input_features"].shape),
                    dtype=str(inputs["input_features"].dtype),
                    sha256=tensor_hash(inputs["input_features"]),
                ),
                lengths=lengths.tolist(),
                token_count=ids.numel(),
                feature_count=len(actual),
                placeholder_count=int((ids == config.audio_token_id).sum()),
                valid_targets=int((batch.labels != -100).sum()),
                position_ids=positions.tolist(),
                input_ids=ids.tolist(),
                labels=batch.labels.tolist(),
                output_error=error_ratio(actual, expected),
            )
        )
    assert len({item["input_features"]["sha256"] for item in audits}) == 2
    sum(losses).backward()
    sum(expected_losses).backward()
    errors = {}
    for name, parameter in native.named_parameters():
        target = reference.get_parameter(name)
        assert parameter.grad is not None and target.grad is not None, name
        torch.testing.assert_close(parameter.grad, target.grad, rtol=2e-4, atol=2e-6, msg=name)
        errors[name] = error_ratio(parameter.grad, target.grad)
    input_errors = []
    for actual, expected in input_pairs:
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6)
        input_errors.append(error_ratio(actual.grad, expected.grad))
    before = {name: parameter.detach().clone() for name, parameter in native.named_parameters()}
    torch.optim.AdamW(native.parameters(), lr=1e-3, weight_decay=0).step()
    updated = [name for name, p in native.named_parameters() if not torch.equal(before[name], p)]
    for prefix in ("conv2d", "layers.", "proj"):
        assert any(name.startswith(prefix) for name in updated), prefix
    report.update(
        processor_revision=bundle["recipe"]["revision"],
        samples=audits,
        gradient_errors=errors,
        input_gradient_errors=input_errors,
        updated_parameters=updated,
        config=config.to_dict(),
        bundle_checksums_sha256=hashlib.sha256(
            (dataset / "checksums.sha256").read_bytes()
        ).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    assert not output.exists(), "Use a new report path"
    assert not output.is_relative_to(dataset), "Evidence must not modify the source bundle"
    assert not torch.cuda.is_initialized()
    report = dict(
        status="RUNNING",
        scope="CPU real audio processor, native encoder and text feature insertion",
        gpu_acceptance="NOT_RUN",
        distributed_pipeline="NOT_RUN",
        checkpoint_resume="NOT_RUN",
        commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(dataset, report)
        assert not torch.cuda.is_initialized()
        report["status"] = "PASSED"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = repr(error)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], output=str(output), commit=report["commit"])))


if __name__ == "__main__":
    main()
