"""Offline real-video audit and opt-in H100 native video training gates.

Uses the existing prepared-bundle reader and pinned processor. The native
consumer trains temporal visual patches, main/DeepStack projections and the
Thinker decoder. Audio tracks and interleaved audio/video remain unsupported.
No downloads, cluster submission, or spending occurs in this entrypoint.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as HFVision,
)

from pithtrain.models.qwen3_omni_video_inputs import validate_video_inputs
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVideoEncoder
from tests.acceptance.omni_image_evidence import fixed_image_batches, validate_target_layout
from tests.acceptance.omni_image_training import REPO, commit, error_ratio, tensor_hash
from tests.acceptance.omni_image_training import (
    HFReference as ImageHFReference,
)
from tests.acceptance.omni_reference_attention import (
    add_reference_attention_argument,
    reference_attention_kwargs,
)
from tests.test_qwen3_omni_video import hf_video_positions


def source_inputs(dataset):
    from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset
    from pithtrain.tasks.prepare_omni_data import processor_for, verify_bundle

    bundle = verify_bundle(dataset)
    processor, config = processor_for(bundle["recipe"], offline=True)
    reader = Qwen3OmniDataset(
        [dataset / name for name in bundle["manifests"]["train"]["video"]],
        media_root=dataset,
        allowed_modalities={"video"},
    )
    assert len(reader) >= 2
    samples = [reader[i] for i in range(2)]
    collator = Qwen3OmniCollator(
        processor,
        config,
        **bundle["recipe"]["batch"],
    )
    batches = [collator([sample]) for sample in samples]
    tiny = json.loads((REPO / "tests/configs/qwen3_omni_text/tiny.json").read_text())
    config.text_config = Qwen3OmniMoeTextConfig(**tiny)
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size = 3, 64, 4, 96
    vision.out_hidden_size = config.text_config.hidden_size
    vision.deepstack_visual_indexes = [0, 2]
    assert all((batch.model_inputs["video_grid_thw"][:, 0] > 1).all() for batch in batches)
    return bundle, samples, batches, config


class HFReference(ImageHFReference):
    """Independent HF vision/text execution and HF temporal position construction."""

    def forward(self, inputs):
        ids = inputs["input_ids"]
        mask = ids == self.config.video_token_id
        embeddings = self.model.embed_tokens(ids)
        output = self.visual(inputs["pixel_values_videos"], inputs["video_grid_thw"])
        embeddings = embeddings.masked_scatter(
            mask[..., None], output.pooler_output.to(embeddings.dtype)
        )
        positions = hf_video_positions(
            ids,
            inputs["video_grid_thw"],
            inputs["video_second_per_grid"],
            self.config,
            inputs["attention_mask"],
        )
        hidden = self.model(
            inputs_embeds=embeddings,
            attention_mask=inputs["attention_mask"],
            position_ids=positions,
            visual_pos_masks=mask,
            deepstack_visual_embeds=output.deepstack_features,
            use_cache=False,
            **reference_attention_kwargs(self),
        ).last_hidden_state
        return self.lm_head(hidden)


def cpu_gate(args):
    assert not torch.cuda.is_initialized()
    assert not args.output.exists(), "Use a new evidence directory"
    bundle, samples, batches, config = source_inputs(args.dataset)
    torch.manual_seed(739)
    reference_config = copy.deepcopy(config.vision_config)
    reference_config._attn_implementation = "eager"
    reference = HFVision(reference_config)
    native = Qwen3OmniMoeVideoEncoder(config.vision_config)
    native.load_state_dict(reference.state_dict(), strict=True)
    audits, losses, expected_losses = [], [], []
    for sample, batch in zip(samples, batches, strict=True):
        inputs = batch.model_inputs
        positions = validate_video_inputs(inputs["input_ids"], inputs, config)
        torch.testing.assert_close(
            positions,
            hf_video_positions(
                inputs["input_ids"],
                inputs["video_grid_thw"],
                inputs["video_second_per_grid"],
                config,
                inputs["attention_mask"],
            ),
            rtol=0,
            atol=0,
        )
        a = native(inputs["pixel_values_videos"], inputs["video_grid_thw"])
        b = reference(inputs["pixel_values_videos"], inputs["video_grid_thw"])
        for actual, expected in zip(
            (a.pooler_output, *a.deepstack_features),
            (b.pooler_output, *b.deepstack_features),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            probe = torch.randn_like(actual)
            losses.append((actual * probe).sum())
            expected_losses.append((expected * probe).sum())
        audits.append(
            dict(
                sample_id=sample["id"],
                text=sample["text"],
                videos=[
                    dict(
                        path=str(Path(media["path"]).relative_to(args.dataset)),
                        sha256=hashlib.sha256(Path(media["path"]).read_bytes()).hexdigest(),
                    )
                    for media in sample["media"]
                ],
                pixel_values=dict(
                    shape=list(inputs["pixel_values_videos"].shape),
                    dtype=str(inputs["pixel_values_videos"].dtype),
                    sha256=tensor_hash(inputs["pixel_values_videos"]),
                ),
                grid=inputs["video_grid_thw"].tolist(),
                second_per_grid=inputs["video_second_per_grid"].tolist(),
                input_ids=inputs["input_ids"].tolist(),
                labels=batch.labels.tolist(),
                valid_targets=int((batch.labels != -100).sum()),
                feature_count=len(a.pooler_output),
                placeholder_count=int((inputs["input_ids"] == config.video_token_id).sum()),
                position_ids=positions.tolist(),
            )
        )
    assert len({a["pixel_values"]["sha256"] for a in audits}) == 2
    sum(losses).backward()
    sum(expected_losses).backward()
    gradients = {}
    for name, p in native.named_parameters():
        q = reference.get_parameter(name)
        assert p.grad is not None and q.grad is not None, name
        gradients[name] = error_ratio(p.grad, q.grad)
        torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=3e-5, msg=name)
    before = {n: p.detach().clone() for n, p in native.named_parameters()}
    torch.optim.AdamW(native.parameters(), lr=1e-3, weight_decay=0).step()
    updated = [n for n, p in native.named_parameters() if not torch.equal(before[n], p)]
    assert len(updated) == len(before)
    args.output.mkdir(parents=True)
    config.save_pretrained(args.output / "model")
    report = dict(
        status="PASSED",
        scope="CPU native video, temporal geometry, gradients and fixture layout",
        staged_target_layouts={
            scenario: validate_target_layout(fixed_image_batches(batches, 2, scenario), 2, scenario)
            for scenario in ("nonuniform", "rank-zero")
        },
        gpu_acceptance="NOT_RUN",
        processor=bundle["recipe"]["revision"],
        samples=audits,
        gradient_errors=gradients,
        updated_parameters=updated,
        torch=torch.__version__,
        commit=commit(),
        input_manifest=hashlib.sha256((args.dataset / "checksums.sha256").read_bytes()).hexdigest(),
    )
    (args.output / "cpu.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    print(
        json.dumps(
            dict(
                status=report["status"],
                samples=[a["sample_id"] for a in audits],
                targets=[a["valid_targets"] for a in audits],
                parameters=len(gradients),
                worst_gradient=max(gradients.items(), key=lambda item: item[1]),
                cuda_initialized=False,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_attention_argument(parser)
    parser.add_argument("--mode", choices=("cpu", "reference", "staged"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--scenario", choices=("nonuniform", "rank-zero"), default="nonuniform")
    args = parser.parse_args()
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    assert not args.output.is_relative_to(args.dataset) and not args.dataset.is_relative_to(
        args.output
    )
    if args.mode == "cpu":
        cpu_gate(args)
        return
    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", (
        "Explicit GPU authorization required"
    )
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(), "Use H100"
    if args.mode == "reference":
        from pithtrain.models.qwen3_omni_video import Qwen3OmniMoeThinkerVideoModel
        from tests.acceptance.omni_image_training import gpu_reference

        gpu_reference(
            args,
            source_inputs_fn=source_inputs,
            reference_class=HFReference,
            native_class=Qwen3OmniMoeThinkerVideoModel,
        )
    else:
        from tests.acceptance.omni_image_staged import run

        run(
            args,
            source_inputs_fn=source_inputs,
            reference_class=HFReference,
            modality="video",
            pixel_key="pixel_values_videos",
            training_flag="native_video",
            sequence_length=2048,
        )


if __name__ == "__main__":
    main()
