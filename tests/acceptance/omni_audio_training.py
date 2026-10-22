"""Real LibriSpeech native audio/text HF and staged DualPipeV/FSDP gates.

Run each topology/scenario in a fresh externally bounded torchrun process.
This driver never submits compute. Only preflight is allowed without GPU opt-in.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig

from pithtrain.models.qwen3_omni_audio_inputs import validate_audio_inputs
from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset
from pithtrain.tasks.prepare_omni_data import processor_for, verify_bundle
from tests.acceptance.omni_audio_reference import HFReference
from tests.acceptance.omni_image_evidence import fixed_image_batches, validate_target_layout
from tests.acceptance.omni_reference_attention import (
    add_reference_attention_argument,
)

REPO = Path(__file__).resolve().parents[2]


def source_inputs(dataset):
    dataset = Path(dataset).resolve()
    bundle = verify_bundle(dataset)
    processor, config = processor_for(bundle["recipe"], offline=True)
    reader = Qwen3OmniDataset(
        [dataset / name for name in bundle["manifests"]["train"]["audio"]],
        media_root=dataset,
        allowed_modalities={"audio"},
    )
    assert len(reader) >= 2, "Two distinct real audio recordings are required"
    samples = [reader[index] for index in range(2)]
    collator = Qwen3OmniCollator(processor, config, max_length=2048)
    batches = [collator([sample]) for sample in samples]
    raw = json.loads((REPO / "tests/configs/qwen3_omni_text/tiny.json").read_text())
    raw["max_position_embeddings"] = 2048
    config.text_config = Qwen3OmniMoeTextConfig(**raw)
    audio = config.audio_config
    audio.encoder_layers, audio.encoder_attention_heads = 2, 4
    audio.encoder_ffn_dim, audio.d_model, audio.downsample_hidden_size = 64, 32, 4
    audio.output_dim, audio.conv_chunksize = config.text_config.hidden_size, 2
    for batch in batches:
        validate_audio_inputs(batch.model_inputs["input_ids"], batch.model_inputs, config)
        assert (batch.labels != -100).any()
    assert batches[0].sample_ids != batches[1].sample_ids
    return bundle, samples, batches, config


def hf_name(name):
    return name if name.startswith(("audio_tower.", "visual.", "lm_head.")) else "model." + name


def training_groups(config):
    return (
        "audio_tower.conv2d1.",
        "audio_tower.conv2d2.",
        "audio_tower.conv2d3.",
        "audio_tower.conv_out.",
        "audio_tower.ln_post.",
        "audio_tower.proj1.",
        "audio_tower.proj2.",
        *(f"audio_tower.layers.{i}." for i in range(config.audio_config.encoder_layers)),
        "model.embed_tokens.",
        "model.norm.",
        "lm_head.",
        *(f"model.layers.{i}." for i in range(config.text_config.num_hidden_layers)),
    )


def preflight(args):
    assert not torch.cuda.is_initialized()
    assert not args.output.exists(), "Use a fresh evidence directory"
    assert not args.output.is_relative_to(args.dataset)
    bundle, samples, batches, config = source_inputs(args.dataset)
    layout = validate_target_layout(
        fixed_image_batches(batches, args.dp, args.scenario), args.dp, args.scenario
    )
    args.output.mkdir(parents=True)
    config.save_pretrained(args.output / "model")
    report = dict(
        status="PASSED",
        scope="CPU real audio fixture and topology/target layout only",
        gpu_acceptance="NOT_RUN",
        pp=args.pp,
        ep=args.ep,
        dp=args.dp,
        cp=1,
        processor_revision=bundle["recipe"]["revision"],
        samples=[sample["id"] for sample in samples],
        target_layout=layout,
        bundle_checksums_sha256=hashlib.sha256(
            (args.dataset / "checksums.sha256").read_bytes()
        ).hexdigest(),
        training_groups=training_groups(config),
    )
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    print(json.dumps(report))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_attention_argument(parser)
    parser.add_argument("--mode", choices=("preflight", "staged"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ep", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--scenario", choices=("nonuniform", "rank-zero"), default="nonuniform")
    args = parser.parse_args(argv)
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    if args.dp % args.ep or 8 // args.dp < 2 * args.pp:
        parser.error("EP must divide DP and GB8 must provide at least 2*PP microbatches")
    if args.scenario == "rank-zero" and args.dp == 1:
        parser.error("rank-zero requires DP=2")
    return args


def main():
    args = parse_args()
    if args.mode == "preflight":
        preflight(args)
        return
    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", "GPU execution is opt-in"
    assert torch.cuda.is_available(), "Requires CUDA"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert "H100" in torch.cuda.get_device_name(), "Requires H100"
    assert int(os.environ["WORLD_SIZE"]) == args.pp * args.dp, "Use the requested CP1 topology"
    from tests.acceptance.omni_image_staged import run

    run(
        args,
        source_inputs_fn=source_inputs,
        reference_class=HFReference,
        modality="audio",
        pixel_key="input_features",
        training_flag="native_audio",
        sequence_length=2048,
        hf_name_fn=hf_name,
        training_groups_fn=training_groups,
    )


if __name__ == "__main__":
    main()
