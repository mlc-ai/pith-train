"""Offline mixed/AV preparation and explicit H100 DualPipeV/FSDP acceptance.

This command never submits jobs or downloads. Prepare first on CPU, then launch
staged with torchrun on the required hardware and the identical source commit.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HFAudio,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFPositions,
)

from pithtrain.models.qwen3_omni_audio_inputs import flatten_audio_features
from pithtrain.models.qwen3_omni_multimodal_inputs import multimodal_position_ids
from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset
from pithtrain.modules.training_data import OmniPretrainData, global_loss_mean, global_target_count
from pithtrain.tasks.prepare_omni_data import processor_for, verify_bundle, write_bundle, write_json
from pithtrain.tasks.tokenize_corpus import Writer
from tests.acceptance.omni_image_evidence import (
    module_update_evidence,
    training_groups,
    validate_target_layout,
)
from tests.acceptance.omni_image_staged import expected_shard, local_tensor
from tests.acceptance.omni_image_training import REPO, commit, error_ratio, tensor_hash
from tests.acceptance.omni_image_training import HFReference as ImageReference
from tests.acceptance.omni_native_resume import equal
from tests.acceptance.omni_optimizer_probe import isolated_optimizer_probe
from tests.acceptance.omni_reference_attention import (
    add_reference_attention_argument,
    configure_reference_attention,
    reference_attention_kwargs,
    reference_contract,
)
from tests.acceptance.omni_reference_precision import FP32GradientReference
from tests.test_qwen3_omni_av import write_av

MEDIA_KEYS = ("pixel_values", "pixel_values_videos", "input_features")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positions(config, inputs):
    owner = SimpleNamespace(
        config=config, spatial_merge_size=config.vision_config.spatial_merge_size
    )
    owner.get_llm_pos_ids_for_vision = HFPositions.get_llm_pos_ids_for_vision.__get__(owner)
    mask = inputs.get("feature_attention_mask")
    return HFPositions.get_rope_index(
        owner,
        inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs["attention_mask"],
        audio_seqlens=None if mask is None else mask.sum(-1),
        second_per_grids=inputs.get("video_second_per_grid"),
        use_audio_in_video=bool(inputs.get("use_audio_in_video", False)),
    )[0]


class HFReference(ImageReference):
    def __init__(self, config, dtype):
        super().__init__(config, dtype)
        self.config.audio_config._attn_implementation = "eager"
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with torch.device("cpu"), torch.random.fork_rng(devices=[]):
                torch.manual_seed(739)
                self.audio_tower = HFAudio(self.config.audio_config)
        finally:
            torch.set_default_dtype(previous)

    def forward(self, inputs):
        ids, cfg = inputs["input_ids"], self.config
        hidden = self.model.embed_tokens(ids)
        visual_mask = (ids == cfg.image_token_id) | (ids == cfg.video_token_id)
        lanes = [torch.zeros_like(hidden) for _ in cfg.vision_config.deepstack_visual_indexes]
        for token, pixel, grid in (
            (cfg.image_token_id, "pixel_values", "image_grid_thw"),
            (cfg.video_token_id, "pixel_values_videos", "video_grid_thw"),
        ):
            if pixel in inputs:
                output = self.visual(inputs[pixel], inputs[grid])
                mask = (ids == token)[..., None]
                hidden = hidden.masked_scatter(mask, output.pooler_output.to(hidden.dtype))
                for i, feature in enumerate(output.deepstack_features):
                    lanes[i] = lanes[i].masked_scatter(mask, feature.to(hidden.dtype))
        if "input_features" in inputs:
            audio = self.audio_tower(
                flatten_audio_features(inputs),
                feature_lens=inputs["feature_attention_mask"].sum(-1),
            ).last_hidden_state
            hidden = hidden.masked_scatter(
                (ids == cfg.audio_token_id)[..., None], audio.to(hidden.dtype)
            )
        hidden = self.model(
            inputs_embeds=hidden,
            attention_mask=inputs["attention_mask"],
            position_ids=positions(cfg, inputs),
            visual_pos_masks=visual_mask if visual_mask.any() else None,
            deepstack_visual_embeds=[lane[visual_mask] for lane in lanes]
            if visual_mask.any()
            else None,
            use_cache=False,
            **reference_attention_kwargs(self),
        ).last_hidden_state
        return self.lm_head(hidden)


def source_inputs(dataset):
    bundle = verify_bundle(dataset)
    processor, config = processor_for(bundle["recipe"], offline=True)
    records = {}
    for kind in ("text", "image", "audio", "video"):
        reader = Qwen3OmniDataset(
            [dataset / p for p in bundle["manifests"]["train"][kind]], media_root=dataset
        )
        records[kind] = reader[0]
        if kind == "video":
            assert len(reader) == 2
            records["av"] = reader[1]
    records["all"] = dict(
        id="controlled-all",
        text="An image, speech and synchronized frames.",
        media=[*records["audio"]["media"], *records["image"]["media"], *records["av"]["media"]],
    )
    collator = Qwen3OmniCollator(processor, config, **bundle["recipe"]["batch"])
    batches = {kind: collator([sample]) for kind, sample in records.items()}
    text = json.loads((REPO / "tests/configs/qwen3_omni_text/tiny.json").read_text())
    config.text_config = Qwen3OmniMoeTextConfig(**text)
    vision, audio = config.vision_config, config.audio_config
    vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size = 3, 64, 4, 96
    vision.out_hidden_size, vision.deepstack_visual_indexes = config.text_config.hidden_size, [0, 2]
    audio.encoder_layers, audio.d_model, audio.encoder_attention_heads, audio.encoder_ffn_dim = (
        2,
        64,
        4,
        96,
    )
    audio.downsample_hidden_size, audio.output_dim = 8, config.text_config.hidden_size
    return bundle, records, batches, config


def prepare(args):
    assert not torch.cuda.is_initialized()
    assert not args.output.exists()
    source = verify_bundle(args.dataset)
    args.output.mkdir(parents=True)
    destination = args.output / "data"
    destination.mkdir()
    (destination / "media").mkdir()
    (destination / "train").mkdir()
    recipe = copy.deepcopy(source["recipe"])
    recipe["batch"].update(
        use_audio_in_video=True, min_pixels=4096, max_pixels=4096, max_length=2048
    )
    recipe["sampling_weights"] = {kind: 1 for kind in ("text", "image", "audio", "video")}
    write_json(destination / "recipe.json", dict(recipe=recipe, stage="video", format_version=3))
    (destination / "rejected.jsonl").write_text("")
    processor, _ = processor_for(recipe, offline=True)
    manifests, statistics, files = {}, {}, ["recipe.json", "rejected.jsonl"]
    for kind in ("text", "image", "audio", "video"):
        reader = Qwen3OmniDataset(
            [args.dataset / p for p in source["manifests"]["train"][kind]], media_root=args.dataset
        )
        row = copy.deepcopy(reader[0])
        for index, item in enumerate(row["media"]):
            path = item["path"]
            name = f"media/{kind}-{index}{path.suffix}"
            shutil.copyfile(path, destination / name)
            files.append(name)
            item["path"] = name
        rows = [row]
        if kind == "text":
            name = "tokens/train/00000.bin"
            (destination / name).parent.mkdir(parents=True)
            tokens = processor.tokenizer.encode(row["text"]) + [processor.tokenizer.eos_token_id]
            writer = Writer(destination / name)
            writer.append(np.asarray(tokens, dtype=np.uint32))
            writer.flush()
            files.append(name)
        if kind == "video":
            name = "media/controlled-av.mkv"
            write_av(destination / name)
            files.append(name)
            rows.append(
                dict(
                    id="controlled-av",
                    text="Frames brighten with a tone.",
                    media=[dict(type="video", path=name)],
                )
            )
        name = f"train/{kind}.jsonl"
        (destination / name).write_text("".join(json.dumps(item) + "\n" for item in rows))
        files.append(name)
        manifests[kind], statistics[kind] = [name], dict(samples=len(rows))
    bundle = dict(
        status="complete",
        recipe=recipe,
        manifests=dict(train=manifests),
        statistics=dict(train=statistics),
        source_checksum=digest(args.dataset / "checksums.sha256"),
        purpose="Controlled mixed acceptance; selected real records plus synthetic timestamped AV",
    )
    write_bundle(destination, bundle, files)
    _, _, batches, config = source_inputs(destination)
    config.save_pretrained(args.output / "model")
    audits = {}
    for kind, batch in batches.items():
        actual = multimodal_position_ids(
            batch.model_inputs["input_ids"], batch.model_inputs, config
        )
        torch.testing.assert_close(actual, positions(config, batch.model_inputs), rtol=0, atol=0)
        audits[kind] = dict(
            sample_ids=batch.sample_ids,
            targets=int((batch.labels != -100).sum()),
            inputs={
                key: dict(
                    shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_hash(value)
                )
                for key, value in batch.model_inputs.items()
            },
        )
    # Verify the actual deterministic production mixture, then prove exact data resume.
    data_cfg = DataCfg()
    data_cfg.dataset, data_cfg.format = destination, "prepared_bundle"
    data_cfg.modalities, data_cfg.epoch_samples = ("text", "image", "audio", "video"), 32
    train_cfg = SimpleNamespace(
        global_batch_size=8, micro_batch_size=1, sequence_length=2048, seed=1234
    )
    data = OmniPretrainData(data_cfg, train_cfg)
    data.validate_model(
        SimpleNamespace(
            input_modalities=data_cfg.modalities, validate_media_batch=multimodal_position_ids
        ),
        config,
    )
    steps, saved = [], None
    for step in range(4):
        micro = data.get_batch(step, "cpu")
        kinds = [batch.sample_ids for batch in micro]
        steps.append(kinds)
        assert any("input_features" in (batch.media_inputs or {}) for batch in micro), (
            f"No audio at step {step}"
        )
        assert any(
            any(
                key in (batch.media_inputs or {}) for key in ("pixel_values", "pixel_values_videos")
            )
            for batch in micro
        ), f"No vision at step {step}"
        data.commit_step(step)
        if step == 1:
            saved = data.state_dict()
        if step == 2:
            expected = batch_hashes(micro)
    restored = OmniPretrainData(data_cfg, train_cfg)
    restored.validate_model(
        SimpleNamespace(
            input_modalities=data_cfg.modalities, validate_media_batch=multimodal_position_ids
        ),
        config,
    )
    restored.load_state_dict(saved)
    equal(batch_hashes(restored.get_batch(2, "cpu")), expected, "restored-mixed-data")
    report = dict(
        status="PASSED",
        scope="CPU real/controlled processor positions and exact next-data restore",
        gpu_acceptance="NOT_RUN",
        commit=commit(),
        inputs=audits,
        mixture_steps=steps,
        dataset_checksum=digest(destination / "checksums.sha256"),
        torch=torch.__version__,
    )
    (args.output / "prepare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def batch_hashes(batches):
    return [
        dict(
            sample_ids=batch.sample_ids,
            labels=tensor_hash(batch.objective_inputs[0]),
            context={key: tensor_hash(value) for key, value in batch.context_for_stage(0).items()},
        )
        for batch in batches
    ]


class ObservedData:
    def __init__(self, source):
        self.source, self.batches = source, {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_batch(self, step, device):
        batches = self.source.get_batch(step, device)
        self.batches[step] = batch_hashes(batches)
        return batches


def component_groups(config):
    return (
        *training_groups(config),
        "audio_tower.conv2d",
        "audio_tower.conv_out.",
        "audio_tower.ln_post.",
        "audio_tower.proj1.",
        "audio_tower.proj2.",
        *(f"audio_tower.layers.{i}." for i in range(config.audio_config.encoder_layers)),
    )


def staged(args):
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.distributed_checkpoint import resume_contract
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import (
        make_adamw_optimizer,
        make_constant_scheduler,
        setup_training,
    )
    from pithtrain.tasks import pretrain_lm
    from tests.acceptance.omni_distributed_resume import snapshot

    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name()
    assert not args.output.exists(), "Use a fresh evidence directory"
    _, _, batches, config = source_inputs(args.dataset)
    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    cfg = pretrain_lm.PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size, cfg.distributed.expert_parallel_size = args.pp, args.ep
    cfg.distributed.timeout = timedelta(seconds=90)
    t = cfg.training
    t.model, t.native_multimodal = args.output / "model", True
    t.fp8, t.moe_load_balance_coef = False, 0.0
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.lr, t.max_steps, t.sequence_length, t.global_batch_size, t.micro_batch_size = (
        1e-4,
        4,
        2048,
        8,
        1,
    )
    t.save_interval, t.strict_resume = None, True
    cfg.data.dataset, cfg.data.format = args.dataset, "prepared_bundle"
    cfg.data.modalities, cfg.data.epoch_samples = ("text", "image", "audio", "video"), 32
    setup_logging(cfg)
    setup_distributed(cfg)
    assert world % args.pp == 0 and (world // args.pp) % args.ep == 0
    assert 8 // distributed.dp_size >= 2 * args.pp
    if rank == 0:
        args.output.mkdir(parents=True)
        config.save_pretrained(t.model)
    dist.barrier()
    data = ObservedData(pretrain_lm.setup_dataset(cfg))
    setup_training(cfg)
    hf_reference = HFReference(config, torch.bfloat16).to(distributed.device)
    attention = configure_reference_attention(hf_reference, config, reference_contract(args))
    (args.output / f"reference-attention-rank-{rank:05d}.json").write_text(
        json.dumps(attention, indent=2) + "\n"
    )
    reference = FP32GradientReference(hf_reference)
    expected, actual = reference.master_parameters, {}
    for module in training.model.module:
        for name, parameter in module.named_parameters():
            key = (
                name
                if name.startswith(("visual.", "audio_tower.", "lm_head."))
                else "model." + name
            )
            assert key not in actual and key in expected, key
            actual[key] = parameter
            with torch.no_grad():
                parameter.copy_(expected_shard(expected[key].detach(), parameter, key, distributed))
    ownership = [None] * world
    dist.all_gather_object(ownership, sorted(actual))
    assert set().union(*(set(keys) for keys in ownership)) == set(expected)
    kinds = ("text", "all", "image", "audio", "video", "all", "text", "av")
    global_batches = [copy.deepcopy(batches[kind]) for kind in kinds]
    for index, batch in enumerate(global_batches):
        if index == 0 or (args.scenario == "rank-zero" and index % distributed.dp_size == 0):
            batch.labels.fill_(-100)
        elif index % 3 == 0:
            batch.labels[:, ::2] = -100
    layout = validate_target_layout(global_batches, distributed.dp_size, args.scenario)
    microbatches = []
    for batch in global_batches[distributed.dp_rank :: distributed.dp_size]:
        inputs = {key: value.to(distributed.device) for key, value in batch.model_inputs.items()}
        media = {key: inputs.pop(key) for key in MEDIA_KEYS if key in inputs}
        microbatches.append(
            Microbatch(
                model_inputs=(inputs["input_ids"],),
                cu_seqlens=None,
                objective_inputs=(batch.labels.to(distributed.device),),
                model_context=inputs,
                media_inputs=media if distributed.pp_rank == 0 else None,
                sample_ids=batch.sample_ids,
            )
        )
    group = distributed.attn_mesh["dp", "cp"]._flatten().get_group()
    count = global_target_count(microbatches, group)
    assert int(count) == layout["global_target_count"]
    hf_loss = 0
    for batch in global_batches:
        inputs = {key: value.to(distributed.device) for key, value in batch.model_inputs.items()}
        logits = reference(inputs)
        loss = F.cross_entropy(
            logits.float().flatten(0, 1),
            batch.labels.to(distributed.device).flatten(),
            reduction="sum",
        )
        loss.backward()
        hf_loss += loss.detach()
    # Match native BF16 compute with FP32 accumulation, then normalize once.
    # The checked layout provides the positive integer API; count is FP32.
    reference.normalize_gradients_(layout["global_target_count"])
    hf_loss = hf_loss / count
    outputs = training.model.step(microbatches, pretrain_lm.objective)
    if distributed.pp_rank == 0:
        loss = global_loss_mean(torch.stack(outputs).sum(), count, group)
        torch.testing.assert_close(loss, hf_loss, rtol=1e-3, atol=1e-3)
    errors = {}
    for name, parameter in actual.items():
        assert parameter.grad is not None and expected[name].grad is not None, name
        parameter.grad.div_(count)
        target = expected_shard(expected[name].grad, parameter, name, distributed)
        errors[name] = error_ratio(local_tensor(parameter.grad), local_tensor(target))
        assert errors[name] < 1e-2, (name, errors[name])
    before = {name: local_tensor(param).detach().clone() for name, param in actual.items()}
    with isolated_optimizer_probe(training.model, lambda: make_adamw_optimizer(t)) as probes:
        for optimizer in probes:
            optimizer.step()
        updates = module_update_evidence(
            actual, before, component_groups(config), reduce=dist.all_reduce
        )
    del probes, optimizer
    for optimizer in training.optimizers:
        optimizer.zero_grad(set_to_none=True)
    contract, checkpoint = resume_contract(cfg), args.output / "checkpoint"
    for step in range(4):
        pretrain_lm.train_step(cfg, data, step)
        if step == 1:
            save_checkpoint(
                checkpoint, 2, data_state=data.checkpoint_state, strict=True, resume_config=contract
            )
            boundary = snapshot(data)
        if step == 2:
            next_inputs, next_state = data.batches[2], snapshot(data)
    assert data.consumed_samples == 32
    load_checkpoint(
        checkpoint, 2, data_state=data.checkpoint_state, strict=True, resume_config=contract
    )
    equal(snapshot(data), boundary, "restored-model-optimizer-scheduler-rng-data")
    pretrain_lm.train_step(cfg, data, 2)
    equal(data.batches[2], next_inputs, "next-mixed-inputs-and-media")
    equal(snapshot(data), next_state, "next-native-update")
    report = dict(
        status="PASSED",
        commit=commit(),
        rank=rank,
        world=world,
        pp=args.pp,
        ep=args.ep,
        scenario=args.scenario,
        target_layout=layout,
        gradient_errors=errors,
        reference_precision=reference.precision_contract(),
        reference_attention=attention,
        module_updates=updates,
        strict_resume="EXACT",
        next_inputs=next_inputs,
        processor=verify_bundle(args.dataset)["recipe"]["revision"],
        dataset_checksum=digest(args.dataset / "checksums.sha256"),
        torch=torch.__version__,
        hardware=torch.cuda.get_device_name(),
        max_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.output / f"rank-{rank:05d}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_attention_argument(parser)
    parser.add_argument("mode", choices=("prepare", "staged"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--scenario", choices=("nonuniform", "rank-zero"), default="nonuniform")
    args = parser.parse_args()
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    (prepare if args.mode == "prepare" else staged)(args)


if __name__ == "__main__":
    main()
