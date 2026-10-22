"""Offline real-image audit and H100 native image training gates.

CPU mode executes the existing PR100 reader/processor and native vision. GPU
modes are opt-in and fail on other hardware. No downloads or cluster submission.
Tiny random weights retain the real tokenizer vocabulary and processor geometry.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel as HFText,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as HFVision,
)

from pithtrain.models.qwen3_omni_image_inputs import validate_image_inputs
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVisionEncoder
from tests.acceptance.omni_image_evidence import (
    fixed_image_batches,
    module_update_evidence,
    training_groups,
    validate_target_layout,
)
from tests.acceptance.omni_reference_attention import (
    add_reference_attention_argument,
    configure_reference_attention,
    reference_attention_kwargs,
    reference_contract,
)
from tests.test_qwen3_omni_vision import hf_positions

REPO = Path(__file__).resolve().parents[2]


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def error_ratio(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    a, b = actual.double(), expected.double()
    denominator = (a.square() + b.square()).sum()
    return 0.0 if denominator == 0 else ((a - b).square().sum() / denominator).item()


def source_inputs(dataset):
    from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset
    from pithtrain.tasks.prepare_omni_data import processor_for, verify_bundle

    bundle = verify_bundle(dataset)
    processor, config = processor_for(bundle["recipe"], offline=True)
    reader = Qwen3OmniDataset(
        [dataset / name for name in bundle["manifests"]["train"]["image"]],
        media_root=dataset,
        allowed_modalities={"image"},
    )
    assert len(reader) >= 2
    samples = [reader[i] for i in range(2)]
    collator = Qwen3OmniCollator(
        processor, config, max_length=512, min_pixels=4096, max_pixels=4096
    )
    batches = [collator([sample]) for sample in samples]
    tiny = json.loads((REPO / "tests/configs/qwen3_omni_text/tiny.json").read_text())
    config.text_config = Qwen3OmniMoeTextConfig(**tiny)
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size = 3, 64, 4, 96
    vision.out_hidden_size = config.text_config.hidden_size
    vision.deepstack_visual_indexes = [0, 2]
    return bundle, samples, batches, config


class HFReference(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        config = copy.deepcopy(config)
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        self.config = config
        # BF16 construction leaves the independent HF rotary buffers in FP32.
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with torch.device("cpu"), torch.random.fork_rng(devices=[]):
                torch.manual_seed(739)
                self.visual = HFVision(config.vision_config)
                self.model = HFText(config.text_config)
                self.lm_head = nn.Linear(
                    config.text_config.hidden_size, config.text_config.vocab_size, bias=False
                )
        finally:
            torch.set_default_dtype(previous)

    def forward(self, inputs):
        ids = inputs["input_ids"]
        mask = ids == self.config.image_token_id
        embeddings = self.model.embed_tokens(ids)
        output = self.visual(inputs["pixel_values"], inputs["image_grid_thw"])
        embeddings = embeddings.masked_scatter(
            mask[..., None], output.pooler_output.to(embeddings.dtype)
        )
        positions = hf_positions(
            ids, inputs["image_grid_thw"], self.config, inputs["attention_mask"]
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
    native = Qwen3OmniMoeVisionEncoder(config.vision_config)
    native.load_state_dict(reference.state_dict(), strict=True)
    audits, losses, expected_losses = [], [], []
    for sample, batch in zip(samples, batches, strict=True):
        inputs = batch.model_inputs
        positions = validate_image_inputs(inputs["input_ids"], inputs, config)
        torch.testing.assert_close(
            positions.float(),
            hf_positions(
                inputs["input_ids"], inputs["image_grid_thw"], config, inputs["attention_mask"]
            ),
            rtol=0,
            atol=0,
        )
        a = native(inputs["pixel_values"], inputs["image_grid_thw"])
        b = reference(inputs["pixel_values"], inputs["image_grid_thw"])
        for actual, expected in zip(
            (a.pooler_output, *a.deepstack_features),
            (b.pooler_output, *b.deepstack_features),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            probe = torch.randn_like(actual)
            losses.append((actual * probe).sum())
            expected_losses.append((expected * probe).sum())
        paths = [Path(media["path"]) for media in sample["media"]]
        audits.append(
            dict(
                sample_id=sample["id"],
                text=sample["text"],
                images=[
                    dict(
                        path=str(path.relative_to(args.dataset)),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    for path in paths
                ],
                pixel_values=dict(
                    shape=list(inputs["pixel_values"].shape),
                    dtype=str(inputs["pixel_values"].dtype),
                    sha256=tensor_hash(inputs["pixel_values"]),
                ),
                grid=inputs["image_grid_thw"].tolist(),
                input_ids=inputs["input_ids"].tolist(),
                labels=batch.labels.tolist(),
                valid_targets=int((batch.labels != -100).sum()),
                feature_count=len(a.pooler_output),
                placeholder_count=int((inputs["input_ids"] == config.image_token_id).sum()),
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
        scope="CPU native vision, image geometry and staged fixture layout only",
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


def commit():
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def hf_name(name):
    return name if name.startswith(("visual.", "lm_head.")) else "model." + name


class FP32MasterAdamW:
    """Reference-only FP32 optimizer storage with BF16 compute materialization.

    Production setup_model constructs FP32 parameters before FSDP, whose policy
    materializes BF16 compute parameters and reduces gradients in FP32. Direct
    AdamW on the standalone BF16 reference instead rounds away small updates,
    including norm=1 at lr=1e-4. Keep the existing forward/gradient comparison
    intact, then promote its gradients for the same FP32 AdamW update semantics.
    This does not emulate FSDP collectives or change gradient accumulation.
    """

    def __init__(self, parameters, **kwargs):
        assert parameters and all(p.dtype == torch.bfloat16 for p in parameters.values())
        self.compute_parameters = dict(parameters)
        self.master_parameters = {
            name: nn.Parameter(p.detach().float().clone(), requires_grad=p.requires_grad)
            for name, p in parameters.items()
        }
        self.optimizer = torch.optim.AdamW(self.master_parameters.values(), **kwargs)

    @torch.no_grad()
    def step(self):
        for name, parameter in self.compute_parameters.items():
            master = self.master_parameters[name]
            master.grad = None if parameter.grad is None else parameter.grad.detach().float()
        self.optimizer.step()
        for name, parameter in self.compute_parameters.items():
            parameter.copy_(self.master_parameters[name])

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self.compute_parameters.values():
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

    def update_evidence(self, before, compute_before, groups):
        """Require real master updates; separately disclose rounded compute changes."""
        evidence = module_update_evidence(self.master_parameters, before, groups)
        for prefix, row in evidence.items():
            row["master_parameter_dtype"] = "torch.float32"
            row["compute_parameter_dtype"] = "torch.bfloat16"
            row["compute_update_squared_sum"] = sum(
                (parameter.detach().double() - compute_before[name].double()).square().sum().item()
                for name, parameter in self.compute_parameters.items()
                if name.startswith(prefix)
            )
        return evidence


def gpu_reference(
    args, *, source_inputs_fn=source_inputs, reference_class=HFReference, native_class=None
):
    from pithtrain.contexts import distributed, training
    from pithtrain.models.qwen3_omni_image import Qwen3OmniMoeThinkerImageModel
    from pithtrain.operators.grouped_linear import GroupedLinear

    assert not args.output.exists()
    _, _, batches, config = source_inputs_fn(args.dataset)
    device = torch.device("cuda", 0)
    distributed.device = device
    distributed.pp_size = distributed.cp_size = distributed.ep_size = 1
    distributed.pp_rank = distributed.cp_rank = distributed.ep_rank = 0
    distributed.pp_group = distributed.cp_group = distributed.ep_group = None
    training.fp8, training.Linear, training.GroupedLinear = False, nn.Linear, GroupedLinear
    reference = reference_class(config, torch.bfloat16).to(device)
    attention = configure_reference_attention(reference, config, reference_contract(args))
    args.output.mkdir(parents=True)
    (args.output / "reference_attention.json").write_text(json.dumps(attention, indent=2) + "\n")
    native_class = native_class or Qwen3OmniMoeThinkerImageModel
    with torch.device(device):
        native = native_class(config, -1).to(torch.bfloat16)
    expected = dict(reference.named_parameters())
    actual = {hf_name(n): p for n, p in native.named_parameters()}
    assert actual.keys() == expected.keys()
    with torch.no_grad():
        for name, p in actual.items():
            p.copy_(expected[name])
    opt = FP32MasterAdamW(actual, lr=1e-4, weight_decay=0)
    ref_opt = FP32MasterAdamW(expected, lr=1e-4, weight_decay=0)
    groups = training_groups(config)
    reports = []
    for step in range(4):
        losses, refs, logits_errors = [], [], []
        count = sum(int((b.labels != -100).sum()) for b in batches)
        # Both forwards remain live before backward: no cross-microbatch cache.
        for batch in batches:
            inputs = {k: v.to(device) for k, v in batch.model_inputs.items()}
            labels = batch.labels.to(device)
            logits = native.reference_forward(inputs["input_ids"], model_context=inputs)
            target = reference(inputs)
            logits_errors.append(error_ratio(logits, target))
            assert logits_errors[-1] < 1e-3
            losses.append(
                F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
                / count
            )
            refs.append(
                F.cross_entropy(target.float().flatten(0, 1), labels.flatten(), reduction="sum")
                / count
            )
        loss, expected_loss = sum(losses), sum(refs)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)
        loss.backward()
        expected_loss.backward()
        errors = {}
        for name, p in actual.items():
            assert p.grad is not None and expected[name].grad is not None, name
            errors[name] = error_ratio(p.grad, expected[name].grad)
            assert errors[name] < 1e-2, (
                name,
                errors[name],
                p.grad.abs().max().item(),
                expected[name].grad.abs().max().item(),
            )
        before = {n: p.detach().clone() for n, p in opt.master_parameters.items()}
        compute_before = {n: p.detach().clone() for n, p in actual.items()}
        opt.step()
        ref_opt.step()
        updates = [n for n, p in opt.master_parameters.items() if not torch.equal(before[n], p)]
        compute_updates = [n for n, p in actual.items() if not torch.equal(compute_before[n], p)]
        modules = opt.update_evidence(before, compute_before, groups)
        reports.append(
            dict(
                step=step,
                loss=loss.item(),
                target_count=count,
                logits_errors=logits_errors,
                gradient_errors=errors,
                updated_parameters=updates,
                updated_parameter_dtype="torch.float32",
                updated_compute_parameters=compute_updates,
                compute_parameter_dtype="torch.bfloat16",
                training_modules=modules,
            )
        )
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
    (args.output / "reference.json").write_text(
        json.dumps(
            dict(status="PASSED", commit=commit(), reference_attention=attention, steps=reports),
            indent=2,
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
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(), "Use Flame H100"
    if args.mode == "reference":
        gpu_reference(args)
    else:
        from tests.acceptance.omni_image_staged import run

        run(args)


if __name__ == "__main__":
    main()
