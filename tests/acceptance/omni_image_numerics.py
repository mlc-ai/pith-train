"""Bounded image numeric diagnosis; never substitutes for the acceptance suite.

Records every gradient failure and free-running router divergence. The optional
visual-eager candidate changes only native vision attention for diagnosis; it
does not replay routes, freeze features, skip gradients, or change thresholds.
"""

import argparse
import copy
import json
import os
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

from tests.acceptance.omni_image_training import (
    HFReference,
    commit,
    error_ratio,
    hf_name,
    source_inputs,
)


def eager_visual_forward(self, hidden_states, lengths, position_embeddings):
    """Candidate: preserve HF eager BF16 score/softmax/value rounding boundaries."""
    q, k, v = self.qkv(hidden_states).reshape(-1, 3, self.num_heads, self.head_dim).unbind(1)
    cos, sin = (value[:, None].float() for value in position_embeddings)

    def rotate(value):
        dtype = value.dtype
        first, second = value.float().chunk(2, dim=-1)
        return (value.float() * cos + torch.cat((-second, first), -1) * sin).to(dtype)

    q, k = rotate(q), rotate(k)
    outputs = []
    for queries, keys, values in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
        queries, keys, values = (x.transpose(0, 1) for x in (queries, keys, values))
        scores = (queries @ keys.transpose(-2, -1)) * self.head_dim**-0.5
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(queries.dtype)
        outputs.append((probabilities @ values).transpose(0, 1))
    return self.proj(torch.cat(outputs).reshape(hidden_states.shape))


def eager_text_forward(self, hidden_states, rotary_posemb, cu_seqlens=None):
    """Diagnostic only: HF eager arithmetic in place of native FA4 attention."""
    assert cu_seqlens is None
    b, length, _ = hidden_states.shape
    q = self.q_norm(self.q_proj(hidden_states).view(b, length, self.num_heads, self.head_dim))
    k = self.k_norm(self.k_proj(hidden_states).view(b, length, self.num_kv_heads, self.head_dim))
    v = self.v_proj(hidden_states).view(b, length, self.num_kv_heads, self.head_dim)
    cos, sin = (value.unsqueeze(2) for value in rotary_posemb)
    q = q * cos + self.rotate_half(q) * sin
    k = k * cos + self.rotate_half(k) * sin
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    k, v = (x.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1) for x in (k, v))
    scores = (q @ k.transpose(-2, -1)) * self.scaling
    mask = torch.full(
        (length, length), torch.finfo(q.dtype).min, dtype=q.dtype, device=q.device
    ).triu(1)
    probabilities = (scores + mask).softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    output = (probabilities @ v).transpose(1, 2).contiguous()
    return self.o_proj(output.reshape(b, length, -1))


def local_expert_checks(native, reference, snapshots):
    """Identical actual normalized inputs, weights, and free routing; local kernel diagnosis."""
    reports = {}
    for index, native_layer in enumerate(native.layers.values()):
        left = copy.deepcopy(native_layer.mlp)
        right = copy.deepcopy(reference.model.layers[index].mlp)
        for model in (left, right):
            for module in model.modules():
                module._forward_hooks.clear()
                module._forward_pre_hooks.clear()
        source = snapshots[f"layer.{index}.router"]["input"]
        a = source.detach().clone().unsqueeze(0).requires_grad_()
        b = source.detach().clone().unsqueeze(0).requires_grad_()
        output = left.reference_forward(a)
        target = right(b)
        probe = torch.linspace(-1, 1, output.numel(), device=output.device).reshape_as(output)
        params, expected = dict(left.named_parameters()), dict(right.named_parameters())
        assert params.keys() == expected.keys()
        (output.float() * probe).sum().backward()
        (target.float() * probe).sum().backward()
        gradients = [a.grad, *(p.grad for p in params.values())]
        targets = [b.grad, *(p.grad for p in expected.values())]
        assert all(value is not None for value in gradients + targets)
        reports[str(index)] = dict(
            output_error=error_ratio(output, target),
            gradient_errors={
                name: error_ratio(actual, expected)
                for name, actual, expected in zip(
                    ["input", *params], gradients, targets, strict=True
                )
            },
        )
    return reports


def capture_modules(model, reference):
    """Hooks observe tensors only; neither their values nor route selection changes."""
    snapshots, handles = {}, []
    layers = model.model.layers if reference else model.layers.values()

    def norm_hook(name):
        def capture(_module, inputs):
            snapshots[name] = inputs[0].detach()

        return capture

    def router_hook(name):
        def capture(_module, inputs, outputs):
            indexes, weights = (outputs[2], outputs[1]) if reference else outputs[:2]
            snapshots[name] = dict(
                input=inputs[0].detach().reshape(-1, inputs[0].shape[-1]),
                indexes=indexes.detach(),
                weights=weights.detach(),
            )
            if reference:
                full = outputs[0].detach().float().softmax(dim=-1)
                ordered = full.sort(dim=-1, descending=True).values
                snapshots[name]["margin"] = (
                    ordered[:, indexes.shape[-1] - 1] - ordered[:, indexes.shape[-1]]
                )

        return capture

    def vision_hook(_module, _inputs, output):
        snapshots["visual"] = [
            v.detach()
            for v in (output.last_hidden_state, output.pooler_output, *output.deepstack_features)
        ]

    handles.append(model.visual.register_forward_hook(vision_hook))
    for index, layer in enumerate(layers):
        handles.append(
            layer.input_layernorm.register_forward_pre_hook(norm_hook(f"layer.{index}.input"))
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                norm_hook(f"layer.{index}.post_attention")
            )
        )
        handles.append(layer.mlp.gate.register_forward_hook(router_hook(f"layer.{index}.router")))
    return snapshots, handles


def compare_snapshots(actual, expected):
    result = {}
    for name, value in actual.items():
        target = expected[name]
        if name == "visual":
            result[name] = [error_ratio(a, b) for a, b in zip(value, target, strict=True)]
        elif name.endswith(".router"):
            same = (value["indexes"].sort(-1).values == target["indexes"].sort(-1).values).all(-1)
            result[name] = dict(
                hidden_error=error_ratio(value["input"], target["input"]),
                weights_error=error_ratio(value["weights"], target["weights"]),
                changed_tokens=int((~same).sum()),
                tokens=len(same),
                min_hf_margin=target["margin"].min().item(),
                changed_hf_margins=target["margin"][~same].tolist(),
                changed_token_indexes=(~same).nonzero().flatten().tolist(),
            )
        else:
            result[name] = error_ratio(value, target)
    return result


def probe(args, variant):
    from pithtrain.contexts import distributed, training
    from pithtrain.models.qwen3_omni_image import Qwen3OmniMoeThinkerImageModel
    from pithtrain.operators.grouped_linear import GroupedLinear

    _, _, batches, config = source_inputs(args.dataset)
    device = torch.device("cuda", 0)
    distributed.device = device
    distributed.pp_size = distributed.cp_size = distributed.ep_size = 1
    distributed.pp_rank = distributed.cp_rank = distributed.ep_rank = 0
    distributed.pp_group = distributed.cp_group = distributed.ep_group = None
    training.fp8, training.Linear, training.GroupedLinear = False, nn.Linear, GroupedLinear
    reference = HFReference(config, torch.bfloat16).to(device)
    with torch.device(device):
        native = Qwen3OmniMoeThinkerImageModel(config, -1).to(torch.bfloat16)
    expected = dict(reference.named_parameters())
    actual = {hf_name(n): p for n, p in native.named_parameters()}
    assert actual.keys() == expected.keys()
    with torch.no_grad():
        for name, p in actual.items():
            p.copy_(expected[name])
    if variant in ("visual-eager", "all-eager"):
        for block in native.visual.blocks:
            block.attn.forward = MethodType(eager_visual_forward, block.attn)
    if variant in ("text-eager", "all-eager"):
        for layer in native.layers.values():
            layer.self_attn.forward = MethodType(eager_text_forward, layer.self_attn)
    snapshots, handles = capture_modules(native, False)
    ref_snapshots, ref_handles = capture_modules(reference, True)
    opt = torch.optim.AdamW(native.parameters(), lr=1e-4, weight_decay=0)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=0)
    reports = []
    expert_checks = None
    count = sum(int((b.labels != -100).sum()) for b in batches)
    for step in range(args.steps):
        losses, refs, logits_errors, layers = [], [], [], []
        for batch in batches:
            inputs = {k: v.to(device) for k, v in batch.model_inputs.items()}
            labels = batch.labels.to(device)
            logits = native.reference_forward(inputs["input_ids"], model_context=inputs)
            target = reference(inputs)
            logits_errors.append(error_ratio(logits, target))
            layers.append(compare_snapshots(snapshots, ref_snapshots))
            if expert_checks is None:
                expert_checks = local_expert_checks(native, reference, ref_snapshots)
                print(
                    json.dumps(dict(variant=variant, local_expert_checks=expert_checks)), flush=True
                )
            losses.append(
                F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
                / count
            )
            refs.append(
                F.cross_entropy(target.float().flatten(0, 1), labels.flatten(), reduction="sum")
                / count
            )
        loss, expected_loss = sum(losses), sum(refs)
        loss.backward()
        expected_loss.backward()
        errors, missing = {}, []
        for name, p in actual.items():
            if p.grad is None or expected[name].grad is None:
                missing.append(name)
            else:
                errors[name] = error_ratio(p.grad, expected[name].grad)
        failures = {name: value for name, value in errors.items() if value >= 1e-2}
        report = dict(
            step=step,
            loss=loss.item(),
            expected_loss=expected_loss.item(),
            loss_close=torch.isclose(loss, expected_loss, rtol=1e-3, atol=1e-3).item(),
            logits_errors=logits_errors,
            gradient_errors=errors,
            gradient_failures=failures,
            missing_gradients=missing,
            layers=layers,
        )
        reports.append(report)
        print(
            json.dumps(
                dict(
                    variant=variant,
                    step=step,
                    failures=failures,
                    missing_gradients=missing,
                    logits_errors=logits_errors,
                )
            ),
            flush=True,
        )
        opt.step()
        ref_opt.step()
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
    for handle in handles + ref_handles:
        handle.remove()
    return dict(
        status="DIAGNOSTIC_ONLY",
        variant=variant,
        commit=commit(),
        steps=reports,
        local_expert_checks=expert_checks,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("baseline", "visual-eager", "text-eager", "all-eager"), required=True
    )
    parser.add_argument("--steps", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    assert not args.output.exists(), "Use a new diagnostic evidence file"
    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1"
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name()
    report = probe(args, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
