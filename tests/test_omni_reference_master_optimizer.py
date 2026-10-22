"""CPU update semantics; these contracts do not execute CUDA model kernels."""

import ast
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.acceptance import omni_audio_reference as audio_reference
from tests.acceptance import omni_image_training as image_reference
from tests.acceptance.omni_image_evidence import module_update_evidence, snapshot
from tests.acceptance.omni_image_training import FP32MasterAdamW


def test_bf16_norm_update_loss_is_reproduced_without_changing_learning_rate():
    parameter = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0)
    before = snapshot({"model.norm.weight": parameter})
    for _ in range(4):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    assert torch.equal(parameter, before["model.norm.weight"])
    with pytest.raises(AssertionError, match="No optimizer update component: model.norm."):
        module_update_evidence({"model.norm.weight": parameter}, before, ("model.norm.",))


def test_master_weights_and_adam_state_match_independent_fp32_updates():
    compute = {
        "model.norm.weight": nn.Parameter(torch.ones(4, dtype=torch.bfloat16)),
        "visual.proj.weight": nn.Parameter(torch.full((4,), 0.125, dtype=torch.bfloat16)),
    }
    optimizer = FP32MasterAdamW(compute, lr=1e-4, weight_decay=0)
    independent = {name: nn.Parameter(p.detach().float()) for name, p in compute.items()}
    reference = torch.optim.AdamW(independent.values(), lr=1e-4, weight_decay=0)
    groups = ("model.norm.", "visual.proj.")
    for step in range(32):
        before, compute_before = snapshot(optimizer.master_parameters), snapshot(compute)
        for index, (name, parameter) in enumerate(compute.items()):
            parameter.grad = torch.full_like(parameter, (index + 1) * (1 + step % 3) / 3)
            independent[name].grad = parameter.grad.float()
        optimizer.step()
        reference.step()
        evidence = optimizer.update_evidence(before, compute_before, groups)
        assert all(row["update_squared_sum"] > 0 for row in evidence.values())
        if step < 4:
            assert evidence["model.norm."]["compute_update_squared_sum"] == 0
        for name, master in optimizer.master_parameters.items():
            assert master.dtype == torch.float32 and master.grad.dtype == torch.float32
            torch.testing.assert_close(master, independent[name], rtol=0, atol=0)
            torch.testing.assert_close(compute[name], master.to(torch.bfloat16), rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                state = optimizer.optimizer.state[master][key]
                assert state.dtype == torch.float32
                torch.testing.assert_close(
                    state, reference.state[independent[name]][key], rtol=0, atol=0
                )
        optimizer.zero_grad(set_to_none=step % 2 == 0)
        for parameter in (*compute.values(), *optimizer.master_parameters.values()):
            if step % 2 == 0:
                assert parameter.grad is None
            else:
                assert torch.count_nonzero(parameter.grad) == 0
    assert not torch.equal(compute["model.norm.weight"], torch.ones(4, dtype=torch.bfloat16))


@pytest.mark.parametrize("bad", ["missing", "zero", "nan"])
def test_master_update_evidence_does_not_accept_invalid_component_gradients(bad):
    parameter = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW({"model.norm.weight": parameter}, lr=1e-4, weight_decay=0)
    before, compute_before = (
        snapshot(optimizer.master_parameters),
        snapshot({"model.norm.weight": parameter}),
    )
    parameter.grad = {
        "missing": None,
        "zero": torch.zeros_like(parameter),
        "nan": torch.full_like(parameter, float("nan")),
    }[bad]
    optimizer.step()
    with pytest.raises(AssertionError, match="gradient"):
        optimizer.update_evidence(before, compute_before, ("model.norm.",))


@pytest.mark.parametrize("mode", ["image_video", "audio"])
@pytest.mark.parametrize("bad_gradient", [False, True])
def test_actual_reference_control_flow_checks_gradients_before_master_updates(
    tmp_path, mode, bad_gradient
):
    """Run the real gate body with tiny CPU modules replacing only CUDA setup."""

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    class Native(nn.Module):
        def __init__(self, config, phase):
            super().__init__()
            self.norm = Norm()

        def reference_forward(self, ids, model_context):
            value = self.norm.weight + model_context.get("input_features", torch.zeros(1)).sum()
            if bad_gradient:
                value = value + value - value.detach()
            return torch.stack((value, -value), dim=-1).expand(*ids.shape, 2)

    class Reference(nn.Module):
        def __init__(self, config, dtype=None):
            super().__init__()
            self.model = nn.Module()
            self.model.config = SimpleNamespace(_attn_implementation="eager")
            self.model.norm = Norm()

        def forward(self, inputs):
            value = self.model.norm.weight + inputs.get("input_features", torch.zeros(1)).sum()
            return torch.stack((value, -value), dim=-1).expand(*inputs["input_ids"].shape, 2)

    class CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def device(self, *args):
            return torch.device("cpu")

    class ObservedOptimizer(FP32MasterAdamW):
        steps = 0

        def step(self):
            type(self).steps += 1
            super().step()

    module = image_reference if mode == "image_video" else audio_reference
    function = module.gpu_reference if mode == "image_video" else module.text_gate
    tree = ast.parse(inspect.getsource(function))
    tree.body[0].body = [
        node for node in tree.body[0].body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    namespace = dict(
        vars(module),
        torch=CpuTorch(),
        distributed=SimpleNamespace(),
        training=SimpleNamespace(),
        GroupedLinear=nn.Linear,
        Qwen3OmniMoeThinkerImageModel=Native,
        Qwen3OmniMoeThinkerAudioModel=Native,
        HFReference=Reference,
        FP32MasterAdamW=ObservedOptimizer,
        training_groups=lambda config: ("model.norm.",),
    )
    exec(compile(tree, "reference-gate-cpu-contract", "exec"), namespace)
    inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    labels = torch.zeros(1, 2, dtype=torch.long)
    if mode == "audio":

        class CopyOnTransfer(torch.Tensor):
            # Real CPU -> CUDA .to() allocates a fresh leaf on every gate step.
            def to(self, *args, **kwargs):
                return self.as_subclass(torch.Tensor).clone()

        inputs["input_features"] = torch.zeros(1, dtype=torch.bfloat16).as_subclass(CopyOnTransfer)

    def execute():
        if mode == "audio":
            return namespace["text_gate"](object(), [(inputs, labels)], torch.device("cpu"))
        output = tmp_path / "report"
        batch = SimpleNamespace(model_inputs=inputs, labels=labels)
        namespace["gpu_reference"](
            SimpleNamespace(output=output, dataset=tmp_path),
            source_inputs_fn=lambda _: (None, None, [batch], object()),
            reference_class=Reference,
            native_class=Native,
        )
        return json.loads((output / "reference.json").read_text())["steps"]

    if bad_gradient:
        error = AssertionError if mode == "image_video" else audio_reference.GradientParityError
        with pytest.raises(error):
            execute()
        assert ObservedOptimizer.steps == 0
    else:
        reports = execute()
        assert len(reports) == 4 and ObservedOptimizer.steps == 8
        for report in reports:
            row = report["training_modules"]["model.norm."]
            assert row["update_squared_sum"] > 0
            assert row["compute_update_squared_sum"] == 0
            assert report["updated_parameter_dtype"] == "torch.float32"
            assert report["updated_compute_parameters"] == []
