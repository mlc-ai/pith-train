"""CPU contract of the split optimizer-update verdict; none of it is GPU acceptance."""

import pytest
import torch

from tests.acceptance import omni_update_gate as gate

LR, WEIGHT_DECAY, THRESHOLD = 1e-5, 0.1, 1e-2
SHAPES = {"gate.weight": (1, 128), "proj.weight": (64, 32), "norm.weight": (32,)}


def values(seed=1234):
    generator = torch.Generator().manual_seed(seed)
    return {name: 0.02 * torch.randn(shape, generator=generator) for name, shape in SHAPES.items()}


def gradients(step, seed=99):
    generator = torch.Generator().manual_seed(seed + step)
    result = {
        name: 1e-4 * torch.randn(shape, generator=generator) for name, shape in SHAPES.items()
    }
    # Elements at or near zero, where AdamW's first steps are most sensitive.
    result["gate.weight"][0, :4] = torch.tensor([1e-8, -1e-8, 0.0, 3e-9])
    return result


class Native:
    """A native run: its parameters, its optimizer and the updates it takes."""

    def __init__(self, initial, *, names=None, **settings):
        self.parameters = {
            name: torch.nn.Parameter(value.clone()) for name, value in initial.items()
        }
        names = list(self.parameters) if names is None else names
        self.optimizer = torch.optim.AdamW([self.parameters[name] for name in names], **settings)

    def step(self, grads):
        before = {name: p.detach().clone() for name, p in self.parameters.items()}
        for name, parameter in self.parameters.items():
            parameter.grad = grads[name].clone()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        delta = {
            name: p.detach().float() - before[name].float() for name, p in self.parameters.items()
        }
        return before, delta


def reference_optimizer(**settings):
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], **settings)


def shadow_like_reference(**storage):
    reference = reference_optimizer(lr=LR, weight_decay=WEIGHT_DECAY)
    return gate.ShadowAdamW.like(reference, SHAPES, **storage)


def test_update_vs_reference_update_fails_when_one_gradient_straddles_zero():
    reference_gradients = gradients(0)
    native_gradients = {name: value.clone() for name, value in reference_gradients.items()}
    reference_gradients["gate.weight"][0, 25] = 1e-6
    native_gradients["gate.weight"][0, 25] = -1e-6
    gradient_rows = gate.delta_rows(native_gradients, reference_gradients, THRESHOLD)
    assert all(row["status"] == "PASS" for row in gradient_rows.values())
    assert gradient_rows["gate.weight"]["nse"] < 1e-5
    initial = values()
    native = Native(initial, lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    reference = Native(initial, lr=LR, weight_decay=WEIGHT_DECAY)
    before, native_delta = native.step(native_gradients)
    _, reference_delta = reference.step(reference_gradients)
    # One element of 128 moving about +lr against -lr exceeds 1e-2 on its own.
    end_to_end = gate.delta_rows(native_delta, reference_delta, THRESHOLD)
    assert end_to_end["gate.weight"]["status"] == "FAIL"
    assert end_to_end["gate.weight"]["nse"] > 0.015
    # The native optimizer is right about the gradients it was given.
    shadow = shadow_like_reference().step(before, native_gradients)
    assert gate.optimizer_gate(native_delta, shadow, THRESHOLD)["status"] == "PASS"


@pytest.mark.parametrize(
    "implementation", [dict(fused=True), dict(foreach=True), dict(foreach=False)]
)
def test_correct_native_adamw_passes_for_three_steps(implementation):
    native = Native(values(), lr=LR, weight_decay=WEIGHT_DECAY, **implementation)
    shadow = shadow_like_reference()
    for step in range(3):
        grads = gradients(step)
        before, delta = native.step(grads)
        verdict = gate.optimizer_gate(delta, shadow.step(before, grads), THRESHOLD)
        assert verdict["status"] == "PASS", verdict["failed"]
        assert max(row["nse"] for row in verdict["parameters"].values()) < 1e-6
        assert verdict["controls"] == dict(
            status="PASS",
            covered_parameters=len(SHAPES),
            not_rejected=dict(double_learning_rate=[], parameter_not_updated=[]),
        )


def test_native_learning_rate_fault_fails():
    native = Native(values(), lr=2 * LR, weight_decay=WEIGHT_DECAY, fused=True)
    before, delta = native.step(gradients(0))
    verdict = gate.optimizer_gate(
        delta, shadow_like_reference().step(before, gradients(0)), THRESHOLD
    )
    assert verdict["status"] == "FAIL"
    assert verdict["failed"] == sorted(SHAPES)


def test_parameter_left_out_of_the_native_optimizer_fails():
    names = ["gate.weight", "proj.weight"]
    native = Native(values(), names=names, lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    before, delta = native.step(gradients(0))
    verdict = gate.optimizer_gate(
        delta, shadow_like_reference().step(before, gradients(0)), THRESHOLD
    )
    assert verdict["failed"] == ["norm.weight"]
    assert verdict["parameters"]["norm.weight"]["nse"] == 1.0


def test_moments_lost_between_steps_fail():
    native = Native(values(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    shadow = shadow_like_reference()
    before, delta = native.step(gradients(0))
    assert (
        gate.optimizer_gate(delta, shadow.step(before, gradients(0)), THRESHOLD)["status"] == "PASS"
    )
    native.optimizer.state.clear()  # e.g. a restore that dropped the Adam moments
    before, delta = native.step(gradients(1))
    verdict = gate.optimizer_gate(delta, shadow.step(before, gradients(1)), THRESHOLD)
    assert verdict["status"] == "FAIL"
    assert verdict["failed"] == sorted(SHAPES)


def test_weight_decay_alone_is_below_the_gate_resolution():
    """The stated limit: an lr * weight_decay * |p| difference per element is not claimed."""
    native = Native(values(), lr=LR, weight_decay=0.0, fused=True)
    before, delta = native.step(gradients(0))
    verdict = gate.optimizer_gate(
        delta, shadow_like_reference().step(before, gradients(0)), THRESHOLD
    )
    assert verdict["status"] == "PASS"
    assert max(row["nse"] for row in verdict["parameters"].values()) < 1e-4


def test_controls_cover_only_nonzero_updates_and_need_at_least_one():
    zero = {name: torch.zeros(shape) for name, shape in SHAPES.items()}
    verdict = gate.optimizer_gate(zero, zero, THRESHOLD)
    assert verdict["status"] == "FAIL"
    assert verdict["controls"]["status"] == "FAIL"
    assert verdict["controls"]["covered_parameters"] == 0
    shadow = dict(zero, **{"proj.weight": torch.full((64, 32), 1e-5)})
    verdict = gate.optimizer_gate(dict(shadow), shadow, THRESHOLD)
    assert verdict["status"] == "PASS"
    assert verdict["controls"]["covered_parameters"] == 1
    assert verdict["parameters"]["gate.weight"]["nse"] == 0.0


def test_a_comparison_that_accepts_a_control_fails_the_verdict():
    shadow = {name: torch.full(shape, 1e-5) for name, shape in SHAPES.items()}
    # Twice the update has NSE 0.2, inside a 0.5 threshold, so that control is not rejected.
    verdict = gate.optimizer_gate(shadow, shadow, threshold=0.5)
    assert verdict["status"] == "FAIL"
    assert verdict["failed"] == []
    assert verdict["controls"]["not_rejected"] == dict(
        double_learning_rate=sorted(SHAPES), parameter_not_updated=[]
    )


def test_update_rows_fail_closed():
    rows = gate.delta_rows(
        {
            "missing_reference": torch.ones(2),
            "shape": torch.ones(2),
            "dtype": torch.ones(2),
            "nonfinite": torch.tensor([1.0, float("nan")]),
        },
        {
            "missing_native": torch.ones(2),
            "shape": torch.ones(3),
            "dtype": torch.ones(2, dtype=torch.float64),
            "nonfinite": torch.ones(2),
        },
        THRESHOLD,
    )
    assert {name: row["status"] for name, row in rows.items()} == dict.fromkeys(
        ["dtype", "missing_native", "missing_reference", "nonfinite", "shape"], "FAIL"
    )
    assert rows["missing_native"]["native_present"] is False


def test_shadow_moments_round_trip_a_checkpoint_boundary(tmp_path):
    first = shadow_like_reference()
    native = Native(values(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    for step in range(2):
        before, _ = native.step(gradients(step))
        first.step(before, gradients(step))
    saved = first.state_dict()
    torch.save(saved, tmp_path / "shadow.pt")
    second = shadow_like_reference()
    second.load_state_dict(torch.load(tmp_path / "shadow.pt", weights_only=True))
    before, delta = native.step(gradients(2))
    expected = first.step(before, gradients(2))
    actual = second.step(before, gradients(2))
    assert all(torch.equal(actual[name], expected[name]) for name in SHAPES)
    assert gate.optimizer_gate(delta, actual, THRESHOLD)["status"] == "PASS"
    # The saved snapshot does not alias the live moments.
    state = first.optimizer.state[first.parameters["gate.weight"]]
    assert float(saved["optimizer"]["state"][0]["step"]) == 2 and float(state["step"]) == 3
    assert not torch.equal(saved["optimizer"]["state"][0]["exp_avg"], state["exp_avg"])


def test_shadow_takes_the_reference_settings_only():
    reference = reference_optimizer(lr=3e-4, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.05)
    group = gate.ShadowAdamW.like(reference, SHAPES).optimizer.param_groups[0]
    assert (group["lr"], group["betas"], group["eps"], group["weight_decay"]) == (
        3e-4,
        (0.8, 0.95),
        1e-6,
        0.05,
    )
    assert group["foreach"] is False and not group["fused"]


def test_shadow_rejects_inputs_it_cannot_judge():
    shadow = shadow_like_reference()
    before, grads = values(), gradients(0)
    with pytest.raises(ValueError, match="identities"):
        shadow.step({name: before[name] for name in ("gate.weight", "proj.weight")}, grads)
    with pytest.raises(ValueError, match="needs a gradient"):
        shadow.step(before, dict(grads, **{"norm.weight": None}))
    with pytest.raises(ValueError, match="shape"):
        shadow.step(dict(before, **{"norm.weight": torch.zeros(31)}), grads)
    with pytest.raises(ValueError, match="dtype"):
        shadow.step(before, dict(grads, **{"norm.weight": grads["norm.weight"].double()}))
    with pytest.raises(ValueError, match="Unsupported"):
        gate.ShadowAdamW(SHAPES, lr=LR, fused=True)
    with pytest.raises(TypeError):
        gate.ShadowAdamW.like(torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]), SHAPES)
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))]} for _ in range(2)]
    with pytest.raises(TypeError):
        gate.ShadowAdamW.like(torch.optim.AdamW(groups), SHAPES)
    with pytest.raises(ValueError, match="names"):
        gate.ShadowAdamW({"other.weight": (2,)}, lr=LR).load_state_dict(shadow.state_dict())


def test_bf16_shadow_matches_a_native_single_tensor_adamw_exactly():
    initial = {name: value.bfloat16() for name, value in values().items()}
    native = Native(initial, lr=1e-4, weight_decay=0.1, foreach=False)
    reference = reference_optimizer(lr=1e-4, weight_decay=0.1)
    shadow = gate.ShadowAdamW.like(reference, SHAPES, dtype=torch.bfloat16)
    for step in range(2):
        grads = {name: value.bfloat16() for name, value in gradients(step).items()}
        before, delta = native.step(grads)
        verdict = gate.optimizer_gate(delta, shadow.step(before, grads), THRESHOLD)
        assert verdict["status"] == "PASS"
        assert all(row["nse"] == 0.0 for row in verdict["parameters"].values())
