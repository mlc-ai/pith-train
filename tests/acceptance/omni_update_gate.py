"""Optimizer-update verdict shared by the Omni acceptance harnesses.

AdamW's first steps move every element by about lr, whatever the size of its
gradient: step one is -lr * g / (|g| + eps) - lr * weight_decay * p. Comparing a
native update with an independent reference's update therefore amplifies gradient
differences that the gradient gates accept. Where an element's two gradients
straddle zero, it moves by about +lr on one side and -lr on the other; in a tensor
of N elements that one element adds about 2 / N to the NSE, 0.0156 for N = 128,
over the 1e-2 threshold with the model and the optimizer both correct. At f828ed0
the Talker G2 gate measured this: gradient NSE at most 2.9e-6 for all 87
parameters, update NSE 0.0135 for shared_expert_gate.weight [1, 128], 99.9999% of
it from two elements whose gradients are near 1e-7.

The update verdict is therefore split, with the thresholds unchanged:

- gradients: each harness's own native-vs-reference gradient gates, unchanged;
- optimizer: ShadowAdamW applies the independent reference AdamW's settings to the
  native run's pre-step parameters and clipped gradients, its moments fed by the
  same native gradient stream, and optimizer_gate requires the native update to
  match it.

Each harness still computes the end-to-end comparison (the reference optimizer on
the reference gradients) and reports it with END_TO_END_ROLE; it no longer decides
PASS. optimizer_gate also requires its comparison to reject two controls wherever
the shadow update is nonzero: twice the shadow update (twice the learning rate,
since AdamW's decay and Adam terms are both linear in lr given the moments) and no
update at all. A weight-decay-only difference, of order lr * weight_decay * |p| per
element, is below a 1e-2 NSE resolution and is not claimed.
"""

import torch

ADAMW_SETTINGS = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")
END_TO_END_ROLE = (
    "diagnostic only: the reference AdamW on the reference gradients amplifies gradient "
    "differences below the gradient gate (tests/acceptance/omni_update_gate.py)"
)


def delta_rows(actual, expected, threshold):
    """NSE of every named update; a missing, reshaped, retyped or nonfinite one fails."""
    rows = {}
    for name in sorted(actual.keys() | expected.keys()):
        left, right = actual.get(name), expected.get(name)
        if left is None or right is None:
            rows[name] = dict(
                status="FAIL",
                reason="missing update",
                native_present=left is not None,
                reference_present=right is not None,
            )
            continue
        if left.shape != right.shape or left.dtype != right.dtype:
            rows[name] = dict(status="FAIL", reason="update shape or dtype differs")
            continue
        a = left.detach().to("cpu", torch.float64)
        b = right.detach().to("cpu", torch.float64)
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            rows[name] = dict(status="FAIL", reason="nonfinite update")
            continue
        denominator = float((a.square() + b.square()).sum())
        error = 0.0 if denominator == 0 else float((a - b).square().sum()) / denominator
        rows[name] = dict(
            status="PASS" if error < threshold else "FAIL",
            nse=error,
            native_norm=float(a.norm()),
            reference_norm=float(b.norm()),
            diff_maxabs=float((a - b).abs().max()) if a.numel() else 0.0,
            shape=list(left.shape),
        )
    return rows


def optimizer_gate(native, shadow, threshold):
    """The native update against ShadowAdamW's on the same gradients, and both controls."""
    rows = delta_rows(native, shadow, threshold)
    covered = sorted(name for name, value in shadow.items() if torch.count_nonzero(value))
    doubled = delta_rows(native, {name: 2 * value for name, value in shadow.items()}, threshold)
    unchanged = delta_rows(
        {name: torch.zeros_like(value) for name, value in shadow.items()}, shadow, threshold
    )
    not_rejected = dict(
        double_learning_rate=[name for name in covered if doubled[name]["status"] == "PASS"],
        parameter_not_updated=[name for name in covered if unchanged[name]["status"] == "PASS"],
    )
    controls = dict(
        status="PASS" if covered and not any(not_rejected.values()) else "FAIL",
        covered_parameters=len(covered),
        not_rejected=not_rejected,
    )
    failed = sorted(name for name, row in rows.items() if row["status"] != "PASS")
    return dict(
        status="PASS" if rows and not failed and controls["status"] == "PASS" else "FAIL",
        threshold=threshold,
        comparison="native update vs the reference AdamW settings on the native gradients",
        expected_count=len(rows),
        failed=failed,
        parameters=rows,
        controls=controls,
    )


class ShadowAdamW:
    """torch.optim.AdamW on the single-tensor path, fed another run's parameters and gradients.

    Every step starts from the caller's pre-step parameters and gradients, so only the
    moments and step counts persist, and they follow the gradients fed in. The parameters,
    and so the moments, take the caller's storage dtype on ``device``.
    """

    def __init__(self, shapes, *, dtype=torch.float32, device="cpu", **settings):
        unknown = sorted(set(settings) - set(ADAMW_SETTINGS))
        if unknown:
            raise ValueError(f"Unsupported AdamW settings: {unknown}")
        if not shapes:
            raise ValueError("ShadowAdamW needs at least one parameter")
        self.parameters = {
            name: torch.nn.Parameter(torch.zeros(tuple(shape), dtype=dtype, device=device))
            for name, shape in shapes.items()
        }
        self.optimizer = torch.optim.AdamW(
            list(self.parameters.values()), foreach=False, **settings
        )

    @classmethod
    def like(cls, reference, shapes, **storage):
        """The settings of the independent reference AdamW, never of the optimizer under test."""
        if type(reference) is not torch.optim.AdamW or len(reference.param_groups) != 1:
            raise TypeError("ShadowAdamW copies a single-group torch.optim.AdamW reference")
        group = reference.param_groups[0]
        return cls(shapes, **storage, **{key: group[key] for key in ADAMW_SETTINGS})

    @torch.no_grad()
    def step(self, before, gradients):
        """The update this optimizer takes from ``before`` with ``gradients``, as FP32 on CPU."""
        if not before.keys() == gradients.keys() == self.parameters.keys():
            raise ValueError("ShadowAdamW parameter identities differ")
        for name, parameter in self.parameters.items():
            value, gradient = before[name], gradients[name]
            if gradient is None:
                raise ValueError(f"ShadowAdamW needs a gradient for {name}")
            if value.shape != parameter.shape or gradient.shape != parameter.shape:
                raise ValueError(f"ShadowAdamW shape differs: {name}")
            if value.dtype != parameter.dtype or gradient.dtype != parameter.dtype:
                raise ValueError(f"ShadowAdamW dtype differs: {name}")
            parameter.copy_(value)
            parameter.grad = gradient.detach().to(parameter.device, copy=True)
        self.optimizer.step()
        delta = {}
        for name, parameter in self.parameters.items():
            delta[name] = parameter.detach().float().cpu() - before[name].detach().float().cpu()
            parameter.grad = None
        return delta

    def state_dict(self):
        """The moments and step counts, cloned, with the parameter names they belong to."""
        return dict(names=list(self.parameters), optimizer=_clone(self.optimizer.state_dict()))

    def load_state_dict(self, state):
        if list(state["names"]) != list(self.parameters):
            raise ValueError("ShadowAdamW checkpoint parameter names differ")
        # Optimizer.load_state_dict keeps tensors that already match; clone to avoid aliasing.
        self.optimizer.load_state_dict(_clone(state["optimizer"]))


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return value
