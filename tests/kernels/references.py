"""
Error metrics shared by the spec-driven kernel tests.
"""

import torch


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    1 - 2 <x, y> / (|x|^2 + |y|^2) in float64.
    """
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum()
    return 0.0 if den == 0 else float(1 - 2 * (x * y).sum() / den)


def rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    """
    |out - ref| / |ref| in float64 (0 when both are zero).
    """
    out, ref = out.double(), ref.double()
    den = ref.norm()
    num = (out - ref).norm()
    return float(num / den) if den > 0 else float(num)


def worst_ratio(out: torch.Tensor, ref: torch.Tensor, tol: torch.Tensor) -> float:
    """
    max |out - ref| / tol elementwise (<= 1 passes); tol broadcasts against ref.
    """
    if out.numel() == 0:
        return 0.0
    err = (out.double() - ref.double()).abs()
    return float((err / (tol.double() + 1e-30)).max())
