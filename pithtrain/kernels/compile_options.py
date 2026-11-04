"""
Inductor options the models merge into the torch.compile options of their compiled stages, chosen
per architecture by the kernel library like its capability queries.
"""

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["stage_compile_options"]


def stage_compile_options() -> dict:
    """
    A fresh dict of Inductor options for the models' compiled stages. SM90: Inductor's pointwise
    autotuning off (triton.autotune_pointwise), so each 1-D pointwise kernel gets the one heuristic
    launch config instead of a pick that can change from run to run. SM100: empty.
    """
    match ARCH_MAJOR:
        case 9:
            return {"triton.autotune_pointwise": False}
        case 10:
            return {}
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
