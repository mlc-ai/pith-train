"""
Inductor options the models merge into the torch.compile options of their compiled stages, chosen per
architecture by the kernel library like its other capability queries.
"""

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["stage_compile_options"]


def stage_compile_options() -> dict:
    """
    A fresh dict of Inductor options for the models' compiled stages. SM90: triton.autotune_pointwise
    off. With it on, Inductor autotunes every 1-D pointwise kernel warm between (XBLOCK 1024, 4 warps)
    and (512, 8 warps) and the pick changes from run to run; the second is 5-9% slower than the first
    when the operands come from DRAM, as they do in training (the SM90 silu_mul a compiled stage
    emits as PyTorch is such a kernel). Off, Inductor emits its one heuristic config, (1024, 4 warps)
    from 128K elements, the launch shape of the Triton silu_mul it replaces. SM100: empty, the stages
    compile as they always have.
    """
    match ARCH_MAJOR:
        case 9:
            return {"triton.autotune_pointwise": False}
        case 10:
            return {}
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
