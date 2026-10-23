"""
Training runtime state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from pithtrain.operators import grouped_linear, linear
    from pithtrain.pipeline import DualPipeV

PARAM_DTYPE = torch.bfloat16
"""
Parameter compute and activation dtype.

Pipeline parallelism cuts at layer boundaries, so the tensors crossing between stages are layer
inputs and carry this dtype. DualPipeV allocates its receive buffers with it.
"""

fp8: bool
"""
Whether layers are built for FP8 compute, False for BF16.

This and the two class bindings below are read from module constructors and kernels that hold no
configuration object, so they live here rather than being passed down.
"""

Linear: type[linear.BF16Linear | linear.FP8Linear]
"""
The dense linear class the models construct.
"""

GroupedLinear: type[grouped_linear.GroupedLinear | grouped_linear.FP8GroupedLinear]
"""
The grouped linear class the routed experts construct.
"""

model: DualPipeV
"""
The pipeline engine for this rank, wrapping the two V-shaped chunks it owns.
"""

optimizers: tuple[Optimizer, ...]
"""
The optimizers stepping this model, one per parameter class under Muon.
"""

schedulers: tuple[LRScheduler, ...]
"""
One learning-rate scheduler per optimizer, in the same order.
"""

current_microbatch: int | None = None
"""
Index into the micro-batch list of the step in flight, or None outside a step.

Routing replay reads it to find which micro-batch a gate call belongs to, which the model cannot
tell from the layer alone: DualPipeV interleaves a rank's two chunks across micro-batches. It is
set before each forward only, which suffices because the backward replays the saved graph without
re-entering module code.
"""

offloaded: bool = False
"""
True between offload_training_state and reload_training_state.

Tensors keep their shapes over zero-byte device storages, so model.step, save_checkpoint,
load_checkpoint and iter_canonical_parameters reject this state.
"""
