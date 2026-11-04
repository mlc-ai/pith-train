"""
fp32 main_grad of expert weights kept as raw parameters (gpt-oss's gate_up_proj / down_proj).

modules/wgrad_accumulation.install_raw_main_grad installs the buffers; the expert ops that write a
raw weight's gradient read them with raw_main_grad. The buffers hang off the module in its __dict__
under MAIN_GRADS (not parameters, buffers or submodules).
"""

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["MAIN_GRADS", "raw_main_grad"]

MAIN_GRADS = "_raw_main_grads"


def raw_main_grad(module: nn.Module, name: str) -> Optional[torch.Tensor]:
    """
    The returned tensor's storage is only allocated from the first backward of a step on, so
    writers check it before writing.
    """
    return module.__dict__.get(MAIN_GRADS, {}).get(name)
