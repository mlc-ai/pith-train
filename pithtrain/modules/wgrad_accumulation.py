"""
Weight-gradient accumulation into a layer-owned fp32 buffer.

BF16Linear, GroupedLinear and FP8GroupedLinear add their weight gradient straight into the fp32
main_grad buffer instead of writing a bf16 .grad that FSDP2 then adds into its own fp32 buffer. It
also keeps the sum across micro-batches in fp32 under DualPipeV, where FSDP2 would otherwise add
most micro-batches together in bf16. setup_model enables it for every run, BF16 or FP8.

install_wgrad_accumulation connects these layers to FSDP2 after fully_shard by extending two methods
of each parameter group:
- pre_backward allocates and zeroes main_grad at the first backward of the step.
- post_backward, when the group reduces, hands main_grad to FSDP2 as the accumulated gradient of the
  parameter and releases its memory once FSDP2 has copied it into the reduce-scatter input.

main_grad keeps the same tensor object across steps, so compiled graphs that take it as an input
stay valid. Without install, main_grad is None and the layer behaves like its torch.nn version.

install_raw_main_grad does the same for expert weights kept as raw parameters (gpt-oss's
gate_up_proj / down_proj), which modules list in ``fp8_raw_params``: the k-grouped FP8 wgrad (or the
BF16 CuTe experts) add their fp32 result straight into ``main_grad`` (``operators/main_grad.py``)
instead of a bf16 ``.grad`` per micro-batch next to FSDP2's fp32 sum: 2 B/param less (4.5 GiB for
gpt-oss-20b at EP8 on one node), and the per-micro-batch gradients are no longer rounded to bf16
before they are summed.
"""

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

from pithtrain.contexts import training
from pithtrain.kernels import (
    bf16_experts_supported,
    fp8_fused_experts_supported,
    fp8_grouped_wgrad_empty,
)
from pithtrain.operators.grouped_linear import FP8GroupedLinear, GroupedLinear
from pithtrain.operators.linear import BF16Linear
from pithtrain.operators.main_grad import MAIN_GRADS

__all__ = ["install_raw_main_grad", "install_wgrad_accumulation"]


def install_wgrad_accumulation(model: nn.Module) -> None:
    """
    Give every BF16Linear, GroupedLinear and FP8GroupedLinear weight in the model a main_grad,
    after fully_shard.
    """
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        for group in fully_shard.state(module)._fsdp_param_groups:
            assert isinstance(group, FSDPParamGroup)
            pairs = []
            for fsdp_param in group.fsdp_params:
                assert isinstance(fsdp_param, FSDPParam)
                owner = fsdp_param._module_info.module
                if fsdp_param._module_info.param_name != "weight":
                    continue
                if not isinstance(owner, (BF16Linear, GroupedLinear, FP8GroupedLinear)):
                    continue
                size, device = fsdp_param._orig_size, fsdp_param.device
                if isinstance(owner, FP8GroupedLinear):
                    # Laid out as the k-grouped FP8 wgrad accumulates in place (transposed where
                    # it computes the product transposed), not transposed twice per micro-batch.
                    owner.main_grad = fp8_grouped_wgrad_empty(size, device)
                else:
                    owner.main_grad = torch.empty(size, dtype=torch.float32, device=device)
                owner.main_grad.untyped_storage().resize_(0)
                pairs.append((fsdp_param, owner.main_grad))
            if pairs:
                _attach_main_grads(group, pairs)


def _attach_main_grads(group: FSDPParamGroup, pairs: List[Tuple[FSDPParam, torch.Tensor]]) -> None:
    """
    Make the group allocate main_grad at the start of backward and reduce it in place of a .grad.
    """
    orig_pre_backward, orig_post_backward = group.pre_backward, group.post_backward

    def pre_backward(*args: Any) -> None:
        for _, main_grad in pairs:
            if main_grad.untyped_storage().size() == 0:
                main_grad.untyped_storage().resize_(main_grad.numel() * main_grad.element_size())
                main_grad.zero_()
        orig_pre_backward(*args)

    def post_backward(*args: Any) -> None:
        if not group.reduce_grads:
            orig_post_backward(*args)
            return
        # FSDP2 reduces a group in one dtype, so move the bf16 gradients into fp32 first.
        for fsdp_param in group.fsdp_params:
            if hasattr(fsdp_param, "_unsharded_param"):
                fsdp_param.accumulate_unsharded_grad_if_needed()
                fsdp_param.to_accumulated_grad_if_needed()
        # Keep any gradient the weight got outside its layer, e.g. kv_b_proj in MLA ring attention.
        for fsdp_param, main_grad in pairs:
            if main_grad.untyped_storage().size() > 0:
                if fsdp_param.unsharded_accumulated_grad is not None:
                    main_grad.add_(fsdp_param.unsharded_accumulated_grad)
                # A transposed main_grad is made contiguous once per step for the reduce-scatter.
                fsdp_param.unsharded_accumulated_grad = main_grad.contiguous()
        orig_post_backward(*args)
        # FSDP2 has copied main_grad into the reduce-scatter input on this stream.
        for _, main_grad in pairs:
            main_grad.untyped_storage().resize_(0)

    group.pre_backward = pre_backward
    group.post_backward = post_backward


def install_raw_main_grad(model: nn.Module) -> None:
    """Give every raw parameter named in a module's ``fp8_raw_params`` an fp32 main_grad, after
    fully_shard (alongside install_wgrad_accumulation). Only the fused FP8 expert path
    (operators/fp8_experts.py) and the BF16 CuTe experts (operators/bf16_experts.py) write it, so
    nothing is installed otherwise."""
    if not ((training.fp8 and fp8_fused_experts_supported()) or (not training.fp8 and bf16_experts_supported())):  # fmt: skip
        return
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        for group in fully_shard.state(module)._fsdp_param_groups:
            pairs = []
            for fsdp_param in group.fsdp_params:
                owner = fsdp_param._module_info.module
                name = fsdp_param._module_info.param_name
                if name not in getattr(owner, "fp8_raw_params", ()):
                    continue
                mg = torch.empty(fsdp_param._orig_size, dtype=torch.float32, device=fsdp_param.device)  # fmt: skip
                mg.untyped_storage().resize_(0)
                owner.__dict__.setdefault(MAIN_GRADS, {})[name] = mg
                pairs.append((fsdp_param, mg))
            if pairs:
                _attach_main_grads(group, pairs)
