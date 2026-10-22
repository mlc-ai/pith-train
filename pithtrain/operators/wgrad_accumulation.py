"""
Weight-gradient accumulation into a layer-owned fp32 buffer.

BF16Linear and GroupedLinear add their weight gradient straight into the fp32 main_grad buffer
instead of writing a bf16 .grad that FSDP2 then adds into its own fp32 buffer. It also keeps the sum
across micro-batches in fp32 under DualPipeV, where FSDP2 would otherwise add most micro-batches
together in bf16. setup_model enables it for every BF16 run.

install_wgrad_accumulation connects these layers to FSDP2 after fully_shard by extending two methods
of each parameter group:
- pre_backward allocates main_grad and installs hooks that promote ordinary autograd gradients
  after each accumulation, before another micro-batch can add into the bf16 leaf's .grad.
- post_backward, when the group reduces, hands main_grad to FSDP2 as the accumulated gradient of the
  parameter and releases its memory once FSDP2 has copied it into the reduce-scatter input.

main_grad keeps the same tensor object across steps, so compiled graphs that take it as an input
stay valid. Without install, main_grad is None and the layer behaves like its torch.nn version.
"""

from functools import partial
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

from pithtrain.operators.grouped_linear import GroupedLinear
from pithtrain.operators.linear import BF16Linear

__all__ = ["install_wgrad_accumulation"]


def install_wgrad_accumulation(model: nn.Module) -> None:
    """
    Give linear weights main_grad buffers and accumulate other autograd gradients in fp32.
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
                if not isinstance(owner, (BF16Linear, GroupedLinear)):
                    continue
                size, device = fsdp_param._orig_size, fsdp_param.device
                owner.main_grad = torch.empty(size, dtype=torch.float32, device=device)
                owner.main_grad.untyped_storage().resize_(0)
                pairs.append((fsdp_param, owner.main_grad))
            _attach_main_grads(group, pairs)


def _accumulate_autograd_grad(fsdp_param: FSDPParam, parameter: torch.Tensor) -> None:
    # FSDP's post-backward hook need not run after every manually scheduled backward.
    # Empty the compute leaf at its autograd boundary, preserving reduce_dtype accumulation.
    fsdp_param.accumulate_unsharded_grad_if_needed()
    fsdp_param.to_accumulated_grad_if_needed()


def _attach_main_grads(group: FSDPParamGroup, pairs: List[Tuple[FSDPParam, torch.Tensor]]) -> None:
    """
    Make the group allocate main_grad at the start of backward and reduce it in place of a .grad.
    """
    orig_pre_backward, orig_post_backward = group.pre_backward, group.post_backward
    hooked_parameters = {}

    def pre_backward(*args: Any) -> None:
        for _, main_grad in pairs:
            if main_grad.untyped_storage().size() == 0:
                main_grad.untyped_storage().resize_(main_grad.numel() * main_grad.element_size())
                main_grad.zero_()
        orig_pre_backward(*args)
        for fsdp_param in group.fsdp_params:
            parameter = fsdp_param.unsharded_param
            if parameter.requires_grad and hooked_parameters.get(fsdp_param) is not parameter:
                parameter.register_post_accumulate_grad_hook(
                    partial(_accumulate_autograd_grad, fsdp_param)
                )
                hooked_parameters[fsdp_param] = parameter

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
                fsdp_param.unsharded_accumulated_grad = main_grad
        orig_post_backward(*args)
        # FSDP2 has copied main_grad into the reduce-scatter input on this stream.
        for _, main_grad in pairs:
            main_grad.untyped_storage().resize_(0)

    group.pre_backward = pre_backward
    group.post_backward = post_backward
