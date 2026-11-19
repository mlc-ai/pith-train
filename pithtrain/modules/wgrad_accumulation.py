"""
Weight-gradient accumulation into a layer-owned fp32 buffer.

BF16Linear, GroupedLinear and FP8GroupedLinear add their weight gradient straight into the fp32
main_grad buffer instead of writing a bf16 gradient per micro-batch that autograd then adds into
FSDP2's fp32 .grad: the GEMM accumulates in fp32, so no micro-batch's gradient is rounded to bf16
and no bf16 temporary is allocated. setup_model enables it for every run, BF16 or FP8.

install_wgrad_accumulation connects these layers to FSDP2 after fully_shard by extending two methods
of each parameter group:
- pre_backward allocates and zeroes main_grad at the first backward of the step.
- post_backward, when the group reduces, hands main_grad to FSDP2 as the gradient of the parameter
  and releases its memory once FSDP2 has copied it into the reduce-scatter input.

main_grad keeps the same tensor object across steps, so compiled graphs that take it as an input
stay valid. Without install, main_grad is None and the layer behaves like its torch.nn version.

install_raw_main_grad does the same for expert weights kept as raw parameters (gpt-oss's
gate_up_proj / down_proj), which modules list in ``fp8_raw_params``: the k-grouped FP8 wgrad (or the
BF16 CuTe experts) add their fp32 result straight into ``main_grad`` (``operators/main_grad.py``)
instead of a bf16 ``.grad`` per micro-batch next to FSDP2's fp32 sum: 2 B/param less (4.5 GiB for
gpt-oss-20b at EP8 on one node), and the per-micro-batch gradients are no longer rounded to bf16
before they are summed.

install_side_grads keeps every other gradient off the unsharded parameter during the step. Torch
2.15's FSDP2 accumulates .grad in the reduce dtype (fp32) on the bf16 unsharded parameter, and the
pipeline defers the reduce to the end of the step, so a parameter would carry an fp32 .grad into
the next micro-batch's compiled forward, which Dynamo rejects ("dtype mismatch between tensor and
its gradient"). A post-accumulate-grad hook moves such a .grad into an fp32 sum and clears it; the
group's post_backward hands the sum back as .grad when the group reduces, before the main_grad hook
above (which adds it into main_grad) and FSDP2's reduce-scatter.
"""

from typing import Any, Dict, List, Tuple

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

__all__ = ["install_raw_main_grad", "install_side_grads", "install_wgrad_accumulation"]


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
        # Keep any gradient the weight got outside its layer, e.g. kv_b_proj in MLA ring attention.
        for fsdp_param, main_grad in pairs:
            if main_grad.untyped_storage().size() > 0:
                param = fsdp_param.unsharded_param
                if param.grad is not None:
                    main_grad.add_(param.grad)
                # A transposed main_grad is made contiguous once per step for the reduce-scatter.
                # The parameter's grad_dtype is fp32 (the reduce dtype) or None, so it takes fp32.
                param.grad = main_grad.contiguous()
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


def install_side_grads(model: nn.Module) -> None:
    """
    Keep each FSDP2 parameter's .grad empty while the step accumulates, after the main_grad
    installs (its post_backward has to run before theirs).
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            for group in fully_shard.state(module)._fsdp_param_groups:
                _attach_side_grads(group)


def _attach_side_grads(group: FSDPParamGroup) -> None:
    """
    Move a .grad whose dtype differs from its parameter's into a per-parameter sum during the step,
    and hand the sums back as .grad when the group reduces.
    """
    orig_pre_backward, orig_post_backward = group.pre_backward, group.post_backward
    sums: Dict[int, torch.Tensor] = {}
    hooked: Dict[int, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(param: torch.Tensor) -> None:
            grad = param.grad
            if grad is None or grad.dtype == param.dtype:
                return
            if index in sums:
                sums[index].add_(grad)
            else:
                sums[index] = grad
            param.grad = None

        return hook

    def pre_backward(*args: Any) -> None:
        # The unsharded parameters exist once the group has been all-gathered; FSDP2 keeps the
        # same objects across steps, so each is hooked once (again if FSDP2 ever replaces it).
        for index, fsdp_param in enumerate(group.fsdp_params):
            param = fsdp_param.unsharded_param
            if hooked.get(index) is not param:
                param.register_post_accumulate_grad_hook(make_hook(index))
                hooked[index] = param
        orig_pre_backward(*args)

    def post_backward(*args: Any) -> None:
        if group.reduce_grads:
            for index, fsdp_param in enumerate(group.fsdp_params):
                grad = sums.pop(index, None)
                if grad is not None:
                    param = fsdp_param.unsharded_param
                    param.grad = grad if param.grad is None else grad.add_(param.grad)
        orig_post_backward(*args)

    group.pre_backward = pre_backward
    group.post_backward = post_backward
