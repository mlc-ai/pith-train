"""Independent HF BF16 compute with FP32 parameter-gradient accumulation."""

import torch
from torch import nn
from torch.func import functional_call


class FP32GradientReference:
    """Match the staged training precision contract without native model code.

    Each forward materializes BF16 parameters from FP32 leaves. Backward through
    that cast promotes each parameter contribution before cross-microbatch
    accumulation, as the training FP32 gradient buffers require. The caller
    backpropagates summed CE, then normalizes once by the global target count.

    The HF module, its attention backend, buffers and forward remain unchanged.
    This does not emulate FSDP communication, native FP32 wgrad GEMMs, or their
    reduction order. Those numerical differences remain subject to every
    original parameter-gradient check.
    """

    def __init__(self, reference):
        self.reference = reference
        parameters = dict(reference.named_parameters())
        if not parameters or any(p.dtype != torch.bfloat16 for p in parameters.values()):
            raise ValueError("Expected a nonempty BF16 HF reference")
        self.master_parameters = {
            name: nn.Parameter(p.detach().float().clone(), requires_grad=p.requires_grad)
            for name, p in parameters.items()
        }
        self.normalized = False

    def __call__(self, *args, **kwargs):
        if self.normalized:
            raise RuntimeError("Clear gradients before starting another reference step")
        compute = {name: p.to(torch.bfloat16) for name, p in self.master_parameters.items()}
        return functional_call(
            self.reference,
            (compute, dict(self.reference.named_buffers())),
            args,
            kwargs,
            strict=True,
        )

    @torch.no_grad()
    def normalize_gradients_(self, target_count):
        if self.normalized:
            raise RuntimeError("Reference gradients were already normalized")
        if isinstance(target_count, torch.Tensor):
            if target_count.ndim != 0 or target_count.dtype not in (torch.int32, torch.int64):
                raise ValueError("Global target count must be a positive integer scalar")
            target_count = target_count.item()
        if type(target_count) is not int or target_count <= 0:
            raise ValueError("Global target count must be a positive integer scalar")
        for parameter in self.master_parameters.values():
            if parameter.grad is not None:
                if parameter.grad.dtype != torch.float32:
                    raise RuntimeError("Reference master gradients must accumulate in FP32")
                parameter.grad.div_(target_count)
        self.normalized = True

    def zero_grad(self):
        for parameter in self.master_parameters.values():
            parameter.grad = None
        self.normalized = False

    @staticmethod
    def precision_contract():
        return {
            "parameter_storage": "torch.float32",
            "parameter_compute": "torch.bfloat16",
            "parameter_gradient_accumulation": "torch.float32",
            "objective": "cross_entropy reduction=sum",
            "normalization": "once after accumulation by global nonignored target count",
            "forward": "unchanged independent HF",
            "distributed_collectives_emulated": False,
        }
