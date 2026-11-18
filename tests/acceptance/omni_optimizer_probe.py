"""Reversible optimizer-update evidence over existing rank-local parameters."""

import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.distributed.tensor import DTensor


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _same_bits(value, saved):
    value = _local(value).detach().contiguous().reshape(-1).view(torch.uint8)
    saved = saved.detach().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(value, saved)


@contextmanager
def isolated_optimizer_probe(model, make_optimizers):
    """Restore values/gradients/RNG even if update evidence raises.

    The factory must create disposable optimizers over this model's parameters.
    Live training optimizers, schedulers and data are not used by the probe.
    Only local shards are cloned; no DTensor gathering or object replacement is
    needed. The caller clears the restored parity gradients at its explicit
    boundary before beginning formal reader training.
    """
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    membership = tuple((name, id(p)) for name, p in model.named_parameters(remove_duplicate=False))
    buffer_membership = tuple(
        (name, id(b)) for name, b in model.named_buffers(remove_duplicate=False)
    )
    values = [(value, _local(value).detach().clone()) for value in (*parameters, *buffers)]
    gradients = [
        (p, p.grad, None if p.grad is None else _local(p.grad).detach().clone()) for p in parameters
    ]
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield make_optimizers()
    finally:
        with torch.no_grad():
            for value, saved in values:
                _local(value).copy_(saved)
            for parameter, gradient, saved in gradients:
                parameter.grad = gradient
                if gradient is not None:
                    _local(gradient).copy_(saved)
        for module, mode in modes:
            module.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if (
            membership
            != tuple((n, id(p)) for n, p in model.named_parameters(remove_duplicate=False))
            or buffer_membership
            != tuple((n, id(b)) for n, b in model.named_buffers(remove_duplicate=False))
            or any(not _same_bits(value, saved) for value, saved in values)
            or any(
                parameter.grad is not gradient
                or (gradient is not None and not _same_bits(gradient, saved))
                for parameter, gradient, saved in gradients
            )
        ):
            raise RuntimeError(
                "Optimizer probe failed to restore model/gradient identity and values"
            )
