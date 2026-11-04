"""
FSDP2 forward methods with a lean path for the repeated calls of a step.

FSDP's register_fsdp_forward_method runs its full pre- and post-forward around every call. Under
reshard_after_forward=False the parameters stay unsharded from a group's first call of the step to
the end-of-step reduce, so from the second call on most of that work is host-side overhead.

register_fsdp_forward_method here takes FSDP's full path whenever a precondition fails (a group
not unsharded or with an all-gather in flight, a pending forward root, forward prefetching, an
output cast, no grad mode, tracing). Otherwise it performs only the per-call side effects, through
FSDP's own functions and in its order: the training-state transitions, the mixed-precision input
cast, RegisterPostBackwardFunction on the grad-requiring inputs, the post-forward order record and
the pre-backward hook on every grad-requiring output. A call to a module in PRE_BACKWARD does only
the input cast.

install_lean_hooks builds the dynamo-disable wrapper of FSDP's hooks once instead of on every call,
and drops the debug log and profiler ranges of a post-backward that reduces nothing.
"""

import functools
from typing import Any

import torch
import torch.nn as nn
from torch._functorch.utils import unwrap_dead_wrappers
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import (
    register_fsdp_forward_method as upstream_register_fsdp_forward_method,  # fmt: skip
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    TrainingState,
    _dynamo_disable,
    is_bw,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPParamGroup,
    RegisterPostBackwardFunction,
)
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

__all__ = ["install_lean_hooks", "register_fsdp_forward_method"]


def is_sequence(x: Any) -> bool:
    return type(x) in (tuple, list) or (isinstance(x, tuple) and hasattr(type(x), "_fields"))


def rebuild_sequence(x: Any, elems) -> Any:
    return type(x)(*elems) if hasattr(type(x), "_fields") else type(x)(elems)


SCALAR_TYPES = frozenset((bool, int, float, str, type(None)))
NOT_PLAIN, NEEDS_CAST = 1, 2


def scan_tree(x: Any, dtype: torch.dtype | None, grads: list) -> int:
    """
    Walk x once and return its flags: NOT_PLAIN unless x holds only tensors, tuples, named tuples,
    lists, None and scalars; NEEDS_CAST if casting its floating tensors to dtype (None: no cast)
    would change one. Appends its grad-requiring tensors to grads in depth-first order, the order
    of FSDP's collect_grad_tensors.
    """
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            grads.append(x)
        if dtype is not None and x.dtype != dtype and torch.is_floating_point(x):
            return NEEDS_CAST
        return 0
    if type(x) in SCALAR_TYPES or isinstance(x, (bool, int, float, str)):
        return 0
    if not is_sequence(x):
        return NOT_PLAIN
    flags = 0
    for elem in x:
        if type(elem) not in SCALAR_TYPES:
            flags |= scan_tree(elem, dtype, grads)
    return flags


def scan_call(args: tuple, kwargs: dict, dtype: torch.dtype | None, grads: list) -> int:
    flags = scan_tree(args, dtype, grads)
    for value in kwargs.values():
        flags |= scan_tree(value, dtype, grads)
    return flags


def cast_tree(x: Any, dtype: torch.dtype) -> Any:
    """
    Matches FSDP's _cast_fp_tensor on a plain structure.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if torch.is_floating_point(x) and x.dtype != dtype else x
    if is_sequence(x):
        return rebuild_sequence(x, [cast_tree(elem, dtype) for elem in x])
    return x


def replace_grad_tensors(x: Any, replacements) -> Any:
    if isinstance(x, torch.Tensor):
        return next(replacements) if x.requires_grad else x
    if is_sequence(x):
        return rebuild_sequence(x, [replace_grad_tensors(elem, replacements) for elem in x])
    return x


def backward_state_call_ok(state: Any) -> bool:
    """
    Whether a call to a module in PRE_BACKWARD needs only the input cast.
    """
    # The pipeline defers FSDP's root post-backward callback to the end of the step, so a module
    # stays in PRE_BACKWARD after its first backward and FSDP takes its checkpointing branch.
    if torch.compiler.is_compiling() or state._mp_policy.output_dtype is not None:
        return False
    return all(group.is_unsharded for group in state._fsdp_param_groups)


def forward_cast_dtype(state: Any) -> torch.dtype | None:
    policy = state._mp_policy
    return policy.param_dtype if policy.cast_forward_inputs else None


def cast_call_inputs(state: Any, args: tuple, kwargs: dict, flags: int) -> tuple[tuple, dict]:
    """
    FSDP's _cast_forward_inputs, given the scan_call flags of the arguments.
    """
    if flags & NOT_PLAIN:
        return state._cast_forward_inputs(args, kwargs)
    if flags & NEEDS_CAST:
        dtype = state._mp_policy.param_dtype
        return cast_tree(args, dtype), {name: cast_tree(value, dtype) for name, value in kwargs.items()}  # fmt: skip
    return args, kwargs


# RegisterPostBackwardFunction.apply without Function.apply's bind_default_args, an
# inspect.signature per call that is a no-op here: forward(param_group, *inputs) has no defaults.
register_post_backward_apply = super(torch.autograd.Function, RegisterPostBackwardFunction).apply


def repeat_call_ok(state: Any) -> bool:
    """
    Whether a call may take the lean path: FSDP's full pre- and post-forward would reduce to the
    side effects the lean path performs.
    """
    if not torch.is_grad_enabled() or torch.compiler.is_compiling() or is_bw():
        return False
    if torch._C._are_functorch_transforms_active():
        return False
    if state._training_state != TrainingState.IDLE or state._states_to_forward_prefetch:
        return False
    if state._mp_policy.output_dtype is not None:
        return False
    # With no forward root yet, FSDP makes this call the root, whose post-forward frees the last
    # all-gather result: take the full path while one is held. Inside another state's forward
    # FSDP does no root work for this call.
    root = state._state_ctx.iter_forward_root
    if root is state:
        return False
    for group in state._fsdp_param_groups:
        if root is None and group.comm_ctx.all_gather_state is not None:
            return False
        if group._training_state != TrainingState.IDLE or group._all_gather_result is not None:
            return False
        if not group.is_unsharded or group._reshard_after_forward:
            return False
    return True


def register_fsdp_forward_method(module: nn.Module, method_name: str) -> None:
    """
    Drop-in for torch.distributed.fsdp.register_fsdp_forward_method that takes the lean path for
    the repeated calls of a step.
    """
    if not isinstance(module, FSDPModule):
        upstream_register_fsdp_forward_method(module, method_name)
        return
    orig_method = getattr(module, method_name)

    @functools.wraps(orig_method)
    def wrapped_method(self, *args, **kwargs):
        state = self._get_fsdp_state()
        if state._training_state == TrainingState.PRE_BACKWARD and backward_state_call_ok(state):
            dtype = forward_cast_dtype(state)
            if dtype is not None:
                flags = scan_call(args, kwargs, dtype, [])
                args, kwargs = cast_call_inputs(state, args, kwargs, flags)
            return orig_method(*args, **kwargs)
        if not repeat_call_ok(state):
            args, kwargs = state._pre_forward(self, args, kwargs)
            out = orig_method(*args, **kwargs)
            return state._post_forward(self, args, out)
        groups = state._fsdp_param_groups
        tensors = []
        flags = scan_call(args, kwargs, forward_cast_dtype(state), tensors)
        plain = not flags & NOT_PLAIN
        state._training_state = TrainingState.FORWARD
        args, kwargs = cast_call_inputs(state, args, kwargs, flags)
        recollect = bool(flags & NEEDS_CAST)  # the cast replaced the tensors it changed
        for group in groups:
            group._training_state = TrainingState.FORWARD
            if not plain:
                args, kwargs = group._register_post_backward_hook(args, kwargs)
                continue
            # As FSDP's _register_post_backward_hook: one RegisterPostBackwardFunction over all
            # grad-requiring inputs (args, then kwargs), its outputs swapped in for them.
            if recollect:
                tensors = []
                scan_call(args, kwargs, None, tensors)
            recollect = True  # this group replaces them for the next one
            if tensors:
                applied = register_post_backward_apply(*unwrap_dead_wrappers((group, *tensors)))
                replacements = iter(applied)
                args = replace_grad_tensors(args, replacements)
                kwargs = {name: replace_grad_tensors(value, replacements) for name, value in kwargs.items()}  # fmt: skip
        out = orig_method(*args, **kwargs)
        for group in groups:
            group._record_post_forward()
            group._training_state = TrainingState.IDLE
        outs = []
        if not scan_tree(out, None, outs) & NOT_PLAIN:
            # As FSDP's _register_pre_backward_hook: the hook on every grad-requiring output.
            for tensor in outs:
                tensor.register_hook(state._pre_backward)
        else:
            out = state._register_pre_backward_hook(out)
        state._training_state = TrainingState.IDLE
        return out

    setattr(module, method_name, wrapped_method.__get__(module, type(module)))


# The FSDP hooks, decorated with _dynamo_disable, that run during a training step.
LEAN_HOOKS = (
    (FSDPState, "_pre_forward"),
    (FSDPState, "_post_forward"),
    (FSDPState, "_pre_backward"),
    (FSDPState, "_root_post_backward_final_callback"),
    (FSDPParamGroup, "pre_backward"),
    (FSDPParamGroup, "post_backward"),
)
PER_CALL_DISABLE_CODE = _dynamo_disable(lambda: None).__code__


def lean_post_backward(post_backward):
    """
    Wrap FSDPParamGroup.post_backward so that a backward that reduces nothing skips FSDP's debug log
    and profiler ranges; one that reduces runs FSDP's own.
    """

    @functools.wraps(post_backward)
    def lean(param_group, *unused):
        if param_group.reduce_grads:
            return post_backward(param_group, *unused)
        param_group._training_state = TrainingState.POST_BACKWARD
        for fsdp_param in param_group.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()
        if param_group.reshard_after_backward:
            param_group.reshard()
        for fsdp_param in param_group.fsdp_params:
            fsdp_param.to_accumulated_grad_if_needed()

    return lean


def install_lean_hooks() -> None:
    """
    Replace the per-call torch._dynamo.disable wrapper of FSDP's step hooks with one built once,
    with lean_post_backward around post_backward. Hooks not decorated that way are left alone.
    """
    for cls, name in LEAN_HOOKS:
        hook = cls.__dict__[name]
        if getattr(hook, "__code__", None) is not PER_CALL_DISABLE_CODE:
            continue
        func = hook.__wrapped__
        if (cls, name) == (FSDPParamGroup, "post_backward"):
            func = lean_post_backward(func)
        setattr(
            cls, name, torch._dynamo.disable(func, recursive=True, reason="skipping FSDP hooks")
        )
