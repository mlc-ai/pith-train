"""
Skip Dynamo guard evaluation on steps that repeat an already guarded step.

Every compiled region pays Dynamo's frame entry on each call, which costs host time on a host-bound
step. Under torch.compiler.set_stance("default", skip_guard_eval_unsafe=True) Dynamo checks only
the differentiating guards of each cache: the tensor-match guards on non-parameter inputs plus
every guard that has failed once. The second compiled variant of a code object (dense vs MoE layer,
sliding vs full attention) fails a guard of the first, so variants still tell themselves apart.

Skipping hides any other recompile condition. GuardSkip therefore runs a step unguarded only when
an identical signature already completed a fully guarded step in this process. The signature
covers grad mode, the ambient Torch state Dynamo guards, the training flag of every module, the
router-replay and load-balance settings of every gate, the decoder layer variants, the parallel
sizes, training.fp8 and every micro-batch's input metadata, including cu_seqlens. If anything
compiles during a skipped step, every signature loses its trust.

Skipping is off when decoder layers differ in module structure (qwen3.5 linear vs full attention,
deepseek-v2 dense vs MoE): guards copied from one variant look up submodules the other lacks, and
Dynamo raises instead of trying the next entry. It is also off on torch releases other than the
one it was validated on.
"""

from typing import Any, Hashable, List, Tuple

import torch
from torch._dynamo.utils import counters

from pithtrain.contexts import distributed, training

__all__ = ["GuardSkip"]

VALIDATED_TORCH = "2.13."


def torch_validated() -> bool:
    return torch.__version__.startswith(VALIDATED_TORCH)


def compiled_graph_count() -> int:
    # Bumped once per graph Dynamo produces: every compile and every recompile.
    return counters["stats"]["unique_graphs"]


def mode_ids(modes) -> Tuple:
    return tuple((type(mode).__name__, id(mode)) for mode in modes)


def ambient_state() -> Tuple:
    """
    Process-global Torch state Dynamo guards on every compiled region (grad mode aside), with the
    identity of every active torch-function or dispatch mode and saved-tensor hook.
    """
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    try:
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(True)
    except (AttributeError, TypeError):
        hooks = None
    matmul = torch.backends.cuda.matmul
    return (
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.is_autocast_cache_enabled(),
        torch.get_default_dtype(),
        str(torch.get_default_device()),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul.allow_tf32,
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cudnn.allow_tf32,
        torch.get_num_threads(),
        mode_ids(_get_current_function_mode_stack()),
        mode_ids(_get_current_dispatch_mode_stack()),
        None if hooks is None else tuple(id(h) for h in hooks),
    )


def object_config(obj: Any) -> Tuple | None:
    if obj is None:
        return None
    scalars = tuple(
        sorted(
            (name, value)
            for name, value in getattr(obj, "__dict__", {}).items()
            if value is None or isinstance(value, (bool, int, float, str))
        )
    )
    return (id(obj), type(obj).__name__, scalars)


def scan_modules(model: torch.nn.Module) -> Tuple[Tuple, Tuple, Tuple, bool]:
    """
    Return the training flag of every module, the router-replay and load-balance configuration of
    every gate, the decoder layer variants, and whether all layers share one module structure. A
    layer's variant is its structure plus the attributes that select a different
    compiled cache entry.
    """
    modes, gates, layers = [], [], []

    def visit(module: torch.nn.Module) -> Tuple:
        modes.append(module.training)
        if hasattr(module, "router_replay"):
            gates.append(
                (
                    object_config(module.router_replay),
                    object_config(getattr(module, "load_balance_loss_fn", None)),
                )
            )
        children = tuple(
            (name, visit(child)) for name, child in module._modules.items() if child is not None
        )
        structure = (type(module).__name__, children)
        if hasattr(module, "idx") and hasattr(module, "mlp"):
            attn = getattr(module, "self_attn", None)
            layers.append(
                (
                    structure,
                    getattr(attn, "is_sliding", None),
                    getattr(module, "is_linear", None),
                    getattr(module, "to_contiguous", None),
                    getattr(module, "to_zigzag", None),
                )
            )
        return structure

    visit(model)
    uniform = len({layer[0] for layer in layers}) <= 1
    return tuple(modes), tuple(gates), tuple(layers), uniform


def tensor_signature(tensor: torch.Tensor | None) -> Tuple | None:
    if tensor is None:
        return None
    return (tensor.shape, tensor.stride(), tensor.dtype, tensor.device, tensor.requires_grad)


class GuardSkip:
    def __init__(self) -> None:
        self.trusted: set[Hashable] = set()
        self.enabled = torch_validated()

    def signature(self, model: torch.nn.Module, microbatches: List[Any]) -> Tuple | None:
        """
        None when the step may never run unguarded (decoder layers of different module structure).
        """
        modes, gates, layers, uniform = scan_modules(model)
        if not uniform:
            return None
        microbatch_meta = tuple(
            (
                tuple(tensor_signature(tensor) for tensor in microbatch.model_inputs),
                tensor_signature(microbatch.cu_seqlens),
            )
            for microbatch in microbatches
        )
        return (
            torch.is_grad_enabled(),
            ambient_state(),
            modes,
            training.fp8,
            distributed.ep_size,
            distributed.cp_size,
            distributed.pp_size,
            gates,
            layers,
            microbatch_meta,
        )

    def run(self, model: torch.nn.Module, microbatches: List[Any], fn, *args, **kwargs):
        if not self.enabled:
            return fn(*args, **kwargs)
        key = self.signature(model, microbatches)
        if key is None:
            return fn(*args, **kwargs)
        trusted = key in self.trusted
        before = compiled_graph_count()
        if trusted:
            with torch.compiler.set_stance("default", skip_guard_eval_unsafe=True):
                out = fn(*args, **kwargs)
            if compiled_graph_count() != before:
                # Something compiled although this signature had run guarded: trust nothing
                # until each signature has run guarded again.
                self.trusted.clear()
                return out
        else:
            out = fn(*args, **kwargs)
        self.trusted.add(key)
        return out
