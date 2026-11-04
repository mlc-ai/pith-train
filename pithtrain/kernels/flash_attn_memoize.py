"""
Host-side memoization of FlashAttention-4's Python entry (memoized_launch): a repeated call
signature replays the recorded compiled-kernel launches instead of running FA4's Python. Pinned to
the validated FA4 version; process-global and not thread-safe. FA4 runs on SM90 and SM100.
"""

import dataclasses
import importlib.metadata
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["FAST_PATH", "MISSING", "memoized_launch", "plans", "signature"]

# FA4's entry spends tens of microseconds per call deriving tile sizes and the compile key, a pure
# function of the call's signature, while its kernels take a few microseconds to launch. A recorded
# call becomes a plan only if FA4 did nothing but allocate, take views and call cached kernels;
# nothing captured is replayed except compiled kernels and immutable arguments.

VALIDATED_FA4 = "4.0.0b28"


def fast_path_enabled() -> bool:
    try:
        return importlib.metadata.version("flash-attn-4") == VALIDATED_FA4
    except importlib.metadata.PackageNotFoundError:
        return False


FAST_PATH = fast_path_enabled()

# aten ops FA4's entry may run without disqualifying a plan.
ALLOC_OPS = {"empty.memory_format", "empty_like.default", "empty_strided.default", "zeros.default", "zeros_like.default", "new_empty.default", "new_zeros.default"}
ZERO_OPS = {"zero_.default"}
VIEW_OPS = {"detach.default", "alias.default", "view.default", "_unsafe_view.default", "t.default", "transpose.int", "permute.default", "as_strided.default", "slice.Tensor", "select.int", "squeeze.dim", "unsqueeze.default", "expand.default"}


class Recording(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.allocs: List[torch.Tensor] = []
        self.zeroed_ptrs: set = set()
        self.other_ops: List[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func).removeprefix("aten.")
        if name in ALLOC_OPS:
            self.allocs.append(out)
            if name.startswith(("zeros", "new_zeros")):
                self.zeroed_ptrs.add(out.data_ptr())
        elif name in ZERO_OPS:
            self.zeroed_ptrs.add(args[0].data_ptr())
        elif name not in VIEW_OPS:
            self.other_ops.append(name)
        return out


class Recorder:
    def __init__(self, fn: Callable, launch_log: list) -> None:
        self.fn, self.launch_log = fn, launch_log

    def __call__(self, *args, **kwargs):
        self.launch_log.append((self.fn, args, kwargs))
        return self.fn(*args, **kwargs)


class RecordingCache:
    def __init__(self, inner: Any, launch_log: list) -> None:
        self.inner, self.launch_log, self.inserted = inner, launch_log, False

    def __contains__(self, key) -> bool:
        return key in self.inner

    def __getitem__(self, key):
        return Recorder(self.inner[key], self.launch_log)

    def __setitem__(self, key, fn) -> None:
        self.inserted = True
        self.inner[key] = fn

    def __getattr__(self, name):
        return getattr(self.inner, name)


@dataclasses.dataclass(frozen=True)
class Plan:
    allocs: Tuple[Tuple[torch.Size, Tuple[int, ...], torch.dtype, bool], ...]
    calls: Tuple[Tuple[Callable, Any, Any], ...]
    outputs: Tuple[Any, ...]
    # calls flattened for replay: (kernel, arguments with the constants filled in, (position,
    # template) of every argument rebuilt per call)
    fast_calls: Tuple[Tuple[Callable, Tuple[Any, ...], Tuple[Tuple[int, Any], ...]], ...] = dataclasses.field(compare=False, default=())


class Unsupported(Exception):
    pass


def template(x: Any, inputs: Dict[str, torch.Tensor], allocs: List[torch.Tensor]) -> Any:
    """
    How to rebuild x from the live inputs and fresh allocations of a later call.
    """
    if isinstance(x, torch.Tensor):
        shares_storage = lambda tensor: tensor.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        meta = (x.shape, x.stride(), x.storage_offset())
        for input_name, tensor in inputs.items():
            if tensor is not None and shares_storage(tensor) and meta == (tensor.shape, tensor.stride(), tensor.storage_offset()):
                return ("in", input_name, None)
        for alloc_idx, tensor in enumerate(allocs):
            if shares_storage(tensor):
                view = None if meta == (tensor.shape, tensor.stride(), tensor.storage_offset()) else (tuple(x.shape), tuple(x.stride()), x.storage_offset())
                return ("new", alloc_idx, view)
        # Inputs only ever pass through whole: a view of one (or a copy) is not planned.
        raise Unsupported("argument tensor that is no input and no allocation of FA4")
    if isinstance(x, (list, tuple)):
        parts = tuple(template(element, inputs, allocs) for element in x)
        if hasattr(x, "_fields"):
            return ("namedtuple", type(x), parts)
        return ("seq", type(x), parts)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        fields = {field.name: template(getattr(x, field.name), inputs, allocs) for field in dataclasses.fields(x)}
        if all(field_recipe[0] == "const" for field_recipe in fields.values()):
            return ("const", x)
        return ("dataclass", type(x), fields)
    if x is None or isinstance(x, (bool, int, float, str)):
        return ("const", x)
    if hasattr(x, "__dict__") and any(isinstance(value, torch.Tensor) for value in vars(x).values()):
        raise Unsupported(f"argument object {type(x).__name__} holds tensors")
    return ("const", x)


def build(recipe: Any, inputs: Dict[str, torch.Tensor], allocs: List[torch.Tensor]) -> Any:
    kind = recipe[0]
    if kind == "const":
        return recipe[1]
    if kind in ("in", "new"):
        base = inputs[recipe[1]] if kind == "in" else allocs[recipe[1]]
        if recipe[2] is None:
            return base.detach()
        shape, stride, offset = recipe[2]
        return base.detach().as_strided(shape, stride, offset)
    if kind == "namedtuple":
        return recipe[1](*(build(item_recipe, inputs, allocs) for item_recipe in recipe[2]))
    if kind == "seq":
        return recipe[1](build(item_recipe, inputs, allocs) for item_recipe in recipe[2])
    return recipe[1](**{name: build(field_recipe, inputs, allocs) for name, field_recipe in recipe[2].items()})


def record(entry: Callable[[], Tuple[torch.Tensor, ...]], inputs: Dict[str, torch.Tensor]):
    """
    Run FA4's entry once, returning (its outputs, a plan), where the plan is None if the call
    cannot be planned and MISSING if it compiled a kernel.
    """
    # Imported here, not at the top: FA4 loads cutlass, and importing pithtrain.kernels must not.
    from flash_attn.cute import interface as fa4_interface
    owners = [owner for owner in vars(fa4_interface).values() if callable(owner) and hasattr(owner, "compile_cache")]
    launch_log: list = []
    caches = [RecordingCache(owner.compile_cache, launch_log) for owner in owners]
    for owner, cache in zip(owners, caches):
        owner.compile_cache = cache
    try:
        with Recording() as recording:
            entry_out = entry()
    finally:
        for owner, cache in zip(owners, caches):
            owner.compile_cache = cache.inner
    if any(cache.inserted for cache in caches):
        return entry_out, MISSING  # compiled a kernel on this call: record the next one
    if recording.other_ops or not launch_log:
        return entry_out, None  # did more than allocate, take views and launch cached kernels
    try:
        allocs = recording.allocs
        alloc_meta = tuple((tensor.shape, tensor.stride(), tensor.dtype, tensor.data_ptr() in recording.zeroed_ptrs) for tensor in allocs)
        calls = tuple((fn, template(args, inputs, allocs), template(kwargs, inputs, allocs)) for fn, args, kwargs in launch_log)
        output_recipes = tuple(template(out, inputs, allocs) for out in entry_out)
    except Unsupported:
        return entry_out, None
    if any(out_recipe[0] != "new" for out_recipe in output_recipes) or any(kwargs for _, _, kwargs in launch_log):
        return entry_out, None
    fast_calls = []
    for fn, args, _ in calls:
        consts, slots = tuple(arg_recipe[1] if arg_recipe[0] == "const" else None for arg_recipe in args[2]), tuple((i, arg_recipe) for i, arg_recipe in enumerate(args[2]) if arg_recipe[0] != "const")
        fast_calls.append((fn, consts, slots))
    return entry_out, Plan(alloc_meta, calls, output_recipes, tuple(fast_calls))


def replay(plan: Plan, inputs: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, ...]:
    allocs = [torch.zeros(shape, dtype=dtype, device=device) if zeroed else torch.empty_strided(shape, stride, dtype=dtype, device=device) for shape, stride, dtype, zeroed in plan.allocs]
    detached = {name: tensor.detach() for name, tensor in inputs.items() if tensor is not None}
    for fn, consts, slots in plan.fast_calls:
        args = list(consts)
        for i, arg_recipe in slots:
            kind, source, view = arg_recipe
            if kind == "in" and view is None:
                args[i] = detached[source]
            elif kind == "new" and view is None:
                args[i] = allocs[source]
            else:
                args[i] = build(arg_recipe, inputs, allocs)
        fn(*args)
    return tuple(allocs[out_recipe[1]] if out_recipe[2] is None else build(out_recipe, inputs, allocs) for out_recipe in plan.outputs)


MISSING, SEEN = object(), object()
plans: Dict[Tuple, Optional[Plan]] = {}


def signature(kind: str, inputs: Dict[str, Optional[torch.Tensor]], *scalars) -> Tuple:
    # Plans pass inputs through whole, so their layout and alignment (FA4 copies misaligned ones)
    # are all that matters; how inputs alias each other does not.
    return (kind, scalars, inputs["q"].device, tuple(None if tensor is None else (tensor.shape, tensor.stride(), tensor.dtype, tensor.data_ptr() % 16) for tensor in inputs.values()))


def memoized_launch(kind: str, inputs: Dict[str, Optional[torch.Tensor]], scalars: Tuple, entry: Callable[[], Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    entry() (a call into FA4's Python entry), replayed through a recorded host-side plan once one
    exists for this call's signature. Returns entry()'s output tuple.

    kind: the FA4 entry point's name. scalars: a hashable tuple of every non-tensor argument.
    inputs: name -> tensor (or None) for every tensor argument, "q" included. entry must be a
    function of exactly these and return freshly allocated tensors.

    The signature is kind, scalars, the device and each input's shape, strides, dtype and address
    modulo 16. Its first call runs entry(), the second runs it while recording, and once a clean
    recording exists (FA4 only allocated, took views and called compiled kernels) later calls
    allocate the same buffers afresh and call the same kernels on the live inputs. The outputs are
    entry()'s (bit for bit where FA4 is deterministic). With any FA4 version other than the
    validated one, every call is entry().
    """
    if not FAST_PATH:
        return entry()
    key = signature(kind, inputs, *scalars)
    plan = plans.get(key, MISSING)
    if plan is MISSING:
        # The plain entry compiles whatever this signature needs.
        plans[key] = SEEN
        return entry()
    if plan is SEEN:
        entry_out, plan = record(entry, inputs)
        if plan is not MISSING:
            plans[key] = plan
        return entry_out
    if plan is None:
        return entry()
    return replay(plan, inputs, inputs["q"].device)
