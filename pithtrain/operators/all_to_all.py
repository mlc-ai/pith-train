"""
All-to-all collectives for the expert-parallel dispatch and combine, without autograd: callers
implement the backward themselves.
"""

import functools
from typing import List, Tuple

import torch
import torch.autograd
import torch.distributed

__all__ = ["all_to_all_async", "all_to_all_sync", "direct_all_to_all"]

all_to_all_single_eager = torch.compiler.disable(torch.distributed.all_to_all_single)

# The lean path calls the process group's binding with prebuilt options, skipping the Python
# layers of torch.distributed.all_to_all_single, which cost more host time than the enqueue itself.
ASYNC_OPTIONS = torch.distributed.AllToAllOptions()
ASYNC_OPTIONS.asyncOp = True
SYNC_OPTIONS = torch.distributed.AllToAllOptions()
SYNC_OPTIONS.asyncOp = False


def without_grad(fn):
    """
    Run fn with grad mode off, like torch.no_grad() but without its per-call context object.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.compiler.is_compiling():
            with torch.no_grad():
                return fn(*args, **kwargs)
        was_grad_enabled = torch.is_grad_enabled()
        torch._C._set_grad_enabled(False)
        try:
            return fn(*args, **kwargs)
        finally:
            torch._C._set_grad_enabled(was_grad_enabled)

    return wrapper


# Output rows are allocated in multiples of this, so varying sizes reuse one allocator block.
ALLTOALL_PAD_ALIGNMENT = 1024


@without_grad
def all_to_all_async(
    x: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: torch.distributed.ProcessGroup,
) -> Tuple[torch.Tensor, torch.distributed.Work]:
    """
    The caller waits on work before reading output. The split sizes and group are
    those of torch.distributed.all_to_all_single.
    """
    valid_size = sum(output_split_sizes)
    # NCCL reads x by these splits; a mismatch would read past the tensor.
    assert sum(input_split_sizes) == x.shape[0], "input splits do not cover the input rows"
    padded_size = (
        (valid_size + ALLTOALL_PAD_ALIGNMENT - 1) // ALLTOALL_PAD_ALIGNMENT * ALLTOALL_PAD_ALIGNMENT
    )
    buf = x.new_empty(padded_size, *x.shape[1:])
    output = buf[:valid_size]
    if group is not None and not torch.compiler.is_compiling():
        work = group.all_to_all_single(
            output, x, output_split_sizes, input_split_sizes, ASYNC_OPTIONS
        )
    else:
        work = all_to_all_single_eager(
            output, x, output_split_sizes, input_split_sizes, group, async_op=True
        )
    return output, work


@without_grad
def all_to_all_sync(
    output: torch.Tensor,
    x: torch.Tensor,
    output_split_sizes: List[int] | None = None,
    input_split_sizes: List[int] | None = None,
    group: torch.distributed.ProcessGroup | None = None,
) -> None:
    if group is None or torch.compiler.is_compiling():
        all_to_all_single_eager(output, x, output_split_sizes, input_split_sizes, group)
        return
    if input_split_sizes:
        assert sum(input_split_sizes) == x.shape[0], "input splits do not cover the input rows"
        assert sum(output_split_sizes) == output.shape[0], "output splits do not cover the output"
    work = group.all_to_all_single(
        output, x, output_split_sizes or [], input_split_sizes or [], SYNC_OPTIONS
    )
    if work is not None:
        work.wait()


@without_grad
def direct_all_to_all(
    x: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    output, comm_work = all_to_all_async(x, output_split_sizes, input_split_sizes, group)
    output.comm_work = comm_work
    return output
