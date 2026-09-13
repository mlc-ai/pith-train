"""
The two sequence layouts under context parallelism, and the collectives that move between them.

Softmax attention uses zigzag, which pairs an early block with a late one so causal work
balances across ranks. A linear-attention recurrence cannot: its state flows from rank to rank,
so it needs contiguous, where rank order is sequence order. A hybrid model runs both kinds of
layer and converts the hidden stream at each boundary. Contiguous is also what lets a causal
convolution fetch the tokens before its shard from the previous rank.
"""

from functools import lru_cache
from typing import Literal

import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_to_all,
    batch_isend_irecv,
    get_global_rank,
    get_rank,
    get_world_size,
    irecv,
    isend,
)
from torch.distributed.distributed_c10d import _resolve_process_group

__all__ = [
    "zigzag_spans",
    "zigzag_to_contiguous",
    "contiguous_to_zigzag",
    "prepend_conv_state",
]

SequenceLayout = Literal["zigzag", "contiguous"]


def _block_ids(cp_rank: int, cp_size: int, layout: SequenceLayout) -> tuple[int, int]:
    """
    Which two of the 2 * cp_size blocks this rank holds, front first.
    """
    match layout:
        case "zigzag":
            return cp_rank, 2 * cp_size - cp_rank - 1
        case "contiguous":
            return 2 * cp_rank, 2 * cp_rank + 1
        case _:
            raise ValueError(f"unknown CP sequence layout: {layout!r}")


def zigzag_spans(cp_rank: int, cp_size: int, seq_len: int) -> tuple[range, range]:
    """
    The two ranges of the global sequence this rank holds under zigzag, front block then back.

    Call this rather than restating the arithmetic: the data loader, every forward_posemb and
    the relayout below must agree on one partition, and a disagreement mistrains silently.
    """
    block = seq_len // (2 * cp_size)
    front, back = _block_ids(cp_rank, cp_size, "zigzag")
    return range(front * block, (front + 1) * block), range(back * block, (back + 1) * block)


@lru_cache(maxsize=None)
def _relayout_plan(
    cp_rank: int, cp_size: int, source: SequenceLayout, target: SequenceLayout
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """
    Which of this rank's two blocks each peer takes, and where each peer's reply lands.

    Entries are (first slot, count), a slice of the rank's two blocks. Both sides walk peers in
    rank order and break ties by block index, so a sender and its receiver agree on the wire
    order without exchanging it.
    """
    assert source != target, f"relayout from {source!r} to itself"
    src = _block_ids(cp_rank, cp_size, source)
    dst = _block_ids(cp_rank, cp_size, target)

    send: list[tuple[int, int]] = []
    recv: list[tuple[int, int]] = []
    for peer in range(cp_size):
        wanted = _block_ids(peer, cp_size, target)
        slots = sorted(slot for slot, block in enumerate(src) if block in wanted)
        send.append((slots[0] if slots else 0, len(slots)))
        offered = _block_ids(peer, cp_size, source)
        slots = sorted(dst.index(block) for block in dst if block in offered)
        recv.append((slots[0] if slots else 0, len(slots)))

    assert sorted(i for slot, count in send for i in range(slot, slot + count)) == [0, 1], send
    assert sorted(i for slot, count in recv for i in range(slot, slot + count)) == [0, 1], recv
    return tuple(send), tuple(recv)


def _relayout(
    x: torch.Tensor, group: ProcessGroup, source: SequenceLayout, target: SequenceLayout
) -> torch.Tensor:
    """
    Move [B, S, D] from the source layout to the target one with a single all-to-all.

    Both layouts cut the sequence into 2 * cp_size blocks and give each rank two of them,
    differing only in which two. So the conversion is an exchange: send each peer the blocks it
    will own, receive the ones this rank will own, and write them straight into the result. That
    result is allocated here because DualPipeV frees stage outputs by hand and needs them to own
    their storage.
    """
    cp_rank, cp_size = get_rank(group), get_world_size(group)
    send_plan, recv_plan = _relayout_plan(cp_rank, cp_size, source, target)

    B, S, D = x.shape
    block = S // 2
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    for src_row, dst_row in zip(x.reshape(B, 2, block, D), out.view(B, 2, block, D)):
        all_to_all(
            [dst_row[slot : slot + count] for slot, count in recv_plan],
            [src_row[slot : slot + count] for slot, count in send_plan],
            group=group,
        )
    return out


@torch.library.custom_op("pithtrain::zigzag_to_contiguous", mutates_args=())
def _z2c(x: torch.Tensor, group_name: str) -> torch.Tensor:
    return _relayout(x, _resolve_process_group(group_name), "zigzag", "contiguous")


@torch.library.custom_op("pithtrain::contiguous_to_zigzag", mutates_args=())
def _c2z(x: torch.Tensor, group_name: str) -> torch.Tensor:
    return _relayout(x, _resolve_process_group(group_name), "contiguous", "zigzag")


@_z2c.register_fake
def _(x: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@_c2z.register_fake
def _(x: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _setup_context(ctx, inputs: tuple, output: torch.Tensor) -> None:
    _, ctx.group_name = inputs


def _z2c_backward(ctx, grad: torch.Tensor) -> tuple:
    return _c2z(grad, ctx.group_name), None


def _c2z_backward(ctx, grad: torch.Tensor) -> tuple:
    return _z2c(grad, ctx.group_name), None


_z2c.register_autograd(_z2c_backward, setup_context=_setup_context)
_c2z.register_autograd(_c2z_backward, setup_context=_setup_context)


def zigzag_to_contiguous(x: torch.Tensor, cp_group: ProcessGroup) -> torch.Tensor:
    """
    Reshard the hidden states for a linear-attention recurrence, which needs each rank to hold
    one contiguous run of the sequence. They arrive zigzag only because the model also has
    softmax attention layers, whose ring attention needs that pairing to balance causal work.
    """
    return _z2c(x, cp_group.group_name)


def contiguous_to_zigzag(x: torch.Tensor, cp_group: ProcessGroup) -> torch.Tensor:
    """
    Reshard the hidden states back to zigzag on the way out of a linear-attention run, for the
    softmax attention layers that follow.
    """
    return _c2z(x, cp_group.group_name)


def _shift(payload: torch.Tensor, into: torch.Tensor, group: ProcessGroup, forward: bool) -> None:
    """
    Send payload one hop (+1 if forward else -1) and receive the peer payload into the buffer.
    """
    cp_rank, cp_size = get_rank(group), get_world_size(group)
    send_to = cp_rank + 1 if forward else cp_rank - 1
    recv_from = cp_rank - 1 if forward else cp_rank + 1
    ops = []
    if 0 <= send_to < cp_size:
        ops.append(P2POp(isend, payload, get_global_rank(group, send_to), group))
    if 0 <= recv_from < cp_size:
        ops.append(P2POp(irecv, into, get_global_rank(group, recv_from), group))
    for work in batch_isend_irecv(ops):
        work.wait()


@torch.library.custom_op("pithtrain::prepend_conv_state", mutates_args=())
def _prepend(x: torch.Tensor, width: int, group_name: str) -> torch.Tensor:
    B, _, D = x.shape
    pad = torch.zeros(B, width, D, dtype=x.dtype, device=x.device)
    _shift(x[:, -width:].contiguous(), pad, _resolve_process_group(group_name), forward=True)
    return torch.cat([pad, x], dim=1)


@_prepend.register_fake
def _(x: torch.Tensor, width: int, group_name: str) -> torch.Tensor:
    B, S, D = x.shape
    return torch.empty((B, S + width, D), dtype=x.dtype, device=x.device)


@torch.library.custom_op("pithtrain::prepend_conv_state_bwd", mutates_args=())
def _prepend_bwd(grad: torch.Tensor, width: int, group_name: str) -> torch.Tensor:
    B, _, D = grad.shape
    tail = torch.zeros(B, width, D, dtype=grad.dtype, device=grad.device)
    _shift(grad[:, :width].contiguous(), tail, _resolve_process_group(group_name), forward=False)
    dx = grad[:, width:].clone(memory_format=torch.contiguous_format)
    dx[:, -width:] += tail
    return dx


@_prepend_bwd.register_fake
def _(grad: torch.Tensor, width: int, group_name: str) -> torch.Tensor:
    B, S, D = grad.shape
    return torch.empty((B, S - width, D), dtype=grad.dtype, device=grad.device)


def _prepend_setup_context(ctx, inputs: tuple, output: torch.Tensor) -> None:
    _, ctx.width, ctx.group_name = inputs


def _prepend_backward(ctx, grad: torch.Tensor) -> tuple:
    return _prepend_bwd(grad, ctx.width, ctx.group_name), None, None


_prepend.register_autograd(_prepend_backward, setup_context=_prepend_setup_context)


def prepend_conv_state(x: torch.Tensor, width: int, cp_group: ProcessGroup) -> torch.Tensor:
    """
    Left-pad [B, S, D] with the width tokens before this shard, fetched from the previous rank.

    Those rows are the conv state a causal depthwise convolution needs for its first width
    outputs. Rank 0 pads with zeros, which is what that convolution assumes at a sequence start.
    Assumes the contiguous layout, which is what puts them there.
    """
    return _prepend(x, width, cp_group.group_name)
