"""
Test CP sequence collectives and the packed zigzag partition.
"""

from dataclasses import dataclass
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from pithtrain.contexts import distributed
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.cp_sequence import (
    contiguous_to_zigzag,
    prepend_conv_state,
    zigzag_spans,
    zigzag_to_contiguous,
    zigzag_varlen_index,
)
from pithtrain.operators.gated_delta_rule import gated_delta_rule
from tests.utilities import launch


@dataclass
class Request:
    B: int
    S: int
    D: int
    width: int = 3


def zigzag_shard(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    blocks = x.chunk(2 * cp_size, dim=1)
    return torch.cat([blocks[cp_rank], blocks[2 * cp_size - cp_rank - 1]], dim=1).contiguous()


def contiguous_shard(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    return x.chunk(cp_size, dim=1)[cp_rank].contiguous()


def verify_relayout(req: Request) -> None:
    """
    Both directions must agree with reference sharding of the same global sequence, and the
    backward of each must be the other, which the gradient of a layout-aware weight pins down.
    """
    cp_group = distributed.cp_group
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()

    torch.manual_seed(42)
    tokens = torch.randn(req.B, req.S, req.D, device=device, dtype=torch.bfloat16)
    weight = torch.randn(req.B, req.S, req.D, device=device, dtype=torch.bfloat16)

    zigzag_in = zigzag_shard(tokens, cp_rank, cp_size).requires_grad_(True)
    contiguous_out = zigzag_to_contiguous(zigzag_in, cp_group)
    torch.testing.assert_close(contiguous_out, contiguous_shard(tokens, cp_rank, cp_size))
    # DualPipeV frees stage outputs by hand, so one has to own its storage.
    assert contiguous_out._base is None, "zigzag_to_contiguous returned a view"

    (contiguous_out * contiguous_shard(weight, cp_rank, cp_size)).sum().backward()
    torch.testing.assert_close(zigzag_in.grad, zigzag_shard(weight, cp_rank, cp_size))

    contiguous_in = contiguous_shard(tokens, cp_rank, cp_size).requires_grad_(True)
    zigzag_out = contiguous_to_zigzag(contiguous_in, cp_group)
    torch.testing.assert_close(zigzag_out, zigzag_shard(tokens, cp_rank, cp_size))
    assert zigzag_out._base is None, "contiguous_to_zigzag returned a view"

    (zigzag_out * zigzag_shard(weight, cp_rank, cp_size)).sum().backward()
    torch.testing.assert_close(contiguous_in.grad, contiguous_shard(weight, cp_rank, cp_size))

    torch.testing.assert_close(contiguous_to_zigzag(contiguous_out, cp_group), zigzag_in.detach())


def verify_prepend(req: Request) -> None:
    """
    The pad must be the trailing tokens of the previous rank taken from the global sequence, and
    the gradient of those pad rows must land back on the rank that owns them.
    """
    cp_group = distributed.cp_group
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()
    shard_len, width = req.S // cp_size, req.width

    torch.manual_seed(42)
    tokens = torch.randn(req.B, req.S, req.D, device=device, dtype=torch.bfloat16)
    # Every rank draws the same table so each can name the weight its successor will use.
    weights = torch.randn(cp_size, req.B, shard_len + width, req.D, device=device, dtype=torch.bfloat16)  # fmt: skip

    shard = contiguous_shard(tokens, cp_rank, cp_size).requires_grad_(True)
    prepended = prepend_conv_state(shard, width, cp_group)

    padded = torch.cat([torch.zeros_like(tokens[:, :width]), tokens], dim=1)
    expected_out = padded[:, cp_rank * shard_len : (cp_rank + 1) * shard_len + width]
    torch.testing.assert_close(prepended, expected_out)
    assert prepended._base is None, "prepend_conv_state returned a view"

    (prepended * weights[cp_rank]).sum().backward()
    expected_grad = weights[cp_rank, :, width:].clone()
    if cp_rank + 1 < cp_size:
        expected_grad[:, -width:] += weights[cp_rank + 1, :, :width]
    torch.testing.assert_close(shard.grad, expected_grad)


REQUESTS = [
    pytest.param(2, Request(B=1, S=256, D=64), id="CP2-B1"),
    # An odd cp_size is the only case where a rank sends both blocks to one peer.
    pytest.param(3, Request(B=2, S=192, D=64), id="CP3-B2"),
    pytest.param(4, Request(B=2, S=256, D=64), id="CP4-B2"),
    pytest.param(8, Request(B=1, S=256, D=64), id="CP8-B1"),
]


@pytest.mark.parametrize("cp_size,req", REQUESTS)
def test_relayout_round_trip(cp_size: int, req: Request) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_relayout, req)


@pytest.mark.parametrize("cp_size,req", REQUESTS)
def test_prepend_conv_state(cp_size: int, req: Request) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_prepend, req)


@pytest.mark.parametrize("cp_size", [2, 3, 4, 8])
def test_zigzag_varlen_index(cp_size: int) -> None:
    """
    Check per-document zigzag ordering and unique token ownership, including empty documents.
    """
    unit = 2 * cp_size
    seqlens = [unit, 7 * unit, 0, 3 * unit, unit, 0, 0]
    bounds = [0, *accumulate(seqlens)]
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32)
    held = []
    for cp_rank in range(cp_size):
        index = zigzag_varlen_index(cp_rank, cp_size, cu_seqlens, bounds[-1]).tolist()
        spans = [
            start + i
            for start, n in zip(bounds, seqlens)
            for span in zigzag_spans(cp_rank, cp_size, n)
            for i in span
        ]
        docs = [torch.arange(start, start + n).view(1, n) for start, n in zip(bounds, seqlens) if n]
        shards = torch.cat([zigzag_shard(doc, cp_rank, cp_size) for doc in docs], dim=1)
        assert index == spans == shards.view(-1).tolist()
        held += index
    assert sorted(held) == list(range(bounds[-1]))
    # Reject unpadded, missing-zero, decreasing and shard-local boundaries or lengths.
    refused = [
        ([0, unit + 2, 2 * unit], 2 * unit),
        ([unit, 2 * unit, 3 * unit], 3 * unit),
        ([0, 2 * unit, unit, 3 * unit], 3 * unit),
        ([0, unit], cp_size * unit),
        (bounds, bounds[-1] // cp_size),
    ]
    for cu, n in refused:
        with pytest.raises(RuntimeError, match="whole packed sample's boundaries"):
            zigzag_varlen_index(0, cp_size, torch.tensor(cu, dtype=torch.int32), n)
    # CP1 accepts odd lengths but still requires whole-sample boundaries.
    cu_seqlens = torch.tensor([0, 1, 7, 7, 10, 15], dtype=torch.int32)
    assert zigzag_varlen_index(0, 1, cu_seqlens, 15).tolist() == list(range(15))
    with pytest.raises(RuntimeError, match="whole packed sample's boundaries"):
        zigzag_varlen_index(0, 1, cu_seqlens, 16)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_packed_positions(cp_size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check positions directly: relative RoPE makes document resets invisible to attention tests.
    Tensors stay on CPU; importing the model requires a GPU.
    """
    from pithtrain.models.qwen3_moe import Qwen3MoeModel

    unit = 2 * cp_size
    seqlens = [unit, 7 * unit, 0, 3 * unit]
    cu_seqlens = torch.tensor([0, *accumulate(seqlens)], dtype=torch.int32)
    T = sum(seqlens)
    model = SimpleNamespace(rotary_emb=lambda S: (torch.arange(S), torch.arange(S)))
    monkeypatch.setattr(distributed, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(distributed, "cp_size", cp_size, raising=False)
    for cp_rank in range(cp_size):
        monkeypatch.setattr(distributed, "cp_rank", cp_rank, raising=False)
        positions, _ = Qwen3MoeModel.forward_posemb(model, T // cp_size, cu_seqlens)
        expected = [i for n in seqlens for span in zigzag_spans(cp_rank, cp_size, n) for i in span]
        assert positions.view(-1).tolist() == expected


@dataclass
class RecurrenceRequest:
    B: int
    S: int
    H: int
    K: int = 128
    V: int = 128
    atol: float = 1e-4


def relative_error(ref: torch.Tensor, got: torch.Tensor) -> float:
    """
    Relative L2 error, magnitude-sensitive: a mis-merged transition summary rescales the state
    rather than rotating it, so a scale-invariant metric would wave that through.
    """
    ref, got = ref.double(), got.double()
    return float((got - ref).norm() / ref.norm().clamp_min(1e-12))


def verify_recurrence(req: RecurrenceRequest) -> None:
    """
    The sharded recurrence must reproduce the dense one on the concatenated sequence.

    This is the gate the layout tests cannot provide: FLA merges a per-shard transition summary
    across ranks, so a wrong merge yields a plausible loss curve rather than a crash.
    """
    cp_group = distributed.cp_group
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()

    torch.manual_seed(42)
    shape = (req.B, req.S, req.H)

    def unit(*dims: int) -> torch.Tensor:
        # The model normalizes q and k per head before the recurrence.
        return F.normalize(torch.randn(*dims, device=device), dim=-1).to(torch.bfloat16)

    def shard(t: torch.Tensor) -> torch.Tensor:
        return contiguous_shard(t, cp_rank, cp_size)

    full = dict(
        q=unit(*shape, req.K),
        k=unit(*shape, req.K),
        v=torch.randn(*shape, req.V, device=device, dtype=torch.bfloat16),
        # Match the model: g is a negative float32 decay, beta a sigmoid in (0, 1).
        g=-torch.rand(*shape, device=device, dtype=torch.float32).mul(0.5),
        beta=torch.rand(*shape, device=device, dtype=torch.bfloat16),
    )
    grad_out = torch.randn(req.B, req.S, req.H, req.V, device=device, dtype=torch.bfloat16)

    def run(inputs: dict, cp: ProcessGroup | None) -> dict:
        leaves = {name: t.clone().requires_grad_(True) for name, t in inputs.items()}
        out = gated_delta_rule(**leaves, cp_group=cp)
        out.backward(shard(grad_out) if cp is not None else grad_out)
        return {"out": out, **{f"d{n}": t.grad for n, t in leaves.items()}}

    dense = run(full, None)
    local = run({name: shard(t) for name, t in full.items()}, cp_group)

    for name, got in local.items():
        error = relative_error(shard(dense[name]), got)
        # The next rank's shard is the cheapest wrong answer, so it calibrates the tolerance.
        control = relative_error(contiguous_shard(dense[name], (cp_rank + 1) % cp_size, cp_size), got)  # fmt: skip
        if error >= req.atol:
            raise AssertionError(f"{name} diverged under cp={cp_size}: {error=:.2e} {control=:.2e}")
        if control <= 10 * error:
            raise AssertionError(f"{name} control too close under cp={cp_size}: {error=:.2e} {control=:.2e}")  # fmt: skip


RECURRENCE = [
    pytest.param(2, RecurrenceRequest(B=1, S=512, H=4), id="CP2-B1"),
    pytest.param(2, RecurrenceRequest(B=3, S=512, H=4), id="CP2-B3"),
    pytest.param(4, RecurrenceRequest(B=1, S=1024, H=8), id="CP4-B1"),
    pytest.param(4, RecurrenceRequest(B=2, S=4096, H=4), id="CP4-B2-S4K"),
]


@pytest.mark.parametrize("cp_size,req", RECURRENCE)
def test_gated_delta_rule_vs_dense(cp_size: int, req: RecurrenceRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_recurrence, req)
