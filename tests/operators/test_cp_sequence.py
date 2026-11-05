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
from pithtrain.kernels import gated_delta_rule_cp_supported
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
    HK: int = 0  # key heads (q, k); 0: H, one per value head
    K: int = 128
    V: int = 128


# Relative L2 bound of the sharded recurrence against the dense one, per CP path. The CuTe CP
# kernels (gated_delta_rule_cp_supported) measured an error of exactly 0 in every output, rank and
# case below at seeds 42, 1, 2 and 3 on the H200 (the gate decays a rank's state to about 1e-28
# within a shard, and a rank's initial state comes from the same kernels the dense path runs); 1e-4
# leaves room for bf16 one-ulp flips from a change of rounding order. The SM100 TIRx CP stages
# measured 3e-3 to 4e-3 against the fp32 recurrence, as does the dense path (2.5e-3 to 3.5e-3), so
# at most about 7e-3 between them, and below 1e-2 when compared directly (B200, one-process
# emulation, 2 and 4 ranks, b 1 to 3, S 1024 to 2048): 1e-2. A mis-merged summary errs by O(1)
# (the next-rank control below reads about 1.4).
CUTE_CP_BOUND, TIRX_CP_BOUND = 1e-4, 1e-2


def fold_batch(t: torch.Tensor) -> torch.Tensor:
    """
    [b, s, heads, ...] -> [1, s, b * heads, ...] contiguous, the operator's fold.
    """
    return t.transpose(0, 1).reshape(1, t.shape[1], -1, *t.shape[3:]).contiguous()


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

    This is the gate the layout tests cannot provide: the CP kernels merge a per-shard transition
    summary across ranks, so a wrong merge yields a plausible loss curve rather than a crash.
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
        q=unit(req.B, req.S, req.HK or req.H, req.K),
        k=unit(req.B, req.S, req.HK or req.H, req.K),
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
    shards = {name: shard(t) for name, t in full.items()}
    local = run(shards, cp_group)
    folded = [fold_batch(shards[n]) for n in ("q", "k", "v", "g", "beta")]
    bound = CUTE_CP_BOUND if gated_delta_rule_cp_supported(*folded) else TIRX_CP_BOUND

    for name, got in local.items():
        error = relative_error(shard(dense[name]), got)
        # The next rank's shard is the cheapest wrong answer, so it calibrates the tolerance.
        control = relative_error(contiguous_shard(dense[name], (cp_rank + 1) % cp_size, cp_size), got)  # fmt: skip
        if error >= bound:
            raise AssertionError(f"{name} diverged under cp={cp_size}: {error=:.2e} {control=:.2e}")
        if control <= 10 * error:
            raise AssertionError(f"{name} control too close under cp={cp_size}: {error=:.2e} {control=:.2e}")  # fmt: skip


# Qwen3.5's geometry (16 key heads, 32 value heads), the one the kernels cover: the CP kernels
# against the pinned dense kernels, with the bounds above. SM100 repeats q / k to the value heads in the operator; SM90's CuTe kernels take the key
# heads. B > 1 folds into the heads on SM90 and runs one sample at a time on SM100.
RECURRENCE = [
    pytest.param(2, RecurrenceRequest(B=1, S=512, H=32, HK=16), id="CP2-B1"),
    pytest.param(2, RecurrenceRequest(B=3, S=512, H=32, HK=16), id="CP2-B3"),
    pytest.param(4, RecurrenceRequest(B=1, S=1024, H=32, HK=16), id="CP4-B1"),
    pytest.param(4, RecurrenceRequest(B=2, S=4096, H=32, HK=16), id="CP4-B2-S4K"),
    pytest.param(2, RecurrenceRequest(B=1, S=4096, H=32, HK=16), id="CP2-B1-S4K"),
    pytest.param(4, RecurrenceRequest(B=1, S=2048, H=32, HK=16), id="CP4-B1-S2K"),
]


@pytest.mark.parametrize("cp_size,req", RECURRENCE)
def test_gated_delta_rule_vs_dense(cp_size: int, req: RecurrenceRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_recurrence, req)
