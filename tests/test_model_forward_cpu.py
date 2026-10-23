"""CPU regression for context forwarding and inference-to-training record reuse.

The real sequential pipeline and backward execute with small dense PyTorch layers.
Only NVTX annotations are stubbed; this does not cover CUDA kernels or EP collectives.
"""

import copy
from dataclasses import fields

import pytest
import torch
from torch import nn

from pithtrain.pipeline import execution


def assert_empty(record):
    assert all(getattr(record, field.name) is None for field in fields(record))


@pytest.mark.parametrize("with_context", [False, True])
@pytest.mark.parametrize("num_layers", [0, 2])
def test_model_forward_clears_inference_records_and_reuses_for_training(
    monkeypatch, with_context, num_layers
):
    monkeypatch.setattr(execution.nvtx, "range_push", lambda *_: None)
    monkeypatch.setattr(execution.nvtx, "range_pop", lambda: None)
    torch.manual_seed(137)
    inputs = torch.randn(1, 4, 3, dtype=torch.float64)
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    context = {"offset": torch.full_like(inputs, 0.25)} if with_context else None
    records = execution.create_chunk_record(num_layers, has_prolog=True, has_epilog=True)

    def assert_layers_empty():
        for layer_record in records.layers:
            for field in fields(layer_record):
                assert_empty(getattr(layer_record, field.name))

    class Layer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
            self.up = nn.Linear(3, 3)
            self.down = nn.Linear(3, 3)

        def forward_stage1(self, hidden, posemb, boundaries):
            assert boundaries is cu_seqlens
            if not torch.is_grad_enabled():
                # The previous layer must release its records before the next one executes.
                assert_empty(records.prolog)
                assert_layers_empty()
            return self.up(hidden), hidden, None

        def forward_stage3(self, hidden, expert_idxs, expand_idx):
            return self.down(hidden).tanh()

        def forward_stage5(self, hidden, moe_local_idxs, topk_weight, residual):
            return hidden + residual

    class Model(nn.Module):
        stage_index = 0
        stage_count = 1

        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(3, 3)
            self.layers = nn.ModuleDict({str(i): Layer(i) for i in range(num_layers)})
            self.head = nn.Linear(3, 2)

        def forward_prolog(self, hidden, **kwargs):
            assert kwargs == {} if context is None else kwargs["model_context"] is context
            offset = 0 if context is None else kwargs["model_context"]["offset"]
            return self.embed(hidden + offset)

        def forward_posemb(self, length, boundaries, **kwargs):
            assert length == inputs.shape[1] and boundaries is cu_seqlens
            assert kwargs == {} if context is None else kwargs["model_context"] is context
            if not torch.is_grad_enabled():
                assert_empty(records.prolog)
            return None

        def forward_epilog(self, hidden):
            if not torch.is_grad_enabled():
                assert_layers_empty()
            return self.head(hidden)

    model = Model().double()
    reference = copy.deepcopy(model)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        expected = reference.embed(inputs + (0 if context is None else context["offset"]))
        for layer in reference.layers.values():
            expected = layer.down(layer.up(expected)).tanh() + expected
        expected = reference.head(expected)
        with torch.no_grad():
            actual = execution.model_forward(model, inputs, records, cu_seqlens, context)
        torch.testing.assert_close(actual, expected.detach())
        assert not actual.requires_grad
        assert_empty(records.prolog)
        assert_layers_empty()
        assert_empty(records.epilog)

        actual = execution.model_forward(model, inputs, records, cu_seqlens, context)
        torch.testing.assert_close(actual, expected)
        execution.model_backward(model, None, actual.square().sum(), records)
        expected.square().sum().backward()
        for (name, parameter), (_, ref_parameter) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, ref_parameter.grad)
        assert_empty(records.prolog)
        assert_layers_empty()
        assert_empty(records.epilog)
