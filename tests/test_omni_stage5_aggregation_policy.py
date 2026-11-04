"""Private semantic aggregation policy; default G gates remain unchanged."""

import copy
import gc
import hashlib
import inspect
import pickle
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pithtrain.models.interface import RoutingInfo, Stage5RouteMetadata, Stage5RoutingInfo
from pithtrain.modules.omni_stage5_policy import validate_omni_stage5_aggregation
from pithtrain.modules.resume_semantics import model_contract
from pithtrain.pipeline import execution
from tests.cpu_helpers import layer_type, load_definitions


def serial_layer(ep):
    cls = layer_type(ep)
    namespace = cls._forward_stage5.__globals__
    load_definitions(
        "pithtrain/models/qwen3_omni_moe.py",
        ("forward_stage5_with_metadata",),
        namespace,
        class_name="Qwen3OmniMoeThinkerTextDecoderLayer",
    )
    cls.forward_stage5_with_metadata = namespace["forward_stage5_with_metadata"]
    layer = cls()
    layer.idx, layer.is_moe = 0, True
    layer.omni_stage5_aggregation = "hf_eager_expert_id_serial"
    return layer


def actual_hf_case(seed, k, dtype, permuted, shared_residual, compiled=False):
    import transformers
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf

    assert transformers.__version__ == "5.17.0"
    assert hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest() == (
        "1c1fc322dfa547c697576e8ad3c88b817aa382abc9db5d966755be3d9fabf7ce"
    )
    torch.manual_seed(seed)
    n, experts, hidden = 11, 16, 32
    config = SimpleNamespace(
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=24,
        hidden_act="silu",
        _experts_implementation="eager",
    )
    model = hf.Qwen3OmniMoeThinkerTextExperts(config).to(dtype)
    with torch.no_grad():
        model.gate_up_proj.normal_(0, 0.2)
        model.down_proj.normal_(0, 0.2)
    x = torch.randn(n, hidden, dtype=dtype, requires_grad=True)
    weights, ids = torch.randn(n, experts).softmax(-1).topk(k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype).detach().requires_grad_()
    pointers = {model.down_proj[e].data_ptr(): e for e in range(experts)}
    observed = []
    linear = torch.nn.functional.linear

    def observe(value, weight, bias=None):
        output = linear(value, weight, bias)
        expert = pointers.get(weight.data_ptr())
        if expert is not None:
            slot, token = torch.where((ids == expert).T)
            observed.append((token * k + slot, output))
        return output

    with patch.object(torch.nn.functional, "linear", observe):
        expected = model(x, ids, weights)
    positions = torch.cat([p for p, _ in observed])
    assert torch.equal(positions.sort().values, torch.arange(n * k))
    outputs = x.new_zeros(n * k, hidden).index_copy(
        0, positions, torch.cat([v for _, v in observed])
    )
    returned = torch.randperm(n * k) if permuted else None
    if returned is not None:
        outputs = outputs[returned]
    residual = (
        x.view(1, n, hidden)
        if shared_residual
        else torch.randn(1, n, hidden, dtype=dtype, requires_grad=True)
    )
    expected = residual + expected.view_as(residual)
    layer = serial_layer(2 if permuted else 1)
    call = layer.forward_stage5_with_metadata if compiled else layer._forward_stage5
    actual = call(
        outputs,
        returned,
        weights,
        residual,
        stage5_metadata=Stage5RouteMetadata.from_expert_ids(ids),
    )
    assert torch.equal(actual, expected)
    targets = (x, weights, model.gate_up_proj, model.down_proj)
    if not shared_residual:
        targets += (residual,)
    cotangent = torch.randn_like(expected)
    a = torch.autograd.grad(actual, targets, cotangent, retain_graph=True)
    b = torch.autograd.grad(expected, targets, cotangent)
    assert all(torch.equal(left, right) for left, right in zip(a, b, strict=True))


@pytest.mark.parametrize("seed", [82, 83])
@pytest.mark.parametrize(
    "k,dtype", [(1, torch.bfloat16), (2, torch.bfloat16), (8, torch.bfloat16), (8, torch.float32)]
)
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("shared_residual", [False, True])
def test_explicit_serial_actual_hf_boundary(seed, k, dtype, permuted, shared_residual):
    actual_hf_case(seed, k, dtype, permuted, shared_residual)


@pytest.mark.parametrize("seed", [82, 83])
def test_explicit_serial_compiled_actual_hf_bf16_boundary(seed):
    actual_hf_case(seed, 8, torch.bfloat16, True, False, compiled=True)


@pytest.mark.parametrize("invalid", [True, False, None, "eager", "grouped_mm", "unknown", 0])
def test_policy_is_explicit_aggregation_enum(invalid):
    with pytest.raises(ValueError, match="omni_stage5_aggregation"):
        validate_omni_stage5_aggregation(invalid)


def test_legacy_tuple_abis_and_wrapper_replace_serialization():
    assert RoutingInfo._fields == (
        "topk_weight",
        "expert_idxs",
        "moe_local_idxs",
        "expand_idx",
        "dispatch_splits",
        "combine_splits",
    )
    assert execution.Stage5Args._fields == ("moe_outs", "topk_weight", "residual")
    base = RoutingInfo(torch.ones(2, 2), torch.arange(4))
    metadata = Stage5RouteMetadata.from_expert_ids(torch.tensor([[3, 0], [1, 2]]))
    wrapped = Stage5RoutingInfo(base, metadata)
    a, b, c, d, e, f = wrapped
    assert a is base.topk_weight and b is base.expert_idxs
    assert (c, d, e, f) == (None, None, None, None)
    assert len(wrapped) == 6 and wrapped[0] is base[0]
    assert tuple(wrapped._asdict()) == RoutingInfo._fields
    assert wrapped._asdict()["topk_weight"] is base.topk_weight
    indexes = torch.randperm(4)
    updated = wrapped._replace(moe_local_idxs=indexes)
    assert isinstance(updated, Stage5RoutingInfo)
    assert updated.stage5_metadata is metadata and updated.moe_local_idxs is indexes
    assert wrapped.moe_local_idxs is None
    for restored in (copy.deepcopy(wrapped), pickle.loads(pickle.dumps(wrapped))):
        assert isinstance(restored, Stage5RoutingInfo)
        assert torch.equal(restored.stage5_metadata.global_expert_ids, metadata.global_expert_ids)
        assert torch.equal(restored.topk_weight, base.topk_weight)
    from torch.utils import _pytree

    leaves, tree = _pytree.tree_flatten(wrapped)
    assert leaves == [wrapped]  # No runtime routing pytree consumer; wrapper stays an owner.
    assert _pytree.tree_unflatten(leaves, tree) is wrapped
    with pytest.raises((AttributeError, TypeError)):
        wrapped.stage5_metadata = metadata


def stage1_namespace(policy, ids, weights):
    namespace = dict(
        torch=torch,
        RoutingInfo=RoutingInfo,
        Stage5RouteMetadata=Stage5RouteMetadata,
        Stage5RoutingInfo=Stage5RoutingInfo,
        replay_indices=lambda *_: None,
        dist=SimpleNamespace(ProcessGroup=object),
        distributed=SimpleNamespace(ep_size=1, ep_group=None),
    )
    load_definitions("pithtrain/operators/ep_dispatch.py", ("prepare_dispatch",), namespace)
    load_definitions(
        "pithtrain/models/qwen3_omni_moe.py",
        ("forward_stage1",),
        namespace,
        class_name="Qwen3OmniMoeThinkerTextDecoderLayer",
    )
    layer = SimpleNamespace(
        is_moe=True,
        omni_stage5_aggregation=policy,
        mlp=SimpleNamespace(
            gate=None,
            num_experts_per_tok=ids.shape[1],
            num_experts=8,
            experts_per_rank=8,
            deterministic_token_layout=False,
        ),
        forward_stage1_compute=lambda hidden, *_: (hidden, hidden, ids, weights, None),
    )
    return namespace, layer


def test_default_stage1_no_extra_metadata_factory_clone_sort_or_wrapper():
    ids, weights = torch.tensor([[7, 0, 3], [2, 6, 1]]), torch.rand(2, 3)
    namespace, layer = stage1_namespace("grouped_sum", ids, weights)

    class Trap:
        @classmethod
        def from_expert_ids(cls, *_):
            raise AssertionError("Default must not clone/sort route metadata")

    namespace["Stage5RouteMetadata"] = Trap
    namespace["Stage5RoutingInfo"] = lambda *_: (_ for _ in ()).throw(
        AssertionError("Default wrapper allocation")
    )
    hidden = torch.randn(1, 2, 4)
    tokens, residual, routing = namespace["forward_stage1"](layer, hidden, None)
    assert type(routing) is RoutingInfo and len(routing) == 6
    assert routing.topk_weight is weights and torch.equal(routing.expert_idxs, ids.flatten())
    assert residual is hidden and tokens.shape == (6, 4)


@pytest.mark.parametrize("compiled", [False, True])
def test_actual_stage4_reverse_microbatch_gc_owns_indices_without_stage5args_metadata(
    compiled, monkeypatch
):
    monkeypatch.setattr(execution.nvtx, "range_push", lambda *_: None)
    monkeypatch.setattr(execution.nvtx, "range_pop", lambda: None)
    layer = serial_layer(1)
    if not compiled:
        layer.forward_stage5_with_metadata = layer._forward_stage5
    ctx = SimpleNamespace(
        fwd_comm_work=None,
        bwd_comm_work=None,
        fwd_comm_deferred_free=[],
        comp_stream=SimpleNamespace(record_event=lambda *_: None),
        comm_stream=SimpleNamespace(wait_event=lambda *_: None),
        fwd_event=None,
        bwd_event=None,
    )
    pending = []
    for seed in (82, 83):
        torch.manual_seed(seed)
        ids = torch.stack([torch.randperm(8) for _ in range(3)])
        weights = torch.rand(3, 8, dtype=torch.bfloat16)
        outputs = torch.randn(24, 8, dtype=torch.bfloat16)
        residual = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        namespace, producer = stage1_namespace("hf_eager_expert_id_serial", ids, weights)
        _, _, routing = namespace["forward_stage1"](producer, residual, None)
        assert isinstance(routing, Stage5RoutingInfo)
        routing_ref = weakref.ref(routing)
        order_ref = weakref.ref(routing.stage5_metadata.expert_order)
        # Actual stage4 consumes the real wrapper's combine_splits, not a fake
        # holder. EP1 has no collective; EP2 communication is not claimed.
        stage4, combined = execution.stage4_f(ctx, layer, outputs, routing.combine_splits)
        assert stage4.splits is None
        record, actual = execution.stage5_f(ctx, layer, combined, routing, residual)
        assert len(record.args) == 3
        o, w, r = record.args  # Legacy adapters keep exactly three float lanes.
        assert all(value.requires_grad for value in (o, w, r))
        reference_outputs = outputs.clone().requires_grad_()
        reference_weights = weights.clone().requires_grad_()
        reference_residual = residual.clone().requires_grad_()
        from pithtrain.models.qwen3_omni_eager_compat import eager_stage5_from_routes

        expected = eager_stage5_from_routes(
            reference_outputs, reference_weights, ids, reference_residual, num_experts=8
        )
        cotangent = torch.randn_like(actual)
        expected_grads = torch.autograd.grad(
            expected, (reference_outputs, reference_weights, reference_residual), cotangent
        )
        pending.append(
            (record, actual.detach(), expected.detach(), expected_grads, cotangent, order_ref)
        )
        del routing, producer, namespace, ids, weights, outputs, residual, combined, stage4, o, w, r
        gc.collect()
        assert routing_ref() is None
        assert order_ref() is not None  # Actual gather/autograd retains what backward needs.
    for record, actual, expected, expected_grads, cotangent, order_ref in reversed(pending):
        gc.collect()
        assert torch.equal(actual, expected)
        actual_grads = execution.stage5_b(ctx, layer, record, (cotangent,))
        assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads, strict=True))


def test_numerical_identity_absent_grouped_serial_and_policy_mismatch_rejection():
    model = torch.nn.Linear(2, 2)
    absent = model_contract(model)
    model.omni_stage5_aggregation = "grouped_sum"
    grouped = model_contract(model)
    model.omni_stage5_aggregation = "hf_eager_expert_id_serial"
    serial = model_contract(model)
    assert len({c["sha256"] for c in (absent, grouped, serial)}) == 3
    from pithtrain.modules.task_checkpoint import TaskCheckpointState

    task = SimpleNamespace(
        state_dict=lambda: dict(step=1, updates=1, data=1),
        validate_state_dict=lambda _: None,
        load_state_dict=lambda _: None,
    )
    envelopes = [
        TaskCheckpointState(
            task,
            task_kind="stage5_policy",
            task_schema=1,
            contract=dict(model=contract),
            transaction_key="step",
            update_key="updates",
            data_key="data",
        )
        for contract in (absent, grouped, serial)
    ]
    for left in envelopes:
        saved = left.state_dict()
        for right in envelopes:
            if left is right:
                right.validate_state_dict(saved)
            else:
                with pytest.raises(ValueError, match="fingerprint"):
                    right.validate_state_dict(saved)
    layer = serial_layer(1)
    with pytest.raises(ValueError, match="requires per-microbatch"):
        layer._forward_stage5(torch.ones(2, 4), None, torch.ones(1, 2), torch.ones(1, 1, 4))
    layer.omni_stage5_aggregation = "grouped_sum"
    with pytest.raises(ValueError, match="disagree"):
        layer._forward_stage5(
            torch.ones(2, 4),
            None,
            torch.ones(1, 2),
            torch.ones(1, 1, 4),
            stage5_metadata=Stage5RouteMetadata.from_expert_ids(torch.tensor([[1, 0]])),
        )


@pytest.mark.parametrize(
    "policy,expected", [("grouped_sum", 1.03125), ("hf_eager_expert_id_serial", 1.0)]
)
def test_sparse_constructor_captures_policy_and_reference_honors_it(policy, expected):
    ids = torch.arange(8).view(1, 8)
    weights = torch.ones(1, 8, dtype=torch.bfloat16)
    values = torch.tensor([1.0] + [0.00390625] * 7, dtype=torch.bfloat16).view(8, 1)
    runtime = SimpleNamespace(omni_stage5_aggregation=policy, deterministic_token_layout=False)

    class Experts(torch.nn.Module):
        def __init__(self, *_):
            super().__init__()

        def forward(self, *_):
            return values

    class Gate(torch.nn.Module):
        def __init__(self, *_):
            super().__init__()

        def forward(self, *_):
            return ids, weights, None

    namespace = dict(
        torch=torch,
        nn=torch.nn,
        training=runtime,
        distributed=SimpleNamespace(ep_size=1),
        validate_omni_stage5_aggregation=validate_omni_stage5_aggregation,
        Qwen3OmniMoeThinkerTextExperts=Experts,
        Qwen3OmniMoeThinkerTextTopKRouter=Gate,
        replay_indices=lambda *_: None,
        scatter_for_grouped_gemm=lambda repeated, *_args, **_kwargs: (
            repeated,
            torch.arange(8),
            None,
            None,
            None,
        ),
        padded_index_gather=lambda tensor, reverse: tensor[reverse],
    )
    load_definitions(
        "pithtrain/models/qwen3_omni_moe.py",
        ("_combine_expert_order", "_combine_expert_sum", "Qwen3OmniMoeThinkerTextSparseMoeBlock"),
        namespace,
    )
    block = namespace["Qwen3OmniMoeThinkerTextSparseMoeBlock"](
        SimpleNamespace(num_experts=8, num_experts_per_tok=8)
    )
    runtime.omni_stage5_aggregation = (
        "grouped_sum" if policy != "grouped_sum" else "hf_eager_expert_id_serial"
    )
    assert block.omni_stage5_aggregation == policy
    if policy == "grouped_sum":
        namespace["_combine_expert_order"] = lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("Default serial path")
        )
    assert block.reference_forward(torch.zeros(1, 1, 1, dtype=torch.bfloat16)).item() == expected


@pytest.mark.parametrize(
    "policy,model_type",
    [
        ("grouped_sum", "qwen3_omni_moe_talker_text"),
        ("hf_eager_expert_id_serial", "qwen3_omni_moe_text"),
        ("hf_eager_expert_id_serial", "qwen3_omni_moe_talker_text"),
    ],
)
def test_public_setup_policy_scope_before_model_allocation(policy, model_type):
    config = SimpleNamespace(model_type=model_type, hidden_size=8)
    runtime = SimpleNamespace()

    class ReachedConstructor(Exception):
        pass

    def stop(*_args, **_kwargs):
        assert runtime.omni_stage5_aggregation == policy
        raise ReachedConstructor

    namespace = dict(
        TrainingCfg=SimpleNamespace,
        DistributedCfg=SimpleNamespace,
        training=runtime,
        distributed=SimpleNamespace(cp_size=1, cp_group=None),
        validate_omni_stage5_aggregation=validate_omni_stage5_aggregation,
        FP8Linear=object,
        BF16Linear=object,
        FP8GroupedLinear=object,
        GroupedLinear=object,
        AutoConfig=SimpleNamespace(from_pretrained=lambda _: config),
        resolve_text_model_config=lambda value: value,
        model_class_for_config=lambda *_a, **_k: stop,
    )
    load_definitions("pithtrain/modules/training.py", ("setup_model",), namespace)
    cfg = SimpleNamespace(
        omni_stage5_aggregation=policy,
        fp8=False,
        model=Path("unused"),
        sequence_length=16,
        native_vision=False,
        native_audio=False,
        native_video=False,
        native_multimodal=False,
    )
    if policy != "grouped_sum" and model_type != "qwen3_omni_moe_text":
        with pytest.raises(ValueError, match="requires an Omni Thinker"):
            namespace["setup_model"](cfg, SimpleNamespace())
    else:
        with pytest.raises(ReachedConstructor):
            namespace["setup_model"](cfg, SimpleNamespace())
