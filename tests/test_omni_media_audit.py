"""CPU regressions for the installed object-broadcast audit wrapper."""

from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch

from tests.omni_media_audit import MediaAudit


def install_audit(stack, audit, original):
    class Data:
        _share_metadata = lambda *_: None
        validate_model = lambda *_: None
        _next_media_microbatch = lambda *_: None

    prepare = SimpleNamespace(verify_bundle=lambda: None, processor_for=lambda: None)
    reader = SimpleNamespace(create_omni_dataloader=lambda: None)
    dist = SimpleNamespace(
        broadcast_object_list=original,
        broadcast=lambda *_: None,
        get_global_rank=lambda group, rank: 11 + rank,
    )
    audit.install(stack, Data, prepare, reader, dist)
    return dist


def broadcast(dist, object_list, keyword, **kwargs):
    if keyword:
        return dist.broadcast_object_list(object_list=object_list, **kwargs)
    return dist.broadcast_object_list(object_list, **kwargs)


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("pp_size,group_kind", [(1, "pipeline"), (2, "foreign"), (2, "default")])
def test_object_broadcast_passthrough_preserves_arguments_and_result(keyword, pp_size, group_kind):
    pp_group = object()
    group = {"pipeline": pp_group, "foreign": object(), "default": None}[group_kind]
    audit = MediaAudit(0, pp_size, pp_group)
    audit.in_batch = True
    audit.specs = specs = [object()]
    # DCP metadata is outside the pipeline audit and may contain tensors.
    payload = [{"checkpoint": torch.ones(1)}]
    result = object()
    calls = []

    def original(object_list, **kwargs):
        assert object_list is payload
        assert not audit.in_object_broadcast
        calls.append(kwargs)
        object_list.append("received")
        return result

    with ExitStack() as stack:
        dist = install_audit(stack, audit, original)
        assert broadcast(dist, payload, keyword, group=group, src=11, device="cpu") is result
    assert calls == [dict(group=group, src=11, device="cpu")]
    assert payload[-1] == "received"
    assert not audit.counts and audit.specs is specs and audit.in_batch
    assert not audit.in_object_broadcast


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("pp_rank", [0, 1])
def test_object_broadcast_audits_pipeline_metadata_for_both_call_styles(keyword, pp_rank):
    group = object()
    audit = MediaAudit(pp_rank, 2, group)
    audit.in_batch = True
    specs = [("input_ids", (1, 2), torch.int64)]
    header = (({"sequence": 2}, specs), None)
    payload = [header if pp_rank == 0 else None]
    result = object()

    def original(object_list, **kwargs):
        assert object_list is payload and audit.in_object_broadcast
        assert kwargs == dict(group=group, src=11)
        object_list[:] = [header]
        return result

    with ExitStack() as stack:
        dist = install_audit(stack, audit, original)
        assert broadcast(dist, payload, keyword, group=group, src=11) is result
    assert audit.counts == {"metadata_headers": 1, "shared_specs": 1}
    assert audit.specs == specs and audit.specs is not specs
    assert not audit.in_object_broadcast


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_object_broadcast_still_rejects_tensors_in_pipeline_metadata(keyword, when):
    group = object()
    audit = MediaAudit(0, 2, group)
    payload = [(torch.ones(1), None)] if when == "before" else [None]
    calls = []

    def original(object_list, **kwargs):
        calls.append(True)
        object_list[:] = [(torch.ones(1), None)]

    with ExitStack() as stack:
        dist = install_audit(stack, audit, original)
        with pytest.raises(AssertionError, match="A media tensor entered pickled metadata"):
            broadcast(dist, payload, keyword, group=group, src=11)
    assert len(calls) == int(when == "after")
    assert not audit.counts and not audit.in_object_broadcast


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
def test_object_broadcast_restores_audit_state_when_original_fails(keyword):
    group = object()
    audit = MediaAudit(0, 2, group)

    def original(object_list, **kwargs):
        assert audit.in_object_broadcast
        raise RuntimeError("collective failed")

    with ExitStack() as stack:
        dist = install_audit(stack, audit, original)
        with pytest.raises(RuntimeError, match="collective failed"):
            broadcast(dist, [None], keyword, group=group, src=11)
    assert not audit.counts and not audit.in_object_broadcast
