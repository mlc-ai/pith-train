"""Execute actual DualPipeV dispatch methods with CPU tensors and mock compute.

Model kernels, CUDA streams/NVTX and overlapped math are mocked. These tests cover
phase/microbatch context selection, including the two chunks on the same PP rank,
and the forward microbatch identity published for routing replay.
"""

import ast
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch

from pithtrain.modules.microbatch import Microbatch


def dispatch_methods(overlapped):
    path = Path(__file__).parents[1] / "pithtrain/pipeline/dualpipev.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DualPipeV"
    )
    names = {"setup_step_metadata", "_forward_compute_chunk", "_forward_backward_compute_chunk"}
    nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ModuleType("cpu_dispatch_fixture")
    ns = module.__dict__
    ns.update(
        torch=torch,
        training=SimpleNamespace(current_microbatch=None),
        distributed=SimpleNamespace(device="cpu"),
        nvtx=SimpleNamespace(range_push=lambda *_: None, range_pop=lambda: None),
        overlapped_forward_backward=overlapped,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    module.DualPipeV = type("DualPipeV", (), {name: ns[name] for name in names})
    return module


@pytest.mark.parametrize("pp_rank", [0, 1])
@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("with_media", [False, True])
def test_pipeline_passes_media_only_to_stage_zero(pp_rank, phase, overlap, with_media):
    seen = []

    class Stage:
        hidden_size = 3

        def __init__(self, index):
            self.stage_index = index

        def __call__(self, inputs, **kwargs):
            seen.append((self.stage_index, kwargs, pipeline_module.training.current_microbatch))
            return torch.ones(1, 3, 3, requires_grad=True)

    def overlapped(module, *args, **kwargs):
        seen.append((module.stage_index, kwargs, pipeline_module.training.current_microbatch))
        return (
            [torch.ones(1, 3, 3, requires_grad=True)],
            torch.tensor(1.0),
            torch.tensor(1.0),
            [torch.ones(1, 3, 3)],
        )

    modules = [Stage(pp_rank), Stage(3 - pp_rank)]
    batches = []
    for index in range(3):
        tokens = torch.full((1, 3), index, dtype=torch.long)
        batches.append(
            Microbatch(
                model_inputs=(tokens,),
                objective_inputs=(tokens,),
                cu_seqlens=None,
                model_context={"input_ids": tokens, "image_grid_thw": torch.tensor([[1, 2, 2]])}
                if with_media
                else None,
                media_inputs={"pixel_values": torch.full((4, 3), float(index))}
                if with_media and pp_rank == 0
                else None,
            )
        )
    tensor = lambda: torch.ones(1, 3, 3, requires_grad=True)
    engine = SimpleNamespace(
        module=modules,
        current_f_chunk_id=[1, 1],
        current_b_chunk_id=[0, 0],
        input_chunks=[[[tensor()] for _ in range(3)] for _ in range(2)],
        output_chunks=[[[tensor()] for _ in range(3)] for _ in range(2)],
        output_grad_chunks=[[[torch.ones(1, 3, 3)] for _ in range(3)] for _ in range(2)],
        input_grad_chunks=[[], []],
        loss_chunks=[torch.tensor(1.0) for _ in range(3)],
        objective_output_chunks=[],
        objective_inputs=[b.objective_inputs for b in batches],
        objective=lambda *_: (torch.tensor(1.0), torch.tensor(1.0)),
        chunk_records=[[object() for _ in range(3)] for _ in range(2)],
        is_first_pp_rank=pp_rank == 0,
        is_last_pp_rank=pp_rank == 1,
        comm_stream=None,
        forward_only=False,
    )
    pipeline_module = dispatch_methods(overlapped)
    for name in (
        "setup_step_metadata",
        "_forward_compute_chunk",
        "_forward_backward_compute_chunk",
    ):
        setattr(engine, name, MethodType(getattr(pipeline_module.DualPipeV, name), engine))
    assert engine.setup_step_metadata(batches) == 3
    assert engine.p2p_shapes == [[(1, 3, 3)]] * 3
    if overlap:
        engine._forward_backward_compute_chunk(phase, 1 - phase)
    else:
        engine._forward_compute_chunk(phase)
    assert len(seen) == 1
    stage, kwargs, replay_microbatch = seen[0]
    assert stage == modules[phase].stage_index
    # The overlapped backward uses batch 0; routing replay must see forward batch 1.
    assert replay_microbatch == 1
    context = kwargs.get("model_context")
    if not with_media:
        assert context is None
        if not overlap:
            assert "model_context" not in kwargs
    else:
        expected = batches[1].context_for_stage(stage)
        assert context.keys() == expected.keys()
        assert all(context[key] is value for key, value in expected.items())
        assert torch.equal(context["input_ids"], batches[1].model_inputs[0])
        assert ("pixel_values" in context) == (stage == 0)
        assert torch.equal(context["image_grid_thw"], torch.tensor([[1, 2, 2]]))
        if stage == 0:
            assert context["pixel_values"] is batches[1].media_inputs["pixel_values"]
    assert engine.current_f_chunk_id[phase] == 2
