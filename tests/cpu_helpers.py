"""Small source loaders for CPU tests that avoid unrelated CUDA kernel imports."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[1]


def load_definitions(relative_path, names, namespace, *, class_name=None):
    path = ROOT / relative_path
    nodes = ast.parse(path.read_text()).body
    if class_name is not None:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    for name in names:
        node = next(
            n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def layer_type(ep):
    name = "Qwen3OmniMoeThinkerTextDecoderLayer"
    methods = ("_forward_stage5", "forward_stage5")
    namespace = load_definitions(
        "pithtrain/models/qwen3_omni_moe.py",
        methods,
        dict(torch=torch, distributed=SimpleNamespace(ep_size=ep)),
        class_name=name,
    )
    return type(name, (torch.nn.Module,), {method: namespace[method] for method in methods})


@pytest.fixture(name="setup_training")
def setup_training_fixture():
    def load(runtime, setup_model, configure_policy):
        cfg = SimpleNamespace(
            training=SimpleNamespace(seed=1234, optimizer=lambda _: (), scheduler=lambda _: ()),
            distributed=SimpleNamespace(),
        )
        noop_seed = SimpleNamespace(seed=lambda _: None)
        namespace = load_definitions(
            "pithtrain/modules/training.py",
            ("setup_training",),
            dict(
                TrainingCfg=SimpleNamespace,
                training=runtime,
                random=noop_seed,
                np=SimpleNamespace(random=noop_seed),
                torch=SimpleNamespace(
                    manual_seed=lambda _: None, cuda=SimpleNamespace(manual_seed_all=lambda _: None)
                ),
                setup_model=setup_model,
                configure_deterministic_training=configure_policy,
            ),
        )
        return cfg, namespace["setup_training"]

    return load
