"""
The training meshes the kernel cases are derived from.

Two sources, both checked in:

- the matrix below: every tokens-per-GPU budget in TOKENS_PER_GPU at every micro-batch size in
  MICRO_BATCH_SIZES, under every context-parallel degree in CONTEXT_PARALLEL and every
  expert-parallel degree in EXPERT_PARALLEL a model admits;
- the meshes the repo ships: benchmarks/end2end/<case>/<hardware>/<mesh>/script.py and
  examples/pretrain_lm/<model>/script.py, read statically (their assignments to
  cfg.distributed.* and cfg.training.*), so a new benchmark or example lands in the spec on the
  next regeneration.

How a mesh maps onto kernel shapes (pithtrain/modules/distributed.py): there is no tensor
parallelism; attention runs on B = micro_batch_size sequences of S_local = sequence_length / cp
tokens; the experts of one rank see T = B * S_local tokens per EP source rank; pipeline
parallelism only decides which layers (and the embedding / lm_head) a rank holds, so it changes no
per-kernel shape.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from .models import REPO_ROOT, Model

TOKENS_PER_GPU = (4096, 8192, 16384, 32768)
MICRO_BATCH_SIZES = (1, 2, 4)
CONTEXT_PARALLEL = (1, 2, 4, 8)
EXPERT_PARALLEL = (1, 2, 4, 8, 16, 32)
MAX_SEQUENCE_LENGTH = 131072
"""
Longest sequence the matrix derives (T = 32768 per GPU at mbs 1 and CP 4, or mbs 2 and CP 8).
"""


@dataclass(frozen=True, order=True)
class Mesh:
    """
    One training mesh of one model. source is "matrix", or the path of the script it came from.
    """

    model: str
    seq: int
    mbs: int
    cp: int
    ep: int
    pp: int = 1
    dp: int | None = None
    fp8: bool = False
    source: str = "matrix"

    @property
    def tokens_per_gpu(self) -> int:
        return self.mbs * self.seq // self.cp

    @property
    def s_local(self) -> int:
        return self.seq // self.cp

    @property
    def label(self) -> str:
        if self.source == "matrix":
            return f"T{self.tokens_per_gpu // 1024}k-mbs{self.mbs}-cp{self.cp}-ep{self.ep}"
        return self.source


def allowed_ep(model: Model) -> list[int]:
    """
    EP degrees with whole experts per rank (E % ep == 0); dense models have only EP 1.
    """
    if model.experts is None:
        return [1]
    return [ep for ep in EXPERT_PARALLEL if model.experts.num_experts % ep == 0]


def allowed_cp(model: Model) -> list[int]:
    """
    CP degrees the model trains with: gpt-oss has no ring attention (window and sinks).
    """
    return list(CONTEXT_PARALLEL) if model.attention.supports_cp else [1]


def matrix_meshes(model: Model) -> list[Mesh]:
    """
    The tokens-per-GPU x micro-batch x CP x EP matrix, keeping sequences of 1k .. 128k tokens whose
    zigzag blocks (seq / (2 * cp)) are whole multiples of 128 tokens.
    """
    meshes = []
    for T in TOKENS_PER_GPU:
        for mbs in MICRO_BATCH_SIZES:
            for cp in allowed_cp(model):
                seq = T * cp // mbs
                if not 1024 <= seq <= MAX_SEQUENCE_LENGTH or seq % (256 * cp):
                    continue
                for ep in allowed_ep(model):
                    meshes.append(Mesh(model.name, seq, mbs, cp, ep))
    return meshes


MESH_DIR = re.compile(r"pp(\d+)-dp(\d+)-cp(\d+)-ep(\d+)")


def script_settings(path: Path) -> dict:
    """
    The constant assignments of a training script: distributed.<field> / training.<field> (and
    cfg.<section>.<field>) = literal, plus training.model = Path("...").
    """
    settings = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Attribute):
            continue
        value = node.value
        if isinstance(value, ast.Call) and getattr(value.func, "id", None) == "Path" and value.args:
            value = value.args[0]
        try:
            settings[target.attr] = ast.literal_eval(value)
        except ValueError:
            continue
    return settings


def script_mesh(path: Path, models: dict[str, Model]) -> Mesh | None:
    settings = script_settings(path)
    model_path = settings.get("model")
    if not isinstance(model_path, str):
        return None
    model = Path(model_path).name
    if model not in models:
        raise ValueError(f"{path}: model {model_path!r} is not under examples/pretrain_lm/")
    return Mesh(
        model=model,
        seq=settings["sequence_length"],
        mbs=settings["micro_batch_size"],
        cp=settings.get("context_parallel_size", 1),
        ep=settings.get("expert_parallel_size", 1),
        pp=settings.get("pipeline_parallel_size", 1),
        dp=int(MESH_DIR.search(str(path)).group(2)) if MESH_DIR.search(str(path)) else None,
        fp8=bool(settings.get("fp8", False)),
        source=str(path.parent.relative_to(REPO_ROOT)),
    )


def shipped_meshes(models: dict[str, Model]) -> list[Mesh]:
    """
    Every end-to-end benchmark and pretraining example script, as a Mesh.
    """
    paths = sorted((REPO_ROOT / "benchmarks" / "end2end").glob("*/*/*/script.py"))
    paths += sorted((REPO_ROOT / "examples" / "pretrain_lm").glob("*/script.py"))
    meshes = [script_mesh(path, models) for path in paths]
    return [mesh for mesh in meshes if mesh is not None]
