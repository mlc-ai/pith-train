"""Strict Omni topology contract; importing it does not import CUDA code."""

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

# Fingerprints bind the AST adapter and independent inventory to reviewed source.
# The entire implementation is additionally bound to --expected-commit at runtime.
SOURCE_HASHES = {
    "tests/test_dualpipev.py": "b113f3999ef860d7baa0709667ba92ae84480e3f1689c64e1fef4482c48f25cd",
    "tests/configs/qwen3_omni_text/tiny.json": "d03f488586dcb3053b10f24338dbe95d2bf4b8f499230adb980cfec98e86c702",
}
TEST_SHA256 = SOURCE_HASHES["tests/test_dualpipev.py"]
GRAD_EPS = 1e-2
SMALL_PARAMETERS = frozenset(
    f"layers.{i}.{suffix}"
    for i in range(4)
    for suffix in ("mlp.gate.weight", "post_attention_layernorm.weight")
)
MODEL = "tests/configs/qwen3_omni_text/tiny.json"


@dataclass(frozen=True)
class Case:
    name: str
    world: int
    pp: int
    ep: int
    cp: int
    packed: bool = False
    ragged: bool = False

    def __post_init__(self):
        if self.world not in (1, 2, 4, 8) or self.pp not in (1, 2):
            raise ValueError("Unsupported world/PP")
        stage = self.world // self.pp
        if self.world % self.pp or self.ep not in (1, 2) or self.cp not in (1, 2):
            raise ValueError("Unsupported mesh")
        if stage % self.ep or stage % self.cp or (self.packed and self.cp != 1):
            raise ValueError("Invalid mesh or unsupported packed CP")

    @property
    def timeout(self):
        return 120 if self.world == 1 else 180

    def environment(self):
        return {
            "PACKED_SEQLEN": str(int(self.packed)),
            "RAGGED_MICROBATCH": str(int(self.ragged)),
            "SEQUENCE_LENGTH": "128",
        }

    def record(self):
        return dict(asdict(self), timeout_seconds=self.timeout, environment=self.environment())


CASES = (
    Case("dense-single", 1, 1, 1, 1),
    Case("dense-dp2", 2, 1, 1, 1),
    Case("dense-pp2", 2, 2, 1, 1),
    Case("dense-ep2", 2, 1, 2, 1),
    Case("dense-pp2-ep2", 4, 2, 2, 1),
    Case("dense-cp2", 2, 1, 1, 2),
    Case("dense-folded-pp2-ep2-cp2", 8, 2, 2, 2),
    Case("packed-ragged-single", 1, 1, 1, 1, True, True),
    Case("packed-ragged-pp2", 2, 2, 1, 1, True, True),
    Case("packed-ragged-ep2", 2, 1, 2, 1, True, True),
    Case("packed-ragged-pp2-ep2", 4, 2, 2, 1, True, True),
    Case("ragged-pp2-ep2", 4, 2, 2, 1, False, True),
    Case("ragged-cp2", 2, 1, 1, 2, False, True),
)
BY_NAME = {c.name: c for c in CASES}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(repo, expected_commit):
    repo = Path(repo).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("An exact 40-character expected commit is required")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, timeout=15
        ).strip()

    head = git("rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError(f"Commit mismatch: {head} != {expected_commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked worktree changes would invalidate commit provenance")
    for name, wanted in SOURCE_HASHES.items():
        if sha256(repo / name) != wanted:
            raise ValueError(f"Reviewed source hash mismatch: {name}")
    return {"commit": head, "source_hashes": dict(SOURCE_HASHES)}


def expected_parameters(case, rank):
    """Independent inventory for the pinned tiny configuration, after EP slicing.

    47 unique full-model keys. Every local key and shape is required even if the
    reference and pipeline accidentally omit the same parameter.
    """
    if not 0 <= rank < case.world:
        raise ValueError("Rank outside world")
    pp_rank = rank // (case.world // case.pp)
    shapes = {}
    for chunk, stage in enumerate((pp_rank, 2 * case.pp - 1 - pp_rank)):
        if stage == 0:
            shapes[f"{chunk}.embed_tokens.weight"] = (152064, 128)
        if stage == 2 * case.pp - 1:
            shapes[f"{chunk}.norm.weight"] = (128,)
            shapes[f"{chunk}.lm_head.weight"] = (152064, 128)
        layers_per_stage = 4 // (2 * case.pp)
        for layer in range(stage * layers_per_stage, (stage + 1) * layers_per_stage):
            prefix = f"{chunk}.layers.{layer}."
            per_layer = {
                "input_layernorm.weight": (128,),
                "post_attention_layernorm.weight": (128,),
                "self_attn.q_proj.weight": (256, 128),
                "self_attn.k_proj.weight": (128, 128),
                "self_attn.v_proj.weight": (128, 128),
                "self_attn.o_proj.weight": (128, 256),
                "self_attn.q_norm.weight": (128,),
                "self_attn.k_norm.weight": (128,),
                "mlp.gate.weight": (4, 128),
                "mlp.experts.gate_up_proj": (4 // case.ep, 128, 128),
                "mlp.experts.down_proj": (4 // case.ep, 128, 64),
            }
            shapes.update({prefix + name: shape for name, shape in per_layer.items()})
    return shapes


def canonical(name):
    prefix, separator, suffix = name.partition(".")
    if prefix not in ("0", "1") or not separator:
        raise ValueError(f"Not a two-chunk runtime parameter: {name}")
    return suffix


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
