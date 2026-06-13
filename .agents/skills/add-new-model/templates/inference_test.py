"""
End-to-end DualPipeV inference test — TEMPLATE.

Copy this to ``tests/test_<model>_inference.py`` (ad-hoc, not committed),
replace the TODO_MODEL markers with the new model's class and HF ID, and
run it through the pp/ep scaling ladder.

The purpose of this test is to verify that **real released weights**
produce coherent text when run through **DualPipeV's 5-stage pipeline**
(not the raw model forward).  This catches:
  • checkpoint-conversion layout bugs (symptom: gibberish even after
    FSDP gradient test passes with random weights),
  • pipeline plumbing issues that only manifest with loaded weights,
  • regressions in the stage scheduler under inference.

Gradual scaling — same ladder as training:
    # 1 GPU
    CUDA_VISIBLE_DEVICES=<g0> timeout 180 torchrun --nproc-per-node=1 \\
        --rdzv-backend=c10d --rdzv-endpoint=localhost:15213 \\
        tests/test_<model>_inference.py --pp-size 1 --ep-size 1

    # 2 GPUs — pp
    CUDA_VISIBLE_DEVICES=<g0>,<g1> timeout 180 torchrun --nproc-per-node=2 ... \\
        tests/test_<model>_inference.py --pp-size 2 --ep-size 1

    # 2 GPUs — ep
    CUDA_VISIBLE_DEVICES=<g0>,<g1> timeout 180 torchrun --nproc-per-node=2 ... \\
        tests/test_<model>_inference.py --pp-size 1 --ep-size 2

    # 4 GPUs
    CUDA_VISIBLE_DEVICES=<g0>,<g1>,<g2>,<g3> timeout 180 torchrun --nproc-per-node=4 ... \\
        tests/test_<model>_inference.py --pp-size 2 --ep-size 2

**ALWAYS** check `nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv`
before each torchrun to pick actually-free GPUs.  Shared cluster.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoTokenizer

from pithtrain.dualpipe import DualPipeV, set_p2p_tensor_dtype, set_p2p_tensor_shapes
from pithtrain.layers.factory import ModelImplMode

# TODO_MODEL: import the model class you're testing.
from pithtrain.models.<model> import <Model>Model  # noqa: F821
from pithtrain.modules.distributed import DistributedCfg, DistributedCtx, distributed_context
from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg
from pithtrain.tasks.convert_checkpoint import launch as convert_launch


# TODO_MODEL: default HF id and local root.  The root holds hf/ and dcp/
# subdirs; both are created lazily on rank 0 when missing.
DEFAULT_HF_ID = "TODO_MODEL"  # e.g. "mistralai/Mixtral-8x7B-v0.1"
DEFAULT_ROOT = Path("/tmp/pithtrain-<model>")  # TODO_MODEL


# ── checkpoint prep (rank 0 only) ─────────────────────────────────────────


def _ensure_dcp_checkpoint(hf_id: str, root: Path):
    """Download HF snapshot and convert to DCP if not already present."""
    hf_dir = root / "hf"
    dcp_dir = root / "dcp"
    if not dcp_dir.exists():
        if not hf_dir.exists() or not (hf_dir / "config.json").exists():
            print(f"[INFO] Downloading {hf_id} into {hf_dir}", flush=True)
            snapshot_download(
                repo_id=hf_id,
                local_dir=hf_dir,
                # TODO_MODEL: tighten the allow_patterns if the repo has
                # extra files you don't need (e.g. original PyTorch weights
                # when you're loading safetensors).
                allow_patterns=["*.json", "model*.safetensors", "*.txt", "tokenizer*"],
                ignore_patterns=["original/*", "metal/*"],
            )
        print(f"[INFO] Converting HF → DCP into {dcp_dir}", flush=True)
        cfg = ConvertCheckpointCfg()
        cfg.operation = "hf2dcp"
        cfg.load_path = hf_dir
        cfg.save_path = dcp_dir
        convert_launch(cfg)
    return dcp_dir, hf_dir


# ── DCP → localized state_dict ────────────────────────────────────────────


def _load_dcp_canonical(dcp_path: Path) -> dict:
    """Read the whole canonical DCP into a CPU dict.

    IMPORTANT: allocate on CPU.  A 20B model is ~42 GB, a 120B model is
    ~230 GB — if the default device is CUDA, this OOMs before the model
    is even built.  See reference/pitfalls.md §load-canonical-to-cpu.
    """
    metadata = FileSystemReader(str(dcp_path)).read_metadata()
    prefix = "app.model."
    sd = {}
    for key, meta in metadata.state_dict_metadata.items():
        if key.startswith(prefix):
            sd[key] = torch.empty(meta.size, dtype=meta.properties.dtype, device="cpu")
    dcp.load(sd, checkpoint_id=dcp_path, no_dist=True)
    return {k.removeprefix(prefix): v for k, v in sd.items()}


def _load_localized_from_canonical(canonical: dict, dualpipev: DualPipeV) -> dict:
    """Remap canonical per-expert indexed keys → local FQNs and stack
    experts for this rank's EP slice.

    Mirrors ``modules.checkpoint.to_localized_model`` but without DTensor
    wrapping since inference doesn't use FSDP-sharded parameters.
    """
    import re

    from pithtrain.modules.checkpoint import MODULE_PREFIX_RE, expert_range

    named_modules = dict(dualpipev.named_modules())
    indexed = re.compile(r"(.*\.mlp\.experts)\.(\d+)\.(.*)")

    fqn_map = {}
    for k in dualpipev.state_dict().keys():
        canon = MODULE_PREFIX_RE.sub("", k)
        fqn_map[canon] = k

    by_local_key: dict[str, dict[int, torch.Tensor]] = {}
    plain: dict[str, torch.Tensor] = {}
    for canon, tensor in canonical.items():
        m = indexed.match(canon)
        if m:
            prefix, idx_str, suffix = m.groups()
            stacked_canon = f"{prefix}.{suffix}"
            local = fqn_map.get(stacked_canon)
            if local is None:
                continue
            moe_path = local.partition(".experts.")[0]
            moe = named_modules.get(moe_path)
            if moe is None or not hasattr(moe, "ep_rank"):
                continue
            start, end = expert_range(moe)
            idx = int(idx_str)
            if start <= idx < end:
                by_local_key.setdefault(local, {})[idx - start] = tensor
        else:
            local = fqn_map.get(canon)
            if local is not None:
                plain[local] = tensor

    for local_key, by_idx in by_local_key.items():
        plain[local_key] = torch.stack([by_idx[i] for i in sorted(by_idx)])

    device = torch.cuda.current_device()
    return {k: v.to(device) for k, v in plain.items()}


# ── model builder ─────────────────────────────────────────────────────────


def build_dualpipev(config, ctx: DistributedCtx, dtype: torch.dtype) -> DualPipeV:
    """Build the DualPipeV wrapper around two model replicas (stage_id and
    its mirror) for this pp_rank.  Inference doesn't wrap with FSDP —
    DP=CP=1, so sharding has no benefit and DTensor-in-state_dict makes
    the localization pass messier.
    """
    pp_size = ctx.pp_size
    pp_rank = ctx.pp_rank
    ep_group = ctx.device_mesh.get_group("ep")
    pp_group = ctx.device_mesh.get_group("pp")

    config.ep_size = ctx.ep_size
    num_stages = pp_size * 2
    modules = [
        # TODO_MODEL: if the model takes cp_group or other kwargs, pass them here.
        <Model>Model(config, num_stages=num_stages, stage_id=pp_rank, ep_group=ep_group),  # noqa: F821
        <Model>Model(  # noqa: F821
            config, num_stages=num_stages, stage_id=num_stages - 1 - pp_rank, ep_group=ep_group
        ),
    ]
    modules = nn.Sequential(*modules)
    modules.to(device=torch.cuda.current_device(), dtype=dtype)

    return DualPipeV(modules, pp_group=pp_group, ep_group=ep_group)


# ── autoregressive decode via DualPipeV ───────────────────────────────────


@torch.no_grad()
def generate(
    dualpipev: DualPipeV,
    ctx: DistributedCtx,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    dtype: torch.dtype,
    hidden_size: int,
) -> list[str]:
    """Greedy decode with static-seq-len buffer.

    DualPipeV requires ``num_chunks >= pp_size * 2``.  We use one prompt
    per chunk so ``num_chunks = len(prompts)``.  The caller must pass a
    batch of at least ``pp_size * 2`` prompts.
    """
    device = torch.cuda.current_device()
    assert len(prompts) >= ctx.pp_size * 2, (
        f"prompts batch={len(prompts)} must be >= pp_size*2={ctx.pp_size * 2}"
    )
    num_chunks = len(prompts)

    enc = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
    pad_id = tokenizer.pad_token_id or 0

    # Trim every prompt to the length of the SHORTEST so the batch is
    # uniform with no left-padding.  Left-padding would contaminate
    # causal attention on the heavily-padded prompts — see
    # reference/pitfalls.md §left-padding.
    prompt_len = min(t.shape[0] for t in enc)

    # Static-seq-len decode: allocate one buffer sized
    # [batch, prompt_len + max_new_tokens] up front and advance a cursor.
    # Keeps DualPipeV's p2p shapes constant and the @torch.compile(fullgraph=True)
    # inside the attention block compiles ONCE instead of once per new
    # seq_len.  See reference/compile.md §inference-compile.
    max_seq_len = prompt_len + max_new_tokens
    buffer = torch.full((len(enc), max_seq_len), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(enc):
        buffer[i, :prompt_len] = t[:prompt_len].to(device)
    cursor = prompt_len
    generated = [[] for _ in range(len(prompts))]

    set_p2p_tensor_shapes([(1, max_seq_len, hidden_size)])
    set_p2p_tensor_dtype(dtype)

    for step in range(max_new_tokens):
        inputs = buffer if ctx.pp_rank == 0 else None
        loss, outputs = dualpipev.step(
            inputs,
            num_chunks=num_chunks,
            criterion=None,
            labels=[],
            return_outputs=True,
        )

        next_tok = torch.empty(len(prompts), dtype=torch.long, device=device)
        if ctx.pp_rank == 0:
            logits = outputs
            if isinstance(logits, tuple):
                logits = logits[0]
            # Use the logit at position (cursor - 1): the prediction made
            # by the last real input position.  Positions beyond the cursor
            # hold pad_id and don't contaminate causal attention from
            # filled positions.
            next_tok = logits[:, cursor - 1, :].float().argmax(dim=-1)
        torch.distributed.broadcast(next_tok, src=0)

        for i, tok in enumerate(next_tok.tolist()):
            generated[i].append(tok)
        buffer[:, cursor] = next_tok
        cursor += 1

    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated]


# ── main ──────────────────────────────────────────────────────────────────


def main(ctx: DistributedCtx, args):
    dtype = torch.bfloat16
    torch.cuda.set_device(ctx.local_rank)
    torch.set_default_device(torch.cuda.current_device())
    # Use the 5-stage pipeline path, not reference_forward — the whole
    # point of this test is to exercise DualPipeV end-to-end.
    ModelImplMode.use_reference_fwd = False

    if ctx.rank == 0:
        dcp_path, hf_path = _ensure_dcp_checkpoint(args.hf_id, args.root)
    else:
        dcp_path = args.root / "dcp"
        hf_path = args.root / "hf"
    torch.distributed.barrier()

    config = AutoConfig.from_pretrained(hf_path)
    dualpipev = build_dualpipev(config, ctx, dtype)

    if ctx.rank == 0:
        print(f"[INFO] Loading canonical DCP from {dcp_path}", flush=True)
    canonical = _load_dcp_canonical(dcp_path)
    local_sd = _load_localized_from_canonical(canonical, dualpipev)
    missing, unexpected = dualpipev.load_state_dict(local_sd, strict=False)
    if ctx.rank == 0 and missing:
        print(f"[WARN] missing keys (first 10): {list(missing)[:10]}", flush=True)
    if ctx.rank == 0 and unexpected:
        print(f"[WARN] unexpected keys (first 10): {list(unexpected)[:10]}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    # Pad the prompt batch up to pp_size * 2 if the user provided fewer —
    # DualPipeV requires num_chunks >= pp_size * 2.
    base_prompts = args.prompts
    min_batch = ctx.pp_size * 2
    if len(base_prompts) >= min_batch:
        prompts = list(base_prompts)
    else:
        prompts = (base_prompts * ((min_batch + len(base_prompts) - 1) // len(base_prompts)))[
            :min_batch
        ]

    if ctx.rank == 0:
        print(
            f"[INFO] Generating with pp={ctx.pp_size}, ep={ctx.ep_size}, batch={len(prompts)}",
            flush=True,
        )
    outputs = generate(
        dualpipev,
        ctx,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        hidden_size=config.hidden_size,
    )
    if ctx.rank == 0:
        for p, o in zip(prompts, outputs):
            print(f"\n[PROMPT] {p!r}", flush=True)
            print(f"[OUTPUT] {o!r}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--hf-id", default=DEFAULT_HF_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "The capital of France is",
            "def fibonacci(n):\n    if n <= 1:",
            "Once upon a time",
            "To compute the Fibonacci sequence, you can use",
        ],
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    cfg, ctx = SimpleNamespace(), SimpleNamespace()
    cfg.distributed = DistributedCfg()
    cfg.distributed.pipeline_parallel_size = args.pp_size
    cfg.distributed.expert_parallel_size = args.ep_size
    ctx.distributed = DistributedCtx()

    with distributed_context(cfg, ctx):
        main(ctx.distributed, args)
