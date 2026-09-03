"""
Test DualPipeV against a single-device reference.
The loss and gradients are compared with the reference implementation.
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.distributed.fsdp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig

from pithtrain.contexts import distributed, training
from pithtrain.dualpipe import DualPipeV, Microbatch
from pithtrain.models.deepseek_v2 import DeepSeekV2Model, DeepSeekV2MoEGate
from pithtrain.models.gpt_oss import GptOssExperts, GptOssModel, GptOssTopKRouter
from pithtrain.models.qwen3_moe import Qwen3MoeGate, Qwen3MoeModel
from pithtrain.models.qwen35_moe import Qwen35MoeModel, Qwen35MoeTopKRouter
from pithtrain.modules.distributed import DistributedCfg, setup_distributed
from pithtrain.operators.grouped_linear import GroupedLinear


def fill_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, GroupedLinear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
    elif isinstance(module, GptOssExperts):
        # Raw nn.Parameter - the GroupedLinear branch above doesn't reach them.
        nn.init.xavier_uniform_(module.gate_up_proj, gain=1.0)
        nn.init.xavier_uniform_(module.down_proj, gain=1.0)
    elif isinstance(
        module, (DeepSeekV2MoEGate, Qwen3MoeGate, GptOssTopKRouter, Qwen35MoeTopKRouter)
    ):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def calculate_difference(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / (x * x + y * y).sum().item()
    return cos_diff


def criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output = output.to(torch.float32)
    target = target.to(torch.float32)
    return F.mse_loss(output, target).clone()


def objective(
    model_outputs: Tuple[torch.Tensor, ...],
    objective_inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrap criterion in the objective contract, returning the detached loss to the caller."""
    (output,) = model_outputs
    (target,) = objective_inputs
    loss = criterion(output, target)
    return loss, loss.detach()


def reference_step(chunks, model: DeepSeekV2Model):
    """Run the reference forward/backward over the same micro-batches DualPipeV will see."""
    ys, ls = [], []
    for micro_x, micro_l, cu in chunks:
        micro_y = model.reference_forward(micro_x, cu)
        loss = criterion(micro_y, micro_l)
        loss.backward()
        ys.append(micro_y)
        ls.append(loss)
    return torch.stack(ls), ys


def shard_layers(layers: nn.ModuleDict, stage_id: int, num_stages: int, config):
    num_local_layers = [config.num_hidden_layers // num_stages for _ in range(num_stages)]
    layers_per_stage_residual = config.num_hidden_layers % num_stages
    for i in range(layers_per_stage_residual):
        num_local_layers[(1 - (i % 2) * 2) * (i // 2) - (i % 2)] += 1
    layer_id_begin = sum(num_local_layers[:stage_id])
    layer_id_end = layer_id_begin + num_local_layers[stage_id]
    return nn.ModuleDict({str(i): layers[str(i)] for i in range(layer_id_begin, layer_id_end)})


def shard_experts(model, ep_rank, ep_size):
    num_experts = None
    for child in model.children():
        if isinstance(child, GroupedLinear):
            num_experts = child.num_groups
            break
    if num_experts is None:
        gu = getattr(model, "gate_up_proj", None)
        if isinstance(gu, nn.Parameter):
            num_experts = getattr(model, "num_experts", None)
    if num_experts is not None and num_experts % ep_size == 0 and num_experts > 1:
        experts_per_ep_rank = num_experts // ep_size
        expert_begin = ep_rank * experts_per_ep_rank
        expert_end = (ep_rank + 1) * experts_per_ep_rank
        for pname, param in list(model.named_parameters(recurse=False)):
            if param.dim() >= 1 and param.shape[0] == num_experts:
                new_param = nn.Parameter(
                    param.data[expert_begin:expert_end].clone(),
                    requires_grad=param.requires_grad,
                )
                if param.grad is not None:
                    new_param.grad = param.grad[expert_begin:expert_end].clone()
                setattr(model, pname, new_param)

    for name, child in model.named_children():
        if isinstance(child, GroupedLinear):
            experts_per_ep_rank = child.num_groups // ep_size
            new_mod = GroupedLinear(experts_per_ep_rank, child.in_features, child.out_features)
            expert_begin = ep_rank * experts_per_ep_rank
            expert_end = (ep_rank + 1) * experts_per_ep_rank
            new_mod.weight.data = child.weight.data[expert_begin:expert_end]
            new_mod.weight.grad = child.weight.grad[expert_begin:expert_end]
            setattr(model, name, new_mod)
        else:
            shard_experts(child, ep_rank, ep_size)


def apply_fsdp(model, mesh: torch.distributed.DeviceMesh, dtype):
    # MoE params are sharded by EP, we only additionally shard on the DP dimension
    moe_fsdp_mesh = mesh["dp"]
    # For other params, we shard on the both DP and EP dimensions
    other_fsdp_mesh = mesh["dp", "ep"]._flatten()
    mp = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    # FSDP recommends shard models from the bottom to the top.
    for i in range(2):
        if model[i].embed_tokens is not None:
            fully_shard(
                model[i].embed_tokens,
                mesh=other_fsdp_mesh,
                reshard_after_forward=True,
                mp_policy=mp,
            )
        if model[i].norm is not None:
            assert model[i].lm_head is not None
            fully_shard(
                model[i].norm, mesh=other_fsdp_mesh, reshard_after_forward=True, mp_policy=mp
            )
            fully_shard(
                model[i].lm_head, mesh=other_fsdp_mesh, reshard_after_forward=True, mp_policy=mp
            )
        for layer in model[i].layers.values():
            if hasattr(layer.mlp, "experts"):
                fully_shard(
                    layer.mlp.experts, mesh=moe_fsdp_mesh, reshard_after_forward=False, mp_policy=mp
                )
            fully_shard(layer, mesh=other_fsdp_mesh, reshard_after_forward=False, mp_policy=mp)
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_stage1")
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_stage3")
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_stage5")
        fully_shard(model[i], mesh=other_fsdp_mesh, reshard_after_forward=False, mp_policy=mp)
    return model


def main(model_name: str):
    """
    Main testing function.

    Parameters
    ----------
    model_name : str
        Model name or local config path.
    """

    ep_group = distributed.ep_group
    dp_size, pp_size, ep_size = distributed.dp_size, distributed.pp_size, distributed.ep_size
    pp_rank, ep_rank = distributed.pp_rank, distributed.ep_rank

    if distributed.rank == 0:
        print("[INFO] Testing FSDP x DualPipeV x EP with model: %s" % model_name, flush=True)
        print(
            "[INFO] DP size: %d, PP size: %d, EP size: %d." % (dp_size, pp_size, ep_size),
            flush=True,
        )
    torch.distributed.barrier()

    torch.manual_seed(1234)
    torch.set_default_device(torch.cuda.current_device())
    dtype = torch.bfloat16

    # This test builds models directly, bypassing setup_model; bind the BF16 linear backend it sets.
    training.fp8 = False
    training.Linear = nn.Linear
    training.GroupedLinear = GroupedLinear

    packed = os.environ.get("PACKED_SEQLEN", "0") == "1"
    ragged = os.environ.get("RAGGED_MICROBATCH", "0") == "1"
    micro_batch_size = 1 if packed else 3  # packing pins mbs to 1
    num_chunks, sequence_length = 20, 128

    config_path = Path(__file__).resolve().parent.parent / model_name
    config = AutoConfig.from_pretrained(config_path)

    if config.model_type == "deepseek_v2":
        ModelClass = DeepSeekV2Model
        config.num_hidden_layers = min(config.num_hidden_layers, 8)
    elif config.model_type == "qwen3_moe":
        ModelClass = Qwen3MoeModel
        config.num_hidden_layers = min(config.num_hidden_layers, 8)
    elif config.model_type == "gpt_oss":
        ModelClass = GptOssModel
        keep = min(config.num_hidden_layers, 8)
        if getattr(config, "layer_types", None) is not None:
            config.layer_types = config.layer_types[:keep]
        config.num_hidden_layers = keep
        config.vocab_size = 8192
    elif config.model_type == "qwen3_5_moe_text":
        ModelClass = Qwen35MoeModel
        keep = min(config.num_hidden_layers, 8)
        if getattr(config, "layer_types", None) is not None:
            config.layer_types = config.layer_types[:keep]
        config.num_hidden_layers = keep
        config.num_experts = 32
        config.vocab_size = 8192
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    torch.distributed.barrier()
    torch.manual_seed(1234)

    vocab_size = config.vocab_size

    # Build the micro-batches, one entry per global chunk, ordered expert-parallel-major so
    # rank r owns chunks [r * num_chunks, (r + 1) * num_chunks).
    #
    # Two steps are run, so the shapes agreed for one step cannot leak into the next: under
    # ragged the second step uses different sequence lengths than the first.
    num_global_chunks = ep_size * num_chunks
    label_scale = 10.0  # scaled up so MSE grads on small bias terms clear the bf16 noise floor

    def build_chunks(step_index):
        if ragged:
            # Vary the sequence length per micro-batch so every P2P buffer must be sized on its
            # own, and shift the pattern per step so the second step's shapes differ.
            pattern = [
                sequence_length,
                sequence_length - 64,
                sequence_length + 32,
                sequence_length - 32,
            ]
            offset = step_index * 2
            seqlens = [pattern[(j + offset) % len(pattern)] for j in range(num_global_chunks)]
        else:
            seqlens = [sequence_length] * num_global_chunks

        chunks = []
        for j, seqlen in enumerate(seqlens):
            chunk_x = torch.randint(0, vocab_size, (micro_batch_size, seqlen))
            chunk_l = label_scale * torch.randn(micro_batch_size, seqlen, vocab_size, dtype=dtype)
            chunk_cu = None
            if packed:
                # Split each sample into documents and check the block-diagonal cu_seqlens path
                # against the reference (attention only; the MSE loss is unaffected). Under
                # ragged the document count varies too, so each micro-batch carries a
                # differently sized cu_seqlens.
                num_docs = 2 + (j % 2) if ragged else 3
                bounds = [round(seqlen * k / num_docs) for k in range(num_docs + 1)]
                chunk_cu = torch.tensor(bounds, dtype=torch.int32)
            chunks.append((chunk_x, chunk_l, chunk_cu))
        return chunks

    chunk_steps = [build_chunks(0), build_chunks(1)]

    # Reference runs single-device (pp=ep=1); restore the real mesh before DualPipeV.
    distributed.pp_size = distributed.ep_size = 1
    full_modules = ModelClass(config, phase=-1)
    full_modules.to(dtype=dtype)
    full_modules.apply(fill_weights)

    # Run the reference step.
    if distributed.rank == 0:
        print("[INFO] Running the reference step.", flush=True)
    torch.distributed.barrier()

    loss_refs = [reference_step(chunks, full_modules)[0] for chunks in chunk_steps]
    distributed.pp_size, distributed.ep_size = pp_size, ep_size

    if distributed.rank == 0:
        print("[INFO] Completed the reference step.", flush=True)
    torch.distributed.barrier()

    # Shard the full modules whose weights and gradients will be used for checking.
    num_stages = pp_size * 2
    local_full_modules = []

    local_full_modules.append(ModelClass(config, phase=0))
    local_full_modules.append(ModelClass(config, phase=1))

    local_full_modules = nn.Sequential(*local_full_modules)
    if pp_rank == 0:
        local_full_modules[0].embed_tokens = full_modules.embed_tokens
        local_full_modules[1].norm = full_modules.norm
        local_full_modules[1].lm_head = full_modules.lm_head
    local_full_modules[0].layers = shard_layers(full_modules.layers, pp_rank, num_stages, config)
    local_full_modules[1].layers = shard_layers(
        full_modules.layers, num_stages - 1 - pp_rank, num_stages, config
    )
    if ep_size > 1:
        shard_experts(local_full_modules[0], ep_rank=ep_rank, ep_size=ep_size)
        shard_experts(local_full_modules[1], ep_rank=ep_rank, ep_size=ep_size)

    # Create the local modules with the same weights but zero gradients.
    local_modules = []

    local_modules.append(ModelClass(config, phase=0))
    local_modules.append(ModelClass(config, phase=1))

    local_modules = nn.Sequential(*local_modules)
    local_modules.to(dtype=dtype)
    local_modules[0].load_state_dict(local_full_modules[0].state_dict())
    local_modules[1].load_state_dict(local_full_modules[1].state_dict())
    local_modules.zero_grad()
    apply_fsdp(local_modules, distributed.device_mesh, dtype)

    # Wrap the modules with DualPipeV.
    dualpipev_model = DualPipeV(local_modules)

    # Run the DualPipeV step.
    for step_index, chunks in enumerate(chunk_steps):
        # Every pipeline rank builds the same micro-batches, so each derives its own P2P
        # shapes; only the first rank reads the tensors.
        local_chunks = chunks[ep_rank * num_chunks : (ep_rank + 1) * num_chunks]
        microbatches = [
            Microbatch(model_inputs=(x,), cu_seqlens=cu, objective_inputs=(lab,))
            for x, lab, cu in local_chunks
        ]

        if distributed.rank == 0:
            print("[INFO] Running DualPipeV step %d." % step_index, flush=True)
        torch.distributed.barrier()

        objective_outputs = dualpipev_model.step(microbatches, objective)

        if distributed.rank == 0:
            print("[INFO] Completed DualPipeV step %d." % step_index, flush=True)
        torch.distributed.barrier()

        # Validate the loss.
        if pp_rank == 0:
            loss = torch.stack(objective_outputs)
            loss_ref = loss_refs[step_index].reshape(ep_size, -1)[ep_rank]
            print(
                "[INFO] rank-%d, step %d, loss: %s, loss_ref: %s"
                % (distributed.rank, step_index, loss, loss_ref),
                flush=True,
            )
            assert torch.allclose(loss, loss_ref, rtol=1e-3, atol=1e-3)
        else:
            assert objective_outputs == []

        if distributed.rank == 0:
            print("[INFO] Loss matches the reference at step %d." % step_index, flush=True)
        torch.distributed.barrier()

    # Validate the gradients.
    eps = 1e-2
    largest_diff = 0
    largest_diff_param = None
    failed = False

    for (n, p), p_ref in zip(local_modules.named_parameters(), local_full_modules.parameters()):
        if p.grad is None:
            print(
                "[warn] rank-%d, Parameter %s doesn't have a gradient, skipping."
                % (distributed.rank, n)
            )
            continue
        p_grad = p.grad
        if isinstance(p_grad, torch.distributed.tensor.DTensor):
            p_grad = p_grad.full_tensor()
        if ".experts." not in n and ep_size > 1:
            p_grad = p_grad.clone()
            torch.distributed.all_reduce(p_grad, group=ep_group)
        if torch.all(p_grad == 0) and torch.all(p_ref.grad == 0):
            print(
                "[warn] rank-%d, Parameter %s has all-zero gradient, skipping."
                % (distributed.rank, n)
            )
            continue
        # Reference accumulates in bf16, DualPipeV in fp32; cosine-diff on
        # noise-floor grads (e.g. gpt-oss router.bias ~1e-8) is meaningless.
        ref_max = p_ref.grad.abs().max().item()
        if ref_max < 1e-5:
            print(
                "[warn] rank-%d, Parameter %s grad max=%.2e at bf16 noise floor, skipping."
                % (distributed.rank, n, ref_max)
            )
            continue
        diff = calculate_difference(p_grad, p_ref.grad)
        if diff > largest_diff:
            largest_diff = diff
            largest_diff_param = n
        # A_log and dt_bias are the gated-delta decay params, both feeding the gate
        # g = -exp(A_log) * softplus(a + dt_bias) that drives the linear-attention recurrence;
        # their tiny grads accumulate over it and are dominated by non-deterministic bf16
        # scatter-add / all-to-all ordering, varying run-to-run (~0.005-0.016) past the eps.
        param_eps = 3e-2 if n.endswith(("linear_attn.A_log", "linear_attn.dt_bias")) else eps
        if diff > param_eps:
            failed = True
            print(
                "[ERROR] rank-%d, Parameter %s grad mismatch: diff=%.6f, eps=%.6f, p_grad:%s..., p_ref.grad:%s..."
                % (
                    distributed.rank,
                    n,
                    diff,
                    param_eps,
                    p_grad.flatten()[:5],
                    p_ref.grad.flatten()[:5],
                )
            )
    assert not failed

    for rank in range(distributed.world_size):
        if rank == distributed.rank:
            print(
                "[INFO] rank-%d, Gradient check completed. Largest diff = %.6f for param %s."
                % (distributed.rank, largest_diff, largest_diff_param)
            )
        torch.distributed.barrier()

    if distributed.rank == 0:
        print("[INFO] All gradients match the reference.", flush=True)
    torch.distributed.barrier()


@record
def _entry() -> None:
    models = []
    models.append("examples/pretrain_lm/deepseek-v2-lite/config.json")
    models.append("examples/pretrain_lm/qwen3-30b-a3b/config.json")
    models.append("examples/pretrain_lm/gpt-oss-20b/config.json")
    models.append("examples/pretrain_lm/gpt-oss-120b/config.json")
    models.append("examples/pretrain_lm/qwen3.5-35b-a3b/config.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--model", type=str, choices=models, required=True)
    parsed = parser.parse_args()

    cfg = SimpleNamespace()
    cfg.distributed = DistributedCfg()
    cfg.distributed.pipeline_parallel_size = parsed.pp_size
    cfg.distributed.expert_parallel_size = parsed.ep_size

    setup_distributed(cfg)
    main(parsed.model)


if __name__ == "__main__":
    _entry()
