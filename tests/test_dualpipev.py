"""
Compare DualPipeV loss, gradients and canonical weight export with a single-device reference.
"""

import argparse
import os
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.distributed.fsdp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig

from pithtrain.contexts import distributed, training
from pithtrain.models.deepseek_v2 import DeepSeekV2Model, DeepSeekV2MoEGate
from pithtrain.models.gpt_oss import GptOssExperts, GptOssModel, GptOssTopKRouter
from pithtrain.models.qwen3_moe import Qwen3MoeGate, Qwen3MoeModel
from pithtrain.models.qwen35_moe import Qwen35MoeModel, Qwen35MoeTopKRouter
from pithtrain.modules.checkpoint import iter_canonical_parameters
from pithtrain.modules.distributed import DistributedCfg, setup_distributed
from pithtrain.modules.wgrad_accumulation import install_raw_main_grad, install_wgrad_accumulation
from pithtrain.operators.cp_sequence import zigzag_varlen_index
from pithtrain.operators.grouped_linear import FP8GroupedLinear, GroupedLinear
from pithtrain.operators.linear import BF16Linear, FP8Linear
from pithtrain.pipeline import DualPipeV, Microbatch


def fill_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (GroupedLinear, FP8GroupedLinear)):
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
    (output,) = model_outputs
    (target,) = objective_inputs
    loss = criterion(output, target)
    return loss, loss.detach()


def records_empty(model: DualPipeV) -> bool:
    records = []
    for chunk in model.chunk_records[0] + model.chunk_records[1]:
        records += [chunk.prolog, chunk.epilog]
        records += [getattr(layer, field.name) for layer in chunk.layers for field in fields(layer)]
    return all(getattr(record, field.name) is None for record in records if record is not None for field in fields(record))  # fmt: skip


def reference_step(chunks, model: DeepSeekV2Model):
    outputs, losses = [], []
    for i, (micro_x, micro_labels, cu_seqlens) in enumerate(chunks):
        training.current_microbatch = i
        micro_y = model.reference_forward(micro_x, cu_seqlens)
        loss = criterion(micro_y, micro_labels)
        loss.backward()
        outputs.append(micro_y)
        losses.append(loss)
    return torch.stack(losses), outputs


def zigzag_shard(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """
    The CP zigzag layout of the loader and forward_posemb, restated here rather
    than calling zigzag_spans.
    """
    if cp_size == 1:
        return x
    block = x.shape[1] // (2 * cp_size)
    front = cp_rank * block
    back = (2 * cp_size - cp_rank - 1) * block
    return torch.cat([x[:, front : front + block], x[:, back : back + block]], dim=1)


def replay_store(layer_idx: int, num_experts: int, micro_batch_size: int, offset: int, chunks):
    """
    Serve deterministic routes by layer and global micro-batch, cut to this rank's zigzag shard.
    Exclude the last expert to detect ignored replay, and alternate int64/int32 across layers.
    """

    def replay(num_tokens: int, top_k: int) -> torch.Tensor:
        microbatch = offset + training.current_microbatch
        seq_len = num_tokens // micro_batch_size * distributed.cp_size
        # A CPU generator draws the same routes on every rank, the reference's included.
        generator = torch.Generator().manual_seed(microbatch * 1000 + layer_idx)
        routes = torch.rand(micro_batch_size, seq_len, num_experts - 1, generator=generator, device="cpu")  # fmt: skip
        routes = routes.topk(top_k).indices
        cu = chunks[training.current_microbatch][2]
        if cu is None:
            routes = zigzag_shard(routes, distributed.cp_rank, distributed.cp_size)
        else:
            # Match the actual packed documents used to shard the hidden states.
            index = zigzag_varlen_index(distributed.cp_rank, distributed.cp_size, cu.cpu(), seq_len)
            routes = routes[:, index]
        dtype = torch.int32 if layer_idx % 2 else torch.int64
        return routes.reshape(num_tokens, top_k).to(distributed.device, dtype)

    return replay


def install_router_replay(model: nn.Module, micro_batch_size: int, offset: int, chunks) -> None:
    for layer in model.layers.values():
        gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
        if gate is not None:
            gate.router_replay = replay_store(
                layer.idx, gate.num_experts, micro_batch_size, offset, chunks
            )


def shard_layers(layers: nn.ModuleDict, local: nn.Module) -> nn.ModuleDict:
    return nn.ModuleDict({i: layers[i] for i in local.layers})


def shard_experts(model, ep_rank, ep_size):
    num_experts = None
    for child in model.children():
        if isinstance(child, (GroupedLinear, FP8GroupedLinear)):
            num_experts = child.num_groups
            break
    if num_experts is None:
        gate_up = getattr(model, "gate_up_proj", None)
        if isinstance(gate_up, nn.Parameter):
            num_experts = getattr(model, "num_experts", None)
    if num_experts is not None and num_experts % ep_size == 0 and num_experts > 1:
        experts_per_ep_rank = num_experts // ep_size
        expert_begin = ep_rank * experts_per_ep_rank
        expert_end = (ep_rank + 1) * experts_per_ep_rank
        for param_name, param in list(model.named_parameters(recurse=False)):
            if param.dim() >= 1 and param.shape[0] == num_experts:
                new_param = nn.Parameter(
                    param.data[expert_begin:expert_end].clone(),
                    requires_grad=param.requires_grad,
                )
                if param.grad is not None:
                    new_param.grad = param.grad[expert_begin:expert_end].clone()
                setattr(model, param_name, new_param)

    for name, child in model.named_children():
        if isinstance(child, (GroupedLinear, FP8GroupedLinear)):
            experts_per_ep_rank = child.num_groups // ep_size
            new_mod = type(child)(experts_per_ep_rank, child.in_features, child.out_features)
            expert_begin = ep_rank * experts_per_ep_rank
            expert_end = (ep_rank + 1) * experts_per_ep_rank
            new_mod.weight.data = child.weight.data[expert_begin:expert_end]
            new_mod.weight.grad = child.weight.grad[expert_begin:expert_end]
            setattr(model, name, new_mod)
        else:
            shard_experts(child, ep_rank, ep_size)


def canonical_state_dict(model: nn.Module, ep_rank: int) -> dict:
    """
    Written independently of checkpoint.unpack, so the export check does not reuse the
    code it tests.
    """
    result = {}
    for name, value in model.state_dict().items():
        if ".experts." not in name:
            result[name] = value
            continue
        for idx, expert in enumerate(value, start=ep_rank * value.shape[0]):
            result[name.replace(".experts.", ".experts.%d." % idx, 1)] = expert
    return result


def apply_fsdp(model, dtype):
    # Mirrors modules.training.apply_fsdp: expert parameters shard over the dp axis of the expert
    # view, attention parameters over the flattened dp x cp stage of the attention view.
    expt_fsdp_mesh = distributed.expt_mesh["dp"]
    attn_fsdp_mesh = distributed.attn_mesh["dp", "cp"]._flatten()
    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    # FSDP recommends sharding from the bottom up.
    for i in range(2):
        if model[i].embed_tokens is not None:
            fully_shard(
                model[i].embed_tokens,
                mesh=attn_fsdp_mesh,
                reshard_after_forward=True,
                mp_policy=mp_policy,
            )
        if model[i].norm is not None:
            assert model[i].lm_head is not None
            fully_shard(
                model[i].norm, mesh=attn_fsdp_mesh, reshard_after_forward=True, mp_policy=mp_policy
            )
            fully_shard(
                model[i].lm_head,
                mesh=attn_fsdp_mesh,
                reshard_after_forward=True,
                mp_policy=mp_policy,
            )
        for layer in model[i].layers.values():
            if hasattr(layer.mlp, "experts"):
                fully_shard(
                    layer.mlp.experts,
                    mesh=expt_fsdp_mesh,
                    reshard_after_forward=False,
                    mp_policy=mp_policy,
                )
                if hasattr(layer.mlp.experts, "forward_from_tokens"):
                    torch.distributed.fsdp.register_fsdp_forward_method(
                        layer.mlp.experts, "forward_from_tokens"
                    )
            fully_shard(layer, mesh=attn_fsdp_mesh, reshard_after_forward=False, mp_policy=mp_policy)  # fmt: skip
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_stage1")
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_stage3")
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_stage5")
        fully_shard(model[i], mesh=attn_fsdp_mesh, reshard_after_forward=False, mp_policy=mp_policy)
    # Plain-sum reduction, as in modules.training.apply_fsdp.
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
    return model


def main(model_name: str):
    dp_size, pp_size, ep_size = distributed.dp_size, distributed.pp_size, distributed.ep_size
    cp_size, cp_rank = distributed.cp_size, distributed.cp_rank
    pp_rank, dp_rank, ep_rank = distributed.pp_rank, distributed.dp_rank, distributed.ep_rank

    if distributed.rank == 0:
        print("[INFO] Testing FSDP x DualPipeV x EP with model: %s" % model_name, flush=True)
        print(
            "[INFO] DP size: %d, PP size: %d, EP size: %d, CP size: %d."
            % (dp_size, pp_size, ep_size, cp_size),
            flush=True,
        )
    torch.distributed.barrier()

    torch.manual_seed(1234)
    torch.set_default_device(torch.cuda.current_device())
    dtype = torch.bfloat16

    # The test builds models directly, bypassing setup_model, so it binds the linear backend itself.
    # The fused classes act as nn.Linear in the single-device reference; fused accumulation is
    # installed on the DualPipeV copy only, as setup_model does.
    training.fp8 = False
    training.Linear = BF16Linear
    training.GroupedLinear = GroupedLinear
    if os.environ.get("FP8", "0") == "1":
        # FP8 on both copies: the reference runs the same FP8 layers unpipelined. Expert weight
        # gradients differ by a few percent, since the 128-row colwise quantization of their wgrad
        # operands follows token order, which the EP dispatch changes.
        training.fp8 = True
        training.Linear = FP8Linear
        training.GroupedLinear = FP8GroupedLinear

    packed = os.environ.get("PACKED_SEQLEN", "0") == "1"
    ragged = os.environ.get("RAGGED_MICROBATCH", "0") == "1"
    replay = os.environ.get("ROUTER_REPLAY", "0") == "1"
    micro_batch_size = 1 if packed else 3  # packing pins mbs to 1
    num_chunks = 20
    sequence_length = int(os.environ.get("SEQUENCE_LENGTH", "128"))

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

    # One micro-batch per global chunk, ordered data-parallel-major so dp rank r owns chunks
    # [r * num_chunks, (r + 1) * num_chunks). dp alone splits the data; the expert rank decides
    # only which experts a rank hosts.
    #
    # Two steps run so the shapes agreed for one step cannot leak into the next.
    num_global_chunks = dp_size * num_chunks
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
                # Ragged batches vary the document count as well as the token count. CP documents
                # are multiples of 2 * cp_size; the first is one token, or one per block under CP.
                num_docs = 2 + (j % 2) if ragged else 3
                unit = 2 * cp_size if cp_size > 1 else 1
                units = seqlen // unit - 1
                bounds = [0] + [
                    unit * (1 + round(units * k / (num_docs - 1))) for k in range(num_docs)
                ]
                chunk_cu = torch.tensor(bounds, dtype=torch.int32)
            chunks.append((chunk_x, chunk_l, chunk_cu))
        return chunks

    chunk_steps = [build_chunks(0), build_chunks(1)]

    # The reference runs single-device and full-sequence (pp=ep=cp=1); restored before DualPipeV.
    # cp_rank is forced too: forward_posemb reads it unguarded, so a nonzero cp_rank would give the
    # reference the wrong slice of the RoPE table.
    distributed.pp_size = distributed.ep_size = distributed.cp_size = 1
    distributed.cp_rank = 0
    full_modules = ModelClass(config, phase=-1)
    full_modules.to(dtype=dtype)
    full_modules.apply(fill_weights)

    if distributed.rank == 0:
        print("[INFO] Running the reference step.", flush=True)
    torch.distributed.barrier()

    loss_refs = []
    for step_index, chunks in enumerate(chunk_steps):
        if replay:
            install_router_replay(
                full_modules, micro_batch_size, step_index * num_global_chunks, chunks
            )
        loss_refs.append(reference_step(chunks, full_modules)[0])
    if replay:
        # Catch replay ignored by both paths: the unused expert must have zero gradient.
        for n, p in full_modules.named_parameters():
            if ".experts." in n:
                assert not p.grad[-1].any(), f"{n}: a layer ignored the replayed routes"
    distributed.pp_size, distributed.ep_size = pp_size, ep_size
    distributed.cp_size, distributed.cp_rank = cp_size, cp_rank

    if distributed.rank == 0:
        print("[INFO] Completed the reference step.", flush=True)
    torch.distributed.barrier()

    local_full_modules = []

    local_full_modules.append(ModelClass(config, phase=0))
    local_full_modules.append(ModelClass(config, phase=1))

    local_full_modules = nn.Sequential(*local_full_modules)
    if pp_rank == 0:
        local_full_modules[0].embed_tokens = full_modules.embed_tokens
        local_full_modules[1].norm = full_modules.norm
        local_full_modules[1].lm_head = full_modules.lm_head
    local_full_modules[0].layers = shard_layers(full_modules.layers, local_full_modules[0])
    local_full_modules[1].layers = shard_layers(full_modules.layers, local_full_modules[1])
    if ep_size > 1:
        shard_experts(local_full_modules[0], ep_rank=ep_rank, ep_size=ep_size)
        shard_experts(local_full_modules[1], ep_rank=ep_rank, ep_size=ep_size)

    local_modules = []

    local_modules.append(ModelClass(config, phase=0))
    local_modules.append(ModelClass(config, phase=1))

    local_modules = nn.Sequential(*local_modules)
    local_modules.to(dtype=dtype)
    local_modules[0].load_state_dict(local_full_modules[0].state_dict())
    local_modules[1].load_state_dict(local_full_modules[1].state_dict())
    local_modules.zero_grad()
    apply_fsdp(local_modules, dtype)
    install_wgrad_accumulation(local_modules)
    install_raw_main_grad(local_modules)

    dualpipev_model = DualPipeV(local_modules)

    def shard(x: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
        # A packed sample is sharded one document at a time; cu_seqlens stays the whole sample's.
        if cu_seqlens is None:
            return zigzag_shard(x, cp_rank, cp_size)
        return x[:, zigzag_varlen_index(cp_rank, cp_size, cu_seqlens, x.shape[1])]

    def collect(model_outputs, objective_inputs):
        # Check before step cleanup can hide retained activations.
        assert records_empty(dualpipev_model), "a forward-only step kept activations mid-step"
        (output,), (target,) = model_outputs, objective_inputs
        return None, criterion(output, target)

    for step_index, chunks in enumerate(chunk_steps):
        # Every pipeline rank builds the same micro-batches, so each derives its own P2P shapes;
        # only the first rank reads the tensors. dp picks the samples, then cp picks the zigzag
        # slice this rank takes from each one.
        local_chunks = chunks[dp_rank * num_chunks : (dp_rank + 1) * num_chunks]
        microbatches = [
            Microbatch(
                model_inputs=(shard(x, cu_seqlens),),
                cu_seqlens=cu_seqlens,
                objective_inputs=(shard(labels, cu_seqlens),),
            )
            for x, labels, cu_seqlens in local_chunks
        ]
        if replay:
            offset = step_index * num_global_chunks + dp_rank * num_chunks
            for module in local_modules:
                install_router_replay(module, micro_batch_size, offset, local_chunks)

        if distributed.rank == 0:
            print("[INFO] Running DualPipeV step %d." % step_index, flush=True)
        torch.distributed.barrier()

        # Training must safely reuse the forward-only step's records.
        with torch.no_grad():
            collected_outputs = dualpipev_model.step(microbatches, collect)
        assert records_empty(dualpipev_model), "a forward-only step kept activations"
        objective_outputs = dualpipev_model.step(microbatches, objective)
        assert records_empty(dualpipev_model), "a training step kept activations"

        if distributed.rank == 0:
            print("[INFO] Completed DualPipeV step %d." % step_index, flush=True)
        torch.distributed.barrier()

        if pp_rank == 0:
            loss = torch.stack(objective_outputs)
            torch.testing.assert_close(torch.stack(collected_outputs), loss, rtol=0, atol=5e-3)
            if cp_size > 1:
                # Every rank takes the mean over its own S/cp tokens, so the mean over the
                # whole sequence is the average of those.
                torch.distributed.all_reduce(loss, group=distributed.cp_group)
                loss = loss / cp_size
            loss_ref = loss_refs[step_index].reshape(dp_size, -1)[dp_rank]
            print(
                "[INFO] rank-%d, step %d, loss: %s, loss_ref: %s"
                % (distributed.rank, step_index, loss, loss_ref),
                flush=True,
            )
            assert torch.allclose(loss, loss_ref, rtol=1e-3, atol=1e-3)
        else:
            assert collected_outputs == objective_outputs == []

        if distributed.rank == 0:
            print("[INFO] Loss matches the reference at step %d." % step_index, flush=True)
        torch.distributed.barrier()

    eps = 1e-2
    largest_diff = 0
    largest_diff_param = None
    failed = False

    for (param_name, param), param_ref in zip(local_modules.named_parameters(), local_full_modules.parameters()):  # fmt: skip
        if param.grad is None:
            print("[ERROR] rank-%d, Parameter %s has no gradient." % (distributed.rank, param_name))
            failed = True
            continue
        # Both parameter classes reduce with a plain sum over their own replica group, and either
        # sum spans every global chunk exactly once: the attn group is the whole dp x cp stage,
        # and the expt replica group covers all the data too, since every member receives the
        # tokens of a whole EP group through the all-to-all. A mismatch here points at the mesh
        # or at the gradient divide factor.
        param_grad = param.grad
        if isinstance(param_grad, torch.distributed.tensor.DTensor):
            param_grad = param_grad.full_tensor()
        if cp_size > 1:
            # Each CP rank takes a mean over S/cp tokens, which is cp times its share of the
            # full-sequence mean, so the summed gradient overshoots the reference by exactly cp.
            param_grad = param_grad / cp_size
        if torch.all(param_grad == 0) and torch.all(param_ref.grad == 0):
            print(
                "[warn] rank-%d, Parameter %s has all-zero gradient, skipping."
                % (distributed.rank, param_name)
            )
            continue
        # The reference accumulates in bf16 and DualPipeV in fp32, so the diff of gradients at the
        # noise floor (gpt-oss router.bias, about 1e-8) is meaningless. Routed expert weights are
        # exempt: their gradients scale as 1/num_experts, so a fixed floor would skip them all on
        # a many-expert config.
        ref_max = param_ref.grad.abs().max().item()
        if ref_max < 1e-5 and ".experts." not in param_name:
            print(
                "[warn] rank-%d, Parameter %s grad max=%.2e at bf16 noise floor, skipping."
                % (distributed.rank, param_name, ref_max)
            )
            continue
        diff = calculate_difference(param_grad, param_ref.grad)
        if diff > largest_diff:
            largest_diff = diff
            largest_diff_param = param_name
        # A_log and dt_bias feed the gated-delta decay g = -exp(A_log) * softplus(a + dt_bias).
        # Their tiny gradients accumulate over the recurrence and are dominated by
        # non-deterministic bf16 scatter-add and all-to-all ordering, which varies past eps.
        param_eps = 3e-2 if param_name.endswith(("linear_attn.A_log", "linear_attn.dt_bias")) else eps  # fmt: skip
        # Under FP8 the expert wgrad quantization follows token order, which EP changes; a broken
        # accumulation shows up as a diff near 1 or above.
        if training.fp8 and ".experts." in param_name:
            param_eps = 0.1
        # Qwen3.5 under FP8: the FP8 rounding of every dense projection is amplified through the
        # gated-delta-net recurrence, on every non-expert weight rather than one broken layer.
        if training.fp8 and config.model_type == "qwen3_5_moe_text":
            param_eps = max(param_eps, 0.1)
            # The decay parameters already need 3e-2 under BF16, and FP8 noise is about 6x the
            # BF16 noise on every dense weight.
            if param_name.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
                param_eps = 0.3
        # DeepSeek-V2 under FP8: with so few rows per expert on this small fixture, a 1-ulp BF16
        # perturbation of the embedding alone moves the FP8 gradients by up to 0.11 on dense
        # weights and 0.29 on experts, also on one GPU with pp=ep=cp=1. A broken dense layer or
        # accumulation reads about 1.
        if training.fp8 and config.model_type == "deepseek_v2":
            param_eps = max(param_eps, 0.6 if ".experts." in param_name else 0.05)
        if diff > param_eps:
            failed = True
            print(
                "[ERROR] rank-%d, Parameter %s grad mismatch: diff=%.6f, eps=%.6f, p_grad:%s..., p_ref.grad:%s..."
                % (
                    distributed.rank,
                    param_name,
                    diff,
                    param_eps,
                    param_grad.flatten()[:5],
                    param_ref.grad.flatten()[:5],
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

    # Export before and after changing the weights to reject caches retained across syncs.
    training.model = dualpipev_model
    for updated in (False, True):
        if updated:
            with torch.no_grad():
                for param in dualpipev_model.parameters():
                    weight = param.to_local()
                    weight.neg_().masked_fill_(weight == 0, 1)
        reference = canonical_state_dict(local_full_modules[0], ep_rank)
        reference.update(canonical_state_dict(local_full_modules[1], ep_rank))
        for name, tensor in iter_canonical_parameters():
            weight = reference.pop(name)
            if updated:
                weight = weight.neg().masked_fill_(weight == 0, 1)
            assert torch.equal(tensor, weight), (distributed.rank, name, updated)
        assert not reference, (distributed.rank, sorted(reference)[:5])
    torch.distributed.barrier()
    if distributed.rank == 0:
        print("[INFO] All canonical parameters match the reference.", flush=True)


@record
def entry() -> None:
    models = []
    models.append("examples/pretrain_lm/deepseek-v2-lite/config.json")
    models.append("examples/pretrain_lm/qwen3-30b-a3b/config.json")
    models.append("examples/pretrain_lm/gpt-oss-20b/config.json")
    models.append("examples/pretrain_lm/gpt-oss-120b/config.json")
    models.append("examples/pretrain_lm/qwen3.5-35b-a3b/config.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--model", type=str, choices=models, required=True)
    parsed = parser.parse_args()

    cfg = SimpleNamespace()
    cfg.distributed = DistributedCfg()
    cfg.distributed.pipeline_parallel_size = parsed.pp_size
    cfg.distributed.expert_parallel_size = parsed.ep_size
    cfg.distributed.context_parallel_size = parsed.cp_size

    setup_distributed(cfg)
    main(parsed.model)


if __name__ == "__main__":
    entry()
