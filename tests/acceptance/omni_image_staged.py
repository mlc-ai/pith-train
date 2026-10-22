"""H100-only DualPipeV/FSDP image gradient gate; invoked by omni_image_training."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, distribute_tensor

from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.training_data import global_loss_mean, global_target_count
from tests.acceptance.omni_image_evidence import (
    OptimizerStepEvidence,
    fixed_image_batches,
    module_update_evidence,
    target_partition_evidence,
    training_groups,
    validate_target_layout,
)
from tests.acceptance.omni_image_training import (
    HFReference,
    commit,
    error_ratio,
    hf_name,
    source_inputs,
)
from tests.acceptance.omni_optimizer_probe import isolated_optimizer_probe
from tests.acceptance.omni_reference_attention import (
    configure_reference_attention,
    reference_contract,
)
from tests.acceptance.omni_reference_precision import FP32GradientReference


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def expected_shard(value, parameter, name, distributed):
    if ".experts." in name:
        count = value.shape[0] // distributed.ep_size
        value = value.narrow(0, distributed.ep_rank * count, count)
    if isinstance(parameter, DTensor):
        value = distribute_tensor(value, parameter.device_mesh, parameter.placements)
    return value


def run(
    args,
    *,
    source_inputs_fn=source_inputs,
    reference_class=HFReference,
    modality="image",
    pixel_key="pixel_values",
    training_flag="native_vision",
    sequence_length=512,
    hf_name_fn=hf_name,
    training_groups_fn=training_groups,
):
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import (
        make_adamw_optimizer,
        make_constant_scheduler,
        setup_training,
    )
    from pithtrain.tasks import pretrain_lm

    world = int(os.environ["WORLD_SIZE"])
    assert world % args.pp == 0 and (world // args.pp) % args.ep == 0
    assert args.scenario != "rank-zero" or world // args.pp > 1
    _, _, batches, config = source_inputs_fn(args.dataset)
    rank = int(os.environ["RANK"])
    # Before the first barrier every rank checks that evidence is fresh.
    assert not args.output.exists(), "Use a fresh shared output"
    cfg = pretrain_lm.PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size, cfg.distributed.expert_parallel_size = args.pp, args.ep
    cfg.distributed.timeout = timedelta(seconds=90)
    t = cfg.training
    t.model = args.output / "model"
    assert (modality, pixel_key, training_flag) in {
        ("image", "pixel_values", "native_vision"),
        ("video", "pixel_values_videos", "native_video"),
        ("audio", "input_features", "native_audio"),
    }
    setattr(t, training_flag, True)
    t.fp8, t.moe_load_balance_coef = False, 0.0
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.lr, t.max_steps, t.sequence_length, t.global_batch_size, t.micro_batch_size = (
        1e-4,
        4,
        sequence_length,
        8,
        1,
    )
    t.save_interval = None
    cfg.data.dataset, cfg.data.format, cfg.data.modalities = (
        args.dataset,
        "prepared_bundle",
        (modality,),
    )
    cfg.data.num_workers = 0
    setup_logging(cfg)
    setup_distributed(cfg)
    if rank == 0:
        args.output.mkdir(parents=True)
        config.save_pretrained(t.model)
    dist.barrier()
    data = pretrain_lm.setup_dataset(cfg)
    setup_training(cfg)
    hf_reference = reference_class(config, torch.bfloat16).to(distributed.device)
    attention = configure_reference_attention(hf_reference, config, reference_contract(args))
    (args.output / f"reference-attention-rank-{rank:05d}.json").write_text(
        json.dumps(attention, indent=2) + "\n"
    )
    reference = FP32GradientReference(hf_reference)
    expected = reference.master_parameters
    actual = {}
    for module in training.model.module:
        for name, parameter in module.named_parameters():
            key = hf_name_fn(name)
            assert key not in actual and key in expected, key
            actual[key] = parameter
            with torch.no_grad():
                parameter.copy_(expected_shard(expected[key].detach(), parameter, key, distributed))
    owned_keys = [None] * world
    dist.all_gather_object(owned_keys, sorted(actual))
    assert set().union(*(set(keys) for keys in owned_keys)) == set(expected)

    # Each scenario has its own process/weights/optimizer and output directory.
    # Preserve nonuniform microbatches; rank-zero also masks every microbatch on
    # attention-DP rank 0, while retaining the other rank's labels.
    global_batches = fixed_image_batches(batches, distributed.dp_size, args.scenario)
    layout = validate_target_layout(global_batches, distributed.dp_size, args.scenario)
    local_batches = global_batches[distributed.dp_rank :: distributed.dp_size]
    microbatches = []
    for batch in local_batches:
        inputs = {k: v.to(distributed.device) for k, v in batch.model_inputs.items()}
        pixels = inputs.pop(pixel_key)
        microbatches.append(
            Microbatch(
                model_inputs=(inputs["input_ids"],),
                cu_seqlens=None,
                objective_inputs=(batch.labels.to(distributed.device),),
                model_context=inputs,
                media_inputs={pixel_key: pixels} if distributed.pp_rank == 0 else None,
                sample_ids=batch.sample_ids,
            )
        )
    group = distributed.attn_mesh["dp", "cp"]._flatten().get_group()
    count = global_target_count(microbatches, group)
    partition = target_partition_evidence(microbatches, layout, distributed.dp_rank, count)
    partition["pp_rank"] = distributed.pp_rank
    partitions = [None] * world
    dist.all_gather_object(partitions, partition)
    assert sorted((row["pp_rank"], row["dp_rank"]) for row in partitions) == [
        (pp, dp) for pp in range(args.pp) for dp in range(distributed.dp_size)
    ]
    for row in partitions:
        assert row["local_target_count"] == layout["rank_target_totals"][row["dp_rank"]]
        assert row["global_target_count"] == layout["global_target_count"] > 0
        if args.scenario == "rank-zero" and row["dp_rank"] == 0:
            assert row["local_target_count"] == 0
    # Independent HF processes all eight global samples. No PP or DP replicas
    # are counted in its denominator.
    hf_loss = 0
    for batch in global_batches:
        inputs = {k: v.to(distributed.device) for k, v in batch.model_inputs.items()}
        logits = reference(inputs)
        labels = batch.labels.to(distributed.device)
        loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
        loss.backward()
        hf_loss += loss.detach()
    # global_target_count returns FP32 for native loss/gradient division. The
    # independently counted integer layout was checked against every rank above.
    reference.normalize_gradients_(layout["global_target_count"])
    hf_loss = hf_loss / count
    outputs = training.model.step(microbatches, pretrain_lm.objective)
    if distributed.pp_rank == 0:
        loss = global_loss_mean(torch.stack(outputs).sum(), count, group)
        torch.testing.assert_close(loss, hf_loss, rtol=1e-3, atol=1e-3)
    errors = {}
    for key, parameter in actual.items():
        assert parameter.grad is not None and expected[key].grad is not None, key
        parameter.grad.div_(count)
        target = expected_shard(expected[key].grad, parameter, key, distributed)
        errors[key] = error_ratio(local_tensor(parameter.grad), local_tensor(target))
        assert errors[key] < 1e-2, (key, errors[key])
    before = {n: local_tensor(p).detach().clone() for n, p in actual.items()}
    with isolated_optimizer_probe(training.model, lambda: make_adamw_optimizer(t)) as probes:
        for optimizer in probes:
            optimizer.step()
        updated = [n for n, p in actual.items() if not torch.equal(before[n], local_tensor(p))]
        groups = training_groups_fn(config)
        modules = module_update_evidence(actual, before, groups, reduce=dist.all_reduce)
    del probes, optimizer
    for optimizer in training.optimizers:
        optimizer.zero_grad(set_to_none=True)
    zero = [
        Microbatch(
            model_inputs=mb.model_inputs,
            cu_seqlens=None,
            objective_inputs=(torch.full_like(mb.objective_inputs[0], -100),),
        )
        for mb in microbatches
    ]
    try:
        global_target_count(zero, group)
    except ValueError as error:
        zero_error = str(error)
    else:
        raise AssertionError("All-zero targets accepted")
    all_errors = [None] * world
    dist.all_gather_object(all_errors, zero_error)
    assert len(set(all_errors)) == 1
    # Observe every real task optimizer step, after backward/gradient scaling,
    # without replacing train_step, model execution or optimizer behavior.
    assert len(training.optimizers) == 1
    observer = OptimizerStepEvidence(actual, groups, reduce=dist.all_reduce)
    optimizer = training.optimizers[0]
    pre_hook = optimizer.register_step_pre_hook(observer.before_step)
    post_hook = optimizer.register_step_post_hook(observer.after_step)
    try:
        for step in range(4):
            pretrain_lm.train_step(cfg, data, step)
    finally:
        pre_hook.remove()
        post_hook.remove()
    assert len(observer.steps) == 4
    assert data.consumed_samples == 32
    report = dict(
        status="PASSED",
        modality=modality,
        commit=commit(),
        rank=rank,
        pp=args.pp,
        ep=args.ep,
        dp=distributed.dp_size,
        cp=1,
        scenario=args.scenario,
        target_layout=layout,
        target_partitions=partitions,
        local_target_count=partition["local_target_count"],
        target_count=int(count),
        gradient_errors=errors,
        reference_precision=reference.precision_contract(),
        reference_attention=attention,
        updated_parameters=updated,
        training_modules=modules,
        real_reader_module_updates=observer.steps,
        real_reader_steps=4,
        consumed_samples=32,
        zero_target_rejection=zero_error,
        max_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.output / f"rank-{rank:05d}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
