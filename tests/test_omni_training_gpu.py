"""GPU integration: real pretrain steps, data checkpoints and pipeline context transport.

Uses a reduced existing Qwen3 model with the real Omni vocabulary. This tests the
training connection, not native Omni encoders or their numerical correctness.
Run with torchrun; --context also exercises a test-only model_context consumer.
"""

import argparse
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()

    from pithtrain.contexts import distributed, training
    from pithtrain.models.qwen3_moe import Qwen3MoeModel
    from pithtrain.modules.checkpoint import find_checkpoint, load_checkpoint
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import (
        make_adamw_optimizer,
        make_constant_scheduler,
        setup_training,
    )
    from pithtrain.modules.training_data import OmniDataCfg
    from pithtrain.pipeline.execution import model_forward
    from pithtrain.tasks import pretrain_lm

    cfg = pretrain_lm.PretrainLMCfg()
    cfg.dataset = args.dataset
    cfg.omni_data = OmniDataCfg()
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = args.cp
    cfg.distributed.expert_parallel_size = args.ep
    cfg.distributed.timeout = timedelta(minutes=5)
    t = cfg.training
    t.model = args.output / "model"
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.lr, t.max_steps, t.sequence_length = 1e-4, 2, 128
    t.global_batch_size, t.micro_batch_size = 8, 1
    t.fp8, t.moe_load_balance_coef = False, 0.0
    t.save_interval, t.save_location = 1, args.output / "checkpoints"
    setup_logging(cfg)
    setup_distributed(cfg)
    if distributed.rank == 0:
        base = Path(__file__).parents[1] / "examples/pretrain_lm/qwen3-30b-a3b/config.json"
        config = json.loads(base.read_text())
        config.update(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=max(4, 2 * args.pp),
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            vocab_size=152064,
        )
        t.model.mkdir(parents=True, exist_ok=True)
        (t.model / "config.json").write_text(json.dumps(config))
        assert find_checkpoint(t.save_location) is None, "Use a fresh output directory"
    torch.distributed.barrier()
    data = pretrain_lm.setup_dataset(cfg)
    setup_training(cfg)

    def weights():
        return {
            name: p.detach().to_local().cpu().clone()
            for name, p in training.model.named_parameters()
        }

    def check_weights(expected):
        for name, value in weights().items():
            assert torch.isfinite(value).all(), name
            torch.testing.assert_close(value, expected[name], rtol=1e-4, atol=1e-6, msg=name)

    # Instrument the existing model only inside this test. Each normal/overlapped
    # posemb call must receive the context for that very microbatch; this consumer
    # deliberately has no audio/vision behavior and declares no media capability.
    seen, normal_calls, batches_seen = Counter(), Counter(), []
    original_prolog, original_posemb = Qwen3MoeModel.forward_prolog, Qwen3MoeModel.forward_posemb
    original_batch = pretrain_lm.get_global_batch

    def get_batch(*a, **kw):
        batches = original_batch(*a, **kw)
        batches_seen.append([mb.model_inputs[0].cpu().clone() for mb in batches])
        if args.context:
            for i, mb in enumerate(batches):
                mb.model_context = {
                    "input_ids": mb.model_inputs[0],
                    "video_second_per_grid": torch.tensor(
                        [2.0 / 3.0], device=distributed.device, dtype=torch.float32
                    ),
                    "batch_index": torch.tensor(i, device=distributed.device),
                }
        return batches

    def forward(self, inputs, cu_seqlens=None, model_context=None):
        assert model_context is not None
        normal_calls[self.stage_index] += 1
        return model_forward(self, inputs, self.chunk_record, cu_seqlens, model_context)

    def prolog(self, inputs, model_context=None):
        torch.testing.assert_close(inputs, model_context["input_ids"], rtol=0, atol=0)
        return original_prolog(self, inputs)

    def posemb(self, length, cu_seqlens=None, model_context=None):
        assert model_context is not None and model_context["input_ids"].shape[1] == length
        timing = model_context["video_second_per_grid"]
        assert timing.dtype == torch.float32, "FSDP must not round timing metadata to BF16"
        torch.testing.assert_close(
            timing, torch.tensor([2.0 / 3.0], device=timing.device), rtol=0, atol=0
        )
        seen[self.stage_index, int(model_context["batch_index"])] += 1
        return original_posemb(self, length, cu_seqlens)

    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(patch.object(pretrain_lm, "get_global_batch", get_batch))
        if args.context:
            for name, method in [
                ("forward", forward),
                ("forward_prolog", prolog),
                ("forward_posemb", posemb),
            ]:
                stack.enter_context(patch.object(Qwen3MoeModel, name, method))
        initial = weights()
        pretrain_lm.train_step(cfg, data, 0)
        saved = weights()
        assert any(not torch.equal(value, initial[name]) for name, value in saved.items())
        del initial
        assert data.consumed_samples == t.global_batch_size
        pretrain_lm.train_step(cfg, data, 1)
        expected = weights()
        restored = pretrain_lm.setup_dataset(cfg)
        load_checkpoint(t.save_location, 1, data_state=restored)
        check_weights(saved)
        del saved
        assert restored.consumed_samples == t.global_batch_size
        t.save_interval = None
        pretrain_lm.train_step(cfg, restored, 1)
        check_weights(expected)
        assert restored.consumed_samples == 2 * t.global_batch_size
    for left, right in zip(batches_seen[1], batches_seen[2], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    if args.context:
        chunks = t.global_batch_size // distributed.dp_size
        for module in training.model.module:
            stage = module.stage_index
            assert all(seen[stage, i] == 3 for i in range(chunks)), (stage, seen)
            assert normal_calls[stage] > 0
        if args.pp > 1:
            assert sum(seen.values()) > sum(normal_calls.values()), "Overlap path was not exercised"
    torch.distributed.barrier()
    if distributed.rank == 0:
        print(
            json.dumps(
                dict(
                    result="PASSED",
                    pp=args.pp,
                    cp=args.cp,
                    ep=args.ep,
                    context=args.context,
                    optimizer_steps=3,
                    consumed_samples=restored.consumed_samples,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
