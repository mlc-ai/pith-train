"""Check the real shuffled pretraining batch path under torchrun, without building a model.

Run after preparing examples/prepare_omni_data/qwen3-omni-training:
    torchrun --standalone --nproc-per-node=1 tests/test_pretrain_data.py
    torchrun --standalone --nproc-per-node=2 tests/test_pretrain_data.py --cp 2
"""

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch


def read_text_reference(root, sequence_length):
    """Read the prepared train-only stream independently of the production loader."""
    from pithtrain.tasks.prepare_omni_data import processor_for, verify_bundle

    root = Path(root).resolve()
    bundle = verify_bundle(root)
    processor, config = processor_for(bundle["recipe"], True)
    checksums = dict(
        (name, digest)
        for digest, name in (
            line.split("  ", 1) for line in (root / "checksums.sha256").read_text().splitlines()
        )
    )
    paths = sorted((root / "tokens/train").rglob("*.bin"))
    listed = sorted(
        name for name in checksums if name.startswith("tokens/train/") and name.endswith(".bin")
    )
    assert paths and [str(path.relative_to(root)) for path in paths] == listed
    expected_inputs, expected_labels, documents = [], [], 0
    for path in paths:
        with path.open("rb") as stream:
            tokens, ends = np.load(stream), np.load(stream)
        assert tokens.dtype == np.uint32
        assert len(ends) and ends[-1] == len(tokens) and np.all(np.diff(ends.astype(np.int64)) > 0)
        assert np.all(tokens[ends.astype(np.int64) - 1] == processor.tokenizer.eos_token_id)
        assert tokens.max() < config.text_config.vocab_size
        documents += len(ends)
        count = (len(tokens) - 1) // sequence_length * sequence_length
        expected_inputs.append(tokens[:count].astype(np.int64).reshape(-1, sequence_length))
        expected_labels.append(tokens[1 : count + 1].astype(np.int64).reshape(-1, sequence_length))
    assert documents == bundle["statistics"]["train"]["text"]["samples"]
    return (
        torch.from_numpy(np.concatenate(expected_inputs)),
        torch.from_numpy(np.concatenate(expected_labels)),
        dict(vocab_size=config.text_config.vocab_size, seed=bundle["recipe"]["seed"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("workspace/datasets/omni-training"))
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if min(args.sequence_length, args.global_batch_size, args.micro_batch_size, args.steps) <= 0:
        parser.error("Sequence/batch sizes and step count must be positive")
    inputs, labels, metadata = read_text_reference(args.dataset, args.sequence_length)
    # Keep pytest collection CPU-safe; this integration check requires a CUDA runtime.
    from pithtrain.contexts import distributed
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.operators.cp_sequence import zigzag_spans
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, get_global_batch, setup_dataset

    cfg = PretrainLMCfg()
    cfg.dataset = args.dataset / "tokens/train"
    cfg.training.sequence_length = args.sequence_length
    cfg.training.global_batch_size = args.global_batch_size
    cfg.training.micro_batch_size = args.micro_batch_size
    cfg.training.max_steps = args.steps
    cfg.training.seed = metadata["seed"] if args.seed is None else args.seed
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = args.cp
    cfg.distributed.expert_parallel_size = args.ep
    cfg.distributed.timeout = timedelta(seconds=60)
    setup_distributed(cfg)
    t, d = cfg.training, distributed
    assert t.global_batch_size % (t.micro_batch_size * d.dp_size) == 0
    assert t.sequence_length % (2 * d.cp_size) == 0

    dataset = setup_dataset(cfg)
    indices = torch.from_numpy(np.array(dataset.indices, dtype=np.int64))
    assert len(dataset) == len(inputs)
    assert torch.equal(indices.sort().values, torch.arange(len(dataset))), "Invalid shuffle"
    digests = [None] * d.world_size
    torch.distributed.all_gather_object(
        digests, hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    )
    assert len(set(digests)) == 1, "Ranks disagree on the shuffled sample order"
    # Reinitializing with the same seed must give the same global sample order.
    torch.distributed.barrier()
    dataset = setup_dataset(cfg)
    assert np.array_equal(dataset.indices, indices.numpy()), "Shuffle is not reproducible"
    front, back = zigzag_spans(d.cp_rank, d.cp_size, t.sequence_length)
    positions = torch.tensor([*front, *back])
    for step in range(t.max_steps):
        batch = get_global_batch(cfg, dataset, step, d.device)
        assert len(batch) == t.global_batch_size // d.dp_size // t.micro_batch_size
        # Global micro-batches alternate between data ranks. EP/PP do not select samples.
        selected = indices[step * t.global_batch_size : (step + 1) * t.global_batch_size]
        selected = selected.reshape(-1, d.dp_size, t.micro_batch_size)[:, d.dp_rank].flatten()
        expected_x = inputs[selected][:, positions]
        expected_y = labels[selected][:, positions]
        for micro in batch:
            (x,), (y,) = micro.model_inputs, micro.objective_inputs
            assert x.shape == y.shape == (t.micro_batch_size, t.sequence_length // d.cp_size)
            assert x.dtype == y.dtype == torch.long and x.device == y.device == d.device
            assert micro.cu_seqlens is None
        actual_x = torch.cat([micro.model_inputs[0] for micro in batch]).cpu()
        actual_y = torch.cat([micro.objective_inputs[0] for micro in batch]).cpu()
        torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(actual_y, expected_y, rtol=0, atol=0)
    torch.distributed.barrier()
    if d.rank == 0:
        print(
            json.dumps(
                dict(
                    result="PASSED",
                    pp=d.pp_size,
                    dp=d.dp_size,
                    cp=d.cp_size,
                    ep=d.ep_size,
                    steps=t.max_steps,
                    samples_per_step=t.global_batch_size,
                    sequence_length=t.sequence_length,
                    dataset_samples=len(dataset),
                    vocab_size=metadata["vocab_size"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
