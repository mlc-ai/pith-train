"""
The fixture every end-to-end benchmark case shares. A case is a directory

    benchmarks/end2end/<model>-seq<N>-<dtype>/h200-<nodes>n<gpus>g/pp<P>-dp<D>-cp<C>-ep<E>/script.py

whose script calls end2end_cfg and sets only its mesh; dp is the attention DP, world / (pp * cp).
Everything that makes runs comparable lives here, so a later run differs from the baseline only
in the code under test.
"""

from pathlib import Path

__all__ = ["MODELS", "MAX_STEPS", "WARMUP_STEPS", "TOKENS_PER_GPU", "end2end_cfg"]

MODELS = {
    # name: (config directory, moe_load_balance_type, moe_load_balance_coef)
    "deepseek-v2-lite": (Path("examples/pretrain_lm/deepseek-v2-lite"), "sequence", 3e-3),
    "gpt-oss-20b": (Path("examples/pretrain_lm/gpt-oss-20b"), "global-batch", 1e-3),
    "gpt-oss-120b": (Path("examples/pretrain_lm/gpt-oss-120b"), "global-batch", 1e-3),
    "qwen3-30b-a3b": (Path("examples/pretrain_lm/qwen3-30b-a3b"), "global-batch", 1e-3),
    "qwen3.5-35b-a3b": (Path("examples/pretrain_lm/qwen3.5-35b-a3b"), "global-batch", 1e-3),
    "qwen3.5-122b-a10b": (
        Path("benchmarks/end2end/models/qwen3.5-122b-a10b"),
        "global-batch",
        1e-3,
    ),
}

MAX_STEPS = 12
"""
Steps per case: WARMUP_STEPS, then 6 timed steps whose median is the step time. Step 1 carries
the torch.compile trace and the JIT warmup, and kernel tuning runs between steps 1 and 2.
"""

WARMUP_STEPS = 6

TOKENS_PER_GPU = 8 * 32768
"""
Tokens per GPU per step: 64 sequences of 32K per 8-GPU node, or 512 of 4K. Each attention DP rank
then runs (TOKENS_PER_GPU / sequence_length) * pp * cp / micro_batch_size micro-batches a step, at
least 8 at every case (16 at 32K, cp >= 2; 8 at 4K, micro-batch 8), so DualPipeV has micro-batches
to overlap at pp1 as at pp2.
"""


def end2end_cfg(
    model: str, sequence_length: int, fp8: bool, world_size: int, micro_batch_size: int = 1
):
    """
    The PretrainLMCfg of one case before its mesh is set.

    training.benchmark selects SyntheticDataset, seeded ids drawn on the GPU, so no corpus is
    prepared and no step reads a file, and replays the routing with the expert-load imbalance of a
    real training run (pithtrain.modules.imbalance, SIGMA = 0.2), seeded by layer and step so every
    run of a case routes alike. Nothing is checkpointed and nothing goes to wandb, so no step touches
    the disk.

    The global batch is TOKENS_PER_GPU tokens per GPU, so tokens per GPU per step are the same at
    every mesh, node count and sequence length, and every mesh of one node count and sequence length
    trains on the same samples. It satisfies the DualPipeV bound global_batch / dp >= 2 * pp at every
    case, as global_batch / dp = (TOKENS_PER_GPU / sequence_length) * pp * cp.
    """
    # Imported here, not at the top: importing pithtrain reads the GPU, and tools that read this
    # module's tables may run on machines without one.
    from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg

    path, lb_type, lb_coef = MODELS[model]
    cfg = PretrainLMCfg()
    training = cfg.training
    training.model = path
    training.optimizer = make_muon_optimizer
    training.scheduler = make_constant_scheduler
    training.lr = 3.0e-4
    training.max_steps = MAX_STEPS
    training.micro_batch_size = micro_batch_size
    training.global_batch_size = TOKENS_PER_GPU // sequence_length * world_size
    training.sequence_length = sequence_length
    training.moe_load_balance_type = lb_type
    training.moe_load_balance_coef = lb_coef
    training.benchmark = True
    training.fp8 = fp8
    return cfg
