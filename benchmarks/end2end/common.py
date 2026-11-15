"""
The fixture every end-to-end benchmark case shares. A case is a directory

    benchmarks/end2end/<model>-seq<N>-<dtype>/h200-<nodes>n<gpus>g/pp<P>-dp<D>-cp<C>-ep<E>/script.py

whose script calls end2end_cfg and sets only its mesh; dp is the attention DP, world / (pp * cp).
Everything that makes runs comparable lives here, so a later run differs from the baseline only
in the code under test.
"""

from pathlib import Path

__all__ = ["MODELS", "MAX_STEPS", "WARMUP_STEPS", "end2end_cfg"]

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


def end2end_cfg(model: str, sequence_length: int, fp8: bool, world_size: int):
    """
    The PretrainLMCfg of one case before its mesh is set.

    Data is SyntheticDataset (cfg.dataset is None): seeded ids drawn on the GPU, so no corpus is
    prepared and no step reads a file. Routing is force-balanced (training.benchmark), which
    removes expert imbalance, the largest source of step-time variance, and makes step time
    comparable across meshes and builds. Nothing is checkpointed and nothing goes to wandb, so no
    step touches the disk.

    The global batch is 4 samples per GPU, so tokens per GPU per step are the same at every mesh
    and node count, and every mesh of one node count trains on the same samples. It satisfies the
    DualPipeV bound global_batch / dp >= 2 * pp at every mesh, as global_batch / dp = 4 * pp * cp.
    """
    # Imported here, not at the top: importing pithtrain reads the GPU, and summarize.py, compare.py
    # and collect.py read this module's tables on machines without one.
    from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg

    path, lb_type, lb_coef = MODELS[model]
    cfg = PretrainLMCfg()
    cfg.dataset = None
    training = cfg.training
    training.model = path
    training.optimizer = make_muon_optimizer
    training.scheduler = make_constant_scheduler
    training.lr = 3.0e-4
    training.max_steps = MAX_STEPS
    training.micro_batch_size = 1
    training.global_batch_size = 4 * world_size
    training.sequence_length = sequence_length
    training.moe_load_balance_type = lb_type
    training.moe_load_balance_coef = lb_coef
    training.benchmark = True
    training.fp8 = fp8
    return cfg
