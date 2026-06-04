"""
CLI entry point for the PithTrain memory estimator.

Usage:
    python -m tools.memory_estimator \
        --model examples/pretrain_lm/qwen3-30b-a3b/config.json \
        --pp-size 2 --ep-size 8 --total-gpus 32 \
        --micro-batch-size 1 --global-batch-size 1024 --sequence-length 2048 \
        --gpu-memory-gb 80

Options:
    --detail          Show per-parameter and per-chunk activation breakdown at peak
    --timeline        Show full schedule timeline with memory at each event
    --timeline-limit  Max timeline events to show (0 = all, default 100)
    --pp-rank N       Show a specific PP rank (default: show worst-case)
    --ep-imbalance    EP token imbalance factor (default 1.0 = uniform)
    --fragmentation   CUDA allocator overhead factor (default 0.10)
"""

from __future__ import annotations

import argparse
import sys

from .activation_profile import ActivationEstimator, compute_token_counts
from .model_profile import ModelConfig, ModelMemoryProfile, ParallelismConfig
from .report import print_detail, print_suggestions, print_summary, print_timeline
from .schedule_simulator import ScheduleSimulator
from .tensor_spec import format_bytes


def estimate_non_pytorch_bytes(parallel_cfg: ParallelismConfig) -> int:
    """
    Estimate non-PyTorch GPU memory (CUDA context + NCCL + FSDP + runtime).

    These are allocated outside PyTorch's caching allocator and consume GPU memory
    that is unavailable for model tensors.  Estimates are calibrated against
    real H100 measurements (pp=4, ep=8, dp=1, 32 GPUs):

    Phase 1 - CUDA init:          ~0.52 GiB
    Phase 2 - NCCL world group:   ~0.39 GiB
    Phase 3 - Device mesh groups: ~1.00 GiB (pp, ep, dp process groups)
    Phase 4 - FSDP + optimizer:   ~0.96 GiB (FSDP state, reduce-scatter buffers)
    Phase 5 - First forward:      ~2.97 GiB (torch.compile codegen, cuBLAS workspace,
              flash_attn workspace, NCCL all-to-all internal buffers)
    Total measured:                ~5.84 GiB
    """
    GiB = 1024**3

    # Phase 1: CUDA context (driver, cuBLAS/cuDNN handles)
    cuda_context = int(0.52 * GiB)

    # Phase 2: NCCL world process group
    nccl_world = int(0.39 * GiB)

    # Phase 3: Device mesh process groups (pp, ep, dp dimensions)
    # Measured: 1.00 GiB for pp=4, ep=8, dp=1 (2 active mesh dims).
    # Scale by number of active mesh dimensions.
    num_mesh_dims = sum(
        [
            parallel_cfg.pp_size > 1,
            parallel_cfg.ep_size > 1,
            parallel_cfg.dp_size > 1,
        ]
    )
    nccl_mesh = int(num_mesh_dims * 0.4 * GiB)
    # PP requires additional send/recv groups
    if parallel_cfg.pp_size > 1:
        nccl_mesh += int(0.2 * GiB)

    # Phase 4: FSDP internal state and reduce-scatter buffers
    fsdp_overhead = int(0.96 * GiB)

    # Phase 5: Runtime overhead allocated on first forward pass
    # - torch.compile codegen + Triton kernel cache: ~1.5 GiB
    # - cuBLAS workspace: ~0.5 GiB
    # - Flash Attention workspace: ~0.5 GiB
    # - NCCL internal send/recv buffers (allocated on first all-to-all): ~0.5 GiB
    runtime_overhead = int(2.97 * GiB)

    return cuda_context + nccl_world + nccl_mesh + fsdp_overhead + runtime_overhead


def run_estimate(
    model_cfg: ModelConfig,
    parallel_cfg: ParallelismConfig,
    pp_rank: int,
    ep_imbalance_factor: float,
    fragmentation_factor: float,
    gpu_memory_gb: float,
    show_detail: bool,
    show_timeline: bool,
    timeline_limit: int,
):
    """Run the memory estimation for a single configuration."""
    profile = ModelMemoryProfile(model_cfg, parallel_cfg, pp_rank)
    token_counts = compute_token_counts(model_cfg, parallel_cfg, ep_imbalance_factor)
    act_est = ActivationEstimator(
        model_cfg,
        parallel_cfg,
        profile,
        token_counts,
    )

    non_pytorch = estimate_non_pytorch_bytes(parallel_cfg)

    simulator = ScheduleSimulator(
        model_cfg=model_cfg,
        parallel_cfg=parallel_cfg,
        profile=profile,
        activation_est=act_est,
        token_counts=token_counts,
        pp_rank=pp_rank,
        fragmentation_factor=fragmentation_factor,
        non_pytorch_bytes=non_pytorch,
    )

    result = simulator.simulate()

    # Build description strings
    num_layers = model_cfg.num_hidden_layers
    num_experts = model_cfg.num_experts
    topk = model_cfg.num_experts_per_tok
    model_desc = (
        f"Model: {model_cfg.model_type} | {num_layers} layers, {num_experts} experts, top-{topk}"
    )

    pp = parallel_cfg.pp_size
    ep = parallel_cfg.ep_size
    dp = parallel_cfg.dp_size
    cp = parallel_cfg.cp_size
    total_gpus = pp * ep * dp * cp
    parallel_desc = (
        f"Parallelism: pp={pp}, ep={ep}, dp={dp}, cp={cp} | {total_gpus} GPUs | pp_rank={pp_rank}"
    )

    mbs = parallel_cfg.micro_batch_size
    gbs = parallel_cfg.global_batch_size
    seq = parallel_cfg.sequence_length
    num_chunks = gbs // (mbs * dp * ep)
    training_desc = (
        f"Training: micro_bs={mbs}, global_bs={gbs}, seq_len={seq}, num_chunks={num_chunks}"
    )

    ep_desc = f"EP token estimate: imbalance_factor={ep_imbalance_factor:.2f}"
    ep_desc += f" (m_dedup={token_counts.m_dedup}, m_recv={token_counts.m_recv})"

    print_summary(result, model_desc, parallel_desc, training_desc, ep_desc, gpu_memory_gb)

    if show_detail:
        print_detail(result)

    if show_timeline:
        print_timeline(result, max_events=timeline_limit)

    print_suggestions(result, gpu_memory_gb)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PithTrain Memory Estimator - estimate peak GPU memory for DualPipeV training"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model config JSON (e.g., examples/.../config.json)",
    )
    parser.add_argument("--pp-size", type=int, required=True, help="Pipeline parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--cp-size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--total-gpus", type=int, required=True, help="Total number of GPUs")
    parser.add_argument("--micro-batch-size", type=int, required=True, help="Micro batch size")
    parser.add_argument("--global-batch-size", type=int, required=True, help="Global batch size")
    parser.add_argument("--sequence-length", type=int, required=True, help="Sequence length")
    parser.add_argument(
        "--fp8-training",
        default="disabled",
        choices=["disabled", "deep-gemm"],
        help="FP8 training mode",
    )
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=80.0,
        help="GPU memory in GB (default: 80 for H100)",
    )
    parser.add_argument(
        "--pp-rank",
        type=int,
        default=-1,
        help="Specific PP rank to analyze (-1 = find worst case)",
    )
    parser.add_argument(
        "--ep-imbalance",
        type=float,
        default=1.0,
        help="EP token imbalance factor (1.0 = uniform, 1.3 = 30%% imbalance)",
    )
    parser.add_argument(
        "--fragmentation",
        type=float,
        default=0.10,
        help="CUDA allocator fragmentation overhead (default: 0.10 = 10%%)",
    )
    parser.add_argument("--detail", action="store_true", help="Show detailed breakdown at peak")
    parser.add_argument("--timeline", action="store_true", help="Show full schedule timeline")
    parser.add_argument(
        "--timeline-limit",
        type=int,
        default=100,
        help="Max timeline events to show (0 = all)",
    )
    args = parser.parse_args()

    # Load model config
    model_cfg = ModelConfig.from_json(args.model)

    # Compute dp_size
    dp_size = args.total_gpus // (args.pp_size * args.cp_size * args.ep_size)
    if dp_size * args.pp_size * args.cp_size * args.ep_size != args.total_gpus:
        print(
            f"Error: total_gpus ({args.total_gpus}) must be divisible by "
            f"pp_size * cp_size * ep_size ({args.pp_size * args.cp_size * args.ep_size})",
            file=sys.stderr,
        )
        sys.exit(1)

    parallel_cfg = ParallelismConfig(
        pp_size=args.pp_size,
        ep_size=args.ep_size,
        dp_size=dp_size,
        cp_size=args.cp_size,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        sequence_length=args.sequence_length,
        fp8_training=args.fp8_training,
    )

    # Validate num_chunks
    chunk_divisor = args.micro_batch_size * dp_size * args.ep_size
    if args.global_batch_size % chunk_divisor != 0:
        print(
            f"Error: global_batch_size ({args.global_batch_size}) must be divisible by "
            f"micro_batch_size * dp_size * ep_size ({chunk_divisor})",
            file=sys.stderr,
        )
        sys.exit(1)
    num_chunks = args.global_batch_size // chunk_divisor
    if num_chunks < args.pp_size * 2:
        print(
            f"Error: num_chunks ({num_chunks}) must be >= pp_size * 2 ({args.pp_size * 2}). "
            f"Increase global_batch_size or decrease micro_batch_size.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.pp_rank >= 0:
        pp_ranks = [args.pp_rank]
    else:
        # Find worst case across all PP ranks
        pp_ranks = list(range(args.pp_size))

    worst_rank = 0
    worst_bytes = 0

    for rank in pp_ranks:
        result = run_estimate(
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            pp_rank=rank,
            ep_imbalance_factor=args.ep_imbalance,
            fragmentation_factor=args.fragmentation,
            gpu_memory_gb=args.gpu_memory_gb,
            show_detail=args.detail and (args.pp_rank >= 0 or len(pp_ranks) == 1),
            show_timeline=args.timeline and (args.pp_rank >= 0 or len(pp_ranks) == 1),
            timeline_limit=args.timeline_limit,
        )
        if result.peak_bytes > worst_bytes:
            worst_bytes = result.peak_bytes
            worst_rank = rank

    # If scanning all ranks, show detail for worst rank
    if args.pp_rank < 0 and len(pp_ranks) > 1:
        print()
        print(f"{'=' * 70}")
        print(f"  Worst-case PP rank: {worst_rank} (peak = {format_bytes(worst_bytes)})")
        print(f"  Showing detailed breakdown for pp_rank={worst_rank}:")
        print(f"{'=' * 70}")
        run_estimate(
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            pp_rank=worst_rank,
            ep_imbalance_factor=args.ep_imbalance,
            fragmentation_factor=args.fragmentation,
            gpu_memory_gb=args.gpu_memory_gb,
            show_detail=args.detail,
            show_timeline=args.timeline,
            timeline_limit=args.timeline_limit,
        )


if __name__ == "__main__":
    main()
