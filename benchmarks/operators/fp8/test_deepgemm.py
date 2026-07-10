"""
Benchmark DeepGEMM FP8 linear layers vs BF16 baselines.

Run all benchmarks:
    python3 -m benchmarks.operators.fp8.test_deepgemm

Section 1 - Non-grouped: FP8Linear vs Linear
Section 2 - Grouped: FP8GroupedLinear vs GroupedLinear

Forward FLOPS:   2*M*N*K
Backward FLOPS:  4*M*N*K  (dgrad: 2MNK, wgrad: 2MNK)

Forward I/O  (BF16, 2B/elem): 2*(M*K + N*K + M*N)
Backward I/O (BF16, 2B/elem): 2*(2*M*N + 2*M*K + 2*N*K)
"""

import itertools
import math
from typing import List, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
from triton.testing import do_bench

from benchmarks.operators.utilities import Metrics
from pithtrain.operators.grouped_linear import FP8GroupedLinear, GroupedLinear
from pithtrain.operators.linear import ARCH_MAJOR, FP8Linear
from pithtrain.operators.token_scatter import scatter_for_grouped_gemm

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling

    HAS_TE = True
except ImportError:
    HAS_TE = False

if HAS_TE:
    _TE_FP8_RECIPE = MXFP8BlockScaling()

# ---------------------------------------------------------------------------
# Non-grouped workloads
# ---------------------------------------------------------------------------
# M values x (N, K) pairs from DeepGEMM generators.py enumerate_normal
M_VALUES = [128, 2048, 4096, 8192]
NK_PAIRS = [
    (2112, 7168),
    (576, 7168),
    (24576, 1536),
    (32768, 512),
    (7168, 16384),
    (4096, 7168),
    (7168, 2048),
]

NormalWorkload = Tuple[int, int, int]  # (M, N, K)
NORMAL_WORKLOADS: List[NormalWorkload] = [
    (m, n, k) for m, (n, k) in itertools.product(M_VALUES, NK_PAIRS)
]

# ---------------------------------------------------------------------------
# Grouped workloads
# ---------------------------------------------------------------------------
GROUP_SHAPES = [(4, 8192), (8, 4096), (16, 2048), (32, 1024)]  # (num_groups, m_per_group)
GROUP_NK_PAIRS = [
    (6144, 7168),
    (7168, 3072),
    (4096, 4096),
    (4096, 2048),
    (4096, 7168),
    (7168, 2048),
]

GroupWorkload = Tuple[int, int, int, int]  # (num_groups, m_per_group, N, K)
GROUP_WORKLOADS: List[GroupWorkload] = [
    (g, mpg, n, k) for (g, mpg), (n, k) in itertools.product(GROUP_SHAPES, GROUP_NK_PAIRS)
]


# ---------------------------------------------------------------------------
# Non-grouped benchmarks
# ---------------------------------------------------------------------------
def bench_normal_fwd(M: int, N: int, K: int) -> Metrics:
    torch.manual_seed(42)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

    ref = nn.Linear(K, N, bias=False).cuda().bfloat16()
    fp8 = FP8Linear(K, N, bias=False).cuda().bfloat16()
    fp8.weight.data.copy_(ref.weight.data)

    def ref_fwd():
        return ref(x)

    def fp8_fwd():
        return fp8(x)

    with nvtx.range(f"test run normal fwd bf16 M={M} N={N} K={K}"):
        ref_fwd()
    with nvtx.range(f"test run normal fwd dg fp8 M={M} N={N} K={K}"):
        fp8_fwd()

    with nvtx.range(f"normal fwd bf16 M={M} N={N} K={K}"):
        ref_time = do_bench(ref_fwd)
    with nvtx.range(f"normal fwd dg fp8 M={M} N={N} K={K}"):
        fp8_time = do_bench(fp8_fwd)

    te_time = float("nan")
    if HAS_TE:
        te_layer = te.Linear(K, N, bias=False).cuda().bfloat16()

        def te_fwd():
            with te.fp8_autocast(enabled=True, fp8_recipe=_TE_FP8_RECIPE):
                return te_layer(x)

        with nvtx.range(f"test run normal fwd te M={M} N={N} K={K}"):
            te_fwd()
        with nvtx.range(f"normal fwd te M={M} N={N} K={K}"):
            te_time = do_bench(te_fwd)

    flops = 2 * M * N * K
    io_bytes = 2 * (M * K + N * K + M * N)
    return Metrics(ref_time, fp8_time, flops, io_bytes, te_ms=te_time)


def bench_normal_bwd(M: int, N: int, K: int) -> Metrics:
    torch.manual_seed(42)

    ref_x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    fp8_x = ref_x.clone().detach().requires_grad_(True)
    grad = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    ref = nn.Linear(K, N, bias=False).cuda().bfloat16()
    fp8 = FP8Linear(K, N, bias=False).cuda().bfloat16()
    fp8.weight.data.copy_(ref.weight.data)

    ref_o = ref(ref_x)
    fp8_o = fp8(fp8_x)

    def ref_bwd():
        ref_x.grad = None
        ref.weight.grad = None
        ref_o.backward(grad, retain_graph=True)

    def fp8_bwd():
        fp8_x.grad = None
        fp8.weight.grad = None
        fp8_o.backward(grad, retain_graph=True)

    with nvtx.range(f"test run normal bwd bf16 M={M} N={N} K={K}"):
        ref_bwd()
    with nvtx.range(f"test run normal bwd dg fp8 M={M} N={N} K={K}"):
        fp8_bwd()

    with nvtx.range(f"normal bwd bf16 M={M} N={N} K={K}"):
        ref_time = do_bench(ref_bwd)
    with nvtx.range(f"normal bwd dg fp8 M={M} N={N} K={K}"):
        fp8_time = do_bench(fp8_bwd)

    te_time = float("nan")
    if HAS_TE:
        te_layer = te.Linear(K, N, bias=False).cuda().bfloat16()
        te_x = ref_x.clone().detach().requires_grad_(True)

        def te_fwd_bwd():
            te_x.grad = None
            te_layer.weight.grad = None
            with te.fp8_autocast(enabled=True, fp8_recipe=_TE_FP8_RECIPE):
                te_o = te_layer(te_x)
            te_o.backward(grad)

        def te_fwd_only():
            with te.fp8_autocast(enabled=True, fp8_recipe=_TE_FP8_RECIPE):
                return te_layer(te_x)

        with nvtx.range(f"test run normal bwd te M={M} N={N} K={K}"):
            te_fwd_bwd()
        with nvtx.range(f"normal fwd+bwd te M={M} N={N} K={K}"):
            te_fwd_bwd_time = do_bench(te_fwd_bwd)
        with nvtx.range(f"normal fwd te (for subtraction) M={M} N={N} K={K}"):
            te_fwd_time = do_bench(te_fwd_only)
        te_time = te_fwd_bwd_time - te_fwd_time

    flops = 4 * M * N * K
    io_bytes = 2 * (2 * M * N + 2 * M * K + 2 * N * K)
    return Metrics(ref_time, fp8_time, flops, io_bytes, te_ms=te_time)


# ---------------------------------------------------------------------------
# Grouped benchmarks
# ---------------------------------------------------------------------------
def _make_group_indices(grouped_mm_offs: torch.Tensor, M: int):
    """Build per-row group indices for Hopper; returns None on Blackwell."""
    if ARCH_MAJOR >= 10:
        return None
    row_indices = torch.arange(M, device=grouped_mm_offs.device)
    return torch.searchsorted(grouped_mm_offs, row_indices, right=True).to(torch.int32)


def _make_grouped_inputs(
    num_groups: int, m_per_group: int, K: int
) -> Tuple[torch.Tensor, torch.Tensor, list, torch.Tensor, int, torch.Tensor | None]:
    """Create scattered input and grouped_mm_offs (outside timed region)."""
    m = num_groups * m_per_group
    tokens = torch.randn((m, K), dtype=torch.bfloat16, device="cuda")
    expert_idxs = torch.randint(0, num_groups, (m,), device="cuda")
    output_tokens, _, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(
        tokens, expert_idxs, num_groups
    )
    actual_M = output_tokens.shape[0]
    gi = _make_group_indices(grouped_mm_offs, actual_M)
    return output_tokens, grouped_mm_offs, ks, ks_tensor, actual_M, gi


def bench_grouped_fwd(num_groups: int, m_per_group: int, N: int, K: int) -> Metrics:
    torch.manual_seed(42)
    input_tokens, grouped_mm_offs, ks, ks_tensor, actual_M, gi = _make_grouped_inputs(
        num_groups, m_per_group, K
    )

    ref = GroupedLinear(num_groups, K, N).cuda().bfloat16()
    nn.init.normal_(ref.weight)
    fp8 = FP8GroupedLinear(num_groups, K, N).cuda().bfloat16()
    fp8.weight.data.copy_(ref.weight.data)

    def ref_fwd():
        return ref(input_tokens, grouped_mm_offs, ks, ks_tensor)

    def fp8_fwd():
        return fp8(input_tokens, grouped_mm_offs, ks, ks_tensor, group_indices=gi)

    with nvtx.range(f"test run grouped fwd bf16 G={num_groups} M/G={m_per_group} N={N} K={K}"):
        ref_fwd()
    with nvtx.range(f"test run grouped fwd dg fp8 G={num_groups} M/G={m_per_group} N={N} K={K}"):
        fp8_fwd()

    with nvtx.range(f"grouped fwd bf16 G={num_groups} M/G={m_per_group} N={N} K={K}"):
        ref_time = do_bench(ref_fwd)
    with nvtx.range(f"grouped fwd dg fp8 G={num_groups} M/G={m_per_group} N={N} K={K}"):
        fp8_time = do_bench(fp8_fwd)

    te_time = float("nan")
    if HAS_TE:
        te_layer = te.GroupedLinear(num_groups, K, N, bias=False).cuda().bfloat16()
        te_m = sum(ks)
        te_input = input_tokens[:te_m]

        def te_fwd():
            with te.fp8_autocast(enabled=True, fp8_recipe=_TE_FP8_RECIPE):
                return te_layer(te_input, ks)

        with nvtx.range(f"test run grouped fwd te G={num_groups} M/G={m_per_group} N={N} K={K}"):
            te_fwd()
        with nvtx.range(f"grouped fwd te G={num_groups} M/G={m_per_group} N={N} K={K}"):
            te_time = do_bench(te_fwd)

    flops = 2 * actual_M * N * K
    io_bytes = 2 * (actual_M * K + num_groups * N * K + actual_M * N)
    return Metrics(ref_time, fp8_time, flops, io_bytes, te_ms=te_time)


def bench_grouped_bwd(num_groups: int, m_per_group: int, N: int, K: int) -> Metrics:
    torch.manual_seed(42)
    input_tokens, grouped_mm_offs, ks, ks_tensor, actual_M, gi = _make_grouped_inputs(
        num_groups, m_per_group, K
    )
    input_tokens = input_tokens.detach().requires_grad_(True)
    grad = torch.randn((actual_M, N), dtype=torch.bfloat16, device="cuda")

    ref = GroupedLinear(num_groups, K, N).cuda().bfloat16()
    nn.init.normal_(ref.weight)
    fp8 = FP8GroupedLinear(num_groups, K, N).cuda().bfloat16()
    fp8.weight.data.copy_(ref.weight.data)

    ref_input = input_tokens.clone().detach().requires_grad_(True)
    fp8_input = input_tokens.clone().detach().requires_grad_(True)

    ref_o = ref(ref_input, grouped_mm_offs, ks, ks_tensor)
    fp8_o = fp8(fp8_input, grouped_mm_offs, ks, ks_tensor, group_indices=gi)

    def ref_bwd():
        ref_input.grad = None
        ref.weight.grad = None
        ref_o.backward(grad, retain_graph=True)

    def fp8_bwd():
        fp8_input.grad = None
        fp8.weight.grad = None
        fp8_o.backward(grad, retain_graph=True)

    with nvtx.range(f"test run grouped bwd bf16 G={num_groups} M/G={m_per_group} N={N} K={K}"):
        ref_bwd()
    with nvtx.range(f"test run grouped bwd dg fp8 G={num_groups} M/G={m_per_group} N={N} K={K}"):
        fp8_bwd()

    with nvtx.range(f"grouped bwd bf16 G={num_groups} M/G={m_per_group} N={N} K={K}"):
        ref_time = do_bench(ref_bwd)
    with nvtx.range(f"grouped bwd dg fp8 G={num_groups} M/G={m_per_group} N={N} K={K}"):
        fp8_time = do_bench(fp8_bwd)

    te_time = float("nan")
    if HAS_TE:
        te_layer = te.GroupedLinear(num_groups, K, N, bias=False).cuda().bfloat16()
        te_m = sum(ks)
        te_input = input_tokens[:te_m].clone().detach().requires_grad_(True)
        te_grad = grad[:te_m]

        def te_fwd_bwd():
            te_input.grad = None
            for p in te_layer.parameters():
                p.grad = None
            with te.fp8_autocast(enabled=True, fp8_recipe=_TE_FP8_RECIPE):
                te_o = te_layer(te_input, ks)
            te_o.backward(te_grad)

        def te_fwd_only():
            with te.fp8_autocast(enabled=True, fp8_recipe=_TE_FP8_RECIPE):
                return te_layer(te_input, ks)

        with nvtx.range(f"test run grouped bwd te G={num_groups} M/G={m_per_group} N={N} K={K}"):
            te_fwd_bwd()
        with nvtx.range(f"grouped fwd+bwd te G={num_groups} M/G={m_per_group} N={N} K={K}"):
            te_fwd_bwd_time = do_bench(te_fwd_bwd)
        with nvtx.range(
            f"grouped fwd te (for subtraction) G={num_groups} M/G={m_per_group} N={N} K={K}"
        ):
            te_fwd_time = do_bench(te_fwd_only)
        te_time = te_fwd_bwd_time - te_fwd_time

    flops = 4 * actual_M * N * K
    io_bytes = 2 * (2 * actual_M * N + 2 * actual_M * K + 2 * num_groups * N * K)
    return Metrics(ref_time, fp8_time, flops, io_bytes, te_ms=te_time)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _format_table(
    title: str,
    dim_labels: str,
    rows: list,
    format_dims,
) -> str:
    has_te = any(not math.isnan(m.te_ms) for *_, fwd, bwd in rows for m in (fwd, bwd))

    if has_te:
        metrics_hdr = f"{'bf16 us':>8} {'dg us':>8}"
        metrics_hdr += f" {'te us':>8}"
        metrics_hdr += f" {'TB/s':>6} {'TF/s':>6} {_GREEN}{'dg spd':>6}{_RESET}"
        metrics_hdr += f" {_BLUE}{'te spd':>6}{_RESET}"
    else:
        metrics_hdr = f"{'bf16 us':>8} {'fp8 us':>8} {'TB/s':>6} {'TF/s':>6} {'speedup':>7}"

    _vis_len = len(metrics_hdr.replace(_GREEN, "").replace(_BLUE, "").replace(_RESET, ""))
    hdr1 = f"{dim_labels}  {' fwd ':-^{_vis_len}}  {' bwd ':-^{_vis_len}}"
    hdr2 = f"{' ' * len(dim_labels)}  {metrics_hdr}  {metrics_hdr}"
    sep = "=" * (len(dim_labels) + 2 + _vis_len + 2 + _vis_len)
    lines = ["", title, sep, hdr1, hdr2, sep]

    for *dims, fwd, bwd in rows:
        dim_str = format_dims(*dims)
        if has_te:
            fwd_str = _format_metrics_multi(fwd, has_te)
            bwd_str = _format_metrics_multi(bwd, has_te)
        else:
            fwd_str = (
                f"{fwd.ref_ms * 1e3:8.1f} {fwd.our_ms * 1e3:8.1f} {fwd.io_bytes / fwd.our_ms / 1e9:6.2f}"
                f" {fwd.flops / fwd.our_ms / 1e9:6.2f} {fwd.ref_ms / fwd.our_ms:6.2f}x"
            )
            bwd_str = (
                f"{bwd.ref_ms * 1e3:8.1f} {bwd.our_ms * 1e3:8.1f} {bwd.io_bytes / bwd.our_ms / 1e9:6.2f}"
                f" {bwd.flops / bwd.our_ms / 1e9:6.2f} {bwd.ref_ms / bwd.our_ms:6.2f}x"
            )
        lines.append(f"{dim_str}  {fwd_str}  {bwd_str}")

    lines.append(sep)
    return "\n".join(lines)


_GREEN = "\033[32m"
_BLUE = "\033[34m"
_RESET = "\033[0m"


def _format_metrics_multi(m: Metrics, has_te: bool) -> str:
    s = f"{m.ref_ms * 1e3:8.1f} {m.our_ms * 1e3:8.1f}"
    if has_te:
        te_us = f"{m.te_ms * 1e3:8.1f}" if not math.isnan(m.te_ms) else f"{'N/A':>8}"
        s += f" {te_us}"
    s += f" {m.io_bytes / m.our_ms / 1e9:6.2f} {m.flops / m.our_ms / 1e9:6.2f}"
    s += f" {_GREEN}{m.ref_ms / m.our_ms:5.2f}x{_RESET}"
    if has_te:
        te_spd = f"{m.ref_ms / m.te_ms:5.2f}x" if not math.isnan(m.te_ms) else f"{'N/A':>6}"
        s += f" {_BLUE}{te_spd}{_RESET}"
    return s


def main() -> None:
    if not HAS_TE:
        print("WARNING: transformer_engine not found - TE columns will be omitted.")

    te_tag = " vs TE FP8" if HAS_TE else ""

    # --- Non-grouped ---
    normal_rows = []
    for M, N, K in NORMAL_WORKLOADS:
        fwd = bench_normal_fwd(M, N, K)
        bwd = bench_normal_bwd(M, N, K)
        normal_rows.append((M, N, K, fwd, bwd))

    dim_labels = f"{'M':>5} {'N':>6} {'K':>6}"
    print(
        _format_table(
            f"Non-grouped: BF16 vs DeepGEMM FP8{te_tag}",
            dim_labels,
            normal_rows,
            lambda m, n, k: f"{m:>5} {n:>6} {k:>6}",
        )
    )

    # --- Grouped ---
    grouped_rows = []
    for num_groups, m_per_group, N, K in GROUP_WORKLOADS:
        fwd = bench_grouped_fwd(num_groups, m_per_group, N, K)
        bwd = bench_grouped_bwd(num_groups, m_per_group, N, K)
        grouped_rows.append((num_groups, m_per_group, N, K, fwd, bwd))

    dim_labels = f"{'G':>3} {'M/G':>6} {'N':>6} {'K':>6}"
    print(
        _format_table(
            f"Grouped: BF16 vs DeepGEMM FP8{te_tag}",
            dim_labels,
            grouped_rows,
            lambda g, mpg, n, k: f"{g:>3} {mpg:>6} {n:>6} {k:>6}",
        )
    )


if __name__ == "__main__":
    main()
