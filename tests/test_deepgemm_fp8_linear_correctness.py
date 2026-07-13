"""
Correctness test for DeepGEMM FP8Linear and FP8GroupedLinear.

Tests compare DeepGEMM FP8 implementations against BF16 reference with
appropriate tolerances for FP8 precision loss.
"""

import deep_gemm  # noqa: F401
import pytest
import torch
import torch.nn as nn

# Threshold for FP8 vs BF16 comparisons using normalized squared-error metric.
ERR_THRESHOLD = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def calc_diff(x, y):
    x, y = x.detach().double(), y.detach().double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _make_bf16(shape, device="cuda"):
    return torch.randn(shape, device=device, dtype=torch.bfloat16)


def _make_group_indices(grouped_mm_offs, M):
    """
    Build per-row group indices needed by the Hopper (SM90) grouped GEMM.

    Returns None on Blackwell (SM100+) where psum layout is used instead.
    """
    arch_major, _ = torch.cuda.get_device_capability()
    if arch_major >= 10:
        return None
    row_indices = torch.arange(M, device=grouped_mm_offs.device)
    return torch.searchsorted(grouped_mm_offs, row_indices, right=True).to(torch.int32)


def _copy_weights(src, dst):
    """Copy weight (and optionally bias) from *src* to *dst*."""
    dst.weight.data.copy_(src.weight.data)
    if hasattr(src, "bias") and src.bias is not None and dst.bias is not None:
        dst.bias.data.copy_(src.bias.data)


# ---------------------------------------------------------------------------
# FP8Linear forward tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_features,out_features",
    [(128, 256), (256, 128), (512, 512), (256, 1024)],
)
def test_fp8_linear_forward(in_features, out_features):
    """FP8Linear forward output is close to BF16 Linear."""
    from pithtrain.operators.linear import FP8Linear

    bf16_linear = nn.Linear(in_features, out_features, bias=False).cuda().to(torch.bfloat16)
    fp8_linear = FP8Linear(in_features, out_features, bias=False).cuda().to(torch.bfloat16)
    _copy_weights(bf16_linear, fp8_linear)

    x = _make_bf16((4, 32, in_features))
    out_bf16 = bf16_linear(x)
    out_fp8 = fp8_linear(x)

    assert out_fp8.shape == out_bf16.shape
    diff = calc_diff(out_fp8, out_bf16)
    assert diff < ERR_THRESHOLD, f"diff = {diff}"


def test_fp8_linear_forward_with_bias():
    """FP8Linear forward with bias produces correct shape and reasonable values."""
    from pithtrain.operators.linear import FP8Linear

    bf16_linear = nn.Linear(128, 256, bias=True).cuda().to(torch.bfloat16)
    fp8_linear = FP8Linear(128, 256, bias=True).cuda().to(torch.bfloat16)
    _copy_weights(bf16_linear, fp8_linear)

    x = _make_bf16((2, 16, 128))
    out_bf16 = bf16_linear(x)
    out_fp8 = fp8_linear(x)

    assert out_fp8.shape == (2, 16, 256)
    diff = calc_diff(out_fp8, out_bf16)
    assert diff < ERR_THRESHOLD, f"diff = {diff}"


# ---------------------------------------------------------------------------
# FP8Linear backward tests
# ---------------------------------------------------------------------------


def test_fp8_linear_backward_input_grad():
    """FP8Linear backward produces input gradients close to BF16."""
    from pithtrain.operators.linear import FP8Linear

    in_f, out_f = 256, 512
    bf16_linear = nn.Linear(in_f, out_f, bias=False).cuda().to(torch.bfloat16)
    fp8_linear = FP8Linear(in_f, out_f, bias=False).cuda().to(torch.bfloat16)
    _copy_weights(bf16_linear, fp8_linear)

    x_bf16 = _make_bf16((4, 32, in_f)).requires_grad_(True)
    x_fp8 = x_bf16.detach().clone().requires_grad_(True)

    grad = _make_bf16((4, 32, out_f))

    bf16_linear(x_bf16).backward(grad)
    fp8_linear(x_fp8).backward(grad)

    diff = calc_diff(x_fp8.grad, x_bf16.grad)
    assert diff < ERR_THRESHOLD, f"input grad diff = {diff}"


def test_fp8_linear_backward_weight_grad():
    """FP8Linear backward produces weight gradients close to BF16."""
    from pithtrain.operators.linear import FP8Linear

    in_f, out_f = 256, 512
    bf16_linear = nn.Linear(in_f, out_f, bias=False).cuda().to(torch.bfloat16)
    fp8_linear = FP8Linear(in_f, out_f, bias=False).cuda().to(torch.bfloat16)
    _copy_weights(bf16_linear, fp8_linear)

    x = _make_bf16((4, 32, in_f))
    grad = _make_bf16((4, 32, out_f))

    bf16_linear(x).backward(grad)
    fp8_linear(x).backward(grad)

    assert fp8_linear.weight.grad is not None
    assert bf16_linear.weight.grad is not None
    diff = calc_diff(fp8_linear.weight.grad, bf16_linear.weight.grad)
    assert diff < ERR_THRESHOLD, f"weight grad diff = {diff}"


# ---------------------------------------------------------------------------
# FP8GroupedLinear forward tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_groups,in_features,out_features",
    [(4, 128, 256), (8, 256, 512), (2, 256, 128)],
)
def test_fp8_grouped_linear_forward(num_groups, in_features, out_features):
    """FP8GroupedLinear forward output is close to BF16 GroupedLinear."""
    from pithtrain.operators.grouped_linear import FP8GroupedLinear, GroupedLinear
    from pithtrain.operators.token_scatter import scatter_for_grouped_gemm

    bf16_gl = GroupedLinear(num_groups, in_features, out_features).cuda().to(torch.bfloat16)
    nn.init.normal_(bf16_gl.weight, std=0.02)
    fp8_gl = FP8GroupedLinear(num_groups, in_features, out_features).cuda().to(torch.bfloat16)
    fp8_gl.weight.data.copy_(bf16_gl.weight.data)

    # Create tokens with known group assignments
    tokens_per_group = 16
    M_total = num_groups * tokens_per_group
    x = _make_bf16((M_total, in_features))
    expert_idxs = torch.randint(0, num_groups, (M_total,), device="cuda")

    # Use scatter_for_grouped_gemm to get proper grouped layout
    output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(
        x, expert_idxs, num_groups
    )

    gi = _make_group_indices(grouped_mm_offs, output_tokens.shape[0])

    out_bf16 = bf16_gl(output_tokens, grouped_mm_offs)
    out_fp8 = fp8_gl(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi)

    assert out_fp8.shape == out_bf16.shape
    # scatter_for_grouped_gemm now returns exactly-sized tensors (no over-allocation),
    # so all rows are valid - compare full output directly.
    diff = calc_diff(out_fp8, out_bf16)
    assert diff < ERR_THRESHOLD, f"diff = {diff}"


# ---------------------------------------------------------------------------
# FP8GroupedLinear backward tests
# ---------------------------------------------------------------------------


def test_fp8_grouped_linear_backward():
    """FP8GroupedLinear backward produces gradients close to BF16."""
    from pithtrain.operators.grouped_linear import FP8GroupedLinear, GroupedLinear
    from pithtrain.operators.token_scatter import scatter_for_grouped_gemm

    num_groups, in_f, out_f = 4, 128, 256
    bf16_gl = GroupedLinear(num_groups, in_f, out_f).cuda().to(torch.bfloat16)
    nn.init.normal_(bf16_gl.weight, std=0.02)
    fp8_gl = FP8GroupedLinear(num_groups, in_f, out_f).cuda().to(torch.bfloat16)
    fp8_gl.weight.data.copy_(bf16_gl.weight.data)

    tokens_per_group = 16
    M_total = num_groups * tokens_per_group
    x_raw = _make_bf16((M_total, in_f))
    expert_idxs = torch.randint(0, num_groups, (M_total,), device="cuda")

    output_tokens, _, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(
        x_raw, expert_idxs, num_groups
    )

    gi = _make_group_indices(grouped_mm_offs, output_tokens.shape[0])

    x_bf16 = output_tokens.detach().clone().requires_grad_(True)
    x_fp8 = output_tokens.detach().clone().requires_grad_(True)

    grad = _make_bf16((output_tokens.shape[0], out_f))

    bf16_gl(x_bf16, grouped_mm_offs).backward(grad)
    fp8_gl(x_fp8, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi).backward(grad)

    # Input grad check - tensors are exactly sized (no over-allocation)
    assert x_fp8.grad is not None
    diff = calc_diff(x_fp8.grad, x_bf16.grad)
    assert diff < ERR_THRESHOLD, f"input grad diff = {diff}"

    # Weight grad check
    assert fp8_gl.weight.grad is not None
    assert bf16_gl.weight.grad is not None
    diff = calc_diff(fp8_gl.weight.grad, bf16_gl.weight.grad)
    assert diff < ERR_THRESHOLD, f"weight grad diff = {diff}"


# ---------------------------------------------------------------------------
# FP8GroupedLinear + WeightGradStore
# ---------------------------------------------------------------------------


def test_fp8_grouped_linear_weight_grad_store():
    """FP8GroupedLinear correctly defers weight gradients via WeightGradStore."""
    from pithtrain.dualpipe.utils import WeightGradStore
    from pithtrain.operators.grouped_linear import FP8GroupedLinear
    from pithtrain.operators.token_scatter import scatter_for_grouped_gemm

    num_groups, in_f, out_f = 4, 128, 256
    tokens_per_group = 16
    M_total = num_groups * tokens_per_group

    # --- Run without WeightGradStore (reference) ---
    fp8_gl_ref = FP8GroupedLinear(num_groups, in_f, out_f).cuda().to(torch.bfloat16)
    nn.init.normal_(fp8_gl_ref.weight, std=0.02)

    x_raw = _make_bf16((M_total, in_f))
    expert_idxs = torch.randint(0, num_groups, (M_total,), device="cuda")
    output_tokens, _, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(
        x_raw, expert_idxs, num_groups
    )
    gi = _make_group_indices(grouped_mm_offs, output_tokens.shape[0])
    grad = _make_bf16((output_tokens.shape[0], out_f))

    x_ref = output_tokens.detach().clone().requires_grad_(True)
    fp8_gl_ref(x_ref, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi).backward(grad)
    ref_weight_grad = fp8_gl_ref.weight.grad.clone()

    # --- Run with WeightGradStore (deferred) ---
    fp8_gl = FP8GroupedLinear(num_groups, in_f, out_f).cuda().to(torch.bfloat16)
    fp8_gl.weight.data.copy_(fp8_gl_ref.weight.data)

    x_def = output_tokens.detach().clone().requires_grad_(True)

    WeightGradStore.enabled = True
    try:
        fp8_gl(x_def, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi).backward(grad)

        # Weight grad should be deferred (None)
        assert fp8_gl.weight.grad is None, "Weight grad should be deferred"

        # Flush and pop to compute deferred gradients
        WeightGradStore.flush()
        WeightGradStore.pop()

        assert fp8_gl.weight.grad is not None, "Weight grad should exist after pop"
        assert fp8_gl.weight.grad.shape == fp8_gl.weight.shape

        # Deferred result should match direct computation
        diff = calc_diff(fp8_gl.weight.grad, ref_weight_grad)
        assert diff < ERR_THRESHOLD, f"deferred vs direct weight grad diff = {diff}"
    finally:
        WeightGradStore.enabled = False
        WeightGradStore.clear()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_fp8_linear_empty_input():
    """FP8Linear handles zero-length input gracefully."""
    from pithtrain.operators.linear import FP8Linear

    fp8_linear = FP8Linear(128, 256, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(0, 128, device="cuda", dtype=torch.bfloat16)
    out = fp8_linear(x)
    assert out.shape == (0, 256)


def test_fp8_grouped_linear_empty_input():
    """FP8GroupedLinear handles zero tokens gracefully."""
    from pithtrain.operators.grouped_linear import FP8GroupedLinear

    fp8_gl = FP8GroupedLinear(8, 128, 256).cuda().to(torch.bfloat16)
    x = torch.randn(0, 128, device="cuda", dtype=torch.bfloat16)
    offs = torch.zeros(8, device="cuda", dtype=torch.int32)
    ks = [0] * 8
    ks_tensor = torch.zeros(8, device="cuda", dtype=torch.int32)
    out = fp8_gl(x, offs, ks=ks, ks_tensor=ks_tensor)
    assert out.shape == (0, 256)
