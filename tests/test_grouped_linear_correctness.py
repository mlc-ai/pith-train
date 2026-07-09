"""
Correctness test for F.grouped_mm and GroupedLinear.

This test compares the F.grouped_mm implementation against a reference
for-loop based matmul implementation to ensure correctness.
"""

from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn.functional as F

from pithtrain.operators.grouped_linear import GroupedLinear
from pithtrain.operators.token_scatter import _GEMM_ALLOC_ALIGNMENT, scatter_for_grouped_gemm


def reference_grouped_linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation using for-loop and regular matmul.

    Args:
        input: [M_total, K] packed input tensor
        weight: [num_groups, N, K] weight tensor
        grouped_mm_offs: [num_groups] cumulative offsets (cumsum of group sizes)

    Returns:
        output: [M_total, N] packed output tensor
    """
    M_total, K = input.shape
    num_groups, N, K_w = weight.shape
    assert K == K_w

    # Extract group sizes from offs
    group_sizes = []
    if len(grouped_mm_offs) > 0:
        offs = [0] + grouped_mm_offs.tolist()
        group_sizes = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]

    # Compute output for each group
    output = torch.zeros((M_total, N), device=input.device, dtype=input.dtype)
    offset = 0
    for g, size in enumerate(group_sizes):
        if size > 0:
            # output[offset:offset+size] = input[offset:offset+size] @ weight[g].T
            output[offset : offset + size] = torch.matmul(
                input[offset : offset + size], weight[g].T
            )
        offset += size

    return output


def reference_grouped_linear_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for backward pass using for-loop and regular matmul.

    Args:
        grad_output: [M_total, N] gradient of output
        input: [M_total, K] input tensor
        weight: [num_groups, N, K] weight tensor
        grouped_mm_offs: [num_groups] cumulative offsets

    Returns:
        grad_input: [M_total, K] gradient of input
        grad_weight: [num_groups, N, K] gradient of weight
    """
    M_total, N = grad_output.shape
    num_groups, N_w, K = weight.shape
    assert N == N_w

    # Extract group sizes from offs
    group_sizes = []
    if len(grouped_mm_offs) > 0:
        group_sizes.append(grouped_mm_offs[0].item())
        for i in range(1, len(grouped_mm_offs)):
            group_sizes.append(grouped_mm_offs[i].item() - grouped_mm_offs[i - 1].item())

    # Compute grad_input: grad_output @ weight
    grad_input = torch.zeros((M_total, K), device=input.device, dtype=input.dtype)
    offset = 0
    for g, size in enumerate(group_sizes):
        if size > 0:
            grad_input[offset : offset + size] = torch.matmul(
                grad_output[offset : offset + size], weight[g]
            )
        offset += size

    # Compute grad_weight: grad_output.T @ input for each group
    grad_weight = torch.zeros((num_groups, N, K), device=weight.device, dtype=weight.dtype)
    offset = 0
    for g, size in enumerate(group_sizes):
        if size > 0:
            grad_weight[g] = torch.matmul(
                grad_output[offset : offset + size].T, input[offset : offset + size]
            )
        offset += size

    return grad_input, grad_weight


def test_grouped_linear_forward():
    """Test forward pass correctness."""
    # Test configurations
    configs = [
        # (num_groups, group_sizes, K, N)
        ("All groups non-empty", 8, [5, 3, 2, 4, 6, 1, 3, 2], 128, 256),
        ("Some empty groups", 8, [5, 0, 3, 0, 0, 2, 0, 1], 128, 256),
        ("Single group", 1, [10], 64, 128),
        ("Many small groups", 16, [1, 2, 1, 0, 3, 1, 0, 2, 1, 0, 1, 2, 0, 1, 1, 2], 64, 64),
        ("Large groups", 4, [100, 50, 75, 125], 256, 512),
    ]

    device = torch.device("cuda")
    dtype = torch.float32

    for test_name, num_groups, group_sizes, K, N in configs:
        # Create packed input
        M_total = sum(group_sizes)
        input = torch.randn(M_total, K, device=device, dtype=dtype)

        # Create weight
        weight = torch.randn(num_groups, N, K, device=device, dtype=dtype)

        # Create grouped_mm_offs (cumsum of group sizes)
        # Note: cumsum returns int64, so we need to cast to int32 after
        grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

        # Reference implementation
        output_ref = reference_grouped_linear_forward(input, weight, grouped_mm_offs)

        # F.grouped_mm implementation
        output_test = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)

        # Compare
        max_diff = (output_test - output_ref).abs().max().item()

        assert torch.allclose(output_test, output_ref, rtol=1e-5, atol=1e-6), (
            f"Forward pass mismatch for '{test_name}'! Max diff: {max_diff}"
        )


def test_grouped_linear_backward():
    """Test backward pass correctness."""
    # Test configurations
    configs = [
        # (num_groups, group_sizes, K, N)
        ("All groups non-empty", 8, [5, 3, 2, 4, 6, 1, 3, 2], 128, 256),
        ("Some empty groups", 8, [5, 0, 3, 0, 0, 2, 0, 1], 128, 256),
        ("Single group", 1, [10], 64, 128),
        ("Many small groups", 16, [1, 2, 1, 0, 3, 1, 0, 2, 1, 0, 1, 2, 0, 1, 1, 2], 64, 64),
    ]

    device = torch.device("cuda")
    dtype = torch.float32

    for test_name, num_groups, group_sizes, K, N in configs:
        # Create packed input
        M_total = sum(group_sizes)
        input = torch.randn(M_total, K, device=device, dtype=dtype, requires_grad=True)

        # Create weight
        weight = torch.randn(num_groups, N, K, device=device, dtype=dtype, requires_grad=True)

        # Create grouped_mm_offs (cumsum of group sizes)
        # Note: cumsum returns int64, so we need to cast to int32 after
        grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

        # Forward pass
        output = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)

        # Create gradient for output
        grad_output = torch.randn_like(output)

        # Backward pass
        output.backward(grad_output)

        # Get computed gradients
        grad_input_test = input.grad.clone()
        grad_weight_test = weight.grad.clone()

        # Compute reference gradients
        grad_input_ref, grad_weight_ref = reference_grouped_linear_backward(
            grad_output, input.detach(), weight.detach(), grouped_mm_offs
        )

        # Compare input gradients
        max_diff_input = (grad_input_test - grad_input_ref).abs().max().item()

        assert torch.allclose(grad_input_test, grad_input_ref, rtol=1e-5, atol=1e-6), (
            f"Input gradient mismatch for '{test_name}'! Max diff: {max_diff_input}"
        )

        # Compare weight gradients
        max_diff_weight = (grad_weight_test - grad_weight_ref).abs().max().item()

        assert torch.allclose(grad_weight_test, grad_weight_ref, rtol=1e-5, atol=1e-6), (
            f"Weight gradient mismatch for '{test_name}'! Max diff: {max_diff_weight}"
        )


def test_grouped_linear_module():
    """Test GroupedLinear module (wrapper around F.grouped_mm)."""
    device = torch.device("cuda")
    dtype = torch.float32

    # Configuration
    num_groups = 8
    group_sizes = [5, 0, 3, 0, 0, 2, 0, 1]
    in_features = 128
    out_features = 256

    # Create module
    module = GroupedLinear(num_groups, in_features, out_features).to(device).to(dtype)

    # Create packed input
    M_total = sum(group_sizes)
    input = torch.randn(M_total, in_features, device=device, dtype=dtype, requires_grad=True)

    # Create grouped_mm_offs
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    # Forward pass
    output = module(input, grouped_mm_offs)

    # Reference forward
    output_ref = reference_grouped_linear_forward(
        input.detach(), module.weight.detach(), grouped_mm_offs
    )

    # Compare forward
    max_diff = (output - output_ref).abs().max().item()
    assert torch.allclose(output, output_ref, rtol=1e-5, atol=1e-6), (
        f"Module forward mismatch! Max diff: {max_diff}"
    )

    # Backward pass
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    grad_input_test = input.grad.clone()
    grad_weight_test = module.weight.grad.clone()

    # Reference backward
    grad_input_ref, grad_weight_ref = reference_grouped_linear_backward(
        grad_output, input.detach(), module.weight.detach(), grouped_mm_offs
    )

    # Compare backward
    max_diff_input = (grad_input_test - grad_input_ref).abs().max().item()
    max_diff_weight = (grad_weight_test - grad_weight_ref).abs().max().item()

    assert torch.allclose(grad_input_test, grad_input_ref, rtol=1e-5, atol=1e-6), (
        f"Module input gradient mismatch! Max diff: {max_diff_input}"
    )
    assert torch.allclose(grad_weight_test, grad_weight_ref, rtol=1e-5, atol=1e-6), (
        f"Module weight gradient mismatch! Max diff: {max_diff_weight}"
    )


def test_edge_cases():
    """Test edge cases."""
    device = torch.device("cuda")
    dtype = torch.float32

    # Test 1: All empty groups
    num_groups = 4
    group_sizes = [0, 0, 0, 0]
    M_total = sum(group_sizes)
    K, N = 64, 128

    input = torch.randn(M_total, K, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_groups, N, K, device=device, dtype=dtype, requires_grad=True)
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    output = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)
    assert output.shape == (M_total, N), f"Shape mismatch: {output.shape}"

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    assert input.grad.shape == input.shape, "Input grad shape mismatch"
    assert weight.grad.shape == weight.shape, "Weight grad shape mismatch"

    # Test 2: Single token per group
    num_groups = 8
    group_sizes = [1, 1, 1, 1, 1, 1, 1, 1]
    M_total = sum(group_sizes)
    K, N = 32, 64

    input = torch.randn(M_total, K, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_groups, N, K, device=device, dtype=dtype, requires_grad=True)
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    output = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)
    output_ref = reference_grouped_linear_forward(input.detach(), weight.detach(), grouped_mm_offs)

    max_diff = (output - output_ref).abs().max().item()
    assert torch.allclose(output, output_ref, rtol=1e-5, atol=1e-6), (
        f"Single token per group mismatch! Max diff: {max_diff}"
    )

    # Test 3: Very large group
    num_groups = 1
    group_sizes = [1000]
    M_total = sum(group_sizes)
    K, N = 128, 256

    input = torch.randn(M_total, K, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_groups, N, K, device=device, dtype=dtype, requires_grad=True)
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    output = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)
    output_ref = reference_grouped_linear_forward(input.detach(), weight.detach(), grouped_mm_offs)

    max_diff = (output - output_ref).abs().max().item()
    assert torch.allclose(output, output_ref, rtol=1e-5, atol=1e-6), (
        f"Large group mismatch! Max diff: {max_diff}"
    )


def _verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out, reverse_idxs, offs):
    """
    Verify scatter_for_grouped_gemm output is semantically correct:
    1. grouped_mm_offs has correct shape and values (padded cumsum)
    2. reverse_idxs correctly maps: out[reverse_idxs[i]] == sorted_tokens[i]
    3. Within each expert group, the correct set of token rows exists
    4. Padding rows are all zeros
    5. Rows beyond offs[-1] (over-allocation) are all zeros
    """
    # Check reverse_idxs recovers original tokens
    recovered = out[reverse_idxs]
    assert torch.equal(recovered, sorted_tokens), (
        f"out[reverse_idxs] != sorted_tokens, max diff: "
        f"{(recovered - sorted_tokens).abs().max().item()}"
    )

    # Check each expert group contains the right tokens
    group_starts = [0] + offs.tolist()
    for g in range(num_groups):
        start = group_starts[g]
        end = group_starts[g + 1]
        # Find which input tokens belong to this expert
        mask = expert_idxs == g
        expected_rows = sorted_tokens[mask]
        # Find which output rows in this group are non-padding
        group_block = out[start:end]
        # The first expected_rows.shape[0] rows (in some order) should match;
        # remaining rows should be zero (padding)
        n_tokens = expected_rows.shape[0]
        # Gather actual token rows via reverse_idxs
        actual_positions = reverse_idxs[mask]
        actual_rows = out[actual_positions]
        # Sort both by first element for comparison (tokens are random so unique)
        if n_tokens > 0:
            assert torch.equal(actual_rows, expected_rows), f"Group {g}: token content mismatch"
        # Check padding rows are zero
        if end - start > n_tokens:
            # Get all positions used by real tokens in this group
            used_positions = set(actual_positions.tolist())
            for pos in range(start, end):
                if pos not in used_positions:
                    assert torch.all(group_block[pos - start] == 0), (
                        f"Group {g}: non-zero padding at position {pos}"
                    )


def test_scatter_for_grouped_gemm():
    """Test scatter_for_grouped_gemm produces semantically correct results."""
    device = torch.device("cuda")
    dtype = torch.float32

    configs = [
        # (test_name, m, hidden_size, num_groups, padding_alignment)
        ("Basic", 64, 256, 8, 128),
        ("Large hidden", 32, 1024, 4, 128),
        ("No padding", 48, 128, 6, 1),
        ("Hidden not multiple of BLOCK_H", 20, 300, 4, 128),
        ("Small hidden", 16, 7, 4, 128),
        ("Many experts", 128, 256, 32, 128),
        ("Single token", 1, 128, 4, 128),
    ]

    for test_name, m, hidden_size, num_groups, padding_alignment in configs:
        sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
        expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

        # New path
        out_new, reverse_new, offs_new, ks_new, ks_tensor_new = scatter_for_grouped_gemm(
            sorted_tokens, expert_idxs, num_groups, padding_alignment
        )

        # Verify grouped_mm_offs shape
        assert offs_new.shape == (num_groups,), f"offs shape {offs_new.shape} != ({num_groups},)"
        assert offs_new.dtype == torch.int32

        # Verify output shape: over-allocated up to _GEMM_ALLOC_ALIGNMENT for
        # allocator-friendliness, and rows in [offs[-1], out.shape[0]) zeroed
        # by the scatter kernel so downstream grouped_mm is safe.
        m_padded = offs_new[-1].item()
        assert out_new.shape[0] >= m_padded, f"output rows {out_new.shape[0]} < offs[-1]={m_padded}"
        assert out_new.shape[0] % _GEMM_ALLOC_ALIGNMENT == 0, (
            f"output rows {out_new.shape[0]} not aligned to {_GEMM_ALLOC_ALIGNMENT}"
        )
        assert out_new.shape[1] == hidden_size, (
            f"output cols {out_new.shape[1]} != hidden_size={hidden_size}"
        )
        if out_new.shape[0] > m_padded:
            tail = out_new[m_padded:]
            assert torch.all(tail == 0), (
                f"over-allocated tail [{m_padded}:{out_new.shape[0]}) must be zero"
            )

        # Verify reverse_shuffle_idxs
        assert reverse_new.shape == (m,)
        assert reverse_new.dtype == torch.int64

        # Full semantic verification
        _verify_scatter_result(
            sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new
        )


def test_scatter_for_grouped_gemm_edge_cases():
    """Test edge cases for scatter_for_grouped_gemm."""
    device = torch.device("cuda")
    dtype = torch.float32

    # Edge case 1: m=0
    sorted_tokens = torch.randn(0, 128, device=device, dtype=dtype)
    expert_idxs = torch.empty(0, device=device, dtype=torch.int64)
    out, reverse, offs, _, _ = scatter_for_grouped_gemm(sorted_tokens, expert_idxs, 8)
    assert out.shape == (0, 128), f"Expected (0, 128), got {out.shape}"
    assert reverse.shape == (0,), f"Expected (0,), got {reverse.shape}"
    assert offs.shape == (0,), f"Expected (0,), got {offs.shape}"

    # Edge case 2: All tokens to one expert
    m, hidden_size, num_groups = 32, 256, 8
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
    expert_idxs = torch.full((m,), 3, device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    _verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)
    # All tokens should be in group 3's block; other groups should be all-zero padding
    group_starts = [0] + offs_new.tolist()
    for g in range(num_groups):
        if g != 3:
            block = out_new[group_starts[g] : group_starts[g + 1]]
            assert torch.all(block == 0), f"Group {g} should be all zeros"

    # Edge case 3: hidden_size not a multiple of BLOCK_H (256)
    m, hidden_size, num_groups = 16, 100, 4
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
    expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    _verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)

    # Edge case 4: hidden_size=1
    m, hidden_size, num_groups = 8, 1, 2
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
    expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    _verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)

    # Edge case 5: bf16 dtype
    m, hidden_size, num_groups = 24, 256, 4
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=torch.bfloat16)
    expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    _verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)

    # Edge case 6: Large hidden_size (multiple BLOCK_H iterations)
    m, hidden_size, num_groups = 16, 768, 4
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
    expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    _verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)


def test_grouped_mm_over_allocated_input():
    """
    Test that F.grouped_mm handles over-allocated input correctly.

    When scatter_for_grouped_gemm uses over-allocation (input.shape[0] > offs[-1]),
    F.grouped_mm should only read up to offs[-1] and produce correct results.
    """
    device = torch.device("cuda")
    dtype = torch.float32

    configs = [
        # (test_name, num_groups, group_sizes, extra_rows, K, N)
        ("Small over-alloc", 4, [5, 3, 2, 4], 50, 128, 256),
        ("Large over-alloc", 8, [10, 5, 8, 3, 7, 2, 6, 4], 200, 128, 256),
        ("Some empty groups", 8, [5, 0, 3, 0, 0, 2, 0, 1], 100, 128, 256),
        ("Single group", 1, [10], 20, 64, 128),
    ]

    for test_name, num_groups, group_sizes, extra_rows, K, N in configs:
        M_exact = sum(group_sizes)
        M_over = M_exact + extra_rows

        # Create over-allocated input (extra trailing zero rows)
        input_exact = torch.randn(M_exact, K, device=device, dtype=dtype)
        input_over = torch.zeros(M_over, K, device=device, dtype=dtype)
        input_over[:M_exact] = input_exact

        weight = torch.randn(num_groups, N, K, device=device, dtype=dtype)
        grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

        # Reference: exact-size input
        output_ref = F.grouped_mm(input_exact, weight.transpose(1, 2), offs=grouped_mm_offs)

        # Test: over-allocated input
        output_over = F.grouped_mm(input_over, weight.transpose(1, 2), offs=grouped_mm_offs)

        # The first M_exact rows should match exactly
        # (output_over may have M_over rows total, but only first M_exact matter)
        output_over_trimmed = output_over[:M_exact]
        max_diff = (output_over_trimmed - output_ref).abs().max().item()
        assert torch.allclose(output_over_trimmed, output_ref, rtol=1e-5, atol=1e-6), (
            f"Over-allocated grouped_mm mismatch for '{test_name}'! Max diff: {max_diff}"
        )


def test_scatter_then_grouped_mm_end_to_end():
    """
    End-to-end test: scatter_for_grouped_gemm -> F.grouped_mm.

    Verifies that the over-allocated output from scatter feeds correctly
    into grouped_mm and produces the same result as exact-size allocation.
    """
    device = torch.device("cuda")
    dtype = torch.float32

    configs = [
        # (test_name, m, hidden_size, num_groups, out_features)
        ("Basic", 64, 128, 8, 256),
        ("Large", 256, 256, 8, 512),
        ("Many experts", 128, 128, 32, 256),
    ]

    for test_name, m, hidden_size, num_groups, out_features in configs:
        sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
        expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)
        weight = torch.randn(num_groups, out_features, hidden_size, device=device, dtype=dtype)

        # scatter (may over-allocate)
        out_tokens, reverse_idxs, offs, _, _ = scatter_for_grouped_gemm(
            sorted_tokens, expert_idxs, num_groups
        )

        # grouped_mm on potentially over-allocated tensor
        mm_result = F.grouped_mm(out_tokens, weight.transpose(1, 2), offs=offs)

        # Gather results back to original order
        result = mm_result[reverse_idxs]

        # Reference: per-expert matmul
        result_ref = torch.zeros(m, out_features, device=device, dtype=dtype)
        for g in range(num_groups):
            mask = expert_idxs == g
            if mask.any():
                result_ref[mask] = sorted_tokens[mask] @ weight[g].T

        max_diff = (result - result_ref).abs().max().item()
        assert torch.allclose(result, result_ref, rtol=1e-4, atol=1e-4), (
            f"End-to-end mismatch for '{test_name}'! Max diff: {max_diff}"
        )


def test_grouped_linear_weight_grad_store():
    """GroupedLinear correctly defers weight gradients via WeightGradStore."""
    from pithtrain.dualpipe.utils import WeightGradStore

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_groups, in_features, out_features = 4, 128, 256
    group_sizes = [16, 8, 12, 4]
    M_total = sum(group_sizes)
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    gl_ref = GroupedLinear(num_groups, in_features, out_features).to(device).to(dtype)
    torch.nn.init.normal_(gl_ref.weight, std=0.02)

    x_raw = torch.randn(M_total, in_features, device=device, dtype=dtype)
    grad = torch.randn(M_total, out_features, device=device, dtype=dtype)

    # Reference: no WeightGradStore.
    x_ref = x_raw.detach().clone().requires_grad_(True)
    gl_ref(x_ref, grouped_mm_offs).backward(grad)
    ref_weight_grad = gl_ref.weight.grad.clone()
    ref_input_grad = x_ref.grad.clone()

    # Deferred path.
    gl = GroupedLinear(num_groups, in_features, out_features).to(device).to(dtype)
    gl.weight.data.copy_(gl_ref.weight.data)
    x_def = x_raw.detach().clone().requires_grad_(True)

    WeightGradStore.enabled = True
    try:
        gl(x_def, grouped_mm_offs).backward(grad)

        assert gl.weight.grad is None, "Weight grad should be deferred"
        assert x_def.grad is not None, "Input grad should be on the critical path"

        input_diff = (x_def.grad - ref_input_grad).abs().max().item()
        assert torch.allclose(x_def.grad, ref_input_grad, rtol=1e-3, atol=1e-3), (
            f"Input grad diff = {input_diff}"
        )

        WeightGradStore.flush()
        WeightGradStore.pop()

        assert gl.weight.grad is not None, "Weight grad should exist after pop"
        assert gl.weight.grad.shape == gl.weight.shape
        weight_diff = (gl.weight.grad - ref_weight_grad).abs().max().item()
        assert torch.allclose(gl.weight.grad, ref_weight_grad, rtol=1e-3, atol=1e-3), (
            f"Deferred vs direct weight grad diff = {weight_diff}"
        )
    finally:
        WeightGradStore.enabled = False
        WeightGradStore.clear()


def test_gpt_oss_experts_weight_grad_store_matches_direct():
    """
    End-to-end sanity check: GptOssExperts (now backed by GroupedLinearFunc for
    the expert GEMMs) produces the same input / weight / bias gradients whether
    WeightGradStore is enabled or disabled.

    Only the expert-weight wgrad is routed through WeightGradStore; the bias
    add lives outside GroupedLinearFunc so its grad is computed eagerly via
    autograd on both paths.
    """
    from pithtrain.dualpipe.utils import WeightGradStore
    from pithtrain.models.gpt_oss import GptOssExperts

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_experts = 4
    hidden_size = 128
    intermediate_size = 256
    swiglu_limit = 7.0
    config = SimpleNamespace(
        hidden_size=hidden_size, intermediate_size=intermediate_size, swiglu_limit=swiglu_limit
    )

    # Random routing with one empty group to exercise the zero-token branch.
    group_sizes = [8, 0, 12, 4]
    M_total = sum(group_sizes)
    ks = list(group_sizes)
    offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    torch.manual_seed(123)
    x_raw = torch.randn(M_total, hidden_size, device=device, dtype=dtype)

    experts_ref = GptOssExperts(config, num_experts)
    experts_ref = experts_ref.to(device=device, dtype=dtype)
    with torch.no_grad():
        experts_ref.gate_up_proj.normal_(std=0.02)
        experts_ref.down_proj.normal_(std=0.02)
        experts_ref.gate_up_proj_bias.normal_(std=0.02)
        experts_ref.down_proj_bias.normal_(std=0.02)

    experts_def = GptOssExperts(config, num_experts)
    experts_def = experts_def.to(device=device, dtype=dtype)
    experts_def.load_state_dict(experts_ref.state_dict())

    # Run 1: WeightGradStore disabled.
    x_ref = x_raw.detach().clone().requires_grad_(True)
    out_ref = experts_ref(x_ref, offs, ks=ks)
    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)

    # Run 2: WeightGradStore enabled (weight wgrads deferred, bias bgrads eager).
    x_def = x_raw.detach().clone().requires_grad_(True)
    WeightGradStore.enabled = True
    try:
        out_def = experts_def(x_def, offs, ks=ks)

        fwd_diff = (out_def - out_ref).abs().max().item()
        assert torch.allclose(out_def, out_ref, rtol=1e-3, atol=1e-3), (
            f"forward mismatch between store on/off, max diff {fwd_diff}"
        )

        out_def.backward(grad)

        # Expert weights go through GroupedLinearFunc's deferred wgrad path.
        for p_def, name in [
            (experts_def.gate_up_proj, "gate_up_proj"),
            (experts_def.down_proj, "down_proj"),
        ]:
            assert p_def.grad is None, f"[deferred] {name} grad should be deferred"

        # Biases are added outside GroupedLinearFunc; autograd populates them eagerly.
        for p_def, name in [
            (experts_def.gate_up_proj_bias, "gate_up_proj_bias"),
            (experts_def.down_proj_bias, "down_proj_bias"),
        ]:
            assert p_def.grad is not None, f"[deferred] {name} grad should be eager"

        in_diff = (x_def.grad - x_ref.grad).abs().max().item()
        assert torch.allclose(x_def.grad, x_ref.grad, rtol=1e-3, atol=1e-3), (
            f"input grad mismatch, max diff {in_diff}"
        )

        # bias grads come from autograd's scatter-add (CUDA index_add_ is
        # non-deterministic, so allow ~5% slack on bf16 accumulation).
        for p_ref, p_def, name in [
            (experts_ref.gate_up_proj_bias, experts_def.gate_up_proj_bias, "gate_up_proj_bias"),
            (experts_ref.down_proj_bias, experts_def.down_proj_bias, "down_proj_bias"),
        ]:
            ref_max = p_ref.grad.abs().max().item()
            diff = (p_def.grad - p_ref.grad).abs().max().item()
            assert diff <= max(0.1, ref_max * 0.05), (
                f"{name} grad mismatch, max diff {diff} (max |ref| {ref_max})"
            )

        WeightGradStore.flush()
        WeightGradStore.pop()

        # grouped_mm is deterministic — weight grads must match tightly.
        for p_ref, p_def, name in [
            (experts_ref.gate_up_proj, experts_def.gate_up_proj, "gate_up_proj"),
            (experts_ref.down_proj, experts_def.down_proj, "down_proj"),
        ]:
            assert p_def.grad is not None, f"[deferred] {name} grad missing after pop"
            diff = (p_def.grad - p_ref.grad).abs().max().item()
            assert torch.allclose(p_def.grad, p_ref.grad, rtol=1e-3, atol=1e-3), (
                f"{name} grad mismatch (deferred vs direct), max diff {diff}"
            )
    finally:
        WeightGradStore.enabled = False
        WeightGradStore.clear()
