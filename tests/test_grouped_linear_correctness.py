from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn.functional as F

from pithtrain.contexts import training
from pithtrain.operators.grouped_linear import GroupedLinear
from pithtrain.operators.token_scatter import scatter_for_grouped_gemm


def reference_grouped_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> torch.Tensor:
    M_total, K = x.shape
    num_groups, N, K_w = weight.shape
    assert K == K_w

    group_sizes = []
    if len(grouped_mm_offs) > 0:
        offs = [0] + grouped_mm_offs.tolist()
        group_sizes = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]

    output = torch.zeros((M_total, N), device=x.device, dtype=x.dtype)
    offset = 0
    for group_idx, size in enumerate(group_sizes):
        if size > 0:
            output[offset : offset + size] = torch.matmul(
                x[offset : offset + size], weight[group_idx].T
            )
        offset += size

    return output


def reference_grouped_linear_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M_total, N = grad_output.shape
    num_groups, N_w, K = weight.shape
    assert N == N_w

    group_sizes = []
    if len(grouped_mm_offs) > 0:
        group_sizes.append(grouped_mm_offs[0].item())
        for i in range(1, len(grouped_mm_offs)):
            group_sizes.append(grouped_mm_offs[i].item() - grouped_mm_offs[i - 1].item())

    grad_input = torch.zeros((M_total, K), device=x.device, dtype=x.dtype)
    offset = 0
    for group_idx, size in enumerate(group_sizes):
        if size > 0:
            grad_input[offset : offset + size] = torch.matmul(
                grad_output[offset : offset + size], weight[group_idx]
            )
        offset += size

    grad_weight = torch.zeros((num_groups, N, K), device=weight.device, dtype=weight.dtype)
    offset = 0
    for group_idx, size in enumerate(group_sizes):
        if size > 0:
            grad_weight[group_idx] = torch.matmul(
                grad_output[offset : offset + size].T, x[offset : offset + size]
            )
        offset += size

    return grad_input, grad_weight


def test_grouped_linear_forward():
    configs = [
        ("All groups non-empty", 8, [5, 3, 2, 4, 6, 1, 3, 2], 128, 256),
        ("Some empty groups", 8, [5, 0, 3, 0, 0, 2, 0, 1], 128, 256),
        ("Single group", 1, [10], 64, 128),
        ("Many small groups", 16, [1, 2, 1, 0, 3, 1, 0, 2, 1, 0, 1, 2, 0, 1, 1, 2], 64, 64),
        ("Large groups", 4, [100, 50, 75, 125], 256, 512),
    ]

    device = torch.device("cuda")
    dtype = torch.float32

    for test_name, num_groups, group_sizes, K, N in configs:
        M_total = sum(group_sizes)
        x = torch.randn(M_total, K, device=device, dtype=dtype)

        weight = torch.randn(num_groups, N, K, device=device, dtype=dtype)

        grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

        output_ref = reference_grouped_linear_forward(x, weight, grouped_mm_offs)

        output_test = F.grouped_mm(x, weight.transpose(1, 2), offs=grouped_mm_offs)

        max_diff = (output_test - output_ref).abs().max().item()

        assert torch.allclose(output_test, output_ref, rtol=1e-5, atol=1e-6), (
            f"Forward pass mismatch for '{test_name}'! Max diff: {max_diff}"
        )


def test_grouped_linear_backward():
    configs = [
        ("All groups non-empty", 8, [5, 3, 2, 4, 6, 1, 3, 2], 128, 256),
        ("Some empty groups", 8, [5, 0, 3, 0, 0, 2, 0, 1], 128, 256),
        ("Single group", 1, [10], 64, 128),
        ("Many small groups", 16, [1, 2, 1, 0, 3, 1, 0, 2, 1, 0, 1, 2, 0, 1, 1, 2], 64, 64),
    ]

    device = torch.device("cuda")
    dtype = torch.float32

    for test_name, num_groups, group_sizes, K, N in configs:
        M_total = sum(group_sizes)
        x = torch.randn(M_total, K, device=device, dtype=dtype, requires_grad=True)

        weight = torch.randn(num_groups, N, K, device=device, dtype=dtype, requires_grad=True)

        grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

        output = F.grouped_mm(x, weight.transpose(1, 2), offs=grouped_mm_offs)

        grad_output = torch.randn_like(output)

        output.backward(grad_output)

        grad_input_test = x.grad.clone()
        grad_weight_test = weight.grad.clone()

        grad_input_ref, grad_weight_ref = reference_grouped_linear_backward(
            grad_output, x.detach(), weight.detach(), grouped_mm_offs
        )

        max_diff_input = (grad_input_test - grad_input_ref).abs().max().item()

        assert torch.allclose(grad_input_test, grad_input_ref, rtol=1e-5, atol=1e-6), (
            f"Input gradient mismatch for '{test_name}'! Max diff: {max_diff_input}"
        )

        max_diff_weight = (grad_weight_test - grad_weight_ref).abs().max().item()

        assert torch.allclose(grad_weight_test, grad_weight_ref, rtol=1e-5, atol=1e-6), (
            f"Weight gradient mismatch for '{test_name}'! Max diff: {max_diff_weight}"
        )


def test_grouped_linear_module():
    device = torch.device("cuda")
    dtype = torch.float32

    num_groups = 8
    group_sizes = [5, 0, 3, 0, 0, 2, 0, 1]
    in_features = 128
    out_features = 256

    module = GroupedLinear(num_groups, in_features, out_features).to(device).to(dtype)
    # The weight is allocated with torch.empty; reused allocator memory can hold NaN.
    torch.nn.init.normal_(module.weight, std=0.02)

    M_total = sum(group_sizes)
    x = torch.randn(M_total, in_features, device=device, dtype=dtype, requires_grad=True)

    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    output = module(x, grouped_mm_offs)

    output_ref = reference_grouped_linear_forward(
        x.detach(), module.weight.detach(), grouped_mm_offs
    )

    max_diff = (output - output_ref).abs().max().item()
    assert torch.allclose(output, output_ref, rtol=1e-5, atol=1e-6), (
        f"Module forward mismatch! Max diff: {max_diff}"
    )

    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    grad_input_test = x.grad.clone()
    grad_weight_test = module.weight.grad.clone()

    grad_input_ref, grad_weight_ref = reference_grouped_linear_backward(
        grad_output, x.detach(), module.weight.detach(), grouped_mm_offs
    )

    max_diff_input = (grad_input_test - grad_input_ref).abs().max().item()
    max_diff_weight = (grad_weight_test - grad_weight_ref).abs().max().item()

    assert torch.allclose(grad_input_test, grad_input_ref, rtol=1e-5, atol=1e-6), (
        f"Module input gradient mismatch! Max diff: {max_diff_input}"
    )
    assert torch.allclose(grad_weight_test, grad_weight_ref, rtol=1e-5, atol=1e-6), (
        f"Module weight gradient mismatch! Max diff: {max_diff_weight}"
    )


def test_edge_cases():
    device = torch.device("cuda")
    dtype = torch.float32

    num_groups = 4
    group_sizes = [0, 0, 0, 0]
    M_total = sum(group_sizes)
    K, N = 64, 128

    x = torch.randn(M_total, K, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_groups, N, K, device=device, dtype=dtype, requires_grad=True)
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    output = F.grouped_mm(x, weight.transpose(1, 2), offs=grouped_mm_offs)
    assert output.shape == (M_total, N), f"Shape mismatch: {output.shape}"

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    assert x.grad.shape == x.shape, "Input grad shape mismatch"
    assert weight.grad.shape == weight.shape, "Weight grad shape mismatch"

    num_groups = 8
    group_sizes = [1, 1, 1, 1, 1, 1, 1, 1]
    M_total = sum(group_sizes)
    K, N = 32, 64

    x = torch.randn(M_total, K, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_groups, N, K, device=device, dtype=dtype, requires_grad=True)
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    output = F.grouped_mm(x, weight.transpose(1, 2), offs=grouped_mm_offs)
    output_ref = reference_grouped_linear_forward(x.detach(), weight.detach(), grouped_mm_offs)

    max_diff = (output - output_ref).abs().max().item()
    assert torch.allclose(output, output_ref, rtol=1e-5, atol=1e-6), (
        f"Single token per group mismatch! Max diff: {max_diff}"
    )

    num_groups = 1
    group_sizes = [1000]
    M_total = sum(group_sizes)
    K, N = 128, 256

    x = torch.randn(M_total, K, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_groups, N, K, device=device, dtype=dtype, requires_grad=True)
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    output = F.grouped_mm(x, weight.transpose(1, 2), offs=grouped_mm_offs)
    output_ref = reference_grouped_linear_forward(x.detach(), weight.detach(), grouped_mm_offs)

    max_diff = (output - output_ref).abs().max().item()
    assert torch.allclose(output, output_ref, rtol=1e-5, atol=1e-6), (
        f"Large group mismatch! Max diff: {max_diff}"
    )


def verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out, reverse_idxs, offs):
    recovered = out[reverse_idxs]
    assert torch.equal(recovered, sorted_tokens), (
        f"out[reverse_idxs] != sorted_tokens, max diff: "
        f"{(recovered - sorted_tokens).abs().max().item()}"
    )

    group_starts = [0] + offs.tolist()
    for group_idx in range(num_groups):
        start = group_starts[group_idx]
        end = group_starts[group_idx + 1]
        mask = expert_idxs == group_idx
        expected_rows = sorted_tokens[mask]
        group_block = out[start:end]
        num_tokens = expected_rows.shape[0]
        actual_positions = reverse_idxs[mask]
        actual_rows = out[actual_positions]
        if num_tokens > 0:
            assert torch.equal(actual_rows, expected_rows), f"Group {group_idx}: token content mismatch"  # fmt: skip
        if end - start > num_tokens:
            used_positions = set(actual_positions.tolist())
            for pos in range(start, end):
                if pos not in used_positions:
                    assert torch.all(group_block[pos - start] == 0), (
                        f"Group {group_idx}: non-zero padding at position {pos}"
                    )


def test_scatter_for_grouped_gemm():
    device = torch.device("cuda")
    dtype = torch.float32

    configs = [
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

        out_new, reverse_new, offs_new, ks_new, ks_tensor_new = scatter_for_grouped_gemm(
            sorted_tokens, expert_idxs, num_groups, padding_alignment
        )

        assert offs_new.shape == (num_groups,), f"offs shape {offs_new.shape} != ({num_groups},)"
        assert offs_new.dtype == torch.int32

        counts = torch.bincount(expert_idxs, minlength=num_groups).tolist()
        expected_ks = [
            (count + padding_alignment - 1) // padding_alignment * padding_alignment
            for count in counts
        ]
        expected_ks_tensor = torch.tensor(expected_ks, device=device, dtype=torch.int32)
        expected_offs = expected_ks_tensor.cumsum(0, dtype=torch.int32)
        assert ks_new == expected_ks, f"{test_name}: incorrect per-expert sizes"
        assert torch.equal(ks_tensor_new, expected_ks_tensor), (
            f"{test_name}: ks_tensor (per-expert padded sizes) mismatch: "
            f"actual={ks_tensor_new.tolist()}, expected={expected_ks}"
        )
        assert torch.equal(offs_new, expected_offs), (
            f"{test_name}: grouped_mm_offs (cumulative padded offsets) mismatch: "
            f"actual={offs_new.tolist()}, expected={expected_offs.tolist()}"
        )
        m_padded = sum(expected_ks)
        if out_new.shape[0] > m_padded:
            tail = out_new[m_padded:]
            assert torch.all(tail == 0), (
                f"{test_name}: over-allocated tail [{m_padded}:{out_new.shape[0]}) must be zero"
            )
        expected_shape = (m_padded, hidden_size)
        assert out_new.shape == expected_shape, (
            f"{test_name}: output shape {out_new.shape} != {expected_shape}"
        )

        assert reverse_new.shape == (m,)
        assert reverse_new.dtype == torch.int64

        verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)  # fmt: skip


def test_scatter_for_grouped_gemm_edge_cases():
    device = torch.device("cuda")
    dtype = torch.float32

    sorted_tokens = torch.randn(0, 128, device=device, dtype=dtype)
    expert_idxs = torch.empty(0, device=device, dtype=torch.int64)
    out, reverse, offs, _, _ = scatter_for_grouped_gemm(sorted_tokens, expert_idxs, 8)
    assert out.shape == (0, 128), f"Expected (0, 128), got {out.shape}"
    assert reverse.shape == (0,), f"Expected (0,), got {reverse.shape}"
    assert offs.shape == (0,), f"Expected (0,), got {offs.shape}"

    m, hidden_size, num_groups = 32, 256, 8
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
    expert_idxs = torch.full((m,), 3, device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)
    group_starts = [0] + offs_new.tolist()
    for group_idx in range(num_groups):
        if group_idx != 3:
            block = out_new[group_starts[group_idx] : group_starts[group_idx + 1]]
            assert torch.all(block == 0), f"Group {group_idx} should be all zeros"

    # hidden_size not a multiple of the 256-column block.
    m, hidden_size, num_groups = 16, 100, 4
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
    expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)

    m, hidden_size, num_groups = 8, 1, 2
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
    expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)

    m, hidden_size, num_groups = 24, 256, 4
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=torch.bfloat16)
    expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)

    # hidden_size spanning several 256-column blocks.
    m, hidden_size, num_groups = 16, 768, 4
    sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
    expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)

    out_new, reverse_new, offs_new, _, _ = scatter_for_grouped_gemm(
        sorted_tokens, expert_idxs, num_groups
    )
    verify_scatter_result(sorted_tokens, expert_idxs, num_groups, out_new, reverse_new, offs_new)


def test_grouped_mm_over_allocated_input():
    """
    With input rows past offs[-1], F.grouped_mm reads only up to offs[-1].
    """
    device = torch.device("cuda")
    dtype = torch.float32

    configs = [
        ("Small over-alloc", 4, [5, 3, 2, 4], 50, 128, 256),
        ("Large over-alloc", 8, [10, 5, 8, 3, 7, 2, 6, 4], 200, 128, 256),
        ("Some empty groups", 8, [5, 0, 3, 0, 0, 2, 0, 1], 100, 128, 256),
        ("Single group", 1, [10], 20, 64, 128),
    ]

    for test_name, num_groups, group_sizes, extra_rows, K, N in configs:
        M_exact = sum(group_sizes)
        M_over = M_exact + extra_rows

        input_exact = torch.randn(M_exact, K, device=device, dtype=dtype)
        input_over = torch.zeros(M_over, K, device=device, dtype=dtype)
        input_over[:M_exact] = input_exact

        weight = torch.randn(num_groups, N, K, device=device, dtype=dtype)
        grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

        output_ref = F.grouped_mm(input_exact, weight.transpose(1, 2), offs=grouped_mm_offs)

        output_over = F.grouped_mm(input_over, weight.transpose(1, 2), offs=grouped_mm_offs)

        # Rows past M_exact are undefined (grouped_mm may write NaN there).
        output_over_trimmed = output_over[:M_exact]
        max_diff = (output_over_trimmed - output_ref).abs().max().item()
        assert torch.allclose(output_over_trimmed, output_ref, rtol=1e-5, atol=1e-6), (
            f"Over-allocated grouped_mm mismatch for '{test_name}'! Max diff: {max_diff}"
        )


def test_scatter_then_grouped_mm_end_to_end():
    device = torch.device("cuda")
    dtype = torch.float32

    configs = [
        ("Basic", 64, 128, 8, 256),
        ("Large", 256, 256, 8, 512),
        ("Many experts", 128, 128, 32, 256),
    ]

    for test_name, m, hidden_size, num_groups, out_features in configs:
        sorted_tokens = torch.randn(m, hidden_size, device=device, dtype=dtype)
        expert_idxs = torch.randint(0, num_groups, (m,), device=device, dtype=torch.int64)
        weight = torch.randn(num_groups, out_features, hidden_size, device=device, dtype=dtype)

        out_tokens, reverse_idxs, offs, _, _ = scatter_for_grouped_gemm(
            sorted_tokens, expert_idxs, num_groups
        )

        mm_result = F.grouped_mm(out_tokens, weight.transpose(1, 2), offs=offs)

        result = mm_result[reverse_idxs]

        result_ref = torch.zeros(m, out_features, device=device, dtype=dtype)
        for group_idx in range(num_groups):
            mask = expert_idxs == group_idx
            if mask.any():
                result_ref[mask] = sorted_tokens[mask] @ weight[group_idx].T

        max_diff = (result - result_ref).abs().max().item()
        assert torch.allclose(result, result_ref, rtol=1e-4, atol=1e-4), (
            f"End-to-end mismatch for '{test_name}'! Max diff: {max_diff}"
        )


def test_grouped_linear_weight_grad_store():
    from pithtrain.contexts.weight_grad_store import WeightGradStore

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_groups, in_features, out_features = 4, 128, 256
    group_sizes = [16, 8, 12, 4]
    M_total = sum(group_sizes)
    grouped_mm_offs = torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

    layer_ref = GroupedLinear(num_groups, in_features, out_features).to(device).to(dtype)
    torch.nn.init.normal_(layer_ref.weight, std=0.02)

    x_raw = torch.randn(M_total, in_features, device=device, dtype=dtype)
    grad = torch.randn(M_total, out_features, device=device, dtype=dtype)

    x_ref = x_raw.detach().clone().requires_grad_(True)
    layer_ref(x_ref, grouped_mm_offs).backward(grad)
    ref_weight_grad = layer_ref.weight.grad.clone()
    ref_input_grad = x_ref.grad.clone()

    layer = GroupedLinear(num_groups, in_features, out_features).to(device).to(dtype)
    layer.weight.data.copy_(layer_ref.weight.data)
    x_def = x_raw.detach().clone().requires_grad_(True)

    WeightGradStore.enabled = True
    try:
        layer(x_def, grouped_mm_offs).backward(grad)

        assert layer.weight.grad is None, "Weight grad should be deferred"
        assert x_def.grad is not None, "Input grad should be on the critical path"

        input_diff = (x_def.grad - ref_input_grad).abs().max().item()
        assert torch.allclose(x_def.grad, ref_input_grad, rtol=1e-3, atol=1e-3), (
            f"Input grad diff = {input_diff}"
        )

        WeightGradStore.flush()
        WeightGradStore.pop()

        assert layer.weight.grad is not None, "Weight grad should exist after pop"
        assert layer.weight.grad.shape == layer.weight.shape
        weight_diff = (layer.weight.grad - ref_weight_grad).abs().max().item()
        assert torch.allclose(layer.weight.grad, ref_weight_grad, rtol=1e-3, atol=1e-3), (
            f"Deferred vs direct weight grad diff = {weight_diff}"
        )
    finally:
        WeightGradStore.enabled = False
        WeightGradStore.clear()


def test_gpt_oss_experts_weight_grad_store_matches_direct(monkeypatch):
    from pithtrain.contexts.weight_grad_store import WeightGradStore
    from pithtrain.models.gpt_oss import GptOssExperts

    # This standalone BF16 test does not call setup_training; monkeypatch restores the previous
    # context value (or its absence) afterwards.
    monkeypatch.setattr(training, "fp8", False, raising=False)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_experts = 4
    hidden_size = 128
    intermediate_size = 256
    swiglu_limit = 7.0
    config = SimpleNamespace(
        hidden_size=hidden_size, intermediate_size=intermediate_size, swiglu_limit=swiglu_limit
    )

    # One empty group exercises the zero-token branch.
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

    x_ref = x_raw.detach().clone().requires_grad_(True)
    out_ref = experts_ref(x_ref, offs, ks=ks)
    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)

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
        for param_def, name in [
            (experts_def.gate_up_proj, "gate_up_proj"),
            (experts_def.down_proj, "down_proj"),
        ]:
            assert param_def.grad is None, f"[deferred] {name} grad should be deferred"

        # Biases are added outside GroupedLinearFunc; autograd populates them eagerly.
        for param_def, name in [
            (experts_def.gate_up_proj_bias, "gate_up_proj_bias"),
            (experts_def.down_proj_bias, "down_proj_bias"),
        ]:
            assert param_def.grad is not None, f"[deferred] {name} grad should be eager"

        in_diff = (x_def.grad - x_ref.grad).abs().max().item()
        assert torch.allclose(x_def.grad, x_ref.grad, rtol=1e-3, atol=1e-3), (
            f"input grad mismatch, max diff {in_diff}"
        )

        # Both paths compute the bias gradients with indexed_bias_add's segment sum; the 5% slack
        # covers bf16 rounding only.
        for param_ref, param_def, name in [
            (experts_ref.gate_up_proj_bias, experts_def.gate_up_proj_bias, "gate_up_proj_bias"),
            (experts_ref.down_proj_bias, experts_def.down_proj_bias, "down_proj_bias"),
        ]:
            ref_max = param_ref.grad.abs().max().item()
            diff = (param_def.grad - param_ref.grad).abs().max().item()
            assert diff <= max(0.1, ref_max * 0.05), (
                f"{name} grad mismatch, max diff {diff} (max |ref| {ref_max})"
            )

        WeightGradStore.flush()
        WeightGradStore.pop()

        # grouped_mm is deterministic, so weight gradients must match tightly.
        for param_ref, param_def, name in [
            (experts_ref.gate_up_proj, experts_def.gate_up_proj, "gate_up_proj"),
            (experts_ref.down_proj, experts_def.down_proj, "down_proj"),
        ]:
            assert param_def.grad is not None, f"[deferred] {name} grad missing after pop"
            diff = (param_def.grad - param_ref.grad).abs().max().item()
            assert torch.allclose(param_def.grad, param_ref.grad, rtol=1e-3, atol=1e-3), (
                f"{name} grad mismatch (deferred vs direct), max diff {diff}"
            )
    finally:
        WeightGradStore.enabled = False
        WeightGradStore.clear()
