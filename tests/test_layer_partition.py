"""Tests for the DualPipeV layer partitioning algorithm."""

import pytest

from pithtrain.dualpipe.layer_partition import layer_partition


@pytest.fixture(autouse=True)
def _silence(monkeypatch):
    """Suppress the rank-0 log print during tests."""
    monkeypatch.setattr("pithtrain.dualpipe.layer_partition.print_msg", lambda *a, **kw: None)


# -- Invariant tests (sweep over many configs) --


@pytest.mark.parametrize("S", range(1, 13), ids=lambda s: f"S={s}")
def test_sum_length_and_positivity(S):
    """sum == num_layers, len == num_stages, every stage >= 1."""
    for N in range(S, S + 30):
        layers = layer_partition(N, S)
        assert len(layers) == S
        assert sum(layers) == N, f"N={N}, S={S}: sum={sum(layers)}"
        assert all(n >= 1 for n in layers), f"N={N}, S={S}: {layers}"


@pytest.mark.parametrize("S", range(1, 13), ids=lambda s: f"S={s}")
def test_max_minus_min_at_most_one(S):
    """Every stage gets either floor or ceil of the average."""
    for N in range(S, S + 30):
        layers = layer_partition(N, S)
        assert max(layers) - min(layers) <= 1, f"N={N}, S={S}: {layers}"


@pytest.mark.parametrize("S", range(3, 13), ids=lambda s: f"S={s}")
def test_edges_le_inner(S):
    """Edge stages (0 and -1) are <= every inner stage."""
    for N in range(S, S + 30):
        layers = layer_partition(N, S)
        inner_min = min(layers[1:-1])
        assert layers[0] <= inner_min, f"N={N}, S={S}: {layers}"
        assert layers[-1] <= inner_min, f"N={N}, S={S}: {layers}"


@pytest.mark.parametrize("S", range(2, 13, 2), ids=lambda s: f"S={s}")
def test_dualpipev_phase_balance(S):
    """abs(phase0_layers - phase1_layers) <= 1 for each pp_rank."""
    pp_size = S // 2
    for N in range(S, S + 30):
        layers = layer_partition(N, S)
        for k in range(pp_size):
            diff = abs(layers[k] - layers[S - 1 - k])
            assert diff <= 1, (
                f"N={N}, S={S}, pp_rank={k}: "
                f"{layers[k]} vs {layers[S - 1 - k]} (diff={diff}), {layers}"
            )


# -- Exact-value tests for production configs --


def test_qwen3_30b_a3b_pp4():
    """Qwen3-30B-A3B: 48 layers, pp=4 (8 stages)."""
    assert layer_partition(48, 8) == [6, 6, 6, 6, 6, 6, 6, 6]


def test_deepseek_v2_pp4():
    """DeepSeek-V2-Lite: 27 layers, pp=4 (8 stages)."""
    assert layer_partition(27, 8) == [3, 4, 4, 4, 3, 3, 3, 3]


def test_small_cases():
    assert layer_partition(1, 1) == [1]
    assert layer_partition(2, 2) == [1, 1]
    assert layer_partition(3, 2) == [2, 1]
    assert layer_partition(5, 3) == [2, 2, 1]
