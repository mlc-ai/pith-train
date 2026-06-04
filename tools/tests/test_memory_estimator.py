"""
Static tests for the memory estimator tool.

CPU-only analytical tests - no GPU or distributed setup needed.
Run from repo root: pytest tools/tests/test_memory_estimator.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Shared helpers to build configs used across multiple tests
# ---------------------------------------------------------------------------


def _qwen3_model_cfg():
    from tools.memory_estimator.model_profile import ModelConfig

    return ModelConfig.from_json(
        REPO_ROOT / "examples" / "pretrain_lm" / "qwen3-30b-a3b" / "config.json"
    )


def _qwen3_parallel_cfg():
    from tools.memory_estimator.model_profile import ParallelismConfig

    return ParallelismConfig(
        pp_size=4,
        ep_size=8,
        dp_size=1,
        cp_size=1,
        micro_batch_size=1,
        global_batch_size=1024,
        sequence_length=4096,
    )


# ---------------------------------------------------------------------------
# Test 1: Token counts computation
# ---------------------------------------------------------------------------


class TestTokenCounts:
    def test_qwen3_token_counts(self):
        from tools.memory_estimator.activation_profile import compute_token_counts

        model_cfg = _qwen3_model_cfg()
        parallel_cfg = _qwen3_parallel_cfg()
        tc = compute_token_counts(model_cfg, parallel_cfg)

        assert tc.m == 4096, f"Expected m=4096, got {tc.m}"
        assert tc.k == 8, f"Expected k=8, got {tc.k}"
        assert tc.m_times_k == 32768, f"Expected m_times_k=32768, got {tc.m_times_k}"

        # Analytical: ceil(4096 * (1 - (7/8)^8))
        expected_m_dedup = math.ceil(4096 * (1 - (7 / 8) ** 8))
        assert expected_m_dedup == 2689
        assert tc.m_dedup == 2689, f"Expected m_dedup=2689, got {tc.m_dedup}"

        # m_recv = m * k / ep_size = 4096 * 8 / 8 = 4096
        assert tc.m_recv == 4096, f"Expected m_recv=4096, got {tc.m_recv}"

        # m_sorted = m_dedup * ep_size = 2689 * 8
        assert tc.m_sorted == 2689 * 8, f"Expected m_sorted={2689 * 8}, got {tc.m_sorted}"
        assert tc.m_sorted == 21512

        # m_expanded = ceil(32768 * 1.2) = 39322
        expected_m_expanded = math.ceil(32768 * 1.2)
        assert expected_m_expanded == 39322
        assert tc.m_expanded == 39322, f"Expected m_expanded=39322, got {tc.m_expanded}"


# ---------------------------------------------------------------------------
# Test 2: Static memory (model parameters and optimizer states)
# ---------------------------------------------------------------------------


class TestStaticMemory:
    def test_qwen3_module_params(self):
        from tools.memory_estimator.model_profile import ModelMemoryProfile

        model_cfg = _qwen3_model_cfg()
        parallel_cfg = _qwen3_parallel_cfg()
        profile = ModelMemoryProfile(model_cfg, parallel_cfg, pp_rank=0)

        mod0_params = profile.compute_module_params(0)
        mod1_params = profile.compute_module_params(1)

        # Module 0 and 1 should each be approximately 1.66 GB (1,779,302,400 bytes)
        expected_bytes = 1_779_302_400
        assert mod0_params.total_bytes == pytest.approx(expected_bytes, rel=0.01), (
            f"module[0] params: expected ~{expected_bytes}, got {mod0_params.total_bytes}"
        )
        assert mod1_params.total_bytes == pytest.approx(expected_bytes, rel=0.01), (
            f"module[1] params: expected ~{expected_bytes}, got {mod1_params.total_bytes}"
        )

    def test_qwen3_optimizer_states(self):
        from tools.memory_estimator.model_profile import ModelMemoryProfile

        model_cfg = _qwen3_model_cfg()
        parallel_cfg = _qwen3_parallel_cfg()
        profile = ModelMemoryProfile(model_cfg, parallel_cfg, pp_rank=0)

        optimizer = profile.compute_optimizer_states()

        # Optimizer states approximately 7.54 GB (8,095,207,424 bytes)
        expected_bytes = 8_095_207_424
        assert optimizer.total_bytes == pytest.approx(expected_bytes, rel=0.01), (
            f"optimizer states: expected ~{expected_bytes}, got {optimizer.total_bytes}"
        )


# ---------------------------------------------------------------------------
# Test 3: Per-chunk activation memory
# ---------------------------------------------------------------------------


class TestChunkActivations:
    def test_qwen3_chunk_total(self):
        from tools.memory_estimator.activation_profile import (
            ActivationEstimator,
            compute_token_counts,
        )
        from tools.memory_estimator.model_profile import ModelMemoryProfile

        model_cfg = _qwen3_model_cfg()
        parallel_cfg = _qwen3_parallel_cfg()
        profile = ModelMemoryProfile(model_cfg, parallel_cfg, pp_rank=0)
        tc = compute_token_counts(model_cfg, parallel_cfg)
        act_est = ActivationEstimator(model_cfg, parallel_cfg, profile, tc)

        records, autograd = act_est.compute_chunk_activations(0, 0, 0)
        total = records.total_bytes + autograd.total_bytes

        # Records + autograd for one chunk should be approximately 4.04 GB (4,334,xxx,xxx bytes)
        expected_bytes = 4_334_000_000
        assert total == pytest.approx(expected_bytes, rel=0.05), (
            f"chunk activation total: expected ~{expected_bytes}, got {total}"
        )

    def test_qwen3_scattered_tokens_dimension(self):
        from tools.memory_estimator.activation_profile import (
            ActivationEstimator,
            compute_token_counts,
        )
        from tools.memory_estimator.model_profile import ModelMemoryProfile

        model_cfg = _qwen3_model_cfg()
        parallel_cfg = _qwen3_parallel_cfg()
        profile = ModelMemoryProfile(model_cfg, parallel_cfg, pp_rank=0)
        tc = compute_token_counts(model_cfg, parallel_cfg)
        act_est = ActivationEstimator(model_cfg, parallel_cfg, profile, tc)

        _, autograd = act_est.compute_chunk_activations(0, 0, 0)

        # Find the scattered_tokens spec in autograd
        # m_expanded = 39322, experts_per_rank = 128/8 = 16, scatter_padding = 16*127 = 2032
        # m_scattered = 39322 + 2032 = 41354
        expected_m_scattered = 39322 + 16 * 127
        assert expected_m_scattered == 41354

        scattered_specs = [s for s in autograd.specs if "scattered_tokens" in s.name]
        assert len(scattered_specs) > 0, "No scattered_tokens spec found in autograd"

        # Check the first dimension of the scattered_tokens shape
        for spec in scattered_specs:
            assert spec.shape[0] == expected_m_scattered, (
                f"scattered_tokens shape[0]: expected {expected_m_scattered}, got {spec.shape[0]}"
            )


# ---------------------------------------------------------------------------
# Test 4: Full schedule simulation
# ---------------------------------------------------------------------------


class TestScheduleSimulation:
    def test_qwen3_simulation(self):
        from tools.memory_estimator.__main__ import estimate_non_pytorch_bytes
        from tools.memory_estimator.activation_profile import (
            ActivationEstimator,
            compute_token_counts,
        )
        from tools.memory_estimator.model_profile import ModelMemoryProfile
        from tools.memory_estimator.schedule_simulator import ScheduleSimulator

        model_cfg = _qwen3_model_cfg()
        parallel_cfg = _qwen3_parallel_cfg()
        pp_rank = 0
        profile = ModelMemoryProfile(model_cfg, parallel_cfg, pp_rank=pp_rank)
        tc = compute_token_counts(model_cfg, parallel_cfg)
        act_est = ActivationEstimator(model_cfg, parallel_cfg, profile, tc)
        non_pytorch = estimate_non_pytorch_bytes(parallel_cfg)

        simulator = ScheduleSimulator(
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            profile=profile,
            activation_est=act_est,
            token_counts=tc,
            pp_rank=pp_rank,
            fragmentation_factor=0.10,
            non_pytorch_bytes=non_pytorch,
        )
        result = simulator.simulate()

        # Peak event name should contain "S4" (Step 4 is steady state)
        peak_event = result.timeline[result.peak_event_idx]
        assert "S4" in peak_event.name, f"Expected peak event in S4, got: {peak_event.name}"

        # Peak total bytes approximately 66.64 GB (71,546,xxx,xxx bytes)
        expected_peak = 71_546_000_000
        assert result.peak_bytes == pytest.approx(expected_peak, rel=0.05), (
            f"peak_bytes: expected ~{expected_peak}, got {result.peak_bytes}"
        )

        # Static bytes approximately 11.06 GB (11,870,712,064 bytes)
        expected_static = 11_870_712_064
        assert result.static_bytes == pytest.approx(expected_static, rel=0.05), (
            f"static_bytes: expected ~{expected_static}, got {result.static_bytes}"
        )

        # Live chunks at peak: phase0=8, phase1=1
        peak_snapshot = peak_event.snapshot
        p0_live, p1_live = peak_snapshot.live_chunk_counts
        assert p0_live == 8, f"Expected 8 phase0 live chunks at peak, got {p0_live}"
        assert p1_live == 1, f"Expected 1 phase1 live chunk at peak, got {p1_live}"


# ---------------------------------------------------------------------------
# Test 5: DeepSeek-V2-Lite smoke test
# ---------------------------------------------------------------------------


class TestDeepSeekV2Lite:
    def test_smoke(self):
        """
        Verify the estimator runs end-to-end on DeepSeek-V2-Lite without errors.

        Known limitation: DeepSeek-V2-Lite config uses ``n_routed_experts``
        instead of ``num_experts``. The current ``ModelConfig.from_json`` only
        reads ``num_experts``, so ``n_routed_experts`` is silently ignored and
        ``num_experts`` defaults to 0 (treated as a dense model). If
        ``from_json`` is updated to handle ``n_routed_experts``, this test
        should still pass since we only assert ``peak_bytes > 0``.
        """
        from tools.memory_estimator.__main__ import estimate_non_pytorch_bytes
        from tools.memory_estimator.activation_profile import (
            ActivationEstimator,
            compute_token_counts,
        )
        from tools.memory_estimator.model_profile import (
            ModelConfig,
            ModelMemoryProfile,
            ParallelismConfig,
        )
        from tools.memory_estimator.schedule_simulator import ScheduleSimulator

        config_path = REPO_ROOT / "examples" / "pretrain_lm" / "deepseek-v2-lite" / "config.json"
        model_cfg = ModelConfig.from_json(config_path)

        parallel_cfg = ParallelismConfig(
            pp_size=2,
            ep_size=2,
            dp_size=1,
            cp_size=1,
            micro_batch_size=1,
            global_batch_size=1024,
            sequence_length=2048,
        )

        pp_rank = 0
        profile = ModelMemoryProfile(model_cfg, parallel_cfg, pp_rank=pp_rank)
        tc = compute_token_counts(model_cfg, parallel_cfg)
        act_est = ActivationEstimator(model_cfg, parallel_cfg, profile, tc)
        non_pytorch = estimate_non_pytorch_bytes(parallel_cfg)

        simulator = ScheduleSimulator(
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            profile=profile,
            activation_est=act_est,
            token_counts=tc,
            pp_rank=pp_rank,
            fragmentation_factor=0.10,
            non_pytorch_bytes=non_pytorch,
        )
        result = simulator.simulate()

        assert result.peak_bytes > 0, "peak_bytes should be positive"
        assert len(result.timeline) > 0, "timeline should not be empty"
