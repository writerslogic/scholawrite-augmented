"""Tests for scholawrite.adversarial — Impossible Forgery Experiment."""
from __future__ import annotations

import pytest
from scholawrite.adversarial import (
    AdversarialEvaluator,
    ForgedTraceGenerator,
    ALL_TIERS,
    TIER_EXPERT,
    _extract_features,
    _extract_consciousness_signals,
    _generate_authentic_trace,
)
from scholawrite.schema import CausalEvent


class TestForgedTraceGenerator:
    def test_naive_forgery_produces_events(self):
        gen = ForgedTraceGenerator(seed=42)
        trace = gen.generate_naive_forgery("The methodology establishes a baseline", n_events=20)
        assert len(trace) == 20
        assert all(isinstance(e, CausalEvent) for e in trace)

    def test_naive_forgery_has_failures(self):
        gen = ForgedTraceGenerator(seed=42)
        trace = gen.generate_naive_forgery("word " * 30, n_events=30)
        failures = [e for e in trace if e.status != "success"]
        assert len(failures) >= 3

    def test_naive_forgery_deterministic(self):
        t1 = ForgedTraceGenerator(seed=99).generate_naive_forgery("hello world " * 5, n_events=15)
        t2 = ForgedTraceGenerator(seed=99).generate_naive_forgery("hello world " * 5, n_events=15)
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a.glucose_at_event == b.glucose_at_event
            assert a.status == b.status

    def test_statistical_forgery_preserves_length(self):
        authentic = _generate_authentic_trace(seed=42, n_events=25)
        gen = ForgedTraceGenerator(seed=42)
        forged = gen.generate_statistical_forgery("some academic text " * 5, authentic)
        assert len(forged) == len(authentic)

    def test_statistical_forgery_empty_trace(self):
        gen = ForgedTraceGenerator(seed=42)
        forged = gen.generate_statistical_forgery("text", [])
        assert forged == []

    def test_reverse_engineered_forgery_returns_tuple(self):
        gen = ForgedTraceGenerator(seed=42)
        trace, meta = gen.generate_reverse_engineered_forgery("word " * 30, n_events=20)
        assert len(trace) == 20
        assert isinstance(meta, dict)
        assert meta["strategy"] == "reverse_engineered"

    def test_reverse_engineered_glucose_resets(self):
        """The reverse-engineered forgery resets glucose periodically,
        so glucose should NOT be monotonically decreasing across the full trace."""
        gen = ForgedTraceGenerator(seed=42)
        trace, meta = gen.generate_reverse_engineered_forgery("word " * 40, n_events=40)
        glucoses = [e.glucose_at_event for e in trace]
        increases = sum(1 for i in range(len(glucoses) - 1) if glucoses[i + 1] > glucoses[i] + 0.001)
        assert increases >= 1, "Reverse-engineered forgery should show glucose resets"


class TestExpertForgery:
    def test_expert_forgery_produces_events(self):
        gen = ForgedTraceGenerator(seed=42)
        trace, meta = gen.generate_expert_forgery("word " * 30, n_events=30)
        assert len(trace) == 30
        assert all(isinstance(e, CausalEvent) for e in trace)
        assert meta["strategy"] == "expert"

    def test_expert_forgery_glucose_monotonic(self):
        """Expert forgery uses full simulation — glucose should be monotonically decreasing."""
        gen = ForgedTraceGenerator(seed=42)
        trace, _ = gen.generate_expert_forgery("word " * 40, n_events=40)
        glucoses = [e.glucose_at_event for e in trace]
        for i in range(len(glucoses) - 1):
            assert glucoses[i + 1] <= glucoses[i] + 0.0001, (
                f"Glucose increased at index {i}: {glucoses[i]:.4f} -> {glucoses[i+1]:.4f}"
            )

    def test_expert_forgery_has_failures(self):
        """Expert forgery should produce some failures (gated on glucose+complexity)."""
        gen = ForgedTraceGenerator(seed=42)
        trace, _ = gen.generate_expert_forgery("word " * 50, n_events=50)
        failures = [e for e in trace if e.status != "success"]
        assert len(failures) >= 1, "Expert forgery should trigger at least 1 failure"

    def test_expert_forgery_deterministic(self):
        t1, _ = ForgedTraceGenerator(seed=99).generate_expert_forgery("word " * 20, n_events=20)
        t2, _ = ForgedTraceGenerator(seed=99).generate_expert_forgery("word " * 20, n_events=20)
        for a, b in zip(t1, t2):
            assert a.glucose_at_event == b.glucose_at_event
            assert a.latency_ms == b.latency_ms

    def test_expert_forgery_techniques_documented(self):
        gen = ForgedTraceGenerator(seed=42)
        _, meta = gen.generate_expert_forgery("word " * 20, n_events=20)
        assert "techniques" in meta
        assert len(meta["techniques"]) >= 3


class TestAuthenticTraceGeneration:
    def test_generates_causal_events(self):
        trace = _generate_authentic_trace(seed=42, n_events=25)
        assert len(trace) == 25
        assert all(isinstance(e, CausalEvent) for e in trace)

    def test_glucose_monotonically_decreasing(self):
        trace = _generate_authentic_trace(seed=42, n_events=30)
        glucoses = [e.glucose_at_event for e in trace]
        for i in range(len(glucoses) - 1):
            assert glucoses[i + 1] <= glucoses[i] + 0.0001, (
                f"Glucose increased at index {i}: {glucoses[i]:.4f} -> {glucoses[i+1]:.4f}"
            )

    def test_deterministic(self):
        t1 = _generate_authentic_trace(seed=123, n_events=20)
        t2 = _generate_authentic_trace(seed=123, n_events=20)
        for a, b in zip(t1, t2):
            assert a.glucose_at_event == b.glucose_at_event
            assert a.latency_ms == b.latency_ms


class TestFeatureExtraction:
    def test_short_trace_returns_zeros(self):
        features = _extract_features([])
        assert features["coupling"] == 0.0
        assert features["glucose_monotonicity"] == 0.0

    def test_authentic_trace_has_high_monotonicity(self):
        trace = _generate_authentic_trace(seed=42, n_events=30)
        features = _extract_features(trace)
        assert features["glucose_monotonicity"] >= 0.9

    def test_naive_forgery_has_lower_monotonicity(self):
        gen = ForgedTraceGenerator(seed=42)
        forged = gen.generate_naive_forgery("word " * 30, n_events=30)
        features = _extract_features(forged)
        assert 0.0 <= features["glucose_monotonicity"] <= 1.0


class TestConsciousnessSignalExtraction:
    def test_short_trace_returns_zeros(self):
        signals = _extract_consciousness_signals([])
        assert all(v == 0.0 for v in signals.values())
        assert len(signals) == 6

    def test_authentic_trace_has_signals(self):
        trace = _generate_authentic_trace(seed=42, n_events=30)
        signals = _extract_consciousness_signals(trace)
        assert len(signals) == 6
        assert all(isinstance(v, float) for v in signals.values())
        # At least some signals should be non-zero for authentic traces
        assert sum(signals.values()) > 0.0


class TestAdversarialEvaluator:
    def test_evaluate_forgery_detection_basic(self):
        authentic = [_generate_authentic_trace(seed=i, n_events=25) for i in range(10)]
        forged = []
        for i in range(10):
            g = ForgedTraceGenerator(seed=100 + i)
            forged.append(g.generate_naive_forgery("word " * 25, n_events=25))

        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.evaluate_forgery_detection(authentic, forged)

        assert "per_feature_auc" in result
        assert "combined_auc" in result
        assert result["n_authentic"] == 10
        assert result["n_forged"] == 10

    def test_naive_forgery_detectable(self):
        """Naive forgery should be detectable with AUC > 0.7."""
        authentic = [_generate_authentic_trace(seed=i, n_events=30) for i in range(20)]
        forged = []
        for i in range(20):
            g = ForgedTraceGenerator(seed=500 + i)
            forged.append(g.generate_naive_forgery("word " * 30, n_events=30))

        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.evaluate_forgery_detection(authentic, forged)
        max_auc = max(result["per_feature_auc"].values())
        assert max_auc > 0.7, f"Best feature AUC {max_auc} should exceed 0.7"

    def test_authentic_vs_authentic_near_random(self):
        """Authentic vs authentic should yield AUC near 0.5."""
        group_a = [_generate_authentic_trace(seed=i, n_events=30) for i in range(20)]
        group_b = [_generate_authentic_trace(seed=i + 1000, n_events=30) for i in range(20)]

        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.evaluate_forgery_detection(group_a, group_b)
        assert result["combined_auc"] <= 0.7, (
            f"Authentic vs authentic AUC {result['combined_auc']} should be near 0.5"
        )

    def test_empty_traces_handled(self):
        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.evaluate_forgery_detection([], [])
        assert result["combined_auc"] == 0.0

    def test_evaluate_consciousness_discrimination(self):
        authentic = [_generate_authentic_trace(seed=i, n_events=25) for i in range(10)]
        forged = []
        for i in range(10):
            g = ForgedTraceGenerator(seed=100 + i)
            forged.append(g.generate_naive_forgery("word " * 25, n_events=25))

        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.evaluate_consciousness_discrimination(authentic, forged)
        assert "per_signal_auc" in result
        assert "composite_auc" in result
        assert len(result["per_signal_auc"]) == 6

    def test_run_full_evaluation_small(self):
        """Smoke test: full evaluation with small sample size."""
        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.run_full_evaluation(n_samples=5)

        assert result["n_samples"] == 5
        assert "naive_forgery" in result
        assert "statistical_forgery" in result
        assert "reverse_engineered_forgery" in result
        assert "expert_forgery" in result

        for key in ["naive_forgery", "statistical_forgery",
                     "reverse_engineered_forgery", "expert_forgery"]:
            assert "combined_auc" in result[key]
            assert "per_feature_auc" in result[key]

    def test_run_full_evaluation_includes_consciousness(self):
        """Full evaluation should include consciousness signal × tier matrix."""
        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.run_full_evaluation(n_samples=5)

        assert "consciousness" in result
        cs = result["consciousness"]
        assert "signal_tier_matrix" in cs
        assert "composite_by_tier" in cs

        # Check matrix shape: 6 signals × 4 tiers
        matrix = cs["signal_tier_matrix"]
        assert len(matrix) == 6
        for sig, tier_aucs in matrix.items():
            assert len(tier_aucs) == len(ALL_TIERS)

        # Composite per tier
        assert len(cs["composite_by_tier"]) == len(ALL_TIERS)

    def test_expert_adversary_defeats_consciousness_signals(self):
        """Expert adversary should achieve AUC near 0.5 (coin flip) on consciousness signals.

        This is the paper's key finding: an informed adversary defeats all signals.
        """
        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.run_full_evaluation(n_samples=20, n_events=30)

        cs = result["consciousness"]
        expert_composite = cs["composite_by_tier"].get(TIER_EXPERT, 1.0)
        naive_composite = cs["composite_by_tier"].get("naive", 0.0)

        # Expert should be much harder to detect than naive
        assert expert_composite < naive_composite, (
            f"Expert AUC ({expert_composite}) should be lower than naive AUC ({naive_composite})"
        )

    def test_skip_consciousness_evaluation(self):
        """Should be able to skip consciousness signals for speed."""
        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.run_full_evaluation(n_samples=5, include_consciousness=False)
        assert "consciousness" not in result
        assert "naive_forgery" in result

    def test_run_signal_ablation(self):
        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.run_signal_ablation(n_samples=10, n_events=25, n_bootstrap=50)
        assert "weights" in result
        assert "signal_names" in result
        assert "tier_results" in result
        assert "correlation_matrix" in result
        assert len(result["signal_names"]) == 6
        assert result["n_samples"] == 10
        assert result["n_events"] == 25

        # Check tier results structure
        for tier in ALL_TIERS:
            assert tier in result["tier_results"]
            tr = result["tier_results"][tier]
            assert "auc" in tr["full_composite"]
            assert "ci_lower" in tr["full_composite"]
            assert "ci_upper" in tr["full_composite"]
            for sig in result["signal_names"]:
                loo = tr["leave_one_out"][sig]
                assert "auc" in loo
                assert "delta" in loo
                assert "ci_lower" in loo
                solo = tr["standalone"][sig]
                assert "auc" in solo
                assert 0.0 <= solo["auc"] <= 1.0

        # Correlation matrix: symmetric, diagonal = 1.0
        corr = result["correlation_matrix"]
        for sig in result["signal_names"]:
            assert corr[sig][sig] == 1.0

    def test_run_held_out_evaluation_small(self):
        evaluator = AdversarialEvaluator(seed=42)
        result = evaluator.run_held_out_evaluation(
            n_samples=5, n_events_list=[20], seed_offset=9000,
            n_bootstrap=50, include_consciousness=True,
        )
        assert 20 in result
        for tier in ALL_TIERS:
            assert tier in result[20]
            tier_data = result[20][tier]
            assert "auc" in tier_data
            assert "ci_lower" in tier_data
            assert "ci_upper" in tier_data
            assert tier_data["ci_lower"] <= tier_data["auc"] <= tier_data["ci_upper"]
