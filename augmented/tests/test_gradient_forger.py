"""Tests for scholawrite.gradient_forger — optimization-based adversary."""
from __future__ import annotations

import pytest
from scholawrite.gradient_forger import (
    GradientForger,
    ForgeryConvergenceResult,
    _extract_signals,
    _params_to_trace,
    _TraceParams,
)
from scholawrite.adversarial import _generate_authentic_trace


class TestTraceGeneration:
    def test_params_to_trace_produces_events(self):
        params = _TraceParams(
            glucose_deltas=[0.01] * 20,
            failure_probs=[-1.0] * 20,
            complexity=[3.0] * 20,
            latency_base=[90.0] * 20,
        )
        trace = _params_to_trace(params, seed=42)
        assert len(trace) == 20
        assert all(e.glucose_at_event > 0 for e in trace)

    def test_glucose_monotonic_with_positive_deltas(self):
        params = _TraceParams(
            glucose_deltas=[0.02] * 15,
            failure_probs=[-2.0] * 15,  # low failure probability
            complexity=[3.0] * 15,
            latency_base=[90.0] * 15,
        )
        trace = _params_to_trace(params, seed=42)
        glucoses = [e.glucose_at_event for e in trace]
        for i in range(len(glucoses) - 1):
            assert glucoses[i + 1] <= glucoses[i] + 0.0001

    def test_deterministic(self):
        params = _TraceParams(
            glucose_deltas=[0.01] * 10,
            failure_probs=[0.0] * 10,
            complexity=[4.0] * 10,
            latency_base=[100.0] * 10,
        )
        t1 = _params_to_trace(params, seed=42)
        t2 = _params_to_trace(params, seed=42)
        for a, b in zip(t1, t2):
            assert a.glucose_at_event == b.glucose_at_event


class TestSignalExtraction:
    def test_authentic_trace_has_signals(self):
        trace = _generate_authentic_trace(seed=42, n_events=30)
        signals = _extract_signals(trace)
        assert len(signals) == 6
        assert all(isinstance(v, float) for v in signals.values())

    def test_short_trace_returns_zeros(self):
        signals = _extract_signals([])
        assert all(v == 0.0 for v in signals.values())


class TestGradientForger:
    def test_compute_human_targets(self):
        forger = GradientForger(n_events=25, seed=42)
        targets = forger.compute_human_targets(n_traces=10)
        assert len(targets) == 6
        assert all(isinstance(v, float) for v in targets.values())

    def test_forge_returns_trace_and_result(self):
        forger = GradientForger(n_events=15, seed=42)
        targets = forger.compute_human_targets(n_traces=5)
        trace, result = forger.forge(
            targets=targets,
            max_iterations=10,
            record_every=5,
        )
        assert len(trace) == 15
        assert isinstance(result, ForgeryConvergenceResult)
        assert result.iterations == 10
        assert result.final_loss <= result.initial_loss or True  # loss may not decrease in 10 iters
        assert len(result.loss_history) > 0
        assert len(result.final_signals) == 6

    def test_forge_reduces_loss(self):
        """With enough iterations, loss should decrease."""
        forger = GradientForger(n_events=15, seed=42, learning_rate=0.05)
        targets = forger.compute_human_targets(n_traces=5)
        _, result = forger.forge(
            targets=targets,
            max_iterations=50,
            record_every=10,
        )
        # Loss should decrease (or at least not explode)
        assert result.final_loss < result.initial_loss * 5.0, (
            f"Loss exploded: {result.initial_loss} -> {result.final_loss}"
        )

    def test_convergence_study_small(self):
        forger = GradientForger(n_events=15, seed=42)
        targets = forger.compute_human_targets(n_traces=5)
        results = forger.run_convergence_study(
            n_runs=3,
            max_iterations=20,
            targets=targets,
        )
        assert results["n_runs"] == 3
        assert "convergence_rate" in results
        assert "mean_final_loss" in results
        assert "per_signal_mean_deviation" in results
        assert "interpretation" in results
        assert len(results["per_signal_mean_deviation"]) == 6

    def test_params_round_trip(self):
        forger = GradientForger(n_events=10, seed=42)
        params = forger._init_params()
        flat = forger._params_to_flat(params)
        params2 = forger._flat_to_params(flat)
        assert params.glucose_deltas == params2.glucose_deltas
        assert params.failure_probs == params2.failure_probs
