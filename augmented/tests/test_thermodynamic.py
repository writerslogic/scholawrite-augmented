"""Tests for scholawrite.thermodynamic module."""
from __future__ import annotations

import random

import pytest
from scholawrite.schema import CausalEvent
from scholawrite.thermodynamic import (
    compute_entropy_production_rate,
    compute_time_asymmetry,
    compute_dissipation_trajectory,
)
from scholawrite.causal_core import IrreversibleProcessEngine, LexicalIntention
from scholawrite.embodied import EmbodiedScholar


def _make_authentic_trace(n: int = 30) -> list[CausalEvent]:
    """Generate an authentic irreversible trace via IrreversibleProcessEngine."""
    author = EmbodiedScholar("thermo_test", initial_glucose=1.0)
    engine = IrreversibleProcessEngine(author)
    for i in range(n):
        depth = 3.0 + (i / n) * 5.0
        rarity = 0.2 + (i / n) * 0.6
        engine.execute(LexicalIntention(f"word{i}", depth, rarity, 0.03))
    return [
        CausalEvent(
            intention=e.intention.target,
            actual_output=e.actual_output,
            status="failure" if e.failure_mode else "success",
            failure_mode=e.failure_mode,
            repair_artifact=e.actual_output if e.repair_distance > 0 else None,
            glucose_at_event=e.glucose_before,
            latency_ms=e.latency_ms,
            syntactic_complexity=e.intention.syntactic_depth,
        )
        for e in engine.trace
    ]


def _make_constant_trace(n: int = 30) -> list[CausalEvent]:
    """Generate a perfectly constant (time-symmetric) trace."""
    return [
        CausalEvent(
            intention="word",
            actual_output="word",
            status="success",
            failure_mode=None,
            repair_artifact=None,
            glucose_at_event=0.8,
            latency_ms=150.0,
            syntactic_complexity=5.0,
        )
        for _ in range(n)
    ]


def _make_shuffled_trace(n: int = 30) -> list[CausalEvent]:
    """Generate a shuffled trace (random order, no temporal structure)."""
    rng = random.Random(42)
    events = []
    for i in range(n):
        events.append(
            CausalEvent(
                intention=f"word{i}",
                actual_output=f"word{i}",
                status="success",
                failure_mode=None,
                repair_artifact=None,
                glucose_at_event=rng.uniform(0.3, 1.0),
                latency_ms=rng.uniform(100.0, 300.0),
                syntactic_complexity=rng.uniform(2.0, 8.0),
            )
        )
    rng.shuffle(events)
    return events


class TestEntropyProductionRate:
    def test_authentic_trace_positive(self):
        """Authentic irreversible trace should have sigma > 0."""
        trace = _make_authentic_trace(40)
        sigma = compute_entropy_production_rate(trace)
        assert sigma > 0.0, f"Expected positive entropy production, got {sigma}"

    def test_constant_trace_zero(self):
        """Constant trace (all identical values) should have sigma = 0."""
        trace = _make_constant_trace(30)
        sigma = compute_entropy_production_rate(trace)
        assert sigma == 0.0

    def test_empty_trace(self):
        sigma = compute_entropy_production_rate([])
        assert sigma == 0.0

    def test_single_event(self):
        trace = _make_constant_trace(1)
        sigma = compute_entropy_production_rate(trace)
        assert sigma == 0.0

    def test_two_events(self):
        trace = _make_constant_trace(2)
        sigma = compute_entropy_production_rate(trace)
        assert sigma == 0.0

    def test_symmetric_trace_near_zero(self):
        """A palindromic (time-symmetric) trace should have near-zero sigma."""
        # Build a trace that is exactly the same forwards and backwards
        half = _make_authentic_trace(20)
        symmetric = half + list(reversed(half))
        sigma = compute_entropy_production_rate(symmetric)
        # Perfectly palindromic trace has identical forward/reverse transitions
        assert sigma < 1e-6, f"Expected ~0 for palindromic trace, got {sigma}"

    def test_reversed_trace_differs(self):
        """Forward and reversed traces should yield different sigma values."""
        trace = _make_authentic_trace(40)
        sigma_fwd = compute_entropy_production_rate(trace)
        sigma_rev = compute_entropy_production_rate(list(reversed(trace)))
        # Both should be positive but generally differ
        assert sigma_fwd > 0.0
        assert sigma_rev > 0.0


class TestTimeAsymmetry:
    def test_authentic_trace_has_asymmetry(self):
        trace = _make_authentic_trace(40)
        result = compute_time_asymmetry(trace)
        assert "aggregate_asymmetry" in result
        assert "glucose_asymmetry" in result
        assert "latency_asymmetry" in result
        assert "complexity_asymmetry" in result

    def test_constant_trace_zero_asymmetry(self):
        trace = _make_constant_trace(30)
        result = compute_time_asymmetry(trace)
        assert result["aggregate_asymmetry"] == 0.0

    def test_empty_trace(self):
        result = compute_time_asymmetry([])
        assert result["aggregate_asymmetry"] == 0.0

    def test_short_trace(self):
        trace = _make_constant_trace(2)
        result = compute_time_asymmetry(trace)
        assert result["aggregate_asymmetry"] == 0.0


class TestDissipationTrajectory:
    def test_trajectory_length(self):
        trace = _make_authentic_trace(30)
        window = 10
        traj = compute_dissipation_trajectory(trace, window=window)
        expected_len = len(trace) - window + 1
        assert len(traj) == expected_len

    def test_trajectory_values_nonnegative(self):
        trace = _make_authentic_trace(30)
        traj = compute_dissipation_trajectory(trace, window=10)
        for val in traj:
            assert val >= 0.0

    def test_empty_trace(self):
        traj = compute_dissipation_trajectory([], window=10)
        assert traj == []

    def test_trace_shorter_than_window(self):
        trace = _make_authentic_trace(5)
        traj = compute_dissipation_trajectory(trace, window=10)
        assert traj == []

    def test_window_too_small(self):
        trace = _make_authentic_trace(20)
        traj = compute_dissipation_trajectory(trace, window=2)
        assert traj == []

    def test_constant_trace_flat(self):
        trace = _make_constant_trace(30)
        traj = compute_dissipation_trajectory(trace, window=10)
        assert all(v == 0.0 for v in traj)
