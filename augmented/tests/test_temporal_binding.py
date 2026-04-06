"""Tests for scholawrite.temporal_binding module."""
from __future__ import annotations

import random
import pytest
from scholawrite.schema import CausalEvent
from scholawrite.causal_core import LexicalIntention, IrreversibleProcessEngine
from scholawrite.embodied import EmbodiedScholar
from scholawrite.temporal_binding import (
    compute_temporal_binding_index,
    compute_binding_decay_curve,
    compute_decay_exponent,
    compute_cross_channel_binding,
)
from scholawrite.metrics import compute_causal_signatures


def _build_authentic_trace(n_tokens: int = 100) -> list[CausalEvent]:
    """Generate an authentic trace via IrreversibleProcessEngine."""
    author = EmbodiedScholar("test_temporal", initial_glucose=1.0)
    engine = IrreversibleProcessEngine(author)
    for i in range(n_tokens):
        depth = 3.0 + (i % 7) * 0.8
        rarity = 0.2 + (i % 5) * 0.15
        cost = 0.02 + (i % 3) * 0.01
        engine.execute(LexicalIntention(f"word{i}", depth, rarity, cost))
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


def _build_independent_trace(n: int = 100) -> list[CausalEvent]:
    """Build a trace with independent random values — no temporal coherence."""
    rng = random.Random(42)
    events = []
    for i in range(n):
        events.append(CausalEvent(
            intention=f"word{i}",
            actual_output=f"word{i}",
            status="success" if rng.random() > 0.3 else "failure",
            failure_mode=None,
            repair_artifact=None,
            glucose_at_event=rng.random(),
            latency_ms=rng.uniform(50, 500),
            syntactic_complexity=rng.uniform(1, 10),
        ))
    return events


class TestTemporalBindingIndex:
    def test_authentic_trace_positive(self):
        """Authentic trace should have TBI > 0."""
        trace = _build_authentic_trace(100)
        tbi = compute_temporal_binding_index(trace)
        assert tbi > 0.0

    def test_shuffled_trace_lower(self):
        """Shuffled trace should have lower TBI than authentic."""
        trace = _build_authentic_trace(100)
        tbi_authentic = compute_temporal_binding_index(trace)

        shuffled = list(trace)
        rng = random.Random(123)
        rng.shuffle(shuffled)
        tbi_shuffled = compute_temporal_binding_index(shuffled)

        assert tbi_authentic > tbi_shuffled

    def test_independent_trace_low(self):
        """Independent random trace should have much lower TBI than authentic."""
        authentic = _build_authentic_trace(100)
        tbi_authentic = compute_temporal_binding_index(authentic)
        trace = _build_independent_trace(100)
        tbi = compute_temporal_binding_index(trace)
        # Independent trace should have substantially lower TBI than authentic
        assert tbi < tbi_authentic * 0.5

    def test_deterministic(self):
        """Same trace should produce the same TBI."""
        trace = _build_authentic_trace(80)
        tbi1 = compute_temporal_binding_index(trace)
        tbi2 = compute_temporal_binding_index(trace)
        assert tbi1 == tbi2

    def test_empty_trace(self):
        assert compute_temporal_binding_index([]) == 0.0

    def test_single_event(self):
        trace = _build_authentic_trace(1)
        assert compute_temporal_binding_index(trace) == 0.0

    def test_short_trace(self):
        """Trace shorter than max_lag should auto-adjust."""
        trace = _build_authentic_trace(10)
        tbi = compute_temporal_binding_index(trace, max_lag=20)
        assert isinstance(tbi, float)


class TestBindingDecayCurve:
    def test_curve_shape(self):
        """Authentic trace should produce a non-empty decay curve."""
        trace = _build_authentic_trace(100)
        curve = compute_binding_decay_curve(trace)
        assert len(curve) > 0
        # Each entry is (lag, mi)
        for lag, mi in curve:
            assert lag >= 1
            assert mi >= 0.0

    def test_slow_decay_authentic(self):
        """Authentic trace MI should not drop to zero immediately."""
        trace = _build_authentic_trace(100)
        curve = compute_binding_decay_curve(trace)
        # Last MI should still be non-negative
        assert curve[-1][1] >= 0.0

    def test_empty_trace(self):
        assert compute_binding_decay_curve([]) == []


class TestDecayExponent:
    def test_returns_required_keys(self):
        trace = _build_authentic_trace(100)
        result = compute_decay_exponent(trace)
        assert "power_law_exponent" in result
        assert "exponential_rate" in result
        assert "power_law_r2" in result
        assert "exponential_r2" in result
        assert "decay_type" in result
        assert result["decay_type"] in ("power_law", "exponential")

    def test_short_trace_defaults(self):
        trace = _build_authentic_trace(5)
        result = compute_decay_exponent(trace, max_lag=20)
        assert isinstance(result["power_law_exponent"], float)

    def test_empty_trace(self):
        result = compute_decay_exponent([])
        assert result["decay_type"] == "exponential"
        assert result["power_law_exponent"] == 0.0


class TestCrossChannelBinding:
    def test_returns_pairs(self):
        trace = _build_authentic_trace(100)
        result = compute_cross_channel_binding(trace, lag=5)
        assert isinstance(result, dict)
        # 4 channels -> C(4,2) = 6 pairs
        assert len(result) == 6

    def test_short_trace_empty(self):
        trace = _build_authentic_trace(3)
        result = compute_cross_channel_binding(trace, lag=5)
        assert result == {}

    def test_values_non_negative(self):
        trace = _build_authentic_trace(100)
        result = compute_cross_channel_binding(trace, lag=3)
        for v in result.values():
            assert v >= 0.0


class TestMetricsIntegration:
    def test_temporal_binding_in_signatures(self):
        """compute_causal_signatures should include temporal_binding."""
        trace = _build_authentic_trace(100)
        sigs = compute_causal_signatures(trace)
        assert "temporal_binding" in sigs
        assert isinstance(sigs["temporal_binding"], float)
