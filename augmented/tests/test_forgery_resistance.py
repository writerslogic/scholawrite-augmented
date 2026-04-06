"""Tests for forgery resistance / sequential process attestation."""
from __future__ import annotations

import pytest

from scholawrite.causal_core import ExecutionEvent, IrreversibleProcessEngine, LexicalIntention
from scholawrite.embodied import EmbodiedScholar
from scholawrite.forgery_resistance import (
    SequentialProcessAttestation,
    compute_forgery_cost,
    verify_trace_authenticity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intention(target: str, depth: float = 2.0, rarity: float = 0.3, cost: float = 0.01) -> LexicalIntention:
    return LexicalIntention(target=target, syntactic_depth=depth, lexical_rarity=rarity, cognitive_cost=cost)


def _generate_authentic_trace(n: int = 20) -> tuple:
    """Run the engine and return (trace, author)."""
    author = EmbodiedScholar(author_id="test-author", initial_glucose=1.0)
    engine = IrreversibleProcessEngine(author)
    words = ["the", "methodology", "demonstrates", "significant", "results",
             "however", "further", "analysis", "reveals", "underlying",
             "patterns", "that", "suggest", "a", "more", "nuanced",
             "interpretation", "of", "these", "findings"]
    for i in range(n):
        word = words[i % len(words)]
        intention = _make_intention(word, depth=1.5 + (i % 5) * 0.5, rarity=0.1 + (i % 3) * 0.15)
        engine.execute(intention)
    return engine.trace, author


def _fabricate_trace_wrong_glucose(n: int = 10) -> list:
    """Manually build a trace with INCREASING glucose (violates monotonicity)."""
    events = []
    for i in range(n):
        events.append(ExecutionEvent(
            intention=_make_intention(f"word{i}"),
            actual_output=f"word{i}",
            failure_mode=None,
            repair_distance=0,
            glucose_before=0.5 + i * 0.03,   # INCREASING -- forgery signal
            glucose_after=0.5 + i * 0.03 - 0.005,
            latency_ms=120.0 - i * 2.0,       # DECREASING latency with rising glucose (consistent direction but wrong glucose)
        ))
    return events


def _fabricate_trace_broken_chain(n: int = 10) -> list:
    """Build a trace where glucose_before[i+1] > glucose_after[i] (chain break)."""
    events = []
    for i in range(n):
        g = 1.0 - i * 0.05
        events.append(ExecutionEvent(
            intention=_make_intention(f"word{i}"),
            actual_output=f"word{i}",
            failure_mode=None,
            repair_distance=0,
            glucose_before=g,
            glucose_after=g - 0.08,           # drops by 0.08
            latency_ms=120.0 + i * 5.0,
        ))
    # glucose_before[i+1] = 1.0 - (i+1)*0.05  but glucose_after[i] = 1.0 - i*0.05 - 0.08
    # so gap = (1.0 - (i+1)*0.05) - (1.0 - i*0.05 - 0.08) = -0.05 + 0.08 = 0.03 > 0
    # This means next_before > current_after -> chain violation
    return events


# ---------------------------------------------------------------------------
# SequentialProcessAttestation tests
# ---------------------------------------------------------------------------

class TestSequentialProcessAttestation:
    def test_authentic_trace_passes_all_checks(self):
        trace, _ = _generate_authentic_trace(20)
        att = SequentialProcessAttestation(trace)
        assert att.verify_monotonic_depletion()
        assert att.verify_temporal_consistency()
        assert att.verify_state_dependency()
        assert att.verify_causal_chain_integrity()
        assert att.attestation_score() == 1.0

    def test_fabricated_increasing_glucose_fails_monotonicity(self):
        trace = _fabricate_trace_wrong_glucose()
        att = SequentialProcessAttestation(trace)
        assert not att.verify_monotonic_depletion()
        assert att.attestation_score() < 1.0

    def test_fabricated_broken_chain_fails_state_dependency(self):
        trace = _fabricate_trace_broken_chain()
        att = SequentialProcessAttestation(trace)
        assert not att.verify_state_dependency()
        assert att.attestation_score() < 1.0

    def test_forged_traces_score_lower_than_authentic(self):
        authentic_trace, _ = _generate_authentic_trace(20)
        forged_glucose = _fabricate_trace_wrong_glucose(20)
        forged_chain = _fabricate_trace_broken_chain(20)

        auth_score = SequentialProcessAttestation(authentic_trace).attestation_score()
        forge1_score = SequentialProcessAttestation(forged_glucose).attestation_score()
        forge2_score = SequentialProcessAttestation(forged_chain).attestation_score()

        assert auth_score > forge1_score
        assert auth_score > forge2_score

    def test_empty_trace_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SequentialProcessAttestation([])

    def test_single_event_trace(self):
        trace, _ = _generate_authentic_trace(1)
        att = SequentialProcessAttestation(trace)
        assert att.attestation_score() == 1.0

    def test_causal_chain_integrity_is_conjunction(self):
        """verify_causal_chain_integrity should fail if any sub-check fails."""
        trace = _fabricate_trace_wrong_glucose()
        att = SequentialProcessAttestation(trace)
        # Monotonicity fails -> causal chain integrity must also fail
        assert not att.verify_monotonic_depletion()
        assert not att.verify_causal_chain_integrity()


# ---------------------------------------------------------------------------
# compute_forgery_cost tests
# ---------------------------------------------------------------------------

class TestComputeForgeryCost:
    def test_basic_output_structure(self):
        result = compute_forgery_cost(10)
        assert "sequential_steps" in result
        assert "state_space" in result
        assert "parallel_speedup" in result

    def test_sequential_steps_equals_trace_length(self):
        for n in [1, 5, 50, 100]:
            assert compute_forgery_cost(n)["sequential_steps"] == n

    def test_parallel_speedup_is_one(self):
        result = compute_forgery_cost(10)
        assert result["parallel_speedup"] == 1.0

    def test_state_space_grows_exponentially(self):
        small = compute_forgery_cost(5, glucose_precision=3)["state_space"]
        large = compute_forgery_cost(10, glucose_precision=3)["state_space"]
        assert large > small

    def test_invalid_trace_length(self):
        with pytest.raises(ValueError):
            compute_forgery_cost(0)

    def test_precision_affects_state_space(self):
        low = compute_forgery_cost(5, glucose_precision=3)["state_space"]
        high = compute_forgery_cost(5, glucose_precision=6)["state_space"]
        assert high > low


# ---------------------------------------------------------------------------
# verify_trace_authenticity tests
# ---------------------------------------------------------------------------

class TestVerifyTraceAuthenticity:
    def test_authentic_trace_verifies(self):
        trace, author = _generate_authentic_trace(15)
        result = verify_trace_authenticity(trace, author)
        assert result["authentic"] is True
        assert result["discrepancies"] == []
        assert result["attestation_score"] == 1.0

    def test_fabricated_trace_has_discrepancies(self):
        """A manually fabricated trace should diverge from engine replay."""
        author = EmbodiedScholar(author_id="test-author", initial_glucose=1.0)
        forged = _fabricate_trace_wrong_glucose(5)
        result = verify_trace_authenticity(forged, author)
        # The forged trace starts at glucose 0.5 but intention replay will
        # diverge in glucose values and possibly outputs
        assert result["authentic"] is False

    def test_empty_trace(self):
        author = EmbodiedScholar(author_id="test-author")
        result = verify_trace_authenticity([], author)
        assert result["authentic"] is False
        assert result["attestation_score"] == 0.0
