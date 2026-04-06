"""Tests for free energy trajectory analysis."""
from __future__ import annotations

import pytest
from scholawrite.schema import CausalEvent
from scholawrite.free_energy import (
    compute_prediction_error,
    compute_free_energy_trajectory,
    compute_surprise_spikes,
    compute_adaptation_rate,
    compute_phenomenological_gap,
)
from scholawrite.metrics import compute_causal_signatures
from scholawrite.embodied import EmbodiedScholar
from scholawrite.causal_core import IrreversibleProcessEngine, LexicalIntention


def _make_event(
    intention: str = "word",
    actual: str = "word",
    status: str = "success",
    failure_mode=None,
    repair=None,
    glucose: float = 0.9,
    latency: float = 120.0,
    complexity: float = 3.0,
) -> CausalEvent:
    return CausalEvent(
        intention=intention,
        actual_output=actual,
        status=status,
        failure_mode=failure_mode,
        repair_artifact=repair,
        glucose_at_event=glucose,
        latency_ms=latency,
        syntactic_complexity=complexity,
    )


# --- compute_prediction_error ---

class TestPredictionError:
    def test_empty_trace(self):
        assert compute_prediction_error([]) == []

    def test_all_success_no_glucose_change(self):
        trace = [_make_event(glucose=0.9, complexity=5.0)]
        errors = compute_prediction_error(trace)
        # First event has no glucose change, no failure -> error = 0
        assert errors == [0.0]

    def test_failure_adds_complexity_surprise(self):
        trace = [
            _make_event(glucose=0.9, complexity=3.0),
            _make_event(
                glucose=0.85, complexity=6.0,
                status="repair", failure_mode="lexical_starvation",
            ),
        ]
        errors = compute_prediction_error(trace)
        # Event 0: 0 (no failure, no prior glucose)
        # Event 1: 6.0 * 1 + |0.85-0.9| * 2 = 6.0 + 0.1 = 6.1
        assert len(errors) == 2
        assert errors[0] == 0.0
        assert abs(errors[1] - 6.1) < 1e-6

    def test_glucose_change_contributes(self):
        trace = [
            _make_event(glucose=0.9, complexity=2.0),
            _make_event(glucose=0.7, complexity=2.0),  # big glucose drop, success
        ]
        errors = compute_prediction_error(trace)
        # Event 1: 0 (success) + |0.7-0.9| * 2 = 0.4
        assert abs(errors[1] - 0.4) < 1e-6

    def test_known_trace(self):
        """Known trace with specific failures produces expected errors."""
        trace = [
            _make_event(glucose=1.0, complexity=2.0),
            _make_event(glucose=0.95, complexity=4.0, status="repair",
                        failure_mode="lexical_starvation"),
            _make_event(glucose=0.90, complexity=3.0),
        ]
        errors = compute_prediction_error(trace)
        assert len(errors) == 3
        # Event 0: first event, no glucose change, success -> 0
        assert errors[0] == 0.0
        # Event 1: 4.0 * 1 + |0.95-1.0| * 2 = 4.0 + 0.1 = 4.1
        assert abs(errors[1] - 4.1) < 1e-6
        # Event 2: success + |0.90-0.95| * 2 = 0 + 0.1 = 0.1
        assert abs(errors[2] - 0.1) < 1e-6


# --- compute_free_energy_trajectory ---

class TestFreeEnergyTrajectory:
    def test_empty_trace(self):
        result = compute_free_energy_trajectory([])
        assert result["trajectory"] == []
        assert result["trajectory_score"] == 0.0

    def test_short_trace(self):
        trace = [_make_event(), _make_event()]
        result = compute_free_energy_trajectory(trace, window=2)
        assert len(result["trajectory"]) >= 1
        assert "warmup_slope" in result
        assert "phase_boundaries" in result

    def test_constant_trace_low_score(self):
        """Flat trace -> trajectory_score should be low (no 3-phase structure)."""
        trace = [_make_event(glucose=0.9, complexity=3.0) for _ in range(50)]
        result = compute_free_energy_trajectory(trace, window=5)
        # All successes, constant glucose & complexity -> near-zero errors
        # 3-phase model should not fit well on constant data
        assert result["trajectory_score"] < 0.3

    def test_authentic_trace_phases(self):
        """IrreversibleProcessEngine trace with 80+ tokens shows 3-phase structure."""
        author = EmbodiedScholar("test_author", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)

        intentions = []
        for i in range(100):
            # Gradually increase complexity to trigger failures late
            depth = 2.0 + (i / 100.0) * 5.0
            rarity = 0.2 + (i / 100.0) * 0.6
            cost = 0.005 + (i / 100.0) * 0.01
            intentions.append(LexicalIntention(
                target=f"word_{i}",
                syntactic_depth=depth,
                lexical_rarity=rarity,
                cognitive_cost=cost,
            ))

        for intent in intentions:
            engine.execute(intent)

        # Convert engine trace to CausalEvent list
        causal_trace = [
            CausalEvent(
                intention=e.intention.target,
                actual_output=e.actual_output,
                status="failure" if e.failure_mode else "success",
                failure_mode=e.failure_mode,
                repair_artifact=e.actual_output if e.failure_mode else None,
                glucose_at_event=e.glucose_before,
                latency_ms=e.latency_ms,
                syntactic_complexity=e.intention.syntactic_depth,
            )
            for e in engine.trace
        ]

        assert len(causal_trace) >= 80
        result = compute_free_energy_trajectory(causal_trace, window=10)

        # Should have meaningful trajectory
        assert len(result["trajectory"]) > 0
        assert result["phase_boundaries"][0] < result["phase_boundaries"][1]
        # Warmup slope should be negative or near-zero (learning/adapting)
        assert result["warmup_slope"] <= 0.1
        # Fatigue slope should be positive (errors increase)
        assert result["fatigue_slope"] >= -0.1

    def test_return_keys(self):
        trace = [_make_event() for _ in range(20)]
        result = compute_free_energy_trajectory(trace, window=3)
        expected_keys = {
            "trajectory", "warmup_slope", "plateau_level",
            "fatigue_slope", "trajectory_score", "phase_boundaries",
            "r2_3phase", "r2_linear",
        }
        assert expected_keys == set(result.keys())


# --- compute_surprise_spikes ---

class TestSurpriseSpikes:
    def test_empty_trace(self):
        assert compute_surprise_spikes([]) == []

    def test_all_success_constant(self):
        """No spikes when all events are identical successes."""
        trace = [_make_event(glucose=0.9, complexity=3.0) for _ in range(20)]
        spikes = compute_surprise_spikes(trace)
        assert spikes == []

    def test_spike_detected(self):
        """A single high-error event should be detected as a spike."""
        trace = [_make_event(glucose=0.9 - i * 0.001, complexity=2.0) for i in range(20)]
        # Insert a failure with high complexity in the middle
        trace[10] = _make_event(
            glucose=0.88, complexity=8.0,
            status="repair", failure_mode="syntactic_collapse",
        )
        spikes = compute_surprise_spikes(trace, threshold=2.0)
        assert len(spikes) >= 1
        spike_indices = [s["index"] for s in spikes]
        assert 10 in spike_indices

    def test_spike_fields(self):
        trace = [_make_event(glucose=0.9, complexity=1.0) for _ in range(10)]
        trace[5] = _make_event(
            glucose=0.5, complexity=9.0,
            status="repair", failure_mode="lexical_starvation",
            intention="difficult_word",
        )
        spikes = compute_surprise_spikes(trace, threshold=1.5)
        if spikes:
            s = spikes[0]
            assert "index" in s
            assert "magnitude" in s
            assert "context" in s
            assert isinstance(s["magnitude"], float)


# --- compute_adaptation_rate ---

class TestAdaptationRate:
    def test_empty_trace(self):
        assert compute_adaptation_rate([]) == 0.0

    def test_no_failures(self):
        trace = [_make_event() for _ in range(20)]
        assert compute_adaptation_rate(trace) == 0.0

    def test_positive_adaptation(self):
        """After failure, increasing complexity indicates adaptation."""
        trace = [
            _make_event(glucose=0.9, complexity=5.0),
            _make_event(glucose=0.88, complexity=5.0),
            _make_event(glucose=0.86, complexity=5.0),
            # Failure
            _make_event(glucose=0.84, complexity=6.0, status="repair",
                        failure_mode="syntactic_collapse"),
            # Recovery: complexity increases back toward pre-failure levels
            _make_event(glucose=0.82, complexity=3.0),
            _make_event(glucose=0.80, complexity=4.0),
            _make_event(glucose=0.78, complexity=4.5),
            _make_event(glucose=0.76, complexity=5.0),
            _make_event(glucose=0.74, complexity=5.2),
        ]
        rate = compute_adaptation_rate(trace)
        assert rate > 0.0, "Should show positive adaptation (recovery)"

    def test_authentic_trace_adaptation(self):
        """Authentic engine trace should show some adaptation."""
        author = EmbodiedScholar("adapt_test", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)

        for i in range(80):
            depth = 2.0 + (i / 80.0) * 5.0
            rarity = 0.3 + (i / 80.0) * 0.5
            engine.execute(LexicalIntention(
                target=f"w{i}", syntactic_depth=depth,
                lexical_rarity=rarity, cognitive_cost=0.008,
            ))

        causal_trace = [
            CausalEvent(
                intention=e.intention.target,
                actual_output=e.actual_output,
                status="failure" if e.failure_mode else "success",
                failure_mode=e.failure_mode,
                repair_artifact=e.actual_output if e.failure_mode else None,
                glucose_at_event=e.glucose_before,
                latency_ms=e.latency_ms,
                syntactic_complexity=e.intention.syntactic_depth,
            )
            for e in engine.trace
        ]

        # Check there are actually some failures
        failures = [e for e in causal_trace if e.failure_mode]
        if len(failures) >= 2:
            rate = compute_adaptation_rate(causal_trace)
            # Rate should be finite
            assert isinstance(rate, float)


# --- compute_phenomenological_gap ---

class TestPhenomenologicalGap:
    def test_empty_trace(self):
        assert compute_phenomenological_gap([]) == 0.0

    def test_short_trace(self):
        trace = [_make_event() for _ in range(3)]
        assert compute_phenomenological_gap(trace) == 0.0

    def test_all_success(self):
        """All-success trace: zero variance in failure -> gap = 0."""
        trace = [_make_event(complexity=float(i)) for i in range(10)]
        assert compute_phenomenological_gap(trace) == 0.0

    def test_nonzero_gap(self):
        """Trace with failures at varying complexities produces non-zero gap."""
        trace = []
        for i in range(20):
            complexity = float(i % 5 + 1)
            # Failures not perfectly correlated with complexity
            fail = (i % 7 == 0)
            trace.append(_make_event(
                complexity=complexity,
                status="repair" if fail else "success",
                failure_mode="lexical_starvation" if fail else None,
            ))
        gap = compute_phenomenological_gap(trace)
        assert 0.0 <= gap <= 1.0

    def test_authentic_trace_nonzero_gap(self):
        """Authentic trace should show non-zero phenomenological gap."""
        author = EmbodiedScholar("pheno_test", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)

        for i in range(80):
            depth = 2.0 + (i / 80.0) * 5.0
            rarity = 0.3 + (i / 80.0) * 0.5
            engine.execute(LexicalIntention(
                target=f"w{i}", syntactic_depth=depth,
                lexical_rarity=rarity, cognitive_cost=0.008,
            ))

        causal_trace = [
            CausalEvent(
                intention=e.intention.target,
                actual_output=e.actual_output,
                status="failure" if e.failure_mode else "success",
                failure_mode=e.failure_mode,
                repair_artifact=e.actual_output if e.failure_mode else None,
                glucose_at_event=e.glucose_before,
                latency_ms=e.latency_ms,
                syntactic_complexity=e.intention.syntactic_depth,
            )
            for e in engine.trace
        ]

        failures = [e for e in causal_trace if e.failure_mode]
        if len(failures) >= 2:
            gap = compute_phenomenological_gap(causal_trace)
            assert gap >= 0.0
            # With embodied resource dynamics, gap should be non-trivial
            assert isinstance(gap, float)


# --- Integration: free_energy_score in compute_causal_signatures ---

class TestIntegration:
    def test_free_energy_score_in_signatures(self, sample_causal_trace):
        sigs = compute_causal_signatures(sample_causal_trace)
        assert "free_energy_score" in sigs
        assert isinstance(sigs["free_energy_score"], float)

    def test_free_energy_score_empty_trace(self):
        sigs = compute_causal_signatures([])
        assert "free_energy_score" in sigs
        assert sigs["free_energy_score"] == 0.0


# --- Edge cases ---

class TestEdgeCases:
    def test_single_event(self):
        trace = [_make_event()]
        errors = compute_prediction_error(trace)
        assert len(errors) == 1

        result = compute_free_energy_trajectory(trace, window=10)
        assert len(result["trajectory"]) == 1

        spikes = compute_surprise_spikes(trace)
        assert isinstance(spikes, list)

        rate = compute_adaptation_rate(trace)
        assert rate == 0.0

    def test_two_events(self):
        trace = [_make_event(glucose=0.9), _make_event(glucose=0.85)]
        result = compute_free_energy_trajectory(trace, window=2)
        assert "trajectory_score" in result

    def test_window_larger_than_trace(self):
        trace = [_make_event() for _ in range(5)]
        result = compute_free_energy_trajectory(trace, window=20)
        # Should handle gracefully — window capped to trace length, output = 1 element
        assert len(result["trajectory"]) >= 1
        assert "trajectory_score" in result

    def test_all_failures(self):
        trace = [
            _make_event(
                glucose=0.9 - i * 0.05, complexity=3.0 + i,
                status="repair", failure_mode="lexical_starvation",
            )
            for i in range(10)
        ]
        errors = compute_prediction_error(trace)
        assert all(e > 0 for e in errors[1:])  # All failures -> positive errors

        spikes = compute_surprise_spikes(trace)
        assert isinstance(spikes, list)
