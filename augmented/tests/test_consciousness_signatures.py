"""Tests for consciousness signature computation and integration."""
from __future__ import annotations

import pytest
from scholawrite.schema import CausalEvent
from scholawrite.consciousness_signatures import (
    ConsciousnessSignatureResult,
    compute_consciousness_signatures,
    compare_consciousness_profiles,
)
from scholawrite.metrics import compute_causal_signatures


def _make_human_trace(n: int = 25) -> list[CausalEvent]:
    """Build a realistic human-like trace with depleting glucose and occasional failures."""
    events = []
    glucose = 0.95
    for i in range(n):
        glucose = max(0.3, glucose - 0.015 - (i % 5) * 0.002)
        failed = i % 7 == 4
        complexity = 3.0 + (i % 4) * 1.5
        latency = 120.0 + i * 2.5 + (30.0 if failed else 0.0)
        events.append(CausalEvent(
            intention=f"word_{i}",
            actual_output=f"word_{i}" if not failed else f"repair_{i}",
            status="failure" if failed else "success",
            failure_mode="lexical_starvation" if failed else None,
            repair_artifact=f"repair_{i}" if failed else None,
            glucose_at_event=round(glucose, 4),
            latency_ms=round(latency, 2),
            syntactic_complexity=complexity,
        ))
    return events


def _make_machine_trace(n: int = 25) -> list[CausalEvent]:
    """Build a machine-like trace: constant glucose, no failures, uniform complexity."""
    events = []
    for i in range(n):
        events.append(CausalEvent(
            intention=f"token_{i}",
            actual_output=f"token_{i}",
            status="success",
            failure_mode=None,
            repair_artifact=None,
            glucose_at_event=0.9,
            latency_ms=100.0,
            syntactic_complexity=4.0,
        ))
    return events


class TestConsciousnessSignatureResult:
    """Test the result dataclass."""

    def test_all_fields_present(self):
        result = compute_consciousness_signatures(_make_human_trace())
        assert isinstance(result, ConsciousnessSignatureResult)
        assert hasattr(result, "entropy_production")
        assert hasattr(result, "integrated_information")
        assert hasattr(result, "temporal_binding")
        assert hasattr(result, "free_energy_score")
        assert hasattr(result, "phenomenological_gap")
        assert hasattr(result, "composite_consciousness_score")
        assert hasattr(result, "is_human_like")
        assert hasattr(result, "signal_breakdown")

    def test_signal_breakdown_structure(self):
        result = compute_consciousness_signatures(_make_human_trace())
        bd = result.signal_breakdown
        assert "raw" in bd
        assert "normalized" in bd
        assert "weights" in bd
        assert set(bd["normalized"].keys()) == {
            "causal_dag", "causal_concentration", "cross_channel_mi",
            "decay_type", "free_energy", "adaptation",
        }

    def test_frozen(self):
        result = compute_consciousness_signatures(_make_human_trace())
        with pytest.raises(AttributeError):
            result.entropy_production = 99.0  # type: ignore[misc]


class TestComputeConsciousnessSignatures:
    """Test the main computation function."""

    def test_human_trace_positive_composite(self):
        result = compute_consciousness_signatures(_make_human_trace())
        assert result.composite_consciousness_score > 0

    def test_machine_trace_lower_than_human(self):
        human = compute_consciousness_signatures(_make_human_trace())
        machine = compute_consciousness_signatures(_make_machine_trace())
        assert human.composite_consciousness_score >= machine.composite_consciousness_score

    def test_empty_trace(self):
        result = compute_consciousness_signatures([])
        assert result.composite_consciousness_score == 0.0
        assert result.is_human_like is False

    def test_very_short_trace(self):
        trace = [CausalEvent(
            intention="a", actual_output="a", status="success",
            failure_mode=None, repair_artifact=None,
            glucose_at_event=0.9, latency_ms=100.0, syntactic_complexity=3.0,
        )]
        result = compute_consciousness_signatures(trace)
        assert result.composite_consciousness_score == 0.0

    def test_normalized_values_in_range(self):
        result = compute_consciousness_signatures(_make_human_trace(40))
        bd = result.signal_breakdown
        for key, val in bd["normalized"].items():
            assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"

    def test_composite_in_range(self):
        result = compute_consciousness_signatures(_make_human_trace())
        assert 0.0 <= result.composite_consciousness_score <= 1.0


class TestCompareConsciousnessProfiles:
    """Test the comparison function."""

    def test_basic_comparison(self):
        human_traces = [_make_human_trace() for _ in range(3)]
        machine_traces = [_make_machine_trace() for _ in range(3)]
        result = compare_consciousness_profiles(human_traces, machine_traces)
        assert "per_metric_auc" in result
        assert "composite_auc" in result
        assert "human_stats" in result
        assert "machine_stats" in result
        assert result["n_human"] == 3
        assert result["n_machine"] == 3

    def test_auc_values_valid(self):
        human_traces = [_make_human_trace() for _ in range(3)]
        machine_traces = [_make_machine_trace() for _ in range(3)]
        result = compare_consciousness_profiles(human_traces, machine_traces)
        for metric, auc_val in result["per_metric_auc"].items():
            assert 0.0 <= auc_val <= 1.0, f"AUC for {metric} out of range: {auc_val}"
        assert 0.0 <= result["composite_auc"] <= 1.0

    def test_empty_groups(self):
        result = compare_consciousness_profiles([], [_make_machine_trace()])
        assert result["composite_auc"] == 0.0

    def test_stats_structure(self):
        human_traces = [_make_human_trace()]
        machine_traces = [_make_machine_trace()]
        result = compare_consciousness_profiles(human_traces, machine_traces)
        for group in ("human_stats", "machine_stats"):
            assert "mean" in result[group]
            assert "min" in result[group]
            assert "max" in result[group]


class TestMetricsIntegration:
    """Test that consciousness_score appears in compute_causal_signatures."""

    def test_consciousness_score_key_present_empty(self):
        result = compute_causal_signatures([])
        assert "consciousness_score" in result
        assert result["consciousness_score"] == 0.0

    def test_consciousness_score_key_present_short(self, sample_causal_trace):
        result = compute_causal_signatures(sample_causal_trace)
        assert "consciousness_score" in result
        # Trace has 12 events (< 15), so consciousness_score should be 0.0
        assert result["consciousness_score"] == 0.0

    def test_consciousness_score_computed_long_trace(self):
        trace = _make_human_trace(20)
        result = compute_causal_signatures(trace)
        assert "consciousness_score" in result
        # 20 events > 15 threshold, should compute a value
        assert isinstance(result["consciousness_score"], float)

    def test_all_existing_keys_preserved(self, sample_causal_trace):
        """Ensure we did not remove any existing keys from compute_causal_signatures."""
        result = compute_causal_signatures(sample_causal_trace)
        expected_keys = {
            "locality", "coupling", "coupling_valid", "plausibility",
            "is_plausible", "causal_asymmetry", "entropy_production",
            "integrated_information", "temporal_binding", "free_energy_score",
            "consciousness_score",
        }
        assert expected_keys.issubset(set(result.keys()))
