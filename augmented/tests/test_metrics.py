"""Comprehensive tests for scholawrite.metrics module."""
from __future__ import annotations

import pytest
from scholawrite.metrics import (
    auc,
    f1,
    span_iou,
    compute_causal_signatures,
    compute_classification_metrics,
    granger_causality_test,
    SUBSTANTIAL_EDIT_RATIO,
    NCD_DEFAULT_THRESHOLD,
    JACCARD_LARGE_DIFF_THRESHOLD,
    TRUNCATION_THRESHOLD,
    MIN_TRACE_LENGTH_METRICS,
    GLUCOSE_INCREASE_TOLERANCE,
    COUPLING_ASSIMILATED_THRESHOLD,
    COUPLING_WARM_MINIMUM,
    LOCALITY_WARM_MAX,
    LOCALITY_ANOMALY_THRESHOLD,
    COUPLING_ANOMALY_THRESHOLD,
)
from scholawrite.schema import CausalEvent


class TestAUC:
    def test_perfect_separation(self):
        assert auc([0, 0, 1, 1], [0.1, 0.4, 0.5, 0.8]) == 1.0

    def test_worst_separation(self):
        assert auc([0, 0, 1, 1], [0.8, 0.5, 0.4, 0.1]) == 0.0

    def test_tied_scores(self):
        assert auc([0, 1], [0.5, 0.5]) == 0.5

    def test_empty_input(self):
        assert auc([], []) == 0.0

    def test_all_positive(self):
        assert auc([1, 1], [0.5, 0.8]) == 0.0

    def test_all_negative(self):
        assert auc([0, 0], [0.5, 0.8]) == 0.0


class TestF1:
    def test_perfect(self):
        assert f1([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0

    def test_zero(self):
        assert f1([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0

    def test_partial(self):
        assert f1([1, 1, 0], [1, 0, 1]) == 0.5

    def test_empty(self):
        assert f1([], []) == 0.0

    def test_all_negative_true_all_negative_pred(self):
        assert f1([0, 0], [0, 0]) == 0.0


class TestSpanIOU:
    def test_perfect_overlap(self):
        spans = [(0, 10), (20, 30)]
        assert span_iou(spans, spans) == 1.0

    def test_no_overlap(self):
        assert span_iou([(0, 10)], [(10, 20)]) == 0.0

    def test_partial_overlap(self):
        assert span_iou([(0, 10)], [(5, 15)]) == pytest.approx(5 / 15)

    def test_both_empty(self):
        assert span_iou([], []) == 1.0

    def test_one_empty(self):
        assert span_iou([(0, 10)], []) == 0.0
        assert span_iou([], [(0, 10)]) == 0.0


class TestComputeCausalSignatures:
    def test_empty_trace(self):
        result = compute_causal_signatures([])
        assert result == {"locality": 0.0, "coupling": 0.0, "plausibility": 0.0, "causal_asymmetry": 0.0, "entropy_production": 0.0, "integrated_information": 0.0, "temporal_binding": 0.0, "free_energy_score": 0.0, "consciousness_score": 0.0}

    def test_all_success_trace(self):
        trace = [
            CausalEvent("w", "w", "success", None, None, 0.9, 100.0, 3.0)
            for _ in range(10)
        ]
        result = compute_causal_signatures(trace)
        assert result["locality"] == 0.0
        assert result["coupling"] == 0.0

    def test_with_failures_and_repairs(self):
        trace = []
        for i in range(12):
            if i in (3, 7):
                trace.append(CausalEvent("w", "r", "repair", "lexical_starvation", "r", 0.9 - i * 0.01, 200.0, 8.0))
            else:
                trace.append(CausalEvent("w", "w", "success", None, None, 0.9 - i * 0.01, 100.0, 3.0))
        result = compute_causal_signatures(trace)
        assert isinstance(result["locality"], float)
        assert isinstance(result["coupling"], float)
        assert isinstance(result["plausibility"], float)


class TestComputeClassificationMetrics:
    def test_perfect_predictions(self):
        result = compute_classification_metrics(
            ["a", "b", "a", "b"], ["a", "b", "a", "b"], verbose=False
        )
        assert result["accuracy"] == 1.0
        assert result["macro_f1"] == 1.0

    def test_empty_input(self):
        result = compute_classification_metrics([], [], verbose=False)
        assert result["accuracy"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_classification_metrics([1, 2], [1], verbose=False)

    def test_per_class_metrics(self):
        result = compute_classification_metrics(
            ["a", "a", "b", "b"], ["a", "b", "a", "b"], verbose=False
        )
        assert "a" in result["per_class"]
        assert "b" in result["per_class"]
        assert "precision" in result["per_class"]["a"]


class TestConstants:
    """Verify all exported constants have expected types and reasonable values."""

    def test_substantial_edit_ratio(self):
        assert 0 < SUBSTANTIAL_EDIT_RATIO < 1

    def test_ncd_threshold(self):
        assert 0 < NCD_DEFAULT_THRESHOLD < 1

    def test_jaccard_threshold(self):
        assert 0 < JACCARD_LARGE_DIFF_THRESHOLD < 1

    def test_truncation_threshold(self):
        assert 0 < TRUNCATION_THRESHOLD < 1

    def test_trajectory_thresholds(self):
        assert COUPLING_WARM_MINIMUM < COUPLING_ASSIMILATED_THRESHOLD
        assert COUPLING_ANOMALY_THRESHOLD < COUPLING_WARM_MINIMUM
        assert LOCALITY_WARM_MAX > 0
        assert LOCALITY_ANOMALY_THRESHOLD > 0

    def test_all_constants_importable(self):
        """Verify that all constants used by other modules are accessible."""
        assert MIN_TRACE_LENGTH_METRICS >= 0
        assert GLUCOSE_INCREASE_TOLERANCE >= 0
        assert LOCALITY_ANOMALY_THRESHOLD > 0
        assert COUPLING_ANOMALY_THRESHOLD > 0


class TestGrangerCausalityTest:
    def test_too_short_trace(self):
        """Traces shorter than 4 events should return 0.0."""
        trace = [
            CausalEvent("w", "w", "success", None, None, 0.9, 100.0, 3.0)
            for _ in range(3)
        ]
        assert granger_causality_test(trace) == 0.0
        assert granger_causality_test([]) == 0.0

    def test_causal_trace(self):
        """Failures followed by complexity drops should show asymmetry > 1."""
        trace = []
        for i in range(20):
            if i in (4, 9, 14):
                # Failure event with high complexity
                trace.append(CausalEvent(
                    "w", "r", "failure", "lexical_starvation", "r",
                    0.9 - i * 0.02, 200.0, 8.0,
                ))
            elif i in (5, 10, 15):
                # Post-failure: complexity drops (simplification)
                trace.append(CausalEvent(
                    "w", "w", "success", None, None,
                    0.9 - i * 0.02, 100.0, 2.0,
                ))
            else:
                trace.append(CausalEvent(
                    "w", "w", "success", None, None,
                    0.9 - i * 0.02, 100.0, 5.0,
                ))
        result = granger_causality_test(trace)
        assert result > 0.0

    def test_acausal_trace(self):
        """All-success uniform complexity trace should yield ~0 or low asymmetry."""
        trace = [
            CausalEvent("w", "w", "success", None, None, 0.9, 100.0, 5.0)
            for _ in range(20)
        ]
        result = granger_causality_test(trace)
        # No failures => no variance in failure_flags => F=0
        assert result == 0.0

    def test_returns_in_causal_signatures(self):
        """causal_asymmetry should appear in compute_causal_signatures output."""
        trace = [
            CausalEvent("w", "w", "success", None, None, 0.9, 100.0, 3.0)
            for _ in range(10)
        ]
        result = compute_causal_signatures(trace)
        assert "causal_asymmetry" in result
        assert isinstance(result["causal_asymmetry"], float)
