"""Tests for scholawrite.detector_harness — cross-detector evaluation."""
from __future__ import annotations

import pytest
from scholawrite.adversarial import _generate_authentic_trace, ForgedTraceGenerator
from scholawrite.detector_harness import (
    ADVERSARY_HIERARCHY,
    CausalCouplingDetector,
    ConsciousnessDetector,
    DetectorHarness,
    GptZeroDetector,
    NcdDetector,
    OriginalityDetector,
    TierResult,
)


class TestHierarchyLevels:
    def test_five_levels_defined(self):
        assert len(ADVERSARY_HIERARCHY) == 5

    def test_levels_ordered(self):
        for i, level in enumerate(ADVERSARY_HIERARCHY):
            assert level.level == i

    def test_names_unique(self):
        names = [l.name for l in ADVERSARY_HIERARCHY]
        assert len(names) == len(set(names))


class TestNcdDetector:
    def test_returns_score(self):
        det = NcdDetector(reference_text="The methodology establishes a baseline")
        result = det.detect("The methodology establishes a baseline for analysis")
        assert 0.0 <= result.score <= 1.0
        assert result.error is None

    def test_no_reference_returns_error(self):
        det = NcdDetector()
        result = det.detect("some text")
        assert result.error is not None

    def test_metadata(self):
        det = NcdDetector()
        assert det.name == "NCD"
        assert det.hierarchy_level_survived == 0
        assert det.signal_type == "statistical"


class TestCausalCouplingDetector:
    def test_with_trace(self):
        trace = _generate_authentic_trace(seed=42, n_events=30)
        det = CausalCouplingDetector()
        result = det.detect("", trace=trace)
        assert 0.0 <= result.score <= 1.0
        assert result.error is None

    def test_without_trace(self):
        det = CausalCouplingDetector()
        result = det.detect("some text")
        assert result.error is not None

    def test_metadata(self):
        det = CausalCouplingDetector()
        assert det.hierarchy_level_survived == 1
        assert det.signal_type == "process_structural"


class TestConsciousnessDetector:
    def test_with_trace(self):
        trace = _generate_authentic_trace(seed=42, n_events=30)
        det = ConsciousnessDetector()
        result = det.detect("", trace=trace)
        assert 0.0 <= result.score <= 1.0
        assert result.error is None
        assert "composite" in result.raw_response

    def test_metadata(self):
        det = ConsciousnessDetector()
        assert det.name == "ConsciousnessSignatures"
        assert det.hierarchy_level_survived == 1


class TestExternalDetectors:
    def test_gptzero_no_key(self):
        # Use a clearly invalid key to ensure error path
        det = GptZeroDetector(api_key="INVALID_KEY_FOR_TEST")
        result = det.detect("Test text")
        # Should either error or return a result (if env key overrides)
        assert isinstance(result.score, float)

    def test_originality_no_key(self):
        det = OriginalityDetector(api_key="INVALID_KEY_FOR_TEST")
        result = det.detect("Test text")
        assert isinstance(result.score, float)

    def test_gptzero_metadata(self):
        det = GptZeroDetector()
        assert det.name == "GPTZero"
        assert det.hierarchy_level_survived == 0
        assert det.signal_type == "product_level"

    def test_originality_metadata(self):
        det = OriginalityDetector()
        assert det.name == "Originality.ai"
        assert det.hierarchy_level_survived == 0
        assert det.signal_type == "product_level"


class TestDetectorHarness:
    def test_evaluate_on_traces_internal(self):
        human_traces = [_generate_authentic_trace(seed=i, n_events=25) for i in range(10)]

        gen = ForgedTraceGenerator(seed=100)
        naive_traces = [
            ForgedTraceGenerator(seed=100 + i).generate_naive_forgery("word " * 25, n_events=25)
            for i in range(10)
        ]

        harness = DetectorHarness()
        harness.add_detector(CausalCouplingDetector())
        harness.add_detector(ConsciousnessDetector())

        results = harness.evaluate_on_traces(
            human_traces=human_traces,
            machine_traces_by_tier={"naive": naive_traces},
        )

        assert "matrix" in results
        assert "tiers" in results
        assert "detectors" in results
        assert len(results["matrix"]) == 2  # Two detectors
        assert "naive" in results["tiers"]

    def test_matrix_has_auc_values(self):
        human_traces = [_generate_authentic_trace(seed=i, n_events=25) for i in range(10)]
        naive_traces = [
            ForgedTraceGenerator(seed=200 + i).generate_naive_forgery("word " * 25, n_events=25)
            for i in range(10)
        ]

        harness = DetectorHarness([ConsciousnessDetector()])
        results = harness.evaluate_on_traces(
            human_traces=human_traces,
            machine_traces_by_tier={"naive": naive_traces},
        )

        matrix = results["matrix"]
        for det, tier_aucs in matrix.items():
            for tier, auc_val in tier_aucs.items():
                assert 0.0 <= auc_val <= 1.0, f"{det}/{tier}: {auc_val}"

    def test_empty_harness(self):
        harness = DetectorHarness()
        results = harness.evaluate_on_traces(
            human_traces=[],
            machine_traces_by_tier={},
        )
        assert results["matrix"] == {}
