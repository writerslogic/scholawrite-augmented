"""Tests for scholawrite.cross_author module."""
from __future__ import annotations

import pytest
from scholawrite.cross_author import (
    AuthorProfile,
    CrossAuthorExperiment,
    _generate_diverse_authors,
    _generate_injected_trace,
)
from scholawrite.metrics import auc


class TestGenerateDiverseAuthors:
    def test_correct_count(self):
        configs = _generate_diverse_authors(5, seed=42)
        assert len(configs) == 5

    def test_distinct_configs(self):
        configs = _generate_diverse_authors(5, seed=42)
        glucose_vals = [c["initial_glucose"] for c in configs]
        # All distinct
        assert len(set(glucose_vals)) == 5

    def test_within_ranges(self):
        configs = _generate_diverse_authors(10, seed=99)
        for c in configs:
            assert 0.85 <= c["initial_glucose"] <= 1.0
            assert 0.9985 <= c["glucose_depletion_rate"] <= 0.9998
            assert 8000.0 <= c["fatigue_divisor"] <= 16000.0

    def test_deterministic(self):
        a = _generate_diverse_authors(5, seed=42)
        b = _generate_diverse_authors(5, seed=42)
        assert a == b


class TestAuthorProfile:
    def test_creation(self):
        p = AuthorProfile(author_id="test")
        assert p.author_id == "test"
        assert p.traces == []
        assert p.signature_stats == {}


class TestGenerateAuthorProfiles:
    def test_creates_distinct_profiles(self):
        exp = CrossAuthorExperiment(n_authors=3, traces_per_author=5, tokens_per_trace=40, seed=42)
        profiles = exp.generate_author_profiles()
        assert len(profiles) == 3
        ids = {p.author_id for p in profiles}
        assert len(ids) == 3

    def test_each_profile_has_traces(self):
        exp = CrossAuthorExperiment(n_authors=2, traces_per_author=4, tokens_per_trace=30, seed=42)
        profiles = exp.generate_author_profiles()
        for p in profiles:
            assert len(p.traces) == 4
            for trace in p.traces:
                assert len(trace) == 30

    def test_signature_stats_populated(self):
        exp = CrossAuthorExperiment(n_authors=2, traces_per_author=3, tokens_per_trace=40, seed=42)
        profiles = exp.generate_author_profiles()
        for p in profiles:
            assert "locality_mean" in p.signature_stats
            assert "coupling_mean" in p.signature_stats
            assert "failure_rate_mean" in p.signature_stats


class TestTrainThresholds:
    def test_produces_valid_ranges(self):
        exp = CrossAuthorExperiment(n_authors=3, traces_per_author=5, tokens_per_trace=40, seed=42)
        profiles = exp.generate_author_profiles()
        thresholds = exp.train_thresholds(profiles)
        assert len(thresholds) > 0
        for key, (lo, hi) in thresholds.items():
            assert lo <= hi, f"Invalid range for {key}: ({lo}, {hi})"

    def test_thresholds_cover_training_data(self):
        exp = CrossAuthorExperiment(n_authors=4, traces_per_author=5, tokens_per_trace=40, seed=42)
        profiles = exp.generate_author_profiles()
        thresholds = exp.train_thresholds(profiles)
        # Most training authors should fall within learned thresholds
        for key, (lo, hi) in thresholds.items():
            in_range = sum(1 for p in profiles if lo <= p.signature_stats[key] <= hi)
            assert in_range >= len(profiles) // 2, f"Too few in range for {key}"


class TestLeaveOneOut:
    def test_completes_without_error(self):
        exp = CrossAuthorExperiment(n_authors=3, traces_per_author=5, tokens_per_trace=40, seed=42)
        result = exp.run_leave_one_out()
        assert "mean_f1" in result
        assert "fold_results" in result
        assert len(result["fold_results"]) == 3

    def test_deterministic(self):
        exp1 = CrossAuthorExperiment(n_authors=3, traces_per_author=5, tokens_per_trace=40, seed=42)
        exp2 = CrossAuthorExperiment(n_authors=3, traces_per_author=5, tokens_per_trace=40, seed=42)
        r1 = exp1.run_leave_one_out()
        r2 = exp2.run_leave_one_out()
        assert r1["mean_f1"] == r2["mean_f1"]


class TestInjectedTracesDistinguishable:
    def test_injected_vs_authentic_auc(self):
        """Injected traces should be distinguishable from authentic (AUC > 0.6)."""
        import random
        from scholawrite.metrics import compute_causal_signatures
        from scholawrite.cross_author import _generate_trace

        rng = random.Random(42)
        author_config = {"initial_glucose": 0.95, "glucose_depletion_rate": 0.999, "fatigue_divisor": 12000.0}

        # Generate authentic traces
        authentic_scores = []
        for i in range(10):
            trace = _generate_trace(author_config, f"auth_{i}", 60, rng)
            sigs = compute_causal_signatures(trace)
            # Use plausibility as detection score
            authentic_scores.append(sigs.get("plausibility", 0.0))

        # Generate injected traces
        injected_scores = []
        for _ in range(10):
            trace = _generate_injected_trace(60, rng)
            sigs = compute_causal_signatures(trace)
            injected_scores.append(sigs.get("plausibility", 0.0))

        # Labels: 1=authentic, 0=injected
        y_true = [1.0] * len(authentic_scores) + [0.0] * len(injected_scores)
        y_score = authentic_scores + injected_scores
        roc_auc = auc(y_true, y_score)
        assert roc_auc > 0.6, f"AUC too low: {roc_auc:.4f}"


class TestAuthorSimilarity:
    def test_completes(self):
        exp = CrossAuthorExperiment(n_authors=3, traces_per_author=5, tokens_per_trace=40, seed=42)
        result = exp.run_author_similarity_analysis()
        assert "pairwise_similarity" in result
        assert "universality_ranking" in result
        assert len(result["universality_ranking"]) == 5
