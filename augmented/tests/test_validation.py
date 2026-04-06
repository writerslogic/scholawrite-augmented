"""Tests for the validation pipeline (scholawrite.validation)."""
from __future__ import annotations

import math
import random

from scholawrite.validation import (
    CheckpointRecord,
    SignalComparison,
    _cohens_d,
    _cliffs_delta,
    _ks_test_2sample,
    _lag1_autocorr,
    _mann_whitney_u,
    _wasserstein_1d,
    _levene_test,
    _trace_to_checkpoint_metrics,
    compare_distributions,
    verify_hash_chain,
)
from scholawrite.schema import CausalEvent


def _make_checkpoint(session_id="s1", seq=0, **kwargs) -> CheckpointRecord:
    defaults = dict(
        session_id=session_id, seq=seq,
        mean_iki_ms=150.0, median_iki_ms=120.0, std_iki_ms=80.0,
        iki_entropy_bits=100.0, lag1_autocorrelation=0.1,
        pause_count=1, burst_count=3,
        planning_pause_count=1, translating_burst_count=2, revising_delete_burst_count=0,
        chars_added=20, chars_deleted=1, revision_density=0.05, wpm=30.0,
        h_prev="aaa", h_content="bbb",
        start_time_ns=0, end_time_ns=1000000000, event_count=20,
    )
    defaults.update(kwargs)
    return CheckpointRecord(**defaults)


def _make_trace(n: int = 30, seed: int = 42) -> list[CausalEvent]:
    rng = random.Random(seed)
    events = []
    glucose = 1.0
    for i in range(n):
        glucose *= 0.998
        depth = rng.uniform(1.0, 6.0)
        failed = rng.random() < 0.15
        events.append(CausalEvent(
            intention=f"word_{i}",
            actual_output=f"word_{i}" if not failed else f"repaired_{i}",
            status="repair" if failed else "success",
            failure_mode="lexical_starvation" if failed else None,
            repair_artifact=f"repaired_{i}" if failed else None,
            glucose_at_event=round(glucose, 4),
            latency_ms=round(100 + 50 * depth * (1.1 - glucose) + rng.gauss(0, 20), 2),
            syntactic_complexity=round(depth, 1),
        ))
    return events


# --- Statistical tests ---

class TestKSTest:
    def test_identical_distributions(self):
        a = list(range(100))
        d, p = _ks_test_2sample(sorted(a), sorted(a))
        assert d < 0.01
        assert p > 0.9

    def test_shifted_distributions(self):
        a = sorted([float(x) for x in range(100)])
        b = sorted([float(x + 50) for x in range(100)])
        d, p = _ks_test_2sample(a, b)
        assert d > 0.3
        assert p < 0.05

    def test_empty_input(self):
        d, p = _ks_test_2sample([], [1.0, 2.0])
        assert d == 1.0
        assert p == 0.0


class TestMannWhitney:
    def test_identical(self):
        a = [float(x) for x in range(50)]
        _, p = _mann_whitney_u(a, a)
        assert p > 0.5

    def test_shifted(self):
        a = [float(x) for x in range(50)]
        b = [float(x + 100) for x in range(50)]
        _, p = _mann_whitney_u(a, b)
        assert p < 0.05


class TestEffectSizes:
    def test_cohens_d_zero(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(_cohens_d(a, a)) < 0.01

    def test_cohens_d_large(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0]
        assert abs(_cohens_d(a, b)) > 2.0

    def test_cliffs_delta_range(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        d = _cliffs_delta(a, b)
        assert -1.0 <= d <= 1.0
        assert d < -0.5  # a < b consistently

    def test_wasserstein_zero(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _wasserstein_1d(a, a) < 0.1

    def test_wasserstein_shift(self):
        a = [float(x) for x in range(100)]
        b = [float(x + 10) for x in range(100)]
        w = _wasserstein_1d(a, b)
        assert 8.0 < w < 12.0


class TestLevene:
    def test_equal_variance(self):
        rng = random.Random(42)
        a = [rng.gauss(0, 1) for _ in range(100)]
        b = [rng.gauss(0, 1) for _ in range(100)]
        _, p = _levene_test(a, b)
        assert p > 0.05

    def test_unequal_variance(self):
        rng = random.Random(42)
        a = [rng.gauss(0, 1) for _ in range(100)]
        b = [rng.gauss(0, 10) for _ in range(100)]
        f, p = _levene_test(a, b)
        assert p < 0.05


# --- Lag-1 autocorrelation ---

class TestLag1Autocorr:
    def test_constant_series(self):
        assert _lag1_autocorr([5.0] * 10) == 0.0

    def test_positive_autocorr(self):
        # Slowly varying series should have positive autocorrelation
        series = [float(x) for x in range(20)]
        assert _lag1_autocorr(series) > 0.5

    def test_short_series(self):
        assert _lag1_autocorr([1.0, 2.0]) == 0.0


# --- Trace to checkpoint metrics ---

class TestTraceToCheckpoint:
    def test_returns_expected_keys(self):
        trace = _make_trace(30)
        m = _trace_to_checkpoint_metrics(trace)
        expected = {"mean_iki_ms", "median_iki_ms", "std_iki_ms",
                    "iki_entropy_bits", "lag1_autocorrelation",
                    "revision_density", "wpm",
                    "planning_ratio", "translating_ratio", "revising_ratio"}
        assert expected == set(m.keys())

    def test_values_in_range(self):
        trace = _make_trace(50)
        m = _trace_to_checkpoint_metrics(trace)
        assert m["mean_iki_ms"] > 0
        assert 0.0 <= m["revision_density"] <= 1.0
        assert m["wpm"] >= 0

    def test_short_trace(self):
        trace = _make_trace(2)
        m = _trace_to_checkpoint_metrics(trace)
        assert m["mean_iki_ms"] == 0.0


# --- Compare distributions ---

class TestCompareDistributions:
    def test_all_fields_populated(self):
        rng = random.Random(42)
        a = [rng.gauss(100, 20) for _ in range(50)]
        b = [rng.gauss(110, 25) for _ in range(50)]
        c = compare_distributions(a, b, "test_signal")
        assert isinstance(c, SignalComparison)
        assert c.signal_name == "test_signal"
        assert c.real_n == 50
        assert c.sim_n == 50
        assert 0.0 <= c.ks_statistic <= 1.0
        assert 0.0 <= c.ks_pvalue <= 1.0
        assert not math.isnan(c.cohens_d)
        assert -1.0 <= c.cliffs_delta <= 1.0


# --- Hash chain verification ---

class TestHashChain:
    def test_valid_chain(self):
        cps = [
            _make_checkpoint(seq=0, h_prev="genesis", h_content="aaa"),
            _make_checkpoint(seq=1, h_prev="aaa", h_content="bbb"),
            _make_checkpoint(seq=2, h_prev="bbb", h_content="ccc"),
        ]
        ok, valid, total = verify_hash_chain(cps, "s1")
        assert ok is True
        assert valid == 2
        assert total == 2

    def test_broken_chain(self):
        cps = [
            _make_checkpoint(seq=0, h_prev="genesis", h_content="aaa"),
            _make_checkpoint(seq=1, h_prev="WRONG", h_content="bbb"),
            _make_checkpoint(seq=2, h_prev="bbb", h_content="ccc"),
        ]
        ok, valid, total = verify_hash_chain(cps, "s1")
        assert ok is False
        assert valid == 1
        assert total == 2

    def test_single_checkpoint(self):
        cps = [_make_checkpoint(seq=0)]
        ok, valid, total = verify_hash_chain(cps, "s1")
        assert ok is True
        assert total == 0
