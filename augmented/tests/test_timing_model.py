"""Tests for timing_model="klicke_ar1": keystroke timing calibrated to KLiCKe (argumentative-text typing, not scholarly LaTeX)."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scholawrite import embodied
from scholawrite.config import SimulationConfig, get_sim_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _use_klicke(monkeypatch, **overrides) -> SimulationConfig:
    """Select the new model for EmbodiedScholar without touching the lru_cached global config."""
    cfg = SimulationConfig(timing_model="klicke_ar1", **overrides)
    monkeypatch.setattr(embodied, "get_sim_config", lambda: cfg)
    return cfg


def _single_row_table(row, lo_ms=2000.0, hi_ms=60000.0):
    return ((row,), (1.0,), lo_ms, hi_ms)


def _row(**kw):
    base = {
        "mu": 5.5,
        "sigma": 0.6,
        "rho": 0.4,
        "p_pause": 0.04,
        "pause_mu": 7.0,
        "pause_sigma": 1.5,
    }
    return embodied._WriterTiming(**{**base, **kw})


class TestConfig:
    def test_default_model_is_the_legacy_one(self):
        assert SimulationConfig().timing_model == "lognormal_iid"
        assert get_sim_config().timing_model == "lognormal_iid"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"timing_model": "klicke_ar2"},
            {"timing_ar_share": 1.01},
            {"timing_ar_share": -0.01},
            {"timing_rho_scale": -0.5},
            {"timing_sigma_scale": 0.0},
        ],
    )
    def test_invalid_timing_settings_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SimulationConfig(**kwargs)

    def test_json_defaults_match_the_dataclass(self):
        cfg, dataclass_default = get_sim_config(), SimulationConfig()
        for name in (
            "timing_model",
            "timing_seed",
            "timing_offset",
            "timing_rho_scale",
            "timing_sigma_scale",
            "timing_ar_share",
            "timing_pause_modulated",
        ):
            assert getattr(cfg, name) == getattr(dataclass_default, name), name


class TestWriterTable:
    def test_real_table_satisfies_the_loader_contract(self):
        rows, cumulative, lo_ms, hi_ms = embodied._timing_writer_table()
        assert len(rows) >= 1000
        assert (lo_ms, hi_ms) == (2000.0, 60000.0)
        assert all(b > a for a, b in zip(cumulative, cumulative[1:])) and cumulative[0] > 0
        for r in rows:
            assert all(
                math.isfinite(v)
                for v in (r.mu, r.sigma, r.rho, r.p_pause, r.pause_mu, r.pause_sigma)
            )
            assert r.sigma > 0 and r.pause_sigma > 0 and 0.0 < r.p_pause < 0.5

    def test_rows_are_drawn_in_proportion_to_their_window_counts(self, monkeypatch):
        rows = (_row(mu=5.0), _row(mu=6.0))
        # cumulative weights (1, 4): the rows weigh 1 and 3
        monkeypatch.setattr(
            embodied, "_timing_writer_table", lambda: (rows, (1.0, 4.0), 2000.0, 60000.0)
        )
        cfg = _use_klicke(monkeypatch)
        heavy = np.mean([embodied._KlickeTiming(f"w{i}", cfg).row.mu == 6.0 for i in range(4000)])
        assert abs(heavy - 0.75) < 0.03


class TestDeterminism:
    _CODE = """
import json
from scholawrite import embodied
from scholawrite.config import SimulationConfig
cfg = SimulationConfig(timing_model="klicke_ar1", timing_seed=%d)
embodied.get_sim_config = lambda: cfg
scholar = embodied.EmbodiedScholar("author_fresh_process", initial_glucose=1.0)
out = []
for i in range(300):
    depth = 1.0 + (i %% 8)
    out.append(scholar.calculate_latency(depth))
    scholar.consume_resources(1, depth)
print(json.dumps(out))
"""

    def _fresh_process(self, hash_seed: str, timing_seed: int = 0):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        proc = subprocess.run(
            [sys.executable, "-c", self._CODE % timing_seed],
            cwd=PACKAGE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return json.loads(proc.stdout)

    def test_same_author_gives_the_same_latencies_in_fresh_processes(self, monkeypatch):
        first, second = (
            self._fresh_process("0"),
            self._fresh_process("987654"),
        )  # different hash() randomisation
        assert first == second and len(set(first)) > 100

        _use_klicke(monkeypatch)
        scholar = embodied.EmbodiedScholar("author_fresh_process", initial_glucose=1.0)
        in_process = []
        for i in range(300):
            depth = 1.0 + (i % 8)
            in_process.append(scholar.calculate_latency(depth))
            scholar.consume_resources(1, depth)
        assert in_process == first

    def test_timing_seed_changes_the_stream(self):
        assert self._fresh_process("0", timing_seed=1) != self._fresh_process("0", timing_seed=0)


class TestStreams:
    def test_legacy_rng_stream_is_not_consumed(self, monkeypatch):
        _use_klicke(monkeypatch)
        scholar = embodied.EmbodiedScholar("author_rng", initial_glucose=1.0)
        state = scholar._rng.getstate()
        for i in range(300):
            scholar.calculate_latency(1.0 + (i % 8))
            scholar.consume_resources(1, 1.0)
        assert scholar._rng.getstate() == state

    def test_timing_stream_is_independent_of_cognitive_state_and_scales_relative_to_the_reference(
        self, monkeypatch
    ):
        """Same author -> same draws. Depth 1 / glucose 1 is the reference (factor 1); other states scale it."""
        _use_klicke(monkeypatch)
        reference = embodied.EmbodiedScholar("author_mod", initial_glucose=1.0)
        loaded = embodied.EmbodiedScholar("author_mod", initial_glucose=0.05)
        factor = loaded.deterministic_latency(8.0) / loaded.deterministic_latency(1.0, 1.0)
        assert factor > 2.0
        modulated = 0
        for _ in range(500):
            base, scaled = reference.calculate_latency(1.0), loaded.calculate_latency(8.0)
            if base >= 2000.0:  # a pause: not modulated by default
                assert scaled == base
            elif base > 50.0:
                assert scaled == pytest.approx(base * factor, rel=3e-4)
                modulated += 1
        assert modulated > 100  # events neither floored at 50 ms nor pauses

    def test_pause_modulation_is_opt_in(self, monkeypatch):
        _use_klicke(monkeypatch, timing_pause_modulated=True)
        reference = embodied.EmbodiedScholar("author_mod", initial_glucose=1.0)
        loaded = embodied.EmbodiedScholar("author_mod", initial_glucose=0.05)
        factor = loaded.deterministic_latency(8.0) / loaded.deterministic_latency(1.0, 1.0)
        pauses = 0
        for _ in range(3000):
            base, scaled = reference.calculate_latency(1.0), loaded.calculate_latency(8.0)
            if base >= 2000.0:
                assert scaled == pytest.approx(base * factor, rel=3e-4)
                pauses += 1
        assert pauses > 20


class TestModel:
    @pytest.mark.parametrize("author_id", ["author_a", "author_b", "author_c"])
    def test_long_run_lag1_autocorrelation_and_pause_share(self, monkeypatch, author_id):
        """5,000 events at constant modulation: non-pause log latency has lag-1 autocorrelation ar_share * rho_ar
        (pauses do not advance the AR state, so the non-pause subsequence is the AR chain) and the pause share is p_pause."""
        row = _row()
        monkeypatch.setattr(embodied, "_timing_writer_table", lambda: _single_row_table(row))
        cfg = _use_klicke(
            monkeypatch,
            timing_offset=0.0,
            timing_rho_scale=1.0,
            timing_sigma_scale=1.0,
            timing_ar_share=0.6,
        )
        scholar = embodied.EmbodiedScholar(
            author_id, initial_glucose=1.0
        )  # glucose never consumed: modulation is constant
        lat = np.array([scholar.calculate_latency(3.0) for _ in range(5000)])
        paused = lat >= 2000.0
        x = np.log(lat[~paused])
        lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        rho_ar = min(0.98, cfg.timing_rho_scale * row.rho)
        assert abs(lag1 - rho_ar * cfg.timing_ar_share) < 0.08
        assert abs(paused.mean() - row.p_pause) < 0.015

    def test_variance_split_between_ar_and_white_noise(self, monkeypatch):
        row = _row(
            sigma=0.4, rho=0.0, p_pause=0.001
        )  # rho 0: all variance is i.i.d. whatever the share
        monkeypatch.setattr(embodied, "_timing_writer_table", lambda: _single_row_table(row))
        _use_klicke(monkeypatch, timing_offset=0.0, timing_sigma_scale=1.5, timing_ar_share=0.3)
        scholar = embodied.EmbodiedScholar("author_var", initial_glucose=1.0)
        lat = np.array([scholar.calculate_latency(1.0) for _ in range(20000)])
        x = np.log(
            lat[lat < 2000.0]
        )  # modulation at depth 1 / glucose 1 is exactly 1; the 50 ms floor is 2.7 sd out
        assert x.std() == pytest.approx(1.5 * row.sigma, rel=0.05)

    @pytest.mark.parametrize(
        "pause_mu,pause_sigma", [(-24.0, 6.0), (3.0, 3.0), (7.0, 1.5), (9.0, 0.3), (12.0, 0.5)]
    )
    def test_pause_draws_stay_inside_the_pause_window(self, monkeypatch, pause_mu, pause_sigma):
        row = _row(pause_mu=pause_mu, pause_sigma=pause_sigma)
        monkeypatch.setattr(embodied, "_timing_writer_table", lambda: _single_row_table(row))
        cfg = _use_klicke(monkeypatch)
        timing = embodied._KlickeTiming("author_pause", cfg)
        rng = random.Random(1)
        draws = [
            timing._pause_ms(u)
            for u in (0.0, 1e-12, 0.5, 1.0 - 1e-12) + tuple(rng.random() for _ in range(2000))
        ]
        assert min(draws) >= 2000.0 - 1e-6 and max(draws) < 60000.0

    def test_real_table_latencies_are_floored_and_rounded(self, monkeypatch):
        _use_klicke(monkeypatch)
        for i in range(25):
            scholar = embodied.EmbodiedScholar(f"author_{i}", initial_glucose=1.0)
            lat = [scholar.calculate_latency(1.0 + (j % 8)) for j in range(200)]
            assert min(lat) >= 50.0 and max(lat) < 60000.0 * 3.0
            assert all(v == round(v, 2) for v in lat)

    def test_default_scholar_does_not_build_the_new_model(self):
        scholar = embodied.EmbodiedScholar("author_legacy")
        assert scholar._timing is None
