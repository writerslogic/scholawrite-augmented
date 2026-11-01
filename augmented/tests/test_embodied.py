"""Tests for scholawrite.embodied module."""
from __future__ import annotations

import dataclasses
import math

import pytest

from scholawrite import embodied
from scholawrite.config import get_sim_config
from scholawrite.embodied import (
    EmbodiedScholar,
    erode_context_deterministically,
    get_embodied_state,
    get_syntactic_demand,
)


class TestEmbodiedScholar:
    def test_creation_defaults(self):
        scholar = EmbodiedScholar("author_1")
        assert scholar.glucose == 1.0
        assert scholar.visual_fatigue == 0.0
        assert scholar.total_tokens_produced == 0

    def test_creation_custom_glucose(self):
        scholar = EmbodiedScholar("author_1", initial_glucose=0.5)
        assert scholar.glucose == 0.5

    def test_consume_resources_depletes_glucose(self):
        scholar = EmbodiedScholar("author_1")
        initial = scholar.glucose
        scholar.consume_resources(100, 5.0)
        assert scholar.glucose < initial

    def test_consume_resources_increases_fatigue(self):
        scholar = EmbodiedScholar("author_1")
        scholar.consume_resources(100, 5.0)
        assert scholar.visual_fatigue > 0.0

    def test_glucose_never_below_floor(self):
        scholar = EmbodiedScholar("author_1")
        scholar.consume_resources(100000, 10.0)
        assert scholar.glucose >= scholar.config.glucose_floor

    def test_fatigue_capped_at_one(self):
        scholar = EmbodiedScholar("author_1")
        for _ in range(100):
            scholar.consume_resources(1000, 5.0)
        assert scholar.visual_fatigue <= 1.0

    def test_allocate_resources(self):
        scholar = EmbodiedScholar("author_1")
        alloc = scholar.allocate_resources(5.0)
        assert 0.0 <= alloc.lexical <= 1.0
        assert 0.0 <= alloc.syntactic <= 1.0
        assert 0.0 <= alloc.attention <= 1.0

    def test_calculate_latency(self):
        scholar = EmbodiedScholar("author_1")
        latency = scholar.calculate_latency(5.0)
        assert latency > 0

    def test_biometric_salt_deterministic(self):
        scholar = EmbodiedScholar("author_1")
        salt1 = scholar.get_biometric_salt(0)
        salt2 = scholar.get_biometric_salt(0)
        assert salt1 == salt2
        salt3 = scholar.get_biometric_salt(1)
        assert salt1 != salt3


class TestGetEmbodiedState:
    def test_returns_cognitive_state(self):
        author = EmbodiedScholar("author_1")
        state = get_embodied_state(author, 5, 10, "Some test text")
        assert hasattr(state, "minute")
        assert hasattr(state, "glucose_level")
        assert hasattr(state, "allocation")

    def test_minute_calculation(self):
        author = EmbodiedScholar("author_1")
        state = get_embodied_state(author, 5, 10)
        assert 0 <= state.minute <= 90


class TestErodeContext:
    def test_high_clarity_preserves_text(self):
        text = "Hello, world!"
        result = erode_context_deterministically(text, 0.95, "salt")
        assert result == text

    def test_empty_text(self):
        assert erode_context_deterministically("", 0.5, "salt") == ""

    def test_low_clarity_modifies_text(self):
        text = "Hello, World! This is a test."
        result = erode_context_deterministically(text, 0.3, "salt")
        # Low clarity should cause some changes
        assert isinstance(result, str)

    def test_deterministic(self):
        text = "Test punctuation: yes! no?"
        r1 = erode_context_deterministically(text, 0.5, "same_salt")
        r2 = erode_context_deterministically(text, 0.5, "same_salt")
        assert r1 == r2


class TestGetSyntacticDemand:
    def test_empty_text(self):
        assert get_syntactic_demand("") == 1.0

    def test_short_text(self):
        result = get_syntactic_demand("hello world")
        assert result > 0

    def test_capped_at_ten(self):
        # Very long text with many markers
        text = " ".join(["however moreover consequently"] * 100)
        result = get_syntactic_demand(text)
        assert result <= 10.0


# Golden outputs of the legacy ("lognormal_iid") latency channel, recorded from the code as it was
# before the timing_model option existed. Each author: calculate_latency at _LEGACY_DEPTHS in order,
# consume_resources(_LEGACY_TOKENS, depth) after each call, starting from initial_glucose=1.0. The second
# item is the next draw of the scholar's own RNG afterwards, which pins the stream and not only the outputs.
_LEGACY_TOKENS = 120
_LEGACY_DEPTHS = [1.0, 4.5, 8.0, 2.2, 6.7, 3.1, 7.4, 1.3, 5.5, 4.0]
_LEGACY_GOLDEN = {
    "author_0": ([50.0, 50.0, 243.89, 185.45, 149.96, 1027.3, 308.76, 187.49, 102.98, 78.75], 0.3040682626463663),
    "author_1": ([53.04, 608.05, 184.24, 176.78, 50.0, 347.97, 522.97, 109.39, 60.74, 393.57], 0.5111800526210155),
    "author_2": ([50.82, 348.78, 101.49, 552.19, 304.66, 280.68, 196.7, 80.63, 250.04, 163.07], 0.15836483199263585),
    "author_3": ([101.58, 346.99, 50.0, 362.39, 294.2, 50.0, 305.78, 75.33, 50.0, 68.43], 0.3360052646789834),
    "author_7": ([257.2, 50.0, 74.73, 103.6, 140.2, 90.3, 168.35, 132.47, 82.51, 50.0], 0.9415839813283133),
    "author_11": ([50.0, 115.05, 438.54, 50.0, 175.95, 64.22, 73.7, 454.65, 113.18, 437.54], 0.5239148157482012),
    "author_42": ([484.42, 108.95, 263.93, 122.32, 106.68, 170.15, 191.48, 110.63, 161.41, 255.68], 0.0036822788711798893),
    "author_99": ([142.73, 348.93, 55.73, 87.41, 346.53, 110.63, 61.23, 228.76, 183.66, 50.0], 0.5444044687542777),
    "author_123": ([88.35, 85.5, 54.15, 110.31, 50.0, 90.25, 360.18, 54.9, 63.34, 309.06], 0.862197061334436),
    "author_1000": ([59.76, 495.37, 51.57, 63.07, 227.7, 50.33, 160.33, 1439.11, 308.65, 151.87], 0.6767153218497981),
    "author_10000": ([171.02, 566.46, 102.89, 64.16, 206.53, 133.87, 367.31, 104.18, 96.28, 131.97], 0.08873131711374316),
    "author_10001": ([111.59, 217.1, 130.87, 50.0, 392.29, 81.34, 50.0, 182.89, 123.34, 982.9], 0.24766592888080996),
    "alice": ([93.33, 50.0, 2532.94, 70.35, 236.04, 305.0, 202.15, 50.0, 236.05, 50.0], 0.6204836272278049),
    "bob": ([50.0, 214.66, 190.57, 65.57, 192.64, 50.86, 305.49, 169.04, 210.23, 140.23], 0.17429221847411347),
    "klicke-writer-17": ([50.0, 673.25, 291.31, 160.51, 107.21, 412.91, 110.05, 127.73, 50.0, 50.0], 0.17605311095168463),
    "writer with spaces": ([108.13, 50.0, 192.77, 276.58, 187.12, 317.38, 354.35, 61.92, 215.13, 61.19], 0.4762253018877508),
    "unicode-é-author": ([51.42, 50.0, 194.78, 50.0, 225.56, 116.37, 138.6, 187.22, 147.18, 74.65], 0.5757762831665214),
    "author_20260101": ([271.64, 50.44, 506.91, 73.43, 141.08, 50.0, 239.77, 50.0, 174.57, 188.33], 0.09648586265206671),
    "x": ([50.0, 61.58, 338.38, 63.22, 611.07, 129.36, 75.37, 371.79, 163.54, 86.57], 0.760704884246003),
    "": ([58.02, 401.74, 52.56, 104.7, 378.41, 257.33, 107.6, 160.75, 345.42, 323.14], 0.6251818513100874),
    "doi:10.1000/182": ([50.0, 55.7, 88.64, 124.3, 50.42, 50.0, 164.85, 121.52, 50.0, 146.93], 0.4603720608067665),
    "author_999999": ([50.0, 76.93, 55.08, 50.0, 869.99, 296.76, 595.83, 82.17, 282.53, 163.33], 0.24113469034920454),
}
# Deterministic path (latency_log_normal_sigma=0) from initial_glucose=0.8.
_LEGACY_GOLDEN_SIGMA0 = {
    "author_0": ([134.96, 176.4, 212.64, 177.1, 234.67, 206.54, 262.11, 176.63, 258.67, 243.89], 0.35655342586582717),
    "author_1": ([134.96, 176.4, 212.64, 177.1, 234.67, 206.54, 262.11, 176.63, 258.67, 243.89], 0.18750342787109553),
    "author_2": ([134.96, 176.4, 212.64, 177.1, 234.67, 206.54, 262.11, 176.63, 258.67, 243.89], 0.39828537484327464),
}


def _legacy_run(author_id, initial_glucose=1.0):
    scholar = EmbodiedScholar(author_id, initial_glucose=initial_glucose)
    latencies = []
    for depth in _LEGACY_DEPTHS:
        latencies.append(scholar.calculate_latency(depth))
        scholar.consume_resources(_LEGACY_TOKENS, depth)
    return latencies, scholar._rng.random()


class TestLegacyLatencyPinned:
    """timing_model="lognormal_iid" (the default) reproduces the pre-timing_model outputs and RNG stream exactly."""

    @pytest.fixture(params=["default", "explicit"])
    def legacy_config(self, request, monkeypatch):
        cfg = get_sim_config()
        if request.param == "explicit":
            cfg = dataclasses.replace(cfg, timing_model="lognormal_iid")
            monkeypatch.setattr(embodied, "get_sim_config", lambda: cfg)
        assert cfg.timing_model == "lognormal_iid"
        return cfg

    def test_golden_latencies_and_rng_stream(self, legacy_config):
        assert len(_LEGACY_GOLDEN) >= 20 and len(_LEGACY_DEPTHS) == 10
        for author_id, (latencies, next_draw) in _LEGACY_GOLDEN.items():
            assert _legacy_run(author_id) == (latencies, next_draw), author_id

    def test_golden_deterministic_path(self, legacy_config, monkeypatch):
        cfg = dataclasses.replace(legacy_config, latency_log_normal_sigma=0.0)
        monkeypatch.setattr(embodied, "get_sim_config", lambda: cfg)
        for author_id, (latencies, next_draw) in _LEGACY_GOLDEN_SIGMA0.items():
            assert _legacy_run(author_id, initial_glucose=0.8) == (latencies, next_draw), author_id

    def test_deterministic_latency_is_the_legacy_formula(self):
        scholar = EmbodiedScholar("author_1")
        assert scholar.deterministic_latency(1.0, 1.0) == 115 + 90 * math.log(2.0) * (1.12 - 1.0)
        scholar.glucose = 0.4
        assert scholar.deterministic_latency(5.0) == 115 + 90 * math.log(6.0) * (1.12 - 0.4)
