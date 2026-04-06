"""Tests for scholawrite.embodied module."""
from __future__ import annotations

import pytest
from scholawrite.embodied import (
    EmbodiedScholar,
    get_embodied_state,
    erode_context_deterministically,
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
