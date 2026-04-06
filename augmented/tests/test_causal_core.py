"""Tests for scholawrite.causal_core module."""
from __future__ import annotations

import pytest
from scholawrite.causal_core import (
    LexicalIntention,
    IrreversibleProcessEngine,
    DeterministicRepairGenerator,
)
from scholawrite.embodied import EmbodiedScholar


class TestLexicalIntention:
    def test_creation(self):
        intent = LexicalIntention("however", 7.0, 0.3, 0.05)
        assert intent.target == "however"
        assert intent.syntactic_depth == 7.0

    def test_frozen(self):
        intent = LexicalIntention("test", 5.0, 0.5, 0.03)
        with pytest.raises(AttributeError):
            intent.target = "changed"


class TestDeterministicRepairGenerator:
    def test_lexical_starvation_repair(self):
        gen = DeterministicRepairGenerator()
        intent = LexicalIntention("epistemological", 5.0, 0.9, 0.05)
        output, dist = gen.generate_repair(intent, "lexical_starvation", 0.5, 0)
        assert isinstance(output, str)
        assert dist == 1

    def test_syntactic_collapse_repair(self):
        gen = DeterministicRepairGenerator()
        intent = LexicalIntention("notwithstanding", 9.0, 0.5, 0.05)
        output, dist = gen.generate_repair(intent, "syntactic_collapse", 0.3, 0)
        assert isinstance(output, str)
        assert dist == 2

    def test_deterministic(self):
        gen = DeterministicRepairGenerator()
        intent = LexicalIntention("test", 5.0, 0.5, 0.03)
        r1, _ = gen.generate_repair(intent, "lexical_starvation", 0.5, 42)
        r2, _ = gen.generate_repair(intent, "lexical_starvation", 0.5, 42)
        assert r1 == r2


class TestIrreversibleProcessEngine:
    def test_execute_simple_token(self):
        author = EmbodiedScholar("test_author", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)
        intent = LexicalIntention("however", 3.0, 0.2, 0.03)
        output = engine.execute(intent)
        assert isinstance(output, str)
        assert len(engine.trace) == 1

    def test_glucose_decreases(self):
        author = EmbodiedScholar("test_author", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)
        initial_glucose = author.glucose
        for i in range(20):
            engine.execute(LexicalIntention(f"word{i}", 5.0, 0.5, 0.03))
        assert author.glucose < initial_glucose

    def test_render_text(self):
        author = EmbodiedScholar("test_author", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)
        words = ["The", "quick", "brown", "fox"]
        for w in words:
            engine.execute(LexicalIntention(w, 3.0, 0.2, 0.01))
        text = engine.render_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_compute_signatures(self):
        author = EmbodiedScholar("test_author", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)
        for i in range(15):
            engine.execute(LexicalIntention(f"word{i}", 5.0, 0.5, 0.03))
        sigs = engine.compute_causal_signatures()
        assert "repair_locality" in sigs
        assert "resource_coupling" in sigs
        assert "is_plausible" in sigs

    def test_trace_glucose_monotonic(self):
        """Glucose in trace should never increase."""
        author = EmbodiedScholar("test_author", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)
        for i in range(20):
            engine.execute(LexicalIntention(f"word{i}", 5.0, 0.5, 0.03))
        for i in range(len(engine.trace) - 1):
            assert engine.trace[i + 1].glucose_after <= engine.trace[i].glucose_before + 0.001
