"""Tests for scholawrite.integrated_information module."""
from __future__ import annotations

import math
import pytest
from scholawrite.schema import CausalEvent
from scholawrite.integrated_information import (
    compute_phi,
    compute_channel_integration_matrix,
    compute_phi_trajectory,
)
from scholawrite.embodied import EmbodiedScholar
from scholawrite.causal_core import (
    IrreversibleProcessEngine,
    LexicalIntention,
)


def _make_event(
    glucose: float = 0.8,
    latency: float = 150.0,
    complexity: float = 5.0,
    status: str = "success",
    failure_mode: str | None = None,
) -> CausalEvent:
    return CausalEvent(
        intention="test",
        actual_output="test",
        status=status,
        failure_mode=failure_mode,
        repair_artifact=None,
        glucose_at_event=glucose,
        latency_ms=latency,
        syntactic_complexity=complexity,
    )


class TestComputePhiEdgeCases:
    def test_empty_trace(self):
        assert compute_phi([]) == 0.0

    def test_single_event(self):
        assert compute_phi([_make_event()]) == 0.0

    def test_two_events(self):
        """Two events is below the minimum of 3."""
        trace = [_make_event(), _make_event(glucose=0.7)]
        assert compute_phi(trace) == 0.0

    def test_constant_channels(self):
        """All channels constant -> zero entropy -> Phi = 0."""
        trace = [_make_event() for _ in range(20)]
        assert compute_phi(trace) == 0.0


class TestComputePhiIndependentChannels:
    def test_independent_channels_low_phi(self):
        """Channels constructed independently should yield low Phi."""
        import hashlib

        trace = []
        for i in range(50):
            # Each channel varies but independently of others
            seed_g = int(hashlib.md5(f"glucose:{i}".encode()).hexdigest(), 16)
            seed_l = int(hashlib.md5(f"latency:{i}".encode()).hexdigest(), 16)
            seed_c = int(hashlib.md5(f"complexity:{i}".encode()).hexdigest(), 16)
            seed_f = int(hashlib.md5(f"failure:{i}".encode()).hexdigest(), 16)

            trace.append(_make_event(
                glucose=0.3 + 0.7 * ((seed_g % 1000) / 1000.0),
                latency=100.0 + 200.0 * ((seed_l % 1000) / 1000.0),
                complexity=1.0 + 9.0 * ((seed_c % 1000) / 1000.0),
                status="failure" if (seed_f % 10) < 3 else "success",
                failure_mode="lexical_starvation" if (seed_f % 10) < 3 else None,
            ))

        phi = compute_phi(trace)
        # Independent channels: Phi should be low
        assert phi < 0.3, f"Expected low Phi for independent channels, got {phi}"


class TestComputePhiCorrelatedChannels:
    def test_perfectly_correlated_high_phi(self):
        """All channels perfectly correlated should yield high Phi."""
        trace = []
        for i in range(50):
            t = i / 50.0
            glucose = 1.0 - t * 0.7
            latency = 100.0 + t * 200.0  # increases as glucose drops
            complexity = 8.0 - t * 5.0   # decreases as glucose drops
            is_fail = glucose < 0.5
            trace.append(_make_event(
                glucose=glucose,
                latency=latency,
                complexity=complexity,
                status="failure" if is_fail else "success",
                failure_mode="lexical_starvation" if is_fail else None,
            ))

        phi = compute_phi(trace)
        assert phi > 0.05, f"Expected positive Phi for correlated channels, got {phi}"


class TestComputePhiAuthenticTrace:
    def test_authentic_trace_positive_phi(self):
        """Trace from IrreversibleProcessEngine should show Phi > 0."""
        author = EmbodiedScholar("phi_test", initial_glucose=1.0)
        engine = IrreversibleProcessEngine(author)

        intentions = [
            LexicalIntention("however", 7.0, 0.3, 0.05),
            LexicalIntention("the", 1.0, 0.01, 0.01),
            LexicalIntention("epistemological", 6.0, 0.85, 0.08),
            LexicalIntention("framework", 4.0, 0.4, 0.04),
            LexicalIntention("notwithstanding", 8.0, 0.7, 0.06),
            LexicalIntention("suggests", 3.0, 0.2, 0.03),
            LexicalIntention("that", 1.0, 0.01, 0.01),
            LexicalIntention("methodological", 7.0, 0.75, 0.07),
            LexicalIntention("rigor", 5.0, 0.6, 0.05),
            LexicalIntention("is", 1.0, 0.01, 0.01),
            LexicalIntention("paramount", 6.0, 0.8, 0.06),
            LexicalIntention("in", 1.0, 0.01, 0.01),
            LexicalIntention("contemporary", 5.0, 0.5, 0.04),
            LexicalIntention("discourse", 6.0, 0.65, 0.05),
            LexicalIntention("particularly", 4.0, 0.35, 0.03),
            LexicalIntention("when", 2.0, 0.05, 0.01),
            LexicalIntention("addressing", 3.0, 0.25, 0.03),
            LexicalIntention("complex", 4.0, 0.3, 0.03),
            LexicalIntention("phenomena", 6.0, 0.7, 0.06),
            LexicalIntention("within", 3.0, 0.2, 0.02),
        ]

        for intent in intentions:
            engine.execute(intent)

        causal_events = [
            CausalEvent(
                intention=e.intention.target,
                actual_output=e.actual_output,
                status="failure" if e.failure_mode else "success",
                failure_mode=e.failure_mode,
                repair_artifact=e.actual_output if e.repair_distance > 0 else None,
                glucose_at_event=e.glucose_before,
                latency_ms=e.latency_ms,
                syntactic_complexity=e.intention.syntactic_depth,
            )
            for e in engine.trace
        ]

        phi = compute_phi(causal_events)
        # Authentic trace should have positive Phi (channels are coupled)
        assert phi > 0.0, f"Expected Phi > 0 for authentic trace, got {phi}"


class TestChannelIntegrationMatrix:
    def test_matrix_structure(self):
        """Matrix should have correct keys and be symmetric for MI."""
        trace = []
        for i in range(30):
            t = i / 30.0
            trace.append(_make_event(
                glucose=1.0 - t * 0.6,
                latency=100.0 + t * 150.0,
                complexity=7.0 - t * 4.0,
                status="failure" if t > 0.6 else "success",
            ))

        matrix = compute_channel_integration_matrix(trace)
        channels = ["glucose", "latency", "complexity", "failure"]

        # Check all keys present
        for c in channels:
            assert c in matrix, f"Missing channel {c} in matrix"
            for c2 in channels:
                assert c2 in matrix[c], f"Missing {c2} in matrix[{c}]"

        # MI should be symmetric
        for a in channels:
            for b in channels:
                if a != b:
                    assert abs(matrix[a][b] - matrix[b][a]) < 1e-10, (
                        f"MI not symmetric: {a},{b}={matrix[a][b]} vs {b},{a}={matrix[b][a]}"
                    )

    def test_matrix_empty_trace(self):
        matrix = compute_channel_integration_matrix([])
        for row in matrix.values():
            for v in row.values():
                assert v == 0.0

    def test_correlated_channels_high_mi(self):
        """Correlated channels should show higher MI than independent ones."""
        trace = []
        for i in range(50):
            t = i / 50.0
            glucose = 1.0 - t * 0.7
            latency = 100.0 + t * 200.0
            complexity = 8.0 - t * 5.0
            is_fail = glucose < 0.5
            trace.append(_make_event(
                glucose=glucose,
                latency=latency,
                complexity=complexity,
                status="failure" if is_fail else "success",
                failure_mode="lexical_starvation" if is_fail else None,
            ))

        matrix = compute_channel_integration_matrix(trace)
        # glucose-latency should show non-trivial MI since they're correlated
        assert matrix["glucose"]["latency"] > 0.0


class TestPhiTrajectory:
    def test_trajectory_length(self):
        trace = [_make_event(glucose=1.0 - i * 0.03, latency=100.0 + i * 5.0)
                 for i in range(30)]
        traj = compute_phi_trajectory(trace, window=15)
        assert len(traj) == 30 - 15 + 1

    def test_trajectory_too_short(self):
        trace = [_make_event() for _ in range(5)]
        traj = compute_phi_trajectory(trace, window=15)
        assert traj == []

    def test_trajectory_empty(self):
        assert compute_phi_trajectory([], window=15) == []

    def test_trajectory_values_bounded(self):
        trace = []
        for i in range(40):
            t = i / 40.0
            trace.append(_make_event(
                glucose=1.0 - t * 0.7,
                latency=100.0 + t * 200.0,
                complexity=8.0 - t * 5.0,
                status="failure" if t > 0.6 else "success",
            ))
        traj = compute_phi_trajectory(trace, window=15)
        for phi in traj:
            assert 0.0 <= phi <= 1.0, f"Phi out of bounds: {phi}"


class TestMetricsIntegration:
    def test_compute_causal_signatures_includes_phi(self):
        """compute_causal_signatures should include integrated_information."""
        from scholawrite.metrics import compute_causal_signatures

        # Empty trace
        result = compute_causal_signatures([])
        assert "integrated_information" in result
        assert result["integrated_information"] == 0.0

        # Non-empty trace
        trace = []
        for i in range(20):
            t = i / 20.0
            trace.append(_make_event(
                glucose=1.0 - t * 0.6,
                latency=100.0 + t * 150.0,
                complexity=7.0 - t * 4.0,
                status="failure" if t > 0.7 else "success",
                failure_mode="lexical_starvation" if t > 0.7 else None,
            ))
        result = compute_causal_signatures(trace)
        assert "integrated_information" in result
        assert isinstance(result["integrated_information"], float)
