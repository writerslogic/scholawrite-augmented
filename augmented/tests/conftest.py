"""Shared test fixtures for scholawrite test suite."""
from __future__ import annotations

import pytest
from scholawrite.schema import (
    AmbiguityFlag,
    AugmentedDocument,
    AugmentedRevision,
    CausalEvent,
    InjectionLevel,
    InjectionSpan,
    Label,
    SeedDocument,
    SeedRevision,
    TrajectoryState,
)


@pytest.fixture
def sample_seed_document() -> SeedDocument:
    """A minimal seed document with two revisions."""
    return SeedDocument(
        doc_id="doc_test001",
        revisions=[
            SeedRevision(
                doc_id="doc_test001",
                revision_id="rev_0",
                revision_index=0,
                text="The initial methodology establishes a baseline for comparative analysis.",
                timestamp="2023-01-01T00:00:00+00:00",
                provenance_hash="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            ),
            SeedRevision(
                doc_id="doc_test001",
                revision_id="rev_1",
                revision_index=1,
                text="The refined methodology establishes a robust baseline for comparative analysis of emerging patterns.",
                timestamp="2023-01-02T00:00:00+00:00",
                provenance_hash="b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6a7",
            ),
        ],
    )


@pytest.fixture
def sample_causal_trace() -> list[CausalEvent]:
    """A realistic causal trace for testing metrics and validation."""
    return [
        CausalEvent(intention="The", actual_output="The", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.95, latency_ms=120.0, syntactic_complexity=2.0),
        CausalEvent(intention="methodology", actual_output="methodology", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.94, latency_ms=135.0, syntactic_complexity=5.0),
        CausalEvent(intention="establishes", actual_output="establishes", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.93, latency_ms=140.0, syntactic_complexity=6.0),
        CausalEvent(intention="a", actual_output="a", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.92, latency_ms=110.0, syntactic_complexity=1.0),
        CausalEvent(intention="robust", actual_output="framework", status="repair", failure_mode="lexical_starvation", repair_artifact="framework", glucose_at_event=0.90, latency_ms=200.0, syntactic_complexity=4.0),
        CausalEvent(intention="baseline", actual_output="baseline", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.89, latency_ms=125.0, syntactic_complexity=3.0),
        CausalEvent(intention="for", actual_output="for", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.88, latency_ms=105.0, syntactic_complexity=1.0),
        CausalEvent(intention="comparative", actual_output="thus, ", status="repair", failure_mode="syntactic_collapse", repair_artifact="thus, ", glucose_at_event=0.86, latency_ms=250.0, syntactic_complexity=7.0),
        CausalEvent(intention="analysis", actual_output="analysis", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.85, latency_ms=130.0, syntactic_complexity=4.0),
        CausalEvent(intention="of", actual_output="of", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.84, latency_ms=100.0, syntactic_complexity=1.0),
        CausalEvent(intention="emerging", actual_output="emerging", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.83, latency_ms=145.0, syntactic_complexity=5.0),
        CausalEvent(intention="patterns", actual_output="patterns", status="success", failure_mode=None, repair_artifact=None, glucose_at_event=0.82, latency_ms=120.0, syntactic_complexity=3.0),
    ]


@pytest.fixture
def sample_injection_span() -> InjectionSpan:
    """A minimal injection span for testing."""
    return InjectionSpan(
        doc_id="doc_test001",
        revision_id="rev_1",
        injection_id="inj_test001",
        injection_level=InjectionLevel.CONTEXTUAL,
        trajectory_state=TrajectoryState.COLD,
        ambiguity_flag=AmbiguityFlag.NONE,
        span_start_char=0,
        span_end_char=50,
        span_start_sentence=0,
        span_end_sentence=0,
        generator_class="strong-frontier",
        prompt_hash="testhash123",
        rng_seed=42,
        provenance_hash="abc123def456",
        label=Label.INJECTION_CONTEXTUAL,
    )
